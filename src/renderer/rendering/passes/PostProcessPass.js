import { BasePass } from "./BasePass.js"
import { Pipeline } from "../../Pipeline.js"

import postProcessingWGSL from "../shaders/postproc.wgsl"

/**
 * PostProcessPass - Final post-processing and tone mapping
 *
 * Pass 7 in the 7-pass pipeline (final pass).
 * Applies tone mapping and outputs to the canvas.
 *
 * Input: HDR lit image from LightingPass
 * Output: Final SDR image to canvas
 */
class PostProcessPass extends BasePass {
    constructor(engine = null) {
        super('PostProcess', engine)

        this.pipeline = null
        this.inputTexture = null
        this.bloomTexture = null
        this.dummyBloomTexture = null  // 1x1 black texture when bloom disabled
        this.noiseTexture = null
        this.noiseSize = 64
        this.noiseAnimated = true
        this.guiCanvas = null          // 2D canvas for GUI overlay
        this.guiTexture = null         // GPU texture for GUI
        this.guiSampler = null

        // CRT support: intermediate texture when CRT is enabled
        this.intermediateTexture = null
        this._outputWidth = 0
        this._outputHeight = 0
        this._lastOutputToTexture = false  // Track output mode changes

        // Store resize dimensions for runtime texture creation
        this._resizeWidth = 0
        this._resizeHeight = 0
    }

    // Convenience getter for exposure setting
    get exposure() { return this.settings?.environment?.exposure ?? 1.6 }

    // Convenience getter for fxaa setting
    get fxaa() { return this.settings?.rendering?.fxaa ?? true }

    // Convenience getter for dithering settings
    get ditheringEnabled() { return this.settings?.dithering?.enabled ?? true }
    get colorLevels() { return this.settings?.dithering?.colorLevels ?? 32 }

    // Tonemap mode: 0=ACES, 1=Reinhard, 2=None/Linear
    get tonemapMode() { return this.settings?.rendering?.tonemapMode ?? 0 }

    // Convenience getters for bloom settings
    get bloomEnabled() { return this.settings?.bloom?.enabled ?? true }
    get bloomIntensity() { return this.settings?.bloom?.intensity ?? 1.0 }
    get bloomRadius() { return this.settings?.bloom?.radius ?? 5 }

    // CRT settings (determines if we output to intermediate texture)
    get crtEnabled() { return this.settings?.crt?.enabled ?? false }
    get crtUpscaleEnabled() { return this.settings?.crt?.upscaleEnabled ?? false }
    get shouldOutputToTexture() { return this.crtEnabled || this.crtUpscaleEnabled }

    /**
     * Set the input texture (HDR image from LightingPass)
     * @param {Texture} texture - Input HDR texture
     */
    setInputTexture(texture) {
        if (this.inputTexture !== texture) {
            this.inputTexture = texture
            this._needsRebuild = true
        }
    }

    /**
     * Set the bloom texture (from BloomPass)
     * @param {Object} bloomTexture - Bloom texture with mip levels
     */
    setBloomTexture(bloomTexture) {
        if (this.bloomTexture !== bloomTexture) {
            this.bloomTexture = bloomTexture
            this._needsRebuild = true
        }
    }

    /**
     * Set the noise texture for dithering
     * @param {Texture} texture - Noise texture (blue noise or bayer dither)
     * @param {number} size - Texture size
     * @param {boolean} animated - Whether to animate noise offset each frame
     */
    setNoise(texture, size = 64, animated = true) {
        this.noiseTexture = texture
        this.noiseSize = size
        this.noiseAnimated = animated
        this._needsRebuild = true
    }

    /**
     * Set the GUI canvas for overlay rendering
     * @param {HTMLCanvasElement} canvas - 2D canvas with GUI content
     */
    setGuiCanvas(canvas) {
        this.guiCanvas = canvas
    }

    /**
     * Get the output texture (for CRT pass to use)
     * Returns null if outputting directly to canvas
     */
    getOutputTexture() {
        return this.shouldOutputToTexture ? this.intermediateTexture : null
    }

    /**
     * Create or resize the intermediate texture for CRT
     */
    async _createIntermediateTexture(width, height) {
        if (!this.shouldOutputToTexture) {
            // Destroy if no longer needed
            if (this.intermediateTexture?.texture) {
                this.intermediateTexture.texture.destroy()
                this.intermediateTexture = null
            }
            return
        }

        // Skip if size hasn't changed
        if (this._outputWidth === width && this._outputHeight === height && this.intermediateTexture) {
            return
        }

        const { device } = this.engine

        // Destroy old texture
        if (this.intermediateTexture?.texture) {
            this.intermediateTexture.texture.destroy()
        }

        // Create intermediate texture (SDR format for CRT input)
        const texture = device.createTexture({
            label: 'PostProcess Intermediate',
            size: [width, height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING |
                   GPUTextureUsage.RENDER_ATTACHMENT |
                   GPUTextureUsage.COPY_SRC,
        })

        const sampler = device.createSampler({
            label: 'PostProcess Intermediate Sampler',
            minFilter: 'nearest',
            magFilter: 'nearest',
        })

        this.intermediateTexture = {
            texture,
            view: texture.createView(),
            sampler,
            width,
            height,
            format: 'rgba8unorm',  // Required by Pipeline.create()
        }

        this._outputWidth = width
        this._outputHeight = height
        this._needsRebuild = true

        console.log(`PostProcessPass: Created intermediate texture ${width}x${height} for CRT`)
    }

    async _init() {
        // Create dummy 1x1 black bloom texture for when bloom is disabled
        // This ensures shader bindings are always valid
        const { device } = this.engine

        const dummyTexture = device.createTexture({
            label: 'Dummy Bloom Texture',
            size: [1, 1, 1],
            format: 'rgba16float',
            mipLevelCount: 1,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })

        // Fill with black (no bloom)
        device.queue.writeTexture(
            { texture: dummyTexture },
            new Float32Array([0, 0, 0, 0]).buffer,
            { bytesPerRow: 8 },
            { width: 1, height: 1 }
        )

        const dummySampler = device.createSampler({
            label: 'Dummy Bloom Sampler',
            minFilter: 'linear',
            magFilter: 'linear',
        })

        this.dummyBloomTexture = {
            texture: dummyTexture,
            view: dummyTexture.createView(),
            sampler: dummySampler,
            mipCount: 1,
        }

        // Create sampler for GUI texture
        this.guiSampler = device.createSampler({
            label: 'GUI Sampler',
            minFilter: 'linear',
            magFilter: 'linear',
        })

        // Create dummy 1x1 transparent GUI texture
        const dummyGuiTexture = device.createTexture({
            label: 'Dummy GUI Texture',
            size: [1, 1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.writeTexture(
            { texture: dummyGuiTexture },
            new Uint8Array([0, 0, 0, 0]),
            { bytesPerRow: 4 },
            { width: 1, height: 1 }
        )
        this.dummyGuiTexture = {
            texture: dummyGuiTexture,
            view: dummyGuiTexture.createView(),
            sampler: this.guiSampler,
        }
    }

    /**
     * Build or rebuild the pipeline
     */
    async _buildPipeline() {
        if (!this.inputTexture) {
            return
        }

        const textures = [this.inputTexture]
        if (this.noiseTexture) {
            textures.push(this.noiseTexture)
        }
        // Always include bloom texture (real or dummy) so shader bindings are valid
        const effectiveBloomTexture = this.bloomTexture || this.dummyBloomTexture
        textures.push(effectiveBloomTexture)

        // Always include GUI texture (real or dummy) so shader bindings are valid
        const effectiveGuiTexture = this.guiTexture || this.dummyGuiTexture
        textures.push(effectiveGuiTexture)

        const hasBloom = this.bloomTexture && this.bloomEnabled

        // Determine render target: intermediate texture (for CRT) or canvas
        const renderTarget = this.shouldOutputToTexture && this.intermediateTexture
            ? this.intermediateTexture
            : null

        this.pipeline = await Pipeline.create(this.engine, {
            label: 'postProcess',
            wgslSource: postProcessingWGSL,
            isPostProcessing: true,
            textures: textures,
            uniforms: () => ({
                noiseParams: [this.noiseSize, this.noiseAnimated ? Math.random() : 0, this.noiseAnimated ? Math.random() : 0, this.fxaa ? 1.0 : 0.0],
                ditherParams: [this.ditheringEnabled ? 1.0 : 0.0, this.colorLevels, this.tonemapMode, 0],
                bloomParams: [hasBloom ? 1.0 : 0.0, this.bloomIntensity, this.bloomRadius, effectiveBloomTexture?.mipCount ?? 1]
            }),
            renderTarget: renderTarget,
        })

        this._needsRebuild = false
    }

    async _execute(context) {
        const { device, canvas } = this.engine

        // Check if CRT state changed (need to switch between canvas/texture output)
        const needsOutputToTexture = this.shouldOutputToTexture
        const hasIntermediateTexture = !!this.intermediateTexture

        // Detect output mode change
        if (needsOutputToTexture !== this._lastOutputToTexture) {
            this._lastOutputToTexture = needsOutputToTexture
            this._needsRebuild = true

            if (needsOutputToTexture && !hasIntermediateTexture) {
                // CRT was just enabled - create intermediate texture at stored resize dimensions
                // Use resize dimensions (render-scaled), not canvas dimensions (full resolution)
                const w = this._resizeWidth || canvas.width
                const h = this._resizeHeight || canvas.height
                await this._createIntermediateTexture(w, h)
            } else if (!needsOutputToTexture && hasIntermediateTexture) {
                // CRT was just disabled - destroy intermediate texture
                if (this.intermediateTexture?.texture) {
                    this.intermediateTexture.texture.destroy()
                    this.intermediateTexture = null
                }
            }
        }

        // Update GUI texture from canvas if available
        if (this.guiCanvas && this.guiCanvas.width > 0 && this.guiCanvas.height > 0) {
            // Check if we need to recreate the texture (size changed)
            const needsNewTexture = !this.guiTexture ||
                this.guiTexture.width !== this.guiCanvas.width ||
                this.guiTexture.height !== this.guiCanvas.height

            if (needsNewTexture) {
                // Destroy old texture if it exists
                if (this.guiTexture?.texture) {
                    this.guiTexture.texture.destroy()
                }

                // Create new texture matching canvas size
                const texture = device.createTexture({
                    label: 'GUI Texture',
                    size: [this.guiCanvas.width, this.guiCanvas.height, 1],
                    format: 'rgba8unorm',
                    usage: GPUTextureUsage.TEXTURE_BINDING |
                           GPUTextureUsage.COPY_DST |
                           GPUTextureUsage.RENDER_ATTACHMENT,
                })

                this.guiTexture = {
                    texture: texture,
                    view: texture.createView(),
                    sampler: this.guiSampler,
                    width: this.guiCanvas.width,
                    height: this.guiCanvas.height,
                }

                // Force pipeline rebuild to use new texture
                this._needsRebuild = true
            }

            // Copy canvas content to GPU texture
            device.queue.copyExternalImageToTexture(
                { source: this.guiCanvas },
                { texture: this.guiTexture.texture },
                [this.guiCanvas.width, this.guiCanvas.height]
            )
        }

        // Rebuild pipeline if needed
        if (this._needsRebuild) {
            await this._buildPipeline()
        }

        // If rebuild was attempted but failed, don't use stale pipeline with old bind groups
        if (!this.pipeline || this._needsRebuild) {
            console.warn('PostProcessPass: Pipeline not ready')
            return
        }

        // Determine if bloom is effectively enabled
        const hasBloom = this.bloomTexture && this.bloomEnabled
        const effectiveBloomTexture = this.bloomTexture || this.dummyBloomTexture

        // Update uniforms each frame
        this.pipeline.uniformValues.set({
            noiseParams: [this.noiseSize, this.noiseAnimated ? Math.random() : 0, this.noiseAnimated ? Math.random() : 0, this.fxaa ? 1.0 : 0.0],
            ditherParams: [this.ditheringEnabled ? 1.0 : 0.0, this.colorLevels, this.tonemapMode, 0],
            bloomParams: [hasBloom ? 1.0 : 0.0, this.bloomIntensity, this.bloomRadius, effectiveBloomTexture?.mipCount ?? 1]
        })

        // Render to canvas
        this.pipeline.render()
    }

    async _resize(width, height) {
        // Store resize dimensions for runtime texture creation
        this._resizeWidth = width
        this._resizeHeight = height

        // Create intermediate texture for CRT if enabled
        await this._createIntermediateTexture(width, height)

        // Pipeline needs rebuild since canvas size changed
        this._needsRebuild = true
    }

    _destroy() {
        this.pipeline = null
        if (this.dummyBloomTexture?.texture) {
            this.dummyBloomTexture.texture.destroy()
            this.dummyBloomTexture = null
        }
        if (this.guiTexture?.texture) {
            this.guiTexture.texture.destroy()
            this.guiTexture = null
        }
        if (this.dummyGuiTexture?.texture) {
            this.dummyGuiTexture.texture.destroy()
            this.dummyGuiTexture = null
        }
        if (this.intermediateTexture?.texture) {
            this.intermediateTexture.texture.destroy()
            this.intermediateTexture = null
        }
    }
}

export { PostProcessPass }
