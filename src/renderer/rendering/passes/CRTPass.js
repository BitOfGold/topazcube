import { BasePass } from "./BasePass.js"
import { Pipeline } from "../../Pipeline.js"

import crtWGSL from "../shaders/crt.wgsl"

/**
 * CRTPass - CRT monitor simulation effect
 *
 * Applies retro CRT effects: screen curvature, scanlines, RGB convergence,
 * phosphor mask, and vignette. Optionally upscales the input for a pixelated look.
 *
 * Input: SDR image from PostProcessPass (rendered to intermediate texture)
 * Output: Final image with CRT effects to canvas
 */
class CRTPass extends BasePass {
    constructor(engine = null) {
        super('CRT', engine)

        this.pipeline = null
        this.inputTexture = null

        // Upscaled texture (for pixelated look)
        this.upscaledTexture = null
        this.upscaledWidth = 0
        this.upscaledHeight = 0

        // Samplers
        this.linearSampler = null
        this.nearestSampler = null

        // Phosphor mask texture (procedural placeholder)
        this.phosphorMaskTexture = null

        // Canvas dimensions
        this.canvasWidth = 0
        this.canvasHeight = 0

        // Render size (before upscale)
        this.renderWidth = 0
        this.renderHeight = 0
    }

    // Settings getters
    get crtSettings() { return this.engine?.settings?.crt ?? {} }
    get crtEnabled() { return this.crtSettings.enabled ?? false }
    get upscaleEnabled() { return this.crtSettings.upscaleEnabled ?? false }
    get upscaleTarget() { return this.crtSettings.upscaleTarget ?? 4 }
    get maxTextureSize() { return this.crtSettings.maxTextureSize ?? 4096 }

    // Geometry
    get curvature() { return this.crtSettings.curvature ?? 0.03 }
    get cornerRadius() { return this.crtSettings.cornerRadius ?? 0.03 }
    get zoom() { return this.crtSettings.zoom ?? 1.0 }

    // Scanlines
    get scanlineIntensity() { return this.crtSettings.scanlineIntensity ?? 0.25 }
    get scanlineWidth() { return this.crtSettings.scanlineWidth ?? 0.5 }
    get scanlineBrightBoost() { return this.crtSettings.scanlineBrightBoost ?? 1.0 }
    get scanlineHeight() { return this.crtSettings.scanlineHeight ?? 3 }  // pixels per scanline

    // Convergence
    get convergence() { return this.crtSettings.convergence ?? [0.5, 0.0, -0.5] }

    // Phosphor mask
    get maskType() {
        const type = this.crtSettings.maskType ?? 'aperture'
        switch (type) {
            case 'none': return 0
            case 'aperture': return 1
            case 'slot': return 2
            case 'shadow': return 3
            default: return 1
        }
    }
    get maskIntensity() { return this.crtSettings.maskIntensity ?? 0.15 }
    get maskScale() { return this.crtSettings.maskScale ?? 1.0 }

    // Vignette
    get vignetteIntensity() { return this.crtSettings.vignetteIntensity ?? 0.15 }
    get vignetteSize() { return this.crtSettings.vignetteSize ?? 0.4 }

    // Blur
    get blurSize() { return this.crtSettings.blurSize ?? 0.5 }

    /**
     * Calculate brightness compensation for phosphor mask
     * Pre-computed on CPU to avoid per-pixel calculation in shader
     * @param {number} maskType - 0=none, 1=aperture, 2=slot, 3=shadow
     * @param {number} intensity - mask intensity 0-1
     * @returns {number} compensation multiplier
     */
    _calculateMaskCompensation(maskType, intensity) {
        if (maskType < 0.5 || intensity <= 0) {
            return 1.0
        }

        // Darkening factors tuned per mask type
        let darkening
        let useLinearOnly = false

        if (maskType < 1.5) {
            // Aperture grille
            darkening = 0.25
        } else if (maskType < 2.5) {
            // Slot mask
            darkening = 0.27
        } else {
            // Shadow mask: linear formula works perfectly, don't blend to exp
            darkening = 0.82
            useLinearOnly = true
        }

        // Linear formula: 1 / (1 - intensity * darkening)
        const linearComp = 1.0 / Math.max(1.0 - intensity * darkening, 0.1)

        // Shadow uses linear only (works well at all intensities)
        if (useLinearOnly) {
            return linearComp
        }

        // Aperture/Slot: blend to exp at high intensities to avoid over-brightening
        const expComp = Math.exp(intensity * darkening)
        const t = Math.max(0, Math.min(1, (intensity - 0.4) / 0.2))
        const blendFactor = t * t * (3 - 2 * t) // smoothstep

        return linearComp * (1 - blendFactor) + expComp * blendFactor
    }

    /**
     * Set the input texture (from PostProcessPass intermediate output)
     * @param {Object} texture - Input texture object
     */
    setInputTexture(texture) {
        if (this.inputTexture !== texture) {
            this.inputTexture = texture
            this._needsRebuild = true
            this._blitPipeline = null  // Invalidate blit pipeline (has bind group with old texture)
        }
    }

    /**
     * Set the render size (before upscaling)
     * @param {number} width - Render width
     * @param {number} height - Render height
     */
    setRenderSize(width, height) {
        if (this.renderWidth !== width || this.renderHeight !== height) {
            this.renderWidth = width
            this.renderHeight = height
            this._needsUpscaleRebuild = true
        }
    }

    /**
     * Calculate the upscaled texture size
     * @returns {{width: number, height: number, scale: number}}
     */
    _calculateUpscaledSize() {
        const renderW = this.renderWidth || this.canvasWidth
        const renderH = this.renderHeight || this.canvasHeight
        const maxSize = this.maxTextureSize

        // Find the largest integer scale that fits within limits
        // This avoids non-integer ratios that cause moiré/checkerboard patterns
        let scale = this.upscaleTarget
        while (scale > 1 && (renderW * scale > maxSize || renderH * scale > maxSize)) {
            scale--
        }

        // Also limit to 2x canvas size (no benefit beyond display resolution)
        const maxCanvasScale = 2.0
        while (scale > 1 && (renderW * scale > this.canvasWidth * maxCanvasScale ||
                             renderH * scale > this.canvasHeight * maxCanvasScale)) {
            scale--
        }

        // Ensure at least 1x
        scale = Math.max(scale, 1)

        const targetW = renderW * scale
        const targetH = renderH * scale

        return { width: targetW, height: targetH, scale }
    }

    /**
     * Check if actual upscaling is needed
     * Returns true only if the upscaled texture would be larger than the input
     */
    _needsUpscaling() {
        if (!this.upscaleEnabled) return false
        const { scale } = this._calculateUpscaledSize()
        return scale > 1
    }

    async _init() {
        const { device } = this.engine

        // Create samplers
        this.linearSampler = device.createSampler({
            label: 'CRT Linear Sampler',
            minFilter: 'linear',
            magFilter: 'linear',
            addressModeU: 'mirror-repeat',
            addressModeV: 'mirror-repeat',
        })

        this.nearestSampler = device.createSampler({
            label: 'CRT Nearest Sampler',
            minFilter: 'nearest',
            magFilter: 'nearest',
            addressModeU: 'mirror-repeat',
            addressModeV: 'mirror-repeat',
        })

        // Create dummy phosphor mask texture (1x1 white - will be replaced)
        await this._createPhosphorMaskTexture()
    }

    /**
     * Create phosphor mask texture
     * This is a simple procedural texture for the aperture grille pattern
     */
    async _createPhosphorMaskTexture() {
        const { device } = this.engine

        // Create a 6x2 texture for aperture grille pattern
        // Pattern: R G B R G B (repeated vertically)
        const size = 6
        const data = new Uint8Array(size * 2 * 4)

        // Aperture grille pattern
        for (let y = 0; y < 2; y++) {
            for (let x = 0; x < size; x++) {
                const idx = (y * size + x) * 4
                const phase = x % 3

                // RGB stripes with some bleed
                if (phase === 0) {
                    data[idx] = 255     // R
                    data[idx + 1] = 50  // G
                    data[idx + 2] = 50  // B
                } else if (phase === 1) {
                    data[idx] = 50      // R
                    data[idx + 1] = 255 // G
                    data[idx + 2] = 50  // B
                } else {
                    data[idx] = 50      // R
                    data[idx + 1] = 50  // G
                    data[idx + 2] = 255 // B
                }
                data[idx + 3] = 255     // A
            }
        }

        const texture = device.createTexture({
            label: 'Phosphor Mask',
            size: [size, 2, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })

        device.queue.writeTexture(
            { texture },
            data,
            { bytesPerRow: size * 4 },
            { width: size, height: 2 }
        )

        const sampler = device.createSampler({
            label: 'Phosphor Mask Sampler',
            minFilter: 'nearest',
            magFilter: 'nearest',
            addressModeU: 'repeat',
            addressModeV: 'repeat',
        })

        this.phosphorMaskTexture = {
            texture,
            view: texture.createView(),
            sampler
        }
    }

    /**
     * Create or resize the upscaled texture
     */
    async _createUpscaledTexture() {
        const { device } = this.engine

        const { width, height, scale } = this._calculateUpscaledSize()

        // Skip if size hasn't changed
        if (this.upscaledWidth === width && this.upscaledHeight === height) {
            return
        }

        // Destroy old texture
        if (this.upscaledTexture?.texture) {
            this.upscaledTexture.texture.destroy()
        }

        // Create new upscaled texture
        const texture = device.createTexture({
            label: 'CRT Upscaled Texture',
            size: [width, height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING |
                   GPUTextureUsage.RENDER_ATTACHMENT |
                   GPUTextureUsage.COPY_DST,
        })

        this.upscaledTexture = {
            texture,
            view: texture.createView(),
            width,
            height,
            scale,
            format: 'rgba8unorm',
        }

        this.upscaledWidth = width
        this.upscaledHeight = height

        console.log(`CRTPass: Created upscaled texture ${width}x${height} (${scale.toFixed(1)}x)`)

        this._needsRebuild = true
        this._blitPipeline = null  // Blit pipeline render target changed
    }

    /**
     * Build or rebuild the CRT pipeline
     */
    async _buildPipeline() {
        if (!this.inputTexture && !this.upscaledTexture) {
            return
        }

        const { device } = this.engine

        // Check if actual upscaling is needed (scale > 1)
        const needsUpscaling = this._needsUpscaling()

        // Determine which texture to use as input
        // When scale = 1, use input directly to avoid unnecessary copy and potential precision issues
        const effectiveInput = (this.crtEnabled || this.upscaleEnabled) && needsUpscaling && this.upscaledTexture
            ? this.upscaledTexture
            : this.inputTexture

        if (!effectiveInput) return

        // Create bind group layout
        const bindGroupLayout = device.createBindGroupLayout({
            label: 'CRT Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ]
        })

        // Create uniform buffer (must match shader struct alignment)
        // Total: 18 floats = 72 bytes, padded to 80 for alignment
        const uniformBuffer = device.createBuffer({
            label: 'CRT Uniforms',
            size: 128, // Padded for alignment
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // Create bind group
        const bindGroup = device.createBindGroup({
            label: 'CRT Bind Group',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: effectiveInput.view },
                { binding: 2, resource: this.linearSampler },
                { binding: 3, resource: this.nearestSampler },
                { binding: 4, resource: this.phosphorMaskTexture.view },
                { binding: 5, resource: this.phosphorMaskTexture.sampler },
            ]
        })

        // Create pipeline layout
        const pipelineLayout = device.createPipelineLayout({
            label: 'CRT Pipeline Layout',
            bindGroupLayouts: [bindGroupLayout]
        })

        // Create shader module
        const shaderModule = device.createShaderModule({
            label: 'CRT Shader',
            code: crtWGSL,
        })

        // Create render pipeline
        const pipeline = device.createRenderPipeline({
            label: 'CRT Pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        })

        this.pipeline = {
            pipeline,
            bindGroup,
            uniformBuffer,
        }

        // Track which textures the pipeline was built with
        this._pipelineInputTexture = this.inputTexture
        // Track effective upscaled texture (null when scale = 1)
        this._pipelineUpscaledTexture = needsUpscaling ? this.upscaledTexture : null

        this._needsRebuild = false
    }

    /**
     * Upscale the input texture using nearest-neighbor filtering
     */
    _upscaleInput() {
        if (!this.inputTexture || !this.upscaledTexture) return

        const { device } = this.engine

        // Create a simple blit pipeline for nearest-neighbor upscaling
        // We use copyTextureToTexture if sizes match, otherwise render with nearest sampling

        const commandEncoder = device.createCommandEncoder({ label: 'CRT Upscale' })

        // If input and upscale sizes are the same, just copy
        if (this.inputTexture.width === this.upscaledTexture.width &&
            this.inputTexture.height === this.upscaledTexture.height) {
            commandEncoder.copyTextureToTexture(
                { texture: this.inputTexture.texture },
                { texture: this.upscaledTexture.texture },
                [this.inputTexture.width, this.inputTexture.height]
            )
        } else {
            // Render with nearest sampling for upscale
            // Check if blit pipeline needs recreation (input texture changed)
            if (!this._blitPipeline || this._blitInputTexture !== this.inputTexture) {
                this._createBlitPipeline()
            }

            if (this._blitPipeline) {
                const renderPass = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: this.upscaledTexture.view,
                        loadOp: 'clear',
                        storeOp: 'store',
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    }]
                })

                renderPass.setPipeline(this._blitPipeline.pipeline)
                renderPass.setBindGroup(0, this._blitPipeline.bindGroup)
                renderPass.draw(3, 1, 0, 0)
                renderPass.end()
            }
        }

        device.queue.submit([commandEncoder.finish()])
    }

    /**
     * Create a simple nearest-neighbor blit pipeline
     */
    _createBlitPipeline() {
        if (!this.inputTexture) return

        const { device } = this.engine

        // Track which texture this pipeline was created for
        this._blitInputTexture = this.inputTexture

        const blitShader = `
            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) uv: vec2f,
            }

            @group(0) @binding(0) var inputTexture: texture_2d<f32>;
            @group(0) @binding(1) var inputSampler: sampler;

            @vertex
            fn vertexMain(@builtin(vertex_index) idx: u32) -> VertexOutput {
                var output: VertexOutput;
                let x = f32(idx & 1u) * 4.0 - 1.0;
                let y = f32(idx >> 1u) * 4.0 - 1.0;
                output.position = vec4f(x, y, 0.0, 1.0);
                output.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
                return output;
            }

            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                return textureSample(inputTexture, inputSampler, input.uv);
            }
        `

        const shaderModule = device.createShaderModule({
            label: 'Blit Shader',
            code: blitShader,
        })

        const bindGroupLayout = device.createBindGroupLayout({
            label: 'Blit Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ]
        })

        const bindGroup = device.createBindGroup({
            label: 'Blit Bind Group',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: this.inputTexture.view },
                { binding: 1, resource: this.nearestSampler },
            ]
        })

        const pipeline = device.createRenderPipeline({
            label: 'Blit Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'rgba8unorm' }],
            },
            primitive: { topology: 'triangle-list' },
        })

        this._blitPipeline = { pipeline, bindGroup }
    }

    async _execute(context) {
        const { device, canvas } = this.engine

        // Check if actual upscaling is needed (scale > 1)
        const needsUpscaling = this._needsUpscaling()

        // Track which texture we'll actually use
        const effectiveUpscaledTexture = needsUpscaling ? this.upscaledTexture : null

        // Check if textures have changed (invalidates pipelines)
        if (this.pipeline && (
            this._pipelineInputTexture !== this.inputTexture ||
            this._pipelineUpscaledTexture !== effectiveUpscaledTexture
        )) {
            this._needsRebuild = true
            this._blitPipeline = null  // Also invalidate blit pipeline
        }

        // Skip if nothing to do
        if (!this.crtEnabled && !this.upscaleEnabled) {
            // Passthrough - but we need an input texture
            if (!this.inputTexture) return

            // Rebuild pipeline if needed for passthrough
            if (this._needsRebuild || !this.pipeline) {
                await this._buildPipeline()
            }
        } else {
            // CRT or upscale enabled
            // Only create/use upscaled texture if actual upscaling is needed
            if (needsUpscaling) {
                if (this._needsUpscaleRebuild) {
                    await this._createUpscaledTexture()
                    this._needsUpscaleRebuild = false
                }

                // Upscale the input (blit pipeline is rebuilt inside if needed)
                if (this.inputTexture && this.upscaledTexture) {
                    this._upscaleInput()
                }
            }

            // Rebuild CRT pipeline if needed
            if (this._needsRebuild || !this.pipeline) {
                await this._buildPipeline()
            }
        }

        if (!this.pipeline) return

        // Update uniforms
        const uniformData = new Float32Array(32) // 128 bytes / 4

        // Canvas size (vec2f)
        uniformData[0] = this.canvasWidth
        uniformData[1] = this.canvasHeight

        // Input size (vec2f) - use upscaled only if actually upscaling
        const inputW = needsUpscaling && this.upscaledTexture ? this.upscaledTexture.width : (this.inputTexture?.width || this.canvasWidth)
        const inputH = needsUpscaling && this.upscaledTexture ? this.upscaledTexture.height : (this.inputTexture?.height || this.canvasHeight)
        uniformData[2] = inputW
        uniformData[3] = inputH

        // Render size (vec2f) - determines scanline count when scanlineCount=0
        const renderW = this.renderWidth || this.canvasWidth
        const renderH = this.renderHeight || this.canvasHeight
        uniformData[4] = renderW
        uniformData[5] = renderH

        // Debug: log dimensions on resize to verify pixel-perfect scanlines
        if (!this._loggedDimensions) {
            const renderScale = renderW > 0 ? (renderW / this.canvasWidth).toFixed(2) : '?'
            const upscaleInfo = needsUpscaling ? `upscale=${this.upscaledTexture?.scale || '?'}x` : 'no-upscale (direct)'
            console.log(`CRT: canvas=${this.canvasWidth}x${this.canvasHeight}, render=${renderW}x${renderH} (scale ${renderScale}), input=${inputW}x${inputH}, ${upscaleInfo}`)
            console.log(`CRT: Scanlines use fragment coords (${this.canvasHeight}px), should repeat every ${this.scanlineHeight}px = ${Math.floor(this.canvasHeight / this.scanlineHeight)} scanlines`)
            this._loggedDimensions = true
        }

        // Geometry (4 floats: curvature, cornerRadius, zoom, pad)
        uniformData[6] = this.curvature
        uniformData[7] = this.cornerRadius
        uniformData[8] = this.zoom
        uniformData[9] = 0 // _padGeom

        // Scanlines (4 floats)
        uniformData[10] = this.scanlineIntensity
        uniformData[11] = this.scanlineWidth
        uniformData[12] = this.scanlineBrightBoost
        uniformData[13] = this.scanlineHeight  // pixels per scanline (e.g. 3)

        // Padding for vec3f alignment (convergence needs 16-byte alignment)
        uniformData[14] = 0
        uniformData[15] = 0

        // Convergence (vec3f at offset 64, aligned to 16 bytes)
        const conv = this.convergence
        uniformData[16] = conv[0]
        uniformData[17] = conv[1]
        uniformData[18] = conv[2]
        uniformData[19] = 0 // _pad2

        // Phosphor mask (3 floats)
        uniformData[20] = this.maskType
        uniformData[21] = this.maskIntensity
        uniformData[22] = this.maskScale

        // Pre-calculate mask brightness compensation (avoid per-pixel calculation)
        const maskCompensation = this._calculateMaskCompensation(this.maskType, this.maskIntensity)
        uniformData[23] = maskCompensation

        // Vignette (2 floats)
        uniformData[24] = this.vignetteIntensity
        uniformData[25] = this.vignetteSize

        // Blur (1 float)
        uniformData[26] = this.blurSize

        // Flags (2 floats)
        uniformData[27] = this.crtEnabled ? 1.0 : 0.0
        uniformData[28] = this.upscaleEnabled ? 1.0 : 0.0

        device.queue.writeBuffer(this.pipeline.uniformBuffer, 0, uniformData)

        // Render to canvas
        const commandEncoder = device.createCommandEncoder({ label: 'CRT Render' })

        const canvasTexture = this.engine.context.getCurrentTexture()
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: canvasTexture.createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }]
        })

        renderPass.setPipeline(this.pipeline.pipeline)
        renderPass.setBindGroup(0, this.pipeline.bindGroup)
        renderPass.draw(3, 1, 0, 0)
        renderPass.end()

        device.queue.submit([commandEncoder.finish()])
    }

    async _resize(width, height) {
        this.canvasWidth = width
        this.canvasHeight = height
        this._needsUpscaleRebuild = true
        this._needsRebuild = true
        this._loggedDimensions = false  // Re-log dimensions on resize
    }

    _destroy() {
        if (this.pipeline?.uniformBuffer) {
            this.pipeline.uniformBuffer.destroy()
        }
        if (this.upscaledTexture?.texture) {
            this.upscaledTexture.texture.destroy()
        }
        if (this.phosphorMaskTexture?.texture) {
            this.phosphorMaskTexture.texture.destroy()
        }
        this.pipeline = null
        this._blitPipeline = null
    }
}

export { CRTPass }
