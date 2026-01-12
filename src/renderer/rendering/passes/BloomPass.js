import { BasePass } from "./BasePass.js"

import bloomExtractWGSL from "../shaders/bloom.wgsl"
import bloomBlurWGSL from "../shaders/bloom_blur.wgsl"

/**
 * BloomPass - HDR Bloom/Glare effect
 *
 * Extracts bright pixels with exponential falloff and applies
 * two-pass diagonal Gaussian blur for X-shaped glare effect.
 *
 * Input: HDR lighting output
 * Output: Blurred bloom texture with X-shaped glare
 */
class BloomPass extends BasePass {
    constructor(engine = null) {
        super('Bloom', engine)

        // Pipelines
        this.extractPipeline = null
        this.blurPipeline = null

        // Textures (ping-pong for blur)
        this.brightTexture = null   // Extracted bright pixels
        this.blurTextureA = null    // After horizontal blur
        this.blurTextureB = null    // After vertical blur (final output)

        // Resources
        this.inputTexture = null
        this.extractUniformBuffer = null
        this.blurUniformBufferH = null  // Horizontal blur uniforms
        this.blurUniformBufferV = null  // Vertical blur uniforms
        this.extractBindGroup = null
        this.blurBindGroupH = null  // Horizontal blur
        this.blurBindGroupV = null  // Vertical blur
        this.sampler = null

        // Textures pending destruction (wait for GPU to finish using them)
        // Use a ring buffer of 3 frames to ensure GPU is definitely done
        this._pendingDestroyRing = [[], [], []]
        this._pendingDestroyIndex = 0

        // Render dimensions (may differ from canvas when effect scaling is applied)
        this.width = 0
        this.height = 0
        // Bloom internal resolution (scaled down for performance)
        this.bloomWidth = 0
        this.bloomHeight = 0
    }

    // Convenience getters for bloom settings
    get bloomEnabled() { return this.settings?.bloom?.enabled ?? true }
    get intensity() { return this.settings?.bloom?.intensity ?? 1.0 }
    get threshold() { return this.settings?.bloom?.threshold ?? 0.8 }
    get softThreshold() { return this.settings?.bloom?.softThreshold ?? 0.5 }
    get radius() { return this.settings?.bloom?.radius ?? 32 }
    get emissiveBoost() { return this.settings?.bloom?.emissiveBoost ?? 2.0 }
    get maxBrightness() { return this.settings?.bloom?.maxBrightness ?? 4.0 }
    get renderScale() { return this.settings?.rendering?.renderScale ?? 1.0 }
    // Bloom resolution scale - 0.5 = half res (faster), 1.0 = full res (quality)
    get bloomScale() { return this.settings?.bloom?.scale ?? 0.5 }

    /**
     * Set the input texture (HDR lighting output)
     * @param {Object} texture - Input texture with view property
     */
    setInputTexture(texture) {
        // Only rebuild if the texture actually changed
        if (this.inputTexture !== texture) {
            this.inputTexture = texture
            this._needsRebuild = true
        }
    }

    async _init() {
        const { device } = this.engine

        // Create sampler for all bloom textures
        this.sampler = device.createSampler({
            label: 'Bloom Sampler',
            minFilter: 'linear',
            magFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        })
    }

    /**
     * Create or recreate bloom textures at scaled resolution
     */
    _createTextures(width, height) {
        const { device } = this.engine

        // Queue old textures for deferred destruction (GPU may still be using them)
        // Add to current slot in ring buffer - will be destroyed 3 frames later
        const slot = this._pendingDestroyRing[this._pendingDestroyIndex]
        if (this.brightTexture?.texture) slot.push(this.brightTexture.texture)
        if (this.blurTextureA?.texture) slot.push(this.blurTextureA.texture)
        if (this.blurTextureB?.texture) slot.push(this.blurTextureB.texture)

        // Calculate scaled resolution for bloom (0.5 = half res for performance)
        // Bloom is blurry anyway, so half-res is usually sufficient
        const scale = this.bloomScale
        const bloomWidth = Math.max(1, Math.floor(width * scale))
        const bloomHeight = Math.max(1, Math.floor(height * scale))

        // Only log if dimensions actually changed
        if (this.bloomWidth !== bloomWidth || this.bloomHeight !== bloomHeight) {
            console.log(`Bloom: ${width}x${height} -> ${bloomWidth}x${bloomHeight} (scale: ${scale})`)
        }

        this.bloomWidth = bloomWidth
        this.bloomHeight = bloomHeight

        // Create textures at scaled resolution
        const createBloomTexture = (label) => {
            const texture = device.createTexture({
                label,
                size: { width: bloomWidth, height: bloomHeight, depthOrArrayLayers: 1 },
                format: 'rgba16float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            })
            return {
                texture,
                view: texture.createView({ label: `${label} View` }),
                sampler: this.sampler,
                width: bloomWidth,
                height: bloomHeight,
            }
        }

        this.brightTexture = createBloomTexture('Bloom Bright')
        this.blurTextureA = createBloomTexture('Bloom Blur A')
        this.blurTextureB = createBloomTexture('Bloom Blur B')
    }

    async _buildPipeline() {
        if (!this.inputTexture) {
            return
        }

        const { device, canvas } = this.engine
        // Store initial dimensions (may be updated by resize)
        this.width = canvas.width
        this.height = canvas.height

        // Create bloom textures
        this._createTextures(this.width, this.height)

        // Create uniform buffers
        this.extractUniformBuffer = device.createBuffer({
            label: 'Bloom Extract Uniforms',
            size: 32, // 5 floats + padding for 16-byte alignment
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        this.blurUniformBufferH = device.createBuffer({
            label: 'Bloom Blur H Uniforms',
            size: 32, // 8 floats (2 vec2 + 2 float + padding)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        this.blurUniformBufferV = device.createBuffer({
            label: 'Bloom Blur V Uniforms',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // ===== EXTRACT PIPELINE =====
        const extractBGL = device.createBindGroupLayout({
            label: 'Bloom Extract BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        })

        const extractModule = device.createShaderModule({
            label: 'Bloom Extract Shader',
            code: bloomExtractWGSL,
        })

        // ===== BLUR PIPELINE =====
        const blurBGL = device.createBindGroupLayout({
            label: 'Bloom Blur BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        })

        const blurModule = device.createShaderModule({
            label: 'Bloom Blur Shader',
            code: bloomBlurWGSL,
        })

        // Create both pipelines in parallel for faster initialization
        const [extractPipeline, blurPipeline] = await Promise.all([
            device.createRenderPipelineAsync({
                label: 'Bloom Extract Pipeline',
                layout: device.createPipelineLayout({ bindGroupLayouts: [extractBGL] }),
                vertex: { module: extractModule, entryPoint: 'vertexMain' },
                fragment: {
                    module: extractModule,
                    entryPoint: 'fragmentMain',
                    targets: [{ format: 'rgba16float' }],
                },
                primitive: { topology: 'triangle-list' },
            }),
            device.createRenderPipelineAsync({
                label: 'Bloom Blur Pipeline',
                layout: device.createPipelineLayout({ bindGroupLayouts: [blurBGL] }),
                vertex: { module: blurModule, entryPoint: 'vertexMain' },
                fragment: {
                    module: blurModule,
                    entryPoint: 'fragmentMain',
                    targets: [{ format: 'rgba16float' }],
                },
                primitive: { topology: 'triangle-list' },
            })
        ])

        this.extractPipeline = extractPipeline
        this.blurPipeline = blurPipeline

        this.extractBindGroup = device.createBindGroup({
            label: 'Bloom Extract Bind Group',
            layout: extractBGL,
            entries: [
                { binding: 0, resource: { buffer: this.extractUniformBuffer } },
                { binding: 1, resource: this.inputTexture.view },
                { binding: 2, resource: this.sampler },
            ],
        })

        // Horizontal blur: brightTexture -> blurTextureA
        this.blurBindGroupH = device.createBindGroup({
            label: 'Bloom Blur H Bind Group',
            layout: blurBGL,
            entries: [
                { binding: 0, resource: { buffer: this.blurUniformBufferH } },
                { binding: 1, resource: this.brightTexture.view },
                { binding: 2, resource: this.sampler },
            ],
        })

        // Vertical blur: blurTextureA -> blurTextureB
        this.blurBindGroupV = device.createBindGroup({
            label: 'Bloom Blur V Bind Group',
            layout: blurBGL,
            entries: [
                { binding: 0, resource: { buffer: this.blurUniformBufferV } },
                { binding: 1, resource: this.blurTextureA.view },
                { binding: 2, resource: this.sampler },
            ],
        })

        this._needsRebuild = false
    }

    async _execute(context) {
        // Skip if bloom is disabled in settings
        if (!this.bloomEnabled) {
            return
        }

        const { device, canvas } = this.engine

        // Rotate ring buffer and destroy textures from 3 frames ago
        this._pendingDestroyIndex = (this._pendingDestroyIndex + 1) % 3
        const toDestroy = this._pendingDestroyRing[this._pendingDestroyIndex]
        for (const tex of toDestroy) {
            tex.destroy()
        }
        this._pendingDestroyRing[this._pendingDestroyIndex] = []

        // Rebuild pipeline if needed
        if (this._needsRebuild) {
            await this._buildPipeline()
        }

        // If rebuild was attempted but failed, don't use stale pipeline with old bind groups
        if (!this.extractPipeline || !this.blurPipeline || !this.inputTexture || this._needsRebuild) {
            return
        }

        // Use bloom-specific dimensions (may be scaled down for performance)
        const bloomWidth = this.bloomWidth
        const bloomHeight = this.bloomHeight

        // Scale radius based on bloom height relative to 1080p (settings are authored for 1080p)
        // Also scale by bloomScale since we're working at lower resolution
        const heightScale = bloomHeight / 1080
        const blurRadius = this.radius * this.renderScale * heightScale

        // Update all uniforms BEFORE creating command encoder
        // (writeBuffer is immediate, commands are batched)
        device.queue.writeBuffer(this.extractUniformBuffer, 0, new Float32Array([
            this.threshold,
            this.softThreshold,
            this.intensity,
            this.emissiveBoost,
            this.maxBrightness,
            0.0, 0.0, 0.0,  // padding
        ]))

        // Diagonal blur uniforms (X-shaped glare)
        const diag = 0.7071067811865476  // 1/sqrt(2)
        device.queue.writeBuffer(this.blurUniformBufferH, 0, new Float32Array([
            diag, diag,         // direction (diagonal: top-left to bottom-right)
            1.0 / bloomWidth, 1.0 / bloomHeight,  // texelSize (at bloom resolution)
            blurRadius, 0.0,    // blurRadius, padding
            0.0, 0.0,           // more padding for alignment
        ]))

        // Second diagonal blur uniforms
        device.queue.writeBuffer(this.blurUniformBufferV, 0, new Float32Array([
            diag, -diag,        // direction (diagonal: bottom-left to top-right)
            1.0 / bloomWidth, 1.0 / bloomHeight,  // texelSize (at bloom resolution)
            blurRadius, 0.0,    // blurRadius, padding
            0.0, 0.0,           // more padding for alignment
        ]))

        const commandEncoder = device.createCommandEncoder({ label: 'Bloom Pass' })

        // ===== PASS 1: Extract bright pixels =====
        {
            const pass = commandEncoder.beginRenderPass({
                label: 'Bloom Extract',
                colorAttachments: [{
                    view: this.brightTexture.view,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            })
            pass.setPipeline(this.extractPipeline)
            pass.setBindGroup(0, this.extractBindGroup)
            pass.draw(3)
            pass.end()
        }

        // ===== PASS 2: Diagonal blur (top-left to bottom-right) =====
        {
            const pass = commandEncoder.beginRenderPass({
                label: 'Bloom Blur Diag1',
                colorAttachments: [{
                    view: this.blurTextureA.view,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            })
            pass.setPipeline(this.blurPipeline)
            pass.setBindGroup(0, this.blurBindGroupH)
            pass.draw(3)
            pass.end()
        }

        // ===== PASS 3: Diagonal blur (bottom-left to top-right) =====
        {
            const pass = commandEncoder.beginRenderPass({
                label: 'Bloom Blur Diag2',
                colorAttachments: [{
                    view: this.blurTextureB.view,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            })
            pass.setPipeline(this.blurPipeline)
            pass.setBindGroup(0, this.blurBindGroupV)
            pass.draw(3)
            pass.end()
        }

        device.queue.submit([commandEncoder.finish()])
    }

    async _resize(width, height) {
        this.width = width
        this.height = height
        this._needsRebuild = true
    }

    _destroy() {
        if (this.brightTexture?.texture) this.brightTexture.texture.destroy()
        if (this.blurTextureA?.texture) this.blurTextureA.texture.destroy()
        if (this.blurTextureB?.texture) this.blurTextureB.texture.destroy()
        // Clean up any pending textures in all ring buffer slots
        for (const slot of this._pendingDestroyRing) {
            for (const tex of slot) {
                tex.destroy()
            }
        }
        this._pendingDestroyRing = [[], [], []]
        this.extractPipeline = null
        this.blurPipeline = null
    }

    /**
     * Get the final bloom texture (after blur)
     */
    getOutputTexture() {
        return this.blurTextureB
    }

    /**
     * Get the bright extraction texture (before blur)
     * Used by SSGITilePass for directional light accumulation
     */
    getBrightTexture() {
        return this.brightTexture
    }
}


export { BloomPass }
