import { BasePass } from "./BasePass.js"
import { Texture } from "../../Texture.js"

import renderPostWGSL from "../shaders/render_post.wgsl"

/**
 * RenderPostPass - Combines SSGI/Planar Reflection with lighting output in HDR space
 *
 * This pass runs after lighting and SSGI, blending the screen-space
 * effects with the lighting result before tone mapping.
 */
class RenderPostPass extends BasePass {
    constructor(engine = null) {
        super('RenderPost', engine)

        this.outputTexture = null   // HDR output (rgba16float)
        this.pipeline = null
        this.bindGroupLayout = null
        this.uniformBuffer = null
        this.uniformData = null
        this.width = 0
        this.height = 0

        // Noise for jittered sampling (blue noise or bayer dither)
        this.noiseTexture = null
        this.noiseSize = 64
        this.noiseAnimated = true
        this.frameCount = 0

        // Ambient capture (6 directional colors buffer)
        this.ambientCaptureBuffer = null

        // Resources pending destruction (wait for GPU to finish using them)
        this._pendingDestroyRing = [[], [], []]
        this._pendingDestroyIndex = 0
    }

    async _init() {
        const { canvas } = this.engine
        await this._createResources(canvas.width, canvas.height)
    }

    async _createResources(width, height) {
        const { device } = this.engine

        this.width = width
        this.height = height

        // Create output texture (full-res HDR)
        this.outputTexture = await Texture.renderTarget(this.engine, 'rgba16float', width, height)
        this.outputTexture.label = 'renderPostOutput'

        // Uniform buffer: screenSize + flags + intensities + planar settings + blue noise + ambient capture
        // vec2f screenSize (8)
        // f32 ssgiEnabled (4)
        // f32 ssgiIntensity (4)
        // f32 planarEnabled (4)
        // f32 planarGroundLevel (4)
        // f32 planarRoughnessCutoff (4)
        // f32 planarNormalPerturbation (4)
        // f32 noiseSize (4)
        // f32 frameCount (4)
        // f32 planarBlurSamples (4)
        // f32 planarIntensity (4)
        // f32 renderScale (4)
        // f32 ssgiSaturateLevel (4)
        // f32 planarDistanceFade (4)
        // f32 ambientCaptureEnabled (4)
        // f32 ambientCaptureIntensity (4)
        // f32 ambientCaptureFadeDistance (4)
        // f32 cameraNear (4)
        // f32 cameraFar (4)
        // f32 ambientCaptureSaturateLevel (4)
        const uniformSize = 96
        this.uniformData = new ArrayBuffer(uniformSize)
        this.uniformBuffer = device.createBuffer({
            label: 'renderPostUniformBuffer',
            size: uniformSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // Create shader module
        const shaderModule = device.createShaderModule({
            label: 'renderPostShaderModule',
            code: renderPostWGSL,
        })

        // Create bind group layout
        this.bindGroupLayout = device.createBindGroupLayout({
            label: 'renderPostBindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },  // lightingOutput
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },  // ssgiTexture
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },  // gbufferARM
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },  // gbufferNormal
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },  // planarReflection
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },  // noise
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },    // linearSampler
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } }, // nearestSampler
                { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // ambientCapture (6 vec4f)
                { binding: 10, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }, // gbufferDepth
            ],
        })

        // Create placeholder buffer for ambient capture (6 vec4f = 96 bytes)
        this.placeholderAmbientBuffer = device.createBuffer({
            label: 'placeholderAmbientBuffer',
            size: 6 * 4 * 4,  // 6 vec4f
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        })
        // Initialize with neutral gray
        const initColors = new Float32Array(6 * 4)
        for (let i = 0; i < 6; i++) {
            initColors[i * 4 + 0] = 0.0
            initColors[i * 4 + 1] = 0.0
            initColors[i * 4 + 2] = 0.0
            initColors[i * 4 + 3] = 1.0
        }
        device.queue.writeBuffer(this.placeholderAmbientBuffer, 0, initColors)

        // Create render pipeline (async for non-blocking initialization)
        this.pipeline = await device.createRenderPipelineAsync({
            label: 'renderPostPipeline',
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'rgba16float' }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        })

        // Create samplers
        this.linearSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'mirror-repeat',  // Mirror repeat for planar reflection edges
            addressModeV: 'mirror-repeat',
        })

        this.nearestSampler = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        })

        // Create placeholder texture for when effects are disabled (black)
        this.placeholderTexture = await Texture.renderTarget(this.engine, 'rgba16float', 1, 1)

        // Clear placeholder to black
        const clearEncoder = device.createCommandEncoder({ label: 'clearPlaceholder' })
        const clearPass = clearEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.placeholderTexture.view,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        })
        clearPass.end()
        device.queue.submit([clearEncoder.finish()])
    }

    /**
     * Set noise texture for jittered sampling
     * @param {Texture} texture - Noise texture (blue noise or bayer dither)
     * @param {number} size - Texture size (width/height)
     * @param {boolean} animated - Whether to animate noise offset each frame
     */
    setNoise(texture, size, animated = true) {
        this.noiseTexture = texture
        this.noiseSize = size || 64
        this.noiseAnimated = animated
    }

    /**
     * Set ambient capture buffer (6 directional colors from AmbientCapturePass)
     * @param {GPUBuffer} buffer - Storage buffer with 6 vec4f colors
     */
    setAmbientCaptureBuffer(buffer) {
        this.ambientCaptureBuffer = buffer
    }

    /**
     * Execute RenderPost pass
     *
     * @param {Object} context
     * @param {Object} context.lightingOutput - HDR output from lighting pass
     * @param {Object} context.gbuffer - GBuffer for ARM and Normal
     * @param {Object} context.ssgi - SSGI texture (or null)
     * @param {Object} context.planarReflection - Planar reflection texture (or null)
     */
    async _execute(context) {
        const { device } = this.engine
        const { lightingOutput, gbuffer, ssgi, planarReflection, camera } = context

        // Process deferred resource destruction (3 frames delayed)
        this._pendingDestroyIndex = (this._pendingDestroyIndex + 1) % 3
        const toDestroy = this._pendingDestroyRing[this._pendingDestroyIndex]
        for (const res of toDestroy) {
            res.destroy()
        }
        this._pendingDestroyRing[this._pendingDestroyIndex] = []

        if (!lightingOutput || !gbuffer) {
            return
        }

        // Increment frame counter for temporal jitter
        this.frameCount++

        // Get settings
        const ssgiSettings = this.settings?.ssgi
        const planarSettings = this.settings?.planarReflection
        const ambientSettings = this.settings?.ambientCapture

        const ssgiEnabled = ssgiSettings?.enabled && ssgi
        const planarEnabled = planarSettings?.enabled && planarReflection
        const ambientEnabled = ambientSettings?.enabled && this.ambientCaptureBuffer

        // Update uniforms
        const view = new DataView(this.uniformData)
        view.setFloat32(0, this.width, true)
        view.setFloat32(4, this.height, true)
        view.setFloat32(8, ssgiEnabled ? 1.0 : 0.0, true)
        view.setFloat32(12, ssgiSettings?.intensity || 1.0, true)  // ssgiIntensity
        view.setFloat32(16, planarEnabled ? 1.0 : 0.0, true)
        view.setFloat32(20, planarSettings?.groundLevel ?? 0.0, true)
        view.setFloat32(24, planarSettings?.roughnessCutoff ?? 0.5, true)
        view.setFloat32(28, planarSettings?.normalPerturbation ?? 0.1, true)
        view.setFloat32(32, this.noiseSize, true)
        view.setFloat32(36, this.noiseAnimated ? this.frameCount : 0, true)
        view.setFloat32(40, planarSettings?.blurSamples ?? 8, true)
        view.setFloat32(44, planarSettings?.intensity ?? 0.9, true)
        view.setFloat32(48, this.settings?.rendering?.renderScale ?? 1.0, true)
        view.setFloat32(52, ssgiSettings?.saturateLevel ?? 0.5, true)
        view.setFloat32(56, planarSettings?.distanceFade ?? 1.0, true)
        view.setFloat32(60, ambientEnabled ? 1.0 : 0.0, true)
        view.setFloat32(64, ambientSettings?.intensity ?? 0.2, true)
        view.setFloat32(68, ambientSettings?.maxDistance ?? 25, true)  // fade distance
        view.setFloat32(72, camera?.near ?? 0.1, true)  // cameraNear
        view.setFloat32(76, camera?.far ?? 1000, true)  // cameraFar
        view.setFloat32(80, ambientSettings?.saturateLevel ?? 0.5, true)  // ambientCaptureSaturateLevel

        device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

        // Use placeholder textures/buffers if effects are disabled
        const ssgiTexture = ssgiEnabled ? ssgi : this.placeholderTexture
        const planarTexture = planarEnabled ? planarReflection : this.placeholderTexture
        const noiseView = this.noiseTexture?.view || this.placeholderTexture.view
        const ambientBuffer = this.ambientCaptureBuffer || this.placeholderAmbientBuffer

        // Create bind group
        const bindGroup = device.createBindGroup({
            label: 'renderPostBindGroup',
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: lightingOutput.view },
                { binding: 2, resource: ssgiTexture.view },
                { binding: 3, resource: gbuffer.arm.view },
                { binding: 4, resource: gbuffer.normal.view },
                { binding: 5, resource: planarTexture.view },
                { binding: 6, resource: noiseView },
                { binding: 7, resource: this.linearSampler },
                { binding: 8, resource: this.nearestSampler },
                { binding: 9, resource: { buffer: ambientBuffer } },
                { binding: 10, resource: gbuffer.depth.view },
            ],
        })

        // Render
        const commandEncoder = device.createCommandEncoder({ label: 'renderPostCommandEncoder' })

        const passEncoder = commandEncoder.beginRenderPass({
            label: 'renderPostRenderPass',
            colorAttachments: [{
                view: this.outputTexture.view,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        })

        passEncoder.setPipeline(this.pipeline)
        passEncoder.setBindGroup(0, bindGroup)
        passEncoder.draw(3)
        passEncoder.end()

        device.queue.submit([commandEncoder.finish()])
    }

    /**
     * Get output texture for ScreenPostPass
     * @returns {Texture} HDR output with SSGI/Planar applied
     */
    getOutputTexture() {
        return this.outputTexture
    }

    async _resize(width, height) {
        this._queueResourcesForDestruction()
        await this._createResources(width, height)
    }

    _queueResourcesForDestruction() {
        const slot = this._pendingDestroyRing[this._pendingDestroyIndex]
        if (this.outputTexture?.texture) {
            slot.push(this.outputTexture.texture)
            this.outputTexture = null
        }
        if (this.placeholderTexture?.texture) {
            slot.push(this.placeholderTexture.texture)
            this.placeholderTexture = null
        }
        if (this.uniformBuffer) {
            slot.push(this.uniformBuffer)
            this.uniformBuffer = null
        }
    }

    _destroyResources() {
        if (this.outputTexture?.texture) {
            this.outputTexture.texture.destroy()
            this.outputTexture = null
        }
        if (this.placeholderTexture?.texture) {
            this.placeholderTexture.texture.destroy()
            this.placeholderTexture = null
        }
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy()
            this.uniformBuffer = null
        }
        if (this.placeholderAmbientBuffer) {
            this.placeholderAmbientBuffer.destroy()
            this.placeholderAmbientBuffer = null
        }
    }

    _destroy() {
        this._destroyResources()
        // Clean up any pending resources in ring buffer
        for (const slot of this._pendingDestroyRing) {
            for (const res of slot) {
                res.destroy()
            }
        }
        this._pendingDestroyRing = [[], [], []]
        this.pipeline = null
        this.bindGroupLayout = null
    }
}

export { RenderPostPass }
