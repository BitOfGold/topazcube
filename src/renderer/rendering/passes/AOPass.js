import { BasePass } from "./BasePass.js"
import { Texture } from "../../Texture.js"

import aoWGSL from "../shaders/ao.wgsl"

/**
 * AOPass - Screen Space Ambient Occlusion
 *
 * Pass 5 in the 7-pass pipeline.
 * Calculates ambient occlusion from depth and normal buffers.
 *
 * Features:
 * - Cavity/corner darkening (traditional SSAO)
 * - Normal-based darkening (plasticity for objects in shadow)
 * - Blue noise jittered sampling
 * - Reflectivity-based fade (reflective surfaces have no AO)
 * - Distance fade (fades to 0 at configurable distance)
 *
 * Inputs: GBuffer (depth, normal, ARM)
 * Output: AO texture (r8unorm - single channel)
 */
class AOPass extends BasePass {
    constructor(engine = null) {
        super('AO', engine)

        this.renderPipeline = null
        this.outputTexture = null
        this.gbuffer = null
        this.noiseTexture = null
        this.noiseSize = 128
        this.noiseAnimated = true

        // Render dimensions (may differ from canvas when effect scaling is applied)
        this.width = 0
        this.height = 0
    }

    // Convenience getters for AO settings (with defaults for backward compatibility)
    get aoIntensity() { return this.settings?.ao?.intensity ?? 1.0 }
    get aoRadius() {
        // Scale radius by renderScale and height relative to 1080p
        // Settings are authored for 1080p, so scale proportionally
        const baseRadius = this.settings?.ao?.radius ?? 64.0
        const renderScale = this.settings?.rendering?.renderScale ?? 1.0
        const heightScale = this.height > 0 ? this.height / 1080 : 1.0
        return baseRadius * renderScale * heightScale
    }
    get aoFadeDistance() { return this.settings?.ao?.fadeDistance ?? 40.0 }
    get aoBias() { return this.settings?.ao?.bias ?? 0.005 }
    get sampleCount() { return this.settings?.ao?.sampleCount ?? 16 }
    get aoLevel() { return this.settings?.ao?.level ?? 0.5 }

    /**
     * Set the GBuffer from GBufferPass
     * @param {GBuffer} gbuffer - GBuffer textures
     */
    async setGBuffer(gbuffer) {
        this.gbuffer = gbuffer
        this._needsRebuild = true
    }

    /**
     * Set the noise texture for jittering
     * @param {Texture} noise - Noise texture (blue noise or bayer dither)
     * @param {number} size - Texture size
     * @param {boolean} animated - Whether to animate noise offset each frame
     */
    setNoise(noise, size = 64, animated = true) {
        this.noiseTexture = noise
        this.noiseSize = size
        this.noiseAnimated = animated
        this._needsRebuild = true
    }

    async _init() {
        // Create output texture (single channel AO)
        this.outputTexture = await Texture.renderTarget(this.engine, 'r8unorm')
    }

    async _buildPipeline() {
        if (!this.gbuffer) {
            return
        }

        const { device } = this.engine

        // Create bind group layout
        const bglEntries = [
            // Uniforms
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            // Depth texture
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
            // Normal texture
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
            // ARM texture (for reflectivity)
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        ]

        // Add blue noise if available
        if (this.noiseTexture) {
            bglEntries.push(
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
            )
        }

        const bindGroupLayout = device.createBindGroupLayout({
            label: 'AO Bind Group Layout',
            entries: bglEntries,
        })

        // Create uniform buffer (256 bytes minimum required by WebGPU)
        // inverseProjection(64) + projection(64) + view(64) + canvasSize(8) + aoParams(16) + noiseParams(16) + cameraParams(8) + padding(16) = 256 bytes
        this.uniformBuffer = device.createBuffer({
            label: 'AO Uniforms',
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // Create shader module
        const shaderModule = device.createShaderModule({
            label: 'AO Shader',
            code: aoWGSL,
        })

        // Check for compilation errors
        const compilationInfo = await shaderModule.getCompilationInfo()
        for (const message of compilationInfo.messages) {
            if (message.type === 'error') {
                console.error('AO Shader Error:', message.message)
                return
            }
        }

        // Create pipeline
        const pipelineLayout = device.createPipelineLayout({
            label: 'AO Pipeline Layout',
            bindGroupLayouts: [bindGroupLayout],
        })

        // Use async pipeline creation for non-blocking initialization
        this.renderPipeline = await device.createRenderPipelineAsync({
            label: 'AO Pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'r8unorm' }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        })

        // Create bind group
        const entries = [
            { binding: 0, resource: { buffer: this.uniformBuffer } },
            { binding: 1, resource: this.gbuffer.depth.view },
            { binding: 2, resource: this.gbuffer.normal.view },
            { binding: 3, resource: this.gbuffer.arm.view },
        ]

        if (this.noiseTexture) {
            entries.push({ binding: 4, resource: this.noiseTexture.view })
        }

        this.bindGroup = device.createBindGroup({
            label: 'AO Bind Group',
            layout: bindGroupLayout,
            entries: entries,
        })

        this.bindGroupLayout = bindGroupLayout
        this._needsRebuild = false
    }

    async _execute(context) {
        const { device, canvas } = this.engine
        const { camera } = context

        // Check if AO is enabled - if not, clear to white (no occlusion)
        if (!this.settings?.ao?.enabled) {
            if (this.outputTexture) {
                const commandEncoder = device.createCommandEncoder({ label: 'AOClear' })
                const passEncoder = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: this.outputTexture.view,
                        clearValue: { r: 1, g: 1, b: 1, a: 1 },  // White = no occlusion
                        loadOp: 'clear',
                        storeOp: 'store',
                    }],
                })
                passEncoder.end()
                device.queue.submit([commandEncoder.finish()])
            }
            return
        }

        // Rebuild pipeline if needed
        if (this._needsRebuild) {
            await this._buildPipeline()
        }

        // If rebuild was attempted but failed, don't use stale pipeline with old bind groups
        if (!this.renderPipeline || !this.gbuffer || this._needsRebuild) {
            return
        }

        // Update uniforms
        const uniformData = new Float32Array(64) // 256 bytes / 4

        // Inverse projection matrix (for reconstructing view-space position)
        if (camera.iProj) {
            uniformData.set(camera.iProj, 0)
        }

        // Projection matrix
        uniformData.set(camera.proj, 16)

        // View matrix
        uniformData.set(camera.view, 32)

        // AO size (vec2f at offset 192 = float index 48)
        // Use stored dimensions (may differ from canvas when effect scaling is applied)
        uniformData[48] = this.width || canvas.width
        uniformData[49] = this.height || canvas.height

        // GBuffer size (vec2f at offset 200 = float index 50)
        // Always full resolution canvas size for GBuffer sampling
        uniformData[50] = canvas.width
        uniformData[51] = canvas.height

        // AO parameters: intensity, radius, fadeDistance, bias (vec4f at offset 208 = float index 52)
        uniformData[52] = this.aoIntensity
        uniformData[53] = this.aoRadius
        uniformData[54] = this.aoFadeDistance
        uniformData[55] = this.aoBias

        // Noise parameters: size, offsetX, offsetY, frame (vec4f at offset 224 = float index 56)
        uniformData[56] = this.noiseSize
        uniformData[57] = this.noiseAnimated ? (Math.random() * 0.1) : 0  // Animated offset X
        uniformData[58] = this.noiseAnimated ? (Math.random() * 0.1) : 0  // Animated offset Y
        uniformData[59] = performance.now() / 1000  // Time for animation

        // Camera near/far (vec2f at offset 240 = float index 60)
        uniformData[60] = camera.near || 0.1
        uniformData[61] = camera.far || 1000

        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData)

        // Render AO pass
        const commandEncoder = device.createCommandEncoder({ label: 'AO Pass' })

        const renderPass = commandEncoder.beginRenderPass({
            label: 'AO Render Pass',
            colorAttachments: [{
                view: this.outputTexture.view,
                clearValue: { r: 1, g: 1, b: 1, a: 1 },  // White = no occlusion
                loadOp: 'clear',
                storeOp: 'store',
            }],
        })

        renderPass.setPipeline(this.renderPipeline)
        renderPass.setBindGroup(0, this.bindGroup)
        renderPass.draw(3)  // Full-screen triangle
        renderPass.end()

        device.queue.submit([commandEncoder.finish()])
    }

    async _resize(width, height) {
        // Store dimensions for height-based scaling
        this.width = width
        this.height = height

        // Recreate output texture at new size
        this.outputTexture = await Texture.renderTarget(this.engine, 'r8unorm')
        this._needsRebuild = true
    }

    _destroy() {
        this.renderPipeline = null
        this.outputTexture = null
    }

    /**
     * Get the output AO texture
     */
    getOutputTexture() {
        return this.outputTexture
    }

    /**
     * Configure AO parameters
     */
    configure(options) {
        if (options.intensity !== undefined) this.aoIntensity = options.intensity
        if (options.radius !== undefined) this.aoRadius = options.radius
        if (options.fadeDistance !== undefined) this.aoFadeDistance = options.fadeDistance
        if (options.bias !== undefined) this.aoBias = options.bias
    }
}

export { AOPass }
