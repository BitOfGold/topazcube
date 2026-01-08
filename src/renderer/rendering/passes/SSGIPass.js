import { BasePass } from "./BasePass.js"
import { Texture } from "../../Texture.js"

import ssgiWGSL from "../shaders/ssgi.wgsl"

/**
 * SSGIPass - Screen Space Global Illumination (Tile-Based)
 *
 * Samples propagated directional light from tiles in a cross pattern.
 * Uses screen-space normal to weight directional contributions.
 *
 * Input: Propagate buffer from SSGITilePass, GBuffer normals
 * Output: SSGI texture (half-res, HDR)
 */

const TILE_SIZE = 64

class SSGIPass extends BasePass {
    constructor(engine = null) {
        super('SSGI', engine)

        this.ssgiTexture = null
        this.pipeline = null
        this.bindGroupLayout = null
        this.uniformBuffer = null
        this.uniformData = null
        this.width = 0
        this.height = 0

        // Propagate buffer from SSGITilePass
        this.propagateBuffer = null
        this.tileCountX = 0
        this.tileCountY = 0

        // GBuffer reference
        this.gbuffer = null

        // Frame counter for temporal jittering
        this.frameIndex = 0

        // Textures pending destruction
        this._pendingDestroyRing = [[], [], []]
        this._pendingDestroyIndex = 0
    }

    async _init() {
        const { canvas } = this.engine
        await this._createResources(canvas.width, canvas.height)
    }

    /**
     * Set the propagate buffer from SSGITilePass
     */
    setPropagateBuffer(buffer, tileCountX, tileCountY) {
        this.propagateBuffer = buffer
        this.tileCountX = tileCountX
        this.tileCountY = tileCountY
    }

    /**
     * Set GBuffer for normal access
     */
    setGBuffer(gbuffer) {
        this.gbuffer = gbuffer
    }

    async _createResources(width, height) {
        const { device } = this.engine

        // Half resolution for performance
        this.width = Math.floor(width / 2)
        this.height = Math.floor(height / 2)

        // Create SSGI output texture (half-res, HDR)
        this.ssgiTexture = await Texture.renderTarget(this.engine, 'rgba16float', this.width, this.height)
        this.ssgiTexture.label = 'ssgiOutput'

        // Uniform buffer: screenParams(16) + tileParams(16) + ssgiParams(16) = 48 bytes
        const uniformSize = 48
        this.uniformData = new ArrayBuffer(uniformSize)
        this.uniformBuffer = device.createBuffer({
            label: 'ssgiUniformBuffer',
            size: uniformSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // Create shader module
        const shaderModule = device.createShaderModule({
            label: 'ssgiShaderModule',
            code: ssgiWGSL,
        })

        // Create bind group layout
        this.bindGroupLayout = device.createBindGroupLayout({
            label: 'ssgiBindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },  // gbufferNormal
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },          // propagateBuffer
            ],
        })

        // Create render pipeline (async for non-blocking initialization)
        this.pipeline = await device.createRenderPipelineAsync({
            label: 'ssgiPipeline',
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
    }

    async _execute(context) {
        const { device, canvas } = this.engine
        const { camera } = context

        // Process deferred texture destruction
        this._pendingDestroyIndex = (this._pendingDestroyIndex + 1) % 3
        const toDestroy = this._pendingDestroyRing[this._pendingDestroyIndex]
        for (const res of toDestroy) {
            res.destroy()
        }
        this._pendingDestroyRing[this._pendingDestroyIndex] = []

        // Check if SSGI is enabled
        const ssgiSettings = this.settings?.ssgi
        if (!ssgiSettings?.enabled) {
            return
        }

        // Check required resources
        if (!this.gbuffer || !this.propagateBuffer) {
            return
        }

        if (!this.pipeline) {
            return
        }

        // Update uniforms
        this._updateUniforms(ssgiSettings, canvas.width, canvas.height)

        // Create bind group
        const bindGroup = device.createBindGroup({
            label: 'ssgiBindGroup',
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.gbuffer.normal.view },
                { binding: 2, resource: { buffer: this.propagateBuffer } },
            ],
        })

        // Render
        const commandEncoder = device.createCommandEncoder({ label: 'ssgiCommandEncoder' })

        const passEncoder = commandEncoder.beginRenderPass({
            label: 'ssgiRenderPass',
            colorAttachments: [{
                view: this.ssgiTexture.view,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
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

    _updateUniforms(ssgiSettings, fullWidth, fullHeight) {
        const { device } = this.engine

        const view = new DataView(this.uniformData)
        let offset = 0

        // screenParams: fullWidth, fullHeight, halfWidth, halfHeight
        view.setFloat32(offset, fullWidth, true); offset += 4
        view.setFloat32(offset, fullHeight, true); offset += 4
        view.setFloat32(offset, this.width, true); offset += 4
        view.setFloat32(offset, this.height, true); offset += 4

        // tileParams: tileCountX, tileCountY, tileSize, sampleRadius
        const sampleRadius = ssgiSettings.sampleRadius ?? 2.0  // Vogel disk radius in tiles
        view.setFloat32(offset, this.tileCountX, true); offset += 4
        view.setFloat32(offset, this.tileCountY, true); offset += 4
        view.setFloat32(offset, TILE_SIZE, true); offset += 4
        view.setFloat32(offset, sampleRadius, true); offset += 4

        // ssgiParams: intensity, frameIndex, unused, unused
        view.setFloat32(offset, ssgiSettings.intensity ?? 1.0, true); offset += 4
        view.setFloat32(offset, this.frameIndex, true); offset += 4
        view.setFloat32(offset, 0.0, true); offset += 4
        view.setFloat32(offset, 0.0, true); offset += 4

        // Increment frame index for temporal jittering
        this.frameIndex++

        device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)
    }

    /**
     * Get SSGI output texture
     */
    getSSGITexture() {
        return this.ssgiTexture
    }

    async _resize(width, height) {
        this._queueResourcesForDestruction()
        await this._createResources(width, height)
    }

    _queueResourcesForDestruction() {
        const slot = this._pendingDestroyRing[this._pendingDestroyIndex]
        if (this.ssgiTexture?.texture) {
            slot.push(this.ssgiTexture.texture)
            this.ssgiTexture = null
        }
        if (this.uniformBuffer) {
            slot.push(this.uniformBuffer)
            this.uniformBuffer = null
        }
    }

    _destroyResources() {
        if (this.ssgiTexture?.texture) {
            this.ssgiTexture.texture.destroy()
            this.ssgiTexture = null
        }
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy()
            this.uniformBuffer = null
        }
    }

    _destroy() {
        this._destroyResources()
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

export { SSGIPass }
