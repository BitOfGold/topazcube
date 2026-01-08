import { BasePass } from "./BasePass.js"

import ssgiAccumulateWGSL from "../shaders/ssgi_accumulate.wgsl"
import ssgiPropagateWGSL from "../shaders/ssgi_propagate.wgsl"

/**
 * SSGITilePass - Two-pass tile-based light propagation
 *
 * Pass 1 (Accumulate): For each tile, average the light content from prev HDR + boosted emissive
 * Pass 2 (Propagate): For each tile, collect indirect light from all other tiles in 4 directions
 *
 * Input: Previous frame HDR, GBuffer emissive
 * Output: Propagated directional light buffer (4 directions per tile)
 */

const TILE_SIZE = 64

class SSGITilePass extends BasePass {
    constructor(engine = null) {
        super('SSGITile', engine)

        // Compute pipelines
        this.accumulatePipeline = null
        this.propagatePipeline = null

        // Bind group layouts
        this.accumulateBGL = null
        this.propagateBGL = null

        // Buffers
        this.tileAccumBuffer = null    // Accumulated light per tile (vec4f per tile)
        this.tilePropagateBuffer = null // Propagated directional light (4 directions per tile)

        // Tile grid dimensions
        this.tileCountX = 0
        this.tileCountY = 0

        // Input textures
        this.prevHDRTexture = null
        this.emissiveTexture = null

        // Uniform buffer
        this.uniformBuffer = null

        // Sampler for HDR texture
        this.sampler = null
    }

    /**
     * Set the previous frame HDR texture
     */
    setPrevHDRTexture(texture) {
        this.prevHDRTexture = texture
    }

    /**
     * Set the emissive texture from GBuffer
     */
    setEmissiveTexture(texture) {
        this.emissiveTexture = texture
    }

    async _init() {
        const { device, canvas } = this.engine

        this.sampler = device.createSampler({
            label: 'SSGI Tile Sampler',
            minFilter: 'linear',
            magFilter: 'linear',
        })

        await this._createResources(canvas.width, canvas.height)
    }

    async _createResources(width, height) {
        const { device } = this.engine

        // Calculate tile grid dimensions
        this.tileCountX = Math.ceil(width / TILE_SIZE)
        this.tileCountY = Math.ceil(height / TILE_SIZE)
        const totalTiles = this.tileCountX * this.tileCountY

        // Destroy old buffers
        if (this.tileAccumBuffer) this.tileAccumBuffer.destroy()
        if (this.tilePropagateBuffer) this.tilePropagateBuffer.destroy()
        if (this.uniformBuffer) this.uniformBuffer.destroy()

        // Create tile accumulation buffer (1 vec4f per tile - RGB + weight)
        this.tileAccumBuffer = device.createBuffer({
            label: 'SSGI Tile Accum Buffer',
            size: totalTiles * 4 * 4,  // tiles × 4 floats × 4 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })

        // Create propagated light buffer (4 directions per tile × vec4f)
        // Directions: 0=left, 1=right, 2=up, 3=down
        this.tilePropagateBuffer = device.createBuffer({
            label: 'SSGI Tile Propagate Buffer',
            size: totalTiles * 4 * 4 * 4,  // tiles × 4 directions × 4 floats × 4 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })

        // Create uniform buffer
        this.uniformBuffer = device.createBuffer({
            label: 'SSGI Tile Uniforms',
            size: 32,  // 8 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // Create accumulate pipeline
        const accumulateModule = device.createShaderModule({
            label: 'SSGI Accumulate Shader',
            code: ssgiAccumulateWGSL,
        })

        this.accumulateBGL = device.createBindGroupLayout({
            label: 'SSGI Accumulate BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        })

        // Create propagate pipeline shader module
        const propagateModule = device.createShaderModule({
            label: 'SSGI Propagate Shader',
            code: ssgiPropagateWGSL,
        })

        this.propagateBGL = device.createBindGroupLayout({
            label: 'SSGI Propagate BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        })

        // Create both compute pipelines in parallel for faster initialization
        const [accumulatePipeline, propagatePipeline] = await Promise.all([
            device.createComputePipelineAsync({
                label: 'SSGI Accumulate Pipeline',
                layout: device.createPipelineLayout({ bindGroupLayouts: [this.accumulateBGL] }),
                compute: { module: accumulateModule, entryPoint: 'main' },
            }),
            device.createComputePipelineAsync({
                label: 'SSGI Propagate Pipeline',
                layout: device.createPipelineLayout({ bindGroupLayouts: [this.propagateBGL] }),
                compute: { module: propagateModule, entryPoint: 'main' },
            })
        ])

        this.accumulatePipeline = accumulatePipeline
        this.propagatePipeline = propagatePipeline

        this._needsRebuild = false
    }

    async _execute(context) {
        const { device, canvas } = this.engine

        // Check if SSGI is enabled
        const ssgiSettings = this.settings?.ssgi
        if (!ssgiSettings?.enabled) {
            return
        }

        // Rebuild if needed
        if (this._needsRebuild) {
            await this._createResources(canvas.width, canvas.height)
        }

        // Check required textures
        if (!this.prevHDRTexture || !this.emissiveTexture) {
            return
        }

        if (!this.accumulatePipeline || !this.propagatePipeline) {
            return
        }

        const width = canvas.width
        const height = canvas.height
        const emissiveBoost = ssgiSettings.emissiveBoost ?? 2.0
        const maxBrightness = ssgiSettings.maxBrightness ?? 4.0

        // Update uniforms
        device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([
            width,
            height,
            this.tileCountX,
            this.tileCountY,
            TILE_SIZE,
            emissiveBoost,
            maxBrightness,
            0,  // padding
        ]))

        // Clear buffers
        const clearAccum = new Float32Array(this.tileCountX * this.tileCountY * 4)
        device.queue.writeBuffer(this.tileAccumBuffer, 0, clearAccum)
        const clearPropagate = new Float32Array(this.tileCountX * this.tileCountY * 4 * 4)
        device.queue.writeBuffer(this.tilePropagateBuffer, 0, clearPropagate)

        // === PASS 1: Accumulate light per tile ===
        const accumulateBindGroup = device.createBindGroup({
            label: 'SSGI Accumulate Bind Group',
            layout: this.accumulateBGL,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.prevHDRTexture.view },
                { binding: 2, resource: this.emissiveTexture.view },
                { binding: 3, resource: this.sampler },
                { binding: 4, resource: { buffer: this.tileAccumBuffer } },
            ],
        })

        const commandEncoder = device.createCommandEncoder({ label: 'SSGI Tile Pass' })

        const accumulatePass = commandEncoder.beginComputePass({ label: 'SSGI Accumulate' })
        accumulatePass.setPipeline(this.accumulatePipeline)
        accumulatePass.setBindGroup(0, accumulateBindGroup)
        accumulatePass.dispatchWorkgroups(this.tileCountX, this.tileCountY, 1)
        accumulatePass.end()

        // === PASS 2: Propagate light between tiles ===
        const propagateBindGroup = device.createBindGroup({
            label: 'SSGI Propagate Bind Group',
            layout: this.propagateBGL,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.tileAccumBuffer } },
                { binding: 2, resource: { buffer: this.tilePropagateBuffer } },
            ],
        })

        const propagatePass = commandEncoder.beginComputePass({ label: 'SSGI Propagate' })
        propagatePass.setPipeline(this.propagatePipeline)
        propagatePass.setBindGroup(0, propagateBindGroup)
        propagatePass.dispatchWorkgroups(this.tileCountX, this.tileCountY, 1)
        propagatePass.end()

        device.queue.submit([commandEncoder.finish()])
    }

    /**
     * Get the propagated light buffer for SSGIPass to sample from
     */
    getPropagateBuffer() {
        return this.tilePropagateBuffer
    }

    /**
     * Get the accumulated light buffer (for debugging)
     */
    getAccumBuffer() {
        return this.tileAccumBuffer
    }

    /**
     * Get tile grid dimensions
     */
    getTileInfo() {
        return {
            tileCountX: this.tileCountX,
            tileCountY: this.tileCountY,
            tileSize: TILE_SIZE,
        }
    }

    async _resize(width, height) {
        await this._createResources(width, height)
    }

    _destroy() {
        if (this.tileAccumBuffer) {
            this.tileAccumBuffer.destroy()
            this.tileAccumBuffer = null
        }
        if (this.tilePropagateBuffer) {
            this.tilePropagateBuffer.destroy()
            this.tilePropagateBuffer = null
        }
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy()
            this.uniformBuffer = null
        }
        this.accumulatePipeline = null
        this.propagatePipeline = null
    }
}

export { SSGITilePass, TILE_SIZE }
