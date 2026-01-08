import { mat4 } from "../math.js"

/**
 * InstanceManager - Manages instance batches for efficient rendering
 *
 * Groups entities by ModelID for instanced draw calls.
 * Handles buffer pools for instance data (model matrix + bsphere + sprite data).
 *
 * Instance data format (112 bytes per instance):
 * - mat4x4f: model matrix (64 bytes)
 * - vec4f: position.xyz + boundingRadius (16 bytes)
 * - vec4f: uvOffset.xy + uvScale.xy (16 bytes) - for sprite sheets
 * - vec4f: color.rgba (16 bytes) - for sprite tinting
 */
class InstanceManager {
    constructor(engine = null) {
        // Reference to engine for settings access
        this.engine = engine

        // Buffer pools: size -> array of available buffers
        this._bufferPool = new Map()

        // Active batches: modelId -> batch info
        this._batches = new Map()

        // Instance data stride (28 floats = 112 bytes)
        this.INSTANCE_STRIDE = 28

        // Default pool size
        this.DEFAULT_POOL_SIZE = 1000
    }

    /**
     * Get or create a buffer from the pool
     * @param {number} capacity - Number of instances the buffer should hold
     * @returns {Object} Buffer info
     */
    _getBuffer(capacity) {
        const { device } = this.engine

        // Round up to nearest pool size
        const poolSize = Math.max(this.DEFAULT_POOL_SIZE, Math.pow(2, Math.ceil(Math.log2(capacity))))

        // Check pool
        if (!this._bufferPool.has(poolSize)) {
            this._bufferPool.set(poolSize, [])
        }

        const pool = this._bufferPool.get(poolSize)
        if (pool.length > 0) {
            return pool.pop()
        }

        // Create new buffer
        const byteSize = poolSize * this.INSTANCE_STRIDE * 4 // 4 bytes per float
        const gpuBuffer = device.createBuffer({
            size: byteSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            label: `Instance buffer (${poolSize})`
        })

        return {
            gpuBuffer,
            cpuData: new Float32Array(poolSize * this.INSTANCE_STRIDE),
            capacity: poolSize,
            count: 0
        }
    }

    /**
     * Return a buffer to the pool
     * @param {Object} bufferInfo - Buffer info to return
     */
    _releaseBuffer(bufferInfo) {
        const pool = this._bufferPool.get(bufferInfo.capacity)
        if (pool) {
            bufferInfo.count = 0
            pool.push(bufferInfo)
        }
    }

    /**
     * Build instance batches from visible entities
     *
     * @param {Map<string, Array>} groups - Groups from CullingSystem.groupByModel()
     * @param {AssetManager} assetManager - Asset manager for mesh lookup
     * @returns {Map<string, Object>} Batches ready for rendering
     */
    buildBatches(groups, assetManager) {
        const { device } = this.engine

        // Release old batches back to pool
        for (const [modelId, batch] of this._batches) {
            if (batch.buffer) {
                this._releaseBuffer(batch.buffer)
            }
        }
        this._batches.clear()

        // Build new batches
        for (const [modelId, entities] of groups) {
            // Get asset
            const asset = assetManager.get(modelId)
            if (!asset?.ready) continue

            // Get or create buffer
            const buffer = this._getBuffer(entities.length)

            // Fill instance data
            let offset = 0
            for (const item of entities) {
                const entity = item.entity

                // Copy model matrix (16 floats)
                buffer.cpuData.set(entity._matrix, offset)

                // Copy position + radius (4 floats)
                buffer.cpuData[offset + 16] = entity._bsphere.center[0]
                buffer.cpuData[offset + 17] = entity._bsphere.center[1]
                buffer.cpuData[offset + 18] = entity._bsphere.center[2]
                buffer.cpuData[offset + 19] = entity._bsphere.radius

                // Copy UV transform (4 floats): offset.xy, scale.xy
                // Default: no transform (offset 0,0, scale 1,1)
                const uvTransform = entity._uvTransform || [0, 0, 1, 1]
                buffer.cpuData[offset + 20] = uvTransform[0]
                buffer.cpuData[offset + 21] = uvTransform[1]
                buffer.cpuData[offset + 22] = uvTransform[2]
                buffer.cpuData[offset + 23] = uvTransform[3]

                // Copy color tint (4 floats): r, g, b, a
                // Default: white (no tint)
                const color = entity.color || [1, 1, 1, 1]
                buffer.cpuData[offset + 24] = color[0]
                buffer.cpuData[offset + 25] = color[1]
                buffer.cpuData[offset + 26] = color[2]
                buffer.cpuData[offset + 27] = color[3]

                offset += this.INSTANCE_STRIDE
            }

            buffer.count = entities.length

            // Upload to GPU
            device.queue.writeBuffer(
                buffer.gpuBuffer,
                0,
                buffer.cpuData,
                0,
                entities.length * this.INSTANCE_STRIDE
            )

            // Store batch
            this._batches.set(modelId, {
                modelId,
                mesh: asset.mesh,
                geometry: asset.geometry,
                material: asset.material,
                skin: asset.skin,
                hasSkin: asset.hasSkin,
                buffer,
                instanceCount: entities.length,
                entities
            })
        }

        return this._batches
    }

    /**
     * Build instance batches for skinned meshes grouped by animation
     *
     * @param {Map<string, Array>} groups - Groups from CullingSystem.groupByModelAndAnimation()
     * @param {AssetManager} assetManager - Asset manager
     * @returns {Map<string, Object>} Batches with animation info
     */
    buildSkinnedBatches(groups, assetManager) {
        const { device } = this.engine
        const batches = new Map()

        for (const [key, entities] of groups) {
            // Parse key: "modelId|animation|phase"
            const parts = key.split('|')
            const modelId = parts[0] + (parts.length > 1 ? '|' + parts[1].split('|')[0] : '')

            // For skinned meshes, we need to check the base modelId
            const baseModelId = key.includes('|') ? key.split('|').slice(0, 2).join('|') : key

            // Try to find the asset
            let asset = null
            for (const [assetKey, assetValue] of Object.entries(assetManager.assets)) {
                if (assetKey.includes('|') && assetKey.startsWith(key.split('|')[0])) {
                    if (assetValue.ready) {
                        asset = assetValue
                        break
                    }
                }
            }

            if (!asset?.ready) continue

            // Get or create buffer
            const buffer = this._getBuffer(entities.length)

            // Fill instance data
            let offset = 0
            for (const item of entities) {
                const entity = item.entity

                buffer.cpuData.set(entity._matrix, offset)
                buffer.cpuData[offset + 16] = entity._bsphere.center[0]
                buffer.cpuData[offset + 17] = entity._bsphere.center[1]
                buffer.cpuData[offset + 18] = entity._bsphere.center[2]
                buffer.cpuData[offset + 19] = entity._bsphere.radius

                // Copy UV transform (4 floats): offset.xy, scale.xy
                const uvTransform = entity._uvTransform || [0, 0, 1, 1]
                buffer.cpuData[offset + 20] = uvTransform[0]
                buffer.cpuData[offset + 21] = uvTransform[1]
                buffer.cpuData[offset + 22] = uvTransform[2]
                buffer.cpuData[offset + 23] = uvTransform[3]

                // Copy color tint (4 floats): r, g, b, a
                const color = entity.color || [1, 1, 1, 1]
                buffer.cpuData[offset + 24] = color[0]
                buffer.cpuData[offset + 25] = color[1]
                buffer.cpuData[offset + 26] = color[2]
                buffer.cpuData[offset + 27] = color[3]

                offset += this.INSTANCE_STRIDE
            }

            buffer.count = entities.length

            // Upload to GPU
            device.queue.writeBuffer(
                buffer.gpuBuffer,
                0,
                buffer.cpuData,
                0,
                entities.length * this.INSTANCE_STRIDE
            )

            // Extract animation info from key
            const animation = parts.length > 2 ? parts[1] : null
            const phase = parts.length > 3 ? parseFloat(parts[2]) : 0

            batches.set(key, {
                modelId: baseModelId,
                mesh: asset.mesh,
                geometry: asset.geometry,
                material: asset.material,
                skin: asset.skin,
                hasSkin: asset.hasSkin,
                animation,
                phase,
                buffer,
                instanceCount: entities.length,
                entities
            })
        }

        return batches
    }

    /**
     * Get current batches
     */
    getBatches() {
        return this._batches
    }

    /**
     * Get instance buffer layout for pipeline creation
     */
    static getBufferLayout() {
        return {
            arrayStride: 112, // 28 floats * 4 bytes
            stepMode: 'instance',
            attributes: [
                { format: "float32x4", offset: 0, shaderLocation: 6 },   // matrix column 0
                { format: "float32x4", offset: 16, shaderLocation: 7 },  // matrix column 1
                { format: "float32x4", offset: 32, shaderLocation: 8 },  // matrix column 2
                { format: "float32x4", offset: 48, shaderLocation: 9 },  // matrix column 3
                { format: "float32x4", offset: 64, shaderLocation: 10 }, // position + radius
                { format: "float32x4", offset: 80, shaderLocation: 11 }, // uvTransform (offset.xy, scale.xy)
                { format: "float32x4", offset: 96, shaderLocation: 12 }, // color (r, g, b, a)
            ]
        }
    }

    /**
     * Clear all batches and return buffers to pool
     */
    clear() {
        for (const [modelId, batch] of this._batches) {
            if (batch.buffer) {
                this._releaseBuffer(batch.buffer)
            }
        }
        this._batches.clear()
    }

    /**
     * Destroy all buffers
     */
    destroy() {
        this.clear()
        for (const [size, pool] of this._bufferPool) {
            for (const buffer of pool) {
                buffer.gpuBuffer.destroy()
            }
        }
        this._bufferPool.clear()
    }

    /**
     * Get statistics about buffer usage
     */
    getStats() {
        let pooledBuffers = 0
        let pooledCapacity = 0
        for (const [size, pool] of this._bufferPool) {
            pooledBuffers += pool.length
            pooledCapacity += pool.length * size
        }

        let activeBuffers = this._batches.size
        let activeInstances = 0
        for (const [modelId, batch] of this._batches) {
            activeInstances += batch.instanceCount
        }

        return {
            pooledBuffers,
            pooledCapacity,
            activeBuffers,
            activeInstances
        }
    }
}

export { InstanceManager }
