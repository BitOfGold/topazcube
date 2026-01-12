import { BasePass } from "./BasePass.js"
import { vec3 } from "../../math.js"

import hizReduceWGSL from "../shaders/hiz_reduce.wgsl"

/**
 * HiZPass - Hierarchical Z-Buffer for occlusion culling
 *
 * Reduces the depth buffer to maximum depth per 64x64 tile.
 * The CPU can then use this data to cull objects that are behind
 * the geometry from the previous frame.
 *
 * Features:
 * - 64x64 pixel tiles for coarse occlusion testing
 * - Camera movement detection to invalidate stale data
 * - Async GPU->CPU readback for non-blocking culling
 */

// Fixed tile size - must match compute shader (hiz_reduce.wgsl)
const TILE_SIZE = 64

// Maximum tiles before disabling occlusion culling (too slow to readback)
const MAX_TILES_FOR_OCCLUSION = 2500

// Thresholds for invalidating HiZ data
const POSITION_THRESHOLD = 0.5      // Units of movement before invalidation
const ROTATION_THRESHOLD = 0.02     // Radians of rotation before invalidation (~1 degree)

// Maximum frames of readback latency before invalidating occlusion data
const MAX_READBACK_LATENCY = 3

class HiZPass extends BasePass {
    constructor(engine = null) {
        super('HiZ', engine)

        // Compute pipeline
        this.pipeline = null
        this.bindGroupLayout = null

        // Buffers
        this.hizBuffer = null           // GPU storage buffer (max Z per tile)
        this.stagingBuffers = [null, null]  // Double-buffered staging for CPU readback
        this.stagingBufferInUse = [false, false]  // Track which buffers are currently mapped
        this.currentStagingIndex = 0    // Which staging buffer to use next
        this.uniformBuffer = null

        // Tile grid dimensions
        this.tileCountX = 0
        this.tileCountY = 0
        this.totalTiles = 0
        this._tooManyTiles = false  // True if resolution is too high for efficient occlusion

        // Frame tracking for readback latency
        this._frameCounter = 0          // Current frame number
        this._lastReadbackFrame = 0     // Frame when last readback completed

        // CPU-side HiZ data (double-buffered to prevent race conditions)
        this.hizDataBuffers = [null, null]  // Two Float32Array buffers for double-buffering
        this.readHizIndex = 0               // Which buffer is currently used for reading
        this.writeHizIndex = 1              // Which buffer is currently used for writing
        this.hizDataReady = false           // Whether CPU data is valid for this frame
        this.pendingReadback = null         // Promise for pending readback

        // Debug stats for occlusion testing
        this.debugStats = {
            tested: 0,
            occluded: 0,
            skippedTileSpan: 0,
            visibleSkyGap: 0,
            visibleInFront: 0,
        }

        // Camera tracking for invalidation
        this.lastCameraPosition = vec3.create()
        this.lastCameraDirection = vec3.create()
        this.hasValidHistory = false    // Whether we have valid previous frame data

        // Depth texture reference
        this.depthTexture = null

        // Screen dimensions
        this.screenWidth = 0
        this.screenHeight = 0

        // Flag to prevent operations during destruction
        this._destroyed = false

        // Warmup frames - occlusion is disabled until scene has rendered for a few frames
        // This prevents false occlusion on engine creation when depth buffer is not yet populated
        this._warmupFramesRemaining = 5
    }

    /**
     * Set the depth texture to read from (from GBuffer)
     * @param {Object} depth - Depth texture object with .texture and .view
     */
    setDepthTexture(depth) {
        this.depthTexture = depth
    }

    /**
     * Invalidate occlusion culling data and reset warmup period.
     * Call this after engine creation, scene loading, or major camera changes
     * to prevent incorrect occlusion culling with stale data.
     */
    invalidate() {
        this.hasValidHistory = false
        this.hizDataReady = false
        this._warmupFramesRemaining = 5  // Wait 5 frames before enabling occlusion
        // Reset camera tracking to avoid false invalidations
        vec3.set(this.lastCameraPosition, 0, 0, 0)
        vec3.set(this.lastCameraDirection, 0, 0, 0)
    }

    async _init() {
        const { device, canvas } = this.engine
        await this._createResources(canvas.width, canvas.height)
    }

    async _createResources(width, height) {
        // Skip if dimensions haven't changed (avoid double init)
        if (this.screenWidth === width && this.screenHeight === height && this.hizBuffer) {
            return
        }

        const { device } = this.engine

        // Mark as destroyed to cancel any pending readback
        this._destroyed = true

        // Wait for any pending readback to complete
        if (this.pendingReadback) {
            try {
                await this.pendingReadback
            } catch (e) {
                // Ignore errors from cancelled readback
            }
            this.pendingReadback = null
        }

        this.screenWidth = width
        this.screenHeight = height

        // Calculate tile grid dimensions (fixed 64px tiles to match compute shader)
        this.tileCountX = Math.ceil(width / TILE_SIZE)
        this.tileCountY = Math.ceil(height / TILE_SIZE)
        this.totalTiles = this.tileCountX * this.tileCountY

        // Check if we have too many tiles for efficient occlusion culling
        this._tooManyTiles = this.totalTiles > MAX_TILES_FOR_OCCLUSION

        console.log(`HiZ: ${width}x${height} -> ${TILE_SIZE}px tiles, ${this.tileCountX}x${this.tileCountY} = ${this.totalTiles} tiles${this._tooManyTiles ? ' (occlusion disabled - too many tiles)' : ''}`)

        // Destroy old buffers
        if (this.hizBuffer) this.hizBuffer.destroy()
        for (let i = 0; i < 2; i++) {
            if (this.stagingBuffers[i]) {
                this.stagingBuffers[i].destroy()
            }
        }
        if (this.uniformBuffer) this.uniformBuffer.destroy()

        // Create HiZ buffer (2 floats per tile: minZ, maxZ)
        const bufferSize = this.totalTiles * 2 * 4  // 2 floats * 4 bytes per float
        this.hizBuffer = device.createBuffer({
            label: 'HiZ Buffer',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })

        // Create double-buffered staging buffers for CPU readback
        for (let i = 0; i < 2; i++) {
            this.stagingBuffers[i] = device.createBuffer({
                label: `HiZ Staging Buffer ${i}`,
                size: bufferSize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            })
            this.stagingBufferInUse[i] = false
        }
        this.currentStagingIndex = 0

        // Create uniform buffer
        this.uniformBuffer = device.createBuffer({
            label: 'HiZ Uniforms',
            size: 32,  // 8 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // Create CPU-side data arrays (double-buffered, 2 floats per tile: minZ, maxZ)
        this.hizDataBuffers[0] = new Float32Array(this.totalTiles * 2)
        this.hizDataBuffers[1] = new Float32Array(this.totalTiles * 2)
        // Initialize: minZ=1.0, maxZ=1.0 (sky) - everything passes occlusion test
        for (let i = 0; i < this.totalTiles; i++) {
            this.hizDataBuffers[0][i * 2] = 1.0      // minZ
            this.hizDataBuffers[0][i * 2 + 1] = 1.0  // maxZ
            this.hizDataBuffers[1][i * 2] = 1.0
            this.hizDataBuffers[1][i * 2 + 1] = 1.0
        }
        this.readHizIndex = 0
        this.writeHizIndex = 1

        // Create compute pipeline
        const shaderModule = device.createShaderModule({
            label: 'HiZ Reduce Shader',
            code: hizReduceWGSL,
        })

        this.bindGroupLayout = device.createBindGroupLayout({
            label: 'HiZ Reduce BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        })

        this.pipeline = await device.createComputePipelineAsync({
            label: 'HiZ Reduce Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'main' },
        })

        // Reset state
        this.hasValidHistory = false
        this._destroyed = false
        this.hizDataReady = false
        this.pendingReadback = null
        this._warmupFramesRemaining = 5  // Wait a few frames after resize before enabling occlusion
    }

    /**
     * Check if camera has moved significantly, requiring HiZ invalidation
     * @param {Camera} camera - Current camera
     * @returns {boolean} True if camera moved too much
     */
    _checkCameraMovement(camera) {
        const position = camera.position
        const direction = camera.direction

        // Calculate position delta
        const dx = position[0] - this.lastCameraPosition[0]
        const dy = position[1] - this.lastCameraPosition[1]
        const dz = position[2] - this.lastCameraPosition[2]
        const positionDelta = Math.sqrt(dx * dx + dy * dy + dz * dz)

        // Calculate rotation delta (dot product of directions)
        const dot = direction[0] * this.lastCameraDirection[0] +
                    direction[1] * this.lastCameraDirection[1] +
                    direction[2] * this.lastCameraDirection[2]
        // Clamp to avoid NaN from acos
        const clampedDot = Math.max(-1, Math.min(1, dot))
        const rotationDelta = Math.acos(clampedDot)

        // Update last camera state
        vec3.copy(this.lastCameraPosition, position)
        vec3.copy(this.lastCameraDirection, direction)

        // Check thresholds
        const positionThreshold = this.settings?.occlusionCulling?.positionThreshold ?? POSITION_THRESHOLD
        const rotationThreshold = this.settings?.occlusionCulling?.rotationThreshold ?? ROTATION_THRESHOLD

        return positionDelta > positionThreshold || rotationDelta > rotationThreshold
    }

    /**
     * Prepare HiZ data for occlusion tests - call BEFORE any culling
     */
    prepareForOcclusionTests(camera) {
        if (!camera) return

        // Reset debug stats at start of frame
        this.resetDebugStats()

        // No camera movement invalidation - the 100% depth gap requirement
        // handles stale data gracefully, and next frame will be correct
    }

    async _execute(context) {
        const { device } = this.engine
        const { camera } = context

        // Increment frame counter
        this._frameCounter++

        // Decrement warmup counter - occlusion is disabled until this reaches 0
        if (this._warmupFramesRemaining > 0) {
            this._warmupFramesRemaining--
        }

        // Check if occlusion culling is enabled
        if (!this.settings?.occlusionCulling?.enabled) {
            return
        }

        if (!this.depthTexture || !this.pipeline) {
            return
        }

        // Note: Camera movement check is now done in prepareForOcclusionTests()
        // which is called before light culling

        // Determine if we should clear or accumulate
        const shouldClear = !this.hasValidHistory

        // Update uniforms
        const near = camera.near || 0.05
        const far = camera.far || 1000
        device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([
            this.screenWidth,
            this.screenHeight,
            this.tileCountX,
            this.tileCountY,
            TILE_SIZE,
            near,
            far,
            shouldClear ? 1.0 : 0.0,  // clearValue
        ]))

        // Create bind group
        const bindGroup = device.createBindGroup({
            label: 'HiZ Reduce Bind Group',
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.depthTexture.view },
                { binding: 2, resource: { buffer: this.hizBuffer } },
            ],
        })

        // Execute compute shader
        const commandEncoder = device.createCommandEncoder({ label: 'HiZ Reduce' })

        const computePass = commandEncoder.beginComputePass({ label: 'HiZ Reduce Pass' })
        computePass.setPipeline(this.pipeline)
        computePass.setBindGroup(0, bindGroup)
        computePass.dispatchWorkgroups(this.tileCountX, this.tileCountY, 1)
        computePass.end()

        // Use double-buffered staging: find a buffer that's not currently in use
        let stagingIndex = this.currentStagingIndex
        let stagingBuffer = this.stagingBuffers[stagingIndex]

        // If current buffer is in use, try the other one
        if (this.stagingBufferInUse[stagingIndex]) {
            stagingIndex = (stagingIndex + 1) % 2
            stagingBuffer = this.stagingBuffers[stagingIndex]
        }

        // If both buffers are in use, skip this frame's copy (use stale data)
        if (this.stagingBufferInUse[stagingIndex]) {
            device.queue.submit([commandEncoder.finish()])
            this.hasValidHistory = true
            return
        }

        // Update for next frame
        this.currentStagingIndex = (stagingIndex + 1) % 2

        // Copy to staging buffer for CPU readback (2 floats per tile)
        commandEncoder.copyBufferToBuffer(
            this.hizBuffer, 0,
            stagingBuffer, 0,
            this.totalTiles * 2 * 4
        )

        device.queue.submit([commandEncoder.finish()])

        // Mark that we now have valid history
        this.hasValidHistory = true

        // Start async readback (non-blocking)
        this._startReadback(stagingBuffer, stagingIndex)
    }

    /**
     * Start async GPU->CPU readback
     * @private
     * @param {GPUBuffer} stagingBuffer - The staging buffer to read from
     * @param {number} stagingIndex - Which buffer index this is
     */
    async _startReadback(stagingBuffer, stagingIndex) {
        // Don't start new readback if destroyed
        if (this._destroyed) return

        // Mark buffer as in use
        this.stagingBufferInUse[stagingIndex] = true

        // Capture which CPU buffer to write to and current frame
        const writeIndex = this.writeHizIndex
        const requestedFrame = this._frameCounter

        // Create the readback promise
        const readbackPromise = (async () => {
            try {
                await stagingBuffer.mapAsync(GPUMapMode.READ)

                // Check if we were destroyed while waiting
                if (this._destroyed) {
                    try { stagingBuffer.unmap() } catch (e) {}
                    return
                }

                // Write to the write buffer (not the read buffer)
                const data = new Float32Array(stagingBuffer.getMappedRange())
                this.hizDataBuffers[writeIndex].set(data)
                stagingBuffer.unmap()

                // Now atomically swap: make the write buffer the new read buffer
                // This ensures readers never see partial data
                this.readHizIndex = writeIndex
                this.writeHizIndex = writeIndex === 0 ? 1 : 0

                // Track when this readback completed
                this._lastReadbackFrame = this._frameCounter

                this.hizDataReady = true
            } catch (e) {
                // Buffer might be destroyed during resize - this is expected
                if (!this._destroyed) {
                    console.warn('HiZ readback failed:', e)
                }
            } finally {
                // Mark buffer as no longer in use
                this.stagingBufferInUse[stagingIndex] = false
            }
        })()

        // Store the promise so we can wait for it during resize
        this.pendingReadback = readbackPromise

        // Don't await here - let it run in background
        readbackPromise.then(() => {
            // Only clear if this is still the pending readback
            if (this.pendingReadback === readbackPromise) {
                this.pendingReadback = null
            }
        })
    }

    /**
     * Get min and max depth for a tile
     * minDepth = closest geometry (occluder surface)
     * maxDepth = farthest geometry (if < 1.0, tile is fully covered)
     * @param {number} tileX - Tile X coordinate
     * @param {number} tileY - Tile Y coordinate
     * @returns {{ minDepth: number, maxDepth: number }} Depth values (0-1, 0=near, 1=far/sky)
     */
    getTileMinMaxDepth(tileX, tileY) {
        const hizData = this.hizDataBuffers[this.readHizIndex]
        if (!this.hizDataReady || !hizData) {
            return { minDepth: 1.0, maxDepth: 1.0 }  // No data - assume sky (no occlusion)
        }

        if (tileX < 0 || tileX >= this.tileCountX ||
            tileY < 0 || tileY >= this.tileCountY) {
            return { minDepth: 1.0, maxDepth: 1.0 }  // Out of bounds - assume sky
        }

        const index = (tileY * this.tileCountX + tileX) * 2
        return {
            minDepth: hizData[index],
            maxDepth: hizData[index + 1]
        }
    }

    /**
     * Get the maximum depth for a tile (backward compatibility)
     * @param {number} tileX - Tile X coordinate
     * @param {number} tileY - Tile Y coordinate
     * @returns {number} Maximum depth (0-1, 0=near, 1=far/sky)
     */
    getTileMaxDepth(tileX, tileY) {
        return this.getTileMinMaxDepth(tileX, tileY).maxDepth
    }

    /**
     * Test if a bounding sphere is occluded
     *
     * Uses MIN/MAX depth per tile to calculate adaptive occlusion threshold.
     * Tile thickness = maxDepth - minDepth (thin for walls, thick for angled ground)
     *
     * Occlusion threshold = max(1m, tileThickness, 2*sphereRadius) in linear depth
     * Object is occluded if sphereFrontDepth > tileMinDepth + threshold for ALL tiles
     *
     * @param {Object} bsphere - Bounding sphere with center[3] and radius
     * @param {mat4} viewProj - View-projection matrix
     * @param {number} near - Near plane distance
     * @param {number} far - Far plane distance
     * @param {Array} cameraPos - Camera position [x, y, z]
     * @returns {boolean} True if definitely occluded, false if potentially visible
     */
    testSphereOcclusion(bsphere, viewProj, near, far, cameraPos) {
        this.debugStats.tested++

        // Warmup period - disable occlusion for first few frames after creation/reset
        // This ensures the depth buffer has valid geometry before we use it for culling
        if (this._warmupFramesRemaining > 0) {
            return false  // Still warming up - assume visible
        }

        // Safety checks
        if (!this.hizDataReady || !this.hasValidHistory) {
            return false  // No valid data - assume visible
        }

        // Skip occlusion at very high resolutions (too many tiles = slow readback)
        if (this._tooManyTiles) {
            return false  // Disabled at this resolution
        }

        // Check for stale data - if readback is lagging too far behind, skip occlusion
        // This prevents objects from being incorrectly culled when GPU is slow
        const readbackLatency = this._frameCounter - this._lastReadbackFrame
        if (readbackLatency > MAX_READBACK_LATENCY) {
            return false  // Data too stale - assume visible
        }

        if (!cameraPos || !viewProj || !bsphere?.center) {
            return false  // Missing required data - assume visible
        }

        if (this.screenWidth <= 0 || this.screenHeight <= 0 || this.tileCountX <= 0) {
            return false  // Invalid screen size - assume visible
        }

        // Use provided near/far or defaults
        near = near || 0.05
        far = far || 1000

        const cx = bsphere.center[0]
        const cy = bsphere.center[1]
        const cz = bsphere.center[2]
        const radius = bsphere.radius

        // Calculate distance to sphere center
        const dx = cx - cameraPos[0]
        const dy = cy - cameraPos[1]
        const dz = cz - cameraPos[2]
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

        // Skip if sphere intersects near plane
        if (distance - radius < near) {
            return false
        }

        // Project sphere CENTER only (more stable than projecting 8 corners)
        const clipX = viewProj[0] * cx + viewProj[4] * cy + viewProj[8] * cz + viewProj[12]
        const clipY = viewProj[1] * cx + viewProj[5] * cy + viewProj[9] * cz + viewProj[13]
        const clipW = viewProj[3] * cx + viewProj[7] * cy + viewProj[11] * cz + viewProj[15]

        // Behind camera
        if (clipW <= near) {
            return false
        }

        // Perspective divide to NDC
        const ndcX = clipX / clipW
        const ndcY = clipY / clipW

        // Skip if center is way off screen (frustum culling handles this)
        if (ndcX < -1.5 || ndcX > 1.5 || ndcY < -1.5 || ndcY > 1.5) {
            return false
        }

        // Convert center to screen coordinates
        const screenCenterX = (ndcX * 0.5 + 0.5) * this.screenWidth
        const screenCenterY = (1.0 - (ndcY * 0.5 + 0.5)) * this.screenHeight  // Flip Y

        // Calculate screen-space radius using clip W for proper perspective
        const ndcRadius = radius / clipW
        const screenRadius = ndcRadius * (this.screenHeight * 0.5)

        // Calculate screen bounds from center and radius
        const minScreenX = screenCenterX - screenRadius
        const maxScreenX = screenCenterX + screenRadius
        const minScreenY = screenCenterY - screenRadius
        const maxScreenY = screenCenterY + screenRadius

        // Calculate tile range from screen bounds
        const rawMinTileX = Math.floor(minScreenX / TILE_SIZE)
        const rawMaxTileX = Math.floor(maxScreenX / TILE_SIZE)
        const rawMinTileY = Math.floor(minScreenY / TILE_SIZE)
        const rawMaxTileY = Math.floor(maxScreenY / TILE_SIZE)

        // If bounding box extends outside screen, object is partially visible
        if (rawMinTileX < 0 || rawMaxTileX >= this.tileCountX ||
            rawMinTileY < 0 || rawMaxTileY >= this.tileCountY) {
            return false  // Partially off-screen - assume visible
        }

        const minTileX = rawMinTileX
        const maxTileX = rawMaxTileX
        const minTileY = rawMinTileY
        const maxTileY = rawMaxTileY

        // If too many tiles, skip (let frustum culling handle large objects)
        const tileSpanX = maxTileX - minTileX + 1
        const tileSpanY = maxTileY - minTileY + 1
        if (tileSpanX * tileSpanY > 25) {
            this.debugStats.skippedTileSpan++
            return false  // Too many tiles to check
        }

        // Calculate LINEAR depth for front of sphere
        const depthRange = far - near
        const frontDistance = Math.max(near, distance - radius)
        const sphereFrontDepth = (frontDistance - near) / depthRange

        // Convert 1 meter to linear depth units (minimum safety margin)
        const minMarginLinear = 1.0 / depthRange

        // Get configurable threshold multiplier (default 1.0 = 100% of maxZ)
        const cullingThreshold = this.settings?.occlusionCulling?.threshold ?? 1.0

        // Check all tiles - if ANY tile shows object might be visible, exit early
        for (let ty = minTileY; ty <= maxTileY; ty++) {
            for (let tx = minTileX; tx <= maxTileX; tx++) {
                const { minDepth, maxDepth } = this.getTileMinMaxDepth(tx, ty)

                // Sky gap - no occlusion possible
                if (maxDepth >= 0.999) {
                    this.debugStats.visibleSkyGap++
                    return false
                }

                // Tile thickness in linear depth
                const tileThickness = maxDepth - minDepth

                // Adaptive occlusion threshold = max(1m, tileThickness, maxZ * cullingThreshold)
                // - 1m: minimum safety margin
                // - tileThickness: thick tiles (angled ground) need more margin
                // - maxZ * threshold: configurable depth-based margin to prevent self-occlusion
                const depthBasedMargin = maxDepth * cullingThreshold
                const threshold = Math.max(minMarginLinear, tileThickness, depthBasedMargin)

                // Object is visible if its front is within threshold of tile's farthest surface
                // Only occlude if sphere front is beyond maxDepth + threshold
                if (sphereFrontDepth <= maxDepth + threshold) {
                    this.debugStats.visibleInFront++
                    return false  // Visible - exit early
                }
            }
        }

        // Sphere is behind all tiles by more than the threshold → occluded
        this.debugStats.occluded++
        return true
    }

    /**
     * Reset debug stats (call at start of each frame)
     */
    resetDebugStats() {
        this.debugStats.tested = 0
        this.debugStats.occluded = 0
        this.debugStats.skippedTileSpan = 0
        this.debugStats.visibleSkyGap = 0
        this.debugStats.visibleInFront = 0
    }

    /**
     * Get debug stats for occlusion testing
     */
    getDebugStats() {
        return this.debugStats
    }

    /**
     * Get tile information for debugging
     */
    getTileInfo() {
        const hizData = this.hizDataBuffers[this.readHizIndex]
        // Calculate stats about tile coverage
        let globalMinDepth = 1.0, globalMaxDepth = 0.0, coveredTiles = 0
        let avgThickness = 0
        if (hizData && this.hizDataReady) {
            for (let i = 0; i < this.totalTiles; i++) {
                const minD = hizData[i * 2]
                const maxD = hizData[i * 2 + 1]
                globalMinDepth = Math.min(globalMinDepth, minD)
                globalMaxDepth = Math.max(globalMaxDepth, maxD)
                if (maxD < 0.999) {
                    coveredTiles++  // Tile has geometry (not just sky)
                    avgThickness += maxD - minD
                }
            }
            if (coveredTiles > 0) {
                avgThickness /= coveredTiles
            }
        }

        return {
            tileCountX: this.tileCountX,
            tileCountY: this.tileCountY,
            tileSize: TILE_SIZE,
            totalTiles: this.totalTiles,
            hasValidData: this.hizDataReady && this.hasValidHistory,
            hizDataReady: this.hizDataReady,
            hasValidHistory: this.hasValidHistory,
            readbackLatency: this._frameCounter - this._lastReadbackFrame,
            coveredTiles,
            globalMinDepth: globalMinDepth.toFixed(4),
            globalMaxDepth: globalMaxDepth.toFixed(4),
            avgTileThickness: avgThickness.toFixed(4),
        }
    }

    /**
     * Get HiZ data for debugging visualization
     */
    getHiZData() {
        return this.hizDataBuffers[this.readHizIndex]
    }

    async _resize(width, height) {
        await this._createResources(width, height)
    }

    _destroy() {
        this._destroyed = true

        if (this.hizBuffer) {
            this.hizBuffer.destroy()
            this.hizBuffer = null
        }
        for (let i = 0; i < 2; i++) {
            if (this.stagingBuffers[i]) {
                this.stagingBuffers[i].destroy()
                this.stagingBuffers[i] = null
            }
        }
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy()
            this.uniformBuffer = null
        }
        this.pipeline = null
        this.hizDataBuffers = [null, null]
        this.hizDataReady = false
        this.hasValidHistory = false
        this.pendingReadback = null
    }
}

export { HiZPass, TILE_SIZE }
