import { Texture } from "../Texture.js"
import { mat4 } from "../math.js"

/**
 * HistoryBufferManager - Manages A/B double-buffered textures for temporal effects
 *
 * Provides previous frame data for:
 * - SSR (Screen Space Reflections)
 * - SSGI (Screen Space Global Illumination)
 * - Temporal anti-aliasing
 * - Motion blur (future)
 */
class HistoryBufferManager {
    constructor(engine) {
        this.engine = engine
        this.frameIndex = 0  // Toggles 0/1 for A/B switching
        this.initialized = false
        this.width = 0
        this.height = 0

        // Double-buffered textures (A/B swap)
        this.colorHistory = [null, null]      // HDR color after lighting (rgba16float)
        this.depthHistory = [null, null]      // Linear depth (r32float)
        this.normalHistory = [null, null]     // World normals (rgba16float)

        // Single buffer - written each frame, read next frame
        this.velocityBuffer = null            // Motion vectors (rg16float)

        // Camera matrices history for reprojection
        this.prevView = mat4.create()
        this.prevProj = mat4.create()
        this.prevViewProj = mat4.create()
        this.prevInvViewProj = mat4.create()
        this.prevCameraPosition = [0, 0, 0]

        // Track if this is the first frame (no valid history)
        this.hasValidHistory = false

        // Textures pending destruction (wait for GPU to finish using them)
        this._pendingDestroyRing = [[], [], []]
        this._pendingDestroyIndex = 0
    }

    /**
     * Initialize or resize all history buffers
     * @param {number} width - Buffer width
     * @param {number} height - Buffer height
     */
    async initialize(width, height) {
        const { device } = this.engine

        // Skip if already initialized at this size
        if (this.initialized && this.width === width && this.height === height) {
            return
        }

        // Queue old buffers for deferred destruction
        this._queueBuffersForDestruction()

        this.width = width
        this.height = height

        // Create A/B color history buffers (HDR)
        for (let i = 0; i < 2; i++) {
            this.colorHistory[i] = await Texture.renderTarget(this.engine, 'rgba16float', width, height)
            this.colorHistory[i].label = `colorHistory${i}`
        }

        // Create A/B depth history buffers (linear depth as r32float)
        for (let i = 0; i < 2; i++) {
            const depthTex = device.createTexture({
                label: `depthHistory${i}`,
                size: [width, height],
                format: 'r32float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            })
            this.depthHistory[i] = {
                texture: depthTex,
                view: depthTex.createView(),
                width,
                height,
                format: 'r32float'
            }
        }

        // Create A/B normal history buffers
        for (let i = 0; i < 2; i++) {
            this.normalHistory[i] = await Texture.renderTarget(this.engine, 'rgba16float', width, height)
            this.normalHistory[i].label = `normalHistory${i}`
        }

        // Create velocity buffer (motion vectors in pixels)
        const velocityTex = device.createTexture({
            label: 'velocityBuffer',
            size: [width, height],
            format: 'rg16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        })
        this.velocityBuffer = {
            texture: velocityTex,
            view: velocityTex.createView(),
            width,
            height,
            format: 'rg16float'
        }

        this.initialized = true
        this.hasValidHistory = false  // Reset - need one frame to build history
    }

    /**
     * Get current frame buffers (write targets)
     * @returns {Object} Current frame buffer references
     */
    getCurrent() {
        return {
            color: this.colorHistory[this.frameIndex],
            depth: this.depthHistory[this.frameIndex],
            normal: this.normalHistory[this.frameIndex],
            velocity: this.velocityBuffer,
        }
    }

    /**
     * Get previous frame buffers (read sources)
     * @returns {Object} Previous frame buffer and camera data
     */
    getPrevious() {
        const prevIdx = 1 - this.frameIndex
        return {
            color: this.colorHistory[prevIdx],
            depth: this.depthHistory[prevIdx],
            normal: this.normalHistory[prevIdx],
            velocity: this.velocityBuffer,
            view: this.prevView,
            proj: this.prevProj,
            viewProj: this.prevViewProj,
            invViewProj: this.prevInvViewProj,
            cameraPosition: this.prevCameraPosition,
            hasValidHistory: this.hasValidHistory,
        }
    }

    /**
     * Get velocity buffer for GBuffer pass to write motion vectors
     * @returns {Object} Velocity buffer texture
     */
    getVelocityBuffer() {
        return this.velocityBuffer
    }

    /**
     * Copy lighting output to current color history
     * Called after lighting pass, before SSR/SSGI
     * @param {GPUCommandEncoder} commandEncoder - Active command encoder
     * @param {Texture} lightingOutput - HDR output from lighting pass
     */
    copyLightingToHistory(commandEncoder, lightingOutput) {
        if (!this.initialized || !lightingOutput) return

        // Skip copy if source size doesn't match history size (resize in progress)
        const srcWidth = lightingOutput.texture.width
        const srcHeight = lightingOutput.texture.height
        if (srcWidth !== this.width || srcHeight !== this.height) {
            return
        }

        const current = this.colorHistory[this.frameIndex]
        commandEncoder.copyTextureToTexture(
            { texture: lightingOutput.texture },
            { texture: current.texture },
            { width: this.width, height: this.height }
        )
    }

    /**
     * Copy GBuffer depth to current depth history
     * Note: Not currently used - particles sample GBuffer depth directly
     * @param {GPUCommandEncoder} commandEncoder - Active command encoder
     * @param {Object} depthTexture - Depth texture from GBuffer
     */
    copyDepthToHistory(commandEncoder, depthTexture) {
        // Not implemented - particles use GBuffer depth directly via texture_depth_2d
        // If needed in future, would require render pass to convert depth32float to r32float
    }

    /**
     * Copy GBuffer normals to current normal history
     * @param {GPUCommandEncoder} commandEncoder - Active command encoder
     * @param {Texture} gbufferNormal - Normal from GBuffer
     */
    copyNormalToHistory(commandEncoder, normalTexture) {
        if (!this.initialized || !normalTexture) return

        // Skip copy if source size doesn't match history size (resize in progress)
        const srcWidth = normalTexture.texture.width
        const srcHeight = normalTexture.texture.height
        if (srcWidth !== this.width || srcHeight !== this.height) {
            return
        }

        const current = this.normalHistory[this.frameIndex]
        commandEncoder.copyTextureToTexture(
            { texture: normalTexture.texture },
            { texture: current.texture },
            { width: this.width, height: this.height }
        )
    }

    /**
     * Swap buffers and save camera matrices
     * Call at end of frame, after all rendering
     * @param {Camera} camera - Current frame camera
     */
    swap(camera) {
        if (!this.initialized) return

        // Process deferred texture destruction (3 frames delayed)
        this._pendingDestroyIndex = (this._pendingDestroyIndex + 1) % 3
        const toDestroy = this._pendingDestroyRing[this._pendingDestroyIndex]
        for (const tex of toDestroy) {
            tex.destroy()
        }
        this._pendingDestroyRing[this._pendingDestroyIndex] = []

        // Save current camera matrices as "previous" for next frame
        mat4.copy(this.prevView, camera.view)
        mat4.copy(this.prevProj, camera.proj)
        mat4.copy(this.prevViewProj, camera.viewProj)
        mat4.copy(this.prevInvViewProj, camera.iViewProj)
        this.prevCameraPosition[0] = camera.position[0]
        this.prevCameraPosition[1] = camera.position[1]
        this.prevCameraPosition[2] = camera.position[2]

        // Swap buffer index for next frame
        this.frameIndex = 1 - this.frameIndex

        // After first swap, we have valid history
        this.hasValidHistory = true
    }

    /**
     * Check if history is valid (at least one frame rendered)
     * @returns {boolean} True if history buffers contain valid data
     */
    isHistoryValid() {
        return this.hasValidHistory
    }

    /**
     * Get current frame index (0 or 1)
     * @returns {number} Current frame index
     */
    getFrameIndex() {
        return this.frameIndex
    }

    /**
     * Resize all buffers
     * @param {number} width - New width
     * @param {number} height - New height
     */
    async resize(width, height) {
        await this.initialize(width, height)
    }

    /**
     * Queue old buffers for deferred destruction (GPU may still be using them)
     */
    _queueBuffersForDestruction() {
        const slot = this._pendingDestroyRing[this._pendingDestroyIndex]
        for (let i = 0; i < 2; i++) {
            if (this.colorHistory[i]?.texture) {
                slot.push(this.colorHistory[i].texture)
                this.colorHistory[i] = null
            }
            if (this.depthHistory[i]?.texture) {
                slot.push(this.depthHistory[i].texture)
                this.depthHistory[i] = null
            }
            if (this.normalHistory[i]?.texture) {
                slot.push(this.normalHistory[i].texture)
                this.normalHistory[i] = null
            }
        }
        if (this.velocityBuffer?.texture) {
            slot.push(this.velocityBuffer.texture)
            this.velocityBuffer = null
        }
    }

    /**
     * Destroy all buffer resources immediately
     */
    _destroyBuffers() {
        for (let i = 0; i < 2; i++) {
            if (this.colorHistory[i]?.texture) {
                this.colorHistory[i].texture.destroy()
                this.colorHistory[i] = null
            }
            if (this.depthHistory[i]?.texture) {
                this.depthHistory[i].texture.destroy()
                this.depthHistory[i] = null
            }
            if (this.normalHistory[i]?.texture) {
                this.normalHistory[i].texture.destroy()
                this.normalHistory[i] = null
            }
        }
        if (this.velocityBuffer?.texture) {
            this.velocityBuffer.texture.destroy()
            this.velocityBuffer = null
        }
    }

    /**
     * Clean up all resources
     */
    destroy() {
        this._destroyBuffers()
        // Clean up any pending textures in ring buffer
        for (const slot of this._pendingDestroyRing) {
            for (const tex of slot) {
                tex.destroy()
            }
        }
        this._pendingDestroyRing = [[], [], []]
        this.initialized = false
        this.hasValidHistory = false
    }
}

export { HistoryBufferManager }
