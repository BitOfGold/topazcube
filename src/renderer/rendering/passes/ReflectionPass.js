import { BasePass } from "./BasePass.js"
import { Texture } from "../../Texture.js"
import { ProbeCapture } from "../ProbeCapture.js"
import { ReflectionProbeManager } from "../ReflectionProbeManager.js"

/**
 * ReflectionPass - Manages reflection probes and environment lighting
 *
 * Pass 2 in the 7-pass pipeline.
 * Handles:
 * - Loading/managing reflection probes
 * - Real-time probe capture (when triggered)
 * - Probe interpolation based on camera position
 *
 * Configuration:
 * - 1024x1024 octahedral HDR texture per probe
 * - Interpolates between closest 2 probes
 */
class ReflectionPass extends BasePass {
    constructor(engine = null) {
        super('Reflection', engine)

        // Probe capture system
        this.probeCapture = null

        // Probe manager for loading/interpolating
        this.probeManager = null

        // Current world ID
        this.currentWorldId = 'default'

        // Capture request queue
        this.captureRequests = []

        // Output: combined/interpolated probe texture
        this.outputTexture = null
    }

    async _init() {
        // Initialize probe manager (lightweight, always works)
        this.probeManager = new ReflectionProbeManager(this.engine)

        // Initialize probe capture system (can fail on some systems)
        try {
            this.probeCapture = new ProbeCapture(this.engine)
            await this.probeCapture.initialize()
        } catch (e) {
            console.warn('ReflectionPass: Probe capture initialization failed:', e)
            this.probeCapture = null
        }
    }

    /**
     * Set fallback environment map (used when no probes available)
     * @param {Texture} envMap - Environment map texture
     * @param {number} encoding - 0 = equirectangular, 1 = octahedral
     */
    setFallbackEnvironment(envMap, encoding = 0) {
        if (this.probeManager) {
            this.probeManager.setFallbackEnvironment(envMap)
        }
        if (this.probeCapture) {
            this.probeCapture.setFallbackEnvironment(envMap)
            this.probeCapture.envEncoding = encoding
        }
    }

    /**
     * Load probes for a world
     */
    async loadWorldProbes(worldId) {
        this.currentWorldId = worldId
        if (this.probeManager) {
            await this.probeManager.loadWorldProbes(worldId)
        }
    }

    /**
     * Request a probe capture at position
     * Will be processed during next execute()
     */
    requestCapture(position, worldId = null) {
        this.captureRequests.push({
            position: [...position],
            worldId: worldId || this.currentWorldId
        })
    }

    /**
     * Load a specific probe from URL
     */
    async loadProbe(url, position, worldId = null) {
        if (this.probeManager) {
            return await this.probeManager.loadProbe(url, position, worldId || this.currentWorldId)
        }
        return null
    }

    async _execute(context) {
        const { camera } = context

        // Update active probes based on camera position
        if (this.probeManager && camera) {
            this.probeManager.updateActiveProbes(camera.position, this.currentWorldId)
        }

        // Process capture requests (one per frame max)
        if (this.captureRequests.length > 0 && !this.probeCapture?.isCapturing) {
            const request = this.captureRequests.shift()
            // Capture would happen here - requires renderGraph reference
            // For now, log the request
            console.log(`ReflectionPass: Capture requested at [${request.position.join(', ')}]`)
        }
    }

    async _resize(width, height) {
        // Reflection maps are fixed size, doesn't resize with screen
    }

    _destroy() {
        if (this.probeCapture) {
            this.probeCapture.destroy()
        }
        if (this.probeManager) {
            this.probeManager.destroy()
        }
    }

    /**
     * Get probe manager for external access
     */
    getProbeManager() {
        return this.probeManager
    }

    /**
     * Get probe capture system for manual triggering
     */
    getProbeCapture() {
        return this.probeCapture
    }

    /**
     * Get active probe data for lighting pass
     */
    getActiveProbeData() {
        if (this.probeManager) {
            return this.probeManager.getActiveProbeData()
        }
        return {
            textures: [null, null],
            weights: [1.0, 0.0]
        }
    }
}

export { ReflectionPass }
