import { Texture } from "../Texture.js"
import { vec3 } from "../math.js"

/**
 * ReflectionProbe - A single reflection probe with position and texture
 */
class ReflectionProbe {
    constructor(id, position, worldId = 'default') {
        this.id = id
        this.position = [...position]
        this.worldId = worldId
        this.texture = null
        this.loaded = false
        this.url = null
        this.mipLevels = 6
    }
}

/**
 * ReflectionProbeManager - Manages loading, caching, and interpolating reflection probes
 *
 * Handles:
 * - Loading probes from server by worldId/position
 * - Finding closest probes to camera
 * - Interpolating between probes
 * - Uploading new captures to server
 */
class ReflectionProbeManager {
    constructor(engine) {
        this.engine = engine

        // All loaded probes: Map<probeId, ReflectionProbe>
        this.probes = new Map()

        // Probes indexed by world: Map<worldId, ReflectionProbe[]>
        this.probesByWorld = new Map()

        // Currently active probes for rendering (closest 2)
        this.activeProbes = [null, null]
        this.activeWeights = [1.0, 0.0]

        // Base/fallback environment map (used when no probes available)
        this.fallbackEnvironment = null

        // Server configuration
        this.serverBaseUrl = '/api/probes'

        // Cache settings
        this.maxCachedProbes = 10
        this.loadingProbes = new Set()  // URLs currently being loaded
    }

    /**
     * Set fallback environment map
     */
    setFallbackEnvironment(envMap) {
        this.fallbackEnvironment = envMap
    }

    /**
     * Register a probe (from capture or loaded from server)
     */
    registerProbe(probe) {
        this.probes.set(probe.id, probe)

        // Index by world
        if (!this.probesByWorld.has(probe.worldId)) {
            this.probesByWorld.set(probe.worldId, [])
        }
        this.probesByWorld.get(probe.worldId).push(probe)

        console.log(`ReflectionProbeManager: Registered probe ${probe.id} at [${probe.position.join(', ')}]`)
    }

    /**
     * Load probe from URL
     * @param {string} url - URL to HDR probe image
     * @param {vec3} position - World position of probe
     * @param {string} worldId - World identifier
     * @returns {Promise<ReflectionProbe>}
     */
    async loadProbe(url, position, worldId = 'default') {
        // Check if already loaded
        const existingId = `${worldId}_${position.join('_')}`
        if (this.probes.has(existingId)) {
            return this.probes.get(existingId)
        }

        // Check if already loading
        if (this.loadingProbes.has(url)) {
            // Wait for existing load
            await new Promise(resolve => {
                const check = () => {
                    if (!this.loadingProbes.has(url)) {
                        resolve()
                    } else {
                        setTimeout(check, 100)
                    }
                }
                check()
            })
            return this.probes.get(existingId)
        }

        this.loadingProbes.add(url)

        try {
            // Load texture (supports HDR)
            const texture = await Texture.fromImage(this.engine, url, {
                flipY: false,
                srgb: false,
                generateMips: true,
                addressMode: 'clamp-to-edge'
            })

            const probe = new ReflectionProbe(existingId, position, worldId)
            probe.texture = texture
            probe.url = url
            probe.loaded = true

            this.registerProbe(probe)

            // Enforce cache limit
            this._enforceeCacheLimit()

            return probe
        } catch (error) {
            console.error(`ReflectionProbeManager: Failed to load probe from ${url}:`, error)
            return null
        } finally {
            this.loadingProbes.delete(url)
        }
    }

    /**
     * Load probes for a world from server
     * @param {string} worldId - World identifier
     * @returns {Promise<ReflectionProbe[]>}
     */
    async loadWorldProbes(worldId) {
        try {
            // Fetch probe manifest from server
            const response = await fetch(`${this.serverBaseUrl}/${worldId}/manifest.json`)
            if (!response.ok) {
                console.warn(`ReflectionProbeManager: No probes found for world ${worldId}`)
                return []
            }

            const manifest = await response.json()
            const loadedProbes = []

            for (const probeInfo of manifest.probes) {
                const probe = await this.loadProbe(
                    `${this.serverBaseUrl}/${worldId}/${probeInfo.file}`,
                    probeInfo.position,
                    worldId
                )
                if (probe) {
                    loadedProbes.push(probe)
                }
            }

            return loadedProbes
        } catch (error) {
            console.warn(`ReflectionProbeManager: Failed to load world probes for ${worldId}:`, error)
            return []
        }
    }

    /**
     * Find closest probes to a position
     * @param {vec3} position - World position
     * @param {string} worldId - World identifier
     * @param {number} count - Number of probes to find (default 2)
     * @returns {{ probes: ReflectionProbe[], weights: number[] }}
     */
    findClosestProbes(position, worldId = 'default', count = 2) {
        const worldProbes = this.probesByWorld.get(worldId) || []

        if (worldProbes.length === 0) {
            return { probes: [], weights: [] }
        }

        // Calculate distances
        const probesWithDistance = worldProbes
            .filter(p => p.loaded && p.texture)
            .map(probe => {
                const dx = probe.position[0] - position[0]
                const dy = probe.position[1] - position[1]
                const dz = probe.position[2] - position[2]
                const distSq = dx * dx + dy * dy + dz * dz
                return { probe, distance: Math.sqrt(distSq) }
            })
            .sort((a, b) => a.distance - b.distance)

        // Get closest N probes
        const closest = probesWithDistance.slice(0, count)

        if (closest.length === 0) {
            return { probes: [], weights: [] }
        }

        if (closest.length === 1) {
            return {
                probes: [closest[0].probe],
                weights: [1.0]
            }
        }

        // Calculate interpolation weights based on inverse distance
        const totalInvDist = closest.reduce((sum, p) => sum + 1.0 / Math.max(p.distance, 0.001), 0)
        const weights = closest.map(p => (1.0 / Math.max(p.distance, 0.001)) / totalInvDist)

        return {
            probes: closest.map(p => p.probe),
            weights
        }
    }

    /**
     * Update active probes based on camera position
     * @param {vec3} cameraPosition - Camera world position
     * @param {string} worldId - Current world
     */
    updateActiveProbes(cameraPosition, worldId = 'default') {
        const { probes, weights } = this.findClosestProbes(cameraPosition, worldId, 2)

        this.activeProbes = [probes[0] || null, probes[1] || null]
        this.activeWeights = [weights[0] || 1.0, weights[1] || 0.0]
    }

    /**
     * Get active probe textures and weights for shader
     * @returns {{ textures: [Texture, Texture], weights: [number, number] }}
     */
    getActiveProbeData() {
        return {
            textures: [
                this.activeProbes[0]?.texture || this.fallbackEnvironment,
                this.activeProbes[1]?.texture || this.fallbackEnvironment
            ],
            weights: this.activeWeights
        }
    }

    /**
     * Upload a captured probe to server
     * @param {Blob} probeData - HDR PNG blob
     * @param {vec3} position - Capture position
     * @param {string} worldId - World identifier
     * @returns {Promise<boolean>} Success
     */
    async uploadProbe(probeData, position, worldId = 'default') {
        try {
            const formData = new FormData()
            formData.append('probe', probeData, 'probe.png')
            formData.append('position', JSON.stringify(position))
            formData.append('worldId', worldId)

            const response = await fetch(`${this.serverBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            })

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`)
            }

            const result = await response.json()
            console.log(`ReflectionProbeManager: Uploaded probe to ${result.url}`)

            // Load the uploaded probe
            await this.loadProbe(result.url, position, worldId)

            return true
        } catch (error) {
            console.error('ReflectionProbeManager: Failed to upload probe:', error)
            return false
        }
    }

    /**
     * Remove a probe
     */
    removeProbe(probeId) {
        const probe = this.probes.get(probeId)
        if (!probe) return

        // Remove from world index
        const worldProbes = this.probesByWorld.get(probe.worldId)
        if (worldProbes) {
            const idx = worldProbes.indexOf(probe)
            if (idx >= 0) worldProbes.splice(idx, 1)
        }

        // Destroy texture
        if (probe.texture?.texture) {
            probe.texture.texture.destroy()
        }

        this.probes.delete(probeId)
    }

    /**
     * Enforce cache limit by removing least recently used probes
     */
    _enforceeCacheLimit() {
        if (this.probes.size <= this.maxCachedProbes) return

        // Simple LRU: remove oldest probes not currently active
        const toRemove = []
        for (const [id, probe] of this.probes) {
            if (probe !== this.activeProbes[0] && probe !== this.activeProbes[1]) {
                toRemove.push(id)
                if (this.probes.size - toRemove.length <= this.maxCachedProbes) {
                    break
                }
            }
        }

        for (const id of toRemove) {
            this.removeProbe(id)
        }
    }

    /**
     * Get statistics
     */
    getStats() {
        return {
            totalProbes: this.probes.size,
            loadedProbes: [...this.probes.values()].filter(p => p.loaded).length,
            worldCount: this.probesByWorld.size,
            activeProbe0: this.activeProbes[0]?.id || 'none',
            activeProbe1: this.activeProbes[1]?.id || 'none',
            weights: this.activeWeights
        }
    }

    /**
     * Destroy all resources
     */
    destroy() {
        for (const [id] of this.probes) {
            this.removeProbe(id)
        }
        this.probes.clear()
        this.probesByWorld.clear()
    }
}

export { ReflectionProbeManager, ReflectionProbe }
