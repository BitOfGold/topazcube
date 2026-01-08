/**
 * ParticleEmitter - Configuration class for particle spawning behavior
 *
 * Defines how particles are spawned, their initial properties, physics behavior,
 * and rendering options. Used by ParticleSystem to create and manage particles.
 */

let _emitterUID = 1

class ParticleEmitter {
    constructor(config = {}) {
        this.uid = _emitterUID++
        this.name = config.name || `emitter_${this.uid}`

        // Apply behavior preset FIRST so user config can override
        const behavior = config.behavior || 'default'
        this.behavior = behavior

        // Get preset defaults (empty for 'default')
        const presetDefaults = this._getPresetDefaults(behavior)

        // Merge: defaults -> preset -> user config
        const merged = { ...presetDefaults, ...config }

        // === Spawning Configuration ===
        // Emitter position in world space
        this.position = merged.position || [0, 0, 0]

        // Spawn volume: 'point' | 'box' | 'sphere'
        this.volume = merged.volume || 'point'

        // Size of spawn volume (for box: [x,y,z], for sphere: [radius])
        this.volumeSize = merged.volumeSize || [1, 1, 1]

        // Particles spawned per second (0 = manual spawn only)
        this.spawnRate = merged.spawnRate ?? 10

        // Initial burst of particles on activation
        this.spawnBurst = merged.spawnBurst ?? 0

        // Maximum particles in this emitter's pool
        this.maxParticles = merged.maxParticles ?? 1000

        // === Particle Initial Properties ===
        // Lifetime range [min, max] in seconds
        this.lifetime = merged.lifetime || [1.0, 2.0]

        // Initial speed range [min, max]
        this.speed = merged.speed || [1.0, 3.0]

        // Emission direction (normalized) - particles shoot this way
        this.direction = merged.direction || [0, 1, 0]

        // Cone spread (0 = tight beam, 1 = hemisphere)
        this.spread = merged.spread ?? 0.5

        // Size over lifetime [start, end]
        this.size = merged.size || [0.5, 0.1]

        // Particle color tint [r, g, b, a]
        this.color = merged.color || [1, 1, 1, 1]

        // Fade timing in seconds
        this.fadeIn = merged.fadeIn ?? 0.1
        this.fadeOut = merged.fadeOut ?? 0.3

        // === Physics ===
        // Gravity acceleration [x, y, z]
        this.gravity = merged.gravity || [0, -9.8, 0]

        // Air resistance (0 = none, 1 = high drag)
        this.drag = merged.drag ?? 0.1

        // Random turbulence strength
        this.turbulence = merged.turbulence ?? 0.0

        // === Rendering ===
        // Texture URL for particles
        this.texture = merged.texture || null

        // Frames per row for sprite sheets
        this.framesPerRow = merged.framesPerRow ?? 1

        // Total frames in sprite sheet (optional, for animation)
        this.totalFrames = merged.totalFrames ?? null

        // Animation FPS (0 = no animation)
        this.animationFPS = merged.animationFPS ?? 0

        // Blend mode: 'additive' | 'alpha'
        this.blendMode = merged.blendMode || 'additive'

        // Depth offset to prevent z-fighting
        this.zOffset = merged.zOffset ?? 0.01

        // Soft particle depth fade distance (meters)
        this.softness = merged.softness ?? 0.25

        // Rotation speed in radians per second
        this.rotationSpeed = merged.rotationSpeed ?? 0.5

        // Whether particles receive basic lighting
        this.lit = merged.lit ?? false

        // Emissive multiplier (1.0 = normal, >1 = brighter for fire/sparks)
        this.emissive = merged.emissive ?? 1.0

        // === Runtime State ===
        this.enabled = true
        this.spawnAccumulator = 0  // Fractional particle spawn buildup
        this.totalSpawned = 0
        this.aliveCount = 0

        // GPU buffer references (set by ParticleSystem)
        this.particleBuffer = null
        this.indirectBuffer = null
    }

    /**
     * Get behavior preset defaults
     * @param {string} behavior - Preset name
     * @returns {Object} Default values for the preset
     */
    _getPresetDefaults(behavior) {
        const presets = {
            smoke: {
                lifetime: [2.0, 4.0],
                speed: [0.5, 1.5],
                direction: [0, 1, 0],
                spread: 0.3,
                size: [0.3, 2.0],
                color: [0.5, 0.5, 0.5, 0.6],
                fadeIn: 0.2,
                fadeOut: 1.0,
                gravity: [0, 0.5, 0],
                drag: 0.3,
                turbulence: 0.3,
                blendMode: 'alpha',
                rotationSpeed: 0.3,
                softness: 0.25
            },
            fire: {
                lifetime: [0.3, 0.8],
                speed: [2.0, 4.0],
                direction: [0, 1, 0],
                spread: 0.2,
                size: [0.5, 0.1],
                color: [1.0, 0.6, 0.2, 1.0],
                fadeIn: 0.05,
                fadeOut: 0.3,
                gravity: [0, 2.0, 0],
                drag: 0.1,
                turbulence: 0.5,
                blendMode: 'additive',
                emissive: 3.0,  // Bright fire
                lit: false      // Fire is self-illuminating
            },
            sparks: {
                lifetime: [0.5, 1.5],
                speed: [5.0, 10.0],
                direction: [0, 1, 0],
                spread: 0.8,
                size: [0.1, 0.05],
                color: [1.0, 0.8, 0.3, 1.0],
                fadeIn: 0.0,
                fadeOut: 0.5,
                gravity: [0, -9.8, 0],
                drag: 0.05,
                turbulence: 0.1,
                blendMode: 'additive',
                emissive: 5.0,  // Very bright sparks
                lit: false      // Sparks are self-illuminating
            },
            fog: {
                lifetime: [5.0, 10.0],
                speed: [0.1, 0.3],
                direction: [0, 0, 0],
                spread: 1.0,
                size: [2.0, 3.0],
                color: [0.8, 0.8, 0.9, 0.3],
                fadeIn: 1.0,
                fadeOut: 2.0,
                gravity: [0, 0, 0],
                drag: 0.5,
                turbulence: 0.1,
                blendMode: 'alpha',
                volume: 'box'
            }
        }
        return presets[behavior] || {}
    }

    /**
     * Get a random value within a [min, max] range
     * @param {number[]} range - [min, max]
     * @param {number} seed - Random seed (0-1)
     * @returns {number}
     */
    static randomInRange(range, seed) {
        return range[0] + (range[1] - range[0]) * seed
    }

    /**
     * Get a random point within the spawn volume
     * @param {number[]} seeds - [seed1, seed2, seed3] random values (0-1)
     * @returns {number[]} - [x, y, z] position
     */
    getSpawnPosition(seeds) {
        const [s1, s2, s3] = seeds
        const [px, py, pz] = this.position

        switch (this.volume) {
            case 'point':
                return [px, py, pz]

            case 'box': {
                const [sx, sy, sz] = this.volumeSize
                return [
                    px + (s1 - 0.5) * sx,
                    py + (s2 - 0.5) * sy,
                    pz + (s3 - 0.5) * sz
                ]
            }

            case 'sphere': {
                // Uniform distribution within sphere
                const radius = this.volumeSize[0] * Math.cbrt(s1)
                const theta = s2 * 2 * Math.PI
                const phi = Math.acos(2 * s3 - 1)
                return [
                    px + radius * Math.sin(phi) * Math.cos(theta),
                    py + radius * Math.cos(phi),
                    pz + radius * Math.sin(phi) * Math.sin(theta)
                ]
            }

            default:
                return [px, py, pz]
        }
    }

    /**
     * Get a random emission direction with spread
     * @param {number[]} seeds - [seed1, seed2] random values (0-1)
     * @returns {number[]} - [x, y, z] normalized direction
     */
    getEmissionDirection(seeds) {
        const [s1, s2] = seeds
        const [dx, dy, dz] = this.direction

        if (this.spread <= 0) {
            // No spread - use exact direction
            return [dx, dy, dz]
        }

        // Create cone around direction
        // Map spread to cone half-angle (0=0°, 1=90°)
        const halfAngle = this.spread * Math.PI * 0.5
        const theta = s1 * 2 * Math.PI
        const cosAngle = Math.cos(halfAngle * s2)
        const sinAngle = Math.sqrt(1 - cosAngle * cosAngle)

        // Random direction in cone around +Y
        let rx = sinAngle * Math.cos(theta)
        let ry = cosAngle
        let rz = sinAngle * Math.sin(theta)

        // Rotate from +Y to emission direction
        // Use simplified rotation (assumes direction is normalized)
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz)
        if (len < 0.001) return [rx, ry, rz]  // No direction

        const ndx = dx / len
        const ndy = dy / len
        const ndz = dz / len

        // If direction is close to +Y, return as-is
        if (ndy > 0.999) return [rx, ry, rz]
        if (ndy < -0.999) return [rx, -ry, rz]  // Flip for -Y

        // Build rotation matrix from +Y to direction
        const ax = -ndz, az = ndx  // Cross product Y × dir (simplified)
        const alen = Math.sqrt(ax * ax + az * az)
        const nax = ax / alen, naz = az / alen
        const c = ndy, s = Math.sqrt(1 - c * c)

        // Rodrigues' rotation formula (simplified for rotation around axis in XZ plane)
        const outX = rx * (c + nax * nax * (1 - c)) + ry * (-naz * s) + rz * (nax * naz * (1 - c))
        const outY = rx * (naz * s) + ry * c + rz * (-nax * s)
        const outZ = rx * (naz * nax * (1 - c)) + ry * (nax * s) + rz * (c + naz * naz * (1 - c))

        return [outX, outY, outZ]
    }

    /**
     * Clone this emitter with optional overrides
     * @param {Object} overrides - Properties to override
     * @returns {ParticleEmitter}
     */
    clone(overrides = {}) {
        const config = {
            ...this.toJSON(),
            ...overrides,
            _presetApplied: true  // Don't re-apply preset
        }
        return new ParticleEmitter(config)
    }

    /**
     * Serialize emitter configuration to JSON
     * @returns {Object}
     */
    toJSON() {
        return {
            name: this.name,
            position: [...this.position],
            volume: this.volume,
            volumeSize: [...this.volumeSize],
            spawnRate: this.spawnRate,
            spawnBurst: this.spawnBurst,
            maxParticles: this.maxParticles,
            lifetime: [...this.lifetime],
            speed: [...this.speed],
            direction: [...this.direction],
            spread: this.spread,
            size: [...this.size],
            color: [...this.color],
            fadeIn: this.fadeIn,
            fadeOut: this.fadeOut,
            gravity: [...this.gravity],
            drag: this.drag,
            turbulence: this.turbulence,
            texture: this.texture,
            framesPerRow: this.framesPerRow,
            totalFrames: this.totalFrames,
            animationFPS: this.animationFPS,
            blendMode: this.blendMode,
            zOffset: this.zOffset,
            softness: this.softness,
            rotationSpeed: this.rotationSpeed,
            lit: this.lit,
            emissive: this.emissive,
            behavior: this.behavior,
            enabled: this.enabled
        }
    }

    /**
     * Create emitter from JSON
     * @param {Object} json - Serialized configuration
     * @returns {ParticleEmitter}
     */
    static fromJSON(json) {
        return new ParticleEmitter({ ...json, _presetApplied: true })
    }
}

export { ParticleEmitter }
