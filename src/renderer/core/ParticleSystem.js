import { ParticleEmitter } from "./ParticleEmitter.js"
import { Texture } from "../Texture.js"

/**
 * ParticleSystem - Main system managing particle emitters and GPU resources
 *
 * Handles:
 * - Emitter registry and lifecycle
 * - GPU buffer allocation with global budget
 * - Texture caching for particle sprites
 * - Compute shader dispatch for particle simulation
 * - Data preparation for particle rendering
 */

// Particle struct size in bytes (must match WGSL)
// position (vec3f) + lifetime (f32) + velocity (vec3f) + maxLifetime (f32) +
// color (vec4f) + size (vec2f) + rotation (f32) + flags (u32) +
// lighting (vec3f) + lightingPad (f32) = 80 bytes
const PARTICLE_STRIDE = 80

// Spawn request struct size: position (vec3f) + velocity (vec3f) + lifetime (f32) +
// maxLifetime (f32) + color (vec4f) + startSize (f32) + endSize (f32) +
// seed (f32) + flags (u32) = 64 bytes
const SPAWN_REQUEST_STRIDE = 64

class ParticleSystem {
    constructor(engine) {
        this.engine = engine
        this.device = engine.device

        // Global particle budget
        this.globalMaxParticles = 50000
        this.globalAliveCount = 0

        // Emitter registry: uid -> ParticleEmitter
        this._emitters = new Map()

        // Active emitter list (for iteration)
        this._activeEmitters = []

        // Texture cache: url -> Texture
        this._textureCache = new Map()

        // Default particle texture (white circle)
        this._defaultTexture = null

        // GPU resources (created on first use)
        this._particleBuffer = null      // Storage buffer for all particles
        this._spawnBuffer = null         // Buffer for spawn requests
        this._emitterBuffer = null       // Buffer for emitter uniforms
        this._counterBuffer = null       // Atomic counters
        this._readbackBuffer = null      // For reading counter values

        // Compute pipeline (created by ParticlePass)
        this.simulatePipeline = null
        this.spawnPipeline = null

        // Time tracking
        this._time = 0
        this._lastSpawnTime = 0

        // Spawn queue: accumulated spawn requests to send to GPU
        this._spawnQueue = []
        this._maxSpawnPerFrame = 1000  // Limit spawns per frame

        // Emitter uniforms buffer data
        this._emitterData = new Float32Array(32)  // Per-emitter uniforms

        this._initialized = false
    }

    /**
     * Initialize GPU resources
     */
    async init() {
        if (this._initialized) return

        // Create particle storage buffer
        this._particleBuffer = this.device.createBuffer({
            size: this.globalMaxParticles * PARTICLE_STRIDE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'ParticleSystem particles'
        })

        // Create spawn request buffer (sized for max spawns per frame)
        this._spawnBuffer = this.device.createBuffer({
            size: this._maxSpawnPerFrame * SPAWN_REQUEST_STRIDE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'ParticleSystem spawn requests'
        })

        // Create counter buffer: [aliveCount, nextFreeIndex, spawnCount, frameCount]
        this._counterBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            label: 'ParticleSystem counters'
        })

        // Readback buffer for counter values
        this._readbackBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            label: 'ParticleSystem counter readback'
        })

        // Create emitter uniforms buffer
        this._emitterBuffer = this.device.createBuffer({
            size: 256,  // Enough for emitter parameters
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'ParticleSystem emitter uniforms'
        })

        // Initialize counters
        const counterData = new Uint32Array([0, 0, 0, 0])  // [aliveCount, nextFreeIndex, spawnCount, frameCount]
        this.device.queue.writeBuffer(this._counterBuffer, 0, counterData)

        // Create default texture (white circle with soft edges)
        this._defaultTexture = await this._createDefaultTexture()

        this._initialized = true
    }

    /**
     * Create a default particle texture (soft white circle)
     */
    async _createDefaultTexture() {
        const size = 64
        const data = new Uint8Array(size * size * 4)

        const center = size / 2
        const maxRadius = size / 2

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - center + 0.5
                const dy = y - center + 0.5
                const dist = Math.sqrt(dx * dx + dy * dy)

                // Soft circular falloff
                const alpha = Math.max(0, 1 - dist / maxRadius)
                const softAlpha = alpha * alpha * (3 - 2 * alpha)  // Smoothstep

                const i = (y * size + x) * 4
                data[i + 0] = 255     // R
                data[i + 1] = 255     // G
                data[i + 2] = 255     // B
                data[i + 3] = Math.floor(softAlpha * 255)  // A
            }
        }

        return Texture.fromRawData(this.engine, data, size, size, {
            srgb: true,
            generateMips: true
        })
    }

    /**
     * Load or get cached particle texture
     * @param {string} url - Texture URL
     * @returns {Promise<Texture>}
     */
    async loadTexture(url) {
        if (!url) return this._defaultTexture

        if (this._textureCache.has(url)) {
            return this._textureCache.get(url)
        }

        const texture = await Texture.fromImage(this.engine, url, {
            srgb: true,
            generateMips: true,
            flipY: true
        })

        this._textureCache.set(url, texture)
        return texture
    }

    /**
     * Register a new particle emitter
     * @param {ParticleEmitter|Object} config - Emitter or configuration
     * @returns {ParticleEmitter}
     */
    addEmitter(config) {
        const emitter = config instanceof ParticleEmitter
            ? config
            : new ParticleEmitter(config)

        this._emitters.set(emitter.uid, emitter)
        this._activeEmitters.push(emitter)

        // Queue initial burst if specified
        if (emitter.spawnBurst > 0) {
            this._queueSpawn(emitter, emitter.spawnBurst)
        }

        return emitter
    }

    /**
     * Remove an emitter
     * @param {number|ParticleEmitter} emitterOrId - Emitter or UID
     */
    removeEmitter(emitterOrId) {
        const uid = typeof emitterOrId === 'number' ? emitterOrId : emitterOrId.uid
        const emitter = this._emitters.get(uid)
        if (!emitter) return

        this._emitters.delete(uid)
        const idx = this._activeEmitters.indexOf(emitter)
        if (idx >= 0) this._activeEmitters.splice(idx, 1)
    }

    /**
     * Get emitter by UID
     * @param {number} uid - Emitter UID
     * @returns {ParticleEmitter|null}
     */
    getEmitter(uid) {
        return this._emitters.get(uid) || null
    }

    /**
     * Queue particles for spawning
     * @param {ParticleEmitter} emitter - Emitter to spawn from
     * @param {number} count - Number of particles to spawn
     */
    _queueSpawn(emitter, count) {
        if (!emitter.enabled) return

        // Respect global budget
        const available = this.globalMaxParticles - this.globalAliveCount
        const toSpawn = Math.min(count, available, emitter.maxParticles - emitter.aliveCount)

        if (toSpawn <= 0) return

        for (let i = 0; i < toSpawn; i++) {
            // Generate random seeds
            const seed1 = Math.random()
            const seed2 = Math.random()
            const seed3 = Math.random()
            const seed4 = Math.random()
            const seed5 = Math.random()

            // Calculate spawn position and velocity
            const position = emitter.getSpawnPosition([seed1, seed2, seed3])
            const direction = emitter.getEmissionDirection([seed4, seed5])
            const speed = ParticleEmitter.randomInRange(emitter.speed, Math.random())
            const velocity = [
                direction[0] * speed,
                direction[1] * speed,
                direction[2] * speed
            ]

            // Calculate lifetime
            const lifetime = ParticleEmitter.randomInRange(emitter.lifetime, Math.random())

            // Random rotation: -PI to +PI (sign determines spin direction)
            const rotation = (Math.random() - 0.5) * Math.PI * 2

            // Flags: bit 0 = alive, bit 1 = additive, bits 8-15 = emitter index
            const emitterIndex = this._activeEmitters.indexOf(emitter)
            const flags = 1 | (emitter.blendMode === 'additive' ? 2 : 0) | ((emitterIndex & 0xFF) << 8)

            this._spawnQueue.push({
                emitter,
                position,
                velocity,
                lifetime,
                maxLifetime: lifetime,
                color: [...emitter.color],
                startSize: emitter.size[0],
                endSize: emitter.size[1],
                rotation,
                flags
            })
        }
    }

    /**
     * Update particle system
     * @param {number} dt - Delta time in seconds
     */
    update(dt) {
        this._time += dt

        // Estimate particle deaths based on average lifetime
        // This prevents globalAliveCount from growing forever
        this._estimateDeaths(dt)

        // Process spawn rates for each emitter
        for (const emitter of this._activeEmitters) {
            if (!emitter.enabled || emitter.spawnRate <= 0) continue

            // Accumulate spawn time
            emitter.spawnAccumulator += dt * emitter.spawnRate
            const toSpawn = Math.floor(emitter.spawnAccumulator)

            if (toSpawn > 0) {
                emitter.spawnAccumulator -= toSpawn
                this._queueSpawn(emitter, toSpawn)
            }
        }
    }

    /**
     * Estimate particle deaths based on spawn history and average lifetime
     * @param {number} dt - Delta time
     */
    _estimateDeaths(dt) {
        // Simple estimation: assume deaths occur at roughly the spawn rate
        // after accounting for average particle lifetime
        // This is approximate but prevents the counter from growing forever

        let totalSpawnRate = 0
        let totalAvgLifetime = 0
        let activeCount = 0

        for (const emitter of this._activeEmitters) {
            if (emitter.enabled && emitter.spawnRate > 0) {
                totalSpawnRate += emitter.spawnRate
                const avgLifetime = (emitter.lifetime[0] + emitter.lifetime[1]) / 2
                totalAvgLifetime += avgLifetime
                activeCount++
            }
        }

        if (activeCount > 0) {
            const avgLifetime = totalAvgLifetime / activeCount
            // At steady state: deaths per second ≈ spawn rate
            // Apply deaths proportional to dt
            const estimatedDeaths = totalSpawnRate * dt
            this.globalAliveCount = Math.max(0, this.globalAliveCount - estimatedDeaths)
        }
    }

    /**
     * Write spawn requests to GPU and execute spawn compute pass
     * @param {GPUCommandEncoder} commandEncoder
     */
    executeSpawn(commandEncoder) {
        if (this._spawnQueue.length === 0) return

        // Limit spawns per frame
        const toProcess = Math.min(this._spawnQueue.length, this._maxSpawnPerFrame)
        const spawnData = new Float32Array(toProcess * (SPAWN_REQUEST_STRIDE / 4))

        for (let i = 0; i < toProcess; i++) {
            const spawn = this._spawnQueue[i]
            const offset = i * 16  // 16 floats per spawn

            // position (vec3f) + lifetime (f32)
            spawnData[offset + 0] = spawn.position[0]
            spawnData[offset + 1] = spawn.position[1]
            spawnData[offset + 2] = spawn.position[2]
            spawnData[offset + 3] = spawn.lifetime

            // velocity (vec3f) + maxLifetime (f32)
            spawnData[offset + 4] = spawn.velocity[0]
            spawnData[offset + 5] = spawn.velocity[1]
            spawnData[offset + 6] = spawn.velocity[2]
            spawnData[offset + 7] = spawn.maxLifetime

            // color (vec4f)
            spawnData[offset + 8] = spawn.color[0]
            spawnData[offset + 9] = spawn.color[1]
            spawnData[offset + 10] = spawn.color[2]
            spawnData[offset + 11] = spawn.color[3]

            // startSize + endSize + rotation + flags
            spawnData[offset + 12] = spawn.startSize
            spawnData[offset + 13] = spawn.endSize
            spawnData[offset + 14] = spawn.rotation
            // flags needs to be written as uint32
        }

        // Write flags separately as uint32
        const flagsView = new Uint32Array(spawnData.buffer)
        for (let i = 0; i < toProcess; i++) {
            flagsView[i * 16 + 15] = this._spawnQueue[i].flags
        }

        // Upload spawn data
        this.device.queue.writeBuffer(this._spawnBuffer, 0, spawnData)

        // Update spawn count in counter buffer
        const counterUpdate = new Uint32Array([0, 0, toProcess, 0])
        this.device.queue.writeBuffer(this._counterBuffer, 8, counterUpdate.subarray(2, 3))

        // Clear processed spawns
        this._spawnQueue.splice(0, toProcess)

        // Update alive count estimate
        this.globalAliveCount += toProcess
        for (const emitter of this._activeEmitters) {
            emitter.totalSpawned += toProcess
        }
    }

    /**
     * Prepare emitter uniforms for compute/render
     * @param {ParticleEmitter} emitter
     * @returns {Float32Array}
     */
    getEmitterUniforms(emitter) {
        const data = this._emitterData

        // Gravity (vec3f) + dt (f32)
        data[0] = emitter.gravity[0]
        data[1] = emitter.gravity[1]
        data[2] = emitter.gravity[2]
        data[3] = 0  // dt will be set by pass

        // drag + turbulence + fadeIn + fadeOut
        data[4] = emitter.drag
        data[5] = emitter.turbulence
        data[6] = emitter.fadeIn
        data[7] = emitter.fadeOut

        // startSize + endSize + time + maxParticles
        data[8] = emitter.size[0]
        data[9] = emitter.size[1]
        data[10] = this._time
        data[11] = emitter.maxParticles

        // Color (vec4f)
        data[12] = emitter.color[0]
        data[13] = emitter.color[1]
        data[14] = emitter.color[2]
        data[15] = emitter.color[3]

        // Softness + zOffset + blendMode (as float) + lit
        data[16] = emitter.softness
        data[17] = emitter.zOffset
        data[18] = emitter.blendMode === 'additive' ? 1.0 : 0.0
        data[19] = emitter.lit ? 1.0 : 0.0

        return data
    }

    /**
     * Get particle buffer for rendering
     * @returns {GPUBuffer}
     */
    getParticleBuffer() {
        return this._particleBuffer
    }

    /**
     * Get counter buffer
     * @returns {GPUBuffer}
     */
    getCounterBuffer() {
        return this._counterBuffer
    }

    /**
     * Get spawn buffer
     * @returns {GPUBuffer}
     */
    getSpawnBuffer() {
        return this._spawnBuffer
    }

    /**
     * Get all active emitters
     * @returns {ParticleEmitter[]}
     */
    getActiveEmitters() {
        return this._activeEmitters.filter(e => e.enabled)
    }

    /**
     * Get total particle count across all emitters
     * @returns {number}
     */
    getTotalParticleCount() {
        return this.globalAliveCount
    }

    /**
     * Get default particle texture
     * @returns {Texture}
     */
    getDefaultTexture() {
        return this._defaultTexture
    }

    /**
     * Reset all particles (clear all emitters)
     */
    reset() {
        // Clear counters
        const counterData = new Uint32Array([0, 0, 0, 0])
        this.device.queue.writeBuffer(this._counterBuffer, 0, counterData)

        this.globalAliveCount = 0
        this._spawnQueue = []

        for (const emitter of this._activeEmitters) {
            emitter.aliveCount = 0
            emitter.spawnAccumulator = 0
        }
    }

    /**
     * Spawn a burst of particles at a specific position
     * @param {Object} config - Burst configuration
     * @param {number[]} config.position - [x, y, z] world position
     * @param {number} config.count - Number of particles
     * @param {Object} [config.overrides] - Emitter property overrides
     * @returns {ParticleEmitter} Temporary emitter used for the burst
     */
    burst(config) {
        const { position, count, overrides = {} } = config

        // Create temporary emitter for this burst
        const emitter = new ParticleEmitter({
            position,
            spawnRate: 0,  // No continuous spawning
            spawnBurst: count,
            ...overrides
        })

        // Add emitter (burst will be queued automatically)
        this.addEmitter(emitter)

        return emitter
    }

    /**
     * Destroy and clean up resources
     */
    destroy() {
        if (this._particleBuffer) {
            this._particleBuffer.destroy()
            this._particleBuffer = null
        }
        if (this._spawnBuffer) {
            this._spawnBuffer.destroy()
            this._spawnBuffer = null
        }
        if (this._counterBuffer) {
            this._counterBuffer.destroy()
            this._counterBuffer = null
        }
        if (this._readbackBuffer) {
            this._readbackBuffer.destroy()
            this._readbackBuffer = null
        }
        if (this._emitterBuffer) {
            this._emitterBuffer.destroy()
            this._emitterBuffer = null
        }

        this._textureCache.clear()
        this._emitters.clear()
        this._activeEmitters = []
        this._spawnQueue = []
        this._initialized = false
    }
}

export { ParticleSystem, PARTICLE_STRIDE, SPAWN_REQUEST_STRIDE }
