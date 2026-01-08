import { mat4, vec3, quat } from "../math.js"

/**
 * EntityManager - Data-oriented entity storage and management
 *
 * Entities are plain objects with:
 * - position: [x, y, z] (Float64)
 * - rotation: [x, y, z, w] (Quaternion)
 * - scale: [x, y, z]
 * - model: "path/to/model.glb|meshName" (ModelID)
 * - animation: "animationName" (optional)
 * - phase: 0.0-1.0 (animation phase, optional)
 * - light: { enabled, position, direction, color, geom, animation } (optional)
 *
 * Sprite properties (for billboard rendering):
 * - sprite: "texture.png|8" (texture path | framesPerRow, optional)
 * - pivot: 'center' | 'bottom' | 'horizontal' (billboard mode)
 * - frame: number or "start|end|fps" (animation frame)
 * - roughness: 0-1 (material roughness, default 0.7)
 * - color: [r, g, b, a] (tint color, default [1,1,1,1])
 *
 * Particle properties (for GPU particle emitters):
 * - particles: ParticleEmitter config object or emitter UID
 * - _emitterUID: internal - UID of registered particle emitter
 *
 * Runtime fields (calculated):
 * - _bsphere: { center: [x,y,z], radius: r }
 * - _matrix: mat4
 * - _uvTransform: [offsetX, offsetY, scaleX, scaleY] (for sprite sheets)
 * - _animState: { currentAnim, time, blendFrom, blendFromTime, blendWeight, blendDuration, isBlending }
 */
class EntityManager {
    constructor() {
        // Main entity storage: id -> entity
        this.entities = {}

        // Indices for fast lookup
        this._byModel = {}      // modelId -> Set<entityId>
        this._byAnimation = {}  // modelId|animation -> Set<entityId>
        this._lights = new Set() // entityIds with lights
        this._particles = new Set() // entityIds with particle emitters

        // ID generation
        this._nextId = 1

        // Dirty tracking for batch updates
        this._dirtyEntities = new Set()
        this._dirtyLights = new Set()
    }

    /**
     * Generate a unique entity ID
     */
    _generateId() {
        return `e${this._nextId++}`
    }

    /**
     * Create a new entity
     * @param {Object} data - Entity data
     * @returns {string} Entity ID
     */
    create(data = {}) {
        const id = data.id || this._generateId()

        const entity = {
            position: data.position || [0, 0, 0],
            rotation: data.rotation || [0, 0, 0, 1], // identity quaternion
            scale: data.scale || [1, 1, 1],
            model: data.model || null,
            animation: data.animation || null,
            phase: data.phase || 0,
            light: data.light || null,
            noRounding: data.noRounding || false,  // Exempt from pixel/position rounding

            // Sprite properties (for billboard rendering)
            sprite: data.sprite || null,           // "texture.png|8" (texture|framesPerRow)
            pivot: data.pivot || 'center',         // 'center', 'bottom', 'horizontal'
            frame: data.frame ?? 0,                // integer or "start|end|fps"
            roughness: data.roughness ?? 0.7,      // Material roughness (0-1)
            color: data.color || null,             // [r, g, b, a] tint color (null = white)

            // Particle properties (for GPU particle emitters)
            particles: data.particles || null,     // ParticleEmitter config or emitter UID
            _emitterUID: null,                     // Internal: UID of registered emitter

            // Runtime fields
            _bsphere: { center: [0, 0, 0], radius: 0 },  // radius 0 = not initialized
            _matrix: mat4.create(),
            _uvTransform: null,                    // [offsetX, offsetY, scaleX, scaleY] for sprites
            _visible: true,
            _dirty: true,

            // Animation state for blending (used by individual skins)
            _animState: {
                currentAnim: data.animation || null,
                time: (data.phase || 0) * 1.0,  // Will be scaled by animation duration
                blendFrom: null,
                blendFromTime: 0,
                blendWeight: 1.0,
                blendDuration: 0.3,
                isBlending: false
            }
        }

        this.entities[id] = entity

        // Update indices
        if (entity.model) {
            this._addToModelIndex(id, entity.model)
            if (entity.animation) {
                this._addToAnimationIndex(id, entity.model, entity.animation)
            }
        }

        if (entity.light && entity.light.enabled) {
            this._lights.add(id)
        }

        if (entity.particles) {
            this._particles.add(id)
        }

        this._dirtyEntities.add(id)
        this._updateEntityMatrix(id)

        return id
    }

    /**
     * Get entity by ID
     */
    get(id) {
        return this.entities[id]
    }

    /**
     * Update entity properties
     */
    update(id, data) {
        const entity = this.entities[id]
        if (!entity) return false

        const oldModel = entity.model
        const oldAnimation = entity.animation

        // Update transform
        if (data.position) entity.position = data.position
        if (data.rotation) entity.rotation = data.rotation
        if (data.scale) entity.scale = data.scale

        // Update model reference
        if (data.model !== undefined && data.model !== oldModel) {
            if (oldModel) {
                this._removeFromModelIndex(id, oldModel)
                if (oldAnimation) {
                    this._removeFromAnimationIndex(id, oldModel, oldAnimation)
                }
            }
            entity.model = data.model
            if (data.model) {
                this._addToModelIndex(id, data.model)
            }
        }

        // Update animation
        if (data.animation !== undefined && data.animation !== oldAnimation) {
            const model = data.model !== undefined ? data.model : entity.model
            if (oldAnimation && oldModel) {
                this._removeFromAnimationIndex(id, oldModel, oldAnimation)
            }
            entity.animation = data.animation
            if (data.animation && model) {
                this._addToAnimationIndex(id, model, data.animation)
            }
        }

        if (data.phase !== undefined) entity.phase = data.phase

        // Update light
        if (data.light !== undefined) {
            entity.light = data.light
            if (data.light && data.light.enabled) {
                this._lights.add(id)
                this._dirtyLights.add(id)
            } else {
                this._lights.delete(id)
            }
        }

        entity._dirty = true
        this._dirtyEntities.add(id)
        this._updateEntityMatrix(id)

        return true
    }

    /**
     * Delete entity
     */
    delete(id) {
        const entity = this.entities[id]
        if (!entity) return false

        // Remove from indices
        if (entity.model) {
            this._removeFromModelIndex(id, entity.model)
            if (entity.animation) {
                this._removeFromAnimationIndex(id, entity.model, entity.animation)
            }
        }

        this._lights.delete(id)
        this._particles.delete(id)
        this._dirtyEntities.delete(id)
        this._dirtyLights.delete(id)

        delete this.entities[id]
        return true
    }

    /**
     * Update entity's model matrix from transform
     */
    _updateEntityMatrix(id) {
        const entity = this.entities[id]
        if (!entity) return

        mat4.identity(entity._matrix)
        mat4.translate(entity._matrix, entity._matrix, entity.position)

        // Apply rotation from quaternion
        const rotMat = mat4.create()
        mat4.fromQuat(rotMat, entity.rotation)
        mat4.multiply(entity._matrix, entity._matrix, rotMat)

        mat4.scale(entity._matrix, entity._matrix, entity.scale)
    }

    /**
     * Update entity's bounding sphere (called when asset is loaded)
     */
    updateBoundingSphere(id, baseBsphere) {
        const entity = this.entities[id]
        if (!entity || !baseBsphere) return

        // Transform bsphere center by entity matrix
        const center = vec3.create()
        vec3.transformMat4(center, baseBsphere.center, entity._matrix)

        // Scale radius by max scale component
        const maxScale = Math.max(
            Math.abs(entity.scale[0]),
            Math.abs(entity.scale[1]),
            Math.abs(entity.scale[2])
        )

        entity._bsphere = {
            center: [center[0], center[1], center[2]],
            radius: baseBsphere.radius * maxScale
        }
    }

    /**
     * Get all entities using a specific model
     */
    getByModel(modelId) {
        const ids = this._byModel[modelId]
        if (!ids) return []
        return Array.from(ids).map(id => ({ id, entity: this.entities[id] }))
    }

    /**
     * Get all entities with a specific model and animation
     */
    getByModelAndAnimation(modelId, animation) {
        const key = `${modelId}|${animation}`
        const ids = this._byAnimation[key]
        if (!ids) return []
        return Array.from(ids).map(id => ({ id, entity: this.entities[id] }))
    }

    /**
     * Get all unique model IDs currently in use
     */
    getActiveModels() {
        return Object.keys(this._byModel)
    }

    /**
     * Get all entities with lights
     */
    getLights() {
        return Array.from(this._lights).map(id => ({ id, entity: this.entities[id] }))
    }

    /**
     * Get all entities with particle emitters
     */
    getParticles() {
        return Array.from(this._particles).map(id => ({ id, entity: this.entities[id] }))
    }

    /**
     * Get dirty entities and clear dirty flags
     */
    consumeDirtyEntities() {
        const dirty = Array.from(this._dirtyEntities)
        this._dirtyEntities.clear()
        for (const id of dirty) {
            const entity = this.entities[id]
            if (entity) entity._dirty = false
        }
        return dirty
    }

    /**
     * Get dirty lights and clear dirty flags
     */
    consumeDirtyLights() {
        const dirty = Array.from(this._dirtyLights)
        this._dirtyLights.clear()
        return dirty
    }

    /**
     * Set visibility for entity
     */
    setVisible(id, visible) {
        const entity = this.entities[id]
        if (entity) {
            entity._visible = visible
        }
    }

    /**
     * Get all visible entities
     */
    getVisible() {
        const result = []
        for (const id in this.entities) {
            if (this.entities[id]._visible) {
                result.push({ id, entity: this.entities[id] })
            }
        }
        return result
    }

    /**
     * Iterate over all entities
     */
    forEach(callback) {
        for (const id in this.entities) {
            callback(id, this.entities[id])
        }
    }

    /**
     * Get entity count
     */
    get count() {
        return Object.keys(this.entities).length
    }

    // Private index management methods

    _addToModelIndex(id, modelId) {
        if (!this._byModel[modelId]) {
            this._byModel[modelId] = new Set()
        }
        this._byModel[modelId].add(id)
    }

    _removeFromModelIndex(id, modelId) {
        if (this._byModel[modelId]) {
            this._byModel[modelId].delete(id)
            if (this._byModel[modelId].size === 0) {
                delete this._byModel[modelId]
            }
        }
    }

    _addToAnimationIndex(id, modelId, animation) {
        const key = `${modelId}|${animation}`
        if (!this._byAnimation[key]) {
            this._byAnimation[key] = new Set()
        }
        this._byAnimation[key].add(id)
    }

    _removeFromAnimationIndex(id, modelId, animation) {
        const key = `${modelId}|${animation}`
        if (this._byAnimation[key]) {
            this._byAnimation[key].delete(id)
            if (this._byAnimation[key].size === 0) {
                delete this._byAnimation[key]
            }
        }
    }

    /**
     * Set animation for entity with optional blending
     * @param {string} id - Entity ID
     * @param {string} animation - Animation name
     * @param {number} blendTime - Blend duration (0 = instant switch)
     */
    setAnimation(id, animation, blendTime = 0.3) {
        const entity = this.entities[id]
        if (!entity) return false

        const animState = entity._animState

        // If same animation, do nothing
        if (animState.currentAnim === animation) return true

        // Start blend if we have a current animation and blend time > 0
        if (blendTime > 0 && animState.currentAnim) {
            animState.blendFrom = animState.currentAnim
            animState.blendFromTime = animState.time
            animState.blendWeight = 0
            animState.blendDuration = blendTime
            animState.isBlending = true
        } else {
            animState.isBlending = false
            animState.blendFrom = null
        }

        animState.currentAnim = animation
        animState.time = 0

        // Also update the legacy animation field for compatibility
        const oldAnimation = entity.animation
        if (oldAnimation !== animation) {
            if (oldAnimation && entity.model) {
                this._removeFromAnimationIndex(id, entity.model, oldAnimation)
            }
            entity.animation = animation
            if (animation && entity.model) {
                this._addToAnimationIndex(id, entity.model, animation)
            }
        }

        return true
    }

    /**
     * Update animation time for entity
     * @param {string} id - Entity ID
     * @param {number} dt - Delta time
     */
    updateAnimationTime(id, dt) {
        const entity = this.entities[id]
        if (!entity) return

        const animState = entity._animState
        animState.time += dt

        if (animState.isBlending) {
            animState.blendFromTime += dt
            animState.blendWeight = Math.min(
                animState.blendWeight + dt / animState.blendDuration,
                1.0
            )

            if (animState.blendWeight >= 1.0) {
                animState.isBlending = false
                animState.blendFrom = null
                animState.blendWeight = 1.0
            }
        }
    }

    /**
     * Check if entity is currently blending animations
     */
    isBlending(id) {
        const entity = this.entities[id]
        return entity?._animState?.isBlending || false
    }

    /**
     * Get animation state for entity
     */
    getAnimationState(id) {
        const entity = this.entities[id]
        return entity?._animState || null
    }

    /**
     * Clear all entities
     */
    clear() {
        this.entities = {}
        this._byModel = {}
        this._byAnimation = {}
        this._lights.clear()
        this._particles.clear()
        this._dirtyEntities.clear()
        this._dirtyLights.clear()
    }

    /**
     * Serialize entities for saving
     */
    serialize() {
        const data = {}
        for (const id in this.entities) {
            const e = this.entities[id]
            data[id] = {
                position: [...e.position],
                rotation: [...e.rotation],
                scale: [...e.scale],
                model: e.model,
                animation: e.animation,
                phase: e.phase,
                light: e.light ? { ...e.light } : null,
                // Sprite properties
                sprite: e.sprite,
                pivot: e.pivot,
                frame: e.frame,
                roughness: e.roughness,
                color: e.color ? [...e.color] : null,
                // Particle properties (serialize config, not runtime UID)
                particles: e.particles
            }
        }
        return data
    }

    /**
     * Deserialize entities from saved data
     */
    deserialize(data) {
        this.clear()
        for (const id in data) {
            this.create({ ...data[id], id })
        }
    }
}

export { EntityManager }
