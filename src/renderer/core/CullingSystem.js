import { Frustum } from "../utils/Frustum.js"
import { transformBoundingSphere } from "../utils/BoundingSphere.js"

/**
 * CullingSystem - Manages visibility culling for entities
 *
 * Performs cone-based frustum culling, distance filtering,
 * and HiZ occlusion culling with configurable limits per pass type.
 */
class CullingSystem {
    constructor(engine = null) {
        // Reference to engine for settings access
        this.engine = engine

        // Frustum for culling
        this.frustum = new Frustum()

        // HiZ pass reference for occlusion culling
        this.hizPass = null

        // Current camera data for HiZ testing
        this._viewProj = null
        this._near = 0.05
        this._far = 1000
        this._cameraPos = null

        // Stats for occlusion culling
        this._occlusionStats = {
            tested: 0,
            culled: 0
        }

        // Cached visible entity lists per pass
        this._visibleCache = {
            shadow: null,
            reflection: null,
            planarReflection: null,
            main: null
        }

        // Frame counter for cache invalidation
        this._frameId = 0
        this._cacheFrameId = -1
    }

    /**
     * Set the HiZ pass for occlusion culling
     * @param {HiZPass} hizPass - The HiZ pass instance
     */
    setHiZPass(hizPass) {
        this.hizPass = hizPass
    }

    // Culling config is now a getter that reads from engine.settings.culling
    get config() {
        return this.engine.settings.culling
    }

    /**
     * Update frustum from camera
     * @param {Camera} camera - Camera object
     * @param {number} screenWidth - Screen width in pixels
     * @param {number} screenHeight - Screen height in pixels
     */
    updateFrustum(camera, screenWidth, screenHeight) {
        // Camera uses: view, proj, fov (degrees), aspect, near, far, position, direction
        const fovRadians = camera.fov * (Math.PI / 180)
        this.frustum.update(
            camera.view,
            camera.proj,
            camera.position,
            camera.direction,
            fovRadians,
            camera.aspect,
            camera.near,
            camera.far,
            screenWidth,
            screenHeight
        )

        // Store camera data for HiZ testing
        // Copy position to avoid issues with mutable references
        this._viewProj = camera.viewProj
        this._near = camera.near
        this._far = camera.far
        this._cameraPos = [camera.position[0], camera.position[1], camera.position[2]]

        // Reset occlusion stats
        this._occlusionStats.tested = 0
        this._occlusionStats.culled = 0

        this._frameId++
    }

    /**
     * Set culling configuration for a pass type
     * @param {string} passType - 'shadow', 'reflection', or 'main'
     * @param {Object} config - { maxDistance, maxSkinned }
     */
    setConfig(passType, config) {
        const cullingConfig = this.engine?.settings?.culling
        if (cullingConfig && cullingConfig[passType]) {
            Object.assign(cullingConfig[passType], config)
        }
    }

    /**
     * Cull entities for a specific pass type
     *
     * @param {EntityManager} entityManager - Entity manager
     * @param {AssetManager} assetManager - Asset manager
     * @param {string} passType - 'shadow', 'reflection', or 'main'
     * @returns {{ visible: Array, skinnedCount: number }}
     */
    cull(entityManager, assetManager, passType = 'main') {
        const config = this.config[passType] || this.config.main
        const visible = []
        let skinnedCount = 0
        let skippedNoVisible = 0
        let skippedNoModel = 0

        entityManager.forEach((id, entity) => {
            // Skip if not visible
            if (!entity._visible) {
                skippedNoVisible++
                return
            }

            // Skip if no model
            if (!entity.model) {
                skippedNoModel++
                return
            }

            // Get bounding sphere from asset and transform by entity matrix
            const asset = assetManager.get(entity.model)
            let bsphere

            if (asset?.bsphere) {
                // Transform asset's bsphere by entity's current matrix
                bsphere = transformBoundingSphere(asset.bsphere, entity._matrix)
                // Cache it on entity for other uses
                entity._bsphere = bsphere
            } else if (entity._bsphere && entity._bsphere.radius > 0) {
                // Use existing bsphere if available
                bsphere = entity._bsphere
            } else {
                // No bsphere available, include by default
                visible.push({ id, entity, distance: 0 })
                return
            }

            // Check if culling is enabled
            const globalCullingEnabled = this.engine?.settings?.culling?.frustumEnabled !== false
            const passFrustumEnabled = config.frustum !== false

            // For planar reflection, mirror the bounding sphere across the ground level
            // This ensures we cull based on where the object appears in the reflection
            let cullBsphere = bsphere
            if (passType === 'planarReflection') {
                const groundLevel = this.engine?.settings?.planarReflection?.groundLevel ?? 0
                // Mirror Y position: mirroredY = 2 * groundLevel - originalY
                cullBsphere = {
                    center: [
                        bsphere.center[0],
                        2 * groundLevel - bsphere.center[1],
                        bsphere.center[2]
                    ],
                    radius: bsphere.radius
                }
            }

            // Distance test (always apply when global culling is enabled)
            const distance = this.frustum.getDistance(cullBsphere)
            if (globalCullingEnabled && distance - cullBsphere.radius > config.maxDistance) {
                return // Too far
            }

            // Pixel size test (always apply when global culling is enabled)
            if (globalCullingEnabled && config.minPixelSize > 0) {
                const projectedSize = this.frustum.getProjectedSize(cullBsphere, distance)
                if (projectedSize < config.minPixelSize) {
                    return // Too small to see
                }
            }

            // Frustum test (only when both global AND per-pass frustum culling is enabled)
            if (globalCullingEnabled && passFrustumEnabled && !this.frustum.testSpherePlanes(cullBsphere)) {
                return // Outside frustum
            }

            // HiZ occlusion culling for entities
            if (passType === 'main' && this.hizPass && this._viewProj && this._cameraPos) {
                const occlusionEnabled = this.engine?.settings?.occlusionCulling?.enabled
                if (occlusionEnabled) {
                    this._occlusionStats.tested++
                    if (this.hizPass.testSphereOcclusion(bsphere, this._viewProj, this._near, this._far, this._cameraPos)) {
                        this._occlusionStats.culled++
                        return // Occluded by previous frame's geometry
                    }
                }
            }

            // Check skinned limit (asset already fetched above)
            const isSkinned = asset?.hasSkin === true

            if (isSkinned) {
                if (skinnedCount >= config.maxSkinned) {
                    return // Too many skinned already
                }
                skinnedCount++
            }

            visible.push({
                id,
                entity,
                distance,
                isSkinned
            })
        })

        // Sort by distance for front-to-back rendering (reduces overdraw)
        visible.sort((a, b) => a.distance - b.distance)

        return { visible, skinnedCount }
    }

    /**
     * Group visible entities by model for instancing
     *
     * @param {Array} visibleEntities - Array from cull()
     * @returns {Map<string, Array>} Map of modelId -> entities
     */
    groupByModel(visibleEntities) {
        const groups = new Map()

        for (const item of visibleEntities) {
            const modelId = item.entity.model
            if (!groups.has(modelId)) {
                groups.set(modelId, [])
            }
            groups.get(modelId).push(item)
        }

        return groups
    }

    /**
     * Group visible entities by model and animation for skinned meshes
     * Entities with same animation can potentially share animation state
     *
     * @param {Array} visibleEntities - Array from cull()
     * @param {number} phaseQuantization - Quantize phase to this step (default 0.05 = 20 groups per animation)
     * @returns {Map<string, Array>} Map of "modelId|animation|quantizedPhase" -> entities
     */
    groupByModelAndAnimation(visibleEntities, phaseQuantization = 0.05) {
        const groups = new Map()

        for (const item of visibleEntities) {
            const entity = item.entity
            let key = entity.model

            if (item.isSkinned && entity.animation) {
                const quantizedPhase = Math.floor(entity.phase / phaseQuantization) * phaseQuantization
                key = `${entity.model}|${entity.animation}|${quantizedPhase.toFixed(2)}`
            }

            if (!groups.has(key)) {
                groups.set(key, [])
            }
            groups.get(key).push(item)
        }

        return groups
    }

    /**
     * Get statistics about culling
     */
    getStats(entityManager, assetManager) {
        const total = entityManager.count
        const { visible } = this.cull(entityManager, assetManager, 'main')
        const culled = total - visible.length

        return {
            total,
            visible: visible.length,
            culled,
            cullPercent: total > 0 ? ((culled / total) * 100).toFixed(1) : 0
        }
    }

    /**
     * Get occlusion culling statistics
     */
    getOcclusionStats() {
        return {
            tested: this._occlusionStats.tested,
            culled: this._occlusionStats.culled,
            cullPercent: this._occlusionStats.tested > 0
                ? ((this._occlusionStats.culled / this._occlusionStats.tested) * 100).toFixed(1)
                : 0
        }
    }
}

export { CullingSystem }
