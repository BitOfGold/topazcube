import { Texture } from "../Texture.js"
import { Material } from "../Material.js"
import { Mesh } from "../Mesh.js"
import { Geometry } from "../Geometry.js"

/**
 * SpriteSystem - Manages sprite entities for billboard rendering
 *
 * Handles:
 * - Parsing sprite definitions (texture path, frames per row)
 * - Loading and caching sprite textures
 * - Computing UV transforms for sprite sheet frames
 * - Animating sprites based on frame property
 * - Creating sprite materials and meshes
 */
class SpriteSystem {
    constructor(engine) {
        this.engine = engine

        // Cache for loaded sprite textures: textureUrl -> Texture
        this._textureCache = new Map()

        // Cache for sprite materials: key -> Material
        this._materialCache = new Map()

        // Cache for sprite geometries: pivot -> Geometry
        this._geometryCache = new Map()

        // Active sprite entity tracking for animation
        this._spriteEntities = new Map()  // entityId -> spriteData
    }

    /**
     * Parse a sprite definition string
     * Format: "texture.png" or "texture.png|8" (texture|framesPerRow)
     * @param {string} spriteString - Sprite definition
     * @returns {{ url: string, framesPerRow: number }}
     */
    parseSprite(spriteString) {
        if (!spriteString || typeof spriteString !== 'string') {
            return null
        }

        const parts = spriteString.split('|')
        return {
            url: parts[0],
            framesPerRow: parts.length > 1 ? parseInt(parts[1], 10) : 1
        }
    }

    /**
     * Parse a frame animation string
     * Format: integer frame OR "startFrame|endFrame|fps"
     * @param {number|string} frame - Frame definition
     * @returns {{ currentFrame: number, startFrame: number, endFrame: number, fps: number, isAnimated: boolean }}
     */
    parseFrame(frame) {
        // Check for animated format first: "startFrame|endFrame|fps"
        if (typeof frame === 'string' && frame.includes('|')) {
            const parts = frame.split('|')
            const start = parseInt(parts[0], 10) || 0
            const end = parseInt(parts[1], 10) || start
            const fps = parseFloat(parts[2]) || 30
            return {
                currentFrame: start,
                startFrame: start,
                endFrame: end,
                fps,
                isAnimated: true
            }
        }

        // Single frame (number or numeric string)
        if (typeof frame === 'number') {
            return {
                currentFrame: frame,
                startFrame: frame,
                endFrame: frame,
                fps: 0,
                isAnimated: false
            }
        }

        if (typeof frame === 'string' && !isNaN(parseInt(frame, 10))) {
            const f = parseInt(frame, 10)
            return {
                currentFrame: f,
                startFrame: f,
                endFrame: f,
                fps: 0,
                isAnimated: false
            }
        }

        // Default: frame 0
        return {
            currentFrame: 0,
            startFrame: 0,
            endFrame: 0,
            fps: 0,
            isAnimated: false
        }
    }

    /**
     * Compute UV offset and scale for a frame in a sprite sheet
     * @param {number} frame - Frame index (0-based)
     * @param {number} framesPerRow - Number of frames per row in the sheet
     * @param {number} [totalFrames] - Optional total frames (for non-square sheets)
     * @returns {{ offset: [number, number], scale: [number, number] }}
     */
    computeFrameUV(frame, framesPerRow, totalFrames = null) {
        if (framesPerRow <= 1) {
            // Single frame - full texture
            return {
                offset: [0, 0],
                scale: [1, 1]
            }
        }

        const col = frame % framesPerRow
        const row = Math.floor(frame / framesPerRow)

        // Calculate number of rows based on totalFrames or assume single row
        // For sprite sheets like 256x32 with 8 frames, there's only 1 row
        // so scaleY should be 1 (full height), not 1/8
        const numRows = totalFrames ? Math.ceil(totalFrames / framesPerRow) : (row + 1)
        const scaleX = 1.0 / framesPerRow
        const scaleY = 1.0 / Math.max(1, numRows)

        // X offset is straightforward: column * frame width
        const xOffset = col * scaleX

        // Y offset with flipY: top of image is at v=1, bottom at v=0
        // For single-row sheets, yOffset is 0 and scaleY is 1
        // For multi-row sheets, calculate based on row
        const yOffset = row * scaleY

        return {
            offset: [xOffset, yOffset],
            scale: [scaleX, scaleY]
        }
    }

    /**
     * Get or load a sprite texture
     * @param {string} url - Texture URL
     * @returns {Promise<Texture>}
     */
    async loadTexture(url) {
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
     * Get or create a material for a sprite type
     * @param {string} textureUrl - Texture URL
     * @param {number} roughness - Roughness value (0-1)
     * @param {string} pivot - Pivot mode
     * @returns {Material}
     */
    async getSpriteMaterial(textureUrl, roughness = 0.7, pivot = 'center') {
        const key = `sprite:${textureUrl}:${pivot}:r${roughness.toFixed(2)}`

        if (this._materialCache.has(key)) {
            return this._materialCache.get(key)
        }

        // Load sprite texture
        const albedoTexture = await this.loadTexture(textureUrl)

        // Create default textures for other slots
        const normalTexture = await Texture.fromRGBA(this.engine, 0.5, 0.5, 1.0, 1.0)  // Flat normal
        const aoTexture = await Texture.fromRGBA(this.engine, 1.0, 1.0, 1.0, 1.0)      // No AO
        const rmTexture = await Texture.fromRGBA(this.engine, 0.0, roughness, 0.0, 1.0) // Roughness, no metallic
        const emissionTexture = await Texture.fromRGBA(this.engine, 0.0, 0.0, 0.0, 1.0) // No emission

        const material = new Material(
            [albedoTexture, normalTexture, aoTexture, rmTexture, emissionTexture],
            {
                billboardMode: this._pivotToMode(pivot),
                spriteRoughness: roughness
            },
            key,
            this.engine
        )

        // Enable alpha hash for cutout transparency
        material.alphaHash = true
        material.alphaHashScale = 1.0

        this._materialCache.set(key, material)
        return material
    }

    /**
     * Get or create billboard quad geometry for a pivot mode
     * @param {string} pivot - Pivot mode: 'center', 'bottom', 'horizontal'
     * @returns {Geometry}
     */
    getGeometry(pivot = 'center') {
        if (this._geometryCache.has(pivot)) {
            return this._geometryCache.get(pivot)
        }

        const geometry = Geometry.billboardQuad(this.engine, pivot)
        this._geometryCache.set(pivot, geometry)
        return geometry
    }

    /**
     * Convert pivot string to billboard mode number
     * @param {string} pivot - Pivot mode
     * @returns {number}
     */
    _pivotToMode(pivot) {
        switch (pivot) {
            case 'center': return 1
            case 'bottom': return 2
            case 'horizontal': return 3
            default: return 0
        }
    }

    /**
     * Register a sprite entity for animation tracking
     * @param {string} entityId - Entity ID
     * @param {Object} entity - Entity object
     */
    registerEntity(entityId, entity) {
        if (!entity.sprite) return

        const spriteInfo = this.parseSprite(entity.sprite)
        if (!spriteInfo) return

        const frameInfo = this.parseFrame(entity.frame || 0)

        this._spriteEntities.set(entityId, {
            entity,
            spriteInfo,
            frameInfo,
            animTime: 0
        })
    }

    /**
     * Unregister a sprite entity
     * @param {string} entityId - Entity ID
     */
    unregisterEntity(entityId) {
        this._spriteEntities.delete(entityId)
    }

    /**
     * Update sprite animations
     * @param {number} dt - Delta time in seconds
     */
    update(dt) {
        for (const [entityId, data] of this._spriteEntities) {
            const { entity, spriteInfo, frameInfo } = data

            if (!frameInfo.isAnimated) continue

            // Advance animation time
            data.animTime += dt * frameInfo.fps

            // Calculate current frame (loop by default)
            const frameCount = frameInfo.endFrame - frameInfo.startFrame + 1
            const frameOffset = Math.floor(data.animTime) % frameCount
            frameInfo.currentFrame = frameInfo.startFrame + frameOffset

            // Update entity's UV transform
            const uv = this.computeFrameUV(frameInfo.currentFrame, spriteInfo.framesPerRow)
            entity._uvTransform = [uv.offset[0], uv.offset[1], uv.scale[0], uv.scale[1]]
        }
    }

    /**
     * Get instance data for a sprite entity
     * @param {Object} entity - Entity with sprite properties
     * @returns {{ uvTransform: [number, number, number, number], color: [number, number, number, number] }}
     */
    getSpriteInstanceData(entity) {
        // If entity has pre-computed _uvTransform (from animation), use it
        if (entity._uvTransform) {
            return {
                uvTransform: entity._uvTransform,
                color: entity.color || [1, 1, 1, 1]
            }
        }

        // Otherwise compute from sprite properties
        const spriteInfo = this.parseSprite(entity.sprite)
        if (!spriteInfo) {
            return {
                uvTransform: [0, 0, 1, 1],
                color: entity.color || [1, 1, 1, 1]
            }
        }

        const frameInfo = this.parseFrame(entity.frame || 0)
        const uv = this.computeFrameUV(frameInfo.currentFrame, spriteInfo.framesPerRow)

        return {
            uvTransform: [uv.offset[0], uv.offset[1], uv.scale[0], uv.scale[1]],
            color: entity.color || [1, 1, 1, 1]
        }
    }

    /**
     * Create a complete sprite asset (geometry + material + mesh)
     * @param {string} spriteString - Sprite definition
     * @param {string} pivot - Pivot mode
     * @param {number} roughness - Roughness value
     * @returns {Promise<{ geometry: Geometry, material: Material, mesh: Mesh }>}
     */
    async createSpriteAsset(spriteString, pivot = 'center', roughness = 0.7) {
        const spriteInfo = this.parseSprite(spriteString)
        if (!spriteInfo) return null

        const geometry = this.getGeometry(pivot)
        const material = await this.getSpriteMaterial(spriteInfo.url, roughness, pivot)
        const mesh = new Mesh(this.engine, geometry, material)

        return { geometry, material, mesh, spriteInfo }
    }

    /**
     * Destroy and clean up resources
     */
    destroy() {
        this._textureCache.clear()
        this._materialCache.clear()
        this._geometryCache.clear()
        this._spriteEntities.clear()
    }
}

export { SpriteSystem }
