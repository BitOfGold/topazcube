var _UID = 10001

class Material {

    constructor(textures = [], uniforms = {}, name = null, engine = null) {
        this.uid = _UID++
        this.name = name || `Material_${this.uid}`
        this.uniforms = uniforms
        this._textures = textures

        this.engine = engine
        // If any textures have .engine, use that
        for (const tex of textures) {
            if (tex && tex.engine) {
                this.engine = tex.engine
                break
            }
        }

        // Transparency settings
        this.transparent = false       // True for alpha blended materials (glass, water)
        this.opacity = 1.0             // Base opacity value (0-1)
        this.opacityTexture = null     // Optional opacity texture

        // Alpha hashing settings (for cutout transparency like leaves)
        this.alphaHash = false
        this.alphaHashScale = 1.0

        // Alpha source settings
        this.luminanceToAlpha = false  // Use base color luminance as alpha (black=transparent)

        // Force emissive: use base color (albedo) as emission
        this.forceEmissive = false

        // Specular boost: enables 3 additional specular lights for shiny materials (0-1, default 0 = disabled)
        this.specularBoost = 0

        // Double-sided rendering: disable backface culling
        this.doubleSided = false
    }

    /**
     * Get textures array, substituting albedo for emission if forceEmissive is true
     * Texture order: [albedo, normal, ambient, rm, emission]
     */
    get textures() {
        if (this.forceEmissive && this._textures.length >= 5 && this._textures[0]) {
            // Return copy with albedo texture as emission (index 4)
            const result = [...this._textures]
            result[4] = this._textures[0]  // Use albedo as emission
            return result
        }
        return this._textures
    }

    /**
     * Set textures array directly
     */
    set textures(value) {
        this._textures = value
    }
}

export { Material }
