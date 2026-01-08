/**
 * BasePass - Abstract base class for render passes
 *
 * All render passes in the 7-pass pipeline inherit from this.
 */
class BasePass {
    constructor(name, engine = null) {
        this.name = name
        this.engine = engine
        this.enabled = true
        this._initialized = false
    }

    // Convenience getter for settings (with fallback for passes without engine)
    get settings() {
        return this.engine?.settings
    }

    /**
     * Initialize the pass (create pipelines, textures, etc.)
     * Called once before first use.
     * @returns {Promise<void>}
     */
    async initialize() {
        if (this._initialized) return
        await this._init()
        this._initialized = true
    }

    /**
     * Override in subclass to perform initialization
     * @protected
     */
    async _init() {
        // Override in subclass
    }

    /**
     * Execute the render pass
     * @param {Object} context - Render context with camera, entities, etc.
     * @returns {Promise<void>}
     */
    async execute(context) {
        if (!this.enabled) return
        if (!this._initialized) {
            await this.initialize()
        }
        await this._execute(context)
    }

    /**
     * Override in subclass to perform rendering
     * @param {Object} context - Render context
     * @protected
     */
    async _execute(context) {
        // Override in subclass
    }

    /**
     * Resize pass resources (called on window resize)
     * @param {number} width - New width
     * @param {number} height - New height
     */
    async resize(width, height) {
        await this._resize(width, height)
    }

    /**
     * Override in subclass to handle resize
     * @protected
     */
    async _resize(width, height) {
        // Override in subclass
    }

    /**
     * Destroy pass resources
     */
    destroy() {
        this._destroy()
        this._initialized = false
    }

    /**
     * Override in subclass to clean up resources
     * @protected
     */
    _destroy() {
        // Override in subclass
    }

    /**
     * Get debug name for profiling
     */
    getDebugName() {
        return `Pass: ${this.name}`
    }
}

export { BasePass }
