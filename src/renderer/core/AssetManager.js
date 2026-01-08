import { loadGltfData, loadGltf } from "../gltf.js"
import { Geometry } from "../Geometry.js"
import { Material } from "../Material.js"
import { Mesh } from "../Mesh.js"
import { Texture } from "../Texture.js"
import { calculateBoundingSphere } from "../utils/BoundingSphere.js"

/**
 * AssetManager - Lazy loading and caching of assets
 *
 * Asset keys:
 * - "path/to/model.glb" - Raw GLTF data (gltf object, ready state)
 * - "path/to/model.glb|meshName" - Processed mesh (geometry, material, skin, bsphere)
 *
 * Assets structure:
 * {
 *   "models/fox.glb": { gltf: {...}, meshNames: [...], ready: true, loading: false }
 *   "models/fox.glb|fox1": { geometry, material, skin, bsphere, ready: true }
 * }
 */
class AssetManager {
    constructor(engine = null) {
        this.engine = engine

        // Asset storage
        this.assets = {}

        // Loading promises for deduplication
        this._loadingPromises = {}

        // Callbacks for when assets become ready
        this._readyCallbacks = {}
    }

    /**
     * Parse a ModelID into path and mesh name
     * @param {string} modelId - Format: "path/to/model.glb|meshName"
     * @returns {{ path: string, meshName: string|null }}
     */
    parseModelId(modelId) {
        const parts = modelId.split("|")
        return {
            path: parts[0],
            meshName: parts[1] || null
        }
    }

    /**
     * Create a ModelID from path and mesh name
     */
    createModelId(path, meshName) {
        return `${path}|${meshName}`
    }

    /**
     * Check if an asset exists and is ready
     */
    isReady(assetKey) {
        return this.assets[assetKey]?.ready === true
    }

    /**
     * Check if an asset is currently loading
     */
    isLoading(assetKey) {
        return this.assets[assetKey]?.loading === true || this._loadingPromises[assetKey] !== undefined
    }

    /**
     * Get an asset if ready, otherwise return null
     */
    get(assetKey) {
        const asset = this.assets[assetKey]
        if (asset?.ready) {
            return asset
        }
        return null
    }

    /**
     * Get or load a GLTF file
     * @param {string} path - Path to the GLTF file
     * @param {Object} options - Loading options
     * @returns {Promise<Object>} The loaded GLTF asset
     */
    async loadGltfFile(path, options = {}) {
        // Check if already loaded
        if (this.assets[path]?.ready) {
            return this.assets[path]
        }

        // Check if already loading (deduplicate)
        if (this._loadingPromises[path]) {
            return this._loadingPromises[path]
        }

        // Mark as loading
        this.assets[path] = { ready: false, loading: true }

        // Create loading promise
        this._loadingPromises[path] = (async () => {
            try {
                const result = await loadGltf(this.engine, path, options)

                // Store the full result
                const meshNames = Object.keys(result.meshes)
                this.assets[path] = {
                    gltf: result,
                    meshes: result.meshes,
                    skins: result.skins,
                    animations: result.animations,
                    nodes: result.nodes,
                    meshNames: meshNames,
                    ready: true,
                    loading: false
                }

                // Auto-register individual meshes
                for (const meshName of meshNames) {
                    const modelId = this.createModelId(path, meshName)
                    await this._registerMesh(path, meshName, result.meshes[meshName])
                }

                // Trigger ready callbacks
                this._triggerReady(path)

                return this.assets[path]
            } catch (error) {
                console.error(`Failed to load GLTF: ${path}`, error)
                this.assets[path] = { ready: false, loading: false, error: error.message }
                throw error
            } finally {
                delete this._loadingPromises[path]
            }
        })()

        return this._loadingPromises[path]
    }

    /**
     * Register a mesh asset (internal)
     */
    async _registerMesh(path, meshName, mesh) {
        const modelId = this.createModelId(path, meshName)

        // Calculate bounding sphere from geometry
        const bsphere = calculateBoundingSphere(mesh.geometry.attributes.position)

        this.assets[modelId] = {
            mesh: mesh,
            geometry: mesh.geometry,
            material: mesh.material,
            skin: mesh.skin || null,
            hasSkin: mesh.hasSkin || false,
            bsphere: bsphere,
            ready: true,
            loading: false
        }

        // Trigger ready callbacks
        this._triggerReady(modelId)
    }

    /**
     * Get or load a specific mesh from a GLTF file
     * @param {string} modelId - Format: "path/to/model.glb|meshName"
     * @param {Object} options - Loading options
     * @returns {Promise<Object>} The mesh asset
     */
    async loadMesh(modelId, options = {}) {
        const { path, meshName } = this.parseModelId(modelId)

        // If mesh is already loaded, return it
        if (this.assets[modelId]?.ready) {
            return this.assets[modelId]
        }

        // Load the GLTF file first (will auto-register meshes)
        await this.loadGltfFile(path, options)

        // Now get the mesh
        const meshAsset = this.assets[modelId]
        if (!meshAsset) {
            throw new Error(`Mesh "${meshName}" not found in "${path}"`)
        }

        return meshAsset
    }

    /**
     * Preload multiple assets
     * @param {string[]} assetKeys - List of asset keys to preload
     * @param {Object} options - Loading options
     * @returns {Promise<void>}
     */
    async preload(assetKeys, options = {}) {
        const promises = assetKeys.map(key => {
            const { path, meshName } = this.parseModelId(key)
            if (meshName) {
                return this.loadMesh(key, options)
            } else {
                return this.loadGltfFile(key, options)
            }
        })

        await Promise.all(promises)
    }

    /**
     * Register a callback for when an asset becomes ready
     */
    onReady(assetKey, callback) {
        // If already ready, call immediately
        if (this.assets[assetKey]?.ready) {
            callback(this.assets[assetKey])
            return
        }

        // Register callback
        if (!this._readyCallbacks[assetKey]) {
            this._readyCallbacks[assetKey] = []
        }
        this._readyCallbacks[assetKey].push(callback)
    }

    /**
     * Trigger ready callbacks
     */
    _triggerReady(assetKey) {
        const callbacks = this._readyCallbacks[assetKey]
        if (callbacks) {
            for (const callback of callbacks) {
                callback(this.assets[assetKey])
            }
            delete this._readyCallbacks[assetKey]
        }
    }

    /**
     * Get all loaded mesh names for a GLTF file
     */
    getMeshNames(path) {
        const asset = this.assets[path]
        return asset?.meshNames || []
    }

    /**
     * Get all unique GLTF paths currently loaded
     */
    getLoadedPaths() {
        return Object.keys(this.assets).filter(key => !key.includes("|") && this.assets[key].ready)
    }

    /**
     * Get all ModelIDs currently loaded
     */
    getLoadedModelIds() {
        return Object.keys(this.assets).filter(key => key.includes("|") && this.assets[key].ready)
    }

    /**
     * Get bounding sphere for a model
     */
    getBoundingSphere(modelId) {
        const asset = this.assets[modelId]
        return asset?.bsphere || null
    }

    /**
     * Create a clone of a mesh (shares geometry/material, but separate instance buffers)
     */
    cloneMesh(modelId) {
        const asset = this.assets[modelId]
        if (!asset?.ready) {
            return null
        }

        // Create a new Mesh with the same geometry and material
        const clone = new Mesh(asset.geometry, asset.material)
        if (asset.skin) {
            clone.skin = asset.skin
            clone.hasSkin = true
        }
        return clone
    }

    /**
     * Unload an asset to free memory
     */
    unload(assetKey) {
        const asset = this.assets[assetKey]
        if (!asset) return

        // If this is a GLTF file, also unload all its meshes
        if (asset.meshNames) {
            for (const meshName of asset.meshNames) {
                const modelId = this.createModelId(assetKey, meshName)
                delete this.assets[modelId]
            }
        }

        delete this.assets[assetKey]
    }

    /**
     * Clear all assets
     */
    clear() {
        this.assets = {}
        this._loadingPromises = {}
        this._readyCallbacks = {}
    }

    /**
     * Get loading status for all assets
     */
    getStatus() {
        const ready = []
        const loading = []
        const failed = []

        for (const key in this.assets) {
            const asset = this.assets[key]
            if (asset.ready) {
                ready.push(key)
            } else if (asset.loading) {
                loading.push(key)
            } else if (asset.error) {
                failed.push({ key, error: asset.error })
            }
        }

        return { ready, loading, failed }
    }

    /**
     * Register a manually created mesh (for procedural geometry)
     */
    registerMesh(modelId, mesh, bsphere = null) {
        if (!bsphere) {
            bsphere = calculateBoundingSphere(mesh.geometry.attributes.position)
        }

        this.assets[modelId] = {
            mesh: mesh,
            geometry: mesh.geometry,
            material: mesh.material,
            skin: mesh.skin || null,
            hasSkin: mesh.hasSkin || false,
            bsphere: bsphere,
            ready: true,
            loading: false
        }

        this._triggerReady(modelId)
    }
}

export { AssetManager }
