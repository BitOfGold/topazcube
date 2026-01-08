import { BasePass } from "./BasePass.js"
import { Pipeline } from "../../Pipeline.js"
import { Texture } from "../../Texture.js"
import { Frustum } from "../../utils/Frustum.js"
import { transformBoundingSphere } from "../../utils/BoundingSphere.js"

import geometryWGSL from "../shaders/geometry.wgsl"

/**
 * GBuffer textures container
 */
class GBuffer {
    constructor() {
        this.isGBuffer = true
        this.albedo = null    // rgba8unorm - Base color
        this.normal = null    // rgba16float - World-space normals
        this.arm = null       // rgba8unorm - Ambient Occlusion, Roughness, Metallic
        this.emission = null  // rgba16float - Emissive color
        this.velocity = null  // rg16float - Motion vectors (screen-space pixels)
        this.depth = null     // depth32float - Scene depth
    }

    static async create(engine, width, height) {
        const gbuffer = new GBuffer()
        gbuffer.albedo = await Texture.renderTarget(engine, 'rgba8unorm', width, height)
        gbuffer.normal = await Texture.renderTarget(engine, 'rgba16float', width, height)
        gbuffer.arm = await Texture.renderTarget(engine, 'rgba8unorm', width, height)
        gbuffer.emission = await Texture.renderTarget(engine, 'rgba16float', width, height)
        gbuffer.velocity = await Texture.renderTarget(engine, 'rg16float', width, height)
        gbuffer.depth = await Texture.depth(engine, width, height)
        gbuffer.width = width
        gbuffer.height = height
        return gbuffer
    }

    getTargets() {
        return [
            { format: "rgba8unorm" },
            { format: "rgba16float" },
            { format: "rgba8unorm" },
            { format: "rgba16float" },
            { format: "rg16float" },
        ]
    }

    getColorAttachments() {
        return [
            {
                view: this.albedo.view,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            },
            {
                view: this.normal.view,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            },
            {
                view: this.arm.view,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            },
            {
                view: this.emission.view,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            },
            {
                view: this.velocity.view,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store',
            },
        ]
    }

    getDepthStencilAttachment() {
        return {
            view: this.depth.view,
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        }
    }
}

/**
 * GBufferPass - Renders scene geometry to GBuffer textures
 *
 * Pass 4 in the 7-pass pipeline.
 * Outputs: Albedo, Normal, ARM, Emission, Depth
 */
class GBufferPass extends BasePass {
    constructor(engine = null) {
        super('GBuffer', engine)

        this.gbuffer = null
        this.pipelines = new Map() // materialId -> pipeline (ready)
        this.skinnedPipelines = new Map() // materialId -> skinned pipeline (ready)
        this.pendingPipelines = new Map() // materialId -> Promise<pipeline> (compiling)

        // Clip plane for planar reflections
        this.clipPlaneY = 0
        this.clipPlaneEnabled = false
        this.clipPlaneDirection = 1.0  // 1.0 = discard below, -1.0 = discard above

        // Distance fade for preventing object popping at culling distance
        this.distanceFadeStart = 0  // Distance where fade begins
        this.distanceFadeEnd = 0    // Distance where fade completes (0 = disabled)

        // Noise texture for alpha hashing
        this.noiseTexture = null
        this.noiseSize = 64
        this.noiseAnimated = true

        // HiZ pass reference for occlusion culling of legacy meshes
        this.hizPass = null

        // Frustum for legacy mesh culling
        this.frustum = new Frustum()

        // Billboard camera vectors (extracted from view matrix)
        this._billboardCameraRight = [1, 0, 0]
        this._billboardCameraUp = [0, 1, 0]
        this._billboardCameraForward = [0, 0, -1]

        // Culling stats for legacy meshes
        this.legacyCullingStats = {
            total: 0,
            rendered: 0,
            culledByFrustum: 0,
            culledByDistance: 0,
            culledByOcclusion: 0
        }
    }

    /**
     * Set the HiZ pass for occlusion culling of legacy meshes
     * @param {HiZPass} hizPass - The HiZ pass instance
     */
    setHiZPass(hizPass) {
        this.hizPass = hizPass
    }

    /**
     * Test if a legacy mesh should be culled (per-instance occlusion culling)
     * @param {Mesh} mesh - The mesh to test
     * @param {Camera} camera - Current camera
     * @param {boolean} canCull - Whether frustum/occlusion culling is available
     * @returns {string|null} - Reason for culling or null if visible
     */
    _shouldCullLegacyMesh(mesh, camera, canCull) {
        // Skip culling if disabled or no HiZ pass
        if (!canCull || !this.hizPass || !camera) {
            return null
        }

        // Skip for entity-managed meshes - they're already culled by CullingSystem
        // Only perform per-mesh occlusion culling for static (non-entity-managed) meshes
        if (!mesh.static) {
            return null
        }

        const occlusionEnabled = this.settings?.occlusionCulling?.enabled
        if (!occlusionEnabled) {
            return null
        }

        // Get local bounding sphere from geometry
        const localBsphere = mesh.geometry?.getBoundingSphere?.()
        if (!localBsphere || localBsphere.radius <= 0) {
            this.legacyCullingStats.skippedNoBsphere = (this.legacyCullingStats.skippedNoBsphere || 0) + 1
            return null  // No valid bsphere, don't cull
        }

        const instanceCount = mesh.geometry?.instanceCount || 0
        if (instanceCount === 0) {
            return null  // No instances to test
        }

        const instanceData = mesh.geometry?.instanceData
        if (!instanceData) {
            return null  // No instance data
        }

        // Test each instance - if ANY is visible, mesh is visible
        // Instance data layout: 20 floats per instance (4x4 matrix + 4 extra)
        const floatsPerInstance = 20
        let allOccluded = true

        // Copy camera data to avoid mutable reference issues
        const cameraPos = [camera.position[0], camera.position[1], camera.position[2]]
        const viewProj = camera.viewProj
        const near = camera.near
        const far = camera.far

        for (let i = 0; i < instanceCount; i++) {
            const offset = i * floatsPerInstance

            // Extract 4x4 matrix from instance data
            const matrix = instanceData.subarray(offset, offset + 16)

            // Transform local bsphere by instance matrix
            const worldBsphere = transformBoundingSphere(localBsphere, matrix)

            // Test against HiZ - if visible, mesh is visible
            const occluded = this.hizPass.testSphereOcclusion(
                worldBsphere,
                viewProj,
                near,
                far,
                cameraPos
            )

            if (!occluded) {
                allOccluded = false
                break  // At least one instance visible, no need to test more
            }
        }

        return allOccluded ? 'occlusion' : null
    }

    /**
     * Extract camera vectors from view matrix for billboarding
     * The view matrix transforms world to view space. Camera basis vectors:
     * - Right: first row of view matrix
     * - Up: second row of view matrix
     * - Forward: negative third row (camera looks down -Z in view space)
     * @param {Float32Array|Array} viewMatrix - 4x4 view matrix
     */
    _extractCameraVectors(viewMatrix) {
        // View matrix is column-major, so row vectors are at indices:
        // Row 0 (right): [0], [4], [8]
        // Row 1 (up): [1], [5], [9]
        // Row 2 (forward): [2], [6], [10] (negated because -Z is forward)
        this._billboardCameraRight[0] = viewMatrix[0]
        this._billboardCameraRight[1] = viewMatrix[4]
        this._billboardCameraRight[2] = viewMatrix[8]

        this._billboardCameraUp[0] = viewMatrix[1]
        this._billboardCameraUp[1] = viewMatrix[5]
        this._billboardCameraUp[2] = viewMatrix[9]

        // Negate Z row for forward direction
        this._billboardCameraForward[0] = -viewMatrix[2]
        this._billboardCameraForward[1] = -viewMatrix[6]
        this._billboardCameraForward[2] = -viewMatrix[10]
    }

    /**
     * Get billboard mode from material
     * @param {Material} material - Material with optional billboardMode uniform
     * @returns {number} Billboard mode: 0=none, 1=center, 2=bottom, 3=horizontal
     */
    _getBillboardMode(material) {
        const mode = material?.uniforms?.billboardMode
        if (typeof mode === 'number') return mode
        if (mode === 'center') return 1
        if (mode === 'bottom') return 2
        if (mode === 'horizontal') return 3
        return 0
    }

    /**
     * Set the noise texture for alpha hashing
     * @param {Texture} noise - Noise texture (blue noise or bayer dither)
     * @param {number} size - Texture size
     * @param {boolean} animated - Whether to animate noise offset each frame
     */
    setNoise(noise, size = 64, animated = true) {
        this.noiseTexture = noise
        this.noiseSize = size
        this.noiseAnimated = animated
        // Mark all pipelines for rebuild (they need noise texture binding)
        this.pipelines.clear()
        this.skinnedPipelines.clear()
    }

    async _init() {
        const { canvas } = this.engine
        this.gbuffer = await GBuffer.create(this.engine, canvas.width, canvas.height)
    }

    /**
     * Get pipeline key for a mesh
     */
    _getPipelineKey(mesh) {
        const isSkinned = mesh.hasSkin && mesh.skin
        const meshId = mesh.uid || mesh.geometry?.uid || 'default'
        const forceEmissive = mesh.material?.forceEmissive ? '_emissive' : ''
        return `${mesh.material.uid}_${meshId}${isSkinned ? '_skinned' : ''}${forceEmissive}`
    }

    /**
     * Check if pipeline is ready for a mesh (non-blocking)
     * @param {Mesh} mesh - The mesh to check
     * @returns {Pipeline|null} The pipeline if ready, null if still compiling
     */
    _getPipelineIfReady(mesh) {
        const isSkinned = mesh.hasSkin && mesh.skin
        const pipelinesMap = isSkinned ? this.skinnedPipelines : this.pipelines
        const key = this._getPipelineKey(mesh)
        return pipelinesMap.get(key) || null
    }

    /**
     * Check if pipeline is ready AND warmed up (stable for rendering)
     * @param {Mesh} mesh - The mesh to check
     * @returns {boolean} True if pipeline is ready and warmed up
     */
    isPipelineStable(mesh) {
        const pipeline = this._getPipelineIfReady(mesh)
        return pipeline && (!pipeline._warmupFrames || pipeline._warmupFrames <= 0)
    }

    /**
     * Start pipeline creation in background (non-blocking)
     * @param {Mesh} mesh - The mesh to create pipeline for
     */
    _startPipelineCreation(mesh) {
        const isSkinned = mesh.hasSkin && mesh.skin
        const pipelinesMap = isSkinned ? this.skinnedPipelines : this.pipelines
        const key = this._getPipelineKey(mesh)

        // Already ready or already pending
        if (pipelinesMap.has(key) || this.pendingPipelines.has(key)) {
            return
        }

        // Start async compilation without awaiting
        const pipelinePromise = Pipeline.create(this.engine, {
            label: `gbuffer-${key}`,
            wgslSource: geometryWGSL,
            geometry: mesh.geometry,
            textures: mesh.material.textures,
            renderTarget: this.gbuffer,
            skin: isSkinned ? mesh.skin : null,
            noiseTexture: this.noiseTexture,
        }).then(pipeline => {
            // Move from pending to ready
            this.pendingPipelines.delete(key)
            // Mark as warming up - needs 2 frames to stabilize
            pipeline._warmupFrames = 2
            pipelinesMap.set(key, pipeline)
            return pipeline
        }).catch(err => {
            console.error(`Failed to create pipeline for ${key}:`, err)
            this.pendingPipelines.delete(key)
            return null
        })

        this.pendingPipelines.set(key, pipelinePromise)
    }

    /**
     * Get or create pipeline for a mesh (blocking - for batch system)
     * @param {Mesh} mesh - The mesh to render
     * @returns {Pipeline} The pipeline for this mesh
     */
    async _getOrCreatePipeline(mesh) {
        const isSkinned = mesh.hasSkin && mesh.skin
        const pipelinesMap = isSkinned ? this.skinnedPipelines : this.pipelines
        const key = this._getPipelineKey(mesh)

        // Return if already ready
        if (pipelinesMap.has(key)) {
            return pipelinesMap.get(key)
        }

        // Wait for pending if exists
        if (this.pendingPipelines.has(key)) {
            return await this.pendingPipelines.get(key)
        }

        // Create new pipeline
        const pipeline = await Pipeline.create(this.engine, {
            label: `gbuffer-${key}`,
            wgslSource: geometryWGSL,
            geometry: mesh.geometry,
            textures: mesh.material.textures,
            renderTarget: this.gbuffer,
            skin: isSkinned ? mesh.skin : null,
            noiseTexture: this.noiseTexture,
        })
        // Mark as warming up - needs 2 frames to stabilize
        pipeline._warmupFrames = 2
        pipelinesMap.set(key, pipeline)
        return pipeline
    }

    /**
     * Execute GBuffer pass
     *
     * @param {Object} context
     * @param {Camera} context.camera - Current camera
     * @param {Object} context.meshes - Legacy mesh dictionary (for backward compatibility)
     * @param {Map} context.batches - Instance batches from InstanceManager (new system)
     * @param {number} context.dt - Delta time for animation
     * @param {HistoryBufferManager} context.historyManager - History buffer manager for motion vectors
     */
    async _execute(context) {
        const { device, canvas, options, stats } = this.engine
        const { camera, meshes, batches, dt = 0, historyManager } = context

        // Get previous frame camera matrices for motion vectors
        // If no valid history, use current viewProj (zero motion)
        const prevData = historyManager?.getPrevious()
        const prevViewProjMatrix = prevData?.hasValidHistory
            ? prevData.viewProj
            : camera.viewProj

        // Get settings from engine (with fallbacks)
        const emissionFactor = this.settings?.environment?.emissionFactor ?? [1.0, 1.0, 1.0, 4.0]
        const mipBias = this.settings?.rendering?.mipBias ?? options.mipBias ?? 0

        stats.drawCalls = 0
        stats.triangles = 0

        // Update camera
        camera.aspect = canvas.width / canvas.height
        camera.updateMatrix()
        camera.updateView()

        // Extract camera vectors for billboarding
        this._extractCameraVectors(camera.view)

        let commandEncoder = null
        let passEncoder = null

        // New system: render batches from InstanceManager
        if (batches && batches.size > 0) {
            for (const [modelId, batch] of batches) {
                const mesh = batch.mesh
                if (!mesh) continue

                // Update skin animation if skinned (skip if externally managed)
                if (batch.hasSkin && batch.skin && !batch.skin.externallyManaged) {
                    batch.skin.update(dt)
                }

                const pipeline = await this._getOrCreatePipeline(mesh)

                // On first render after pipeline creation, force skin update to ensure proper state
                if (pipeline._warmupFrames > 0) {
                    pipeline._warmupFrames--
                    if (batch.hasSkin && batch.skin) {
                        // Force immediate skin update to avoid stale joint matrices
                        batch.skin.update(0)
                    }
                }

                // Update bind group if skinned
                if (batch.hasSkin && batch.skin) {
                    pipeline.updateBindGroupForSkin(batch.skin)
                }

                // Update geometry buffers
                mesh.geometry.update()

                // Set uniforms
                const jitterFadeDistance = this.settings?.rendering?.jitterFadeDistance ?? 30.0
                // Get alpha hash settings (per-material or global)
                const alphaHashEnabled = mesh.material?.alphaHash ?? this.settings?.rendering?.alphaHash ?? false
                const alphaHashScale = mesh.material?.alphaHashScale ?? this.settings?.rendering?.alphaHashScale ?? 1.0
                const luminanceToAlpha = mesh.material?.luminanceToAlpha ?? this.settings?.rendering?.luminanceToAlpha ?? false

                pipeline.uniformValues.set({
                    viewMatrix: camera.view,
                    projectionMatrix: camera.proj,
                    prevViewProjMatrix: prevViewProjMatrix,
                    mipBias: mipBias,
                    skinEnabled: batch.hasSkin ? 1.0 : 0.0,
                    numJoints: batch.hasSkin && batch.skin ? batch.skin.numJoints : 0,
                    near: camera.near || 0.05,
                    far: camera.far || 1000,
                    jitterFadeDistance: jitterFadeDistance,
                    jitterOffset: camera.jitterOffset || [0, 0],
                    screenSize: camera.screenSize || [canvas.width, canvas.height],
                    emissionFactor: emissionFactor,
                    clipPlaneY: this.clipPlaneY,
                    clipPlaneEnabled: this.clipPlaneEnabled ? 1.0 : 0.0,
                    clipPlaneDirection: this.clipPlaneDirection,
                    pixelRounding: this.settings?.rendering?.pixelRounding || 0.0,
                    pixelExpansion: this.settings?.rendering?.pixelExpansion ?? 0.05,
                    positionRounding: this.settings?.rendering?.positionRounding || 0.0,
                    alphaHashEnabled: alphaHashEnabled ? 1.0 : 0.0,
                    alphaHashScale: alphaHashScale,
                    luminanceToAlpha: luminanceToAlpha ? 1.0 : 0.0,
                    noiseSize: this.noiseSize,
                    // Always use static noise for alpha hash to avoid shimmer on cutout edges
                    noiseOffsetX: 0,
                    noiseOffsetY: 0,
                    cameraPosition: camera.position,
                    distanceFadeStart: this.distanceFadeStart,
                    distanceFadeEnd: this.distanceFadeEnd,
                    // Billboard uniforms
                    billboardMode: this._getBillboardMode(mesh.material),
                    billboardCameraRight: this._billboardCameraRight,
                    billboardCameraUp: this._billboardCameraUp,
                    billboardCameraForward: this._billboardCameraForward,
                    // Per-material specular boost (0-1, default 0 = disabled)
                    specularBoost: mesh.material?.specularBoost ?? 0,
                })

                // Render
                if (commandEncoder) {
                    pipeline.render({
                        commandEncoder,
                        passEncoder,
                        dontFinish: true,
                        instanceBuffer: batch.buffer?.gpuBuffer,
                        instanceCount: batch.instanceCount
                    })
                } else {
                    const result = pipeline.render({
                        dontFinish: true,
                        instanceBuffer: batch.buffer?.gpuBuffer,
                        instanceCount: batch.instanceCount
                    })
                    commandEncoder = result.commandEncoder
                    passEncoder = result.passEncoder
                }
            }
        }

        // Legacy system: render individual meshes with progressive loading
        // Meshes appear as their shaders compile - no blocking wait
        let totalInstances = 0

        if (meshes && Object.keys(meshes).length > 0) {
            // Update frustum for legacy mesh culling (only if camera has required properties)
            const canCull = camera.view && camera.proj && camera.position && camera.direction
            if (canCull) {
                const fovRadians = (camera.fov || 60) * (Math.PI / 180)
                this.frustum.update(
                    camera.view,
                    camera.proj,
                    camera.position,
                    camera.direction,
                    fovRadians,
                    camera.aspect || (canvas.width / canvas.height),
                    camera.near || 0.05,
                    camera.far || 1000,
                    canvas.width,
                    canvas.height
                )
            }

            // Reset culling stats
            this.legacyCullingStats.total = 0
            this.legacyCullingStats.rendered = 0
            this.legacyCullingStats.culledByFrustum = 0
            this.legacyCullingStats.culledByDistance = 0
            this.legacyCullingStats.culledByOcclusion = 0
            this.legacyCullingStats.skippedNoBsphere = 0

            // Start pipeline creation for ALL meshes (non-blocking)
            // This kicks off parallel shader compilation in the background
            for (const name in meshes) {
                const mesh = meshes[name]
                if (!mesh || !mesh.geometry || !mesh.material) continue
                this._startPipelineCreation(mesh)
            }

            // Render only meshes with READY pipelines (others will appear next frame)
            for (const name in meshes) {
                const mesh = meshes[name]
                const instanceCount = mesh.geometry?.instanceCount || 0
                totalInstances += instanceCount

                // Skip meshes with no instances
                if (instanceCount === 0) continue

                this.legacyCullingStats.total++

                // Apply culling to legacy meshes (frustum, distance, occlusion)
                const cullReason = this._shouldCullLegacyMesh(mesh, camera, canCull)
                if (cullReason) {
                    if (cullReason === 'frustum') this.legacyCullingStats.culledByFrustum++
                    else if (cullReason === 'distance') this.legacyCullingStats.culledByDistance++
                    else if (cullReason === 'occlusion') this.legacyCullingStats.culledByOcclusion++
                    continue
                }

                // Check if pipeline is ready (non-blocking)
                const pipeline = this._getPipelineIfReady(mesh)
                if (!pipeline) continue  // Still compiling, skip for now

                // Track warmup frames (pipeline just became ready)
                if (pipeline._warmupFrames > 0) {
                    pipeline._warmupFrames--
                }

                this.legacyCullingStats.rendered++

                // Update skin animation (skip if externally managed - already updated by RenderGraph)
                if (mesh.skin && mesh.hasSkin && !mesh.skin.externallyManaged) {
                    mesh.skin.update(dt)
                }

                // Ensure pipeline geometry matches mesh geometry
                if (pipeline.geometry !== mesh.geometry) {
                    pipeline.geometry = mesh.geometry
                }

                // Update bind group if skinned
                if (mesh.hasSkin && mesh.skin) {
                    pipeline.updateBindGroupForSkin(mesh.skin)
                }

                // Update geometry buffers
                mesh.geometry.update()

                // Set uniforms
                const jitterFadeDist = this.settings?.rendering?.jitterFadeDistance ?? 30.0
                // Get alpha hash settings - check mesh material first, then global setting
                const meshAlphaHash = mesh.material?.alphaHash ?? mesh.alphaHash
                const alphaHashEnabled = meshAlphaHash ?? this.settings?.rendering?.alphaHash ?? false
                const alphaHashScale = mesh.material?.alphaHashScale ?? this.settings?.rendering?.alphaHashScale ?? 1.0
                const luminanceToAlpha = mesh.material?.luminanceToAlpha ?? this.settings?.rendering?.luminanceToAlpha ?? false

                pipeline.uniformValues.set({
                    viewMatrix: camera.view,
                    projectionMatrix: camera.proj,
                    prevViewProjMatrix: prevViewProjMatrix,
                    mipBias: mipBias,
                    skinEnabled: mesh.hasSkin ? 1.0 : 0.0,
                    numJoints: mesh.hasSkin && mesh.skin ? mesh.skin.numJoints : 0,
                    near: camera.near || 0.05,
                    far: camera.far || 1000,
                    jitterFadeDistance: jitterFadeDist,
                    jitterOffset: camera.jitterOffset || [0, 0],
                    screenSize: camera.screenSize || [canvas.width, canvas.height],
                    emissionFactor: emissionFactor,
                    clipPlaneY: this.clipPlaneY,
                    clipPlaneEnabled: this.clipPlaneEnabled ? 1.0 : 0.0,
                    clipPlaneDirection: this.clipPlaneDirection,
                    pixelRounding: this.settings?.rendering?.pixelRounding || 0.0,
                    pixelExpansion: this.settings?.rendering?.pixelExpansion ?? 0.05,
                    positionRounding: this.settings?.rendering?.positionRounding || 0.0,
                    alphaHashEnabled: alphaHashEnabled ? 1.0 : 0.0,
                    alphaHashScale: alphaHashScale,
                    luminanceToAlpha: luminanceToAlpha ? 1.0 : 0.0,
                    noiseSize: this.noiseSize,
                    // Always use static noise for alpha hash to avoid shimmer on cutout edges
                    noiseOffsetX: 0,
                    noiseOffsetY: 0,
                    cameraPosition: camera.position,
                    distanceFadeStart: this.distanceFadeStart,
                    distanceFadeEnd: this.distanceFadeEnd,
                    // Billboard uniforms
                    billboardMode: this._getBillboardMode(mesh.material),
                    billboardCameraRight: this._billboardCameraRight,
                    billboardCameraUp: this._billboardCameraUp,
                    billboardCameraForward: this._billboardCameraForward,
                    // Per-material specular boost (0-1, default 0 = disabled)
                    specularBoost: mesh.material?.specularBoost ?? 0,
                })

                // Render
                if (commandEncoder) {
                    pipeline.render({ commandEncoder, passEncoder, dontFinish: true })
                } else {
                    const result = pipeline.render({ dontFinish: true })
                    commandEncoder = result.commandEncoder
                    passEncoder = result.passEncoder
                }
            }
        }

        // Finish the pass
        if (passEncoder && commandEncoder) {
            passEncoder.end()
            device.queue.submit([commandEncoder.finish()])
        }
    }

    async _resize(width, height) {
        // Recreate GBuffer at new size
        this.gbuffer = await GBuffer.create(this.engine, width, height)

        // Clear pipeline caches (they reference old GBuffer)
        this.pipelines.clear()
        this.skinnedPipelines.clear()
    }

    _destroy() {
        this.pipelines.clear()
        this.skinnedPipelines.clear()
        this.gbuffer = null
    }

    /**
     * Get the GBuffer for use by subsequent passes
     */
    getGBuffer() {
        return this.gbuffer
    }
}

export { GBuffer, GBufferPass }
