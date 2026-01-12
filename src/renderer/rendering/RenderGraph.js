import { ShadowPass } from "./passes/ShadowPass.js"
import { ReflectionPass } from "./passes/ReflectionPass.js"
import { PlanarReflectionPass } from "./passes/PlanarReflectionPass.js"
import { GBufferPass } from "./passes/GBufferPass.js"
import { HiZPass } from "./passes/HiZPass.js"
import { AOPass } from "./passes/AOPass.js"
import { LightingPass } from "./passes/LightingPass.js"
import { SSGITilePass } from "./passes/SSGITilePass.js"
import { SSGIPass } from "./passes/SSGIPass.js"
import { RenderPostPass } from "./passes/RenderPostPass.js"
import { BloomPass } from "./passes/BloomPass.js"
import { TransparentPass } from "./passes/TransparentPass.js"
import { ParticlePass } from "./passes/ParticlePass.js"
import { FogPass } from "./passes/FogPass.js"
import { VolumetricFogPass } from "./passes/VolumetricFogPass.js"
import { PostProcessPass } from "./passes/PostProcessPass.js"
import { CRTPass } from "./passes/CRTPass.js"
import { AmbientCapturePass } from "./passes/AmbientCapturePass.js"
import { HistoryBufferManager } from "./HistoryBufferManager.js"
import { CullingSystem } from "../core/CullingSystem.js"
import { InstanceManager } from "../core/InstanceManager.js"
import { SpriteSystem } from "../core/SpriteSystem.js"
import { ParticleSystem } from "../core/ParticleSystem.js"
import { transformBoundingSphere, calculateShadowBoundingSphere, sphereInCascade } from "../utils/BoundingSphere.js"
import { vec3, mat4 } from "../math.js"
import { Texture } from "../Texture.js"
import { Geometry } from "../Geometry.js"

/**
 * RenderGraph - Orchestrates the multi-pass rendering pipeline
 *
 * Manages pass execution order, resource dependencies, and integrates
 * with the entity/asset system for data-oriented rendering.
 *
 * Pipeline order:
 * 1. Shadow Pass (CSM + spotlight)
 * 2. Reflection Pass (octahedral probes)
 * 3. Planar Reflection Pass (mirrored camera for water/floors)
 * 4. GBuffer Pass (geometry -> albedo, normal, ARM, emission, velocity, depth)
 * 4b. HiZ Pass (hierarchical-Z for occlusion culling)
 * 5. AO Pass (SSAO)
 * 6. Lighting Pass (deferred lighting)
 * 7. Bloom Pass (HDR bright extraction + blur - moved before SSGI)
 * 8. SSGITile Pass (compute - tile light accumulation)
 * 9. SSGI Pass (screen-space global illumination)
 * 9b. Ambient Capture Pass (6-directional sky-aware ambient)
 * 10. RenderPost Pass (combine SSGI/Planar/Ambient with lighting)
 * 11. Transparent Pass (forward rendering for alpha-blended)
 * 12. Bloom Pass (applied to transparent highlights)
 * 13. PostProcess Pass (bloom composite + tone mapping -> canvas)
 */
class RenderGraph {
    constructor(engine = null) {
        // Reference to engine for settings access
        this.engine = engine

        // Passes (in execution order)
        this.passes = {
            shadow: null,              // Pass 1: Shadow maps
            reflection: null,          // Pass 2: Reflection probes
            planarReflection: null,    // Pass 3: Planar reflection (mirrored camera)
            gbuffer: null,             // Pass 4: GBuffer generation
            hiz: null,                 // Pass 4b: HiZ reduction (for next frame's occlusion culling)
            ao: null,                  // Pass 5: SSAO
            lighting: null,            // Pass 6: Deferred lighting
            bloom: null,               // Pass 7: HDR bloom/glare (moved before SSGI)
            ssgiTile: null,            // Pass 8: SSGI tile accumulation (compute)
            ssgi: null,                // Pass 9: Screen-space global illumination
            ambientCapture: null,      // Pass 9b: 6-directional ambient capture for sky-aware GI
            renderPost: null,          // Pass 10: Combine SSGI/Planar with lighting
            transparent: null,         // Pass 11: Forward transparent objects
            particles: null,           // Pass 12: GPU particle rendering
            postProcess: null,         // Pass 13: Tone mapping + bloom composite
            crt: null,                 // Pass 14: CRT effect (optional)
        }

        // History buffer manager for temporal effects
        this.historyManager = null

        // Support systems (pass engine reference)
        this.cullingSystem = new CullingSystem(engine)
        this.instanceManager = new InstanceManager(engine)
        this.spriteSystem = new SpriteSystem(engine)
        this.particleSystem = new ParticleSystem(engine)

        // Environment map
        this.environmentMap = null

        // Noise texture for dithering/jittering (can be blue noise or bayer)
        this.noiseTexture = null
        this.noiseSize = 64  // Will be updated when texture loads
        this.noiseAnimated = true  // Whether to animate noise offset each frame

        // Effect scaling for expensive passes (bloom, AO, SSGI, planar reflection)
        // When autoScale.enabledForEffects is true and height > maxHeight, effects render at reduced resolution
        this.effectWidth = 0
        this.effectHeight = 0
        this.effectScale = 1.0

        // Cache for cloned skins per phase group: "modelId|animation|phase" -> { skin, mesh }
        this._skinnedPhaseCache = new Map()

        // Cache for individual skins per entity: entityId -> { skin, mesh, geometry }
        this._individualSkinCache = new Map()

        // Debug/stats
        this.stats = {
            passTimings: {},
            visibleEntities: 0,
            culledEntities: 0,
            drawCalls: 0,
            triangles: 0
        }

        // Last render context for probe capture
        this._lastRenderContext = null

        // Probe-specific passes (256x256 for probe face capture)
        this.probePasses = {
            gbuffer: null,
            lighting: null
        }
    }

    // Convenience getter for individualRenderDistance from settings
    get individualRenderDistance() {
        return this.engine?.settings?.skinning?.individualRenderDistance ?? 20.0
    }

    /**
     * Create and initialize the render graph
     * @param {Engine} engine - Engine instance for settings access
     * @param {Texture} environmentMap - HDR environment map for IBL
     * @param {number} encoding - 0 = equirectangular, 1 = octahedral
     * @returns {Promise<RenderGraph>}
     */
    static async create(engine, environmentMap, encoding = 0) {
        const graph = new RenderGraph(engine)
        await graph.initialize(environmentMap, encoding)
        return graph
    }

    /**
     * Initialize all passes
     * @param {Texture} environmentMap - HDR environment map
     * @param {number} encoding - 0 = equirectangular, 1 = octahedral
     */
    async initialize(environmentMap, encoding = 0) {
        const timings = []
        const startTotal = performance.now()

        this.environmentMap = environmentMap
        this.environmentEncoding = encoding

        // Load noise texture based on settings
        let start = performance.now()
        await this._loadNoiseTexture()
        timings.push({ name: 'loadNoiseTexture', time: performance.now() - start })

        // Create passes (pass engine reference)
        this.passes.shadow = new ShadowPass(this.engine)
        this.passes.reflection = new ReflectionPass(this.engine)
        this.passes.planarReflection = new PlanarReflectionPass(this.engine)
        this.passes.gbuffer = new GBufferPass(this.engine)
        this.passes.hiz = new HiZPass(this.engine)
        this.passes.ao = new AOPass(this.engine)
        this.passes.lighting = new LightingPass(this.engine)
        this.passes.bloom = new BloomPass(this.engine)
        this.passes.ssgiTile = new SSGITilePass(this.engine)
        this.passes.ssgi = new SSGIPass(this.engine)
        this.passes.ambientCapture = new AmbientCapturePass(this.engine)
        this.passes.renderPost = new RenderPostPass(this.engine)
        this.passes.transparent = new TransparentPass(this.engine)
        this.passes.particles = new ParticlePass(this.engine)
        this.passes.fog = new FogPass(this.engine)
        this.passes.volumetricFog = new VolumetricFogPass(this.engine)
        this.passes.postProcess = new PostProcessPass(this.engine)
        this.passes.crt = new CRTPass(this.engine)

        // Create history buffer manager for temporal effects
        const { canvas } = this.engine
        this.historyManager = new HistoryBufferManager(this.engine)
        start = performance.now()
        await this.historyManager.initialize(canvas.width, canvas.height)
        timings.push({ name: 'init:historyManager', time: performance.now() - start })

        // Initialize passes
        start = performance.now()
        await this.passes.shadow.initialize()
        timings.push({ name: 'init:shadow', time: performance.now() - start })

        start = performance.now()
        await this.passes.reflection.initialize()
        timings.push({ name: 'init:reflection', time: performance.now() - start })

        start = performance.now()
        await this.passes.planarReflection.initialize()
        timings.push({ name: 'init:planarReflection', time: performance.now() - start })

        start = performance.now()
        await this.passes.gbuffer.initialize()
        timings.push({ name: 'init:gbuffer', time: performance.now() - start })

        start = performance.now()
        await this.passes.hiz.initialize()
        timings.push({ name: 'init:hiz', time: performance.now() - start })

        start = performance.now()
        await this.passes.ao.initialize()
        timings.push({ name: 'init:ao', time: performance.now() - start })

        start = performance.now()
        await this.passes.lighting.initialize()
        timings.push({ name: 'init:lighting', time: performance.now() - start })

        start = performance.now()
        await this.passes.bloom.initialize()
        timings.push({ name: 'init:bloom', time: performance.now() - start })

        start = performance.now()
        await this.passes.ssgiTile.initialize()
        timings.push({ name: 'init:ssgiTile', time: performance.now() - start })

        start = performance.now()
        await this.passes.ssgi.initialize()
        timings.push({ name: 'init:ssgi', time: performance.now() - start })

        start = performance.now()
        await this.passes.renderPost.initialize()
        timings.push({ name: 'init:renderPost', time: performance.now() - start })

        start = performance.now()
        await this.passes.ambientCapture.initialize()
        timings.push({ name: 'init:ambientCapture', time: performance.now() - start })

        start = performance.now()
        await this.passes.transparent.initialize()
        timings.push({ name: 'init:transparent', time: performance.now() - start })

        start = performance.now()
        await this.passes.particles.initialize()
        timings.push({ name: 'init:particles', time: performance.now() - start })

        start = performance.now()
        await this.passes.fog.initialize()
        timings.push({ name: 'init:fog', time: performance.now() - start })

        start = performance.now()
        await this.passes.volumetricFog.initialize()
        timings.push({ name: 'init:volumetricFog', time: performance.now() - start })

        start = performance.now()
        await this.passes.postProcess.initialize()
        timings.push({ name: 'init:postProcess', time: performance.now() - start })

        start = performance.now()
        await this.passes.crt.initialize()
        timings.push({ name: 'init:crt', time: performance.now() - start })

        // Wire up dependencies
        start = performance.now()
        this.passes.reflection.setFallbackEnvironment(environmentMap, this.environmentEncoding)
        this.passes.lighting.setEnvironmentMap(environmentMap, this.environmentEncoding)
        await this.passes.lighting.setGBuffer(this.passes.gbuffer.getGBuffer())
        this.passes.lighting.setShadowPass(this.passes.shadow)
        this.passes.lighting.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        timings.push({ name: 'wire:lighting', time: performance.now() - start })

        // Wire up planar reflection pass (shares shadows, environment with main)
        start = performance.now()
        this.passes.planarReflection.setDependencies({
            environmentMap,
            encoding: this.environmentEncoding,
            shadowPass: this.passes.shadow,
            lightingPass: this.passes.lighting,
            noise: this.noiseTexture,
            noiseSize: this.noiseSize
        })
        this.passes.planarReflection.setParticleSystem(this.particleSystem)
        timings.push({ name: 'wire:planarReflection', time: performance.now() - start })

        // Wire up ambient capture pass (shares shadows, environment with main)
        start = performance.now()
        this.passes.ambientCapture.setDependencies({
            environmentMap,
            encoding: this.environmentEncoding,
            shadowPass: this.passes.shadow,
            noise: this.noiseTexture,
            noiseSize: this.noiseSize
        })
        // Wire ambient capture output to RenderPost
        this.passes.renderPost.setAmbientCaptureBuffer(this.passes.ambientCapture.getFaceColorsBuffer())
        timings.push({ name: 'wire:ambientCapture', time: performance.now() - start })

        // Initialize probe-specific passes at 256x256 (for probe face capture)
        start = performance.now()
        await this._initProbePasses()
        timings.push({ name: 'init:probePasses', time: performance.now() - start })

        // Set up probe capture to use the renderer
        const probeCapture = this.passes.reflection.getProbeCapture()
        if (probeCapture) {
            probeCapture.setSceneRenderCallback(this._renderSceneForProbe.bind(this))
        }

        // Set up GBuffer pass with noise texture for alpha hashing
        start = performance.now()
        this.passes.gbuffer.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        timings.push({ name: 'wire:gbuffer', time: performance.now() - start })

        // Wire up HiZ pass with GBuffer depth and CullingSystem
        start = performance.now()
        this.passes.hiz.setDepthTexture(this.passes.gbuffer.getGBuffer()?.depth)
        this.cullingSystem.setHiZPass(this.passes.hiz)
        // Also wire HiZ to passes that use it for occlusion culling
        this.passes.gbuffer.setHiZPass(this.passes.hiz)
        this.passes.lighting.setHiZPass(this.passes.hiz)
        this.passes.transparent.setHiZPass(this.passes.hiz)
        this.passes.shadow.setHiZPass(this.passes.hiz)
        timings.push({ name: 'wire:hiz', time: performance.now() - start })

        // Set up Shadow pass with noise texture for alpha hashing
        start = performance.now()
        this.passes.shadow.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        timings.push({ name: 'wire:shadow', time: performance.now() - start })

        // Set up AO pass
        start = performance.now()
        await this.passes.ao.setGBuffer(this.passes.gbuffer.getGBuffer())
        this.passes.ao.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        timings.push({ name: 'wire:ao', time: performance.now() - start })

        // Pass AO texture to lighting
        this.passes.lighting.setAOTexture(this.passes.ao.getOutputTexture())

        // Set up RenderPost pass with blue noise
        this.passes.renderPost.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)

        // SSGI passes are wired dynamically per frame (prev HDR, emissive, propagate buffer)

        // Wire up transparent pass (forward rendering for alpha-blended materials)
        this.passes.transparent.setGBuffer(this.passes.gbuffer.getGBuffer())
        this.passes.transparent.setShadowPass(this.passes.shadow)
        this.passes.transparent.setEnvironmentMap(environmentMap, this.environmentEncoding)
        this.passes.transparent.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)

        // Wire up particle pass (GPU particle system rendering)
        this.passes.particles.setParticleSystem(this.particleSystem)
        this.passes.particles.setGBuffer(this.passes.gbuffer.getGBuffer())
        this.passes.particles.setShadowPass(this.passes.shadow)
        this.passes.particles.setEnvironmentMap(environmentMap, this.environmentEncoding)
        this.passes.particles.setLightingPass(this.passes.lighting)

        // Wire up volumetric fog pass
        this.passes.volumetricFog.setGBuffer(this.passes.gbuffer.getGBuffer())
        this.passes.volumetricFog.setShadowPass(this.passes.shadow)
        this.passes.volumetricFog.setLightingPass(this.passes.lighting)
        this.passes.volumetricFog.setHiZPass(this.passes.hiz)

        this.passes.postProcess.setInputTexture(this.passes.lighting.getOutputTexture())
        this.passes.postProcess.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)

        // Wire up GUI canvas for overlay rendering
        if (this.engine?.guiCanvas) {
            this.passes.postProcess.setGuiCanvas(this.engine.guiCanvas)
        }

        // Invalidate occlusion culling after initialization to ensure warmup starts fresh
        // This prevents stale depth data from causing incorrect culling on first frames
        this.invalidateOcclusionCulling()
    }

    /**
     * Render a frame using the new entity/asset system
     *
     * @param {Object} context
     * @param {EntityManager} context.entityManager - Entity manager
     * @param {AssetManager} context.assetManager - Asset manager
     * @param {Camera} context.camera - Current camera
     * @param {Object} context.meshes - Legacy meshes (optional, for hybrid rendering)
     * @param {number} context.dt - Delta time
     */
    async renderEntities(context) {
        // Skip main render while probe capture is in progress
        // The main render modifies shared mesh instance counts which corrupts probe data
        if (this._isCapturingProbe) {
            return
        }

        const { entityManager, assetManager, camera, meshes, dt = 0 } = context
        const { canvas, stats } = this.engine

        // Register sprite entities for animation tracking (before update)
        entityManager.forEach((id, entity) => {
            if (entity.sprite && !entity._spriteRegistered) {
                this.spriteSystem.registerEntity(id, entity)
                entity._spriteRegistered = true
            }
        })

        // Update sprite animations
        this.spriteSystem.update(dt)

        // Process entities with particle emitters
        const particleEntities = entityManager.getParticles()
        for (const { id, entity } of particleEntities) {
            // Register emitter if not already registered
            if (entity.particles && !entity._emitterUID) {
                const emitterConfig = typeof entity.particles === 'object'
                    ? { ...entity.particles, position: entity.position }
                    : { position: entity.position }
                const emitter = this.particleSystem.addEmitter(emitterConfig)
                entity._emitterUID = emitter.uid
            }
            // Update emitter position from entity position
            if (entity._emitterUID) {
                const emitter = this.particleSystem.getEmitter(entity._emitterUID)
                if (emitter) {
                    emitter.position = [...entity.position]
                }
            }
        }

        // Ensure camera matrices are up to date
        camera.aspect = canvas.width / canvas.height
        camera.screenSize[0] = canvas.width
        camera.screenSize[1] = canvas.height
        camera.jitterEnabled = false  // Disable for shadow pass first
        camera.updateMatrix()
        camera.updateView()

        // Update frustum with screen dimensions for pixel size culling
        this.cullingSystem.updateFrustum(camera, canvas.width, canvas.height)

        // Prepare HiZ for occlusion tests (check camera movement, invalidate if needed)
        const hizPass = this.passes.hiz
        if (hizPass) {
            hizPass.prepareForOcclusionTests(camera)
        }

        // Cull entities
        const { visible, skinnedCount } = this.cullingSystem.cull(
            entityManager,
            assetManager,
            'main'
        )

        this.stats.visibleEntities = visible.length
        this.stats.culledEntities = entityManager.count - visible.length

        // Group by model for instancing
        const groups = this.cullingSystem.groupByModel(visible)

        // Build instance batches
        const batches = this.instanceManager.buildBatches(groups, assetManager)

        // For shadow pass, we need entities within shadow range, not just camera-visible ones
        // Apply pixel size culling and distance culling based on cascade coverage
        // NEW: Use shadow bounding spheres for frustum/occlusion culling (only for main light)
        const allEntities = []
        const shadowConfig = this.cullingSystem.config.shadow

        // Get max shadow distance from culling.shadow.maxDistance setting
        const maxShadowDistance = shadowConfig?.maxDistance ?? 100

        // Check if main light is enabled - shadow bounding sphere culling only applies to main light
        const mainLight = this.engine?.settings?.mainLight
        const mainLightEnabled = mainLight?.enabled !== false

        // Get light direction for shadow bounding sphere calculation (only used when main light enabled)
        const lightDir = vec3.fromValues(
            mainLight?.direction?.[0] ?? -1,
            mainLight?.direction?.[1] ?? 1,
            mainLight?.direction?.[2] ?? -0.5
        )
        vec3.normalize(lightDir, lightDir)

        // Ground level for shadow projection (default 0, or from planarReflection settings)
        const groundLevel = this.engine?.settings?.planarReflection?.groundLevel ?? 0

        // Shadow culling settings - only apply shadow bounding sphere culling when main light is enabled
        // Spotlights have their own frustum culling in ShadowPass
        const shadowCullingEnabled = mainLightEnabled && shadowConfig?.frustum !== false
        const shadowHiZEnabled = mainLightEnabled && shadowConfig?.hiZ !== false && this.passes.hiz

        // Track shadow culling stats
        let shadowFrustumCulled = 0
        let shadowHiZCulled = 0
        let shadowDistanceCulled = 0
        let shadowPixelCulled = 0

        // Collect sprite-only entities (entities with .sprite but no .model)
        const spriteOnlyEntities = []
        entityManager.forEach((id, entity) => {
            if (!entity._visible) return
            if (entity.sprite && !entity.model) {
                // Calculate bounding sphere for sprite entity based on scale
                const scale = entity.scale || [1, 1, 1]
                const radius = Math.max(scale[0], scale[1]) * 0.5
                entity._bsphere = {
                    center: [...entity.position],
                    radius: radius
                }
                spriteOnlyEntities.push({ id, entity })
            }
        })

        entityManager.forEach((id, entity) => {
            // Skip invisible entities (same check as main cull)
            if (!entity._visible) return

            if (entity.model) {
                // Update bsphere from asset for shadow culling
                // Note: For skinned models, bsphere is pre-computed as combined sphere of all submeshes
                const asset = assetManager.get(entity.model)
                if (asset?.bsphere) {
                    entity._bsphere = transformBoundingSphere(asset.bsphere, entity._matrix)
                }

                if (entity._bsphere) {
                    // Calculate shadow bounding sphere only when main light is enabled
                    // For spotlights, we use the object's regular bsphere (spotlight culling is in ShadowPass)
                    if (mainLightEnabled) {
                        entity._shadowBsphere = calculateShadowBoundingSphere(
                            entity._bsphere,
                            lightDir,
                            groundLevel
                        )
                    } else {
                        // When main light is off, use regular bsphere for distance/pixel culling
                        entity._shadowBsphere = entity._bsphere
                    }

                    // Use shadow bounding sphere for distance culling
                    // This ensures objects whose shadows are visible are included
                    const distance = this.cullingSystem.frustum.getDistance(entity._shadowBsphere)
                    if (distance - entity._shadowBsphere.radius > maxShadowDistance) {
                        shadowDistanceCulled++
                        return // Shadow too far to be visible
                    }

                    // Skip if projected size is too small (shadow won't be visible)
                    // Use shadow bounding sphere for pixel size calculation
                    if (shadowConfig.minPixelSize > 0) {
                        const projectedSize = this.cullingSystem.frustum.getProjectedSize(entity._shadowBsphere, distance)
                        if (projectedSize < shadowConfig.minPixelSize) {
                            shadowPixelCulled++
                            return // Shadow too small to see
                        }
                    }

                    // Frustum cull using shadow bounding sphere (only for main light)
                    // Spotlights have their own frustum culling in ShadowPass._buildFilteredInstances
                    if (shadowCullingEnabled) {
                        if (!this.cullingSystem.frustum.testSpherePlanes(entity._shadowBsphere)) {
                            shadowFrustumCulled++
                            return // Shadow not in camera frustum
                        }
                    }

                    // HiZ occlusion cull using shadow bounding sphere (only for main light)
                    // Spotlights have their own distance/frustum culling in ShadowPass
                    if (shadowHiZEnabled && this.cullingSystem.frustum.hiZValid) {
                        const occluded = this.passes.hiz.testSphereOcclusion(
                            entity._shadowBsphere,
                            this.cullingSystem.frustum.viewProj
                        )
                        if (occluded) {
                            shadowHiZCulled++
                            return // Shadow occluded by depth buffer
                        }
                    }
                }

                allEntities.push({ id, entity })
            }
        })

        // Store shadow culling stats
        stats.shadowFrustumCulled = shadowFrustumCulled
        stats.shadowHiZCulled = shadowHiZCulled
        stats.shadowDistanceCulled = shadowDistanceCulled
        stats.shadowPixelCulled = shadowPixelCulled

        const allGroups = this.cullingSystem.groupByModel(allEntities)

        // Process lights BEFORE shadow pass so shadow can use processed light data
        // Pass camera for frustum culling and distance ordering of point lights
        const rawLights = entityManager.getLights()
        this.passes.lighting.updateLightsFromEntities(rawLights, camera)

        // Update meshes for shadow pass (includes entities within shadow range, even if outside main frustum)
        this._updateMeshInstancesFromEntities(allGroups, assetManager, meshes, true, camera, 0, null)

        // Execute shadow pass FIRST with shadow-culled data
        const passContext = {
            camera,
            meshes,
            dt,
            lights: this.passes.lighting.lights, // Use processed lights with lightType
            mainLight: this.engine?.settings?.mainLight   // Main directional light settings
        }

        // Pass 1: Shadow (uses shadow-culled entities - can include off-screen objects)
        // Always execute - ShadowPass internally skips cascades when main light is off,
        // but still renders spotlight shadows
        await this.passes.shadow.execute(passContext)

        // Pass 2: Reflection (updates active probes based on camera position)
        await this.passes.reflection.execute(passContext)

        // Pass 2b: Planar Reflection (render scene from mirrored camera)
        // Only execute when enabled - skipped entirely when off
        if (this.passes.planarReflection && this.engine?.settings?.planarReflection?.enabled) {
            // Cull entities specifically for planar reflection (distance, skinned limit, pixel size)
            const { visible: planarVisible } = this.cullingSystem.cull(
                entityManager,
                assetManager,
                'planarReflection'
            )
            const planarGroups = this.cullingSystem.groupByModel(planarVisible)

            // Filter out horizontal sprites from planar reflection (they're flat on ground, shouldn't reflect)
            const planarSpriteEntities = spriteOnlyEntities.filter(item => {
                const pivot = item.entity.pivot || item.entity.sprite?.pivot
                return pivot !== 'horizontal'
            })

            // Update meshes with planar reflection culled entities (including sprites)
            this._updateMeshInstancesFromEntities(planarGroups, assetManager, meshes, true, camera, dt, entityManager, planarSpriteEntities)

            // Set distance fade for planar reflection (prevents object popping at maxDistance)
            const planarCulling = this.engine?.settings?.culling?.planarReflection
            const planarFadeMaxDist = planarCulling?.maxDistance ?? 50
            const planarFadeStart = planarCulling?.fadeStart ?? 0.7
            if (this.passes.planarReflection.gbufferPass) {
                this.passes.planarReflection.gbufferPass.distanceFadeEnd = planarFadeMaxDist
                this.passes.planarReflection.gbufferPass.distanceFadeStart = planarFadeMaxDist * planarFadeStart
            }

            await this.passes.planarReflection.execute(passContext)
        } else {
            // Zero stats when disabled
            stats.planarDrawCalls = 0
            stats.planarTriangles = 0
        }

        // NOW update meshes for main render - overwrites shadow data with main-culled instances
        // Pass sprite-only entities for sprite rendering
        this._updateMeshInstancesFromEntities(groups, assetManager, meshes, true, camera, dt, entityManager, spriteOnlyEntities)

        // Enable TAA jitter for main render (after shadow, before GBuffer)
        // Updates projection matrix with sub-pixel offset for temporal anti-aliasing
        camera.jitterEnabled = this.engine?.settings?.rendering?.jitter ?? true
        camera.updateView()  // Recompute proj with jitter

        // Set distance fade for main render (prevents object popping at maxDistance)
        const mainCulling = this.engine?.settings?.culling?.main
        const mainMaxDist = mainCulling?.maxDistance ?? 1000
        const mainFadeStart = mainCulling?.fadeStart ?? 0.9
        this.passes.gbuffer.distanceFadeEnd = mainMaxDist
        this.passes.gbuffer.distanceFadeStart = mainMaxDist * mainFadeStart

        // Pass 3: GBuffer (uses main-culled entities, outputs velocity for motion vectors)
        passContext.historyManager = this.historyManager
        await this.passes.gbuffer.execute(passContext)

        // Pass 3b: HiZ reduction (for next frame's occlusion culling)
        // Must run after GBuffer to have depth data, before next frame's culling
        if (this.passes.hiz) {
            this.passes.hiz.setDepthTexture(this.passes.gbuffer.getGBuffer()?.depth)
            await this.passes.hiz.execute(passContext)
        }

        // Pass 4: AO (screen-space ambient occlusion)
        await this.passes.ao.execute(passContext)

        // Pass 5: Lighting
        await this.passes.lighting.execute(passContext)

        // Copy lighting and normal to history buffers for temporal effects
        const { device } = this.engine
        const commandEncoder = device.createCommandEncoder({ label: 'historyCommandEncoder' })
        this.historyManager.copyLightingToHistory(commandEncoder, this.passes.lighting.getOutputTexture())
        this.historyManager.copyNormalToHistory(commandEncoder, this.passes.gbuffer.getGBuffer()?.normal)
        device.queue.submit([commandEncoder.finish()])

        const gbuffer = this.passes.gbuffer.getGBuffer()
        const lightingOutput = this.passes.lighting.getOutputTexture()

        // Pass 7: SSGITile (compute shader - accumulate + propagate light between tiles)
        const ssgiEnabled = this.engine?.settings?.ssgi?.enabled
        const prevData = this.historyManager.getPrevious()
        if (this.passes.ssgiTile && ssgiEnabled && prevData.hasValidHistory) {
            // Use previous frame HDR and emissive for tile accumulation
            this.passes.ssgiTile.setPrevHDRTexture(prevData.color)
            this.passes.ssgiTile.setEmissiveTexture(gbuffer.emission)
            await this.passes.ssgiTile.execute(passContext)
        }

        // Pass 8: SSGI (screen-space global illumination - sample from propagated tiles)
        if (this.passes.ssgi && ssgiEnabled && prevData.hasValidHistory) {
            // Pass propagate buffer to SSGI for sampling
            const tileInfo = this.passes.ssgiTile.getTileInfo()
            this.passes.ssgi.setPropagateBuffer(
                this.passes.ssgiTile.getPropagateBuffer(),
                tileInfo.tileCountX,
                tileInfo.tileCountY
            )
            this.passes.ssgi.setGBuffer(gbuffer)
            await this.passes.ssgi.execute({
                camera,
                gbuffer,
            })
        }

        // Pass 9b: Ambient Capture (6-directional sky-aware ambient)
        // Captures sky visibility in 6 directions for ambient lighting
        if (this.passes.ambientCapture && this.engine?.settings?.ambientCapture?.enabled) {
            await this.passes.ambientCapture.execute(passContext)
        }

        // Pass 10: RenderPost (combine SSGI/PlanarReflection/AmbientCapture with lighting)
        let hdrSource = lightingOutput
        if (this.passes.renderPost) {
            const planarEnabled = this.engine?.settings?.planarReflection?.enabled
            const ambientCaptureEnabled = this.engine?.settings?.ambientCapture?.enabled
            await this.passes.renderPost.execute({
                lightingOutput,
                gbuffer,
                camera,
                ssgi: this.passes.ssgi?.getSSGITexture(),
                // Pass null when disabled - renderPost uses black placeholder
                planarReflection: planarEnabled ? this.passes.planarReflection?.getReflectionTexture() : null,
            })

            // Use RenderPost output if any screen-space effect is enabled
            if (ssgiEnabled || planarEnabled || ambientCaptureEnabled) {
                hdrSource = this.passes.renderPost.getOutputTexture()
            }
        }

        // Pass 11: Transparent (forward rendering for alpha-blended materials)
        if (this.passes.transparent) {
            this.passes.transparent.setOutputTexture(hdrSource)
            // Set distance fade for transparent pass (same as main render)
            const transparentCulling = this.engine?.settings?.culling?.main
            const transparentMaxDist = transparentCulling?.maxDistance ?? 1000
            const transparentFadeStart = transparentCulling?.fadeStart ?? 0.9
            this.passes.transparent.distanceFadeEnd = transparentMaxDist
            this.passes.transparent.distanceFadeStart = transparentMaxDist * transparentFadeStart
            await this.passes.transparent.execute(passContext)
        }

        // Pass 12: Fog (distance-based fog with height fade)
        // Applied BEFORE particles - scene fog uses scene depth
        // Particles will apply their own fog based on particle position
        const fogEnabled = this.engine?.settings?.environment?.fog?.enabled
        if (this.passes.fog && fogEnabled) {
            this.passes.fog.setInputTexture(hdrSource)
            this.passes.fog.setGBuffer(gbuffer)
            await this.passes.fog.execute(passContext)
            const fogOutput = this.passes.fog.getOutputTexture()
            if (fogOutput && fogOutput !== hdrSource) {
                hdrSource = fogOutput
            }
        }

        // Pass 12b: Particles (GPU particle system)
        // Rendered AFTER simple fog, BEFORE volumetric fog
        // Particles apply their own fog based on particle world position
        if (this.passes.particles && this.particleSystem.getActiveEmitters().length > 0) {
            this.passes.particles.setOutputTexture(hdrSource)
            await this.passes.particles.execute(passContext)
        }

        // Pass 13: Volumetric Fog (light scattering through particles)
        // Applied last - additive light scattering on top of everything
        const volumetricFogEnabled = this.engine?.settings?.volumetricFog?.enabled
        if (this.passes.volumetricFog && volumetricFogEnabled) {
            this.passes.volumetricFog.setInputTexture(hdrSource)
            this.passes.volumetricFog.setGBuffer(gbuffer)
            await this.passes.volumetricFog.execute(passContext)
            const volFogOutput = this.passes.volumetricFog.getOutputTexture()
            if (volFogOutput && volFogOutput !== hdrSource) {
                hdrSource = volFogOutput
            }
        }

        // Pass 14: Bloom (HDR bright extraction + blur)
        // Runs after transparent so glass/water highlights contribute to bloom
        const bloomEnabled = this.engine?.settings?.bloom?.enabled
        if (this.passes.bloom && bloomEnabled) {
            this.passes.bloom.setInputTexture(hdrSource)
            await this.passes.bloom.execute(passContext)
            this.passes.postProcess.setBloomTexture(this.passes.bloom.getOutputTexture())
        } else {
            this.passes.postProcess.setBloomTexture(null)
        }

        // Pass 13: PostProcess (bloom composite + tone mapping)
        // When CRT is enabled, outputs to intermediate texture instead of canvas
        this.passes.postProcess.setInputTexture(hdrSource)
        await this.passes.postProcess.execute(passContext)

        // Pass 14: CRT effect (optional - outputs to canvas)
        const crtEnabled = this.engine?.settings?.crt?.enabled
        const crtUpscaleEnabled = this.engine?.settings?.crt?.upscaleEnabled
        if (crtEnabled || crtUpscaleEnabled) {
            // Wire CRT pass to receive PostProcess intermediate output
            const postProcessOutput = this.passes.postProcess.getOutputTexture()
            if (postProcessOutput) {
                this.passes.crt.setInputTexture(postProcessOutput)
                this.passes.crt.setRenderSize(
                    this.passes.gbuffer.getGBuffer()?.depth?.width || canvas.width,
                    this.passes.gbuffer.getGBuffer()?.depth?.height || canvas.height
                )
                await this.passes.crt.execute(passContext)
            }
        }

        // Swap history buffers and save camera matrices for next frame
        this.historyManager.swap(camera)

        // Store render context for probe capture (entityManager/assetManager for building fresh batches)
        this._lastRenderContext = {
            meshes,
            entityManager,
            assetManager
        }

        // Update stats
        this.stats.drawCalls = stats.drawCalls
        this.stats.triangles = stats.triangles
    }

    /**
     * Update legacy mesh instances from entity transforms
     * This bridges the entity system with existing mesh rendering
     *
     * @param {Map} groups - Entity groups by model ID
     * @param {AssetManager} assetManager - Asset manager
     * @param {Object} meshes - Meshes dictionary
     * @param {boolean} resetAll - Reset all mesh instance counts
     * @param {Camera} camera - Camera for proximity calculation (null = no individual skins)
     * @param {number} dt - Delta time for animation updates
     * @param {EntityManager} entityManager - Entity manager for animation state
     */
    _updateMeshInstancesFromEntities(groups, assetManager, meshes, resetAll = false, camera = null, dt = 0, entityManager = null, spriteOnlyEntities = []) {
        if (!meshes) return

        const { device } = this.engine

        // Reset ALL mesh instance counts first if requested (for frustum-culled passes)
        // Skip meshes marked as static (manually placed, not entity-managed)
        if (resetAll) {
            for (const name in meshes) {
                const mesh = meshes[name]
                if (mesh.geometry && !mesh.static) {
                    mesh.geometry.instanceCount = 0
                    mesh.geometry._instanceDataDirty = true
                }
            }
        }

        // Track which meshes we've updated
        const updatedMeshes = new Set()

        // Get camera position for proximity check
        const cameraPos = camera ? [camera.position[0], camera.position[1], camera.position[2]] : null
        const individualDistSq = this.individualRenderDistance * this.individualRenderDistance

        // Collect sprite entities and group by material key for batching
        const spriteGroups = new Map()  // materialKey -> { entities, spriteInfo }

        // Separate entities by type and proximity
        const nonSkinnedGroups = new Map()
        const skinnedIndividualEntities = []  // Close entities needing individual skins
        const skinnedInstancedGroups = new Map()  // Far entities for phase-grouped instancing

        for (const [modelId, entities] of groups) {
            // Check for sprite entities (entities with .sprite property but no model asset)
            // These are handled separately with billboard geometry
            for (const item of entities) {
                const entity = item.entity
                if (entity.sprite) {
                    // Parse sprite and compute UV transform
                    const spriteInfo = this.spriteSystem.parseSprite(entity.sprite)
                    if (spriteInfo) {
                        // Compute UV transform from frame (uses animated _uvTransform if available)
                        const instanceData = this.spriteSystem.getSpriteInstanceData(entity)
                        entity._uvTransform = instanceData.uvTransform

                        // Group sprites by material key for batching
                        const pivot = entity.pivot || 'center'
                        const roughness = entity.roughness ?? 0.7
                        const materialKey = `sprite:${spriteInfo.url}:${pivot}:r${roughness.toFixed(2)}`

                        if (!spriteGroups.has(materialKey)) {
                            spriteGroups.set(materialKey, {
                                entities: [],
                                spriteInfo,
                                pivot,
                                roughness
                            })
                        }
                        spriteGroups.get(materialKey).entities.push(item)
                    }
                }
            }

            const asset = assetManager.get(modelId)

            // Handle parent GLTF paths (expand to all submeshes)
            // If entity.model is "model.glb" instead of "model.glb|meshName", expand to all meshes
            if (asset?.meshNames && !asset.mesh) {
                // This is a parent GLTF asset - expand to all submeshes
                // Each submesh will share the same entity transform/animation/phase
                for (const meshName of asset.meshNames) {
                    const submeshId = assetManager.createModelId(modelId, meshName)
                    const submeshAsset = assetManager.get(submeshId)
                    if (!submeshAsset?.mesh) continue

                    // Add this submesh to the appropriate group
                    if (submeshAsset.hasSkin && submeshAsset.skin) {
                        // Skinned submesh - process each entity for this submesh
                        // Expanded submeshes ALWAYS use phase-grouped instancing (even when close)
                        // Individual rendering doesn't support multi-submesh expansion
                        for (const item of entities) {
                            const entity = item.entity

                            const animation = entity.animation || 'default'
                            const phase = entity.phase || 0
                            const quantizedPhase = Math.floor(phase / 0.05) * 0.05
                            const key = `${submeshId}|${animation}|${quantizedPhase.toFixed(2)}`

                            if (!skinnedInstancedGroups.has(key)) {
                                skinnedInstancedGroups.set(key, {
                                    modelId: submeshId, animation, phase: quantizedPhase, asset: submeshAsset, entities: []
                                })
                            }
                            skinnedInstancedGroups.get(key).entities.push(item)
                        }
                    } else {
                        // Non-skinned submesh - add to non-skinned groups
                        if (!nonSkinnedGroups.has(submeshId)) {
                            nonSkinnedGroups.set(submeshId, { asset: submeshAsset, entities: [] })
                        }
                        for (const item of entities) {
                            nonSkinnedGroups.get(submeshId).entities.push(item)
                        }
                    }
                }
                continue  // Skip normal processing, we've handled expansion
            }

            if (!asset?.mesh) continue

            if (asset.hasSkin && asset.skin) {
                // For skinned meshes, check proximity for each entity
                for (const item of entities) {
                    const entity = item.entity
                    const entityId = item.id

                    // Calculate distance to camera
                    let useIndividual = false
                    if (cameraPos && entity._bsphere) {
                        const dx = entity._bsphere.center[0] - cameraPos[0]
                        const dy = entity._bsphere.center[1] - cameraPos[1]
                        const dz = entity._bsphere.center[2] - cameraPos[2]
                        const distSq = dx * dx + dy * dy + dz * dz
                        useIndividual = distSq < individualDistSq
                    }

                    // Check if individual mesh is ready and pipeline is stable
                    // If not, keep in instanced group to avoid flash during transition
                    let addToIndividualList = false
                    if (useIndividual) {
                        const cached = this._individualSkinCache.get(entityId)
                        if (cached?.mesh && this.passes.gbuffer) {
                            // Only switch if pipeline is stable
                            if (!this.passes.gbuffer.isPipelineStable(cached.mesh)) {
                                addToIndividualList = true  // Create/warm pipeline
                                useIndividual = false  // But keep in instanced until ready
                            }
                        } else if (!cached) {
                            // No cache yet - will be created, but keep in instanced for this frame
                            addToIndividualList = true  // Create the mesh/pipeline
                            useIndividual = false  // Keep in instanced for this frame
                        }
                    }

                    if (useIndividual || addToIndividualList) {
                        skinnedIndividualEntities.push({ id: entityId, entity, asset, modelId })
                    }
                    if (!useIndividual) {
                        // Group by animation and phase for instancing
                        const animation = entity.animation || 'default'
                        const phase = entity.phase || 0
                        const quantizedPhase = Math.floor(phase / 0.05) * 0.05
                        const key = `${modelId}|${animation}|${quantizedPhase.toFixed(2)}`

                        if (!skinnedInstancedGroups.has(key)) {
                            skinnedInstancedGroups.set(key, {
                                modelId, animation, phase: quantizedPhase, asset, entities: []
                            })
                        }
                        skinnedInstancedGroups.get(key).entities.push(item)
                    }
                }
            } else {
                nonSkinnedGroups.set(modelId, { asset, entities })
            }
        }

        // Process non-skinned meshes (simple path)
        for (const [modelId, { asset, entities }] of nonSkinnedGroups) {
            const mesh = asset.mesh
            const geometry = mesh.geometry

            let meshName = null
            for (const name in meshes) {
                if (meshes[name] === mesh || meshes[name].geometry === geometry) {
                    meshName = name
                    break
                }
            }

            if (!meshName) {
                // Use sanitized modelId as base name, but ensure uniqueness
                let baseName = modelId.replace(/[^a-zA-Z0-9]/g, '_')
                meshName = baseName
                // If name already exists with a DIFFERENT mesh, make it unique
                let counter = 1
                while (meshes[meshName] && meshes[meshName] !== mesh && meshes[meshName].geometry !== geometry) {
                    meshName = `${baseName}_${counter++}`
                }
                meshes[meshName] = mesh
            }

            geometry.instanceCount = 0

            for (const item of entities) {
                const entity = item.entity
                const idx = geometry.instanceCount

                if (idx >= geometry.maxInstances) {
                    geometry.growInstanceBuffer(entities.length)
                }

                geometry.instanceCount++
                const base = idx * 28
                geometry.instanceData.set(entity._matrix, base)
                geometry.instanceData[base + 16] = entity._bsphere.center[0]
                geometry.instanceData[base + 17] = entity._bsphere.center[1]
                geometry.instanceData[base + 18] = entity._bsphere.center[2]
                // Negative radius signals shader to skip pixel/position rounding
                geometry.instanceData[base + 19] = entity.noRounding
                    ? -Math.max(entity._bsphere.radius, 1)
                    : entity._bsphere.radius

                // uvTransform: [offsetX, offsetY, scaleX, scaleY] - default full texture
                const uvTransform = entity._uvTransform || [0, 0, 1, 1]
                geometry.instanceData[base + 20] = uvTransform[0]
                geometry.instanceData[base + 21] = uvTransform[1]
                geometry.instanceData[base + 22] = uvTransform[2]
                geometry.instanceData[base + 23] = uvTransform[3]

                // color: [r, g, b, a] - default white
                const color = entity.color || [1, 1, 1, 1]
                geometry.instanceData[base + 24] = color[0]
                geometry.instanceData[base + 25] = color[1]
                geometry.instanceData[base + 26] = color[2]
                geometry.instanceData[base + 27] = color[3]
            }

            geometry._instanceDataDirty = true
            updatedMeshes.add(meshName)
        }

        // Process individual skinned entities (close to camera, with blending support)
        // Animation time can be scaled by settings.animation.speed (default 1.0)
        const animationSpeed = this.engine?.settings?.animation?.speed ?? 1.0
        const globalTime = (performance.now() / 1000) * animationSpeed

        for (const { id: entityId, entity, asset, modelId } of skinnedIndividualEntities) {
            const entityAnimation = entity.animation || 'default'
            const entityPhase = entity.phase || 0

            // Get or create individual skin for this entity
            let cached = this._individualSkinCache.get(entityId)

            if (!cached) {
                // Create individual skin with local transforms for blending
                const individualSkin = asset.skin.cloneForIndividual()

                // Create geometry wrapper for single instance
                const originalGeom = asset.mesh.geometry
                const geomUid = `individual_${entityId}_${Date.now()}`

                const individualGeometry = {
                    uid: geomUid,
                    vertexBuffer: originalGeom.vertexBuffer,
                    indexBuffer: originalGeom.indexBuffer,
                    vertexBufferLayout: originalGeom.vertexBufferLayout,
                    instanceBufferLayout: originalGeom.instanceBufferLayout,
                    vertexCount: originalGeom.vertexCount,
                    indexArray: originalGeom.indexArray,
                    attributes: originalGeom.attributes,
                    maxInstances: 1,
                    instanceCount: 1,
                    instanceData: new Float32Array(28),  // 28 floats: matrix(16) + posRadius(4) + uvTransform(4) + color(4)
                    instanceBuffer: device.createBuffer({
                        size: 112,  // 28 floats * 4 bytes
                        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                    }),
                    _instanceDataDirty: true,
                    writeInstanceBuffer() {
                        device.queue.writeBuffer(this.instanceBuffer, 0, this.instanceData)
                    },
                    update() {
                        if (this._instanceDataDirty) {
                            this.writeInstanceBuffer()
                            this._instanceDataDirty = false
                        }
                    }
                }

                const individualMesh = {
                    geometry: individualGeometry,
                    material: asset.mesh.material,
                    skin: individualSkin,
                    hasSkin: true,
                    uid: `individual_${entityId}`,
                    // Use asset's combined bsphere for culling (all skinned submeshes share one sphere)
                    combinedBsphere: asset.bsphere || null
                }

                cached = {
                    skin: individualSkin,
                    mesh: individualMesh,
                    geometry: individualGeometry,
                    lastAnimation: entityAnimation,
                    // Blending state
                    blendStartTime: 0,
                    blendFromAnim: null,
                    blendFromPhaseOffset: 0
                }
                this._individualSkinCache.set(entityId, cached)

                // Initialize animation
                individualSkin.currentAnimation = entityAnimation
            }

            const { skin: individualSkin, mesh: individualMesh, geometry: individualGeometry } = cached

            // Get animation info
            const anim = individualSkin.animations[entityAnimation]
            if (!anim) continue

            // Calculate time EXACTLY the same way as far mode
            // This ensures no glitch when switching between modes
            const phaseOffset = entityPhase * anim.duration
            const baseTime = globalTime + phaseOffset

            // Check if animation changed - start blend
            if (dt > 0 && cached.lastAnimation !== entityAnimation) {
                // Animation changed! Start blending
                cached.blendFromAnim = cached.lastAnimation
                cached.blendFromPhaseOffset = entityPhase * (individualSkin.animations[cached.lastAnimation]?.duration || anim.duration)
                cached.blendStartTime = globalTime
                cached.lastAnimation = entityAnimation
                individualSkin.currentAnimation = entityAnimation
                individualSkin.isBlending = true
                individualSkin.blendDuration = 0.3
            }

            // Handle blending with global time (not dt-based)
            if (individualSkin.isBlending && cached.blendFromAnim) {
                const blendElapsed = globalTime - cached.blendStartTime
                const blendWeight = Math.min(blendElapsed / individualSkin.blendDuration, 1.0)

                if (blendWeight >= 1.0) {
                    // Blend complete
                    individualSkin.isBlending = false
                    cached.blendFromAnim = null
                    individualSkin.time = baseTime
                    individualSkin.update(0)  // Apply current animation only
                } else {
                    // Blending - manually apply both animations
                    const fromAnim = individualSkin.animations[cached.blendFromAnim]
                    const fromTime = globalTime + cached.blendFromPhaseOffset

                    // Set up blend state
                    individualSkin.blendFromAnimation = cached.blendFromAnim
                    individualSkin.blendFromTime = fromTime
                    individualSkin.blendWeight = blendWeight
                    individualSkin.time = baseTime

                    // Update applies blended animation
                    individualSkin.update(0)
                }
            } else {
                // No blending - just set time and update
                individualSkin.time = baseTime
                individualSkin.update(0)
            }

            // Update instance data
            individualGeometry.instanceCount = 1
            individualGeometry.instanceData.set(entity._matrix, 0)
            individualGeometry.instanceData[16] = entity._bsphere.center[0]
            individualGeometry.instanceData[17] = entity._bsphere.center[1]
            individualGeometry.instanceData[18] = entity._bsphere.center[2]
            individualGeometry.instanceData[19] = entity._bsphere.radius
            // uvTransform: default full texture
            const uvTransform = entity._uvTransform || [0, 0, 1, 1]
            individualGeometry.instanceData[20] = uvTransform[0]
            individualGeometry.instanceData[21] = uvTransform[1]
            individualGeometry.instanceData[22] = uvTransform[2]
            individualGeometry.instanceData[23] = uvTransform[3]
            // color: default white
            const color = entity.color || [1, 1, 1, 1]
            individualGeometry.instanceData[24] = color[0]
            individualGeometry.instanceData[25] = color[1]
            individualGeometry.instanceData[26] = color[2]
            individualGeometry.instanceData[27] = color[3]
            individualGeometry._instanceDataDirty = true

            // Register in meshes dict
            const meshName = `individual_${entityId}`
            meshes[meshName] = individualMesh
            updatedMeshes.add(meshName)
        }

        // Process instanced skinned entities (far from camera, phase-grouped)
        for (const [key, group] of skinnedInstancedGroups) {
            const { modelId, animation, phase, asset, entities } = group

            // Get or create cloned skin and geometry wrapper for this phase group
            let cached = this._skinnedPhaseCache.get(key)
            if (!cached) {
                const clonedSkin = asset.skin.clone()
                clonedSkin.currentAnimation = animation

                const originalGeom = asset.mesh.geometry
                const phaseGeomUid = `phase_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

                const phaseGeometry = {
                    uid: phaseGeomUid,
                    vertexBuffer: originalGeom.vertexBuffer,
                    indexBuffer: originalGeom.indexBuffer,
                    vertexBufferLayout: originalGeom.vertexBufferLayout,
                    instanceBufferLayout: originalGeom.instanceBufferLayout,
                    vertexCount: originalGeom.vertexCount,
                    indexArray: originalGeom.indexArray,
                    attributes: originalGeom.attributes,
                    maxInstances: 64,
                    instanceCount: 0,
                    instanceData: new Float32Array(28 * 64),  // 28 floats per instance
                    instanceBuffer: device.createBuffer({
                        size: 112 * 64,  // 112 bytes per instance
                        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                    }),
                    _instanceDataDirty: true,
                    growInstanceBuffer(minCapacity) {
                        let newMax = this.maxInstances * 2
                        while (newMax < minCapacity) newMax *= 2
                        const newBuffer = device.createBuffer({
                            size: 112 * newMax,  // 112 bytes per instance
                            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                        })
                        const newData = new Float32Array(28 * newMax)  // 28 floats per instance
                        newData.set(this.instanceData)
                        this.instanceBuffer.destroy()
                        this.instanceBuffer = newBuffer
                        this.instanceData = newData
                        this.maxInstances = newMax
                        this._instanceDataDirty = true
                    },
                    writeInstanceBuffer() {
                        device.queue.writeBuffer(this.instanceBuffer, 0, this.instanceData)
                    },
                    update() {
                        if (this._instanceDataDirty) {
                            this.writeInstanceBuffer()
                            this._instanceDataDirty = false
                        }
                    }
                }

                const phaseMesh = {
                    geometry: phaseGeometry,
                    material: asset.mesh.material,
                    skin: clonedSkin,
                    hasSkin: true,
                    uid: asset.mesh.uid + '_phase_' + key.replace(/[^a-zA-Z0-9]/g, '_'),
                    // Use asset's combined bsphere for culling (all skinned submeshes share one sphere)
                    combinedBsphere: asset.bsphere || null
                }

                cached = { skin: clonedSkin, mesh: phaseMesh, geometry: phaseGeometry }
                this._skinnedPhaseCache.set(key, cached)
            }

            const { skin: clonedSkin, mesh: phaseMesh, geometry: phaseGeometry } = cached

            const meshName = `skinned_${key.replace(/[^a-zA-Z0-9]/g, '_')}`
            meshes[meshName] = phaseMesh

            const anim = clonedSkin.animations[animation]
            if (!anim) continue

            const phaseOffset = phase * anim.duration
            clonedSkin.currentAnimation = animation
            clonedSkin.updateAtTime(globalTime + phaseOffset)

            phaseGeometry.instanceCount = 0

            for (const item of entities) {
                const entity = item.entity
                const idx = phaseGeometry.instanceCount

                if (idx >= phaseGeometry.maxInstances) {
                    phaseGeometry.growInstanceBuffer(entities.length)
                }

                phaseGeometry.instanceCount++
                const base = idx * 28
                phaseGeometry.instanceData.set(entity._matrix, base)
                phaseGeometry.instanceData[base + 16] = entity._bsphere.center[0]
                phaseGeometry.instanceData[base + 17] = entity._bsphere.center[1]
                phaseGeometry.instanceData[base + 18] = entity._bsphere.center[2]
                phaseGeometry.instanceData[base + 19] = entity._bsphere.radius
                // uvTransform: default full texture
                const uvTransform = entity._uvTransform || [0, 0, 1, 1]
                phaseGeometry.instanceData[base + 20] = uvTransform[0]
                phaseGeometry.instanceData[base + 21] = uvTransform[1]
                phaseGeometry.instanceData[base + 22] = uvTransform[2]
                phaseGeometry.instanceData[base + 23] = uvTransform[3]
                // color: default white
                const color = entity.color || [1, 1, 1, 1]
                phaseGeometry.instanceData[base + 24] = color[0]
                phaseGeometry.instanceData[base + 25] = color[1]
                phaseGeometry.instanceData[base + 26] = color[2]
                phaseGeometry.instanceData[base + 27] = color[3]
            }

            phaseGeometry._instanceDataDirty = true
            updatedMeshes.add(meshName)
        }

        // Add sprite-only entities (entities with .sprite but no .model) to spriteGroups
        for (const item of spriteOnlyEntities) {
            const entity = item.entity
            const spriteInfo = this.spriteSystem.parseSprite(entity.sprite)
            if (!spriteInfo) continue

            // Compute UV transform for sprite-only entities
            const instanceData = this.spriteSystem.getSpriteInstanceData(entity)
            entity._uvTransform = instanceData.uvTransform

            // Group by material key (same format as model+sprite entities)
            const pivot = entity.pivot || 'center'
            const roughness = entity.roughness ?? 0.7
            const materialKey = `sprite:${spriteInfo.url}:${pivot}:r${roughness.toFixed(2)}`

            if (!spriteGroups.has(materialKey)) {
                spriteGroups.set(materialKey, {
                    entities: [],
                    spriteInfo,
                    pivot,
                    roughness
                })
            }
            spriteGroups.get(materialKey).entities.push(item)
        }

        // Process sprite entities (billboard quads)
        // This is done synchronously - sprite assets are cached after first load
        for (const [materialKey, group] of spriteGroups) {
            const { entities, spriteInfo, pivot, roughness } = group

            // Get or create sprite mesh (async on first call, cached thereafter)
            // Note: We use a simple approach - if the asset isn't loaded yet, skip this frame
            const meshName = `sprite_${materialKey.replace(/[^a-zA-Z0-9]/g, '_')}`

            let spriteMesh = meshes[meshName]
            if (!spriteMesh) {
                // Check if material is ready
                const material = this.spriteSystem._materialCache.get(materialKey)

                if (!material) {
                    // Material not loaded yet - trigger async load and skip this frame
                    this.spriteSystem.getSpriteMaterial(spriteInfo.url, roughness, pivot)
                    continue
                }

                // Create a NEW geometry for each sprite batch (not shared)
                // This is needed because each batch has its own instance data
                const geometry = Geometry.billboardQuad(this.engine, pivot)

                // Create mesh with sprite geometry and material
                spriteMesh = {
                    geometry: geometry,
                    material: material,
                    hasSkin: false,
                    uid: meshName
                }
                meshes[meshName] = spriteMesh
            }

            const geometry = spriteMesh.geometry

            // Reset instance count for this frame
            geometry.instanceCount = 0

            // Ensure geometry has instance buffer large enough
            if (entities.length > geometry.maxInstances) {
                geometry.growInstanceBuffer(entities.length)
            }

            // Add sprite instances
            for (const item of entities) {
                const entity = item.entity
                const idx = geometry.instanceCount

                geometry.instanceCount++
                const base = idx * 28
                geometry.instanceData.set(entity._matrix, base)

                // Bounding sphere (use scale for radius approximation)
                const center = entity.position || [0, 0, 0]
                const radius = Math.max(entity.scale?.[0] || 1, entity.scale?.[1] || 1) * 0.5
                geometry.instanceData[base + 16] = center[0]
                geometry.instanceData[base + 17] = center[1]
                geometry.instanceData[base + 18] = center[2]
                geometry.instanceData[base + 19] = entity.noRounding ? -radius : radius

                // uvTransform from sprite frame
                const uvTransform = entity._uvTransform || [0, 0, 1, 1]
                geometry.instanceData[base + 20] = uvTransform[0]
                geometry.instanceData[base + 21] = uvTransform[1]
                geometry.instanceData[base + 22] = uvTransform[2]
                geometry.instanceData[base + 23] = uvTransform[3]

                // color tint
                const color = entity.color || [1, 1, 1, 1]
                geometry.instanceData[base + 24] = color[0]
                geometry.instanceData[base + 25] = color[1]
                geometry.instanceData[base + 26] = color[2]
                geometry.instanceData[base + 27] = color[3]
            }

            geometry._instanceDataDirty = true
            updatedMeshes.add(meshName)
        }
    }

    /**
     * Render a frame using the legacy mesh system (backward compatibility)
     *
     * @param {Object} meshes - Dictionary of meshes
     * @param {Camera} camera - Current camera
     * @param {number} dt - Delta time
     */
    async render(meshes, camera, dt = 0) {
        const { stats } = this.engine

        stats.drawCalls = 0
        stats.triangles = 0

        const passContext = {
            camera,
            meshes,
            dt
        }

        // Pass 4: GBuffer
        await this.passes.gbuffer.execute(passContext)

        // Pass 6: Lighting
        await this.passes.lighting.execute(passContext)

        // Pass 7: PostProcess
        await this.passes.postProcess.execute(passContext)

        this.stats.drawCalls = stats.drawCalls
        this.stats.triangles = stats.triangles
    }

    /**
     * Handle window resize
     * @param {number} width - Canvas width (full device pixels)
     * @param {number} height - Canvas height (full device pixels)
     * @param {number} renderScale - Scale for internal rendering (1.0 = full resolution)
     */
    async resize(width, height, renderScale = 1.0) {
        const timings = []
        const startTotal = performance.now()

        // Store full canvas dimensions (for CRT pixel-perfect output)
        this.canvasWidth = width
        this.canvasHeight = height

        // Calculate internal render dimensions (scaled for performance)
        const renderWidth = Math.max(1, Math.round(width * renderScale))
        const renderHeight = Math.max(1, Math.round(height * renderScale))

        // Store render dimensions
        this.renderWidth = renderWidth
        this.renderHeight = renderHeight
        this.renderScale = renderScale

        // Calculate effect scale for expensive passes
        // When autoScale.enabled is false but enabledForEffects is true and height > maxHeight,
        // expensive effects (bloom, AO, SSGI, planar reflection) render at reduced resolution
        const autoScale = this.engine?.settings?.rendering?.autoScale
        let effectScale = 1.0

        if (autoScale && !autoScale.enabled && autoScale.enabledForEffects) {
            if (renderHeight > (autoScale.maxHeight ?? 1536)) {
                effectScale = autoScale.scaleFactor ?? 0.5
                if (!this._effectScaleWarned) {
                    console.log(`Effect auto-scale: Reducing effect resolution by ${effectScale} (height: ${renderHeight}px > ${autoScale.maxHeight}px)`)
                    this._effectScaleWarned = true
                }
            } else if (this._effectScaleWarned) {
                console.log(`Effect auto-scale: Restoring full effect resolution (height: ${renderHeight}px <= ${autoScale.maxHeight}px)`)
                this._effectScaleWarned = false
            }
        }

        // Passes that render at full canvas resolution (for pixel-perfect output)
        const fullResPasses = new Set(['crt'])

        // Expensive passes that should be scaled down at high resolutions
        // Note: ssgiTile and ssgi must be at the same resolution (they share tile grid)
        // Note: planarReflection combines effectScale with its own resolution setting
        const effectScaledPasses = new Set(['bloom', 'ao', 'planarReflection'])

        // Calculate scaled dimensions for expensive effects (relative to render dimensions)
        const effectWidth = Math.max(1, Math.floor(renderWidth * effectScale))
        const effectHeight = Math.max(1, Math.floor(renderHeight * effectScale))

        // Store effect dimensions for use in rendering
        this.effectWidth = effectWidth
        this.effectHeight = effectHeight
        this.effectScale = effectScale

        // Resize all passes
        for (const passName in this.passes) {
            if (this.passes[passName]) {
                const start = performance.now()
                let w, h
                if (fullResPasses.has(passName)) {
                    // CRT and similar passes render at full canvas resolution
                    w = width
                    h = height
                } else if (effectScaledPasses.has(passName) && effectScale < 1.0) {
                    // Expensive effects use effect-scaled dimensions
                    w = effectWidth
                    h = effectHeight
                } else {
                    // All other passes use render-scaled dimensions
                    w = renderWidth
                    h = renderHeight
                }
                await this.passes[passName].resize(w, h)
                timings.push({ name: `pass:${passName}`, time: performance.now() - start })
            }
        }

        // Resize history buffer manager (uses render dimensions)
        if (this.historyManager) {
            const start = performance.now()
            await this.historyManager.resize(renderWidth, renderHeight)
            timings.push({ name: 'historyManager', time: performance.now() - start })
        }

        // Rewire dependencies after resize
        let start = performance.now()
        await this.passes.lighting.setGBuffer(this.passes.gbuffer.getGBuffer())
        timings.push({ name: 'rewire:lighting.setGBuffer', time: performance.now() - start })

        start = performance.now()
        this.passes.lighting.setShadowPass(this.passes.shadow)
        timings.push({ name: 'rewire:lighting.setShadowPass', time: performance.now() - start })

        start = performance.now()
        this.passes.gbuffer.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        timings.push({ name: 'rewire:gbuffer.setNoise', time: performance.now() - start })

        // Rewire HiZ pass with new GBuffer depth and all passes that use it
        start = performance.now()
        if (this.passes.hiz) {
            this.passes.hiz.setDepthTexture(this.passes.gbuffer.getGBuffer()?.depth)
            this.passes.gbuffer.setHiZPass(this.passes.hiz)
            this.passes.lighting.setHiZPass(this.passes.hiz)
            this.passes.transparent.setHiZPass(this.passes.hiz)
            this.passes.shadow.setHiZPass(this.passes.hiz)
            if (this.passes.volumetricFog) {
                this.passes.volumetricFog.setHiZPass(this.passes.hiz)
            }
        }
        timings.push({ name: 'rewire:hiz', time: performance.now() - start })

        // Rewire volumetric fog pass
        if (this.passes.volumetricFog) {
            start = performance.now()
            this.passes.volumetricFog.setGBuffer(this.passes.gbuffer.getGBuffer())
            this.passes.volumetricFog.setShadowPass(this.passes.shadow)
            this.passes.volumetricFog.setLightingPass(this.passes.lighting)
            timings.push({ name: 'rewire:volumetricFog', time: performance.now() - start })
        }

        start = performance.now()
        this.passes.shadow.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        timings.push({ name: 'rewire:shadow.setNoise', time: performance.now() - start })

        start = performance.now()
        await this.passes.ao.setGBuffer(this.passes.gbuffer.getGBuffer())
        timings.push({ name: 'rewire:ao.setGBuffer', time: performance.now() - start })

        start = performance.now()
        this.passes.lighting.setAOTexture(this.passes.ao.getOutputTexture())
        timings.push({ name: 'rewire:lighting.setAOTexture', time: performance.now() - start })

        start = performance.now()
        this.passes.postProcess.setInputTexture(this.passes.lighting.getOutputTexture())
        timings.push({ name: 'rewire:postProcess.setInputTexture', time: performance.now() - start })

        start = performance.now()
        this.passes.postProcess.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        timings.push({ name: 'rewire:postProcess.setNoise', time: performance.now() - start })

        // Rewire transparent pass
        start = performance.now()
        this.passes.transparent.setGBuffer(this.passes.gbuffer.getGBuffer())
        this.passes.transparent.setShadowPass(this.passes.shadow)
        this.passes.transparent.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        timings.push({ name: 'rewire:transparent', time: performance.now() - start })

        // Rewire particle pass
        start = performance.now()
        this.passes.particles.setGBuffer(this.passes.gbuffer.getGBuffer())
        timings.push({ name: 'rewire:particles', time: performance.now() - start })

        // SSGI passes are wired dynamically per frame (no static rewire needed)

    }

    /**
     * Update environment map
     * @param {Texture} environmentMap - New environment map
     * @param {number} encoding - 0 = equirectangular (default), 1 = octahedral
     */
    setEnvironmentMap(environmentMap, encoding = 0) {
        this.environmentMap = environmentMap
        this.environmentEncoding = encoding
        this.passes.lighting.setEnvironmentMap(environmentMap, encoding)
        if (this.passes.reflection) {
            this.passes.reflection.setFallbackEnvironment(environmentMap, encoding)
        }
        if (this.passes.transparent) {
            this.passes.transparent.setEnvironmentMap(environmentMap, encoding)
        }
        if (this.passes.ambientCapture) {
            this.passes.ambientCapture.setDependencies({
                environmentMap,
                encoding
            })
        }
    }

    /**
     * Load reflection probes for a world
     * @param {string} worldId - World identifier
     */
    async loadWorldProbes(worldId) {
        if (this.passes.reflection) {
            await this.passes.reflection.loadWorldProbes(worldId)
        }
    }

    /**
     * Load a specific reflection probe
     * @param {string} url - URL to probe HDR image
     * @param {vec3} position - World position
     * @param {string} worldId - World identifier
     */
    async loadProbe(url, position, worldId = 'default') {
        if (this.passes.reflection) {
            return await this.passes.reflection.loadProbe(url, position, worldId)
        }
        return null
    }

    /**
     * Request a probe capture at position
     * @param {vec3} position - Capture position
     * @param {string} worldId - World identifier
     */
    requestProbeCapture(position, worldId = 'default') {
        if (this.passes.reflection) {
            this.passes.reflection.requestCapture(position, worldId)
        }
    }

    /**
     * Get the reflection probe manager
     */
    getProbeManager() {
        return this.passes.reflection?.getProbeManager()
    }

    /**
     * Capture a probe at position and optionally save/use it
     * @param {vec3} position - Capture position
     * @param {Object} options - { save: bool, filename: string, format: 'hdr'|'jpg', useAsEnvironment: bool, saveDebug: bool, saveFaces: bool }
     */
    async captureProbe(position, options = {}) {
        const {
            save = true,
            filename = 'probe',  // Base filename without extension
            format = 'jpg',  // 'hdr' or 'jpg' (jpg = RGB + exp pair)
            useAsEnvironment = false,
            saveDebug = false,  // Save tone-mapped PNG for preview
            saveFaces = false
        } = options
        const probeCapture = this.passes.reflection?.getProbeCapture()

        if (!probeCapture) {
            console.error('RenderGraph: ProbeCapture not initialized')
            return
        }

        // CRITICAL: Pause main render loop during probe capture
        // The main render modifies mesh.geometry.instanceCount on shared mesh objects
        // which corrupts the probe capture data
        this._isCapturingProbe = true

        try {
            // Clear probe pass pipeline caches to ensure fresh creation
            // This fixes issues where pipelines from previous captures may be stale
            if (this.probePasses.gbuffer) {
                this.probePasses.gbuffer.pipelines.clear()
                this.probePasses.gbuffer.skinnedPipelines.clear()
            }

            // Clear probe meshes dictionary to avoid stale entries
            this._probeMeshes = {}

            await probeCapture.capture(position)

            // Save cube faces for debugging (before octahedral conversion)
            if (saveFaces) {
                await probeCapture.saveCubeFaces('face')
            }

            if (save) {
                if (format === 'hdr') {
                    // Save as Radiance HDR file
                    await probeCapture.saveAsHDR(`${filename}.hdr`)
                } else {
                    // Save as JPG pair (RGB + exponent)
                    await probeCapture.saveAsJPG(filename)
                }
            }

            if (saveDebug) {
                // Save tone-mapped PNG for preview/debugging
                await probeCapture.saveAsDebugPNG(`${filename}_debug.png`)
            }

            if (useAsEnvironment) {
                const envTex = await probeCapture.getAsEnvironmentTexture()
                if (envTex) {
                    this.passes.lighting.setEnvironmentMap(envTex, 1) // 1 = octahedral encoding
                    console.log('RenderGraph: Using captured probe as environment (octahedral, RGBE)')
                }
            }

            return probeCapture.getProbeTexture()
        } finally {
            // Always reset the flag, even if capture fails
            this._isCapturingProbe = false
        }
    }

    /**
     * Convert an equirectangular HDR environment map to octahedral format
     * and save as RGBI JPG pair for efficient storage
     *
     * @param {Object} options - Conversion options
     * @param {string} options.url - URL to HDR file (uses current environment if not provided)
     * @param {string} options.filename - Base filename without extension (default: 'environment')
     * @param {boolean} options.useAsEnvironment - Set converted map as active environment (default: true)
     * @param {boolean} options.saveDebug - Also save tone-mapped PNG for preview (default: false)
     * @returns {Promise<Object>} The converted texture
     */
    async convertEquirectToOctahedral(options = {}) {
        const {
            url = null,
            filename = 'environment',
            useAsEnvironment = true,
            saveDebug = false
        } = options

        const probeCapture = this.passes.reflection?.getProbeCapture()

        if (!probeCapture) {
            console.error('RenderGraph: ProbeCapture not initialized')
            return null
        }

        // Load HDR file if URL provided, otherwise use current environment
        let sourceEnvMap = this.environmentMap
        if (url) {
            console.log(`RenderGraph: Loading HDR from ${url}`)
            sourceEnvMap = await Texture.fromImage(this.engine, url)
        }

        if (!sourceEnvMap) {
            console.error('RenderGraph: No environment map available')
            return null
        }

        console.log('RenderGraph: Converting equirectangular to octahedral format...')

        // Convert equirectangular to octahedral
        await probeCapture.convertEquirectToOctahedral(sourceEnvMap)

        // Save as RGBI JPG pair
        await probeCapture.saveAsJPG(filename)

        if (saveDebug) {
            // Save tone-mapped PNG for preview
            await probeCapture.saveAsDebugPNG(`${filename}_debug.png`)
        }

        // Optionally set as environment
        if (useAsEnvironment) {
            const envTex = await probeCapture.getAsEnvironmentTexture()
            if (envTex) {
                this.passes.lighting.setEnvironmentMap(envTex, 1) // 1 = octahedral encoding
                console.log('RenderGraph: Using converted environment (octahedral, RGBE)')
            }
        }

        console.log(`RenderGraph: Saved octahedral environment as ${filename}.jpg + ${filename}.int.jpg`)
        return probeCapture.getProbeTexture()
    }

    /**
     * Initialize probe-specific passes at fixed 256x256 size
     * These are used for probe face capture (smaller than main passes)
     */
    async _initProbePasses() {
        const { device } = this.engine
        const probeSize = 1024

        // Create probe GBuffer pass
        this.probePasses.gbuffer = new GBufferPass(this.engine)
        await this.probePasses.gbuffer.initialize()
        await this.probePasses.gbuffer.resize(probeSize, probeSize)

        // Create probe Lighting pass
        this.probePasses.lighting = new LightingPass(this.engine)
        await this.probePasses.lighting.initialize()
        await this.probePasses.lighting.resize(probeSize, probeSize)
        // Use exposure = 1.0 for probe capture (raw HDR values, no display exposure)
        this.probePasses.lighting.exposureOverride = 1.0

        // Create a simple white AO texture for probes (skip AO computation)
        const dummyAOTexture = device.createTexture({
            label: 'probeAO',
            size: [probeSize, probeSize],
            format: 'r8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
        })
        // Fill with white (AO = 1.0 = no occlusion)
        const whiteData = new Uint8Array(probeSize * probeSize).fill(255)
        device.queue.writeTexture(
            { texture: dummyAOTexture },
            whiteData,
            { bytesPerRow: probeSize },
            { width: probeSize, height: probeSize }
        )
        this.probePasses.dummyAO = {
            texture: dummyAOTexture,
            view: dummyAOTexture.createView()
        }

        // Wire up probe passes
        await this.probePasses.lighting.setGBuffer(this.probePasses.gbuffer.getGBuffer())
        this.probePasses.lighting.setEnvironmentMap(this.environmentMap, this.environmentEncoding)
        this.probePasses.lighting.setShadowPass(this.passes.shadow)  // Share shadow pass
        this.probePasses.lighting.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        this.probePasses.lighting.setAOTexture(this.probePasses.dummyAO)
    }

    /**
     * Render scene for a probe face
     * Called by ProbeCapture for each of the 6 cube faces
     *
     * @param {mat4} viewMatrix - View matrix for this face
     * @param {mat4} projMatrix - Projection matrix (90° FOV)
     * @param {Object} colorTarget - Target texture object with .texture and .view
     * @param {Object} depthTarget - Depth texture object with .texture and .view
     * @param {number} faceIndex - Which face (0-5) for debugging
     * @param {vec3} position - Capture position
     */
    async _renderSceneForProbe(viewMatrix, projMatrix, colorTarget, depthTarget, faceIndex, position) {
        if (!this._lastRenderContext) {
            console.warn('RenderGraph: No render context available for probe capture')
            return
        }

        if (!this.probePasses.gbuffer || !this.probePasses.lighting) {
            console.warn('RenderGraph: Probe passes not initialized')
            return
        }

        const { device } = this.engine
        const { entityManager, assetManager } = this._lastRenderContext

        // CRITICAL: Use a SEPARATE meshes dictionary for probe capture
        // The main render loop (via requestAnimationFrame) can overwrite instance counts
        // in the shared meshes dictionary while probe capture is running
        if (!this._probeMeshes) {
            this._probeMeshes = {}
        }
        const meshes = this._probeMeshes

        // Build entity list for probe capture (ALL entities, no frustum culling)
        const probeEntities = []
        entityManager.forEach((id, entity) => {
            if (entity.model) {
                probeEntities.push({ id, entity, distance: 0 })
            }
        })
        const probeGroups = this.cullingSystem.groupByModel(probeEntities)

        // Update meshes dictionary with ALL entities for probe rendering
        this._updateMeshInstancesFromEntities(probeGroups, assetManager, meshes, true, null, 0, null)

        // Ensure all geometry buffers are written to GPU before rendering
        for (const name in meshes) {
            const mesh = meshes[name]
            if (mesh?.geometry?.update) {
                mesh.geometry.update()
            }
        }

        // Wait for GPU to complete buffer writes before rendering each face
        await device.queue.onSubmittedWorkDone()

        // Create inverse matrices for lighting calculations
        const iView = mat4.create()
        const iProj = mat4.create()
        const iViewProj = mat4.create()
        const viewProj = mat4.create()
        mat4.invert(iView, viewMatrix)
        mat4.invert(iProj, projMatrix)
        mat4.multiply(viewProj, projMatrix, viewMatrix)
        mat4.invert(iViewProj, viewProj)

        // Create a temporary camera-like object with the probe matrices
        // No jitter for probe capture - we want clean, stable environment maps
        const probeCamera = {
            view: viewMatrix,
            proj: projMatrix,
            iView,
            iProj,
            iViewProj,
            position: position || [0, 0, 0],
            near: 0.1,
            far: 10000,
            aspect: 1.0,
            jitterEnabled: false,
            jitterOffset: [0, 0],
            updateMatrix: () => {},
            updateView: () => {}
        }

        // Execute probe GBuffer pass (uses meshes with ALL entity instances)
        await this.probePasses.gbuffer.execute({
            camera: probeCamera,
            meshes,
            dt: 0
        })

        // Copy light data and environment from main lighting pass to probe pass
        this.probePasses.lighting.lights = this.passes.lighting.lights
        // Ensure probe lighting has current environment map with correct encoding
        if (this.environmentMap) {
            this.probePasses.lighting.setEnvironmentMap(this.environmentMap, this.environmentEncoding)
        }

        // Execute probe Lighting pass
        await this.probePasses.lighting.execute({
            camera: probeCamera,
            meshes,
            dt: 0,
            lights: this.passes.lighting.lights,
            mainLight: this.engine?.settings?.mainLight
        })

        // Copy lighting output to the probe face texture
        const lightingOutput = this.probePasses.lighting.getOutputTexture()
        const copySize = colorTarget.texture.width  // Get size from target texture

        if (lightingOutput && colorTarget.texture) {
            const commandEncoder = device.createCommandEncoder()
            commandEncoder.copyTextureToTexture(
                { texture: lightingOutput.texture },
                { texture: colorTarget.texture },
                { width: copySize, height: copySize }
            )
            device.queue.submit([commandEncoder.finish()])
        }

        // Copy GBuffer depth to face depth texture (for skybox depth testing)
        const gbufferDepth = this.probePasses.gbuffer.getGBuffer()?.depth
        if (gbufferDepth && depthTarget.texture) {
            const commandEncoder = device.createCommandEncoder()
            commandEncoder.copyTextureToTexture(
                { texture: gbufferDepth.texture },
                { texture: depthTarget.texture },
                { width: copySize, height: copySize }
            )
            device.queue.submit([commandEncoder.finish()])
        }
    }

    /**
     * Get culling system for external configuration
     */
    getCullingSystem() {
        return this.cullingSystem
    }

    /**
     * Get instance manager for stats
     */
    getInstanceManager() {
        return this.instanceManager
    }

    /**
     * Get sprite system for external access
     */
    getSpriteSystem() {
        return this.spriteSystem
    }

    /**
     * Get particle system for external access
     */
    getParticleSystem() {
        return this.particleSystem
    }

    /**
     * Get render stats
     */
    getStats() {
        return {
            ...this.stats,
            instance: this.instanceManager.getStats(),
            culling: this.cullingSystem.getStats(),
            occlusion: this.cullingSystem.getOcclusionStats()
        }
    }

    /**
     * Enable/disable a specific pass
     * @param {string} passName - Name of pass ('gbuffer', 'lighting', 'postProcess')
     * @param {boolean} enabled - Whether to enable
     */
    setPassEnabled(passName, enabled) {
        if (this.passes[passName]) {
            this.passes[passName].enabled = enabled
        }
    }

    /**
     * Get a specific pass for configuration
     * @param {string} passName - Name of pass
     * @returns {BasePass} The pass instance
     */
    getPass(passName) {
        return this.passes[passName]
    }

    /**
     * Invalidate occlusion culling data and reset warmup period.
     * Call this after scene loading or major camera changes to prevent
     * incorrect occlusion culling with stale depth buffer data.
     */
    invalidateOcclusionCulling() {
        if (this.passes.hiz) {
            this.passes.hiz.invalidate()
        }
    }

    /**
     * Load noise texture based on settings
     * Supports: 'bluenoise' (loaded from file), 'bayer8' (generated 8x8 ordered dither)
     */
    async _loadNoiseTexture() {
        const noiseSettings = this.engine?.settings?.noise || { type: 'bluenoise', animated: true }
        this.noiseAnimated = noiseSettings.animated !== false

        if (noiseSettings.type === 'bayer8') {
            // Create 8x8 Bayer ordered dither pattern
            this.noiseTexture = this._createBayerTexture()
            this.noiseSize = 8
            console.log('RenderGraph: Using Bayer 8x8 dither pattern')
        } else {
            // Default: load blue noise
            try {
                this.noiseTexture = await Texture.fromImage(this.engine, '/bluenoise.png', {
                    flipY: false,
                    srgb: false,  // Linear data
                    generateMips: false,
                    addressMode: 'repeat'  // Tile across screen
                })
                if (this.noiseTexture.width) {
                    this.noiseSize = this.noiseTexture.width
                }
            } catch (e) {
                console.warn('RenderGraph: Failed to load blue noise texture:', e)
            }
        }
    }

    /**
     * Create an 8x8 Bayer ordered dither texture
     * @returns {Object} Texture object with texture and view properties
     */
    _createBayerTexture() {
        const { device } = this.engine

        // Bayer 8x8 matrix (values 0-63, we normalize to 0-1)
        const bayer8x8 = [
             0, 32,  8, 40,  2, 34, 10, 42,
            48, 16, 56, 24, 50, 18, 58, 26,
            12, 44,  4, 36, 14, 46,  6, 38,
            60, 28, 52, 20, 62, 30, 54, 22,
             3, 35, 11, 43,  1, 33,  9, 41,
            51, 19, 59, 27, 49, 17, 57, 25,
            15, 47,  7, 39, 13, 45,  5, 37,
            63, 31, 55, 23, 61, 29, 53, 21
        ]

        // Create RGBA texture data (normalized to 0-255)
        const data = new Uint8Array(8 * 8 * 4)
        for (let i = 0; i < 64; i++) {
            const value = Math.round((bayer8x8[i] / 63) * 255)
            data[i * 4 + 0] = value  // R
            data[i * 4 + 1] = value  // G
            data[i * 4 + 2] = value  // B
            data[i * 4 + 3] = 255    // A
        }

        // Create GPU texture
        const texture = device.createTexture({
            label: 'Bayer 8x8 Dither',
            size: [8, 8, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        })

        device.queue.writeTexture(
            { texture },
            data,
            { bytesPerRow: 8 * 4 },
            { width: 8, height: 8 }
        )

        // Create sampler with repeat addressing for tiling
        const sampler = device.createSampler({
            label: 'Bayer 8x8 Sampler',
            addressModeU: 'repeat',
            addressModeV: 'repeat',
            magFilter: 'nearest',
            minFilter: 'nearest',
        })

        return {
            texture,
            view: texture.createView(),
            sampler,
            width: 8,
            height: 8
        }
    }

    /**
     * Reload noise texture and update all passes that use it
     * Called when noise settings change at runtime
     */
    async reloadNoiseTexture() {
        // Destroy old texture if it exists
        if (this.noiseTexture?.texture) {
            this.noiseTexture.texture.destroy()
        }

        // Load new noise texture based on current settings
        await this._loadNoiseTexture()

        // Update all passes that use noise
        if (this.passes.gbuffer) {
            this.passes.gbuffer.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        }
        if (this.passes.shadow) {
            this.passes.shadow.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        }
        if (this.passes.lighting) {
            this.passes.lighting.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        }
        if (this.passes.ao) {
            this.passes.ao.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        }
        if (this.passes.transparent) {
            this.passes.transparent.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        }
        if (this.passes.postProcess) {
            this.passes.postProcess.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        }
        if (this.passes.renderPost) {
            this.passes.renderPost.setNoise(this.noiseTexture, this.noiseSize, this.noiseAnimated)
        }

        console.log(`RenderGraph: Reloaded noise texture (${this.engine?.settings?.noise?.type || 'bluenoise'})`)
    }

    /**
     * Destroy all resources
     */
    destroy() {
        for (const passName in this.passes) {
            if (this.passes[passName]) {
                this.passes[passName].destroy()
            }
        }
        if (this.historyManager) {
            this.historyManager.destroy()
        }
        this.instanceManager.destroy()
        this.spriteSystem.destroy()
        this.particleSystem.destroy()
    }
}

export { RenderGraph }
