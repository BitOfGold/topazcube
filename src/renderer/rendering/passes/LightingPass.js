import { BasePass } from "./BasePass.js"
import { Pipeline } from "../../Pipeline.js"
import { Texture } from "../../Texture.js"
import { mat4, vec3 } from "../../math.js"
import { Frustum } from "../../utils/Frustum.js"

import lightingWGSL from "../shaders/lighting.wgsl"
import lightCullingWGSL from "../shaders/light_culling.wgsl"

/**
 * LightingPass - Tiled deferred lighting calculation
 *
 * Pass 6 in the 7-pass pipeline.
 * Uses compute shader to cull lights per tile, then fragment shader for lighting.
 *
 * Inputs: GBuffer (albedo, normal, ARM, emission, depth), environment map
 * Output: HDR lit image (rgba16float)
 */
class LightingPass extends BasePass {
    constructor(engine = null) {
        super('Lighting', engine)

        this.pipeline = null
        this.computePipeline = null
        this.outputTexture = null
        this.environmentMap = null
        this.gbuffer = null
        this.shadowPass = null

        // Light data
        this.lights = []

        // Noise texture for shadow jittering (blue noise or bayer dither)
        this.noiseTexture = null
        this.noiseSize = 64
        this.noiseAnimated = true

        // AO texture from AOPass
        this.aoTexture = null

        // Environment encoding: 0 = equirectangular (default), 1 = octahedral (for captured probes)
        this.envEncoding = 0

        // Exposure override for probe rendering (null = use settings, number = override)
        // When capturing probes, we want raw HDR values without exposure
        this.exposureOverride = null

        // Reflection mode: flips environment sampling Y for planar reflections
        this.reflectionMode = false

        // Ambient capture mode: disables IBL on geometry, only direct lights + emissive + skybox background
        this.ambientCaptureMode = false

        // Tiled lighting buffers
        this.lightBuffer = null        // Storage buffer for light data
        this.tileLightBuffer = null    // Storage buffer for per-tile light indices
        this.tileCountBuffer = null    // For debug: light counts per tile

        // Stats
        this.stats = {
            totalLights: 0,
            visibleLights: 0,
            pointLights: 0,
            spotLights: 0,
            culledByFrustum: 0,
            culledByDistance: 0,
            culledByOcclusion: 0
        }

        // HiZ pass reference for occlusion culling
        this.hizPass = null
    }

    /**
     * Set the HiZ pass for occlusion culling of lights
     * @param {HiZPass} hizPass - The HiZ pass instance
     */
    setHiZPass(hizPass) {
        this.hizPass = hizPass
    }

    // Convenience getters for lighting settings (with defaults for backward compatibility)
    get maxLights() { return this.settings?.lighting?.maxLights ?? 768 }
    get tileSize() { return this.settings?.lighting?.tileSize ?? 16 }
    get maxLightsPerTile() { return this.settings?.lighting?.maxLightsPerTile ?? 256 }
    get lightMaxDistance() { return this.settings?.lighting?.maxDistance ?? 240 }
    get lightCullingEnabled() { return this.settings?.lighting?.cullingEnabled ?? true }
    get shadowBias() { return this.settings?.shadow?.bias ?? 0.0005 }
    get shadowNormalBias() { return this.settings?.shadow?.normalBias ?? 0.015 }
    get shadowStrength() { return this.settings?.shadow?.strength ?? 1.0 }

    /**
     * Set the environment map for IBL
     * @param {Texture} envMap - HDR environment map
     * @param {number} encoding - 0 = equirectangular (default), 1 = octahedral (for captured probes)
     */
    setEnvironmentMap(envMap, encoding = 0) {
        this.environmentMap = envMap
        this.envEncoding = encoding
        this._needsRebuild = true
    }

    /**
     * Set the environment encoding type
     * @param {number} encoding - 0 = equirectangular (default), 1 = octahedral (for captured probes)
     */
    setEnvironmentEncoding(encoding) {
        this.envEncoding = encoding
    }

    /**
     * Set the GBuffer from GBufferPass
     * @param {GBuffer} gbuffer - GBuffer textures
     */
    async setGBuffer(gbuffer) {
        this.gbuffer = gbuffer
        this._needsRebuild = true
        this._computeBindGroupDirty = true
        // Create/recreate compute pipeline now that we have gbuffer
        if (!this.computePipeline && this.lightBuffer) {
            await this._createComputePipeline()
        }
    }

    /**
     * Set the shadow pass for shadow mapping
     * @param {ShadowPass} shadowPass - Shadow pass instance
     */
    setShadowPass(shadowPass) {
        this.shadowPass = shadowPass
        this._needsRebuild = true
    }

    /**
     * Set the noise texture for shadow jittering
     * @param {Texture} noise - Noise texture (blue noise or bayer dither)
     * @param {number} size - Texture size (assumed square)
     * @param {boolean} animated - Whether to animate noise offset each frame
     */
    setNoise(noise, size = 64, animated = true) {
        this.noiseTexture = noise
        this.noiseSize = size
        this.noiseAnimated = animated
        this._needsRebuild = true
    }

    /**
     * Set the AO texture from AOPass
     * @param {Texture} aoTexture - AO texture (r8unorm)
     */
    setAOTexture(aoTexture) {
        this.aoTexture = aoTexture
        this._needsRebuild = true
    }

    async _init() {
        const { canvas, device } = this.engine

        // Create output texture (HDR)
        this.outputTexture = await Texture.renderTarget(this.engine, 'rgba16float')

        // Initialize tiled lighting resources
        await this._initTiledLighting(canvas.width, canvas.height)
    }

    /**
     * Initialize tiled lighting compute resources
     */
    async _initTiledLighting(width, height) {
        const { device } = this.engine

        // Calculate tile counts
        this.tileCountX = Math.ceil(width / this.tileSize)
        this.tileCountY = Math.ceil(height / this.tileSize)
        const totalTiles = this.tileCountX * this.tileCountY

        // Light buffer: store all light data for GPU access
        // Each light: 96 bytes to match WGSL storage buffer alignment
        const lightBufferSize = this.maxLights * 96
        this.lightBuffer = device.createBuffer({
            label: 'Light Buffer',
            size: lightBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.lightBufferData = new ArrayBuffer(lightBufferSize)

        // Tile light indices buffer: for each tile, store count + up to maxLightsPerTile indices
        // Each entry is a u32 (4 bytes)
        const tileLightBufferSize = totalTiles * (this.maxLightsPerTile + 1) * 4
        this.tileLightBuffer = device.createBuffer({
            label: 'Tile Light Indices',
            size: tileLightBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })

        // Uniform buffer for compute shader
        // viewMatrix(64) + projectionMatrix(64) + inverseProjection(64) + screenSize(8) + tileCount(8) + lightCount(4) + near(4) + far(4) + padding(4) = 224 bytes
        this.cullUniformBuffer = device.createBuffer({
            label: 'Light Cull Uniforms',
            size: 224,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // Create compute pipeline for light culling
        await this._createComputePipeline()

    }

    /**
     * Create compute pipeline for light culling
     */
    async _createComputePipeline() {
        const { device } = this.engine

        if (!this.gbuffer) return

        const computeModule = device.createShaderModule({
            label: 'Light Culling Compute',
            code: lightCullingWGSL,
        })

        const computeBindGroupLayout = device.createBindGroupLayout({
            label: 'Light Culling Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
            ],
        })

        // Use async pipeline creation for non-blocking initialization
        this.computePipeline = await device.createComputePipelineAsync({
            label: 'Light Culling Pipeline',
            layout: device.createPipelineLayout({
                bindGroupLayouts: [computeBindGroupLayout],
            }),
            compute: {
                module: computeModule,
                entryPoint: 'main',
            },
        })

        this.computeBindGroupLayout = computeBindGroupLayout
        this._computeBindGroupDirty = true
    }

    /**
     * Create or update compute bind group
     */
    _updateComputeBindGroup() {
        const { device } = this.engine

        if (!this.gbuffer || !this.computePipeline) return

        this.computeBindGroup = device.createBindGroup({
            label: 'Light Culling Bind Group',
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cullUniformBuffer } },
                { binding: 1, resource: { buffer: this.lightBuffer } },
                { binding: 2, resource: { buffer: this.tileLightBuffer } },
                { binding: 3, resource: this.gbuffer.depth.view },
            ],
        })

        this._computeBindGroupDirty = false
    }

    /**
     * Build or rebuild the fragment pipeline
     */
    async _buildPipeline() {
        if (!this.gbuffer || !this.environmentMap) {
            console.warn('LightingPass: Missing gbuffer or environmentMap')
            return
        }

        // Check all required textures
        if (!this.noiseTexture) {
            console.warn('LightingPass: Missing noiseTexture')
            return
        }
        if (!this.aoTexture) {
            console.warn('LightingPass: Missing aoTexture')
            return
        }

        this.pipeline = await Pipeline.create(this.engine, {
            label: 'lighting',
            wgslSource: lightingWGSL,
            isPostProcessing: true,
            textures: [this.gbuffer, this.environmentMap, this.noiseTexture, this.aoTexture],
            renderTarget: this.outputTexture,
            shadowPass: this.shadowPass,
            tileLightBuffer: this.tileLightBuffer,
            lightBuffer: this.lightBuffer,
            tileSize: this.tileSize,
            tileCountX: this.tileCountX,
            maxLightsPerTile: this.maxLightsPerTile,
        })

        this._needsRebuild = false
    }

    /**
     * Add a light to the scene
     * @param {Object} light - Light data
     */
    addLight(light) {
        if (this.lights.length < this.maxLights) {
            this.lights.push({
                enabled: light.enabled !== false ? 1 : 0,
                position: light.position || [0, 0, 0],
                color: light.color || [1, 1, 1, 1],
                direction: light.direction || [0, -1, 0],
                geom: light.geom || [10, 0.3, 0.5, 0], // radius, innerCone, outerCone
                lightType: light.lightType || 0, // 0=dir, 1=point, 2=spot
                shadowIndex: light.shadowIndex || -1
            })
        }
    }

    /**
     * Clear all lights
     */
    clearLights() {
        this.lights = []
    }

    /**
     * Update lights from entity system with frustum culling and distance ordering
     * @param {Array} lightEntities - Array of entities with lights
     * @param {Camera} camera - Camera for frustum culling and distance calculation
     */
    updateLightsFromEntities(lightEntities, camera) {
        this.clearLights()

        // Reset stats
        this.stats.totalLights = 0
        this.stats.visibleLights = 0
        this.stats.culledByFrustum = 0
        this.stats.culledByDistance = 0
        this.stats.culledByOcclusion = 0

        // Separate lights by type for different handling
        const spotlights = []
        const pointLights = []

        // Create camera frustum for culling
        let cameraFrustum = null
        if (camera && this.lightCullingEnabled) {
            cameraFrustum = new Frustum()
            cameraFrustum.update(
                camera.view,
                camera.proj,
                camera.position,
                camera.direction,
                camera.fov || Math.PI / 4,
                camera.aspect || 1.0,
                camera.near || 0.1,
                camera.far || 1000
            )
        }

        for (const { id, entity } of lightEntities) {
            if (!entity.light || !entity.light.enabled) continue

            const light = entity.light
            this.stats.totalLights++

            // Transform light position to world space
            const worldPos = [
                entity.position[0] + (light.position?.[0] || 0),
                entity.position[1] + (light.position?.[1] || 0),
                entity.position[2] + (light.position?.[2] || 0)
            ]

            const lightData = {
                enabled: true,
                position: worldPos,
                direction: light.direction || [0, -1, 0],
                color: light.color || [1, 1, 1, 1],
                geom: [...(light.geom || [10, 0.3, 0.5, 0])], // Copy array to avoid modifying original
                lightType: light.lightType || 1
            }

            const lightRadius = lightData.geom[0] || 10

            // Calculate distance and fade for ALL light types (including spotlights)
            let distance = 0
            let distanceFade = 1.0
            if (camera) {
                const dx = worldPos[0] - camera.position[0]
                const dy = worldPos[1] - camera.position[1]
                const dz = worldPos[2] - camera.position[2]
                distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

                let maxDist = lightData.geom[0] < 5 ? this.lightMaxDistance * 0.25 : this.lightMaxDistance
                // Distance cull: skip if light is too far (accounting for light radius)
                const effectiveDistance = distance - lightRadius
                if (effectiveDistance > maxDist) {
                    this.stats.culledByDistance++
                    continue
                }

                // Calculate fade: 1.0 at 80% distance, 0.0 at 100% distance (smooth fade to avoid popping)
                const fadeStart = maxDist * 0.8
                if (effectiveDistance > fadeStart) {
                    distanceFade = 1.0 - (effectiveDistance - fadeStart) / (maxDist - fadeStart)
                    distanceFade = Math.max(0, Math.min(1, distanceFade))
                }
            }

            // Store distance fade in geom.w (will be applied in shader)
            lightData.geom[3] = distanceFade

            // Skip lights that are nearly invisible (faded out)
            if (distanceFade <= 0.01) {
                this.stats.culledByDistance++
                continue
            }

            // Create bounding sphere for culling tests
            const lightBsphere = { center: worldPos, radius: lightRadius }

            // Spotlights
            if (lightData.lightType === 2) {
                // Frustum cull spotlights
                if (cameraFrustum && this.lightCullingEnabled) {
                    if (!cameraFrustum.testSpherePlanes(lightBsphere)) {
                        this.stats.culledByFrustum++
                        continue
                    }
                }

                // HiZ occlusion cull spotlights
                if (this.hizPass && camera && this.settings?.occlusionCulling?.enabled) {
                    if (this.hizPass.testSphereOcclusion(lightBsphere, camera.viewProj, camera.near, camera.far, camera.position)) {
                        this.stats.culledByOcclusion++
                        continue
                    }
                }

                lightData._distance = distance
                spotlights.push(lightData)
                continue
            }

            // Point lights: apply frustum culling
            if (cameraFrustum && this.lightCullingEnabled) {
                if (!cameraFrustum.testSpherePlanes(lightBsphere)) {
                    this.stats.culledByFrustum++
                    continue
                }
            }

            // HiZ occlusion cull point lights
            if (this.hizPass && camera && this.settings?.occlusionCulling?.enabled) {
                if (this.hizPass.testSphereOcclusion(lightBsphere, camera.viewProj, camera.near, camera.far, camera.position)) {
                    this.stats.culledByOcclusion++
                    continue
                }
            }

            // Store distance for sorting
            lightData._distance = distance
            pointLights.push(lightData)
        }

        // Sort point lights by distance (closest first)
        pointLights.sort((a, b) => a._distance - b._distance)

        // Add spotlights first (they need consistent indices for shadow mapping)
        for (const light of spotlights) {
            this.addLight(light)
        }
        this.stats.spotLights = spotlights.length

        // Add point lights (closest first, up to remaining capacity)
        let addedPointLights = 0
        for (const light of pointLights) {
            if (this.lights.length >= this.maxLights) break
            this.addLight(light)
            addedPointLights++
        }
        this.stats.pointLights = addedPointLights

        this.stats.visibleLights = this.lights.length

    }

    async _execute(context) {
        const { device, canvas } = this.engine
        const { camera, lights: contextLights, mainLight } = context

        // Use output texture size (not canvas size) for proper probe rendering
        const targetWidth = this.outputTexture?.texture?.width || canvas.width
        const targetHeight = this.outputTexture?.texture?.height || canvas.height

        // Get environment settings from engine (with fallbacks)
        // In ambient capture mode (emissive only): disable ALL lighting, only emissive + skybox
        const ambientColor = this.ambientCaptureMode
            ? [0, 0, 0, 0]  // No ambient in emissive-only mode
            : (this.settings?.environment?.ambientColor ?? [0.7, 0.75, 0.9, 0.2])
        const environmentDiffuse = this.ambientCaptureMode ? 0.0 : (this.settings?.environment?.diffuse ?? 4.0)
        const environmentSpecular = this.ambientCaptureMode ? 0.0 : (this.settings?.environment?.specular ?? 4.0)
        // Use override if set (for probe capture), otherwise use settings
        const exposure = this.exposureOverride ?? this.settings?.environment?.exposure ?? 1.6

        // Rebuild pipeline if needed
        if (this._needsRebuild) {
            await this._buildPipeline()
        }

        // If rebuild was attempted but failed, don't use stale pipeline with old bind groups
        if (!this.pipeline || this._needsRebuild) {
            return
        }

        // Update compute bind group if needed
        if (this._computeBindGroupDirty && this.gbuffer) {
            this._updateComputeBindGroup()
        }

        // ===================
        // LIGHT CULLING COMPUTE PASS
        // ===================
        // Skip light culling if:
        // - In ambient capture mode (emissive only - no point/spot lights)
        // - Light culling is disabled in settings
        // - No lights to process
        const shouldRunLightCulling = this.computePipeline &&
                                       this.computeBindGroup &&
                                       this.lights.length > 0 &&
                                       !this.ambientCaptureMode &&
                                       this.lightCullingEnabled

        if (shouldRunLightCulling) {
            // Update light buffer with current lights data
            this._updateLightBuffer()

            // Update cull uniforms
            const cullUniformData = new Float32Array(56) // 224 bytes / 4
            cullUniformData.set(camera.view, 0)        // viewMatrix (16 floats)
            cullUniformData.set(camera.proj, 16)       // projectionMatrix (16 floats)
            cullUniformData.set(camera.iProj || mat4.create(), 32) // inverseProjection (16 floats)
            cullUniformData[48] = targetWidth          // screenSize.x
            cullUniformData[49] = targetHeight         // screenSize.y
            const cullUniformDataU32 = new Uint32Array(cullUniformData.buffer)
            cullUniformDataU32[50] = this.tileCountX   // tileCount.x
            cullUniformDataU32[51] = this.tileCountY   // tileCount.y
            cullUniformDataU32[52] = this.lights.length // lightCount
            cullUniformData[53] = camera.near || 0.1   // nearPlane from camera
            cullUniformData[54] = camera.far || 10000  // farPlane from camera (use large default)
            cullUniformData[55] = 0                    // padding

            device.queue.writeBuffer(this.cullUniformBuffer, 0, cullUniformData)

            // Run compute pass
            const computeEncoder = device.createCommandEncoder({ label: 'Light Culling' })
            const computePass = computeEncoder.beginComputePass({ label: 'Light Culling Pass' })

            computePass.setPipeline(this.computePipeline)
            computePass.setBindGroup(0, this.computeBindGroup)
            computePass.dispatchWorkgroups(this.tileCountX, this.tileCountY, 1)
            computePass.end()

            device.queue.submit([computeEncoder.finish()])
        }

        // ===================
        // LIGHTING FRAGMENT PASS
        // ===================

        // Get main light settings (use defaults if not provided)
        // In ambient capture mode (emissive only): disable main light
        const mainLightEnabled = this.ambientCaptureMode ? false : (mainLight?.enabled !== false)
        const mainLightIntensity = mainLight?.intensity ?? 1.0
        const mainLightColor = mainLight?.color || [1.0, 0.95, 0.9]
        const lightDir = vec3.fromValues(
            mainLight?.direction?.[0] ?? -1.0,
            mainLight?.direction?.[1] ?? 1.0,
            mainLight?.direction?.[2] ?? -0.5
        )
        vec3.normalize(lightDir, lightDir)

        // Get shadow info
        let shadowMapSize = 2048
        if (this.shadowPass) {
            shadowMapSize = this.shadowPass.shadowMapSize

        }

        // Get cascade sizes from shadow pass
        const cascadeSizes = this.shadowPass ? this.shadowPass.getCascadeSizes() : [50, 200, 1000]

        // Noise offset (0..1) - random each frame if animated, 0 if static
        const noiseOffsetX = this.noiseAnimated ? Math.random() : 0
        const noiseOffsetY = this.noiseAnimated ? Math.random() : 0

        // Set uniforms
        this.pipeline.uniformValues.set({
            inverseViewProjection: camera.iViewProj,
            inverseProjection: camera.iProj,
            inverseView: camera.iView,
            cameraPosition: camera.position,
            canvasSize: [targetWidth, targetHeight],
            lightDir: lightDir,
            lightColor: [
                mainLightColor[0],
                mainLightColor[1],
                mainLightColor[2],
                mainLightEnabled ? mainLightIntensity * 12.0 : 0.0
            ],
            ambientColor: ambientColor,
            environmentParams: [
                environmentDiffuse,
                environmentSpecular,
                this.environmentMap?.mipCount || 1,
                exposure
            ],
            shadowParams: [
                this.shadowBias,
                this.shadowNormalBias,
                this.shadowStrength,
                shadowMapSize
            ],
            cascadeSizes: [cascadeSizes[0], cascadeSizes[1], cascadeSizes[2], 0],
            tileParams: [this.tileSize, this.tileCountX, this.maxLightsPerTile, this.ambientCaptureMode ? 0 : this.lights.length],
            noiseParams: [this.noiseSize, noiseOffsetX, noiseOffsetY, this.envEncoding],
            cameraParams: [camera.near || 0.05, camera.far || 1000, this.reflectionMode ? 1.0 : 0.0, this.settings?.lighting?.directSpecularMultiplier ?? 3.0],
            specularBoost: [
                this.settings?.lighting?.specularBoost ?? 0.0,
                this.settings?.lighting?.specularBoostRoughnessCutoff ?? 0.5,
                0.0,
                0.0
            ],
        })

        // Render lighting pass
        this.pipeline.render()
    }

    /**
     * Update light buffer with current lights data
     * Must match WGSL struct Light alignment in storage buffer:
     * - enabled: u32 at offset 0
     * - position: vec3f at offset 16 (aligned to 16)
     * - color: vec4f at offset 32
     * - direction: vec3f at offset 48
     * - geom: vec4f at offset 64
     * - shadowIndex: i32 at offset 80
     * - struct size: 96 bytes (24 floats)
     */
    _updateLightBuffer() {
        const { device } = this.engine

        // Get spotlight shadow slot assignments
        let spotShadowSlots = null
        if (this.shadowPass) {
            spotShadowSlots = this.shadowPass.getSpotShadowSlots()
        }

        // Each light is 96 bytes (24 floats) to match WGSL storage buffer alignment
        const lightData = new Float32Array(this.maxLights * 24)
        const lightDataU32 = new Uint32Array(lightData.buffer)
        const lightDataI32 = new Int32Array(lightData.buffer)

        for (let i = 0; i < this.maxLights; i++) {
            const light = this.lights[i]
            const offset = i * 24

            if (light) {
                // enabled: u32 at offset 0
                lightDataU32[offset + 0] = light.enabled ? 1 : 0
                // padding: 12 bytes (3 floats)

                // position: vec3f at offset 16 (4 floats)
                lightData[offset + 4] = light.position[0]
                lightData[offset + 5] = light.position[1]
                lightData[offset + 6] = light.position[2]
                // padding: 4 bytes

                // color: vec4f at offset 32 (8 floats)
                lightData[offset + 8] = light.color[0]
                lightData[offset + 9] = light.color[1]
                lightData[offset + 10] = light.color[2]
                lightData[offset + 11] = light.color[3]

                // direction: vec3f at offset 48 (12 floats)
                lightData[offset + 12] = light.direction[0]
                lightData[offset + 13] = light.direction[1]
                lightData[offset + 14] = light.direction[2]
                // padding: 4 bytes

                // geom: vec4f at offset 64 (16 floats)
                lightData[offset + 16] = light.geom[0]
                lightData[offset + 17] = light.geom[1]
                lightData[offset + 18] = light.geom[2]
                lightData[offset + 19] = light.geom[3]

                // shadowIndex: i32 at offset 80 (20 floats)
                const shadowIndex = spotShadowSlots ? (spotShadowSlots[i] !== undefined ? spotShadowSlots[i] : -1) : -1
                lightDataI32[offset + 20] = shadowIndex
                // padding: 12 bytes to reach 96
            } else {
                lightDataU32[offset] = 0 // disabled
            }
        }

        device.queue.writeBuffer(this.lightBuffer, 0, lightData)
    }

    async _resize(width, height) {
        // Recreate output texture at new size
        this.outputTexture = await Texture.renderTarget(this.engine, 'rgba16float', width, height)

        // Recreate tiled lighting buffers for new size
        await this._initTiledLighting(width, height)

        this._needsRebuild = true
        this._computeBindGroupDirty = true
    }

    _destroy() {
        this.pipeline = null
        this.outputTexture = null
    }

    /**
     * Get the output texture for use by subsequent passes
     */
    getOutputTexture() {
        return this.outputTexture
    }
}

export { LightingPass }
