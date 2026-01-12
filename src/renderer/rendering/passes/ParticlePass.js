import { BasePass } from "./BasePass.js"
import { Texture } from "../../Texture.js"

import particleSimulateWGSL from "../shaders/particle_simulate.wgsl"
import particleRenderWGSL from "../shaders/particle_render.wgsl"

/**
 * ParticlePass - GPU particle simulation and rendering
 *
 * Handles:
 * - Compute shader dispatch for particle spawning and simulation
 * - Forward rendering with additive or alpha blending
 * - Soft particle depth fade
 * - Billboard rendering using camera vectors
 */
class ParticlePass extends BasePass {
    constructor(engine = null) {
        super('Particles', engine)

        // Compute pipelines
        this.spawnPipeline = null
        this.simulatePipeline = null
        this.resetPipeline = null

        // Render pipelines (by blend mode)
        this.renderPipelineAdditive = null
        this.renderPipelineAlpha = null

        // Bind groups
        this.computeBindGroupLayout = null
        this.renderBindGroupLayout = null

        // Uniform buffers
        this.simulationUniformBuffer = null
        this.renderUniformBuffer = null
        this.emitterSettingsBuffer = null
        this.emitterRenderSettingsBuffer = null  // For lit + emissive per emitter

        // References
        this.particleSystem = null
        this.gbuffer = null
        this.outputTexture = null
        this.shadowPass = null  // For shadow maps
        this.environmentMap = null  // For IBL
        this.environmentEncoding = 0  // 0=equirect, 1=octahedral
        this.lightingPass = null  // For point/spot lights buffer

        // Per-emitter texture cache
        this._textureCache = new Map()  // textureUrl -> {texture, bindGroup}

        // Placeholder resources
        this._placeholderTexture = null
        this._placeholderSampler = null

        // Frame timing
        this._lastTime = 0
        this._deltaTime = 0
    }

    /**
     * Set particle system reference
     */
    setParticleSystem(particleSystem) {
        this.particleSystem = particleSystem
    }

    /**
     * Set GBuffer for depth testing
     */
    setGBuffer(gbuffer) {
        this.gbuffer = gbuffer
    }

    /**
     * Set output texture to render onto
     */
    setOutputTexture(texture) {
        this.outputTexture = texture
    }

    /**
     * Set shadow pass for accessing shadow maps
     */
    setShadowPass(shadowPass) {
        this.shadowPass = shadowPass
    }

    /**
     * Set environment map for IBL lighting
     * @param {Texture} envMap - Environment map texture
     * @param {number} encoding - 0=equirectangular, 1=octahedral
     */
    setEnvironmentMap(envMap, encoding = 0) {
        this.environmentMap = envMap
        this.environmentEncoding = encoding
    }

    /**
     * Set lighting pass for accessing point/spot lights buffer
     * @param {LightingPass} lightingPass
     */
    setLightingPass(lightingPass) {
        this.lightingPass = lightingPass
    }

    async _init() {
        const { device } = this.engine

        // Create uniform buffers
        // SimulationUniforms: dt, time, maxParticles, emitterCount + lighting params
        // dt(f32) + time(f32) + maxParticles(u32) + emitterCount(u32) = 16 bytes
        // cameraPosition(vec3f) + shadowBias(f32) = 16 bytes
        // lightDir(vec3f) + shadowStrength(f32) = 16 bytes
        // lightColor(vec4f) = 16 bytes
        // ambientColor(vec4f) = 16 bytes
        // cascadeSizes(vec4f) = 16 bytes
        // lightCount(u32) + pad(3xu32) = 16 bytes
        // Total: 112 bytes
        this.simulationUniformBuffer = device.createBuffer({
            size: 112,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Particle Simulation Uniforms'
        })

        this.renderUniformBuffer = device.createBuffer({
            size: 368,  // ParticleUniforms struct (increased for lighting + IBL + light count)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Particle Render Uniforms'
        })

        // Emitter settings buffer: 16 emitters * 48 bytes each = 768 bytes
        this.emitterSettingsBuffer = device.createBuffer({
            size: 16 * 48,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Particle Emitter Settings'
        })

        // Emitter render settings buffer: 16 emitters * 16 bytes each = 256 bytes
        // Contains: lit, emissive, softness, zOffset per emitter
        this.emitterRenderSettingsBuffer = device.createBuffer({
            size: 16 * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Particle Emitter Render Settings'
        })

        // Create placeholder resources
        this._placeholderTexture = await Texture.fromRGBA(this.engine, 1, 1, 1, 1)
        this._placeholderSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear'
        })

        // Placeholder comparison sampler for shadow sampling
        this._placeholderComparisonSampler = device.createSampler({
            compare: 'less',
            magFilter: 'linear',
            minFilter: 'linear'
        })

        // Placeholder depth texture array for shadow maps (when shadows disabled)
        this._placeholderDepthTexture = device.createTexture({
            size: [1, 1, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
        })
        this._placeholderDepthView = this._placeholderDepthTexture.createView({
            dimension: '2d-array',
            arrayLayerCount: 1
        })

        // Placeholder buffer for cascade matrices (3 mat4 = 192 bytes)
        this._placeholderMatricesBuffer = device.createBuffer({
            size: 192,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Placeholder Cascade Matrices'
        })

        // Placeholder lights buffer (64 lights * 64 bytes each = 4096 bytes)
        // Light struct: enabled(u32) + position(vec3f) + color(vec4f) + direction(vec3f) + geom(vec4f) + shadowIndex(i32)
        this._placeholderLightsBuffer = device.createBuffer({
            size: 64 * 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Placeholder Lights Buffer'
        })

        // Placeholder spot shadow atlas (1x1 depth texture)
        this._placeholderSpotShadowTexture = device.createTexture({
            size: [1, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
        })
        this._placeholderSpotShadowView = this._placeholderSpotShadowTexture.createView()

        // Placeholder spot shadow matrices buffer (8 mat4 = 512 bytes)
        this._placeholderSpotMatricesBuffer = device.createBuffer({
            size: 8 * 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Placeholder Spot Shadow Matrices'
        })

        // Create compute pipelines
        await this._createComputePipelines()

        // Create render pipelines
        await this._createRenderPipelines()
    }

    async _createComputePipelines() {
        const { device } = this.engine

        // Compute shader module
        const computeModule = device.createShaderModule({
            label: 'Particle Compute Shader',
            code: particleSimulateWGSL
        })

        // Bind group layout for compute (includes lighting bindings for shadow sampling)
        this.computeBindGroupLayout = device.createBindGroupLayout({
            label: 'Particle Compute BindGroup Layout',
            entries: [
                // Core bindings
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                // Lighting bindings
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // emitterRenderSettings
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth', viewDimension: '2d-array' } }, // shadowMapArray
                { binding: 7, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'comparison' } }, // shadowSampler
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cascadeMatrices
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // lights
                { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } }, // spotShadowAtlas
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // spotMatrices
            ]
        })

        const computePipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.computeBindGroupLayout]
        })

        // Spawn pipeline
        this.spawnPipeline = await device.createComputePipelineAsync({
            label: 'Particle Spawn Pipeline',
            layout: computePipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: 'spawn'
            }
        })

        // Simulate pipeline
        this.simulatePipeline = await device.createComputePipelineAsync({
            label: 'Particle Simulate Pipeline',
            layout: computePipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: 'simulate'
            }
        })

        // Reset counters pipeline
        this.resetPipeline = await device.createComputePipelineAsync({
            label: 'Particle Reset Pipeline',
            layout: computePipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: 'resetCounters'
            }
        })
    }

    async _createRenderPipelines() {
        const { device } = this.engine

        // Render shader module
        const renderModule = device.createShaderModule({
            label: 'Particle Render Shader',
            code: particleRenderWGSL
        })

        // Bind group layout for rendering
        this.renderBindGroupLayout = device.createBindGroupLayout({
            label: 'Particle Render BindGroup Layout',
            entries: [
                // Uniforms
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                // Particle storage buffer (read only in render)
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                // Particle texture
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                // Depth texture for soft particles (GBuffer depth32float)
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
                // Shadow map array (cascade shadows)
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
                // Shadow comparison sampler
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
                // Cascade matrices (storage buffer)
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                // Emitter render settings (lit, emissive, softness, zOffset)
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                // Environment map for IBL
                { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 10, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                // Point/spot lights buffer
                { binding: 11, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                // Spot shadow atlas
                { binding: 12, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
                { binding: 13, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
                // Spot shadow matrices
                { binding: 14, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            ]
        })

        const renderPipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.renderBindGroupLayout]
        })

        // Additive blend pipeline
        this.renderPipelineAdditive = await device.createRenderPipelineAsync({
            label: 'Particle Render Pipeline (Additive)',
            layout: renderPipelineLayout,
            vertex: {
                module: renderModule,
                entryPoint: 'vertexMain',
                buffers: []  // All data from storage buffer
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fragmentMainAdditive',
                targets: [{
                    format: 'rgba16float',
                    blend: {
                        color: {
                            srcFactor: 'one',  // Additive: src + dst
                            dstFactor: 'one',
                            operation: 'add'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one',
                            operation: 'add'
                        }
                    }
                }]
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: false,  // No depth write for particles
                depthCompare: 'less-equal',  // Standard depth test with linear depth
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            }
        })

        // Alpha blend pipeline
        this.renderPipelineAlpha = await device.createRenderPipelineAsync({
            label: 'Particle Render Pipeline (Alpha)',
            layout: renderPipelineLayout,
            vertex: {
                module: renderModule,
                entryPoint: 'vertexMain',
                buffers: []
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fragmentMainAlpha',
                targets: [{
                    format: 'rgba16float',
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        }
                    }
                }]
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: false,
                depthCompare: 'less-equal',  // Standard depth test with linear depth
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            }
        })
    }

    async _execute(context) {
        const { device, canvas, stats } = this.engine
        const { camera, time, mainLight } = context

        if (!this.particleSystem || !this.outputTexture || !this.gbuffer) {
            return
        }

        // Initialize particle system if needed
        await this.particleSystem.init()

        // Calculate delta time
        const currentTime = time ?? (performance.now() / 1000)
        this._deltaTime = this._lastTime > 0 ? currentTime - this._lastTime : 0.016
        this._lastTime = currentTime

        // Clamp delta time to prevent huge jumps
        this._deltaTime = Math.min(this._deltaTime, 0.1)

        // Update particle system (CPU side - spawn rate accumulation)
        this.particleSystem.update(this._deltaTime)

        const emitters = this.particleSystem.getActiveEmitters()
        if (emitters.length === 0) {
            return
        }

        // Create command encoder
        const commandEncoder = device.createCommandEncoder({ label: 'Particle Pass' })

        // === COMPUTE PHASE (includes lighting calculation) ===
        await this._executeCompute(commandEncoder, emitters, camera, mainLight)

        // === RENDER PHASE ===
        await this._executeRender(commandEncoder, emitters, camera, mainLight)

        // Submit
        device.queue.submit([commandEncoder.finish()])

        // Update stats
        if (stats) {
            stats.particleEmitters = emitters.length
            stats.particleCount = this.particleSystem.getTotalParticleCount()
        }
    }

    async _executeCompute(commandEncoder, emitters, camera, mainLight) {
        const { device } = this.engine

        // Get spawn count BEFORE executing (executeSpawn clears the queue)
        const spawnCount = Math.min(this.particleSystem._spawnQueue?.length || 0, 1000)

        // Execute spawns - writes data to GPU buffers
        this.particleSystem.executeSpawn(commandEncoder)

        // Get lighting settings
        const mainLightEnabled = mainLight?.enabled !== false
        const mainLightIntensity = mainLight?.intensity ?? 1.0
        const mainLightColor = mainLight?.color || [1.0, 0.95, 0.9]
        const lightDir = mainLight?.direction || [-1, 1, -0.5]

        // Normalize light direction
        const lightDirLen = Math.sqrt(lightDir[0] * lightDir[0] + lightDir[1] * lightDir[1] + lightDir[2] * lightDir[2])
        const normalizedLightDir = [
            lightDir[0] / lightDirLen,
            lightDir[1] / lightDirLen,
            lightDir[2] / lightDirLen
        ]

        // Get ambient and shadow settings
        const ambientColor = this.settings?.ambient?.color || [0.1, 0.12, 0.15]
        const ambientIntensity = this.settings?.ambient?.intensity ?? 0.3
        const shadowConfig = this.settings?.shadow || {}
        const shadowBias = shadowConfig.bias ?? 0.0005
        const shadowStrength = shadowConfig.strength ?? 0.8
        const cascadeSizes = shadowConfig.cascadeSizes || [10, 25, 100]
        const lightCount = this.lightingPass?.lights?.length ?? 0

        // Update simulation uniforms (28 floats = 112 bytes)
        const uniformData = new Float32Array(28)
        uniformData[0] = this._deltaTime  // dt
        uniformData[1] = this._lastTime   // time
        const uniformIntView = new Uint32Array(uniformData.buffer)
        uniformIntView[2] = this.particleSystem.globalMaxParticles  // maxParticles
        uniformIntView[3] = emitters.length  // emitterCount
        // cameraPosition (vec3f) + shadowBias (f32) - floats 4-7
        uniformData[4] = camera.position[0]
        uniformData[5] = camera.position[1]
        uniformData[6] = camera.position[2]
        uniformData[7] = shadowBias
        // lightDir (vec3f) + shadowStrength (f32) - floats 8-11
        uniformData[8] = normalizedLightDir[0]
        uniformData[9] = normalizedLightDir[1]
        uniformData[10] = normalizedLightDir[2]
        uniformData[11] = shadowStrength
        // lightColor (vec4f) - floats 12-15
        uniformData[12] = mainLightColor[0]
        uniformData[13] = mainLightColor[1]
        uniformData[14] = mainLightColor[2]
        uniformData[15] = mainLightEnabled ? mainLightIntensity * 12.0 : 0.0
        // ambientColor (vec4f) - floats 16-19
        uniformData[16] = ambientColor[0]
        uniformData[17] = ambientColor[1]
        uniformData[18] = ambientColor[2]
        uniformData[19] = ambientIntensity
        // cascadeSizes (vec4f) - floats 20-23
        uniformData[20] = cascadeSizes[0]
        uniformData[21] = cascadeSizes[1]
        uniformData[22] = cascadeSizes[2]
        uniformData[23] = 0.0  // padding
        // lightCount (u32) + padding - floats 24-27
        uniformIntView[24] = lightCount
        uniformIntView[25] = 0
        uniformIntView[26] = 0
        uniformIntView[27] = 0

        device.queue.writeBuffer(this.simulationUniformBuffer, 0, uniformData)

        // Update per-emitter settings buffer
        // EmitterSettings: gravity(vec3f), drag, turbulence, fadeIn, fadeOut, rotationSpeed, startSize, endSize, baseAlpha, padding
        // = 12 floats = 48 bytes per emitter
        const emitterData = new Float32Array(16 * 12)  // 16 emitters max
        for (let i = 0; i < Math.min(emitters.length, 16); i++) {
            const e = emitters[i]
            const offset = i * 12
            // gravity (vec3f) + drag (f32)
            emitterData[offset + 0] = e.gravity[0]
            emitterData[offset + 1] = e.gravity[1]
            emitterData[offset + 2] = e.gravity[2]
            emitterData[offset + 3] = e.drag
            // turbulence + fadeIn + fadeOut + rotationSpeed
            emitterData[offset + 4] = e.turbulence
            emitterData[offset + 5] = e.fadeIn
            emitterData[offset + 6] = e.fadeOut
            emitterData[offset + 7] = e.rotationSpeed ?? 0.5
            // startSize + endSize + baseAlpha + padding
            emitterData[offset + 8] = e.size[0]
            emitterData[offset + 9] = e.size[1]
            emitterData[offset + 10] = e.color[3]  // baseAlpha
            emitterData[offset + 11] = 0  // padding
        }

        device.queue.writeBuffer(this.emitterSettingsBuffer, 0, emitterData)

        // Create compute bind group with lighting resources
        const computeBindGroup = device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                // Core bindings
                { binding: 0, resource: { buffer: this.simulationUniformBuffer } },
                { binding: 1, resource: { buffer: this.particleSystem.getParticleBuffer() } },
                { binding: 2, resource: { buffer: this.particleSystem.getCounterBuffer() } },
                { binding: 3, resource: { buffer: this.particleSystem.getSpawnBuffer() } },
                { binding: 4, resource: { buffer: this.emitterSettingsBuffer } },
                // Lighting bindings
                { binding: 5, resource: { buffer: this.emitterRenderSettingsBuffer } },
                { binding: 6, resource: this.shadowPass?.getShadowMapView() || this._placeholderDepthView },
                { binding: 7, resource: this.shadowPass?.getShadowSampler() || this._placeholderComparisonSampler },
                { binding: 8, resource: { buffer: this.shadowPass?.getCascadeMatricesBuffer() || this._placeholderMatricesBuffer } },
                { binding: 9, resource: { buffer: this.lightingPass?.lightBuffer || this._placeholderLightsBuffer } },
                { binding: 10, resource: this.shadowPass?.getSpotShadowAtlasView?.() || this._placeholderSpotShadowView },
                { binding: 11, resource: { buffer: this.shadowPass?.getSpotMatricesBuffer?.() || this._placeholderSpotMatricesBuffer } },
            ]
        })

        // Dispatch spawn compute (using spawnCount captured before executeSpawn cleared the queue)
        if (spawnCount > 0) {
            const spawnPass = commandEncoder.beginComputePass({ label: 'Particle Spawn' })
            spawnPass.setPipeline(this.spawnPipeline)
            spawnPass.setBindGroup(0, computeBindGroup)
            spawnPass.dispatchWorkgroups(Math.ceil(spawnCount / 64))
            spawnPass.end()
        }

        // Dispatch simulate compute
        const maxParticles = this.particleSystem.globalMaxParticles
        const simulatePass = commandEncoder.beginComputePass({ label: 'Particle Simulate' })
        simulatePass.setPipeline(this.simulatePipeline)
        simulatePass.setBindGroup(0, computeBindGroup)
        simulatePass.dispatchWorkgroups(Math.ceil(maxParticles / 64))
        simulatePass.end()
    }

    async _executeRender(commandEncoder, emitters, camera, mainLight) {
        const { device, canvas } = this.engine

        // Extract camera right/up from view matrix if not provided
        // View matrix rows are: right, up, forward (transposed world axes)
        let cameraRight = camera.right
        let cameraUp = camera.up

        if (!cameraRight && camera.view) {
            // First row of view matrix is camera right vector
            cameraRight = [camera.view[0], camera.view[4], camera.view[8]]
        }
        if (!cameraUp && camera.view) {
            // Second row of view matrix is camera up vector
            cameraUp = [camera.view[1], camera.view[5], camera.view[9]]
        }

        // Fallback defaults
        cameraRight = cameraRight || [1, 0, 0]
        cameraUp = cameraUp || [0, 1, 0]

        // Get lighting settings
        const mainLightEnabled = mainLight?.enabled !== false
        const mainLightIntensity = mainLight?.intensity ?? 1.0
        const mainLightColor = mainLight?.color || [1.0, 0.95, 0.9]
        const lightDir = mainLight?.direction || [-1, 1, -0.5]

        // Normalize light direction
        const lightDirLen = Math.sqrt(lightDir[0] * lightDir[0] + lightDir[1] * lightDir[1] + lightDir[2] * lightDir[2])
        const normalizedLightDir = [
            lightDir[0] / lightDirLen,
            lightDir[1] / lightDirLen,
            lightDir[2] / lightDirLen
        ]

        // Get ambient color from settings
        const ambientColor = this.settings?.ambient?.color || [0.1, 0.12, 0.15]
        const ambientIntensity = this.settings?.ambient?.intensity ?? 0.3

        // Get shadow settings
        const shadowConfig = this.settings?.shadow || {}
        const shadowBias = shadowConfig.bias ?? 0.0005
        const shadowStrength = shadowConfig.strength ?? 0.8
        const shadowMapSize = shadowConfig.mapSize ?? 2048
        const cascadeSizes = shadowConfig.cascadeSizes || [10, 25, 100]

        // Get IBL settings - use actual mip count from environment map
        const envSettings = this.settings?.environment || {}
        const envDiffuseLevel = envSettings.diffuseLevel ?? 0.5
        const envMipCount = this.environmentMap?.mipCount ?? envSettings.envMipCount ?? 8
        const envExposure = envSettings.exposure ?? 1.0

        // Get light count from lighting pass
        const lightCount = this.lightingPass?.lights?.length ?? 0

        // Update render uniforms (92 floats = 368 bytes)
        const uniformData = new Float32Array(92)
        // viewMatrix (mat4x4f) - floats 0-15
        uniformData.set(camera.view, 0)
        // projectionMatrix (mat4x4f) - floats 16-31
        uniformData.set(camera.proj, 16)
        // cameraPosition (vec3f) + time (f32) - floats 32-35
        uniformData.set(camera.position, 32)
        uniformData[35] = this._lastTime
        // cameraRight (vec3f) + softness (f32) - floats 36-39
        uniformData.set(cameraRight, 36)
        uniformData[39] = emitters[0]?.softness ?? 0.25
        // cameraUp (vec3f) + zOffset (f32) - floats 40-43
        uniformData.set(cameraUp, 40)
        uniformData[43] = emitters[0]?.zOffset ?? 0.01
        // screenSize (vec2f) + near + far - floats 44-47
        uniformData[44] = canvas.width
        uniformData[45] = canvas.height
        uniformData[46] = camera.near ?? 0.1
        uniformData[47] = camera.far ?? 1000
        // blendMode + lit + shadowBias + shadowStrength - floats 48-51
        // blendMode set per-pass below
        uniformData[49] = emitters[0]?.lit ? 1.0 : 0.0
        uniformData[50] = shadowBias
        uniformData[51] = shadowStrength
        // lightDir (vec3f) + shadowMapSize (f32) - floats 52-55
        uniformData.set(normalizedLightDir, 52)
        uniformData[55] = shadowMapSize
        // lightColor (vec4f) - floats 56-59
        // Multiply intensity by 12.0 to match LightingPass convention
        uniformData[56] = mainLightColor[0]
        uniformData[57] = mainLightColor[1]
        uniformData[58] = mainLightColor[2]
        uniformData[59] = mainLightEnabled ? mainLightIntensity * 12.0 : 0.0
        // ambientColor (vec4f) - floats 60-63
        uniformData[60] = ambientColor[0]
        uniformData[61] = ambientColor[1]
        uniformData[62] = ambientColor[2]
        uniformData[63] = ambientIntensity
        // cascadeSizes (vec4f) - floats 64-67
        uniformData[64] = cascadeSizes[0]
        uniformData[65] = cascadeSizes[1]
        uniformData[66] = cascadeSizes[2]
        uniformData[67] = 0.0  // padding
        // envParams (vec4f) - floats 68-71
        // x = diffuse level, y = mip count, z = encoding (0=equirect, 1=octahedral), w = exposure
        uniformData[68] = this.environmentMap ? envDiffuseLevel : 0.0
        uniformData[69] = envMipCount
        uniformData[70] = this.environmentEncoding === 'octahedral' || this.environmentEncoding === 1 ? 1.0 : 0.0
        uniformData[71] = envExposure

        // lightParams (vec4u) - floats 72-75 (stored as u32)
        // x = light count, y = unused, z = unused, w = unused
        const uniformIntView = new Uint32Array(uniformData.buffer)
        uniformIntView[72] = lightCount
        uniformIntView[73] = 0
        uniformIntView[74] = 0
        uniformIntView[75] = 0

        // Fog uniforms - floats 76-91
        const fogSettings = this.settings?.environment?.fog || {}
        const fogEnabled = fogSettings.enabled ?? false
        const fogColor = fogSettings.color ?? [0.8, 0.85, 0.9]
        const fogDistances = fogSettings.distances ?? [0, 50, 200]
        const fogAlphas = fogSettings.alpha ?? [0.0, 0.3, 0.8]
        const fogHeightFade = fogSettings.heightFade ?? [-10, 100]
        const fogBrightResist = fogSettings.brightResist ?? 0.8

        // fogColor (vec3f) + fogEnabled (f32) - floats 76-79
        uniformData[76] = fogColor[0]
        uniformData[77] = fogColor[1]
        uniformData[78] = fogColor[2]
        uniformData[79] = fogEnabled ? 1.0 : 0.0
        // fogDistances (vec3f) + fogBrightResist (f32) - floats 80-83
        uniformData[80] = fogDistances[0]
        uniformData[81] = fogDistances[1]
        uniformData[82] = fogDistances[2]
        uniformData[83] = fogBrightResist
        // fogAlphas (vec3f) + fogPad1 (f32) - floats 84-87
        uniformData[84] = fogAlphas[0]
        uniformData[85] = fogAlphas[1]
        uniformData[86] = fogAlphas[2]
        uniformData[87] = 0.0  // padding
        // fogHeightFade (vec2f) + fogDebug (f32) + fogPad2 (f32) - floats 88-91
        uniformData[88] = fogHeightFade[0]  // bottomY
        uniformData[89] = fogHeightFade[1]  // topY
        uniformData[90] = fogSettings.debug ?? 0  // fogDebug (0=off, 2=show distance)
        uniformData[91] = 0.0  // padding

        // Update emitter render settings buffer (lit, emissive, softness, zOffset per emitter)
        const emitterRenderData = new Float32Array(16 * 4)  // 16 emitters * 4 floats
        for (let i = 0; i < Math.min(emitters.length, 16); i++) {
            const e = emitters[i]
            const offset = i * 4
            emitterRenderData[offset + 0] = e.lit ? 1.0 : 0.0
            emitterRenderData[offset + 1] = e.emissive ?? 1.0
            emitterRenderData[offset + 2] = e.softness ?? 0.25
            emitterRenderData[offset + 3] = e.zOffset ?? 0.01
        }
        device.queue.writeBuffer(this.emitterRenderSettingsBuffer, 0, emitterRenderData)

        // Get or load particle texture
        const textureUrl = emitters[0]?.texture
        const particleTexture = textureUrl
            ? await this.particleSystem.loadTexture(textureUrl)
            : this.particleSystem.getDefaultTexture()

        // Create render bind group - use GBuffer depth directly (depthReadOnly allows this)
        const renderBindGroup = device.createBindGroup({
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.renderUniformBuffer } },
                { binding: 1, resource: { buffer: this.particleSystem.getParticleBuffer() } },
                { binding: 2, resource: particleTexture?.view || this._placeholderTexture.view },
                { binding: 3, resource: particleTexture?.sampler || this._placeholderSampler },
                { binding: 4, resource: this.gbuffer.depth.view },
                { binding: 5, resource: this.shadowPass?.getShadowMapView() || this._placeholderDepthView },
                { binding: 6, resource: this.shadowPass?.getShadowSampler() || this._placeholderComparisonSampler },
                { binding: 7, resource: { buffer: this.shadowPass?.getCascadeMatricesBuffer() || this._placeholderMatricesBuffer } },
                { binding: 8, resource: { buffer: this.emitterRenderSettingsBuffer } },
                { binding: 9, resource: this.environmentMap?.view || this._placeholderTexture.view },
                { binding: 10, resource: this.environmentMap?.sampler || this._placeholderSampler },
                // Point/spot lights buffer from lighting pass
                { binding: 11, resource: { buffer: this.lightingPass?.lightBuffer || this._placeholderLightsBuffer } },
                // Spot shadow atlas (uses same sampler as cascade shadows)
                { binding: 12, resource: this.shadowPass?.getSpotShadowAtlasView?.() || this._placeholderSpotShadowView },
                { binding: 13, resource: this.shadowPass?.getShadowSampler?.() || this._placeholderComparisonSampler },
                // Spot shadow matrices
                { binding: 14, resource: { buffer: this.shadowPass?.getSpotMatricesBuffer?.() || this._placeholderSpotMatricesBuffer } },
            ]
        })

        const maxParticles = this.particleSystem.globalMaxParticles

        // Check which blend modes are in use
        const hasAlpha = emitters.some(e => e.blendMode !== 'additive')
        const hasAdditive = emitters.some(e => e.blendMode === 'additive')

        // Draw alpha-blended particles first (back-to-front would be ideal, but we draw all)
        // NOTE: Must submit each pass separately because writeBuffer is immediate
        // but render passes execute when command buffer is submitted
        if (hasAlpha) {
            uniformData[48] = 0.0  // blendMode = alpha
            device.queue.writeBuffer(this.renderUniformBuffer, 0, uniformData)

            const alphaEncoder = device.createCommandEncoder({ label: 'Particle Alpha Pass' })
            const renderPass = alphaEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.outputTexture.view,
                    loadOp: 'load',
                    storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: this.gbuffer.depth.view,
                    depthReadOnly: true,
                },
                label: 'Particle Render (Alpha)'
            })

            renderPass.setPipeline(this.renderPipelineAlpha)
            renderPass.setBindGroup(0, renderBindGroup)
            renderPass.draw(6, maxParticles, 0, 0)
            renderPass.end()
            device.queue.submit([alphaEncoder.finish()])
        }

        // Draw additive particles (order doesn't matter for additive)
        if (hasAdditive) {
            uniformData[48] = 1.0  // blendMode = additive
            device.queue.writeBuffer(this.renderUniformBuffer, 0, uniformData)

            const additiveEncoder = device.createCommandEncoder({ label: 'Particle Additive Pass' })
            const renderPass = additiveEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.outputTexture.view,
                    loadOp: 'load',
                    storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: this.gbuffer.depth.view,
                    depthReadOnly: true,
                },
                label: 'Particle Render (Additive)'
            })

            renderPass.setPipeline(this.renderPipelineAdditive)
            renderPass.setBindGroup(0, renderBindGroup)
            renderPass.draw(6, maxParticles, 0, 0)
            renderPass.end()
            device.queue.submit([additiveEncoder.finish()])
        }
    }

    async _resize(width, height) {
        // Nothing to resize - uses shared resources
    }

    _destroy() {
        this._textureCache.clear()
        this.particleSystem = null
    }
}

export { ParticlePass }
