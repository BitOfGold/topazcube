import { BasePass } from "./BasePass.js"
import { Texture } from "../../Texture.js"

import volumetricRaymarchWGSL from "../shaders/volumetric_raymarch.wgsl"
import volumetricBlurWGSL from "../shaders/volumetric_blur.wgsl"
import volumetricCompositeWGSL from "../shaders/volumetric_composite.wgsl"

/**
 * VolumetricFogPass - Simple ray marching volumetric fog
 *
 * Pipeline:
 * 1. Ray Marching (Fragment) - March rays, sample shadows directly at each step
 * 2. Blur (Fragment) - Gaussian blur to soften edges
 * 3. Composite (Fragment) - Blend fog into scene
 */
class VolumetricFogPass extends BasePass {
    constructor(engine = null) {
        super('VolumetricFog', engine)

        // Pipelines
        this.raymarchPipeline = null
        this.blurHPipeline = null
        this.blurVPipeline = null
        this.compositePipeline = null

        // Bind group layouts
        this.raymarchBGL = null
        this.blurBGL = null
        this.compositeBGL = null

        // 2D textures
        this.raymarchTexture = null
        this.blurTempTexture = null
        this.blurredTexture = null
        this.outputTexture = null

        // Uniform buffers
        this.raymarchUniformBuffer = null
        this.blurHUniformBuffer = null
        this.blurVUniformBuffer = null
        this.compositeUniformBuffer = null

        // Samplers
        this.linearSampler = null

        // External dependencies
        this.inputTexture = null
        this.gbuffer = null
        this.shadowPass = null
        this.lightingPass = null

        // Dimensions
        this.canvasWidth = 0
        this.canvasHeight = 0
        this.renderWidth = 0
        this.renderHeight = 0

        // Adaptive scatter state
        this._currentMainLightScatter = null  // Will be initialized on first execute
        this._lastUpdateTime = 0
        this._cameraInShadowSmooth = 0.0  // 0 = in light, 1 = in shadow

        // Sky detection state (for adaptive scatter)
        this._skyVisible = true  // Assume sky visible until proven otherwise
        this._skyCheckPending = false
        this._lastSkyCheckTime = 0
    }

    // Settings getters
    get volumetricSettings() { return this.settings?.volumetricFog ?? {} }
    get fogSettings() { return this.settings?.fog ?? {} }
    get isVolumetricEnabled() { return this.volumetricSettings.enabled ?? false }
    get resolution() { return this.volumetricSettings.resolution ?? 0.25 }
    get maxSamples() { return this.volumetricSettings.maxSamples ?? 32 }
    get blurRadius() { return this.volumetricSettings.blurRadius ?? 4.0 }
    get fogDensity() { return this.volumetricSettings.density ?? this.volumetricSettings.densityMultiplier ?? 0.5 }
    get scatterStrength() { return this.volumetricSettings.scatterStrength ?? 1.0 }
    get maxDistance() { return this.volumetricSettings.maxDistance ?? 20.0 }
    get heightRange() { return this.volumetricSettings.heightRange ?? [-5, 20] }
    get shadowsEnabled() { return this.volumetricSettings.shadowsEnabled ?? true }
    get noiseStrength() { return this.volumetricSettings.noiseStrength ?? 1.0 }  // 0 = uniform fog, 1 = full noise
    get noiseAnimated() { return this.volumetricSettings.noiseAnimated ?? true }
    get noiseScale() { return this.volumetricSettings.noiseScale ?? 0.25 }  // Noise frequency (higher = finer detail)
    get mainLightScatter() { return this.volumetricSettings.mainLightScatter ?? 1.0 }  // Scatter when camera in light
    get mainLightScatterDark() { return this.volumetricSettings.mainLightScatterDark ?? 3.0 }  // Scatter when camera in shadow
    get mainLightSaturation() { return this.volumetricSettings.mainLightSaturation ?? 1.0 }  // Max brightness cap
    // Brightness-based attenuation (fog less visible over bright surfaces)
    get brightnessThreshold() { return this.volumetricSettings.brightnessThreshold ?? 1.0 }  // Scene luminance where fog starts fading
    get minVisibility() { return this.volumetricSettings.minVisibility ?? 0.15 }  // Minimum fog visibility over very bright surfaces
    get skyBrightness() { return this.volumetricSettings.skyBrightness ?? 5.0 }  // Virtual brightness for sky (far depth)
    // Debug mode: 0=normal, 1=depth, 2=ray dir, 3=noise, 4=viewDir.z, 5=worldPos, 6=accum, 7=light dist, 8=light pos
    get debugMode() { return this.volumetricSettings.debug ?? 0 }

    setInputTexture(texture) { this.inputTexture = texture }
    setGBuffer(gbuffer) { this.gbuffer = gbuffer }
    setShadowPass(shadowPass) { this.shadowPass = shadowPass }
    setLightingPass(lightingPass) { this.lightingPass = lightingPass }
    getOutputTexture() { return this.outputTexture }

    // Unused setters (kept for API compatibility)
    setHiZPass() {}

    async _init() {
        const { device } = this.engine

        this.linearSampler = device.createSampler({
            label: 'Volumetric Linear Sampler',
            minFilter: 'linear',
            magFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        })

        // Fallback shadow resources
        this.fallbackShadowSampler = device.createSampler({
            label: 'Volumetric Fallback Shadow Sampler',
            compare: 'less',
        })

        this.fallbackCascadeShadowMap = device.createTexture({
            label: 'Volumetric Fallback Cascade Shadow',
            size: [1, 1, 3],
            format: 'depth32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        })

        const identityMatrix = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
        const matrixData = new Float32Array(16 * 3)
        for (let i = 0; i < 3; i++) matrixData.set(identityMatrix, i * 16)

        this.fallbackCascadeMatrices = device.createBuffer({
            label: 'Volumetric Fallback Cascade Matrices',
            size: 16 * 4 * 3,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        device.queue.writeBuffer(this.fallbackCascadeMatrices, 0, matrixData)

        // Fallback lights buffer (empty) - 96 bytes per light to match WGSL alignment
        this.fallbackLightsBuffer = device.createBuffer({
            label: 'Volumetric Fallback Lights',
            size: 768 * 96, // MAX_LIGHTS * 96 bytes per light
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })

        // Fallback spot shadow atlas (1x1 depth texture)
        this.fallbackSpotShadowAtlas = device.createTexture({
            label: 'Volumetric Fallback Spot Shadow Atlas',
            size: [1, 1, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        })

        // Fallback spot matrices buffer
        const spotMatrixData = new Float32Array(16 * 16) // 16 spot shadows max
        for (let i = 0; i < 16; i++) spotMatrixData.set(identityMatrix, i * 16)
        this.fallbackSpotMatrices = device.createBuffer({
            label: 'Volumetric Fallback Spot Matrices',
            size: 16 * 4 * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        device.queue.writeBuffer(this.fallbackSpotMatrices, 0, spotMatrixData)

        await this._createResources(this.engine.canvas.width, this.engine.canvas.height)
    }

    async _createResources(width, height) {
        const { device } = this.engine

        this.canvasWidth = width
        this.canvasHeight = height
        this.renderWidth = Math.max(1, Math.floor(width * this.resolution))
        this.renderHeight = Math.max(1, Math.floor(height * this.resolution))

        this._destroyTextures()

        // Create 2D textures at reduced resolution
        this.raymarchTexture = this._create2DTexture('Raymarch Output', this.renderWidth, this.renderHeight)
        this.blurTempTexture = this._create2DTexture('Blur Temp', this.renderWidth, this.renderHeight)
        this.blurredTexture = this._create2DTexture('Blurred Fog', this.renderWidth, this.renderHeight)
        this.outputTexture = await Texture.renderTarget(this.engine, 'rgba16float', width, height)

        // Uniform buffers
        this.raymarchUniformBuffer = this._createUniformBuffer('Raymarch Uniforms', 256)
        this.blurHUniformBuffer = this._createUniformBuffer('Blur H Uniforms', 32)
        this.blurVUniformBuffer = this._createUniformBuffer('Blur V Uniforms', 32)
        this.compositeUniformBuffer = this._createUniformBuffer('Composite Uniforms', 48)

        await this._createPipelines()
    }

    _create2DTexture(label, width, height) {
        const texture = this.engine.device.createTexture({
            label,
            size: [width, height, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        })
        return { texture, view: texture.createView(), width, height }
    }

    _createUniformBuffer(label, size) {
        return this.engine.device.createBuffer({
            label,
            size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
    }

    async _createPipelines() {
        const { device } = this.engine

        // === Raymarch Pipeline ===
        const raymarchModule = device.createShaderModule({
            label: 'Volumetric Raymarch Shader',
            code: volumetricRaymarchWGSL,
        })

        this.raymarchBGL = device.createBindGroupLayout({
            label: 'Volumetric Raymarch BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // cascade matrices
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // lights
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } }, // spot shadow atlas
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // spot matrices
            ],
        })

        this.raymarchPipeline = await device.createRenderPipelineAsync({
            label: 'Volumetric Raymarch Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.raymarchBGL] }),
            vertex: { module: raymarchModule, entryPoint: 'vertexMain' },
            fragment: {
                module: raymarchModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'rgba16float' }],
            },
            primitive: { topology: 'triangle-list' },
        })

        // === Blur Pipeline ===
        const blurModule = device.createShaderModule({
            label: 'Volumetric Blur Shader',
            code: volumetricBlurWGSL,
        })

        this.blurBGL = device.createBindGroupLayout({
            label: 'Volumetric Blur BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        })

        const blurPipelineDesc = {
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.blurBGL] }),
            vertex: { module: blurModule, entryPoint: 'vertexMain' },
            fragment: {
                module: blurModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'rgba16float' }],
            },
            primitive: { topology: 'triangle-list' },
        }

        this.blurHPipeline = await device.createRenderPipelineAsync({ ...blurPipelineDesc, label: 'Volumetric Blur H' })
        this.blurVPipeline = await device.createRenderPipelineAsync({ ...blurPipelineDesc, label: 'Volumetric Blur V' })

        // === Composite Pipeline ===
        const compositeModule = device.createShaderModule({
            label: 'Volumetric Composite Shader',
            code: volumetricCompositeWGSL,
        })

        this.compositeBGL = device.createBindGroupLayout({
            label: 'Volumetric Composite BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
            ],
        })

        this.compositePipeline = await device.createRenderPipelineAsync({
            label: 'Volumetric Composite Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.compositeBGL] }),
            vertex: { module: compositeModule, entryPoint: 'vertexMain' },
            fragment: {
                module: compositeModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'rgba16float' }],
            },
            primitive: { topology: 'triangle-list' },
        })
    }

    async _execute(context) {
        if (!this.isVolumetricEnabled) return
        if (!this.inputTexture || !this.gbuffer) return

        const { device } = this.engine
        const { camera, mainLight, lights } = context
        const time = performance.now() / 1000

        // Update adaptive main light scatter based on camera shadow state
        this._updateAdaptiveScatter(camera, mainLight, time)

        // Get light count from context or lighting pass
        const lightCount = lights?.length ?? this.lightingPass?.lightCount ?? 0

        const commandEncoder = device.createCommandEncoder({ label: 'Volumetric Fog Pass' })

        // === Stage 1: Ray Marching ===
        this._updateRaymarchUniforms(camera, mainLight, time, lightCount)
        const raymarchBindGroup = this._createRaymarchBindGroup()
        if (raymarchBindGroup) {
            const raymarchPass = commandEncoder.beginRenderPass({
                label: 'Volumetric Raymarch',
                colorAttachments: [{
                    view: this.raymarchTexture.view,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                }],
            })
            raymarchPass.setPipeline(this.raymarchPipeline)
            raymarchPass.setBindGroup(0, raymarchBindGroup)
            raymarchPass.draw(3)
            raymarchPass.end()
        }

        // === Stage 2: Blur ===
        this._updateBlurUniforms(this.blurHUniformBuffer, 1, 0)
        this._updateBlurUniforms(this.blurVUniformBuffer, 0, 1)

        const blurHBindGroup = this._createBlurBindGroup(this.raymarchTexture, this.blurHUniformBuffer)
        const blurHPass = commandEncoder.beginRenderPass({
            label: 'Volumetric Blur H',
            colorAttachments: [{
                view: this.blurTempTexture.view,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
            }],
        })
        blurHPass.setPipeline(this.blurHPipeline)
        blurHPass.setBindGroup(0, blurHBindGroup)
        blurHPass.draw(3)
        blurHPass.end()

        const blurVBindGroup = this._createBlurBindGroup(this.blurTempTexture, this.blurVUniformBuffer)
        const blurVPass = commandEncoder.beginRenderPass({
            label: 'Volumetric Blur V',
            colorAttachments: [{
                view: this.blurredTexture.view,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
            }],
        })
        blurVPass.setPipeline(this.blurVPipeline)
        blurVPass.setBindGroup(0, blurVBindGroup)
        blurVPass.draw(3)
        blurVPass.end()

        // === Stage 3: Composite ===
        this._updateCompositeUniforms()
        const compositeBindGroup = this._createCompositeBindGroup()
        if (compositeBindGroup) {
            const compositePass = commandEncoder.beginRenderPass({
                label: 'Volumetric Composite',
                colorAttachments: [{
                    view: this.outputTexture.view,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                }],
            })
            compositePass.setPipeline(this.compositePipeline)
            compositePass.setBindGroup(0, compositeBindGroup)
            compositePass.draw(3)
            compositePass.end()
        }

        device.queue.submit([commandEncoder.finish()])
    }

    /**
     * Update adaptive main light scatter based on camera shadow state and sky visibility
     * Smoothly transitions between light/dark scatter values
     * Only uses dark scatter if camera is in shadow AND there's something overhead (no sky)
     */
    _updateAdaptiveScatter(camera, mainLight, currentTime) {
        // Initialize on first call
        if (this._currentMainLightScatter === null) {
            this._currentMainLightScatter = this.mainLightScatter
        }

        const deltaTime = this._lastUpdateTime > 0 ? (currentTime - this._lastUpdateTime) : 0.016
        this._lastUpdateTime = currentTime

        // Check if camera is in shadow using cascade shadow matrices
        let cameraInShadow = false

        if (this.shadowPass && this.shadowsEnabled && mainLight?.enabled !== false) {
            const cameraPos = camera.position || [0, 0, 0]
            cameraInShadow = this._isCameraInShadow(cameraPos)

            // Periodically check sky visibility using raycaster (every 0.5s)
            if (!this._skyCheckPending && currentTime - this._lastSkyCheckTime > 0.5) {
                this._checkSkyVisibility(cameraPos)
            }
        }

        // Only consider "in dark area" if in shadow AND sky is NOT visible
        // If sky is visible (outdoors), don't boost scatter even in shadow
        const inDarkArea = cameraInShadow && !this._skyVisible

        // Smooth transition toward target state
        // Going into dark area: 5 seconds (slow), exiting: 1 second (fast)
        const targetShadowState = inDarkArea ? 1.0 : 0.0
        const transitionSpeed = inDarkArea ? (1.0 / 5.0) : (1.0 / 1.0)  // per second

        // Exponential smoothing toward target
        const t = 1.0 - Math.exp(-transitionSpeed * deltaTime * 3.0)
        this._cameraInShadowSmooth += (targetShadowState - this._cameraInShadowSmooth) * t

        // Clamp to valid range
        this._cameraInShadowSmooth = Math.max(0, Math.min(1, this._cameraInShadowSmooth))

        // Interpolate scatter value
        const lightScatter = this.mainLightScatter
        const darkScatter = this.mainLightScatterDark
        this._currentMainLightScatter = lightScatter + (darkScatter - lightScatter) * this._cameraInShadowSmooth
    }

    /**
     * Check if sky is visible above the camera using raycaster
     * This is async and updates _skyVisible when complete
     */
    _checkSkyVisibility(cameraPos) {
        const raycaster = this.engine?.raycaster
        if (!raycaster) return

        this._skyCheckPending = true
        this._lastSkyCheckTime = this._lastUpdateTime

        // Cast ray upward from camera position
        const skyCheckDistance = this.volumetricSettings.skyCheckDistance ?? 100
        const debugSkyCheck = this.volumetricSettings.debugSkyCheck ?? false

        raycaster.cast(
            cameraPos,
            [0, 1, 0], // Straight up
            skyCheckDistance,
            (result) => {
                const wasVisible = this._skyVisible
                this._skyVisible = !result.hit
                this._skyCheckPending = false

                if (debugSkyCheck) {
                    const pos = cameraPos.map(v => v.toFixed(1)).join(', ')
                    if (result.hit) {
                        console.log(`Sky check from [${pos}]: HIT ${result.meshName || result.candidateId} at dist=${result.distance?.toFixed(1)}`)
                    } else {
                        console.log(`Sky check from [${pos}]: NO HIT (sky visible)`)
                    }
                }
            },
            { backfaces: true, debug: debugSkyCheck }  // Need backfaces to hit ceilings from below
        )
    }

    /**
     * Check if camera position is in shadow
     * Uses the shadow pass's isCameraInShadow method if available,
     * otherwise falls back to checking fog height bounds
     */
    _isCameraInShadow(cameraPos) {
        // Try shadow pass method first (if it has GPU shadow readback)
        if (typeof this.shadowPass?.isCameraInShadow === 'function') {
            return this.shadowPass.isCameraInShadow(cameraPos)
        }

        // Fallback: check if camera is below the fog height range
        // This is a simple heuristic - if camera is at the bottom of fog volume,
        // it's more likely to be in a shadowed area (cave, corridor, etc.)
        const heightRange = this.heightRange
        const fogBottom = heightRange[0]
        const fogTop = heightRange[1]
        const fogMiddle = (fogBottom + fogTop) / 2

        // Consider "in shadow" if camera is in the lower third of fog volume
        // This is a rough approximation - works for indoor/cave scenarios
        const lowerThird = fogBottom + (fogTop - fogBottom) * 0.33
        if (cameraPos[1] < lowerThird) {
            return true
        }

        // Also check if main light direction is mostly blocked (sun very low)
        // This helps with sunset/sunrise scenarios
        const mainLightDir = this.engine?.settings?.mainLight?.direction
        if (mainLightDir) {
            // If sun direction Y component is positive (sun below horizon in our convention)
            // or very low angle, consider shadowed
            const sunAngle = mainLightDir[1]  // Y component of direction
            if (sunAngle > 0.7) {  // Sun mostly pointing up = below horizon
                return true
            }
        }

        return false
    }

    _updateRaymarchUniforms(camera, mainLight, time, lightCount) {
        const { device } = this.engine

        // Verify camera matrices exist
        if (!camera.iProj || !camera.iView) {
            console.warn('VolumetricFogPass: Camera missing iProj or iView matrices')
            return
        }

        const invProj = camera.iProj
        const invView = camera.iView
        const cameraPos = camera.position || [0, 0, 0]

        const mainLightEnabled = mainLight?.enabled !== false
        const mainLightDir = mainLight?.direction ?? [-1.0, 1.0, -0.5]
        const mainLightColor = mainLight?.color ?? [1, 0.95, 0.9]
        const mainLightIntensity = mainLightEnabled ? (mainLight?.intensity ?? 1.0) : 0.0

        const fogColor = this.fogSettings.color ?? [0.8, 0.85, 0.9]
        const heightFade = this.heightRange

        const shadowsEnabled = this.shadowsEnabled && this.shadowPass != null

        const data = new Float32Array(64)
        let offset = 0

        // mat4 inverseProjection
        data.set(invProj, offset); offset += 16

        // mat4 inverseView
        data.set(invView, offset); offset += 16

        // vec3 cameraPosition + nearPlane
        data[offset++] = cameraPos[0]
        data[offset++] = cameraPos[1]
        data[offset++] = cameraPos[2]
        data[offset++] = camera.near ?? 0.1

        // farPlane + maxSamples + time + fogDensity
        data[offset++] = camera.far ?? 1000
        data[offset++] = this.maxSamples
        data[offset++] = time
        data[offset++] = this.fogDensity

        // vec3 fogColor + shadowsEnabled
        data[offset++] = fogColor[0]
        data[offset++] = fogColor[1]
        data[offset++] = fogColor[2]
        data[offset++] = shadowsEnabled ? 1.0 : 0.0

        // vec3 mainLightDir + mainLightIntensity
        data[offset++] = mainLightDir[0]
        data[offset++] = mainLightDir[1]
        data[offset++] = mainLightDir[2]
        data[offset++] = mainLightIntensity

        // vec3 mainLightColor + scatterStrength
        data[offset++] = mainLightColor[0]
        data[offset++] = mainLightColor[1]
        data[offset++] = mainLightColor[2]
        data[offset++] = this.scatterStrength

        // vec2 fogHeightFade + maxDistance + lightCount
        data[offset++] = heightFade[0]
        data[offset++] = heightFade[1]
        data[offset++] = this.maxDistance
        data[offset++] = lightCount

        // debugMode + noiseStrength + noiseAnimated + mainLightScatter (adaptive)
        data[offset++] = this.debugMode
        data[offset++] = this.noiseStrength
        data[offset++] = this.noiseAnimated ? 1.0 : 0.0
        data[offset++] = this._currentMainLightScatter  // Uses adaptive value

        // noiseScale + mainLightSaturation (ends the struct)
        data[offset++] = this.noiseScale
        data[offset++] = this.mainLightSaturation

        device.queue.writeBuffer(this.raymarchUniformBuffer, 0, data)
    }

    _updateBlurUniforms(buffer, dirX, dirY) {
        const data = new Float32Array([
            dirX, dirY,
            1.0 / this.renderWidth, 1.0 / this.renderHeight,
            this.blurRadius, 0, 0, 0,
        ])
        this.engine.device.queue.writeBuffer(buffer, 0, data)
    }

    _updateCompositeUniforms() {
        const data = new Float32Array([
            this.canvasWidth, this.canvasHeight,
            this.renderWidth, this.renderHeight,
            1.0 / this.canvasWidth, 1.0 / this.canvasHeight,
            this.brightnessThreshold, this.minVisibility,
            this.skyBrightness, 0, // skyBrightness + padding
            0, 0, // extra padding to 48 bytes
        ])
        this.engine.device.queue.writeBuffer(this.compositeUniformBuffer, 0, data)
    }

    _createRaymarchBindGroup() {
        const { device } = this.engine

        const depthTexture = this.gbuffer?.depth
        if (!depthTexture) return null

        const cascadeShadows = this.shadowPass?.getShadowMap?.() ?? this.fallbackCascadeShadowMap
        const cascadeMatrices = this.shadowPass?.getCascadeMatricesBuffer?.() ?? this.fallbackCascadeMatrices
        const shadowSampler = this.shadowPass?.getShadowSampler?.() ?? this.fallbackShadowSampler

        // Light resources
        const lightsBuffer = this.lightingPass?.getLightBuffer?.() ?? this.fallbackLightsBuffer
        const spotShadowAtlas = this.shadowPass?.getSpotShadowAtlasView?.() ?? this.fallbackSpotShadowAtlas.createView()
        const spotMatrices = this.shadowPass?.getSpotMatricesBuffer?.() ?? this.fallbackSpotMatrices

        return device.createBindGroup({
            label: 'Volumetric Raymarch Bind Group',
            layout: this.raymarchBGL,
            entries: [
                { binding: 0, resource: { buffer: this.raymarchUniformBuffer } },
                { binding: 1, resource: depthTexture.texture.createView({ aspect: 'depth-only' }) },
                { binding: 2, resource: cascadeShadows.createView({ dimension: '2d-array', aspect: 'depth-only' }) },
                { binding: 3, resource: shadowSampler },
                { binding: 4, resource: { buffer: cascadeMatrices } },
                { binding: 5, resource: { buffer: lightsBuffer } },
                { binding: 6, resource: spotShadowAtlas },
                { binding: 7, resource: { buffer: spotMatrices } },
            ],
        })
    }

    _createBlurBindGroup(inputTexture, uniformBuffer) {
        return this.engine.device.createBindGroup({
            label: 'Volumetric Blur Bind Group',
            layout: this.blurBGL,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: inputTexture.view },
                { binding: 2, resource: this.linearSampler },
            ],
        })
    }

    _createCompositeBindGroup() {
        if (!this.inputTexture) return null

        const depthTexture = this.gbuffer?.depth
        if (!depthTexture) return null

        return this.engine.device.createBindGroup({
            label: 'Volumetric Composite Bind Group',
            layout: this.compositeBGL,
            entries: [
                { binding: 0, resource: { buffer: this.compositeUniformBuffer } },
                { binding: 1, resource: this.inputTexture.view },
                { binding: 2, resource: this.blurredTexture.view },
                { binding: 3, resource: this.linearSampler },
                { binding: 4, resource: depthTexture.texture.createView({ aspect: 'depth-only' }) },
            ],
        })
    }

    _destroyTextures() {
        const textures = [this.raymarchTexture, this.blurTempTexture, this.blurredTexture, this.outputTexture]
        for (const tex of textures) {
            if (tex?.texture) tex.texture.destroy()
        }
        this.raymarchTexture = null
        this.blurTempTexture = null
        this.blurredTexture = null
        this.outputTexture = null
    }

    async _resize(width, height) {
        await this._createResources(width, height)
    }

    _destroy() {
        this._destroyTextures()

        const buffers = [this.raymarchUniformBuffer, this.blurHUniformBuffer, this.blurVUniformBuffer, this.compositeUniformBuffer]
        for (const buf of buffers) {
            if (buf) buf.destroy()
        }

        this.raymarchPipeline = null
        this.blurHPipeline = null
        this.blurVPipeline = null
        this.compositePipeline = null
    }
}

export { VolumetricFogPass }
