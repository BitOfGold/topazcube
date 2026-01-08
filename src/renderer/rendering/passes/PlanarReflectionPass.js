import { BasePass } from "./BasePass.js"
import { GBufferPass } from "./GBufferPass.js"
import { LightingPass } from "./LightingPass.js"
import { ParticlePass } from "./ParticlePass.js"
import { FogPass } from "./FogPass.js"
import { mat4 } from "../../math.js"

/**
 * PlanarReflectionPass - Renders the scene from a mirrored camera for water/floor reflections
 *
 * This pass creates a true planar reflection by:
 * 1. Mirroring the camera position across the ground plane (Y = groundLevel)
 * 2. Inverting the camera's Y view direction
 * 3. Rendering the full scene from this mirrored viewpoint
 * 4. Storing the result for later compositing with reflective surfaces
 *
 * Settings (from engine.settings.planarReflection):
 * - enabled: boolean - Enable/disable planar reflections
 * - groundLevel: number - Y coordinate of the reflection plane
 * - resolution: number - Resolution multiplier (0.5 = half res, 1.0 = full res)
 */
class PlanarReflectionPass extends BasePass {
    constructor(engine = null) {
        super('PlanarReflection', engine)

        // Output texture (HDR, same format as lighting output)
        this.outputTexture = null

        // Internal passes for rendering the mirrored view
        this.gbufferPass = null
        this.lightingPass = null
        this.particlePass = null
        this.fogPass = null

        // Particle system reference (set by RenderGraph)
        this.particleSystem = null

        // Mirrored camera matrices
        this.mirrorView = mat4.create()
        this.mirrorProj = mat4.create()

        // Dimensions
        this.width = 0
        this.height = 0

        // Dummy AO texture (white = no occlusion)
        this.dummyAO = null

        // Textures pending destruction (wait for GPU to finish using them)
        this._pendingDestroyRing = [[], [], []]
        this._pendingDestroyIndex = 0
    }

    async _init() {
        const { canvas } = this.engine

        // Get resolution multiplier from settings
        const resScale = this.settings?.planarReflection?.resolution ?? 0.5
        this.width = Math.floor(canvas.width * resScale)
        this.height = Math.floor(canvas.height * resScale)

        await this._createResources()
    }

    async _createResources() {
        const { device } = this.engine

        // Create internal GBuffer pass at reflection resolution
        this.gbufferPass = new GBufferPass(this.engine)
        await this.gbufferPass.initialize()
        await this.gbufferPass.resize(this.width, this.height)

        // Create internal Lighting pass
        this.lightingPass = new LightingPass(this.engine)
        await this.lightingPass.initialize()
        await this.lightingPass.resize(this.width, this.height)
        // Use exposure = 1.0 to avoid double exposure (main render applies exposure)
        this.lightingPass.exposureOverride = 1.0

        // Create dummy AO texture (white = no occlusion for reflections)
        const dummyAOTexture = device.createTexture({
            label: 'planarReflectionAO',
            size: [this.width, this.height],
            format: 'r8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
        })
        const whiteData = new Uint8Array(this.width * this.height).fill(255)
        device.queue.writeTexture(
            { texture: dummyAOTexture },
            whiteData,
            { bytesPerRow: this.width },
            { width: this.width, height: this.height }
        )
        this.dummyAO = {
            texture: dummyAOTexture,
            view: dummyAOTexture.createView()
        }

        // Wire up passes
        await this.lightingPass.setGBuffer(this.gbufferPass.getGBuffer())
        this.lightingPass.setAOTexture(this.dummyAO)

        // Create internal Particle pass for reflected particles
        this.particlePass = new ParticlePass(this.engine)
        await this.particlePass.initialize()
        this.particlePass.setGBuffer(this.gbufferPass.getGBuffer())

        // Create internal Fog pass for reflected fog
        this.fogPass = new FogPass(this.engine)
        await this.fogPass.initialize()
        await this.fogPass.resize(this.width, this.height)

        // Set output texture to lighting output (fog will copy to this if enabled)
        this.outputTexture = this.lightingPass.getOutputTexture()
    }

    /**
     * Set particle system reference for rendering particles in reflection
     * @param {ParticleSystem} particleSystem
     */
    setParticleSystem(particleSystem) {
        this.particleSystem = particleSystem
        if (this.particlePass) {
            this.particlePass.setParticleSystem(particleSystem)
        }
    }

    /**
     * Set dependencies from main render pipeline
     * @param {Object} options
     * @param {Texture} options.environmentMap - Environment map for IBL
     * @param {number} options.encoding - Environment encoding (0=equirect, 1=octahedral)
     * @param {Object} options.shadowPass - Shadow pass for shadows in reflection
     * @param {Object} options.lightingPass - Lighting pass for point/spot lights
     * @param {Texture} options.noise - Noise texture (blue noise or bayer dither)
     * @param {number} options.noiseSize - Noise texture size
     */
    setDependencies(options) {
        const { environmentMap, encoding, shadowPass, lightingPass, noise, noiseSize } = options

        if (environmentMap) {
            this.lightingPass.setEnvironmentMap(environmentMap, encoding ?? 0)
            // Also set environment map for particle IBL in reflections
            if (this.particlePass) {
                this.particlePass.setEnvironmentMap(environmentMap, encoding ?? 0)
            }
        }
        if (shadowPass) {
            this.lightingPass.setShadowPass(shadowPass)
            // Also set shadow pass for particle lighting in reflections
            if (this.particlePass) {
                this.particlePass.setShadowPass(shadowPass)
            }
        }
        if (lightingPass) {
            // Set lighting pass for particle point/spot lights in reflections
            if (this.particlePass) {
                this.particlePass.setLightingPass(lightingPass)
            }
        }
        if (noise) {
            // Planar reflection always uses static noise (no animation)
            this.gbufferPass.setNoise(noise, noiseSize, false)
            this.lightingPass.setNoise(noise, noiseSize, false)
        }
    }

    /**
     * Execute planar reflection pass
     *
     * @param {Object} context
     * @param {Camera} context.camera - Main camera
     * @param {Object} context.meshes - Meshes to render
     * @param {Array} context.lights - Processed lights array
     * @param {Object} context.mainLight - Main directional light settings
     * @param {number} context.dt - Delta time
     */
    async _execute(context) {
        const { device, canvas, stats } = this.engine
        const { camera, meshes, lights, mainLight, dt = 0 } = context

        // Process deferred texture destruction (3 frames delayed)
        this._pendingDestroyIndex = (this._pendingDestroyIndex + 1) % 3
        const toDestroy = this._pendingDestroyRing[this._pendingDestroyIndex]
        for (const tex of toDestroy) {
            tex.destroy()
        }
        this._pendingDestroyRing[this._pendingDestroyIndex] = []

        // Check if enabled (RenderGraph already skips when disabled, but double-check)
        if (!this.settings?.planarReflection?.enabled) {
            stats.planarDrawCalls = 0
            stats.planarTriangles = 0
            return
        }

        // Get ground level from settings (can be changed in real-time)
        const groundLevel = this.settings?.planarReflection?.groundLevel ?? 0

        // Create mirrored camera
        const mirrorCamera = this._createMirrorCamera(camera, groundLevel)

        // Copy lights to internal lighting pass
        this.lightingPass.lights = lights

        // Disable reflection mode for now (focus on geometry first)
        this.lightingPass.reflectionMode = false

        // Set clip plane based on camera position relative to ground level
        // - Camera above ground: show objects above ground in reflection (discard below)
        // - Camera below ground: show objects below ground in reflection (discard above)
        const cameraY = camera.position[1]
        const cameraAboveGround = cameraY >= groundLevel

        this.gbufferPass.clipPlaneEnabled = true
        if (cameraAboveGround) {
            // Camera above water: render objects above water (discard below)
            this.gbufferPass.clipPlaneY = groundLevel + 0.001
            this.gbufferPass.clipPlaneDirection = 1.0  // discard below
        } else {
            // Camera below water: render objects below water (discard above)
            this.gbufferPass.clipPlaneY = groundLevel - 0.2
            this.gbufferPass.clipPlaneDirection = -1.0  // discard above
        }

        // Execute GBuffer pass with mirrored camera
        await this.gbufferPass.execute({
            camera: mirrorCamera,
            meshes,
            dt
        })

        // Capture planar reflection stats after gbuffer (before main pass resets them)
        stats.planarDrawCalls = stats.drawCalls
        stats.planarTriangles = stats.triangles

        // Reset clip plane after reflection render
        this.gbufferPass.clipPlaneEnabled = false

        // Execute Lighting pass
        await this.lightingPass.execute({
            camera: mirrorCamera,
            meshes,
            dt,
            lights,
            mainLight
        })

        // Reset reflection mode
        this.lightingPass.reflectionMode = false

        // Execute Particle pass with mirrored camera (renders onto lighting output)
        let hdrOutput = this.lightingPass.getOutputTexture()
        if (this.particlePass && this.particleSystem?.getActiveEmitters().length > 0) {
            this.particlePass.setOutputTexture(hdrOutput)
            await this.particlePass.execute({
                camera: mirrorCamera,
                dt,
                mainLight
            })
        }

        // Execute Fog pass for reflected fog
        const fogEnabled = this.engine?.settings?.environment?.fog?.enabled
        if (this.fogPass && fogEnabled) {
            this.fogPass.setInputTexture(hdrOutput)
            this.fogPass.setGBuffer(this.gbufferPass.getGBuffer())
            await this.fogPass.execute({
                camera: mirrorCamera,
                dt
            })
            const fogOutput = this.fogPass.getOutputTexture()
            if (fogOutput && fogOutput !== hdrOutput) {
                hdrOutput = fogOutput
            }
        }

        // Copy final output to our output texture (if they're different textures)
        if (hdrOutput?.texture && this.outputTexture?.texture &&
            hdrOutput.texture !== this.outputTexture.texture) {
            const commandEncoder = device.createCommandEncoder({ label: 'planarReflectionCopy' })
            commandEncoder.copyTextureToTexture(
                { texture: hdrOutput.texture },
                { texture: this.outputTexture.texture },
                [this.width, this.height, 1]
            )
            device.queue.submit([commandEncoder.finish()])
        }
    }

    /**
     * Create a mirrored camera for reflection rendering
     *
     * Classic planar reflection: mirror the camera below the ground plane
     * and render the scene from that mirrored viewpoint.
     *
     * @param {Camera} camera - Original camera
     * @param {number} groundLevel - Y coordinate of reflection plane
     * @returns {Object} Camera-like object with mirrored matrices
     */
    _createMirrorCamera(camera, groundLevel) {
        // We want: same camera position, but world geometry mirrored at groundLevel
        //
        // To achieve this:
        // - Camera stays at original position
        // - Apply reflection matrix to mirror world geometry
        // - mirrorView = camera.view * reflectionMatrix
        //   This transforms: world point P -> reflected P' -> view space

        // User wants: same camera position/angle, but world geometry mirrored at groundLevel
        //
        // To mirror world geometry without moving camera:
        // Apply reflection matrix to world coordinates BEFORE view transform
        //
        // Normal: clipPos = proj * view * world
        // Mirrored: clipPos = proj * view * reflection * world
        //
        // We combine: mirrorView = view * reflection
        // Result: geometry is reflected, camera stays the same

        // Reflection matrix for Y = groundLevel
        const reflectionMatrix = mat4.create()
        reflectionMatrix[5] = -1  // Flip Y
        reflectionMatrix[13] = 2 * groundLevel  // Offset

        // Combine view with reflection: view * reflection
        mat4.multiply(this.mirrorView, camera.view, reflectionMatrix)

        // Copy projection and flip Y to fix winding order
        // The reflection matrix changes coordinate system handedness, which reverses
        // triangle winding. Flipping projection Y reverses it back, but also flips
        // the rendered image vertically - we compensate by flipping UV.y when sampling.
        mat4.copy(this.mirrorProj, camera.proj)
        this.mirrorProj[5] *= -1  // Flip projection Y

        // Create inverse matrices
        const newIView = mat4.create()
        const newIProj = mat4.create()
        const newIViewProj = mat4.create()
        const viewProj = mat4.create()
        mat4.invert(newIView, this.mirrorView)
        mat4.invert(newIProj, this.mirrorProj)
        mat4.multiply(viewProj, this.mirrorProj, this.mirrorView)
        mat4.invert(newIViewProj, viewProj)

        // Return camera with mirrored view matrix
        return {
            view: this.mirrorView,
            proj: this.mirrorProj,
            iView: newIView,
            iProj: newIProj,
            iViewProj: newIViewProj,
            viewProj,
            position: camera.position,  // Same camera position
            near: camera.near,
            far: camera.far,
            aspect: camera.aspect,
            jitterEnabled: false,
            jitterOffset: [0, 0],
            screenSize: [this.width, this.height],
            updateMatrix: () => {},
            updateView: () => {}
        }
    }

    /**
     * Get the reflection texture for compositing
     * @returns {Texture} HDR reflection texture
     */
    getReflectionTexture() {
        return this.outputTexture
    }

    async _resize(width, height) {
        // Get resolution multiplier from settings
        const resScale = this.settings?.planarReflection?.resolution ?? 0.5
        this.width = Math.floor(width * resScale)
        this.height = Math.floor(height * resScale)

        // Resize internal passes
        if (this.gbufferPass) {
            await this.gbufferPass.resize(this.width, this.height)
        }
        if (this.lightingPass) {
            await this.lightingPass.resize(this.width, this.height)
            await this.lightingPass.setGBuffer(this.gbufferPass.getGBuffer())
        }
        if (this.particlePass) {
            this.particlePass.setGBuffer(this.gbufferPass.getGBuffer())
        }
        if (this.fogPass) {
            await this.fogPass.resize(this.width, this.height)
        }

        // Queue old dummy AO for deferred destruction
        if (this.dummyAO?.texture) {
            this._pendingDestroyRing[this._pendingDestroyIndex].push(this.dummyAO.texture)
        }

        const { device } = this.engine
        const dummyAOTexture = device.createTexture({
            label: 'planarReflectionAO',
            size: [this.width, this.height],
            format: 'r8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
        })
        const whiteData = new Uint8Array(this.width * this.height).fill(255)
        device.queue.writeTexture(
            { texture: dummyAOTexture },
            whiteData,
            { bytesPerRow: this.width },
            { width: this.width, height: this.height }
        )
        this.dummyAO = {
            texture: dummyAOTexture,
            view: dummyAOTexture.createView()
        }
        this.lightingPass.setAOTexture(this.dummyAO)

        // Update output reference
        this.outputTexture = this.lightingPass.getOutputTexture()
    }

    _destroy() {
        if (this.gbufferPass) {
            this.gbufferPass.destroy()
            this.gbufferPass = null
        }
        if (this.lightingPass) {
            this.lightingPass.destroy()
            this.lightingPass = null
        }
        if (this.particlePass) {
            this.particlePass.destroy()
            this.particlePass = null
        }
        if (this.fogPass) {
            this.fogPass.destroy()
            this.fogPass = null
        }
        if (this.dummyAO?.texture) {
            this.dummyAO.texture.destroy()
            this.dummyAO = null
        }
        // Clean up any pending textures in ring buffer
        for (const slot of this._pendingDestroyRing) {
            for (const tex of slot) {
                tex.destroy()
            }
        }
        this._pendingDestroyRing = [[], [], []]
        this.outputTexture = null
    }
}

export { PlanarReflectionPass }
