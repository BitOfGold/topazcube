import { BasePass } from "./BasePass.js"
import { Texture } from "../../Texture.js"
import { GBufferPass } from "./GBufferPass.js"
import { LightingPass } from "./LightingPass.js"
import { mat4, vec3 } from "../../math.js"

/**
 * AmbientCapturePass - Captures 6 directional ambient light samples for sky-aware GI
 *
 * Renders simplified views in 6 directions (up, down, left, right, front, back)
 * to capture sky visibility and distant lighting. This provides ambient lighting
 * that responds to sky visibility (blue tint from sky when outdoors, darker when
 * under a roof).
 *
 * Features:
 * - 64x64 resolution per direction (configurable)
 * - Staggered updates: 2 faces per frame (full cycle in 3 frames)
 * - Direct lights + emissive + skybox only (no IBL on geometry)
 * - Aggressive distance culling (25m default)
 * - Output: 6 average colors applied in RenderPost based on surface normal
 *
 * Settings (from engine.settings.ambientCapture):
 * - enabled: boolean - Enable/disable ambient capture
 * - maxDistance: number - Distance culling (default 25m)
 * - intensity: number - Output intensity multiplier
 * - resolution: number - Capture resolution (default 64)
 */
class AmbientCapturePass extends BasePass {
    constructor(engine = null) {
        super('AmbientCapture', engine)

        // Capture settings
        this.captureSize = 64
        this.maxDistance = 25
        this.currentFaceIndex = 0  // cycles 0-5, update 2 per frame

        // 6 face directions (calculated each frame from camera)
        // Order: up, down, left, right, front, back
        this.directions = [
            [0, 1, 0],
            [0, -1, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, -1]
        ]

        // Output: 6 average colors (vec4 each)
        this.faceColors = new Float32Array(6 * 4)
        this.faceColorsBuffer = null  // GPU storage buffer

        // Internal passes
        this.gbufferPass = null
        this.lightingPass = null

        // Render targets (reused for each face)
        this.captureTexture = null
        this.dummyAO = null

        // Compute pipeline for reduction
        this.reducePipeline = null
        this.reduceBindGroupLayout = null
        this.reduceBindGroups = []  // One per face

        // Camera matrices (reused)
        this.faceView = mat4.create()
        this.faceProj = mat4.create()

        // Textures pending destruction
        this._pendingDestroyRing = [[], [], []]
        this._pendingDestroyIndex = 0
    }

    async _init() {
        const { device } = this.engine

        // Get settings
        this.captureSize = this.settings?.ambientCapture?.resolution ?? 64
        this.maxDistance = this.settings?.ambientCapture?.maxDistance ?? 25

        await this._createResources()
        await this._createComputePipeline()
    }

    async _createResources() {
        const { device } = this.engine

        // Create capture render target
        this.captureTexture = await Texture.renderTarget(this.engine, 'rgba16float', this.captureSize, this.captureSize)
        this.captureTexture.label = 'ambientCaptureRT'

        // Create internal GBuffer pass
        this.gbufferPass = new GBufferPass(this.engine)
        await this.gbufferPass.initialize()
        await this.gbufferPass.resize(this.captureSize, this.captureSize)

        // Create internal Lighting pass
        this.lightingPass = new LightingPass(this.engine)
        await this.lightingPass.initialize()
        await this.lightingPass.resize(this.captureSize, this.captureSize)
        this.lightingPass.exposureOverride = 1.0

        // Disable IBL on geometry for ambient capture (skybox still renders as background)
        this.lightingPass.ambientCaptureMode = true

        // Create dummy AO texture (white = no occlusion)
        const dummyAOTexture = device.createTexture({
            label: 'ambientCaptureAO',
            size: [this.captureSize, this.captureSize],
            format: 'r8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
        })
        const whiteData = new Uint8Array(this.captureSize * this.captureSize).fill(255)
        device.queue.writeTexture(
            { texture: dummyAOTexture },
            whiteData,
            { bytesPerRow: this.captureSize },
            { width: this.captureSize, height: this.captureSize }
        )
        this.dummyAO = {
            texture: dummyAOTexture,
            view: dummyAOTexture.createView()
        }

        // Wire up passes
        await this.lightingPass.setGBuffer(this.gbufferPass.getGBuffer())
        this.lightingPass.setAOTexture(this.dummyAO)

        // Create output buffer for 6 face colors
        this.faceColorsBuffer = device.createBuffer({
            label: 'ambientCaptureFaceColors',
            size: 6 * 4 * 4,  // 6 vec4f
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        // Initialize with distinct colors per direction for debugging
        // up=cyan, down=brown, left=red, right=green, front=blue, back=yellow
        const initColors = new Float32Array(6 * 4)
        // Up: sky blue
        initColors[0] = 0.4; initColors[1] = 0.6; initColors[2] = 1.0; initColors[3] = 1.0
        // Down: brown/ground
        initColors[4] = 0.3; initColors[5] = 0.2; initColors[6] = 0.1; initColors[7] = 1.0
        // Left (-X): dark
        initColors[8] = 0.2; initColors[9] = 0.2; initColors[10] = 0.2; initColors[11] = 1.0
        // Right (+X): dark
        initColors[12] = 0.2; initColors[13] = 0.2; initColors[14] = 0.2; initColors[15] = 1.0
        // Front (+Z): dark
        initColors[16] = 0.2; initColors[17] = 0.2; initColors[18] = 0.2; initColors[19] = 1.0
        // Back (-Z): dark
        initColors[20] = 0.2; initColors[21] = 0.2; initColors[22] = 0.2; initColors[23] = 1.0
        device.queue.writeBuffer(this.faceColorsBuffer, 0, initColors)
    }

    async _createComputePipeline() {
        const { device } = this.engine

        // Bind group layout for reduction compute
        this.reduceBindGroupLayout = device.createBindGroupLayout({
            label: 'ambientReduceBindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        })

        // Create compute shader for reduction with temporal smoothing
        const shaderCode = `
            // Reduce a texture to a single average color using parallel reduction
            // Includes temporal smoothing to blend with previous frame's value
            // Workgroup: 8x8 threads, each samples region of the texture
            // Uses shared memory for efficient reduction

            struct ReduceParams {
                faceIndex: u32,
                textureSize: u32,
                blendFactor: f32,  // 0 = keep old, 1 = use new
                emissiveBoost: f32,
            }

            @group(0) @binding(0) var inputTexture: texture_2d<f32>;
            @group(0) @binding(1) var<storage, read_write> outputColors: array<vec4f>;
            @group(0) @binding(2) var<uniform> params: ReduceParams;

            var<workgroup> sharedColors: array<vec4f, 64>;

            @compute @workgroup_size(8, 8, 1)
            fn main(
                @builtin(local_invocation_id) localId: vec3u,
                @builtin(local_invocation_index) localIndex: u32
            ) {
                let faceIndex = params.faceIndex;
                let textureSize = params.textureSize;
                let tilesPerThread = textureSize / 8u;

                // Each thread samples a region and accumulates
                var sum = vec4f(0.0);
                let baseX = localId.x * tilesPerThread;
                let baseY = localId.y * tilesPerThread;

                for (var dy = 0u; dy < tilesPerThread; dy++) {
                    for (var dx = 0u; dx < tilesPerThread; dx++) {
                        let coord = vec2i(i32(baseX + dx), i32(baseY + dy));
                        var sample = textureLoad(inputTexture, coord, 0);

                        // Clamp max luminance to prevent sun/bright HDR from dominating
                        // This preserves color ratios while limiting brightness
                        var lum = max(sample.r, max(sample.g, sample.b));
                        let maxLum = 2.0;  // Cap brightness (sun can be 100+)
                        if (lum > maxLum) {
                            sample = sample * (maxLum / lum);
                            lum = maxLum;
                        }

                        // Gentle boost for emissive after clamping (logarithmic)
                        if (lum > 0.5 && params.emissiveBoost > 0.0) {
                            let boost = 1.0 + log2(lum + 1.0) * params.emissiveBoost;
                            sample = sample * boost;
                        }

                        sum += sample;
                    }
                }

                // Average this thread's samples
                let pixelCount = f32(tilesPerThread * tilesPerThread);
                sharedColors[localIndex] = sum / pixelCount;

                workgroupBarrier();

                // Parallel reduction: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
                if (localIndex < 32u) {
                    sharedColors[localIndex] += sharedColors[localIndex + 32u];
                }
                workgroupBarrier();

                if (localIndex < 16u) {
                    sharedColors[localIndex] += sharedColors[localIndex + 16u];
                }
                workgroupBarrier();

                if (localIndex < 8u) {
                    sharedColors[localIndex] += sharedColors[localIndex + 8u];
                }
                workgroupBarrier();

                if (localIndex < 4u) {
                    sharedColors[localIndex] += sharedColors[localIndex + 4u];
                }
                workgroupBarrier();

                if (localIndex < 2u) {
                    sharedColors[localIndex] += sharedColors[localIndex + 2u];
                }
                workgroupBarrier();

                // Final reduction with temporal smoothing
                if (localIndex == 0u) {
                    let finalSum = sharedColors[0] + sharedColors[1];
                    let newColor = finalSum / 64.0;

                    // Blend with previous value for temporal smoothing
                    let oldColor = outputColors[faceIndex];
                    let blendFactor = params.blendFactor;
                    outputColors[faceIndex] = mix(oldColor, newColor, blendFactor);
                }
            }
        `

        const shaderModule = device.createShaderModule({
            label: 'ambientReduceShader',
            code: shaderCode
        })

        this.reducePipeline = await device.createComputePipelineAsync({
            label: 'ambientReducePipeline',
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.reduceBindGroupLayout]
            }),
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        })

        // Create uniform buffers for each face (stores faceIndex, textureSize, blendFactor, emissiveBoost)
        this.reduceUniformBuffers = []
        this.reduceUniformData = []
        for (let i = 0; i < 6; i++) {
            const buffer = device.createBuffer({
                label: `ambientReduceParams_${i}`,
                size: 16,  // u32 + u32 + f32 + f32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            })
            // Initialize with default values
            const data = new ArrayBuffer(16)
            const view = new DataView(data)
            view.setUint32(0, i, true)  // faceIndex
            view.setUint32(4, this.captureSize, true)  // textureSize
            view.setFloat32(8, 1.0, true)  // blendFactor (1.0 = instant, for first frame)
            view.setFloat32(12, 2.0, true)  // emissiveBoost
            device.queue.writeBuffer(buffer, 0, data)
            this.reduceUniformBuffers.push(buffer)
            this.reduceUniformData.push(data)
        }

        // Smoothing time in seconds
        this.smoothingTime = 1.0
        this.lastUpdateTime = performance.now() / 1000

        // Create bind groups for each face
        this._createReduceBindGroups()
    }

    _createReduceBindGroups() {
        const { device } = this.engine

        this.reduceBindGroups = []
        const lightingOutput = this.lightingPass.getOutputTexture()

        for (let i = 0; i < 6; i++) {
            const bindGroup = device.createBindGroup({
                label: `ambientReduceBindGroup_${i}`,
                layout: this.reduceBindGroupLayout,
                entries: [
                    { binding: 0, resource: lightingOutput.view },
                    { binding: 1, resource: { buffer: this.faceColorsBuffer } },
                    { binding: 2, resource: { buffer: this.reduceUniformBuffers[i] } }
                ]
            })
            this.reduceBindGroups.push(bindGroup)
        }
    }

    /**
     * Set dependencies from main render pipeline
     */
    setDependencies(options) {
        const { environmentMap, encoding, shadowPass, noise, noiseSize } = options

        if (environmentMap) {
            this.lightingPass.setEnvironmentMap(environmentMap, encoding ?? 0)
        }
        if (shadowPass) {
            this.lightingPass.setShadowPass(shadowPass)
        }
        if (noise) {
            this.gbufferPass.setNoise(noise, noiseSize, false)
            this.lightingPass.setNoise(noise, noiseSize, false)
        }
    }

    /**
     * Calculate 6 world-space directions for ambient capture
     * Uses fixed world axes so shader can apply using world-space normals
     */
    _calculateDirections(camera) {
        // 6 directions in world space: up, down, left (-X), right (+X), front (+Z), back (-Z)
        // These match the shader's normal-direction mapping
        this.directions = [
            [0, 1, 0],    // up (+Y)
            [0, -1, 0],   // down (-Y)
            [-1, 0, 0],   // left (-X)
            [1, 0, 0],    // right (+X)
            [0, 0, 1],    // front (+Z)
            [0, 0, -1]    // back (-Z)
        ]
    }

    /**
     * Create a camera looking in a specific direction
     */
    _createFaceCamera(camera, direction, faceIndex) {
        const position = camera.position

        // Create look-at view matrix
        // Target = position + direction
        const target = [
            position[0] + direction[0],
            position[1] + direction[1],
            position[2] + direction[2]
        ]

        // Up vector: for up/down faces, use world +Z; otherwise use world up
        let up
        if (faceIndex === 0) {
            // Looking up (+Y): use +Z as up
            up = [0, 0, 1]
        } else if (faceIndex === 1) {
            // Looking down (-Y): use +Z as up
            up = [0, 0, 1]
        } else {
            // Horizontal directions: use world up (+Y)
            up = [0, 1, 0]
        }

        mat4.lookAt(this.faceView, position, target, up)

        // 90 degree FOV perspective projection
        const fov = Math.PI / 2  // 90 degrees
        const aspect = 1.0
        const near = 0.1
        const far = this.maxDistance
        mat4.perspective(this.faceProj, fov, aspect, near, far)

        // Create inverse matrices
        const iView = mat4.create()
        const iProj = mat4.create()
        const viewProj = mat4.create()
        const iViewProj = mat4.create()
        mat4.invert(iView, this.faceView)
        mat4.invert(iProj, this.faceProj)
        mat4.multiply(viewProj, this.faceProj, this.faceView)
        mat4.invert(iViewProj, viewProj)

        return {
            view: this.faceView,
            proj: this.faceProj,
            iView,
            iProj,
            iViewProj,
            viewProj,
            position,
            near,
            far,
            aspect,
            jitterEnabled: false,
            jitterOffset: [0, 0],
            screenSize: [this.captureSize, this.captureSize],
            forward: direction,
            updateMatrix: () => {},
            updateView: () => {}
        }
    }

    /**
     * Execute ambient capture pass
     */
    async _execute(context) {
        const { device, stats } = this.engine
        const { camera, meshes, lights, mainLight, dt = 0.016 } = context

        // Process deferred texture destruction
        this._pendingDestroyIndex = (this._pendingDestroyIndex + 1) % 3
        const toDestroy = this._pendingDestroyRing[this._pendingDestroyIndex]
        for (const tex of toDestroy) {
            tex.destroy()
        }
        this._pendingDestroyRing[this._pendingDestroyIndex] = []

        // Check if enabled
        if (!this.settings?.ambientCapture?.enabled) {
            return
        }

        // Update settings
        this.maxDistance = this.settings?.ambientCapture?.maxDistance ?? 25
        this.smoothingTime = this.settings?.ambientCapture?.smoothingTime ?? 1.0

        // Calculate directions based on camera orientation
        this._calculateDirections(camera)

        // Copy lights to internal lighting pass
        this.lightingPass.lights = lights

        // Determine which 2 faces to update this frame
        const facesToUpdate = [
            this.currentFaceIndex,
            (this.currentFaceIndex + 1) % 6
        ]

        // Render and reduce each face with temporal smoothing
        for (const faceIndex of facesToUpdate) {
            await this._renderFace(context, faceIndex)
            this._reduceFace(faceIndex, dt)
        }

        // Advance to next pair of faces
        this.currentFaceIndex = (this.currentFaceIndex + 2) % 6
    }

    async _renderFace(context, faceIndex) {
        const { camera, meshes, lights, mainLight, dt } = context

        const direction = this.directions[faceIndex]
        const faceCamera = this._createFaceCamera(camera, direction, faceIndex)

        // Execute GBuffer pass
        await this.gbufferPass.execute({
            camera: faceCamera,
            meshes,
            dt,
            // Custom culling for ambient capture
            cullingOverride: {
                maxDistance: this.maxDistance,
                frustum: true,
                hiZ: false
            }
        })

        // Execute Lighting pass
        await this.lightingPass.execute({
            camera: faceCamera,
            meshes,
            dt,
            lights,
            mainLight
        })
    }

    _reduceFace(faceIndex, dt) {
        const { device } = this.engine

        // Calculate blend factor for temporal smoothing
        // blend = 1 - exp(-dt / smoothingTime) gives exponential smoothing
        // With smoothingTime=1s, after 1s we've blended ~63% of the way to target
        const blendFactor = 1.0 - Math.exp(-dt / this.smoothingTime)

        // Get emissive boost from settings
        const emissiveBoost = this.settings?.ambientCapture?.emissiveBoost ?? 2.0

        // Update uniform buffer with blend factor and emissive boost
        const data = this.reduceUniformData[faceIndex]
        const view = new DataView(data)
        view.setFloat32(8, blendFactor, true)  // blendFactor
        view.setFloat32(12, emissiveBoost, true)  // emissiveBoost
        device.queue.writeBuffer(this.reduceUniformBuffers[faceIndex], 0, data)

        const commandEncoder = device.createCommandEncoder({
            label: `ambientReduce_face${faceIndex}`
        })

        const computePass = commandEncoder.beginComputePass({
            label: `ambientReducePass_face${faceIndex}`
        })

        computePass.setPipeline(this.reducePipeline)
        computePass.setBindGroup(0, this.reduceBindGroups[faceIndex])
        computePass.dispatchWorkgroups(1)  // Single workgroup handles entire 64x64 texture
        computePass.end()

        device.queue.submit([commandEncoder.finish()])
    }

    /**
     * Get the face colors buffer for RenderPost
     */
    getFaceColorsBuffer() {
        return this.faceColorsBuffer
    }

    /**
     * Get the 6 directions (for RenderPost to know orientation)
     */
    getDirections() {
        return this.directions
    }

    async _resize(width, height) {
        // Ambient capture doesn't resize with screen - fixed resolution
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
        if (this.dummyAO?.texture) {
            this.dummyAO.texture.destroy()
            this.dummyAO = null
        }
        if (this.faceColorsBuffer) {
            this.faceColorsBuffer.destroy()
            this.faceColorsBuffer = null
        }
        for (const buffer of this.reduceUniformBuffers || []) {
            buffer.destroy()
        }
        this.reduceUniformBuffers = []
        for (const slot of this._pendingDestroyRing) {
            for (const tex of slot) {
                tex.destroy()
            }
        }
        this._pendingDestroyRing = [[], [], []]
    }
}

export { AmbientCapturePass }
