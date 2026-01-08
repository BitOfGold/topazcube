import { Texture, generateMips, numMipLevels } from "../Texture.js"
import { mat4, vec3 } from "../math.js"

/**
 * ProbeCapture - Captures environment reflections from a position
 *
 * Renders 6 cube faces and encodes to octahedral format.
 * Can export as HDR PNG for server storage.
 */
class ProbeCapture {
    constructor(engine) {
        this.engine = engine

        // Capture resolution per cube face
        this.faceSize = 1024

        // Output octahedral texture size
        this.octahedralSize = 4096

        // Cube face render targets (6 faces)
        this.faceTextures = []

        // Depth textures for each face
        this.faceDepthTextures = []

        // Output octahedral texture
        this.octahedralTexture = null

        // Mip levels for roughness
        this.mipLevels = 6

        // Previous environment map (used during capture to avoid recursion)
        this.fallbackEnvironment = null

        // Environment encoding: 0 = equirectangular, 1 = octahedral
        this.envEncoding = 0

        // Scene render callback (set by RenderGraph)
        this.sceneRenderCallback = null

        // Capture state
        this.isCapturing = false
        this.capturePosition = [0, 0, 0]

        // Cube face camera directions
        // +X, -X, +Y, -Y, +Z, -Z
        this.faceDirections = [
            { dir: [1, 0, 0], up: [0, 1, 0] },   // +X
            { dir: [-1, 0, 0], up: [0, 1, 0] },  // -X
            { dir: [0, 1, 0], up: [0, 0, -1] },  // +Y (up)
            { dir: [0, -1, 0], up: [0, 0, 1] },  // -Y (down)
            { dir: [0, 0, 1], up: [0, 1, 0] },   // +Z
            { dir: [0, 0, -1], up: [0, 1, 0] },  // -Z
        ]

        // Pipelines
        this.skyboxPipeline = null
        this.scenePipeline = null
        this.convertPipeline = null
        this.faceBindGroupLayout = null
        this.faceSampler = null

        // Scene rendering resources
        this.sceneBindGroupLayout = null
        this.sceneUniformBuffer = null
    }

    /**
     * Set scene render callback - called to render scene for each face
     * @param {Function} callback - (viewMatrix, projMatrix, colorTarget, depthTarget) => void
     */
    setSceneRenderCallback(callback) {
        this.sceneRenderCallback = callback
    }

    /**
     * Initialize capture resources
     */
    async initialize() {
        const { device } = this.engine

        // Create 6 face render targets with depth
        for (let i = 0; i < 6; i++) {
            // Color target (needs COPY_DST for scene render callback, COPY_SRC for debug save)
            const colorTexture = device.createTexture({
                label: `probeFace${i}`,
                size: [this.faceSize, this.faceSize],
                format: 'rgba16float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
            })
            this.faceTextures.push({
                texture: colorTexture,
                view: colorTexture.createView()
            })

            // Depth target (depth32float to match GBuffer depth for copying)
            const depthTexture = device.createTexture({
                label: `probeFaceDepth${i}`,
                size: [this.faceSize, this.faceSize],
                format: 'depth32float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
            })
            this.faceDepthTextures.push({
                texture: depthTexture,
                view: depthTexture.createView()
            })
        }

        // Create octahedral output texture
        const octTexture = device.createTexture({
            label: 'probeOctahedral',
            size: [this.octahedralSize, this.octahedralSize],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
        })
        this.octahedralTexture = {
            texture: octTexture,
            view: octTexture.createView()
        }

        // Create sampler
        this.faceSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        })

        // Create skybox render pipeline (samples environment as background)
        await this._createSkyboxPipeline()

        // Create compute pipeline for cubemap → octahedral conversion
        await this._createConvertPipeline()
    }

    /**
     * Set fallback environment map (used during capture)
     */
    setFallbackEnvironment(envMap) {
        this.fallbackEnvironment = envMap
    }

    /**
     * Create pipeline to render skybox (environment map as background)
     */
    async _createSkyboxPipeline() {
        const { device } = this.engine

        const shaderCode = /* wgsl */`
            struct Uniforms {
                invViewProj: mat4x4f,
                envEncoding: f32,       // 0 = equirectangular, 1 = octahedral
                padding: vec3f,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var envTexture: texture_2d<f32>;
            @group(0) @binding(2) var envSampler: sampler;

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) uv: vec2f,
            }

            @vertex
            fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                var output: VertexOutput;
                let x = f32(vertexIndex & 1u) * 4.0 - 1.0;
                let y = f32(vertexIndex >> 1u) * 4.0 - 1.0;
                output.position = vec4f(x, y, 0.0, 1.0);
                output.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
                return output;
            }

            const PI: f32 = 3.14159265359;

            // Equirectangular UV mapping
            fn SphToUV(n: vec3f) -> vec2f {
                var uv: vec2f;
                uv.x = atan2(-n.x, n.z);
                uv.x = (uv.x + PI / 2.0) / (PI * 2.0) + PI * (28.670 / 360.0);
                uv.y = acos(n.y) / PI;
                return uv;
            }

            // Octahedral UV mapping
            fn octEncode(n: vec3f) -> vec2f {
                var p = n.xz / (abs(n.x) + abs(n.y) + abs(n.z));
                if (n.y < 0.0) {
                    p = (1.0 - abs(p.yx)) * vec2f(
                        select(-1.0, 1.0, p.x >= 0.0),
                        select(-1.0, 1.0, p.y >= 0.0)
                    );
                }
                return p * 0.5 + 0.5;
            }

            fn getEnvUV(dir: vec3f) -> vec2f {
                if (uniforms.envEncoding > 0.5) {
                    return octEncode(dir);
                }
                return SphToUV(dir);
            }

            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                // Convert UV to clip space
                let clipPos = vec4f(input.uv * 2.0 - 1.0, 1.0, 1.0);

                // Transform to world direction
                let worldPos = uniforms.invViewProj * clipPos;
                var dir = normalize(worldPos.xyz / worldPos.w);

                // For octahedral, negate direction (same as main lighting shader)
                if (uniforms.envEncoding > 0.5) {
                    dir = -dir;
                }

                // Sample environment map using correct UV mapping
                let uv = getEnvUV(dir);
                let envRGBE = textureSample(envTexture, envSampler, uv);
                var color = envRGBE.rgb * pow(2.0, envRGBE.a * 255.0 - 128.0);

                // No exposure or environment levels for probe capture
                // We capture raw HDR values - exposure is applied in main render only
                return vec4f(color, 1.0);
            }
        `

        const shaderModule = device.createShaderModule({
            label: 'probeSkybox',
            code: shaderCode
        })

        this.faceBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
            ]
        })

        this.skyboxPipeline = device.createRenderPipeline({
            label: 'probeSkybox',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.faceBindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'rgba16float' }]
            },
            primitive: { topology: 'triangle-list' },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: false,  // Don't write depth
                depthCompare: 'greater-equal',  // Only draw where depth is at far plane (1.0)
            }
        })
    }

    /**
     * Create compute pipeline for cubemap to octahedral conversion
     */
    async _createConvertPipeline() {
        const { device } = this.engine

        const shaderCode = /* wgsl */`
            @group(0) @binding(0) var outputTex: texture_storage_2d<rgba16float, write>;
            @group(0) @binding(1) var cubeFace0: texture_2d<f32>;
            @group(0) @binding(2) var cubeFace1: texture_2d<f32>;
            @group(0) @binding(3) var cubeFace2: texture_2d<f32>;
            @group(0) @binding(4) var cubeFace3: texture_2d<f32>;
            @group(0) @binding(5) var cubeFace4: texture_2d<f32>;
            @group(0) @binding(6) var cubeFace5: texture_2d<f32>;
            @group(0) @binding(7) var cubeSampler: sampler;

            // Octahedral decode: UV to direction
            // Maps 2D octahedral UV to 3D direction vector
            fn octDecode(uv: vec2f) -> vec3f {
                var uv2 = uv * 2.0 - 1.0;

                // Upper hemisphere mapping
                var n = vec3f(uv2.x, 1.0 - abs(uv2.x) - abs(uv2.y), uv2.y);

                // Lower hemisphere wrapping
                if (n.y < 0.0) {
                    let signX = select(-1.0, 1.0, n.x >= 0.0);
                    let signZ = select(-1.0, 1.0, n.z >= 0.0);
                    n = vec3f(
                        (1.0 - abs(n.z)) * signX,
                        n.y,
                        (1.0 - abs(n.x)) * signZ
                    );
                }
                return normalize(n);
            }

            // Sample cubemap from direction
            // Standard cubemap UV mapping for faces rendered with lookAt
            // V coordinate is negated for Y because texture v=0 is top, world Y=+1 is up
            // Y faces use Z for vertical since they look along Y axis with Z as up vector
            fn sampleCube(dir: vec3f) -> vec4f {
                let absDir = abs(dir);
                var uv: vec2f;
                var faceColor: vec4f;

                if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
                    if (dir.x > 0.0) {
                        // +X face: use face1
                        uv = vec2f(-dir.z, -dir.y) / absDir.x * 0.5 + 0.5;
                        faceColor = textureSampleLevel(cubeFace1, cubeSampler, uv, 0.0);
                    } else {
                        // -X face: use face0
                        uv = vec2f(dir.z, -dir.y) / absDir.x * 0.5 + 0.5;
                        faceColor = textureSampleLevel(cubeFace0, cubeSampler, uv, 0.0);
                    }
                } else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
                    if (dir.y > 0.0) {
                        // +Y face: looking at +Y (up), up vector = -Z, right = +X
                        // Screen top (-Z) should map to v=0, so use +z
                        uv = vec2f(dir.x, dir.z) / absDir.y * 0.5 + 0.5;
                        faceColor = textureSampleLevel(cubeFace2, cubeSampler, uv, 0.0);
                    } else {
                        // -Y face: looking at -Y (down), up vector = +Z, right = +X
                        // Screen top (+Z) should map to v=0, so use -z
                        uv = vec2f(dir.x, -dir.z) / absDir.y * 0.5 + 0.5;
                        faceColor = textureSampleLevel(cubeFace3, cubeSampler, uv, 0.0);
                    }
                } else {
                    if (dir.z > 0.0) {
                        // +Z face: looking at +Z, up = +Y, right = +X
                        uv = vec2f(dir.x, -dir.y) / absDir.z * 0.5 + 0.5;
                        faceColor = textureSampleLevel(cubeFace4, cubeSampler, uv, 0.0);
                    } else {
                        // -Z face: looking at -Z, up = +Y, right = -X
                        uv = vec2f(-dir.x, -dir.y) / absDir.z * 0.5 + 0.5;
                        faceColor = textureSampleLevel(cubeFace5, cubeSampler, uv, 0.0);
                    }
                }

                return faceColor;
            }

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) gid: vec3u) {
                let size = textureDimensions(outputTex);
                if (gid.x >= size.x || gid.y >= size.y) {
                    return;
                }

                let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
                let dir = octDecode(uv);
                let color = sampleCube(dir);
                textureStore(outputTex, gid.xy, color);
            }
        `

        const shaderModule = device.createShaderModule({
            label: 'probeConvert',
            code: shaderCode
        })

        this.convertPipeline = device.createComputePipeline({
            label: 'probeConvert',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        })

        // Create equirectangular to octahedral conversion pipeline
        await this._createEquirectToOctPipeline()
    }

    /**
     * Create compute pipeline for equirectangular to octahedral conversion
     */
    async _createEquirectToOctPipeline() {
        const { device } = this.engine

        const shaderCode = /* wgsl */`
            @group(0) @binding(0) var outputTex: texture_storage_2d<rgba16float, write>;
            @group(0) @binding(1) var envTexture: texture_2d<f32>;
            @group(0) @binding(2) var envSampler: sampler;

            const PI: f32 = 3.14159265359;

            // Octahedral decode: UV to direction
            fn octDecode(uv: vec2f) -> vec3f {
                var uv2 = uv * 2.0 - 1.0;
                var n = vec3f(uv2.x, 1.0 - abs(uv2.x) - abs(uv2.y), uv2.y);
                if (n.y < 0.0) {
                    let signX = select(-1.0, 1.0, n.x >= 0.0);
                    let signZ = select(-1.0, 1.0, n.z >= 0.0);
                    n = vec3f(
                        (1.0 - abs(n.z)) * signX,
                        n.y,
                        (1.0 - abs(n.x)) * signZ
                    );
                }
                return normalize(n);
            }

            // Equirectangular UV from direction (matching lighting.wgsl SphToUV)
            fn SphToUV(n: vec3f) -> vec2f {
                var uv: vec2f;
                uv.x = atan2(-n.x, n.z);
                uv.x = (uv.x + PI / 2.0) / (PI * 2.0) + PI * (28.670 / 360.0);
                uv.y = acos(n.y) / PI;
                return uv;
            }

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) gid: vec3u) {
                let size = textureDimensions(outputTex);
                if (gid.x >= size.x || gid.y >= size.y) {
                    return;
                }

                let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
                let dir = octDecode(uv);

                // Sample equirectangular environment
                let envUV = SphToUV(dir);
                let envRGBE = textureSampleLevel(envTexture, envSampler, envUV, 0.0);

                // Decode RGBE to linear HDR
                let color = envRGBE.rgb * pow(2.0, envRGBE.a * 255.0 - 128.0);

                textureStore(outputTex, gid.xy, vec4f(color, 1.0));
            }
        `;

        const shaderModule = device.createShaderModule({
            label: 'equirectToOct',
            code: shaderCode
        })

        this.equirectToOctPipeline = device.createComputePipeline({
            label: 'equirectToOct',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        })
    }

    /**
     * Convert equirectangular environment map to octahedral format
     * @param {Texture} envMap - Equirectangular RGBE environment map
     */
    async convertEquirectToOctahedral(envMap) {
        const { device } = this.engine

        if (!this.equirectToOctPipeline) {
            await this._createEquirectToOctPipeline()
        }

        // Reset RGBE texture so it gets regenerated
        if (this.octahedralRGBE?.texture) {
            this.octahedralRGBE.texture.destroy()
            this.octahedralRGBE = null
        }

        const bindGroup = device.createBindGroup({
            layout: this.equirectToOctPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.octahedralTexture.view },
                { binding: 1, resource: envMap.view },
                { binding: 2, resource: this.faceSampler },
            ]
        })

        const commandEncoder = device.createCommandEncoder()
        const passEncoder = commandEncoder.beginComputePass()

        passEncoder.setPipeline(this.equirectToOctPipeline)
        passEncoder.setBindGroup(0, bindGroup)

        const workgroupsX = Math.ceil(this.octahedralSize / 8)
        const workgroupsY = Math.ceil(this.octahedralSize / 8)
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY)

        passEncoder.end()
        device.queue.submit([commandEncoder.finish()])

        // Wait for GPU to finish
        await device.queue.onSubmittedWorkDone()

        console.log('ProbeCapture: Converted equirectangular to octahedral')
    }

    /**
     * Create view/projection matrix for a cube face
     */
    _createFaceMatrix(faceIndex, position) {
        const face = this.faceDirections[faceIndex]

        const view = mat4.create()
        const target = [
            position[0] + face.dir[0],
            position[1] + face.dir[1],
            position[2] + face.dir[2]
        ]
        mat4.lookAt(view, position, target, face.up)

        const proj = mat4.create()
        mat4.perspective(proj, Math.PI / 2, 1.0, 0.1, 1000.0)

        const viewProj = mat4.create()
        mat4.multiply(viewProj, proj, view)

        const invViewProj = mat4.create()
        mat4.invert(invViewProj, viewProj)

        return invViewProj
    }

    /**
     * Create view and projection matrices for a cube face
     * Returns both matrices separately (for scene rendering)
     */
    _createFaceMatrices(faceIndex, position) {
        const face = this.faceDirections[faceIndex]

        const view = mat4.create()
        const target = [
            position[0] + face.dir[0],
            position[1] + face.dir[1],
            position[2] + face.dir[2]
        ]
        mat4.lookAt(view, position, target, face.up)

        const proj = mat4.create()
        mat4.perspective(proj, Math.PI / 2, 1.0, 0.1, 10000.0)  // Far plane 10000 for distant objects

        return { view, proj }
    }

    /**
     * Capture probe from a position
     * Renders scene geometry (if callback set) then skybox as background
     * @param {vec3} position - World position to capture from
     * @returns {Promise<void>}
     */
    async capture(position) {
        if (this.isCapturing) {
            console.warn('ProbeCapture: Already capturing')
            return
        }

        if (!this.fallbackEnvironment) {
            console.error('ProbeCapture: No environment map set')
            return
        }

        this.isCapturing = true
        this.capturePosition = [...position]

        // Reset RGBE texture so it gets regenerated after new capture
        if (this.octahedralRGBE?.texture) {
            this.octahedralRGBE.texture.destroy()
            this.octahedralRGBE = null
        }

        const { device } = this.engine

        // Wait for any previous GPU work to complete before starting capture
        // This ensures all previous frame's buffer writes are complete
        await device.queue.onSubmittedWorkDone()

        // Capture started - position logged by caller

        try {
            // Create uniform buffer for skybox (mat4x4 + envEncoding + padding)
            const uniformBuffer = device.createBuffer({
                size: 80, // mat4x4 (64) + vec4 (16) for envEncoding + padding
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            })

            // Render each cube face
            for (let i = 0; i < 6; i++) {
                const { view, proj } = this._createFaceMatrices(i, position)
                const invViewProj = this._createFaceMatrix(i, position)

                if (this.sceneRenderCallback) {
                    // Render scene using RenderGraph's deferred pipeline
                    // LightingPass handles BOTH scene geometry AND background (environment)
                    // No separate skybox render needed - LightingPass does it all
                    await this.sceneRenderCallback(
                        view,
                        proj,
                        this.faceTextures[i],
                        this.faceDepthTextures[i],
                        i,
                        position
                    )
                } else {
                    // Fallback: No scene callback - render skybox only
                    const commandEncoder = device.createCommandEncoder()
                    const passEncoder = commandEncoder.beginRenderPass({
                        colorAttachments: [{
                            view: this.faceTextures[i].view,
                            clearValue: { r: 0, g: 0, b: 0, a: 0 },
                            loadOp: 'clear',
                            storeOp: 'store'
                        }],
                        depthStencilAttachment: {
                            view: this.faceDepthTextures[i].view,
                            depthClearValue: 1.0,
                            depthLoadOp: 'clear',
                            depthStoreOp: 'store'
                        }
                    })
                    passEncoder.end()
                    device.queue.submit([commandEncoder.finish()])

                    // Render skybox for fallback path only
                    const uniformData = new Float32Array(20)
                    uniformData.set(invViewProj, 0)
                    uniformData[16] = this.envEncoding  // 0 = equirect, 1 = octahedral
                    uniformData[17] = 0  // padding
                    uniformData[18] = 0  // padding
                    uniformData[19] = 0  // padding
                    device.queue.writeBuffer(uniformBuffer, 0, uniformData)

                    const skyboxBindGroup = device.createBindGroup({
                        layout: this.faceBindGroupLayout,
                        entries: [
                            { binding: 0, resource: { buffer: uniformBuffer } },
                            { binding: 1, resource: this.fallbackEnvironment.view },
                            { binding: 2, resource: this.faceSampler },
                        ]
                    })

                    const skyboxEncoder = device.createCommandEncoder()
                    const skyboxPass = skyboxEncoder.beginRenderPass({
                        colorAttachments: [{
                            view: this.faceTextures[i].view,
                            loadOp: 'load',
                            storeOp: 'store'
                        }],
                        depthStencilAttachment: {
                            view: this.faceDepthTextures[i].view,
                            depthLoadOp: 'load',
                            depthStoreOp: 'store'
                        }
                    })

                    skyboxPass.setPipeline(this.skyboxPipeline)
                    skyboxPass.setBindGroup(0, skyboxBindGroup)
                    skyboxPass.draw(3)
                    skyboxPass.end()

                    device.queue.submit([skyboxEncoder.finish()])
                }
            }

            uniformBuffer.destroy()

            // Convert cubemap to octahedral
            await this._convertToOctahedral()

            // Capture complete

        } finally {
            this.isCapturing = false
        }
    }

    /**
     * Convert 6 cube faces to octahedral encoding
     */
    async _convertToOctahedral() {
        const { device } = this.engine

        const bindGroup = device.createBindGroup({
            layout: this.convertPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.octahedralTexture.view },
                { binding: 1, resource: this.faceTextures[0].view },
                { binding: 2, resource: this.faceTextures[1].view },
                { binding: 3, resource: this.faceTextures[2].view },
                { binding: 4, resource: this.faceTextures[3].view },
                { binding: 5, resource: this.faceTextures[4].view },
                { binding: 6, resource: this.faceTextures[5].view },
                { binding: 7, resource: this.faceSampler },
            ]
        })

        const commandEncoder = device.createCommandEncoder()
        const passEncoder = commandEncoder.beginComputePass()

        passEncoder.setPipeline(this.convertPipeline)
        passEncoder.setBindGroup(0, bindGroup)

        const workgroupsX = Math.ceil(this.octahedralSize / 8)
        const workgroupsY = Math.ceil(this.octahedralSize / 8)
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY)

        passEncoder.end()
        device.queue.submit([commandEncoder.finish()])
    }

    /**
     * Export probe as DEBUG PNG (regular LDR, tone mapped) for visual inspection
     * @param {string} filename - Download filename
     * @param {number} exposure - Exposure value for tone mapping (uses engine setting if not provided)
     */
    async saveAsDebugPNG(filename = 'probe_debug.png', exposure = null) {
        // Use engine's exposure setting if not explicitly provided
        if (exposure === null) {
            exposure = this.engine?.settings?.environment?.exposure ?? 1.6
        }
        const { device } = this.engine

        // Read back texture data
        const bytesPerPixel = 8 // rgba16float = 4 * 2 bytes
        const bytesPerRow = Math.ceil(this.octahedralSize * bytesPerPixel / 256) * 256
        const bufferSize = bytesPerRow * this.octahedralSize

        const readBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })

        const commandEncoder = device.createCommandEncoder()
        commandEncoder.copyTextureToBuffer(
            { texture: this.octahedralTexture.texture },
            { buffer: readBuffer, bytesPerRow: bytesPerRow },
            { width: this.octahedralSize, height: this.octahedralSize }
        )
        device.queue.submit([commandEncoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)
        const data = new Uint16Array(readBuffer.getMappedRange())

        // Convert float16 to regular 8-bit RGBA with ACES tone mapping (matches PostProcess)
        const pixelsPerRow = bytesPerRow / 8  // 8 bytes per pixel (rgba16float)
        const rgbaData = new Uint8ClampedArray(this.octahedralSize * this.octahedralSize * 4)

        for (let y = 0; y < this.octahedralSize; y++) {
            for (let x = 0; x < this.octahedralSize; x++) {
                const srcIdx = (y * pixelsPerRow + x) * 4
                const dstIdx = (y * this.octahedralSize + x) * 4

                // Decode float16 values
                const r = this._float16ToFloat32(data[srcIdx])
                const g = this._float16ToFloat32(data[srcIdx + 1])
                const b = this._float16ToFloat32(data[srcIdx + 2])

                // ACES tone mapping (matches postproc.wgsl)
                const [tr, tg, tb] = this._acesToneMap(r, g, b)

                // ACES already outputs in sRGB, just clamp and convert to 8-bit
                rgbaData[dstIdx] = Math.min(255, Math.max(0, tr * 255))
                rgbaData[dstIdx + 1] = Math.min(255, Math.max(0, tg * 255))
                rgbaData[dstIdx + 2] = Math.min(255, Math.max(0, tb * 255))
                rgbaData[dstIdx + 3] = 255
            }
        }

        readBuffer.unmap()
        readBuffer.destroy()

        // Create canvas and draw image data
        const canvas = document.createElement('canvas')
        canvas.width = this.octahedralSize
        canvas.height = this.octahedralSize
        const ctx = canvas.getContext('2d')
        const imageData = new ImageData(rgbaData, this.octahedralSize, this.octahedralSize)
        ctx.putImageData(imageData, 0, 0)

        // Trigger download
        const link = document.createElement('a')
        link.download = filename
        link.href = canvas.toDataURL('image/png')
        link.click()

        console.log(`ProbeCapture: Saved debug PNG as ${filename}`)
    }

    /**
     * Save individual cube faces for debugging
     * @param {string} prefix - Filename prefix
     * @param {number} exposure - Exposure value for tone mapping (uses engine setting if not provided)
     */
    async saveCubeFaces(prefix = 'face', exposure = null) {
        // Use engine's exposure setting if not explicitly provided
        if (exposure === null) {
            exposure = this.engine?.settings?.environment?.exposure ?? 1.6
        }
        const { device } = this.engine
        const faceNames = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']

        for (let i = 0; i < 6; i++) {
            const bytesPerPixel = 8
            const bytesPerRow = Math.ceil(this.faceSize * bytesPerPixel / 256) * 256
            const bufferSize = bytesPerRow * this.faceSize

            const readBuffer = device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            })

            const commandEncoder = device.createCommandEncoder()
            commandEncoder.copyTextureToBuffer(
                { texture: this.faceTextures[i].texture },
                { buffer: readBuffer, bytesPerRow },
                { width: this.faceSize, height: this.faceSize }
            )
            device.queue.submit([commandEncoder.finish()])

            await readBuffer.mapAsync(GPUMapMode.READ)
            const data = new Uint16Array(readBuffer.getMappedRange())

            const pixelsPerRow = bytesPerRow / 8
            const rgbaData = new Uint8ClampedArray(this.faceSize * this.faceSize * 4)

            for (let y = 0; y < this.faceSize; y++) {
                for (let x = 0; x < this.faceSize; x++) {
                    const srcIdx = (y * pixelsPerRow + x) * 4
                    const dstIdx = (y * this.faceSize + x) * 4

                    const r = this._float16ToFloat32(data[srcIdx])
                    const g = this._float16ToFloat32(data[srcIdx + 1])
                    const b = this._float16ToFloat32(data[srcIdx + 2])

                    // ACES tone mapping (matches postproc.wgsl)
                    const [tr, tg, tb] = this._acesToneMap(r, g, b)

                    rgbaData[dstIdx] = Math.min(255, Math.max(0, tr * 255))
                    rgbaData[dstIdx + 1] = Math.min(255, Math.max(0, tg * 255))
                    rgbaData[dstIdx + 2] = Math.min(255, Math.max(0, tb * 255))
                    rgbaData[dstIdx + 3] = 255
                }
            }

            readBuffer.unmap()
            readBuffer.destroy()

            const canvas = document.createElement('canvas')
            canvas.width = this.faceSize
            canvas.height = this.faceSize
            const ctx = canvas.getContext('2d')
            const imageData = new ImageData(rgbaData, this.faceSize, this.faceSize)
            ctx.putImageData(imageData, 0, 0)

            const link = document.createElement('a')
            link.download = `${prefix}_${i}_${faceNames[i].replace(/[+-]/g, m => m === '+' ? 'pos' : 'neg')}.png`
            link.href = canvas.toDataURL('image/png')
            link.click()

            // Small delay between downloads
            await new Promise(r => setTimeout(r, 100))
        }

        console.log('ProbeCapture: Saved all 6 cube faces')
    }

    /**
     * Convert float16 to float32
     */
    _float16ToFloat32(h) {
        const s = (h & 0x8000) >> 15
        const e = (h & 0x7C00) >> 10
        const f = h & 0x03FF

        if (e === 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
        } else if (e === 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity)
        }

        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
    }

    /**
     * ACES tone mapping (matches postproc.wgsl)
     * @param {number} r - Red HDR value
     * @param {number} g - Green HDR value
     * @param {number} b - Blue HDR value
     * @returns {number[]} Tone-mapped [r, g, b] in 0-1 range
     */
    _acesToneMap(r, g, b) {
        // Input transform matrix (RRT_SAT)
        const m1 = [
            [0.59719, 0.35458, 0.04823],
            [0.07600, 0.90834, 0.01566],
            [0.02840, 0.13383, 0.83777]
        ]
        // Output transform matrix (ODT_SAT)
        const m2 = [
            [ 1.60475, -0.53108, -0.07367],
            [-0.10208,  1.10813, -0.00605],
            [-0.00327, -0.07276,  1.07602]
        ]

        // Apply input transform
        const v0 = m1[0][0] * r + m1[0][1] * g + m1[0][2] * b
        const v1 = m1[1][0] * r + m1[1][1] * g + m1[1][2] * b
        const v2 = m1[2][0] * r + m1[2][1] * g + m1[2][2] * b

        // RRT and ODT fit
        const a0 = v0 * (v0 + 0.0245786) - 0.000090537
        const b0 = v0 * (0.983729 * v0 + 0.4329510) + 0.238081
        const a1 = v1 * (v1 + 0.0245786) - 0.000090537
        const b1 = v1 * (0.983729 * v1 + 0.4329510) + 0.238081
        const a2 = v2 * (v2 + 0.0245786) - 0.000090537
        const b2 = v2 * (0.983729 * v2 + 0.4329510) + 0.238081

        const c0 = a0 / b0
        const c1 = a1 / b1
        const c2 = a2 / b2

        // Apply output transform
        const out0 = m2[0][0] * c0 + m2[0][1] * c1 + m2[0][2] * c2
        const out1 = m2[1][0] * c0 + m2[1][1] * c1 + m2[1][2] * c2
        const out2 = m2[2][0] * c0 + m2[2][1] * c1 + m2[2][2] * c2

        // Clamp to 0-1
        return [
            Math.max(0, Math.min(1, out0)),
            Math.max(0, Math.min(1, out1)),
            Math.max(0, Math.min(1, out2))
        ]
    }

    /**
     * Export probe as two JPG files: RGB + Multiplier (RGBM format)
     * RGB stores actual color (with sRGB gamma) - values <= 1.0 stored directly
     * Multiplier only encodes values > 1.0: black = 1.0, white = 32768, logarithmic
     * This means most pixels have multiplier = 1.0 (black), compressing very well
     * Blue noise dithering is applied to reduce banding
     * @param {string} basename - Base filename (without extension), e.g. 'probe_01'
     * @param {number} quality - JPG quality 0-1 (default 0.95 = 95%)
     */
    async saveAsJPG(basename = 'probe_01', quality = 0.95) {
        const { device } = this.engine

        // Load blue noise texture for dithering
        let blueNoiseData = null
        let blueNoiseSize = 1024
        try {
            const response = await fetch('/bluenoise1024.png')
            const blob = await response.blob()
            const bitmap = await createImageBitmap(blob)
            blueNoiseSize = bitmap.width
            const noiseCanvas = document.createElement('canvas')
            noiseCanvas.width = blueNoiseSize
            noiseCanvas.height = blueNoiseSize
            const noiseCtx = noiseCanvas.getContext('2d')
            noiseCtx.drawImage(bitmap, 0, 0)
            blueNoiseData = noiseCtx.getImageData(0, 0, blueNoiseSize, blueNoiseSize).data
        } catch (e) {
            console.warn('ProbeCapture: Could not load blue noise, using white noise fallback')
        }

        // Read back texture data
        const bytesPerPixel = 8 // rgba16float = 4 * 2 bytes
        const bytesPerRow = Math.ceil(this.octahedralSize * bytesPerPixel / 256) * 256
        const bufferSize = bytesPerRow * this.octahedralSize

        const readBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })

        const commandEncoder = device.createCommandEncoder()
        commandEncoder.copyTextureToBuffer(
            { texture: this.octahedralTexture.texture },
            { buffer: readBuffer, bytesPerRow: bytesPerRow },
            { width: this.octahedralSize, height: this.octahedralSize }
        )
        device.queue.submit([commandEncoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)
        const float16Data = new Uint16Array(readBuffer.getMappedRange())

        // RGBM encoding parameters
        // Multiplier range: 1.0 (black) to 32768 (white), logarithmic
        // Most pixels will have multiplier = 1.0, so intensity image is mostly black
        const MULT_MAX = 32768  // 2^15
        const LOG_MULT_MAX = 15  // log2(32768)
        const SRGB_GAMMA = 2.2

        const size = this.octahedralSize
        const rgbData = new Uint8ClampedArray(size * size * 4)  // RGBA for canvas
        const multData = new Uint8ClampedArray(size * size * 4)  // RGBA grayscale for canvas
        const stride = bytesPerRow / 2  // stride in uint16 elements

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const srcIdx = y * stride + x * 4
                const dstIdx = (y * size + x) * 4

                // Decode float16 to float32
                let r = this._float16ToFloat32(float16Data[srcIdx])
                let g = this._float16ToFloat32(float16Data[srcIdx + 1])
                let b = this._float16ToFloat32(float16Data[srcIdx + 2])

                // Find max component
                const maxVal = Math.max(r, g, b, 1e-10)

                let multiplier = 1.0
                if (maxVal > 1.0) {
                    // HDR range: scale down so max = 1.0
                    multiplier = Math.min(maxVal, MULT_MAX)
                    const scale = 1.0 / multiplier
                    r *= scale
                    g *= scale
                    b *= scale
                }

                // Apply sRGB gamma and store RGB
                rgbData[dstIdx] = Math.round(Math.pow(Math.min(1, r), 1 / SRGB_GAMMA) * 255)
                rgbData[dstIdx + 1] = Math.round(Math.pow(Math.min(1, g), 1 / SRGB_GAMMA) * 255)
                rgbData[dstIdx + 2] = Math.round(Math.pow(Math.min(1, b), 1 / SRGB_GAMMA) * 255)
                rgbData[dstIdx + 3] = 255

                // Encode multiplier: 1.0 = black (0), 32768 = white (255), logarithmic
                // log2(1) = 0 → 0, log2(32768) = 15 → 255
                const logMult = Math.log2(Math.max(1.0, multiplier))  // 0 to 15
                const multNorm = logMult / LOG_MULT_MAX  // 0 to 1

                // Add blue noise dithering (-0.5 to +0.5) before quantization
                let dither = 0
                if (blueNoiseData) {
                    const noiseX = x % blueNoiseSize
                    const noiseY = y % blueNoiseSize
                    const noiseIdx = (noiseY * blueNoiseSize + noiseX) * 4
                    dither = (blueNoiseData[noiseIdx] / 255) - 0.5
                } else {
                    dither = Math.random() - 0.5
                }

                const multByte = Math.min(255, Math.max(0, Math.round(multNorm * 255 + dither)))

                multData[dstIdx] = multByte
                multData[dstIdx + 1] = multByte
                multData[dstIdx + 2] = multByte
                multData[dstIdx + 3] = 255
            }
        }

        readBuffer.unmap()
        readBuffer.destroy()

        // Create RGB canvas and save as JPG
        const rgbCanvas = document.createElement('canvas')
        rgbCanvas.width = size
        rgbCanvas.height = size
        const rgbCtx = rgbCanvas.getContext('2d')
        const rgbImageData = new ImageData(rgbData, size, size)
        rgbCtx.putImageData(rgbImageData, 0, 0)

        // Create multiplier canvas and save as JPG
        const multCanvas = document.createElement('canvas')
        multCanvas.width = size
        multCanvas.height = size
        const multCtx = multCanvas.getContext('2d')
        const multImageData = new ImageData(multData, size, size)
        multCtx.putImageData(multImageData, 0, 0)

        // Download RGB JPG
        const rgbLink = document.createElement('a')
        rgbLink.download = `${basename}.jpg`
        rgbLink.href = rgbCanvas.toDataURL('image/jpeg', quality)
        rgbLink.click()

        // Small delay between downloads
        await new Promise(r => setTimeout(r, 100))

        // Download multiplier JPG
        const multLink = document.createElement('a')
        multLink.download = `${basename}.mult.jpg`
        multLink.href = multCanvas.toDataURL('image/jpeg', quality)
        multLink.click()

        console.log(`ProbeCapture: Saved RGBM pair as ${basename}.jpg + ${basename}.mult.jpg (${size}x${size})`)
    }

    /**
     * Export probe as Radiance HDR file and trigger download
     * @param {string} filename - Download filename
     */
    async saveAsHDR(filename = 'probe.hdr') {
        const { device } = this.engine

        // Read back texture data
        const bytesPerPixel = 8 // rgba16float = 4 * 2 bytes
        const bytesPerRow = Math.ceil(this.octahedralSize * bytesPerPixel / 256) * 256
        const bufferSize = bytesPerRow * this.octahedralSize

        const readBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })

        const commandEncoder = device.createCommandEncoder()
        commandEncoder.copyTextureToBuffer(
            { texture: this.octahedralTexture.texture },
            { buffer: readBuffer, bytesPerRow: bytesPerRow },
            { width: this.octahedralSize, height: this.octahedralSize }
        )
        device.queue.submit([commandEncoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)
        const float16Data = new Uint16Array(readBuffer.getMappedRange())

        // Convert to RGBE format
        const rgbeData = this._float16ToRGBE(float16Data, bytesPerRow / 2)

        readBuffer.unmap()
        readBuffer.destroy()

        // Build Radiance HDR file
        const hdrData = this._buildHDRFile(rgbeData, this.octahedralSize, this.octahedralSize)

        // Download
        const blob = new Blob([hdrData], { type: 'application/octet-stream' })
        const link = document.createElement('a')
        link.download = filename
        link.href = URL.createObjectURL(blob)
        link.click()
        URL.revokeObjectURL(link.href)

        console.log(`ProbeCapture: Saved HDR as ${filename}`)
    }

    /**
     * Build Radiance HDR file from RGBE data
     * Uses simple RLE compression per scanline
     */
    _buildHDRFile(rgbeData, width, height) {
        // Header
        const header = `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y ${height} +X ${width}\n`
        const headerBytes = new TextEncoder().encode(header)

        // Encode scanlines with RLE
        const scanlines = []
        for (let y = 0; y < height; y++) {
            const scanline = this._encodeRLEScanline(rgbeData, y, width)
            scanlines.push(scanline)
        }

        // Calculate total size
        const dataSize = scanlines.reduce((sum, s) => sum + s.length, 0)
        const totalSize = headerBytes.length + dataSize

        // Build final buffer
        const result = new Uint8Array(totalSize)
        result.set(headerBytes, 0)

        let offset = headerBytes.length
        for (const scanline of scanlines) {
            result.set(scanline, offset)
            offset += scanline.length
        }

        return result
    }

    /**
     * Encode a single scanline with RLE compression
     */
    _encodeRLEScanline(rgbeData, y, width) {
        // New RLE format for width >= 8 and <= 32767
        if (width < 8 || width > 32767) {
            // Fallback to uncompressed
            const scanline = new Uint8Array(width * 4)
            for (let x = 0; x < width; x++) {
                const srcIdx = (y * width + x) * 4
                scanline[x * 4 + 0] = rgbeData[srcIdx + 0]
                scanline[x * 4 + 1] = rgbeData[srcIdx + 1]
                scanline[x * 4 + 2] = rgbeData[srcIdx + 2]
                scanline[x * 4 + 3] = rgbeData[srcIdx + 3]
            }
            return scanline
        }

        // New RLE format: 4 bytes header + RLE encoded channels
        const header = new Uint8Array([2, 2, (width >> 8) & 0xFF, width & 0xFF])

        // Separate channels
        const channels = [[], [], [], []]
        for (let x = 0; x < width; x++) {
            const srcIdx = (y * width + x) * 4
            channels[0].push(rgbeData[srcIdx + 0])
            channels[1].push(rgbeData[srcIdx + 1])
            channels[2].push(rgbeData[srcIdx + 2])
            channels[3].push(rgbeData[srcIdx + 3])
        }

        // RLE encode each channel
        const encodedChannels = channels.map(ch => this._rleEncodeChannel(ch))

        // Combine
        const totalLen = header.length + encodedChannels.reduce((s, c) => s + c.length, 0)
        const result = new Uint8Array(totalLen)
        result.set(header, 0)

        let offset = header.length
        for (const encoded of encodedChannels) {
            result.set(encoded, offset)
            offset += encoded.length
        }

        return result
    }

    /**
     * RLE encode a single channel
     */
    _rleEncodeChannel(data) {
        const result = []
        let i = 0

        while (i < data.length) {
            // Check for run
            let runLen = 1
            while (i + runLen < data.length && runLen < 127 && data[i + runLen] === data[i]) {
                runLen++
            }

            if (runLen > 2) {
                // Encode run
                result.push(128 + runLen)
                result.push(data[i])
                i += runLen
            } else {
                // Encode non-run (literal)
                let litLen = 1
                while (i + litLen < data.length && litLen < 128) {
                    // Check if next would start a run
                    if (i + litLen + 2 < data.length &&
                        data[i + litLen] === data[i + litLen + 1] &&
                        data[i + litLen] === data[i + litLen + 2]) {
                        break
                    }
                    litLen++
                }
                result.push(litLen)
                for (let j = 0; j < litLen; j++) {
                    result.push(data[i + j])
                }
                i += litLen
            }
        }

        return new Uint8Array(result)
    }

    /**
     * Export probe as RGBE PNG (for debugging/preview)
     * @param {string} filename - Download filename
     */
    async saveAsPNG(filename = 'probe.png') {
        const { device } = this.engine

        // Read back texture data
        const bytesPerPixel = 8 // rgba16float = 4 * 2 bytes
        const bytesPerRow = Math.ceil(this.octahedralSize * bytesPerPixel / 256) * 256
        const bufferSize = bytesPerRow * this.octahedralSize

        const readBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })

        const commandEncoder = device.createCommandEncoder()
        commandEncoder.copyTextureToBuffer(
            { texture: this.octahedralTexture.texture },
            { buffer: readBuffer, bytesPerRow: bytesPerRow },
            { width: this.octahedralSize, height: this.octahedralSize }
        )
        device.queue.submit([commandEncoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)
        const data = new Uint16Array(readBuffer.getMappedRange())

        // Convert float16 to RGBE format for HDR storage
        const rgbeData = this._float16ToRGBE(data, bytesPerRow / 2)

        readBuffer.unmap()
        readBuffer.destroy()

        // Create PNG with RGBE data
        const canvas = document.createElement('canvas')
        canvas.width = this.octahedralSize
        canvas.height = this.octahedralSize
        const ctx = canvas.getContext('2d')
        const imageData = ctx.createImageData(this.octahedralSize, this.octahedralSize)
        imageData.data.set(rgbeData)
        ctx.putImageData(imageData, 0, 0)

        // Download
        const link = document.createElement('a')
        link.download = filename
        link.href = canvas.toDataURL('image/png')
        link.click()

        console.log(`ProbeCapture: Saved as ${filename}`)
    }

    /**
     * Convert float16 RGBA to RGBE format
     */
    _float16ToRGBE(float16Data, stride) {
        const size = this.octahedralSize
        const rgbe = new Uint8ClampedArray(size * size * 4)

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const srcIdx = y * stride + x * 4
                const dstIdx = (y * size + x) * 4

                // Decode float16 to float32
                const r = this._float16ToFloat32(float16Data[srcIdx])
                const g = this._float16ToFloat32(float16Data[srcIdx + 1])
                const b = this._float16ToFloat32(float16Data[srcIdx + 2])

                // Find max component
                const maxVal = Math.max(r, g, b)

                if (maxVal < 1e-32) {
                    rgbe[dstIdx] = 0
                    rgbe[dstIdx + 1] = 0
                    rgbe[dstIdx + 2] = 0
                    rgbe[dstIdx + 3] = 0
                } else {
                    // Compute exponent
                    let exp = Math.ceil(Math.log2(maxVal))
                    const scale = Math.pow(2, -exp) * 255

                    rgbe[dstIdx] = Math.min(255, Math.max(0, r * scale))
                    rgbe[dstIdx + 1] = Math.min(255, Math.max(0, g * scale))
                    rgbe[dstIdx + 2] = Math.min(255, Math.max(0, b * scale))
                    rgbe[dstIdx + 3] = exp + 128
                }
            }
        }

        return rgbe
    }

    /**
     * Convert float16 (as uint16) to float32
     */
    _float16ToFloat32(h) {
        const sign = (h & 0x8000) >> 15
        const exp = (h & 0x7C00) >> 10
        const frac = h & 0x03FF

        if (exp === 0) {
            if (frac === 0) return sign ? -0 : 0
            // Denormalized
            return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024)
        } else if (exp === 31) {
            return frac ? NaN : (sign ? -Infinity : Infinity)
        }

        return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024)
    }

    /**
     * Capture and save as PNG
     * @param {vec3} position - Capture position
     * @param {string} filename - Download filename
     */
    async captureAndSave(position, filename = 'probe.png') {
        await this.capture(position)
        await this.saveAsPNG(filename)
    }

    /**
     * Get the octahedral probe texture (raw float16)
     */
    getProbeTexture() {
        return this.octahedralTexture
    }

    /**
     * Convert float16 octahedral texture to RGBE-encoded rgba8unorm texture with mips
     * This makes it compatible with the standard environment map pipeline
     */
    async _createRGBETexture() {
        const { device } = this.engine

        // Read back float16 data
        const bytesPerPixel = 8 // rgba16float = 4 * 2 bytes
        const bytesPerRow = Math.ceil(this.octahedralSize * bytesPerPixel / 256) * 256
        const bufferSize = bytesPerRow * this.octahedralSize

        const readBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })

        const commandEncoder = device.createCommandEncoder()
        commandEncoder.copyTextureToBuffer(
            { texture: this.octahedralTexture.texture },
            { buffer: readBuffer, bytesPerRow: bytesPerRow },
            { width: this.octahedralSize, height: this.octahedralSize }
        )
        device.queue.submit([commandEncoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)
        const float16Data = new Uint16Array(readBuffer.getMappedRange())

        // Convert to RGBE
        const rgbeData = this._float16ToRGBE(float16Data, bytesPerRow / 2)

        readBuffer.unmap()
        readBuffer.destroy()

        // Calculate mip levels
        const mipCount = numMipLevels(this.octahedralSize, this.octahedralSize)

        // Create RGBE texture with mips (rgba8unorm like loaded HDR files)
        const rgbeTexture = device.createTexture({
            label: 'probeOctahedralRGBE',
            size: [this.octahedralSize, this.octahedralSize],
            mipLevelCount: mipCount,
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        })

        device.queue.writeTexture(
            { texture: rgbeTexture },
            rgbeData,
            { bytesPerRow: this.octahedralSize * 4 },
            { width: this.octahedralSize, height: this.octahedralSize }
        )

        // Generate mip levels with RGBE-aware filtering
        generateMips(device, rgbeTexture, true) // true = RGBE mode

        this.octahedralRGBE = {
            texture: rgbeTexture,
            view: rgbeTexture.createView(),
            mipCount: mipCount
        }
    }

    /**
     * Get texture in format compatible with lighting pass (RGBE encoded with mips)
     * Creates a wrapper with .view and .sampler properties
     */
    async getAsEnvironmentTexture() {
        if (!this.octahedralTexture) return null

        // Create RGBE texture with mips if not already done
        if (!this.octahedralRGBE) {
            await this._createRGBETexture()
        }

        return {
            texture: this.octahedralRGBE.texture,
            view: this.octahedralRGBE.view,
            sampler: this.faceSampler,
            width: this.octahedralSize,
            height: this.octahedralSize,
            mipCount: this.octahedralRGBE.mipCount,
            isHDR: true  // Mark as HDR for proper handling
        }
    }

    /**
     * Destroy resources
     */
    destroy() {
        for (const tex of this.faceTextures) {
            if (tex?.texture) tex.texture.destroy()
        }
        for (const tex of this.faceDepthTextures) {
            if (tex?.texture) tex.texture.destroy()
        }
        if (this.octahedralTexture?.texture) {
            this.octahedralTexture.texture.destroy()
        }
        if (this.octahedralRGBE?.texture) {
            this.octahedralRGBE.texture.destroy()
        }
        this.faceTextures = []
        this.faceDepthTextures = []
        this.octahedralTexture = null
        this.octahedralRGBE = null
    }
}

export { ProbeCapture }
