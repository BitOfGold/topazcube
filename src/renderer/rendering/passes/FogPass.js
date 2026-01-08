import { BasePass } from "./BasePass.js"

/**
 * FogPass - Distance-based fog with height fade
 *
 * Features:
 * - Distance fog based on camera distance (not Z-depth)
 * - Two-gradient system: distance[0]->distance[1] and distance[1]->distance[2]
 * - Height fade: maximum fog at bottomY, zero fog at topY
 * - Emissive/bright colors show through fog more (HDR resistance)
 * - Applied before bloom so bright objects still bloom through fog
 *
 * Input: HDR lighting output, GBuffer (for depth texture)
 * Output: Fog-applied HDR texture
 */
class FogPass extends BasePass {
    constructor(engine = null) {
        super('Fog', engine)

        this.renderPipeline = null
        this.outputTexture = null
        this.inputTexture = null
        this.gbuffer = null
        this.uniformBuffer = null
        this.bindGroup = null
        this.sampler = null

        // Render dimensions
        this.width = 0
        this.height = 0
    }

    // Fog settings getters
    get fogEnabled() { return this.settings?.environment?.fog?.enabled ?? false }
    get fogColor() { return this.settings?.environment?.fog?.color ?? [0.8, 0.85, 0.9] }
    get fogDistances() { return this.settings?.environment?.fog?.distances ?? [0, 50, 200] }
    get fogAlpha() { return this.settings?.environment?.fog?.alpha ?? [0.0, 0.3, 0.8] }
    get fogHeightFade() { return this.settings?.environment?.fog?.heightFade ?? [-10, 100] }
    get fogBrightResist() { return this.settings?.environment?.fog?.brightResist ?? 0.8 }

    /**
     * Set the input texture (HDR lighting output)
     */
    setInputTexture(texture) {
        if (this.inputTexture !== texture) {
            this.inputTexture = texture
            this._needsRebuild = true
        }
    }

    /**
     * Set the GBuffer for depth/normal reconstruction
     */
    setGBuffer(gbuffer) {
        if (this.gbuffer !== gbuffer) {
            this.gbuffer = gbuffer
            this._needsRebuild = true
        }
    }

    async _init() {
        const { device } = this.engine

        // Create sampler
        this.sampler = device.createSampler({
            label: 'Fog Sampler',
            minFilter: 'linear',
            magFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        })

        // Create uniform buffer
        // inverseProj (64) + inverseView (64) + cameraPosition (12) + near (4) +
        // far (4) + fogEnabled (4) + fogColor (12) + brightResist (4) +
        // distances (12) + pad (4) + alphas (12) + pad (4) + heightFade (8) + screenSize (8) = 224 bytes
        // Round to 256 for alignment
        this.uniformBuffer = device.createBuffer({
            label: 'Fog Uniforms',
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
    }

    async _buildPipeline() {
        if (!this.inputTexture || !this.gbuffer) {
            return
        }

        const { device } = this.engine

        // Destroy old output texture before creating new one
        if (this.outputTexture) {
            this.outputTexture.destroy()
            this.outputTexture = null
            this.outputTextureView = null
        }

        // Create output texture (same format as input)
        // Include COPY_SRC for planar reflection pass which copies fog output
        this.outputTexture = device.createTexture({
            label: 'Fog Output',
            size: { width: this.width || 1, height: this.height || 1, depthOrArrayLayers: 1 },
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        })
        this.outputTextureView = this.outputTexture.createView({ label: 'Fog Output View' })

        // Create bind group layout
        const bindGroupLayout = device.createBindGroupLayout({
            label: 'Fog BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
            ],
        })

        const shaderModule = device.createShaderModule({
            label: 'Fog Shader',
            code: `
                // Uniforms packed for proper WGSL alignment (vec3f needs 16-byte alignment)
                struct Uniforms {
                    inverseProjection: mat4x4f,  // floats 0-15
                    inverseView: mat4x4f,        // floats 16-31
                    cameraPosition: vec3f,       // floats 32-34
                    near: f32,                   // float 35
                    fogColor: vec3f,             // floats 36-38
                    far: f32,                    // float 39
                    distances: vec3f,            // floats 40-42
                    fogEnabled: f32,             // float 43
                    alphas: vec3f,               // floats 44-46
                    brightResist: f32,           // float 47
                    heightFade: vec2f,           // floats 48-49
                    screenSize: vec2f,           // floats 50-51
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                @group(0) @binding(1) var inputTexture: texture_2d<f32>;
                @group(0) @binding(2) var inputSampler: sampler;
                @group(0) @binding(3) var depthTexture: texture_depth_2d;

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) uv: vec2f,
                }

                @vertex
                fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                    var output: VertexOutput;
                    // Full-screen triangle
                    let x = f32((vertexIndex << 1u) & 2u);
                    let y = f32(vertexIndex & 2u);
                    output.position = vec4f(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
                    output.uv = vec2f(x, 1.0 - y);
                    return output;
                }

                @fragment
                fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                    let color = textureSample(inputTexture, inputSampler, input.uv);

                    // If fog disabled, pass through
                    if (uniforms.fogEnabled < 0.5) {
                        return color;
                    }

                    // Sample depth from GBuffer depth texture (same as lighting shader)
                    let pixelCoords = vec2i(input.uv * uniforms.screenSize);
                    let depth = textureLoad(depthTexture, pixelCoords, 0);

                    // Check if this is sky (depth = 1 means far plane / cleared depth buffer)
                    let isSky = depth >= 0.9999;

                    // Get view ray direction (needed for both sky and geometry)
                    var iuv = input.uv;
                    iuv.y = 1.0 - iuv.y;
                    let ndc = vec4f(iuv * 2.0 - 1.0, 0.0, 1.0);
                    let viewRay = uniforms.inverseProjection * ndc;
                    let rayDir = normalize(viewRay.xyz / viewRay.w);

                    // For sky, treat as far distance; otherwise reconstruct linear depth
                    var cameraDistance: f32;
                    var worldPosY: f32;

                    if (isSky) {
                        // Sky is at far distance
                        cameraDistance = uniforms.far;

                        // Map view ray elevation to effective Y for height fade
                        // The fog clear zone in sky depends on topY:
                        // - Low topY (near camera): small clear zone, fog only at horizon
                        // - Medium topY (25m): clear zone extends to ~60° above horizon
                        // - High topY (100m+): almost no clear zone, fog fills sky
                        let worldRayDir = normalize((uniforms.inverseView * vec4f(rayDir, 0.0)).xyz);
                        let camY = uniforms.cameraPosition.y;
                        let topY = uniforms.heightFade.y;
                        let bottomY = uniforms.heightFade.x;

                        if (worldRayDir.y > 0.0) {
                            // Looking up: map elevation to effective Y
                            // Use a reference height to control how much sky can be clear
                            // At referenceHeight (50m), zenith reaches topY (fully clear)
                            // Above that, zenith stays fogged
                            let referenceHeight = 50.0;
                            let fogHeight = topY - camY;
                            let clearRange = min(fogHeight, referenceHeight);
                            worldPosY = camY + worldRayDir.y * clearRange;
                        } else {
                            // Looking down or at horizon: below camera, full fog
                            worldPosY = camY + worldRayDir.y * (camY - bottomY);
                        }
                    } else {
                        // Reconstruct linear depth: depth = (z - near) / (far - near), so z = near + depth * (far - near)
                        let linearDepth = uniforms.near + depth * (uniforms.far - uniforms.near);

                        // Use linear depth directly as camera distance
                        cameraDistance = linearDepth;

                        // Reconstruct world position for height fade
                        let viewPos = rayDir * (linearDepth / -rayDir.z);
                        let worldPos4 = uniforms.inverseView * vec4f(viewPos, 1.0);
                        worldPosY = worldPos4.y;
                    }

                    // Distance fog - two gradients
                    var distanceFog: f32;
                    let d0 = uniforms.distances.x;
                    let d1 = uniforms.distances.y;
                    let d2 = uniforms.distances.z;
                    let a0 = uniforms.alphas.x;
                    let a1 = uniforms.alphas.y;
                    let a2 = uniforms.alphas.z;

                    if (cameraDistance <= d0) {
                        distanceFog = a0;
                    } else if (cameraDistance <= d1) {
                        let t = (cameraDistance - d0) / max(d1 - d0, 0.001);
                        distanceFog = mix(a0, a1, t);
                    } else if (cameraDistance <= d2) {
                        let t = (cameraDistance - d1) / max(d2 - d1, 0.001);
                        distanceFog = mix(a1, a2, t);
                    } else {
                        distanceFog = a2;
                    }

                    // Height fade - full fog at bottomY, zero fog at topY
                    let bottomY = uniforms.heightFade.x;
                    let topY = uniforms.heightFade.y;
                    var heightFactor = clamp((worldPosY - bottomY) / max(topY - bottomY, 0.001), 0.0, 1.0);
                    // Below bottomY = maximum fog
                    if (worldPosY < bottomY) {
                        heightFactor = 0.0;
                    }
                    var fogAlpha = distanceFog * (1.0 - heightFactor);

                    // Emissive/bright resistance - HDR values show through fog
                    // Calculate luminance
                    let luminance = dot(color.rgb, vec3f(0.299, 0.587, 0.114));
                    // For HDR values > 1.0, reduce fog effect
                    let brightnessResist = clamp((luminance - 1.0) / 2.0, 0.0, 1.0);
                    fogAlpha *= (1.0 - brightnessResist * uniforms.brightResist);

                    // Apply fog
                    let foggedColor = mix(color.rgb, uniforms.fogColor, fogAlpha);

                    return vec4f(foggedColor, color.a);
                }
            `,
        })

        this.renderPipeline = await device.createRenderPipelineAsync({
            label: 'Fog Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: { module: shaderModule, entryPoint: 'vertexMain' },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'rgba16float' }],
            },
            primitive: { topology: 'triangle-list' },
        })

        this.bindGroup = device.createBindGroup({
            label: 'Fog Bind Group',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.inputTexture.view },
                { binding: 2, resource: this.sampler },
                { binding: 3, resource: this.gbuffer.depth.view },
            ],
        })

        this._needsRebuild = false
    }

    async _execute(context) {
        const { device } = this.engine
        const { camera } = context

        // Skip if fog disabled
        if (!this.fogEnabled) {
            // Just copy input to output or return input directly
            this.outputTexture = null
            return
        }

        // Rebuild pipeline if needed
        if (this._needsRebuild) {
            await this._buildPipeline()
        }

        if (!this.renderPipeline || !this.inputTexture || !this.gbuffer || this._needsRebuild) {
            return
        }

        // Update uniforms - 256 bytes = 64 floats (matches shader struct layout)
        const uniformData = new Float32Array(64)

        // inverseProjection (floats 0-15)
        if (camera.iProj) {
            uniformData.set(camera.iProj, 0)
        }

        // inverseView (floats 16-31)
        if (camera.iView) {
            uniformData.set(camera.iView, 16)
        }

        // cameraPosition (floats 32-34) + near (float 35)
        uniformData[32] = camera.position[0]
        uniformData[33] = camera.position[1]
        uniformData[34] = camera.position[2]
        uniformData[35] = camera.near || 0.1

        // fogColor (floats 36-38) + far (float 39)
        const fogColor = this.fogColor
        uniformData[36] = fogColor[0]
        uniformData[37] = fogColor[1]
        uniformData[38] = fogColor[2]
        uniformData[39] = camera.far || 1000

        // distances (floats 40-42) + fogEnabled (float 43)
        const distances = this.fogDistances
        uniformData[40] = distances[0]
        uniformData[41] = distances[1]
        uniformData[42] = distances[2]
        uniformData[43] = this.fogEnabled ? 1.0 : 0.0

        // alphas (floats 44-46) + brightResist (float 47)
        const alphas = this.fogAlpha
        uniformData[44] = alphas[0]
        uniformData[45] = alphas[1]
        uniformData[46] = alphas[2]
        uniformData[47] = this.fogBrightResist

        // heightFade (floats 48-49) + screenSize (floats 50-51)
        const heightFade = this.fogHeightFade
        uniformData[48] = heightFade[0]  // bottomY
        uniformData[49] = heightFade[1]  // topY
        uniformData[50] = this.width
        uniformData[51] = this.height

        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData)

        // Render fog pass
        const commandEncoder = device.createCommandEncoder({ label: 'Fog Pass' })

        const renderPass = commandEncoder.beginRenderPass({
            label: 'Fog Render Pass',
            colorAttachments: [{
                view: this.outputTextureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        })

        renderPass.setPipeline(this.renderPipeline)
        renderPass.setBindGroup(0, this.bindGroup)
        renderPass.draw(3)
        renderPass.end()

        device.queue.submit([commandEncoder.finish()])
    }

    async _resize(width, height) {
        this.width = width
        this.height = height
        this._needsRebuild = true
    }

    _destroy() {
        if (this.outputTexture) {
            this.outputTexture.destroy()
            this.outputTexture = null
        }
        this.renderPipeline = null
    }

    /**
     * Get the output texture (fog-applied HDR)
     */
    getOutputTexture() {
        // If fog is disabled, return input texture
        if (!this.fogEnabled || !this.outputTexture) {
            return this.inputTexture
        }
        return {
            texture: this.outputTexture,
            view: this.outputTextureView,
            sampler: this.sampler,
        }
    }
}

export { FogPass }
