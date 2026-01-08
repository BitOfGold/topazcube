import { BasePass } from "./BasePass.js"
import { Texture } from "../../Texture.js"
import { Frustum } from "../../utils/Frustum.js"

import lightingCommonWGSL from "../shaders/lighting_common.wgsl"

/**
 * TransparentPass - Forward rendering pass for transparent objects
 *
 * Renders transparent meshes with full PBR lighting on top of the HDR buffer.
 * Uses back-to-front sorting for correct alpha blending.
 * Reads depth from GBuffer for occlusion testing but does NOT write depth,
 * so fog pass uses scene depth (what's behind transparent objects).
 */
class TransparentPass extends BasePass {
    constructor(engine = null) {
        super('Transparent', engine)

        this.pipeline = null
        this.bindGroupLayout = null
        this.uniformBuffer = null
        this.pipelineCache = new Map()

        // References to shared resources
        this.gbuffer = null
        this.lightingUniforms = null
        this.shadowPass = null
        this.environmentMap = null
        this.noiseTexture = null
        this.noiseSize = 64
        this.noiseAnimated = true

        // Light buffers
        this.tileLightBuffer = null
        this.lightBuffer = null

        // HiZ pass reference for occlusion culling
        this.hizPass = null

        // Frustum for transparent mesh culling
        this.frustum = new Frustum()

        // Distance fade for preventing object popping at culling distance
        this.distanceFadeStart = 0  // Distance where fade begins
        this.distanceFadeEnd = 0    // Distance where fade completes (0 = disabled)

        // Culling stats for transparent meshes
        this.cullingStats = {
            total: 0,
            rendered: 0,
            culledByFrustum: 0,
            culledByDistance: 0,
            culledByOcclusion: 0
        }
    }

    /**
     * Set the HiZ pass for occlusion culling
     * @param {HiZPass} hizPass - The HiZ pass instance
     */
    setHiZPass(hizPass) {
        this.hizPass = hizPass
    }

    /**
     * Test if a transparent mesh should be culled
     * @param {Mesh} mesh - The mesh to test
     * @param {Camera} camera - Current camera
     * @param {boolean} canCull - Whether frustum/occlusion culling is available
     * @returns {string|null} - Reason for culling or null if visible
     */
    _shouldCullMesh(mesh, camera, canCull) {
        // Transparent mesh culling is disabled - same issues as legacy mesh culling
        // where bounding spheres may be in local space or represent root nodes
        return null
    }

    async _init() {
        const { device } = this.engine

        // Create uniform buffer (same layout as lighting pass + material params)
        this.uniformBuffer = device.createBuffer({
            size: 512, // Plenty of space for uniforms
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Transparent Uniforms'
        })

        // Create placeholder resources for missing bindings
        await this._createPlaceholders()
    }

    /**
     * Set the GBuffer for depth testing
     */
    setGBuffer(gbuffer) {
        this.gbuffer = gbuffer
    }

    /**
     * Set the HDR output texture to render onto
     */
    setOutputTexture(texture) {
        this.outputTexture = texture
    }

    /**
     * Set shadow pass reference
     */
    setShadowPass(shadowPass) {
        this.shadowPass = shadowPass
    }

    /**
     * Set environment map
     */
    setEnvironmentMap(envMap, encoding = 'equirect') {
        this.environmentMap = envMap
        this.environmentEncoding = encoding
    }

    /**
     * Set noise texture for effects
     */
    setNoise(noise, size = 64, animated = true) {
        this.noiseTexture = noise
        this.noiseSize = size
        this.noiseAnimated = animated
    }

    /**
     * Set light buffers for tiled lighting
     */
    setLightBuffers(tileLightBuffer, lightBuffer) {
        this.tileLightBuffer = tileLightBuffer
        this.lightBuffer = lightBuffer
    }

    /**
     * Create or get pipeline for a mesh
     */
    async _getOrCreatePipeline(mesh) {
        const { device } = this.engine
        const key = `transparent_${mesh.material.uid}`

        if (this.pipelineCache.has(key)) {
            return this.pipelineCache.get(key)
        }

        // Build the shader
        const shaderCode = this._buildShaderCode()

        const shaderModule = device.createShaderModule({
            label: 'Transparent Shader',
            code: shaderCode
        })

        // Bind group layout
        const bindGroupLayout = device.createBindGroupLayout({
            label: 'Transparent BindGroup Layout',
            entries: [
                // Uniforms
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                // Albedo texture
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                // Normal texture
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                // ARM texture
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                // Environment map
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                // Shadow map array (cascades)
                { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
                { binding: 10, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
                // Cascade matrices
                { binding: 11, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                // Noise texture
                { binding: 12, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
            ]
        })

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        })

        // Vertex buffer layout (same as geometry pass)
        const vertexBufferLayout = {
            arrayStride: 80,
            attributes: [
                { format: "float32x3", offset: 0, shaderLocation: 0 },  // position
                { format: "float32x2", offset: 12, shaderLocation: 1 }, // uv
                { format: "float32x3", offset: 20, shaderLocation: 2 }, // normal
                { format: "float32x4", offset: 32, shaderLocation: 3 }, // color
                { format: "float32x4", offset: 48, shaderLocation: 4 }, // weights
                { format: "uint32x4", offset: 64, shaderLocation: 5 },  // joints
            ],
            stepMode: 'vertex'
        }

        const instanceBufferLayout = {
            arrayStride: 112,  // 28 floats: matrix(16) + posRadius(4) + uvTransform(4) + color(4)
            stepMode: 'instance',
            attributes: [
                { format: "float32x4", offset: 0, shaderLocation: 6 },
                { format: "float32x4", offset: 16, shaderLocation: 7 },
                { format: "float32x4", offset: 32, shaderLocation: 8 },
                { format: "float32x4", offset: 48, shaderLocation: 9 },
                { format: "float32x4", offset: 64, shaderLocation: 10 },
                { format: "float32x4", offset: 80, shaderLocation: 11 },  // uvTransform
                { format: "float32x4", offset: 96, shaderLocation: 12 },  // color
            ]
        }

        const pipeline = await device.createRenderPipelineAsync({
            label: 'Transparent Pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
                buffers: [vertexBufferLayout, instanceBufferLayout]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
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
                depthWriteEnabled: false, // Don't write depth - fog needs scene depth behind transparent objects
                depthCompare: 'less',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none', // No culling for transparent objects
            }
        })

        this.pipelineCache.set(key, { pipeline, bindGroupLayout })
        return { pipeline, bindGroupLayout }
    }

    /**
     * Build the forward transparent shader
     */
    _buildShaderCode() {
        return `
${lightingCommonWGSL}

struct TransparentUniforms {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
    cameraPosition: vec3f,
    opacity: f32,
    lightDir: vec3f,
    lightIntensity: f32,
    lightColor: vec3f,
    ambientIntensity: f32,
    ambientColor: vec3f,
    envMipCount: f32,
    envDiffuse: f32,
    envSpecular: f32,
    exposure: f32,
    envEncoding: f32, // 0=equirect, 1=octahedral
    shadowBias: f32,
    shadowNormalBias: f32,
    shadowStrength: f32,
    cascadeSize0: f32,
    cascadeSize1: f32,
    cascadeSize2: f32,
    noiseSize: f32,
    noiseOffsetX: f32,
    noiseOffsetY: f32,
    distanceFadeStart: f32,  // Distance where fade begins
    distanceFadeEnd: f32,    // Distance where fade completes (0 = disabled)
    fogEnabled: f32,         // Whether fog is enabled
    fogPad1: f32,            // Padding for vec3 alignment
    fogPad2: f32,
    fogPad3: f32,
    fogColor: vec3f,         // Fog color
    fogBrightResist: f32,    // HDR brightness resistance
    fogDistances: vec3f,     // Distance thresholds [near, mid, far]
    fogPad4: f32,
    fogAlphas: vec3f,        // Alpha values at distances
    fogPad5: f32,
    fogHeightFade: vec2f,    // [bottomY, topY]
    cameraNear: f32,
    cameraFar: f32,
}

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
    @location(3) color: vec4f,
    @location(4) weights: vec4f,
    @location(5) joints: vec4u,
    @location(6) model0: vec4f,
    @location(7) model1: vec4f,
    @location(8) model2: vec4f,
    @location(9) model3: vec4f,
    @location(10) posRadius: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
}

@group(0) @binding(0) var<uniform> uniforms: TransparentUniforms;
@group(0) @binding(1) var albedoTexture: texture_2d<f32>;
@group(0) @binding(2) var albedoSampler: sampler;
@group(0) @binding(3) var normalTexture: texture_2d<f32>;
@group(0) @binding(4) var normalSampler: sampler;
@group(0) @binding(5) var armTexture: texture_2d<f32>;
@group(0) @binding(6) var armSampler: sampler;
@group(0) @binding(7) var envTexture: texture_2d<f32>;
@group(0) @binding(8) var envSampler: sampler;
@group(0) @binding(9) var shadowMapArray: texture_depth_2d_array;
@group(0) @binding(10) var shadowSampler: sampler_comparison;
@group(0) @binding(11) var<storage, read> cascadeMatrices: CascadeMatrices;
@group(0) @binding(12) var noiseTexture: texture_2d<f32>;

fn getEnvUV(dir: vec3f) -> vec2f {
    if (uniforms.envEncoding > 0.5) {
        return octEncode(dir);
    }
    return SphToUV(dir);
}

fn sampleNoise(screenPos: vec2f) -> f32 {
    let noiseSize = i32(uniforms.noiseSize);
    let noiseOffsetX = i32(uniforms.noiseOffsetX * f32(noiseSize));
    let noiseOffsetY = i32(uniforms.noiseOffsetY * f32(noiseSize));
    let texCoord = vec2i(
        (i32(screenPos.x) + noiseOffsetX) % noiseSize,
        (i32(screenPos.y) + noiseOffsetY) % noiseSize
    );
    return textureLoad(noiseTexture, texCoord, 0).r;
}

fn getNoiseJitter(screenPos: vec2f) -> vec2f {
    let noiseSize = i32(uniforms.noiseSize);
    let noiseOffsetX = i32(uniforms.noiseOffsetX * f32(noiseSize));
    let noiseOffsetY = i32(uniforms.noiseOffsetY * f32(noiseSize));
    let texCoord1 = vec2i(
        (i32(screenPos.x) + noiseOffsetX) % noiseSize,
        (i32(screenPos.y) + noiseOffsetY) % noiseSize
    );
    let texCoord2 = vec2i(
        (i32(screenPos.x) + 37 + noiseOffsetX) % noiseSize,
        (i32(screenPos.y) + 17 + noiseOffsetY) % noiseSize
    );
    let n1 = textureLoad(noiseTexture, texCoord1, 0).r;
    let n2 = textureLoad(noiseTexture, texCoord2, 0).r;
    return vec2f(n1, n2) * 2.0 - 1.0;
}

fn getIBLSample(reflection: vec3f, lod: f32) -> vec3f {
    let envRGBE = textureSampleLevel(envTexture, envSampler, getEnvUV(reflection), lod);
    let envColor = envRGBE.rgb * pow(2.0, envRGBE.a * 255.0 - 128.0);
    return envColor;
}

// Sample cascade shadow
fn sampleCascadeShadow(worldPos: vec3f, normal: vec3f, cascadeIndex: i32, screenPos: vec2f) -> f32 {
    let bias = uniforms.shadowBias;
    let normalBias = uniforms.shadowNormalBias;
    let biasedPos = worldPos + normal * normalBias;
    let lightMatrix = cascadeMatrices.matrices[cascadeIndex];
    let lightSpacePos = lightMatrix * vec4f(biasedPos, 1.0);
    let projCoords = lightSpacePos.xyz / lightSpacePos.w;
    let shadowUV = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);
    let currentDepth = projCoords.z - bias;

    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0) {
        return 1.0;
    }

    // Simple shadow sampling with jitter
    let jitter = getNoiseJitter(screenPos);
    let texelSize = 1.0 / 2048.0;
    var shadow = 0.0;
    for (var i = 0; i < 4; i++) {
        let offset = vogelDiskSample(i, 4, jitter.x * PI) * texelSize * 2.0;
        shadow += textureSampleCompareLevel(shadowMapArray, shadowSampler, shadowUV + offset, cascadeIndex, currentDepth);
    }
    return shadow / 4.0;
}

fn calculateShadow(worldPos: vec3f, normal: vec3f, screenPos: vec2f) -> f32 {
    let camXZ = vec2f(uniforms.cameraPosition.x, uniforms.cameraPosition.z);
    let posXZ = vec2f(worldPos.x, worldPos.z);
    let offsetXZ = posXZ - camXZ;

    let dist0 = squircleDistanceXZ(offsetXZ, uniforms.cascadeSize0);
    let dist1 = squircleDistanceXZ(offsetXZ, uniforms.cascadeSize1);
    let dist2 = squircleDistanceXZ(offsetXZ, uniforms.cascadeSize2);

    var shadow = 1.0;
    if (dist0 < 1.0) {
        shadow = sampleCascadeShadow(worldPos, normal, 0, screenPos);
    } else if (dist1 < 1.0) {
        shadow = sampleCascadeShadow(worldPos, normal, 1, screenPos);
    } else if (dist2 < 1.0) {
        shadow = sampleCascadeShadow(worldPos, normal, 2, screenPos);
    }

    return mix(1.0 - uniforms.shadowStrength, 1.0, shadow);
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let modelMatrix = mat4x4f(
        input.model0,
        input.model1,
        input.model2,
        input.model3
    );

    let worldPos = (modelMatrix * vec4f(input.position, 1.0)).xyz;
    let viewPos = uniforms.viewMatrix * vec4f(worldPos, 1.0);
    output.position = uniforms.projectionMatrix * viewPos;
    output.worldPos = worldPos;
    output.uv = input.uv;

    let normalMatrix = mat3x3f(
        modelMatrix[0].xyz,
        modelMatrix[1].xyz,
        modelMatrix[2].xyz
    );
    output.normal = normalize(normalMatrix * input.normal);

    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    // Sample textures
    let albedoSample = textureSample(albedoTexture, albedoSampler, input.uv);
    let armSample = textureSample(armTexture, armSampler, input.uv);

    // Calculate alpha (albedo alpha * material opacity)
    var alpha = albedoSample.a * uniforms.opacity;

    // Distance fade: modulate alpha to prevent popping at culling distance
    if (uniforms.distanceFadeEnd > 0.0) {
        let distToCamera = length(input.worldPos - uniforms.cameraPosition);
        if (distToCamera >= uniforms.distanceFadeEnd) {
            discard;  // Beyond fade end - fully invisible
        }
        if (distToCamera > uniforms.distanceFadeStart) {
            // Calculate fade factor: 1.0 at fadeStart, 0.0 at fadeEnd
            let fadeRange = uniforms.distanceFadeEnd - uniforms.distanceFadeStart;
            let fadeFactor = 1.0 - (distToCamera - uniforms.distanceFadeStart) / fadeRange;
            alpha *= fadeFactor;
        }
    }

    // Discard very transparent fragments
    if (alpha < 0.01) {
        discard;
    }

    let baseColor = albedoSample.rgb;
    let ao = armSample.r;
    let roughness = max(armSample.g, 0.04);
    let metallic = armSample.b;
    let alphaRoughness = roughness * roughness;

    // Normal mapping
    let normalSample = textureSample(normalTexture, normalSampler, input.uv).rgb;
    let tangentNormal = normalize(normalSample * 2.0 - 1.0);
    let N = normalize(input.normal);
    let refVec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
    let T = normalize(cross(N, refVec));
    let B = cross(N, T);
    let TBN = mat3x3f(T, B, N);
    let n = normalize(TBN * tangentNormal);

    // View direction
    let v = normalize(uniforms.cameraPosition - input.worldPos);
    let NdotV = clampedDot(n, v);

    // PBR setup
    var f0 = vec3f(0.04);
    f0 = mix(f0, baseColor, metallic);
    let f90 = vec3f(1.0);
    let c_diff = mix(baseColor, vec3f(0.0), metallic);
    let specularWeight = 1.0;

    var diffuse = vec3f(0.0);
    var specular = vec3f(0.0);

    // Ambient lighting
    diffuse += ao * uniforms.ambientColor * uniforms.ambientIntensity * BRDF_lambertian(f0, f90, c_diff, specularWeight, 1.0);

    // Environment diffuse
    let envDiffuseSample = getIBLSample(n, uniforms.envMipCount - 2.0);
    diffuse += ao * uniforms.envDiffuse * envDiffuseSample * BRDF_lambertian(f0, f90, c_diff, specularWeight, 1.0);

    // Environment specular
    let reflection = normalize(reflect(-v, n));
    let lod = roughness * (uniforms.envMipCount - 1.0);
    let Fr = max(vec3f(1.0 - roughness), f0) - f0;
    let k_S = f0 + Fr * pow(1.0 - NdotV, 5.0);
    let envSpecSample = getIBLSample(reflection, lod);
    specular += ao * uniforms.envSpecular * envSpecSample * k_S;

    // Directional light
    let l = normalize(uniforms.lightDir);
    let h = normalize(l + v);
    let NdotL = clampedDot(n, l);
    let NdotH = clampedDot(n, h);
    let VdotH = clampedDot(v, h);

    let shadow = calculateShadow(input.worldPos, n, input.position.xy);
    let lightContrib = uniforms.lightIntensity * uniforms.lightColor * NdotL * shadow;

    diffuse += lightContrib * BRDF_lambertian(f0, f90, c_diff, specularWeight, VdotH);
    specular += lightContrib * BRDF_specularGGX(f0, f90, alphaRoughness, specularWeight, VdotH, NdotL, NdotV, NdotH);

    // Final color
    var color = (diffuse + specular) * uniforms.exposure;

    // Fog is applied as post-process after all transparent/particle rendering

    // For glass-like materials, reduce color intensity based on transparency
    // More transparent = more of background shows through
    color *= alpha;

    return vec4f(color, alpha);
}
`
    }

    /**
     * Execute the transparent pass
     */
    async _execute(context) {
        const { device, canvas, stats } = this.engine
        const { camera, meshes, mainLight } = context

        // Initialize transparent stats
        stats.transparentDrawCalls = 0
        stats.transparentTriangles = 0

        if (!this.outputTexture || !this.gbuffer) {
            return
        }

        // Update frustum for transparent mesh culling (only if camera has required properties)
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
        this.cullingStats.total = 0
        this.cullingStats.rendered = 0
        this.cullingStats.culledByFrustum = 0
        this.cullingStats.culledByDistance = 0
        this.cullingStats.culledByOcclusion = 0

        // Collect transparent meshes with culling
        const transparentMeshes = []
        for (const name in meshes) {
            const mesh = meshes[name]
            if (mesh.material?.transparent && mesh.geometry?.instanceCount > 0) {
                this.cullingStats.total++

                // Apply culling (frustum, distance, occlusion)
                const cullReason = this._shouldCullMesh(mesh, camera, canCull)
                if (cullReason) {
                    if (cullReason === 'frustum') this.cullingStats.culledByFrustum++
                    else if (cullReason === 'distance') this.cullingStats.culledByDistance++
                    else if (cullReason === 'occlusion') this.cullingStats.culledByOcclusion++
                    continue
                }

                this.cullingStats.rendered++
                transparentMeshes.push({ name, mesh })
            }
        }

        if (transparentMeshes.length === 0) {
            return
        }

        // Sort back-to-front by distance to camera
        transparentMeshes.sort((a, b) => {
            // Get center position from first instance (simplified)
            const aPos = a.mesh.geometry.instanceData ?
                [a.mesh.geometry.instanceData[12], a.mesh.geometry.instanceData[13], a.mesh.geometry.instanceData[14]] :
                [0, 0, 0]
            const bPos = b.mesh.geometry.instanceData ?
                [b.mesh.geometry.instanceData[12], b.mesh.geometry.instanceData[13], b.mesh.geometry.instanceData[14]] :
                [0, 0, 0]

            const aDist = (aPos[0] - camera.position[0]) ** 2 +
                         (aPos[1] - camera.position[1]) ** 2 +
                         (aPos[2] - camera.position[2]) ** 2
            const bDist = (bPos[0] - camera.position[0]) ** 2 +
                         (bPos[1] - camera.position[1]) ** 2 +
                         (bPos[2] - camera.position[2]) ** 2

            return bDist - aDist // Back to front
        })

        // Render each transparent mesh
        const commandEncoder = device.createCommandEncoder({ label: 'Transparent Pass' })

        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.outputTexture.view,
                loadOp: 'load', // Preserve existing content
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.gbuffer.depth.view,
                depthLoadOp: 'load',
                depthStoreOp: 'store',
            }
        })

        for (const { mesh } of transparentMeshes) {
            const { pipeline, bindGroupLayout } = await this._getOrCreatePipeline(mesh)

            // Update uniforms
            const uniformData = new Float32Array(96)
            uniformData.set(camera.view, 0)
            uniformData.set(camera.proj, 16)
            uniformData.set(camera.position, 32)
            uniformData[35] = mesh.material.opacity ?? 1.0

            // Light direction
            const lightDir = mainLight?.direction || [-1, 1, -0.5]
            uniformData.set(lightDir, 36)
            uniformData[39] = mainLight?.color?.[3] ?? 1.0 // intensity

            // Light color
            const lightColor = mainLight?.color || [1, 1, 1]
            uniformData.set(lightColor, 40)
            uniformData[43] = this.settings?.environment?.ambientIntensity ?? 0.3

            // Ambient color
            const ambientColor = this.settings?.environment?.ambientColor || [0.5, 0.5, 0.6]
            uniformData.set(ambientColor, 44)
            uniformData[47] = this.settings?.environment?.envMipCount ?? 8 // envMipCount

            // Environment params
            uniformData[48] = this.settings?.environment?.diffuseLevel ?? 0.5
            uniformData[49] = this.settings?.environment?.specularLevel ?? 1.0
            uniformData[50] = this.settings?.environment?.exposure ?? 1.0
            uniformData[51] = this.environmentEncoding === 'octahedral' ? 1.0 : 0.0

            // Shadow params
            uniformData[52] = this.settings?.shadow?.bias ?? 0.001
            uniformData[53] = this.settings?.shadow?.normalBias ?? 0.02
            uniformData[54] = this.settings?.shadow?.strength ?? 0.7

            // Cascade sizes
            const cascadeSizes = this.shadowPass?.getCascadeSizes() || [20, 60, 300]
            uniformData[55] = cascadeSizes[0]
            uniformData[56] = cascadeSizes[1]
            uniformData[57] = cascadeSizes[2]

            // Noise params
            uniformData[58] = this.noiseSize
            uniformData[59] = this.noiseAnimated ? Math.random() : 0
            uniformData[60] = this.noiseAnimated ? Math.random() : 0

            // Distance fade params
            uniformData[61] = this.distanceFadeStart
            uniformData[62] = this.distanceFadeEnd

            // Fog params
            const fogSettings = this.settings?.environment?.fog
            uniformData[63] = fogSettings?.enabled ? 1.0 : 0.0
            // Padding at 64-66 for vec3 alignment (indices 64, 65, 66 are pad)
            // fogColor at indices 68-70 (aligned to 16 bytes at byte 272)
            const fogColor = fogSettings?.color ?? [0.8, 0.85, 0.9]
            uniformData[68] = fogColor[0]
            uniformData[69] = fogColor[1]
            uniformData[70] = fogColor[2]
            uniformData[71] = fogSettings?.brightResist ?? 0.8
            // fogDistances at indices 72-74
            const fogDistances = fogSettings?.distances ?? [0, 50, 200]
            uniformData[72] = fogDistances[0]
            uniformData[73] = fogDistances[1]
            uniformData[74] = fogDistances[2]
            // fogAlphas at indices 76-78
            const fogAlphas = fogSettings?.alpha ?? [0, 0.3, 0.8]
            uniformData[76] = fogAlphas[0]
            uniformData[77] = fogAlphas[1]
            uniformData[78] = fogAlphas[2]
            // fogHeightFade at indices 80-81
            const fogHeightFade = fogSettings?.heightFade ?? [-10, 100]
            uniformData[80] = fogHeightFade[0]
            uniformData[81] = fogHeightFade[1]
            // cameraNear/Far at indices 82-83
            uniformData[82] = camera.near || 0.1
            uniformData[83] = camera.far || 1000

            device.queue.writeBuffer(this.uniformBuffer, 0, uniformData)

            // Create bind group
            const bindGroup = device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: mesh.material.textures[0]?.view || this._placeholderTexture.view },
                    { binding: 2, resource: mesh.material.textures[0]?.sampler || this._placeholderSampler },
                    { binding: 3, resource: mesh.material.textures[1]?.view || this._placeholderNormal.view },
                    { binding: 4, resource: mesh.material.textures[1]?.sampler || this._placeholderSampler },
                    { binding: 5, resource: mesh.material.textures[3]?.view || this._placeholderTexture.view },
                    { binding: 6, resource: mesh.material.textures[3]?.sampler || this._placeholderSampler },
                    { binding: 7, resource: this.environmentMap?.view || this._placeholderTexture.view },
                    { binding: 8, resource: this.environmentMap?.sampler || this._placeholderSampler },
                    { binding: 9, resource: this.shadowPass?.getShadowMapView() || this._placeholderDepth.view },
                    { binding: 10, resource: this.shadowPass?.getShadowSampler() || this._placeholderComparisonSampler },
                    { binding: 11, resource: { buffer: this.shadowPass?.getCascadeMatricesBuffer() || this._placeholderBuffer } },
                    { binding: 12, resource: this.noiseTexture?.view || this._placeholderTexture.view },
                ]
            })

            // Update geometry
            mesh.geometry.update()

            // Render
            passEncoder.setPipeline(pipeline)
            passEncoder.setBindGroup(0, bindGroup)
            passEncoder.setVertexBuffer(0, mesh.geometry.vertexBuffer)
            passEncoder.setVertexBuffer(1, mesh.geometry.instanceBuffer)
            passEncoder.setIndexBuffer(mesh.geometry.indexBuffer, 'uint32')
            passEncoder.drawIndexed(mesh.geometry.indexArray.length, mesh.geometry.instanceCount)

            // Track stats
            stats.transparentDrawCalls++
            stats.transparentTriangles += (mesh.geometry.indexArray.length / 3) * mesh.geometry.instanceCount
        }

        passEncoder.end()
        device.queue.submit([commandEncoder.finish()])
    }

    /**
     * Create placeholder resources for missing bindings
     */
    async _createPlaceholders() {
        const { device } = this.engine

        // 1x1 white texture
        this._placeholderTexture = await Texture.fromRGBA(this.engine, 1, 1, 1, 1)

        // 1x1 normal texture (pointing up)
        this._placeholderNormal = await Texture.fromColor(this.engine, "#8080FF")

        // Sampler
        this._placeholderSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        })

        // Comparison sampler
        this._placeholderComparisonSampler = device.createSampler({
            compare: 'less',
            magFilter: 'linear',
            minFilter: 'linear',
        })

        // 1x1 depth texture
        this._placeholderDepthTexture = device.createTexture({
            size: [1, 1, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.TEXTURE_BINDING,
            dimension: '2d',
        })
        this._placeholderDepth = {
            view: this._placeholderDepthTexture.createView({ dimension: '2d-array', arrayLayerCount: 1 })
        }

        // Placeholder buffer
        this._placeholderBuffer = device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.STORAGE,
        })
    }

    async _resize(width, height) {
        // Nothing to resize
    }

    _destroy() {
        this.pipelineCache.clear()
    }
}

export { TransparentPass }
