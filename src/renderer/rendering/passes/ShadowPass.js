import { BasePass } from "./BasePass.js"
import { mat4, vec3 } from "../../math.js"
import { Frustum } from "../../utils/Frustum.js"
import { calculateShadowBoundingSphere, sphereInCascade, transformBoundingSphere } from "../../utils/BoundingSphere.js"

/**
 * ShadowPass - Shadow map generation for directional and spot lights
 *
 * Pass 1 in the 7-pass pipeline.
 * Generates depth maps from light perspectives.
 */
class ShadowPass extends BasePass {
    constructor(engine = null) {
        super('Shadow', engine)

        // Shadow textures - use texture array for cascades
        this.directionalShadowMap = null // Depth texture array for cascades
        this.spotShadowMaps = []         // Array of depth textures for spot lights

        // Spotlight shadow atlas
        this.spotShadowAtlas = null
        this.spotShadowAtlasView = null

        // Light matrices
        this.directionalLightMatrix = mat4.create() // For backward compatibility
        this.cascadeMatrices = []        // Array of mat4 for each cascade
        this.cascadeViews = []           // Texture views for each cascade layer
        this.spotLightMatrices = []      // Array of mat4 for each shadow slot

        // Shadow slot assignments (lightIndex -> slotIndex, -1 if no shadow)
        this.spotShadowSlots = new Int32Array(128) // Max 128 lights
        this.spotShadowMatrices = []     // Matrices for lights with shadows

        // Pipeline and bind groups
        this.pipeline = null
        this.uniformBuffer = null
        this.bindGroup = null

        // Scene bounds for directional light
        this.sceneBounds = {
            min: [-50, -10, -50],
            max: [50, 50, 50]
        }

        // Noise texture for alpha hashing
        this.noiseTexture = null
        this.noiseSize = 64
        this.noiseAnimated = true

        // HiZ pass reference for occlusion culling of static meshes
        this.hizPass = null

        // Per-mesh bind group cache (for alpha hashing with different albedo textures)
        this._meshBindGroups = new WeakMap()

        // Camera shadow detection state
        this._cameraShadowBuffer = null
        this._cameraShadowReadBuffer = null
        this._cameraShadowPipeline = null
        this._cameraShadowBindGroup = null
        this._cameraShadowUniformBuffer = null
        this._cameraInShadow = false
        this._cameraShadowPending = false
    }

    /**
     * Set the HiZ pass for occlusion culling of static meshes
     * @param {HiZPass} hizPass - HiZ pass instance
     */
    setHiZPass(hizPass) {
        this.hizPass = hizPass
    }

    /**
     * Set the noise texture for alpha hashing in shadows
     * @param {Texture} noise - Noise texture (blue noise or bayer dither)
     * @param {number} size - Texture size
     * @param {boolean} animated - Whether to animate noise offset each frame
     */
    setNoise(noise, size = 64, animated = true) {
        this.noiseTexture = noise
        this.noiseSize = size
        this.noiseAnimated = animated
        // Clear bind group cache since noise texture changed
        this._meshBindGroups = new WeakMap()
        this._skinBindGroups = new WeakMap()
    }

    // Convenience getters for shadow settings (with defaults for backward compatibility)
    get shadowMapSize() { return this.settings?.shadow?.mapSize ?? 2048 }
    get cascadeCount() { return this.settings?.shadow?.cascadeCount ?? 3 }
    get cascadeSizes() { return this.settings?.shadow?.cascadeSizes ?? [20, 60, 300] }
    get maxSpotShadows() { return this.settings?.shadow?.maxSpotShadows ?? 16 }
    get spotTileSize() { return this.settings?.shadow?.spotTileSize ?? 512 }
    get spotAtlasSize() { return this.settings?.shadow?.spotAtlasSize ?? 2048 }
    get spotTilesPerRow() { return this.spotAtlasSize / this.spotTileSize }
    get shadowMaxDistance() { return this.settings?.shadow?.spotMaxDistance ?? 60 }
    get shadowFadeStart() { return this.settings?.shadow?.spotFadeStart ?? 55 }

    async _init() {
        const { device } = this.engine

        // Create directional shadow map as 2D texture array (one layer per cascade)
        this.directionalShadowMap = device.createTexture({
            size: [this.shadowMapSize, this.shadowMapSize, this.cascadeCount],
            format: 'depth32float',
            dimension: '2d',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            label: 'Cascaded Shadow Map'
        })

        // Create view for entire array (for sampling in shader)
        this.directionalShadowMapView = this.directionalShadowMap.createView({
            dimension: '2d-array',
            arrayLayerCount: this.cascadeCount,
        })

        // Create individual views for each cascade layer (for rendering)
        this.cascadeViews = []
        for (let i = 0; i < this.cascadeCount; i++) {
            this.cascadeViews.push(this.directionalShadowMap.createView({
                dimension: '2d',
                baseArrayLayer: i,
                arrayLayerCount: 1,
            }))
        }

        // Initialize cascade matrices
        this.cascadeMatrices = []
        for (let i = 0; i < this.cascadeCount; i++) {
            this.cascadeMatrices.push(mat4.create())
        }

        // Create storage buffer for cascade matrices (3 matrices * 64 bytes = 192 bytes)
        this.cascadeMatricesBuffer = device.createBuffer({
            label: 'Cascade Shadow Matrices',
            size: this.cascadeCount * 16 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.cascadeMatricesData = new Float32Array(this.cascadeCount * 16)

        // Create spotlight shadow atlas
        // spotAtlasSize, spotTileSize, spotTilesPerRow, maxSpotShadows come from settings via getters
        const atlasSize = this.spotAtlasSize
        this.spotAtlasHeight = atlasSize

        this.spotShadowAtlas = device.createTexture({
            size: [this.spotAtlasSize, this.spotAtlasHeight, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            label: 'Spot Shadow Atlas'
        })

        this.spotShadowAtlasView = this.spotShadowAtlas.createView()

        // Initialize spot shadow slot data
        this.spotShadowSlots.fill(-1)
        for (let i = 0; i < this.maxSpotShadows; i++) {
            this.spotLightMatrices.push(mat4.create())
        }

        // Create storage buffer for spot shadow matrices (8 matrices * 64 bytes = 512 bytes)
        this.spotMatricesBuffer = device.createBuffer({
            label: 'Spot Shadow Matrices',
            size: this.maxSpotShadows * 16 * 4, // 8 mat4x4 * 16 floats * 4 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.spotMatricesData = new Float32Array(this.maxSpotShadows * 16)

        // Create sampler for shadow map
        this.shadowSampler = device.createSampler({
            compare: 'less',
            magFilter: 'linear',
            minFilter: 'linear',
        })

        // Create regular sampler for reading depth
        this.depthSampler = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        })

        // Create uniform buffer for light matrix + alpha hash params + surface bias
        // mat4 (64) + vec3+f32 (16) + alpha hash (16) + surfaceBias+padding (16) + lightDir+padding (16) = 128 bytes
        this.uniformBuffer = device.createBuffer({
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Shadow Uniforms'
        })

        // Create placeholder textures
        this._createPlaceholderTextures()

        // Create shadow pipeline
        await this._createPipeline()

        // Create camera shadow detection resources
        await this._createCameraShadowDetection()
    }

    async _createPipeline() {
        const { device } = this.engine

        const shaderModule = device.createShaderModule({
            label: 'Shadow Shader',
            code: `
                struct Uniforms {
                    lightViewProjection: mat4x4f,
                    lightPosition: vec3f,
                    lightType: f32,
                    // Alpha hash params
                    alphaHashEnabled: f32,
                    alphaHashScale: f32,
                    luminanceToAlpha: f32,
                    noiseSize: f32,
                    noiseOffsetX: f32,
                    surfaceBias: f32,           // Expand triangles along normals (meters)
                    _padding: vec2f,
                    lightDirection: vec3f,      // Light direction (for surface bias)
                    _padding2: f32,
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
                    @location(10) instancePosRadius: vec4f,
                }

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) uv: vec2f,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                @group(0) @binding(1) var jointTexture: texture_2d<f32>;
                @group(0) @binding(2) var jointSampler: sampler;
                @group(0) @binding(3) var albedoTexture: texture_2d<f32>;
                @group(0) @binding(4) var albedoSampler: sampler;
                @group(0) @binding(5) var noiseTexture: texture_2d<f32>;

                // Get a 4x4 matrix from the joint texture
                fn getJointMatrix(jointIndex: u32) -> mat4x4f {
                    let row = i32(jointIndex);
                    let col0 = textureLoad(jointTexture, vec2i(0, row), 0);
                    let col1 = textureLoad(jointTexture, vec2i(1, row), 0);
                    let col2 = textureLoad(jointTexture, vec2i(2, row), 0);
                    let col3 = textureLoad(jointTexture, vec2i(3, row), 0);
                    return mat4x4f(col0, col1, col2, col3);
                }

                // Apply skinning to a position
                fn applySkinning(position: vec3f, joints: vec4u, weights: vec4f) -> vec3f {
                    // Check if skinning is active (weights sum > 0)
                    let weightSum = weights.x + weights.y + weights.z + weights.w;
                    if (weightSum < 0.001) {
                        return position;
                    }

                    var skinnedPos = vec3f(0.0);
                    let m0 = getJointMatrix(joints.x);
                    let m1 = getJointMatrix(joints.y);
                    let m2 = getJointMatrix(joints.z);
                    let m3 = getJointMatrix(joints.w);

                    skinnedPos += (m0 * vec4f(position, 1.0)).xyz * weights.x;
                    skinnedPos += (m1 * vec4f(position, 1.0)).xyz * weights.y;
                    skinnedPos += (m2 * vec4f(position, 1.0)).xyz * weights.z;
                    skinnedPos += (m3 * vec4f(position, 1.0)).xyz * weights.w;

                    return skinnedPos;
                }

                // Sample noise at screen position (tiled, no animation for shadows)
                fn sampleNoise(screenPos: vec2f) -> f32 {
                    let noiseSize = i32(uniforms.noiseSize);
                    let noiseOffsetX = i32(uniforms.noiseOffsetX * f32(noiseSize));

                    let texCoord = vec2i(
                        (i32(screenPos.x) + noiseOffsetX) % noiseSize,
                        i32(screenPos.y) % noiseSize
                    );
                    return textureLoad(noiseTexture, texCoord, 0).r;
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

                    // Apply skinning
                    let skinnedPos = applySkinning(input.position, input.joints, input.weights);
                    let worldPos = modelMatrix * vec4f(skinnedPos, 1.0);

                    var clipPos = uniforms.lightViewProjection * worldPos;

                    // Apply surface bias - scale shadow projection to make shadows larger
                    // surfaceBias is treated as a percentage (0.01 = 1% larger shadows)
                    if (uniforms.surfaceBias > 0.0) {
                        let scale = 1.0 + uniforms.surfaceBias;
                        clipPos = vec4f(clipPos.xy * scale, clipPos.z, clipPos.w);
                    }

                    output.position = clipPos;
                    output.uv = input.uv;

                    return output;
                }

                @fragment
                fn fragmentMain(input: VertexOutput) {
                    // Luminance to alpha: hard discard for pure black (no noise)
                    if (uniforms.luminanceToAlpha > 0.5) {
                        let albedo = textureSample(albedoTexture, albedoSampler, input.uv);
                        let luminance = dot(albedo.rgb, vec3f(0.299, 0.587, 0.114));
                        if (luminance < 0.004) {
                            discard;
                        }
                    }
                    // Simple alpha cutoff for shadows (no hashing - too noisy at shadow resolution)
                    else if (uniforms.alphaHashEnabled > 0.5) {
                        let albedo = textureSample(albedoTexture, albedoSampler, input.uv);
                        let alpha = albedo.a * uniforms.alphaHashScale;
                        if (alpha < 0.5) {
                            discard;
                        }
                    }
                    // Depth-only pass, no color output needed
                }
            `
        })

        // Vertex buffer layout (must match geometry)
        const vertexBufferLayout = {
            arrayStride: 80,
            attributes: [
                { format: "float32x3", offset: 0, shaderLocation: 0 },
                { format: "float32x2", offset: 12, shaderLocation: 1 },
                { format: "float32x3", offset: 20, shaderLocation: 2 },
                { format: "float32x4", offset: 32, shaderLocation: 3 },
                { format: "float32x4", offset: 48, shaderLocation: 4 },
                { format: "uint32x4", offset: 64, shaderLocation: 5 },
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

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    texture: { sampleType: 'unfilterable-float' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX,
                    sampler: { type: 'non-filtering' }
                },
                // Albedo texture for alpha hashing
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: 'filtering' }
                },
                // Noise texture for alpha hashing
                {
                    binding: 5,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                }
            ]
        })

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        })

        // Use async pipeline creation for non-blocking initialization
        this.pipeline = await device.createRenderPipelineAsync({
            label: 'Shadow Pipeline',
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
                buffers: [vertexBufferLayout, instanceBufferLayout]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [] // No color attachments
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none', // No culling for shadow map (debug)
            }
        })

        // Create placeholder joint texture for non-skinned meshes
        this._createPlaceholderJointTexture()

        // Default bind group with placeholder textures
        this.bindGroup = device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.placeholderJointTextureView },
                { binding: 2, resource: this.placeholderJointSampler },
                { binding: 3, resource: this.placeholderAlbedoTextureView },
                { binding: 4, resource: this.placeholderAlbedoSampler },
                { binding: 5, resource: this.placeholderNoiseTextureView }
            ]
        })
    }

    _createPlaceholderJointTexture() {
        const { device } = this.engine

        // Create a 4x1 rgba32float texture (one identity matrix)
        this.placeholderJointTexture = device.createTexture({
            size: [4, 1, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })

        // Write identity matrix
        const identityData = new Float32Array([
            1, 0, 0, 0,  // column 0
            0, 1, 0, 0,  // column 1
            0, 0, 1, 0,  // column 2
            0, 0, 0, 1,  // column 3
        ])
        device.queue.writeTexture(
            { texture: this.placeholderJointTexture },
            identityData,
            { bytesPerRow: 4 * 4 * 4, rowsPerImage: 1 },
            [4, 1, 1]
        )

        this.placeholderJointTextureView = this.placeholderJointTexture.createView()
        this.placeholderJointSampler = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        })
    }

    _createPlaceholderTextures() {
        const { device } = this.engine

        // Create placeholder albedo texture (1x1 white with alpha=1)
        // Used for meshes without alpha hashing - alpha=1 means no discard
        this.placeholderAlbedoTexture = device.createTexture({
            size: [1, 1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.writeTexture(
            { texture: this.placeholderAlbedoTexture },
            new Uint8Array([255, 255, 255, 255]),
            { bytesPerRow: 4, rowsPerImage: 1 },
            [1, 1, 1]
        )
        this.placeholderAlbedoTextureView = this.placeholderAlbedoTexture.createView()
        this.placeholderAlbedoSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        })

        // Create placeholder noise texture (1x1 gray = 0.5)
        // Used when no noise texture is configured
        this.placeholderNoiseTexture = device.createTexture({
            size: [1, 1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        device.queue.writeTexture(
            { texture: this.placeholderNoiseTexture },
            new Uint8Array([128, 128, 128, 255]),
            { bytesPerRow: 4, rowsPerImage: 1 },
            [1, 1, 1]
        )
        this.placeholderNoiseTextureView = this.placeholderNoiseTexture.createView()
    }

    /**
     * Get or create a bind group for a mesh (handles skin and albedo for alpha hashing)
     * @param {Mesh} mesh - The mesh to get bind group for
     * @returns {GPUBindGroup} The bind group for this mesh
     */
    getBindGroupForMesh(mesh) {
        const { device } = this.engine

        const skin = mesh?.skin
        const material = mesh?.material
        const hasAlphaHash = material?.alphaHash || mesh?.alphaHash
        const hasLuminanceToAlpha = material?.luminanceToAlpha
        const needsAlbedo = hasAlphaHash || hasLuminanceToAlpha

        // Get albedo texture (first texture in material) or placeholder
        let albedoView = this.placeholderAlbedoTextureView
        let albedoSampler = this.placeholderAlbedoSampler
        if (needsAlbedo && material?.textures?.[0]) {
            albedoView = material.textures[0].view
            albedoSampler = material.textures[0].sampler
        }

        // Get noise texture or placeholder
        const noiseView = this.noiseTexture?.view || this.placeholderNoiseTextureView

        // Get joint texture (from skin or placeholder)
        let jointView = this.placeholderJointTextureView
        let jointSampler = this.placeholderJointSampler
        if (skin?.jointTexture) {
            jointView = skin.jointTextureView
            jointSampler = skin.jointSampler
        }

        // For meshes without alpha hash, luminanceToAlpha, and without skin, use default bind group
        if (!needsAlbedo && !skin?.jointTexture) {
            return this.bindGroup
        }

        // Cache bind groups by mesh
        let bindGroup = this._meshBindGroups.get(mesh)
        if (!bindGroup) {
            bindGroup = device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: jointView },
                    { binding: 2, resource: jointSampler },
                    { binding: 3, resource: albedoView },
                    { binding: 4, resource: albedoSampler },
                    { binding: 5, resource: noiseView }
                ]
            })
            this._meshBindGroups.set(mesh, bindGroup)
        }

        return bindGroup
    }

    /**
     * Get or create a bind group for a specific joint texture (legacy, for backward compatibility)
     */
    getBindGroupForSkin(skin) {
        // For backward compatibility, create a minimal bind group for skinned meshes
        // without alpha hashing
        const { device } = this.engine

        if (!skin || !skin.jointTexture) {
            return this.bindGroup
        }

        // Cache bind groups by skin
        if (!this._skinBindGroups) {
            this._skinBindGroups = new WeakMap()
        }

        let bindGroup = this._skinBindGroups.get(skin)
        if (!bindGroup) {
            bindGroup = device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: skin.jointTextureView },
                    { binding: 2, resource: skin.jointSampler },
                    { binding: 3, resource: this.placeholderAlbedoTextureView },
                    { binding: 4, resource: this.placeholderAlbedoSampler },
                    { binding: 5, resource: this.noiseTexture?.view || this.placeholderNoiseTextureView }
                ]
            })
            this._skinBindGroups.set(skin, bindGroup)
        }

        return bindGroup
    }

    /**
     * Create a frustum from a view-projection matrix for culling
     */
    _createFrustumFromMatrix(viewProj) {
        const frustum = new Frustum()
        frustum._extractPlanes(viewProj)
        return frustum
    }

    /**
     * Test if an instance's bounding sphere is visible to a spotlight using cone culling
     * @param {Object} bsphere - Bounding sphere { center: [x,y,z], radius: r }
     * @param {Array} lightPos - Light position [x, y, z]
     * @param {Array} lightDir - Normalized light direction
     * @param {number} maxDistance - Max shadow distance
     * @param {number} coneAngle - Half-angle of spotlight cone in radians
     * @returns {boolean} True if instance should be rendered
     */
    _isInstanceVisibleToSpotlight(bsphere, lightPos, lightDir, maxDistance, coneAngle) {
        // Vector from light to sphere center
        const dx = bsphere.center[0] - lightPos[0]
        const dy = bsphere.center[1] - lightPos[1]
        const dz = bsphere.center[2] - lightPos[2]
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)

        // Distance test: closest surface must be within max shadow distance
        if (dist - bsphere.radius > maxDistance) {
            return false
        }

        // Skip objects too close (behind light or at light position)
        if (dist < 0.1) {
            return true // Include objects at light position
        }

        // Cone test: check if sphere intersects the spotlight cone
        // Normalize direction to sphere
        const invDist = 1.0 / dist
        const toDirX = dx * invDist
        const toDirY = dy * invDist
        const toDirZ = dz * invDist

        // Dot product with light direction = cos(angle to sphere center)
        const cosAngle = toDirX * lightDir[0] + toDirY * lightDir[1] + toDirZ * lightDir[2]

        // Angular radius of sphere as seen from light (sin approximation for small angles)
        // For larger spheres, use proper asin
        const sinAngularRadius = Math.min(bsphere.radius / dist, 1.0)
        const angularRadius = Math.asin(sinAngularRadius)

        // Sphere is visible if: angle to center - angular radius < cone angle
        // cos(angle) > cos(coneAngle + angularRadius)
        // For efficiency, compare cosines (reversed inequality since cos is decreasing)
        const expandedConeAngle = coneAngle + angularRadius
        const cosExpandedCone = Math.cos(Math.min(expandedConeAngle, Math.PI))

        if (cosAngle < cosExpandedCone) {
            return false // Sphere is outside the expanded cone
        }

        return true
    }

    /**
     * Build filtered instance data for a cascade
     * Returns a Float32Array with only the instances visible to this cascade
     * @param {Object} geometry - Geometry with instanceData
     * @param {mat4} cascadeMatrix - Cascade's view-projection matrix
     * @param {Array} lightDir - Normalized light direction (pointing to light)
     * @param {number} groundLevel - Ground plane Y coordinate
     * @param {Object|null} combinedBsphere - Combined bsphere for skinned models (optional)
     * @returns {{ data: Float32Array, count: number }}
     */
    _buildCascadeFilteredInstances(geometry, cascadeMatrix, lightDir, groundLevel, combinedBsphere = null) {
        const instanceStride = 28 // floats per instance (matrix + posRadius + uvTransform + color)
        const visibleIndices = []

        // Use combined bsphere for skinned models, otherwise fall back to geometry's sphere
        const localBsphere = combinedBsphere || geometry.getBoundingSphere?.()

        for (let i = 0; i < geometry.instanceCount; i++) {
            const offset = i * instanceStride
            let bsphere = {
                center: [
                    geometry.instanceData[offset + 16],
                    geometry.instanceData[offset + 17],
                    geometry.instanceData[offset + 18]
                ],
                radius: Math.abs(geometry.instanceData[offset + 19])
            }

            // If no valid bsphere in instance data, use geometry's local bsphere + transform
            if (bsphere.radius <= 0 && localBsphere && localBsphere.radius > 0) {
                // Extract transform matrix from instance data
                const matrix = geometry.instanceData.subarray(offset, offset + 16)
                // Transform local bsphere by instance matrix
                bsphere = transformBoundingSphere(localBsphere, matrix)
            }

            // Still no valid bsphere - include by default
            if (!bsphere || bsphere.radius <= 0) {
                visibleIndices.push(i)
                continue
            }

            // Calculate shadow bounding sphere for this instance
            const shadowBsphere = calculateShadowBoundingSphere(bsphere, lightDir, groundLevel)

            // Test if shadow bounding sphere intersects this cascade's box
            if (sphereInCascade(shadowBsphere, cascadeMatrix)) {
                visibleIndices.push(i)
            }
        }

        if (visibleIndices.length === 0) {
            return { data: null, count: 0 }
        }

        // If all instances are visible, no need to copy data
        if (visibleIndices.length === geometry.instanceCount) {
            return { data: null, count: geometry.instanceCount, useOriginal: true }
        }

        // Build filtered instance data
        const filteredData = new Float32Array(visibleIndices.length * instanceStride)
        for (let i = 0; i < visibleIndices.length; i++) {
            const srcOffset = visibleIndices[i] * instanceStride
            const dstOffset = i * instanceStride
            for (let j = 0; j < instanceStride; j++) {
                filteredData[dstOffset + j] = geometry.instanceData[srcOffset + j]
            }
        }

        return { data: filteredData, count: visibleIndices.length }
    }

    /**
     * Build filtered instance data for a spotlight using cone culling
     * Returns a Float32Array with only the instances visible to this light
     * @param {Object} geometry - Geometry with instanceData
     * @param {Array} lightPos - Light position
     * @param {Array} lightDir - Normalized light direction
     * @param {number} maxDistance - Max shadow distance (min of light radius and spotMaxDistance)
     * @param {number} coneAngle - Half-angle of spotlight cone in radians
     * @param {Object|null} combinedBsphere - Combined bsphere for skinned models (optional)
     * @returns {{ data: Float32Array, count: number }}
     */
    _buildFilteredInstances(geometry, lightPos, lightDir, maxDistance, coneAngle, combinedBsphere = null) {
        const instanceStride = 28 // floats per instance (matrix + posRadius + uvTransform + color)
        const visibleIndices = []

        // Use combined bsphere for skinned models, otherwise fall back to geometry's sphere
        const localBsphere = combinedBsphere || geometry.getBoundingSphere?.()

        for (let i = 0; i < geometry.instanceCount; i++) {
            const offset = i * instanceStride
            let bsphere = {
                center: [
                    geometry.instanceData[offset + 16],
                    geometry.instanceData[offset + 17],
                    geometry.instanceData[offset + 18]
                ],
                radius: Math.abs(geometry.instanceData[offset + 19])
            }

            // If no valid bsphere in instance data, use geometry's local bsphere + transform
            if (bsphere.radius <= 0 && localBsphere && localBsphere.radius > 0) {
                // Extract transform matrix from instance data
                const matrix = geometry.instanceData.subarray(offset, offset + 16)
                // Transform local bsphere by instance matrix
                bsphere = transformBoundingSphere(localBsphere, matrix)
            }

            // Still no valid bsphere - include by default
            if (!bsphere || bsphere.radius <= 0) {
                visibleIndices.push(i)
                continue
            }

            if (this._isInstanceVisibleToSpotlight(bsphere, lightPos, lightDir, maxDistance, coneAngle)) {
                visibleIndices.push(i)
            }
        }

        if (visibleIndices.length === 0) {
            return { data: null, count: 0 }
        }

        // If all instances are visible, no need to copy data
        if (visibleIndices.length === geometry.instanceCount) {
            return { data: null, count: geometry.instanceCount, useOriginal: true }
        }

        // Build filtered instance data
        const filteredData = new Float32Array(visibleIndices.length * instanceStride)
        for (let i = 0; i < visibleIndices.length; i++) {
            const srcOffset = visibleIndices[i] * instanceStride
            const dstOffset = i * instanceStride
            for (let j = 0; j < instanceStride; j++) {
                filteredData[dstOffset + j] = geometry.instanceData[srcOffset + j]
            }
        }

        return { data: filteredData, count: visibleIndices.length }
    }

    /**
     * Create perspective projection matrix for WebGPU (0-1 depth range)
     */
    perspectiveZO(out, fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2)
        const nf = 1 / (near - far)

        out[0] = f / aspect
        out[1] = 0
        out[2] = 0
        out[3] = 0
        out[4] = 0
        out[5] = f
        out[6] = 0
        out[7] = 0
        out[8] = 0
        out[9] = 0
        out[10] = far * nf        // WebGPU: f/(n-f)
        out[11] = -1
        out[12] = 0
        out[13] = 0
        out[14] = near * far * nf  // WebGPU: n*f/(n-f)
        out[15] = 0

        return out
    }

    /**
     * Create orthographic projection matrix for WebGPU (0-1 depth range)
     * gl-matrix uses OpenGL convention (-1 to 1), so we need a custom version
     */
    orthoZO(out, left, right, bottom, top, near, far) {
        const lr = 1 / (left - right)
        const bt = 1 / (bottom - top)
        const nf = 1 / (near - far)

        out[0] = -2 * lr
        out[1] = 0
        out[2] = 0
        out[3] = 0
        out[4] = 0
        out[5] = -2 * bt
        out[6] = 0
        out[7] = 0
        out[8] = 0
        out[9] = 0
        out[10] = nf           // WebGPU: -1/(f-n) = 1/(n-f) = nf
        out[11] = 0
        out[12] = (left + right) * lr
        out[13] = (top + bottom) * bt
        out[14] = near * nf    // WebGPU: -n/(f-n) = n/(n-f) = near*nf
        out[15] = 1

        return out
    }

    /**
     * Calculate light view-projection matrices for all cascades
     * Each cascade is centered on camera's XZ position for best shadow utilization
     */
    calculateCascadeMatrices(lightDir, camera) {
        const dir = vec3.create()
        vec3.normalize(dir, lightDir)

        // Fixed up vector
        const up = Math.abs(dir[1]) > 0.99
            ? vec3.fromValues(0, 0, 1)
            : vec3.fromValues(0, 1, 0)

        // Camera's XZ position (center cascades here)
        const cameraXZ = vec3.fromValues(camera.position[0], 0, camera.position[2])

        for (let i = 0; i < this.cascadeCount; i++) {
            const lightView = mat4.create()
            const lightProj = mat4.create()

            const frustumSize = this.cascadeSizes[i]
            // Light needs to be far enough to avoid near-plane clipping
            const lightDistance = frustumSize * 2 + 50
            const nearPlane = 1
            const farPlane = lightDistance * 2 + frustumSize

            // Light position: camera XZ + light direction * distance
            const lightPos = vec3.fromValues(
                cameraXZ[0] + dir[0] * lightDistance,
                dir[1] * lightDistance,
                cameraXZ[2] + dir[2] * lightDistance
            )

            // Target is camera's XZ position
            const target = vec3.clone(cameraXZ)

            mat4.lookAt(lightView, lightPos, target, up)
            this.orthoZO(lightProj, -frustumSize, frustumSize, -frustumSize, frustumSize, nearPlane, farPlane)
            mat4.multiply(this.cascadeMatrices[i], lightProj, lightView)
        }

        // For backward compatibility, copy cascade 0 to directionalLightMatrix
        mat4.copy(this.directionalLightMatrix, this.cascadeMatrices[0])

        return this.cascadeMatrices
    }

    /**
     * Calculate spotlight view-projection matrix
     * @param {Object} light - Light with position, direction, geom (radius, innerCone, outerCone)
     * @param {number} slotIndex - Which slot this light is assigned to
     * @returns {mat4} Light view-projection matrix
     */
    calculateSpotLightMatrix(light, slotIndex) {
        const lightView = mat4.create()
        const lightProj = mat4.create()

        const pos = vec3.fromValues(light.position[0], light.position[1], light.position[2])
        const dir = vec3.create()
        vec3.normalize(dir, light.direction)

        // Target = position + direction
        const target = vec3.create()
        vec3.add(target, pos, dir)

        // Up vector - avoid parallel with direction
        const up = Math.abs(dir[1]) > 0.9
            ? vec3.fromValues(1, 0, 0)
            : vec3.fromValues(0, 1, 0)

        mat4.lookAt(lightView, pos, target, up)

        // FOV based on outer cone angle (geom.z is cosine of angle)
        // Convert from cosine to angle, double it for full cone
        // Cap at 120 degrees (60 degree half-angle) for shadow quality
        const outerCone = light.geom[2] || 0.7
        const coneAngle = Math.acos(outerCone)
        const maxShadowAngle = Math.PI / 3 // 60 degrees = 120 degree total FOV
        const shadowAngle = Math.min(coneAngle, maxShadowAngle)
        const fov = shadowAngle * 2.0 + 0.05 // Small margin

        const near = 0.5 // Close enough to capture nearby shadows
        const far = light.geom[0] || 10 // radius

        this.perspectiveZO(lightProj, fov, 1.0, near, far)

        const matrix = this.spotLightMatrices[slotIndex]
        mat4.multiply(matrix, lightProj, lightView)

        return matrix
    }

    /**
     * Assign shadow slots to spotlights based on distance to camera and frustum visibility
     * @param {Array} lights - Array of light objects
     * @param {vec3} cameraPosition - Camera position
     * @param {Frustum} cameraFrustum - Camera frustum for culling
     * @returns {Object} Mapping info for shader
     */
    assignSpotShadowSlots(lights, cameraPosition, cameraFrustum) {
        // Reset all slots
        this.spotShadowSlots.fill(-1)

        // Filter to spotlights (lightType == 2) that affect the visible area
        const spotLights = []
        let culledByFrustum = 0
        let culledByDistance = 0

        for (let i = 0; i < lights.length; i++) {
            const light = lights[i]
            if (!light || !light.enabled) continue
            if (light.lightType !== 2) continue // Only spotlights

            const lightRadius = light.geom?.[0] || 10

            // Distance from light to camera
            const dx = light.position[0] - cameraPosition[0]
            const dy = light.position[1] - cameraPosition[1]
            const dz = light.position[2] - cameraPosition[2]
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

            // Skip lights too far from camera (shadow max distance + light radius)
            // The light could still affect visible geometry even if the light itself is far
            if (distance - lightRadius > this.shadowMaxDistance) {
                culledByDistance++
                continue
            }

            // Frustum cull: check if light's bounding sphere intersects camera frustum
            // The bounding sphere is the light's area of effect
            if (cameraFrustum) {
                const lightBsphere = {
                    center: light.position,
                    radius: lightRadius
                }
                if (!cameraFrustum.testSpherePlanes(lightBsphere)) {
                    culledByFrustum++
                    continue
                }
            }

            spotLights.push({
                index: i,
                light: light,
                distance: distance,
                radius: lightRadius
            })
        }

        // Sort by distance (closest first)
        spotLights.sort((a, b) => a.distance - b.distance)

        // Assign closest visible lights to shadow slots (up to maxSpotShadows)
        const assignments = []
        for (let slot = 0; slot < Math.min(spotLights.length, this.maxSpotShadows); slot++) {
            const spotLight = spotLights[slot]

            this.spotShadowSlots[spotLight.index] = slot

            // Calculate shadow fade factor (1.0 at shadowFadeStart, 0.0 at shadowMaxDistance)
            let fadeFactor = 1.0
            if (spotLight.distance > this.shadowFadeStart) {
                fadeFactor = 1.0 - (spotLight.distance - this.shadowFadeStart) /
                                   (this.shadowMaxDistance - this.shadowFadeStart)
                fadeFactor = Math.max(0, fadeFactor)
            }

            assignments.push({
                slot: slot,
                lightIndex: spotLight.index,
                light: spotLight.light,
                distance: spotLight.distance,
                fadeFactor: fadeFactor
            })
        }

        return {
            assignments: assignments,
            totalSpotLights: spotLights.length,
            culledByFrustum: culledByFrustum,
            culledByDistance: culledByDistance
        }
    }

    async _execute(context) {
        const { device, stats } = this.engine
        const { camera, meshes, mainLight, lights } = context

        if (!this.pipeline || !meshes) {
            console.warn('ShadowPass: No pipeline or meshes', { pipeline: !!this.pipeline, meshes: !!meshes })
            return
        }

        // Clear bind group caches for skinned meshes to ensure fresh joint textures are bound
        // This prevents stale bind groups from causing shadow artifacts on animated meshes
        this._meshBindGroups = new WeakMap()
        if (this._skinBindGroups) {
            this._skinBindGroups = new WeakMap()
        }

        // Track shadow pass stats
        let shadowDrawCalls = 0
        let shadowTriangles = 0
        let shadowCulledInstances = 0

        // Check if main directional light is enabled
        const mainLightEnabled = !mainLight || mainLight.enabled !== false

        // Calculate cascade matrices (centered on camera XZ) - even if disabled, for consistent state
        const dir = vec3.fromValues(
            mainLight?.direction?.[0] ?? -1,
            mainLight?.direction?.[1] ?? 1,
            mainLight?.direction?.[2] ?? -0.5
        )
        this.calculateCascadeMatrices(dir, camera)

        // ===================
        // CASCADED DIRECTIONAL SHADOWS
        // ===================

        // Uniform buffer layout: mat4 (16) + vec3+f32 (4) + alpha hash params (5) + padding (3) = 28 floats (112 bytes)
        const uniformData = new Float32Array(28)
        let totalInstances = 0
        let totalTriangles = 0

        // Noise offset for alpha hashing - always static to avoid shimmer on cutout edges
        const noiseOffsetX = 0
        const noiseOffsetY = 0

        // Get light direction for shadow bounding sphere calculation
        const lightDir = vec3.fromValues(
            mainLight?.direction?.[0] ?? -1,
            mainLight?.direction?.[1] ?? 1,
            mainLight?.direction?.[2] ?? -0.5
        )
        vec3.normalize(lightDir, lightDir)

        // Ground level for shadow projection
        const groundLevel = this.settings?.planarReflection?.groundLevel ?? 0

        // Create camera frustum for culling static meshes
        const cameraFrustum = this._createFrustumFromMatrix(camera.viewProj)
        const shadowConfig = this.settings?.culling?.shadow
        const shadowFrustumCullingEnabled = shadowConfig?.frustum !== false
        const shadowHiZEnabled = shadowConfig?.hiZ !== false && this.hizPass
        const shadowMaxDistance = shadowConfig?.maxDistance ?? 100

        // Pre-filter static meshes by shadow bounding sphere visibility
        // This is used for BOTH cascade and spotlight shadows
        // Entity-managed meshes are already filtered in RenderGraph, but static meshes aren't
        const visibleMeshes = {}
        let meshFrustumCulled = 0
        let meshDistanceCulled = 0
        let meshOcclusionCulled = 0
        let meshNoBsphere = 0

        for (const name in meshes) {
            const mesh = meshes[name]
            const geometry = mesh.geometry
            if (!geometry || geometry.instanceCount === 0) continue

            // Entity-managed meshes (not static) are already culled - include them
            if (!mesh.static) {
                visibleMeshes[name] = mesh
                continue
            }

            // For static meshes, apply shadow bounding sphere culling
            // For skinned meshes with multiple submeshes, use combined bsphere if available
            // This ensures all submeshes are culled together as a unit
            const localBsphere = mesh.combinedBsphere || geometry.getBoundingSphere?.()
            if (!localBsphere || localBsphere.radius <= 0) {
                // No bsphere - include but track it
                meshNoBsphere++
                visibleMeshes[name] = mesh
                continue
            }

            // Has valid bsphere - apply culling
            // Get world bounding sphere (transform by first instance matrix)
            const matrix = geometry.instanceData?.subarray(0, 16)
            const worldBsphere = matrix ?
                transformBoundingSphere(localBsphere, matrix) :
                localBsphere

            // Calculate shadow bounding sphere (only if main light enabled)
            // For spotlights only, use object's own bsphere for culling
            const shadowBsphere = mainLightEnabled
                ? calculateShadowBoundingSphere(worldBsphere, lightDir, groundLevel)
                : worldBsphere

            // For skinned meshes, expand the shadow bsphere to account for animation
            // Animated poses can extend beyond the rest pose bounding sphere
            const skinnedExpansion = this.engine?.settings?.shadow?.skinnedBsphereExpansion ?? 2.0
            const cullBsphere = mesh.hasSkin ? {
                center: shadowBsphere.center,
                radius: shadowBsphere.radius * skinnedExpansion
            } : shadowBsphere

            // Distance culling - skip if shadow sphere is too far from camera
            const dx = cullBsphere.center[0] - camera.position[0]
            const dy = cullBsphere.center[1] - camera.position[1]
            const dz = cullBsphere.center[2] - camera.position[2]
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) - cullBsphere.radius
            if (distance > shadowMaxDistance) {
                meshDistanceCulled++
                continue
            }

            // Frustum culling - skip if shadow not visible to camera
            if (shadowFrustumCullingEnabled && cameraFrustum) {
                if (!cameraFrustum.testSpherePlanes(cullBsphere)) {
                    meshFrustumCulled++
                    continue
                }
            }

            // HiZ occlusion culling - skip if shadow sphere is fully occluded
            if (shadowHiZEnabled && this.hizPass) {
                const occluded = this.hizPass.testSphereOcclusion(
                    cullBsphere,
                    camera.viewProj,
                    camera.near,
                    camera.far,
                    camera.position
                )
                if (occluded) {
                    meshOcclusionCulled++
                    continue
                }
            }

            visibleMeshes[name] = mesh
        }

        // Store mesh culling stats for reporting
        this._lastMeshFrustumCulled = meshFrustumCulled
        this._lastMeshDistanceCulled = meshDistanceCulled
        this._lastMeshOcclusionCulled = meshOcclusionCulled
        this._lastMeshNoBsphere = meshNoBsphere

        // Only render cascade shadows if main light is enabled
        if (mainLightEnabled) {
            // Check if per-cascade filtering is enabled
            const cascadeFilterEnabled = this.settings?.culling?.shadow?.cascadeFilter !== false

            // Per-cascade culling stats
            let cascadeCulledInstances = 0

            // Render each cascade - submit separately to ensure correct matrix
            for (let cascade = 0; cascade < this.cascadeCount; cascade++) {
                // Update uniform buffer with this cascade's matrix
                uniformData.set(this.cascadeMatrices[cascade], 0)  // 0-15
                uniformData.set([0, 100, 0], 16)                    // 16-18 (lightPosition)
                uniformData[19] = 0                                  // lightType: directional
                // Alpha hash params (enabled globally - per-mesh control via albedo texture)
                const globalLuminanceToAlpha = this.settings?.rendering?.luminanceToAlpha ? 1.0 : 0.0
                uniformData[20] = 1.0                                // alphaHashEnabled
                uniformData[21] = 1.0                                // alphaHashScale
                uniformData[22] = globalLuminanceToAlpha             // luminanceToAlpha
                uniformData[23] = this.noiseSize                     // noiseSize
                uniformData[24] = noiseOffsetX                       // noiseOffsetX
                uniformData[25] = this.settings?.shadow?.surfaceBias ?? 0  // surfaceBias
                // 26-27 are padding
                uniformData[28] = lightDir[0]                        // lightDirection.x
                uniformData[29] = lightDir[1]                        // lightDirection.y
                uniformData[30] = lightDir[2]                        // lightDirection.z
                // 31 is padding
                device.queue.writeBuffer(this.uniformBuffer, 0, uniformData)

                // Create command encoder for this cascade
                const cascadeEncoder = device.createCommandEncoder({
                    label: `Shadow Cascade ${cascade}`
                })

                // Render to this cascade's layer
                const cascadePass = cascadeEncoder.beginRenderPass({
                    colorAttachments: [],
                    depthStencilAttachment: {
                        view: this.cascadeViews[cascade],
                        depthClearValue: 1.0,
                        depthLoadOp: 'clear',
                        depthStoreOp: 'store',
                    }
                })

                cascadePass.setPipeline(this.pipeline)

                // Collect filtered instances for this cascade
                const meshFilters = []
                let totalFilteredFloats = 0
                const instanceStride = 28 // floats per instance (matrix + posRadius + uvTransform + color)

                for (const name in visibleMeshes) {
                    const mesh = visibleMeshes[name]
                    const geometry = mesh.geometry
                    if (geometry.instanceCount === 0) continue
                    if (cascade === 0) geometry.update() // Only update geometry once

                    // Apply per-cascade filtering if enabled
                    let filtered = null
                    if (cascadeFilterEnabled) {
                        filtered = this._buildCascadeFilteredInstances(
                            geometry,
                            this.cascadeMatrices[cascade],
                            lightDir,
                            groundLevel,
                            mesh.combinedBsphere // Use combined bsphere for skinned models
                        )

                        if (filtered.count === 0) {
                            cascadeCulledInstances += geometry.instanceCount
                            continue
                        }

                        cascadeCulledInstances += geometry.instanceCount - filtered.count
                    }

                    // If filtering returned useOriginal, or filtering disabled, use original buffer
                    if (!cascadeFilterEnabled || filtered?.useOriginal) {
                        meshFilters.push({
                            mesh,
                            geometry,
                            useOriginal: true,
                            count: geometry.instanceCount
                        })
                    } else {
                        // Need to use filtered data
                        meshFilters.push({
                            mesh,
                            geometry,
                            filtered,
                            byteOffset: totalFilteredFloats * 4
                        })
                        totalFilteredFloats += filtered.count * instanceStride
                    }
                }

                // Create/resize cascade temp buffer if needed
                const totalBufferSize = totalFilteredFloats * 4
                if (totalBufferSize > 0) {
                    if (!this._cascadeTempBuffer || this._cascadeTempBufferSize < totalBufferSize) {
                        if (this._cascadeTempBuffer) {
                            this._cascadeTempBuffer.destroy()
                        }
                        const allocSize = Math.max(totalBufferSize, 65536) // Min 64KB
                        this._cascadeTempBuffer = device.createBuffer({
                            size: allocSize,
                            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                            label: 'Cascade Shadow Temp Instance Buffer'
                        })
                        this._cascadeTempBufferSize = allocSize
                    }

                    // Write filtered instance data at their respective offsets
                    for (const mf of meshFilters) {
                        if (!mf.useOriginal && mf.filtered?.data) {
                            device.queue.writeBuffer(this._cascadeTempBuffer, mf.byteOffset, mf.filtered.data)
                        }
                    }
                }

                // Separate meshes by luminanceToAlpha flag for proper uniform handling
                const regularMeshes = meshFilters.filter(mf => !mf.mesh.material?.luminanceToAlpha)
                const luminanceMeshes = meshFilters.filter(mf => mf.mesh.material?.luminanceToAlpha)

                // Render regular meshes (luminanceToAlpha = 0)
                for (const mf of regularMeshes) {
                    const bindGroup = this.getBindGroupForMesh(mf.mesh)
                    cascadePass.setBindGroup(0, bindGroup)

                    cascadePass.setVertexBuffer(0, mf.geometry.vertexBuffer)

                    if (mf.useOriginal) {
                        cascadePass.setVertexBuffer(1, mf.geometry.instanceBuffer)
                        cascadePass.setIndexBuffer(mf.geometry.indexBuffer, 'uint32')
                        cascadePass.drawIndexed(mf.geometry.indexArray.length, mf.count)

                        shadowDrawCalls++
                        shadowTriangles += (mf.geometry.indexArray.length / 3) * mf.count
                    } else {
                        cascadePass.setVertexBuffer(1, this._cascadeTempBuffer, mf.byteOffset)
                        cascadePass.setIndexBuffer(mf.geometry.indexBuffer, 'uint32')
                        cascadePass.drawIndexed(mf.geometry.indexArray.length, mf.filtered.count)

                        shadowDrawCalls++
                        shadowTriangles += (mf.geometry.indexArray.length / 3) * mf.filtered.count
                    }

                    if (cascade === 0) {
                        const count = mf.useOriginal ? mf.count : mf.filtered.count
                        totalInstances += count
                        totalTriangles += (mf.geometry.indexArray.length / 3) * count
                    }
                }

                cascadePass.end()
                device.queue.submit([cascadeEncoder.finish()])

                // Render luminanceToAlpha meshes in separate pass with updated uniform
                if (luminanceMeshes.length > 0) {
                    uniformData[22] = 1.0  // Enable luminanceToAlpha
                    device.queue.writeBuffer(this.uniformBuffer, 0, uniformData)

                    const lumEncoder = device.createCommandEncoder({ label: `Shadow Cascade ${cascade} LumAlpha` })
                    const lumPass = lumEncoder.beginRenderPass({
                        colorAttachments: [],
                        depthStencilAttachment: {
                            view: this.cascadeViews[cascade],
                            depthClearValue: 1.0,
                            depthLoadOp: 'load',  // Keep existing depth
                            depthStoreOp: 'store',
                        }
                    })

                    lumPass.setPipeline(this.pipeline)

                    for (const mf of luminanceMeshes) {
                        const bindGroup = this.getBindGroupForMesh(mf.mesh)
                        lumPass.setBindGroup(0, bindGroup)

                        lumPass.setVertexBuffer(0, mf.geometry.vertexBuffer)

                        if (mf.useOriginal) {
                            lumPass.setVertexBuffer(1, mf.geometry.instanceBuffer)
                            lumPass.setIndexBuffer(mf.geometry.indexBuffer, 'uint32')
                            lumPass.drawIndexed(mf.geometry.indexArray.length, mf.count)

                            shadowDrawCalls++
                            shadowTriangles += (mf.geometry.indexArray.length / 3) * mf.count
                        } else {
                            lumPass.setVertexBuffer(1, this._cascadeTempBuffer, mf.byteOffset)
                            lumPass.setIndexBuffer(mf.geometry.indexBuffer, 'uint32')
                            lumPass.drawIndexed(mf.geometry.indexArray.length, mf.filtered.count)

                            shadowDrawCalls++
                            shadowTriangles += (mf.geometry.indexArray.length / 3) * mf.filtered.count
                        }

                        if (cascade === 0) {
                            const count = mf.useOriginal ? mf.count : mf.filtered.count
                            totalInstances += count
                            totalTriangles += (mf.geometry.indexArray.length / 3) * count
                        }
                    }

                    lumPass.end()
                    device.queue.submit([lumEncoder.finish()])

                    // Reset luminanceToAlpha for next cascade
                    uniformData[22] = 0.0
                }
            }

            // Store cascade culling stats
            shadowCulledInstances += cascadeCulledInstances

        } // End if (mainLightEnabled)

        // Update cascade matrices storage buffer (always, for consistent state)
        for (let i = 0; i < this.cascadeCount; i++) {
            this.cascadeMatricesData.set(this.cascadeMatrices[i], i * 16)
        }
        device.queue.writeBuffer(this.cascadeMatricesBuffer, 0, this.cascadeMatricesData)

        // Update camera shadow detection (for adaptive volumetric fog)
        if (mainLightEnabled) {
            this._updateCameraShadowDetection(camera)
        }

        // ===================
        // SPOTLIGHT SHADOWS (always runs, even when main light is disabled)
        // ===================

        // If main light was disabled, we need to update geometry buffers here
        // (normally done in cascade loop, but that was skipped)
        if (!mainLightEnabled) {
            for (const name in meshes) {
                const mesh = meshes[name]
                const geometry = mesh.geometry
                if (geometry.instanceCount > 0) {
                    geometry.update()
                }
            }
        }

        // Reset slot info
        this.lastSlotInfo = { assignments: [], totalSpotLights: 0, culledByFrustum: 0, culledByDistance: 0 }
        this.spotShadowSlots.fill(-1)

        // Assign shadow slots to closest spotlights that affect visible area
        if (lights && lights.length > 0 && this.spotShadowAtlas) {
            // Create camera frustum for culling spotlights
            const cameraFrustum = this._createFrustumFromMatrix(camera.viewProj)
            const slotInfo = this.assignSpotShadowSlots(lights, camera.position, cameraFrustum)
            this.lastSlotInfo = slotInfo

            // Clear the atlas first - use separate encoder and submit immediately
            const clearEncoder = device.createCommandEncoder({ label: 'Spot Shadow Clear' })
            const clearPass = clearEncoder.beginRenderPass({
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this.spotShadowAtlasView,
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                }
            })
            clearPass.end()
            device.queue.submit([clearEncoder.finish()])

            // Render each spotlight shadow - IMPORTANT: Submit each one separately
            // because writeBuffer calls are queued and would all execute before any render pass
            for (const assignment of slotInfo.assignments) {
                // Calculate spotlight matrix
                this.calculateSpotLightMatrix(assignment.light, assignment.slot)
                const spotMatrix = this.spotLightMatrices[assignment.slot]

                // Extract spotlight parameters for cone culling
                const lightPos = assignment.light.position
                const lightRadius = assignment.light.geom[0] || 10

                // Normalize light direction
                const spotLightDir = vec3.create()
                vec3.normalize(spotLightDir, assignment.light.direction)

                // Max shadow distance is minimum of light radius and spotMaxDistance setting
                const spotShadowMaxDist = Math.min(lightRadius, this.shadowMaxDistance)

                // Cone angle from outer cone (geom[2] is cosine of half-angle)
                const outerConeCos = assignment.light.geom[2] || 0.7
                const coneAngle = Math.acos(outerConeCos)

                // Update uniform buffer with spotlight matrix and alpha hash params
                uniformData.set(spotMatrix, 0)                       // 0-15
                uniformData.set(lightPos, 16)                        // 16-18 (lightPosition)
                uniformData[19] = 2                                  // lightType: spotlight
                // Alpha hash params (same as cascaded shadows)
                const spotLuminanceToAlpha = this.settings?.rendering?.luminanceToAlpha ? 1.0 : 0.0
                uniformData[20] = 1.0                                // alphaHashEnabled
                uniformData[21] = 1.0                                // alphaHashScale
                uniformData[22] = spotLuminanceToAlpha               // luminanceToAlpha
                uniformData[23] = this.noiseSize                     // noiseSize
                uniformData[24] = noiseOffsetX                       // noiseOffsetX
                uniformData[25] = this.settings?.shadow?.surfaceBias ?? 0  // surfaceBias
                // 26-27 are padding
                uniformData[28] = spotLightDir[0]                    // lightDirection.x
                uniformData[29] = spotLightDir[1]                    // lightDirection.y
                uniformData[30] = spotLightDir[2]                    // lightDirection.z
                // 31 is padding
                device.queue.writeBuffer(this.uniformBuffer, 0, uniformData)

                // Calculate viewport for this slot in atlas
                const col = assignment.slot % this.spotTilesPerRow
                const row = Math.floor(assignment.slot / this.spotTilesPerRow)
                const x = col * this.spotTileSize
                const y = row * this.spotTileSize

                // Create a separate command encoder for each spotlight to ensure
                // the writeBuffer takes effect before rendering
                const spotEncoder = device.createCommandEncoder({
                    label: `Spot Shadow ${assignment.slot}`
                })

                // Render to this tile using viewport
                const spotPass = spotEncoder.beginRenderPass({
                    colorAttachments: [],
                    depthStencilAttachment: {
                        view: this.spotShadowAtlasView,
                        depthClearValue: 1.0,
                        depthLoadOp: 'load', // Don't clear, we already did
                        depthStoreOp: 'store',
                    }
                })

                spotPass.setPipeline(this.pipeline)
                spotPass.setViewport(x, y, this.spotTileSize, this.spotTileSize, 0, 1)
                spotPass.setScissorRect(x, y, this.spotTileSize, this.spotTileSize)

                // First pass: collect all filtered instances and calculate offsets
                const meshFilters = []
                let totalFilteredFloats = 0
                let spotCulledInstances = 0
                const instanceStride = 28 // floats per instance (matrix + posRadius + uvTransform + color)

                // Use visibleMeshes for spotlight shadows - applies same static mesh
                // culling (frustum, distance, HiZ) as cascade shadows
                for (const name in visibleMeshes) {
                    const mesh = visibleMeshes[name]
                    const geometry = mesh.geometry
                    if (geometry.instanceCount === 0) continue

                    // Build filtered instances using cone culling
                    const filtered = this._buildFilteredInstances(
                        geometry, lightPos, spotLightDir, spotShadowMaxDist, coneAngle,
                        mesh.combinedBsphere // Use combined bsphere for skinned models
                    )

                    if (filtered.count === 0) {
                        spotCulledInstances += geometry.instanceCount
                        continue
                    }

                    spotCulledInstances += geometry.instanceCount - filtered.count

                    // Handle useOriginal optimization (all instances visible)
                    if (filtered.useOriginal) {
                        meshFilters.push({
                            mesh,
                            geometry,
                            useOriginal: true,
                            count: filtered.count
                        })
                    } else {
                        meshFilters.push({
                            mesh,
                            geometry,
                            filtered,
                            byteOffset: totalFilteredFloats * 4 // offset in bytes
                        })
                        totalFilteredFloats += filtered.count * instanceStride
                    }
                }

                // Create/resize buffer if needed for filtered instances
                const totalBufferSize = totalFilteredFloats * 4
                if (totalBufferSize > 0) {
                    if (!this._tempInstanceBuffer || this._tempInstanceBufferSize < totalBufferSize) {
                        if (this._tempInstanceBuffer) {
                            this._tempInstanceBuffer.destroy()
                        }
                        const allocSize = Math.max(totalBufferSize, 16384) // Min 16KB
                        this._tempInstanceBuffer = device.createBuffer({
                            size: allocSize,
                            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                            label: 'Spot Shadow Temp Instance Buffer'
                        })
                        this._tempInstanceBufferSize = allocSize
                    }

                    // Write filtered instance data at their respective offsets
                    for (const mf of meshFilters) {
                        if (!mf.useOriginal && mf.filtered?.data) {
                            device.queue.writeBuffer(this._tempInstanceBuffer, mf.byteOffset, mf.filtered.data)
                        }
                    }
                }

                // Separate meshes by luminanceToAlpha flag
                const regularMeshes = meshFilters.filter(mf => !mf.mesh.material?.luminanceToAlpha)
                const luminanceMeshes = meshFilters.filter(mf => mf.mesh.material?.luminanceToAlpha)

                // Render regular meshes (luminanceToAlpha = 0)
                for (const mf of regularMeshes) {
                    const bindGroup = this.getBindGroupForMesh(mf.mesh)
                    spotPass.setBindGroup(0, bindGroup)
                    spotPass.setVertexBuffer(0, mf.geometry.vertexBuffer)

                    if (mf.useOriginal) {
                        spotPass.setVertexBuffer(1, mf.geometry.instanceBuffer)
                        spotPass.setIndexBuffer(mf.geometry.indexBuffer, 'uint32')
                        spotPass.drawIndexed(mf.geometry.indexArray.length, mf.count)
                    } else {
                        spotPass.setVertexBuffer(1, this._tempInstanceBuffer, mf.byteOffset)
                        spotPass.setIndexBuffer(mf.geometry.indexBuffer, 'uint32')
                        spotPass.drawIndexed(mf.geometry.indexArray.length, mf.filtered.count)
                    }
                }

                spotPass.end()
                device.queue.submit([spotEncoder.finish()])

                // Render luminanceToAlpha meshes in separate pass
                if (luminanceMeshes.length > 0) {
                    uniformData[22] = 1.0  // Enable luminanceToAlpha
                    device.queue.writeBuffer(this.uniformBuffer, 0, uniformData)

                    const lumEncoder = device.createCommandEncoder({ label: `Spot Shadow ${assignment.slot} LumAlpha` })
                    const lumPass = lumEncoder.beginRenderPass({
                        colorAttachments: [],
                        depthStencilAttachment: {
                            view: this.spotShadowAtlasView,
                            depthClearValue: 1.0,
                            depthLoadOp: 'load',  // Keep existing depth
                            depthStoreOp: 'store',
                        }
                    })

                    lumPass.setPipeline(this.pipeline)
                    lumPass.setViewport(x, y, this.spotTileSize, this.spotTileSize, 0, 1)
                    lumPass.setScissorRect(x, y, this.spotTileSize, this.spotTileSize)

                    for (const mf of luminanceMeshes) {
                        const bindGroup = this.getBindGroupForMesh(mf.mesh)
                        lumPass.setBindGroup(0, bindGroup)
                        lumPass.setVertexBuffer(0, mf.geometry.vertexBuffer)

                        if (mf.useOriginal) {
                            lumPass.setVertexBuffer(1, mf.geometry.instanceBuffer)
                            lumPass.setIndexBuffer(mf.geometry.indexBuffer, 'uint32')
                            lumPass.drawIndexed(mf.geometry.indexArray.length, mf.count)
                        } else {
                            lumPass.setVertexBuffer(1, this._tempInstanceBuffer, mf.byteOffset)
                            lumPass.setIndexBuffer(mf.geometry.indexBuffer, 'uint32')
                            lumPass.drawIndexed(mf.geometry.indexArray.length, mf.filtered.count)
                        }
                    }

                    lumPass.end()
                    device.queue.submit([lumEncoder.finish()])

                    // Reset luminanceToAlpha
                    uniformData[22] = 0.0
                }

                // Track stats for spotlight shadows
                for (const mf of meshFilters) {
                    shadowDrawCalls++
                    const count = mf.useOriginal ? mf.count : mf.filtered.count
                    shadowTriangles += (mf.geometry.indexArray.length / 3) * count
                }
                shadowCulledInstances += spotCulledInstances
            }

            // Update spot matrices storage buffer with all calculated matrices
            for (let i = 0; i < this.maxSpotShadows; i++) {
                this.spotMatricesData.set(this.spotLightMatrices[i], i * 16)
            }
            device.queue.writeBuffer(this.spotMatricesBuffer, 0, this.spotMatricesData)


            // IMPORTANT: Restore directional light matrix to uniform buffer
            // This prevents the last spotlight matrix from corrupting subsequent passes
            uniformData.set(this.directionalLightMatrix, 0)
            uniformData.set([0, 100, 0], 16)
            uniformData[19] = 0 // Light type: directional
            device.queue.writeBuffer(this.uniformBuffer, 0, uniformData)
        }

        // Add shadow stats to global stats
        stats.shadowDrawCalls = shadowDrawCalls
        stats.shadowTriangles = shadowTriangles
        stats.shadowCulledInstances = shadowCulledInstances

        // Add mesh culling stats (stored from mainLightEnabled block)
        stats.shadowMeshFrustumCulled = this._lastMeshFrustumCulled || 0
        stats.shadowMeshDistanceCulled = this._lastMeshDistanceCulled || 0
        stats.shadowMeshOcclusionCulled = this._lastMeshOcclusionCulled || 0
        stats.shadowMeshNoBsphere = this._lastMeshNoBsphere || 0
    }

    async _resize(width, height) {
        // Shadow maps don't resize with screen
    }

    _destroy() {
        if (this.directionalShadowMap) {
            this.directionalShadowMap.destroy()
        }
        if (this.spotShadowAtlas) {
            this.spotShadowAtlas.destroy()
        }
    }

    /**
     * Get shadow map texture for lighting pass (texture array)
     */
    getShadowMap() {
        return this.directionalShadowMap
    }

    /**
     * Get shadow map view for binding (2d-array view)
     */
    getShadowMapView() {
        return this.directionalShadowMapView
    }

    /**
     * Get cascade matrices storage buffer
     */
    getCascadeMatricesBuffer() {
        return this.cascadeMatricesBuffer
    }

    /**
     * Get cascade sizes array
     */
    getCascadeSizes() {
        return this.cascadeSizes
    }

    /**
     * Get number of cascades
     */
    getCascadeCount() {
        return this.cascadeCount
    }

    /**
     * Get shadow sampler (comparison sampler)
     */
    getShadowSampler() {
        return this.shadowSampler
    }

    /**
     * Get depth sampler (regular sampler)
     */
    getDepthSampler() {
        return this.depthSampler
    }

    /**
     * Get light matrix for shader
     */
    getLightMatrix() {
        return this.directionalLightMatrix
    }

    /**
     * Get spot shadow atlas texture
     */
    getSpotShadowAtlas() {
        return this.spotShadowAtlas
    }

    /**
     * Get spot shadow atlas view
     */
    getSpotShadowAtlasView() {
        return this.spotShadowAtlasView
    }

    /**
     * Get spotlight shadow matrices (array of mat4)
     */
    getSpotLightMatrices() {
        return this.spotLightMatrices
    }

    /**
     * Get the storage buffer containing spot shadow matrices
     */
    getSpotMatricesBuffer() {
        return this.spotMatricesBuffer
    }

    /**
     * Get slot assignments for each light (-1 if no shadow)
     */
    getSpotShadowSlots() {
        return this.spotShadowSlots
    }

    /**
     * Get shadow parameters for shader
     */
    getSpotShadowParams() {
        return {
            atlasSize: [this.spotAtlasSize, this.spotAtlasHeight],
            tileSize: this.spotTileSize,
            tilesPerRow: this.spotTilesPerRow,
            maxSlots: this.maxSpotShadows,
            fadeStart: this.shadowFadeStart,
            maxDistance: this.shadowMaxDistance
        }
    }

    /**
     * Get last frame's slot assignment info (for debugging)
     */
    getLastSlotInfo() {
        return this.lastSlotInfo
    }

    /**
     * Get cascade matrices as JavaScript arrays (for CPU-side calculations)
     */
    getCascadeMatrices() {
        return this.cascadeMatrices
    }

    /**
     * Create resources for camera shadow detection
     * Uses a compute shader to sample shadow at camera position
     */
    async _createCameraShadowDetection() {
        const { device } = this.engine

        // Create uniform buffer for camera position and cascade matrices
        // vec3 cameraPos + pad + 3 x mat4 cascadeMatrices = 4 + 48 = 52 floats = 208 bytes
        this._cameraShadowUniformBuffer = device.createBuffer({
            label: 'Camera Shadow Detection Uniforms',
            size: 256, // Aligned
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // Create output buffer (1 float for shadow result)
        this._cameraShadowBuffer = device.createBuffer({
            label: 'Camera Shadow Result',
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        })

        // Create readback buffer
        this._cameraShadowReadBuffer = device.createBuffer({
            label: 'Camera Shadow Readback',
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        })

        // Create compute shader
        const shaderModule = device.createShaderModule({
            label: 'Camera Shadow Detection Shader',
            code: `
                struct Uniforms {
                    cameraPosition: vec3f,
                    _pad0: f32,
                    cascadeMatrix0: mat4x4f,
                    cascadeMatrix1: mat4x4f,
                    cascadeMatrix2: mat4x4f,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                @group(0) @binding(1) var shadowMap: texture_depth_2d_array;
                @group(0) @binding(2) var shadowSampler: sampler_comparison;
                @group(0) @binding(3) var<storage, read_write> result: f32;

                fn sampleShadowCascade(worldPos: vec3f, cascadeMatrix: mat4x4f, cascadeIndex: i32) -> f32 {
                    let lightSpacePos = cascadeMatrix * vec4f(worldPos, 1.0);
                    let projCoords = lightSpacePos.xyz / lightSpacePos.w;

                    // Convert to UV space
                    let uv = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);

                    // Check bounds
                    if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99 ||
                        projCoords.z < 0.0 || projCoords.z > 1.0) {
                        return -1.0; // Out of bounds, try next cascade
                    }

                    let bias = 0.005;
                    let depth = projCoords.z - bias;
                    return textureSampleCompareLevel(shadowMap, shadowSampler, uv, cascadeIndex, depth);
                }

                @compute @workgroup_size(1)
                fn main() {
                    let pos = uniforms.cameraPosition;

                    // Sample multiple points around camera (5m sphere)
                    var totalShadow = 0.0;
                    var sampleCount = 0.0;

                    let offsets = array<vec3f, 7>(
                        vec3f(0.0, 0.0, 0.0),   // Center
                        vec3f(0.0, 3.0, 0.0),   // Above
                        vec3f(0.0, -2.0, 0.0),  // Below
                        vec3f(4.0, 0.0, 0.0),   // Right
                        vec3f(-4.0, 0.0, 0.0),  // Left
                        vec3f(0.0, 0.0, 4.0),   // Front
                        vec3f(0.0, 0.0, -4.0),  // Back
                    );

                    for (var i = 0; i < 7; i++) {
                        let samplePos = pos + offsets[i];

                        // Try cascade 0 first (closest)
                        var shadow = sampleShadowCascade(samplePos, uniforms.cascadeMatrix0, 0);
                        if (shadow < 0.0) {
                            // Try cascade 1
                            shadow = sampleShadowCascade(samplePos, uniforms.cascadeMatrix1, 1);
                        }
                        if (shadow < 0.0) {
                            // Try cascade 2
                            shadow = sampleShadowCascade(samplePos, uniforms.cascadeMatrix2, 2);
                        }

                        if (shadow >= 0.0) {
                            totalShadow += shadow;
                            sampleCount += 1.0;
                        }
                    }

                    // Average shadow (0 = all in shadow, 1 = all lit)
                    // If no valid samples, assume lit
                    if (sampleCount > 0.0) {
                        result = totalShadow / sampleCount;
                    } else {
                        result = 1.0;
                    }
                }
            `
        })

        // Create bind group layout
        this._cameraShadowBGL = device.createBindGroupLayout({
            label: 'Camera Shadow Detection BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'comparison' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        })

        // Create pipeline
        this._cameraShadowPipeline = await device.createComputePipelineAsync({
            label: 'Camera Shadow Detection Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._cameraShadowBGL] }),
            compute: { module: shaderModule, entryPoint: 'main' },
        })
    }

    /**
     * Update camera shadow detection (called during execute)
     * Dispatches compute shader and starts async readback
     */
    _updateCameraShadowDetection(camera) {
        if (!this._cameraShadowPipeline || !this.directionalShadowMap) return

        // Skip if a readback is already pending (buffer is mapped)
        if (this._cameraShadowPending) return

        const { device } = this.engine
        const cameraPos = camera.position || [0, 0, 0]

        // Update uniform buffer
        const data = new Float32Array(64) // 256 bytes / 4
        data[0] = cameraPos[0]
        data[1] = cameraPos[1]
        data[2] = cameraPos[2]
        data[3] = 0 // padding

        // Copy cascade matrices
        if (this.cascadeMatrices[0]) data.set(this.cascadeMatrices[0], 4)
        if (this.cascadeMatrices[1]) data.set(this.cascadeMatrices[1], 20)
        if (this.cascadeMatrices[2]) data.set(this.cascadeMatrices[2], 36)

        device.queue.writeBuffer(this._cameraShadowUniformBuffer, 0, data)

        // Create bind group (recreated each frame as shadow map view might change)
        const bindGroup = device.createBindGroup({
            layout: this._cameraShadowBGL,
            entries: [
                { binding: 0, resource: { buffer: this._cameraShadowUniformBuffer } },
                { binding: 1, resource: this.directionalShadowMapView },
                { binding: 2, resource: this.shadowSampler },
                { binding: 3, resource: { buffer: this._cameraShadowBuffer } },
            ]
        })

        // Dispatch compute shader
        const encoder = device.createCommandEncoder({ label: 'Camera Shadow Detection' })
        const pass = encoder.beginComputePass()
        pass.setPipeline(this._cameraShadowPipeline)
        pass.setBindGroup(0, bindGroup)
        pass.dispatchWorkgroups(1)
        pass.end()

        // Copy result to readback buffer
        encoder.copyBufferToBuffer(this._cameraShadowBuffer, 0, this._cameraShadowReadBuffer, 0, 4)
        device.queue.submit([encoder.finish()])

        // Start async readback
        this._cameraShadowPending = true
        this._cameraShadowReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const data = new Float32Array(this._cameraShadowReadBuffer.getMappedRange())
            const shadowValue = data[0]
            this._cameraShadowReadBuffer.unmap()
            this._cameraShadowPending = false

            // Camera is "in shadow" if average shadow value is low
            // Threshold of 0.3 means mostly in shadow
            this._cameraInShadow = shadowValue < 0.3
        }).catch(() => {
            this._cameraShadowPending = false
        })
    }

    /**
     * Check if camera is in shadow (uses async readback result from previous frames)
     * @returns {boolean} True if camera is mostly in shadow
     */
    isCameraInShadow() {
        return this._cameraInShadow
    }
}

export { ShadowPass }
