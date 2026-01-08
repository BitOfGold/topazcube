// Particle rendering shader
// Renders billboarded quads for each alive particle with soft depth fade
// Supports lighting, shadows, point/spot lights, IBL, and emissive brightness

const PI = 3.14159265359;
const CASCADE_COUNT = 3;
const MAX_EMITTERS = 16u;
const MAX_LIGHTS = 64u;
const MAX_SPOT_SHADOWS = 8;

// Spot shadow atlas constants (must match LightingPass)
const SPOT_ATLAS_WIDTH: f32 = 2048.0;
const SPOT_ATLAS_HEIGHT: f32 = 2048.0;
const SPOT_TILE_SIZE: f32 = 512.0;
const SPOT_TILES_PER_ROW: i32 = 4;

// Particle data structure (must match particle_simulate.wgsl)
struct Particle {
    position: vec3f,
    lifetime: f32,
    velocity: vec3f,
    maxLifetime: f32,
    color: vec4f,
    size: vec2f,
    rotation: f32,    // Current rotation in radians
    flags: u32,
    lighting: vec3f,  // Pre-computed lighting (smoothed in compute shader)
    lightingPad: f32,
}

// Light structure (must match LightingPass)
struct Light {
    enabled: u32,
    position: vec3f,
    color: vec4f,
    direction: vec3f,
    geom: vec4f, // x = radius, y = inner cone, z = outer cone, w = distance fade
    shadowIndex: i32, // -1 if no shadow, 0-7 for spot shadow slot
}

// Per-emitter settings (must match ParticlePass.js)
struct EmitterRenderSettings {
    lit: f32,           // 0 = unlit, 1 = lit
    emissive: f32,      // Brightness multiplier (1 = normal, >1 = glow)
    softness: f32,
    zOffset: f32,
}

// Spot shadow matrices
struct SpotShadowMatrices {
    matrices: array<mat4x4<f32>, MAX_SPOT_SHADOWS>,
}

struct ParticleUniforms {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
    cameraPosition: vec3f,
    time: f32,
    cameraRight: vec3f,
    softness: f32,
    cameraUp: vec3f,
    zOffset: f32,
    screenSize: vec2f,
    near: f32,
    far: f32,
    blendMode: f32,       // 0 = alpha, 1 = additive
    lit: f32,             // 0 = unlit, 1 = simple lighting (global fallback)
    shadowBias: f32,
    shadowStrength: f32,
    // Lighting uniforms
    lightDir: vec3f,
    shadowMapSize: f32,
    lightColor: vec4f,
    ambientColor: vec4f,
    cascadeSizes: vec4f,  // x, y, z = cascade half-widths
    // IBL uniforms
    envParams: vec4f,     // x = diffuse level, y = mip count, z = encoding (0=equirect, 1=octahedral), w = exposure
    // Light count
    lightParams: vec4u,   // x = light count, y = unused, z = unused, w = unused
    // Fog uniforms
    fogColor: vec3f,
    fogEnabled: f32,
    fogDistances: vec3f,  // [near, mid, far]
    fogBrightResist: f32,
    fogAlphas: vec3f,     // [nearAlpha, midAlpha, farAlpha]
    fogPad1: f32,
    fogHeightFade: vec2f, // [bottomY, topY]
    fogPad2: vec2f,
}

struct CascadeMatrices {
    matrices: array<mat4x4<f32>, CASCADE_COUNT>,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) viewZ: f32,
    @location(3) linearDepth: f32,  // For frag_depth output
    @location(4) lighting: vec3f,   // Pre-computed lighting from particle
    @location(5) @interpolate(flat) emitterIdx: u32,  // For per-emitter settings
    @location(6) worldPos: vec3f,   // For fog height fade
}

@group(0) @binding(0) var<uniform> uniforms: ParticleUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var particleTexture: texture_2d<f32>;
@group(0) @binding(3) var particleSampler: sampler;
@group(0) @binding(4) var depthTexture: texture_depth_2d;
@group(0) @binding(5) var shadowMapArray: texture_depth_2d_array;
@group(0) @binding(6) var shadowSampler: sampler_comparison;
@group(0) @binding(7) var<storage, read> cascadeMatrices: CascadeMatrices;
@group(0) @binding(8) var<storage, read> emitterSettings: array<EmitterRenderSettings, MAX_EMITTERS>;
@group(0) @binding(9) var envMap: texture_2d<f32>;
@group(0) @binding(10) var envSampler: sampler;
// Point/spot lights
@group(0) @binding(11) var<storage, read> lights: array<Light, MAX_LIGHTS>;
// Spot shadow atlas
@group(0) @binding(12) var spotShadowAtlas: texture_depth_2d;
@group(0) @binding(13) var spotShadowSampler: sampler_comparison;
@group(0) @binding(14) var<storage, read> spotMatrices: SpotShadowMatrices;

// Quad vertices: 0=bottom-left, 1=bottom-right, 2=top-left, 3=top-right
// Two triangles: 0,1,2 and 1,3,2
fn getQuadVertex(vertexId: u32) -> vec2f {
    // Map 6 vertices to 4 quad corners
    // Triangle 1: 0,1,2 -> BL, BR, TL
    // Triangle 2: 3,4,5 -> BR, TR, TL
    var corners = array<vec2f, 6>(
        vec2f(-0.5, -0.5),  // 0: BL
        vec2f(0.5, -0.5),   // 1: BR
        vec2f(-0.5, 0.5),   // 2: TL
        vec2f(0.5, -0.5),   // 3: BR
        vec2f(0.5, 0.5),    // 4: TR
        vec2f(-0.5, 0.5)    // 5: TL
    );
    return corners[vertexId];
}

fn getQuadUV(vertexId: u32) -> vec2f {
    var uvs = array<vec2f, 6>(
        vec2f(0.0, 1.0),  // 0: BL
        vec2f(1.0, 1.0),  // 1: BR
        vec2f(0.0, 0.0),  // 2: TL
        vec2f(1.0, 1.0),  // 3: BR
        vec2f(1.0, 0.0),  // 4: TR
        vec2f(0.0, 0.0)   // 5: TL
    );
    return uvs[vertexId];
}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    var output: VertexOutput;

    // Get particle data
    let particle = particles[instanceIndex];

    // Get emitter index from flags (bits 8-15)
    let emitterIdx = (particle.flags >> 8u) & 0xFFu;
    output.emitterIdx = emitterIdx;

    // Check if particle is alive
    if ((particle.flags & 1u) == 0u || particle.lifetime <= 0.0) {
        // Dead particle - render degenerate triangle
        output.position = vec4f(0.0, 0.0, 0.0, 0.0);
        output.uv = vec2f(0.0, 0.0);
        output.color = vec4f(0.0, 0.0, 0.0, 0.0);
        output.viewZ = 0.0;
        output.linearDepth = 0.0;
        output.lighting = vec3f(0.0);
        output.worldPos = vec3f(0.0);
        return output;
    }

    // Check blend mode: bit 1 of flags = additive (1) or alpha (0)
    // uniforms.blendMode: 1.0 = additive, 0.0 = alpha
    let particleIsAdditive = (particle.flags & 2u) != 0u;
    let renderingAdditive = uniforms.blendMode > 0.5;
    if (particleIsAdditive != renderingAdditive) {
        // Wrong blend mode for this pass - skip
        output.position = vec4f(0.0, 0.0, 0.0, 0.0);
        output.uv = vec2f(0.0, 0.0);
        output.color = vec4f(0.0, 0.0, 0.0, 0.0);
        output.viewZ = 0.0;
        output.linearDepth = 0.0;
        output.lighting = vec3f(0.0);
        output.worldPos = vec3f(0.0);
        return output;
    }

    // Get quad vertex position and UV
    let localVertexId = vertexIndex % 6u;
    var quadPos = getQuadVertex(localVertexId);
    output.uv = getQuadUV(localVertexId);

    // Apply rotation to quad position
    let cosR = cos(particle.rotation);
    let sinR = sin(particle.rotation);
    let rotatedPos = vec2f(
        quadPos.x * cosR - quadPos.y * sinR,
        quadPos.x * sinR + quadPos.y * cosR
    );

    // Billboard: create quad facing camera
    let particleWorldPos = particle.position;

    // Scale rotated quad by particle size
    let scaledOffset = rotatedPos * particle.size;

    // Create billboard position using camera vectors
    let right = uniforms.cameraRight;
    let up = uniforms.cameraUp;
    let billboardPos = particleWorldPos + right * scaledOffset.x + up * scaledOffset.y;

    // Apply z-offset along view direction to prevent z-fighting
    let toCamera = normalize(uniforms.cameraPosition - particleWorldPos);
    let offsetPos = billboardPos + toCamera * uniforms.zOffset;

    // Transform to clip space
    let viewPos = uniforms.viewMatrix * vec4f(offsetPos, 1.0);
    output.position = uniforms.projectionMatrix * viewPos;
    output.viewZ = -viewPos.z;  // Positive depth

    // Calculate linear depth matching GBuffer format: (z - near) / (far - near)
    let z = -viewPos.z;  // View space Z (positive into screen)
    output.linearDepth = (z - uniforms.near) / (uniforms.far - uniforms.near);

    // Pass pre-computed lighting from particle (calculated in compute shader)
    output.lighting = particle.lighting;

    // Pass through particle color
    output.color = particle.color;

    // Pass world position for fog
    output.worldPos = particleWorldPos;

    return output;
}

// Equirectangular UV from direction
fn SphToUV(n: vec3f) -> vec2f {
    var uv: vec2f;
    uv.x = atan2(-n.x, n.z);
    uv.x = (uv.x + PI / 2.0) / (PI * 2.0) + PI * (28.670 / 360.0);
    uv.y = acos(n.y) / PI;
    return uv;
}

// Octahedral encoding: direction to UV
fn octEncode(n: vec3f) -> vec2f {
    var n2 = n / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n2.y < 0.0) {
        let signX = select(-1.0, 1.0, n2.x >= 0.0);
        let signZ = select(-1.0, 1.0, n2.z >= 0.0);
        n2 = vec3f(
            (1.0 - abs(n2.z)) * signX,
            n2.y,
            (1.0 - abs(n2.x)) * signZ
        );
    }
    return n2.xz * 0.5 + 0.5;
}

// Get environment UV based on encoding type
fn getEnvUV(dir: vec3f) -> vec2f {
    if (uniforms.envParams.z > 0.5) {
        return octEncode(dir);
    }
    return SphToUV(dir);
}

// Sample IBL at direction with LOD
fn getIBLSample(dir: vec3f, lod: f32) -> vec3f {
    let envRGBE = textureSampleLevel(envMap, envSampler, getEnvUV(dir), lod);
    // RGBE decode
    let envColor = envRGBE.rgb * pow(2.0, envRGBE.a * 255.0 - 128.0);
    return envColor;
}

// Squircle distance - returns distance normalized to cascade size
fn squircleDistanceXZ(offset: vec2f, size: f32) -> f32 {
    let normalized = offset / size;
    let absNorm = abs(normalized);
    return pow(pow(absNorm.x, 4.0) + pow(absNorm.y, 4.0), 0.25);
}

// Sample shadow from a specific cascade (simplified for particles)
fn sampleCascadeShadow(worldPos: vec3f, normal: vec3f, cascadeIndex: i32) -> f32 {
    let bias = uniforms.shadowBias;
    let shadowMapSize = uniforms.shadowMapSize;

    // Apply normal bias
    let biasedPos = worldPos + normal * bias * 0.5;

    // Get cascade matrix
    let lightMatrix = cascadeMatrices.matrices[cascadeIndex];

    // Transform to light space
    let lightSpacePos = lightMatrix * vec4f(biasedPos, 1.0);
    let projCoords = lightSpacePos.xyz / lightSpacePos.w;

    // Transform to [0,1] UV space
    let shadowUV = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);
    let currentDepth = projCoords.z - bias;

    // Check bounds
    let inBoundsX = shadowUV.x >= 0.0 && shadowUV.x <= 1.0;
    let inBoundsY = shadowUV.y >= 0.0 && shadowUV.y <= 1.0;
    let inBoundsZ = currentDepth >= 0.0 && currentDepth <= 1.0;

    if (!inBoundsX || !inBoundsY || !inBoundsZ) {
        return 1.0;  // Out of bounds = lit
    }

    let clampedUV = clamp(shadowUV, vec2f(0.001), vec2f(0.999));
    let clampedDepth = clamp(currentDepth, 0.001, 0.999);

    // Simple 4-tap PCF for particles (fast)
    let texelSize = 1.0 / shadowMapSize;
    var shadow = 0.0;
    shadow += textureSampleCompareLevel(shadowMapArray, shadowSampler, clampedUV + vec2f(-texelSize, 0.0), cascadeIndex, clampedDepth);
    shadow += textureSampleCompareLevel(shadowMapArray, shadowSampler, clampedUV + vec2f(texelSize, 0.0), cascadeIndex, clampedDepth);
    shadow += textureSampleCompareLevel(shadowMapArray, shadowSampler, clampedUV + vec2f(0.0, -texelSize), cascadeIndex, clampedDepth);
    shadow += textureSampleCompareLevel(shadowMapArray, shadowSampler, clampedUV + vec2f(0.0, texelSize), cascadeIndex, clampedDepth);
    shadow /= 4.0;

    return shadow;
}

// Calculate cascaded shadow for particles (simplified cascade selection)
fn calculateParticleShadow(worldPos: vec3f, normal: vec3f) -> f32 {
    let shadowStrength = uniforms.shadowStrength;

    // Calculate XZ offset from camera
    let camXZ = vec2f(uniforms.cameraPosition.x, uniforms.cameraPosition.z);
    let posXZ = vec2f(worldPos.x, worldPos.z);
    let offsetXZ = posXZ - camXZ;

    // Cascade sizes
    let cascade0Size = uniforms.cascadeSizes.x;
    let cascade1Size = uniforms.cascadeSizes.y;
    let cascade2Size = uniforms.cascadeSizes.z;

    let dist0 = squircleDistanceXZ(offsetXZ, cascade0Size);
    let dist1 = squircleDistanceXZ(offsetXZ, cascade1Size);
    let dist2 = squircleDistanceXZ(offsetXZ, cascade2Size);

    var shadow = 1.0;

    // Simple cascade selection (no blending for performance)
    if (dist0 < 0.95) {
        shadow = sampleCascadeShadow(worldPos, normal, 0);
    } else if (dist1 < 0.95) {
        shadow = sampleCascadeShadow(worldPos, normal, 1);
    } else if (dist2 < 0.95) {
        shadow = sampleCascadeShadow(worldPos, normal, 2);
    }

    // Apply shadow strength
    return mix(1.0 - shadowStrength, 1.0, shadow);
}

// Calculate spot light shadow (simplified for particles)
fn calculateSpotShadow(worldPos: vec3f, normal: vec3f, slotIndex: i32) -> f32 {
    if (slotIndex < 0 || slotIndex >= MAX_SPOT_SHADOWS) {
        return 1.0;  // No shadow
    }

    let bias = uniforms.shadowBias;
    let normalBias = bias * 2.0;  // Increased for spot lights

    // Apply normal bias
    let biasedPos = worldPos + normal * normalBias;

    // Get the light matrix from storage buffer
    let lightMatrix = spotMatrices.matrices[slotIndex];

    // Transform to light space
    let lightSpacePos = lightMatrix * vec4f(biasedPos, 1.0);

    // Perspective divide
    let w = max(abs(lightSpacePos.w), 0.0001) * sign(lightSpacePos.w + 0.0001);
    let projCoords = lightSpacePos.xyz / w;

    // Check if outside frustum
    if (projCoords.z < 0.0 || projCoords.z > 1.0 ||
        abs(projCoords.x) > 1.0 || abs(projCoords.y) > 1.0) {
        return 1.0;  // Outside shadow frustum
    }

    // Calculate tile position in atlas
    let col = slotIndex % SPOT_TILES_PER_ROW;
    let row = slotIndex / SPOT_TILES_PER_ROW;

    // Transform to [0,1] UV within tile
    let localUV = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);

    // Transform to atlas UV
    let tileOffset = vec2f(f32(col), f32(row)) * SPOT_TILE_SIZE;
    let atlasUV = (tileOffset + localUV * SPOT_TILE_SIZE) / vec2f(SPOT_ATLAS_WIDTH, SPOT_ATLAS_HEIGHT);

    // Sample shadow with simple 4-tap PCF
    let texelSize = 1.0 / SPOT_TILE_SIZE;
    let currentDepth = clamp(projCoords.z - bias * 3.0, 0.001, 0.999);

    var shadowSample = 0.0;
    shadowSample += textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, atlasUV + vec2f(-texelSize, 0.0), currentDepth);
    shadowSample += textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, atlasUV + vec2f(texelSize, 0.0), currentDepth);
    shadowSample += textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, atlasUV + vec2f(0.0, -texelSize), currentDepth);
    shadowSample += textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, atlasUV + vec2f(0.0, texelSize), currentDepth);
    shadowSample /= 4.0;

    return shadowSample;
}

// Apply pre-computed lighting to particle color
// Lighting is calculated per-particle in the compute shader with temporal smoothing
fn applyLighting(baseColor: vec3f, lighting: vec3f) -> vec3f {
    // Just multiply base color by pre-computed lighting
    // Lighting already includes ambient, shadows, point/spot lights, and emissive
    return baseColor * lighting;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

// Calculate soft particle fade based on depth difference
fn calcSoftFade(fragPos: vec4f, particleLinearDepth: f32) -> f32 {
    if (uniforms.softness <= 0.0) {
        return 1.0;
    }

    // Get screen coordinates
    let screenPos = vec2i(fragPos.xy);
    let screenSize = vec2i(uniforms.screenSize);

    // Bounds check
    if (screenPos.x < 0 || screenPos.x >= screenSize.x ||
        screenPos.y < 0 || screenPos.y >= screenSize.y) {
        return 1.0;
    }

    // Sample scene depth (linear depth in 0-1 range from GBuffer)
    let sceneDepthNorm = textureLoad(depthTexture, screenPos, 0);

    // If depth is 0 (no valid data), skip soft fade
    if (sceneDepthNorm <= 0.0) {
        return 1.0;
    }

    // Convert both to world units for comparison
    let sceneDepth = uniforms.near + sceneDepthNorm * (uniforms.far - uniforms.near);
    let particleDepth = uniforms.near + particleLinearDepth * (uniforms.far - uniforms.near);

    // Fade based on depth difference (in world units)
    let depthDiff = sceneDepth - particleDepth;
    return saturate(depthDiff / uniforms.softness);
}

// Fragment for alpha blend mode
@fragment
fn fragmentMainAlpha(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    // Sample texture directly
    let texColor = textureSample(particleTexture, particleSampler, input.uv);

    // Combine texture with particle color
    var alpha = texColor.a * input.color.a;

    // Apply soft particle fade
    alpha *= calcSoftFade(input.position, input.linearDepth);

    // Discard nearly transparent pixels
    if (alpha < 0.001) {
        discard;
    }

    // Apply pre-computed lighting from particle
    let baseColor = texColor.rgb * input.color.rgb;
    let litColor = applyLighting(baseColor, input.lighting);

    // Fog is applied as post-process to the combined HDR buffer
    output.color = vec4f(litColor, alpha);
    output.depth = input.linearDepth;
    return output;
}

// Fragment for additive mode
@fragment
fn fragmentMainAdditive(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    // Sample texture directly
    let texColor = textureSample(particleTexture, particleSampler, input.uv);

    // Combine texture with particle color
    var alpha = texColor.a * input.color.a;

    // Apply soft particle fade
    alpha *= calcSoftFade(input.position, input.linearDepth);

    // Discard nearly transparent pixels
    if (alpha < 0.001) {
        discard;
    }

    // Apply pre-computed lighting from particle
    let baseColor = texColor.rgb * input.color.rgb;
    let litColor = applyLighting(baseColor, input.lighting);

    // Fog is applied as post-process to the combined HDR buffer
    // Premultiply for additive blending
    let rgb = litColor * alpha;
    output.color = vec4f(rgb, alpha);
    output.depth = input.linearDepth;
    return output;
}
