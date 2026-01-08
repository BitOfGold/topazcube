const PI = 3.14159;
const MAX_LIGHTS = 768;
const CASCADE_COUNT = 3;
const MAX_LIGHTS_PER_TILE = 256;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const MAX_SPOT_SHADOWS = 16;

struct Light {
    enabled: u32,
    position: vec3f,
    color: vec4f,
    direction: vec3f,
    geom: vec4f, // x = radius, y = inner cone, z = outer cone
    shadowIndex: i32, // -1 if no shadow, 0-7 for spot shadow slot
}

struct Uniforms {
    inverseViewProjection: mat4x4<f32>,
    inverseProjection: mat4x4<f32>,  // For logarithmic depth reconstruction
    inverseView: mat4x4<f32>,        // For logarithmic depth reconstruction
    cameraPosition: vec3f,
    canvasSize: vec2f,
    lightDir: vec3f,
    lightColor: vec4f,
    ambientColor: vec4f,
    environmentParams: vec4f, // x = diffuse level, y = specular level, z = env mip count, a = exposure
    shadowParams: vec4f, // x = bias, y = normalBias, z = shadow strength, w = shadow map size
    cascadeSizes: vec4f, // x, y, z = cascade half-widths in meters
    tileParams: vec4u, // x = tile size, y = tile count X, z = max lights per tile, w = actual light count
    noiseParams: vec4f, // x = noise texture size, y = noise offset X, z = noise offset Y, w = env encoding (0=equirect, 1=octahedral)
    cameraParams: vec4f, // x = near, y = far, z = reflection mode (flip env Y), w = direct specular multiplier
    specularBoost: vec4f, // x = intensity, y = roughness cutoff, z = unused, w = unused
}

// Hard-coded spot shadow parameters (to avoid uniform buffer alignment issues)
const SPOT_ATLAS_WIDTH: f32 = 2048.0;
const SPOT_ATLAS_HEIGHT: f32 = 2048.0;
const SPOT_TILE_SIZE: f32 = 512.0;
const SPOT_TILES_PER_ROW: i32 = 4;
const SPOT_FADE_START: f32 = 25.0;
const SPOT_MAX_DISTANCE: f32 = 30.0;
const SPOT_MIN_SHADOW: f32 = 0.5;

// Storage buffer for spotlight matrices (avoids uniform buffer size limits)
struct SpotShadowMatrices {
    matrices: array<mat4x4<f32>, MAX_SPOT_SHADOWS>,
}

// Storage buffer for cascade matrices
struct CascadeMatrices {
    matrices: array<mat4x4<f32>, CASCADE_COUNT>,
}


@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(12) var<storage, read> spotMatrices: SpotShadowMatrices;
@group(0) @binding(13) var<storage, read> cascadeMatrices: CascadeMatrices;
@group(0) @binding(14) var<storage, read> tileLightIndices: array<u32>;
@group(0) @binding(15) var<storage, read> lights: array<Light, MAX_LIGHTS>;
@group(0) @binding(1) var gAlbedo: texture_2d<f32>;
@group(0) @binding(2) var gNormal: texture_2d<f32>;
@group(0) @binding(3) var gArm: texture_2d<f32>;
@group(0) @binding(4) var gEmission: texture_2d<f32>;
@group(0) @binding(5) var gDepth: texture_depth_2d;
@group(0) @binding(6) var env: texture_2d<f32>;
@group(0) @binding(7) var envSampler: sampler;
@group(0) @binding(8) var shadowMapArray: texture_depth_2d_array;
@group(0) @binding(9) var shadowSampler: sampler_comparison;
@group(0) @binding(10) var spotShadowAtlas: texture_depth_2d;
@group(0) @binding(11) var spotShadowSampler: sampler_comparison;
@group(0) @binding(16) var noiseTexture: texture_2d<f32>;
@group(0) @binding(17) var noiseSampler: sampler;
@group(0) @binding(18) var aoTexture: texture_2d<f32>;

fn SphToUV(n: vec3<f32>) -> vec2<f32> {
    var uv: vec2<f32>;

    uv.x = atan2(-n.x, n.z);
    uv.x = (uv.x + PI / 2.0) / (PI * 2.0) + PI * (28.670 / 360.0);

    uv.y = acos(n.y) / PI;

    return uv;
}

// Octahedral encoding: direction to UV
// Maps sphere to square: yaw around edges, pitch from center (top) to edges (bottom)
fn octEncode(n: vec3<f32>) -> vec2<f32> {
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

// Octahedral decoding: UV to direction
fn octDecode(uv: vec2<f32>) -> vec3<f32> {
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

// Sample noise texture at screen position with optional animation offset
// Uses textureLoad to avoid uniform control flow issues
fn sampleNoise(screenPos: vec2f) -> f32 {
    let noiseSize = i32(uniforms.noiseParams.x);
    let noiseOffsetX = i32(uniforms.noiseParams.y * f32(noiseSize));
    let noiseOffsetY = i32(uniforms.noiseParams.z * f32(noiseSize));

    // Tile the noise across screen space using modulo
    let texCoord = vec2i(
        (i32(screenPos.x) + noiseOffsetX) % noiseSize,
        (i32(screenPos.y) + noiseOffsetY) % noiseSize
    );
    return textureLoad(noiseTexture, texCoord, 0).r;
}

// Get 2D jitter offset from noise (-1 to 1 range)
// Uses textureLoad to avoid uniform control flow issues
fn getNoiseJitter(screenPos: vec2f) -> vec2f {
    let noiseSize = i32(uniforms.noiseParams.x);
    let noiseOffsetX = i32(uniforms.noiseParams.y * f32(noiseSize));
    let noiseOffsetY = i32(uniforms.noiseParams.z * f32(noiseSize));

    // Load at two offset positions for X and Y
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

// Get 4-channel noise for IBL multisampling
fn getNoise4(screenPos: vec2f) -> vec4f {
    let noiseSize = i32(uniforms.noiseParams.x);
    let noiseOffsetX = i32(uniforms.noiseParams.y * f32(noiseSize));
    let noiseOffsetY = i32(uniforms.noiseParams.z * f32(noiseSize));

    let texCoord = vec2i(
        (i32(screenPos.x) + noiseOffsetX) % noiseSize,
        (i32(screenPos.y) + noiseOffsetY) % noiseSize
    );
    return textureLoad(noiseTexture, texCoord, 0);
}

// Vogel disk sample pattern - gives good 2D distribution
fn vogelDiskSample(sampleIndex: i32, numSamples: i32, rotation: f32) -> vec2f {
    let goldenAngle = 2.399963229728653;  // pi * (3 - sqrt(5))
    let r = sqrt((f32(sampleIndex) + 0.5) / f32(numSamples));
    let theta = f32(sampleIndex) * goldenAngle + rotation;
    return vec2f(r * cos(theta), r * sin(theta));
}

// Build orthonormal basis around a direction (for cone sampling)
fn buildOrthonormalBasis(n: vec3f) -> mat3x3<f32> {
    // Choose a vector not parallel to n
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return mat3x3<f32>(tangent, bitangent, n);
}

fn clampedDot(x: vec3<f32>, y: vec3<f32>) -> f32 {
    return clamp(dot(x, y), 0.0, 1.0);
}

fn F_Schlick(f0: vec3<f32>, f90: vec3<f32>, VdotH: f32) -> vec3<f32> {
    return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

fn V_GGX(NdotL: f32, NdotV: f32, alphaRoughness: f32) -> f32 {
    let alphaRoughnessSq = alphaRoughness * alphaRoughness;

    let GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
    let GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

    let GGX = GGXV + GGXL;
    if (GGX > 0.0) {
        return 0.5 / GGX;
    }
    return 0.0;
}

fn D_GGX(NdotH: f32, alphaRoughness: f32) -> f32 {
    let alphaRoughnessSq = alphaRoughness * alphaRoughness;
    let f = (NdotH * NdotH) * (alphaRoughnessSq - 1.0) + 1.0;
    return alphaRoughnessSq / (3.14159 * f * f);
}

fn BRDF_lambertian(f0: vec3<f32>, f90: vec3<f32>, diffuseColor: vec3<f32>, specularWeight: f32, VdotH: f32) -> vec3<f32> {
    return (1.0 - specularWeight * F_Schlick(f0, f90, VdotH)) * (diffuseColor / 3.14159);
}

fn BRDF_specularGGX(f0: vec3<f32>, f90: vec3<f32>, alphaRoughness: f32, specularWeight: f32, VdotH: f32, NdotL: f32, NdotV: f32, NdotH: f32) -> vec3<f32> {
    let F = F_Schlick(f0, f90, VdotH);
    let Vis = V_GGX(NdotL, NdotV, alphaRoughness);
    let D = D_GGX(NdotH, alphaRoughness);

    return specularWeight * F * Vis * D;
}

// Get environment UV based on encoding type (0=equirectangular, 1=octahedral)
fn getEnvUV(dir: vec3<f32>) -> vec2<f32> {
    if (uniforms.noiseParams.w > 0.5) {
        return octEncode(dir);
    }
    return SphToUV(dir);
}

fn getIBLSample(reflection: vec3<f32>, lod: f32) -> vec3<f32> {
    let envRGBE = textureSampleLevel(env, envSampler, getEnvUV(reflection), lod);
    // Both equirectangular and octahedral textures are RGBE encoded
    let envColor = envRGBE.rgb * pow(2.0, envRGBE.a * 255.0 - 128.0);
    return envColor;
}

// IBL sample with noise-jittered UV to reduce banding
fn getIBLSampleJittered(reflection: vec3<f32>, lod: f32, screenPos: vec2f) -> vec3<f32> {
    let baseUV = getEnvUV(reflection);

    // Get 2D jitter from blue noise
    let jitter = getNoiseJitter(screenPos);

    // Jitter by 0.5 texels of the environment texture
    let envSize = vec2f(textureDimensions(env, 0));
    let jitterScale = 16 / envSize;
    let jitteredUV = baseUV + jitter * jitterScale;

    let envRGBE = textureSampleLevel(env, envSampler, jitteredUV, lod);
    // Both equirectangular and octahedral textures are RGBE encoded
    let envColor = envRGBE.rgb * pow(2.0, envRGBE.a * 255.0 - 128.0);
    return envColor;
}

// IBL specular with multisampled cone jitter based on roughness
// Uses Vogel disk sampling with blue noise rotation (same technique as planar reflections)
fn getIBLRadianceGGX(n: vec3<f32>, v: vec3<f32>, roughness: f32, F0: vec3<f32>, specularWeight: f32, screenPos: vec2f) -> vec3<f32> {
    let NdotV = clampedDot(n, v);
    let lod = roughness * (uniforms.environmentParams.z - 1);
    let reflection = normalize(reflect(-v, n));

    // Fresnel calculation
    let Fr = max(vec3<f32>(1.0 - roughness), F0) - F0;
    let k_S = F0 + Fr * pow(1.0 - NdotV, 5.0);

    // For very smooth surfaces (roughness < 0.1), use single sample
    if (roughness < 0.1) {
        let specularSample = getIBLSampleJittered(reflection, lod, screenPos);
        return specularWeight * specularSample * k_S;
    }

    // Multisample with roughness-based cone jitter
    let numSamples = 4;

    // Get blue noise for rotation and radius jitter
    let noise = getNoise4(screenPos);
    let rotationAngle = noise.r * 6.283185307;  // 0 to 2*PI rotation
    let radiusJitter = noise.g;

    // Build orthonormal basis around reflection direction
    let basis = buildOrthonormalBasis(reflection);

    // Cone angle based on roughness (roughness^2 scaling, max ~30 degrees)
    // Similar to planar reflection blur scaling
    let maxConeAngle = 0.5;  // ~30 degrees in radians
    let coneAngle = roughness * roughness * maxConeAngle;

    var colorSum = vec3f(0.0);

    for (var i = 0; i < numSamples; i++) {
        // Get Vogel disk sample position (unit disk, rotated by blue noise)
        let diskSample = vogelDiskSample(i, numSamples, rotationAngle);

        // Apply radius jitter per-sample
        let sampleRadius = mix(0.5, 1.0, fract(radiusJitter + f32(i) * 0.618033988749895));

        // Convert disk position to cone offset
        let offset2D = diskSample * sampleRadius * coneAngle;

        // Perturb reflection direction within cone
        let perturbedDir = normalize(
            basis[2] +  // reflection direction (z-axis of basis)
            basis[0] * offset2D.x +  // tangent
            basis[1] * offset2D.y    // bitangent
        );

        // Sample environment at perturbed direction
        let sampleColor = getIBLSample(perturbedDir, lod);
        colorSum += sampleColor;
    }

    let specularLight = colorSum / f32(numSamples);
    return specularWeight * specularLight * k_S;
}

// Sample shadow from a specific cascade with blue noise jittered PCF
// Returns: x = shadow value (0=shadow, 1=lit), y = 1 if in bounds, 0 if out of bounds
fn sampleCascadeShadowWithBounds(worldPos: vec3<f32>, normal: vec3<f32>, cascadeIndex: i32, screenPos: vec2f) -> vec2f {
    let baseBias = uniforms.shadowParams.x;
    let baseNormalBias = uniforms.shadowParams.y;
    let shadowMapSize = uniforms.shadowParams.w;

    // Slightly increase bias for cascade 2 (larger coverage, lower resolution)
    var biasScale = 1.0;
    if (cascadeIndex == 2) {
        biasScale = 1.5;
    }
    let bias = baseBias * biasScale;
    let normalBias = baseNormalBias * biasScale;

    // Apply normal bias
    let biasedPos = worldPos + normal * normalBias;

    // Get cascade matrix from storage buffer
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
    let inBounds = inBoundsX && inBoundsY && inBoundsZ;

    if (!inBounds) {
        return vec2f(1.0, 0.0); // Out of bounds
    }

    let clampedUV = clamp(shadowUV, vec2f(0.001), vec2f(0.999));
    let clampedDepth = clamp(currentDepth, 0.001, 0.999);

    // Get blue noise jitter for this pixel
    let jitter = getNoiseJitter(screenPos);

    // Blue noise jittered PCF - rotated disk pattern
    // Use noise to rotate and offset sample positions
    let texelSize = 1.0 / shadowMapSize;
    // More blur for closer cascades, less for distant ones
    var cascadeBlurScale = 1.0;  // Default for cascade 2

    if (cascadeIndex == 0) {
        cascadeBlurScale = 2.0;
    } else if (cascadeIndex == 1) {
        cascadeBlurScale = 1.7;
    }

    let filterRadius = texelSize * cascadeBlurScale;

    // Create rotation matrix from noise
    let angle = jitter.x * PI;
    let cosA = cos(angle);
    let sinA = sin(angle);

    // Poisson disk sample offsets (pre-computed, 8 samples)
    let offsets = array<vec2f, 8>(
        vec2f(-0.94201624, -0.39906216),
        vec2f(0.94558609, -0.76890725),
        vec2f(-0.094184101, -0.92938870),
        vec2f(0.34495938, 0.29387760),
        vec2f(-0.91588581, 0.45771432),
        vec2f(-0.81544232, -0.87912464),
        vec2f(-0.38277543, 0.27676845),
        vec2f(0.97484398, 0.75648379)
    );

    var shadow = 0.0;
    for (var i = 0; i < 8; i++) {
        // Rotate offset by noise angle
        let offset = offsets[i];
        let rotatedOffset = vec2f(
            offset.x * cosA - offset.y * sinA,
            offset.x * sinA + offset.y * cosA
        ) * filterRadius;

        // Add subtle jitter offset
        let jitteredOffset = rotatedOffset + jitter * texelSize * 0.15;

        shadow += textureSampleCompareLevel(shadowMapArray, shadowSampler, clampedUV + jitteredOffset, cascadeIndex, clampedDepth);
    }
    shadow /= 8.0;

    return vec2f(shadow, 1.0); // In bounds
}

// Wrapper for simple usage
fn sampleCascadeShadow(worldPos: vec3<f32>, normal: vec3<f32>, cascadeIndex: i32, screenPos: vec2f) -> f32 {
    return sampleCascadeShadowWithBounds(worldPos, normal, cascadeIndex, screenPos).x;
}

// Squircle distance function for smooth fade at edges
// Returns 0-1 where 1 is at the edge of the squircle
fn squircleDistance(uv: vec2<f32>, power: f32) -> f32 {
    // Squircle: |x|^n + |y|^n = 1 where n > 2 gives squircle shape
    let centered = (uv - 0.5) * 2.0; // Convert to -1 to 1
    let absCentered = abs(centered);
    return pow(pow(absCentered.x, power) + pow(absCentered.y, power), 1.0 / power);
}

// Squircle distance - returns distance normalized to cascade size
// Power 4 gives nice rounded corners
fn squircleDistanceXZ(offset: vec2f, size: f32) -> f32 {
    let normalized = offset / size;
    let absNorm = abs(normalized);
    return pow(pow(absNorm.x, 4.0) + pow(absNorm.y, 4.0), 0.25);
}

// Calculate cascaded shadow with smooth blending between cascades
fn calculateShadow(worldPos: vec3<f32>, normal: vec3<f32>, screenPos: vec2f) -> f32 {
    let shadowStrength = uniforms.shadowParams.z;

    // Calculate XZ offset from camera to world position
    let camXZ = vec2f(uniforms.cameraPosition.x, uniforms.cameraPosition.z);
    let posXZ = vec2f(worldPos.x, worldPos.z);
    let offsetXZ = posXZ - camXZ;

    // Cascade sizes from uniforms
    let cascade0Size = uniforms.cascadeSizes.x;
    let cascade1Size = uniforms.cascadeSizes.y;
    let cascade2Size = uniforms.cascadeSizes.z;

    // Use squircle distance for cascade selection - rounded square shape
    let dist0 = squircleDistanceXZ(offsetXZ, cascade0Size);
    let dist1 = squircleDistanceXZ(offsetXZ, cascade1Size);
    let dist2 = squircleDistanceXZ(offsetXZ, cascade2Size);

    // Determine which cascade(s) to sample based on squircle distance
    var shadow = 1.0;
    var cascadeUsed = -1;
    var blendFactor = 0.0;

    // Sample cascades with bounds checking (pass screen position for blue noise)
    let sample0 = sampleCascadeShadowWithBounds(worldPos, normal, 0, screenPos);
    let sample1 = sampleCascadeShadowWithBounds(worldPos, normal, 1, screenPos);
    let sample2 = sampleCascadeShadowWithBounds(worldPos, normal, 2, screenPos);

    let inBounds0 = sample0.y > 0.5;
    let inBounds1 = sample1.y > 0.5;
    let inBounds2 = sample2.y > 0.5;

    // Cascade 0: highest detail, smallest area
    if (dist0 < 1.0) {
        if (inBounds0) {
            // Blend towards cascade 1 at 90-100% of cascade 0 size
            let blend0to1 = smoothstep(0.9, 1.0, dist0);
            if (blend0to1 > 0.0 && inBounds1) {
                shadow = mix(sample0.x, sample1.x, blend0to1);
            } else {
                shadow = sample0.x;
            }
            cascadeUsed = 0;
        } else if (inBounds1) {
            // Cascade 0 out of bounds, use cascade 1 100%
            shadow = sample1.x;
            cascadeUsed = 1;
        } else if (inBounds2) {
            // Cascade 0 and 1 out of bounds, use cascade 2 100%
            shadow = sample2.x;
            cascadeUsed = 2;
        } else {
            return mix(1.0 - shadowStrength, 1.0, 0.5);
        }
    }
    // Cascade 1: medium detail
    else if (dist1 < 1.0) {
        if (inBounds1) {
            // Blend towards cascade 2 at 90-100% of cascade 1 size
            let blend1to2 = smoothstep(0.9, 1.0, dist1);
            if (blend1to2 > 0.0 && inBounds2) {
                shadow = mix(sample1.x, sample2.x, blend1to2);
            } else {
                shadow = sample1.x;
            }
            cascadeUsed = 1;
        } else if (inBounds2) {
            // Cascade 1 out of bounds, use cascade 2 100%
            shadow = sample2.x;
            cascadeUsed = 2;
        } else {
            return mix(1.0 - shadowStrength, 1.0, 0.5);
        }
    }
    // Cascade 2: lowest detail, largest area
    else if (dist2 < 1.0) {
        if (inBounds2) {
            shadow = sample2.x;
            cascadeUsed = 2;

            // Gradual fade for cascade 2:
            // 50-95%: fade shadow towards half-lit
            // 95-100%: fade from shadow to half-lit (0.5)
            let lightnessFade = smoothstep(0.5, 0.95, dist2);
            shadow = mix(shadow, 0.5, lightnessFade);

            let edgeFade = smoothstep(0.95, 1.0, dist2);
            shadow = mix(shadow, 0.5, edgeFade);
        } else {
            return mix(1.0 - shadowStrength, 1.0, 0.5);
        }
    }
    // Beyond all cascades: half-lit
    else {
        return mix(1.0 - shadowStrength, 1.0, 0.5);
    }

    // Apply shadow strength
    let shadowResult = mix(1.0 - shadowStrength, 1.0, shadow);
    return shadowResult;
}

// Calculate spot light shadow from atlas using storage buffer for matrices
// lightPos: spotlight position (for calculating distance to camera)
// slotIndex: shadow atlas slot (-1 if no shadow)
fn calculateSpotShadow(worldPos: vec3<f32>, normal: vec3<f32>, lightPos: vec3<f32>, slotIndex: i32, screenPos: vec2f) -> f32 {
    // Use hard-coded constants to avoid uniform buffer alignment issues
    let fadeStart = SPOT_FADE_START;
    let maxDist = SPOT_MAX_DISTANCE;
    let minShadow = SPOT_MIN_SHADOW;
    let bias = uniforms.shadowParams.x;
    let normalBias = uniforms.shadowParams.y;
    let shadowStrength = uniforms.shadowParams.z;

    // Calculate distance from spotlight to camera (not light to pixel!)
    let lightToCam = lightPos - uniforms.cameraPosition;
    let lightDistance = length(lightToCam);

    // Get atlas parameters from constants
    let atlasWidth = SPOT_ATLAS_WIDTH;
    let atlasHeight = SPOT_ATLAS_HEIGHT;
    let tileSize = SPOT_TILE_SIZE;
    let tilesPerRow = SPOT_TILES_PER_ROW;

    // Check validity
    let hasValidSlot = f32(slotIndex >= 0 && slotIndex < MAX_SPOT_SHADOWS);

    // Clamp slot index to valid range for matrix access
    let safeSlot = clamp(slotIndex, 0, MAX_SPOT_SHADOWS - 1);

    // Calculate tile position in atlas
    let col = safeSlot % tilesPerRow;
    let row = safeSlot / tilesPerRow;

    // Apply normal bias (increased for spot lights due to perspective projection)
    let spotNormalBias = normalBias * 2.0;
    let biasedPos = worldPos + normal * spotNormalBias;

    // Get the light matrix from storage buffer
    let lightMatrix = spotMatrices.matrices[safeSlot];

    // Transform to light space
    let lightSpacePos = lightMatrix * vec4f(biasedPos, 1.0);

    // Perspective divide (avoid divide by zero)
    let w = max(abs(lightSpacePos.w), 0.0001) * sign(lightSpacePos.w + 0.0001);
    let projCoords = lightSpacePos.xyz / w;

    // Calculate edge fade for soft frustum boundaries
    // Use distance from center in clip space (max of abs x,y)
    let edgeDist = max(abs(projCoords.x), abs(projCoords.y));
    // Fade from 0.8 to 1.0 in clip space (soft edge)
    let edgeFade = 1.0 - smoothstep(0.8, 1.0, edgeDist);
    // Also fade based on depth (behind camera or too far)
    let depthFade = smoothstep(-0.1, 0.1, projCoords.z) * (1.0 - smoothstep(0.9, 1.0, projCoords.z));
    let frustumFade = edgeFade * depthFade;

    // Transform to [0,1] UV space within the tile
    let tileUV = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);

    // Calculate UV in atlas (normalized to atlas size)
    let tileOffsetX = f32(col) * tileSize;
    let tileOffsetY = f32(row) * tileSize;
    let atlasUV = vec2f(
        (tileOffsetX + tileUV.x * tileSize) / atlasWidth,
        (tileOffsetY + tileUV.y * tileSize) / atlasHeight
    );

    // Blue noise jittered PCF for spot shadows
    let jitter = getNoiseJitter(screenPos);
    let texelSize = 1.0 / tileSize;
    let filterRadius = texelSize * 0.5;  // Subtle filter for spot shadows

    // Create rotation from noise
    let angle = jitter.x * PI;
    let cosA = cos(angle);
    let sinA = sin(angle);

    // 4-sample rotated PCF for spot shadows (less samples for performance)
    let offsets = array<vec2f, 4>(
        vec2f(-0.7, -0.7),
        vec2f(0.7, -0.7),
        vec2f(-0.7, 0.7),
        vec2f(0.7, 0.7)
    );

    // Current depth with bias (increased for spot lights)
    let spotBias = bias * 3.0;
    let currentDepth = clamp(projCoords.z - spotBias, 0.001, 0.999);

    var shadowSample = 0.0;
    for (var i = 0; i < 4; i++) {
        let offset = offsets[i];
        let rotatedOffset = vec2f(
            offset.x * cosA - offset.y * sinA,
            offset.x * sinA + offset.y * cosA
        ) * filterRadius;

        let sampleUV = clamp(atlasUV + rotatedOffset, vec2f(0.001), vec2f(0.999));
        shadowSample += textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, sampleUV, currentDepth);
    }
    shadowSample /= 4.0;

    // Apply shadow strength
    let shadowWithStrength = mix(1.0 - shadowStrength, 1.0, shadowSample);

    // Apply frustum edge fade (fade to 1.0 = no shadow at edges)
    let edgeFadedShadow = mix(1.0, shadowWithStrength, frustumFade);

    // Apply distance-based fade
    let fadeT = clamp((lightDistance - fadeStart) / (maxDist - fadeStart), 0.0, 1.0);
    let fadedShadow = mix(edgeFadedShadow, minShadow, fadeT);

    // Calculate result for no-slot case (distance-based constant shadow)
    let noSlotFadeT = clamp((lightDistance - fadeStart) / (maxDist - fadeStart), 0.0, 1.0);
    let noSlotResult = mix(1.0, minShadow, noSlotFadeT);

    // Select final result based on slot validity (frustum fade already applied)
    let shadowResult = select(noSlotResult, fadedShadow, hasValidSlot > 0.5);

    // Debug visualization stored in global for fragment shader to use
    // Using projCoords.z as a debug value

    return shadowResult;
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
var output : VertexOutput;
    let x = f32(vertexIndex & 1u) * 4.0 - 1.0;
    let y = f32(vertexIndex >> 1u) * 4.0 - 1.0;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let cs = uniforms.canvasSize;
    let fuv = vec2i(floor(input.uv * cs));
    let baseColor = textureLoad(gAlbedo, fuv, 0);
    let normal = normalize(textureLoad(gNormal, fuv, 0).xyz);
    let arm = textureLoad(gArm, fuv, 0);
    let emissive = textureLoad(gEmission, fuv, 0).rgb;
    let depth = textureLoad(gDepth, fuv, 0);

    // Unpack ARM values (Ambient, Roughness, Metallic, per-material SpecularBoost)
    var ao = arm.r;
    var roughness = arm.g;
    var metallic = arm.b;
    let materialSpecularBoost = arm.a;  // Per-material boost (0-1)

    // Sample SSAO texture - this affects only ambient/environment lighting
    let ssao = textureLoad(aoTexture, fuv, 0).r;

    var specular = vec3<f32>(0.0);
    var diffuse = vec3<f32>(0.0);

    let ior = 1.5;
    var f0 = vec3<f32>(0.04);
    let f90 = vec3<f32>(1.0);
    let specularWeight = 1.0;

    // Reconstruct position from linear depth and UV
    // Linear depth: depth = (z - near) / (far - near), so z = near + depth * (far - near)
    let near = uniforms.cameraParams.x;
    let far = uniforms.cameraParams.y;
    let linearDepth = near + depth * (far - near);

    // Get view-space ray direction from UV
    var iuv = input.uv;
    iuv.y = 1.0 - iuv.y;
    let ndc = vec4f(iuv * 2.0 - 1.0, 0.0, 1.0);
    let viewRay = uniforms.inverseProjection * ndc;
    let rayDir = normalize(viewRay.xyz / viewRay.w);

    // Reconstruct view-space position using linear depth
    let viewPos = rayDir * (linearDepth / -rayDir.z);

    // Transform to world space
    let worldPos4 = uniforms.inverseView * vec4f(viewPos, 1.0);
    let position = worldPos4.xyz;
    let toCamera = uniforms.cameraPosition.xyz - position;
    let v = normalize(toCamera);

    let n = normal;

    f0 = mix(f0, baseColor.rgb, metallic);
    roughness = max(roughness, 0.04);
    let alphaRoughness = roughness * roughness;

    let c_diff = mix(baseColor.rgb, vec3<f32>(0.0), metallic);

    // Combine material AO with SSAO for ambient/environment lighting only
    let combinedAO = ao * ssao;

    diffuse += combinedAO * uniforms.ambientColor.rgb * uniforms.ambientColor.a * BRDF_lambertian(f0, f90, c_diff, specularWeight, 1.0);

    // Use jittered IBL sampling to reduce banding
    let screenPos = input.position.xy;
    let iblSample = getIBLSampleJittered(n, uniforms.environmentParams.z - 2, screenPos);
    diffuse += combinedAO * uniforms.environmentParams.x * iblSample.rgb * BRDF_lambertian(f0, f90, c_diff, specularWeight, 1.0);

    let reflection = normalize(reflect(-v, n));
    specular += combinedAO * uniforms.environmentParams.y * getIBLRadianceGGX(n, v, roughness, f0, specularWeight, screenPos);

    // calculate lighting

    let onormal = n;
    let l = normalize(uniforms.lightDir.xyz);
    let h = normalize(l + v);
    let NdotV = clampedDot(n, v);
    let ONdotV = clampedDot(onormal, v);
    let ONdotL = clampedDot(onormal, l);
    let NdotLF = dot(n, l);
    let NdotL = clamp(NdotLF, 0.0, 1.0);
    let NdotLF_adjusted = NdotLF * 0.5 + 0.5;
    let NdotH = clampedDot(n, h);
    let LdotH = clampedDot(l, h);
    let VdotH = clampedDot(v, h);

    let intensity = uniforms.lightColor.a;
    // Calculate shadow outside of non-uniform control flow
    let shadow = calculateShadow(position, n, screenPos);
    let fallOff = smoothstep(-0.1, 0.3, NdotLF);  // Soft gradual falloff

    // Apply lighting with shadow (using select to avoid non-uniform branching)
    let lightActive = select(0.0, 1.0, (NdotL > 0.0 || NdotV > 0.0) && intensity > 0.0);
    diffuse += lightActive * intensity * uniforms.lightColor.rgb * fallOff * shadow * BRDF_lambertian(f0, f90, c_diff, specularWeight, VdotH);
    specular += lightActive * intensity * uniforms.lightColor.rgb * fallOff * shadow * BRDF_specularGGX(f0, f90, alphaRoughness, specularWeight, VdotH, fallOff, NdotV, NdotH) * uniforms.cameraParams.w;

    // Tiled lighting: get tile index from pixel position
    let tileSize = uniforms.tileParams.x;
    let tileCountX = uniforms.tileParams.y;
    let maxLightsPerTile = uniforms.tileParams.z;
    let actualLightCount = uniforms.tileParams.w;

    // Calculate tileCountY from canvas size
    let tileCountY = (u32(uniforms.canvasSize.y) + tileSize - 1u) / tileSize;

    let tileX = u32(floor(input.uv.x * uniforms.canvasSize.x)) / tileSize;
    // Flip Y because UV.y increases downward but NDC Y (used in compute shader) increases upward
    let rawTileY = u32(floor(input.uv.y * uniforms.canvasSize.y)) / tileSize;
    let tileY = tileCountY - 1u - rawTileY;
    let tileIndex = tileY * tileCountX + tileX;
    let tileDataOffset = tileIndex * (maxLightsPerTile + 1u);

    // Read number of lights affecting this tile
    let tileLightCount = min(tileLightIndices[tileDataOffset], maxLightsPerTile);

    // Process only lights that affect this tile
    for (var i = 0u; i < tileLightCount; i++) {
        let lightIndex = tileLightIndices[tileDataOffset + 1u + i];
        if (lightIndex >= actualLightCount) {
            continue;
        }

        let light = lights[lightIndex];
        let isEnabled = f32(light.enabled);

        // Calculate light direction and distance
        let lightVec = light.position - position;
        let dist = length(lightVec);
        let lightDir = normalize(lightVec);

        // Distance fade from CPU culling (stored in geom.w) - smooth fade to avoid popping
        let distanceFade = light.geom.w;

        // Attenuation (inverse square with radius falloff)
        let radius = max(light.geom.x, 0.001);
        let attenuation = max(0.0, 1.0 - dist / radius);
        let attenuationSq = attenuation * attenuation * distanceFade;

        // Spot light cone attenuation
        let innerCone = light.geom.y;
        let outerCone = light.geom.z;
        let spotCos = dot(-lightDir, normalize(light.direction + vec3f(0.0001)));
        let spotAttenuation = select(1.0, smoothstep(outerCone, innerCone, spotCos), outerCone > 0.0);

        // Spot shadow calculation (only for spotlights, not point lights)
        let isSpotlight = outerCone > 0.0;
        let spotShadow = calculateSpotShadow(position, n, light.position, light.shadowIndex, screenPos);
        let lightShadow = select(1.0, spotShadow, isSpotlight); // Point lights: no shadow

        // Calculate lighting vectors
        let pl = lightDir;
        let ph = normalize(pl + v);
        let pNdotL = clampedDot(n, pl);
        let pNdotH = clampedDot(n, ph);
        let pVdotH = clampedDot(v, ph);

        // Light intensity from color.a, masked by enabled, attenuation, and shadow
        let pIntensity = isEnabled * light.color.a * attenuationSq * spotAttenuation * lightShadow * pNdotL;

        // Add light contribution (always compute, multiply by intensity mask)
        // Use simpler diffuse calculation for point lights
        diffuse += pIntensity * light.color.rgb * c_diff / PI;
        specular += pIntensity * light.color.rgb * BRDF_specularGGX(f0, f90, alphaRoughness, specularWeight, pVdotH, pNdotL, NdotV, pNdotH) * uniforms.cameraParams.w;
    }

    // Specular boost: 3 fake lights at 60° elevation, 120° apart
    // Only affects materials with specularBoost >= 0.04 and roughness < cutoff, ignores shadows
    let boostIntensity = uniforms.specularBoost.x;
    let boostRoughnessCutoff = uniforms.specularBoost.y;

    // Per-material specularBoost controls this effect (0 = disabled, 1 = full effect)
    // Skip calculation entirely if material boost is below threshold (default 0 means no boost)
    if (materialSpecularBoost >= 0.04 && boostIntensity > 0.0 && roughness < boostRoughnessCutoff && depth < 0.999999) {

        // 3 light directions at 30° above horizon (sin30=0.5, cos30=0.866), 120° apart
        // Pre-calculated normalized directions:
        let boostDir0 = vec3f(0.0, 0.5, 0.866025);           // 0° yaw
        let boostDir1 = vec3f(0.75, 0.5, -0.433013);         // 120° yaw
        let boostDir2 = vec3f(-0.75, 0.5, -0.433013);        // 240° yaw

        // Fade based on roughness: full at 0, zero at cutoff
        let roughnessFade = 1.0 - (roughness / boostRoughnessCutoff);
        // Include per-material boost in final strength
        let boostStrength = boostIntensity * roughnessFade * roughnessFade * materialSpecularBoost;

        // Use sharper roughness for boost highlights (smaller, more concentrated)
        // Square the roughness to make highlights much tighter
        let boostRoughness = roughness * 0.5;
        let boostAlphaRoughness = boostRoughness * 0.5;

        // Calculate specular for each boost light
        var boostSpecular = vec3f(0.0);

        // Light 0
        let bNdotL0 = max(dot(n, boostDir0), 0.0);
        if (bNdotL0 > 0.0) {
            let bH0 = normalize(boostDir0 + v);
            let bNdotH0 = max(dot(n, bH0), 0.0);
            let bVdotH0 = max(dot(v, bH0), 0.0);
            boostSpecular += bNdotL0 * BRDF_specularGGX(f0, f90, boostAlphaRoughness, specularWeight, bVdotH0, bNdotL0, NdotV, bNdotH0);
        }

        // Light 1
        let bNdotL1 = max(dot(n, boostDir1), 0.0);
        if (bNdotL1 > 0.0) {
            let bH1 = normalize(boostDir1 + v);
            let bNdotH1 = max(dot(n, bH1), 0.0);
            let bVdotH1 = max(dot(v, bH1), 0.0);
            boostSpecular += bNdotL1 * BRDF_specularGGX(f0, f90, boostAlphaRoughness, specularWeight, bVdotH1, bNdotL1, NdotV, bNdotH1);
        }

        // Light 2
        let bNdotL2 = max(dot(n, boostDir2), 0.0);
        if (bNdotL2 > 0.0) {
            let bH2 = normalize(boostDir2 + v);
            let bNdotH2 = max(dot(n, bH2), 0.0);
            let bVdotH2 = max(dot(v, bH2), 0.0);
            boostSpecular += bNdotL2 * BRDF_specularGGX(f0, f90, boostAlphaRoughness, specularWeight, bVdotH2, bNdotL2, NdotV, bNdotH2);
        }

        // Add boost specular (white light, same intensity as main light would have)
        specular += boostSpecular * boostStrength;
    }

    // calculate final color

    let specularColor = specular + emissive;
    let diffuseColor = diffuse;

    var color = diffuseColor + specularColor;
    color *= uniforms.environmentParams.a;

    if (depth > 0.999999) {
        var bgDir = v;
        if (uniforms.noiseParams.w > 0.5) {
            // Octahedral encoding: captured probe uses standard coordinate system
            // Just negate view direction to get world direction
            bgDir = -v;
        } else {
            // Equirectangular encoding: flip to match IBL lighting orientation
            bgDir.x *= -1.0;
            bgDir.z *= -1.0;
            bgDir.y *= -1.0;
        }
        // In reflection mode, flip Y to mirror the sky
        if (uniforms.cameraParams.z > 0.5) {
            bgDir.y *= -1.0;
        }
        // Background sky: only apply exposure, NOT diffuse/specular IBL levels
        // Those levels are for PBR lighting, not for displaying the actual sky
        let bgSample = getIBLSample(bgDir, 0.0) * uniforms.environmentParams.a;
        color = bgSample;
    }

    return vec4f(color, 1.0);
}
