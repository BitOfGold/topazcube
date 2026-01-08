// Shared lighting functions for deferred and forward rendering
// This file is imported as a string and concatenated with other shaders

const PI = 3.14159;
const MAX_LIGHTS = 768;
const CASCADE_COUNT = 3;
const MAX_LIGHTS_PER_TILE = 256;
const MAX_SPOT_SHADOWS = 16;

// Hard-coded spot shadow parameters
const SPOT_ATLAS_WIDTH: f32 = 2048.0;
const SPOT_ATLAS_HEIGHT: f32 = 2048.0;
const SPOT_TILE_SIZE: f32 = 512.0;
const SPOT_TILES_PER_ROW: i32 = 4;
const SPOT_FADE_START: f32 = 25.0;
const SPOT_MAX_DISTANCE: f32 = 30.0;
const SPOT_MIN_SHADOW: f32 = 0.5;

struct Light {
    enabled: u32,
    position: vec3f,
    color: vec4f,
    direction: vec3f,
    geom: vec4f,
    shadowIndex: i32,
}

struct SpotShadowMatrices {
    matrices: array<mat4x4<f32>, MAX_SPOT_SHADOWS>,
}

struct CascadeMatrices {
    matrices: array<mat4x4<f32>, CASCADE_COUNT>,
}

// ============================================
// Environment mapping functions
// ============================================

fn SphToUV(n: vec3<f32>) -> vec2<f32> {
    var uv: vec2<f32>;
    uv.x = atan2(-n.x, n.z);
    uv.x = (uv.x + PI / 2.0) / (PI * 2.0) + PI * (28.670 / 360.0);
    uv.y = acos(n.y) / PI;
    return uv;
}

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

// ============================================
// Noise sampling functions
// ============================================

fn vogelDiskSample(sampleIndex: i32, numSamples: i32, rotation: f32) -> vec2f {
    let goldenAngle = 2.399963229728653;
    let r = sqrt((f32(sampleIndex) + 0.5) / f32(numSamples));
    let theta = f32(sampleIndex) * goldenAngle + rotation;
    return vec2f(r * cos(theta), r * sin(theta));
}

fn buildOrthonormalBasis(n: vec3f) -> mat3x3<f32> {
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return mat3x3<f32>(tangent, bitangent, n);
}

// ============================================
// PBR BRDF functions
// ============================================

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
    return alphaRoughnessSq / (PI * f * f);
}

fn BRDF_lambertian(f0: vec3<f32>, f90: vec3<f32>, diffuseColor: vec3<f32>, specularWeight: f32, VdotH: f32) -> vec3<f32> {
    return (1.0 - specularWeight * F_Schlick(f0, f90, VdotH)) * (diffuseColor / PI);
}

fn BRDF_specularGGX(f0: vec3<f32>, f90: vec3<f32>, alphaRoughness: f32, specularWeight: f32, VdotH: f32, NdotL: f32, NdotV: f32, NdotH: f32) -> vec3<f32> {
    let F = F_Schlick(f0, f90, VdotH);
    let Vis = V_GGX(NdotL, NdotV, alphaRoughness);
    let D = D_GGX(NdotH, alphaRoughness);
    return specularWeight * F * Vis * D;
}

// ============================================
// Squircle distance for cascade blending
// ============================================

fn squircleDistanceXZ(offset: vec2f, size: f32) -> f32 {
    let normalized = offset / size;
    let absNorm = abs(normalized);
    return pow(pow(absNorm.x, 4.0) + pow(absNorm.y, 4.0), 0.25);
}
