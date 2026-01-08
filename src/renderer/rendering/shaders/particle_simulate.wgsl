// Particle simulation compute shader
// Updates particle positions, velocities, lifetimes, and handles spawning
// Also calculates per-particle lighting with temporal smoothing

const WORKGROUP_SIZE: u32 = 64u;
const MAX_EMITTERS: u32 = 16u;
const CASCADE_COUNT = 3;
const MAX_LIGHTS = 64u;
const MAX_SPOT_SHADOWS = 8;
const LIGHTING_FADE_TIME: f32 = 0.3;  // Seconds to smooth lighting changes

// Spot shadow atlas constants
const SPOT_ATLAS_WIDTH: f32 = 2048.0;
const SPOT_ATLAS_HEIGHT: f32 = 2048.0;
const SPOT_TILE_SIZE: f32 = 512.0;
const SPOT_TILES_PER_ROW: i32 = 4;

// Particle data structure (80 bytes, matches JS PARTICLE_STRIDE)
// flags: bit 0 = alive, bit 1 = additive, bits 8-15 = emitter index
struct Particle {
    position: vec3f,      // World position
    lifetime: f32,        // Remaining life (seconds), <=0 = dead
    velocity: vec3f,      // Movement per second
    maxLifetime: f32,     // Initial lifetime for fade calculation
    color: vec4f,         // RGBA with computed alpha
    size: vec2f,          // Current width/height
    rotation: f32,        // Current rotation in radians
    flags: u32,           // Bit 0 = alive, bit 1 = additive, bits 8-15 = emitter index
    lighting: vec3f,      // Pre-computed lighting (smoothed over time)
    lightingPad: f32,     // Padding for alignment
}

// Light structure (must match LightingPass)
struct Light {
    enabled: u32,
    position: vec3f,
    color: vec4f,
    direction: vec3f,
    geom: vec4f, // x = radius, y = inner cone, z = outer cone, w = distance fade
    shadowIndex: i32,
}

struct CascadeMatrices {
    matrices: array<mat4x4<f32>, CASCADE_COUNT>,
}

struct SpotShadowMatrices {
    matrices: array<mat4x4<f32>, MAX_SPOT_SHADOWS>,
}

// Spawn request structure (64 bytes, matches JS SPAWN_REQUEST_STRIDE)
struct SpawnRequest {
    position: vec3f,
    lifetime: f32,
    velocity: vec3f,
    maxLifetime: f32,
    color: vec4f,
    startSize: f32,
    endSize: f32,
    rotation: f32,        // Initial rotation (random)
    flags: u32,           // Bit 0 = alive, bit 1 = additive, bits 8-15 = emitter index
}

// Per-emitter simulation settings (64 bytes each)
struct EmitterSettings {
    gravity: vec3f,
    drag: f32,
    turbulence: f32,
    fadeIn: f32,
    fadeOut: f32,
    rotationSpeed: f32,
    startSize: f32,
    endSize: f32,
    baseAlpha: f32,
    padding: f32,
}

struct SimulationUniforms {
    dt: f32,
    time: f32,
    maxParticles: u32,
    emitterCount: u32,
    // Lighting parameters
    cameraPosition: vec3f,
    shadowBias: f32,
    lightDir: vec3f,
    shadowStrength: f32,
    lightColor: vec4f,
    ambientColor: vec4f,
    cascadeSizes: vec4f,
    lightCount: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

struct Counters {
    aliveCount: atomic<u32>,
    nextFreeIndex: atomic<u32>,
    spawnCount: u32,
    frameCount: u32,
}

// Core bindings
@group(0) @binding(0) var<uniform> uniforms: SimulationUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> counters: Counters;
@group(0) @binding(3) var<storage, read> spawnRequests: array<SpawnRequest>;
@group(0) @binding(4) var<storage, read> emitterSettings: array<EmitterSettings>;

// Lighting bindings
@group(0) @binding(5) var<storage, read> emitterRenderSettings: array<EmitterRenderSettings>;
@group(0) @binding(6) var shadowMapArray: texture_depth_2d_array;
@group(0) @binding(7) var shadowSampler: sampler_comparison;
@group(0) @binding(8) var<storage, read> cascadeMatrices: CascadeMatrices;
@group(0) @binding(9) var<storage, read> lights: array<Light, MAX_LIGHTS>;
@group(0) @binding(10) var spotShadowAtlas: texture_depth_2d;
@group(0) @binding(11) var<storage, read> spotMatrices: SpotShadowMatrices;

// Emitter render settings (lit, emissive)
struct EmitterRenderSettings {
    lit: f32,
    emissive: f32,
    softness: f32,
    zOffset: f32,
}

// Simple noise function for turbulence
fn hash(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise3D(p: vec3f) -> vec3f {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return vec3f(
        mix(hash(i), hash(i + vec3f(1.0, 0.0, 0.0)), u.x),
        mix(hash(i + vec3f(0.0, 1.0, 0.0)), hash(i + vec3f(1.0, 1.0, 0.0)), u.x),
        mix(hash(i + vec3f(0.0, 0.0, 1.0)), hash(i + vec3f(1.0, 0.0, 1.0)), u.x)
    ) * 2.0 - 1.0;
}

// ============= LIGHTING FUNCTIONS =============

// Squircle distance for cascade selection
fn squircleDistanceXZ(offset: vec2f, size: f32) -> f32 {
    let normalized = offset / size;
    let absNorm = abs(normalized);
    return pow(pow(absNorm.x, 4.0) + pow(absNorm.y, 4.0), 0.25);
}

// Sample shadow from cascade (single tap for compute shader performance)
fn sampleCascadeShadow(worldPos: vec3f, cascadeIndex: i32) -> f32 {
    let bias = uniforms.shadowBias;
    let lightMatrix = cascadeMatrices.matrices[cascadeIndex];
    let lightSpacePos = lightMatrix * vec4f(worldPos, 1.0);
    let projCoords = lightSpacePos.xyz / lightSpacePos.w;
    let shadowUV = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);
    let currentDepth = projCoords.z - bias;

    // Check bounds
    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
        currentDepth < 0.0 || currentDepth > 1.0) {
        return 1.0;
    }

    // Single tap shadow for compute shader (faster)
    return textureSampleCompareLevel(shadowMapArray, shadowSampler, shadowUV, cascadeIndex, currentDepth);
}

// Calculate cascade shadow for particle
fn calculateParticleShadow(worldPos: vec3f) -> f32 {
    let camXZ = vec2f(uniforms.cameraPosition.x, uniforms.cameraPosition.z);
    let posXZ = vec2f(worldPos.x, worldPos.z);
    let offsetXZ = posXZ - camXZ;

    let dist0 = squircleDistanceXZ(offsetXZ, uniforms.cascadeSizes.x);
    let dist1 = squircleDistanceXZ(offsetXZ, uniforms.cascadeSizes.y);
    let dist2 = squircleDistanceXZ(offsetXZ, uniforms.cascadeSizes.z);

    var shadow = 1.0;
    if (dist0 < 0.95) {
        shadow = sampleCascadeShadow(worldPos, 0);
    } else if (dist1 < 0.95) {
        shadow = sampleCascadeShadow(worldPos, 1);
    } else if (dist2 < 0.95) {
        shadow = sampleCascadeShadow(worldPos, 2);
    }

    return mix(1.0 - uniforms.shadowStrength, 1.0, shadow);
}

// Calculate spot shadow (single tap)
fn calculateSpotShadow(worldPos: vec3f, slotIndex: i32) -> f32 {
    if (slotIndex < 0 || slotIndex >= MAX_SPOT_SHADOWS) {
        return 1.0;
    }

    let lightMatrix = spotMatrices.matrices[slotIndex];
    let lightSpacePos = lightMatrix * vec4f(worldPos, 1.0);
    let w = max(abs(lightSpacePos.w), 0.0001) * sign(lightSpacePos.w + 0.0001);
    let projCoords = lightSpacePos.xyz / w;

    if (projCoords.z < 0.0 || projCoords.z > 1.0 || abs(projCoords.x) > 1.0 || abs(projCoords.y) > 1.0) {
        return 1.0;
    }

    let col = slotIndex % SPOT_TILES_PER_ROW;
    let row = slotIndex / SPOT_TILES_PER_ROW;
    let localUV = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);
    let tileOffset = vec2f(f32(col), f32(row)) * SPOT_TILE_SIZE;
    let atlasUV = (tileOffset + localUV * SPOT_TILE_SIZE) / vec2f(SPOT_ATLAS_WIDTH, SPOT_ATLAS_HEIGHT);
    let currentDepth = clamp(projCoords.z - uniforms.shadowBias * 3.0, 0.001, 0.999);

    return textureSampleCompareLevel(spotShadowAtlas, shadowSampler, atlasUV, currentDepth);
}

// Calculate full lighting for a particle at given position
fn calculateParticleLighting(worldPos: vec3f, emitterIdx: u32) -> vec3f {
    let renderSettings = emitterRenderSettings[min(emitterIdx, MAX_EMITTERS - 1u)];

    // Unlit particles - just return emissive multiplier as 1.0
    if (renderSettings.lit < 0.5) {
        return vec3f(renderSettings.emissive);
    }

    // Particle normal is always up
    let normal = vec3f(0.0, 1.0, 0.0);

    // Start with ambient
    var lighting = uniforms.ambientColor.rgb * uniforms.ambientColor.a;

    // Main directional light with shadow
    let shadow = calculateParticleShadow(worldPos);
    let NdotL = max(dot(normal, uniforms.lightDir), 0.0);
    lighting += uniforms.lightColor.rgb * uniforms.lightColor.a * NdotL * shadow;

    // Point and spot lights
    let lightCount = uniforms.lightCount;
    for (var i = 0u; i < min(lightCount, MAX_LIGHTS); i++) {
        let light = lights[i];
        if (light.enabled == 0u) { continue; }

        let lightVec = light.position - worldPos;
        let dist = length(lightVec);
        let lightDir = normalize(lightVec);
        let distanceFade = light.geom.w;

        // Attenuation
        let radius = max(light.geom.x, 0.001);
        let attenuation = max(0.0, 1.0 - dist / radius);
        let attenuationSq = attenuation * attenuation * distanceFade;
        if (attenuationSq <= 0.001) { continue; }

        // Spot cone
        let innerCone = light.geom.y;
        let outerCone = light.geom.z;
        var spotAttenuation = 1.0;
        let isSpotlight = outerCone > 0.0;
        if (isSpotlight) {
            let spotCos = dot(-lightDir, normalize(light.direction));
            spotAttenuation = smoothstep(outerCone, innerCone, spotCos);
        }
        if (spotAttenuation <= 0.001) { continue; }

        // Spot shadow
        var lightShadow = 1.0;
        if (isSpotlight && light.shadowIndex >= 0) {
            lightShadow = calculateSpotShadow(worldPos, light.shadowIndex);
        }

        // Add contribution
        let pNdotL = max(dot(normal, lightDir), 0.0);
        let pIntensity = light.color.a * attenuationSq * spotAttenuation * lightShadow * pNdotL;
        lighting += pIntensity * light.color.rgb;
    }

    return lighting * renderSettings.emissive;
}

// Spawn pass: initialize new particles from spawn requests
@compute @workgroup_size(WORKGROUP_SIZE)
fn spawn(@builtin(global_invocation_id) globalId: vec3u) {
    let idx = globalId.x;

    // Check if this thread should process a spawn request
    if (idx >= counters.spawnCount) {
        return;
    }

    // Find a free particle slot (try multiple slots if needed)
    let maxAttempts = 8u;  // Limit search to avoid infinite loop
    var particleIdx = 0u;
    var foundSlot = false;

    for (var attempt = 0u; attempt < maxAttempts; attempt++) {
        let rawIdx = atomicAdd(&counters.nextFreeIndex, 1u);
        particleIdx = rawIdx % uniforms.maxParticles;

        // Check if this slot is free (particle is dead)
        let existing = particles[particleIdx];
        if ((existing.flags & 1u) == 0u || existing.lifetime <= 0.0) {
            foundSlot = true;
            break;
        }
    }

    // If no free slot found, skip this spawn
    if (!foundSlot) {
        return;
    }

    // Read spawn request
    let req = spawnRequests[idx];

    // Initialize particle from spawn request
    var p: Particle;
    p.position = req.position;
    p.lifetime = req.lifetime;
    p.velocity = req.velocity;
    p.maxLifetime = req.maxLifetime;
    p.color = req.color;
    p.size = vec2f(req.startSize, req.startSize);
    p.rotation = req.rotation;  // Random initial rotation
    p.flags = req.flags;
    // Initialize lighting to ambient (will smooth towards actual lighting)
    p.lighting = uniforms.ambientColor.rgb * uniforms.ambientColor.a;
    p.lightingPad = 0.0;

    // Store particle
    particles[particleIdx] = p;

    // Increment alive count
    atomicAdd(&counters.aliveCount, 1u);
}

// Simulate pass: update existing particles
@compute @workgroup_size(WORKGROUP_SIZE)
fn simulate(@builtin(global_invocation_id) globalId: vec3u) {
    let idx = globalId.x;

    // Bounds check
    if (idx >= uniforms.maxParticles) {
        return;
    }

    var p = particles[idx];

    // Skip dead particles
    if ((p.flags & 1u) == 0u || p.lifetime <= 0.0) {
        return;
    }

    // Get emitter index from flags (bits 8-15)
    let emitterIdx = (p.flags >> 8u) & 0xFFu;
    let settings = emitterSettings[min(emitterIdx, MAX_EMITTERS - 1u)];

    let dt = uniforms.dt;

    // Update lifetime
    p.lifetime -= dt;

    // Check if particle died
    if (p.lifetime <= 0.0) {
        p.flags = 0u;  // Mark as dead
        p.lifetime = 0.0;
        atomicSub(&counters.aliveCount, 1u);
        particles[idx] = p;
        return;
    }

    // Calculate life progress (0 = just born, 1 = about to die)
    let lifeProgress = 1.0 - (p.lifetime / p.maxLifetime);

    // Apply gravity (per-emitter)
    p.velocity += settings.gravity * dt;

    // Apply drag (per-emitter)
    let dragFactor = 1.0 - settings.drag * dt;
    p.velocity *= max(dragFactor, 0.0);

    // Apply turbulence (per-emitter)
    if (settings.turbulence > 0.0) {
        let noisePos = p.position * 0.5 + vec3f(uniforms.time * 0.1, 0.0, 0.0);
        let turbulenceForce = noise3D(noisePos + vec3f(p.rotation * 10.0)) * settings.turbulence;
        p.velocity += turbulenceForce * dt * 10.0;
    }

    // Update position
    p.position += p.velocity * dt;

    // Interpolate size over lifetime (per-emitter)
    let currentSize = mix(settings.startSize, settings.endSize, lifeProgress);
    p.size = vec2f(currentSize, currentSize);

    // Update rotation (per-emitter)
    let rotationDir = select(-1.0, 1.0, p.rotation >= 0.0);
    p.rotation += rotationDir * settings.rotationSpeed * dt;

    // Calculate alpha with fade in/out (per-emitter)
    var alpha = settings.baseAlpha;

    // Fade in
    let fadeInDuration = settings.fadeIn / p.maxLifetime;
    if (lifeProgress < fadeInDuration && fadeInDuration > 0.0) {
        alpha *= lifeProgress / fadeInDuration;
    }

    // Fade out
    let fadeOutStart = 1.0 - (settings.fadeOut / p.maxLifetime);
    if (lifeProgress > fadeOutStart && fadeOutStart < 1.0) {
        let fadeOutProgress = (lifeProgress - fadeOutStart) / (1.0 - fadeOutStart);
        alpha *= 1.0 - fadeOutProgress;
    }

    // Only update alpha, preserve RGB from spawn
    p.color.a = alpha;

    // Calculate target lighting at particle center
    let targetLighting = calculateParticleLighting(p.position, emitterIdx);

    // Smooth lighting over time (exponential moving average)
    // lerpFactor = 1 - exp(-dt / fadeTime) ≈ dt / fadeTime for small dt
    let lerpFactor = clamp(dt / LIGHTING_FADE_TIME, 0.0, 1.0);
    p.lighting = mix(p.lighting, targetLighting, lerpFactor);

    // Write updated particle
    particles[idx] = p;
}

// Reset counters (run once per frame before spawn)
@compute @workgroup_size(1)
fn resetCounters() {
    // Reset spawn-related counter but preserve alive count
    atomicStore(&counters.nextFreeIndex, 0u);
}
