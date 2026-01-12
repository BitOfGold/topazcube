// Volumetric Fog - Simple Ray Marching with Direct Shadow Sampling
// Supports main directional light + point/spot lights with shadows

const MAX_LIGHTS: u32 = 768u;
const MAX_SPOT_SHADOWS: i32 = 16;
const SPOT_ATLAS_SIZE: f32 = 2048.0;
const SPOT_TILE_SIZE: f32 = 512.0;
const SPOT_TILES_PER_ROW: i32 = 4;

struct Uniforms {
    inverseProjection: mat4x4f,
    inverseView: mat4x4f,
    cameraPosition: vec3f,
    nearPlane: f32,
    farPlane: f32,
    maxSamples: f32,
    time: f32,
    fogDensity: f32,
    fogColor: vec3f,
    shadowsEnabled: f32,
    mainLightDir: vec3f,
    mainLightIntensity: f32,
    mainLightColor: vec3f,
    scatterStrength: f32,
    fogHeightFade: vec2f,
    maxDistance: f32,
    lightCount: f32,
    debugMode: f32,
    noiseStrength: f32,    // 0 = uniform fog, 1 = full noise variation
    noiseAnimated: f32,    // 0 = static noise, 1 = animated
    mainLightScatter: f32, // Separate scatter strength for main directional light
    noiseScale: f32,       // Noise frequency multiplier (higher = smaller details)
    mainLightSaturation: f32, // Max brightness for main light (logarithmic cap)
}

// Light struct must match LightingPass buffer layout (96 bytes per light)
struct Light {
    enabled: u32,        // offset 0
    _pad0: u32,          // offset 4
    _pad1: u32,          // offset 8
    _pad2: u32,          // offset 12
    position: vec3f,     // offset 16 (vec3f has 16-byte alignment)
    _pad3: f32,          // offset 28
    color: vec4f,        // offset 32 (rgb + intensity in alpha)
    direction: vec3f,    // offset 48
    _pad4: f32,          // offset 60
    geom: vec4f,         // offset 64 (x = radius, y = inner cone, z = outer cone)
    shadowIndex: i32,    // offset 80 (-1 if no shadow, 0-15 for spot shadow slot)
    _pad5: u32,          // offset 84
    _pad6: u32,          // offset 88
    _pad7: u32,          // offset 92
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depthTexture: texture_depth_2d;
@group(0) @binding(2) var cascadeShadowMaps: texture_depth_2d_array;
@group(0) @binding(3) var shadowSampler: sampler_comparison;
@group(0) @binding(4) var<storage, read> cascadeMatrices: array<mat4x4f>;
@group(0) @binding(5) var<storage, read> lights: array<Light, MAX_LIGHTS>;
@group(0) @binding(6) var spotShadowAtlas: texture_depth_2d;
@group(0) @binding(7) var<storage, read> spotMatrices: array<mat4x4f, MAX_SPOT_SHADOWS>;

// Full-screen triangle vertex shader
@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(vertexIndex & 1u) * 4.0 - 1.0;
    let y = f32(vertexIndex >> 1u) * 4.0 - 1.0;
    output.position = vec4f(x, y, 0.0, 1.0);
    output.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

// Simple 3D noise for fog variation
fn hash(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise3d(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash(i + vec3f(0,0,0)), hash(i + vec3f(1,0,0)), u.x),
            mix(hash(i + vec3f(0,1,0)), hash(i + vec3f(1,1,0)), u.x), u.y),
        mix(mix(hash(i + vec3f(0,0,1)), hash(i + vec3f(1,0,1)), u.x),
            mix(hash(i + vec3f(0,1,1)), hash(i + vec3f(1,1,1)), u.x), u.y),
        u.z
    );
}

fn fbm(p: vec3f) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = p;
    for (var i = 0; i < 3; i++) {
        value += amplitude * noise3d(pos);
        pos *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// Get fog density at world position
fn getFogDensity(worldPos: vec3f, dist: f32) -> f32 {
    // Soft height-based fade at boundaries (smooth transition rather than hard cutoff)
    let heightFade = uniforms.fogHeightFade;
    var heightMod = 1.0;
    if (heightFade.y > heightFade.x) {
        let range = heightFade.y - heightFade.x;
        let fadeZone = range * 0.1; // 10% fade zone at top
        let topFade = 1.0 - smoothstep(heightFade.y - fadeZone, heightFade.y, worldPos.y);
        let bottomFade = smoothstep(heightFade.x, heightFade.x + fadeZone, worldPos.y);
        heightMod = topFade * bottomFade;
    }

    // If noise strength is 0, return uniform fog
    if (uniforms.noiseStrength < 0.001) {
        return uniforms.fogDensity * heightMod;
    }

    // Time offset: 0 if not animated, actual time if animated
    let timeOffset = select(0.0, uniforms.time, uniforms.noiseAnimated > 0.5);

    // Noise variation for natural look - use multiple scales
    // noiseScale controls detail size: higher = finer detail, lower = larger billows
    let scale1 = uniforms.noiseScale;        // Fine detail
    let scale2 = uniforms.noiseScale * 0.32; // Large-scale variation (about 1/3 of fine)

    let noisePos = worldPos * scale1 + vec3f(timeOffset * 0.15, timeOffset * 0.02, timeOffset * 0.08);
    let noiseVal = fbm(noisePos);

    // Add a second layer of larger-scale variation
    let noisePos2 = worldPos * scale2 + vec3f(timeOffset * 0.05, 0.0, timeOffset * 0.03);
    let noiseVal2 = fbm(noisePos2);

    // Combine noises: fbm returns [0, ~0.875], combine for more range
    let combinedNoise = noiseVal * 0.6 + noiseVal2 * 0.4;

    // Map noise to density multiplier based on noiseStrength
    // noiseStrength=0: multiplier = 1.0 (uniform)
    // noiseStrength=1: multiplier = [0.2, 1.2] (full variation)
    let noiseRange = combinedNoise * 1.0 + 0.2;  // [0.2, 1.2]
    let noiseMapped = mix(1.0, noiseRange, uniforms.noiseStrength);

    return uniforms.fogDensity * heightMod * noiseMapped;
}

// Sample cascade shadow at world position
fn sampleShadow(worldPos: vec3f) -> f32 {
    if (uniforms.shadowsEnabled < 0.5) {
        return 1.0;
    }

    // Try each cascade
    for (var cascade = 0; cascade < 3; cascade++) {
        let lightSpacePos = cascadeMatrices[cascade] * vec4f(worldPos, 1.0);
        let projCoords = lightSpacePos.xyz / lightSpacePos.w;

        // Convert to UV space
        let uv = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);

        // Check bounds
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 &&
            projCoords.z >= 0.0 && projCoords.z <= 1.0) {

            let bias = 0.003 * f32(cascade + 1);
            let depth = projCoords.z - bias;
            let clampedUV = clamp(uv, vec2f(0.002), vec2f(0.998));
            let clampedDepth = clamp(depth, 0.001, 0.999);

            return textureSampleCompareLevel(cascadeShadowMaps, shadowSampler, clampedUV, cascade, clampedDepth);
        }
    }

    return 1.0;
}

// Sample spot light shadow from atlas
fn sampleSpotShadow(worldPos: vec3f, slotIndex: i32) -> f32 {
    if (slotIndex < 0 || slotIndex >= MAX_SPOT_SHADOWS) {
        return 1.0;
    }

    // Get the light matrix
    let lightMatrix = spotMatrices[slotIndex];

    // Transform to light space
    let lightSpacePos = lightMatrix * vec4f(worldPos, 1.0);

    // Perspective divide
    let w = max(abs(lightSpacePos.w), 0.0001) * sign(lightSpacePos.w + 0.0001);
    let projCoords = lightSpacePos.xyz / w;

    // Check if in frustum
    if (abs(projCoords.x) > 1.0 || abs(projCoords.y) > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0) {
        return 1.0;
    }

    // Calculate tile position in atlas
    let col = slotIndex % SPOT_TILES_PER_ROW;
    let row = slotIndex / SPOT_TILES_PER_ROW;

    // Transform to [0,1] UV space within the tile
    let tileUV = vec2f(projCoords.x * 0.5 + 0.5, 0.5 - projCoords.y * 0.5);

    // Calculate UV in atlas
    let tileOffsetX = f32(col) * SPOT_TILE_SIZE;
    let tileOffsetY = f32(row) * SPOT_TILE_SIZE;
    let atlasUV = vec2f(
        (tileOffsetX + tileUV.x * SPOT_TILE_SIZE) / SPOT_ATLAS_SIZE,
        (tileOffsetY + tileUV.y * SPOT_TILE_SIZE) / SPOT_ATLAS_SIZE
    );

    // Sample shadow with bias
    let bias = 0.005;
    let depth = clamp(projCoords.z - bias, 0.001, 0.999);

    return textureSampleCompareLevel(spotShadowAtlas, shadowSampler, atlasUV, depth);
}

// Calculate lighting contribution from a single light
fn calculateLightContribution(worldPos: vec3f, light: Light, worldRayDir: vec3f) -> vec3f {
    let toLight = light.position - worldPos;
    let dist = length(toLight);
    let radius = light.geom.x;

    // Distance check
    if (dist > radius) {
        return vec3f(0.0);
    }

    // Smooth falloff for volumetric fog - gentler than inverse square
    // so lights are visible throughout their radius
    let normalizedDist = dist / radius;

    // Smooth falloff: strong near light, gradual fade to edge
    // Using smoothstep gives S-curve that looks natural
    let attenuation = 1.0 - smoothstep(0.0, 1.0, normalizedDist);

    // Spotlight cone check
    let innerCone = light.geom.y;
    let outerCone = light.geom.z;
    var spotAttenuation = 1.0;
    let isSpotlight = outerCone > 0.0;

    if (isSpotlight) {
        let lightDir = normalize(-toLight);
        let spotCos = dot(lightDir, normalize(light.direction));
        spotAttenuation = smoothstep(outerCone, innerCone, spotCos);
        if (spotAttenuation <= 0.0) {
            return vec3f(0.0);
        }
    }

    // Shadow
    var shadow = 1.0;
    if (isSpotlight && light.shadowIndex >= 0 && uniforms.shadowsEnabled > 0.5) {
        shadow = sampleSpotShadow(worldPos, light.shadowIndex);
    }

    // Phase function for scattering (isotropic for point/spot)
    let phase = 0.25;  // Isotropic

    // Final contribution
    let intensity = light.color.a;
    return light.color.rgb * intensity * attenuation * spotAttenuation * shadow * phase;
}

// Henyey-Greenstein phase function for forward scattering
fn phaseHG(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(denom, 1.5));
}

// Convert depth buffer value to linear distance
fn linearizeDepth(depth: f32) -> f32 {
    let near = uniforms.nearPlane;
    let far = uniforms.farPlane;
    return near + depth * (far - near);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;

    // Debug modes: 0=normal, 1=depth, 2=ray dir, 3=noise, 4=viewDir.z, 5=worldPos, 6=accum color, 7=light positions
    let DEBUG = i32(uniforms.debugMode);

    // Reconstruct world ray direction from screen UV
    // Use clip space at z=0 (near plane) for ray direction - matches FogPass pattern
    var clipUV = uv;
    clipUV.y = 1.0 - clipUV.y;  // Flip Y for clip space
    let ndc = vec4f(clipUV * 2.0 - 1.0, 0.0, 1.0);

    // Transform to view space
    let viewRay4 = uniforms.inverseProjection * ndc;
    let viewDir = normalize(viewRay4.xyz / viewRay4.w);

    // Transform view direction to world space
    let worldRayDir = normalize((uniforms.inverseView * vec4f(viewDir, 0.0)).xyz);

    // Get scene depth from full-resolution depth texture
    let depthTexSize = vec2f(textureDimensions(depthTexture));
    let depthCoord = vec2i(uv * depthTexSize);
    let rawDepth = textureLoad(depthTexture, depthCoord, 0);

    // Debug: show raw depth
    if (DEBUG == 1) {
        return vec4f(rawDepth, rawDepth, rawDepth, 1.0);
    }

    // Debug: show world ray direction (should vary across screen)
    if (DEBUG == 2) {
        return vec4f(worldRayDir * 0.5 + 0.5, 1.0);
    }

    // Debug: show viewDir.z (should vary across screen)
    if (DEBUG == 4) {
        let vzVis = abs(viewDir.z);
        return vec4f(vzVis, vzVis, vzVis, 1.0);
    }

    // Debug: visualize first light position (should show light location, not camera)
    // Red = distance from camera to light (normalized by 50m)
    // Green = light enabled (0.5 if enabled)
    // Blue = light radius / 50
    if (DEBUG == 7) {
        let numLights = i32(uniforms.lightCount);
        if (numLights > 0) {
            let light = lights[0];
            // Distance from CAMERA to light (not from ray sample)
            let camToLightDist = length(light.position - uniforms.cameraPosition);
            let radius = light.geom.x;
            // Show: R=distance/50, G=enabled, B=radius/50
            return vec4f(
                clamp(camToLightDist / 50.0, 0.0, 1.0),
                f32(light.enabled) * 0.5,
                clamp(radius / 50.0, 0.0, 1.0),
                1.0
            );
        }
        return vec4f(1.0, 0.0, 0.0, 1.0); // Red = no lights
    }

    // Debug: visualize light position in world space
    // Shows light position components normalized
    if (DEBUG == 8) {
        let numLights = i32(uniforms.lightCount);
        if (numLights > 0) {
            let light = lights[0];
            // Show light position components (scaled to [0,1] range assuming -100 to 100 world coords)
            return vec4f(
                (light.position.x + 100.0) / 200.0,
                (light.position.y + 10.0) / 60.0,
                (light.position.z + 100.0) / 200.0,
                1.0
            );
        }
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // Calculate max ray distance to geometry (or infinite for sky)
    var geometryDist = 1000000.0;
    if (rawDepth < 0.9999) {
        let linearDepth = linearizeDepth(rawDepth);
        geometryDist = linearDepth / max(0.001, -viewDir.z) * 0.98;
    }

    // Calculate ray start/end based on fog height bounds
    // Fog exists between fogHeightFade.x (bottom) and fogHeightFade.y (top)
    let fogBottom = uniforms.fogHeightFade.x;
    let fogTop = uniforms.fogHeightFade.y;
    let camY = uniforms.cameraPosition.y;

    // Find where ray intersects fog volume (height planes)
    // Ray: P = camPos + t * worldRayDir
    // For y = fogBottom: t = (fogBottom - camY) / worldRayDir.y
    // For y = fogTop: t = (fogTop - camY) / worldRayDir.y
    var tStart = 0.0;
    var tEnd = geometryDist;

    if (abs(worldRayDir.y) > 0.0001) {
        let tBottom = (fogBottom - camY) / worldRayDir.y;
        let tTop = (fogTop - camY) / worldRayDir.y;

        // Determine entry and exit based on ray direction
        let tEnter = min(tBottom, tTop);
        let tExit = max(tBottom, tTop);

        // If camera is outside fog, start from where ray enters
        if (camY < fogBottom || camY > fogTop) {
            tStart = max(0.0, tEnter);
        }

        // End at fog boundary or geometry
        tEnd = min(tEnd, max(0.0, tExit));
    } else {
        // Ray is horizontal - check if we're in the fog layer
        if (camY < fogBottom || camY > fogTop) {
            return vec4f(0.0);  // Camera outside fog, horizontal ray never enters
        }
    }

    // Skip if ray doesn't travel through fog
    if (tStart >= tEnd || (tEnd - tStart) < 0.5) {
        return vec4f(0.0);
    }

    // Debug 9: show ray distance coverage
    // R = tStart/50, G = tEnd/50, B = rayLength/50
    if (DEBUG == 9) {
        let rayLength = tEnd - tStart;
        return vec4f(tStart / 50.0, tEnd / 50.0, rayLength / 50.0, 1.0);
    }

    // Debug 10: test if ray intersects first light's volume
    // Sample at 10 points along ray and check if any are inside light
    if (DEBUG == 10) {
        let numLights = i32(uniforms.lightCount);
        if (numLights > 0) {
            let light = lights[0];
            let radius = light.geom.x;
            var minDist = 9999.0;
            var hitCount = 0.0;
            let rayLength = tEnd - tStart;
            for (var ti = 0; ti < 20; ti++) {
                let testT = tStart + rayLength * f32(ti) / 20.0;
                let testPos = uniforms.cameraPosition + worldRayDir * testT;
                let dist = length(light.position - testPos);
                minDist = min(minDist, dist);
                if (dist < radius) {
                    hitCount += 1.0;
                }
            }
            // R = min distance / 50, G = hit count / 20, B = light radius / 50
            return vec4f(minDist / 50.0, hitCount / 20.0, radius / 50.0, 1.0);
        }
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }

    // Debug: show noise at sample position (should vary across screen)
    if (DEBUG == 3) {
        let samplePos = uniforms.cameraPosition + worldRayDir * min(10.0, tEnd);
        let n = getFogDensity(samplePos, 10.0);
        return vec4f(n, n, n, 1.0);
    }

    // Debug: show world position at fixed distance (should vary)
    if (DEBUG == 5) {
        let worldPos = uniforms.cameraPosition + worldRayDir * min(5.0, tEnd);
        return vec4f(fract(worldPos * 0.1), 1.0);
    }

    // Ray marching with fixed step sizes (not scaled to ray length)
    // This ensures we always have good sample density near lights
    let numSamples = i32(uniforms.maxSamples);
    let fNumSamples = f32(numSamples);

    // Spatial jitter to reduce banding
    let jitter = fract(sin(dot(input.position.xy, vec2f(12.9898, 78.233))) * 43758.5453);

    var accumulatedColor = vec3f(0.0);
    var accumulatedMainLight = vec3f(0.0);  // Track main light separately (unshaded)
    var accumulatedShadow = 0.0;  // Weighted shadow accumulation
    var shadowWeight = 0.0;  // Total weight for shadow averaging
    var accumulatedAlpha = 0.0;

    // Phase function for main light scattering
    let lightDir = normalize(uniforms.mainLightDir);
    let viewToLight = dot(worldRayDir, lightDir);
    // Boosted phase - more visible from all angles, extra bright when looking toward sun
    let hgPhase = phaseHG(viewToLight, 0.6);
    let phase = max(0.4, hgPhase);  // Minimum 0.4 so shadows are always visible

    // Adaptive step size: cover full ray but with minimum step for quality

    let rayLength = tEnd - tStart;
    let minStepSize = 0.25;  // Never step smaller than this
    let maxStepSize = 2.0;   // Never step larger than this (to not skip lights)
    let stepSize = clamp(rayLength / fNumSamples, minStepSize, maxStepSize);
    //let stepSize = 0.25;

    // Track current position along ray
    var t = tStart + jitter * stepSize;

    // Debug 11: track max light contribution found
    var debugMaxLight = vec3f(0.0);
    var debugLightHits = 0.0;

    for (var i = 0; i < numSamples; i++) {
        // Only stop at fog boundary or surface, never due to density
        if (t > tEnd) { break; }

        let samplePos = uniforms.cameraPosition + worldRayDir * t;

        // Advance t for next iteration (do this early so continue doesn't skip it)
        t += stepSize;

        // Get fog density at this point
        let density = getFogDensity(samplePos, t);
        // Don't skip low density - lights still illuminate thin fog
        // Only skip if truly zero
        if (density <= 0.0) { continue; }

        // Calculate lighting at this point - only from actual lights, no ambient
        // Separate main light from point/spot lights for independent control
        var mainLighting = vec3f(0.0);
        var pointLighting = vec3f(0.0);

        // Main directional light - accumulate shadow weighted by density
        if (uniforms.mainLightIntensity > 0.0) {
            let rawShadow = sampleShadow(samplePos);

            // Accumulate shadow weighted by density (where fog actually is)
            let weight = density * stepSize;
            accumulatedShadow += rawShadow * weight;
            shadowWeight += weight;

            // Accumulate main light base (without shadow factor - applied at end)
            mainLighting = uniforms.mainLightColor * uniforms.mainLightIntensity * phase * uniforms.mainLightScatter;
        }

        // Point and spot lights
        let numLights = i32(uniforms.lightCount);
        for (var li = 0; li < numLights; li++) {
            let light = lights[li];
            if (light.enabled == 0u) { continue; }

            let contrib = calculateLightContribution(samplePos, light, worldRayDir);
            pointLighting += contrib;

            // Track for debug
            if (length(contrib) > 0.001) {
                debugLightHits += 1.0;
                debugMaxLight = max(debugMaxLight, contrib);
            }
        }

        // Apply fog color - scatterStrength only affects point/spot lights
        let pointColor = pointLighting * uniforms.scatterStrength * uniforms.fogColor;
        let mainColor = mainLighting * uniforms.fogColor;

        // Accumulate with density - multiply by step size for proper integration
        // Cap per-sample alpha to prevent rapid saturation near camera
        // This ensures distant lights remain visible even when inside fog
        let rawSampleAlpha = density * stepSize;
        let sampleAlpha = min(rawSampleAlpha, 0.03);  // Max 3% opacity per sample

        // Color contribution is proportional to density but uses uncapped value for brightness
        let colorWeight = rawSampleAlpha * (1.0 - accumulatedAlpha);
        accumulatedColor += pointColor * colorWeight;
        accumulatedMainLight += mainColor * colorWeight;
        accumulatedAlpha += sampleAlpha * (1.0 - accumulatedAlpha);
    }

    // Clamp alpha to valid range
    accumulatedAlpha = min(accumulatedAlpha, 1.0);

    // Calculate weighted average shadow (where fog density is)
    let avgShadow = select(1.0, accumulatedShadow / shadowWeight, shadowWeight > 0.001);

    // Apply strong contrast curve to shadow boundary
    // This creates high contrast god rays at shadow edges
    // smoothstep with tight range pushes values toward 0 or 1
    let contrastShadow = smoothstep(0.25, 0.75, avgShadow);
    // Mix with original to keep some softness
    let finalShadow = mix(avgShadow, contrastShadow, 0.88);

    // Apply saturation cap with logarithmic curve
    let sat = uniforms.mainLightSaturation;
    let saturatedShadow = sat * (1.0 - exp(-finalShadow * 3.0 / max(sat, 0.01)));

    // Add main light contribution with contrast-boosted shadow
    accumulatedColor += accumulatedMainLight * saturatedShadow;

    // Debug 11: show max light contribution found along ray
    // R = max light intensity, G = number of lit samples / 100, B = 0
    if (DEBUG == 11) {
        return vec4f(length(debugMaxLight), debugLightHits / 100.0, 0.0, 1.0);
    }

    // Debug 12: show actual ray march distance and iterations
    // R = distance actually marched / 50, G = iterations / numSamples, B = accumulatedAlpha
    if (DEBUG == 12) {
        let distMarched = t - tStart;
        let iterCount = distMarched / stepSize;
        return vec4f(distMarched / 50.0, iterCount / fNumSamples, accumulatedAlpha, 1.0);
    }

    // Debug 13: check light distances at sample points along ray
    // Sample 10 points and show min distance to first light
    if (DEBUG == 13) {
        let numLights = i32(uniforms.lightCount);
        if (numLights > 0) {
            let light = lights[0];
            let radius = light.geom.x;
            var minDistToLight = 9999.0;
            var closestT = 0.0;
            let testRayLen = t - tStart; // actual marched distance
            for (var ti = 0; ti < 32; ti++) {
                let testT = tStart + testRayLen * f32(ti) / 32.0;
                let testPos = uniforms.cameraPosition + worldRayDir * testT;
                let distToLight = length(light.position - testPos);
                if (distToLight < minDistToLight) {
                    minDistToLight = distToLight;
                    closestT = testT;
                }
            }
            // R = min dist to light / radius (< 1 means inside light)
            // G = closestT / 50 (where along ray is closest point)
            // B = radius / 50
            return vec4f(minDistToLight / radius, closestT / 50.0, radius / 50.0, 1.0);
        }
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }

    // Debug: show raw accumulated color (before compositing)
    if (DEBUG == 6) {
        return vec4f(accumulatedColor, 1.0);
    }

    return vec4f(accumulatedColor, accumulatedAlpha);
}
