const PI = 3.14159265359;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Uniforms {
    inverseProjection: mat4x4<f32>,
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    aoSize: vec2f,        // AO render target size (may be scaled)
    gbufferSize: vec2f,   // GBuffer full resolution size
    // AO params: x = intensity, y = radius, z = fadeDistance, w = bias
    aoParams: vec4f,
    // Noise params: x = size, y = offsetX, z = offsetY, w = time
    noiseParams: vec4f,
    // Camera params: x = near, y = far
    cameraParams: vec2f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var gDepth: texture_depth_2d;
@group(0) @binding(2) var gNormal: texture_2d<f32>;
@group(0) @binding(3) var gArm: texture_2d<f32>;
@group(0) @binding(4) var noiseTexture: texture_2d<f32>;

// Poisson disk offsets for 16 samples - well distributed pattern
const OFFSETS_16 = array<vec2f, 16>(
    vec2f(0.063, 0.000),
    vec2f(0.079, 0.097),
    vec2f(-0.039, 0.183),
    vec2f(-0.223, 0.113),
    vec2f(-0.285, -0.127),
    vec2f(-0.097, -0.362),
    vec2f(0.257, -0.354),
    vec2f(0.499, -0.026),
    vec2f(0.376, 0.418),
    vec2f(-0.098, 0.617),
    vec2f(-0.595, 0.344),
    vec2f(-0.700, -0.269),
    vec2f(-0.251, -0.773),
    vec2f(0.477, -0.734),
    vec2f(0.932, -0.098),
    vec2f(0.707, 0.707)
);

// Convert linear depth buffer value back to view-space distance
// Inverse of: depth = (z - near) / (far - near)
// Result: z = near + depth * (far - near)
fn depthToLinear(depth: f32) -> f32 {
    let near = uniforms.cameraParams.x;
    let far = uniforms.cameraParams.y;
    return near + depth * (far - near);
}

// Get linear depth at pixel coordinate (converts from depth buffer)
fn getDepth(coord: vec2i) -> f32 {
    let bufferDepth = textureLoad(gDepth, coord, 0);
    return depthToLinear(bufferDepth);
}

// Sample noise at screen position
fn sampleNoise(screenPos: vec2f) -> f32 {
    let noiseSize = i32(uniforms.noiseParams.x);
    let noiseOffsetX = i32(uniforms.noiseParams.y * f32(noiseSize));
    let noiseOffsetY = i32(uniforms.noiseParams.z * f32(noiseSize));

    let texCoord = vec2i(
        (i32(screenPos.x) + noiseOffsetX) % noiseSize,
        (i32(screenPos.y) + noiseOffsetY) % noiseSize
    );
    return textureLoad(noiseTexture, texCoord, 0).r;
}

// Rotate a 2D vector by angle
fn rotate2D(v: vec2f, angle: f32) -> vec2f {
    let s = sin(angle);
    let c = cos(angle);
    return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
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

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) f32 {
    // Use UV to sample GBuffer at correct position regardless of AO resolution
    let gbufferCoord = vec2i(floor(input.uv * uniforms.gbufferSize));
    let aoCoord = vec2i(floor(input.uv * uniforms.aoSize));
    let screenPos = input.position.xy;

    // Get linear depth in meters (sample from full-res GBuffer)
    let depth = getDepth(gbufferCoord);
    let normal = normalize(textureLoad(gNormal, gbufferCoord, 0).xyz);

    // Parameters
    let aoIntensity = uniforms.aoParams.x;
    let aoRadius = uniforms.aoParams.y;
    let aoFadeDistance = uniforms.aoParams.z;
    let aoBias = uniforms.aoParams.w;
    let far = uniforms.cameraParams.y;

    // Early out for sky
    if (depth > far * 0.95) {
        return 1.0;
    }

    // Distance fade
    let distanceFade = 1.0 - smoothstep(aoFadeDistance * 0.5, aoFadeDistance, depth);
    if (distanceFade < 0.01) {
        return 1.0;
    }

    // Blue noise jitter
    let noise = sampleNoise(screenPos);
    let rotationAngle = noise * PI * 2.0;

    // Sample radius in pixels - scale with depth so AO is consistent across distances
    let sampleRadius = aoRadius / max(depth * 0.2, 1.0);

    var occlusion = 0.0;
    var validSamples = 0.0;

    // Occlusion threshold in meters - scales with depth
    let occlusionThreshold = 0.1 + depth * 0.02;  // 10cm + 2% of depth
    let maxOcclusionDist = occlusionThreshold * 5.0;

    // Scale factor from AO space to GBuffer space
    let scaleToGBuffer = uniforms.gbufferSize / uniforms.aoSize;

    for (var i = 0; i < 16; i++) {
        let offset = rotate2D(OFFSETS_16[i], rotationAngle) * sampleRadius;
        // Scale offset from AO-space to GBuffer-space pixels
        let scaledOffset = offset * scaleToGBuffer;
        let sampleCoord = gbufferCoord + vec2i(i32(scaledOffset.x), i32(scaledOffset.y));

        // Bounds check against GBuffer size
        if (sampleCoord.x < 0 || sampleCoord.x >= i32(uniforms.gbufferSize.x) ||
            sampleCoord.y < 0 || sampleCoord.y >= i32(uniforms.gbufferSize.y)) {
            continue;
        }

        let sampleDepth = getDepth(sampleCoord);

        // Simple depth-based AO like the reference shader
        // ddiff > 0 means sample is closer to camera (we're behind it = occluded)
        let ddiff = depth - sampleDepth;

        // If sample is closer (ddiff > 0), we're occluded
        // If sample is further or same (ddiff <= 0), we're not occluded
        let unoccluded = select(1.0, 0.0, ddiff > aoBias);

        // Ignore samples that are much closer (edge/discontinuity)
        let relevant = 1.0 - smoothstep(occlusionThreshold, maxOcclusionDist, ddiff);

        occlusion += unoccluded * relevant;
        validSamples += relevant;
    }

    // Final AO calculation
    // occlusion = sum of unoccluded samples, validSamples = sum of relevant weights
    // Higher unoccluded ratio = less darkening
    var ao = 1.0;
    if (validSamples > 0.5) {
        let unoccludedRatio = occlusion / (validSamples * 0.5);  // Like reference shader
        let aoFactor = 1.0 - clamp(unoccludedRatio, 0.0, 1.0);
        ao = 1.0 - aoFactor * aoIntensity * 0.5;
    }

    // Apply distance fade
    ao = mix(1.0, ao, distanceFade);

    return ao;
}
