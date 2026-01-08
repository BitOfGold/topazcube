// Screen Space Global Illumination - Vogel Disk Sampling
//
// Samples propagated directional light from tiles using 16-sample Vogel disk.
// Uses screen-space normal to weight directional contributions.

struct SSGIUniforms {
    // screenParams: fullWidth, fullHeight, halfWidth, halfHeight
    screenParams: vec4f,
    // tileParams: tileCountX, tileCountY, tileSize, sampleRadius (in tiles)
    tileParams: vec4f,
    // ssgiParams: intensity, frameIndex, unused, unused
    ssgiParams: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: SSGIUniforms;
@group(0) @binding(1) var gbufferNormal: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> propagateBuffer: array<vec4f>;

// Direction indices (must match ssgi_propagate.wgsl)
const DIR_LEFT: u32 = 0u;
const DIR_RIGHT: u32 = 1u;
const DIR_UP: u32 = 2u;
const DIR_DOWN: u32 = 3u;

const PI: f32 = 3.14159265359;
const GOLDEN_ANGLE: f32 = 2.39996323;  // PI * (3 - sqrt(5))
const SAMPLE_COUNT: i32 = 16;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

// Hash function for noise
fn hash21(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, vec3f(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2f) -> vec2f {
    let n = sin(dot(p, vec2f(41.0, 289.0)));
    return fract(vec2f(262144.0, 32768.0) * n);
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );
    var uvs = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0)
    );

    var output: VertexOutput;
    output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
    output.uv = uvs[vertexIndex];
    return output;
}

// Get propagated light for a tile and direction (with clamping)
// Returns vec4f: RGB = light, A = weight (for detecting sky/empty tiles)
fn getTileLightWithWeight(tileX: i32, tileY: i32, direction: u32) -> vec4f {
    let tileCountX = i32(uniforms.tileParams.x);
    let tileCountY = i32(uniforms.tileParams.y);

    // Clamp to valid tile range
    let clampedX = clamp(tileX, 0, tileCountX - 1);
    let clampedY = clamp(tileY, 0, tileCountY - 1);

    let tileIdx = u32(clampedY * tileCountX + clampedX);
    let bufferIdx = tileIdx * 4u + direction;

    return propagateBuffer[bufferIdx];
}

// Bilinear sample tile light at fractional position
// Returns vec4f: RGB = light, A = validity weight (0 = sky, 1+ = has geometry)
fn sampleTileLightWeighted(tilePos: vec2f, direction: u32) -> vec4f {
    let tileX = i32(floor(tilePos.x));
    let tileY = i32(floor(tilePos.y));
    let fracX = fract(tilePos.x);
    let fracY = fract(tilePos.y);

    // Bilinear weights
    let w_tl = (1.0 - fracX) * (1.0 - fracY);
    let w_tr = fracX * (1.0 - fracY);
    let w_bl = (1.0 - fracX) * fracY;
    let w_br = fracX * fracY;

    // Sample 4 neighboring tiles (RGBA: light + validity weight)
    let tl = getTileLightWithWeight(tileX, tileY, direction);
    let tr = getTileLightWithWeight(tileX + 1, tileY, direction);
    let bl = getTileLightWithWeight(tileX, tileY + 1, direction);
    let br = getTileLightWithWeight(tileX + 1, tileY + 1, direction);

    // Weight-aware bilinear interpolation
    // Use tile weight (.w) to reduce contribution from sky tiles
    let validity_tl = min(tl.w, 1.0);
    let validity_tr = min(tr.w, 1.0);
    let validity_bl = min(bl.w, 1.0);
    let validity_br = min(br.w, 1.0);

    // Combined weights (bilinear * validity)
    let cw_tl = w_tl * validity_tl;
    let cw_tr = w_tr * validity_tr;
    let cw_bl = w_bl * validity_bl;
    let cw_br = w_br * validity_br;
    let totalWeight = cw_tl + cw_tr + cw_bl + cw_br;

    // Weight-averaged light
    var light = vec3f(0.0);
    if (totalWeight > 0.001) {
        light = (tl.rgb * cw_tl + tr.rgb * cw_tr + bl.rgb * cw_bl + br.rgb * cw_br) / totalWeight;
    }

    return vec4f(light, totalWeight);
}

// Sample all 4 directions at a position, weighted by normal
// Returns vec4f: RGB = weighted light, A = total validity weight
fn sampleAllDirectionsWeighted(samplePos: vec2f, leftWeight: f32, rightWeight: f32, upWeight: f32, downWeight: f32, ambient: f32) -> vec4f {
    let fromLeft = sampleTileLightWeighted(samplePos, DIR_LEFT);
    let fromRight = sampleTileLightWeighted(samplePos, DIR_RIGHT);
    let fromUp = sampleTileLightWeighted(samplePos, DIR_UP);
    let fromDown = sampleTileLightWeighted(samplePos, DIR_DOWN);

    // Direction weights
    let dw_left = leftWeight + ambient;
    let dw_right = rightWeight + ambient;
    let dw_up = upWeight + ambient;
    let dw_down = downWeight + ambient;

    // Combine direction weight with tile validity
    let w_left = dw_left * fromLeft.w;
    let w_right = dw_right * fromRight.w;
    let w_up = dw_up * fromUp.w;
    let w_down = dw_down * fromDown.w;
    let totalValidity = fromLeft.w + fromRight.w + fromUp.w + fromDown.w;

    var light = vec3f(0.0);
    light += fromLeft.rgb * dw_left;
    light += fromRight.rgb * dw_right;
    light += fromUp.rgb * dw_up;
    light += fromDown.rgb * dw_down;

    return vec4f(light, totalValidity);
}

// Vogel disk sample position
fn vogelDiskSample(sampleIndex: i32, samplesCount: i32, rotation: f32) -> vec2f {
    let theta = f32(sampleIndex) * GOLDEN_ANGLE + rotation;
    let radius = sqrt((f32(sampleIndex) + 0.5) / f32(samplesCount));
    return vec2f(cos(theta), sin(theta)) * radius;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;
    let halfWidth = uniforms.screenParams.z;
    let halfHeight = uniforms.screenParams.w;
    let fullWidth = uniforms.screenParams.x;
    let fullHeight = uniforms.screenParams.y;
    let pixelCoord = vec2i(i32(uv.x * halfWidth), i32(uv.y * halfHeight));

    // Sample GBuffer normal using UV (works regardless of resolution ratio)
    let fullPixelCoord = vec2i(i32(uv.x * fullWidth), i32(uv.y * fullHeight));
    let normalSample = textureLoad(gbufferNormal, fullPixelCoord, 0).xyz;

    // Skip sky pixels (zero normal)
    let normalLen = length(normalSample);
    if (normalLen < 0.1) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let normal = normalize(normalSample);

    // Convert pixel position to tile position
    let tileSize = uniforms.tileParams.z;

    // Current pixel in full resolution
    let fullPixelX = uv.x * fullWidth;
    let fullPixelY = uv.y * fullHeight;

    // Current tile position (fractional)
    let tilePos = vec2f(fullPixelX / tileSize, fullPixelY / tileSize);

    // Sample radius in tiles (default 2.0)
    let sampleRadius = uniforms.tileParams.w;

    // Convert screen-space normal to 2D directional weights
    // Light FROM a direction hits surfaces FACING toward that direction
    let screenNormalX = normal.x;  // Positive = faces right in screen
    let screenNormalY = -normal.y; // Flip for screen coords (positive = faces down)

    // Calculate directional weights based on normal
    let leftWeight = max(-screenNormalX, 0.0);  // Faces left (toward left light)
    let rightWeight = max(screenNormalX, 0.0);  // Faces right (toward right light)
    let upWeight = max(-screenNormalY, 0.0);    // Faces up in world (toward up light)
    let downWeight = max(screenNormalY, 0.0);   // Faces down in world (toward down light)

    // Ambient contribution from all directions
    let ambient = 0.25;

    // Per-pixel noise for jittering the Vogel disk rotation
    let frameIndex = uniforms.ssgiParams.y;
    let noiseInput = vec2f(f32(pixelCoord.x), f32(pixelCoord.y)) + frameIndex * 0.618;
    let rotation = hash21(noiseInput) * 2.0 * PI;

    // Accumulate irradiance from 16 Vogel disk samples
    var irradiance = vec3f(0.0);
    var totalSampleWeight = 0.0;
    let totalWeight = leftWeight + rightWeight + upWeight + downWeight + ambient * 4.0;

    for (var i = 0; i < SAMPLE_COUNT; i++) {
        // Get Vogel disk offset, scaled by radius
        let diskOffset = vogelDiskSample(i, SAMPLE_COUNT, rotation) * sampleRadius;
        let samplePos = tilePos + diskOffset;

        // Sample all 4 directions at this position
        let sampleResult = sampleAllDirectionsWeighted(samplePos, leftWeight, rightWeight, upWeight, downWeight, ambient);

        // Calculate brightness of this sample
        let sampleBrightness = max(max(sampleResult.rgb.r, sampleResult.rgb.g), sampleResult.rgb.b);

        // Weight by validity from tile data (propagation weights)
        // Tiles with low validity are likely at screen edges or sky
        let validityWeight = smoothstep(0.0, 2.0, sampleResult.w);

        // Also weight by minimum absolute brightness (very dark = likely sky before sky pass)
        let absoluteBrightnessWeight = smoothstep(0.001, 0.02, sampleBrightness);

        // Combined weight
        let sampleWeight = validityWeight * absoluteBrightnessWeight;

        irradiance += sampleResult.rgb * sampleWeight;
        totalSampleWeight += sampleWeight;
    }

    // Average over weighted samples
    if (totalSampleWeight > 0.1) {
        irradiance /= totalSampleWeight;
    }

    // Normalize by directional weights
    if (totalWeight > 0.0) {
        irradiance /= totalWeight;
    }

    // Apply intensity
    let intensity = uniforms.ssgiParams.x;
    irradiance *= intensity;

    // Clamp to prevent fireflies
    let maxBrightness = 4.0;
    let brightness = max(max(irradiance.r, irradiance.g), irradiance.b);
    if (brightness > maxBrightness) {
        irradiance *= maxBrightness / brightness;
    }

    return vec4f(irradiance, 1.0);
}
