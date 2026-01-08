// SSGI Tile Accumulation Compute Shader
//
// For each tile, accumulates the average light from:
// - Previous frame HDR (lit surfaces)
// - Emissive texture (boosted)
//
// Output: Single vec4f per tile (RGB = average color, A = count)

struct TileUniforms {
    screenWidth: f32,
    screenHeight: f32,
    tileCountX: f32,
    tileCountY: f32,
    tileSize: f32,
    emissiveBoost: f32,
    maxBrightness: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> uniforms: TileUniforms;
@group(0) @binding(1) var prevHDR: texture_2d<f32>;
@group(0) @binding(2) var emissiveTexture: texture_2d<f32>;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var<storage, read_write> tileAccumBuffer: array<vec4f>;

const WORKGROUP_SIZE: u32 = 8u;
const PIXELS_PER_THREAD: u32 = 8u;
const TILE_SIZE: u32 = 64u;

// Shared memory for parallel reduction (64 threads × 4 floats)
var<workgroup> sharedAccum: array<vec4f, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(local_invocation_id) localId: vec3u,
    @builtin(workgroup_id) workgroupId: vec3u,
) {
    let tileX = workgroupId.x;
    let tileY = workgroupId.y;
    let threadIdx = localId.y * WORKGROUP_SIZE + localId.x;

    // Base pixel position for this tile
    let tileBaseX = tileX * TILE_SIZE;
    let tileBaseY = tileY * TILE_SIZE;

    // This thread processes an 8×8 block of pixels
    let blockBaseX = tileBaseX + localId.x * PIXELS_PER_THREAD;
    let blockBaseY = tileBaseY + localId.y * PIXELS_PER_THREAD;

    // Local accumulator
    var localAccum = vec4f(0.0);

    // Process 8×8 pixels
    for (var py = 0u; py < PIXELS_PER_THREAD; py++) {
        for (var px = 0u; px < PIXELS_PER_THREAD; px++) {
            let pixelX = blockBaseX + px;
            let pixelY = blockBaseY + py;

            // Skip pixels outside screen bounds
            if (pixelX >= u32(uniforms.screenWidth) || pixelY >= u32(uniforms.screenHeight)) {
                continue;
            }

            let pixelCoord = vec2i(i32(pixelX), i32(pixelY));

            // Sample previous frame HDR
            let hdrColor = textureLoad(prevHDR, pixelCoord, 0).rgb;

            // Sample emissive and boost
            let emissive = textureLoad(emissiveTexture, pixelCoord, 0).rgb;
            let boostedEmissive = emissive * uniforms.emissiveBoost;

            // Combine HDR + boosted emissive
            var totalLight = hdrColor + boostedEmissive;

            // Clamp brightness to maxBrightness (excludes specular highlights)
            // Use max RGB instead of luminance to treat all colors equally
            let bright = max(max(totalLight.r, totalLight.g), totalLight.b);
            if (bright > uniforms.maxBrightness) {
                totalLight *= uniforms.maxBrightness / bright;
            }

            // Accumulate
            localAccum += vec4f(totalLight, 1.0);
        }
    }

    // Store to shared memory
    sharedAccum[threadIdx] = localAccum;

    workgroupBarrier();

    // Parallel reduction - 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (threadIdx < stride) {
            sharedAccum[threadIdx] += sharedAccum[threadIdx + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes final averaged result to tile buffer
    if (threadIdx == 0u) {
        let tileIndex = tileY * u32(uniforms.tileCountX) + tileX;
        let accum = sharedAccum[0];

        // Average the accumulated light
        var avgColor = vec4f(0.0);
        if (accum.w > 0.0) {
            avgColor = vec4f(accum.rgb / accum.w, accum.w);
        }

        tileAccumBuffer[tileIndex] = avgColor;
    }
}
