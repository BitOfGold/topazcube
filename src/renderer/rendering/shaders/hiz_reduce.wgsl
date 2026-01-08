// HiZ Reduce Compute Shader
// Reduces the depth buffer to MIN and MAX depth per 64x64 tile for occlusion culling
// MIN depth = closest geometry in tile (occluder surface)
// MAX depth = farthest geometry in tile (if < 1.0, tile is fully covered)
// Tile thickness = MAX - MIN (thin for walls, thick for angled ground)

struct Uniforms {
    screenWidth: f32,
    screenHeight: f32,
    tileCountX: f32,
    tileCountY: f32,
    tileSize: f32,
    near: f32,
    far: f32,
    clearValue: f32,  // 1.0 to clear, 0.0 to accumulate
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depthTexture: texture_depth_2d;
@group(0) @binding(2) var<storage, read_write> hizBuffer: array<f32>;

// Workgroup size: 8x8 threads, each thread processes 8x8 pixels = 64x64 per workgroup
var<workgroup> sharedMinDepth: array<f32, 64>;  // 8x8 = 64 threads
var<workgroup> sharedMaxDepth: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) globalId: vec3u,
    @builtin(local_invocation_id) localId: vec3u,
    @builtin(workgroup_id) workgroupId: vec3u
) {
    let tileX = workgroupId.x;
    let tileY = workgroupId.y;
    let tileIndex = tileY * u32(uniforms.tileCountX) + tileX;

    // Check if this tile is within bounds
    if (tileX >= u32(uniforms.tileCountX) || tileY >= u32(uniforms.tileCountY)) {
        return;
    }

    // Each thread processes an 8x8 block within the 64x64 tile
    let localIndex = localId.y * 8u + localId.x;
    let blockStartX = tileX * 64u + localId.x * 8u;
    let blockStartY = tileY * 64u + localId.y * 8u;

    // Find MIN and MAX depth in this thread's 8x8 block
    var minDepth: f32 = 1.0;  // Start at far plane
    var maxDepth: f32 = 0.0;  // Start at near plane

    for (var y = 0u; y < 8u; y = y + 1u) {
        for (var x = 0u; x < 8u; x = x + 1u) {
            let pixelX = blockStartX + x;
            let pixelY = blockStartY + y;

            // Skip pixels outside screen bounds
            if (pixelX < u32(uniforms.screenWidth) && pixelY < u32(uniforms.screenHeight)) {
                let depth = textureLoad(depthTexture, vec2u(pixelX, pixelY), 0);
                minDepth = min(minDepth, depth);
                maxDepth = max(maxDepth, depth);
            }
        }
    }

    // Store in shared memory
    sharedMinDepth[localIndex] = minDepth;
    sharedMaxDepth[localIndex] = maxDepth;
    workgroupBarrier();

    // Parallel reduction in shared memory
    // Step 1: 64 -> 32
    if (localIndex < 32u) {
        sharedMinDepth[localIndex] = min(sharedMinDepth[localIndex], sharedMinDepth[localIndex + 32u]);
        sharedMaxDepth[localIndex] = max(sharedMaxDepth[localIndex], sharedMaxDepth[localIndex + 32u]);
    }
    workgroupBarrier();

    // Step 2: 32 -> 16
    if (localIndex < 16u) {
        sharedMinDepth[localIndex] = min(sharedMinDepth[localIndex], sharedMinDepth[localIndex + 16u]);
        sharedMaxDepth[localIndex] = max(sharedMaxDepth[localIndex], sharedMaxDepth[localIndex + 16u]);
    }
    workgroupBarrier();

    // Step 3: 16 -> 8
    if (localIndex < 8u) {
        sharedMinDepth[localIndex] = min(sharedMinDepth[localIndex], sharedMinDepth[localIndex + 8u]);
        sharedMaxDepth[localIndex] = max(sharedMaxDepth[localIndex], sharedMaxDepth[localIndex + 8u]);
    }
    workgroupBarrier();

    // Step 4: 8 -> 4
    if (localIndex < 4u) {
        sharedMinDepth[localIndex] = min(sharedMinDepth[localIndex], sharedMinDepth[localIndex + 4u]);
        sharedMaxDepth[localIndex] = max(sharedMaxDepth[localIndex], sharedMaxDepth[localIndex + 4u]);
    }
    workgroupBarrier();

    // Step 5: 4 -> 2
    if (localIndex < 2u) {
        sharedMinDepth[localIndex] = min(sharedMinDepth[localIndex], sharedMinDepth[localIndex + 2u]);
        sharedMaxDepth[localIndex] = max(sharedMaxDepth[localIndex], sharedMaxDepth[localIndex + 2u]);
    }
    workgroupBarrier();

    // Step 6: 2 -> 1 (final result)
    if (localIndex == 0u) {
        let finalMinDepth = min(sharedMinDepth[0], sharedMinDepth[1]);
        let finalMaxDepth = max(sharedMaxDepth[0], sharedMaxDepth[1]);

        // Store both min and max depth (2 floats per tile, interleaved)
        hizBuffer[tileIndex * 2u] = finalMinDepth;
        hizBuffer[tileIndex * 2u + 1u] = finalMaxDepth;
    }
}
