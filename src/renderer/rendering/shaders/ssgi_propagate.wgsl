// SSGI Light Propagation Compute Shader
//
// For each tile, collects indirect lighting from all other tiles in 4 directions.
// Attenuation: tiles more than half screen away contribute 0.
//
// Output: 4 vec4f per tile (left, right, up, down directions)
// Each direction stores: RGB = accumulated light from that direction, A = weight

struct TileUniforms {
    screenWidth: f32,
    screenHeight: f32,
    tileCountX: f32,
    tileCountY: f32,
    tileSize: f32,
    emissiveBoost: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> uniforms: TileUniforms;
@group(0) @binding(1) var<storage, read> tileAccumBuffer: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tilePropagateBuffer: array<vec4f>;

// Direction indices
const DIR_LEFT: u32 = 0u;
const DIR_RIGHT: u32 = 1u;
const DIR_UP: u32 = 2u;
const DIR_DOWN: u32 = 3u;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
    let tileX = globalId.x;
    let tileY = globalId.y;

    let tileCountX = u32(uniforms.tileCountX);
    let tileCountY = u32(uniforms.tileCountY);

    // Skip if out of bounds
    if (tileX >= tileCountX || tileY >= tileCountY) {
        return;
    }

    let thisTileIdx = tileY * tileCountX + tileX;

    // Half screen distance in tiles (for attenuation cutoff)
    // Use max of X/Y so vertical propagation has same range as horizontal
    let halfScreenTiles = max(f32(tileCountX), f32(tileCountY)) * 0.5;

    // Accumulators for each direction
    var accumLeft = vec4f(0.0);
    var accumRight = vec4f(0.0);
    var accumUp = vec4f(0.0);
    var accumDown = vec4f(0.0);

    // === Collect light from LEFT (all tiles with smaller X) ===
    // Light traveling RIGHT toward surfaces facing left
    // No self-lighting (dist=0), neighbor (dist=1) contributes 1.0, linear falloff to 0 at half screen
    for (var x = 0u; x < tileX; x++) {
        let dist = f32(tileX - x);
        var weight = 0.0;
        if (dist >= 1.0 && dist < halfScreenTiles) {
            weight = 1.0 - (dist - 1.0) / (halfScreenTiles - 1.0);
            weight = max(weight, 0.0);
        }
        if (weight > 0.0) {
            let srcIdx = tileY * tileCountX + x;
            let srcLight = tileAccumBuffer[srcIdx];
            accumLeft += vec4f(srcLight.rgb * weight, weight);
        }
    }

    // === Collect light from RIGHT (all tiles with larger X) ===
    // Light traveling LEFT toward surfaces facing right
    for (var x = tileX + 1u; x < tileCountX; x++) {
        let dist = f32(x - tileX);
        var weight = 0.0;
        if (dist >= 1.0 && dist < halfScreenTiles) {
            weight = 1.0 - (dist - 1.0) / (halfScreenTiles - 1.0);
            weight = max(weight, 0.0);
        }
        if (weight > 0.0) {
            let srcIdx = tileY * tileCountX + x;
            let srcLight = tileAccumBuffer[srcIdx];
            accumRight += vec4f(srcLight.rgb * weight, weight);
        }
    }

    // === Collect light from UP (all tiles with smaller Y) ===
    // Light traveling DOWN toward surfaces facing up
    for (var y = 0u; y < tileY; y++) {
        let dist = f32(tileY - y);
        var weight = 0.0;
        if (dist >= 1.0 && dist < halfScreenTiles) {
            weight = 1.0 - (dist - 1.0) / (halfScreenTiles - 1.0);
            weight = max(weight, 0.0);
        }
        if (weight > 0.0) {
            let srcIdx = y * tileCountX + tileX;
            let srcLight = tileAccumBuffer[srcIdx];
            accumUp += vec4f(srcLight.rgb * weight, weight);
        }
    }

    // === Collect light from DOWN (all tiles with larger Y) ===
    // Light traveling UP toward surfaces facing down
    for (var y = tileY + 1u; y < tileCountY; y++) {
        let dist = f32(y - tileY);
        var weight = 0.0;
        if (dist >= 1.0 && dist < halfScreenTiles) {
            weight = 1.0 - (dist - 1.0) / (halfScreenTiles - 1.0);
            weight = max(weight, 0.0);
        }
        if (weight > 0.0) {
            let srcIdx = y * tileCountX + tileX;
            let srcLight = tileAccumBuffer[srcIdx];
            accumDown += vec4f(srcLight.rgb * weight, weight);
        }
    }

    // Normalize accumulated light by weight
    if (accumLeft.w > 0.0) { accumLeft = vec4f(accumLeft.rgb / accumLeft.w, accumLeft.w); }
    if (accumRight.w > 0.0) { accumRight = vec4f(accumRight.rgb / accumRight.w, accumRight.w); }
    if (accumUp.w > 0.0) { accumUp = vec4f(accumUp.rgb / accumUp.w, accumUp.w); }
    if (accumDown.w > 0.0) { accumDown = vec4f(accumDown.rgb / accumDown.w, accumDown.w); }

    // Write to propagate buffer (4 directions per tile)
    let baseIdx = thisTileIdx * 4u;
    tilePropagateBuffer[baseIdx + DIR_LEFT] = accumLeft;
    tilePropagateBuffer[baseIdx + DIR_RIGHT] = accumRight;
    tilePropagateBuffer[baseIdx + DIR_UP] = accumUp;
    tilePropagateBuffer[baseIdx + DIR_DOWN] = accumDown;
}
