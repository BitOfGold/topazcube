// Light culling compute shader for tiled deferred lighting
// Divides screen into tiles and determines which lights affect each tile

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 256u;
const MAX_LIGHTS: u32 = 512u;

struct Light {
    enabled: u32,
    position: vec3f,
    color: vec4f,
    direction: vec3f,
    geom: vec4f,     // x = radius, y = inner cone, z = outer cone
    shadowIndex: i32,
}

struct LightCullUniforms {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
    inverseProjection: mat4x4f,
    screenSize: vec2f,
    tileCount: vec2u,
    lightCount: u32,
    nearPlane: f32,
    farPlane: f32,
    padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: LightCullUniforms;
@group(0) @binding(1) var<storage, read> lights: array<Light, MAX_LIGHTS>;
@group(0) @binding(2) var<storage, read_write> tileLightIndices: array<u32>;
@group(0) @binding(3) var gDepth: texture_depth_2d;

// Get view-space frustum planes for a tile
// Returns 4 planes (left, right, bottom, top) in view space
fn getTileFrustumPlanes(tileX: u32, tileY: u32, tileCount: vec2u) -> array<vec4f, 4> {
    var planes: array<vec4f, 4>;

    // Calculate tile bounds in NDC space [-1, 1]
    let tileMinX = f32(tileX) / f32(tileCount.x) * 2.0 - 1.0;
    let tileMaxX = f32(tileX + 1u) / f32(tileCount.x) * 2.0 - 1.0;
    let tileMinY = f32(tileY) / f32(tileCount.y) * 2.0 - 1.0;
    let tileMaxY = f32(tileY + 1u) / f32(tileCount.y) * 2.0 - 1.0;

    // Create planes from NDC frustum edges
    // Each plane normal points inward

    // Left plane: normal = (1, 0, -tileMinX)
    let leftNormal = normalize(vec3f(1.0, 0.0, -tileMinX));
    planes[0] = vec4f(leftNormal, 0.0);

    // Right plane: normal = (-1, 0, tileMaxX)
    let rightNormal = normalize(vec3f(-1.0, 0.0, tileMaxX));
    planes[1] = vec4f(rightNormal, 0.0);

    // Bottom plane: normal = (0, 1, -tileMinY)
    let bottomNormal = normalize(vec3f(0.0, 1.0, -tileMinY));
    planes[2] = vec4f(bottomNormal, 0.0);

    // Top plane: normal = (0, -1, tileMaxY)
    let topNormal = normalize(vec3f(0.0, -1.0, tileMaxY));
    planes[3] = vec4f(topNormal, 0.0);

    return planes;
}

// Test if a sphere intersects with a frustum plane
fn sphereInsidePlane(center: vec3f, radius: f32, plane: vec4f) -> bool {
    let dist = dot(plane.xyz, center) + plane.w;
    return dist >= -radius;
}

// Test if a light affects a tile
fn lightAffectsTile(lightIndex: u32, tileX: u32, tileY: u32, minDepth: f32, maxDepth: f32) -> bool {
    let light = lights[lightIndex];

    if (light.enabled == 0u) {
        return false;
    }

    let lightRadius = light.geom.x;
    if (lightRadius <= 0.0) {
        return false;
    }

    // Transform light position to view space
    let lightPosView = uniforms.viewMatrix * vec4f(light.position, 1.0);
    let lightCenter = lightPosView.xyz;

    // View-space depth (positive = in front of camera)
    let lightDepth = -lightCenter.z;

    // Skip lights that are entirely behind the camera
    if (lightDepth + lightRadius < 0.0) {
        return false;
    }

    // Quick depth test: check if light sphere overlaps tile depth range
    // Only cull if entirely in front of near plane (behind camera already handled above)
    let lightMinZ = lightDepth - lightRadius;
    if (lightMinZ > maxDepth) {
        return false;
    }

    // Project light center to clip space for tile bounds test
    let lightPosClip = uniforms.projectionMatrix * lightPosView;

    // Handle lights very close to or behind camera - use conservative estimate
    let w = lightPosClip.w;
    if (w < 0.1) {
        // Light is very close to or behind camera - include in all relevant tiles
        // Just do a simple distance check instead
        return true;
    }

    let lightPosNDC = lightPosClip.xyz / w;

    // Calculate tile bounds in NDC
    let tileCount = uniforms.tileCount;
    let tileMinX = f32(tileX) / f32(tileCount.x) * 2.0 - 1.0;
    let tileMaxX = f32(tileX + 1u) / f32(tileCount.x) * 2.0 - 1.0;
    let tileMinY = f32(tileY) / f32(tileCount.y) * 2.0 - 1.0;
    let tileMaxY = f32(tileY + 1u) / f32(tileCount.y) * 2.0 - 1.0;

    // Minimum radius in NDC - ensure lights are assigned to at least their containing tile
    // Scale minimum with depth to handle grazing angle views where distant lights project thin
    let tileWidthNDC = 2.0 / f32(tileCount.x);
    let tileHeightNDC = 2.0 / f32(tileCount.y);
    let depthFactor = 1.0 + lightDepth * 0.002; // Grow minimum radius with distance
    let minRadiusX = tileWidthNDC * depthFactor;
    let minRadiusY = tileHeightNDC * depthFactor;

    // Approximate radius in NDC - use separate X and Y scales due to aspect ratio
    // Add 1.5x multiplier to account for perspective distortion at screen edges
    let depthScale = 1.0 / max(lightDepth, 0.1);
    let radiusNDC_X = max(lightRadius * depthScale * uniforms.projectionMatrix[0][0] * 1.5, minRadiusX);
    let radiusNDC_Y = max(lightRadius * depthScale * uniforms.projectionMatrix[1][1] * 1.5, minRadiusY);

    // AABB test: check if light sphere (in NDC) overlaps tile bounds
    if (lightPosNDC.x + radiusNDC_X < tileMinX || lightPosNDC.x - radiusNDC_X > tileMaxX) {
        return false;
    }
    if (lightPosNDC.y + radiusNDC_Y < tileMinY || lightPosNDC.y - radiusNDC_Y > tileMaxY) {
        return false;
    }

    return true;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u, @builtin(workgroup_id) workgroupId: vec3u) {
    let tileX = workgroupId.x;
    let tileY = workgroupId.y;
    let tileCount = uniforms.tileCount;

    // Bounds check
    if (tileX >= tileCount.x || tileY >= tileCount.y) {
        return;
    }

    let tileIndex = tileY * tileCount.x + tileX;
    let tileDataOffset = tileIndex * (MAX_LIGHTS_PER_TILE + 1u);

    // Local thread coordinates within the tile
    let localX = globalId.x % TILE_SIZE;
    let localY = globalId.y % TILE_SIZE;
    let localIndex = localY * TILE_SIZE + localX;

    // Calculate pixel coordinates for this thread
    let pixelX = tileX * TILE_SIZE + localX;
    let pixelY = tileY * TILE_SIZE + localY;

    // Sample depth at this pixel (for min/max depth calculation)
    var depth = 1.0;
    if (pixelX < u32(uniforms.screenSize.x) && pixelY < u32(uniforms.screenSize.y)) {
        depth = textureLoad(gDepth, vec2i(i32(pixelX), i32(pixelY)), 0);
    }

    // Use shared memory for min/max depth reduction
    // For simplicity, we'll use a conservative estimate here
    // In a real implementation, you'd use workgroup shared memory for reduction

    // Convert depth to view-space Z
    let minDepth = uniforms.nearPlane;
    let maxDepth = uniforms.farPlane;

    // Only thread 0 of each tile does the light culling
    if (localIndex == 0u) {
        var lightCount = 0u;

        for (var i = 0u; i < uniforms.lightCount && i < MAX_LIGHTS; i++) {
            if (lightAffectsTile(i, tileX, tileY, minDepth, maxDepth)) {
                if (lightCount < MAX_LIGHTS_PER_TILE) {
                    // Store light index (offset by 1 to leave room for count)
                    tileLightIndices[tileDataOffset + 1u + lightCount] = i;
                    lightCount++;
                }
            }
        }

        // Store light count at the start of tile data
        tileLightIndices[tileDataOffset] = lightCount;
    }
}
