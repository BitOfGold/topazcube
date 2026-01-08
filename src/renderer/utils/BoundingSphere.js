import { vec3 } from "../math.js"

/**
 * Calculate a bounding sphere from position data
 * Uses Ritter's bounding sphere algorithm for a good approximation
 *
 * @param {Float32Array|Array} positions - Position data (x, y, z, x, y, z, ...)
 * @returns {{ center: [number, number, number], radius: number }}
 */
function calculateBoundingSphere(positions) {
    if (!positions || positions.length < 3) {
        return { center: [0, 0, 0], radius: 0 }
    }

    const vertexCount = Math.floor(positions.length / 3)

    // Step 1: Find the centroid (average of all points)
    let cx = 0, cy = 0, cz = 0
    for (let i = 0; i < vertexCount; i++) {
        cx += positions[i * 3]
        cy += positions[i * 3 + 1]
        cz += positions[i * 3 + 2]
    }
    cx /= vertexCount
    cy /= vertexCount
    cz /= vertexCount

    // Step 2: Find the point farthest from the centroid
    let maxDistSq = 0
    let farthestIdx = 0
    for (let i = 0; i < vertexCount; i++) {
        const dx = positions[i * 3] - cx
        const dy = positions[i * 3 + 1] - cy
        const dz = positions[i * 3 + 2] - cz
        const distSq = dx * dx + dy * dy + dz * dz
        if (distSq > maxDistSq) {
            maxDistSq = distSq
            farthestIdx = i
        }
    }

    // Step 3: Find the point farthest from that point
    let p1x = positions[farthestIdx * 3]
    let p1y = positions[farthestIdx * 3 + 1]
    let p1z = positions[farthestIdx * 3 + 2]

    maxDistSq = 0
    let oppositeIdx = 0
    for (let i = 0; i < vertexCount; i++) {
        const dx = positions[i * 3] - p1x
        const dy = positions[i * 3 + 1] - p1y
        const dz = positions[i * 3 + 2] - p1z
        const distSq = dx * dx + dy * dy + dz * dz
        if (distSq > maxDistSq) {
            maxDistSq = distSq
            oppositeIdx = i
        }
    }

    let p2x = positions[oppositeIdx * 3]
    let p2y = positions[oppositeIdx * 3 + 1]
    let p2z = positions[oppositeIdx * 3 + 2]

    // Step 4: Initial sphere from these two extreme points
    let sphereCx = (p1x + p2x) * 0.5
    let sphereCy = (p1y + p2y) * 0.5
    let sphereCz = (p1z + p2z) * 0.5
    let radius = Math.sqrt(maxDistSq) * 0.5

    // Step 5: Ritter's expansion - grow sphere to include all points
    for (let i = 0; i < vertexCount; i++) {
        const px = positions[i * 3]
        const py = positions[i * 3 + 1]
        const pz = positions[i * 3 + 2]

        const dx = px - sphereCx
        const dy = py - sphereCy
        const dz = pz - sphereCz
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)

        if (dist > radius) {
            // Point is outside sphere, expand to include it
            const newRadius = (radius + dist) * 0.5
            const ratio = (newRadius - radius) / dist

            sphereCx += dx * ratio
            sphereCy += dy * ratio
            sphereCz += dz * ratio
            radius = newRadius
        }
    }

    // Add a small epsilon to avoid floating-point issues
    radius *= 1.001

    return {
        center: [sphereCx, sphereCy, sphereCz],
        radius: radius
    }
}

/**
 * Calculate axis-aligned bounding box from positions
 * @param {Float32Array|Array} positions - Position data
 * @returns {{ min: [number, number, number], max: [number, number, number] }}
 */
function calculateAABB(positions) {
    if (!positions || positions.length < 3) {
        return { min: [0, 0, 0], max: [0, 0, 0] }
    }

    const vertexCount = Math.floor(positions.length / 3)

    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

    for (let i = 0; i < vertexCount; i++) {
        const x = positions[i * 3]
        const y = positions[i * 3 + 1]
        const z = positions[i * 3 + 2]

        if (x < minX) minX = x
        if (y < minY) minY = y
        if (z < minZ) minZ = z
        if (x > maxX) maxX = x
        if (y > maxY) maxY = y
        if (z > maxZ) maxZ = z
    }

    return {
        min: [minX, minY, minZ],
        max: [maxX, maxY, maxZ]
    }
}

/**
 * Create bounding sphere from AABB
 */
function boundingSphereFromAABB(aabb) {
    const center = [
        (aabb.min[0] + aabb.max[0]) * 0.5,
        (aabb.min[1] + aabb.max[1]) * 0.5,
        (aabb.min[2] + aabb.max[2]) * 0.5
    ]

    const dx = aabb.max[0] - aabb.min[0]
    const dy = aabb.max[1] - aabb.min[1]
    const dz = aabb.max[2] - aabb.min[2]
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5

    return { center, radius }
}

/**
 * Merge two bounding spheres
 */
function mergeBoundingSpheres(a, b) {
    const dx = b.center[0] - a.center[0]
    const dy = b.center[1] - a.center[1]
    const dz = b.center[2] - a.center[2]
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)

    // Check if one sphere contains the other
    if (dist + b.radius <= a.radius) {
        return { center: [...a.center], radius: a.radius }
    }
    if (dist + a.radius <= b.radius) {
        return { center: [...b.center], radius: b.radius }
    }

    // Calculate new sphere that contains both
    const newRadius = (dist + a.radius + b.radius) * 0.5
    const ratio = (newRadius - a.radius) / dist

    return {
        center: [
            a.center[0] + dx * ratio,
            a.center[1] + dy * ratio,
            a.center[2] + dz * ratio
        ],
        radius: newRadius
    }
}

/**
 * Transform a bounding sphere by a matrix
 * @param {Object} bsphere - Bounding sphere
 * @param {mat4} matrix - Transform matrix
 * @returns {Object} Transformed bounding sphere
 */
function transformBoundingSphere(bsphere, matrix) {
    // Transform center
    const center = vec3.create()
    vec3.transformMat4(center, bsphere.center, matrix)

    // Get scale factor from matrix (approximate as max axis scale)
    const sx = Math.sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1] + matrix[2] * matrix[2])
    const sy = Math.sqrt(matrix[4] * matrix[4] + matrix[5] * matrix[5] + matrix[6] * matrix[6])
    const sz = Math.sqrt(matrix[8] * matrix[8] + matrix[9] * matrix[9] + matrix[10] * matrix[10])
    const maxScale = Math.max(sx, sy, sz)

    return {
        center: [center[0], center[1], center[2]],
        radius: bsphere.radius * maxScale
    }
}

/**
 * Test if a bounding sphere is visible in a frustum
 * @param {Object} bsphere - Bounding sphere { center, radius }
 * @param {Float32Array} planes - Frustum planes (6 vec4s)
 * @returns {boolean} True if visible
 */
function sphereInFrustum(bsphere, planes) {
    for (let i = 0; i < 6; i++) {
        const offset = i * 4
        const nx = planes[offset]
        const ny = planes[offset + 1]
        const nz = planes[offset + 2]
        const d = planes[offset + 3]

        const dist = nx * bsphere.center[0] + ny * bsphere.center[1] + nz * bsphere.center[2] + d

        if (dist < -bsphere.radius) {
            return false // Completely outside this plane
        }
    }
    return true
}

/**
 * Test if a bounding sphere is within a distance from a point
 * @param {Object} bsphere - Bounding sphere
 * @param {Array} point - Test point [x, y, z]
 * @param {number} maxDistance - Maximum distance
 * @returns {boolean} True if within distance
 */
function sphereWithinDistance(bsphere, point, maxDistance) {
    const dx = bsphere.center[0] - point[0]
    const dy = bsphere.center[1] - point[1]
    const dz = bsphere.center[2] - point[2]
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) - bsphere.radius
    return dist <= maxDistance
}

/**
 * Calculate a "shadow bounding sphere" that encompasses both the object
 * and its shadow projection on the ground plane.
 *
 * This is used for shadow map culling optimization - we can cull objects
 * whose shadow spheres are not visible to the camera (frustum/occlusion).
 *
 * The algorithm:
 * 1. Start with the object's world-space bounding sphere
 * 2. Project the sphere onto the ground plane along the light direction
 *    - This creates an ellipse on the ground
 * 3. Calculate a bounding sphere that contains both the object sphere
 *    and the shadow ellipse
 *
 * @param {Object} bsphere - Object's bounding sphere { center: [x,y,z], radius: r }
 * @param {Array} lightDir - Normalized light direction vector [x,y,z] (pointing TO light)
 * @param {number} groundLevel - Y coordinate of the ground/shadow receiver plane
 * @returns {Object} Shadow bounding sphere { center: [x,y,z], radius: r }
 */
function calculateShadowBoundingSphere(bsphere, lightDir, groundLevel = 0) {
    // Light direction should point FROM object TO light (i.e., towards the sky for sun)
    // If Y component is positive, light comes from above
    const lx = lightDir[0]
    const ly = lightDir[1]
    const lz = lightDir[2]

    // Object sphere properties
    const cx = bsphere.center[0]
    const cy = bsphere.center[1]
    const cz = bsphere.center[2]
    const r = bsphere.radius

    // If object is below ground level, its shadow is above it (not on ground)
    // In that case, just return the original sphere with some margin
    if (cy + r < groundLevel) {
        return {
            center: [...bsphere.center],
            radius: r * 1.1
        }
    }

    // Calculate the height of sphere center above ground
    const heightAboveGround = cy - groundLevel

    // If light is nearly horizontal (ly close to 0), shadow extends very far
    // Clamp to a reasonable maximum shadow distance
    const maxShadowDistance = 100

    // Calculate shadow center on ground:
    // The center of the sphere projects along light direction to ground
    // shadowPos = center + t * (-lightDir)  where t = heightAboveGround / ly
    let shadowCenterX, shadowCenterZ

    if (Math.abs(ly) < 0.01) {
        // Nearly horizontal light - shadow extends very far
        // Use a reasonable estimate: shadow at maxShadowDistance in light direction
        const horizLen = Math.sqrt(lx * lx + lz * lz)
        if (horizLen > 0.001) {
            shadowCenterX = cx - (lx / horizLen) * maxShadowDistance
            shadowCenterZ = cz - (lz / horizLen) * maxShadowDistance
        } else {
            shadowCenterX = cx
            shadowCenterZ = cz
        }
    } else {
        // Normal case: calculate projection along light ray
        const t = heightAboveGround / ly
        // Clamp shadow distance
        const clampedT = Math.min(Math.abs(t), maxShadowDistance) * Math.sign(t)
        shadowCenterX = cx - lx * clampedT
        shadowCenterZ = cz - lz * clampedT
    }

    // The shadow of a sphere is an ellipse when light is not vertical
    // The ellipse's major axis extends along the light direction
    // For simplicity, we approximate the shadow as a circle with radius
    // that encompasses the ellipse

    // Shadow radius depends on the angle of the light
    // When light is vertical (ly=1), shadow radius = object radius
    // When light is at an angle, shadow stretches along the light direction
    const cosAngle = Math.abs(ly)
    const sinAngle = Math.sqrt(1 - cosAngle * cosAngle)

    // Shadow elongation factor (how much the shadow stretches)
    // When light angle is 45°, shadow is ~1.4x longer
    // When light angle is 30° (from horizon), shadow is ~2x longer
    const elongation = cosAngle > 0.01 ? 1 / cosAngle : maxShadowDistance / r
    const clampedElongation = Math.min(elongation, maxShadowDistance / Math.max(r, 0.1))

    // Shadow "radius" (bounding circle of the ellipse)
    const shadowRadius = r * Math.max(1, clampedElongation)

    // Now we have two spheres to encompass:
    // 1. Object sphere: center=(cx, cy, cz), radius=r
    // 2. Shadow "sphere" on ground: center=(shadowCenterX, groundLevel, shadowCenterZ), radius=shadowRadius

    // Create a sphere that contains both
    // Method: find the enclosing sphere of two spheres
    const s1 = {
        center: [cx, cy, cz],
        radius: r
    }
    const s2 = {
        center: [shadowCenterX, groundLevel, shadowCenterZ],
        radius: shadowRadius
    }

    // Distance between sphere centers
    const dx = s2.center[0] - s1.center[0]
    const dy = s2.center[1] - s1.center[1]
    const dz = s2.center[2] - s1.center[2]
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)

    // Check containment
    if (dist + s2.radius <= s1.radius) {
        // Shadow is inside object sphere (unlikely but handle it)
        return { center: [...s1.center], radius: s1.radius * 1.01 }
    }
    if (dist + s1.radius <= s2.radius) {
        // Object is inside shadow sphere
        return { center: [...s2.center], radius: s2.radius * 1.01 }
    }

    // Calculate enclosing sphere
    const newRadius = (dist + s1.radius + s2.radius) * 0.5

    // Center is along the line between sphere centers
    // Position it so both spheres fit
    const ratio = dist > 0.001 ? (newRadius - s1.radius) / dist : 0.5

    return {
        center: [
            s1.center[0] + dx * ratio,
            s1.center[1] + dy * ratio,
            s1.center[2] + dz * ratio
        ],
        radius: newRadius * 1.01 // Small margin
    }
}

/**
 * Test if a bounding sphere intersects a cascade's orthographic box
 *
 * @param {Object} bsphere - Bounding sphere { center: [x,y,z], radius: r }
 * @param {mat4} cascadeMatrix - Cascade's light view-projection matrix
 * @returns {boolean} True if sphere intersects the cascade box
 */
function sphereInCascade(bsphere, cascadeMatrix) {
    // Transform sphere center to cascade clip space
    const cx = bsphere.center[0]
    const cy = bsphere.center[1]
    const cz = bsphere.center[2]

    // Apply cascade view-projection matrix to center
    // clipPos = cascadeMatrix * vec4(center, 1)
    const m = cascadeMatrix
    const w = m[3] * cx + m[7] * cy + m[11] * cz + m[15]

    if (Math.abs(w) < 0.0001) return true // Degenerate case, include it

    const invW = 1.0 / w
    const clipX = (m[0] * cx + m[4] * cy + m[8] * cz + m[12]) * invW
    const clipY = (m[1] * cx + m[5] * cy + m[9] * cz + m[13]) * invW
    const clipZ = (m[2] * cx + m[6] * cy + m[10] * cz + m[14]) * invW

    // Get scale factor for radius (approximate as max axis scale)
    // For orthographic projections, this is simpler
    const sx = Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
    const sy = Math.sqrt(m[4] * m[4] + m[5] * m[5] + m[6] * m[6])
    const sz = Math.sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10])
    const maxScale = Math.max(sx, sy, sz)
    const clipRadius = bsphere.radius * maxScale * invW

    // NDC box is [-1, 1] for X/Y, [0, 1] for Z in WebGPU
    // Check if sphere (with radius) intersects this box
    if (clipX + clipRadius < -1 || clipX - clipRadius > 1) return false
    if (clipY + clipRadius < -1 || clipY - clipRadius > 1) return false
    if (clipZ + clipRadius < 0 || clipZ - clipRadius > 1) return false

    return true
}

export {
    calculateBoundingSphere,
    calculateAABB,
    boundingSphereFromAABB,
    mergeBoundingSpheres,
    transformBoundingSphere,
    sphereInFrustum,
    sphereWithinDistance,
    calculateShadowBoundingSphere,
    sphereInCascade
}
