/**
 * Raycaster - Async ray intersection testing
 *
 * Performs ray-geometry intersection tests without blocking rendering.
 * Uses a Web Worker for heavy triangle intersection calculations.
 *
 * Usage:
 *   const raycaster = new Raycaster(engine)
 *   await raycaster.initialize()
 *
 *   raycaster.cast(origin, direction, maxDistance, (result) => {
 *     if (result.hit) {
 *       console.log('Hit:', result.entity, 'at distance:', result.distance)
 *     }
 *   })
 */

import { vec3, mat4 } from "../math.js"

class Raycaster {
    constructor(engine) {
        this.engine = engine
        this.worker = null
        this._pendingCallbacks = new Map()
        this._nextRequestId = 0
        this._initialized = false
    }

    async initialize() {
        // Create worker from inline code (avoids separate file issues with bundlers)
        const workerCode = this._getWorkerCode()
        const blob = new Blob([workerCode], { type: 'application/javascript' })
        const workerUrl = URL.createObjectURL(blob)

        this.worker = new Worker(workerUrl)
        this.worker.onmessage = this._handleWorkerMessage.bind(this)
        this.worker.onerror = (e) => console.error('Raycaster worker error:', e)

        this._initialized = true
        URL.revokeObjectURL(workerUrl)
    }

    /**
     * Cast a ray and get the closest intersection
     * @param {Array|vec3} origin - Ray start point [x, y, z]
     * @param {Array|vec3} direction - Ray direction (will be normalized)
     * @param {number} maxDistance - Maximum ray length
     * @param {Function} callback - Called with result: { hit, distance, point, normal, entity, mesh, triangleIndex }
     * @param {Object} options - Optional settings
     * @param {Array} options.entities - Specific entities to test (default: all scene entities)
     * @param {Array} options.meshes - Specific meshes to test (default: all scene meshes)
     * @param {boolean} options.backfaces - Test backfaces (default: false)
     * @param {Array} options.exclude - Entities/meshes to exclude
     */
    cast(origin, direction, maxDistance, callback, options = {}) {
        if (!this._initialized) {
            console.warn('Raycaster not initialized')
            callback({ hit: false, error: 'not initialized' })
            return
        }

        const ray = {
            origin: Array.from(origin),
            direction: this._normalize(Array.from(direction)),
            maxDistance
        }

        // Collect candidates from scene
        const candidates = this._collectCandidates(ray, options)

        if (candidates.length === 0) {
            callback({ hit: false })
            return
        }

        // Send to worker for triangle intersection
        const requestId = this._nextRequestId++
        this._pendingCallbacks.set(requestId, { callback, candidates })

        this.worker.postMessage({
            type: 'raycast',
            requestId,
            ray,
            debug: options.debug ?? false,
            candidates: candidates.map(c => ({
                id: c.id,
                vertices: c.vertices,
                indices: c.indices,
                matrix: c.matrix,
                backfaces: options.backfaces ?? false
            }))
        })
    }

    /**
     * Cast a ray upward from a position to check for sky visibility
     * Useful for determining if camera is under cover
     * @param {Array|vec3} position - Position to test from
     * @param {number} maxDistance - How far to check (default: 100)
     * @param {Function} callback - Called with { hitSky: boolean, distance?: number, entity?: object }
     */
    castToSky(position, maxDistance, callback) {
        this.cast(
            position,
            [0, 1, 0], // Straight up
            maxDistance ?? 100,
            (result) => {
                callback({
                    hitSky: !result.hit,
                    distance: result.distance,
                    entity: result.entity,
                    mesh: result.mesh
                })
            }
        )
    }

    /**
     * Cast a ray from screen coordinates (mouse picking)
     * @param {number} screenX - Screen X coordinate
     * @param {number} screenY - Screen Y coordinate
     * @param {Object} camera - Camera with projection/view matrices
     * @param {Function} callback - Called with intersection result
     * @param {Object} options - Cast options
     */
    castFromScreen(screenX, screenY, camera, callback, options = {}) {
        const { width, height } = this.engine.canvas

        // Convert screen to NDC
        const ndcX = (screenX / width) * 2 - 1
        const ndcY = 1 - (screenY / height) * 2  // Flip Y

        // Unproject to world space
        const invViewProj = mat4.create()
        mat4.multiply(invViewProj, camera.proj, camera.view)
        mat4.invert(invViewProj, invViewProj)

        // Near and far points
        const nearPoint = this._unproject([ndcX, ndcY, 0], invViewProj)
        const farPoint = this._unproject([ndcX, ndcY, 1], invViewProj)

        // Ray direction
        const direction = [
            farPoint[0] - nearPoint[0],
            farPoint[1] - nearPoint[1],
            farPoint[2] - nearPoint[2]
        ]

        const maxDistance = options.maxDistance ?? camera.far ?? 1000

        this.cast(nearPoint, direction, maxDistance, callback, options)
    }

    /**
     * Collect candidate geometries that pass bounding sphere test
     */
    _collectCandidates(ray, options) {
        const candidates = []
        const exclude = new Set(options.exclude ?? [])
        const debug = options.debug

        // Test entities - entities reference models via string ID, geometry is in asset
        const entities = options.entities ?? this._getAllEntities()
        const assetManager = this.engine.assetManager

        for (const entity of entities) {
            if (exclude.has(entity)) continue
            if (!entity.model) continue

            // Get geometry from asset manager
            const asset = assetManager?.get(entity.model)
            if (!asset?.geometry) continue

            // Entity has world-space bsphere in _bsphere
            const bsphere = this._getEntityBoundingSphere(entity)
            if (!bsphere) continue

            if (this._raySphereIntersect(ray, bsphere)) {
                const geometryData = this._extractGeometry(asset.geometry)
                if (geometryData) {
                    const matrix = entity._matrix ?? mat4.create()
                    candidates.push({
                        id: entity.id ?? entity.name ?? `entity_${candidates.length}`,
                        type: 'entity',
                        entity,
                        asset,
                        vertices: geometryData.vertices,
                        indices: geometryData.indices,
                        matrix: Array.from(matrix),
                        bsphereDistance: this._raySphereDistance(ray, bsphere)
                    })
                }
            }
        }

        // Test standalone meshes
        const meshes = options.meshes ?? this._getAllMeshes()
        let debugStats = debug ? { total: 0, noGeom: 0, noBsphere: 0, noData: 0, sphereMiss: 0, candidates: 0 } : null

        for (const [name, mesh] of Object.entries(meshes)) {
            if (exclude.has(mesh)) continue
            if (!mesh.geometry) {
                if (debug) debugStats.noGeom++
                continue
            }

            const bsphere = this._getMeshBoundingSphere(mesh)
            if (!bsphere) {
                if (debug) debugStats.noBsphere++
                continue
            }

            const geometryData = this._extractGeometry(mesh.geometry)
            if (!geometryData) {
                if (debug) debugStats.noData++
                continue
            }

            if (debug) debugStats.total++

            // For instanced meshes, test each instance
            // Static meshes keep their instanceCount; dynamic meshes may have it reset mid-frame
            let instanceCount = mesh.geometry.instanceCount ?? 0

            // For meshes with instanceCount=0, still test if they have instance data
            // (transparent/static meshes may have valid transforms even when instanceCount is 0)
            if (instanceCount === 0) {
                if (mesh.geometry.instanceData) {
                    // Use maxInstances for static meshes (instance data persists)
                    // For dynamic meshes, test at least 1 instance
                    instanceCount = mesh.static ? (mesh.geometry.maxInstances ?? 1) : 1
                } else {
                    // Non-instanced mesh - test with identity matrix
                    instanceCount = 1
                }
            }

            for (let i = 0; i < instanceCount; i++) {
                const matrix = this._getInstanceMatrix(mesh.geometry, i)
                const instanceBsphere = this._transformBoundingSphere(bsphere, matrix)

                if (this._raySphereIntersect(ray, instanceBsphere)) {
                    if (debug) debugStats.candidates++
                    candidates.push({
                        id: `${name}_${i}`,
                        type: 'mesh',
                        mesh,
                        meshName: name,
                        instanceIndex: i,
                        vertices: geometryData.vertices,
                        indices: geometryData.indices,
                        matrix: Array.from(matrix),
                        bsphereDistance: this._raySphereDistance(ray, instanceBsphere)
                    })
                } else {
                    if (debug) debugStats.sphereMiss++
                }
            }
        }

        if (debug && debugStats) {
            console.log(`Raycaster: meshes=${debugStats.total}, sphereHit=${debugStats.candidates}, sphereMiss=${debugStats.sphereMiss}`)
            // Show which candidates passed bounding sphere test
            if (candidates.length > 0 && candidates.length < 50) {
                const candInfo = candidates.map(c => {
                    const m = c.matrix
                    const pos = [m[12], m[13], m[14]]  // Translation from matrix
                    return `${c.id}@[${pos.map(v=>v.toFixed(1)).join(',')}]`
                }).join(', ')
                console.log(`Candidates: ${candInfo}`)
            }
        }

        // Sort by bounding sphere distance (closest first) for early termination
        candidates.sort((a, b) => a.bsphereDistance - b.bsphereDistance)

        return candidates
    }

    _getAllEntities() {
        // Get entities from engine's entity manager
        // engine.entities is a plain object { id: entity }
        const entities = this.engine.entities
        if (!entities) return []
        return Object.values(entities)
    }

    _getAllMeshes() {
        // Get meshes from engine
        return this.engine.meshes ?? {}
    }

    _getEntityBoundingSphere(entity) {
        // Entities have pre-calculated _bsphere in world space
        if (entity._bsphere && entity._bsphere.radius > 0) {
            return {
                center: Array.from(entity._bsphere.center),
                radius: entity._bsphere.radius
            }
        }

        // Fallback to mesh geometry bounding sphere
        const geometry = entity.mesh?.geometry
        if (!geometry) return null

        const localBsphere = geometry.getBoundingSphere?.()
        if (!localBsphere || localBsphere.radius <= 0) return null

        // Transform by entity matrix
        const matrix = entity._matrix ?? entity.matrix ?? mat4.create()
        return this._transformBoundingSphere(localBsphere, matrix)
    }

    _getMeshBoundingSphere(mesh) {
        const geometry = mesh.geometry
        if (!geometry) return null

        return geometry.getBoundingSphere?.() ?? null
    }

    _transformBoundingSphere(bsphere, matrix) {
        // Transform center
        const center = vec3.create()
        vec3.transformMat4(center, bsphere.center, matrix)

        // Scale radius by max scale factor
        const scaleX = Math.sqrt(matrix[0]*matrix[0] + matrix[1]*matrix[1] + matrix[2]*matrix[2])
        const scaleY = Math.sqrt(matrix[4]*matrix[4] + matrix[5]*matrix[5] + matrix[6]*matrix[6])
        const scaleZ = Math.sqrt(matrix[8]*matrix[8] + matrix[9]*matrix[9] + matrix[10]*matrix[10])
        const maxScale = Math.max(scaleX, scaleY, scaleZ)

        return {
            center: Array.from(center),
            radius: bsphere.radius * maxScale
        }
    }

    _getInstanceMatrix(geometry, instanceIndex) {
        // Always try to read from instanceData - transforms are stored there even for single instances
        if (!geometry.instanceData) {
            return mat4.create()
        }

        const stride = 28 // floats per instance (matrix + posRadius + uvTransform + color)
        const offset = instanceIndex * stride

        // Check if we have data at this offset
        if (offset + 16 > geometry.instanceData.length) {
            return mat4.create()
        }

        const matrix = mat4.create()

        // Copy 16 floats for matrix
        for (let i = 0; i < 16; i++) {
            matrix[i] = geometry.instanceData[offset + i]
        }

        return matrix
    }

    _extractGeometry(geometry) {
        // Get vertex positions from CPU arrays if available
        if (!geometry.vertexArray || !geometry.indexArray) {
            return null
        }

        // Extract positions (assuming stride of 20 floats: pos(3) + uv(2) + normal(3) + color(4) + weights(4) + joints(4))
        const stride = 20 // floats per vertex
        const vertexCount = geometry.vertexArray.length / stride
        const vertices = new Float32Array(vertexCount * 3)

        for (let i = 0; i < vertexCount; i++) {
            vertices[i * 3] = geometry.vertexArray[i * stride]
            vertices[i * 3 + 1] = geometry.vertexArray[i * stride + 1]
            vertices[i * 3 + 2] = geometry.vertexArray[i * stride + 2]
        }

        return {
            vertices,
            indices: geometry.indexArray
        }
    }

    /**
     * Ray-sphere intersection test
     * Returns true if ray intersects sphere within maxDistance
     * Handles case where ray origin is inside the sphere
     */
    _raySphereIntersect(ray, sphere) {
        // Vector from sphere center to ray origin
        const oc = [
            ray.origin[0] - sphere.center[0],
            ray.origin[1] - sphere.center[1],
            ray.origin[2] - sphere.center[2]
        ]

        // Check if we're inside the sphere
        const distToCenter = Math.sqrt(oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2])
        if (distToCenter < sphere.radius) {
            // Inside sphere - ray will definitely exit through it
            // Just check if exit point is within maxDistance
            // Exit distance is approximately radius - distToCenter (simplified)
            return true
        }

        const a = this._dot(ray.direction, ray.direction)
        const b = 2.0 * this._dot(oc, ray.direction)
        const c = this._dot(oc, oc) - sphere.radius * sphere.radius
        const discriminant = b * b - 4 * a * c

        if (discriminant < 0) return false

        const sqrtDisc = Math.sqrt(discriminant)
        const t1 = (-b - sqrtDisc) / (2.0 * a)
        const t2 = (-b + sqrtDisc) / (2.0 * a)

        // Check if either intersection is within valid range [0, maxDistance]
        if (t1 >= 0 && t1 <= ray.maxDistance) return true
        if (t2 >= 0 && t2 <= ray.maxDistance) return true

        return false
    }

    /**
     * Get distance to sphere along ray (for sorting)
     */
    _raySphereDistance(ray, sphere) {
        const oc = [
            ray.origin[0] - sphere.center[0],
            ray.origin[1] - sphere.center[1],
            ray.origin[2] - sphere.center[2]
        ]

        const a = this._dot(ray.direction, ray.direction)
        const b = 2.0 * this._dot(oc, ray.direction)
        const c = this._dot(oc, oc) - sphere.radius * sphere.radius
        const discriminant = b * b - 4 * a * c

        if (discriminant < 0) return Infinity

        const t = (-b - Math.sqrt(discriminant)) / (2.0 * a)
        return Math.max(0, t)
    }

    _handleWorkerMessage(event) {
        const { type, requestId, result } = event.data

        if (type === 'raycastResult') {
            const pending = this._pendingCallbacks.get(requestId)
            if (pending) {
                this._pendingCallbacks.delete(requestId)

                // Enrich result with original entity/mesh references
                if (result.hit && pending.candidates) {
                    const candidate = pending.candidates.find(c => c.id === result.candidateId)
                    if (candidate) {
                        result.entity = candidate.entity
                        result.mesh = candidate.mesh
                        result.meshName = candidate.meshName
                        result.instanceIndex = candidate.instanceIndex
                    }
                }

                pending.callback(result)
            }
        }
    }

    _normalize(v) {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if (len === 0) return [0, 0, 1]
        return [v[0]/len, v[1]/len, v[2]/len]
    }

    _dot(a, b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    }

    /**
     * Get perpendicular distance from a point to a ray
     */
    _pointToRayDistance(point, ray) {
        // Vector from ray origin to point
        const op = [
            point[0] - ray.origin[0],
            point[1] - ray.origin[1],
            point[2] - ray.origin[2]
        ]

        // Project onto ray direction
        const t = this._dot(op, ray.direction)

        // Closest point on ray
        const closest = [
            ray.origin[0] + ray.direction[0] * t,
            ray.origin[1] + ray.direction[1] * t,
            ray.origin[2] + ray.direction[2] * t
        ]

        // Distance from point to closest point on ray
        const dx = point[0] - closest[0]
        const dy = point[1] - closest[1]
        const dz = point[2] - closest[2]

        return Math.sqrt(dx*dx + dy*dy + dz*dz)
    }

    _unproject(ndc, invViewProj) {
        const x = ndc[0]
        const y = ndc[1]
        const z = ndc[2]

        // Multiply by inverse view-projection
        const w = invViewProj[3]*x + invViewProj[7]*y + invViewProj[11]*z + invViewProj[15]

        return [
            (invViewProj[0]*x + invViewProj[4]*y + invViewProj[8]*z + invViewProj[12]) / w,
            (invViewProj[1]*x + invViewProj[5]*y + invViewProj[9]*z + invViewProj[13]) / w,
            (invViewProj[2]*x + invViewProj[6]*y + invViewProj[10]*z + invViewProj[14]) / w
        ]
    }

    /**
     * Generate Web Worker code as string
     */
    _getWorkerCode() {
        return `
// Raycaster Web Worker
// Performs triangle intersection tests off the main thread

self.onmessage = function(event) {
    const { type, requestId, ray, candidates, debug } = event.data

    if (type === 'raycast') {
        const result = raycastTriangles(ray, candidates, debug)
        self.postMessage({ type: 'raycastResult', requestId, result })
    }
}

function raycastTriangles(ray, candidates, debug) {
    let closestHit = null
    let closestDistance = ray.maxDistance
    let debugInfo = debug ? { totalTris: 0, testedCandidates: 0, scales: [] } : null

    for (const candidate of candidates) {
        if (debug) debugInfo.testedCandidates++
        const result = testCandidate(ray, candidate, closestDistance, debug ? debugInfo : null)

        if (result && result.distance < closestDistance) {
            closestDistance = result.distance
            closestHit = {
                hit: true,
                distance: result.distance,
                point: result.point,
                normal: result.normal,
                triangleIndex: result.triangleIndex,
                candidateId: candidate.id,
                localT: result.localT,
                scale: result.scale
            }
        }
    }

    if (debug) {
        let msg = 'Worker: candidates=' + debugInfo.testedCandidates + ', triangles=' + debugInfo.totalTris
        if (closestHit) {
            msg += ', hit=' + closestHit.distance.toFixed(2) + ' (localT=' + closestHit.localT.toFixed(2) + ', scale=' + closestHit.scale.toFixed(2) + ')'
        } else {
            msg += ', hit=none'
        }
        console.log(msg)
    }

    return closestHit ?? { hit: false }
}

function testCandidate(ray, candidate, maxDistance, debugInfo) {
    const { vertices, indices, matrix, backfaces } = candidate

    // Compute inverse matrix for transforming ray to local space
    const invMatrix = invertMatrix4(matrix)

    // Transform ray to local space
    const localOrigin = transformPoint(ray.origin, invMatrix)
    const localDir = transformDirection(ray.direction, invMatrix)

    // Calculate the scale factor of the transformation (for correct distance)
    const dirScale = Math.sqrt(localDir[0]*localDir[0] + localDir[1]*localDir[1] + localDir[2]*localDir[2])
    const localDirNorm = [localDir[0]/dirScale, localDir[1]/dirScale, localDir[2]/dirScale]

    let closestHit = null
    let closestT = maxDistance

    // Test each triangle
    const triangleCount = indices.length / 3
    if (debugInfo) debugInfo.totalTris += triangleCount
    for (let i = 0; i < triangleCount; i++) {
        const i0 = indices[i * 3]
        const i1 = indices[i * 3 + 1]
        const i2 = indices[i * 3 + 2]

        const v0 = [vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]]
        const v1 = [vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]]
        const v2 = [vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]]

        const hit = rayTriangleIntersect(localOrigin, localDirNorm, v0, v1, v2, backfaces)

        if (hit && hit.t > 0) {
            // Transform hit point and normal back to world space
            const worldPoint = transformPoint(hit.point, matrix)

            // Calculate world-space distance (local t may be wrong due to matrix scale)
            const worldDist = Math.sqrt(
                (worldPoint[0] - ray.origin[0]) ** 2 +
                (worldPoint[1] - ray.origin[1]) ** 2 +
                (worldPoint[2] - ray.origin[2]) ** 2
            )

            if (worldDist < closestT) {
                closestT = worldDist
                const worldNormal = transformDirection(hit.normal, matrix)

                closestHit = {
                    distance: worldDist,
                    point: worldPoint,
                    normal: normalize(worldNormal),
                    triangleIndex: i,
                    localT: hit.t,
                    scale: dirScale
                }
            }
        }
    }

    return closestHit
}

// Möller–Trumbore intersection algorithm
function rayTriangleIntersect(origin, dir, v0, v1, v2, backfaces) {
    const EPSILON = 0.0000001

    const edge1 = sub(v1, v0)
    const edge2 = sub(v2, v0)
    const h = cross(dir, edge2)
    const a = dot(edge1, h)

    // Check if ray is parallel to triangle
    if (a > -EPSILON && a < EPSILON) return null

    // Check backface
    if (!backfaces && a < 0) return null

    const f = 1.0 / a
    const s = sub(origin, v0)
    const u = f * dot(s, h)

    if (u < 0.0 || u > 1.0) return null

    const q = cross(s, edge1)
    const v = f * dot(dir, q)

    if (v < 0.0 || u + v > 1.0) return null

    const t = f * dot(edge2, q)

    if (t > EPSILON) {
        const point = [
            origin[0] + dir[0] * t,
            origin[1] + dir[1] * t,
            origin[2] + dir[2] * t
        ]
        const normal = normalize(cross(edge1, edge2))
        return { t, point, normal, u, v }
    }

    return null
}

// Matrix and vector utilities
function invertMatrix4(m) {
    const inv = new Array(16)

    inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10]
    inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10]
    inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9]
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9]
    inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10]
    inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10]
    inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9]
    inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9]
    inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6]
    inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6]
    inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5]
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5]
    inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6]
    inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6]
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5]
    inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5]

    let det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12]
    if (det === 0) return m // Return original if singular

    det = 1.0 / det
    for (let i = 0; i < 16; i++) inv[i] *= det

    return inv
}

function transformPoint(p, m) {
    const w = m[3]*p[0] + m[7]*p[1] + m[11]*p[2] + m[15]
    return [
        (m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12]) / w,
        (m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13]) / w,
        (m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14]) / w
    ]
}

function transformDirection(d, m) {
    return [
        m[0]*d[0] + m[4]*d[1] + m[8]*d[2],
        m[1]*d[0] + m[5]*d[1] + m[9]*d[2],
        m[2]*d[0] + m[6]*d[1] + m[10]*d[2]
    ]
}

function normalize(v) {
    const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if (len === 0) return [0, 0, 1]
    return [v[0]/len, v[1]/len, v[2]/len]
}

function dot(a, b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

function cross(a, b) {
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]
}

function sub(a, b) {
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}
`
    }

    destroy() {
        if (this.worker) {
            this.worker.terminate()
            this.worker = null
        }
        this._pendingCallbacks.clear()
        this._initialized = false
    }
}

export { Raycaster }
