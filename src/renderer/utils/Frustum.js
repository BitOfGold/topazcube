import { vec3, vec4, mat4 } from "../math.js"

/**
 * Frustum - Camera frustum for culling
 *
 * Implements cone-based frustum culling which is faster than plane-based
 * for rectangular viewports while maintaining accuracy.
 */
class Frustum {
    constructor() {
        // Standard frustum planes (for fallback)
        // Order: left, right, bottom, top, near, far
        this.planes = new Float32Array(24) // 6 planes * 4 components

        // Cone-based frustum (more efficient for spheres)
        this.coneAxis = vec3.create()      // Camera forward direction
        this.coneOrigin = vec3.create()    // Camera position
        this.coneAngle = 0                 // Half-angle of the cone (radians)
        this.coneCos = 0                   // cos(coneAngle) for fast tests
        this.coneSin = 0                   // sin(coneAngle) for fast tests
        this.nearPlane = 0.1
        this.farPlane = 1000

        // View-projection matrix for plane extraction
        this._viewProj = mat4.create()

        // Screen dimensions for pixel size culling
        this.screenWidth = 1920
        this.screenHeight = 1080
        this.projectionScale = 1  // screenHeight / (2 * tan(fov/2))
    }

    /**
     * Update frustum from camera matrices
     * @param {mat4} viewMatrix - View matrix
     * @param {mat4} projectionMatrix - Projection matrix
     * @param {vec3} cameraPosition - Camera world position
     * @param {vec3} cameraForward - Camera forward direction (normalized)
     * @param {number} fov - Field of view in radians (vertical)
     * @param {number} aspect - Aspect ratio (width/height)
     * @param {number} near - Near plane distance
     * @param {number} far - Far plane distance
     * @param {number} screenWidth - Screen width in pixels (optional)
     * @param {number} screenHeight - Screen height in pixels (optional)
     */
    update(viewMatrix, projectionMatrix, cameraPosition, cameraForward, fov, aspect, near, far, screenWidth, screenHeight) {
        // Store near/far for distance culling
        this.nearPlane = near
        this.farPlane = far

        // Store screen dimensions for pixel size culling
        if (screenWidth) this.screenWidth = screenWidth
        if (screenHeight) this.screenHeight = screenHeight

        // Compute projection scale: converts world-space radius to screen pixels
        // projectedPixels = radius * projectionScale / distance
        const halfFov = fov * 0.5
        this.projectionScale = this.screenHeight / (2 * Math.tan(halfFov))

        // Compute view-projection matrix
        mat4.multiply(this._viewProj, projectionMatrix, viewMatrix)

        // Extract standard frustum planes from view-projection matrix
        this._extractPlanes(this._viewProj)

        // Setup cone-based frustum
        vec3.copy(this.coneOrigin, cameraPosition)
        vec3.copy(this.coneAxis, cameraForward)

        // Calculate cone angle that encompasses the frustum
        // For a rectangular viewport, use the diagonal angle
        const halfFovH = Math.atan(Math.tan(halfFov) * aspect)
        // Diagonal half-angle (slightly larger to ensure we don't cull visible objects)
        this.coneAngle = Math.sqrt(halfFov * halfFov + halfFovH * halfFovH) * 1.1

        this.coneCos = Math.cos(this.coneAngle)
        this.coneSin = Math.sin(this.coneAngle)
    }

    /**
     * Extract frustum planes from view-projection matrix
     * Planes are in the form: ax + by + cz + d = 0
     * @private
     */
    _extractPlanes(vp) {
        // Left plane
        this.planes[0] = vp[3] + vp[0]
        this.planes[1] = vp[7] + vp[4]
        this.planes[2] = vp[11] + vp[8]
        this.planes[3] = vp[15] + vp[12]
        this._normalizePlane(0)

        // Right plane
        this.planes[4] = vp[3] - vp[0]
        this.planes[5] = vp[7] - vp[4]
        this.planes[6] = vp[11] - vp[8]
        this.planes[7] = vp[15] - vp[12]
        this._normalizePlane(1)

        // Bottom plane
        this.planes[8] = vp[3] + vp[1]
        this.planes[9] = vp[7] + vp[5]
        this.planes[10] = vp[11] + vp[9]
        this.planes[11] = vp[15] + vp[13]
        this._normalizePlane(2)

        // Top plane
        this.planes[12] = vp[3] - vp[1]
        this.planes[13] = vp[7] - vp[5]
        this.planes[14] = vp[11] - vp[9]
        this.planes[15] = vp[15] - vp[13]
        this._normalizePlane(3)

        // Near plane (WebGPU: z >= 0, so just row 2)
        this.planes[16] = vp[2]
        this.planes[17] = vp[6]
        this.planes[18] = vp[10]
        this.planes[19] = vp[14]
        this._normalizePlane(4)

        // Far plane (z <= w)
        this.planes[20] = vp[3] - vp[2]
        this.planes[21] = vp[7] - vp[6]
        this.planes[22] = vp[11] - vp[10]
        this.planes[23] = vp[15] - vp[14]
        this._normalizePlane(5)
    }

    /**
     * Normalize a frustum plane
     * @private
     */
    _normalizePlane(index) {
        const offset = index * 4
        const length = Math.sqrt(
            this.planes[offset] * this.planes[offset] +
            this.planes[offset + 1] * this.planes[offset + 1] +
            this.planes[offset + 2] * this.planes[offset + 2]
        )
        if (length > 0) {
            this.planes[offset] /= length
            this.planes[offset + 1] /= length
            this.planes[offset + 2] /= length
            this.planes[offset + 3] /= length
        }
    }

    /**
     * Test if a sphere is visible using cone-based culling
     * This is faster than plane testing for large numbers of objects
     *
     * @param {Object} bsphere - Bounding sphere { center: [x,y,z], radius: r }
     * @returns {boolean} True if potentially visible
     */
    testSphere(bsphere) {
        // Vector from cone origin to sphere center
        const dx = bsphere.center[0] - this.coneOrigin[0]
        const dy = bsphere.center[1] - this.coneOrigin[1]
        const dz = bsphere.center[2] - this.coneOrigin[2]

        // Distance along cone axis
        const distAlongAxis = dx * this.coneAxis[0] + dy * this.coneAxis[1] + dz * this.coneAxis[2]

        // Quick near/far plane test
        if (distAlongAxis + bsphere.radius < this.nearPlane || distAlongAxis - bsphere.radius > this.farPlane) {
            return false
        }

        // For objects behind the camera
        if (distAlongAxis < -bsphere.radius) {
            return false
        }

        // Distance from cone axis
        const dist2 = dx * dx + dy * dy + dz * dz
        const distFromAxis2 = dist2 - distAlongAxis * distAlongAxis
        const distFromAxis = Math.sqrt(Math.max(0, distFromAxis2))

        // Angle from cone axis to sphere edge
        // sphere is visible if its closest edge is within the cone angle
        const sphereAngle = Math.atan2(distFromAxis - bsphere.radius, Math.max(0.001, distAlongAxis))

        return sphereAngle < this.coneAngle
    }

    /**
     * Test if a sphere is visible using standard plane-based culling
     * More accurate but slower than cone test
     *
     * @param {Object} bsphere - Bounding sphere { center: [x,y,z], radius: r }
     * @returns {boolean} True if potentially visible
     */
    testSpherePlanes(bsphere) {
        for (let i = 0; i < 6; i++) {
            const offset = i * 4
            const dist =
                this.planes[offset] * bsphere.center[0] +
                this.planes[offset + 1] * bsphere.center[1] +
                this.planes[offset + 2] * bsphere.center[2] +
                this.planes[offset + 3]

            if (dist < -bsphere.radius) {
                return false
            }
        }
        return true
    }

    /**
     * Test if a sphere is within a maximum distance from the camera
     *
     * @param {Object} bsphere - Bounding sphere
     * @param {number} maxDistance - Maximum distance
     * @returns {boolean} True if within distance
     */
    testSphereDistance(bsphere, maxDistance) {
        const dx = bsphere.center[0] - this.coneOrigin[0]
        const dy = bsphere.center[1] - this.coneOrigin[1]
        const dz = bsphere.center[2] - this.coneOrigin[2]
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) - bsphere.radius
        return dist <= maxDistance
    }

    /**
     * Get the distance from camera to sphere center
     *
     * @param {Object} bsphere - Bounding sphere
     * @returns {number} Distance
     */
    getDistance(bsphere) {
        const dx = bsphere.center[0] - this.coneOrigin[0]
        const dy = bsphere.center[1] - this.coneOrigin[1]
        const dz = bsphere.center[2] - this.coneOrigin[2]
        return Math.sqrt(dx * dx + dy * dy + dz * dz)
    }

    /**
     * Get the projected size of a bounding sphere in pixels
     *
     * @param {Object} bsphere - Bounding sphere
     * @param {number} distance - Pre-computed distance (optional)
     * @returns {number} Projected diameter in pixels
     */
    getProjectedSize(bsphere, distance) {
        if (distance === undefined) {
            distance = this.getDistance(bsphere)
        }
        // Avoid division by zero for very close objects
        if (distance < this.nearPlane) {
            return this.screenHeight // Fill screen
        }
        // Projected diameter = 2 * radius * projectionScale / distance
        return 2 * bsphere.radius * this.projectionScale / distance
    }

    /**
     * Test both frustum visibility and distance
     *
     * @param {Object} bsphere - Bounding sphere
     * @param {number} maxDistance - Maximum distance (optional)
     * @returns {boolean} True if visible and within distance
     */
    test(bsphere, maxDistance = Infinity) {
        if (!this.testSphere(bsphere)) {
            return false
        }
        if (maxDistance < Infinity) {
            return this.testSphereDistance(bsphere, maxDistance)
        }
        return true
    }

    /**
     * Get frustum planes as Float32Array for GPU upload
     */
    getPlanesBuffer() {
        return this.planes
    }
}

export { Frustum }
