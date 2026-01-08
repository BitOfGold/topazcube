import { Node } from './Node.js'

const UP = vec3.fromValues(0, 1, 0)

// Perspective projection for WebGPU's [0, 1] depth range
function perspectiveZO(out, fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy / 2)
    const nf = 1 / (near - far)

    out[0] = f / aspect
    out[1] = 0
    out[2] = 0
    out[3] = 0
    out[4] = 0
    out[5] = f
    out[6] = 0
    out[7] = 0
    out[8] = 0
    out[9] = 0
    out[10] = far * nf
    out[11] = -1
    out[12] = 0
    out[13] = 0
    out[14] = near * far * nf
    out[15] = 0

    return out
}

class Camera extends Node {

    constructor(engine = null) {
        super()
        this.engine = engine

        // Initialize from engine settings or use defaults
        const cameraSettings = engine?.settings?.camera || { fov: 70, near: 0.05, far: 5000 }

        this.fov = cameraSettings.fov
        this.direction = vec3.fromValues(0, 0, -1)
        this.right = vec3.fromValues(1, 0, 0)
        this.target = vec3.create()
        this.aspect = 16 / 9
        this.near = cameraSettings.near
        this.far = cameraSettings.far
        this.view = mat4.create()
        this.proj = mat4.create()
        this.viewProj = mat4.create()
        this.iViewProj = mat4.create()
        this.iProj = mat4.create()
        this.iView = mat4.create()
        this.planes = new Array(6).fill(null).map(() => ({
            normal: vec3.create(),
            distance: 0
        }));

        // TAA jitter - sub-pixel offset for temporal anti-aliasing
        this.jitterEnabled = false
        this.jitterOffset = [0, 0]  // Current frame's jitter in pixels
        this.jitterAngle = 0  // Current jitter angle in radians
        this.screenSize = [1920, 1080]  // Updated by RenderGraph
    }

    updateView() {
        // this.limitPitch(), this.updateMatrix() before updating view
        vec3.add(this.target, this.position, this.direction)
        mat4.lookAt(this.view, this.position, this.target, UP)
        // Use perspectiveZO for WebGPU's [0, 1] depth range (not OpenGL's [-1, 1])
        perspectiveZO(this.proj, this.fov * (Math.PI / 180), this.aspect, this.near, this.far)

        // Update TAA jitter offset (applied in vertex shader, not projection matrix)
        if (this.jitterEnabled) {
            // Rotate jitter vector by golden angle (137.5°) for optimal coverage
            // Golden angle ensures successive samples are maximally spread apart
            this.jitterAngle += 137.5 * Math.PI / 180
            const amount = this.engine?.settings?.rendering?.jitterAmount ?? 0.39
            this.jitterOffset[0] = Math.cos(this.jitterAngle) * amount
            this.jitterOffset[1] = Math.sin(this.jitterAngle) * amount
        } else {
            this.jitterOffset[0] = 0
            this.jitterOffset[1] = 0
        }

        mat4.multiply(this.viewProj, this.proj, this.view)
        mat4.invert(this.iViewProj, this.viewProj)
        mat4.invert(this.iProj, this.proj)
        mat4.invert(this.iView, this.view)
        this._updatePlanes()
    }

    isBoxVisible(min, max) {
        // For each plane
        for (let i = 0; i < 6; i++) {
            const plane = this.planes[i];

            // Calculate the positive vertex (p-vertex)
            const px = plane.normal[0] > 0 ? max[0] : min[0];
            const py = plane.normal[1] > 0 ? max[1] : min[1];
            const pz = plane.normal[2] > 0 ? max[2] : min[2];

            // If the positive vertex is outside, the whole box is outside
            if (px * plane.normal[0] +
                py * plane.normal[1] +
                pz * plane.normal[2] +
                plane.distance < 0) {
                return false;
            }
        }

        return true;
    }

    // Check if a sphere is visible
    isSphereVisible(center, radius) {
        // For each plane
        for (let i = 0; i < 6; i++) {
            const plane = this.planes[i];

            // Calculate signed distance from sphere center to plane
            const distance = vec3.dot(plane.normal, center) + plane.distance;

            // If the distance is less than -radius, the sphere is completely outside
            if (distance < -radius) {
                return false;
            }
        }

        return true;
    }

    // Extract frustum planes from view-projection matrix (column-major format)
    // For column-major: row i elements are at indices [i, i+4, i+8, i+12]
    _updatePlanes() {
        const m = this.viewProj;

        // Left plane: row3 + row0
        this._setPlane(0,
            m[3] + m[0],
            m[7] + m[4],
            m[11] + m[8],
            m[15] + m[12]);

        // Right plane: row3 - row0
        this._setPlane(1,
            m[3] - m[0],
            m[7] - m[4],
            m[11] - m[8],
            m[15] - m[12]);

        // Bottom plane: row3 + row1
        this._setPlane(2,
            m[3] + m[1],
            m[7] + m[5],
            m[11] + m[9],
            m[15] + m[13]);

        // Top plane: row3 - row1
        this._setPlane(3,
            m[3] - m[1],
            m[7] - m[5],
            m[11] - m[9],
            m[15] - m[13]);

        // Near plane (WebGPU: z >= 0, so just row2)
        this._setPlane(4,
            m[2],
            m[6],
            m[10],
            m[14]);

        // Far plane: row3 - row2
        this._setPlane(5,
            m[3] - m[2],
            m[7] - m[6],
            m[11] - m[10],
            m[15] - m[14]);
    }

    _setPlane(index, x, y, z, w) {
        const length = sqrt(x * x + y * y + z * z);
        const plane = this.planes[index];

        plane.normal[0] = x / length;
        plane.normal[1] = y / length;
        plane.normal[2] = z / length;
        plane.distance = w / length;
    }

}

export { Camera }
