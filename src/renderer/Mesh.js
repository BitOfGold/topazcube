import { mat4, vec3 } from "./math.js"

var _UID = 20001

class Mesh {
    constructor(geometry, material, name = null) {
        this.engine = geometry.engine || material.engine
        this.uid = _UID++
        this.name = name || `Mesh_${this.uid}`
        this.geometry = geometry
        this.material = material

        // Cache for rotation to avoid recalculating from matrix
        this._rotation = { yaw: 0, pitch: 0, roll: 0 }
        this._scale = [1, 1, 1]
    }

    /**
     * Add an instance with position and optional bounding sphere radius
     * @param {Array} position - Position [x, y, z] or bounding sphere center
     * @param {number} radius - Bounding sphere radius (default 1)
     * @param {Array} uvTransform - UV transform [offsetX, offsetY, scaleX, scaleY] (default [0,0,1,1])
     * @param {Array} color - RGBA color tint (default [1,1,1,1])
     */
    addInstance(position, radius = 1, uvTransform = [0, 0, 1, 1], color = [1, 1, 1, 1]) {
        this.geometry.addInstance(position, radius, uvTransform, color)
    }

    updateInstance(index, matrix) {
        this.geometry.updateInstance(index, matrix)
    }

    /**
     * Get position of the first instance (legacy mesh)
     * @returns {[number, number, number]} Position [x, y, z]
     */
    get position() {
        const data = this.geometry.instanceData
        if (!data) return [0, 0, 0]
        return [data[12], data[13], data[14]]
    }

    /**
     * Set position of the first instance (legacy mesh)
     * @param {[number, number, number]} pos - Position [x, y, z]
     */
    set position(pos) {
        const data = this.geometry.instanceData
        if (!data) return
        data[12] = pos[0]
        data[13] = pos[1]
        data[14] = pos[2]
        this.geometry._instanceDataDirty = true
    }

    /**
     * Get rotation of the first instance as euler angles
     * @returns {{yaw: number, pitch: number, roll: number}} Rotation in radians
     */
    get rotation() {
        const data = this.geometry.instanceData
        if (!data) return { yaw: 0, pitch: 0, roll: 0 }

        // Extract rotation from matrix (assuming no shear)
        // Matrix layout: column-major [m0,m1,m2,m3, m4,m5,m6,m7, m8,m9,m10,m11, m12,m13,m14,m15]
        // Column 0: [m0,m1,m2] = right * scaleX
        // Column 1: [m4,m5,m6] = up * scaleY
        // Column 2: [m8,m9,m10] = forward * scaleZ

        // Get scale to normalize rotation matrix
        const scaleX = Math.sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2])
        const scaleY = Math.sqrt(data[4]*data[4] + data[5]*data[5] + data[6]*data[6])
        const scaleZ = Math.sqrt(data[8]*data[8] + data[9]*data[9] + data[10]*data[10])

        // Normalized rotation matrix elements
        const m00 = data[0] / scaleX, m01 = data[4] / scaleY, m02 = data[8] / scaleZ
        const m10 = data[1] / scaleX, m11 = data[5] / scaleY, m12 = data[9] / scaleZ
        const m20 = data[2] / scaleX, m21 = data[6] / scaleY, m22 = data[10] / scaleZ

        // Extract euler angles (YXZ order: yaw, pitch, roll)
        let pitch, yaw, roll

        if (Math.abs(m12) < 0.99999) {
            pitch = Math.asin(-m12)
            yaw = Math.atan2(m02, m22)
            roll = Math.atan2(m10, m11)
        } else {
            // Gimbal lock
            pitch = m12 < 0 ? Math.PI / 2 : -Math.PI / 2
            yaw = Math.atan2(-m20, m00)
            roll = 0
        }

        return { yaw, pitch, roll }
    }

    /**
     * Set rotation of the first instance using euler angles
     * Rebuilds the transform matrix preserving position and scale
     * @param {{yaw?: number, pitch?: number, roll?: number}} rot - Rotation in radians
     */
    set rotation(rot) {
        const data = this.geometry.instanceData
        if (!data) return

        // Get current position
        const pos = [data[12], data[13], data[14]]

        // Get current scale
        const scaleX = Math.sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2])
        const scaleY = Math.sqrt(data[4]*data[4] + data[5]*data[5] + data[6]*data[6])
        const scaleZ = Math.sqrt(data[8]*data[8] + data[9]*data[9] + data[10]*data[10])

        // Update cached rotation
        const yaw = rot.yaw ?? this._rotation.yaw
        const pitch = rot.pitch ?? this._rotation.pitch
        const roll = rot.roll ?? this._rotation.roll
        this._rotation = { yaw, pitch, roll }

        // Build rotation matrix (YXZ order)
        const cy = Math.cos(yaw), sy = Math.sin(yaw)
        const cp = Math.cos(pitch), sp = Math.sin(pitch)
        const cr = Math.cos(roll), sr = Math.sin(roll)

        // Combined rotation matrix R = Ry * Rx * Rz
        const m00 = cy * cr + sy * sp * sr
        const m01 = -cy * sr + sy * sp * cr
        const m02 = sy * cp
        const m10 = cp * sr
        const m11 = cp * cr
        const m12 = -sp
        const m20 = -sy * cr + cy * sp * sr
        const m21 = sy * sr + cy * sp * cr
        const m22 = cy * cp

        // Apply scale and write to instance data
        data[0] = m00 * scaleX; data[1] = m10 * scaleX; data[2] = m20 * scaleX; data[3] = 0
        data[4] = m01 * scaleY; data[5] = m11 * scaleY; data[6] = m21 * scaleY; data[7] = 0
        data[8] = m02 * scaleZ; data[9] = m12 * scaleZ; data[10] = m22 * scaleZ; data[11] = 0
        data[12] = pos[0]; data[13] = pos[1]; data[14] = pos[2]; data[15] = 1

        this.geometry._instanceDataDirty = true
    }

    /**
     * Get scale of the first instance
     * @returns {[number, number, number]} Scale [x, y, z]
     */
    get scale() {
        const data = this.geometry.instanceData
        if (!data) return [1, 1, 1]

        const scaleX = Math.sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2])
        const scaleY = Math.sqrt(data[4]*data[4] + data[5]*data[5] + data[6]*data[6])
        const scaleZ = Math.sqrt(data[8]*data[8] + data[9]*data[9] + data[10]*data[10])

        return [scaleX, scaleY, scaleZ]
    }

    /**
     * Set scale of the first instance
     * Rebuilds the transform matrix preserving position and rotation
     * @param {[number, number, number]} s - Scale [x, y, z]
     */
    set scale(s) {
        const data = this.geometry.instanceData
        if (!data) return

        // Get current scale to compute ratio
        const oldScaleX = Math.sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2])
        const oldScaleY = Math.sqrt(data[4]*data[4] + data[5]*data[5] + data[6]*data[6])
        const oldScaleZ = Math.sqrt(data[8]*data[8] + data[9]*data[9] + data[10]*data[10])

        // Scale ratio
        const rx = oldScaleX > 0 ? s[0] / oldScaleX : s[0]
        const ry = oldScaleY > 0 ? s[1] / oldScaleY : s[1]
        const rz = oldScaleZ > 0 ? s[2] / oldScaleZ : s[2]

        // Apply new scale to rotation columns
        data[0] *= rx; data[1] *= rx; data[2] *= rx
        data[4] *= ry; data[5] *= ry; data[6] *= ry
        data[8] *= rz; data[9] *= rz; data[10] *= rz

        this.geometry._instanceDataDirty = true
    }

    /**
     * Get the full transform matrix of the first instance
     * @returns {Float32Array} 4x4 transform matrix (16 floats)
     */
    get matrix() {
        const data = this.geometry.instanceData
        if (!data) return mat4.create()
        return new Float32Array(data.buffer, data.byteOffset, 16)
    }

    /**
     * Set the full transform matrix of the first instance
     * @param {Float32Array|Array} m - 4x4 transform matrix
     */
    set matrix(m) {
        const data = this.geometry.instanceData
        if (!data) return
        for (let i = 0; i < 16; i++) {
            data[i] = m[i]
        }
        this.geometry._instanceDataDirty = true
    }
}

export { Mesh }
