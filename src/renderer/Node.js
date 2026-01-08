import { fromEuler, toEuler, UP, RIGHT, FORWARD, V_T, V_T2 } from "./math.js"

class Node {

    constructor() {
        this.name = null
        this.position = vec3.create()
        this.rotation = quat.create()
        this.scale = vec3.fromValues(1, 1, 1)
        this.children = []

        // calculated matrices
        this.matrix = mat4.create()
        this.world = mat4.create()
        this.inv = mat4.create()
        this.normal = mat4.create()
    }

    fromEuler(euler) {
        //console.log('before fromEuler', this.rotation)
        fromEuler(euler, this.rotation)
        //console.log('after fromEuler', this.rotation)
    }

    get yaw() {
        toEuler(this.rotation, V_T)
        //console.log('get yaw', V_T[0])
        return V_T[1]
    }

    set yaw(yaw) {
        //console.log('--- set yaw', yaw)
        //console.log('--- set before rotation', this.rotation)
        toEuler(this.rotation, V_T2)
        //console.log('toEuler', V_T2)
        V_T2[1] = yaw
        this.fromEuler(V_T2)
    }

    rotateYaw(yaw) {
        this.yaw += yaw
    }

    get pitch() {
        toEuler(this.rotation, V_T)
        return V_T[0]
    }

    set pitch(pitch) {
        toEuler(this.rotation, V_T2)
        V_T2[0] = pitch
        this.fromEuler(V_T2)
    }

    rotatePitch(pitch) {
        this.pitch += pitch
    }

    limitPitch() {
        // Convert current pitch to degrees
        let pitchDegrees = this.pitch * (180 / Math.PI);
        
        // Clamp pitch between -89.9 and 89.9 degrees
        pitchDegrees = Math.max(-89, Math.min(89, pitchDegrees));
        
        // Convert back to radians and set the pitch
        this.pitch = pitchDegrees * (Math.PI / 180);
    }

    get roll() {
        toEuler(this.rotation, V_T)
        return V_T[2]
    }

    set roll(roll) {
        toEuler(this.rotation, V_T2)
        V_T2[2] = roll
        this.fromEuler(V_T2)
    }

    rotateRoll(roll) {
        this.roll += roll
    }

    updateMatrix(parentMatrix) {
        mat4.fromRotationTranslationScale(this.matrix, this.rotation, this.position, this.scale)
        mat4.identity(this.world)
        mat4.multiply(this.world, this.world, this.matrix)
        if (parentMatrix) {
            mat4.multiply(this.world, this.world, parentMatrix)
        }
        mat4.invert(this.inv, this.world)
        mat4.transpose(this.normal, this.inv)
        if (this.direction) {
            vec3.set(this.direction, 0, 0, -1)
            vec3.transformQuat(this.direction, this.direction, this.rotation)
        }
        if (this.right) {
            vec3.set(this.right, 1, 0, 0)
            vec3.transformQuat(this.right, this.right, this.rotation)
        }
        for (let child of this.children) {
            child.updateMatrix(this.world)
        }
    }

    addChild(child) {
        this.children.push(child)
    }
}

export { Node, toEuler, fromEuler }
