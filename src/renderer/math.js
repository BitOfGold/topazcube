import * as glMatrix from "gl-matrix"

// Extract gl-matrix modules for both window globals and ES module exports
const { mat4, mat3, mat2, mat2d, vec2, vec3, vec4, quat, quat2 } = glMatrix

window.mat4 = mat4
window.mat2 = mat2
window.mat2d = mat2d
window.mat3 = mat3
window.quat = quat
window.quat2 = quat2
window.vec2 = vec2
window.vec3 = vec3
window.vec4 = vec4

// Make all Math functions globally accessible
for (let name of Object.getOwnPropertyNames(Math)) {
  window[name] = Math[name]
}
// Add PI2 constant
window.PI2 = Math.PI * 2

const UP = vec3.fromValues(0, 1, 0)
const RIGHT = vec3.fromValues(1, 0, 0)
const FORWARD = vec3.fromValues(0, 0, -1)

let V_T = vec3.create()
let V_T2 = vec3.create()

// YXZ
function toEuler(q, dst) {
    var qx = q[0]
    var qy = q[1]
    var qz = q[2]
    var qw = q[3]

    var sqw = qw * qw
    var sqz = qz * qz
    var sqx = qx * qx
    var sqy = qy * qy

    var zAxisY = qy * qz - qx * qw
    var limit = 0.4999999

    if (zAxisY < -limit) {
      dst[1] = 2 * atan2(qy, qw)
      dst[0] = PI / 2
      dst[2] = 0
    } else if (zAxisY > limit) {
      dst[1] = 2 * atan2(qy, qw)
      dst[0] = -PI / 2
      dst[2] = 0
    } else {
      dst[2] = atan2(2.0 * (qx * qy + qz * qw), -sqz - sqx + sqy + sqw)
      dst[0] = asin(-2.0 * (qz * qy - qx * qw))
      dst[1] = atan2(2.0 * (qz * qx + qy * qw), sqz - sqx - sqy + sqw)
    }
    return dst
}

function fromEuler(v, dst) {
    let hx = v[0] * 0.5
    let hy = v[1] * 0.5
    let hz = v[2] * 0.5
    let sx = sin(hx)
    let cx = cos(hx)
    let sy = sin(hy)
    let cy = cos(hy)
    let sz = sin(hz)
    let cz = cos(hz)
    
    // YXZ
    dst[0] = sx * cy * cz + cx * sy * sz
    dst[1] = cx * sy * cz - sx * cy * sz
    dst[2] = cx * cy * sz - sx * sy * cz
    dst[3] = cx * cy * cz + sx * sy * sz
    return dst
}

/*
Octahedron-normal vectors

// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/

float2 OctWrap(float2 v)
{
    return (1.0 - abs(v.yx)) * (v.xy >= 0.0 ? 1.0 : -1.0);
}
 
float2 Encode(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = n.z >= 0.0 ? n.xy : OctWrap(n.xy);
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}
 
float3 Decode(float2 f)
{
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += n.xy >= 0.0 ? -t : t;
    return normalize(n);
}
*/

function octWrap(out, v) {
    out[0] = (1.0 - Math.abs(v[1])) * (v[0] >= 0.0 ? 1.0 : -1.0)
    out[1] = (1.0 - Math.abs(v[0])) * (v[1] >= 0.0 ? 1.0 : -1.0)
    return out
}

function encodeNormal(out, n) {
    let sum = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2])
    let nx = n[0] / sum
    let ny = n[1] / sum
    
    if (n[2] >= 0.0) {
        out[0] = nx
        out[1] = ny
    } else {
        let temp = vec2.create()
        temp[0] = nx
        temp[1] = ny
        octWrap(out, temp)
    }
    
    out[0] = out[0] * 0.5 + 0.5
    out[1] = out[1] * 0.5 + 0.5
    return out
}

function decodeNormal(out, f) {
    let fx = f[0] * 2.0 - 1.0
    let fy = f[1] * 2.0 - 1.0

    let fz = 1.0 - Math.abs(fx) - Math.abs(fy)
    
    let t = Math.max(0.0, Math.min(1.0, -fz))
    
    out[0] = fx + (fx >= 0.0 ? -t : t)
    out[1] = fy + (fy >= 0.0 ? -t : t) 
    out[2] = fz
    
    vec3.normalize(out, out)
    return out
}

export {
    toEuler,
    fromEuler,
    UP, RIGHT, FORWARD,
    V_T, V_T2,
    // gl-matrix exports
    mat4, mat3, mat2,
    vec2, vec3, vec4,
    quat, quat2
}
