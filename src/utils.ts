import { Packr } from 'msgpackr';
import { FLOAT32_OPTIONS } from 'msgpackr';
const { ALWAYS, DECIMAL_ROUND, DECIMAL_FIT } = FLOAT32_OPTIONS;

let packr = new Packr({
  useFloat32: ALWAYS
});

export function encode(obj) {
  return packr.pack(obj)
}

export function decode(data) {
  return packr.unpack(data)
}

export function reactive(name, object, callback, path = '', excludedProperties = false) {
  if (object === null || typeof object !== 'object') {
    //console.log('--- Type not object', typeof object)
    return object
  }

  function isReactive(p) {
    let r = true
    if (p.startsWith('_')) {
      r = false
    }
    if (excludedProperties) {
      if (excludedProperties[p]) {
        r = false
      }
    }
    if (path == '/entities') {
      r = false
    }
    return r
  }

  for (const property in object) {
    if (isReactive(property)) {
      //console.log(`path '${path}', prop '${property}' is reactive`)
      object[property] = reactive(
        name,
        object[property],
        callback,
        path + '/' + property,
        excludedProperties
      )
    } else {
      //console.log(`--- path '${path}', property '${property}' is NOT reactive`)
    }
  }
  //console.log(`path '${path}' is reactive`)
  return new Proxy(object, {
    get(target, property, receiver) { // ...arguments
      return Reflect.get(target, property, receiver)
    },
    set(target, property, value) {
      let newvalue
      let pn = path + '/' + String(property)
      if (isReactive(property)) {
        newvalue = reactive(name, value, callback, pn, excludedProperties)
        callback(name, 'replace', target, pn, newvalue)
      } else {
        newvalue = value
      }
      return Reflect.set(target, property, newvalue)
    },
    deleteProperty(target, property) {
      let pn = path + '/' + String(property)
      delete target[property]
      if (isReactive(property)) {
        callback(name, 'delete', target, pn, null)
      }
      return true
    },
  })

}

export function deepGet(obj, path) {
  path = path.trimLeft('/')
  let paths = ('' + path).split('/').filter((p) => p)
  let len = paths.length
  for (let i = 0; i < len; i++) {
    if (obj[paths[i]] == undefined) {
      return undefined
    } else {
      obj = obj[paths[i]]
    }
  }
  return obj
}

export function deepSet(obj, path, value) {
  path = path.trimLeft('/')
  let paths = ('' + path).split('/').filter((p) => p)
  let len = paths.length
  let i
  for (i = 0; i < len - 1; i++) {
    obj = obj[paths[i]]
  }
  obj[paths[i]] = value
}

// recursive clone oject, without properties that starts with _ (or __)

export function clonewo_(obj, excludeStart = '_') {
  if (obj === null || typeof obj !== 'object') {
    return obj
  }

  function isExcluded(key) {
    let e = false
    if (typeof (excludeStart) == 'string' && key.startsWith(excludeStart)) {
      e = true
    } else if (typeof(excludeStart) == 'object') {
      if (excludeStart[key] || key.startsWith('_')) {
        e = true
      }
    }
    return e
  }

  if (obj instanceof Map) {
    const mapClone = new Map()
    for (let [key, value] of obj) {
      mapClone.set(clonewo_(key, excludeStart), clonewo_(value, excludeStart))
    }
    return mapClone
  }

  let clone
  if (Array.isArray(obj)) {
    clone = []
    for (let i = 0; i < obj.length; i++) {
      clone[i] = clonewo_(obj[i], excludeStart)
    }
  } else {
    clone = {}
    for (let key in obj) {
      if (obj.hasOwnProperty(key) && !isExcluded(key)) {
        if (typeof obj[key] === 'object') {
          clone[key] = clonewo_(obj[key], excludeStart)
        } else {
          clone[key] = obj[key]
        }
      }
    }
  }

  return clone
}

export function limitPrecision(obj) {
  if (Array.isArray(obj)) {
    return obj.map(limitPrecision)
  } else if (obj !== null && typeof obj === 'object') {
    const result = {}
    for (const key in obj) {
      result[key] = limitPrecision(obj[key])
    }
    return result
  } else if (typeof obj === 'number') {
    if (Number.isInteger(obj)) {
      return obj
    } else {
      // Limit to max 3 decimal digits, not fixed
      return parseFloat(obj.toFixed(3))
    }
  } else {
    return obj
  }
}

export function msgop(op:any) {
  let nop:any = {}
  if (!op.o) {
    nop.op = 'replace'
  } else {
    nop.op = {
      a: 'add',
      r: 'remove',
      d: 'delete',
      t: 'test',
    }[op.o]
  }
  nop.path = op.p
  nop.value = op.v
  return nop
}

export function opmsg(op, target, path, value) {
  let c:any = { p: path, v: value }
  if (op != 'replace') {
    c.o = {
      add: 'a',
      remove: 'r',
      delete: 'd',
      test: 't',
    }[op]
  }
  return c
}

// a function that converts an int to a hexa string
// (8 characters long)
export function int2hex(int) {
  return int.toString(16)
}

// a function that converts a hexa string to an int
export function hex2int(str) {
  if (str.length % 2) {
    str = '0' + str
  }
  return BigInt('0x' + str)
}

// - Fixed point encoding/decoding functions

// 32-bit unsigned integer encoding
export function encode_uint32(uint, byteArray, offset = 0) {
  if (!byteArray) {
    byteArray = new Uint8Array(4)
  }
  let p = offset + 3
  byteArray[p--] = uint & 0xff
  uint >>= 8
  byteArray[p--] = uint & 0xff
  uint >>= 8
  byteArray[p--] = uint & 0xff
  uint >>= 8
  byteArray[p] = uint
  return byteArray
}

// 32-bit unsigned integer decoding
export function decode_uint32(byteArray, offset = 0) {
  let p = offset
  return (
    ((byteArray[p++] & 0x7f) << 24) |
    (byteArray[p++] << 16) |
    (byteArray[p++] << 8) |
    byteArray[p]
  )
}

// 24-bit unsigned integer encoding
export function encode_uint24(uint, byteArray, offset = 0) {
  if (!byteArray) {
    byteArray = new Uint8Array(3)
  }
  let p = offset + 2
  byteArray[p--] = uint & 0xff
  uint >>= 8
  byteArray[p--] = uint & 0xff
  uint >>= 8
  byteArray[p] = uint
  return byteArray
}

// 24-bit unsigned integer decoding
export function decode_uint24(byteArray, offset = 0) {
  let p = offset
  return (
    (byteArray[p++] << 16) |
    (byteArray[p++] << 8) |
    byteArray[p]
  )
}

// 16-bit unsigned integer encoding
export function encode_uint16(uint, byteArray, offset = 0) {
  if (!byteArray) {
    byteArray = new Uint8Array(2)
  }
  let p = offset + 1
  byteArray[p--] = uint & 0xff
  uint >>= 8
  byteArray[p] = uint
  return byteArray
}

// 16-bit unsigned integer decoding
export function decode_uint16(byteArray, offset = 0) {
  let p = offset
  return (byteArray[p++] << 8) | byteArray[p]
}

// 24.8 bit ====================================================================

// 24.8-bit fixed point encoding
export function encode_fp248(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 256)
  encode_uint32(fp, byteArray, offset)
  if (float < 0) {
    byteArray[offset] |= 0x80
  }
}

// 24.8-bit fixed point decoding
export function decode_fp248(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -256 : 256
  byteArray[offset] &= 0x7f
  const fp = decode_uint32(byteArray, offset)
  return fp / divider
}

// 16.8 bit ====================================================================

// 16.8-bit fixed point encoding (3 bytes)
export function encode_fp168(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 256)
  encode_uint24(fp, byteArray, offset)
  if (float < 0) {
    byteArray[offset] |= 0x80
  }
}

// 16.8-bit fixed point decoding (3 bytes)
export function decode_fp168(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -256 : 256
  byteArray[offset] &= 0x7f
  const fp = decode_uint24(byteArray, offset)
  return fp / divider
}

// 16.16 bit ===================================================================

// 16.16-bit fixed point encoding
export function encode_fp1616(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 65536)
  encode_uint32(fp, byteArray, offset)
  if (float < 0) {
    byteArray[offset] |= 0x80
  }
}

// 16.16-bit fixed point decoding
export function decode_fp1616(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -65536 : 65536
  byteArray[offset] &= 0x7f
  const fp = decode_uint32(byteArray, offset)
  return fp / divider
}

// 8.8 bit =====================================================================

// 8.8-bit fixed point encoding
export function encode_fp88(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 256)
  encode_uint16(fp, byteArray, offset)
  if (float < 0) {
    byteArray[offset] |= 0x80
  }
}

// 8.8-bit fixed point decoding
export function decode_fp88(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -256 : 256
  byteArray[offset] &= 0x7f
  const fp = decode_uint16(byteArray, offset)
  return fp / divider
}

// 4.12 bit ====================================================================

// 4.12-bit fixed point encoding
export function encode_fp412(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 4096)
  encode_uint16(fp, byteArray, offset)
  if (float < 0) {
    byteArray[offset] |= 0x80
  }
}

// 4.12-bit fixed point decoding
export function decode_fp412(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -4096 : 4096
  byteArray[offset] &= 0x7f
  const fp = decode_uint16(byteArray, offset)
  return fp / divider
}

// 1.7 bit =====================================================================

// 1.7-bit fixed point encoding
export function encode_fp17(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 128)
  byteArray[offset] = fp
  if (float < 0) {
    byteArray[offset] |= 0x80
  }
}

// 1.7-bit fixed point decoding
export function decode_fp17(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -128.0 : 128.0
  byteArray[offset] &= 0x7f
  return byteArray[offset] / divider
}
