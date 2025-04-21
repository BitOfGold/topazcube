export function reactive(name, object, callback, path = "", ID = null) {
  if (object === null || typeof object !== "object") {
    return object
  }
  for (const property in object) {
    object[property] = reactive(name, object[property], callback, path + "/" + property, ID)
  }

  return new Proxy(object, {
    get(target, property) {
      return Reflect.get(...arguments)
    },
    set(target, property, value) {
      let pn = path + "/" + property
      let newvalue = reactive(name, value, callback, pn, ID)
      callback(name, "replace", target, pn, newvalue, ID)
      return Reflect.set(target, property, newvalue)
    },
    deleteProperty(target, property) {
      let pn = path + "/" + property
      delete target[property]
      callback(name, "delete", target, pn, null, ID)
      return true
    },
  })
}

export function deepGet(obj, path) {
  let paths = ("" + path).split("/")
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
  let paths = ("" + path).split("/")
  let len = paths.length
  let i
  for (i = 0; i < len - 1; i++) {
    obj = obj[paths[i]]
  }
  obj[paths[i]] = value
}

// recursive clone oject, without properties that starts with _ (or __)

export function clonewo_(obj, excludeStart = "_") {
  if (obj === null || typeof obj !== "object") {
    return obj
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
      if (obj.hasOwnProperty(key) && !key.startsWith(excludeStart)) {
        if (typeof obj[key] === "object") {
          clone[key] = clonewo_(obj[key], excludeStart)
        } else {
          clone[key] = obj[key]
        }
      }
    }
  }

  return clone
}

export function msgop(op) {
  let nop = {}
  if (!op.o) {
    nop.op = "replace"
  } else {
    nop.op = {
      a: "add",
      r: "remove",
      d: "delete",
      t: "test",
    }[op.o]
  }
  nop.path = op.p
  nop.value = op.v
  return nop
}

export function opmsg(op, target, path, value) {
  let c = { p: path, v: value }
  if (op != "replace") {
    c.o = {
      add: "a",
      remove: "r",
      delete: "d",
      test: "t",
    }[op]
  }
  return c
}

var _lastUID = 1

export function sdate() {
  return (Date.now() / 1000 - 1715000000) | 0
}

export function getUID() {
  let uid = sdate()
  if (uid <= _lastUID) {
    uid = _lastUID + 1
  }
  _lastUID = uid
  return uid
}

// - Fixed point encoding/decoding functions
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

export function decode_uint32(byteArray, offset = 0) {
  let p = offset
  return ((byteArray[p++] & 0x7f) << 24) | (byteArray[p++] << 16) | (byteArray[p++] << 8) | byteArray[p]
}

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

export function decode_uint16(byteArray, offset = 0) {
  let p = offset
  return (byteArray[p++] << 8) | byteArray[p]
}

export function encode_fp248(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 256)
  const enc = encode_uint32(fp, byteArray, offset)
  if (float < 0) {
    enc[offset] |= 0x80
  }
  return enc
}

export function decode_fp248(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -256 : 256
  byteArray[offset] &= 0x7f
  const fp = decode_uint32(byteArray, offset)
  return fp / divider
}

export function encode_fp1616(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 65536)
  const enc = encode_uint32(fp, byteArray, offset)
  if (float < 0) {
    enc[offset] |= 0x80
  }
  return enc
}

export function decode_fp1616(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -65536 : 65536
  byteArray[offset] &= 0x7f
  const fp = decode_uint32(byteArray, offset)
  return fp / divider
}

export function encode_fp88(float, byteArray, offset = 0) {
  const fp = Math.round(Math.abs(float) * 256)
  const enc = encode_uint16(fp, byteArray, offset)
  if (float < 0) {
    enc[offset] |= 0x80
  }
  return enc
}

export function decode_fp88(byteArray, offset = 0) {
  const divider = (byteArray[offset] & 0x80) === 0x80 ? -256 : 256
  byteArray[offset] &= 0x7f
  const fp = decode_uint16(byteArray, offset)
  return fp / divider
}
