import { Packr } from 'msgpackr';
import { FLOAT32_OPTIONS } from 'msgpackr';
const { ALWAYS } = FLOAT32_OPTIONS;

let packr = new Packr({
  useFloat32: ALWAYS
});

export function encode(obj: any): Uint8Array {
  return packr.pack(obj)
}

export function decode(data: Uint8Array): any {
  return packr.unpack(data)
}

type ReactiveCallback = (name: string, operation: string, target: any, path: string, value: any) => void;

export function reactive(name: string, object: any, callback: ReactiveCallback, path: string = '', excludedProperties: Record<string, boolean> | false = false): any {
  if (object === null || typeof object !== 'object') {
    //console.log('--- Type not object', typeof object)
    return object
  }

  function isReactive(p: string): boolean {
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
    get(target: any, property: string | symbol, receiver: any): any { // ...arguments
      return Reflect.get(target, property, receiver)
    },
    set(target: any, property: string | symbol, value: any): boolean {
      let newvalue: any
      let pn = path + '/' + String(property)
      if (isReactive(String(property))) {
        newvalue = reactive(name, value, callback, pn, excludedProperties)
        callback(name, 'replace', target, pn, newvalue)
      } else {
        newvalue = value
      }
      return Reflect.set(target, property, newvalue)
    },
    deleteProperty(target: any, property: string | symbol): boolean {
      let pn = path + '/' + String(property)
      if (isReactive(String(property))) {
        callback(name, 'remove', target, pn, null)
      }
      delete target[property]
      return true
    },
  })

}

export function deepGet(obj: any, path: string): any {
  //path = path.replace(/^\/+/, '')
  let paths = ('' + path).split('/').filter((p) => p)
  let len = paths.length
  for (let i = 0; i < len; i++) {
    if (obj[paths[i]!] == undefined) {
      return undefined
    } else {
      obj = obj[paths[i]!]
    }
  }
  return obj
}

export function deepSet(obj: any, path: string, value: any): void {
  //path = path.replace(/^\/+/, '')
  let paths = ('' + path).split('/').filter((p) => p)
  let len = paths.length
  let i: number
  for (i = 0; i < len - 1; i++) {
    obj = obj[paths[i]!]
  }
  obj[paths[i]!] = value
}

// recursive clone oject, without properties that starts with _ (or __)

export function clonewo_(obj: any, excludeStart: string | Record<string, boolean> = '_'): any {
  if (obj === null || typeof obj !== 'object') {
    return obj
  }

  function isExcluded(key: string): boolean {
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
    Array.from(obj.entries()).forEach(([key, value]) => {
      mapClone.set(clonewo_(key, excludeStart), clonewo_(value, excludeStart))
    })
    return mapClone
  }

  let clone: any
  if (Array.isArray(obj)) {
    clone = []
    for (let i = 0; i < obj.length; i++) {
      clone[i] = clonewo_(obj[i], excludeStart)
    }
  } else {
    clone = {} as Record<string, any>
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

export function limitPrecision(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map(limitPrecision)
  } else if (obj !== null && typeof obj === 'object') {
    const result: Record<string, any> = {}
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

export function msgop(op: any): any {
  let nop: any = {}
  if (!op.o) {
    nop.op = 'replace'
  } else {
    nop.op = ({
      a: 'add',
      r: 'remove',
      d: 'delete',
      t: 'test',
    } as Record<string, string>)[op.o]
  }
  nop.path = op.p
  nop.value = op.v
  return nop
}

export function opmsg(op: string, target: any, path: string, value: any): any {
  let c: any = { p: path, v: value }
  if (op != 'replace') {
    c.o = ({
      add: 'a',
      remove: 'r',
      delete: 'd',
      test: 't',
    } as Record<string, string>)[op]
  }
  return c
}

// a function that converts an int to a hexa string
// (8 characters long)
export function int2hex(int: number): string {
  return int.toString(16)
}

// a function that converts a hexa string to an int
export function hex2int(str: string): bigint {
  if (str.length % 2) {
    str = '0' + str
  }
  return BigInt('0x' + str)
}

// - Fixed point encoding/decoding functions

// 32-bit unsigned integer encoding
export function encode_uint32(uint: number, byteArray?: Uint8Array, offset: number = 0): Uint8Array {
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
export function decode_uint32(byteArray: Uint8Array, offset: number = 0): number {
  let p = offset
  return (
    ((byteArray[p++]! & 0x7f) << 24) |
    (byteArray[p++]! << 16) |
    (byteArray[p++]! << 8) |
    byteArray[p]!
  )
}

// 24-bit unsigned integer encoding
export function encode_uint24(uint: number, byteArray?: Uint8Array, offset: number = 0): Uint8Array {
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
export function decode_uint24(byteArray: Uint8Array, offset: number = 0): number {
  let p = offset
  return (
    (byteArray[p++]! << 16) |
    (byteArray[p++]! << 8) |
    byteArray[p]!
  )
}

// 16-bit unsigned integer encoding
export function encode_uint16(uint: number, byteArray?: Uint8Array, offset: number = 0): Uint8Array {
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
export function decode_uint16(byteArray: Uint8Array, offset: number = 0): number {
  let p = offset
  return (byteArray[p++]! << 8) | byteArray[p]!
}

// 24.8 bit ====================================================================

// 24.8-bit fixed point encoding
export function encode_fp248(float: number, byteArray?: Uint8Array, offset: number = 0): void {
  const fp = Math.round(Math.abs(float) * 256)
  encode_uint32(fp, byteArray, offset)
  if (float < 0 && byteArray) {
    byteArray[offset]! |= 0x80
  }
}

// 24.8-bit fixed point decoding
export function decode_fp248(byteArray: Uint8Array, offset: number = 0): number {
  const divider = (byteArray[offset]! & 0x80) === 0x80 ? -256 : 256
  byteArray[offset]! &= 0x7f
  const fp = decode_uint32(byteArray, offset)
  return fp / divider
}

// 16.8 bit ====================================================================

// 16.8-bit fixed point encoding (3 bytes)
export function encode_fp168(float: number, byteArray?: Uint8Array, offset: number = 0): void {
  const fp = Math.round(Math.abs(float) * 256)
  encode_uint24(fp, byteArray, offset)
  if (float < 0 && byteArray) {
    byteArray[offset]! |= 0x80
  }
}

// 16.8-bit fixed point decoding (3 bytes)
export function decode_fp168(byteArray: Uint8Array, offset: number = 0): number {
  const divider = (byteArray[offset]! & 0x80) === 0x80 ? -256 : 256
  byteArray[offset]! &= 0x7f
  const fp = decode_uint24(byteArray, offset)
  return fp / divider
}

// 16.16 bit ===================================================================

// 16.16-bit fixed point encoding
export function encode_fp1616(float: number, byteArray?: Uint8Array, offset: number = 0): void {
  const fp = Math.round(Math.abs(float) * 65536)
  encode_uint32(fp, byteArray, offset)
  if (float < 0 && byteArray) {
    byteArray[offset]! |= 0x80
  }
}

// 16.16-bit fixed point decoding
export function decode_fp1616(byteArray: Uint8Array, offset: number = 0): number {
  const divider = (byteArray[offset]! & 0x80) === 0x80 ? -65536 : 65536
  byteArray[offset]! &= 0x7f
  const fp = decode_uint32(byteArray, offset)
  return fp / divider
}

// 8.8 bit =====================================================================

// 8.8-bit fixed point encoding
export function encode_fp88(float: number, byteArray?: Uint8Array, offset: number = 0): void {
  const fp = Math.round(Math.abs(float) * 256)
  encode_uint16(fp, byteArray, offset)
  if (float < 0 && byteArray) {
    byteArray[offset]! |= 0x80
  }
}

// 8.8-bit fixed point decoding
export function decode_fp88(byteArray: Uint8Array, offset: number = 0): number {
  const divider = (byteArray[offset]! & 0x80) === 0x80 ? -256 : 256
  byteArray[offset]! &= 0x7f
  const fp = decode_uint16(byteArray, offset)
  return fp / divider
}

// 4.12 bit ====================================================================

// 4.12-bit fixed point encoding
export function encode_fp412(float: number, byteArray?: Uint8Array, offset: number = 0): void {
  const fp = Math.round(Math.abs(float) * 4096)
  encode_uint16(fp, byteArray, offset)
  if (float < 0 && byteArray) {
    byteArray[offset]! |= 0x80
  }
}

// 4.12-bit fixed point decoding
export function decode_fp412(byteArray: Uint8Array, offset: number = 0): number {
  const divider = (byteArray[offset]! & 0x80) === 0x80 ? -4096 : 4096
  byteArray[offset]! &= 0x7f
  const fp = decode_uint16(byteArray, offset)
  return fp / divider
}

// 1.7 bit =====================================================================

// 1.7-bit fixed point encoding
export function encode_fp17(float: number, byteArray: Uint8Array, offset: number = 0): void {
  const fp = Math.round(Math.abs(float) * 128)
  byteArray[offset] = fp
  if (float < 0) {
    byteArray[offset] |= 0x80
  }
}

// 1.7-bit fixed point decoding
export function decode_fp17(byteArray: Uint8Array, offset: number = 0): number {
  const divider = (byteArray[offset]! & 0x80) === 0x80 ? -128.0 : 128.0
  byteArray[offset]! &= 0x7f
  return byteArray[offset]! / divider
}
