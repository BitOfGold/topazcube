import { promisify }  from 'util'
import { gzip, gunzip, constants }  from 'zlib'

const MIN_COMPRESSED_BUFFER_SIZE = 256
const MAX_COMPRESSED_BUFFER_SIZE = 999999

const lib_compress = promisify(gzip)
const lib_decompress = promisify(gunzip)

export async function compress(buffer:Uint8Array<ArrayBufferLike>) {
  if (buffer.byteLength <= MIN_COMPRESSED_BUFFER_SIZE || buffer.byteLength >= MAX_COMPRESSED_BUFFER_SIZE) return buffer
  try {
    let t1 = Date.now()
    let cbytes = await lib_compress(buffer, {
      level: constants.Z_BEST_SPEED
    })
    let t2 = Date.now()
    let cbuffer = Buffer.from(cbytes)
    let t3 = Date.now()

    //console.log(`Node compression ${buffer.byteLength} -> ${cbuffer.byteLength}, time: ${t2 - t1}ms`)

    return cbuffer
  } catch (error) {
    console.error('Error compressing buffer:', error)
    return buffer
  }
}

export async function decompress(buffer:Uint8Array<ArrayBufferLike>) {
  try {
    let cbytes = await lib_decompress(buffer)
    let cbuffer = Buffer.from(cbytes)
    return cbuffer
  } catch (error) {
    console.error('Error decompressing buffer:', error)
    return buffer
  }
}
