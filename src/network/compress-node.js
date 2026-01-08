import { promisify } from 'util'
import { gzip, gunzip, constants } from 'zlib'

const MIN_COMPRESSED_BUFFER_SIZE = 256
const MAX_COMPRESSED_BUFFER_SIZE = 999999

const lib_compress = promisify(gzip)
const lib_decompress = promisify(gunzip)

export async function compress(buffer) {
  if (buffer.byteLength <= MIN_COMPRESSED_BUFFER_SIZE || buffer.byteLength >= MAX_COMPRESSED_BUFFER_SIZE) return buffer
  try {
    const cbytes = await lib_compress(buffer, {
      level: constants.Z_BEST_SPEED
    })
    const cbuffer = Buffer.from(cbytes)
    return cbuffer
  } catch (error) {
    console.error('Error compressing buffer:', error)
    return buffer
  }
}

export async function decompress(buffer) {
  try {
    const cbytes = await lib_decompress(buffer)
    const cbuffer = Buffer.from(cbytes)
    return cbuffer
  } catch (error) {
    console.error('Error decompressing buffer:', error)
    return buffer
  }
}
