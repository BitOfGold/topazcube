const browserFormat = 'gzip'

const MIN_COMPRESSED_BUFFER_SIZE = 256

export async function compress(buffer) {
  if (buffer.byteLength <= MIN_COMPRESSED_BUFFER_SIZE) return buffer
  if (typeof CompressionStream !== 'undefined') {
    const cs = new CompressionStream(browserFormat)
    const compressed = await new Response(
      new Blob([buffer]).stream().pipeThrough(cs)
    ).arrayBuffer()
    return compressed
  } else {
    throw new Error('CompressionStream not supported')
  }
}

export async function decompress(buffer) {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const ds = new DecompressionStream(browserFormat)
      const decompressed = await new Response(
        new Blob([buffer]).stream().pipeThrough(ds)
      ).arrayBuffer()
      //console.log(` ${buffer.byteLength} -> ${decompressed.byteLength}`)
      return decompressed
    } catch (e) {
      //console.error('Decompression failed:', e)
      return buffer
    }
  } else {
    throw new Error('DecompressionStream not supported')
  }
}
