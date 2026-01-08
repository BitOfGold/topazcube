// Texture class for WebGPU texture management


// https://enkimute.github.io/hdrpng.js/
//From https://raw.githubusercontent.com/enkimute/hdrpng.js/refs/heads/master/hdrpng.js

function loadHDR( url ) {
  function m(a,b) { for (var i in b) a[i]=b[i]; return a; };

  let p = new Promise((resolve, reject) => {
    var req = m(new XMLHttpRequest(),{responseType:"arraybuffer"});
    req.onerror = reject.bind(req,false);
    req.onload  = function() {
      if (this.status>=400) return this.onerror();
      var header='',pos=0,d8=new Uint8Array(this.response),format;
    // read header.
      while (!header.match(/\n\n[^\n]+\n/g)) header += String.fromCharCode(d8[pos++]);
    // check format.
      format = header.match(/FORMAT=(.*)$/m)[1];
      if (format!='32-bit_rle_rgbe') return console.warn('unknown format : '+format),this.onerror();
    // parse resolution
      var rez=header.split(/\n/).reverse()[1].split(' '), width=rez[3]*1, height=rez[1]*1;
    // Create image.
      var img=new Uint8Array(width*height*4),ipos=0;
    // Read all scanlines
      for (var j=0; j<height; j++) {
        var rgbe=d8.slice(pos,pos+=4),scanline=[];
        if (rgbe[0]!=2||(rgbe[1]!=2)||(rgbe[2]&0x80)) {
          var len=width,rs=0; pos-=4; while (len>0) {
            img.set(d8.slice(pos,pos+=4),ipos);
            if (img[ipos]==1&&img[ipos+1]==1&&img[ipos+2]==1) {
              for (img[ipos+3]<<rs; i>0; i--) {
                img.set(img.slice(ipos-4,ipos),ipos);
                ipos+=4;
                len--
              }
              rs+=8;
            } else { len--; ipos+=4; rs=0; }
          }
        } else {
          if ((rgbe[2]<<8)+rgbe[3]!=width) return console.warn('HDR line mismatch ..'),this.onerror();
          for (var i=0;i<4;i++) {
              var ptr=i*width,ptr_end=(i+1)*width,buf,count;
              while (ptr<ptr_end){
                  buf = d8.slice(pos,pos+=2);
                  if (buf[0] > 128) { count = buf[0]-128; while(count-- > 0) scanline[ptr++] = buf[1]; }
                              else { count = buf[0]-1; scanline[ptr++]=buf[1]; while(count-->0) scanline[ptr++]=d8[pos++]; }
              }
          }
          for (var i=0;i<width;i++) { img[ipos++]=scanline[i]; img[ipos++]=scanline[i+width]; img[ipos++]=scanline[i+2*width]; img[ipos++]=scanline[i+3*width]; }
        }
      }
      resolve({data:img,width: width,height: height});
    }
    req.open("GET",url,true);
    req.send(null);
    return req;
  })
  return p
}

//https://webgpufundamentals.org/webgpu/lessons/webgpu-importing-textures.html

const numMipLevels = (...sizes) => {
    const maxSize = Math.max(sizes[0], sizes[1]);
    const mipFactor = 2
    return 1 + Math.floor(Math.log(maxSize) / Math.log(mipFactor));
}

const generateMips = (() => {
    let sampler, module, pipeline;
    let modules = {};
    const pipelines = {};
    const samplers = {};

    return function generateMips(device, texture, rgbe = false) {
      let v = rgbe ? 'rgbe' : 'rgb';
      v = v + texture.format
      if (modules[v]) {
        module = modules[v]
        pipeline = pipelines[v]
        sampler = samplers[v]
      } else {
        let code = ''
        code = `
          struct VSOutput {
            @builtin(position) position: vec4f,
            @location(0) texcoord: vec2f,
          };

          @vertex fn vs(
            @builtin(vertex_index) vertexIndex : u32
          ) -> VSOutput {
            let pos = array(
              // 1st triangle
              vec2f( 0.0,  0.0),  // center
              vec2f( 1.0,  0.0),  // right, center
              vec2f( 0.0,  1.0),  // center, top

              // 2nd triangle
              vec2f( 0.0,  1.0),  // center, top
              vec2f( 1.0,  0.0),  // right, center
              vec2f( 1.0,  1.0),  // right, top
            );

            var vsOutput: VSOutput;
            let xy = pos[vertexIndex];
            vsOutput.position = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
            vsOutput.texcoord = vec2f(xy.x, 1.0 - xy.y);
            return vsOutput;
          }

          @group(0) @binding(0) var ourSampler: sampler;
          @group(0) @binding(1) var ourTexture: texture_2d<f32>;

          fn gaussian3x3(uv: vec2f) -> vec4f {
            let texelSize = 1.0 / vec2f(textureDimensions(ourTexture));
            var color = vec3f(0.0);

            // 3x3 Gaussian kernel weights
            let weights = array(
              0.0625, 0.125, 0.0625,  // 1/16, 2/16, 1/16
              0.125,  0.25,  0.125,   // 2/16, 4/16, 2/16
              0.0625, 0.125, 0.0625   // 1/16, 2/16, 1/16
            );

            for (var i = -1; i <= 1; i++) {
              for (var j = -1; j <= 1; j++) {
                let offset = vec2f(f32(i), f32(j)) * texelSize * 1.0;
                let weight = weights[(i + 1) * 3 + (j + 1)];

                var rgbe = textureSample(ourTexture, ourSampler, uv + offset);
                var rgb = rgbe.rgb * pow(2.0, rgbe.a * 255.0 - 128.0);
                color += rgb * weight;
              }
            }

            // Encode back to RGBE format
            let maxComponent = max(max(color.r, color.g), color.b);
            let exponent = ceil(log2(maxComponent));
            let mantissa = color * pow(2.0, -exponent);
            return vec4f(mantissa, (exponent + 128.0) / 255.0);
          }

          @fragment fn fs(fsInput: VSOutput) -> @location(0) vec4f {
          `
          if (rgbe) {
              code += `return gaussian3x3(fsInput.texcoord);\n`
          } else {
              code += `return textureSample(ourTexture, ourSampler, fsInput.texcoord);\n`
          }
          code += `
          }
        `
        module = device.createShaderModule({
          label: 'textured quad shaders for mip level generation',
          code: code
        });
        modules[v] = module

        sampler = device.createSampler({
          minFilter: 'linear',
          magFilter: 'linear',
          addressModeU: 'mirror-repeat',
          addressModeV: 'mirror-repeat',
        });
        samplers[v] = sampler

        pipeline = device.createRenderPipeline({
          label: 'mip level generator pipeline',
          layout: 'auto',
          vertex: {
            module,
          },
          fragment: {
            module,
            targets: [{ format: texture.format }],
          },
        });
        pipelines[v] = pipeline
      }

      const encoder = device.createCommandEncoder({
        label: 'mip gen encoder',
      });

      let width = texture.width;
      let height = texture.height;
      let baseMipLevel = 0;
      while (width > 1 || height > 1) {
        width = Math.max(1, width / 2 | 0);
        height = Math.max(1, height / 2 | 0);

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: texture.createView({baseMipLevel, mipLevelCount: 1}) },
          ],
        });

        ++baseMipLevel;

        const renderPassDescriptor = {
          label: 'our basic canvas renderPass',
          colorAttachments: [
            {
              view: texture.createView({baseMipLevel, mipLevelCount: 1}),
              loadOp: 'clear',
              storeOp: 'store',
            },
          ],
        };

        const pass = encoder.beginRenderPass(renderPassDescriptor);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(6);  // call our vertex shader 6 times
        pass.end();
      }

      const commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);
    };
  })();

class Texture {
    texture = null
    sampler = null
    engine = null
    sourceUrl = ''

    constructor(engine = null) {
        this.engine = engine
    }

    // Static async factory method
    static async fromImage(engine, url_or_image, options = {}) {
        options = {
            flipY: true,
            srgb: true, // Whether to interpret image as sRGB (true) or linear (false)
            generateMips: true,
            ...options
        }
        const { device, options: webgpuOptions } = engine

        let imageBitmap
        let isHDR = false
        if (typeof url_or_image === 'string') {
            if (url_or_image.endsWith('.hdr')) {
                isHDR = true
                imageBitmap = await loadHDR(url_or_image)
            } else {
                const response = await fetch(url_or_image)
                imageBitmap = await createImageBitmap(await response.blob())
            }
        } else {
            imageBitmap = url_or_image
        }

        let format = options.srgb ? "rgba8unorm-srgb" : "rgba8unorm"
        if (isHDR) {
            format = "rgba8unorm"
        }
        const texture = await device.createTexture({
            size: [imageBitmap.width, imageBitmap.height, 1],
            mipLevelCount: options.generateMips ? numMipLevels(imageBitmap.width, imageBitmap.height) : 1,
            format: format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        })

        let source = imageBitmap
        if (isHDR) {
          source = imageBitmap.data
          device.queue.writeTexture(
            { texture, mipLevel: 0, origin: [0, 0, 0], aspect: "all" },
            source,
            { offset: 0, bytesPerRow: imageBitmap.width * 4, rowsPerImage: imageBitmap.height },
            [ imageBitmap.width, imageBitmap.height, 1 ]
          )
        } else {
          device.queue.copyExternalImageToTexture(
            { source: source, flipY: options.flipY },
            { texture: texture },
            [imageBitmap.width, imageBitmap.height]
          )
        }

        let mipCount = options.generateMips ? numMipLevels(imageBitmap.width, imageBitmap.height) : 1
        if (options.generateMips) {
            generateMips(device, texture, isHDR)
        }

        let t = Texture._createTexture(engine, texture, {
            forceLinear: isHDR,
            addressModeU: options.addressModeU,
            addressModeV: options.addressModeV
        })
        t.isHDR = isHDR
        t.mipCount = mipCount
        if (typeof url_or_image === 'string') {
          t.sourceUrl = url_or_image
        }
        return t
    }

    static async fromColor(engine, color) {
        // Parse hex color to rgb values
        color = color.replace('#', '')
        const r = parseInt(color.substring(0,2), 16) / 255
        const g = parseInt(color.substring(2,4), 16) / 255
        const b = parseInt(color.substring(4,6), 16) / 255

        return Texture.fromRGBA(engine, r, g, b, 1.0)
    }

    /**
     * Create a texture from raw RGBA Uint8Array data
     * @param {Engine} engine
     * @param {Uint8Array} data - RGBA pixel data (4 bytes per pixel)
     * @param {number} width - Texture width
     * @param {number} height - Texture height
     * @param {Object} options - { srgb, generateMips }
     */
    static async fromRawData(engine, data, width, height, options = {}) {
        options = {
            srgb: true,
            generateMips: false,
            ...options
        }
        const { device } = engine

        const format = options.srgb ? "rgba8unorm-srgb" : "rgba8unorm"
        const mipCount = options.generateMips ? numMipLevels(width, height) : 1

        const texture = await device.createTexture({
            size: [width, height, 1],
            mipLevelCount: mipCount,
            format: format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        })

        device.queue.writeTexture(
            { texture, mipLevel: 0, origin: [0, 0, 0] },
            data,
            { bytesPerRow: width * 4, rowsPerImage: height },
            [width, height, 1]
        )

        if (options.generateMips && mipCount > 1) {
            generateMips(device, texture, false)
        }

        let t = Texture._createTexture(engine, texture)
        t.mipCount = mipCount
        return t
    }

    /**
     * Create a 1x1 texture from RGBA values (0-1 range)
     */
    static async fromRGBA(engine, r, g, b, a = 1.0) {
        const { device } = engine

        // Create 1x1 pixel data with the color (as Uint8)
        const colorData = new Uint8Array([
            Math.round(r * 255),
            Math.round(g * 255),
            Math.round(b * 255),
            Math.round(a * 255)
        ])

        const texture = await device.createTexture({
            size: [1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        })

        device.queue.writeTexture(
            { texture },
            colorData,
            { bytesPerRow: 4 },
            [1, 1]
        )

        let t = Texture._createTexture(engine, texture)
        return t
    }

    /**
     * Load HDR texture from RGBM JPG pair (RGB with sRGB gamma + log multiplier)
     * RGB stores actual color values (gamma corrected) - values <= 1.0 stored directly
     * Multiplier: black = 1.0, white = 32768, logarithmic encoding
     * @param {string} rgbUrl - URL to RGB JPG (e.g., 'probe_01.jpg')
     * @param {string} multUrl - URL to multiplier JPG (e.g., 'probe_01.mult.jpg'), or null to auto-derive
     * @param {Object} options - { generateMips: true }
     */
    static async fromJPGPair(engine, rgbUrl, multUrl = null, options = {}) {
        options = {
            generateMips: true,
            ...options
        }
        const { device } = engine

        // Auto-derive multiplier URL if not provided
        if (!multUrl) {
            multUrl = rgbUrl.replace(/\.jpg$/i, '.mult.jpg')
        }

        // Load both images
        const [rgbResponse, multResponse] = await Promise.all([
            fetch(rgbUrl),
            fetch(multUrl)
        ])

        const [rgbBlob, multBlob] = await Promise.all([
            rgbResponse.blob(),
            multResponse.blob()
        ])

        const [rgbBitmap, multBitmap] = await Promise.all([
            createImageBitmap(rgbBlob),
            createImageBitmap(multBlob)
        ])

        const width = rgbBitmap.width
        const height = rgbBitmap.height

        // Draw to canvases to get pixel data
        const rgbCanvas = document.createElement('canvas')
        rgbCanvas.width = width
        rgbCanvas.height = height
        const rgbCtx = rgbCanvas.getContext('2d')
        rgbCtx.drawImage(rgbBitmap, 0, 0)
        const rgbImageData = rgbCtx.getImageData(0, 0, width, height)

        const multCanvas = document.createElement('canvas')
        multCanvas.width = width
        multCanvas.height = height
        const multCtx = multCanvas.getContext('2d')
        multCtx.drawImage(multBitmap, 0, 0)
        const multImageData = multCtx.getImageData(0, 0, width, height)

        // RGBM decoding parameters (must match encoder in ProbeCapture.saveAsJPG)
        const LOG_MULT_MAX = 15  // log2(32768)
        const SRGB_GAMMA = 2.2

        // Convert RGBM to RGBE for GPU texture
        // We store as RGBE because that's what the shader expects
        const rgbeData = new Uint8Array(width * height * 4)
        for (let i = 0; i < width * height; i++) {
            const idx = i * 4

            // Get sRGB gamma encoded RGB (0-255 -> 0-1)
            const rGamma = rgbImageData.data[idx] / 255
            const gGamma = rgbImageData.data[idx + 1] / 255
            const bGamma = rgbImageData.data[idx + 2] / 255

            // Convert from sRGB to linear
            const rLinear = Math.pow(rGamma, SRGB_GAMMA)
            const gLinear = Math.pow(gGamma, SRGB_GAMMA)
            const bLinear = Math.pow(bGamma, SRGB_GAMMA)

            // Decode multiplier: 0 = 1.0, 255 = 32768, logarithmic
            const multByte = multImageData.data[idx]
            const multNorm = multByte / 255  // 0 to 1
            const logMult = multNorm * LOG_MULT_MAX  // 0 to 15
            const multiplier = Math.pow(2, logMult)  // 1 to 32768

            // Reconstruct HDR color
            const r = rLinear * multiplier
            const g = gLinear * multiplier
            const b = bLinear * multiplier

            // Convert to RGBE for GPU
            const maxVal = Math.max(r, g, b)
            if (maxVal < 1e-32) {
                rgbeData[idx] = 0
                rgbeData[idx + 1] = 0
                rgbeData[idx + 2] = 0
                rgbeData[idx + 3] = 0
            } else {
                const exp = Math.ceil(Math.log2(maxVal))
                const scale = Math.pow(2, -exp) * 255

                rgbeData[idx] = Math.min(255, Math.max(0, Math.round(r * scale)))
                rgbeData[idx + 1] = Math.min(255, Math.max(0, Math.round(g * scale)))
                rgbeData[idx + 2] = Math.min(255, Math.max(0, Math.round(b * scale)))
                rgbeData[idx + 3] = exp + 128
            }
        }

        // Create texture
        const mipCount = options.generateMips ? numMipLevels(width, height) : 1
        const texture = device.createTexture({
            size: [width, height],
            mipLevelCount: mipCount,
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        })

        device.queue.writeTexture(
            { texture },
            rgbeData,
            { bytesPerRow: width * 4 },
            [width, height]
        )

        // Generate mips with RGBE-aware filtering
        if (options.generateMips) {
            generateMips(device, texture, true) // true = RGBE mode
        }

        let t = Texture._createTexture(engine, texture, {forceLinear: true})
        t.isHDR = true
        t.mipCount = mipCount
        return t
    }

    static async renderTarget(engine, format = 'rgba8unorm', width = null, height = null) {
        const { device, canvas, options } = engine
        const w = width ?? canvas.width
        const h = height ?? canvas.height
        const texture =  await device.createTexture({
            size: [w, h],
            format: format,
            //sampleCount: options.msaa > 1 ? options.msaa : 1,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
        });
        let t = Texture._createTexture(engine, texture)
        t.renderTarget = true
        t.format = format
        return t
    }

    static async depth(engine, width = null, height = null) {
        const { device, canvas, options } = engine
        const w = width ?? canvas.width
        const h = height ?? canvas.height

        const texture = await device.createTexture({
            size: [w, h, 1],
            //sampleCount: options.msaa > 1 ? options.msaa : 1,
            format: "depth32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        })

        let t = Texture._createTexture(engine, texture)
        t.depth = true
        return t
    }

    static _createTexture(engine, texture, options = {}) {
        const { device, canvas, settings } = engine
        let t = new Texture(engine)
        t.texture = texture
        t.view = texture.createView()
        // Default to mirror-repeat to avoid black border artifacts
        const addressModeU = options.addressModeU || 'mirror-repeat'
        const addressModeV = options.addressModeV || 'mirror-repeat'
        if (settings.rendering.nearestFiltering && !options.forceLinear) {
            t.sampler = device.createSampler({
                magFilter: "nearest",
                minFilter: "nearest",
                addressModeU: addressModeU,
                addressModeV: addressModeV,
                mipmapFilter: "nearest",
                maxAnisotropy: 1,
            })
        } else {
            t.sampler = device.createSampler({
                magFilter: "linear",
                minFilter: "linear",
                addressModeU: addressModeU,
                addressModeV: addressModeV,
                mipmapFilter: "linear",
                maxAnisotropy: 1,
            })
        }
        return t
    }
}

export { Texture, generateMips, numMipLevels }
