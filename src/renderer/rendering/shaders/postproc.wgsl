struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
}

struct Uniforms {
    canvasSize: vec2f,
    noiseParams: vec4f,  // x = size, y = offsetX, z = offsetY, w = fxaaEnabled
    ditherParams: vec4f, // x = enabled, y = colorLevels (32 = 5-bit PS1 style), z = unused, w = unused
    bloomParams: vec4f,  // x = enabled, y = intensity, z = radius (mip levels to sample), w = mipCount
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var inputTexture : texture_2d<f32>;
@group(0) @binding(2) var inputSampler : sampler;
@group(0) @binding(3) var noiseTexture : texture_2d<f32>;
@group(0) @binding(4) var noiseSampler : sampler;
@group(0) @binding(5) var bloomTexture : texture_2d<f32>;
@group(0) @binding(6) var bloomSampler : sampler;
@group(0) @binding(7) var guiTexture : texture_2d<f32>;
@group(0) @binding(8) var guiSampler : sampler;

// FXAA constants - more aggressive settings for visible effect
const FXAA_EDGE_THRESHOLD: f32 = 0.125;      // Lower = more edges detected (was 0.0625)
const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0156; // Lower = catch more subtle edges (was 0.0312)
const FXAA_SUBPIX_QUALITY: f32 = 1.0;        // Higher = more sub-pixel smoothing (was 0.75)

// Sample noise for dithering
fn sampleNoise(screenPos: vec2f) -> f32 {
    let noiseSize = i32(uniforms.noiseParams.x);
    let noiseOffsetX = i32(uniforms.noiseParams.y * f32(noiseSize));
    let noiseOffsetY = i32(uniforms.noiseParams.z * f32(noiseSize));

    let texCoord = vec2i(
        (i32(screenPos.x) + noiseOffsetX) % noiseSize,
        (i32(screenPos.y) + noiseOffsetY) % noiseSize
    );
    return textureLoad(noiseTexture, texCoord, 0).r;
}

// Convert RGB to luminance
fn rgb2luma(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.299, 0.587, 0.114));
}

// Load pixel with clamping
fn loadPixel(coord: vec2i) -> vec3f {
    let size = vec2i(textureDimensions(inputTexture, 0));
    let clampedCoord = clamp(coord, vec2i(0), size - vec2i(1));
    return textureLoad(inputTexture, clampedCoord, 0).rgb;
}

// Manual bilinear interpolation using textureLoad
fn sampleBilinear(uv: vec2f) -> vec3f {
    let texSize = vec2f(textureDimensions(inputTexture, 0));
    let texelPos = uv * texSize - 0.5;
    let baseCoord = vec2i(floor(texelPos));
    let frac = fract(texelPos);

    let c00 = loadPixel(baseCoord);
    let c10 = loadPixel(baseCoord + vec2i(1, 0));
    let c01 = loadPixel(baseCoord + vec2i(0, 1));
    let c11 = loadPixel(baseCoord + vec2i(1, 1));

    let c0 = mix(c00, c10, frac.x);
    let c1 = mix(c01, c11, frac.x);
    return mix(c0, c1, frac.y);
}

// Simplified FXAA without early returns or textureSample
fn fxaa(uv: vec2f) -> vec3f {
    let texSize = vec2f(textureDimensions(inputTexture, 0));
    let pixelCoord = vec2i(uv * texSize);

    // Sample center and neighbors
    let rgbM = loadPixel(pixelCoord);
    let rgbN = loadPixel(pixelCoord + vec2i(0, -1));
    let rgbS = loadPixel(pixelCoord + vec2i(0, 1));
    let rgbW = loadPixel(pixelCoord + vec2i(-1, 0));
    let rgbE = loadPixel(pixelCoord + vec2i(1, 0));
    let rgbNW = loadPixel(pixelCoord + vec2i(-1, -1));
    let rgbNE = loadPixel(pixelCoord + vec2i(1, -1));
    let rgbSW = loadPixel(pixelCoord + vec2i(-1, 1));
    let rgbSE = loadPixel(pixelCoord + vec2i(1, 1));

    // Get luma values
    let lumaM = rgb2luma(rgbM);
    let lumaN = rgb2luma(rgbN);
    let lumaS = rgb2luma(rgbS);
    let lumaW = rgb2luma(rgbW);
    let lumaE = rgb2luma(rgbE);
    let lumaNW = rgb2luma(rgbNW);
    let lumaNE = rgb2luma(rgbNE);
    let lumaSW = rgb2luma(rgbSW);
    let lumaSE = rgb2luma(rgbSE);

    // Luma range
    let lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaW, lumaE)));
    let lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaW, lumaE)));
    let lumaRange = lumaMax - lumaMin;

    // Edge threshold check - compute but don't early return
    let isEdge = lumaRange >= max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD);

    // Compute sub-pixel aliasing factor
    let lumaL = (lumaN + lumaS + lumaW + lumaE) * 0.25;
    let rangeL = abs(lumaL - lumaM);
    var blendL = max(0.0, (rangeL / max(lumaRange, 0.0001)) - 0.25) * (1.0 / 0.75);
    blendL = min(1.0, blendL) * blendL * FXAA_SUBPIX_QUALITY;

    // Determine edge orientation
    let edgeHorz = abs((lumaNW + lumaNE) - 2.0 * lumaN) +
                   2.0 * abs((lumaW + lumaE) - 2.0 * lumaM) +
                   abs((lumaSW + lumaSE) - 2.0 * lumaS);
    let edgeVert = abs((lumaNW + lumaSW) - 2.0 * lumaW) +
                   2.0 * abs((lumaN + lumaS) - 2.0 * lumaM) +
                   abs((lumaNE + lumaSE) - 2.0 * lumaE);
    let isHorizontal = edgeHorz >= edgeVert;

    // Choose gradient direction
    let luma1 = select(lumaE, lumaS, isHorizontal);
    let luma2 = select(lumaW, lumaN, isHorizontal);
    let gradient1 = abs(luma1 - lumaM);
    let gradient2 = abs(luma2 - lumaM);
    let is1Steepest = gradient1 >= gradient2;
    let gradientScaled = 0.25 * max(gradient1, gradient2);

    // Step direction
    let stepSign = select(-1.0, 1.0, is1Steepest);
    let lumaLocalAverage = 0.5 * (select(luma2, luma1, is1Steepest) + lumaM);

    // Search for edge endpoints (simplified - 4 steps each direction)
    let searchDir = select(vec2i(0, 1), vec2i(1, 0), isHorizontal);

    let luma1_1 = rgb2luma(loadPixel(pixelCoord - searchDir)) - lumaLocalAverage;
    let luma2_1 = rgb2luma(loadPixel(pixelCoord + searchDir)) - lumaLocalAverage;
    let luma1_2 = rgb2luma(loadPixel(pixelCoord - searchDir * 2)) - lumaLocalAverage;
    let luma2_2 = rgb2luma(loadPixel(pixelCoord + searchDir * 2)) - lumaLocalAverage;
    let luma1_3 = rgb2luma(loadPixel(pixelCoord - searchDir * 3)) - lumaLocalAverage;
    let luma2_3 = rgb2luma(loadPixel(pixelCoord + searchDir * 3)) - lumaLocalAverage;
    let luma1_4 = rgb2luma(loadPixel(pixelCoord - searchDir * 4)) - lumaLocalAverage;
    let luma2_4 = rgb2luma(loadPixel(pixelCoord + searchDir * 4)) - lumaLocalAverage;

    // Find distance to edge end - check from closest to farthest
    let reached1_1 = abs(luma1_1) >= gradientScaled;
    let reached1_2 = abs(luma1_2) >= gradientScaled;
    let reached1_3 = abs(luma1_3) >= gradientScaled;
    let reached2_1 = abs(luma2_1) >= gradientScaled;
    let reached2_2 = abs(luma2_2) >= gradientScaled;
    let reached2_3 = abs(luma2_3) >= gradientScaled;

    // Distance = first position where edge was found (or 4 if not found)
    let dist1 = select(select(select(4.0, 3.0, reached1_3), 2.0, reached1_2), 1.0, reached1_1);
    let dist2 = select(select(select(4.0, 3.0, reached2_3), 2.0, reached2_2), 1.0, reached2_1);

    // Get the luma at the edge end
    let lumaEnd1 = select(select(select(luma1_4, luma1_3, reached1_3), luma1_2, reached1_2), luma1_1, reached1_1);
    let lumaEnd2 = select(select(select(luma2_4, luma2_3, reached2_3), luma2_2, reached2_2), luma2_1, reached2_1);

    // Compute offset
    let distFinal = min(dist1, dist2);
    let edgeThickness = dist1 + dist2;
    let lumaEndCloser = select(lumaEnd2, lumaEnd1, dist1 < dist2);
    let correctVariation = (lumaEndCloser < 0.0) != (lumaM < lumaLocalAverage);
    var pixelOffset = select(0.0, -distFinal / max(edgeThickness, 0.0001) + 0.5, correctVariation);

    // Ensure minimum blending for detected edges
    let finalOffset = max(max(pixelOffset, blendL), 0.5);

    // Compute final UV
    let inverseVP = 1.0 / texSize;
    var finalUv = uv;
    let offsetAmount = finalOffset * stepSign;
    finalUv.x += select(offsetAmount * inverseVP.x, 0.0, isHorizontal);
    finalUv.y += select(0.0, offsetAmount * inverseVP.y, isHorizontal);

    // Sample at offset position using bilinear
    let offsetColor = sampleBilinear(finalUv);

    // Blend original with offset sample for smoother result
    // Also blend with perpendicular neighbors for better coverage
    let perpDir = select(vec2i(0, 1), vec2i(1, 0), isHorizontal);
    let perpColor1 = loadPixel(pixelCoord + perpDir);
    let perpColor2 = loadPixel(pixelCoord - perpDir);
    let neighborAvg = (perpColor1 + perpColor2) * 0.5;

    // Weighted blend: offset sample + neighbor average + original
    let fxaaColor = mix(mix(offsetColor, neighborAvg, 0.7), rgbM, 0.1);

    // Return original if not an edge, otherwise return FXAA result
    return select(rgbM, fxaaColor, isEdge);
}

// Sample bloom texture (already blurred by BloomPass)
// Masked by scene brightness - bloom only shows in darker areas around bright pixels
fn sampleBloom(uv: vec2f, sceneBrightness: f32) -> vec3f {
    let bloom = textureSample(bloomTexture, bloomSampler, uv).rgb;

    // Mask: bloom is visible where scene is dark, fades out where scene is bright
    // Using smooth falloff so bloom doesn't abruptly cut off
    let threshold = 0.5;  // Start fading bloom at this brightness
    let mask = saturate(1.0 - (sceneBrightness - threshold) / (1.0 - threshold));

    return bloom * mask * mask;  // Square for smoother falloff
}

// ACES tone mapping
fn aces_tone_map(hdr: vec3<f32>) -> vec3<f32> {
    let m1 = mat3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777,
    );
    let m2 = mat3x3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602,
    );
    let v = m1 * hdr;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(m2 * (a / b), vec3(0.0), vec3(1.0));
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var output : VertexOutput;
    let x = f32(vertexIndex & 1u) * 4.0 - 1.0;
    let y = f32(vertexIndex >> 1u) * 4.0 - 1.0;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    // Always sample - choose FXAA or direct based on uniform
    let fxaaEnabled = uniforms.noiseParams.w > 0.5;
    let fxaaColor = fxaa(input.uv);
    let directColor = textureSample(inputTexture, inputSampler, input.uv).rgb;
    var color = select(directColor, fxaaColor, fxaaEnabled);

    // Add bloom before tone mapping (in HDR space)
    // Bloom is masked by scene brightness - only shows in darker areas around bright pixels
    if (uniforms.bloomParams.x > 0.5) {
        let sceneBrightness = rgb2luma(color);
        let bloom = sampleBloom(input.uv, sceneBrightness);
        color += bloom * uniforms.bloomParams.y;  // y = intensity
    }

    var sdr = aces_tone_map(color);

    // Blend GUI overlay (after tone mapping, before dithering)
    // GUI is premultiplied alpha blending
    let gui = textureSample(guiTexture, guiSampler, input.uv);
    sdr = sdr * (1.0 - gui.a) + gui.rgb;

    // PS1-style ordered dithering with configurable color levels
    // Quantizes to reduced bit depth (e.g., 32 levels = 5-bit per channel)
    if (uniforms.ditherParams.x > 0.5 && uniforms.noiseParams.x > 0.0) {
        let levels = uniforms.ditherParams.y;
        let noise = sampleNoise(input.position.xy);

        // Scale color to target levels, add dither, round, scale back
        // dither value is -0.5 to +0.5, which shifts the rounding threshold
        let dither = noise - 0.5;
        sdr = floor(sdr * (levels - 1.0) + dither + 0.5) / (levels - 1.0);
        sdr = clamp(sdr, vec3f(0.0), vec3f(1.0));
    }

    return vec4<f32>(sdr, 1.0);
}
