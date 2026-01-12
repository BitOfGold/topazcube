// Volumetric Fog - Gaussian Blur
// Blurs the ray-marched fog for softer edges

struct Uniforms {
    direction: vec2f,
    texelSize: vec2f,
    radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var inputSampler: sampler;

// Full-screen triangle vertex shader
@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    let x = f32(vertexIndex & 1u) * 4.0 - 1.0;
    let y = f32(vertexIndex >> 1u) * 4.0 - 1.0;

    output.position = vec4f(x, y, 0.0, 1.0);
    output.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return output;
}

// Gaussian weight function
fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;
    let radius = uniforms.radius;
    let sigma = radius / 3.0;  // 99.7% of Gaussian within 3 sigma

    // Sample center pixel
    var color = textureSample(inputTexture, inputSampler, uv);
    var totalWeight = 1.0;

    // Step size (sample every 1.5 texels for efficiency)
    let stepSize = 1.5;
    let numSamples = i32(ceil(radius / stepSize));

    // Blur direction (scaled by texel size)
    let dir = uniforms.direction * uniforms.texelSize;

    // Bidirectional sampling
    for (var i = 1; i <= numSamples; i++) {
        let offset = f32(i) * stepSize;
        let weight = gaussian(offset, sigma);

        // Positive direction
        let uvPos = uv + dir * offset;
        let samplePos = textureSample(inputTexture, inputSampler, uvPos);
        color += samplePos * weight;

        // Negative direction
        let uvNeg = uv - dir * offset;
        let sampleNeg = textureSample(inputTexture, inputSampler, uvNeg);
        color += sampleNeg * weight;

        totalWeight += weight * 2.0;
    }

    // Normalize
    color /= totalWeight;

    return color;
}
