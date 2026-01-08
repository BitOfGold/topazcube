// Bloom blur shader - Separable Gaussian blur
// Run twice: once horizontal, once vertical

struct Uniforms {
    direction: vec2f,     // (1,0) for horizontal, (0,1) for vertical
    texelSize: vec2f,     // 1.0 / textureSize
    blurRadius: f32,      // Blur radius in pixels
    padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    // Full-screen triangle
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );
    var uvs = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0)
    );

    var output: VertexOutput;
    output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
    output.uv = uvs[vertexIndex];
    return output;
}

// Gaussian weight function
fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;
    let direction = uniforms.direction;
    let texelSize = uniforms.texelSize;
    let radius = uniforms.blurRadius;

    // Sigma is radius / 3 for good coverage (99.7% of gaussian within 3 sigma)
    let sigma = radius / 3.0;

    // Sample center
    var color = textureSample(inputTexture, inputSampler, uv).rgb;
    var totalWeight = 1.0;

    // Use incremental offsets for better cache coherency
    // Sample in both directions from center
    let stepSize = 1.5;  // Sample every 1.5 pixels for quality/performance balance
    let numSamples = i32(ceil(radius / stepSize));

    for (var i = 1; i <= numSamples; i++) {
        let offset = f32(i) * stepSize;
        let weight = gaussian(offset, sigma);

        // Positive direction
        let uvPos = uv + direction * texelSize * offset;
        color += textureSample(inputTexture, inputSampler, uvPos).rgb * weight;

        // Negative direction
        let uvNeg = uv - direction * texelSize * offset;
        color += textureSample(inputTexture, inputSampler, uvNeg).rgb * weight;

        totalWeight += weight * 2.0;
    }

    return vec4f(color / totalWeight, 1.0);
}
