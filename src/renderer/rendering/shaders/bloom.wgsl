// Bloom bright pass shader
// Extracts bright and emissive pixels from HDR lighting output
// Uses exponential falloff so dark pixels contribute exponentially less

struct Uniforms {
    threshold: f32,        // Brightness threshold (e.g., 0.8)
    softThreshold: f32,    // Soft knee (0 = hard cutoff, 1 = very soft)
    intensity: f32,        // Overall bloom intensity
    emissiveBoost: f32,    // Extra boost for emissive pixels
    maxBrightness: f32,    // Clamp input brightness (prevents specular halos)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;  // HDR lighting output
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

// Calculate brightness as max RGB (treats all colors equally for bloom threshold)
fn brightness(color: vec3f) -> f32 {
    return max(max(color.r, color.g), color.b);
}

// Soft threshold with knee curve
// Creates smooth transition around threshold instead of hard cutoff
fn softThresholdCurve(brightness: f32, threshold: f32, knee: f32) -> f32 {
    let soft = threshold * knee;
    let softMin = threshold - soft;
    let softMax = threshold + soft;

    if (brightness <= softMin) {
        // Below threshold - exponential falloff
        // Instead of 0, use exponential curve so bright-ish pixels still contribute slightly
        let ratio = brightness / max(softMin, 0.001);
        return pow(ratio, 4.0) * brightness;  // Very aggressive falloff
    } else if (brightness >= softMax) {
        // Above threshold - full contribution
        return brightness;
    } else {
        // In the soft knee region - smooth interpolation
        let t = (brightness - softMin) / (softMax - softMin);
        let smoothT = t * t * (3.0 - 2.0 * t);  // Smoothstep
        return mix(pow(brightness / threshold, 4.0) * brightness, brightness, smoothT);
    }
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    var color = textureSample(inputTexture, inputSampler, input.uv).rgb;

    // Clamp extremely bright values (specular highlights) to prevent excessive bloom
    let bright = brightness(color);
    if (bright > uniforms.maxBrightness) {
        color = color * (uniforms.maxBrightness / bright);
    }
    let clampedBright = min(bright, uniforms.maxBrightness);

    // Apply soft threshold with exponential falloff
    let contribution = softThresholdCurve(clampedBright, uniforms.threshold, uniforms.softThreshold);

    // Calculate the extraction factor (how much of the original color to keep)
    let factor = contribution / max(clampedBright, 0.001);

    // Extract bloom color
    var bloomColor = color * factor * uniforms.intensity;

    // Note: Emissive is already included in the HDR color from lighting pass
    // Very bright pixels (brightness > 1.0) are likely emissive or highly lit
    // Give them extra boost based on how much they exceed 1.0
    let emissiveFactor = max(0.0, clampedBright - 1.0);
    bloomColor *= 1.0 + emissiveFactor * uniforms.emissiveBoost;

    return vec4f(bloomColor, 1.0);
}
