// Volumetric Fog - Composite
// Blends the blurred fog into the scene (additive)

struct Uniforms {
    canvasSize: vec2f,
    renderSize: vec2f,
    texelSize: vec2f,
    // Brightness-based fog attenuation (like bloom)
    brightnessThreshold: f32,  // Scene luminance where fog starts fading
    minVisibility: f32,        // Minimum fog visibility over bright surfaces (0-1)
    skyBrightness: f32,        // Virtual brightness for sky pixels (depth at far plane)
    _pad: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var sceneTexture: texture_2d<f32>;
@group(0) @binding(2) var fogTexture: texture_2d<f32>;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var depthTexture: texture_depth_2d;

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

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;

    // Sample scene color at full resolution
    let sceneColor = textureSample(sceneTexture, linearSampler, uv).rgb;

    // Sample fog at reduced resolution (will be upsampled by linear filtering)
    let fog = textureSample(fogTexture, linearSampler, uv);

    // Sample depth to detect sky (depth near 1.0 = far plane = sky)
    let depthCoord = vec2i(input.position.xy);
    let depth = textureLoad(depthTexture, depthCoord, 0);

    // Calculate scene luminance for brightness-based attenuation
    var luminance = dot(sceneColor, vec3f(0.299, 0.587, 0.114));

    // Sky detection: if depth is very close to 1.0 (far plane), treat as bright
    // This ensures fog is less visible over sky even if sky color isn't bright
    let isSky = depth > 0.9999;
    if (isSky) {
        luminance = max(luminance, uniforms.skyBrightness);
    }

    // Fog visibility decreases over bright surfaces (like bloom behavior)
    // Full visibility at dark, reduced visibility at bright, but never zero
    let threshold = uniforms.brightnessThreshold;
    let minVis = uniforms.minVisibility;

    // Soft falloff using inverse relationship
    // At luminance=0: visibility=1
    // At luminance=threshold: visibility≈0.5
    // At luminance>>threshold: visibility→minVis
    let falloff = 1.0 / (1.0 + luminance / max(threshold, 0.01));
    let fogVisibility = mix(minVis, 1.0, falloff);

    // Additive blend with brightness attenuation
    let finalColor = sceneColor + fog.rgb * fogVisibility;

    return vec4f(finalColor, 1.0);
}
