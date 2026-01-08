// Shadow map generation shader - depth only rendering

struct Uniforms {
    lightViewProjection: mat4x4f,
    lightPosition: vec3f,
    lightType: f32, // 0 = directional, 1 = point, 2 = spot
    lightDirection: vec3f,      // Light direction (for surface bias)
    surfaceBias: f32,           // Expand triangles along normals (meters)
}

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
    @location(3) color: vec4f,
    @location(4) weights: vec4f,
    @location(5) joints: vec4u,
    // Instance data
    @location(6) model0: vec4f,
    @location(7) model1: vec4f,
    @location(8) model2: vec4f,
    @location(9) model3: vec4f,
    @location(10) instancePosRadius: vec4f,
}

struct VertexOutput {
    @invariant @builtin(position) position: vec4f,
    @location(0) depth: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Reconstruct model matrix from instance data
    let modelMatrix = mat4x4f(
        input.model0,
        input.model1,
        input.model2,
        input.model3
    );

    // Transform to world space
    let worldPos = modelMatrix * vec4f(input.position, 1.0);

    // Transform to light clip space
    var clipPos = uniforms.lightViewProjection * worldPos;

    // Apply surface bias - scale shadow projection to make shadows larger
    // surfaceBias is treated as a percentage (0.01 = 1% larger shadows)
    if (uniforms.surfaceBias > 0.0) {
        let scale = 1.0 + uniforms.surfaceBias;
        clipPos = vec4f(clipPos.xy * scale, clipPos.z, clipPos.w);
    }

    output.position = clipPos;

    // For point lights, we might need linear depth
    if (uniforms.lightType > 0.5) {
        let lightToVertex = worldPos.xyz - uniforms.lightPosition;
        output.depth = length(lightToVertex);
    } else {
        output.depth = output.position.z / output.position.w;
    }

    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) f32 {
    // For VSM (Variance Shadow Maps), we could output depth and depth^2
    // For now, simple depth output
    return input.depth;
}
