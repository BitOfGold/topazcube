// Shadow map generation shader - depth only rendering

struct Uniforms {
    lightViewProjection: mat4x4f,
    lightPosition: vec3f,
    lightType: f32, // 0 = directional, 1 = point, 2 = spot
    lightDirection: vec3f,      // Light direction (for surface bias)
    surfaceBias: f32,           // Expand triangles along normals (meters)
    skinEnabled: f32,           // 1.0 if skinning enabled, 0.0 otherwise
    numJoints: f32,             // Number of joints in the skin
    _pad: vec2f,
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
@group(0) @binding(1) var jointTexture: texture_2d<f32>;
@group(0) @binding(2) var jointSampler: sampler;

// Get a 4x4 matrix from the joint texture
fn getJointMatrix(jointIndex: u32) -> mat4x4f {
    let row = i32(jointIndex);
    let col0 = textureLoad(jointTexture, vec2i(0, row), 0);
    let col1 = textureLoad(jointTexture, vec2i(1, row), 0);
    let col2 = textureLoad(jointTexture, vec2i(2, row), 0);
    let col3 = textureLoad(jointTexture, vec2i(3, row), 0);
    return mat4x4f(col0, col1, col2, col3);
}

// Apply skinning to a position
fn applySkinning(position: vec3f, joints: vec4u, weights: vec4f) -> vec3f {
    var skinnedPos = vec3f(0.0);

    let m0 = getJointMatrix(joints.x);
    let m1 = getJointMatrix(joints.y);
    let m2 = getJointMatrix(joints.z);
    let m3 = getJointMatrix(joints.w);

    skinnedPos += (m0 * vec4f(position, 1.0)).xyz * weights.x;
    skinnedPos += (m1 * vec4f(position, 1.0)).xyz * weights.y;
    skinnedPos += (m2 * vec4f(position, 1.0)).xyz * weights.z;
    skinnedPos += (m3 * vec4f(position, 1.0)).xyz * weights.w;

    return skinnedPos;
}

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

    // Apply skinning if enabled
    var localPos = input.position;
    if (uniforms.skinEnabled > 0.5) {
        let weightSum = input.weights.x + input.weights.y + input.weights.z + input.weights.w;
        if (weightSum > 0.001) {
            localPos = applySkinning(input.position, input.joints, input.weights);
        }
    }

    // Transform to world space
    let worldPos = modelMatrix * vec4f(localPos, 1.0);

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
