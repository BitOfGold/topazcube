// Depth copy shader - converts depth32float to r32float
// Used to copy GBuffer depth to Hi-Z base level

@group(0) @binding(0) var depthTexture: texture_depth_2d;
@group(0) @binding(1) var outputTexture: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
    let size = textureDimensions(outputTexture);

    if (globalId.x >= size.x || globalId.y >= size.y) {
        return;
    }

    let depth = textureLoad(depthTexture, vec2i(globalId.xy), 0);
    textureStore(outputTexture, vec2i(globalId.xy), vec4f(depth, 0.0, 0.0, 0.0));
}
