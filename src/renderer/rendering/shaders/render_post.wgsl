// Render Post-Processing shader
// Combines SSGI/Planar Reflection with lighting output in HDR space

struct RenderPostUniforms {
    screenSize: vec2f,
    ssgiEnabled: f32,
    ssgiIntensity: f32,
    planarEnabled: f32,
    planarGroundLevel: f32,
    planarRoughnessCutoff: f32,
    planarNormalPerturbation: f32,
    noiseSize: f32,
    frameCount: f32,
    planarBlurSamples: f32,
    planarIntensity: f32,
    renderScale: f32,  // Render resolution multiplier for scaling pixel-based effects
    ssgiSaturateLevel: f32,  // Logarithmic saturation level for indirect light
    planarDistanceFade: f32,  // Distance from ground for full reflection (meters)
    ambientCaptureEnabled: f32,  // Enable 6-directional ambient capture
    ambientCaptureIntensity: f32,  // Intensity of ambient capture contribution
    ambientCaptureFadeDistance: f32,  // Distance from camera where ambient fades to zero
    cameraNear: f32,  // Camera near plane
    cameraFar: f32,  // Camera far plane
    ambientCaptureSaturateLevel: f32,  // Logarithmic saturation level for ambient capture
}

@group(0) @binding(0) var<uniform> uniforms: RenderPostUniforms;
@group(0) @binding(1) var lightingOutput: texture_2d<f32>;   // HDR lighting result
@group(0) @binding(2) var ssgiTexture: texture_2d<f32>;      // SSGI result (half-res)
@group(0) @binding(3) var gbufferARM: texture_2d<f32>;       // For metallic/roughness
@group(0) @binding(4) var gbufferNormal: texture_2d<f32>;    // For surface normal
@group(0) @binding(5) var planarReflection: texture_2d<f32>; // Planar reflection (mirrored render)
@group(0) @binding(6) var noiseTexture: texture_2d<f32>;     // Noise texture for jitter
@group(0) @binding(7) var linearSampler: sampler;            // Mirror-repeat sampler
@group(0) @binding(8) var nearestSampler: sampler;
@group(0) @binding(9) var<storage, read> ambientCapture: array<vec4f, 6>;  // 6 directional colors: up, down, left, right, front, back
@group(0) @binding(10) var gbufferDepth: texture_2d<f32>;    // Depth buffer for distance calculation

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

// Get noise value at pixel position with frame-based offset
fn getNoise(pixelCoord: vec2f) -> vec4f {
    let noiseSize = uniforms.noiseSize;
    // Offset by frame count for temporal variation (0 if not animated)
    let frameOffset = vec2f(
        fract(uniforms.frameCount * 0.618033988749895),  // Golden ratio
        fract(uniforms.frameCount * 0.381966011250105)
    ) * noiseSize;
    let coord = vec2i((pixelCoord + frameOffset) % noiseSize);
    return textureLoad(noiseTexture, coord, 0);
}

// Bilateral upscale for half-res textures
fn bilateralUpscale(tex: texture_2d<f32>, uv: vec2f, fullSize: vec2f) -> vec4f {
    // Simple bilinear for now - could add depth-aware bilateral
    return textureSampleLevel(tex, linearSampler, uv, 0.0);
}

// Vogel disk sample pattern - gives good 2D distribution
fn vogelDiskSample(sampleIndex: i32, numSamples: i32, rotation: f32) -> vec2f {
    let goldenAngle = 2.399963229728653;  // pi * (3 - sqrt(5))
    let r = sqrt((f32(sampleIndex) + 0.5) / f32(numSamples));
    let theta = f32(sampleIndex) * goldenAngle + rotation;
    return vec2f(r * cos(theta), r * sin(theta));
}

// Sample planar reflection with roughness-based blur using blue noise jitter
fn samplePlanarReflection(baseUV: vec2f, roughness: f32, pixelCoord: vec2f) -> vec3f {
    let numSamples = i32(uniforms.planarBlurSamples);

    // Calculate blur radius based on roughness
    // Minimum 1px blur, scaling up to 128px at roughness cutoff
    // Scale by renderScale to maintain consistent visual blur at different render resolutions
    let maxBlurPixels = 128.0 * uniforms.renderScale;
    let minBlurPixels = 1.0;
    let normalizedRoughness = roughness / uniforms.planarRoughnessCutoff;
    let blurRadius = max(minBlurPixels, normalizedRoughness * normalizedRoughness * maxBlurPixels);

    // Convert blur radius to UV space
    let blurUV = blurRadius / uniforms.screenSize;

    // Get blue noise for this pixel - use for rotation and radius jitter
    let noise = getNoise(pixelCoord);
    let rotationAngle = noise.r * 6.283185307;  // 0 to 2*PI rotation
    let radiusJitter = noise.g;  // 0 to 1 radius variation

    var colorSum = vec3f(0.0);

    for (var i = 0; i < numSamples; i++) {
        // Get Vogel disk sample position (unit disk, rotated by blue noise)
        let diskSample = vogelDiskSample(i, numSamples, rotationAngle);

        // Apply radius jitter per-sample using additional noise channels
        let sampleRadius = mix(0.5, 1.0, fract(radiusJitter + f32(i) * 0.618033988749895));

        // Calculate final offset
        let offset = diskSample * sampleRadius * blurUV;

        // Sample with mirror-repeat sampler handles edge cases automatically
        let jitteredUV = baseUV + offset;
        let sampleColor = textureSampleLevel(planarReflection, linearSampler, jitteredUV, 0.0).rgb;
        colorSum += sampleColor;
    }

    // Apply reflection intensity (default 0.9 = 90% brightness for realism)
    return (colorSum / f32(numSamples)) * uniforms.planarIntensity;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;
    let pixelCoord = input.position.xy;

    // Sample lighting output
    var color = textureSampleLevel(lightingOutput, nearestSampler, uv, 0.0);

    // Sample material properties for blending
    let arm = textureSampleLevel(gbufferARM, nearestSampler, uv, 0.0);
    let roughness = arm.g;
    let metallic = arm.b;

    // Sample world-space normal (xyz) and world Y position (w)
    let normalSample = textureSampleLevel(gbufferNormal, nearestSampler, uv, 0.0);
    let normal = normalSample.xyz;
    let worldY = normalSample.w;

    // Apply Planar Reflection (for water/floor)
    // Blends between environment reflection (in lighting) and planar reflection based on:
    // - Roughness: r=0 → full planar, r=cutoff/2 → half, r=cutoff → full env
    // - Normal direction: up → full weight, 45° → 0.5×, horizontal/down → 0
    // - Distance from ground: within distanceFade → full, 2x distanceFade → 0
    if (uniforms.planarEnabled > 0.5 && roughness < uniforms.planarRoughnessCutoff && normal.y > 0.0) {
        // Calculate distance from ground level (in any direction)
        let distFromGround = abs(worldY - uniforms.planarGroundLevel);
        let fadeDist = uniforms.planarDistanceFade;

        // Distance-based weight: full at fadeDist, fade to 0 at 2x fadeDist
        // within fadeDist → 1.0, at 2x fadeDist → 0.0
        let distanceWeight = saturate(1.0 - (distFromGround - fadeDist) / fadeDist);

        // Skip if too far from ground
        if (distanceWeight > 0.0) {
            // Flip UV.y to compensate for projection Y flip in PlanarReflectionPass
            var reflectUV = vec2f(uv.x, 1.0 - uv.y);

            // Apply normal perturbation for water ripples
            let perturbAmount = uniforms.planarNormalPerturbation;
            reflectUV.x += normal.x * perturbAmount;
            reflectUV.y += normal.z * perturbAmount;

            // Sample planar reflection with roughness-based blur
            let reflectionColor = samplePlanarReflection(reflectUV, roughness, pixelCoord);

            // Calculate roughness-based weight (aggressive fade near cutoff)
            // r≤0.40 → 1.0 (full planar), r=0.45 → 0.5, r≥0.50 → 0.0 (full env)
            let fadeRange = 0.10;
            let fadeStart = uniforms.planarRoughnessCutoff - fadeRange;
            let roughnessWeight = saturate(1.0 - (roughness - fadeStart) / fadeRange);

            // Calculate normal-based weight (how "up-facing" is the surface)
            // up (y=1) → 1.0, 45° from up (y≈0.707) → 0.5, horizontal (y=0) → 0.0
            // Formula: 1 - acos(y) * (2/PI) maps angle linearly
            let upDot = clamp(normal.y, 0.0, 1.0);
            let normalWeight = saturate(1.0 - acos(upDot) * 0.6366197723675814);  // 2/PI

            // Final blend: planar vs environment (already in color from lighting pass)
            let planarBlend = roughnessWeight * normalWeight * distanceWeight;
            color = vec4f(mix(color.rgb, reflectionColor, planarBlend), color.a);
        }
    }

    // Apply SSGI
    if (uniforms.ssgiEnabled > 0.5) {
        let ssgi = bilateralUpscale(ssgiTexture, uv, uniforms.screenSize);

        // Add indirect lighting - ssgi.a = 1 means valid data
        let ssgiBlend = ssgi.a * uniforms.ssgiIntensity;

        // SSGI adds to diffuse component (less effect on metallic surfaces)
        // Also apply AO - indirect light is occluded in corners/crevices
        let ao = arm.r;
        let diffuseBlend = (1.0 - metallic * 0.5) * ssgiBlend * ao;

        // Calculate raw indirect light contribution
        var indirectLight = ssgi.rgb * diffuseBlend;

        // Apply logarithmic saturation to prevent indirect light from going too bright
        // Formula: saturateLevel * (1 - exp(-value / saturateLevel))
        // This smoothly approaches saturateLevel as value increases
        let satLevel = uniforms.ssgiSaturateLevel;
        if (satLevel > 0.0) {
            let indirectLum = max(indirectLight.r, max(indirectLight.g, indirectLight.b));
            if (indirectLum > 0.0) {
                let saturatedLum = satLevel * (1.0 - exp(-indirectLum / satLevel));
                indirectLight = indirectLight * (saturatedLum / indirectLum);
            }
        }

        color = vec4f(color.rgb + indirectLight, color.a);
    }

    // Apply Ambient Capture (6-directional sky/environment sampling)
    // Adds ambient light based on captured sky visibility in each direction
    // Main benefit: sky-facing surfaces get blue tint when outdoors, darker when under roof
    // Effect is local: fades to zero at ambientCaptureFadeDistance from camera
    if (uniforms.ambientCaptureEnabled > 0.5) {
        // Calculate linear depth for distance fade
        let depth = textureSampleLevel(gbufferDepth, nearestSampler, uv, 0.0).r;
        let near = uniforms.cameraNear;
        let far = uniforms.cameraFar;
        // GBuffer stores linear depth normalized to [0,1]
        let linearDepth = near + depth * (far - near);

        // Distance fade: full effect at 0, fades to 0 at fadeDistance (smooth curve)
        let fadeDistance = uniforms.ambientCaptureFadeDistance;
        // Square root fade: starts fading immediately but with gentler curve than linear
        // At distance 0: fade = 1.0, at fadeDistance: fade = 0
        let normalizedDist = saturate(linearDepth / fadeDistance);
        let distanceFade = 1.0 - sqrt(normalizedDist);

        // Only process if within fade distance
        if (distanceFade > 0.0) {
            let N = normal;

            // Sample ambient from 6 directions weighted by how much the surface faces each direction
            // Directions: 0=up, 1=down, 2=left(-X), 3=right(+X), 4=front(+Z), 5=back(-Z)
            var ambient = vec3f(0.0);

            // Up/Down (Y axis)
            ambient += ambientCapture[0].rgb * max(N.y, 0.0);   // up: surface facing up receives sky
            ambient += ambientCapture[1].rgb * max(-N.y, 0.0);  // down: surface facing down receives ground

            // Left/Right (X axis in world space)
            ambient += ambientCapture[2].rgb * max(-N.x, 0.0);  // left (-X)
            ambient += ambientCapture[3].rgb * max(N.x, 0.0);   // right (+X)

            // Front/Back (Z axis in world space)
            ambient += ambientCapture[4].rgb * max(N.z, 0.0);   // front (+Z)
            ambient += ambientCapture[5].rgb * max(-N.z, 0.0);  // back (-Z)

            // Apply AO from ARM texture (ambient only affects diffuse surfaces)
            let ao = arm.r;
            let diffuseFactor = 1.0 - metallic * 0.5;  // Metallic surfaces less affected

            // Calculate ambient contribution
            var ambientContrib = ambient * ao * diffuseFactor * uniforms.ambientCaptureIntensity * distanceFade;

            // Apply logarithmic saturation to prevent ambient from going too bright
            // Formula: saturateLevel * (1 - exp(-value / saturateLevel))
            // This smoothly approaches saturateLevel as value increases
            let ambientSatLevel = uniforms.ambientCaptureSaturateLevel;
            if (ambientSatLevel > 0.0) {
                let ambientLum = max(ambientContrib.r, max(ambientContrib.g, ambientContrib.b));
                if (ambientLum > 0.0) {
                    let saturatedLum = ambientSatLevel * (1.0 - exp(-ambientLum / ambientSatLevel));
                    ambientContrib = ambientContrib * (saturatedLum / ambientLum);
                }
            }

            // Add ambient contribution
            color = vec4f(color.rgb + ambientContrib, color.a);
        }
    }

    return color;
}
