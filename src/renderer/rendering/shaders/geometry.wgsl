struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
    @location(3) color: vec4f,
    @location(4) weights: vec4f,
    @location(5) joints: vec4u,
}

struct VertexOutput {
    @invariant @builtin(position) position: vec4f,  // @invariant ensures shared edges compute identical positions
    @location(0) worldPos: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
    @location(3) viewZ: f32,           // Linear view-space depth for logarithmic encoding
    @location(4) currClipPos: vec4f,   // Current frame clip position (for motion vectors)
    @location(5) prevClipPos: vec4f,   // Previous frame clip position (for motion vectors)
    @location(6) instanceColor: vec4f, // Per-instance color tint for sprites
    @location(7) anchorY: f32,         // Entity anchor Y position (for billboard clip plane testing)
}

struct Uniforms {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
    prevViewProjMatrix: mat4x4f,  // Previous frame view-projection for motion vectors
    mipBias: f32,
    skinEnabled: f32,  // 1.0 if skinning enabled, 0.0 otherwise
    numJoints: f32,    // Number of joints in the skin
    near: f32,         // Camera near plane
    far: f32,          // Camera far plane
    jitterFadeDistance: f32,  // Distance at which jitter fades to 0 (meters)
    jitterOffset: vec2f,      // TAA jitter offset in pixels
    screenSize: vec2f,        // Screen dimensions for pixel-to-clip conversion
    emissionFactor: vec4f,
    clipPlaneY: f32,          // Y level for clip plane
    clipPlaneEnabled: f32,    // 1.0 to enable clip plane, 0.0 to disable
    clipPlaneDirection: f32,  // 1.0 = discard below clipPlaneY, -1.0 = discard above clipPlaneY
    pixelRounding: f32,       // Pixel grid size (0=off, 1=every pixel, 2=every 2px, etc.)
    pixelExpansion: f32,      // Sub-pixel expansion to convert gaps to overlaps (0=off, 0.05=default)
    positionRounding: f32,    // If > 0, round view-space position to this precision (simulates fixed-point)
    alphaHashEnabled: f32,    // 1.0 to enable alpha hashing/dithering for cutout transparency
    alphaHashScale: f32,      // Scale factor for alpha hash (default 1.0, higher = more opaque)
    luminanceToAlpha: f32,    // 1.0 to derive alpha from base color luminance (black=transparent)
    noiseSize: f32,           // Size of noise texture (8 for bayer, 64/128 for blue noise)
    noiseOffsetX: f32,        // Animated noise offset X
    noiseOffsetY: f32,        // Animated noise offset Y
    cameraPosition: vec3f,    // World-space camera position for distance fade
    distanceFadeStart: f32,   // Distance where fade begins (A)
    distanceFadeEnd: f32,     // Distance where fade completes (B) - object invisible
    // Billboard/sprite uniforms
    billboardMode: f32,       // 0=none, 1=center (spherical), 2=bottom (cylindrical), 3=horizontal
    billboardCameraRight: vec3f,  // Camera right vector for billboarding
    _pad1: f32,
    billboardCameraUp: vec3f,     // Camera up vector for billboarding
    _pad2: f32,
    billboardCameraForward: vec3f, // Camera forward vector for billboarding
    specularBoost: f32,            // Per-material specular boost (0-1), scales the 3-light specular effect
}

struct InstanceData {
    modelMatrix: mat4x4f,
    posRadius: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var albedoTexture: texture_2d<f32>;
@group(0) @binding(2) var albedoSampler: sampler;
@group(0) @binding(3) var normalTexture: texture_2d<f32>;
@group(0) @binding(4) var normalSampler: sampler;
@group(0) @binding(5) var ambientTexture: texture_2d<f32>;
@group(0) @binding(6) var ambientSampler: sampler;
@group(0) @binding(7) var rmTexture: texture_2d<f32>;
@group(0) @binding(8) var rmSampler: sampler;
@group(0) @binding(9) var emissionTexture: texture_2d<f32>;
@group(0) @binding(10) var emissionSampler: sampler;
@group(0) @binding(11) var jointTexture: texture_2d<f32>;
@group(0) @binding(12) var jointSampler: sampler;
@group(0) @binding(13) var prevJointTexture: texture_2d<f32>;  // Previous frame joint matrices for skinned motion vectors
@group(0) @binding(14) var noiseTexture: texture_2d<f32>;    // Noise texture for alpha hashing (blue noise or bayer)

// Sample noise at screen position (tiled with animated offset)
fn sampleNoise(screenPos: vec2f) -> f32 {
    let noiseSize = i32(uniforms.noiseSize);
    let noiseOffsetX = i32(uniforms.noiseOffsetX * f32(noiseSize));
    let noiseOffsetY = i32(uniforms.noiseOffsetY * f32(noiseSize));

    let texCoord = vec2i(
        (i32(screenPos.x) + noiseOffsetX) % noiseSize,
        (i32(screenPos.y) + noiseOffsetY) % noiseSize
    );
    return textureLoad(noiseTexture, texCoord, 0).r;
}

// Get a 4x4 matrix from the joint texture
// Each row of the texture stores one joint's matrix (4 RGBA pixels = 16 floats)
fn getJointMatrix(jointIndex: u32) -> mat4x4f {
    let row = i32(jointIndex);
    let col0 = textureLoad(jointTexture, vec2i(0, row), 0);
    let col1 = textureLoad(jointTexture, vec2i(1, row), 0);
    let col2 = textureLoad(jointTexture, vec2i(2, row), 0);
    let col3 = textureLoad(jointTexture, vec2i(3, row), 0);
    return mat4x4f(col0, col1, col2, col3);
}

// Get a 4x4 matrix from the previous frame joint texture (for motion vectors)
fn getPrevJointMatrix(jointIndex: u32) -> mat4x4f {
    let row = i32(jointIndex);
    let col0 = textureLoad(prevJointTexture, vec2i(0, row), 0);
    let col1 = textureLoad(prevJointTexture, vec2i(1, row), 0);
    let col2 = textureLoad(prevJointTexture, vec2i(2, row), 0);
    let col3 = textureLoad(prevJointTexture, vec2i(3, row), 0);
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

// Apply skinning with previous frame joint matrices (for motion vectors)
fn applySkinningPrev(position: vec3f, joints: vec4u, weights: vec4f) -> vec3f {
    var skinnedPos = vec3f(0.0);

    let m0 = getPrevJointMatrix(joints.x);
    let m1 = getPrevJointMatrix(joints.y);
    let m2 = getPrevJointMatrix(joints.z);
    let m3 = getPrevJointMatrix(joints.w);

    skinnedPos += (m0 * vec4f(position, 1.0)).xyz * weights.x;
    skinnedPos += (m1 * vec4f(position, 1.0)).xyz * weights.y;
    skinnedPos += (m2 * vec4f(position, 1.0)).xyz * weights.z;
    skinnedPos += (m3 * vec4f(position, 1.0)).xyz * weights.w;

    return skinnedPos;
}

// Apply skinning to a normal (using 3x3 rotation part of matrices)
fn applySkinningNormal(normal: vec3f, joints: vec4u, weights: vec4f) -> vec3f {
    var skinnedNormal = vec3f(0.0);

    let m0 = getJointMatrix(joints.x);
    let m1 = getJointMatrix(joints.y);
    let m2 = getJointMatrix(joints.z);
    let m3 = getJointMatrix(joints.w);

    // Extract 3x3 rotation matrices
    let r0 = mat3x3f(m0[0].xyz, m0[1].xyz, m0[2].xyz);
    let r1 = mat3x3f(m1[0].xyz, m1[1].xyz, m1[2].xyz);
    let r2 = mat3x3f(m2[0].xyz, m2[1].xyz, m2[2].xyz);
    let r3 = mat3x3f(m3[0].xyz, m3[1].xyz, m3[2].xyz);

    skinnedNormal += (r0 * normal) * weights.x;
    skinnedNormal += (r1 * normal) * weights.y;
    skinnedNormal += (r2 * normal) * weights.z;
    skinnedNormal += (r3 * normal) * weights.w;

    return normalize(skinnedNormal);
}

@vertex
fn vertexMain(
    input: VertexInput,
    @builtin(instance_index) instanceIdx : u32,
    @location(6) instanceModelMatrix0: vec4f,
    @location(7) instanceModelMatrix1: vec4f,
    @location(8) instanceModelMatrix2: vec4f,
    @location(9) instanceModelMatrix3: vec4f,
    @location(10) posRadius: vec4f,
    @location(11) uvTransform: vec4f,     // UV offset (xy) and scale (zw) for sprite sheets
    @location(12) instanceColor: vec4f,   // Per-instance color tint
) -> VertexOutput {
    var output: VertexOutput;

    // Construct instance model matrix
    let instanceModelMatrix = mat4x4f(
        instanceModelMatrix0,
        instanceModelMatrix1,
        instanceModelMatrix2,
        instanceModelMatrix3
    );

    var localPos = input.position;
    var prevLocalPos = input.position;  // For motion vectors
    var localNormal = input.normal;
    var billboardWorldPos: vec3f;
    var useBillboardWorldPos = false;

    // Apply billboard transform if enabled
    if (uniforms.billboardMode > 0.5) {
        // Extract entity position (translation) from instance matrix
        let entityPos = instanceModelMatrix[3].xyz;

        // Extract scale from matrix (length of each column's xyz)
        let scaleX = length(instanceModelMatrix[0].xyz);
        let scaleY = length(instanceModelMatrix[1].xyz);

        if (uniforms.billboardMode < 1.5) {
            // Mode 1: Center (spherical billboard) - faces camera, centered on entity
            // Calculate from camera position and world-up to avoid view matrix flip issues
            let toCamera = uniforms.cameraPosition - entityPos;
            let toCameraDist = length(toCamera);

            var right: vec3f;
            var up: vec3f;

            if (toCameraDist > 0.001) {
                let forward = toCamera / toCameraDist;
                // Right = cross(worldUp, forward)
                right = cross(vec3f(0.0, 1.0, 0.0), forward);
                let rightLen = length(right);
                if (rightLen < 0.001) {
                    // Camera directly above/below - use camera right as fallback
                    right = uniforms.billboardCameraRight;
                    up = vec3f(0.0, 0.0, 1.0);  // Use Z as up when looking straight down
                } else {
                    right = right / rightLen;
                    // Up is perpendicular to both forward and right
                    up = cross(forward, right);
                }
            } else {
                right = uniforms.billboardCameraRight;
                up = vec3f(0.0, 1.0, 0.0);
            }

            let offset = right * input.position.x * scaleX + up * input.position.y * scaleY;
            billboardWorldPos = entityPos + offset;
            useBillboardWorldPos = true;
            localNormal = vec3f(0.0, 1.0, 0.0);  // Normal points up

        } else if (uniforms.billboardMode < 2.5) {
            // Mode 2: Bottom (face camera position) - tilts toward camera, pivots at bottom
            // Like a sunflower facing the sun - when viewed from above, nearly horizontal
            let toCamera = uniforms.cameraPosition - entityPos;
            let toCameraDist = length(toCamera);

            var right: vec3f;
            var billUp: vec3f;

            if (toCameraDist > 0.001) {
                let forward = toCamera / toCameraDist;  // Direction toward camera
                // Right = cross(worldUp, forward)
                right = cross(vec3f(0.0, 1.0, 0.0), forward);
                let rightLen = length(right);
                if (rightLen < 0.001) {
                    // Camera directly above/below - use camera right as fallback
                    right = uniforms.billboardCameraRight;
                    billUp = uniforms.billboardCameraUp;
                } else {
                    right = right / rightLen;
                    // Billboard "up" is perpendicular to both forward and right
                    billUp = cross(forward, right);
                }
            } else {
                // Camera at entity position - use camera vectors as fallback
                right = uniforms.billboardCameraRight;
                billUp = uniforms.billboardCameraUp;
            }

            let offset = right * input.position.x * scaleX + billUp * input.position.y * scaleY;
            billboardWorldPos = entityPos + offset;
            useBillboardWorldPos = true;
            localNormal = vec3f(0.0, 1.0, 0.0);  // Normal points up

        } else {
            // Mode 3: Horizontal - fixed in world space on XZ plane
            // Uses entity's model matrix rotation (yaw only in practice)
            // Don't modify localPos - let model matrix apply the transform
            // Geometry is already on XZ plane from Geometry.billboardQuad('horizontal')
            localNormal = vec3f(0.0, 1.0, 0.0);  // Normal points up
        }
    }

    // Apply skinning if enabled
    if (uniforms.skinEnabled > 0.5) {
        // Check if weights sum to something meaningful
        let weightSum = input.weights.x + input.weights.y + input.weights.z + input.weights.w;
        if (weightSum > 0.001) {
            localPos = applySkinning(input.position, input.joints, input.weights);
            prevLocalPos = applySkinningPrev(input.position, input.joints, input.weights);
            localNormal = applySkinningNormal(input.normal, input.joints, input.weights);
        }
    }

    // Apply instance transform
    // For billboard modes 1 and 2, we computed worldPos directly (bypassing matrix rotation)
    // For billboard mode 3 and non-billboard, use standard matrix transform
    var worldPos: vec3f;
    if (useBillboardWorldPos) {
        worldPos = billboardWorldPos;
    } else {
        worldPos = (instanceModelMatrix * vec4f(localPos, 1.0)).xyz;
    }
    var viewPos = uniforms.viewMatrix * vec4f(worldPos, 1.0);

    // Check if this instance allows rounding (negative radius = no rounding)
    let allowRounding = posRadius.w >= 0.0;

    // Position rounding: simulate fixed-point by rounding view-space position
    // Only round X and Y - rounding Z can cause clipping issues when vertices are near camera
    if (allowRounding && uniforms.positionRounding > 0.0) {
        let snap = uniforms.positionRounding;
        viewPos = vec4f(
            floor(viewPos.x / snap) * snap,
            floor(viewPos.y / snap) * snap,
            viewPos.z,
            viewPos.w
        );
    }

    var clipPos = uniforms.projectionMatrix * viewPos;

    // Compute previous world position (using same instance matrix - static objects)
    // For moving objects, we'd need prevInstanceModelMatrix in the instance buffer
    // For billboards, use same worldPos (billboard faces camera each frame)
    var prevWorldPos: vec3f;
    if (useBillboardWorldPos) {
        prevWorldPos = billboardWorldPos;  // Billboard - same as current (TODO: proper motion vectors)
    } else {
        prevWorldPos = (instanceModelMatrix * vec4f(prevLocalPos, 1.0)).xyz;
    }
    let prevClipPos = uniforms.prevViewProjMatrix * vec4f(prevWorldPos, 1.0);

    // Apply TAA jitter with distance-based fade
    // Full jitter near camera, fading to 0 at jitterFadeDistance
    let viewDist = -viewPos.z;  // Positive distance from camera
    let jitterFade = saturate(1.0 - viewDist / uniforms.jitterFadeDistance);

    // Convert pixel offset to clip space and apply with fade
    // In clip space, 1 pixel = 2/screenSize (since clip space is -1 to 1)
    let jitterClip = uniforms.jitterOffset * 2.0 / uniforms.screenSize * jitterFade;
    clipPos.x += jitterClip.x * clipPos.w;
    clipPos.y += jitterClip.y * clipPos.w;

    // Pixel rounding: snap vertices to pixel grid for retro/PSX aesthetic
    // pixelRounding value = grid size in pixels (1.0 = every pixel, 2.0 = every 2 pixels, etc.)
    if (allowRounding && uniforms.pixelRounding > 0.5) {
        let gridSize = uniforms.pixelRounding;  // Grid size in pixels
        let ndc = clipPos.xy / clipPos.w;

        // Convert NDC (-1..1) to pixel coordinates (0..screenSize)
        let pixelCoords = (ndc + 1.0) * 0.5 * uniforms.screenSize;

        // Snap to coarse grid (every gridSize pixels)
        var snappedPixel = floor(pixelCoords / gridSize) * gridSize;

        // Expansion to convert gaps to overlaps (push vertices slightly outward from screen center)
        if (uniforms.pixelExpansion > 0.0) {
            let screenCenter = uniforms.screenSize * 0.5;
            let fromCenter = snappedPixel - screenCenter;
            snappedPixel += sign(fromCenter) * uniforms.pixelExpansion;
        }

        // Convert back to NDC
        let snappedNDC = (snappedPixel / uniforms.screenSize) * 2.0 - 1.0;

        // Convert back to clip space
        clipPos.x = snappedNDC.x * clipPos.w;
        clipPos.y = snappedNDC.y * clipPos.w;
    }

    output.position = clipPos;
    output.worldPos = worldPos;
    // Apply UV transform for sprite sheets: uv * scale + offset
    output.uv = input.uv * uvTransform.zw + uvTransform.xy;
    output.viewZ = viewDist;
    output.currClipPos = clipPos;
    output.prevClipPos = prevClipPos;
    output.instanceColor = instanceColor;

    // For billboard sprites, use entity position Y for clip plane testing
    // In planar reflections, the billboard camera vectors are flipped, which causes
    // worldPos to be computed at wrong Y values. Using entityPos directly is simpler
    // and more reliable - if the entity is above ground, the sprite shows.
    if (uniforms.billboardMode > 0.5) {
        let entityPos = instanceModelMatrix[3].xyz;
        output.anchorY = entityPos.y;
    } else {
        output.anchorY = worldPos.y;  // For regular meshes, use fragment world Y
    }

    // Apply instance transform to normal
    // For billboard sprites, the normal is already in world space (0, 1, 0)
    if (uniforms.billboardMode > 0.5) {
        output.normal = localNormal;  // Already world-space (0, 1, 0) for all billboard modes
    } else {
        let normalMatrix = mat3x3f(
            instanceModelMatrix[0].xyz,
            instanceModelMatrix[1].xyz,
            instanceModelMatrix[2].xyz
        );
        output.normal = normalMatrix * localNormal;
    }
    return output;
}

struct GBufferOutput {
    @location(0) albedo: vec4f,
    @location(1) normal: vec4f,
    @location(2) arm: vec4f,    // Ambient, Roughness, Metallic
    @location(3) emission: vec4f,
    @location(4) velocity: vec2f,  // Motion vectors in pixels
    @builtin(frag_depth) depth: f32,  // Linear depth
}

// Interleaved Gradient Noise for alpha hashing
// From: http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
fn interleavedGradientNoise(screenPos: vec2f) -> f32 {
    let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screenPos, magic.xy)));
}

// Alternative: simple hash for variety
fn screenHash(screenPos: vec2f) -> f32 {
    let p = fract(screenPos * vec2f(0.1031, 0.1030));
    let p3 = p.xyx * (p.yxy + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fragmentMain(input: VertexOutput, @builtin(front_facing) frontFacing: bool) -> GBufferOutput {
    // Clip plane: discard fragments based on clip plane Y level and direction
    // Used for planar reflections to avoid rendering geometry on the wrong side of the water
    // Direction > 0: discard below clipPlaneY (camera above water, show above-water objects)
    // Direction < 0: discard above clipPlaneY (camera below water, show below-water objects)
    // For billboard sprites (anchorY != worldPos.y), skip clip plane - entity placement handles it
    if (uniforms.clipPlaneEnabled > 0.5 && uniforms.billboardMode < 0.5) {
        let clipDist = (input.worldPos.y - uniforms.clipPlaneY) * uniforms.clipPlaneDirection;
        if (clipDist < 0.0) {
            discard;
        }
    }

    // Distance fade: noise-based dithered fade to prevent popping at culling distance
    // Fade from fully visible at distanceFadeStart to invisible at distanceFadeEnd
    if (uniforms.distanceFadeEnd > 0.0) {
        let distToCamera = length(input.worldPos - uniforms.cameraPosition);
        if (distToCamera >= uniforms.distanceFadeEnd) {
            discard;  // Beyond fade end - fully invisible
        }
        if (distToCamera > uniforms.distanceFadeStart) {
            // Calculate fade factor: 1.0 at fadeStart, 0.0 at fadeEnd
            let fadeRange = uniforms.distanceFadeEnd - uniforms.distanceFadeStart;
            let fadeFactor = 1.0 - (distToCamera - uniforms.distanceFadeStart) / fadeRange;
            // Use noise-based dithering for smooth fade
            let noise = sampleNoise(input.position.xy);
            if (fadeFactor < noise) {
                discard;  // Dithered fade out
            }
        }
    }

    var output: GBufferOutput;

    // Write albedo with mip offset
    let mipBias = uniforms.mipBias; // Offset mip level
    output.albedo = textureSampleBias(albedoTexture, albedoSampler, input.uv, mipBias);

    // Apply instance color tint (for sprites)
    output.albedo = output.albedo * input.instanceColor;

    // Luminance to alpha: derive alpha from base color brightness (for old game assets where black=transparent)
    // Only pure black (luminance < 1/255) becomes transparent - HARD discard, no noise
    if (uniforms.luminanceToAlpha > 0.5) {
        let luminance = dot(output.albedo.rgb, vec3f(0.299, 0.587, 0.114));
        if (luminance < 0.004) {
            discard;  // Hard discard for pure black - no noise dithering
        }
        output.albedo.a = 1.0;  // Everything else is fully opaque
    }

    // Alpha hashing: screen-space dithered alpha test (only for non-luminanceToAlpha materials)
    // Uses noise texture (blue noise or bayer) for stable, temporally coherent cutout
    if (uniforms.alphaHashEnabled > 0.5 && uniforms.luminanceToAlpha < 0.5) {
        let alpha = output.albedo.a * uniforms.alphaHashScale;
        let noise = sampleNoise(input.position.xy);

        // Hard cutoff for very transparent areas (avoid random dots)
        if (alpha < 0.5) {
            discard;
        }
        // Hash only the semi-transparent edge (0.5 to 1.0 range)
        // Remap alpha from [0.5, 1.0] to [0.0, 1.0] for noise comparison
        let remappedAlpha = (alpha - 0.5) * 2.0;
        if (remappedAlpha < noise) {
            discard;
        }
    }


    // Write world-space normal

    // Sample normal map and convert from [0,1] to [-1,1] range
    let nsample = textureSampleBias(normalTexture, normalSampler, input.uv, mipBias).rgb;
    let tangentNormal = normalize(nsample * 2.0 - 1.0);

    // Calculate cotangent frame from screen-space derivatives (runtime tangent generation)
    // Based on: http://www.thetenthplanet.de/archives/1180
    let dPdx = dpdx(input.worldPos);
    let dPdy = dpdy(input.worldPos);
    let dUVdx = dpdx(input.uv);
    let dUVdy = dpdy(input.uv);

    let N = normalize(input.normal);

    // Get edge vectors perpendicular to N
    let dp2perp = cross(dPdy, N);
    let dp1perp = cross(N, dPdx);

    // Construct tangent and bitangent
    // Negate T to match glTF/OpenGL tangent space convention
    var T = -(dp2perp * dUVdx.x + dp1perp * dUVdy.x);
    var B = dp2perp * dUVdx.y + dp1perp * dUVdy.y;

    // Scale-invariant normalization
    let invmax = inverseSqrt(max(dot(T, T), dot(B, B)));
    T = T * invmax;
    B = B * invmax;

    // Handle degenerate cases (no valid UVs)
    if (length(T) < 0.001 || length(B) < 0.001) {
        let refVec = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
        T = normalize(cross(N, refVec));
        B = cross(N, T);
    }

    // For double-sided materials: flip tangent and bitangent for back faces
    if (!frontFacing) {
        T = -T;
        B = -B;
    }

    let TBN = mat3x3f(T, B, N);

    // Transform tangent space normal to world space
    let normal = normalize(TBN * tangentNormal);
    // Store world Y position in .w for planar reflection distance fade
    output.normal = vec4f(normal, input.worldPos.y);

    // Write ARM values (Ambient, Roughness, Metallic, SpecularBoost)
    var ambient = textureSampleBias(ambientTexture, ambientSampler, input.uv, mipBias).rgb;
    if (ambient.r < 0.04) {
        ambient.r = 1.0;
    }
    let rm = textureSampleBias(rmTexture, rmSampler, input.uv, mipBias).rgb;
    output.arm = vec4f(ambient.r, rm.g, rm.b, uniforms.specularBoost);

    // Write emission
    output.emission = textureSampleBias(emissionTexture, emissionSampler, input.uv, mipBias) * uniforms.emissionFactor;

    // Compute motion vectors (velocity) in pixels
    // Convert clip positions to NDC
    let currNDC = input.currClipPos.xy / input.currClipPos.w;
    let prevNDC = input.prevClipPos.xy / input.prevClipPos.w;

    // Convert NDC difference to pixel velocity
    // NDC is -1 to 1, screen is 0 to screenSize
    // velocity = (currNDC - prevNDC) * screenSize / 2
    let velocityNDC = currNDC - prevNDC;
    output.velocity = velocityNDC * uniforms.screenSize * 0.5;

    // Linear depth: maps [near, far] to [0, 1]
    let near = uniforms.near;
    let far = uniforms.far;
    let z = input.viewZ;
    output.depth = (z - near) / (far - near);

    return output;
}
