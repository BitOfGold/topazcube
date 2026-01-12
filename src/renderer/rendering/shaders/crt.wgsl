// CRT Effect Shader
// Simulates a CRT monitor with geometry distortion, scanlines,
// RGB convergence errors, phosphor mask, and vignette.

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

struct Uniforms {
    canvasSize: vec2f,          // Output canvas size
    inputSize: vec2f,           // Upscaled input texture size
    renderSize: vec2f,          // Original render size (before upscale)

    // Geometry
    curvature: f32,             // Screen curvature (0-0.15)
    cornerRadius: f32,          // Corner rounding (0-0.1)
    zoom: f32,                  // Zoom to compensate for curvature shrinkage
    _padGeom: f32,

    // Scanlines
    scanlineIntensity: f32,     // Scanline darkness (0-1)
    scanlineWidth: f32,         // Beam width (0=thin line, 1=no gap)
    scanlineBrightBoost: f32,   // Bright pixels widen beam to fill gaps
    scanlineHeight: f32,        // Scanline height in canvas pixels (e.g. 3)

    // Convergence (RGB X offset in source pixels)
    convergence: vec3f,
    _pad2: f32,

    // Phosphor mask
    maskType: f32,              // 0=none, 1=aperture, 2=slot, 3=shadow
    maskIntensity: f32,         // Mask strength (0-1)
    maskScale: f32,             // Mask size multiplier
    maskCompensation: f32,      // Pre-calculated brightness compensation

    // Vignette
    vignetteIntensity: f32,
    vignetteSize: f32,

    // Blur
    blurSize: f32,              // Horizontal blur size in pixels (0-2)

    // Flags
    crtEnabled: f32,            // 1.0 = CRT effects on, 0.0 = passthrough
    upscaleEnabled: f32,        // 1.0 = use upscaled texture, 0.0 = direct
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var inputSampler: sampler;        // Linear sampler for CRT
@group(0) @binding(3) var nearestSampler: sampler;      // Nearest sampler for sharp pixels
@group(0) @binding(4) var phosphorMask: texture_2d<f32>;
@group(0) @binding(5) var phosphorSampler: sampler;

// Constants
const PI: f32 = 3.14159265359;

// ============================================================================
// Geometry Distortion (Barrel/Curvature)
// ============================================================================

// Apply barrel distortion to simulate CRT screen curvature
fn applyBarrelDistortion(uv: vec2f, curvature: f32) -> vec2f {
    // Center UV around origin
    let centered = uv - 0.5;

    // Calculate radial distance squared from center
    let r2 = dot(centered, centered);

    // Apply barrel distortion
    let distorted = centered * (1.0 + curvature * r2);

    return distorted + 0.5;
}

// Check if UV is outside the visible area (for corner masking)
fn getCornerMask(uv: vec2f, radius: f32) -> f32 {
    // Distance from edges
    let d = abs(uv - 0.5) * 2.0;

    // Rounded rectangle SDF
    let corner = max(d.x, d.y);
    let roundedCorner = length(max(d - (1.0 - radius), vec2f(0.0)));

    // Soft edge
    let mask = 1.0 - smoothstep(1.0 - radius * 0.5, 1.0, max(corner, roundedCorner));

    return mask;
}

// ============================================================================
// Scanlines (Electron Beam Simulation - Gaussian Profile)
// ============================================================================

fn applyScanlines(color: vec3f, scanlinePos: f32, intensity: f32, width: f32, brightBoost: f32) -> vec3f {
    if (intensity <= 0.0) {
        return color;
    }

    // scanlinePos is position within one scanline period (0-1)
    // 0.5 = center of the beam (brightest), 0.0/1.0 = between scanlines (darkest)
    let pos = fract(scanlinePos);

    // Distance from scanline center (0 at center, 0.5 at edges)
    // For 3px scanlines: center px at dist=0, edge px at dist≈0.33
    let distFromCenter = abs(pos - 0.5);

    // Calculate beam width based on base width and brightness
    let brightness = dot(color, vec3f(0.299, 0.587, 0.114));

    // Width 0 = thin beam (center bright, edges black)
    // Width 1 = full beam (all pixels equally bright, no visible scanline)
    // BrightBoost: bright pixels widen beam toward filling gaps
    let baseWidth = width;
    let brightWidening = brightBoost * brightness * (1.0 - width);
    let effectiveWidth = clamp(baseWidth + brightWidening, 0.0, 1.0);

    // Gaussian sigma: controls how quickly brightness falls off from center
    // At width=0: sigma small → sharp falloff, edges dark
    // At width=1: sigma large → flat response, edges bright
    let sigma = mix(0.08, 0.8, effectiveWidth);
    let gaussian = exp(-0.5 * pow(distFromCenter / sigma, 2.0));

    // Apply intensity: 0 = no effect, 1 = full scanline effect
    let scanline = clamp(gaussian, 0.0, 1.0);
    let darkening = mix(1.0, scanline, intensity);

    return color * darkening;
}

// ============================================================================
// RGB Convergence Error
// ============================================================================

fn sampleWithConvergence(uv: vec2f, convergence: vec3f, texelSize: vec2f) -> vec3f {
    // Sample each color channel with its own offset
    // Use textureSampleLevel to avoid non-uniform control flow issues
    let offsetR = vec2f(convergence.r * texelSize.x, 0.0);
    let offsetG = vec2f(convergence.g * texelSize.x, 0.0);
    let offsetB = vec2f(convergence.b * texelSize.x, 0.0);

    let r = textureSampleLevel(inputTexture, inputSampler, uv + offsetR, 0.0).r;
    let g = textureSampleLevel(inputTexture, inputSampler, uv + offsetG, 0.0).g;
    let b = textureSampleLevel(inputTexture, inputSampler, uv + offsetB, 0.0).b;

    return vec3f(r, g, b);
}

// ============================================================================
// Phosphor Mask
// ============================================================================

fn applyPhosphorMask(color: vec3f, screenPos: vec2f, maskType: f32, intensity: f32, scale: f32) -> vec3f {
    if (intensity <= 0.0 || maskType < 0.5) {
        return color;
    }

    // Sample prerendered phosphor mask texture
    // Scale determines how many mask tiles fit on screen
    // Use textureSampleLevel to avoid non-uniform control flow issues
    let maskUV = screenPos / (3.0 * scale);
    let mask = textureSampleLevel(phosphorMask, phosphorSampler, maskUV, 0.0).rgb;

    // Blend mask with color
    // The mask modulates each RGB channel independently
    return mix(color, color * mask, intensity);
}

// Procedural aperture grille (fallback if no texture)
fn proceduralApertureGrille(screenX: f32, scale: f32) -> vec3f {
    let phase = fract(screenX / (3.0 * scale));

    // RGB stripes
    var mask = vec3f(0.0);
    if (phase < 0.333) {
        mask = vec3f(1.0, 0.2, 0.2);  // Red phosphor
    } else if (phase < 0.666) {
        mask = vec3f(0.2, 1.0, 0.2);  // Green phosphor
    } else {
        mask = vec3f(0.2, 0.2, 1.0);  // Blue phosphor
    }

    return mask;
}

// Procedural slot mask - height matches scanline height for alignment
fn proceduralSlotMask(screenPos: vec2f, scale: f32, scanlineHeight: f32) -> vec3f {
    // X uses 3-pixel RGB cells, Y uses scanline height for alignment
    let cellWidth = 3.0 * scale;
    let cellHeight = max(scanlineHeight, 1.0);
    let cellPos = fract(vec2f(screenPos.x / cellWidth, screenPos.y / cellHeight));

    // Horizontal offset every other row for staggered pattern
    var adjusted = cellPos;
    if (fract(screenPos.y / (cellHeight * 2.0)) > 0.5) {
        adjusted.x = fract(adjusted.x + 0.5);
    }

    // RGB slots
    var mask = vec3f(0.0);
    if (adjusted.x < 0.333) {
        mask = vec3f(1.0, 0.1, 0.1);
    } else if (adjusted.x < 0.666) {
        mask = vec3f(0.1, 1.0, 0.1);
    } else {
        mask = vec3f(0.1, 0.1, 1.0);
    }

    // Vertical gaps between scanlines
    mask *= smoothstep(0.0, 0.15, cellPos.y) * smoothstep(1.0, 0.85, cellPos.y);

    return mask;
}

// Procedural shadow mask (delta pattern)
fn proceduralShadowMask(screenPos: vec2f, scale: f32) -> vec3f {
    let cellSize = 2.0 * scale;
    let cell = floor(screenPos / cellSize);
    let cellPos = fract(screenPos / cellSize);

    // Offset pattern based on row
    let rowOffset = select(0.0, 0.5, fract(cell.y * 0.5) > 0.25);
    let adjustedX = fract(cellPos.x + rowOffset);

    // Circular phosphors
    let dist = length(cellPos - 0.5) * 2.0;
    let circle = 1.0 - smoothstep(0.6, 0.8, dist);

    // RGB color based on position
    let phase = fract((cell.x + rowOffset) / 3.0);
    var mask = vec3f(0.1);
    if (phase < 0.333) {
        mask = vec3f(1.0, 0.1, 0.1) * circle;
    } else if (phase < 0.666) {
        mask = vec3f(0.1, 1.0, 0.1) * circle;
    } else {
        mask = vec3f(0.1, 0.1, 1.0) * circle;
    }

    return max(mask, vec3f(0.1));
}

// ============================================================================
// Vignette
// ============================================================================

fn applyVignette(color: vec3f, uv: vec2f, intensity: f32, size: f32) -> vec3f {
    if (intensity <= 0.0) {
        return color;
    }

    // Distance from center
    let centered = uv - 0.5;
    let dist = length(centered);

    // Smooth vignette falloff
    let vignette = 1.0 - smoothstep(size * 0.5, size, dist);
    let darkening = mix(1.0 - intensity, 1.0, vignette);

    return color * darkening;
}

// ============================================================================
// Horizontal Blur (CRT beam softness)
// ============================================================================

// 5-tap Gaussian horizontal blur
fn sampleWithHorizontalBlur(uv: vec2f, blurSize: f32, texelSize: vec2f) -> vec3f {
    if (blurSize <= 0.0) {
        return textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb;
    }

    // Gaussian weights for 5 taps (sigma ~= 0.84 for nice falloff)
    // Weights: 0.0625, 0.25, 0.375, 0.25, 0.0625 (sum = 1.0)
    let w0 = 0.0625;
    let w1 = 0.25;
    let w2 = 0.375;

    let offset = blurSize * texelSize.x;

    let c0 = textureSampleLevel(inputTexture, inputSampler, uv + vec2f(-2.0 * offset, 0.0), 0.0).rgb;
    let c1 = textureSampleLevel(inputTexture, inputSampler, uv + vec2f(-1.0 * offset, 0.0), 0.0).rgb;
    let c2 = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).rgb;
    let c3 = textureSampleLevel(inputTexture, inputSampler, uv + vec2f(1.0 * offset, 0.0), 0.0).rgb;
    let c4 = textureSampleLevel(inputTexture, inputSampler, uv + vec2f(2.0 * offset, 0.0), 0.0).rgb;

    return c0 * w0 + c1 * w1 + c2 * w2 + c3 * w1 + c4 * w0;
}

// 5-tap Gaussian horizontal blur with RGB convergence
fn sampleWithBlurAndConvergence(uv: vec2f, convergence: vec3f, blurSize: f32, texelSize: vec2f) -> vec3f {
    if (blurSize <= 0.0) {
        // No blur, just convergence
        let offsetR = vec2f(convergence.r * texelSize.x, 0.0);
        let offsetG = vec2f(convergence.g * texelSize.x, 0.0);
        let offsetB = vec2f(convergence.b * texelSize.x, 0.0);
        let r = textureSampleLevel(inputTexture, inputSampler, uv + offsetR, 0.0).r;
        let g = textureSampleLevel(inputTexture, inputSampler, uv + offsetG, 0.0).g;
        let b = textureSampleLevel(inputTexture, inputSampler, uv + offsetB, 0.0).b;
        return vec3f(r, g, b);
    }

    // Gaussian weights
    let w0 = 0.0625;
    let w1 = 0.25;
    let w2 = 0.375;
    let offset = blurSize * texelSize.x;

    // Sample each channel with convergence offset + blur
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;

    for (var i = -2; i <= 2; i++) {
        let weight = select(select(w1, w2, i == 0), w0, abs(i) == 2);
        let blurOffset = f32(i) * offset;

        r += textureSampleLevel(inputTexture, inputSampler, uv + vec2f(convergence.r * texelSize.x + blurOffset, 0.0), 0.0).r * weight;
        g += textureSampleLevel(inputTexture, inputSampler, uv + vec2f(convergence.g * texelSize.x + blurOffset, 0.0), 0.0).g * weight;
        b += textureSampleLevel(inputTexture, inputSampler, uv + vec2f(convergence.b * texelSize.x + blurOffset, 0.0), 0.0).b * weight;
    }

    return vec3f(r, g, b);
}

// ============================================================================
// Main Shader
// ============================================================================

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    // Full-screen triangle
    let x = f32(vertexIndex & 1u) * 4.0 - 1.0;
    let y = f32(vertexIndex >> 1u) * 4.0 - 1.0;
    output.position = vec4f(x, y, 0.0, 1.0);
    output.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    // Use fragment coordinates (input.position) for pixel-perfect effects
    let pixelPos = input.position.xy;
    var uv = input.uv;

    // Check if CRT effects are enabled
    let crtEnabled = uniforms.crtEnabled > 0.5;

    // If CRT disabled and upscale disabled, just sample directly with nearest
    // (avoids interpolation artifacts when input matches canvas size)
    if (!crtEnabled && uniforms.upscaleEnabled < 0.5) {
        return textureSampleLevel(inputTexture, nearestSampler, uv, 0.0);
    }

    // Apply geometry distortion (only when CRT enabled)
    var distortedUV = uv;
    var cornerMask = 1.0;

    if (crtEnabled && uniforms.curvature > 0.0) {
        // Apply zoom first (scales UV inward to enlarge image, compensating for curvature shrinkage)
        var zoomedUV = uv;
        if (uniforms.zoom > 1.0) {
            let centered = uv - 0.5;
            zoomedUV = centered / uniforms.zoom + 0.5;
        }
        distortedUV = applyBarrelDistortion(zoomedUV, uniforms.curvature);
        cornerMask = getCornerMask(distortedUV, uniforms.cornerRadius);

        // Clamp to edges for mirror-repeat effect
        // If outside bounds, return black (corner area)
        if (distortedUV.x < 0.0 || distortedUV.x > 1.0 ||
            distortedUV.y < 0.0 || distortedUV.y > 1.0) {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
    }

    // Calculate texel size for blur and convergence
    let texelSize = 1.0 / uniforms.inputSize;

    // Sample color with optional blur and RGB convergence
    var color: vec3f;
    if (crtEnabled) {
        let hasConvergence = any(uniforms.convergence != vec3f(0.0));
        if (hasConvergence) {
            // Blur + convergence combined
            color = sampleWithBlurAndConvergence(distortedUV, uniforms.convergence, uniforms.blurSize, texelSize);
        } else {
            // Just blur (or no blur if blurSize <= 0)
            color = sampleWithHorizontalBlur(distortedUV, uniforms.blurSize, texelSize);
        }
    } else {
        // Upscale-only mode
        // Check if actual upscaling occurred (input larger than canvas)
        let hasUpscaling = uniforms.inputSize.x > uniforms.canvasSize.x + 0.5 ||
                           uniforms.inputSize.y > uniforms.canvasSize.y + 0.5;

        if (hasUpscaling) {
            // Actual upscaling: use linear for smooth downsampling to canvas
            color = textureSampleLevel(inputTexture, inputSampler, distortedUV, 0.0).rgb;
        } else {
            // No upscaling (1:1 mapping): use nearest to avoid interpolation artifacts
            // This prevents checkerboard patterns from floating-point UV precision issues
            color = textureSampleLevel(inputTexture, nearestSampler, distortedUV, 0.0).rgb;
        }
    }

    // Apply CRT effects only when enabled
    if (crtEnabled) {
        // Pixel-perfect scanlines using fragment coordinates
        // scanlineHeight = exact pixels per scanline (e.g. 3 = repeats every 3 pixels)
        let scanlinePos = pixelPos.y / max(uniforms.scanlineHeight, 1.0);

        // Apply Gaussian scanlines
        color = applyScanlines(
            color,
            scanlinePos,
            uniforms.scanlineIntensity,
            uniforms.scanlineWidth,
            uniforms.scanlineBrightBoost
        );

        // Apply pixel-perfect phosphor mask using fragment coordinates
        if (uniforms.maskType > 0.5 && uniforms.maskIntensity > 0.0) {
            var mask: vec3f;
            let maskPos = pixelPos / uniforms.maskScale;
            if (uniforms.maskType < 1.5) {
                // Aperture grille (type 1)
                mask = proceduralApertureGrille(maskPos.x, 1.0);
            } else if (uniforms.maskType < 2.5) {
                // Slot mask (type 2) - height matches scanline height
                mask = proceduralSlotMask(maskPos, 1.0, uniforms.scanlineHeight / uniforms.maskScale);
            } else {
                // Shadow mask (type 3)
                mask = proceduralShadowMask(maskPos, 1.0);
            }

            // Apply mask with pre-calculated brightness compensation
            let maskedColor = color * mask;
            color = mix(color, maskedColor, uniforms.maskIntensity);
            color *= uniforms.maskCompensation;
        }

        // Apply vignette
        color = applyVignette(color, uv, uniforms.vignetteIntensity, uniforms.vignetteSize);

        // Apply corner mask (darken rounded corners)
        color *= cornerMask;
    }

    return vec4f(color, 1.0);
}
