import "./math.js"
import { Texture } from "./Texture.js"
import { Geometry } from "./Geometry.js";
import { Material } from "./Material.js";
import { Mesh } from "./Mesh.js";
import { Camera } from "./Camera.js"
import { RenderGraph } from "./rendering/RenderGraph.js"
import { loadGltf } from "./gltf.js"
import { EntityManager } from "./core/EntityManager.js"
import { AssetManager } from "./core/AssetManager.js"
import { CullingSystem } from "./core/CullingSystem.js"
import { InstanceManager } from "./core/InstanceManager.js"
import { ParticleSystem } from "./core/ParticleSystem.js"
import { ParticleEmitter } from "./core/ParticleEmitter.js"
import { DebugUI } from "./DebugUI.js"
import { Raycaster } from "./utils/Raycaster.js"


// Display a failure message and stop rendering
function fail(engine, msg, data) {
    if (engine?.canvas) {
        engine.canvas.style.display = "none"
    }
    console.error(msg, data)

    if (typeof document !== 'undefined') {
        let ecanvas = document.createElement("canvas")
        document.body.appendChild(ecanvas)
        if (ecanvas) {
            ecanvas.width = window.innerWidth
            ecanvas.height = window.innerHeight
            let ctx = ecanvas.getContext("2d")
            ctx.clearRect(0, 0, ecanvas.width, ecanvas.height)
            ctx.fillStyle = "rgba(255, 128, 128, 0.5)"
            ctx.fillRect(0, 0, ecanvas.width, ecanvas.height)
            ctx.fillStyle = "#ffffff"
            ctx.textAlign = "center"
            ctx.textBaseline = "middle"
            ctx.font = "64px Arial"
            ctx.fillText("😒", ecanvas.width / 2, ecanvas.height / 2 - 72)
            ctx.font = "bold 20px Arial"
            ctx.fillText(msg, ecanvas.width / 2, ecanvas.height / 2)
            if (data) {
                ctx.font = "10px Arial"
                ctx.fillText(data, ecanvas.width / 2, ecanvas.height / 2 + 20)
            }
        } else {
            alert(msg, data)
        }
    }
    if (engine) {
        engine.rendering = false
    }
}


// Default settings for the entire engine - consolidated from all subsystems
const DEFAULT_SETTINGS = {
    // Engine/Runtime settings
    engine: {
        debugMode: false,           // F10 toggle: false=character controller, true=fly camera
        mouseSmoothing: 0.2,        // Lower = more smoothing
        mouseIdleThreshold: 0.1,    // Seconds before stopping mouse callbacks
    },

    // Camera defaults
    camera: {
        fov: 70,                    // Field of view in degrees
        near: 0.05,                 // Near plane
        far: 5000,                  // Far plane
    },

    // Rendering options
    rendering: {
        debug: false,
        nearestFiltering: false,    // Use linear filtering by default
        mipBias: 0,                 // MIP map bias
        fxaa: false,                // Fast approximate anti-aliasing
        renderScale: 1,             // Render resolution multiplier (1.5-2.0 for supersampling AA)
        autoScale: {
            enabled: true,         // Auto-reduce renderScale for high resolutions
            enabledForEffects: true,// Auto scale effects at high resolutions (when main autoScale disabled)
            maxHeight: 1536,        // Height threshold (above this, scale is reduced)
            scaleFactor: 0.5,       // Factor to apply when above threshold
        },
        jitter: false,              // TAA-like sub-pixel jitter
        jitterAmount: 0.37,         // Jitter amplitude in pixels
        jitterFadeDistance: 25.0,   // Distance at which jitter fades to 0
        pixelRounding: 0,           // Pixel grid size for vertex snapping (0=off, 1=every pixel, 2=every 2px, etc.)
        pixelExpansion: 0,       // Sub-pixel expansion to convert gaps to overlaps (0=off, 0.05=default)
        positionRounding: 0,        // Round view-space position to this precision (0 = disabled, simulates fixed-point)
        alphaHash: false,           // Enable alpha hashing/dithering for cutout transparency (global default)
        alphaHashScale: 1.0,        // Scale factor for alpha hash threshold (higher = more opaque)
        luminanceToAlpha: false,    // Derive alpha from color luminance (for old game assets where black=transparent)
        tonemapMode: 0,             // 0=ACES, 1=Reinhard, 2=None (linear clamp)
    },

    // Noise settings for dithering, jittering, etc.
    noise: {
        type: 'bluenoise',             // 'bluenoise', 'bayer8' (8x8 ordered dither)
        animated: false,            // Animate noise offset each frame (temporal variation)
    },

    // Dithering settings (PS1-style color quantization)
    dithering: {
        enabled: false,              // Enable/disable dithering
        colorLevels: 32,            // Color levels per channel (32 = 5-bit like PS1, 256 = 8-bit, 16 = 4-bit)
    },

    // Bloom/Glare settings (HDR glow effect)
    bloom: {
        enabled: true,              // Enable/disable bloom
        intensity: 0.12,            // Overall bloom intensity
        threshold: 0.98,            // Brightness threshold (pixels below this contribute exponentially less)
        softThreshold: 0.5,         // Soft knee for threshold (0 = hard, 1 = very soft)
        radius: 64,                 // Blur radius in pixels (scaled by renderScale)
        emissiveBoost: 6.0,         // Extra boost for emissive pixels
        maxBrightness: 6.0,         // Clamp input brightness (prevents specular halos)
        scale: 0.5,                 // Resolution scale (0.5 = half res for performance, 1.0 = full)
    },

    // Environment/IBL settings
    environment: {
        texture: "alps_field.jpg",  // .jpg = octahedral RGBM pair, .hdr = equirectangular
        //texture: "alps_field_8k.hdr",
        diffuse: 3.0,
        specular: 1.0,
        emissionFactor: [1.0, 1.0, 1.0, 1.0],
        ambientColor: [0.7, 0.75, 0.9, 0.1],
        exposure: 1.6,
        fog: {
            enabled: true,
            color: [100/255.0, 135/255.0, 170/255.0],
            distances: [0, 15, 50],
            alpha: [0.0, 0.5, 0.9],
            heightFade: [-2, 185],  // [bottomY, topY] - full fog at bottomY, zero at topY
            brightResist: 0.0,        // How much bright/emissive colors resist fog (0-1)
            debug: 0,
        }
    },

    // Main directional light
    mainLight: {
        enabled: true,
        intensity: 1,
        color: [1.0, 0.78, 0.47],    // Mid-morning / mid-afternoon (solar elevation ~20°–40°, less red) - soft warm white
        direction: [-0.4, 0.45, 0.35],
    },

    // Shadow settings
    shadow: {
        mapSize: 2048,
        cascadeCount: 3,
        cascadeSizes: [10, 25, 125], // Half-widths in meters
        maxSpotShadows: 16,
        spotTileSize: 512,
        spotAtlasSize: 2048,
        spotMaxDistance: 60,        // No spot shadow beyond this distance
        spotFadeStart: 55,          // Spot shadow fade out starts here
        bias: 0.0005,
        normalBias: 0.015,          // ~2-3 texels for shadow acne
        surfaceBias: 0,             // Scale shadow projection larger (0.01 = 1% larger)
        strength: 1.0,
        //frustum: false,
        //hiZ: false,
    },

    // Ambient Occlusion settings
    ao: {
        enabled: true,              // Enable/disable AO
        intensity: 1.6,             // Overall AO strength
        radius: 64.0,               // Sample radius in pixels
        fadeDistance: 40.0,         // Distance at which AO fades to 0
        bias: 0.005,                // Depth bias to avoid self-occlusion
        sampleCount: 16,            // Number of samples
        level: 1,                   // AO level multiplier
    },

    // Lighting pass settings
    lighting: {
        maxLights: 768,
        tileSize: 16,               // Tile size for tiled deferred
        maxLightsPerTile: 256,
        maxDistance: 250,           // Max distance for point lights from camera
        cullingEnabled: true,
        directSpecularMultiplier: 3.0, // Multiplier for direct light specular highlights
        specularBoost: 64.0,         // Extra specular from 3 fake lights (0 = disabled)
        specularBoostRoughnessCutoff: 0.70, // Only boost materials with roughness < this
    },

    // Culling configuration per pass type
    culling: {
        frustumEnabled: true,           // Enable frustum culling
        shadow: {
            frustum: true,              // Enable frustum culling using shadow bounding spheres
            hiZ: true,                  // Enable HiZ occlusion culling using shadow bounding spheres
            cascadeFilter: true,        // Enable per-cascade instance filtering
            maxDistance: 250,
            maxSkinned: 32,
            minPixelSize: 1,
            fadeStart: 0.8,             // Distance fade starts at 80% of maxDistance
        },
        reflection: {
            frustum: true,
            maxDistance: 50,
            maxSkinned: 0,
            minPixelSize: 4,
            fadeStart: 0.8,
        },
        planarReflection: {
            frustum: true,
            maxDistance: 40,
            maxSkinned: 32,
            minPixelSize: 4,
            fadeStart: 0.7,             // Earlier fade for reflections (70%)
        },
        main: {
            frustum: true,
            maxDistance: 250,
            maxSkinned: 500,
            minPixelSize: 2,
            fadeStart: 0.8,             // Distance fade starts at 90% of maxDistance
        },
    },

    // HiZ Occlusion Culling - uses previous frame's depth buffer
    occlusionCulling: {
        enabled: true,                  // Enable HiZ occlusion culling
        threshold: 0.7,                 // Depth threshold multiplier (0.5 = 50% of maxZ, 1.0 = 100%)
        positionThreshold: 1.0,         // Camera movement (units) before invalidation
        rotationThreshold: 0.1,        // Camera rotation (radians, ~1 deg) before invalidation
        maxTileSpan: 16,                 // Max tiles a bounding sphere can span for occlusion test
    },

    // Skinned mesh rendering
    skinning: {
        individualRenderDistance: 20.0,  // Proximity threshold for individual rendering
    },

    // Screen Space Global Illumination (tile-based light propagation)
    ssgi: {
        enabled: true,
        intensity: 1.0,                 // GI intensity multiplier
        emissiveBoost: 2.0,             // Boost factor for emissive surfaces
        maxBrightness: 4.0,             // Clamp luminance (excludes specular highlights)
        sampleRadius: 3,              // Vogel disk sample radius in tiles
        saturateLevel: 0.5,             // Logarithmic saturation level for indirect light
    },

    // Volumetric Fog (light scattering through particles)
    volumetricFog: {
        enabled: false,                 // Disabled by default (performance impact)
        resolution: 0.125,               // 1/4 render resolution for ray marching
        maxSamples: 32,                 // Ray march samples (8-32)
        blurRadius: 8.0,                // Gaussian blur radius
        densityMultiplier: 1.0,         // Multiplies base fog density
        scatterStrength: 0.35,           // Light scattering intensity
        mainLightScatter: 1.4,          // Main directional light scattering boost
        mainLightScatterDark: 5.0,      // Main directional light scattering boost
        mainLightSaturation: 0.15,      // Main light color saturation in fog
        maxFogOpacity: 0.3,             // Maximum fog opacity (0-1)
        heightRange: [-2, 8],           // [bottom, top] Y bounds for fog (low ground fog)
        windDirection: [1, 0, 0.2],     // Wind direction for fog animation
        windSpeed: 0.5,                 // Wind speed multiplier
        noiseScale: 0.9,                // 3D noise frequency (higher = finer detail)
        noiseStrength: 0.8,             // Noise intensity (0 = uniform, 1 = full variation)
        noiseOctaves: 6,                // Noise detail layers
        noiseEnabled: true,             // Enable 3D noise (disable for debug)
        lightingEnabled: true,          // Light fog from scene lights
        shadowsEnabled: true,           // Apply shadows to fog
        brightnessThreshold: 0.8,       // Scene luminance where fog starts fading (like bloom)
        minVisibility: 0.15,            // Minimum fog visibility over bright surfaces (0-1)
        skyBrightness: 1.2,             // Virtual brightness for sky pixels (depth at far plane)
        //debugSkyCheck: true
    },

    // Planar Reflections (alternative to SSR for water/floor)
    planarReflection: {
        enabled: true,                  // Disabled by default (use SSR instead)
        groundLevel: 0.1,               // Y coordinate of reflection plane (real-time adjustable)
        resolution: 1,                  // Resolution multiplier (0.5 = half res)
        roughnessCutoff: 0.4,           // Only reflect on surfaces with roughness < this
        normalPerturbation: 0.25,       // Amount of normal-based distortion (for water)
        blurSamples: 4,                 // Blur samples based on roughness
        intensity: 1.0,                 // Reflection brightness (0.9 = 90% for realism)
        distanceFade: 0.5,              // Distance from ground for full reflection (meters)
    },

    // Ambient Capture (6-directional sky-aware GI)
    ambientCapture: {
        enabled: true,                  // Enable 6-directional ambient capture
        intensity: 1.0,                 // Output intensity multiplier (subtle effect)
        maxDistance: 50,                // Distance fade & culling in meters
        resolution: 64,                 // Capture resolution per face (default 64)
        emissiveBoost: 10.0,            // Boost for emissive surfaces in capture
        smoothingTime: 0.3,             // Temporal smoothing duration in seconds
        saturateLevel: 0.2,             // Logarithmic saturation level (0 = disabled)
    },

    // Temporal accumulation settings
    temporal: {
        blendFactor: 0.5,               // Default history blend (conservative)
        motionScale: 10.0,              // Motion rejection sensitivity
        depthThreshold: 0.1,            // Depth rejection threshold
        normalThreshold: 0.9,           // Normal rejection threshold (dot product)
    },

    // Performance auto-tuning
    performance: {
        autoDisable: true,              // Auto-disable SSR/SSGI on low FPS
        fpsThreshold: 60,               // FPS threshold for auto-disable
        disableDelay: 3.0,              // Seconds below threshold before disabling
    },

    // CRT effect (retro monitor simulation)
    crt: {
        enabled: false,                 // Enable CRT effect (geometry, scanlines, etc.)
        upscaleEnabled: false,          // Enable upscaling (pixelated look) even when CRT disabled
        upscaleTarget: 4,               // Target upscale multiplier (4x render resolution)
        maxTextureSize: 4096,           // Max upscaled texture dimension

        // Geometry distortion
        curvature: 0.14,                // Screen curvature amount (0-0.15)
        cornerRadius: 0.055,             // Rounded corner radius (0-0.1)
        zoom: 1.06,                      // Zoom to compensate for curvature shrinkage

        // Scanlines (electron beam simulation - Gaussian profile)
        scanlineIntensity: 0.4,         // Scanline effect strength (0-1)
        scanlineWidth: 0.0,            // Beam width (0=thin/center only, 1=no gap)
        scanlineBrightBoost: 0.8,       // Bright pixels widen beam to fill gaps (0-1)
        scanlineHeight: 5,              // Scanline height in canvas pixels

        // RGB convergence error (color channel misalignment)
        convergence: [0.79, 0.0, -0.77],  // RGB X offset in source pixels

        // Phosphor mask
        maskType: 'aperture',           // 'aperture', 'slot', 'shadow', 'none'
        maskIntensity: 0.25,             // Mask strength (0-1)
        maskScale: 1.0,                 // Mask size multiplier

        // Vignette (edge darkening)
        vignetteIntensity: 0.54,        // Edge darkening strength (0-1)
        vignetteSize: 0.85,              // Vignette size (larger = more visible)

        // Horizontal blur (beam softness)
        blurSize: 0.79,                  // Horizontal blur in pixels (0-2)
    },
}

// Function to create WebGPU context
async function createWebGPUContext(engine, canvasId) {
    try {
        const canvas = document.getElementById(canvasId)
        if (!canvas) throw new Error(`Canvas with id ${canvasId} not found`)
        engine.canvas = canvas

        // Detailed WebGPU availability check
        console.log("WebGPU check:", {
            hasNavigatorGpu: !!navigator.gpu,
            isSecureContext: window.isSecureContext,
            protocol: window.location.protocol,
            hostname: window.location.hostname
        })

        if (!navigator.gpu) {
            if (!window.isSecureContext) {
                throw new Error("WebGPU requires HTTPS or localhost (secure context)")
            }
            throw new Error("WebGPU not supported on this browser.")
        }

        // Try high-performance adapter first, fall back to any adapter
        let adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
        })
        if (!adapter) {
            console.warn("High-performance adapter not found, trying default...")
            adapter = await navigator.gpu.requestAdapter()
        }
        if (!adapter) throw new Error("No appropriate GPUAdapter found.")

        // Log adapter info for debugging
        const adapterInfo = await adapter.requestAdapterInfo?.() || {}
        console.log("WebGPU Adapter:", adapterInfo.vendor, adapterInfo.device, adapterInfo.description)
        console.log("Adapter limits:", {
            maxColorAttachmentBytesPerSample: adapter.limits.maxColorAttachmentBytesPerSample,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize
        })

        const canTimestamp = adapter.features.has('timestamp-query');
        const requiredFeatures = []
        if (canTimestamp) {
            requiredFeatures.push('timestamp-query')
            engine.canTimestamp = true
        } else {
            engine.canTimestamp = false
        }

        // Request higher limits for GBuffer with multiple render targets
        // Default is 32 bytes, but we need 36+ for albedo + normal + ARM + emission + velocity
        const requiredLimits = {}
        const adapterLimits = adapter.limits
        if (adapterLimits.maxColorAttachmentBytesPerSample >= 64) {
            requiredLimits.maxColorAttachmentBytesPerSample = 64
        } else if (adapterLimits.maxColorAttachmentBytesPerSample >= 48) {
            requiredLimits.maxColorAttachmentBytesPerSample = 48
        }

        // Try to create device with requested features/limits, fall back if it fails
        let device
        try {
            device = await adapter.requestDevice({
                requiredFeatures: requiredFeatures,
                requiredLimits: requiredLimits
            })
        } catch (deviceError) {
            console.warn("Device creation failed with custom limits, trying defaults...", deviceError)
            // Try without custom limits
            try {
                device = await adapter.requestDevice({
                    requiredFeatures: requiredFeatures
                })
            } catch (deviceError2) {
                console.warn("Device creation failed with features, trying minimal...", deviceError2)
                // Try with no features at all
                device = await adapter.requestDevice()
                engine.canTimestamp = false
            }
        }

        if (!device) throw new Error("Failed to create GPU device")

        const context = canvas.getContext("webgpu")
        if (!context) throw new Error("Failed to get WebGPU context from canvas")

        const canvasFormat = navigator.gpu.getPreferredCanvasFormat()

        engine.adapter = adapter
        engine.device = device
        engine.context = context
        engine.canvasFormat = canvasFormat
        engine.rendering = true

        function configureContext() {
            // Use exact device pixel size if available (from ResizeObserver)
            // This ensures pixel-perfect rendering for CRT effects
            let pixelWidth, pixelHeight
            if (engine._devicePixelSize) {
                pixelWidth = engine._devicePixelSize.width
                pixelHeight = engine._devicePixelSize.height
            } else {
                // Fallback to clientWidth * devicePixelRatio
                const devicePixelRatio = window.devicePixelRatio || 1
                pixelWidth = Math.round(canvas.clientWidth * devicePixelRatio)
                pixelHeight = Math.round(canvas.clientHeight * devicePixelRatio)
            }

            // Canvas is ALWAYS at full device pixel resolution for pixel-perfect CRT
            // Render scale only affects internal render passes, not the final canvas
            canvas.width = pixelWidth
            canvas.height = pixelHeight

            // Store device pixel size for CRT pass
            engine._canvasPixelSize = { width: pixelWidth, height: pixelHeight }

            context.configure({
                device: device,
                format: canvasFormat,
                alphaMode: "opaque",
            })
        }

        configureContext()
        engine.configureContext = configureContext

        // Make available globally for debugging
        if (typeof window !== 'undefined') {
            window.engine = engine
        }
    } catch (error) {
        console.error("WebGPU initialization failed:", error)
        // Provide more specific error message
        let errorTitle = "WebGPU Error"
        let errorDetail = error.message
        if (error.message.includes("not supported")) {
            errorTitle = "WebGPU Not Available"
            errorDetail = "Check if WebGPU is enabled in browser flags"
        } else if (error.message.includes("Adapter")) {
            errorTitle = "GPU Not Found"
            errorDetail = "No compatible GPU adapter found"
        } else if (error.message.includes("device")) {
            errorTitle = "Device Creation Failed"
            errorDetail = error.message + " - Try updating GPU drivers"
        }
        fail(engine, errorTitle, errorDetail)
    }
    return engine
}

/**
 * Deep merge source into target (mutates target)
 * Arrays are replaced, not merged
 */
function deepMerge(target, source) {
    if (!source) return target
    for (const key in source) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
            if (!target[key] || typeof target[key] !== 'object') {
                target[key] = {}
            }
            deepMerge(target[key], source[key])
        } else {
            target[key] = source[key]
        }
    }
    return target
}

class Engine {

    constructor(settings = {}) {
        this.lastTime = performance.now()
        // Deep clone DEFAULT_SETTINGS and merge with provided settings
        this.settings = deepMerge(
            JSON.parse(JSON.stringify(DEFAULT_SETTINGS)),
            settings
        )

        // GPU state properties (populated by createWebGPUContext)
        this.device = null
        this.context = null
        this.canvas = null
        this.canvasFormat = null
        this.canTimestamp = false
        this.configureContext = null

        // Runtime state
        this.rendering = true
        this._renderInProgress = false  // Prevents frame pileup when GPU is slow
        this.renderTextures = []
        this.renderScale = 1
        this.options = {
            debug: false,
            nearestFiltering: false,
            mipBias: 0,
        }
        this.stats = {
            fps: 0,
            ms: 0,
            drawCalls: 0,
            triangles: 0,
        }

        // Debug UI (lazy initialization - created on first debug mode)
        this.debugUI = new DebugUI(this)

        this.init()
    }

    // Convenience getter/setter for debugMode (used frequently)
    get debugMode() { return this.settings.engine.debugMode }
    set debugMode(value) { this.settings.engine.debugMode = value }

    async init() {
        try {
            await createWebGPUContext(this, "webgpu-canvas")
            if (!this.rendering) return

            // Legacy mesh storage (for backward compatibility)
            this.meshes = {}

            // New data-oriented systems
            this.entityManager = new EntityManager()
            this.assetManager = new AssetManager(this)

            // Expose for convenience
            this.entities = this.entityManager.entities
            this.assets = this.assetManager.assets

            await this._create()
            await this.create()
            await this._after_create()

            this._lastTime = performance.now()
            this.time = 0.0
            this.frame = 0
            this.stats.avg_dt = 17
            this.stats.avg_fps = 60
            this.stats.avg_dt_render = 0.1

            requestAnimationFrame(() => this._frame())

            // Use ResizeObserver with devicePixelContentBoxSize for pixel-perfect sizing
            this._devicePixelSize = null
            try {
                const resizeObserver = new ResizeObserver((entries) => {
                    for (const entry of entries) {
                        // Prefer devicePixelContentBoxSize for exact device pixels
                        if (entry.devicePixelContentBoxSize) {
                            const size = entry.devicePixelContentBoxSize[0]
                            this._devicePixelSize = {
                                width: size.inlineSize,
                                height: size.blockSize
                            }
                        } else if (entry.contentBoxSize) {
                            // Fallback to contentBoxSize * devicePixelRatio
                            const size = entry.contentBoxSize[0]
                            const dpr = window.devicePixelRatio || 1
                            this._devicePixelSize = {
                                width: Math.round(size.inlineSize * dpr),
                                height: Math.round(size.blockSize * dpr)
                            }
                        }
                        this.needsResize = true
                    }
                })
                resizeObserver.observe(this.canvas, { box: 'device-pixel-content-box' })
            } catch (e) {
                // Fallback if device-pixel-content-box not supported
                console.log('ResizeObserver device-pixel-content-box not supported, falling back to window resize')
                window.addEventListener("resize", () => {
                    this.needsResize = true
                })
            }

            setInterval(() => {
              if (this.needsResize && !this._resizing) {
                  this.needsResize = false
                  this._resize()
              }
            }, 100)
            this._resize()
        } catch (error) {
            fail(this, "Error", error.message)
            console.error(error)
        }
    }

    _frame() {
        // Skip if previous frame is still rendering (prevents GPU command queue backup)
        if (this._renderInProgress) {
            requestAnimationFrame(() => this._frame())
            return
        }

        let t1 = performance.now()
        let dt = t1 - this._lastTime
        this._lastTime = t1
        if (this.rendering && dt > 0 && dt < 100 && !this._resizing) {
            this.stats.dt = dt
            this.stats.fps = 1000 / dt
            this.stats.avg_dt = this.stats.avg_dt * 0.98 + this.stats.dt * 0.02
            this.stats.avg_fps = this.stats.avg_fps * 0.98 + this.stats.fps * 0.02
            let dtt = dt / 1000.0
            this.time += dtt
            this._update(dtt)
            this.update(dtt)
            let t2 = performance.now()

            // Mark render in progress to prevent frame pileup
            this._renderInProgress = true
            this._render().finally(() => {
                this._renderInProgress = false
            })

            let t3 = performance.now()
            let dtr = t3 - t2
            this.stats.dt_render = dtr
            this.stats.avg_dt_render = this.stats.avg_dt_render * 0.98 + dtr * 0.02

            // Calculate totals including all passes
            const shadowDC = this.stats.shadowDrawCalls || 0
            const shadowTri = this.stats.shadowTriangles || 0
            const planarDC = this.stats.planarDrawCalls || 0
            const planarTri = this.stats.planarTriangles || 0
            const transparentDC = this.stats.transparentDrawCalls || 0
            const transparentTri = this.stats.transparentTriangles || 0
            const totalDC = this.stats.drawCalls + shadowDC + planarDC + transparentDC
            const totalTri = this.stats.triangles + shadowTri + planarTri + transparentTri
            this.stats.shadowDC = shadowDC
            this.stats.shadowTri = shadowTri
            this.stats.planarDC = planarDC
            this.stats.planarTri = planarTri
            this.stats.transparentDC = transparentDC
            this.stats.transparentTri = transparentTri
            this.stats.totalDC = totalDC
            this.stats.totalTri = totalTri
            this.frame++

            // Update debug UI (checks debug mode internally, lazy init)
            if (this.debugUI) {
                this.debugUI.update()
            }
        }
        requestAnimationFrame(() => this._frame())
    }

    async _render() {
        // Pass delta time to renderer for animation updates
        const dt = this.stats.dt ? this.stats.dt / 1000.0 : 0.016

        await this.renderer.renderEntities({
            entityManager: this.entityManager,
            assetManager: this.assetManager,
            camera: this.camera,
            meshes: this.meshes,
            dt
        })
    }

    async loadGltf(url, options = {}) {
        const result = await loadGltf(this, url, options)
        // Handle both old format (just meshes) and new format (with skins, animations)
        const meshes = result.meshes || result
        for (const [name, mesh] of Object.entries(meshes)) {
            this.meshes[name] = mesh
        }
        // Store skins and animations for access
        if (result.skins) {
            this.skins = this.skins || []
            this.skins.push(...result.skins)
        }
        if (result.animations) {
            this.animations = this.animations || []
            this.animations.push(...result.animations)
        }
        return result
    }

    /**
     * Load a GLTF file and register with asset manager (new data-oriented API)
     * @param {string} url - Path to the GLTF file
     * @param {Object} options - Loading options
     * @returns {Promise<Object>} Asset manager entry
     */
    async loadAsset(url, options = {}) {
        const result = await this.assetManager.loadGltfFile(url, options)

        // Also store in legacy meshes for backward compatibility
        if (result.meshes) {
            for (const [name, mesh] of Object.entries(result.meshes)) {
                this.meshes[name] = mesh
                // Ensure at least one instance exists for rendering
                if (mesh.geometry.instanceCount === 0) {
                    // Use geometry bounding sphere for correct shadow culling
                    const localBsphere = mesh.geometry.getBoundingSphere?.()
                    const center = localBsphere?.center || [0, 0, 0]
                    const radius = localBsphere?.radius || 1
                    mesh.addInstance(center, radius)
                    mesh.updateInstance(0, mat4.create())
                }
            }
        }

        return result
    }

    /**
     * Load a GLTF/GLB file and render meshes directly in the scene at their original positions.
     * Unlike loadAsset, this doesn't hide meshes - they render immediately with their transforms.
     * Handles Blender's Z-up coordinate system by respecting the full node hierarchy.
     *
     * @param {string} url - Path to the GLTF/GLB file
     * @param {Object} options - Loading options
     * @param {Array} options.position - Optional position offset [x, y, z]
     * @param {Array} options.rotation - Optional rotation offset [x, y, z] in radians
     * @param {number} options.scale - Optional uniform scale multiplier
     * @param {boolean} options.doubleSided - Optional: force all materials to be double-sided
     * @returns {Promise<Object>} Object containing { meshes, nodes, skins, animations }
     */
    async loadScene(url, options = {}) {
        const result = await loadGltf(this, url, options)
        const { meshes, nodes } = result

        // Apply scene-wide doubleSided option if specified
        if (options.doubleSided) {
            for (const mesh of Object.values(meshes)) {
                if (mesh.material) {
                    mesh.material.doubleSided = true
                }
            }
        }

        // Update node world matrices from their hierarchy
        // This handles Blender's Z-up to Y-up rotation in parent nodes
        for (const node of nodes) {
            if (!node.parent) {
                // Root nodes - start the hierarchy update
                node.updateMatrix(null)
            }
        }

        // Optional root transform from options
        const rootTransform = mat4.create()
        if (options.position || options.rotation || options.scale) {
            const pos = options.position || [0, 0, 0]
            const rot = options.rotation || [0, 0, 0]
            const scl = options.scale || 1

            const rotQuat = quat.create()
            quat.fromEuler(rotQuat, rot[0] * 180 / Math.PI, rot[1] * 180 / Math.PI, rot[2] * 180 / Math.PI)

            mat4.fromRotationTranslationScale(
                rootTransform,
                rotQuat,
                pos,
                [scl, scl, scl]
            )
        }

        // For skinned models with multiple submeshes, compute a combined bounding sphere
        // This ensures all submeshes are culled together as a unit (especially for shadows)
        let combinedBsphere = null
        const hasAnySkin = Object.values(meshes).some(m => m.hasSkin)

        if (hasAnySkin) {
            // Collect all vertex positions from ALL meshes
            const allPositions = []
            for (const mesh of Object.values(meshes)) {
                const positions = mesh.geometry?.attributes?.position
                if (positions) {
                    for (let i = 0; i < positions.length; i += 3) {
                        allPositions.push(positions[i], positions[i + 1], positions[i + 2])
                    }
                }
            }

            if (allPositions.length > 0) {
                // Calculate combined bounding sphere
                const { calculateBoundingSphere } = await import('./utils/BoundingSphere.js')
                combinedBsphere = calculateBoundingSphere(new Float32Array(allPositions))
            }
        }

        // For each mesh, find its node and compute world transform
        for (const [name, mesh] of Object.entries(meshes)) {
            // Find the node that references this mesh by nodeIndex
            let meshNode = null
            if (mesh.nodeIndex !== null && mesh.nodeIndex !== undefined) {
                meshNode = nodes[mesh.nodeIndex]
            }

            // Compute final world matrix
            const worldMatrix = mat4.create()
            if (meshNode) {
                mat4.copy(worldMatrix, meshNode.world)
            }

            // Apply optional root transform
            if (options.position || options.rotation || options.scale) {
                mat4.multiply(worldMatrix, rootTransform, worldMatrix)
            }

            // Compute world bounding sphere from geometry bsphere + world transform
            // For skinned models, use the combined bsphere so all submeshes are culled together
            const localBsphere = (hasAnySkin && combinedBsphere) ? combinedBsphere : mesh.geometry.getBoundingSphere?.()
            let worldCenter = [0, 0, 0]
            let worldRadius = 1

            if (localBsphere && localBsphere.radius > 0) {
                // Transform local bsphere center by world matrix
                const c = localBsphere.center
                worldCenter = [
                    worldMatrix[0] * c[0] + worldMatrix[4] * c[1] + worldMatrix[8] * c[2] + worldMatrix[12],
                    worldMatrix[1] * c[0] + worldMatrix[5] * c[1] + worldMatrix[9] * c[2] + worldMatrix[13],
                    worldMatrix[2] * c[0] + worldMatrix[6] * c[1] + worldMatrix[10] * c[2] + worldMatrix[14]
                ]
                // Scale radius by the largest axis scale in the transform
                const scaleX = Math.sqrt(worldMatrix[0]**2 + worldMatrix[1]**2 + worldMatrix[2]**2)
                const scaleY = Math.sqrt(worldMatrix[4]**2 + worldMatrix[5]**2 + worldMatrix[6]**2)
                const scaleZ = Math.sqrt(worldMatrix[8]**2 + worldMatrix[9]**2 + worldMatrix[10]**2)
                worldRadius = localBsphere.radius * Math.max(scaleX, scaleY, scaleZ)
            }

            // Store combined bsphere on mesh for shadow pass culling
            if (hasAnySkin && combinedBsphere) {
                mesh.combinedBsphere = combinedBsphere
            }

            // Add instance with world bounding sphere
            mesh.addInstance(worldCenter, worldRadius)
            mesh.updateInstance(0, worldMatrix)

            // Mark as static so instance count doesn't get reset by entity system
            mesh.static = true

            // Update geometry buffers
            if (mesh.geometry?.update) {
                mesh.geometry.update()
            }

            // Register mesh for rendering
            this.meshes[name] = mesh
        }

        // Store skins and animations
        if (result.skins) {
            this.skins = this.skins || []
            this.skins.push(...result.skins)
        }
        if (result.animations) {
            this.animations = this.animations || []
            this.animations.push(...result.animations)
        }

        return result
    }

    /**
     * Create a new entity (new data-oriented API)
     * @param {Object} data - Entity data
     * @returns {string} Entity ID
     */
    createEntity(data = {}) {
        const entityId = this.entityManager.create(data)

        // If entity has a model, ensure asset is loaded
        if (data.model) {
            const { path, meshName } = this.assetManager.parseModelId(data.model)

            // Check if asset is ready, if so update bounding sphere
            const meshAsset = this.assetManager.get(data.model)
            if (meshAsset) {
                this.entityManager.updateBoundingSphere(entityId, meshAsset.bsphere)
            } else {
                // Register callback to update bsphere when asset loads
                this.assetManager.onReady(data.model, (asset) => {
                    this.entityManager.updateBoundingSphere(entityId, asset.bsphere)
                })
            }
        }

        return entityId
    }

    /**
     * Update an entity
     * @param {string} id - Entity ID
     * @param {Object} data - Properties to update
     */
    updateEntity(id, data) {
        const result = this.entityManager.update(id, data)

        // If model changed, update bounding sphere
        if (data.model) {
            const meshAsset = this.assetManager.get(data.model)
            if (meshAsset) {
                this.entityManager.updateBoundingSphere(id, meshAsset.bsphere)
            }
        }

        return result
    }

    /**
     * Delete an entity
     * @param {string} id - Entity ID
     */
    deleteEntity(id) {
        return this.entityManager.delete(id)
    }

    /**
     * Get entity by ID
     * @param {string} id - Entity ID
     * @returns {Object|null} Entity or null if not found
     */
    getEntity(id) {
        return this.entityManager.get(id)
    }

    /**
     * Invalidate occlusion culling data and reset warmup period.
     * Call this after scene loading or major camera teleportation to prevent
     * incorrect occlusion culling with stale depth buffer data.
     */
    invalidateOcclusionCulling() {
        if (this.renderer) {
            this.renderer.invalidateOcclusionCulling()
        }
    }

    async _create() {
        let camera = new Camera(this)  // Pass engine reference
        camera.updateMatrix()
        camera.updateView()
        this.camera = camera

        // Create hidden GUI canvas for 2D overlay (UI, debugging)
        this.guiCanvas = document.createElement('canvas')
        this.guiCanvas.style.display = 'none'
        this.guiCtx = this.guiCanvas.getContext('2d')
        // Initial size will be set by _resize()

        // Load environment based on file extension
        // .jpg = octahedral RGBM pair, .hdr = equirectangular
        const envTexture = this.settings.environment.texture
        if (envTexture.toLowerCase().endsWith('.jpg') || envTexture.toLowerCase().endsWith('.jpeg')) {
            // Load octahedral RGBM JPG pair
            this.environment = await Texture.fromJPGPair(this, envTexture)
            this.environmentEncoding = 1  // octahedral
        } else {
            // Load equirectangular HDR
            this.environment = await Texture.fromImage(this, envTexture)
            this.environmentEncoding = 0  // equirectangular
        }
        this._setupInput()
    }

    async create() {
    }

    async _after_create() {
        this.renderer = await RenderGraph.create(this, this.environment, this.environmentEncoding)

        // Initialize raycaster for async ray intersection tests
        this.raycaster = new Raycaster(this)
        await this.raycaster.initialize()
    }

    _update(dt) {
        // Process input every frame
        this._updateInput();
        const ctx = this.guiCtx;
        const w = this.guiCanvas.width;
        const h = this.guiCanvas.height;
        ctx.clearRect(0, 0, w, h);

        // Debug: render light positions (uses engine's _debugSettings from DebugUI)
        if (this._debugSettings?.showLights) {
            this.debugRenderLights();
        }

/*
        // Debug: visualize HiZ occlusion buffer
        const hizPass = this.renderer?.getPass('hiz');
        if (hizPass) {
            const info = hizPass.getTileInfo();
            const data = hizPass.getHiZData();

            // Log sample depth values once per second
            if (data && info.hizDataReady && Math.floor(this.time) !== this._lastHizLog) {
                this._lastHizLog = Math.floor(this.time);
                const centerIdx = Math.floor(info.tileCountY / 2) * info.tileCountX + Math.floor(info.tileCountX / 2);
                console.log(`HiZ center tile: depth=${data[centerIdx].toFixed(6)}, near=${this.camera.near}, far=${this.camera.far}`);
            }
            if (data && info.hizDataReady) {
                const tileW = w / info.tileCountX;
                const tileH = h / info.tileCountY;

                // Get camera near/far for depth linearization
                const near = this.camera?.near ?? 0.05;
                const far = this.camera?.far ?? 5000;

                ctx.font = '12px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                for (let y = 0; y < info.tileCountY; y++) {
                    for (let x = 0; x < info.tileCountX; x++) {
                        const idx = y * info.tileCountX + x;
                        const maxZ = data[idx];

                        // Linear depth: depth 0 = near, depth 1 = far
                        // z = near + depth * (far - near)
                        const linearZ = near + maxZ * (far - near);
                        const dist = linearZ.toFixed(1);
                        const label = `${dist}m`;

                        ctx.fillStyle = 'rgba(0,0,0,0.2)';
                        ctx.fillText(label, x * tileW + tileW / 2 + 1, y * tileH + tileH / 2 + 1);

                        // White if gap (far/sky), red if geometry
                        ctx.fillStyle = maxZ >= 0.999 ? 'rgba(255,255,255,0.2)' : 'rgba(255,32,32,0.2)';
                        ctx.fillText(label, x * tileW + tileW / 2, y * tileH + tileH / 2);
                    }
                }
            }
        }
*/
    }

    update(dt) {
    }

    async _resize() {
        // Wait for any in-progress render to complete before resizing
        if (this._renderInProgress) {
            await new Promise(resolve => {
                const checkRender = () => {
                    if (!this._renderInProgress) {
                        resolve()
                    } else {
                        setTimeout(checkRender, 5)
                    }
                }
                checkRender()
            })
        }

        this._resizing = true
        let t1 = performance.now()
        const { canvas, configureContext } = this

        // Calculate effective render scale with auto-scaling
        const autoScale = this.settings?.rendering?.autoScale
        const configuredScale = this.settings?.rendering?.renderScale ?? 1.0
        let effectiveScale = configuredScale

        if (autoScale?.enabled) {
            const devicePixelRatio = window.devicePixelRatio || 1
            const nativeHeight = canvas.clientHeight * devicePixelRatio

            if (nativeHeight > autoScale.maxHeight) {
                // Apply scale reduction for high-res displays
                effectiveScale = configuredScale * (autoScale.scaleFactor ?? 0.5)
                if (!this._autoScaleWarned) {
                    console.log(`Auto-scale: Reducing render scale from ${configuredScale} to ${effectiveScale.toFixed(2)} (native height: ${nativeHeight}px > ${autoScale.maxHeight}px)`)
                    this._autoScaleWarned = true
                }
            } else {
                // Restore configured scale for lower resolutions
                if (this._autoScaleWarned) {
                    console.log(`Auto-scale: Restoring render scale to ${configuredScale} (native height: ${nativeHeight}px <= ${autoScale.maxHeight}px)`)
                    this._autoScaleWarned = false
                }
            }
        }

        // Update the effective render scale
        this.renderScale = effectiveScale

        configureContext()

        // Resize GUI canvas to match render size
        if (this.guiCanvas) {
            this.guiCanvas.width = canvas.width
            this.guiCanvas.height = canvas.height
            this.guiCtx.clearRect(0, 0, canvas.width, canvas.height)
        }

        // Pass render scale to RenderGraph - internal passes use scaled dimensions
        // CRT pass will still output at full canvas resolution
        await this.renderer.resize(canvas.width, canvas.height, this.renderScale)
        this.resize()

        // Small delay before allowing renders to ensure all GPU resources are ready
        await new Promise(resolve => setTimeout(resolve, 16))
        this._resizing = false
    }

    async resize() {
    }

    _setupInput() {
        this.keys = {};

        // Mouse/touch movement state
        this._inputDeltaX = 0;  // Accumulated raw input delta
        this._inputDeltaY = 0;
        this._smoothedX = 0;    // Smoothed output
        this._smoothedY = 0;
        this._mouseSmoothing = this.settings.engine.mouseSmoothing;

        // Idle detection for normal mode
        this._mouseIdleTime = 0;
        this._mouseIdleThreshold = this.settings.engine.mouseIdleThreshold;
        this._mouseMovedThisFrame = false;

        // Pointer lock state
        this._pointerLocked = false;
        this._mouseOnCanvas = false;

        // Touch tracking
        this._lastTouchX = 0;
        this._lastTouchY = 0;

        // Keyboard events
        window.addEventListener('keydown', (e) => {
            this.keys[e.key.toLowerCase()] = true;

            // F10 toggles debug mode (both fly camera and debug panel)
            if (e.key === 'F10') {
                this.debugMode = !this.debugMode;
                this.settings.rendering.debug = this.debugMode;
                console.log(`Debug mode: ${this.debugMode ? 'ON' : 'OFF'}`);

                // Exit pointer lock when entering debug mode
                if (this.debugMode && document.pointerLockElement) {
                    document.exitPointerLock();
                }
                e.preventDefault();
            }
        });
        window.addEventListener('keyup', (e) => { this.keys[e.key.toLowerCase()] = false; });
        window.addEventListener('blur', (e) => {
            this.keys = {}
        })
        // Mouse events
        window.addEventListener('mousedown', (e) => {
            this.keys['lmb'] = true;
            this._mouseOnCanvas = e.target === this.canvas;

            // In normal mode, click requests pointer lock for character controller
            if (!this.debugMode && e.button === 0 && this._mouseOnCanvas) {
                this.canvas.requestPointerLock();
            }
        });
        window.addEventListener('mouseup', (e) => {
            this.keys['lmb'] = false;
            this._mouseOnCanvas = false;
        });

        // Pointer lock change handler
        document.addEventListener('pointerlockchange', () => {
            this._pointerLocked = document.pointerLockElement !== null;
        });

        window.addEventListener('mousemove', (e) => {
            // In debug mode: only track when LMB pressed on canvas
            // In normal mode: always track (for character controller) when pointer locked
            if (this.debugMode) {
                if (this.keys['lmb'] && this._mouseOnCanvas) {
                    this._inputDeltaX += e.movementX;
                    this._inputDeltaY += e.movementY;
                }
            } else {
                // Normal mode: only when pointer is locked (clicked on canvas)
                if (this._pointerLocked) {
                    this._inputDeltaX += e.movementX;
                    this._inputDeltaY += e.movementY;
                    this._mouseMovedThisFrame = true;
                }
            }
        });

        // Touch events - simulate LMB + mouse movement
        window.addEventListener('touchstart', (e) => {
            this.keys['lmb'] = true;
            if (e.touches.length > 0) {
                this._lastTouchX = e.touches[0].clientX;
                this._lastTouchY = e.touches[0].clientY;
            }
            e.preventDefault();
        }, { passive: false });

        window.addEventListener('touchend', (e) => {
            if (e.touches.length === 0) {
                this.keys['lmb'] = false;
            }
        });

        window.addEventListener('touchcancel', (e) => {
            this.keys['lmb'] = false;
        });

        window.addEventListener('touchmove', (e) => {
            if (e.touches.length > 0) {
                const touch = e.touches[0];
                const dx = touch.clientX - this._lastTouchX;
                const dy = touch.clientY - this._lastTouchY;
                this._inputDeltaX += dx;
                this._inputDeltaY += dy;
                this._lastTouchX = touch.clientX;
                this._lastTouchY = touch.clientY;
                this._mouseMovedThisFrame = true;
            }
            e.preventDefault();
        }, { passive: false });

        // Escape exits pointer lock in normal mode
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this._pointerLocked) {
                document.exitPointerLock();
            }
        });
    }

    /**
     * Called every frame to process smoothed input
     * Call this from your _update() method
     */
    _updateInput() {
        const dt = this.stats.dt ? this.stats.dt / 1000.0 : 0.016;
        let camera = this.camera;

        if (this.debugMode) {
            let moveSpeed = 0.1;
            if (this.keys["shift"]) moveSpeed *= 5;
            if (this.keys[" "]) moveSpeed *= 0.1;

          // Debug mode: fly camera with WASD
            // Camera movement
            if (this.keys["w"]) {
                camera.position[0] += camera.direction[0] * moveSpeed;
                camera.position[1] += camera.direction[1] * moveSpeed;
                camera.position[2] += camera.direction[2] * moveSpeed;
            }
            if (this.keys["s"]) {
                camera.position[0] -= camera.direction[0] * moveSpeed;
                camera.position[1] -= camera.direction[1] * moveSpeed;
                camera.position[2] -= camera.direction[2] * moveSpeed;
            }
            if (this.keys["a"]) {
                camera.position[0] -= camera.right[0] * moveSpeed;
                camera.position[1] -= camera.right[1] * moveSpeed;
                camera.position[2] -= camera.right[2] * moveSpeed;
            }
            if (this.keys["d"]) {
                camera.position[0] += camera.right[0] * moveSpeed;
                camera.position[1] += camera.right[1] * moveSpeed;
                camera.position[2] += camera.right[2] * moveSpeed;
            }
            if (this.keys["space"] || this.keys["e"]) {
                camera.position[1] += moveSpeed;
            }
            if (this.keys["c"] || this.keys["q"]) {
                camera.position[1] -= moveSpeed;
            }

            if (this.keys["arrowleft"]) camera.yaw += 0.02;
            if (this.keys["arrowright"]) camera.yaw -= 0.02;
            if (this.keys["arrowup"]) camera.pitch += 0.02;
            if (this.keys["arrowdown"]) camera.pitch -= 0.02;

            // Debug mode: original behavior - only when LMB pressed
            if (this.keys['lmb']) {
                // Apply smoothing
                const dx = this._inputDeltaX - this._smoothedX;
                const dy = this._inputDeltaY - this._smoothedY;
                this._smoothedX += dx * this._mouseSmoothing;
                this._smoothedY += dy * this._mouseSmoothing;

                // Call handler with smoothed movement
                this.onMouseMove(this._smoothedX, this._smoothedY);

                // Reset accumulated input (it's been consumed)
                this._inputDeltaX = 0;
                this._inputDeltaY = 0;
            } else {
                // Decay smoothed values when not pressing
                this._smoothedX *= 0.8;
                this._smoothedY *= 0.8;
                this._inputDeltaX = 0;
                this._inputDeltaY = 0;

                // Still call onMouseMove with decaying values for smooth stop
                if (Math.abs(this._smoothedX) > 0.01 || Math.abs(this._smoothedY) > 0.01) {
                    this.onMouseMove(this._smoothedX, this._smoothedY);
                }
            }
        } else {
            // Normal mode: always smooth, call onMouseMove, stop when idle

            // Check if mouse moved this frame
            if (this._mouseMovedThisFrame) {
                this._mouseIdleTime = 0;
                this._mouseMovedThisFrame = false;
            } else {
                this._mouseIdleTime += dt;
            }

            // Apply smoothing to accumulated input
            const dx = this._inputDeltaX - this._smoothedX;
            const dy = this._inputDeltaY - this._smoothedY;
            this._smoothedX += dx * this._mouseSmoothing;
            this._smoothedY += dy * this._mouseSmoothing;

            // Reset accumulated input
            this._inputDeltaX = 0;
            this._inputDeltaY = 0;

            // Check if there's significant movement
            const hasMovement = Math.abs(this._smoothedX) > 0.01 || Math.abs(this._smoothedY) > 0.01;

            // Only call onMouseMove if there's movement and not idle for too long
            if (hasMovement && this._mouseIdleTime < this._mouseIdleThreshold) {
                this.onMouseMove(this._smoothedX, this._smoothedY);
            }

            // Decay smoothed values when idle
            if (this._mouseIdleTime >= this._mouseIdleThreshold) {
                this._smoothedX *= 0.8;
                this._smoothedY *= 0.8;
            }
        }
    }

    /**
     * Debug render: draw crosses at all light positions
     * Green = visible (in front of camera), Red = not visible (behind camera or culled)
     */
    debugRenderLights() {
        const ctx = this.guiCtx;
        const w = this.guiCanvas.width;
        const h = this.guiCanvas.height;
        const crossSize = this._debugSettings?.lightCrossSize || 10;

        // Get camera viewProj matrix (already computed by camera)
        const viewProj = this.camera.viewProj;
        if (!viewProj) return;

        // Iterate through all entities with lights
        for (const entityId in this.entityManager.entities) {
            const entity = this.entityManager.entities[entityId];
            if (!entity.light?.enabled) continue;

            // Get world position of light
            const lightPos = [
                entity.position[0] + (entity.light.position?.[0] || 0),
                entity.position[1] + (entity.light.position?.[1] || 0),
                entity.position[2] + (entity.light.position?.[2] || 0)
            ];

            // Transform to clip space
            const clipPos = vec4.fromValues(lightPos[0], lightPos[1], lightPos[2], 1.0);
            vec4.transformMat4(clipPos, clipPos, viewProj);

            // Check if behind camera (w <= 0 means behind)
            const isBehindCamera = clipPos[3] <= 0;

            // Perspective divide to get NDC
            let ndcX, ndcY;
            if (!isBehindCamera) {
                ndcX = clipPos[0] / clipPos[3];
                ndcY = clipPos[1] / clipPos[3];
            } else {
                // For lights behind camera, project them to edge of screen
                ndcX = clipPos[0] < 0 ? -2 : 2;
                ndcY = clipPos[1] < 0 ? -2 : 2;
            }

            // Convert NDC (-1 to 1) to screen coordinates
            const screenX = (ndcX + 1) * 0.5 * w;
            const screenY = (1 - ndcY) * 0.5 * h;  // Y is inverted

            // Check if on screen
            const isOnScreen = !isBehindCamera &&
                              screenX >= 0 && screenX <= w &&
                              screenY >= 0 && screenY <= h;

            // Color: green if visible, red if not
            ctx.strokeStyle = isOnScreen ? 'rgba(0, 255, 0, 0.9)' : 'rgba(255, 0, 0, 0.7)';
            ctx.lineWidth = 2;

            // Clamp to screen bounds for drawing
            const drawX = Math.max(crossSize, Math.min(w - crossSize, screenX));
            const drawY = Math.max(crossSize, Math.min(h - crossSize, screenY));

            // Draw cross
            ctx.beginPath();
            ctx.moveTo(drawX - crossSize, drawY);
            ctx.lineTo(drawX + crossSize, drawY);
            ctx.moveTo(drawX, drawY - crossSize);
            ctx.lineTo(drawX, drawY + crossSize);
            ctx.stroke();

            // Draw light type indicator
            const lightType = entity.light.lightType || 0;
            if (lightType === 2) {
                // Spotlight: draw a small cone indicator
                ctx.beginPath();
                ctx.arc(drawX, drawY, crossSize * 0.5, 0, Math.PI * 2);
                ctx.stroke();
            } else if (lightType === 1) {
                // Point light: draw small circle
                ctx.beginPath();
                ctx.arc(drawX, drawY, crossSize * 0.3, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
    }

    onMouseMove(dx, dy) {
    }
}

export {
    Engine,
    fail,
    Texture,
    Material,
    Geometry,
    Mesh,
    Camera,
    EntityManager,
    AssetManager,
    CullingSystem,
    InstanceManager,
    RenderGraph,
    ParticleSystem,
    ParticleEmitter,
    DebugUI,
    Raycaster,
}
