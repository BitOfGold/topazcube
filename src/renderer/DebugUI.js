import GUI from 'lil-gui'

/**
 * DebugUI - Stats display and settings GUI for the renderer
 *
 * Created lazily on first debug mode activation.
 * Shows stats panel and collapsible settings folders.
 */
class DebugUI {
    constructor(engine) {
        this.engine = engine
        this.gui = null
        this.statsFolder = null
        this.initialized = false

        // Stats display elements
        this._statsController = null
        this._statsText = ''

        // Stats update throttling (5 updates per second)
        this._lastStatsUpdate = 0
        this._statsUpdateInterval = 200 // ms

        // Folder references for updating
        this.folders = {}
    }

    /**
     * Initialize the debug UI (called lazily on first debug mode)
     */
    init() {
        if (this.initialized) return

        this.gui = new GUI({ title: 'Debug Panel' })
        this.gui.close() // Start collapsed

        // Add CSS for stats styling
        this._addStyles()

        // Create stats folder at the top
        this._createStatsFolder()

        // Create settings folders
        this._createRenderingFolder()
        this._createCameraFolder()
        this._createEnvironmentFolder()
        this._createLightingFolder()
        this._createMainLightFolder()
        this._createShadowFolder()
        this._createPlanarReflectionFolder()
        this._createAOFolder()
        this._createAmbientCaptureFolder()
        this._createSSGIFolder()
        this._createVolumetricFogFolder()
        this._createBloomFolder()
        this._createTonemapFolder()
        this._createDitheringFolder()
        this._createCRTFolder()
        this._createDebugFolder()

        // Close all folders by default
        for (const folder of Object.values(this.folders)) {
            folder.close()
        }

        this.initialized = true
    }

    /**
     * Add custom CSS for the debug UI
     */
    _addStyles() {
        if (document.getElementById('debug-ui-styles')) return

        const style = document.createElement('style')
        style.id = 'debug-ui-styles'
        style.textContent = `
            .lil-gui .stats-display {
                font-family: monospace;
                font-size: 11px;
                line-height: 1.4;
                white-space: pre;
                padding: 4px 8px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 3px;
                color: #8f8;
            }
            .lil-gui .stats-title {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .lil-gui .stats-title::before {
                content: '📊';
            }
        `
        document.head.appendChild(style)
    }

    /**
     * Create stats folder with live updating display
     */
    _createStatsFolder() {
        this.statsFolder = this.gui.addFolder('Stats')
        this.statsFolder.$title.classList.add('stats-title')

        // Create a custom stats display element
        const statsDiv = document.createElement('div')
        statsDiv.className = 'stats-display'
        statsDiv.textContent = 'Loading...'
        this._statsElement = statsDiv

        // Add to folder
        this.statsFolder.$children.appendChild(statsDiv)

        // Keep stats folder open by default
        this.statsFolder.open()
    }

    /**
     * Update stats display (throttled to 5 updates/sec, only when panel is open)
     */
    updateStats() {
        if (!this._statsElement || !this.engine.stats) return

        // Only update if GUI panel is open
        if (this.gui?._closed) return

        // Throttle updates to 5 per second
        const now = performance.now()
        if (now - this._lastStatsUpdate < this._statsUpdateInterval) return
        this._lastStatsUpdate = now

        const s = this.engine.stats
        const fmt = this._fmt.bind(this)

        // Basic timing stats
        const fps = s.avg_fps?.toFixed(0) || '0'
        const ms = s.avg_dt?.toFixed(1) || '0'
        const renderMs = s.avg_dt_render?.toFixed(2) || '0'

        // Draw calls breakdown
        const dc = s.totalDC || s.drawCalls || 0
        const dcParts = []
        if (s.shadowDC) dcParts.push(`sh:${fmt(s.shadowDC)}`)
        if (s.planarDC) dcParts.push(`pl:${fmt(s.planarDC)}`)
        if (s.transparentDC) dcParts.push(`tr:${fmt(s.transparentDC)}`)
        const dcBreakdown = dcParts.length > 0 ? ` (${dcParts.join(' ')})` : ''

        // Triangles breakdown
        const tri = s.totalTri || s.triangles || 0
        const triStr = this._formatNumber(tri)
        const triParts = []
        if (s.shadowTri) triParts.push(`sh:${this._formatNumber(s.shadowTri)}`)
        if (s.planarTri) triParts.push(`pl:${this._formatNumber(s.planarTri)}`)
        if (s.transparentTri) triParts.push(`tr:${this._formatNumber(s.transparentTri)}`)
        const triBreakdown = triParts.length > 0 ? ` (${triParts.join(' ')})` : ''

        // Get renderer stats
        const renderer = this.engine.renderer
        const lightingStats = renderer?.getPass?.('lighting')?.stats || {}
        const gbufferStats = renderer?.getPass?.('gbuffer')?.legacyCullingStats || {}
        const occStats = renderer?.cullingSystem?.getOcclusionStats?.() || {}
        const rendererStats = renderer?.stats || {}

        // Entity culling stats
        const visibleEntities = rendererStats.visibleEntities || 0
        const occCulled = occStats.culled || 0

        // Mesh culling stats
        const meshRendered = gbufferStats.rendered || 0
        const meshTotal = gbufferStats.total || 0
        const meshOccCulled = gbufferStats.culledByOcclusion || 0
        const meshSkipped = gbufferStats.skippedNoBsphere || 0

        // Shadow entity culling
        const shEntCull = []
        if (s.shadowFrustumCulled) shEntCull.push(`fr:${fmt(s.shadowFrustumCulled)}`)
        if (s.shadowDistanceCulled) shEntCull.push(`dist:${fmt(s.shadowDistanceCulled)}`)
        if (s.shadowHiZCulled) shEntCull.push(`hiz:${fmt(s.shadowHiZCulled)}`)
        if (s.shadowPixelCulled) shEntCull.push(`px:${fmt(s.shadowPixelCulled)}`)
        const shEntStr = shEntCull.length > 0 ? ` ${shEntCull.join(' ')}` : ''

        // Shadow mesh culling
        const shMeshCull = []
        if (s.shadowMeshFrustumCulled) shMeshCull.push(`fr:${fmt(s.shadowMeshFrustumCulled)}`)
        if (s.shadowMeshDistanceCulled) shMeshCull.push(`dist:${fmt(s.shadowMeshDistanceCulled)}`)
        if (s.shadowMeshOcclusionCulled) shMeshCull.push(`occ:${fmt(s.shadowMeshOcclusionCulled)}`)
        const shMeshStr = shMeshCull.length > 0 ? ` ${shMeshCull.join(' ')}` : ''

        // Lights stats
        const visibleLights = lightingStats.visibleLights || 0
        const lightOccCulled = lightingStats.culledByOcclusion || 0

        this._statsElement.textContent =
            `FPS: ${fps} (${ms}ms) render: ${renderMs}ms\n` +
            `DC: ${fmt(dc)}${dcBreakdown}\n` +
            `Tri: ${triStr}${triBreakdown}\n` +
            `Entities: ${fmt(visibleEntities)} (occ:${fmt(occCulled)})\n` +
            `Meshes: ${fmt(meshRendered)}/${fmt(meshTotal)} (occ:${fmt(meshOccCulled)} skip:${fmt(meshSkipped)})\n` +
            `Shadow ent:${shEntStr || ' -'} mesh:${shMeshStr || ' -'}\n` +
            `Lights: ${fmt(visibleLights)} (occ:${fmt(lightOccCulled)})`
    }

    /**
     * Format number with thin space as thousand separator
     */
    _fmt(n) {
        if (n === undefined || n === null) return '0'
        return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, '\u2009')
    }

    /**
     * Format large numbers with K/M suffix
     */
    _formatNumber(n) {
        if (n > 1000000) return (n / 1000000).toFixed(1) + 'M'
        if (n > 1000) return (n / 1000).toFixed(0) + 'K'
        return n.toString()
    }

    _createRenderingFolder() {
        const s = this.engine.settings
        const folder = this.gui.addFolder('Rendering')
        this.folders.rendering = folder

        folder.add(s.rendering, 'renderScale', 0.25, 2.0, 0.25).name('Render Scale')
            .onChange(() => this.engine.needsResize = true)
        folder.add(s.rendering.autoScale, 'enabled').name('Auto-Scale (4K)')
            .onChange(() => this.engine.needsResize = true)
        folder.add(s.rendering.autoScale, 'maxHeight', 720, 2160, 1).name('Max Height')
        folder.add(s.rendering.autoScale, 'scaleFactor', 0.25, 1.0, 0.05).name('Scale Factor')
        folder.add(s.rendering, 'fxaa').name('FXAA')
        folder.add(s.rendering, 'jitter').name('TAA Jitter')
        if (s.rendering.jitterAmount !== undefined) {
            folder.add(s.rendering, 'jitterAmount', 0, 1, 0.01).name('Jitter Amount')
        }
        if (s.rendering.jitterFadeDistance !== undefined) {
            folder.add(s.rendering, 'jitterFadeDistance', 5, 100, 1).name('Jitter Fade Dist')
        }
        folder.add(s.rendering, 'debug').name('Debug Mode')
            .onChange((value) => {
                // Immediately hide panel when debug mode is turned off
                if (!value && this.gui) {
                    this.gui.domElement.style.display = 'none'
                }
            })

        // Culling subfolder
        if (s.culling) {
            const cullFolder = folder.addFolder('Culling')
            cullFolder.add(s.culling, 'frustumEnabled').name('Frustum Culling')

            if (s.occlusionCulling) {
                cullFolder.add(s.occlusionCulling, 'enabled').name('Occlusion Culling')
                cullFolder.add(s.occlusionCulling, 'threshold', 0.1, 2.0, 0.1).name('Occlusion Threshold')
            }

            // Planar reflection culling sub-folder
            if (s.culling.planarReflection) {
                const prFolder = cullFolder.addFolder('Planar Reflection')
                prFolder.add(s.culling.planarReflection, 'frustum').name('Frustum Culling')
                prFolder.add(s.culling.planarReflection, 'maxDistance', 10, 200, 10).name('Max Distance')
                prFolder.add(s.culling.planarReflection, 'maxSkinned', 0, 100, 1).name('Max Skinned')
                prFolder.add(s.culling.planarReflection, 'minPixelSize', 0, 16, 1).name('Min Pixel Size')
                prFolder.close()
            }
            cullFolder.close()
        }

        // Noise subfolder
        if (s.noise) {
            const noiseFolder = folder.addFolder('Noise')
            noiseFolder.add(s.noise, 'type', ['bluenoise', 'bayer8']).name('Type')
                .onChange(() => {
                    // Reload noise texture when type changes
                    if (this.engine.renderer?.renderGraph) {
                        this.engine.renderer.renderGraph.reloadNoiseTexture()
                    }
                })
            noiseFolder.add(s.noise, 'animated').name('Animated')
            noiseFolder.close()
        }
    }

    _createAOFolder() {
        const s = this.engine.settings
        if (!s.ao) return

        const folder = this.gui.addFolder('Ambient Occlusion')
        this.folders.ao = folder

        folder.add(s.ao, 'enabled').name('Enabled')
        folder.add(s.ao, 'intensity', 0, 2, 0.1).name('Intensity')
        folder.add(s.ao, 'radius', 8, 128, 1).name('Radius')
        folder.add(s.ao, 'fadeDistance', 10, 100, 1).name('Fade Distance')
        folder.add(s.ao, 'bias', 0, 0.05, 0.001).name('Bias')
        folder.add(s.ao, 'level', 0, 1, 0.05).name('Level')
    }

    _createShadowFolder() {
        const s = this.engine.settings
        if (!s.shadow) return

        const folder = this.gui.addFolder('Shadows')
        this.folders.shadow = folder

        folder.add(s.shadow, 'strength', 0, 1, 0.05).name('Strength')
        if (s.shadow.spotMaxDistance !== undefined) {
            folder.add(s.shadow, 'spotMaxDistance', 10, 200, 5).name('Spot Max Distance')
        }
        if (s.shadow.spotFadeStart !== undefined) {
            folder.add(s.shadow, 'spotFadeStart', 10, 150, 5).name('Spot Fade Start')
        }
        folder.add(s.shadow, 'bias', 0, 0.01, 0.0001).name('Bias')
        folder.add(s.shadow, 'normalBias', 0, 0.05, 0.001).name('Normal Bias')
        folder.add(s.shadow, 'surfaceBias', 0, 0.1, 0.001).name('Surface Bias')
    }

    _createMainLightFolder() {
        const s = this.engine.settings
        if (!s.mainLight) return

        const folder = this.gui.addFolder('Main Light')
        this.folders.mainLight = folder

        folder.add(s.mainLight, 'enabled').name('Enabled')
        folder.add(s.mainLight, 'intensity', 0, 2, 0.05).name('Intensity')

        // Color picker
        folder.addColor({ color: this._rgbToHex(s.mainLight.color) }, 'color')
            .name('Color')
            .onChange((hex) => {
                const rgb = this._hexToRgb(hex)
                s.mainLight.color = [rgb.r, rgb.g, rgb.b]
            })

        // Direction controls
        const dirProxy = {
            x: s.mainLight.direction[0],
            y: s.mainLight.direction[1],
            z: s.mainLight.direction[2]
        }
        const updateDir = () => {
            s.mainLight.direction = [dirProxy.x, dirProxy.y, dirProxy.z]
        }
        folder.add(dirProxy, 'x', -1, 1, 0.05).name('Direction X').onChange(updateDir)
        folder.add(dirProxy, 'y', -1, 1, 0.05).name('Direction Y').onChange(updateDir)
        folder.add(dirProxy, 'z', -1, 1, 0.05).name('Direction Z').onChange(updateDir)
    }

    _createEnvironmentFolder() {
        const s = this.engine.settings
        if (!s.environment) return

        const folder = this.gui.addFolder('Environment')
        this.folders.environment = folder

        folder.add(s.environment, 'exposure', 0.1, 4, 0.1).name('Exposure')
        folder.add(s.environment, 'diffuse', 0, 10, 0.1).name('Diffuse IBL')
        folder.add(s.environment, 'specular', 0, 10, 0.1).name('Specular IBL')

        // Fog subfolder
        if (s.environment.fog) {
            const fogFolder = folder.addFolder('Fog')
            fogFolder.add(s.environment.fog, 'enabled').name('Enabled')
            fogFolder.addColor({ color: this._rgbToHex(s.environment.fog.color) }, 'color')
                .name('Color')
                .onChange((hex) => {
                    const rgb = this._hexToRgb(hex)
                    s.environment.fog.color[0] = rgb.r
                    s.environment.fog.color[1] = rgb.g
                    s.environment.fog.color[2] = rgb.b
                })
            fogFolder.add(s.environment.fog.distances, '0', 0, 50, 1).name('Near Distance')
            fogFolder.add(s.environment.fog.distances, '1', 0, 200, 5).name('Mid Distance')
            fogFolder.add(s.environment.fog.distances, '2', 50, 500, 10).name('Far Distance')
            fogFolder.add(s.environment.fog.alpha, '0', 0, 1, 0.01).name('Near Alpha')
            fogFolder.add(s.environment.fog.alpha, '1', 0, 1, 0.01).name('Mid Alpha')
            fogFolder.add(s.environment.fog.alpha, '2', 0, 1, 0.01).name('Far Alpha')
            fogFolder.add(s.environment.fog.heightFade, '0', -100, 100, 1).name('Bottom Y')
            fogFolder.add(s.environment.fog.heightFade, '1', -50, 200, 5).name('Top Y')
            fogFolder.add(s.environment.fog, 'brightResist', 0, 1, 0.05).name('Bright Resist')
            // Initialize debug if not present
            if (s.environment.fog.debug === undefined) s.environment.fog.debug = 0
            fogFolder.add(s.environment.fog, 'debug', 0, 10, 1).name('Debug Mode')
        }
    }

    _createLightingFolder() {
        const s = this.engine.settings
        if (!s.lighting) return

        const folder = this.gui.addFolder('Lighting System')
        this.folders.lighting = folder

        folder.add(s.lighting, 'cullingEnabled').name('Light Culling')
        folder.add(s.lighting, 'maxDistance', 50, 500, 10).name('Max Distance')
        if (s.lighting.specularBoost !== undefined) {
            folder.add(s.lighting, 'specularBoost', 0, 2, 0.05).name('Specular Boost')
        }
        if (s.lighting.specularBoostRoughnessCutoff !== undefined) {
            folder.add(s.lighting, 'specularBoostRoughnessCutoff', 0.1, 1.0, 0.05).name('Boost Roughness Cutoff')
        }
    }

    _createSSGIFolder() {
        const s = this.engine.settings
        if (!s.ssgi) return

        const folder = this.gui.addFolder('SSGI (Indirect Light)')
        this.folders.ssgi = folder

        folder.add(s.ssgi, 'enabled').name('Enabled')
        folder.add(s.ssgi, 'intensity', 0.0, 5.0, 0.1).name('Intensity')
        folder.add(s.ssgi, 'emissiveBoost', 0.0, 50.0, 1.0).name('Emissive Boost')
        folder.add(s.ssgi, 'maxBrightness', 1.0, 50.0, 1.0).name('Max Brightness')
        folder.add(s.ssgi, 'sampleRadius', 0.5, 4.0, 0.5).name('Sample Radius')
        folder.add(s.ssgi, 'saturateLevel', 0.1, 2.0, 0.1).name('Saturate Level')
    }

    _createVolumetricFogFolder() {
        const s = this.engine.settings
        if (!s.volumetricFog) return

        // Initialize defaults for missing properties
        const vf = s.volumetricFog
        if (vf.density === undefined && vf.densityMultiplier === undefined) vf.density = 0.5
        if (vf.scatterStrength === undefined) vf.scatterStrength = 1.0
        if (!vf.heightRange) vf.heightRange = [-5, 20]
        if (vf.resolution === undefined) vf.resolution = 0.25
        if (vf.maxSamples === undefined) vf.maxSamples = 32
        if (vf.blurRadius === undefined) vf.blurRadius = 4
        if (vf.noiseStrength === undefined) vf.noiseStrength = 1.0
        if (vf.noiseScale === undefined) vf.noiseScale = 0.25
        if (vf.noiseAnimated === undefined) vf.noiseAnimated = true
        if (vf.shadowsEnabled === undefined) vf.shadowsEnabled = true
        if (vf.mainLightScatter === undefined) vf.mainLightScatter = 1.0
        if (vf.mainLightScatterDark === undefined) vf.mainLightScatterDark = 3.0
        if (vf.mainLightSaturation === undefined) vf.mainLightSaturation = 1.0
        if (vf.brightnessThreshold === undefined) vf.brightnessThreshold = 1.0
        if (vf.minVisibility === undefined) vf.minVisibility = 0.15
        if (vf.skyBrightness === undefined) vf.skyBrightness = 5.0
        if (vf.debug === undefined) vf.debug = 0

        const folder = this.gui.addFolder('Volumetric Fog')
        this.folders.volumetricFog = folder

        folder.add(vf, 'enabled').name('Enabled')

        // Density - use whichever property exists
        if (vf.density !== undefined) {
            folder.add(vf, 'density', 0.0, 2.0, 0.05).name('Density')
        } else if (vf.densityMultiplier !== undefined) {
            folder.add(vf, 'densityMultiplier', 0.0, 2.0, 0.05).name('Density')
        }

        folder.add(vf, 'scatterStrength', 0.0, 10.0, 0.1).name('Scatter (Lights)')
        folder.add(vf, 'mainLightScatter', 0.0, 5.0, 0.1).name('Sun Scatter (Light)')
        folder.add(vf, 'mainLightScatterDark', 0.0, 10.0, 0.1).name('Sun Scatter (Dark)')
        folder.add(vf, 'mainLightSaturation', 0.0, 1.0, 0.01).name('Sun Saturation')
        folder.add(vf, 'brightnessThreshold', 0.1, 5.0, 0.1).name('Bright Threshold')
        folder.add(vf, 'minVisibility', 0.0, 1.0, 0.05).name('Min Visibility')
        folder.add(vf, 'skyBrightness', 0.0, 10.0, 0.5).name('Sky Brightness')
        folder.add(vf.heightRange, '0', -50, 50, 1).name('Height Bottom')
        folder.add(vf.heightRange, '1', -10, 100, 1).name('Height Top')

        // Quality sub-folder
        const qualityFolder = folder.addFolder('Quality')
        qualityFolder.add(vf, 'resolution', 0.125, 0.5, 0.125).name('Resolution')
        qualityFolder.add(vf, 'maxSamples', 16, 128, 8).name('Max Samples')
        qualityFolder.add(vf, 'blurRadius', 0, 8, 1).name('Blur Radius')
        qualityFolder.close()

        // Noise sub-folder
        const noiseFolder = folder.addFolder('Noise')
        noiseFolder.add(vf, 'noiseStrength', 0.0, 1.0, 0.1).name('Strength')
        noiseFolder.add(vf, 'noiseScale', 0.05, 1.0, 0.05).name('Scale (Detail)')
        noiseFolder.add(vf, 'noiseAnimated').name('Animated')
        noiseFolder.close()

        // Shadows & Debug
        folder.add(vf, 'shadowsEnabled').name('Shadows')
        folder.add(vf, 'debug', 0, 12, 1).name('Debug Mode')
    }

    _createBloomFolder() {
        const s = this.engine.settings
        if (!s.bloom) return

        const folder = this.gui.addFolder('Bloom')
        this.folders.bloom = folder

        folder.add(s.bloom, 'enabled').name('Enabled')
        folder.add(s.bloom, 'intensity', 0.0, 1.0, 0.01).name('Intensity')
        folder.add(s.bloom, 'threshold', 0.0, 2.0, 0.01).name('Threshold')
        folder.add(s.bloom, 'softThreshold', 0.0, 1.0, 0.1).name('Soft Threshold')
        folder.add(s.bloom, 'radius', 8, 128, 4).name('Blur Radius')
        folder.add(s.bloom, 'emissiveBoost', 0.0, 20.0, 0.1).name('Emissive Boost')
        folder.add(s.bloom, 'maxBrightness', 1.0, 10.0, 0.5).name('Max Brightness')
        if (s.bloom.scale !== undefined) {
            folder.add(s.bloom, 'scale', 0.25, 1.0, 0.25).name('Resolution Scale')
        }
    }

    _createTonemapFolder() {
        const s = this.engine.settings
        if (!s.rendering) return

        // Ensure tonemapMode exists
        if (s.rendering.tonemapMode === undefined) s.rendering.tonemapMode = 0

        const folder = this.gui.addFolder('Tone Mapping')
        this.folders.tonemap = folder

        folder.add(s.rendering, 'tonemapMode', { 'ACES': 0, 'Reinhard': 1, 'None (Linear)': 2 }).name('Mode')
    }

    _createPlanarReflectionFolder() {
        const s = this.engine.settings
        if (!s.planarReflection) return

        const folder = this.gui.addFolder('Planar Reflection')
        this.folders.planarReflection = folder

        folder.add(s.planarReflection, 'enabled').name('Enabled')
        folder.add(s.planarReflection, 'groundLevel', -10, 10, 0.1).name('Ground Level')
        folder.add(s.planarReflection, 'resolution', 0.25, 1.0, 0.25).name('Resolution')
        folder.add(s.planarReflection, 'roughnessCutoff', 0.0, 1.0, 0.05).name('Roughness Cutoff')
        folder.add(s.planarReflection, 'normalPerturbation', 0.0, 2.0, 0.1).name('Normal Perturbation')
        folder.add(s.planarReflection, 'distanceFade', 0.1, 10.0, 0.1).name('Distance Fade')
    }

    _createAmbientCaptureFolder() {
        const s = this.engine.settings
        if (!s.ambientCapture) return

        const folder = this.gui.addFolder('Probe GI (Ambient Capture)')
        this.folders.ambientCapture = folder

        folder.add(s.ambientCapture, 'enabled').name('Enabled')
        folder.add(s.ambientCapture, 'intensity', 0.0, 2.0, 0.05).name('Intensity')
        folder.add(s.ambientCapture, 'maxDistance', 5, 100, 5).name('Max Distance')
        folder.add(s.ambientCapture, 'emissiveBoost', 0.0, 10.0, 0.1).name('Emissive Boost')
        folder.add(s.ambientCapture, 'saturateLevel', 0.0, 2.0, 0.05).name('Saturate Level')
    }

    _createDitheringFolder() {
        const s = this.engine.settings
        if (!s.dithering) return

        const folder = this.gui.addFolder('Dithering')
        this.folders.dithering = folder

        folder.add(s.dithering, 'enabled').name('Enabled')
        folder.add(s.dithering, 'colorLevels', 4, 256, 1).name('Color Levels')
    }

    _createCRTFolder() {
        const s = this.engine.settings
        if (!s.crt) return

        const folder = this.gui.addFolder('CRT Effect')
        this.folders.crt = folder

        folder.add(s.crt, 'enabled').name('CRT Enabled')
        folder.add(s.crt, 'upscaleEnabled').name('Upscale Only')
        folder.add(s.crt, 'upscaleTarget', 1, 8, 1).name('Upscale Target')

        // Geometry sub-folder
        const geomFolder = folder.addFolder('Geometry')
        geomFolder.add(s.crt, 'curvature', 0, 0.25, 0.005).name('Curvature')
        geomFolder.add(s.crt, 'cornerRadius', 0, 0.2, 0.005).name('Corner Radius')
        geomFolder.add(s.crt, 'zoom', 1.0, 1.25, 0.005).name('Zoom')
        geomFolder.close()

        // Scanlines sub-folder
        const scanFolder = folder.addFolder('Scanlines')
        scanFolder.add(s.crt, 'scanlineIntensity', 0, 1, 0.05).name('Intensity')
        scanFolder.add(s.crt, 'scanlineWidth', 0, 1, 0.05).name('Width')
        scanFolder.add(s.crt, 'scanlineBrightBoost', 0, 2, 0.05).name('Bright Boost')
        scanFolder.add(s.crt, 'scanlineHeight', 1, 10, 1).name('Height (px)')
        scanFolder.close()

        // Convergence sub-folder
        const convFolder = folder.addFolder('RGB Convergence')
        convFolder.add(s.crt.convergence, '0', -3, 3, 0.1).name('Red X Offset')
        convFolder.add(s.crt.convergence, '1', -3, 3, 0.1).name('Green X Offset')
        convFolder.add(s.crt.convergence, '2', -3, 3, 0.1).name('Blue X Offset')
        convFolder.close()

        // Phosphor mask sub-folder
        const maskFolder = folder.addFolder('Phosphor Mask')
        maskFolder.add(s.crt, 'maskType', ['none', 'aperture', 'slot', 'shadow']).name('Type')
        maskFolder.add(s.crt, 'maskIntensity', 0, 1, 0.05).name('Intensity')
        maskFolder.close()

        // Vignette sub-folder
        const vigFolder = folder.addFolder('Vignette')
        vigFolder.add(s.crt, 'vignetteIntensity', 0, 1, 0.05).name('Intensity')
        vigFolder.add(s.crt, 'vignetteSize', 0.1, 1, 0.05).name('Size')
        vigFolder.close()

        // Blur (top level, not in subfolder)
        folder.add(s.crt, 'blurSize', 0, 8, 0.1).name('H-Blur (px)')
    }

    _createCameraFolder() {
        const s = this.engine.settings
        if (!s.camera) return

        const folder = this.gui.addFolder('Camera')
        this.folders.camera = folder

        folder.add(s.camera, 'fov', 30, 120, 1).name('FOV').onChange(() => {
            if (this.engine.camera) this.engine.camera.fov = s.camera.fov
        })
        folder.add(s.camera, 'near', 0.01, 1, 0.01).name('Near Plane').onChange(() => {
            if (this.engine.camera) this.engine.camera.near = s.camera.near
        })
        folder.add(s.camera, 'far', 100, 10000, 100).name('Far Plane').onChange(() => {
            if (this.engine.camera) this.engine.camera.far = s.camera.far
        })
    }

    _createDebugFolder() {
        const folder = this.gui.addFolder('Debug Options')
        this.folders.debug = folder

        // Add engine-level debug options
        if (!this.engine._debugSettings) {
            this.engine._debugSettings = {
                showLights: false,
                lightCrossSize: 10
            }
        }

        folder.add(this.engine._debugSettings, 'showLights').name('Show Lights')
        folder.add(this.engine._debugSettings, 'lightCrossSize', 5, 30, 1).name('Cross Size')
    }

    /**
     * Show or hide the debug UI
     */
    setVisible(visible) {
        if (visible && !this.initialized) {
            this.init()
        }

        if (this.gui) {
            this.gui.domElement.style.display = visible ? '' : 'none'
        }
    }

    /**
     * Check if debug mode is on and update visibility
     */
    update() {
        const debugMode = this.engine.settings?.rendering?.debug ?? false

        if (debugMode && !this.initialized) {
            this.init()
        }

        if (this.gui) {
            this.gui.domElement.style.display = debugMode ? '' : 'none'
        }

        if (debugMode && this.initialized) {
            this.updateStats()
        }
    }

    /**
     * Destroy the debug UI
     */
    destroy() {
        if (this.gui) {
            this.gui.destroy()
            this.gui = null
        }
        this.initialized = false
    }

    // Utility functions
    _rgbToHex(rgb) {
        const r = Math.round((rgb[0] || 0) * 255)
        const g = Math.round((rgb[1] || 0) * 255)
        const b = Math.round((rgb[2] || 0) * 255)
        return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
    }

    _hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
        return result ? {
            r: parseInt(result[1], 16) / 255,
            g: parseInt(result[2], 16) / 255,
            b: parseInt(result[3], 16) / 255
        } : { r: 1, g: 1, b: 1 }
    }
}

export { DebugUI }
