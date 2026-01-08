import {
    makeShaderDataDefinitions,
    makeStructuredView,
  } from 'webgpu-utils';

// Placeholder joint texture for non-skinned meshes
let _placeholderJointTexture = null
let _placeholderJointTextureView = null
let _placeholderJointSampler = null

function getPlaceholderJointTexture(engine) {
    const { device } = engine
    if (!_placeholderJointTexture) {
        // Create a 4x1 rgba32float texture (one identity matrix)
        _placeholderJointTexture = device.createTexture({
            size: [4, 1, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        // Write identity matrix
        const identityData = new Float32Array([
            1, 0, 0, 0,  // column 0
            0, 1, 0, 0,  // column 1
            0, 0, 1, 0,  // column 2
            0, 0, 0, 1,  // column 3
        ])
        device.queue.writeTexture(
            { texture: _placeholderJointTexture },
            identityData,
            { bytesPerRow: 4 * 4 * 4, rowsPerImage: 1 },
            [4, 1, 1]
        )
        _placeholderJointTextureView = _placeholderJointTexture.createView()
        _placeholderJointSampler = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        })
    }
    return {
        texture: _placeholderJointTexture,
        view: _placeholderJointTextureView,
        sampler: _placeholderJointSampler,
    }
}

// Placeholder noise texture for alpha hashing when no noise texture is configured
let _placeholderNoiseTexture = null
let _placeholderNoiseTextureView = null

function getPlaceholderNoiseTexture(engine) {
    const { device } = engine
    if (!_placeholderNoiseTexture) {
        // Create a 1x1 rgba8unorm texture with 0.5 value (middle gray)
        _placeholderNoiseTexture = device.createTexture({
            size: [1, 1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        // Write 0.5 gray value
        const noiseData = new Uint8Array([128, 128, 128, 255])
        device.queue.writeTexture(
            { texture: _placeholderNoiseTexture },
            noiseData,
            { bytesPerRow: 4, rowsPerImage: 1 },
            [1, 1, 1]
        )
        _placeholderNoiseTextureView = _placeholderNoiseTexture.createView()
    }
    return {
        texture: _placeholderNoiseTexture,
        view: _placeholderNoiseTextureView,
    }
}

class Pipeline {
    engine = null

    static async create(engine, {
        wgslSource,
        geometry,
        textures = [],
        isPostProcessing = false,
        renderTarget = null,
        uniforms = null,
        label = 'pipeline',
        skin = null,  // Optional skin for skeletal animation
        shadowPass = null, // Optional shadow pass for shadow mapping
        tileLightBuffer = null, // Optional tile light indices buffer for tiled lighting
        lightBuffer = null, // Optional light storage buffer for tiled lighting
        noiseTexture = null, // Optional noise texture for alpha hashing
    }) {
        let texture = textures[0]
        const { canvas, device, canvasFormat, options } = engine

        const shaderModule = device.createShaderModule({
            code: wgslSource,
        })
    
        const compilationInfo = await shaderModule.getCompilationInfo()
        for (const message of compilationInfo.messages) {
            let formattedMessage = ""
            if (message.lineNum) {
                formattedMessage += `Shader Error, ${label} Line ${message.lineNum}:${message.linePos} - ${wgslSource.substr(
                    message.offset,
                    message.length
                )}\n`
            }
            formattedMessage += message.message

            switch (message.type) {
                case "error":
                    console.error(formattedMessage)
                    engine.rendering = false
                    return false
                    break
                case "warning":
                    console.warn(formattedMessage)
                    break
                case "info":
                    console.log(formattedMessage)
                    break
            }
        }

        let targets = []
        if (renderTarget) {
            if (renderTarget && renderTarget.isGBuffer) {
                targets = renderTarget.getTargets()
            } else {
                targets = [{ format: renderTarget.format }]
            }
        } else {
            targets = [{ format: canvasFormat }]
        }
        let pipelineDescriptor = {
            label: label,
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "vertexMain",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: targets,
            },
            primitive: {
                topology: "triangle-list",
            },
        };

        if (!isPostProcessing) {
            pipelineDescriptor.vertex.buffers = [
                geometry.vertexBufferLayout,
                geometry.instanceBufferLayout
            ];
            pipelineDescriptor.primitive.cullMode = "back";
            pipelineDescriptor.depthStencil = {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth32float",
            };
        }

        // Use async pipeline creation for parallel shader compilation
        let pipeline = await device.createRenderPipelineAsync(pipelineDescriptor);

        const defs = makeShaderDataDefinitions(wgslSource);
        const uniformValues = makeStructuredView(defs.uniforms.uniforms);

        const uniformBuffer = device.createBuffer({
            label: label+'_uniformBuffer',
            size: uniformValues.arrayBuffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })


        // Create bind group
        let bindGroup;
        let entries, bgl
        if (isPostProcessing) {
            if (texture.isGBuffer) {
                let env = textures[1]
                if (!env || !env.view || !env.sampler) {
                    console.error('Pipeline: environment texture missing or incomplete', env)
                    throw new Error('Pipeline: environment texture required for lighting')
                }
                let sampleType = 'unfilterable-float'
                let sampleCount = 1

                // Base bind group layout entries
                let bglEntries = [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.FRAGMENT,
                        buffer: { type: 'uniform' },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: {
                            sampleType: sampleType,
                            sampleCount: sampleCount,
                        },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: {
                            sampleType: sampleType,
                            sampleCount: sampleCount,
                        },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: {
                            sampleType: sampleType,
                            sampleCount: sampleCount,
                        },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: {
                            sampleType: sampleType,
                            sampleCount: sampleCount,
                        },
                    },
                    {
                        binding: 5,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: {
                            sampleType: 'depth',
                            sampleCount: sampleCount,
                        },
                    },
                    {
                        binding: 6,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: { sampleType: 'float' },
                    },
                    {
                        binding: 7,
                        visibility: GPUShaderStage.FRAGMENT,
                        sampler: { type: 'filtering' },
                    },
                ];

                // Base bind group entries
                entries = [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: texture.albedo.view },
                    { binding: 2, resource: texture.normal.view },
                    { binding: 3, resource: texture.arm.view },
                    { binding: 4, resource: texture.emission.view },
                    { binding: 5, resource: texture.depth.view },
                    { binding: 6, resource: env.view },
                    { binding: 7, resource: env.sampler },
                ];

                // Add shadow map bindings if shadowPass is provided
                if (shadowPass) {
                    // Cascaded directional shadow map (2d-array texture)
                    bglEntries.push({
                        binding: 8,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: {
                            sampleType: 'depth',
                            viewDimension: '2d-array',
                            sampleCount: 1,
                        },
                    });
                    bglEntries.push({
                        binding: 9,
                        visibility: GPUShaderStage.FRAGMENT,
                        sampler: { type: 'comparison' },
                    });

                    // Cascade matrices storage buffer
                    bglEntries.push({
                        binding: 13,
                        visibility: GPUShaderStage.FRAGMENT,
                        buffer: { type: 'read-only-storage' },
                    });

                    entries.push({ binding: 8, resource: shadowPass.getShadowMapView() });
                    entries.push({ binding: 9, resource: shadowPass.getShadowSampler() });
                    entries.push({ binding: 13, resource: { buffer: shadowPass.getCascadeMatricesBuffer() } });

                    // Spot shadow atlas
                    bglEntries.push({
                        binding: 10,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: {
                            sampleType: 'depth',
                            sampleCount: 1,
                        },
                    });
                    bglEntries.push({
                        binding: 11,
                        visibility: GPUShaderStage.FRAGMENT,
                        sampler: { type: 'comparison' },
                    });

                    // Spot shadow matrices storage buffer
                    bglEntries.push({
                        binding: 12,
                        visibility: GPUShaderStage.FRAGMENT,
                        buffer: { type: 'read-only-storage' },
                    });

                    entries.push({ binding: 10, resource: shadowPass.getSpotShadowAtlasView() });
                    entries.push({ binding: 11, resource: shadowPass.getShadowSampler() }); // Reuse same sampler
                    entries.push({ binding: 12, resource: { buffer: shadowPass.getSpotMatricesBuffer() } });
                }

                // Add tile light indices buffer binding (binding 14) if provided
                if (tileLightBuffer) {
                    bglEntries.push({
                        binding: 14,
                        visibility: GPUShaderStage.FRAGMENT,
                        buffer: { type: 'read-only-storage' },
                    });
                    entries.push({ binding: 14, resource: { buffer: tileLightBuffer } });
                }

                // Add light buffer binding (binding 15) if provided
                if (lightBuffer) {
                    bglEntries.push({
                        binding: 15,
                        visibility: GPUShaderStage.FRAGMENT,
                        buffer: { type: 'read-only-storage' },
                    });
                    entries.push({ binding: 15, resource: { buffer: lightBuffer } });
                }

                // Add noise texture (binding 16, 17) if provided in textures[2]
                let noiseTexture = textures[2]
                if (noiseTexture && noiseTexture.view && noiseTexture.sampler) {
                    bglEntries.push({
                        binding: 16,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: { sampleType: 'float' },
                    });
                    bglEntries.push({
                        binding: 17,
                        visibility: GPUShaderStage.FRAGMENT,
                        sampler: { type: 'filtering' },
                    });
                    entries.push({ binding: 16, resource: noiseTexture.view });
                    entries.push({ binding: 17, resource: noiseTexture.sampler });
                } else if (noiseTexture) {
                    console.warn('Pipeline: noise texture missing view or sampler')
                }

                // Add AO texture (binding 18) if provided in textures[3]
                let aoTexture = textures[3]
                if (aoTexture && aoTexture.view) {
                    bglEntries.push({
                        binding: 18,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: { sampleType: 'unfilterable-float' },
                    });
                    entries.push({ binding: 18, resource: aoTexture.view });
                } else if (aoTexture) {
                    console.warn('Pipeline: aoTexture missing view')
                }

                // Create bind group layout
                bgl = device.createBindGroupLayout({
                    label: label,
                    entries: bglEntries
                });

                // Update pipeline descriptor to use explicit layout
                pipelineDescriptor.layout = device.createPipelineLayout({
                    label: label+'_layout',
                    bindGroupLayouts: [bgl]
                });

                // Create pipeline with explicit layout (async for parallel compilation)
                pipeline = await device.createRenderPipelineAsync(pipelineDescriptor);

                let bgdesc = {
                    label: label,
                    entries: entries,
                }
                if (bgl) {
                    bgdesc.layout = bgl
                }
                bindGroup = device.createBindGroup(bgdesc);
            } else {
                let pb = await Pipeline.pipelineFromTextures(engine, pipelineDescriptor, label, textures, uniformBuffer)
                pipeline = pb[0]
                bindGroup = pb[1]
            }
        } else {
            let pb = await Pipeline.pipelineFromTextures(engine, pipelineDescriptor, label, textures, uniformBuffer, skin, noiseTexture)
            pipeline = pb[0]
            bindGroup = pb[1]
            bgl = pb[2]
        }

        let p = new Pipeline()
        p.engine = engine
        p.label = label
        p.wgslSource = wgslSource
        p.geometry = geometry
        p.textures = textures
        p.pipeline = pipeline
        p.uniformBuffer = uniformBuffer
        p.uniformValues = uniformValues
        p.bindGroup = bindGroup
        p.bindGroupLayout = bgl
        p.isPostProcessing = isPostProcessing
        p.renderTarget = renderTarget
        p.skin = skin
        p.shadowPass = shadowPass
        p.tileLightBuffer = tileLightBuffer
        p.lightBuffer = lightBuffer
        p.noiseTexture = noiseTexture
        return p
    }

    static async pipelineFromTextures(engine, pipelineDescriptor, label, textures, uniformBuffer, skin = null, noiseTexture = null) {
        const { device } = engine

        // Get joint texture (either from skin or placeholder)
        let jointTextureData
        if (skin && skin.jointTexture) {
            jointTextureData = {
                view: skin.jointTextureView,
                sampler: skin.jointSampler,
            }
        } else {
            jointTextureData = getPlaceholderJointTexture(engine)
        }

        let entries = [
            { binding: 0, resource: { buffer: uniformBuffer } },
        ]
        let bgle = [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
                buffer: { type: 'uniform' },
            },
        ]
        let b = 1, b2 = 1
        for (let i = 0; i < textures.length; i++) {
            entries.push({ binding: b++, resource: textures[i].view })
            entries.push({ binding: b++, resource: textures[i].sampler })
            bgle.push({ binding: b2++, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, texture: { sampleType: 'float' } })
            bgle.push({ binding: b2++, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, sampler: { type: 'filtering' } })
        }

        // Add joint texture bindings (bindings 11 and 12 after 5 textures with 2 bindings each = 10 + uniform = 11)
        entries.push({ binding: b++, resource: jointTextureData.view })
        entries.push({ binding: b++, resource: jointTextureData.sampler })
        bgle.push({ binding: b2++, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } })
        bgle.push({ binding: b2++, visibility: GPUShaderStage.VERTEX, sampler: { type: 'non-filtering' } })

        // Add previous joint texture for motion vectors (binding 13)
        // Use same texture as current for placeholder (no motion when no history)
        let prevJointTextureView = skin?.prevJointTextureView ?? jointTextureData.view
        entries.push({ binding: b++, resource: prevJointTextureView })
        bgle.push({ binding: b2++, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } })

        // Add noise texture for alpha hashing (binding 14)
        // Always add binding - use placeholder if no noise texture provided
        const noiseTextureView = (noiseTexture && noiseTexture.view) ? noiseTexture.view : getPlaceholderNoiseTexture(engine).view
        entries.push({ binding: b++, resource: noiseTextureView })
        bgle.push({ binding: b2++, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } })

        let bgl = device.createBindGroupLayout({
            label: label+'_bgl',
            entries: bgle
        });

        pipelineDescriptor.layout = device.createPipelineLayout({
            label: label+'_pipelineLayout',
            bindGroupLayouts: [bgl]
        });

        // Create pipeline with explicit layout (async for parallel compilation)
        let pipeline = await device.createRenderPipelineAsync(pipelineDescriptor);

        let bgdesc = {
            label: label,
            entries: entries,
        }
        if (bgl) {
            bgdesc.layout = bgl
        }
        let bindGroup = device.createBindGroup(bgdesc);
        return [ pipeline, bindGroup, bgl ]
    }

    // Update bind group with new skin's joint texture (for reusing pipeline with different skins)
    updateBindGroupForSkin(skin) {
        // Cache bind group per skin to avoid recreating every frame
        const skinId = skin?.jointTexture ? skin : null
        if (this._lastSkin === skinId && this.bindGroup) {
            return  // Same skin, reuse existing bind group
        }
        this._lastSkin = skinId

        const { device } = this.engine

        // Get joint texture (either from skin or placeholder)
        let jointTextureData
        if (skin && skin.jointTexture) {
            jointTextureData = {
                view: skin.jointTextureView,
                sampler: skin.jointSampler,
            }
        } else {
            jointTextureData = getPlaceholderJointTexture(this.engine)
        }

        let entries = [
            { binding: 0, resource: { buffer: this.uniformBuffer } },
        ]
        let b = 1
        for (let i = 0; i < this.textures.length; i++) {
            entries.push({ binding: b++, resource: this.textures[i].view })
            entries.push({ binding: b++, resource: this.textures[i].sampler })
        }
        // Add joint texture bindings
        entries.push({ binding: b++, resource: jointTextureData.view })
        entries.push({ binding: b++, resource: jointTextureData.sampler })

        // Add previous joint texture for motion vectors
        let prevJointTextureView = skin?.prevJointTextureView ?? jointTextureData.view
        entries.push({ binding: b++, resource: prevJointTextureView })

        // Add noise texture for alpha hashing
        // Always add binding - use placeholder if no noise texture provided
        const noiseTextureView = (this.noiseTexture && this.noiseTexture.view) ? this.noiseTexture.view : getPlaceholderNoiseTexture(this.engine).view
        entries.push({ binding: b++, resource: noiseTextureView })

        this.bindGroup = device.createBindGroup({
            label: this.label + '_bindGroup',
            layout: this.bindGroupLayout,
            entries: entries,
        })
    }

    render(options = {}) {
        let { renderTarget } = this
        const { device, context, stats } = this.engine
        const depthTexture = this.engine.depthTexture
        
        let commandEncoder, passEncoder
        if (options.passEncoder) {
            commandEncoder = options.commandEncoder
            passEncoder = options.passEncoder
        } else {
            commandEncoder = device.createCommandEncoder()
        
            const textureView = renderTarget ? renderTarget.view : context.getCurrentTexture().createView()
        
            let renderPassDescriptor = {}
            if (renderTarget && renderTarget.isGBuffer) {
                renderPassDescriptor.colorAttachments = renderTarget.getColorAttachments()
            } else {
                renderPassDescriptor.colorAttachments = [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: "clear",
                    storeOp: "store",
                }]
            }

            if (!this.isPostProcessing) {
                if (renderTarget && renderTarget.isGBuffer) {
                    renderPassDescriptor.depthStencilAttachment = renderTarget.getDepthStencilAttachment()
                } else {
                    renderPassDescriptor.depthStencilAttachment = {
                        view: depthTexture.view,
                        depthClearValue: 1.0,
                        depthLoadOp: "clear",
                        depthStoreOp: "store",
                    }
                }
            }

            passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor)
        }

        // Draw actual pipeline

        device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformValues.arrayBuffer)
        passEncoder.setPipeline(this.pipeline)
        if (!this.isPostProcessing) {
            passEncoder.setVertexBuffer(0, this.geometry.vertexBuffer)
            passEncoder.setVertexBuffer(1, this.geometry.instanceBuffer)
            passEncoder.setIndexBuffer(this.geometry.indexBuffer, "uint32")
        }
        passEncoder.setBindGroup(0, this.bindGroup)
        if (this.isPostProcessing) {
            passEncoder.draw(3);
            stats.triangles += 1
            stats.drawCalls++
        } else {
            const instanceCount = this.geometry.instanceCount ?? 0
            if (instanceCount > 0) {
                passEncoder.drawIndexed(this.geometry.indexArray.length, instanceCount)
                stats.triangles += this.geometry.indexArray.length / 3 * instanceCount
                stats.drawCalls++
            }
        }

        if (!options.dontFinish) {
            passEncoder.end()
            device.queue.submit([commandEncoder.finish()])
        }


        return {
            commandEncoder: commandEncoder,
            passEncoder: passEncoder,
        }
    }

    finish(options = {}) {
        const { device } = this.engine
        let commandEncoder, passEncoder
        if (options.passEncoder) {
            commandEncoder = options.commandEncoder
            passEncoder = options.passEncoder
            passEncoder.end()
            device.queue.submit([commandEncoder.finish()])
        } else {
            console.warn('finish: no passEncoder')
        }
    }
}

export {
    Pipeline
}
