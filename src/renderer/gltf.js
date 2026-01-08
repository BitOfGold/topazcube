import { parse } from '@loaders.gl/core';
import { GLTFLoader } from '@loaders.gl/gltf';
import { Geometry } from "./Geometry.js"
import { Texture } from "./Texture.js"
import { Material } from "./Material.js"
import { Mesh } from "./Mesh.js"
import { Skin, Joint } from "./Skin.js"

/**
 * Expand triangles by moving vertices away from triangle centroids
 * This creates small overlaps to eliminate gaps between adjacent triangles
 * @param {Object} attributes - Mesh attributes (position, normal, uv, indices, etc.)
 * @param {number} expansion - Expansion distance in model units (e.g., 0.025 meters)
 * @returns {Object} New attributes with expanded triangles (vertices duplicated per triangle)
 */
function expandTriangles(attributes, expansion) {
    const positions = attributes.position
    const normals = attributes.normal
    const uvs = attributes.uv
    const indices = attributes.indices
    const weights = attributes.weights
    const joints = attributes.joints

    if (!positions || !indices) return attributes

    const triCount = indices.length / 3

    // Create new arrays - each triangle gets its own 3 vertices
    const newPositions = new Float32Array(triCount * 3 * 3)
    const newNormals = normals ? new Float32Array(triCount * 3 * 3) : null
    const newUvs = uvs ? new Float32Array(triCount * 3 * 2) : null
    const newWeights = weights ? new Float32Array(triCount * 3 * 4) : null
    const newJoints = joints ? new Uint16Array(triCount * 3 * 4) : null
    const newIndices = new Uint32Array(triCount * 3)

    for (let t = 0; t < triCount; t++) {
        const i0 = indices[t * 3 + 0]
        const i1 = indices[t * 3 + 1]
        const i2 = indices[t * 3 + 2]

        // Get original positions
        const p0 = [positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]]
        const p1 = [positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]]
        const p2 = [positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]]

        // Calculate centroid
        const cx = (p0[0] + p1[0] + p2[0]) / 3
        const cy = (p0[1] + p1[1] + p2[1]) / 3
        const cz = (p0[2] + p1[2] + p2[2]) / 3

        // Expand each vertex away from centroid
        for (let v = 0; v < 3; v++) {
            const origIdx = [i0, i1, i2][v]
            const p = [p0, p1, p2][v]
            const newIdx = t * 3 + v

            // Direction from centroid to vertex
            let dx = p[0] - cx
            let dy = p[1] - cy
            let dz = p[2] - cz
            const len = Math.sqrt(dx * dx + dy * dy + dz * dz)

            if (len > 0.0001) {
                dx /= len
                dy /= len
                dz /= len
            }

            // Expanded position
            newPositions[newIdx * 3 + 0] = p[0] + dx * expansion
            newPositions[newIdx * 3 + 1] = p[1] + dy * expansion
            newPositions[newIdx * 3 + 2] = p[2] + dz * expansion

            // Copy normals
            if (newNormals) {
                newNormals[newIdx * 3 + 0] = normals[origIdx * 3 + 0]
                newNormals[newIdx * 3 + 1] = normals[origIdx * 3 + 1]
                newNormals[newIdx * 3 + 2] = normals[origIdx * 3 + 2]
            }

            // Copy UVs
            if (newUvs) {
                newUvs[newIdx * 2 + 0] = uvs[origIdx * 2 + 0]
                newUvs[newIdx * 2 + 1] = uvs[origIdx * 2 + 1]
            }

            // Copy weights
            if (newWeights) {
                newWeights[newIdx * 4 + 0] = weights[origIdx * 4 + 0]
                newWeights[newIdx * 4 + 1] = weights[origIdx * 4 + 1]
                newWeights[newIdx * 4 + 2] = weights[origIdx * 4 + 2]
                newWeights[newIdx * 4 + 3] = weights[origIdx * 4 + 3]
            }

            // Copy joints
            if (newJoints) {
                newJoints[newIdx * 4 + 0] = joints[origIdx * 4 + 0]
                newJoints[newIdx * 4 + 1] = joints[origIdx * 4 + 1]
                newJoints[newIdx * 4 + 2] = joints[origIdx * 4 + 2]
                newJoints[newIdx * 4 + 3] = joints[origIdx * 4 + 3]
            }

            // New indices are sequential
            newIndices[newIdx] = newIdx
        }
    }

    return {
        position: newPositions,
        normal: newNormals,
        uv: newUvs,
        indices: newIndices,
        weights: newWeights,
        joints: newJoints
    }
}

async function loadGltfData(engine, url, options = {}) {
    options = {
        scale: 1.0,
        rotation: [0, 0, 0],
        expandTriangles: 0,  // Expansion distance in model units (0 = disabled)
        ...options
    }
    const gltf = await parse(fetch(url), GLTFLoader)
    //console.log('loaded gltf '+url, gltf)
    // Transform GLTF into desired format
    const meshes = {};
    const nodes = [];
    const skins = [];
    const animations = [];

    function getTexture(texture_id) {
        let tex = gltf.json.textures[texture_id]
        let filter = 'linear'
        let wrapS = 'mirror-repeat'  // Default to mirror-repeat to avoid black borders
        let wrapT = 'mirror-repeat'
        if (tex.sampler !== undefined) {
            const samplerData = gltf.json.samplers[tex.sampler]
            // Filter mode
            if (samplerData.magFilter === 9728) filter = 'nearest'
            else if (samplerData.magFilter === 9729) filter = 'linear'
            // Wrap modes (GLTF: 33071=CLAMP, 33648=MIRROR, 10497=REPEAT)
            // Default to mirror-repeat unless explicitly set to repeat
            if (samplerData.wrapS === 10497) wrapS = 'repeat'
            else if (samplerData.wrapS === 33648) wrapS = 'mirror-repeat'
            else wrapS = 'mirror-repeat'  // CLAMP or undefined -> mirror
            if (samplerData.wrapT === 10497) wrapT = 'repeat'
            else if (samplerData.wrapT === 33648) wrapT = 'mirror-repeat'
            else wrapT = 'mirror-repeat'  // CLAMP or undefined -> mirror
        }
        let image = gltf.images[tex.source]
        // Get image URI from GLTF JSON (for sourceUrl tracking)
        let imageUri = gltf.json.images?.[tex.source]?.uri || null
        return {image: image, filter: filter, wrapS: wrapS, wrapT: wrapT, uri: imageUri}
    }

    function getTextures(property) {
        let textures = {}
        for (const [key, value] of Object.entries(property)) {
            if (key.includes('Texture') && value.index !== undefined) {
                property[key] = getTexture(value.index)
            }
        }
        return property
    }

    function getMaterial(material_id) {
        if (material_id === undefined) {
            return {
                pbrMetallicRoughness: {
                    baseColorFactor: [1, 1, 1, 1],
                    metallicFactor: 0,
                    roughnessFactor: 1
                }
            }
        }
        let mat = gltf.json.materials[material_id]
        mat = getTextures(mat)
        if (mat.pbrMetallicRoughness) {
            mat.pbrMetallicRoughness = getTextures(mat.pbrMetallicRoughness)
        }
        return mat
    }

    function getAccessor(accessor_id, name) {
        if (accessor_id == undefined) return false
        let acc = gltf.json.accessors[accessor_id]
        let bufferView = gltf.json.bufferViews[acc.bufferView]
        let buffer = gltf.buffers[bufferView.buffer]

        // Get number of components based on type
        let numComponents = 0
        switch(acc.type) {
            case 'SCALAR': numComponents = 1; break;
            case 'VEC2': numComponents = 2; break;
            case 'VEC3': numComponents = 3; break;
            case 'VEC4': numComponents = 4; break;
            case 'MAT2': numComponents = 4; break;
            case 'MAT3': numComponents = 9; break;
            case 'MAT4': numComponents = 16; break;
        }

        // Create typed array based on componentType
        let byteOffset = (bufferView.byteOffset || 0) + (acc.byteOffset || 0) + (buffer.byteOffset || 0)
        let ab = buffer.arrayBuffer
        switch(acc.componentType) {
            case 5120: // BYTE
                acc.data = new Int8Array(ab, byteOffset, acc.count * numComponents)
                break
            case 5121: // UNSIGNED_BYTE
                acc.data = new Uint8Array(ab, byteOffset, acc.count * numComponents)
                break
            case 5122: // SHORT
                acc.data = new Int16Array(ab, byteOffset, acc.count * numComponents)
                break
            case 5123: // UNSIGNED_SHORT
                acc.data = new Uint16Array(ab, byteOffset, acc.count * numComponents)
                break
            case 5125: // UNSIGNED_INT
                acc.data = new Uint32Array(ab, byteOffset, acc.count * numComponents)
                break
            case 5126: // FLOAT
                acc.data = new Float32Array(ab, byteOffset, acc.count * numComponents)
                break
        }
        return acc.data
    }

    // Parse all nodes first
    if (gltf.json.nodes) {
        for (let i = 0; i < gltf.json.nodes.length; i++) {
            const nodeData = gltf.json.nodes[i]
            const joint = new Joint(nodeData.name || `node_${i}`)

            // Set transform
            if (nodeData.matrix) {
                joint.setMatrix(new Float32Array(nodeData.matrix))
            } else {
                if (nodeData.translation) {
                    vec3.set(joint.position, nodeData.translation[0], nodeData.translation[1], nodeData.translation[2])
                }
                if (nodeData.rotation) {
                    quat.set(joint.rotation, nodeData.rotation[0], nodeData.rotation[1], nodeData.rotation[2], nodeData.rotation[3])
                }
                if (nodeData.scale) {
                    vec3.set(joint.scale, nodeData.scale[0], nodeData.scale[1], nodeData.scale[2])
                }
            }

            joint.saveBindPose()
            joint.nodeIndex = i
            nodes.push(joint)
        }

        // Build parent-child relationships
        for (let i = 0; i < gltf.json.nodes.length; i++) {
            const nodeData = gltf.json.nodes[i]
            if (nodeData.children) {
                for (const childIndex of nodeData.children) {
                    nodes[i].addChild(nodes[childIndex])
                }
            }
        }
    }

    // Parse skins
    if (gltf.json.skins) {
        for (const skinData of gltf.json.skins) {
            const skin = new Skin(engine)

            // Get joint nodes
            const jointNodes = skinData.joints.map(jointIndex => nodes[jointIndex])

            // Get inverse bind matrices
            const inverseBindMatrices = getAccessor(skinData.inverseBindMatrices, 'inverseBindMatrices')

            // Find root node (skeleton)
            let rootNode = skinData.skeleton !== undefined ? nodes[skinData.skeleton] : jointNodes[0]
            // Find the topmost parent
            while (rootNode.parent) {
                rootNode = rootNode.parent
            }

            skin.init(jointNodes, inverseBindMatrices, rootNode)
            skins.push(skin)
        }
    }

    // Parse animations
    if (gltf.json.animations) {
        for (const animData of gltf.json.animations) {
            const animation = {
                name: animData.name || 'default',
                duration: 0,
                channels: []
            }

            // Parse samplers
            const samplers = animData.samplers.map(samplerData => {
                const input = getAccessor(samplerData.input, 'animation_input')  // times
                const output = getAccessor(samplerData.output, 'animation_output') // values

                // Update animation duration
                if (input && input.length > 0) {
                    animation.duration = Math.max(animation.duration, input[input.length - 1])
                }

                return {
                    input: input,
                    output: output,
                    interpolation: samplerData.interpolation || 'LINEAR'
                }
            })

            // Parse channels
            for (const channelData of animData.channels) {
                const targetNode = nodes[channelData.target.node]
                const sampler = samplers[channelData.sampler]

                animation.channels.push({
                    target: targetNode,
                    path: channelData.target.path,
                    sampler: sampler
                })
            }

            animations.push(animation)

            // Add animation to relevant skins
            for (const skin of skins) {
                skin.addAnimation(animation.name, animation)
            }
        }
    }

    // Parse meshes
    for (let mi = 0; mi < gltf.json.meshes.length; mi++) {
        const mesh = gltf.json.meshes[mi]

        // Find which node uses this mesh
        let meshNodeIndex = null
        let skinIndex = null
        if (gltf.json.nodes) {
            for (let ni = 0; ni < gltf.json.nodes.length; ni++) {
                if (gltf.json.nodes[ni].mesh === mi) {
                    meshNodeIndex = ni
                    skinIndex = gltf.json.nodes[ni].skin
                    break
                }
            }
        }

        // Handle all primitives in the mesh (some GLTF files have multiple primitives per mesh)
        for (let pi = 0; pi < mesh.primitives.length; pi++) {
            const primitive = mesh.primitives[pi]
            const attributes = primitive.attributes;

            // Generate unique name: meshName_primitiveIndex or mesh_meshIndex_primitiveIndex
            const baseName = mesh.name || `mesh_${mi}`
            const meshName = mesh.primitives.length > 1 ? `${baseName}_${pi}` : baseName

            // Collect mesh attributes
            let attrs = {
                position: getAccessor(attributes.POSITION, 'position'),
                normal: getAccessor(attributes.NORMAL, 'normal'),
                uv: getAccessor(attributes.TEXCOORD_0, 'uv'),
                indices: getAccessor(primitive.indices, 'indices'),
                weights: getAccessor(attributes.WEIGHTS_0, 'weights'),
                joints: getAccessor(attributes.JOINTS_0, 'joints')
            }

            // Apply triangle expansion if requested (helps eliminate gaps between triangles)
            if (options.expandTriangles > 0) {
                attrs = expandTriangles(attrs, options.expandTriangles)
            }

            meshes[meshName] = {
                attributes: attrs,
                material: getMaterial(primitive.material),
                scale: options.scale,
                rotation: options.rotation,
                skinIndex: skinIndex,
                nodeIndex: meshNodeIndex
            };
        }
    }

    return { meshes, nodes, skins, animations }
}

async function loadGltf(engine, url, options = {}) {
    options = {
        flipY: false,
        ...options
    }
    const data = await loadGltfData(engine, url, options)
    const { meshes: mdata, skins, animations, nodes } = data
    const meshes = {}

    for (const name in mdata) {
        const mesh = mdata[name]
        const geometry = new Geometry(engine, mesh.attributes)
        const pbr = mesh.material.pbrMetallicRoughness || {}

        // Albedo/Base Color texture
        let albedo
        if (pbr.baseColorTexture) {
            const tex = pbr.baseColorTexture
            albedo = await Texture.fromImage(engine, tex.image, {
                srgb: true,
                flipY: options.flipY,
                addressModeU: tex.wrapS,
                addressModeV: tex.wrapT
            })
            if (tex.uri) albedo.sourceUrl = tex.uri
        } else {
            // Use baseColorFactor if available, otherwise white
            const bcf = pbr.baseColorFactor || [1, 1, 1, 1]
            albedo = await Texture.fromRGBA(engine, bcf[0], bcf[1], bcf[2], bcf[3])
        }

        // Normal map
        let normal
        if (mesh.material.normalTexture) {
            const tex = mesh.material.normalTexture
            normal = await Texture.fromImage(engine, tex.image, {
                srgb: false,
                flipY: options.flipY,
                addressModeU: tex.wrapS,
                addressModeV: tex.wrapT
            })
            if (tex.uri) normal.sourceUrl = tex.uri
        } else {
            normal = await Texture.fromColor(engine, "#8080FF")
        }

        // Metallic/Roughness texture
        // Format: R=ambient occlusion (if packed), G=roughness, B=metallic
        let rm
        if (pbr.metallicRoughnessTexture) {
            const tex = pbr.metallicRoughnessTexture
            rm = await Texture.fromImage(engine, tex.image, {
                srgb: false,
                flipY: options.flipY,
                addressModeU: tex.wrapS,
                addressModeV: tex.wrapT
            })
            if (tex.uri) rm.sourceUrl = tex.uri
        } else {
            // Use material factors or defaults
            // Default: roughness 0.9, metallic 0.0
            const roughness = pbr.roughnessFactor !== undefined ? pbr.roughnessFactor : 0.9
            const metallic = pbr.metallicFactor !== undefined ? pbr.metallicFactor : 0.0
            // R=1 (no AO in this channel), G=roughness, B=metallic
            rm = await Texture.fromRGBA(engine, 1.0, roughness, metallic, 1.0)
        }

        // Ambient Occlusion texture
        let ambient
        if (mesh.material.occlusionTexture) {
            const tex = mesh.material.occlusionTexture
            ambient = await Texture.fromImage(engine, tex.image, {
                srgb: false,
                flipY: options.flipY,
                addressModeU: tex.wrapS,
                addressModeV: tex.wrapT
            })
            if (tex.uri) ambient.sourceUrl = tex.uri
        } else {
            // Default: no occlusion (white = full light)
            ambient = await Texture.fromColor(engine, "#FFFFFF")
        }

        // Emission texture
        let emission
        if (mesh.material.emissiveTexture) {
            const tex = mesh.material.emissiveTexture
            emission = await Texture.fromImage(engine, tex.image, {
                srgb: true,
                flipY: options.flipY,
                addressModeU: tex.wrapS,
                addressModeV: tex.wrapT
            })
            if (tex.uri) emission.sourceUrl = tex.uri
        } else {
            // Use emissiveFactor if available, otherwise black (no emission)
            const ef = mesh.material.emissiveFactor || [0, 0, 0]
            emission = await Texture.fromRGBA(engine, ef[0], ef[1], ef[2], 1.0)
        }

        // Get material name from GLTF or generate from index
        const materialName = mesh.material.name || null
        const material = new Material([albedo, normal, ambient, rm, emission], {}, materialName)

        // Check GLTF alphaMode for transparency handling
        // "MASK" = alpha cutout (use alpha hashing for deferred)
        // "BLEND" = true alpha blending (requires forward transparent pass)
        // "OPAQUE" = fully opaque (default)
        const alphaMode = mesh.material.alphaMode || 'OPAQUE'
        if (alphaMode === 'MASK') {
            // Alpha cutout: use alpha hashing in deferred renderer
            material.alphaHash = true
            // GLTF alphaCutoff defaults to 0.5, use as scale factor
            const alphaCutoff = mesh.material.alphaCutoff ?? 0.5
            material.alphaHashScale = 1.0 / Math.max(alphaCutoff, 0.01)
        } else if (alphaMode === 'BLEND') {
            // True transparency: render in forward transparent pass
            material.transparent = true
            // Get base opacity from baseColorFactor alpha if available
            const bcf = pbr.baseColorFactor || [1, 1, 1, 1]
            material.opacity = bcf[3]
        }

        // Handle KHR_materials_transmission extension (for glass-like materials)
        if (mesh.material.extensions?.KHR_materials_transmission) {
            material.transparent = true
            const transmission = mesh.material.extensions.KHR_materials_transmission
            material.opacity = 1.0 - (transmission.transmissionFactor ?? 0)
        }

        // Create mesh with name from GLTF (name is the key from mesh data)
        const nmesh = new Mesh(geometry, material, name)

        // Attach skin if this mesh has one
        if (mesh.skinIndex !== undefined && mesh.skinIndex !== null && skins[mesh.skinIndex]) {
            nmesh.skin = skins[mesh.skinIndex]
            nmesh.hasSkin = true
        }

        // Preserve node index for scene placement
        nmesh.nodeIndex = mesh.nodeIndex

        meshes[name] = nmesh
    }

    return { meshes, skins, animations, nodes }
}

// Simpler version that returns just meshes for backward compatibility
async function loadGltfMeshes(engine, url, options = {}) {
    const result = await loadGltf(engine, url, options)
    return result.meshes
}

export { loadGltf, loadGltfMeshes, loadGltfData }
