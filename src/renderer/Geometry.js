import { calculateBoundingSphere } from "./utils/BoundingSphere.js"

var _UID = 30001

class Geometry {

    constructor(engine, attributes) {
        this.uid = _UID++
        this.engine = engine
        const { device } = engine

        // Bounding sphere (lazy calculated)
        this._bsphere = null

        this.vertexBufferLayout = {
            arrayStride: 80, // 20 floats * 4 bytes each
            attributes: [
                { format: "float32x3", offset: 0, shaderLocation: 0 },  // position
                { format: "float32x2", offset: 12, shaderLocation: 1 }, // uv
                { format: "float32x3", offset: 20, shaderLocation: 2 }, // normal
                { format: "float32x4", offset: 32, shaderLocation: 3 }, // color
                { format: "float32x4", offset: 48, shaderLocation: 4 }, // weights
                { format: "uint32x4", offset: 64, shaderLocation: 5 }, // joints
            ],
            stepMode: 'vertex'
        };

        // Instance buffer layout for model matrix + sprite data
        this.instanceBufferLayout = {
            arrayStride: 112, // 28 floats * 4 bytes each
            stepMode: 'instance',
            attributes: [
                { format: "float32x4", offset: 0, shaderLocation: 6 },   // matrix column 0
                { format: "float32x4", offset: 16, shaderLocation: 7 },  // matrix column 1
                { format: "float32x4", offset: 32, shaderLocation: 8 },  // matrix column 2
                { format: "float32x4", offset: 48, shaderLocation: 9 },  // matrix column 3
                { format: "float32x4", offset: 64, shaderLocation: 10 }, // position + radius
                { format: "float32x4", offset: 80, shaderLocation: 11 }, // uvTransform (offset.xy, scale.xy)
                { format: "float32x4", offset: 96, shaderLocation: 12 }, // color (r, g, b, a)
            ]
        };

        const vertexCount = attributes.position.length / 3

        if (attributes.indices == false) {
            // Generate indices for non-indexed geometry
            const positions = attributes.position;
            const vertexCount = positions.length / 3; // 3 components per vertex
            const indices = new Uint32Array(vertexCount);
            for (let i = 0; i < vertexCount; i++) {
                indices[i] = i;
            }
            attributes.indices = indices
        }

        if (attributes.normal == false) {
            // Generate flat normals for non-normal geometry
            const positions = attributes.position;
            const indices = attributes.indices;
            const normals = new Float32Array(positions.length);

            // Calculate normals for each triangle
            for (let i = 0; i < indices.length; i += 3) {
                const i0 = indices[i] * 3;
                const i1 = indices[i + 1] * 3;
                const i2 = indices[i + 2] * 3;

                // Get vertices of triangle
                const v0 = [positions[i0], positions[i0 + 1], positions[i0 + 2]];
                const v1 = [positions[i1], positions[i1 + 1], positions[i1 + 2]];
                const v2 = [positions[i2], positions[i2 + 1], positions[i2 + 2]];

                // Calculate vectors for cross product
                const vec1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                const vec2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

                // Calculate cross product
                const normal = [
                    vec1[1] * vec2[2] - vec1[2] * vec2[1],
                    vec1[2] * vec2[0] - vec1[0] * vec2[2],
                    vec1[0] * vec2[1] - vec1[1] * vec2[0]
                ];

                // Normalize
                const length = Math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
                normal[0] /= length;
                normal[1] /= length;
                normal[2] /= length;

                // Assign same normal to all vertices of triangle
                normals[i0] = normal[0];
                normals[i0 + 1] = normal[1];
                normals[i0 + 2] = normal[2];
                normals[i1] = normal[0];
                normals[i1 + 1] = normal[1];
                normals[i1 + 2] = normal[2];
                normals[i2] = normal[0];
                normals[i2 + 1] = normal[1];
                normals[i2 + 2] = normal[2];
            }

            attributes.normal = normals
        }


        const vertexArray = new Float32Array(vertexCount * 20);
        this.vertexCount = vertexCount;
        const vertexArrayUint32 = new Uint32Array(vertexArray.buffer);
        const vertexBuffer = device.createBuffer({
            size: vertexArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        })

        // Create instance buffer (start small, grows dynamically)
        const initialMaxInstances = 16;
        const instanceBuffer = device.createBuffer({
            size: 112 * initialMaxInstances, // 28 floats per instance
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.instanceBuffer = instanceBuffer;
        this.maxInstances = initialMaxInstances;
        this.instanceCount = 0;
        this.instanceData = new Float32Array(28 * initialMaxInstances);
        this._instanceDataDirty = true

        const indexArray = new Uint32Array(attributes.indices.length);        
        const indexBuffer = device.createBuffer({
            size: indexArray.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        })

        this.attributes = attributes
        this.vertexArray = vertexArray
        this.vertexArrayUint32 = vertexArrayUint32
        this.indexArray = indexArray

        this.vertexBuffer = vertexBuffer
        this.indexBuffer = indexBuffer

        this.updateVertexBuffer()
    }


    /**
     * Grow instance buffer capacity by doubling it
     * @param {number} minCapacity - Minimum required capacity (optional)
     */
    growInstanceBuffer(minCapacity = 0) {
        const { device } = this.engine;

        // Calculate new size: double current, or enough for minCapacity
        let newMaxInstances = this.maxInstances * 2;
        while (newMaxInstances < minCapacity) {
            newMaxInstances *= 2;
        }

        // Create new larger buffer
        const newInstanceBuffer = device.createBuffer({
            size: 112 * newMaxInstances, // 28 floats per instance
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        // Create new larger data array and copy existing data
        const newInstanceData = new Float32Array(28 * newMaxInstances);
        newInstanceData.set(this.instanceData);

        // Destroy old buffer
        this.instanceBuffer.destroy();

        // Update references
        this.instanceBuffer = newInstanceBuffer;
        this.instanceData = newInstanceData;
        this.maxInstances = newMaxInstances;
        this._instanceDataDirty = true;
    }

    addInstance(position, radius = 1, uvTransform = [0, 0, 1, 1], color = [1, 1, 1, 1]) {
        if (this.instanceCount >= this.maxInstances) {
            this.growInstanceBuffer();
        }

        // Create a 4x4 transform matrix for this instance
        const matrix = mat4.create();
        mat4.translate(matrix, matrix, position);

        // Add the matrix to instance data at the next available slot
        const offset = this.instanceCount * 28;
        this.instanceData.set(matrix, offset);
        // Position + radius (would normally be set from bsphere)
        this.instanceData[offset + 16] = position[0];
        this.instanceData[offset + 17] = position[1];
        this.instanceData[offset + 18] = position[2];
        this.instanceData[offset + 19] = radius;
        // UV transform
        this.instanceData[offset + 20] = uvTransform[0];
        this.instanceData[offset + 21] = uvTransform[1];
        this.instanceData[offset + 22] = uvTransform[2];
        this.instanceData[offset + 23] = uvTransform[3];
        // Color
        this.instanceData[offset + 24] = color[0];
        this.instanceData[offset + 25] = color[1];
        this.instanceData[offset + 26] = color[2];
        this.instanceData[offset + 27] = color[3];

        this.instanceCount++;
        this._instanceDataDirty = true
    }

    updateAllInstances(matrices) {
        const { device } = this.engine;
        this.instanceCount = matrices.length;
        if (this.instanceCount > this.maxInstances) {
            this.growInstanceBuffer(this.instanceCount);
        }

        // Copy all matrices into instance data (28 floats per instance)
        for (let i = 0; i < matrices.length; i++) {
            const offset = i * 28;
            this.instanceData.set(matrices[i], offset);
            // Set default UV transform and color if not provided
            this.instanceData[offset + 20] = 0;  // uvOffset.x
            this.instanceData[offset + 21] = 0;  // uvOffset.y
            this.instanceData[offset + 22] = 1;  // uvScale.x
            this.instanceData[offset + 23] = 1;  // uvScale.y
            this.instanceData[offset + 24] = 1;  // color.r
            this.instanceData[offset + 25] = 1;  // color.g
            this.instanceData[offset + 26] = 1;  // color.b
            this.instanceData[offset + 27] = 1;  // color.a
        }

        this._instanceDataDirty = true
    }

    updateInstance(index, matrix) {
        this.instanceData.set(matrix, index * 28);
        this._instanceDataDirty = true
    }

    writeInstanceBuffer() {
        const { device } = this.engine;

        device.queue.writeBuffer(this.instanceBuffer, 0, this.instanceData);
    }

    updateVertexBuffer() {
        const { device } = this.engine
        const { attributes, vertexArray, vertexArrayUint32 } = this
        // Interleave vertex attributes into a single array
        for (let i = 0; i < this.vertexCount; i++) {
            const vertexOffset = i * 20; // 20 floats per vertex
            
            // Position (xyz)
            vertexArray[vertexOffset] = attributes.position[i * 3];
            vertexArray[vertexOffset + 1] = attributes.position[i * 3 + 1];
            vertexArray[vertexOffset + 2] = attributes.position[i * 3 + 2];

            // UV (xy) 
            vertexArray[vertexOffset + 3] = attributes.uv ? attributes.uv[i * 2] : 0;
            vertexArray[vertexOffset + 4] = attributes.uv ? attributes.uv[i * 2 + 1] : 0;

            // Normal (xyz)
            vertexArray[vertexOffset + 5] = attributes.normal[i * 3]
            vertexArray[vertexOffset + 6] = attributes.normal[i * 3 + 1]
            vertexArray[vertexOffset + 7] = attributes.normal[i * 3 + 2]

            // Color (rgba)
            vertexArray[vertexOffset + 8] = attributes.color ? attributes.color[i * 4] : 1;
            vertexArray[vertexOffset + 9] = attributes.color ? attributes.color[i * 4 + 1] : 1;
            vertexArray[vertexOffset + 10] = attributes.color ? attributes.color[i * 4 + 2] : 1;
            vertexArray[vertexOffset + 11] = attributes.color ? attributes.color[i * 4 + 3] : 1;

            // Weights (xyzw)
            vertexArray[vertexOffset + 12] = attributes.weights ? attributes.weights[i * 4] : 0;
            vertexArray[vertexOffset + 13] = attributes.weights ? attributes.weights[i * 4 + 1] : 0;
            vertexArray[vertexOffset + 14] = attributes.weights ? attributes.weights[i * 4 + 2] : 0;
            vertexArray[vertexOffset + 15] = attributes.weights ? attributes.weights[i * 4 + 3] : 0;

            // Joints (xyzw) - using Float32Array view to write uint32 values
            const jointsOffset = vertexOffset + 16;
            vertexArrayUint32[jointsOffset] = attributes.joints ? attributes.joints[i * 4] : 0;
            vertexArrayUint32[jointsOffset + 1] = attributes.joints ? attributes.joints[i * 4 + 1] : 0;
            vertexArrayUint32[jointsOffset + 2] = attributes.joints ? attributes.joints[i * 4 + 2] : 0;
            vertexArrayUint32[jointsOffset + 3] = attributes.joints ? attributes.joints[i * 4 + 3] : 0;
        }
        
        // Copy indices
        for (let i = 0; i < attributes.indices.length; i++) {
            this.indexArray[i] = attributes.indices[i];
        }

        device.queue.writeBuffer(this.vertexBuffer, 0, this.vertexArray)
        device.queue.writeBuffer(this.indexBuffer, 0, this.indexArray)
    }

    update() {
        if (this._instanceDataDirty) {
            this.writeInstanceBuffer()
            this._instanceDataDirty = false
        }
    }

    /**
     * Get bounding sphere for this geometry (lazy calculated)
     * @returns {{ center: [number, number, number], radius: number }}
     */
    getBoundingSphere() {
        if (!this._bsphere) {
            this._bsphere = calculateBoundingSphere(this.attributes.position)
        }
        return this._bsphere
    }

    /**
     * Invalidate cached bounding sphere (call after modifying positions)
     */
    invalidateBoundingSphere() {
        this._bsphere = null
    }

    static createCullingBuffers(device, maxInstances) {
        // Input buffer containing instance data (position + radius)
        const instanceDataBuffer = device.createBuffer({
            size: maxInstances * 16, // vec3 position + float radius
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Output buffer containing active instance indices and count
        const culledInstancesBuffer = device.createBuffer({
            size: (maxInstances + 1) * 4, // indices + count at start
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Uniform buffer for camera frustum planes
        const frustumBuffer = device.createBuffer({
            size: 96, // 6 planes * vec4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
                        struct Instance {
                            position: vec3f,
                            radius: f32
                        }

                        struct FrustumPlanes {
                            planes: array<vec4f, 6>
                        }

                        @group(0) @binding(0) var<storage, read> instances: array<Instance>;
                        @group(0) @binding(1) var<storage, read_write> culledInstances: array<u32>;
                        @group(0) @binding(2) var<uniform> frustum: FrustumPlanes;

                        fn sphereAgainstPlane(center: vec3f, radius: f32, plane: vec4f) -> bool {
                            let dist = dot(vec4f(center, 1.0), plane);
                            return dist > -radius;
                        }

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) global_id: vec3u) {
                            let index = global_id.x;
                            if (index >= arrayLength(&instances)) {
                                return;
                            }

                            let instance = instances[index];
                            
                            // Test against all frustum planes
                            var visible = true;
                            for (var i = 0u; i < 6u; i++) {
                                if (!sphereAgainstPlane(instance.position, instance.radius, frustum.planes[i])) {
                                    visible = false;
                                    break;
                                }
                            }

                            if (visible) {
                                let oldCount = atomicAdd(&culledInstances[0], 1);
                                culledInstances[oldCount + 1] = index;
                            }
                        }
                    `
                }),
                entryPoint: 'main'
            }
        });

        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: instanceDataBuffer }
                },
                {
                    binding: 1, 
                    resource: { buffer: culledInstancesBuffer }
                },
                {
                    binding: 2,
                    resource: { buffer: frustumBuffer }
                }
            ]
        });

        return {
            pipeline: computePipeline,
            bindGroup: bindGroup,
            instanceDataBuffer,
            culledInstancesBuffer,
            frustumBuffer
        };
    }

    static cullInstances(engine, commandEncoder, cullingData, instances, frustumPlanes) {
        const { device } = engine;
        const { pipeline, bindGroup, instanceDataBuffer, culledInstancesBuffer, frustumBuffer } = cullingData;

        // Write instance data
        device.queue.writeBuffer(instanceDataBuffer, 0, instances);
        
        // Write frustum planes
        device.queue.writeBuffer(frustumBuffer, 0, frustumPlanes);

        // Clear count to 0
        const zeros = new Uint32Array(1);
        device.queue.writeBuffer(culledInstancesBuffer, 0, zeros);

        // Dispatch compute shader
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(pipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(Math.ceil(instances.length / 64));
        computePass.end();

        return culledInstancesBuffer;
    }

    static cube(engine) {
        return Geometry.box(engine, 1, 1, 1, 1)
    }

    static box(engine, width = 1, height = 1, depth = 1, uvSize = 1) {
        const w = width / 2
        const h = height / 2 
        const d = depth / 2

        // UV scale based on dimensions and uvSize
        const uw = width / uvSize
        const uh = height / uvSize
        const ud = depth / uvSize

        return new Geometry(engine, {
            position: new Float32Array([
                // Front face
                -w, -h,  d,
                 w, -h,  d,
                 w,  h,  d,
                -w,  h,  d,
                // Back face
                -w, -h, -d,
                -w,  h, -d,
                 w,  h, -d,
                 w, -h, -d,
                // Top face
                -w,  h, -d,
                -w,  h,  d,
                 w,  h,  d,
                 w,  h, -d,
                // Bottom face
                -w, -h, -d,
                 w, -h, -d,
                 w, -h,  d,
                -w, -h,  d,
                // Right face
                 w, -h, -d,
                 w,  h, -d,
                 w,  h,  d,
                 w, -h,  d,
                // Left face
                -w, -h, -d,
                -w, -h,  d,
                -w,  h,  d,
                -w,  h, -d,
            ]),
            uv: new Float32Array([
                // Front face
                0, 0,
                uw, 0,
                uw, uh,
                0, uh,
                // Back face
                0, 0,
                0, uh,
                uw, uh,
                uw, 0,
                // Top face
                0, 0,
                0, ud,
                uw, ud,
                uw, 0,
                // Bottom face
                0, 0,
                uw, 0,
                uw, ud,
                0, ud,
                // Right face
                0, 0,
                0, uh,
                ud, uh,
                ud, 0,
                // Left face
                0, 0,
                ud, 0,
                ud, uh,
                0, uh,
            ]),
            normal: new Float32Array([
                // Front face
                0,  0,  1,
                0,  0,  1,
                0,  0,  1,
                0,  0,  1,
                // Back face
                0,  0, -1,
                0,  0, -1,
                0,  0, -1,
                0,  0, -1,
                // Top face
                0,  1,  0,
                0,  1,  0,
                0,  1,  0,
                0,  1,  0,
                // Bottom face
                0, -1,  0,
                0, -1,  0,
                0, -1,  0,
                0, -1,  0,
                // Right face
                1,  0,  0,
                1,  0,  0,
                1,  0,  0,
                1,  0,  0,
                // Left face
                -1,  0,  0,
                -1,  0,  0,
                -1,  0,  0,
                -1,  0,  0,
            ]),
            indices: new Uint32Array([
                0,  1,  2,  2,  3,  0,  // Front face
                4,  5,  6,  6,  7,  4,  // Back face
                8,  9,  10, 10, 11, 8,  // Top face
                12, 13, 14, 14, 15, 12, // Bottom face
                16, 17, 18, 18, 19, 16, // Right face
                20, 21, 22, 22, 23, 20  // Left face
            ])
        })
    }
    static sphere(engine, radius = 1, widthSegments = 32, heightSegments = 16) {
        const positions = []
        const normals = []
        const uvs = []
        const indices = []

        // Generate vertices
        for (let y = 0; y <= heightSegments; y++) {
            const v = y / heightSegments
            const phi = v * Math.PI

            for (let x = 0; x <= widthSegments; x++) {
                const u = x / widthSegments
                const theta = u * Math.PI * 2

                // Calculate vertex position
                const px = -radius * Math.cos(theta) * Math.sin(phi)
                const py = radius * Math.cos(phi)
                const pz = radius * Math.sin(theta) * Math.sin(phi)

                positions.push(px, py, pz)

                // Normal is just the normalized position for a sphere
                const length = Math.sqrt(px * px + py * py + pz * pz)
                normals.push(px / length, py / length, pz / length)

                // UV coordinates
                uvs.push(u, 1 - v)
            }
        }

        // Generate indices
        for (let y = 0; y < heightSegments; y++) {
            for (let x = 0; x < widthSegments; x++) {
                const a = y * (widthSegments + 1) + x
                const b = a + 1
                const c = a + widthSegments + 1
                const d = c + 1

                // Generate two triangles for each quad (counter-clockwise winding)
                indices.push(a, d, b)
                indices.push(a, c, d)
            }
        }

        return new Geometry(engine, {
            position: new Float32Array(positions),
            normal: new Float32Array(normals),
            uv: new Float32Array(uvs),
            indices: new Uint32Array(indices)
        })
    }

    static quad(engine) {
        return new Geometry(engine, {
            position: [
                -1, -1,  0,
                 1, -1,  0,
                 1,  1,  0,
                -1,  1,  0,
            ],
            uv: [
                0, 0,
                1, 0,
                1, 1,
                0, 1,
            ],
            normal: [
                // Normals up
                0,  1,  0,
                0,  1,  0,
                0,  1,  0,
                0,  1,  0,
            ],
            index: [
                0, 1, 2,
                2, 3, 0
            ]

        })
    }

    /**
     * Create a quad geometry for billboard/sprite rendering
     * @param {Engine} engine - Engine instance
     * @param {string} pivot - Pivot mode: 'center', 'bottom', or 'horizontal'
     * @returns {Geometry} Billboard quad geometry
     */
    static billboardQuad(engine, pivot = 'center') {
        let position, normal

        if (pivot === 'center') {
            // Center pivot: quad centered at origin, facing +Z
            position = new Float32Array([
                -0.5, -0.5, 0,
                 0.5, -0.5, 0,
                 0.5,  0.5, 0,
                -0.5,  0.5, 0,
            ])
            normal = new Float32Array([
                0, 0, 1,
                0, 0, 1,
                0, 0, 1,
                0, 0, 1,
            ])
        } else if (pivot === 'bottom') {
            // Bottom pivot: quad with bottom edge at origin, facing +Z
            position = new Float32Array([
                -0.5, 0, 0,
                 0.5, 0, 0,
                 0.5, 1, 0,
                -0.5, 1, 0,
            ])
            normal = new Float32Array([
                0, 0, 1,
                0, 0, 1,
                0, 0, 1,
                0, 0, 1,
            ])
        } else if (pivot === 'horizontal') {
            // Horizontal pivot: quad flat on XZ plane (ground decal)
            // Front face points UP (+Y) with CCW winding when viewed from above
            position = new Float32Array([
                -0.5, 0, -0.5,  // 0: back left
                 0.5, 0, -0.5,  // 1: back right
                 0.5, 0,  0.5,  // 2: front right
                -0.5, 0,  0.5,  // 3: front left
            ])
            normal = new Float32Array([
                0, 1, 0,
                0, 1, 0,
                0, 1, 0,
                0, 1, 0,
            ])
            // Use reversed winding for horizontal (CCW from above)
            return new Geometry(engine, {
                position,
                uv: new Float32Array([
                    0, 1,  // back left -> top-left of texture
                    1, 1,  // back right -> top-right
                    1, 0,  // front right -> bottom-right
                    0, 0,  // front left -> bottom-left
                ]),
                normal,
                indices: new Uint32Array([
                    0, 3, 2,  // CCW: back-left, front-left, front-right
                    2, 1, 0   // CCW: front-right, back-right, back-left
                ])
            })
        } else {
            // Default to center pivot
            return Geometry.billboardQuad(engine, 'center')
        }

        return new Geometry(engine, {
            position,
            uv: new Float32Array([
                0, 0,  // Bottom-left: sample from v=0 (bottom of texture)
                1, 0,  // Bottom-right
                1, 1,  // Top-right: sample from v=1 (top of texture)
                0, 1,  // Top-left
            ]),
            normal,
            indices: new Uint32Array([
                0, 1, 2,
                2, 3, 0
            ])
        })
    }

    simplify(angleThreshold = 0.1) {
        // Early return if no UV coordinates or not enough vertices
        if (!this.attributes.uv || this.attributes.indices.length < 6) {
            return this;
        }

        const positions = this.attributes.position;
        const normals = this.attributes.normal;
        const uvs = this.attributes.uv;
        const indices = this.attributes.indices;
        const colors = this.attributes.color;
        const joints = this.attributes.joints;
        const weights = this.attributes.weights;

        let newPositions = [];
        let newNormals = [];
        let newUVs = [];
        let newIndices = [];
        let newColors = colors ? [] : null;
        let newJoints = joints ? [] : null;
        let newWeights = weights ? [] : null;
        
        // Process triangles in pairs
        for (let i = 0; i < indices.length; i += 6) {
            if (i + 5 >= indices.length) {
                // Add remaining triangle if we can't make a pair
                for (let j = 0; j < 3; j++) {
                    const idx = indices[i + j];
                    newPositions.push(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
                    newNormals.push(normals[idx * 3], normals[idx * 3 + 1], normals[idx * 3 + 2]);
                    newUVs.push(uvs[idx * 2], uvs[idx * 2 + 1]);
                    if (colors) {
                        newColors.push(colors[idx * 4], colors[idx * 4 + 1], colors[idx * 4 + 2], colors[idx * 4 + 3]);
                    }
                    if (joints) {
                        newJoints.push(joints[idx * 4], joints[idx * 4 + 1], joints[idx * 4 + 2], joints[idx * 4 + 3]);
                    }
                    if (weights) {
                        newWeights.push(weights[idx * 4], weights[idx * 4 + 1], weights[idx * 4 + 2], weights[idx * 4 + 3]);
                    }
                }
                newIndices.push(newPositions.length / 3 - 3, newPositions.length / 3 - 2, newPositions.length / 3 - 1);
                continue;
            }

            // Get the two triangles
            const tri1 = [indices[i], indices[i + 1], indices[i + 2]];
            const tri2 = [indices[i + 3], indices[i + 4], indices[i + 5]];

            // Calculate normals of both triangles
            const normal1 = new Vector3(
                normals[tri1[0] * 3], normals[tri1[0] * 3 + 1], normals[tri1[0] * 3 + 2]
            );
            const normal2 = new Vector3(
                normals[tri2[0] * 3], normals[tri2[0] * 3 + 1], normals[tri2[0] * 3 + 2]
            );

            // Check if triangles are nearly coplanar
            const angle = Math.acos(normal1.dot(normal2));
            
            if (angle < angleThreshold) {
                // Find shared vertices
                const sharedVertices = tri1.filter(v => tri2.includes(v));
                
                if (sharedVertices.length === 2) {
                    // Get unique vertices
                    const uniqueFromTri1 = tri1.find(v => !tri2.includes(v));
                    const uniqueFromTri2 = tri2.find(v => !tri1.includes(v));
                    
                    // Create new quad from the four points
                    const quadIndices = [uniqueFromTri1, ...sharedVertices, uniqueFromTri2];
                    
                    // Add vertices to new arrays
                    for (const idx of quadIndices) {
                        newPositions.push(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
                        newNormals.push(normals[idx * 3], normals[idx * 3 + 1], normals[idx * 3 + 2]);
                        newUVs.push(uvs[idx * 2], uvs[idx * 2 + 1]);
                        if (colors) {
                            newColors.push(colors[idx * 4], colors[idx * 4 + 1], colors[idx * 4 + 2], colors[idx * 4 + 3]);
                        }
                        if (joints) {
                            newJoints.push(joints[idx * 4], joints[idx * 4 + 1], joints[idx * 4 + 2], joints[idx * 4 + 3]);
                        }
                        if (weights) {
                            newWeights.push(weights[idx * 4], weights[idx * 4 + 1], weights[idx * 4 + 2], weights[idx * 4 + 3]);
                        }
                    }
                    
                    // Add indices for the simplified quad (as two triangles)
                    const baseIndex = newPositions.length / 3 - 4;
                    newIndices.push(
                        baseIndex, baseIndex + 1, baseIndex + 2,
                        baseIndex + 2, baseIndex + 3, baseIndex
                    );
                    continue;
                }
            }
            
            // If we can't simplify, add original triangles
            for (let j = 0; j < 6; j++) {
                const idx = indices[i + j];
                newPositions.push(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
                newNormals.push(normals[idx * 3], normals[idx * 3 + 1], normals[idx * 3 + 2]);
                newUVs.push(uvs[idx * 2], uvs[idx * 2 + 1]);
                if (colors) {
                    newColors.push(colors[idx * 4], colors[idx * 4 + 1], colors[idx * 4 + 2], colors[idx * 4 + 3]);
                }
                if (joints) {
                    newJoints.push(joints[idx * 4], joints[idx * 4 + 1], joints[idx * 4 + 2], joints[idx * 4 + 3]);
                }
                if (weights) {
                    newWeights.push(weights[idx * 4], weights[idx * 4 + 1], weights[idx * 4 + 2], weights[idx * 4 + 3]);
                }
                newIndices.push(newPositions.length / 3 - 1);
            }
        }

        const geometry = {
            position: new Float32Array(newPositions),
            normal: new Float32Array(newNormals),
            uv: new Float32Array(newUVs),
            index: new Uint32Array(newIndices)
        };

        if (colors) {
            geometry.color = new Float32Array(newColors);
        }
        if (joints) {
            geometry.joints = new Uint32Array(newJoints);
        }
        if (weights) {
            geometry.weights = new Float32Array(newWeights);
        }

        return new Geometry(geometry);
    }

    static unrepetizeUVs(geometry) {
        const positions = geometry.attributes.position;
        const normals = geometry.attributes.normal;
        const uvs = geometry.attributes.uv;
        const indices = geometry.attributes.indices;
        const colors = geometry.attributes.color;
        const joints = geometry.attributes.joints;
        const weights = geometry.attributes.weights;

        const newPositions = [];
        const newNormals = [];
        const newUVs = [];
        const newIndices = [];
        const newColors = colors ? [] : null;
        const newJoints = joints ? [] : null;
        const newWeights = weights ? [] : null;

        // Process each triangle
        for (let i = 0; i < indices.length; i += 3) {
            const tri = [indices[i], indices[i + 1], indices[i + 2]];
            
            // Get UV coordinates for the triangle
            const uvCoords = tri.map(idx => [uvs[idx * 2], uvs[idx * 2 + 1]]);
            
            // Get integer and fractional parts of UVs
            const uvInts = uvCoords.map(uv => [
                Math.floor(Math.abs(uv[0])),
                Math.floor(Math.abs(uv[1]))
            ]);
            const uvFracs = uvCoords.map(uv => [
                Math.abs(uv[0]) % 1,
                Math.abs(uv[1]) % 1
            ]);
            
            // Find max UV integer values to determine grid size
            const maxU = Math.max(...uvInts.map(uv => uv[0]));
            const maxV = Math.max(...uvInts.map(uv => uv[1]));
            
            if (maxU > 0 || maxV > 0) {
                // Need to split into grid
                for (let u = 0; u <= maxU; u++) {
                    for (let v = 0; v <= maxV; v++) {
                        // Generate vertices for this grid cell
                        const cellVertices = [];
                        
                        // For each vertex of original triangle
                        for (let j = 0; j < 3; j++) {
                            const origPos = [
                                positions[tri[j] * 3],
                                positions[tri[j] * 3 + 1], 
                                positions[tri[j] * 3 + 2]
                            ];
                            const origNorm = [
                                normals[tri[j] * 3],
                                normals[tri[j] * 3 + 1],
                                normals[tri[j] * 3 + 2]
                            ];
                            
                            // Calculate UV for this cell
                            const cellUV = [
                                (uvFracs[j][0] + u) / (maxU + 1),
                                (uvFracs[j][1] + v) / (maxV + 1)
                            ];
                            
                            const idx = newPositions.length / 3;
                            newPositions.push(...origPos);
                            newNormals.push(...origNorm);
                            newUVs.push(...cellUV);

                            if (colors) {
                                newColors.push(
                                    colors[tri[j] * 4],
                                    colors[tri[j] * 4 + 1],
                                    colors[tri[j] * 4 + 2],
                                    colors[tri[j] * 4 + 3]
                                );
                            }
                            if (joints) {
                                newJoints.push(
                                    joints[tri[j] * 4],
                                    joints[tri[j] * 4 + 1],
                                    joints[tri[j] * 4 + 2],
                                    joints[tri[j] * 4 + 3]
                                );
                            }
                            if (weights) {
                                newWeights.push(
                                    weights[tri[j] * 4],
                                    weights[tri[j] * 4 + 1],
                                    weights[tri[j] * 4 + 2],
                                    weights[tri[j] * 4 + 3]
                                );
                            }

                            cellVertices.push(idx);
                        }
                        
                        // Add triangle indices for this cell
                        newIndices.push(
                            cellVertices[0],
                            cellVertices[1],
                            cellVertices[2]
                        );
                    }
                }
                
            } else {
                // Triangle doesn't need splitting, add as-is with normalized UVs
                for (let j = 0; j < 3; j++) {
                    const idx = tri[j];
                    newPositions.push(
                        positions[idx * 3],
                        positions[idx * 3 + 1],
                        positions[idx * 3 + 2]
                    );
                    newNormals.push(
                        normals[idx * 3],
                        normals[idx * 3 + 1],
                        normals[idx * 3 + 2]
                    );
                    newUVs.push(
                        uvFracs[j][0],
                        uvFracs[j][1]
                    );

                    if (colors) {
                        newColors.push(
                            colors[idx * 4],
                            colors[idx * 4 + 1],
                            colors[idx * 4 + 2],
                            colors[idx * 4 + 3]
                        );
                    }
                    if (joints) {
                        newJoints.push(
                            joints[idx * 4],
                            joints[idx * 4 + 1],
                            joints[idx * 4 + 2],
                            joints[idx * 4 + 3]
                        );
                    }
                    if (weights) {
                        newWeights.push(
                            weights[idx * 4],
                            weights[idx * 4 + 1],
                            weights[idx * 4 + 2],
                            weights[idx * 4 + 3]
                        );
                    }
                }
                
                const baseIdx = newPositions.length / 3 - 3;
                newIndices.push(baseIdx, baseIdx + 1, baseIdx + 2);
            }
        }

        const result = {
            position: new Float32Array(newPositions),
            normal: new Float32Array(newNormals),
            uv: new Float32Array(newUVs),
            index: new Uint32Array(newIndices)
        };

        if (colors) {
            result.color = new Float32Array(newColors);
        }
        if (joints) {
            result.joints = new Uint32Array(newJoints);
        }
        if (weights) {
            result.weights = new Float32Array(newWeights);
        }

        return new Geometry(result);
    }
    
}


export { Geometry }
