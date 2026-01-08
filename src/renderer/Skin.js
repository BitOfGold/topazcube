/**
 * Skin class for skeletal animation
 * Manages joint hierarchy, animations, and joint matrices texture for GPU skinning
 *
 * Supports animation blending for smooth transitions between animations.
 */
class Skin {
    engine = null

    constructor(engine = null, options = {}) {
        this.engine = engine
        this.joints = []                    // Array of joint nodes
        this.inverseBindMatrices = []       // Inverse bind matrices for each joint
        this.jointMatrices = []             // Final joint matrices (worldMatrix * inverseBindMatrix)
        this.jointData = null               // Float32Array for texture upload
        this.jointTexture = null            // GPU texture storing joint matrices
        this.animations = {}                // Named animations { name: Animation }
        this.currentAnimation = null        // Currently playing animation name
        this.time = 0                       // Current animation time
        this.speed = 1.0                    // Playback speed multiplier
        this.loop = true                    // Whether to loop animation
        this.rootNode = null                // Root node of skeleton hierarchy

        // Animation blending state
        this.blendFromAnimation = null      // Previous animation name (during blend)
        this.blendFromTime = 0              // Time in previous animation
        this.blendWeight = 1.0              // Blend weight (0 = previous, 1 = current)
        this.blendDuration = 0.3            // Default blend duration in seconds
        this.blendElapsed = 0               // Elapsed time in current blend
        this.isBlending = false             // Whether currently blending

        // Per-skin local transforms (for individual skins that don't share joint state)
        this.localTransforms = null         // Array of { position, rotation, scale } per joint
        this.worldMatrices = null           // Array of mat4 for world transforms
        this.useLocalTransforms = false     // Whether to use per-skin transforms instead of shared joints

        // Previous frame joint matrices for motion vectors
        this.prevJointData = null           // Float32Array for previous frame
        this.prevJointTexture = null        // GPU texture storing previous joint matrices
        this.prevJointTextureView = null    // View for binding
    }

    /**
     * Initialize the skin with joints and inverse bind matrices
     * @param {Array} joints - Array of joint nodes
     * @param {Float32Array} inverseBindMatrixData - Flat array of inverse bind matrices
     * @param {Object} rootNode - Root node of the skeleton
     */
    init(joints, inverseBindMatrixData, rootNode) {
        const { device } = this.engine

        this.joints = joints
        this.rootNode = rootNode
        this.inverseBindMatrices = []
        this.jointMatrices = []

        // Allocate joint data array (16 floats per joint for mat4)
        this.jointData = new Float32Array(joints.length * 16)

        // Parse inverse bind matrices and create views for joint matrices
        for (let i = 0; i < joints.length; i++) {
            // Create view into inverse bind matrix data
            const ibm = new Float32Array(
                inverseBindMatrixData.buffer,
                inverseBindMatrixData.byteOffset + i * 16 * 4,
                16
            )
            this.inverseBindMatrices.push(ibm)

            // Create view into joint data for this joint's matrix
            const jointMatrix = new Float32Array(this.jointData.buffer, i * 16 * 4, 16)
            mat4.identity(jointMatrix)
            this.jointMatrices.push(jointMatrix)

            // Link joint to this skin
            joints[i].skin = this
        }

        // Create GPU texture for joint matrices (4 pixels wide x numJoints high, RGBA32F)
        this.jointTexture = device.createTexture({
            size: [4, joints.length, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })

        this.jointTextureView = this.jointTexture.createView()

        // Create previous frame joint texture for motion vectors
        this.prevJointData = new Float32Array(joints.length * 16)
        this.prevJointTexture = device.createTexture({
            size: [4, joints.length, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        this.prevJointTextureView = this.prevJointTexture.createView()

        // Initialize prevJointData with identity matrices
        for (let i = 0; i < joints.length; i++) {
            const offset = i * 16
            // Identity matrix column-major
            this.prevJointData[offset + 0] = 1; this.prevJointData[offset + 1] = 0; this.prevJointData[offset + 2] = 0; this.prevJointData[offset + 3] = 0
            this.prevJointData[offset + 4] = 0; this.prevJointData[offset + 5] = 1; this.prevJointData[offset + 6] = 0; this.prevJointData[offset + 7] = 0
            this.prevJointData[offset + 8] = 0; this.prevJointData[offset + 9] = 0; this.prevJointData[offset + 10] = 1; this.prevJointData[offset + 11] = 0
            this.prevJointData[offset + 12] = 0; this.prevJointData[offset + 13] = 0; this.prevJointData[offset + 14] = 0; this.prevJointData[offset + 15] = 1
        }

        // Create sampler for joint texture (nearest filtering, no interpolation)
        this.jointSampler = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        })
    }

    /**
     * Add an animation to this skin
     * @param {string} name - Animation name
     * @param {Object} animation - Animation data { duration, channels }
     */
    addAnimation(name, animation) {
        this.animations[name] = animation
        if (!this.currentAnimation) {
            this.currentAnimation = name
        }
    }

    /**
     * Play a named animation with optional blending from current animation
     * @param {string} name - Animation name
     * @param {boolean} loop - Whether to loop
     * @param {number} phase - Starting phase (0-1)
     * @param {number} blendTime - Blend duration in seconds (0 = instant switch)
     */
    play(name, loop = true, phase = 0.0, blendTime = 0) {
        if (!this.animations[name]) return

        const anim = this.animations[name]
        phase = Math.max(Math.min(phase, 1.0), 0.0)

        // If we have a current animation and blend time > 0, start blending
        if (blendTime > 0 && this.currentAnimation && this.currentAnimation !== name) {
            this.blendFromAnimation = this.currentAnimation
            this.blendFromTime = this.time
            this.blendDuration = blendTime
            this.blendElapsed = 0
            this.blendWeight = 0  // Start fully on previous animation
            this.isBlending = true
        } else {
            this.isBlending = false
            this.blendFromAnimation = null
        }

        this.currentAnimation = name
        this.loop = loop
        this.time = phase * anim.duration
    }

    /**
     * Transition to a new animation with blending
     * @param {string} name - Target animation name
     * @param {number} blendTime - Blend duration (default 0.3s)
     */
    blendTo(name, blendTime = 0.3) {
        this.play(name, this.loop, 0, blendTime)
    }

    /**
     * Update animation and joint matrices
     * @param {number} dt - Delta time in seconds
     */
    update(dt) {
        const { device } = this.engine

        // Copy current joint data to previous BEFORE updating (for motion vectors)
        if (this.prevJointData && this.jointData) {
            this.prevJointData.set(this.jointData)
        }

        // Update animation time
        this.time += dt * this.speed

        // Update blend progress (only if dt > 0, otherwise assume externally managed)
        if (this.isBlending && dt > 0) {
            this.blendElapsed += dt
            this.blendWeight = Math.min(this.blendElapsed / this.blendDuration, 1.0)

            // Also advance the "from" animation time
            this.blendFromTime += dt * this.speed

            // Blend complete
            if (this.blendWeight >= 1.0) {
                this.isBlending = false
                this.blendFromAnimation = null
                this.blendWeight = 1.0
            }
        }

        // Apply animations (with blending if active)
        if (this.useLocalTransforms) {
            this._applyAnimationToLocalTransforms(dt)
        } else {
            this._applyAnimationToSharedJoints()
        }

        // Update world matrices
        if (this.useLocalTransforms) {
            this._updateWorldMatricesFromLocal()
        } else if (this.rootNode) {
            this.rootNode.updateMatrix()
        }

        // Calculate final joint matrices: jointMatrix = worldMatrix * inverseBindMatrix
        for (let i = 0; i < this.joints.length; i++) {
            const worldMat = this.useLocalTransforms ? this.worldMatrices[i] : this.joints[i].world
            const dst = this.jointMatrices[i]
            mat4.multiply(dst, worldMat, this.inverseBindMatrices[i])
        }

        // Upload previous joint matrices to GPU (for motion vectors)
        if (this.prevJointTexture && this.prevJointData) {
            device.queue.writeTexture(
                { texture: this.prevJointTexture },
                this.prevJointData,
                { bytesPerRow: 4 * 4 * 4, rowsPerImage: this.joints.length },
                [4, this.joints.length, 1]
            )
        }

        // Upload current joint matrices to GPU
        device.queue.writeTexture(
            { texture: this.jointTexture },
            this.jointData,
            { bytesPerRow: 4 * 4 * 4, rowsPerImage: this.joints.length },
            [4, this.joints.length, 1]
        )
    }

    /**
     * Apply animation to shared joint objects (original behavior)
     */
    _applyAnimationToSharedJoints() {
        if (!this.currentAnimation || !this.animations[this.currentAnimation]) return

        const currentAnim = this.animations[this.currentAnimation]

        // Handle looping for current animation
        let currentTime = this.time
        if (this.loop && currentAnim.duration > 0) {
            currentTime = currentTime % currentAnim.duration
        } else {
            currentTime = Math.min(currentTime, currentAnim.duration)
        }

        if (this.isBlending && this.blendFromAnimation && this.animations[this.blendFromAnimation]) {
            // Blending between two animations
            const fromAnim = this.animations[this.blendFromAnimation]

            // Handle looping for from animation
            let fromTime = this.blendFromTime
            if (this.loop && fromAnim.duration > 0) {
                fromTime = fromTime % fromAnim.duration
            } else {
                fromTime = Math.min(fromTime, fromAnim.duration)
            }

            // Apply blended animation
            this._applyBlendedAnimation(fromAnim, fromTime, currentAnim, currentTime, this.blendWeight)
        } else {
            // Single animation
            this._applyAnimation(currentAnim, currentTime)
        }
    }

    /**
     * Apply animation with blending to local transforms (for individual skins)
     */
    _applyAnimationToLocalTransforms(dt) {
        if (!this.localTransforms) return
        if (!this.currentAnimation || !this.animations[this.currentAnimation]) return

        const currentAnim = this.animations[this.currentAnimation]

        // Handle looping for current animation
        let currentTime = this.time
        if (this.loop && currentAnim.duration > 0) {
            currentTime = currentTime % currentAnim.duration
        } else {
            currentTime = Math.min(currentTime, currentAnim.duration)
        }

        if (this.isBlending && this.blendFromAnimation && this.animations[this.blendFromAnimation]) {
            const fromAnim = this.animations[this.blendFromAnimation]

            let fromTime = this.blendFromTime
            if (this.loop && fromAnim.duration > 0) {
                fromTime = fromTime % fromAnim.duration
            } else {
                fromTime = Math.min(fromTime, fromAnim.duration)
            }

            this._applyBlendedAnimationToLocal(fromAnim, fromTime, currentAnim, currentTime, this.blendWeight)
        } else {
            this._applyAnimationToLocal(currentAnim, currentTime)
        }
    }

    /**
     * Apply blended animation between two animations to shared joints
     */
    _applyBlendedAnimation(fromAnim, fromTime, toAnim, toTime, weight) {
        // Temporary storage for blended values
        const tempPos = vec3.create()
        const tempRot = quat.create()
        const tempScale = vec3.fromValues(1, 1, 1)

        // First apply "from" animation to get base pose
        this._applyAnimation(fromAnim, fromTime)

        // Store the "from" pose for each joint
        const fromPoses = this.joints.map(joint => ({
            position: vec3.clone(joint.position),
            rotation: quat.clone(joint.rotation),
            scale: vec3.clone(joint.scale)
        }))

        // Apply "to" animation
        this._applyAnimation(toAnim, toTime)

        // Blend between stored "from" pose and current "to" pose
        for (let i = 0; i < this.joints.length; i++) {
            const joint = this.joints[i]
            const fromPose = fromPoses[i]

            // Lerp position
            vec3.lerp(joint.position, fromPose.position, joint.position, weight)

            // Slerp rotation
            quat.slerp(joint.rotation, fromPose.rotation, joint.rotation, weight)

            // Lerp scale
            vec3.lerp(joint.scale, fromPose.scale, joint.scale, weight)
        }
    }

    /**
     * Apply animation keyframes to joints
     * @param {Object} anim - Animation data
     * @param {number} time - Current time
     */
    _applyAnimation(anim, time) {
        for (const channel of anim.channels) {
            const joint = channel.target
            const sampler = channel.sampler

            // Find keyframes
            const times = sampler.input
            const values = sampler.output

            // Find the two keyframes to interpolate between
            let prevIndex = 0
            let nextIndex = 0

            for (let i = 0; i < times.length - 1; i++) {
                if (time >= times[i] && time < times[i + 1]) {
                    prevIndex = i
                    nextIndex = i + 1
                    break
                }
                if (i === times.length - 2) {
                    prevIndex = i + 1
                    nextIndex = i + 1
                }
            }

            // Calculate interpolation factor
            let t = 0
            if (nextIndex !== prevIndex) {
                t = (time - times[prevIndex]) / (times[nextIndex] - times[prevIndex])
            }

            // Apply based on path type
            const path = channel.path
            const numComponents = path === 'rotation' ? 4 : 3

            if (path === 'translation') {
                const prev = [
                    values[prevIndex * 3],
                    values[prevIndex * 3 + 1],
                    values[prevIndex * 3 + 2]
                ]
                const next = [
                    values[nextIndex * 3],
                    values[nextIndex * 3 + 1],
                    values[nextIndex * 3 + 2]
                ]
                vec3.lerp(joint.position, prev, next, t)
            } else if (path === 'rotation') {
                const prev = quat.fromValues(
                    values[prevIndex * 4],
                    values[prevIndex * 4 + 1],
                    values[prevIndex * 4 + 2],
                    values[prevIndex * 4 + 3]
                )
                const next = quat.fromValues(
                    values[nextIndex * 4],
                    values[nextIndex * 4 + 1],
                    values[nextIndex * 4 + 2],
                    values[nextIndex * 4 + 3]
                )
                quat.slerp(joint.rotation, prev, next, t)
            } else if (path === 'scale') {
                const prev = [
                    values[prevIndex * 3],
                    values[prevIndex * 3 + 1],
                    values[prevIndex * 3 + 2]
                ]
                const next = [
                    values[nextIndex * 3],
                    values[nextIndex * 3 + 1],
                    values[nextIndex * 3 + 2]
                ]
                vec3.lerp(joint.scale, prev, next, t)
            }
        }
    }

    /**
     * Initialize local transforms for individual skin (enables per-skin animation state)
     * Call this to make the skin independent from the shared joint hierarchy
     */
    initLocalTransforms() {
        this.useLocalTransforms = true
        this.localTransforms = []
        this.worldMatrices = []

        // Create local transform storage for each joint
        for (let i = 0; i < this.joints.length; i++) {
            const joint = this.joints[i]
            this.localTransforms.push({
                position: vec3.clone(joint.position),
                rotation: quat.clone(joint.rotation),
                scale: vec3.clone(joint.scale)
            })
            this.worldMatrices.push(mat4.create())
        }
    }

    /**
     * Apply animation to local transforms (for individual skins)
     */
    _applyAnimationToLocal(anim, time) {
        for (const channel of anim.channels) {
            const joint = channel.target
            const jointIndex = this.joints.indexOf(joint)
            if (jointIndex === -1) continue

            const localTrans = this.localTransforms[jointIndex]
            const sampler = channel.sampler
            const times = sampler.input
            const values = sampler.output

            // Find keyframes
            let prevIndex = 0
            let nextIndex = 0
            for (let i = 0; i < times.length - 1; i++) {
                if (time >= times[i] && time < times[i + 1]) {
                    prevIndex = i
                    nextIndex = i + 1
                    break
                }
                if (i === times.length - 2) {
                    prevIndex = i + 1
                    nextIndex = i + 1
                }
            }

            let t = 0
            if (nextIndex !== prevIndex) {
                t = (time - times[prevIndex]) / (times[nextIndex] - times[prevIndex])
            }

            const path = channel.path
            if (path === 'translation') {
                const prev = [values[prevIndex * 3], values[prevIndex * 3 + 1], values[prevIndex * 3 + 2]]
                const next = [values[nextIndex * 3], values[nextIndex * 3 + 1], values[nextIndex * 3 + 2]]
                vec3.lerp(localTrans.position, prev, next, t)
            } else if (path === 'rotation') {
                const prev = quat.fromValues(values[prevIndex * 4], values[prevIndex * 4 + 1], values[prevIndex * 4 + 2], values[prevIndex * 4 + 3])
                const next = quat.fromValues(values[nextIndex * 4], values[nextIndex * 4 + 1], values[nextIndex * 4 + 2], values[nextIndex * 4 + 3])
                quat.slerp(localTrans.rotation, prev, next, t)
            } else if (path === 'scale') {
                const prev = [values[prevIndex * 3], values[prevIndex * 3 + 1], values[prevIndex * 3 + 2]]
                const next = [values[nextIndex * 3], values[nextIndex * 3 + 1], values[nextIndex * 3 + 2]]
                vec3.lerp(localTrans.scale, prev, next, t)
            }
        }
    }

    /**
     * Apply blended animation to local transforms
     */
    _applyBlendedAnimationToLocal(fromAnim, fromTime, toAnim, toTime, weight) {
        // First apply "from" animation
        this._applyAnimationToLocal(fromAnim, fromTime)

        // Store from pose
        const fromPoses = this.localTransforms.map(lt => ({
            position: vec3.clone(lt.position),
            rotation: quat.clone(lt.rotation),
            scale: vec3.clone(lt.scale)
        }))

        // Apply "to" animation
        this._applyAnimationToLocal(toAnim, toTime)

        // Blend
        for (let i = 0; i < this.localTransforms.length; i++) {
            const lt = this.localTransforms[i]
            const from = fromPoses[i]

            vec3.lerp(lt.position, from.position, lt.position, weight)
            quat.slerp(lt.rotation, from.rotation, lt.rotation, weight)
            vec3.lerp(lt.scale, from.scale, lt.scale, weight)
        }
    }

    /**
     * Update world matrices from local transforms (replicates joint hierarchy traversal)
     */
    _updateWorldMatricesFromLocal() {
        // Build parent-child mapping from joints
        const parentIndices = this.joints.map(joint => {
            if (joint.parent) {
                return this.joints.indexOf(joint.parent)
            }
            return -1
        })

        // Process joints in order (assuming they're topologically sorted - parents before children)
        for (let i = 0; i < this.joints.length; i++) {
            const lt = this.localTransforms[i]
            const worldMat = this.worldMatrices[i]
            const parentIndex = parentIndices[i]

            // Build local matrix
            const localMat = mat4.create()
            mat4.fromRotationTranslationScale(localMat, lt.rotation, lt.position, lt.scale)

            // Combine with parent
            if (parentIndex >= 0) {
                mat4.multiply(worldMat, this.worldMatrices[parentIndex], localMat)
            } else {
                mat4.copy(worldMat, localMat)
            }
        }
    }

    /**
     * Get the number of joints
     */
    get numJoints() {
        return this.joints.length
    }

    /**
     * Get available animation names
     */
    getAnimationNames() {
        return Object.keys(this.animations)
    }

    /**
     * Clone this skin with its own joint texture but sharing animation data
     * This allows multiple instances to play at different phases independently
     * @param {boolean} individual - If true, create local transforms for independent animation/blending
     * @returns {Skin} A new Skin instance
     */
    clone(individual = false) {
        const { device } = this.engine

        const clonedSkin = new Skin(this.engine)

        // Share references to joints hierarchy and animations (read-only during playback)
        clonedSkin.joints = this.joints
        clonedSkin.inverseBindMatrices = this.inverseBindMatrices
        clonedSkin.animations = this.animations
        clonedSkin.rootNode = this.rootNode
        clonedSkin.speed = this.speed
        clonedSkin.loop = this.loop
        clonedSkin.currentAnimation = this.currentAnimation
        clonedSkin.blendDuration = this.blendDuration

        // Mark as externally managed - GBufferPass should skip calling update()
        clonedSkin.externallyManaged = true

        // Create own joint matrices array (these get modified during update)
        clonedSkin.jointMatrices = []
        clonedSkin.jointData = new Float32Array(this.joints.length * 16)

        for (let i = 0; i < this.joints.length; i++) {
            const jointMatrix = new Float32Array(clonedSkin.jointData.buffer, i * 16 * 4, 16)
            mat4.identity(jointMatrix)
            clonedSkin.jointMatrices.push(jointMatrix)
        }

        // Create own GPU texture for joint matrices
        clonedSkin.jointTexture = device.createTexture({
            size: [4, this.joints.length, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })

        clonedSkin.jointTextureView = clonedSkin.jointTexture.createView()

        // Create previous frame joint texture for motion vectors
        clonedSkin.prevJointData = new Float32Array(this.joints.length * 16)
        clonedSkin.prevJointTexture = device.createTexture({
            size: [4, this.joints.length, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
        clonedSkin.prevJointTextureView = clonedSkin.prevJointTexture.createView()

        // Create own sampler
        clonedSkin.jointSampler = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        })

        clonedSkin.time = this.time

        // For individual skins, initialize local transforms for independent animation/blending
        if (individual) {
            clonedSkin.initLocalTransforms()
        }

        return clonedSkin
    }

    /**
     * Create an individual clone with its own animation state and blending support
     * Use this for entities that need smooth animation transitions (close to camera)
     * @returns {Skin} A new independent Skin instance
     */
    cloneForIndividual() {
        return this.clone(true)
    }

    /**
     * Update animation at a specific time (absolute, not delta)
     * Used for phase-offset playback where multiple skins share the same animation
     * @param {number} absoluteTime - Absolute time in the animation
     */
    updateAtTime(absoluteTime) {
        const { device } = this.engine

        // Copy current joint data to previous BEFORE updating (for motion vectors)
        if (this.prevJointData && this.jointData) {
            this.prevJointData.set(this.jointData)
        }

        // Apply current animation if any
        if (this.currentAnimation && this.animations[this.currentAnimation]) {
            const anim = this.animations[this.currentAnimation]

            // Handle looping
            let t = absoluteTime
            if (this.loop && anim.duration > 0) {
                t = t % anim.duration
            } else {
                t = Math.min(t, anim.duration)
            }

            // Apply animation to joints
            this._applyAnimation(anim, t)
        }

        // Update world matrices starting from root
        if (this.rootNode) {
            this.rootNode.updateMatrix()
        }

        // Calculate final joint matrices: jointMatrix = worldMatrix * inverseBindMatrix
        for (let i = 0; i < this.joints.length; i++) {
            const joint = this.joints[i]
            const dst = this.jointMatrices[i]

            // dst = joint.world * inverseBindMatrix
            mat4.multiply(dst, joint.world, this.inverseBindMatrices[i])
        }

        // Upload previous joint matrices to GPU (for motion vectors)
        if (this.prevJointTexture && this.prevJointData) {
            device.queue.writeTexture(
                { texture: this.prevJointTexture },
                this.prevJointData,
                { bytesPerRow: 4 * 4 * 4, rowsPerImage: this.joints.length },
                [4, this.joints.length, 1]
            )
        }

        // Upload current joint matrices to GPU
        device.queue.writeTexture(
            { texture: this.jointTexture },
            this.jointData,
            { bytesPerRow: 4 * 4 * 4, rowsPerImage: this.joints.length },
            [4, this.joints.length, 1]
        )
    }
}

/**
 * Joint node for skeletal animation
 * Represents a bone in the skeleton hierarchy
 */
class Joint {
    constructor(name = 'joint') {
        this.name = name
        this.position = vec3.create()
        this.rotation = quat.create()
        this.scale = vec3.fromValues(1, 1, 1)
        this.children = []
        this.parent = null
        this.skin = null

        // Matrices
        this.matrix = mat4.create()      // Local transform
        this.world = mat4.create()       // World transform (accumulated from parents)

        // Original pose (bind pose)
        this.bindPosition = vec3.create()
        this.bindRotation = quat.create()
        this.bindScale = vec3.fromValues(1, 1, 1)
    }

    /**
     * Set the local transform from a matrix
     */
    setMatrix(m) {
        mat4.getTranslation(this.position, m)
        mat4.getRotation(this.rotation, m)
        mat4.getScaling(this.scale, m)
    }

    /**
     * Save current pose as bind pose
     */
    saveBindPose() {
        vec3.copy(this.bindPosition, this.position)
        quat.copy(this.bindRotation, this.rotation)
        vec3.copy(this.bindScale, this.scale)
    }

    /**
     * Reset to bind pose
     */
    resetToBindPose() {
        vec3.copy(this.position, this.bindPosition)
        quat.copy(this.rotation, this.bindRotation)
        vec3.copy(this.scale, this.bindScale)
    }

    /**
     * Add a child joint
     */
    addChild(child) {
        child.parent = this
        this.children.push(child)
    }

    /**
     * Update local and world matrices
     * @param {mat4} parentWorld - Parent's world matrix (optional)
     */
    updateMatrix(parentWorld = null) {
        // Build local matrix from TRS
        mat4.fromRotationTranslationScale(this.matrix, this.rotation, this.position, this.scale)

        // Combine with parent world matrix
        if (parentWorld) {
            mat4.multiply(this.world, parentWorld, this.matrix)
        } else {
            mat4.copy(this.world, this.matrix)
        }

        // Update children
        for (const child of this.children) {
            child.updateMatrix(this.world)
        }
    }
}

export { Skin, Joint }
