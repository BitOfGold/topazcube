import TopazCubeClient from 'topazcube/client'
import * as THREE from 'three'
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls'
import { CSM } from 'three/examples/jsm/csm/CSM'

class GameClient extends TopazCubeClient {
  canvas: HTMLCanvasElement
  controls: PointerLockControls
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene

  constructor() {
    super({
      url: 'ws://192.168.0.200:4849',
      allowWebRTC: true,
      allowSync: false,
    })
    this.init()
    window.addEventListener('resize', () => this.resize())
    this.resize()
    this.renderer.setAnimationLoop(() => this.gameLoop())
    setInterval(() => this.updateUI(), 50)
  }

  onConnect() {
    this.subscribe('world')
    console.log('onConnect')
  }

  onDisconnect() {
    console.log('onDisconnect')
  }

  onMessage(message: string) {
    console.log('onMessage', message)
  }

  onChange(name: string, doc: object) {
    console.log('onChange', name, doc)
  }

  resize() {
    let width = window.innerWidth
    let height = window.innerHeight
    this.canvas.width = width
    this.canvas.height = height
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height)
  }

  init() {
    this.canvas = document.getElementById('game-canvas') as HTMLCanvasElement

    // THREE.js setup
    this.scene = new THREE.Scene()

    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      5000
    )
    this.camera.position.set(74, 25, -74)
    this.camera.lookAt(45, 0, -45)
    this.viewOrigin = [50000000, 0, 0]
    const csm = new CSM({
      mode: 'practical',
      fade: true,
      far: this.camera.far,
      maxFar: 200,
      cascades: 4,
      shadowMapSize: 2048,
      lightDirection: new THREE.Vector3(-8, -8, -3),
      lightColor: 0xffeedd,
      lightIntensity: 2.0,
      camera: this.camera,
      parent: this.scene,
    })
    this.csm = csm

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
    })
    this.renderer.setSize(this.canvas.width, this.canvas.height)
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap

    // Configure controls for FPS-style movement
    //
    this.moveForward = false
    this.moveBackward = false
    this.moveLeft = false
    this.moveRight = false
    this.velocity = new THREE.Vector3()
    this.direction = new THREE.Vector3()
    this.prevTime = performance.now()

    // Set up pointer lock controls
    this.controls = new PointerLockControls(this.camera, document.body)
    this.scene.add(this.controls.object)

    // Add click event to lock pointer
    document.addEventListener('click', () => {
      this.controls.lock()
    })

    // Add event listeners for movement
    const onKeyDown = (event) => {
      switch (event.code) {
        case 'ArrowUp':
        case 'KeyW':
          this.moveForward = true
          break
        case 'ArrowLeft':
        case 'KeyA':
          this.moveLeft = true
          break
        case 'ArrowDown':
        case 'KeyS':
          this.moveBackward = true
          break
        case 'ArrowRight':
        case 'KeyD':
          this.moveRight = true
          break
      }
    }

    const onKeyUp = (event) => {
      switch (event.code) {
        case 'ArrowUp':
        case 'KeyW':
          this.moveForward = false
          break
        case 'ArrowLeft':
        case 'KeyA':
          this.moveLeft = false
          break
        case 'ArrowDown':
        case 'KeyS':
          this.moveBackward = false
          break
        case 'ArrowRight':
        case 'KeyD':
          this.moveRight = false
          break
      }
    }

    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)

    // Create skybox
    const skyColor = 0x87ceeb // Gray color
    this.scene.background = new THREE.Color(skyColor)

    // Add fog with the same color as skybox
    this.scene.fog = new THREE.FogExp2(skyColor, 0.025)

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x405060)
    ambientLight.intensity = 2.0
    this.scene.add(ambientLight)

    // Add grid helper
    const gridHelper = new THREE.GridHelper(200, 200, 0x444444, 0x666666)
    gridHelper.position.y = 0.01 // Slightly above ground to avoid z-fighting
    this.scene.add(gridHelper)

    // Create ground
    const groundGeometry = new THREE.PlaneGeometry(200, 200)
    const groundMaterial = new THREE.MeshStandardMaterial({ color: 0xaaaaaa })
    this.csm.setupMaterial(groundMaterial)
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.rotation.x = -Math.PI / 2
    ground.receiveShadow = true
    this.scene.add(ground)

    // Create instanced cubes
    const instancedGeometry = new THREE.BoxGeometry(1, 2, 1)
    const instancedMaterial = new THREE.MeshStandardMaterial({
      color: 0xff4500,
    })
    this.csm.setupMaterial(instancedMaterial)

    // Create instance mesh with 3 instances
    this.instancedCubes = new THREE.InstancedMesh(
      instancedGeometry,
      instancedMaterial,
      20000
    )
    this.instancedCubes.castShadow = true
    this.instancedCubes.receiveShadow = true
    this.instancedCubes.frustumCulled = false
    this.scene.add(this.instancedCubes)

    // Set instance count to 0 initially
    this.instancedCubes.count = 0
  }

  updateUI() {
    document.getElementById('status').textContent = this.isConnected
      ? 'Connected'
      : 'Disconnected'
    document.getElementById('client-id').textContent = this.ID
    document.getElementById('ping').textContent = this.stats.ping
    document.getElementById('down').textContent = this.stats.recBps
    document.getElementById('down-rtc').textContent = this.stats.recRTCBps
    document.getElementById('up').textContent = this.stats.sendBps
    if (this.documents['world'] && this.documents['world'].entities) {
      let eids = Object.keys(this.documents['world'].entities)
      let e1 = this.documents['world'].entities[eids[0]]
      document.getElementById('doc').textContent = JSON.stringify(e1, null, 2)
    }
  }

  updateObjects() {
    if (!this.documents['world'] || !this.documents['world'].entities) {
      return
    }
    let elist = this.documents['world'].entities
    let eids = Object.keys(elist)
    let len = eids.length
    const matrix = new THREE.Matrix4()
    const tr = new THREE.Quaternion()
    this.instancedCubes.count = Math.min(len, 10000)
    let i = 0
    for (let id in elist) {
      let e = elist[id]
      matrix.identity()
      //matrix.fromQuat(matrix, e.rotation)

      if (e.rotation) {
        tr.fromArray(e.rotation)
        matrix.makeRotationFromQuaternion(tr)
      }
      if (e.scale) {
        matrix.scale(e.scale[0], e.scale[1], e.scale[2]);
      }
      if (e.position) {
        matrix.setPosition(
          e.position[0] - this.viewOrigin[0],
          e.position[1] - this.viewOrigin[1],
          e.position[2] - this.viewOrigin[2]
        )
      } else {
        matrix.setPosition(0, -100000, 0)
      }
      this.instancedCubes.setMatrixAt(i++, matrix)
    }
    this.instancedCubes.instanceMatrix.needsUpdate = true
  }

  control() {
    /* FPS controls */
    const time = performance.now()

    if (this.controls.isLocked) {
      const delta = (time - this.prevTime) / 1000

      this.velocity.x *= 0.9
      this.velocity.z *= 0.9
      this.velocity.y *= 0.9

      // Get camera direction vector
      const cameraDirection = new THREE.Vector3()
      this.camera.getWorldDirection(cameraDirection)

      // Create right vector relative to camera direction
      const rightVector = new THREE.Vector3()
      rightVector.crossVectors(this.camera.up, cameraDirection).normalize()

      // Calculate movement direction based on camera orientation
      this.direction.set(0, 0, 0)

      if (this.moveForward) this.direction.add(cameraDirection)
      if (this.moveBackward) this.direction.sub(cameraDirection)
      if (this.moveRight) this.direction.sub(rightVector)
      if (this.moveLeft) this.direction.add(rightVector)

      // Normalize only if there's movement
      if (
        this.moveForward ||
        this.moveBackward ||
        this.moveLeft ||
        this.moveRight
      ) {
        this.direction.normalize() // Ensures consistent speed in all directions

        // Apply movement speed
        const speed = 400.0 * delta
        this.velocity.x = this.direction.x * speed
        this.velocity.y = this.direction.y * speed
        this.velocity.z = this.direction.z * speed
      }

      // Move the player
      //this.controls.moveRight(this.velocity.x * delta);
      //this.controls.moveForward(this.velocity.z * delta);

      this.controls.object.position.x += this.velocity.x * delta
      this.controls.object.position.y += this.velocity.y * delta
      this.controls.object.position.z += this.velocity.z * delta
    }

    this.prevTime = time
  }

  render() {
    this.csm.update()
    this.csm.updateFrustums()
    this.renderer.render(this.scene, this.camera)
  }

  gameLoop() {
    if (this.isConnected) {
      this.interpolate()
      this.updateObjects()
    }

    this.control()
    // Render
    this.render()
  }
}

let client = new GameClient()
client.connect()
globalThis.client = client
