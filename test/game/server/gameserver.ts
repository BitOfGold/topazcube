import { vec3, quat } from 'gl-matrix'
import TopazCubeServer from 'topazcube/server'
import * as terminal from 'topazcube/terminal'

const SMALL = 1e-5
const BIGDIFF = 50000000
const TEST_OBJECTS = 500

const newRotation = quat.create()
const UP = vec3.fromValues(0, 1, 0)

function grandom() {
  return (Math.random() +
    Math.random() +
    Math.random() +
    Math.random() +
    Math.random() +
    Math.random() +
    Math.random()) /
    7.0
}

class GameServer extends TopazCubeServer {
  constructor() {
    super({
      name: 'Test-GameServer',
      port: 4849,
      database: 'game3d',
      allowSave: false,
      //simulateLatency: 500,
      cycle: 200,
      allowSync: false,
      allowWebRTC: true,
      allowFastPatch: true,
      allowCompression: true,
    })
  }

  onCreate(name:string):any {
    let entities = {}
    for (let i = 0; i < TEST_OBJECTS; i++) {
      entities[this.getUID()] = {
        type: 'enemy',
        position: [
          BIGDIFF + Math.random() * 198 - 99,
          1,
          Math.random() * 198 - 99,
        ],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        model: 'models.glb|goblin',
        animation: 'walk',
        sound: '',
        effect: '',

        _velocity: [grandom() * 20 - 10, 0, grandom() * 20 - 10],
        _av: Math.random() * 0.64,
      }
    }

    return {
      origin: [BIGDIFF, 0, 0],
      entities: entities,
    }
  }

  async onHydrate(name, doc) {
    doc.rand = Math.random()
  }

  onUpdate(name, doc, dt) {
    //console.log(`${name} update ${this.update} dt: ${dt}`, )
    for (let id in doc.entities) {
      let entity = doc.entities[id]
      vec3.scaleAndAdd(entity.position, entity.position, entity._velocity, dt)
      if (entity.position[0] > BIGDIFF + 100) {
        if (entity._velocity[0] > 0) {
          entity._velocity[0] *= -1
        }
        entity.position[0] = BIGDIFF + 100 - SMALL
      }
      if (entity.position[0] < BIGDIFF - 100) {
        if (entity._velocity[0] < 0) {
          entity._velocity[0] *= -1
        }
        entity.position[0] = BIGDIFF - 100 + SMALL
      }
      if (entity.position[2] > 100) {
        if (entity._velocity[2] > 0) {
          entity._velocity[2] *= -1
        }
        entity.position[2] = 100 - SMALL
      }
      if (entity.position[2] < -100) {
        if (entity._velocity[2] < 0) {
          entity._velocity[2] *= -1
        }
        entity.position[2] = -100 + SMALL
      }
      this.propertyChange(name, id, 'position')
      // Randomly change velocity direction
      if (Math.random() < 0.01) { // 1% chance per frame to change direction
        entity._velocity[0] = grandom() * 20 - 10
        entity._velocity[2] = grandom() * 20 - 10
      }

      // Calculate target rotation based on velocity direction
      const velocityAngle = Math.atan2(entity._velocity[0], entity._velocity[2])
      const currentRotationY = quat.getAxisAngle(UP, entity.rotation);
      let angleDiff = currentRotationY - velocityAngle
      // Normalize angle difference to [-π, π]
      while (angleDiff > Math.PI) angleDiff -= Math.PI
      while (angleDiff < -Math.PI) angleDiff += Math.PI

      // If rotation differs by more than 4 degree, rotate towards velocity
      if (Math.abs(angleDiff) > Math.PI / 180 * 4.) { // 4 degree in radians
        const turnSpeed = 0.5 // radians per second
        const maxTurn = turnSpeed * dt
        const turnAmount = Math.sign(angleDiff) * maxTurn

        quat.copy(newRotation, entity.rotation)
        quat.rotateY(newRotation, newRotation, turnAmount)
        quat.copy(entity.rotation, newRotation)
        this.propertyChange(name, id, 'rotation')
      }
    }
  }

  onConnect(client) {
    console.log('Game: Client connected:', client.ID)
  }

  onMessage(client, message) {
    console.log('Game: Client message:', client.ID, message)
  }
}

let server = new GameServer()

terminal.setTitle('Server: Test Game')
