import { vec3, quat } from 'gl-matrix'
import TopazCubeServer from 'topazcube/server'
import * as terminal from 'topazcube/terminal'

const SMALL = 1e-5
const BIGDIFF = 50000000
const TEST_OBJECTS = 5000

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
        entity._velocity[0] *= -1
        entity.position[0] = BIGDIFF + 100 - SMALL
      }
      if (entity.position[0] < BIGDIFF - 100) {
        entity._velocity[0] *= -1
        entity.position[0] = BIGDIFF - 100 + SMALL
      }
      if (entity.position[2] > 100) {
        entity._velocity[2] *= -1
        entity.position[2] = 100 - SMALL
      }
      if (entity.position[2] < -100) {
        entity._velocity[2] *= -1
        entity.position[2] = -100 + SMALL
      }
      this.propertyChange(name, id, 'position')
      quat.rotateY(entity.rotation, entity.rotation, entity._av * dt)
      this.propertyChange(name, id, 'rotation')
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
