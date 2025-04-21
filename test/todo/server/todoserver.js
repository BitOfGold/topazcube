import TopazCubeServer from "../../src/server.js"

class TodoServer extends TopazCubeServer {
  constructor() {
    super({
      port: 4799,
    })
  }

  canCreate(client, name) {
    return true;
  }

  onCreate(name) {
    return {
      todos: [],
    }
  }

  canUpdate(client, name) {
    return true;
  }
  
  onHydrate(name, doc) {
    doc.random = Math.random()
  }
  onUpdate(name, doc, dt) {}
  onMessage(client, message) {}


}

let server = new TodoServer()
