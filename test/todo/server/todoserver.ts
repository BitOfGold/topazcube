import TopazCubeServer from "topazcube/server"

class TodoServer extends TopazCubeServer {
  constructor() {
    super({
      name: "Test-TodoServer",
      port: 4799,
      database: "todo",
    })
  }

  onCreate(name) {
    return {
      todos: [],
    }
  }
  
}

let server = new TodoServer()
