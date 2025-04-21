import https from "https"
import fs from "fs"
import { reactive, clonewo_, opmsg, msgop, getUID } from "./utils.js"
import fastjsonpatch from "fast-json-patch"
import { encode, decode } from "@msgpack/msgpack"
import WebSocket, { WebSocketServer } from "ws"
import { MongoClient } from "mongodb"

const { applyPatch, applyOperation, observe } = fastjsonpatch

const LITTLE_ENDIAN = (() => {
  const buffer = new ArrayBuffer(2)
  new DataView(buffer).setInt16(0, 256, true)
  return new Int16Array(buffer)[0] === 256
})()

export default class TopazCubeServer {
  allowSync = true // allow clients to sync their changes (no server authorization)
  CYCLE = 100 // update/patch rate in ms
  clients = new Set()
  _documentChanges = {}
  documents = {}
  isLoading = {}

  constructor({ port = 8799, https = false }) {
    this.port = port
    if (https) {
      const httpsServer = https.createServer({
        key: fs.readFileSync("./.cert/privkey.pem"),
        cert: fs.readFileSync("./.cert/fullchain.pem"),
      })
      httpsServer.listen(this.port)
      this.wss = new WebSocketServer({ server: httpsServer })
      console.log("TopazCubeServer running on HTTPS port " + this.port)
    } else {
      this.wss = new WebSocketServer({ port: this.port })
      console.log("TopazCubeServer running on port " + this.port)
    }
    this.wss.on("connection", (client) => {
      this._onConnected(client)
    })

    this._initDB()
    this._exited = false
    process.stdin.resume()
    process.on("SIGINT", () => {
      this._exitSignal("SIGINT")
    })
    process.on("SIGQUIT", () => {
      this._exitSignal("SIGQUIT")
    })
    process.on("SIGTERM", () => {
      this._exitSignal("SIGTERM")
    })
    process.on("SIGUSR2", () => {
      this._exitSignal("SIGUSR2")
    })
    this._startLoop()
  }
  
  canCreate(client, name) {
    return true;
  }
  
  onCreate(name) {
    return {
      data: {}
    }
  }

  // to be redefined, to be called when a new document is hydrated
  // (created, or loaded from db)
  onHydrate(name, doc) {
  }

  _makeReactive(name) {
    //console.log(`Making document '${name}' reactive`, this.documents[name])
    this.documents[name] = reactive(name, this.documents[name], this._onDocumentChange.bind(this))
  }

  _createEmptyDocument(name) {
    let doc = this.onCreate(name)
    if (!doc) {
      return
    }
    this.documents[name] = doc
  }

  async _waitLoad(name) {
    if (this.isLoading[name]) {
      while (this.isLoading[name]) {
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    }
  }
  
  async _checkDocument(name, client) {
    await this._waitLoad(name)
    if (!this.documents[name]) {
      this.isLoading[name] = true;
      await this._loadDocument(name);
      if (!this.documents[name] && this.canCreate(client, name)) {
        this._createEmptyDocument(name)
      }
      if (this.documents[name]) {
        if (!this._documentChanges[name]) {
          this._documentChanges[name] = []
        }  
        this._makeReactive(name)
        this.onHydrate(name, this.documents[name])
      }
      this.isLoading[name] = false
    }
  }

  /*= UPDATE =================================================================*/

  // to be redefined. called every 1/20s
  onUpdate(name, doc, dt) {}

  _startLoop() {
    this.lastUpdate = Date.now()
    if (this._loopiv) {
      clearInterval(this._loopiv)
    }
    this._loopiv = setInterval(() => {
      this._loop()
    }, this.CYCLE)
  }

  _loop() {
    let dt = Date.now() - this.lastUpdate
    for (let name in this.documents) {
      this.onUpdate(name, this.documents[name], dt)
    }
    this.lastUpdate = Date.now()
    this._sendPatches()
  }

  /*= MESSAGES ===============================================================*/

  // to be redefined. Called on message (operation) from client
  onMessage(client, message) {}

  // to be redefined. Called when a client connects
  onConnect(client) {}

  // to be redefined. Called when a client disconnects
  onDisconnect(client) {}

  _onConnected(client) {
    client.ID = getUID()
    client.subscribed = {}
    console.log("client connected", client.ID)
    this.clients.add(client)
    client.on("error", () => {
      this._onError(client, arguments)
    })
    client.on("message", (message) => {
      let dec = decode(message)
      this._onMessage(client, dec)
    })
    client.on("close", (message) => {
      this._onDisconnected(client)
      this.onDisconnect(client)
    })
    this.onConnect(client)
  }

  async _onMessage(client, message) {
    if (message.c == "hello") {
      //console.log('client hello')
      this.send(client, { c: "hello", t: Date.now(), ct: message.ct })
    } else if (message.c == "sync" && this.allowSync && client.subscribed[message.n] && this.documents[message.n]) {
      let name = message.n
      if (!this._documentChanges[name]) {
        this._documentChanges[name] = []
      }
      for (let op of message.p) {
        this._documentChanges[name].push(op)
        let dop = msgop(op)
        applyOperation(this.documents[name], dop)
      }
    } else if (message.c == "sub") {
      await this._checkDocument(message.n, client)
      if (!this.documents[message.n]) {
        this.send(client, { c: "error", t: Date.now(), message: "Document not found" })
        return
      }
      client.subscribed[message.n] = true
      this._sendFullState(message.n, client)
    } else if (message.c == "unsub") {
      client.subscribed[message.n] = false
    } else {
      this.onMessage(client, message)
    }
  }

  _onError(client, args) {
    console.error("onError:", args)
  }

  _onDisconnected(client) {
    console.log("client disconnected")
  }

  send(client, message) {
    let enc = encode(message)
    client.send(enc)
  }

  broadcast(message) {
    let enc = encode(message)
    for (let client of this.clients) {
      client.send(enc)
    }
  }

  async _sendFullState(name, client) {
    await this._waitLoad(name)
    let fullState = {
      c: "full",
      le: LITTLE_ENDIAN,
      t: Date.now(),
      n: name,
      doc: clonewo_(this.documents[name]),
    }
    this.send(client, fullState)
  }

  _sendPatches() {
    let now = Date.now()

    for (let name in this._documentChanges) {
      let dc = this._documentChanges[name];
      if (dc.length > 0) {
        let record = {
          c: "patch",
          t: now, // server time
          n: name,
          doc: dc
        }
        for (let client of this.clients) {
          if (client.subscribed[name]) {
            this.send(client, record)
          }
        }
        this._documentChanges[name] = []
      }
    }
  }

  _onDocumentChange(name, op, target, path, value) {
    if (path.indexOf("/_") >= 0) {
      return
    }
    this._documentChanges[name].push(opmsg(op, target, path, value))
  }
  
  async _initDB() {
    await this._connectDB()
  }

  async _connectDB() {
    this.mongoClient = new MongoClient("mongodb://localhost:27017")
    try {
      await this.mongoClient.connect()
      console.log("Connected to MongoDB")
      const db = this.mongoClient.db("topazcube")
      this._DB = db.collection("documents")
    } catch (error) {
      console.error("Error connecting to MongoDB:", error)
      this.mongoClient = null
    }
  }

  async _loadDocument(name) {
    if (this._DB) {
      try {
        const doc = await this._DB.findOne({ name: name })
        if (doc) {
          delete doc._id
          this.documents[name] = doc
        }
      } catch (error) {
        console.error("Error loading document from MongoDB:", error)
      }
    } else {
      console.warn("MongoDB client not initialized. Document not loaded.")
    }
  }
  
  async _saveDocument(name) {
    if (this._DB) {
      try {
        const doc = this.documents[name]
        let newdoc = clonewo_(doc, "__")
        console.log(`Saving document '${name}' to MongoDB`)
        await this._DB.updateOne({ name: name }, { $set: newdoc }, { upsert: true })
        console.log("Document saved to MongoDB")
      } catch (error) {
        console.error("Error saving document to MongoDB:", error)
      }
    } else {
      console.warn("MongoDB client not initialized. Document not saved.")
    }
  }

  async _saveAllDocuments() {
    for (let name in this.documents) {
      await this._saveDocument(name)
    }
  }
  
  async onHydrate(name, document) {
    document._hydrated = true
  }
  
  _exitSignal(signal) {
    if (!this._exited) {
      console.log("\nEXIT: Caught interrupt signal " + signal)
      this._exited = true
      clearInterval(this._loopiv)
      this.onBeforeExit()
      this.broadcast({ server: "Going down" })
      this._saveAllDocuments()
      this.wss.close()
      setTimeout(() => process.exit(0), 1000)
    }
  }

  // Called on program exit

  onBeforeExit() {}
}
