import { applyOperation, applyPatch } from "fast-json-patch"
import { encode, decode } from "@msgpack/msgpack"
import { reactive, opmsg, msgop } from "./utils.js"

export default class TopazCubeClient {
  CYCLE = 100 // update/patch rate in ms
  url = ""
  documents = {}
  autoReconnect = true
  allowSync = true
  isConnected = false
  isConnecting = false
  isPatched = false
  stats = {
    send: 0,
    rec: 0,
    sendBps: 0,
    recBps: 0,
    ping: 0,
    stdiff: 0, // server time difference
  }
  lastFullState = 0
  lastPatch = 0
  le = true // Server is little endian
  _documentChanges = {}
  constructor({
    url, // server url
  }) {
    this.url = url
    this.socket = null
    this._startLoop()
  }

  /*= UPDATE ===================================================================*/

  _startLoop() {
    if (this._loopiv) {
      clearInterval(this._loopiv)
    }
    this._loopiv = setInterval(() => {
      this._loop()
    }, this.CYCLE)
  }

  _loop() {
    if (!this.isConnected) {
      return
    }
    this._sendPatches()
  }

  _countLoop() {
    this._stats.sendBps = this._stats.send
    this._stats.recBps = this._stats.rec
    this._stats.send = 0
    this._stats.rec = 0
  }

  /*= CONNECTION ===============================================================*/

  subscribe(name) {
    this.documents[name] = {}
    this.send({ c: "sub", n: name })
  }

  unsubscribe(name) {
    this.send({ c: "unsub", n: name })
    delete this.documents[name]
  }

  connect() {
    if (this.isConnecting) {
      return
    }
    this.isConnecting = true
    this._clear()
    console.log("connecting...")

    this.socket = new WebSocket(this.url)

    // message received
    this.socket.onmessage = async (event) => {
      let buffer = await event.data.arrayBuffer()
      this.stats.rec += buffer.byteLength
      let message = decode(buffer)
      let time = Date.now()
      if (message.c == "full") {
        let name = message.n
        let doc = message.doc
        this.documents[name] = doc
        this.isPatched = false
        this.document = reactive(name, this.documents[name], this._onDocumentChange.bind(this))
        this.isPatched = false
        this.lastFullState = message.t
        this.le = message.le
        this.onChange(name)
      } else if (message.c == "patch") {
        // patch
        this.lastPatch = message.t
        let name = message.n
        if (message.doc) {
          this.isPatched = true
          for (let op of message.doc) {
            let dop = msgop(op)
            applyOperation(this.documents[name], dop)
          }
          this.isPatched = false
        }
        this.onChange(name)
      } else if (message.c == "hello") {
        time = Date.now()
        let lastct = message.ct
        let ping = time - lastct
        let stime = message.t
        this.stats.stdiff = stime + ping / 2 - time
        this.stats.ping = ping
        console.log("ping", ping, "ms", "stdiff", this.stats.stdiff, "ms")
      }
    }

    // connection closed
    this.socket.onclose = (event) => {
      this.isConnected = false
      this.isConnecting = false
      this.lastFullState = 0
      this.socket = null
      this.onDisconnect()
      if (this.autoReconnect) {
        setTimeout(() => {
          this._reconnect()
        }, 500 + Math.random() * 500)
      }
    }

    this.socket.onerror = (event) => {
      this.isConnected = false
      this.isConnecting = false
      this.lastFullState = 0
      this.socket = null
      this.onDisconnect()

      if (this.autoReconnect) {
        setTimeout(() => {
          this._reconnect()
        }, 500 + Math.random() * 500)
      }
    }

    this.socket.onopen = (event) => {
      this.isConnecting = false
      this.isConnected = true
      this.lastFullState = 0
      this._ping()
      this.onConnect()
    }
  }

  disconnect() {
    this.isConnected = false
    this.isConnecting = false
    this.lastFullState = 0
    this.socket.close()
    this.socket = null
  }

  destroy() {
    this.autoReconnect = false
    this.disconnect()
    this.socket = null
  }

  onConnect() {}

  onDisconnect() {}

  _clear() {
    this.stats.sendBps = 0
    this.stats.recBps = 0
    this.stats.send = 0
    this.stats.rec = 0
    this.documents = {}
    this._documentChanges = {}
    this.lastFullState = 0
    this.lastPatch = 0
    this.isPatched = false
    this.le = true
  }

  _reconnect() {
    if (!this.isConnected) {
      if (!this.isConnecting) {
        this.connect()
      }
    }
  }

  _ping() {
    if (this.isConnected) {
      this.send({ c: "hello", ct: Date.now() })
    }
  }

  /*= MESSAGES =================================================================*/

  onChange(name) {}

  send(operation) {
    try {
      let enc = encode(operation)
      this.stats.send += enc.byteLength
      this.socket.send(enc)
    } catch (e) {
      console.error("send failed", e)
    }
  }

  _onDocumentChange(name, op, target, path, value) {
    if (this.isPatched || !this.allowSync) {
      return
    }
    if (path.indexOf("/_") >= 0) {
      return
    }
    if (!this._documentChanges[name]) {
      this._documentChanges[name] = []
    }
    this._documentChanges[name].push(opmsg(op, target, path, value))
  }

  _sendPatches() {
    for (let name in this._documentChanges) {
      let dc = this._documentChanges[name]
      if (dc.length == 0) {
        continue
      }
      let record = {
        n: name,
        c: "sync",
        ct: Date.now(),
      }

      if (dc.length > 0) {
        record.p = dc
      }
      this.send(record)
      this._documentChanges[name].length = 0
      this.onChange(name)
    }
  }

}
