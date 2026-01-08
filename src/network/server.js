import * as https from 'node:https'
import * as fs from 'node:fs'
import {
  reactive,
  deepGet,
  clonewo_,
  opmsg,
  msgop,
  encode_uint32,
  encode_fp412,
  encode_fp168,
  encode_fp1616,
  limitPrecision,
  encode,
  decode
} from './utils.js'
import { compress, decompress } from './compress-node.js'
import fastjsonpatch from 'fast-json-patch'
import { WebSocketServer } from 'ws'
import { MongoClient } from 'mongodb'
import { glMatrix, vec3, quat } from 'gl-matrix'
import { RTCPeerConnection, RTCSessionDescription } from "werift"

glMatrix.setMatrixArrayType(Array)

const fastPatchProperties = {
  'type': true,
  'status': true,
  'level': true,
  'race': true,
  'class': true,
  'model': true,
  'animation': true,
  'sprite': true,
  'frame': true,
  'pivot': true, // 'center' | 'bottom' | 'horizontal'
  'color': true,
  'sound': true,
  'effect': true,
  'position': true,
  'rotation': true,
  'scale': true,
}

const dictionaryProperties = {
  'type': true,
  'status': true,
  'level': true,
  'race': true,
  'class': true,
  'model': true,
  'animation': true,
  'sprite': true,
  'frame':true,
  'pivot':true,
  'color': true,
  'sound': true,
  'effect': true,
}

const { applyOperation } = fastjsonpatch
const LITTLE_ENDIAN = (() => {
  const buffer = new ArrayBuffer(2)
  new DataView(buffer).setInt16(0, 256, true)
  return new Int16Array(buffer)[0] === 256
})()

const MAX_PACKAGE_SIZE = 65400

export default class TopazCubeServer {
  DEBUG = false
  name = 'TopazCubeServer'
  cycle = 100
  saveCheckCycle = 1000
  saveCycle = 10000
  patchCycleDivider = 1
  port = 8799
  useHttps = false
  key = './cert/key.pem'
  cert = './cert/cert.pem'
  MongoUrl = 'mongodb://localhost:27017'
  mongoClient = null
  DB = null
  database = 'topazcube'
  collection = 'documents'
  allowSave = true
  allowSync = true
  allowWebRTC = false
  allowFastPatch = false
  allowCompression = false
  simulateLatency = 0
  _lastUID = 100
  clients = []
  documents = {}
  isLoading = {}
  _documentChanges = {}
  _documentChanged = {}
  _documentState = {}

  update = 0
  lastUpdate = 0
  _saveiv = null
  _lastSave = {}
  _loopiv = null
  _statsiv = null
  _stillUpdating = false
  stats = {
    tUpdate: [],
    tPatch: [],
    send: 0,
    sendRTC: 0,
    _sendRTCUpdate: 0
  }

  _wss = null
  _exited = false

  log(...args) {
    if (this.DEBUG) {
      console.log(this.name + ':', ...args)
    }
  }

  warn(...args) {
    if (this.DEBUG) {
      console.warn(this.name + ':', ...args)
    }
  }

  error(...args) {
    console.error(this.name + ':', ...args)
  }

  constructor({
    name = 'TopazCubeServer',
    cycle = 100,
    saveCheckCycle = 1000,
    saveCycle = 10000,
    port = 8799,
    useHttps = false,
    key = './cert/key.pem',
    cert = './cert/cert.pem',
    MongoUrl = 'mongodb://localhost:27017',
    database = 'topazcube',
    collection = 'documents',
    allowSave = true,
    allowSync = true,
    allowWebRTC = false,
    allowFastPatch = false,
    allowCompression = false,
    simulateLatency = 0,
    DEBUG = false
  } = {}) {
    this.name = name
    this.cycle = cycle
    this.saveCheckCycle = saveCheckCycle
    this.saveCycle = saveCycle
    this.port = port
    this.useHttps = useHttps
    this.key = key
    this.cert = cert
    this.MongoUrl = MongoUrl
    this.database = database
    this.collection = collection
    this.allowSave = allowSave
    this.allowSync = allowSync
    this.allowWebRTC = allowWebRTC
    this.allowFastPatch = allowFastPatch
    this.allowCompression = allowCompression
    this.simulateLatency = simulateLatency
    this.DEBUG = DEBUG

    this._initDB()

    if (useHttps) {
      let httpsServer = https.createServer({
        key: fs.readFileSync(this.key),
        cert: fs.readFileSync(this.cert),
      }, (req, res) => {
        res.writeHead(200)
        res.end('<b>Hello World!</b>')
      }).listen(this.port)
      this._wss = new WebSocketServer({ server: httpsServer })
      httpsServer = null
      this.log(this.name + ' running on HTTPS port ' + this.port)
    } else {
      this._wss = new WebSocketServer({ port: this.port })
      this.log(this.name + ' running on port ' + this.port)
    }
    this._wss.on('connection', (client) => {
      this._onConnected(client)
    })

    process.stdin.resume()
    process.on('SIGINT', () => {
      this._exitSignal('SIGINT')
    })
    process.on('SIGQUIT', () => {
      this._exitSignal('SIGQUIT')
    })
    process.on('SIGTERM', () => {
      this._exitSignal('SIGTERM')
    })
    process.on('SIGUSR2', () => {
      this._exitSignal('SIGUSR2')
    })

    process.stdin.resume()
    process.stdin.setEncoding('utf8')

    process.stdin.on('data', (key) => {
      key = ('' + key).trim()

      if (key == '\u0003') {
        this._exitSignal('SIGINT')
        return
      }

      this.log(`Key pressed: ${key}`)

      if (key == 's') {
        this.log('Saving all documents...')
        this._saveAllDocuments()
      }

      if (key == 'i') {
        this.log(
          `Server: ${this.name}, Clients: ${this.clients.length}, Documents: ${Object.keys(this.documents).length}`
        )
      }
    })
    this._startLoop()
  }

  canCreate(client, name) {
    return true
  }

  onCreate(name) {
    return {
      data: {},
    }
  }

  canSync(client, name, op) {
    return true
  }

  async onHydrate(name, document) {
    document.__hydrated = true
  }

  _makeReactive(name) {
    let ep = false
    if (this.allowFastPatch) {
      ep = fastPatchProperties
    }
    this.documents[name] = reactive(
      name,
      this.documents[name],
      this._onDocumentChange.bind(this),
      '',
      ep
    )
    if (!this._documentChanges[name]) {
      this._documentChanges[name] = []
      this._documentChanged[name] = false
    }
  }

  _createEmptyDocument(name) {
    const doc = this.onCreate(name)
    if (!doc) {
      return
    }
    this.documents[name] = doc
  }

  async _waitLoad(name) {
    if (this.isLoading[name]) {
      while (this.isLoading[name]) {
        await new Promise((resolve) => setTimeout(resolve, 50))
      }
    }
  }

  async _checkDocument(name, client) {
    await this._waitLoad(name)
    if (!this.documents[name]) {
      this.isLoading[name] = true
      await this._loadDocument(name)
      if (!this.documents[name] && this.canCreate(client, name)) {
        this._createEmptyDocument(name)
      }
      if (this.documents[name]) {
        this._makeReactive(name)
        this.onHydrate(name, this.documents[name])
      }
      this.isLoading[name] = false
      this._documentState[name] = {
        subscibers: 0,
        lastModified: Date.now(),
      }
    }
  }

  _updateAllDocumentsState() {
    for (const name in this.documents) {
      if (name != '_server') {
        const doc = this.documents[name]
        this._documentState[name].subscibers = 0
        for (const client of this.clients) {
          if (client.subscribed && client.subscribed[name]) {
            this._documentState[name].subscibers++
          }
        }
      }
    }
  }

  onUpdate(name, doc, dt) {}

  _startLoop() {
    this.lastUpdate = Date.now()
    this._loop()
    this._statsiv = setInterval(() => {
      this._doStats()
    }, 1000)
    this._saveiv = setInterval(() => {
      this._saveChanges()
    }, 1000)
  }

  _loop() {
    const now = Date.now()
    const dtms = (now - this.lastUpdate)
    const dt = dtms / 1000.0
    this.lastUpdate = now

    this._stillUpdating = true
    for (const name in this.documents) {
      this.onUpdate(name, this.documents[name], dt)
    }
    const t1 = Date.now()
    this._stillUpdating = false
    const updateTime = t1 - now
    this.stats.tUpdate.push(updateTime)

    let patchTime = 0
    if (this.update % this.patchCycleDivider == 0) {
      this._sendPatches()
      const t2 = Date.now()
      patchTime = t2 - t1
      this.stats.tPatch.push(patchTime)
      if (this.allowFastPatch) {
        this.log(`update ${this.update} dt:${dtms}ms RTC:${this.stats._sendRTCUpdate}bytes, tUpdate: ${updateTime}ms, tPatch: ${patchTime}ms`)
      }
      this.stats._sendRTCUpdate = 0
    }

    this.update++
    const endUpdate = Date.now()
    const totalUpdate = endUpdate - now

    setTimeout(() => {
      this._loop()
    }, Math.max(this.cycle - totalUpdate, 10))
  }

  _doStats() {
    for (const key in this.stats) {
      const i = this.stats[key]
      if (Array.isArray(i) && i.length > 0) {
        while (i.length > 60) {
          i.shift()
        }
        this.stats['_avg_' + key] = i.reduce((a, b) => a + b, 0) / i.length
      } else if (!key.startsWith('_')) {
        this.stats['_persec_' + key] = i / 3.0
        this.stats[key] = 0
      }
    }
  }

  onMessage(client, message) {}

  onConnect(client) {}

  onDisconnect(client) {}

  _onConnected(client) {
    client.ID = this.getUID()
    client.ping = 0
    client.ctdiff = 0
    client.subscribed = {}
    client.dataChannel = null
    client.peerConnection = null

    this.log('client connected', client.ID)
    this.clients.push(client)
    client.on('error', (...args) => {
      this._onError(client, args)
    })
    client.on('message', (message) => {
      const dec = decode(message)
      if (this.simulateLatency) {
        setTimeout(() => {
          this._onMessage(client, dec)
        }, this.simulateLatency)
      } else {
        this._onMessage(client, dec)
      }
    })
    client.on('close', (message) => {
      this._onDisconnected(client)
      this.onDisconnect(client)
    })
    this.onConnect(client)
  }

  async _onMessage(client, message) {
    if (
      message.c == 'sync' &&
      this.allowSync &&
      client.subscribed &&
      client.subscribed[message.n] &&
      this.documents[message.n]
    ) {
      const name = message.n
      if (!this._documentChanges[name]) {
        this._documentChanges[name] = []
        this._documentChanged[name] = false
      }
      for (const op of message.p) {
        if (!this.canSync(client, name, op)) {
          continue
        }
        this._documentChanges[name].push(op)
        this._documentChanged[name] = true
        const dop = msgop(op)
        applyOperation(this.documents[name], dop)
      }
    } else if (message.c == 'ping') {
      this.send(client, {
        c: 'pong',
        t: Date.now(),
        ct: message.ct,
        ID: client.ID,
      })
    } else if (message.c == 'peng') {
      const time = Date.now()
      const ping = time - message.st
      client.ctdiff = message.ct + ping / 2 - time
      client.ping = ping
    } else if (message.c == 'rtc-offer') {
      this._processOffer(client, message)
    } else if (message.c == 'rtc-candidate') {
      this._processICECandidate(client, message)
    } else if (message.c == 'sub') {
      await this._checkDocument(message.n, client)
      if (!this.documents[message.n]) {
        this.send(client, {
          c: 'error',
          t: Date.now(),
          message: 'Document not found',
        })
        return
      }
      if (client.subscribed) {
        client.subscribed[message.n] = true
      }
      this._sendFullState(message.n, client)
    } else if (message.c == 'unsub') {
      if (client.subscribed) {
        client.subscribed[message.n] = false
      }
    } else {
      this.onMessage(client, message)
    }
  }

  _onError(client, args) {
    this.error('onError:', args)
  }

  _onDisconnected(client) {
    if (client.dataChannel) {
      client.dataChannel.close()
    }
    if (client.peerConnection) {
      client.peerConnection.close()
    }
    this.log('client disconnected')
    const index = this.clients.indexOf(client)
    if (index !== -1) {
      this.clients.splice(index, 1)
    }
  }

  async send(client, message) {
    try {
      const t1 = Date.now()
      let data = encode(message)
      const t2 = Date.now()
      const dl = data.byteLength
      if (this.allowCompression) {
        data = await compress(data)
      }
      const t3 = Date.now()
      if (data.length > 4096) {
        this.log(`Big message ${dl} -> ${data.length} (${(100.0 * data.length / dl).toFixed()}%) encoding:${t2 - t1}ms compression:${t3 - t1}ms`)
      }
      this.stats.send += data.byteLength
      if (this.simulateLatency) {
        setTimeout(() => {
          client.send(data)
        }, this.simulateLatency)
      } else {
        client.send(data)
      }
    } catch (e) {
      this.error('Error sending message:', e, message)
    }
  }

  async broadcast(message, clients = false) {
    if (!clients) {
      clients = this.clients
    }
    let data = encode(message)
    if (this.allowCompression) {
      data = await compress(data)
    }
    for (const client of this.clients) {
      this.stats.send += data.byteLength
      if (this.simulateLatency) {
        setTimeout(() => {
          client.send(data)
        }, this.simulateLatency)
      } else {
        client.send(data)
      }
    }
  }

  async _sendFullState(name, client) {
    await this._waitLoad(name)
    let excluded = '_'
    if (this.allowFastPatch) {
      excluded = fastPatchProperties
    }
    const doc = clonewo_(this.documents[name], excluded)
    limitPrecision(doc)
    let fdata = false
    if (this.allowFastPatch) {
      fdata = this._encodeFastChanges(name, false)
    }
    const fullState = {
      c: 'full',
      le: LITTLE_ENDIAN,
      t: Date.now(),
      n: name,
      doc: doc,
      fdata: fdata
    }
    this.send(client, fullState)
  }

  _encodeFastChanges(name, changesOnly = true) {
    const doc = this.documents[name]
    if (!doc) { return false }
    let origin = this.documents[name].origin
    if (!origin) {
      origin = [0, 0, 0]
      this.documents[name].origin = origin
    }

    const entities = doc.entities
    const ids = Object.keys(entities)
    if (!entities) { return false }
    const count = {}
    const changed = {}
    const hasChanges = {}
    const dictionary = {}
    const encodedChanges = {}

    for (const key in fastPatchProperties) {
      if (changesOnly) {
        count[key] = 0
        changed[key] = {}
        hasChanges[key] = false
      } else {
        count[key] = ids.length
        changed[key] = {}
        hasChanges[key] = true
      }
      dictionary[key] = {}
    }

    if (changesOnly) {
      for (const id in entities) {
        const e = entities[id]
        for (const key in fastPatchProperties) {
          if (e['__changed_' + key]) {
            changed['' + key]['' + id] = true
            count['' + key] = parseInt('' + count['' + key]) + 1
            hasChanges['' + key] = true
            e['__changed_' + key] = false
          }
        }
      }
    } else {
      for (const id in entities) {
        for (const key in fastPatchProperties) {
          changed['' + key]['' + id] = true
        }
      }
    }

    let dictUID = 1
    for (const key in hasChanges) {
      if (hasChanges[key] && dictionaryProperties[key]) {
        for (const id in changed[key]) {
          const e = entities[id]
          const value = e[key]
          if (!dictionary[key][value]) {
            dictionary[key][value] = dictUID++
          }
        }
      }
    }

    this.log("--------------------------------------------------")

    for (const key in hasChanges) {
      if (hasChanges[key]) {
        const size = parseInt('' + count['' + key])
        const encoded = {}
        if (dictionaryProperties[key]) {
          encoded.dict = dictionary[key]

          const pdata = new Uint8Array(size * 8)
          let offset = 0
          for (const id in changed[key]) {
            const e = entities[id]
            const nid = parseInt(id)
            encode_uint32(nid, pdata, offset)
            offset += 4
            const value = e[key]
            const did = parseInt(dictionary[key][value])
            encode_uint32(did, pdata, offset)
            offset += 4
          }
          encoded.pdata = pdata
        } else {
          let pdata
          if (key == 'position') {
            pdata = new Uint8Array(size * 13)
          } else if (key == 'rotation') {
            pdata = new Uint8Array(size * 16)
          } else if (key == 'scale') {
            pdata = new Uint8Array(size * 16)
          } else {
            pdata = new Uint8Array(0)
          }

          let offset = 0
          for (const id in changed[key]) {
            const e = entities[id]
            const nid = parseInt(id)
            encode_uint32(nid, pdata, offset)
            offset += 4
            if (key == 'position') {
              encode_fp168(e.position[0] - origin[0], pdata, offset)
              offset += 3
              encode_fp168(e.position[1] - origin[1], pdata, offset)
              offset += 3
              encode_fp168(e.position[2] - origin[2], pdata, offset)
              offset += 3
            } else if (key == 'rotation') {
              encode_fp412(e.rotation[0], pdata, offset)
              offset += 2
              encode_fp412(e.rotation[1], pdata, offset)
              offset += 2
              encode_fp412(e.rotation[2], pdata, offset)
              offset += 2
              encode_fp412(e.rotation[3], pdata, offset)
              offset += 2
            } else if (key == 'scale') {
              encode_fp1616(e.scale[0], pdata, offset)
              offset += 4
              encode_fp1616(e.scale[1], pdata, offset)
              offset += 4
              encode_fp1616(e.scale[2], pdata, offset)
              offset += 4
            }
          }
          encoded.pdata = pdata
        }
        encodedChanges[key] = encoded
      }
    }

    return encodedChanges
  }

  _sendPatches() {
    const now = Date.now()

    for (const name in this._documentChanges) {
      const dc = this._documentChanges[name]
      this._documentChanges[name] = []
      const sus = this.clients.filter((client) => client.subscribed && client.subscribed[name])
      if (sus.length > 0) {
        if (dc && dc.length > 0) {
          const record = {
            c: 'patch',
            t: now,
            u: this.update,
            n: name,
            doc: dc,
          }
          this.broadcast(record, sus)
        }
      }

      if (this.allowFastPatch) {
        if (sus.length > 0) {
          const t1 = Date.now()
          const changes = this._encodeFastChanges(name)
          const t2 = Date.now()
          const record = {
            c: 'fpatch',
            t: now,
            u: this.update,
            n: name,
            fdata: changes
          }
          this.broadcastRTC(record, sus)
          const t3 = Date.now()
          this.log(`_sendPatches: ${name} encode_changes: ${t2 - t1}ms broadcast:${t3 - t2}ms`)
        }
      }
    }
  }

  _onDocumentChange(name, op, target, path, value) {
    this._documentChanges[name]?.push(opmsg(op, target, path, value))
    this._documentChanged[name] = true
  }

  propertyChange(name, id, property) {
    const doc = this.documents[name]
    if (!doc) { return }
    const entities = doc.entities
    if (!entities) { return }
    const e = entities[id]
    if (!e) { return }
    e['__changed_' + property] = true
  }

  async _processOffer(client, data) {
    if (!this.allowWebRTC) {
      this.warn('WebRTC is disabled')
      return
    }

    this.log("RTC: Processing offer from client", client.ID, data)

    const peerConnection = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun.cloudflare.com:3478' },
        { urls: 'stun:freestun.net:3478' },
      ],
    })

    client.peerConnection = peerConnection

    peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        this.log("RTC: ICE candidate generated", event.candidate.candidate.substring(0, 50) + "...")
        this.send(client, {
          c: 'rtc-candidate',
          type: 'ice-candidate',
          candidate: event.candidate,
        })
      } else {
        this.log("RTC: ICE candidate gathering complete")
      }
    }

    peerConnection.onconnectionstatechange = () => {
      this.log(`RTC: Connection state changed: ${peerConnection.connectionState}`)
      if (peerConnection.connectionState === 'connected') {
        client.webRTCConnected = true
        this.log(`RTC: Connection established with client ${client.ID}`)
      } else if (
        peerConnection.connectionState === 'failed' ||
        peerConnection.connectionState === 'disconnected' ||
        peerConnection.connectionState === 'closed'
      ) {
        client.webRTCConnected = false
        this.log(`RTC: Connection failed or closed with client ${client.ID}`)
      }
    }

    peerConnection.onicegatheringstatechange = () => {
      this.log(`RTC: ICE gathering state: ${peerConnection.iceGatheringState}`)
    }

    peerConnection.oniceconnectionstatechange = () => {
      this.log(`RTC: ICE connection state: ${peerConnection.iceConnectionState}`)
      if (
        peerConnection.iceConnectionState === 'connected' ||
        peerConnection.iceConnectionState === 'completed'
      ) {
        this.log(`RTC: ICE connection established with client ${client.ID}`)
      }
    }

    try {
      this.log("RTC: Remote description set from data", data)
      await peerConnection.setRemoteDescription(
        new RTCSessionDescription(data.sdp, data.type)
      )
      this.log("RTC: Remote description set successfully")

      client.dataChannel = peerConnection.createDataChannel('serverchannel', {
        ordered: true,
        maxRetransmits: 1,
      })

      client.dataChannel.onopen = () => {
        this.log(`RTC: Data channel opened for client ${client.ID}`)
        try {
          const testData = { c: 'test', message: 'Hello WebRTC' }
          this.sendRTC(client, testData)
        } catch (e) {
          this.error(
            `RTC: Error sending test message to client ${client.ID}`,
            e
          )
        }
      }

      client.dataChannel.onclose = () => {
        this.log(`RTC: Data channel closed for client ${client.ID}`)
      }

      client.dataChannel.onerror = (error) => {
        this.error(`RTC: Data channel error for client ${client.ID}:`, error)
      }

      client.dataChannel.onmessage = (event) => {
        try {
          const data = decode(event.data)
          this.log(
            `RTC: Data channel message from client ${client.ID}:`,
            data
          )
        } catch (error) {
          this.error(
            `RTC: Error decoding message from client ${client.ID}:`,
            error
          )
        }
      }

      const answer = await peerConnection.createAnswer()
      await peerConnection.setLocalDescription(answer)

      this.log(`RTC: Sending answer to client ${client.ID}`)
      this.send(client, {
        c: 'rtc-answer',
        type: answer.type,
        sdp: answer.sdp,
      })
    } catch (error) {
      this.error(
        `RTC: Error processing offer from client ${client.ID}:`,
        error
      )
    }
  }

  async _processICECandidate(client, data) {
    this.log(`RTC: Processing ICE candidate from client ${client.ID}`)
    try {
      if (data.candidate && typeof (data.candidate) == 'object') {
        data.candidate = data.candidate.candidate
      }
      if (client.peerConnection && data.candidate) {
        await client.peerConnection.addIceCandidate(
          data.candidate
        )
        this.log(`RTC: ICE candidate added successfully for client ${client.ID}`)
      } else {
        this.warn(`RTC: Cannot add ICE candidate for client ${client.ID} - peerConnection not ready or candidate missing`, client.peerConnection, data)
      }
    } catch (error) {
      this.error(`RTC: Error adding ICE candidate for client ${client.ID}`)
    }
  }

  _clientRTCOpen(client) {
    return client.dataChannel !== null && client.dataChannel !== undefined && client.dataChannel.readyState === 'open'
  }

  async sendRTC(client, message) {
    let data = encode(message)
    if (this.allowCompression) {
      data = await compress(data)
    }
    this.stats.sendRTC += data.byteLength
    this.stats._sendRTCUpdate += data.byteLength

    const packages = this._splitRTCMessage(data)

    if (this.simulateLatency) {
      setTimeout(() => {
        if (this._clientRTCOpen(client)) {
          packages.forEach((p) => {
            client.dataChannel.send(p)
          })
        }
      }, this.simulateLatency)
    } else {
      if (this._clientRTCOpen(client)) {
        packages.forEach((p) => {
          client.dataChannel.send(p)
        })
      }
    }
  }

  async broadcastRTC(message, clients = []) {
    if (clients.length == 0) {
      clients = this.clients
    }
    const t1 = Date.now()
    let data = encode(message)
    const dl = data.byteLength
    const t2 = Date.now()
    if (this.allowCompression) {
      data = await compress(data)
    }
    const t3 = Date.now()

    if (data.length > 16384) {
      this.log(`BroadcastRTC message ${dl} -> ${data.length} (${(100.0 * data.length / dl).toFixed()}%) encoding:${t2 - t1}ms compression:${t3 - t1}ms`)
    }

    const packages = this._splitRTCMessage(data)

    for (const client of this.clients) {
      this.stats.sendRTC += data.byteLength
      this.stats._sendRTCUpdate += data.byteLength
      if (this.simulateLatency) {
        setTimeout(() => {
          if (client.dataChannel && client.dataChannel.readyState === 'open') {
            packages.forEach((p) => {
              client?.dataChannel?.send(p)
            })
          }
        }, this.simulateLatency)
      } else {
        if (client.dataChannel && client.dataChannel.readyState === 'open') {
          packages.forEach((p) => {
            client?.dataChannel?.send(p)
          })
        }
      }
    }
  }

  _splitRTCMessage(data) {
    let packages
    if (data.byteLength > 65535) {
      const now = Date.now()
      this.warn(`RTC: Message too large: ${data.byteLength} bytes`)
      packages = []
      let offset = 0
      const mid = this.update + '-' + now
      let seq = 0

      while (offset < data.byteLength) {
        const remaining = data.byteLength - offset
        const chunkSize = Math.min(remaining, MAX_PACKAGE_SIZE)
        const chunk = new Uint8Array(data.buffer, offset, chunkSize)
        const cmessage = {
          c: 'chunk',
          t: now,
          mid: mid,
          seq: seq,
          ofs: offset,
          chs: chunkSize,
          ts: data.byteLength,
          data: chunk,
          last: remaining <= MAX_PACKAGE_SIZE,
        }
        packages.push(encode(cmessage))
        offset += chunkSize
        seq++
      }

      this.log(`RTC: Large message split into ${packages.length} packages`)
    } else {
      packages = [data]
      this.log(`RTC: Message - ${data.byteLength} bytes`)
    }
    return packages
  }

  getUID() {
    this.documents['_server'].nextUID++
    return this.documents['_server'].nextUID
  }

  async _initDB() {
    await this._connectDB()
    await this._loadDocument('_server')
    if (!this.documents['_server']) {
      this._initServerDocument()
    }
  }

  async _connectDB() {
    this.mongoClient = new MongoClient(this.MongoUrl)
    try {
      await this.mongoClient.connect()
      this.log('Connected to MongoDB')
      const db = this.mongoClient.db(this.database)
      this.DB = db
    } catch (error) {
      this.error('Error connecting to MongoDB:', error)
      this.mongoClient = null
    }
  }

  async _loadDocument(name) {
    this.log(`Loading document '${name}' from MongoDB`)
    if (this.DB) {
      try {
        const doc = await this.DB.collection(this.collection).findOne({
          name: name,
        })
        if (doc) {
          delete doc._id
          this.documents[name] = doc
        }
      } catch (error) {
        this.error('Error loading document from MongoDB:', error)
      }
    } else {
      this.warn('MongoDB client not initialized. Document not loaded.')
    }
  }

  async _saveDocument(name) {
    if (this.DB) {
      try {
        const doc = this.documents[name]
        const newdoc = clonewo_(doc, '__')
        this.log(`Saving document '${name}' to MongoDB`)
        await this.DB.collection(this.collection).updateOne(
          { name: name },
          { $set: newdoc },
          { upsert: true }
        )
        this.log('Document saved to MongoDB')
      } catch (error) {
        this.error('Error saving document to MongoDB:', error)
      }
    } else {
      this.warn('MongoDB client not initialized. Document not saved.')
    }
  }

  async _saveAllDocuments() {
    if (!this.allowSave) {
      return
    }
    for (const name in this.documents) {
      await this._saveDocument(name)
    }
  }

  async _saveChanges() {
    if (!this.allowSave) {
      return
    }
    for (const name in this._documentChanged) {
      if (!this._lastSave[name]) {
        this._lastSave[name] = Date.now() - 60000
      }
      const lastSave = this._lastSave[name]
      if (Date.now() - lastSave < 10000) {
        continue
      }
      if (this._documentChanged[name]) {
        await this._saveDocument(name)
        this._documentChanged[name] = false
        this._lastSave[name] = Date.now()
      }
    }
  }

  _initServerDocument() {
    this.documents['_server'] = {
      nextUID: 100,
    }
  }

  _exitSignal(signal) {
    if (!this._exited) {
      this.log('\nEXIT: Caught interrupt signal ' + signal)
      this._exited = true
      clearInterval(this._loopiv)
      clearInterval(this._statsiv)
      clearInterval(this._saveiv)
      this.onBeforeExit()
      this.broadcast({ server: 'Going down' })
      this._saveAllDocuments()
      this._wss.close()
      setTimeout(() => process.exit(0), 1000)
    }
  }

  onBeforeExit() {}
}
