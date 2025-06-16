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
} from './utils'
import { compress, decompress } from './compress-node'
import fastjsonpatch from 'fast-json-patch'
import { WebSocketServer, WebSocket } from 'ws'
import { MongoClient, Db } from 'mongodb'
import * as wrtc from '@roamhq/wrtc' // Server-side WebRTC implementation
import { doesNotThrow } from 'assert'
import { glMatrix, vec3, quat } from 'gl-matrix'
glMatrix.setMatrixArrayType(Array)

// entities/ID/
const fastPatchProperties: Record<string, boolean> = {
  'type': true, // string 'enemy'
  'status': true, // string 'idle'
  'level': true, // number 2
  'race': true, // string 'goblin'
  'class': true, // string 'warrior'
  'model': true, // string 'models/models.glb|goblin'
  'animation': true, // string 'idle2'
  'sound': true, // string 'sound/goblin.snd|snarl'
  'effect': true, // 'selected'
  'position': true, // [0, 0, 0] Vector (Number)
  'rotation': true, // [0, 0, 0, 1] Quaternion (Number)
  'scale': true, // [1, 1, 1] Vector (Number)
}

const dictionaryProperties: Record<string, boolean> = {
  'type': true,
  'status': true,
  'level': true,
  'race': true,
  'class': true,
  'model': true,
  'animation': true,
  'sound': true,
  'effect': true,
}

const { applyOperation } = fastjsonpatch
const LITTLE_ENDIAN = (() => {
  const buffer = new ArrayBuffer(2)
  new DataView(buffer).setInt16(0, 256, true)
  return new Int16Array(buffer)[0] === 256
})()

const MAX_PACKAGE_SIZE = 65400; // Slightly below the 65535 limit to allow for overhead


type ClientType = any

interface StatsType {
  tUpdate: number[]
  tPatch: number[]
  send: number
  sendRTC: number
  _sendRTCUpdate: number
  [key: string]: any
}

export default class TopazCubeServer {
  name = 'TopazCubeServer'
  cycle = 100
  patchCycleDivider = 1
  port = 8799
  useHttps = false
  key = './cert/key.pem'
  cert = './cert/fullchain.pem'
  MongoUrl = 'mongodb://localhost:27017'
  mongoClient: MongoClient | null = null
  DB: Db | null = null
  database = 'topazcube'
  collection = 'documents'
  allowSave = true
  allowSync = true
  allowWebRTC = false
  allowFastPatch = false
  allowCompression = false
  simulateLatency = 0

  _lastUID = 100
  clients: ClientType[] = []
  documents: Record<string, any> = {}
  isLoading: Record<string, boolean> = {}
  _documentChanges: Record<string, any[]> = {}
  _documentState: Record<string, any> = {}

  update = 0
  lastUpdate = 0
  _loopiv: any = null
  _statsiv: any = null
  _stillUpdating = false
  stats: StatsType = {
    tUpdate: [],
    tPatch: [],
    send: 0,
    sendRTC: 0,
    _sendRTCUpdate: 0
  }

  _wss: WebSocketServer | null = null
  _exited = false

  constructor({
    name = 'TopazCubeServer',
    cycle = 100,
    port = 8799,
    useHttps = false,
    key = './cert/key.pem',
    cert = './cert/fullchain.pem',
    MongoUrl = 'mongodb://localhost:27017',
    database = 'topazcube',
    collection = 'documents',
    allowSave = true,
    allowSync = true,
    allowWebRTC = false,
    allowFastPatch = false,
    allowCompression = false,
    simulateLatency = 0,
  }: {
    name?: string
    cycle?: number
    port?: number
    useHttps?: boolean
    key?: string
    cert?: string
    MongoUrl?: string
    database?: string
    collection?: string
    allowSave?: boolean
    allowSync?: boolean
    allowWebRTC?: boolean
    allowFastPatch?: boolean
    allowCompression?: boolean
    simulateLatency?: number
  } = {}) {
    this.name = name
    this.cycle = cycle
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

    this._initDB()

    if (useHttps) {
      let httpsServer: https.Server | null = https.createServer({
        key: fs.readFileSync(this.key),
        cert: fs.readFileSync(this.cert),
      }, (req, res) => {
        res.writeHead(200)
        res.end('<b>Hello World!</b>')
      }).listen(this.port)
      this._wss = new WebSocketServer({ server: httpsServer })
      httpsServer = null
      console.log(this.name + ' running on HTTPS port ' + this.port)
    } else {
      this._wss = new WebSocketServer({ port: this.port })
      console.log(this.name + ' running on port ' + this.port)
    }
    this._wss.on('connection', (client: WebSocket) => {
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

    // Setup keypress handling for console input
    process.stdin.resume()
    process.stdin.setEncoding('utf8')

    process.stdin.on('data', (key: any) => {
      key = (''+key).trim()

      // ctrl-c ( end of text )
      if (key == '\u0003') {
        this._exitSignal('SIGINT')
        return
      }

      // Process other keypresses
      console.log(`Key pressed: ${key}`)

      // Example: 's' to save all documents
      if (key == 's') {
        console.log('Saving all documents...')
        this._saveAllDocuments()
      }

      // Example: 'i' to print server info
      if (key == 'i') {
        console.log(
          `Server: ${this.name}, Clients: ${this.clients.length}, Documents: ${Object.keys(this.documents).length}`
        )
      }
    })
    this._startLoop()
  }

  /*= DOCUMENTS ==============================================================*/

  // to be redefined. Called before a new document is created. Returns true if
  // the client has the right to create an empty document
  canCreate(client: ClientType, name: string): boolean {
    return true
  }

  // to be redefined. Called when a new document is created
  // (returns an empty document)
  onCreate(name: string): any {
    return {
      data: {},
    }
  }

  // to be redefined. Called when a client wants to sync (modify) a document.
  // Returns true if the client has the right to sync that operation.
  canSync(client: ClientType, name: string, op: any): boolean {
    return true
  }

  // to be redefined. Called when a new document is hydrated
  // (created, or loaded from db)
  async onHydrate(name: string, document: any): Promise<void> {
    document.__hydrated = true
  }

  _makeReactive(name: string): void {
    //console.log(`Making document '${name}' reactive`, this.documents[name])
    let ep: any = false
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
    }
  }

  _createEmptyDocument(name: string): void {
    let doc = this.onCreate(name)
    if (!doc) {
      return
    }
    this.documents[name] = doc
  }

  async _waitLoad(name: string): Promise<void> {
    if (this.isLoading[name]) {
      while (this.isLoading[name]) {
        await new Promise((resolve) => setTimeout(resolve, 50))
      }
    }
  }

  async _checkDocument(name: string, client: ClientType): Promise<void> {
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

  _updateAllDocumentsState(): void {
    for (let name in this.documents) {
      if (name != '_server') {
        let doc = this.documents[name]
        this._documentState[name].subscibers = 0
        for (let client of this.clients) {
          if (client.subscribed && client.subscribed[name]) {
            this._documentState[name].subscibers++
          }
        }
      }
    }
  }

  /*= UPDATE LOOP ============================================================*/

  // to be redefined. called every this.cycle ms
  onUpdate(name: string, doc: any, dt: number): void {}

  _startLoop(): void {
    this.lastUpdate = Date.now()
    this._loop()
    this._statsiv = setInterval(() => {
      this._doStats()
    }, 1000)
  }

  _loop(): void {
    let now = Date.now()
    let dtms = (now - this.lastUpdate)
    let dt = dtms / 1000.0 // Convert to seconds
    this.lastUpdate = now

    /*
    if (this._stillUpdating) {
      return
    }
    */
    this._stillUpdating = true
    for (let name in this.documents) {
      this.onUpdate(name, this.documents[name], dt)
    }
    let t1 = Date.now()
    this._stillUpdating = false
    let updateTime = t1 - now
    this.stats.tUpdate.push(updateTime)

    //console.log(`update ${this.update} patch: ${this.update % this.patchCycleDivider}`, )

    let patchTime = 0
    if (this.update % this.patchCycleDivider == 0) {
      this._sendPatches()
      let t2 = Date.now()
      patchTime = t2 - t1
      this.stats.tPatch.push(patchTime)
      if (this.allowFastPatch) {
        console.log(`update ${this.update} dt:${dtms}ms RTC:${this.stats._sendRTCUpdate}bytes, tUpdate: ${updateTime}ms, tPatch: ${patchTime}ms`, )
      }
      this.stats._sendRTCUpdate = 0
    }

    this.update++
    let endUpdate = Date.now()
    let totalUpdate = endUpdate - now

    setTimeout(() => {
      this._loop()
    }, Math.max(this.cycle - totalUpdate, 10))
  }

  _doStats(): void {
    for (let key in this.stats) {
      let i = this.stats[key]
      if (Array.isArray(i) && i.length > 0) {
        while (i.length > 60) {
          i.shift()
        }
        this.stats['_avg_' + key] = i.reduce((a: number, b: number) => a + b, 0) / i.length
      } else if (!key.startsWith('_')) {
        this.stats['_persec_' + key] = i / 3.0
        this.stats[key] = 0
      }
    }
    //console.log('stats', this.stats)
  }

  /*= MESSAGES ===============================================================*/

  // to be redefined. Called on message (operation) from client
  onMessage(client: ClientType, message: any): void {}

  // to be redefined. Called when a client connects
  onConnect(client: ClientType): void {}

  // to be redefined. Called when a client disconnects
  onDisconnect(client: ClientType): void {}

  _onConnected(client: ClientType): void {
    client.ID = this.getUID()
    client.ping = 0
    client.ctdiff = 0
    client.subscribed = {}
    client.dataChannel = null
    client.peerConnection = null

    console.log('client connected', client.ID)
    this.clients.push(client)
    client.on('error', () => {
      this._onError(client, arguments)
    })
    client.on('message', (message:any) => {
      let dec = decode(message)
      if (this.simulateLatency) {
        setTimeout(() => {
          this._onMessage(client, dec)
        }, this.simulateLatency)
      } else {
        this._onMessage(client, dec)
      }
    })
    client.on('close', (message:any) => {
      this._onDisconnected(client)
      this.onDisconnect(client)
    })
    this.onConnect(client)
  }

  async _onMessage(client: ClientType, message: any): Promise<void> {
    if (
      message.c == 'sync' &&
      this.allowSync &&
      client.subscribed &&
      client.subscribed[message.n] &&
      this.documents[message.n]
    ) {
      let name = message.n
      if (!this._documentChanges[name]) {
        this._documentChanges[name] = []
      }
      for (let op of message.p) {
        if (!this.canSync(client, name, op)) {
          continue
        }
        this._documentChanges[name].push(op)
        let dop = msgop(op)
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
      let time = Date.now()
      let ping = time - message.st
      client.ctdiff = message.ct + ping / 2 - time
      client.ping = ping
      //console.log(time, "PENG ping, ctdiff", message, ping, client.ctdiff, "ms")
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

  _onError(client: ClientType, args: IArguments): void {
    console.error('onError:', args)
  }

  _onDisconnected(client: ClientType): void {
    if (client.dataChannel) {
      client.dataChannel.close()
    }
    if (client.peerConnection) {
      client.peerConnection.close()
    }
    console.log('client disconnected')
    let index = this.clients.indexOf(client)
    if (index !== -1) {
      this.clients.splice(index, 1)
    }
  }

  async send(client: ClientType, message: any): Promise<void> {
    try {
      let t1 = Date.now()
      let data = encode(message)
      let t2 = Date.now()
      let dl = data.byteLength
      if (this.allowCompression) {
        data = await compress(data)
      }
      let t3 = Date.now()
      if (data.length > 4096) {
        console.log(`Big message ${dl} -> ${data.length} (${(100.0 * data.length / dl).toFixed()}%) encoding:${t2 - t1}ms compression:${t3 - t1}ms`)
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
      console.error('Error sending message:', e, message)
    }
  }

  async broadcast(message: object, clients: ClientType[] | false = false): Promise<void> {
    if (!clients) {
      clients = this.clients
    }
    let data = encode(message)
    if (this.allowCompression) {
      data = await compress(data)
    }
    for (let client of this.clients) {
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

  async _sendFullState(name: string, client: ClientType): Promise<void> {
    await this._waitLoad(name)
    let excluded: any = '_'
    if (this.allowFastPatch) {
      excluded = fastPatchProperties
    }
    let doc = clonewo_(this.documents[name], excluded)
    limitPrecision(doc)
    let fdata: any = false
    if (this.allowFastPatch) {
      fdata = this._encodeFastChanges(name, false)
    }
    let fullState = {
      c: 'full',
      le: LITTLE_ENDIAN,
      t: Date.now(),
      n: name,
      doc: doc,
      fdata: fdata
    }
    this.send(client, fullState)
  }

  _encodeFastChanges(name: string, changesOnly = true): any {
    let doc = this.documents[name]
    if (!doc) { return false }
    let origin = this.documents[name].origin
    if (!origin) {
      origin = [0, 0, 0]
      this.documents[name].origin = origin
    }

    let entities = doc.entities
    let ids = Object.keys(entities)
    if (!entities) { return false }
    let count: Record<string, number> = {}
    let changed: any = {}
    let hasChanges: any = {}
    let dictionary: any = {}
    let encodedChanges: any = {}

    for (let key in fastPatchProperties) {
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

    // search for changes

    if (changesOnly) {
      for (let id in entities) {
        let e = entities[id]
        for (let key in fastPatchProperties) {
          if (e['__changed_' + key]) {
            changed[''+key][''+id] = true
            count[''+key] = parseInt(''+count[''+key]) + 1
            hasChanges[''+key] = true
            e['__changed_' + key] = false
          }
        }
      }
    } else {
      for (let id in entities) {
        for (let key in fastPatchProperties) {
          changed[''+key][''+id] = true
        }
      }
    }

    // create dictionaries

    let dictUID = 1
    for (let key in hasChanges) {
      if (hasChanges[key] && dictionaryProperties[key]) {
        for (let id in changed[key]) {
          let e = entities[id]
          let value = e[key]
          if (!dictionary[key][value]) {
            dictionary[key][value] = dictUID++
          }
        }
      }
    }

    console.log("--------------------------------------------------")
    //console.log("changed", changed)
    //console.log("count", count)

    // create encoded changes
    //
    for (let key in hasChanges) {
      if (hasChanges[key]) {
        let size = parseInt(''+count[''+key])
        let encoded: any = {}
        if (dictionaryProperties[key]) {
          encoded.dict = dictionary[key]

          let pdata = new Uint8Array(size * 8)
          let offset = 0
          for (let id in changed[key]) {
            let e = entities[id]
            let nid = parseInt(id)
            encode_uint32(nid, pdata, offset)
            offset += 4
            let value = e[key]
            let did = parseInt(dictionary[key][value])
            encode_uint32(did, pdata, offset)
            offset += 4
          }
          encoded.pdata = pdata
        } else {

          let pdata: Uint8Array
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
          for (let id in changed[key]) {
            let e = entities[id]
            let nid = parseInt(id)
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

  _sendPatches(): void {
    let now = Date.now()

    for (let name in this._documentChanges) {
      let dc = this._documentChanges[name]
      this._documentChanges[name] = []
      let sus = this.clients.filter((client) => client.subscribed && client.subscribed[name])
      if (sus.length > 0) {
        if (dc && dc.length > 0) {
          let record = {
            c: 'patch',
            t: now, // server time
            u: this.update,
            n: name,
            doc: dc,
          }
          this.broadcast(record, sus)
        }
      }

      if (this.allowFastPatch) {
        if (sus.length > 0) {
          let t1 = Date.now()
          let changes = this._encodeFastChanges(name)
          let t2 = Date.now()
          let record = {
            c: 'fpatch',
            t: now, // server time
            u: this.update,
            n: name,
            fdata: changes
          }
          this.broadcastRTC(record, sus)
          let t3 = Date.now()
          console.log(`_sendPatches: ${name} encode_changes: ${t2-t1}ms broadcast:${t3-t2}ms`)
        }
      }
    }
  }

  _onDocumentChange(name: string, op: any, target: any, path: any, value: any): void {
    this._documentChanges[name]?.push(opmsg(op, target, path, value))
  }

  propertyChange(name: string, id: string | number, property: string): void {
    let doc = this.documents[name]
    if (!doc) { return }
    let entities = doc.entities
    if (!entities) { return }
    let e = entities[id]
    if (!e) { return }
    e['__changed_'+property] = true
    //console.log('propertyChange', e)
  }


  /*= WEBRTC ===================================================================*/

  async _processOffer(client: ClientType, data: any): Promise<void> {
    //console.log("RTC: Offer received", data);
    const peerConnection = new wrtc.RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun.cloudflare.com:3478' },
        { urls: 'stun:freestun.net:3478' },
      ],
      iceCandidatePoolSize: 10,
    })

    client.peerConnection = peerConnection

    peerConnection.onicecandidate = (event: any) => {
      if (event.candidate) {
        //console.log("RTC: ICE candidate generated", event.candidate.candidate.substring(0, 50) + "...");
        this.send(client, {
          c: 'rtc-candidate',
          type: 'ice-candidate',
          candidate: event.candidate, // .toJSON()
        })
      } else {
        //console.log("RTC: ICE candidate gathering complete");
      }
    }

    peerConnection.onconnectionstatechange = () => {
      //console.log(`RTC: Connection state changed: ${peerConnection.connectionState}`);
      if (peerConnection.connectionState === 'connected') {
        client.webRTCConnected = true
        console.log(`RTC: Connection established with client ${client.ID}`)
      } else if (
        peerConnection.connectionState === 'failed' ||
        peerConnection.connectionState === 'disconnected' ||
        peerConnection.connectionState === 'closed'
      ) {
        client.webRTCConnected = false
        console.log(`RTC: Connection failed or closed with client ${client.ID}`)
      }
    }

    peerConnection.onicegatheringstatechange = () => {
      //console.log(`RTC: ICE gathering state: ${peerConnection.iceGatheringState}`);
    }

    peerConnection.oniceconnectionstatechange = () => {
      //console.log(`RTC: ICE connection state: ${peerConnection.iceConnectionState}`);
      if (
        peerConnection.iceConnectionState === 'connected' ||
        peerConnection.iceConnectionState === 'completed'
      ) {
        //console.log(`RTC: ICE connection established with client ${client.ID}`);
      }
    }

    try {
      await peerConnection.setRemoteDescription(
        new wrtc.RTCSessionDescription(data)
      )
      //console.log("RTC: Remote description set successfully");

      client.dataChannel = peerConnection.createDataChannel('serverchannel', {
        ordered: true,
        maxRetransmits: 1,
      })

      client.dataChannel.onopen = () => {
        //console.log(`RTC: Data channel opened for client ${client.ID}`);
        // Try sending a test message
        try {
          const testData = { c: 'test', message: 'Hello WebRTC' }
          this.sendRTC(client, testData)
        } catch (e) {
          console.error(
            `RTC: Error sending test message to client ${client.ID}`,
            e
          )
        }
      }

      client.dataChannel.onclose = () => {
        console.log(`RTC: Data channel closed for client ${client.ID}`)
      }

      client.dataChannel.onerror = (error: Event) => {
        console.error(`RTC: Data channel error for client ${client.ID}:`, error)
      }

      client.dataChannel.onmessage = (event: MessageEvent) => {
        try {
          const data = decode(event.data)
          console.log(
            `RTC: Data channel message from client ${client.ID}:`,
            data
          )
          //this.onMessage(client, data);
        } catch (error) {
          console.error(
            `RTC: Error decoding message from client ${client.ID}:`,
            error
          )
        }
      }

      // Create and send answer
      const answer = await peerConnection.createAnswer()
      await peerConnection.setLocalDescription(answer)

      //console.log(`RTC: Sending answer to client ${client.ID}`);
      this.send(client, {
        c: 'rtc-answer',
        type: answer.type,
        sdp: answer.sdp,
      })
    } catch (error) {
      console.error(
        `RTC: Error processing offer from client ${client.ID}:`,
        error
      )
    }
  }

  async _processICECandidate(client: ClientType, data: any): Promise<void> {
    //console.log(`RTC: Processing ICE candidate from client ${client.ID}`);
    try {
      if (client.peerConnection && data.candidate) {
        await client.peerConnection.addIceCandidate(
          data.candidate
          //new wrtc.RTCIceCandidate(data.candidate)
        )
        //console.log(`RTC: ICE candidate added successfully for client ${client.ID}`);
      } else {
        //console.warn(`RTC: Cannot add ICE candidate for client ${client.ID} - peerConnection not ready or candidate missing`);
      }
    } catch (error) {
      console.error(`RTC: Error adding ICE candidate for client ${client.ID}`)
    }
  }

  _clientRTCOpen(client: ClientType): boolean {
    return client.dataChannel !== null && client.dataChannel !== undefined && client.dataChannel.readyState === 'open'
  }

  async sendRTC(client: ClientType, message: any): Promise<void> {
    let data = encode(message)
    if (this.allowCompression) {
      data = await compress(data)
    }
    this.stats.sendRTC += data.byteLength
    this.stats._sendRTCUpdate += data.byteLength

    let packages = this._splitRTCMessage(data)

    if (this.simulateLatency) {
      setTimeout(() => {
        if (this._clientRTCOpen(client)) {
          packages.forEach((p) => {
            client.dataChannel!.send(p)
          })
        }
      }, this.simulateLatency)
    } else {
      if (this._clientRTCOpen(client)) {
        packages.forEach((p) => {
          client.dataChannel!.send(p)
        })
      }
    }
  }

  async broadcastRTC(message: any, clients: ClientType[] = []): Promise<void> {
    if (clients.length == 0) {
      clients = this.clients
    }
    let t1 = Date.now()
    let data = encode(message)
    let dl = data.byteLength
    let t2 = Date.now()
    if (this.allowCompression) {
      data = await compress(data)
    }
    let t3 = Date.now()


    if (data.length > 16384) {
      console.log(`BroadcastRTC message ${dl} -> ${data.length} (${(100.0 * data.length / dl).toFixed()}%) encoding:${t2 - t1}ms compression:${t3 - t1}ms`)
    }

    let packages = this._splitRTCMessage(data)

    for (let client of this.clients) {
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

  _splitRTCMessage(data: Uint8Array): Uint8Array[] {
    let packages: Uint8Array[]
    if (data.byteLength > 65535) {
      const now = Date.now()
      console.warn(`RTC: Message too large: ${data.byteLength} bytes`)
      // Split the message into smaller packages
      packages = [];
      let offset = 0;
      let mid = this.update +'-'+ now
      let seq = 0

      // Create subsequent packages if needed
      while (offset < data.byteLength) {
        const remaining = data.byteLength - offset;
        const chunkSize = Math.min(remaining, MAX_PACKAGE_SIZE);
        const chunk = new Uint8Array(data.buffer, offset, chunkSize);
        let cmessage = {
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
        offset += chunkSize;
        seq++;
      }

      console.log(`RTC: Large message split into ${packages.length} packages`);
    } else {
      packages = [data]
      console.log(`RTC: Message - ${data.byteLength} bytes`)
    }
    return packages
  }

  /*= DATABASE =================================================================*/
  // properties (of the documents) that starts with __ are not saved to the database.
  // __properties are restored on hydration. (for example __physicsBody or __bigObject)

  getUID(): number {
    this.documents['_server'].nextUID++
    return this.documents['_server'].nextUID
  }

  async _initDB(): Promise<void> {
    await this._connectDB()
    await this._loadDocument('_server')
    if (!this.documents['_server']) {
      this._initServerDocument()
    }
  }

  async _connectDB(): Promise<void> {
    this.mongoClient = new MongoClient(this.MongoUrl)
    try {
      await this.mongoClient.connect()
      console.log('Connected to MongoDB')
      const db = this.mongoClient.db(this.database)
      this.DB = db
    } catch (error) {
      console.error('Error connecting to MongoDB:', error)
      this.mongoClient = null
    }
  }

  async _loadDocument(name: string): Promise<void> {
    console.log(`Loading document '${name}' from MongoDB`)
    if (this.DB) {
      try {
        const doc = await this.DB.collection(this.collection).findOne({
          name: name,
        })
        if (doc) {
          delete (doc as any)._id
          this.documents[name] = doc
        }
      } catch (error) {
        console.error('Error loading document from MongoDB:', error)
      }
    } else {
      console.warn('MongoDB client not initialized. Document not loaded.')
    }
  }

  async _saveDocument(name: string): Promise<void> {
    if (this.DB) {
      try {
        const doc = this.documents[name]
        let newdoc = clonewo_(doc, '__')
        console.log(`Saving document '${name}' to MongoDB`)
        await this.DB.collection(this.collection).updateOne(
          { name: name },
          { $set: newdoc },
          { upsert: true }
        )
        console.log('Document saved to MongoDB')
      } catch (error) {
        console.error('Error saving document to MongoDB:', error)
      }
    } else {
      console.warn('MongoDB client not initialized. Document not saved.')
    }
  }

  async _saveAllDocuments(): Promise<void> {
    if (!this.allowSave) {
      return
    }
    for (let name in this.documents) {
      await this._saveDocument(name)
    }
  }

  _initServerDocument(): void {
    this.documents['_server'] = {
      nextUID: 100,
    }
  }

  /*= EXIT ===================================================================*/

  _exitSignal(signal: string): void {
    if (!this._exited) {
      console.log('\nEXIT: Caught interrupt signal ' + signal)
      this._exited = true
      clearInterval(this._loopiv)
      this.onBeforeExit()
      this.broadcast({ server: 'Going down' })
      this._saveAllDocuments()
      this._wss!.close()
      setTimeout(() => process.exit(0), 1000)
    }
  }

  // To be redefined. Called BEFORE program exit, and saving all documents
  onBeforeExit(): void {}
}
