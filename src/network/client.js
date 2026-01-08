import { applyOperation } from 'fast-json-patch'
import {
  reactive,
  opmsg,
  msgop,
  decode_uint32,
  decode_fp412,
  decode_fp168,
  decode_fp1616,
  deepGet,
  encode,
  decode,
} from './utils.js'
import { compress, decompress } from './compress-browser.js'
import { vec3, quat } from 'gl-matrix'

const MAX_PACKAGE_SIZE = 65400

export default class TopazCubeClient {
  DEBUG = false
  CYCLE = 200
  url = ''
  documents = {}
  autoReconnect = true
  allowSync = true
  allowWebRTC = false
  isConnected = false
  isConnecting = false
  isPatched = false
  stats = {
    send: 0,
    rec: 0,
    recRTC: 0,
    sendBps: 0,
    recBps: 0,
    recRTCBps: 0,
    ping: 0,
    stdiff: 0,
  }
  lastFullState = 0
  lastPatch = 0
  _chunks = {}
  le = true
  _documentChanges = {}

  ID = 0
  socket = null
  _peerConnection = null
  _candidates = []
  _remoteCandidates = []
  _offerSent = false
  _dataChannel = null
  _serverDataChannel = null
  _webRTCConnected = false

  isInterpolated = false
  _lastInterpolate = Date.now()
  _lastUpdateId = {}
  _dpos = [0, 0, 0]
  _drot = [0, 0, 0, 1]
  _sca = [1, 1, 1]
  _notifyChanges = true
  _siv = null
  _loopiv = null
  _updateiv = null
  _pingiv = null

  constructor({
    url,
    autoReconnect = true,
    allowSync = true,
    allowWebRTC = false,
    DEBUG = false
  }) {
    this.url = url
    this.autoReconnect = autoReconnect
    this.allowSync = allowSync
    this.allowWebRTC = allowWebRTC
    this.socket = null
    this.DEBUG = DEBUG
    this._startLoop()
    this.log('Client initialized')
  }

  log(...args) {
    if (this.DEBUG) {
      console.log(...args)
    }
  }

  warn(...args) {
    if (this.DEBUG) {
      console.warn(...args)
    }
  }

  error(...args) {
    console.error(...args)
  }

  _startLoop() {
    if (this._loopiv) {
      clearInterval(this._loopiv)
    }
    this._loopiv = setInterval(() => {
      this._loop()
    }, this.CYCLE)
    this._siv = setInterval(() => {
      this._updateStats()
    }, 1000)
    this._pingiv = setInterval(() => {
      this._ping()
    }, 10000)
  }

  _loop() {
    if (!this.isConnected) {
      return
    }
    this._sendPatches()
  }

  _updateStats() {
    this.stats.recBps = this.stats.rec
    this.stats.rec = 0
    this.stats.recRTCBps = this.stats.recRTC
    this.stats.recRTC = 0
    this.stats.sendBps = this.stats.send
    this.stats.send = 0
  }

  _clear() {
    this.stats.sendBps = 0
    this.stats.recBps = 0
    this.stats.recRTC = 0
    this.stats.recRTCBps = 0
    this.stats.send = 0
    this.stats.rec = 0

    this.ID = 0
    this.documents = {}
    this._documentChanges = {}
    this._lastUpdateId = {}
    this.lastFullState = 0
    this.lastPatch = 0
    this._lastInterpolate = 0
    this.isPatched = false
    this.le = true
  }

  interpolate() {
    const now = Date.now()
    const dt = now - this._lastInterpolate
    this._lastInterpolate = now
    if (dt <= 0 || dt > 200) { return }
    this.isInterpolated = true
    for (const name in this.documents) {
      const doc = this.documents[name]
      const entities = doc?.entities
      if (!entities) { continue }
      for (const id in entities) {
        const e = entities[id]
        if (e._lpostime1 && e._lpostime2) {
          const t1 = e._lpostime1
          const t2 = e._lpostime2
          const interval = t2 - t1
          const elapsed = now - t1
          e.pelapsed = elapsed
          if (elapsed > 1000) {
            vec3.copy(e.position, e._lpos2)
            e._changed_position = now
          } else {
            const alpha = Math.max(0, elapsed / interval)
            vec3.lerp(this._dpos, e._lpos1, e._lpos2, alpha)
            vec3.lerp(e.position, e.position, this._dpos, 0.07)
            e._changed_position = now
          }
        }
        if (e._lrottime1 && e._lrottime2) {
          const t1 = e._lrottime1
          const t2 = e._lrottime2
          const interval = t2 - t1
          const elapsed = now - t1
          e.relapsed = elapsed
          if (elapsed > 1000) {
            quat.copy(e.rotation, e._lrot2)
            e._changed_rotation = now
          } else {
            const alpha = Math.max(0, elapsed / interval)
            quat.slerp(this._drot, e._lrot1, e._lrot2, alpha)
            quat.slerp(e.rotation, e.rotation, this._drot, 0.07)
            e._changed_rotation = now
          }
        }
      }
    }
    this.isInterpolated = false
  }

  subscribe(name) {
    this.documents[name] = {}
    this.send({ c: 'sub', n: name })
  }

  unsubscribe(name) {
    this.send({ c: 'unsub', n: name })
    delete this.documents[name]
  }

  connect() {
    if (this.isConnecting) {
      return
    }
    this.isConnecting = true
    this._clear()
    this.log('connecting...')

    this.socket = new WebSocket(this.url)

    this.socket.onmessage = async (event) => {
      const buffer = await event.data.arrayBuffer()
      this.stats.rec += buffer.byteLength
      const dec = await decompress(buffer)
      const decu = new Uint8Array(dec)
      const message = decode(decu)
      this._onMessage(message)
    }

    this.socket.onclose = (_event) => {
      this._clear()
      this.isConnected = false
      this.isConnecting = false
      this.lastFullState = 0
      this.socket = null
      this.onDisconnect()
      if (this.allowWebRTC) {
        this._destroyWebRTC()
      }
      if (this.autoReconnect) {
        setTimeout(
          () => {
            this._reconnect()
          },
          500 + Math.random() * 500
        )
      }
    }

    this.socket.onerror = (_event) => {
      this._clear()
      this.isConnected = false
      this.isConnecting = false
      this.lastFullState = 0
      this.socket = null
      this.onDisconnect()
      if (this.allowWebRTC) {
        this._destroyWebRTC()
      }

      if (this.autoReconnect) {
        setTimeout(
          () => {
            this._reconnect()
          },
          500 + Math.random() * 500
        )
      }
    }

    this.socket.onopen = async (_event) => {
      this._clear()
      this.isConnecting = false
      this.isConnected = true
      this.lastFullState = 0
      this._ping()
      this.onConnect()
      if (this.allowWebRTC) {
        await this._initializeWebRTC()
      }
    }
  }

  disconnect() {
    this._clear()
    this.isConnected = false
    this.isConnecting = false
    this.lastFullState = 0
    if (this.socket) {
      this.socket.close()
    }
    this.socket = null
  }

  destroy() {
    this._clear()
    this.autoReconnect = false
    this.disconnect()
    this.socket = null
    if (this._siv) clearInterval(this._siv)
    if (this._loopiv) clearInterval(this._loopiv)
  }

  onConnect() {}

  onDisconnect() {}

  _reconnect() {
    if (!this.isConnected) {
      if (!this.isConnecting) {
        this.connect()
      }
    }
  }

  _ping() {
    if (this.isConnected) {
      this.send({ c: 'ping', ct: Date.now() })
    }
  }

  onChange(_name, _doc, patch) {}

  onMessage(_message) {}

  send(operation) {
    if (!this.isConnected) {
      return
    }
    try {
      const enc = encode(operation)
      this.stats.send += enc.byteLength
      if (this.socket) {
        this.socket.send(enc)
      }
    } catch (e) {
      this.error('send failed', e)
    }
  }

  get document() {
    const names = '' + Object.keys(this.documents)
    return this.documents['' + names[0]]
  }

  async _onMessage(message) {
    let time = Date.now()
    if (message.c == 'full') {
      const name = '' + message.n
      const doc = message.doc
      this.documents[name] = doc
      this._decodeFastChanges(message)
      this.isPatched = false
      if (this.allowSync) {
        this.documents[name] = reactive(
          name,
          this.documents[name],
          this._onDocumentChange.bind(this)
        )
      }
      this.isPatched = false
      this.lastFullState = message.t
      this.le = message.le
      if (this._notifyChanges) {
        this.onChange(name, this.documents[name], null)
      }
    } else if (message.c == 'patch') {
      this.lastPatch = message.t
      const name = message.n
      if (!this.documents[name]) {
        this.warn('Patch for unknown document', name)
        return
      }
      if (message.doc) {
        this.isPatched = true
        for (const op of message.doc) {
          const dop = msgop(op)
          try {
            applyOperation(this.documents[name], dop)
          } catch (e) {
            this.warn('applyOperation failed for', name, 'with op', dop, e)
          }
        }
        this.isPatched = false
      }
      if (this._notifyChanges) {
        this.onChange(name, this.documents[name], message.doc)
      }
    } else if (message.c == 'chunk') {
      this._chunks[message.mid + '-' + message.seq] = message
      if (message.last) {
        let cfound = 0
        const ts = message.ts
        const cdata = new Uint8Array(ts)
        for (const cid in this._chunks) {
          const chunk = this._chunks[cid]
          if (chunk.mid == message.mid) {
            const offset = chunk.ofs
            const _csize = chunk.chs
            cdata.set(new Uint8Array(chunk.data), offset)
            cfound++
            delete this._chunks[cid]
          }
        }
        if (cfound == message.seq + 1) {
          try {
            const cdec = await decompress(cdata)
            const cdecu = new Uint8Array(cdec)
            const nmessage = decode(cdecu)
            this._onMessage(nmessage)
          } catch (error) {
            this.error('Error decoding chunks:', error)
          }
        } else {
          this.warn('missing chunks', cfound, 'of', message.seq + 1)
        }
      }
    } else if (message.c == 'fpatch') {
      time = Date.now()
      const name = message.n
      let doPatch = true
      if (!this._lastUpdateId[name]) {
        this._lastUpdateId[name] = message.u
      } else {
        if (this._lastUpdateId[name] < message.u) {
          const lp = message.u - this._lastUpdateId[name] - 1
          if (lp > 0) {
            this.warn('Lost ' + lp + ' updates')
          }
          this._lastUpdateId[name] = message.u
        } else if (this._lastUpdateId[name] > message.u) {
          this.warn(`Received outdated update ID for document ${name}: ${message.u} < ${this._lastUpdateId[name]}`)
          doPatch = false
        }
      }
      if (doPatch) {
        this._decodeFastChanges(message)
      }
    } else if (message.c == 'pong') {
      this.ID = message.ID
      time = Date.now()
      const lastct = message.ct
      const ping = time - lastct
      const stime = message.t
      this.send({ c: 'peng', ct: Date.now(), st: stime })
      this.stats.stdiff = stime + ping / 2 - time
      this.stats.ping = ping
      this.log('ping', ping, 'ms', 'stdiff', this.stats.stdiff, 'ms')
    } else if (message.c == 'rtc-offer') {
      this.log("RTC: offer received:", message)
    } else if (message.c == 'rtc-answer') {
      this.log("RTC: answer received:", message)
      try {
        const sessionDesc = new RTCSessionDescription({
          type: message.type,
          sdp: message.sdp,
        })
        if (this._peerConnection) {
          await this._peerConnection.setRemoteDescription(sessionDesc)
        }
        this.log("RTC: Remote description set successfully")

        for (const candidate of this._remoteCandidates) {
          try {
            await this._peerConnection?.addIceCandidate(candidate)
            this.log("RTC: Added remote ICE candidate:", candidate)
          } catch (error) {
            this.error("RTC: Error adding remote ICE candidate:", error)
          }
        }
      } catch (error) {
        this.error('RTC: Error setting remote description:', error)
      }
    } else if (message.c == 'rtc-candidate') {
      this.log("RTC: candidate received", message)
      if (this._peerConnection && message.candidate) {
        this._remoteCandidates.push(message.candidate)
      }
    } else {
      this.onMessage(message)
    }
  }

  _onDocumentChange(name, op, target, path, value) {
    if (this.DEBUG) {
      this.log('Document change:', name, op, target, path, value)
    }
    if (this.isPatched || !this.allowSync) {
      return
    }
    if (path.indexOf('/_') >= 0) {
      return
    }
    if (!this._documentChanges[name]) {
      this._documentChanges[name] = []
    }
    this._documentChanges[name].push(opmsg(op, target, path, value))
  }

  _sendPatches() {
    for (const name in this._documentChanges) {
      const dc = this._documentChanges[name]
      if (!dc || dc.length == 0) {
        continue
      }
      const record = {
        n: name,
        c: 'sync',
        ct: Date.now(),
        p: null
      }

      if (dc.length > 0) {
        record.p = dc
      }
      this.send(record)
      dc.length = 0
      if (this._notifyChanges) {
        this.onChange(name, this.documents['' + name], record.p)
      }
    }
  }

  _decodeFastChanges(message) {
    const time = Date.now()
    const name = message.n
    const fdata = message.fdata
    if (!fdata) {
      return
    }
    const doc = this.documents[name]
    if (!doc) {
      return
    }
    const entities = doc.entities
    if (!entities) {
      return
    }
    let origin = this.documents['' + name]?.origin
    if (!origin) {
      origin = [0, 0, 0]
    }
    for (const key in fdata) {
      const changes = fdata[key]
      if (changes.dict) {
        const pdata = changes.pdata
        const dict = changes.dict
        const rdict = {}
        for (const key in dict) {
          rdict[dict[key]] = key
        }
        let offset = 0

        while (offset < pdata.byteLength) {
          const id = '' + decode_uint32(pdata, offset)
          offset += 4
          const did = '' + decode_uint32(pdata, offset)
          offset += 4
          const e = entities[id]
          if (!e) {
            continue
          }
          const value = rdict[did]
          e[key] = value
          e['_changed_' + key] = time
        }
      } else {
        const pdata = changes.pdata
        let offset = 0
        while (offset < pdata.byteLength) {
          const id = '' + decode_uint32(pdata, offset)
          let e = entities[id]
          if (!e) {
            if (key == 'position') {
              offset += 13
            } else if (key == 'rotation') {
              offset += 8
            } else if (key == 'scale') {
              offset += 16
            }
            continue
          }
          offset += 4

          if (key == 'position') {
            if (!e._lpos2) {
              e._lpos1 = [0, 0, 0]
              e._lpos2 = [0, 0, 0]
            } else {
              e._lpos1[0] = e._lpos2[0]
              e._lpos1[1] = e._lpos2[1]
              e._lpos1[2] = e._lpos2[2]
              e._lpostime1 = e._lpostime2
            }
            e._lpostime2 = time
            e._lpos2[0] = origin[0] + decode_fp168(pdata, offset)
            offset += 3
            e._lpos2[1] = origin[1] + decode_fp168(pdata, offset)
            offset += 3
            e._lpos2[2] = origin[2] + decode_fp168(pdata, offset)
            offset += 3
            if (!e.position) {
              e.position = [
                e._lpos2[0],
                e._lpos2[1],
                e._lpos2[2],
              ]
            }
          } else if (key == 'rotation') {
            if (!e._lrot2) {
              e._lrot1 = [0, 0, 0, 1]
              e._lrot2 = [0, 0, 0, 1]
            } else {
              e._lrot1[0] = e._lrot2[0]
              e._lrot1[1] = e._lrot2[1]
              e._lrot1[2] = e._lrot2[2]
              e._lrot1[3] = e._lrot2[3]
              e._lrottime1 = e._lrottime2
            }
            e._lrottime2 = time
            e._lrot2[0] = decode_fp412(pdata, offset)
            offset += 2
            e._lrot2[1] = decode_fp412(pdata, offset)
            offset += 2
            e._lrot2[2] = decode_fp412(pdata, offset)
            offset += 2
            e._lrot2[3] = decode_fp412(pdata, offset)
            offset += 2
            quat.normalize(e._lrot2, e._lrot2)
            if (!e.rotation) {
              e.rotation = [
                e._lrot2[0],
                e._lrot2[1],
                e._lrot2[2],
                e._lrot2[3],
              ]
            }
          } else if (key == 'scale') {
            if (!e._lsca2) {
              e._lsca1 = [0, 0, 0]
              e._lsca2 = [0, 0, 0]
            } else {
              e._lsca1[0] = e._lsca2[0]
              e._lsca1[1] = e._lsca2[1]
              e._lsca1[2] = e._lsca2[2]
              e._lscatime1 = e._lscatime2
            }
            e._lscatime2 = time
            e._lsca2[0] = decode_fp1616(pdata, offset)
            offset += 4
            e._lsca2[1] = decode_fp1616(pdata, offset)
            offset += 4
            e._lsca2[2] = decode_fp1616(pdata, offset)
            offset += 4
            if (!e.sca) {
              e.sca = [
                e._lsca2[0],
                e._lsca2[1],
                e._lsca2[2],
              ]
            }
          }
        }
      }
    }
  }

  sendRTC(message) {
    if (this._dataChannel && this._dataChannel.readyState === 'open') {
      this._dataChannel.send(encode(message))
    }
  }

  _onRTCConnect() {
    this.log('RTC: Connected')
    this.send({ c: 'test', message: 'Hello RTC from client' })
  }

  _onRTCDisconnect() {
    this._webRTCConnected = false
    this.log('RTC: Disconnected')
  }

  async _onRTCMessage(data) {
    this.stats.recRTC += data.byteLength
    const datau = new Uint8Array(data)
    const dec = await decompress(datau)
    const decu = new Uint8Array(dec)
    const message = decode(decu)
    this._onMessage(message)
  }

  async _initializeWebRTC() {
    this._peerConnection = null
    this._candidates = []
    this._remoteCandidates = []
    this._offerSent = false
    try {
      this._peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun.cloudflare.com:3478' },
          { urls: 'stun:freestun.net:3478' },
        ],
        iceCandidatePoolSize: 10,
      })

      this._peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          this._candidates.push(event.candidate)
        } else {
          this.log("RTC: ICE candidate gathering complete")
        }
      }

      this._peerConnection.onconnectionstatechange = () => {
        if (this._peerConnection && this._peerConnection.connectionState === 'connected') {
          this._webRTCConnected = true
          this.log('RTC: Peer connection established!')
        } else if (
          this._peerConnection && (
            this._peerConnection.connectionState === 'failed' ||
            this._peerConnection.connectionState === 'disconnected' ||
            this._peerConnection.connectionState === 'closed')
        ) {
          this._webRTCConnected = false
          this.log('RTC: Peer connection closed or failed')
        }
      }

      this._peerConnection.onicegatheringstatechange = () => {
        this.log(`RTC: ICE gathering state. _candidates:`, this._candidates.length, this._peerConnection?.iceGatheringState)
        if (this._peerConnection?.iceGatheringState == 'complete' && this._offerSent) {
          for (const candidate of this._candidates) {
            this.send({
              c: 'rtc-candidate',
              type: 'ice-candidate',
              candidate: candidate,
            })
          }
        }
      }

      this._peerConnection.oniceconnectionstatechange = () => {
        if (
          this._peerConnection && (
            this._peerConnection.iceConnectionState === 'connected' ||
            this._peerConnection.iceConnectionState === 'completed')
        ) {
          // ICE connection established
        }
      }

      this._dataChannel = this._peerConnection.createDataChannel(
        'clientchannel',
        {
          ordered: true,
          maxRetransmits: 1,
        }
      )

      this._dataChannel.onopen = () => {
        this._onRTCConnect()
      }

      this._dataChannel.onclose = () => {
        this._onRTCDisconnect()
      }

      this._dataChannel.onerror = (_error) => {
        this.error('RTC: Client data channel error', _error)
      }

      this._peerConnection.ondatachannel = (event) => {
        const dataChannel = event.channel
        this._serverDataChannel = dataChannel

        dataChannel.onopen = () => {}

        dataChannel.onclose = () => {}

        dataChannel.onerror = (_error) => {}

        dataChannel.onmessage = (event) => {
          this._onRTCMessage(event.data)
        }
      }

      const offerOptions = {
        offerToReceiveAudio: false,
        offerToReceiveVideo: false,
        iceRestart: true,
      }

      const offer = await this._peerConnection.createOffer(offerOptions)
      await this._peerConnection.setLocalDescription(offer)

      await new Promise((resolve) => setTimeout(resolve, 100))

      const ld = this._peerConnection.localDescription

      if (ld) {
        const offerPayload = {
          c: 'rtc-offer',
          type: ld.type,
          sdp: ld.sdp,
        }
        this.send(offerPayload)
        this._offerSent = true
      }

      setTimeout(() => {
        if (!this._webRTCConnected && this._peerConnection) {
          if (this._peerConnection.iceConnectionState === 'failed') {
            this.log('RTC: Attempting ICE restart')
            this._restartIce()
          }
        }
      }, 5000)
    } catch (error) {
      this.error('RTC: error:', error)
    }
  }

  async _restartIce() {
    try {
      const offerOptions = {
        offerToReceiveAudio: false,
        offerToReceiveVideo: false,
        iceRestart: true,
      }

      const offer = await this._peerConnection?.createOffer(offerOptions)
      await this._peerConnection?.setLocalDescription(offer)

      const offerPayload = {
        c: 'rtc-offer',
        type: offer?.type,
        sdp: offer?.sdp,
      }
      this.send(offerPayload)
    } catch (error) {
      // Error during ICE restart
    }
  }

  async _destroyWebRTC() {
    if (this._peerConnection) {
      if (this._peerConnection.dataChannel) {
        this._peerConnection.dataChannel.close()
      }
      this._peerConnection.close()
      this._peerConnection = null
    }
    if (this._dataChannel) {
      this._dataChannel.close()
      this._dataChannel = null
    }
    this._webRTCConnected = false
  }
}
