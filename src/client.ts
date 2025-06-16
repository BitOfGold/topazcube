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
} from './utils'
import { compress, decompress } from './compress-browser'
import { glMatrix, vec3, quat } from 'gl-matrix'
import { argv0 } from 'process'

const MAX_PACKAGE_SIZE = 65400; // Slightly below the 65535 limit to allow for overhead

export default class TopazCubeClient {
  CYCLE = 200 // update/patch rate in ms
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
    stdiff: 0, // server time difference
  }
  lastFullState = 0
  lastPatch = 0
  _chunks = {}
  le = true // Server is little endian
  _documentChanges:any = {}

  ID = 0
  socket:any = null
  _peerConnection:any = null
  _dataChannel:any = null // our data channel
  _serverDataChannel:any = null // server data channel
  _webRTCConnected:any = null

  isInterpolated = false
  _lastInterpolate = Date.now()
  _lastUpdateId = {}
  _dpos:vec3 = [0, 0, 0]
  _drot:quat = [0, 0, 0, 1]
  _sca:vec3 = [1, 1, 1]
  _notifyChanges = true
  _siv:any = null
  _loopiv:any = null
  _updateiv:any = null
  _pingiv:any = null


  constructor({
    url, // server url
    autoReconnect = true, // auto reconnect on disconnect
    allowSync = true, // allow sync on connect
    allowWebRTC = false,
  }) {
    this.url = url
    this.autoReconnect = autoReconnect
    this.allowSync = allowSync
    this.allowWebRTC = allowWebRTC
    this.socket = null
    this._startLoop()
    console.log('Client initialized')
  }

  /*= UPDATE ===================================================================*/

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

  /*= INTERPOLATION ============================================================*/

  // to be called in display rate (like 60fps) to interpolate .position, .rotation and .scale
  interpolate() {
    let now = Date.now()
    let dt = now - this._lastInterpolate
    this._lastInterpolate = now
    if (dt <= 0 || dt > 200) { return }
    this.isInterpolated = true
    for (let name in this.documents) {
      let doc = this.documents[name]
      let entities = doc.entities
      if (!entities) { continue }
      for (let id in entities) {
        let e = entities[id]
        if (e._lpostime1 && e._lpostime2) {
          let t1 = e._lpostime1
          let t2 = e._lpostime2
          const interval = t2 - t1;
          const elapsed = now - t1;
          const alpha = Math.max(0, elapsed / interval)
          vec3.lerp(this._dpos, e._lpos1, e._lpos2, alpha)
          vec3.lerp(e.position, e.position, this._dpos, 0.07)
          e._changed_position = now
        }
        if (e._lrottime1 && e._lrottime2) {

          let t1 = e._lrottime1
          let t2 = e._lrottime2
          const interval = t2 - t1;
          const elapsed = now - t1;
          const alpha = Math.max(0, elapsed / interval)
          quat.slerp(this._drot, e._lrot1, e._lrot2, alpha)
          quat.slerp(e.rotation, e.rotation, this._drot, 0.07)
          e._changed_rotation = now
        }

      }
    }
    this.isInterpolated = false
  }

  /*= CONNECTION ===============================================================*/

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
    console.log('connecting...')

    this.socket = new WebSocket(this.url)

    // message received
    this.socket.onmessage = async (event) => {
      let buffer = await event.data.arrayBuffer()
      this.stats.rec += buffer.byteLength
      let dec = await decompress(buffer)
      let message = decode(dec)
      this._onMessage(message)
    }

    // connection closed
    this.socket.onclose = (event) => {
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

    this.socket.onerror = (event) => {
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

    this.socket.onopen = async (event) => {
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
    this.socket.close()
    this.socket = null
  }

  destroy() {
    this._clear()
    this.autoReconnect = false
    this.disconnect()
    this.socket = null
    clearInterval(this._siv)
    clearInterval(this._loopiv)
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

  /*= MESSAGES =================================================================*/

  onChange(name, doc) {}

  onMessage(message) {}

  send(operation) {
    try {
      let enc = encode(operation)
      this.stats.send += enc.byteLength
      this.socket.send(enc)
    } catch (e) {
      console.error('send failed', e)
    }
  }

  get document() {
    let names = Object.keys(this.documents)
    return this.documents[names[0]]
  }

  async _onMessage(message) {
    let time = Date.now()
    if (message.c == 'full') {
      console.log('full:', message)
      let name = message.n
      let doc = message.doc
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
        this.onChange(name, this.documents[name])
      }
    } else if (message.c == 'patch') {
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
      if (this._notifyChanges) {
        this.onChange(name, this.documents[name])
      }
    } else if (message.c == 'chunk') {
      //console.log('chunk', message)
      this._chunks[message.mid+'-'+message.seq] = message
      if (message.last) {
        let cfound = 0
        let ts = message.ts
        let cdata = new Uint8Array(ts)
        for (const cid in this._chunks) {
          let chunk = this._chunks[cid]
          if (chunk.mid == message.mid) {
            let offset = chunk.ofs
            let csize = chunk.chs
            cdata.set(new Uint8Array(chunk.data), offset);
            cfound++
            delete this._chunks[cid]
          }
        }
        //console.log('found chunks ', cfound, 'of', message.seq + 1)
        if (cfound == message.seq + 1) {
          try {
            let cdec = await decompress(cdata)
            let nmessage = decode(cdec)
            //console.log('decoded message', nmessage)
            this._onMessage(nmessage)
          } catch (error) {
            console.error('Error decoding chunks:', error)
          }
        } else {
          console.warn('missing chunks', cfound, 'of', message.seq + 1)
        }
      }
    } else if (message.c == 'fpatch') {
      time = Date.now()
      let name = message.n
      //console.log('fpatch', message)
      let doPatch = true
      if (!this._lastUpdateId[name]) {
        this._lastUpdateId[name] = message.u
      } else {
        if (this._lastUpdateId[name] < message.u) {
          let lp = message.u - this._lastUpdateId[name] - 1
          if (lp > 0) {
            console.warn('Lost ' + lp + ' updates')
          }
          this._lastUpdateId[name] = message.u
        } else if (this._lastUpdateId[name] > message.u) {
          // Handle the case where the server's update ID is older than the client's
          // This could be due to a network issue or a clock skew
          console.warn(`Received outdated update ID for document ${name}: ${message.u} < ${this._lastUpdateId[name]}`)
          doPatch = false
        }
      }
      if (doPatch) {
        this._decodeFastChanges(message)
      }
    } else if (message.c == 'pong') {
      this.ID = message.ID
      time = Date.now()
      let lastct = message.ct
      let ping = time - lastct
      let stime = message.t
      this.send({ c: 'peng', ct: Date.now(), st: stime })
      this.stats.stdiff = stime + ping / 2 - time
      this.stats.ping = ping
      console.log('ping', ping, 'ms', 'stdiff', this.stats.stdiff, 'ms')
    } else if (message.c == 'rtc-offer') {
      //console.log("RTC: offer received:", message);
      // You might need to handle this if the server sends offers
    } else if (message.c == 'rtc-answer') {
      //console.log("RTC: answer received:", message);
      try {
        const sessionDesc = new RTCSessionDescription({
          type: message.type,
          sdp: message.sdp,
        })
        await this._peerConnection.setRemoteDescription(sessionDesc)
        //console.log("RTC: Remote description set successfully");

        // Log the current state after setting remote description
        //console.log("RTC: Current connection state:", this._peerConnection.connectionState);
        //console.log("RTC: Current ICE connection state:", this._peerConnection.iceConnectionState);
      } catch (error) {
        console.error('RTC: Error setting remote description:', error)
      }
    } else if (message.c == 'rtc-candidate') {
      //console.log("RTC: candidate received", message);
      try {
        if (this._peerConnection && message.candidate) {
          await this._peerConnection.addIceCandidate(
            new RTCIceCandidate(message.candidate)
          )
          //console.log("RTC: ICE candidate added successfully");
        } else {
          console.warn(
            'RTC: Received candidate but peerConnection not ready or candidate missing'
          )
        }
      } catch (error) {
        //console.error("RTC: Error adding ICE candidate:", error);
      }
    } else {
      this.onMessage(message)
    }
  }

  _onDocumentChange(name, op, target, path, value) {
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
    for (let name in this._documentChanges) {
      let dc = this._documentChanges[name]
      if (dc.length == 0) {
        continue
      }
      let record = {
        n: name,
        c: 'sync',
        ct: Date.now(),
        p: null
      }

      if (dc.length > 0) {
        record.p = dc
      }
      this.send(record)
      this._documentChanges[name].length = 0
      if (this._notifyChanges) {
        this.onChange(name, this.documents[name])
      }
    }
  }

  _decodeFastChanges(message) {
    let time = Date.now()
    let name = message.n
    let fdata = message.fdata
    if (!fdata) {
      return
    }
    let doc = this.documents[name]
    if (!doc) {
      return
    }
    let entities = doc.entities
    if (!entities) {
      return
    }
    let origin = this.documents[name].origin
    if (!origin) {
      origin = [0, 0, 0]
    }
    for (let key in fdata) {
      let changes = fdata[key]
      if (changes.dict) {
        let pdata = changes.pdata
        let dict = changes.dict
        // Reverse the dictionary for lookup (value to key)
        let rdict = {};
        for (let key in dict) {
          rdict[dict[key]] = key;
        }
        let offset = 0

        while (offset < pdata.byteLength) {
          let id = ''+decode_uint32(pdata, offset)
          offset += 4
          let did = ''+decode_uint32(pdata, offset)
          offset += 4
          let e = entities[id]
          if (!e) {
            //console.log('Entity not found:', id)
            continue
          }
          let value = rdict[did]
          e[key] = value
          //console.log('FCHANGE', key, id, did, value, rdict)
          e['_changed_'+key] = time
        }
      } else {
        let pdata = changes.pdata
        let offset = 0
        while (offset < pdata.byteLength) {
          let id = ''+decode_uint32(pdata, offset)
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
              e._lsca1 = [0, 0, 0, 1]
              e._lsca2 = [0, 0, 0, 1]
            } else {
              e._lsca1[0] = e._lsca2[0]
              e._lsca1[1] = e._lsca2[1]
              e._lsca1[2] = e._lsca2[2]
              e._lsca1[3] = e._lsca2[3]
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

  /*= WEBRTC ===================================================================*/

  sendRTC(message) {
    if (this._dataChannel && this._dataChannel.readyState === 'open') {
      this._dataChannel.send(encode(message))
    }
  }

  _onRTCConnect() {
    console.log('RTC: Connected')
    this.send({ c: 'test', message: 'Hello RTC from client' })
  }

  _onRTCDisconnect() {
    this._webRTCConnected = true
    console.log('RTC: Disconnected')
  }

  async _onRTCMessage(data) {
    this.stats.recRTC += data.byteLength
    let dec = await decompress(data)
    let message = decode(dec)
    this._onMessage(message)
  }

  async _initializeWebRTC() {
    //console.log("RTC: _initializeWebRTC")
    this._peerConnection = null
    try {
      // Create RTCPeerConnection with more comprehensive STUN server list
      this._peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun.cloudflare.com:3478' },
          { urls: 'stun:freestun.net:3478' },
        ],
        iceCandidatePoolSize: 10,
      })

      //console.log("RTC: peerConnection created", this._peerConnection)

      // Handle ICE candidates
      this._peerConnection.onicecandidate = (event) => {
        //console.log("RTC: onicecandidate", event.candidate)
        if (event.candidate) {
          this.send({
            c: 'rtc-candidate',
            type: 'ice-candidate',
            candidate: event.candidate,
          })
        } else {
          //console.log("RTC: ICE candidate gathering complete")
        }
      }

      // Log connection state changes
      this._peerConnection.onconnectionstatechange = () => {
        //console.log(`RTC: Connection state changed: ${this._peerConnection.connectionState}`)
        if (this._peerConnection.connectionState === 'connected') {
          this._webRTCConnected = true
        } else if (
          this._peerConnection.connectionState === 'failed' ||
          this._peerConnection.connectionState === 'disconnected' ||
          this._peerConnection.connectionState === 'closed'
        ) {
          this._webRTCConnected = false
        }
      }

      this._peerConnection.onicegatheringstatechange = () => {
        //console.log(`RTC: ICE gathering state: ${this._peerConnection.iceGatheringState}`)
      }

      this._peerConnection.oniceconnectionstatechange = () => {
        //console.log(`RTC: ICE connection state: ${this._peerConnection.iceConnectionState}`)

        // This is critical - when ICE succeeds, the connection should be established
        if (
          this._peerConnection.iceConnectionState === 'connected' ||
          this._peerConnection.iceConnectionState === 'completed'
        ) {
          //console.log("RTC: ICE connection established!")
        }
      }

      // Create a data channel on our side as well (belt and suspenders approach)
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

      this._dataChannel.onerror = (error) => {
        console.error('RTC: Client data channel error', error)
      }

      // Handle data channels created by the server
      this._peerConnection.ondatachannel = (event) => {
        //console.log("RTC: Server data channel received", event.channel.label);
        const dataChannel = event.channel
        this._serverDataChannel = dataChannel

        dataChannel.onopen = () => {
          //console.log("RTC: Server data channel open");
        }

        dataChannel.onclose = () => {
          //console.log("RTC: Server data channel closed");
        }

        dataChannel.onerror = (error) => {
          //console.error("RTC: Server data channel error", error);
        }

        dataChannel.onmessage = (event) => {
          this._onRTCMessage(event.data)
        }
      }

      // Create and send offer with specific constraints
      const offerOptions = {
        offerToReceiveAudio: false,
        offerToReceiveVideo: false,
        iceRestart: true,
      }

      const offer = await this._peerConnection.createOffer(offerOptions)
      //console.log("RTC: our offer:", offer);
      await this._peerConnection.setLocalDescription(offer)

      // Wait a moment to ensure the local description is set
      await new Promise((resolve) => setTimeout(resolve, 100))

      let ld = this._peerConnection.localDescription
      //console.log("RTC: our localDescription", ld);

      const offerPayload = {
        c: 'rtc-offer',
        type: ld.type,
        sdp: ld.sdp,
      }
      //console.log("RTC: our offer payload", offerPayload);
      this.send(offerPayload)

      // Set a timeout to check connection status
      setTimeout(() => {
        if (!this._webRTCConnected && this._peerConnection) {
          /*
          console.log("RTC: Connection not established after timeout, current states:");
          console.log("Connection state:", this._peerConnection.connectionState);
          console.log("ICE connection state:", this._peerConnection.iceConnectionState);
          console.log("ICE gathering state:", this._peerConnection.iceGatheringState);
          */
          // Attempt to restart ICE if needed
          if (this._peerConnection.iceConnectionState === 'failed') {
            console.log('RTC: Attempting ICE restart')
            this._restartIce()
          }
        }
      }, 5000)
    } catch (error) {
      console.error('RTC: error:', error)
    }
  }

  // Add this method to restart ICE if needed
  async _restartIce() {
    try {
      const offerOptions = {
        offerToReceiveAudio: false,
        offerToReceiveVideo: false,
        iceRestart: true,
      }

      const offer = await this._peerConnection.createOffer(offerOptions)
      await this._peerConnection.setLocalDescription(offer)

      const offerPayload = {
        c: 'rtc-offer',
        type: offer.type,
        sdp: offer.sdp,
      }
      //console.log("RTC: ICE restart offer payload", offerPayload);
      this.send(offerPayload)
    } catch (error) {
      //console.error("RTC: Error during ICE restart:", error);
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
