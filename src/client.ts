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

const MAX_PACKAGE_SIZE = 65400; // Slightly below the 65535 limit to allow for overhead

interface Stats {
  send: number;
  rec: number;
  recRTC: number;
  sendBps: number;
  recBps: number;
  recRTCBps: number;
  ping: number;
  stdiff: number; // server time difference
}

interface ConstructorParams {
  url: string; // server url
  autoReconnect?: boolean; // auto reconnect on disconnect
  allowSync?: boolean; // allow sync on connect
  allowWebRTC?: boolean;
  DEBUG?: boolean;
}

interface Message {
  c: string;
  [key: string]: any;
}

interface Document {
  entities?: { [key: string]: any };
  origin?: vec3;
  [key: string]: any;
}

interface Entity {
  position?: vec3;
  rotation?: quat;
  _lpos1?: vec3;
  _lpos2?: vec3;
  _lpostime1?: number | undefined;
  _lpostime2?: number | undefined;
  _lrot1?: quat;
  _lrot2?: quat;
  _lrottime1?: number | undefined;
  _lrottime2?: number | undefined;
  _lsca1?: vec3;
  _lsca2?: vec3;
  _lscatime1?: number | undefined;
  _lscatime2?: number | undefined;
  sca?: vec3;
  [key: string]: any;
}
// WebRTC type declarations for environments that don't have them built-in
declare global {
  interface RTCPeerConnection {
    new(configuration?: RTCConfiguration): RTCPeerConnection;
    createOffer(options?: RTCOfferOptions): Promise<RTCSessionDescriptionInit>;
    createAnswer(options?: RTCAnswerOptions): Promise<RTCSessionDescriptionInit>;
    setLocalDescription(description?: RTCSessionDescriptionInit): Promise<void>;
    setRemoteDescription(description: RTCSessionDescriptionInit): Promise<void>;
    addIceCandidate(candidate?: RTCIceCandidateInit): Promise<void>;
    createDataChannel(label: string, dataChannelDict?: RTCDataChannelInit): RTCDataChannel;
    close(): void;
    readonly connectionState: RTCPeerConnectionState;
    readonly iceConnectionState: RTCIceConnectionState;
    readonly iceGatheringState: RTCIceGatheringState;
    readonly localDescription: RTCSessionDescription | null;
    readonly remoteDescription: RTCSessionDescription | null;
    onconnectionstatechange: ((this: RTCPeerConnection, ev: Event) => any) | null;
    oniceconnectionstatechange: ((this: RTCPeerConnection, ev: Event) => any) | null;
    onicegatheringstatechange: ((this: RTCPeerConnection, ev: Event) => any) | null;
    onicecandidate: ((this: RTCPeerConnection, ev: RTCPeerConnectionIceEvent) => any) | null;
    ondatachannel: ((this: RTCPeerConnection, ev: RTCDataChannelEvent) => any) | null;
    dataChannel?: RTCDataChannel;
  }

  interface RTCDataChannel {
    readonly label: string;
    readonly readyState: RTCDataChannelState;
    send(data: string | Blob | ArrayBuffer | ArrayBufferView): void;
    close(): void;
    onopen: ((this: RTCDataChannel, ev: Event) => any) | null;
    onclose: ((this: RTCDataChannel, ev: Event) => any) | null;
    onerror: ((this: RTCDataChannel, ev: Event) => any) | null;
    onmessage: ((this: RTCDataChannel, ev: MessageEvent) => any) | null;
  }

  interface RTCSessionDescription {
    readonly type: RTCSdpType;
    readonly sdp: string;
  }

  interface RTCIceCandidate {
    new(candidateInitDict?: RTCIceCandidateInit): RTCIceCandidate;
  }

  interface RTCPeerConnectionIceEvent extends Event {
    readonly candidate: RTCIceCandidate | null;
  }

  interface RTCDataChannelEvent extends Event {
    readonly channel: RTCDataChannel;
  }

  type RTCPeerConnectionState = 'closed' | 'connected' | 'connecting' | 'disconnected' | 'failed' | 'new';
  type RTCIceConnectionState = 'checking' | 'closed' | 'completed' | 'connected' | 'disconnected' | 'failed' | 'new';
  type RTCIceGatheringState = 'complete' | 'gathering' | 'new';
  type RTCDataChannelState = 'closed' | 'closing' | 'connecting' | 'open';
  type RTCSdpType = 'answer' | 'offer' | 'pranswer' | 'rollback';

  interface RTCConfiguration {
    iceServers?: RTCIceServer[];
    iceCandidatePoolSize?: number;
  }

  interface RTCIceServer {
    urls: string | string[];
    username?: string;
    credential?: string;
  }

  interface RTCOfferOptions {
    offerToReceiveAudio?: boolean;
    offerToReceiveVideo?: boolean;
    iceRestart?: boolean;
  }

  interface RTCAnswerOptions {
    // Currently no standard options for answers
  }

  interface RTCSessionDescriptionInit {
    type: RTCSdpType;
    sdp?: string;
  }

  interface RTCIceCandidateInit {
    candidate?: string;
    sdpMLineIndex?: number | null;
    sdpMid?: string | null;
    usernameFragment?: string | null;
  }

  interface RTCDataChannelInit {
    ordered?: boolean;
    maxPacketLifeTime?: number;
    maxRetransmits?: number;
    protocol?: string;
    negotiated?: boolean;
    id?: number;
  }

  const RTCPeerConnection: {
    prototype: RTCPeerConnection;
    new(configuration?: RTCConfiguration): RTCPeerConnection;
  };

  const RTCSessionDescription: {
    prototype: RTCSessionDescription;
    new(descriptionInitDict: RTCSessionDescriptionInit): RTCSessionDescription;
  };

  const RTCIceCandidate: {
    prototype: RTCIceCandidate;
    new(candidateInitDict?: RTCIceCandidateInit): RTCIceCandidate;
  };
}

export default class TopazCubeClient {
  DEBUG = false
  CYCLE = 200 // update/patch rate in ms
  url = ''
  documents: { [key: string]: Document } = {}
  autoReconnect = true
  allowSync = true
  allowWebRTC = false
  isConnected = false
  isConnecting = false
  isPatched = false
  stats: Stats = {
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
  _chunks: { [key: string]: any } = {}
  le = true // Server is little endian
  _documentChanges: { [key: string]: any[] } = {}

  ID = 0
  socket: WebSocket | null = null
  _peerConnection: RTCPeerConnection | null = null
  _candidates: RTCIceCandidate[] = []
  _remoteCandidates: RTCIceCandidateInit[] = []
  _offerSent: boolean = false
  _dataChannel: RTCDataChannel | null = null // our data channel
  _serverDataChannel: RTCDataChannel | null = null // server data channel
  _webRTCConnected: boolean = false

  isInterpolated = false
  _lastInterpolate = Date.now()
  _lastUpdateId: { [key: string]: number } = {}
  _dpos: vec3 = [0, 0, 0]
  _drot: quat = [0, 0, 0, 1]
  _sca: vec3 = [1, 1, 1]
  _notifyChanges = true
  _siv: NodeJS.Timeout | null = null
  _loopiv: NodeJS.Timeout | null = null
  _updateiv: NodeJS.Timeout | null = null
  _pingiv: NodeJS.Timeout | null = null


  constructor({
    url, // server url
    autoReconnect = true, // auto reconnect on disconnect
    allowSync = true, // allow sync on connect
    allowWebRTC = false,
    DEBUG = false
  }: ConstructorParams) {
    this.url = url
    this.autoReconnect = autoReconnect
    this.allowSync = allowSync
    this.allowWebRTC = allowWebRTC
    this.socket = null
    this.DEBUG = DEBUG
    this._startLoop()
    this.log('Client initialized')
  }

  log(...args: any[]) {
    if (this.DEBUG) {
      console.log(...args);
    }
  }

  warn(...args: any[]) {
    if (this.DEBUG) {
      console.warn(...args);
    }
  }

  error(...args: any[]) {
    console.error(...args);
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
      let entities = doc?.entities
      if (!entities) { continue }
      for (let id in entities) {
        let e: Entity = entities[id]
        if (e._lpostime1 && e._lpostime2) {
          let t1 = e._lpostime1
          let t2 = e._lpostime2
          const interval = t2 - t1;
          const elapsed = now - t1;
          e.pelapsed = elapsed
          /*if (elapsed > 5000) {
          } else */if (elapsed > 1000) {
            vec3.copy(e.position!, e._lpos2!)
            e._changed_position = now
          } else {
            const alpha = Math.max(0, elapsed / interval)
            vec3.lerp(this._dpos, e._lpos1!, e._lpos2!, alpha)
            vec3.lerp(e.position!, e.position!, this._dpos, 0.07)
            e._changed_position = now
          }
        }
        if (e._lrottime1 && e._lrottime2) {
          let t1 = e._lrottime1
          let t2 = e._lrottime2
          const interval = t2 - t1;
          const elapsed = now - t1;
          e.relapsed = elapsed
          /*if (elapsed > 5000) {
          } else */if (elapsed > 1000) {
            quat.copy(e.rotation!, e._lrot2!)
            e._changed_rotation = now
          } else {
            const alpha = Math.max(0, elapsed / interval)
            quat.slerp(this._drot, e._lrot1!, e._lrot2!, alpha)
            quat.slerp(e.rotation!, e.rotation!, this._drot, 0.07)
            e._changed_rotation = now
          }
        }

      }
    }
    this.isInterpolated = false
  }

  /*= CONNECTION ===============================================================*/

  subscribe(name: string) {
    this.documents[name] = {}
    this.send({ c: 'sub', n: name })
  }

  unsubscribe(name: string) {
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

    // message received
    this.socket.onmessage = async (event: MessageEvent) => {
      let buffer = await event.data.arrayBuffer()
      this.stats.rec += buffer.byteLength
      let dec = await decompress(buffer)
      let decu = new Uint8Array(dec)
      let message = decode(decu)
      this._onMessage(message)
    }

    // connection closed
    this.socket.onclose = (_event: any) => {
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

    this.socket.onerror = (_event: Event) => {
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

    this.socket.onopen = async (_event: Event) => {
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

  /*= MESSAGES =================================================================*/

  onChange(_name: string, _doc: Document | undefined, patch: any | undefined) {}

  onMessage(_message: Message) {}

  send(operation: any) {
    try {
      let enc = encode(operation)
      this.stats.send += enc.byteLength
      if (this.socket) {
        this.socket.send(enc)
      }
    } catch (e) {
      this.error('send failed', e)
    }
  }

  get document(): Document | undefined {
    let names:string = ''+Object.keys(this.documents)
    return this.documents[''+names[0]]
  }

  async _onMessage(message: Message) {
    let time = Date.now()
    if (message.c == 'full') {
      //this.log('full:', message)
      let name:string = ''+message.n
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
        this.onChange(name, this.documents[name], null)
      }
    } else if (message.c == 'patch') {
      // patch
      this.lastPatch = message.t
      let name = message.n
      if (message.doc) {
        this.isPatched = true
        for (let op of message.doc) {
          let dop = msgop(op)
          try {
            applyOperation(this.documents[name], dop)
          } catch (e) {
            this.error('applyOperation failed for', name, 'with op', dop, e)
          }
        }
        this.isPatched = false
      }
      if (this._notifyChanges) {
        this.onChange(name, this.documents[name], message.doc)
      }
    } else if (message.c == 'chunk') {
      //this.log('chunk', message)
      this._chunks[message.mid+'-'+message.seq] = message
      if (message.last) {
        let cfound = 0
        let ts = message.ts
        let cdata = new Uint8Array(ts)
        for (const cid in this._chunks) {
          let chunk = this._chunks[cid]
          if (chunk.mid == message.mid) {
            let offset = chunk.ofs
            let _csize = chunk.chs
            cdata.set(new Uint8Array(chunk.data), offset);
            cfound++
            delete this._chunks[cid]
          }
        }
        //this.log('found chunks ', cfound, 'of', message.seq + 1)
        if (cfound == message.seq + 1) {
          try {
            let cdec = await decompress(cdata)
            let cdecu = new Uint8Array(cdec)
            let nmessage = decode(cdecu)
            //this.log('decoded message', nmessage)
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
      let name = message.n
      //this.log('fpatch', message)
      let doPatch = true
      if (!this._lastUpdateId[name]) {
        this._lastUpdateId[name] = message.u
      } else {
        if (this._lastUpdateId[name] < message.u) {
          let lp = message.u - this._lastUpdateId[name] - 1
          if (lp > 0) {
            this.warn('Lost ' + lp + ' updates')
          }
          this._lastUpdateId[name] = message.u
        } else if (this._lastUpdateId[name] > message.u) {
          // Handle the case where the server's update ID is older than the client's
          // This could be due to a network issue or a clock skew
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
      let lastct = message.ct
      let ping = time - lastct
      let stime = message.t
      this.send({ c: 'peng', ct: Date.now(), st: stime })
      this.stats.stdiff = stime + ping / 2 - time
      this.stats.ping = ping
      this.log('ping', ping, 'ms', 'stdiff', this.stats.stdiff, 'ms')
    } else if (message.c == 'rtc-offer') {
      this.log("RTC: offer received:", message);
      // You might need to handle this if the server sends offers
    } else if (message.c == 'rtc-answer') {
      this.log("RTC: answer received:", message);
      try {
        const sessionDesc = new RTCSessionDescription({
          type: message.type,
          sdp: message.sdp,
        })
        if (this._peerConnection) {
          await this._peerConnection.setRemoteDescription(sessionDesc)
        }
        this.log("RTC: Remote description set successfully");

        // Log the current state after setting remote description
        //
        //this.log("RTC: Current connection state:", this._peerConnection.connectionState);
        //this.log("RTC: Current ICE connection state:", this._peerConnection.iceConnectionState);
        for (let candidate of this._remoteCandidates) {
          try {
            await this._peerConnection?.addIceCandidate(candidate);
            this.log("RTC: Added remote ICE candidate:", candidate);
          } catch (error) {
            this.error("RTC: Error adding remote ICE candidate:", error);
          }
        }
      } catch (error) {
        this.error('RTC: Error setting remote description:', error)
      }
    } else if (message.c == 'rtc-candidate') {
      this.log("RTC: candidate received", message);
        if (this._peerConnection && message.candidate) {
          this._remoteCandidates.push(message.candidate);
        }
    } else {
      this.onMessage(message)
    }
  }

  _onDocumentChange(name: string, op: any, target: any, path: string, value: any) {
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
    for (let name in this._documentChanges) {
      let dc = this._documentChanges[name]
      if (!dc || dc.length == 0) {
        continue
      }
      let record: any = {
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
        this.onChange(name, this.documents[''+name], record.p)
      }
    }
  }

  _decodeFastChanges(message: Message) {
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
    let origin = this.documents[''+name]?.origin
    if (!origin) {
      origin = [0, 0, 0]
    }
    for (let key in fdata) {
      let changes = fdata[key]
      if (changes.dict) {
        let pdata = changes.pdata
        let dict = changes.dict
        // Reverse the dictionary for lookup (value to key)
        let rdict: { [key: string]: string } = {};
        for (let key in dict) {
          rdict[dict[key]] = key;
        }
        let offset = 0

        while (offset < pdata.byteLength) {
          let id = ''+decode_uint32(pdata, offset)
          offset += 4
          let did = ''+decode_uint32(pdata, offset)
          offset += 4
          let e: Entity = entities[id]
          if (!e) {
            //this.log('Entity not found:', id)
            continue
          }
          let value = rdict[did]
          e[key] = value
          //this.log('FCHANGE', key, id, did, value, rdict)
          e['_changed_'+key] = time
        }
      } else {
        let pdata = changes.pdata
        let offset = 0
        while (offset < pdata.byteLength) {
          let id = ''+decode_uint32(pdata, offset)
          let e: Entity = entities[id]
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
              e._lpos1![0] = e._lpos2[0]
              e._lpos1![1] = e._lpos2[1]
              e._lpos1![2] = e._lpos2[2]
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
              e._lrot1![0] = e._lrot2[0]
              e._lrot1![1] = e._lrot2[1]
              e._lrot1![2] = e._lrot2[2]
              e._lrot1![3] = e._lrot2[3]
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
              e._lsca1![0] = e._lsca2[0]
              e._lsca1![1] = e._lsca2[1]
              e._lsca1![2] = e._lsca2[2]
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

  sendRTC(message: Message) {
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

  async _onRTCMessage(data: ArrayBuffer) {
    this.stats.recRTC += data.byteLength
    let datau = new Uint8Array(data)
    let dec = await decompress(datau)
    let decu = new Uint8Array(dec)
    let message = decode(decu)
    this._onMessage(message)
  }

  async _initializeWebRTC() {
    //this.log("RTC: _initializeWebRTC")
    this._peerConnection = null
    this._candidates = []
    this._remoteCandidates = []
    this._offerSent = false
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

      //this.log("RTC: peerConnection created", this._peerConnection)

      // Handle ICE candidates
      this._peerConnection.onicecandidate = (event: RTCPeerConnectionIceEvent) => {
        //this.log("RTC: onicecandidate", event.candidate)
        if (event.candidate) {
          this._candidates.push(event.candidate)
        } else {
          this.log("RTC: ICE candidate gathering complete")
        }
      }

      // Log connection state changes
      this._peerConnection.onconnectionstatechange = () => {
        //this.log(`RTC: Connection state changed: ${this._peerConnection.connectionState}`)
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
          for (let candidate of this._candidates) {
            this.send({
              c: 'rtc-candidate',
              type: 'ice-candidate',
              candidate: candidate,
            })
          }
        }
      }

      this._peerConnection.oniceconnectionstatechange = () => {
        //this.log(`RTC: ICE connection state: ${this._peerConnection.iceConnectionState}`)

        // This is critical - when ICE succeeds, the connection should be established
        if (
          this._peerConnection && (
          this._peerConnection.iceConnectionState === 'connected' ||
          this._peerConnection.iceConnectionState === 'completed')
        ) {
          //this.log("RTC: ICE connection established!")
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

      this._dataChannel.onerror = (_error: Event) => {
        this.error('RTC: Client data channel error', _error)
      }

      // Handle data channels created by the server
      this._peerConnection.ondatachannel = (event: RTCDataChannelEvent) => {
        //this.log("RTC: Server data channel received", event.channel.label);
        const dataChannel = event.channel
        this._serverDataChannel = dataChannel

        dataChannel.onopen = () => {
          //this.log("RTC: Server data channel open");
        }

        dataChannel.onclose = () => {
          //this.log("RTC: Server data channel closed");
        }

        dataChannel.onerror = (_error: Event) => {
          //this.error("RTC: Server data channel error", error);
        }

        dataChannel.onmessage = (event: MessageEvent) => {
          this._onRTCMessage(event.data)
        }
      }

      // Create and send offer with specific constraints
      const offerOptions: RTCOfferOptions = {
        offerToReceiveAudio: false,
        offerToReceiveVideo: false,
        iceRestart: true,
      }

      const offer = await this._peerConnection.createOffer(offerOptions)
      //this.log("RTC: our offer:", offer);
      await this._peerConnection.setLocalDescription(offer)

      // Wait a moment to ensure the local description is set
      await new Promise((resolve) => setTimeout(resolve, 100))

      let ld = this._peerConnection.localDescription
      //this.log("RTC: our localDescription", ld);

      if (ld) {
        const offerPayload = {
          c: 'rtc-offer',
          type: ld.type,
          sdp: ld.sdp,
        }
        //this.log("RTC: our offer payload", offerPayload);
        this.send(offerPayload)
        this._offerSent = true
      }

      // Set a timeout to check connection status
      setTimeout(() => {
        if (!this._webRTCConnected && this._peerConnection) {
          /*
          this.log("RTC: Connection not established after timeout, current states:");
          this.log("Connection state:", this._peerConnection.connectionState);
          this.log("ICE connection state:", this._peerConnection.iceConnectionState);
          this.log("ICE gathering state:", this._peerConnection.iceGatheringState);
          */
          // Attempt to restart ICE if needed
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

  // Add this method to restart ICE if needed
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
      //this.log("RTC: ICE restart offer payload", offerPayload);
      this.send(offerPayload)
    } catch (error) {
      //this.error("RTC: Error during ICE restart:", error);
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
