# TopazCube

TopazCube is a real-time collaborative document editing and multiplayer game library.

## Installation

```bash
npm install topazcube
```

## Usage

### Basic Server Setup

```typescript
import TopazCubeServer from 'topazcube/server'

const server = new TopazCubeServer({
  port: 8799,
  allowWebRTC: false // WebRTC is optional
})
```

### WebRTC Support (Optional)

WebRTC functionality is optional and requires platform-specific binary packages. **Note: WebRTC is only available when using the CommonJS build due to native module limitations.**

If you want to use WebRTC features:

```bash
# Install the appropriate platform package
npm install @roamhq/wrtc-darwin-x64  # for macOS x64
npm install @roamhq/wrtc-linux-x64   # for Linux x64  
npm install @roamhq/wrtc-win32-x64   # for Windows x64
```

Then use the CommonJS build and enable WebRTC:

```javascript
// Use require() and .cjs build for WebRTC support
const TopazCubeServer = require('topazcube/server').default;

const server = new TopazCubeServer({
  port: 8799,
  allowWebRTC: true // Enable WebRTC functionality
})
```

For ESM imports without WebRTC:

```typescript
// ESM import works fine without WebRTC
import TopazCubeServer from 'topazcube/server'

const server = new TopazCubeServer({
  port: 8799,
  allowWebRTC: false // WebRTC not available in ESM mode
})
```

### Client Setup

```typescript
import TopazCubeClient from 'topazcube/client'

const client = new TopazCubeClient({
  url: 'ws://localhost:8799'
})
```

## Features

- Real-time collaborative document editing
- Multiplayer game support
- WebSocket communication
- Optional WebRTC for peer-to-peer connections
- MongoDB integration for persistence
- Fast binary patches for performance
