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

WebRTC functionality is optional and requires platform-specific binary packages. If you want to use WebRTC features:

```bash
# Install the appropriate platform package
npm install @roamhq/wrtc-darwin-x64  # for macOS x64
npm install @roamhq/wrtc-linux-x64   # for Linux x64  
npm install @roamhq/wrtc-win32-x64   # for Windows x64
```

Then enable WebRTC in your server:

```typescript
const server = new TopazCubeServer({
  port: 8799,
  allowWebRTC: true // Enable WebRTC functionality
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
