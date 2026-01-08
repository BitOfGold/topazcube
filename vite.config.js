import { defineConfig } from 'vite'
import { resolve } from 'path'
import glsl from 'vite-plugin-glsl'

export default defineConfig({
  plugins: [
    glsl()
  ],
  build: {
    lib: {
      entry: {
        client: resolve(__dirname, 'src/network/client.js'),
        server: resolve(__dirname, 'src/network/server.js'),
        terminal: resolve(__dirname, 'src/network/terminal.js'),
        Renderer: resolve(__dirname, 'src/renderer/Renderer.js'),
      },
      formats: ['es', 'cjs'],
      fileName: (format, entryName) => {
        const ext = format === 'es' ? 'js' : 'cjs'
        return `${entryName}.${ext}`
      }
    },
    rollupOptions: {
      external: [
        'node:https',
        'node:fs',
        'util',
        'zlib',
        'ws',
        'mongodb',
        'werift',
        'fast-json-patch',
        'msgpackr',
        'gl-matrix',
        '@loaders.gl/core',
        '@loaders.gl/gltf',
        'lil-gui',
        'webgpu-utils',
        'wgsl_reflect'
      ],
      output: {
        preserveModules: false,
        globals: {
          'ws': 'ws',
          'mongodb': 'mongodb',
          'werift': 'werift',
          'fast-json-patch': 'fastjsonpatch',
          'msgpackr': 'msgpackr',
          'gl-matrix': 'glMatrix'
        }
      }
    },
    sourcemap: true,
    minify: false,
    target: 'es2022',
    outDir: 'dist'
  }
})
