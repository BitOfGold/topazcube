// vite.config.js
import { resolve } from 'path'
import { defineConfig } from 'vite'
import { externalizeDeps } from 'vite-plugin-externalize-deps'

const config = {
  server: {
    entry: resolve(__dirname, './src/server.ts'),
    name: 'TopazCube Server',
    fileName: 'topazcube-server',
  },
  client: {
    entry: resolve(__dirname, './src/topazcube.ts'),
    name: 'TopazCube Client',
    fileName: 'topazcube-client',
  },
}

const currentConfig = config[process.env.LIB_NAME]

if (currentConfig === undefined) {
  throw new Error('LIB_NAME is not defined or is not valid')
}

export default defineConfig({
  root: resolve(__dirname, 'src'),
  plugins: [externalizeDeps()],
  build: {
    lib: {
      ...currentConfig,
      formats: ['es'],
      target: 'esnext',
    },
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: false,
    esbuild: {
      comments: 'none',
      legalComments: 'none',
    },
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
})
