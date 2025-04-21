// vite.config.js
import { resolve } from "path"
import { defineConfig } from "vite"

export default defineConfig({
  root: resolve(__dirname, "src"),
  build: {
    lib: {
      entry: resolve(__dirname, "src/topazcube.js"),
      name: "TopazCube",
      // the proper extensions will be added
      fileName: "topazcube",
      formats: ["es", "cjs"],
    },
    outDir: resolve(__dirname, "dist"),
    emptyOutDir: true, 
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
})
