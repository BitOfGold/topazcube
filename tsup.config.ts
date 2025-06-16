import { defineConfig } from 'tsup'

export default defineConfig({
  entry: [
    'src/client.ts',
    'src/server.ts',
    'src/terminal.js'
  ],
  format: ['cjs', 'esm'],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  external: [],
  target: 'node18',
  /*
  esbuildOptions: (options) => {
    options.banner = {
      js: '"use client";',
    };
  },
  */
});
