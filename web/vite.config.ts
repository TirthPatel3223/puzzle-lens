import { defineConfig } from 'vitest/config';

/** The local Python server (scanner/server.py): vision and solver. */
const API = process.env.SCANNER_API ?? 'http://127.0.0.1:8000';

export default defineConfig({
  base: './',
  server: {
    // The browser talks to the Python server through the dev server, so there is one origin.
    proxy: { '/api': { target: API, changeOrigin: true } },
    // shared/ (generated layout and solver map) lives next to web/.
    fs: { allow: ['..'] },
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    chunkSizeWarningLimit: 800,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
