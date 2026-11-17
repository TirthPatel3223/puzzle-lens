import { defineConfig } from 'vitest/config';

/** Generators that write files from the engine (run with `npm run export:shared`), not tests. */
export default defineConfig({
  test: {
    environment: 'node',
    include: ['tools/**/*.gen.ts'],
  },
});
