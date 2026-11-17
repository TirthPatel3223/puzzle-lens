import { configureRendering } from './common/random';
import { malteseDatagen } from './maltese/main';
import type { DatagenPage } from './page';
import { tangramDatagen } from './tangram/main';

/**
 * The synthetic-photo page:
 *
 *   datagen.html?puzzle=maltese-gear-cube&size=640      (the default puzzle)
 *   datagen.html?puzzle=tangram-cube-extreme&size=640
 *   …&layouts=0.5&shades=wide      half the puzzles stickered another way, wider sticker shades (common/random.ts)
 *
 * Driven by tools/gen_dataset.py; opened by hand it is a viewer (one sample per click, labels drawn over it).
 */

const params = new URLSearchParams(location.search);
const page: DatagenPage = {
  params,
  size: Math.max(128, Math.min(2048, parseInt(params.get('size') ?? '640', 10) || 640)),
  canvas: document.getElementById('shot') as HTMLCanvasElement,
  overlay: document.getElementById('overlay') as HTMLCanvasElement,
  info: document.getElementById('info') as HTMLElement,
};

configureRendering(params);
const puzzle = params.get('puzzle') ?? 'maltese-gear-cube';
window.datagen = (puzzle.startsWith('tangram') ? tangramDatagen(page) : malteseDatagen(page)) as typeof window.datagen;
