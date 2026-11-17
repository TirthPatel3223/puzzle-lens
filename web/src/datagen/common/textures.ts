import { mulberry32 } from '../../vendor/puzzle/rng';
import { type TextureSpec, chance, pick, uni } from './random';

/**
 * Procedural stand-ins for desks, walls and rooms, painted on a 2D canvas. They do not need to look
 * like anything in particular: they need to be varied, so that the models learn that what surrounds
 * the cube carries no information.
 */

const SIZE = 256;

const rgb = (r: number, g: number, b: number, a = 1) => `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${a})`;

function anyColour(rng: () => number, a = 1): string {
  // Mostly dull colours (desks, walls, cloth), now and then a saturated one (toys, book covers).
  const sat = chance(rng, 0.25) ? uni(rng, 0.5, 1) : uni(rng, 0, 0.3);
  const h = rng() * 360;
  const l = uni(rng, 0.08, 0.95);
  return `hsla(${h.toFixed(0)},${(sat * 100).toFixed(0)}%,${(l * 100).toFixed(0)}%,${a})`;
}

export function paintTexture(canvas: HTMLCanvasElement | OffscreenCanvas, spec: TextureSpec): void {
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
  const rng = mulberry32(spec.seed);
  ctx.filter = 'none';
  ctx.globalAlpha = 1;
  ctx.fillStyle = anyColour(rng);
  ctx.fillRect(0, 0, SIZE, SIZE);

  switch (spec.kind) {
    case 'solid':
      break;
    case 'gradient': {
      const radial = chance(rng, 0.4);
      const g = radial
        ? ctx.createRadialGradient(uni(rng, 0, SIZE), uni(rng, 0, SIZE), 0, SIZE / 2, SIZE / 2, SIZE * uni(rng, 0.6, 1.2))
        : ctx.createLinearGradient(uni(rng, 0, SIZE), 0, uni(rng, 0, SIZE), SIZE);
      const stops = 2 + Math.floor(rng() * 2);
      for (let i = 0; i < stops; i++) g.addColorStop(i / (stops - 1), anyColour(rng));
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, SIZE, SIZE);
      break;
    }
    case 'wood': {
      const h = uni(rng, 15, 45);
      const s = uni(rng, 20, 60);
      const l = uni(rng, 18, 70);
      ctx.fillStyle = `hsl(${h},${s}%,${l}%)`;
      ctx.fillRect(0, 0, SIZE, SIZE);
      ctx.save();
      ctx.translate(SIZE / 2, SIZE / 2);
      ctx.rotate(uni(rng, 0, Math.PI));
      ctx.translate(-SIZE, -SIZE);
      let y = 0;
      while (y < 2 * SIZE) {
        const w = uni(rng, 1, 9);
        ctx.fillStyle = `hsla(${h + uni(rng, -6, 6)},${s}%,${l + uni(rng, -12, 12)}%,${uni(rng, 0.25, 0.8)})`;
        ctx.fillRect(0, y, 2 * SIZE, w);
        y += w + uni(rng, 0, 4);
      }
      // Board joints.
      if (chance(rng, 0.5)) {
        ctx.fillStyle = rgb(0, 0, 0, uni(rng, 0.2, 0.6));
        const gap = uni(rng, 50, 140);
        for (let x = uni(rng, 0, gap); x < 2 * SIZE; x += gap) ctx.fillRect(0, x, 2 * SIZE, uni(rng, 1, 3));
      }
      ctx.restore();
      break;
    }
    case 'blobs': {
      const count = 20 + Math.floor(rng() * 120);
      for (let i = 0; i < count; i++) {
        ctx.fillStyle = anyColour(rng, uni(rng, 0.15, 0.9));
        ctx.beginPath();
        ctx.ellipse(uni(rng, 0, SIZE), uni(rng, 0, SIZE), uni(rng, 3, 70), uni(rng, 3, 70), uni(rng, 0, Math.PI), 0, 2 * Math.PI);
        ctx.fill();
      }
      break;
    }
    case 'clutter': {
      // Rectangles, bars and rows of small marks: shelves, papers, keyboards, screens.
      const count = 8 + Math.floor(rng() * 40);
      for (let i = 0; i < count; i++) {
        ctx.save();
        ctx.translate(uni(rng, 0, SIZE), uni(rng, 0, SIZE));
        ctx.rotate(chance(rng, 0.7) ? 0 : uni(rng, -0.5, 0.5));
        const w = uni(rng, 8, 150);
        const h = uni(rng, 4, 110);
        ctx.fillStyle = anyColour(rng, uni(rng, 0.5, 1));
        ctx.fillRect(-w / 2, -h / 2, w, h);
        if (chance(rng, 0.4)) {
          ctx.fillStyle = anyColour(rng, 0.9);
          const rows = 1 + Math.floor(rng() * 6);
          for (let r = 0; r < rows; r++) {
            for (let x = -w / 2 + 3; x < w / 2 - 4; x += uni(rng, 4, 12)) ctx.fillRect(x, -h / 2 + 3 + (r * (h - 6)) / rows, uni(rng, 2, 8), Math.max(1.5, (h - 6) / rows - 2));
          }
        }
        ctx.restore();
      }
      break;
    }
    case 'tiles': {
      const n = pick(rng, [2, 3, 4, 6, 8, 12]);
      const a = anyColour(rng);
      const b = anyColour(rng);
      const grout = chance(rng, 0.5) ? uni(rng, 1, 4) : 0;
      const step = SIZE / n;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          ctx.fillStyle = (i + j) % 2 ? a : b;
          ctx.fillRect(i * step + grout / 2, j * step + grout / 2, step - grout, step - grout);
        }
      }
      break;
    }
  }

  // Fine grain: nothing real is perfectly flat in colour.
  if (chance(rng, 0.7)) {
    const grains = 400 + Math.floor(rng() * 2500);
    for (let i = 0; i < grains; i++) {
      ctx.fillStyle = chance(rng, 0.5) ? rgb(1, 1, 1, uni(rng, 0.02, 0.12)) : rgb(0, 0, 0, uni(rng, 0.02, 0.12));
      ctx.fillRect(uni(rng, 0, SIZE), uni(rng, 0, SIZE), uni(rng, 1, 3), uni(rng, 1, 3));
    }
  }
  // Out of focus.
  if (spec.blur > 0.3) {
    ctx.filter = `blur(${spec.blur.toFixed(1)}px)`;
    ctx.drawImage(canvas as CanvasImageSource, 0, 0);
    ctx.filter = 'none';
  }
}
