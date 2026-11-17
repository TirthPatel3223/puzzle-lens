import { mulberry32 } from '../../vendor/puzzle/rng';
import type { PuzzleState } from '../../vendor/tangram/state';
import { COLOUR_HEX, EMPTY, FACE_MIN_COS, FACE_VIEWS, TABLES } from '../../tangram/model';
import { dot } from '../../vendor/puzzle/vec';
import { SKIN, colourLayout, pick, puzzleLook } from '../common/random';
import { type DatagenApi, type DatagenPage, save } from '../page';
import { type StateKind, type TangramSpec, randomState, tangramSpec } from './randomise';
import { TangramStudio } from './scene';

/**
 * Synthetic photos of the Tangram Cube Extreme (datagen.html?puzzle=tangram-cube-extreme): each picture
 * is one face roughly face-on, in any state and shape, labelled exactly.
 */

export interface TangramMeta {
  readonly seed: number;
  readonly puzzle: 'tangram-cube-extreme';
  /** The face photographed: index into FACE_IDS (+x -x +y -y +z -z) and layout.json faces. */
  readonly face: number;
  /** The face's corners in the picture (fractions, y down): top-left, top-right, bottom-right (its turning corner), bottom-left, in the face's own frame. */
  readonly corners: number[][];
  /** Per position (120): colour 0-5 or 6 (empty), as the puzzle shows it (its colour layout applied). */
  readonly content: number[];
  /** The puzzle's colour layout: paint[c] is the colour it shows where the model shows colour c. */
  readonly paint: number[];
  /** Positions facing the camera whose sticker it really sees. */
  readonly visible: number[];
  /** Orientation of every piece (the state). */
  readonly orient: number[];
  readonly stateKind: StateKind | 'scan';
  readonly setting: string;
  readonly framing: TangramSpec['framing'];
  readonly fov: number;
  /** Pictures of one scan share this number: shots 0-5 are the faces +x -x +y -y +z -z. */
  readonly scan?: number;
  readonly shot?: number;
}

const WHITE = 3;

export function tangramDatagen(page: DatagenPage): DatagenApi<TangramMeta> {
  const { params, size, canvas, overlay, info } = page;
  let studio: TangramStudio;

  function shoot(seed: number, spec: TangramSpec, state: PuzzleState, kind: TangramMeta['stateKind'], extra: Partial<TangramMeta> = {}, paint = colourLayout(seed)) {
    const r = studio.render(spec, state, paint);
    const meta: TangramMeta = {
      seed,
      puzzle: 'tangram-cube-extreme',
      face: spec.face,
      corners: r.corners.map((p) => [+p[0].toFixed(5), +p[1].toFixed(5)]),
      content: r.content,
      paint,
      visible: r.visible,
      orient: Array.from(state.orient),
      stateKind: kind,
      setting: spec.setting,
      framing: spec.framing,
      fov: spec.camera.fov,
      ...extra,
    };
    return { meta, spec };
  }

  function draw(seed: number) {
    const rng = mulberry32(seed);
    const { state, kind } = randomState(rng);
    return shoot(seed, tangramSpec(seed, rng, kind, state), state, kind);
  }

  function one(seed: number, quality = 0.92) {
    const { meta } = draw(seed);
    return { meta, dataUrl: canvas.toDataURL('image/jpeg', quality) };
  }

  /** The six photos of one scan: the same puzzle, look and room; each its own roll, tilt, light and hand. */
  function* scanShots(scan: number) {
    const rng = mulberry32(scan ^ 0x2545f491);
    const { state } = randomState(rng);
    const lookRng = mulberry32(scan ^ 0x68bc21eb);
    const look = { ...puzzleLook(lookRng, COLOUR_HEX, WHITE), toneMapping: pick(lookRng, ['none', 'linear', 'neutral', 'neutral', 'aces', 'agx'] as const), skin: pick(lookRng, SKIN) };
    const setting = (['floating', 'table-top', 'table-side', 'hand', 'hand'] as const)[Math.floor(rng() * 5)];
    for (let shot = 0; shot < 6; shot++) {
      const seed = scan * 8 + shot;
      const spec = tangramSpec(seed, mulberry32(seed ^ 0x51ed270b), 'scramble', state, { face: shot, setting, look });
      yield shoot(seed, spec, state, 'scan', { scan, shot }, colourLayout(scan * 8));       // one layout for the whole scan
    }
  }

  async function run(seeds: number[], quality = 0.92): Promise<number> {
    let done = 0;
    for (const seed of seeds) {
      const { meta } = draw(seed);
      await save(canvas, meta, quality);
      done++;
    }
    return done;
  }

  async function runScans(scans: number[], quality = 0.92): Promise<number> {
    let done = 0;
    for (const scan of scans) {
      for (const { meta } of scanShots(scan)) {
        await save(canvas, meta, quality);
        done++;
      }
    }
    return done;
  }

  /** The labels over the picture: the face's corners, and a dot on every position in the photo. */
  function annotate(meta: TangramMeta): void {
    overlay.width = overlay.height = size;
    const ctx = overlay.getContext('2d')!;
    ctx.clearRect(0, 0, size, size);
    const px = (p: readonly number[]) => [p[0] * size, p[1] * size] as const;
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    meta.corners.forEach((p, i) => (i ? ctx.lineTo(...px(p)) : ctx.moveTo(...px(p))));
    ctx.closePath();
    ctx.stroke();
    ctx.font = '600 13px system-ui';
    ctx.fillStyle = '#00e5ff';
    meta.corners.forEach((p, i) => ctx.fillText(['TL', 'TR', 'turning corner', 'BL'][i], px(p)[0] + 5, px(p)[1] - 5));
    const visible = new Set(meta.visible);
    const view = FACE_VIEWS[meta.face];
    const facing = TABLES.positions.filter((p) => dot(p.normal, view.normal) >= FACE_MIN_COS);
    const where = studio.project(facing.map((p) => p.centre));
    facing.forEach((p, j) => {
      const [x, y] = px(where[j]);
      const c = meta.content[p.index];
      ctx.beginPath();
      ctx.arc(x, y, c === EMPTY ? 2.5 : 5, 0, 2 * Math.PI);
      if (c !== EMPTY) {
        ctx.fillStyle = COLOUR_HEX[c];
        ctx.fill();
      }
      ctx.lineWidth = visible.has(p.index) ? 1.5 : 3;
      ctx.strokeStyle = c === EMPTY ? '#888' : visible.has(p.index) ? '#000' : '#ff2d55';
      ctx.stroke();
    });
  }

  function viewer(): void {
    document.body.classList.add('viewer');
    let seed = parseInt(params.get('seed') ?? '0', 10) || 0;
    const show = () => {
      const { meta, spec } = draw(seed);
      if ((document.getElementById('labels') as HTMLInputElement).checked) annotate(meta);
      else overlay.getContext('2d')!.clearRect(0, 0, overlay.width, overlay.height);
      const view = FACE_VIEWS[meta.face];
      const facing = TABLES.positions.filter((p) => dot(p.normal, view.normal) >= FACE_MIN_COS && meta.content[p.index] !== EMPTY).length;
      info.textContent =
        `seed ${seed} · face ${view.face} (${view.end} corner) · ${meta.stateKind} · ${meta.setting} · fov ${spec.camera.fov.toFixed(0)}° · ` +
        `tilt ${spec.framing.tiltDeg.toFixed(1)}° · roll ${spec.framing.rollDeg.toFixed(0)}° · fill ${(spec.framing.fill * 100).toFixed(0)}% · ` +
        `${meta.visible.filter((p) => dot(TABLES.positions[p].normal, view.normal) >= FACE_MIN_COS).length} of ${facing} stickers in the photo visible (red rings: hidden)`;
    };
    document.getElementById('next')!.addEventListener('click', () => {
      seed++;
      show();
    });
    document.getElementById('prev')!.addEventListener('click', () => {
      seed = Math.max(0, seed - 1);
      show();
    });
    document.getElementById('labels')!.addEventListener('change', show);
    show();
  }

  return {
    ready: (async () => {
      canvas.width = canvas.height = size;
      studio = new TangramStudio(canvas, size);
      if (!window.__save) viewer();
      return { renderer: studio.rendererName, size, puzzle: 'tangram-cube-extreme', shotsPerScan: 6 };
    })(),
    one,
    run,
    runScans,
  };
}

