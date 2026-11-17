import wasmUrl from 'manifold-3d/manifold.wasm?url';
import { FACES } from '../../vendor/maltese/geometry';
import { CUBE_GROUP } from '../../vendor/maltese/model';
import { generateScramble } from '../../vendor/maltese/scramble';
import { MalteseState } from '../../vendor/maltese/state';
import { buildMalteseGeometry, defaultFillet } from '../../vendor/maltese/render/pieces';
import { mulberry32 } from '../../vendor/puzzle/rng';
import { loadManifoldKernel } from '../../vendor/render/manifold';
import { buildLayout } from '../../maltese/layout';
import { observe } from '../../maltese/observation';
import type { DatagenApi, DatagenPage } from '../page';
import { type SampleOptions, type SampleSpec, chance, randomState, sampleSpec } from './randomise';
import { colourLayout, painted } from '../common/random';
import { Studio } from './scene';

/**
 * Synthetic photos of the Maltese Gear Cube (datagen.html?puzzle=maltese-gear-cube, the default): one
 * face roughly face-on per picture, labelled with what the whole cube shows.
 */

export interface SampleMeta {
  readonly seed: number;
  /** Index into FACES (U D F B L R) of the face photographed. */
  readonly face: number;
  /** The face's corners in the picture (fractions, y down): top-left, top-right, bottom-right, bottom-left in the face's own orientation. */
  readonly quad: number[][];
  readonly cells: number[];
  /** The cube's colour layout: paint[c] is the colour it shows where the model shows colour c (cells and gears are as shown). */
  readonly paint: number[];
  readonly gearColour: number[];
  readonly gearQuarter: number[];
  readonly slot: number[];
  readonly ori: number[];
  readonly stateKind: string;
  readonly setting: string;
  readonly framing: SampleSpec['framing'];
  readonly fov: number;
  /** Pictures of one scan share this number and are shots 0-5 of it: front, right, back, left, top, bottom. */
  readonly scan?: number;
  readonly shot?: number;
}

export function malteseDatagen(page: DatagenPage): DatagenApi<SampleMeta> {
  const { params, size, canvas, overlay, info } = page;
  let studio: Studio;
  const layout = buildLayout();

  function draw(seed: number): { meta: SampleMeta; spec: SampleSpec } {
    const rng = mulberry32(seed);
    const { state, kind } = randomState(rng);
    return shoot(seed, sampleSpec(seed, rng, kind), state, kind);
  }

  function shoot(seed: number, spec: SampleSpec, state: MalteseState, kind: string, extra: Partial<SampleMeta> = {}, paint = colourLayout(seed)): { meta: SampleMeta; spec: SampleSpec } {
    const { quad } = studio.render(spec, state, paint);
    const obs = observe(state);
    const meta: SampleMeta = {
      seed,
      face: spec.face,
      quad: quad.map((p) => [+p[0].toFixed(5), +p[1].toFixed(5)]),
      // the colours the cube shows (its colour layout applied)
      cells: painted(obs.cells, paint),
      gearColour: painted(obs.gearColour, paint),
      paint,
      gearQuarter: Array.from(obs.gearQuarter),
      slot: Array.from(state.slot),
      ori: Array.from(state.ori),
      stateKind: kind,
      setting: spec.setting,
      framing: spec.framing,
      fov: spec.camera.fov,
      ...extra,
    };
    return { meta, spec };
  }

  function one(seed: number, quality = 0.92) {
    const { meta } = draw(seed);
    return { meta, dataUrl: canvas.toDataURL('image/jpeg', quality) };
  }

  /** The faces of a scan in the order they are photographed (indices into FACES): F R B L with U up, then U, then D. */
  const SCAN_FACES = (['F', 'R', 'B', 'L', 'U', 'D'] as const).map((f) => FACES.indexOf(f));

  /**
   * Six pictures of one scrambled cube, as a person would take them: the same cube and room, a new
   * hand position and light direction for every face, each face the right way up (now and then one is
   * not, as happens). The cube is held any way round: the labels are what it shows held that way.
   */
  function* scanShots(scan: number): Generator<{ meta: SampleMeta; spec: SampleSpec }> {
    const rng = mulberry32(scan ^ 0x2545f491);
    const state = MalteseState.solved();
    state.applyAll(generateScramble(rng, 6 + Math.floor(rng() * 45)));
    const held = state.rotated(Math.floor(rng() * CUBE_GROUP.size));
    const look = sampleSpec(scan, mulberry32(scan ^ 0x68bc21eb), 'scramble');
    for (let shot = 0; shot < 6; shot++) {
      const seed = scan * 8 + shot;
      const options: SampleOptions = { face: SCAN_FACES[shot], quarter: chance(rng, 0.1) ? 1 + Math.floor(rng() * 3) : 0, setting: look.setting, look };
      yield shoot(seed, sampleSpec(seed, mulberry32(seed ^ 0x51ed270b), 'scramble', options), held, 'scan', { scan, shot }, colourLayout(scan * 8));
    }
  }

  async function runScans(scans: number[], quality = 0.92): Promise<number> {
    if (!window.__save) throw new Error('runScans() needs the __save binding (tools/gen_dataset.py provides it)');
    let done = 0;
    for (const scan of scans) {
      for (const { meta } of scanShots(scan)) {
        const dataUrl = canvas.toDataURL('image/jpeg', quality);
        await window.__save(JSON.stringify(meta), dataUrl.slice(dataUrl.indexOf(',') + 1));
        done++;
      }
    }
    return done;
  }

  async function run(seeds: number[], quality = 0.92): Promise<number> {
    if (!window.__save) throw new Error('run() needs the __save binding (tools/gen_dataset.py provides it)');
    let done = 0;
    for (const seed of seeds) {
      const { meta, dataUrl } = one(seed, quality);
      await window.__save(JSON.stringify(meta), dataUrl.slice(dataUrl.indexOf(',') + 1));
      done++;
    }
    return done;
  }

  /** Draw the labels over the picture: a dot of the labelled colour on every cell, the quarter on every gear. */
  function annotate(meta: SampleMeta): void {
    overlay.width = overlay.height = size;
    const ctx = overlay.getContext('2d')!;
    ctx.clearRect(0, 0, size, size);
    const px = (p: number[]) => [p[0] * size, p[1] * size] as const;
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    meta.quad.forEach((p, i) => (i ? ctx.lineTo(...px(p)) : ctx.moveTo(...px(p))));
    ctx.closePath();
    ctx.stroke();
    ctx.font = '600 13px system-ui';
    ctx.fillStyle = '#00e5ff';
    meta.quad.forEach((p, i) => ctx.fillText(['TL', 'TR', 'BR', 'BL'][i], px(p)[0] + 5, px(p)[1] - 5));
    const cells = studio.project(meta.face, layout.geometry.cellPoints);
    cells.forEach((p, c) => {
      const [x, y] = px(p);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = layout.colours[meta.cells[16 * meta.face + c]].hex;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#000';
      ctx.stroke();
    });
    const hubs = studio.project(meta.face, layout.geometry.gearPositions);
    hubs.forEach((p, g) => {
      const [x, y] = px(p);
      ctx.fillStyle = layout.colours[meta.gearColour[4 * meta.face + g]].hex;
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 3;
      ctx.font = '700 18px system-ui';
      const text = String(meta.gearQuarter[4 * meta.face + g]);
      ctx.strokeText(text, x - 5, y + 6);
      ctx.fillText(text, x - 5, y + 6);
    });
  }

  function viewer(): void {
    document.body.classList.add('viewer');
    let seed = parseInt(params.get('seed') ?? '0', 10) || 0;
    const show = () => {
      const { meta, spec } = draw(seed);
      if ((document.getElementById('labels') as HTMLInputElement).checked) annotate(meta);
      else overlay.getContext('2d')!.clearRect(0, 0, overlay.width, overlay.height);
      info.textContent =
        `seed ${seed} · face ${FACES[meta.face]} · ${meta.stateKind} · ${meta.setting} · fov ${spec.camera.fov.toFixed(0)}° · ` +
        `tilt ${spec.framing.tiltDeg.toFixed(1)}° · roll ${spec.framing.rollDeg.toFixed(1)}° + ${spec.framing.quarter}×90° · ` +
        `fill ${(spec.framing.fill * 100).toFixed(0)}% · ${spec.toneMapping} ×${spec.exposure.toFixed(2)}`;
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
      const wasm = await loadManifoldKernel(wasmUrl);
      const geo = buildMalteseGeometry(wasm, params.get('round') === '0' ? null : defaultFillet());
      studio = new Studio(canvas, geo, size);
      if (!window.__save) viewer();
      return { renderer: studio.rendererName, size, puzzle: 'maltese-gear-cube', shotsPerScan: 6 };
    })(),
    one,
    run,
    runScans,
  };
}
