import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildLayout } from '../src/maltese/layout';
import {
  CENTRE_CELLS,
  FACE_RIGHT,
  FACE_UP,
  GEAR_SITE,
  canonicalise,
  colourOf,
  fromObservation,
  observe,
  observeLabels,
  radialCells,
} from '../src/maltese/observation';
import { colourAt, depthAt, rasterise } from '../src/maltese/rasterise';
import { applyObservationMove, observationMoveOf } from '../src/maltese/solverMap';
import { COLORS } from '../src/vendor/maltese/config';
import { CENTRE_SLOT_COUNT, FACES, FACE_NORMAL, type Face, GEAR_SLOTS, GEAR_TILT, RADIAL_SLOTS } from '../src/vendor/maltese/geometry';
import { CUBE_GROUP, PIECES } from '../src/vendor/maltese/model';
import { buildCoreMesh, buildMalteseVisuals, setPieceTransform } from '../src/vendor/maltese/render/meshes';
import { GEAR_CAP_NORMAL, buildMalteseGeometry } from '../src/vendor/maltese/render/pieces';
import { generateScramble } from '../src/vendor/maltese/scramble';
import { MalteseState } from '../src/vendor/maltese/state';
import { mulberry32 } from '../src/vendor/puzzle/rng';
import { dot, mat3Vec } from '../src/vendor/puzzle/vec';
import { loadManifoldKernel } from '../src/vendor/render/manifold';

const scrambled = (seed: number, length = 25): MalteseState => {
  const st = MalteseState.solved();
  st.applyAll(generateScramble(mulberry32(seed), length));
  return st;
};
const same = (a: ReturnType<typeof observe>, b: ReturnType<typeof observe>) =>
  a.cells.every((v, i) => v === b.cells[i]) && a.gearColour.every((v, i) => v === b.gearColour[i]) && a.gearQuarter.every((v, i) => v === b.gearQuarter[i]);

describe('observation of the engine state', () => {
  it('the solved cube shows each face in its own colour, every gear flush', () => {
    const obs = observe(MalteseState.solved());
    FACES.forEach((f, fi) => {
      for (let i = 0; i < 16; i++) expect(obs.cells[16 * fi + i]).toBe(colourOf(f));
      for (let i = 0; i < 4; i++) {
        expect(obs.gearColour[4 * fi + i]).toBe(colourOf(f));
        expect(obs.gearQuarter[4 * fi + i]).toBe(0);
      }
    });
  });

  it('fills every cell and gear place exactly once, in every state', () => {
    for (let seed = 1; seed <= 50; seed++) {
      const lab = observeLabels(scrambled(seed));
      expect(new Set(lab.cells).size).toBe(96);
      expect(lab.cells.every((v) => v >= 0)).toBe(true);
      expect(new Set(lab.gears).size).toBe(24);
    }
  });

  it('every colour appears on 4 stars, 4 gears and 12 centre-or-edge cells (4 of its centre, 8 of its four edges)', () => {
    const corner = new Set([0, 3, 12, 15]);
    for (let seed = 1; seed <= 20; seed++) {
      const obs = observe(scrambled(seed));
      for (let c = 0; c < 6; c++) {
        expect(obs.cells.filter((v, i) => v === c && corner.has(i % 16)).length).toBe(4);
        expect(obs.cells.filter((v, i) => v === c && !corner.has(i % 16)).length).toBe(12);
        expect(obs.gearColour.filter((v) => v === c).length).toBe(4);
      }
    }
  });

  it('U R from solved: the FR edge lies in the UF slot turned 90°, both colours side by side on both faces', () => {
    const st = MalteseState.solved();
    st.apply({ face: 'U', amount: 1 });
    st.apply({ face: 'R', amount: 1 });
    const obs = observe(st);
    const uf = RADIAL_SLOTS.findIndex((s) => s.name === 'UF');
    const shown = radialCells(uf).map((c) => ({ face: Math.floor(c / 16), colour: obs.cells[c] }));
    for (const face of [colourOf('U'), colourOf('F')]) {
      const onFace = shown.filter((s) => s.face === face).map((s) => s.colour).sort();
      expect(onFace).toEqual([colourOf('F'), colourOf('R')].sort());
    }
  });

  it('an edge standing in a centre slot shows two cells of each colour, split along its ridge', () => {
    let seen = 0;
    for (let seed = 1; seed <= 30; seed++) {
      const st = scrambled(seed);
      const obs = observe(st);
      for (const p of PIECES) {
        if (p.kind !== 'edge' || st.slot[p.id] >= CENTRE_SLOT_COUNT) continue;
        const face = st.slot[p.id];
        const [a, b, c, d] = CENTRE_CELLS.map((i) => obs.cells[16 * face + i]); // 5 6 / 9 10
        expect([a, b, c, d].sort()).toEqual([colourOf(p.colors[0]), colourOf(p.colors[0]), colourOf(p.colors[1]), colourOf(p.colors[1])].sort());
        expect((a === b && c === d) || (a === c && b === d)).toBe(true);
        seen++;
      }
    }
    expect(seen).toBeGreaterThan(20);
  });

  it('a state is recovered from what it shows', () => {
    for (let seed = 1; seed <= 100; seed++) {
      const st = scrambled(seed, 40);
      const back = fromObservation(observe(st));
      expect(back.problems).toEqual([]);
      expect(back.state!.poseProblems()).toEqual([]);
      expect(same(observe(back.state!), observe(st))).toBe(true);
      // Corners, centres and edges are told apart by their colours, so their poses come back exactly
      // (a centre's own quarter turn is invisible; it comes back as 0).
      for (const p of PIECES) {
        if (p.kind === 'gear') continue;
        expect(back.state!.slot[p.id]).toBe(st.slot[p.id]);
        if (p.kind !== 'centre') expect(back.state!.ori[p.id]).toBe(st.ori[p.id]);
      }
    }
  });

  it('reports what is wrong when the cells cannot be a cube', () => {
    const obs = observe(MalteseState.solved());
    const cells = obs.cells.slice();
    cells[0] = colourOf('D'); // a star of the wrong colour: no white-yellow-... corner like that
    const bad = fromObservation({ ...obs, cells });
    expect(bad.state).toBeNull();
    expect(bad.problems.length).toBeGreaterThan(0);
    expect(bad.problems[0].cells).toContain(0);
    const gearColour = obs.gearColour.slice();
    gearColour[0] = colourOf('D');
    const fiveYellow = fromObservation({ ...obs, gearColour });
    expect(fiveYellow.state).toBeNull();
    expect(fiveYellow.problems[0].message).toMatch(/more than four yellow gears/);
  });

  it('moves act on observations as on states, and turning the whole cube changes nothing but the frame', () => {
    const rng = mulberry32(3);
    for (let n = 0; n < 60; n++) {
      const st = scrambled(100 + n);
      const face = FACES[Math.floor(rng() * 6)];
      const amount = rng() < 0.5 ? 1 : -1;
      const moved = st.clone();
      moved.apply({ face, amount });
      expect(same(applyObservationMove(observe(st), observationMoveOf({ face, amount })), observe(moved))).toBe(true);
      // A whole-cube rotation and its inverse.
      const rot = Math.floor(rng() * 24);
      expect(st.rotated(rot).rotated(CUBE_GROUP.inv(rot)).equals(st)).toBe(true);
      expect(st.rotated(rot).poseProblems()).toEqual([]);
    }
  });

  it('whole-cube rotations commute with moves: turning the cube, then the face that arrived, is the same', () => {
    const rng = mulberry32(8);
    for (let n = 0; n < 60; n++) {
      const st = scrambled(300 + n);
      const rot = Math.floor(rng() * 24);
      const face = FACES[Math.floor(rng() * 6)];
      const n2 = mat3Vec(CUBE_GROUP.elements[rot], FACE_NORMAL[face]);
      const face2 = FACES.find((f) => dot(FACE_NORMAL[f], n2) > 0.5)!;
      const a = st.clone();
      a.apply({ face, amount: 1 });
      const b = st.rotated(rot);
      b.apply({ face: face2, amount: 1 });
      expect(a.rotated(rot).equals(b)).toBe(true);
    }
  });

  it('canonicalise brings the white-green-red corner home whatever the cube was turned to', () => {
    const layout = buildLayout();
    for (let seed = 1; seed <= 40; seed++) {
      const st = scrambled(seed);
      const c = canonicalise(st);
      const obs = observe(c);
      layout.pegged.cells.forEach((cell, i) => expect(obs.cells[cell]).toBe(layout.pegged.colours[i]));
      for (let rot = 0; rot < 24; rot += 5) expect(canonicalise(st.rotated(rot)).equals(c)).toBe(true);
    }
  });
});

describe('the layout exported for Python', () => {
  const layout = buildLayout();

  it('lists, for every slot, exactly what each piece would show there', () => {
    expect(layout.corners).toHaveLength(8);
    for (const c of layout.corners) expect(c.options).toHaveLength(24);
    expect(layout.radial).toHaveLength(18);
    for (const r of layout.radial) expect(r.options).toHaveLength(6 + 12 * 4);
    expect(layout.gears).toHaveLength(24);
    // Every option is a different picture, so a slot's cells identify the piece and its turn.
    for (const r of layout.radial) expect(new Set(r.options.map((o) => o.colours.join())).size).toBe(r.options.length);
    for (const c of layout.corners) expect(new Set(c.options.map((o) => o.colours.join())).size).toBe(c.options.length);
  });

  it('agrees with observe() on random states', () => {
    for (let seed = 1; seed <= 30; seed++) {
      const st = scrambled(seed);
      const obs = observe(st);
      for (const c of layout.corners) {
        const piece = PIECES.find((p) => p.kind === 'corner' && st.slot[p.id] === c.slot)!;
        const opt = c.options.find((o) => o.piece === piece.id && o.ori === st.ori[piece.id])!;
        expect(c.cells.map((i) => obs.cells[i])).toEqual(opt.colours);
      }
      for (const r of layout.radial) {
        const piece = PIECES.find((p) => (p.kind === 'centre' || p.kind === 'edge') && st.slot[p.id] === r.slot)!;
        const opt = r.options.find((o) => o.piece === piece.id && o.k === (piece.kind === 'centre' ? 0 : st.ori[piece.id]))!;
        expect(r.cells.map((i) => obs.cells[i])).toEqual(opt.colours);
      }
    }
  });

  it('has moves and rotations that are permutations', () => {
    expect(Object.keys(layout.moves)).toHaveLength(12);
    expect(layout.rotations).toHaveLength(24);
    for (const m of [...Object.values(layout.moves), ...layout.rotations]) {
      expect(new Set(m.cells).size).toBe(96);
      expect(new Set(m.gears).size).toBe(24);
    }
    // Turning the whole cube never tilts a gear.
    for (const r of layout.rotations) expect(r.dk.every((d) => d === 0)).toBe(true);
  });
});

/**
 * The labels have to mean what a camera sees. The pieces are built exactly as the app builds them and
 * drawn face-on; at a point on each cell's sticker the picture must show the colour observe() gives.
 */
describe('labels against the rendered puzzle', () => {
  let visuals: ReturnType<typeof buildMalteseVisuals>;
  let root: THREE.Group;
  const layout = buildLayout();
  const U = 1 / 30;
  const hexOf = (colour: number) => new THREE.Color(COLORS[FACES[colour]]).getHexString();

  beforeAll(async () => {
    const wasm = await loadManifoldKernel();
    const geo = buildMalteseGeometry(wasm, null);
    visuals = buildMalteseVisuals(geo);
    root = new THREE.Group();
    for (const v of visuals) root.add(v.group);
    root.add(buildCoreMesh(geo));
  }, 120_000);

  const viewOf = (f: Face) => [FACE_RIGHT[f], FACE_UP[f], FACE_NORMAL[f]] as const;
  const show = (st: MalteseState) => {
    for (const v of visuals) setPieceTransform(v, st.matrix(v.pieceId));
  };

  /** Where to look for each cell's colour: it depends on the piece that fills the cell. */
  function samplePoints(st: MalteseState): Map<number, [number, number]> {
    const pts = new Map<number, [number, number]>();
    for (let f = 0; f < 6; f++) for (const c of layout.grid.cornerCells) pts.set(16 * f + c, layout.geometry.cellPoints[c] as [number, number]);
    for (const p of PIECES) {
      if (p.kind !== 'centre' && p.kind !== 'edge') continue;
      const slot = st.slot[p.id];
      for (const cell of radialCells(slot)) {
        const f = Math.floor(cell / 16);
        const local = cell % 16;
        if (slot < CENTRE_SLOT_COUNT) {
          // A centre's square sticker, or the two roof stickers of an edge standing there: 1.5 mm off the middle.
          const col = local % 4;
          const row = Math.floor(local / 4);
          pts.set(cell, [(col === 2 ? 1.5 : -1.5) * U, (row === 1 ? 1.5 : -1.5) * U]);
          continue;
        }
        const side = layout.grid.edgeCells.findIndex((pair) => pair.includes(local));
        const second = layout.grid.edgeCells[side][1] === local;
        // An edge's roof sticker is 27.3 mm out; a centre sunk into the slot is seen 21.2 mm out, deep down.
        const out = (p.kind === 'edge' ? 27.3 : 21.2) * U;
        const along = (second ? 1 : -1) * (p.kind === 'edge' ? 1.5 : 1.0) * U;
        pts.set(cell, side === 0 ? [along, out] : side === 2 ? [along, -out] : side === 1 ? [out, -along] : [-out, -along]);
        void f;
      }
    }
    return pts;
  }

  it('shows the labelled colour at every cell and every gear hub, face-on, in scrambled states', () => {
    let checked = 0;
    for (const seed of [0, 1, 2, 3, 4, 5, 6, 7]) {
      const st = seed === 0 ? MalteseState.solved() : scrambled(seed, 30);
      show(st);
      const obs = observe(st);
      const pts = samplePoints(st);
      FACES.forEach((face, fi) => {
        const r = rasterise(root, viewOf(face), 640);
        for (let c = 0; c < 16; c++) {
          const [x, y] = pts.get(16 * fi + c)!;
          const seen = colourAt(r, x, y).replace('#', '');
          if (seen !== hexOf(obs.cells[16 * fi + c])) {
            throw new Error(`seed ${seed}, face ${face}, cell ${c}: the picture shows #${seen} where the label says ${layout.colours[obs.cells[16 * fi + c]].name}`);
          }
          checked++;
        }
        layout.geometry.gearPositions.forEach(([x, y], pos) => {
          const seen = colourAt(r, x, y).replace('#', '');
          expect(seen).toBe(hexOf(obs.gearColour[4 * fi + pos]));
          checked++;
        });
      });
    }
    expect(checked).toBe(8 * 6 * 20);
  }, 300_000);

  it('a gear quarter is a tilt the camera can see: flush at 0, proud on the side its cap leans away from otherwise', () => {
    const d = 1.2 * U;
    let tilted = 0;
    for (const seed of [1, 2, 3]) {
      const st = scrambled(seed, 30);
      show(st);
      const obs = observe(st);
      FACES.forEach((face, fi) => {
        const r = rasterise(root, viewOf(face), 640);
        for (const p of PIECES) {
          if (p.kind !== 'gear') continue;
          const slot = st.slot[p.id];
          if (GEAR_SLOTS[slot].face !== face) continue;
          const site = GEAR_SITE[slot];
          const [hx, hy] = layout.geometry.gearPositions[site % 4];
          const k = obs.gearQuarter[site];
          const n = mat3Vec(st.matrix(p.id), GEAR_CAP_NORMAL);
          const lean: [number, number] = [dot(n, FACE_RIGHT[face]), dot(n, FACE_UP[face])];
          const size = Math.hypot(lean[0], lean[1]);
          if (k === 0) {
            expect(size).toBeLessThan(1e-9);
            for (const [ax, ay] of [[d, 0], [-d, 0], [0, d], [0, -d]]) expect(Math.abs(depthAt(r, hx + ax, hy + ay) - 1)).toBeLessThan(0.004);
            continue;
          }
          // k = 2 leans the cap 45° (twice the gear axis' 22.5°); k = 1 and 3 lean it about 31°.
          const tilt = (Math.asin(size) * 180) / Math.PI;
          expect(tilt).toBeCloseTo(k === 2 ? (2 * GEAR_TILT * 180) / Math.PI : 31.4, 0);
          const ux = lean[0] / size;
          const uy = lean[1] / size;
          const high = depthAt(r, hx - d * ux, hy - d * uy);
          const low = depthAt(r, hx + d * ux, hy + d * uy);
          expect(high - low).toBeGreaterThan(d * Math.tan((tilt * Math.PI) / 180));
          tilted++;
        }
        void fi;
      });
    }
    expect(tilted).toBeGreaterThan(20);
  }, 300_000);
});
