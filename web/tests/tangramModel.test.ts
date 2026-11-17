import { describe, expect, it } from 'vitest';
import { generateScramble, mulberry32 } from '../src/vendor/tangram/moves';
import { PuzzleState } from '../src/vendor/tangram/state';
import { dot, mat3Vec, sub } from '../src/vendor/puzzle/vec';
import { DIAGONAL, EMPTY, FACE_MIN_COS, FACE_VIEWS, MODEL, POSITION_COUNT, TABLES, cellKeyOf, inFaceFrame, observe, observeLabels, placementOf, stateFromOrient } from '../src/tangram/model';

const scrambled = (seed: number, rounds = 20): PuzzleState => {
  const st = PuzzleState.solved(MODEL);
  st.applyAll(generateScramble(MODEL.variant, mulberry32(seed), rounds));
  return st;
};

describe('Tangram positions, cells and placements', () => {
  it('has 120 positions: 66 on the cube (the home positions), 60 facing each turning corner', () => {
    expect(POSITION_COUNT).toBe(120);
    const homes = TABLES.positions.filter((p) => p.home);
    expect(homes.length).toBe(66);
    expect(homes.every((p) => p.onCube)).toBe(true);
    expect(TABLES.positions.filter((p) => p.onCube).length).toBe(66);
    expect(TABLES.positions.filter((p) => p.facing === 'T').length).toBe(60);
    // Every normal makes 54.7° with the diagonal: each sticker faces exactly one turning corner.
    for (const p of TABLES.positions) expect(Math.abs(dot(p.normal, DIAGONAL))).toBeCloseTo(1 / Math.sqrt(3), 6);
  });

  it('home positions come first, in piece and sticker order', () => {
    let i = 0;
    MODEL.pieces.forEach((piece, id) => {
      piece.stickers.forEach((_, k) => {
        expect(TABLES.positions[i].home).toEqual({ piece: id, sticker: k });
        i++;
      });
    });
  });

  it('every state puts exactly one piece in each of the 45 cells', () => {
    for (let seed = 1; seed <= 60; seed++) {
      const st = scrambled(seed, 1 + (seed % 25));
      const cells = MODEL.pieces.map((_, i) => placementOf(i, st.orient[i])!.cell);
      expect(new Set(cells).size).toBe(45);
      // The cell is where the piece's centre is.
      MODEL.pieces.forEach((p, i) => {
        expect(cellKeyOf(mat3Vec(st.pieceMatrix(i), p.poly.centroid))).toBe(TABLES.cells[cells[i]].key);
      });
    }
  });

  it('the move tables move what is shown exactly as the engine moves the pieces', () => {
    for (let seed = 1; seed <= 30; seed++) {
      const st = scrambled(seed, seed % 12);
      for (const axis of ['U', 'D', 'F', 'R', 'L'] as const) {
        const before = observeLabels(st);
        const after = st.clone();
        after.apply({ axis, amount: 1 });
        const want = observeLabels(after);
        const from = TABLES.moveFrom[axis];
        const got = Array.from(want, (_, i) => before[from[i]]);
        expect(got).toEqual(Array.from(want));
      }
    }
  });

  it('a solved puzzle shows its home colours on the cube and nothing off it', () => {
    const obs = observe(PuzzleState.solved(MODEL));
    TABLES.positions.forEach((p, i) => expect(obs[i] === EMPTY).toBe(!p.home));
    expect(Array.from(obs).filter((c) => c !== EMPTY).length).toBe(66);
  });

  it('stateFromOrient is the inverse of reading the orientations', () => {
    const st = scrambled(99, 30);
    expect(stateFromOrient(Array.from(st.orient)).equals(st)).toBe(true);
  });
});

describe('the six face-on photos', () => {
  const key = (v: readonly number[]) => v.map((x) => (Math.abs(x) < 5e-7 ? 0 : x).toFixed(4)).join(',');
  const facing = (f: (typeof FACE_VIEWS)[number]) => TABLES.positions.filter((p) => dot(p.normal, f.normal) >= FACE_MIN_COS);

  it('each face frame is right-handed, with the turning corner at the bottom-right', () => {
    for (const f of FACE_VIEWS) {
      const [r, u, n] = [f.right, f.up, f.normal];
      const c = [r[1] * u[2] - r[2] * u[1], r[2] * u[0] - r[0] * u[2], r[0] * u[1] - r[1] * u[0]];
      c.forEach((ci, i) => expect(ci).toBeCloseTo(n[i], 9));
      expect(f.corners[2].map((x) => Math.round(x))).toEqual(f.end === 'T' ? [1, 1, 1] : [-1, -1, -1]);
      expect(f.end).toBe(dot(n, DIAGONAL) > 0 ? 'T' : 'B');
    }
  });

  it('every face sees the same 29 positions in its own frame (11 flat, 18 turned out at 48°)', () => {
    const sets = FACE_VIEWS.map((f) =>
      facing(f)
        .map((p) => key([...inFaceFrame(f, p.centre), dot(p.normal, f.right), dot(p.normal, f.up), dot(p.normal, f.normal)]))
        .sort(),
    );
    for (const s of sets) expect(s).toEqual(sets[0]);
    const f = FACE_VIEWS[0];
    expect(facing(f).length).toBe(29);
    expect(facing(f).filter((p) => dot(p.normal, f.normal) > 0.999).length).toBe(11);
    // Every position is in at least one photo; the turned-out ones in two.
    for (const p of TABLES.positions) {
      const n = FACE_VIEWS.filter((v) => dot(p.normal, v.normal) >= FACE_MIN_COS).length;
      expect(n).toBe(p.onCube ? 1 : 2);
    }
  });

  it('look-alikes down the diagonal (same centre, turned 60°) are told apart: one is flat in a photo where the other is not', () => {
    const P = TABLES.positions;
    let pairs = 0;
    for (let i = 0; i < P.length; i++) {
      for (let j = i + 1; j < P.length; j++) {
        if (Math.hypot(...sub(P[i].centre, P[j].centre)) > 0.05) continue;
        pairs++;
        const told = FACE_VIEWS.some((f) => Math.abs(dot(P[i].normal, f.normal) - dot(P[j].normal, f.normal)) > 0.3);
        expect(told).toBe(true);
      }
    }
    expect(pairs).toBe(12);
  });
});
