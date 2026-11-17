import { describe, expect, it } from 'vitest';
import layoutJson from '../../shared/maltese/layout.json';
import solverMapJson from '../../shared/maltese/solver_map.json';
import { CUBE_GROUP } from '../src/vendor/maltese/model';
import { normalizeAmount } from '../src/vendor/maltese/notation';
import { MalteseState } from '../src/vendor/maltese/state';
import { mulberry32 } from '../src/vendor/puzzle/rng';
import { mat3Vec } from '../src/vendor/puzzle/vec';
import { FACE_NORMAL, faceOf } from '../src/vendor/maltese/geometry';
import { generateScramble } from '../src/vendor/maltese/scramble';
import { observe } from '../src/maltese/observation';
import { type SolverMap, fromSolver144, solverWordToMoves, toSolver144 } from '../src/maltese/solverMap';
import type { ViewPreset } from '../src/vendor/render/scene';
import {
  COLOUR_NAMES,
  HELD_VIEWS,
  HOLD_POSITION,
  SOLVER_ROWS,
  SOLVER_VIEWS,
  countDifferences,
  describeSolverMove,
  faceView,
  heldTruth,
  parseSolverWord,
  solverScramble,
  stateFromPoses,
  stateOf,
  toPlain,
} from '../src/app/maltese/model';
import fixture from './fixtures/real-scan-1.json';

const MAP = solverMapJson as unknown as SolverMap;
type Rearrangement = { cells: number[]; gears: number[]; dk: number[] };
const ROTATIONS = (layoutJson as unknown as { rotations: Rearrangement[] }).rotations;

function applyRearrangement(o: ReturnType<typeof toPlain>, r: Rearrangement) {
  return {
    cells: r.cells.map((c) => o.cells[c]),
    gearColour: r.gears.map((g) => o.gearColour[g]),
    gearQuarter: r.gears.map((g, i) => (o.gearQuarter[g] + r.dk[i]) % 4),
  };
}

function randomState(seed: number): MalteseState {
  const s = MalteseState.solved();
  s.applyAll(generateScramble(mulberry32(seed), 40));
  return s;
}

describe('frames shared with the Python side', () => {
  it('layout.json rotation r is the engine turning the whole cube by r (what the server uses to turn a scan back)', () => {
    for (let seed = 1; seed <= 6; seed++) {
      const s = randomState(seed);
      const base = toPlain(observe(s));
      for (let r = 0; r < CUBE_GROUP.size; r++) {
        expect(applyRearrangement(base, ROTATIONS[r])).toEqual(toPlain(observe(s.rotated(r))));
      }
    }
  });

  it('a decoded cube turned by the inverse rotation is the cube as photographed', () => {
    // Python: held = rotations[inv(rotation)].apply(solver-frame observation)
    for (let seed = 1; seed <= 4; seed++) {
      const canonical = randomState(100 + seed);
      for (const r of [0, 5, 13, 22]) {
        const heldState = canonical.rotated(CUBE_GROUP.inv(r));
        const held = applyRearrangement(toPlain(observe(canonical)), ROTATIONS[CUBE_GROUP.inv(r)]);
        expect(toPlain(observe(heldState))).toEqual(held);
        // and the rotation takes it back
        expect(heldState.rotated(r).equals(canonical)).toBe(true);
      }
    }
  });

  it('stateOf rebuilds what the faces show', () => {
    const s = randomState(7);
    const st = stateOf(toPlain(observe(s)))!;
    expect(st).not.toBeNull();
    expect(toPlain(observe(st))).toEqual(toPlain(observe(s)));
  });

  it('heldTruth finds how a known cube was held', () => {
    const truth = randomState(8);
    for (const r of [0, 3, 11, 19]) {
      const seen = toPlain(observe(truth.rotated(r)));
      expect(countDifferences(heldTruth(truth, seen), seen)).toBe(0);
    }
  });
});

describe('the real scan and its solution', () => {
  it('the decoder poses show exactly the 144 array', () => {
    const st = stateFromPoses(fixture.slot, fixture.ori);
    expect(Array.from(toSolver144(observe(st), MAP))).toEqual(fixture.state144);
    expect(toPlain(fromSolver144(fixture.state144, MAP))).toEqual(toPlain(observe(st)));
  });

  it("the solver's solution solves the 3D model", () => {
    const moves = parseSolverWord(fixture.word);
    expect(moves.map((m) => m.engine)).toEqual(solverWordToMoves(fixture.word, MAP).map((m) => ({ face: m.face, amount: normalizeAmount(m.amount) })));
    expect(moves.map((m) => m.engine)).toEqual(fixture.engineMoves.map((m) => ({ face: m.face, amount: normalizeAmount(m.amount) })));
    const st = stateFromPoses(fixture.slot, fixture.ori);
    expect(st.isSolved()).toBe(false);
    st.applyAll(moves.map((m) => m.engine));
    expect(st.isSolved()).toBe(true);
  });
});

describe('solver moves', () => {
  it('reads every way the solver writes a move', () => {
    const m = parseSolverWord("U L' F2 U'2 L2' F4 L'3");
    expect(m.map((x) => [x.letter, x.quarters])).toEqual([
      ['U', 1],
      ['L', -1],
      ['F', 2],
      ['U', -2],
      ['L', -2],
      ['F', 4],
      ['L', -3],
    ]);
    expect(m.map((x) => x.engine.face)).toEqual(['B', 'R', 'D', 'B', 'R', 'D', 'R']);
    expect(describeSolverMove(m[6])).toBe('Left (purple) face: 3 quarter turns counterclockwise');
    expect(describeSolverMove(m[0])).toBe('Top (blue) face: a quarter turn clockwise');
    expect(() => parseSolverWord('R')).toThrow();
  });

  it('agrees with the dictionary derived from the move tables', () => {
    for (const [letter, v] of Object.entries(MAP.moves)) {
      if (letter.endsWith("'")) continue;
      expect(parseSolverWord(letter)[0].engine).toEqual({ face: v.face, amount: v.amount });
    }
  });

  it('a practice scramble never moves the white-green-red corner', () => {
    // The solver's U, L, F are the engine's B, R, D, none of which touches the engine's UFL corner.
    const s = MalteseState.solved();
    s.applyAll(solverScramble(mulberry32(5), 60).map((m) => m.engine));
    const st = observe(s);
    const home = observe(MalteseState.solved());
    // UFL corner cells: U face (index 0) bottom-left cell 12, F face (2) top-left cell 0, L face (4) top-right cell 3.
    const ufl = [16 * 0 + 12, 16 * 2 + 0, 16 * 4 + 3];
    expect(ufl.map((c) => st.cells[c])).toEqual(ufl.map((c) => home.cells[c]));
    expect(s.isSolved()).toBe(false);
  });
});

describe('views', () => {
  const det = (r: number[][]) =>
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);

  it('every face view is a rotation that puts that face towards the viewer', () => {
    for (const f of ['U', 'D', 'F', 'B', 'L', 'R'] as const) {
      const rows = faceView(f).rows;
      expect(det(rows)).toBe(1);
      const n = FACE_NORMAL[f];
      expect(rows.map((row) => row[0] * n[0] + row[1] * n[1] + row[2] * n[2])).toEqual([0, 0, 1]);
    }
  });

  it("the solver's view: blue (engine B) up, yellow (D) in front, purple (R) on the left", () => {
    expect(det(SOLVER_ROWS as number[][])).toBe(1);
    const world = (v: readonly number[]) => SOLVER_ROWS.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
    expect(world(FACE_NORMAL.B)).toEqual([0, 1, 0]);
    expect(world(FACE_NORMAL.D)).toEqual([0, 0, 1]);
    expect(world(FACE_NORMAL.R)).toEqual([-1, 0, 0]);
    expect(HOLD_POSITION.B).toBe('top');
  });

  /** The faces a preset shows: those whose outward normal, once turned by the rows, points at the camera. */
  const facesSeen = (v: ViewPreset) => {
    const az = (v.azimuth * Math.PI) / 180;
    const el = (v.elevation * Math.PI) / 180;
    const eye = [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
    return (['U', 'D', 'F', 'B', 'L', 'R'] as const)
      .filter((f) => {
        const n = FACE_NORMAL[f];
        const w = v.rows.map((row) => row[0] * n[0] + row[1] * n[1] + row[2] * n[2]);
        return w[0] * eye[0] + w[1] * eye[1] + w[2] * eye[2] > 1e-9;
      })
      .sort();
  };
  const upFace = (v: ViewPreset) =>
    (['U', 'D', 'F', 'B', 'L', 'R'] as const).find((f) => v.rows[1].every((x, i) => x === FACE_NORMAL[f][i]));

  it("solving, Corner: the blue-yellow-red corner (engine B, D, L) towards you, blue on top", () => {
    expect(facesSeen(SOLVER_VIEWS.corner)).toEqual(['B', 'D', 'L']);
    expect(upFace(SOLVER_VIEWS.corner)).toBe('B');
    expect(['B', 'D', 'L'].map((f) => COLOUR_NAMES[['U', 'D', 'F', 'B', 'L', 'R'].indexOf(f)])).toEqual(['blue', 'yellow', 'red']);
  });

  it('solving, Face-on: only yellow (engine D) towards you, blue on top', () => {
    expect(facesSeen(SOLVER_VIEWS.face)).toEqual(['D']);
    expect(upFace(SOLVER_VIEWS.face)).toBe('B');
  });

  it("checking: the simulator site's Corner, Face-on and Edge views of the frame shown", () => {
    expect(facesSeen(HELD_VIEWS.corner)).toEqual(['F', 'R', 'U']);
    expect(facesSeen(HELD_VIEWS.face)).toEqual(['F']);
    expect(facesSeen(HELD_VIEWS.edge)).toEqual(['F', 'U']);
    for (const v of Object.values(HELD_VIEWS)) expect(upFace(v)).toBe('U');
  });

  it('where each photographed face is while solving follows the decoder rotation', () => {
    // For a cube photographed turned by inv(r), the face the scan calls f is at faceOf(R_r n_f) in the solver's frame.
    const canonical = randomState(9);
    const r = 17;
    const held = canonical.rotated(CUBE_GROUP.inv(r));
    const R = CUBE_GROUP.elements[r];
    const ho = observe(held);
    const co = observe(canonical);
    for (const f of ['U', 'D', 'F', 'B', 'L', 'R'] as const) {
      const g = faceOf(mat3Vec(R, FACE_NORMAL[f]))!;
      const fi = ['U', 'D', 'F', 'B', 'L', 'R'].indexOf(f);
      const gi = ['U', 'D', 'F', 'B', 'L', 'R'].indexOf(g);
      // The centre slot of face f (cells 5, 6, 9, 10) holds the same piece as the centre slot of g.
      const centre = (o: typeof ho, k: number) => [5, 6, 9, 10].map((c) => o.cells[16 * k + c]).sort();
      expect(centre(ho, fi)).toEqual(centre(co, gi));
    }
  });
});
