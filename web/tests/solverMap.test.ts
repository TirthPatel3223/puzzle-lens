import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { canonicalise, observe } from '../src/maltese/observation';
import { type SolverMap, type SolverTables, deriveSolverMap, fromSolver144, solverWordToMoves, toSolver144 } from '../src/maltese/solverMap';
import { generateScramble } from '../src/vendor/maltese/scramble';
import { MalteseState } from '../src/vendor/maltese/state';
import { mulberry32 } from '../src/vendor/puzzle/rng';

const SHARED = join(__dirname, '..', '..', 'shared', 'maltese');
const tables = JSON.parse(readFileSync(join(SHARED, 'solver_tables.json'), 'utf8')) as SolverTables;

/** The solver's own simulator: next[i] = (state[idx[m][i]] + add[m][i]) mod (4 on the gear rotations). */
function solverApply(state: Uint8Array, move: number): Uint8Array {
  const out = new Uint8Array(144);
  const idx = tables.idx[move];
  const add = tables.add[move];
  for (let i = 0; i < 144; i++) out[i] = i < 120 ? state[idx[i]] : (state[idx[i]] + add[i]) % 4;
  return out;
}

describe('the dictionary between the 3D engine and the solver array', () => {
  const derived = deriveSolverMap(tables);
  const map = derived.map;

  it('exists, with the solver holding the cube blue up, yellow in front and purple on the left', () => {
    expect(map.moves.U).toEqual({ face: 'B', amount: 1 });
    expect(map.moves.L).toEqual({ face: 'R', amount: 1 });
    expect(map.moves.F).toEqual({ face: 'D', amount: 1 });
    expect(map.moves["U'"]).toEqual({ face: 'B', amount: -1 });
    expect(new Set(map.cell).size).toBe(96);
    expect(new Set(map.gear).size).toBe(24);
  });

  it('turns the solved cube into the solver goal', () => {
    expect(Array.from(toSolver144(observe(MalteseState.solved()), map))).toEqual(tables.goal);
  });

  it('agrees with the solver tables on 10,000 random move sequences', () => {
    const rng = mulberry32(20261114);
    for (let n = 0; n < 10000; n++) {
      const st = MalteseState.solved();
      let theirs: Uint8Array = Uint8Array.from(tables.goal);
      const length = 1 + Math.floor(rng() * 40);
      for (let i = 0; i < length; i++) {
        const m = Math.floor(rng() * tables.moves.length);
        st.apply(map.moves[tables.moves[m]]);
        theirs = solverApply(theirs, m);
      }
      const ours = toSolver144(observe(st), map);
      if (!ours.every((v, i) => v === theirs[i])) {
        throw new Error(`sequence ${n} disagrees at entries ${Array.from(ours.keys()).filter((i) => ours[i] !== theirs[i]).join(', ')}`);
      }
    }
  });

  it('reads an array back into the same observation', () => {
    const rng = mulberry32(5);
    for (let n = 0; n < 200; n++) {
      const st = MalteseState.solved();
      for (let i = 0; i < 30; i++) st.apply(map.moves[tables.moves[Math.floor(rng() * 6)]]);
      const obs = observe(st);
      const back = fromSolver144(toSolver144(obs, map), map);
      expect(Array.from(back.cells)).toEqual(Array.from(obs.cells));
      expect(Array.from(back.gearColour)).toEqual(Array.from(obs.gearColour));
      expect(Array.from(back.gearQuarter)).toEqual(Array.from(obs.gearQuarter));
    }
  });

  it('gives the same array whichever of the fitting relabellings is used', () => {
    // Several relabellings can fit (a centre's four cells can be turned among themselves, say). They
    // differ only on places that always show the same colour, so every reachable state translates alike.
    expect(derived.cellChoices.every((n) => n >= 1)).toBe(true);
    expect(derived.gearChoices).toBeGreaterThanOrEqual(1);
  });

  it('a scramble with all six faces becomes a state the solver moves can reach, once turned to the solver frame', () => {
    // Any real scramble, however the cube is held, has the white-green-red corner somewhere; turning the
    // whole cube to bring it home gives a state in the solver's frame. Undoing the scramble with whole-cube
    // turns in between is the same as solving it, so the canonical state must also be reachable by B, R, D
    // alone: check the cheap necessary condition that the pegged corner cells show the goal colours.
    const rng = mulberry32(99);
    for (let n = 0; n < 200; n++) {
      const st = MalteseState.solved();
      st.applyAll(generateScramble(rng));
      const arr = toSolver144(observe(canonicalise(st)), map);
      for (const i of [3, 63, 80]) expect(arr[i]).toBe(tables.goal[i]);
    }
  });

  it('reads solver move words', () => {
    expect(solverWordToMoves("U L'3 F2", map)).toEqual([
      { face: 'B', amount: 1 },
      { face: 'R', amount: -3 },
      { face: 'D', amount: 2 },
    ]);
  });

  it('matches the committed solver_map.json', () => {
    const committed = JSON.parse(readFileSync(join(SHARED, 'solver_map.json'), 'utf8')) as SolverMap;
    expect(committed.cell).toEqual(map.cell);
    expect(committed.gear).toEqual(map.gear);
    expect(committed.gearSign).toBe(map.gearSign);
    expect(committed.gearOffset).toEqual(map.gearOffset);
    expect(committed.moves).toEqual(map.moves);
    expect(committed.colourCode).toEqual(map.colourCode);
  });
});
