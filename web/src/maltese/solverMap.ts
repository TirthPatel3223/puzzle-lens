import { FACES, type Face } from '../vendor/maltese/geometry';
import type { Move } from '../vendor/maltese/notation';
import { MalteseState } from '../vendor/maltese/state';
import { CELL_COUNT, GEAR_COUNT, type Observation, observe, observeLabels } from './observation';

/**
 * The dictionary between the 3D engine and the solver's 144-entry array.
 *
 * The two were written separately and agree on nothing by convention: the solver holds the cube blue
 * up and yellow in front, numbers its cells and gears its own way, and counts gear rotation in the
 * other direction from its own zero. So the dictionary is not written down, it is *derived*: it is
 * the one relabelling under which the engine's moves and the solver's move tables do the same thing.
 * If no such relabelling exists the two model different puzzles, and `deriveSolverMap` says so.
 */

/** The solver's tables, as exported by tools/export_solver_tables.py. */
export interface SolverTables {
  readonly moves: readonly string[];
  /** next[i] = (state[idx[m][i]] + add[m][i]) mod (4 for the 24 gear rotations). */
  readonly idx: readonly (readonly number[])[];
  readonly add: readonly (readonly number[])[];
  readonly goal: readonly number[];
}

export interface SolverMap {
  /** Solver colour code of each colour index (FACES order: white, yellow, green, blue, red, purple). */
  readonly colourCode: readonly number[];
  /** The engine move each solver move is, e.g. U -> B (the solver's "up" face is the blue one). */
  readonly moves: Readonly<Record<string, { readonly face: Face; readonly amount: 1 | -1 }>>;
  /** Solver index (0-95) of each of our 96 cells. */
  readonly cell: readonly number[];
  /** Solver gear number (0-23) of each of our 24 gear places. */
  readonly gear: readonly number[];
  /** Solver rotation = gearSign · quarter + gearOffset[place] (mod 4). */
  readonly gearSign: 1 | -1;
  readonly gearOffset: readonly number[];
}

export const STATE_DIM = 144;
const SOLVER_CELLS = 96;
const GEAR_POS = 96;
const GEAR_ROT = 120;

/**
 * Colour codes of the solver (0 White, 1 Yellow, 2 Blue, 3 Green, 4 Purple, 5 Red; the user's
 * convention), by our colour index (white, yellow, green, blue, red, purple).
 */
export const SOLVER_COLOUR_CODE: readonly number[] = [0, 1, 3, 2, 5, 4];

/** How one engine quarter turn rearranges what is shown: new[i] = old[cells[i]], and for gears new quarter = old quarter + dk. */
export interface ObservationMove {
  readonly cells: readonly number[];
  readonly gears: readonly number[];
  readonly dk: readonly number[];
}

/** The rearrangement of cells and gear places made by taking the solved state to `after`. */
export function observationMove(after: MalteseState): ObservationMove {
  const a = observeLabels(MalteseState.solved());
  const b = observeLabels(after);
  const cellAt = new Map<number, number>();
  a.cells.forEach((label, i) => cellAt.set(label, i));
  const gearAt = new Map<number, number>();
  a.gears.forEach((label, i) => gearAt.set(label, i));
  const quarter = observe(after).gearQuarter;
  return {
    cells: Array.from(b.cells, (label) => cellAt.get(label)!),
    gears: Array.from(b.gears, (label) => gearAt.get(label)!),
    // From solved every gear starts at quarter 0, so what a place shows afterwards is the change itself.
    dk: Array.from(quarter),
  };
}

export function observationMoveOf(move: Move): ObservationMove {
  const st = MalteseState.solved();
  st.apply(move);
  return observationMove(st);
}

/** Apply a rearrangement to an observation. */
export function applyObservationMove(obs: Observation, m: ObservationMove): Observation {
  return {
    cells: Uint8Array.from(m.cells, (from) => obs.cells[from]),
    gearColour: Uint8Array.from(m.gears, (from) => obs.gearColour[from]),
    gearQuarter: Uint8Array.from(m.gears, (from, i) => (obs.gearQuarter[from] + m.dk[i]) & 3),
  };
}

/** Orbits of 0..n-1 under the given maps. */
function orbits(n: number, maps: readonly (readonly number[])[]): number[][] {
  const seen = new Array<boolean>(n).fill(false);
  const out: number[][] = [];
  for (let s = 0; s < n; s++) {
    if (seen[s]) continue;
    const orbit = [s];
    seen[s] = true;
    for (let i = 0; i < orbit.length; i++) {
      for (const m of maps) {
        const t = m[orbit[i]];
        if (!seen[t]) {
          seen[t] = true;
          orbit.push(t);
        }
      }
    }
    out.push(orbit);
  }
  return out;
}

/**
 * Every relabelling pi of one orbit with pi(ours[g][c]) = theirs[g][pi(c)] for all generators g that
 * also carries our colouring onto theirs. Fixing the image of the orbit's first place fixes the rest,
 * so this tries each candidate image once.
 */
function matchOrbit(
  orbit: readonly number[],
  ours: readonly (readonly number[])[],
  theirs: readonly (readonly number[])[],
  ourColour: (i: number) => number,
  theirColour: (i: number) => number,
  theirCount: number,
): Map<number, number>[] {
  const found: Map<number, number>[] = [];
  for (let image = 0; image < theirCount; image++) {
    if (theirColour(image) !== ourColour(orbit[0])) continue;
    const pi = new Map<number, number>([[orbit[0], image]]);
    const queue = [orbit[0]];
    let ok = true;
    for (let i = 0; i < queue.length && ok; i++) {
      const c = queue[i];
      for (let g = 0; g < ours.length && ok; g++) {
        const next = ours[g][c];
        const to = theirs[g][pi.get(c)!];
        const had = pi.get(next);
        if (had === undefined) {
          pi.set(next, to);
          queue.push(next);
        } else if (had !== to) ok = false;
      }
    }
    if (!ok || pi.size !== orbit.length) continue;
    if (new Set(pi.values()).size !== pi.size) continue;
    if (orbit.some((c) => ourColour(c) !== theirColour(pi.get(c)!))) continue;
    found.push(pi);
  }
  return found;
}

export interface Derivation {
  readonly map: SolverMap;
  /** How many relabellings fit, per orbit of cells and for the gears. More than one is harmless: see the tests. */
  readonly cellChoices: readonly number[];
  readonly gearChoices: number;
}

/**
 * Derive the dictionary from the solver's tables. Throws, with the reason, when the engine and the
 * tables cannot be the same puzzle.
 */
export function deriveSolverMap(tables: SolverTables): Derivation {
  const goal = tables.goal;
  const solved = observe(MalteseState.solved());
  const code = SOLVER_COLOUR_CODE;

  // Which engine face each solver move turns: the face whose 16 cells the move keeps among themselves.
  const base = tables.moves.filter((m) => !m.endsWith("'"));
  const faceOfMove = new Map<string, Face>();
  for (const name of base) {
    const idx = tables.idx[tables.moves.indexOf(name)];
    let block = -1;
    for (let f = 0; f < 6; f++) {
      const own = Array.from({ length: 16 }, (_, i) => 16 * f + i);
      if (own.every((i) => Math.floor(idx[i] / 16) === f) && own.some((i) => idx[i] !== i)) block = f;
    }
    if (block < 0) throw new Error(`solver move ${name} turns no face in place`);
    const colour = code.indexOf(goal[16 * block]);
    faceOfMove.set(name, FACES[colour]);
  }

  const failures: string[] = [];
  // The turning direction is not assumed: each solver move is tried as the engine's clockwise and
  // anticlockwise turn, and exactly one combination may fit.
  const fits: Derivation[] = [];
  for (let mask = 0; mask < 1 << base.length; mask++) {
    const amounts = base.map((_, i) => ((mask >> i) & 1 ? -1 : 1) as 1 | -1);
    const ourMoves = base.map((name, i) => observationMoveOf({ face: faceOfMove.get(name)!, amount: amounts[i] }));
    const theirIdx = base.map((name) => tables.idx[tables.moves.indexOf(name)]);
    const theirAdd = base.map((name) => tables.add[tables.moves.indexOf(name)]);
    const label = base.map((name, i) => `${name}=${faceOfMove.get(name)}${amounts[i] < 0 ? "'" : ''}`).join(' ');

    // Cells.
    const ourCells = ourMoves.map((m) => m.cells);
    const theirCells = theirIdx.map((idx) => idx.slice(0, SOLVER_CELLS));
    const cell = new Array<number>(CELL_COUNT).fill(-1);
    const cellChoices: number[] = [];
    let ok = true;
    for (const orbit of orbits(CELL_COUNT, ourCells)) {
      const options = matchOrbit(
        orbit,
        ourCells,
        theirCells,
        (i) => code[solved.cells[i]],
        (i) => goal[i],
        SOLVER_CELLS,
      );
      // A fixed place matches every fixed solver cell of its colour; the colour then has to be unique.
      const usable = options.filter((pi) => [...pi.values()].every((v) => !cell.includes(v)));
      if (!usable.length) {
        ok = false;
        break;
      }
      cellChoices.push(usable.length);
      for (const [c, s] of usable[0]) cell[c] = s;
    }
    if (!ok) {
      failures.push(`${label}: the sticker cells do not match`);
      continue;
    }

    // Gears: places first, then rotation = sign · quarter + offset.
    const ourGears = ourMoves.map((m) => m.gears);
    const theirGears = theirIdx.map((idx) => idx.slice(GEAR_POS, GEAR_ROT).map((v) => v - GEAR_POS));
    const gearOrbits = orbits(GEAR_COUNT, ourGears);
    const perOrbit = gearOrbits.map((orbit) =>
      matchOrbit(
        orbit,
        ourGears,
        theirGears,
        (i) => code[solved.gearColour[i]],
        (i) => goal[GEAR_POS + i],
        GEAR_COUNT,
      ),
    );
    if (perOrbit.some((o) => !o.length)) {
      failures.push(`${label}: the gear places do not match`);
      continue;
    }
    // All combinations of the per-orbit relabellings (normally there is one orbit).
    let combos: Map<number, number>[] = [new Map()];
    for (const options of perOrbit) combos = combos.flatMap((c) => options.map((o) => new Map([...c, ...o])));
    const gearFits: { gear: number[]; sign: 1 | -1; offset: number[] }[] = [];
    for (const rho of combos) {
      if (new Set(rho.values()).size !== GEAR_COUNT) continue;
      const gear = Array.from({ length: GEAR_COUNT }, (_, i) => rho.get(i)!);
      const offset = gear.map((g) => goal[GEAR_ROT + g]);
      for (const sign of [1, -1] as const) {
        let good = true;
        for (let g = 0; g < ourMoves.length && good; g++) {
          for (let site = 0; site < GEAR_COUNT && good; site++) {
            const from = ourGears[g][site];
            const lhs = sign * ourMoves[g].dk[site] + offset[site];
            const rhs = offset[from] + theirAdd[g][GEAR_ROT + gear[site]];
            if ((((lhs - rhs) % 4) + 4) % 4 !== 0) good = false;
          }
        }
        if (good) gearFits.push({ gear, sign, offset });
      }
    }
    if (!gearFits.length) {
      failures.push(`${label}: cells and gear places match, but no rule "rotation = ±quarter + offset" fits the gear rotations`);
      continue;
    }
    const moves: Record<string, { face: Face; amount: 1 | -1 }> = {};
    base.forEach((name, i) => {
      moves[name] = { face: faceOfMove.get(name)!, amount: amounts[i] };
      moves[`${name}'`] = { face: faceOfMove.get(name)!, amount: -amounts[i] as 1 | -1 };
    });
    fits.push({
      map: { colourCode: code.slice(), moves, cell, gear: gearFits[0].gear, gearSign: gearFits[0].sign, gearOffset: gearFits[0].offset },
      cellChoices,
      gearChoices: gearFits.length,
    });
  }

  if (!fits.length) {
    throw new Error(`The 3D engine and the solver tables are not the same puzzle under any relabelling:\n  ${failures.join('\n  ')}`);
  }
  if (fits.length > 1) {
    const which = fits.map((f) => Object.entries(f.map.moves).map(([k, v]) => `${k}=${v.face}${v.amount < 0 ? "'" : ''}`).join(' '));
    throw new Error(`More than one turning direction fits, so the dictionary is ambiguous:\n  ${which.join('\n  ')}`);
  }
  return fits[0];
}

/** The solver's 144-entry array for what the cube shows. The observation must be in the solver's frame (see canonicalise). */
export function toSolver144(obs: Observation, map: SolverMap): Uint8Array {
  const out = new Uint8Array(STATE_DIM);
  for (let i = 0; i < CELL_COUNT; i++) out[map.cell[i]] = map.colourCode[obs.cells[i]];
  for (let i = 0; i < GEAR_COUNT; i++) {
    out[GEAR_POS + map.gear[i]] = map.colourCode[obs.gearColour[i]];
    out[GEAR_ROT + map.gear[i]] = (((map.gearSign * obs.gearQuarter[i] + map.gearOffset[i]) % 4) + 4) % 4;
  }
  return out;
}

/** What the cube shows, from the solver's 144-entry array. */
export function fromSolver144(state: ArrayLike<number>, map: SolverMap): Observation {
  if (state.length !== STATE_DIM) throw new Error(`the solver array has ${STATE_DIM} entries, not ${state.length}`);
  const colour = (c: number) => {
    const i = map.colourCode.indexOf(c);
    if (i < 0) throw new Error(`${c} is not a colour code (0-5)`);
    return i;
  };
  const cells = new Uint8Array(CELL_COUNT);
  const gearColour = new Uint8Array(GEAR_COUNT);
  const gearQuarter = new Uint8Array(GEAR_COUNT);
  for (let i = 0; i < CELL_COUNT; i++) cells[i] = colour(state[map.cell[i]]);
  for (let i = 0; i < GEAR_COUNT; i++) {
    gearColour[i] = colour(state[GEAR_POS + map.gear[i]]);
    const r = state[GEAR_ROT + map.gear[i]] - map.gearOffset[i];
    gearQuarter[i] = (((map.gearSign * r) % 4) + 4) % 4;
  }
  return { cells, gearColour, gearQuarter };
}

/** A solver move word ("U L' F2 ...", a number repeats the turn) as engine moves. */
export function solverWordToMoves(word: string, map: SolverMap): Move[] {
  const out: Move[] = [];
  for (const token of word.split(/\s+/).filter(Boolean)) {
    const m = /^([A-Za-z]'?)(\d*)$/.exec(token);
    const base = m && map.moves[m[1]];
    if (!m || !base) throw new Error(`cannot read solver move "${token}"`);
    const n = m[2] ? parseInt(m[2], 10) : 1;
    out.push({ face: base.face, amount: base.amount * n });
  }
  return out;
}
