import { COLORS } from '../../vendor/maltese/config';
import { FACES, FACE_NORMAL, type Face } from '../../vendor/maltese/geometry';
import { CUBE_GROUP } from '../../vendor/maltese/model';
import { type Move, normalizeAmount } from '../../vendor/maltese/notation';
import { MalteseState } from '../../vendor/maltese/state';
import type { ViewPreset } from '../../vendor/render/scene';
import { CELL_COUNT, FACE_RIGHT, FACE_UP, GEAR_COUNT, type Observation, fromObservation, observe } from '../../maltese/observation';
import { Palette } from '../palette';

/**
 * Shared vocabulary of the scanner app: colours, faces, frames and moves.
 *
 * Frames. A scan is described in the frame the cube was photographed in ("held"): photo 1 is its F
 * face with that photo's top edge up. The solver works in its own frame: the white-green-red corner
 * home, held blue up, yellow in front, purple on the left. The 3D engine names faces by their solved
 * colours (U white, D yellow, F green, B blue, L red, R purple), so in the solver's frame the engine's
 * B face is on top, D in front and R on the left: the solver's moves U, L, F are the engine's B, R, D.
 */

export const COLOUR_NAMES = ['white', 'yellow', 'green', 'blue', 'red', 'purple'] as const;
/** Display colour of each colour index (FACES order: the face that colour belongs to when solved). */
export const COLOUR_HEX: readonly string[] = FACES.map((f) => COLORS[f]);
/** Text colour that reads on top of each colour. */
export const COLOUR_INK: readonly string[] = ['#1a202c', '#1a202c', '#ffffff', '#ffffff', '#ffffff', '#ffffff'];

/**
 * The colours shown: the scanned cube's colour layout (see ../palette.ts). Observations and states stay
 * in the model's colours; everything drawn or said goes through this.
 */
export const PALETTE = new Palette(COLOUR_HEX, COLOUR_NAMES, COLOUR_INK);

/** A JSON-friendly observation (as the server sends it). */
export interface PlainObservation {
  cells: number[];
  gearColour: number[];
  gearQuarter: number[];
}

export function toObservation(p: PlainObservation): Observation {
  return { cells: Uint8Array.from(p.cells), gearColour: Uint8Array.from(p.gearColour), gearQuarter: Uint8Array.from(p.gearQuarter) };
}

export function toPlain(o: Observation): PlainObservation {
  return { cells: Array.from(o.cells), gearColour: Array.from(o.gearColour), gearQuarter: Array.from(o.gearQuarter) };
}

export function clonePlain(p: PlainObservation): PlainObservation {
  return { cells: p.cells.slice(), gearColour: p.gearColour.slice(), gearQuarter: p.gearQuarter.slice() };
}

export function samePlain(a: PlainObservation, b: PlainObservation): boolean {
  return a.cells.every((c, i) => c === b.cells[i]) && a.gearColour.every((c, i) => c === b.gearColour[i]) && a.gearQuarter.every((q, i) => q === b.gearQuarter[i]);
}

/** Number of places (cells + gear colours + gear quarters) where two observations differ. */
export function countDifferences(a: PlainObservation, b: PlainObservation): number {
  let n = 0;
  for (let i = 0; i < CELL_COUNT; i++) if (a.cells[i] !== b.cells[i]) n++;
  for (let i = 0; i < GEAR_COUNT; i++) {
    if (a.gearColour[i] !== b.gearColour[i]) n++;
    if (a.gearQuarter[i] !== b.gearQuarter[i]) n++;
  }
  return n;
}

/** The 3D state an observation shows (null when its stickers are not a set of real pieces). */
export function stateOf(p: PlainObservation): MalteseState | null {
  return fromObservation(toObservation(p)).state;
}

/** The engine state from the decoder's poses (they are in the solver's frame). */
export function stateFromPoses(slot: readonly number[], ori: readonly number[]): MalteseState {
  return MalteseState.fromPoses(slot, ori);
}

/**
 * A known cube (in the solver's frame) as it was held for a scan: of the 24 ways to hold it, the one
 * that agrees best with what the scan saw. Used to compare a practice scan with its scramble.
 */
export function heldTruth(truth: MalteseState, seen: PlainObservation): PlainObservation {
  let best: PlainObservation | null = null;
  let bestScore = -1;
  for (let r = 0; r < CUBE_GROUP.size; r++) {
    const o = observe(truth.rotated(r));
    let score = 0;
    for (let i = 0; i < CELL_COUNT; i++) if (o.cells[i] === seen.cells[i]) score++;
    for (let i = 0; i < GEAR_COUNT; i++) if (o.gearColour[i] === seen.gearColour[i]) score++;
    if (score > bestScore) {
      bestScore = score;
      best = toPlain(o);
    }
  }
  return best!;
}

// ------------------------------------------------------------------------------------- views

/** Straight at one face of the frame the cube is shown in, that face's own "up" at the top (slightly tilted). */
export function faceView(f: Face, tilt = true): ViewPreset {
  return { rows: [[...FACE_RIGHT[f]], [...FACE_UP[f]], [...FACE_NORMAL[f]]], azimuth: tilt ? 16 : 0, elevation: tilt ? 18 : 0 };
}

/** The solver's way of holding the cube (blue up, yellow in front, purple on the left), seen from front-left-above. */
export const SOLVER_ROWS: ViewPreset['rows'] = [
  [-1, 0, 0],
  [0, 0, -1],
  [0, -1, 0],
];
export const SOLVER_VIEW: ViewPreset = { rows: SOLVER_ROWS, azimuth: -34, elevation: 26 };

/** The two ways to turn the model while following a solution (held the solver's way: blue on top). */
export const SOLVER_VIEWS: Readonly<Record<'corner' | 'face', ViewPreset>> = {
  // The blue-yellow-red corner (top, front, right) towards the viewer.
  corner: { rows: SOLVER_ROWS, azimuth: 45, elevation: 28 },
  // Yellow straight on.
  face: { rows: SOLVER_ROWS, azimuth: 0, elevation: 0 },
};

const IDENTITY_ROWS: ViewPreset['rows'] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
/**
 * The 3D simulator site's three camera presets, for a cube in the frame it is shown in (for a scan:
 * photo 1 is F with its top edge up). The same values as the site's MALTESE_VIEWS.
 */
export const HELD_VIEWS: Readonly<Record<'corner' | 'face' | 'edge', ViewPreset>> = {
  // U on top, F front-left, R front-right.
  corner: { rows: IDENTITY_ROWS, azimuth: 45, elevation: 28 },
  // F straight on.
  face: { rows: IDENTITY_ROWS, azimuth: 0, elevation: 0 },
  // The UF edge towards the viewer.
  edge: { rows: IDENTITY_ROWS, azimuth: 0, elevation: 45 },
};

/** An overview of a held cube: F front-left, R front-right, U on top. */
export const HELD_OVERVIEW: ViewPreset = HELD_VIEWS.corner;

/** Where each engine face is while following a solution (the engine names faces by solved colour). */
export const HOLD_POSITION: Readonly<Record<Face, string>> = { B: 'top', D: 'front', R: 'left', L: 'right', U: 'back', F: 'bottom' };

// ------------------------------------------------------------------------------------- solver moves

/** The solver's move letters and the engine faces they turn. */
export const SOLVER_FACE: Readonly<Record<'U' | 'L' | 'F', Face>> = { U: 'B', L: 'R', F: 'D' };
/** The solver's faces in words, in the colours of the cube in hand. */
export const SOLVER_FACE_WORDS: Readonly<Record<'U' | 'L' | 'F', string>> = {
  get U() { return PALETTE.say('top (blue) face'); },
  get L() { return PALETTE.say('left (purple) face'); },
  get F() { return PALETTE.say('front (yellow) face'); },
};

export interface SolverMove {
  /** As the solver writes it, e.g. "L'3". */
  readonly token: string;
  readonly letter: 'U' | 'L' | 'F';
  /** Signed quarter turns (clockwise positive, seen looking at that face). */
  readonly quarters: number;
  /** The same move on the 3D model. */
  readonly engine: Move;
}

/** Read a solver word ("U L'3 F2 ..."): every token becomes one move. */
export function parseSolverWord(word: string): SolverMove[] {
  return word
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => {
      const m = /^([ULF])(\d*)('?)(\d*)$/.exec(token);
      if (!m || (m[2] && m[4])) throw new Error(`cannot read the solver move "${token}"`);
      const letter = m[1] as 'U' | 'L' | 'F';
      const n = parseInt(m[2] || m[4] || '1', 10);
      const quarters = m[3] ? -n : n;
      return { token, letter, quarters, engine: { face: SOLVER_FACE[letter], amount: normalizeAmount(quarters) } };
    });
}

/** A solver move in words: "Left (purple) face: 3 quarter turns counterclockwise". */
export function describeSolverMove(m: SolverMove): string {
  const n = Math.abs(m.quarters);
  const turns = n === 1 ? 'a quarter turn' : `${n} quarter turns`;
  const way = m.quarters > 0 ? 'clockwise' : 'counterclockwise';
  const face = SOLVER_FACE_WORDS[m.letter];
  return `${face[0].toUpperCase()}${face.slice(1)}: ${turns} ${way}`;
}

/**
 * A random scramble in the solver's moves (so the white-green-red corner stays put and the cube stays
 * in the solver's frame): `length` moves, never the same face twice in a row.
 */
export function solverScramble(rng: () => number, length = 30): SolverMove[] {
  const letters = ['U', 'L', 'F'] as const;
  const amounts = [1, -1, 1, -1, 2, -2];
  const out: SolverMove[] = [];
  let prev = '';
  while (out.length < length) {
    const options = letters.filter((l) => l !== prev);
    const letter = options[Math.floor(rng() * options.length)];
    const q = amounts[Math.floor(rng() * amounts.length)];
    const token = `${letter}${Math.abs(q) === 1 ? '' : Math.abs(q)}${q < 0 ? "'" : ''}`;
    out.push(parseSolverWord(token)[0]);
    prev = letter;
  }
  return out;
}

/** Group solver tokens four to a line, as in the solution files. */
export function linesOfFour(tokens: readonly string[]): string[] {
  const lines: string[] = [];
  for (let i = 0; i < tokens.length; i += 4) lines.push(tokens.slice(i, i + 4).join(' '));
  return lines;
}

export const FACE_LIST: readonly Face[] = FACES;
