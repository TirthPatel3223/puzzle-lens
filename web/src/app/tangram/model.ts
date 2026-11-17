import { type Move, formatMove, generateScramble, invertSequence, parseAlgorithm } from '../../vendor/tangram/moves';
import { PuzzleState } from '../../vendor/tangram/state';
import { type Vec3, dot, sub } from '../../vendor/puzzle/vec';
import type { ViewPreset } from '../../vendor/render/scene';
import { TANGRAM_VIEWS } from '../../vendor/render/scene';
import { COLOUR_HEX, COLOUR_NAMES, EMPTY, FACE_MIN_COS, FACE_VIEWS, MODEL, TABLES, inFaceFrame, observe } from '../../tangram/model';
import { Palette } from '../palette';

/**
 * The Tangram Cube Extreme as the app talks about it: colours, faces, the simulator's notation in
 * words, the pieces that mark where each move is on a scrambled puzzle, and each face's drawing.
 *
 * Frame. Everything is in the core's frame (the hidden middle piece never moves), which is also the
 * solver's and the 3D simulator's. A scan needs no other frame: the scanner works out which face each
 * photo shows.
 */

export { COLOUR_HEX, COLOUR_NAMES, EMPTY };
/** Text colour that reads on top of each colour (orange red yellow white green blue). */
export const COLOUR_INK: readonly string[] = ['#1a202c', '#ffffff', '#1a202c', '#1a202c', '#ffffff', '#ffffff'];

/**
 * The colours shown: the scanned cube's colour layout (see ../palette.ts). Observations and states stay
 * in the model's colours; everything drawn or said goes through this.
 */
export const PALETTE = new Palette(COLOUR_HEX, COLOUR_NAMES, COLOUR_INK);

/** A model colour as the cube in hand shows it, in words. */
export const colourWord = (c: number): string => (c === EMPTY ? 'no sticker' : c >= 0 && c < 6 ? PALETTE.nameOf(c) : '?');

/** Face i (FACE_IDS order) by its solved colour as the cube in hand shows it: "orange face". */
export const faceName = (i: number): string => `${PALETTE.nameOf(i)} face`;

/**
 * Where each move is on a puzzle in any state. The turning corner pieces only twist in place and the
 * middle-layer edge pieces only flip in place, so they always mark their part of the puzzle.
 */
const LANDMARK_WORDS = {
  U: 'the orange-yellow-green corner (the T corner)',
  D: 'the red-white-blue corner (the B corner)',
  F: 'the edge with the orange-white middle piece',
  R: 'the edge with the yellow-blue middle piece',
  L: 'the edge with the red-green middle piece',
} as const;

export type Axis = keyof typeof LANDMARK_WORDS;

/** The landmarks in the colours of the cube in hand. */
export const LANDMARKS: Readonly<Record<Axis, string>> = {
  get U() { return PALETTE.say(LANDMARK_WORDS.U); },
  get D() { return PALETTE.say(LANDMARK_WORDS.D); },
  get F() { return PALETTE.say(LANDMARK_WORDS.F); },
  get R() { return PALETTE.say(LANDMARK_WORDS.R); },
  get L() { return PALETTE.say(LANDMARK_WORDS.L); },
};

/** Read moves in the simulator's notation ("U2' F D R"). */
export function parseMoves(text: string): Move[] {
  const r = parseAlgorithm(text, MODEL.variant);
  if (!r.ok) throw new Error(r.error);
  return r.moves;
}

export const moveToken = (m: Move): string => formatMove(m);

/** A move in words: "Top corner (orange-yellow-green): 2 steps (120°) clockwise". */
export function describeMove(m: Move): string {
  if (m.axis === 'F' || m.axis === 'R' || m.axis === 'L') {
    const which = { F: 'orange-white', R: 'yellow-blue', L: 'red-green' }[m.axis];
    return PALETTE.say(`Flip the ${which} edge half a turn (180°)`);
  }
  const n = Math.abs(m.amount);
  const corner = m.axis === 'U' ? 'Top corner (orange-yellow-green)' : 'Bottom corner (red-white-blue)';
  const steps = n === 1 ? 'one step (60°)' : `${n} steps (${60 * n}°)`;
  return PALETTE.say(`${corner}: ${steps} ${m.amount > 0 ? 'clockwise' : 'counterclockwise'}, seen from that corner`);
}

/** Group tokens six to a line. */
export function linesOf(tokens: readonly string[], per = 6): string[] {
  const lines: string[] = [];
  for (let i = 0; i < tokens.length; i += per) lines.push(tokens.slice(i, i + per).join(' '));
  return lines;
}

/** A practice scramble (the simulator's: each round turns the corners, then flips an edge). */
export function practiceScramble(rng: () => number, rounds = 12): { moves: Move[]; truth: PuzzleState } {
  const moves = generateScramble(MODEL.variant, rng, rounds);
  const truth = PuzzleState.solved(MODEL);
  truth.applyAll(moves);
  return { moves, truth };
}

export const undoMoves = (moves: readonly Move[]): Move[] => invertSequence(moves);

export function stateAfter(moves: readonly Move[], from: PuzzleState = PuzzleState.solved(MODEL)): PuzzleState {
  const s = from.clone();
  s.applyAll(moves);
  return s;
}

export const observeState = (s: PuzzleState): number[] => Array.from(observe(s));

/** Places where two observations differ. */
export function countDifferences(a: readonly number[], b: readonly number[]): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
  return n;
}

/**
 * Positions that share a centre with another one (turned 60° from it): only one of them can hold a
 * sticker. Setting a colour on one clears the others.
 */
export const LOOKALIKES: readonly (readonly number[])[] = TABLES.positions.map((p) =>
  TABLES.positions.filter((q) => q.index !== p.index && Math.hypot(...sub(p.centre, q.centre)) < 0.05).map((q) => q.index),
);

// ------------------------------------------------------------------------------------------ faces

export interface FacePosition {
  readonly position: number;
  /** Centre in the face's frame: x right, y up (the face is [-1, 1]²), z out of the face. */
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly flat: boolean;
  /** The sticker's outline seen straight at the face (x right, y up). */
  readonly outline: readonly (readonly [number, number])[];
}

/** Per face (FACE_IDS order): the 29 positions in its photo, as a straight-on drawing shows them. */
export const FACE_POSITIONS: readonly (readonly FacePosition[])[] = FACE_VIEWS.map((f) =>
  TABLES.positions
    .filter((p) => dot(p.normal, f.normal) >= FACE_MIN_COS)
    .map((p) => {
      const [x, y, z] = inFaceFrame(f, p.centre);
      return {
        position: p.index,
        x,
        y,
        z,
        flat: dot(p.normal, f.normal) > 0.999,
        outline: p.polygon.map((v) => {
          const q = inFaceFrame(f, v as Vec3);
          return [q[0], q[1]] as const;
        }),
      };
    })
    // Drawn back to front: what stands further out covers what is behind it.
    .sort((a, b) => a.z - b.z),
);

/** The faces' turning corner: T (orange-yellow-green) or B (red-white-blue). */
export const FACE_END: readonly ('T' | 'B')[] = FACE_VIEWS.map((f) => f.end);

// ------------------------------------------------------------------------------------------ views

/** The simulator's three views, and one straight at each face (turning corner at the bottom-right, slightly tilted). */
export const MODEL_VIEWS: Readonly<Record<string, ViewPreset>> = {
  corner: TANGRAM_VIEWS.corner,
  face: TANGRAM_VIEWS.face,
  edge: TANGRAM_VIEWS.edge,
  ...Object.fromEntries(
    FACE_VIEWS.map((f, i) => [`face-${i}`, { rows: [[...f.right], [...f.up], [...f.normal]], azimuth: -14, elevation: 16 } as ViewPreset]),
  ),
};
