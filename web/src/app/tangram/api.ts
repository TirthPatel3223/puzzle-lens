import { ApiError, call, json, puzzleApi } from '../api';
import type { ColoursInfo } from '../palette';

/** The Tangram Cube Extreme's endpoints (api/tangram-cube-extreme/…, scanner/puzzles/tangram/api.py). */

export { ApiError };

export const ID = 'tangram-cube-extreme';

export interface Problem {
  readonly code: string;
  readonly message: string;
  /** Positions (0-119) involved. */
  readonly positions: number[];
}

export interface Correction {
  readonly position: number;
  readonly read: number;
  readonly decoded: number;
  readonly probability: number;
}

export interface TangramDecode {
  /** Orientation of every piece (core frame), or null when no arrangement fits. */
  readonly orient: number[] | null;
  /** What that state shows: per position a colour 0-5 or 6 (no sticker). */
  readonly observation: number[] | null;
  readonly legal: boolean;
  readonly problems: Problem[];
  readonly corrections: Correction[];
  /** Probability the reader gave the decoded value at each position (1 for hand-edited input). */
  readonly confidence: number[];
  readonly loglik: number;
  /** Changes the decoder made to reach a legal state, in words. */
  readonly repairs: string[];
}

export interface ScanPosition {
  readonly position: number;
  /** Where it appears in the flattened photo, as fractions. */
  readonly x: number;
  readonly y: number;
  /** In the face plane (else turned out at 48°). */
  readonly flat: boolean;
  /** What the reader saw there (0-5 colour, 6 no sticker, 7 hidden) and how sure it was. */
  readonly read: number;
  readonly prob: number;
}

export interface ScanPhoto {
  readonly shot: number;
  readonly name: string;
  /** The face this photo shows ('+x' …) and its index (FACE_IDS order, also its solved colour). */
  readonly face: string;
  readonly faceIndex: number;
  /** The turning corner on this face: T (orange-yellow-green on the model) or B (red-white-blue). */
  readonly end: 'T' | 'B';
  readonly colour: string;
  readonly quad: number[][];
  readonly locateConfidence: number;
  /** Quarter turns the scanner turned the photo by (it found the turning corner elsewhere). */
  readonly turned: number;
  /** The face flattened (JPEG data URL), its turning corner at the bottom-right. */
  readonly flat: string;
  readonly positions: ScanPosition[];
}

export interface TangramScan {
  readonly id: string;
  readonly seconds: number;
  readonly decode: TangramDecode;
  /** Log-likelihood margin over the next most likely answer (nats); null when there was none. */
  readonly margin: number | null;
  readonly doubtful: boolean;
  /** What each position was read as, all photos combined, and how sure. */
  readonly read: number[];
  readonly readProb: number[];
  /** The cube's colour layout (everything above is in the model's colours), and the next closest ones. */
  readonly colours: ColoursInfo;
  readonly alternatives: ColoursInfo[];
  readonly photos: ScanPhoto[];
}

export interface TangramSolution {
  readonly word: string;
  readonly moves: { axis: 'U' | 'D' | 'F' | 'R' | 'L'; amount: number; token: string }[];
  readonly turns: number;
  readonly steps: number;
  readonly seconds: number;
  readonly verified: boolean;
  readonly source: string;
}

const base = puzzleApi(ID);
const p = (path: string) => `${ID}/${path}`;

export const api = {
  ...base,
  decode: (observation: number[]) => call<TangramDecode>(p('decode'), json({ observation })),
  fromMoves: (moves: string) => call<{ orient: number[]; observation: number[]; moves: number }>(p('from-moves'), json({ moves })),
  solve: (orient: number[]) => call<TangramSolution>(p('solve'), json({ orient })),
  scan: (photos: { blob: Blob; name: string }[], quads?: (number[][] | null)[], colours?: readonly number[]) => base.scan<TangramScan>(photos, quads, colours),
  confirm: (id: string, observation: number[], source: 'checked' | 'known-scramble', colours?: readonly number[]) =>
    base.confirm(id, { observation, source, colours: colours ?? null }),
};
