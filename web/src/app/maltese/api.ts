import { ApiError, type Checked, type Located, call, json, puzzleApi } from '../api';
import type { PlainObservation } from './model';
import type { ColoursInfo } from '../palette';

/** The Maltese Gear Cube's endpoints (api/maltese-gear-cube/…, scanner/puzzles/maltese/api.py). */

export { ApiError };
export type { Checked, Located };

export const ID = 'maltese-gear-cube';

export interface Problem {
  readonly code: string;
  readonly message: string;
  /** Cells (0-95) and gear places (0-23) involved, in the frame the observation was given in. */
  readonly cells: number[];
  readonly gears: number[];
}

export interface Correction {
  readonly kind: 'cell' | 'gearColour' | 'gearQuarter';
  readonly index: number;
  readonly read: number;
  readonly decoded: number;
  readonly probability: number;
}

export interface Decode {
  /** The arrangement in the solver's frame (null when no arrangement of real pieces fits). */
  readonly observation: PlainObservation | null;
  /** Whole-cube rotation (engine CUBE_GROUP index) from the frame given to the solver's frame. */
  readonly rotation: number;
  /** Engine poses of the 50 pieces in the solver's frame. */
  readonly slot: number[] | null;
  readonly ori: number[] | null;
  readonly state144: number[] | null;
  readonly legal: boolean;
  readonly problems: Problem[];
  readonly corrections: Correction[];
  /** Probability the reader gave each decoded value, in the frame given (1 for hand-edited input). */
  readonly cellConfidence: number[];
  readonly gearConfidence: number[];
}

export interface ScanFace {
  readonly shot: number;
  readonly name: string;
  /** The face this photo shows, in the frame the cube was photographed in (photo 1 is F). */
  readonly face: 'U' | 'D' | 'F' | 'B' | 'L' | 'R';
  /** Its colour when solved (the model's colour name), and where it is while following the solution. */
  readonly solvedColour: string;
  readonly holdPosition: string;
  readonly quad: number[][];
  readonly locateConfidence: number;
  readonly turn: number;
  /** The face flattened (JPEG data URL), turned the way its cells are laid out. */
  readonly flat: string;
}

export interface Scan {
  readonly id: string;
  readonly seconds: number;
  /** The decoded cube in the frame photographed. */
  readonly observation: PlainObservation | null;
  readonly decode: Decode;
  readonly turns: number[];
  /** The cube's colour layout (everything here is in the model's colours), and the next closest ones. */
  readonly colours: ColoursInfo;
  readonly alternatives: ColoursInfo[];
  readonly arrangement: { readonly margin: number | null; readonly doubtful: boolean };
  readonly reading: {
    readonly cells: number[];
    readonly cellProb: number[];
    readonly gearColour: number[];
    readonly gearColourProb: number[];
    readonly gearQuarter: number[];
    readonly gearQuarterProb: number[];
  };
  readonly faces: ScanFace[];
}

export interface Solution {
  readonly word: string;
  readonly moves: { name: string; amount: number }[];
  readonly engineMoves: { face: 'U' | 'D' | 'F' | 'B' | 'L' | 'R'; amount: number }[];
  readonly quarterTurns: number;
  readonly faceTurns: number;
  readonly perRung: Record<string, number>;
  readonly seconds: number;
  readonly verified: boolean;
}

const base = puzzleApi(ID);
const p = (path: string) => `${ID}/${path}`;

export const api = {
  ...base,
  decode: (obs: PlainObservation) => call<Decode>(p('decode'), json(obs)),
  fromArray: (state144: number[]) => call<Decode>(p('from-array'), json({ state144 })),
  solve: (state144: number[]) => call<Solution>(p('solve'), json({ state144 })),
  scan: (photos: { blob: Blob; name: string }[], quads?: (number[][] | null)[], colours?: readonly number[]) => base.scan<Scan>(photos, quads, colours),
  confirm: (id: string, obs: PlainObservation, source: 'checked' | 'known-scramble', colours?: readonly number[]) =>
    base.confirm(id, { ...obs, source, colours: colours ?? null }) as Promise<{ saved: string; scans: number; readErrors: Record<string, number> | null; decodedRight: boolean | null }>,
};
