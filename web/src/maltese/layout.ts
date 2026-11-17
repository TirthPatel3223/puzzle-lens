import { mat3Vec } from '../vendor/puzzle/vec';
import { COLORS } from '../vendor/maltese/config';
import { CORNER_SLOTS, FACES, FACE_NORMAL, type Face, GEAR_SLOTS, GEAR_TILT, RADIAL_SLOTS, faceOf } from '../vendor/maltese/geometry';
import { CORNER_SLOT_OF_ORI, CUBE_GROUP, PIECES } from '../vendor/maltese/model';
import { MalteseState } from '../vendor/maltese/state';
import { STAR_MM } from '../vendor/maltese/stickers';
import {
  CENTRE_CELLS,
  CORNER_CELLS,
  EDGE_CELLS,
  FACE_RIGHT,
  FACE_UP,
  GEAR_SLOT_AT,
  PEGGED_CORNER,
  RADIAL_POINTS,
  colourName,
  colourOf,
  cornerCell,
  observe,
  radialCells,
} from './observation';
import { type ObservationMove, observationMove, observationMoveOf } from './solverMap';

/**
 * Everything the Python side needs to know about the Maltese Gear Cube, generated from the engine so
 * that Python has no geometry or move logic of its own: the pieces, what each of them shows in every
 * slot and turn, how the twelve quarter turns and the 24 whole-cube rotations rearrange what is shown,
 * and where things sit on a face. Written to shared/maltese/layout.json by tools/export-shared.gen.ts.
 */
export interface Layout {
  readonly version: number;
  readonly puzzle: 'maltese-gear-cube';
  /** Colour index -> name and display colour; the index is also the face (FACES order). */
  readonly colours: readonly { readonly name: string; readonly hex: string; readonly face: Face }[];
  readonly faces: readonly {
    readonly name: Face;
    readonly normal: readonly number[];
    readonly right: readonly number[];
    readonly up: readonly number[];
  }[];
  readonly grid: {
    readonly cornerCells: readonly number[];
    readonly centreCells: readonly number[];
    /** By side (top, right, bottom, left): this face's two cells of that edge slot. */
    readonly edgeCells: readonly (readonly number[])[];
  };
  readonly pieces: readonly { readonly id: number; readonly kind: string; readonly name: string; readonly colours: readonly number[] }[];
  /** Per corner slot: its three cells, and every (piece, orientation) that can sit there with the colours it shows on them. */
  readonly corners: readonly {
    readonly slot: number;
    readonly name: string;
    readonly cells: readonly number[];
    readonly options: readonly { readonly piece: number; readonly ori: number; readonly colours: readonly number[] }[];
  }[];
  /** Per centre/edge slot: its four cells and every (piece, quarter turn) with the colours shown. */
  readonly radial: readonly {
    readonly slot: number;
    readonly name: string;
    readonly kind: 'centre' | 'edge';
    readonly cells: readonly number[];
    readonly options: readonly { readonly piece: number; readonly k: number; readonly colours: readonly number[] }[];
  }[];
  /** Per gear place (4·face + position): the engine's gear slot. */
  readonly gears: readonly { readonly site: number; readonly slot: number; readonly name: string; readonly face: number; readonly position: number }[];
  /** The twelve quarter turns: new[i] = old[cells[i]]; gear quarter new[i] = old[gears[i]] + dk[i]. */
  readonly moves: Readonly<Record<string, ObservationMove>>;
  /** The 24 whole-cube rotations, by CUBE_GROUP index, in the same form. */
  readonly rotations: readonly ObservationMove[];
  readonly cubeGroup: {
    readonly mul: readonly (readonly number[])[];
    readonly inv: readonly number[];
    readonly cornerSlotOfOri: readonly number[];
  };
  /** The corner the solver never moves: its piece, home slot and orientation, and the colours its cells show at home. */
  readonly pegged: { readonly piece: number; readonly slot: number; readonly ori: number; readonly cells: readonly number[]; readonly colours: readonly number[] };
  readonly solved: { readonly cells: readonly number[]; readonly gearColour: readonly number[]; readonly gearQuarter: readonly number[] };
  /**
   * Where things are on a face, in face coordinates: x right, y up, the face is [-1, 1]² (1 = 30 mm).
   * `cellPoints[cell]` is a point on the sticker that fills the cell when the piece there is flush.
   */
  readonly geometry: {
    readonly unitMm: number;
    readonly gearHub: number;
    readonly gearPositions: readonly (readonly number[])[];
    readonly cellPoints: readonly (readonly number[])[];
  };
}

const CORNER_NORMALS = [
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 0],
] as const;

const MOVE_NAMES: readonly { name: string; face: Face; amount: 1 | -1 }[] = FACES.flatMap((face) => [
  { name: face, face, amount: 1 as const },
  { name: `${face}'`, face, amount: -1 as const },
]);

/** A point on the flush sticker of each of the 16 cells, in face coordinates. */
function cellPoints(): number[][] {
  const out: number[][] = new Array(16);
  const U = 1 / 30;
  // Corner stars: the elbow where the inner and outer arms meet, on the star's diagonal.
  const star = ((STAR_MM.hc + (STAR_MM.yl + STAR_MM.yu) / 2) / 2) * U;
  // Centre and edge stickers are 3.9 mm squares on a 5.2 mm column; each of their cells is a quarter of one.
  const q = 1.0 * U;
  // An edge's roof sticker lies near the face's border (see docs: the roof's mid-line is about 27.3 mm out).
  const roof = 27.3 * U;
  CORNER_CELLS.forEach((cell) => {
    const col = cell % 4;
    const row = Math.floor(cell / 4);
    out[cell] = [col === 3 ? star : -star, row === 0 ? star : -star];
  });
  CENTRE_CELLS.forEach((cell) => {
    const col = cell % 4;
    const row = Math.floor(cell / 4);
    out[cell] = [col === 2 ? q : -q, row === 1 ? q : -q];
  });
  EDGE_CELLS.forEach((pair, side) => {
    pair.forEach((cell, second) => {
      const along = second ? q : -q;
      out[cell] = side === 0 ? [along, roof] : side === 2 ? [along, -roof] : side === 1 ? [roof, -along] : [-roof, -along];
    });
  });
  return out;
}

export function buildLayout(): Layout {
  const solved = observe(MalteseState.solved());
  const moves: Record<string, ObservationMove> = {};
  for (const m of MOVE_NAMES) moves[m.name] = observationMoveOf({ face: m.face, amount: m.amount });
  const hub = Math.tan(GEAR_TILT);

  return {
    version: 1,
    puzzle: 'maltese-gear-cube',
    colours: FACES.map((f) => ({ name: colourName(colourOf(f)), hex: COLORS[f], face: f })),
    faces: FACES.map((f) => ({ name: f, normal: [...FACE_NORMAL[f]], right: [...FACE_RIGHT[f]], up: [...FACE_UP[f]] })),
    grid: { cornerCells: [...CORNER_CELLS], centreCells: [...CENTRE_CELLS], edgeCells: EDGE_CELLS.map((p) => [...p]) },
    pieces: PIECES.map((p) => ({ id: p.id, kind: p.kind, name: p.name, colours: p.colors.map(colourOf) })),
    corners: CORNER_SLOTS.map((cs, s) => {
      const cells = cs.faces.map((f) => cornerCell(s, f));
      const options: { piece: number; ori: number; colours: number[] }[] = [];
      for (const p of PIECES) {
        if (p.kind !== 'corner') continue;
        for (let o = 0; o < CUBE_GROUP.size; o++) {
          if (CORNER_SLOT_OF_ORI[o] !== s) continue;
          const g = CUBE_GROUP.elements[o];
          const colours = cs.faces.map((f) => {
            const i = CORNER_NORMALS.findIndex((n) => faceOf(mat3Vec(g, n)) === f);
            return colourOf(p.colors[i]);
          });
          options.push({ piece: p.id, ori: o, colours });
        }
      }
      return { slot: s, name: cs.name, cells, options };
    }),
    radial: RADIAL_SLOTS.map((rs, s) => {
      const cells = [...radialCells(s)];
      const options: { piece: number; k: number; colours: number[] }[] = [];
      for (const p of PIECES) {
        if (p.kind !== 'centre' && p.kind !== 'edge') continue;
        for (let k = 0; k < (p.kind === 'centre' ? 1 : 4); k++) {
          // Put the piece there alone and read what the slot's four cells show.
          const slot = Int8Array.from(PIECES, (q) => q.home);
          const ori = Int8Array.from(PIECES, (q) => (q.kind === 'corner' ? q.homeOri : 0));
          const other = PIECES.find((q) => (q.kind === 'centre' || q.kind === 'edge') && q.home === s)!;
          slot[other.id] = p.home;
          slot[p.id] = s;
          ori[p.id] = k;
          const shown = observe(MalteseState.fromPoses(slot, ori)).cells;
          options.push({ piece: p.id, k, colours: cells.map((c) => shown[c]) });
        }
      }
      return { slot: s, name: rs.name, kind: rs.kind, cells, options };
    }),
    gears: GEAR_SLOT_AT.map((slot, site) => ({
      site,
      slot,
      name: GEAR_SLOTS[slot].name,
      face: Math.floor(site / 4),
      position: site % 4,
    })),
    moves,
    rotations: CUBE_GROUP.elements.map((_, rot) => observationMove(MalteseState.solved().rotated(rot))),
    cubeGroup: {
      mul: CUBE_GROUP.elements.map((_, a) => CUBE_GROUP.elements.map((__, b) => CUBE_GROUP.mul(a, b))),
      inv: CUBE_GROUP.elements.map((_, a) => CUBE_GROUP.inv(a)),
      cornerSlotOfOri: [...CORNER_SLOT_OF_ORI],
    },
    pegged: (() => {
      const s = PEGGED_CORNER.home;
      const cells = CORNER_SLOTS[s].faces.map((f) => cornerCell(s, f));
      return { piece: PEGGED_CORNER.id, slot: s, ori: PEGGED_CORNER.homeOri, cells, colours: cells.map((c) => solved.cells[c]) };
    })(),
    solved: { cells: Array.from(solved.cells), gearColour: Array.from(solved.gearColour), gearQuarter: Array.from(solved.gearQuarter) },
    geometry: {
      unitMm: 30,
      gearHub: hub,
      gearPositions: [
        [0, hub],
        [hub, 0],
        [0, -hub],
        [-hub, 0],
      ],
      cellPoints: cellPoints(),
    },
  };
}

/** Number of sticker points on a centre or edge piece (kept here so the layout and its readers agree). */
export const RADIAL_POINT_COUNT = RADIAL_POINTS.length;
