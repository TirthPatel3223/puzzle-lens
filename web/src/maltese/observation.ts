import { type Vec3, dot, mat3Vec } from '../vendor/puzzle/vec';
import { CORNER_SLOTS, FACES, FACE_NORMAL, type Face, GEAR_SLOTS, RADIAL_SLOTS, RZ, faceOf } from '../vendor/maltese/geometry';
import { CORNER_SLOT_OF_ORI, CUBE_GROUP, PIECES } from '../vendor/maltese/model';
import { MalteseState } from '../vendor/maltese/state';

/**
 * What a Maltese Gear Cube shows, face by face: the thing a camera can see and the thing the solver's
 * 144-entry array stores. It is computed from the exact engine state, so it is the label of every
 * training image and the meeting point of vision, 3D model and solver.
 *
 * Each face is a 4×4 grid of colour cells plus 4 gears:
 *
 *        col 0   1   2   3
 *   row 0  [ 0][ 1][ 2][ 3]     0, 3, 12, 15   the four corner stars
 *   row 1  [ 4][ 5][ 6][ 7]     5, 6, 9, 10    the centre slot (one piece, four cells)
 *   row 2  [ 8][ 9][10][11]     1 2 / 4 8 / 7 11 / 13 14   this face's half of an edge slot
 *   row 3  [12][13][14][15]
 *
 * A centre piece shows one colour on all four of its cells. An edge piece shows its two colours two
 * cells each: in an edge slot lying along the ridge that is one colour per face; turned 90° it is both
 * colours side by side on both faces; standing in a centre slot it is [A A / B B] in one of four turns.
 *
 * Rows run down and columns run right as the face is seen from outside, with this side up:
 *   F, B, L, R: U is up.   U: B is up.   D: F is up.     (the usual unfolded-cube picture)
 *
 * Colours are indices into FACES: 0 U white, 1 D yellow, 2 F green, 3 B blue, 4 L red, 5 R purple.
 * Faces are stored in FACES order (U D F B L R): cell `16·face + 4·row + col`, gear `4·face + position`
 * with positions 0 top, 1 right, 2 bottom, 3 left. A gear's quarter is its pose k: 0 = flush with its
 * long teeth towards the cube edge, each step a quarter turn counterclockwise seen from outside.
 */
export interface Observation {
  /** 96 colour indices. */
  readonly cells: Uint8Array;
  /** 24 colour indices. */
  readonly gearColour: Uint8Array;
  /** 24 quarter turns, 0-3. */
  readonly gearQuarter: Uint8Array;
}

export const CELL_COUNT = 96;
export const GEAR_COUNT = 24;
/** Marks a cell nobody has filled in yet. */
export const UNKNOWN = 255;

/** "Right" and "up" on each face, seen from outside (right × up = the outward normal). */
export const FACE_RIGHT: Readonly<Record<Face, Vec3>> = {
  F: [1, 0, 0],
  B: [-1, 0, 0],
  R: [0, 0, -1],
  L: [0, 0, 1],
  U: [1, 0, 0],
  D: [1, 0, 0],
};
export const FACE_UP: Readonly<Record<Face, Vec3>> = {
  F: [0, 1, 0],
  B: [0, 1, 0],
  R: [0, 1, 0],
  L: [0, 1, 0],
  U: [0, 0, -1],
  D: [0, 0, 1],
};

export const colourOf = (f: Face): number => FACES.indexOf(f);
const faceIndex = colourOf;

/** Cells of the grid by role. */
export const CORNER_CELLS = [0, 3, 12, 15] as const;
export const CENTRE_CELLS = [5, 6, 9, 10] as const;
/** The two cells of this face's half of each edge slot, by side: top, right, bottom, left. */
export const EDGE_CELLS = [
  [1, 2],
  [7, 11],
  [13, 14],
  [4, 8],
] as const;

/** Stickers of the corner prototype (at UFR): its U, F and R faces, in the order of `MaltesePiece.colors`. */
const CORNER_NORMALS: readonly Vec3[] = [
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 0],
];

/** The four sticker points of a centre or edge piece, in its own frame: (sx, sy), index sx>0 ? 1 : 0 plus sy>0 ? 2 : 0. */
export const RADIAL_POINTS: readonly (readonly [number, number])[] = [
  [-1, -1],
  [1, -1],
  [-1, 1],
  [1, 1],
];
const quadrant = (x: number, y: number) => (x > 0 ? 1 : 0) + (y > 0 ? 2 : 0);

/** Side of face `f` that neighbour `g` lies on: 0 top, 1 right, 2 bottom, 3 left. */
function sideOf(f: Face, g: Face): number {
  const n = FACE_NORMAL[g];
  const u = dot(n, FACE_UP[f]);
  const r = dot(n, FACE_RIGHT[f]);
  if (u > 0.5) return 0;
  if (r > 0.5) return 1;
  if (u < -0.5) return 2;
  if (r < -0.5) return 3;
  throw new Error(`${g} is not a neighbour of ${f}`);
}

/** Per corner slot: the cell its sticker on each of its three faces fills. */
const CORNER_CELL: readonly ReadonlyMap<Face, number>[] = CORNER_SLOTS.map((slot) => {
  const m = new Map<Face, number>();
  for (const f of slot.faces) {
    const col = dot(FACE_RIGHT[f], slot.dir) > 0 ? 3 : 0;
    const row = dot(FACE_UP[f], slot.dir) > 0 ? 0 : 3;
    m.set(f, 16 * faceIndex(f) + 4 * row + col);
  }
  return m;
});

/**
 * Per radial slot: the cell under each quadrant of the slot frame (offsets (±1, ±1) from the slot's
 * axis, indexed like RADIAL_POINTS). In an edge slot the half leaning towards a face belongs to that face.
 */
const RADIAL_CELL: readonly (readonly number[])[] = RADIAL_SLOTS.map((slot) =>
  RADIAL_POINTS.map(([ox, oy]) => {
    const w = mat3Vec(slot.frame, [ox, oy, 0]);
    if (slot.kind === 'centre') {
      const f = slot.faces[0];
      const col = dot(FACE_RIGHT[f], w) > 0 ? 2 : 1;
      const row = dot(FACE_UP[f], w) > 0 ? 1 : 2;
      return 16 * faceIndex(f) + 4 * row + col;
    }
    const [a, b] = slot.faces;
    const f = dot(w, FACE_NORMAL[a]) > 0 ? a : b;
    const g = f === a ? b : a;
    const side = sideOf(f, g);
    // Along the edge: left/right for the top and bottom sides, up/down for the left and right sides.
    const second = side === 0 || side === 2 ? dot(FACE_RIGHT[f], w) > 0 : dot(FACE_UP[f], w) < 0;
    return 16 * faceIndex(f) + EDGE_CELLS[side][second ? 1 : 0];
  }),
);

/** Per gear slot: its place in the gear arrays (4·face + position). */
export const GEAR_SITE: readonly number[] = GEAR_SLOTS.map((g) => 4 * faceIndex(g.face) + sideOf(g.face, g.toward));
/** The gear slot at each place of the gear arrays. */
export const GEAR_SLOT_AT: readonly number[] = (() => {
  const out = new Array<number>(GEAR_COUNT).fill(-1);
  GEAR_SITE.forEach((site, slot) => {
    out[site] = slot;
  });
  return out;
})();

/** The cells a radial slot fills, by slot-frame quadrant. */
export const radialCells = (slot: number): readonly number[] => RADIAL_CELL[slot];
/** The cell a corner slot fills on face `f`. */
export const cornerCell = (slot: number, f: Face): number => CORNER_CELL[slot].get(f)!;

/** Where sticker point `i` of a centre or edge piece turned k quarter turns lands: a slot-frame quadrant. */
function pointQuadrant(i: number, k: number): number {
  const [sx, sy] = RADIAL_POINTS[i];
  const v = mat3Vec(RZ[k], [sx, sy, 0]);
  return quadrant(v[0], v[1]);
}
/** POINT_QUADRANT[k][i]. */
const POINT_QUADRANT: readonly (readonly number[])[] = [0, 1, 2, 3].map((k) => RADIAL_POINTS.map((_, i) => pointQuadrant(i, k)));

/** Colour a centre or edge piece shows at its sticker point i: an edge's +y half is its first colour. */
function radialPointColour(pieceId: number, i: number): number {
  const p = PIECES[pieceId];
  return colourOf(p.kind === 'centre' ? p.colors[0] : RADIAL_POINTS[i][1] > 0 ? p.colors[0] : p.colors[1]);
}

/**
 * Walk every sticker point of the state: `cell(index, pieceId, point)` for the 96 cells (point 0-2 for a
 * corner's U/F/R sticker, 0-3 for a centre or edge) and `gear(site, pieceId, quarter)` for the 24 gears.
 */
function walk(
  state: MalteseState,
  cell: (index: number, pieceId: number, point: number) => void,
  gear: (site: number, pieceId: number, quarter: number) => void,
): void {
  for (const p of PIECES) {
    const s = state.slot[p.id];
    const o = state.ori[p.id];
    if (p.kind === 'corner') {
      const g = CUBE_GROUP.elements[o];
      for (let i = 0; i < 3; i++) {
        const f = faceOf(mat3Vec(g, CORNER_NORMALS[i]));
        if (!f) throw new Error('corner sticker off the faces');
        cell(CORNER_CELL[s].get(f)!, p.id, i);
      }
    } else if (p.kind === 'gear') {
      gear(GEAR_SITE[s], p.id, o);
    } else {
      for (let i = 0; i < 4; i++) cell(RADIAL_CELL[s][POINT_QUADRANT[o][i]], p.id, i);
    }
  }
}

/** What the puzzle shows in this state. */
export function observe(state: MalteseState): Observation {
  const cells = new Uint8Array(CELL_COUNT).fill(UNKNOWN);
  const gearColour = new Uint8Array(GEAR_COUNT).fill(UNKNOWN);
  const gearQuarter = new Uint8Array(GEAR_COUNT);
  walk(
    state,
    (index, id, point) => {
      const p = PIECES[id];
      cells[index] = p.kind === 'corner' ? colourOf(p.colors[point]) : radialPointColour(id, point);
    },
    (site, id, quarter) => {
      gearColour[site] = colourOf(PIECES[id].colors[0]);
      gearQuarter[site] = quarter;
    },
  );
  return { cells, gearColour, gearQuarter };
}

/**
 * Which sticker point sits at every cell (4·pieceId + point) and which gear at every gear place
 * (pieceId). Unlike colours these are all different, so they show exactly how a move permutes places.
 */
export function observeLabels(state: MalteseState): { cells: Int16Array; gears: Int16Array } {
  const cells = new Int16Array(CELL_COUNT).fill(-1);
  const gears = new Int16Array(GEAR_COUNT).fill(-1);
  walk(
    state,
    (index, id, point) => {
      cells[index] = 4 * id + point;
    },
    (site, id) => {
      gears[site] = id;
    },
  );
  return { cells, gears };
}

export interface FromObservation {
  /** The arrangement, when every slot could be filled. Gears of one colour are interchangeable; any is taken. */
  readonly state: MalteseState | null;
  /** What is wrong, in words, with the cells involved. Empty when `state` is a proper arrangement. */
  readonly problems: readonly ObservationProblem[];
}

export interface ObservationProblem {
  readonly message: string;
  /** Cell indices (0-95) involved. */
  readonly cells: readonly number[];
  /** Gear places (0-23) involved. */
  readonly gears: readonly number[];
}

const COLOUR_NAME = ['white', 'yellow', 'green', 'blue', 'red', 'purple'] as const;
export const colourName = (c: number): string => COLOUR_NAME[c] ?? 'unknown';

/**
 * The arrangement an observation shows: which piece sits in every slot, and how it is turned. This
 * only checks that the pieces exist and are used once each; whether moves can reach the arrangement is
 * the legality check's job.
 */
export function fromObservation(obs: Observation): FromObservation {
  const slot = new Int8Array(PIECES.length).fill(-1);
  const ori = new Int8Array(PIECES.length);
  const problems: ObservationProblem[] = [];
  const place = (id: number, s: number, o: number, cells: number[], what: string) => {
    if (slot[id] >= 0) {
      problems.push({ message: `The ${what} appears twice.`, cells, gears: [] });
      return;
    }
    slot[id] = s;
    ori[id] = o;
  };

  // Corners: the piece and orientation whose three stickers match the three cells.
  CORNER_SLOTS.forEach((cs, s) => {
    const cells = cs.faces.map((f) => CORNER_CELL[s].get(f)!);
    let found = false;
    for (const p of PIECES) {
      if (p.kind !== 'corner' || found) continue;
      for (let o = 0; o < CUBE_GROUP.size; o++) {
        if (CORNER_SLOT_OF_ORI[o] !== s) continue;
        const g = CUBE_GROUP.elements[o];
        const ok = CORNER_NORMALS.every((n, i) => obs.cells[CORNER_CELL[s].get(faceOf(mat3Vec(g, n))!)!] === colourOf(p.colors[i]));
        if (ok) {
          place(p.id, s, o, cells, `${p.colors.map((c) => colourName(colourOf(c))).join('-')} corner`);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      const seen = cells.map((c) => colourName(obs.cells[c])).join(', ');
      problems.push({ message: `No corner piece has the colours ${seen} in this order (corner ${cs.name}).`, cells, gears: [] });
    }
  });

  // Centre and edge slots: four cells of one colour are a centre; two and two are an edge in one of four turns.
  RADIAL_SLOTS.forEach((rs, s) => {
    const cells = RADIAL_CELL[s].slice();
    const shown = cells.map((c) => obs.cells[c]);
    let found = false;
    for (const p of PIECES) {
      if ((p.kind !== 'centre' && p.kind !== 'edge') || found) continue;
      for (let k = 0; k < (p.kind === 'centre' ? 1 : 4); k++) {
        const ok = RADIAL_POINTS.every((_, i) => shown[POINT_QUADRANT[k][i]] === radialPointColour(p.id, i));
        if (ok) {
          const what = p.kind === 'centre' ? `${colourName(colourOf(p.colors[0]))} centre` : `${p.colors.map((c) => colourName(colourOf(c))).join('-')} edge`;
          place(p.id, s, k, cells, what);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      problems.push({
        message: `The ${rs.kind} slot ${rs.name} shows ${shown.map(colourName).join(', ')}, which no centre or edge piece can show.`,
        cells,
        gears: [],
      });
    }
  });

  // Gears: any unused gear of the colour shown.
  const spare = new Map<number, number[]>();
  for (const p of PIECES) {
    if (p.kind !== 'gear') continue;
    const c = colourOf(p.colors[0]);
    spare.set(c, [...(spare.get(c) ?? []), p.id]);
  }
  for (let site = 0; site < GEAR_COUNT; site++) {
    const c = obs.gearColour[site];
    const id = spare.get(c)?.shift();
    if (id === undefined) {
      const message = c === UNKNOWN || c > 5 ? 'A gear has no colour yet.' : `There are more than four ${colourName(c)} gears.`;
      problems.push({ message, cells: [], gears: [site] });
      continue;
    }
    slot[id] = GEAR_SLOT_AT[site];
    ori[id] = obs.gearQuarter[site] & 3;
  }

  if (problems.length || slot.some((s) => s < 0)) {
    if (!problems.length) problems.push({ message: 'Some pieces are missing from the cube.', cells: [], gears: [] });
    return { state: null, problems };
  }
  return { state: MalteseState.fromPoses(slot, ori), problems };
}

/** The UFL corner (white, green, red): the piece the solver never moves. */
export const PEGGED_CORNER = PIECES.find((p) => p.kind === 'corner' && p.name === 'UFL')!;

/**
 * The whole-cube rotation (a CUBE_GROUP index) that brings the white-green-red corner home, which is
 * the frame the solver works in. A cube scanned in any orientation becomes the same canonical state.
 */
export function canonicalRotation(state: MalteseState): number {
  return CUBE_GROUP.mul(PEGGED_CORNER.homeOri, CUBE_GROUP.inv(state.ori[PEGGED_CORNER.id]));
}

export function canonicalise(state: MalteseState): MalteseState {
  return state.rotated(canonicalRotation(state));
}
