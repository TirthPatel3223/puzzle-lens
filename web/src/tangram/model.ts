import { CUT_S, FACE_COLORS } from '../vendor/tangram/config';
import { type PuzzleModel, createModel } from '../vendor/tangram/model';
import { AXES, type Axis, type Move, formatMove, inRegion } from '../vendor/tangram/moves';
import { FACE_IDS, FACE_NORMALS, type FaceId } from '../vendor/tangram/pieces';
import { PuzzleState } from '../vendor/tangram/state';
import { type Vec3, add, dot, mat3Vec, normalize, scale, sub } from '../vendor/puzzle/vec';

/**
 * What a Tangram Cube Extreme shows, as the scanner, the decoder and the solver see it.
 *
 * Every sticker always sits in one of 120 **positions** (a centre point and an outward normal): the 66
 * where the stickers are on the solved cube, and 54 where they are when part of the puzzle has turned
 * 60° and stands out of the cube outline. An observation is one value per position: the colour of the
 * sticker there (0-5), or EMPTY. This is also the solver's state vector (one value per sticker frame,
 * EMPTY for an unoccupied one), up to the order of the positions.
 *
 * Every piece always fills one of 45 **cells**: the three layers along the T-B diagonal, each cut into
 * columns by the edge-cut planes (which are parallel to the diagonal). A piece's orientation (one of the
 * 12 rotations of D6) decides its cell and the positions of its stickers: a **placement**.
 *
 * It is photographed face-on, one photo per face, whatever its shape. The faces are those of the
 * middle layer, which never changes shape (it only flips in place), so every face always shows some flat
 * stickers to square the camera to. Every face has one turning corner (T or B) at one of its corners,
 * and the T and B corner pieces never leave their corners: that corner fixes which way up a photo is.
 * A sticker that has turned 60° out of the cube outline is seen at 48° in two photos, so it can never be
 * mistaken for the flat sticker it looks like straight down the diagonal (see FACE_VIEWS).
 *
 * Colours are indices into FACE_IDS (+x -x +y -y +z -z): orange, red, yellow, white, green, blue.
 */

export const MODEL: PuzzleModel = createModel('extreme', CUT_S);
export const EMPTY = 6;
export const COLOUR_COUNT = 6;
export const COLOUR_NAMES = ['orange', 'red', 'yellow', 'white', 'green', 'blue'] as const;
export const COLOUR_HEX: readonly string[] = FACE_IDS.map((f) => FACE_COLORS[f]);
export const colourOfFace = (f: FaceId): number => FACE_IDS.indexOf(f);
export const colourName = (c: number): string => (c === EMPTY ? 'empty' : (COLOUR_NAMES[c] ?? 'unknown'));

/** Unit vector from the centre to the T corner (+x+y+z). */
export const DIAGONAL: Vec3 = normalize([1, 1, 1]);

export type End = 'T' | 'B';

export interface Position {
  readonly index: number;
  /** Mean of the sticker polygon's corners, model frame (the cube is [-1, 1]^3). */
  readonly centre: Vec3;
  readonly normal: Vec3;
  /** The sticker outline (the cell face, before the black border) at this position. */
  readonly polygon: readonly Vec3[];
  /** On the cube outline (true for the 66 home positions). */
  readonly onCube: boolean;
  /** The turning corner this position faces (its normal leans towards it). */
  readonly facing: End;
  /** The cell of the piece whose sticker sits here. */
  readonly cell: number;
  /** The sticker that is here when the puzzle is solved, if any. */
  readonly home: { readonly piece: number; readonly sticker: number } | null;
}

export interface Cell {
  readonly index: number;
  readonly layer: 'T' | 'M' | 'B';
  /** Layer and the side of each edge-cut plane, e.g. "T101011". */
  readonly key: string;
}

export interface Placement {
  readonly piece: number;
  /** Index into MODEL.group. */
  readonly ori: number;
  readonly cell: number;
  /** Position of each of the piece's stickers (in the order of PieceDef.stickers). */
  readonly positions: readonly number[];
}

const KEY_DIGITS = 5;
const roundKey = (v: Vec3) => v.map((x) => (Math.abs(x) < 5e-7 ? 0 : x).toFixed(KEY_DIGITS)).join(',');
const frameKey = (c: Vec3, n: Vec3) => `${roundKey(c)}|${roundKey(n)}`;

function polygonMean(poly: readonly Vec3[]): Vec3 {
  let s: Vec3 = [0, 0, 0];
  for (const p of poly) s = add(s, p);
  return scale(s, 1 / poly.length);
}

/** The cell (layer + column) that contains point p. */
export function cellKeyOf(p: Vec3, model: PuzzleModel = MODEL): string {
  const { b, c } = model.cut;
  const s = p[0] + p[1] + p[2];
  const layer = s > c ? 'T' : s < -c ? 'B' : 'M';
  const sides = [p[0] - p[1], p[1] - p[2], p[2] - p[0]].flatMap((u) => [u > b ? 1 : 0, u > -b ? 1 : 0]).join('');
  return layer + sides;
}

/** Orientations each piece can reach: a piece moves alone, so this is a search over the 12 rotations. */
function reachableOrientations(model: PuzzleModel): number[][] {
  const steps = AXES.map((axis) => ({ axis, g: model.moveElement({ axis, amount: 1 }) }));
  return model.pieces.map((piece) => {
    const seen = new Set<number>([model.group.identity]);
    const queue = [model.group.identity];
    while (queue.length) {
      const o = queue.shift()!;
      const centre = mat3Vec(model.group.elements[o], piece.poly.centroid);
      for (const { axis, g } of steps) {
        if (!inRegion(model.axes[axis].region, centre)) continue;
        const next = model.group.mul(g, o);
        if (!seen.has(next)) {
          seen.add(next);
          queue.push(next);
        }
      }
    }
    return [...seen].sort((a, b) => a - b);
  });
}

export interface TangramTables {
  readonly positions: readonly Position[];
  readonly cells: readonly Cell[];
  /** placements[piece] = every placement the piece can reach. */
  readonly placements: readonly (readonly Placement[])[];
  /** Colour of each sticker of each piece (its home face). */
  readonly stickerColour: readonly (readonly number[])[];
  /** One step of each axis (U, D: 60° clockwise seen from that corner; F, R, L: the 180° flip) as a map of positions: content new[i] = old[from[i]]. */
  readonly moveFrom: Readonly<Record<Axis, readonly number[]>>;
}

function buildTables(model: PuzzleModel): TangramTables {
  const reach = reachableOrientations(model);
  const stickerColour = model.pieces.map((p) => p.stickers.map((s) => colourOfFace(s.face)));

  // Every sticker frame any placement produces.
  type Raw = { centre: Vec3; normal: Vec3; polygon: Vec3[]; cellKey: string; home: Position['home'] };
  const raw = new Map<string, Raw>();
  const placementKeys: { piece: number; ori: number; cellKey: string; keys: string[] }[] = [];
  model.pieces.forEach((piece, i) => {
    for (const o of reach[i]) {
      const g = model.group.elements[o];
      const cellKey = cellKeyOf(mat3Vec(g, piece.poly.centroid), model);
      const keys = piece.stickers.map((s, k) => {
        const polygon = s.polygon.map((v) => mat3Vec(g, v));
        const centre = polygonMean(polygon);
        const normal = mat3Vec(g, FACE_NORMALS[s.face]);
        const key = frameKey(centre, normal);
        const home = o === model.group.identity ? { piece: i, sticker: k } : null;
        const had = raw.get(key);
        if (!had) raw.set(key, { centre, normal, polygon, cellKey, home });
        else {
          if (had.cellKey !== cellKey) throw new Error(`position ${key} belongs to two cells`);
          if (home) had.home = home;
        }
        return key;
      });
      placementKeys.push({ piece: i, ori: o, cellKey, keys });
    }
  });

  // Cells in the order of the pieces that fill them when solved.
  const cellKeys = model.pieces.map((p) => cellKeyOf(p.poly.centroid, model));
  if (new Set(cellKeys).size !== cellKeys.length) throw new Error('two pieces share a cell when solved');
  const cells: Cell[] = cellKeys.map((key, index) => ({ index, key, layer: key[0] as Cell['layer'] }));
  const cellIndex = new Map(cellKeys.map((k, i) => [k, i]));

  // Positions: home positions first (by piece, then sticker), then the rest by coordinates.
  const entries = [...raw.entries()];
  const homes = entries.filter(([, r]) => r.home).sort(([, a], [, b]) => a.home!.piece - b.home!.piece || a.home!.sticker - b.home!.sticker);
  const others = entries.filter(([, r]) => !r.home).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  const ordered = [...homes, ...others];
  const posIndex = new Map(ordered.map(([k], i) => [k, i]));
  const positions: Position[] = ordered.map(([, r], index) => {
    const cell = cellIndex.get(r.cellKey);
    if (cell === undefined) throw new Error(`position in an unknown cell ${r.cellKey}`);
    const onCube = Math.max(...r.normal.map(Math.abs)) > 1 - 1e-6 && Math.max(...r.centre.map(Math.abs)) > 1 - 1e-6;
    return { index, centre: r.centre, normal: r.normal, polygon: r.polygon, onCube, facing: dot(r.normal, DIAGONAL) > 0 ? 'T' : 'B', cell, home: r.home };
  });

  const placements: Placement[][] = model.pieces.map(() => []);
  for (const p of placementKeys) {
    placements[p.piece].push({ piece: p.piece, ori: p.ori, cell: cellIndex.get(p.cellKey)!, positions: p.keys.map((k) => posIndex.get(k)!) });
  }

  // Moves as maps of positions: a sticker moves when its piece's cell is in the turning region.
  const cellInside = (axis: Axis, cell: number) => inRegion(model.axes[axis].region, model.pieces[cell].poly.centroid);
  const moveFrom = {} as Record<Axis, number[]>;
  for (const axis of AXES) {
    const g = model.group.elements[model.moveElement({ axis, amount: 1 })];
    const from = positions.map((p) => p.index);
    for (const p of positions) {
      if (!cellInside(axis, p.cell)) continue;
      const to = posIndex.get(frameKey(mat3Vec(g, p.centre), mat3Vec(g, p.normal)));
      if (to === undefined) throw new Error(`move ${axis} takes position ${p.index} off the position set`);
      from[to] = p.index;
    }
    if (new Set(from).size !== positions.length) throw new Error(`move ${axis} is not a permutation of the positions`);
    moveFrom[axis] = from;
  }
  return { positions, cells, placements, stickerColour, moveFrom };
}

export const TABLES: TangramTables = buildTables(MODEL);
export const POSITION_COUNT = TABLES.positions.length;

/** The placement of piece i at orientation o (undefined if the piece cannot get there). */
const PLACEMENT_AT: readonly ReadonlyMap<number, Placement>[] = TABLES.placements.map((ps) => new Map(ps.map((p) => [p.ori, p])));
export const placementOf = (piece: number, ori: number): Placement | undefined => PLACEMENT_AT[piece].get(ori);

/** What the puzzle shows: per position, the colour of the sticker there or EMPTY. */
export function observe(state: PuzzleState): Uint8Array {
  const out = new Uint8Array(POSITION_COUNT).fill(EMPTY);
  for (let i = 0; i < MODEL.pieces.length; i++) {
    const pl = placementOf(i, state.orient[i]);
    if (!pl) throw new Error(`piece ${i} cannot be at orientation ${state.orient[i]}`);
    pl.positions.forEach((p, k) => {
      out[p] = TABLES.stickerColour[i][k];
    });
  }
  return out;
}

/** Which sticker is at every position (4·piece + sticker), -1 where empty. Unlike colours these are all different. */
export function observeLabels(state: PuzzleState): Int16Array {
  const out = new Int16Array(POSITION_COUNT).fill(-1);
  for (let i = 0; i < MODEL.pieces.length; i++) {
    placementOf(i, state.orient[i])!.positions.forEach((p, k) => (out[p] = 4 * i + k));
  }
  return out;
}

/** A state from an orientation per piece (as the server's decoder returns it). */
export function stateFromOrient(orient: readonly number[]): PuzzleState {
  const st = PuzzleState.solved(MODEL);
  if (orient.length !== st.orient.length) throw new Error(`need ${st.orient.length} orientations, got ${orient.length}`);
  orient.forEach((o, i) => (st.orient[i] = o));
  return st;
}

// ------------------------------------------------------------------------------------------- faces

/**
 * The six photos: each face straight on. A face's frame has x right and y up as the face is seen from
 * outside, and is turned so that the face's turning corner (T for +x +y +z, B for the others) is at its
 * bottom-right: corners are listed top-left, top-right, bottom-right (the turning corner), bottom-left,
 * which is the order the locator finds them in. The D3 symmetry of the cuts (120° turns about the
 * diagonal, the three flips) carries any face's frame onto any other's, so the positions a face shows
 * sit at the same places in every face's frame.
 */
export interface FaceView {
  readonly face: FaceId;
  readonly end: End;
  readonly normal: Vec3;
  readonly right: Vec3;
  readonly up: Vec3;
  /** Model points of the face's corners: top-left, top-right, bottom-right (the turning corner), bottom-left. */
  readonly corners: readonly Vec3[];
}

/** A position takes part in a face's photo when its normal is within 60° of the face's (flat: 0°, turned out: 48°). */
export const FACE_MIN_COS = 0.5;

function faceView(face: FaceId): FaceView {
  const n = FACE_NORMALS[face];
  const end: End = dot(n, DIAGONAL) > 0 ? 'T' : 'B';
  const apex: Vec3 = end === 'T' ? [1, 1, 1] : [-1, -1, -1];
  const w = sub(apex, n); // from the face centre to its turning corner, = right - up
  const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const right = scale(add(w, cross(n, w)), 0.5);
  const up = cross(n, right);
  const at = (x: number, y: number) => add(n, add(scale(right, x), scale(up, y)));
  return { face, end, normal: n, right, up, corners: [at(-1, 1), at(1, 1), at(1, -1), at(-1, -1)] };
}

export const FACE_VIEWS: readonly FaceView[] = FACE_IDS.map(faceView);

/** A model point in a face's frame: (right, up, height above the face plane). */
export function inFaceFrame(view: FaceView, p: Vec3): Vec3 {
  return [dot(p, view.right), dot(p, view.up), dot(p, view.normal) - 1];
}

/** A move written the simulator's way ("U2'", "F"). */
export const moveName = (m: Move): string => formatMove(m);
