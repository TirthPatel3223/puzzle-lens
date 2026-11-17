import { CUT_S } from '../vendor/tangram/config';
import { AXES, inRegion } from '../vendor/tangram/moves';
import { FACE_IDS } from '../vendor/tangram/pieces';
import { PuzzleState } from '../vendor/tangram/state';
import { COLOUR_HEX, COLOUR_NAMES, EMPTY, FACE_MIN_COS, FACE_VIEWS, MODEL, TABLES, observe } from './model';

/**
 * Everything the Python side needs to know about the Tangram Cube Extreme, generated from the engine
 * (so Python has no geometry or move logic of its own). Written to shared/tangram/layout.json by
 * tools/export-shared.gen.ts.
 */
export function buildTangramLayout() {
  const g = MODEL.group;
  const round = (x: number) => Math.round(x * 1e6) / 1e6;
  const vec = (v: readonly number[]) => v.map(round);
  return {
    version: 1,
    puzzle: 'tangram-cube-extreme',
    variant: MODEL.variant.id,
    cutS: CUT_S,
    edgeMm: MODEL.variant.edgeMm,
    empty: EMPTY,
    colours: FACE_IDS.map((f, i) => ({ name: COLOUR_NAMES[i], hex: COLOUR_HEX[i], face: f })),
    /** The 12 rotations of D6 (row-major 3x3) and their multiplication table: mul[a][b] = index of a·b. */
    group: {
      identity: g.identity,
      matrices: g.elements.map((m) => vec(m)),
      mul: g.elements.map((_, a) => g.elements.map((__, b) => g.mul(a, b))),
      inv: g.elements.map((_, a) => g.inv(a)),
    },
    /** Per axis: one step's rotation (an index into group) and the cells it turns. */
    axes: Object.fromEntries(
      AXES.map((axis) => {
        const def = MODEL.axes[axis];
        const cells = TABLES.cells.filter((c) => inRegion(def.region, MODEL.pieces[c.index].poly.centroid)).map((c) => c.index);
        return [
          axis,
          { kind: def.kind, order: def.order, rotAxis: vec(def.rotAxis), region: { n: vec(def.region.n), d: round(def.region.d) }, step: MODEL.moveElement({ axis, amount: 1 }), cells },
        ];
      }),
    ),
    pieces: MODEL.pieces.map((p, i) => ({
      id: p.id,
      layer: p.layer,
      code: p.code,
      colours: TABLES.stickerColour[i],
      homePositions: TABLES.placements[i].find((pl) => pl.ori === g.identity)!.positions,
      reachable: TABLES.placements[i].map((pl) => pl.ori),
    })),
    cells: TABLES.cells.map((c) => ({ id: c.index, layer: c.layer, key: c.key })),
    positions: TABLES.positions.map((p) => ({
      centre: vec(p.centre),
      normal: vec(p.normal),
      polygon: p.polygon.map(vec),
      onCube: p.onCube,
      facing: p.facing,
      cell: p.cell,
      home: p.home,
    })),
    placements: TABLES.placements.flat().map((pl) => ({ piece: pl.piece, ori: pl.ori, cell: pl.cell, positions: pl.positions })),
    /** One step of each axis as a map of positions: new[i] = old[from[i]]. */
    moves: Object.fromEntries(AXES.map((a) => [a, TABLES.moveFrom[a]])),
    solved: Array.from(observe(PuzzleState.solved(MODEL))),
    /** The six face-on photos: each face's frame (turning corner at the bottom-right) and corners. */
    faces: FACE_VIEWS.map((f) => ({
      face: f.face,
      end: f.end,
      normal: vec(f.normal),
      right: vec(f.right),
      up: vec(f.up),
      corners: f.corners.map(vec),
    })),
    faceMinCos: FACE_MIN_COS,
  };
}

export type TangramLayout = ReturnType<typeof buildTangramLayout>;
