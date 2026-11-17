import { generateScramble } from '../../vendor/tangram/moves';
import { PuzzleState } from '../../vendor/tangram/state';
import { type Vec3, dot, mat3Vec } from '../../vendor/puzzle/vec';
import { COLOUR_HEX, FACE_VIEWS, MODEL, TABLES } from '../../tangram/model';
import { type PuzzleLook, type Rng, type SceneSpec, type Setting, chance, clamp, gauss, puzzleLook, rotateAbout, sceneAround, shuffled, uni } from '../common/random';

/**
 * What one synthetic photo of the Tangram Cube Extreme looks like: a state, and a camera roughly
 * face-on to one face (as defined by the middle layer, whatever shape the puzzle is in), rolled any way
 * (mostly close to square with the picture, as people take photos) and tilted a little. All plain
 * numbers drawn from a seeded stream; scene.ts only applies them.
 */

export type StateKind = 'scramble' | 'arrangement' | 'cube' | 'near-solved';

/** White's colour index (FACE_IDS order: orange red yellow white green blue). */
const WHITE = 3;

/**
 * Any piece in any cell it can fill, turned any way that fills it. Most of these cannot be reached by
 * moves, which is the point: every position shows every possible content independently of its
 * neighbours, so the reader learns to look, not to guess from the rest of the puzzle.
 */
export function randomArrangement(rng: Rng): PuzzleState | null {
  const n = MODEL.pieces.length;
  const byCell = new Map<number, { piece: number; ori: number }[]>();
  TABLES.placements.forEach((ps) => ps.forEach((p) => byCell.set(p.cell, [...(byCell.get(p.cell) ?? []), { piece: p.piece, ori: p.ori }])));
  for (let attempt = 0; attempt < 40; attempt++) {
    const used = new Set<number>();
    const orient = new Array<number>(n).fill(-1);
    let stuck = false;
    // Cells with the fewest candidates first, in random order among equals.
    const cells = shuffled(rng, TABLES.cells.length).sort((a, b) => byCell.get(a)!.length - byCell.get(b)!.length);
    for (const c of cells) {
      const options = byCell.get(c)!.filter((o) => !used.has(o.piece));
      if (!options.length) {
        stuck = true;
        break;
      }
      const o = options[Math.floor(rng() * options.length)];
      used.add(o.piece);
      orient[o.piece] = o.ori;
    }
    if (stuck) continue;
    const st = PuzzleState.solved(MODEL);
    orient.forEach((o, i) => (st.orient[i] = o));
    return st;
  }
  return null;
}

export function randomState(rng: Rng): { state: PuzzleState; kind: StateKind } {
  const r = rng();
  const state = PuzzleState.solved(MODEL);
  if (r < 0.3) {
    const a = randomArrangement(rng);
    if (a) return { state: a, kind: 'arrangement' };
  }
  if (r < 0.82) {
    state.applyAll(generateScramble(MODEL.variant, rng, 1 + Math.floor(rng() * 40)));
    return { state, kind: 'scramble' };
  }
  if (r < 0.92) {
    // Kept in cube shape (only 120° corner turns and flips), as many people leave it.
    const rounds = 1 + Math.floor(rng() * 30);
    for (let i = 0; i < rounds; i++) {
      state.apply({ axis: chance(rng, 0.5) ? 'U' : 'D', amount: chance(rng, 0.5) ? 2 : -2 });
      state.apply({ axis: (['F', 'R', 'L'] as const)[Math.floor(rng() * 3)], amount: 1 });
    }
    return { state, kind: 'cube' };
  }
  state.applyAll(generateScramble(MODEL.variant, rng, Math.floor(rng() * 3)));
  return { state, kind: 'near-solved' };
}

export interface TangramSpec extends SceneSpec, PuzzleLook {
  readonly seed: number;
  readonly stateKind: StateKind;
  /** The face photographed (index into FACE_IDS / FACE_VIEWS). */
  readonly face: number;
  readonly framing: { readonly tiltDeg: number; readonly rollDeg: number; readonly fill: number; readonly distance: number };
}

export interface TangramOptions {
  readonly face?: number;
  readonly setting?: Setting;
  /** The puzzle's own look and the camera's tone curve, shared by the shots of a scan. */
  readonly look?: PuzzleLook & Partial<Pick<SceneSpec, 'toneMapping' | 'skin'>>;
}

/** Every corner of every piece where the state puts it. */
function vertices(state: PuzzleState): Vec3[] {
  const out: Vec3[] = [];
  MODEL.pieces.forEach((p, i) => {
    const m = state.pieceMatrix(i);
    for (const v of p.poly.vertices) out.push(mat3Vec(m, v));
  });
  return out;
}

const CUBE_NORMALS: readonly Vec3[] = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

export function tangramSpec(seed: number, rng: Rng, kind: StateKind, state: PuzzleState, options: TangramOptions = {}): TangramSpec {
  const drawnFace = Math.floor(rng() * 6);
  const face = options.face ?? drawnFace;
  const view = FACE_VIEWS[face];
  const toward = view.normal;
  // Mostly close to square with the picture (any of the four ways up), sometimes anyhow.
  const square = chance(rng, 0.65);
  const rollDeg = square ? 90 * Math.floor(rng() * 4) + clamp(gauss(rng) * 7, -25, 25) : uni(rng, 0, 360);
  const up = rotateAbout(view.up, toward, (rollDeg * Math.PI) / 180);
  const pts = vertices(state);
  const extent = Math.max(...pts.map((p) => Math.hypot(p[0], p[1], p[2])));
  const support = (n: Vec3) => Math.max(...pts.map((p) => -dot(n, p)));
  // Round the side: the four faces next to this one.
  const sides = CUBE_NORMALS.filter((n) => Math.abs(dot(n, toward)) < 0.5);
  const { scene, framing } = sceneAround(rng, {
    toward,
    up,
    extent,
    subject: 1.25,
    support,
    restOn: sides,
    sides,
    setting: options.setting,
    toneMapping: options.look?.toneMapping,
    skin: options.look?.skin,
    maxTiltDeg: 20,
  });
  const look = options.look ?? puzzleLook(rng, COLOUR_HEX, WHITE);
  return {
    ...scene,
    stickers: look.stickers,
    stickerLook: look.stickerLook,
    body: look.body,
    seed,
    stateKind: kind,
    face,
    framing: { ...framing, rollDeg },
  };
}
