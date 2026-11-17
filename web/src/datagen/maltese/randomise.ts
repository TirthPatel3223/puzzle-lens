import { COLORS } from '../../vendor/maltese/config';
import { CORNER_SLOTS, FACES, FACE_NORMAL, type Face, GEAR_SLOTS, RADIAL_SLOTS } from '../../vendor/maltese/geometry';
import { CORNER_SLOT_OF_ORI, CUBE_GROUP, PIECES } from '../../vendor/maltese/model';
import { generateScramble } from '../../vendor/maltese/scramble';
import { MalteseState } from '../../vendor/maltese/state';
import { type Vec3, add, cross, dot, normalize, scale } from '../../vendor/puzzle/vec';
import { FACE_RIGHT, FACE_UP } from '../../maltese/observation';
import {
  BACKDROPS,
  type BackdropKind,
  type FingerSpec,
  type LightSpec,
  type Rng,
  SKIN,
  type Setting,
  type TextureSpec,
  chance,
  clamp,
  gauss,
  hexToRgb,
  hslToRgb,
  lightColour,
  pick,
  rgbToHsl,
  rotateAbout,
  shuffled,
  textureSpec,
  uni,
} from '../common/random';

export { chance, gauss, pick, uni };
export type { FingerSpec, LightSpec, Rng, Setting, TextureSpec };

/**
 * What one synthetic photo of the Maltese Gear Cube looks like, drawn from a seeded random stream. Everything random about a
 * sample is decided here as plain numbers, so a sample can be reproduced from its seed, inspected, and
 * tested without a GPU; `scene.ts` only applies it.
 *
 * The ranges are deliberately wider than real photos (domain randomisation): a model that copes with
 * all of this treats the real camera, lamp and desk as one more variation.
 */

export type StateKind = 'arrangement' | 'scramble' | 'near-solved';

/**
 * Any piece in any slot of its kind, turned any way. Most of these cannot be reached by moves, which
 * is the point: every slot shows every possible content equally often and independently of its
 * neighbours, so the reader learns to look, not to guess from the rest of the face.
 */
export function randomArrangement(rng: Rng): MalteseState {
  const slot = new Int8Array(PIECES.length);
  const ori = new Int8Array(PIECES.length);
  const corners = PIECES.filter((p) => p.kind === 'corner');
  const radial = PIECES.filter((p) => p.kind === 'centre' || p.kind === 'edge');
  const gears = PIECES.filter((p) => p.kind === 'gear');
  const cs = shuffled(rng, CORNER_SLOTS.length);
  corners.forEach((p, i) => {
    const options = CUBE_GROUP.elements.map((_, o) => o).filter((o) => CORNER_SLOT_OF_ORI[o] === cs[i]);
    slot[p.id] = cs[i];
    ori[p.id] = pick(rng, options);
  });
  const rs = shuffled(rng, RADIAL_SLOTS.length);
  radial.forEach((p, i) => {
    slot[p.id] = rs[i];
    ori[p.id] = Math.floor(rng() * 4);
  });
  const gs = shuffled(rng, GEAR_SLOTS.length);
  gears.forEach((p, i) => {
    slot[p.id] = gs[i];
    ori[p.id] = Math.floor(rng() * 4);
  });
  return MalteseState.fromPoses(slot, ori);
}

export function randomState(rng: Rng): { state: MalteseState; kind: StateKind } {
  const r = rng();
  if (r < 0.45) return { state: randomArrangement(rng), kind: 'arrangement' };
  const state = MalteseState.solved();
  if (r < 0.9) {
    state.applyAll(generateScramble(rng, 1 + Math.floor(rng() * 50)));
    return { state, kind: 'scramble' };
  }
  state.applyAll(generateScramble(rng, Math.floor(rng() * 4)));
  return { state, kind: 'near-solved' };
}

export interface SampleSpec {
  readonly seed: number;
  readonly stateKind: StateKind;
  /** Index into FACES of the face photographed. */
  readonly face: number;
  readonly camera: {
    readonly position: Vec3;
    readonly lookAt: Vec3;
    readonly up: Vec3;
    /** Vertical field of view, degrees. */
    readonly fov: number;
  };
  /** For the record: how the shot was framed. */
  readonly framing: { readonly tiltDeg: number; readonly rollDeg: number; readonly quarter: number; readonly fill: number; readonly distance: number };
  readonly setting: Setting;
  readonly key: LightSpec;
  readonly fills: readonly LightSpec[];
  readonly shadow: { readonly intensity: number; readonly radius: number };
  readonly hemisphere: { readonly sky: readonly [number, number, number]; readonly ground: readonly [number, number, number]; readonly intensity: number };
  readonly environment: { readonly index: number; readonly rotation: number; readonly intensity: number };
  readonly toneMapping: 'none' | 'linear' | 'neutral' | 'aces' | 'agx';
  readonly exposure: number;
  /** Sticker colour of each of the six colours (linear-light is the renderer's business; these are sRGB 0-1). */
  readonly stickers: readonly (readonly [number, number, number])[];
  readonly stickerLook: { readonly roughness: number; readonly clearcoat: number; readonly spread: number };
  readonly body: { readonly colour: readonly [number, number, number]; readonly roughness: number; readonly clearcoat: number };
  readonly backdrop: {
    readonly texture: TextureSpec;
    /** Distance of the plane behind the cube's centre, along the viewing direction. */
    readonly distance: number;
    readonly lit: boolean;
    readonly brightness: number;
  };
  readonly ground: { readonly normal: Vec3; readonly offset: number; readonly texture: TextureSpec; readonly roughness: number } | null;
  readonly fingers: readonly FingerSpec[];
  readonly skin: readonly [number, number, number];
  readonly palm: { readonly centre: Vec3; readonly scale: Vec3 } | null;
}

/** The six sticker colours as this particular cube and print run has them. */
function stickerColours(rng: Rng): [number, number, number][] {
  return FACES.map((f: Face) => {
    const [h, s, l] = rgbToHsl(hexToRgb(COLORS[f]));
    if (f === 'U') return hslToRgb(rng(), uni(rng, 0, 0.14), uni(rng, 0.8, 0.97)); // white, maybe faintly tinted
    return hslToRgb(h + clamp(gauss(rng) * 0.012, -0.03, 0.03), clamp(s * uni(rng, 0.68, 1.08), 0, 1), clamp(l * uni(rng, 0.78, 1.18), 0.05, 0.92));
  });
}

/** What a caller may fix instead of leaving to chance (a scan is six shots of one cube in one room). */
export interface SampleOptions {
  readonly face?: number;
  /** Quarter turns the face is rolled in the picture. */
  readonly quarter?: number;
  readonly setting?: Setting;
  /** The cube's own look, shared by all shots of a scan. */
  readonly look?: Pick<SampleSpec, 'stickers' | 'stickerLook' | 'body' | 'toneMapping' | 'skin'>;
}

export function sampleSpec(seed: number, rng: Rng, stateKind: StateKind, options: SampleOptions = {}): SampleSpec {
  const drawnFace = Math.floor(rng() * 6);
  const face = options.face ?? drawnFace;
  const f = FACES[face];
  const n = FACE_NORMAL[f];
  const right = FACE_RIGHT[f];
  const up = FACE_UP[f];

  // --- camera: roughly face-on, as a person holding a phone would manage -----------------------------
  const drawnQuarter = Math.floor(rng() * 4);
  const quarter = options.quarter ?? drawnQuarter;
  const rollDeg = clamp(gauss(rng) * 6, -20, 20);
  const phi = (quarter * Math.PI) / 2 + (rollDeg * Math.PI) / 180;
  const camUp = rotateAbout(up, n, phi);
  const camRight = cross(camUp, n);
  const tiltDeg = Math.min(22, Math.abs(gauss(rng)) * 7.5);
  const tiltAz = uni(rng, 0, 2 * Math.PI);
  const t = Math.tan((tiltDeg * Math.PI) / 180);
  const dir = normalize(add(n, add(scale(camRight, t * Math.cos(tiltAz)), scale(camUp, t * Math.sin(tiltAz)))));
  const fov = uni(rng, 18, 66);
  const fill = uni(rng, 0.25, 0.86); // face width as a fraction of the picture
  const halfVisible = 1 / fill; // half-width of the picture at the face plane, model units
  const distance = 1 + halfVisible / Math.tan((fov * Math.PI) / 360);
  const slack = Math.max(0, halfVisible - 1.12) * 0.92;
  const lookAt = add(scale(camRight, uni(rng, -slack, slack)), scale(camUp, uni(rng, -slack, slack)));
  const position = add(lookAt, scale(dir, distance));

  // --- light ----------------------------------------------------------------------------------------
  const lightDir = (minElevation: number): Vec3 => {
    const el = uni(rng, minElevation, Math.PI / 2);
    const az = uni(rng, 0, 2 * Math.PI);
    return normalize(add(scale(dir, Math.sin(el)), add(scale(camRight, Math.cos(el) * Math.cos(az)), scale(camUp, Math.cos(el) * Math.sin(az)))));
  };
  const key: LightSpec = { dir: lightDir(0.16), intensity: uni(rng, 0.5, 2.8), colour: lightColour(rng) };
  const fills: LightSpec[] = [0, 1].map(() => ({
    dir: lightDir(-0.3),
    intensity: chance(rng, 0.7) ? uni(rng, 0.05, 1.2) : 0,
    colour: lightColour(rng),
  }));
  const sky = lightColour(rng);
  const groundTint = lightColour(rng);

  const drawnSetting: Setting = pick(rng, ['floating', 'table-top', 'table-top', 'table-side', 'table-side', 'hand', 'hand', 'hand'] as const);
  const setting = options.setting ?? drawnSetting;

  // --- what is behind and under the cube ---------------------------------------------------------------
  const tableKinds: readonly BackdropKind[] = ['solid', 'solid', 'wood', 'wood', 'gradient', 'tiles', 'blobs'];
  const backdrop = {
    texture: textureSpec(rng, setting === 'table-top' ? tableKinds : BACKDROPS),
    // On a table seen from above the "backdrop" is the table itself, right under the cube.
    distance: setting === 'table-top' ? uni(rng, 1.0, 1.38) : uni(rng, 3, 14),
    lit: setting === 'table-top' || chance(rng, 0.3),
    brightness: uni(rng, 0.25, 1.15),
  };
  let ground: SampleSpec['ground'] = null;
  if (setting === 'table-side') {
    // The cube rests on the face that is lowest in the picture.
    const axes: Vec3[] = [up, scale(up, -1), right, scale(right, -1)];
    const down = axes.reduce((best, a) => (dot(a, camUp) < dot(best, camUp) ? a : best));
    ground = { normal: scale(down, -1), offset: uni(rng, 1.0, 1.38), texture: textureSpec(rng, tableKinds), roughness: uni(rng, 0.35, 1) };
  }

  // --- a hand ---------------------------------------------------------------------------------------
  const fingers: FingerSpec[] = [];
  let palm: SampleSpec['palm'] = null;
  if (setting === 'hand') {
    // The hand comes in from one side of the picture (mostly from below) and grips the two side faces
    // across from each other, fingers lying along them. Tips sometimes curl over the front edge.
    const wanted = pick(rng, [scale(camUp, -1), scale(camUp, -1), scale(camUp, -1), camRight, scale(camRight, -1), camUp]);
    const axes: Vec3[] = [up, scale(up, -1), right, scale(right, -1)];
    const from = axes.reduce((best, a) => (dot(a, wanted) > dot(best, wanted) ? a : best));
    const grip = cross(n, from);
    const count = 1 + Math.floor(rng() * 4);
    for (let i = 0; i < count; i++) {
      const side = scale(grip, chance(rng, 0.5) ? 1 : -1);
      const radius = uni(rng, 0.2, 0.33);
      const length = uni(rng, 1.6, 3.8);
      const axis = normalize(add(scale(from, -1), add(scale(n, uni(rng, -0.45, 0.45)), scale(side, -uni(rng, 0, 0.22)))));
      const tip = add(add(scale(side, 1 + radius * uni(rng, 0.5, 1.1)), scale(from, -uni(rng, -0.7, 0.95))), scale(n, uni(rng, -0.7, 0.95)));
      fingers.push({ centre: add(tip, scale(axis, -(length / 2 - radius))), axis, radius, length });
    }
    if (chance(rng, 0.45)) {
      palm = {
        centre: add(scale(n, -uni(rng, 1.7, 3.2)), add(scale(from, uni(rng, 0.2, 2.2)), scale(grip, uni(rng, -0.9, 0.9)))),
        scale: [uni(rng, 1.6, 3.2), uni(rng, 1.6, 3.2), uni(rng, 0.8, 1.6)],
      };
    }
  }

  // --- exposure: a camera meters the scene, so brighter light does not mean a brighter picture ---------
  const lit = (l: LightSpec) => l.intensity * Math.max(0.12, dot(l.dir, n));
  const hemisphereIntensity = uni(rng, 0.05, 1.0);
  const environmentIntensity = uni(rng, 0.1, 1.2);
  const illumination = lit(key) + fills.reduce((a, l) => a + lit(l), 0) + 0.6 * hemisphereIntensity + 0.8 * environmentIntensity;
  const exposure = clamp((uni(rng, 0.62, 1.42) * 1.7) / illumination, 0.3, 3);

  const grey = uni(rng, 0.02, 0.1);
  return {
    seed,
    stateKind,
    face,
    camera: { position, lookAt, up: camUp, fov },
    framing: { tiltDeg, rollDeg, quarter, fill, distance },
    setting,
    key,
    fills,
    shadow: { intensity: uni(rng, 0.35, 1), radius: uni(rng, 0.5, 6) },
    hemisphere: { sky, ground: groundTint.map((c) => c * uni(rng, 0.2, 0.8)) as [number, number, number], intensity: hemisphereIntensity },
    environment: { index: Math.floor(rng() * 1e6), rotation: uni(rng, 0, 2 * Math.PI), intensity: environmentIntensity },
    toneMapping: options.look?.toneMapping ?? pick(rng, ['none', 'linear', 'neutral', 'neutral', 'aces', 'agx'] as const),
    exposure,
    stickers: options.look?.stickers ?? stickerColours(rng),
    stickerLook: options.look?.stickerLook ?? { roughness: uni(rng, 0.22, 0.75), clearcoat: chance(rng, 0.5) ? uni(rng, 0, 0.7) : 0, spread: uni(rng, 0, 0.035) },
    body: options.look?.body ?? { colour: [grey * uni(rng, 0.9, 1.1), grey, grey * uni(rng, 0.9, 1.15)], roughness: uni(rng, 0.22, 0.65), clearcoat: uni(rng, 0, 0.6) },
    backdrop,
    ground,
    fingers,
    skin: options.look?.skin ?? (pick(rng, SKIN).map((c) => clamp(c * uni(rng, 0.88, 1.1), 0, 1)) as [number, number, number]),
    palm,
  };
}
