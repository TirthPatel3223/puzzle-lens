import { mulberry32 } from '../../vendor/puzzle/rng';
import { type Vec3, add, cross, dot, normalize, scale } from '../../vendor/puzzle/vec';

/**
 * Randomness every puzzle's synthetic photos share: the generic draws, light colours, backdrops, sticker
 * shades, and a whole photo studio around a viewing direction (`sceneAround`). Everything random about a
 * sample is decided as plain numbers, so a sample can be reproduced from its seed, inspected, and tested
 * without a GPU; the studio only applies it.
 *
 * The ranges are deliberately wider than real photos (domain randomisation): a model that copes with all
 * of this treats the real camera, lamp and desk as one more variation.
 */

export type Rng = () => number;

export const uni = (rng: Rng, a: number, b: number): number => a + (b - a) * rng();
export const pick = <T>(rng: Rng, items: readonly T[]): T => items[Math.min(items.length - 1, Math.floor(rng() * items.length))];
export const chance = (rng: Rng, p: number): boolean => rng() < p;
/** Standard normal (Box-Muller). */
export function gauss(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
export const clamp = (x: number, a: number, b: number): number => Math.min(b, Math.max(a, x));

export function shuffled(rng: Rng, n: number): number[] {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export type Setting = 'floating' | 'table-top' | 'table-side' | 'hand';
export type BackdropKind = 'solid' | 'gradient' | 'wood' | 'blobs' | 'clutter' | 'tiles';
export type Rgb = readonly [number, number, number];

export interface TextureSpec {
  readonly kind: BackdropKind;
  readonly seed: number;
  readonly blur: number;
  /** Texture repeats across the plane. */
  readonly repeat: number;
}

export interface LightSpec {
  /** Direction from the puzzle towards the light, world frame, unit. */
  readonly dir: Vec3;
  readonly intensity: number;
  readonly colour: Rgb;
}

export interface FingerSpec {
  /** Centre of the capsule, world frame. */
  readonly centre: Vec3;
  /** Unit axis of the capsule. */
  readonly axis: Vec3;
  readonly radius: number;
  readonly length: number;
}

/** Everything about a photo except the puzzle itself: camera, light, what is behind and under it, a hand. */
export interface SceneSpec {
  readonly camera: {
    readonly position: Vec3;
    readonly lookAt: Vec3;
    readonly up: Vec3;
    /** Vertical field of view, degrees. */
    readonly fov: number;
  };
  readonly setting: Setting;
  readonly key: LightSpec;
  readonly fills: readonly LightSpec[];
  readonly shadow: { readonly intensity: number; readonly radius: number };
  readonly hemisphere: { readonly sky: Rgb; readonly ground: Rgb; readonly intensity: number };
  readonly environment: { readonly index: number; readonly rotation: number; readonly intensity: number };
  readonly toneMapping: 'none' | 'linear' | 'neutral' | 'aces' | 'agx';
  readonly exposure: number;
  readonly backdrop: {
    readonly texture: TextureSpec;
    /** Distance of the plane behind the puzzle's centre, along the viewing direction. */
    readonly distance: number;
    readonly lit: boolean;
    readonly brightness: number;
  };
  readonly ground: { readonly normal: Vec3; readonly offset: number; readonly texture: TextureSpec; readonly roughness: number } | null;
  readonly fingers: readonly FingerSpec[];
  readonly skin: Rgb;
  readonly palm: { readonly centre: Vec3; readonly scale: Vec3 } | null;
}

/** The puzzle's own look: sticker shades and plastic, shared by all photos of one puzzle. */
export interface PuzzleLook {
  /** Sticker colour of each colour index (sRGB 0-1). */
  readonly stickers: readonly Rgb[];
  readonly stickerLook: { readonly roughness: number; readonly clearcoat: number; readonly spread: number };
  readonly body: { readonly colour: Rgb; readonly roughness: number; readonly clearcoat: number };
}

export const BACKDROPS: readonly BackdropKind[] = ['solid', 'gradient', 'wood', 'blobs', 'clutter', 'tiles'];
export const TABLE_KINDS: readonly BackdropKind[] = ['solid', 'solid', 'wood', 'wood', 'gradient', 'tiles', 'blobs'];
export const SKIN: readonly Rgb[] = [
  [0.96, 0.8, 0.69],
  [0.89, 0.69, 0.56],
  [0.78, 0.57, 0.44],
  [0.62, 0.43, 0.32],
  [0.45, 0.3, 0.22],
  [0.3, 0.2, 0.15],
];

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export function rgbToHsl([r, g, b]: readonly number[]): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h / 6, s, l];
}

export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const f = (p: number, q: number, t: number) => {
    t = ((t % 1) + 1) % 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [f(p, q, h + 1 / 3), f(p, q, h), f(p, q, h - 1 / 3)];
}

/** White with a warm or cool cast, as lamps and daylight are. */
export function lightColour(rng: Rng): [number, number, number] {
  const warm = [1.0, 0.8, 0.58];
  const cool = [0.76, 0.87, 1.0];
  const t = rng();
  const amount = Math.pow(rng(), 1.5) * 0.85;
  return [0, 1, 2].map((i) => 1 + (warm[i] + (cool[i] - warm[i]) * t - 1) * amount) as [number, number, number];
}

export function textureSpec(rng: Rng, kinds: readonly BackdropKind[] = BACKDROPS): TextureSpec {
  return { kind: pick(rng, kinds), seed: Math.floor(rng() * 2 ** 31), blur: chance(rng, 0.6) ? uni(rng, 0, 7) : 0, repeat: uni(rng, 1, 3) };
}

/** Rotate v about the unit axis k by `angle`. */
export function rotateAbout(v: Vec3, k: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return add(add(scale(v, c), scale(cross(k, v), s)), scale(k, dot(k, v) * (1 - c)));
}

/**
 * Sticker colours as this particular puzzle and print run has them: each nominal colour nudged in hue,
 * saturation and lightness; `white` (an index) may be faintly tinted instead.
 */
/**
 * Options of the whole page (datagen.html?…&layouts=0.5&shades=wide), set once by `configureRendering`.
 *
 * layouts  the share of puzzles stickered another way: a random colour layout (any of the 720), as real
 *          cubes are (scanner/core/colours.py). Labels are always the colours the puzzle shows.
 * shades   'wide': sticker shades vary more from cube to cube, reds as far as orange-red and salmon and
 *          oranges as far as red-orange (cameras bring the two close), always keeping red redder than
 *          orange on one cube.
 */
export const RENDER = { layouts: 0, wideShades: false };

export function configureRendering(params: URLSearchParams): void {
  RENDER.layouts = Math.min(1, Math.max(0, parseFloat(params.get('layouts') ?? '0') || 0));
  RENDER.wideShades = params.get('shades') === 'wide';
}

export const USUAL_PAINT: readonly number[] = [0, 1, 2, 3, 4, 5];

/**
 * The colour layout of the puzzle drawn for `seed`: paint[c] is the colour it shows where the model
 * shows colour c. The usual one, or with probability RENDER.layouts any permutation. It has its own
 * random stream, so everything else about the picture is the same as without it.
 */
export function colourLayout(seed: number): number[] {
  if (!RENDER.layouts) return USUAL_PAINT.slice();
  const rng = mulberry32((seed ^ 0x3c6ef372) >>> 0);
  if (rng() >= RENDER.layouts) return USUAL_PAINT.slice();
  const p = USUAL_PAINT.slice();
  for (let i = p.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  return p;
}

/** Labels in the model's colours -> the colours a puzzle with layout `paint` shows (empty etc. kept). */
export const painted = (values: ArrayLike<number>, paint: readonly number[]): number[] =>
  Array.from(values, (v) => (v >= 0 && v < paint.length ? paint[v] : v));

/** Hue (0-1) of red-like and orange-like nominal colours. */
const isRed = (h: number) => h > 0.94 || h < 0.02;
const isOrange = (h: number) => h >= 0.03 && h < 0.11;

export function stickerShades(rng: Rng, nominal: readonly string[], white: number): [number, number, number][] {
  if (RENDER.wideShades) return wideShades(rng, nominal, white);
  return nominal.map((hex, i) => {
    const [h, s, l] = rgbToHsl(hexToRgb(hex));
    if (i === white) return hslToRgb(rng(), uni(rng, 0, 0.14), uni(rng, 0.8, 0.97));
    return hslToRgb(h + clamp(gauss(rng) * 0.012, -0.03, 0.03), clamp(s * uni(rng, 0.68, 1.08), 0, 1), clamp(l * uni(rng, 0.78, 1.18), 0.05, 0.92));
  });
}

/** Sticker shades with the wide spread (RENDER.wideShades). */
function wideShades(rng: Rng, nominal: readonly string[], white: number): [number, number, number][] {
  const hsl = nominal.map((hex) => rgbToHsl(hexToRgb(hex)));
  const hue = hsl.map(([h], i) => {
    if (i === white) return rng();
    if (isRed(h)) return h + uni(rng, -0.015, 0.045);          // towards orange-red
    if (isOrange(h)) return h + uni(rng, -0.03, 0.025);         // towards red-orange, or yellower
    return h + clamp(gauss(rng) * 0.02, -0.05, 0.05);
  });
  // On one cube red stays redder than orange by at least 9° of hue (else even a person could not tell).
  const red = hsl.findIndex(([h], i) => i !== white && isRed(h));
  const orange = hsl.findIndex(([h], i) => i !== white && isOrange(h));
  if (red >= 0 && orange >= 0) {
    const r = hue[red] > 0.5 ? hue[red] - 1 : hue[red];
    if (hue[orange] - r < 0.025) hue[orange] = r + 0.025;
  }
  return hsl.map(([, s, l], i) => {
    if (i === white) return hslToRgb(hue[i], uni(rng, 0, 0.16), uni(rng, 0.78, 0.97));
    const h = ((hue[i] % 1) + 1) % 1;
    const light = isRed(hsl[i][0]) ? uni(rng, 0.8, 1.4) : uni(rng, 0.75, 1.28);
    return hslToRgb(h, clamp(s * uni(rng, 0.55, 1.08), 0, 1), clamp(l * light, 0.05, 0.92));
  });
}

export function puzzleLook(rng: Rng, nominal: readonly string[], white: number): PuzzleLook {
  const grey = uni(rng, 0.02, 0.1);
  return {
    stickers: stickerShades(rng, nominal, white),
    stickerLook: { roughness: uni(rng, 0.22, 0.75), clearcoat: chance(rng, 0.5) ? uni(rng, 0, 0.7) : 0, spread: uni(rng, 0, 0.035) },
    body: { colour: [grey * uni(rng, 0.9, 1.1), grey, grey * uni(rng, 0.9, 1.15)], roughness: uni(rng, 0.22, 0.65), clearcoat: uni(rng, 0, 0.6) },
  };
}

export interface Around {
  /** Unit vector from the puzzle towards the camera (before tilt). */
  readonly toward: Vec3;
  /** The camera's up, in the plane square to `toward` (the picture's roll is already in it). */
  readonly up: Vec3;
  /** Radius of a sphere round the centre that holds the whole puzzle (model units). */
  readonly extent: number;
  /** Half-width of what must be in the picture, model units (the cube is [-1, 1]^3). */
  readonly subject: number;
  /** Lowest point of the puzzle along each direction it may rest on, for the table: support(n) = max over the puzzle of -n·x. */
  readonly support: (n: Vec3) => number;
  /** Directions the puzzle may rest on a table on (its faces as they are now). */
  readonly restOn: readonly Vec3[];
  /** Directions round the side the hand may hold it from. */
  readonly sides: readonly Vec3[];
  readonly setting?: Setting;
  readonly toneMapping?: SceneSpec['toneMapping'];
  readonly skin?: Rgb;
  readonly maxTiltDeg?: number;
}

/**
 * A photo studio round one viewing direction: the camera roughly along `toward` (tilted a little, as a
 * person holding a phone manages), lights, what is behind and under the puzzle, and sometimes a hand.
 * Returns the spec and how the shot was framed.
 */
export function sceneAround(rng: Rng, a: Around): { scene: SceneSpec; framing: { tiltDeg: number; fill: number; distance: number } } {
  const n = normalize(a.toward);
  const camUp0 = normalize(a.up);
  const tiltDeg = Math.min(a.maxTiltDeg ?? 22, Math.abs(gauss(rng)) * 7.5);
  const tiltAz = uni(rng, 0, 2 * Math.PI);
  const t = Math.tan((tiltDeg * Math.PI) / 180);
  const camRight0 = cross(camUp0, n);
  const dir = normalize(add(n, add(scale(camRight0, t * Math.cos(tiltAz)), scale(camUp0, t * Math.sin(tiltAz)))));
  const camRight = normalize(cross(camUp0, dir));
  const camUp = normalize(cross(dir, camRight));
  const fov = uni(rng, 18, 66);
  const fill = uni(rng, 0.25, 0.86); // subject width as a fraction of the picture
  const halfVisible = a.subject / fill; // half-width of the picture at the subject, model units
  const distance = a.extent + halfVisible / Math.tan((fov * Math.PI) / 360);
  const slack = Math.max(0, halfVisible - a.subject * 1.12) * 0.92;
  const lookAt = add(scale(camRight, uni(rng, -slack, slack)), scale(camUp, uni(rng, -slack, slack)));
  const position = add(lookAt, scale(dir, distance));

  const lightDir = (minElevation: number): Vec3 => {
    const el = uni(rng, minElevation, Math.PI / 2);
    const az = uni(rng, 0, 2 * Math.PI);
    return normalize(add(scale(dir, Math.sin(el)), add(scale(camRight, Math.cos(el) * Math.cos(az)), scale(camUp, Math.cos(el) * Math.sin(az)))));
  };
  const key: LightSpec = { dir: lightDir(0.16), intensity: uni(rng, 0.5, 2.8), colour: lightColour(rng) };
  const fills: LightSpec[] = [0, 1].map(() => ({ dir: lightDir(-0.3), intensity: chance(rng, 0.7) ? uni(rng, 0.05, 1.2) : 0, colour: lightColour(rng) }));
  const sky = lightColour(rng);
  const groundTint = lightColour(rng);

  const setting: Setting = a.setting ?? pick(rng, ['floating', 'table-top', 'table-top', 'table-side', 'table-side', 'hand', 'hand', 'hand'] as const);
  const backdrop = {
    texture: textureSpec(rng, setting === 'table-top' ? TABLE_KINDS : BACKDROPS),
    // On a table seen from above the "backdrop" is the table itself, right under the puzzle.
    distance: setting === 'table-top' ? a.support(n) + uni(rng, 0.0, 0.38) : uni(rng, 3, 14) + a.extent,
    lit: setting === 'table-top' || chance(rng, 0.3),
    brightness: uni(rng, 0.25, 1.15),
  };
  let ground: SceneSpec['ground'] = null;
  if (setting === 'table-side') {
    // The puzzle rests on whichever of its faces is lowest in the picture.
    const down = a.restOn.reduce((best, d) => (dot(d, camUp) < dot(best, camUp) ? d : best));
    ground = { normal: scale(down, -1), offset: a.support(scale(down, -1)) + uni(rng, 0.0, 0.38), texture: textureSpec(rng, TABLE_KINDS), roughness: uni(rng, 0.35, 1) };
  }

  const fingers: FingerSpec[] = [];
  let palm: SceneSpec['palm'] = null;
  if (setting === 'hand') {
    // The hand comes in from one side of the picture (mostly from below) and grips the puzzle's sides,
    // fingers lying along them. Tips sometimes curl over the front.
    const wanted = pick(rng, [scale(camUp, -1), scale(camUp, -1), scale(camUp, -1), camRight, scale(camRight, -1), camUp]);
    const from = a.sides.reduce((best, d) => (dot(d, wanted) > dot(best, wanted) ? d : best));
    const grip = normalize(cross(n, from));
    const count = 1 + Math.floor(rng() * 4);
    const r = a.subject;
    for (let i = 0; i < count; i++) {
      const side = scale(grip, chance(rng, 0.5) ? 1 : -1);
      const radius = uni(rng, 0.2, 0.33);
      const length = uni(rng, 1.6, 3.8);
      const axis = normalize(add(scale(from, -1), add(scale(n, uni(rng, -0.45, 0.45)), scale(side, -uni(rng, 0, 0.22)))));
      const tip = add(add(scale(side, r + radius * uni(rng, 0.5, 1.1)), scale(from, -uni(rng, -0.7, 0.95) * r)), scale(n, uni(rng, -0.7, 0.95)));
      fingers.push({ centre: add(tip, scale(axis, -(length / 2 - radius))), axis, radius, length });
    }
    if (chance(rng, 0.45)) {
      palm = {
        centre: add(scale(n, -uni(rng, 1.7, 3.2)), add(scale(from, uni(rng, 0.2, 2.2) * r), scale(grip, uni(rng, -0.9, 0.9)))),
        scale: [uni(rng, 1.6, 3.2), uni(rng, 1.6, 3.2), uni(rng, 0.8, 1.6)],
      };
    }
  }

  // Exposure: a camera meters the scene, so brighter light does not mean a brighter picture.
  const lit = (l: LightSpec) => l.intensity * Math.max(0.12, dot(l.dir, n));
  const hemisphereIntensity = uni(rng, 0.05, 1.0);
  const environmentIntensity = uni(rng, 0.1, 1.2);
  const illumination = lit(key) + fills.reduce((s, l) => s + lit(l), 0) + 0.6 * hemisphereIntensity + 0.8 * environmentIntensity;
  const exposure = clamp((uni(rng, 0.62, 1.42) * 1.7) / illumination, 0.3, 3);

  return {
    scene: {
      camera: { position, lookAt, up: camUp, fov },
      setting,
      key,
      fills,
      shadow: { intensity: uni(rng, 0.35, 1), radius: uni(rng, 0.5, 6) },
      hemisphere: { sky, ground: groundTint.map((c) => c * uni(rng, 0.2, 0.8)) as [number, number, number], intensity: hemisphereIntensity },
      environment: { index: Math.floor(rng() * 1e6), rotation: uni(rng, 0, 2 * Math.PI), intensity: environmentIntensity },
      toneMapping: a.toneMapping ?? pick(rng, ['none', 'linear', 'neutral', 'neutral', 'aces', 'agx'] as const),
      exposure,
      backdrop,
      ground,
      fingers,
      skin: a.skin ?? (pick(rng, SKIN).map((c) => clamp(c * uni(rng, 0.88, 1.1), 0, 1)) as [number, number, number]),
      palm,
    },
    framing: { tiltDeg, fill, distance },
  };
}
