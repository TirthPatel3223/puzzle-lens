/**
 * The colour layout of the cube in the person's hands (scanner/core/colours.py on the server).
 *
 * Cubes are not all stickered alike: the same six colours can sit on a puzzle another way (a Tangram
 * Cube Extreme whose turning corners are red-yellow-blue and orange-white-green instead of the model's
 * orange-yellow-green and red-white-blue), in a mirror image, or with other colours opposite each other.
 * One permutation says it all:
 *
 *     paint[c] = the colour this cube shows where the model shows colour c
 *
 * Everything the app keeps (observations, states, the solver's input) stays in the model's colours. Only
 * what is shown goes through the palette: a model colour c is drawn as hex[paint[c]] and named
 * names[paint[c]], and colour words in instructions ("the orange-yellow-green corner") are said in the
 * cube's colours. A scan tells the page its cube's layout; anything else (typed moves, a practice
 * scramble before its photos) uses the model's own colours.
 */

/** A colour layout as the server reports it. */
export interface ColoursInfo {
  readonly paint: readonly number[];
  readonly kind: 'default' | 'rearranged' | 'mirrored' | 'other';
  readonly usual: boolean;
  /** The colours that differ: where the model has `model`, the cube has `cube`. */
  readonly changed: readonly { model: string; cube: string }[];
  /** For an alternative: how much less likely than the scan's answer (nats). */
  readonly nats?: number;
}

export const USUAL_PAINT: readonly number[] = [0, 1, 2, 3, 4, 5];

/** Something with a colour that can be set from a hex string (a three.js material's `color`). */
export interface Colourable {
  readonly color: { getHexString(): string; set(hex: string): unknown };
}

export class Palette {
  private paintNow: readonly number[] = USUAL_PAINT;
  private readonly listeners = new Set<() => void>();
  private readonly word: RegExp;

  /**
   * @param hex   the model's colours, by colour index
   * @param names their names (lower case)
   * @param ink   a text colour that reads on each colour
   */
  constructor(
    readonly hex: readonly string[],
    readonly names: readonly string[],
    readonly ink: readonly string[] = [],
  ) {
    this.word = new RegExp(`\\b(${names.join('|')})\\b`, 'gi');
  }

  /** The layout now shown: paint[c] for each model colour c. */
  get paint(): readonly number[] {
    return this.paintNow;
  }

  /** Is the cube shown in the model's own colours? */
  get usual(): boolean {
    return this.paintNow.every((p, c) => p === c);
  }

  /** Show the cube with layout `paint` (null or undefined: the model's own colours). */
  set(paint: readonly number[] | null | undefined): void {
    const next = paint && isLayout(paint) ? paint.slice() : USUAL_PAINT;
    if (next.every((p, c) => p === this.paintNow[c])) return;
    this.paintNow = next;
    for (const f of this.listeners) f();
  }

  /** Called whenever the layout changes. Returns a function that stops it. */
  onChange(f: () => void): () => void {
    this.listeners.add(f);
    return () => this.listeners.delete(f);
  }

  /** The colour drawn for model colour c (anything outside 0-5 is returned unchanged). */
  shown(c: number): number {
    return c >= 0 && c < this.paintNow.length ? this.paintNow[c] : c;
  }

  hexOf(c: number): string {
    return this.hex[this.shown(c)] ?? '#888888';
  }

  nameOf(c: number): string {
    return this.names[this.shown(c)] ?? '?';
  }

  inkOf(c: number): string {
    return this.ink[this.shown(c)] ?? '#1a202c';
  }

  /** The model colour shown as the cube's colour k (what a person picking colour k means). */
  modelOf(k: number): number {
    const i = this.paintNow.indexOf(k);
    return i < 0 ? k : i;
  }

  /** A text in the cube's colours: every colour word (a whole word, any case) is said as the cube shows it. */
  say(text: string): string {
    if (this.usual) return text;
    return text.replace(this.word, (w) => {
      const c = this.names.indexOf(w.toLowerCase());
      const out = this.nameOf(c);
      return w[0] === w[0].toUpperCase() ? out[0].toUpperCase() + out.slice(1) : out;
    });
  }

  /**
   * Recolour sticker materials that were built in the model's colours. The model colour of each is
   * worked out the first time it is seen (from the colour it was built with) and remembered.
   */
  repaint(materials: Iterable<Colourable>, memo: WeakMap<object, number>): void {
    const own = this.hex.map((x) => x.replace('#', '').toLowerCase());
    for (const m of materials) {
      let c = memo.get(m);
      if (c === undefined) {
        c = own.indexOf(m.color.getHexString().toLowerCase());
        if (c < 0) continue;
        memo.set(m, c);
      }
      m.color.set(this.hexOf(c));
    }
  }
}

/** Six colours, each once. */
export function isLayout(paint: readonly number[]): boolean {
  return paint.length === 6 && [...paint].sort((a, b) => a - b).every((p, i) => p === i);
}

/** "orange and red swapped, green and blue swapped", from what the server says changed. */
export function describeLayout(info: ColoursInfo): string {
  if (info.usual) return 'the usual colours';
  const seen = new Set<string>();
  const parts: string[] = [];
  const to = new Map(info.changed.map((x) => [x.model, x.cube]));
  for (const { model } of info.changed) {
    if (seen.has(model)) continue;
    const cycle: string[] = [];
    let c: string | undefined = model;
    while (c !== undefined && !seen.has(c)) {
      seen.add(c);
      cycle.push(c);
      c = to.get(c);
    }
    if (cycle.length === 2) parts.push(`${cycle[0]} and ${cycle[1]} swapped`);
    else parts.push(cycle.map((m) => `${to.get(m)} where the usual has ${m}`).join(', '));
  }
  return parts.join('; ');
}
