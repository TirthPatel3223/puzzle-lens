import type { Checked } from './api';

/**
 * The webcam's live check: is the frame a good photo of a face, and how long has it been one?
 *
 * About four times a second the page sends a small copy of the frame to /api/vision/check, which
 * finds the face and reads it once. The guide turns green when the networks find one face, it fills
 * the square, it is steady, the reader is sure of nearly every sticker, and it is not a face that was
 * already taken. Held green for HOLD_MS, the page captures it.
 */

export const LIVE = {
  /** The locator must be at least this sure (a real face scores about 0.7, an empty picture 0.002). */
  minConfidence: 0.45,
  /** The face's side must be at least this fraction of the guide square's. */
  minFill: 0.5,
  /** The face's centre may be at most this fraction of the guide's side from the guide's centre. */
  maxOffCentre: 0.22,
  /** Corners closer than this to the picture's edge (fraction of the picture) may be cut off. */
  edgeMargin: 0.01,
  /** Between two checks, no corner may move more than this fraction of the face's side. */
  maxShift: 0.04,
  /**
   * Maltese Gear Cube: at most this many of the 20 stickers and gears may be under 90% sure. Real photos
   * that scanned to a legal cube had 0 to 4 (2026-11-14); a blurred or glary face has many more. Other
   * puzzles pass their own limit to `judge`.
   */
  maxUnsure: 4,
  /** A face showing at least this many of the same 16 cells as a photo already taken (any way up) is that face. */
  sameCells: 13,
  /** How long the outline must stay green before the photo is taken. */
  holdMs: 3000,
  /** One bad check in a run of good ones is forgiven (a single misread frame does not restart the count). */
  forgive: 1,
} as const;

/** A rectangle in fractions of the picture. */
export interface FrameRect {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

export type LiveState = 'none' | 'cut-off' | 'far' | 'off-centre' | 'duplicate' | 'unsure' | 'moving' | 'good';

export interface Verdict {
  readonly state: LiveState;
  readonly message: string;
  /** For 'duplicate': the photo (1-based) that already shows this face. */
  readonly photo?: number;
}

const dist = (a: readonly number[], b: readonly number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** The face's mean side in pixels of the checked picture. */
export function quadSide(quad: readonly (readonly number[])[], width: number, height: number): number {
  const px = quad.map(([x, y]) => [x * width, y * height]);
  return (dist(px[0], px[1]) + dist(px[1], px[2]) + dist(px[2], px[3]) + dist(px[3], px[0])) / 4;
}

/**
 * Maltese Gear Cube: the photo (index into `taken`) that shows the same face as `cellTurns`, held any way
 * up, or -1. `taken[i]` is that photo's cells as read (cellTurns[0] of its check), or undefined when unknown.
 */
export function sameFaceAs(cellTurns: readonly (readonly number[])[], taken: readonly unknown[]): number {
  for (let i = 0; i < taken.length; i++) {
    const t = taken[i] as readonly number[] | undefined;
    if (!t) continue;
    for (const cells of cellTurns) {
      let same = 0;
      for (let c = 0; c < 16; c++) if (cells[c] === t[c]) same++;
      if (same >= LIVE.sameCells) return i;
    }
  }
  return -1;
}

export interface JudgeOptions {
  /** The photo (index into `taken`) that shows the same face, or -1. Default: the Maltese cells test. */
  readonly sameAs?: (c: Checked, taken: readonly unknown[]) => number;
  /** At most this many unsure places (default LIVE.maxUnsure, for the Maltese Gear Cube's 20). */
  readonly maxUnsure?: number;
}

/**
 * What one checked frame says. `guide` is the guide square in fractions of the picture, `previous`
 * the face's corners at the check before (to tell a moving puzzle), `taken` what each photo taken showed.
 */
export function judge(
  c: Checked,
  guide: FrameRect,
  previous: readonly (readonly number[])[] | null,
  taken: readonly unknown[],
  options: JudgeOptions = {},
): Verdict {
  if (!c.reading || c.confidence < LIVE.minConfidence) return { state: 'none', message: 'Show one face in the square' };
  const { quad, width: w, height: h } = c;
  const m = LIVE.edgeMargin;
  if (quad.some(([x, y]) => x < m || y < m || x > 1 - m || y > 1 - m)) {
    return { state: 'cut-off', message: 'Move back: the whole face must be in view' };
  }
  const side = quadSide(quad, w, h);
  const guideSide = Math.min((guide.x1 - guide.x0) * w, (guide.y1 - guide.y0) * h);
  if (side < LIVE.minFill * guideSide) return { state: 'far', message: 'Closer: fill the square' };
  const cx = (quad.reduce((s, p) => s + p[0], 0) / 4) * w;
  const cy = (quad.reduce((s, p) => s + p[1], 0) / 4) * h;
  const gx = ((guide.x0 + guide.x1) / 2) * w;
  const gy = ((guide.y0 + guide.y1) / 2) * h;
  if (Math.hypot(cx - gx, cy - gy) > LIVE.maxOffCentre * guideSide) return { state: 'off-centre', message: 'Centre the face in the square' };
  const dup = options.sameAs ? options.sameAs(c, taken) : sameFaceAs(c.reading.cellTurns ?? [], taken);
  if (dup >= 0) return { state: 'duplicate', message: `Photo ${dup + 1} is this face: turn to another`, photo: dup + 1 };
  if (c.reading.unsure > (options.maxUnsure ?? LIVE.maxUnsure)) {
    return { state: 'unsure', message: 'Some stickers are hard to read: avoid glare' };
  }
  if (previous) {
    const shift = Math.max(...quad.map((p, i) => Math.hypot((p[0] - previous[i][0]) * w, (p[1] - previous[i][1]) * h)));
    if (shift > LIVE.maxShift * side) return { state: 'moving', message: 'Hold the cube steady' };
  }
  return { state: 'good', message: 'Hold still' };
}

/** How long the outline has been green (with one bad check forgiven), and whether it is green now. */
export class HoldTimer {
  private since: number | null = null;
  private misses = 0;

  /** Feed one check; true while the outline is green. */
  update(good: boolean, now: number): boolean {
    if (good) {
      this.misses = 0;
      if (this.since === null) this.since = now;
      return true;
    }
    this.misses++;
    if (this.since !== null && this.misses <= LIVE.forgive) return true;
    this.since = null;
    return false;
  }

  get green(): boolean {
    return this.since !== null;
  }

  /** 0 to 1: how much of the hold is done. */
  progress(now: number): number {
    return this.since === null ? 0 : Math.min(1, (now - this.since) / LIVE.holdMs);
  }

  reset(): void {
    this.since = null;
    this.misses = 0;
  }
}
