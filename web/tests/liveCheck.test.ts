import { describe, expect, it } from 'vitest';
import type { Checked } from '../src/app/api';
import { type FrameRect, HoldTimer, LIVE, judge, sameFaceAs } from '../src/app/liveCheck';

// A 16:9 frame with the guide square (76% of the height) in the middle, as the webcam view draws it.
const W = 960;
const H = 540;
const GUIDE: FrameRect = { x0: 0.5 - (0.38 * H) / W, y0: 0.12, x1: 0.5 + (0.38 * H) / W, y1: 0.88 };
const CELLS = [0, 0, 1, 2, 3, 4, 5, 0, 1, 1, 2, 3, 4, 5, 5, 2];

/** A face `side` pixels across, centred at (cx, cy) pixels. */
function frame(side = 360, cx = W / 2, cy = H / 2, extra: Partial<Checked> = {}, cellTurns = [CELLS, CELLS, CELLS, CELLS]): Checked {
  const s = side / 2;
  const quad = [
    [cx - s, cy - s],
    [cx + s, cy - s],
    [cx + s, cy + s],
    [cx - s, cy + s],
  ].map(([x, y]) => [x / W, y / H]);
  return { width: W, height: H, quad, confidence: 0.71, reading: { unsure: 0, minProb: 0.99, cellTurns }, ...extra };
}

describe('the live check', () => {
  it('a steady, centred, readable new face is good', () => {
    expect(judge(frame(), GUIDE, frame().quad, []).state).toBe('good');
    expect(judge(frame(), GUIDE, null, [undefined]).state).toBe('good');
  });

  it('no face, or one the locator is unsure of', () => {
    expect(judge(frame(360, W / 2, H / 2, { reading: null }), GUIDE, null, []).state).toBe('none');
    expect(judge(frame(360, W / 2, H / 2, { confidence: 0.2 }), GUIDE, null, []).state).toBe('none');
  });

  it('too small, off centre, or cut off by the picture edge', () => {
    expect(judge(frame(150), GUIDE, null, []).state).toBe('far');
    expect(judge(frame(360, W / 2 + 140), GUIDE, null, []).state).toBe('off-centre');
    expect(judge(frame(560), GUIDE, null, []).state).toBe('cut-off');
  });

  it('too many unsure stickers', () => {
    const c = frame(360, W / 2, H / 2, { reading: { unsure: LIVE.maxUnsure + 1, minProb: 0.4, cellTurns: [CELLS, CELLS, CELLS, CELLS] } });
    expect(judge(c, GUIDE, null, []).state).toBe('unsure');
  });

  it('moving between checks', () => {
    expect(judge(frame(), GUIDE, frame(360, W / 2 + 30).quad, []).state).toBe('moving');
    expect(judge(frame(), GUIDE, frame(360, W / 2 + 5).quad, []).state).toBe('good');
  });

  it('a face already taken, held any way up, even with a couple of misread stickers', () => {
    const turned = [...CELLS].reverse();
    const misread = CELLS.map((c, i) => (i < 3 ? (c + 1) % 6 : c));
    // The live face read turned: one of its four turns is the photo's cells.
    const v = judge(frame(360, W / 2, H / 2, {}, [turned, turned, misread, turned]), GUIDE, null, [undefined, CELLS]);
    expect(v.state).toBe('duplicate');
    expect(v.photo).toBe(2);
    // Four or more stickers different: another face.
    const other = CELLS.map((c, i) => (i < 4 ? (c + 1) % 6 : c));
    expect(sameFaceAs([other, other, other, other], [CELLS])).toBe(-1);
  });
});

describe('holding green', () => {
  it('counts from the first good check to the hold time', () => {
    const t = new HoldTimer();
    expect(t.update(true, 1000)).toBe(true);
    expect(t.update(true, 2500)).toBe(true);
    expect(t.progress(2500)).toBeCloseTo(1500 / LIVE.holdMs);
    expect(t.progress(1000 + LIVE.holdMs + 50)).toBe(1);
  });

  it('forgives one bad check, but not two', () => {
    const t = new HoldTimer();
    t.update(true, 0);
    expect(t.update(false, 250)).toBe(true);
    expect(t.update(true, 500)).toBe(true);
    expect(t.progress(1000)).toBeCloseTo(1000 / LIVE.holdMs);
    t.update(false, 750);
    expect(t.update(false, 1000)).toBe(false);
    expect(t.progress(1000)).toBe(0);
  });

  it('a bad check with nothing green yet stays red', () => {
    const t = new HoldTimer();
    expect(t.update(false, 0)).toBe(false);
    expect(t.green).toBe(false);
  });
});
