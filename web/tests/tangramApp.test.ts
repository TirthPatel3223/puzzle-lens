import { describe, expect, it } from 'vitest';
import layoutJson from '../../shared/tangram/layout.json';
import type { Checked } from '../src/app/api';
import { judge } from '../src/app/liveCheck';
import { LOOKALIKES, FACE_POSITIONS, describeMove, moveToken, parseMoves, practiceScramble, stateAfter, undoMoves } from '../src/app/tangram/model';
import { buildTangramLayout } from '../src/tangram/layout';
import { EMPTY, MODEL, TABLES } from '../src/tangram/model';
import { mulberry32 } from '../src/vendor/puzzle/rng';
import { PuzzleState } from '../src/vendor/tangram/state';

describe('shared/tangram/layout.json', () => {
  it('matches the engine (run npm run export:shared after changing the model)', () => {
    expect(JSON.parse(JSON.stringify(buildTangramLayout()))).toEqual(layoutJson);
  });
});

describe('moves in the simulator notation', () => {
  it('reads and writes them', () => {
    const moves = parseMoves("U2' F D3 R L U");
    expect(moves.map(moveToken)).toEqual(["U2'", 'F', 'D3', 'R', 'L', 'U']);
    expect(() => parseMoves('U X')).toThrow();
  });

  it('says them in words, naming the pieces that mark each part', () => {
    expect(describeMove({ axis: 'U', amount: 2 })).toContain('orange-yellow-green');
    expect(describeMove({ axis: 'U', amount: 2 })).toContain('120°');
    expect(describeMove({ axis: 'D', amount: -1 })).toContain('counterclockwise');
    expect(describeMove({ axis: 'F', amount: 1 })).toContain('orange-white');
  });

  it("a practice scramble's undo solves it", () => {
    const { moves, truth } = practiceScramble(mulberry32(5), 12);
    expect(truth.isSolved()).toBe(false);
    expect(stateAfter(undoMoves(moves), truth).isSolved()).toBe(true);
    expect(stateAfter(moves).equals(truth)).toBe(true);
  });
});

describe('the Check step', () => {
  it('look-alikes are the 12 pairs of positions that share a centre', () => {
    const pairs = LOOKALIKES.reduce((n, l) => n + l.length, 0) / 2;
    expect(pairs).toBe(12);
    // Never both occupied.
    for (let seed = 1; seed < 30; seed++) {
      const s = PuzzleState.solved(MODEL);
      s.applyAll(practiceScramble(mulberry32(seed), 20).moves);
      const occupied = new Set<number>();
      MODEL.pieces.forEach((_, i) => TABLES.placements[i].find((p) => p.ori === s.orient[i])!.positions.forEach((p) => occupied.add(p)));
      LOOKALIKES.forEach((l, p) => {
        if (occupied.has(p)) for (const q of l) expect(occupied.has(q)).toBe(false);
      });
    }
  });

  it('each face drawing shows its 29 positions inside the picture, back to front', () => {
    for (const f of FACE_POSITIONS) {
      expect(f.length).toBe(29);
      expect(f.filter((p) => p.flat).length).toBe(11);
      for (const p of f) expect(Math.max(Math.abs(p.x), Math.abs(p.y))).toBeLessThan(1.5);
      for (let i = 1; i < f.length; i++) expect(f[i].z).toBeGreaterThanOrEqual(f[i - 1].z);
    }
  });

  it('the live check knows a face it has already taken (29 positions, at least 26 the same)', () => {
    const labels = Array.from({ length: 29 }, (_, i) => (i % 3 === 0 ? EMPTY : i % 6));
    const check: Checked = {
      width: 960, height: 540, confidence: 0.7,
      quad: [[0.35, 0.2], [0.65, 0.2], [0.65, 0.8], [0.35, 0.8]],
      reading: { unsure: 0, minProb: 0.99, labels },
    };
    const guide = { x0: 0.29, y0: 0.12, x1: 0.71, y1: 0.88 };
    const sameAs = (c: Checked, taken: readonly unknown[]) =>
      taken.findIndex((t) => Array.isArray(t) && t.filter((v, i) => v === c.reading!.labels![i]).length >= 26);
    expect(judge(check, guide, null, [], { sameAs, maxUnsure: 6 }).state).toBe('good');
    const near = labels.slice();
    near[0] = 1;
    near[1] = 2;
    expect(judge(check, guide, null, [near], { sameAs, maxUnsure: 6 }).state).toBe('duplicate');
    expect(judge({ ...check, reading: { unsure: 7, minProb: 0.4, labels } }, guide, null, [], { sameAs, maxUnsure: 6 }).state).toBe('unsure');
  });
});
