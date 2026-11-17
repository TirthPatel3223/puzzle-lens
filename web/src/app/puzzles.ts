import type { PuzzleUI } from './app';
import { MalteseUI } from './maltese/ui';
import { TangramUI } from './tangram/ui';

/**
 * The puzzles the app offers, in the switcher's order. Adding one: write its PuzzleUI in
 * src/app/<puzzle>/ui.ts (its photo protocol, Check and Solve screens) and list it here; the server side
 * is scanner/puzzles/<key>/ (see scanner/puzzles/__init__.py).
 */
export const PUZZLES = (): PuzzleUI[] => [new MalteseUI(), new TangramUI()];
