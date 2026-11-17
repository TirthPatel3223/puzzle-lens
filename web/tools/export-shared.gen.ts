/**
 * Writes the files both sides share, from the engines:
 *   shared/maltese/solver_map.json   the dictionary between the 3D engine and the solver's 144 array
 *   shared/maltese/layout.json       pieces, what they show, moves and rotations, for the Python side
 *   shared/tangram/layout.json       positions, cells, placements, moves and the two photo views
 *
 * Run: npm run export:shared        (after tools/export_solver_tables.py if the Maltese solver tables changed)
 * The tests then check that the committed files still match the engines.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { it } from 'vitest';
import { buildLayout } from '../src/maltese/layout';
import { type SolverTables, deriveSolverMap } from '../src/maltese/solverMap';
import { buildTangramLayout } from '../src/tangram/layout';

const SHARED = join(__dirname, '..', '..', 'shared');

it('writes shared/maltese', () => {
  const dir = join(SHARED, 'maltese');
  const tables = JSON.parse(readFileSync(join(dir, 'solver_tables.json'), 'utf8')) as SolverTables;
  const d = deriveSolverMap(tables);
  writeFileSync(join(dir, 'solver_map.json'), JSON.stringify(d.map) + '\n');
  writeFileSync(join(dir, 'layout.json'), JSON.stringify(buildLayout()) + '\n');
  const moves = Object.entries(d.map.moves)
    .filter(([k]) => !k.endsWith("'"))
    .map(([k, v]) => `${k} = ${v.face}${v.amount < 0 ? "'" : ''}`)
    .join(', ');
  console.log(`solver moves: ${moves}; gear rotation = ${d.map.gearSign > 0 ? '+' : '-'}quarter + offset`);
  console.log(`fitting relabellings: cells ${d.cellChoices.join(' x ')}, gears ${d.gearChoices}`);
});

it('writes shared/tangram', () => {
  const dir = join(SHARED, 'tangram');
  mkdirSync(dir, { recursive: true });
  const layout = buildTangramLayout();
  writeFileSync(join(dir, 'layout.json'), JSON.stringify(layout) + '\n');
  console.log(`tangram: ${layout.positions.length} positions, ${layout.cells.length} cells, ${layout.placements.length} placements`);
});
