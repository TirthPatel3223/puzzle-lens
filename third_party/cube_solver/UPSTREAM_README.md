# Cube Solving Pipeline

A solver for twisty and permutation puzzles that works from the move tables alone. It
splits the puzzle into a ladder of subgroups and solves one rung at a time: small rungs
with exact cost tables, large rungs with a value network trained by deep approximate value
iteration (the DeepCubeA / DeepXube method) on that rung's coset graph.

The first puzzle is Meffert's Maltese Gear Cube (about 4.5×10^58 visible states). The
approach is the one outlined in `docs/cayley_graphs_twisty_puzzles_and_hierarchical_rl_solvers.md`.
What was built, what was measured, and where the build departs from that outline are in
`docs/hierarchical_solver.md`.

The second is the DaYan Tangram Cube Extreme (4.75×10^28 states), whose move tables are not written
by hand: they are compiled from the puzzle's 3D model (`core/geometry.py`, `docs/tangram.md`). The
same route works for any puzzle whose moves are always legal.

Every solution is replayed on the reference simulator (the verified move tables) before it
is reported.

## Setup

```bash
pip install -e ".[test]"        # numpy, sympy, pytest
pip install torch               # the CUDA build for your driver; Colab has it already
pytest                          # about a minute; uses the GPU if there is one
```

## Solving the Maltese Gear Cube

```bash
# 1. Build the table rungs (corners, gear parity, gear rotation). About 3 minutes on a GPU.
python scripts/build_ladder.py --ladder learned --radius 7

# 2. Train the two value networks. Each run resumes from its checkpoint.
python scripts/train_rung.py --rung gear_col --minutes 45 --states 200000
python scripts/train_rung.py --rung ring --minutes 30

# 3. Solve.
python scripts/solve.py --scramble "U L F' U2 L'3 F"
python scripts/solve.py --random 50 --beam 4096
```

Everything a build produces is cached under `data/ladders/` (not versioned), so each step
resumes where it stopped. Moves are `U L F` and their inverses `U' L' F'`; the corner
opposite those three faces is held fixed, so the other three faces are not needed.

## Solving the Tangram Cube Extreme

```bash
# 1. The exact rung (turning corners and middle layer), and the macros of the two learned rungs.
python scripts/build_ladder.py --puzzle tangram_extreme --ladder learned --radius 7 --test 0

# 2. Train the two value networks.
python scripts/train_rung.py --puzzle tangram_extreme --rung large --minutes 45 --states 200000 --walk 40
python scripts/train_rung.py --puzzle tangram_extreme --rung small --minutes 15 --states 200000 --walk 20

# 3. Solve, in the simulator's notation (U, U2, U3, U2', U' are 60°-step turns of the top corner layer).
python scripts/solve.py --puzzle tangram_extreme --scramble "U2' F R U3 D' L"
python scripts/solve.py --puzzle tangram_extreme --random 50 --replay-json runs/tangram_solutions.json
```

The learned ladder solves random scrambles in about 91 steps (85 turns), 4.5 s each on an RTX 5070 Ti;
a worked solution is in `solutions/tangram_extreme_solution.md`. `--ladder table` instead builds an
all-table ladder that needs no training (about 4 minutes; about 138 steps). Every `--puzzle` defaults to `maltese`, so the commands above it are unchanged.

## How it works

1. **Labelled model** (`core/perm_model.py`). The move tables become permutations of 216
   labelled points, so identical-looking pieces can be told apart internally.
2. **Structure discovery** (`core/structure.py`). Orbits, pieces, orientation sub-blocks and
   the candidate rung features are computed from the permutations. For the Maltese cube it
   finds 7 corners × 3, 18 ring pieces × 4, and 24 gears with a colour and a rotation.
3. **Rung order** (`core/discover.py`). The note's ordering rules, measured: generator
   periods, the shortest macros each choice leaves behind, and bits resolved per macro length.
4. **Macro harvesting** (`core/harvest.py`). The moves of a rung must keep earlier rungs
   solved. They are found exhaustively by meet in the middle and as Schreier generators of
   the previous rung's table, then checked to generate the whole stabiliser.
5. **Rungs** (`core/quotient.py`, `core/coset_table.py`, `core/davi.py`). Each rung is a
   small puzzle on its own feature vector. It gets an exact Dijkstra table if its coset
   space fits in memory, otherwise a residual value network trained by value iteration.
6. **Search and solve** (`core/search.py`, `core/ladder.py`). Table rungs are a table walk;
   learned rungs use batched beam search on the GPU. The rung solutions are concatenated,
   adjacent turns are merged, and the result is replayed on the simulator.

## Layout

| Path | Contents |
|---|---|
| `core/gather_engine.py`, `core/torch_engine.py` | Reference simulator: a move is `next[i] = (state[idx[m,i]] + add[m,i]) mod modulus[i]` |
| `core/perm_model.py`, `core/structure.py`, `core/discover.py` | Labelled model, structure discovery, rung ordering |
| `core/harvest.py`, `core/quotient.py`, `core/coset_table.py`, `core/ladder.py` | Macros, quotient puzzles, exact tables, the ladder |
| `core/nnet.py`, `core/davi.py`, `core/search.py` | Value network, coset value iteration, beam search |
| `core/geometry.py` | Compiles a 3D model (stickers as frames in space, moves as rotations of regions) into a `GatherPuzzle` |
| `core/notation.py` | Per-puzzle notation and length metrics |
| `puzzles/registry.py` | Every puzzle the scripts know (`--puzzle`): tables, ladders, notation, cache paths |
| `puzzles/maltese_gear/` | Verified move tables and the ladders for the Maltese Gear Cube |
| `puzzles/tangram/` | The Tangram Cube Extreme and original: exported 3D geometry, and their ladders |
| `puzzles/toy/` | A 29,160-state puzzle used to check everything against brute force |
| `scripts/` | `build_ladder.py`, `train_rung.py`, `solve.py`, `analyse_puzzle.py`, `group_order.py`, `benchmark_deep_scrambles.py` |
| `tests/` | Unit and end-to-end tests |
| `docs/` | The design note and the implementation notes |
| `archive/` | The earlier attempts (single-network beam search), kept for reference |

## Adding a puzzle

1. **The move tables**, one of two ways:
   - *From a 3D model* (preferred). Export the model's stickers (centre, normal, colour), pieces
     (vertices) and moves (rotation matrix, the region `n·x > d` it turns) as `puzzle-geometry/1`
     JSON, with a few walks played on the model's own engine. `core/geometry.py` compiles it: it closes
     the sticker frames under the moves, turns each move into a permutation, proves that no move ever
     cuts a piece, and replays the walks. The Tangram exporter (`tools/export/geometry.gen.ts` in the
     Cube_Modeling project) is the example.
   - *By hand*, as a `GatherPuzzle` (index and addend tables plus the solved state), like the Maltese
     Gear Cube.
2. Add an entry to `puzzles/registry.py`: the builder, the ladders, and the notation and metrics.
3. Measure the structure and a proposed rung order:

   ```bash
   python scripts/analyse_puzzle.py --puzzle <name>
   ```

4. Turn the proposal into a list of `RungSpec` (see `puzzles/tangram/ladder.py`) and run the three
   steps above with `--puzzle <name>`. Rungs whose coset space fits in memory get exact tables; the
   large ones get value networks.

## Conventions

- Runs on Windows and Linux (Colab). No hardcoded paths.
- Long jobs cache and resume.
- DeepXube (MIT) is credited where its design is followed: `core/nnet.py`, `core/davi.py`.
