# The Tangram Cube Extreme: from the 3D model to a solver

This note records how the DaYan Tangram Cube Extreme was brought into the pipeline. The Maltese Gear
Cube's move tables were written by hand. For this puzzle they come from its 3D model, through a compiler
that works for any puzzle whose moves are always legal. Then come the ladder that solves it, and what
was measured on the way.

**Legend.** `[measured]` = computed in this project; the test or script that reproduces it is named.

**Result.** The learned ladder (one exact table, then two value networks) solves random scrambles in
a mean of 90.8 steps (85.4 turns). The all-table ladder needs 138.3 steps on the same scrambles.
Every solution reaches the solved state on the compiled tables. The learned ladder's 50 solutions
and the worked one in `solutions/tangram_extreme_solution.md` were also replayed on the simulator's
own 3D engine, and all 51 solve the puzzle. `[measured: section 3]`

---

## 1. From the 3D model to move tables

The puzzle's simulator (the Cube_Modeling project, `src/puzzle/*.ts`) describes it geometrically. Each
piece is a convex cell of the cube cut by planes. Each move turns the pieces whose centroid lies in a
half-space `n·x > d` by a rotation: 60° about the body diagonal for the corner layers `U` and `D`,
180° about an edge axis for the flips `F`, `R`, `L`. `tools/export/geometry.gen.ts` there writes the
pieces, stickers and moves to JSON (`puzzles/tangram/geometry_extreme.json`). It also writes 20 random
walks played on the simulator's own engine, with every piece's final orientation.

`core/geometry.py` compiles that file:

1. **Positions.** A sticker is a frame in space: its centre and its outward normal. Every move is
   applied to every frame inside its region, until no new frame appears. The result is the set of
   positions a sticker can ever occupy: 120 on the Extreme. 66 are on the cube; 54 are where stickers
   go when a 60° turn changes the puzzle's shape. `[measured: tests/test_geometry.py]`
2. **Moves.** Each move is a permutation of the 120 positions, hence a gather table. The state is the
   colour at every position, or `EMPTY` (6) where no sticker is. This is an ordinary `GatherPuzzle`,
   and nothing geometric is used after this point.
3. **Checks.** The compiler refuses the model unless every one of these holds:
   - every reachable pose of every piece lies on one side of every move's boundary, so a move never
     cuts a piece (it is doctrinaire: every move is legal in every state);
   - no position lies within 0.09 of a boundary (cube half-width 1);
   - distinct positions are at least 0.25 apart, against a matching tolerance of 1e-7;
   - each move has its stated order and is undone by its inverse;
   - the tables reproduce the simulator's 20 walks, and 50 walks of a piece-pose engine that moves whole
     pieces, as the simulator does.

   `tests/test_geometry.py` also checks that it refuses a model whose flip cuts through pieces, and
   tables that disagree with the model.

**A sticker's centre alone is not a position.** Twelve pairs of positions share a centre and differ
only in their normal: a sticker turned 60° lands exactly where another kind of piece has a sticker,
tilted. With centres alone a move would not be a permutation. This holds for any cut fraction from
0.25 to 0.32, so it comes from the design (the cuts meet on the cube's edges), not from the measured
value 0.31. The original Tangram Cube, which never changes shape, has no such pairs.
`[measured: tests/test_geometry.py::test_a_sticker_centre_alone_is_not_a_position]`

| | Extreme | Original |
|---|---|---|
| Pieces / stickers | 45 / 66 | 21 / 42 |
| Positions (occupied when solved) | 120 (66) | 42 (42) |
| Piece orientations | 12 (D6) | 6 (D3) |
| Group order | 47,478,243,018,446,808,809,472,000,000 | 111,974,400 |
| Matches | Cube_Modeling's notes, "about 4.75 × 10²⁸" | Jaap Scherphuis: 3,110,400 × 36 |
| Compile time | 0.1 s | 0.05 s |

The empty positions add no hidden states: the elements that fix every occupied position fix
everything. The core never moves, so the state is in a fixed frame. No turned copy of the solved state
is reachable with the core held still, so the goal is unique. `[measured: the group order, by sympy, in
tests/test_geometry.py]`

## 2. Structure

`core/structure.py` runs unchanged on the compiled tables and finds 14 orbits.
`puzzles/tangram/ladder.py` names them, and `tests/test_tangram.py` checks each name:

| Orbit | Pieces | Positions | Notes |
|---|---|---|---|
| `SMALL_TRI_A`, `SMALL_TRI_B` | 2 × 6 | 12 each | small triangular sticker |
| `LARGE_TRI_A`, `LARGE_TRI_B` | 2 × 6 | 12 each | larger triangular sticker |
| `PENTAGON` | 6 | 12 | pentagonal sticker |
| `CORNER`, `CORNER_2`, `CORNER_3` | 6 | 12 each | the six cube corners off the turning axis; one orbit per sticker |
| `TOP`, `BOTTOM` | 1 each | 6 each | the turning corners: they only spin in place |
| `MID_EDGE_F`, `_R`, `_L` | 1 each | 2 each | middle-layer edges on the flip axes: they only flip in place |
| `MIDDLE` | 3 | 6 | the other middle-layer pieces, permuted by the flips |

Each 12-position orbit holds 6 pieces in 12 places (6 on the cube, 6 shifted) and reaches all
12!/6! = 665,280 arrangements. `[measured: BFS on each orbit's feature]`

Facts the ladder rests on, all `[measured: tests/test_tangram.py]`:

- **A turning corner can be turned 120° with every other piece solved** (3 states), and **two middle
  edges can be flipped together** (4 states), although one middle edge alone cannot. No word of up to
  12 moves does either once the other pieces are solved.
- A cube corner's first sticker determines the other two, so a rung needs only `CORNER`.
- The cube shape alone is a large problem: more than 40 million shapes by depth 11, still growing
  about 4.5× per level. So no rung restores the shape. It comes back as the pieces are placed.
  `[measured: BFS on the occupancy pattern]`

### A correction to `core/discover.py`

Run on these tables, the rung-order proposal placed the middle trio and the six classes and stopped.
It treated the turning corners and middle edges as solved because no allowed word of up to 12 moves
changed them. Its ladder would have left a corner turned 120° in some solves. The proposal now ends
with an exact test: does fixing the chosen features force each orbit to its goal (a pointwise
stabiliser in sympy, `forced`)? Orbits that are not forced go into a first rung, solved while the
primitive moves are still allowed. `linked_orbits` also no longer groups single-piece orbits: such an
orbit is permuted trivially, so "moves like" any other.

On the original Tangram Cube (radius 4, 72 s), the greedy loop places seven orbits and leaves out the
bottom turning corner. The new check finds it unsolved and puts it first, which completes the ladder.
`[measured: propose_ladder on tangram_original]`

## 3. The ladders

Both ladders start with the same exact rung, `frame`: the turning corners, the middle trio and the
middle edges. It has 3,456 cosets and is solved with the primitive moves in at most 13 steps. The
features of each ladder together force every orbit, so solving the rungs solves the puzzle.
`[measured: tests/test_tangram.py::test_each_ladders_features_solve_the_whole_puzzle]`

### All-table baseline (`TABLE_LADDER`)

One exact table per piece class, in the order `core/discover.py` measured: the small triangles have
the cheapest pure macros, so they go last. `[measured: scripts/build_ladder.py --puzzle tangram_extreme
--ladder table --radius 7 --test 100; built in 215 s]`

| # | Rung | Cosets | Macros (steps) | Mean steps |
|---|---|---|---|---|
| 0 | frame | 3,456 | 7 primitive moves (1) | 7.5 |
| 1 | large_tri_a | 665,280 | 884 (4–8) | 19.5 |
| 2 | large_tri_b | 360 | 359 (6–14) | 9.8 |
| 3 | pentagons | 665,280 | 844 (10–14) | 34.6 |
| 4 | corners | 360 | 338 (12–84) | 25.6 |
| 5 | small_tri_a | 665,280 | 736 (10–12) | 27.8 |
| 6 | small_tri_b | 360 | 359 (12–24) | 16.8 |
| | total after merging turns | 4.75 × 10²⁸ (product = \|G\|) | | **137.2** (min 112, max 158) |

100 of 100 random scrambles solved and replayed on the reference simulator.

The cost grows down the ladder: each rung must keep the earlier ones solved, so its macros get longer
(the corner rung needs words of up to 84 steps).

### Learned ladder (`LEARNED_LADDER`)

The six classes in two rungs with value networks (`core/davi.py`), so most of the work is done with the
cheap macros.

| # | Rung | Cosets | Heuristic | Macros (steps) |
|---|---|---|---|---|
| 0 | frame | 3,456 | exact table | 7 primitive moves (1) |
| 1 | large: large triangles, pentagons, cube corners | 5.7 × 10¹⁶ | value network | 163 (4–6) |
| 2 | small: the small triangles | 2.4 × 10⁸ | value network | 192 (10) |

The coset counts are the orders of the groups the macros generate on the rung's pieces. They match the
exact stabiliser indices computed with sympy, and the three multiply to |G|. The `small` rung would fit
in memory as a table, but its feature vector does not fit the 63-bit key that tables use (two 12-slot
orbits of 7 values each), so it is learned.

**Training** (`scripts/train_rung.py`, one RTX 5070 Ti; commands in section 4). Each network is the
fully connected residual network of `core/nnet.py` (1,000 units, 4 residual blocks). Each update
draws 200,000 states by random walks of the rung's macros from its goal, and computes their targets
with the target network over every macro. That takes 30 s of each 34 s update, so most of the training
time goes into making data.

| Rung | Network | Training | Walks | Test: beam 4096, earlier rungs solved |
|---|---|---|---|---|
| large | 8.36 M parameters; 48 slots × 7 classes | 45 min, 77 updates | up to 40 macros | 50/50 solved, mean 47.4 steps, 3.9 s each |
| small | 8.20 M parameters; 24 slots × 7 classes | 15 min, 27 updates | up to 20 macros | 20/20 solved, mean 41.0 steps, 0.7 s each |

Followed greedily, without search, the networks solve only 13 % (large) and 25 % (small) of the
training monitor's test states. The beam search does the rest. `[measured: runs/tangram_train_*.log]`

**The whole ladder.** On 50 random scrambles of 200 steps, against the table ladder on the same
scrambles `[measured: scripts/solve.py --puzzle tangram_extreme --random 50, beam 4096, with and
without --ladder table]`:

| | Learned ladder | Table ladder |
|---|---|---|
| Solved and replayed | 50 / 50 | 50 / 50 |
| Steps: mean (sd), min–max | **90.8** (5.2), 82–104 | 138.3 (10.1), 112–156 |
| Turns: mean (sd), min–max | **85.4** (4.9), 75–97 | 126.7 (9.4), 105–144 |
| Steps per rung, before merging | frame 7.7, large 47.4, small 40.4 | 7.7, then 19.8 + 10.0 + 34.8 + 25.6, then 27.6 + 17.3 |
| Time per solve | 4.5 s (max 5.7 s) | under 0.05 s |

Both ladders solve the frame the same way, so the large pieces start from the same states: the
network spends 47.4 steps on them where four tables spend 90.2. On the small triangles the gain is
smaller (40.4 against 44.9).

The final word is merged across rungs, and across moves that commute (`WordAlgebra.simplify`). The two
corner layers commute, so `D' U' D` is `U'`. Merging only adjacent turns of one layer misses this. The
commuting merge saves 0.6 steps and 1.1 turns per solve. No two faces of the Maltese Gear Cube commute,
so its solutions do not change. `[measured: tests/test_tangram.py::test_turns_merge_across_the_commuting_corner_layers]`

`solutions/tangram_extreme_solution.md` solves a scramble made the way the simulator's Scramble button
makes them (20 rounds of `U^a D^b` and a flip) in 84 steps (80 turns). Cube_Modeling's
`tools/export/replay.gen.ts` replays that solution and the 50 random ones on the simulator's own 3D
engine. All 51 solve the puzzle.

## 4. Reproducing

```bash
# in Cube_Modeling, only after a change to the 3D model: re-export, then copy the two files
npx vitest run --config tools/export/vitest.config.ts       # writes tools/export/out/tangram-*.json

# here
pytest tests/test_geometry.py tests/test_tangram.py
python scripts/group_order.py --puzzle tangram_extreme
python scripts/analyse_puzzle.py --puzzle tangram_extreme --radius 6
python scripts/build_ladder.py --puzzle tangram_extreme --ladder table --test 100
python scripts/build_ladder.py --puzzle tangram_extreme --ladder learned --test 0
python scripts/train_rung.py --puzzle tangram_extreme --rung large --minutes 45 --states 200000 --walk 40
python scripts/train_rung.py --puzzle tangram_extreme --rung small --minutes 15 --states 200000 --walk 20
python scripts/solve.py --puzzle tangram_extreme --random 50 --replay-json runs/tangram_solutions.json

# in Cube_Modeling: replay those solutions on the 3D model's own engine
SOLUTIONS=<path>/runs/tangram_solutions.json npx vitest run --config tools/export/vitest.config.ts
```

## 5. Other puzzles

- Any puzzle whose moves rotate half-space regions and are always legal: write an exporter, then
  compile. No new pipeline code is needed.
- Curved cuts (the Curvy Copter's cones) need another kind of region in the JSON. With 180° turns only,
  the compile is otherwise the same.
- Geared moves (the Maltese Gear Cube: face layer 90°, middle ring 45°, each gear spinning about its
  own axis) need moves made of several (region, rotation) parts.
- Jumbling (the Curvy Copter's partial turns, blocked by bandaging) is different. The compiler refuses
  such a model, because a turn can cut pieces. The positions can still be listed, and "this turn is
  blocked" becomes "these positions are occupied". But the moves are no longer permutations, so the
  subgroup ladder does not apply as it stands; the solver needs moves that depend on the state.
