# Cube Scanner

Take six photos of a scrambled twisty puzzle. The scanner reads every sticker, rebuilds the puzzle on its exact 3D model so you can check it, hands the state to the puzzle's solver, and plays the solution back move by move.

| Puzzle | Photos | Vision | Solver |
|---|---|---|---|
| **Maltese Gear Cube** | six faces | trained (see [Accuracy](#accuracy-2026-11-14)) | connected (ladder solver) |
| **Tangram Cube Extreme** | six faces, straight on, **in any shape** | code ready, datasets rendered; [train it](#training-the-tangram-networks) | connected ([ladder solver](#the-tangram-solver): about 90 steps, about 5 s) |

```
6 photos ─▶ find each face ─▶ read every place on it ─▶ fit the 6 photos together ─▶ snap to a legal state
         ─▶ check it on the 3D model ─▶ solver ─▶ follow the solution in 3D
```

Everything runs locally: a small Python server (vision networks, decoders, solvers) and a browser app in the same look as the 3D simulator site. Pick the puzzle in the panel's switcher (or open `?puzzle=tangram-cube-extreme`).

---

## Quick start

**Once:**

```bash
# Python 3.10+; install PyTorch with CUDA for your driver first (https://pytorch.org)
pip install -e ".[test]"
pip install pillow-heif          # iPhone photos (HEIC)

cd web
npm install
npm run build                    # the app the server shows
cd ..
```

The trained networks (Maltese: `data/checkpoints/locator.pt`, `reader.pt`; Tangram: `data/checkpoints/tangram/`) and the Maltese solver's ladder cache (`data/ladders/learned_r7`) must be in `data/`. See [Data](#data-not-in-git).

**Every time:**

```bash
python -m scanner serve --open   # http://127.0.0.1:8000
```

The first scan loads the networks (a few seconds), and the first solve loads the solver. Add `--warm` to load the solver at start-up.

---

## Using the app (Maltese Gear Cube)

The Tangram Cube Extreme's steps are the same; what differs is in [its own section](#tangram-cube-extreme).

### 1. Photos

Take **one photo of each of the six faces**. Upload them, or use the webcam.

- **Any order, any way up.** The scanner works out which photo shows which face, and how each is turned, from the corner pieces. Every corner shows on three photos, and only one arrangement makes all eight corners real pieces.
- **Square-on, filling most of the picture**, with the face's edges in view. The cube can sit on a table or be held, but keep fingers off the face's edges.
- **Even light.** Avoid strong glare on the stickers.
- Each upload shows the face the scanner found, outlined in blue.
- Webcam mode has a square guide, a focus meter, and <kbd>Space</kbd> to capture. The live picture is mirrored for a camera facing you (move the cube left and it moves left on screen); **Mirror** flips it. The photos themselves are never mirrored.
- **Auto-capture** (webcam): about four times a second the networks check the picture (`/api/vision/check`: find the face, read it once; nothing is kept). The guide turns **green** when they find one face that fills the square, holds steady, is read surely (at most 4 of its 20 stickers and gears under 90%), and is not a face already taken. It turns **red**, with the reason, otherwise. Held green for 3 seconds, the photo is taken by itself. The same face is recognised however it is turned, so turn the cube to the next face. The thresholds are in `web/src/app/liveCheck.ts`.
- The side panel is on the left. Drag the bar beside it to make it wider or narrower (double-click the bar to reset).

**Other ways in:**

- **Practice with a known scramble.** The app gives you a scramble. Do it from solved and take the photos. The app then knows the exact state, so it can show how well the scan did, and the photos can be kept as *labelled real data* for measuring and fine-tuning the model.
- **Paste a 144 array.** Check or solve an array you already have.

### 2. Check

Each face is shown as a card: the photo, flattened and turned upright, with a **ring on every sticker and gear** in the colour the scanner decided on.

| Mark | Meaning |
|---|---|
| plain ring | the decoded colour |
| dashed amber circle | the scanner was unsure (under 90%): look at it |
| amber dot | the decoder overruled the camera, to make the cube legal |
| blue dot | you changed it |
| red circle | part of a problem that makes the cube impossible |
| white line on a gear | where its long teeth point |

**Select any ring to fix it.** The editor shows a zoomed crop of the photo, what the camera read and how sure it was, the six colours (keys <kbd>1</kbd> to <kbd>6</kbd>), and for gears the four tooth directions.

- **3D model:** rebuilt from the faces on every change. The numbers on it are the photos. *Corner*, *Face-on* and *Edge* are the 3D simulator site's views (photo 1 is the front, its top edge up), and *Photo N* turns it to that face.
- **Legality check:** every piece must appear once, the corner twists and gear turns must add up, and the centre/edge permutation must be even. Problems are explained in words, and their stickers are ringed in red.
- **When it is legal:**
  - **Solve this cube**;
  - copy the 144 array;
  - save the scan as JSON;
  - keep the photos as labelled data.

### 3. Solve

**How to hold the cube:** blue on top, yellow in front, purple on the left. The white-green-red corner is then at the back, bottom, right, and it never moves.

The app also tells you which of your photos goes where.

- **Turn the model:** *Corner* shows the blue-yellow-red corner with blue on top; *Face-on* shows the yellow face with blue on top.
- **Solution:** the solver finds a verified solution, usually in 10 to 20 seconds. It is checked again on the 3D model.
- **Follow along:** each move is shown large, in the solver's letters and in words (*"Left (purple) face: 3 quarter turns counterclockwise"*). It is animated on the model, and the layer that turns next is lit up.
- **Moves:** U, L and F are the top, left and front faces. ′ means counterclockwise, and a number repeats the quarter turn: `L'3` is three counterclockwise quarter turns.
- **Keys:** <kbd>→</kbd> or <kbd>Space</kbd> next, <kbd>←</kbd> back, <kbd>P</kbd> play/pause, <kbd>Home</kbd>/<kbd>End</kbd>.
- **Extras:** a speed slider, a clickable move list (four to a line), copy, and download as `.txt`.

---

## How it works (Maltese Gear Cube)

| Stage | What | Where |
|---|---|---|
| Find the face | `FaceLocator`: a heatmap network for the four face corners. It looks once at the whole photo, then again close up. | `scanner/puzzles/maltese/vision/models.py`, `pipeline.py` |
| Flatten and cut | A homography from the corners, then 13 crops per face (4 stars, 4 gears, 4 edge halves, the centre). Each crop is turned to one canonical pose. | `scanner/puzzles/maltese/vision/geometry.py` |
| Read | `SlotReader`: one ResNet trunk with a head per slot kind. It gives the colour of every cell, each gear's colour and quarter turn, and a probability for each. It reads 5 slightly nudged cut-outs per face and averages them. | `models.py` |
| Fit the photos together | Every photo can be any face, turned 4 ways (120 × 1024 arrangements). A quick score uses the corner and edge slots; the 24 best are then decoded in full, and the most likely decoded cube wins. | `pipeline.py` `find_arrangement` |
| Decode | The most likely **legal** cube under those probabilities. Pieces go into slots by Hungarian assignment, then the invariants are repaired at the cheapest cost. A repair that would overrule a confident reading is flagged instead. | `scanner/puzzles/maltese/decode.py` |
| 144 array | The decoded cube is turned to the solver's frame and relabelled with the dictionary between engine and solver, which was derived from the move tables. | `shared/maltese/solver_map.json` |
| Solve | The copied ladder solver; every solution is replayed on its own simulator. | `scanner/puzzles/maltese/solver.py`, `third_party/cube_solver` |

**The 3D engine is the ground truth.** Every training label comes from it.

- **Training pictures:** synthetic, drawn by the real 3D model in Chrome with randomised light, camera, sticker shades, backgrounds and hands (`web/src/datagen/maltese`, `tools/gen_dataset.py`).
- **The arrangement rules:** the decoder's invariants are proven from the order of the move group (`tools/derive_invariants.py`): a visible arrangement can be reached exactly when they hold.

### Accuracy (2026-11-14)

| Test | Result |
|---|---|
| Synthetic, single faces (4,000), corners found by the network | 94.3% of faces read with no mistake; edge stickers 99.1%, stars 99.9%, gear turns 99.97% |
| Synthetic, whole scans (400), photos shuffled | **99.5% exactly right** after decoding, 0.23 s per scan |
| Synthetic, whole scans (400), fixed order | 98.75% exactly right (69.5% before decoding) |
| First real scan (6 iPhone photos, any order) | Every sticker I could check against the photos agrees. 3 places were fixed by the decoder, and the solver found a 299-move solution. Not yet confirmed by solving the physical cube. |

Reports are in `data/reports/`.

### Frames and conventions

- **Colours** are indices 0-5: white, yellow, green, blue, red, purple. These are the 3D model's U D F B L R faces, with white up and green in front. **The solver's codes differ:** 0 White, 1 Yellow, 2 Blue, 3 Green, 4 Purple, 5 Red. The dictionary handles that.
- **A scan is described as it was photographed:** photo 1 is its F face, with that photo's top edge up. The decoder turns it to the **solver's frame**: white-green-red corner home, blue up, yellow in front, purple on the left. The solver's U, L, F are the model's B, R, D.
- **A face is a 4×4 grid of cells** (rows down, columns right, seen from outside) plus 4 gears. See `web/src/maltese/observation.ts`.

---

## Tangram Cube Extreme

The Tangram Cube Extreme changes shape: a 60° corner turn tilts pieces out of the cube outline. The scanner reads it in any shape.

### Taking the photos: six faces, straight on (decided before training)

A face is a side of the **middle layer**, which never changes shape. Every face always shows some flat stickers to square the camera to. Every face also has one **turning corner** at one of its corners: the three-coloured corner piece (orange-yellow-green or red-white-blue), which only ever twists in place.

- **One photo of each face, straight on**, whatever shape the puzzle is in. Pieces that stick out are fine.
- **Any order, any way up.** The locator finds the face's corners *starting from its turning corner*, so it knows which way up each photo is. The decoder works out which face each photo shows.
- **Not corner-on.** Looking straight at a turning corner, a sticker a 60° turn has tilted out of the cube looks exactly like the flat sticker in its place.

Measured on the 3D model (`web/tests/tangramModel.test.ts` checks the geometry):

| Photos | Look-alike pairs told apart | Pieces hidden from every photo (60 scrambles) |
|---|---|---|
| 2, straight at the turning corners | 2 of 12 (10 pairs overlap 98-100%) | 39 of 60 scrambles had one |
| **6, face-on** | **12 of 12** (no overlap at all) | **none** |
| 6, face-on but 15° off square | 12 of 12 (at most 28% overlap) | none |

- Face-on, every sticker on the cube outline is seen flat in one photo.
- Every sticker turned out of it is seen at 48° in two photos.

The app's Photos step says all this, and the live webcam check marks the turning corner it found.

### Check and Solve

- **Check:**
  - **Each face card:** the flattened photo, its turning corner at the bottom-right, with a marker on each of the face's 29 places. A circle is a flat sticker, a diamond a turned-out one, a small dot no sticker. Select one to fix it; setting a colour clears the place it overlaps.
  - **The 3D model** is the simulator's, with its U D F R L labels, painted in **your cube's colours** (see [Cubes stickered another way](#cubes-stickered-another-way)).
  - **Without photos** (practice scramble, typed moves), the cards draw each face.
- **Solve:** the simulator's notation (`U U2 U3 U2' U' D … F R L`). Every move is named by a piece that never leaves its place:

  | Move | Where it is on the puzzle |
  |---|---|
  | U | the orange-yellow-green corner |
  | D | the red-white-blue corner |
  | F | the edge with the orange-white middle piece |
  | R | the edge with the yellow-blue middle piece |
  | L | the edge with the red-green middle piece |

  These hold in any shape, so you can follow a solution without knowing where the hidden core is. They are the
  model's colours; for a cube stickered another way the app says them in that cube's colours (a DaYan cube's
  U corner is red-yellow-blue).
- **If the solver is missing** (its ladder not copied in), the Solve step says so. After a practice scramble it still offers to follow the scramble's undo.

### Cubes stickered another way

Not every cube carries its colours as the model does. A **DaYan Tangram Cube Extreme** has the usual six colours
(same opposite pairs, same handedness), but its turning corners are **red-yellow-blue** and **orange-white-green**,
where the model (built from pictures on the internet) has orange-yellow-green and red-white-blue: every
sticker is the model's colour with orange↔red and green↔blue swapped. Others may be mirror images, or have
other colours opposite each other.

- **One permutation says it all**, the cube's *colour layout*: `paint[c]` is the colour the cube shows where
  the model shows colour c (`scanner/core/colours.py`). The networks read the colours they see; the decoder,
  every state and the solver stay in the model's colours (**the solver needs nothing new**); the 3D model,
  the face cards, the colour picker and every instruction are shown in the cube's colours
  (`web/src/app/palette.ts`).
- **Worked out on every scan**, nothing remembered. Each layout class (120 for the Tangram: 720 permutations
  up to its 6 symmetries; 30 for the Maltese Gear Cube, whose decoder already absorbs any whole-cube turn)
  is tried together with every way to give the photos the faces (`vision/search.py`: 0.5 s for all 86,400
  pairs, by scoring the 30 one-colour pieces once per colour and bounding before the exact assignments).
- **A usual cube is never relabelled by a poor photo.** The usual scheme is decoded first, exactly as
  before (retries of a misjudged turning corner included). Another layout is believed only when its answer
  is legal, clear (margin ≥ 10 nats) and better than the usual one by its prior plus 10 nats (prior: 12 for
  a rearrangement like the DaYan's, 16 for a mirror image or other pairs). Otherwise it is only offered.
- **On the Check step**, one line says so ("Your cube is stickered differently from the usual: orange and
  red swapped; green and blue swapped") with *Not right? Change the colours*: the photos are read again
  with the layout chosen (the usual colours are always among the choices). When a usual-coloured scan is
  doubtful and another layout came close, that is offered too.
- **A solved cube** looks the same in every layout (six one-colour faces), so it keeps the usual colours,
  and is decoded right regardless.
- **Kept scans** store the layout and what the cube itself shows (`labels.colours`, `labels.seen`), so real
  scans of such cubes train the readers correctly.

Measured on the 300 rendered scans (cached readings of the networks trained on v1; no retries):

| | Before | Now |
|---|---|---|
| Usual cubes, exactly right | 67.3% | 67.3% (every scan the same answer; never another layout) |
| Cubes stickered another way (readings relabelled), exactly right | 3.3% | 65.3% (rearranged 71%, mirrored 62%, other pairs 63%) |
| A camera that makes red look orange (80% of red's probability moved to orange) | 62.7% | 62.7% (2 of 300 already-wrong scans get a wrong layout) |
| Your three DaYan scans (2026-11-15) | doubtful (margins −9, 3, −4.5) | DaYan layout, legal, margins 16, 20, 22 |

### How it works

| Stage | What | Where |
|---|---|---|
| Positions | Every sticker is always in one of **120 positions** (centre + normal): 66 on the cube, 54 turned out. An observation is one value per position: colour 0-5 or 6 for none (the solver's state vector, up to order). | `web/src/tangram/model.ts` → `shared/tangram/layout.json` |
| Find the face | `CornerLocator`: heatmaps for the four corners in the face's own order (turning corner third). Two looks: whole photo, then close up. | `scanner/puzzles/tangram/vision/models.py`, `pipeline.py` |
| Cut | The camera's pose from the four corners (IPPE, an assumed lens; a wrong guess moves off-plane points by about a millimetre). Each of the face's 29 positions is cut out as a square in its own sticker's plane, so a tilted sticker looks like a flat one. | `vision/geometry.py` |
| Read | `PositionReader`: a ResNet trunk per crop, a context network for the light, and a vector per position. It reads 8 classes: six colours, **empty**, and **hidden** (a sticker is there but something stands in front). Three nudged looks are averaged. | `vision/models.py` |
| Fit the photos together | All 720 ways to give the photos the faces, each decoded fast, for the usual colours and then for every other colour layout (above). Six always fit equally well, because the cuts look the same after a whole-puzzle turn by any of the 6 symmetries that keep the diagonal. **Only one of them is reachable** with the core fixed, and that is the answer. | `vision/pipeline.py` `find_faces`, `vision/search.py` |
| Decode | Every piece fills one of **45 cells**; its orientation decides which cell and which positions. The best arrangement is a linear assignment of pieces to cells (Hungarian). Then an exact reachability test on the moves' stabiliser chain (Schreier-Sims; 0.6 s once, 0.65 ms a test; group order 4.75 × 10²⁸, matching the solver pipeline). If needed, the cheapest single or double change that makes it reachable. | `scanner/puzzles/tangram/decode.py`, `scanner/core/permgroup.py` |
| Solve | The ladder solver trained in Cube_Solving_Pipeline, through `scanner/puzzles/tangram/solver.py`. Every solution is replayed on the solver's tables and on this repository's model before it is shown. | [below](#the-tangram-solver) |

### Training the Tangram networks

**Before training, check that the GPU is free** (another job, such as the solver's training, may be using it):

```bash
nvidia-smi
```

The datasets render in Chrome, on the integrated GPU unless `--discrete-gpu` is given, so they can be made while the NVIDIA GPU trains something else. Rendering runs at about 160 pictures a second; the set below takes about 7 minutes. **They are rendered** (2026-11-15) in `C:/Users/tirth/cube_scanner_data/tangram/v1/{train,val,scans}`. Keep datasets outside OneDrive:

```bash
# 1. synthetic photos: one face straight on per picture, any state and shape, exact labels
python tools/gen_dataset.py --puzzle tangram --out C:/Users/tirth/cube_scanner_data/tangram/v1/train --count 60000 --tabs 4
python tools/gen_dataset.py --puzzle tangram --out C:/Users/tirth/cube_scanner_data/tangram/v1/val   --count 4000 --start-seed 1000000 --tabs 4
python tools/gen_dataset.py --puzzle tangram --out C:/Users/tirth/cube_scanner_data/tangram/v1/scans --scans --count 300 --start-seed 3000000 --tabs 2

# 2. train (best checkpoint -> data/checkpoints/tangram/)
python -m scanner.vision.train locator --puzzle tangram --data C:/Users/tirth/cube_scanner_data/tangram/v1 --epochs 10 --log 200
python -m scanner.vision.train reader  --puzzle tangram --data C:/Users/tirth/cube_scanner_data/tangram/v1 --epochs 8  --log 200

# 3. evaluate
python -m scanner.vision.evaluate photos --puzzle tangram --data C:/Users/tirth/cube_scanner_data/tangram/v1/val --out data/reports/tangram_photos_v1.json
python -m scanner.vision.evaluate scans  --puzzle tangram --data C:/Users/tirth/cube_scanner_data/tangram/v1/scans --out data/reports/tangram_scans_v1.json
python -m scanner.vision.evaluate real   --puzzle tangram      # labelled real scans kept from the app (data/real/tangram/scans)
```

**v2 datasets (for the next training): cubes stickered other ways, and reds that look orange.** On real webcam
photos the readers trained on v1 call most red stickers orange (21 orange and 1 red on the first DaYan scan; the
decoder still put the 11 reddest right). Two changes target that:

- `--layouts 0.5`: half the rendered puzzles get a random colour layout (labels are the colours shown), so the
  readers never learn where the usual scheme puts its colours.
- `--wide-shades`: sticker shades vary more from cube to cube, reds as far as orange-red and salmon, oranges as
  far as red-orange (red always stays redder than orange on one cube).
- Training also now and then makes a picture look like a warm, bright webcam (red runs into its ceiling), in
  `scanner/vision/augment.py`, for both puzzles.

```bash
python tools/gen_dataset.py --puzzle tangram --out C:/Users/tirth/cube_scanner_data/tangram/v2/train --count 60000 --tabs 4 --layouts 0.5 --wide-shades
python tools/gen_dataset.py --puzzle tangram --out C:/Users/tirth/cube_scanner_data/tangram/v2/val   --count 4000 --start-seed 1000000 --tabs 4 --layouts 0.5 --wide-shades
python tools/gen_dataset.py --puzzle tangram --out C:/Users/tirth/cube_scanner_data/tangram/v2/scans --scans --count 300 --start-seed 3000000 --tabs 2 --layouts 0.5 --wide-shades
python -m scanner.vision.train locator --puzzle tangram --data C:/Users/tirth/cube_scanner_data/tangram/v2 --epochs 10 --log 200
python -m scanner.vision.train reader  --puzzle tangram --data C:/Users/tirth/cube_scanner_data/tangram/v2 --epochs 8  --log 200
python -m scanner.vision.evaluate scans --puzzle tangram --data C:/Users/tirth/cube_scanner_data/tangram/v2/scans --out data/reports/tangram_scans_v2.json
```

The scans report gives exactly-right % for usual cubes and for cubes stickered another way, and how often the
colour layout is found. Training on v2 overwrites `data/checkpoints/tangram/`; copy the v1 checkpoints aside first
to compare.

- **Watching it work:** `web/datagen.html?puzzle=tangram-cube-extreme` (with `npm run dev`, or `--headed` on `gen_dataset.py`) shows one sample per click with its labels drawn over it. Red rings are stickers the camera cannot see.
- **The locator** reports the corner error and how often the photo is found the right way up. The checkpoint kept weighs both.
- **The reader** reports accuracy for colours, empty, hidden and sticker-or-not. The checkpoint kept is the best per-position accuracy, with a fitted temperature.
- **Options:** both take `--batch`, `--workers`, `--resume` and `--device`. Use `--device cpu` for a smoke test that must not touch the GPU.

### The Tangram solver

The solver is the hierarchical subgroup ladder trained in Cube_Solving_Pipeline:

- one exact table for the frame (the turning corners and the middle layer);
- a value network for the large pieces;
- a value network for the small triangles.

On 200-move scrambles it finds solutions of about 90 steps in about 5 s on the GPU. It loads in about 8 s, on the first solve or with `serve --warm`.

- **Code and ladder:**
  - code in `third_party/cube_solver/`;
  - the trained ladder (about 330 MB) in `data/ladders/tangram_extreme/learned_r7`.
- **The bridge** is `scanner/puzzles/tangram/solver.py`:
  - Both sides describe a state the same way (per sticker position, colour 0-5 or 6 for none), but list the 120 positions in different orders.
  - On loading, each position is matched by its frame (same normal, nearest centre).
  - The solved colours and all seven moves are then checked to agree, or the solver is refused.
- **Every solution is replayed twice:** on the solver's own tables, then on this repository's model of the puzzle. The browser checks it once more on the 3D model.
- **To refresh it** after the pipeline changes:
  1. Copy its `core/` and `puzzles/` into `third_party/cube_solver/`.
  2. Run `python tools/import_ladder.py --puzzle tangram`.
  3. Run `pytest tests/test_tangram_solver.py`.

```bash
python -m scanner solve --puzzle tangram --scramble "U2 F D' R U3 L D2 F"   # from the command line
```

---

## Training and evaluating (Maltese Gear Cube)

`--puzzle maltese` is the default, so the commands from before still work.

```bash
# synthetic photos (Chrome via Playwright); keep them outside OneDrive
python tools/gen_dataset.py --out C:/Users/tirth/cube_scanner_data/synth/v1/train --count 60000 --tabs 4
python tools/gen_dataset.py --out C:/Users/tirth/cube_scanner_data/synth/v1/val   --count 4000 --start-seed 1000000 --tabs 4
python tools/gen_dataset.py --out C:/Users/tirth/cube_scanner_data/synth/v1/scans --scans --count 400 --start-seed 3000000

# train (best checkpoint -> data/checkpoints/, or data/checkpoints/maltese/ once that folder exists)
python -m scanner.vision.train locator --data C:/Users/tirth/cube_scanner_data/synth/v1 --epochs 8
python -m scanner.vision.train reader  --data C:/Users/tirth/cube_scanner_data/synth/v1 --epochs 6

# evaluate
python -m scanner.vision.evaluate faces --data C:/Users/tirth/cube_scanner_data/synth/v1/val
python -m scanner.vision.evaluate scans --data C:/Users/tirth/cube_scanner_data/synth/v1/scans --any-order
python -m scanner.vision.evaluate real    # labelled real scans kept from the app (data/real/scans)

# v2: cubes stickered other ways (mirror images, other opposite pairs) and wider sticker shades
python tools/gen_dataset.py --out C:/Users/tirth/cube_scanner_data/synth/v2/train --count 60000 --tabs 4 --layouts 0.5 --wide-shades
python tools/gen_dataset.py --out C:/Users/tirth/cube_scanner_data/synth/v2/val   --count 4000 --start-seed 1000000 --tabs 4 --layouts 0.5 --wide-shades
python tools/gen_dataset.py --out C:/Users/tirth/cube_scanner_data/synth/v2/scans --scans --count 400 --start-seed 3000000 --layouts 0.5 --wide-shades
```

The Maltese Gear Cube's scan works out the cube's colour layout too ([how](#cubes-stickered-another-way)): any whole-cube
turn of the usual scheme is already handled by its decoder, so the layouts told apart are the mirror image and the
28 with other opposite pairs.

**Labelled real scans.** Every scan you keep from the Check step is saved with its photos and labels:

- Maltese Gear Cube: `data/real/scans/<id>/`;
- Tangram Cube Extreme: `data/real/tangram/scans/<id>/`.

The labels are the state you checked, or the known scramble's exact state.

---

## Development

```bash
python -m scanner serve          # API on :8000
cd web && npm run dev            # app on http://localhost:5173 with hot reload; /api is proxied to :8000

pytest                           # decoders, arrangement searches, APIs, solver round trip (the solver test takes ~1 min, GPU)
pytest -m "not solver"           # everything else
cd web && npm test               # engines, observations, layouts, solver dictionary, app frames, the real scan's solution
cd web && npm run typecheck
cd web && npm run export:shared  # regenerate shared/maltese/*.json and shared/tangram/layout.json from the engines
```

`web/datagen.html?puzzle=<id>` is the synthetic-photo page (opened by `tools/gen_dataset.py`; also usable by hand).

## Repository layout

```
scanner/                        Python package
  server.py                     FastAPI app: mounts each puzzle's endpoints at /api/<puzzle id>/ (and the Maltese ones at /api/)
  core/                         shared by every puzzle: photos and pending scans, lazy vision and solver status, permgroup
  vision/                       shared vision: common (ResNet, heatmap locator, letterbox, training plumbing), augment,
                                and the CLIs `python -m scanner.vision.train|evaluate --puzzle <key>`
  puzzles/                      the registry (REGISTRY) and one package per puzzle, each with the same parts:
    maltese/                      layout, decode, solver, api, vision/{geometry,dataset,models,pipeline,train,evaluate}
    tangram/                      layout, decode, solver (bridge), api, vision/{…same…}
web/                            Vite + TypeScript + three.js
  src/app/                      the app shell shared by every puzzle: steps, photos (upload, webcam, live check), API client
    puzzles.ts                  the puzzles the app offers
    maltese/, tangram/          each puzzle's PuzzleUI: its photo protocol, Check and Solve screens, 3D view, model
  src/maltese/, src/tangram/    what each puzzle shows (observation/positions) and its layout export for Python
  src/datagen/                  the synthetic photo studio: common/ (light, rooms, backdrops, hands) + one folder per puzzle
  src/vendor/                   copied 3D engines and renderers (from Cube_Modeling)
  tests/                        vitest
shared/maltese/, shared/tangram/  generated from the engines; read by both sides
third_party/cube_solver/        copied solver (from Cube_Solving_Pipeline)
tools/                          gen_dataset (--puzzle), import_ladder (--puzzle), export_solver_tables, derive_invariants
tests/                          pytest
data/                           not in git: checkpoints, ladders, reports, real scans, cache
```

## Adding a puzzle

The pipeline is the same for every puzzle: photos, then locator, crops, reader, then decoder, then a check on the 3D model, then the solver. Each part has one place:

1. **Engine** (from Cube_Modeling): copy into `web/src/vendor/<puzzle>/`.
2. **What it shows** (`web/src/<puzzle>/`): an observation computed from the engine state, and a layout export for Python. Add it to `web/tools/export-shared.gen.ts`, with tests tying it to the engine.
3. **Server side** (`scanner/puzzles/<key>/`): `layout.py` (reads `shared/<key>/`), `decode.py`, `solver.py`, `api.py` with `create_service()`. Then add it to `REGISTRY` in `scanner/puzzles/__init__.py`.
   - `scanner/core/permgroup.py` gives any puzzle an exact "can moves reach this?" test from its move permutations.
4. **Vision** (`scanner/puzzles/<key>/vision/`): geometry, dataset, models, pipeline, train and evaluate.
   - `scanner/vision/common.py` has the shared parts; `--puzzle <key>` finds the training and evaluation scripts.
5. **Synthetic photos:** `web/src/datagen/<puzzle>/`, which extends `StudioBase` and uses `sceneAround`. Add it to `web/src/datagen/main.ts` and `tools/gen_dataset.py`.
6. **App** (`web/src/app/<puzzle>/ui.ts`): a `PuzzleUI` with its photo protocol and its Check and Solve screens. List it in `web/src/app/puzzles.ts`.

**Decide the camera angles first.** Measure on the 3D model which views tell apart every state the reader must read, as was done for the Tangram's look-alike stickers. Then generate the dataset.

## Copied code

- **From Cube_Modeling** (commit `c4d0120`), into `web/src/vendor/`:
  - the Maltese engine and renderer: `src/maltese/*`, `src/maltese/render/*`;
  - the Tangram engine and renderer (2026-11-15):
    - `src/puzzle/{model,moves,pieces,planes,polyhedron,state,variants}.ts` and `src/config.ts` into `vendor/tangram/`;
    - `src/render/{meshes,polygon2d,animator,labels}.ts` into `vendor/tangram/render/`;
    - imports rewritten only;
  - `src/puzzle/{vec,group,history}.ts` and `src/render/*`;
  - the stylesheet;
  - the engines' tests (`web/tests/vendor/`).
  - **Local additions:** `MalteseState.fromPoses`, `poseProblems`, `rotated`, the whole-cube rotation table in `moves.ts`, and `rng.ts`.
- **From Cube_Solving_Pipeline** (working tree of 2026-11-15 13:18, uncommitted, on HEAD `141c053`), into `third_party/cube_solver/`: `core/` and `puzzles/`.
  - This includes the Tangram compiler (`core/geometry.py`), `core/notation.py`, `puzzles/registry.py` and `puzzles/tangram/`.
  - Its README and Tangram notes are copied beside them as `UPSTREAM_*.md`.
  - The ladder caches are copied with `python tools/import_ladder.py [--puzzle tangram]`.

Both source repos are untouched. When they change, copy again and run both test suites.

## Data (not in git)

| What | Where |
|---|---|
| Trained networks | Maltese: `data/checkpoints/locator.pt`, `reader.pt`; Tangram: `data/checkpoints/tangram/` (+ `.json` scores) |
| Solver ladders | Maltese (~515 MB): `data/ladders/learned_r7`; Tangram (~330 MB): `data/ladders/tangram_extreme/learned_r7` (`python tools/import_ladder.py --puzzle …`) |
| Synthetic photos | `C:/Users/tirth/cube_scanner_data/synth/v1/` (Maltese), `…/tangram/v1/` (Tangram), each `{train,val,scans}`; Tangram v2 (half the cubes in other colour layouts, wider shades) rendered 2026-11-15 in `…/tangram/v2/` |
| Evaluation reports | `data/reports/` |
| Real scans | Maltese `data/real/{pending,scans}/`, Tangram `data/real/tangram/{pending,scans}/` |
| Caches | `data/cache/` (the Tangram move group's stabiliser chain) |

## Next steps

1. **Tangram: render v2 and retrain** ([commands above](#training-the-tangram-networks)) so the readers tell red from orange on real webcams.
   - Keep checked real scans from the app (the DaYan scans included): they measure the real-photo accuracy, and are the data to fine-tune on.
2. **Maltese: measure on real photos.** Do about 30 practice scrambles in the app and keep each scan. Then run `python -m scanner.vision.evaluate real`.
   - Fine-tune if real photos fall short of the targets: 99.8% per sticker, and 95% of scans needing no manual fix. Mix 20 real scans with the synthetic set and hold 10 out for testing.

## License

[MIT](LICENSE)
