"""Photos in, a decoded cube out.

    vision = Vision()                       # loads {locator,reader}.pt from the Maltese checkpoint folder
    quad, confidence = vision.locate(photo) # the face's corners in the photo
    reading = vision.read(photo, quad)      # probabilities for the face's 16 cells and 4 gears
    result = vision.scan(photos)            # six faces -> the legal cube they show (decode.py)

Photos are HxWx3 uint8 RGB arrays. A scan is six photos, one per face, **in any order and
any way up**: which face each photo shows, and how it is turned, is worked out from the
corners (``find_arrangement``): every corner piece shows on three photos, and only one way of
fitting the six photos together makes all eight of them real pieces. The first photo is
called the front (F) and its top edge "up"; that naming does not matter, because the decoder
turns the whole cube to the solver's frame by finding the white-green-red corner.

``SCAN_ORDER`` (front, right, back, left, top, bottom) is the order the synthetic test scans
were rendered in; ``scan(..., faces=SCAN_ORDER)`` still reads a scan in that fixed order.

**Colour layouts** (``scanner.core.colours``). A cube may carry its six colours another way: a mirror
image of the usual scheme, or other colours opposite each other (any whole-cube turn of the scheme is
already the usual one, since the decoder turns the cube itself). ``find_colours`` also tries every other
layout (29 classes): each is scored by the quick arrangement score, the best few are decoded in full,
and one is believed only when its cube is legal and clear and explains the photos better than the
usual colours by its prior (``COLOUR_PRIOR``) plus ``ARRANGEMENT_DOUBT_NATS``. Readings stay in the
colours the cube shows; ``FaceReading.coloured`` puts them in the model's.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from scanner.core.colours import DEFAULT_PRIOR, ColourLayout
from scanner.paths import checkpoints_dir
from scanner.puzzles.maltese.decode import DecodeResult, decode_probs
from scanner.puzzles.maltese.layout import (
    FACE_NAMES,
    N_CELLS,
    N_COLOURS,
    N_GEARS,
    Layout,
    load_layout,
)
from scanner.vision.common import LOCATOR_INPUT, letterbox

from .geometry import (
    cut_crops,
    face_cells,
    picture_order,
    prepare_photo,
    quad_side,
    rectify,
    turn_face,
)
from .models import FaceLocator, SlotReader

#: The faces of a scan in the order they are photographed, as indices into FACE_NAMES (U D F B L R).
SCAN_ORDER = tuple(FACE_NAMES.index(f) for f in ("F", "R", "B", "L", "U", "D"))
#: A wrongly turned photo has to beat the expected orientation by this much (nats) to be believed.
TURN_PRIOR = 4.0
#: Nats a colour layout of each kind must explain the photos better by than the model's own.
COLOUR_PRIOR = DEFAULT_PRIOR
#: Other colour layouts decoded in full (the best by the quick score), and offered to the person.
COLOUR_CANDIDATES = 3


@dataclass
class FaceReading:
    """What one photo shows, in the orientation of the photo."""

    cells: np.ndarray           # (16, 6) probabilities
    gear_colour: np.ndarray     # (4, 6)
    gear_quarter: np.ndarray    # (4, 4)

    def turned(self, k: int, layout: Layout) -> "FaceReading":
        """The same reading for a face that appears turned k quarter turns counterclockwise in its photo."""
        return FaceReading(*turn_face(self.cells, self.gear_colour, self.gear_quarter, k, layout))

    def coloured(self, colours: ColourLayout | None) -> "FaceReading":
        """The reading in the model's colours, for a cube with colour layout ``colours``."""
        if colours is None or colours.is_default:
            return self
        return FaceReading(colours.model_columns(self.cells), colours.model_columns(self.gear_colour), self.gear_quarter)


@dataclass
class Arrangement:
    """Which face each photo shows and how it is turned, found from the corners."""

    #: Index into FACE_NAMES of the face each photo shows (photo 0 is F by definition).
    faces: tuple[int, ...]
    #: Quarter turns (counterclockwise) each photo's face appears turned in it.
    turns: list[int]
    #: Log-likelihood of the cube decoded from the best arrangement, and from the best one that decodes
    #: to a different cube (nats).
    score: float
    runner_up: float
    #: The cube the best arrangement decodes to.
    decode: DecodeResult | None = field(default=None, repr=False)

    @property
    def margin(self) -> float:
        """How much more likely the chosen arrangement is than any other (nats). Under ~10 it is doubtful."""
        return float(self.score - self.runner_up)


#: An arrangement whose margin over the next best is below this is reported as doubtful.
ARRANGEMENT_DOUBT_NATS = 10.0


@dataclass
class ScanResult:
    decode: DecodeResult
    #: Quarter turns (counterclockwise) each photo was found to be turned from the expected way up.
    turns: list[int]
    quads: list[np.ndarray]
    locate_confidence: list[float]
    readings: list[FaceReading] = field(repr=False, default_factory=list)
    #: Index into FACE_NAMES of the face each photo shows, in the frame the cube was photographed in.
    faces: tuple[int, ...] = SCAN_ORDER
    arrangement: Arrangement | None = None
    #: The cube's colour layout, and the next closest ones with how much less likely each is (nats).
    colours: ColourLayout = field(default_factory=ColourLayout.default)
    alternatives: list[tuple[ColourLayout, float]] = field(default_factory=list)

    def model_readings(self) -> list[FaceReading]:
        """The readings in the model's colours."""
        return [r.coloured(self.colours) for r in self.readings]


class Vision:
    def __init__(self, checkpoints: Path | None = None, device: str | None = None) -> None:
        self.dir = Path(checkpoints or checkpoints_dir("maltese"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.layout = load_layout()
        self.locator: FaceLocator | None = None
        self.reader: SlotReader | None = None
        self.info: dict[str, dict] = {}

    # -- loading --------------------------------------------------------------------------
    def available(self) -> dict[str, bool]:
        return {"locator": (self.dir / "locator.pt").exists(), "reader": (self.dir / "reader.pt").exists()}

    def load(self) -> "Vision":
        if self.locator is None and (self.dir / "locator.pt").exists():
            ck = torch.load(self.dir / "locator.pt", map_location="cpu", weights_only=False)
            m = FaceLocator(pretrained=False)
            m.load_state_dict(ck["model"])
            self.locator = m.to(self.device).eval()
            self.info["locator"] = {"epoch": ck.get("epoch"), "scores": ck.get("scores")}
        if self.reader is None and (self.dir / "reader.pt").exists():
            ck = torch.load(self.dir / "reader.pt", map_location="cpu", weights_only=False)
            m = SlotReader(pretrained=False)
            m.load_state_dict(ck["model"])
            self.reader = m.to(self.device).eval()
            self.info["reader"] = {"epoch": ck.get("epoch"), "scores": ck.get("scores")}
        return self

    def _need(self, what: str):
        self.load()
        model = getattr(self, what)
        if model is None:
            raise RuntimeError(f"No trained {what} in {self.dir}. Train one: python -m scanner.vision.train {what} --puzzle maltese --data <dataset>")
        return model

    # -- the face's corners ------------------------------------------------------------------
    @torch.no_grad()
    def _corners(self, square: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = self._need("locator")
        x = torch.from_numpy(square).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits = model(x)
        pts, conf = FaceLocator.corners(logits)
        return pts[0].cpu().numpy().astype(np.float64) * square.shape[0], conf[0].cpu().numpy().astype(np.float64)

    def locate(self, image: np.ndarray, passes: int = 2) -> tuple[np.ndarray, float]:
        """The face's corners in the photo (4, 2) in pixels, top-left first as the photo shows it, and a
        confidence in 0-1. The first look is at the whole photo; each further pass looks again at a
        window just around the face, where its corners are several times larger."""
        s = LOCATOR_INPUT
        square, scale, dx, dy = letterbox(image, s)
        pts, conf = self._corners(square)
        quad = (pts - np.array([dx, dy])) / scale
        confidence = float(conf.min())
        for _ in range(max(0, passes - 1)):
            lo, hi = quad.min(0), quad.max(0)
            side = 1.75 * float(max(hi - lo))
            if side < 8:
                break
            centre = (lo + hi) / 2
            k = s / side
            m = np.array([[k, 0.0, s / 2 - k * centre[0]], [0.0, k, s / 2 - k * centre[1]]])
            window = cv2.warpAffine(image, m, (s, s), flags=cv2.INTER_AREA if k < 0.7 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            pts, conf = self._corners(window)
            refined = (pts - m[:, 2]) / k
            # Keep the closer look only if it agrees roughly with the first one (it cannot have found another cube).
            if float(conf.min()) >= 0.5 * confidence and np.abs(refined - quad).max() < 0.35 * quad_side(quad):
                quad, confidence = refined, max(confidence, float(conf.min()))
        quad, _ = picture_order(quad)
        return quad, confidence

    # -- what a face shows ---------------------------------------------------------------------
    @torch.no_grad()
    def read(self, image: np.ndarray, quad: np.ndarray, looks: int = 5, jitter: float = 0.006) -> FaceReading:
        """Probabilities for the face inside ``quad``. The face is cut out ``looks`` times with its
        corners nudged a little differently, and the readings are averaged: no single placement of
        the corners is exactly right, and a slot that reads differently when nudged should say so."""
        model = self._need("reader")
        image, quad = prepare_photo(image, np.asarray(quad, dtype=np.float64))
        rng = np.random.default_rng(12345)
        side = quad_side(quad)
        crops, ctx = [], []
        for i in range(max(1, looks)):
            q = quad if i == 0 else quad + rng.normal(0.0, jitter * side, quad.shape)
            c, x = cut_crops(image, q)
            crops.append(c)
            ctx.append(x)
        cr = torch.from_numpy(np.stack(crops)).permute(0, 1, 4, 2, 3).contiguous().to(self.device)
        cx = torch.from_numpy(np.stack(ctx)).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits = model(cr, cx)
        p = {k: v.mean(0).cpu().numpy().astype(np.float64) for k, v in model.probabilities(logits).items()}
        fc = face_cells(self.layout)
        cells = np.zeros((16, N_COLOURS))
        cells[fc.star] = p["star"]
        cells[fc.edge.reshape(-1)] = p["edge"].reshape(-1, N_COLOURS)
        cells[fc.centre] = p["centre"]
        return FaceReading(cells, p["gear_colour"], p["gear_quarter"])

    # -- six faces -> a cube ---------------------------------------------------------------------
    def assemble(self, readings: list[FaceReading], turns: list[int], faces: tuple[int, ...] = SCAN_ORDER) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cell_p = np.zeros((N_CELLS, N_COLOURS))
        gc_p = np.zeros((N_GEARS, N_COLOURS))
        gq_p = np.zeros((N_GEARS, 4))
        for reading, k, f in zip(readings, turns, faces):
            r = reading.turned(k, self.layout)
            cell_p[16 * f:16 * f + 16] = r.cells
            gc_p[4 * f:4 * f + 4] = r.gear_colour
            gq_p[4 * f:4 * f + 4] = r.gear_quarter
        return cell_p, gc_p, gq_p

    def find_turns(self, readings: list[FaceReading], faces: tuple[int, ...] = SCAN_ORDER) -> list[int]:
        """Which way up each photo was taken, from the corners: each corner piece shows on three
        photos, and only when all three are the right way up do its three stars make a real piece."""
        layout = self.layout
        logp = np.zeros((6, 4, 16, N_COLOURS))            # face index, turn, cell, colour
        for reading, f in zip(readings, faces):
            for k in range(4):
                logp[f, k] = np.log(np.clip(reading.turned(k, layout).cells, 1e-9, 1.0))
        tables = []
        for slot in layout.corners:
            fs = [c // 16 for c in slot.cells]
            t = np.full((4, 4, 4), -np.inf)
            for ks in itertools.product(range(4), repeat=3):
                best = -np.inf
                for o in slot.options:
                    v = sum(logp[fs[i], ks[i], slot.cells[i] % 16, o.colours[i]] for i in range(3))
                    best = max(best, v)
                t[ks] = best
            tables.append((fs, t))
        best_score, best_turns = -np.inf, (0,) * 6
        for turns in itertools.product(range(4), repeat=6):
            score = sum(t[turns[fs[0]], turns[fs[1]], turns[fs[2]]] for fs, t in tables) - TURN_PRIOR * sum(1 for k in turns if k)
            if score > best_score:
                best_score, best_turns = score, turns
        return [int(best_turns[f]) for f in faces]

    def find_colours(self, readings: list[FaceReading], candidates: int = 24) -> tuple[Arrangement, ColourLayout, list[tuple[ColourLayout, float]]]:
        """The arrangement and the cube's colour layout: the model's own unless another is believed (see the
        module notes). Returns (arrangement, layout, the closest other layouts with how much less likely)."""
        layouts = self.layout.colour_layouts
        usual = self.find_arrangement(readings, candidates)
        quick = []
        for c in layouts[1:]:
            q = self._quick_arrangements([r.coloured(c) for r in readings], 1)[0][2]
            quick.append((q - COLOUR_PRIOR[c.kind], c))
        quick.sort(key=lambda x: -x[0])
        tried = []
        for _, c in quick[:COLOUR_CANDIDATES]:
            a = self.find_arrangement(readings, candidates, colours=c)
            tried.append((a.score - COLOUR_PRIOR[c.kind], c, a))
        tried.sort(key=lambda x: -x[0])
        best_score, best_c, best = tried[0]
        believed = (best.decode is not None and best.decode.legal and best.margin >= ARRANGEMENT_DOUBT_NATS
                    and best_score >= usual.score + ARRANGEMENT_DOUBT_NATS)
        if believed:
            best.runner_up = max(best.runner_up, usual.score + COLOUR_PRIOR[best_c.kind])
            others = [(layouts[0], float(best_score - usual.score))] + [(c, float(best_score - s)) for s, c, _ in tried[1:]]
            return best, best_c, others[:COLOUR_CANDIDATES]
        return usual, layouts[0], [(c, float(usual.score - s)) for s, c, _ in tried]

    def find_arrangement(self, readings: list[FaceReading], candidates: int = 24, colours: ColourLayout | None = None) -> Arrangement:
        """Which face each of six photos shows and how it is turned, taken in any order and any way up.

        Photo 0 is called F and its top edge up (that fixes the whole-cube rotation, which does not
        matter). Every other photo can be any of the five other faces, turned any of four ways:
        120 x 1024 arrangements.

        1. A quick score for every arrangement: how well each slot that spans two or three faces (the
           8 corners and 12 edge slots) can be filled with *some* real piece.
        2. The best ``candidates`` of those are decoded in full (every piece exactly once, the puzzle's
           invariants), and the one whose decoded cube best explains the photos wins. Step 1 alone can
           tie: it does not stop two slots from claiming the same piece.

        With ``colours`` the cube is taken to have that colour layout.
        """
        readings = [r.coloured(colours) for r in readings]
        quick = self._quick_arrangements(readings, candidates)
        scored = []
        for faces, turns, _ in quick:
            cell_p, gc_p, gq_p = self.assemble(readings, turns, faces)
            d = decode_probs(cell_p, gc_p, gq_p, self.layout)
            scored.append((self._decoded_loglik(d, cell_p, gc_p, gq_p), faces, turns, d))
        scored.sort(key=lambda s: -s[0])
        best = scored[0]
        # The runner-up is the best arrangement that decodes to a *different* cube (two arrangements that
        # give the same cube are the same answer).
        key = best[3].observation
        runner = next((s[0] for s in scored[1:] if s[3].observation is None or key is None or s[3].observation != key), -np.inf)
        return Arrangement(best[1], best[2], float(best[0]), float(runner), best[3])

    def _decoded_loglik(self, d: DecodeResult, cell_p: np.ndarray, gc_p: np.ndarray, gq_p: np.ndarray) -> float:
        """Log-likelihood of a decoded cube under the readings, in the frame the readings are in."""
        if d.observation is None:
            return -np.inf
        layout = self.layout
        held = layout.rotations[int(layout.group_inv[d.rotation])].apply(d.observation)
        eps = 1e-9
        ll = np.log(np.clip(cell_p[np.arange(N_CELLS), held.cells], eps, 1)).sum()
        ll += np.log(np.clip(gc_p[np.arange(N_GEARS), held.gear_colour], eps, 1)).sum()
        ll += np.log(np.clip(gq_p[np.arange(N_GEARS), held.gear_quarter], eps, 1)).sum()
        return float(ll) - (0.0 if d.legal else 1e3)

    def _quick_arrangements(self, readings: list[FaceReading], top: int) -> list[tuple[tuple[int, ...], list[int], float]]:
        """The ``top`` arrangements by the quick slot-by-slot score: (faces of photos, turns of photos, score)."""
        layout = self.layout
        if len(readings) != 6:
            raise ValueError("an arrangement needs six photos")
        # logp[a, cell, colour] for a = photo * 4 + turn
        logp = np.empty((24, 16, N_COLOURS))
        for p, reading in enumerate(readings):
            for k in range(4):
                logp[4 * p + k] = np.log(np.clip(reading.turned(k, layout).cells, 1e-6, 1.0))
        tables = []
        for slot in [*layout.corners, *layout.radial]:
            slot_faces = sorted({c // 16 for c in slot.cells})
            if len(slot_faces) < 2:
                continue
            shape = (24,) * len(slot_faces)
            best = np.full(shape, -np.inf)
            for option in slot.options:
                v = np.zeros(shape)
                for cell, colour in zip(slot.cells, option.colours):
                    axis = slot_faces.index(cell // 16)
                    v = v + logp[:, cell % 16, colour].reshape([24 if i == axis else 1 for i in range(len(slot_faces))])
                best = np.maximum(best, v)
            tables.append((slot_faces, best))
        front = FACE_NAMES.index("F")
        others = [f for f in range(6) if f != front]
        perms = np.array(list(itertools.permutations(range(1, 6))))         # photo shown on each face of `others`
        turns = np.array(list(itertools.product(range(4), repeat=5)))
        a = np.zeros((len(perms), len(turns), 6), dtype=np.int64)          # per face: photo * 4 + turn
        for j, f in enumerate(others):
            a[:, :, f] = perms[:, None, j] * 4 + turns[None, :, j]
        a = a.reshape(-1, 6)
        score = np.zeros(len(a))
        for slot_faces, table in tables:
            score += table[tuple(a[:, f] for f in slot_faces)]
        top = min(top, len(score))
        order = np.argpartition(-score, top - 1)[:top]
        order = order[np.argsort(-score[order], kind="stable")]
        out = []
        for i in order:
            row = a[i]
            faces_of_photo = [0] * 6
            turn_of_photo = [0] * 6
            for f in range(6):
                faces_of_photo[row[f] // 4] = f
                turn_of_photo[row[f] // 4] = int(row[f] % 4)
            out.append((tuple(faces_of_photo), turn_of_photo, float(score[i])))
        return out

    def scan(self, images: list[np.ndarray], quads: list[np.ndarray] | None = None, turns: list[int] | None = None,
             faces: tuple[int, ...] | None = None, colours: ColourLayout | None = None) -> ScanResult:
        """Six photos -> the legal cube they show.

        With ``faces`` (and optionally ``turns``) the photos are taken to show those faces; without, the
        photos may be in any order and any way up and ``find_arrangement`` works out which is which.
        With ``colours`` the cube is taken to have that colour layout; otherwise it is worked out.
        """
        if len(images) != 6:
            raise ValueError("a scan is six photos, one of each face")
        confidence = [1.0] * 6
        if quads is None:
            located = [self.locate(im) for im in images]
            quads = [q for q, _ in located]
            confidence = [c for _, c in located]
        readings = [self.read(im, q) for im, q in zip(images, quads)]
        arrangement = None
        decoded = None
        alternatives: list[tuple[ColourLayout, float]] = []
        if faces is None:
            if colours is None:
                arrangement, colours, alternatives = self.find_colours(readings)
            else:
                arrangement = self.find_arrangement(readings, colours=colours)
                # The person chose this layout: still offer the others that fit (so they can go back).
                best, found, others = self.find_colours(readings)
                offered = [(found, 0.0)] + [(c, d) for c, d in others]
                alternatives = [(c, float(arrangement.score - (best.score - d))) for c, d in offered if c != colours]
                if not colours.is_default and all(not c.is_default for c, _ in alternatives):
                    alternatives.insert(0, (self.layout.colour_layouts[0], 0.0))
                alternatives = alternatives[:COLOUR_CANDIDATES]
            faces = arrangement.faces
            if turns is None:
                turns, decoded = arrangement.turns, arrangement.decode
        colours = colours or ColourLayout.default()
        model = [r.coloured(colours) for r in readings]
        if faces is not None and turns is None:
            turns = self.find_turns(model, faces)
        if decoded is None:
            cell_p, gc_p, gq_p = self.assemble(model, list(turns), faces)
            decoded = decode_probs(cell_p, gc_p, gq_p, self.layout)
        return ScanResult(decoded, list(turns), [np.asarray(q) for q in quads], confidence, readings, tuple(faces), arrangement,
                          colours, alternatives)

    def flatten(self, image: np.ndarray, quad: np.ndarray, size: int = 384) -> np.ndarray:
        """The face flattened, for showing beside the 3D model."""
        image, quad = prepare_photo(image, np.asarray(quad, dtype=np.float64), face_px=size)
        return rectify(image, quad, size)
