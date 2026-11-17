"""Six face-on photos of a Tangram Cube Extreme in, a decoded legal state out.

    vision = TangramVision()                  # loads data/checkpoints/tangram/{locator,reader}.pt
    quad, confidence = vision.locate(photo)   # the face's corners, turning corner third
    probs = vision.read(photo, quad)          # (29, 8): per face position, colour 0-5 / empty / hidden
    result = vision.scan(photos)              # six photos, any order and any way up -> TangramScan

**Which way up** each photo is comes from the locator: it finds the face's corners starting from the
face's turning corner. **Which face** each photo shows, and **which colour layout** the cube has (how its
colours sit on the pieces; ``scanner.core.colours``), are worked out together from what the photos show:
every pair of a layout (120 classes) and a way to give the six photos the six faces (720) is scored by a
fast assignment of pieces to cells (``search.py``), and the best few are checked for reachability. A
layout other than the model's own must explain the photos clearly better (``COLOUR_PRIOR``), so a cube
stickered the usual way is read as before. The puzzle's cuts look the same after a whole-puzzle turn by
any of the six cube symmetries that keep its diagonal, so six assignments always fit equally well; only
one of them is a state moves can reach with the hidden core fixed, and that one is the answer.

**A camera that makes red look orange.** The cube has 11 stickers of each colour, and every arrangement of
its real pieces has too, so the decoder already takes, say, the 11 reddest of 21 stickers the networks
call orange as the red ones: the state decoded is right as long as red stickers look *more* red than the
orange ones (see scanner/core/colours.py for why there is no per-colour rebalancing on top).

Each photo's reading is evidence for the positions it shows (a turned-out sticker is in two photos);
"hidden" says a sticker is there but not which colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from scanner.core.colours import DEFAULT_PRIOR, ColourLayout
from scanner.paths import checkpoints_dir
from scanner.vision.common import LOCATOR_INPUT, letterbox, prepare_photo, quad_side

from ..decode import TangramDecode, decode_logp, reachable
from ..layout import EMPTY, N_COLOURS, TangramLayout, load_layout
from .geometry import (
    HIDDEN,
    N_CLASSES,
    FaceGeometry,
    cut_crops,
    face_geometry,
    face_pose,
    flat_points,
    rectify,
)
from .models import CornerLocator, PositionReader
from .search import FaceSearch

#: An answer less than this many nats more likely than the next different one is reported as doubtful.
DOUBT_NATS = 10.0
#: Face assignments checked for reachability after the fast first look.
CANDIDATES = 36
#: The face is cut out about this many pixels across before reading.
FACE_PX = 520.0
#: Nats a colour layout of each kind must explain the photos better by than the model's own.
COLOUR_PRIOR = DEFAULT_PRIOR
#: Other colour layouts offered to the person when the scan's is not the usual one.
ALTERNATIVES = 3


@dataclass
class FaceAnswer:
    """Which face each photo shows and the cube's colour layout."""

    faces: tuple[int, ...]
    decode: TangramDecode
    #: Log-likelihood margin over the next most likely different answer (nats).
    margin: float
    colours: ColourLayout = field(default_factory=ColourLayout.default)
    #: Other layouts that came closest (best first), with how much less likely each is (nats).
    alternatives: list[tuple[ColourLayout, float]] = field(default_factory=list)


@dataclass
class TangramScan:
    decode: TangramDecode
    #: Index into the layout's faces (+x -x +y -y +z -z) of the face each photo shows.
    faces: tuple[int, ...]
    quads: list[np.ndarray]
    locate_confidence: list[float]
    #: Per photo (29, 8) probabilities, in its face's frame.
    readings: list[np.ndarray] = field(repr=False, default_factory=list)
    #: Log-likelihood margin over the next most likely different answer (nats).
    margin: float = float("inf")
    #: Photos whose corners were turned from what the locator said (it misjudged the turning corner).
    returned: dict[int, int] = field(default_factory=dict)
    #: The cube's colour layout, and the next closest ones.
    colours: ColourLayout = field(default_factory=ColourLayout.default)
    alternatives: list[tuple[ColourLayout, float]] = field(default_factory=list)

    def evidence(self) -> list[np.ndarray]:
        """Per photo (29, 7) log-likelihoods in the model's colours, as the decode used them."""
        return [self.colours.model_columns(evidence(r)) for r in self.readings]

    @property
    def doubtful(self) -> bool:
        return self.margin < DOUBT_NATS


def evidence(probs: np.ndarray) -> np.ndarray:
    """A reading (29, 8) as log-likelihoods over (colour 0-5, empty) per face position (29, 7). A hidden
    sticker is there, colour unknown."""
    p = np.asarray(probs, dtype=np.float64)
    out = np.empty((len(p), N_COLOURS + 1))
    out[:, :N_COLOURS] = p[:, :N_COLOURS] + p[:, HIDDEN:HIDDEN + 1] / N_COLOURS
    out[:, EMPTY] = p[:, EMPTY]
    return np.log(np.clip(out, 1e-9, 1.0))


def assemble(evid: list[np.ndarray], faces: tuple[int, ...], geom: FaceGeometry, n_positions: int) -> np.ndarray:
    """Per puzzle position, log-probabilities over (colour 0-5, empty) from every photo that shows it."""
    logp = np.zeros((n_positions, N_COLOURS + 1))
    for e, f in zip(evid, faces):
        np.add.at(logp, geom.position_of[f], e)
    # Normalise each position (a constant per position: it does not change which state is best).
    m = logp.max(1, keepdims=True)
    return logp - (m + np.log(np.exp(logp - m).sum(1, keepdims=True)))


class TangramVision:
    def __init__(self, checkpoints: Path | None = None, device: str | None = None) -> None:
        self.dir = Path(checkpoints or checkpoints_dir("tangram"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.layout: TangramLayout = load_layout()
        self.geom = face_geometry(self.layout)
        self.search = FaceSearch(self.layout, self.geom)
        self.locator: CornerLocator | None = None
        self.reader: PositionReader | None = None
        self.info: dict[str, dict] = {}

    # -- loading --------------------------------------------------------------------------------
    def available(self) -> dict[str, bool]:
        return {"locator": (self.dir / "locator.pt").exists(), "reader": (self.dir / "reader.pt").exists()}

    def load(self) -> "TangramVision":
        if self.locator is None and (self.dir / "locator.pt").exists():
            ck = torch.load(self.dir / "locator.pt", map_location="cpu", weights_only=False)
            m = CornerLocator(pretrained=False)
            m.load_state_dict(ck["model"])
            self.locator = m.to(self.device).eval()
            self.info["locator"] = {"epoch": ck.get("epoch"), "scores": ck.get("scores")}
        if self.reader is None and (self.dir / "reader.pt").exists():
            ck = torch.load(self.dir / "reader.pt", map_location="cpu", weights_only=False)
            m = PositionReader(self.geom.n, pretrained=False)
            m.load_state_dict(ck["model"])
            self.reader = m.to(self.device).eval()
            self.info["reader"] = {"epoch": ck.get("epoch"), "scores": ck.get("scores")}
        return self

    def _need(self, what: str):
        self.load()
        model = getattr(self, what)
        if model is None:
            raise RuntimeError(f"No trained {what} in {self.dir}. Train one: python -m scanner.vision.train {what} --puzzle tangram --data <dataset>")
        return model

    # -- the face's corners -----------------------------------------------------------------------
    @torch.no_grad()
    def _corners(self, square: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = self._need("locator")
        x = torch.from_numpy(square).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits = model(x)
        pts, conf = CornerLocator.corners(logits)
        return pts[0].cpu().numpy().astype(np.float64) * square.shape[0], conf[0].cpu().numpy().astype(np.float64)

    def locate(self, image: np.ndarray, passes: int = 2) -> tuple[np.ndarray, float]:
        """The face's corners (4, 2) in pixels, in the face's order (turning corner third), and a confidence
        in 0-1. The first look is at the whole photo; each further pass looks again, closer, round the face."""
        s = LOCATOR_INPUT
        square, scale, dx, dy = letterbox(image, s)
        pts, conf = self._corners(square)
        quad = (pts - np.array([dx, dy])) / scale
        confidence = float(conf.min())
        for _ in range(max(0, passes - 1)):
            lo, hi = quad.min(0), quad.max(0)
            side = 1.9 * float(max(hi - lo))           # room for pieces standing out of the face
            if side < 8:
                break
            centre = (lo + hi) / 2
            k = s / side
            m = np.array([[k, 0.0, s / 2 - k * centre[0]], [0.0, k, s / 2 - k * centre[1]]])
            window = cv2.warpAffine(image, m, (s, s), flags=cv2.INTER_AREA if k < 0.7 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            pts, conf = self._corners(window)
            refined = (pts - m[:, 2]) / k
            if float(conf.min()) >= 0.5 * confidence and np.abs(refined - quad).max() < 0.35 * quad_side(quad):
                quad, confidence = refined, max(confidence, float(conf.min()))
        return quad, confidence

    # -- what a face shows -----------------------------------------------------------------------
    @torch.no_grad()
    def read(self, image: np.ndarray, quad: np.ndarray, looks: int = 3, jitter: float = 0.006) -> np.ndarray:
        """(29, 8) probabilities for the face inside ``quad`` (face order). The face is cut out ``looks``
        times with its corners nudged a little differently, and the readings are averaged."""
        model = self._need("reader")
        image, quad = prepare_photo(image, np.asarray(quad, dtype=np.float64), FACE_PX)
        rng = np.random.default_rng(12345)
        side = quad_side(quad)
        crops, ctx = [], []
        for i in range(max(1, looks)):
            q = quad if i == 0 else quad + rng.normal(0.0, jitter * side, quad.shape)
            c, x = cut_crops(image, q, self.geom)
            crops.append(c)
            ctx.append(x)
        cr = torch.from_numpy(np.stack(crops)).permute(0, 1, 4, 2, 3).contiguous().to(self.device)
        cx = torch.from_numpy(np.stack(ctx)).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits = model(cr, cx)
        return model.probabilities(logits).mean(0).cpu().numpy().astype(np.float64)

    # -- six faces -> a state ----------------------------------------------------------------------
    def find_faces(self, readings: list[np.ndarray], candidates: int = CANDIDATES, colours: list[ColourLayout] | None = None) -> FaceAnswer:
        """Which face each photo shows, and the cube's colour layout: one of ``colours``, or (by default)
        the model's own unless another is clearly better (``other_colours``)."""
        if colours:
            return self._find(readings, list(colours), candidates)
        usual = self._find(readings, self.layout.colour_layouts[:1], candidates)
        return self.other_colours(readings, usual, candidates)

    def other_colours(self, readings: list[np.ndarray], usual: FaceAnswer, candidates: int = CANDIDATES) -> FaceAnswer:
        """``usual`` (the answer for the model's own colour layout), or the answer for another layout when
        that one is believed: it must be legal and clear (margin at least ``DOUBT_NATS``), and explain the
        photos better than ``usual`` by its prior (``COLOUR_PRIOR``) plus ``DOUBT_NATS``. A cube stickered
        the usual way is so never read as another layout because of a poor photo; when no other layout is
        believed the closest ones are only offered (``alternatives``)."""
        layouts = self.layout.colour_layouts
        anyway = self._find(readings, layouts, candidates)
        believed = (not anyway.colours.is_default and anyway.decode.legal and anyway.margin >= DOUBT_NATS
                    and anyway.decode.loglik - COLOUR_PRIOR[anyway.colours.kind] >= usual.decode.loglik + DOUBT_NATS)
        if believed:
            gap = anyway.decode.loglik - COLOUR_PRIOR[anyway.colours.kind] - usual.decode.loglik
            anyway.margin = float(min(anyway.margin, gap))
            if all(not c.is_default for c, _ in anyway.alternatives):
                anyway.alternatives = [(layouts[0], float(gap))] + anyway.alternatives[:ALTERNATIVES - 1]
            return anyway
        others = [(c, d) for c, d in anyway.alternatives if not c.is_default]
        if not anyway.colours.is_default:
            others.insert(0, (anyway.colours, float(usual.decode.loglik - anyway.decode.loglik + COLOUR_PRIOR[anyway.colours.kind])))
        usual.alternatives = others[:ALTERNATIVES]
        return usual

    def _find(self, readings: list[np.ndarray], layouts: list[ColourLayout], candidates: int) -> FaceAnswer:
        lay = self.layout
        prior = [0.0 if len(layouts) == 1 else COLOUR_PRIOR[c.kind] for c in layouts]
        evid = [evidence(r) for r in readings]
        top, rest = self.search.rank(evid, layouts, prior, candidates)
        sym = lay.symmetries

        def key(li: int, orient: np.ndarray) -> tuple:
            """The same answer: same layout, and the same arrangement up to a whole-puzzle turn."""
            return (li, min(tuple(lay.turned(orient, g).tolist()) for g in sym))

        looked = []                                                     # (candidate, orient, reachable)
        for c in top:
            orient, _, _ = self.search.arrangement(evid, layouts[c.layout], c.faces)
            looked.append((c, orient, reachable(orient, lay)))
        best = next((x for x in looked if x[2]), None)

        def full(c) -> TangramDecode:
            return decode_logp(self.search.arrangement(evid, layouts[c.layout], c.faces)[2], lay)

        if best is not None:
            chosen = best[0]
            d = full(chosen)
            score = chosen.score
        else:
            # Nothing among the first candidates is reachable as read: decode the best few in full (with repairs).
            tried, seen = [], set()
            for c, orient, _ in looked:
                k = key(c.layout, orient)
                if k in seen:
                    continue
                seen.add(k)
                tried.append((c, full(c)))
                if len(tried) >= 6:
                    break
            tried.sort(key=lambda t: (not t[1].legal, -(t[1].loglik - (t[0].loglik - t[0].score))))
            chosen, d = tried[0]
            score = d.loglik - (chosen.loglik - chosen.score)
        k0 = key(chosen.layout, d.orient)
        runner = next((c.score for c, o, ok in looked if ok and key(c.layout, o) != k0), None)
        if runner is None:
            runner = max(rest, next((c.score for c, o, _ in looked if key(c.layout, o) != k0), -np.inf))
        alternatives: list[tuple[ColourLayout, float]] = []
        for c, _, _ in looked:
            other = layouts[c.layout]
            if c.layout != chosen.layout and all(other != a for a, _ in alternatives):
                alternatives.append((other, float(score - c.score)))
        alternatives = alternatives[:ALTERNATIVES]
        return FaceAnswer(tuple(chosen.faces), d, float(score - runner), layouts[chosen.layout], alternatives)

    def scan(self, images: list[np.ndarray], quads: list[np.ndarray] | None = None, faces: tuple[int, ...] | None = None,
             retry_turns: bool = True, colours: ColourLayout | None = None) -> TangramScan:
        """Six photos -> the legal state they show. With ``faces`` the photos are taken to show those faces;
        with ``colours`` the cube is taken to have that colour layout."""
        if len(images) != 6:
            raise ValueError("a scan is six photos, one of each face")
        confidence = [1.0] * 6
        if quads is None:
            located = [self.locate(im) for im in images]
            quads = [q for q, _ in located]
            confidence = [c for _, c in located]
        quads = [np.asarray(q, dtype=np.float64) for q in quads]
        first_quads = [q.copy() for q in quads]
        readings = [self.read(im, q) for im, q in zip(images, quads)]
        if faces is not None:
            cl = colours or ColourLayout.default()
            evid = [cl.model_columns(evidence(r)) for r in readings]
            d = decode_logp(assemble(evid, tuple(faces), self.geom, self.layout.n_positions), self.layout)
            return TangramScan(d, tuple(faces), quads, confidence, readings, colours=cl)
        # The model's own colour layout first (or the one asked for), exactly as before colour layouts were
        # looked at: so the photos that were turned and retried are the same for a usually stickered cube.
        first = list(readings)
        ans = self._find(readings, [colours or self.layout.colour_layouts[0]], CANDIDATES)
        returned: dict[int, int] = {}
        if retry_turns and (not ans.decode.legal or ans.margin < DOUBT_NATS):
            # Maybe the locator took the wrong corner for a turning corner: try each photo turned, keep what helps.
            keep = [ans.colours]
            for i in np.argsort(confidence):
                for k in (1, 3, 2):
                    q = np.roll(quads[i], -k, axis=0)
                    r = self.read(images[i], q)
                    trial = readings[:i] + [r] + readings[i + 1:]
                    a2 = self.find_faces(trial, colours=keep)
                    d, d2 = ans.decode, a2.decode
                    if (d2.legal and not d.legal) or (d2.legal == d.legal and d2.loglik > d.loglik + 5.0):
                        quads[i], readings, ans = q, trial, a2
                        returned[int(i)] = k
                        break
                if ans.decode.legal and ans.margin >= DOUBT_NATS:
                    break
        if colours is None:
            # Then any other colour layout, on the photos as first read (a turned retry fitted the usual one).
            other = self.other_colours(first, ans)
            if not other.colours.is_default:
                ans, readings, quads, returned = other, first, [np.asarray(q, dtype=np.float64) for q in first_quads], {}
            else:
                ans = other
        else:
            # The person chose this layout: still offer the others that fit (so they can go back).
            anyway = self._find(first, self.layout.colour_layouts, CANDIDATES)
            offered = [(anyway.colours, 0.0)] + list(anyway.alternatives)
            ans.alternatives = [(c, float(ans.decode.loglik - (anyway.decode.loglik - d))) for c, d in offered if c != colours][:ALTERNATIVES]
            if all(not c.is_default for c, _ in ans.alternatives) and not colours.is_default:
                ans.alternatives = [(self.layout.colour_layouts[0], 0.0)] + ans.alternatives[:ALTERNATIVES - 1]
        return TangramScan(ans.decode, ans.faces, quads, confidence, readings, ans.margin, returned, ans.colours, ans.alternatives)

    # -- for showing -------------------------------------------------------------------------------
    def flatten(self, image: np.ndarray, quad: np.ndarray, size: int = 448, half: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        """The face flattened (turning corner at the bottom-right) and where its 29 positions appear in it."""
        image, quad = prepare_photo(image, np.asarray(quad, dtype=np.float64), face_px=size)
        h, w = image.shape[:2]
        pose = face_pose(quad, w, h)
        return rectify(image, quad, size, half), flat_points(quad, self.geom.centre, w, h, size, half, pose)


assert N_CLASSES == HIDDEN + 1
