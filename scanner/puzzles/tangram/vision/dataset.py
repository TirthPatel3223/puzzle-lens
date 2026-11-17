"""The synthetic photos written by ``tools/gen_dataset.py --puzzle tangram``, as training examples.

A dataset folder holds ``images/*.jpg`` and ``meta.jsonl``: per picture the face photographed, its four
corners in the picture (the locator's order: top-left, top-right, bottom-right = turning corner,
bottom-left), what every one of the 120 positions shows, and which of them the camera really sees.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scanner.vision import augment
from scanner.vision.common import LOCATOR_INPUT, quad_side

from ..layout import EMPTY, load_layout
from .geometry import CROP, HIDDEN, FaceGeometry, cut_crops, face_geometry

_FIELDS = ("files", "seeds", "faces", "quads", "content", "visible", "sizes", "scan", "shot", "setting", "kind", "paint")


class Photos:
    """The index of a dataset folder, as arrays (cheap to hand to worker processes)."""

    def __init__(self, root: str | Path, limit: int | None = None) -> None:
        self.root = Path(root)
        cache = self.root / "meta.npz"
        meta = self.root / "meta.jsonl"
        z = np.load(cache, allow_pickle=False) if cache.exists() and cache.stat().st_mtime >= meta.stat().st_mtime else None
        if z is not None and all(name in z for name in _FIELDS):
            for name in _FIELDS:
                setattr(self, name, z[name])
        else:
            rows = []
            with meta.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            if rows and rows[0].get("puzzle") != "tangram-cube-extreme":
                raise ValueError(f"{meta} is not a Tangram Cube Extreme dataset")
            rows.sort(key=lambda m: m["seed"])
            n_pos = len(rows[0]["content"]) if rows else 120
            self.files = np.array([m["file"] for m in rows])
            self.seeds = np.array([m["seed"] for m in rows], dtype=np.int64)
            self.faces = np.array([m["face"] for m in rows], dtype=np.int64)
            self.sizes = np.array([m["size"] for m in rows], dtype=np.int64)
            self.quads = np.array([m["corners"] for m in rows], dtype=np.float64) * self.sizes[:, None, None]
            self.content = np.array([m["content"] for m in rows], dtype=np.uint8)                   # (N, 120)
            self.visible = np.zeros((len(rows), n_pos), dtype=bool)                                   # (N, 120)
            for i, m in enumerate(rows):
                self.visible[i, m["visible"]] = True
            self.scan = np.array([m.get("scan", -1) for m in rows], dtype=np.int64)
            self.shot = np.array([m.get("shot", -1) for m in rows], dtype=np.int64)
            self.setting = np.array([m.get("setting", "") for m in rows])
            self.kind = np.array([m.get("stateKind", "") for m in rows])
            # The puzzle's colour layout (``content`` is what it shows: the model's colour c as paint[c]).
            self.paint = np.array([m.get("paint", list(range(6))) for m in rows], dtype=np.int64).reshape(len(rows), 6)
            np.savez(cache, **{name: getattr(self, name) for name in _FIELDS})
        if limit is not None:
            for name in _FIELDS:
                setattr(self, name, getattr(self, name)[:limit])

    def __len__(self) -> int:
        return len(self.files)

    def image(self, i: int) -> np.ndarray:
        """The picture as RGB uint8."""
        bgr = cv2.imread(str(self.root / str(self.files[i])), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / str(self.files[i]))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def face_labels(self, i: int, geom: FaceGeometry) -> np.ndarray:
        """What the reader should say for each of the photo's face positions: colour, EMPTY, or HIDDEN."""
        pos = geom.position_of[int(self.faces[i])]
        c = self.content[i, pos].astype(np.int64)
        hidden = (c != EMPTY) & ~self.visible[i, pos]
        return np.where(hidden, HIDDEN, c)


class _Seeded(Dataset):
    """Training examples draw fresh randomness every time; evaluation examples are the same every time."""

    def __init__(self, photos: Photos, train: bool) -> None:
        self.photos = photos
        self.train = train
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.photos)

    def rng(self, i: int) -> np.random.Generator:
        if not self.train:
            return np.random.default_rng(int(self.photos.seeds[i]) + 77)
        if self._rng is None:                       # one stream per worker process, seeded by the OS
            self._rng = np.random.default_rng()
        return self._rng


class ReaderData(_Seeded):
    """One face photo: its position crops, the context picture, and what each position shows.

    The corners are jittered, because neither the locator nor a person dragging handles puts them
    exactly on the face's corners (and a turned-out piece can carry a corner away)."""

    def __init__(self, photos: Photos, train: bool, jitter: float = 0.022, eval_jitter: float = 0.008, strength: float = 1.0) -> None:
        super().__init__(photos, train)
        self.jitter = jitter
        self.eval_jitter = eval_jitter
        self.strength = strength
        self.geom = face_geometry(load_layout())

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = self.rng(i)
        img = self.photos.image(i)
        quad = self.photos.quads[i].copy()
        side = quad_side(quad)
        if self.train:
            img = augment.camera(img, rng, self.strength)
            sigma = side * (rng.uniform(0, self.jitter) if rng.random() < 0.85 else rng.uniform(self.jitter, 2 * self.jitter))
        else:
            sigma = side * self.eval_jitter
        quad = quad + rng.normal(0.0, sigma, quad.shape)
        crops, ctx = cut_crops(img, quad, self.geom, CROP)
        return {
            "crops": torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous(),     # (29, 3, S, S) uint8
            "context": torch.from_numpy(ctx).permute(2, 0, 1).contiguous(),        # (3, C, C) uint8
            "labels": torch.from_numpy(self.photos.face_labels(i, self.geom)),     # (29,)
        }


ASPECTS = (1.0, 1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16)


class LocatorData(_Seeded):
    """One picture, framed as a phone or webcam might frame it, and where the face's corners are in it
    (in the face's own order: the network has to find the turning corner, not just the outline)."""

    def __init__(self, photos: Photos, train: bool, strength: float = 1.0) -> None:
        super().__init__(photos, train)
        self.strength = strength

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = self.rng(i)
        img = self.photos.image(i)
        quad = self.photos.quads[i].copy()
        s = LOCATOR_INPUT
        aspect = ASPECTS[int(rng.integers(len(ASPECTS)))] if self.train else 1.0
        lo, hi = quad.min(0), quad.max(0)
        # Turned-out pieces reach past the face: frame a little more than the face itself.
        lo, hi = lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo)
        bw, bh = hi - lo
        fill = rng.uniform(0.3, 0.95) if self.train else rng.uniform(0.4, 0.85)
        wh = max(bh / fill, bw / (fill * aspect))
        ww = aspect * wh
        centre = (lo + hi) / 2 + np.array([rng.uniform(-1, 1) * (ww - bw) / 2 * 0.9, rng.uniform(-1, 1) * (wh - bh) / 2 * 0.9])
        angle = np.deg2rad(rng.normal(0, 8.0)) if self.train else 0.0
        scale = s / max(ww, wh)
        c, sn = np.cos(angle) * scale, np.sin(angle) * scale
        m = np.array([[c, -sn, s / 2 - (c * centre[0] - sn * centre[1])],
                      [sn, c, s / 2 - (sn * centre[0] + c * centre[1])]])
        out = cv2.warpAffine(img, m, (s, s), flags=cv2.INTER_AREA if scale < 0.7 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if self.train:
            out = augment.camera(out, rng, self.strength)
        barx, bary = round((s - ww * scale) / 2), round((s - wh * scale) / 2)
        if barx > 0:
            out[:, :barx] = 0
            out[:, s - barx:] = 0
        if bary > 0:
            out[:bary] = 0
            out[s - bary:] = 0
        q = quad @ m[:, :2].T + m[:, 2]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).contiguous(),   # (3, s, s) uint8
            "corners": torch.from_numpy((q / s).astype(np.float32)),                               # (4, 2) in 0-1, face order
        }
