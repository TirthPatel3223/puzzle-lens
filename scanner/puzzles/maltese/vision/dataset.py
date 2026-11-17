"""The synthetic photos written by tools/gen_dataset.py, as training examples for the two networks.

A dataset folder holds ``images/*.jpg`` and ``meta.jsonl`` (one line per picture: the face
photographed, its four corners in the picture, and what the whole cube shows).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scanner.vision import augment
from scanner.vision.common import LOCATOR_INPUT, letterbox

from .geometry import CROP, cut_crops, face_cells, picture_order, quad_side

#: Side of the locator's heatmaps.
HEATMAP = 80


class Photos:
    """The index of a dataset folder, as arrays (cheap to hand to worker processes)."""

    def __init__(self, root: str | Path, limit: int | None = None) -> None:
        self.root = Path(root)
        cache = self.root / "meta.npz"
        meta = self.root / "meta.jsonl"
        if cache.exists() and cache.stat().st_mtime >= meta.stat().st_mtime:
            z = np.load(cache, allow_pickle=False)
            self.files, self.seeds, self.faces, self.quads = z["files"], z["seeds"], z["faces"], z["quads"]
            self.cells, self.gear_colour, self.gear_quarter, self.sizes = z["cells"], z["gear_colour"], z["gear_quarter"], z["sizes"]
            self.scan, self.setting = z["scan"], z["setting"]
            # (a cache written before colour layouts existed: every cube is stickered the usual way)
            self.paint = z["paint"] if "paint" in z else np.tile(np.arange(6, dtype=np.int64), (len(self.files), 1))
        else:
            rows = []
            with meta.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            rows.sort(key=lambda m: m["seed"])
            self.files = np.array([m["file"] for m in rows])
            self.seeds = np.array([m["seed"] for m in rows], dtype=np.int64)
            self.faces = np.array([m["face"] for m in rows], dtype=np.int64)
            self.sizes = np.array([m["size"] for m in rows], dtype=np.int64)
            self.quads = np.array([m["quad"] for m in rows], dtype=np.float64) * self.sizes[:, None, None]
            self.cells = np.array([m["cells"] for m in rows], dtype=np.uint8)                  # (N, 96)
            self.gear_colour = np.array([m["gearColour"] for m in rows], dtype=np.uint8)       # (N, 24)
            # The cube's colour layout (cells and gear colours are what it shows: the model's colour c as paint[c]).
            self.paint = np.array([m.get("paint", list(range(6))) for m in rows], dtype=np.int64).reshape(len(rows), 6)
            self.gear_quarter = np.array([m["gearQuarter"] for m in rows], dtype=np.uint8)     # (N, 24)
            self.scan = np.array([m.get("scan", -1) for m in rows], dtype=np.int64)
            self.setting = np.array([m.get("setting", "") for m in rows])
            np.savez(cache, paint=self.paint, files=self.files, seeds=self.seeds, faces=self.faces, quads=self.quads, cells=self.cells,
                     gear_colour=self.gear_colour, gear_quarter=self.gear_quarter, sizes=self.sizes, scan=self.scan, setting=self.setting)
        if limit is not None:
            for name in ("files", "seeds", "faces", "quads", "cells", "gear_colour", "gear_quarter", "sizes", "scan", "setting", "paint"):
                setattr(self, name, getattr(self, name)[:limit])

    def __len__(self) -> int:
        return len(self.files)

    def image(self, i: int) -> np.ndarray:
        """The picture as RGB uint8."""
        bgr = cv2.imread(str(self.root / str(self.files[i])), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / str(self.files[i]))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def face_labels(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The photographed face's 16 cells, 4 gear colours and 4 gear quarters, in its own orientation."""
        f = int(self.faces[i])
        return self.cells[i, 16 * f:16 * f + 16], self.gear_colour[i, 4 * f:4 * f + 4], self.gear_quarter[i, 4 * f:4 * f + 4]


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
    """One face: its 13 slot crops, the context picture, and what each slot shows.

    The corners are jittered, because neither the locator nor a person dragging handles puts
    them exactly on the face's corners.
    """

    def __init__(self, photos: Photos, train: bool, jitter: float = 0.022, eval_jitter: float = 0.008, strength: float = 1.0) -> None:
        super().__init__(photos, train)
        self.jitter = jitter
        self.eval_jitter = eval_jitter
        self.strength = strength
        self.cells = face_cells()

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
        crops, ctx = cut_crops(img, quad, CROP)
        cells, gear_colour, gear_quarter = self.photos.face_labels(i)
        fc = self.cells
        return {
            "crops": torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous(),          # (13, 3, S, S) uint8
            "context": torch.from_numpy(ctx).permute(2, 0, 1).contiguous(),             # (3, C, C) uint8
            "star": torch.from_numpy(cells[fc.star].astype(np.int64)),                  # (4,)
            "gear_colour": torch.from_numpy(gear_colour.astype(np.int64)),              # (4,)
            "gear_quarter": torch.from_numpy(gear_quarter.astype(np.int64)),            # (4,)
            "edge": torch.from_numpy(cells[fc.edge].astype(np.int64)),                  # (4, 2)
            "centre": torch.from_numpy(cells[fc.centre].astype(np.int64)),              # (4,)
        }


ASPECTS = (1.0, 1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16)


class LocatorData(_Seeded):
    """One picture, framed as a phone or webcam might frame it, and where the face's corners are in it."""

    def __init__(self, photos: Photos, train: bool, strength: float = 1.0) -> None:
        super().__init__(photos, train)
        self.strength = strength

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = self.rng(i)
        img = self.photos.image(i)
        quad = self.photos.quads[i].copy()
        s = LOCATOR_INPUT

        # A window of a random shape around the face, then letterboxed into the square the network sees.
        aspect = ASPECTS[int(rng.integers(len(ASPECTS)))] if self.train else 1.0
        lo, hi = quad.min(0), quad.max(0)
        bw, bh = hi - lo
        fill = rng.uniform(0.22, 0.9) if self.train else rng.uniform(0.3, 0.8)
        wh = max(bh / fill, bw / (fill * aspect))
        ww = aspect * wh
        centre = (lo + hi) / 2 + np.array([rng.uniform(-1, 1) * (ww - bw) / 2 * 0.9, rng.uniform(-1, 1) * (wh - bh) / 2 * 0.9])
        angle = np.deg2rad(rng.normal(0, 5.0)) if self.train else 0.0
        scale = s / max(ww, wh)
        c, sn = np.cos(angle) * scale, np.sin(angle) * scale
        m = np.array([[c, -sn, s / 2 - (c * centre[0] - sn * centre[1])],
                      [sn, c, s / 2 - (sn * centre[0] + c * centre[1])]])
        out = cv2.warpAffine(img, m, (s, s), flags=cv2.INTER_AREA if scale < 0.7 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if self.train:
            out = augment.camera(out, rng, self.strength)
        # The bars of the letterbox are black, as they are when a real photo is letterboxed.
        barx, bary = round((s - ww * scale) / 2), round((s - wh * scale) / 2)
        if barx > 0:
            out[:, :barx] = 0
            out[:, s - barx:] = 0
        if bary > 0:
            out[:bary] = 0
            out[s - bary:] = 0
        q = quad @ m[:, :2].T + m[:, 2]
        q, _ = picture_order(q)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).contiguous(),   # (3, s, s) uint8
            "corners": torch.from_numpy((q / s).astype(np.float32)),                               # (4, 2) in 0-1
        }
