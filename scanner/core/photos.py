"""Photos as the server receives and returns them, and scans waiting to be checked.

Every puzzle's scan keeps its photos in ``<real>/pending/<id>/`` until the person has checked the
result; a confirmed scan moves to ``<real>/scans/<id>/`` with its labels (labelled real data for
measuring and fine-tuning the networks). ``<real>`` is ``scanner.paths.real_dir(puzzle)``.
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

#: Photos are shrunk so their longer side is at most this (a phone photo is far larger than the networks need).
MAX_SIDE = 2400
#: Longer side of the previews sent back for the upload grid.
PREVIEW_SIDE = 520
#: Scans waiting to be confirmed; older ones are deleted.
KEEP_PENDING = 12

try:                                   # iPhone photos (HEIC), when pillow-heif is installed
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:                    # pragma: no cover
    pass


def read_photo(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    """An uploaded photo as RGB uint8, the right way up (phones store the rotation beside the pixels)."""
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError) as err:
        raise HTTPException(415, "That file is not a picture this tool can read (JPEG, PNG, WebP or HEIC).") from err
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    return np.asarray(im)


def jpeg_data_url(rgb: np.ndarray, quality: int = 86) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode a picture")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def preview(rgb: np.ndarray, side: int = PREVIEW_SIDE) -> str:
    f = side / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else rgb
    return jpeg_data_url(small, 82)


def fractions(points: np.ndarray, image: np.ndarray, digits: int = 5) -> list:
    """Pixel points as fractions of the picture (the browser shows it at its own size)."""
    h, w = image.shape[:2]
    return (np.asarray(points, dtype=np.float64) / np.array([w, h])).round(digits).tolist()


class PendingScans:
    """Scans kept until they are checked, and the labelled ones they become."""

    def __init__(self, real: Path, keep: int = KEEP_PENDING) -> None:
        self.real = Path(real)
        self.keep = keep

    @property
    def pending(self) -> Path:
        return self.real / "pending"

    @property
    def kept(self) -> Path:
        return self.real / "scans"

    def labelled_count(self) -> int:
        return sum(1 for p in self.kept.iterdir() if p.is_dir()) if self.kept.is_dir() else 0

    def new(self, images: list[np.ndarray], meta: dict, arrays: dict[str, np.ndarray] | None = None) -> str:
        """Save the photos and what was read from them; returns the scan id."""
        scan_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        folder = self.pending / scan_id
        folder.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            cv2.imwrite(str(folder / f"shot{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
        if arrays:
            np.savez_compressed(folder / "reading.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
        (folder / "scan.json").write_text(json.dumps({"id": scan_id, **meta}, indent=1), encoding="utf-8")
        for old in sorted(p for p in self.pending.iterdir() if p.is_dir())[: -self.keep]:
            shutil.rmtree(old, ignore_errors=True)
        return scan_id

    def folder(self, scan_id: str) -> Path:
        if not scan_id.replace("-", "").isalnum():
            raise HTTPException(422, "bad scan id")
        folder = self.pending / scan_id
        if not folder.is_dir():
            raise HTTPException(404, "That scan is no longer waiting (it was confirmed already, or too many newer scans came after it).")
        return folder

    def read_meta(self, scan_id: str) -> dict:
        return json.loads((self.folder(scan_id) / "scan.json").read_text(encoding="utf-8"))

    def confirm(self, scan_id: str, meta: dict) -> Path:
        """Write the labelled metadata and move the scan to the kept ones."""
        folder = self.folder(scan_id)
        (folder / "scan.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        kept = self.kept / scan_id
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(kept))
        return kept
