"""Where things are in a face-on photo of the Tangram Cube Extreme, and how one is cut into crops.

**Why face-on.** Straight down the T-B diagonal, a sticker that a 60° turn has tilted out of the cube
looks exactly like the flat sticker in its place (10 of the 12 such pairs overlap 98-100%). Face-on,
the flat one is seen square and the tilted one at 48°, somewhere else: every pair is told apart, and in
random scrambles no piece is hidden from all six photos. So a scan is six photos, one per face, as for
the Maltese Gear Cube. A face is the middle layer's: it never changes shape, so every face always shows
some flat stickers to square the camera to.

**Face frame.** x right, y up as the face is seen from outside, z out of the face; the face sits at
z = 0 and spans [-1, 1]². It is turned so that the face's turning corner (the T or B corner piece, which
never leaves its corner) is at the bottom-right. The locator finds the face's corners in that order:
top-left, top-right, bottom-right (the turning corner), bottom-left, so it also says which way up the
photo is. Corners a turned-out piece has carried away are still found where the face's corners are.

**Face positions.** Every face shows the same 29 positions in its own frame (11 flat, 18 turned out at
48°): the D3 symmetry of the cuts carries any face's frame onto any other's. ``FaceGeometry`` lists them
and, per face, which of the puzzle's 120 positions each one is.

**Crops.** From the four corners (all in the face plane) and a guess at the lens, the camera's pose is
solved (IPPE), so any position, flat or standing out, can be projected. Each position is cut out as a
square in its own sticker's plane, so a sticker looks the same wherever and however tilted it is. A
wrong guess at the lens matters only for points off the face plane, and only slightly (about a millimetre
at a normal distance).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from ..layout import TangramLayout, load_layout

#: The face's corners in its own frame (z = 0 is the face plane), in the locator's order.
FACE_CORNERS = np.array([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]])
#: Side of every position crop, pixels.
CROP = 72
#: Side of the context picture (the whole face, small), pixels, and how much of the face plane it shows.
CONTEXT = 128
CONTEXT_HALF = 1.5
#: The lens assumed when nothing better is known: this field of view across the picture's longer side.
ASSUMED_FOV_DEG = 60.0
N_CLASSES = 8                   # colours 0-5, 6 empty, 7 hidden (a sticker is there but something is in front)
HIDDEN = 7


@dataclass(frozen=True)
class FaceGeometry:
    """The 29 positions a face shows, in the face's frame, and which puzzle position each one is per face."""

    centre: np.ndarray        # (n, 3) face frame
    normal: np.ndarray        # (n, 3)
    u: np.ndarray             # (n, 3) crop's right, in the sticker plane
    w: np.ndarray             # (n, 3) crop's up
    half: np.ndarray          # (n,)   crop half-width, face units
    flat: np.ndarray          # (n,)   lies in the face plane, facing straight out
    position_of: np.ndarray   # (6, n) puzzle position of face position j in the photo of face f

    @property
    def n(self) -> int:
        return len(self.centre)

    def evidence_index(self) -> list[list[tuple[int, int]]]:
        """For each puzzle position, the (face, face position) pairs that show it."""
        out: list[list[tuple[int, int]]] = [[] for _ in range(int(self.position_of.max()) + 1)]
        for f in range(self.position_of.shape[0]):
            for j, p in enumerate(self.position_of[f]):
                out[int(p)].append((f, j))
        return out


def _key(v: np.ndarray) -> tuple:
    return tuple(np.round(v, 4) + 0.0)


@lru_cache(maxsize=2)
def _face_geometry(layout_id: int) -> FaceGeometry:
    lay = _LAYOUTS[layout_id]
    rows = lay.face_rows                                   # (6, 3, 3) right, up, normal
    per_face = []
    for f in range(6):
        r = rows[f]
        sel = np.nonzero(lay.normal @ r[2] >= lay.face_min_cos - 1e-9)[0]
        c = lay.centre[sel] @ r.T - np.array([0.0, 0.0, 1.0])
        n = lay.normal[sel] @ r.T
        polys = [lay.polygons[p] @ r.T - np.array([0.0, 0.0, 1.0]) for p in sel]
        per_face.append((sel, c, n, polys))
    # Face 0's positions, in a fixed order: flat ones first, then top to bottom, left to right.
    sel0, c0, n0, polys0 = per_face[0]
    order = sorted(range(len(sel0)), key=lambda j: (n0[j, 2] < 0.999, -round(c0[j, 1], 3), round(c0[j, 0], 3), round(c0[j, 2], 3)))
    centre, normal = c0[order], n0[order]
    key0 = [(_key(centre[j]), _key(normal[j])) for j in range(len(order))]
    position_of = np.zeros((6, len(order)), dtype=np.int64)
    for f, (sel, c, n, _) in enumerate(per_face):
        where = {(_key(c[j]), _key(n[j])): int(sel[j]) for j in range(len(sel))}
        if len(where) != len(key0) or any(k not in where for k in key0):
            raise RuntimeError(f"face {lay.face_ids[f]} does not show the same positions as face {lay.face_ids[0]}")
        position_of[f] = [where[k] for k in key0]
    # Crop frames: right is the face's right laid into the sticker plane; up completes it.
    u = np.empty_like(centre)
    w = np.empty_like(centre)
    half = np.empty(len(order))
    for j, k in enumerate(order):
        nj = normal[j]
        x = np.array([1.0, 0.0, 0.0]) - nj * nj[0]
        if np.linalg.norm(x) < 1e-6:                       # the sticker faces sideways: use the face's up instead
            x = np.array([0.0, 1.0, 0.0]) - nj * nj[1]
        u[j] = x / np.linalg.norm(x)
        w[j] = np.cross(nj, u[j])
        poly = polys0[k]
        reach = np.linalg.norm(poly - poly.mean(0), axis=1).max()
        # The middle of a sticker and a margin round it: long stickers are cut short, which is fine.
        half[j] = float(np.clip(0.8 * reach + 0.05, 0.2, 0.42))
    flat = normal[:, 2] > 0.999
    return FaceGeometry(centre, normal, u, w, half, flat, position_of)


_LAYOUTS: dict[int, TangramLayout] = {}


def face_geometry(layout: TangramLayout | None = None) -> FaceGeometry:
    layout = layout or load_layout()
    _LAYOUTS[id(layout)] = layout
    return _face_geometry(id(layout))


# ---------------------------------------------------------------------------------------- the camera

def camera_matrix(width: int, height: int, fov_deg: float = ASSUMED_FOV_DEG) -> np.ndarray:
    f = (max(width, height) / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Pose:
    """A camera looking at a face: project(points in the face frame) -> pixels."""

    K: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray

    def project(self, points: np.ndarray) -> np.ndarray:
        pts, _ = cv2.projectPoints(np.asarray(points, dtype=np.float64).reshape(-1, 3), self.rvec, self.tvec, self.K, None)
        return pts.reshape(-1, 2)


def face_pose(quad: np.ndarray, width: int, height: int, fov_deg: float = ASSUMED_FOV_DEG) -> Pose:
    """The camera's pose from the face's four corners (pixels, the locator's order)."""
    K = camera_matrix(width, height, fov_deg)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(FACE_CORNERS, np.asarray(quad, dtype=np.float64).reshape(4, 1, 2), K, None,
                                                 flags=cv2.SOLVEPNP_IPPE)
    if not ok or not len(rvecs):
        raise ValueError("no camera pose fits those corners")
    # Of IPPE's two solutions take the one with the face in front of the camera, facing it.
    best = None
    for rv, tv, e in zip(rvecs, tvecs, np.ravel(errs) if errs is not None else [0.0] * len(rvecs)):
        R, _ = cv2.Rodrigues(rv)
        facing = (R @ np.array([0.0, 0.0, 1.0])) @ (-(tv.ravel())) > 0     # the face normal points at the camera
        score = float(e) + (0.0 if facing and tv.ravel()[2] > 0 else 1e6)
        if best is None or score < best[0]:
            best = (score, rv, tv)
    return Pose(K, best[1].ravel(), best[2].ravel())


# ------------------------------------------------------------------------------------------- crops

def _square(centre: np.ndarray, u: np.ndarray, w: np.ndarray, half: float) -> np.ndarray:
    """Corners of a square in a plane (top-left, top-right, bottom-right, bottom-left as the crop shows it)."""
    return np.array([centre - half * u + half * w, centre + half * u + half * w, centre + half * u - half * w, centre - half * u - half * w])


def _pixel_square(size: int) -> np.ndarray:
    """Where those corners are in a crop of ``size`` pixels: OpenCV puts pixel i's centre at i, so the
    square's edges are the outer edges of the outer pixels, at -0.5 and size - 0.5."""
    a, b = -0.5, size - 0.5
    return np.array([[a, a], [b, a], [b, b], [a, b]], dtype=np.float32)


def cut_crops(image: np.ndarray, quad: np.ndarray, geom: FaceGeometry, size: int = CROP, context: int = CONTEXT,
              pose: Pose | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The position crops (n, size, size, 3) and the context picture (context, context, 3) of one face photo.
    ``quad`` is the face's corners in pixels, in the locator's order."""
    h, w = image.shape[:2]
    pose = pose or face_pose(quad, w, h)
    dst = _pixel_square(size)
    crops = np.empty((geom.n, size, size, 3), dtype=image.dtype)
    corners = np.stack([_square(geom.centre[j], geom.u[j], geom.w[j], geom.half[j]) for j in range(geom.n)])   # (n, 4, 3)
    src = pose.project(corners.reshape(-1, 3)).reshape(geom.n, 4, 2).astype(np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    for j in range(geom.n):
        H = cv2.getPerspectiveTransform(dst, src[j])
        crops[j] = cv2.warpPerspective(image, H, (size, size), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    ctx = rectify(image, quad, context, CONTEXT_HALF)
    return crops, ctx


def plane_homography(quad: np.ndarray, size: int, half: float) -> np.ndarray:
    """3x3 map from the pixels of a ``size`` square showing [-half, half]² of the face plane to the photo."""
    to_photo = cv2.getPerspectiveTransform(FACE_CORNERS[:, :2].astype(np.float32), np.asarray(quad, dtype=np.float32))
    m = np.array([[2 * half / size, 0.0, -half + half / size], [0.0, -2 * half / size, half - half / size], [0.0, 0.0, 1.0]])
    return to_photo @ m


def rectify(image: np.ndarray, quad: np.ndarray, size: int = 448, half: float = 1.5) -> np.ndarray:
    """The face plane flattened, ``half`` face units from the centre to each border, turning corner at the bottom-right."""
    return cv2.warpPerspective(image, plane_homography(quad, size, half), (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def flat_points(quad: np.ndarray, points: np.ndarray, width: int, height: int, size: int = 448, half: float = 1.5,
                pose: Pose | None = None) -> np.ndarray:
    """Where points of the face frame (3D) appear in ``rectify``'s picture: projected into the photo with the
    camera's pose, then through the face plane's rectification. A point standing out of the face lands
    where it is seen, over its sticker in the flattened photo."""
    pose = pose or face_pose(quad, width, height)
    px = pose.project(points)
    back = np.linalg.inv(plane_homography(quad, size, half))
    q = np.c_[px, np.ones(len(px))] @ back.T
    return q[:, :2] / q[:, 2:3]
