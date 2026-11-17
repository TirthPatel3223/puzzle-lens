"""Where things are on a face, and how a photo is cut into slot crops.

A face is described in *face coordinates*: x to the right, y up, the face square is [-1, 1]²
(1 = 30 mm), seen from outside. A photo of a face is tied to them by the face's four corners
in the picture (a "quad", in pixels: top-left, top-right, bottom-right, bottom-left), which
give a homography.

A face has 13 slots, read from 13 square crops, every one turned so that slots of one kind
look the same wherever they sit on the face:

    crops 0-3    the four corner stars      top-left, top-right, bottom-right, bottom-left
    crops 4-7    the four gears             top, right, bottom, left
    crops 8-11   this face's half of the four edge slots   top, right, bottom, left
    crop  12     the centre slot

Crops of kind k at position p are the crop at position 0 turned p quarter turns clockwise
about the face centre. Which cells of the 4x4 grid each crop reports is worked out here from
the layout's cell points, not written down by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from scanner.puzzles.maltese.layout import Layout, load_layout

#: Face corners in face coordinates, in quad order.
FACE_CORNERS = np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])

#: Side of every crop, pixels.
CROP = 96
#: Side of the context picture (the whole face, small), pixels, and how much of the face plane it shows.
CONTEXT = 96
CONTEXT_HALF = 1.25

KINDS = ("star", "gear", "edge", "centre")
#: Centre (face coordinates) and half-width of each kind's crop at position 0.
CANON = {
    "star": ((-0.5, 0.5), 0.47),      # the whole star, top-left corner
    "gear": ((0.0, 0.44), 0.38),      # hub at tan 22.5° = 0.414; teeth reach 0.16 to 0.72
    "edge": ((0.0, 0.86), 0.28),      # the roof sticker on this face's side of the top edge
    "centre": ((0.0, 0.0), 0.30),     # the 5.2 mm column, with room for an edge standing 11 mm proud
}
N_CROPS = 13
#: (kind, position) of each crop.
CROPS: tuple[tuple[str, int], ...] = (
    *(("star", p) for p in range(4)),
    *(("gear", p) for p in range(4)),
    *(("edge", p) for p in range(4)),
    ("centre", 0),
)
STAR, GEAR, EDGE, CENTRE = slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 13)


def _rot_cw(p: int) -> np.ndarray:
    """Quarter turns clockwise (seen from outside, y up): (x, y) -> (y, -x), p times. 3x3."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][p % 4]
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def crop_to_face(kind: str, position: int, size: int = CROP) -> np.ndarray:
    """3x3 map from crop pixels (u right, v down, pixel centres at +0.5) to face coordinates."""
    (cx, cy), h = CANON[kind]
    a = np.array([
        [2 * h / size, 0.0, cx - h + h / size],
        [0.0, -2 * h / size, cy + h - h / size],
        [0.0, 0.0, 1.0],
    ])
    return _rot_cw(position) @ a


def context_to_face(size: int = CONTEXT) -> np.ndarray:
    h = CONTEXT_HALF
    return np.array([[2 * h / size, 0.0, -h + h / size], [0.0, -2 * h / size, h - h / size], [0.0, 0.0, 1.0]])


def face_to_photo(quad: np.ndarray) -> np.ndarray:
    """Homography from face coordinates to photo pixels, from the face's corners in the photo."""
    return cv2.getPerspectiveTransform(FACE_CORNERS.astype(np.float32), np.asarray(quad, dtype=np.float32))


def picture_order(quad: np.ndarray) -> tuple[np.ndarray, int]:
    """The quad starting from the corner that is top-left *in the picture*, and how far it was shifted.

    A face photographed upright (within 45°) has its own top-left corner there and the shift is 0;
    a shift of k means the face appears turned k quarter turns counterclockwise in the picture.
    """
    quad = np.asarray(quad, dtype=np.float64)
    k = int(np.argmin(quad[:, 0] + quad[:, 1]))
    return np.roll(quad, -k, axis=0), k


def quad_side(quad: np.ndarray) -> float:
    """Mean side length of the quad, pixels."""
    q = np.asarray(quad, dtype=np.float64)
    return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))


def prepare_photo(image: np.ndarray, quad: np.ndarray, face_px: float = 640.0) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a large photo so the face is about ``face_px`` across (crops then sample it without aliasing)."""
    side = quad_side(quad)
    if side <= face_px * 1.25:
        return image, np.asarray(quad, dtype=np.float64)
    f = face_px / side
    small = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return small, np.asarray(quad, dtype=np.float64) * f


def cut_crops(image: np.ndarray, quad: np.ndarray, size: int = CROP, context: int = CONTEXT) -> tuple[np.ndarray, np.ndarray]:
    """The 13 slot crops (13, size, size, 3) and the context picture of one face.

    ``image`` is HxWx3 (any channel order, kept); ``quad`` its face corners in pixels, in the
    order the face should be read in (see ``picture_order``).
    """
    h = face_to_photo(quad)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    crops = np.empty((N_CROPS, size, size, 3), dtype=image.dtype)
    for i, (kind, position) in enumerate(CROPS):
        crops[i] = cv2.warpPerspective(image, h @ crop_to_face(kind, position, size), (size, size), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    ctx = cv2.warpPerspective(image, h @ context_to_face(context), (context, context), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    return crops, ctx


def rectify(image: np.ndarray, quad: np.ndarray, size: int = 512, half: float = 1.15) -> np.ndarray:
    """The whole face flattened (for showing to a person): ``half`` face units from the centre to each border."""
    m = np.array([[2 * half / size, 0.0, -half + half / size], [0.0, -2 * half / size, half - half / size], [0.0, 0.0, 1.0]])
    return cv2.warpPerspective(image, face_to_photo(quad) @ m, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


@dataclass(frozen=True)
class FaceCells:
    """Which of a face's 16 cells each crop reports, in the order the reader gives them."""

    star: np.ndarray      # (4,)    cell of each star crop
    edge: np.ndarray      # (4, 2)  cells of each edge crop: left, right as seen in the crop
    centre: np.ndarray    # (4,)    cells of the centre crop: top-left, top-right, bottom-left, bottom-right


@lru_cache(maxsize=4)
def _face_cells(layout_id: int) -> FaceCells:
    layout = _LAYOUTS[layout_id]
    pts = np.array(layout.geometry["cellPoints"], dtype=np.float64)        # (16, 2), x right, y up
    grid = layout.grid

    def in_crop(kind: str, position: int, cell: int) -> np.ndarray:
        """The cell's point in the crop's own picture: u right, v down, in crop widths from the centre."""
        # With a crop one pixel wide, that pixel's centre is the origin and the crop spans -0.5 to 0.5.
        m = np.linalg.inv(crop_to_face(kind, position, 1))
        u, v, w = m @ np.array([pts[cell, 0], pts[cell, 1], 1.0])
        return np.array([u / w, v / w])

    corner_cells = list(grid["cornerCells"])
    star = np.empty(4, dtype=np.int64)
    for p in range(4):
        # The star crop at position p shows the corner whose point lies nearest its middle.
        d = [np.linalg.norm(in_crop("star", p, c)) for c in corner_cells]
        star[p] = corner_cells[int(np.argmin(d))]
    if len(set(star.tolist())) != 4:
        raise RuntimeError("the star crops do not cover the four corners")

    edge = np.empty((4, 2), dtype=np.int64)
    sides = [list(pair) for pair in grid["edgeCells"]]
    for p in range(4):
        # The side whose two cells lie inside this crop; left to right as the crop shows them.
        inside = [pair for pair in sides if all(np.abs(in_crop("edge", p, c)).max() < 0.5 for c in pair)]
        if len(inside) != 1:
            raise RuntimeError(f"edge crop {p} does not contain exactly one side's cells")
        pair = sorted(inside[0], key=lambda c: in_crop("edge", p, c)[0])
        edge[p] = pair

    centre_cells = list(grid["centreCells"])
    centre = np.empty(4, dtype=np.int64)
    for c in centre_cells:
        u, v = in_crop("centre", 0, c)
        centre[(1 if u > 0 else 0) + (2 if v > 0 else 0)] = c
    if len(set(centre.tolist())) != 4:
        raise RuntimeError("the centre crop does not separate its four cells")
    return FaceCells(star, edge, centre)


_LAYOUTS: dict[int, Layout] = {}


def face_cells(layout: Layout | None = None) -> FaceCells:
    layout = layout or load_layout()
    _LAYOUTS[id(layout)] = layout
    return _face_cells(id(layout))


@lru_cache(maxsize=4)
def _face_turns(layout_id: int) -> tuple[np.ndarray, np.ndarray]:
    """How turning a face in the picture relabels it, taken from the whole-cube rotations.

    Returns ``cells`` (4, 16) and ``gears`` (4, 4). A face that appears turned k quarter turns
    counterclockwise in the picture and is read as if upright gives ``read``; its labels in its
    own orientation are ``true = read[cells[k]]`` (and the same with ``gears``).
    """
    layout = _LAYOUTS[layout_id]
    pts = np.array(layout.geometry["cellPoints"], dtype=np.float64)
    hubs = np.array(layout.geometry["gearPositions"], dtype=np.float64)
    cells = np.empty((4, 16), dtype=np.int64)
    gears = np.empty((4, 4), dtype=np.int64)
    for k in range(4):
        # The face point q appears in the picture at q turned k quarter turns counterclockwise,
        # so the label of the cell at q is what was read at the cell there.
        r = _rot_cw(-k)[:2, :2]
        for c in range(16):
            target = r @ pts[c]
            cells[k, c] = int(np.argmin(np.linalg.norm(pts - target, axis=1)))
        for g in range(4):
            target = r @ hubs[g]
            gears[k, g] = int(np.argmin(np.linalg.norm(hubs - target, axis=1)))
        if len(set(cells[k].tolist())) != 16 or len(set(gears[k].tolist())) != 4:
            raise RuntimeError("a quarter turn does not permute the face's cells")
    return cells, gears


def turn_face(cells: np.ndarray, gear_colour: np.ndarray, gear_quarter: np.ndarray, k: int,
              layout: Layout | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels (or probabilities, first axis = cell / gear) of a face read from a picture in which
    it appears turned k quarter turns counterclockwise, put back into the face's own orientation.

    A gear's quarter is counted from its own slot, so it does not change; it just moves with its gear.
    """
    layout = layout or load_layout()
    _LAYOUTS[id(layout)] = layout
    c, g = _face_turns(id(layout))
    k %= 4
    return np.asarray(cells)[c[k]], np.asarray(gear_colour)[g[k]], np.asarray(gear_quarter)[g[k]]
