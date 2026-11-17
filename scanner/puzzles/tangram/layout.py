"""The Tangram Cube Extreme as the Python side knows it.

Nothing here is geometry or move logic of its own: it is all read from ``shared/tangram/layout.json``,
which the browser side generates from the 3D engine (``web/src/tangram/model.ts``; see there for what
positions, cells and placements are).

An **observation** is one value per position (120): the colour of the sticker there (0-5) or EMPTY (6).
A **state** is one orientation per piece (45, indices into the 12-element rotation group); the core
piece is always at the identity, which fixes the frame.

Colours are indices into the engine's faces (+x -x +y -y +z -z): orange, red, yellow, white, green, blue.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from scanner.paths import SHARED

N_COLOURS = 6
EMPTY = 6
ENDS = ("T", "B")
FACE_IDS = ("+x", "-x", "+y", "-y", "+z", "-z")
AXES = ("U", "D", "F", "R", "L")


@dataclass(frozen=True)
class Placement:
    """Where piece ``piece`` is and what it shows at orientation ``ori``."""

    piece: int
    ori: int
    cell: int
    positions: tuple[int, ...]          # position of each sticker, in the piece's sticker order


class TangramLayout:
    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.colour_names: list[str] = [c["name"] for c in raw["colours"]]
        self.colour_hex: list[str] = [c["hex"] for c in raw["colours"]]
        g = raw["group"]
        self.group_identity = int(g["identity"])
        self.group_matrices = np.array(g["matrices"], dtype=np.float64).reshape(-1, 3, 3)
        self.group_mul = np.array(g["mul"], dtype=np.int64)
        self.group_inv = np.array(g["inv"], dtype=np.int64)
        self.n_pieces = len(raw["pieces"])
        self.piece_colours: list[tuple[int, ...]] = [tuple(p["colours"]) for p in raw["pieces"]]
        self.piece_layer: list[str] = [p["layer"] for p in raw["pieces"]]
        self.core = next(i for i, c in enumerate(self.piece_colours) if not c)
        self.n_cells = len(raw["cells"])
        pos = raw["positions"]
        self.n_positions = len(pos)
        self.centre = np.array([p["centre"] for p in pos], dtype=np.float64)
        self.normal = np.array([p["normal"] for p in pos], dtype=np.float64)
        self.polygons = [np.array(p["polygon"], dtype=np.float64) for p in pos]
        self.on_cube = np.array([p["onCube"] for p in pos], dtype=bool)
        self.facing = np.array([p["facing"] for p in pos])
        self.position_cell = np.array([p["cell"] for p in pos], dtype=np.int64)
        self.position_home: list[tuple[int, int] | None] = [(p["home"]["piece"], p["home"]["sticker"]) if p["home"] else None for p in pos]
        self.placements: list[dict[int, Placement]] = [dict() for _ in range(self.n_pieces)]
        for pl in raw["placements"]:
            self.placements[pl["piece"]][pl["ori"]] = Placement(pl["piece"], pl["ori"], pl["cell"], tuple(pl["positions"]))
        self.home_positions = [tuple(p["homePositions"]) for p in raw["pieces"]]
        self.axes = raw["axes"]
        self.axis_cells = {a: frozenset(v["cells"]) for a, v in raw["axes"].items()}
        self.axis_step = {a: int(v["step"]) for a, v in raw["axes"].items()}
        self.axis_order = {a: int(v["order"]) for a, v in raw["axes"].items()}
        self.move_from = {a: np.array(v, dtype=np.int64) for a, v in raw["moves"].items()}
        self.solved = np.array(raw["solved"], dtype=np.int64)
        #: The six face-on photos: per face (FACE_IDS order) its frame rows (right, up, normal) and corners.
        self.face_ids: list[str] = [f["face"] for f in raw["faces"]]
        self.face_end: list[str] = [f["end"] for f in raw["faces"]]
        self.face_rows = np.array([[f["right"], f["up"], f["normal"]] for f in raw["faces"]], dtype=np.float64)   # (6, 3, 3)
        self.face_corners = np.array([f["corners"] for f in raw["faces"]], dtype=np.float64)                     # (6, 4, 3)
        self.face_min_cos = float(raw["faceMinCos"])

    # -- states and observations ------------------------------------------------------------------
    @property
    def solved_orient(self) -> np.ndarray:
        return np.full(self.n_pieces, self.group_identity, dtype=np.int64)

    def cell_of(self, orient: Sequence[int]) -> np.ndarray:
        return np.array([self.placements[i][int(o)].cell for i, o in enumerate(orient)], dtype=np.int64)

    def observe(self, orient: Sequence[int]) -> np.ndarray:
        """What a state shows: per position, a colour or EMPTY."""
        out = np.full(self.n_positions, EMPTY, dtype=np.int64)
        for i, o in enumerate(orient):
            pl = self.placements[i].get(int(o))
            if pl is None:
                raise ValueError(f"piece {i} cannot be at orientation {o}")
            for k, p in enumerate(pl.positions):
                out[p] = self.piece_colours[i][k]
        return out

    def sticker_positions(self, orient: Sequence[int]) -> np.ndarray:
        """Where each sticker is (66 entries, in the order of the home positions)."""
        out = []
        for i, o in enumerate(orient):
            out.extend(self.placements[i][int(o)].positions)
        return np.array(out, dtype=np.int64)

    @cached_property
    def sticker_homes(self) -> np.ndarray:
        """The home position of each sticker, same order as ``sticker_positions``."""
        return np.array([p for hp in self.home_positions for p in hp], dtype=np.int64)

    @cached_property
    def symmetries(self) -> list[int]:
        """The group elements that are symmetries of the cube (D3: 120° turns about the diagonal and the
        three flips). Turning the whole puzzle by one of them looks the same from outside, cut for cut."""
        m = self.group_matrices
        return [g for g in range(len(m)) if np.allclose(np.abs(m[g]).sum(0), 1) and np.allclose(np.abs(m[g]), np.round(np.abs(m[g])))]

    # -- colour layouts (scanner.core.colours) ------------------------------------------------------
    @cached_property
    def colour_normals(self) -> np.ndarray:
        """(6, 3): the direction of the face each colour covers when solved."""
        out = np.zeros((N_COLOURS, 3))
        for c in range(N_COLOURS):
            n = self.normal[(self.solved == c) & self.on_cube].mean(0)
            out[c] = np.round(n / np.linalg.norm(n))
        return out

    @cached_property
    def colour_actions(self) -> list[tuple[int, ...]]:
        """How each whole-puzzle symmetry relabels the colours."""
        from scanner.core.colours import colour_action

        return [colour_action(self.group_matrices[g], self.colour_normals) for g in self.symmetries]

    @cached_property
    def colour_layouts(self) -> list:
        """One ``ColourLayout`` per class a scan can tell apart (120), the model's own first."""
        from scanner.core.colours import layout_classes

        return layout_classes(self.colour_actions, self.colour_normals)

    def turned(self, orient: Sequence[int], g: int) -> np.ndarray:
        """The same arrangement with the whole puzzle (core included) turned by group element g."""
        return self.group_mul[g, np.asarray(orient, dtype=np.int64)]

    # -- moves --------------------------------------------------------------------------------------
    def apply(self, orient: Sequence[int], word: str | Sequence[tuple[str, int]]) -> np.ndarray:
        """Apply moves in the simulator's notation (``"U2' F D R"``) to a state."""
        moves = parse_word(word) if isinstance(word, str) else list(word)
        o = np.array(orient, dtype=np.int64)
        for axis, amount in moves:
            cells = self.axis_cells[axis]
            g = self.axis_step[axis] if amount > 0 else int(self.group_inv[self.axis_step[axis]])
            for _ in range(abs(amount)):
                inside = [i for i in range(self.n_pieces) if self.placements[i][int(o[i])].cell in cells]
                for i in inside:
                    o[i] = self.group_mul[g, o[i]]
        return o

    def random_state(self, rng: np.random.Generator, moves: int = 60) -> np.ndarray:
        word = [(AXES[int(rng.integers(5))], 1) for _ in range(moves)]
        return self.apply(self.solved_orient, word)

    @cached_property
    def generators(self) -> list[np.ndarray]:
        """One step of each axis as a map of positions: the sticker at p goes to gen[p]."""
        out = []
        for a in AXES:
            frm = self.move_from[a]
            to = np.empty_like(frm)
            to[frm] = np.arange(len(frm))
            out.append(to)
        return out


_TOKEN = re.compile(r"([UDFRLudfrl])(\d*)(['’`]?)(\d*)")


def parse_word(word: str) -> list[tuple[str, int]]:
    """``"U2' D F"`` -> ``[("U", -2), ("D", 1), ("F", 1)]``. A flip is its own inverse (F' = F)."""
    out = []
    for token in word.replace(",", " ").split():
        m = _TOKEN.fullmatch(token)
        if m is None or (m.group(2) and m.group(4)):
            raise ValueError(f"cannot read move {token!r}")
        axis = m.group(1).upper()
        n = int(m.group(2) or m.group(4) or 1)
        if axis in "FRL":
            if n % 2 == 0:
                raise ValueError(f"{token} is a full turn of a flip")
            out.append((axis, 1))
            continue
        a = (-n if m.group(3) else n) % 6
        if a == 0:
            raise ValueError(f"{token} is a full turn")
        out.append((axis, a - 6 if a > 3 else a))
    return out


def format_move(axis: str, amount: int) -> str:
    if axis in "FRL":
        return axis
    n = abs(amount)
    return axis + ("" if n == 1 else str(n)) + ("'" if amount < 0 else "")


@lru_cache(maxsize=1)
def load_layout(shared: Path | None = None) -> TangramLayout:
    return TangramLayout(json.loads(((shared or SHARED) / "tangram" / "layout.json").read_text(encoding="utf-8")))
