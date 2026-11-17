"""Compile a puzzle's 3D model into a GatherPuzzle.

A 3D model describes a puzzle by its geometry: every sticker is a frame in space (its centre and its
outward normal), and every move turns the pieces inside a region {x : n·x > d} by a rotation. Starting
from the stickers of the solved puzzle and applying every move to every frame inside its region until
no new frame appears gives the finite set of *positions* a sticker can ever occupy. Each move is then
a permutation of the positions, i.e. a gather table, and the state is the colour at each position
(``empty`` where no sticker is). Nothing about the geometry is needed after this: the solver runs on
the tables like it does on the hand-made Maltese ones.

Two details matter:

* A position is the centre *and* the normal. On the Tangram Cube Extreme twelve pairs of positions share
  a centre and differ only in their normal (a sticker turned 60° lands where another piece's sticker
  sits, tilted). With centres alone a move would not be a permutation.
* Applying a move to single positions (rather than to whole pieces) is only right if a move never cuts
  a piece: every piece in every pose it can reach must lie on one side of every move's boundary. That is
  what makes the puzzle a permutation group (doctrinaire, every move always legal). It is checked here
  from the pieces' vertices before anything else, and a model that fails it (a jumbling or bandaged
  puzzle, where a turn can be blocked) is refused with GeometryError.

The input is the JSON written by an exporter next to the 3D model (for the Tangram Cube:
``tools/export/geometry.gen.ts`` in the Cube_Modeling project), format ``puzzle-geometry/1``. It also
carries check walks played on the model's own engine; the compiled tables must reproduce them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from core.gather_engine import GatherPuzzle

FORMAT = "puzzle-geometry/1"
COLOUR_MODULUS = 256              # colours are only moved, never incremented


class GeometryError(ValueError):
    """The model cannot be compiled into permutations, or the tables disagree with the model."""


@dataclass(frozen=True)
class Move3D:
    name: str
    inverse: str
    rotation: np.ndarray          # (3, 3), acts on column vectors
    normal: np.ndarray            # (3,) unit normal of the region the move turns
    offset: float                 # the region is {x : normal·x > offset}
    order: int

    def inside(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal > self.offset


@dataclass(frozen=True)
class Geometry:
    name: str
    title: str
    colours: tuple[str, ...]                 # colour id -> name (the cube face it belongs to)
    sticker_piece: np.ndarray                # (S,) piece each sticker belongs to
    sticker_colour: np.ndarray               # (S,) colour id
    sticker_centre: np.ndarray               # (S, 3)
    sticker_normal: np.ndarray               # (S, 3)
    sticker_polygon: tuple[np.ndarray, ...]  # outline of each sticker (for descriptions only)
    piece_centroid: np.ndarray               # (P, 3)
    piece_vertices: tuple[np.ndarray, ...]
    moves: tuple[Move3D, ...]
    checks: tuple[tuple[tuple[str, ...], np.ndarray], ...]   # (move names, (P, 3, 3) piece orientations after them)
    source: dict

    @property
    def n_pieces(self) -> int:
        return len(self.piece_centroid)

    @property
    def n_stickers(self) -> int:
        return len(self.sticker_colour)

    def move_index(self, name: str) -> int:
        for i, m in enumerate(self.moves):
            if m.name == name:
                return i
        raise KeyError(f"no move named {name!r}")


def load_geometry(path: str | Path) -> Geometry:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise GeometryError(f"{path}: expected format {FORMAT!r}, found {data.get('format')!r}")
    stickers = data["stickers"]
    moves = []
    for m in data["moves"]:
        n = np.asarray(m["region"]["normal"], dtype=np.float64)
        moves.append(Move3D(name=m["name"], inverse=m["inverse"], rotation=np.asarray(m["rotation"], dtype=np.float64),
                            normal=n, offset=float(m["region"]["offset"]), order=int(m["order"])))
    return Geometry(
        name=data["name"],
        title=data.get("title", data["name"]),
        colours=tuple(c["face"] for c in data["colours"]),
        sticker_piece=np.array([s["piece"] for s in stickers], dtype=np.int64),
        sticker_colour=np.array([s["colour"] for s in stickers], dtype=np.int64),
        sticker_centre=np.array([s["centre"] for s in stickers], dtype=np.float64),
        sticker_normal=np.array([s["normal"] for s in stickers], dtype=np.float64),
        sticker_polygon=tuple(np.asarray(s.get("polygon", []), dtype=np.float64) for s in stickers),
        piece_centroid=np.array([p["centroid"] for p in data["pieces"]], dtype=np.float64),
        piece_vertices=tuple(np.asarray(p["vertices"], dtype=np.float64) for p in data["pieces"]),
        moves=tuple(moves),
        checks=tuple((tuple(c["moves"]), np.asarray(c["orientation"], dtype=np.float64)) for c in data.get("checks", [])),
        source=data.get("source", {}),
    )


# -- the compiled puzzle ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledGeometry:
    geometry: Geometry
    puzzle: GatherPuzzle
    frames: np.ndarray            # (N, 6) centre and normal of every position
    home_sticker: np.ndarray      # (N,) sticker at the position when solved, -1 if none
    empty: int                    # colour value of a position no sticker covers
    report: dict                  # what was checked, and the margins found

    def describe(self) -> str:
        r = self.report
        return (f"{self.geometry.title}: {r['pieces']} pieces, {r['stickers']} stickers, {r['positions']} positions "
                f"({r['stickers']} occupied when solved), {r['orientations']} piece orientations; every move legal in "
                f"every state; positions >= {r['boundary_margin']:.3f} from any move boundary and >= {r['separation']:.3f} "
                f"apart; {r['centre_collisions']} pairs of positions share a centre; tables agree with "
                f"{r['model_walks']} walks on the model and {r['pose_walks']} walks of the piece-pose engine")


class _FrameStore:
    """Positions found so far, matched with a tolerance (positions are far apart compared with it)."""

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.rows: list[np.ndarray] = []
        self._arr = np.empty((0, 6))

    def find(self, frame: np.ndarray) -> int:
        if len(self._arr) != len(self.rows):
            self._arr = np.array(self.rows)
        if not len(self._arr):
            return -1
        hit = np.flatnonzero(np.abs(self._arr - frame).max(axis=1) < self.tol)
        return int(hit[0]) if len(hit) else -1

    def add(self, frame: np.ndarray) -> int:
        self.rows.append(np.asarray(frame, dtype=np.float64))
        return len(self.rows) - 1


def _turn(frames: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate (k, 6) frames: centre and normal."""
    return np.concatenate([frames[:, :3] @ rotation.T, frames[:, 3:] @ rotation.T], axis=1)


def _check_moves(geom: Geometry, tol: float) -> None:
    names = [m.name for m in geom.moves]
    if len(set(names)) != len(names):
        raise GeometryError("move names are not unique")
    for m in geom.moves:
        r = m.rotation
        if not (np.allclose(r @ r.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(r) - 1) < 1e-9):
            raise GeometryError(f"move {m.name}: not a rotation matrix")
        if not np.allclose(r @ m.normal, m.normal, atol=1e-9):
            raise GeometryError(f"move {m.name}: the rotation does not keep its own region in place")
        if m.inverse not in names:
            raise GeometryError(f"move {m.name}: inverse {m.inverse!r} is not a move")
        inv = geom.moves[geom.move_index(m.inverse)]
        if not np.allclose(inv.rotation @ r, np.eye(3), atol=1e-9) or not np.allclose(inv.normal, m.normal, atol=tol) \
                or abs(inv.offset - m.offset) > tol:
            raise GeometryError(f"move {m.name}: {m.inverse} does not undo it")
        if not np.allclose(np.linalg.matrix_power(r, m.order), np.eye(3), atol=1e-9):
            raise GeometryError(f"move {m.name}: the rotation does not have order {m.order}")


def reachable_poses(geom: Geometry, tol: float = 1e-7) -> tuple[list[np.ndarray], list[list[int]]]:
    """Every orientation each piece can reach (a piece moves when its centroid is in a move's region,
    as in the model's engine). Returns the distinct orientation matrices and, per piece, their indices."""
    mats: list[np.ndarray] = [np.eye(3)]

    def intern(m: np.ndarray) -> int:
        for i, e in enumerate(mats):
            if np.abs(e - m).max() < tol:
                return i
        mats.append(m)
        return len(mats) - 1

    per_piece: list[list[int]] = []
    for p in range(geom.n_pieces):
        seen = {0}
        stack = [0]
        while stack:
            g = stack.pop()
            c = mats[g] @ geom.piece_centroid[p]
            for mv in geom.moves:
                if c @ mv.normal > mv.offset:
                    h = intern(mv.rotation @ mats[g])
                    if h not in seen:
                        seen.add(h)
                        stack.append(h)
        per_piece.append(sorted(seen))
    return mats, per_piece


def _check_legal(geom: Geometry, mats: list[np.ndarray], poses: list[list[int]], tol: float) -> None:
    """No piece, in any pose it can reach, crosses the boundary of any move's region."""
    bad = []
    for p, verts in enumerate(geom.piece_vertices):
        for g in poses[p]:
            world = verts @ mats[g].T
            for mv in geom.moves:
                s = world @ mv.normal - mv.offset
                if s.min() < -tol and s.max() > tol:
                    bad.append((p, g, mv.name))
    if bad:
        raise GeometryError(
            f"{len(bad)} (piece, pose, move) combinations where a move would cut a piece, e.g. {bad[:5]}: "
            "this puzzle jumbles or is bandaged, so its moves are not permutations of positions")


def _replay(images: np.ndarray, moves: Sequence[int], n: int) -> np.ndarray:
    """Position of every label after a sequence of moves (labels start at their own positions)."""
    where = np.arange(n)
    for m in moves:
        where = images[m][where]
    return where


def compile_geometry(geom: Geometry, tol: float = 1e-7, pose_walks: int = 50, seed: int = 0) -> CompiledGeometry:
    """Compile a 3D model into a GatherPuzzle, with every check described in the module docstring."""
    _check_moves(geom, tol)
    mats, poses = reachable_poses(geom, tol)
    _check_legal(geom, mats, poses, tol)

    # positions: closure of the sticker frames under the moves (breadth first, deterministic order)
    store = _FrameStore(tol)
    home = np.concatenate([geom.sticker_centre, geom.sticker_normal], axis=1)
    for f in home:
        if store.find(f) >= 0:
            raise GeometryError("two stickers of the solved puzzle have the same centre and normal")
        store.add(f)
    pending = list(range(len(home)))
    while pending:
        rows = np.array(pending)
        pending = []
        frames = np.array(store.rows)
        for mv in geom.moves:
            sel = rows[mv.inside(frames[rows, :3])]
            for f in _turn(frames[sel], mv.rotation):
                if store.find(f) < 0:
                    pending.append(store.add(f))
    frames = np.array(store.rows)
    n = len(frames)

    # every position must be clearly inside or outside every region
    margins = np.abs(np.stack([frames[:, :3] @ mv.normal - mv.offset for mv in geom.moves]))
    if margins.min() < 1e3 * tol:
        m, k = np.unravel_index(int(margins.argmin()), margins.shape)
        raise GeometryError(f"position {k} lies on the boundary of move {geom.moves[m].name}")
    dist = np.abs(frames[:, None, :] - frames[None, :, :]).max(axis=-1)
    np.fill_diagonal(dist, np.inf)
    if dist.min() < 1e3 * tol:
        raise GeometryError("two positions are too close to tell apart reliably")
    centre_dist = np.abs(frames[:, None, :3] - frames[None, :, :3]).max(axis=-1)
    np.fill_diagonal(centre_dist, np.inf)

    # each move as a permutation of the positions: images[m, i] = where the content of i goes
    images = np.tile(np.arange(n), (len(geom.moves), 1))
    for mi, mv in enumerate(geom.moves):
        sel = np.flatnonzero(mv.inside(frames[:, :3]))
        for i, f in zip(sel, _turn(frames[sel], mv.rotation)):
            images[mi, i] = store.find(f)
        if (images[mi] < 0).any() or len(set(images[mi].tolist())) != n:
            raise GeometryError(f"move {mv.name} is not a permutation of the positions")
        power = np.arange(n)
        for _ in range(mv.order):
            power = images[mi][power]
        if not np.array_equal(power, np.arange(n)):
            raise GeometryError(f"move {mv.name} does not have order {mv.order} on the positions")
    for mi, mv in enumerate(geom.moves):
        if not np.array_equal(images[geom.move_index(mv.inverse)][images[mi]], np.arange(n)):
            raise GeometryError(f"move {mv.inverse} does not undo {mv.name} on the positions")

    def expected_positions(orient: np.ndarray) -> np.ndarray:
        """Where each sticker is when piece p has orientation orient[p]."""
        out = np.empty(geom.n_stickers, dtype=np.int64)
        for s in range(geom.n_stickers):
            g = orient[geom.sticker_piece[s]]
            out[s] = store.find(np.concatenate([g @ geom.sticker_centre[s], g @ geom.sticker_normal[s]]))
        return out

    # the tables against the model's own walks
    for k, (names, orient) in enumerate(geom.checks):
        where = _replay(images, [geom.move_index(x) for x in names], n)
        if not np.array_equal(where[:geom.n_stickers], expected_positions(orient)):
            raise GeometryError(f"check walk {k} of the model ends in a different state than the compiled tables")

    # the tables against a piece-pose engine (whole pieces move when their centroid is in a region)
    rng = np.random.default_rng(seed)
    for _ in range(pose_walks):
        word = rng.integers(0, len(geom.moves), size=80).tolist()
        orient = np.tile(np.eye(3), (geom.n_pieces, 1, 1))
        for m in word:
            mv = geom.moves[m]
            inside = mv.inside(np.einsum("pij,pj->pi", orient, geom.piece_centroid))
            orient[inside] = mv.rotation @ orient[inside]
        if not np.array_equal(_replay(images, word, n)[:geom.n_stickers], expected_positions(orient)):
            raise GeometryError("the position tables disagree with moving whole pieces")

    empty = len(geom.colours)
    goal = np.full(n, empty, dtype=np.uint8)
    goal[:geom.n_stickers] = geom.sticker_colour
    idx = np.empty(images.shape, dtype=np.intp)
    for mi in range(len(geom.moves)):
        idx[mi, images[mi]] = np.arange(n)                  # position images[i] receives what was at i
    puzzle = GatherPuzzle(
        name=geom.name,
        move_names=tuple(m.name for m in geom.moves),
        idx=idx,
        add=np.zeros(images.shape, dtype=np.int16),
        modulus=np.full(n, COLOUR_MODULUS, dtype=np.int16),
        goal=goal,
        inverse=np.array([geom.move_index(m.inverse) for m in geom.moves], dtype=np.intp),
    )
    home_sticker = np.full(n, -1, dtype=np.int64)
    home_sticker[:geom.n_stickers] = np.arange(geom.n_stickers)
    report = dict(
        pieces=geom.n_pieces, stickers=geom.n_stickers, positions=n, orientations=len(mats),
        poses_per_piece=[len(p) for p in poses],
        boundary_margin=float(margins.min()), separation=float(dist.min()),
        centre_collisions=int((centre_dist < 1e3 * tol).sum() // 2),
        model_walks=len(geom.checks), pose_walks=pose_walks,
    )
    return CompiledGeometry(geometry=geom, puzzle=puzzle, frames=frames, home_sticker=home_sticker,
                            empty=empty, report=report)
