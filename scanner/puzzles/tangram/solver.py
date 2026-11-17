"""The Tangram Cube Extreme's solver: the ladder solver trained in Cube_Solving_Pipeline.

The solver is a hierarchical subgroup ladder over the puzzle compiled from the same 3D model (one exact
table for the frame, then value networks for the large and the small pieces; about 91 steps and 4.5 s
per solve on the GPU). Its code is copied into third_party/cube_solver/ and its built ladder into
data/ladders/tangram_extreme/learned_r7 (``python tools/import_ladder.py --puzzle tangram``).

**The two sides' states are the same thing in a different order.** Both are one value per sticker
position (colour 0-5, or 6 where no sticker is), and both list the same 120 positions, but in their own
order. When the solver loads, each of this repository's positions is matched to the solver's by its
frame (the same normal, the nearest centre: the solver measures a sticker's centre as its area centroid,
this repository as the mean of its corners, so they differ by up to 0.03 while positions with one normal
are at least 0.25 apart). The match is then checked: the solved colours and every move must agree, or
the solver is refused.

The state given to ``solve`` is the decoder's: one orientation per piece (indices into the engine's
12-element rotation group), core fixed. A solution is replayed twice before it is returned: on the
solver's own tables, and on this repository's copy of the puzzle.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from scanner.core.services import NotSolvable, SolverNotReady, SolverStatus
from scanner.paths import DATA, SOLVER_DIR, use_solver

from .layout import EMPTY, TangramLayout, format_move, load_layout, parse_word

__all__ = ["NotSolvable", "SolverNotReady", "SolverStatus", "TangramSolution", "TangramSolverService", "check_agreement",
           "match_positions", "solver_state"]


@dataclass
class TangramSolution:
    word: str                          # "U2 F D' R ..."
    moves: list[dict]                  # [{"axis": "U", "amount": 2, "token": "U2"}, ...]
    turns: int                         # moves as written (any U^k counts one)
    steps: int                         # 60° steps of the corner layers plus flips
    seconds: float
    verified: bool = True
    source: str = "solver"
    detail: dict | None = None

    def to_json(self) -> dict:
        return {"word": self.word, "moves": self.moves, "turns": self.turns, "steps": self.steps,
                "seconds": round(self.seconds, 2), "verified": self.verified, "source": self.source,
                "detail": self.detail or {}}


def solver_state(layout: TangramLayout, orient: Sequence[int]) -> np.ndarray:
    """The state as the solver's vector: per sticker position, the colour there or 6 (empty)."""
    return layout.observe(orient).astype(np.uint8)


def solution_from_word(layout: TangramLayout, orient: Sequence[int], word: str, seconds: float, source: str = "solver",
                       detail: dict | None = None) -> TangramSolution:
    """Check a word on the scanner's own copy of the puzzle and describe it; raises if it does not solve."""
    moves = parse_word(word)
    end = layout.apply(orient, moves)
    if not np.array_equal(end, layout.solved_orient):
        raise AssertionError("the solution does not solve the state on the scanner's model of the puzzle")
    return TangramSolution(
        word=" ".join(format_move(a, n) for a, n in moves),
        moves=[{"axis": a, "amount": int(n), "token": format_move(a, n)} for a, n in moves],
        turns=len(moves),
        steps=sum(abs(n) for _, n in moves),
        seconds=seconds,
        source=source,
        detail=detail,
    )


class TangramSolverService:
    def __init__(self, cache_dir: Path | None = None, ladder: str = "learned", radius: int = 7, beam: int = 4096) -> None:
        self.layout = load_layout()
        self.cache_dir = Path(cache_dir or os.environ.get("CUBE_SCANNER_TANGRAM_LADDER", DATA / "ladders" / "tangram_extreme" / f"{ladder}_r{radius}"))
        self.ladder_name = ladder
        self.radius = radius
        self.beam = beam
        self.status = SolverStatus()
        self._lock = threading.Lock()
        self._backend = None

    # -- the solver ------------------------------------------------------------------------------------
    def _load_backend(self) -> "_Backend":
        """Build the solver from its cache. Raises SolverNotReady with what is missing."""
        code = SOLVER_DIR / "puzzles" / "tangram"
        if not code.is_dir():
            raise SolverNotReady(f"The Tangram Cube Extreme solver's code is not in {SOLVER_DIR}. Copy core/ and puzzles/ from Cube_Solving_Pipeline there.")
        if not any(self.cache_dir.glob("rung*.json")):
            raise SolverNotReady(f"No built Tangram ladder in {self.cache_dir}. Copy it there: python tools/import_ladder.py --puzzle tangram")
        use_solver()
        from core.ladder import Ladder
        from core.perm_model import build_point_model
        from core.structure import analyse
        from puzzles.registry import get_puzzle
        from puzzles.tangram.puzzle import compiled

        entry = get_puzzle("tangram_extreme")
        geo = compiled("extreme")
        puzzle = geo.puzzle
        to_theirs = match_positions(self.layout, geo.frames)
        check_agreement(self.layout, puzzle, to_theirs, geo.empty)
        ladders = entry.ladders()
        if self.ladder_name not in ladders:
            raise SolverNotReady(f"The Tangram solver has the ladders {sorted(ladders)}, not {self.ladder_name!r}.")
        model = build_point_model(puzzle)
        structure = analyse(model)
        ladder = Ladder(model, structure, ladders[self.ladder_name], mitm_radius=self.radius, log=self.status.log.append)
        ladder.build(cache_dir=self.cache_dir)
        if not ladder.ready():
            missing = [r.spec.name for r in ladder.rungs if r.table is None and r.heuristic is None]
            raise SolverNotReady(f"The Tangram ladder in {self.cache_dir} is incomplete; untrained rungs: {missing}")
        return _Backend(puzzle, ladder, entry.notation(ladder.alg), to_theirs, [r.spec.name for r in ladder.rungs])

    def _backend_solve(self, b: "_Backend", state: np.ndarray, beam: int) -> tuple[str, dict]:
        """One solver call: our state vector -> a word in the simulator's notation, and per-rung costs."""
        theirs = np.empty(len(state), dtype=np.uint8)
        theirs[b.to_theirs] = state
        try:
            word, per_rung = b.ladder.solve(theirs, beam=beam)
        except (LookupError, ValueError) as err:
            raise NotSolvable(str(err)) from err
        if not np.array_equal(b.puzzle.apply_word(theirs, list(word)), b.puzzle.goal):
            raise AssertionError("the solution does not solve the state on the solver's own tables")
        return b.notation.format(word), {"perRung": {n: int(c) for n, c in zip(b.rungs, per_rung)}, "beam": int(beam),
                                         "ladder": f"{self.ladder_name}_r{self.radius}"}

    # -- everything else is ready ---------------------------------------------------------------------
    def load(self) -> SolverStatus:
        with self._lock:
            if self.status.state in ("ready", "missing"):
                return self.status
            self.status = SolverStatus("loading", "Loading the Tangram solver")
            t0 = time.time()
            try:
                self._backend = self._load_backend()
                self.status = SolverStatus("ready", f"Loaded in {time.time() - t0:.1f} s from {self.cache_dir}")
            except SolverNotReady as err:
                self.status = SolverStatus("missing", str(err))
            except Exception as err:  # the reason is shown in the UI
                self.status = SolverStatus("error", f"{type(err).__name__}: {err}")
            return self.status

    def solve(self, orient: Sequence[int], beam: int | None = None) -> TangramSolution:
        orient = np.asarray(orient, dtype=np.int64)
        if orient.shape != (self.layout.n_pieces,):
            raise ValueError(f"a state is {self.layout.n_pieces} orientations, got {orient.shape}")
        if self.status.state != "ready":
            status = self.load()
            if status.state != "ready":
                raise SolverNotReady(status.detail)
        t0 = time.time()
        with self._lock:                   # one search at a time on the GPU
            word, detail = self._backend_solve(self._backend, solver_state(self.layout, orient), beam or self.beam)
        return solution_from_word(self.layout, orient, word, time.time() - t0, detail=detail)


@dataclass
class _Backend:
    puzzle: object                     # the solver's GatherPuzzle
    ladder: object                     # its built Ladder
    notation: object                   # the simulator's notation over its moves
    to_theirs: np.ndarray              # (120,) the solver's position for each of ours
    rungs: list[str]


def match_positions(layout: TangramLayout, frames: np.ndarray, tol_normal: float = 1e-4, tol_centre: float = 0.1) -> np.ndarray:
    """For each of our positions, the solver's position with the same frame (see the module's notes)."""
    frames = np.asarray(frames, dtype=np.float64)
    if len(frames) != layout.n_positions:
        raise SolverNotReady(f"the solver's puzzle has {len(frames)} positions, this repository's {layout.n_positions}")
    out = np.full(layout.n_positions, -1, dtype=np.int64)
    for p in range(layout.n_positions):
        same = np.flatnonzero(np.abs(frames[:, 3:] - layout.normal[p]).max(1) < tol_normal)
        if not len(same):
            raise SolverNotReady(f"no solver position has the normal of position {p}")
        d = np.linalg.norm(frames[same, :3] - layout.centre[p], axis=1)
        k = int(np.argmin(d))
        if d[k] > tol_centre:
            raise SolverNotReady(f"no solver position is near position {p} (nearest {d[k]:.3f} away)")
        out[p] = same[k]
    if len(set(out.tolist())) != layout.n_positions:
        raise SolverNotReady("the solver's positions do not match this repository's one to one")
    return out


def check_agreement(layout: TangramLayout, puzzle, to_theirs: np.ndarray, empty: int, seed: int = 0) -> None:
    """The solved colours and every move must be the same on both sides under the position match."""
    if empty != EMPTY:
        raise SolverNotReady(f"the solver marks an empty position {empty}, this repository {EMPTY}")
    if not np.array_equal(np.asarray(puzzle.goal)[to_theirs], layout.solved):
        raise SolverNotReady("the solver's solved state shows other colours than this repository's")
    rng = np.random.default_rng(seed)
    orient = layout.random_state(rng, 40)
    theirs = np.empty(layout.n_positions, dtype=np.uint8)
    theirs[to_theirs] = layout.observe(orient)
    for k, name in enumerate(puzzle.move_names):
        after = np.asarray(puzzle.apply_word(theirs, [k]))[to_theirs]
        if not np.array_equal(after, layout.observe(layout.apply(orient, parse_word(name)))):
            raise SolverNotReady(f"the solver's move {name} does not do what the 3D model's does")
