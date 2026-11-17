"""The Maltese Gear Cube's ladder solver (copied into third_party/cube_solver), loaded once and kept warm.

The solver works on its own 144-entry array and answers in its own move names (U, L, F and
their inverses: it holds the cube blue up, yellow in front, purple on the left, and never
moves the white-green-red corner). Every solution is replayed on the solver's reference
simulator before it is returned.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scanner.core.services import NotSolvable, SolverStatus
from scanner.paths import DATA, use_solver

__all__ = ["NotSolvable", "Solution", "SolverService", "SolverStatus"]


@dataclass
class Solution:
    word: str                              # "U L'3 F2 ..."
    moves: list[dict]                      # [{"name": "U", "amount": 1}, {"name": "L", "amount": -3}, ...]
    quarter_turns: int
    face_turns: int
    per_rung: dict[str, int]
    seconds: float
    verified: bool = True

    def to_json(self) -> dict:
        return {"word": self.word, "moves": self.moves, "quarterTurns": self.quarter_turns, "faceTurns": self.face_turns,
                "perRung": self.per_rung, "seconds": round(self.seconds, 2), "verified": self.verified}


class SolverService:
    def __init__(self, cache_dir: Path | None = None, ladder: str = "learned", radius: int = 7, beam: int = 4096) -> None:
        self.cache_dir = Path(cache_dir or os.environ.get("CUBE_SCANNER_LADDER", DATA / "ladders" / f"{ladder}_r{radius}"))
        self.ladder_name = ladder
        self.radius = radius
        self.beam = beam
        self.status = SolverStatus()
        self._lock = threading.Lock()
        self._puzzle = None
        self._ladder = None

    def load(self) -> SolverStatus:
        """Build the ladder from its cache. Safe to call from several threads; the first does the work."""
        with self._lock:
            if self.status.state in ("ready", "missing"):
                return self.status
            if not any(self.cache_dir.glob("rung*.json")):
                self.status = SolverStatus("missing", f"No built ladder in {self.cache_dir}. Run: python tools/import_ladder.py")
                return self.status
            self.status = SolverStatus("loading", "Loading the solver's tables and value networks")
            try:
                use_solver()
                from core.ladder import Ladder
                from core.perm_model import build_point_model
                from core.structure import analyse
                from puzzles.maltese_gear.ladder import LADDERS
                from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle

                t0 = time.time()
                puzzle = build_maltese_gear_puzzle()
                model = build_point_model(puzzle)
                structure = analyse(model)
                log = self.status.log
                ladder = Ladder(model, structure, LADDERS[self.ladder_name], mitm_radius=self.radius, log=log.append)
                ladder.build(cache_dir=self.cache_dir)
                if not ladder.ready():
                    missing = [r.spec.name for r in ladder.rungs if r.table is None and r.heuristic is None]
                    self.status = SolverStatus("missing", f"The ladder in {self.cache_dir} is incomplete; untrained rungs: {missing}", log)
                    return self.status
                self._puzzle, self._ladder = puzzle, ladder
                self.status = SolverStatus("ready", f"Loaded in {time.time() - t0:.1f} s from {self.cache_dir}", log)
            except Exception as err:  # the reason is shown in the UI
                self.status = SolverStatus("error", f"{type(err).__name__}: {err}")
            return self.status

    def solve(self, state144, beam: int | None = None) -> Solution:
        if self.status.state != "ready":
            status = self.load()
            if status.state != "ready":
                raise RuntimeError(status.detail)
        puzzle, ladder = self._puzzle, self._ladder
        state = np.asarray(state144, dtype=np.uint8)
        if state.shape != puzzle.goal.shape:
            raise ValueError(f"the solver array has {puzzle.goal.shape[0]} entries, got {state.shape}")
        t0 = time.time()
        with self._lock:                   # one search at a time on the GPU
            try:
                word, per_rung = ladder.solve(state, beam=beam or self.beam)
            except (LookupError, ValueError) as err:
                raise NotSolvable(str(err)) from err
        seconds = time.time() - t0
        if not np.array_equal(puzzle.apply_word(state, list(word)), puzzle.goal):
            raise AssertionError("the solution does not solve the state on the reference simulator")
        alg = ladder.alg
        names = puzzle.move_names
        return Solution(
            word=alg.format(word),
            moves=[{"name": names[b], "amount": int(e)} for b, e in alg.runs(word)],
            quarter_turns=len(word),
            face_turns=int(alg.face_turns(word)),
            per_rung={r.spec.name: int(c) for r, c in zip(ladder.rungs, per_rung)},
            seconds=seconds,
        )
