"""The Maltese Gear Cube as the Python side knows it.

Nothing here is geometry or move logic of its own: it is all read from
``shared/maltese/layout.json`` and ``solver_map.json``, which the browser side generates from
the 3D engine (``web/tools/export-shared.gen.ts``). See ``web/src/maltese/observation.ts`` for
what an observation is: per face a 4x4 grid of colour cells and 4 gears.

Colours are indices 0-5: white, yellow, green, blue, red, purple (the engine's U D F B L R).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from scanner.paths import SHARED

N_CELLS = 96
N_GEARS = 24
N_COLOURS = 6
STATE_DIM = 144
FACE_NAMES = ("U", "D", "F", "B", "L", "R")


@dataclass(frozen=True)
class Observation:
    """What the cube shows: 96 cell colours, 24 gear colours, 24 gear quarter turns."""

    cells: np.ndarray
    gear_colour: np.ndarray
    gear_quarter: np.ndarray

    def __post_init__(self) -> None:
        for name, n in (("cells", N_CELLS), ("gear_colour", N_GEARS), ("gear_quarter", N_GEARS)):
            a = np.asarray(getattr(self, name), dtype=np.int64)
            if a.shape != (n,):
                raise ValueError(f"{name} needs {n} entries, got shape {a.shape}")
            object.__setattr__(self, name, a)

    def copy(self) -> "Observation":
        return Observation(self.cells.copy(), self.gear_colour.copy(), self.gear_quarter.copy())

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Observation)
            and np.array_equal(self.cells, other.cells)
            and np.array_equal(self.gear_colour, other.gear_colour)
            and np.array_equal(self.gear_quarter, other.gear_quarter)
        )

    def to_json(self) -> dict:
        return {
            "cells": self.cells.tolist(),
            "gearColour": self.gear_colour.tolist(),
            "gearQuarter": self.gear_quarter.tolist(),
        }

    @staticmethod
    def from_json(d: dict) -> "Observation":
        return Observation(np.array(d["cells"]), np.array(d["gearColour"]), np.array(d["gearQuarter"]))


@dataclass(frozen=True)
class Rearrangement:
    """How a move or a whole-cube rotation rearranges what is shown.

    ``new.cells[i] = old.cells[cells[i]]``; a gear arrives from place ``gears[i]`` and its
    quarter grows by ``dk[i]``.
    """

    cells: np.ndarray
    gears: np.ndarray
    dk: np.ndarray

    def apply(self, obs: Observation) -> Observation:
        return Observation(
            obs.cells[self.cells],
            obs.gear_colour[self.gears],
            (obs.gear_quarter[self.gears] + self.dk) % 4,
        )

    @staticmethod
    def from_json(d: dict) -> "Rearrangement":
        return Rearrangement(np.array(d["cells"]), np.array(d["gears"]), np.array(d["dk"]))


@dataclass(frozen=True)
class SlotOption:
    """One way to fill a slot: a piece, how it is turned, and the colours its cells then show."""

    piece: int
    turn: int
    colours: tuple[int, ...]


@dataclass(frozen=True)
class Slot:
    index: int
    name: str
    kind: str                       # "corner", "centre" or "edge"
    cells: tuple[int, ...]
    options: tuple[SlotOption, ...]


@dataclass(frozen=True)
class SolverMap:
    colour_code: np.ndarray         # solver colour code of each colour index
    moves: dict[str, tuple[str, int]]  # solver move -> (engine face, +1 or -1)
    cell: np.ndarray                # solver index of each of our cells
    gear: np.ndarray                # solver gear number of each of our gear places
    gear_sign: int
    gear_offset: np.ndarray


class Layout:
    def __init__(self, layout: dict, solver_map: dict) -> None:
        self.raw = layout
        self.colour_names: list[str] = [c["name"] for c in layout["colours"]]
        self.colour_hex: list[str] = [c["hex"] for c in layout["colours"]]
        self.pieces: list[dict] = layout["pieces"]
        self.corners = [
            Slot(c["slot"], c["name"], "corner", tuple(c["cells"]),
                 tuple(SlotOption(o["piece"], o["ori"], tuple(o["colours"])) for o in c["options"]))
            for c in layout["corners"]
        ]
        self.radial = [
            Slot(r["slot"], r["name"], r["kind"], tuple(r["cells"]),
                 tuple(SlotOption(o["piece"], o["k"], tuple(o["colours"])) for o in r["options"]))
            for r in layout["radial"]
        ]
        #: engine gear slot at each gear place
        self.gear_slot = np.array([g["slot"] for g in layout["gears"]])
        self.gear_names = [g["name"] for g in layout["gears"]]
        self.moves = {name: Rearrangement.from_json(m) for name, m in layout["moves"].items()}
        self.rotations = [Rearrangement.from_json(m) for m in layout["rotations"]]
        self.group_mul = np.array(layout["cubeGroup"]["mul"])
        self.group_inv = np.array(layout["cubeGroup"]["inv"])
        self.corner_slot_of_ori = np.array(layout["cubeGroup"]["cornerSlotOfOri"])
        self.pegged = layout["pegged"]
        s = layout["solved"]
        self.solved = Observation(np.array(s["cells"]), np.array(s["gearColour"]), np.array(s["gearQuarter"]))
        self.grid = layout["grid"]
        self.geometry = layout["geometry"]
        self.gear_pieces_by_colour: dict[int, list[int]] = {}
        for p in self.pieces:
            if p["kind"] == "gear":
                self.gear_pieces_by_colour.setdefault(p["colours"][0], []).append(p["id"])
        self.solver = SolverMap(
            np.array(solver_map["colourCode"]),
            {k: (v["face"], int(v["amount"])) for k, v in solver_map["moves"].items()},
            np.array(solver_map["cell"]),
            np.array(solver_map["gear"]),
            int(solver_map["gearSign"]),
            np.array(solver_map["gearOffset"]),
        )

    # -- moves -------------------------------------------------------------------
    def apply(self, obs: Observation, word: str | Sequence[str]) -> Observation:
        """Apply engine moves, e.g. ``"U R' F2"`` (a number repeats the turn, U'2 or U2' both work)."""
        for name, count in parse_word(word):
            m = self.moves[name]
            for _ in range(count):
                obs = m.apply(obs)
        return obs

    # -- colour layouts (scanner.core.colours) ----------------------------------------------------
    @cached_property
    def colour_normals(self) -> np.ndarray:
        """(6, 3): the direction of the face each colour covers when solved (colour c is face c: U D F B L R)."""
        face_dir = {"U": (0, 1, 0), "D": (0, -1, 0), "F": (0, 0, 1), "B": (0, 0, -1), "L": (-1, 0, 0), "R": (1, 0, 0)}
        out = np.zeros((N_COLOURS, 3))
        for f, name in enumerate(FACE_NAMES):
            colour = int(np.bincount(self.solved.cells[16 * f:16 * f + 16], minlength=N_COLOURS).argmax())
            out[colour] = face_dir[name]
        return out

    @cached_property
    def colour_actions(self) -> list[tuple[int, ...]]:
        """How each whole-cube turn relabels the colours: all 24, since a scan's frame is free (the
        decoder turns the cube to the solver's frame itself)."""
        from scanner.core.colours import colour_action, cube_rotations

        return [colour_action(m, self.colour_normals) for m in cube_rotations()]

    @cached_property
    def colour_layouts(self) -> list:
        """One ``ColourLayout`` per class a scan can tell apart (30), the model's own first."""
        from scanner.core.colours import layout_classes

        return layout_classes(self.colour_actions, self.colour_normals)

    def random_state(self, rng: np.random.Generator, length: int = 40) -> Observation:
        names = list(self.moves)
        obs = self.solved
        for i in rng.integers(0, len(names), size=length):
            obs = self.moves[names[int(i)]].apply(obs)
        return obs

    # -- the solver's array ---------------------------------------------------------
    def to_solver(self, obs: Observation) -> np.ndarray:
        """The solver's 144-entry array. The observation must be in the solver's frame."""
        sm = self.solver
        out = np.zeros(STATE_DIM, dtype=np.uint8)
        out[sm.cell] = sm.colour_code[obs.cells]
        out[96 + sm.gear] = sm.colour_code[obs.gear_colour]
        out[120 + sm.gear] = (sm.gear_sign * obs.gear_quarter + sm.gear_offset) % 4
        return out

    def from_solver(self, state: Sequence[int]) -> Observation:
        sm = self.solver
        state = np.asarray(state, dtype=np.int64)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"the solver array has {STATE_DIM} entries, got {state.shape}")
        if state[:120].max() > 5 or state.min() < 0 or state[120:].max() > 3:
            raise ValueError("colour codes are 0-5 and gear rotations 0-3")
        back = np.argsort(sm.colour_code)
        return Observation(
            back[state[sm.cell]],
            back[state[96 + sm.gear]],
            (sm.gear_sign * (state[120 + sm.gear] - sm.gear_offset)) % 4,
        )

    def solver_word_to_engine(self, word: str | Sequence[str]) -> list[tuple[str, int]]:
        """A solver word ("U L'3 F2") as engine moves [(face, signed quarter turns)]."""
        out = []
        for name, count in parse_word(word):
            face, amount = self.solver.moves[name]
            out.append((face, amount * count))
        return out


def parse_word(word: str | Sequence[str]) -> list[tuple[str, int]]:
    """'U L2 F'3' -> [("U", 1), ("L", 2), ("F'", 3)]."""
    import re

    tokens = word.split() if isinstance(word, str) else list(word)
    out = []
    for token in tokens:
        m = re.fullmatch(r"([A-Za-z])(\d*)('?)(\d*)", token)
        if m is None or (m.group(2) and m.group(4)):
            raise ValueError(f"cannot read move {token!r}")
        out.append((m.group(1).upper() + m.group(3), int(m.group(2) or m.group(4) or 1)))
    return out


@lru_cache(maxsize=1)
def load_layout(shared: Path | None = None) -> Layout:
    d = (shared or SHARED) / "maltese"
    return Layout(
        json.loads((d / "layout.json").read_text(encoding="utf-8")),
        json.loads((d / "solver_map.json").read_text(encoding="utf-8")),
    )
