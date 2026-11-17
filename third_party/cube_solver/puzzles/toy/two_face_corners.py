"""A small puzzle for tests and demonstrations: the corners under two faces.

The corner stickers of the Maltese Gear Cube under U and L only. It is the two-generator
subgroup of the 2x2x2 cube: six corners move, each with three orientations, and there are
29,160 states (120 corner arrangements times 3^5 twists).

It is small enough to enumerate completely, so every part of the ladder machinery can be
checked against brute force, yet it has the structure that matters: pieces with several
stickers, an orientation invariant, and a natural two-rung ladder (place the corners, then
twist them).
"""
from __future__ import annotations

from core.gather_engine import GatherPuzzle, restrict
from core.ladder import RungSpec
from puzzles.maltese_gear import tables as md
from puzzles.maltese_gear.puzzle import build_maltese_gear_puzzle

STATE_COUNT = 29_160


def build_two_face_corners() -> GatherPuzzle:
    full = build_maltese_gear_puzzle()
    stickers = [s for corner in md.CORNERS for s in corner]
    moves = [md.MOVE_LIST.index(m) for m in ("U", "U'", "L", "L'")]
    return restrict(full, stickers, moves, name="two_face_corners")


# Place the corners (which corner is where), then twist them.
LADDER = [
    RungSpec("place", [(0, "multiset")], max_actions=16),
    RungSpec("twist", [(0, "full")], max_actions=64),
]
