"""The DaYan Tangram Cube Extreme, and the original Tangram Cube, as GatherPuzzles.

The move tables are not written by hand. They are compiled (``core/geometry.py``) from the 3D model of
the puzzle in the Cube_Modeling project (``src/puzzle/*.ts``), exported there by
``tools/export/geometry.gen.ts`` into ``geometry_<variant>.json`` next to this file. To refresh them
after a change to the model, run the exporter and copy the two files here.

Moves (one 60° step on the Extreme, 120° on the original; a flip is 180° and its own inverse):

    U, U'   top corner layer, clockwise / anticlockwise seen from the top corner (1,1,1)
    D, D'   bottom corner layer, seen from the bottom corner (-1,-1,-1)
    F R L   flips of the front, right-back and left-back edges, about (1,-1,0), (0,1,-1), (-1,0,1)

State: the colour at every position a sticker can occupy, or ``EMPTY``. The Extreme has 120 positions:
66 on the cube and 54 where stickers go when a 60° turn changes the puzzle's shape. The hidden core never
moves, so the state is in a fixed frame and the solved state is unique (no turned copy of it is reachable).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.gather_engine import GatherPuzzle
from core.geometry import CompiledGeometry, compile_geometry, load_geometry

VARIANTS = ("extreme", "original")
EMPTY = 6                          # colour value of a position no sticker covers (colours are 0-5)

# Group orders, with every piece counted as distinct. The Extreme's is the count in Cube_Modeling's notes
# (docs/tangram-cube.md: about 4.75e28); the original's is Jaap Scherphuis' 3,110,400 x 36.
GROUP_ORDER = {"extreme": 47_478_243_018_446_808_809_472_000_000, "original": 111_974_400}


def geometry_file(variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"unknown Tangram variant {variant!r}; expected one of {VARIANTS}")
    return Path(__file__).with_name(f"geometry_{variant}.json")


@lru_cache(maxsize=None)
def compiled(variant: str = "extreme") -> CompiledGeometry:
    out = compile_geometry(load_geometry(geometry_file(variant)))
    assert out.empty == EMPTY
    return out


def build_tangram_puzzle(variant: str = "extreme") -> GatherPuzzle:
    return compiled(variant).puzzle
