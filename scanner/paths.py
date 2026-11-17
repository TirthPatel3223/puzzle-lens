"""Where things live. Everything is relative to the repository; nothing is hardcoded."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared"
#: Everything a run produces (renders, real photos, checkpoints, the solver's ladder cache). Not versioned.
DATA = Path(os.environ.get("CUBE_SCANNER_DATA", ROOT / "data"))
SOLVER_DIR = ROOT / "third_party" / "cube_solver"
WEB_DIST = ROOT / "web" / "dist"

#: The puzzle the scanner started with keeps its files where they were before there were several puzzles.
_LEGACY = "maltese"


def checkpoints_dir(puzzle: str) -> Path:
    """The trained networks of one puzzle (``maltese``, ``tangram``): data/checkpoints/<puzzle>.

    The Maltese Gear Cube's are still read from data/checkpoints itself until a data/checkpoints/maltese
    folder exists."""
    d = DATA / "checkpoints" / puzzle
    if puzzle == _LEGACY and not d.exists():
        return DATA / "checkpoints"
    return d


def real_dir(puzzle: str) -> Path:
    """Real photos kept from the app: <dir>/pending (waiting to be checked) and <dir>/scans (labelled).
    data/real for the Maltese Gear Cube, data/real/<puzzle> for the others."""
    return DATA / "real" if puzzle == _LEGACY else DATA / "real" / puzzle


def use_solver() -> None:
    """Make the copied solver importable as it imports itself (``core``, ``puzzles``)."""
    path = str(SOLVER_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
