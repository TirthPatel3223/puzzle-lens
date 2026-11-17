"""Copy the solver's built ladder (tables and trained value networks) into this repository.

The ladder is what makes the solver fast: three exact tables and two value networks, about
430 MB, built and trained in Cube_Solving_Pipeline. It is not versioned there or here.

    python tools/import_ladder.py                       # Maltese Gear Cube, from ../Cube_Solving_Pipeline
    python tools/import_ladder.py --from D:/somewhere/data/ladders/learned_r7
    python tools/import_ladder.py --puzzle tangram      # Tangram Cube Extreme (once its solver is trained)

To rebuild it from nothing instead, see third_party/cube_solver/UPSTREAM_README.md.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scanner.paths import DATA  # noqa: E402

PIPELINE_LADDERS = ROOT.parent / "Cube_Solving_Pipeline" / "data" / "ladders"
#: Per puzzle: the ladder folder under data/ladders, the same in the pipeline and here
#: (scanner/puzzles/<key>/solver.py reads it; the pipeline's puzzles/registry.py PuzzleEntry.cache_dir writes it).
NAMES = {"maltese": "learned_r7", "tangram": "tangram_extreme/learned_r7"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", choices=sorted(NAMES), default="maltese")
    ap.add_argument("--from", dest="source", help="a built ladder directory (default: the pipeline's for --puzzle)")
    ap.add_argument("--name", help="folder name under data/ladders (default: the one the puzzle's solver reads)")
    args = ap.parse_args()
    args.name = args.name or NAMES[args.puzzle]
    args.source = args.source or str(PIPELINE_LADDERS / args.name)

    source = Path(args.source)
    if not source.is_dir():
        sys.exit(f"no ladder at {source}")
    target = DATA / "ladders" / args.name
    target.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for f in sorted(source.iterdir()):
        if not f.is_file() or f.suffix == ".bak":
            continue
        dest = target / f.name
        if dest.exists() and dest.stat().st_size == f.stat().st_size and dest.stat().st_mtime >= f.stat().st_mtime:
            skipped += 1
            continue
        shutil.copy2(f, dest)
        copied += 1
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print(f"ladder in {target}: {copied} files copied, {skipped} already up to date")


if __name__ == "__main__":
    main()
