"""Score a puzzle's networks the way they are used: from the picture alone.

    python -m scanner.vision.evaluate faces|scans|real --puzzle maltese [options]
    python -m scanner.vision.evaluate photos|scans|real --puzzle tangram [options]

``--puzzle`` defaults to the Maltese Gear Cube. Everything after it is the puzzle's own evaluation script's
(scanner/puzzles/<key>/vision/evaluate.py; run with -h to list its options).
"""
from __future__ import annotations

import sys

from scanner.vision.common import split_puzzle


def main() -> None:
    puzzle, rest = split_puzzle(sys.argv[1:])
    from scanner.puzzles import module

    module(puzzle, "vision.evaluate").main(rest)


if __name__ == "__main__":
    main()
