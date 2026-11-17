"""Train a puzzle's networks on its synthetic photos.

    python -m scanner.vision.train <reader|locator> --puzzle <maltese|tangram> --data <dataset> [options]

``--puzzle`` defaults to the Maltese Gear Cube. Everything after it is the puzzle's own training script's
(scanner/puzzles/<key>/vision/train.py; run with -h to list its options).
"""
from __future__ import annotations

import sys

from scanner.vision.common import split_puzzle


def main() -> None:
    puzzle, rest = split_puzzle(sys.argv[1:])
    from scanner.puzzles import module

    module(puzzle, "vision.train").main(rest)


if __name__ == "__main__":
    main()
