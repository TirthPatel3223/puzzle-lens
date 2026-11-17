"""Reading and writing move sequences of one puzzle, and the lengths reported for them.

Each puzzle has its own conventions. The Maltese Gear Cube writes three anticlockwise quarter turns as
``U'3`` and reports quarter turns and face turns. The Tangram Cube writes ``U3'`` (as its simulator
does) and reports steps (60° turns of a corner layer, flips) and turns (a turn of a layer by any amount,
or a flip, counts one). A ``Notation`` holds those conventions on top of the word algebra, which merges
consecutive turns of one layer.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

from core.harvest import WordAlgebra

Metric = Callable[[WordAlgebra, Sequence[int]], int]

_TOKEN = re.compile(r"([A-Za-z])('?)(\d*)('?)")


def steps(alg: WordAlgebra, word: Sequence[int]) -> int:
    """Number of primitive moves, as written."""
    return len(word)


def turns(alg: WordAlgebra, word: Sequence[int]) -> int:
    """Turns of a layer by any amount (or flips) after merging consecutive turns of one layer."""
    return len(alg.runs(word))


def face_turns(alg: WordAlgebra, word: Sequence[int]) -> int:
    """The Maltese Gear Cube's face-turn metric (see ``WordAlgebra.face_turns``)."""
    return alg.face_turns(word)


class Notation:
    """``prime_then_count``: ``U'3`` (True) or ``U3'`` (False). ``metrics``: (name, function) pairs,
    the first being the one a solution's length is quoted in. ``legend``: one paragraph explaining the
    notation, for solution pages."""

    def __init__(self, alg: WordAlgebra, prime_then_count: bool, metrics: Sequence[tuple[str, Metric]],
                 legend: str = "") -> None:
        self.alg = alg
        self.names = tuple(alg.model.move_names)
        self.prime_then_count = prime_then_count
        self.metrics = tuple(metrics)
        self.legend = legend

    def format(self, word: Sequence[int]) -> str:
        """Consecutive turns of one layer merged, e.g. ``U2'`` for two anticlockwise steps."""
        if self.prime_then_count:
            return self.alg.format(word)
        parts = []
        for b, e in self.alg.runs(word):
            n = abs(e)
            parts.append(self.names[b] + (str(n) if n > 1 else "") + ("'" if e < 0 else ""))
        return " ".join(parts)

    def parse(self, text: str) -> list[int]:
        """Move indices of a sequence such as ``U2' F R U3``. ``U'2`` and ``U2'`` are both read. The count
        repeats the move; a prime on a move that is its own inverse (a flip) changes nothing."""
        out: list[int] = []
        for token in text.replace(",", " ").split():
            m = _TOKEN.fullmatch(token)
            if m is None or (m.group(2) and m.group(4)):
                raise ValueError(f"cannot read move {token!r}; moves are {' '.join(self.names)}")
            base, prime, count = m.group(1), bool(m.group(2) or m.group(4)), int(m.group(3) or 1)
            name = base + ("'" if prime else "")
            if name not in self.names and prime and base in self.names \
                    and self.alg.inverse[self.names.index(base)] == self.names.index(base):
                name = base
            if name not in self.names:
                raise ValueError(f"cannot read move {token!r}; moves are {' '.join(self.names)}")
            out.extend([self.names.index(name)] * count)
        return out

    def lengths(self, word: Sequence[int]) -> dict[str, int]:
        return {name: fn(self.alg, word) for name, fn in self.metrics}

    def describe(self, word: Sequence[int]) -> str:
        """For example "476 quarter turns, 411 face turns"."""
        return ", ".join(f"{v} {k}" for k, v in self.lengths(word).items())
