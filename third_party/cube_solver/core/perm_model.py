"""Labelled point model of a GatherPuzzle, and words compiled into single tables.

A GatherPuzzle state is a vector of visible values. That hides which piece is where when
pieces look alike, and it mixes two kinds of coordinate (colours that are only moved, and
counters that are moved and incremented). The point model removes both problems:

* every coordinate that is only moved becomes one point;
* every counter coordinate with modulus q becomes q points (j, 0..q-1), and a move that
  adds `a` to the counter shifts those points cyclically by `a`.

A move is then a plain permutation of N labelled points. A *labelled state* is the vector
``lab`` with ``lab[pos]`` = the home label of the point now at position ``pos``. Applying a
move with gather table ``g`` gives ``lab_next = lab[g]``. The solved labelled state is
``arange(N)``.

A word (sequence of moves) compiles into one gather table, so a macro costs one gather
however long it is. Tables compose as ``compose(a, b) = a[b]`` for "a, then b".

The visible state is recovered from a labelled state by colouring: a moved-only point
shows the goal value of its home coordinate; a counter point shows 1 if it is its counter's
marker (the residue equal to the goal value) and 0 otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from core.gather_engine import GatherPuzzle

Word = tuple[int, ...]


@dataclass(frozen=True)
class PointModel:
    puzzle: GatherPuzzle
    n: int                      # number of points
    gens: np.ndarray            # (M, n) gather tables of the primitive moves
    inverse: np.ndarray         # (M,) index of the inverse move
    coord: np.ndarray           # (n,) coordinate of the GatherPuzzle each point belongs to
    residue: np.ndarray         # (n,) residue r of a counter point, 0 for a moved-only point
    offset: np.ndarray          # (D,) first point of each coordinate
    count: np.ndarray           # (D,) points per coordinate: 1, or the modulus for a counter
    visible: np.ndarray         # (n,) visible colour of each home label
    move_names: tuple[str, ...] = field(default=())

    @property
    def num_moves(self) -> int:
        return self.gens.shape[0]

    # -- words ------------------------------------------------------------------
    def identity(self) -> np.ndarray:
        return np.arange(self.n, dtype=np.int32)

    def compile(self, word: Sequence[int]) -> np.ndarray:
        """Gather table of a word: applying it to labels ``lab`` gives ``lab[table]``."""
        table = self.identity()
        for m in word:
            table = table[self.gens[m]]
        return table

    def compile_many(self, words: Iterable[Sequence[int]]) -> np.ndarray:
        words = list(words)
        out = np.empty((len(words), self.n), dtype=np.int32)
        for k, w in enumerate(words):
            out[k] = self.compile(w)
        return out

    def invert_word(self, word: Sequence[int]) -> Word:
        return tuple(int(self.inverse[m]) for m in reversed(word))

    def word_str(self, word: Sequence[int]) -> str:
        return " ".join(self.move_names[m] for m in word)

    # -- labelled <-> visible ---------------------------------------------------
    def labels_to_state(self, lab: np.ndarray) -> np.ndarray:
        """Visible GatherPuzzle state(s) shown by labelled state(s). lab: (..., n)."""
        lab = np.asarray(lab)
        colours = self.visible[lab]                       # colour shown at each position
        d = self.puzzle.dim
        out = np.zeros(lab.shape[:-1] + (d,), dtype=np.uint8)
        single = self.count == 1
        out[..., single] = colours[..., self.offset[single]]
        for j in np.nonzero(~single)[0]:
            block = colours[..., self.offset[j]:self.offset[j] + self.count[j]]
            out[..., j] = np.argmax(block, axis=-1)       # position of the marker
        return out

    def state_to_colours(self, state: np.ndarray) -> np.ndarray:
        """Colour shown at every point position by visible state(s). state: (..., D)."""
        state = np.asarray(state)
        out = np.zeros(state.shape[:-1] + (self.n,), dtype=np.int16)
        out[...] = state[..., self.coord]
        counter = self.count[self.coord] > 1
        out[..., counter] = (state[..., self.coord[counter]] == self.residue[counter])
        return out


def build_point_model(puzzle: GatherPuzzle) -> PointModel:
    idx, add, modulus = puzzle.idx, puzzle.add, puzzle.modulus
    m, d = idx.shape

    # A coordinate is a counter if any move ever adds to it, or to a coordinate it can
    # be carried to. Coordinates in one orbit of the gather maps must be expanded alike.
    parent = list(range(d))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for mv in range(m):
        for i in range(d):
            a, b = find(i), find(int(idx[mv, i]))
            if a != b:
                parent[a] = b
    root = np.array([find(i) for i in range(d)])
    touched = (add != 0).any(axis=0)
    counter_roots = set(root[touched].tolist())
    is_counter = np.array([r in counter_roots for r in root])
    for r in counter_roots:
        mods = set(modulus[root == r].tolist())
        assert len(mods) == 1, f"coordinates in one orbit have different moduli: {mods}"

    count = np.where(is_counter, modulus.astype(np.int64), 1)
    offset = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64)
    n = int(count.sum())
    coord = np.repeat(np.arange(d), count)
    residue = np.concatenate([np.arange(c) for c in count]).astype(np.int64)

    gens = np.empty((m, n), dtype=np.int32)
    for mv in range(m):
        src = idx[mv, coord]                                   # source coordinate
        q = count[coord]
        # position (j, r) receives the point that was at (src(j), r - add) before the move
        r_src = np.where(q > 1, (residue - add[mv, coord]) % q, 0)
        gens[mv] = offset[src] + r_src
        assert np.array_equal(np.sort(gens[mv]), np.arange(n)), f"move {mv} is not a permutation"

    goal = puzzle.goal.astype(np.int64)
    visible = np.where(count[coord] > 1, (residue == goal[coord]).astype(np.int64), goal[coord])

    model = PointModel(puzzle=puzzle, n=n, gens=gens, inverse=np.asarray(puzzle.inverse),
                       coord=coord, residue=residue, offset=offset, count=count,
                       visible=visible, move_names=tuple(puzzle.move_names))
    return model


# -- free-group word utilities ---------------------------------------------------

def reduce_word(word: Sequence[int], inverse: np.ndarray, orders: Sequence[int] | None = None) -> Word:
    """Cancel adjacent inverse pairs. With ``orders``, also drop a run of one move that
    reaches the move's order (e.g. eight identical quarter turns of an order-8 face)."""
    out: list[int] = []
    for m in word:
        m = int(m)
        if out and out[-1] == int(inverse[m]):
            out.pop()
            continue
        out.append(m)
        if orders is not None:
            k = orders[m]
            if len(out) >= k and all(x == m for x in out[-k:]):
                del out[-k:]
    return tuple(out)


def concat_reduce(a: Sequence[int], b: Sequence[int], inverse: np.ndarray) -> Word:
    """Concatenate two reduced words and cancel across the seam."""
    a = list(a)
    i = 0
    while a and i < len(b) and a[-1] == int(inverse[int(b[i])]):
        a.pop()
        i += 1
    return tuple(a) + tuple(int(x) for x in b[i:])


def move_orders(model: PointModel) -> list[int]:
    """Order of each primitive move as a permutation of the points."""
    orders = []
    ident = model.identity()
    for m in range(model.num_moves):
        table = model.gens[m].copy()
        k = 1
        while not np.array_equal(table, ident):
            table = table[model.gens[m]]
            k += 1
        orders.append(k)
    return orders
