"""Colour layouts: which colour a particular cube has where the model has each of its six colours.

Cubes are not all stickered alike. The same six colours can sit on a puzzle in other arrangements (a
Tangram Cube Extreme whose turning corners are red-yellow-blue and orange-white-green instead of the
model's orange-yellow-green and red-white-blue), in a mirror image, or with other colours opposite
each other. Every sticker of such a cube still has one colour throughout, so a single permutation says
it all:

    paint[c] = the colour this cube shows where the model shows colour c

The networks read the colours they see (the cube's), the decoder and the solver work in the model's
colours (so the solver needs nothing new), and the 3D model is painted in the cube's colours.

**Which layouts can be told apart.** Turning the whole puzzle by one of its symmetries relabels the
colours too, and a scan cannot tell that apart from a different layout: the layouts ``paint`` and
``paint o a`` (for a symmetry's action ``a`` on the colours) describe the same cube. A puzzle names its
symmetries, and ``layout_classes`` keeps one layout per class (720 / number of symmetries).

**Kinds.** Relative to the model's own scheme, a layout is

- ``default``     the model's scheme (perhaps with the whole puzzle turned);
- ``rearranged``  the same scheme turned against the puzzle's cuts (same opposite colours, same
                  handedness), which only a puzzle with fewer symmetries than the cube has;
- ``mirrored``    the mirror image of the scheme (same opposite colours, other handedness);
- ``other``       other colours opposite each other.

A scan picks the layout that explains its photos best, after a **prior** that favours the usual scheme:
a layout other than the default has to explain the photos better by that many nats. This keeps a cube
stickered the usual way from ever being read as another layout by chance.

**No per-colour balancing.** A camera that makes red look orange shows as too many "orange" readings.
Shifting each colour's log-probability until the counts match (11 of each) was tried and dropped: every
arrangement of the real pieces already has exactly those counts, so the decoder already takes the 11
reddest stickers as the red ones, and a per-colour shift adds the same to every arrangement (it cannot
change the decoded state). Shown to the person it only moves the doubt from the red stickers to the
orange ones. Telling red from orange better takes the readers themselves (trained on shades that vary
more, and on real scans).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

N = 6
IDENTITY = tuple(range(N))
KINDS = ("default", "rearranged", "mirrored", "other")

#: Nats a layout of each kind must explain the photos better by than the default, to be believed.
#: Measured on rendered scans of usually stickered cubes (see the puzzle's evaluate.py ``layouts``).
DEFAULT_PRIOR = {"default": 0.0, "rearranged": 12.0, "mirrored": 16.0, "other": 16.0}


@dataclass(frozen=True)
class ColourLayout:
    #: paint[c]: the colour this cube shows where the model shows colour c.
    paint: tuple[int, ...]
    kind: str

    @property
    def is_default(self) -> bool:
        return self.kind == "default"

    @property
    def inverse(self) -> tuple[int, ...]:
        inv = [0] * N
        for c, p in enumerate(self.paint):
            inv[p] = c
        return tuple(inv)

    def to_model(self, values: np.ndarray) -> np.ndarray:
        """Colour indices as the cube shows them -> the model's (values >= 6, e.g. 'empty', are kept)."""
        v = np.asarray(values, dtype=np.int64)
        inv = np.array(self.inverse + tuple(range(N, max(N, int(v.max(initial=0)) + 1))))
        return inv[v]

    def to_cube(self, values: np.ndarray) -> np.ndarray:
        """The model's colour indices -> the colours this cube shows there."""
        v = np.asarray(values, dtype=np.int64)
        paint = np.array(self.paint + tuple(range(N, max(N, int(v.max(initial=0)) + 1))))
        return paint[v]

    def model_columns(self, x: np.ndarray) -> np.ndarray:
        """Per-colour scores whose first six columns are the cube's colours -> the model's (column c of
        the result is column paint[c] of ``x``); further columns (empty, hidden) are kept."""
        idx = list(self.paint) + list(range(N, x.shape[-1]))
        return x[..., idx]

    def to_json(self) -> dict:
        return {"paint": list(self.paint), "kind": self.kind}

    @staticmethod
    def default() -> "ColourLayout":
        return ColourLayout(IDENTITY, "default")


def colour_action(matrix: np.ndarray, normals: np.ndarray) -> tuple[int, ...]:
    """How a whole-puzzle turn ``matrix`` moves the colours: colour c goes where colour a[c]'s face was."""
    n = np.asarray(normals, dtype=np.float64)
    out = []
    for c in range(N):
        v = np.asarray(matrix, dtype=np.float64) @ n[c]
        out.append(int(np.argmax(n @ v)))
    if sorted(out) != list(range(N)):
        raise ValueError("the matrix does not map the faces onto each other")
    return tuple(out)


def cube_rotations() -> list[np.ndarray]:
    """The 24 rotations of a cube, as 3x3 matrices."""
    out = []
    for axes in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, a in enumerate(axes):
                m[a, i] = signs[i]
            if round(np.linalg.det(m)) == 1:
                out.append(m)
    return out


def kind_of(paint: Sequence[int], normals: np.ndarray, default_class: Iterable[tuple[int, ...]]) -> str:
    """Which kind of layout ``paint`` is (see the module notes). ``normals[c]``: the direction of the
    face colour c covers on the solved model."""
    paint = tuple(int(p) for p in paint)
    if paint in set(default_class):
        return "default"
    n = np.asarray(normals, dtype=np.float64)
    opposite = [int(np.argmin(n @ n[c])) for c in range(N)]
    if any(paint[opposite[c]] != opposite[paint[c]] for c in range(N)):
        return "other"
    # The linear map taking each face's direction to its new colour's face direction (sum n n^T = 2 I).
    m = sum(np.outer(n[paint[c]], n[c]) for c in range(N)) / 2.0
    return "rearranged" if np.linalg.det(m) > 0 else "mirrored"


def _simplicity(p: tuple[int, ...]) -> tuple:
    """Prefer, as a class's representative, the layout that is quickest to say: fewest colours changed,
    then swaps rather than longer cycles."""
    moved = sum(p[c] != c for c in range(N))
    seen, long_cycles = set(), 0
    for c in range(N):
        if c in seen:
            continue
        k, j = 0, c
        while j not in seen:
            seen.add(j)
            j = p[j]
            k += 1
        long_cycles += k > 2
    return (moved, long_cycles, p)


def layout_classes(actions: Sequence[Sequence[int]], normals: np.ndarray) -> list[ColourLayout]:
    """One layout per class of layouts a scan cannot tell apart (``paint ~ paint o a`` for each
    symmetry action ``a``), the default first, then by kind."""
    acts = [tuple(int(x) for x in a) for a in actions]
    if IDENTITY not in acts:
        acts.append(IDENTITY)
    default_class = {tuple(a) for a in acts}
    seen: set[tuple[int, ...]] = set()
    reps: list[ColourLayout] = []
    for p in itertools.permutations(range(N)):
        if p in seen:
            continue
        orbit = {tuple(p[a[c]] for c in range(N)) for a in acts}
        seen |= orbit
        rep = IDENTITY if IDENTITY in orbit else min(orbit, key=_simplicity)
        reps.append(ColourLayout(rep, kind_of(rep, normals, default_class)))
    reps.sort(key=lambda lay: (KINDS.index(lay.kind), _simplicity(lay.paint)))
    return reps


def find_class(layouts: Sequence[ColourLayout], paint: Sequence[int], actions: Sequence[Sequence[int]]) -> int:
    """Index in ``layouts`` of the class that ``paint`` belongs to."""
    p = tuple(int(x) for x in paint)
    acts = [tuple(a) for a in actions] + [IDENTITY]
    orbit = {tuple(p[a[c]] for c in range(N)) for a in acts}
    for i, lay in enumerate(layouts):
        if lay.paint in orbit:
            return i
    raise ValueError(f"{p} is not a colour layout")


def representative(layouts: Sequence[ColourLayout], paint: Sequence[int] | None, actions: Sequence[Sequence[int]]) -> ColourLayout:
    """The layout a page asked for (``paint``; None: the model's own), as the class representative the
    scanner uses. Raises ValueError if ``paint`` is not six colours each once."""
    if paint is None:
        return layouts[0]
    p = [int(x) for x in paint]
    if sorted(p) != list(range(N)):
        raise ValueError("colours must list each of the six colours 0-5 once")
    return layouts[find_class(layouts, p, actions)]


def page_json(layout: ColourLayout, names: Sequence[str], nats: float | None = None) -> dict:
    """A colour layout for the page: its permutation, kind, and the colours that differ by name."""
    out = {"paint": list(layout.paint), "kind": layout.kind, "usual": layout.is_default,
           "changed": [{"model": names[i], "cube": names[p]} for i, p in enumerate(layout.paint) if p != i]}
    if nats is not None:
        out["nats"] = round(float(nats), 1)
    return out


def describe(layout: ColourLayout, names: Sequence[str]) -> str:
    """In words, for a person: 'orange and red swapped, green and blue swapped'."""
    if layout.is_default:
        return "the usual colours"
    parts, seen = [], set()
    for c in range(N):
        if c in seen or layout.paint[c] == c:
            continue
        cyc, j = [], c
        while j not in seen:
            seen.add(j)
            cyc.append(j)
            j = layout.paint[j]
        if len(cyc) == 2:
            parts.append(f"{names[cyc[0]]} and {names[cyc[1]]} swapped")
        else:
            parts.append(", ".join(f"{names[a]} where the model has {names[b]}" for b, a in ((x, layout.paint[x]) for x in cyc)))
    return "; ".join(parts)
