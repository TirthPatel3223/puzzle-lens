"""Which arrangements of the Maltese Gear Cube can moves reach? A proof by counting.

The pieces can be arranged in a huge "ambient" group A: 7 free corners in any order and twist,
18 centre/edge pieces in any order and quarter turn, 24 gears in any order and quarter turn
(the eighth corner is held fixed; the solver's three faces never move it). Every map from A to
an abelian group factors through six numbers:

    corner twist sum (mod 3)        sign of the corner permutation
    centre/edge turn sum (mod 4)    sign of the centre/edge permutation
    gear turn sum (mod 4)           sign of the gear permutation

This script computes, exactly (Schreier-Sims, sympy):

1. the order of the group G the moves really generate, on 192 labelled points;
2. the values the six numbers take on G, and the order of "everything in A with those values".

If the two orders are equal, G *is* that set: the six numbers are the only constraints. What
survives in what a camera can see (a centre's own turn and which same-coloured gear is which
are invisible) is then exactly the list in scanner/puzzles/maltese/decode.py.

    python tools/derive_invariants.py          # about half a minute
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sympy.combinatorics import Permutation, PermutationGroup  # noqa: E402

from scanner.puzzles.maltese.layout import load_layout  # noqa: E402

N = 96 + 96            # 96 sticker cells, then 24 gear places x 4 quarter turns
RADIAL_CYCLE = [0, 1, 3, 2]   # a slot's four cells (stored by quadrant) in rotational order
MODULI = (3, 2, 4, 2, 4, 2)
NAMES = ("corner twist sum", "corner sign", "centre/edge turn sum", "centre/edge sign", "gear turn sum", "gear sign")


def sign_bit(perm: list[int]) -> int:
    seen = [False] * len(perm)
    odd = 0
    for i in range(len(perm)):
        n, j = 0, i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            n += 1
        if n and n % 2 == 0:
            odd ^= 1
    return odd


def main() -> None:
    layout = load_layout()
    faces = sorted({face for face, _ in layout.solver.moves.values()})     # the three faces the solver turns
    print(f"generators: the engine's {', '.join(faces)} (the solver's U, L, F)")

    perms = {}
    for face in faces:
        m = layout.moves[face]
        p = np.arange(N)
        p[m.cells] = np.arange(96)                           # the content of cell m.cells[i] goes to i
        for i, src in enumerate(m.gears):
            for q in range(4):
                p[96 + 4 * src + q] = 96 + 4 * i + (q + m.dk[i]) % 4
        perms[face] = p

    t0 = time.time()
    order = PermutationGroup([Permutation(p.tolist()) for p in perms.values()]).order()
    print(f"order of the move group: {order}  ({float(order):.4e}, {time.time() - t0:.0f} s)")

    # Corner slots with one handedness: carried round from the first by the moves themselves.
    pegged = set(layout.pegged["cells"])
    free = [c for c in layout.corners if not set(c.cells) & pegged]
    slot_of = {c: s.index for s in free for c in s.cells}
    cyc = {free[0].index: list(free[0].cells)}
    grew = True
    while grew:
        grew = False
        for p in perms.values():
            for s, cells in list(cyc.items()):
                img = [int(p[c]) for c in cells]
                s2 = slot_of[img[0]]
                if s2 not in cyc:
                    cyc[s2] = img
                    grew = True
    slots = sorted(cyc)
    radial_of = {c: r.index for r in layout.radial for c in r.cells}

    def numbers(p: np.ndarray) -> tuple[int, ...]:
        cperm, twist = [], 0
        for s in slots:
            img = int(p[cyc[s][0]])
            s2 = slot_of[img]
            cperm.append(slots.index(s2))
            twist += cyc[s2].index(img)
        rperm, turn = [], 0
        for r in layout.radial:
            img = int(p[r.cells[RADIAL_CYCLE[0]]])
            r2 = layout.radial[radial_of[img]]
            rperm.append(r2.index)
            turn += [r2.cells[i] for i in RADIAL_CYCLE].index(img)
        gperm, gturn = [], 0
        for a in range(24):
            img = int(p[96 + 4 * a]) - 96
            gperm.append(img // 4)
            gturn += img % 4
        return (twist % 3, sign_bit(cperm), turn % 4, sign_bit(rperm), gturn % 4, sign_bit(gperm))

    images = {face: numbers(p) for face, p in perms.items()}
    print("what one quarter turn does to the six numbers:")
    for face, v in images.items():
        print(f"  {face}: " + ", ".join(f"{n} {x}" for n, x in zip(NAMES, v)))
    reached = {(0,) * 6}
    todo = [(0,) * 6]
    while todo:
        x = todo.pop()
        for v in images.values():
            y = tuple((a + b) % m for a, b, m in zip(x, v, MODULI))
            if y not in reached:
                reached.add(y)
                todo.append(y)

    ambient = (3**7 * math.factorial(7)) * (4**18 * math.factorial(18)) * (4**24 * math.factorial(24))
    expected = ambient * len(reached) // math.prod(MODULI)
    print(f"values reached: {len(reached)} of {math.prod(MODULI)}: {sorted(reached)}")
    print(f"arrangements with those values: {expected}")
    if order != expected:
        sys.exit("NOT EQUAL: there are constraints beyond the six numbers; decode.py's legality check is incomplete.")
    print("equal: the six numbers are the only constraints.")
    print()
    print("Seen through a camera (a centre's turn and the identity of same-coloured gears are invisible):")
    print("  corner twists sum to 0 mod 3; the centre/edge permutation is even; gear turns sum to 0 mod 4.")


if __name__ == "__main__":
    main()
