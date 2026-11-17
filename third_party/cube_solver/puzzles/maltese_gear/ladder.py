"""Subgroup ladders for the Maltese Gear Cube.

The structure module finds four moving orbits in the move tables. They are named here once
so the rung lists below read as intended:

    CORNER   7 pieces x 3 stickers   (the eighth corner is pegged and never moves)
    RING    18 pieces x 4 stickers   (12 edges and 6 centres; any piece can sit in any slot)
    GEARCOL 24 gear slots, the colour of the gear in each
    GEARROT 24 gear slots x 4 residues, the rotation of the gear in each

The rung order follows the design note's rules (docs/cayley_graphs_...md, part V), checked
against the tables (docs/hierarchical_solver.md records the measurements):

* corners first: each face turn has order 4 on corners and 8 elsewhere, so X^4 fixes the
  corners for free and the corner stabiliser has generators of 4 to 8 quarter turns;
* gear rotation before gear colour: the words that keep every rotation value in place
  still permute the gears as the whole alternating group, with generators of 4 to 12 moves.
  The other order needs colour-preserving, rotation-changing words, and none exists
  within 14 moves;
* the ring last: there are words of only 12 moves that fix all corners and all gears.

``TABLE_LADDER`` uses exact tables only. Rungs that would be too large are split with nested
colour masks (place the pieces showing colour a, then a and b, ...). It needs no training
and always solves, but its late rungs use long macros.

``LEARNED_LADDER`` replaces each split chain by a single rung with a trained value network:
all gear colours at once (3.2e15 cosets) and the whole ring at once (1.3e22 cosets). The rungs
are far too large for tables, but their macros are the cheap ones (4 to 14 quarter turns),
so solutions are several times shorter. This is the ladder the solver uses.
"""
from __future__ import annotations

from core.ladder import RungSpec

CORNER, RING, GEARCOL, GEARROT = 0, 1, 2, 3

# Colours of the gears, in the order they are placed by the table ladder.
_GEAR_ORDER = [2, 1, 4, 0, 5]          # the sixth colour is then in place automatically
_RING_ORDER = [0, 1, 2, 3, 4]


def _nested(orbit: int, kind: str, order: list[int]) -> list[list[tuple]]:
    return [[(orbit, kind, tuple(order[:k + 1]))] for k in range(len(order))]


# Table rungs can afford many actions: an exact table over 8.4 million cosets and 600 moves
# takes about a minute on a GPU, and more moves only ever shorten the solutions.
_GEAR_PREFIX = [
    RungSpec("corners", [(CORNER, "full")], max_actions=6),
    RungSpec("gear_parity", [(GEARROT, "sub0")], max_actions=600),
    RungSpec("gear_rot", [(GEARROT, "full")], max_actions=600),
]

TABLE_LADDER: list[RungSpec] = (
    _GEAR_PREFIX
    + [RungSpec(f"gear_col{k + 1}", feats, max_actions=160) for k, feats in enumerate(_nested(GEARCOL, "full", _GEAR_ORDER))]
    + [RungSpec("ring_type", [(RING, "type")], max_actions=160),
       RungSpec("ring_shape", [(RING, "shape")], max_actions=160)]
    + [RungSpec(f"ring_place{k + 1}", feats, max_actions=160) for k, feats in enumerate(_nested(RING, "multiset", _RING_ORDER))]
    + [RungSpec("ring_flip", [(RING, "full")], max_actions=160)]
)

# gear_col: the 132 moves of up to 10 quarter turns only carry the gears around as 12 rigid
# pairs, so the selection also keeps the 244 cheapest moves that split pairs (376 actions).
# With a single one of them the rung is solvable on paper and untrainable in practice.
LEARNED_LADDER: list[RungSpec] = (
    _GEAR_PREFIX
    + [RungSpec("gear_col", [(GEARCOL, "full")], mode="learned", max_actions=128),
       RungSpec("ring", [(RING, "full")], mode="learned", max_actions=72)]
)

LADDERS = {"table": TABLE_LADDER, "learned": LEARNED_LADDER}
