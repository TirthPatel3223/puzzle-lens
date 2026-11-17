"""Subgroup ladders for the Tangram Cube Extreme (and the original Tangram Cube).

The structure module finds these orbits in the compiled tables. ``tests/test_tangram.py`` checks that
each name below holds the pieces it says. Positions are sticker frames, so a piece with several stickers
that never trade places shows up as one orbit per sticker; a rung needs only one of them.

Extreme (120 positions, 14 orbits). Every 12-position orbit holds 6 pieces in 12 places: 6 on the
cube and 6 where a 60° turn puts them. The empty places show ``EMPTY``.

    SMALL_TRI_A, SMALL_TRI_B   two classes of 6 pieces with a small triangular sticker
    LARGE_TRI_A, LARGE_TRI_B   two classes of 6 pieces with a larger triangular sticker
    PENTAGON                   6 pieces with a pentagonal sticker
    CORNER (+ CORNER_2, _3)    the 6 cube corners away from the turning axis (3 stickers; orbit per sticker)
    TOP, BOTTOM                the turning corners: they only spin in place (6 states each)
    MID_EDGE_F, _R, _L         the middle-layer edges on the flip axes: each only flips in place
    MIDDLE                     the 3 other middle-layer pieces, permuted by the flips

Rung order, measured (docs/tangram.md):

* ``frame`` first: the turning corners, the middle trio and the middle edges. 3,456 cosets, solved
  with the primitive moves in at most 13 steps. They need a rung of their own: with every other piece
  solved a turning corner can still be turned 120° (3 states) and two middle edges flipped (4 states).
* then the large pieces, then the small triangles: the small triangles have the cheapest pure
  macros (``core/discover.py``), so they go last.

``LEARNED_LADDER`` solves the large pieces (about 5.7e16 cosets) and the small triangles (about
2.4e8) with value networks. ``TABLE_LADDER`` splits them into one exact table per class (at most
665,280 cosets each); it needs no training, and its later rungs use longer macros.
"""
from __future__ import annotations

from core.ladder import RungSpec

# -- Extreme ----------------------------------------------------------------------------------------------
(SMALL_TRI_A, LARGE_TRI_A, PENTAGON, CORNER, CORNER_2, CORNER_3, SMALL_TRI_B, TOP, LARGE_TRI_B,
 MID_EDGE_F, MID_EDGE_R, MIDDLE, MID_EDGE_L, BOTTOM) = range(14)

FRAME = [(TOP, "full"), (BOTTOM, "full"), (MIDDLE, "full"), (MID_EDGE_F, "full"), (MID_EDGE_R, "full"),
         (MID_EDGE_L, "full")]
LARGE = [(LARGE_TRI_A, "full"), (LARGE_TRI_B, "full"), (PENTAGON, "full"), (CORNER, "full")]
SMALL = [(SMALL_TRI_A, "full"), (SMALL_TRI_B, "full")]

TABLE_LADDER: list[RungSpec] = [
    RungSpec("frame", FRAME, max_actions=7),
    RungSpec("large_tri_a", [(LARGE_TRI_A, "full")], max_actions=600),
    RungSpec("large_tri_b", [(LARGE_TRI_B, "full")], max_actions=600),
    RungSpec("pentagons", [(PENTAGON, "full")], max_actions=600),
    RungSpec("corners", [(CORNER, "full")], max_actions=600),
    RungSpec("small_tri_a", [(SMALL_TRI_A, "full")], max_actions=600),
    RungSpec("small_tri_b", [(SMALL_TRI_B, "full")], max_actions=600),
]

LEARNED_LADDER: list[RungSpec] = [
    RungSpec("frame", FRAME, max_actions=7),
    RungSpec("large", LARGE, mode="learned", max_actions=128),
    RungSpec("small", SMALL, mode="learned", max_actions=128),
]

LADDERS = {"table": TABLE_LADDER, "learned": LEARNED_LADDER}

# -- original Tangram Cube (111,974,400 states; 42 positions, nothing changes shape) ----------------------
(O_TOP, O_CORNER, O_CORNER_2, O_CORNER_3, O_SINGLE, O_MID_EDGE_F, O_MID_EDGE_R, O_MIDDLE, O_MID_EDGE_L,
 O_BOTTOM) = range(10)

ORIGINAL_LADDERS = {
    "table": [
        RungSpec("frame", [(O_TOP, "full"), (O_BOTTOM, "full"), (O_MIDDLE, "full"), (O_MID_EDGE_F, "full"),
                           (O_MID_EDGE_R, "full"), (O_MID_EDGE_L, "full")], max_actions=7),
        RungSpec("pieces", [(O_CORNER, "full"), (O_SINGLE, "full")], max_actions=600),
    ],
}
