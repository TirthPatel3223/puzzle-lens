"""Colour layouts (scanner.core.colours): which layouts a scan can tell apart, and their kinds."""
from __future__ import annotations

from collections import Counter

import numpy as np

from scanner.core.colours import (
    ColourLayout,
    colour_action,
    cube_rotations,
    describe,
    find_class,
    kind_of,
    layout_classes,
)

#: Face directions of the six colours, in the Tangram's order (+x -x +y -y +z -z).
AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)


def test_a_cube_with_every_rotation_has_30_layout_classes():
    """The Maltese Gear Cube: any whole-cube turn is allowed, so only mirror images and other opposite
    pairs are told apart."""
    acts = [colour_action(m, AXES) for m in cube_rotations()]
    classes = layout_classes(acts, AXES)
    assert len(classes) == 30
    assert Counter(c.kind for c in classes) == {"default": 1, "mirrored": 1, "other": 28}
    assert classes[0] == ColourLayout.default()


def test_the_tangram_has_120_classes_and_the_dayan_layout_is_a_rearrangement():
    from scanner.puzzles.tangram.layout import load_layout

    lay = load_layout()
    classes = lay.colour_layouts
    assert len(classes) == 120
    assert Counter(c.kind for c in classes) == {"default": 1, "rearranged": 3, "mirrored": 4, "other": 112}
    dayan = classes[find_class(classes, (1, 0, 2, 3, 5, 4), lay.colour_actions)]
    assert dayan.paint == (1, 0, 2, 3, 5, 4) and dayan.kind == "rearranged"
    assert describe(dayan, lay.colour_names) == "orange and red swapped; green and blue swapped"


def test_kinds():
    default = {tuple(range(6))}
    assert kind_of((1, 0, 2, 3, 4, 5), AXES, default) == "mirrored"          # one pair swapped: a mirror image
    assert kind_of((1, 0, 3, 2, 4, 5), AXES, default) == "rearranged"        # two swapped: a half turn
    assert kind_of((0, 2, 1, 3, 4, 5), AXES, default) == "other"             # red now opposite yellow


def test_colours_go_both_ways():
    lay = ColourLayout((1, 0, 2, 3, 5, 4), "rearranged")
    model = np.array([0, 1, 2, 3, 4, 5, 6])
    assert lay.to_cube(model).tolist() == [1, 0, 2, 3, 5, 4, 6]
    assert lay.to_model(lay.to_cube(model)).tolist() == model.tolist()
    x = np.arange(8.0)[None]                                                   # 6 colours, then empty, hidden
    assert lay.model_columns(x).tolist() == [[1, 0, 2, 3, 5, 4, 6, 7]]
