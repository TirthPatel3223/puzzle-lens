"""The six photos of a scan may come in any order and any way up: find_arrangement works out which
face each shows from the corners. These tests feed it readings made from known cubes (no networks
needed), shuffled and turned, and check that the decoded cube is the true one."""
from __future__ import annotations

import numpy as np
import pytest

from scanner.puzzles.maltese.decode import decode_exact
from scanner.puzzles.maltese.layout import N_COLOURS, Observation, load_layout
from scanner.puzzles.maltese.vision.pipeline import FaceReading, Vision


def soft(labels: np.ndarray, n: int, sure: float, rng: np.random.Generator, wrong: float = 0.0) -> np.ndarray:
    """One row of probabilities per label, ``sure`` on the label; with probability ``wrong`` a row
    instead favours a random other label (a misread the decoder has to overrule)."""
    p = np.full((len(labels), n), (1 - sure) / (n - 1))
    for i, c in enumerate(labels):
        pick = c
        if rng.random() < wrong:
            pick = int(rng.choice([k for k in range(n) if k != c]))
            p[i, c] = 0.3
            p[i, pick] = sure - 0.3 + (1 - sure) / (n - 1)
            p[i] /= p[i].sum()
            continue
        p[i, pick] = sure
    return p


def photos_of(obs: Observation, layout, rng, wrong: float = 0.0):
    """Readings of the six faces of ``obs`` in a random order, each turned a random number of quarter turns."""
    order = rng.permutation(6)
    turns = rng.integers(0, 4, size=6)
    readings = []
    for f, k in zip(order, turns):
        own = FaceReading(
            soft(obs.cells[16 * f:16 * f + 16], N_COLOURS, 0.97, rng, wrong),
            soft(obs.gear_colour[4 * f:4 * f + 4], N_COLOURS, 0.97, rng),
            soft(obs.gear_quarter[4 * f:4 * f + 4], 4, 0.97, rng),
        )
        shown = own.turned(-int(k) % 4, layout)            # what a photo of the face turned k shows
        assert np.allclose(shown.turned(int(k), layout).cells, own.cells)
        readings.append(shown)
    return readings, order, turns


@pytest.mark.parametrize("seed", range(12))
def test_any_order_and_rotation(seed):
    layout = load_layout()
    rng = np.random.default_rng(seed)
    truth = layout.random_state(rng, length=60)
    readings, _, _ = photos_of(truth, layout, rng)
    vision = Vision()
    arrangement = vision.find_arrangement(readings)
    d = arrangement.decode
    assert d is not None and d.legal
    assert d.observation == decode_exact(truth, layout).observation
    assert arrangement.margin > 10


@pytest.mark.parametrize("seed", range(6))
def test_any_order_with_misreads(seed):
    """A few wrong stickers (5%) do not throw the arrangement off; the decoder overrules them."""
    layout = load_layout()
    rng = np.random.default_rng(100 + seed)
    truth = layout.random_state(rng, length=60)
    readings, _, _ = photos_of(truth, layout, rng, wrong=0.05)
    arrangement = Vision().find_arrangement(readings)
    assert arrangement.decode.legal
    assert arrangement.decode.observation == decode_exact(truth, layout).observation


def relabelled(readings, paint):
    """What the networks read on a cube whose colour layout is ``paint`` (model colour c shows as paint[c])."""
    out = []
    for r in readings:
        cells = r.cells.copy()
        cells[:, list(paint)] = r.cells
        gears = r.gear_colour.copy()
        gears[:, list(paint)] = r.gear_colour
        out.append(FaceReading(cells, gears, r.gear_quarter))
    return out


@pytest.mark.parametrize("kind", ["mirrored", "other"])
@pytest.mark.parametrize("seed", range(2))
def test_a_cube_stickered_another_way_is_read_in_its_own_colours(kind, seed):
    """A mirror-image cube, or one with other colours opposite each other: found, and decoded right."""
    layout = load_layout()
    want = next(c for c in layout.colour_layouts if c.kind == kind)
    rng = np.random.default_rng(40 + seed)
    truth = layout.random_state(rng, length=60)
    readings, _, _ = photos_of(truth, layout, rng, wrong=0.02)
    arrangement, colours, alternatives = Vision().find_colours(relabelled(readings, want.paint))
    assert colours == want
    assert arrangement.decode.legal and arrangement.decode.observation == decode_exact(truth, layout).observation
    assert arrangement.margin > 10
    assert alternatives[0][0].is_default


def test_the_maltese_layouts_are_its_rotations_mirror_and_other_pairs():
    from collections import Counter

    layout = load_layout()
    assert len(layout.colour_layouts) == 30
    assert Counter(c.kind for c in layout.colour_layouts) == {"default": 1, "mirrored": 1, "other": 28}


@pytest.mark.parametrize("seed", range(3))
def test_a_usual_cube_read_badly_keeps_the_usual_colours(seed):
    layout = load_layout()
    rng = np.random.default_rng(70 + seed)
    truth = layout.random_state(rng, length=60)
    readings, _, _ = photos_of(truth, layout, rng, wrong=0.12)
    _, colours, _ = Vision().find_colours(readings)
    assert colours.is_default
