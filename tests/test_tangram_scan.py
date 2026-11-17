"""The Tangram Cube Extreme from six face-on photos, without the networks: readings are made from known
states (with misreads and hidden stickers), the photos shuffled, and the pipeline must work out which
face each shows and decode the true state."""
from __future__ import annotations

import numpy as np
import pytest

from scanner.puzzles.tangram.decode import decode_exact
from scanner.puzzles.tangram.layout import EMPTY, load_layout
from scanner.puzzles.tangram.vision.geometry import HIDDEN, N_CLASSES, face_geometry
from scanner.puzzles.tangram.vision.pipeline import TangramVision, assemble, evidence


def reading(content: np.ndarray, rng: np.random.Generator, sure: float = 0.95, wrong: float = 0.0, hidden: float = 0.0) -> np.ndarray:
    """(29, 8) probabilities for one face's positions with true ``content`` (29 values, colour or EMPTY)."""
    p = np.full((len(content), N_CLASSES), (1 - sure) / (N_CLASSES - 1))
    for j, c in enumerate(content):
        if c != EMPTY and rng.random() < hidden:
            p[j] = 0.01
            p[j, HIDDEN] = 0.93
        elif rng.random() < wrong:
            w = int(rng.choice([k for k in range(7) if k != c]))
            p[j, c] = 0.3
            p[j, w] = 0.6
        else:
            p[j, c] = sure
        p[j] /= p[j].sum()
    return p


@pytest.fixture(scope="module")
def vision() -> TangramVision:
    return TangramVision(device="cpu")


def photos_of(orient: np.ndarray, rng, **noise) -> tuple[list[np.ndarray], list[int]]:
    lay = load_layout()
    geom = face_geometry(lay)
    obs = lay.observe(orient)
    order = [int(f) for f in rng.permutation(6)]
    return [reading(obs[geom.position_of[f]], rng, **noise) for f in order], order


def test_every_face_reading_lands_on_the_right_positions():
    lay = load_layout()
    geom = face_geometry(lay)
    rng = np.random.default_rng(1)
    o = lay.random_state(rng, 80)
    obs = lay.observe(o)
    logp = assemble([evidence(reading(obs[geom.position_of[f]], rng, sure=0.99)) for f in range(6)], tuple(range(6)), geom, lay.n_positions)
    assert np.array_equal(logp.argmax(1), obs)
    # Every position is in a photo; the turned-out ones in two.
    counts = np.bincount(geom.position_of.ravel(), minlength=lay.n_positions)
    assert set(counts[lay.on_cube]) == {1} and set(counts[~lay.on_cube]) == {2}


@pytest.mark.parametrize("seed", range(6))
def test_shuffled_photos_decode_to_the_true_state(vision, seed):
    lay = load_layout()
    rng = np.random.default_rng(100 + seed)
    o = lay.random_state(rng, 120)
    readings, order = photos_of(o, rng, wrong=0.02, hidden=0.05)
    a = vision.find_faces(readings)
    assert a.decode.legal
    assert np.array_equal(a.decode.orient, o)
    assert list(a.faces) == order
    assert a.margin > 0
    assert a.colours.is_default


def test_solved_puzzle_is_found_in_the_core_frame(vision):
    """Six symmetric assignments fit a solved puzzle equally well; only one is reachable with the core fixed."""
    lay = load_layout()
    rng = np.random.default_rng(7)
    readings, order = photos_of(lay.solved_orient, rng)
    a = vision.find_faces(readings)
    assert a.decode.legal and np.array_equal(a.decode.orient, lay.solved_orient)
    assert list(a.faces) == order


def relabelled(readings: list[np.ndarray], paint) -> list[np.ndarray]:
    """What the networks read on a cube whose colour layout is ``paint`` (model colour c shows as paint[c])."""
    out = []
    for r in readings:
        x = r.copy()
        x[:, list(paint)] = r[:, :6]
        out.append(x)
    return out


#: A DaYan Tangram Cube Extreme: turning corners red-yellow-blue and orange-white-green.
DAYAN = (1, 0, 2, 3, 5, 4)


@pytest.mark.parametrize("seed", range(3))
def test_a_cube_stickered_another_way_is_read_in_its_own_colours(vision, seed):
    lay = load_layout()
    rng = np.random.default_rng(300 + seed)
    o = lay.random_state(rng, 120)
    readings, order = photos_of(o, rng, wrong=0.02, hidden=0.05)
    a = vision.find_faces(relabelled(readings, DAYAN))
    assert a.colours.paint == DAYAN and a.colours.kind == "rearranged"
    assert a.decode.legal and np.array_equal(a.decode.orient, o) and list(a.faces) == order
    assert a.margin >= 10
    assert a.alternatives and a.alternatives[0][0].is_default       # the usual colours are offered to switch back


@pytest.mark.parametrize("kind", ["rearranged", "mirrored", "other"])
def test_every_kind_of_layout_is_read(vision, kind):
    lay = load_layout()
    want = next(c for c in lay.colour_layouts if c.kind == kind)
    rng = np.random.default_rng(41)
    o = lay.random_state(rng, 120)
    readings, order = photos_of(o, rng, wrong=0.01)
    a = vision.find_faces(relabelled(readings, want.paint))
    assert a.colours == want
    assert a.decode.legal and np.array_equal(a.decode.orient, o) and list(a.faces) == order


def test_a_solved_cube_keeps_the_usual_colours(vision):
    """Every layout looks the same on a solved cube (six one-colour faces), so nothing is switched."""
    lay = load_layout()
    rng = np.random.default_rng(8)
    readings, _ = photos_of(lay.solved_orient, rng)
    a = vision.find_faces(relabelled(readings, DAYAN))
    assert a.colours.is_default
    assert a.decode.legal and np.array_equal(a.decode.orient, lay.solved_orient)


def test_a_usual_cube_read_badly_is_not_given_another_layout(vision):
    lay = load_layout()
    for seed in range(4):
        rng = np.random.default_rng(500 + seed)
        readings, _ = photos_of(lay.random_state(rng, 120), rng, wrong=0.12, hidden=0.1)
        assert vision.find_faces(readings).colours.is_default


def test_the_fast_search_scores_as_the_plain_assignment(vision):
    from scanner.puzzles.tangram.decode import assign_logp

    lay = load_layout()
    rng = np.random.default_rng(9)
    readings, _ = photos_of(lay.random_state(rng, 120), rng, wrong=0.05, hidden=0.05)
    evid = [evidence(r) for r in readings]
    layouts = lay.colour_layouts[:5]
    top, rest = vision.search.rank(evid, layouts, [0.0] * 5, 12)
    assert all(a.score >= b.score for a, b in zip(top, top[1:])) and rest <= top[-1].score + 1e-9
    for c in top:
        lp = assemble([layouts[c.layout].model_columns(e) for e in evid], c.faces, vision.geom, lay.n_positions)
        assert abs(assign_logp(lp, lay)[1] - c.loglik) < 1e-6


def test_decode_exact_reports_impossible_states():
    lay = load_layout()
    obs = lay.solved.copy()
    p = int(np.nonzero(obs != EMPTY)[0][0])
    obs[p] = (obs[p] + 1) % 6
    d = decode_exact(obs)
    assert not d.legal and d.problems
    assert any(p in q.positions for q in d.problems)
