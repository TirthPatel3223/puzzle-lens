"""The decoder: from what the cube shows to a legal arrangement in the solver's frame."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scanner.paths import SHARED
from scanner.puzzles.maltese.decode import decode_exact, decode_probs
from scanner.puzzles.maltese.layout import N_CELLS, N_COLOURS, N_GEARS, Observation, load_layout

L = load_layout()
SOLVER_MOVES = ["U", "U'", "L", "L'", "F", "F'"]


def solver_frame_state(rng: np.random.Generator, length: int = 60) -> tuple[Observation, list[str]]:
    """A state reached with the solver's own moves, so it is in the solver's frame by construction."""
    obs = L.solved
    word = [SOLVER_MOVES[int(i)] for i in rng.integers(0, 6, size=length)]
    for name in word:
        face, amount = L.solver.moves[name]
        obs = L.moves[face if amount > 0 else face + "'"].apply(obs)
    return obs, word


def probabilities(obs: Observation, sure: float = 0.97) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = np.full((N_CELLS, N_COLOURS), (1 - sure) / 5)
    cell[np.arange(N_CELLS), obs.cells] = sure
    gc = np.full((N_GEARS, N_COLOURS), (1 - sure) / 5)
    gc[np.arange(N_GEARS), obs.gear_colour] = sure
    gq = np.full((N_GEARS, 4), (1 - sure) / 3)
    gq[np.arange(N_GEARS), obs.gear_quarter] = sure
    return cell, gc, gq


def test_layout_matches_the_solver_tables_on_random_words():
    tables = json.loads((SHARED / "maltese" / "solver_tables.json").read_text())
    idx, add, goal = np.array(tables["idx"]), np.array(tables["add"]), np.array(tables["goal"])
    mod = np.array([256] * 120 + [4] * 24)
    rng = np.random.default_rng(1)
    for _ in range(300):
        obs, word = solver_frame_state(rng, int(rng.integers(1, 50)))
        theirs = goal.copy()
        for name in word:
            m = tables["moves"].index(name)
            theirs = (theirs[idx[m]] + add[m]) % mod
        assert np.array_equal(L.to_solver(obs), theirs)
        assert L.from_solver(theirs) == obs


def test_solved_cube_is_legal_and_is_the_goal():
    r = decode_exact(L.solved)
    assert r.legal and r.problems == []
    assert r.rotation == 0
    assert r.state144 == L.to_solver(L.solved).tolist()
    assert r.observation == L.solved


def test_any_orientation_decodes_to_the_same_solver_state():
    rng = np.random.default_rng(2)
    for _ in range(40):
        obs, _ = solver_frame_state(rng)
        want = L.to_solver(obs).tolist()
        for rot in rng.choice(24, size=4, replace=False):
            turned = L.rotations[int(rot)].apply(obs)
            r = decode_exact(turned)
            assert r.legal, [p.message for p in r.problems]
            assert r.state144 == want
            assert r.observation == obs
            # cell_source says where each solver-frame cell was in the frame given
            assert np.array_equal(turned.cells[r.cell_source], obs.cells)
            assert np.array_equal(turned.gear_colour[r.gear_source], obs.gear_colour)


def test_scrambles_with_all_six_faces_are_legal():
    rng = np.random.default_rng(3)
    for _ in range(60):
        r = decode_exact(L.random_state(rng, 50))
        assert r.legal, [p.message for p in r.problems]
        assert all(r.observation.cells[c] == col for c, col in zip(L.pegged["cells"], L.pegged["colours"]))


def test_poses_reproduce_the_observation():
    rng = np.random.default_rng(4)
    for _ in range(30):
        r = decode_exact(L.random_state(rng, 50))
        assert sorted(r.slot[p["id"]] for p in L.pieces if p["kind"] == "corner") == list(range(8))
        assert sorted(r.slot[p["id"]] for p in L.pieces if p["kind"] in ("centre", "edge")) == list(range(18))
        assert sorted(r.slot[p["id"]] for p in L.pieces if p["kind"] == "gear") == list(range(24))
        for s in [*L.corners, *L.radial]:
            piece = next(p["id"] for p in L.pieces
                         if (p["kind"] == "corner") == (s.kind == "corner") and p["kind"] != "gear" and r.slot[p["id"]] == s.index)
            option = next(o for o in s.options if o.piece == piece and o.turn == r.ori[piece])
            assert tuple(r.observation.cells[list(s.cells)]) == option.colours


def _twist_one_corner(obs: Observation) -> Observation:
    s = next(c for c in L.corners if c.index != L.pegged["slot"])
    cells = obs.cells.copy()
    a, b, c = s.cells
    cells[[a, b, c]] = cells[[b, c, a]]
    return Observation(cells, obs.gear_colour, obs.gear_quarter)


def _swap_two_edges(obs: Observation) -> Observation:
    edges = [s for s in L.radial if s.kind == "edge"]
    a, b = edges[0], edges[1]
    cells = obs.cells.copy()
    cells[list(a.cells)], cells[list(b.cells)] = obs.cells[list(b.cells)], obs.cells[list(a.cells)]
    return Observation(cells, obs.gear_colour, obs.gear_quarter)


def test_a_twisted_corner_is_reported():
    r = decode_exact(_twist_one_corner(L.solved))
    assert not r.legal and r.state144 is None
    assert [p.code for p in r.problems] == ["corner-twist"]


def test_two_swapped_edges_are_reported():
    # In the solved cube two edges exchanged show colours no edge slot can show at home, so use a
    # state where the exchange keeps every slot a real piece: swap the pieces, keeping their own turns.
    r = decode_exact(_swap_two_edges(L.solved))
    assert not r.legal
    assert {p.code for p in r.problems} <= {"ring-parity", "no-such-piece"}
    rng = np.random.default_rng(5)
    seen = 0
    for _ in range(20):
        obs = L.random_state(rng, 40)
        good = decode_exact(obs)
        # exchange two centre pieces: a centre looks the same in any slot it is in, turned any way
        centres = [p["id"] for p in L.pieces if p["kind"] == "centre"]
        sa, sb = L.radial[good.slot[centres[0]]], L.radial[good.slot[centres[1]]]
        cells = good.observation.cells.copy()
        ca, cb = cells[sa.cells[0]], cells[sb.cells[0]]
        cells[list(sa.cells)] = cb
        cells[list(sb.cells)] = ca
        bad = decode_exact(Observation(cells, good.observation.gear_colour, good.observation.gear_quarter))
        assert not bad.legal
        assert [p.code for p in bad.problems] == ["ring-parity"]
        seen += 1
    assert seen == 20


def test_a_gear_turned_alone_is_reported():
    q = L.solved.gear_quarter.copy()
    q[5] = 1
    r = decode_exact(Observation(L.solved.cells, L.solved.gear_colour, q))
    assert not r.legal
    assert [p.code for p in r.problems] == ["gear-sum"]
    # two quarter turns that cancel are fine
    q[9] = 3
    assert decode_exact(Observation(L.solved.cells, L.solved.gear_colour, q)).legal


def test_impossible_pieces_are_named():
    cells = L.solved.cells.copy()
    cells[0] = 1                                   # a white star turned yellow
    r = decode_exact(Observation(cells, L.solved.gear_colour, L.solved.gear_quarter))
    assert not r.legal and r.observation is None
    assert r.problems[0].code == "no-such-piece" and 0 in r.problems[0].cells
    gc = L.solved.gear_colour.copy()
    gc[0] = 1
    r = decode_exact(Observation(L.solved.cells, gc, L.solved.gear_quarter))
    assert {p.code for p in r.problems} == {"gear-count"}
    assert "5 yellow gears" in " ".join(p.message for p in r.problems)


def test_probabilities_of_a_true_state_decode_to_it():
    rng = np.random.default_rng(6)
    for _ in range(20):
        obs = L.random_state(rng, 40)
        want = decode_exact(obs)
        got = decode_probs(*probabilities(obs))
        assert got.legal and got.corrections == []
        assert got.state144 == want.state144


def test_a_misread_gear_quarter_is_repaired_from_the_sum():
    rng = np.random.default_rng(7)
    for _ in range(30):
        obs = L.random_state(rng, 40)
        want = decode_exact(obs)
        cell, gc, gq = probabilities(obs)
        g = int(rng.integers(0, N_GEARS))
        wrong = (obs.gear_quarter[g] + int(rng.integers(1, 4))) % 4
        gq[g] = 0.02
        gq[g, wrong] = 0.55                         # the reader prefers the wrong quarter...
        gq[g, obs.gear_quarter[g]] = 0.41           # ...but not by much
        got = decode_probs(cell, gc, gq)
        assert got.legal
        assert got.state144 == want.state144
        assert [(c.kind, c.index) for c in got.corrections] == [("gearQuarter", g)]


def test_a_confident_wrong_gear_quarter_is_flagged_not_changed():
    obs = L.random_state(np.random.default_rng(8), 40)
    cell, gc, gq = probabilities(obs, sure=0.9999)
    gq[3] = 1e-5
    gq[3, (obs.gear_quarter[3] + 1) % 4] = 1 - 3e-5
    got = decode_probs(cell, gc, gq)
    assert not got.legal
    assert [p.code for p in got.problems] == ["gear-sum"]
    assert got.observation is not None              # the best guess is still shown for correction


def test_a_misread_sticker_is_repaired_from_the_piece_inventory():
    rng = np.random.default_rng(9)
    repaired = 0
    for _ in range(30):
        obs = L.random_state(rng, 40)
        want = decode_exact(obs)
        cell, gc, gq = probabilities(obs)
        i = int(rng.integers(0, N_CELLS))
        wrong = (obs.cells[i] + int(rng.integers(1, 6))) % 6
        cell[i] = 0.01
        cell[i, wrong] = 0.60
        cell[i, obs.cells[i]] = 0.36
        got = decode_probs(cell, gc, gq)
        assert got.legal
        if got.state144 == want.state144:
            repaired += 1
            assert [(c.kind, c.index) for c in got.corrections] == [("cell", i)]
    assert repaired >= 28                           # a single weak misread is almost always recoverable


@pytest.mark.parametrize("seed", range(3))
def test_noisy_probabilities_still_give_a_legal_state(seed):
    rng = np.random.default_rng(100 + seed)
    obs = L.random_state(rng, 40)
    cell, gc, gq = probabilities(obs, sure=0.8)
    cell = rng.dirichlet(np.ones(6) * 0.3, size=N_CELLS) * 0.15 + cell * 0.85
    got = decode_probs(cell, gc, gq)
    assert got.observation is not None
    assert got.legal == (got.state144 is not None)
