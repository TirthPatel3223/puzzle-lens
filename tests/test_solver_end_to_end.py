"""The whole loop without vision: what a scrambled cube shows -> decode -> solver -> replay.

The replay is done on the *engine's* move tables (through the dictionary), not on the solver's
own simulator, so it checks that a solution the solver believes in also solves the cube the 3D
model (and the real puzzle) describes.

Needs the built ladder (python tools/import_ladder.py) and takes about a minute.
"""
from __future__ import annotations

import numpy as np
import pytest

from scanner.puzzles.maltese.decode import decode_exact
from scanner.puzzles.maltese.layout import load_layout
from scanner.puzzles.maltese.solver import NotSolvable, SolverService

pytestmark = pytest.mark.solver

L = load_layout()


@pytest.fixture(scope="module")
def solver() -> SolverService:
    service = SolverService()
    status = service.load()
    if status.state != "ready":
        pytest.skip(status.detail)
    return service


def replay(obs, solution):
    for move in solution.moves:
        face, direction = L.solver.moves[move["name"]]
        turns = direction * move["amount"]                    # signed quarter turns of the engine face
        step = L.moves[face if turns > 0 else face + "'"]
        for _ in range(abs(turns)):
            obs = step.apply(obs)
    return obs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_scramble_held_any_way_is_solved_on_the_engine(solver, seed):
    rng = np.random.default_rng(seed)
    scrambled = L.random_state(rng, 60)                       # all six faces
    scrambled = L.rotations[int(rng.integers(0, 24))].apply(scrambled)   # held some other way
    decoded = decode_exact(scrambled)
    assert decoded.legal
    solution = solver.solve(decoded.state144)
    assert solution.verified
    assert sum(abs(m["amount"]) for m in solution.moves) == solution.quarter_turns
    assert replay(decoded.observation, solution) == L.solved


def test_the_solved_cube_needs_no_moves(solver):
    solution = solver.solve(L.to_solver(L.solved))
    assert solution.moves == [] and solution.quarter_turns == 0


def test_an_unreachable_array_is_refused(solver):
    state = L.to_solver(L.solved).copy()
    state[120] = (state[120] + 1) % 4                         # one gear turned alone
    with pytest.raises(NotSolvable):
        solver.solve(state)
