"""The Tangram Cube Extreme's solver (copied from Cube_Solving_Pipeline) against this repository's model.

The first tests need only the copied solver code (they compile its puzzle, 0.1 s). The last one loads the
trained ladder (data/ladders/tangram_extreme/learned_r7) and solves; it is marked ``solver``.
"""
from __future__ import annotations

import numpy as np
import pytest

from scanner.paths import SOLVER_DIR, use_solver
from scanner.puzzles.tangram.layout import load_layout
from scanner.puzzles.tangram.solver import TangramSolverService, check_agreement, match_positions, solver_state

pytestmark = pytest.mark.skipif(not (SOLVER_DIR / "puzzles" / "tangram").is_dir(), reason="the Tangram solver's code is not copied in")


@pytest.fixture(scope="module")
def compiled():
    use_solver()
    from puzzles.tangram.puzzle import compiled as compile_tangram

    return compile_tangram("extreme")


def test_positions_match_one_to_one_and_moves_agree(compiled):
    lay = load_layout()
    to_theirs = match_positions(lay, compiled.frames)
    assert sorted(to_theirs.tolist()) == list(range(120))
    check_agreement(lay, compiled.puzzle, to_theirs, compiled.empty)        # raises if anything differs
    # Scrambles too: the same moves give the same state on both sides.
    rng = np.random.default_rng(3)
    names = compiled.puzzle.move_names
    word = [int(k) for k in rng.integers(0, len(names), size=150)]
    theirs = np.asarray(compiled.puzzle.apply_word(compiled.puzzle.goal, word))
    ours = lay.observe(lay.apply(lay.solved_orient, " ".join(names[k] for k in word)))
    assert np.array_equal(theirs[to_theirs], ours)


def test_a_mismatched_solver_is_refused(compiled):
    from scanner.core.services import SolverNotReady

    lay = load_layout()
    to_theirs = match_positions(lay, compiled.frames)
    swapped = to_theirs.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    with pytest.raises(SolverNotReady):
        check_agreement(lay, compiled.puzzle, swapped, compiled.empty)


@pytest.mark.solver
def test_solves_scrambles_and_every_solution_checks_out():
    service = TangramSolverService()
    status = service.load()
    if status.state == "missing":
        pytest.skip(status.detail)
    assert status.state == "ready", status.detail
    lay = load_layout()
    rng = np.random.default_rng(11)
    for _ in range(3):
        orient = lay.random_state(rng, 200)
        s = service.solve(orient)
        assert s.verified and 40 < s.steps < 200
        assert np.array_equal(lay.apply(orient, s.word), lay.solved_orient)
        assert set(s.detail["perRung"]) == {"frame", "large", "small"}
    assert solver_state(lay, lay.solved_orient).tolist() == lay.solved.tolist()
