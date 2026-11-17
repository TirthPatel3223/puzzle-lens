"""The Tangram Cube Extreme's API, as the browser app uses it (no trained networks needed)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_tangram_scan import reading

from scanner.core.photos import PendingScans
from scanner.puzzles.tangram.api import TangramService
from scanner.puzzles.tangram.layout import EMPTY, load_layout
from scanner.puzzles.tangram.vision.geometry import face_geometry
from scanner.puzzles.tangram.vision.pipeline import TangramVision
from scanner.server import create_app

BASE = "/api/tangram-cube-extreme"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_listed_and_healthy(client):
    ids = [p["id"] for p in client.get("/api/puzzles").json()["puzzles"]]
    assert "tangram-cube-extreme" in ids and "maltese-gear-cube" in ids
    h = client.get(f"{BASE}/health").json()
    assert h["ok"] and h["puzzle"] == "tangram-cube-extreme"
    assert "tangram-cube-extreme" in client.get("/api/health").json()["puzzles"]


def test_decode_and_from_moves(client):
    lay = load_layout()
    r = client.post(f"{BASE}/decode", json={"observation": lay.solved.tolist()}).json()
    assert r["legal"] and r["orient"] == lay.solved_orient.tolist() and r["problems"] == []
    m = client.post(f"{BASE}/from-moves", json={"moves": "U F D2' R U3 L"}).json()
    assert m["moves"] == 6
    r = client.post(f"{BASE}/decode", json={"observation": m["observation"]}).json()
    assert r["legal"] and r["orient"] == m["orient"]
    bad = lay.solved.tolist()
    p = next(i for i, c in enumerate(bad) if c != EMPTY)
    bad[p] = (bad[p] + 1) % 6
    r = client.post(f"{BASE}/decode", json={"observation": bad}).json()
    assert not r["legal"] and r["problems"] and all(q["message"] for q in r["problems"])


def test_bad_input_is_refused(client):
    assert client.post(f"{BASE}/decode", json={"observation": [9] * 120}).status_code == 422
    assert client.post(f"{BASE}/decode", json={"observation": [0] * 10}).status_code == 422
    assert client.post(f"{BASE}/from-moves", json={"moves": "U X"}).status_code == 422


def test_solve_says_plainly_when_the_solver_is_missing(tmp_path):
    from scanner.puzzles.tangram.solver import TangramSolverService

    service = TangramService(solver=TangramSolverService(cache_dir=tmp_path))
    c = TestClient(create_app(services={"tangram-cube-extreme": service}))
    lay = load_layout()
    r = c.post(f"{BASE}/solve", json={"orient": lay.apply(lay.solved_orient, "U F").tolist()})
    assert r.status_code == 503
    assert "ladder" in r.json()["detail"]
    assert c.get(f"{BASE}/health").json()["solver"]["state"] == "missing"


class _UndoSolver:
    """Stands in for the trained solver: it knows the scramble and answers with its undo (or a wrong word)."""

    def __init__(self, word: str) -> None:
        from scanner.puzzles.tangram.solver import TangramSolverService

        self.real = TangramSolverService()
        self.word = word
        self.real._load_backend = lambda: object()
        self.real._backend_solve = lambda backend, state, beam: (self.word, {"perRung": {"frame": 1}})

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_solve_returns_a_solution_checked_on_this_model():
    lay = load_layout()
    state = lay.apply(lay.solved_orient, "U2 F D' R")
    c = TestClient(create_app(services={"tangram-cube-extreme": TangramService(solver=_UndoSolver("R D F U2'"))}))
    r = c.post(f"{BASE}/solve", json={"orient": state.tolist()})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["verified"] and j["word"] == "R D F U2'" and j["turns"] == 4
    assert [m["token"] for m in j["moves"]] == ["R", "D", "F", "U2'"]
    # A word that does not solve the state is refused, whatever the solver says.
    c2 = TestClient(create_app(services={"tangram-cube-extreme": TangramService(solver=_UndoSolver("R D F"))}))
    r2 = c2.post(f"{BASE}/solve", json={"orient": state.tolist()})
    assert r2.status_code == 500 and "does not solve" in r2.json()["detail"]


def test_solution_check_accepts_only_real_solutions():
    from scanner.puzzles.tangram.solver import solution_from_word

    lay = load_layout()
    state = lay.apply(lay.solved_orient, "U2 F D' R")
    s = solution_from_word(lay, state, "R D F U2'", 1.0)
    assert s.verified and s.turns == 4 and s.steps == 5
    with pytest.raises(AssertionError):
        solution_from_word(lay, state, "R D F", 1.0)


class _FakeVision(TangramVision):
    """The real pipeline with stand-ins for the two networks: photo i (a flat grey picture of value i)
    shows face order[i] of ``orient``, read with a few misreads."""

    def __init__(self, orient: np.ndarray, order: list[int], paint: tuple[int, ...] = tuple(range(6))) -> None:
        super().__init__(device="cpu")
        self.reader = object()
        self.orient, self.order, self.paint = orient, order, paint
        self.rng = np.random.default_rng(4)

    def available(self) -> dict:
        return {"locator": True, "reader": True}

    def load(self) -> "_FakeVision":
        return self

    def locate(self, image, passes=2):
        h, w = image.shape[:2]
        return np.array([[0.25 * w, 0.2 * h], [0.75 * w, 0.2 * h], [0.75 * w, 0.8 * h], [0.25 * w, 0.8 * h]]), 0.8

    def read(self, image, quad, looks=3, jitter=0.006):
        f = self.order[int(image[0, 0, 0]) // 10]
        obs = self.layout.observe(self.orient)
        r = reading(obs[face_geometry(self.layout).position_of[f]], self.rng, wrong=0.01)
        out = r.copy()
        out[:, list(self.paint)] = r[:, :6]          # the cube shows model colour c as paint[c]
        return out


def _png(value: int) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((90, 120, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def test_scan_works_out_the_faces_and_decodes_the_state(tmp_path):
    lay = load_layout()
    orient = lay.random_state(np.random.default_rng(9), 100)
    order = [3, 0, 5, 1, 4, 2]
    service = TangramService(vision=_FakeVision(orient, order))
    service.scans = PendingScans(tmp_path)
    c = TestClient(create_app(services={"tangram-cube-extreme": service}))
    files = [("photos", (f"face{i}.png", _png(10 * i), "image/png")) for i in range(6)]
    r = c.post(f"{BASE}/vision/scan", files=files)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["decode"]["legal"] and j["decode"]["orient"] == orient.tolist()
    assert [p["faceIndex"] for p in j["photos"]] == order
    assert all(len(p["positions"]) == 29 and p["flat"].startswith("data:image/jpeg") for p in j["photos"])
    # Keep it as labelled data.
    k = c.post(f"{BASE}/scans/{j['id']}/confirm", json={"observation": j["decode"]["observation"], "source": "checked"}).json()
    assert k["scans"] == 1 and k["decodedRight"] is True
    assert (tmp_path / "scans" / j["id"] / "shot5.jpg").exists()


def test_scan_of_a_cube_stickered_another_way(tmp_path):
    """A DaYan cube (turning corners red-yellow-blue and orange-white-green): the layout is found, the state
    is the model's, and the person can switch back to the usual colours."""
    lay = load_layout()
    orient = lay.random_state(np.random.default_rng(12), 100)
    order = [2, 4, 0, 5, 3, 1]
    dayan = (1, 0, 2, 3, 5, 4)
    service = TangramService(vision=_FakeVision(orient, order, dayan))
    service.scans = PendingScans(tmp_path)
    c = TestClient(create_app(services={"tangram-cube-extreme": service}))
    files = [("photos", (f"face{i}.png", _png(10 * i), "image/png")) for i in range(6)]
    j = c.post(f"{BASE}/vision/scan", files=files).json()
    assert j["colours"]["paint"] == list(dayan) and not j["colours"]["usual"] and j["colours"]["kind"] == "rearranged"
    assert {"model": "orange", "cube": "red"} in j["colours"]["changed"]
    assert j["decode"]["legal"] and j["decode"]["orient"] == orient.tolist()
    assert j["alternatives"][0]["usual"]
    # What was read is reported in the model's colours, like the decoded state.
    assert sum(a != b for a, b in zip(j["read"], j["decode"]["observation"])) <= 3
    # The person says: no, the usual colours. The scan is decoded that way (badly, but as asked).
    k = c.post(f"{BASE}/vision/scan", files=files, data={"colours": "[0,1,2,3,4,5]"}).json()
    assert k["colours"]["usual"]
    assert any(alt["paint"] == list(dayan) for alt in k["alternatives"])     # and the way back is offered
    assert c.post(f"{BASE}/vision/scan", files=files, data={"colours": "[0,0,1,2,3,4]"}).status_code == 422
    # Kept as labelled data: the cube's own colours are stored for the readers.
    r = c.post(f"{BASE}/scans/{j['id']}/confirm", json={"observation": j["decode"]["observation"], "colours": list(dayan)}).json()
    assert r["decodedRight"] is True
    import json as _json
    meta = _json.loads((tmp_path / "scans" / j["id"] / "scan.json").read_text())
    assert meta["labels"]["colours"] == list(dayan)
    seen = np.array(meta["labels"]["seen"]); obs = np.array(j["decode"]["observation"])
    assert (seen[obs == 0] == 1).all() and (seen[obs == 6] == 6).all()
