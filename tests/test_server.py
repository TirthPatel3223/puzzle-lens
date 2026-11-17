"""The local server's API, as the browser app uses it."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scanner.paths import DATA
from scanner.puzzles.maltese.decode import decode_exact
from scanner.puzzles.maltese.layout import Observation, load_layout
from scanner.puzzles.maltese.api import solver_frame_faces
from scanner.server import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["puzzle"] == "maltese-gear-cube"
    assert set(h["solver"]) == {"state", "detail"}


def test_decode_solved_and_scrambled(client):
    layout = load_layout()
    r = client.post("/api/decode", json=layout.solved.to_json()).json()
    assert r["legal"] and r["state144"] is not None and r["problems"] == []
    obs = layout.random_state(np.random.default_rng(3), 50)
    r = client.post("/api/decode", json=obs.to_json()).json()
    assert r["legal"]
    assert r["state144"] == decode_exact(obs, layout).state144
    # The same cube from its 144 array.
    r2 = client.post("/api/from-array", json={"state144": r["state144"]}).json()
    assert r2["legal"] and r2["state144"] == r["state144"]


def test_decode_reports_impossible_cubes_in_words(client):
    layout = load_layout()
    obs = layout.solved.copy()
    cells = obs.cells.copy()
    cells[0] = (cells[0] + 2) % 6                 # one star repainted: that corner has no such colours
    r = client.post("/api/decode", json=Observation(cells, obs.gear_colour, obs.gear_quarter).to_json()).json()
    assert not r["legal"] and r["state144"] is None
    assert r["problems"] and all(p["message"] for p in r["problems"])
    assert any(0 in p["cells"] for p in r["problems"])
    gears = obs.gear_colour.copy()
    gears[0] = (gears[0] + 1) % 6                 # five gears of one colour
    r = client.post("/api/decode", json=Observation(obs.cells, gears, obs.gear_quarter).to_json()).json()
    assert not r["legal"] and any(p["code"] == "gear-count" for p in r["problems"])


def test_bad_input_is_refused(client):
    assert client.post("/api/decode", json={"cells": [9] * 96, "gearColour": [0] * 24, "gearQuarter": [0] * 24}).status_code == 422
    assert client.post("/api/from-array", json={"state144": [0] * 10}).status_code == 422


class _FakeVision:
    """Stands in for scanner.puzzles.maltese.vision.pipeline.Vision: a fixed face in the middle of the picture."""

    dir = Path(".")
    device = "cpu"
    reader = object()

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        self.looks: list[int] = []

    def available(self) -> dict:
        return {"locator": True, "reader": True}

    def load(self) -> "_FakeVision":
        return self

    def locate(self, image):
        h, w = image.shape[:2]
        return np.array([[0.2 * w, 0.1 * h], [0.8 * w, 0.1 * h], [0.8 * w, 0.9 * h], [0.2 * w, 0.9 * h]]), self.confidence

    def read(self, image, quad, looks=5):
        from types import SimpleNamespace

        self.looks.append(looks)
        cells = np.full((16, 6), 0.004)
        cells[np.arange(16), np.arange(16) % 6] = 0.98
        cells[3] = [0.0, 0.0, 0.0, 0.6, 0.4, 0.0]          # one sticker the reader is unsure of
        gear_colour = np.full((4, 6), 0.002)
        gear_colour[np.arange(4), [1, 2, 3, 4]] = 0.99
        gear_quarter = np.full((4, 4), 0.003)
        gear_quarter[np.arange(4), [0, 1, 2, 3]] = 0.99
        return SimpleNamespace(cells=cells, gear_colour=gear_colour, gear_quarter=gear_quarter)


def _png(w: int = 160, h: int = 120) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


def test_live_check_reads_a_face_quickly_and_keeps_nothing():
    from scanner.puzzles.maltese.vision.geometry import turn_face

    vision = _FakeVision(0.71)
    c = TestClient(create_app(vision=vision))
    pending = DATA / "real" / "pending"
    before = sorted(pending.iterdir()) if pending.is_dir() else []
    r = c.post("/api/vision/check", files={"photo": ("frame.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["width"], j["height"]) == (160, 120)
    assert j["quad"][0] == [0.2, 0.1] and j["confidence"] == 0.71
    assert vision.looks == [1]                          # one look, not a scan's five
    reading = j["reading"]
    assert reading["unsure"] == 1 and reading["minProb"] == 0.6
    labels = np.arange(16) % 6
    labels[3] = 3
    layout = load_layout()
    assert len(reading["cellTurns"]) == 4
    assert reading["cellTurns"][0] == labels.tolist()
    for k in range(4):
        assert reading["cellTurns"][k] == turn_face(labels, np.zeros(4, int), np.zeros(4, int), k, layout)[0].tolist()
    assert (sorted(pending.iterdir()) if pending.is_dir() else []) == before


def test_live_check_does_not_read_when_there_is_no_face():
    vision = _FakeVision(0.002)
    c = TestClient(create_app(vision=vision))
    j = c.post("/api/vision/check", files={"photo": ("frame.png", _png(), "image/png")}).json()
    assert j["reading"] is None and vision.looks == []


def test_solver_frame_faces_is_a_permutation():
    layout = load_layout()
    for r in range(24):
        faces = solver_frame_faces(layout, r)
        assert sorted(faces) == list(range(6))
    assert solver_frame_faces(layout, 0) == list(range(6))


SCANS = Path(os.environ.get("CUBE_SCANNER_SYNTH_SCANS", "C:/Users/tirth/cube_scanner_data/synth/v1/scans"))


@pytest.mark.skipif(not (SCANS / "meta.jsonl").exists() or not (DATA / "checkpoints" / "reader.pt").exists(),
                    reason="needs the rendered test scans and trained checkpoints")
def test_scan_endpoint_on_a_rendered_scan(client):
    every = [json.loads(line) for line in (SCANS / "meta.jsonl").read_text().splitlines()]
    first = min(r["scan"] for r in every)
    rows = sorted((r for r in every if r["scan"] == first), key=lambda r: r["seed"])     # shots 0-5 of one scan
    assert len(rows) == 6
    order = [3, 0, 5, 1, 4, 2]                      # shuffled: the server works out which face is which
    files = [("photos", (f"shot{i}.jpg", (SCANS / rows[i]["file"]).read_bytes(), "image/jpeg")) for i in order]
    r = client.post("/api/vision/scan", files=files)
    assert r.status_code == 200, r.text
    j = r.json()
    import shutil

    shutil.rmtree(DATA / "real" / "pending" / j["id"], ignore_errors=True)    # the server keeps scans until confirmed
    truth = decode_exact(Observation(np.array(rows[0]["cells"]), np.array(rows[0]["gearColour"]), np.array(rows[0]["gearQuarter"])))
    assert j["decode"]["legal"]
    assert j["decode"]["state144"] == truth.state144
    assert sorted(f["face"] for f in j["faces"]) == sorted("UDFBLR")
    assert {f["holdPosition"] for f in j["faces"]} == {"top", "front", "left", "right", "back", "bottom"}
    assert len(j["reading"]["cellProb"]) == 96 and len(j["decode"]["cellConfidence"]) == 96
    assert not j["arrangement"]["doubtful"]
    assert j["colours"]["usual"] and j["colours"]["paint"] == [0, 1, 2, 3, 4, 5]       # stickered the usual way
