"""The Tangram Cube Extreme's endpoints (mounted at /api/tangram-cube-extreme).

    GET  /health              what is loaded
    POST /solver/load         start loading the solver in the background
    POST /vision/load         start loading the vision networks in the background
    POST /decode              {observation: 120 values} -> the legal state it is, or what is wrong
    POST /from-moves          {moves: "U2' F D R"} -> the state those moves make from solved (practice scrambles)
    POST /solve               {orient: 45 values} -> a solution in the simulator's notation, checked on the model
    POST /vision/locate       a photo -> the face's corners (turning corner third) (+ a preview with ?preview=true)
    POST /vision/check        a live webcam frame -> the face's corners and a quick reading (nothing is kept)
    POST /vision/scan         six face-on photos, any order and any way up -> the decoded state and the
                              cube's colour layout (form field ``colours`` forces one)
    POST /scans/{id}/confirm  the checked state of a scan: kept with its photos as labelled real data

States are in the core's frame (the hidden middle piece never moves): one orientation per piece, indices
into the 12 rotations in shared/tangram/layout.json. Observations are one value per position: colour
0-5 (orange red yellow white green blue) or 6 for no sticker there.

**Colours are always the model's**, in every request and answer. A scan also says the cube's colour
layout (``colours.paint``: the colour the cube shows where the model shows colour c; see
scanner/core/colours.py), and the page paints the model's colour c as ``paint[c]``. So the solver, the
decoder and every state stay the same for a cube stickered another way.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from scanner.core.colours import ColourLayout, page_json, representative
from scanner.core.photos import (
    PendingScans,
    fractions,
    jpeg_data_url,
    preview,
    read_photo,
)
from scanner.core.services import LazyVision, NotSolvable, SolverNotReady
from scanner.paths import checkpoints_dir, real_dir

from .decode import decode_exact
from .layout import EMPTY, load_layout, parse_word
from .solver import TangramSolverService

ID = "tangram-cube-extreme"
KEY = "tangram"
TITLE = "Tangram Cube Extreme"
N_POSITIONS = 120
N_PIECES = 45

#: A live check reads the face only when the locator is at least this sure there is one.
CHECK_MIN_CONFIDENCE = 0.3
#: A position the reader gives less than this probability counts as unsure in a live check.
CHECK_SURE = 0.85
#: Side of the flattened face sent back for the Check step.
FLAT = 448
FLAT_HALF = 1.5


class ObservationIn(BaseModel):
    observation: list[int] = Field(min_length=N_POSITIONS, max_length=N_POSITIONS)

    @field_validator("observation")
    @classmethod
    def _values(cls, v: list[int]) -> list[int]:
        if any(c < 0 or c > EMPTY for c in v):
            raise ValueError("each position is a colour 0-5 (orange, red, yellow, white, green, blue) or 6 for no sticker")
        return v


class ConfirmIn(ObservationIn):
    source: str = Field(default="checked", pattern="^(checked|known-scramble)$")
    #: The cube's colour layout the person checked against (default: the model's own).
    colours: list[int] | None = Field(default=None, min_length=6, max_length=6)


class MovesIn(BaseModel):
    moves: str = Field(max_length=4000)


class SolveIn(BaseModel):
    orient: list[int] = Field(min_length=N_PIECES, max_length=N_PIECES)
    beam: int | None = Field(default=None, ge=1, le=65536)


def colour_layout(paint: list[int] | None) -> ColourLayout:
    """A colour layout from the page (``paint``), as the class representative the scanner uses."""
    lay = load_layout()
    try:
        return representative(lay.colour_layouts, paint, lay.colour_actions)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err


colours_json = page_json


def _make_vision():
    from .vision.pipeline import TangramVision

    return TangramVision()


class TangramService:
    id = ID
    key = KEY
    title = TITLE

    def __init__(self, solver: TangramSolverService | None = None, vision=None) -> None:
        self.layout = load_layout()
        self.solver = solver or TangramSolverService()
        self.vision = LazyVision(_make_vision, "python -m scanner.vision.train reader --puzzle tangram --data <dataset>", vision)
        self.scans = PendingScans(real_dir(KEY))

    def describe(self) -> dict:
        return {"id": ID, "key": KEY, "title": TITLE, "photos": 6}

    def health(self) -> dict:
        ck = checkpoints_dir(KEY)
        return {
            "puzzle": ID, "solver": self.solver.status.to_json(),
            "vision": self.vision.status({"locator": (ck / "locator.pt").exists(), "reader": (ck / "reader.pt").exists()}),
            "labelledScans": self.scans.labelled_count(),
        }

    def router(self) -> APIRouter:
        r = APIRouter()
        lay = self.layout
        solver = self.solver
        lazy = self.vision
        scans = self.scans

        @r.get("/health")
        def health() -> dict:
            return {"ok": True, **self.health()}

        @r.post("/solver/load")
        def solver_load() -> dict:
            if solver.status.state == "not-loaded":
                solver.status.state = "loading"
                threading.Thread(target=solver.load, daemon=True).start()
            return solver.status.to_json()

        @r.post("/vision/load")
        def vision_load() -> dict:
            lazy.load_in_background()
            return {"started": True}

        @r.post("/decode")
        def decode(body: ObservationIn) -> dict:
            return decode_exact(np.array(body.observation), lay).to_json()

        @r.post("/from-moves")
        def from_moves(body: MovesIn) -> dict:
            try:
                orient = lay.apply(lay.solved_orient, body.moves)
            except ValueError as err:
                raise HTTPException(422, str(err)) from err
            return {"orient": orient.tolist(), "observation": lay.observe(orient).tolist(), "moves": len(parse_word(body.moves))}

        @r.post("/solve")
        def solve(body: SolveIn) -> dict:
            orient = np.array(body.orient)
            if any(int(o) not in lay.placements[i] for i, o in enumerate(orient)):
                raise HTTPException(422, "Some piece is at an orientation it can never have.")
            try:
                return solver.solve(orient, beam=body.beam).to_json()
            except NotSolvable as err:
                raise HTTPException(409, f"The solver cannot reach this state: {err}") from err
            except SolverNotReady as err:
                raise HTTPException(503, str(err)) from err
            except AssertionError as err:            # the answer failed a replay: never shown as a solution
                raise HTTPException(500, f"The solver's answer was refused: {err}") from err

        # -- vision -----------------------------------------------------------------------------

        @r.post("/vision/locate")
        def locate(photo: UploadFile = File(...), preview: bool = False) -> dict:   # noqa: F811 (the query flag)
            image = read_photo(photo.file.read())
            v = lazy.get()
            with lazy.lock:
                quad, confidence = v.locate(image)
            h, w = image.shape[:2]
            out = {"width": w, "height": h, "quad": fractions(quad, image), "confidence": round(confidence, 4)}
            if preview:
                out["preview"] = _preview(image)
            return out

        @r.post("/vision/check")
        def check(photo: UploadFile = File(...)) -> dict:
            """A live webcam frame: is there a face the networks can read? One look, nothing kept. ``labels``
            (what each of the face's 29 positions shows) tells the page a face it has already taken."""
            image = read_photo(photo.file.read())
            v = lazy.get()
            with lazy.lock:
                quad, confidence = v.locate(image)
                probs = v.read(image, quad, looks=1) if confidence >= CHECK_MIN_CONFIDENCE else None
            h, w = image.shape[:2]
            out: dict = {"width": w, "height": h, "quad": fractions(quad, image), "confidence": round(confidence, 4), "reading": None}
            if probs is not None:
                p = probs.max(1)
                out["reading"] = {"unsure": int((p < CHECK_SURE).sum()), "minProb": round(float(p.min()), 4), "labels": probs.argmax(1).tolist()}
            return out

        @r.post("/vision/scan")
        def scan(photos: list[UploadFile] = File(...), quads: str | None = Form(default=None),
                 colours: str | None = Form(default=None)) -> dict:
            """Six face-on photos, one of each face, in any order and any way up. ``quads`` is optional JSON:
            per photo its four corners as fractions of the picture in the face's order (turning corner
            third), or null to have that photo located. ``colours`` is optional JSON: the cube's colour
            layout (six colours, ``paint``) when the person chose one; otherwise it is worked out."""
            if len(photos) != 6:
                raise HTTPException(422, f"A scan is six photos, one of each face; got {len(photos)}.")
            images = [read_photo(p.file.read()) for p in photos]
            names = [p.filename or f"photo {i + 1}" for i, p in enumerate(photos)]
            given: list = [None] * 6
            if quads:
                try:
                    given = json.loads(quads)
                    if len(given) != 6:
                        raise ValueError
                except ValueError as err:
                    raise HTTPException(422, "quads must be a JSON list of six entries") from err
            forced = None
            if colours:
                try:
                    forced = colour_layout(json.loads(colours))
                except (ValueError, TypeError) as err:
                    raise HTTPException(422, "colours must be a JSON list of six colours") from err
            v = lazy.get()
            t0 = time.time()
            with lazy.lock:
                px, confidence = [], []
                for image, q in zip(images, given):
                    h, w = image.shape[:2]
                    if q is None:
                        quad, c = v.locate(image)
                    else:
                        quad, c = np.asarray(q, dtype=np.float64).reshape(4, 2) * np.array([w, h]), 1.0
                    px.append(quad)
                    confidence.append(c)
                result = v.scan(images, px, colours=forced)
                flats = [v.flatten(image, q, FLAT, FLAT_HALF) for image, q in zip(images, result.quads)]
            seconds = time.time() - t0
            from .vision.pipeline import assemble

            geom = v.geom
            cl = result.colours
            # What was read, in the model's colours (the cube's colour layout applied), like the decoded state.
            logp = assemble(result.evidence(), result.faces, geom, lay.n_positions)
            read = logp.argmax(1)
            d = result.decode
            scan_id = scans.new(images, {
                "names": names, "faces": [lay.face_ids[f] for f in result.faces],
                "quads": [fractions(q, im) for q, im in zip(result.quads, images)],
                "sizes": [[im.shape[1], im.shape[0]] for im in images], "legal": d.legal, "margin": round(result.margin, 3),
                "read": read.tolist(), "decoded": d.orient.tolist() if d.orient is not None else None,
                "colours": list(cl.paint), "coloursKind": cl.kind,
            }, {"readings": np.stack(result.readings)})
            photos_out = []
            for i, f in enumerate(result.faces):
                flat, pts = flats[i]
                probs = result.readings[i]
                shown = cl.to_model(probs.argmax(1))                  # colours as the model's; empty / hidden kept
                photos_out.append({
                    "shot": i, "name": names[i], "face": lay.face_ids[f], "faceIndex": int(f), "end": lay.face_end[f],
                    "colour": lay.colour_names[f],
                    "quad": fractions(result.quads[i], images[i]), "locateConfidence": round(float(confidence[i]), 4),
                    "turned": int(result.returned.get(i, 0)),
                    "flat": jpeg_data_url(flat),
                    # Where each of the face's positions appears in the flattened photo, and what was read there.
                    "positions": [
                        {"position": int(geom.position_of[f][j]), "x": round(float(pts[j][0] / FLAT), 4), "y": round(float(pts[j][1] / FLAT), 4),
                         "flat": bool(geom.flat[j]), "read": int(shown[j]), "prob": round(float(probs[j].max()), 4)}
                        for j in range(geom.n)
                    ],
                })
            return {
                "id": scan_id, "seconds": round(seconds, 2), "decode": d.to_json(),
                "margin": round(result.margin, 2) if np.isfinite(result.margin) else None, "doubtful": result.doubtful,
                "read": read.tolist(), "readProb": [round(float(x), 4) for x in np.exp(logp.max(1))],
                "colours": colours_json(cl, lay.colour_names),
                "alternatives": [colours_json(c, lay.colour_names, nats) for c, nats in result.alternatives],
                "photos": photos_out,
            }

        @r.post("/scans/{scan_id}/confirm")
        def confirm(scan_id: str, body: ConfirmIn) -> dict:
            meta = scans.read_meta(scan_id)
            obs = np.array(body.observation)
            checked = decode_exact(obs, lay)
            if not checked.legal:
                raise HTTPException(409, "Only a legal state can be kept as a labelled scan: " + "; ".join(p.message for p in checked.problems))
            cl = colour_layout(body.colours if body.colours is not None else meta.get("colours"))
            meta["labels"] = {"source": body.source, "orient": checked.orient.tolist(), "observation": obs.tolist(),
                              # what the cube itself shows (the readers' labels), and its colour layout
                              "colours": list(cl.paint), "seen": cl.to_cube(obs).tolist()}
            if meta.get("read"):
                meta["readErrors"] = int((np.array(meta["read"]) != obs).sum())
            if meta.get("decoded"):
                meta["decodedRight"] = bool(np.array_equal(np.array(meta["decoded"]), checked.orient))
            kept = scans.confirm(scan_id, meta)
            return {"saved": str(kept), "scans": scans.labelled_count(), "readErrors": meta.get("readErrors"), "decodedRight": meta.get("decodedRight")}

        return r


_preview = preview


def create_service(**kwargs) -> TangramService:
    return TangramService(**kwargs)
