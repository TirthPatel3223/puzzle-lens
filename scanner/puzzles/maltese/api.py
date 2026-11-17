"""The Maltese Gear Cube's endpoints (mounted at /api/maltese-gear-cube, and at /api for older pages).

    GET  /health              what is loaded
    POST /solver/load         start loading the solver in the background
    POST /vision/load         start loading the vision networks in the background
    POST /decode              {cells, gearColour, gearQuarter} -> legal arrangement in the solver's frame, or what is wrong
    POST /from-array          {state144} -> the same, starting from the solver's 144-entry array
    POST /solve               {state144} -> a verified solution in the solver's move names (and as 3D-model moves)
    POST /vision/locate       a photo -> the face's four corners in it (+ a small preview with ?preview=true)
    POST /vision/check        a live webcam frame -> its face's corners and a quick reading (nothing is kept)
    POST /vision/scan         six photos, any order and any way up -> what they show, decoded to a legal cube,
                              and the cube's colour layout (form field ``colours`` forces one)
    POST /scans/{id}/confirm  the checked result of a scan: kept with its photos as labelled real data

Colours are always the model's, in every request and answer (white yellow green blue red purple). A scan
also says the cube's colour layout (``colours.paint``: the colour the cube shows where the model shows
colour c; scanner/core/colours.py), and the page paints the model's colour c as ``paint[c]``.

Frames. A scan's answer is given in the frame the cube was photographed in: the first photo is
its F face, with that photo's top edge up. ``decode`` holds the same cube turned to the solver's
frame (the white-green-red corner home), which is what the solver and the 144 array use.
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
from scanner.core.services import LazyVision
from scanner.paths import checkpoints_dir, real_dir

from .decode import decode_exact
from .layout import (
    FACE_NAMES,
    N_CELLS,
    N_GEARS,
    STATE_DIM,
    Layout,
    Observation,
    load_layout,
)
from .solver import NotSolvable, SolverService

ID = "maltese-gear-cube"
KEY = "maltese"
TITLE = "Maltese Gear Cube"

#: A live check (/vision/check) reads the face only when the locator is at least this sure there is
#: one (a real face scores about 0.7, an empty picture about 0.002; the app calls under 0.45 "unsure").
CHECK_MIN_CONFIDENCE = 0.3
#: A slot the reader gives less than this probability counts as unsure in a live check.
CHECK_SURE = 0.9
#: An arrangement of the photos less than this many nats more likely than the next is reported as doubtful
#: (the same value as vision.pipeline.ARRANGEMENT_DOUBT_NATS; not imported, to keep torch out of start-up).
ARRANGEMENT_DOUBT_NATS = 10.0

#: Where each face of the solver's frame is while following a solution (blue up, yellow in front,
#: purple on the left), by engine face. The engine's faces are named by their solved colours:
#: U white, D yellow, F green, B blue, L red, R purple.
HOLD_POSITION = {"B": "top", "D": "front", "R": "left", "L": "right", "U": "back", "F": "bottom"}


class ObservationIn(BaseModel):
    cells: list[int] = Field(min_length=N_CELLS, max_length=N_CELLS)
    gearColour: list[int] = Field(min_length=N_GEARS, max_length=N_GEARS)
    gearQuarter: list[int] = Field(min_length=N_GEARS, max_length=N_GEARS)

    @field_validator("cells", "gearColour")
    @classmethod
    def _colours(cls, v: list[int]) -> list[int]:
        if any(c < 0 or c > 5 for c in v):
            raise ValueError("colours are 0-5 (white, yellow, green, blue, red, purple)")
        return v

    @field_validator("gearQuarter")
    @classmethod
    def _quarters(cls, v: list[int]) -> list[int]:
        if any(q < 0 or q > 3 for q in v):
            raise ValueError("gear quarter turns are 0-3")
        return v

    def to_observation(self) -> Observation:
        return Observation(np.array(self.cells), np.array(self.gearColour), np.array(self.gearQuarter))


class ConfirmIn(ObservationIn):
    #: How the labels were obtained: "checked" (a person looked at the 3D model) or "known-scramble"
    #: (the cube was scrambled with moves the app gave, so the state is known exactly).
    source: str = Field(default="checked", pattern="^(checked|known-scramble)$")
    #: The cube's colour layout the person checked against (default: the scan's).
    colours: list[int] | None = Field(default=None, min_length=6, max_length=6)


def colour_layout(paint: list[int] | None) -> ColourLayout:
    """A colour layout from the page (``paint``), as the class representative the scanner uses."""
    lay = load_layout()
    try:
        return representative(lay.colour_layouts, paint, lay.colour_actions)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err


class ArrayIn(BaseModel):
    state144: list[int] = Field(min_length=STATE_DIM, max_length=STATE_DIM)


class SolveIn(ArrayIn):
    beam: int | None = Field(default=None, ge=1, le=65536)


def solver_frame_faces(layout: Layout, rotation: int) -> list[int]:
    """For each face of the frame a scan was photographed in, the face it is in the solver's frame
    (both as indices into FACE_NAMES). ``rotation`` takes the photographed frame to the solver's."""
    rot = layout.rotations[rotation]
    held_of = [int(rot.cells[16 * g + 5]) // 16 for g in range(6)]        # solver face g shows held face held_of[g]
    out = [0] * 6
    for g, h in enumerate(held_of):
        out[h] = g
    return out


def _make_vision():
    from .vision.pipeline import Vision

    return Vision()


class MalteseService:
    id = ID
    key = KEY
    title = TITLE

    def __init__(self, solver: SolverService | None = None, vision=None) -> None:
        self.layout = load_layout()
        self.solver = solver or SolverService()
        self.vision = LazyVision(_make_vision, "python -m scanner.vision.train reader --puzzle maltese --data <dataset>", vision)
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
        layout = self.layout
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
            """Load the networks in the background, so the first scan does not wait for torch."""
            lazy.load_in_background()
            return {"started": True}

        @r.post("/decode")
        def decode(body: ObservationIn) -> dict:
            return decode_exact(body.to_observation(), layout).to_json()

        @r.post("/from-array")
        def from_array(body: ArrayIn) -> dict:
            try:
                obs = layout.from_solver(body.state144)
            except ValueError as err:
                raise HTTPException(422, str(err)) from err
            return decode_exact(obs, layout).to_json()

        @r.post("/solve")
        def solve(body: SolveIn) -> dict:
            try:
                s = solver.solve(body.state144, beam=body.beam).to_json()
            except NotSolvable as err:
                raise HTTPException(409, f"The solver cannot reach this state: {err}") from err
            except RuntimeError as err:                 # the ladder is missing or failed to load
                raise HTTPException(503, str(err)) from err
            # The same moves on the 3D model (whose faces are named by their solved colours: the solver's
            # U, L, F are the model's B, R, D), so the browser can play them back.
            s["engineMoves"] = [{"face": f, "amount": int(a)} for f, a in layout.solver_word_to_engine(s["word"])]
            return s

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
            """A live webcam frame: is there a face the networks can read, and what does it show?

            For the webcam's live outline and auto-capture: the face's corners, and when one was found, a
            quick reading (one look, not the five a scan uses). The cells are given in all four turns, so
            the page can tell a face it has already captured however it is held. Nothing is kept."""
            image = read_photo(photo.file.read())
            v = lazy.get()
            with lazy.lock:
                quad, confidence = v.locate(image)
                reading = v.read(image, quad, looks=1) if confidence >= CHECK_MIN_CONFIDENCE else None
            h, w = image.shape[:2]
            out: dict = {"width": w, "height": h, "quad": fractions(quad, image), "confidence": round(confidence, 4), "reading": None}
            if reading is not None:
                from .vision.geometry import turn_face

                cell_p = reading.cells.max(1)
                gear_p = np.minimum(reading.gear_colour.max(1), reading.gear_quarter.max(1))
                labels = reading.cells.argmax(1)
                gc, gq = reading.gear_colour.argmax(1), reading.gear_quarter.argmax(1)
                out["reading"] = {
                    # Stickers (16) and gears (4) the reader was less than CHECK_SURE sure of.
                    "unsure": int((cell_p < CHECK_SURE).sum() + (gear_p < CHECK_SURE).sum()),
                    "minProb": round(float(min(cell_p.min(), gear_p.min())), 4),
                    # cellTurns[k]: the cells if the face appears turned k quarter turns counterclockwise.
                    "cellTurns": [turn_face(labels, gc, gq, k, layout)[0].tolist() for k in range(4)],
                }
            return out

        @r.post("/vision/scan")
        def scan(photos: list[UploadFile] = File(...), quads: str | None = Form(default=None),
                 colours: str | None = Form(default=None)) -> dict:
            """Six photos, one of each face, in any order and any way up. ``quads`` is optional JSON: per
            photo its four corners as fractions of the picture (top-left first, as the picture shows
            them), or null to have that photo located. ``colours`` is optional JSON: the cube's colour
            layout (six colours) when the person chose one; otherwise it is worked out."""
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
                flats = [v.flatten(image, np.roll(q, k, axis=0)) for image, q, k in zip(images, px, result.turns)]
                # In the model's colours (the cube's colour layout applied), like everything answered.
                cell_p, gc_p, gq_p = v.assemble(result.model_readings(), result.turns, result.faces)
            seconds = time.time() - t0
            d = result.decode
            arrangement = result.arrangement

            # The decoded cube, turned back to the way it was held for the photos.
            held = layout.rotations[int(layout.group_inv[d.rotation])].apply(d.observation) if d.observation is not None else None
            solver_face = solver_frame_faces(layout, d.rotation)

            # Keep the photos until the person has checked the result: a confirmed scan is labelled real data.
            scan_id = scans.new(images, {
                "names": names, "faces": [FACE_NAMES[f] for f in result.faces], "turns": result.turns,
                "quads": [fractions(q, im) for q, im in zip(px, images)],
                "sizes": [[im.shape[1], im.shape[0]] for im in images], "rotation": d.rotation, "legal": d.legal,
                "arrangementMargin": round(arrangement.margin, 3) if arrangement else None,
                "read": Observation(cell_p.argmax(1), gc_p.argmax(1), gq_p.argmax(1)).to_json(),
                "decoded": held.to_json() if held is not None else None,
                "colours": list(result.colours.paint), "coloursKind": result.colours.kind,
            }, {"cells": cell_p, "gear_colour": gc_p, "gear_quarter": gq_p})

            def rounded(a: np.ndarray) -> list[float]:
                return [round(float(x), 4) for x in a]

            return {
                "id": scan_id,
                "seconds": round(seconds, 2),
                "observation": held.to_json() if held is not None else None,
                "decode": d.to_json(),
                "turns": result.turns,
                "colours": page_json(result.colours, layout.colour_names),
                "alternatives": [page_json(c, layout.colour_names, nats) for c, nats in result.alternatives],
                "arrangement": {
                    "margin": round(arrangement.margin, 2) if arrangement else None,
                    "doubtful": bool(arrangement and arrangement.margin < ARRANGEMENT_DOUBT_NATS),
                },
                # What each place was read as before decoding, and how sure the reader was, in the frame photographed.
                "reading": {
                    "cells": cell_p.argmax(1).tolist(), "cellProb": rounded(cell_p.max(1)),
                    "gearColour": gc_p.argmax(1).tolist(), "gearColourProb": rounded(gc_p.max(1)),
                    "gearQuarter": gq_p.argmax(1).tolist(), "gearQuarterProb": rounded(gq_p.max(1)),
                },
                "faces": [
                    {
                        "shot": i,
                        "name": names[i],
                        "face": FACE_NAMES[f],
                        "solvedColour": layout.colour_names[solver_face[f]],
                        "holdPosition": HOLD_POSITION[FACE_NAMES[solver_face[f]]],
                        "quad": fractions(px[i], images[i]),
                        "locateConfidence": round(float(confidence[i]), 4),
                        "turn": result.turns[i],
                        # The face flattened and turned the way its labels are laid out (rows down, columns right).
                        "flat": jpeg_data_url(flats[i]),
                    }
                    for i, f in enumerate(result.faces)
                ],
            }

        @r.post("/scans/{scan_id}/confirm")
        def confirm(scan_id: str, body: ConfirmIn) -> dict:
            """The cube a scan really showed, in the frame it was photographed in, as checked by the person
            (or known from the scramble). The photos and these labels are kept as labelled real data for
            testing and fine-tuning the networks."""
            meta = scans.read_meta(scan_id)
            held = body.to_observation()
            checked = decode_exact(held, layout)
            if not checked.legal:
                raise HTTPException(409, "Only a legal cube can be kept as a labelled scan: " + "; ".join(p.message for p in checked.problems))
            cl = colour_layout(body.colours if body.colours is not None else meta.get("colours"))
            meta["labels"] = {"source": body.source, "solverFrame": checked.observation.to_json(), "scanFrame": held.to_json(),
                              "state144": checked.state144,
                              # the cube's colour layout, and what the cube itself shows (the readers' labels)
                              "colours": list(cl.paint),
                              "seen": {"cells": cl.to_cube(held.cells).tolist(), "gearColour": cl.to_cube(held.gear_colour).tolist(),
                                       "gearQuarter": np.asarray(held.gear_quarter).tolist()}}
            if meta.get("read"):
                rd = Observation.from_json(meta["read"])
                meta["readErrors"] = {
                    "cells": int((rd.cells != held.cells).sum()), "gearColour": int((rd.gear_colour != held.gear_colour).sum()),
                    "gearQuarter": int((rd.gear_quarter != held.gear_quarter).sum()),
                }
            if meta.get("decoded"):
                meta["decodedRight"] = bool(Observation.from_json(meta["decoded"]) == held)
            kept = scans.confirm(scan_id, meta)
            return {"saved": str(kept), "scans": scans.labelled_count(),
                    "readErrors": meta.get("readErrors"), "decodedRight": meta.get("decodedRight")}

        return r


_preview = preview


def create_service(**kwargs) -> MalteseService:
    return MalteseService(**kwargs)
