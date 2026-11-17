"""Score the trained networks the way they are used: from the picture alone.

    python -m scanner.vision.evaluate faces --puzzle maltese --data C:/Users/tirth/cube_scanner_data/synth/v1/val
    python -m scanner.vision.evaluate scans --puzzle maltese --data C:/Users/tirth/cube_scanner_data/synth/v1/scans [--any-order]
    python -m scanner.vision.evaluate real --puzzle maltese                  # the labelled real scans kept from the app

``faces``  every picture is located and read with no help; reports how often each kind of
           slot is right and how often a whole face is read without a mistake.
``scans``  every six-picture scan goes through the whole pipeline (locate, read, work out
           which way up each photo is, decode to a legal cube) and is compared with the true
           cube; reports how many scans come out exactly right, before and after decoding.
           With --any-order the photos are shuffled first, as the app allows.
``real``   the same for real photos kept from the app's Check step (data/real/scans).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scanner.puzzles.maltese.decode import decode_exact
from scanner.puzzles.maltese.layout import Observation

from .dataset import Photos
from .geometry import picture_order, quad_side
from .pipeline import ARRANGEMENT_DOUBT_NATS, SCAN_ORDER, Vision


def evaluate_faces(vision: Vision, photos: Photos, limit: int | None = None, true_corners: bool = False) -> dict:
    layout = vision.layout
    n = len(photos) if limit is None else min(limit, len(photos))
    hits = defaultdict(list)
    corner_err = []
    by_setting = defaultdict(list)
    confusion = np.zeros((6, 6), dtype=np.int64)
    quarter_confusion = np.zeros((4, 4), dtype=np.int64)
    t0 = time.time()
    for i in range(n):
        img = photos.image(i)
        true_quad, shift = picture_order(photos.quads[i])
        if true_corners:
            quad = true_quad
        else:
            quad, _ = vision.locate(img)
            corner_err.append(float(np.linalg.norm(quad - true_quad, axis=1).max() / quad_side(true_quad)))
        r = vision.read(img, quad).turned(shift, layout)        # the dataset's labels are in the face's own orientation
        cells, gc, gq = photos.face_labels(i)
        pc, pgc, pgq = r.cells.argmax(1), r.gear_colour.argmax(1), r.gear_quarter.argmax(1)
        grid = layout.grid
        star = list(grid["cornerCells"])
        centre = list(grid["centreCells"])
        edge = [c for pair in grid["edgeCells"] for c in pair]
        hits["star"].append(np.mean(pc[star] == cells[star]))
        hits["edge"].append(np.mean(pc[edge] == cells[edge]))
        hits["centre"].append(np.mean(pc[centre] == cells[centre]))
        hits["gear colour"].append(np.mean(pgc == gc))
        hits["gear quarter"].append(np.mean(pgq == gq))
        ok = bool((pc == cells).all() and (pgc == gc).all() and (pgq == gq).all())
        hits["whole face"].append(ok)
        by_setting[str(photos.setting[i])].append(ok)
        np.add.at(confusion, (cells, pc), 1)
        np.add.at(quarter_confusion, (gq, pgq), 1)
    out = {
        "pictures": n,
        "seconds per picture": round((time.time() - t0) / max(1, n), 3),
        "accuracy %": {k: round(100 * float(np.mean(v)), 3) for k, v in hits.items()},
        "whole face % by setting": {k: round(100 * float(np.mean(v)), 2) for k, v in sorted(by_setting.items())},
        "colour confusion (rows true, columns read; " + ", ".join(layout.colour_names) + ")": confusion.tolist(),
        "gear quarter confusion (rows true, columns read)": quarter_confusion.tolist(),
    }
    if corner_err:
        e = np.array(corner_err)
        out["worst corner error, % of the face's side"] = {"median": round(100 * float(np.median(e)), 2), "p95": round(100 * float(np.quantile(e, 0.95)), 2),
                                                           "over 4%": round(100 * float((e > 0.04).mean()), 2)}
    return out


def evaluate_scans(vision: Vision, photos: Photos, limit: int | None = None, any_order: bool = False, shuffle_seed: int = 0) -> dict:
    """With ``any_order`` the six photos are shuffled before the scan and the pipeline has to work out
    which face each shows (as the app does); otherwise they are given in capture order."""
    layout = vision.layout
    scans = defaultdict(list)
    for i in range(len(photos)):
        scans[int(photos.scan[i])].append(i)
    ids = sorted(s for s in scans if s >= 0 and len(scans[s]) == 6)
    if limit is not None:
        ids = ids[:limit]
    rng = np.random.default_rng(shuffle_seed)
    exact, legal, raw_exact, raw_wrong, decoded_wrong, corrected, turn_ok, margins, flagged = [], [], [], [], [], [], [], [], []
    failures = []
    t0 = time.time()
    for s in ids:
        idx = sorted(scans[s], key=lambda i: int(photos.seeds[i]))          # shots 0-5: F R B L U D
        if tuple(int(photos.faces[i]) for i in idx) != SCAN_ORDER:
            raise ValueError(f"scan {s} is not in capture order")
        i0 = idx[0]
        # What the cube shows (its colour layout applied), in the frame it was held in.
        shown = Observation(photos.cells[i0].astype(np.int64), photos.gear_colour[i0].astype(np.int64), photos.gear_quarter[i0].astype(np.int64))
        shifts = [picture_order(photos.quads[i])[1] for i in idx]
        if any_order:
            order = rng.permutation(6)
            result = vision.scan([photos.image(idx[j]) for j in order])
            margins.append(result.arrangement.margin)
            flagged.append(result.arrangement.margin < ARRANGEMENT_DOUBT_NATS)
            wrong = -1                                                       # the held frame differs from the rendered one
        else:
            order = np.arange(6)
            result = vision.scan([photos.image(i) for i in idx], faces=SCAN_ORDER)
            # What the photos were read as before any decoding, in the frame the cube was held in.
            cell_p, gc_p, gq_p = vision.assemble(result.readings, result.turns)        # the colours seen
            wrong = int((cell_p.argmax(1) != shown.cells).sum() + (gc_p.argmax(1) != shown.gear_colour).sum()
                        + (gq_p.argmax(1) != shown.gear_quarter).sum())
            raw_wrong.append(wrong)
            raw_exact.append(wrong == 0)
            # A photo taken a quarter turn off shows its face turned that much; the pipeline should say so.
            turn_ok.append(result.turns == shifts)
        d = result.decode
        legal.append(d.legal)
        # The truth in the model's colours, through the layout the scanner found (a layout that is the true one
        # with the whole cube turned gives the same cube once the decoder turns it to the solver's frame).
        cl = result.colours
        truth = decode_exact(Observation(cl.to_model(shown.cells), cl.to_model(shown.gear_colour), shown.gear_quarter), layout)
        same = truth.legal and d.observation is not None and d.observation == truth.observation
        exact.append(bool(d.legal and same))
        if d.observation is not None and truth.observation is not None:
            o, t = d.observation, truth.observation
            decoded_wrong.append(int((o.cells != t.cells).sum() + (o.gear_colour != t.gear_colour).sum() + (o.gear_quarter != t.gear_quarter).sum()))
        corrected.append(len(d.corrections))
        if not exact[-1]:
            failures.append({"scan": s, "legal": d.legal, "wrong before decoding": wrong, "turns found": result.turns,
                             "turns true": [shifts[j] for j in order], "faces found": list(result.faces),
                             "arrangement margin": round(result.arrangement.margin, 2) if result.arrangement else None,
                             "problems": [p.message for p in d.problems][:3], "setting": str(photos.setting[i0])})
    n = len(ids)
    out = {
        "scans": n,
        "photos in any order": any_order,
        "seconds per scan": round((time.time() - t0) / max(1, n), 2),
        "exactly right after decoding %": round(100 * float(np.mean(exact)), 2),
        "decoded to a legal cube %": round(100 * float(np.mean(legal)), 2),
        "mean wrong entries per scan after decoding": round(float(np.mean(decoded_wrong)), 3) if decoded_wrong else None,
        "mean entries the decoder changed": round(float(np.mean(corrected)), 3),
    }
    if any_order:
        m = np.array(margins)
        wrong_flagged = [f for f, e in zip(flagged, exact) if not e]
        out["arrangement margin (nats)"] = {"min": round(float(m.min()), 2), "median": round(float(np.median(m)), 2)}
        out["arrangements flagged as doubtful %"] = round(100 * float(np.mean(flagged)), 2)
        out["wrong scans that were flagged"] = f"{sum(wrong_flagged)} of {len(wrong_flagged)}"
    else:
        out["exactly right before decoding %"] = round(100 * float(np.mean(raw_exact)), 2)
        out["photo orientation found %"] = round(100 * float(np.mean(turn_ok)), 2)
        out["mean wrong entries per scan before decoding (of 144)"] = round(float(np.mean(raw_wrong)), 3)
    out["failures"] = failures[:25]
    return out


def evaluate_real(vision: Vision, folder: Path, limit: int | None = None) -> dict:
    """The real scans kept from the app (data/real/scans/<id>: shot0-5.jpg and scan.json with the
    labels a person checked, or that a known scramble gave), read again with the current networks."""
    import cv2

    dirs = sorted(p for p in Path(folder).iterdir() if (p / "scan.json").exists())
    rows = []
    t0 = time.time()
    for d in dirs:
        meta = json.loads((d / "scan.json").read_text(encoding="utf-8"))
        labels = meta.get("labels")
        if not labels:
            continue
        images = [cv2.cvtColor(cv2.imread(str(d / f"shot{i}.jpg")), cv2.COLOR_BGR2RGB) for i in range(6)]
        result = vision.scan(images)
        truth = Observation.from_json(labels["solverFrame"])
        dec = result.decode
        o = dec.observation
        wrong = None if o is None else int((o.cells != truth.cells).sum() + (o.gear_colour != truth.gear_colour).sum()
                                           + (o.gear_quarter != truth.gear_quarter).sum())
        rows.append({"scan": d.name, "source": labels.get("source", "checked"), "exact": bool(dec.legal and o == truth),
                     "wrong after decoding": wrong, "decoder changed": len(dec.corrections),
                     "arrangement margin": round(result.arrangement.margin, 1) if result.arrangement else None})
        if limit is not None and len(rows) >= limit:
            break
    n = len(rows)
    if not n:
        return {"scans": 0, "note": f"no labelled scans in {folder}: keep some from the app's Check step first"}
    return {
        "scans": n,
        "from a known scramble": sum(r["source"] == "known-scramble" for r in rows),
        "seconds per scan": round((time.time() - t0) / n, 2),
        "exactly right after decoding %": round(100 * float(np.mean([r["exact"] for r in rows])), 2),
        "mean wrong entries per scan after decoding": round(float(np.mean([r["wrong after decoding"] or 0 for r in rows])), 3),
        "mean entries the decoder changed": round(float(np.mean([r["decoder changed"] for r in rows])), 3),
        "per scan": rows,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scanner.vision.evaluate --puzzle maltese", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["faces", "scans", "real"])
    ap.add_argument("--data", help="dataset folder (real: default data/real/scans)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--checkpoints", help="folder with locator.pt and reader.pt (default data/checkpoints)")
    ap.add_argument("--true-corners", action="store_true", help="faces: give the reader the true corners instead of the locator's")
    ap.add_argument("--any-order", action="store_true", help="scans: shuffle each scan's photos; the pipeline works out which face each is")
    ap.add_argument("--out", help="also write the report here as JSON")
    args = ap.parse_args(argv)
    vision = Vision(Path(args.checkpoints) if args.checkpoints else None).load()
    if args.what == "real":
        from scanner.paths import real_dir

        report = evaluate_real(vision, Path(args.data) if args.data else real_dir("maltese") / "scans", args.limit)
    else:
        if not args.data:
            ap.error("--data is needed for faces and scans")
        photos = Photos(args.data)
        if args.what == "faces":
            report = evaluate_faces(vision, photos, args.limit, args.true_corners)
        else:
            report = evaluate_scans(vision, photos, args.limit, any_order=args.any_order)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
