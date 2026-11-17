"""Score the Tangram Cube Extreme's networks the way they are used: from the picture alone.

    python -m scanner.vision.evaluate photos --puzzle tangram --data ../puzzle-data/tangram/v1/val
    python -m scanner.vision.evaluate scans  --puzzle tangram --data ../puzzle-data/tangram/v1/scans
    python -m scanner.vision.evaluate real   --puzzle tangram         # labelled real scans kept from the app

``photos``  every picture is located and read with no help: how often the turning corner is found (the
            photo the right way up), how far off the corners are, and how often each kind of answer
            (a colour, empty, hidden) is right.
``scans``   every six-photo scan goes through the whole pipeline with its photos shuffled (locate, read,
            work out which face each photo is, decode to a legal state) and is compared with the truth.
``real``    the same for real photos kept from the app's Check step (data/real/tangram/scans).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scanner.core.colours import find_class
from scanner.vision.common import quad_side

from ..layout import EMPTY, load_layout
from .dataset import Photos
from .geometry import HIDDEN, face_geometry
from .pipeline import DOUBT_NATS, TangramVision


def _pct(xs: list) -> float | None:
    return round(100 * float(np.mean(xs)), 2) if xs else None


def evaluate_photos(vision: TangramVision, photos: Photos, limit: int | None = None, true_corners: bool = False) -> dict:
    geom = face_geometry(vision.layout)
    n = len(photos) if limit is None else min(limit, len(photos))
    hits = defaultdict(list)
    corner_err, turn_ok = [], []
    by_setting = defaultdict(list)
    t0 = time.time()
    for i in range(n):
        img = photos.image(i)
        truth = photos.quads[i]
        if true_corners:
            quad = truth
        else:
            quad, _ = vision.locate(img)
            corner_err.append(float(np.linalg.norm(quad - truth, axis=1).max() / quad_side(truth)))
            turn_ok.append(int(np.argmin(np.linalg.norm(truth - quad[2], axis=1))) == 2)
        pred = vision.read(img, quad).argmax(1)
        lab = photos.face_labels(i, geom)
        for name, mask in (("colour", lab < 6), ("empty", lab == EMPTY), ("hidden", lab == HIDDEN)):
            if mask.any():
                hits[name].append(float((pred[mask] == lab[mask]).mean()))
        hits["sticker or not"].append(float(((pred == EMPTY) == (lab == EMPTY)).mean()))
        ok = bool((pred == lab).all())
        hits["whole photo"].append(ok)
        by_setting[str(photos.setting[i])].append(ok)
    out = {
        "pictures": n,
        "seconds per picture": round((time.time() - t0) / max(1, n), 3),
        "accuracy %": {k: round(100 * float(np.mean(v)), 3) for k, v in hits.items()},
        "whole photo % by setting": {k: round(100 * float(np.mean(v)), 2) for k, v in sorted(by_setting.items())},
    }
    if corner_err:
        e = np.array(corner_err)
        out["photo the right way up %"] = round(100 * float(np.mean(turn_ok)), 3)
        out["worst corner error, % of the face's side"] = {"median": round(100 * float(np.median(e)), 2), "p95": round(100 * float(np.quantile(e, 0.95)), 2),
                                                           "over 6%": round(100 * float((e > 0.06).mean()), 2)}
    return out


def evaluate_scans(vision: TangramVision, photos: Photos, limit: int | None = None, shuffle_seed: int = 0) -> dict:
    lay = vision.layout
    scans = defaultdict(list)
    for i in range(len(photos)):
        scans[int(photos.scan[i])].append(i)
    ids = sorted(s for s in scans if s >= 0 and len(scans[s]) == 6)[:limit]
    rng = np.random.default_rng(shuffle_seed)
    rows = []
    t0 = time.time()
    for s in ids:
        idx = sorted(scans[s], key=lambda i: int(photos.shot[i]))
        paint = tuple(int(x) for x in photos.paint[idx[0]])
        shown = photos.content[idx[0]].astype(np.int64)            # what the puzzle shows (its colour layout applied)
        order = rng.permutation(6)
        result = vision.scan([photos.image(idx[j]) for j in order])
        d = result.decode
        # Right when the decoded state, painted in the layout found, shows exactly what the puzzle shows.
        seen = result.colours.to_cube(d.observation) if d.observation is not None else None
        exact = bool(d.legal and np.array_equal(seen, shown))
        layout_right = lay.colour_layouts[find_class(lay.colour_layouts, paint, lay.colour_actions)] == result.colours
        faces_ok = [int(photos.faces[idx[j]]) for j in order] == list(result.faces)
        rows.append({"scan": s, "exact": exact, "legal": d.legal, "faces right": faces_ok, "margin": result.margin,
                     "usual colours": paint == tuple(range(6)), "colour layout right": layout_right,
                     "wrong positions": int((seen != shown).sum()) if seen is not None else None,
                     "repaired": bool(d.repairs), "turned": result.returned, "setting": str(photos.setting[idx[0]])})
    n = len(rows)
    if not n:
        return {"scans": 0, "note": "no complete six-photo scans in this folder (render them with gen_dataset.py --scans)"}
    m = np.array([r["margin"] for r in rows])
    wrong = [r for r in rows if not r["exact"]]
    return {
        "scans": n,
        "seconds per scan": round((time.time() - t0) / n, 2),
        "exactly right %": round(100 * float(np.mean([r["exact"] for r in rows])), 2),
        "decoded to a legal state %": round(100 * float(np.mean([r["legal"] for r in rows])), 2),
        "faces worked out right %": round(100 * float(np.mean([r["faces right"] for r in rows])), 2),
        "colour layout right %": round(100 * float(np.mean([r["colour layout right"] for r in rows])), 2),
        "exactly right %, usual colours": _pct([r["exact"] for r in rows if r["usual colours"]]),
        "exactly right %, stickered another way": _pct([r["exact"] for r in rows if not r["usual colours"]]),
        "mean wrong positions per scan": round(float(np.mean([r["wrong positions"] or 0 for r in rows])), 3),
        "repaired by the decoder %": round(100 * float(np.mean([r["repaired"] for r in rows])), 2),
        "margin (nats)": {"min": round(float(m.min()), 2), "median": round(float(np.median(m)), 2)},
        "flagged as doubtful %": round(100 * float((m < DOUBT_NATS).mean()), 2),
        "wrong scans that were flagged": f"{sum(r['margin'] < DOUBT_NATS for r in wrong)} of {len(wrong)}",
        "failures": wrong[:25],
    }


def evaluate_real(vision: TangramVision, folder: Path, limit: int | None = None) -> dict:
    """The real scans kept from the app (data/real/tangram/scans/<id>: shot0-5.jpg and scan.json with the
    state a person checked, or that a known scramble gave), read again with the current networks."""
    import cv2

    lay = load_layout()
    dirs = sorted(p for p in Path(folder).iterdir() if (p / "scan.json").exists()) if Path(folder).is_dir() else []
    rows = []
    t0 = time.time()
    for d in dirs:
        meta = json.loads((d / "scan.json").read_text(encoding="utf-8"))
        labels = meta.get("labels")
        if not labels:
            continue
        images = [cv2.cvtColor(cv2.imread(str(d / f"shot{i}.jpg")), cv2.COLOR_BGR2RGB) for i in range(6)]
        result = vision.scan(images)
        truth = np.array(labels["orient"], dtype=np.int64)
        dec = result.decode
        rows.append({"scan": d.name, "source": labels.get("source", "checked"), "exact": bool(dec.legal and np.array_equal(dec.orient, truth)),
                     "wrong positions": int((lay.observe(dec.orient) != lay.observe(truth)).sum()) if dec.orient is not None else None,
                     "margin": round(result.margin, 1)})
        if limit is not None and len(rows) >= limit:
            break
    n = len(rows)
    if not n:
        return {"scans": 0, "note": f"no labelled scans in {folder}: keep some from the app's Check step first"}
    return {
        "scans": n,
        "from a known scramble": sum(r["source"] == "known-scramble" for r in rows),
        "seconds per scan": round((time.time() - t0) / n, 2),
        "exactly right %": round(100 * float(np.mean([r["exact"] for r in rows])), 2),
        "mean wrong positions per scan": round(float(np.mean([r["wrong positions"] or 0 for r in rows])), 3),
        "per scan": rows,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scanner.vision.evaluate --puzzle tangram", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["photos", "scans", "real"])
    ap.add_argument("--data", help="dataset folder (real: default data/real/tangram/scans)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--checkpoints", help="folder with locator.pt and reader.pt (default data/checkpoints/tangram)")
    ap.add_argument("--true-corners", action="store_true", help="photos: give the reader the true corners instead of the locator's")
    ap.add_argument("--out", help="also write the report here as JSON")
    ap.add_argument("--device", help="cuda or cpu (default: the GPU if there is one)")
    args = ap.parse_args(argv)
    vision = TangramVision(Path(args.checkpoints) if args.checkpoints else None, device=args.device).load()
    if args.what == "real":
        from scanner.paths import real_dir

        report = evaluate_real(vision, Path(args.data) if args.data else real_dir("tangram") / "scans", args.limit)
    else:
        if not args.data:
            ap.error("--data is needed for photos and scans")
        photos = Photos(args.data)
        report = evaluate_photos(vision, photos, args.limit, args.true_corners) if args.what == "photos" else evaluate_scans(vision, photos, args.limit)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
