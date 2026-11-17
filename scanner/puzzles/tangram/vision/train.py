"""Train the Tangram Cube Extreme's two networks on the synthetic photos.

    python -m scanner.vision.train locator --puzzle tangram --data ../puzzle-data/tangram/v1 --epochs 10
    python -m scanner.vision.train reader  --puzzle tangram --data ../puzzle-data/tangram/v1 --epochs 8

``--data`` holds ``train/`` and ``val/`` (tools/gen_dataset.py --puzzle tangram). The best checkpoint on the
validation set is written to data/checkpoints/tangram/<what>.pt, with its scores beside it as JSON.
``--resume`` starts from a checkpoint's weights (to fine-tune, or to go on after a stop).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scanner.paths import checkpoints_dir
from scanner.vision.common import device as pick_device
from scanner.vision.common import fit_temperature, loader, schedule, to_device

from .dataset import LocatorData, Photos, ReaderData
from .geometry import face_geometry
from .models import CornerLocator, PositionReader, locator_loss, reader_hits, reader_loss

KEY = "tangram"


def _save(out: Path, payload: dict, scores: dict) -> None:
    torch.save(payload, out)
    out.with_suffix(".json").write_text(json.dumps({"epoch": payload["epoch"], "scores": scores, **({"temperature": payload["temperature"]} if "temperature" in payload else {})}, indent=2))


# ------------------------------------------------------------------------------------------ reader

@torch.no_grad()
def evaluate_reader(model: PositionReader, dl: DataLoader, device: torch.device) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    hits: dict[str, list[torch.Tensor]] = {}
    logits_all, labels_all = [], []
    for batch in dl:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch["crops"], batch["context"])
        for k, v in reader_hits(logits, batch["labels"]).items():
            hits.setdefault(k, []).append(v.cpu())
        hits.setdefault("position", []).append((logits.argmax(-1) == batch["labels"]).flatten().cpu())
        logits_all.append(logits.float().cpu())
        labels_all.append(batch["labels"].cpu())
    scores = {k: float(torch.cat(v).float().mean()) if sum(len(x) for x in v) else float("nan") for k, v in hits.items()}
    return scores, torch.cat(logits_all), torch.cat(labels_all)


def train_reader(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else pick_device()
    geom = face_geometry()
    train = ReaderData(Photos(Path(args.data) / "train", args.limit), train=True, strength=args.strength)
    val = ReaderData(Photos(Path(args.data) / "val", args.val_limit), train=False)
    tl, vl = loader(train, args.batch, args.workers, True), loader(val, args.batch * 2, min(args.workers, 6), False)
    model = PositionReader(geom.n, pretrained=not args.scratch).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(tl)
    sched = schedule(opt, total, min(600, total // 10))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.out or checkpoints_dir(KEY) / "reader.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    print(f"reader: {len(train)} training photos, {len(val)} validation photos, {geom.n} positions each, {len(tl)} steps per epoch, on {device}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        t0, seen, run = time.time(), 0, 0.0
        for step, batch in enumerate(tl):
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["crops"], batch["context"])
            loss = reader_loss(logits, batch["labels"], args.smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.detach()) * len(batch["crops"])
            seen += len(batch["crops"])
            if args.log and (step + 1) % args.log == 0:
                print(f"  epoch {epoch + 1} step {step + 1}/{len(tl)}  loss {run / seen:.4f}  {seen / (time.time() - t0):.0f} photos/s", flush=True)
        scores, logits_v, labels_v = evaluate_reader(model, vl, device)
        line = "  ".join(f"{k} {100 * v:.2f}" for k, v in scores.items())
        print(f"epoch {epoch + 1}/{args.epochs}  loss {run / seen:.4f}  {seen / (time.time() - t0):.0f} photos/s  |  val %: {line}", flush=True)
        if scores["position"] > best:
            best = scores["position"]
            t = fit_temperature(logits_v, labels_v)
            model.temperature.fill_(t)
            _save(out, {"model": model.state_dict(), "kind": "reader", "puzzle": KEY, "epoch": epoch + 1, "scores": scores,
                        "temperature": [t], "positions": geom.n, "args": vars(args)}, scores)
    print(f"best: {100 * best:.2f}% of validation positions read right -> {out}", flush=True)


# ------------------------------------------------------------------------------------------ locator

@torch.no_grad()
def evaluate_locator(model: CornerLocator, dl: DataLoader, device: torch.device) -> dict[str, float]:
    """Corner error as a fraction of the face's side; and how often the turning corner is the right one."""
    model.eval()
    errs, turn_ok = [], []
    for batch in dl:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch["image"])
        pred, _ = CornerLocator.corners(logits)
        c = batch["corners"]
        side = (c - c.roll(-1, dims=1)).norm(dim=-1).mean(1, keepdim=True)
        errs.append(((pred - c).norm(dim=-1) / side).cpu())
        # The photo is the right way up when the predicted turning corner is nearest the true one.
        d = (pred[:, 2:3, :] - c).norm(dim=-1)
        turn_ok.append((d.argmin(1) == 2).cpu())
    e = torch.cat(errs)
    worst = e.max(1).values
    return {"mean": float(e.mean()), "p95": float(e.flatten().quantile(0.95)), "worst_corner_under_3pct": float((worst < 0.03).float().mean()),
            "worst_corner_under_6pct": float((worst < 0.06).float().mean()), "turn_right": float(torch.cat(turn_ok).float().mean())}


def train_locator(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else pick_device()
    train = LocatorData(Photos(Path(args.data) / "train", args.limit), train=True, strength=args.strength)
    val = LocatorData(Photos(Path(args.data) / "val", args.val_limit), train=False)
    tl, vl = loader(train, args.batch, args.workers, True), loader(val, args.batch * 2, min(args.workers, 6), False)
    model = CornerLocator(pretrained=not args.scratch).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(tl)
    sched = schedule(opt, total, min(600, total // 10))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.out or checkpoints_dir(KEY) / "locator.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    best = np.inf
    print(f"locator: {len(train)} training pictures, {len(val)} validation pictures, {len(tl)} steps per epoch, on {device}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        t0, seen, run = time.time(), 0, 0.0
        for step, batch in enumerate(tl):
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["image"])
            loss, _ = locator_loss(logits, batch["corners"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.detach()) * len(batch["image"])
            seen += len(batch["image"])
            if args.log and (step + 1) % args.log == 0:
                print(f"  epoch {epoch + 1} step {step + 1}/{len(tl)}  loss {run / seen:.4f}  {seen / (time.time() - t0):.0f} pictures/s", flush=True)
        s = evaluate_locator(model, vl, device)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {run / seen:.4f}  {seen / (time.time() - t0):.0f} pictures/s  |  val, % of the face's side: "
              f"mean {100 * s['mean']:.2f}  p95 {100 * s['p95']:.2f}  all four within 3%: {100 * s['worst_corner_under_3pct']:.1f}%  "
              f"within 6%: {100 * s['worst_corner_under_6pct']:.1f}%  right way up: {100 * s['turn_right']:.2f}%", flush=True)
        # Being the right way up matters most (a photo read upside down is useless); then the corners.
        key = s["mean"] + (1.0 - s["turn_right"])
        if key < best:
            best = key
            _save(out, {"model": model.state_dict(), "kind": "locator", "puzzle": KEY, "epoch": epoch + 1, "scores": s, "args": vars(args)}, s)
    print(f"best checkpoint -> {out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scanner.vision.train --puzzle tangram", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["reader", "locator"])
    ap.add_argument("--data", required=True, help="folder with train/ and val/")
    ap.add_argument("--out", help="checkpoint file (default data/checkpoints/tangram/<what>.pt)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--strength", type=float, default=1.0, help="how hard the camera augmentation is, 0-1")
    ap.add_argument("--limit", type=int, help="use only the first N training pictures")
    ap.add_argument("--val-limit", type=int, help="use only the first N validation pictures")
    ap.add_argument("--scratch", action="store_true", help="do not start from ImageNet weights")
    ap.add_argument("--resume", help="start from this checkpoint's weights")
    ap.add_argument("--log", type=int, default=0, help="print every N steps")
    ap.add_argument("--device", help="cuda, cuda:1 or cpu (default: the GPU if there is one)")
    args = ap.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    (train_reader if args.what == "reader" else train_locator)(args)


if __name__ == "__main__":
    main()
