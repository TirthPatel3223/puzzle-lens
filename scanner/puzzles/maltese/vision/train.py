"""Train the two networks on the synthetic photos.

    python -m scanner.vision.train reader  --puzzle maltese --data D:/cube_data/synth/v1 --epochs 12
    python -m scanner.vision.train locator --puzzle maltese --data D:/cube_data/synth/v1 --epochs 12

``--data`` holds ``train/`` and ``val/`` (tools/gen_dataset.py). The best checkpoint on the
validation set is written to the Maltese checkpoint folder (scanner.paths.checkpoints_dir), with its
scores beside it as JSON.
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
from scanner.vision.common import fit_temperature
from scanner.vision.common import loader as _loader
from scanner.vision.common import schedule as _schedule
from scanner.vision.common import to_device as _to

from .dataset import LocatorData, Photos, ReaderData
from .models import FaceLocator, SlotReader, locator_loss, reader_hits, reader_loss


# ------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_reader(model: SlotReader, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.eval()
    hits: dict[str, list[torch.Tensor]] = {}
    logits_all: dict[str, list[torch.Tensor]] = {k: [] for k in SlotReader.OUTPUTS}
    labels_all: dict[str, list[torch.Tensor]] = {k: [] for k in SlotReader.OUTPUTS}
    for batch in loader:
        batch = _to(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch["crops"], batch["context"])
        for k, v in reader_hits(logits, batch).items():
            hits.setdefault(k, []).append(v.cpu())
        for k in SlotReader.OUTPUTS:
            logits_all[k].append(logits[k].float().cpu())
            labels_all[k].append(batch[k].cpu())
    scores = {k: float(torch.cat(v).float().mean()) for k, v in hits.items()}
    return scores, {k: torch.cat(v) for k, v in logits_all.items()}, {k: torch.cat(v) for k, v in labels_all.items()}


def train_reader(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = ReaderData(Photos(Path(args.data) / "train", args.limit), train=True, strength=args.strength)
    val = ReaderData(Photos(Path(args.data) / "val"), train=False)
    tl, vl = _loader(train, args.batch, args.workers, True), _loader(val, args.batch * 2, min(args.workers, 6), False)
    model = SlotReader(pretrained=not args.scratch).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu")["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(tl)
    sched = _schedule(opt, total, min(600, total // 10))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.out or checkpoints_dir("maltese") / "reader.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    print(f"reader: {len(train)} training faces, {len(val)} validation faces, {len(tl)} steps per epoch, on {device}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        t0, seen, run = time.time(), 0, 0.0
        for step, batch in enumerate(tl):
            batch = _to(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["crops"], batch["context"])
            loss = reader_loss(logits, batch, args.smoothing)
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
                print(f"  epoch {epoch + 1} step {step + 1}/{len(tl)}  loss {run / seen:.4f}  {seen / (time.time() - t0):.0f} faces/s", flush=True)
        scores, logits_v, labels_v = evaluate_reader(model, vl, device)
        line = "  ".join(f"{k} {100 * v:.2f}" for k, v in scores.items())
        print(f"epoch {epoch + 1}/{args.epochs}  loss {run / seen:.4f}  {seen / (time.time() - t0):.0f} faces/s  |  val %: {line}", flush=True)
        if scores["face"] > best:
            best = scores["face"]
            temps = [fit_temperature(logits_v[k], labels_v[k]) for k in SlotReader.OUTPUTS]
            model.temperature.copy_(torch.tensor(temps))
            torch.save({"model": model.state_dict(), "kind": "reader", "epoch": epoch + 1, "scores": scores, "temperature": temps,
                        "args": vars(args)}, out)
            out.with_suffix(".json").write_text(json.dumps({"epoch": epoch + 1, "scores": scores, "temperature": temps}, indent=2))
    print(f"best: {100 * best:.2f}% of validation faces read without a single mistake -> {out}", flush=True)


# ------------------------------------------------------------------------------------------
# Locator
# ------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_locator(model: FaceLocator, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Corner error as a fraction of the face's side (so 0.01 is 0.6 mm on the cube)."""
    model.eval()
    errs = []
    for batch in loader:
        batch = _to(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch["image"])
        pred, _ = FaceLocator.corners(logits)
        c = batch["corners"]
        side = (c - c.roll(-1, dims=1)).norm(dim=-1).mean(1, keepdim=True)
        errs.append(((pred - c).norm(dim=-1) / side).cpu())
    e = torch.cat(errs)
    worst = e.max(1).values
    return {"mean": float(e.mean()), "p95": float(e.flatten().quantile(0.95)), "worst_corner_under_2pct": float((worst < 0.02).float().mean()),
            "worst_corner_under_4pct": float((worst < 0.04).float().mean())}


def train_locator(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = LocatorData(Photos(Path(args.data) / "train", args.limit), train=True, strength=args.strength)
    val = LocatorData(Photos(Path(args.data) / "val"), train=False)
    tl, vl = _loader(train, args.batch, args.workers, True), _loader(val, args.batch * 2, min(args.workers, 6), False)
    model = FaceLocator(pretrained=not args.scratch).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu")["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(tl)
    sched = _schedule(opt, total, min(600, total // 10))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.out or checkpoints_dir("maltese") / "locator.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    best = np.inf
    print(f"locator: {len(train)} training pictures, {len(val)} validation pictures, {len(tl)} steps per epoch, on {device}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        t0, seen, run, run_err = time.time(), 0, 0.0, 0.0
        for step, batch in enumerate(tl):
            batch = _to(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["image"])
            loss, err = locator_loss(logits, batch["corners"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n = len(batch["image"])
            run += float(loss.detach()) * n
            run_err += float(err.detach()) * n
            seen += n
            if args.log and (step + 1) % args.log == 0:
                print(f"  epoch {epoch + 1} step {step + 1}/{len(tl)}  loss {run / seen:.4f}  {seen / (time.time() - t0):.0f} pictures/s", flush=True)
        scores = evaluate_locator(model, vl, device)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {run / seen:.4f}  train corner error {100 * run_err / seen:.2f}% of the picture  "
              f"{seen / (time.time() - t0):.0f} pictures/s  |  val, % of the face's side: mean {100 * scores['mean']:.2f}  p95 {100 * scores['p95']:.2f}  "
              f"all four corners within 2%: {100 * scores['worst_corner_under_2pct']:.1f}%  within 4%: {100 * scores['worst_corner_under_4pct']:.1f}%", flush=True)
        if scores["mean"] < best:
            best = scores["mean"]
            torch.save({"model": model.state_dict(), "kind": "locator", "epoch": epoch + 1, "scores": scores, "args": vars(args)}, out)
            out.with_suffix(".json").write_text(json.dumps({"epoch": epoch + 1, "scores": scores}, indent=2))
    print(f"best: mean corner error {100 * best:.2f}% of the face's side -> {out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scanner.vision.train --puzzle maltese", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["reader", "locator"])
    ap.add_argument("--data", required=True, help="folder with train/ and val/")
    ap.add_argument("--out", help="checkpoint file (default data/checkpoints/<what>.pt)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--strength", type=float, default=1.0, help="how hard the camera augmentation is, 0-1")
    ap.add_argument("--limit", type=int, help="use only the first N training pictures")
    ap.add_argument("--scratch", action="store_true", help="do not start from ImageNet weights")
    ap.add_argument("--resume", help="start from this checkpoint's weights")
    ap.add_argument("--log", type=int, default=0, help="print every N steps")
    args = ap.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    (train_reader if args.what == "reader" else train_locator)(args)


if __name__ == "__main__":
    main()
