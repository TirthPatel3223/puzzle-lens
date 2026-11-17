"""Building blocks every puzzle's networks and training share.

``Normalise``, ``resnet18``      uint8 pictures in, an ImageNet ResNet-18 trunk
``HeatmapNet``                   a ResNet-18 + small FPN that turns a picture into heatmaps (one per point
                                 kind); the Maltese face locator and the Tangram corner locator are both one
``soft_argmax``                  a heatmap's centre of mass and how peaked it is
``letterbox``                    fit a picture of any shape into the square a locator looks at
``loader``, ``schedule``, ``fit_temperature``, ``to_device``   training plumbing
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def resnet18(pretrained: bool) -> torchvision.models.ResNet:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.resnet18(weights=weights)


class Normalise(nn.Module):
    """uint8 RGB in, normalised float out (so the data loader ships bytes)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255.0, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / self.std


class HeatmapNet(nn.Module):
    """(B, 3, S, S) uint8 -> (B, maps, S/4, S/4) heatmap logits: ResNet-18 with a top-down FPN."""

    def __init__(self, maps: int, pretrained: bool = True, width: int = 96) -> None:
        super().__init__()
        r = resnet18(pretrained)
        self.norm = Normalise()
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)       # 1/4
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])   # 1/4, 1/8, 1/16, 1/32
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512)])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
            for _ in range(3)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, maps, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(self.norm(image))
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        y = self.lateral[3](feats[3])
        for i in (2, 1, 0):
            y = self.smooth[i](self.lateral[i](feats[i]) + F.interpolate(y, size=feats[i].shape[-2:], mode="bilinear", align_corners=False))
        return self.head(y)


def soft_argmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per map, its centre of mass (B, K, 2) as fractions of the picture, and a confidence (B, K): the
    probability mass within about two heatmap pixels of it (near 1 when the map has one sharp peak)."""
    b, k, h, w = logits.shape
    p = logits.float().flatten(2).softmax(-1).view(b, k, h, w)
    xs = (torch.arange(w, device=logits.device, dtype=torch.float32) + 0.5) / w
    ys = (torch.arange(h, device=logits.device, dtype=torch.float32) + 0.5) / h
    x = (p.sum(2) * xs).sum(-1)
    y = (p.sum(3) * ys).sum(-1)
    d2 = (xs.view(1, 1, 1, w) - x.view(b, k, 1, 1)) ** 2 + (ys.view(1, 1, h, 1) - y.view(b, k, 1, 1)) ** 2
    conf = (p * (d2 < (2.5 / w) ** 2)).sum((2, 3))
    return torch.stack([x, y], -1), conf


def gaussian_targets(points: torch.Tensor, h: int, w: int, sigma: float) -> torch.Tensor:
    """(B, K, 2) points in 0-1 -> (B, K, h, w) Gaussians normalised to sum 1."""
    b, k = points.shape[:2]
    xs = torch.arange(w, device=points.device, dtype=torch.float32) + 0.5
    ys = torch.arange(h, device=points.device, dtype=torch.float32) + 0.5
    tx, ty = points[..., 0] * w, points[..., 1] * h
    g = torch.exp(-((xs.view(1, 1, 1, w) - tx.view(b, k, 1, 1)) ** 2 + (ys.view(1, 1, h, 1) - ty.view(b, k, 1, 1)) ** 2) / (2 * sigma ** 2))
    return g / g.sum((2, 3), keepdim=True).clamp_min(1e-8)


LOCATOR_INPUT = 320


def letterbox(image: np.ndarray, size: int = LOCATOR_INPUT) -> tuple[np.ndarray, float, float, float]:
    """Fit a picture of any shape into a black square. Returns it with the scale and offsets used:
    a point (x, y) of the picture is at (x * scale + dx, y * scale + dy) in the square."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), dtype=image.dtype)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = small
    return out, scale, float(dx), float(dy)


# ------------------------------------------------------------------------------------------ training

def _worker(_: int) -> None:
    cv2.setNumThreads(1)        # every worker is one thread; OpenCV's own pool would fight the others


def loader(ds, batch: int, workers: int, train: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=train, drop_last=train, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
                      worker_init_fn=_worker if workers > 0 else None)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def schedule(opt: torch.optim.Optimizer, total: int, warmup: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up, then a cosine down to 2% of the peak."""
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """The one number that makes the probabilities honest: divide the logits by it (Guo et al. 2017)."""
    logits = logits.flatten(0, -2).float()
    labels = labels.flatten()
    keep = labels != ignore_index
    logits, labels = logits[keep], labels[keep]
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------------ photos

def quad_side(quad: np.ndarray) -> float:
    """Mean side length of a quadrilateral, pixels."""
    q = np.asarray(quad, dtype=np.float64)
    return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))


def prepare_photo(image: np.ndarray, quad: np.ndarray, face_px: float = 640.0) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a large photo so the face is about ``face_px`` across (crops then sample it without aliasing)."""
    side = quad_side(quad)
    if side <= face_px * 1.25:
        return image, np.asarray(quad, dtype=np.float64)
    f = face_px / side
    small = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return small, np.asarray(quad, dtype=np.float64) * f


def heatmap_point_loss(logits: torch.Tensor, points: torch.Tensor, sigma: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    """For heatmaps with one point each: distance of each map's centre of mass from its point, plus a term
    that makes each map one sharp bump there. Returns (loss, mean point error as a fraction of the picture)."""
    b, k, h, w = logits.shape
    pred, _ = soft_argmax(logits)
    g = gaussian_targets(points, h, w, sigma)
    ce = -(g.flatten(2) * logits.float().flatten(2).log_softmax(-1)).sum(-1).mean()
    err = (pred - points).norm(dim=-1)
    return F.smooth_l1_loss(pred, points, beta=0.01) * 100.0 + ce * 0.2, err.mean()


def split_puzzle(argv: list[str], default: str = "maltese") -> tuple[str, list[str]]:
    """Take ``--puzzle X`` (or ``--puzzle=X``) out of a command line; the rest goes to the puzzle's script."""
    rest, puzzle, i = [], default, 0
    while i < len(argv):
        a = argv[i]
        if a == "--puzzle" and i + 1 < len(argv):
            puzzle = argv[i + 1]
            i += 2
            continue
        if a.startswith("--puzzle="):
            puzzle = a.split("=", 1)[1]
        else:
            rest.append(a)
        i += 1
    return puzzle, rest
