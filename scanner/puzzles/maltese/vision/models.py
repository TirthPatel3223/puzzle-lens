"""The two networks.

``FaceLocator``  one picture in, the face's four corners out (top-left, top-right, bottom-right,
                 bottom-left as the picture shows them). A ResNet-18 with a small decoder gives
                 one heatmap per corner; the corner is the heatmap's centre of mass, so it is
                 found to a fraction of a heatmap pixel.

``SlotReader``   the 13 slot crops of a flattened face in, what each slot shows out: a colour
                 for each star, a colour and a quarter turn for each gear, two colours for each
                 edge half and four for the centre slot. One ResNet-18 trunk reads every crop;
                 a small second network looks at the whole face and tells every crop what the
                 light is like (a white sticker under a warm lamp looks like a yellow one unless
                 you can see the rest of the face).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scanner.vision.common import HeatmapNet, Normalise, resnet18, soft_argmax

from .geometry import CENTRE, EDGE, GEAR, STAR

# Kept for code that imported them from here.
_resnet18 = resnet18
_Normalise = Normalise


# ------------------------------------------------------------------------------------------
# FaceLocator
# ------------------------------------------------------------------------------------------

class FaceLocator(HeatmapNet):
    """(B, 3, S, S) uint8 -> (B, 4, S/4, S/4) heatmap logits, one per face corner."""

    def __init__(self, pretrained: bool = True, width: int = 96) -> None:
        super().__init__(4, pretrained, width)

    @staticmethod
    def corners(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Corners (B, 4, 2) as fractions of the picture, and a confidence (B, 4): the probability
        mass within about two heatmap pixels of the corner (near 1 when the heatmap has one sharp peak)."""
        return soft_argmax(logits)


def locator_loss(logits: torch.Tensor, corners: torch.Tensor, sigma: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance of the predicted corners from the true ones, plus a term that makes each heatmap
    one sharp bump at its corner. Returns (loss, mean corner error as a fraction of the picture)."""
    b, k, h, w = logits.shape
    pred, _ = FaceLocator.corners(logits)
    xs = torch.arange(w, device=logits.device, dtype=torch.float32) + 0.5
    ys = torch.arange(h, device=logits.device, dtype=torch.float32) + 0.5
    tx, ty = corners[..., 0] * w, corners[..., 1] * h
    g = torch.exp(-((xs.view(1, 1, 1, w) - tx.view(b, k, 1, 1)) ** 2 + (ys.view(1, 1, h, 1) - ty.view(b, k, 1, 1)) ** 2) / (2 * sigma ** 2))
    g = g / g.sum((2, 3), keepdim=True).clamp_min(1e-8)
    ce = -(g.flatten(2) * logits.float().flatten(2).log_softmax(-1)).sum(-1).mean()
    err = (pred - corners).norm(dim=-1)
    return F.smooth_l1_loss(pred, corners, beta=0.01) * 100.0 + ce * 0.2, err.mean()


# ------------------------------------------------------------------------------------------
# SlotReader
# ------------------------------------------------------------------------------------------

class _Context(nn.Module):
    def __init__(self, out: int) -> None:
        super().__init__()
        chans = (3, 24, 48, 96, 128)
        self.net = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True))
            for i in range(4)
        ])
        self.fc = nn.Linear(128, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).mean((2, 3)))


class SlotReader(nn.Module):
    #: Outputs of each kind of crop.
    HEADS = {"star": 6, "gear": 6 + 4, "edge": 2 * 6, "centre": 4 * 6}

    def __init__(self, pretrained: bool = True, context: int = 64, hidden: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        r = resnet18(pretrained)
        self.norm = Normalise()
        # No max-pool: the crops are small (96 px), and what matters in them is small too.
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.layer1, r.layer2, r.layer3, r.layer4)   # 1/16
        self.squeeze = nn.Sequential(nn.Conv2d(512, 96, 1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.context = _Context(context)
        feat = 96 * 6 * 6 + context
        self.heads = nn.ModuleDict({
            kind: nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, n))
            for kind, n in self.HEADS.items()
        })
        #: Divides each kind's logits before the softmax; fitted on held-out data after training.
        self.register_buffer("temperature", torch.ones(5))    # star, gear colour, gear quarter, edge, centre

    def forward(self, crops: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        """crops (B, 13, 3, 96, 96) uint8 and context (B, 3, C, C) uint8 -> logits:
        star (B, 4, 6), gear_colour (B, 4, 6), gear_quarter (B, 4, 4), edge (B, 4, 2, 6), centre (B, 4, 6)."""
        b, n = crops.shape[:2]
        x = self.norm(crops.flatten(0, 1))
        f = self.squeeze(self.trunk(x)).flatten(1).view(b, n, -1)
        ctx = self.context(self.norm(context)).unsqueeze(1).expand(b, n, -1)
        f = torch.cat([f, ctx], -1)
        gear = self.heads["gear"](f[:, GEAR])
        return {
            "star": self.heads["star"](f[:, STAR]),
            "gear_colour": gear[..., :6],
            "gear_quarter": gear[..., 6:],
            "edge": self.heads["edge"](f[:, EDGE]).view(b, 4, 2, 6),
            "centre": self.heads["centre"](f[:, CENTRE]).view(b, 4, 6),
        }

    OUTPUTS = ("star", "gear_colour", "gear_quarter", "edge", "centre")

    def probabilities(self, logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: (logits[k].float() / self.temperature[i]).softmax(-1) for i, k in enumerate(self.OUTPUTS)}


def reader_loss(logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], smoothing: float = 0.0) -> torch.Tensor:
    total = logits["star"].new_zeros(())
    for k in SlotReader.OUTPUTS:
        total = total + F.cross_entropy(logits[k].flatten(0, -2).float(), batch[k].flatten(), label_smoothing=smoothing)
    return total


@torch.no_grad()
def reader_hits(logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per output kind (B, n) booleans: was every part of it read right? And per face."""
    hits = {k: (logits[k].argmax(-1) == batch[k]).flatten(1) for k in SlotReader.OUTPUTS}
    hits["face"] = torch.cat(list(hits.values()), 1).all(1, keepdim=True)
    return hits
