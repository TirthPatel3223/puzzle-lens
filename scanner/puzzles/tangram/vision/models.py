"""The Tangram Cube Extreme's two networks.

``CornerLocator``    one picture in, the photographed face's four corners out, in the face's own order
                     (top-left, top-right, bottom-right = its turning corner, bottom-left). A ResNet-18 with
                     a small FPN gives one heatmap per corner; each corner is its map's centre of mass.
                     Finding the turning corner (the T or B corner piece, which only ever twists in place)
                     tells the scanner which way up the photo is.

``PositionReader``   the 29 position crops of a face photo in, what each position shows out: one of the
                     six colours, empty (no sticker is there: the piece that would put one there is
                     elsewhere), or hidden (a sticker is there, but something stands in front of it). One
                     ResNet-18 trunk reads every crop; a small network looks at the whole face and tells
                     every crop what the light is like; a learned vector per position tells the head which
                     position it is looking at (flat or turned out, and its shape).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scanner.vision.common import HeatmapNet, Normalise, heatmap_point_loss, resnet18, soft_argmax

from .geometry import N_CLASSES


class CornerLocator(HeatmapNet):
    """(B, 3, S, S) uint8 -> (B, 4, S/4, S/4) heatmap logits, one per face corner (face order)."""

    def __init__(self, pretrained: bool = True, width: int = 96) -> None:
        super().__init__(4, pretrained, width)

    @staticmethod
    def corners(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Corners (B, 4, 2) as fractions of the picture, and a confidence per corner (B, 4)."""
        return soft_argmax(logits)


def locator_loss(logits: torch.Tensor, corners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return heatmap_point_loss(logits, corners)


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


class PositionReader(nn.Module):
    def __init__(self, positions: int = 29, pretrained: bool = True, context: int = 64, embed: int = 32, hidden: int = 384,
                 dropout: float = 0.1, squeeze: int = 64) -> None:
        super().__init__()
        r = resnet18(pretrained)
        self.positions = positions
        self.norm = Normalise()
        # No max-pool: the crops are small, and the cut lines that tell flat from turned out are thin.
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.layer1, r.layer2, r.layer3, r.layer4)   # 1/16
        self.squeeze = nn.Sequential(nn.Conv2d(512, squeeze, 1, bias=False), nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.context = _Context(context)
        self.position = nn.Embedding(positions, embed)
        feat = squeeze * 16 + context + embed
        self.head = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, N_CLASSES))
        #: Divides the logits before the softmax; fitted on held-out data after training.
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, crops: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """crops (B, n, 3, S, S) uint8 and context (B, 3, C, C) uint8 -> logits (B, n, 8)."""
        b, n = crops.shape[:2]
        x = self.norm(crops.flatten(0, 1))
        f = self.pool(self.squeeze(self.trunk(x))).flatten(1).view(b, n, -1)
        ctx = self.context(self.norm(context)).unsqueeze(1).expand(b, n, -1)
        pos = self.position.weight[:n].unsqueeze(0).expand(b, n, -1)
        return self.head(torch.cat([f, ctx, pos], -1))

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.float() / self.temperature).softmax(-1)


def reader_loss(logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), label_smoothing=smoothing)


@torch.no_grad()
def reader_hits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per kind of answer, (k,) booleans: was it read right? And per photo, all 29 right."""
    pred = logits.argmax(-1)
    ok = pred == labels
    colour = labels < 6
    out = {
        "colour": ok[colour],
        "empty": ok[labels == 6],
        "hidden": ok[labels == 7],
        # Sticker or not, whatever the colour: the shape of the puzzle.
        "shape": ((pred == 6) == (labels == 6)).flatten(),
        "photo": ok.all(1),
    }
    return out
