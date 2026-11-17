"""What a real camera does to a picture, applied to the clean synthetic renders during training.

Everything works on HxWx3 uint8 RGB arrays and returns the same. The renders already vary the
light, the sticker shades and the background; this adds the sensor and the file format: white
balance that is a little off (now and then a warm webcam that makes red look orange), exposure and contrast, soft focus, hand shake, low resolution,
uneven light, noise, and JPEG blocks.

Speed matters here (it runs on every training picture), so everything that acts on one pixel
at a time is folded into a single 256-entry lookup table per channel.
"""
from __future__ import annotations

import cv2
import numpy as np

_RAMP = np.arange(256, dtype=np.float32) / 255.0


def tone_table(rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """One lookup table per channel (256, 3): white balance, exposure, gamma, contrast, lifted blacks."""
    warm = rng.normal(0.0, 0.07) * strength           # warm (+) or cool (-), as auto white balance leaves behind
    green = rng.normal(0.0, 0.03) * strength          # green (+) or magenta (-)
    if rng.random() < 0.2 * strength:
        # A warm, bright webcam: red runs into its ceiling and red stickers come out salmon or orange-red, close
        # to the orange ones (real scans show this); the difference left is in green and blue.
        warm += rng.uniform(0.08, 0.22)
        green += rng.uniform(0.0, 0.08)
    gains = np.array([1.0 + warm, 1.0 + green, 1.0 - warm], dtype=np.float32) * np.exp(rng.normal(0.0, 0.14) * strength)
    gamma = np.exp(rng.normal(0.0, 0.16) * strength)
    contrast = np.exp(rng.normal(0.0, 0.12) * strength)
    lift = rng.uniform(0.02, 0.12) if rng.random() < 0.15 * strength else 0.0    # haze, a dirty lens
    x = np.power(_RAMP[:, None] * gains[None, :], gamma)
    x = (x - 0.5) * contrast + 0.5
    x = x * (1.0 - lift) + lift
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_table(img: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``img`` (..., 3) uint8 through a (256, 3) table."""
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = table[:, c][img[..., c]]
    return out


def saturate(img: np.ndarray, s: float) -> np.ndarray:
    """Scale the colourfulness of an HxWx3 picture by ``s``."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.addWeighted(img, s, cv2.merge([grey, grey, grey]), 1.0 - s, 0.0)


def soften(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Out of focus, hand shake, or simply few pixels."""
    r = rng.random()
    if r < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 2.4))
    elif r < 0.5:
        n = int(rng.integers(3, 10))
        kernel = np.zeros((n, n), dtype=np.float32)
        angle = rng.uniform(0, np.pi)
        c = (n - 1) / 2
        for t in np.linspace(-c, c, 2 * n):
            kernel[int(round(c + t * np.sin(angle))), int(round(c + t * np.cos(angle)))] = 1.0
        img = cv2.filter2D(img, -1, kernel / kernel.sum())
    if rng.random() < 0.3:
        h, w = img.shape[:2]
        f = rng.uniform(0.3, 0.85)
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR if rng.random() < 0.7 else cv2.INTER_CUBIC)
    return img


def uneven_light(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Darker towards the corners (a small lens) or brighter towards one side (a window)."""
    n = 24
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / (n - 1) - 0.5
    gain = np.ones((n, n), dtype=np.float32)
    if rng.random() < 0.5:
        cx, cy = rng.uniform(-0.15, 0.15, 2)
        gain *= 1.0 - rng.uniform(0.15, 0.7) * ((x - cx) ** 2 + (y - cy) ** 2)
    else:
        a = rng.uniform(0, 2 * np.pi)
        gain *= 1.0 + (x * np.cos(a) + y * np.sin(a)) * rng.uniform(0.15, 0.7)
    full = cv2.resize(gain, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.multiply(img, cv2.merge([full, full, full]), dtype=cv2.CV_8U)


def sensor_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = np.empty(img.shape, dtype=np.int16)
    cv2.setRNGSeed(int(rng.integers(1 << 31)))
    cv2.randn(noise, 0, float(rng.uniform(0.5, 9.0)))
    return cv2.add(img, noise, dtype=cv2.CV_8U)


def jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img      # the channel order does not matter to JPEG blocks


def camera(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """The whole chain. ``strength`` 0 leaves the picture alone, 1 is the full training range."""
    if strength <= 0:
        return img
    if rng.random() < 0.6 * strength:
        img = soften(img, rng)
    img = apply_table(img, tone_table(rng, strength))
    if rng.random() < 0.7 * strength:
        img = saturate(img, float(np.exp(rng.normal(0.0, 0.16) * strength)))
    if rng.random() < 0.4 * strength:
        img = uneven_light(img, rng)
    if rng.random() < 0.6 * strength:
        img = sensor_noise(img, rng)
    if rng.random() < 0.6 * strength:
        img = jpeg(img, rng)
    return img
