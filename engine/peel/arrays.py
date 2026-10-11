"""Array aliases and small image helpers shared by every stage."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]
Bool = npt.NDArray[np.bool_]
I32 = npt.NDArray[np.int32]


def to_float(rgb: U8) -> F32:
    return rgb.astype(np.float32) / 255.0


def to_u8(rgb: F32) -> U8:
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def rgb_to_lab(rgb: F32) -> F32:
    """sRGB (0..1 float32) to CIELAB D65. L in 0..100, a/b roughly -128..127."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def lab_to_rgb(lab: F32) -> F32:
    rgb = cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(rgb, 0.0, 1.0)


def luminance_u8(rgb: U8) -> U8:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def resize_long_side(rgb: U8, long_side: int) -> U8:
    """Downscale so the long side is at most ``long_side``. Never upscales."""
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return rgb
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def resize_to(img: npt.NDArray[np.generic], width: int, height: int) -> npt.NDArray[np.generic]:
    """Resize picking the right filter for the direction (area down, cubic up)."""
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    shrinking = width * height < w * h
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, (width, height), interpolation=interp)


def resize_mask(mask: Bool, width: int, height: int) -> Bool:
    out = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)


def disk(radius: int) -> U8:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(mask: Bool, radius: int) -> Bool:
    if radius <= 0:
        return mask
    return cv2.dilate(mask.astype(np.uint8), disk(radius)).astype(bool)


def erode(mask: Bool, radius: int) -> Bool:
    if radius <= 0:
        return mask
    return cv2.erode(mask.astype(np.uint8), disk(radius)).astype(bool)


def robust_stats(values: npt.NDArray[np.floating]) -> tuple[float, float]:
    """Median and MAD scaled to be a consistent estimator of sigma."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826
    return med, mad
