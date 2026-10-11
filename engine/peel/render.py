"""Renderings for the web app: the change map and amplified difference images."""

from __future__ import annotations

import cv2
import numpy as np

from peel.arrays import U8, resize_long_side
from peel.classify import RegionMask

# Process inks. Cyan = kept, magenta = peeled, yellow = needs review.
CYAN = (0, 174, 239)
MAGENTA = (236, 0, 140)
YELLOW = (255, 221, 0)
MAP_LONG_SIDE = 1600
DIFF_GAIN = 6.0


def _ink(rm: RegionMask) -> tuple[int, int, int]:
    region = rm.region
    if region.classification == "ambiguous" and not region.overridden:
        return YELLOW
    return CYAN if region.decision == "keep" else MAGENTA


def change_map(original: U8, regions: list[RegionMask], mode: str) -> U8:
    """Regions inked over a desaturated original."""
    base = resize_long_side(original, MAP_LONG_SIDE)
    h, w = base.shape[:2]
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    canvas = np.repeat((gray * 0.7 + 255.0 * 0.18)[..., None], 3, axis=2)
    # Paint large regions first so small ones stay visible on top.
    for rm in sorted(regions, key=lambda r: -r.region.area_pct):
        mask = cv2.resize(rm.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.medianBlur(mask, 5).astype(bool)
        if not mask.any():
            continue
        ink = np.array(_ink(rm), dtype=np.float32)
        whole_frame = mode == "global" and rm.region.classification == "intended"
        opacity = 0.16 if whole_frame else 0.5
        canvas[mask] = canvas[mask] * (1.0 - opacity) + ink * opacity
        if not whole_frame:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            thickness = max(2, round(max(h, w) / 500))
            cv2.drawContours(canvas, contours, -1, tuple(float(c) for c in ink), thickness)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def amplified_diff(image: U8, reference: U8) -> U8:
    """|image - reference| times a fixed gain. Black means byte-identical."""
    diff = cv2.absdiff(image, reference).astype(np.float32) * DIFF_GAIN
    out = np.clip(diff, 0, 255).astype(np.uint8)
    return resize_long_side(out, MAP_LONG_SIDE)
