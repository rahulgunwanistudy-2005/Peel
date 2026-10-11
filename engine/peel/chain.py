"""Anchored chains: edit 20 still has the background of edit 0.

Each step peels the new edit against the previous *peeled* frame, so drift cannot accumulate.
Every step is measured against the step-0 original, for the raw chain and the peeled chain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from peel import models
from peel.arrays import U8, Bool, dilate, resize_mask
from peel.io import aspect_matches
from peel.pipeline import PeelOutput, peel_arrays
from peel.types import ChainStepMetrics, PeelOptions

LPIPS_SIDE = 512


@dataclass
class ChainStep:
    output: PeelOutput
    raw: U8  # the raw edit in the original's frame
    metrics: ChainStepMetrics


@dataclass
class ChainResult:
    original: U8
    steps: list[ChainStep]


def psnr(a: U8, b: U8, mask: Bool) -> float:
    if not mask.any():
        return float("inf")
    diff = a[mask].astype(np.float64) - b[mask].astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse < 1e-10 else float(10.0 * np.log10(255.0**2 / mse))


def masked_lpips(a: U8, b: U8, mask: Bool) -> float:
    """Mean spatial LPIPS over ``mask`` (full-resolution boolean)."""
    h, w = a.shape[:2]
    scale = min(1.0, LPIPS_SIDE / max(h, w))
    size = (max(64, round(w * scale)), max(64, round(h * scale)))
    small_a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    dist = models.lpips_model().spatial(small_a, small_b)
    small_mask = resize_mask(mask, size[0], size[1])
    return float(dist[small_mask].mean()) if small_mask.any() else 0.0


def raw_in_original_frame(original: U8, edited: U8, aligned: U8) -> U8:
    """What a user without Peel has: the edit, resized to the original (no alignment help)."""
    if edited.shape[:2] == original.shape[:2]:
        return edited
    if aspect_matches(original, edited):
        h, w = original.shape[:2]
        shrinking = edited.shape[0] > h
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
        return cv2.resize(edited, (w, h), interpolation=interp)
    return aligned


def run_chain(
    original: U8,
    edits: list[U8],
    instructions: list[str],
    options: PeelOptions | None = None,
    *,
    on_step: Callable[[int, int], None] | None = None,
) -> ChainResult:
    """``edits[k]`` is the raw editor output at step k+1, made from the previous raw output."""
    if len(edits) != len(instructions):
        raise ValueError("edits and instructions must have the same length")
    options = options or PeelOptions()
    height, width = original.shape[:2]
    edited_so_far = np.zeros((height, width), dtype=bool)
    anchor = original
    steps: list[ChainStep] = []
    for index, (edit, instruction) in enumerate(zip(edits, instructions, strict=True), start=1):
        if on_step is not None:
            on_step(index, len(edits))
        output = peel_arrays(anchor, edit, instruction, options)
        kept = np.zeros(output.analysis.change.mask.shape, dtype=bool)
        for rm in output.analysis.regions:
            if rm.region.decision == "keep":
                kept |= rm.mask
        edited_so_far |= dilate(resize_mask(kept, width, height), max(2, round(0.01 * width)))
        unedited = ~edited_so_far
        raw = raw_in_original_frame(original, edit, output.analysis.aligned)
        metrics = ChainStepMetrics(
            step=index,
            instruction=instruction,
            raw_lpips=round(masked_lpips(raw, original, unedited), 4),
            peeled_lpips=round(masked_lpips(output.peeled, original, unedited), 4),
            raw_psnr=round(psnr(raw, original, unedited), 2),
            peeled_psnr=round(psnr(output.peeled, original, unedited), 2),
            drift_score=output.ledger.drift_score,
        )
        steps.append(ChainStep(output=output, raw=raw, metrics=metrics))
        anchor = output.peeled
    return ChainResult(original=original, steps=steps)
