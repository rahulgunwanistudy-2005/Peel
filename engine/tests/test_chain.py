from __future__ import annotations

import numpy as np

from peel.arrays import U8
from peel.chain import psnr, run_chain
from peel.synth import COLORS, drift_blur, drift_cast, drift_shift, paste_sprite, render_sprite


def test_psnr_of_identical_images_is_capped(scene: U8) -> None:
    assert psnr(scene, scene, np.ones(scene.shape[:2], dtype=bool)) == 99.0


def test_anchored_chain_does_not_accumulate_drift(astronaut: U8) -> None:
    """Three chained edits, each adding one sprite and a little more drift to the raw frame."""
    spots = [("ball", "red", 380, 40), ("star", "yellow", 30, 380), ("heart", "blue", 400, 400)]
    raw = astronaut
    edits, instructions = [], []
    for name, color, x, y in spots:
        sprite, mask = render_sprite(name, COLORS[color], 90)  # type: ignore[arg-type]
        raw, _ = paste_sprite(raw, sprite, mask, x, y)
        raw = drift_shift(drift_cast(drift_blur(raw, 0.7), db=2.0), 0.8, 0.5)
        edits.append(raw)
        instructions.append(f"Add a {color} {name}")
    result = run_chain(astronaut, edits, instructions)
    assert len(result.steps) == 3
    raw_psnr = [s.metrics.raw_psnr for s in result.steps]
    peeled_psnr = [s.metrics.peeled_psnr for s in result.steps]
    assert raw_psnr[0] > raw_psnr[-1], "the raw chain degrades"
    assert all(p > r + 10 for p, r in zip(peeled_psnr, raw_psnr, strict=True))
    assert result.steps[-1].metrics.peeled_lpips < 0.2 * result.steps[-1].metrics.raw_lpips
    # All three sprites survive in the final peeled frame.
    final = result.steps[-1].output.peeled
    for _, color, x, y in spots:
        centre = final[y + 45, x + 45].astype(int)
        assert np.abs(centre - np.array(COLORS[color])).mean() < 60
