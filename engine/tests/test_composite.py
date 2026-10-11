from __future__ import annotations

import numpy as np

from peel.arrays import U8
from peel.composite import composite, guided_filter


def test_guided_filter_preserves_a_constant() -> None:
    guide = np.random.default_rng(0).random((40, 50), dtype=np.float32)
    out = guided_filter(guide, np.full((40, 50), 0.5, dtype=np.float32), 4, 1e-3)
    assert np.allclose(out, 0.5, atol=1e-4)


def test_empty_mask_returns_the_base(scene: U8, other_scene: U8) -> None:
    mask = np.zeros(scene.shape[:2], dtype=bool)
    result = composite(scene, other_scene, mask, stable_work=~mask)
    assert np.array_equal(result.image, scene) and not result.support.any()


def test_outside_the_support_is_bit_identical(scene: U8, other_scene: U8) -> None:
    mask = np.zeros(scene.shape[:2], dtype=bool)
    mask[120:200, 180:300] = True
    result = composite(scene, other_scene, mask, stable_work=~mask)
    assert np.array_equal(result.image[~result.support], scene[~result.support])
    assert result.support.mean() < 0.2, "the support stays close to the mask"
    inner = (slice(135, 185), slice(195, 285))
    pasted = np.abs(result.image[inner].astype(int) - other_scene[inner].astype(int)).mean()
    assert pasted < 6.0, "inside the mask the pasted image comes through"


def test_poisson_blend_keeps_the_guarantee(scene: U8, other_scene: U8) -> None:
    mask = np.zeros(scene.shape[:2], dtype=bool)
    mask[120:200, 180:300] = True
    result = composite(scene, other_scene, mask, stable_work=~mask, blend="poisson")
    assert np.array_equal(result.image[~result.support], scene[~result.support])


def test_harmonization_closes_a_brightness_step() -> None:
    base = np.full((200, 200, 3), 120, dtype=np.uint8)
    top = np.full((200, 200, 3), 150, dtype=np.uint8)  # same content, uniformly brighter
    mask = np.zeros((200, 200), dtype=bool)
    mask[70:130, 70:130] = True
    on = composite(base, top, mask, stable_work=~mask, harmonize=True).image
    off = composite(base, top, mask, stable_work=~mask, harmonize=False).image
    edge = (slice(72, 76), slice(90, 110))
    assert abs(int(on[edge].mean()) - 120) < abs(int(off[edge].mean()) - 120)
