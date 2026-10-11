from __future__ import annotations

import numpy as np

from peel.arrays import U8
from peel.change import detect_change
from peel.synth import drift_cast, drift_patch_color


def _valid(img: U8) -> np.ndarray:
    return np.ones(img.shape[:2], dtype=bool)


def test_identical_images_have_no_change(scene: U8) -> None:
    result = detect_change(scene, scene.copy(), _valid(scene))
    assert not result.mask.any() and result.n_labels == 0
    assert not result.cast.detected


def test_uniform_tint_is_a_cast_not_regions(astronaut: U8) -> None:
    tinted = drift_cast(astronaut, db=5.0, da=1.5)
    result = detect_change(astronaut, tinted, _valid(astronaut))
    assert result.cast.detected
    dl, da, db = result.cast.shift
    assert abs(db - 5.0) < 1.0 and abs(da - 1.5) < 1.0 and abs(dl) < 1.0
    assert result.mask.mean() < 0.01, "a uniform tint must not light up regions"
    assert result.delta_e_raw.mean() > 3.0 > result.delta_e.mean()


def test_without_cast_separation_the_tint_reads_as_change(astronaut: U8) -> None:
    tinted = drift_cast(astronaut, db=9.0, da=4.0)
    off = detect_change(astronaut, tinted, _valid(astronaut), separate_cast=False)
    on = detect_change(astronaut, tinted, _valid(astronaut), separate_cast=True)
    assert off.score.mean() > 3 * on.score.mean()


def test_local_change_is_found_under_a_cast(astronaut: U8) -> None:
    box = (40, 300, 110, 90)  # on the orange suit, where a hue shift is visible
    edited = drift_cast(drift_patch_color(astronaut, box, hue_shift=40), db=4.0)
    result = detect_change(astronaut, edited, _valid(astronaut))
    truth = np.zeros(astronaut.shape[:2], dtype=bool)
    truth[box[1] : box[1] + box[3], box[0] : box[0] + box[2]] = True
    inter = (result.mask & truth).sum()
    assert inter / truth.sum() > 0.7, "most of the patch is detected"
    assert inter / max(1, result.mask.sum()) > 0.7, "little is detected outside it"
    assert result.cast.detected


def test_invalid_pixels_never_count(scene: U8) -> None:
    edited = scene.copy()
    edited[:50] = 0
    valid = _valid(scene)
    valid[:50] = False
    result = detect_change(scene, edited, valid)
    assert not result.mask[:50].any()
