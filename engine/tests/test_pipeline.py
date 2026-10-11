from __future__ import annotations

import cv2
import numpy as np
import pytest

from peel.arrays import U8, dilate
from peel.pipeline import PeelOutput, content_hash, peel_arrays, peel_bytes, recomposite
from peel.synth import SynthCase, drift_cast, make_case
from peel.types import ErrorCode, PeelError, PeelOptions

from .conftest import png_bytes


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def kept_mask(output: PeelOutput) -> np.ndarray:
    h, w = output.peeled.shape[:2]
    mask = np.zeros(output.analysis.change.mask.shape, dtype=bool)
    for rm in output.analysis.regions:
        if rm.region.decision == "keep":
            mask |= rm.mask
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


@pytest.fixture(scope="module")
def add_output(add_case: SynthCase) -> PeelOutput:
    return peel_arrays(add_case.original, add_case.edited, add_case.instruction)


def test_named_pasted_object_is_classified_intended(
    add_case: SynthCase, add_output: PeelOutput
) -> None:
    intended = [r for r in add_output.ledger.regions if r.classification == "intended"]
    assert len(intended) == 1 and intended[0].decision == "keep"
    assert iou(kept_mask(add_output), add_case.intended) > 0.6


def test_collateral_drift_is_peeled(add_case: SynthCase, add_output: PeelOutput) -> None:
    ledger = add_output.ledger
    assert ledger.summary.n_collateral >= 1
    assert ledger.global_cast.detected
    assert ledger.registration.shift_px > 0.5
    assert ledger.drift_score > 5.0
    # The result is far closer to the ideal (edit only) than the raw edit is.
    ideal = add_case.ideal.astype(np.float32)
    raw_error = np.abs(add_case.edited.astype(np.float32) - ideal).mean()
    peeled_error = np.abs(add_output.peeled.astype(np.float32) - ideal).mean()
    assert peeled_error < 0.25 * raw_error


def test_pixels_outside_the_final_mask_are_bit_identical(
    add_case: SynthCase, add_output: PeelOutput
) -> None:
    changed = np.any(add_output.peeled != add_case.original, axis=-1)
    allowed = dilate(kept_mask(add_output), 24)
    assert not (changed & ~allowed).any(), "a pixel outside the kept edit differs from the original"
    assert add_output.ledger.summary.bit_identical_pct > 85.0
    # And the ledger's number is the measured one.
    measured = 100.0 * float(np.mean(np.all(add_output.peeled == add_case.original, axis=-1)))
    assert add_output.ledger.summary.bit_identical_pct == pytest.approx(measured, abs=0.01)


def test_every_region_has_a_plain_sentence_and_geometry(add_output: PeelOutput) -> None:
    h, w = add_output.peeled.shape[:2]
    for region in add_output.ledger.regions:
        assert region.sentence.endswith(".") and len(region.sentence) > 20
        assert len(region.polygon) >= 3
        assert 0 <= region.bbox.x < w and 0 <= region.bbox.y < h
    stages = [t.stage for t in add_output.timings]
    assert stages == ["register", "change", "intent", "faces", "classify", "composite", "ledger"]


def test_pipeline_is_deterministic(add_case: SynthCase, add_output: PeelOutput) -> None:
    again = peel_arrays(add_case.original, add_case.edited, add_case.instruction)
    assert np.array_equal(again.peeled, add_output.peeled)
    assert again.ledger.model_dump() == add_output.ledger.model_dump()


def test_region_override_recomposites_without_reanalysis(add_case: SynthCase) -> None:
    output = peel_arrays(add_case.original, add_case.edited, add_case.instruction)
    intended = next(r for r in output.ledger.regions if r.classification == "intended")
    flipped = recomposite(output, {intended.id: "revert"})
    assert np.array_equal(flipped.peeled, add_case.original), (
        "reverting the only edit gives the original"
    )
    region = next(r for r in flipped.ledger.regions if r.id == intended.id)
    assert region.decision == "revert" and region.overridden
    assert flipped.ledger.summary.bit_identical_pct == 100.0
    restored = recomposite(flipped, {})
    assert not np.array_equal(restored.peeled, add_case.original)
    with pytest.raises(PeelError) as err:
        recomposite(output, {"r999": "keep"})
    assert err.value.code == ErrorCode.INVALID_OPTIONS


@pytest.mark.parametrize("kind", ["remove", "recolor", "replace"])
def test_other_edit_kinds_keep_the_target(scene: U8, kind: str) -> None:
    case = make_case(scene, seed=11, kind=kind, drifts=["cast", "jpeg"])  # type: ignore[arg-type]
    output = peel_arrays(case.original, case.edited, case.instruction)
    assert iou(kept_mask(output), case.intended) > 0.5, output.ledger.headline
    ideal = case.ideal.astype(np.float32)
    raw_error = np.abs(case.edited.astype(np.float32) - ideal).mean()
    assert np.abs(output.peeled.astype(np.float32) - ideal).mean() < 0.5 * raw_error


def test_not_same_scene(scene: U8, astronaut: U8) -> None:
    with pytest.raises(PeelError) as err:
        peel_bytes(png_bytes(scene), png_bytes(astronaut), "add a hat")
    assert err.value.code == ErrorCode.NOT_SAME_SCENE and err.value.status == 422


def test_empty_instruction(scene: U8) -> None:
    with pytest.raises(PeelError) as err:
        peel_arrays(scene, scene, "   ")
    assert err.value.code == ErrorCode.EMPTY_INSTRUCTION


def test_replica_instruction_restores_the_original(astronaut: U8) -> None:
    drifted = drift_cast(cv2.GaussianBlur(astronaut, (0, 0), 1.0), db=4.0)
    output = peel_arrays(
        astronaut, drifted, "Produce an exact replica of the provided image, with no alterations."
    )
    assert np.array_equal(output.peeled, astronaut)
    assert output.ledger.summary.bit_identical_pct == 100.0
    assert output.ledger.global_cast.detected and output.ledger.drift_score > 0


def test_face_identity_lock(astronaut: U8) -> None:
    face_box = (160, 50, 150, 170)
    case = make_case(astronaut, seed=5, kind="add", drifts=["patch_warp"], face_box=face_box)
    output = peel_arrays(case.original, case.edited, case.instruction)
    faces = [r for r in output.ledger.regions if r.type == "face_identity"]
    assert len(faces) == 1, [r.sentence for r in output.ledger.regions]
    assert faces[0].decision == "revert" and faces[0].face_distance is not None
    assert "face" in faces[0].sentence and "re-rendered face" in output.ledger.headline
    x, y, w, h = face_box
    crop = (slice(y + 30, y + h - 30), slice(x + 30, x + w - 30))
    assert np.array_equal(output.peeled[crop], astronaut[crop]), (
        "the face is the original face again"
    )
    # Without the lock the warped face can only be caught as a generic change.
    unlocked = peel_arrays(
        case.original, case.edited, case.instruction, PeelOptions(use_face_lock=False)
    )
    assert not [r for r in unlocked.ledger.regions if r.type == "face_identity"]


def test_untouched_face_is_not_flagged(astronaut: U8) -> None:
    case = make_case(astronaut, seed=5, kind="add", drifts=["jpeg"], face_box=(160, 50, 150, 170))
    output = peel_arrays(case.original, case.edited, case.instruction)
    assert not [r for r in output.ledger.regions if r.type == "face_identity"]


def test_global_instruction_keeps_the_edit_everywhere(astronaut: U8) -> None:
    night = (astronaut.astype(np.float32) * np.array([0.45, 0.5, 0.75])).astype(np.uint8)
    output = peel_arrays(astronaut, night, "make it night")
    assert output.ledger.mode == "global"
    assert output.ledger.summary.kept_area_pct > 90.0
    assert np.abs(output.peeled.astype(int) - night.astype(int)).mean() < 3.0
    assert any("whole-image" in w for w in output.ledger.warnings)


def test_ablation_switches_change_behaviour(add_case: SynthCase, add_output: PeelOutput) -> None:
    no_intent = peel_arrays(
        add_case.original, add_case.edited, add_case.instruction, PeelOptions(use_intent=False)
    )
    assert all(r.decision == "keep" for r in no_intent.ledger.regions)
    assert no_intent.ledger.summary.kept_area_pct > add_output.ledger.summary.kept_area_pct


def test_content_hash_depends_on_every_input() -> None:
    base = content_hash(b"a", b"b", "x", PeelOptions())
    assert base == content_hash(b"a", b"b", "x", PeelOptions())
    assert (
        len(
            {
                base,
                content_hash(b"a", b"c", "x", PeelOptions()),
                content_hash(b"a", b"b", "y", PeelOptions()),
                content_hash(b"a", b"b", "x", PeelOptions(k_mad=5.0)),
                content_hash(b"ab", b"", "x", PeelOptions()),
            }
        )
        == 5
    )


def test_identical_pair_is_reported_calmly(astronaut: U8) -> None:
    output = peel_arrays(astronaut, astronaut.copy(), "Add a yellow balloon")
    assert np.array_equal(output.peeled, astronaut)
    assert output.ledger.regions == [] and output.ledger.warnings == []
    assert output.ledger.confidence == "high" and output.ledger.drift_score == 0.0
    assert output.ledger.headline.startswith("Nothing changed")


def test_downscale_is_reported_with_the_actual_limit(astronaut: U8) -> None:
    from .conftest import png_bytes

    output = peel_bytes(
        png_bytes(astronaut), png_bytes(astronaut), "Add a hat", PeelOptions(max_long_side=256)
    )
    assert output.peeled.shape[:2] == (256, 256)
    assert any("larger than 256 px" in w for w in output.ledger.warnings)
