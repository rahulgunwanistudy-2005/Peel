"""Orchestration: decode → register → change → intent → faces → classify → composite → ledger."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from peel import render
from peel.arrays import U8, Bool, resize_long_side, resize_mask
from peel.change import ChangeResult, detect_change
from peel.classify import RegionMask, classify_regions, default_decision
from peel.composite import composite
from peel.faces import Face, compare_faces
from peel.intent import IntentResult, compute_intent, empty_intent
from peel.io import decode_image, require_same_scene
from peel.ledger import build_ledger
from peel.register import Registration, register
from peel.types import (
    STAGES,
    Decision,
    ErrorCode,
    Ledger,
    PeelError,
    PeelOptions,
    StageTiming,
)

Progress = Callable[[str, float], None]


def _no_progress(stage: str, pct: float) -> None:
    return None


@dataclass
class Analysis:
    """Everything needed to recomposite with different region decisions, without re-analysing."""

    original: U8
    alpha: U8 | None
    aligned: U8
    valid_work: Bool
    change: ChangeResult
    intent: IntentResult
    faces: list[Face]
    regions: list[RegionMask]
    registration: Registration
    instruction: str
    options: PeelOptions
    downscaled: bool


@dataclass
class PeelOutput:
    peeled: U8
    ledger: Ledger
    analysis: Analysis
    timings: list[StageTiming] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return float(sum(t.ms for t in self.timings))

    @property
    def size(self) -> tuple[int, int]:
        return (self.peeled.shape[1], self.peeled.shape[0])

    def change_map(self) -> U8:
        a = self.analysis
        return render.change_map(a.original, a.regions, a.intent.mode)

    def diff(self) -> U8:
        return render.amplified_diff(self.peeled, self.analysis.original)

    def raw_diff(self) -> U8:
        return render.amplified_diff(self.analysis.aligned, self.analysis.original)


class _Clock:
    def __init__(self, progress: Progress) -> None:
        self.timings: list[StageTiming] = []
        self._progress = progress
        self._start = time.perf_counter()

    def begin(self, stage: str) -> None:
        self._progress(stage, STAGES.index(stage) / len(STAGES))
        self._start = time.perf_counter()

    def end(self, stage: str) -> None:
        ms = (time.perf_counter() - self._start) * 1000.0
        self.timings.append(StageTiming(stage=stage, ms=round(ms, 1)))


def content_hash(original: bytes, edited: bytes, instruction: str, options: PeelOptions) -> str:
    digest = hashlib.sha256()
    for part in (original, edited, instruction.encode(), options.model_dump_json().encode()):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _compose(analysis: Analysis) -> tuple[U8, float]:
    """Build the peeled image from the current region decisions."""
    a = analysis
    shape = a.change.mask.shape
    keep = np.zeros(shape, dtype=bool)
    revert = np.zeros(shape, dtype=bool)
    for rm in a.regions:
        if rm.region.decision == "keep":
            keep |= rm.mask
        else:
            revert |= rm.mask

    if a.intent.mode == "global":
        body_kept = any(
            rm.region.decision == "keep" and rm.region.classification == "intended"
            for rm in a.regions
        )
        if not body_kept:
            peeled = a.original.copy()
        else:
            restore = revert | ~a.valid_work
            peeled = composite(
                a.aligned,
                a.original,
                restore,
                stable_work=a.valid_work & ~restore,
                harmonize=a.options.harmonize,
                harmonize_falloff=False,
                blend=a.options.blend,
            ).image
    else:
        peeled = composite(
            a.original,
            a.aligned,
            keep & a.valid_work,
            stable_work=a.valid_work & ~a.change.mask,
            cast=a.change.cast,
            harmonize=a.options.harmonize,
            blend=a.options.blend,
        ).image
    identical = float(np.mean(np.all(peeled == a.original, axis=-1))) * 100.0
    return peeled, identical


def _ledger(analysis: Analysis, bit_identical_pct: float) -> Ledger:
    a = analysis
    return build_ledger(
        instruction=a.instruction,
        regions=a.regions,
        change=a.change,
        intent=a.intent,
        registration=a.registration,
        valid=a.valid_work,
        full_size=(a.original.shape[1], a.original.shape[0]),
        bit_identical_pct=bit_identical_pct,
        downscaled=a.downscaled,
    )


def peel_arrays(
    original: U8,
    edited: U8,
    instruction: str,
    options: PeelOptions | None = None,
    *,
    alpha: U8 | None = None,
    downscaled: bool = False,
    progress: Progress = _no_progress,
    clock: _Clock | None = None,
) -> PeelOutput:
    """Run the pipeline on decoded sRGB uint8 images."""
    options = options or PeelOptions()
    instruction = instruction.strip()
    if not instruction:
        raise PeelError(
            ErrorCode.EMPTY_INSTRUCTION,
            "Tell Peel what you asked the editor to do, for example “make the jacket red”.",
        )
    clock = clock or _Clock(progress)
    height, width = original.shape[:2]

    clock.begin("register")
    require_same_scene(original, edited)
    registration = register(
        original,
        edited,
        work_long_side=options.work_long_side,
        background_lock=options.use_reregistration,
        allow_flow=options.use_reregistration,
    )
    orig_work = resize_long_side(original, options.work_long_side)
    wh, ww = orig_work.shape[:2]
    aligned_work = cv2.resize(registration.aligned, (ww, wh), interpolation=cv2.INTER_AREA)
    valid_work = resize_mask(registration.valid, ww, wh)
    clock.end("register")

    clock.begin("change")
    change = detect_change(
        orig_work,
        aligned_work,
        valid_work,
        k_mad=options.k_mad,
        min_area_pct=options.min_area_pct,
        separate_cast=options.use_cast_separation,
    )
    clock.end("change")

    clock.begin("intent")
    if options.use_intent:
        intent = compute_intent(
            orig_work, aligned_work, instruction, threshold=options.intent_threshold
        )
    else:
        intent = empty_intent((wh, ww), instruction)
    clock.end("intent")

    clock.begin("faces")
    faces = (
        compare_faces(orig_work, aligned_work, original, registration.aligned)
        if options.use_face_lock
        else []
    )
    clock.end("faces")

    clock.begin("classify")
    regions = classify_regions(
        change, intent, faces, valid_work, options, instruction, (width, height)
    )
    clock.end("classify")

    analysis = Analysis(
        original=original,
        alpha=alpha,
        aligned=registration.aligned,
        valid_work=valid_work,
        change=change,
        intent=intent,
        faces=faces,
        regions=regions,
        registration=registration,
        instruction=instruction,
        options=options,
        downscaled=downscaled,
    )

    clock.begin("composite")
    peeled, identical = _compose(analysis)
    clock.end("composite")

    clock.begin("ledger")
    ledger = _ledger(analysis, identical)
    clock.end("ledger")
    progress("done", 1.0)
    return PeelOutput(peeled=peeled, ledger=ledger, analysis=analysis, timings=clock.timings)


def peel_bytes(
    original: bytes,
    edited: bytes,
    instruction: str,
    options: PeelOptions | None = None,
    *,
    progress: Progress = _no_progress,
) -> PeelOutput:
    """Run the pipeline on encoded image files."""
    options = options or PeelOptions()
    clock = _Clock(progress)
    clock.begin("decode")
    orig = decode_image(original, max_long_side=options.max_long_side, label="original image")
    edit = decode_image(edited, max_long_side=options.max_long_side, label="edited image")
    clock.end("decode")
    return peel_arrays(
        orig.rgb,
        edit.rgb,
        instruction,
        options,
        alpha=orig.alpha,
        downscaled=orig.downscaled,
        progress=progress,
        clock=clock,
    )


def recomposite(output: PeelOutput, overrides: dict[str, Decision]) -> PeelOutput:
    """Apply per-region keep/revert decisions and rebuild the image and ledger."""
    analysis = output.analysis
    known = {rm.region.id for rm in analysis.regions}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise PeelError(
            ErrorCode.INVALID_OPTIONS, f"Unknown region id: {', '.join(unknown)}.", status=422
        )
    start = time.perf_counter()
    for rm in analysis.regions:
        default = default_decision(rm.region, analysis.options.keep_ambiguous)
        decision = overrides.get(rm.region.id, default)
        rm.region.decision = decision
        rm.region.overridden = decision != default
    peeled, identical = _compose(analysis)
    ledger = _ledger(analysis, identical)
    ms = (time.perf_counter() - start) * 1000.0
    timings = [StageTiming(stage="composite", ms=round(ms, 1))]
    return PeelOutput(peeled=peeled, ledger=ledger, analysis=analysis, timings=timings)
