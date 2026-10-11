"""Turn change components into regions and decide: intended, collateral or ambiguous.

A component is intended when most of it lies where the instruction points, collateral when almost
none of it does. Faces are handled first: a face whose identity moved and that the instruction
did not target is always collateral, whatever the overlap says.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from peel.arrays import Bool, dilate
from peel.change import ChangeResult, label_components
from peel.faces import Face, face_lock_mask, instruction_names_faces
from peel.intent import IntentResult
from peel.types import (
    BBox,
    Classification,
    Confidence,
    Decision,
    PeelOptions,
    Region,
    RegionType,
)

TEXT_HEAT = 0.4
SPILL_RATIO = 2.5
SPILL_STRENGTH = 2.5
TARGET_SHARE = 0.3
SLIVER_PX = 7.0
FACE_TARGET_OVERLAP = 0.5
FACE_TARGET_OVERLAP_NAMED = 0.2


@dataclass
class RegionMask:
    region: Region
    mask: Bool  # working resolution


@dataclass(frozen=True)
class _Stats:
    area_pct: float
    delta_e: float
    lpips: float
    structure: float
    delta_low: float
    intent_overlap: float
    text_overlap: float
    thickness: float  # widest inscribed radius, in working pixels: slivers are thin


def _stats(mask: Bool, change: ChangeResult, intent: IntentResult, text_mask: Bool) -> _Stats:
    n = int(mask.sum())
    if n == 0:
        return _Stats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return _Stats(
        area_pct=100.0 * n / mask.size,
        delta_e=float(change.delta_e[mask].mean()),
        lpips=float(change.lpips[mask].mean()),
        structure=float(change.structure[mask].mean()),
        delta_low=float(change.delta_low[mask].mean()),
        intent_overlap=float(intent.mask[mask].mean()),
        text_overlap=float(text_mask[mask].mean()),
        thickness=float(cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3).max()),
    )


def _region_type(stats: _Stats, classification: Classification) -> RegionType:
    if classification != "intended" and stats.text_overlap > 0.35 and stats.structure > 0.08:
        return "text"
    if stats.thickness < SLIVER_PX and classification != "intended":
        return "structure"  # a thin sliver along an edge: the edge itself moved
    if stats.structure < 0.10 and stats.lpips < 0.15:
        return "color_cast"  # colour moved, edges did not
    if stats.delta_low >= 6.0 and (stats.structure >= 0.15 or stats.lpips >= 0.30):
        return "object"  # even the blurred image differs: content was added, removed or replaced
    if stats.structure >= 0.20 and stats.lpips >= 0.30:
        return "structure"  # same colours, edges redrawn
    return "texture"  # same content, fine detail re-rendered


def _outline(mask: Bool, scale_x: float, scale_y: float) -> tuple[BBox, list[tuple[int, int]]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    bbox = BBox(
        x=round(x0 * scale_x),
        y=round(y0 * scale_y),
        w=max(1, round((x1 - x0) * scale_x)),
        h=max(1, round((y1 - y0) * scale_y)),
    )
    largest = max(contours, key=cv2.contourArea)
    epsilon = max(0.75, 0.004 * cv2.arcLength(largest, True))
    approx = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)
    polygon = [(round(float(x) * scale_x), round(float(y) * scale_y)) for x, y in approx]
    return bbox, polygon


def _confidence(classification: Classification, overlap: float) -> Confidence:
    if classification == "ambiguous":
        return "low"
    if classification == "intended":
        return "high" if overlap >= 0.8 else "medium"
    return "high" if overlap <= 0.05 else "medium"


def _split_by_intent(
    component: Bool, intent: IntentResult, threshold: float, min_area: int
) -> tuple[Bool, Bool] | None:
    """Hysteresis split of a component that straddles the intent boundary.

    The intended part is the strong-intent core grown through pixels of weaker intent. Returns
    (intended part, remainder) or None when the split would leave a sliver.
    """
    core = component & intent.mask
    if not core.any():
        return None
    weak = component & (intent.mask | (intent.heat >= 0.5 * threshold))
    labels, n = label_components(weak)
    if n == 0:
        return None
    touching = np.unique(labels[core])
    intended = np.isin(labels, touching[touching > 0])
    remainder = component & ~intended
    if int(intended.sum()) < min_area or int(remainder.sum()) < min_area:
        return None
    if remainder.sum() < 0.15 * component.sum():
        return None  # nearly all of it is the target: not worth cutting a rim off
    return intended, remainder


def _remainder_class(remainder: Bool, intended: Bool, change: ChangeResult) -> Classification:
    """What is left of a component once the intended part is cut out.

    A strongly changed area of similar size attached to the edit is usually part of it (a cast
    shadow, a reflection): it is kept, flagged for review. A faint halo, a thin sliver along an
    edge, or something much larger than the edit is drift.
    """
    similar_size = remainder.sum() <= SPILL_RATIO * intended.sum()
    strong = float(change.score[remainder].mean()) >= SPILL_STRENGTH * change.threshold
    thick = (
        float(cv2.distanceTransform(remainder.astype(np.uint8), cv2.DIST_L2, 3).max()) >= SLIVER_PX
    )
    return "ambiguous" if (similar_size and strong and thick) else "collateral"


def _face_targeted(face: Face, intent: IntentResult, named: bool, shape: tuple[int, int]) -> bool:
    overlap = float(intent.mask[face_lock_mask(shape, face, expand=1.0)].mean())
    return overlap >= (FACE_TARGET_OVERLAP_NAMED if named else FACE_TARGET_OVERLAP)


def classify_regions(
    change: ChangeResult,
    intent: IntentResult,
    faces: list[Face],
    valid: Bool,
    options: PeelOptions,
    instruction: str,
    full_size: tuple[int, int],
) -> list[RegionMask]:
    shape = change.mask.shape
    scale_x = full_size[0] / shape[1]
    scale_y = full_size[1] / shape[0]
    min_area = max(4, round(options.min_area_pct / 100.0 * change.mask.size))
    text_mask = dilate(intent.text_heat >= TEXT_HEAT, 2)
    named_faces = instruction_names_faces(instruction)
    results: list[tuple[Bool, Classification, RegionType | None, float | None, _Stats]] = []

    # 1. Face identity locks.
    locked = np.zeros(shape, dtype=bool)
    if options.use_face_lock:
        for face in faces:
            if face.distance is None or face.distance <= options.face_distance_threshold:
                continue
            if _face_targeted(face, intent, named_faces, shape):
                continue
            mask = face_lock_mask(shape, face) & valid & ~locked
            if not mask.any():
                continue
            locked |= mask
            stats = _stats(mask, change, intent, text_mask)
            results.append((mask, "collateral", "face_identity", face.distance, stats))

    if intent.mode == "global":
        # 2a. The requested change is the whole frame; text is locked unless it was named.
        text_locked = np.zeros(shape, dtype=bool)
        if not intent.names_text:
            labels, n = label_components(text_mask & valid & ~locked)
            for i in range(1, n + 1):
                mask = labels == i
                if int(mask.sum()) < min_area:
                    continue
                stats = _stats(mask, change, intent, text_mask)
                if stats.structure < 0.12:
                    continue
                text_locked |= mask
                results.append((mask, "collateral", "text", None, stats))
        body = valid & ~locked & ~text_locked
        if body.any():
            stats = _stats(body, change, intent, text_mask)
            results.insert(0, (body, "intended", None, None, stats))
    else:
        # 2b. Local edit: classify each change component by overlap with the intent map.
        remaining = change.mask & ~locked
        labels, n = label_components(remaining)
        for i in range(1, n + 1):
            component = labels == i
            if int(component.sum()) < min_area:
                continue
            if not options.use_intent:
                stats = _stats(component, change, intent, text_mask)
                results.append((component, "intended", None, None, stats))
                continue
            overlap = float(intent.mask[component].mean())
            inside = int((intent.mask & component).sum())
            # A blob of drift can swallow the edit and push the overlap under the collateral
            # threshold. If a real share of the instruction's target lies in this component, it
            # is split rather than thrown away whole.
            holds_target = inside >= max(min_area, TARGET_SHARE * int(intent.mask.sum()))
            parts: list[tuple[Bool, Classification]]
            if overlap < options.collateral_overlap and not holds_target:
                parts = [(component, "collateral")]
            else:
                split = _split_by_intent(component, intent, options.intent_threshold, min_area)
                if split is None:
                    whole: Classification = (
                        "intended"
                        if overlap > options.intended_overlap
                        else (
                            "ambiguous" if overlap >= options.collateral_overlap else "collateral"
                        )
                    )
                    parts = [(component, whole)]
                else:
                    intended_part, remainder = split
                    parts = [
                        (intended_part, "intended"),
                        (remainder, _remainder_class(remainder, intended_part, change)),
                    ]
            for mask, classification in parts:
                stats = _stats(mask, change, intent, text_mask)
                results.append((mask, classification, None, None, stats))

        # 3. Grounding failed: do not throw the user's edit away, keep the strongest change.
        no_intended = not any(c == "intended" for _, c, _, _, _ in results)
        if options.use_intent and not intent.grounded and not intent.is_noop and no_intended:
            candidates = [
                (float(change.score[m].sum()), i)
                for i, (m, c, t, _, _) in enumerate(results)
                if c == "collateral" and t is None
            ]
            if candidates:
                _, index = max(candidates)
                mask, _, kind, dist, stats = results[index]
                results[index] = (mask, "ambiguous", kind, dist, stats)

    results.sort(key=lambda r: -r[4].area_pct)
    regions: list[RegionMask] = []
    for index, (mask, classification, kind, distance, stats) in enumerate(results, start=1):
        region_id = f"r{index}"
        bbox, polygon = _outline(mask, scale_x, scale_y)
        default: Decision = (
            "keep"
            if classification == "intended"
            or (classification == "ambiguous" and options.keep_ambiguous)
            else "revert"
        )  # same rule as ``default_decision``, before the Region exists
        decision = options.overrides.get(region_id, default)
        regions.append(
            RegionMask(
                region=Region(
                    id=region_id,
                    bbox=bbox,
                    polygon=polygon,
                    area_pct=round(stats.area_pct, 3),
                    type=kind or _region_type(stats, classification),
                    classification=classification,
                    decision=decision,
                    overridden=decision != default,
                    confidence=_confidence(classification, stats.intent_overlap),
                    intent_overlap=round(stats.intent_overlap, 3),
                    mean_delta_e=round(stats.delta_e, 2),
                    mean_lpips=round(stats.lpips, 3),
                    face_distance=None if distance is None else round(distance, 3),
                    sentence="",
                ),
                mask=mask,
            )
        )
    return regions


def default_decision(region: Region, keep_ambiguous: bool) -> Decision:
    if region.classification == "intended":
        return "keep"
    if region.classification == "ambiguous" and keep_ambiguous:
        return "keep"
    return "revert"
