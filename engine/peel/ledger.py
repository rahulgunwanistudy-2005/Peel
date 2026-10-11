"""The drift ledger: every change, located, typed, measured and said in plain words."""

from __future__ import annotations

import numpy as np

from peel.arrays import Bool
from peel.change import CastFit, ChangeResult
from peel.classify import RegionMask
from peel.intent import IntentResult
from peel.register import Registration
from peel.types import (
    Confidence,
    GlobalCast,
    Ledger,
    LedgerSummary,
    Region,
    RegistrationReport,
)

DELTA_E_VISIBLE = 1.0  # just-noticeable difference
DELTA_E_SATURATED = 10.0
LPIPS_VISIBLE = 0.05
LPIPS_SATURATED = 0.5


def describe_cast(cast: CastFit, *, requested: bool) -> GlobalCast:
    dl, da, db = cast.shift
    if not cast.detected:
        sentence = "No whole-image colour shift."
    else:
        parts: list[tuple[float, str]] = []
        if abs(db) >= 0.5:
            parts.append((abs(db), f"{db:+.1f} b* ({'yellow' if db > 0 else 'blue'})"))
        if abs(da) >= 0.5:
            parts.append((abs(da), f"{da:+.1f} a* ({'magenta' if da > 0 else 'green'})"))
        if abs(dl) >= 0.5:
            parts.append((abs(dl), f"{dl:+.1f} L* ({'lighter' if dl > 0 else 'darker'})"))
        parts.sort(reverse=True)
        dominant = max((abs(db), "b"), (abs(da), "a"), (abs(dl), "l"))[1]
        word = {
            "b": "warmer" if db > 0 else "cooler",
            "a": "pinker" if da > 0 else "greener",
            "l": "brighter" if dl > 0 else "darker",
        }[dominant]
        detail = ", ".join(text for _, text in parts) or f"ΔE {cast.magnitude:.1f}"
        suffix = " This matches the instruction, so it was kept." if requested else ""
        sentence = f"Whole image shifted {word}: {detail}.{suffix}"
    return GlobalCast(
        detected=cast.detected,
        delta_l=round(dl, 2),
        delta_a=round(da, 2),
        delta_b=round(db, 2),
        magnitude=round(cast.magnitude, 2),
        sentence=sentence,
    )


def describe_registration(reg: Registration) -> RegistrationReport:
    if reg.confidence == "low":
        sentence = (
            "Alignment is uncertain: the edit could not be matched to the original reliably. "
            "Check the result closely before using it."
        )
    elif reg.shift_px < 0.5 and not reg.flow_applied and not reg.resized:
        sentence = "The edit was already pixel-aligned with the original."
    else:
        bits = []
        if reg.resized:
            bits.append("came back at a different size")
        if reg.shift_px >= 0.5:
            bits.append(f"was shifted by up to {reg.shift_px:.1f} px")
        if reg.flow_applied:
            bits.append(f"was subtly warped (about {reg.residual_flow_px:.1f} px)")
        sentence = f"The edit {' and '.join(bits)}; Peel aligned it before comparing."
    return RegistrationReport(
        confidence=reg.confidence,
        inlier_ratio=round(reg.inlier_ratio, 3),
        reprojection_error_px=round(reg.reprojection_error_px, 3),
        n_matches=reg.n_matches,
        shift_px=round(reg.shift_px, 2),
        flow_applied=reg.flow_applied,
        residual_flow_px=round(reg.residual_flow_px, 2),
        resized=reg.resized,
        sentence=sentence,
    )


def _where(region: Region, width: int, height: int) -> str:
    if region.area_pct >= 40.0:
        return "across much of the frame"
    cx = (region.bbox.x + region.bbox.w / 2) / width
    cy = (region.bbox.y + region.bbox.h / 2) / height
    col = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
    row = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
    names = (
        ("in the top left", "at the top", "in the top right"),
        ("on the left", "in the centre", "on the right"),
        ("in the bottom left", "at the bottom", "in the bottom right"),
    )
    return names[row][col]


def describe_region(region: Region, width: int, height: int, mode: str) -> str:
    where = _where(region, width, height)
    area = f"{region.area_pct:.1f}% of the image"
    if region.type == "face_identity":
        distance = (
            ""
            if region.face_distance is None
            else f" (identity distance {region.face_distance:.2f})"
        )
        return f"The face {where} was re-rendered as a different-looking person{distance}."
    if region.classification == "intended":
        if mode == "global":
            return "The whole-image change you asked for."
        flavour = {
            "color_cast": "Recoloured as requested",
            "object": "New or replaced content, as requested",
            "structure": "Reshaped as requested",
            "texture": "Reworked as requested",
            "text": "Text changed as requested",
        }.get(region.type, "Changed as requested")
        return f"{flavour}, {where} ({area})."
    if region.classification == "ambiguous":
        return (
            f"A change {where} sits partly inside what you asked for ({area}). "
            "Kept unless you peel it."
        )
    unasked = {
        "color_cast": f"A patch {where} changed colour without being asked "
        f"(ΔE {region.mean_delta_e:.1f}).",
        "texture": f"Fine detail {where} was re-rendered without being asked "
        f"(LPIPS {region.mean_lpips:.2f}).",
        "structure": f"Shapes and edges {where} were redrawn without being asked "
        f"(LPIPS {region.mean_lpips:.2f}).",
        "object": f"Content {where} was replaced without being asked "
        f"(LPIPS {region.mean_lpips:.2f}).",
        "text": f"Text or fine print {where} was redrawn without being asked.",
    }
    return unasked[region.type]


def drift_score(change: ChangeResult, unrequested: Bool) -> float:
    """Share of the frame that changed without being asked, weighted by how visible it is.

    Each unrequested pixel contributes 0 (below the just-noticeable difference) to 1 (clearly
    different: ΔE2000 >= 10 or LPIPS >= 0.5). The score is the mean over all pixels, times 100.
    """
    if not unrequested.any():
        return 0.0
    colour = (change.delta_e_raw - DELTA_E_VISIBLE) / (DELTA_E_SATURATED - DELTA_E_VISIBLE)
    percept = (change.lpips - LPIPS_VISIBLE) / (LPIPS_SATURATED - LPIPS_VISIBLE)
    magnitude = np.clip(np.maximum(colour, percept), 0.0, 1.0)
    return float(100.0 * magnitude[unrequested].sum() / unrequested.size)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _headline(regions: list[Region], cast: GlobalCast, mode: str, is_noop: bool) -> str:
    kept = [r for r in regions if r.decision == "keep"]
    peeled = [r for r in regions if r.decision == "revert"]
    cast_drift = cast.detected and mode == "local"
    if not regions and not cast_drift:
        return "Nothing changed beyond pixel noise. The result is the original."
    extras = []
    if any(r.type == "face_identity" for r in peeled):
        extras.append("a re-rendered face")
    if any(r.type == "text" for r in peeled):
        extras.append("redrawn text")
    if cast_drift:
        extras.append("a whole-image colour shift")
    kept_part = (
        f"Kept {_plural(len(kept), 'requested change')}."
        if kept
        else ("Nothing was requested, so nothing was kept." if is_noop else "Nothing was kept.")
    )
    if not peeled and not cast_drift:
        return f"{kept_part} No unrequested changes found."
    n_unrequested = len(peeled)
    peeled_part = f"Peeled {_plural(n_unrequested, 'unrequested change')}" if peeled else "Peeled"
    if extras:
        joiner = ", including " if peeled else " "
        peeled_part += joiner + " and ".join(extras)
    return f"{kept_part} {peeled_part}."


def build_ledger(
    *,
    instruction: str,
    regions: list[RegionMask],
    change: ChangeResult,
    intent: IntentResult,
    registration: Registration,
    valid: Bool,
    full_size: tuple[int, int],
    bit_identical_pct: float,
    downscaled: bool,
) -> Ledger:
    width, height = full_size
    mode = intent.mode
    shape = change.mask.shape
    for rm in regions:
        rm.region.sentence = describe_region(rm.region, width, height, mode)

    kept = np.zeros(shape, dtype=bool)
    peeled = np.zeros(shape, dtype=bool)
    requested = np.zeros(shape, dtype=bool)
    for rm in regions:
        if rm.region.decision == "keep":
            kept |= rm.mask
        else:
            peeled |= rm.mask
        if rm.region.classification in ("intended", "ambiguous"):
            requested |= rm.mask
    unrequested = (peeled if mode == "global" else ~requested) & valid

    cast = describe_cast(change.cast, requested=mode == "global")
    reg = describe_registration(registration)
    items = [rm.region for rm in regions]
    outside = valid & ~kept
    summary = LedgerSummary(
        n_regions=len(items),
        n_intended=sum(r.classification == "intended" for r in items),
        n_collateral=sum(r.classification == "collateral" for r in items),
        n_ambiguous=sum(r.classification == "ambiguous" for r in items),
        changed_area_pct=round(100.0 * float(change.mask.mean()), 2),
        kept_area_pct=round(100.0 * float(kept.mean()), 2),
        peeled_area_pct=round(100.0 * float(peeled.mean()), 2),
        bit_identical_pct=round(bit_identical_pct, 3),
        background_delta_e=round(float(change.delta_e_raw[outside].mean()), 2)
        if outside.any()
        else 0.0,
    )

    warnings: list[str] = []
    grounding_failed = (
        mode == "local" and not intent.grounded and not intent.is_noop and bool(intent.phrases)
    )
    if registration.confidence == "low":
        warnings.append(reg.sentence)
    if grounding_failed:
        warnings.append(
            "Peel could not find what the instruction refers to in either image. The strongest "
            "change was kept for you to review; flip regions by hand if that is wrong."
        )
    if mode == "local" and not kept.any() and items and not intent.is_noop:
        warnings.append(
            "No change matched the instruction, so the result equals the original. "
            "Keep a region by hand if one of them is your edit."
        )
    if float(valid.mean()) < 0.99:
        pct = 100.0 * (1.0 - float(valid.mean()))
        warnings.append(
            f"The edit does not cover {pct:.0f}% of the original frame (it was cropped or "
            "shifted). Those pixels come from the original."
        )
    if downscaled:
        warnings.append("The original was larger than 4096 px and was downscaled first.")
    if mode == "global":
        warnings.append(
            "This is a whole-image instruction, so the edit is kept everywhere except locked "
            "faces and text. Nothing is bit-identical to the original in this mode."
        )

    confidence: Confidence = "high"
    if registration.confidence == "low" or grounding_failed:
        confidence = "low"
    elif registration.confidence == "medium" or summary.n_ambiguous > 0:
        confidence = "medium"

    return Ledger(
        instruction=instruction,
        target_phrases=intent.phrases,
        mode=mode,
        drift_score=round(drift_score(change, unrequested), 1),
        headline=_headline(items, cast, mode, intent.is_noop),
        confidence=confidence,
        summary=summary,
        global_cast=cast,
        registration=reg,
        regions=items,
        warnings=warnings,
    )
