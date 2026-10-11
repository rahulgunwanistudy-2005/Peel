"""Where did the edit differ from the original, and by how much?

A uniform tint is measured and removed first (the "global cast"), so it is reported once in the
ledger instead of lighting up every pixel. Local change is then a fusion of three views:
CIEDE2000 (colour), gradient-magnitude similarity (structure) and spatial LPIPS (perception).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage.color import deltaE_ciede2000
from skimage.morphology import reconstruction

from peel import models
from peel.arrays import F32, I32, U8, Bool, disk, rgb_to_lab, robust_stats, to_float

F64 = npt.NDArray[np.float64]

W_COLOR, W_STRUCTURE, W_LPIPS = 0.45, 0.20, 0.35
DELTA_E_FULL = 12.0  # a colour difference this large counts as fully changed
LPIPS_FULL = 0.55
THRESHOLD_FLOOR = 0.07
THRESHOLD_CEIL = 0.60  # high: a frame that was re-synthesized everywhere has a high noise floor
CAST_DETECT_DELTA_E = 1.0
LPIPS_SIDE = 512


@dataclass
class CastFit:
    """Affine Lab transform that maps the edit back onto the original's colour."""

    matrix: F64  # 3x3
    offset: F64  # 3
    shift: tuple[float, float, float]  # mean (dL, da, db) of edit minus original on stable pixels
    detected: bool

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.shift))

    def apply(self, lab: F32) -> F32:
        """Full affine correction. Only valid on colours like the ones it was fitted on."""
        flat = lab.reshape(-1, 3).astype(np.float64)
        out = flat @ self.matrix.T + self.offset
        return out.reshape(lab.shape).astype(np.float32)

    def remove_shift(self, lab: F32) -> F32:
        """Undo only the mean shift. Safe on new content (a red apple on a beige desk), where
        extrapolating the affine fit would bend saturated colours."""
        return np.asarray(lab - np.array(self.shift, dtype=np.float32), dtype=np.float32)


IDENTITY_CAST = CastFit(np.eye(3), np.zeros(3), (0.0, 0.0, 0.0), False)


@dataclass
class ChangeResult:
    score: F32  # fused change score, 0..1
    mask: Bool
    labels: I32  # connected components of ``mask``; 0 is background
    n_labels: int
    delta_e: F32  # CIEDE2000 after cast correction
    delta_e_raw: F32  # CIEDE2000 before cast correction (what the viewer actually sees)
    lpips: F32
    structure: F32  # 1 - gradient magnitude similarity
    delta_low: F32  # Lab distance of the low-pass images: did the content change, or only detail?
    cast: CastFit
    threshold: float
    noise_floor: float


def fit_cast(orig_lab: F32, edit_lab: F32, valid: Bool, *, iterations: int = 6) -> CastFit:
    """Robust (Huber IRLS) least squares for ``orig ≈ M · edit + t`` on stable pixels.

    A ridge term pulls M toward identity so near-monochrome images stay well conditioned.
    """
    h, w = valid.shape
    stride = max(1, int(np.sqrt(h * w / 40_000)))
    sub = valid[::stride, ::stride].ravel()
    x = edit_lab[::stride, ::stride].reshape(-1, 3)[sub].astype(np.float64)
    y = orig_lab[::stride, ::stride].reshape(-1, 3)[sub].astype(np.float64)
    if len(x) < 200:
        return IDENTITY_CAST
    design = np.hstack([x, np.ones((len(x), 1))])
    prior = np.vstack([np.eye(3), np.zeros((1, 3))])
    ridge = np.diag([1.0, 1.0, 1.0, 0.0]) * (1e-3 * len(x))
    weights: F64 = np.ones(len(x))
    params = prior.copy()
    for _ in range(iterations):
        dw = design * weights[:, None]
        params = np.asarray(
            np.linalg.solve(design.T @ dw + ridge, dw.T @ y + ridge @ prior), dtype=np.float64
        )
        residual = np.linalg.norm(design @ params - y, axis=1)
        scale = max(float(np.median(residual)) * 1.4826, 0.5)
        cutoff = 1.5 * scale
        weights = np.where(residual <= cutoff, 1.0, cutoff / np.maximum(residual, 1e-9))
        weights = np.where(residual > 4.0 * scale, 0.0, weights)  # hard reject: edited pixels
    stable = weights > 0.5
    if stable.sum() < 100:
        return IDENTITY_CAST
    shift = (x[stable] - y[stable]).mean(axis=0)
    detected = bool(np.linalg.norm(shift) >= CAST_DETECT_DELTA_E)
    return CastFit(
        matrix=params[:3].T.copy(),
        offset=params[3].copy(),
        shift=(float(shift[0]), float(shift[1]), float(shift[2])),
        detected=detected,
    )


def _gradient_dissimilarity(orig_l: F32, edit_l: F32) -> F32:
    """1 - GMS. Zero where edges agree, toward one where edges appear, vanish or move."""

    def magnitude(x: F32) -> F32:
        blurred = cv2.GaussianBlur(x, (0, 0), 1.0)
        return np.hypot(cv2.Sobel(blurred, cv2.CV_32F, 1, 0), cv2.Sobel(blurred, cv2.CV_32F, 0, 1))

    ga, gb = magnitude(orig_l), magnitude(edit_l)
    c = 60.0  # stabilizer on the L* (0..100) scale; keeps flat regions from reading as "changed"
    gms = (2.0 * ga * gb + c) / (ga * ga + gb * gb + c)
    return np.asarray(1.0 - gms, dtype=np.float32)


def _spatial_lpips(original: U8, edited: U8) -> F32:
    h, w = original.shape[:2]
    scale = min(1.0, LPIPS_SIDE / max(h, w))
    size = (max(64, round(w * scale)), max(64, round(h * scale)))
    a = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(edited, size, interpolation=cv2.INTER_AREA)
    dist = models.lpips_model().spatial(a, b)
    return cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)


def _clean_mask(raw: Bool, min_area: int) -> Bool:
    """Close small gaps, open by reconstruction (drops specks, keeps shapes), fill holes."""
    closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, disk(3))
    seed = cv2.erode(closed, disk(2))
    opened = reconstruction(seed, closed, method="dilation").astype(bool)
    filled = ndimage.binary_fill_holes(opened)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.asarray(keep[labels], dtype=bool)


def label_components(mask: Bool) -> tuple[I32, int]:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32), n - 1


def detect_change(
    original: U8,
    aligned: U8,
    valid: Bool,
    *,
    k_mad: float = 6.0,
    min_area_pct: float = 0.05,
    separate_cast: bool = True,
) -> ChangeResult:
    """All inputs at working resolution."""
    orig_lab = rgb_to_lab(to_float(original))
    edit_lab = rgb_to_lab(to_float(aligned))

    delta_raw = np.asarray(deltaE_ciede2000(orig_lab, edit_lab), dtype=np.float32)
    cast = fit_cast(orig_lab, edit_lab, valid) if separate_cast else IDENTITY_CAST
    if cast.detected:
        corrected = cast.apply(edit_lab)
        delta = np.asarray(deltaE_ciede2000(orig_lab, corrected), dtype=np.float32)
    else:
        corrected = edit_lab
        delta = delta_raw

    structure = _gradient_dissimilarity(orig_lab[..., 0], corrected[..., 0])
    sigma_low = max(3.0, 0.015 * max(original.shape[:2]))
    low = cv2.GaussianBlur(orig_lab, (0, 0), sigma_low) - cv2.GaussianBlur(
        corrected, (0, 0), sigma_low
    )
    delta_low = np.linalg.norm(low, axis=2).astype(np.float32)
    lpips_map = _spatial_lpips(original, aligned)

    # LPIPS comes from coarse feature maps and bleeds well past the pixels that changed, so it
    # only counts where there is pixel-level evidence (colour or edges) of a change as well.
    evidence = np.clip(np.maximum(delta / 4.0, structure / 0.15), 0.0, 1.0)
    evidence = cv2.GaussianBlur(evidence, (0, 0), 2.0)
    score = (
        W_COLOR * np.clip(delta / DELTA_E_FULL, 0.0, 1.0)
        + W_STRUCTURE * structure
        + W_LPIPS * np.clip(lpips_map / LPIPS_FULL, 0.0, 1.0) * evidence
    ).astype(np.float32)
    score = cv2.GaussianBlur(score, (0, 0), 1.5)
    score[~valid] = 0.0

    noise_floor, mad = robust_stats(score[valid]) if valid.any() else (0.0, 0.0)
    threshold = float(np.clip(noise_floor + k_mad * mad, THRESHOLD_FLOOR, THRESHOLD_CEIL))
    min_area = max(4, round(min_area_pct / 100.0 * score.size))
    mask = _clean_mask(score > threshold, min_area) & valid
    labels, n_labels = label_components(mask)

    return ChangeResult(
        score=score,
        mask=mask,
        labels=labels,
        n_labels=n_labels,
        delta_e=delta,
        delta_e_raw=delta_raw,
        lpips=lpips_map,
        structure=structure,
        delta_low=delta_low,
        cast=cast,
        threshold=threshold,
        noise_floor=noise_floor,
    )
