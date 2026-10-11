"""Align the AI edit to the original.

Editors return frames that are resized, shifted by a pixel or two, or subtly warped. Compositing
without undoing that produces ghost edges, so registration is the first real stage:

  1. SIFT on CLAHE-equalized luminance, ratio test, MAGSAC homography.
  2. Background lock: refit using only matches outside the first-pass change mask, so the edit
     itself (which legitimately moved pixels) cannot drag the alignment.
  3. DIS optical flow for the residual non-rigid shift, applied only if it measurably helps.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from peel.arrays import F32, U8, Bool, dilate, luminance_u8, resize_long_side, robust_stats
from peel.io import aspect_matches
from peel.types import Confidence

F64 = npt.NDArray[np.float64]

IDENTITY_TOLERANCE_PX = 0.35
MIN_MATCHES_HOMOGRAPHY = 12
MIN_MATCHES_SIMILARITY = 6


@dataclass
class Registration:
    aligned: U8  # the edit, warped into the original's frame at full resolution
    valid: Bool  # pixels of the original frame that the edit actually covers
    homography: F64  # full-resolution edit -> original
    confidence: Confidence
    inlier_ratio: float
    reprojection_error_px: float
    n_matches: int
    shift_px: float
    flow_applied: bool
    residual_flow_px: float
    resized: bool
    background_locked: bool


@dataclass
class _Fit:
    h: F64
    inlier_ratio: float
    error: float
    n_matches: int
    src: F32
    dst: F32
    inliers: Bool


def _equalized(gray: U8) -> U8:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _match(orig_gray: U8, edit_gray: U8) -> tuple[F32, F32]:
    """Ratio-tested SIFT correspondences as (edit points, original points)."""
    sift = cv2.SIFT.create(nfeatures=4000, contrastThreshold=0.03)
    ko, do = sift.detectAndCompute(_equalized(orig_gray), None)
    ke, de = sift.detectAndCompute(_equalized(edit_gray), None)
    empty = np.zeros((0, 2), dtype=np.float32)
    if do is None or de is None or len(ko) < 4 or len(ke) < 4:
        return empty, empty
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(de, do, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if not good:
        return empty, empty
    src = np.array([ke[m.queryIdx].pt for m in good], dtype=np.float32)
    dst = np.array([ko[m.trainIdx].pt for m in good], dtype=np.float32)
    return src, dst


def _fit(src: F32, dst: F32) -> _Fit | None:
    """Homography when well supported, similarity when matches are scarce."""
    n = len(src)
    if n < MIN_MATCHES_SIMILARITY:
        return None
    h: F64 | None = None
    mask = None
    if n >= MIN_MATCHES_HOMOGRAPHY:
        h, mask = cv2.findHomography(
            src, dst, cv2.USAC_MAGSAC, 2.0, maxIters=4000, confidence=0.999
        )
    if h is None or mask is None or int(mask.sum()) < MIN_MATCHES_HOMOGRAPHY:
        affine, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        if affine is None or mask is None or int(mask.sum()) < MIN_MATCHES_SIMILARITY:
            return None
        h = np.vstack([affine, [0.0, 0.0, 1.0]])
    inliers = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src[inliers][None].astype(np.float64), h)[0]
    error = float(np.mean(np.linalg.norm(projected - dst[inliers], axis=1)))
    return _Fit(
        h=np.asarray(h, dtype=np.float64),
        inlier_ratio=float(inliers.mean()),
        error=error,
        n_matches=n,
        src=src,
        dst=dst,
        inliers=inliers,
    )


def _corner_shift(h: F64, width: int, height: int) -> float:
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    moved = cv2.perspectiveTransform(corners[None], h)[0]
    return float(np.max(np.linalg.norm(moved - corners, axis=1)))


def _plausible(h: F64, width: int, height: int) -> bool:
    """Reject degenerate fits (folded, collapsed or wildly scaled frames)."""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    moved = cv2.perspectiveTransform(corners[None], h)[0].astype(np.float32)
    if not cv2.isContourConvex(moved):
        return False
    area = abs(float(cv2.contourArea(moved)))
    return 0.25 * width * height < area < 4.0 * width * height


def _rough_change(orig_gray: U8, aligned_gray: U8) -> Bool:
    a = cv2.GaussianBlur(orig_gray, (0, 0), 2.0).astype(np.float32)
    b = cv2.GaussianBlur(aligned_gray, (0, 0), 2.0).astype(np.float32)
    diff = np.abs(a - b)
    med, mad = robust_stats(diff)
    return np.asarray(diff > max(med + 4.0 * mad, 12.0), dtype=bool)


def _phase_shift(orig_gray: U8, edit_gray: U8) -> tuple[F64, float]:
    """Translation-only fallback for scenes too plain for keypoints."""
    window = cv2.createHanningWindow((orig_gray.shape[1], orig_gray.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(
        edit_gray.astype(np.float32), orig_gray.astype(np.float32), window
    )
    h = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return h, float(response)


def _background_flow(
    orig_gray: U8, aligned_gray: U8, changed: Bool, valid: Bool
) -> tuple[F32, float] | None:
    """Smooth residual flow measured on the background. Returns (flow, median magnitude)."""
    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(orig_gray, aligned_gray, None)
    background = valid & ~dilate(changed, 6)
    if background.mean() < 0.15:
        return None
    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    median = float(np.median(magnitude[background]))
    if median <= 0.5:
        return None
    # Flow inside the edit is meaningless; fill it from the surrounding background.
    weight = background.astype(np.float32)
    sigma = max(4.0, 0.02 * max(orig_gray.shape))
    blurred_w = cv2.GaussianBlur(weight, (0, 0), sigma) + 1e-4
    smooth = np.dstack(
        [cv2.GaussianBlur(flow[..., c] * weight, (0, 0), sigma) / blurred_w for c in range(2)]
    ).astype(np.float32)
    return smooth, median


def _detail_residual(a: U8, b: U8, mask: Bool) -> float:
    """Mean difference of the high-pass images. Blind to a brightness or colour offset, which a
    plain difference would let a spurious shift along a gradient "explain"."""

    def high_pass(x: U8) -> F32:
        f = x.astype(np.float32)
        return np.asarray(f - cv2.GaussianBlur(f, (0, 0), 4.0), dtype=np.float32)

    return float(np.mean(np.abs(high_pass(a) - high_pass(b))[mask]))


def _ecc_refine(orig_gray: U8, edit_gray: U8, h_work: F64, *, trusted: bool) -> F64 | None:
    """Sub-pixel affine refinement by direct intensity alignment (ECC), on the background only.

    Keypoints need texture; smooth scenes (sky, sand, studio walls) have little, and that is where
    a one-pixel error shows as a ghost edge. ECC works on gradients of the whole frame instead.
    Returns the refined homography, or None when it does not measurably improve the alignment.
    """
    height, width = orig_gray.shape
    aligned = cv2.warpPerspective(
        edit_gray, h_work, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    covered = (
        cv2.warpPerspective(np.full(edit_gray.shape, 255, np.uint8), h_work, (width, height)) > 0
    )
    background = covered & ~dilate(_rough_change(orig_gray, aligned), 8)
    if background.mean() < 0.2:
        return None
    half = (max(32, width // 2), max(32, height // 2))
    template = cv2.resize(orig_gray, half, interpolation=cv2.INTER_AREA)
    moving = cv2.resize(aligned, half, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(background.astype(np.uint8) * 255, half, interpolation=cv2.INTER_NEAREST)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6)
    try:
        _, warp = cv2.findTransformECC(template, moving, warp, cv2.MOTION_AFFINE, criteria, mask, 5)
    except cv2.error:
        return None
    # ``warp`` maps original (half-res) coordinates to aligned-edit coordinates.
    to_half = _scale_matrix(half[0] / width, half[1] / height)
    residual = (
        np.linalg.inv(to_half) @ np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]]) @ to_half
    )
    candidate = np.linalg.inv(residual) @ h_work
    realigned = cv2.warpPerspective(
        edit_gray, candidate, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    before = _detail_residual(orig_gray, aligned, background)
    after = _detail_residual(orig_gray, realigned, background)
    needed = 0.97 if trusted else 0.9
    return candidate if after < needed * before else None


def _beats_identity(orig_gray: U8, edit_gray: U8, h_work: F64) -> bool:
    """True when warping by ``h_work`` leaves a smaller background residual than not warping."""
    if np.allclose(h_work, np.eye(3), atol=1e-9):
        return False
    height, width = orig_gray.shape
    aligned = cv2.warpPerspective(
        edit_gray, h_work, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    covered = (
        cv2.warpPerspective(np.full(edit_gray.shape, 255, np.uint8), h_work, (width, height)) > 0
    )
    changed = dilate(_rough_change(orig_gray, aligned) | _rough_change(orig_gray, edit_gray), 6)
    background = covered & ~changed
    if background.mean() < 0.1:
        background = covered
    moved = _detail_residual(orig_gray, aligned, background)
    still = _detail_residual(orig_gray, edit_gray, background)
    return moved < 0.98 * still


def _warp(
    edited: U8,
    h_full: F64,
    size: tuple[int, int],
    flow_full: F32 | None,
) -> tuple[U8, Bool]:
    """One resampling pass from the edit's native pixels into the original frame."""
    width, height = size
    ones = np.full(edited.shape[:2], 255, dtype=np.uint8)
    if flow_full is None:
        aligned = cv2.warpPerspective(
            edited, h_full, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        cover = cv2.warpPerspective(ones, h_full, size, flags=cv2.INTER_NEAREST)
        return aligned, cover > 0
    inv = np.linalg.inv(h_full)
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    px = xs + flow_full[..., 0]
    py = ys + flow_full[..., 1]
    denom = inv[2, 0] * px + inv[2, 1] * py + inv[2, 2]
    map_x = ((inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]) / denom).astype(np.float32)
    map_y = ((inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]) / denom).astype(np.float32)
    aligned = cv2.remap(edited, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cover = cv2.remap(ones, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    return aligned, cover > 0


def _scale_matrix(sx: float, sy: float) -> F64:
    """Pixel-centre-correct scaling, the same convention ``cv2.resize`` uses."""
    return np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])


def _confidence(inlier_ratio: float, error: float, keypoint_based: bool) -> Confidence:
    if not keypoint_based:
        return "medium"
    if inlier_ratio < 0.25 or error > 3.0:
        return "low"
    if inlier_ratio < 0.5 or error > 1.5:
        return "medium"
    return "high"


def register(
    original: U8,
    edited: U8,
    *,
    work_long_side: int = 768,
    background_lock: bool = True,
    allow_flow: bool = True,
) -> Registration:
    height, width = original.shape[:2]
    same_aspect = aspect_matches(original, edited)
    resized = edited.shape[:2] != (height, width)
    if max(edited.shape[:2]) > 1.25 * max(height, width):
        # A larger edit is area-averaged down first; warping straight from it would alias.
        if same_aspect:
            edited = cv2.resize(edited, (width, height), interpolation=cv2.INTER_AREA)
        else:
            edited = resize_long_side(edited, max(height, width))
    eh, ew = edited.shape[:2]

    orig_work = resize_long_side(original, work_long_side)
    wh, ww = orig_work.shape[:2]
    if same_aspect:
        edit_work = cv2.resize(edited, (ww, wh), interpolation=cv2.INTER_AREA)
    else:
        edit_work = resize_long_side(edited, work_long_side)
    orig_gray = luminance_u8(orig_work)
    edit_gray = luminance_u8(edit_work)

    # Scale matrices between full resolution and the working resolution.
    to_work_orig = _scale_matrix(ww / width, wh / height)
    to_work_edit = _scale_matrix(edit_work.shape[1] / ew, edit_work.shape[0] / eh)
    from_work_orig = np.linalg.inv(to_work_orig)

    src, dst = _match(orig_gray, edit_gray)
    fit = _fit(src, dst)
    if fit is not None and not _plausible(fit.h, edit_work.shape[1], edit_work.shape[0]):
        fit = None

    keypoint_based = fit is not None
    locked = False
    if fit is not None:
        h_work = fit.h
        if background_lock:
            aligned_gray = cv2.warpPerspective(edit_gray, h_work, (ww, wh))
            changed = dilate(_rough_change(orig_gray, aligned_gray), 10)
            xs = np.clip(np.rint(fit.dst[:, 0]).astype(int), 0, ww - 1)
            ys = np.clip(np.rint(fit.dst[:, 1]).astype(int), 0, wh - 1)
            keep = fit.inliers & ~changed[ys, xs]
            if MIN_MATCHES_HOMOGRAPHY <= int(keep.sum()) < int(fit.inliers.sum()):
                refit = _fit(fit.src[keep], fit.dst[keep])
                if (
                    refit is not None
                    and refit.error <= fit.error + 0.25
                    and _plausible(refit.h, edit_work.shape[1], edit_work.shape[0])
                ):
                    h_work = refit.h
                    locked = True
                    fit = _Fit(
                        h=refit.h,
                        inlier_ratio=fit.inlier_ratio,
                        error=refit.error,
                        n_matches=fit.n_matches,
                        src=fit.src,
                        dst=fit.dst,
                        inliers=fit.inliers,
                    )
        inlier_ratio, error, n_matches = fit.inlier_ratio, fit.error, fit.n_matches
    elif same_aspect:
        h_work, response = _phase_shift(orig_gray, edit_gray)
        if response < 0.3 or _corner_shift(h_work, ww, wh) > 0.05 * max(ww, wh):
            h_work = np.eye(3)
        inlier_ratio, error, n_matches = 0.0, 0.0, len(src)
    else:
        # Different aspect and nothing to match on: centre-fit and say so.
        scale = min(ww / edit_work.shape[1], wh / edit_work.shape[0])
        tx = (ww - edit_work.shape[1] * scale) / 2
        ty = (wh - edit_work.shape[0] * scale) / 2
        h_work = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1.0]])
        inlier_ratio, error, n_matches = 0.0, 99.0, len(src)

    ecc_refined = False
    if background_lock:
        refined = _ecc_refine(orig_gray, edit_gray, h_work, trusted=keypoint_based)
        if refined is not None and _plausible(refined, edit_work.shape[1], edit_work.shape[0]):
            h_work = refined
            ecc_refined = True

    # "The edit did not move" is the null hypothesis. A transform has to beat it on the actual
    # pixels; on smooth scenes a confident-looking fit can otherwise invent a shift.
    if same_aspect and not _beats_identity(orig_gray, edit_gray, h_work):
        h_work = np.eye(3)

    shift_work = _corner_shift(h_work, edit_work.shape[1], edit_work.shape[0])
    if same_aspect and shift_work < IDENTITY_TOLERANCE_PX:
        h_work = np.eye(3)  # do not resample for sub-pixel noise in the fit
    h_full = from_work_orig @ h_work @ to_work_edit
    shift_full = shift_work * (width / ww)

    aligned_gray = cv2.warpPerspective(edit_gray, h_work, (ww, wh))
    valid_work = cv2.warpPerspective(np.full(edit_gray.shape, 255, np.uint8), h_work, (ww, wh)) > 0
    flow_full: F32 | None = None
    residual = 0.0
    if allow_flow:
        changed = _rough_change(orig_gray, aligned_gray)
        result = _background_flow(orig_gray, aligned_gray, changed, valid_work)
        if result is not None:
            flow_work, residual = result
            xs, ys = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(wh, dtype=np.float32))
            warped = cv2.remap(
                aligned_gray, xs + flow_work[..., 0], ys + flow_work[..., 1], cv2.INTER_LINEAR
            )
            background = valid_work & ~dilate(changed, 6)
            before = _detail_residual(orig_gray, aligned_gray, background)
            after = _detail_residual(orig_gray, warped, background)
            if after < 0.92 * before:
                flow_full = cv2.resize(flow_work, (width, height), interpolation=cv2.INTER_LINEAR)
                flow_full = flow_full * np.float32(width / ww)

    identity = np.allclose(h_full, np.eye(3), atol=1e-9) and (eh, ew) == (height, width)
    if identity and flow_full is None:
        aligned = edited.copy()
        valid = np.ones((height, width), dtype=bool)
    else:
        aligned, valid = _warp(edited, h_full, (width, height), flow_full)

    confidence = _confidence(inlier_ratio, error, keypoint_based)
    if not keypoint_based and not same_aspect and not ecc_refined:
        confidence = "low"
    return Registration(
        aligned=aligned,
        valid=valid,
        homography=h_full,
        confidence=confidence,
        inlier_ratio=inlier_ratio,
        reprojection_error_px=error,
        n_matches=n_matches,
        shift_px=shift_full,
        flow_applied=flow_full is not None,
        residual_flow_px=residual,
        resized=resized,
        background_locked=locked,
    )
