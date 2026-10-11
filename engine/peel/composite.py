"""Blend the kept regions of one image into another, and leave everything else untouched.

``composite(base, top, mask)`` pastes ``top`` into ``base`` where ``mask`` is set:
  * the mask is feathered with an edge-aware guided filter,
  * ``top`` is tone-matched to ``base`` in a thin ring around the mask (harmonization),
  * the blend is a 5-level Laplacian pyramid (or Poisson on request),
  * every pixel outside the blend support is copied from ``base`` byte for byte.

For a local edit ``base`` is the original and ``top`` the aligned edit. For a global edit the
roles swap: the edit is the base and locked regions of the original are pasted back in.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from peel.arrays import F32, U8, Bool, dilate, lab_to_rgb, rgb_to_lab, to_float, to_u8
from peel.change import CastFit
from peel.types import BlendMode

PYRAMID_LEVELS = 5
ALPHA_EPS = 1.0 / 255.0


@dataclass
class CompositeResult:
    image: U8
    support: Bool  # full-resolution pixels that may differ from ``base``


def guided_filter(guide: F32, src: F32, radius: int, eps: float) -> F32:
    """He et al. guided filter with a grayscale guide, built from box filters."""
    k = (2 * radius + 1, 2 * radius + 1)

    def box(x: npt.NDArray[np.floating]) -> F32:
        return cv2.boxFilter(x, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)

    mean_g, mean_s = box(guide), box(src)
    cov = box(guide * src) - mean_g * mean_s
    var = box(guide * guide) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return np.asarray(box(a) * guide + box(b), dtype=np.float32)


def _laplacian_blend(base: F32, top: F32, alpha: F32, levels: int) -> F32:
    gauss_a = [alpha]
    lap_base, lap_top = [], []
    cur_b, cur_t = base, top
    for _ in range(levels):
        next_b, next_t = cv2.pyrDown(cur_b), cv2.pyrDown(cur_t)
        size = (cur_b.shape[1], cur_b.shape[0])
        lap_base.append(cur_b - cv2.pyrUp(next_b, dstsize=size))
        lap_top.append(cur_t - cv2.pyrUp(next_t, dstsize=size))
        gauss_a.append(cv2.pyrDown(gauss_a[-1]))
        cur_b, cur_t = next_b, next_t
    a = gauss_a[-1][..., None]
    out = cur_t * a + cur_b * (1.0 - a)
    for level in range(levels - 1, -1, -1):
        a = gauss_a[level][..., None]
        size = (lap_base[level].shape[1], lap_base[level].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + lap_top[level] * a + lap_base[level] * (1.0 - a)
    return np.asarray(out, dtype=np.float32)


def _poisson_blend(base: U8, top: U8, alpha: F32) -> F32 | None:
    mask = (alpha > 0.5).astype(np.uint8) * 255
    if not mask.any():
        return None
    x, y, w, h = cv2.boundingRect(mask)
    if x <= 1 or y <= 1 or x + w >= mask.shape[1] - 1 or y + h >= mask.shape[0] - 1:
        return None  # seamlessClone needs a margin around the mask
    center = (x + w // 2, y + h // 2)
    try:
        cloned = cv2.seamlessClone(top[..., ::-1], base[..., ::-1], mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return None
    return to_float(np.ascontiguousarray(cloned[..., ::-1]))


def _ring_offset(
    base_work: U8, top_work: U8, mask_work: Bool, stable_work: Bool, falloff: bool
) -> F32 | None:
    """Low-frequency colour offset that makes ``top`` meet ``base`` at the mask boundary.

    Measured in a ring just outside the mask, on pixels that did not otherwise change, then
    diffused into the mask (normalized convolution).
    """
    short = min(mask_work.shape)
    ring = dilate(mask_work, max(3, round(0.03 * short))) & ~dilate(mask_work, 1) & stable_work
    if int(ring.sum()) < 30:
        return None
    diff = (to_float(base_work) - to_float(top_work)) * ring[..., None]
    sigma = max(4.0, 0.05 * short)
    weight = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), sigma)
    offset = cv2.GaussianBlur(diff, (0, 0), sigma) / np.maximum(weight, 1e-3)[..., None]
    offset = np.clip(offset, -0.25, 0.25)
    offset[weight < 1e-3] = 0.0
    if falloff:
        inside = cv2.distanceTransform(mask_work.astype(np.uint8), cv2.DIST_L2, 3)
        offset = offset * np.exp(-inside / max(2.0, 0.04 * short))[..., None]
    return np.asarray(offset * mask_work[..., None], dtype=np.float32)


def _levels_for(long_side: int) -> int:
    """Five levels at 2048 px and up, fewer for small images so the blend stays local."""
    return int(np.clip(round(np.log2(max(long_side, 64) / 64.0)), 3, PYRAMID_LEVELS))


def _crop_box(support: Bool, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(support)
    h, w = support.shape
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + 1 + pad),
        min(h, int(ys.max()) + 1 + pad),
    )


def composite(
    base: U8,
    top: U8,
    mask_work: Bool,
    *,
    stable_work: Bool,
    cast: CastFit | None = None,
    harmonize: bool = True,
    harmonize_falloff: bool = True,
    blend: BlendMode = "laplacian",
) -> CompositeResult:
    """Paste ``top`` into ``base`` where ``mask_work`` (working resolution) is set.

    When ``cast`` was detected, its mean colour shift is removed from ``top`` before blending.
    ``stable_work`` marks pixels where the two images agree apart from noise.
    """
    height, width = base.shape[:2]
    if not mask_work.any():
        return CompositeResult(base.copy(), np.zeros((height, width), dtype=bool))

    wh, ww = mask_work.shape
    scale = width / ww
    grown = dilate(mask_work, 2)  # take the antialiased rim of the change with it
    soft = cv2.resize(grown.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    if scale > 1.0:
        soft = cv2.GaussianBlur(soft, (0, 0), 0.5 * scale)

    levels = _levels_for(max(height, width))
    pad = 2 ** (levels + 1)
    x0, y0, x1, y1 = _crop_box(soft > ALPHA_EPS, pad + round(4 * scale))
    base_crop = base[y0:y1, x0:x1]
    top_crop = top[y0:y1, x0:x1]
    soft_crop = soft[y0:y1, x0:x1]
    top_f = to_float(top_crop)

    if cast is not None and cast.detected:
        top_f = lab_to_rgb(cast.remove_shift(rgb_to_lab(top_f)))
    if harmonize:
        base_work = cv2.resize(base, (ww, wh), interpolation=cv2.INTER_AREA)
        top_work = cv2.resize(top, (ww, wh), interpolation=cv2.INTER_AREA)
        if cast is not None and cast.detected:
            top_work = to_u8(lab_to_rgb(cast.remove_shift(rgb_to_lab(to_float(top_work)))))
        offset = _ring_offset(base_work, top_work, grown, stable_work, harmonize_falloff)
        if offset is not None:
            full = cv2.resize(offset, (width, height), interpolation=cv2.INTER_LINEAR)
            top_f = np.clip(top_f + full[y0:y1, x0:x1], 0.0, 1.0)

    # Edge-aware feather: snap the soft mask to edges of the image being pasted.
    guide = cv2.cvtColor(top_f, cv2.COLOR_RGB2GRAY)
    radius = max(2, round(0.004 * max(height, width)))
    refined = guided_filter(guide, soft_crop, radius, 1e-3)
    alpha = np.clip((refined - 0.08) / 0.84, 0.0, 1.0).astype(np.float32)
    alpha[soft_crop <= ALPHA_EPS] = 0.0  # the feather never reaches past the grown mask

    base_f = to_float(base_crop)
    blended: F32 | None = None
    if blend == "poisson":
        blended = _poisson_blend(base_crop, to_u8(top_f), alpha)
    if blended is None:
        while levels > 1 and min(alpha.shape) < 2 ** (levels + 1):
            levels -= 1
        blended = _laplacian_blend(base_f, top_f, alpha, levels)

    # Fade the low-frequency leakage of the pyramid to exactly zero a fixed distance from the mask,
    # so everything beyond is the base image, byte for byte.
    active = alpha > ALPHA_EPS
    distance = cv2.distanceTransform((~active).astype(np.uint8), cv2.DIST_L2, 3)
    window = np.clip(1.0 - distance / float(pad // 2), 0.0, 1.0)
    out_f = base_f + (blended - base_f) * window[..., None]
    support_crop = window > 0.0

    image = base.copy()
    image[y0:y1, x0:x1] = np.where(support_crop[..., None], to_u8(out_f), base_crop)
    support = np.zeros((height, width), dtype=bool)
    support[y0:y1, x0:x1] = support_crop
    return CompositeResult(image=image, support=support)
