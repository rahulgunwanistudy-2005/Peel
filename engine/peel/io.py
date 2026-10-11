"""Decoding, size policy and the same-scene gate."""

from __future__ import annotations

import io as _io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from skimage.metrics import structural_similarity

from peel.arrays import U8, luminance_u8
from peel.types import ErrorCode, PeelError

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 80_000_000
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "MPO", "HEIF", "HEIC"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

try:  # HEIC is optional; the product works without it.
    import pillow_heif  # type: ignore[import-not-found,unused-ignore]

    pillow_heif.register_heif_opener()
except ImportError:
    pass


@dataclass(frozen=True)
class DecodedImage:
    rgb: U8
    alpha: U8 | None
    source_size: tuple[int, int]
    downscaled: bool


def decode_image(data: bytes, *, max_long_side: int = 4096, label: str = "image") -> DecodedImage:
    """Bytes to oriented sRGB uint8, with alpha kept apart."""
    if len(data) == 0:
        raise PeelError(ErrorCode.INVALID_IMAGE, f"The {label} is empty.")
    if len(data) > MAX_BYTES:
        mb = len(data) / 1e6
        raise PeelError(
            ErrorCode.IMAGE_TOO_LARGE,
            f"The {label} is {mb:.1f} MB. The limit is 15 MB.",
            status=413,
        )
    try:
        img: Image.Image = Image.open(_io.BytesIO(data))
        fmt = (img.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            raise PeelError(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"The {label} is a {fmt or 'unknown'} file. Use PNG, JPEG or WebP.",
                status=415,
            )
        img.load()
    except PeelError:
        raise
    except Image.DecompressionBombError as exc:
        raise PeelError(
            ErrorCode.IMAGE_TOO_LARGE, f"The {label} has too many pixels to process.", status=413
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise PeelError(
            ErrorCode.INVALID_IMAGE, f"The {label} could not be read. Use PNG, JPEG or WebP."
        ) from exc

    transposed = ImageOps.exif_transpose(img)
    img = transposed if transposed is not None else img
    img = _to_srgb(img)
    source_size = img.size

    if img.mode in ("I", "F") or img.mode.startswith("I;16"):
        # 16-bit and float images: scale to 8 bits ourselves; PIL's convert() would clip them.
        deep = np.asarray(img, dtype=np.float32)
        top = 65535.0 if float(deep.max(initial=0.0)) > 255.0 else 255.0
        if img.mode == "F" and float(deep.max(initial=0.0)) <= 1.0:
            top = 1.0
        img = Image.fromarray(np.clip(deep / top * 255.0, 0, 255).astype(np.uint8), mode="L")

    alpha: U8 | None = None
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha_arr = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        alpha = None if bool((alpha_arr == 255).all()) else alpha_arr
        img = rgba.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    downscaled = False
    w, h = img.size
    if max(w, h) > max_long_side:
        scale = max_long_side / max(w, h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(size, Image.Resampling.LANCZOS)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        downscaled = True

    rgb = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    if min(rgb.shape[:2]) < 32:
        raise PeelError(ErrorCode.INVALID_IMAGE, f"The {label} is too small (under 32 px).")
    return DecodedImage(rgb=rgb, alpha=alpha, source_size=source_size, downscaled=downscaled)


def _to_srgb(img: Image.Image) -> Image.Image:
    profile = img.info.get("icc_profile")
    if not profile or img.mode not in ("RGB", "RGBA"):
        return img
    try:
        src = ImageCms.ImageCmsProfile(_io.BytesIO(profile))
        dst = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(img, src, dst, outputMode=img.mode)
    except (ImageCms.PyCMSError, OSError, ValueError):
        return img
    return converted if converted is not None else img


def encode_png(rgb: U8, alpha: U8 | None = None, *, compression: int = 3) -> bytes:
    if alpha is not None:
        bgra = np.dstack([rgb[..., ::-1], alpha])
        ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    else:
        ok, buf = cv2.imencode(".png", rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise PeelError(ErrorCode.INTERNAL, "Could not encode the result image.", status=500)
    return bytes(buf.tobytes())


def aspect_matches(a: U8, b: U8, tolerance: float = 0.01) -> bool:
    ra = a.shape[1] / a.shape[0]
    rb = b.shape[1] / b.shape[0]
    return abs(ra - rb) / ra <= tolerance


SSIM_SAME = 0.45
GRADIENT_SAME = 0.78
ORB_INLIERS_SAME = 14


@dataclass(frozen=True)
class SceneCheck:
    same: bool
    ssim: float
    gradient_agreement: float
    orb_inliers: int


def _gate_view(rgb: U8, size: tuple[int, int]) -> U8:
    gray = luminance_u8(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))
    return cv2.GaussianBlur(gray, (0, 0), 1.2)


def _gradient_agreement(a: U8, b: U8) -> float:
    """Magnitude-weighted |cos| between gradient directions. Survives relighting and recolor.

    Unrelated images score about 2/pi (0.64), the mean of |cos| over random angles; 1.0 is perfect.
    """
    fa = cv2.GaussianBlur(a, (0, 0), 2.0).astype(np.float32)
    fb = cv2.GaussianBlur(b, (0, 0), 2.0).astype(np.float32)
    ax, ay = cv2.Sobel(fa, cv2.CV_32F, 1, 0), cv2.Sobel(fa, cv2.CV_32F, 0, 1)
    bx, by = cv2.Sobel(fb, cv2.CV_32F, 1, 0), cv2.Sobel(fb, cv2.CV_32F, 0, 1)
    ma, mb = np.hypot(ax, ay), np.hypot(bx, by)
    weight = ma * mb
    total = float(weight.sum())
    if total < 1e-6:
        return 0.0
    return float(np.abs(ax * bx + ay * by).sum() / total)


def _orb_inliers(a: U8, b: U8) -> int:
    orb = cv2.ORB.create(nfeatures=800)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.78 * n.distance]
    if len(good) < 8:
        return 0
    src = np.array([ka[m.queryIdx].pt for m in good], dtype=np.float32)
    dst = np.array([kb[m.trainIdx].pt for m in good], dtype=np.float32)
    _, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return 0 if inliers is None else int(inliers.sum())


def check_same_scene(original: U8, edited: U8) -> SceneCheck:
    """Cheap test that the two images show the same scene.

    Three independent cues, any one of which is enough: plain structural similarity (most edits),
    gradient-direction agreement (survives night/recolor/style edits), and geometrically verified
    ORB matches (survives crops and shifts).
    """
    h, w = original.shape[:2]
    scale = 256 / max(h, w)
    size = (max(32, round(w * scale)), max(32, round(h * scale)))
    a = _gate_view(original, size)
    b = _gate_view(edited, size)
    ssim = float(structural_similarity(a, b, data_range=255))
    grad = _gradient_agreement(a, b)
    if ssim >= SSIM_SAME or grad >= GRADIENT_SAME:
        return SceneCheck(True, ssim, grad, -1)
    inliers = _orb_inliers(a, _gate_view(edited, _fit_size(edited, 256)))
    return SceneCheck(inliers >= ORB_INLIERS_SAME, ssim, grad, inliers)


def _fit_size(rgb: U8, long_side: int) -> tuple[int, int]:
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    return (max(32, round(w * scale)), max(32, round(h * scale)))


def require_same_scene(original: U8, edited: U8) -> SceneCheck:
    check = check_same_scene(original, edited)
    if not check.same:
        raise PeelError(
            ErrorCode.NOT_SAME_SCENE,
            "These look like two different photos. Peel compares an image with its edited "
            "version, so upload the original and the AI edit of that same picture.",
            status=422,
        )
    return check
