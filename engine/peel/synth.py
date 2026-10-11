"""Synthetic edits with ground truth by construction.

Used by the unit tests and by the evaluation suite. A case is built from a base photo by:
  1. an *intended* edit with a known mask and a nameable target (a sprite is added, removed,
     recoloured or swapped), and
  2. *collateral* drift applied to the whole frame or to patches outside that mask (colour cast,
     small affine shift, JPEG re-encode, blur, sharpen, noise, local warp).

Because the intended edit and the drift are applied separately, the ideal output is known exactly:
the base photo with only the intended edit applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from peel.arrays import U8, Bool

SpriteName = Literal["ball", "star", "balloon", "banana", "heart"]
EditKind = Literal["add", "remove", "recolor", "replace"]
DriftKind = Literal[
    "cast", "shift", "jpeg", "blur", "sharpen", "noise", "patch_blur", "patch_color", "patch_warp"
]

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (214, 40, 40),
    "blue": (40, 90, 220),
    "green": (40, 170, 70),
    "yellow": (245, 205, 40),
    "purple": (140, 60, 190),
    "orange": (245, 130, 30),
    "pink": (240, 110, 170),
}
SPRITE_WORDS: dict[str, str] = {
    "ball": "ball",
    "star": "star",
    "balloon": "balloon",
    "banana": "banana",
    "heart": "heart",
}


@dataclass
class SynthCase:
    name: str
    original: U8
    edited: U8  # intended edit + collateral drift: what an AI editor would hand back
    ideal: U8  # intended edit only: what Peel should produce
    instruction: str
    intended: Bool  # ground-truth mask of the requested change
    kind: EditKind
    drifts: list[str] = field(default_factory=list)


def procedural_scene(seed: int, width: int = 480, height: int = 360) -> U8:
    """A textured, feature-rich scene made from noise, blocks and strokes. No assets needed."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)
    for octave, weight in ((8, 0.5), (24, 0.3), (72, 0.2)):
        small = rng.random((max(2, height // octave), max(2, width // octave), 3), dtype=np.float32)
        img += weight * cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img * 0.65 + 0.2, 0, 1)
    canvas = (img * 255).astype(np.uint8)
    for _ in range(26):
        color = tuple(int(c) for c in rng.integers(20, 235, size=3))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        w, h = int(rng.integers(12, width // 5)), int(rng.integers(12, height // 5))
        if rng.random() < 0.5:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1 if rng.random() < 0.6 else 2)
        else:
            cv2.ellipse(
                canvas, (x, y), (w // 2, h // 2), float(rng.integers(0, 180)), 0, 360, color, 2
            )
    for _ in range(18):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.line(canvas, p1, p2, tuple(int(c) for c in rng.integers(0, 255, size=3)), 1)
    grain = rng.normal(0, 3.0, canvas.shape).astype(np.float32)
    return np.clip(canvas.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def _sprite_mask(name: SpriteName, size: int) -> Bool:
    s = size
    mask = np.zeros((s, s), dtype=np.uint8)
    c = s // 2
    if name == "ball":
        cv2.circle(mask, (c, c), int(s * 0.46), 1, -1)
    elif name == "balloon":
        cv2.ellipse(mask, (c, int(s * 0.4)), (int(s * 0.34), int(s * 0.39)), 0, 0, 360, 1, -1)
        knot = np.array(
            [[c, int(s * 0.76)], [c - s // 14, int(s * 0.86)], [c + s // 14, int(s * 0.86)]]
        )
        cv2.fillPoly(mask, [knot], 1)
        cv2.line(mask, (c, int(s * 0.86)), (c, s - 1), 1, max(1, s // 40))
    elif name == "star":
        pts = []
        for i in range(10):
            radius = s * (0.48 if i % 2 == 0 else 0.2)
            angle = -np.pi / 2 + i * np.pi / 5
            pts.append([c + radius * np.cos(angle), c + radius * np.sin(angle)])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
    elif name == "heart":
        r = int(s * 0.24)
        cv2.circle(mask, (c - r, int(s * 0.36)), r, 1, -1)
        cv2.circle(mask, (c + r, int(s * 0.36)), r, 1, -1)
        tri = np.array([[c - 2 * r, int(s * 0.44)], [c + 2 * r, int(s * 0.44)], [c, int(s * 0.92)]])
        cv2.fillPoly(mask, [tri], 1)
    elif name == "banana":
        cv2.ellipse(mask, (c, int(s * 0.3)), (int(s * 0.46), int(s * 0.5)), 0, 20, 160, 1, -1)
        cut = np.zeros_like(mask)
        cv2.ellipse(cut, (c, int(s * 0.12)), (int(s * 0.44), int(s * 0.46)), 0, 0, 360, 1, -1)
        mask[cut > 0] = 0
    return mask.astype(bool)


def render_sprite(name: SpriteName, color: tuple[int, int, int], size: int) -> tuple[U8, Bool]:
    """A shaded sprite and its mask. Shading gives it a little volume."""
    mask = _sprite_mask(name, size)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    light = 1.15 - 0.55 * np.hypot(xs - size * 0.36, ys - size * 0.32) / size
    rgb = np.clip(np.array(color, dtype=np.float32)[None, None] * light[..., None], 0, 255)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    rgb[edge > 0] *= 0.55
    return rgb.astype(np.uint8), mask


def paste_sprite(image: U8, sprite: U8, mask: Bool, x: int, y: int) -> tuple[U8, Bool]:
    out = image.copy()
    h, w = mask.shape
    region = out[y : y + h, x : x + w]
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.7)[..., None]
    region[:] = (sprite * soft + region * (1 - soft)).astype(np.uint8)
    full = np.zeros(image.shape[:2], dtype=bool)
    full[y : y + h, x : x + w] = soft[..., 0] > 0.02
    return out, full


# ----------------------------------------------------------------------------- drift


def drift_cast(image: U8, db: float = 4.0, da: float = 1.0, dl: float = 0.0) -> U8:
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    lab += np.array([dl, da, db], dtype=np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def drift_shift(image: U8, dx: float, dy: float, scale: float = 1.0, angle: float = 0.0) -> U8:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def drift_jpeg(image: U8, quality: int = 60) -> U8:
    ok, buf = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert ok and decoded is not None
    return np.ascontiguousarray(decoded[..., ::-1])


def drift_blur(image: U8, sigma: float = 1.2) -> U8:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def drift_sharpen(image: U8, amount: float = 1.0) -> U8:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5).astype(np.float32)
    sharp = image.astype(np.float32) * (1 + amount) - blurred * amount
    return np.clip(sharp, 0, 255).astype(np.uint8)


def drift_noise(image: U8, sigma: float, rng: np.random.Generator) -> U8:
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _patch(
    shape: tuple[int, int], avoid: Bool, rng: np.random.Generator, frac: float = 0.22
) -> tuple[int, int, int, int] | None:
    """A rectangle that stays clear of the intended region."""
    h, w = shape
    ph, pw = max(16, int(h * frac)), max(16, int(w * frac))
    guard = cv2.dilate(avoid.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    for _ in range(40):
        x, y = int(rng.integers(4, w - pw - 4)), int(rng.integers(4, h - ph - 4))
        if not guard[y : y + ph, x : x + pw].any():
            return x, y, pw, ph
    return None


def drift_patch_blur(image: U8, box: tuple[int, int, int, int], sigma: float = 3.0) -> U8:
    x, y, w, h = box
    out = image.copy()
    out[y : y + h, x : x + w] = cv2.GaussianBlur(out[y : y + h, x : x + w], (0, 0), sigma)
    return out


def drift_patch_color(image: U8, box: tuple[int, int, int, int], hue_shift: int = 18) -> U8:
    x, y, w, h = box
    out = image.copy()
    hsv = cv2.cvtColor(out[y : y + h, x : x + w], cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + 25, 0, 255)
    out[y : y + h, x : x + w] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def drift_patch_warp(
    image: U8, box: tuple[int, int, int, int], strength: float = 0.11, seed: int = 0
) -> U8:
    """Smooth random (elastic) warp inside the patch: the same content with different proportions,
    which is what a re-rendered face looks like."""
    x, y, w, h = box
    rng = np.random.default_rng(seed)
    coarse_x = rng.uniform(-1, 1, (4, 4)).astype(np.float32)
    coarse_y = rng.uniform(-1, 1, (4, 4)).astype(np.float32)
    window = np.sqrt(np.outer(np.hanning(h), np.hanning(w))).astype(np.float32)
    flow_x = cv2.resize(coarse_x, (w, h), interpolation=cv2.INTER_CUBIC) * strength * w * window
    flow_y = cv2.resize(coarse_y, (w, h), interpolation=cv2.INTER_CUBIC) * strength * h * window
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    out = image.copy()
    out[y : y + h, x : x + w] = cv2.remap(
        image[y : y + h, x : x + w],
        xs + flow_x,
        ys + flow_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    return out


# ----------------------------------------------------------------------------- cases


def _place(shape: tuple[int, int], size: int, rng: np.random.Generator) -> tuple[int, int]:
    h, w = shape
    margin = max(8, size // 6)
    return (
        int(rng.integers(margin, max(margin + 1, w - size - margin))),
        int(rng.integers(margin, max(margin + 1, h - size - margin))),
    )


def make_case(
    base: U8,
    seed: int,
    kind: EditKind,
    drifts: list[DriftKind],
    *,
    name: str = "",
    face_box: tuple[int, int, int, int] | None = None,
) -> SynthCase:
    """Build one case. ``face_box`` (x, y, w, h) makes ``patch_warp`` land on a face."""
    rng = np.random.default_rng(seed)
    h, w = base.shape[:2]
    size = int(min(h, w) * float(rng.uniform(0.2, 0.3)))
    sprite_names: list[SpriteName] = ["ball", "star", "balloon", "banana", "heart"]
    color_names = list(COLORS)
    sprite_a: SpriteName = sprite_names[int(rng.integers(len(sprite_names)))]
    color_a = color_names[int(rng.integers(len(color_names)))]
    avoid = np.zeros((h, w), dtype=bool)
    if face_box is not None:
        fx, fy, fw, fh = face_box
        avoid[
            max(0, fy - fh // 3) : fy + fh + fh // 3, max(0, fx - fw // 3) : fx + fw + fw // 3
        ] = True
    for _ in range(60):
        x, y = _place((h, w), size, rng)
        if not avoid[y : y + size, x : x + size].any():
            break
    sprite, mask = render_sprite(sprite_a, COLORS[color_a], size)
    thing = f"{color_a} {SPRITE_WORDS[sprite_a]}"

    if kind == "add":
        original = base
        ideal, intended = paste_sprite(base, sprite, mask, x, y)
        instruction = f"Add a {thing}"
    elif kind == "remove":
        original, intended = paste_sprite(base, sprite, mask, x, y)
        ideal = base
        instruction = f"Remove the {thing}"
    elif kind == "recolor":
        original, intended = paste_sprite(base, sprite, mask, x, y)
        color_b = color_names[
            (color_names.index(color_a) + 1 + int(rng.integers(len(color_names) - 1)))
            % len(color_names)
        ]
        recolored, _ = render_sprite(sprite_a, COLORS[color_b], size)
        ideal, _ = paste_sprite(base, recolored, mask, x, y)
        instruction = f"Make the {thing} {color_b}"
    else:  # replace
        original, mask_a = paste_sprite(base, sprite, mask, x, y)
        sprite_b: SpriteName = sprite_names[
            (sprite_names.index(sprite_a) + 1 + int(rng.integers(len(sprite_names) - 1)))
            % len(sprite_names)
        ]
        color_b = color_names[int(rng.integers(len(color_names)))]
        other, other_mask = render_sprite(sprite_b, COLORS[color_b], size)
        ideal, mask_b = paste_sprite(base, other, other_mask, x, y)
        intended = mask_a | mask_b
        instruction = f"Replace the {thing} with a {color_b} {SPRITE_WORDS[sprite_b]}"

    edited = ideal
    applied: list[str] = []
    for drift in drifts:
        if drift == "cast":
            edited = drift_cast(
                edited,
                db=float(rng.uniform(2.5, 6.0)) * float(rng.choice([-1, 1])),
                da=float(rng.uniform(-2, 2)),
                dl=float(rng.uniform(-2, 2)),
            )
        elif drift == "shift":
            pct = float(rng.uniform(0.005, 0.02))
            angle = float(rng.uniform(0, 2 * np.pi))
            edited = drift_shift(
                edited,
                pct * w * np.cos(angle),
                pct * h * np.sin(angle),
                scale=float(rng.uniform(0.995, 1.005)),
            )
        elif drift == "jpeg":
            edited = drift_jpeg(edited, int(rng.integers(45, 80)))
        elif drift == "blur":
            edited = drift_blur(edited, float(rng.uniform(0.8, 1.4)))
        elif drift == "sharpen":
            edited = drift_sharpen(edited, float(rng.uniform(0.6, 1.2)))
        elif drift == "noise":
            edited = drift_noise(edited, float(rng.uniform(3, 7)), rng)
        else:
            box = (
                face_box
                if (drift == "patch_warp" and face_box is not None)
                else _patch((h, w), intended | avoid if drift != "patch_warp" else intended, rng)
            )
            if box is None:
                continue
            if drift == "patch_blur":
                edited = drift_patch_blur(edited, box, float(rng.uniform(2.5, 4.0)))
            elif drift == "patch_color":
                edited = drift_patch_color(edited, box, int(rng.integers(12, 30)))
            else:
                edited = drift_patch_warp(edited, box, float(rng.uniform(0.1, 0.14)), seed)
        applied.append(drift)

    return SynthCase(
        name=name or f"{kind}-{seed}",
        original=original,
        edited=edited,
        ideal=ideal,
        instruction=instruction,
        intended=intended,
        kind=kind,
        drifts=applied,
    )
