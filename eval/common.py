"""Shared helpers for the evaluation scripts."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from peel.arrays import U8

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
CACHE = ROOT / "eval" / "cache"
WEB_DATA = ROOT / "web" / "src" / "data" / "eval"
WEB_PUBLIC = ROOT / "web" / "public"


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unknown"


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def read_rgb(path: Path) -> U8:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(bgr[..., ::-1])


def write_webp(path: Path, rgb: U8, *, lossless: bool = False, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_WEBP_QUALITY, 101 if lossless else quality]
    if not cv2.imwrite(str(path), rgb[..., ::-1], params):
        raise OSError(f"could not write {path}")


def write_png(path: Path, rgb: U8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6])


def write_result(name: str, payload: dict[str, Any]) -> Path:
    """Write eval/results/<name>.json and mirror it into the web app's data folder."""
    text = json.dumps(payload, indent=2) + "\n"
    for folder in (RESULTS, WEB_DATA):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.json").write_text(text)
    return RESULTS / f"{name}.json"


def fit_long_side(rgb: U8, long_side: int) -> U8:
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def heat_to_rgb(heat: np.ndarray, vmax: float) -> U8:
    """Grayscale heatmap: black is zero, white is ``vmax`` or more."""
    gray = np.clip(heat / max(vmax, 1e-6), 0, 1)
    return np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)


def side_by_side(left: U8, right: U8, gap: int = 12) -> U8:
    h = min(left.shape[0], right.shape[0])
    left = cv2.resize(
        left, (round(left.shape[1] * h / left.shape[0]), h), interpolation=cv2.INTER_AREA
    )
    right = cv2.resize(
        right, (round(right.shape[1] * h / right.shape[0]), h), interpolation=cv2.INTER_AREA
    )
    spacer = np.full((h, gap, 3), 119, dtype=np.uint8)
    return np.hstack([left, spacer, right])
