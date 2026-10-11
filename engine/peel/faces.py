"""Identity lock for faces.

Editors routinely re-render bystanders into slightly different people. For every face in the
original we find its counterpart in the aligned edit and compare SFace embeddings. Embeddings are
compared within the uploaded pair only and are never stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from peel import models
from peel.arrays import U8, Bool

MIN_FACE_PX = 18  # at working resolution; smaller faces carry no usable identity signal
_FACE_WORDS = re.compile(
    r"\b(face|faces|head|person|people|man|men|woman|women|girl|boy|child|kid|baby|guy|lady|"
    r"portrait|selfie|hair|hairstyle|beard|mustache|moustache|eyes?|eyebrows?|nose|mouth|lips?|"
    r"smile|smiling|expression|teeth|skin|makeup|glasses|sunglasses|hat|cap|helmet|mask|"
    r"him|her|his|she|he|they|them|me|myself|age|older|younger)\b",
    re.I,
)


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h at working resolution, in the original
    distance: float | None  # SFace cosine distance to the same face in the edit
    found_in_edit: bool
    score: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)


def instruction_names_faces(instruction: str) -> bool:
    return bool(_FACE_WORDS.search(instruction))


def _detect(rgb: U8) -> npt.NDArray[np.float32]:
    fm = models.face_models()
    h, w = rgb.shape[:2]
    with models.detector_lock():
        fm.detector.setInputSize((w, h))
        _, faces = fm.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if faces is None:
        return np.zeros((0, 15), dtype=np.float32)
    return np.asarray(faces, dtype=np.float32)


def _iou(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def _embedding(full_rgb: U8, row: npt.NDArray[np.float32], scale: float) -> npt.NDArray[np.float32]:
    """Embed from full-resolution pixels; landmarks were found at working resolution."""
    fm = models.face_models()
    scaled = row.copy()
    scaled[:14] *= scale
    bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
    with models.detector_lock():
        crop = fm.recognizer.alignCrop(bgr, scaled[None])
        feature = fm.recognizer.feature(crop)
    return np.asarray(feature, dtype=np.float32).ravel().copy()


def _cosine_distance(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 1.0
    return float(1.0 - float(a @ b) / denom)


def compare_faces(orig_work: U8, aligned_work: U8, orig_full: U8, aligned_full: U8) -> list[Face]:
    scale = orig_full.shape[1] / orig_work.shape[1]
    orig_rows = _detect(orig_work)
    if len(orig_rows) == 0:
        return []
    edit_rows = _detect(aligned_work)
    faces: list[Face] = []
    for row in orig_rows:
        x, y, w, h = (round(float(v)) for v in row[:4])
        box = (max(0, x), max(0, y), w, h)
        if min(w, h) < MIN_FACE_PX:
            continue
        ious = [_iou(row, other) for other in edit_rows]
        best = int(np.argmax(ious)) if ious else -1
        found = best >= 0 and ious[best] >= 0.25
        # The images are registered, so when the detector misses the edited face we can still
        # crop it with the original's landmarks and measure how far the identity moved.
        edit_row = edit_rows[best] if found else row
        a = _embedding(orig_full, row, scale)
        b = _embedding(aligned_full, edit_row, scale)
        faces.append(Face(box, _cosine_distance(a, b), found, float(row[14])))
    return faces


def face_lock_mask(shape: tuple[int, int], face: Face, expand: float = 1.35) -> Bool:
    """Ellipse over the face, a little larger than the detector box."""
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy = face.center
    _, _, w, h = face.box
    axes = (max(2, round(w * expand / 2)), max(2, round(h * expand / 2)))
    cv2.ellipse(mask, (round(cx), round(cy)), axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)
