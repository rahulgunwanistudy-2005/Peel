from __future__ import annotations

import cv2
import numpy as np

from peel.arrays import U8
from peel.register import register
from peel.synth import drift_shift


def _corner_error(h: np.ndarray, expected: np.ndarray, width: int, height: int) -> float:
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)[None]
    got = cv2.perspectiveTransform(corners, h)[0]
    want = cv2.perspectiveTransform(corners, expected)[0]
    return float(np.max(np.linalg.norm(got - want, axis=1)))


def test_identical_images_are_not_resampled(scene: U8) -> None:
    reg = register(scene, scene.copy())
    assert reg.confidence == "high"
    assert np.array_equal(reg.aligned, scene)
    assert reg.valid.all() and not reg.flow_applied


def test_shifted_copy_registers_within_half_a_pixel(astronaut: U8) -> None:
    dx, dy = 5.3, -3.6
    shifted = drift_shift(astronaut, dx, dy)
    reg = register(astronaut, shifted, allow_flow=False)
    undo = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]])
    h, w = astronaut.shape[:2]
    assert _corner_error(reg.homography, undo, w, h) < 0.5
    assert reg.confidence == "high"
    interior = (slice(20, -20), slice(20, -20))
    error = np.abs(reg.aligned[interior].astype(int) - astronaut[interior].astype(int)).mean()
    assert error < 4.0


def test_rotation_and_scale_are_recovered(astronaut: U8) -> None:
    warped = drift_shift(astronaut, 2.0, 1.0, scale=1.01, angle=0.6)
    reg = register(astronaut, warped, allow_flow=False)
    h, w = astronaut.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0.6, 1.01)
    m[0, 2] += 2.0
    m[1, 2] += 1.0
    undo = np.linalg.inv(np.vstack([m, [0, 0, 1.0]]))
    assert _corner_error(reg.homography, undo, w, h) < 0.75


def test_resized_edit_comes_back_at_original_size(astronaut: U8) -> None:
    small = cv2.resize(astronaut, (320, 320), interpolation=cv2.INTER_AREA)
    reg = register(astronaut, small)
    assert reg.resized and reg.aligned.shape == astronaut.shape
    assert np.abs(reg.aligned.astype(int) - astronaut.astype(int)).mean() < 8.0


def test_background_lock_ignores_a_moved_object(astronaut: U8) -> None:
    edited = astronaut.copy()
    block = astronaut[40:240, 150:350].copy()
    edited[60:260, 170:370] = block  # the "edit" moves the head; the frame itself did not move
    reg = register(astronaut, edited)
    assert reg.shift_px < 0.5


def test_featureless_images_fall_back_gracefully() -> None:
    flat = np.full((200, 300, 3), 128, dtype=np.uint8)
    reg = register(flat, flat.copy())
    assert reg.aligned.shape == flat.shape and reg.confidence != "high"
