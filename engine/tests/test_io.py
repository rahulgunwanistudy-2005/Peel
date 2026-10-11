from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from skimage import data

from peel.arrays import U8
from peel.io import MAX_BYTES, check_same_scene, decode_image, encode_png, require_same_scene
from peel.synth import drift_cast, drift_jpeg, drift_shift
from peel.types import ErrorCode, PeelError

from .conftest import png_bytes


def test_decode_png_roundtrip(scene: U8) -> None:
    decoded = decode_image(png_bytes(scene))
    assert decoded.alpha is None
    assert np.array_equal(decoded.rgb, scene)
    assert np.array_equal(decode_image(encode_png(scene)).rgb, scene)


def test_exif_orientation_is_applied(scene: U8) -> None:
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    Image.fromarray(scene).save(buf, format="JPEG", exif=exif, quality=95)
    decoded = decode_image(buf.getvalue())
    assert decoded.rgb.shape[:2] == (scene.shape[1], scene.shape[0])


def test_alpha_is_kept_apart(scene: U8) -> None:
    alpha = np.full(scene.shape[:2], 255, dtype=np.uint8)
    alpha[:40] = 0
    buf = io.BytesIO()
    Image.fromarray(np.dstack([scene, alpha])).save(buf, format="PNG")
    decoded = decode_image(buf.getvalue())
    assert decoded.alpha is not None and np.array_equal(decoded.alpha, alpha)
    assert decoded.rgb.shape == scene.shape


def test_large_image_is_downscaled() -> None:
    big = np.zeros((300, 5000, 3), dtype=np.uint8)
    decoded = decode_image(png_bytes(big), max_long_side=4096)
    assert decoded.downscaled and max(decoded.rgb.shape[:2]) == 4096


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"", ErrorCode.INVALID_IMAGE),
        (b"definitely not an image", ErrorCode.INVALID_IMAGE),
        (b"\x00" * (MAX_BYTES + 1), ErrorCode.IMAGE_TOO_LARGE),
    ],
)
def test_bad_input_is_a_typed_error(payload: bytes, code: str) -> None:
    with pytest.raises(PeelError) as err:
        decode_image(payload)
    assert err.value.code == code


def test_unsupported_format(scene: U8) -> None:
    buf = io.BytesIO()
    Image.fromarray(scene).save(buf, format="BMP")
    with pytest.raises(PeelError) as err:
        decode_image(buf.getvalue())
    assert err.value.code == ErrorCode.UNSUPPORTED_FORMAT


def test_same_scene_accepts_drifted_copy(scene: U8) -> None:
    drifted = drift_jpeg(drift_cast(drift_shift(scene, 3.0, -2.0), db=5.0), 60)
    assert check_same_scene(scene, drifted).same


def test_same_scene_accepts_a_darkened_recoloured_copy(astronaut: U8) -> None:
    night = (astronaut.astype(np.float32) * np.array([0.35, 0.4, 0.6])).astype(np.uint8)
    assert check_same_scene(astronaut, night).same


def test_same_scene_accepts_resized_copy(astronaut: U8) -> None:
    small = np.asarray(Image.fromarray(astronaut).resize((300, 300)))
    assert check_same_scene(astronaut, small).same


def test_not_same_scene_on_unrelated_images(scene: U8, other_scene: U8, astronaut: U8) -> None:
    for a, b in (
        (scene, other_scene),
        (astronaut, np.ascontiguousarray(data.coffee())),
        (astronaut, scene),
    ):
        with pytest.raises(PeelError) as err:
            require_same_scene(a, b)
        assert err.value.code == ErrorCode.NOT_SAME_SCENE
        assert "different photos" in err.value.message
