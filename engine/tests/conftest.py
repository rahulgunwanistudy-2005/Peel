"""Shared fixtures. Everything is generated in code; the only bundled photo is scikit-image's
public-domain astronaut, used for the face tests."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from skimage import data

from peel.arrays import U8
from peel.synth import SynthCase, make_case, procedural_scene


def png_bytes(rgb: U8) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def scene() -> U8:
    return procedural_scene(7)


@pytest.fixture(scope="session")
def other_scene() -> U8:
    return procedural_scene(1234)


@pytest.fixture(scope="session")
def astronaut() -> U8:
    return np.ascontiguousarray(data.astronaut())


@pytest.fixture(scope="session")
def add_case(scene: U8) -> SynthCase:
    return make_case(scene, seed=3, kind="add", drifts=["cast", "shift", "patch_blur"])
