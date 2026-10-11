from __future__ import annotations

import io
import os

os.environ.setdefault("PEEL_WARM", "0")
os.environ.setdefault("PEEL_RATE_CAPACITY", "10000")
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

import numpy as np
import pytest
from app.main import app
from fastapi.testclient import TestClient
from peel.synth import SynthCase, make_case
from PIL import Image
from skimage import data


def png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def case() -> SynthCase:
    base = np.ascontiguousarray(data.astronaut())
    return make_case(base, seed=3, kind="add", drifts=["cast", "shift", "patch_blur"])


@pytest.fixture(scope="session")
def files(case: SynthCase) -> dict[str, tuple[str, bytes, str]]:
    return {
        "original": ("original.png", png(case.original), "image/png"),
        "edited": ("edited.png", png(case.edited), "image/png"),
    }
