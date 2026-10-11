"""Lazy model singletons and warm-up.

Four small models, all CPU:
  * CLIPSeg (CIDAS/clipseg-rd64-refined) for instruction grounding,
  * LPIPS (AlexNet, spatial) for perceptual change,
  * YuNet + SFace (OpenCV Zoo) for face detection and identity.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from peel.arrays import F32, U8

CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"
CLIPSEG_SIZE = 352
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
_WEIGHT_URLS = {
    YUNET_FILE: f"{_ZOO}/face_detection_yunet/{YUNET_FILE}",
    SFACE_FILE: f"{_ZOO}/face_recognition_sface/{SFACE_FILE}",
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_lock = threading.RLock()
_threads_configured = False


def model_dir() -> Path:
    """Where the OpenCV Zoo weights live. Baked into the Docker image at build time."""
    root = os.environ.get("PEEL_MODEL_DIR")
    path = Path(root) if root else Path.home() / ".cache" / "peel"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _weight_path(name: str) -> Path:
    path = model_dir() / name
    if not path.exists() or path.stat().st_size < 100_000:
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(_WEIGHT_URLS[name], tmp)
        tmp.replace(path)
    return path


def thread_count() -> int:
    """Threads to use. ``PEEL_THREADS`` wins: in a container ``os.cpu_count()`` reports the host's
    cores, not the CPU quota, and oversubscribing two vCPUs makes everything slower."""
    configured = os.environ.get("PEEL_THREADS", "")
    if configured.isdigit() and int(configured) > 0:
        return int(configured)
    return max(1, min(os.cpu_count() or 1, 8))


def configure_torch() -> None:
    global _threads_configured
    with _lock:
        if _threads_configured:
            return
        torch.set_num_threads(thread_count())
        cv2.setNumThreads(thread_count())
        torch.set_grad_enabled(False)
        cv2.setRNGSeed(0)
        _threads_configured = True


@dataclass
class ClipSeg:
    model: Any
    tokenizer: Any

    def heatmaps(self, images: list[U8], phrases: list[str]) -> F32:
        """Sigmoid heatmaps, shape (len(images), len(phrases), 352, 352).

        The vision tower runs once per image; only the light decoder runs per phrase.
        """
        batch = np.stack([_clipseg_preprocess(img) for img in images])
        tokens = self.tokenizer(phrases, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            vision = self.model.clip.vision_model(
                pixel_values=torch.from_numpy(batch), output_hidden_states=True
            )
            hidden = vision.hidden_states
            text = self.model.clip.get_text_features(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
            )
            if not isinstance(text, torch.Tensor):
                text = text.pooler_output
            n_img, n_txt = len(images), len(phrases)
            activations = [
                hidden[i + 1].repeat_interleave(n_txt, dim=0) for i in self.model.extract_layers
            ]
            cond = text.repeat(n_img, 1)
            logits = self.model.decoder(activations, cond).logits
            heat = torch.sigmoid(logits).reshape(n_img, n_txt, CLIPSEG_SIZE, CLIPSEG_SIZE)
        return heat.numpy().astype(np.float32)


def _clipseg_preprocess(rgb: U8) -> F32:
    resized = cv2.resize(rgb, (CLIPSEG_SIZE, CLIPSEG_SIZE), interpolation=cv2.INTER_AREA)
    x = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


@dataclass
class Lpips:
    net: Any

    def spatial(self, a: U8, b: U8) -> F32:
        """Per-pixel LPIPS distance map at the resolution of the inputs."""
        ta = torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0)
        tb = torch.from_numpy(b.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0)
        with torch.inference_mode():
            out = self.net(ta, tb)
        return out[0, 0].numpy().astype(np.float32)


@dataclass
class FaceModels:
    detector: Any
    recognizer: Any


_clipseg: ClipSeg | None = None
_lpips: Lpips | None = None
_faces: FaceModels | None = None


def clipseg() -> ClipSeg:
    global _clipseg
    with _lock:
        if _clipseg is None:
            configure_torch()
            from transformers import AutoTokenizer, CLIPSegForImageSegmentation

            model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID)
            model.eval()
            _clipseg = ClipSeg(model=model, tokenizer=AutoTokenizer.from_pretrained(CLIPSEG_ID))
        return _clipseg


def lpips_model() -> Lpips:
    global _lpips
    with _lock:
        if _lpips is None:
            configure_torch()
            import lpips

            net = lpips.LPIPS(net="alex", spatial=True, verbose=False)
            net.eval()
            _lpips = Lpips(net=net)
        return _lpips


def face_models() -> FaceModels:
    global _faces
    with _lock:
        if _faces is None:
            detector = cv2.FaceDetectorYN.create(
                str(_weight_path(YUNET_FILE)), "", (320, 320), 0.7, 0.3, 200
            )
            recognizer = cv2.FaceRecognizerSF.create(str(_weight_path(SFACE_FILE)), "")
            _faces = FaceModels(detector=detector, recognizer=recognizer)
        return _faces


def warm_flags() -> dict[str, bool]:
    return {
        "clipseg": _clipseg is not None,
        "lpips": _lpips is not None,
        "faces": _faces is not None,
    }


def warmup() -> dict[str, bool]:
    """Load every model and push a 64-px dummy through it so the first request is not slow."""
    dummy = np.full((64, 64, 3), 127, dtype=np.uint8)
    clipseg().heatmaps([dummy], ["warm up"])
    lpips_model().spatial(dummy, dummy)
    fm = face_models()
    with _lock:
        fm.detector.setInputSize((64, 64))
        fm.detector.detect(cv2.cvtColor(dummy, cv2.COLOR_RGB2BGR))
    return warm_flags()


def detector_lock() -> threading.RLock:
    """OpenCV's DNN face models are not thread-safe; callers serialize on this lock."""
    return _lock
