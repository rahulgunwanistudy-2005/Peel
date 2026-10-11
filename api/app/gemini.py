"""Optional "Edit with Nano Banana" step, via the google-genai SDK.

Nothing else depends on this. Without a key, or when no image model is reachable, the status is
``available: false`` and ``/v1/edit`` answers ``EDIT_UNAVAILABLE``.
"""

from __future__ import annotations

import logging
import threading

from peel.types import ErrorCode, PeelError

from app.config import settings
from app.schemas import EditStatus

log = logging.getLogger("peel.gemini")

# Preference order. Only IDs that the account's model list actually contains are used.
CANDIDATES = (
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-3-pro-image-preview",
)

_lock = threading.Lock()
_status: EditStatus | None = None
_client = None


def _pick_model(available: list[str]) -> str | None:
    names = {name.removeprefix("models/") for name in available}
    if settings.edit_model:
        wanted = settings.edit_model.removeprefix("models/")
        if wanted in names:
            return wanted
        log.warning("PEEL_EDIT_MODEL=%s is not in the model list; falling back", wanted)
    for candidate in CANDIDATES:
        if candidate in names:
            return candidate
    flash_image = sorted(n for n in names if "image" in n and "gemini" in n)
    return flash_image[-1] if flash_image else None


def status(refresh: bool = False) -> EditStatus:
    """Resolve the edit model once, from the live model list (IDs are never guessed)."""
    global _status, _client
    with _lock:
        if _status is not None and not refresh:
            return _status
        if not settings.gemini_api_key:
            _status = EditStatus(available=False)
            return _status
        try:
            from google import genai

            _client = genai.Client(api_key=settings.gemini_api_key)
            available = [m.name for m in _client.models.list() if m.name]
            model = _pick_model(available)
            _status = EditStatus(available=model is not None, model=model)
            log.info("edit model: %s", model)
        except Exception as exc:
            log.warning("Gemini unavailable: %s", exc)
            _status = EditStatus(available=False)
        return _status


def edit_image(image: bytes, mime_type: str, instruction: str) -> tuple[bytes, str]:
    """Ask the image model to apply ``instruction``. Returns (image bytes, model id)."""
    current = status()
    if not current.available or current.model is None or _client is None:
        raise PeelError(
            ErrorCode.EDIT_UNAVAILABLE,
            "Live editing is not available on this server. Upload your own AI edit instead.",
            status=503,
        )
    from google.genai import types

    try:
        response = _client.models.generate_content(
            model=current.model,
            contents=[types.Part.from_bytes(data=image, mime_type=mime_type), instruction],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as exc:
        log.warning("Gemini edit failed: %s", exc)
        raise PeelError(
            ErrorCode.EDIT_UNAVAILABLE,
            "The image model did not respond (quota or outage). Upload your own AI edit instead.",
            status=503,
        ) from exc
    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else None) or []:
            if part.inline_data is not None and part.inline_data.data:
                return bytes(part.inline_data.data), current.model
    raise PeelError(
        ErrorCode.EDIT_UNAVAILABLE,
        "The image model answered without an image (it may have refused the request). "
        "Try a different instruction or upload your own AI edit.",
        status=502,
    )
