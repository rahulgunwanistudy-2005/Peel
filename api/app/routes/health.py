from __future__ import annotations

from fastapi import APIRouter
from peel import models

from app import gemini, service
from app.config import VERSION, settings
from app.schemas import Health

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health, summary="Liveness, model warm flags and version")
def healthz() -> Health:
    flags = models.warm_flags()
    return Health(
        status="ok" if all(flags.values()) else "warming",
        version=VERSION,
        git_sha=settings.git_sha,
        models=flags,
        edit=gemini.status(),
        examples=len(service.examples()),
    )
