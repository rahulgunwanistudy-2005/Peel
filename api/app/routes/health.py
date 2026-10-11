from __future__ import annotations

import os
import platform

from fastapi import APIRouter
from peel import models

from app import gemini, service
from app.config import VERSION, settings
from app.limits import MAX_BYTES
from app.schemas import Health, Limits

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
        cpus=len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1),
        platform=f"{platform.system()} {platform.machine()}",
        limits=Limits(
            max_upload_bytes=min(MAX_BYTES, settings.max_upload_bytes),
            max_request_bytes=settings.max_request_bytes,
            max_long_side=settings.max_long_side,
            max_chain_steps=settings.max_chain_steps,
        ),
        stateless=settings.stateless,
    )
