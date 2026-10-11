from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from peel.types import ErrorCode, PeelError
from starlette.concurrency import run_in_threadpool

from app import service
from app.cache import store
from app.config import settings
from app.limits import rate_limit, read_upload
from app.routes.peel import ERRORS, parse_instruction_list
from app.schemas import ChainResult, JobCreated

router = APIRouter(prefix="/v1", tags=["chain"])

Original = Annotated[UploadFile, File(description="The step-0 image.")]
Edits = Annotated[list[UploadFile], File(description="Raw editor outputs, in order.")]
Instructions = Annotated[
    str,
    Form(
        description="One instruction per edit: a JSON array, or one per line. A single line is reused for every step."
    ),
]
Options = Annotated[str | None, Form(description="PeelOptions as a JSON string (optional).")]


async def _read(
    original: UploadFile, edited: list[UploadFile], instructions: str
) -> tuple[bytes, list[bytes], list[str]]:
    if not edited:
        raise PeelError(ErrorCode.INVALID_OPTIONS, "Upload at least one edited image.", status=422)
    if len(edited) > settings.max_chain_steps:
        raise PeelError(
            ErrorCode.CHAIN_TOO_LONG,
            f"A chain can have at most {settings.max_chain_steps} steps on this server; you sent {len(edited)}.",
            status=422,
        )
    parsed = parse_instruction_list(instructions)
    if len(parsed) == 1 and len(edited) > 1:
        parsed = parsed * len(edited)
    if len(parsed) != len(edited) or not all(parsed):
        raise PeelError(
            ErrorCode.INVALID_OPTIONS,
            f"Send one instruction per edit: got {len(parsed)} instructions for {len(edited)} edits.",
            status=422,
        )
    original_bytes = await read_upload(original, "original image")
    edits = [await read_upload(e, f"edit {i}") for i, e in enumerate(edited, start=1)]
    return original_bytes, edits, parsed


@router.post(
    "/chain",
    response_model=ChainResult,
    responses=ERRORS,
    summary="Peel a sequence of chained edits, anchored to the step-0 image",
    dependencies=[Depends(rate_limit(2.0))],
)
async def chain(
    original: Original, edited: Edits, instructions: Instructions, options: Options = None
) -> ChainResult:
    original_bytes, edits, parsed = await _read(original, edited, instructions)
    opts = service.parse_options(options)
    job = store.create("chain")
    try:
        return await run_in_threadpool(
            service.run_chain_job, job, original_bytes, edits, parsed, opts
        )
    except Exception as exc:
        service.fail(job, exc)
        raise


@router.post(
    "/chain/jobs",
    response_model=JobCreated,
    status_code=202,
    responses=ERRORS,
    summary="Start a chain and follow it over Server-Sent Events",
    dependencies=[Depends(rate_limit(2.0))],
)
async def create_chain_job(
    original: Original, edited: Edits, instructions: Instructions, options: Options = None
) -> JobCreated:
    original_bytes, edits, parsed = await _read(original, edited, instructions)
    opts = service.parse_options(options)
    job = store.create("chain")
    job.emit("queued", 0.0, service.STAGE_LABELS["queued"])
    service.submit(job, lambda j: service.run_chain_job(j, original_bytes, edits, parsed, opts))
    return JobCreated(
        id=job.id,
        kind="chain",
        status_url=f"/v1/jobs/{job.id}",
        events_url=f"/v1/jobs/{job.id}/events",
    )
