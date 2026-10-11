from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from peel.types import ErrorCode, PeelError
from starlette.concurrency import run_in_threadpool

from app import service
from app.cache import Job, store
from app.limits import rate_limit, read_upload
from app.schemas import ErrorBody, JobCreated, JobStatus, PeelResult, RecompositeRequest

router = APIRouter(prefix="/v1", tags=["peel"])

ERRORS = {
    413: {"model": ErrorBody, "description": "An upload is over 15 MB."},
    415: {"model": ErrorBody, "description": "Not a PNG, JPEG or WebP."},
    422: {
        "model": ErrorBody,
        "description": "NOT_SAME_SCENE, INVALID_OPTIONS or EMPTY_INSTRUCTION.",
    },
    429: {"model": ErrorBody, "description": "Rate limited."},
}

Original = Annotated[UploadFile, File(description="The image before the AI edit.")]
Edited = Annotated[UploadFile, File(description="The image the AI editor returned.")]
Instruction = Annotated[str, Form(description="What you asked the editor to do.", max_length=2000)]
Options = Annotated[str | None, Form(description="PeelOptions as a JSON string (optional).")]


def _job_or_404(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise PeelError(
            ErrorCode.JOB_NOT_FOUND,
            "This result has expired. Results are kept in memory for 30 minutes; run it again.",
            status=404,
        )
    return job


def _status(job: Job) -> JobStatus:
    return JobStatus(
        id=job.id,
        kind=job.kind,
        state=job.state,
        progress=job.progress,
        result=job.result,
        chain=job.chain,
        error=job.error,
    )


@router.post(
    "/peel",
    response_model=PeelResult,
    responses=ERRORS,
    summary="Peel an edit and wait for the result",
    dependencies=[Depends(rate_limit())],
)
async def peel(
    original: Original, edited: Edited, instruction: Instruction, options: Options = None
) -> PeelResult:
    original_bytes = await read_upload(original, "original image")
    edited_bytes = await read_upload(edited, "edited image")
    parsed = service.parse_options(options)
    job = store.create("peel")
    try:
        return await run_in_threadpool(
            service.run_peel, job, original_bytes, edited_bytes, instruction, parsed
        )
    except Exception as exc:
        service.fail(job, exc)
        raise


@router.post(
    "/peel/jobs",
    response_model=JobCreated,
    status_code=202,
    responses=ERRORS,
    summary="Start a peel and follow its progress over Server-Sent Events",
    dependencies=[Depends(rate_limit())],
)
async def create_job(
    original: Original, edited: Edited, instruction: Instruction, options: Options = None
) -> JobCreated:
    original_bytes = await read_upload(original, "original image")
    edited_bytes = await read_upload(edited, "edited image")
    parsed = service.parse_options(options)
    if not instruction.strip():
        raise PeelError(
            ErrorCode.EMPTY_INSTRUCTION,
            "Tell Peel what you asked the editor to do, for example “make the jacket red”.",
            status=422,
        )
    job = store.create("peel")
    job.emit("queued", 0.0, service.STAGE_LABELS["queued"])
    service.submit(
        job, lambda j: service.run_peel(j, original_bytes, edited_bytes, instruction, parsed)
    )
    return JobCreated(
        id=job.id,
        kind="peel",
        status_url=f"/v1/jobs/{job.id}",
        events_url=f"/v1/jobs/{job.id}/events",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    responses={404: {"model": ErrorBody}},
    summary="Job state and, when finished, the result",
)
def job_status(job_id: str) -> JobStatus:
    return _status(_job_or_404(job_id))


@router.get(
    "/jobs/{job_id}/events",
    summary="Stage progress as Server-Sent Events",
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Events: `progress` (ProgressEvent), then `done` (JobStatus) or `error` (ErrorDetail).",
        },
        404: {"model": ErrorBody},
    },
)
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    job = _job_or_404(job_id)

    def frame(event: str, payload: str) -> str:
        return f"event: {event}\ndata: {payload}\n\n"

    async def stream() -> AsyncIterator[str]:
        sent = 0
        idle = 0
        while True:
            while sent < len(job.events):
                event = job.events[sent]
                sent += 1
                if event.stage not in ("done", "error"):
                    yield frame("progress", event.model_dump_json())
            if job.state == "done":
                yield frame("done", _status(job).model_dump_json())
                return
            if job.state == "error" and job.error is not None:
                yield frame("error", job.error.model_dump_json())
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.08)
            idle += 1
            if idle % 100 == 0:
                yield ": keep-alive\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@router.get(
    "/jobs/{job_id}/images/{name}.png",
    summary="One rendered image of a result",
    responses={200: {"content": {"image/png": {}}}, 404: {"model": ErrorBody}},
)
def job_image(
    job_id: str,
    name: str,
    max: Annotated[
        int,
        Query(
            ge=0, le=4096, description="Downscale so the long side is at most this (0 = full size)."
        ),
    ] = 0,
    download: Annotated[bool, Query(description="Send as an attachment.")] = False,
) -> Response:
    job = _job_or_404(job_id)
    data = service.render_image(job, name, max)
    headers = {"Cache-Control": "private, max-age=1800"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="peel-{name}.png"'
    return Response(content=data, media_type="image/png", headers=headers)


@router.post(
    "/recomposite",
    response_model=PeelResult,
    responses={404: {"model": ErrorBody}, 422: {"model": ErrorBody}},
    summary="Flip regions between keep and revert without re-analysing",
    dependencies=[Depends(rate_limit(0.25))],
)
def recomposite(body: RecompositeRequest) -> PeelResult:
    job = _job_or_404(body.job_id)
    if job.kind != "peel" or job.state != "done":
        raise PeelError(
            ErrorCode.JOB_NOT_FOUND, "That job has no finished result to recomposite.", status=404
        )
    return service.run_recomposite(job, body.overrides)


def parse_instruction_list(raw: str) -> list[str]:
    """Instructions for a chain: a JSON array, or one instruction per line."""
    text = raw.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise PeelError(
                ErrorCode.INVALID_OPTIONS,
                "The instructions are not a valid JSON array.",
                status=422,
            ) from exc
        if not isinstance(parsed, list) or not all(isinstance(i, str) for i in parsed):
            raise PeelError(
                ErrorCode.INVALID_OPTIONS,
                "The instructions must be a JSON array of strings.",
                status=422,
            )
        return [i.strip() for i in parsed]
    return [line.strip() for line in text.splitlines() if line.strip()]
