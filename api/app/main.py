"""Peel HTTP API."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from peel import models
from peel.types import ErrorCode, PeelError

from app import gemini, service
from app.config import VERSION, settings
from app.routes import chain, edit, examples, health, peel
from app.schemas import ErrorBody, ErrorDetail

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("peel.api")

DESCRIPTION = """
**Keep the edit. Peel the drift.**

Give Peel the original image, the AI-edited image and the instruction. It returns a composite that
keeps the requested change and restores every other pixel, plus a ledger of everything the editor
changed without being asked.

Images are held in memory for 30 minutes and never written to disk. Errors are
`{"error": {"code", "message", "request_id"}}` with messages written for people.
"""


def _startup_work() -> None:
    try:
        flags = models.warmup()
        log.info("models warm: %s", flags)
    except Exception:
        log.exception("model warm-up failed")
    gemini.status()
    service.warm_examples()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    loaded = service.load_examples()
    log.info("peel %s (%s): %d examples", VERSION, settings.git_sha, len(loaded))
    if settings.warm_on_start:
        threading.Thread(target=_startup_work, name="peel-warm", daemon=True).start()
    yield


app = FastAPI(
    title="Peel",
    version=VERSION,
    description=DESCRIPTION,
    lifespan=lifespan,
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Peel-Edit-Model", "Retry-After"],
    max_age=3600,
)


@app.middleware("http")
async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    if not request.url.path.startswith(("/assets", "/v1/jobs/")) or response.status_code >= 400:
        log.info(
            "%s %s -> %d %.0fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            ms,
            request_id,
        )
    return response


def _error(request: Request, code: str, message: str, status: int) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = ErrorBody(error=ErrorDetail(code=code, message=message, request_id=request_id))
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


@app.exception_handler(PeelError)
async def peel_error(request: Request, exc: PeelError) -> JSONResponse:
    return _error(request, exc.code, exc.message, exc.status)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    missing = [str(e["loc"][-1]) for e in exc.errors() if e.get("type") == "missing"]
    if missing:
        message = f"The request is missing: {', '.join(missing)}."
    else:
        first = exc.errors()[0] if exc.errors() else {}
        message = f"The request is not valid: {first.get('msg', 'check the fields')} ({first.get('loc', ['?'])[-1]})."
    return _error(request, ErrorCode.INVALID_OPTIONS, message, 422)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return _error(
        request,
        ErrorCode.INTERNAL,
        "Something went wrong on our side while processing these images. Try again.",
        500,
    )


for module in (health, peel, chain, edit, examples):
    app.include_router(module.router)

# The same static build that Vercel serves, as a fallback URL on the Space itself.
if (settings.web_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")
