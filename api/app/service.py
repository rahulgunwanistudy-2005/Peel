"""Glue between HTTP and the engine: run jobs, encode images, build response models."""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import cv2
import numpy as np
from peel.arrays import U8
from peel.chain import run_chain
from peel.io import decode_image, encode_png
from peel.pipeline import PeelOutput, content_hash, peel_bytes, recomposite
from peel.types import (
    Decision,
    ErrorCode,
    PeelError,
    PeelImages,
    PeelOptions,
    PeelResult,
    StageTiming,
)
from PIL import Image

from app.cache import Job, store
from app.config import settings
from app.schemas import ChainResult, ChainStepResult, ErrorDetail, Example

log = logging.getLogger("peel.api")

IMAGE_NAMES = ("original", "edited", "peeled", "change_map", "diff", "raw_diff")
VERSIONED = {"peeled", "change_map", "diff"}

# One pipeline at a time: the box has two vCPUs and the models are not all thread-safe.
_pipeline_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peel-job")

STAGE_LABELS = {
    "queued": "Waiting for a free worker",
    "decode": "Reading the images",
    "register": "Registering the edit to the original",
    "change": "Measuring what changed",
    "intent": "Locating what you asked for",
    "faces": "Checking faces",
    "classify": "Sorting intended from collateral",
    "composite": "Compositing",
    "ledger": "Writing the ledger",
    "done": "Done",
}


def parse_options(raw: str | None) -> PeelOptions:
    if raw is None or not raw.strip():
        return PeelOptions(max_long_side=settings.max_long_side)
    try:
        options = PeelOptions.model_validate_json(raw)
        if options.max_long_side > settings.max_long_side:
            options = options.model_copy(update={"max_long_side": settings.max_long_side})
        return options
    except ValueError as exc:
        first = str(exc).splitlines()[1:3]
        detail = " ".join(line.strip() for line in first) or "it is not valid JSON"
        raise PeelError(
            ErrorCode.INVALID_OPTIONS, f"The options are not valid: {detail}.", status=422
        ) from exc


def image_url(job: Job, name: str) -> str:
    suffix = f"?v={job.version}" if name in VERSIONED and job.version else ""
    return f"/v1/jobs/{job.id}/images/{name}.png{suffix}"


def images_for(job: Job) -> PeelImages:
    return PeelImages(**{name: image_url(job, name) for name in IMAGE_NAMES})


def _attach(job: Job, output: PeelOutput) -> None:
    """Point the job at a pipeline output and drop any stale encodings."""
    job.output = output
    analysis = output.analysis
    job.renderers = {
        "original": lambda: analysis.original,
        "edited": lambda: analysis.aligned,
        "peeled": lambda: output.peeled,
        "change_map": output.change_map,
        "diff": output.diff,
        "raw_diff": output.raw_diff,
    }
    job.encoded = {k: v for k, v in job.encoded.items() if k[0] not in VERSIONED}


InlineMode = Literal["none", "dataurl", "base64"]


def inline_images(output: PeelOutput, mode: InlineMode) -> PeelImages:
    """Every image of a result as WebP inside the response (lossless for the three photographs).

    For hosts that cannot serve a follow-up request from the same process, and for callers who
    want one round trip. ``dataurl`` suits browsers; ``base64`` is the bare payload.
    """
    analysis = output.analysis
    sources: dict[str, tuple[U8, bool]] = {
        "original": (analysis.original, True),
        "edited": (analysis.aligned, True),
        "peeled": (output.peeled, True),
        "change_map": (output.change_map(), False),
        "diff": (output.diff(), False),
        "raw_diff": (output.raw_diff(), False),
    }
    encoded: dict[str, str] = {}
    for name, (rgb, lossless) in sources.items():
        ok, buf = cv2.imencode(
            ".webp", rgb[..., ::-1], [cv2.IMWRITE_WEBP_QUALITY, 101 if lossless else 86]
        )
        if not ok:
            raise PeelError(ErrorCode.INTERNAL, "Could not encode the result image.", status=500)
        payload = base64.b64encode(buf.tobytes()).decode("ascii")
        encoded[name] = f"data:image/webp;base64,{payload}" if mode == "dataurl" else payload
    return PeelImages(**encoded)


def with_inline(result: PeelResult, output: PeelOutput, mode: InlineMode) -> PeelResult:
    if mode == "none":
        return result
    return result.model_copy(update={"images": inline_images(output, mode)})


def _result(job: Job, output: PeelOutput, *, cached: bool = False) -> PeelResult:
    width, height = output.size
    return PeelResult(
        id=job.id,
        width=width,
        height=height,
        ledger=output.ledger,
        timings=output.timings,
        total_ms=round(output.total_ms, 1),
        images=images_for(job),
        cached=cached,
    )


def run_peel(
    job: Job, original: bytes, edited: bytes, instruction: str, options: PeelOptions
) -> PeelResult:
    """Run the pipeline for ``job``. Identical requests are answered from the cache."""
    key = content_hash(original, edited, instruction, options)
    hit = store.by_hash(key)
    if hit is not None and hit.output is not None and hit.id != job.id:
        _attach(job, hit.output)
        job.encoded = dict(hit.encoded)
        job.version = hit.version
        job.result = _result(job, hit.output, cached=True)
        job.state = "done"
        job.emit("done", 1.0, STAGE_LABELS["done"])
        return job.result

    def progress(stage: str, pct: float) -> None:
        job.emit(stage, pct, STAGE_LABELS.get(stage, stage))

    job.state = "running"
    with _pipeline_lock:
        output = peel_bytes(original, edited, instruction, options, progress=progress)
    _attach(job, output)
    job.result = _result(job, output)
    job.state = "done"
    store.index(job, key)
    return job.result


def fail(job: Job, exc: Exception, request_id: str | None = None) -> None:
    if isinstance(exc, PeelError):
        job.error = ErrorDetail(code=exc.code, message=exc.message, request_id=request_id)
    else:
        log.exception("job %s failed", job.id)
        job.error = ErrorDetail(
            code=ErrorCode.INTERNAL,
            message="Something went wrong on our side while processing these images.",
            request_id=request_id,
        )
    job.state = "error"
    job.emit("error", 1.0, job.error.message)


def submit(job: Job, work) -> None:  # type: ignore[no-untyped-def]
    """Queue ``work(job)`` on the single worker thread; failures land on the job."""

    def task() -> None:
        try:
            work(job)
        except Exception as exc:
            fail(job, exc)

    _executor.submit(task)


def ensure_analysed(job: Job) -> PeelOutput:
    """Examples ship pre-rendered; analyse them the first time someone flips a region."""
    with job.lock:
        if job.output is None:
            if job.source is None:
                raise PeelError(ErrorCode.JOB_NOT_FOUND, "This result has expired.", status=404)
            original, edited, instruction = job.source
            with _pipeline_lock:
                output = peel_bytes(original, edited, instruction, parse_options(None))
            _attach(job, output)
        return job.output


def run_recomposite(job: Job, overrides: dict[str, Decision]) -> PeelResult:
    output = ensure_analysed(job)
    with job.lock, _pipeline_lock:
        updated = recomposite(output, overrides)
        job.version += 1
        _attach(job, updated)
        job.result = _result(job, updated)
        job.result.timings = [StageTiming(stage="composite", ms=round(updated.total_ms, 1))]
    return job.result


def render_image(job: Job, name: str, max_side: int) -> bytes:
    """PNG bytes for one image of a job, optionally downscaled for display. Memoized."""
    key = (name, max_side)
    cached = job.encoded.get(key)
    if cached is not None:
        return cached
    renderer = job.renderers.get(name)
    if renderer is None:
        raise PeelError(
            ErrorCode.JOB_NOT_FOUND, f"No image named {name} for this result.", status=404
        )
    with job.lock:
        cached = job.encoded.get(key)
        if cached is not None:
            return cached
        rgb: U8 = renderer()
        if max_side and max(rgb.shape[:2]) > max_side:
            scale = max_side / max(rgb.shape[:2])
            size = (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = None
        if name == "peeled" and not max_side and job.output is not None:
            alpha = job.output.analysis.alpha
        data = encode_png(rgb, alpha)
        job.encoded[key] = data
        return data


# ----------------------------------------------------------------------------- chains


def run_chain_job(
    job: Job, original: bytes, edits: list[bytes], instructions: list[str], options: PeelOptions
) -> ChainResult:
    job.state = "running"
    orig = decode_image(original, max_long_side=options.max_long_side, label="original image")
    decoded = [
        decode_image(e, max_long_side=options.max_long_side, label=f"edit {i}").rgb
        for i, e in enumerate(edits, start=1)
    ]

    def on_step(index: int, total: int) -> None:
        job.emit("chain", (index - 1) / total, f"Peeling step {index} of {total}")

    with _pipeline_lock:
        result = run_chain(orig.rgb, decoded, instructions, options, on_step=on_step)

    job.renderers["original"] = lambda: orig.rgb
    steps: list[ChainStepResult] = []
    for step in result.steps:
        k = step.metrics.step
        child = store.create("peel", job_id=f"{job.id}-s{k}")
        _attach(child, step.output)
        child.result = _result(child, step.output)
        child.state = "done"
        job.renderers[f"step{k}_raw"] = lambda s=step: s.raw
        steps.append(
            ChainStepResult(
                step=k,
                instruction=step.metrics.instruction,
                raw=f"/v1/jobs/{job.id}/images/step{k}_raw.png",
                peeled=image_url(child, "peeled"),
                result=child.result,
            )
        )
    job.chain = ChainResult(
        id=job.id,
        original=f"/v1/jobs/{job.id}/images/original.png",
        width=orig.rgb.shape[1],
        height=orig.rgb.shape[0],
        steps=steps,
        metrics=[s.metrics for s in result.steps],
        total_ms=round(sum(s.output.total_ms for s in result.steps), 1),
    )
    job.state = "done"
    job.emit("done", 1.0, STAGE_LABELS["done"])
    return job.chain


# ----------------------------------------------------------------------------- examples

_examples: list[Example] = []


def _decode(data: bytes) -> U8:
    return np.ascontiguousarray(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))


def load_examples() -> list[Example]:
    """Register every precomputed example as a permanent job. No compute at request time.

    Examples are stored as WebP (lossless for the photographs) and encoded to PNG on first request.
    """
    _examples.clear()
    root = settings.examples_dir
    index = root / "index.json"
    if not index.exists():
        return _examples
    for entry in json.loads(index.read_text()):
        folder = root / entry["id"]
        try:
            result = PeelResult.model_validate_json((folder / "result.json").read_text())
            files = {name: (folder / f"{name}.webp").read_bytes() for name in IMAGE_NAMES}
            source_edit = (folder / "source_edit.webp").read_bytes()
        except (OSError, ValueError) as exc:
            log.warning("skipping example %s: %s", entry.get("id"), exc)
            continue
        job = store.create("peel", job_id=f"ex-{entry['id']}", permanent=True)
        job.renderers = {name: (lambda d=data: _decode(d)) for name, data in files.items()}
        job.source = (files["original"], source_edit, entry["instruction"])
        job.result = result.model_copy(update={"id": job.id, "images": images_for(job)})
        job.state = "done"
        _examples.append(
            Example(
                id=entry["id"],
                title=entry["title"],
                blurb=entry["blurb"],
                instruction=entry["instruction"],
                source=entry["source"],
                synthetic=bool(entry.get("synthetic", False)),
                result=job.result,
            )
        )
    return _examples


def examples() -> list[Example]:
    return _examples


def warm_examples() -> None:
    """Analyse examples in the background so the first region flip is instant too."""
    for job in store.permanent_jobs():
        try:
            ensure_analysed(job)
        except Exception:
            log.exception("could not warm example %s", job.id)
