"""HTTP contract. Engine types are re-exported so OpenAPI describes them once."""

from __future__ import annotations

from typing import Literal

from peel.types import (
    ChainStepMetrics,
    Decision,
    Ledger,
    PeelImages,
    PeelOptions,
    PeelResult,
    Region,
    StageTiming,
)
from pydantic import BaseModel, Field

__all__ = [
    "ChainResult",
    "ChainStepMetrics",
    "ChainStepResult",
    "Decision",
    "EditStatus",
    "ErrorBody",
    "ErrorDetail",
    "Example",
    "Health",
    "JobCreated",
    "JobStatus",
    "Ledger",
    "Limits",
    "PeelImages",
    "PeelOptions",
    "PeelResult",
    "ProgressEvent",
    "RecompositeRequest",
    "Region",
    "StageTiming",
]

JobState = Literal["queued", "running", "done", "error"]
JobKind = Literal["peel", "chain"]


class ErrorDetail(BaseModel):
    code: str = Field(examples=["NOT_SAME_SCENE"])
    message: str = Field(description="Written for the person using the app.")
    request_id: str | None = None


class ErrorBody(BaseModel):
    error: ErrorDetail


class EditStatus(BaseModel):
    available: bool
    model: str | None = None


class Limits(BaseModel):
    max_upload_bytes: int = Field(description="Largest single image this server accepts.")
    max_request_bytes: int = Field(
        description="Largest whole request body this host accepts; 0 means no cap beyond the per-image one."
    )
    max_long_side: int = Field(
        description="Images are downscaled to this long side before analysis."
    )
    max_chain_steps: int


class Health(BaseModel):
    status: Literal["ok", "warming"]
    version: str
    git_sha: str
    models: dict[str, bool] = Field(description="Warm flag per model.")
    edit: EditStatus
    examples: int
    cpus: int = Field(description="CPU cores visible to the process.")
    platform: str
    limits: Limits


class ProgressEvent(BaseModel):
    stage: str
    pct: float = Field(ge=0.0, le=1.0)
    label: str = ""


class ChainStepResult(BaseModel):
    step: int
    instruction: str
    raw: str = Field(description="URL of the raw edit in the original's frame.")
    peeled: str
    result: PeelResult


class ChainResult(BaseModel):
    id: str
    original: str
    width: int
    height: int
    steps: list[ChainStepResult]
    metrics: list[ChainStepMetrics]
    total_ms: float


class JobCreated(BaseModel):
    id: str
    kind: JobKind
    status_url: str
    events_url: str


class JobStatus(BaseModel):
    id: str
    kind: JobKind
    state: JobState
    progress: ProgressEvent
    result: PeelResult | None = None
    chain: ChainResult | None = None
    error: ErrorDetail | None = None


class RecompositeRequest(BaseModel):
    job_id: str
    overrides: dict[str, Decision] = Field(
        default_factory=dict,
        description=(
            "Region id to keep/revert. Regions not listed go back to the classifier's decision."
        ),
        examples=[{"r2": "keep"}],
    )


class Example(BaseModel):
    id: str
    title: str
    blurb: str
    instruction: str
    source: str = Field(description="Where the image pair came from, with licence.")
    synthetic: bool = Field(
        description="True when the drift was generated in code, not by an AI editor."
    )
    result: PeelResult
