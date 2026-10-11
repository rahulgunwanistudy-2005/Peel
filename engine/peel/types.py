"""Public data contract for the Peel engine.

Everything the API returns and the web app renders is defined here. Coordinates
(``bbox``, ``polygon``) are in full-resolution pixels of the original image.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RegionType = Literal["color_cast", "texture", "structure", "face_identity", "text", "object"]
Classification = Literal["intended", "collateral", "ambiguous"]
Decision = Literal["keep", "revert"]
Confidence = Literal["high", "medium", "low"]
Mode = Literal["local", "global"]
BlendMode = Literal["laplacian", "poisson"]

STAGES: tuple[str, ...] = (
    "decode",
    "register",
    "change",
    "intent",
    "faces",
    "classify",
    "composite",
    "ledger",
)


class PeelOptions(BaseModel):
    """Tunable knobs. Defaults are the values tuned on the synthetic drift suite."""

    model_config = ConfigDict(extra="forbid")

    work_long_side: int = Field(768, ge=256, le=1536, description="Analysis resolution.")
    max_long_side: int = Field(4096, ge=256, le=4096, description="Full-resolution cap.")
    k_mad: float = Field(6.0, ge=1.0, le=20.0, description="Change threshold: median + k*MAD.")
    min_area_pct: float = Field(0.05, ge=0.0, le=5.0, description="Drop regions smaller than this.")
    intent_threshold: float = Field(0.35, ge=0.05, le=0.95)
    intended_overlap: float = Field(0.6, ge=0.0, le=1.0)
    collateral_overlap: float = Field(0.2, ge=0.0, le=1.0)
    face_distance_threshold: float = Field(
        0.30, ge=0.0, le=2.0, description="SFace cosine distance above which a face has drifted."
    )
    blend: BlendMode = "laplacian"
    harmonize: bool = True
    keep_ambiguous: bool = True
    overrides: dict[str, Decision] = Field(
        default_factory=dict, description="Per-region decisions that win over the classifier."
    )
    # Ablation switches (all on in the shipped product; the eval suite turns them off one by one).
    use_reregistration: bool = True
    use_intent: bool = True
    use_cast_separation: bool = True
    use_face_lock: bool = True


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Region(BaseModel):
    id: str
    bbox: BBox
    polygon: list[tuple[int, int]] = Field(description="Simplified outer contour.")
    area_pct: float
    type: RegionType
    classification: Classification
    decision: Decision
    overridden: bool = False
    confidence: Confidence
    intent_overlap: float
    mean_delta_e: float = Field(description="Mean CIEDE2000 inside the region.")
    mean_lpips: float
    face_distance: float | None = None
    sentence: str = Field(description="One plain-language sentence describing the change.")


class GlobalCast(BaseModel):
    detected: bool
    delta_l: float
    delta_a: float
    delta_b: float
    magnitude: float = Field(description="Euclidean length of the Lab shift.")
    sentence: str


class RegistrationReport(BaseModel):
    confidence: Confidence
    inlier_ratio: float
    reprojection_error_px: float
    n_matches: int
    shift_px: float = Field(description="Largest corner displacement that was corrected.")
    flow_applied: bool
    residual_flow_px: float
    resized: bool = Field(description="True when the edit came back at a different size.")
    sentence: str


class LedgerSummary(BaseModel):
    n_regions: int
    n_intended: int
    n_collateral: int
    n_ambiguous: int
    changed_area_pct: float
    kept_area_pct: float
    peeled_area_pct: float
    bit_identical_pct: float = Field(
        description="Share of output pixels that are byte-for-byte the original."
    )
    background_delta_e: float = Field(
        description="Mean CIEDE2000 of the raw edit against the original, outside kept regions."
    )


class Ledger(BaseModel):
    instruction: str
    target_phrases: list[str]
    mode: Mode
    drift_score: float = Field(ge=0.0, le=100.0)
    headline: str
    confidence: Confidence
    summary: LedgerSummary
    global_cast: GlobalCast
    registration: RegistrationReport
    regions: list[Region]
    warnings: list[str] = Field(default_factory=list)


class StageTiming(BaseModel):
    stage: str
    ms: float


class PeelImages(BaseModel):
    """URLs (relative to the API root) for the rendered images of one result."""

    original: str
    edited: str = Field(description="The AI edit after alignment to the original.")
    peeled: str
    change_map: str
    diff: str


class PeelResult(BaseModel):
    id: str
    width: int
    height: int
    ledger: Ledger
    timings: list[StageTiming]
    total_ms: float
    images: PeelImages
    cached: bool = False


class ChainStepMetrics(BaseModel):
    step: int
    instruction: str
    raw_lpips: float = Field(description="Unedited-region LPIPS of the raw chain vs step 0.")
    peeled_lpips: float
    raw_psnr: float
    peeled_psnr: float
    drift_score: float


class ErrorCode:
    NOT_SAME_SCENE = "NOT_SAME_SCENE"
    INVALID_IMAGE = "INVALID_IMAGE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    EMPTY_INSTRUCTION = "EMPTY_INSTRUCTION"
    INVALID_OPTIONS = "INVALID_OPTIONS"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    EDIT_UNAVAILABLE = "EDIT_UNAVAILABLE"
    CHAIN_TOO_LONG = "CHAIN_TOO_LONG"
    INTERNAL = "INTERNAL"


class PeelError(Exception):
    """A typed failure with a message written for the person using the app."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
