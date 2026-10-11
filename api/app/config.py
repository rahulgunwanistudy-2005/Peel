"""Settings read once from the environment."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"


def _git_sha() -> str:
    sha = os.environ.get("PEEL_GIT_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _origins() -> list[str]:
    raw = os.environ.get("PEEL_CORS_ORIGINS", "")
    extra = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", *extra]


@dataclass(frozen=True)
class Settings:
    git_sha: str = field(default_factory=_git_sha)
    cors_origins: list[str] = field(default_factory=_origins)
    # The deployed web app (any Vercel preview or .xyz domain), Spaces and n8n Cloud.
    cors_origin_regex: str = os.environ.get(
        "PEEL_CORS_REGEX",
        r"https://([a-z0-9-]+\.)*(vercel\.app|hf\.space|n8n\.cloud|xyz)",
    )
    examples_dir: Path = Path(
        os.environ.get("PEEL_EXAMPLES_DIR", REPO_ROOT / "api" / "examples_data")
    )
    web_dist: Path = Path(os.environ.get("PEEL_WEB_DIST", REPO_ROOT / "web" / "dist"))
    job_ttl_seconds: int = int(os.environ.get("PEEL_JOB_TTL", 30 * 60))
    max_jobs: int = int(os.environ.get("PEEL_MAX_JOBS", 16))
    max_chain_steps: int = int(os.environ.get("PEEL_MAX_CHAIN_STEPS", 8))
    rate_capacity: int = int(os.environ.get("PEEL_RATE_CAPACITY", 12))
    rate_refill_seconds: float = float(os.environ.get("PEEL_RATE_REFILL", 5.0))
    warm_on_start: bool = os.environ.get("PEEL_WARM", "1") != "0"
    gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    edit_model: str | None = os.environ.get("PEEL_EDIT_MODEL")


settings = Settings()
