"""Push the API and web build inputs to a Hugging Face Space (Docker SDK).

    python deploy/deploy_space.py <user>/<space>

The Space builds deploy/Dockerfile itself. Needs a Hugging Face write token (`hf auth login`).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "engine/pyproject.toml",
    "engine/peel",
    "api/requirements.txt",
    "api/app",
    "api/examples_data",
    "web/package.json",
    "web/package-lock.json",
    "web/index.html",
    "web/vite.config.ts",
    "web/tsconfig.json",
    "web/tsconfig.app.json",
    "web/tsconfig.node.json",
    "web/src",
    "web/public",
]


def main() -> None:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        raise SystemExit("usage: python deploy/deploy_space.py <user>/<space>")
    repo_id = sys.argv[1]
    sha = (
        subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for item in INCLUDE:
            source, target = ROOT / item, stage / item
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
                )
            else:
                shutil.copy2(source, target)
        dockerfile = (
            (ROOT / "deploy" / "Dockerfile")
            .read_text()
            .replace("ARG PEEL_GIT_SHA=unknown", f"ARG PEEL_GIT_SHA={sha}")
        )
        (stage / "Dockerfile").write_text(dockerfile)
        shutil.copy2(ROOT / "deploy" / "space_README.md", stage / "README.md")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=stage,
            commit_message=f"Deploy Peel {sha}",
            delete_patterns=["*"],
        )
    print(f"deployed {sha} to https://huggingface.co/spaces/{repo_id}")
    print(f"app URL: https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
