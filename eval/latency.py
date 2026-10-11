"""Per-stage latency, p50 and p95, on a 1024-px pair.

    python -m eval.latency --api https://<space>.hf.space    # measure the deployed hardware
    python -m eval.latency --local --threads 2               # this machine, limited to 2 threads

Against an API, each request gets a negligibly different option value so the content-hash cache
cannot answer it.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any

import numpy as np

from eval.common import CACHE, fit_long_side, git_sha, now, read_rgb, write_result

PAIR = CACHE / "banana100" / "table_add_apple_denoise_run1"


def percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(float(np.percentile(values, 50)), 1),
        "p95": round(float(np.percentile(values, 95)), 1),
    }


def summarize(runs: list[dict[str, float]], wall: list[float]) -> dict[str, Any]:
    stages = list(runs[0])
    return {
        "stages": [{"stage": s, **percentiles([r[s] for r in runs])} for s in stages],
        "total": percentiles([sum(r.values()) for r in runs]),
        "wall": percentiles(wall),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args()

    from peel.io import encode_png

    original = fit_long_side(read_rgb(PAIR / "step_000.png"), 1024)
    edited = fit_long_side(read_rgb(PAIR / "step_001.png"), 1024)
    instruction = json.loads((PAIR / "manifest.json").read_text())["steps"][0]["instruction"]
    orig_png, edit_png = encode_png(original), encode_png(edited)
    runs: list[dict[str, float]] = []
    wall: list[float] = []

    if args.api:
        import httpx

        base = args.api.rstrip("/")
        health = httpx.get(f"{base}/healthz", timeout=60).json()
        hardware = f"{base.split('//')[1]} ({health.get('cpus', '?')} vCPU, {health.get('platform', 'unknown')})"
        with httpx.Client(timeout=180) as client:
            for i in range(args.n + 1):
                options = json.dumps({"min_area_pct": 0.05 + i * 1e-7})
                start = time.perf_counter()
                response = client.post(
                    f"{base}/v1/peel",
                    files={
                        "original": ("o.png", orig_png, "image/png"),
                        "edited": ("e.png", edit_png, "image/png"),
                    },
                    data={"instruction": instruction, "options": options},
                    headers={"x-forwarded-for": f"198.51.100.{i % 250 + 1}"},
                )
                elapsed = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                if i == 0:
                    continue  # warm-up request
                runs.append({t["stage"]: t["ms"] for t in response.json()["timings"]})
                wall.append(elapsed)
                print(
                    f"{i}/{args.n}: {sum(runs[-1].values()):.0f} ms server, {elapsed:.0f} ms round trip",
                    flush=True,
                )
        name = "latency"
    else:
        import cv2
        import torch
        from peel import models
        from peel.pipeline import peel_bytes

        models.warmup()
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
        hardware = (
            f"{platform.machine()} {platform.system()} laptop, limited to {args.threads} threads"
        )
        for i in range(args.n + 1):
            start = time.perf_counter()
            output = peel_bytes(orig_png, edit_png, instruction)
            elapsed = (time.perf_counter() - start) * 1000
            if i == 0:
                continue
            runs.append({t.stage: t.ms for t in output.timings})
            wall.append(elapsed)
        name = "latency_local"

    payload = {
        "meta": {
            "hardware": hardware,
            "n": len(runs),
            "image": f"a {original.shape[1]} × {original.shape[0]} px original with a {edited.shape[1]} × {edited.shape[0]} px edit",
            "generated_at": now(),
            "git_sha": git_sha(),
        },
        **summarize(runs, wall),
    }
    write_result(name, payload)
    print(json.dumps(payload["total"]), hardware)


if __name__ == "__main__":
    main()
