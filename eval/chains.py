"""Banana100 chains: does drift still accumulate when every step is peeled?

Uses the chains fetched by ``eval.banana100`` (real Nano Banana Pro outputs). For each chain the
raw sequence and the anchored Peel sequence are measured against the step-0 image in the regions
no edit touched, at every step. Also writes the bundled chain example for the web app.

    python -m eval.chains
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from peel.chain import ChainResult, run_chain

from eval.common import (
    CACHE,
    RESULTS,
    WEB_PUBLIC,
    fit_long_side,
    git_sha,
    now,
    read_rgb,
    write_result,
    write_webp,
)

LONG_SIDE = 1024
CHECKPOINTS = (1, 5, 10, 20)
KEYS = ("raw_lpips", "peeled_lpips", "raw_psnr", "peeled_psnr", "drift_score")
WEB_CHAIN = WEB_PUBLIC / "chain"


def load_chain(folder: Path, max_steps: int) -> tuple[dict[str, Any], Any, list[Any], list[str]]:
    meta = json.loads((folder / "manifest.json").read_text())
    steps = meta["steps"][:max_steps]
    original = fit_long_side(read_rgb(folder / "step_000.png"), LONG_SIDE)
    edits = [
        fit_long_side(read_rgb(folder / f"step_{s['step']:03d}.png"), LONG_SIDE) for s in steps
    ]
    return meta, original, edits, [s["instruction"] for s in steps]


def mean_curve(chains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n_steps = min(len(c["steps"]) for c in chains)
    curve = []
    for i in range(n_steps):
        point: dict[str, Any] = {"step": chains[0]["steps"][i]["step"], "n_chains": len(chains)}
        for key in KEYS:
            point[key] = round(float(np.mean([c["steps"][i][key] for c in chains])), 4)
        curve.append(point)
    return curve


def write_web_example(meta: dict[str, Any], result: ChainResult, instructions: list[str]) -> None:
    """The chain shown on the Chains page before anyone uploads anything."""
    WEB_CHAIN.mkdir(parents=True, exist_ok=True)
    for old in WEB_CHAIN.glob("*.webp"):
        old.unlink()
    write_webp(WEB_CHAIN / "original.webp", result.original, quality=88)
    steps = []
    for step, instruction in zip(result.steps, instructions, strict=True):
        k = step.metrics.step
        write_webp(WEB_CHAIN / f"step{k:02d}_raw.webp", step.raw, quality=86)
        write_webp(WEB_CHAIN / f"step{k:02d}_peeled.webp", step.output.peeled, quality=86)
        steps.append(
            {
                "step": k,
                "instruction": instruction,
                "raw": f"./chain/step{k:02d}_raw.webp",
                "peeled": f"./chain/step{k:02d}_peeled.webp",
            }
        )
    height, width = result.original.shape[:2]
    payload = {
        "title": f"{len(steps)} chained Nano Banana edits, raw and peeled",
        "source": (
            f"Banana100 ({meta['dataset']}, {meta['license']}): {meta['model']}/{meta['image']}, chain "
            f"“{meta['chain']}”, run {meta['run']}. Real model outputs, shown at reduced size."
        ),
        "chain": {
            "id": "example",
            "original": "./chain/original.webp",
            "width": width,
            "height": height,
            "steps": steps,
            "metrics": [json.loads(s.metrics.model_dump_json()) for s in result.steps],
        },
    }
    (WEB_CHAIN / "chain.json").write_text(json.dumps(payload, indent=2) + "\n")


def plot(groups: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 3.8), squeeze=False)
    for ax, group in zip(axes[0], groups, strict=True):
        curve = group["mean_curve"]
        xs = [p["step"] for p in curve]
        ax.plot(
            xs,
            [p["raw_lpips"] for p in curve],
            color="#ec008c",
            marker="o",
            ms=3,
            label="Raw chain",
        )
        ax.plot(
            xs,
            [p["peeled_lpips"] for p in curve],
            color="#00aeef",
            marker="o",
            ms=3,
            label="Peeled chain",
        )
        ax.set_title(f"{group['label']} (n = {len(group['chains'])})", fontsize=10, loc="left")
        ax.set_xlabel("Edit step")
        ax.set_ylabel("Unedited-region LPIPS vs step 0")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(RESULTS / "plots" / f"banana100_chains.{ext}", dpi=160)
    plt.close(fig)


def main(max_steps: int = 20) -> None:
    folders = sorted((CACHE / "banana100").glob("*_run*"))
    folders = [f for f in folders if (f / "manifest.json").exists()]
    if not folders:
        write_result(
            "banana100",
            {
                "meta": {
                    "generated_at": now(),
                    "dataset": "kenantang/Banana100",
                    "license": "CC BY-NC-SA 4.0",
                    "note": "",
                },
                "groups": [],
            },
        )
        print("no cached Banana100 chains; run `python -m eval.banana100 fetch` first")
        return

    edit_chains: list[dict[str, Any]] = []
    replica_chains: list[dict[str, Any]] = []
    for folder in folders:
        meta, original, edits, instructions = load_chain(folder, max_steps)
        start = time.time()
        result = run_chain(original, edits, instructions)
        record = {
            "image": meta["image"],
            "chain": meta["chain"],
            "run": meta["run"],
            "instruction": instructions[0],
            "seconds": round(time.time() - start, 1),
            "steps": [json.loads(s.metrics.model_dump_json()) for s in result.steps],
        }
        is_replica = meta["chain"].startswith("replica")
        (replica_chains if is_replica else edit_chains).append(record)
        last = record["steps"][-1]
        print(
            f"{meta['image']}/{meta['chain']}: {len(edits)} steps in {record['seconds']}s | step {last['step']}: "
            f"raw LPIPS {last['raw_lpips']} PSNR {last['raw_psnr']} -> peeled LPIPS {last['peeled_lpips']} PSNR {last['peeled_psnr']}",
            flush=True,
        )
        if (
            not is_replica
            and not (WEB_CHAIN / "chain.json").exists()
            or (not is_replica and meta["image"] == "table")
        ):
            write_web_example(meta, result, instructions)

    groups = []
    if edit_chains:
        groups.append(
            {
                "id": "edit",
                "label": "Edit chain: one object added per step",
                "note": (
                    "Each step asks for one more apple. The raw chain feeds every output back into the "
                    "editor; the peeled chain keeps each new apple and anchors everything else to step 0."
                ),
                "chains": edit_chains,
            }
        )
    if replica_chains:
        groups.append(
            {
                "id": "replica",
                "label": "Replica chains: “produce an exact replica”",
                "note": (
                    "Nothing is requested, so the correct output at every step is the step-0 image. Peel "
                    "returns it byte for byte by construction: this group shows the size of the raw drift, "
                    "not a hard test of Peel."
                ),
                "chains": replica_chains,
            }
        )
    for group in groups:
        group["mean_curve"] = mean_curve(group["chains"])
        group["checkpoints"] = [p for p in group["mean_curve"] if p["step"] in CHECKPOINTS]
    write_result(
        "banana100",
        {
            "meta": {
                "generated_at": now(),
                "git_sha": git_sha(),
                "dataset": "kenantang/Banana100",
                "license": "CC BY-NC-SA 4.0",
                "long_side": LONG_SIDE,
                "note": (
                    "Images were read at reduced size (long side 1280, evaluated at 1024). Only chains at "
                    "the start of a parquet row group can be fetched cheaply, which is why these chains "
                    "were chosen; see eval/BANANA100_NOTES.md."
                ),
            },
            "groups": groups,
        },
    )
    plot(groups)


if __name__ == "__main__":
    main()
