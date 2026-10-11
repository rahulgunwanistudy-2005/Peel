"""Synthetic drift suite and ablations: ground truth by construction.

    python -m eval.synthetic              # full suite (about 216 cases x 7 methods)
    python -m eval.synthetic --limit 24   # quick pilot

Writes eval/results/synthetic.json (aggregates), synthetic_cases.json (every case) and plots.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from typing import Any

import cv2
import numpy as np
from peel.arrays import U8, Bool, dilate, resize_mask
from peel.chain import masked_lpips
from peel.pipeline import peel_arrays
from peel.synth import DriftKind, EditKind, SynthCase, make_case
from peel.types import PeelError, PeelOptions
from skimage import data

from eval.common import CACHE, RESULTS, fit_long_side, git_sha, now, read_rgb, write_result

LONG_SIDE = 448
PSNR_CAP = 60.0  # identical pixels would be infinite; capped so means stay meaningful
ASTRONAUT_FACE = (140, 44, 131, 149)  # x, y, w, h at 448 px

KINDS: list[EditKind] = ["add", "remove", "recolor", "replace"]
RECIPES: list[list[DriftKind]] = [
    ["cast", "shift"],
    ["jpeg", "patch_color"],
    ["blur", "patch_blur", "cast"],
    ["noise", "shift", "patch_warp"],
    ["sharpen", "patch_color", "shift"],
    ["cast", "jpeg", "patch_blur", "shift"],
]
VARIANTS: list[tuple[str, str, str, dict[str, bool] | None]] = [
    ("raw", "Raw AI edit", "No repair at all.", None),
    (
        "diff_merge",
        "Diff-merge baseline",
        "Composite every pixel whose difference exceeds a threshold.",
        None,
    ),
    ("peel", "Peel", "Full pipeline.", {}),
    (
        "no_rereg",
        "Peel without re-registration",
        "No background-locked refit, no flow.",
        {"use_reregistration": False},
    ),
    (
        "no_intent",
        "Peel without intent grounding",
        "Every changed region is kept.",
        {"use_intent": False},
    ),
    (
        "no_cast",
        "Peel without cast separation",
        "The whole-image tint is not removed first.",
        {"use_cast_separation": False},
    ),
    (
        "no_face",
        "Peel without face lock",
        "Faces are treated like any other region.",
        {"use_face_lock": False},
    ),
]
METRIC_INFO = [
    {
        "key": "intended_iou",
        "label": "Intended-region IoU",
        "better": "higher",
        "digits": 3,
        "unit": "",
    },
    {
        "key": "collateral_reverted_pct",
        "label": "Collateral pixels reverted",
        "better": "higher",
        "digits": 1,
        "unit": "%",
    },
    {
        "key": "bit_identical_pct",
        "label": "Byte-identical outside the edit",
        "better": "higher",
        "digits": 1,
        "unit": "%",
    },
    {
        "key": "unedited_psnr",
        "label": "Unedited-region PSNR",
        "better": "higher",
        "digits": 1,
        "unit": "dB, capped at 60",
    },
    {
        "key": "unedited_lpips",
        "label": "Unedited-region LPIPS",
        "better": "lower",
        "digits": 4,
        "unit": "",
    },
    {
        "key": "edit_psnr",
        "label": "Kept-edit fidelity PSNR",
        "better": "higher",
        "digits": 1,
        "unit": "dB",
    },
]


def bases() -> list[tuple[str, U8, tuple[int, int, int, int] | None]]:
    out: list[tuple[str, U8, tuple[int, int, int, int] | None]] = [
        (
            "astronaut (NASA, public domain)",
            fit_long_side(np.ascontiguousarray(data.astronaut()), LONG_SIDE),
            ASTRONAUT_FACE,
        ),
        ("coffee (CC0)", fit_long_side(np.ascontiguousarray(data.coffee()), LONG_SIDE), None),
        (
            "chelsea the cat (CC0)",
            fit_long_side(np.ascontiguousarray(data.chelsea()), LONG_SIDE),
            None,
        ),
        ("rocket (CC0)", fit_long_side(np.ascontiguousarray(data.rocket()), LONG_SIDE), None),
    ]
    for folder in sorted((CACHE / "banana100").glob("*_run1")):
        name = folder.name.split("_")[0]
        out.append(
            (
                f"Banana100 input: {name} (CC BY-NC-SA 4.0)",
                fit_long_side(read_rgb(folder / "step_000.png"), LONG_SIDE),
                None,
            )
        )
    return out


def case_specs() -> list[tuple[int, int, int, int]]:
    """(base index, kind index, recipe index, seed) for every case."""
    n_bases = len(bases())
    specs = []
    seed = 1000
    for b in range(n_bases):
        for k in range(len(KINDS)):
            for r in range(len(RECIPES)):
                specs.append((b, k, r, seed))
                seed += 1
    return specs


def psnr(a: U8, b: U8, mask: Bool) -> float:
    if not mask.any():
        return PSNR_CAP
    diff = a[mask].astype(np.float64) - b[mask].astype(np.float64)
    mse = float(np.mean(diff * diff))
    return PSNR_CAP if mse < 1e-10 else float(min(PSNR_CAP, 10.0 * np.log10(255.0**2 / mse)))


def diff_merge(original: U8, edited: U8, threshold: float = 13.0) -> tuple[U8, Bool]:
    """What existing diff-merge nodes do: keep every pixel that differs enough, feathered.

    No registration, no notion of which change was asked for.
    """
    diff = cv2.absdiff(
        cv2.GaussianBlur(original, (0, 0), 1.0), cv2.GaussianBlur(edited, (0, 0), 1.0)
    ).max(axis=2)
    mask = dilate(diff > threshold, 3)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)[..., None]
    out = edited.astype(np.float32) * soft + original.astype(np.float32) * (1.0 - soft)
    out = np.where(soft > 0, np.clip(np.rint(out), 0, 255), original).astype(np.uint8)
    return out, mask


def measure(case: SynthCase, output: U8, kept: Bool) -> dict[str, float]:
    intended = case.intended
    outside = ~dilate(intended, 3)
    drifted = outside & np.any(case.edited != case.original, axis=-1)
    identical = np.all(output == case.original, axis=-1)
    union = (kept | intended).sum()
    return {
        "intended_iou": float((kept & intended).sum() / union) if union else 1.0,
        "collateral_reverted_pct": 100.0 * float(identical[drifted].mean())
        if drifted.any()
        else 100.0,
        "bit_identical_pct": 100.0 * float(identical[outside].mean()),
        "unedited_psnr": psnr(output, case.original, outside),
        "unedited_lpips": masked_lpips(output, case.original, outside),
        "edit_psnr": psnr(output, case.ideal, intended),
    }


def run_case(spec: tuple[int, int, int, int]) -> dict[str, Any]:
    base_index, kind_index, recipe_index, seed = spec
    name, base, face_box = _BASES[base_index]
    case = make_case(base, seed, KINDS[kind_index], RECIPES[recipe_index], face_box=face_box)
    h, w = case.original.shape[:2]
    row: dict[str, Any] = {
        "base": name,
        "kind": case.kind,
        "drifts": case.drifts,
        "instruction": case.instruction,
        "seed": seed,
        "metrics": {},
        "ms": {},
    }
    for variant, _, _, flags in VARIANTS:
        start = time.perf_counter()
        if variant == "raw":
            output, kept = case.edited, np.ones((h, w), dtype=bool)
        elif variant == "diff_merge":
            output, kept = diff_merge(case.original, case.edited)
        else:
            try:
                result = peel_arrays(
                    case.original, case.edited, case.instruction, PeelOptions(**(flags or {}))
                )
            except PeelError as exc:
                row["metrics"][variant] = None
                row.setdefault("errors", {})[variant] = exc.code
                continue
            output = result.peeled
            small = np.zeros(result.analysis.change.mask.shape, dtype=bool)
            for rm in result.analysis.regions:
                if rm.region.decision == "keep":
                    small |= rm.mask
            kept = resize_mask(small, w, h)
        row["metrics"][variant] = measure(case, output, kept)
        row["ms"][variant] = round((time.perf_counter() - start) * 1000.0, 1)
    return row


_BASES: list[tuple[str, U8, tuple[int, int, int, int] | None]] = []


def _init() -> None:
    global _BASES
    import torch

    torch.set_num_threads(3)
    cv2.setNumThreads(3)
    _BASES = bases()


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant, label, note, _ in VARIANTS:
        values = [r["metrics"][variant] for r in rows if r["metrics"].get(variant)]
        metrics = {m["key"]: float(np.mean([v[m["key"]] for v in values])) for m in METRIC_INFO}
        out.append(
            {"id": variant, "label": label, "note": note, "n": len(values), "metrics": metrics}
        )
    return out


def breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    """Mean intended IoU and collateral-reverted share for full Peel, per edit kind or drift kind."""
    groups: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        metrics = row["metrics"].get("peel")
        if not metrics:
            continue
        labels = [row[key]] if key == "kind" else row["drifts"]
        for label in labels:
            groups.setdefault(label, []).append(metrics)
    return {
        label: {
            "n": len(items),
            "intended_iou": round(float(np.mean([i["intended_iou"] for i in items])), 3),
            "collateral_reverted_pct": round(
                float(np.mean([i["collateral_reverted_pct"] for i in items])), 1
            ),
        }
        for label, items in sorted(groups.items())
    }


def plot(variants: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["diff_merge", "no_intent", "no_rereg", "no_cast", "no_face", "peel"]
    rows = [next(v for v in variants if v["id"] == i) for i in order]
    labels = [r["label"].replace("Peel without ", "− ").replace(" baseline", "") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, title in zip(
        axes,
        ("intended_iou", "collateral_reverted_pct", "unedited_lpips"),
        (
            "Intended-region IoU (higher is better)",
            "Collateral pixels reverted, % (higher is better)",
            "Unedited-region LPIPS (lower is better)",
        ),
        strict=True,
    ):
        values = [r["metrics"][key] for r in rows]
        colors = [
            "#ec008c" if r["id"] == "diff_merge" else "#00aeef" if r["id"] == "peel" else "#8c8c8c"
            for r in rows
        ]
        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(RESULTS / "plots" / f"ablation.{ext}", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Run only N evenly spaced cases.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    specs = case_specs()
    if args.limit:
        step = max(1, len(specs) // args.limit)
        specs = specs[::step][: args.limit]
    start = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        rows = []
        for i, row in enumerate(pool.imap(run_case, specs), start=1):
            rows.append(row)
            if i % 12 == 0 or i == len(specs):
                peel = [r["metrics"]["peel"] for r in rows if r["metrics"].get("peel")]
                print(
                    f"{i}/{len(specs)} cases, {time.time() - start:.0f}s | peel IoU "
                    f"{np.mean([p['intended_iou'] for p in peel]):.3f}, reverted "
                    f"{np.mean([p['collateral_reverted_pct'] for p in peel]):.1f}%",
                    flush=True,
                )
    variants = aggregate(rows)
    meta = {
        "generated_at": now(),
        "git_sha": git_sha(),
        "n_cases": len(rows),
        "long_side": LONG_SIDE,
        "bases": [name for name, _, _ in bases()],
        "edit_kinds": KINDS,
        "drift_kinds": sorted({d for recipe in RECIPES for d in recipe}),
        "psnr_cap_db": PSNR_CAP,
        "errors": sum(1 for r in rows if r.get("errors")),
        "seconds": round(time.time() - start, 1),
    }
    name = "synthetic" if not args.limit else "synthetic_pilot"
    payload = {
        "meta": meta,
        "metric_info": METRIC_INFO,
        "variants": variants,
        "peel_by_edit_kind": breakdown(rows, "kind"),
        "peel_by_drift_kind": breakdown(rows, "drifts"),
    }
    if args.limit:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    else:
        write_result(name, payload)
        (RESULTS / "synthetic_cases.json").write_text(json.dumps(rows) + "\n")
        plot(variants)
    for v in variants:
        print(f"{v['label']:34}", {k: round(x, 3) for k, x in v["metrics"].items()})
    print("by kind", payload["peel_by_edit_kind"])
    print("by drift", payload["peel_by_drift_kind"])


if __name__ == "__main__":
    main()
