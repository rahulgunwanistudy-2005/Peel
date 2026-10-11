"""Hand-made Nano Banana pairs with a rough mask of the intended region.

Layout: eval/pairs/<name>/{original,edited,mask}.png + instruction.txt. The mask is white where
the edit was meant to land. If no pairs are present, the result says so; nothing is invented.

    python -m eval.real_pairs
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from peel.arrays import dilate, resize_mask
from peel.pipeline import peel_arrays
from peel.types import PeelError

from eval.common import ROOT, git_sha, now, read_rgb, write_result

PAIRS = ROOT / "eval" / "pairs"


def main() -> None:
    folders = (
        sorted(
            f
            for f in PAIRS.iterdir()
            if f.is_dir()
            and all(
                (f / n).exists()
                for n in ("original.png", "edited.png", "mask.png", "instruction.txt")
            )
        )
        if PAIRS.exists()
        else []
    )
    if not folders:
        write_result(
            "real_pairs",
            {
                "meta": {"n": 0, "generated_at": now(), "git_sha": git_sha()},
                "skipped": "No hand-labelled pairs were in eval/pairs/ when the evaluation ran.",
            },
        )
        print("no pairs in eval/pairs; wrote a skipped result")
        return

    rows: list[dict[str, Any]] = []
    for folder in folders:
        original = read_rgb(folder / "original.png")
        edited = read_rgb(folder / "edited.png")
        instruction = (folder / "instruction.txt").read_text().strip()
        mask_img = cv2.imread(str(folder / "mask.png"), cv2.IMREAD_GRAYSCALE)
        h, w = original.shape[:2]
        truth = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST) > 127
        try:
            out = peel_arrays(original, edited, instruction)
        except PeelError as exc:
            rows.append({"name": folder.name, "instruction": instruction, "error": exc.code})
            continue
        shape = out.analysis.change.mask.shape
        kept = np.zeros(shape, dtype=bool)
        for rm in out.analysis.regions:
            if rm.region.decision == "keep":
                kept |= rm.mask
        kept_full = resize_mask(kept, w, h)
        reverted = resize_mask(out.analysis.change.mask & ~kept, w, h)
        loose = dilate(truth, max(2, round(0.01 * max(h, w))))  # the hand mask is rough
        rows.append(
            {
                "name": folder.name,
                "instruction": instruction,
                "intended_recall": round(float((kept_full & truth).sum() / max(1, truth.sum())), 4),
                "collateral_revert_precision": round(
                    float((reverted & ~loose).sum() / max(1, reverted.sum())), 4
                )
                if reverted.any()
                else 1.0,
                "drift_score": out.ledger.drift_score,
                "n_regions": out.ledger.summary.n_regions,
                "headline": out.ledger.headline,
            }
        )
        print(rows[-1], flush=True)
    ok = [r for r in rows if "error" not in r]
    summary = (
        {
            "intended_recall": round(float(np.mean([r["intended_recall"] for r in ok])), 4),
            "collateral_revert_precision": round(
                float(np.mean([r["collateral_revert_precision"] for r in ok])), 4
            ),
            "mean_drift_score": round(float(np.mean([r["drift_score"] for r in ok])), 2),
        }
        if ok
        else None
    )
    payload: dict[str, Any] = {
        "meta": {"n": len(ok), "generated_at": now(), "git_sha": git_sha()},
        "pairs": rows,
    }
    if summary:
        payload["summary"] = summary
    else:
        payload["skipped"] = "Every pair failed to process; see the per-pair errors."
    write_result("real_pairs", payload)


if __name__ == "__main__":
    main()
