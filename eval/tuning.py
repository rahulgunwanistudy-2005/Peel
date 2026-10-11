"""Threshold sweep on a subset of the synthetic suite: how k (in median + k·MAD) was chosen.

python -m eval.tuning
"""

from __future__ import annotations

import json

import numpy as np
import peel.change as change_module
from peel.arrays import resize_mask
from peel.pipeline import peel_arrays
from peel.synth import make_case
from peel.types import PeelOptions

import eval.synthetic as suite
from eval.common import RESULTS, git_sha, now

GRID = [(4.0, 0.3), (4.0, 0.4), (6.0, 0.4), (6.0, 0.6), (8.0, 0.4), (8.0, 0.6)]


def main() -> None:
    suite._init()
    specs = suite.case_specs()[::5]
    cases = [
        make_case(
            suite._BASES[b][1], seed, suite.KINDS[k], suite.RECIPES[r], face_box=suite._BASES[b][2]
        )
        for b, k, r, seed in specs
    ]
    default_ceiling = change_module.THRESHOLD_CEIL
    rows = []
    for k_mad, ceiling in GRID:
        change_module.THRESHOLD_CEIL = ceiling
        metrics = []
        for case in cases:
            out = peel_arrays(
                case.original, case.edited, case.instruction, PeelOptions(k_mad=k_mad)
            )
            small = np.zeros(out.analysis.change.mask.shape, dtype=bool)
            for rm in out.analysis.regions:
                if rm.region.decision == "keep":
                    small |= rm.mask
            kept = resize_mask(small, case.original.shape[1], case.original.shape[0])
            metrics.append(suite.measure(case, out.peeled, kept))
        row = {
            "k_mad": k_mad,
            "threshold_ceiling": ceiling,
            "is_default": k_mad == PeelOptions().k_mad and ceiling == default_ceiling,
            "lost_edits": sum(1 for m in metrics if m["intended_iou"] < 0.1),
            **{key: round(float(np.mean([m[key] for m in metrics])), 4) for key in metrics[0]},
        }
        rows.append(row)
        print(row, flush=True)
    change_module.THRESHOLD_CEIL = default_ceiling
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"generated_at": now(), "git_sha": git_sha(), "n_cases": len(cases)},
        "grid": rows,
    }
    (RESULTS / "tuning.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
