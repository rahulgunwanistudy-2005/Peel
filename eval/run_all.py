"""Reproduce every number: `make eval`."""

from __future__ import annotations

import subprocess
import sys

STEPS = [
    ["-m", "eval.synthetic"],
    ["-m", "eval.tuning"],
    ["-m", "eval.chains"],
    ["-m", "eval.real_pairs"],
    ["-m", "eval.latency", "--local", "--threads", "2"],
    ["-m", "eval.build_examples"],
]


def main() -> None:
    for step in STEPS:
        print("\n==>", " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], check=True)


if __name__ == "__main__":
    main()
