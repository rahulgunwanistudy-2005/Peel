"""Command line: ``peel original.png edited.png "make the jacket red" -o out/``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from peel.io import encode_png
from peel.pipeline import peel_bytes
from peel.types import PeelError, PeelOptions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="peel", description="Keep the edit. Peel the drift.")
    parser.add_argument("original", type=Path)
    parser.add_argument("edited", type=Path)
    parser.add_argument("instruction")
    parser.add_argument("-o", "--out", type=Path, default=Path("peel-out"))
    parser.add_argument("--blend", choices=["laplacian", "poisson"], default="laplacian")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    def progress(stage: str, pct: float) -> None:
        if not args.quiet:
            print(f"[{pct * 100:5.1f}%] {stage}", file=sys.stderr)

    try:
        output = peel_bytes(
            args.original.read_bytes(),
            args.edited.read_bytes(),
            args.instruction,
            PeelOptions(blend=args.blend),
            progress=progress,
        )
    except PeelError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "peeled.png").write_bytes(encode_png(output.peeled, output.analysis.alpha))
    (args.out / "change_map.png").write_bytes(encode_png(output.change_map()))
    (args.out / "diff.png").write_bytes(encode_png(output.diff()))
    (args.out / "ledger.json").write_text(output.ledger.model_dump_json(indent=2))
    if not args.quiet:
        print(output.ledger.headline)
        print(f"Drift score {output.ledger.drift_score:.1f} · {output.total_ms:.0f} ms")
        for region in output.ledger.regions:
            print(f"  {region.id} [{region.decision:6}] {region.sentence}")
        print(f"Wrote {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
