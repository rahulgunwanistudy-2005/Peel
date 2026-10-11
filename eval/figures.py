"""Contact sheet of the bundled chain example: raw chain on the left, peeled chain on the right.

python -m eval.figures
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from eval.common import ROOT, WEB_PUBLIC, read_rgb

STEPS = (1, 5, 10, 20)
TARGET = ROOT / "docs" / "screenshots" / "chain-raw-vs-peeled.jpg"


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(out, (0, 0), (w + 16, h + 16), (20, 20, 20), -1)
    cv2.putText(
        out, text, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def main() -> None:
    chain = json.loads((WEB_PUBLIC / "chain" / "chain.json").read_text())["chain"]
    available = {s["step"] for s in chain["steps"]}
    rows = []
    for step in [s for s in STEPS if s in available]:
        raw = read_rgb(WEB_PUBLIC / "chain" / f"step{step:02d}_raw.webp")
        peeled = read_rgb(WEB_PUBLIC / "chain" / f"step{step:02d}_peeled.webp")
        size = (640, round(640 * raw.shape[0] / raw.shape[1]))
        left = label(cv2.resize(raw, size, interpolation=cv2.INTER_AREA), f"Raw chain, step {step}")
        right = label(
            cv2.resize(peeled, size, interpolation=cv2.INTER_AREA), f"Peeled chain, step {step}"
        )
        gap = np.full((size[1], 8, 3), 119, dtype=np.uint8)
        rows.append(np.hstack([left, gap, right]))
        rows.append(np.full((8, rows[-1].shape[1], 3), 119, dtype=np.uint8))
    sheet = np.vstack(rows[:-1])
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TARGET), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {TARGET} ({TARGET.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
