"""Assemble the README hero GIF from frames captured by web/scripts/hero.mjs.

    cd web && node scripts/hero.mjs && cd .. && python -m eval.hero
"""

from __future__ import annotations

from PIL import Image

from eval.common import ROOT

FRAMES = ROOT / "eval" / "out" / "hero-frames"
TARGET = ROOT / "docs" / "screenshots" / "hero.gif"


def main() -> None:
    paths = sorted(FRAMES.glob("frame-*.png"))
    if not paths:
        raise SystemExit("no frames; run `node scripts/hero.mjs` in web/ first")
    frames = [Image.open(p).convert("RGB").resize((960, 540), Image.Resampling.LANCZOS) for p in paths]
    palette = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(colors=128, palette=palette, dither=Image.Dither.NONE) for f in frames]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(TARGET, save_all=True, append_images=quantized[1:], duration=110, loop=0, optimize=True)
    print(f"wrote {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
