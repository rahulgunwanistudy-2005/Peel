"""Build the curated examples and the "How it works" figures.

Every example is produced by running the real pipeline; nothing is drawn by hand. Outputs:
  * api/examples_data/        served by GET /v1/examples
  * web/public/examples/      bundled with the web build so the first view needs no server
  * web/public/how/           intermediate images of the first example
  * web/src/api/fixtures/     the same JSON, as the contract fixture for the web app

    python -m eval.build_examples
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
from peel.arrays import U8
from peel.pipeline import PeelOutput, peel_arrays
from peel.synth import (
    COLORS,
    drift_cast,
    drift_jpeg,
    drift_patch_color,
    drift_patch_warp,
    drift_shift,
    paste_sprite,
    render_sprite,
)
from peel.types import PeelImages, PeelResult
from skimage import data

from eval.common import (
    CACHE,
    ROOT,
    WEB_PUBLIC,
    fit_long_side,
    heat_to_rgb,
    read_rgb,
    side_by_side,
    write_webp,
)

API_DIR = ROOT / "api" / "examples_data"
WEB_DIR = WEB_PUBLIC / "examples"
HOW_DIR = WEB_PUBLIC / "how"
FIXTURES = ROOT / "web" / "src" / "api" / "fixtures"
LONG_SIDE = 1024
IMAGE_NAMES = ("original", "edited", "peeled", "change_map", "diff", "raw_diff")
LOSSLESS = {"original", "edited", "peeled", "source_edit"}
BANANA_SOURCE = (
    "Banana100 (Tang et al., CC BY-NC-SA 4.0): a real Nano Banana Pro edit, shown at reduced size."
)


@dataclass
class Spec:
    id: str
    title: str
    blurb: str
    instruction: str
    source: str
    synthetic: bool
    original: U8
    edited: U8


def banana_pair(folder: str, step: int) -> tuple[U8, U8, str] | None:
    root = CACHE / "banana100" / folder
    manifest = root / "manifest.json"
    if not manifest.exists():
        return None
    meta = json.loads(manifest.read_text())
    instruction = next(s["instruction"] for s in meta["steps"] if s["step"] == step)
    original = fit_long_side(read_rgb(root / "step_000.png"), LONG_SIDE)
    edited = fit_long_side(read_rgb(root / f"step_{step:03d}.png"), LONG_SIDE)
    return original, edited, instruction


def specs() -> list[Spec]:
    out: list[Spec] = []

    pair = banana_pair("table_add_apple_denoise_run1", 1)
    if pair is not None:
        original, edited, instruction = pair
        out.append(
            Spec(
                id="desk-apple",
                title="One apple, please",
                blurb=(
                    "The prompt asked for one apple and said to keep everything else unchanged. The editor "
                    "also reframed the shot, re-rendered the lamp and plant, and repainted the wall."
                ),
                instruction=instruction,
                source=BANANA_SOURCE,
                synthetic=False,
                original=original,
                edited=edited,
            )
        )

    for folder, title in (
        ("library_replica_run1", "An exact replica"),
        ("peacock_replica_run1", "An exact replica"),
    ):
        pair = banana_pair(folder, 1)
        if pair is not None:
            original, edited, instruction = pair
            out.append(
                Spec(
                    id=folder.split("_")[0] + "-replica",
                    title=title,
                    blurb=(
                        "The model was asked to return the image untouched. The ledger shows what it changed "
                        "anyway; Peel hands back the original, byte for byte."
                    ),
                    instruction=instruction,
                    source=BANANA_SOURCE,
                    synthetic=False,
                    original=original,
                    edited=edited,
                )
            )
            break

    astronaut = np.ascontiguousarray(data.astronaut())
    sprite, mask = render_sprite("balloon", COLORS["yellow"], 120)
    edited, _ = paste_sprite(astronaut, sprite, mask, 372, 40)
    edited = drift_patch_warp(edited, (160, 50, 150, 170), 0.12, seed=1)
    edited = drift_shift(drift_cast(edited, db=4.5, da=1.0), 1.8, -1.2)
    out.append(
        Spec(
            id="astronaut-face",
            title="A re-rendered face",
            blurb=(
                "A balloon was requested. The face was also redrawn with different proportions, the frame "
                "shifted two pixels and the whole image went warmer. Here the drift was generated in code, "
                "so the right answer is known exactly."
            ),
            instruction="Add a yellow balloon",
            source="Photo: NASA (public domain), via scikit-image. Edit and drift are synthetic.",
            synthetic=True,
            original=astronaut,
            edited=edited,
        )
    )

    coffee = np.ascontiguousarray(data.coffee())
    sprite, mask = render_sprite("ball", COLORS["blue"], 110)
    with_ball, _ = paste_sprite(coffee, sprite, mask, 60, 60)
    recolored, _ = render_sprite("ball", COLORS["red"], 110)
    edited, _ = paste_sprite(coffee, recolored, mask, 60, 60)
    edited = drift_patch_color(edited, (330, 150, 150, 130), hue_shift=14)
    edited = drift_jpeg(drift_cast(edited, db=-3.5, dl=1.5), 55)
    out.append(
        Spec(
            id="coffee-recolor",
            title="A recolour with side effects",
            blurb=(
                "The ball was meant to turn red. The coffee also changed hue, the frame went cooler and "
                "the image was recompressed. Synthetic drift with a known answer."
            ),
            instruction="Make the blue ball red",
            source="Photo: Rachel Michetti (CC0), via scikit-image. Edit and drift are synthetic.",
            synthetic=True,
            original=with_ball,
            edited=edited,
        )
    )
    return out


def render(output: PeelOutput) -> dict[str, U8]:
    a = output.analysis
    return {
        "original": a.original,
        "edited": a.aligned,
        "peeled": output.peeled,
        "change_map": output.change_map(),
        "diff": output.diff(),
        "raw_diff": output.raw_diff(),
    }


def to_result(spec: Spec, output: PeelOutput, images: PeelImages) -> PeelResult:
    width, height = output.size
    return PeelResult(
        id=f"ex-{spec.id}",
        width=width,
        height=height,
        ledger=output.ledger,
        timings=output.timings,
        total_ms=round(output.total_ms, 1),
        images=images,
    )


def how_figures(output: PeelOutput, face_output: PeelOutput | None) -> None:
    """Real intermediate images for the seven steps on the How it works page."""
    a = output.analysis
    shape = a.change.mask.shape
    work = (shape[1], shape[0])
    original = cv2.resize(a.original, work, interpolation=cv2.INTER_AREA)
    aligned = cv2.resize(a.aligned, work, interpolation=cv2.INTER_AREA)
    gain = 6.0

    def amp(x: U8, y: U8) -> U8:
        return np.clip(cv2.absdiff(x, y).astype(np.float32) * gain, 0, 255).astype(np.uint8)

    HOW_DIR.mkdir(parents=True, exist_ok=True)
    unaligned = cv2.resize(SOURCE_EDIT, work, interpolation=cv2.INTER_AREA)
    write_webp(
        HOW_DIR / "1-register.webp", side_by_side(amp(unaligned, original), amp(aligned, original))
    )
    write_webp(
        HOW_DIR / "2-cast.webp",
        side_by_side(heat_to_rgb(a.change.delta_e_raw, 8.0), heat_to_rgb(a.change.delta_e, 8.0)),
    )
    write_webp(HOW_DIR / "3-change.webp", heat_to_rgb(a.change.score, 0.6))
    heat = heat_to_rgb(a.intent.heat, 1.0).astype(np.float32) / 255.0
    dimmed = aligned.astype(np.float32) * (0.25 + 0.75 * heat)
    write_webp(HOW_DIR / "4-intent.webp", dimmed.astype(np.uint8))
    write_webp(HOW_DIR / "6-classify.webp", output.change_map())
    write_webp(HOW_DIR / "7-composite.webp", side_by_side(output.raw_diff(), output.diff()))

    if face_output is not None and face_output.analysis.faces:
        fa = face_output.analysis
        canvas = cv2.resize(
            fa.aligned,
            (fa.change.mask.shape[1], fa.change.mask.shape[0]),
            interpolation=cv2.INTER_AREA,
        ).copy()
        for face in fa.faces:
            x, y, w, h = face.box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 2)
            label = (
                f"identity distance {face.distance:.2f}" if face.distance is not None else "face"
            )
            cv2.rectangle(canvas, (x, y + h), (x + 250, y + h + 28), (20, 20, 20), -1)
            cv2.putText(
                canvas,
                label,
                (x + 6, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        write_webp(HOW_DIR / "5-faces.webp", canvas)


SOURCE_EDIT: U8 = np.zeros((1, 1, 3), dtype=np.uint8)


def main() -> None:
    global SOURCE_EDIT
    for folder in (API_DIR, WEB_DIR):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    api_index, web_index = [], []
    outputs: dict[str, PeelOutput] = {}
    all_specs = specs()
    for spec in all_specs:
        output = peel_arrays(spec.original, spec.edited, spec.instruction)
        outputs[spec.id] = output
        images = render(output)
        for folder in (API_DIR / spec.id, WEB_DIR / spec.id):
            for name, rgb in images.items():
                write_webp(folder / f"{name}.webp", rgb, lossless=name in LOSSLESS, quality=90)
        # Lossy display copies and a thumbnail keep the first view light; the loupe reads the
        # lossless files, so its pixel-identity readout is exact.
        for name in sorted(LOSSLESS & set(images)):
            write_webp(WEB_DIR / spec.id / f"{name}.display.webp", images[name], quality=84)
        thumb = fit_long_side(images["edited"], 360)
        write_webp(WEB_DIR / spec.id / "thumb.webp", thumb, quality=78)
        # The API re-analyses an example from its inputs the first time a region is flipped.
        write_webp(API_DIR / spec.id / "source_edit.webp", spec.edited, lossless=True)
        # The web build carries it too: a stateless API host is sent the pair again to flip a region.
        write_webp(WEB_DIR / spec.id / "source_edit.webp", spec.edited, lossless=True)
        assert np.array_equal(read_rgb(WEB_DIR / spec.id / "peeled.webp"), output.peeled), (
            "lossless check"
        )

        entry = {
            "id": spec.id,
            "title": spec.title,
            "blurb": spec.blurb,
            "instruction": spec.instruction,
            "source": spec.source,
            "synthetic": spec.synthetic,
        }
        api_images = PeelImages(**{n: f"/v1/jobs/ex-{spec.id}/images/{n}.png" for n in IMAGE_NAMES})
        web_images = PeelImages(**{n: f"./examples/{spec.id}/{n}.webp" for n in IMAGE_NAMES})
        (API_DIR / spec.id / "result.json").write_text(
            to_result(spec, output, api_images).model_dump_json(indent=2)
        )
        api_index.append(entry)
        web_index.append(
            {**entry, "result": json.loads(to_result(spec, output, web_images).model_dump_json())}
        )
        ledger = output.ledger
        print(
            f"{spec.id}: {ledger.headline} | drift {ledger.drift_score} | {output.total_ms:.0f} ms",
            flush=True,
        )
        for region in ledger.regions:
            print(f"    {region.id} {region.decision:6} {region.type:13} {region.sentence}")

    (API_DIR / "index.json").write_text(json.dumps(api_index, indent=2) + "\n")
    (WEB_DIR / "index.json").write_text(json.dumps(web_index, indent=2) + "\n")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "examples.json").write_text(json.dumps(web_index, indent=2) + "\n")

    first = all_specs[0]
    if first.id != "desk-apple":
        print(
            f"NOTE: web/index.html preloads the first example's image; update it for {first.id!r}."
        )
    SOURCE_EDIT = first.edited
    how_figures(outputs[first.id], outputs.get("astronaut-face"))
    size = sum(f.stat().st_size for f in WEB_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {len(all_specs)} examples ({size:.1f} MB for the web build)")


if __name__ == "__main__":
    main()
