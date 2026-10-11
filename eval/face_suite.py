"""Face-lock ablation where it matters: an edit that sits right next to a re-rendered face.

A drifted face that touches the requested edit forms one change component with it. Without the
face lock the face is "spill" attached to the edit and is kept; with the lock it is restored.
The general synthetic suite keeps edits away from faces, so it cannot show this.

    python -m eval.face_suite
"""

from __future__ import annotations

import numpy as np
from peel.faces import compare_faces
from peel.pipeline import peel_arrays
from peel.synth import COLORS, drift_cast, drift_patch_warp, paste_sprite, render_sprite
from peel.types import PeelOptions
from skimage import data

from eval.common import git_sha, now, write_result

FACE = (160, 50, 150, 170)  # x, y, w, h in the 512-px astronaut photo
SPRITES = ["ball", "balloon", "heart", "star"]


def identity_distance(original: np.ndarray, image: np.ndarray) -> float:
    faces = compare_faces(original, image, original, image)
    return float(faces[0].distance) if faces and faces[0].distance is not None else float("nan")


def main() -> None:
    astronaut = np.ascontiguousarray(data.astronaut())
    x, y, w, h = FACE
    rows = []
    rng = np.random.default_rng(7)
    for index in range(24):
        sprite_name = SPRITES[index % len(SPRITES)]
        color = list(COLORS)[index % len(COLORS)]
        size = int(rng.integers(90, 130))
        # Left or right of the face, overlapping its edge by a few pixels.
        side = -1 if index % 2 == 0 else 1
        sx = x - size + 14 if side < 0 else x + w - 14
        sy = int(y + rng.integers(0, h - size // 2))
        sprite, mask = render_sprite(sprite_name, COLORS[color], size)  # type: ignore[arg-type]
        ideal, intended = paste_sprite(
            astronaut,
            sprite,
            mask,
            int(np.clip(sx, 0, 511 - size)),
            int(np.clip(sy, 0, 511 - size)),
        )
        edited = drift_patch_warp(ideal, FACE, float(rng.uniform(0.10, 0.14)), seed=index)
        edited = drift_cast(edited, db=float(rng.uniform(2, 4)))
        instruction = f"Add a {color} {sprite_name}"
        face_box = np.zeros(astronaut.shape[:2], dtype=bool)
        face_box[y + 20 : y + h - 20, x + 20 : x + w - 20] = True
        face_only = face_box & ~intended
        row = {
            "instruction": instruction,
            "raw_identity_distance": round(identity_distance(astronaut, edited), 4),
        }
        for name, options in (
            ("peel", PeelOptions()),
            ("no_face", PeelOptions(use_face_lock=False)),
        ):
            out = peel_arrays(astronaut, edited, instruction, options)
            identical = np.all(out.peeled == astronaut, axis=-1)
            row[name] = {
                "face_restored_pct": round(100.0 * float(identical[face_only].mean()), 2),
                "identity_distance": round(identity_distance(astronaut, out.peeled), 4),
                "face_regions": sum(r.type == "face_identity" for r in out.ledger.regions),
            }
        rows.append(row)
        print(row, flush=True)

    def mean(variant: str, key: str) -> float:
        return round(float(np.nanmean([r[variant][key] for r in rows])), 4)

    write_result(
        "face_lock",
        {
            "meta": {
                "generated_at": now(),
                "git_sha": git_sha(),
                "n_cases": len(rows),
                "note": "One photo (astronaut, NASA, public domain) with a synthetic elastic face warp; the requested object touches the face.",
            },
            "raw_identity_distance": round(
                float(np.nanmean([r["raw_identity_distance"] for r in rows])), 4
            ),
            "variants": [
                {
                    "id": variant,
                    "label": label,
                    "face_restored_pct": mean(variant, "face_restored_pct"),
                    "identity_distance": mean(variant, "identity_distance"),
                    "faces_flagged_pct": round(
                        100.0 * float(np.mean([r[variant]["face_regions"] > 0 for r in rows])), 1
                    ),
                }
                for variant, label in (("peel", "Peel"), ("no_face", "Peel without face lock"))
            ],
            "cases": rows,
        },
    )


if __name__ == "__main__":
    main()
