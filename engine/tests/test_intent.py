from __future__ import annotations

import pytest

from peel.arrays import U8
from peel.intent import compute_intent, parse_instruction
from peel.synth import COLORS, paste_sprite, render_sprite


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Change the shirt to red", [("shirt", "both")]),
        ("make the car blue", [("car", "both")]),
        ("Please remove the person in the background.", [("person", "original")]),
        ("Add a hat to the dog", [("hat", "edited")]),
        ("add sunglasses", [("sunglasses", "edited")]),
        ("Replace the sky with a sunset", [("sky", "original"), ("sunset", "edited")]),
        ("change the color of the sofa to green", [("sofa", "both")]),
        ("Turn the cat into a tiger", [("cat", "original"), ("tiger", "edited")]),
        ("give the man a red tie", [("red tie", "edited")]),
        ("Blur the license plate", [("license plate", "both")]),
        (
            "Could you remove the cup and add a vase on the table",
            [("cup", "original"), ("vase", "edited")],
        ),
    ],
)
def test_target_phrases(instruction: str, expected: list[tuple[str, str]]) -> None:
    parsed = parse_instruction(instruction)
    assert [(p.text, p.side) for p in parsed.phrases] == expected
    assert not parsed.is_global and not parsed.is_noop


@pytest.mark.parametrize(
    "instruction",
    [
        "make it night",
        "Convert the photo to watercolor style",
        "warmer lighting",
        "in the style of Van Gogh",
        "Make the image black and white",
        "change the time of day to sunset",
        "enhance",
    ],
)
def test_global_instructions(instruction: str) -> None:
    assert parse_instruction(instruction).is_global


@pytest.mark.parametrize(
    "instruction",
    ["make this car blue", "change her hair style", "brighten the lamp", "add a night stand"],
)
def test_local_instructions_are_not_global(instruction: str) -> None:
    assert not parse_instruction(instruction).is_global


def test_noop_instruction() -> None:
    parsed = parse_instruction(
        "Produce an exact replica of the provided image, with no alterations."
    )
    assert parsed.is_noop and not parsed.phrases and not parsed.is_global


def test_added_object_is_grounded_in_the_edit_only(astronaut: U8) -> None:
    sprite, mask = render_sprite("ball", COLORS["red"], 110)
    edited, truth = paste_sprite(astronaut, sprite, mask, 370, 60)
    intent = compute_intent(astronaut, edited, "add a red ball")
    assert intent.grounded and intent.mode == "local"
    assert intent.mask[truth].mean() > 0.9, "the ball is inside the intent map"
    assert intent.mask.mean() < 0.25, "and the map is not the whole frame"


def test_unfindable_target_is_reported_as_ungrounded(scene: U8) -> None:
    intent = compute_intent(scene, scene, "remove the giraffe")
    assert not intent.grounded
