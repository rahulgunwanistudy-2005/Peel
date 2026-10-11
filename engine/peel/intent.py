"""Ground the instruction in the image: which pixels was the user talking about?

Target phrases come from plain rules (no LLM needed). Each phrase is located with CLIPSeg on both
images, because an added object only exists in the edit and a removed one only in the original.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from peel import models
from peel.arrays import F32, U8, Bool, dilate
from peel.types import Mode

Side = Literal["both", "original", "edited"]

MAX_PHRASES = 4
GLOBAL_COVERAGE = 0.55
TEXT_PROMPTS = ("text", "a label with writing")


@dataclass(frozen=True)
class Phrase:
    text: str
    side: Side = "both"


@dataclass(frozen=True)
class ParsedInstruction:
    phrases: tuple[Phrase, ...]
    is_global: bool
    is_noop: bool
    names_text: bool


@dataclass
class IntentResult:
    heat: F32  # max over phrases and images, 0..1, working resolution
    mask: Bool  # thresholded and dilated
    text_heat: F32  # where the original has text-like content
    phrases: list[str]
    mode: Mode
    grounded: bool  # False when no phrase could be located in either image
    coverage: float
    is_noop: bool
    names_text: bool


_POLITE = re.compile(
    r"^\s*(please|kindly|can you|could you|would you|i want you to|i'd like you to|"
    r"i would like you to|i want to|let's|lets|just)\s+",
    re.I,
)
_ARTICLE = re.compile(
    r"^(the|a|an|some|any|his|her|their|my|our|this|that|these|those|its|all the|all)\s+", re.I
)
_COLORS = (
    "red|orange|yellow|green|blue|purple|violet|pink|magenta|cyan|teal|turquoise|brown|black|"
    "white|gray|grey|gold|golden|silver|beige|navy|maroon|crimson|lime|olive|tan|blonde|blond"
)
_TRAILING_ATTR = re.compile(
    rf"\s+(?:(?:bright|dark|light|deep|pale|neon|pastel)\s+)?(?:{_COLORS})(?:\s+colou?red)?$", re.I
)
_PREP = (
    r"on top of|in front of|next to|to the left of|to the right of|"
    r"onto|into|on|to|in|at|beside|behind|above|under|underneath|below|near|over|around|inside|"
    r"across|along|between|from"
)

_NOOP = re.compile(
    r"\b(exact (replica|copy|duplicate)|identical copy|no (alterations?|changes?|modifications?)|"
    r"without (any )?(alterations?|changes?)|do(n't| not) change anything|leave .* unchanged|"
    r"reproduce (the|this) image)\b",
    re.I,
)
_GLOBAL_PATTERNS = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bin the style of\b|\bstyle transfer\b|\bstyle of\b|"
        r"\b(in|as|into)\s+(a\s+|an\s+)?[\w-]+(\s+[\w-]+)?[- ]style\b",
        r"\b(watercolou?r|oil painting|pencil sketch|charcoal|line art|anime|cartoon|comic|"
        r"pixel art|black[- ]and[- ]white|gr[ae]yscale|monochrome|sepia|cinematic|hdr|"
        r"film (look|grain)|vintage|retro look|colou?r[- ]grad(e|ing)|duotone|noir|cyberpunk)\b",
        r"\b(make|turn|change|convert|set|render|transform|shift)\s+it\b",
        r"\b(the\s+)?(whole|entire|overall)\s+(image|photo|picture|scene|frame|shot)\b",
        r"\b(make|turn|change|convert|set|render|transform|adjust|fix|improve)\s+"
        r"((the|this|that)\s+)?"
        r"(image|photo|picture|scene|shot|time of day|lighting|light|weather|season|mood|"
        r"atmosphere|exposure|contrast|saturation|white balance|colou?rs|tones?)\b",
        r"\b(warmer|cooler|brighter|darker|softer|harsher|moodier|more dramatic)\s+"
        r"(lighting|light|tones?|colou?rs?|look|mood)\b",
        r"\b(at|to|into)\s+(night|nighttime|dusk|dawn|sunset|sunrise|golden hour|"
        r"winter|summer|autumn|spring)\b",
        r"^\s*(relight|upscale|enhance|denoise|sharpen|restore|colou?rize|desaturate|"
        r"brighten|darken)\b(?!\s+the\s+(?!image|photo|picture|scene))",
    )
)
_TEXT_WORDS = re.compile(
    r"\b(text|label|sign|logo|caption|title|word|words|letter|letters|writing|headline|"
    r"lettering|font|typography|slogan|number|price|date|watermark|name)\b",
    re.I,
)

_QUOTED = re.compile(r"[\"“']([^\"”']{2,60})[\"”']")
_REPLACE = re.compile(
    r"^(?:replace|swap|substitute|exchange)\s+(?P<x>.+?)\s+(?:with|for|by)\s+(?P<y>.+)$", re.I
)
_INTO = re.compile(
    r"^(?:change|turn|transform|convert|make)\s+(?P<x>.+?)\s+(?:into|to be)\s+(?P<y>.+)$", re.I
)
_REMOVE = re.compile(
    r"^(?:remove|delete|erase|get rid of|take out|take away|eliminate|clear|hide)\s+"
    rf"(?P<x>.+?)(?:\s+(?:{_PREP})\s+.+)?$",
    re.I,
)
_ADD_AT = re.compile(
    r"^(?:add|insert|put|place|draw|include|paint|attach|overlay)\s+(?P<x>.+?)\s+"
    rf"(?:{_PREP})\s+(?P<y>.+)$",
    re.I,
)
_ADD = re.compile(r"^(?:add|insert|put|place|draw|include|paint|attach|overlay)\s+(?P<x>.+)$", re.I)
_GIVE = re.compile(r"^give\s+(?P<y>.+?)\s+(?P<x>(?:a|an|some|the)\s+.+)$", re.I)
_COLOR_OF = re.compile(
    r"^(?:change|make|turn|set|adjust)\s+the\s+colou?r\s+of\s+(?P<x>.+?)(?:\s+(?:to|into)\s+.+)?$",
    re.I,
)
_CHANGE_TO = re.compile(
    r"^(?:change|turn|make|colou?r|paint|recolou?r|dye|tint|set)\s+(?P<x>.+?)\s+(?:to|in)\s+.+$",
    re.I,
)
_MAKE = re.compile(r"^(?:make|turn|colou?r|paint|recolou?r|dye|tint)\s+(?P<x>.+)$", re.I)
_VERB_OBJECT = re.compile(
    r"^(?:change|edit|modify|update|alter|adjust|fix|blur|brighten|darken|sharpen|enlarge|"
    r"shrink|resize|move|rotate|flip|open|close|clean|straighten|whiten|smooth|retouch|"
    r"highlight|lighten|extend|shorten|widen|repaint|redo|restyle|refine)\s+(?P<x>.+)$",
    re.I,
)


_VERBS = (
    "add|insert|put|place|draw|include|paint|attach|overlay|remove|delete|erase|get rid of|"
    "take out|take away|replace|swap|change|turn|make|give|colou?r|recolou?r|blur|brighten|"
    "darken|move|rotate|flip|fix|clean|resize|enlarge|shrink"
)
_CLAUSE_SPLIT = re.compile(
    rf"(?:[.;\n]|,?\s+and then\s+|,?\s+then\s+|,\s+and\s+|,?\s+and\s+(?=(?:{_VERBS})\b))", re.I
)


def _clean(phrase: str) -> str:
    text = phrase.strip().strip(".,;:!?\"'“” ")
    text = re.sub(r"\s+", " ", text)
    previous = None
    while previous != text:
        previous = text
        text = _ARTICLE.sub("", text)
    return text.strip()


def _object_of(phrase: str) -> str:
    """Drop a trailing colour so 'the car blue' grounds as 'car'."""
    cleaned = _clean(phrase)
    stripped = _TRAILING_ATTR.sub("", cleaned)
    return stripped or cleaned


def parse_instruction(instruction: str) -> ParsedInstruction:
    text = instruction.strip()
    is_noop = bool(_NOOP.search(text))
    is_global = (not is_noop) and any(p.search(text) for p in _GLOBAL_PATTERNS)
    names_text = bool(_TEXT_WORDS.search(text))

    phrases: list[Phrase] = []

    def add(raw: str, side: Side = "both") -> None:
        cleaned = _clean(raw)
        if len(cleaned) >= 2 and all(cleaned.lower() != p.text.lower() for p in phrases):
            phrases.append(Phrase(cleaned[:80], side))

    if not is_noop:
        # Sentences are handled one by one; "and"/"then" clauses too.
        clauses = _CLAUSE_SPLIT.split(text)
        for clause in clauses:
            body = clause.strip()
            while (stripped := _POLITE.sub("", body)) != body:
                body = stripped
            body = body.strip(" .,;:!?")
            if not body:
                continue
            if m := _REPLACE.match(body):
                add(m["x"], "original")
                add(m["y"], "edited")
            elif m := _REMOVE.match(body):
                add(m["x"], "original")
            elif m := _COLOR_OF.match(body):
                add(m["x"])
            elif m := _INTO.match(body):
                add(m["x"], "original")
                add(m["y"], "edited")
            elif (m := _GIVE.match(body)) or (m := _ADD_AT.match(body)) or (m := _ADD.match(body)):
                add(m["x"], "edited")
            elif (
                (m := _CHANGE_TO.match(body))
                or (m := _MAKE.match(body))
                or (m := _VERB_OBJECT.match(body))
            ):
                add(_object_of(m["x"]))
            else:
                add(body)
        for quoted in _QUOTED.findall(text):
            add(quoted)
        if not phrases:
            add(text)

    return ParsedInstruction(tuple(phrases[:MAX_PHRASES]), is_global, is_noop, names_text)


def _resize_heat(heat: F32, shape: tuple[int, int]) -> F32:
    return cv2.resize(heat, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def compute_intent(
    original: U8,
    aligned: U8,
    instruction: str,
    *,
    threshold: float = 0.35,
    extra_phrases: list[str] | None = None,
) -> IntentResult:
    """Images at working resolution. ``extra_phrases`` lets an optional LLM add targets."""
    parsed = parse_instruction(instruction)
    phrases = list(parsed.phrases)
    for extra in extra_phrases or []:
        if len(phrases) < MAX_PHRASES and all(extra.lower() != p.text.lower() for p in phrases):
            phrases.append(Phrase(extra))

    shape = original.shape[:2]
    prompts = [p.text for p in phrases] + list(TEXT_PROMPTS)
    heat_all = models.clipseg().heatmaps([original, aligned], prompts)
    n = len(phrases)

    heat = np.zeros(shape, dtype=np.float32)
    for i, phrase in enumerate(phrases):
        in_original = _resize_heat(heat_all[0, i], shape)
        in_edit = _resize_heat(heat_all[1, i], shape)
        if phrase.side == "edited":
            # Differential grounding: something that was added lights up in the edit and not in
            # the original. This stops "red ball" from also selecting a red jacket.
            located = np.clip(in_edit - in_original, 0.0, 1.0)
        elif phrase.side == "original":
            located = np.clip(in_original - in_edit, 0.0, 1.0)
        else:
            located = np.maximum(in_original, in_edit)
        heat = np.maximum(heat, located)
    text_heat = _resize_heat(heat_all[0, n:].max(axis=0), shape)

    core = heat >= threshold
    grounded = bool(core.any())
    mask = dilate(core, max(1, round(0.03 * min(shape))))
    coverage = float(mask.mean())
    is_global = parsed.is_global or coverage > GLOBAL_COVERAGE
    return IntentResult(
        heat=heat,
        mask=mask,
        text_heat=text_heat,
        phrases=[p.text for p in phrases],
        mode="global" if is_global else "local",
        grounded=grounded,
        coverage=coverage,
        is_noop=parsed.is_noop,
        names_text=parsed.names_text,
    )


def empty_intent(shape: tuple[int, int], instruction: str) -> IntentResult:
    """Used by the 'without intent grounding' ablation: nothing is known about the target."""
    zeros = np.zeros(shape, dtype=np.float32)
    return IntentResult(
        heat=zeros,
        mask=np.zeros(shape, dtype=bool),
        text_heat=zeros,
        phrases=[],
        mode="local",
        grounded=False,
        coverage=0.0,
        is_noop=False,
        names_text=bool(_TEXT_WORDS.search(instruction)),
    )
