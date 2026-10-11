# Peel — keep the edit, peel the drift

## Inspiration

Ask an AI image editor for one change and you get several. We asked Nano Banana for one apple on a
desk, with "keep other elements unchanged" in the prompt. It added the apple, and also reframed the
shot, re-rendered the lamp and the plant, and repainted the wall. The Banana100 paper shows this
compounds across chained edits and that standard quality metrics do not flag it. People fix it by
hand-masking in Photoshop.

## What it does

Peel takes the original, the AI edit and the instruction. It returns:

- a composite that keeps the requested change and restores every other pixel, byte for byte;
- a drift ledger: each unrequested change, located, typed and measured, in one plain sentence;
- a chain mode in which edit 20 still has the background of edit 0.

A three-pane loupe lets anyone verify the "byte-identical" claim on any spot of the image.

## How we built it

A CPU-only Python engine, seven stages: register the edit to the original (SIFT + MAGSAC, a
background-locked refit, ECC, optical flow); remove the whole-image colour cast; measure change with
CIEDE2000, gradient structure and LPIPS; locate the instruction with CLIPSeg in both images; compare
faces with SFace; classify regions as intended, collateral or ambiguous; composite with an
edge-aware feather and a Laplacian pyramid. No generative model: every output pixel comes from one
of the two inputs.

Around it: a FastAPI service with live stage progress, a hand-written React interface built on a
print-proofing metaphor (neutral gray stage, process inks for keep / peel / review, registration
marks that slide into register), and an n8n workflow that gates automated product-photo edits on
the drift score.

## The numbers

Measured by code in the repository; nothing is estimated.

<!-- BEGIN NUMBERS -->
- Not measured yet.
<!-- END NUMBERS -->

## Challenges

- Existing diff-merge tools keep everything that changed. Telling a requested change from collateral
  needed the instruction, and "red ball" kept matching a red jacket until we required added objects
  to light up in the edit but not in the original.
- A colour-cast correction that was right for the background turned a kept red apple orange. The
  fit is now used for detection only.
- Banana100 is a single 218 GB parquet whose images sit in 2 GB compressed pages. We wrote a
  streaming Thrift and Snappy reader to pull single frames over range requests.

## What's next

More real edited pairs with hand-drawn masks, text-region locks, and a ComfyUI node so Peel can sit
at the end of any editing graph.
