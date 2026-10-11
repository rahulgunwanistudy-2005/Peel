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
print-proofing metaphor (neutral gray stage, process inks for keep, peel and review), and an n8n
workflow that gates automated product-photo edits on the drift score.

## The numbers

<!-- BEGIN NUMBERS -->
- On 216 synthetic cases with known ground truth, Peel restores 98.7% of unrequested pixel changes byte for byte; the diff-merge baseline restores 38.1%.
- Peel's kept region matches the requested edit with IoU 0.76 (diff-merge: 0.08), and unedited-region LPIPS drops from 0.164 in the raw edit to 0.028.
- On a real Nano Banana chain from Banana100, after 20 chained edits the untouched areas of the raw chain are at LPIPS 0.388 and 17.5 dB PSNR against step 0; the peeled chain is at LPIPS 0.087 and 56.7 dB.
<!-- END NUMBERS -->

## Challenges

- Diff-merge tools keep everything that changed. Telling a requested change from collateral needed
  the instruction, and "red ball" kept matching a red jacket until added objects had to light up in
  the edit but not in the original.
- A cast correction that was right for the background turned a kept red apple orange.
- Banana100 is one 218 GB parquet with images in 2 GB compressed pages; we wrote a streaming Thrift
  and Snappy reader to pull single frames.

## What's next

More real pairs with hand-drawn masks, text-region locks, and a ComfyUI node.
