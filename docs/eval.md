# Evaluation: method and limits

All numbers are in [numbers.md](numbers.md), generated from `eval/results/*.json`. `make eval`
reproduces them. This page says how each was measured and what it cannot tell you.

## 1. Synthetic drift suite

**Why synthetic.** For a real AI edit nobody knows, pixel for pixel, what the "right" output is.
Here it is known by construction.

**Construction** (`engine/peel/synth.py`, `eval/synthetic.py`). For each case:

1. Take a base photo (four scikit-image photos and the five Banana100 seed images we could fetch).
2. Apply an *intended* edit with a known mask and a nameable target: a shaded sprite (ball, star,
   balloon, banana, heart) is added, removed, recoloured or replaced. The instruction is generated
   with it ("Replace the green ball with a pink heart").
3. Apply *collateral* drift: a whole-image colour cast, a 0.5–2 % shift with slight scaling, JPEG
   re-encoding, blur, sharpening, noise, and local patches that are blurred, hue-shifted or
   elastically warped (on the face when the photo has one).

9 photos × 4 edit kinds × 6 drift recipes = 216 cases. The ideal output is the base photo with only
step 2 applied.

**Metrics.**

| Metric | Definition |
| --- | --- |
| Intended-region IoU | IoU of the region the method kept against the ground-truth mask. |
| Collateral pixels reverted | Of the pixels outside the mask that the drift changed, the share that is byte-identical to the original in the output. |
| Byte-identical outside the edit | Share of all pixels outside the mask that are byte-identical to the original. |
| Unedited-region PSNR / LPIPS | Output against original, outside the mask. PSNR is capped at 60 dB because identical pixels would be infinite. |
| Kept-edit fidelity PSNR | Output against the ideal image, inside the mask: was the edit itself kept intact? |

**Methods compared.** The raw edit; a **diff-merge baseline** (threshold the blurred pixel
difference at 13/255, dilate, feather, composite: what existing nodes do, without their alignment
step); full Peel; and Peel with one component switched off at a time.

**Limits.**
- Sprites are easier to locate than real edited objects, and synthetic drift is cleaner than a
  diffusion model's. Absolute numbers here are optimistic for real edits; the *differences between
  methods* are the point.
- The diff-merge baseline has no registration. Nodes that align first would do better on shifted
  cases; they would still keep every drifted region.
- The "all cases" ablation table dilutes each component, because a component only matters on the
  cases that exercise it. The second table restricts each ablation to those cases.
- The general suite keeps edits away from faces, so the face lock shows no effect there. It is
  measured separately (section 2).
- Kept regions include "review" regions (kept by default), which lowers IoU by design.
- **Re-registration shows no measurable effect here.** The synthetic shifts are global, so the
  first-pass homography already solves them. What the background lock protects against (an edit
  that moves many keypoints and drags the fit) is covered by a unit test, not by this suite.
- **Cast separation is a trade.** It costs a little intended IoU and buys kept-edit fidelity: the
  kept object sits in the original's colour balance instead of the edit's.

## 2. Face lock

`eval/face_suite.py`: 24 cases on one public-domain portrait. The requested object touches a face
that has been elastically warped. We measure the share of face pixels restored to the original and
the SFace identity distance between the original face and the face in the output.

**Limits.** One face, synthetic warps. This shows the mechanism works, not that the threshold is
right across people, poses or real model re-renders.

## 3. Threshold sweep

`eval/tuning.py` sweeps k in *median + k·MAD* and the threshold ceiling on a 44-case subset. The
shipped default is k = 6 with a ceiling of 0.6. The sweep shows that the ceiling matters more than
k, and that k = 8 is slightly ahead on intended IoU while k = 6 is slightly ahead on kept-edit
fidelity; we did not move the default after seeing this, to avoid tuning on the test cases any
further. The subset overlaps the main suite, so the main suite's Peel numbers are mildly
tuned-on-test; the sweep table shows how much the choice matters.

## 4. Banana100 chains

`eval/chains.py` on real Nano Banana Pro chains. For each step we compare the raw chain image and
the peeled chain image with the step-0 image, in the regions no kept edit has touched (LPIPS at
512 px, PSNR at full evaluation size).

**What was measured:** one local-edit chain (`table / add_apple_denoise`, 20 steps) and four
replica chains (`fog`, `library`, `peacock`, `sand`, 20 steps each).

**Limits.**
- The sample is small and was chosen by what could be fetched from a 218 GB file, not at random
  (see `eval/BANANA100_NOTES.md`).
- For replica chains the instruction requests nothing, so Peel returns the step-0 image by
  construction. Those rows measure the raw drift, not Peel's skill.
- The raw chain is compared after a plain resize to the original's size. Part of its error is
  reframing, which is real damage for a user but is not "noise" in the paper's sense.
- "Untouched region" is defined by Peel's own kept masks. A requested object that Peel failed to
  keep would not be counted against it here; the synthetic suite covers that failure instead.
- Images were evaluated at 1024 px, not at the dataset's native 2816 px.

## 5. Real pairs

`eval/real_pairs.py` expects hand-made Nano Banana edits with a rough mask in `eval/pairs/`. If the
folder is empty the result file says "skipped" and the app and README show no number.

## 6. Latency

`eval/latency.py` posts the same 1024-px pair 20 times and reads the per-stage timings the pipeline
records, with a varying option so the cache cannot answer. `--api` measures the deployed hardware;
`--local --threads 2` measures a laptop limited to two threads. The hardware is named next to every
latency number.

## What is not evaluated

Human preference, real edits at scale, non-photographic images, text regions, and global
instructions. The unit tests cover behaviour for those paths; no accuracy is claimed.
