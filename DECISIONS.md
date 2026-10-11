# Decisions

One line each, newest last.

- One Python venv at the repo root (uv, 3.11) serves engine, api and eval; fewer moving parts than three.
- Guided filter is implemented with box filters instead of pulling opencv-contrib for `ximgproc`.
- `PeelResult.images` are URLs relative to the API root, not base64: smaller JSON, cacheable, and n8n can fetch them.
- Region coordinates are full-resolution original pixels so the web overlay needs no knowledge of the analysis scale.
- SSE is hand-written on `StreamingResponse`; no extra dependency for a 20-line protocol.
- Face drift threshold is on SFace cosine *distance* (1 − similarity), stricter than the "different person" cutoff, because the goal is detecting re-rendering, not verifying identity.
- CLIPSeg's vision tower runs once per image and the decoder once per phrase (internals called directly); the stock forward pass would re-encode the image for every phrase.
- Change components that straddle the intent boundary are split with hysteresis on the intent heatmap, so a collateral change touching the edit can be peeled separately.
- When a global cast is detected it is also removed inside kept regions, so the kept edit sits in the original's lighting.
- The synthetic drift generator lives in the engine package (`peel.synth`) because the unit tests and the eval suite both need it.
- LPIPS only counts where there is pixel-level evidence of change: its coarse feature maps bleed far past the changed pixels and made kept regions blobby.
- The change threshold has a high ceiling (0.6): real AI edits re-render every pixel, so a low ceiling merged the edit with the noise and lost it.
- Differential grounding for added/removed objects (edit heat minus original heat): CLIPSeg matched "red ball" to a red jacket.
- Cast correction is detected with an affine Lab fit but only its mean shift is applied to kept pixels: the affine extrapolated badly and turned a red apple orange.
- Alignment must beat "the edit did not move" on the high-pass background residual; smooth scenes and brightness changes otherwise produced invented shifts.
- ECC refinement after the homography for sub-pixel accuracy on low-texture scenes.
- A remainder attached to the edit is kept for review only if it is of similar size, strongly changed and not a sliver (shadows, reflections); otherwise it is drift.
- Face drift threshold 0.25 (SFace cosine distance): JPEG and mild blur measure below 0.1, elastic re-rendering above 0.3 on our one test portrait.
- Banana100 is read by decoding the parquet dictionary page by hand (Thrift header + raw Snappy prefix); pyarrow needs the whole 2 GB page in memory.
- Bundled examples come from Banana100 (real Nano Banana Pro edits) because `eval/pairs/` was empty; two more are synthetic and labelled so.
- Examples are compiled into the web bundle with lossy display copies; the loupe reads the lossless files, so the first view is light and the pixel check stays exact.
- Hash routing in the web app: no server rewrites needed on Vercel or on the Space.
- k = 6 stays the default although k = 8 scored higher IoU in the sweep: moving it after seeing test results would be tuning on the test set.
- Root `ruff.toml` relaxes line length for `api/`, `eval/`, `deploy/`; the engine keeps the strict config.
- TypeScript pinned to 5.9: `openapi-typescript` does not accept 6.x yet. ESLint replaces the template's oxlint because the brief asks for it.
- Vercel project uses Standard Protection (production domain public, previews private) and `VITE_API_BASE=off` until an API host exists.
- Hugging Face refused the Space (Docker Spaces on free CPU need PRO as of this date); deployment files are ready and the API host is a human decision.
- n8n workflow uses plain HTTP Request nodes for Gemini rather than a Gemini node, so it does not depend on a node version that supports image editing.
