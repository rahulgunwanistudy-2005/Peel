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
