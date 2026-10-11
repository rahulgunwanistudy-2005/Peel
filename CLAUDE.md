# Peel — working rules

Peel takes (original image, AI-edited image, instruction) and returns a composite that keeps the
requested change and restores every other pixel, plus a drift ledger. Banana Hacks 2026 entry.
Tagline: **Keep the edit. Peel the drift.**

## Non-negotiables
- Never fabricate numbers. Every metric in README, UI, write-up or video comes from
  `eval/results/*.json`, produced by code in this repo (`make eval`).
- Pixels outside the final mask are bit-identical to the original. There is a test for it; keep it.
- Never silently composite garbage: low registration confidence is surfaced in the ledger and UI.
- No placeholder code, no TODOs in shipped paths, no mocked results outside `web/src/api/fixtures`.
- Commit after every working step with a real message. Credit every library, model and dataset.
- Log each non-obvious decision as one line in `DECISIONS.md`.

## Layout
- `engine/` — Python package `peel` (the real work). Contract lives in `engine/peel/types.py`.
- `api/` — FastAPI wrapper. `api/openapi.json` is exported (`make openapi`) and is the web contract.
- `web/` — Vite + React + TypeScript, hand-written CSS, no UI kit. Design plan in `web/DESIGN.md`.
- `eval/` — synthetic drift suite, ablations, Banana100 runner, real pairs, latency → `eval/results/`.
- `integrations/n8n/`, `deploy/`, `docs/`.

## Commands
- `make setup` — venv (uv, Python 3.11) + npm install. `make dev` — API on :7860, web on :5173.
- `make test` / `make lint` — pytest (engine, api), ruff, `mypy -p peel` (strict), tsc, eslint.
- `cd web && npm run test:e2e` — Playwright smoke tests against the real API and web app.
- `make openapi` — re-export `api/openapi.json` and regenerate `web/src/api/types.ts`.
- `make eval` — reproduce every number, then render them into README and docs (`eval/report.py`).
- `make examples` — rebuild curated examples (`api/examples_data`, `web/public/examples`, `web/public/how`).
- `python deploy/deploy_space.py <user>/<space>` — push the API to a Hugging Face Space.

## Things that bit us
- Run mypy as `mypy -p peel` from `engine/`; `mypy peel` reports that `peel/types.py` shadows stdlib `types`.
- First pipeline call in a process loads the models (several seconds). Timings of that call are not latency.
- Banana100 cannot be read with pyarrow on a laptop: one 2 GB dictionary page per row group. Use `eval/banana100.py`.
- The font's `tnum` feature widens the full stop; use the `Num` component for decimals.
- Colour-cast correction is fitted as an affine Lab transform but only its mean shift may be applied to kept pixels.
- In containers set `PEEL_THREADS` to the CPU quota; `os.cpu_count()` reports the host.

## Engine pipeline (order matters)
decode → same-scene gate → register (SIFT + MAGSAC, background-locked second pass, DIS flow)
→ change (global cast separated first; ΔE2000 + gradient structure + spatial LPIPS; median + k·MAD)
→ intent (rule-parsed phrases → CLIPSeg on both images; global-instruction detection)
→ faces (YuNet + SFace identity lock) → classify → composite (guided feather, Laplacian pyramid,
ring harmonization, bit-identical outside) → ledger.

Analysis runs at long side 768; compositing at full resolution (cap 4096). CPU only, deterministic.

## Design direction (web)
Print-proofing desk: neutral mid-gray stage (#777), light gray chrome, process inks used only where
they carry meaning — cyan = kept, magenta = peeled, yellow = review. One motion moment:
registration crosshairs sliding into alignment. Sentence case. No UI kit, no gradients, no emoji.
