# Devpost submission sheet

Copy from here into the Devpost form.

**Name:** Peel

**Tagline:** Keep the edit. Peel the drift.

**Elevator pitch (one sentence):** Peel takes an AI image edit, keeps the change you asked for,
restores every other pixel to your original byte for byte, and shows a ledger of everything the
editor changed without being asked.

**Links**
- Live: https://peel-xi.vercel.app
- Code: https://github.com/rahulgunwanistudy-2005/Peel
- API docs: https://peel-api-rahulgunwanistudy-5466s-projects.vercel.app/docs

**Description:** use `docs/writeup.md` as is (under 500 words).

**Built with**

python, opencv, pytorch, transformers, clipseg, lpips, scikit-image, numpy, scipy, pillow, pydantic,
fastapi, uvicorn, server-sent-events, google-genai, gemini, nano-banana, react, typescript, vite,
css, playwright, pytest, ruff, mypy, github-actions, docker, hugging-face-spaces, vercel, n8n,
huggingface-hub, pyarrow, matplotlib

**Sponsor technologies used**
- n8n: importable drift-gate workflow (`integrations/n8n/`).
- .xyz: custom domain for the web app (fill in once claimed).

**Images to upload** (in `docs/screenshots/`)
1. `hero.gif` — change map, then the compare sweep.
2. `01-workbench.jpg` — stage and ledger on a real Nano Banana edit.
3. `03-loupe.jpg` — the pixel-identity loupe.
4. `05-face-example.jpg` — the identity lock.
5. `07-chain.jpg` — raw versus peeled chain and the drift chart.
6. `08-numbers.jpg` — evaluation tables.

**AI tools disclosure:** written with Claude Code (Claude Opus) during the hackathon; the optional
live-edit feature calls Google Gemini.

**Datasets and models credited:** see the Credits section of the README.
