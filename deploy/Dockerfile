# Peel API + static web build, for a Hugging Face Space (Docker SDK, CPU basic, port 7860).
# Build context: the repository root.

# ---- web build -----------------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
# Same-origin API when the Space serves the page itself.
ENV VITE_API_BASE=""
RUN npm run build

# ---- runtime -------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    PEEL_MODEL_DIR=/home/user/.cache/peel \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    PEEL_THREADS=2
USER user
WORKDIR /home/user/app

# CPU-only torch first: the default wheel drags in gigabytes of CUDA libraries.
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user engine/pyproject.toml engine/pyproject.toml
COPY --chown=user engine/peel engine/peel
COPY --chown=user api/requirements.txt api/requirements.txt
RUN pip install --user ./engine -r api/requirements.txt

# Bake every model into the image so the first request never downloads anything.
RUN python -c "from peel import models; print(models.warmup())"
ENV HF_HUB_OFFLINE=1

COPY --chown=user api/app api/app
COPY --chown=user api/examples_data api/examples_data
COPY --chown=user --from=web /src/web/dist web/dist

ARG PEEL_GIT_SHA=unknown
ENV PEEL_GIT_SHA=$PEEL_GIT_SHA \
    PEEL_EXAMPLES_DIR=/home/user/app/api/examples_data \
    PEEL_WEB_DIST=/home/user/app/web/dist

EXPOSE 7860
# One worker: results live in process memory (30-minute TTL), so there is nothing to share.
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "api", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
