# syntax=docker/dockerfile:1.7
# Person Search API: one CPU-only image for laptops, CI and the server (decision 0013).
#
#   docker build -t person-search .                       # or: make docker-build
#   docker run -p 8000:8000 -v ps-data:/data \
#       -e PS_ASSETS__REPO=<hf-user>/person-search-assets -e HF_TOKEN=... person-search
#
# Weights and the index are NOT in the image: on start, `person-search serve`
# downloads what is missing into /data (configs/assets.yaml, SHA-256 checked),
# then loads the models. Keep /data on a volume and later starts skip the download.

# Official python:3.10-slim-bookworm, pinned (multi-arch index: amd64 + arm64).
ARG PYTHON_IMAGE=python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370

FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

# ------------------------------------------------------------------ build
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
# git: the CLIP tokenizer is a git dependency (pyproject.toml)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first, from the lockfile only: this layer is reused until uv.lock
# changes. torch/torchvision resolve to the CPU wheels (pytorch-cpu index in
# pyproject.toml), so no CUDA libraries end up in the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group serve --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group serve --no-editable

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
# libGL + glib: opencv-python (pinned for golden parity) needs them at import
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /data /home/user/.config \
    && chown user:user /data /home/user/.config
COPY --from=build /opt/venv /opt/venv
COPY configs /app/configs
WORKDIR /app
USER user
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    YOLO_CONFIG_DIR=/home/user/.config \
    PS_DETECTOR__WEIGHTS=/data/models/yolo11n_person.pt \
    PS_OSNET__WEIGHTS=/data/models/osnet_campus_v1.pth.tar \
    PS_IRRA__WEIGHTS=/data/models/irra_v1.pth \
    PS_API__INDEX_PATH=/data/index \
    PS_API__UPLOAD_DIR=/data/uploads \
    PS_API__VIDEO_DIR=/data/videos \
    PS_API__CLIP_DIR=/data/clips \
    PS_ASSETS__MANIFEST=/app/configs/assets.yaml \
    PORT=8000
# No VOLUME instruction: mount one at /data yourself (-v ps-data:/data); on a
# Hugging Face Space an image-declared volume could hide the uid-1000 /data (0015).
EXPOSE 8000
# Generous start period: the first start downloads ~1.1 GB before the models load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health\", timeout=4)"]
CMD ["person-search", "serve", "--host", "0.0.0.0"]
