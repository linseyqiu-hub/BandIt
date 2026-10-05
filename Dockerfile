# ============================================================
# BandIt API — multi-stage Dockerfile
# ============================================================
#
# Stage 1 (builder): install Python dependencies into a venv.
#   - Uses the PyTorch CPU index so we get the ~200MB CPU wheel
#     instead of the ~2GB CUDA wheel.
#   - Compilers, pip cache, and build headers stay in this stage
#     and never reach the final image.
#
# Stage 2 (runtime): copy the venv + source code, run the app.
#   - Starts from the same base image (required — compiled
#     extensions must match the runtime's glibc and Python).
#   - Runs as a non-root user.
#   - Model weights are NOT in the image — they're downloaded
#     from HuggingFace Hub at startup and cached in a mounted
#     volume at /models.
#
# Build:   docker build -t bandit-api .
# Run:     docker compose up  (see compose.yaml)
# ============================================================


# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.13-slim AS builder

# Create a venv at a fixed path so we can copy it wholesale
# into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirements file first — this layer is cached
# as long as dependencies don't change, so a source-code edit
# doesn't re-install torch.
COPY requirements_prod.txt .

# Install with the PyTorch CPU index.
# --no-cache-dir keeps the layer smaller (no pip wheel cache).
RUN pip install --no-cache-dir -r requirements_prod.txt


# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.13-slim

# Non-root user — if the app is compromised, the attacker gets
# a restricted user, not root.
RUN adduser --disabled-password --no-create-home bandit

# Copy the fully-installed venv from the builder stage.
# Nothing else from the builder comes along — no pip cache,
# no compilers, no build headers.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the serving source code. eval/, notebooks/,
# frontend/, data/, checkpoints/ are blocked by .dockerignore
# and never enter the build context.
WORKDIR /app
COPY src/ ./src/

# HuggingFace cache directory — must match the mounted volume
# in compose.yaml. Set here (not in Python) because the
# huggingface_hub library reads it at import time.
ENV HF_HOME=/models

# Create the mount point so it exists even before compose
# attaches the volume.
RUN mkdir -p /models && chown bandit:bandit /models

# Documentation: this container listens on 8000.
# Does NOT publish the port — that's compose's job.
EXPOSE 8000

# Liveness probe. Docker marks the container unhealthy after
# 3 consecutive failures (90s of downtime). The first check
# waits 30s for startup (model download + load).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Drop to non-root before running.
USER bandit

# Start the API. --host 0.0.0.0 is required — without it
# uvicorn binds to 127.0.0.1 (loopback only), which means
# published ports and other containers can't reach it.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]