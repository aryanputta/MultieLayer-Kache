# =============================================================================
# Dockerfile — Adaptive Multi-Tier KV Cache Orchestrator
# Multi-stage build: base → deps → app
# =============================================================================

# ── Stage 1: CUDA base ───────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3      1

# ── Stage 2: Python dependencies ─────────────────────────────────────────────
FROM base AS deps

WORKDIR /deps

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt --no-cache-dir

# ── Stage 3: Application ─────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /workspace

# Copy source code
COPY . .

# Install the package in editable mode
RUN pip install -e . --no-cache-dir

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /workspace
USER appuser

# Runtime directories
RUN mkdir -p data/traces results/tables results/plots results/logs

# Expose serving port and Prometheus scrape port
EXPOSE 8000 9090

# Default command: start the FastAPI inference server
CMD ["uvicorn", "src.serving.api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]
