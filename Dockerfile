# syntax=docker/dockerfile:1.4
# ==============================================================================
# AudioPipeline V1 — Production GPU Inference Container
# Optimized for CUDA 12.4 + Python 3.10 with FFmpeg and Chatterbox TTS
# ==============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install Python 3.10, FFmpeg, and audio processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Upgrade pip and install PyTorch with CUDA 12.4 support
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Copy consolidated dependencies
COPY requirements.txt .

# Install application dependencies
RUN pip install -r requirements.txt

# Create application non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/backend/storage/tmp /app/backend/storage/output /app/backend/logs /app/backend/voices && \
    chown -R appuser:appuser /app

# Copy backend source code
COPY --chown=appuser:appuser backend /app/backend

# Switch to non-root user
USER appuser
WORKDIR /app/backend

# Pre-create cache directories
RUN mkdir -p storage/tmp/cache/conditionals

# Expose API port
EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Launch ASGI server via main.py
CMD ["python", "main.py"]
