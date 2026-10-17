# ==============================================================================
# AI Voice Studio - Backend Production Dockerfile
# Multi-stage / lightweight runtime container with FFmpeg and PyTorch support
# ==============================================================================

FROM python:3.10-slim-bullseye AS base

# Prevent Python from writing .pyc files and force unbuffered standard output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install critical OS dependencies: FFmpeg, libsndfile, git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root unprivileged service user
RUN groupadd -g 10001 studio && \
    useradd -u 10000 -g studio -s /bin/bash -m studio

WORKDIR /app

# Install pip build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in cached layers
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy backend codebase
COPY core ./core
COPY commons ./commons
COPY voices ./voices
COPY main.py pytest.ini ./

# Create data directories with appropriate user ownership
RUN mkdir -p storage/tmp storage/output storage/models logs && \
    chown -R studio:studio /app

# Switch to unprivileged runtime user
USER studio

# Expose HTTP API port
EXPOSE 8000

# Docker healthcheck pinging /v1/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "core.apis.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
