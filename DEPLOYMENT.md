# AudioPipeline V1 — Production Deployment Runbook

This document defines the deployment architecture, hosting requirements, hardware sizing, environment configuration, and operational runbook for client-facing V1 release.

---

## 1. Architecture Overview & Hosting Strategy

AudioPipeline consists of two distinct components with fundamentally different operational profiles:

1. **Client-Facing SPA Frontend**: React 19 + TypeScript + Vite. Static web bundle with negligible compute requirements (~303 KB gzipped).
2. **Audio ML Inference Backend**: FastAPI + PyTorch + CUDA + Chatterbox Multilingual TTS. Requires NVIDIA GPU acceleration, FFmpeg, and ~3.5 GB VRAM.

### Recommended Deployment Options

| Architecture | Frontend Hosting | Backend Hosting | Estimated Cost | Best For |
|---|---|---|---|---|
| **Option A (Recommended Split)** | Vercel / Cloudflare Pages | RunPod Serverless / Modal / GPU VPS | **$5 – $25 / month** | Lowest cost, high availability, zero idle waste |
| **Option B (Dedicated Single Node)** | Nginx Container | FastAPI CUDA Container | **$120 – $380 / month** | Enterprise client intranet / air-gapped on-premise |
| **Option C (Client Demo / Low Cost)** | Static Hosting ($0) | RunPod Community Pod (RTX 3090/4000) | **~$0.20 – $0.29 / hour** | Client evaluation demos, on-demand activation |

> [!IMPORTANT]
> **Free Tier Warning**: Generic free web hosts (Render, Heroku, Fly.io free tiers) **cannot** run CUDA inference and will run out of memory (OOM) or time out during PyTorch imports. Do not attempt to host the ML inference pipeline on free CPU-only tiers.

---

## 2. Hardware Sizing & VRAM Budget

Empirically measured inference resource profile (measured on NVIDIA RTX 3050 6GB Laptop GPU):

| Resource | Baseline (Subprocess) | Optimized V1 (Resident Singleton) | Production Recommendation |
|---|---|---|---|
| **Peak VRAM** | 3,513 MB | **2,850 MB – 3,513 MB** | 6 GB minimum (8 GB+ recommended) |
| **System RAM** | 4.2 GB | **4.5 GB** | 8 GB minimum |
| **Cold Start** | 59.8s – 65.9s | **41.49s** | ~40s (model load into GPU) |
| **Warm Preview Latency** | 125.7s (2x cold calls) | **7.28s** (HTTP round-trip) | **< 8.0s** |
| **Conditioning Cache Hit** | N/A (recomputed) | **0.031s** (`Conditionals.load`) | Sub-50ms |

---

## 3. Environment Variables Configuration

Copy `backend/.env.example` to `backend/.env` on the host:

```env
# Runtime Environment
ENV=production
DEBUG=false

# Allowed CORS Origins (Replace with production domain)
ALLOWED_ORIGINS=https://audiopipeline.client.com,http://localhost:5173

# Database Connection URI (SQLite default; PostgreSQL supported)
DATABASE_URL=sqlite:///./storage/voice_studio.db

# Storage Directories
TMP_DIR=storage/tmp
VOICES_DIR=voices

# Inference Engine Configuration
CHATTERBOX_IN_PROCESS=true
EXPOSE_EXPERIMENTAL_VOICES=false

# Security & Quota Limits
MAX_UPLOAD_MB=25
MAX_AUDIO_SECONDS=120
MAX_TEXT_CHARS=30000
CHUNK_MAX_CHARS=220
RATE_LIMIT_PER_MINUTE=10
CLIENT_DAILY_MINUTES=120

# Optional External API Keys (Offline rule-based fallback active by default)
GROQ_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=
```

---

## 4. Docker Deployment Instructions

### Prerequisites
- Docker Engine 24+ with NVIDIA Container Toolkit (`nvidia-smi` accessible inside containers)
- Docker Compose v2+

### Step-by-Step Launch

```bash
# 1. Clone repository
git clone <repository_url> audiopipeline
cd audiopipeline

# 2. Configure production environment
cp backend/.env.example backend/.env
# Edit backend/.env with production ALLOWED_ORIGINS

# 3. Build and launch containers
docker compose up -d --build

# 4. Verify running services and health
docker compose ps
curl -f http://localhost:8000/v1/health
```

---

## 5. Persistent Storage & Backup Strategy

The following directories MUST be mounted to persistent volumes:

1. `/app/backend/storage/voice_studio.db`: Stores jobs, status metadata, and audit logs.
2. `/app/backend/storage/tmp/cache/conditionals/`: Stores pre-computed speaker conditioning tensors (`.pt`). Persisting this volume guarantees 7-second warm preview generation across server restarts.
3. `/app/backend/voices/`: Stores client reference recordings and voice catalog manifests.
4. `/home/appuser/.cache/huggingface/`: Model checkpoint cache (~1.5 GB). Prevents re-downloading Chatterbox model weights on every container start.

---

## 6. Smoke Tests & Verification Runbook

Run these checks immediately after any deployment:

```bash
# 1. Healthcheck Probe
curl -i http://localhost:8000/v1/health
# Expected: HTTP 200 OK, {"status": "ok", "version": "0.1.0"}

# 2. Voice Catalog Listing
curl -i http://localhost:8000/v1/voices
# Expected: HTTP 200 OK, client_ref voice available

# 3. Test Preview Generation
curl -X POST http://localhost:8000/v1/tts/preview \
  -H "Content-Type: application/json" \
  -d '{"voice_id": "client_ref", "text": "Smoke test audio pipeline.", "language": "en"}'
# Expected: HTTP 200 OK with job_id, completed within 8-10 seconds on warm server
```

---

## 7. Rollback Procedure

If a deployment fails:
1. Revert container image:
   ```bash
   docker compose down
   docker tag audiopipeline-backend:v1-backup audiopipeline-backend:v1
   docker compose up -d
   ```
2. Database rollback: SQLite database is backed up automatically before restarts; restore `storage/voice_studio.db.bak` if schema changes were applied.
