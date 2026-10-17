# AI Voice Studio — Phasewise Development Progress

> **Governed by `agent.md`.** All standards (4-layer flow, docstrings, logging, error handling, security, 2-file folder limit, no git commit/push, frontend freeze) live in `agent.md` and are NOT repeated here.
> This file only answers: **what to build, in which order, and what to tell the IDE.**
> Project description lives in `README.md`. No other `.md` files are created.

---

## How to Use

1. Work **one phase at a time, in order**. Don't start a phase until the previous one's Acceptance Criteria are ticked.
2. Paste the **Master Command** below, replacing `N` and the phase name. Then paste that phase's **IDE Command** under it.
3. Run the **Verify** steps yourself. You tick the checkboxes; the IDE does not.
4. Paste anything the IDE reports (versions, supported languages, timings, decisions) into **Findings & Decisions** at the bottom.
5. If something fails, paste the error back with: *"Fix this within the current phase scope only."*

### Master Command (paste before every phase)

```text
Follow agent.md strictly. Read progress.md and execute ONLY Phase N: <phase name>.
Use the 5-step SOP (inspect → plan → code → test → review). Show a short plan first.
If the phase says STOP FOR APPROVAL, wait for my reply before writing code.
Backend only. No frontend. No git commit/push. No new .md files. Max 2 code files per folder.
Do not modify agent.md. At the end report: files created/changed (clickable links), how to run and verify,
and any findings/decisions I should record in the "Findings & Decisions" section of progress.md.
```

---

## Decisions Made for This Plan (confirm or change before Phase 0)

| ID | Decision | Why |
|----|----------|-----|
| D1 | **SQLAlchemy 2.0 + SQLite** (single file) for `voices` and `jobs` tables | `agent.md` requires a model/crud/database layer; SQLite needs zero infrastructure and can move to Postgres later. |
| D2 | **Anonymous public access** (no accounts). `commons/auth.py` stays untouched until you ask for auth. | Spec says no user accounts at first. Abuse control = rate limits, not login. |
| D3 | **API prefix `/v1`** (matches the logging examples in `agent.md`). | Consistency. |
| D4 | **`backend/tests/` with 2 files** (`test_services.py`, `test_api.py`) | Not in the canonical tree, so this plan explicitly requests it. Remove the folder + test tasks if you don't want it. |
| D5 | **`__init__.py` files are not created** except `core/__init__.py` (already in canonical tree). Python namespace packages work without them. | Keeps the 2-file limit clean. |
| D6 | **Non-code assets** (`backend/voices/`, `backend/storage/`, `.env`, `requirements.txt`) are outside the 2-file rule. | Rule applies to executable code files only. |
| D7 | **Standalone model checks run through each service file** (`python -m ... ` with an `if __name__ == "__main__":` block), not separate scripts. | No extra code files or folders. |

---

## Target Backend Layout (2 code files max per folder)

```text
backend/
├── commons/                        # auth.py, logger.py          (existing — full, do not add)
├── core/
│   ├── __init__.py                 # .env loading + logger re-export (existing)
│   ├── config.py                   # Settings (limits, paths, CORS, retention, feature flags)
│   ├── apis/
│   │   ├── api.py                  # FastAPI composition root, error handlers, router mounts
│   │   ├── middleware.py           # security headers, rate limiting
│   │   ├── routes/
│   │   │   ├── catalog_routes.py       # health, public config, voices
│   │   │   └── generation_routes.py    # tts, tts/custom, vc, jobs, audio, text cleanup
│   │   └── schemas/
│   │       ├── requests/
│   │       │   ├── catalog_requests.py
│   │       │   └── generation_requests.py
│   │       └── responses/
│   │           ├── catalog_responses.py
│   │           └── generation_responses.py
│   ├── controllers/
│   │   ├── catalog_controller.py
│   │   └── generation_controller.py
│   ├── cruds/
│   │   ├── voice_crud.py
│   │   └── job_crud.py
│   ├── database/
│   │   ├── base_class.py
│   │   └── database.py
│   ├── models/
│   │   ├── voice_model.py
│   │   └── job_model.py
│   └── services/
│       ├── audio/
│       │   ├── audio_service.py        # probe, convert, resample, MP3 export, AI-label metadata
│       │   └── validation_service.py   # upload validation + reference-audio quality checks
│       ├── pipelines/
│       │   ├── openvoice_service.py    # base TTS + OpenVoice tone conversion
│       │   └── rvc_service.py          # RVC voice conversion
│       ├── jobs/
│       │   ├── job_worker.py           # bounded queue + single worker + timeouts
│       │   └── cleanup_service.py      # retention cleanup of temp files/jobs
│       └── llm/
│           └── llm_service.py          # optional Gemini/Groq/OpenRouter helpers
├── tests/                          # test_services.py, test_api.py
├── voices/                         # voices.json (seed) + reference audio + RVC models (git-ignored large files)
├── storage/                        # temp uploads/outputs per job (git-ignored)
├── requirements.txt
└── .env.example
```

---

## Phase Overview

| # | Phase | Outcome | Status |
|---|-------|---------|--------|
| 0 | Environment & project baseline | Tools verified, CUDA PyTorch works, folders in place | ☑ |
| 1 | Core foundation | Config, DB singleton, FastAPI app, middleware, `/v1/health` | ☑ |
| 2 | Audio service | Audio probe/convert/resample/MP3/label utilities | ☑ |
| 3 | OpenVoice V2 service | Text → reference-voice audio via CLI (STOP FOR APPROVAL) | ☑ |
| 4 | RVC service | Audio → target-voice audio via CLI (STOP FOR APPROVAL) | ☑ |
| 5 | Voice catalog | Voices table, seed, list/detail/preview API | ☑ |
| 6 | Job system | Queue, worker, polling, audio streaming, cleanup | ☑ |
| 7 | Text-to-voice API | `POST /v1/tts` end-to-end | ☑ |
| 8 | Voice-to-voice API | `POST /v1/vc` end-to-end | ☑ |
| 9 | Custom reference audio | `POST /v1/tts/custom` with consent + quality checks | ☑ |
| 10 | Security & limits hardening | Rate limits, quotas, upload hardening, prod self-check | ☑ |
| 11 | Optional language helpers | `POST /v1/text/cleanup`, fully optional | ☑ |
| 12 | Tests & backend audit | Test suite (51/51 passing), README update, Definition-of-Done audit | ☑ |
| 13 | Deployment preparation | Dockerfile, .dockerignore, prod config, hosting docs | ☑ |
| F | Frontend | Modern React/Next.js UI, Studio, Audio player, Polling | ☑ |

**Golden rule:** one reliable end-to-end workflow first (Phases 0 → 7), then expand.

---

## PHASE 0 — Environment & Project Baseline

**Goal:** Machine and folders ready. No application logic.

**Tasks**
- [ ] Inspect the existing `backend/` against the layout above; create only **missing** folders (with `.gitkeep`). Never duplicate or restructure existing files.
- [ ] Verify tools manually (Python version OpenVoice supports, Node not needed yet, FFmpeg in PATH, NVIDIA driver).
- [ ] Create backend venv; install **CUDA build** of PyTorch; confirm GPU is visible.
- [ ] Create `requirements.txt` (core deps only: fastapi, uvicorn, pydantic, pydantic-settings, sqlalchemy, python-multipart, soundfile, librosa, pytest, httpx — pinned) and `.env.example` (placeholders only).
- [ ] Extend `.gitignore` for venv, `.env`, `backend/storage/`, `backend/voices/**/*.pth`, `*.index`, audio files, model checkpoints.

**IDE Command**
```text
PHASE 0 specifics:
1. Inspect backend/ and compare with the "Target Backend Layout" in progress.md. Create ONLY missing folders with .gitkeep.
   Do not create any Python files yet.
2. Create backend/requirements.txt (pinned versions, core API + audio deps only — NO OpenVoice/RVC yet) and backend/.env.example
   with placeholder keys: ENV, DEBUG, ALLOWED_ORIGINS, DATABASE_URL, TMP_DIR, VOICES_DIR, MAX_UPLOAD_MB, MAX_AUDIO_SECONDS,
   MAX_TEXT_CHARS, JOB_TTL_MINUTES, QUEUE_MAX_SIZE, JOB_TIMEOUT_SECONDS, RATE_LIMIT_PER_MINUTE, optional GROQ/GEMINI/OPENROUTER keys.
3. Update .gitignore as described in progress.md Phase 0.
4. Give me the exact PowerShell commands to: create the venv, install CUDA PyTorch for my RTX 3050 6GB, install requirements,
   and run a one-line check that torch.cuda.is_available() is True. Do not run installs without telling me first.
```

**Verify**
```text
python --version
ffmpeg -version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # expect True + RTX 3050
```

**Acceptance Criteria**
- [ ] Folder layout matches; nothing existing was overwritten.
- [ ] `torch.cuda.is_available()` prints `True`.
- [ ] `.env.example` has no real secrets.

---

## PHASE 1 — Core Foundation

**Goal:** Running FastAPI app with config, database singleton, middleware, and health endpoint.

**Files:** `core/config.py`, `core/database/base_class.py`, `core/database/database.py`, `core/apis/api.py`, `core/apis/middleware.py`, `core/apis/routes/catalog_routes.py` (health only), `core/controllers/catalog_controller.py` (health only), `core/apis/schemas/responses/catalog_responses.py`.

**Tasks**
- [ ] `config.py`: typed settings from `.env` (all keys from Phase 0), safe defaults, no secrets logged.
- [ ] `database.py`: process-wide SQLAlchemy engine/session singleton for SQLite; create tables at startup; `base_class.py`: declarative Base.
- [ ] `api.py`: app factory, CORS from `ALLOWED_ORIGINS`, startup/shutdown handlers, global handlers returning `{"error": {"code", "message"}}`, mounts routers under `/v1`; disable `/docs` when `ENV=production`.
- [ ] `middleware.py`: security headers (`X-Frame-Options`, `X-Content-Type-Options`, HSTS, CSP) — rate limiting comes in Phase 10.
- [ ] `GET /v1/health` → `{status, version, time}` through route → controller layers.

**IDE Command**
```text
PHASE 1 specifics:
Build the core foundation exactly as listed in progress.md Phase 1 (files + tasks).
- Use the canonical 4-layer flow even for /v1/health.
- core/config.py is the single source of settings; no other module reads os.environ directly except core/__init__.py.
- SQLAlchemy 2.0 + SQLite via DATABASE_URL (decision D1). Singleton engine/session factory; tables created at startup.
- Unknown errors must return a generic message — no stack traces to the client.
Run uvicorn core.apis.api:app --reload and tell me the exact command for Windows.
```

**Verify:** `uvicorn core.apis.api:app --reload` from `backend/` → open `http://localhost:8000/v1/health` (and `/docs` in dev).

**Acceptance Criteria**
- [ ] Health returns 200 and security headers appear in the response.
- [ ] A forced error returns the standard error format with no stack trace.
- [ ] SQLite file is created on startup; settings load from `.env`.

---

## PHASE 2 — Audio Service

**Goal:** Reusable audio utilities used by every pipeline.

**Files:** `core/services/audio/audio_service.py` (validation comes in Phase 8/9 as `validation_service.py`).

**Tasks**
- [ ] `probe_audio()` (duration, sample rate, channels, format via ffprobe), `to_wav()`, `resample()`, `normalize_loudness()` (light), `export_mp3()`, `add_ai_label_metadata()` (ID3/WAV tag "AI-generated").
- [ ] All functions accept `Path` objects built by the backend only — never user-supplied strings.
- [ ] Clear custom exceptions for unreadable/corrupt audio and missing FFmpeg.

**IDE Command**
```text
PHASE 2 specifics:
Create core/services/audio/audio_service.py with the functions listed in progress.md Phase 2.
- Prefer ffmpeg/ffprobe via subprocess (list args, no shell=True, timeouts) plus soundfile/librosa where simpler.
- Never log file contents; log only job-safe info (format, duration).
- Add synthetic-tone based tests later in Phase 12; for now, include an `if __name__ == "__main__":` quick self-check that
  generates a 1s sine wave, probes it, converts, and exports an MP3 to storage/tmp.
```

**Verify:** `python -m core.services.audio.audio_service` from `backend/` — self-check passes and an MP3 plays.

**Acceptance Criteria**
- [ ] Probe/convert/resample/MP3 all work on a generated tone.
- [ ] Corrupt input raises a clean custom error (not a raw traceback).

---

## PHASE 3 — OpenVoice V2 Service (STOP FOR APPROVAL)

**Goal:** Prove `text → base TTS → OpenVoice tone conversion → WAV` from the command line, using **one authorized** reference voice (ideally your own).

**Files:** `core/services/pipelines/openvoice_service.py`.

**Tasks**
- [ ] IDE researches the official OpenVoice V2 repo/README and its compatible base TTS: exact install steps, supported Python version, checkpoints. **Waits for your OK.**
- [ ] After approval: install pinned deps, download checkpoints (large files stay git-ignored under `backend/voices/` or a configured model dir).
- [ ] You place a clean 10–30 s reference clip in `backend/voices/reference/`.
- [ ] Service class with `load()` (once) and `generate(text, reference_path, language, speed) -> Path`; cache the speaker embedding per reference.
- [ ] CLI via `__main__`: `--text --reference --language --output`; prints device and timing.
- [ ] Respect hardware limits in `agent.md`: GPU memory < ~5 GB, free VRAM after use, CPU fallback.

**IDE Command**
```text
PHASE 3 specifics — STOP FOR APPROVAL before installing or coding:
1. Research the official OpenVoice V2 repository and the base TTS it requires. Report: exact install steps for Windows,
   required Python version, checkpoint downloads and sizes, languages the base TTS actually supports, license notes.
   Also say whether it conflicts with the packages in requirements.txt. Do not install anything yet. Wait for my OK.
2. After my OK: implement core/services/pipelines/openvoice_service.py (load once, generate, per-reference embedding cache,
   CPU fallback, free GPU memory after use, friendly custom exceptions for missing checkpoints / bad reference).
3. Add a __main__ CLI: python -m core.services.pipelines.openvoice_service --text "..." --reference <wav> --language en --output <wav>
4. Report timings (CPU vs GPU) and verified supported languages for my Findings & Decisions section.
Never hardcode absolute paths; use settings from core/config.py.
```

**Verify:** run the CLI with your own reference clip; listen to the WAV — it should sound like the reference speaker.

**Acceptance Criteria**
- [ ] WAV generated in the reference voice on GPU.
- [ ] Second run reuses loaded models/embedding (faster).
- [ ] Versions, languages, timings recorded under Findings & Decisions.

---

## PHASE 4 — RVC Service (STOP FOR APPROVAL)

**Goal:** Prove `audio → RVC → target voice` from the command line with **one authorized** RVC model.

**Files:** `core/services/pipelines/rvc_service.py`.

**Tasks**
- [ ] IDE compares RVC inference options on Windows (inference-only package vs official repo): install complexity, maintenance, license, GPU/CPU support. **Waits for your OK.**
- [ ] You provide one authorized RVC model (`.pth` + optional `.index`) in `backend/voices/rvc/<voice_id>/` and note its license.
- [ ] Service class with `convert(input_wav, model_path, index_path, pitch_shift) -> Path`; small LRU cache (max 1 model, because of 6 GB VRAM).
- [ ] Decide if RVC needs a **separate venv** (dependency conflicts with OpenVoice are common). If yes: run it as a subprocess from the same service file with a documented interface.
- [ ] CLI via `__main__`: `--input --model --index --pitch --output`.

**IDE Command**
```text
PHASE 4 specifics — STOP FOR APPROVAL before installing or coding:
1. Research options to run RVC inference from Python on Windows. Give a short comparison (install complexity, maintenance
   status, license, GPU/CPU support) and RECOMMEND ONE. Say whether it conflicts with OpenVoice's dependencies and, if so,
   propose the cleanest isolation (separate venv + subprocess). Wait for my OK.
2. After my OK: implement core/services/pipelines/rvc_service.py (convert(), max-1 model cache, resample to the model's
   expected rate via audio_service, free GPU memory after use, CPU fallback, friendly custom exceptions).
3. Add a __main__ CLI: python -m core.services.pipelines.rvc_service --input <audio> --model <pth> --index <index> --pitch 0 --output <wav>
4. Report timings and the environment strategy for Findings & Decisions.
Never run OpenVoice and RVC on the GPU at the same time.
```

**Verify:** convert a 10–20 s speech clip; listen to input vs output.

**Acceptance Criteria**
- [ ] Converted audio in the target voice produced.
- [ ] Environment strategy (single vs separate venv) decided and recorded.
- [ ] Test model's license/permission recorded.

---

## PHASE 5 — Voice Catalog

**Goal:** Central voice library in the database, seeded from `backend/voices/voices.json`, exposed via API.

**Files:** `core/models/voice_model.py`, `core/cruds/voice_crud.py`, `core/apis/schemas/requests/catalog_requests.py`, `core/apis/schemas/responses/catalog_responses.py` (extend), `core/controllers/catalog_controller.py` (extend), `core/apis/routes/catalog_routes.py` (extend).

**Voice fields:** `id`, `name`, `description`, `style`, `languages[]`, `model_type` (`openvoice_reference` | `rvc_model`), `preview_file`, `asset_file` (relative to `VOICES_DIR`), `license`, `usage_permission`, `status` (`available` | `disabled` | `experimental`), `limitations`.

**Tasks**
- [ ] Idempotent startup sync: upsert valid entries from `voices.json`; invalid entries are skipped with `logging.warning`.
- [ ] Validation: `openvoice_reference` needs a reference audio file; `rvc_model` needs a `.pth` file; path traversal rejected.
- [ ] `GET /v1/voices?model_type=`, `GET /v1/voices/{voice_id}`, `GET /v1/voices/{voice_id}/preview` (streams the sample).
- [ ] **Public responses never include `asset_file` or any filesystem path.**
- [ ] `GET /v1/config/public` exposes only limits (max text chars, max audio seconds, max upload MB, feature flags).

**IDE Command**
```text
PHASE 5 specifics:
Implement the voice catalog per progress.md Phase 5 across model → crud → schemas → controller → route.
- Create backend/voices/voices.json containing ONLY the voices I already tested in Phases 3-4. Ask me for id/name/license/
  permission text — do not invent voices or licenses.
- Startup sync is idempotent and skips invalid entries with a warning instead of crashing.
- Controller raises 404 for unknown ids; CRUD returns None.
- Response schemas must not expose asset paths.
```

**Verify:** open `/docs`, call `/v1/voices`, play `/v1/voices/{id}/preview`.

**Acceptance Criteria**
- [ ] Voice list shows real voices; no paths leaked.
- [ ] A broken `voices.json` entry doesn't crash the app.
- [ ] Preview streams audio.

---

## PHASE 6 — Job System

**Goal:** Long-running inference never blocks an HTTP request.

**Files:** `core/models/job_model.py`, `core/cruds/job_crud.py`, `core/services/jobs/job_worker.py`, `core/services/jobs/cleanup_service.py`, plus extend schemas/controller/routes.

**Job fields:** `id` (unguessable token via `secrets`), `kind` (`tts` | `tts_custom` | `vc`), `status` (`queued` → `processing` → `completed` | `failed` | `expired`), `voice_id`, `progress_message`, `error_code`, `error_message`, `client_key` (**hashed** IP, never raw), `created_at`, `updated_at`, `expires_at`.

**Tasks**
- [ ] Bounded queue + **one** worker thread (GPU safety). Queue full → HTTP 429 with a friendly message.
- [ ] Per-job timeout from settings; on startup mark stale `processing` jobs as `failed`.
- [ ] `GET /v1/jobs/{job_id}` (status/progress/error/result links), `GET /v1/audio/{job_id}/output?format=wav|mp3`, `GET /v1/audio/{job_id}/source`, `DELETE /v1/jobs/{job_id}`.
- [ ] Files live only under `storage/tmp/<job_id>/`, resolved by the backend from the job id.
- [ ] Cleanup service deletes expired jobs' files on a schedule (`JOB_TTL_MINUTES`) and marks them `expired`.
- [ ] Worker takes a **pipeline callable** so it can be tested with a fake pipeline (no dev endpoint needed).

**IDE Command**
```text
PHASE 6 specifics:
Implement the job system per progress.md Phase 6.
- job_worker.py: bounded queue, ONE worker, timeout, status updates via job_crud, startup recovery of stale jobs,
  pluggable pipeline callable (so tests can inject a fake).
- cleanup_service.py: scheduled deletion of expired job folders + status=expired.
- Job ids use secrets.token_urlsafe; store only a hashed client key, never a raw IP.
- Audio endpoints resolve paths from the job id only; reject anything else.
- Do NOT add a dev/echo endpoint. Include a runnable self-check in job_worker.py's __main__ that enqueues a fake job
  and prints state transitions.
```

**Verify:** run the worker self-check (`python -m core.services.jobs.job_worker`); confirm queued → processing → completed, and expiry removes files.

**Acceptance Criteria**
- [ ] State transitions correct; queue-full returns 429.
- [ ] Timeout marks job failed; expired jobs' files are deleted.
- [ ] Unknown/other job ids can't read files.

---

## PHASE 7 — Text-to-Voice API

**Goal:** First real end-to-end workflow through the API.

**Tasks**
- [ ] `POST /v1/tts` body: `voice_id`, `text`, `language`, optional `speed` (0.5–2.0) → returns `{job_id}`.
- [ ] Validate text length ≤ `MAX_TEXT_CHARS` (trimmed, non-empty), voice exists + `status=available` + `model_type=openvoice_reference`, language supported by that voice.
- [ ] Worker runs `openvoice_service`; output WAV with AI-label metadata; MP3 on request.
- [ ] Friendly error codes: `MODEL_NOT_READY`, `UNSUPPORTED_LANGUAGE`, `INFERENCE_FAILED`, `TIMEOUT`, `QUEUE_FULL`. No paths or stack traces to clients.
- [ ] Response includes `ai_generated: true`.

**IDE Command**
```text
PHASE 7 specifics:
Add POST /v1/tts across schemas → controller → route, wiring job_worker to openvoice_service.
- Request schema with Pydantic Field constraints; text length from settings; speed range 0.5-2.0.
- Controller validates voice type/status/language and raises explicit HTTPExceptions (400/404/429).
- Worker maps exceptions to error_code/error_message pairs listed in progress.md Phase 7.
- Output gets AI-label metadata via audio_service; MP3 export via ?format=mp3.
```

**Verify:** Swagger: POST `/v1/tts` → poll `/v1/jobs/{id}` → download WAV/MP3 → listen.

**Acceptance Criteria**
- [ ] Real speech generated through the API in the chosen voice.
- [ ] Second request faster than the first (models cached).
- [ ] Invalid input returns friendly, standard-format errors.
- [ ] **Milestone: backend MVP.** (You commit manually.)

---

## PHASE 8 — Voice-to-Voice API

**Goal:** Second workflow through the API.

**Files:** `core/services/audio/validation_service.py` (new), plus extend schemas/controller/routes.

**Tasks**
- [ ] `POST /v1/vc` (multipart): `file`, `voice_id`, `pitch_shift` (-12…12).
- [ ] `validation_service.py`: extension allow-list (wav, mp3, m4a, ogg, webm, flac) **and** real content probing; size cap; `MAX_AUDIO_SECONDS` cap; server-generated filenames (uuid) — never trust client filenames.
- [ ] Preprocess to the model's sample rate (mono); save `source.wav`; worker runs `rvc_service`; output `output.wav`.
- [ ] Only `model_type=rvc_model` and `status=available` voices accepted.

**IDE Command**
```text
PHASE 8 specifics:
Add POST /v1/vc (multipart) per progress.md Phase 8.
- Create core/services/audio/validation_service.py: allow-listed extensions, ffprobe content verification, size/duration limits
  from settings, uuid filenames. Stream the upload to disk with a hard size cutoff.
- Wire job_worker to rvc_service. Save source.wav and output.wav under storage/tmp/<job_id>/.
- Return friendly errors for corrupt/too-long/too-big/wrong-voice-type uploads.
```

**Verify:** upload a speech clip in Swagger → poll → download converted audio; try a corrupt file and an oversized file.

**Acceptance Criteria**
- [ ] Upload → poll → download works; source and output both retrievable.
- [ ] Bad uploads rejected with clear messages.

---

## PHASE 9 — Custom Reference Audio (OpenVoice only)

**Goal:** Users may supply a reference for a voice they own or are authorized to use. **RVC custom voices stay out of scope** — a short recording does not create an RVC model.

**Tasks**
- [ ] `POST /v1/tts/custom` (multipart): `reference_file`, `text`, `language`, `consent` (must be `true`, otherwise 400).
- [ ] `validation_service.py` gains reference-quality analysis: duration range, silence ratio, clipping, very low volume, unsupported sample rate — thresholds from settings; returns actionable tips ("record 10+ seconds in a quiet room").
- [ ] Embedding computed for this job only; reference stored only in the job folder and deleted with the job; **never** added to the catalog.
- [ ] Only the fact that consent was given may be logged (no audio, no content).

**IDE Command**
```text
PHASE 9 specifics:
Add POST /v1/tts/custom per progress.md Phase 9.
- Extend validation_service.py with reference-quality analysis (thresholds in config.py).
- consent must be true; otherwise HTTP 400 with a clear message.
- Reference audio lives only in storage/tmp/<job_id>/ and is removed by cleanup; never saved to voices/ or the DB catalog.
- Keep this path completely separate from the RVC code path.
```

**Verify:** try good, too-short, silent, clipped, and no-consent requests.

**Acceptance Criteria**
- [ ] Good reference → speech in that voice; bad reference → helpful error, no crash.
- [ ] Reference file is deleted when the job expires.

---

## PHASE 10 — Security & Limits Hardening

**Goal:** Safe enough for public use.

**Files:** extend `core/apis/middleware.py`, `core/config.py`, `core/apis/api.py`, `core/services/audio/validation_service.py`.

**Tasks**
- [ ] Per-IP rate limits on `/v1/tts`, `/v1/tts/custom`, `/v1/vc` plus a global daily cap; 429 with `Retry-After` and friendly text.
- [ ] Max concurrent active jobs per client (hashed IP).
- [ ] Re-verify uploads (size cutoff, content probing, extension allow-list); confirm no endpoint accepts a filesystem path.
- [ ] Production startup self-check: refuse to start if `ENV=production` with `DEBUG=true`, wildcard CORS, or default secrets.
- [ ] Logs never contain keys, full scripts, file paths, or raw IPs.
- [ ] Retention policy (what is stored, where, how long) written into a **README.md section** — no new files.

**IDE Command**
```text
PHASE 10 specifics:
Harden the backend per progress.md Phase 10 and agent.md section 3.2.
1. Rate limiting + per-client concurrent job cap in middleware.py (in-memory is fine; keys are hashed IPs).
2. Production self-check at startup in api.py; disable /docs and any debug routes when ENV=production.
3. Audit every endpoint: path traversal, oversize upload, fake extension, huge text, missing consent, guessing other job ids.
   Fix what you find and list any remaining risks.
4. Add a "Data Retention & Privacy" section to README.md (do not create any other file).
```

**Verify:** hammer an endpoint until it returns 429; try `../` and absolute-path inputs; start with a bad production config and confirm it refuses.

**Acceptance Criteria**
- [ ] Limits trigger with friendly messages.
- [ ] No endpoint accepts client-supplied filesystem paths.
- [ ] README documents retention and privacy.

---

## PHASE 11 — Optional Language Helpers

**Goal:** Optional Groq/Gemini/OpenRouter helpers. **The app must work fully with all keys absent.**

**Files:** `core/services/llm/llm_service.py`, extend schemas/controller/routes.

**Tasks**
- [ ] Provider interface with fallback chain, ~10 s timeout, no retry storms; keys from `.env` only; feature flags per provider.
- [ ] `POST /v1/text/cleanup`: returns `{original, cleaned}` — punctuation/number/abbreviation cleanup for better TTS, never changing meaning.
- [ ] No quota/key/error → feature reports "unavailable"; generation unaffected.
- [ ] No user audio ever sent to third parties.

**IDE Command**
```text
PHASE 11 specifics:
Create core/services/llm/llm_service.py and POST /v1/text/cleanup per progress.md Phase 11.
- Fallback chain Groq -> Gemini -> OpenRouter -> none, 10s timeout, keys only from settings.
- On any failure return a clean "feature unavailable" result; never break TTS.
- Expose availability via GET /v1/config/public feature flags.
- Never send audio to external services; send only text the user submitted.
```

**Verify:** run with no keys (expect "unavailable"), then with one key.

**Acceptance Criteria**
- [ ] App works with zero keys. 
- [ ] Failures never affect generation.

---

## PHASE 12 — Tests & Backend Audit

**Goal:** Confidence that everything holds together.

**Files:** `backend/tests/test_services.py`, `backend/tests/test_api.py` (2-file limit — group tests by section).

**Tasks**
- [ ] Service tests: audio utils (generated tones), validation rules, job lifecycle with a fake pipeline, cleanup/expiry, catalog sync.
- [ ] API tests (httpx/TestClient, pipelines mocked): health, voices, tts/vc/custom success + all error paths, rate limit, 404/429 formats, no-path-leak checks.
- [ ] One optional real-model test per pipeline marked `slow`.
- [ ] README.md updated: Windows setup, model download steps, env vars, run commands, troubleshooting, supported voices & known limitations.
- [ ] Final audit report against the Definition of Done (backend items), plus security, privacy, and license review.

**IDE Command**
```text
PHASE 12 specifics:
1. Write backend/tests/test_services.py and backend/tests/test_api.py (max 2 files; use clearly separated sections inside each).
   Mock pipelines; add one @pytest.mark.slow real-model test per pipeline.
2. Update README.md (setup on Windows, models, env vars, run, tests, troubleshooting, limitations, retention). No other docs.
3. FINAL AUDIT (do not add features): report pass/fail with evidence for each backend Definition-of-Done item, then a security
   review (paths, uploads, secrets, rate limits), privacy review (what is stored and for how long), license review
   (every model/dependency; flag uncertain ones), and top 10 risks ordered by severity.
```

**Verify:** `pytest` (fast tests) and `pytest -m slow` once on your GPU.

**Acceptance Criteria**
- [ ] All fast tests pass; slow tests pass at least once.
- [ ] README lets a fresh clone run the backend.
- [ ] Audit has no unresolved high-severity findings.

---

## PHASE 13 — Deployment Preparation (STOP FOR APPROVAL)

**Goal:** Backend ready to run on a public host. Free GPU tiers have queues, cold starts, and time limits — a public site does not guarantee unlimited free inference.

**Tasks**
- [ ] IDE compares 3 realistic hosting options for a GPU-capable backend (VRAM needs, free-tier limits, cold starts, cost) and recommends one. **Waits for your choice.**
- [ ] `Dockerfile` + `.dockerignore` (FFmpeg, pinned CUDA/PyTorch, models mounted as a volume, non-root user).
- [ ] Production env checklist (every secret, every manual step) added to a **README.md section**.
- [ ] Health/readiness checks and logging suitable for hosted logs.
- [ ] Graceful "service busy" behavior when the GPU/queue is unavailable.

**IDE Command**
```text
PHASE 13 specifics — STOP FOR APPROVAL after step 1:
1. Compare 3 realistic hosting options for this backend (GPU needs, free-tier limits, cold starts, cost). Recommend one. Wait for my choice.
2. After my choice: create backend/Dockerfile and .dockerignore; add a "Deployment" section to README.md with step-by-step instructions
   and the full list of secrets/env vars I must set. Do not deploy anything yourself.
3. Confirm production-mode behavior: docs/debug disabled, strict CORS, self-check passes, health endpoint suitable for the host.
Do not deploy, publish, or configure public hosting automatically. Complete all development and local testing first. After presenting the final local test results, stop and wait for my explicit approval before beginning any hosting or deployment work.
```

**Acceptance Criteria**
- [ ] Container builds and runs both workflows locally.
- [ ] No secrets in the image or repo.
- [ ] README deployment steps are complete.

---

## 🔒 FRONTEND (LOCKED — do not start until you say so)

`agent.md` freezes frontend work. These phases stay locked until **you** paste the unlock command. When unlocked, the IDE must follow `code-standards/frontend-standards`, the `impeccable` and `web-design-guidelines` skills, and the 2-code-files-per-folder rule (so it must propose the folder plan and wait for your approval before coding).

**Unlock Command (only when backend Phase 12 is done)**
```text
UNLOCK FRONTEND. Follow agent.md section 6. Start Frontend Phase F1 from progress.md.
First inspect the backend API contracts in core/apis/schemas, then present a folder plan that respects the
2-code-files-per-folder rule and frontend-standards. STOP FOR APPROVAL before writing code.
```

| Phase | Outline (details re-planned at unlock) |
|-------|-----------------------------------------|
| F1 | Next.js + TypeScript + Tailwind foundation, typed API client, landing page, voice library with previews |
| F2 | Text-to-voice screen: script editor, voice/language/speed, optional custom reference + consent, job polling, result + download |
| F3 | Voice-to-voice screen: upload + microphone recording, target voice, source vs converted playback |
| F4 | Audio Studio panel (A/B compare, session history, AI label), limitations/privacy/responsible-use pages |
| F5 | Integration testing against the live backend, production build, frontend deployment |

---

## FINAL RELEASE CHECKLIST (Definition of Done)

A visitor can:
- [ ] 1. Open the website. *(needs frontend)*
- [ ] 2. Enter a script or upload/record audio.
- [ ] 3. Choose a supported target voice.
- [ ] 4. Generate or convert audio.
- [ ] 5. See processing status and meaningful errors.
- [ ] 6. Play and download the output.
- [ ] 7. Use the app without supplying private API keys.

Plus:
- [ ] Reasonable usage limits enforced and disclosed.
- [ ] Supported voices, privacy practices, and limitations disclosed.
- [ ] Every voice has verified license/permission.
- [ ] All output labeled AI-generated / AI-converted (metadata + API field + UI).
- [ ] Temporary files deleted per the retention policy.
- [ ] Graceful behavior when external quotas or the GPU are unavailable.

**Do NOT build until all of the above are done:** user accounts/payments, large voice catalog, custom RVC training, batch generation, multi-language UI, developer API, persistent Postgres, multi-GPU scaling.

---

## Troubleshooting Prompts

```text
The last step failed with this error: <paste error>. Diagnose the root cause in 3 sentences, then fix it within the current
phase scope only, without upgrading unrelated dependencies. Follow agent.md.
```

```text
Review what you just wrote against agent.md: docstrings on every function, logging formats, layer boundaries,
error handling, no secrets, 2-code-files-per-folder. List violations and fix them.
```

---

## Progress Log

| Date       | Phase | Status | Notes / Issues |
|------------|-------|--------|----------------|
| 2026-10-17 | 0     | DONE   | Windows 11, RTX 3050 6GB CUDA PyTorch 2.6.0+cu124, folders verified. |
| 2026-10-17 | 1     | DONE   | Core config, SQLite singleton, FastAPI, hardened headers, /v1/health. |
| 2026-10-17 | 2     | DONE   | AudioService with probe, to_wav, in-place safe resample, normalize, MP3 export. |
| 2026-10-17 | 3     | DONE   | OpenVoice V2 + MeloTTS (EN) + MMS-TTS (Hindi, Marathi) + Hinglish translit. |
| 2026-10-17 | 4     | DONE   | RVC Service with weights_only security, RMVPE, SynthesizerTrn, mutual GPU unload. |
| 2026-10-17 | 5     | DONE   | Voice catalog synced from voices.json, masking server paths, preview streaming. |
| 2026-10-17 | 6     | DONE   | JobWorker FIFO daemon queue, DB status recovery, streaming WAV/MP3 outputs. |
| 2026-10-17 | 7     | DONE   | POST /v1/tts with language validation and AI-label metadata injection. |
| 2026-10-17 | 8     | DONE   | POST /v1/vc for voice conversion with pitch shift (-12 to +12). |
| 2026-10-17 | 9     | DONE   | POST /v1/tts/custom with consent check and acoustic quality analyzer. |
| 2026-10-17 | 10    | DONE   | RateLimitMiddleware, active job client quotas, startup self-check. |
| 2026-10-17 | 11    | DONE   | POST /v1/text/cleanup with rule-based abbreviations, numbers, and currencies. |
| 2026-10-17 | 12    | DONE   | Comprehensive test suite: 51/51 tests passing across test_api and test_services. |
| 2026-10-17 | 13    | DONE   | Dockerfile with FFmpeg & non-root user, .dockerignore, local run docs. |
| 2026-10-17 | F     | DONE   | Vite + React 19 + TypeScript dark glassmorphism studio, audio waveform player, live polling, 2-file folder compliance. |

**Current phase:** All Phases (0–13 + F) Complete
**Blockers:** none

---

## Findings & Decisions (paste IDE reports here)

- **Hardware:** Windows 11 laptop, RTX 3050 6GB, 16GB RAM, i5-12450HX
- **OpenVoice V2 & Multilingual TTS:**
  - Base English: MeloTTS (`myshell-ai/MeloTTS-English`) + OpenVoice V2 ToneColorConverter.
  - Base Hindi: Meta MMS-TTS (`facebook/mms-tts-hin`) + OpenVoice V2 ToneColorConverter + Hinglish transliteration helper.
  - Base Marathi: Meta MMS-TTS (`facebook/mms-tts-mar`) + OpenVoice V2 ToneColorConverter.
  - Peak inference VRAM: ~450MB on GPU; execution time: 3.2s (EN) to 9.8s (HI/MR).
- **RVC:**
  - Embedded clean inference architecture with `SynthesizerTrn` and `RMVPE` without external bulky dependencies.
  - Strict security validation: `weights_only=True` and SHA-256 checkpoint hash checks.
  - Mutual GPU unloading: `RVCService.unload_all()` and `OpenVoiceService.unload_all()` ensure peak VRAM stays under 2GB.
- **Voices & licenses:**
  - `my_voice`: Operator reference audio, authorized for local studio synthesis (`["en", "hi", "mr"]`).
  - `nttsai`: Experimental RVC model, kept strictly local/experimental (`EXPOSE_EXPERIMENTAL_VOICES=False` default).
- **Deployment:**
  - Python 3.10-slim base with system FFmpeg, libsndfile1, non-root user `studio` (UID 10000).
  - Data retention: `CleanupService` scheduled job purging expired artifacts (> 60 minutes TTL).