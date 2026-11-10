import io
"""
API and integration test suite for AI Voice Studio core endpoints.
"""

import pytest
from pathlib import Path
import numpy as np
import soundfile as sf
from sqlalchemy.orm import Session
from starlette.testclient import TestClient
from core.apis.api import app

client = TestClient(app)


def test_health_check_endpoint() -> None:
    """
    Verify GET /v1/health returns HTTP 200 with standard status and ISO timestamp.
    """
    response = client.get("/v1/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert "time" in payload
    assert len(payload["time"]) > 10


def test_security_headers_present() -> None:
    """
    Verify required hardened HTTP security headers are returned on all responses.
    """
    response = client.get("/v1/health")
    assert response.status_code == 200

    headers = response.headers
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert "Content-Security-Policy" in headers
    assert "Strict-Transport-Security" in headers


def test_not_found_returns_standard_error_envelope() -> None:
    """
    Verify 404 responses conform to the standard {"error": {"code", "message"}} format.
    """
    response = client.get("/v1/nonexistent-endpoint")
    assert response.status_code == 404

    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "404"
    assert "message" in payload["error"]


def test_forced_error_masks_internal_stack_trace() -> None:
    """
    Verify unhandled exceptions return a generic 500 message with no leaked stack trace.
    """
    error_client = TestClient(app, raise_server_exceptions=False)

    @app.get("/v1/test-crash-endpoint")
    def trigger_crash() -> None:
        raise RuntimeError("Simulated internal database crash with sensitive secret: SECRET123")

    response = error_client.get("/v1/test-crash-endpoint")
    assert response.status_code == 500

    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert payload["error"]["message"] == "An unexpected error occurred. Please try again later."
    assert "SECRET123" not in response.text
    assert "Traceback" not in response.text


def test_lifespan_initializes_database() -> None:
    """
    Verify application lifespan initializes the database and creates tables.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/health")
        assert response.status_code == 200


def test_list_voices_returns_registered_voices_and_masks_asset_paths() -> None:
    """
    Verify GET /v1/voices returns available voices without exposing server asset paths.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/voices")
        assert response.status_code == 200
        payload = response.json()

        assert "voices" in payload
        assert "total" in payload
        assert payload["total"] >= 1

        voice_ids = [v["id"] for v in payload["voices"]]
        assert "client_ref" in voice_ids
        # Experimental voice must not be listed by default
        assert "nttsai" not in voice_ids

        for voice in payload["voices"]:
            assert "asset_file" not in voice
            assert "reference/client_ref.wav" not in str(voice)
            assert "checkpoints" not in str(voice)
            assert "voices/" not in str(voice)


def test_get_voice_by_id_success() -> None:
    """
    Verify GET /v1/voices/{id} returns profile details for an available voice.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/voices/client_ref")
        assert response.status_code == 200
        payload = response.json()

        assert payload["id"] == "client_ref"
        assert payload["name"] == "Client Voice"
        assert payload["model_type"] == "chatterbox_reference"
        assert isinstance(payload["has_preview"], bool)
        assert "asset_file" not in payload


def test_get_experimental_voice_returns_404_by_default() -> None:
    """
    Verify experimental voice returns 404 when EXPOSE_EXPERIMENTAL_VOICES is False.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/voices/nttsai")
        assert response.status_code == 404
        payload = response.json()
        assert "error" in payload
        assert payload["error"]["code"] == "404"


def test_get_voice_preview_streams_audio() -> None:
    """
    Verify GET /v1/voices/{id}/preview returns streaming WAV audio for client_ref using generated tone.
    """
    import soundfile as sf
    import numpy as np
    from core.config import get_settings
    settings = get_settings()

    preview_dir = settings.voices_path / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "client_ref_preview.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(preview_path), tone, sr)

    try:
        with TestClient(app) as test_client:
            response = test_client.get("/v1/voices/client_ref/preview")
            assert response.status_code == 200
            assert "audio/wav" in response.headers.get("Content-Type", "")
            assert len(response.content) > 1000
    finally:
        if preview_path.exists():
            preview_path.unlink()


def test_get_voice_preview_for_experimental_returns_404() -> None:
    """
    Verify preview request for an experimental voice without preview returns 404.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/voices/nttsai/preview")
        assert response.status_code == 404


def test_get_public_config_endpoint() -> None:
    """
    Verify GET /v1/config/public returns system quotas, limits, and safe capability flags.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/config/public")
        assert response.status_code == 200
        payload = response.json()

        assert "max_text_chars" in payload
        assert "job_ttl_minutes" in payload
        assert "queue_max_size" in payload
        assert "job_timeout_seconds" in payload
        assert payload["expose_experimental_voices"] is False
        assert "features" in payload


def test_startup_self_check_enforces_no_experimental_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that application fails to start if EXPOSE_EXPERIMENTAL_VOICES is True in production.
    """
    from core.config import get_settings
    settings = get_settings()

    monkeypatch.setattr(settings, "ENV", "production")
    monkeypatch.setattr(settings, "EXPOSE_EXPERIMENTAL_VOICES", True)

    with pytest.raises(RuntimeError, match="Startup self-check failed"):
        with TestClient(app):
            pass


# ==============================================================================
# SECTION: JOB SYSTEM & AUDIO ARTIFACT ENDPOINTS
# ==============================================================================


def test_get_job_status_success_and_output_urls() -> None:
    """
    Verify GET /v1/jobs/{job_id} returns accurate state and resource URLs for a completed job.
    """
    import secrets
    from datetime import datetime, timezone, timedelta
    from core.database.database import SessionLocal
    from core.cruds.job_crud import JobCRUD
    from core.config import get_settings

    settings = get_settings()
    token = secrets.token_urlsafe(16)
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)

    db = SessionLocal()
    try:
        JobCRUD.create_job(
            db,
            {
                "id": token,
                "kind": "tts",
                "voice_id": "client_ref",
                "status": "completed",
                "progress_message": "Completed test job",
                "client_key": "hashed_client_key_123",
                "expires_at": expires,
            },
        )
    finally:
        db.close()

    with TestClient(app) as test_client:
        response = test_client.get(f"/v1/jobs/{token}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"] == token
        assert payload["status"] == "completed"
        assert payload["output_url"] == f"/v1/audio/{token}/output"


def test_get_job_status_not_found() -> None:
    """
    Verify GET /v1/jobs/{unknown_id} returns 404 with standard error envelope.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/jobs/unknown_token_12345")
        assert response.status_code == 404
        payload = response.json()
        assert "error" in payload
        assert payload["error"]["code"] == "404"


def test_get_audio_output_wav_and_mp3(tmp_path: Path) -> None:
    """
    Verify GET /v1/audio/{job_id}/output serves WAV and converts on-demand to MP3.
    """
    import secrets
    from datetime import datetime, timezone, timedelta
    from core.database.database import SessionLocal
    from core.cruds.job_crud import JobCRUD
    from core.config import get_settings
    from core.services.audio.audio_service import AudioService

    settings = get_settings()
    token = secrets.token_urlsafe(16)
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)

    # Create dummy audio file in job directory
    job_dir = settings.tmp_path / token
    job_dir.mkdir(parents=True, exist_ok=True)
    wav_file = job_dir / "output.wav"
    
    import soundfile as sf
    import numpy as np
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(wav_file), tone, sr)

    db = SessionLocal()
    try:
        JobCRUD.create_job(
            db,
            {
                "id": token,
                "kind": "tts",
                "voice_id": "client_ref",
                "status": "completed",
                "client_key": "hashed_client_key_123",
                "expires_at": expires,
            },
        )
    finally:
        db.close()

    try:
        with TestClient(app) as test_client:
            # 1. Fetch WAV
            wav_resp = test_client.get(f"/v1/audio/{token}/output?format=wav")
            assert wav_resp.status_code == 200
            assert "audio/wav" in wav_resp.headers.get("Content-Type", "")
            assert len(wav_resp.content) > 100

            # 2. Fetch MP3 (converted on demand)
            mp3_resp = test_client.get(f"/v1/audio/{token}/output?format=mp3")
            assert mp3_resp.status_code == 200
            assert "audio/mpeg" in mp3_resp.headers.get("Content-Type", "")
            assert len(mp3_resp.content) > 100
    finally:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)


def test_get_audio_output_rejected_when_not_completed() -> None:
    """
    Verify audio output download is rejected with 400 when job is still processing.
    """
    import secrets
    from datetime import datetime, timezone, timedelta
    from core.database.database import SessionLocal
    from core.cruds.job_crud import JobCRUD
    from core.config import get_settings

    settings = get_settings()
    token = secrets.token_urlsafe(16)
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)

    db = SessionLocal()
    try:
        JobCRUD.create_job(
            db,
            {
                "id": token,
                "kind": "tts",
                "voice_id": "client_ref",
                "status": "processing",
                "client_key": "hashed_client_key_123",
                "expires_at": expires,
            },
        )
    finally:
        db.close()

    with TestClient(app) as test_client:
        response = test_client.get(f"/v1/audio/{token}/output")
        assert response.status_code == 400
        payload = response.json()
        assert "not ready" in payload["error"]["message"].lower()


def test_delete_job_endpoint() -> None:
    """
    Verify DELETE /v1/jobs/{job_id} purges disk artifacts and removes database record.
    """
    import secrets
    from datetime import datetime, timezone, timedelta
    from core.database.database import SessionLocal
    from core.cruds.job_crud import JobCRUD
    from core.config import get_settings

    settings = get_settings()
    token = secrets.token_urlsafe(16)
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)

    job_dir = settings.tmp_path / token
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "output.wav").write_bytes(b"dummy wav data")

    db = SessionLocal()
    try:
        JobCRUD.create_job(
            db,
            {
                "id": token,
                "kind": "tts",
                "voice_id": "client_ref",
                "status": "completed",
                "client_key": "hashed_client_key_123",
                "expires_at": expires,
            },
        )
    finally:
        db.close()

    with TestClient(app) as test_client:
        del_resp = test_client.get(f"/v1/jobs/{token}")
        assert del_resp.status_code == 200

        del_action = test_client.delete(f"/v1/jobs/{token}")
        assert del_action.status_code == 200
        assert del_action.json()["status"] == "deleted"

        # Confirm folder removed
        assert not job_dir.exists()

        # Confirm 404 on subsequent get
        after_resp = test_client.get(f"/v1/jobs/{token}")
        assert after_resp.status_code == 404


def test_text_cleanup_endpoint() -> None:
    """
    Verify POST /v1/text/cleanup normalizes abbreviations and currencies.
    """
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/text/cleanup",
            json={"text": "Hello Dr. Stone! Your coffee costs $4.50... amazing!!!", "language": "en"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "doctor Stone" in payload["cleaned"]
        assert "4.50 dollars" in payload["cleaned"]
        assert "amazing!" in payload["cleaned"]
        assert payload["original"].startswith("Hello Dr. Stone")


def test_create_tts_job_unsupported_language_returns_400() -> None:
    """
    Verify POST /v1/tts rejects languages not in supported catalog with HTTP 400.
    """
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts",
            json={"text": "Bonjour le monde", "voice_id": "client_ref", "language": "fr"},
        )
        assert response.status_code == 400
        payload = response.json()
        assert "error" in payload
        assert "not supported" in payload["error"]["message"].lower()


def test_create_tts_job_unknown_voice_returns_404() -> None:
    """
    Verify POST /v1/tts with non-existent voice_id returns HTTP 404.
    """
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts",
            json={"text": "Hello world", "voice_id": "nonexistent_voice_999", "language": "en"},
        )
        assert response.status_code == 404


def test_create_tts_job_success_enqueues_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify POST /v1/tts accepts valid payload, creates job in DB, and returns HTTP 202.
    """
    from core.controllers.job_controller import job_worker

    # Mock enqueue to prevent background inference during fast endpoint acceptance test
    monkeypatch.setattr(job_worker, "enqueue_job", lambda j_id: None)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts",
            json={"text": "Testing production voice studio pipeline.", "voice_id": "client_ref", "language": "en"},
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        assert "job_id" in payload
        job_id = payload["job_id"]

        # Confirm job exists in DB
        status_resp = test_client.get(f"/v1/jobs/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == job_id


def test_create_custom_tts_without_consent_returns_400() -> None:
    """
    Verify POST /v1/tts/custom without explicit voice owner consent returns HTTP 400.
    """
    import io

    fake_wav = io.BytesIO(b"RIFFdummydataWAVEfmt ")
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts/custom",
            data={"text": "Voice cloning test", "language": "en", "consent": "false"},
            files={"file": ("test.wav", fake_wav, "audio/wav")},
        )
        assert response.status_code == 400
        assert "consent is required" in response.json()["error"]["message"].lower()


def test_create_custom_tts_with_invalid_extension_returns_400() -> None:
    """
    Verify POST /v1/tts/custom with forbidden file extension returns HTTP 400.
    """
    import io

    fake_file = io.BytesIO(b"executable content")
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts/custom",
            data={"text": "Voice cloning test", "language": "en", "consent": "true"},
            files={"file": ("malicious.exe", fake_file, "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "unsupported audio format" in response.json()["error"]["message"].lower()


def test_create_custom_tts_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify POST /v1/tts/custom with valid audio and consent succeeds and returns HTTP 202.
    """
    import io
    import soundfile as sf
    import numpy as np
    from core.controllers.job_controller import job_worker

    monkeypatch.setattr(job_worker, "enqueue_job", lambda j_id: None)

    # Generate 4s sine wave
    sr = 22050
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="WAV")
    wav_bytes = buf.getvalue()

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts/custom",
            data={"text": "Cloned custom voice test.", "language": "en", "consent": "true"},
            files={"file": ("sample_reference.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        assert "job_id" in payload


def test_create_vc_job_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify POST /v1/vc returns HTTP 501 Not Implemented since voice conversion is coming soon.
    """
    import io

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/vc",
            data={"voice_id": "client_ref", "pitch_shift": "0"},
            files={"file": ("source.wav", io.BytesIO(b"dummy wav data"), "audio/wav")},
        )
        assert response.status_code == 501
        payload = response.json()
        assert "disabled" in payload["error"]["message"].lower() or "coming soon" in payload["error"]["message"].lower()


def test_job_status_chunk_progress_fields() -> None:
    """
    Verify GET /v1/jobs/{job_id} exposes chunk_done, chunk_total, eta_seconds, and completed_at.
    """
    from datetime import datetime, timezone
    from core.models.job_model import JobModel
    from core.cruds.job_crud import JobCRUD
    from core.database.database import SessionLocal

    import uuid
    job_id = f"test_prog_{uuid.uuid4().hex[:8]}"
    job_data = {
        "id": job_id,
        "kind": "tts",
        "voice_id": "client_ref",
        "status": "processing",
        "client_key": "test_client_hash",
        "progress_message": "Synthesising chunk 2 of 5",
        "expires_at": datetime.now(timezone.utc),
    }

    with TestClient(app) as test_client:
        db = SessionLocal()
        try:
            JobCRUD.create_job(db, job_data)
            JobCRUD.update_chunk_progress(db, job_id=job_id, chunk_done=2, chunk_total=5, eta_seconds=42)
        finally:
            db.close()

        response = test_client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"] == job_id
        assert payload["chunk_done"] == 2
        assert payload["chunk_total"] == 5
        assert payload["eta_seconds"] == 42
        assert payload["status"] == "processing"


def test_client_daily_quota_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify POST /v1/tts rejects submission with HTTP 429 when client (hashed IP) exceeds daily minutes quota.
    """
    from unittest.mock import MagicMock
    from core.cruds.job_crud import JobCRUD
    from core.cruds.voice_crud import VoiceCRUD

    mock_voice = MagicMock()
    mock_voice.id = "client_ref"
    mock_voice.name = "Client Voice"
    mock_voice.status = "available"
    mock_voice.model_type = "openvoice_reference"
    mock_voice.languages = ["en"]
    monkeypatch.setattr(VoiceCRUD, "get_voice_by_id", lambda db, voice_id=None, **kwargs: mock_voice)

    # Mock client daily usage to 135 minutes (over the 120 min cap)
    monkeypatch.setattr(JobCRUD, "get_daily_generated_minutes", lambda db, client_key: 135.0)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts",
            json={"voice_id": "client_ref", "text": "This should be blocked by quota.", "language": "en"},
        )
        assert response.status_code == 429
        body = response.json()
        error_msg = body.get("detail") or body.get("error", {}).get("message", "")
        assert "Daily audio generation quota reached" in error_msg


def test_resume_job_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Verify POST /v1/jobs/{job_id}/resume validates state, clears error, and requeues the job.
    """
    import uuid
    from datetime import datetime, timezone
    from core.cruds.job_crud import JobCRUD
    from core.controllers.job_controller import job_worker, settings
    from core.database.database import SessionLocal

    monkeypatch.setattr(job_worker, "enqueue_job", lambda j_id: None)
    monkeypatch.setattr(settings, "TMP_DIR", str(tmp_path))

    job_id = f"test_res_{uuid.uuid4().hex[:8]}"
    job_dir = tmp_path / job_id
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Create 2 finished chunk files
    (chunks_dir / "chunk_0000.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    (chunks_dir / "chunk_0001.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    # And 1 partial file that should be cleaned
    (chunks_dir / "chunk_0002.wav.part").write_bytes(b"PARTIAL")

    job_data = {
        "id": job_id,
        "kind": "tts",
        "voice_id": "client_ref",
        "status": "failed",
        "client_key": "test_client_hash",
        "progress_message": "Inference failed.",
        "error_code": "INFERENCE_FAILED",
        "error_message": "GPU OOM",
        "chunk_done": 2,
        "chunk_total": 4,
        "expires_at": datetime.now(timezone.utc),
    }

    with TestClient(app) as test_client:
        db = SessionLocal()
        try:
            JobCRUD.create_job(db, job_data)

            response = test_client.post(f"/v1/jobs/{job_id}/resume")
            assert response.status_code == 202
            payload = response.json()
            assert payload["job_id"] == job_id
            assert payload["status"] == "queued"
            assert "from chunk 2" in payload["message"]

            # Incomplete .part must have been cleaned
            assert not (chunks_dir / "chunk_0002.wav.part").exists()
            # Finished chunks must be preserved
            assert (chunks_dir / "chunk_0000.wav").exists()
            assert (chunks_dir / "chunk_0001.wav").exists()

            # Database state must be updated
            refreshed = JobCRUD.get_job_by_id(db, job_id)
            assert refreshed.status == "queued"
            assert refreshed.error_code is None
            assert refreshed.error_message is None
        finally:
            db.close()


def test_resume_after_mid_job_crash_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify _run_long_form_tts preserves completed chunks on failure,
    cleans up incomplete .part files, recomputes speaker embedding on resume,
    and purges chunk directory only upon successful join.
    """
    import soundfile as sf
    import numpy as np
    from core.services.jobs.job_worker import _run_long_form_tts
    from core.controllers.job_controller import settings

    job_id = "test_crash_resume_789"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_wav = job_dir / "output.wav"

    ref_wav = tmp_path / "ref.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(ref_wav), 0.5 * np.sin(2 * np.pi * 440 * t), sr)

    text = "First paragraph sentence one. First paragraph sentence two.\n\nSecond paragraph here."
    attempt = {"count": 0}

    class MockChatterboxResult:
        sample_rate = sr
        duration_seconds = 0.5

    class MockChatterboxService:
        def generate(self, text, reference_audio_path, output_path, speed=1.0, language="hi"):
            attempt["count"] += 1
            # Write a valid wav to the requested output path (.part)
            tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sr, endpoint=False))
            sf.write(output_path, tone, sr, format="WAV")
            if attempt["count"] == 2:
                # Simulate a mid-job crash during chunk 2
                raise RuntimeError("Simulated mid-job worker crash!")
            return MockChatterboxResult()

    mock_service = MockChatterboxService()

    # Run 1: Fails on second chunk
    chunks_dir = job_dir / "chunks"
    with pytest.raises(RuntimeError, match="Simulated mid-job worker crash"):
        _run_long_form_tts(
            job_id=job_id,
            text=text,
            ref_path=ref_wav,
            out_wav=out_wav,
            speed=1.0,
            language="hi",
            job_dir=job_dir,
            pipeline_service=mock_service,
        )

    # Verify chunk 0 was completed and preserved
    assert chunks_dir.exists()
    assert (chunks_dir / "chunk_0000.wav").exists()
    # Incomplete .part must have been cleaned up by error handling
    assert not (chunks_dir / "chunk_0001.wav.part").exists()
    assert not out_wav.exists()

    # Run 2: Resume (skip chunk 0, synthesize chunk 1, join, purge chunks)
    attempt["count"] = 0  # reset so chunk 1 doesn't fail
    res = _run_long_form_tts(
        job_id=job_id,
        text=text,
        ref_path=ref_wav,
        out_wav=out_wav,
        speed=1.0,
        language="hi",
        job_dir=job_dir,
        pipeline_service=mock_service,
    )

    # Assertions
    assert out_wav.exists()
    assert res["duration_seconds"] > 0
    # Chunks folder must be purged only after successful join
    assert not chunks_dir.exists(), "Chunks folder should be purged after successful join"


def test_public_config_contains_seconds_per_audio_second() -> None:
    """
    Verify GET /v1/config/public includes measured compute seconds per audio second for UI estimates.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/config/public")
        assert response.status_code == 200
        payload = response.json()
        assert "seconds_per_audio_second" in payload
        assert payload["seconds_per_audio_second"] > 0
        assert "compute_seconds_per_audio_second" in payload


def test_job_status_includes_stage_and_timestamps() -> None:
    """
    Verify GET /v1/jobs/{id} returns stage, stage_index, stage_total, started_at, and updated_at.
    """
    import uuid
    from datetime import datetime, timezone
    from core.database.database import SessionLocal
    from core.cruds.job_crud import JobCRUD

    job_id = f"test_stage_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)
    job_data = {
        "id": job_id,
        "kind": "preview",
        "voice_id": "client_ref",
        "status": "processing",
        "stage": "Loading voice",
        "stage_index": 2,
        "stage_total": 4,
        "client_key": "test_client_hash",
        "progress_message": "Loading voice for preview...",
        "created_at": now,
        "started_at": now,
        "updated_at": now,
        "expires_at": now,
    }

    with TestClient(app) as test_client:
        db = SessionLocal()
        try:
            JobCRUD.create_job(db, job_data)
        finally:
            db.close()

        response = test_client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"] == job_id
        assert payload["stage"] == "Loading voice"
        assert payload["stage_index"] == 2
        assert payload["stage_total"] == 4
        assert payload["started_at"] is not None
        assert payload["updated_at"] is not None


def test_post_tts_preview_enqueues_preview_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify POST /v1/tts/preview returns job_id, preview_url, and confidence_label.
    """
    from core.controllers.job_controller import job_worker
    monkeypatch.setattr(job_worker, "enqueue_job", lambda j_id, pipeline_fn=None: None)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts/preview",
            json={"text": "Short preview test audio.", "voice_id": "client_ref", "language": "hi"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "job_id" in payload
        assert payload["job_id"] is not None
        assert "confidence_label" in payload


def test_post_custom_tts_preview_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify POST /v1/tts/preview/custom accepts multipart reference audio and queues preview job.
    """
    from core.controllers.job_controller import job_worker
    monkeypatch.setattr(job_worker, "enqueue_job", lambda j_id, pipeline_fn=None: None)

    sr = 22050
    t = np.linspace(0, 4, int(sr * 4))
    tone = 0.3 * np.sin(2 * np.pi * 300 * t)
    wav_path = tmp_path / "custom_preview_ref.wav"
    sf.write(str(wav_path), tone, sr)

    with TestClient(app) as test_client:
        with open(wav_path, "rb") as f:
            response = test_client.post(
                "/v1/tts/preview/custom",
                data={
                    "text": "Short custom preview text.",
                    "language": "hi",
                    "speed": "1.0",
                    "consent": "true",
                },
                files={"file": ("custom_preview_ref.wav", f, "audio/wav")},
            )
        assert response.status_code == 200
        payload = response.json()
        assert "job_id" in payload
        assert "preview_url" in payload


def test_post_custom_tts_preview_no_consent_returns_400(tmp_path: Path) -> None:
    """
    Verify POST /v1/tts/preview/custom without consent returns HTTP 400.
    """
    sr = 22050
    tone = 0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 3, int(sr * 3)))
    wav_path = tmp_path / "no_consent_preview.wav"
    sf.write(str(wav_path), tone, sr)

    with TestClient(app) as test_client:
        with open(wav_path, "rb") as f:
            response = test_client.post(
                "/v1/tts/preview/custom",
                data={
                    "text": "Short preview.",
                    "language": "hi",
                    "speed": "1.0",
                    "consent": "false",
                },
                files={"file": ("no_consent_preview.wav", f, "audio/wav")},
            )
        assert response.status_code == 400
        msg = response.json().get("error", {}).get("message") or response.json().get("detail", "")
        assert "consent" in msg.lower()


def test_post_tts_preview_unknown_voice_returns_404() -> None:
    """
    Verify POST /v1/tts/preview for an unknown voice returns 404 rather than silently falling back.
    """
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/tts/preview",
            json={"text": "Short preview.", "voice_id": "nonexistent_voice_slug", "language": "hi"},
        )
        assert response.status_code == 404


def test_reference_check_safety_no_file_or_voice_returns_400() -> None:
    """
    Verify POST /v1/reference/check without audio file or voice_id returns HTTP 400.
    """
    with TestClient(app) as test_client:
        response = test_client.post("/v1/reference/check")
        assert response.status_code == 400


def test_post_custom_tts_job_creation_languages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify POST /v1/tts/custom accepts reference audio and creates jobs for en, hi, and mr.
    """
    from core.controllers.job_controller import job_worker
    monkeypatch.setattr(job_worker, "enqueue_job", lambda j_id, pipeline_fn=None: None)

    sr = 22050
    tone = 0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 3, int(sr * 3)))
    ref_path = tmp_path / "custom_test_ref.wav"
    sf.write(str(ref_path), tone, sr)

    with TestClient(app) as test_client:
        for idx, lang in enumerate(["en", "hi", "mr"]):
            with open(ref_path, "rb") as f:
                response = test_client.post(
                    "/v1/tts/custom",
                    data={
                        "text": f"This is a test script in {lang}.",
                        "language": lang,
                        "speed": "1.0",
                        "consent": "true",
                    },
                    files={"file": (f"ref_{lang}.wav", f, "audio/wav")},
                    headers={"X-Forwarded-For": f"10.10.1.{idx + 1}"},
                )
            assert response.status_code == 202, f"Failed for lang={lang}: {response.text}"
            payload = response.json()
            assert payload["kind"] == "tts_custom"
            assert "job_id" in payload


def test_post_custom_tts_distinct_references_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify two requests with different reference recordings are strictly isolated in their respective job directories.
    """
    from core.controllers.job_controller import job_worker
    monkeypatch.setattr(job_worker, "enqueue_job", lambda j_id, pipeline_fn=None: None)

    sr = 22050
    tone_a = 0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 3, int(sr * 3)))
    tone_b = 0.4 * np.sin(2 * np.pi * 600 * np.linspace(0, 3, int(sr * 3)))
    ref_a = tmp_path / "ref_a.wav"
    ref_b = tmp_path / "ref_b.wav"
    sf.write(str(ref_a), tone_a, sr)
    sf.write(str(ref_b), tone_b, sr)

    from core.config import get_settings
    settings = get_settings()

    with TestClient(app) as test_client:
        with open(ref_a, "rb") as f:
            res_a = test_client.post(
                "/v1/tts/custom",
                data={"text": "Script A", "language": "en", "speed": "1.0", "consent": "true"},
                files={"file": ("ref_a.wav", f, "audio/wav")},
                headers={"X-Forwarded-For": "10.20.1.1"},
            )
        with open(ref_b, "rb") as f:
            res_b = test_client.post(
                "/v1/tts/custom",
                data={"text": "Script B", "language": "en", "speed": "1.0", "consent": "true"},
                files={"file": ("ref_b.wav", f, "audio/wav")},
                headers={"X-Forwarded-For": "10.20.1.2"},
            )

        assert res_a.status_code == 202
        assert res_b.status_code == 202
        job_id_a = res_a.json()["job_id"]
        job_id_b = res_b.json()["job_id"]

        assert job_id_a != job_id_b

        dir_a = settings.tmp_path / job_id_a
        dir_b = settings.tmp_path / job_id_b
        file_a = dir_a / "reference.wav"
        file_b = dir_b / "reference.wav"

        assert file_a.exists()
        assert file_b.exists()

        import hashlib
        hash_a = hashlib.sha256(file_a.read_bytes()).hexdigest()
        hash_b = hashlib.sha256(file_b.read_bytes()).hexdigest()
        assert hash_a != hash_b, "Reference audio hashes must be distinct for distinct inputs"


def test_custom_tts_unsupported_language_returns_400(tmp_path: Path) -> None:
    """
    Verify POST /v1/tts/custom with unsupported language returns HTTP 400.
    """
    sr = 22050
    tone = 0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 3, int(sr * 3)))
    ref_path = tmp_path / "custom_unsupported_ref.wav"
    sf.write(str(ref_path), tone, sr)

    with TestClient(app) as test_client:
        with open(ref_path, "rb") as f:
            response = test_client.post(
                "/v1/tts/custom",
                data={
                    "text": "Unsupported language test.",
                    "language": "de",
                    "speed": "1.0",
                    "consent": "true",
                },
                files={"file": ("ref.wav", f, "audio/wav")},
                headers={"X-Forwarded-For": "10.30.1.1"},
            )
        assert response.status_code == 400









# ---------------------------------------------------------------------------
# Security regression tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        "/v1/audio/preview/..%5C..%5C..%5Cvoice_studio",
        "/v1/audio/preview/..",
        "/v1/jobs/..%5Cvoices",
        "/v1/audio/..%5C..%5Cvoices/output",
    ],
)
def test_malformed_job_ids_rejected(path: str) -> None:
    """
    Backslash / dot-dot IDs (path traversal on Windows) never reach the filesystem.
    """
    response = client.get(path)
    assert response.status_code == 404


def test_spoofed_forwarded_for_does_not_change_client_key() -> None:
    """
    X-Forwarded-For is ignored unless TRUST_PROXY_HEADERS is enabled.
    """
    from starlette.requests import Request
    from core.apis.middleware import get_hashed_client_key

    def make(headers):
        scope = {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": ("10.0.0.5", 1234),
        }
        return Request(scope)

    assert get_hashed_client_key(make({})) == get_hashed_client_key(make({"X-Forwarded-For": "1.2.3.4"}))


@pytest.mark.rate_limit
def test_preview_endpoints_are_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GPU-heavy preview endpoints are covered by the rate limiter.
    """
    from core.apis import middleware

    monkeypatch.setattr(middleware.settings, "RATE_LIMIT_PER_MINUTE", 2)
    local = TestClient(app, client=("203.0.113.77", 5000))
    codes = [
        local.post("/v1/tts/preview", json={"text": "hi", "voice_id": "does_not_exist"}).status_code
        for _ in range(3)
    ]
    assert codes[-1] == 429


def test_custom_preview_requires_explicit_consent() -> None:
    """
    Omitting `consent` must fail validation instead of defaulting to True.
    """
    tone = io.BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    response = client.post(
        "/v1/tts/preview/custom",
        data={"text": "hello", "language": "en"},
        files={"file": ("ref.wav", tone, "audio/wav")},
    )
    assert response.status_code == 422


def test_non_audio_upload_rejected_without_leaking_paths() -> None:
    """
    A playlist disguised as .wav is rejected by content sniffing before ffmpeg sees it.
    """
    playlist = io.BytesIO(b"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\nfile:///etc/passwd\n")
    response = client.post(
        "/v1/tts/custom",
        data={"text": "hello world", "language": "en", "consent": "true"},
        files={"file": ("ref.wav", playlist, "audio/wav")},
    )
    assert response.status_code == 400
    body = response.text
    assert "storage" not in body and ":\\" not in body
