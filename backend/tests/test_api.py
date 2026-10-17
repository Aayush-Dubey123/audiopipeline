"""
API and integration test suite for AI Voice Studio core endpoints.
"""

import pytest
from pathlib import Path
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
        assert "my_voice" in voice_ids
        # Experimental voice must not be listed by default
        assert "nttsai" not in voice_ids

        for voice in payload["voices"]:
            assert "asset_file" not in voice
            assert "reference/my_voice.wav" not in str(voice)
            assert "checkpoints" not in str(voice)
            assert "voices/" not in str(voice)


def test_get_voice_by_id_success() -> None:
    """
    Verify GET /v1/voices/{id} returns profile details for an available voice.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/voices/my_voice")
        assert response.status_code == 200
        payload = response.json()

        assert payload["id"] == "my_voice"
        assert payload["name"] == "My Voice"
        assert payload["model_type"] == "openvoice_reference"
        assert payload["has_preview"] is True
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
    Verify GET /v1/voices/{id}/preview returns streaming WAV audio for my_voice.
    """
    with TestClient(app) as test_client:
        response = test_client.get("/v1/voices/my_voice/preview")
        assert response.status_code == 200
        assert "audio/wav" in response.headers.get("Content-Type", "")
        assert len(response.content) > 1000


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
                "voice_id": "my_voice",
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
                "voice_id": "my_voice",
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
                "voice_id": "my_voice",
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
                "voice_id": "my_voice",
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



