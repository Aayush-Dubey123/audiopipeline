"""
Controller managing inference job lifecycle, submissions, audio artifact streaming, and cleanup.
"""

import os
import re
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from core import logger
from core.config import get_settings
from core.cruds.job_crud import JobCRUD
from core.cruds.voice_crud import VoiceCRUD
from core.services.jobs.cleanup_service import CleanupService
from core.services.jobs.job_worker import job_worker, QueueFullError
from core.services.audio.audio_service import AudioService
from core.services.audio.validation_service import (
    ValidationService,
    AudioValidationError,
    AudioFileSizeExceededError,
    AudioDurationExceededError,
    InvalidAudioExtensionError,
    ReferenceQualityError,
)
from core.services.llm.llm_service import LLMService
from core.apis.schemas.requests.job_requests import (
    TTSJobRequest,
    TextCleanupRequest,
    TTSPreviewRequest,
)
from core.apis.schemas.responses.job_responses import (
    JobCreatedResponse,
    JobStatusResponse,
    JobDeleteResponse,
    TextCleanupResponse,
    ReferenceCheckResponse,
    TTSPreviewResponse,
)

logging = logger(__name__)
settings = get_settings()

MAX_CONCURRENT_JOBS_PER_CLIENT = 3

# Job IDs are secrets.token_urlsafe(16) or uuid4().hex. Anything else (dots, slashes,
# backslashes) is rejected before it can touch the filesystem.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def validate_job_id(job_id: str) -> str:
    """
    Reject job/preview identifiers that are not in the server-issued token format.

    Raises:
        HTTPException 404: If the identifier is malformed (prevents path traversal).
    """
    if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job_id


class JobController:
    """
    Business logic and orchestration for speech synthesis, voice conversion, and job operations.
    """

    @staticmethod
    def _check_client_active_job_quota(db: Session, client_key: str) -> None:
        """
        Verify that a client does not exceed the allowed concurrent active job limit.

        Args:
            db (Session): Active database session.
            client_key (str): Hashed client identifier token.

        Raises:
            HTTPException 429: If the client already has too many pending/processing jobs.
        """
        active_count = JobCRUD.count_active_jobs_for_client(db=db, client_key=client_key)
        if active_count >= MAX_CONCURRENT_JOBS_PER_CLIENT:
            logging.warning(f"Client '{client_key}' exceeded active job limit ({active_count}/{MAX_CONCURRENT_JOBS_PER_CLIENT})")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"You already have {active_count} active job(s) in progress. Please wait for completion before submitting more.",
            )

    @staticmethod
    def _check_client_daily_quota(db: Session, client_key: str) -> None:
        """
        Verify that a client has not exceeded their daily generated audio minutes cap.

        Args:
            db (Session): Active database session.
            client_key (str): Hashed client identifier token.

        Raises:
            HTTPException 429: If daily quota is exceeded.
        """
        daily_minutes = JobCRUD.get_daily_generated_minutes(db=db, client_key=client_key)
        if daily_minutes >= settings.CLIENT_DAILY_MINUTES:
            logging.warning(
                f"Client '{client_key}' exceeded daily quota ({daily_minutes:.1f}/{settings.CLIENT_DAILY_MINUTES} min)"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily audio generation quota reached ({daily_minutes:.1f}/{settings.CLIENT_DAILY_MINUTES} minutes used in last 24h). Please try again tomorrow.",
            )

    @classmethod
    def create_tts_job(
        cls,
        db: Session,
        client_key: str,
        request: TTSJobRequest,
    ) -> JobCreatedResponse:
        """
        Validate, register, and queue a standard text-to-speech synthesis job.

        Args:
            db (Session): Active database session.
            client_key (str): Hashed client token.
            request (TTSJobRequest): Validated input parameters.

        Returns:
            JobCreatedResponse: Job token and queued status envelope.

        Raises:
            HTTPException 400: If voice does not exist, is disabled, or does not support requested language.
            HTTPException 429: If queue is full or per-client concurrent job limit reached.
        """
        logging.info(f"Executing JobController.create_tts_job for voice '{request.voice_id}' (lang={request.language})")
        cls._check_client_active_job_quota(db=db, client_key=client_key)
        cls._check_client_daily_quota(db=db, client_key=client_key)

        voice = VoiceCRUD.get_voice_by_id(db=db, voice_id=request.voice_id)
        if not voice:
            logging.warning(f"TTS submission rejected: voice '{request.voice_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{request.voice_id}' was not found in the catalog.",
            )

        if voice.status == "disabled":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice '{voice.name}' is currently unavailable.",
            )

        if voice.status == "experimental" and not settings.EXPOSE_EXPERIMENTAL_VOICES:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{request.voice_id}' was not found.",
            )

        if voice.model_type not in ("chatterbox_reference", "openvoice_reference"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice '{voice.name}' has invalid model type '{voice.model_type}'.",
            )

        lang = request.language.lower().strip()
        supported_langs = [l.lower() for l in voice.languages]
        if lang not in supported_langs:
            logging.warning(f"Language '{lang}' rejected for voice '{voice.id}'. Supported: {supported_langs}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Language '{request.language}' is not supported by voice '{voice.name}'. Supported: {', '.join(voice.languages)}",
            )

        job_id = secrets.token_urlsafe(16)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)

        job_data = {
            "id": job_id,
            "kind": "tts",
            "voice_id": voice.id,
            "status": "queued",
            "client_key": client_key,
            "progress_message": "Queued for synthesis.",
            "payload": {
                "text": request.text.strip(),
                "language": lang,
                "speed": request.speed,
            },
            "expires_at": expires_at,
        }

        JobCRUD.create_job(db=db, job_data=job_data)

        try:
            job_worker.enqueue_job(job_id)
        except QueueFullError as q_err:
            JobCRUD.update_job_status(db=db, job_id=job_id, status="failed", error_code="QUEUE_FULL", error_message=str(q_err))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(q_err),
            )

        return JobCreatedResponse(
            job_id=job_id,
            kind="tts",
            status="queued",
            message="Text-to-speech job queued successfully.",
            ai_generated=True,
        )

    @classmethod
    async def create_custom_tts_job(
        cls,
        db: Session,
        client_key: str,
        text: str,
        language: str,
        speed: float,
        consent: bool,
        file: UploadFile,
    ) -> JobCreatedResponse:
        """
        Validate reference voice audio and queue a custom voice-cloned text-to-speech job.

        Args:
            db (Session): Active database session.
            client_key (str): Hashed client token.
            text (str): Script text to synthesize.
            language (str): Target language code ('en', 'hi', or 'mr').
            speed (float): Playback rate multiplier.
            consent (bool): Explicit user confirmation of voice ownership or permission.
            file (UploadFile): FastApi multipart audio reference upload.

        Returns:
            JobCreatedResponse: Queued job envelope.

        Raises:
            HTTPException 400: If consent is missing, audio is corrupt, or quality thresholds fail.
            HTTPException 429: If queue is full or per-client quota is reached.
        """
        logging.info("Executing JobController.create_custom_tts_job")
        if not consent:
            logging.warning("Custom TTS rejected: user consent was not given")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Explicit consent is required. You must verify you have the right or authorization to clone this voice.",
            )

        cls._check_client_active_job_quota(db=db, client_key=client_key)
        cls._check_client_daily_quota(db=db, client_key=client_key)

        clean_text = text.strip()
        if not clean_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text script cannot be empty.",
            )

        if len(clean_text) > settings.MAX_TEXT_CHARS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text exceeds maximum character limit of {settings.MAX_TEXT_CHARS}.",
            )

        lang = language.lower().strip()
        if lang not in ("en", "hi", "mr"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language '{language}'. Supported: English ('en'), Hindi ('hi'), Marathi ('mr').",
            )

        job_id = secrets.token_urlsafe(16)
        job_dir = (settings.tmp_path / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        temp_upload_path = None
        try:
            # 1. Stream upload to disk with hard byte cutoff
            temp_upload_path = await ValidationService.stream_upload_to_disk(
                upload_file=file,
                destination_dir=job_dir,
            )

            # 2. Transcode once into standard 24kHz mono WAV (no extra temp copies)
            dest_wav = job_dir / "reference.wav"
            AudioService.to_wav(temp_upload_path, dest_wav, sample_rate=24000, mono=True)

            # 3. Inspect acoustic reference quality on the transcoded WAV
            quality = ValidationService.inspect_reference_audio_quality(
                file_path=dest_wav,
                min_duration=3.0,
                max_duration=60.0,
            )
            logging.info(f"Custom voice reference quality: score={quality.quality_score}, warnings={quality.warnings}")

        except (AudioFileSizeExceededError, InvalidAudioExtensionError, AudioDurationExceededError, ReferenceQualityError) as val_err:
            CleanupService.cleanup_job_directory(job_id)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
        except Exception as exc:
            CleanupService.cleanup_job_directory(job_id)
            logging.error(f"Error processing custom reference upload: {exc}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process reference audio. Upload a clear speech recording (wav/mp3/m4a/ogg/webm/flac).")
        finally:
            if temp_upload_path and temp_upload_path.exists() and temp_upload_path.name != "reference.wav":
                try:
                    temp_upload_path.unlink()
                except OSError:
                    pass

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)
        job_data = {
            "id": job_id,
            "kind": "tts_custom",
            "voice_id": None,
            "status": "queued",
            "client_key": client_key,
            "progress_message": "Custom reference voice queued for synthesis.",
            "payload": {
                "text": clean_text,
                "language": lang,
                "speed": float(speed),
                "consent": True,
            },
            "expires_at": expires_at,
        }

        JobCRUD.create_job(db=db, job_data=job_data)

        try:
            job_worker.enqueue_job(job_id)
        except QueueFullError as q_err:
            CleanupService.cleanup_job_directory(job_id)
            JobCRUD.update_job_status(db=db, job_id=job_id, status="failed", error_code="QUEUE_FULL", error_message=str(q_err))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(q_err),
            )

        return JobCreatedResponse(
            job_id=job_id,
            kind="tts_custom",
            status="queued",
            message="Custom voice cloning job queued successfully.",
            ai_generated=True,
        )

    @classmethod
    async def create_vc_job(
        cls,
        db: Session,
        client_key: str,
        voice_id: str,
        pitch_shift: int,
        file: UploadFile,
    ) -> JobCreatedResponse:
        """
        Validate input audio and queue an RVC voice conversion job.

        Args:
            db (Session): Active database session.
            client_key (str): Hashed client token.
            voice_id (str): Target RVC voice profile slug.
            pitch_shift (int): Pitch shift in semitones (-12 to +12).
            file (UploadFile): FastApi multipart audio upload.

        Returns:
            JobCreatedResponse: Queued job outcome envelope.

        Raises:
            HTTPException 400: If voice model or audio upload parameters are invalid.
            HTTPException 404: If voice cannot be found.
            HTTPException 429: If queue is full or per-client quota is reached.
        """
        logging.info(f"Executing JobController.create_vc_job for voice '{voice_id}' (pitch={pitch_shift})")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Voice-to-voice conversion is currently disabled (coming soon).",
        )
        if not voice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' was not found in catalog.",
            )

        if voice.status == "disabled":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice '{voice.name}' is currently unavailable.",
            )

        if voice.status == "experimental" and not settings.EXPOSE_EXPERIMENTAL_VOICES:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' was not found.",
            )

        if voice.model_type != "rvc_model":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice '{voice.name}' is not an RVC voice conversion model.",
            )

        if pitch_shift < -12 or pitch_shift > 12:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="pitch_shift must be between -12 and +12 semitones.",
            )

        job_id = secrets.token_urlsafe(16)
        job_dir = (settings.tmp_path / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        temp_upload_path = None
        try:
            # 1. Stream upload to disk
            temp_upload_path = await ValidationService.stream_upload_to_disk(
                upload_file=file,
                destination_dir=job_dir,
            )

            # 2. Probe content and enforce duration limits
            ValidationService.validate_audio_duration(
                file_path=temp_upload_path,
                max_seconds=settings.MAX_AUDIO_SECONDS,
                min_seconds=0.5,
            )

            # 3. Transcode to source.wav
            dest_wav = job_dir / "source.wav"
            AudioService.to_wav(temp_upload_path, dest_wav)

        except (AudioFileSizeExceededError, InvalidAudioExtensionError, AudioDurationExceededError) as val_err:
            CleanupService.cleanup_job_directory(job_id)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
        except Exception as exc:
            CleanupService.cleanup_job_directory(job_id)
            logging.error(f"Error processing VC audio upload: {exc}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process audio.")
        finally:
            if temp_upload_path and temp_upload_path.exists() and temp_upload_path.name != "source.wav":
                try:
                    temp_upload_path.unlink()
                except OSError:
                    pass

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)
        job_data = {
            "id": job_id,
            "kind": "vc",
            "voice_id": voice.id,
            "status": "queued",
            "client_key": client_key,
            "progress_message": "Queued for voice conversion.",
            "payload": {
                "pitch_shift": pitch_shift,
            },
            "expires_at": expires_at,
        }

        JobCRUD.create_job(db=db, job_data=job_data)

        try:
            job_worker.enqueue_job(job_id)
        except QueueFullError as q_err:
            JobCRUD.update_job_status(db=db, job_id=job_id, status="failed", error_code="QUEUE_FULL", error_message=str(q_err))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(q_err),
            )

        return JobCreatedResponse(
            job_id=job_id,
            kind="vc",
            status="queued",
            message="Voice conversion job queued successfully.",
            ai_generated=True,
        )

    @staticmethod
    def cleanup_text(request: TextCleanupRequest) -> TextCleanupResponse:
        """
        Normalize script text using optional LLM or deterministic rule-based cleaners.

        Args:
            request (TextCleanupRequest): Raw text payload and language specification.

        Returns:
            TextCleanupResponse: Cleaned text and engine attribution.
        """
        logging.info("Executing JobController.cleanup_text")
        result = LLMService.clean_text(text=request.text, language=request.language)
        return TextCleanupResponse(
            original=result.original,
            cleaned=result.cleaned,
            provider_used=result.provider_used,
            service_available=result.service_available,
        )

    @staticmethod
    def get_job_status(db: Session, job_id: str) -> JobStatusResponse:
        """
        Query current status, execution progress, and outcome links for an inference job.

        Args:
            db (Session): Active database session.
            job_id (str): Unique job identifier token.

        Returns:
            JobStatusResponse: Structured status payload with client-safe resource URLs.

        Raises:
            HTTPException 404: If job identifier is not registered.
        """
        validate_job_id(job_id)
        logging.info(f"Executing JobController.get_job_status for job '{job_id}'")
        job = JobCRUD.get_job_by_id(db=db, job_id=job_id)
        if not job:
            logging.warning(f"Job '{job_id}' not found in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found",
            )

        if job.kind in ("preview", "tts_preview"):
            output_url = f"/v1/audio/preview/{job_id}" if job.status == "completed" else None
        else:
            output_url = f"/v1/audio/{job_id}/output" if job.status == "completed" else None
        src_dir = settings.tmp_path / job_id
        has_source = (src_dir / "source.wav").is_file() or (src_dir / "reference.wav").is_file()
        source_url = f"/v1/audio/{job_id}/source" if job.kind in ("vc", "tts_custom") and has_source else None

        output_files = None
        confidence_label = None
        if job.status == "completed":
            job_dir = (settings.tmp_path / job_id).resolve()
            if job.kind in ("preview", "tts_preview"):
                prev_file = job_dir / "preview.wav"
                if not prev_file.is_file():
                    prev_file = job_dir / "output.wav"
                output_files = {
                    "wav": {
                        "url": f"/v1/audio/preview/{job_id}",
                        "size_bytes": prev_file.stat().st_size if prev_file.is_file() else 0,
                        "hint": "Generated preview audio sample",
                    }
                }
            else:
                wav_file = job_dir / "output.wav"
                mp3_file = job_dir / "output.mp3"
                output_files = {
                    "wav": {
                        "url": f"/v1/audio/{job_id}/output?format=wav",
                        "size_bytes": wav_file.stat().st_size if wav_file.is_file() else 0,
                        "hint": "Master uncompressed audio for video editing",
                    },
                    "mp3": {
                        "url": f"/v1/audio/{job_id}/output?format=mp3",
                        "size_bytes": mp3_file.stat().st_size if mp3_file.is_file() else 0,
                        "hint": "192 kbps MP3 ready for X / Instagram / YouTube uploads",
                    },
                }
            if job.output_metadata and isinstance(job.output_metadata, dict):
                confidence_label = job.output_metadata.get("confidence_label")

        stage = job.stage
        stage_index = job.stage_index
        stage_total = job.stage_total or 4
        if stage is None:
            if job.status == "queued":
                stage = "Queued"
                stage_index = 1
            elif job.status == "processing":
                stage = "Generating speech"
                stage_index = 3
            elif job.status == "completed":
                stage = "Completed"
                stage_index = 4

        return JobStatusResponse(
            job_id=job.id,
            kind=job.kind,
            status=job.status,
            voice_id=job.voice_id,
            progress_message=job.progress_message,
            error_code=job.error_code,
            error_message=job.error_message,
            stage=stage,
            stage_index=stage_index,
            stage_total=stage_total,
            chunk_done=job.chunk_done,
            chunk_total=job.chunk_total,
            eta_seconds=job.eta_seconds,
            created_at=job.created_at,
            started_at=job.started_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            expires_at=job.expires_at,
            output_url=output_url,
            source_url=source_url,
            output_files=output_files,
            confidence_label=confidence_label,
        )

    @staticmethod
    def get_job_output_file(db: Session, job_id: str, audio_format: str = "wav") -> Tuple[Path, str, str]:
        """
        Verify and locate the synthesized audio output file for a completed job.

        Args:
            db (Session): Active database session.
            job_id (str): Unique job identifier.
            audio_format (str): Requested audio format ('wav', 'mp3', or 'mp3_small').

        Returns:
            Tuple[Path, str, str]: File path, MIME media type, and download filename.

        Raises:
            HTTPException 400: If job is not in completed state or format is unsupported.
            HTTPException 404: If job or audio file is missing.
        """
        validate_job_id(job_id)
        logging.info(f"Executing JobController.get_job_output_file for job '{job_id}' (format={audio_format})")
        fmt = audio_format.lower().strip()
        if fmt not in ("wav", "mp3", "mp3_small"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Supported audio formats are 'wav', 'mp3', and 'mp3_small'.",
            )

        job = JobCRUD.get_job_by_id(db=db, job_id=job_id)
        if not job:
            logging.warning(f"Job '{job_id}' not found for audio output retrieval")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found",
            )

        if job.status != "completed":
            logging.warning(f"Audio retrieval rejected: job '{job_id}' status is '{job.status}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job output is not ready (status: {job.status})",
            )

        job_dir = (settings.tmp_path / job_id).resolve()
        try:
            job_dir.relative_to(settings.tmp_path.resolve())
        except ValueError:
            logging.error(f"Security error: path traversal attempt for job '{job_id}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio artifact not found",
            )

        wav_path = job_dir / "output.wav"
        if not wav_path.is_file() and (job_dir / "preview.wav").is_file():
            wav_path = job_dir / "preview.wav"
        if not wav_path.is_file():
            logging.warning(f"Audio file missing on disk for completed job '{job_id}': {wav_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio artifact not found on server",
            )

        if fmt == "wav":
            return wav_path, "audio/wav", f"{job_id}_output.wav"

        if fmt == "mp3":
            mp3_path = job_dir / "output.mp3"
            if not mp3_path.is_file():
                try:
                    AudioService.export_mp3(wav_path, mp3_path, bitrate="192k", mono=False)
                except Exception as mp3_err:
                    logging.error(f"Failed to export MP3 for job '{job_id}': {mp3_err}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Audio format conversion failed",
                    )
            return mp3_path, "audio/mpeg", f"{job_id}_output.mp3"

        # Legacy 'mp3_small' requests get the single MP3 so a job never holds
        # more than two audio files (output.wav + output.mp3).
        return JobController.get_job_output_file(db=db, job_id=job_id, audio_format="mp3")

    @staticmethod
    def get_job_source_file(db: Session, job_id: str) -> Tuple[Path, str, str]:
        """
        Verify and locate the original uploaded source audio for conversion jobs.

        Args:
            db (Session): Active database session.
            job_id (str): Unique job identifier.

        Returns:
            Tuple[Path, str, str]: File path, MIME media type, and download filename.

        Raises:
            HTTPException 404: If job or source audio file is missing.
        """
        validate_job_id(job_id)
        logging.info(f"Executing JobController.get_job_source_file for job '{job_id}'")
        job = JobCRUD.get_job_by_id(db=db, job_id=job_id)
        if not job:
            logging.warning(f"Job '{job_id}' not found for source audio retrieval")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found",
            )

        job_dir = (settings.tmp_path / job_id).resolve()
        try:
            job_dir.relative_to(settings.tmp_path.resolve())
        except ValueError:
            logging.error(f"Security error: path traversal attempt for job '{job_id}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source audio not found",
            )

        # Look for source.wav or reference.wav
        source_path = job_dir / "source.wav"
        if not source_path.is_file():
            source_path = job_dir / "reference.wav"

        if not source_path.is_file():
            logging.warning(f"Source audio missing on disk for job '{job_id}': {source_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source audio not found",
            )

        return source_path, "audio/wav", f"{job_id}_source.wav"

    @staticmethod
    def delete_job(db: Session, job_id: str) -> JobDeleteResponse:
        """
        Cancel a pending job, purge its temporary disk artifacts, and delete its database record.

        Args:
            db (Session): Active database session.
            job_id (str): Unique job identifier token.

        Returns:
            JobDeleteResponse: Confirmation message of completed deletion.

        Raises:
            HTTPException 404: If job record does not exist.
        """
        validate_job_id(job_id)
        logging.info(f"Executing JobController.delete_job for job '{job_id}'")
        job = JobCRUD.get_job_by_id(db=db, job_id=job_id)
        if not job:
            logging.warning(f"Cannot delete non-existent job '{job_id}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found",
            )

        CleanupService.cleanup_job_directory(job_id)
        JobCRUD.delete_job(db=db, job_id=job_id)

        return JobDeleteResponse(
            job_id=job_id,
            status="deleted",
            message=f"Job '{job_id}' and all associated audio artifacts were deleted.",
        )

    @classmethod
    def resume_job(
        cls,
        db: Session,
        client_key: str,
        job_id: str,
    ) -> JobCreatedResponse:
        """
        Resume a failed or interrupted long-form inference job.

        Args:
            db (Session): Active database session.
            client_key (str): Client (hashed IP).
            job_id (str): Unique job identifier token.

        Returns:
            JobCreatedResponse: Queued status confirmation.

        Raises:
            HTTPException 400: If job is already completed or invalid kind.
            HTTPException 404: If job does not exist.
            HTTPException 429: If queue is full or quotas are exceeded.
        """
        validate_job_id(job_id)
        logging.info(f"Executing JobController.resume_job for job '{job_id}'")
        job = JobCRUD.get_job_by_id(db=db, job_id=job_id)
        if not job:
            logging.warning(f"Resume rejected: job '{job_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' was not found.",
            )

        if job.status == "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is already completed.",
            )
        if job.status == "processing":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is currently processing.",
            )

        if job.kind not in ("tts", "tts_custom"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job kind '{job.kind}' does not support resuming.",
            )

        cls._check_client_active_job_quota(db=db, client_key=client_key)
        cls._check_client_daily_quota(db=db, client_key=client_key)

        job_dir = (settings.tmp_path / job_id).resolve()
        chunks_dir = job_dir / "chunks"

        if chunks_dir.exists():
            for part_file in chunks_dir.glob("*.part"):
                part_file.unlink(missing_ok=True)
            finished_chunks = [
                p for p in chunks_dir.glob("chunk_*.wav") if p.stat().st_size > 44
            ]
            finished_count = len(finished_chunks)
        else:
            finished_count = 0

        JobCRUD.update_job_status(
            db=db,
            job_id=job_id,
            status="queued",
            progress_message=f"Resuming job from chunk {finished_count}...",
            error_code=None,
            error_message=None,
        )

        try:
            job_worker.enqueue_job(job_id)
        except QueueFullError as q_err:
            JobCRUD.update_job_status(
                db=db,
                job_id=job_id,
                status="failed",
                error_code="QUEUE_FULL",
                error_message=str(q_err),
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(q_err),
            )

        return JobCreatedResponse(
            job_id=job_id,
            kind=job.kind,
            status="queued",
            message=f"Job resumed successfully from chunk {finished_count}.",
            ai_generated=True,
        )

    @classmethod
    async def check_reference_safety(
        cls,
        db: Session,
        client_key: str,
        file: Optional[UploadFile] = None,
        voice_id: Optional[str] = None,
    ) -> ReferenceCheckResponse:
        """
        Evaluate reference audio acoustic safety (length, loudness, noise, clipping, reverb, speaker, music).
        """
        logging.info("Executing JobController.check_reference_safety")
        if file is not None:
            temp_dir = settings.tmp_path / "reference_checks"
            temp_dir.mkdir(parents=True, exist_ok=True)
            saved_path = await ValidationService.stream_upload_to_disk(
                upload_file=file,
                destination_dir=temp_dir,
                max_bytes=settings.MAX_UPLOAD_MB * 1024 * 1024,
            )
            try:
                res = ValidationService.check_reference_safety(saved_path)
                return ReferenceCheckResponse(**res)
            finally:
                saved_path.unlink(missing_ok=True)
        elif voice_id:
            voice = VoiceCRUD.get_voice_by_id(db, voice_id)
            if not voice:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Voice '{voice_id}' not found")
            ref_path = (settings.voices_path / voice.asset_file).resolve()
            if not ref_path.exists():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice audio asset not found on disk")
            res = ValidationService.check_reference_safety(ref_path)
            return ReferenceCheckResponse(**res)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either an audio file or voice_id must be provided to evaluate acoustic safety.",
            )

    @classmethod
    def generate_tts_preview(
        cls,
        db: Session,
        client_key: str,
        request: TTSPreviewRequest,
    ) -> TTSPreviewResponse:
        """
        Generate rapid preview of up to 200 characters using the catalog voice pipeline.

        Supports both asynchronous job queuing with stage tracking and direct synchronous generation.

        Args:
            db (Session): Active database session.
            client_key (str): Hashed client key for quota enforcement.
            request (TTSPreviewRequest): Preview parameters including text, voice, and speed.

        Returns:
            TTSPreviewResponse: Structured preview confirmation with job_id and preview_url.

        Raises:
            HTTPException 400: If text or voice_id is empty.
            HTTPException 404: If voice or reference audio cannot be found.
            HTTPException 500: If synchronous synthesis fails.
        """
        logging.info("Executing JobController.generate_tts_preview")
        import hashlib
        import uuid
        from datetime import datetime, timezone, timedelta
        from core.services.jobs.job_worker import job_worker

        from core.services.llm.llm_service import LLMService
        preview_text = LLMService.rule_based_cleanup(request.text.strip()[:200], language=request.language or "hi")
        if not preview_text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text cannot be empty.")

        if not request.voice_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A voice_id is required for saved voice preview. For custom reference audio, use the custom preview endpoint.",
            )

        voice = VoiceCRUD.get_voice_by_id(db, request.voice_id)
        if not voice:
            logging.warning(f"Preview submission rejected: voice '{request.voice_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{request.voice_id}' was not found in the catalog.",
            )

        ref_path = (settings.voices_path / voice.asset_file).resolve()
        if not ref_path.exists():
            logging.error(f"Voice '{voice.id}' preview audio asset missing on disk: {ref_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice asset for '{voice.name}' not found on disk.",
            )

        ref_hash = hashlib.sha256(ref_path.read_bytes()).hexdigest()[:16]
        logging.info(f"Generating preview for voice '{voice.id}': ref_hash='{ref_hash}'")

        job_id = uuid.uuid4().hex
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.JOB_TTL_MINUTES)
        now = datetime.now(timezone.utc)

        if not request.sync:
            # Enqueue asynchronous preview job
            job_data = {
                "id": job_id,
                "kind": "preview",
                "status": "queued",
                "stage": "Queued",
                "stage_index": 1,
                "stage_total": 4,
                "voice_id": voice.id,
                "client_key": client_key,
                "progress_message": "Preview job queued.",
                "payload": {
                    "text": preview_text,
                    "voice_id": voice.id,
                    "language": request.language or "hi",
                    "speed": request.speed,
                    "ref_hash": ref_hash,
                },
                "created_at": now,
                "updated_at": now,
                "expires_at": expires_at,
            }
            cls._check_client_active_job_quota(db=db, client_key=client_key)
            JobCRUD.create_job(db, job_data)
            try:
                job_worker.enqueue_job(job_id)
            except QueueFullError as q_err:
                JobCRUD.update_job_status(db=db, job_id=job_id, status="failed", error_code="QUEUE_FULL", error_message=str(q_err))
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(q_err))

            return TTSPreviewResponse(
                job_id=job_id,
                preview_url=f"/v1/audio/preview/{job_id}",
                duration_seconds=0.0,
                text_preview=preview_text,
                confidence_label="good",
            )

        # Synchronous execution fallback
        preview_dir = settings.tmp_path / job_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        out_wav = preview_dir / "preview.wav"

        cls._check_client_active_job_quota(db=db, client_key=client_key)
        from core.services.pipelines.chatterbox.chatterbox_service import ChatterboxService
        pipeline_service = ChatterboxService()

        try:
            pipeline_service.generate(
                text=preview_text,
                reference_audio_path=str(ref_path),
                output_path=str(out_wav),
                language=request.language or "hi",
                target_sample_rate=settings.OUTPUT_SAMPLE_RATE,
            )
            AudioService.polish(
                out_wav, out_wav, speed=request.speed, sample_rate=settings.OUTPUT_SAMPLE_RATE,
                ai_label=AudioService.AI_LABEL,
            )
            meta = AudioService.probe_audio(out_wav)

            # Persist completed job record for queries
            job_data = {
                "id": job_id,
                "kind": "preview",
                "status": "completed",
                "stage": "Completed",
                "stage_index": 4,
                "stage_total": 4,
                "voice_id": voice.id,
                "client_key": client_key,
                "progress_message": "Preview generation completed.",
                "payload": {
                    "text": preview_text,
                    "voice_id": voice.id,
                    "language": request.language or "hi",
                    "speed": request.speed,
                    "ref_hash": ref_hash,
                },
                "output_metadata": {
                    "duration_seconds": round(meta.duration_seconds, 2),
                    "confidence_label": "good",
                    "ref_hash": ref_hash,
                },
                "created_at": now,
                "started_at": now,
                "updated_at": now,
                "completed_at": now,
                "expires_at": expires_at,
            }
            JobCRUD.create_job(db, job_data)
        except Exception as err:
            logging.error(f"Preview generation failed: {err}")
            CleanupService.cleanup_job_directory(job_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Preview synthesis failed.",
            )

        return TTSPreviewResponse(
            job_id=job_id,
            preview_url=f"/v1/audio/preview/{job_id}",
            duration_seconds=round(meta.duration_seconds, 2),
            text_preview=preview_text,
            confidence_label="good",
        )

    @classmethod
    async def generate_custom_tts_preview(
        cls,
        db: Session,
        client_key: str,
        text: str,
        file: UploadFile,
        language: str = "hi",
        speed: float = 1.0,
        consent: bool = False,
    ) -> TTSPreviewResponse:
        """
        Generate rapid preview of up to 200 characters conditioned on custom reference audio.

        Args:
            db (Session): Active database session.
            client_key (str): Hashed client token.
            text (str): Input text (up to 200 characters).
            file (UploadFile): Reference audio recording or file.
            language (str): Target language ('hi', 'en', 'mr').
            speed (float): Playback speed modifier.
            consent (bool): Explicit voice cloning authorization confirmation.

        Returns:
            TTSPreviewResponse: Structured preview confirmation with job_id and preview_url.

        Raises:
            HTTPException 400: If consent is missing, text is empty, or reference audio is invalid.
        """
        logging.info("Executing JobController.generate_custom_tts_preview")
        if not consent:
            logging.warning("Custom TTS preview rejected: user consent was not given")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Explicit consent is required before generating speech conditioned on custom reference audio.",
            )

        cls._check_client_active_job_quota(db=db, client_key=client_key)
        cls._check_client_daily_quota(db=db, client_key=client_key)

        language = (language or "hi").lower().strip()
        if language not in ("en", "hi", "mr"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported language. Use 'en', 'hi' or 'mr'.")
        if not (0.5 <= float(speed) <= 2.0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="speed must be between 0.5 and 2.0.")

        from core.services.llm.llm_service import LLMService
        clean_text = LLMService.rule_based_cleanup(text.strip()[:200], language=language)
        if not clean_text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text cannot be empty.")

        import hashlib
        import uuid
        from datetime import datetime, timezone, timedelta
        from core.services.jobs.job_worker import job_worker
        from core.services.audio.audio_service import AudioService
        from core.services.jobs.cleanup_service import CleanupService

        job_id = uuid.uuid4().hex
        job_dir = (settings.tmp_path / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        temp_upload_path = None
        try:
            temp_upload_path = await ValidationService.stream_upload_to_disk(
                upload_file=file,
                destination_dir=job_dir,
            )
            dest_wav = job_dir / "reference.wav"
            AudioService.to_wav(temp_upload_path, dest_wav, sample_rate=24000, mono=True)
            ValidationService.inspect_reference_audio_quality(
                file_path=dest_wav,
                min_duration=3.0,
                max_duration=60.0,
            )

            ref_hash = hashlib.sha256(dest_wav.read_bytes()).hexdigest()[:16]
            logging.info(f"Custom preview reference saved: job_id='{job_id}', hash='{ref_hash}'")
        except (AudioFileSizeExceededError, InvalidAudioExtensionError, AudioDurationExceededError, ReferenceQualityError) as val_err:
            CleanupService.cleanup_job_directory(job_id)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
        except Exception as exc:
            CleanupService.cleanup_job_directory(job_id)
            logging.error(f"Error processing custom preview reference: {exc}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process reference audio. Upload a clear speech recording (wav/mp3/m4a/ogg/webm/flac).")
        finally:
            if temp_upload_path and temp_upload_path.exists() and temp_upload_path.name != "reference.wav":
                temp_upload_path.unlink(missing_ok=True)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=settings.JOB_TTL_MINUTES)

        job_data = {
            "id": job_id,
            "kind": "preview",
            "status": "queued",
            "stage": "Queued",
            "stage_index": 1,
            "stage_total": 4,
            "voice_id": None,
            "client_key": client_key,
            "progress_message": "Custom preview job queued.",
            "payload": {
                "text": clean_text,
                "voice_id": None,
                "language": language or "hi",
                "speed": float(speed),
                "is_custom": True,
                "ref_hash": ref_hash,
            },
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
        }
        JobCRUD.create_job(db, job_data)
        try:
            job_worker.enqueue_job(job_id)
        except QueueFullError as q_err:
            CleanupService.cleanup_job_directory(job_id)
            JobCRUD.update_job_status(db=db, job_id=job_id, status="failed", error_code="QUEUE_FULL", error_message=str(q_err))
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(q_err))

        return TTSPreviewResponse(
            job_id=job_id,
            preview_url=f"/v1/audio/preview/{job_id}",
            duration_seconds=0.0,
            text_preview=clean_text,
            confidence_label="good",
        )

    @classmethod
    def get_preview_file(cls, preview_id: str) -> Path:
        """
        Locate generated preview WAV audio file across staging locations.

        Args:
            preview_id (str): Identifier of preview job or session.

        Returns:
            Path: Verified path to the output WAV file.

        Raises:
            HTTPException 404: If preview file is not found.
        """
        validate_job_id(preview_id)
        logging.info(f"Executing JobController.get_preview_file for preview '{preview_id}'")
        candidate_paths = [
            (settings.tmp_path / preview_id / "preview.wav"),
            (settings.tmp_path / preview_id / "output.wav"),
        ]
        for p in candidate_paths:
            if p.is_file():
                return p.resolve()
        logging.warning(f"Preview audio '{preview_id}' not found on disk")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview audio not found or expired.")

