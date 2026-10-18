"""
Controller managing inference job lifecycle, submissions, audio artifact streaming, and cleanup.
"""

import os
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
from core.apis.schemas.requests.job_requests import TTSJobRequest, TextCleanupRequest
from core.apis.schemas.responses.job_responses import (
    JobCreatedResponse,
    JobStatusResponse,
    JobDeleteResponse,
    TextCleanupResponse,
)

logging = logger(__name__)
settings = get_settings()

MAX_CONCURRENT_JOBS_PER_CLIENT = 3


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

        if voice.model_type != "openvoice_reference":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice '{voice.name}' is a voice conversion model and cannot be used for text-to-speech.",
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

            # 2. Inspect acoustic reference quality
            quality = ValidationService.inspect_reference_audio_quality(
                file_path=temp_upload_path,
                min_duration=3.0,
                max_duration=60.0,
            )
            logging.info(f"Custom voice reference quality: score={quality.quality_score}, warnings={quality.warnings}")

            # 3. Transcode reference into standard 24kHz mono WAV
            dest_wav = job_dir / "reference.wav"
            AudioService.to_wav(temp_upload_path, dest_wav)
            AudioService.resample(dest_wav, dest_wav, target_sample_rate=24000)

        except (AudioFileSizeExceededError, InvalidAudioExtensionError, AudioDurationExceededError, ReferenceQualityError) as val_err:
            CleanupService.cleanup_job_directory(job_id)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
        except Exception as exc:
            CleanupService.cleanup_job_directory(job_id)
            logging.error(f"Error processing custom reference upload: {exc}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to process reference audio: {exc}")
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
        cls._check_client_active_job_quota(db=db, client_key=client_key)

        voice = VoiceCRUD.get_voice_by_id(db=db, voice_id=voice_id)
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
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to process audio: {exc}")
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
        logging.info(f"Executing JobController.get_job_status for job '{job_id}'")
        job = JobCRUD.get_job_by_id(db=db, job_id=job_id)
        if not job:
            logging.warning(f"Job '{job_id}' not found in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found",
            )

        output_url = f"/v1/audio/{job_id}/output" if job.status == "completed" else None
        source_url = f"/v1/audio/{job_id}/source" if job.kind in ("vc", "tts_custom") else None

        return JobStatusResponse(
            job_id=job.id,
            kind=job.kind,
            status=job.status,
            voice_id=job.voice_id,
            progress_message=job.progress_message,
            error_code=job.error_code,
            error_message=job.error_message,
            chunk_done=job.chunk_done,
            chunk_total=job.chunk_total,
            eta_seconds=job.eta_seconds,
            created_at=job.created_at,
            completed_at=job.completed_at,
            expires_at=job.expires_at,
            output_url=output_url,
            source_url=source_url,
        )

    @staticmethod
    def get_job_output_file(db: Session, job_id: str, audio_format: str = "wav") -> Tuple[Path, str, str]:
        """
        Verify and locate the synthesized audio output file for a completed job.

        Args:
            db (Session): Active database session.
            job_id (str): Unique job identifier.
            audio_format (str): Requested audio format ('wav' or 'mp3').

        Returns:
            Tuple[Path, str, str]: File path, MIME media type, and download filename.

        Raises:
            HTTPException 400: If job is not in completed state or format is unsupported.
            HTTPException 404: If job or audio file is missing.
        """
        logging.info(f"Executing JobController.get_job_output_file for job '{job_id}' (format={audio_format})")
        fmt = audio_format.lower().strip()
        if fmt not in ("wav", "mp3"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Supported audio formats are 'wav' and 'mp3'.",
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
        if not wav_path.is_file():
            logging.warning(f"Audio file missing on disk for completed job '{job_id}': {wav_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio artifact not found on server",
            )

        if fmt == "wav":
            return wav_path, "audio/wav", f"{job_id}_output.wav"

        # On-demand MP3 export
        mp3_path = job_dir / "output.mp3"
        if not mp3_path.is_file():
            try:
                AudioService.export_mp3(wav_path, mp3_path)
            except Exception as mp3_err:
                logging.error(f"Failed to export MP3 for job '{job_id}': {mp3_err}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Audio format conversion failed",
                )

        return mp3_path, "audio/mpeg", f"{job_id}_output.mp3"

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
