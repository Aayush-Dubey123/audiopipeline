"""
Controller managing inference job status, audio artifact retrieval, and job deletion.
"""

from pathlib import Path
from typing import Tuple
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from core import logger
from core.config import get_settings
from core.cruds.job_crud import JobCRUD
from core.services.jobs.cleanup_service import CleanupService
from core.services.audio.audio_service import AudioService
from core.apis.schemas.responses.job_responses import JobStatusResponse, JobDeleteResponse

logging = logger(__name__)
settings = get_settings()


class JobController:
    """
    Business logic and access control for job lifecycles and artifacts.
    """

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
            created_at=job.created_at,
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

        source_path = job_dir / "source.wav"
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
