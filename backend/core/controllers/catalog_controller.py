"""
Controller handling voice catalog, public configuration, and system lifecycle operations.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from core import logger
from core.config import get_settings
from core.cruds.voice_crud import VoiceCRUD
from core.apis.schemas.responses.catalog_responses import (
    HealthResponse,
    VoiceResponse,
    VoiceListResponse,
    PublicConfigResponse,
)

logging = logger(__name__)
settings = get_settings()


class CatalogController:
    """
    Business logic and operational control for catalog and service health.
    """

    @staticmethod
    def get_health_status() -> HealthResponse:
        """
        Evaluate and assemble the current operational health of the application.

        Constructs a timestamped health payload confirming system availability.

        Returns:
            HealthResponse: Structured health contract containing status, version, and UTC time.

        Raises:
            Exception: If an unexpected error occurs during status generation.
        """
        logging.info("Executing CatalogController.get_health_status")
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            return HealthResponse(
                status="ok",
                version="0.1.0",
                time=current_time,
            )
        except Exception as error:
            logging.error(f"Error in CatalogController.get_health_status: {error}")
            raise error

    @staticmethod
    def list_voices(db: Session, model_type: Optional[str] = None) -> VoiceListResponse:
        """
        List registered voices permitted for client consumption.

        Hides experimental voices unless EXPOSE_EXPERIMENTAL_VOICES is enabled.

        Args:
            db (Session): Active database session.
            model_type (Optional[str]): Optional model type filter.

        Returns:
            VoiceListResponse: Collection of accessible voices with internal paths stripped.

        Raises:
            Exception: If catalog query fails.
        """
        logging.info("Executing CatalogController.list_voices")
        try:
            voices = VoiceCRUD.list_voices(
                db=db,
                model_type=model_type,
                include_experimental=settings.EXPOSE_EXPERIMENTAL_VOICES,
            )
            voice_items = [
                VoiceResponse(
                    id=v.id,
                    name=v.name,
                    description=v.description,
                    style=v.style,
                    languages=v.languages or ["en"],
                    model_type=v.model_type,
                    has_preview=bool(v.preview_file),
                    license=v.license,
                    usage_permission=v.usage_permission,
                    status=v.status,
                    limitations=v.limitations,
                )
                for v in voices
            ]
            return VoiceListResponse(voices=voice_items, total=len(voice_items))
        except Exception as error:
            logging.error(f"Error in CatalogController.list_voices: {error}")
            raise error

    @staticmethod
    def get_voice_by_id(db: Session, voice_id: str) -> VoiceResponse:
        """
        Retrieve a single voice by its identifier with access controls applied.

        Args:
            db (Session): Active database session.
            voice_id (str): Unique voice identifier.

        Returns:
            VoiceResponse: Voice profile details without server asset paths.

        Raises:
            HTTPException 404: If the voice does not exist or is an unexposed experimental model.
        """
        logging.info(f"Executing CatalogController.get_voice_by_id for voice '{voice_id}'")
        voice = VoiceCRUD.get_voice_by_id(db=db, voice_id=voice_id)
        if not voice:
            logging.warning(f"Voice '{voice_id}' not found in catalog")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        if voice.status == "experimental" and not settings.EXPOSE_EXPERIMENTAL_VOICES:
            logging.warning(f"Access to experimental voice '{voice_id}' rejected by policy")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        return VoiceResponse(
            id=voice.id,
            name=voice.name,
            description=voice.description,
            style=voice.style,
            languages=voice.languages or ["en"],
            model_type=voice.model_type,
            has_preview=bool(voice.preview_file),
            license=voice.license,
            usage_permission=voice.usage_permission,
            status=voice.status,
            limitations=voice.limitations,
        )

    @staticmethod
    def get_voice_preview_path(db: Session, voice_id: str) -> Path:
        """
        Locate and validate the static audio preview file for a voice.

        Args:
            db (Session): Active database session.
            voice_id (str): Unique voice identifier.

        Returns:
            Path: Absolute filesystem path to the verified preview audio file.

        Raises:
            HTTPException 404: If voice or preview file is missing or unexposed.
        """
        logging.info(f"Executing CatalogController.get_voice_preview_path for voice '{voice_id}'")
        voice = VoiceCRUD.get_voice_by_id(db=db, voice_id=voice_id)
        if not voice:
            logging.warning(f"Voice '{voice_id}' not found for preview")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        if voice.status == "experimental" and not settings.EXPOSE_EXPERIMENTAL_VOICES:
            logging.warning(f"Preview access for experimental voice '{voice_id}' rejected by policy")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{voice_id}' not found",
            )

        if not voice.preview_file:
            logging.warning(f"Voice '{voice_id}' does not have a registered preview file")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No preview available for voice '{voice_id}'",
            )

        preview_path = (settings.voices_path / voice.preview_file).resolve()
        try:
            preview_path.relative_to(settings.voices_path.resolve())
        except ValueError:
            logging.error(f"Security error: preview path traversal detected for voice '{voice_id}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preview file not found",
            )

        if not preview_path.is_file():
            logging.warning(f"Preview file for voice '{voice_id}' missing on disk: {preview_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preview file not found",
            )

        return preview_path

    @staticmethod
    def get_public_config() -> PublicConfigResponse:
        """
        Assemble safe public system limits, operational quotas, and feature flags.

        Returns:
            PublicConfigResponse: Publicly disclosable system metadata.
        """
        logging.info("Executing CatalogController.get_public_config")
        return PublicConfigResponse(
            max_text_chars=settings.MAX_TEXT_CHARS,
            job_ttl_minutes=settings.JOB_TTL_MINUTES,
            queue_max_size=settings.QUEUE_MAX_SIZE,
            job_timeout_seconds=settings.JOB_TIMEOUT_SECONDS,
            max_audio_seconds=30,
            max_upload_mb=10,
            expose_experimental_voices=settings.EXPOSE_EXPERIMENTAL_VOICES,
            features={
                "openvoice": True,
                "rvc": True,
                "llm_cleanup": False,
            },
        )
