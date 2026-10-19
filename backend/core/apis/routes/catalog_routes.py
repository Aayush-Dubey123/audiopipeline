"""
HTTP routes for voice catalog, configuration, and operational health endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from core import logger
from core.database.database import get_db
from core.apis.schemas.responses.catalog_responses import (
    HealthResponse,
    VoiceResponse,
    VoiceListResponse,
    PublicConfigResponse,
)
from core.controllers.catalog_controller import CatalogController

logging = logger(__name__)

router = APIRouter(tags=["Catalog & Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Readiness Check",
    description="Returns service availability, build version, and current UTC server time.",
)
def get_health() -> HealthResponse:
    """
    Handle GET /v1/health requests.

    Delegates status assembly to the controller layer and wraps execution
    in standardized error handling.

    Returns:
        HealthResponse: Status payload with current system state.

    Raises:
        HTTPException 500: If health check evaluation fails unexpectedly.
    """
    logging.info("Calling GET /v1/health endpoint")
    try:
        return CatalogController.get_health_status()
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/health endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/health endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get(
    "/voices",
    response_model=VoiceListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Registered Voices",
    description="Retrieves available registered voices, filtering out experimental models if disabled.",
)
def list_voices(
    model_type: Optional[str] = Query(None, description="Optional pipeline type filter (chatterbox_reference)"),
    db: Session = Depends(get_db),
) -> VoiceListResponse:
    """
    Handle GET /v1/voices endpoint requests.

    Args:
        model_type (Optional[str]): Optional query parameter filtering by synthesis engine.
        db (Session): Injected database session dependency.

    Returns:
        VoiceListResponse: Collection of accessible voices without backend filepaths.

    Raises:
        HTTPException 500: If voice listing fails unexpectedly.
    """
    logging.info("Calling GET /v1/voices endpoint")
    try:
        return CatalogController.list_voices(db=db, model_type=model_type)
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/voices endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/voices endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get(
    "/voices/{voice_id}",
    response_model=VoiceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Voice Profile Details",
    description="Fetches public profile details for a specific voice by identifier.",
)
def get_voice(
    voice_id: str,
    db: Session = Depends(get_db),
) -> VoiceResponse:
    """
    Handle GET /v1/voices/{voice_id} endpoint requests.

    Args:
        voice_id (str): Target voice identifier slug.
        db (Session): Injected database session dependency.

    Returns:
        VoiceResponse: Voice metadata and capabilities.

    Raises:
        HTTPException 404: If voice is not found or is an unexposed experimental voice.
        HTTPException 500: If voice lookup encounters an internal server error.
    """
    logging.info(f"Calling GET /v1/voices/{voice_id} endpoint")
    try:
        return CatalogController.get_voice_by_id(db=db, voice_id=voice_id)
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/voices/{voice_id} endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/voices/{voice_id} endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get(
    "/voices/{voice_id}/preview",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Stream Audio Preview Sample",
    description="Streams the reference audio preview sample for a supported voice.",
)
def get_voice_preview(
    voice_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Handle GET /v1/voices/{voice_id}/preview endpoint requests.

    Args:
        voice_id (str): Target voice identifier slug.
        db (Session): Injected database session dependency.

    Returns:
        FileResponse: Streaming audio WAV preview file.

    Raises:
        HTTPException 404: If voice or preview audio is unavailable.
        HTTPException 500: If streaming encounters an unexpected error.
    """
    logging.info(f"Calling GET /v1/voices/{voice_id}/preview endpoint")
    try:
        preview_path = CatalogController.get_voice_preview_path(db=db, voice_id=voice_id)
        return FileResponse(
            path=str(preview_path),
            media_type="audio/wav",
            filename=f"{voice_id}_preview.wav",
        )
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/voices/{voice_id}/preview endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/voices/{voice_id}/preview endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get(
    "/config/public",
    response_model=PublicConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Public System Configuration",
    description="Provides safe limits, quotas, and feature flags to the frontend client.",
)
def get_public_config() -> PublicConfigResponse:
    """
    Handle GET /v1/config/public requests.

    Returns:
        PublicConfigResponse: Public limits and capability flags.

    Raises:
        HTTPException 500: If config assembly fails.
    """
    logging.info("Calling GET /v1/config/public endpoint")
    try:
        return CatalogController.get_public_config()
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/config/public endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/config/public endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )
