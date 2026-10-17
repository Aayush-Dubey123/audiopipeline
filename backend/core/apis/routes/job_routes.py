"""
HTTP routes for asynchronous job lifecycle polling and audio artifact streaming.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from core import logger
from core.database.database import get_db
from core.controllers.job_controller import JobController
from core.apis.schemas.responses.job_responses import JobStatusResponse, JobDeleteResponse

logging = logger(__name__)

router = APIRouter(tags=["Inference Jobs & Audio"])


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Job Status & Progress",
    description="Polls current status, execution percentage or message, and download links for a job.",
)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """
    Handle GET /v1/jobs/{job_id} status queries.

    Args:
        job_id (str): Unique URL-safe job token identifier.
        db (Session): Database session dependency.

    Returns:
        JobStatusResponse: Current execution progress and outcome metadata.

    Raises:
        HTTPException 404: If job does not exist.
        HTTPException 500: If internal status query fails.
    """
    logging.info(f"Calling GET /v1/jobs/{job_id} endpoint")
    try:
        return JobController.get_job_status(db=db, job_id=job_id)
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/jobs/{job_id} endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/jobs/{job_id} endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get(
    "/audio/{job_id}/output",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Download Synthesized Audio Output",
    description="Streams or downloads the completed synthesized or converted audio artifact.",
)
def get_audio_output(
    job_id: str,
    format: Optional[str] = Query("wav", description="Audio format ('wav' or 'mp3')"),
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Handle GET /v1/audio/{job_id}/output requests.

    Args:
        job_id (str): Unique job token identifier.
        format (Optional[str]): Desired audio format ('wav' or 'mp3').
        db (Session): Database session dependency.

    Returns:
        FileResponse: Streaming audio binary payload.

    Raises:
        HTTPException 400: If job is incomplete or format unsupported.
        HTTPException 404: If job or audio file is missing.
        HTTPException 500: If audio retrieval fails unexpectedly.
    """
    logging.info(f"Calling GET /v1/audio/{job_id}/output endpoint (format={format})")
    try:
        file_path, media_type, filename = JobController.get_job_output_file(
            db=db,
            job_id=job_id,
            audio_format=format or "wav",
        )
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename,
        )
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/audio/{job_id}/output endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/audio/{job_id}/output endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get(
    "/audio/{job_id}/source",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Download Uploaded Source Audio",
    description="Streams or downloads the original input audio uploaded for conversion.",
)
def get_audio_source(
    job_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Handle GET /v1/audio/{job_id}/source requests.

    Args:
        job_id (str): Unique job token identifier.
        db (Session): Database session dependency.

    Returns:
        FileResponse: Streaming source audio WAV payload.

    Raises:
        HTTPException 404: If source audio artifact is not found.
        HTTPException 500: If audio streaming fails.
    """
    logging.info(f"Calling GET /v1/audio/{job_id}/source endpoint")
    try:
        file_path, media_type, filename = JobController.get_job_source_file(
            db=db,
            job_id=job_id,
        )
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename,
        )
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/audio/{job_id}/source endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/audio/{job_id}/source endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.delete(
    "/jobs/{job_id}",
    response_model=JobDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel or Delete Job",
    description="Cancels an active or pending job, removes disk artifacts, and deletes the record.",
)
def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> JobDeleteResponse:
    """
    Handle DELETE /v1/jobs/{job_id} requests.

    Args:
        job_id (str): Target job identifier.
        db (Session): Database session dependency.

    Returns:
        JobDeleteResponse: Confirmation of job deletion.

    Raises:
        HTTPException 404: If job was not found.
        HTTPException 500: If deletion failed unexpectedly.
    """
    logging.info(f"Calling DELETE /v1/jobs/{job_id} endpoint")
    try:
        return JobController.delete_job(db=db, job_id=job_id)
    except HTTPException as httperror:
        logging.error(f"Error in DELETE /v1/jobs/{job_id} endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in DELETE /v1/jobs/{job_id} endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )
