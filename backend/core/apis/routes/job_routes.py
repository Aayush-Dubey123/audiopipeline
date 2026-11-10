"""
HTTP routes for inference job creation, status polling, and audio artifact streaming.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from core import logger
from core.database.database import get_db
from core.controllers.job_controller import JobController
from core.apis.middleware import get_hashed_client_key
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

router = APIRouter(tags=["Inference Jobs & Audio"])


@router.post(
    "/tts",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Text-to-Speech Job",
    description="Submits a script for asynchronous speech synthesis using a registered voice.",
)
def create_tts(
    request: TTSJobRequest,
    http_request: Request,
    db: Session = Depends(get_db),
) -> JobCreatedResponse:
    """
    Handle POST /v1/tts endpoint requests.

    Args:
        request (TTSJobRequest): Validated text and voice parameters.
        http_request (Request): Inbound HTTP request for client IP hashing.
        db (Session): Database session dependency.

    Returns:
        JobCreatedResponse: Queued job token and status confirmation.

    Raises:
        HTTPException 400: If voice or language constraints are violated.
        HTTPException 404: If voice does not exist.
        HTTPException 429: If queue is full or active job quota is reached.
        HTTPException 500: If unexpected server error occurs.
    """
    logging.info("Calling POST /v1/tts endpoint")
    try:
        client_key = get_hashed_client_key(http_request)
        return JobController.create_tts_job(db=db, client_key=client_key, request=request)
    except HTTPException as httperror:
        logging.error(f"Error in POST /v1/tts endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in POST /v1/tts endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post(
    "/tts/custom",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Custom Reference Voice TTS Job",
    description="Synthesizes speech cloned from a user-supplied reference audio recording.",
)
async def create_custom_tts(
    http_request: Request,
    text: str = Form(..., description="Script text to synthesize"),
    language: str = Form("en", description="Target language ('en', 'hi', or 'mr')"),
    speed: float = Form(1.0, description="Speech rate multiplier (0.5 to 2.0)"),
    consent: bool = Form(..., description="Verification of voice ownership or authorization"),
    file: UploadFile = File(..., description="Reference audio recording file"),
    db: Session = Depends(get_db),
) -> JobCreatedResponse:
    """
    Handle POST /v1/tts/custom multipart endpoint requests.

    Args:
        http_request (Request): Inbound HTTP request.
        text (str): Script text from form body.
        language (str): Language code.
        speed (float): Playback speed.
        consent (bool): Explicit voice cloning consent.
        file (UploadFile): Audio reference file upload.
        db (Session): Database session dependency.

    Returns:
        JobCreatedResponse: Queued job metadata.

    Raises:
        HTTPException 400: If consent is missing, audio fails quality inspection, or parameters invalid.
        HTTPException 429: If queue is full or per-client quota is reached.
        HTTPException 500: If unexpected internal error occurs.
    """
    logging.info("Calling POST /v1/tts/custom endpoint")
    try:
        client_key = get_hashed_client_key(http_request)
        return await JobController.create_custom_tts_job(
            db=db,
            client_key=client_key,
            text=text,
            language=language,
            speed=speed,
            consent=consent,
            file=file,
        )
    except HTTPException as httperror:
        logging.error(f"Error in POST /v1/tts/custom endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in POST /v1/tts/custom endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post(
    "/vc",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Voice Conversion Job",
    description="Converts uploaded source audio into a target RVC voice model.",
)
async def create_vc(
    http_request: Request,
    voice_id: str = Form(..., description="Target RVC voice identifier slug"),
    pitch_shift: int = Form(0, description="Pitch modification in semitones (-12 to +12)"),
    file: UploadFile = File(..., description="Source speech audio file to convert"),
    db: Session = Depends(get_db),
) -> JobCreatedResponse:
    """
    Handle POST /v1/vc multipart endpoint requests.

    Args:
        http_request (Request): Inbound HTTP request.
        voice_id (str): Target voice profile identifier.
        pitch_shift (int): Pitch shift in semitones.
        file (UploadFile): Audio file to convert.
        db (Session): Database session dependency.

    Returns:
        JobCreatedResponse: Queued conversion job envelope.

    Raises:
        HTTPException 400: If voice model or audio upload parameters are invalid.
        HTTPException 404: If target voice is not found.
        HTTPException 429: If queue is full or quota exceeded.
        HTTPException 500: If unexpected internal failure occurs.
    """
    logging.info(f"Calling POST /v1/vc endpoint for voice '{voice_id}'")
    try:
        client_key = get_hashed_client_key(http_request)
        return await JobController.create_vc_job(
            db=db,
            client_key=client_key,
            voice_id=voice_id,
            pitch_shift=pitch_shift,
            file=file,
        )
    except HTTPException as httperror:
        logging.error(f"Error in POST /v1/vc endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in POST /v1/vc endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post(
    "/text/cleanup",
    response_model=TextCleanupResponse,
    status_code=status.HTTP_200_OK,
    summary="Enhance and Clean Script Text",
    description="Expands abbreviations, normalizes numbers, and cleans punctuation for optimal synthesis.",
)
def cleanup_text(
    request: TextCleanupRequest,
) -> TextCleanupResponse:
    """
    Handle POST /v1/text/cleanup endpoint requests.

    Args:
        request (TextCleanupRequest): Input text and target language parameters.

    Returns:
        TextCleanupResponse: Enhanced script text.

    Raises:
        HTTPException 500: If text cleaning encounters unexpected error.
    """
    logging.info("Calling POST /v1/text/cleanup endpoint")
    try:
        return JobController.cleanup_text(request=request)
    except HTTPException as httperror:
        logging.error(f"Error in POST /v1/text/cleanup endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in POST /v1/text/cleanup endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


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


@router.post(
    "/jobs/{job_id}/resume",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resume or Retry Inference Job",
    description="Resumes a failed, interrupted, or stalled long-form inference job, reusing finished chunks.",
)
def resume_job(
    job_id: str,
    http_request: Request,
    db: Session = Depends(get_db),
) -> JobCreatedResponse:
    """
    Handle POST /v1/jobs/{job_id}/resume endpoint requests.

    Args:
        job_id (str): Unique URL-safe job token identifier.
        http_request (Request): Inbound HTTP request.
        db (Session): Database session dependency.

    Returns:
        JobCreatedResponse: Queued job confirmation payload.

    Raises:
        HTTPException 400: If job is already completed or cannot be resumed.
        HTTPException 404: If job does not exist.
        HTTPException 429: If queue is full or client quota is exceeded.
        HTTPException 500: If unexpected internal error occurs.
    """
    logging.info(f"Calling POST /v1/jobs/{job_id}/resume endpoint")
    try:
        client_key = get_hashed_client_key(http_request)
        return JobController.resume_job(db=db, client_key=client_key, job_id=job_id)
    except HTTPException as httperror:
        logging.error(f"Error in POST /v1/jobs/{job_id}/resume endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in POST /v1/jobs/{job_id}/resume endpoint: {error}")
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
        format (Optional[str]): Desired audio format ('wav', 'mp3', or 'mp3_small').
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


@router.post(
    "/reference/check",
    response_model=ReferenceCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Reference Audio Safety",
    description="Analyzes reference audio acoustic safety (speech length, loudness, noise, clipping, reverb, speaker, music).",
)
async def check_reference(
    http_request: Request,
    file: Optional[UploadFile] = File(None),
    voice_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> ReferenceCheckResponse:
    """
    Handle POST /v1/reference/check requests.
    """
    logging.info("Calling POST /v1/reference/check endpoint")
    try:
        client_key = get_hashed_client_key(http_request)
        return await JobController.check_reference_safety(
            db=db,
            client_key=client_key,
            file=file,
            voice_id=voice_id,
        )
    except HTTPException as httperror:
        logging.error(f"Error in POST /v1/reference/check: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in POST /v1/reference/check: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error during reference check",
        )


@router.post(
    "/tts/preview",
    response_model=TTSPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate TTS Preview",
    description="Generates rapid preview of up to 200 characters with the chosen reference voice.",
)
def tts_preview(
    request: TTSPreviewRequest,
    http_request: Request,
    db: Session = Depends(get_db),
) -> TTSPreviewResponse:
    """
    Handle POST /v1/tts/preview requests.
    """
    logging.info("Calling POST /v1/tts/preview endpoint")
    try:
        client_key = get_hashed_client_key(http_request)
        return JobController.generate_tts_preview(
            db=db,
            client_key=client_key,
            request=request,
        )
    except HTTPException as httperror:
        logging.error(f"Error in POST /v1/tts/preview: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in POST /v1/tts/preview: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error during TTS preview",
        )


@router.post(
    "/tts/preview/custom",
    response_model=TTSPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Custom Reference TTS Preview",
    description="Generates rapid preview of up to 200 characters conditioned on uploaded reference voice audio.",
)
async def tts_preview_custom(
    http_request: Request,
    text: str = Form(..., description="Script text (up to 200 characters)"),
    file: UploadFile = File(..., description="Custom reference voice audio file"),
    language: str = Form("hi", description="Language code ('hi', 'en', 'mr')"),
    speed: float = Form(1.0, description="Speech playback rate multiplier"),
    consent: bool = Form(..., description="Confirmation of voice ownership or authorization"),
    db: Session = Depends(get_db),
) -> TTSPreviewResponse:
    """
    Handle POST /v1/tts/preview/custom requests.
    """
    logging.info("Calling POST /v1/tts/preview/custom endpoint")
    try:
        client_key = get_hashed_client_key(http_request)
        return await JobController.generate_custom_tts_preview(
            db=db,
            client_key=client_key,
            text=text,
            file=file,
            language=language,
            speed=speed,
            consent=consent,
        )
    except HTTPException as httperror:
        logging.error(f"Error in POST /v1/tts/preview/custom: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in POST /v1/tts/preview/custom: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error during custom TTS preview",
        )


@router.get(
    "/audio/preview/{preview_id}",
    summary="Stream Preview Audio",
    description="Streams generated preview audio WAV file.",
)
def get_preview_audio(
    preview_id: str,
) -> FileResponse:
    """
    Handle GET /v1/audio/preview/{preview_id} requests.
    """
    logging.info(f"Calling GET /v1/audio/preview/{preview_id} endpoint")
    try:
        file_path = JobController.get_preview_file(preview_id)
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=f"preview_{preview_id}.wav",
        )
    except HTTPException as httperror:
        raise httperror
    except Exception as error:
        logging.error(f"Error streaming preview audio: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )

