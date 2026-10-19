"""
Response data contracts for inference job status, progress, and lifecycle operations.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class JobCreatedResponse(BaseModel):
    """
    Immediate acknowledgement returned upon successfully enqueuing an inference job.
    """

    job_id: str = Field(..., description="Unique URL-safe job token identifier")
    kind: str = Field(..., description="Execution pipeline type: tts, tts_custom, or vc")
    status: str = Field(default="queued", description="Initial lifecycle state: queued")
    message: str = Field(..., description="User-facing status confirmation message")
    ai_generated: bool = Field(default=True, description="Attribution flag marking synthetic output")


class JobStatusResponse(BaseModel):
    """
    Client-facing response model representing status and outcomes of an asynchronous job.
    """

    job_id: str = Field(..., description="Unique job token identifier")
    kind: str = Field(..., description="Synthesis or conversion pipeline type")
    status: str = Field(..., description="Current job status: queued, processing, completed, failed, expired")
    voice_id: Optional[str] = Field(default=None, description="Voice identifier used for generation")
    progress_message: Optional[str] = Field(default=None, description="Current progress status description")
    error_code: Optional[str] = Field(default=None, description="Standardized error code if failed")
    error_message: Optional[str] = Field(default=None, description="Human-readable error explanation if failed")
    created_at: datetime = Field(..., description="Job creation UTC timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Job execution start UTC timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Job record last modification UTC timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion UTC timestamp")
    expires_at: datetime = Field(..., description="Job artifact expiry UTC timestamp")
    stage: Optional[str] = Field(default=None, description="Current execution stage name: Queued, Loading voice, Generating speech, Polishing audio, Completed")
    stage_index: Optional[int] = Field(default=None, description="1-based index of active execution stage")
    stage_total: Optional[int] = Field(default=None, description="Total count of stages in pipeline")
    chunk_done: Optional[int] = Field(default=None, description="Completed chunk count for long-form synthesis jobs")
    chunk_total: Optional[int] = Field(default=None, description="Total planned chunks for long-form synthesis jobs")
    eta_seconds: Optional[int] = Field(default=None, description="Rolling ETA in seconds based on chunk processing times")
    output_url: Optional[str] = Field(default=None, description="Relative URL to download completed audio")
    source_url: Optional[str] = Field(default=None, description="Relative URL to download uploaded source audio")
    output_files: Optional[dict] = Field(default=None, description="Available output files and byte sizes")
    confidence_label: Optional[str] = Field(default=None, description="Quality confidence label: good, fair, or low")


class JobDeleteResponse(BaseModel):
    """
    Confirmation payload returned upon successful cancellation or deletion of a job.
    """

    job_id: str = Field(..., description="Identifier of the deleted job")
    status: str = Field(default="deleted", description="Confirmation status")
    message: str = Field(..., description="Informative status message")


class TextCleanupResponse(BaseModel):
    """
    Response payload returned by the text cleanup and enhancement endpoint.
    """

    original: str = Field(..., description="Unmodified original input text")
    cleaned: str = Field(..., description="Cleaned, expanded, and punctuated text")
    provider_used: str = Field(..., description="Enhancement engine utilized ('rule_based', 'groq', 'gemini')")
    service_available: bool = Field(default=True, description="Whether LLM service provider keys are configured")


class ReferenceCheckMetrics(BaseModel):
    """
    Acoustic measurements for reference safety evaluation.
    """

    speech_length_s: float
    total_duration_s: float
    loudness_dbfs: float
    snr_db: float
    clipping_ratio: float
    echo_reverb: str
    speaker_consistency: float
    multi_speaker_detected: bool
    music_detected: bool


class ReferenceCheckResponse(BaseModel):
    """
    Response contract for voice reference acoustic safety inspection.
    """

    score: int = Field(..., ge=0, le=100, description="Overall quality score from 0 to 100")
    is_usable: bool = Field(..., description="Whether audio is usable (hard blocks if False)")
    advice: list[str] = Field(..., description="Plain-language diagnostics and recommendations")
    metrics: ReferenceCheckMetrics = Field(..., description="Detailed acoustic metric measurements")


class TTSPreviewResponse(BaseModel):
    """
    Rapid <=200 character voice synthesis preview.
    """

    job_id: Optional[str] = Field(default=None, description="Job identifier token for tracking progress stages")
    preview_url: str = Field(..., description="Direct URL to listen to the generated preview audio")
    duration_seconds: float = Field(default=0.0, description="Duration of preview audio in seconds")
    text_preview: str = Field(..., description="Text synthesized for the preview")
    confidence_label: Optional[str] = Field(
        default="good", description="Predicted similarity confidence: good, fair, or low"
    )

