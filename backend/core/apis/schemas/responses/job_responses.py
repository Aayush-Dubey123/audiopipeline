"""
Response data contracts for inference job status, progress, and lifecycle operations.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


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
    expires_at: datetime = Field(..., description="Job artifact expiry UTC timestamp")
    output_url: Optional[str] = Field(default=None, description="Relative URL to download completed audio")
    source_url: Optional[str] = Field(default=None, description="Relative URL to download uploaded source audio")


class JobDeleteResponse(BaseModel):
    """
    Confirmation payload returned upon successful cancellation or deletion of a job.
    """

    job_id: str = Field(..., description="Identifier of the deleted job")
    status: str = Field(default="deleted", description="Confirmation status")
    message: str = Field(..., description="Informative status message")
