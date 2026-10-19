"""
Response data contracts for voice catalog, configuration, and health endpoints.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """
    Payload returned by the system health status endpoint.
    """

    status: str = Field(default="ok", description="Operational health status indicator")
    version: str = Field(default="0.1.0", description="Backend service version string")
    time: str = Field(..., description="Current server UTC timestamp in ISO-8601 format")


class VoiceResponse(BaseModel):
    """
    Public representation of a catalog voice profile.

    Omits all internal filesystem paths and asset file locations.
    """

    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(..., description="Unique voice identifier slug")
    name: str = Field(..., description="Human-readable voice display name")
    description: Optional[str] = Field(default=None, description="Detailed voice description")
    style: Optional[str] = Field(default=None, description="Voice style or genre")
    languages: List[str] = Field(default_factory=list, description="Supported ISO language codes")
    model_type: str = Field(..., description="Synthesis pipeline model type")
    has_preview: bool = Field(default=False, description="Flag indicating if audio preview is available")
    license: str = Field(..., description="Declared license or provenance attribution")
    usage_permission: str = Field(..., description="Permitted usage scope")
    status: str = Field(..., description="Voice availability status")
    limitations: Optional[str] = Field(default=None, description="Known operational limitations")


class VoiceListResponse(BaseModel):
    """
    Paginated or aggregated collection of registered catalog voices.
    """

    voices: List[VoiceResponse] = Field(default_factory=list, description="List of matching voice entries")
    total: int = Field(..., description="Total count of matching voices")


class PublicConfigResponse(BaseModel):
    """
    Publicly safe system configuration, limits, and feature flags.
    """

    max_text_chars: int = Field(..., description="Maximum allowed characters per text-to-speech request")
    job_ttl_minutes: int = Field(..., description="Retention time for temporary generated job artifacts")
    queue_max_size: int = Field(..., description="Maximum pending jobs in inference queue")
    job_timeout_seconds: int = Field(..., description="Execution timeout for inference jobs")
    max_audio_seconds: int = Field(default=30, description="Maximum allowed audio duration for conversions")
    max_upload_mb: int = Field(default=10, description="Maximum allowed audio upload file size in megabytes")
    expose_experimental_voices: bool = Field(default=False, description="Whether experimental voices are enabled")
    seconds_per_audio_second: float = Field(default=0.8, description="Measured seconds of compute per second of audio")
    compute_seconds_per_audio_second: float = Field(default=0.8, description="Measured seconds of compute per second of audio")
    features: Dict[str, bool] = Field(default_factory=dict, description="Active system capability flags")
