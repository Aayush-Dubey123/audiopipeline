"""
Response data contracts for voice catalog and health endpoints.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Payload returned by the system health status endpoint.
    """

    status: str = Field(default="ok", description="Operational health status indicator")
    version: str = Field(default="0.1.0", description="Backend service version string")
    time: str = Field(..., description="Current server UTC timestamp in ISO-8601 format")
