"""
Request schemas and parameter contracts for job queries and audio streaming.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AudioFormatQuery(BaseModel):
    """
    Query parameter specification for requested output audio formats.
    """

    format: Optional[str] = Field(
        default="wav",
        description="Target audio format ('wav' or 'mp3')",
    )
