"""
Request schemas and parameter contracts for job creation, queries, and audio streaming.
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


class TTSJobRequest(BaseModel):
    """
    Payload contract for submitting a standard text-to-speech synthesis job.
    """

    voice_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Registered catalog voice identifier slug",
        examples=["my_voice"],
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Script text to synthesize into spoken audio",
        examples=["Hello and welcome to AI Voice Studio."],
    )
    language: str = Field(
        default="en",
        description="Language code for synthesis: English ('en'), Hindi ('hi'), or Marathi ('mr')",
        examples=["en"],
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech rate multiplier (0.5 to 2.0)",
    )


class TextCleanupRequest(BaseModel):
    """
    Payload contract for requesting rule-based or LLM text cleaning prior to synthesis.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Raw script text to normalize and clean",
    )
    language: str = Field(
        default="en",
        description="Target language code ('en', 'hi', or 'mr')",
    )
