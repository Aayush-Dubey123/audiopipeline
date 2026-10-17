"""
Request schemas and parameter contracts for voice catalog endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class VoiceFilterParams(BaseModel):
    """
    Query parameters for filtering catalog voices.
    """

    model_type: Optional[str] = Field(
        default=None,
        description="Filter voices by pipeline type ('openvoice_reference' or 'rvc_model')",
    )
