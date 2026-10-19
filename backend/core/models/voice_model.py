"""
Voice model entity representing registered AI voice profiles.

Defines persistence schema for Chatterbox reference voice models.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, JSON, DateTime
from core.database.base_class import Base


class VoiceModel(Base):
    """
    SQLAlchemy persistence model for voices stored in the catalog.
    """

    __tablename__ = "voices"

    id = Column(String(64), primary_key=True, index=True, doc="Unique slug identifier for the voice")
    name = Column(String(128), nullable=False, doc="Human-readable voice display name")
    description = Column(Text, nullable=True, doc="Detailed description of voice characteristics")
    style = Column(String(64), nullable=True, doc="Voice delivery style or genre")
    languages = Column(JSON, nullable=False, default=list, doc="List of supported ISO language codes")
    model_type = Column(String(32), nullable=False, index=True, doc="Model pipeline type: chatterbox_reference")
    preview_file = Column(String(256), nullable=True, doc="Relative path to preview audio sample under voices dir")
    asset_file = Column(String(256), nullable=False, doc="Relative path to model checkpoint or reference wav")
    license = Column(Text, nullable=False, doc="Declared license or provenance text")
    usage_permission = Column(Text, nullable=False, doc="Permitted operational usage scope")
    status = Column(String(32), nullable=False, default="available", index=True, doc="Status: available, disabled, or experimental")
    limitations = Column(Text, nullable=True, doc="Known quality boundaries or operational constraints")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
