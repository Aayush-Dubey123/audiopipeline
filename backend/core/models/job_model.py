"""
SQLAlchemy persistence model for asynchronous inference jobs.

Tracks execution state, pipeline kind, timing, quotas, and expiration.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, JSON, DateTime
from core.database.base_class import Base


class JobModel(Base):
    """
    Persistence entity representing an inference job and its lifecycle.
    """

    __tablename__ = "jobs"

    id = Column(String(64), primary_key=True, index=True, doc="Unguessable URL-safe token identifier")
    kind = Column(String(32), nullable=False, index=True, doc="Pipeline kind: tts, tts_custom, or vc")
    status = Column(
        String(32),
        nullable=False,
        default="queued",
        index=True,
        doc="Execution status: queued, processing, completed, failed, expired",
    )
    voice_id = Column(String(64), nullable=True, index=True, doc="Associated voice identifier slug")
    progress_message = Column(String(256), nullable=True, doc="Human-readable progress notification")
    error_code = Column(String(64), nullable=True, doc="Standardized system error code")
    error_message = Column(Text, nullable=True, doc="Sanitized client-facing error description")
    client_key = Column(String(64), nullable=False, index=True, doc="Hashed client identifier for quota tracking")
    payload = Column(JSON, nullable=True, doc="Job input configuration and sanitized parameters")
    output_metadata = Column(JSON, nullable=True, doc="Generated audio duration, format, and size metadata")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="Timestamp when job was created",
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        doc="Timestamp when job record was last modified",
    )
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Timestamp after which job artifacts are cleaned up",
    )
