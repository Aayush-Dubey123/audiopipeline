"""
SQLAlchemy persistence model for asynchronous inference jobs.

Tracks execution state, pipeline kind, timing, quotas, expiration, and chunk progress.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, JSON, DateTime, Integer
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
    client_key = Column(String(64), nullable=False, index=True, doc="Client (hashed IP) identifier for quota tracking")
    payload = Column(JSON, nullable=True, doc="Job input configuration and sanitized parameters")
    output_metadata = Column(JSON, nullable=True, doc="Generated audio duration, format, and size metadata")

    # Long-form chunk progress fields (null for short / VC jobs)
    chunk_done = Column(Integer, nullable=True, doc="Number of synthesis chunks completed so far")
    chunk_total = Column(Integer, nullable=True, doc="Total number of synthesis chunks in this job")
    eta_seconds = Column(Integer, nullable=True, doc="Rolling ETA in seconds based on average chunk wall time")

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
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when job reached completed or failed status; TTL counts from this",
    )
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Timestamp after which job artifacts are cleaned up (set on completion)",
    )
