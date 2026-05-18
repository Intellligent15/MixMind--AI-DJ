import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class TranscriptionStatus(str, enum.Enum):
    not_attempted = "not_attempted"
    success = "success"
    skipped_instrumental = "skipped_instrumental"
    error = "error"


class Transcription(Base):
    __tablename__ = "transcriptions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String, nullable=False, default="large-v3"
    )
    status: Mapped[TranscriptionStatus] = mapped_column(
        Enum(TranscriptionStatus, name="transcription_status"),
        nullable=False,
        default=TranscriptionStatus.not_attempted,
    )

    # Detected language (ISO 639-1, e.g. "en"). Null for skipped/error rows.
    language: Mapped[str | None] = mapped_column(String, nullable=True)

    # List of {start, end, text, words: [{start, end, word}]}.
    # Empty list for skipped/error rows.
    segments: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)

    # Threshold used at the time of the skip decision. Persisted so a future
    # threshold change doesn't make a historical "skipped" row look wrong.
    vocal_rms_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Vocal RMS we read from the Stems row at decision time. Useful for
    # debugging "why was this skipped?" without joining back.
    vocal_rms_observed: Mapped[float | None] = mapped_column(Float, nullable=True)

    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
