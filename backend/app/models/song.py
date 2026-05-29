import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class SongStatus(str, enum.Enum):
    pending = "pending"
    downloading = "downloading"
    downloaded = "downloaded"
    analyzing = "analyzing"
    analyzed = "analyzed"
    separating = "separating"
    transcribing = "transcribing"
    ready = "ready"
    failed = "failed"


class Song(Base):
    __tablename__ = "songs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    youtube_video_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    artist: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    thumbnail_url: Mapped[str | None] = mapped_column(String, nullable=True)
    audio_path: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[SongStatus] = mapped_column(
        Enum(SongStatus, name="song_status"),
        nullable=False,
        default=SongStatus.pending,
    )

    # Set to True when the user signals they want full pipeline processing
    # (queue lock, or manual /analyze /separate /transcribe API call). Each
    # worker checks this on success and only auto-dispatches the next stage
    # if True. Library-added songs default to False so they stop at
    # `downloaded` until queued.
    pipeline_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
