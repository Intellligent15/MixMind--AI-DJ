import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, String, func
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
