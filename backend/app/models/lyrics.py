import enum
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, String, Enum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.song import Song


class LyricsFetchStatus(str, enum.Enum):
    not_attempted = "not_attempted"
    success = "success"
    not_found = "not_found"
    error = "error"


class LyricsAlignmentStatus(str, enum.Enum):
    not_attempted = "not_attempted"
    success = "success"
    whisper_only = "whisper_only"
    low_quality = "low_quality"
    error = "error"


class Lyrics(Base):
    __tablename__ = "lyrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    
    genius_id: Mapped[int | None] = mapped_column(nullable=True)
    text: Mapped[str | None] = mapped_column(String, nullable=True)
    fetch_status: Mapped[LyricsFetchStatus] = mapped_column(
        Enum(LyricsFetchStatus, name="lyricsfetchstatus"),
        nullable=False,
        default=LyricsFetchStatus.not_attempted,
    )

    aligned_words: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    alignment_status: Mapped[LyricsAlignmentStatus] = mapped_column(
        Enum(LyricsAlignmentStatus, name="lyricsalignmentstatus"),
        nullable=False,
        default=LyricsAlignmentStatus.not_attempted,
    )
    alignment_quality: Mapped[float | None] = mapped_column(Float, nullable=True)

    song: Mapped["Song"] = relationship("Song")
