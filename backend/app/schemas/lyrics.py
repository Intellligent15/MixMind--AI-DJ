import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.lyrics import LyricsAlignmentStatus, LyricsFetchStatus


class AlignedWord(BaseModel):
    word: str
    start: float
    end: float
    confidence: float
    source: str


class LyricsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    song_id: uuid.UUID
    genius_id: int | None
    text: str | None
    fetch_status: LyricsFetchStatus
    aligned_words: list[AlignedWord] | None
    alignment_status: LyricsAlignmentStatus
    alignment_quality: float | None
