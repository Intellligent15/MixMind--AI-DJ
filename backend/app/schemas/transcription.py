from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import TranscriptionStatus


class WordRead(BaseModel):
    start: float
    end: float
    word: str


class SegmentRead(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordRead] = []


class TranscriptionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    song_id: uuid.UUID
    model_name: str
    status: TranscriptionStatus
    language: str | None
    segments: list[SegmentRead]
    vocal_rms_threshold: float | None
    vocal_rms_observed: float | None
    duration_seconds: float | None
    created_at: datetime
    updated_at: datetime
