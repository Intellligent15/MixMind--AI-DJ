from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import TranscriptionStatus


class WordRead(BaseModel):
    start: float
    end: float
    word: str
    # mlx-whisper per-word probability. Input to the planned vocal-safety
    # logic (ai-dj-spec.md → Vocal Safety Model). Nullable for historical
    # rows transcribed before we started preserving it.
    probability: float | None = None


class SegmentRead(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordRead] = []
    # Per-segment confidence signals from mlx-whisper. Same nullable
    # rationale as WordRead.probability — historical rows from earlier
    # Phase 6 work won't have these fields populated.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None


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
