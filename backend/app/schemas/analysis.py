from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SectionSchema(BaseModel):
    start: float
    end: float
    label: str


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    song_id: uuid.UUID
    bpm: float
    key: str
    camelot_key: str
    time_signature: int
    beat_grid: list[float]
    downbeats: list[float]
    sections: list[SectionSchema]
    energy_curve: list[float]
    vocal_segments: list[list[float]]
    created_at: datetime
    updated_at: datetime
