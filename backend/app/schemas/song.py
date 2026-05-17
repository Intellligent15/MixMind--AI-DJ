from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.song import SongStatus


class SearchResultSchema(BaseModel):
    youtube_video_id: str
    title: str
    artist: str | None = None
    duration_seconds: float
    thumbnail_url: str | None = None


class SongCreate(BaseModel):
    youtube_video_id: str
    title: str
    artist: str | None = None
    duration_seconds: float = Field(ge=0)
    thumbnail_url: str | None = None


class SongRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    youtube_video_id: str
    title: str
    artist: str | None
    duration_seconds: float
    thumbnail_url: str | None
    audio_path: str | None
    status: SongStatus
    created_at: datetime
    updated_at: datetime
