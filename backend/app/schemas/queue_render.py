import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict

from app.models.queue_render import QueueRenderStatus


class QueueRenderBase(BaseModel):
    queue_id: uuid.UUID
    status: QueueRenderStatus
    rendered_audio_path: str | None
    error_text: str | None


class QueueRenderRead(QueueRenderBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    # Phase 10 player timeline (see QueueRender.timeline). Null until stitched.
    timeline: dict[str, Any] | None = None

    model_config = ConfigDict(from_attributes=True)
