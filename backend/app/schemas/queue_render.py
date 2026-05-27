import uuid
from datetime import datetime
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

    model_config = ConfigDict(from_attributes=True)
