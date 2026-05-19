import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class MixPlanStatus(str, enum.Enum):
    pending = "pending"
    rendering = "rendering"
    ready = "ready"
    failed = "failed"


class MixPlan(Base):
    __tablename__ = "mix_plans"
    __table_args__ = (
        UniqueConstraint(
            "queue_id", "from_song_id", "to_song_id", name="uq_mix_plan_pair"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    queue_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queues.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        nullable=False,
    )
    to_song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Null at lock time (generated lazily at render time so Phase 9's LLM
    # call doesn't fire for plans the user never asks to render). Filled
    # with the tool-call list before the executor walks it.
    plan_json: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)

    # StorageBackend key, e.g. "mixes/<mix_plan_id>.wav". Null until rendered.
    rendered_audio_path: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[MixPlanStatus] = mapped_column(
        Enum(MixPlanStatus, name="mix_plan_status"),
        nullable=False,
        default=MixPlanStatus.pending,
    )

    # Populated when status=failed so the debug UI can show the user *why*
    # without asking them to scrape worker logs.
    error_text: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
