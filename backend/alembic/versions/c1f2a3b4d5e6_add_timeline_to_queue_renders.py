"""add timeline to queue_renders

Revision ID: c1f2a3b4d5e6
Revises: 76d2da8e5066
Create Date: 2026-06-07 00:00:00.000000

Phase 10: additive JSONB column holding the stitched-mix output timeline
(per-song + per-transition time spans) consumed by the player's transition
indicator. Nullable — existing QueueRender rows read as null and are
re-stitched to populate.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c1f2a3b4d5e6"
down_revision: Union[str, None] = "76d2da8e5066"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "queue_renders",
        sa.Column("timeline", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("queue_renders", "timeline")
