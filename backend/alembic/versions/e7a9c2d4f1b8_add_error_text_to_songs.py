"""add error_text to songs

Revision ID: e7a9c2d4f1b8
Revises: c1f2a3b4d5e6
Create Date: 2026-06-08 00:00:00.000000

Phase 11: additive nullable column holding the reason a song last failed,
so the Processing view can surface per-song failures (and offer a retry)
instead of leaving a silent stall. Mirrors MixPlan.error_text.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e7a9c2d4f1b8"
down_revision: Union[str, None] = "c1f2a3b4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "songs",
        sa.Column("error_text", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("songs", "error_text")
