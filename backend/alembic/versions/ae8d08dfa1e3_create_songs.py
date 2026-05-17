"""create songs

Revision ID: ae8d08dfa1e3
Revises:
Create Date: 2026-05-16 23:25:44.229152

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "ae8d08dfa1e3"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


song_status = sa.Enum(
    "pending",
    "downloading",
    "downloaded",
    "analyzing",
    "analyzed",
    "separating",
    "transcribing",
    "ready",
    "failed",
    name="song_status",
)


def upgrade() -> None:
    op.create_table(
        "songs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("youtube_video_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("artist", sa.String(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("thumbnail_url", sa.String(), nullable=True),
        sa.Column("audio_path", sa.String(), nullable=True),
        sa.Column("status", song_status, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_accessed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("youtube_video_id"),
    )


def downgrade() -> None:
    op.drop_table("songs")
    song_status.drop(op.get_bind(), checkfirst=False)
