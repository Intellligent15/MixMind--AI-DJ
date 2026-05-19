"""add vocal_envelope_path to stems

Revision ID: db729e2f9c53
Revises: 4065b47d2200
Create Date: 2026-05-19 00:00:00.000000

Additive column for the frame-wise vocal envelope sidecar (10 Hz RMS+peak
JSON written by separate_stems). Nullable: existing Stems rows stay valid
and read as null — the user will re-separate to populate them.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "db729e2f9c53"
down_revision: Union[str, None] = "4065b47d2200"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "stems",
        sa.Column("vocal_envelope_path", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("stems", "vocal_envelope_path")
