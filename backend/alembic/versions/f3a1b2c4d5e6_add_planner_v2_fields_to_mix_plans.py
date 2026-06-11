"""add planner v2 fields to mix_plans

Revision ID: f3a1b2c4d5e6
Revises: e7a9c2d4f1b8
Create Date: 2026-06-09
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f3a1b2c4d5e6"
down_revision: Union[str, None] = "e7a9c2d4f1b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("mix_plans", sa.Column("plan_source", sa.String(), nullable=True))
    op.add_column("mix_plans", sa.Column("style", sa.String(), nullable=True))
    op.add_column("mix_plans", sa.Column("rationale", sa.String(), nullable=True))
    op.add_column("mix_plans", sa.Column("style_hint", sa.String(), nullable=True))
    op.add_column("mix_plans", sa.Column("style_override", sa.String(), nullable=True))
    op.add_column(
        "mix_plans",
        sa.Column("reroll_nonce", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("mix_plans", "reroll_nonce")
    op.drop_column("mix_plans", "style_override")
    op.drop_column("mix_plans", "style_hint")
    op.drop_column("mix_plans", "rationale")
    op.drop_column("mix_plans", "style")
    op.drop_column("mix_plans", "plan_source")
