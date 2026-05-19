"""Test fixtures.

DB-touching tests use the dev Postgres (the one docker compose brings up).
Each test runs inside a transaction that is rolled back at teardown so
in-test writes don't persist. We also TRUNCATE the small set of tables
that the queue/songs API reads at fixture setup — without this, rows left
over from interactive dev (or earlier test runs that committed outside
the fixture) leak into list/count assertions. The whole arrangement can
be swapped for testcontainers in a later polish phase.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.db import SessionLocal, engine
from app.models import Song  # noqa: F401 — ensure Song table is registered


# Tables to wipe before each db_session test. Order doesn't matter — TRUNCATE
# ... CASCADE handles the FKs. queue_items and queues are listed for clarity
# even though the CASCADE from songs would catch them via queue_items.song_id.
_TRUNCATE_TABLES = (
    "queue_items",
    "queues",
    "transcriptions",
    "stems",
    "analyses",
    "songs",
)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    connection = engine.connect()
    # Truncate outside the rollback envelope so the wipe persists even when
    # tests don't commit. RESTART IDENTITY isn't needed (all our PKs are
    # UUIDs) but CASCADE is — queue_items → songs FK would otherwise block.
    connection.execute(
        text(f"TRUNCATE TABLE {', '.join(_TRUNCATE_TABLES)} CASCADE")
    )
    connection.commit()

    trans = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()
