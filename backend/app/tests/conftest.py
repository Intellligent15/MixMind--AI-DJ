"""Test fixtures.

DB-touching tests use the dev Postgres (the one docker compose brings up).
Each test runs inside a SAVEPOINT-wrapped transaction that is rolled back at
teardown, so tests neither persist data nor depend on ordering. Testcontainers
can replace this in a later polish phase.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy.orm import Session

from app.core.db import SessionLocal, engine
from app.models import Song  # noqa: F401 — ensure Song table is registered


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    connection = engine.connect()
    trans = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()
