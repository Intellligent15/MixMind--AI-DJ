"""Phase 10 eager-stitch dispatcher tests.

Covers the precondition gating in auto_stitch.maybe_dispatch_stitch — the
hazard the brief calls out is firing the stitch chord before every pair's
songs are ``ready`` (every render would no-op). These assert the dispatcher
only fires when it's safe and never twice.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    QueueRender,
    QueueRenderStatus,
    Song,
    SongStatus,
)
from app.workers.auto_stitch import maybe_dispatch_stitch


def _song(db: Session, status: SongStatus) -> Song:
    s = Song(
        youtube_video_id=f"as-{uuid.uuid4().hex[:10]}",
        title="t",
        duration_seconds=180.0,
        audio_path="audio/x.wav",
        status=status,
    )
    db.add(s)
    db.flush()
    return s


def _locked_queue(db: Session, statuses: list[SongStatus]) -> Queue:
    q = Queue(locked=True)
    db.add(q)
    db.flush()
    songs = [_song(db, st) for st in statuses]
    for i, s in enumerate(songs):
        db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
    for i in range(len(songs) - 1):
        db.add(MixPlan(
            queue_id=q.id,
            from_song_id=songs[i].id,
            to_song_id=songs[i + 1].id,
            status=MixPlanStatus.pending,
        ))
    db.flush()
    return q


def test_no_dispatch_when_a_song_not_ready(db_session: Session):
    q = _locked_queue(db_session, [SongStatus.ready, SongStatus.separating])
    with patch("app.workers.auto_stitch.celery_app") as celery, \
            patch("app.workers.auto_stitch.chord"):
        assert maybe_dispatch_stitch(q.id, db_session) is False
    celery.signature.assert_not_called()
    assert db_session.scalar(
        select(QueueRender).where(QueueRender.queue_id == q.id)
    ) is None


def test_dispatch_when_all_ready_creates_pending_row(db_session: Session):
    q = _locked_queue(db_session, [SongStatus.ready, SongStatus.ready])
    with patch("app.workers.auto_stitch.celery_app") as celery, \
            patch("app.workers.auto_stitch.chord") as chord_mock:
        assert maybe_dispatch_stitch(q.id, db_session) is True
    # 1 pair, still pending → render chord fans out before the stitch callback.
    chord_mock.assert_called_once()
    row = db_session.scalar(select(QueueRender).where(QueueRender.queue_id == q.id))
    assert row is not None and row.status == QueueRenderStatus.pending


def test_no_double_dispatch_when_already_in_flight(db_session: Session):
    q = _locked_queue(db_session, [SongStatus.ready, SongStatus.ready])
    db_session.add(QueueRender(queue_id=q.id, status=QueueRenderStatus.rendering))
    db_session.flush()
    with patch("app.workers.auto_stitch.celery_app") as celery, \
            patch("app.workers.auto_stitch.chord"):
        assert maybe_dispatch_stitch(q.id, db_session) is False
    celery.signature.assert_not_called()


def test_no_dispatch_for_single_song_queue(db_session: Session):
    q = _locked_queue(db_session, [SongStatus.ready])
    with patch("app.workers.auto_stitch.celery_app") as celery, \
            patch("app.workers.auto_stitch.chord"):
        assert maybe_dispatch_stitch(q.id, db_session) is False
    celery.signature.assert_not_called()


def test_no_dispatch_when_queue_unlocked(db_session: Session):
    q = _locked_queue(db_session, [SongStatus.ready, SongStatus.ready])
    q.locked = False
    db_session.flush()
    with patch("app.workers.auto_stitch.celery_app") as celery, \
            patch("app.workers.auto_stitch.chord"):
        assert maybe_dispatch_stitch(q.id, db_session) is False
    celery.signature.assert_not_called()
