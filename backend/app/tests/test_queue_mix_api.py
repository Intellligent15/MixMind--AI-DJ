"""Tests for the Phase 8 queue-mix HTTP surface: kick off a stitch,
read the QueueRender row, stream the rendered FLAC."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.main import app
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


def _client(db_session: Session) -> TestClient:
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def _make_song(db: Session, vid: str) -> Song:
    s = Song(
        youtube_video_id=vid,
        title=f"song-{vid}",
        duration_seconds=180.0,
        audio_path=f"audio/{vid}.wav",
        status=SongStatus.ready,
    )
    db.add(s)
    db.flush()
    return s


def _seed_locked_queue(db: Session, n_songs: int = 2) -> Queue:
    q = Queue(locked=True)
    db.add(q)
    db.flush()
    songs = [_make_song(db, f"qm-{i}-{uuid.uuid4().hex[:8]}") for i in range(n_songs)]
    for i, s in enumerate(songs):
        db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
    for i in range(n_songs - 1):
        db.add(MixPlan(
            queue_id=q.id,
            from_song_id=songs[i].id,
            to_song_id=songs[i + 1].id,
            status=MixPlanStatus.ready,
            rendered_audio_path=f"mixes/m{i}.wav",
            plan_json=[],
        ))
    db.flush()
    return q


# --- POST /api/queues/{id}/stitch --------------------------------------


def test_stitch_404_when_queue_missing(db_session: Session):
    client = _client(db_session)
    r = client.post(f"/api/queues/{uuid.uuid4()}/stitch")
    assert r.status_code == 404


def test_stitch_409_when_queue_not_locked(db_session: Session):
    q = Queue(locked=False)
    db_session.add(q)
    db_session.flush()
    client = _client(db_session)
    r = client.post(f"/api/queues/{q.id}/stitch")
    assert r.status_code == 409


def test_stitch_dispatches_stitch_task_when_all_mixes_ready(db_session: Session):
    q = _seed_locked_queue(db_session, n_songs=2)
    client = _client(db_session)

    with patch("app.api.queues._TaskShim") as shim_cls:
        shim_cls.return_value = MagicMock()
        r = client.post(f"/api/queues/{q.id}/stitch")

    assert r.status_code == 202
    # One _TaskShim instantiation for the stitch task itself; no
    # render_transition tasks because all mixes are ready.
    instantiations = [c.args for c in shim_cls.call_args_list]
    assert ("app.workers.stitch_queue.stitch_queue",) in instantiations


def test_stitch_chords_renders_when_some_mixes_pending(db_session: Session):
    q = Queue(locked=True)
    db_session.add(q)
    db_session.flush()
    songs = [_make_song(db_session, f"qm-{i}-{uuid.uuid4().hex[:8]}") for i in range(3)]
    for i, s in enumerate(songs):
        db_session.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
    # First mix is ready, second is still pending — the chord should
    # render the pending one before stitching.
    db_session.add(MixPlan(
        queue_id=q.id,
        from_song_id=songs[0].id, to_song_id=songs[1].id,
        status=MixPlanStatus.ready,
        rendered_audio_path="mixes/ready.wav", plan_json=[],
    ))
    db_session.add(MixPlan(
        queue_id=q.id,
        from_song_id=songs[1].id, to_song_id=songs[2].id,
        status=MixPlanStatus.pending,
        plan_json=None,
    ))
    db_session.flush()
    client = _client(db_session)

    with patch("app.api.queues._TaskShim") as shim_cls, \
         patch("celery.chord") as chord_mock:
        shim_cls.return_value = MagicMock()
        r = client.post(f"/api/queues/{q.id}/stitch")

    assert r.status_code == 202
    # chord(render_tasks)(stitch_task)
    chord_mock.assert_called_once()


def test_stitch_creates_queue_render_row_if_missing(db_session: Session):
    q = _seed_locked_queue(db_session, n_songs=2)
    client = _client(db_session)

    with patch("app.api.queues._TaskShim") as shim_cls:
        shim_cls.return_value = MagicMock()
        client.post(f"/api/queues/{q.id}/stitch")

    row = db_session.scalar(select(QueueRender).where(QueueRender.queue_id == q.id))
    assert row is not None
    assert row.status == QueueRenderStatus.pending
    assert row.error_text is None


def test_stitch_resets_existing_failed_row_to_pending(db_session: Session):
    q = _seed_locked_queue(db_session, n_songs=2)
    db_session.add(QueueRender(
        queue_id=q.id,
        status=QueueRenderStatus.failed,
        error_text="earlier error",
    ))
    db_session.flush()
    client = _client(db_session)

    with patch("app.api.queues._TaskShim") as shim_cls:
        shim_cls.return_value = MagicMock()
        client.post(f"/api/queues/{q.id}/stitch")

    db_session.expire_all()
    row = db_session.scalar(select(QueueRender).where(QueueRender.queue_id == q.id))
    assert row.status == QueueRenderStatus.pending
    assert row.error_text is None


# --- GET /api/queues/{id}/mix ------------------------------------------


def test_get_mix_404_when_missing(db_session: Session):
    q = Queue(locked=True)
    db_session.add(q)
    db_session.flush()
    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix")
    assert r.status_code == 404


def test_get_mix_returns_row(db_session: Session):
    q = Queue(locked=True)
    db_session.add(q)
    db_session.flush()
    db_session.add(QueueRender(
        queue_id=q.id,
        status=QueueRenderStatus.ready,
        rendered_audio_path="queue_mixes/x.flac",
    ))
    db_session.flush()
    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["rendered_audio_path"] == "queue_mixes/x.flac"


# --- GET /api/queues/{id}/mix/audio -------------------------------------


def test_get_mix_audio_404_when_not_ready(db_session: Session):
    q = Queue(locked=True)
    db_session.add(q)
    db_session.flush()
    db_session.add(QueueRender(
        queue_id=q.id,
        status=QueueRenderStatus.rendering,
    ))
    db_session.flush()
    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix/audio")
    assert r.status_code == 404


def test_get_mix_audio_404_when_no_render_row(db_session: Session):
    q = Queue(locked=True)
    db_session.add(q)
    db_session.flush()
    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix/audio")
    assert r.status_code == 404


def test_get_mix_audio_streams_flac_with_download_disposition(
    db_session: Session,
):
    """When the mix is ready, the route streams the FLAC bytes through
    the storage backend with a Content-Disposition: attachment header."""
    q = Queue(locked=True)
    db_session.add(q)
    db_session.flush()
    key = f"queue_mixes/{q.id}.flac"
    db_session.add(QueueRender(
        queue_id=q.id,
        status=QueueRenderStatus.ready,
        rendered_audio_path=key,
    ))
    db_session.flush()

    # Put a tiny placeholder file at the LocalFilesystemStorage root
    # (conftest pins storage to LocalFilesystemStorage under a tmp dir).
    import asyncio
    from app.services.storage import get_storage as _get_storage
    asyncio.run(_get_storage().write(key, b"fLaC-placeholder-bytes"))

    client = _client(db_session)
    r = client.get(f"/api/queues/{q.id}/mix/audio")
    assert r.status_code == 200
    assert r.content == b"fLaC-placeholder-bytes"
    cd = r.headers.get("content-disposition", "")
    assert "attachment" in cd
    assert "mix.flac" in cd
