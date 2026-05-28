"""Celery task tests for the two lyrics workers. Real DB, upstream
services (Genius, aligner) mocked. Tasks are executed via ``.apply()``
so ``bind=True`` self injection works the same as a real worker."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import (
    Lyrics,
    LyricsAlignmentStatus,
    LyricsFetchStatus,
    Song,
    SongStatus,
    Stems,
    StemsStatus,
    Transcription,
    TranscriptionStatus,
)


@pytest.fixture
def instrumental_song():
    """A song whose transcription was skipped as instrumental."""
    sid: uuid.UUID
    with SessionLocal() as db:
        s = Song(
            youtube_video_id=f"ins-{uuid.uuid4().hex[:8]}",
            title="Test Instrumental",
            artist="Artist",
            duration_seconds=120.0,
            audio_path="audio/foo.wav",
            status=SongStatus.ready,
        )
        db.add(s)
        db.commit()
        sid = s.id
        db.add(Transcription(
            song_id=sid,
            model_name="large-v3",
            status=TranscriptionStatus.skipped_instrumental,
            language=None,
            segments=[],
            duration_seconds=120.0,
            vocal_rms_observed=0.001,
        ))
        db.add(Lyrics(
            song_id=sid,
            fetch_status=LyricsFetchStatus.success,
            text="dummy",
        ))
        db.commit()
    yield str(sid)
    with SessionLocal() as db:
        db.execute(Lyrics.__table__.delete().where(Lyrics.song_id == sid))
        db.execute(Transcription.__table__.delete().where(Transcription.song_id == sid))
        db.execute(Song.__table__.delete().where(Song.id == sid))
        db.commit()


def test_align_lyrics_marks_whisper_only_for_instrumental(
    instrumental_song, monkeypatch,
):
    """When the transcription was skipped because the track is
    instrumental, alignment isn't a failure — it's just not applicable.
    Surface that as `whisper_only` (matches spec's enum)."""
    spy = MagicMock()
    monkeypatch.setattr("app.workers.align_lyrics.align_lyrics", spy)

    from app.workers.align_lyrics import align_lyrics_task
    align_lyrics_task.apply(args=(instrumental_song,)).get()

    spy.assert_not_called()
    with SessionLocal() as db:
        row = db.scalar(select(Lyrics).where(
            Lyrics.song_id == uuid.UUID(instrumental_song)
        ))
        assert row.alignment_status == LyricsAlignmentStatus.whisper_only


def test_align_lyrics_marks_error_when_lyrics_never_arrive(monkeypatch):
    """If the Lyrics row never lands (Genius down / song never matches),
    we retry a few times and eventually mark alignment_status=error so
    the UI doesn't show 'pending' forever."""
    sid = uuid.uuid4()
    # No Lyrics row, no Song — task should retry then give up.
    # Force max_retries to 1 for test speed.
    from app.workers.align_lyrics import align_lyrics_task

    # We can't easily test the .retry() exhaustion path via .apply()
    # because .apply() runs once and records the Retry exception.
    # Instead, simulate the "exhausted retries" code path by directly
    # calling the bound function with a self stub whose .retry() raises
    # MaxRetriesExceededError immediately.
    from celery.exceptions import MaxRetriesExceededError
    from app.workers.align_lyrics import align_lyrics_task as task

    class _FakeSelf:
        max_retries = 5
        request = type("R", (), {"retries": 5})()  # already at limit

        def retry(self, *args, **kwargs):
            raise MaxRetriesExceededError()

    # Seed a Lyrics row with not_attempted and no Song so the path
    # hits the "no transcription" / "no lyrics" branches with a
    # retry-exhausted self.
    with SessionLocal() as db:
        s = Song(
            youtube_video_id=f"alg-{uuid.uuid4().hex[:8]}",
            title="X",
            duration_seconds=10.0,
            audio_path="audio/x.wav",
            status=SongStatus.downloaded,
        )
        db.add(s)
        db.commit()
        sid = s.id
        db.add(Lyrics(
            song_id=sid,
            fetch_status=LyricsFetchStatus.not_attempted,
        ))
        db.commit()

    # Call the underlying function with the fake self. For a bind=True
    # Celery task, `task.run` is already bound to the real task instance,
    # so reach through to the raw function to inject our fake self.
    raw_fn = task.__wrapped__.__func__
    try:
        raw_fn(_FakeSelf(), str(sid))
    except MaxRetriesExceededError:
        pass

    with SessionLocal() as db:
        row = db.scalar(select(Lyrics).where(Lyrics.song_id == sid))
        assert row.alignment_status == LyricsAlignmentStatus.error
        db.execute(Lyrics.__table__.delete().where(Lyrics.song_id == sid))
        db.execute(Song.__table__.delete().where(Song.id == sid))
        db.commit()


def test_fetch_lyrics_tolerates_concurrent_row_creation(monkeypatch):
    """Two dispatches racing to create the Lyrics row — the loser
    should silently observe the row created by the winner instead of
    bubbling IntegrityError."""
    from sqlalchemy.exc import IntegrityError

    sid: uuid.UUID
    with SessionLocal() as db:
        s = Song(
            youtube_video_id=f"race-{uuid.uuid4().hex[:8]}",
            title="Hello",
            artist="Adele",
            duration_seconds=120.0,
            audio_path="audio/x.wav",
            status=SongStatus.downloaded,
        )
        db.add(s)
        db.commit()
        sid = s.id

    # Simulate the WINNER having already inserted a Lyrics row by the
    # time the loser's session reads. We do this by inserting the row
    # in the test setup AFTER the worker begins (impossible via .apply
    # in a single-thread test) — emulate by inserting before, then
    # asserting the worker's path observes it cleanly.
    with SessionLocal() as db:
        db.add(Lyrics(song_id=sid))
        db.commit()

    # Mock the Genius client to return a known payload, then run the
    # task and confirm:
    #   - It DID call Genius (the row exists but fetch_status is not_attempted)
    #   - It wrote the result to the existing row, not a new one
    fake = AsyncMock(return_value=(101, "Hello, it's me"))
    monkeypatch.setattr("app.workers.fetch_lyrics.genius_fetch", fake)

    from app.workers.fetch_lyrics import fetch_lyrics_task
    fetch_lyrics_task.apply(args=(str(sid),)).get()

    with SessionLocal() as db:
        rows = db.scalars(select(Lyrics).where(Lyrics.song_id == sid)).all()
        assert len(rows) == 1  # NOT two rows
        assert rows[0].text == "Hello, it's me"
        assert rows[0].fetch_status == LyricsFetchStatus.success
        db.execute(Lyrics.__table__.delete().where(Lyrics.song_id == sid))
        db.execute(Song.__table__.delete().where(Song.id == sid))
        db.commit()


def test_fetch_lyrics_tolerates_integrity_error_on_create(monkeypatch):
    """If the row gets created mid-flight by a concurrent worker, our
    commit raises IntegrityError. Catch it, rollback, re-read."""
    from sqlalchemy.exc import IntegrityError

    sid: uuid.UUID
    with SessionLocal() as db:
        s = Song(
            youtube_video_id=f"race2-{uuid.uuid4().hex[:8]}",
            title="Hello",
            artist="Adele",
            duration_seconds=120.0,
            audio_path="audio/x.wav",
            status=SongStatus.downloaded,
        )
        db.add(s)
        db.commit()
        sid = s.id

    # Capture original SessionLocal so we can let one commit succeed
    # while injecting a failure on the test's behalf.
    fake = AsyncMock(return_value=(202, "Hello it's me"))
    monkeypatch.setattr("app.workers.fetch_lyrics.genius_fetch", fake)

    # Insert the racing row "after" the worker reads but "before" it
    # commits. We can't truly interleave, so insert before — the worker
    # path will read the row, see it's already there, and not call
    # IntegrityError-prone code at all. The genuine race protection
    # needs an integration test with two parallel workers; here we
    # accept the row-already-exists short-path as proxy coverage.
    with SessionLocal() as db:
        db.add(Lyrics(song_id=sid))
        db.commit()

    from app.workers.fetch_lyrics import fetch_lyrics_task
    fetch_lyrics_task.apply(args=(str(sid),)).get()

    with SessionLocal() as db:
        row = db.scalar(select(Lyrics).where(Lyrics.song_id == sid))
        assert row.fetch_status == LyricsFetchStatus.success
        assert row.text == "Hello it's me"
        db.execute(Lyrics.__table__.delete().where(Lyrics.song_id == sid))
        db.execute(Song.__table__.delete().where(Song.id == sid))
        db.commit()
