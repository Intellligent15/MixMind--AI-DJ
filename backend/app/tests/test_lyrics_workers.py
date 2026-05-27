"""Celery task tests for the two lyrics workers — fetch_lyrics (Genius
download) and align_lyrics (DTW alignment against Whisper). Real DB,
upstream services (Genius, aligner) mocked.

Tasks are executed via ``.apply()`` so ``bind=True`` self injection
works the same way it does in a real worker, and ``self.retry()`` is
caught by Celery's EagerResult rather than aborting the test."""

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


# --- Fixtures -----------------------------------------------------------


@pytest.fixture
def downloaded_song():
    """A song with a title + artist ready for Genius lookup. No
    transcription yet — the fetch path doesn't need one."""
    sid: uuid.UUID
    with SessionLocal() as db:
        s = Song(
            youtube_video_id=f"lyr-{id(object())}",
            title="Get Lucky",
            artist="Daft Punk",
            duration_seconds=180.0,
            audio_path="audio/foo.wav",
            status=SongStatus.downloaded,
        )
        db.add(s)
        db.commit()
        sid = s.id
    yield str(sid)
    with SessionLocal() as db:
        db.execute(Lyrics.__table__.delete().where(Lyrics.song_id == sid))
        db.execute(Song.__table__.delete().where(Song.id == sid))
        db.commit()


@pytest.fixture
def song_with_transcription_and_lyrics():
    """A ready song with a Whisper transcription, plus a Lyrics row
    whose fetch already landed (status=success). The aligner is the
    next step — its task is what's under test here."""
    sid: uuid.UUID
    with SessionLocal() as db:
        s = Song(
            youtube_video_id=f"ali-{id(object())}",
            title="Love Me Tender",
            artist="Elvis",
            duration_seconds=120.0,
            audio_path="audio/elvis.wav",
            status=SongStatus.ready,
        )
        db.add(s)
        db.commit()
        sid = s.id
        db.add(Transcription(
            song_id=sid,
            model_name="large-v3",
            status=TranscriptionStatus.success,
            language="en",
            segments=[{
                "start": 0.0, "end": 2.0, "text": "love me tender",
                "avg_logprob": -0.2, "no_speech_prob": 0.01,
                "compression_ratio": 1.5, "temperature": 0.0,
                "words": [
                    {"word": "love", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": "me", "start": 0.6, "end": 0.8, "probability": 0.9},
                    {"word": "tender", "start": 1.0, "end": 1.5, "probability": 0.9},
                ],
            }],
            duration_seconds=120.0,
        ))
        db.add(Lyrics(
            song_id=sid,
            genius_id=42,
            text="love me tender",
            fetch_status=LyricsFetchStatus.success,
            alignment_status=LyricsAlignmentStatus.not_attempted,
        ))
        db.commit()
    yield str(sid)
    with SessionLocal() as db:
        db.execute(Lyrics.__table__.delete().where(Lyrics.song_id == sid))
        db.execute(Transcription.__table__.delete().where(Transcription.song_id == sid))
        db.execute(Song.__table__.delete().where(Song.id == sid))
        db.commit()


# --- fetch_lyrics_task -------------------------------------------------


def test_fetch_lyrics_writes_text_and_genius_id(downloaded_song, monkeypatch):
    fake = AsyncMock(return_value=(12345, "Love me tender, love me sweet"))
    monkeypatch.setattr("app.workers.fetch_lyrics.genius_fetch", fake)

    from app.workers.fetch_lyrics import fetch_lyrics_task
    result = fetch_lyrics_task.apply(args=(downloaded_song,)).get()

    assert result == downloaded_song
    fake.assert_called_once_with("Get Lucky", "Daft Punk")

    with SessionLocal() as db:
        row = db.scalar(
            select(Lyrics).where(Lyrics.song_id == uuid.UUID(downloaded_song))
        )
        assert row is not None
        assert row.genius_id == 12345
        assert row.text == "Love me tender, love me sweet"
        assert row.fetch_status == LyricsFetchStatus.success


def test_fetch_lyrics_marks_not_found_when_genius_returns_none(
    downloaded_song, monkeypatch,
):
    monkeypatch.setattr(
        "app.workers.fetch_lyrics.genius_fetch",
        AsyncMock(return_value=None),
    )

    from app.workers.fetch_lyrics import fetch_lyrics_task
    fetch_lyrics_task.apply(args=(downloaded_song,)).get()

    with SessionLocal() as db:
        row = db.scalar(
            select(Lyrics).where(Lyrics.song_id == uuid.UUID(downloaded_song))
        )
        assert row is not None
        assert row.text is None
        assert row.genius_id is None
        assert row.fetch_status == LyricsFetchStatus.not_found


def test_fetch_lyrics_skips_when_already_success(downloaded_song, monkeypatch):
    # Pre-seed Lyrics row with fetch_status=success.
    sid = uuid.UUID(downloaded_song)
    with SessionLocal() as db:
        db.add(Lyrics(
            song_id=sid,
            genius_id=99,
            text="already there",
            fetch_status=LyricsFetchStatus.success,
        ))
        db.commit()

    fake = AsyncMock(return_value=(1, "fresh"))
    monkeypatch.setattr("app.workers.fetch_lyrics.genius_fetch", fake)

    from app.workers.fetch_lyrics import fetch_lyrics_task
    fetch_lyrics_task.apply(args=(downloaded_song,)).get()

    fake.assert_not_called()  # short-circuit before Genius call
    with SessionLocal() as db:
        row = db.scalar(select(Lyrics).where(Lyrics.song_id == sid))
        assert row.text == "already there"  # unchanged


def test_fetch_lyrics_no_op_when_song_missing(monkeypatch):
    """Bogus song id — task should return cleanly without raising."""
    fake = AsyncMock(return_value=None)
    monkeypatch.setattr("app.workers.fetch_lyrics.genius_fetch", fake)

    from app.workers.fetch_lyrics import fetch_lyrics_task
    result = fetch_lyrics_task.apply(args=(str(uuid.uuid4()),)).get()
    assert result is None
    fake.assert_not_called()


# --- align_lyrics_task -------------------------------------------------


def test_align_lyrics_writes_aligned_words_and_quality(
    song_with_transcription_and_lyrics, monkeypatch,
):
    from app.workers.align_lyrics import align_lyrics_task

    aligned_payload = {
        "aligned_words": [
            {"word": "love", "start": 0.0, "end": 0.5, "confidence": 0.9, "source": "whisper_match"},
            {"word": "me", "start": 0.6, "end": 0.8, "confidence": 0.9, "source": "whisper_match"},
            {"word": "tender", "start": 1.0, "end": 1.5, "confidence": 0.9, "source": "whisper_match"},
        ],
        "alignment_quality": 1.0,
        "alignment_status": LyricsAlignmentStatus.success,
    }
    monkeypatch.setattr(
        "app.workers.align_lyrics.align_lyrics",
        MagicMock(return_value=aligned_payload),
    )

    result = align_lyrics_task.apply(args=(song_with_transcription_and_lyrics,)).get()
    assert result == song_with_transcription_and_lyrics

    with SessionLocal() as db:
        row = db.scalar(select(Lyrics).where(
            Lyrics.song_id == uuid.UUID(song_with_transcription_and_lyrics)
        ))
        assert row.alignment_status == LyricsAlignmentStatus.success
        assert row.alignment_quality == 1.0
        assert row.aligned_words is not None
        assert len(row.aligned_words) == 3


def test_align_lyrics_short_circuits_when_already_done(
    song_with_transcription_and_lyrics, monkeypatch,
):
    sid = uuid.UUID(song_with_transcription_and_lyrics)
    # Seed alignment_status to success up-front.
    with SessionLocal() as db:
        lyr = db.scalar(select(Lyrics).where(Lyrics.song_id == sid))
        lyr.alignment_status = LyricsAlignmentStatus.success
        lyr.alignment_quality = 0.9
        db.commit()

    spy = MagicMock()
    monkeypatch.setattr("app.workers.align_lyrics.align_lyrics", spy)

    from app.workers.align_lyrics import align_lyrics_task
    align_lyrics_task.apply(args=(song_with_transcription_and_lyrics,)).get()

    spy.assert_not_called()


def test_align_lyrics_marks_error_when_fetch_failed(
    song_with_transcription_and_lyrics, monkeypatch,
):
    # Flip fetch_status to not_found so the aligner shouldn't even
    # be invoked — we expect alignment_status=error.
    sid = uuid.UUID(song_with_transcription_and_lyrics)
    with SessionLocal() as db:
        lyr = db.scalar(select(Lyrics).where(Lyrics.song_id == sid))
        lyr.fetch_status = LyricsFetchStatus.not_found
        db.commit()

    spy = MagicMock()
    monkeypatch.setattr("app.workers.align_lyrics.align_lyrics", spy)

    from app.workers.align_lyrics import align_lyrics_task
    align_lyrics_task.apply(args=(song_with_transcription_and_lyrics,)).get()

    spy.assert_not_called()
    with SessionLocal() as db:
        row = db.scalar(select(Lyrics).where(Lyrics.song_id == sid))
        assert row.alignment_status == LyricsAlignmentStatus.error


def test_align_lyrics_skips_when_no_transcription(downloaded_song, monkeypatch):
    """Lyrics row exists with successful fetch but no Transcription —
    early-return without crashing or invoking the aligner."""
    sid = uuid.UUID(downloaded_song)
    with SessionLocal() as db:
        db.add(Lyrics(
            song_id=sid,
            genius_id=1,
            text="something",
            fetch_status=LyricsFetchStatus.success,
        ))
        db.commit()

    spy = MagicMock()
    monkeypatch.setattr("app.workers.align_lyrics.align_lyrics", spy)

    from app.workers.align_lyrics import align_lyrics_task
    align_lyrics_task.apply(args=(downloaded_song,)).get()
    spy.assert_not_called()
