"""transcribe_song Celery task tests.

Same pattern as test_separate_task: real DB, clean up by song id at the
end, mock out the TranscriptionService so we don't pull down MLX weights.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.db import SessionLocal
from app.models import (
    Song,
    SongStatus,
    Stems,
    StemsStatus,
    Transcription,
    TranscriptionStatus,
)
from app.services.transcription import TranscriptionResult


@pytest.fixture
def analyzed_song_with_stems():
    """A song in `analyzed` status with a Stems row whose vocal_rms is
    well above the 0.005 default threshold — i.e. transcription will run."""
    sid_str = None
    with SessionLocal() as db:
        song = Song(
            youtube_video_id=f"trtest-{id(object())}",
            title="T",
            artist=None,
            duration_seconds=10.0,
            thumbnail_url=None,
            audio_path="audio/fake.wav",
            status=SongStatus.analyzed,
        )
        db.add(song)
        db.flush()
        stems = Stems(
            song_id=song.id,
            model_name="htdemucs",
            status=StemsStatus.separated,
            vocals_path=f"stems/{song.youtube_video_id}/vocals.wav",
            drums_path=f"stems/{song.youtube_video_id}/drums.wav",
            bass_path=f"stems/{song.youtube_video_id}/bass.wav",
            other_path=f"stems/{song.youtube_video_id}/other.wav",
            vocal_rms=0.15,  # > 0.005 threshold
        )
        db.add(stems)
        db.commit()
        sid_str = str(song.id)
    yield sid_str
    sid = uuid.UUID(sid_str)
    with SessionLocal() as db:
        # Cascade from songs.id deletes Stems + Transcription rows.
        song = db.get(Song, sid)
        if song is not None:
            db.delete(song)
        db.commit()


def _fake_result() -> TranscriptionResult:
    return TranscriptionResult(
        language="en",
        segments=[
            {
                "start": 0.0,
                "end": 1.2,
                "text": " hi",
                "words": [{"start": 0.0, "end": 1.2, "word": " hi"}],
            }
        ],
        duration_seconds=1.2,
    )


def _patched_service(result: TranscriptionResult | None = None):
    service = MagicMock()
    service.transcribe.return_value = result or _fake_result()
    service.model_name = "large-v3"
    return service


def test_transcribe_song_happy_path(analyzed_song_with_stems: str):
    service = _patched_service()
    storage = AsyncMock()
    storage.path.side_effect = lambda key: f"/tmp/{key}"

    with (
        patch("app.workers.transcribe.get_storage", return_value=storage),
        patch(
            "app.workers.transcribe.TranscriptionService", return_value=service
        ),
    ):
        from app.workers.transcribe import transcribe_song

        result = transcribe_song(analyzed_song_with_stems)

    assert result == analyzed_song_with_stems
    service.transcribe.assert_called_once()

    sid = uuid.UUID(analyzed_song_with_stems)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        assert song is not None
        # Phase 6 is the gate that promotes analyzed -> ready.
        assert song.status == SongStatus.ready

        row = (
            db.query(Transcription).filter(Transcription.song_id == sid).one()
        )
        assert row.status == TranscriptionStatus.success
        assert row.model_name == "large-v3"
        assert row.language == "en"
        assert len(row.segments) == 1
        assert row.segments[0]["text"] == " hi"
        assert row.vocal_rms_observed == pytest.approx(0.15)
        assert row.vocal_rms_threshold == pytest.approx(0.005)
        assert row.duration_seconds == pytest.approx(1.2)


def test_transcribe_song_skips_when_vocal_rms_below_threshold(
    analyzed_song_with_stems: str,
):
    sid = uuid.UUID(analyzed_song_with_stems)
    with SessionLocal() as db:
        stems = db.query(Stems).filter(Stems.song_id == sid).one()
        stems.vocal_rms = 0.001  # below 0.005
        db.commit()

    service = _patched_service()
    with (
        patch("app.workers.transcribe.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.transcribe.TranscriptionService", return_value=service
        ),
    ):
        from app.workers.transcribe import transcribe_song

        result = transcribe_song(analyzed_song_with_stems)

    assert result == analyzed_song_with_stems
    # Service was never invoked.
    service.transcribe.assert_not_called()

    with SessionLocal() as db:
        song = db.get(Song, sid)
        assert song is not None
        # Skipped-instrumental songs are still treated as ready downstream.
        assert song.status == SongStatus.ready
        row = (
            db.query(Transcription).filter(Transcription.song_id == sid).one()
        )
        assert row.status == TranscriptionStatus.skipped_instrumental
        assert row.segments == []
        assert row.language is None
        assert row.vocal_rms_observed == pytest.approx(0.001)
        assert row.duration_seconds is None


def test_transcribe_song_marks_failed_on_error(analyzed_song_with_stems: str):
    service = MagicMock()
    service.transcribe.side_effect = RuntimeError("whisper boom")
    service.model_name = "large-v3"

    with (
        patch("app.workers.transcribe.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.transcribe.TranscriptionService", return_value=service
        ),
    ):
        from app.workers.transcribe import transcribe_song

        with pytest.raises(RuntimeError, match="whisper boom"):
            transcribe_song(analyzed_song_with_stems)

    sid = uuid.UUID(analyzed_song_with_stems)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        assert song is not None
        assert song.status == SongStatus.failed
        # Error rows are persisted so the UI can show why transcription
        # failed without consulting Celery logs.
        row = (
            db.query(Transcription).filter(Transcription.song_id == sid).one()
        )
        assert row.status == TranscriptionStatus.error
        assert row.segments == []


def test_transcribe_song_fails_when_no_stems_row():
    """No Stems row -> the separate_stems chain step never ran. Mark the
    song failed and bail. No Transcription row is written (no useful
    context to record)."""
    sid_str = None
    with SessionLocal() as db:
        song = Song(
            youtube_video_id=f"trtest-no-stems-{id(object())}",
            title="T",
            artist=None,
            duration_seconds=10.0,
            thumbnail_url=None,
            audio_path="audio/fake.wav",
            status=SongStatus.analyzed,
        )
        db.add(song)
        db.commit()
        sid_str = str(song.id)

    try:
        with (
            patch("app.workers.transcribe.get_storage", return_value=AsyncMock()),
            patch(
                "app.workers.transcribe.TranscriptionService",
                return_value=_patched_service(),
            ),
        ):
            from app.workers.transcribe import transcribe_song

            result = transcribe_song(sid_str)

        assert result is None
        sid = uuid.UUID(sid_str)
        with SessionLocal() as db:
            song = db.get(Song, sid)
            assert song is not None
            assert song.status == SongStatus.failed
            assert (
                db.query(Transcription)
                .filter(Transcription.song_id == sid)
                .one_or_none()
                is None
            )
    finally:
        sid = uuid.UUID(sid_str)
        with SessionLocal() as db:
            song = db.get(Song, sid)
            if song is not None:
                db.delete(song)
            db.commit()


def test_transcribe_song_skips_wrong_status(analyzed_song_with_stems: str):
    """A song still downloading -> atomic claim refuses, transcribe is a no-op."""
    sid = uuid.UUID(analyzed_song_with_stems)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        song.status = SongStatus.downloading
        db.commit()

    service = _patched_service()
    with (
        patch("app.workers.transcribe.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.transcribe.TranscriptionService", return_value=service
        ),
    ):
        from app.workers.transcribe import transcribe_song

        result = transcribe_song(analyzed_song_with_stems)

    assert result is None
    service.transcribe.assert_not_called()


def test_transcribe_song_skips_if_already_transcribing(
    analyzed_song_with_stems: str,
):
    """Concurrent dispatch: the loser sees status=transcribing and bails."""
    sid = uuid.UUID(analyzed_song_with_stems)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        song.status = SongStatus.transcribing
        db.commit()

    service = _patched_service()
    with (
        patch("app.workers.transcribe.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.transcribe.TranscriptionService", return_value=service
        ),
    ):
        from app.workers.transcribe import transcribe_song

        result = transcribe_song(analyzed_song_with_stems)

    assert result is None
    service.transcribe.assert_not_called()


def test_transcribe_song_missing_row_logs_and_returns():
    with (
        patch("app.workers.transcribe.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.transcribe.TranscriptionService",
            return_value=_patched_service(),
        ),
    ):
        from app.workers.transcribe import transcribe_song

        assert transcribe_song(str(uuid.uuid4())) is None


def test_transcribe_song_replaces_existing_transcription_row(
    analyzed_song_with_stems: str,
):
    sid = uuid.UUID(analyzed_song_with_stems)
    with SessionLocal() as db:
        db.add(
            Transcription(
                song_id=sid,
                model_name="large-v3",
                status=TranscriptionStatus.error,
                language=None,
                segments=[],
                vocal_rms_threshold=0.005,
                vocal_rms_observed=0.15,
            )
        )
        db.commit()

    service = _patched_service()
    with (
        patch("app.workers.transcribe.get_storage", return_value=AsyncMock()),
        patch(
            "app.workers.transcribe.TranscriptionService", return_value=service
        ),
    ):
        from app.workers.transcribe import transcribe_song

        transcribe_song(analyzed_song_with_stems)

    with SessionLocal() as db:
        rows = (
            db.query(Transcription).filter(Transcription.song_id == sid).all()
        )
        assert len(rows) == 1
        # New status replaces the old error row.
        assert rows[0].status == TranscriptionStatus.success
        assert len(rows[0].segments) == 1
