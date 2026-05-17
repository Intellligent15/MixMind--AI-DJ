"""Download task tests.

The Celery task opens its own SessionLocal, so these tests use the real DB
(not the rollback-on-teardown fixture) and clean up by id at the end.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.db import SessionLocal
from app.models import Song, SongStatus
from app.services.youtube.service import YouTubeDownloadError


@pytest.fixture
def song_id():
    with SessionLocal() as db:
        song = Song(
            youtube_video_id=f"test-{id(object())}",
            title="T",
            artist=None,
            duration_seconds=10.0,
            thumbnail_url=None,
            status=SongStatus.pending,
        )
        db.add(song)
        db.commit()
        sid = str(song.id)
    yield sid
    with SessionLocal() as db:
        row = db.get(Song, __import__("uuid").UUID(sid))
        if row is not None:
            db.delete(row)
            db.commit()


def test_download_song_happy_path(song_id: str, tmp_path: Path):
    dest = tmp_path / "audio" / "out.wav"

    storage = MagicMock()
    storage.path.return_value = dest

    yt = MagicMock()

    def fake_download(video_id, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"RIFFwave")

    yt.download.side_effect = fake_download

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        result = download_song(song_id)

    with SessionLocal() as db:
        import uuid

        row = db.get(Song, uuid.UUID(song_id))
        assert row is not None
        assert row.status == SongStatus.downloaded
        assert row.audio_path == f"audio/{row.youtube_video_id}.wav"
        assert result == row.audio_path


def test_download_song_marks_failed_on_error(song_id: str, tmp_path: Path):
    storage = MagicMock()
    storage.path.return_value = tmp_path / "out.wav"
    yt = MagicMock()
    yt.download.side_effect = YouTubeDownloadError("boom")

    with (
        patch("app.workers.download.get_storage", return_value=storage),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        with pytest.raises(YouTubeDownloadError):
            download_song(song_id)

    with SessionLocal() as db:
        import uuid

        row = db.get(Song, uuid.UUID(song_id))
        assert row is not None
        assert row.status == SongStatus.failed
        assert row.audio_path is None


def test_download_song_missing_row_logs_and_returns():
    """A stale broker message for a deleted song must not crash the worker."""
    import uuid as _uuid

    with (
        patch("app.workers.download.get_storage", return_value=MagicMock()),
        patch("app.workers.download.YouTubeService", return_value=MagicMock()),
    ):
        from app.workers.download import download_song

        assert download_song(str(_uuid.uuid4())) is None


def test_download_song_skips_if_already_downloading(song_id: str):
    """Concurrent dispatch: the loser sees status=downloading and bails."""
    import uuid as _uuid

    # Pre-mark the row as downloading to simulate another task winning the
    # atomic claim first.
    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        row.status = SongStatus.downloading
        db.commit()

    yt = MagicMock()
    with (
        patch("app.workers.download.get_storage", return_value=MagicMock()),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        result = download_song(song_id)

    assert result is None
    yt.download.assert_not_called()
    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        # Status unchanged: the loser did not touch it.
        assert row.status == SongStatus.downloading


def test_download_song_skips_if_already_downloaded(song_id: str):
    """Re-dispatch after success is a no-op (no re-download)."""
    import uuid as _uuid

    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        row.status = SongStatus.downloaded
        row.audio_path = f"audio/{row.youtube_video_id}.wav"
        db.commit()

    yt = MagicMock()
    with (
        patch("app.workers.download.get_storage", return_value=MagicMock()),
        patch("app.workers.download.YouTubeService", return_value=yt),
    ):
        from app.workers.download import download_song

        result = download_song(song_id)

    assert result is not None and result.endswith(".wav")
    yt.download.assert_not_called()
    with SessionLocal() as db:
        row = db.get(Song, _uuid.UUID(song_id))
        assert row is not None
        assert row.status == SongStatus.downloaded
