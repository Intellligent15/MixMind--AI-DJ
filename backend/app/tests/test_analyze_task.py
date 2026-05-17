"""analyze_song Celery task tests.

Same pattern as test_download_task: real DB, clean up by id at the end,
mock out the heavy analysis service so the test stays fast.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.db import SessionLocal
from app.models import Analysis, Song, SongStatus
from app.services.analysis.sections.base import Section
from app.services.analysis.service import AnalysisResult


@pytest.fixture
def downloaded_song():
    sid_str = None
    with SessionLocal() as db:
        song = Song(
            youtube_video_id=f"analtest-{id(object())}",
            title="T",
            artist=None,
            duration_seconds=10.0,
            thumbnail_url=None,
            audio_path="audio/fake.wav",
            status=SongStatus.downloaded,
        )
        db.add(song)
        db.commit()
        sid_str = str(song.id)
    yield sid_str
    sid = uuid.UUID(sid_str)
    with SessionLocal() as db:
        analysis = (
            db.query(Analysis).filter(Analysis.song_id == sid).one_or_none()
        )
        if analysis is not None:
            db.delete(analysis)
        row = db.get(Song, sid)
        if row is not None:
            db.delete(row)
        db.commit()


def _fake_result() -> AnalysisResult:
    return AnalysisResult(
        bpm=124.0,
        key="Am",
        camelot_key="8A",
        time_signature=4,
        beat_grid=[0.0, 0.5, 1.0, 1.5],
        downbeats=[0.0],
        sections=[Section(start=0.0, end=2.0, label="section_1")],
        energy_curve=[0.1, 0.2],
        vocal_segments=[],
    )


def test_analyze_song_happy_path(downloaded_song: str, tmp_path: Path):
    storage = MagicMock()
    storage.path.return_value = tmp_path / "fake.wav"

    service = MagicMock()
    service.analyze.return_value = _fake_result()

    with (
        patch("app.workers.analyze.get_storage", return_value=storage),
        patch("app.workers.analyze.AnalysisService", return_value=service),
    ):
        from app.workers.analyze import analyze_song

        analyze_song(downloaded_song)

    sid = uuid.UUID(downloaded_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        assert song is not None
        assert song.status == SongStatus.analyzed
        analysis = (
            db.query(Analysis).filter(Analysis.song_id == sid).one()
        )
        assert analysis.bpm == 124.0
        assert analysis.key == "Am"
        assert analysis.camelot_key == "8A"
        assert analysis.beat_grid == [0.0, 0.5, 1.0, 1.5]
        assert analysis.sections == [
            {"start": 0.0, "end": 2.0, "label": "section_1"}
        ]
        assert analysis.vocal_segments == []


def test_analyze_song_marks_failed_on_error(downloaded_song: str, tmp_path: Path):
    storage = MagicMock()
    storage.path.return_value = tmp_path / "fake.wav"
    service = MagicMock()
    service.analyze.side_effect = RuntimeError("librosa boom")

    with (
        patch("app.workers.analyze.get_storage", return_value=storage),
        patch("app.workers.analyze.AnalysisService", return_value=service),
    ):
        from app.workers.analyze import analyze_song

        with pytest.raises(RuntimeError, match="librosa boom"):
            analyze_song(downloaded_song)

    sid = uuid.UUID(downloaded_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        assert song is not None
        assert song.status == SongStatus.failed
        assert (
            db.query(Analysis).filter(Analysis.song_id == sid).one_or_none()
            is None
        )


def test_analyze_song_skips_wrong_status(downloaded_song: str):
    """A song still pending/downloading is silently skipped — the worker
    doesn't raise. The atomic claim will fail rowcount == 0."""
    sid = uuid.UUID(downloaded_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        song.status = SongStatus.pending
        db.commit()

    service = MagicMock()
    with (
        patch("app.workers.analyze.get_storage", return_value=MagicMock()),
        patch("app.workers.analyze.AnalysisService", return_value=service),
    ):
        from app.workers.analyze import analyze_song

        result = analyze_song(downloaded_song)

    assert result is None
    service.analyze.assert_not_called()


def test_analyze_song_missing_row_logs_and_returns():
    with (
        patch("app.workers.analyze.get_storage", return_value=MagicMock()),
        patch("app.workers.analyze.AnalysisService", return_value=MagicMock()),
    ):
        from app.workers.analyze import analyze_song

        assert analyze_song(str(uuid.uuid4())) is None


def test_analyze_song_skips_if_already_analyzing(downloaded_song: str):
    """Concurrent dispatch: the loser sees status=analyzing and bails."""
    sid = uuid.UUID(downloaded_song)
    with SessionLocal() as db:
        song = db.get(Song, sid)
        song.status = SongStatus.analyzing
        db.commit()

    service = MagicMock()
    with (
        patch("app.workers.analyze.get_storage", return_value=MagicMock()),
        patch("app.workers.analyze.AnalysisService", return_value=service),
    ):
        from app.workers.analyze import analyze_song

        result = analyze_song(downloaded_song)

    assert result is None
    service.analyze.assert_not_called()
    with SessionLocal() as db:
        row = db.get(Song, sid)
        assert row is not None
        assert row.status == SongStatus.analyzing


def test_analyze_song_replaces_existing_analysis(downloaded_song: str, tmp_path: Path):
    sid = uuid.UUID(downloaded_song)
    with SessionLocal() as db:
        db.add(
            Analysis(
                song_id=sid,
                bpm=99.0,
                key="X",
                camelot_key="0Z",
                time_signature=4,
                beat_grid=[],
                downbeats=[],
                sections=[],
                energy_curve=[],
                vocal_segments=[],
            )
        )
        song = db.get(Song, sid)
        # Re-running requires a non-pending status; analyzed is the realistic case.
        song.status = SongStatus.analyzed
        db.commit()

    storage = MagicMock()
    storage.path.return_value = tmp_path / "fake.wav"
    service = MagicMock()
    service.analyze.return_value = _fake_result()

    with (
        patch("app.workers.analyze.get_storage", return_value=storage),
        patch("app.workers.analyze.AnalysisService", return_value=service),
    ):
        from app.workers.analyze import analyze_song

        analyze_song(downloaded_song)

    with SessionLocal() as db:
        analyses = (
            db.query(Analysis).filter(Analysis.song_id == sid).all()
        )
        assert len(analyses) == 1
        assert analyses[0].bpm == 124.0
