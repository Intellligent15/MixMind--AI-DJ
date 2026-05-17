from __future__ import annotations

import logging
import uuid

from app.core.db import SessionLocal
from app.models import Analysis, Song, SongStatus
from app.services.analysis.service import AnalysisService
from app.services.storage import get_storage
from app.workers import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.analyze.analyze_song")
def analyze_song(song_id: str) -> str:
    """Run the analysis pipeline for a downloaded Song.

    Transitions: downloaded -> analyzing -> analyzed (or failed).
    Re-running on a song that already has an Analysis row overwrites it.
    """
    song_uuid = uuid.UUID(song_id)
    storage = get_storage()
    service = AnalysisService()

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        if song is None:
            raise RuntimeError(f"Song {song_id} not found")
        if song.status not in (SongStatus.downloaded, SongStatus.analyzed, SongStatus.failed):
            raise RuntimeError(
                f"Song {song_id} not ready for analysis (status={song.status.value})"
            )
        if not song.audio_path:
            raise RuntimeError(f"Song {song_id} has no audio_path")
        audio_key = song.audio_path
        song.status = SongStatus.analyzing
        db.commit()

    audio_path = storage.path(audio_key)

    try:
        result = service.analyze(audio_path)
    except Exception:
        logger.exception("analysis failed for song %s", song_id)
        with SessionLocal() as db:
            song = db.get(Song, song_uuid)
            if song is not None:
                song.status = SongStatus.failed
                db.commit()
        raise

    with SessionLocal() as db:
        existing = (
            db.query(Analysis).filter(Analysis.song_id == song_uuid).one_or_none()
        )
        if existing is not None:
            db.delete(existing)
            db.flush()
        analysis = Analysis(
            song_id=song_uuid,
            bpm=result.bpm,
            key=result.key,
            camelot_key=result.camelot_key,
            time_signature=result.time_signature,
            beat_grid=result.beat_grid,
            downbeats=result.downbeats,
            sections=[s.to_dict() for s in result.sections],
            energy_curve=result.energy_curve,
            vocal_segments=[list(seg) for seg in result.vocal_segments],
        )
        db.add(analysis)
        song = db.get(Song, song_uuid)
        assert song is not None
        song.status = SongStatus.analyzed
        db.commit()

    return str(song_uuid)
