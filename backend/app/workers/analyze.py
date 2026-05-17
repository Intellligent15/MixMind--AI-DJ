from __future__ import annotations

import logging
import uuid

from sqlalchemy import update

from app.core.db import SessionLocal
from app.models import Analysis, Song, SongStatus
from app.services.analysis.service import AnalysisService
from app.services.storage import get_storage
from app.workers import celery_app

logger = logging.getLogger(__name__)

# Statuses we can transition into `analyzing`. Mirrors the API gate so
# manual re-analyze and recovery from a failed run both work.
CLAIMABLE_STATUSES = (SongStatus.downloaded, SongStatus.analyzed, SongStatus.failed)


@celery_app.task(name="app.workers.analyze.analyze_song")
def analyze_song(song_id: str) -> str | None:
    """Run the analysis pipeline for a downloaded Song.

    Idempotent under concurrent dispatch: the transition into `analyzing`
    is an atomic SQL UPDATE that only one task can win. Losers log and
    return without touching the row, leaving the winner to produce the
    Analysis.

    Transitions: downloaded/analyzed/failed -> analyzing -> analyzed
    (or failed). Re-running on a song that already has an Analysis row
    overwrites it.
    """
    song_uuid = uuid.UUID(song_id)
    storage = get_storage()
    service = AnalysisService()

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        if song is None:
            logger.warning("analyze_song: song %s not found, skipping", song_id)
            return None
        if not song.audio_path:
            logger.warning(
                "analyze_song: song %s has no audio_path yet, skipping", song_id
            )
            return None

        claim = db.execute(
            update(Song)
            .where(Song.id == song_uuid)
            .where(Song.status.in_(CLAIMABLE_STATUSES))
            .values(status=SongStatus.analyzing)
        )
        db.commit()

        if claim.rowcount == 0:
            db.refresh(song)
            logger.info(
                "analyze_song: %s already %s, skipping duplicate dispatch",
                song_id,
                song.status.value,
            )
            return None

        audio_key = song.audio_path

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
