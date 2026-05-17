from __future__ import annotations

import logging
import uuid

from app.core.db import SessionLocal
from app.models import Song, SongStatus
from app.services.storage import get_storage
from app.services.youtube import YouTubeService
from app.workers import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.download.download_song")
def download_song(song_id: str) -> str:
    """Download the audio for a Song to local storage.

    Transitions: pending/failed -> downloading -> downloaded (or failed).
    Returns the absolute audio path on success. Re-raises on failure after
    marking the row failed so Celery records the error.
    """
    song_uuid = uuid.UUID(song_id)
    storage = get_storage()
    yt = YouTubeService()

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        if song is None:
            raise RuntimeError(f"Song {song_id} not found")
        song.status = SongStatus.downloading
        db.commit()
        video_id = song.youtube_video_id

    key = f"audio/{video_id}.wav"
    dest = storage.path(key)

    try:
        yt.download(video_id, dest)
    except Exception as exc:
        logger.exception("download failed for %s", video_id)
        with SessionLocal() as db:
            song = db.get(Song, song_uuid)
            if song is not None:
                song.status = SongStatus.failed
                db.commit()
        raise

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        assert song is not None
        # Store the logical storage key, not the absolute filesystem path.
        # The host worker and the dockerized backend resolve this key
        # against different roots (./cache vs /app/cache).
        song.audio_path = key
        song.status = SongStatus.downloaded
        db.commit()

    return key
