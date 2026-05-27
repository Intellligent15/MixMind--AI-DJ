import logging
import uuid
from typing import Any

from sqlalchemy import select

from app.workers import celery_app
from app.core.db import SessionLocal
from app.models.lyrics import Lyrics, LyricsAlignmentStatus, LyricsFetchStatus
from app.models.transcription import Transcription, TranscriptionStatus
from app.services.lyrics_alignment.aligner import align_lyrics

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.align_lyrics.align_lyrics_task", bind=True, max_retries=5)
def align_lyrics_task(self: Any, song_id: uuid.UUID | str) -> str | None:
    if isinstance(song_id, str):
        song_id = uuid.UUID(song_id)

    with SessionLocal() as db:
        lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
        if not lyrics:
            # fetch_lyrics probably hasn't created the row yet
            raise self.retry(countdown=10)

        if lyrics.fetch_status == LyricsFetchStatus.not_attempted:
            raise self.retry(countdown=10)
            
        if lyrics.fetch_status != LyricsFetchStatus.success:
            logger.info(f"align_lyrics: fetch_status is {lyrics.fetch_status}, skipping alignment")
            lyrics.alignment_status = LyricsAlignmentStatus.error
            db.commit()
            return str(song_id)

        transcription = db.scalar(select(Transcription).where(Transcription.song_id == song_id))
        if not transcription:
            logger.info(f"align_lyrics: no transcription for {song_id}")
            return str(song_id)
            
        if transcription.status != TranscriptionStatus.success:
            logger.info(f"align_lyrics: transcription status is {transcription.status}, skipping alignment")
            lyrics.alignment_status = LyricsAlignmentStatus.error
            db.commit()
            return str(song_id)

        if lyrics.alignment_status in (LyricsAlignmentStatus.success, LyricsAlignmentStatus.low_quality):
            return str(song_id)

        try:
            result = align_lyrics(transcription.segments, lyrics.text or "")
            lyrics.aligned_words = result["aligned_words"]
            lyrics.alignment_quality = result["alignment_quality"]
            lyrics.alignment_status = result["alignment_status"]
            logger.info(f"align_lyrics: {song_id} aligned with quality {lyrics.alignment_quality:.2f}")
        except Exception as e:
            logger.error(f"align_lyrics: error for {song_id}: {e}", exc_info=True)
            lyrics.alignment_status = LyricsAlignmentStatus.error
            db.commit()
            raise self.retry(exc=e, countdown=30)

        db.commit()

    return str(song_id)
