from __future__ import annotations

import logging
import uuid

from sqlalchemy import update

from app.core.db import SessionLocal
from app.models import Song, SongStatus, Stems, StemsStatus
from app.services.stems import STEM_NAMES, StemSeparationService
from app.services.storage import get_storage
from app.workers import celery_app

logger = logging.getLogger(__name__)

# Statuses we can transition into `separating`. analyzed/ready/failed all
# legitimately want a (re-)separation; downloaded is too early (analysis
# hasn't run yet); pending/downloading/analyzing/separating/transcribing
# are mid-flight and would clobber a winning task.
CLAIMABLE_STATUSES = (SongStatus.analyzed, SongStatus.ready, SongStatus.failed)


def _stem_key(video_id: str, stem: str) -> str:
    return f"stems/{video_id}/{stem}.wav"


@celery_app.task(name="app.workers.separate.separate_stems")
def separate_stems(song_id: str) -> str | None:
    """Run Demucs htdemucs_ft over the song's audio and persist four stems.

    Idempotent under concurrent dispatch via the same atomic-claim pattern
    as download_song / analyze_song: the analyzed/ready/failed -> separating
    transition is a single UPDATE; losers log and return.

    Transitions: analyzed/ready/failed -> separating -> analyzed (or failed).
    Re-running on a song with an existing Stems row deletes the old row and
    its on-disk files before writing the new ones.

    Note we exit at `analyzed` not `separated` — `separated` lives on the
    Stems row, not the Song row. The Song status reflects "what's the most
    advanced pipeline stage the song has fully cleared"; until Phase 6
    transcription lands, analyzed is still the terminal Song status.
    """
    song_uuid = uuid.UUID(song_id)
    storage = get_storage()
    service = StemSeparationService()

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        if song is None:
            logger.warning("separate_stems: song %s not found, skipping", song_id)
            return None
        if not song.audio_path:
            logger.warning(
                "separate_stems: song %s has no audio_path yet, skipping", song_id
            )
            return None

        claim = db.execute(
            update(Song)
            .where(Song.id == song_uuid)
            .where(Song.status.in_(CLAIMABLE_STATUSES))
            .values(status=SongStatus.separating)
        )
        db.commit()

        if claim.rowcount == 0:
            db.refresh(song)
            logger.info(
                "separate_stems: %s already %s, skipping duplicate dispatch",
                song_id,
                song.status.value,
            )
            return None

        audio_key = song.audio_path
        video_id = song.youtube_video_id

    audio_path = storage.path(audio_key)

    try:
        result = service.separate(audio_path)
    except Exception:
        logger.exception("separation failed for song %s", song_id)
        with SessionLocal() as db:
            song = db.get(Song, song_uuid)
            if song is not None:
                song.status = SongStatus.failed
                db.commit()
        raise

    # Write the four WAVs through the storage path() resolver. Demucs gives
    # us tensors; torchaudio.save needs a real filesystem path (which the
    # local backend resolves directly, and a future S3 backend would buffer
    # to a tempfile + sync on close).
    keys: dict[str, str] = {}
    for stem_name in STEM_NAMES:
        key = _stem_key(video_id, stem_name)
        dest = storage.path(key)
        service.write_stem(result.stems[stem_name], result.sample_rate, dest)
        keys[stem_name] = key

    with SessionLocal() as db:
        existing = db.query(Stems).filter(Stems.song_id == song_uuid).one_or_none()
        if existing is not None:
            db.delete(existing)
            db.flush()
        stems_row = Stems(
            song_id=song_uuid,
            model_name=service.model_name,
            status=StemsStatus.separated,
            vocals_path=keys["vocals"],
            drums_path=keys["drums"],
            bass_path=keys["bass"],
            other_path=keys["other"],
            vocal_rms=result.vocal_rms,
        )
        db.add(stems_row)
        song = db.get(Song, song_uuid)
        assert song is not None
        # Bounce back to `analyzed` — that's still the terminal Song status
        # until Phase 6 lands transcription and Phase 7+ lands `ready`.
        song.status = SongStatus.analyzed
        db.commit()

    return str(song_uuid)
