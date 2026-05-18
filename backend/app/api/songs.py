from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import Analysis, Song, SongStatus, Stems, Transcription
from app.schemas import (
    AnalysisRead,
    SongCreate,
    SongRead,
    StemsRead,
    TranscriptionRead,
)
from app.services.storage import get_storage
from app.workers import celery_app
from app.workers.analyze import analyze_song
from app.workers.download import download_song

# Dispatched via celery_app.send_task so the API container doesn't have to
# import app.workers.separate / app.workers.transcribe — those modules pull
# in torch + demucs + mlx-whisper, which only the native worker needs.
SEPARATE_TASK = "app.workers.separate.separate_stems"
TRANSCRIBE_TASK = "app.workers.transcribe.transcribe_song"
# Mirrors STEM_NAMES in app.services.stems (kept inline so the API doesn't
# import the stems service module either — same reason as above).
STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "other")

router = APIRouter(prefix="/api/songs", tags=["songs"])


@router.get("", response_model=list[SongRead])
def list_songs(db: Session = Depends(get_db)) -> list[Song]:
    return list(
        db.scalars(select(Song).order_by(Song.created_at.desc())).all()
    )


@router.post("", response_model=SongRead, status_code=status.HTTP_201_CREATED)
def create_song(payload: SongCreate, db: Session = Depends(get_db)) -> Song:
    existing = db.scalar(
        select(Song).where(Song.youtube_video_id == payload.youtube_video_id)
    )
    if existing is not None:
        return existing

    song = Song(
        youtube_video_id=payload.youtube_video_id,
        title=payload.title,
        artist=payload.artist,
        duration_seconds=payload.duration_seconds,
        thumbnail_url=payload.thumbnail_url,
        status=SongStatus.pending,
    )
    db.add(song)
    db.commit()
    db.refresh(song)
    download_song.delay(str(song.id))
    return song


@router.get("/{song_id}", response_model=SongRead)
def get_song(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Song:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    return song


@router.get("/{song_id}/audio")
def get_song_audio(song_id: uuid.UUID, db: Session = Depends(get_db)) -> FileResponse:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status in (SongStatus.pending, SongStatus.downloading) or not song.audio_path:
        raise HTTPException(
            status_code=409,
            detail=f"audio not available (status={song.status.value})",
        )
    path = get_storage().path(song.audio_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="audio file missing on disk")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@router.post(
    "/{song_id}/analyze",
    response_model=SongRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_analyze_song(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Song:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status not in (SongStatus.downloaded, SongStatus.analyzed, SongStatus.failed):
        raise HTTPException(
            status_code=409,
            detail=f"song not ready for analysis (status={song.status.value})",
        )
    analyze_song.delay(str(song.id))
    return song


@router.get("/{song_id}/analysis", response_model=AnalysisRead)
def get_song_analysis(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Analysis:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    analysis = db.scalar(select(Analysis).where(Analysis.song_id == song_id))
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis not available")
    return analysis


# Same gate as the separate_stems worker's CLAIMABLE_STATUSES — keep these
# in sync; the API rejects early so the frontend gets a useful error
# instead of a silently-dropped task.
SEPARATE_GATE = (SongStatus.analyzed, SongStatus.ready, SongStatus.failed)


@router.post(
    "/{song_id}/separate",
    response_model=SongRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_separate_song(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Song:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status not in SEPARATE_GATE:
        raise HTTPException(
            status_code=409,
            detail=f"song not ready for separation (status={song.status.value})",
        )
    celery_app.send_task(SEPARATE_TASK, args=[str(song.id)])
    return song


@router.get("/{song_id}/stems", response_model=StemsRead)
def get_song_stems(song_id: uuid.UUID, db: Session = Depends(get_db)) -> Stems:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    stems = db.scalar(select(Stems).where(Stems.song_id == song_id))
    if stems is None:
        raise HTTPException(status_code=404, detail="stems not available")
    return stems


@router.get("/{song_id}/stems/{stem_name}")
def get_song_stem_audio(
    song_id: uuid.UUID, stem_name: str, db: Session = Depends(get_db)
) -> FileResponse:
    if stem_name not in STEM_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"unknown stem (expected one of {list(STEM_NAMES)})",
        )
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    stems = db.scalar(select(Stems).where(Stems.song_id == song_id))
    if stems is None:
        raise HTTPException(status_code=404, detail="stems not available")

    key = {
        "vocals": stems.vocals_path,
        "drums": stems.drums_path,
        "bass": stems.bass_path,
        "other": stems.other_path,
    }[stem_name]
    if key is None:
        raise HTTPException(status_code=409, detail=f"{stem_name} stem not written yet")

    path = get_storage().path(key)
    if not path.exists():
        raise HTTPException(status_code=410, detail="stem file missing on disk")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


# Same gate as the transcribe_song worker's CLAIMABLE_STATUSES — keep in
# sync. The API rejects early so the frontend gets a useful error instead
# of a silently-dropped task.
TRANSCRIBE_GATE = (SongStatus.analyzed, SongStatus.ready, SongStatus.failed)


@router.post(
    "/{song_id}/transcribe",
    response_model=SongRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_transcribe_song(
    song_id: uuid.UUID, db: Session = Depends(get_db)
) -> Song:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    if song.status not in TRANSCRIBE_GATE:
        raise HTTPException(
            status_code=409,
            detail=f"song not ready for transcription (status={song.status.value})",
        )
    # Stems must exist — the worker will fail loudly otherwise, but
    # rejecting up-front gives the frontend a usable error.
    stems = db.scalar(select(Stems).where(Stems.song_id == song_id))
    if stems is None or stems.vocals_path is None:
        raise HTTPException(
            status_code=409,
            detail="song has no separated vocal stem yet",
        )
    celery_app.send_task(TRANSCRIBE_TASK, args=[str(song.id)])
    return song


@router.get("/{song_id}/transcription", response_model=TranscriptionRead)
def get_song_transcription(
    song_id: uuid.UUID, db: Session = Depends(get_db)
) -> Transcription:
    song = db.get(Song, song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    transcription = db.scalar(
        select(Transcription).where(Transcription.song_id == song_id)
    )
    if transcription is None:
        raise HTTPException(status_code=404, detail="transcription not available")
    return transcription
