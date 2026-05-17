from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import Song, SongStatus
from app.schemas import SongCreate, SongRead
from app.workers.download import download_song

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
    if song.status != SongStatus.downloaded or not song.audio_path:
        raise HTTPException(
            status_code=409,
            detail=f"audio not available (status={song.status.value})",
        )
    path = Path(song.audio_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="audio file missing on disk")
    return FileResponse(path, media_type="audio/wav", filename=path.name)
