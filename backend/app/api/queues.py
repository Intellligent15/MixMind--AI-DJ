from __future__ import annotations

import uuid
from datetime import datetime, timezone

from celery import chain
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import Queue, QueueItem, Song, SongStatus, Stems
from app.schemas import QueueItemAdd, QueueRead, QueueReorder
from app.workers.analyze import analyze_song
from app.workers.download import download_song
from app.workers.separate import separate_stems

router = APIRouter(prefix="/api/queues", tags=["queues"])


QUEUE_CAP = 20


def _compact_positions(db: Session, queue_id: uuid.UUID) -> None:
    """Re-pack item positions to be 0..N-1 in their current order."""
    items = list(
        db.scalars(
            select(QueueItem)
            .where(QueueItem.queue_id == queue_id)
            .order_by(QueueItem.position)
        ).all()
    )
    # Two passes to avoid colliding with the (queue_id, position) unique
    # constraint while we shuffle.
    for idx, item in enumerate(items):
        item.position = -1000 - idx
    db.flush()
    for idx, item in enumerate(items):
        item.position = idx
    db.flush()


@router.post("", response_model=QueueRead, status_code=status.HTTP_201_CREATED)
def create_queue(db: Session = Depends(get_db)) -> Queue:
    existing_unlocked = db.scalar(
        select(Queue).where(Queue.locked.is_(False)).order_by(Queue.created_at.desc())
    )
    if existing_unlocked is not None:
        raise HTTPException(
            status_code=409,
            detail="an unlocked queue already exists",
        )
    queue = Queue()
    db.add(queue)
    db.commit()
    db.refresh(queue)
    return queue


@router.get("/current", response_model=QueueRead)
def get_current_queue(db: Session = Depends(get_db)) -> Queue:
    queue = db.scalar(
        select(Queue).where(Queue.locked.is_(False)).order_by(Queue.created_at.desc())
    )
    if queue is None:
        queue = db.scalar(select(Queue).order_by(Queue.created_at.desc()))
    if queue is None:
        raise HTTPException(status_code=404, detail="no queue exists")
    return queue


@router.get("/{queue_id}", response_model=QueueRead)
def get_queue(queue_id: uuid.UUID, db: Session = Depends(get_db)) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    return queue


@router.post(
    "/{queue_id}/items",
    response_model=QueueRead,
    status_code=status.HTTP_201_CREATED,
)
def add_queue_item(
    queue_id: uuid.UUID, payload: QueueItemAdd, db: Session = Depends(get_db)
) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is locked")

    song = db.get(Song, payload.song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")

    current_count = len(queue.items)
    if current_count >= QUEUE_CAP:
        raise HTTPException(
            status_code=409,
            detail=f"queue is full (cap={QUEUE_CAP})",
        )

    item = QueueItem(
        queue_id=queue_id,
        song_id=song.id,
        position=current_count,
    )
    db.add(item)
    db.commit()

    # NOTE: POST /api/songs already dispatches download_song on song
    # creation, and POST /lock catches anything still pending. Dispatching
    # again here used to race with the songs-API dispatch — two parallel
    # yt-dlp processes writing the same .wav, one would fail and mark the
    # song failed even though the audio file was on disk.

    db.refresh(queue)
    return queue


@router.delete(
    "/{queue_id}/items/{item_id}",
    response_model=QueueRead,
)
def remove_queue_item(
    queue_id: uuid.UUID, item_id: uuid.UUID, db: Session = Depends(get_db)
) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is locked")

    item = db.get(QueueItem, item_id)
    if item is None or item.queue_id != queue_id:
        raise HTTPException(status_code=404, detail="queue item not found")

    db.delete(item)
    db.flush()
    _compact_positions(db, queue_id)
    db.commit()
    db.refresh(queue)
    return queue


@router.patch(
    "/{queue_id}/items",
    response_model=QueueRead,
)
def reorder_queue_items(
    queue_id: uuid.UUID, payload: QueueReorder, db: Session = Depends(get_db)
) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is locked")

    current_ids = {item.id for item in queue.items}
    requested_ids = list(payload.ordered_item_ids)
    if set(requested_ids) != current_ids or len(requested_ids) != len(current_ids):
        raise HTTPException(
            status_code=400,
            detail="ordered_item_ids must be a permutation of the queue's items",
        )

    by_id = {item.id: item for item in queue.items}
    # Two-pass shuffle to avoid the unique (queue_id, position) collision.
    for idx, item_id in enumerate(requested_ids):
        by_id[item_id].position = -1000 - idx
    db.flush()
    for idx, item_id in enumerate(requested_ids):
        by_id[item_id].position = idx
    db.commit()
    db.refresh(queue)
    return queue


def _enqueue_pipeline_for_song(song: Song, db: Session) -> None:
    """Kick off whatever pipeline stages a song still needs.

    Phase 5 extends Phase 4 with stem separation. The chain length depends
    on what the song already has:
      pending/failed-no-audio  -> download -> analyze -> separate
      downloaded/failed-w-audio -> analyze -> separate
      analyzed (no stems row)   -> separate
      analyzed (has stems row)  -> nothing

    Later phases will extend this with transcription, lyrics, mix planning.
    """
    sid = str(song.id)
    # Has the song already cleared the separation stage? One Stems row
    # per song; presence means we've already separated (or are mid-flight).
    has_stems = (
        db.scalar(select(Stems.id).where(Stems.song_id == song.id)) is not None
    )

    if song.status in (SongStatus.pending, SongStatus.failed) and not song.audio_path:
        # download -> analyze -> separate. .si() ignores upstream returns
        # so each task only sees its own song_id arg.
        if has_stems:
            chain(download_song.s(sid), analyze_song.si(sid)).delay()
        else:
            chain(
                download_song.s(sid),
                analyze_song.si(sid),
                separate_stems.si(sid),
            ).delay()
    elif song.status in (SongStatus.downloaded, SongStatus.failed):
        if has_stems:
            analyze_song.delay(sid)
        else:
            chain(analyze_song.s(sid), separate_stems.si(sid)).delay()
    elif song.status in (SongStatus.analyzed, SongStatus.ready) and not has_stems:
        separate_stems.delay(sid)
    # analyzed/ready with stems already: nothing to do.


@router.post(
    "/{queue_id}/lock",
    response_model=QueueRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def lock_queue(queue_id: uuid.UUID, db: Session = Depends(get_db)) -> Queue:
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if queue.locked:
        raise HTTPException(status_code=409, detail="queue is already locked")
    if not queue.items:
        raise HTTPException(status_code=409, detail="queue is empty")

    queue.locked = True
    queue.locked_at = datetime.now(timezone.utc)
    # Snapshot songs while the session is open; we'll enqueue after commit
    # so a transient task-broker failure doesn't roll back the lock.
    songs_to_pipeline = [item.song for item in queue.items]
    db.commit()
    db.refresh(queue)

    for song in songs_to_pipeline:
        _enqueue_pipeline_for_song(song, db)

    return queue
