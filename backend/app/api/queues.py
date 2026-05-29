from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import (
    MixPlan,
    Queue,
    QueueItem,
    Song,
    SongStatus,
    Stems,
    Transcription,
)
from app.schemas import QueueItemAdd, QueueRead, QueueReorder, QueueRenderRead
from app.workers import (
    PRI_ANALYZE,
    PRI_DOWNLOAD,
    PRI_SEPARATE,
    PRI_TRANSCRIBE,
    celery_app,
)
from app.workers.analyze import analyze_song
from app.workers.download import download_song

# Dispatched by task name so the API container doesn't have to import
# app.workers.separate / app.workers.transcribe (which pull torch + demucs
# + mlx-whisper — native-worker only).
SEPARATE_TASK = "app.workers.separate.separate_stems"
TRANSCRIBE_TASK = "app.workers.transcribe.transcribe_song"


class _TaskShim:
    """Adapter exposing the .s/.si/.delay surface tests + chain composition
    expect, without importing the underlying ML-bound modules."""

    def __init__(self, task_name: str) -> None:
        self._task_name = task_name

    def s(self, *args):
        return celery_app.signature(self._task_name, args=args)

    def si(self, *args):
        return celery_app.signature(self._task_name, args=args, immutable=True)

    def delay(self, *args):
        return celery_app.send_task(self._task_name, args=list(args))

    def apply_async(self, args=None, priority=None, **kwargs):
        send_kwargs = {"args": list(args or []), **kwargs}
        if priority is not None:
            send_kwargs["priority"] = priority
        return celery_app.send_task(self._task_name, **send_kwargs)


separate_stems = _TaskShim(SEPARATE_TASK)
transcribe_song = _TaskShim(TRANSCRIBE_TASK)

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
async def create_queue(db: Session = Depends(get_db)) -> Queue:
    from app.models.queue_render import QueueRender
    from app.services.storage import get_storage
    import logging
    
    storage = get_storage()
    logger = logging.getLogger(__name__)
    old_queues = list(db.scalars(select(Queue)).all())
    for old_q in old_queues:
        # Collect every storage blob the queue owns before the cascade
        # delete removes the rows that point to them. Both the stitched
        # queue mix AND each per-pair transition render (mixes/<id>.wav)
        # must go — otherwise the per-pair WAVs orphan in object storage.
        keys: list[str] = []
        render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == old_q.id))
        if render_row and render_row.rendered_audio_path:
            keys.append(render_row.rendered_audio_path)
        plans = db.scalars(select(MixPlan).where(MixPlan.queue_id == old_q.id)).all()
        for plan in plans:
            if plan.rendered_audio_path:
                keys.append(plan.rendered_audio_path)

        for key in keys:
            try:
                await storage.delete(key)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"create_queue: failed to delete old blob {key!r}: {e}")
        db.delete(old_q)
    db.commit()

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
    """Kick the appropriate next pipeline stage for one song.

    Workers auto-chain on success (download→analyze→separate→transcribe)
    when `song.pipeline_requested` is True, so this only needs to dispatch
    the SINGLE current-needed stage. Songs already in flight
    (`downloading`/`analyzing`/etc.) are no-ops — the in-flight worker
    will auto-dispatch its successor on completion. That auto-chain is
    what fixes the "lock-during-download stalls the pipeline" bug.

    Lyrics fetch is fire-and-forget, independent of the audio pipeline.
    """
    sid = str(song.id)
    # Signal that the user wants the full pipeline for this song. Workers
    # gate their auto-dispatch on this flag, so flipping it true here is
    # what makes a currently-running download or analyze auto-progress
    # all the way through transcribe.
    song.pipeline_requested = True
    db.commit()

    has_stems = (
        db.scalar(select(Stems.id).where(Stems.song_id == song.id)) is not None
    )
    has_transcription = (
        db.scalar(select(Transcription.id).where(Transcription.song_id == song.id))
        is not None
    )

    from app.models.lyrics import Lyrics, LyricsFetchStatus
    has_lyrics = (
        db.scalar(select(Lyrics.id).where(
            (Lyrics.song_id == song.id) &
            (Lyrics.fetch_status.in_([
                LyricsFetchStatus.success,
                LyricsFetchStatus.not_found,
                LyricsFetchStatus.error,
            ]))
        )) is not None
    )
    if not has_lyrics:
        _TaskShim("app.workers.fetch_lyrics.fetch_lyrics").delay(sid)

    # In-flight states: a running worker will carry the pipeline forward
    # on its own via auto-chain (which now sees pipeline_requested=True).
    if song.status in (
        SongStatus.downloading,
        SongStatus.analyzing,
        SongStatus.separating,
        SongStatus.transcribing,
    ):
        return

    # Idle states: kick the next-needed stage. Auto-chain handles the rest.
    if song.status in (SongStatus.pending, SongStatus.failed) and not song.audio_path:
        download_song.apply_async(args=[sid], priority=PRI_DOWNLOAD)
    elif song.status in (SongStatus.downloaded, SongStatus.failed):
        analyze_song.apply_async(args=[sid], priority=PRI_ANALYZE)
    elif song.status in (SongStatus.analyzed, SongStatus.ready):
        if not has_stems:
            separate_stems.apply_async(args=[sid], priority=PRI_SEPARATE)
        elif not has_transcription:
            transcribe_song.apply_async(args=[sid], priority=PRI_TRANSCRIBE)
        # else: fully processed — nothing to do.


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

    # Phase 7: seed MixPlan rows for each adjacent pair. plan_json is
    # generated lazily at render time so the LLM call in Phase 9 doesn't
    # fire for plans the user never asks to render. Local import to dodge
    # the api/queues ↔ api/mix_plans circular at module load.
    from app.api.mix_plans import _seed_mix_plans
    _seed_mix_plans(queue, db)

    return queue


@router.post(
    "/{queue_id}/stitch",
    status_code=status.HTTP_202_ACCEPTED,
)
def stitch_queue_route(queue_id: uuid.UUID, db: Session = Depends(get_db)):
    from app.models.queue_render import QueueRender, QueueRenderStatus
    
    queue = db.get(Queue, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="queue not found")
    if not queue.locked:
        raise HTTPException(status_code=409, detail="queue must be locked to stitch")
        
    render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_id))
    if not render_row:
        render_row = QueueRender(queue_id=queue_id)
        db.add(render_row)
    else:
        render_row.status = QueueRenderStatus.pending
        render_row.error_text = None
    db.commit()
    db.refresh(render_row)
    
    from app.models import MixPlan, MixPlanStatus
    from celery import chord

    mix_plans = db.scalars(select(MixPlan).where(MixPlan.queue_id == queue_id)).all()
    render_tasks = []
    for mp in mix_plans:
        if mp.status != MixPlanStatus.ready:
            render_tasks.append(_TaskShim("app.workers.render_transition.render_transition").si(str(mp.id)))

    stitch_task = _TaskShim("app.workers.stitch_queue.stitch_queue").si(str(queue_id))
    
    if render_tasks:
        chord(render_tasks)(stitch_task)
    else:
        stitch_task.delay()

    return {"message": "Stitching started"}


@router.get("/{queue_id}/mix", response_model=QueueRenderRead)
def get_queue_mix(queue_id: uuid.UUID, db: Session = Depends(get_db)):
    from app.models.queue_render import QueueRender
    
    render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_id))
    if not render_row:
        raise HTTPException(status_code=404, detail="no mix found for queue")
    return render_row


@router.get("/{queue_id}/mix/audio")
async def get_queue_mix_audio(
    queue_id: uuid.UUID,
    db: Session = Depends(get_db),
    range: str | None = Header(default=None),
):
    from app.models.queue_render import QueueRender, QueueRenderStatus
    from app.api.songs import _stream_audio_response
    
    render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_id))
    if not render_row or render_row.status != QueueRenderStatus.ready or not render_row.rendered_audio_path:
        raise HTTPException(status_code=404, detail="mix audio not ready")
        
    return await _stream_audio_response(render_row.rendered_audio_path, "audio/flac", range, download_filename="mix.flac")
