from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import (
    MixPlan,
    MixPlanStatus,
    Queue,
    Song,
    SongStatus,
)
from app.schemas import MixPlanRead
from app.services.storage import get_storage
from app.workers import celery_app

RENDER_TASK = "app.workers.render_transition.render_transition"
RENDER_GATE = (
    MixPlanStatus.pending,
    MixPlanStatus.failed,
    MixPlanStatus.ready,
)

router = APIRouter(tags=["mix_plans"])


def _seed_mix_plans(queue: Queue, db: Session) -> list[MixPlan]:
    """Create one MixPlan row per adjacent pair in the locked queue.

    Idempotent: existing rows (unique (queue_id, from_song_id, to_song_id))
    are left untouched and returned alongside any newly-created ones.
    """
    items = sorted(queue.items, key=lambda it: it.position)
    if len(items) < 2:
        return []

    existing = {
        (row.from_song_id, row.to_song_id): row
        for row in db.scalars(
            select(MixPlan).where(MixPlan.queue_id == queue.id)
        )
    }
    rows: list[MixPlan] = []
    for a, b in zip(items[:-1], items[1:]):
        existing_row = existing.get((a.song_id, b.song_id))
        if existing_row is not None:
            rows.append(existing_row)
            continue
        row = MixPlan(
            queue_id=queue.id,
            from_song_id=a.song_id,
            to_song_id=b.song_id,
            plan_json=None,
            status=MixPlanStatus.pending,
        )
        db.add(row)
        rows.append(row)
    db.commit()
    for r in rows:
        db.refresh(r)
    return rows


@router.post(
    "/api/queues/{queue_id}/mix_plans",
    response_model=list[MixPlanRead],
    status_code=status.HTTP_201_CREATED,
)
def seed_mix_plans(queue_id: uuid.UUID, db: Session = Depends(get_db)):
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    if not queue.locked:
        raise HTTPException(
            status_code=409,
            detail="queue must be locked before seeding mix plans",
        )
    rows = _seed_mix_plans(queue, db)
    return rows


@router.get(
    "/api/queues/{queue_id}/mix_plans",
    response_model=list[MixPlanRead],
)
def list_mix_plans(queue_id: uuid.UUID, db: Session = Depends(get_db)):
    queue = db.get(Queue, queue_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="queue not found")
    # Order by from_song's queue position so the list reads "first
    # transition first."
    items_by_song = {
        it.song_id: it.position for it in queue.items
    }
    rows = list(
        db.scalars(
            select(MixPlan).where(MixPlan.queue_id == queue_id)
        )
    )
    rows.sort(key=lambda r: items_by_song.get(r.from_song_id, 1_000_000))
    return rows


@router.get(
    "/api/mix_plans/{mix_plan_id}",
    response_model=MixPlanRead,
)
def get_mix_plan(mix_plan_id: uuid.UUID, db: Session = Depends(get_db)):
    row = db.get(MixPlan, mix_plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="mix plan not found")
    return row


@router.post(
    "/api/mix_plans/{mix_plan_id}/render",
    status_code=status.HTTP_202_ACCEPTED,
)
def render_mix_plan(mix_plan_id: uuid.UUID, db: Session = Depends(get_db)):
    row = db.get(MixPlan, mix_plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="mix plan not found")
    if row.status not in RENDER_GATE:
        raise HTTPException(
            status_code=409,
            detail=f"mix plan is {row.status.value}, cannot render",
        )
    a = db.get(Song, row.from_song_id)
    b = db.get(Song, row.to_song_id)
    if a is None or b is None:
        raise HTTPException(status_code=409, detail="pair songs missing")
    if a.status != SongStatus.ready or b.status != SongStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=f"pair songs not ready (a={a.status.value}, b={b.status.value})",
        )
    celery_app.send_task(RENDER_TASK, args=[str(mix_plan_id)])
    return {"status": "accepted"}


@router.get(
    "/api/mix_plans/{mix_plan_id}/audio",
)
def get_mix_plan_audio(mix_plan_id: uuid.UUID, db: Session = Depends(get_db)):
    row = db.get(MixPlan, mix_plan_id)
    if row is None or row.rendered_audio_path is None:
        raise HTTPException(status_code=404, detail="no rendered audio")
    storage = get_storage()
    path = storage.path(row.rendered_audio_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="rendered audio missing on disk")
    return FileResponse(path, media_type="audio/wav")
