"""Eager queue-stitch dispatch.

Phase 10 renders transitions DURING the Processing state instead of only
when the Player mounts. The hazard (called out in the Phase 10 brief): a
``render_transition`` no-ops unless BOTH of its songs are ``ready``, and
``stitch_queue`` fails unless every per-pair render is ready. So we can't
just fire the chord at lock time — every render would no-op and the stitch
would run on incomplete output.

The correct trigger is therefore "once every song in the locked queue has
reached ``ready``." At that point every pair's render is renderable, so the
``chord(render not-ready plans)(stitch_queue)`` is guaranteed to produce a
complete mix. ``maybe_dispatch_stitch`` enforces that precondition and is
idempotent, so it's safe to call from every place a song might finish:

- ``transcribe_song`` terminal ``ready`` (the usual path — the last song to
  finish processing fires the chord),
- ``lock_queue`` (covers the all-cached case where songs are already
  ``ready`` at lock and no worker re-runs to trigger the hook).

Dispatch is by task NAME (``celery_app.signature``) so the API container can
call this without importing the ML-bound worker modules.
"""

from __future__ import annotations

import logging
import uuid

from celery import chain, chord
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueRender,
    QueueRenderStatus,
    Song,
    SongStatus,
)
from app.workers import celery_app

logger = logging.getLogger(__name__)

RENDER_TASK = "app.workers.render_transition.render_transition"
STITCH_TASK = "app.workers.stitch_queue.stitch_queue"
PLAN_SET_TASK = "app.workers.plan_set.plan_set"

# Render-row states that mean "a stitch is already in flight or done."
# The eager path must not re-dispatch while the row sits here, otherwise a
# second song finishing would re-render every (already-ready) pair.
_LATCHED = (
    QueueRenderStatus.pending,
    QueueRenderStatus.rendering,
    QueueRenderStatus.ready,
)


def _dispatch_chord(queue_id: uuid.UUID, db: Session) -> None:
    """Fire ``chord(render not-ready plans)(stitch_queue)``.

    Assumes the caller has already (re)set the ``QueueRender`` row to
    ``pending``. Plans already ``ready`` are skipped — only the missing ones
    are rendered before the stitch callback runs.
    """
    plans = db.scalars(
        select(MixPlan).where(MixPlan.queue_id == queue_id)
    ).all()
    render_tasks = [
        celery_app.signature(RENDER_TASK, args=[str(mp.id)], immutable=True)
        for mp in plans
        if mp.status != MixPlanStatus.ready
    ]
    stitch_sig = celery_app.signature(
        STITCH_TASK, args=[str(queue_id)], immutable=True
    )
    if render_tasks:
        # Set-level pass first (one LLM call assigning each pair a style
        # hint), then the per-pair renders, then the stitch. plan_set is
        # failure-tolerant — on any error it logs and returns, and the
        # renders proceed without hints.
        plan_set_sig = celery_app.signature(
            PLAN_SET_TASK, args=[str(queue_id)], immutable=True
        )
        chain(plan_set_sig, chord(render_tasks, stitch_sig)).delay()
    else:
        stitch_sig.delay()


def _upsert_render_row(queue_id: uuid.UUID, db: Session) -> None:
    """Create the 1:1 ``QueueRender`` row (or reset it to ``pending``)."""
    row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_id))
    if row is None:
        row = QueueRender(queue_id=queue_id)
        db.add(row)
    else:
        row.status = QueueRenderStatus.pending
        row.error_text = None
    db.commit()


def reset_and_dispatch_stitch(queue_id: uuid.UUID, db: Session) -> None:
    """Force path used by the manual ``POST /stitch`` endpoint.

    (Re)sets the render row to ``pending`` and fires the chord
    unconditionally — the user explicitly asked to (re)render the mix, and
    ``render_transition`` / ``stitch_queue`` each gate on song-readiness
    themselves and fail loudly if something isn't ready.
    """
    _upsert_render_row(queue_id, db)
    _dispatch_chord(queue_id, db)


def maybe_dispatch_stitch(queue_id: uuid.UUID, db: Session) -> bool:
    """Eager path: dispatch the stitch chord IFF it's safe and not already
    running. Returns True if it dispatched.

    Safe to call from anywhere a song might have just reached ``ready`` —
    it self-gates on:

      * the queue exists and is locked,
      * it has >= 2 songs (a single song has no transition / mix),
      * EVERY song is ``ready`` (so every pair's render is renderable),
      * no stitch is already pending/rendering/ready (idempotency latch).

    Concurrency note: the native worker runs the heavy ML tasks effectively
    serially, so the check-then-act on the latch is sufficient for this
    single-user app. A second concurrent caller that slipped past the latch
    would at worst re-render already-ready pairs (each ``render_transition``
    re-claims atomically) — wasteful but not corrupting.
    """
    queue = db.get(Queue, queue_id)
    if queue is None or not queue.locked:
        return False

    items = sorted(queue.items, key=lambda it: it.position)
    if len(items) < 2:
        return False

    song_ids = [it.song_id for it in items]
    statuses = db.scalars(
        select(Song.status).where(Song.id.in_(song_ids))
    ).all()
    if len(statuses) != len(song_ids) or any(
        s != SongStatus.ready for s in statuses
    ):
        return False

    row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_id))
    if row is not None and row.status in _LATCHED:
        return False

    _upsert_render_row(queue_id, db)
    _dispatch_chord(queue_id, db)
    logger.info("auto_stitch: dispatched eager stitch for queue %s", queue_id)
    return True


def maybe_dispatch_stitch_for_song(song_id: uuid.UUID, db: Session) -> bool:
    """Find the locked queue containing ``song_id`` and try the eager stitch.

    Called from the terminal ``ready`` transition of ``transcribe_song``.
    Returns True if a stitch was dispatched.
    """
    from app.models import QueueItem

    queue = db.scalar(
        select(Queue)
        .join(QueueItem, QueueItem.queue_id == Queue.id)
        .where(QueueItem.song_id == song_id)
        .where(Queue.locked.is_(True))
        .order_by(Queue.created_at.desc())
    )
    if queue is None:
        return False
    return maybe_dispatch_stitch(queue.id, db)
