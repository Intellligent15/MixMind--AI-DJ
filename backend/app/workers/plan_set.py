"""Set-level planning pass.

The v1 prompt demanded "variety across the set" while each per-pair call
saw exactly one pair — the model had no idea what the previous
transition was. This task runs ONCE per locked queue, before any
per-pair render: a single LLM call sees the whole ordered queue
(titles, artists, BPMs, keys, energy shape) and assigns each adjacent
pair a suggested transition style plus the set's energy arc. The
suggestion lands in MixPlan.style_hint; the per-pair planner treats it
as a strong default and the user's style_override (if any) still wins.

Failure-tolerant by design: any error logs and returns — per-pair
planning works fine without hints, just with less set-level coherence.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import Analysis, MixPlan, Queue, Song
from app.services.llm import get_llm_provider
from app.services.llm.prompts import SET_PLAN_SYSTEM_PROMPT, set_plan_user_prompt
from app.services.mixer.decision import TransitionStyle
from app.workers import celery_app

logger = logging.getLogger(__name__)


def _peak_energy_position(energy_curve: list[float]) -> float | None:
    if not energy_curve:
        return None
    peak_idx = max(range(len(energy_curve)), key=lambda i: energy_curve[i])
    return round(peak_idx / max(1, len(energy_curve) - 1), 2)


@celery_app.task(name="app.workers.plan_set.plan_set")
def plan_set(queue_id: str) -> str | None:
    """Write style_hint onto each of the queue's MixPlan rows."""
    if not settings.use_llm_planner or settings.planner_version != "v2":
        return None
    try:
        return _plan_set_inner(uuid.UUID(queue_id))
    except Exception as exc:  # never block the render chord on this
        logger.error("plan_set: failed for queue %s: %s", queue_id, exc)
        return None


def _plan_set_inner(queue_uuid: uuid.UUID) -> str | None:
    with SessionLocal() as db:
        queue = db.get(Queue, queue_uuid)
        if queue is None or not queue.locked:
            return None
        items = sorted(queue.items, key=lambda it: it.position)
        if len(items) < 2:
            return None

        songs_payload: list[dict] = []
        for idx, item in enumerate(items):
            song = db.get(Song, item.song_id)
            analysis = db.scalar(
                select(Analysis).where(Analysis.song_id == item.song_id)
            )
            if song is None or analysis is None:
                logger.info("plan_set: song %s not analyzed yet; skipping pass",
                            item.song_id)
                return None
            songs_payload.append({
                "index": idx,
                "title": song.title,
                "artist": song.artist,
                "bpm": analysis.bpm,
                "key": analysis.key,
                "camelot_key": analysis.camelot_key,
                "duration": round(song.duration_seconds or 0.0, 1),
                "peak_energy_position": _peak_energy_position(
                    list(analysis.energy_curve or [])
                ),
            })

        plans = db.scalars(
            select(MixPlan).where(MixPlan.queue_id == queue_uuid)
        ).all()
        song_order = {item.song_id: idx for idx, item in enumerate(items)}
        plan_by_pair_index = {
            song_order[p.from_song_id]: p
            for p in plans
            if p.from_song_id in song_order
        }

    provider = get_llm_provider()
    obj = asyncio.run(
        provider.complete_json(
            system=SET_PLAN_SYSTEM_PROMPT,
            user=set_plan_user_prompt(songs_payload),
            cache_namespace="set_plan_logs",
        )
    )
    if not isinstance(obj, dict) or not isinstance(obj.get("pairs"), list):
        logger.warning("plan_set: malformed response, skipping hints")
        return None

    legal = {s.value for s in TransitionStyle}
    hints: dict[int, str] = {}
    for entry in obj["pairs"]:
        if not isinstance(entry, dict):
            continue
        idx, style = entry.get("index"), entry.get("style")
        if isinstance(idx, int) and isinstance(style, str) and style in legal:
            hints[idx] = style

    if not hints:
        return None

    with SessionLocal() as db:
        wrote = 0
        for idx, style in hints.items():
            stale = plan_by_pair_index.get(idx)
            if stale is None:
                continue
            row = db.get(MixPlan, stale.id)
            if row is None:
                continue
            row.style_hint = style
            wrote += 1
        db.commit()
    arc = obj.get("arc")
    logger.info(
        "plan_set: wrote %d style hints for queue %s (arc: %s)",
        wrote, queue_uuid, arc,
    )
    return str(queue_uuid)
