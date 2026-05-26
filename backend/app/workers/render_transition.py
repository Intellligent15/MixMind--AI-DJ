"""Render a MixPlan's transition into a WAV via the mixer executor.

Atomic-claim pattern mirrors separate_stems / transcribe_song: a single
UPDATE WHERE status=pending|failed|ready transitions the row to
`rendering`. Losers (status already rendering, or row missing) return
None.

`plan_json` is generated lazily on the first render so we don't burn
work for plans the user never asks to render — and so Phase 9's LLM
call (which replaces this generator) also fires lazily.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select, update

from app.core.db import SessionLocal
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Song,
    SongStatus,
    Stems,
)
from app.services.mixer.executor import render
from app.services.mixer.plan import build_pair_plan
from app.services.mixer.types import (
    AnalysisBundle,
    SongRenderInputs,
)
from app.services.storage import get_storage
from app.workers import celery_app

logger = logging.getLogger(__name__)

CLAIMABLE_STATUSES = (
    MixPlanStatus.pending,
    MixPlanStatus.failed,
    MixPlanStatus.ready,  # allows re-render of an already-rendered pair
)


def _to_bundle(analysis: Analysis, duration: float) -> AnalysisBundle:
    return AnalysisBundle(
        bpm=analysis.bpm,
        key=analysis.key,
        camelot_key=analysis.camelot_key,
        time_signature=analysis.time_signature,
        beat_grid=list(analysis.beat_grid),
        downbeats=list(analysis.downbeats),
        sections=list(analysis.sections),
        duration=duration,
    )


def _stem_paths(stems: Stems) -> dict[str, str]:
    return {
        "vocals": stems.vocals_path,
        "drums": stems.drums_path,
        "bass": stems.bass_path,
        "other": stems.other_path,
    }


@celery_app.task(name="app.workers.render_transition.render_transition")
def render_transition(mix_plan_id: str) -> str | None:
    plan_uuid = uuid.UUID(mix_plan_id)
    storage = get_storage()

    # Phase 1: load the row, validate, atomically claim.
    with SessionLocal() as db:
        row = db.get(MixPlan, plan_uuid)
        if row is None:
            logger.warning("render_transition: %s not found", mix_plan_id)
            return None

        a = db.get(Song, row.from_song_id)
        b = db.get(Song, row.to_song_id)
        if a is None or b is None:
            logger.error("render_transition: %s missing songs", mix_plan_id)
            return None
        if a.status != SongStatus.ready or b.status != SongStatus.ready:
            logger.warning(
                "render_transition: %s songs not ready (a=%s, b=%s)",
                mix_plan_id, a.status.value, b.status.value,
            )
            return None

        claim = db.execute(
            update(MixPlan)
            .where(MixPlan.id == plan_uuid)
            .where(MixPlan.status.in_(CLAIMABLE_STATUSES))
            .values(status=MixPlanStatus.rendering, error_text=None)
        )
        db.commit()
        if claim.rowcount == 0:
            db.refresh(row)
            logger.info(
                "render_transition: %s already %s, skipping",
                mix_plan_id, row.status.value,
            )
            return None

        # Snapshot what we need outside the session.
        a_analysis = db.scalar(select(Analysis).where(Analysis.song_id == a.id))
        b_analysis = db.scalar(select(Analysis).where(Analysis.song_id == b.id))
        a_stems = db.scalar(select(Stems).where(Stems.song_id == a.id))
        b_stems = db.scalar(select(Stems).where(Stems.song_id == b.id))
        if not (a_analysis and b_analysis and a_stems and b_stems):
            logger.error("render_transition: %s missing analysis/stems", mix_plan_id)
            _mark_failed(plan_uuid, "missing analysis or stems")
            return None

        a_bundle = _to_bundle(a_analysis, a.duration_seconds)
        b_bundle = _to_bundle(b_analysis, b.duration_seconds)
        existing_plan_json = row.plan_json

    plan_json = existing_plan_json or build_pair_plan(a_bundle, b_bundle)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        async def _download_stems(stems: Stems, prefix: str):
            paths = {}
            for k, key in _stem_paths(stems).items():
                dest = tmp / f"{prefix}_{k}.wav"
                await storage.download_file(key, dest)
                paths[k] = str(dest)
            return paths

        a_paths = asyncio.run(_download_stems(a_stems, "a"))
        b_paths = asyncio.run(_download_stems(b_stems, "b"))
        
        a_inputs = SongRenderInputs(stem_paths=a_paths, analysis=a_bundle)
        b_inputs = SongRenderInputs(stem_paths=b_paths, analysis=b_bundle)

        try:
            result = render(plan_json, a_inputs, b_inputs)
        except Exception as exc:
            logger.exception("render_transition: %s render failed", mix_plan_id)
            _mark_failed(plan_uuid, f"{type(exc).__name__}: {exc}")
            raise

    # Phase 4: persist output via storage, flip to ready.
    key = f"mixes/{mix_plan_id}.wav"
    asyncio.run(storage.write(key, result.wav_bytes))

    with SessionLocal() as db:
        row = db.get(MixPlan, plan_uuid)
        if row is None:
            return None
        row.plan_json = plan_json
        row.rendered_audio_path = key
        row.status = MixPlanStatus.ready
        row.error_text = None
        db.commit()
    return mix_plan_id


def _mark_failed(plan_uuid: uuid.UUID, message: str) -> None:
    with SessionLocal() as db:
        row = db.get(MixPlan, plan_uuid)
        if row is None:
            return
        row.status = MixPlanStatus.failed
        row.error_text = message[:1000]  # cap for sanity
        db.commit()
