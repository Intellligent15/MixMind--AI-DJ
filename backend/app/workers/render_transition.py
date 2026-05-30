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
    Transcription,
)
from app.models.lyrics import Lyrics, LyricsAlignmentStatus
from app.services.llm import get_llm_provider
from app.core.config import settings
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

# Permanent pitch shifts beyond ±2 semitones produce pyrubberband artifacts
# that outweigh the harmonic benefit (see build_pair_plan's matching cap).
# Enforced post-LLM since the model can still emit larger values despite
# being told the limit in the system prompt.
PERMANENT_PITCH_SHIFT_CAP = 2

_LEGAL_TOOLS = {
    "set_transition_window",
    "set_tempo_ramp",
    "temporary_pitch_shift",
    "pitch_shift",
    "crossfade_stem",
}
_CANONICAL_STEMS = {"vocals", "drums", "bass", "other"}


def _validate_llm_plan(plan: list[dict]) -> None:
    """Raise ValueError if the LLM's plan won't survive the executor.

    Checks the same invariants the executor enforces, up-front, so a
    malformed plan triggers the deterministic fallback instead of
    failing the render and stranding the row in `failed`.
    """
    if not isinstance(plan, list) or not plan:
        raise ValueError("LLM plan is not a non-empty list")
    windows = [c for c in plan if c.get("tool") == "set_transition_window"]
    if len(windows) != 1:
        raise ValueError(f"expected 1 set_transition_window, got {len(windows)}")
    stem_calls = [c for c in plan if c.get("tool") == "crossfade_stem"]
    if len(stem_calls) != 4:
        raise ValueError(f"expected 4 crossfade_stem calls, got {len(stem_calls)}")
    if {c.get("stem") for c in stem_calls} != _CANONICAL_STEMS:
        raise ValueError("crossfade_stem calls must cover vocals/drums/bass/other")
    illegal = {c.get("tool") for c in plan} - _LEGAL_TOOLS
    if illegal:
        raise ValueError(f"illegal tools in plan: {sorted(illegal)}")


def _clamp_pitch_shifts(plan: list[dict]) -> list[dict]:
    """Clamp permanent `pitch_shift.semitones` to ±PERMANENT_PITCH_SHIFT_CAP."""
    clamped = []
    for call in plan:
        if call.get("tool") == "pitch_shift":
            n = call.get("semitones", 0)
            capped = max(-PERMANENT_PITCH_SHIFT_CAP, min(PERMANENT_PITCH_SHIFT_CAP, n))
            if capped != n:
                logger.warning(
                    "render_transition: clamping LLM permanent pitch_shift %s -> %s",
                    n, capped,
                )
                call = {**call, "semitones": capped}
        clamped.append(call)
    return clamped


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

        # Fetch optional inputs for LLM planning
        a_transcription = db.scalar(select(Transcription).where(Transcription.song_id == a.id))
        b_transcription = db.scalar(select(Transcription).where(Transcription.song_id == b.id))
        a_lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == a.id))
        b_lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == b.id))

        a_bundle = _to_bundle(a_analysis, a.duration_seconds)
        b_bundle = _to_bundle(b_analysis, b.duration_seconds)
        existing_plan_json = row.plan_json
        a_audio_key = a.audio_path
        b_audio_key = b.audio_path

        # Capture vocal envelope paths to load via storage
        a_envelope_path = a_stems.vocal_envelope_path
        b_envelope_path = b_stems.vocal_envelope_path

        # LLM-only inputs that don't ride through AnalysisBundle (which
        # the executor consumes and intentionally stays narrow). Snapshot
        # them here so the rest of the worker can stay session-free.
        a_energy_curve = list(a_analysis.energy_curve or [])
        b_energy_curve = list(b_analysis.energy_curve or [])
        a_vocal_segments = list(a_analysis.vocal_segments or [])
        b_vocal_segments = list(b_analysis.vocal_segments or [])
        a_raw_segments = list(a_transcription.segments) if a_transcription else None
        b_raw_segments = list(b_transcription.segments) if b_transcription else None
        a_alignment_ok = (
            a_lyrics is not None
            and a_lyrics.alignment_status == LyricsAlignmentStatus.success
        )
        b_alignment_ok = (
            b_lyrics is not None
            and b_lyrics.alignment_status == LyricsAlignmentStatus.success
        )
        a_aligned_words = a_lyrics.aligned_words if a_alignment_ok else None
        b_aligned_words = b_lyrics.aligned_words if b_alignment_ok else None

    import json
    from app.services.vocal_safety.safety import vocal_safe_regions

    async def _fetch_safe_regions(transcription, lyrics, envelope_path, duration):
        if not transcription or not envelope_path:
            return []
        try:
            envelope_data = await storage.read(envelope_path)
            envelope = json.loads(envelope_data.decode("utf-8"))
        except Exception:
            return []
        
        aligned_words = None
        if lyrics and lyrics.alignment_status == LyricsAlignmentStatus.success:
            aligned_words = lyrics.aligned_words
            
        return vocal_safe_regions(
            transcription_segments=transcription.segments,
            envelope=envelope,
            aligned_words=aligned_words,
            duration_seconds=duration or 0.0,
        )

    def _bundle_to_llm_dict(bundle: AnalysisBundle, energy_curve, vocal_segments) -> dict:
        return {
            "bpm": bundle.bpm,
            "key": bundle.key,
            "camelot_key": bundle.camelot_key,
            "time_signature": bundle.time_signature,
            "downbeats": bundle.downbeats,
            "sections": bundle.sections,
            "duration": bundle.duration,
            "energy_curve": energy_curve,
            "vocal_segments": vocal_segments,
        }

    async def _build_plan() -> list[dict]:
        if not settings.use_llm_planner:
            return build_pair_plan(a_bundle, b_bundle)

        a_regions = await _fetch_safe_regions(a_transcription, a_lyrics, a_envelope_path, a.duration_seconds)
        b_regions = await _fetch_safe_regions(b_transcription, b_lyrics, b_envelope_path, b.duration_seconds)

        a_llm_input = {
            "analysis": _bundle_to_llm_dict(a_bundle, a_energy_curve, a_vocal_segments),
            "aligned_lyrics": a_aligned_words,
            "raw_transcription": None if a_alignment_ok else a_raw_segments,
            "vocal_safe_regions": a_regions,
        }
        b_llm_input = {
            "analysis": _bundle_to_llm_dict(b_bundle, b_energy_curve, b_vocal_segments),
            "aligned_lyrics": b_aligned_words,
            "raw_transcription": None if b_alignment_ok else b_raw_segments,
            "vocal_safe_regions": b_regions,
        }

        # Provide tools schema definition to the LLM. `beat_grid` is
        # intentionally omitted from per-song inputs — `downbeats` is
        # enough for the LLM to reason about bars and avoids hundreds
        # of redundant floats per prompt.
        tools_schema = json.dumps([
            {"tool": "set_transition_window", "from_song_time_start": "float", "to_song_time_start": "float", "duration_bars": "int"},
            {"tool": "crossfade_stem", "stem": "str", "from_song": "str", "to_song": "str", "start_bar": "int", "duration_bars": "int", "curve": "str"},
            {"tool": "pitch_shift", "song": "str", "semitones": "float"},
            {"tool": "temporary_pitch_shift", "song": "str", "start_time": "float", "semitones": "float", "fade_in_bars": "int", "hold_bars": "int", "fade_out_bars": "int"},
            {"tool": "set_tempo_ramp", "song": "str", "start_time": "float", "end_time": "float", "start_bpm": "float", "end_bpm": "float"},
        ], indent=2)

        provider = get_llm_provider()
        try:
            plan = await provider.plan_transition(a_llm_input, b_llm_input, tools_schema)
            _validate_llm_plan(plan)
            return _clamp_pitch_shifts(plan)
        except Exception as exc:
            logger.error("render_transition: LLM planner failed, falling back to deterministic: %s", exc)
            return build_pair_plan(a_bundle, b_bundle)
            
    plan_json = existing_plan_json or asyncio.run(_build_plan())

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

        async def _download_original(key: str | None, prefix: str) -> str | None:
            if not key:
                return None
            dest = tmp / f"{prefix}_original.wav"
            await storage.download_file(key, dest)
            return str(dest)

        a_paths = asyncio.run(_download_stems(a_stems, "a"))
        b_paths = asyncio.run(_download_stems(b_stems, "b"))
        a_orig = asyncio.run(_download_original(a_audio_key, "a"))
        b_orig = asyncio.run(_download_original(b_audio_key, "b"))

        a_inputs = SongRenderInputs(
            stem_paths=a_paths, analysis=a_bundle, original_audio_path=a_orig
        )
        b_inputs = SongRenderInputs(
            stem_paths=b_paths, analysis=b_bundle, original_audio_path=b_orig
        )

        try:
            result = render(plan_json, a_inputs, b_inputs)
        except Exception as exc:
            logger.exception("render_transition: %s render failed", mix_plan_id)
            _mark_failed(plan_uuid, f"{type(exc).__name__}: {exc}")
            return None

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
