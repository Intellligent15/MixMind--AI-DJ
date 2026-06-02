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
    "filter_sweep",
    "echo_out",
    "loop_section",
    "swap_stem",
}
_CANONICAL_STEMS = {"vocals", "drums", "bass", "other"}
_LEGAL_SONG_REFS = {"A", "B"}
# Which fields on each tool carry a song reference. The set is small
# enough that an explicit map beats reflection over the TypedDicts.
_SONG_FIELDS_BY_TOOL = {
    "crossfade_stem": ("from_song", "to_song"),
    "pitch_shift": ("song",),
    "temporary_pitch_shift": ("song",),
    "set_tempo_ramp": ("song",),
    "filter_sweep": ("song",),
    "echo_out": ("song",),
    "loop_section": ("song",),
    "swap_stem": ("from_song", "to_song"),
}


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
    # Song refs must be the single-char strings the executor expects.
    # Models like to drift to "Song A" / "Song B" if the prompt
    # introduces the songs that way; reject so we fall back instead of
    # silently rendering with a misnamed field.
    for call in plan:
        for field in _SONG_FIELDS_BY_TOOL.get(call.get("tool"), ()):
            val = call.get(field)
            if val not in _LEGAL_SONG_REFS:
                raise ValueError(
                    f"{call['tool']}.{field}={val!r} must be 'A' or 'B'"
                )


# Maximum crossfade length the deterministic planner uses; we treat
# it as the worst case the LLM might ask for and reserve room for it.
_MAX_CROSSFADE_BARS = 16
# Safety buffer (seconds) past the crossfade end. Absorbs stem-WAV
# drift vs `Song.duration_seconds` metadata (Demucs trims/pads, yt-dlp
# rounds), which we've seen reach ~25s in pathological cases.
_SEAM_SAFETY_SECONDS = 5.0


def _max_seam_time(duration: float, bpm: float, time_signature: int) -> float:
    """Latest seam time that leaves room for a full-length crossfade.

    Returned in original-song seconds. Clamped to 0 if the song is
    shorter than the reserved tail — in that case the validator will
    reject any LLM plan and trigger the deterministic fallback (which
    handles short songs via its own duration-bars clamp).
    """
    if not bpm or not duration:
        return 0.0
    sec_per_bar = (60.0 / bpm) * time_signature
    return max(0.0, duration - _MAX_CROSSFADE_BARS * sec_per_bar - _SEAM_SAFETY_SECONDS)


def _enrich_sections(sections: list[dict], energy_curve: list[float]) -> list[dict]:
    """Annotate each section with mean energy, normalized 0..1 over the song.

    The analyzer's section `label`s are opaque cluster IDs (`section_1`
    …`section_5`) — they tell the LLM nothing about musical role. The
    energy curve is sampled at 1 Hz, so its index ≈ second; we average
    it over each section's span and normalize to the song's hottest
    section. The LLM uses that to read structure (low = intro/breakdown,
    ~1.0 = drop/chorus), pick a style, and place a musical seam. We drop
    the meaningless `label` and the standalone energy_curve in favor of
    this.
    """
    if not sections:
        return []
    n = len(energy_curve)
    raw: list[float] = []
    for s in sections:
        lo = int(s["start"])
        hi = max(lo + 1, int(round(s["end"])))
        window = energy_curve[lo:hi] if n else []
        raw.append(sum(window) / len(window) if window else 0.0)
    peak = max(raw) or 1.0
    return [
        {"start": round(s["start"], 1), "end": round(s["end"], 1),
         "energy": round(e / peak, 2)}
        for s, e in zip(sections, raw)
    ]


def _validate_seam_headroom(
    plan: list[dict], a: AnalysisBundle, b: AnalysisBundle
) -> None:
    """Reject plans whose seam + crossfade extends past either song.

    The executor will gracefully clamp short overlaps, but a clamp from
    ~30s down to ~4s produces a perceptually abrupt cut — defeats the
    point of a DJ-style transition. Reject so we fall back to the
    deterministic planner, which picks seams in A's outro section
    (where there's room by construction).
    """
    window = next((c for c in plan if c.get("tool") == "set_transition_window"), None)
    if window is None:
        return  # _validate_llm_plan already rejected
    duration_bars = window.get("duration_bars", 0)
    from_t = window.get("from_song_time_start", 0.0)
    to_t = window.get("to_song_time_start", 0.0)
    sec_per_bar_a = (60.0 / a.bpm) * a.time_signature if a.bpm else 0.0
    sec_per_bar_b = (60.0 / b.bpm) * b.time_signature if b.bpm else 0.0
    crossfade_a = duration_bars * sec_per_bar_a
    crossfade_b = duration_bars * sec_per_bar_b
    if from_t + crossfade_a > a.duration - _SEAM_SAFETY_SECONDS:
        raise ValueError(
            f"A's seam leaves too little headroom: "
            f"from={from_t:.1f}s + crossfade={crossfade_a:.1f}s > "
            f"duration={a.duration:.1f}s - safety={_SEAM_SAFETY_SECONDS}s"
        )
    if to_t + crossfade_b > b.duration - _SEAM_SAFETY_SECONDS:
        raise ValueError(
            f"B's seam leaves too little headroom: "
            f"to={to_t:.1f}s + crossfade={crossfade_b:.1f}s > "
            f"duration={b.duration:.1f}s - safety={_SEAM_SAFETY_SECONDS}s"
        )


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
        #
        # `energy_curve` is sampled at 1Hz in the analyzer. We don't send
        # the raw curve to the LLM (no timestamps → unusable); instead we
        # average it per section in `_enrich_sections` so structure and
        # energy arrive together.
        a_energy_curve = list(a_analysis.energy_curve or [])
        b_energy_curve = list(b_analysis.energy_curve or [])
        # Per-word lyrics + raw Whisper segments + vocal_segments are
        # deliberately NOT carried into the LLM input. Reasons:
        #   - `vocal_safe_regions` already distills "where can/can't I
        #     cut" from those signals; sending the raw text on top is
        #     redundant and was the dominant cost in our prompts (we
        #     hit Groq's 8K TPM limit at ~35K tokens).
        #   - The LLM plans transitions, not lyric matching, so word-
        #     level timestamps don't change its decisions.
        # If a future phase wants lyrics for clever same-word vocal
        # swaps, add a separate summary field (e.g. chorus hook lyric)
        # — don't dump the full alignment.

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

    def _bundle_to_llm_dict(bundle: AnalysisBundle, energy_curve) -> dict:
        sec_per_bar = (
            round((60.0 / bundle.bpm) * bundle.time_signature, 3)
            if bundle.bpm else 0.0
        )
        return {
            "bpm": bundle.bpm,
            "key": bundle.key,
            "camelot_key": bundle.camelot_key,
            "time_signature": bundle.time_signature,
            # Bar length so the LLM can convert section times <-> bars
            # without us shipping the full downbeats array (which was
            # ~40% of the prompt and pure noise).
            "seconds_per_bar": sec_per_bar,
            "duration": bundle.duration,
            # Sections carry their normalized mean energy (see
            # _enrich_sections) — this is the LLM's structural map.
            "sections": _enrich_sections(bundle.sections, energy_curve),
            # Latest seam time that leaves a full 16-bar crossfade +
            # 5s safety buffer in the song. The LLM must keep its
            # seam at or before this value; the validator rejects
            # plans that violate it.
            "max_seam_time": _max_seam_time(
                bundle.duration, bundle.bpm, bundle.time_signature
            ),
        }

    async def _build_plan() -> list[dict]:
        if not settings.use_llm_planner:
            return build_pair_plan(a_bundle, b_bundle)

        a_regions = await _fetch_safe_regions(a_transcription, a_lyrics, a_envelope_path, a.duration_seconds)
        b_regions = await _fetch_safe_regions(b_transcription, b_lyrics, b_envelope_path, b.duration_seconds)

        a_llm_input = {
            "analysis": _bundle_to_llm_dict(a_bundle, a_energy_curve),
            "vocal_safe_regions": a_regions,
        }
        b_llm_input = {
            "analysis": _bundle_to_llm_dict(b_bundle, b_energy_curve),
            "vocal_safe_regions": b_regions,
        }

        # Provide tools schema definition to the LLM. `beat_grid` is
        # intentionally omitted from per-song inputs — `downbeats` is
        # enough for the LLM to reason about bars and avoids hundreds
        # of redundant floats per prompt.
        tools_schema = json.dumps([
            {"tool": "set_transition_window", "from_song_time_start": "float", "to_song_time_start": "float", "duration_bars": "int"},
            {"tool": "crossfade_stem", "stem": "str", "from_song": "str", "to_song": "str", "start_bar": "int", "duration_bars": "int", "curve": "str", "a_fade_out_bars": "int (optional; bars over which A fades out, <= duration_bars; default duration_bars)"},
            {"tool": "pitch_shift", "song": "str", "semitones": "float"},
            {"tool": "temporary_pitch_shift", "song": "str", "start_time": "float", "semitones": "float", "fade_in_bars": "int", "hold_bars": "int", "fade_out_bars": "int"},
            {"tool": "set_tempo_ramp", "song": "str", "start_time": "float", "end_time": "float", "start_bpm": "float", "end_bpm": "float"},
            {"tool": "filter_sweep", "song": "str", "type": "str (lowpass|highpass)", "start_time": "float", "end_time": "float", "start_cutoff_hz": "float", "end_cutoff_hz": "float"},
            {"tool": "echo_out", "song": "str", "start_time": "float", "beats": "int", "feedback": "float (0..0.9)", "bpm": "float"},
            {"tool": "loop_section", "song": "str", "start_time": "float", "beats": "int", "repeats": "int", "bpm": "float"},
            {"tool": "swap_stem", "from_song": "str", "to_song": "str", "stem": "str (vocals|drums|bass|other)", "time": "float (output-timeline seconds)"},
        ], indent=2)

        provider = get_llm_provider()
        try:
            plan = await provider.plan_transition(a_llm_input, b_llm_input, tools_schema)
            _validate_llm_plan(plan)
            _validate_seam_headroom(plan, a_bundle, b_bundle)
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
