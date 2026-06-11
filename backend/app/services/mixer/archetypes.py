"""Archetype expander: TransitionDecision → exact MixPlanJSON.

This is the deterministic half of planner v2. The LLM chooses *what*
(candidates, style, a couple of knobs); this module computes *how* —
every timestamp, every tempo-ramp boundary, every pitch decision — using
the same math as the battle-tested deterministic planner. The output
satisfies the executor's invariants by construction:

  * exactly one set_transition_window, first;
  * exactly four crossfade_stem calls covering vocals/drums/bass/other;
  * seams at downbeats within the headroom budget;
  * tempo ramp / temporary pitch return strictly after the crossfade;
  * never a permanent pitch_shift.

Pure module: no I/O, no DB, no settings.
"""

from __future__ import annotations

import logging

from app.services.mixer.candidates import PairCandidates, SeamCandidate
from app.services.mixer.decision import (
    TransitionDecision,
    TransitionExtra,
    TransitionStyle,
)
from app.services.mixer.plan import compute_pitch_shift
from app.services.mixer.types import AnalysisBundle, MixPlanJSON

logger = logging.getLogger(__name__)

STEMS = ("vocals", "drums", "bass", "other")

# Tempo/pitch settle windows after the crossfade, mirroring v1.
TEMPO_RAMP_BARS = 16
PITCH_RETURN_BARS = 4
# Pitch shifts past ±2 semitones trade a key match for pyrubberband
# artifacts that sound worse than the clash. Hard cap.
PITCH_SHIFT_CAP = 2
# Bars of A's bass killed early for the bass_kill extra.
BASS_KILL_BARS = 4
# stutter_buildup loop parameters: half-beat slices repeated 8 times = 1
# bar of stutter at A's tempo, right before the seam.
STUTTER_BEAT_FRACTION = 0.5
STUTTER_REPEATS = 8
# vinyl_stop brake length.
VINYL_STOP_BARS = 1.5


class ArchetypeError(ValueError):
    """Raised when a decision can't be expanded (bad candidate id, no
    overlap room, …). Callers treat it like an invalid LLM plan."""


def _sec_per_bar(bundle: AnalysisBundle) -> float:
    if not bundle.bpm:
        raise ArchetypeError("song has no bpm")
    return (60.0 / bundle.bpm) * bundle.time_signature


def camelot_compatible(a_camelot: str | None, b_camelot: str | None) -> bool:
    """Equal or adjacent on the Camelot wheel (same number other letter,
    or ±1 number same letter) → harmonically mixable, no pitch shift.

    The Camelot wheel is the DJ shorthand for key compatibility: '8A' is
    A minor, '8B' is C major; neighbours share most of their notes, so
    blending them sounds consonant. Unknown keys → assume compatible
    (a wrong no-op beats a wrong shift).
    """
    if not a_camelot or not b_camelot:
        return True
    try:
        a_num, a_letter = int(a_camelot[:-1]), a_camelot[-1].upper()
        b_num, b_letter = int(b_camelot[:-1]), b_camelot[-1].upper()
    except (ValueError, IndexError):
        return True
    if a_num == b_num:
        return True  # same or relative major/minor
    if a_letter == b_letter and (a_num - b_num) % 12 in (1, 11):
        return True
    return False


def _clamp_duration_bars(
    duration_bars: int,
    a: AnalysisBundle,
    b: AnalysisBundle,
    seam_a: float,
    seam_b: float,
) -> int:
    """Shrink the crossfade if either song lacks the room — same clamp the
    deterministic planner applies, so the executor never has to."""
    spb_a = _sec_per_bar(a)
    stretch = b.bpm / a.bpm if a.bpm and b.bpm else 1.0
    available_a = a.duration - seam_a
    available_b_stretched = (b.duration - seam_b) * stretch
    clamped = min(
        duration_bars,
        int(available_a / spb_a),
        int(available_b_stretched / spb_a),
    )
    if clamped < 2:
        raise ArchetypeError(
            f"no overlap room at chosen seams (a={available_a:.1f}s, "
            f"b_stretched={available_b_stretched:.1f}s)"
        )
    return clamped


def _crossfade_calls(
    duration_bars: int,
    a_fade_out_bars: int,
    *,
    drums_start_bar: int = 0,
    drums_duration_bars: int | None = None,
) -> list[dict]:
    """The four stem crossfades. `drums_*` lets drum_bridge offset the
    drum stem; everything else shares the main window."""
    calls = []
    for stem in STEMS:
        if stem == "drums" and (drums_start_bar or drums_duration_bars):
            calls.append({
                "tool": "crossfade_stem", "stem": stem,
                "from_song": "A", "to_song": "B",
                "start_bar": drums_start_bar,
                "duration_bars": drums_duration_bars or duration_bars,
                "curve": "equal_power",
            })
            continue
        call = {
            "tool": "crossfade_stem", "stem": stem,
            "from_song": "A", "to_song": "B",
            "start_bar": 0, "duration_bars": duration_bars,
            "curve": "equal_power",
        }
        if a_fade_out_bars < duration_bars:
            call["a_fade_out_bars"] = a_fade_out_bars
        calls.append(call)
    return calls


def _tempo_and_pitch_calls(
    a: AnalysisBundle,
    b: AnalysisBundle,
    seam_b: float,
    crossfade_total_bars: int,
) -> list[dict]:
    """Tempo ramp + temporary pitch return, both anchored strictly AFTER
    the bar where the last stem finishes its fade. Computed, not prompted."""
    calls: list[dict] = []
    spb_b = _sec_per_bar(b)
    crossfade_end_b = seam_b + crossfade_total_bars * spb_b

    needs_ramp = a.bpm and b.bpm and abs(a.bpm - b.bpm) / a.bpm > 0.02
    ramp_end_b = crossfade_end_b + TEMPO_RAMP_BARS * spb_b
    if needs_ramp and b.duration >= ramp_end_b:
        calls.append({
            "tool": "set_tempo_ramp", "song": "B",
            "start_time": round(crossfade_end_b, 3),
            "end_time": round(ramp_end_b, 3),
            "start_bpm": a.bpm, "end_bpm": b.bpm,
        })
    elif needs_ramp:
        logger.info(
            "archetypes: B too short for a tempo ramp (%.1fs < %.1fs); "
            "B stays at A's tempo", b.duration, ramp_end_b,
        )

    if not camelot_compatible(a.camelot_key, b.camelot_key):
        try:
            n_steps = compute_pitch_shift(a.key, b.key)
        except ValueError:
            # Unparseable key string — skip pitch handling rather than
            # killing the whole expansion (a wrong no-op beats no plan).
            n_steps = 0
        n_steps = max(-PITCH_SHIFT_CAP, min(PITCH_SHIFT_CAP, n_steps))
        return_end_b = crossfade_end_b + PITCH_RETURN_BARS * spb_b
        if n_steps != 0 and b.duration >= return_end_b:
            calls.append({
                "tool": "temporary_pitch_shift", "song": "B",
                "start_time": round(crossfade_end_b, 3),
                "semitones": n_steps,
                "fade_in_bars": 0, "hold_bars": 0,
                "fade_out_bars": PITCH_RETURN_BARS,
            })
    return calls


def expand(
    decision: TransitionDecision,
    a: AnalysisBundle,
    b: AnalysisBundle,
    candidates: PairCandidates,
) -> MixPlanJSON:
    """Expand a validated decision into the final tool-call list."""
    out_c = candidates.find(decision.out)
    in_c = candidates.find(decision.in_)
    if out_c is None or not decision.out.startswith("A"):
        raise ArchetypeError(f"unknown OUT candidate {decision.out!r}")
    if in_c is None or not decision.in_.startswith("B"):
        raise ArchetypeError(f"unknown IN candidate {decision.in_!r}")

    seam_a, seam_b = out_c.time, in_c.time
    duration = _clamp_duration_bars(
        decision.normalized_duration(), a, b, seam_a, seam_b
    )
    a_fade = decision.normalized_a_fade(duration)
    spb_a = _sec_per_bar(a)

    style = decision.style
    drums_start, drums_dur = 0, None
    pre_window_calls: list[dict] = []   # effects placed before/around the seam

    if style == TransitionStyle.drop_swap:
        # A snaps out fast; coupled short fade reads as an instant swap.
        a_fade = min(a_fade, duration)
    elif style == TransitionStyle.drum_bridge:
        # Drums hold longest: they start late on the grid but run past the
        # other stems, bridging the grooves (mirrors the classic shape).
        bridge = max(4, duration // 2)
        drums_start = min(bridge, duration - 2)
        drums_dur = duration + bridge - drums_start
        # Keep total within the clamp budget.
        total = drums_start + drums_dur
        room = _clamp_duration_bars(total, a, b, seam_a, seam_b)
        if total > room:
            drums_dur = max(2, room - drums_start)
    elif style == TransitionStyle.wash_out:
        pre_window_calls.append({
            "tool": "apply_reverb", "song": "A",
            "start_time": round(seam_a, 3),
            "tail_duration_bars": float(max(2, a_fade // 2)),
            "wet_level": 0.8, "bpm": a.bpm,
        })
        if TransitionExtra.filter_sweep_out not in decision.extras:
            decision.extras.append(TransitionExtra.filter_sweep_out)
    elif style == TransitionStyle.stutter_buildup:
        if out_c.vocal_safe:
            stutter_start = max(0.0, seam_a - spb_a)  # last bar before the seam
            pre_window_calls.append({
                "tool": "loop_section", "song": "A",
                "start_time": round(stutter_start, 3),
                "beats": STUTTER_BEAT_FRACTION,
                "repeats": STUTTER_REPEATS, "bpm": a.bpm,
            })
        else:
            logger.info(
                "archetypes: OUT point not vocal-safe; dropping stutter loop"
            )
    elif style == TransitionStyle.vinyl_stop:
        pre_window_calls.append({
            "tool": "turntable_stop", "song": "A",
            "start_time": round(seam_a, 3),
            "duration_bars": VINYL_STOP_BARS, "bpm": a.bpm,
        })
        a_fade = min(a_fade, 2)

    extra_calls: list[dict] = []
    for extra in decision.extras:
        if extra == TransitionExtra.bass_kill:
            kill_start = max(0.0, seam_a - BASS_KILL_BARS * spb_a)
            extra_calls.append({
                "tool": "volume_fade", "song": "A", "stem": "bass",
                "start_time": round(kill_start, 3),
                "duration_bars": float(BASS_KILL_BARS),
                "start_gain": 1.0, "end_gain": 0.0, "bpm": a.bpm,
            })
        elif extra == TransitionExtra.filter_sweep_out:
            sweep_end = seam_a + a_fade * spb_a
            extra_calls.append({
                "tool": "filter_sweep", "song": "A", "type": "lowpass",
                "start_time": round(seam_a, 3),
                "end_time": round(sweep_end, 3),
                "start_cutoff_hz": 20000.0, "end_cutoff_hz": 120.0,
            })
        elif extra == TransitionExtra.echo_tail:
            echo_start = seam_a + a_fade * spb_a
            extra_calls.append({
                "tool": "echo_out", "song": "A",
                "start_time": round(echo_start, 3),
                "beats": 4, "feedback": 0.5, "bpm": a.bpm,
            })
        elif extra == TransitionExtra.reverb_tail:
            if style != TransitionStyle.wash_out:  # wash_out already has one
                extra_calls.append({
                    "tool": "apply_reverb", "song": "A",
                    "start_time": round(seam_a, 3),
                    "tail_duration_bars": 4.0,
                    "wet_level": 0.6, "bpm": a.bpm,
                })

    stem_calls = _crossfade_calls(
        duration, a_fade,
        drums_start_bar=drums_start, drums_duration_bars=drums_dur,
    )
    crossfade_total_bars = max(
        int(c["start_bar"]) + int(c["duration_bars"]) for c in stem_calls
    )

    plan: MixPlanJSON = [
        {
            "tool": "set_transition_window",
            "from_song_time_start": round(seam_a, 3),
            "to_song_time_start": round(seam_b, 3),
            "duration_bars": duration,
        },
        *pre_window_calls,
        *extra_calls,
        *_tempo_and_pitch_calls(a, b, seam_b, crossfade_total_bars),
        *stem_calls,
    ]
    return plan


def default_decision(candidates: PairCandidates, style: TransitionStyle | None = None) -> TransitionDecision:
    """A sensible decision when the LLM is unavailable but a style was
    pinned (e.g. user override): latest OUT, earliest IN, default knobs."""
    if not candidates.out_candidates or not candidates.in_candidates:
        raise ArchetypeError("no seam candidates available")
    chosen = style or TransitionStyle.smooth_blend
    # Prefer a high-energy IN for energetic styles; first candidate else.
    in_c: SeamCandidate = candidates.in_candidates[0]
    if chosen in (TransitionStyle.drop_swap, TransitionStyle.stutter_buildup):
        for c in candidates.in_candidates:
            if c.energy >= 0.8:
                in_c = c
                break
    from app.services.mixer.decision import STYLE_DURATION_CHOICES

    duration = STYLE_DURATION_CHOICES[chosen][-1]
    return TransitionDecision(
        **{
            "out": candidates.out_candidates[-1].id,
            "in": in_c.id,
            "style": chosen,
            "duration_bars": duration,
            "a_fade_out_bars": max(1, duration // 2),
            "rationale": "default expansion (no LLM decision available)",
        }
    )
