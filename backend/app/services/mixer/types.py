"""Shared types for the mixer service.

Tool-call dicts mirror the spec's Mix Plan Schema. Dataclasses are the
shape the executor's caller assembles from the DB (Song / Analysis /
Stems rows) — keeps the executor pure (no SQLAlchemy import).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

# Tool-call dicts as the LLM (Phase 9) and our hand-built plan generator
# (Phase 7) both emit. The mixer executor walks these in order.

SongRef = Literal["A", "B"]
StemName = Literal["vocals", "drums", "bass", "other"]
CrossfadeCurve = Literal["linear", "exponential", "s_curve"]


class SetTransitionWindow(TypedDict):
    tool: Literal["set_transition_window"]
    from_song_time_start: float  # seconds in A (original time)
    to_song_time_start: float    # seconds in B (original time, pre-stretch)
    duration_bars: int


class PitchShift(TypedDict):
    """Static pitch shift applied to the entire song. Phase 9's LLM may
    emit this for harmonic matching when keys are incompatible. Phase 7's
    hand-built plan generator does NOT emit it — incompatible keys are
    left alone, since pyrubberband artifacts on shifts > 2 semitones
    can be worse than the dissonance they're trying to fix."""

    tool: Literal["pitch_shift"]
    song: SongRef
    semitones: float  # may be fractional; positive=up, negative=down


class TemporaryPitchShift(TypedDict):
    """Time-limited pitch shift that fades in, holds, and fades back to
    the song's original key. Lets the planner introduce a brief key
    excursion (e.g. shift B up 3 semitones at the seam, hold for 4 bars
    of the chorus, then glide back over 8 bars to land on B's true key).

    All bar counts are measured at the planner's target BPM (typically
    A.bpm post-stretch). Total duration of the effect on the audio is
    (fade_in_bars + hold_bars + fade_out_bars) bars at that tempo.

    NOT implemented in Phase 7's executor — Phase 9 territory."""

    tool: Literal["temporary_pitch_shift"]
    song: SongRef
    start_time: float          # seconds in original song time
    semitones: float           # peak shift; fractional allowed
    fade_in_bars: int          # bars to glide from 0 → semitones
    hold_bars: int             # bars to hold at full shift
    fade_out_bars: int         # bars to glide back from semitones → 0


class SetTempoRamp(TypedDict):
    """Gradual tempo change for one song over a time window. Lets the
    planner avoid a hard tempo lock at the seam — e.g. ramp B from its
    original BPM up to A's BPM over the last 8 bars of B's intro, so the
    listener doesn't hear an instantaneous tempo jump when B enters.

    The mixer interpolates the time-stretch rate linearly across the ramp
    window. Outside the window the song plays at the closer endpoint BPM
    (the planner is responsible for ensuring no audible jump at the
    boundaries — typically by anchoring start_bpm to the song's natural
    BPM at the start of the ramp).

    NOT implemented in Phase 7's executor — Phase 9 territory."""

    tool: Literal["set_tempo_ramp"]
    song: SongRef
    start_time: float          # seconds in original song time
    end_time: float            # seconds in original song time
    start_bpm: float           # BPM at start_time
    end_bpm: float             # BPM at end_time


class CrossfadeStem(TypedDict):
    tool: Literal["crossfade_stem"]
    stem: StemName
    from_song: SongRef
    to_song: SongRef
    start_bar: int
    duration_bars: int
    curve: CrossfadeCurve


# Phase 9 will grow this union with apply_filter, apply_echo, swap_stem,
# loop_section, set_reasoning. Phase 7's executor only handles
# {set_transition_window, pitch_shift, crossfade_stem}; the rest are
# vocabulary the LLM can already emit but the executor will refuse with
# NotImplementedError until Phase 9 grows the dispatch table.
ToolCall = (
    SetTransitionWindow
    | PitchShift
    | TemporaryPitchShift
    | SetTempoRamp
    | CrossfadeStem
)
MixPlanJSON = list[dict]  # list[ToolCall] but JSONB persists as plain dicts


@dataclass(frozen=True)
class AnalysisBundle:
    """The slice of an Analysis row the plan generator needs.

    Kept explicit so the generator can be tested with literal fixtures
    instead of building a real Analysis ORM object.
    """

    bpm: float
    key: str            # e.g. "C", "F#m"
    camelot_key: str    # e.g. "8B", "11A"
    time_signature: int
    beat_grid: list[float]
    downbeats: list[float]
    sections: list[dict]   # each has at least {"start": float, "end": float}
    duration: float


@dataclass(frozen=True)
class SongRenderInputs:
    """Per-song inputs the executor needs to load and process audio.

    `stem_paths` map canonical stem names to StorageBackend keys. Keys
    are resolved to filesystem paths via `storage.path(...)`.
    """

    stem_paths: dict[str, str]
    analysis: AnalysisBundle


@dataclass(frozen=True)
class RenderedTransition:
    """Executor output. Caller persists via StorageBackend.write."""

    wav_bytes: bytes
    sample_rate: int
    duration_seconds: float
    pitch_shift_warning: bool  # true when |δ| > 2 was applied


class MixerPreconditionError(ValueError):
    """Raised when the inputs can't be mixed (wrong sample rate, missing
    stems on disk, etc.). The worker translates this into a `failed`
    MixPlan row with the message in `error_text`."""
