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
    tool: Literal["pitch_shift"]
    song: SongRef
    semitones: int  # may exceed ±2; mixer logs a WARN at WARN level


class CrossfadeStem(TypedDict):
    tool: Literal["crossfade_stem"]
    stem: StemName
    from_song: SongRef
    to_song: SongRef
    start_bar: int
    duration_bars: int
    curve: CrossfadeCurve


# Phase 9 will grow this union with apply_filter, apply_echo, swap_stem,
# loop_section, set_reasoning. Phase 7's executor refuses any tool not
# in {set_transition_window, pitch_shift, crossfade_stem}.
ToolCall = SetTransitionWindow | PitchShift | CrossfadeStem
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
