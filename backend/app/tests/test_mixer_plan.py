"""Tests for `build_pair_plan` — the hand-built stand-in for Phase 9's
LLM call. Pure function over two AnalysisBundles → list of tool-call
dicts. No DB, no audio."""

from __future__ import annotations

from app.services.mixer.plan import build_pair_plan
from app.services.mixer.types import AnalysisBundle


def _bundle(
    bpm: float = 120.0,
    key: str = "C",
    camelot: str = "8B",
    duration: float = 182.0,
    sections: list[dict] | None = None,
    beat_grid: list[float] | None = None,
    downbeats: list[float] | None = None,
    time_signature: int = 4,
) -> AnalysisBundle:
    # Default fixture: 120 BPM, ~3-minute song with exactly 32s of outro
    # (so a 16-bar crossfade starting at the last section fits to the
    # sample). Downbeats every 2s, beats every 0.5s, 3 sections
    # (intro 0-30, body 30-150, outro 150-end).
    sec_per_beat = 60.0 / bpm
    sec_per_bar = sec_per_beat * time_signature
    if beat_grid is None:
        n_beats = int(duration / sec_per_beat)
        beat_grid = [i * sec_per_beat for i in range(n_beats)]
    if downbeats is None:
        n_bars = int(duration / sec_per_bar)
        downbeats = [i * sec_per_bar for i in range(n_bars)]
    if sections is None:
        sections = [
            {"start": 0.0, "end": 30.0, "label": "intro"},
            {"start": 30.0, "end": 150.0, "label": "body"},
            {"start": 150.0, "end": duration, "label": "outro"},
        ]
    return AnalysisBundle(
        bpm=bpm,
        key=key,
        camelot_key=camelot,
        time_signature=time_signature,
        beat_grid=beat_grid,
        downbeats=downbeats,
        sections=sections,
        duration=duration,
    )


def test_build_pair_plan_identical_songs():
    a = _bundle()
    b = _bundle()
    plan = build_pair_plan(a, b)

    # set_transition_window first, then 4 crossfade_stem calls. No
    # pitch_shift (identical keys).
    assert plan[0]["tool"] == "set_transition_window"
    stem_calls = [c for c in plan if c["tool"] == "crossfade_stem"]
    assert len(stem_calls) == 4
    assert {c["stem"] for c in stem_calls} == {"vocals", "drums", "bass", "other"}
    assert not any(c["tool"] == "pitch_shift" for c in plan)

    window = plan[0]
    # A's outro starts at 150.0, snapped to nearest downbeat ≥ 150.0.
    # Bars are 2s at 120/4, so first downbeat ≥ 150.0 is 150.0 itself.
    assert window["from_song_time_start"] == 150.0
    # B's intro ends at 30.0, first downbeat ≥ 30.0 is 30.0.
    assert window["to_song_time_start"] == 30.0
    # 16 bars by default; both songs have plenty of room.
    assert window["duration_bars"] == 16

    # All 4 crossfade_stem calls share the same envelope params (Phase 7).
    for call in stem_calls:
        assert call["from_song"] == "A"
        assert call["to_song"] == "B"
        assert call["start_bar"] == 0
        assert call["duration_bars"] == 16
        assert call["curve"] == "linear"


def test_build_pair_plan_tempo_difference():
    a = _bundle(bpm=120.0)
    b = _bundle(bpm=128.0)
    plan = build_pair_plan(a, b)
    # Window present, no pitch shift (same key).
    window = next(c for c in plan if c["tool"] == "set_transition_window")
    assert window["duration_bars"] == 16  # both songs have plenty


def test_build_pair_plan_key_difference_with_minor_shift():
    a = _bundle(key="C")    # C major, root=0
    b = _bundle(key="D")    # D major, root=2 → δ = -2 (B shifts down 2)
    plan = build_pair_plan(a, b)
    shift = next(c for c in plan if c["tool"] == "pitch_shift")
    assert shift["semitones"] == -2
    assert shift["song"] == "B"


def test_build_pair_plan_large_shift_applied_with_warning(caplog):
    a = _bundle(key="C")    # root=0
    b = _bundle(key="F#")   # root=6 → δ = -6 or +6; ((0-6+6)%12)-6 = -6
    with caplog.at_level("WARNING"):
        plan = build_pair_plan(a, b)
    shift = next(c for c in plan if c["tool"] == "pitch_shift")
    assert abs(shift["semitones"]) == 6  # full shift applied
    assert any("large pitch shift" in rec.message for rec in caplog.records)


def test_build_pair_plan_relative_major_minor_no_shift():
    # C major's relative minor is A minor (same Camelot position 8B/8A).
    a = _bundle(key="C")
    b = _bundle(key="Am")
    plan = build_pair_plan(a, b)
    assert not any(c["tool"] == "pitch_shift" for c in plan)


def test_build_pair_plan_short_a_outro_clamps_to_end_window():
    # A is 60s long, last section starts at 55s (only 5s of section-outro
    # but we still anchor seam to "no more than 16 bars before end"). At
    # 120 BPM/4, 16 bars = 32s, so seam should be at max(55, 60-32) = 55.
    a = _bundle(duration=60.0, sections=[
        {"start": 0.0, "end": 55.0, "label": "body"},
        {"start": 55.0, "end": 60.0, "label": "outro"},
    ])
    b = _bundle()
    plan = build_pair_plan(a, b)
    window = next(c for c in plan if c["tool"] == "set_transition_window")
    # seam_a = nearest downbeat ≥ 55.0. Downbeats are 0,2,4,...,58 → snap to 56.
    assert window["from_song_time_start"] == 56.0
    # duration_bars clamped: available_a = 60-56 = 4s = 2 bars. Below floor → 4.
    assert window["duration_bars"] == 4


def test_build_pair_plan_b_intro_skipped():
    # B's first section is 0-10s of intro; seam_b should land at first
    # downbeat ≥ 10. With 128 BPM/4, sec_per_bar = 60/128*4 = 1.875s.
    # Downbeats at 0, 1.875, 3.75, ..., first ≥ 10 is 9*1.875=16.875.
    # Actually 5*1.875=9.375 (no), 6*1.875=11.25 (yes).
    b = _bundle(bpm=128.0, sections=[
        {"start": 0.0, "end": 10.0, "label": "intro"},
        {"start": 10.0, "end": 180.0, "label": "body"},
    ])
    a = _bundle(bpm=128.0)  # same tempo so we can ignore stretch math
    plan = build_pair_plan(a, b)
    window = next(c for c in plan if c["tool"] == "set_transition_window")
    sec_per_bar = (60.0 / 128.0) * 4
    expected = 6 * sec_per_bar  # 11.25
    assert abs(window["to_song_time_start"] - expected) < 1e-9
