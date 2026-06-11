"""Unit tests for planner v2: candidates, decision schema, archetype
expansion, repair-not-reject validation, and orchestration. All pure —
no DB, no storage, no network."""

from __future__ import annotations

import asyncio

import pytest

from app.services.mixer.archetypes import (
    ArchetypeError,
    camelot_compatible,
    default_decision,
    expand,
)
from app.services.mixer.candidates import (
    build_in_candidates,
    build_out_candidates,
    build_pair_candidates,
    enrich_sections,
    max_seam_time,
)
from app.services.mixer.decision import (
    STYLE_DURATION_CHOICES,
    TransitionDecision,
    TransitionStyle,
)
from app.services.mixer.planner_v2 import SongMeta, build_plan_v2
from app.services.mixer.types import AnalysisBundle
from app.services.mixer.validation import (
    enforce_revert_after_crossfade,
    repair_plan,
    validate_plan,
)


def make_bundle(
    bpm: float = 120.0,
    duration: float = 240.0,
    key: str = "C",
    camelot: str = "8B",
    n_sections: int = 5,
) -> AnalysisBundle:
    spb = 60.0 / bpm
    beat_grid = [i * spb for i in range(int(duration / spb))]
    downbeats = beat_grid[::4]
    sec_len = duration / n_sections
    sections = [
        {"start": i * sec_len, "end": (i + 1) * sec_len, "label": f"section_{i}"}
        for i in range(n_sections)
    ]
    return AnalysisBundle(
        bpm=bpm, key=key, camelot_key=camelot, time_signature=4,
        beat_grid=beat_grid, downbeats=downbeats, sections=sections,
        duration=duration,
    )


def energy_curve_for(duration: float, peak_at: float = 0.5) -> list[float]:
    n = int(duration)
    return [
        1.0 - abs((i / max(1, n - 1)) - peak_at) for i in range(n)
    ]


FULL_SAFE = [{"start": 0.0, "end": 10_000.0, "safe": True}]


# ---------------------------------------------------------------- candidates

def test_out_candidates_respect_headroom_and_downbeats():
    a = make_bundle()
    cands = build_out_candidates(a, energy_curve_for(a.duration), FULL_SAFE)
    ceiling = max_seam_time(a.duration, a.bpm, a.time_signature)
    assert 1 <= len(cands) <= 5
    for c in cands:
        assert c.time <= ceiling
        assert any(abs(c.time - d) < 1e-6 for d in a.downbeats)
        assert c.id.startswith("A")
        assert c.vocal_safe is True


def test_in_candidates_stay_in_first_half():
    b = make_bundle()
    cands = build_in_candidates(b, energy_curve_for(b.duration, peak_at=0.3), [])
    assert cands
    for c in cands:
        assert c.time <= b.duration * 0.5 + 1e-6
        assert c.id.startswith("B")
        # No vocal-safety data → conservatively unsafe.
        assert c.vocal_safe is False


def test_candidates_empty_for_too_short_song():
    a = make_bundle(duration=20.0)
    assert build_out_candidates(a, [], []) == []


def test_enrich_sections_normalizes_energy_and_keeps_real_labels():
    sections = [
        {"start": 0.0, "end": 10.0, "label": "section_0"},
        {"start": 10.0, "end": 20.0, "label": "chorus"},
    ]
    curve = [0.1] * 10 + [0.4] * 10
    out = enrich_sections(sections, curve)
    assert out[1]["energy"] == 1.0
    assert out[0]["energy"] == 0.25
    assert "label" not in out[0]      # opaque cluster id dropped
    assert out[1]["label"] == "chorus"  # real label kept


# ------------------------------------------------------------------ decision

def test_decision_caps_extras_and_normalizes_duration():
    d = TransitionDecision.model_validate({
        "out": "A1", "in": "B1", "style": "drop_swap",
        "duration_bars": 9,
        "extras": ["echo_tail", "bass_kill", "reverb_tail"],
    })
    assert len(d.extras) == 2
    assert d.normalized_duration() in STYLE_DURATION_CHOICES[TransitionStyle.drop_swap]


def test_camelot_compatibility_rules():
    assert camelot_compatible("8A", "8B")    # relative major/minor
    assert camelot_compatible("8A", "9A")    # neighbour
    assert camelot_compatible("12B", "1B")   # wheel wraps
    assert not camelot_compatible("8A", "3B")
    assert camelot_compatible(None, "3B")    # unknown → assume fine


# ---------------------------------------------------------------- archetypes

@pytest.mark.parametrize("style", list(TransitionStyle))
def test_every_archetype_expands_to_a_valid_plan(style):
    a = make_bundle(bpm=126.0, key="Am", camelot="8A")
    b = make_bundle(bpm=120.0, key="Fm", camelot="4A")  # key clash
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=style)
    plan = expand(decision, a, b, cands)

    validate_plan(plan)  # must not raise
    assert plan[0]["tool"] == "set_transition_window"

    window = plan[0]
    spb_b = (60.0 / b.bpm) * b.time_signature
    stem_calls = [c for c in plan if c["tool"] == "crossfade_stem"]
    crossfade_end_b = window["to_song_time_start"] + spb_b * max(
        c["start_bar"] + c["duration_bars"] for c in stem_calls
    )
    for call in plan:
        if call["tool"] == "set_tempo_ramp":
            assert call["start_time"] >= crossfade_end_b - 1e-6
        if call["tool"] == "temporary_pitch_shift":
            assert call["start_time"] >= crossfade_end_b - 1e-6
            assert abs(call["semitones"]) <= 2


def test_drum_bridge_offsets_drums():
    a, b = make_bundle(), make_bundle(bpm=124.0)
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    decision = default_decision(cands, style=TransitionStyle.drum_bridge)
    plan = expand(decision, a, b, cands)
    drums = next(c for c in plan if c.get("stem") == "drums")
    others = [c for c in plan if c["tool"] == "crossfade_stem" and c["stem"] != "drums"]
    assert drums["start_bar"] > 0
    assert all(c["start_bar"] == 0 for c in others)


def test_stutter_skips_loop_when_not_vocal_safe():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        [], [],  # no vocal data → unsafe
    )
    decision = default_decision(cands, style=TransitionStyle.stutter_buildup)
    plan = expand(decision, a, b, cands)
    assert not any(c["tool"] == "loop_section" for c in plan)
    validate_plan(plan)


def test_expand_rejects_unknown_candidate():
    a, b = make_bundle(), make_bundle()
    cands = build_pair_candidates(
        a, b, energy_curve_for(a.duration), energy_curve_for(b.duration),
        FULL_SAFE, FULL_SAFE,
    )
    bad = TransitionDecision.model_validate({
        "out": "A99", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    with pytest.raises(ArchetypeError):
        expand(bad, a, b, cands)


# ---------------------------------------------------------------- validation

def _minimal_plan(seam_a=100.0, seam_b=20.0, bars=8):
    plan = [{
        "tool": "set_transition_window",
        "from_song_time_start": seam_a,
        "to_song_time_start": seam_b,
        "duration_bars": bars,
    }]
    for stem in ("vocals", "drums", "bass", "other"):
        plan.append({
            "tool": "crossfade_stem", "stem": stem,
            "from_song": "A", "to_song": "B",
            "start_bar": 0, "duration_bars": bars, "curve": "equal_power",
        })
    return plan


def test_repair_normalizes_song_refs():
    a, b = make_bundle(), make_bundle()
    plan = _minimal_plan()
    for c in plan[1:]:
        c["from_song"], c["to_song"] = "Song A", "song_b"
    repaired = repair_plan(plan, a, b)
    validate_plan(repaired)
    for c in repaired:
        if c["tool"] == "crossfade_stem":
            assert (c["from_song"], c["to_song"]) == ("A", "B")


def test_repair_converts_permanent_pitch_shift():
    a, b = make_bundle(), make_bundle()
    plan = _minimal_plan() + [
        {"tool": "pitch_shift", "song": "B", "semitones": -5}
    ]
    repaired = repair_plan(plan, a, b)
    validate_plan(repaired)
    tps = [c for c in repaired if c["tool"] == "temporary_pitch_shift"]
    assert len(tps) == 1 and tps[0]["semitones"] == -2  # capped
    assert not any(c["tool"] == "pitch_shift" for c in repaired)


def test_repair_clamps_late_seam_and_fills_missing_stems():
    a, b = make_bundle(duration=200.0), make_bundle()
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 199.0,  # way past headroom
         "to_song_time_start": 20.0, "duration_bars": 8},
        {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A",
         "to_song": "B", "start_bar": 0, "duration_bars": 8,
         "curve": "equal_power"},
    ]
    repaired = repair_plan(plan, a, b)
    validate_plan(repaired)
    ceiling = max_seam_time(a.duration, a.bpm, a.time_signature)
    assert repaired[0]["from_song_time_start"] <= ceiling
    stems = {c["stem"] for c in repaired if c["tool"] == "crossfade_stem"}
    assert stems == {"vocals", "drums", "bass", "other"}


def test_repair_drops_unknown_tools():
    a, b = make_bundle(), make_bundle()
    plan = _minimal_plan() + [{"tool": "explode_speakers", "song": "A"}]
    repaired = repair_plan(plan, a, b)
    validate_plan(repaired)


def test_validate_rejects_planless_garbage():
    with pytest.raises(ValueError):
        validate_plan([{"tool": "crossfade_stem", "stem": "vocals",
                        "from_song": "A", "to_song": "B",
                        "start_bar": 0, "duration_bars": 8,
                        "curve": "equal_power"}])


def test_enforce_revert_defers_early_ramp():
    b = make_bundle(bpm=120.0)
    plan = _minimal_plan(seam_b=20.0, bars=8) + [{
        "tool": "set_tempo_ramp", "song": "B",
        "start_time": 21.0, "end_time": 40.0,
        "start_bpm": 126.0, "end_bpm": 120.0,
    }]
    out = enforce_revert_after_crossfade(plan, b)
    ramp = next(c for c in out if c["tool"] == "set_tempo_ramp")
    spb_b = (60.0 / b.bpm) * b.time_signature
    assert ramp["start_time"] >= 20.0 + 8 * spb_b - 1e-6


# ---------------------------------------------------------------- planner v2

class StubProvider:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    async def complete_json(self, *, system, user, nonce=0, cache_namespace="x"):
        self.calls.append({"system": system, "user": user, "nonce": nonce})
        if self.exc:
            raise self.exc
        return self.response


def _metas():
    a = make_bundle(bpm=126.0)
    b = make_bundle(bpm=120.0)
    return (
        SongMeta("Levels", "Avicii", a, energy_curve_for(a.duration), FULL_SAFE),
        SongMeta("One More Time", "Daft Punk", b,
                 energy_curve_for(b.duration), FULL_SAFE),
    )


def test_planner_v2_happy_path_uses_llm_decision():
    a, b = _metas()
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend",
        "duration_bars": 16, "a_fade_out_bars": 8,
        "rationale": "both four-on-the-floor at similar energy",
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.source == "llm_v2"
    assert outcome.style == "smooth_blend"
    validate_plan(outcome.plan)
    # Identity made it into the prompt.
    assert "Avicii" in provider.calls[0]["user"]
    assert "Daft Punk" in provider.calls[0]["user"]


def test_planner_v2_repairs_near_miss_decision():
    a, b = _metas()
    provider = StubProvider(response={
        "style": "drum_bridge", "out": "A1",   # missing "in", bad shape
        "duration_bars": "lots",
    })
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.source == "llm_v2_repaired"
    assert outcome.style == "drum_bridge"
    validate_plan(outcome.plan)


def test_planner_v2_falls_back_when_llm_dies():
    a, b = _metas()
    provider = StubProvider(exc=RuntimeError("api down"))
    outcome = asyncio.run(build_plan_v2(provider, a, b))
    assert outcome.source == "style_default"
    validate_plan(outcome.plan)


def test_planner_v2_pinned_style_wins():
    a, b = _metas()
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    outcome = asyncio.run(
        build_plan_v2(provider, a, b, style_override="vinyl_stop")
    )
    assert outcome.style == "vinyl_stop"
    assert any(c["tool"] == "turntable_stop" for c in outcome.plan)
    validate_plan(outcome.plan)


def test_planner_v2_passes_nonce_and_context():
    a, b = _metas()
    provider = StubProvider(response={
        "out": "A1", "in": "B1", "style": "smooth_blend", "duration_bars": 16,
    })
    asyncio.run(build_plan_v2(
        provider, a, b, style_hint="wash_out",
        previous_styles=["drop_swap"], nonce=3,
    ))
    call = provider.calls[0]
    assert call["nonce"] == 3
    assert "wash_out" in call["user"]
    assert "drop_swap" in call["user"]
