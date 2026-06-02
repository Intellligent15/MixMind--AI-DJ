"""render_transition Celery task tests.

Same pattern as test_separate_task / test_transcribe_task: real DB, set
up two ready songs + a MixPlan row, mock the executor + storage so we
don't run real rubberband or load real WAVs.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.db import SessionLocal
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    Song,
    SongStatus,
    Stems,
    StemsStatus,
)
from app.services.mixer.types import RenderedTransition


def _make_analysis(song_id, bpm: float = 120.0, key: str = "C") -> Analysis:
    return Analysis(
        song_id=song_id,
        bpm=bpm,
        key=key,
        camelot_key="8B",
        time_signature=4,
        beat_grid=[i * 0.5 for i in range(360)],
        downbeats=[i * 2.0 for i in range(90)],
        sections=[
            {"start": 0.0, "end": 30.0, "label": "intro"},
            {"start": 30.0, "end": 150.0, "label": "body"},
            {"start": 150.0, "end": 180.0, "label": "outro"},
        ],
        energy_curve=[0.5] * 180,
        vocal_segments=[],
    )


def _make_stems(song_id, video_id: str) -> Stems:
    return Stems(
        song_id=song_id,
        model_name="htdemucs",
        status=StemsStatus.separated,
        vocals_path=f"stems/{video_id}/vocals.wav",
        drums_path=f"stems/{video_id}/drums.wav",
        bass_path=f"stems/{video_id}/bass.wav",
        other_path=f"stems/{video_id}/other.wav",
        vocal_rms=0.15,
    )


@pytest.fixture
def pair_with_plan():
    """Two ready songs in a locked queue + a pending MixPlan row."""
    payload = {}
    with SessionLocal() as db:
        a = Song(
            youtube_video_id=f"rta-{id(object())}",
            title="A", duration_seconds=180.0, audio_path="audio/a.wav",
            status=SongStatus.ready,
        )
        b = Song(
            youtube_video_id=f"rtb-{id(object())}",
            title="B", duration_seconds=180.0, audio_path="audio/b.wav",
            status=SongStatus.ready,
        )
        q = Queue(locked=True)
        db.add_all([a, b, q])
        db.flush()
        db.add_all([
            _make_analysis(a.id), _make_analysis(b.id, bpm=128.0),
            _make_stems(a.id, a.youtube_video_id),
            _make_stems(b.id, b.youtube_video_id),
            QueueItem(queue_id=q.id, song_id=a.id, position=0),
            QueueItem(queue_id=q.id, song_id=b.id, position=1),
        ])
        plan = MixPlan(
            queue_id=q.id, from_song_id=a.id, to_song_id=b.id,
            plan_json=None, status=MixPlanStatus.pending,
        )
        db.add(plan)
        db.commit()
        payload = {
            "queue_id": str(q.id),
            "a_id": str(a.id), "b_id": str(b.id),
            "plan_id": str(plan.id),
        }
    yield payload
    with SessionLocal() as db:
        # Cascade from queue + songs cleans MixPlan, Analysis, Stems.
        for sid_key in ("a_id", "b_id"):
            song = db.get(Song, uuid.UUID(payload[sid_key]))
            if song is not None:
                db.delete(song)
        q = db.get(Queue, uuid.UUID(payload["queue_id"]))
        if q is not None:
            db.delete(q)
        db.commit()


def _patched_render():
    return RenderedTransition(
        wav_bytes=b"RIFF....fakewavbytes",
        sample_rate=44100,
        duration_seconds=42.0,
        pitch_shift_warning=False,
    )


def test_render_transition_happy_path(pair_with_plan):
    storage = AsyncMock()
    storage.write = MagicMock()
    # storage.write is async in the protocol; the worker calls it sync.
    # We use a sync MagicMock — the worker code uses asyncio.run for it.
    # See implementation. To keep this simple here, patch the awaitable.
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write

    with (
        patch(
            "app.workers.render_transition.render",
            return_value=_patched_render(),
        ),
        patch(
            "app.workers.render_transition.get_storage", return_value=storage
        ),
    ):
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result == pair_with_plan["plan_id"]
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        assert row is not None
        assert row.status == MixPlanStatus.ready
        assert row.rendered_audio_path == f"mixes/{pair_with_plan['plan_id']}.wav"
        # plan_json was lazily generated on first render.
        assert row.plan_json is not None
        assert row.plan_json[0]["tool"] == "set_transition_window"


def test_render_transition_atomic_claim_loser(pair_with_plan):
    """A second dispatch while the row is already `rendering` should
    no-op and return None."""
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        row.status = MixPlanStatus.rendering
        db.commit()

    with patch(
        "app.workers.render_transition.render", return_value=_patched_render()
    ) as render_mock:
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result is None
    render_mock.assert_not_called()


def test_render_transition_marks_failed_on_executor_error(pair_with_plan):
    storage = AsyncMock()
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write

    with (
        patch(
            "app.workers.render_transition.render",
            side_effect=RuntimeError("rubberband not on PATH"),
        ),
        patch(
            "app.workers.render_transition.get_storage", return_value=storage
        ),
    ):
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])
        assert result is None

    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        assert row.status == MixPlanStatus.failed
        assert "rubberband not on PATH" in (row.error_text or "")


def test_render_transition_refuses_when_song_not_ready(pair_with_plan):
    with SessionLocal() as db:
        a = db.get(Song, uuid.UUID(pair_with_plan["a_id"]))
        a.status = SongStatus.analyzed
        db.commit()

    with patch(
        "app.workers.render_transition.render", return_value=_patched_render()
    ) as render_mock:
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result is None
    render_mock.assert_not_called()
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        # Still pending — the worker bailed before claiming.
        assert row.status == MixPlanStatus.pending


def test_render_transition_missing_row_returns_none():
    from app.workers.render_transition import render_transition
    result = render_transition(str(uuid.uuid4()))
    assert result is None


def test_enrich_sections_helper():
    """Each section gets mean energy normalized to the hottest section."""
    from app.workers.render_transition import _enrich_sections
    # 1 Hz energy curve: 0-10s quiet, 10-20s loud.
    curve = [0.1] * 10 + [0.8] * 10
    sections = [
        {"start": 0.0, "end": 10.0, "label": "intro"},
        {"start": 10.0, "end": 20.0, "label": "drop"},
    ]
    out = _enrich_sections(sections, curve)
    assert out[0]["energy"] == round(0.1 / 0.8, 2)  # quiet section, relative
    assert out[1]["energy"] == 1.0  # hottest section normalizes to 1.0
    assert "label" not in out[0]  # opaque cluster id dropped
    assert _enrich_sections([], curve) == []


def _valid_plan(extra: list[dict] | None = None) -> list[dict]:
    """A minimally-valid LLM plan: one window + 4 stems (+ optional extras)."""
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0, "duration_bars": 4},
    ]
    if extra:
        plan.extend(extra)
    for stem in ("vocals", "drums", "bass", "other"):
        plan.append({
            "tool": "crossfade_stem", "stem": stem,
            "from_song": "A", "to_song": "B",
            "start_bar": 0, "duration_bars": 4, "curve": "equal_power",
        })
    return plan


def test_render_transition_llm_planner(pair_with_plan):
    storage = AsyncMock()
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write
    storage.read.return_value = b'{"rms": [0.1], "peak": [0.2], "frame_hz": 10}'

    mock_llm_provider = AsyncMock()
    mock_plan = _valid_plan()
    mock_llm_provider.plan_transition.return_value = mock_plan

    with (
        patch("app.workers.render_transition.render", return_value=_patched_render()),
        patch("app.workers.render_transition.get_storage", return_value=storage),
        patch("app.workers.render_transition.get_llm_provider", return_value=mock_llm_provider),
        patch("app.workers.render_transition.settings.use_llm_planner", True),
    ):
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result == pair_with_plan["plan_id"]
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        assert row.plan_json == mock_plan
    mock_llm_provider.plan_transition.assert_called_once()

    # Verify the LLM saw the signals it actually uses for planning, and
    # NOT the redundant ones that dominated the prompt cost.
    call_args = mock_llm_provider.plan_transition.call_args.args
    from_song_input, to_song_input = call_args[0], call_args[1]
    assert "sections" in from_song_input["analysis"]
    assert "seconds_per_bar" in from_song_input["analysis"]
    assert "max_seam_time" in from_song_input["analysis"]
    assert "max_seam_time" in to_song_input["analysis"]
    assert "vocal_safe_regions" in to_song_input
    # Sections carry normalized energy; the standalone energy_curve and
    # the raw downbeats array are intentionally dropped — energy is
    # folded into sections, and bar length comes from seconds_per_bar.
    assert "energy_curve" not in from_song_input["analysis"]
    assert "downbeats" not in from_song_input["analysis"]
    assert all("energy" in s for s in from_song_input["analysis"]["sections"])
    # Lyrics, raw transcription, and vocal_segments are intentionally
    # absent — vocal_safe_regions already distills what the LLM needs,
    # and including the raw signals blew us past Groq's TPM ceiling.
    assert "aligned_lyrics" not in from_song_input
    assert "raw_transcription" not in from_song_input
    assert "vocal_segments" not in to_song_input["analysis"]


def test_render_transition_llm_planner_fallback(pair_with_plan):
    storage = AsyncMock()
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write
    storage.read.return_value = b'{"rms": [0.1], "peak": [0.2], "frame_hz": 10}'

    mock_llm_provider = AsyncMock()
    mock_llm_provider.plan_transition.side_effect = Exception("LLM timed out")

    with (
        patch("app.workers.render_transition.render", return_value=_patched_render()),
        patch("app.workers.render_transition.get_storage", return_value=storage),
        patch("app.workers.render_transition.get_llm_provider", return_value=mock_llm_provider),
        patch("app.workers.render_transition.settings.use_llm_planner", True),
        patch("app.workers.render_transition.build_pair_plan") as mock_fallback,
    ):
        fallback_plan = _valid_plan()
        mock_fallback.return_value = fallback_plan
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result == pair_with_plan["plan_id"]
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        assert row.plan_json == fallback_plan
    mock_llm_provider.plan_transition.assert_called_once()
    mock_fallback.assert_called_once()


def test_render_transition_llm_invalid_plan_falls_back(pair_with_plan):
    """LLM returns a plan that won't pass shape validation (missing stem
    calls). The worker must fall back to the deterministic planner
    instead of letting the executor blow up mid-render."""
    storage = AsyncMock()
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write
    storage.read.return_value = b'{"rms": [0.1], "peak": [0.2], "frame_hz": 10}'

    mock_llm_provider = AsyncMock()
    mock_llm_provider.plan_transition.return_value = [
        {"tool": "set_transition_window", "from_song_time_start": 0.0,
         "to_song_time_start": 0.0, "duration_bars": 4},
        # only 2 of 4 required stems
        {"tool": "crossfade_stem", "stem": "vocals",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 4, "curve": "equal_power"},
        {"tool": "crossfade_stem", "stem": "drums",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 4, "curve": "equal_power"},
    ]

    with (
        patch("app.workers.render_transition.render", return_value=_patched_render()),
        patch("app.workers.render_transition.get_storage", return_value=storage),
        patch("app.workers.render_transition.get_llm_provider", return_value=mock_llm_provider),
        patch("app.workers.render_transition.settings.use_llm_planner", True),
        patch("app.workers.render_transition.build_pair_plan") as mock_fallback,
    ):
        fallback_plan = _valid_plan()
        mock_fallback.return_value = fallback_plan
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result == pair_with_plan["plan_id"]
    mock_fallback.assert_called_once()
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        assert row.plan_json == fallback_plan


def test_max_seam_time_helper():
    """Latest seam = duration - 16 bars - 5s safety, clamped to 0."""
    from app.workers.render_transition import _max_seam_time
    # 240s @ 120 BPM, 4/4: sec_per_bar = 2.0, 16 bars = 32s. Latest = 240 - 32 - 5 = 203.
    assert _max_seam_time(240.0, 120.0, 4) == 203.0
    # Short song: should clamp to 0, not go negative.
    assert _max_seam_time(10.0, 60.0, 4) == 0.0
    # No BPM / no duration → 0 (don't crash).
    assert _max_seam_time(0.0, 120.0, 4) == 0.0
    assert _max_seam_time(240.0, 0.0, 4) == 0.0


def test_render_transition_rejects_seam_without_headroom(pair_with_plan):
    """LLM picks a seam too close to A's end so the executor would clamp
    the crossfade to ~nothing. Validator rejects → deterministic
    fallback gets persisted instead."""
    storage = AsyncMock()
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write
    storage.read.return_value = b'{"rms": [0.1], "peak": [0.2], "frame_hz": 10}'

    # _make_analysis sets bpm=120, time_signature=4; pair_with_plan
    # sets duration_seconds=180. So max_seam_time = 180 - 16*2 - 5 = 143s.
    # Picking from_song_time_start=170s + 16-bar crossfade (32s) puts
    # the seam end at 202s, well past the 180s song.
    bad_plan = _valid_plan()
    bad_plan[0]["from_song_time_start"] = 170.0
    bad_plan[0]["duration_bars"] = 16
    for c in bad_plan:
        if c["tool"] == "crossfade_stem":
            c["duration_bars"] = 16

    mock_llm_provider = AsyncMock()
    mock_llm_provider.plan_transition.return_value = bad_plan

    with (
        patch("app.workers.render_transition.render", return_value=_patched_render()),
        patch("app.workers.render_transition.get_storage", return_value=storage),
        patch("app.workers.render_transition.get_llm_provider", return_value=mock_llm_provider),
        patch("app.workers.render_transition.settings.use_llm_planner", True),
        patch("app.workers.render_transition.build_pair_plan") as mock_fallback,
    ):
        fallback_plan = _valid_plan()
        mock_fallback.return_value = fallback_plan
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result == pair_with_plan["plan_id"]
    mock_fallback.assert_called_once()
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        assert row.plan_json == fallback_plan


def test_render_transition_rejects_wrong_song_refs(pair_with_plan):
    """LLM uses 'Song A'/'Song B' instead of 'A'/'B' — validator rejects
    and falls back to the deterministic planner."""
    storage = AsyncMock()
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write
    storage.read.return_value = b'{"rms": [0.1], "peak": [0.2], "frame_hz": 10}'

    mock_llm_provider = AsyncMock()
    bad_plan = [
        {"tool": "set_transition_window", "from_song_time_start": 0.0,
         "to_song_time_start": 0.0, "duration_bars": 4},
    ]
    for stem in ("vocals", "drums", "bass", "other"):
        bad_plan.append({
            "tool": "crossfade_stem", "stem": stem,
            "from_song": "Song A", "to_song": "Song B",  # wrong shape
            "start_bar": 0, "duration_bars": 4, "curve": "equal_power",
        })
    mock_llm_provider.plan_transition.return_value = bad_plan

    with (
        patch("app.workers.render_transition.render", return_value=_patched_render()),
        patch("app.workers.render_transition.get_storage", return_value=storage),
        patch("app.workers.render_transition.get_llm_provider", return_value=mock_llm_provider),
        patch("app.workers.render_transition.settings.use_llm_planner", True),
        patch("app.workers.render_transition.build_pair_plan") as mock_fallback,
    ):
        mock_fallback.return_value = _valid_plan()
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result == pair_with_plan["plan_id"]
    mock_fallback.assert_called_once()
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        # Persisted plan is the deterministic fallback, not the bad LLM one.
        assert all(
            c.get("from_song", "A") == "A" and c.get("to_song", "B") == "B"
            for c in row.plan_json
            if c.get("tool") == "crossfade_stem"
        )


def test_validate_llm_plan_accepts_per_stem_envelopes():
    """Validator must accept 4 stem calls with DIFFERENT start_bar/
    duration_bars/curve — that's the whole point of per-stem envelopes."""
    from app.workers.render_transition import _validate_llm_plan
    plan = [
        {"tool": "set_transition_window", "from_song_time_start": 0.0,
         "to_song_time_start": 0.0, "duration_bars": 16},
        {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 8, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "drums", "from_song": "A", "to_song": "B",
         "start_bar": 4, "duration_bars": 12, "curve": "equal_power"},
        {"tool": "crossfade_stem", "stem": "bass", "from_song": "A", "to_song": "B",
         "start_bar": 2, "duration_bars": 6, "curve": "equal_power"},
        {"tool": "crossfade_stem", "stem": "other", "from_song": "A", "to_song": "B",
         "start_bar": 8, "duration_bars": 4, "curve": "linear"},
    ]
    # No exception → accepted.
    _validate_llm_plan(plan)


def test_render_transition_accepts_per_stem_envelope_plan(pair_with_plan):
    """End-to-end: LLM returns a per-stem envelope plan; worker persists it
    unchanged (no fallback) and downstream render is called with it."""
    storage = AsyncMock()
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write
    storage.read.return_value = b'{"rms": [0.1], "peak": [0.2], "frame_hz": 10}'

    # Plan with per-stem envelopes: vocals fade out early, drums hold.
    per_stem_plan = [
        {"tool": "set_transition_window", "from_song_time_start": 0.0,
         "to_song_time_start": 0.0, "duration_bars": 16},
        {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 8, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "drums", "from_song": "A", "to_song": "B",
         "start_bar": 8, "duration_bars": 8, "curve": "equal_power"},
        {"tool": "crossfade_stem", "stem": "bass", "from_song": "A", "to_song": "B",
         "start_bar": 8, "duration_bars": 8, "curve": "equal_power"},
        {"tool": "crossfade_stem", "stem": "other", "from_song": "A", "to_song": "B",
         "start_bar": 8, "duration_bars": 8, "curve": "equal_power"},
    ]
    mock_llm_provider = AsyncMock()
    mock_llm_provider.plan_transition.return_value = per_stem_plan

    with (
        patch("app.workers.render_transition.render", return_value=_patched_render()),
        patch("app.workers.render_transition.get_storage", return_value=storage),
        patch("app.workers.render_transition.get_llm_provider", return_value=mock_llm_provider),
        patch("app.workers.render_transition.settings.use_llm_planner", True),
        patch("app.workers.render_transition.build_pair_plan") as mock_fallback,
    ):
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result == pair_with_plan["plan_id"]
    mock_fallback.assert_not_called()  # per-stem envelopes are valid; no fallback
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        assert row.plan_json == per_stem_plan


def test_render_transition_clamps_llm_permanent_pitch_shift(pair_with_plan):
    """LLM emits pitch_shift=+5; worker must clamp to +2 before persist."""
    storage = AsyncMock()
    async def _write(key, data):
        return f"/abs/{key}"
    storage.write = _write
    storage.read.return_value = b'{"rms": [0.1], "peak": [0.2], "frame_hz": 10}'

    mock_llm_provider = AsyncMock()
    mock_llm_provider.plan_transition.return_value = _valid_plan(extra=[
        {"tool": "pitch_shift", "song": "B", "semitones": 5},
    ])

    with (
        patch("app.workers.render_transition.render", return_value=_patched_render()),
        patch("app.workers.render_transition.get_storage", return_value=storage),
        patch("app.workers.render_transition.get_llm_provider", return_value=mock_llm_provider),
        patch("app.workers.render_transition.settings.use_llm_planner", True),
    ):
        from app.workers.render_transition import render_transition
        result = render_transition(pair_with_plan["plan_id"])

    assert result == pair_with_plan["plan_id"]
    with SessionLocal() as db:
        row = db.get(MixPlan, uuid.UUID(pair_with_plan["plan_id"]))
        pitch_calls = [c for c in row.plan_json if c["tool"] == "pitch_shift"]
        assert len(pitch_calls) == 1
        assert pitch_calls[0]["semitones"] == 2  # clamped from +5
