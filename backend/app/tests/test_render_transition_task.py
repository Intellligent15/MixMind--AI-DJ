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

    # Verify the LLM saw the richer signals (energy_curve + sections + downbeats).
    call_args = mock_llm_provider.plan_transition.call_args.args
    from_song_input, to_song_input = call_args[0], call_args[1]
    assert "energy_curve" in from_song_input["analysis"]
    assert "downbeats" in from_song_input["analysis"]
    assert "sections" in from_song_input["analysis"]
    assert "vocal_segments" in to_song_input["analysis"]


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
