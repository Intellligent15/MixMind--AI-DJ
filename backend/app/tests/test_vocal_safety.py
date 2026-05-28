"""Tests for vocal_safe_regions. Pure function, no DB, no audio."""

from __future__ import annotations

from app.services.vocal_safety.safety import (
    _hop_seconds,
    vocal_safe_regions,
)


def _envelope(frame_hz: int, rms: list[float], peak: list[float] | None = None) -> dict:
    return {
        "frame_hz": frame_hz,
        "rms": rms,
        "peak": peak if peak is not None else [r * 1.5 for r in rms],
    }


def _whisper_seg(words: list[tuple[str, float, float, float]], avg_logprob: float = -0.2) -> dict:
    return {
        "start": words[0][1] if words else 0.0,
        "end": words[-1][2] if words else 0.0,
        "text": " ".join(w[0] for w in words),
        "avg_logprob": avg_logprob,
        "no_speech_prob": 0.01,
        "compression_ratio": 1.5,
        "temperature": 0.0,
        "words": [
            {"word": w, "start": s, "end": e, "probability": p}
            for (w, s, e, p) in words
        ],
    }


def test_hop_seconds_from_frame_hz():
    assert _hop_seconds({"frame_hz": 10}) == 0.1
    assert _hop_seconds({"frame_hz": 20}) == 0.05


def test_hop_seconds_legacy_fallback():
    assert _hop_seconds({"hop_seconds": 0.025}) == 0.025
    assert _hop_seconds({}) == 0.1


def test_peak_signal_blocks_quiet_check():
    # RMS is below quiet threshold, but peak is hot enough to mark
    # 30% of frames noisy → fails the 15% tolerance.
    env_rms = [0.001] * 50
    env_peak = [0.005] * 50
    for i in range(15):
        env_peak[i * 3] = 0.2
    env = _envelope(10, env_rms, env_peak)
    out = vocal_safe_regions(
        transcription_segments=[],
        envelope=env,
        duration_seconds=5.0,
    )
    assert out == []


def test_pure_instrumental_returns_full_duration_safe():
    env = _envelope(10, rms=[0.001] * 100, peak=[0.005] * 100)
    out = vocal_safe_regions(
        transcription_segments=[],
        envelope=env,
        duration_seconds=10.0,
    )
    assert len(out) == 1
    assert out[0]["start"] == 0.0
    assert abs(out[0]["end"] - 10.0) < 0.2


def test_aligned_match_word_blocks_region():
    env_rms = [0.001] * 50 + [0.05] * 5 + [0.001] * 45  # hot at 5.0-5.5
    env_peak = [0.005] * 50 + [0.15] * 5 + [0.005] * 45
    env = _envelope(10, env_rms, env_peak)
    aligned = [{
        "word": "hello", "start": 5.0, "end": 5.5,
        "confidence": 0.9, "source": "whisper_match",
    }]
    out = vocal_safe_regions(
        transcription_segments=[], envelope=env, aligned_words=aligned,
        duration_seconds=10.0,
    )
    assert len(out) == 2
    assert out[0]["end"] <= 5.0
    assert out[1]["start"] >= 5.5


def test_aligned_interpolated_filtered_when_silent():
    # Interpolated word at 5s but envelope is silent → filter out.
    env = _envelope(10, rms=[0.001] * 100, peak=[0.005] * 100)
    aligned = [{
        "word": "phantom", "start": 5.0, "end": 5.5,
        "confidence": 0.2, "source": "interpolated",
    }]
    out = vocal_safe_regions(
        transcription_segments=[], envelope=env, aligned_words=aligned,
        duration_seconds=10.0,
    )
    assert len(out) == 1
    assert out[0]["reason"] == "quiet_gap"


def test_raw_whisper_hallucination_filtered_by_envelope():
    env = _envelope(10, rms=[0.001] * 100, peak=[0.005] * 100)
    segs = [_whisper_seg([("Thank", 4.0, 4.5, 0.95), ("you", 4.5, 5.0, 0.95)])]
    out = vocal_safe_regions(
        transcription_segments=segs, envelope=env, duration_seconds=10.0,
    )
    assert len(out) == 1


def test_raw_whisper_low_prob_word_does_not_block():
    env = _envelope(10, rms=[0.001] * 100, peak=[0.005] * 100)
    segs = [_whisper_seg([("phantom", 5.0, 5.5, 0.2)])]
    out = vocal_safe_regions(
        transcription_segments=segs, envelope=env, duration_seconds=10.0,
    )
    assert len(out) == 1


def test_min_safe_region_filters_short_gaps():
    env_rms = (
        [0.001] * 20    # 0-2s safe
        + [0.05] * 5    # 2.0-2.5s hot
        + [0.001] * 5   # 2.5-3.0s quiet < 1.5s
        + [0.05] * 5    # 3.0-3.5s hot
        + [0.001] * 65  # 3.5-10.0s safe
    )
    env_peak = [r * 5 for r in env_rms]
    env = _envelope(10, env_rms, env_peak)
    aligned = [
        {"word": "first", "start": 2.0, "end": 2.5, "confidence": 0.9, "source": "whisper_match"},
        {"word": "second", "start": 3.0, "end": 3.5, "confidence": 0.9, "source": "whisper_match"},
    ]
    out = vocal_safe_regions(
        transcription_segments=[], envelope=env, aligned_words=aligned,
        duration_seconds=10.0,
    )
    assert all((r["end"] - r["start"]) >= 1.5 for r in out)
    # The short gap 2.5-3.0 is filtered out.
    assert not any(2.0 <= r["start"] < 3.0 for r in out)


def test_empty_envelope_returns_no_envelope_marker():
    out = vocal_safe_regions(
        transcription_segments=[],
        envelope={"frame_hz": 10, "rms": [], "peak": []},
        duration_seconds=180.0,
    )
    assert out == [{"start": 0.0, "end": 180.0, "safe": True, "reason": "no_envelope"}]
