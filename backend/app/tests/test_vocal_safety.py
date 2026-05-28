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
