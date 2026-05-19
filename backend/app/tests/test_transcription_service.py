"""Unit tests for TranscriptionService.

mlx-whisper.transcribe is patched at module scope so the tests never
download the ~3 GB MLX weights or run actual inference. The thing the
service layer owns is the raw -> wire-format normalization (segment shape,
word entries, language, duration heuristic), which is verifiable against
a stubbed return value.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.services.transcription import TranscriptionService


def _stub_raw(
    segments: list[dict] | None = None,
    language: str | None = "en",
) -> dict:
    return {
        "language": language,
        "segments": segments or [],
        "text": "".join(s.get("text", "") for s in (segments or [])),
    }


def test_transcribe_normalizes_segments_and_words():
    raw = _stub_raw(
        segments=[
            {
                "start": 0.0,
                "end": 1.5,
                "text": " hello world",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": " hello"},
                    {"start": 0.6, "end": 1.5, "word": " world"},
                ],
            },
            {
                "start": 1.6,
                "end": 3.0,
                "text": " more text",
                "words": [
                    {"start": 1.6, "end": 3.0, "word": " more text"},
                ],
            },
        ],
    )
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=raw,
    ):
        result = svc.transcribe(Path("/fake/vocals.wav"))

    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.segments[0] == {
        "start": 0.0,
        "end": 1.5,
        "text": " hello world",
        "words": [
            {"start": 0.0, "end": 0.5, "word": " hello"},
            {"start": 0.6, "end": 1.5, "word": " world"},
        ],
    }
    # Duration is the last segment's end timestamp.
    assert result.duration_seconds == 3.0


def test_transcribe_handles_missing_words_key():
    """mlx-whisper sometimes returns segments with no `words` key when
    word_timestamps fails on a chunk — the normalizer should treat that as
    an empty word list, not error."""
    raw = _stub_raw(
        segments=[
            {"start": 0.0, "end": 2.0, "text": " hello"},
        ]
    )
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=raw,
    ):
        result = svc.transcribe(Path("/fake/vocals.wav"))

    assert result.segments[0]["words"] == []
    assert result.duration_seconds == 2.0


def test_transcribe_empty_result_zero_duration():
    raw = _stub_raw(segments=[], language=None)
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=raw,
    ):
        result = svc.transcribe(Path("/fake/vocals.wav"))

    assert result.segments == []
    assert result.duration_seconds == 0.0
    assert result.language is None


def test_transcribe_passes_full_override_set():
    """Every override from WHISPER_OPTIONS must flow through to mlx-whisper.

    These are the tuned values for music + Demucs-leaked vocals — a
    silent regression would either re-introduce the "Thank you"
    hallucination loop, eat real lyric repetition, or hallucinate
    "(music)" / "[applause]" tags. The test pins every knob so a
    one-line edit to WHISPER_OPTIONS gets flagged."""
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=_stub_raw(),
    ) as mock_transcribe:
        svc.transcribe(Path("/fake/vocals.wav"))

    mock_transcribe.assert_called_once()
    _, kwargs = mock_transcribe.call_args
    assert "mlx-community" in kwargs.get("path_or_hf_repo", "")

    # Hallucination / quality guards
    assert kwargs.get("compression_ratio_threshold") == 3.0
    assert kwargs.get("logprob_threshold") == -1.2
    assert kwargs.get("no_speech_threshold") == 0.45
    assert kwargs.get("hallucination_silence_threshold") == 1.5
    assert kwargs.get("condition_on_previous_text") is False
    assert kwargs.get("temperature") == (0.0, 0.2)

    # Content / formatting
    assert "vocals" in (kwargs.get("initial_prompt") or "").lower()
    assert kwargs.get("word_timestamps") is True
    assert kwargs.get("clip_timestamps") == "0"

    # Decoder options
    assert kwargs.get("language") == "en"
    assert kwargs.get("task") == "transcribe"
    assert kwargs.get("beam_size") == 5
    assert kwargs.get("best_of") == 3
