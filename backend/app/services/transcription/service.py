from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mlx_whisper

logger = logging.getLogger(__name__)

# MLX-converted weights for whisper large-v3 — the spec's locked V1 model.
# We briefly ran large-v3-turbo (fewer decoder layers, ~5-6× faster on
# MPS) but it was measurably more prone to the "Thank you" hallucination
# loop on sparse-vocal tracks (Charli xcx "Everything is romantic" was
# the canonical failure). large-v3's extra decoder depth carries enough
# context to ride out the same silent stretches. mlx-whisper downloads +
# caches the weights from HuggingFace on first call; subsequent calls are
# fast. Transcription.model_name is stored per-row so historical rows
# from earlier model choices stay accurate.
DEFAULT_MODEL_REPO = "mlx-community/whisper-large-v3-mlx"
DEFAULT_MODEL_NAME = "large-v3"


@dataclass
class TranscriptionResult:
    """In-memory output of mlx-whisper.

    `segments` is the canonical wire shape persisted into the JSONB column:
      {start, end, text, words: [{start, end, word}]}
    Word entries preserve mlx-whisper's leading-space convention on `word`,
    so re-joining segments by concatenation produces fluent text.
    """

    language: str | None
    segments: list[dict]
    duration_seconds: float


class TranscriptionService:
    """Wraps mlx-whisper for vocal-stem transcription.

    Pure function: no DB, no storage I/O beyond reading the vocal WAV path.
    mlx-whisper exposes a module-level `transcribe()` and caches the model
    weights internally between calls, so there's no model handle to load
    eagerly — the first call inside a fresh worker pays the warm-up cost.
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.model_repo = model_repo
        self.model_name = model_name

    def transcribe(self, vocals_path: Path) -> TranscriptionResult:
        logger.info(
            "transcribing %s with mlx-whisper %s", vocals_path, self.model_repo
        )
        # hallucination_silence_threshold=2.0 leans on word_timestamps to
        # skip text that lines up with >2s of silence when the model
        # suspects it has started hallucinating — a free guard since we
        # already turn word_timestamps on.
        #
        # condition_on_previous_text=False decodes each 30 s window
        # independently, breaking Whisper's window-to-window feedback
        # loop. We tried it both ways under large-v3-turbo and bounced
        # back; under large-v3 the decoder depth carries enough context
        # within a window that we don't need the previous-text crutch,
        # so we get the loop protection without paying as much narrative
        # cost. Currently under evaluation against real-world tracks.
        raw = mlx_whisper.transcribe(
            str(vocals_path),
            path_or_hf_repo=self.model_repo,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
        )

        segments: list[dict] = []
        for seg in raw.get("segments") or []:
            words: list[dict] = []
            for w in seg.get("words") or []:
                words.append(
                    {
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        # mlx-whisper preserves the leading space; keep it.
                        "word": str(w["word"]),
                    }
                )
            segments.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": str(seg["text"]),
                    "words": words,
                }
            )

        # mlx-whisper doesn't surface the source duration; use the last
        # segment's end timestamp as a usable proxy. Empty transcriptions
        # (Whisper produced no segments) report duration 0.0 — the
        # transcription-status column already tells callers it's not useful.
        duration = float(segments[-1]["end"]) if segments else 0.0
        language = raw.get("language")
        return TranscriptionResult(
            language=str(language) if language else None,
            segments=segments,
            duration_seconds=duration,
        )
