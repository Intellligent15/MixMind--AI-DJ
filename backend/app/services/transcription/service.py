from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mlx_whisper

logger = logging.getLogger(__name__)

# MLX-converted weights for whisper large-v3 — spec's locked V1 model.
# We A/B-tested large-v3-turbo (fewer decoder layers, ~5-6× faster on
# MPS) twice. Both times turbo was measurably worse on sparse-vocal
# tracks (the "Everything is romantic" failure was the clearest signal),
# and the speed win wasn't worth the quality loss for a single-user DJ
# app where latency is bounded by Demucs separation anyway. Decision
# locked to large-v3; do not re-attempt turbo without a quality plan.
# mlx-whisper downloads + caches the weights from HuggingFace on first
# call; subsequent calls are fast. Transcription.model_name is stored
# per-row so historical rows from earlier model choices stay accurate.
DEFAULT_MODEL_REPO = "mlx-community/whisper-large-v3-mlx"
DEFAULT_MODEL_NAME = "large-v3"


# Full mlx-whisper override set. Kept as a module-level dict so tuning is
# one place to look. Inline comments explain why each knob is set where it
# is — defaults that are functionally no-ops are included for documentation
# (so a future reader can see the full set of decisions). See
# docs/the notes deviation #9 for the model history.
WHISPER_OPTIONS: dict = {
    # --- Hallucination / quality guards ---
    # gzip-ratio decode-fail threshold. Default 2.4 was eating real lyric
    # repetition ("hotter than the bluest flame hotter than the bluest
    # flame" → deduped). 3.0 still catches "Thank you. Thank you. Thank
    # you." style hallucinations.
    "compression_ratio_threshold": 3.0,
    # avg log-prob floor below which a decode is marked failed. Default
    # -1.0; slightly more permissive at -1.2 since music is harder than
    # clean speech.
    "logprob_threshold": -1.2,
    # P(no_speech) threshold above which a segment is treated as silence
    # (and dropped). Default 0.6; more aggressive at 0.45 to trim
    # near-silent Demucs leakage.
    "no_speech_threshold": 0.45,
    # Skip text aligned with >1.5s of silence when the model suspects
    # hallucination. Slightly tighter than the previous 2.0.
    "hallucination_silence_threshold": 1.5,
    # Decode each 30s window independently. Blocks the "Thank you"
    # feedback loop on sparse vocals — under evaluation against real-
    # world tracks.
    "condition_on_previous_text": False,
    # Shorter fallback ladder. If decoding at 0.0 fails, retry once at
    # 0.2 then give up — high-temperature retries (0.4..1.0) tend to
    # hallucinate more than they recover.
    "temperature": (0.0, 0.2),

    # --- Content / formatting ---
    # Biases the model toward vocal content. Whisper's tendency to add
    # "(music)" / "[applause]" disappears with this prompt.
    "initial_prompt": (
        "Transcribe only the sung or spoken vocals. Do not add "
        "descriptions of music, applause, silence, or instrumental "
        "sounds."
    ),
    # DTW pass for per-word timestamps. Required for word-anchored DJ
    # cuts in Phase 7+ and for the planned vocal-safety logic.
    "word_timestamps": True,
    # mlx-whisper defaults, listed explicitly for documentation. These
    # control which punctuation merges into adjacent word tokens.
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    # Transcribe the full file. Could be narrowed to specific ranges
    # later for partial re-runs.
    "clip_timestamps": "0",

    # --- Decoder options (mlx-whisper routes these to DecodingOptions) ---
    # Force English. Speeds up first-pass (skips lang detect) and
    # prevents accidental non-English decoding on instrumental-heavy
    # tracks where Whisper sometimes guesses Italian / Japanese. Change
    # / remove if you start queueing multilingual songs.
    "language": "en",
    "task": "transcribe",
    # mlx-whisper hasn't implemented beam search yet (decoding.py raises
    # NotImplementedError if beam_size is set), so we stick with greedy
    # decoding at temperature 0.0 and let `best_of` kick in only when the
    # temperature-fallback ladder advances past 0.0. `patience` is also
    # off — mlx-whisper's _verify_options rejects "patience without
    # beam_size" as a contradiction. Re-enable both together when
    # mlx-whisper ships beam search.
    # "beam_size": 5,
    # "patience": 1.0,
    "best_of": 3,
    # [-1] is mlx-whisper's standard "suppress all special / non-speech
    # tokens" sentinel. Don't change without reading mlx-whisper docs.
    "suppress_tokens": [-1],
    "suppress_blank": True,
    "prompt": None,
    "prefix": None,
    "length_penalty": None,
}


def _maybe_float(d: dict, key: str) -> float | None:
    """Read a numeric field from an mlx-whisper segment/word dict.

    Returns None when the field is absent (stubbed test fixtures) or the
    raw value is None. Keeps the normalizer terse and the JSONB payload
    consistently typed for downstream consumers.
    """
    v = d.get(key)
    return None if v is None else float(v)


@dataclass
class TranscriptionResult:
    """In-memory output of mlx-whisper.

    `segments` is the canonical wire shape persisted into the JSONB column:
      {
        start, end, text,
        avg_logprob, no_speech_prob, compression_ratio, temperature,
        words: [{start, end, word, probability}],
      }
    Confidence fields are nullable — mlx-whisper populates them on every
    real decode but a stubbed test result might omit them. Word entries
    preserve mlx-whisper's leading-space convention on `word`, so
    re-joining segments by concatenation produces fluent text.

    The per-word `probability` and per-segment `avg_logprob` are the
    inputs to Phase 7+'s vocal-safety logic (see ai-dj-spec.md → Vocal
    Safety Model). Persisting them now avoids re-transcribing every song
    when we land that work.
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

    def transcribe(self, vocals_path: Path, initial_prompt: str | None = None) -> TranscriptionResult:
        logger.info(
            "transcribing %s with mlx-whisper %s", vocals_path, self.model_repo
        )
        options = dict(WHISPER_OPTIONS)
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        raw = mlx_whisper.transcribe(
            str(vocals_path),
            path_or_hf_repo=self.model_repo,
            **options,
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
                        # Per-word confidence — Phase 7+ vocal-safety
                        # input. None on stubbed test data.
                        "probability": _maybe_float(w, "probability"),
                    }
                )
            segments.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": str(seg["text"]),
                    "words": words,
                    # Per-segment confidence signals from mlx-whisper. All
                    # are populated on real decodes; we keep them nullable
                    # so stubbed tests + future model swaps that drop a
                    # field don't break the normalizer.
                    "avg_logprob": _maybe_float(seg, "avg_logprob"),
                    "no_speech_prob": _maybe_float(seg, "no_speech_prob"),
                    "compression_ratio": _maybe_float(
                        seg, "compression_ratio"
                    ),
                    "temperature": _maybe_float(seg, "temperature"),
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
