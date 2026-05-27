"""Compute vocal-safe regions of a song — intervals where a transition
can safely place a hard cut, stem swap, or drop swap without chopping
a syllable.

See ai-dj-spec.md → "Vocal Safety Model" for the design. The short
version: Whisper alone gives false positives (hallucinated text on
silence) and false negatives (missed quiet vocals), so we cross-reference
Whisper words against the vocal stem's frame-wise RMS + peak envelope
written by ``separate_stems``. A region is "safe" when:

  - No usable vocal word overlaps it.
  - All envelope frames inside it are below the quiet thresholds (with
    a small tolerance for Demucs stem bleed — see ``QUIET_NOISE_TOLERANCE``).
  - It is at least ``min_safe_region_seconds`` wide.

The envelope sidecar's shape is
``{"frame_hz": int, "rms": [...], "peak": [...]}`` (written by
``app/services/stems/service.py::_compute_vocal_envelope``). The legacy
``hop_seconds`` key is also tolerated for forward compatibility.
"""

from __future__ import annotations

from typing import Any


# Allow up to this fraction of frames in a candidate quiet gap to exceed
# the quiet thresholds. Demucs stem bleed lets transient drum / synth
# energy leak into the vocal stem; without this tolerance every
# percussion-heavy song would have zero safe regions.
QUIET_NOISE_TOLERANCE = 0.15


def _hop_seconds(envelope: dict[str, Any]) -> float:
    """Resolve the per-frame interval from the envelope sidecar.

    Accepts either ``frame_hz`` (canonical, written by Phase 5) or
    ``hop_seconds`` (legacy). Falls back to 100 ms (= frame_hz 10).
    """
    if "frame_hz" in envelope and envelope["frame_hz"]:
        return 1.0 / float(envelope["frame_hz"])
    if "hop_seconds" in envelope and envelope["hop_seconds"]:
        return float(envelope["hop_seconds"])
    return 0.1


def _envelope_value(values: list[float], idx: int) -> float:
    if not values:
        return 0.0
    if idx < 0:
        return values[0]
    if idx >= len(values):
        return values[-1]
    return float(values[idx])


def _quiet_fraction_ok(
    rms_list: list[float],
    peak_list: list[float],
    start_idx: int,
    end_idx: int,
    rms_quiet: float,
    peak_quiet: float,
) -> bool:
    """True if the (start_idx, end_idx) frame window passes the quiet
    test: at most QUIET_NOISE_TOLERANCE of frames may exceed either
    threshold. Empty windows are vacuously quiet."""
    total = 0
    noisy = 0
    for i in range(start_idx, end_idx):
        if i < 0 or i >= len(rms_list):
            continue
        total += 1
        rms_val = rms_list[i]
        peak_val = _envelope_value(peak_list, i) if peak_list else 0.0
        if rms_val >= rms_quiet or peak_val >= peak_quiet:
            noisy += 1
    if total == 0:
        return True
    return (noisy / total) < QUIET_NOISE_TOLERANCE


def _word_has_audio_support(
    rms_list: list[float],
    peak_list: list[float],
    hop: float,
    word_start: float,
    word_end: float,
    rms_presence: float,
    peak_presence: float,
) -> bool:
    """Confirm the vocal stem actually has energy near the word's
    timestamp. Without this check a hallucinated "Thank you" on silence
    would block transitions in a perfectly safe region."""
    mid = (word_start + word_end) / 2.0
    idx = int(mid / hop)
    rms_val = _envelope_value(rms_list, idx)
    peak_val = _envelope_value(peak_list, idx) if peak_list else 0.0
    # Either signal hot enough is sufficient — drum-heavy tracks can
    # have low RMS but real percussive vocal hits.
    return rms_val >= rms_presence or peak_val >= peak_presence


def _collect_word_intervals_from_alignment(
    aligned_words: list[dict[str, Any]],
    rms_list: list[float],
    peak_list: list[float],
    hop: float,
    rms_presence: float,
    peak_presence: float,
) -> list[tuple[float, float]]:
    """Filter aligned-words → intervals worth respecting in transition
    planning. ``whisper_match`` and ``whisper_substitution`` are the
    high-confidence sources; ``interpolated`` words are tracked too
    when the audio actually has energy at the interpolated time."""
    out: list[tuple[float, float]] = []
    for w in aligned_words:
        src = w.get("source")
        if src not in ("whisper_match", "whisper_substitution", "interpolated"):
            continue
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            continue
        if not _word_has_audio_support(
            rms_list, peak_list, hop, start, end, rms_presence, peak_presence
        ):
            continue
        out.append((float(start), float(end)))
    return out


def _collect_word_intervals_from_whisper(
    segments: list[dict[str, Any]],
    rms_list: list[float],
    peak_list: list[float],
    hop: float,
    word_prob_min: float,
    segment_logprob_min: float,
    rms_presence: float,
    peak_presence: float,
) -> list[tuple[float, float]]:
    """Same job but starting from raw Whisper segments — the fallback
    when no Genius alignment is available. Applies the per-segment and
    per-word confidence filters from the spec's ``usable_vocal_word``."""
    out: list[tuple[float, float]] = []
    for seg in segments or []:
        if seg.get("avg_logprob", 0.0) < segment_logprob_min:
            continue
        for w in seg.get("words") or []:
            if w.get("probability", 0.0) < word_prob_min:
                continue
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                continue
            if not _word_has_audio_support(
                rms_list, peak_list, hop, start, end, rms_presence, peak_presence
            ):
                continue
            out.append((float(start), float(end)))
    return out


def _merge_intervals(
    intervals: list[tuple[float, float]],
    join_gap: float = 0.2,
) -> list[tuple[float, float]]:
    """Sort + coalesce overlapping or nearly-adjacent intervals so the
    safe-gap inversion sees clean blocks."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + join_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def vocal_safe_regions(
    transcription_segments: list[dict[str, Any]],
    envelope: dict[str, Any],
    aligned_words: list[dict[str, Any]] | None = None,
    word_prob_min: float = 0.35,
    segment_logprob_min: float = -1.2,
    stem_rms_presence: float = 0.02,
    stem_peak_presence: float = 0.08,
    stem_rms_quiet: float = 0.01,
    stem_peak_quiet: float = 0.04,
    min_safe_region_seconds: float = 1.5,
    duration_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Compute the list of intervals during which a transition can
    safely place a hard cut, stem swap, or drop swap.

    Returns a list of ``{start, end, safe, reason}`` dicts. When no
    envelope is supplied the whole duration is treated as safe (callers
    can still bail before trusting it).
    """
    rms_list: list[float] = list(envelope.get("rms") or [])
    peak_list: list[float] = list(envelope.get("peak") or [])
    hop = _hop_seconds(envelope)

    if not rms_list:
        return [{
            "start": 0.0,
            "end": float(duration_seconds),
            "safe": True,
            "reason": "no_envelope",
        }]

    duration = float(duration_seconds) or (len(rms_list) * hop)

    if aligned_words is not None:
        intervals = _collect_word_intervals_from_alignment(
            aligned_words, rms_list, peak_list, hop,
            stem_rms_presence, stem_peak_presence,
        )
    else:
        intervals = _collect_word_intervals_from_whisper(
            transcription_segments, rms_list, peak_list, hop,
            word_prob_min, segment_logprob_min,
            stem_rms_presence, stem_peak_presence,
        )

    merged = _merge_intervals(intervals)

    safe_regions: list[dict[str, Any]] = []
    cursor = 0.0
    bounds = merged + [(duration, duration)]  # synthetic tail to flush the final gap
    for word_start, word_end in bounds:
        gap_start = cursor
        gap_end = word_start
        if gap_end - gap_start >= min_safe_region_seconds:
            start_idx = max(0, int(gap_start / hop))
            end_idx = min(len(rms_list), int(gap_end / hop))
            if _quiet_fraction_ok(
                rms_list, peak_list, start_idx, end_idx,
                stem_rms_quiet, stem_peak_quiet,
            ):
                safe_regions.append({
                    "start": gap_start,
                    "end": gap_end,
                    "safe": True,
                    "reason": "quiet_gap",
                })
        cursor = word_end

    return safe_regions
