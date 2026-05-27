"""Lyrics alignment via Needleman-Wunsch over word sequences.

Aligns authoritative Genius lyric text against Whisper's per-word
timestamps to produce a per-Genius-word
`(word, start, end, confidence, source)` sequence. See
ai-dj-spec.md → "Lyrics Alignment Model".

Algorithm:
  1. Normalize Genius text — strip ``[Chorus]`` style section headers,
     expand ``(×N)`` / ``(xN)`` repeat markers into the actual repeated
     block (Genius's shorthand for repeated choruses).
  2. Tokenize both sides into words; compute lowercase-alnum normal form
     and Soundex phonetic code for each.
  3. Standard Needleman-Wunsch DP with substitution scoring:
       exact match  → +2  (whisper_match)
       phonetic hit → +1  (whisper_substitution — Whisper got the word
                            wrong but the sound is right, so the timing
                            is almost certainly trustworthy)
       other        → -1  (whisper_substitution, lower confidence)
       gap          → -1
  4. Walk the traceback. Gaps in Whisper (Genius word the model missed)
     are placeholders that pass 2 fills in by linear interpolation
     between flanking anchored timestamps. Gaps in Genius (a word
     Whisper transcribed that doesn't exist in the lyrics, i.e.
     hallucinations) are dropped.
  5. Quality aggregate: ``(matches + 0.5 * substitutions) / total``.
     Below 0.3 we mark the alignment ``low_quality`` so downstream
     consumers (vocal_safety, Phase 9 LLM) know to fall back to
     vocal-safety-only logic.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.models.lyrics import LyricsAlignmentStatus

logger = logging.getLogger(__name__)


# --- Text normalization --------------------------------------------------

_BRACKET_RE = re.compile(r"\[.*?\]")
# Genius uses U+00D7 (×) or ASCII 'x' followed by a digit, optionally in
# parens. "(×4)", "(x4)", "x4" all show up in the wild.
_REPEAT_RE = re.compile(r"\(\s*[x×]\s*(\d+)\s*\)", re.IGNORECASE)


def _strip_section_headers(text: str) -> str:
    """Remove ``[Chorus]`` / ``[Verse 2]`` / etc."""
    return _BRACKET_RE.sub("", text)


def _expand_repeat_markers(text: str) -> str:
    """Expand ``(×N)`` markers: repeat the immediately preceding
    block (the contiguous run of non-blank lines ending at the marker
    line) so it appears N times total.

    Example::
        Verse A
        Verse B
        (×3)

    becomes::
        Verse A
        Verse B
        Verse A
        Verse B
        Verse A
        Verse B
    """
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        m = _REPEAT_RE.search(line)
        if not m:
            out.append(line)
            continue

        n = int(m.group(1))
        # The block we're repeating is everything appended to `out`
        # since the last blank line (or since the start of the doc).
        block_end = len(out)
        block_start = block_end
        for k in range(block_end - 1, -1, -1):
            if out[k].strip() == "":
                block_start = k + 1
                break
            block_start = k
        block = out[block_start:block_end]

        # The block currently appears once. Add (n - 1) more copies.
        for _ in range(max(0, n - 1)):
            out.extend(block)

        # Preserve any remaining text on the marker line.
        remainder = _REPEAT_RE.sub("", line).strip()
        if remainder:
            out.append(remainder)
    return "\n".join(out)


def _normalize(word: str) -> str:
    """Lowercase, strip non-alphanumeric. Empty string for pure punctuation."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


# Soundex consonant classes. Vowels (aeiouy) and h/w get dropped during
# the encoding pass, matching the classical algorithm.
_SOUNDEX_MAP = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
}
# H and W are "transparent" in classical American Soundex — they don't
# count toward the code, but they also don't reset the dedupe state, so
# ``Ashcraft`` collapses S+C (both class 2) into a single ``2`` digit
# even though an H sits between them. Vowels (and Y) DO reset.
_SOUNDEX_TRANSPARENT = {"h", "w"}


def _soundex(word: str) -> str:
    """Classical American Soundex: first letter (uppercase) + up to 3
    digit codes. Returns empty string for words with no alphabetic
    content."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    first = w[0]
    digits: list[str] = []
    last_code = _SOUNDEX_MAP.get(first, "")
    for ch in w[1:]:
        code = _SOUNDEX_MAP.get(ch)
        if code is None:
            if ch in _SOUNDEX_TRANSPARENT:
                # Don't reset last_code — adjacent same-class
                # consonants separated by H/W still collapse.
                continue
            last_code = ""  # vowel (or y) — resets dedupe state
            continue
        if code == last_code:
            continue
        digits.append(code)
        last_code = code
        if len(digits) == 3:
            break
    return first.upper() + "".join(digits).ljust(3, "0")


def _tokenize_genius(text: str) -> list[str]:
    """Genius → ordered list of display words (original casing/punctuation)."""
    cleaned = _strip_section_headers(text)
    expanded = _expand_repeat_markers(cleaned)
    return [w for w in re.split(r"\s+", expanded) if w and _normalize(w)]


def _extract_whisper_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten Whisper segments → ordered list of word entries with
    normalized form + Soundex + start/end/probability/seg_logprob."""
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        seg_logprob = seg.get("avg_logprob", 0.0)
        for w in seg.get("words") or []:
            if not w.get("word"):
                continue
            norm = _normalize(w["word"])
            if not norm:
                continue
            out.append({
                "word": w["word"],
                "norm": norm,
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "probability": float(w.get("probability", 1.0)),
                "seg_logprob": float(seg_logprob),
            })
    return out


# --- Needleman-Wunsch ----------------------------------------------------

SCORE_EXACT = 2
SCORE_PHONETIC = 1
SCORE_SUBSTITUTION = -1
SCORE_GAP = -1


def _sub_score(g_norm: str, w_norm: str, g_phon: str, w_phon: str) -> int:
    if g_norm == w_norm:
        return SCORE_EXACT
    if g_phon and g_phon == w_phon:
        return SCORE_PHONETIC
    return SCORE_SUBSTITUTION


def _nw_traceback(
    g_norm: list[str],
    w_norm: list[str],
    g_phon: list[str],
    w_phon: list[str],
) -> list[tuple[int | None, int | None]]:
    """Run Needleman-Wunsch over the two sequences and return a list of
    (g_idx, w_idx) operations from left to right:

      - (gi, wj)         match or substitution
      - (gi, None)       Genius word with no Whisper counterpart (gap in Whisper)
      - (None, wj)       Whisper word with no Genius counterpart (dropped)
    """
    n, m = len(g_norm), len(w_norm)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        H[i][0] = i * SCORE_GAP
    for j in range(m + 1):
        H[0][j] = j * SCORE_GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _sub_score(g_norm[i-1], w_norm[j-1], g_phon[i-1], w_phon[j-1])
            H[i][j] = max(
                H[i-1][j-1] + s,
                H[i-1][j] + SCORE_GAP,
                H[i][j-1] + SCORE_GAP,
            )

    ops: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 and j > 0:
        s = _sub_score(g_norm[i-1], w_norm[j-1], g_phon[i-1], w_phon[j-1])
        if H[i][j] == H[i-1][j-1] + s:
            ops.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif H[i][j] == H[i-1][j] + SCORE_GAP:
            ops.append((i - 1, None))
            i -= 1
        else:
            ops.append((None, j - 1))
            j -= 1
    while i > 0:
        ops.append((i - 1, None))
        i -= 1
    while j > 0:
        ops.append((None, j - 1))
        j -= 1
    ops.reverse()
    return ops


def _interpolate_gaps(aligned: list[dict[str, Any]]) -> None:
    """In-place: fill (start, end) for entries whose source is
    ``interpolated`` and whose timestamps are None, using the nearest
    anchored neighbours on each side. Leaves timestamps as None when
    there is no anchor at all (pathological case)."""
    for i, a in enumerate(aligned):
        if a["start"] is not None:
            continue

        prev_idx: int | None = None
        next_idx: int | None = None
        for k in range(i - 1, -1, -1):
            if aligned[k]["start"] is not None:
                prev_idx = k
                break
        for k in range(i + 1, len(aligned)):
            if aligned[k]["start"] is not None:
                next_idx = k
                break

        if prev_idx is not None and next_idx is not None:
            prev_end = aligned[prev_idx]["end"]
            next_start = aligned[next_idx]["start"]
            gap_span = max(next_start - prev_end, 1e-3)
            slot = (i - prev_idx) / (next_idx - prev_idx)
            t_center = prev_end + slot * gap_span
            per_word = gap_span / (next_idx - prev_idx)
            a["start"] = t_center
            a["end"] = t_center + per_word
            a["confidence"] = 0.2
        elif prev_idx is not None:
            prev_end = aligned[prev_idx]["end"]
            a["start"] = prev_end + 0.05
            a["end"] = prev_end + 0.15
            a["confidence"] = 0.1
        elif next_idx is not None:
            next_start = aligned[next_idx]["start"]
            a["start"] = max(0.0, next_start - 0.15)
            a["end"] = max(0.0, next_start - 0.05)
            a["confidence"] = 0.1
        # else: no anchor on either side — leave start/end as None.


# --- Public API ---------------------------------------------------------

QUALITY_LOW_THRESHOLD = 0.3


def align_lyrics(
    transcription_segments: list[dict[str, Any]],
    genius_text: str,
) -> dict[str, Any]:
    """Align Genius text against Whisper output.

    Returns a dict with three keys:
      - ``aligned_words``      list of {word, start, end, confidence, source}
      - ``alignment_quality``  float in [0, 1]
      - ``alignment_status``   LyricsAlignmentStatus
    """
    if not genius_text or not genius_text.strip():
        return {
            "aligned_words": [],
            "alignment_quality": 0.0,
            "alignment_status": LyricsAlignmentStatus.error,
        }

    genius_words = _tokenize_genius(genius_text)
    if not genius_words:
        return {
            "aligned_words": [],
            "alignment_quality": 0.0,
            "alignment_status": LyricsAlignmentStatus.error,
        }

    whisper_words = _extract_whisper_words(transcription_segments)

    if not whisper_words:
        # Genius text exists but Whisper produced nothing — emit Genius
        # words with no timestamps. Mark low_quality so downstream
        # consumers know not to use these for time-anchored cuts.
        aligned = [{
            "word": g,
            "start": None,
            "end": None,
            "confidence": 0.0,
            "source": "interpolated",
        } for g in genius_words]
        return {
            "aligned_words": aligned,
            "alignment_quality": 0.0,
            "alignment_status": LyricsAlignmentStatus.low_quality,
        }

    g_norm = [_normalize(w) for w in genius_words]
    w_norm = [w["norm"] for w in whisper_words]
    g_phon = [_soundex(w) for w in g_norm]
    w_phon = [_soundex(w) for w in w_norm]

    ops = _nw_traceback(g_norm, w_norm, g_phon, w_phon)

    aligned: list[dict[str, Any]] = []
    for gi, wi in ops:
        if gi is None:
            # Whisper word with no Genius counterpart — drop.
            continue
        if wi is None:
            # Genius word the model missed — placeholder, pass 2 fills.
            aligned.append({
                "word": genius_words[gi],
                "start": None,
                "end": None,
                "confidence": 0.0,
                "source": "interpolated",
            })
            continue

        w = whisper_words[wi]
        if g_norm[gi] == w_norm[wi]:
            aligned.append({
                "word": genius_words[gi],
                "start": w["start"],
                "end": w["end"],
                "confidence": w["probability"],
                "source": "whisper_match",
            })
        else:
            # Substitution. Whisper's timestamp is the trustworthy part —
            # the word it picked is the unreliable one. Confidence
            # bakes in whether the substitution at least sounded right.
            phonetic_hit = bool(g_phon[gi]) and g_phon[gi] == w_phon[wi]
            sub_factor = 0.7 if phonetic_hit else 0.4
            aligned.append({
                "word": genius_words[gi],
                "start": w["start"],
                "end": w["end"],
                "confidence": w["probability"] * sub_factor,
                "source": "whisper_substitution",
            })

    _interpolate_gaps(aligned)

    # Quality aggregate: matches weigh 1.0; phonetically-supported
    # substitutions (Whisper got the sound right, just the word wrong)
    # weigh 0.5; substitutions where even the phonetics don't match
    # weigh 0.1 — they're effectively noise that NW paired up because
    # the gap penalty would have been worse. Interpolations contribute
    # nothing — we don't know if those words are really where we put them.
    matches = 0
    phonetic_subs = 0
    other_subs = 0
    for gi, wi in ops:
        if gi is None or wi is None:
            continue
        if g_norm[gi] == w_norm[wi]:
            matches += 1
        elif g_phon[gi] and g_phon[gi] == w_phon[wi]:
            phonetic_subs += 1
        else:
            other_subs += 1
    total = matches + phonetic_subs + other_subs
    # Add interpolated entries to the denominator so a Genius word the
    # model missed drags quality down.
    interpolated_count = sum(1 for a in aligned if a["source"] == "interpolated")
    denom = total + interpolated_count
    quality = (
        (matches + 0.5 * phonetic_subs + 0.1 * other_subs) / denom
        if denom > 0 else 0.0
    )
    status = (
        LyricsAlignmentStatus.low_quality
        if quality < QUALITY_LOW_THRESHOLD
        else LyricsAlignmentStatus.success
    )

    logger.info(
        "align_lyrics: %d genius / %d whisper → %d matches, %d phonetic subs, "
        "%d other subs, %d interpolated, quality=%.2f",
        len(genius_words), len(whisper_words),
        matches, phonetic_subs, other_subs, interpolated_count, quality,
    )

    return {
        "aligned_words": aligned,
        "alignment_quality": float(quality),
        "alignment_status": status,
    }
