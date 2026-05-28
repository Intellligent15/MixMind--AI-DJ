import difflib
import re
from typing import Any

from app.models.lyrics import LyricsAlignmentStatus


def _clean_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


_SOUNDEX_MAP = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
}
# H and W are "transparent" in classical American Soundex — they don't
# count toward the code, but they also don't reset the dedupe state.
# Vowels (and Y) DO reset.
_SOUNDEX_TRANSPARENT = {"h", "w"}


def _soundex(word: str) -> str:
    """Classical American Soundex: first letter + up to 3 digit codes."""
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
                continue  # transparent — preserve last_code
            last_code = ""  # vowel/y — resets dedupe
            continue
        if code == last_code:
            continue
        digits.append(code)
        last_code = code
        if len(digits) == 3:
            break
    return first.upper() + "".join(digits).ljust(3, "0")


# Genius uses U+00D7 (×) or ASCII 'x' followed by a digit, optionally
# parenthesised. "(×4)", "(x4)" are the common shapes.
_REPEAT_RE = re.compile(r"\(\s*[x×]\s*(\d+)\s*\)", re.IGNORECASE)


def _expand_repeat_markers(text: str) -> str:
    """Expand ``(×N)`` markers: repeat the immediately preceding block
    (the contiguous run of non-blank lines ending at the marker line)
    so it appears N times total. Strips the marker."""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        m = _REPEAT_RE.search(line)
        if not m:
            out.append(line)
            continue

        n = int(m.group(1))
        # Find the block we're repeating — everything appended since
        # the last blank line (or the doc start).
        block_end = len(out)
        block_start = block_end
        for k in range(block_end - 1, -1, -1):
            if out[k].strip() == "":
                block_start = k + 1
                break
            block_start = k
        block = out[block_start:block_end]

        # Block currently appears once. Add (n - 1) more copies.
        for _ in range(max(0, n - 1)):
            out.extend(block)

        # Preserve any text on the marker line beyond the marker itself.
        remainder = _REPEAT_RE.sub("", line).strip()
        if remainder:
            out.append(remainder)
    return "\n".join(out)


def align_lyrics(transcription_segments: list[dict[str, Any]], genius_text: str) -> dict[str, Any]:
    """
    Align authoritative Genius text with Whisper transcription timestamps.
    Returns a dict with aligned_words, alignment_quality, and alignment_status.
    """
    if not genius_text:
        return {
            "aligned_words": [],
            "alignment_quality": 0.0,
            "alignment_status": LyricsAlignmentStatus.error,
        }

    # Split genius lyrics into words
    # Remove bracketed section headers like [Chorus]
    # NEW: strip section headers, then expand (×N) chorus markers
    # before tokenising. Order matters — markers like (×3) sit on
    # their own lines and must be visible to _expand_repeat_markers.
    clean_text = re.sub(r"\[.*?\]", "", genius_text)
    clean_text = _expand_repeat_markers(clean_text)
    genius_words = [w for w in re.split(r"\s+", clean_text) if w]
    genius_clean = [_clean_word(w) for w in genius_words]

    whisper_words = []
    for seg in transcription_segments:
        for w in seg.get("words", []):
            if w.get("word"):
                whisper_words.append({
                    "word": w["word"],
                    "clean": _clean_word(w["word"]),
                    "start": w["start"],
                    "end": w["end"],
                    "probability": w.get("probability", 1.0),
                    "seg_logprob": seg.get("avg_logprob", 0.0),
                })

    w_clean = [w["clean"] for w in whisper_words]

    matcher = difflib.SequenceMatcher(None, genius_clean, w_clean)

    aligned = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                w = whisper_words[j]
                aligned.append({
                    "word": genius_words[i],
                    "start": w["start"],
                    "end": w["end"],
                    "confidence": w["probability"],
                    "source": "whisper_match",
                })
        elif tag == "replace":
            for idx, i in enumerate(range(i1, i2)):
                j = j1 + idx
                if j < j2:
                    w = whisper_words[j]
                    g_phon = _soundex(genius_clean[i])
                    w_phon = _soundex(w["clean"])
                    # Phonetic match → trustworthy timing despite wrong
                    # word ("there"/"their"). Non-phonetic → likely
                    # pairing two unrelated words; low conf.
                    phonetic_hit = bool(g_phon) and g_phon == w_phon
                    sub_factor = 0.7 if phonetic_hit else 0.4
                    aligned.append({
                        "word": genius_words[i],
                        "start": w["start"],
                        "end": w["end"],
                        "confidence": w["probability"] * sub_factor,
                        "source": "whisper_substitution",
                        # Internal bookkeeping — stripped before return.
                        "_phonetic_hit": phonetic_hit,
                    })
                else:
                    aligned.append({
                        "word": genius_words[i],
                        "start": None,
                        "end": None,
                        "confidence": 0.0,
                        "source": "unmapped",
                    })
        elif tag == "delete":
            for i in range(i1, i2):
                aligned.append({
                    "word": genius_words[i],
                    "start": None,
                    "end": None,
                    "confidence": 0.0,
                    "source": "unmapped",
                })
        elif tag == "insert":
            # Words in Whisper not in Genius. Drop them.
            pass

    # Pass 2: Interpolate missing timestamps. Non-zero duration for
    # interpolated words; leave None when neither neighbour anchors.
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
            per_word = gap_span / (next_idx - prev_idx)
            t_center = prev_end + slot * gap_span
            a["start"] = t_center
            a["end"] = t_center + per_word
            a["source"] = "interpolated"
        elif prev_idx is not None:
            prev_end = aligned[prev_idx]["end"]
            a["start"] = prev_end + 0.05
            a["end"] = prev_end + 0.15
            a["source"] = "interpolated"
        elif next_idx is not None:
            next_start = aligned[next_idx]["start"]
            a["start"] = max(0.0, next_start - 0.15)
            a["end"] = max(0.0, next_start - 0.05)
            a["source"] = "interpolated"
        else:
            # No anchor on either side — leave timestamps None.
            # Downstream consumers (vocal_safety) must skip these.
            a["source"] = "interpolated"
            # a["start"] and a["end"] remain None.

    matched_count = sum(1 for a in aligned if a["source"] == "whisper_match")
    quality = matched_count / len(aligned) if aligned else 0.0
    status = LyricsAlignmentStatus.success
    if quality < 0.3:
        status = LyricsAlignmentStatus.low_quality

    for a in aligned:
        a.pop("_phonetic_hit", None)

    return {
        "aligned_words": aligned,
        "alignment_quality": quality,
        "alignment_status": status,
    }
