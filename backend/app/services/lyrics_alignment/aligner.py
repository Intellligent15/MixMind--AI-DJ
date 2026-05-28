import difflib
import re
from typing import Any

from app.models.lyrics import LyricsAlignmentStatus


def _clean_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


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
    clean_text = re.sub(r"\[.*?\]", "", genius_text)
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
                    sim = difflib.SequenceMatcher(None, genius_clean[i], w["clean"]).ratio()
                    aligned.append({
                        "word": genius_words[i],
                        "start": w["start"],
                        "end": w["end"],
                        "confidence": w["probability"] * sim,
                        "source": "whisper_substitution",
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

    # Pass 2: Interpolate missing timestamps
    for i, a in enumerate(aligned):
        if a["start"] is None:
            prev_end = None
            prev_idx = -1
            for k in range(i - 1, -1, -1):
                if aligned[k]["start"] is not None:
                    prev_end = aligned[k]["end"]
                    prev_idx = k
                    break

            next_start = None
            next_idx = len(aligned)
            for k in range(i + 1, len(aligned)):
                if aligned[k]["start"] is not None:
                    next_start = aligned[k]["start"]
                    next_idx = k
                    break

            if prev_end is not None and next_start is not None:
                fraction = (i - prev_idx) / (next_idx - prev_idx)
                duration = next_start - prev_end
                a["start"] = prev_end + fraction * duration
                a["end"] = prev_end + fraction * duration
                a["source"] = "interpolated"
            elif prev_end is not None:
                a["start"] = prev_end + 0.1
                a["end"] = prev_end + 0.2
                a["source"] = "interpolated"
            elif next_start is not None:
                a["start"] = max(0.0, next_start - 0.2)
                a["end"] = max(0.0, next_start - 0.1)
                a["source"] = "interpolated"
            else:
                a["start"] = 0.0
                a["end"] = 0.0
                a["source"] = "interpolated"

    matched_count = sum(1 for a in aligned if a["source"] == "whisper_match")
    quality = matched_count / len(aligned) if aligned else 0.0
    status = LyricsAlignmentStatus.success
    if quality < 0.3:
        status = LyricsAlignmentStatus.low_quality

    return {
        "aligned_words": aligned,
        "alignment_quality": quality,
        "alignment_status": status,
    }
