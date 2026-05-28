"""Unit tests for the difflib-backed lyrics aligner with Soundex
phonetic backoff and (×N) chorus expansion. Pure function, no DB."""

from __future__ import annotations

from app.models.lyrics import LyricsAlignmentStatus
from app.services.lyrics_alignment.aligner import (
    _soundex,
    _expand_repeat_markers,
    align_lyrics,
)


def test_soundex_known_codes():
    # Classical American Soundex — transparent H/W, vowel reset.
    assert _soundex("Robert") == "R163"
    assert _soundex("Rupert") == "R163"
    assert _soundex("Ashcraft") == "A261"  # H is transparent, S+C collapse
    assert _soundex("Tymczak") == "T522"
    assert _soundex("Pfister") == "P236"
    assert _soundex("Honeyman") == "H555"


def test_soundex_empty_and_punctuation():
    assert _soundex("") == ""
    assert _soundex("!!!") == ""
    assert _soundex("a") == "A000"


def test_soundex_homophones_collide():
    assert _soundex("their") == _soundex("there") == "T600"


def test_expand_repeat_marker_unicode_times():
    text = "Line one\nLine two\n(×3)"
    out = _expand_repeat_markers(text)
    assert out.count("Line one") == 3
    assert out.count("Line two") == 3


def test_expand_repeat_marker_ascii_x():
    text = "A\nB\n(x2)"
    out = _expand_repeat_markers(text)
    assert out.count("A") == 2 and out.count("B") == 2


def test_expand_repeat_marker_respects_blank_line_boundary():
    text = "Intro\n\nChorus line A\nChorus line B\n(×2)"
    out = _expand_repeat_markers(text)
    assert out.count("Intro") == 1  # NOT repeated
    assert out.count("Chorus line A") == 2


def test_expand_repeat_marker_no_repeats_when_absent():
    text = "Just a verse\nNo marker here"
    out = _expand_repeat_markers(text)
    assert out == text


def _whisper(words: list[tuple[str, float, float, float]], avg_logprob: float = -0.2) -> list[dict]:
    return [{
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
    }]


def test_align_expands_repeated_chorus_marker():
    # Whisper transcribes the chorus 3×; Genius lists it once with "(×3)".
    segs = _whisper([
        ("Hey", 0.0, 0.5, 0.95), ("now", 0.5, 1.0, 0.95),
        ("Hey", 2.0, 2.5, 0.95), ("now", 2.5, 3.0, 0.95),
        ("Hey", 4.0, 4.5, 0.95), ("now", 4.5, 5.0, 0.95),
    ])
    out = align_lyrics(segs, "Hey now\n(×3)")
    aligned = out["aligned_words"]
    assert len(aligned) == 6
    assert sum(1 for a in aligned if a["source"] == "whisper_match") == 6
    assert aligned[0]["start"] == 0.0
    assert aligned[-1]["end"] == 5.0


def test_align_phonetic_substitution_high_confidence():
    # "there" vs "their" — same Soundex (T600). Substitution with
    # elevated confidence because phonetics match.
    segs = _whisper([("there", 1.0, 1.3, 0.9)])
    out = align_lyrics(segs, "their")
    a = out["aligned_words"][0]
    assert a["source"] == "whisper_substitution"
    # Phonetic substitution factor = 0.7; raw prob 0.9 → 0.63
    assert a["confidence"] > 0.5


def test_align_nonphonetic_substitution_lower_confidence():
    # "flame" vs "name" — completely different phonetics, low conf.
    segs = _whisper([("flame", 0.7, 1.0, 0.9)])
    out = align_lyrics(segs, "name")
    a = out["aligned_words"][0]
    assert a["source"] == "whisper_substitution"
    # Non-phonetic factor 0.4 → 0.36; well below the phonetic case.
    assert a["confidence"] < 0.5


def test_align_interpolated_word_has_nonzero_duration():
    # "the" is missing in Whisper between "hold" and "line".
    segs = _whisper([
        ("hold", 0.0, 0.3, 0.95),
        ("line", 1.0, 1.3, 0.95),
    ])
    out = align_lyrics(segs, "hold the line")
    interp = [a for a in out["aligned_words"] if a["source"] == "interpolated"]
    assert len(interp) == 1
    assert interp[0]["start"] < interp[0]["end"]  # non-zero duration
    assert 0.3 <= interp[0]["start"] <= 1.0


def test_align_no_anchors_leaves_timestamps_none():
    # Whisper produced no usable words at all.
    out = align_lyrics([], "love me tender")
    assert all(a["start"] is None for a in out["aligned_words"])
    assert all(a["end"] is None for a in out["aligned_words"])
    assert out["alignment_status"] == LyricsAlignmentStatus.low_quality


def test_align_quality_weighs_phonetic_subs_at_half():
    # 3 matches + 1 phonetic substitution (there ↔ their) →
    # (3 + 0.5) / 4 = 0.875.
    segs = _whisper([
        ("love", 0.0, 0.3, 0.95),
        ("me", 0.4, 0.6, 0.95),
        ("there", 0.7, 0.9, 0.95),
        ("forever", 1.0, 1.5, 0.95),
    ])
    out = align_lyrics(segs, "love me their forever")
    assert abs(out["alignment_quality"] - 0.875) < 1e-6
    assert out["alignment_status"] == LyricsAlignmentStatus.success


def test_align_low_quality_when_mostly_misaligned():
    # 0 matches + 4 non-phonetic subs → (0 + 0.4) / 4 = 0.1 < 0.3.
    segs = _whisper([
        ("zzz", 0.0, 0.3, 0.9), ("qqq", 0.3, 0.6, 0.9),
        ("xxx", 0.6, 0.9, 0.9), ("yyy", 0.9, 1.2, 0.9),
    ])
    out = align_lyrics(segs, "love me tender forever")
    assert out["alignment_quality"] < 0.3
    assert out["alignment_status"] == LyricsAlignmentStatus.low_quality
