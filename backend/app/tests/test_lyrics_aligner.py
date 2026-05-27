"""Tests for the lyrics aligner — DTW alignment of Genius text against
Whisper timestamps. Pure function, no DB, no audio."""

from __future__ import annotations

from app.models.lyrics import LyricsAlignmentStatus
from app.services.lyrics_alignment.aligner import (
    _expand_repeat_markers,
    _soundex,
    _strip_section_headers,
    _tokenize_genius,
    align_lyrics,
)


# --- Soundex unit tests -------------------------------------------------


def test_soundex_known_codes():
    # Classical Soundex examples — Robert/Rupert collapse to R163, Ashcraft
    # to A261, Tymczak to T522, Pfister to P236, Honeyman to H555.
    assert _soundex("Robert") == "R163"
    assert _soundex("Rupert") == "R163"
    assert _soundex("Ashcraft") == "A261"
    assert _soundex("Tymczak") == "T522"
    assert _soundex("Pfister") == "P236"
    assert _soundex("Honeyman") == "H555"


def test_soundex_empty_and_punctuation():
    assert _soundex("") == ""
    assert _soundex("!!!") == ""
    assert _soundex("a") == "A000"


def test_soundex_homophones_collide():
    # Whisper "flame" vs Genius "name" — different first letters so they
    # *don't* collide (Soundex's known weakness). But "their / there /
    # they're" all share T600.
    assert _soundex("their") == _soundex("there") == "T600"


# --- Text-cleaning unit tests -------------------------------------------


def test_strip_section_headers():
    text = "[Chorus]\nLove me\n[Verse 2]\nTender"
    assert "[" not in _strip_section_headers(text)
    assert "Chorus" not in _strip_section_headers(text)


def test_expand_repeat_marker_basic():
    text = "Line one\nLine two\n(×3)"
    out = _expand_repeat_markers(text)
    # The two-line block should now appear 3 times.
    assert out.count("Line one") == 3
    assert out.count("Line two") == 3


def test_expand_repeat_marker_ascii_x():
    # Ascii 'x' variant — Genius isn't consistent.
    text = "A\nB\n(x2)"
    out = _expand_repeat_markers(text)
    assert out.count("A") == 2 and out.count("B") == 2


def test_expand_repeat_marker_respects_blank_line_block_boundary():
    text = "Intro\n\nChorus line A\nChorus line B\n(×2)"
    out = _expand_repeat_markers(text)
    # Only the post-blank chorus block repeats, not the intro.
    assert out.count("Intro") == 1
    assert out.count("Chorus line A") == 2


def test_tokenize_genius_drops_brackets_and_blank_chunks():
    text = "[Chorus]\nHello world\n[Verse]"
    words = _tokenize_genius(text)
    assert words == ["Hello", "world"]


# --- Whisper segment shape helper --------------------------------------


def _whisper(words: list[tuple[str, float, float, float]], avg_logprob: float = -0.2) -> list[dict]:
    """Build a single-segment Whisper transcription from a list of
    (word, start, end, probability) tuples."""
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


# --- align_lyrics behaviour tests --------------------------------------


def test_align_happy_path_all_matches():
    segs = _whisper([
        ("Love", 0.0, 0.5, 0.95),
        ("me", 0.6, 0.8, 0.95),
        ("tender", 1.0, 1.5, 0.95),
    ])
    out = align_lyrics(segs, "Love me tender")
    assert out["alignment_status"] == LyricsAlignmentStatus.success
    assert out["alignment_quality"] == 1.0
    assert len(out["aligned_words"]) == 3
    assert all(a["source"] == "whisper_match" for a in out["aligned_words"])
    # Timestamps come from Whisper.
    assert out["aligned_words"][0]["start"] == 0.0
    assert out["aligned_words"][2]["end"] == 1.5


def test_align_substitution_borrows_whisper_timing():
    # Whisper got "flame" — Genius truth is "name". Different first
    # letter (F vs N), so Soundex won't help — it's a low-confidence
    # substitution, but the timing is Whisper's.
    segs = _whisper([
        ("Hotter", 0.0, 0.5, 0.9),
        ("than", 0.5, 0.7, 0.9),
        ("flame", 0.7, 1.0, 0.9),
    ])
    out = align_lyrics(segs, "Hotter than name")
    assert out["alignment_status"] == LyricsAlignmentStatus.success
    aligned = out["aligned_words"]
    assert aligned[2]["word"] == "name"
    assert aligned[2]["source"] == "whisper_substitution"
    # Borrowed Whisper's start/end.
    assert aligned[2]["start"] == 0.7
    assert aligned[2]["end"] == 1.0
    # Lower confidence than a match.
    assert aligned[2]["confidence"] < 1.0


def test_align_phonetic_substitution_keeps_higher_confidence():
    # "there" vs "their" — same Soundex (T600). Substitution but the
    # timing is trustworthy and confidence stays elevated.
    segs = _whisper([("there", 1.0, 1.3, 0.9)])
    out = align_lyrics(segs, "their")
    aligned = out["aligned_words"]
    assert aligned[0]["source"] == "whisper_substitution"
    # Phonetic-hit confidence is the higher of the two factors (0.7
    # rather than 0.4) — so 0.9 * 0.7 = 0.63.
    assert aligned[0]["confidence"] > 0.5


def test_align_interpolates_genius_word_missed_by_whisper():
    # Whisper missed "the" in the middle of a sequence.
    segs = _whisper([
        ("hold", 0.0, 0.3, 0.95),
        ("line", 1.0, 1.3, 0.95),
    ])
    out = align_lyrics(segs, "hold the line")
    aligned = out["aligned_words"]
    assert len(aligned) == 3
    assert [a["word"] for a in aligned] == ["hold", "the", "line"]
    assert aligned[1]["source"] == "interpolated"
    # Should sit between the two anchored timestamps.
    assert 0.3 <= aligned[1]["start"] <= 1.0
    assert aligned[1]["confidence"] < aligned[0]["confidence"]


def test_align_drops_whisper_hallucination():
    # Whisper produced a phantom "thank" that isn't in Genius.
    segs = _whisper([
        ("hello", 0.0, 0.3, 0.95),
        ("thank", 0.5, 0.7, 0.4),  # hallucination
        ("world", 1.0, 1.3, 0.95),
    ])
    out = align_lyrics(segs, "hello world")
    aligned = out["aligned_words"]
    assert [a["word"] for a in aligned] == ["hello", "world"]
    # Both should be matches — the hallucination was dropped.
    assert all(a["source"] == "whisper_match" for a in aligned)


def test_align_strips_section_headers_before_alignment():
    segs = _whisper([
        ("love", 0.0, 0.3, 0.95),
        ("me", 0.4, 0.6, 0.95),
    ])
    out = align_lyrics(segs, "[Chorus]\nLove me")
    aligned = out["aligned_words"]
    # The "[Chorus]" header should not appear as a word.
    assert all("[" not in a["word"] and "]" not in a["word"] for a in aligned)
    assert len(aligned) == 2


def test_align_expands_repeated_chorus_marker():
    # Whisper transcribed the chorus 3 times; Genius only lists it once
    # with "(×3)".
    segs = _whisper([
        ("Hey", 0.0, 0.5, 0.95),
        ("now", 0.5, 1.0, 0.95),
        ("Hey", 2.0, 2.5, 0.95),
        ("now", 2.5, 3.0, 0.95),
        ("Hey", 4.0, 4.5, 0.95),
        ("now", 4.5, 5.0, 0.95),
    ])
    out = align_lyrics(segs, "Hey now\n(×3)")
    aligned = out["aligned_words"]
    # 6 Genius words after expansion, all anchored to the 6 Whisper words.
    assert len(aligned) == 6
    assert sum(1 for a in aligned if a["source"] == "whisper_match") == 6
    # Timestamps span the full 5 seconds.
    assert aligned[0]["start"] == 0.0
    assert aligned[-1]["end"] == 5.0


def test_align_empty_genius_returns_error():
    segs = _whisper([("love", 0.0, 0.3, 0.95)])
    out = align_lyrics(segs, "")
    assert out["alignment_status"] == LyricsAlignmentStatus.error
    assert out["aligned_words"] == []


def test_align_whitespace_only_genius_returns_error():
    out = align_lyrics([], "   \n  \n")
    assert out["alignment_status"] == LyricsAlignmentStatus.error


def test_align_no_whisper_words_marks_low_quality():
    # Genius text but no Whisper output: every Genius word goes in
    # unanchored. Status is low_quality so the LLM falls back.
    out = align_lyrics([], "Love me tender")
    assert out["alignment_status"] == LyricsAlignmentStatus.low_quality
    assert len(out["aligned_words"]) == 3
    assert all(a["start"] is None for a in out["aligned_words"])
    assert out["alignment_quality"] == 0.0


def test_align_low_quality_when_mostly_misaligned():
    # Genius truth and Whisper output share nothing. Quality should
    # fall below QUALITY_LOW_THRESHOLD.
    segs = _whisper([
        ("zzz", 0.0, 0.3, 0.9),
        ("qqq", 0.3, 0.6, 0.9),
        ("xxx", 0.6, 0.9, 0.9),
        ("yyy", 0.9, 1.2, 0.9),
    ])
    out = align_lyrics(segs, "love me tender forever")
    assert out["alignment_status"] == LyricsAlignmentStatus.low_quality
    assert out["alignment_quality"] < 0.3


def test_align_quality_aggregate_counts_subs_at_half_weight():
    # Two exact matches + two phonetic substitutions:
    # quality = (2 + 0.5*2) / 4 = 0.75
    segs = _whisper([
        ("love", 0.0, 0.3, 0.95),
        ("me", 0.4, 0.6, 0.95),
        ("there", 0.7, 0.9, 0.95),  # phonetic sub for "their"
        ("forever", 1.0, 1.5, 0.95),
    ])
    out = align_lyrics(segs, "love me their forever")
    # love + me + forever match; their/there is a substitution.
    assert out["alignment_quality"] == 0.875  # (3 + 0.5*1) / 4
    assert out["alignment_status"] == LyricsAlignmentStatus.success


def test_align_preserves_genius_word_casing_and_punctuation():
    # The output should keep Genius's display form ("Love") even when
    # Whisper produced lowercase.
    segs = _whisper([("love", 0.0, 0.3, 0.95)])
    out = align_lyrics(segs, "Love!")
    assert out["aligned_words"][0]["word"] == "Love!"
