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
