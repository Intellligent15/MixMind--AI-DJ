"""LibrosaLaplacianDetector smoke tests against synthesised audio.

We synthesise two contrasting "sections" back-to-back (different chord +
timbre) so a structural segmenter should put a boundary somewhere in the
middle. We don't assert exact boundaries — librosa Laplacian on synthetic
audio is approximate — only shape invariants and that the timeline is
contiguous from 0 to duration.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.analysis.sections.base import Section
from app.services.analysis.sections.factory import get_section_detector
from app.services.analysis.sections.librosa_laplacian import (
    LibrosaLaplacianDetector,
    _merge_short_sections,
)

SR = 22050


def _tone_section(frequencies: list[float], seconds: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = np.zeros_like(t)
    for f in frequencies:
        sig += np.sin(2 * np.pi * f * t)
    return sig / max(1, len(frequencies))


def _click_track(bpm: float, seconds: float, sr: int = SR) -> np.ndarray:
    beats = int(seconds * bpm / 60.0)
    audio = np.zeros(int(seconds * sr))
    period = int(sr * 60.0 / bpm)
    for i in range(beats):
        idx = i * period
        if idx + 200 < len(audio):
            audio[idx : idx + 200] += np.linspace(1.0, 0.0, 200)
    return audio


def _synth_two_section_song(seconds_per: float = 10.0) -> np.ndarray:
    # Section A: C major triad (C-E-G), Section B: F minor triad (F-Ab-C)
    a = _tone_section([261.63, 329.63, 392.00], seconds_per)
    b = _tone_section([349.23, 415.30, 261.63], seconds_per)
    click = _click_track(120.0, seconds_per * 2)
    melodic = np.concatenate([a, b])
    n = min(len(melodic), len(click))
    return 0.7 * melodic[:n] + 0.3 * click[:n]


def test_short_clip_returns_single_section():
    audio = _tone_section([440.0], 1.0)
    detector = LibrosaLaplacianDetector()
    out = detector.detect(audio, SR)
    assert len(out) == 1
    assert out[0].start == 0.0
    assert out[0].end == pytest.approx(1.0, abs=0.05)


def test_two_section_song_yields_contiguous_sections():
    audio = _synth_two_section_song(seconds_per=12.0)
    detector = LibrosaLaplacianDetector()
    out = detector.detect(audio, SR)

    assert len(out) >= 1
    assert isinstance(out[0], Section)
    assert out[0].start == 0.0
    assert out[-1].end == pytest.approx(len(audio) / SR, abs=0.5)
    # Contiguity: every section's end equals the next section's start.
    for prev, nxt in zip(out, out[1:]):
        assert nxt.start == prev.end
        assert prev.end > prev.start
    # Labels follow the section_N convention.
    for s in out:
        assert s.label.startswith("section_")


def test_factory_returns_librosa_by_default():
    get_section_detector.cache_clear()
    det = get_section_detector()
    assert isinstance(det, LibrosaLaplacianDetector)
    get_section_detector.cache_clear()


def test_section_to_dict_is_json_friendly():
    s = Section(start=1.0, end=2.5, label="section_3")
    d = s.to_dict()
    assert d == {"start": 1.0, "end": 2.5, "label": "section_3"}


def test_merge_short_absorbs_into_longer_neighbour():
    # Short middle section flanked by a long left and short-but-longer right;
    # the left is longer so the short one should merge left.
    sections = [
        Section(0.0, 20.0, "section_1"),
        Section(20.0, 23.0, "section_2"),  # 3s — under the 8s floor
        Section(23.0, 35.0, "section_3"),
    ]
    out = _merge_short_sections(sections, min_seconds=8.0)
    assert len(out) == 2
    assert out[0] == Section(0.0, 23.0, "section_1")  # absorbed left
    assert out[1] == Section(23.0, 35.0, "section_3")


def test_merge_short_picks_longer_of_two_neighbours():
    sections = [
        Section(0.0, 10.0, "A"),
        Section(10.0, 12.0, "B"),
        Section(12.0, 40.0, "C"),  # longer than the left neighbour
    ]
    out = _merge_short_sections(sections, min_seconds=8.0)
    assert out == [
        Section(0.0, 10.0, "A"),
        Section(10.0, 40.0, "C"),
    ]


def test_merge_short_handles_edges():
    # Short leading section has only a right neighbour.
    sections = [
        Section(0.0, 3.0, "intro"),
        Section(3.0, 30.0, "main"),
    ]
    out = _merge_short_sections(sections, min_seconds=8.0)
    assert out == [Section(0.0, 30.0, "main")]


def test_merge_short_cascades_until_clean():
    # Several short sections in a row should all dissolve.
    sections = [
        Section(0.0, 20.0, "A"),
        Section(20.0, 22.0, "B"),
        Section(22.0, 24.0, "C"),
        Section(24.0, 26.0, "D"),
        Section(26.0, 50.0, "E"),
    ]
    out = _merge_short_sections(sections, min_seconds=8.0)
    durations = [s.end - s.start for s in out]
    assert all(d >= 8.0 for d in durations)
    # No gaps, full timeline preserved.
    assert out[0].start == 0.0
    assert out[-1].end == 50.0
    for prev, nxt in zip(out, out[1:]):
        assert nxt.start == prev.end


def test_merge_short_never_empties_input():
    # If the whole track is shorter than the threshold, return the one section.
    sections = [Section(0.0, 5.0, "tiny")]
    out = _merge_short_sections(sections, min_seconds=8.0)
    assert out == sections


def test_merge_short_coalesces_adjacent_same_label_after_merge():
    # [A, short, A] — merging the short into either neighbour creates
    # adjacent same-label sections which should collapse into one.
    sections = [
        Section(0.0, 20.0, "A"),
        Section(20.0, 22.0, "B"),
        Section(22.0, 50.0, "A"),
    ]
    out = _merge_short_sections(sections, min_seconds=8.0)
    assert out == [Section(0.0, 50.0, "A")]
