from __future__ import annotations

import numpy as np
import pytest

from app.services.analysis.key import PITCH_CLASSES, detect_key


def _profile(tonic_pitch_class: int, kind: str) -> np.ndarray:
    """Build a synthetic chroma vector for a given tonic + mode by rotating
    the Krumhansl-Kessler profile. This is the test's reference signal."""
    if kind == "major":
        profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
    else:
        profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )
    return np.roll(profile, tonic_pitch_class)


@pytest.mark.parametrize(
    "pitch,kind,expected_key,expected_camelot",
    [
        (0, "major", "C", "8B"),
        (7, "major", "G", "9B"),
        (9, "minor", "Am", "8A"),
        (5, "minor", "Fm", "4A"),
        (6, "major", "F#", "2B"),
        (11, "minor", "Bm", "10A"),
    ],
)
def test_detect_key_resolves_clean_profiles(
    pitch: int, kind: str, expected_key: str, expected_camelot: str
):
    chroma = _profile(pitch, kind)
    name, camelot = detect_key(chroma)
    assert name == expected_key
    assert camelot == expected_camelot


def test_detect_key_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        detect_key(np.zeros(10))


def test_pitch_classes_cover_octave():
    assert len(PITCH_CLASSES) == 12
    assert PITCH_CLASSES[0] == "C"
    assert PITCH_CLASSES[-1] == "B"


def test_mode_flip_kicks_in_when_relative_tonic_dominates():
    # Build a chroma that scores nearly equally well as C major and A minor
    # under K-S, but where the A bin is much louder than the C bin. The
    # mode-flip tiebreaker should pick A minor.
    # Start with an even blend of the two profiles, then boost A.
    chroma = (
        _profile(0, "major") + _profile(9, "minor")
    ) / 2
    chroma[9] *= 3.0  # boost A
    name, camelot = detect_key(chroma)
    assert name == "Am"
    assert camelot == "8A"


def test_mode_flip_does_not_trigger_on_clear_major():
    # Pure major profile — relative minor's correlation is much lower,
    # so the tiebreaker should NOT fire even if we artificially boost
    # the relative tonic.
    chroma = _profile(0, "major").copy()
    chroma[9] *= 1.5  # mild boost on A — not enough to override clear K-S signal
    name, _ = detect_key(chroma)
    assert name == "C"
