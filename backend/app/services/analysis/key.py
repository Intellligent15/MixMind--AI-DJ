"""Key detection via the Krumhansl-Schmuckler algorithm.

Given a 12-bin chroma vector summed/averaged over a song, correlate against
the 24 standard major/minor key profiles and return the best match plus its
Camelot wheel code (used for harmonic mixing decisions in later phases).

Profiles from Krumhansl & Kessler (1982).
"""

from __future__ import annotations

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Camelot wheel: major keys are "B" lane, minor keys are "A" lane.
# Index is the pitch class (0=C, 1=C#, ..., 11=B).
_CAMELOT_MAJOR = {
    "C": "8B", "C#": "3B", "D": "10B", "D#": "5B", "E": "12B", "F": "7B",
    "F#": "2B", "G": "9B", "G#": "4B", "A": "11B", "A#": "6B", "B": "1B",
}
_CAMELOT_MINOR = {
    "C": "5A", "C#": "12A", "D": "7A", "D#": "2A", "E": "9A", "F": "4A",
    "F#": "11A", "G": "6A", "G#": "1A", "A": "8A", "A#": "3A", "B": "10A",
}


def detect_key(chroma_vector: np.ndarray) -> tuple[str, str]:
    """Return (key_name, camelot_code) for a 12-bin chroma vector.

    Key names are like "C", "F#m". Camelot codes are like "8B", "5A".
    Ties between major/minor (rare) resolve to major.
    """
    if chroma_vector.shape != (12,):
        raise ValueError(f"chroma must be shape (12,), got {chroma_vector.shape}")

    best_score = -np.inf
    best_pitch = 0
    best_is_major = True

    for shift in range(12):
        rotated = np.roll(chroma_vector, -shift)
        major_score = float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1])
        minor_score = float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1])
        if major_score >= best_score:
            best_score = major_score
            best_pitch = shift
            best_is_major = True
        if minor_score > best_score:
            best_score = minor_score
            best_pitch = shift
            best_is_major = False

    pitch_name = PITCH_CLASSES[best_pitch]
    if best_is_major:
        return pitch_name, _CAMELOT_MAJOR[pitch_name]
    return f"{pitch_name}m", _CAMELOT_MINOR[pitch_name]
