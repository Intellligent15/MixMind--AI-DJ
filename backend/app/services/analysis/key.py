"""Key detection via the Krumhansl-Schmuckler algorithm.

Given a 12-bin chroma vector summed/averaged over a song, correlate against
the 24 standard major/minor key profiles and return the best match plus its
Camelot wheel code (used for harmonic mixing decisions in later phases).

Profiles from Krumhansl & Kessler (1982).

The naive K-S algorithm is known to confuse relative-mode pairs (C major
↔ A minor share all 7 notes — the only difference is which one feels like
home). When the top two correlation scores are within
MODE_FLIP_SCORE_THRESHOLD of each other, we use the chroma intensity at
the two candidate tonics as a tiebreaker. This catches the common case
where the K-S correlation barely distinguishes the relative pair.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

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

# When K-S correlation difference between the top candidate and its
# relative-pair counterpart is below this threshold, we treat them as
# indistinguishable on profile correlation alone and fall back to tonic
# chroma comparison. 0.05 is conservative — only triggers on genuinely
# ambiguous keys.
MODE_FLIP_SCORE_THRESHOLD = 0.05


def _relative_pair(pitch: int, is_major: bool) -> tuple[int, bool]:
    """Return (pitch, is_major) of the relative-mode counterpart.

    Relative minor of major M is at root (M_root + 9) % 12 (e.g. C major
    -> A minor). Relative major of minor m is at root (m_root + 3) % 12
    (e.g. A minor -> C major).
    """
    if is_major:
        return (pitch + 9) % 12, False
    return (pitch + 3) % 12, True


def detect_key(chroma_vector: np.ndarray) -> tuple[str, str]:
    """Return (key_name, camelot_code) for a 12-bin chroma vector.

    Key names are like "C", "F#m". Camelot codes are like "8B", "5A".
    """
    if chroma_vector.shape != (12,):
        raise ValueError(f"chroma must be shape (12,), got {chroma_vector.shape}")

    # Score all 24 keys (12 pitch classes × 2 modes).
    scores: dict[tuple[int, bool], float] = {}
    for shift in range(12):
        rotated = np.roll(chroma_vector, -shift)
        scores[(shift, True)] = float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1])
        scores[(shift, False)] = float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1])

    (best_pitch, best_is_major), best_score = max(
        scores.items(), key=lambda kv: kv[1]
    )

    # Mode-flip tiebreaker for relative pairs. K-S can't reliably tell
    # C major from A minor when both correlations come out nearly equal —
    # in that case, the louder tonic in the actual chroma wins.
    rel_pitch, rel_is_major = _relative_pair(best_pitch, best_is_major)
    rel_score = scores[(rel_pitch, rel_is_major)]
    if best_score - rel_score < MODE_FLIP_SCORE_THRESHOLD:
        if chroma_vector[rel_pitch] > chroma_vector[best_pitch]:
            logger.info(
                "key mode-flipped from %s%s to %s%s "
                "(score diff %.3f < %.3f, tonic emphasis %.3f > %.3f)",
                PITCH_CLASSES[best_pitch],
                "" if best_is_major else "m",
                PITCH_CLASSES[rel_pitch],
                "" if rel_is_major else "m",
                best_score - rel_score,
                MODE_FLIP_SCORE_THRESHOLD,
                chroma_vector[rel_pitch],
                chroma_vector[best_pitch],
            )
            best_pitch = rel_pitch
            best_is_major = rel_is_major

    pitch_name = PITCH_CLASSES[best_pitch]
    if best_is_major:
        return pitch_name, _CAMELOT_MAJOR[pitch_name]
    return f"{pitch_name}m", _CAMELOT_MINOR[pitch_name]
