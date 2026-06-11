"""Optional All-In-One (allin1) neural analysis source.

The librosa stack (beat_track + octave correction, Laplacian sections)
is a solid v1, but it's now the quality ceiling: a mis-detected beat
grid or an opaque `section_3` label degrades every transition no matter
how good the planner is. The `allin1` package (Taejun Kim's
"All-In-One" music structure model) jointly predicts beats, downbeats,
tempo, AND *functionally labeled* segments — intro / verse / chorus /
bridge / outro — from one forward pass, and those labels are exactly
the language the seam-candidate generator and the LLM want.

This module is import-guarded and failure-tolerant: it returns None
whenever the package is missing or inference fails, and the analysis
service falls back to the librosa path field-by-field. Install with:

    uv pip install allin1   # plus its torch dependency; GPU recommended

and set `SECTION_DETECTOR=allin1` (or `section_detector` in settings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when allin1 is installed
    import allin1  # type: ignore

    ALLIN1_AVAILABLE = True
except Exception:  # ImportError or any of allin1's heavy deps failing
    allin1 = None  # type: ignore
    ALLIN1_AVAILABLE = False


@dataclass(frozen=True)
class Allin1Result:
    bpm: float
    beats: list[float]
    downbeats: list[float]
    sections: list[dict]  # {"start", "end", "label"} with REAL labels


def analyze_with_allin1(audio_path: Path) -> Allin1Result | None:
    """Run allin1 on `audio_path`. None on any failure (caller falls back)."""
    if not ALLIN1_AVAILABLE:
        return None
    try:
        result = allin1.analyze(str(audio_path))
        sections = [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "label": str(seg.label),
            }
            for seg in (result.segments or [])
        ]
        beats = [float(t) for t in (result.beats or [])]
        downbeats = [float(t) for t in (result.downbeats or [])]
        bpm = float(result.bpm)
        if not beats or not downbeats or bpm <= 0:
            logger.warning("allin1: degenerate output for %s; falling back", audio_path)
            return None
        return Allin1Result(
            bpm=bpm, beats=beats, downbeats=downbeats, sections=sections
        )
    except Exception as exc:  # noqa: BLE001 — any model failure → fallback
        logger.warning("allin1: analysis failed for %s (%s); falling back",
                       audio_path, exc)
        return None
