"""Pure analysis service.

Takes an audio path and returns a dataclass of analysis primitives. No DB
writes, no storage I/O beyond `librosa.load`. The Celery task is responsible
for persistence.

Steps (per spec, Spotify path excluded):
  1. Load audio mono @ 22050 Hz
  2. librosa beat tracking -> BPM + beat times
  3. Downbeats: every Nth beat where N = time signature (fixed 4)
  4. CQT chroma -> Krumhansl-Schmuckler key detection -> Camelot
  5. RMS resampled to 1 Hz -> energy curve
  6. Sections via the injected SectionDetector
  7. vocal_segments left empty (Phase 6 populates from Whisper)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from app.services.analysis.key import detect_key
from app.services.analysis.sections.base import Section, SectionDetector
from app.services.analysis.sections.factory import get_section_detector

ANALYSIS_SR = 22050
TIME_SIGNATURE = 4  # see deviation note in the notes


@dataclass(frozen=True)
class AnalysisResult:
    bpm: float
    key: str
    camelot_key: str
    time_signature: int
    beat_grid: list[float]
    downbeats: list[float]
    sections: list[Section]
    energy_curve: list[float]
    vocal_segments: list[tuple[float, float]] = field(default_factory=list)


class AnalysisService:
    def __init__(self, section_detector: SectionDetector | None = None) -> None:
        self.section_detector = section_detector or get_section_detector()

    def analyze(self, audio_path: Path) -> AnalysisResult:
        y, sr = librosa.load(str(audio_path), sr=ANALYSIS_SR, mono=True)

        bpm_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        bpm = float(np.atleast_1d(bpm_arr)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        downbeats = beat_times[::TIME_SIGNATURE]

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_name, camelot = detect_key(chroma_mean)

        energy_curve = _rms_at_1hz(y, sr)

        sections = self.section_detector.detect(y, sr)

        return AnalysisResult(
            bpm=bpm,
            key=key_name,
            camelot_key=camelot,
            time_signature=TIME_SIGNATURE,
            beat_grid=beat_times,
            downbeats=downbeats,
            sections=sections,
            energy_curve=energy_curve,
            vocal_segments=[],
        )


def _rms_at_1hz(y: np.ndarray, sr: int) -> list[float]:
    """Mean RMS energy per 1-second window over the track."""
    duration = len(y) / sr
    n_seconds = max(1, int(np.ceil(duration)))
    window = sr
    out: list[float] = []
    for i in range(n_seconds):
        start = i * window
        end = min(start + window, len(y))
        if start >= len(y):
            out.append(0.0)
            continue
        chunk = y[start:end]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
        out.append(rms)
    return out
