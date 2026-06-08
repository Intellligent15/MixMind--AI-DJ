"""Pure analysis service.

Takes an audio path and returns a dataclass of analysis primitives. No DB
writes, no storage I/O beyond `librosa.load`. The Celery task is responsible
for persistence.

Steps (per spec, Spotify path excluded):
  1. Load audio mono @ 22050 Hz
  2. librosa beat tracking -> BPM + beat times
  2a. Tempo octave/multiple correction (see _correct_tempo_octave)
  3. Downbeats: pick the phase offset (0..time_signature-1) whose beats
     align with the strongest onsets (see _pick_downbeat_phase)
  4. CQT chroma -> Krumhansl-Schmuckler key detection -> Camelot
  5. RMS resampled to 1 Hz -> energy curve
  6. Sections via the injected SectionDetector
  7. vocal_segments left empty (Phase 6 populates from Whisper)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from app.services.analysis.key import detect_key
from app.services.analysis.sections.base import Section, SectionDetector
from app.services.analysis.sections.factory import get_section_detector

logger = logging.getLogger(__name__)

ANALYSIS_SR = 22050
TIME_SIGNATURE = 4  # assumed 4/4 (the common case for the target material)

# Tempo octave-correction parameters. librosa.beat.beat_track uses a Gaussian
# tempo prior centered on `start_bpm` (default 120) — it can pick half-time,
# 3:2 multiples, or other "musically wrong" tempos when the song has strong
# off-beat emphasis. When the initial BPM lands outside this preferred range,
# we re-run beat_track with alternate hints and pick the result whose beats
# coincide most strongly with the onset envelope.
# Upper bound (Hz) of the "low band" used to phase-align downbeats. The kick
# fundamental and the bass note live here; snare (~200 Hz + broadband) and hats
# are mostly excluded. Picking the downbeat phase off a full-spectrum onset
# envelope gets fooled by loud hats/snares on the backbeat — beat 1 is carried
# by the kick + bass, so a low-band onset envelope is a cleaner "where's the
# bar?" signal.
DOWNBEAT_LOW_BAND_HZ = 250.0

TEMPO_PREFERRED_MIN = 85.0
TEMPO_PREFERRED_MAX = 170.0
# Multiplier candidates to try as start_bpm hints. Covers the common
# octave (×2, ÷2), 3:2 (×1.5, ÷1.5), and triplet (×3, ÷3) errors.
TEMPO_CORRECTION_MULTIPLIERS = (0.5, 2.0 / 3.0, 1.5, 2.0)
# Penalty applied to a candidate's score when it falls outside the preferred
# range. Lower = stronger preference for in-range candidates. 0.6 means an
# out-of-range candidate has to have ≥1.67× the onset strength of an
# in-range candidate to win.
OUT_OF_RANGE_PENALTY = 0.6


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


def _correct_tempo_octave(
    y: np.ndarray,
    sr: int,
    bpm0: float,
    beat_frames0: np.ndarray,
    onset_env: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Try to correct librosa octave/multiple errors.

    Librosa's beat tracker can pick half-time, ×1.5, or ÷1.5 multiples of the
    true tempo on songs with strong syncopation, half-time grooves, or
    dotted-quarter emphasis (e.g. the "Massive" case where 125 BPM was
    detected as 80 BPM — a 1:1.5625 ratio that aligned with the song's
    dotted-quarter emphasis pattern).

    Strategy: if the detected BPM is outside the musically-preferred range
    [TEMPO_PREFERRED_MIN, TEMPO_PREFERRED_MAX], re-run beat_track with
    alternative tempo hints (×0.5, ×2/3, ×1.5, ×2). Score each candidate by
    mean onset strength at the detected beat frames, with a penalty for
    out-of-range BPMs. Return the highest-scoring candidate.

    Returns the original (bpm0, beat_frames0) when in range or when no
    candidate scores higher — never makes things worse than librosa's
    default output.
    """
    if TEMPO_PREFERRED_MIN <= bpm0 <= TEMPO_PREFERRED_MAX:
        return bpm0, beat_frames0

    def _strength(frames: np.ndarray) -> float:
        valid = frames[frames < len(onset_env)]
        return float(onset_env[valid].mean()) if len(valid) else 0.0

    def _score(bpm: float, strength: float) -> float:
        in_range = TEMPO_PREFERRED_MIN <= bpm <= TEMPO_PREFERRED_MAX
        return strength * (1.0 if in_range else OUT_OF_RANGE_PENALTY)

    candidates: list[tuple[float, np.ndarray]] = [(bpm0, beat_frames0)]
    best_score = _score(bpm0, _strength(beat_frames0))

    for multiplier in TEMPO_CORRECTION_MULTIPLIERS:
        hint = bpm0 * multiplier
        # Don't bother with hints that are wildly out of range — they'd
        # converge to the same answer anyway.
        if not (50.0 <= hint <= 220.0):
            continue
        try:
            _, cand_frames = librosa.beat.beat_track(
                y=y, sr=sr, onset_envelope=onset_env, start_bpm=hint, trim=False
            )
        except Exception:  # noqa: BLE001 — librosa can throw on edge inputs
            continue
        if len(cand_frames) < 2:
            continue
        beat_times = librosa.frames_to_time(cand_frames, sr=sr)
        intervals = np.diff(beat_times)
        if intervals.min() <= 0:
            continue
        cand_bpm = 60.0 / float(np.median(intervals))
        cand_score = _score(cand_bpm, _strength(cand_frames))
        candidates.append((cand_bpm, cand_frames))
        if cand_score > best_score:
            best_score = cand_score

    # Find the candidate matching the best score (in case of ties, the
    # original wins since it appears first).
    for bpm, frames in candidates:
        if _score(bpm, _strength(frames)) == best_score:
            if bpm != bpm0:
                logger.info(
                    "tempo octave-corrected from %.2f BPM to %.2f BPM "
                    "(in-range bonus %.2f)",
                    bpm0,
                    bpm,
                    1.0 / OUT_OF_RANGE_PENALTY,
                )
            return bpm, frames
    return bpm0, beat_frames0


def _low_band_onset_env(y: np.ndarray, sr: int) -> np.ndarray:
    """Onset-strength envelope built from only the low mel bands (kick + bass).

    Used for downbeat phase-picking. A full-spectrum onset envelope weights
    hats/snares (often loudest on the offbeat / backbeat) as heavily as the
    kick, so `_pick_downbeat_phase` can lock onto beat 2/4. Restricting the
    onset flux to <= DOWNBEAT_LOW_BAND_HZ makes beat 1 (kick fundamental + bass
    note) stand out. Uses the default hop (512), matching `beat_track`'s frames
    so the offset search indexes consistently.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=sr / 2.0)
    low = freqs <= DOWNBEAT_LOW_BAND_HZ
    if not low.any():
        low[0] = True
    return librosa.onset.onset_strength(
        S=librosa.power_to_db(mel[low], ref=np.max), sr=sr
    )


def _pick_downbeat_phase(
    beat_frames: np.ndarray,
    onset_env: np.ndarray,
    time_signature: int,
) -> int:
    """Choose which beat-grid offset (0..time_signature-1) carries the
    downbeats by aggregating onset strength at each candidate phase.

    Librosa's beat tracker doesn't model bar structure — its "beat 0" can
    land on any of the 4 beats of an actual 4/4 bar. Without correction,
    `beat_times[::4]` picks every 4th beat starting from a phase that's
    only accidentally correct ~25% of the time, which is why the rendered
    crossfade hears downbeats land off-grid on most material.

    Onset strength tends to be higher on downbeats (kick drum on 1, snare
    on 2 & 4 — kick usually punchier). Summing onset_env at each candidate
    phase and picking the max is a simple but surprisingly effective proxy
    for "where's beat 1?".
    """
    best_offset = 0
    best_strength = -np.inf
    for offset in range(time_signature):
        candidate_frames = beat_frames[offset::time_signature]
        valid = candidate_frames[candidate_frames < len(onset_env)]
        if len(valid) == 0:
            continue
        strength = float(onset_env[valid].mean())
        if strength > best_strength:
            best_strength = strength
            best_offset = offset
    if best_offset != 0:
        logger.info(
            "downbeat phase-aligned: offset=%d (was assuming 0)", best_offset
        )
    return best_offset


class AnalysisService:
    def __init__(self, section_detector: SectionDetector | None = None) -> None:
        self.section_detector = section_detector or get_section_detector()

    def analyze(self, audio_path: Path) -> AnalysisResult:
        y, sr = librosa.load(str(audio_path), sr=ANALYSIS_SR, mono=True)

        # Compute the onset envelope once and reuse it for beat_track,
        # tempo octave correction, and downbeat phase search.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        bpm_arr, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, onset_envelope=onset_env, trim=False
        )
        bpm_raw = float(np.atleast_1d(bpm_arr)[0])

        bpm, beat_frames = _correct_tempo_octave(
            y, sr, bpm_raw, beat_frames, onset_env
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Phase-pick downbeats off a LOW-BAND onset envelope (kick + bass), not
        # the full-spectrum one — beat 1 is carried by the kick/bass, so this
        # is far less likely to lock onto a loud backbeat snare/hat.
        phase_env = _low_band_onset_env(y, sr)
        downbeat_offset = _pick_downbeat_phase(
            beat_frames, phase_env, TIME_SIGNATURE
        )
        downbeats = beat_times[downbeat_offset::TIME_SIGNATURE]

        # Key detection: three layered improvements over bare chroma_cqt.
        #   1. HPSS (harmonic-percussive source separation) strips drum
        #      content from the signal. Percussion adds broadband chroma
        #      noise that dilutes the actual tonal information.
        #   2. chroma_cens (Chroma Energy Normalized Statistics) is a
        #      quantized + smoothed chroma designed for global descriptors
        #      like key — more robust than chroma_cqt's per-frame output.
        #   3. Energy-weighted averaging: per-frame RMS as weight so loud
        #      sections (where harmonic content is clearest) dominate the
        #      estimate. Quiet intros / outros stop diluting the average.
        y_harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
        rms = librosa.feature.rms(y=y_harmonic)[0]
        n = min(rms.shape[0], chroma.shape[1])
        weights = rms[:n] / (rms[:n].sum() + 1e-9)
        chroma_weighted = (chroma[:, :n] * weights[None, :]).sum(axis=1)
        key_name, camelot = detect_key(chroma_weighted)

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
