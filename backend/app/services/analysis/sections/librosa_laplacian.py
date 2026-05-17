"""Section detection via librosa's Laplacian segmentation example.

Follows McFee & Ellis (2014): build a beat-synchronous CQT recurrence matrix,
add a path enhancement, run spectral clustering on its Laplacian, then segment
on cluster boundaries. Labels are `section_N` cluster IDs - same N means
"structurally similar segments" (e.g. all choruses share an ID).

`k` (number of clusters) is chosen by the largest eigengap inside a bounded
range so single-section detection or pathological over-segmentation are
avoided. Returns sections sorted by start time with no gaps.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans

from app.services.analysis.sections.base import Section, SectionDetector

_DEFAULT_SR = 22050
_K_MIN = 4
_K_MAX = 10


class LibrosaLaplacianDetector(SectionDetector):
    def __init__(self, k_min: int = _K_MIN, k_max: int = _K_MAX) -> None:
        self.k_min = k_min
        self.k_max = k_max

    def detect_file(self, path: Path) -> list[Section]:
        y, sr = librosa.load(str(path), sr=_DEFAULT_SR, mono=True)
        return self.detect(y, sr)

    def detect(self, audio: np.ndarray, sr: int) -> list[Section]:
        duration = len(audio) / sr
        if duration < 4.0:
            return [Section(start=0.0, end=float(duration), label="section_1")]

        # Beat-synchronous CQT features for repetition detection.
        bpm, beats = librosa.beat.beat_track(y=audio, sr=sr, trim=False)
        if len(beats) < self.k_min * 2:
            return [Section(start=0.0, end=float(duration), label="section_1")]

        cqt = np.abs(
            librosa.cqt(y=audio, sr=sr, bins_per_octave=12 * 3, n_bins=7 * 12 * 3)
        )
        cqt_sync = librosa.util.sync(cqt, beats, aggregate=np.median)
        # Log-amplitude smoothing — emphasises pitch class over spectral magnitude.
        cqt_sync = librosa.amplitude_to_db(cqt_sync, ref=np.max)

        # Recurrence: which beats look like which other beats.
        rec = librosa.segment.recurrence_matrix(
            cqt_sync, width=3, mode="affinity", sym=True
        )
        # Path enhancement: emphasise temporal proximity.
        path = librosa.segment.path_enhance(rec, n=9)
        graph = np.maximum(rec, path)

        # Spectral clustering on the symmetric normalised Laplacian.
        degree = graph.sum(axis=1)
        if np.any(degree == 0):
            return [Section(start=0.0, end=float(duration), label="section_1")]
        d_inv_sqrt = 1.0 / np.sqrt(degree)
        laplacian = scipy.sparse.eye(graph.shape[0]) - (
            (graph * d_inv_sqrt).T * d_inv_sqrt
        ).T

        eigvals, eigvecs = scipy.sparse.linalg.eigsh(
            laplacian, k=self.k_max + 1, which="SM"
        )
        # Sort ascending (eigsh with which="SM" returns ascending already, but
        # be defensive — the order matters for the eigengap pick).
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        k = _choose_k(eigvals, self.k_min, self.k_max)
        embedding = eigvecs[:, :k]
        # Normalise rows (Ng-Jordan-Weiss): treat embedding rows as unit vectors.
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding = embedding / norms

        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embedding)

        # Convert beat-frame labels into time-domain segments.
        beat_times = librosa.frames_to_time(beats, sr=sr)
        # Each label[i] corresponds to the beat span [beat_times[i], beat_times[i+1]).
        # The last beat extends to the end of the track.
        boundaries: list[tuple[float, float, int]] = []
        for i, lab in enumerate(labels):
            start = float(beat_times[i]) if i < len(beat_times) else float(duration)
            end = float(beat_times[i + 1]) if i + 1 < len(beat_times) else float(duration)
            boundaries.append((start, end, int(lab)))

        # Coalesce contiguous beats with the same label into single sections.
        sections: list[Section] = []
        cur_start, cur_end, cur_lab = boundaries[0]
        for start, end, lab in boundaries[1:]:
            if lab == cur_lab:
                cur_end = end
            else:
                sections.append(
                    Section(start=cur_start, end=cur_end, label=f"section_{cur_lab + 1}")
                )
                cur_start, cur_end, cur_lab = start, end, lab
        sections.append(
            Section(start=cur_start, end=cur_end, label=f"section_{cur_lab + 1}")
        )

        # Snap first section to t=0 and last section to the track duration so
        # there are no implicit gaps in the timeline. Then drop any
        # zero/negative-duration tail sections caused by librosa's beat-frame
        # times slightly exceeding the audio duration. Repeat the snap on
        # each removal so the last surviving section still ends at duration.
        if not sections:
            return [Section(start=0.0, end=float(duration), label="section_1")]
        sections[0] = Section(start=0.0, end=sections[0].end, label=sections[0].label)
        sections[-1] = Section(
            start=sections[-1].start, end=float(duration), label=sections[-1].label
        )
        while len(sections) > 1 and sections[-1].end <= sections[-1].start:
            sections.pop()
            sections[-1] = Section(
                start=sections[-1].start,
                end=float(duration),
                label=sections[-1].label,
            )
        return sections


def _choose_k(eigvals: np.ndarray, k_min: int, k_max: int) -> int:
    """Pick k by the largest eigengap in [k_min, k_max]."""
    if len(eigvals) <= k_min:
        return max(1, len(eigvals) - 1)
    gaps = np.diff(eigvals)
    # gaps[i] = eigvals[i+1] - eigvals[i]. We want k such that gap at position k-1
    # is largest within the allowed range.
    lo = max(k_min - 1, 0)
    hi = min(k_max - 1, len(gaps) - 1)
    if hi < lo:
        return k_min
    k = int(np.argmax(gaps[lo : hi + 1])) + lo + 1
    return max(k_min, min(k, k_max))
