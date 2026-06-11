import os
from functools import lru_cache

from app.services.analysis.sections.base import SectionDetector
from app.services.analysis.sections.librosa_laplacian import LibrosaLaplacianDetector


@lru_cache(maxsize=1)
def get_section_detector() -> SectionDetector:
    """Pick the section backend.

    "librosa" / "librosa_laplacian" — the default Laplacian segmentation
    (opaque section_N cluster labels).

    "allin1" — sections come from the All-In-One neural model with REAL
    functional labels (intro/verse/chorus/...). The allin1 path is wired
    at the AnalysisService level (it also supplies beats/downbeats/bpm),
    so here it just means "keep librosa as the in-process fallback
    detector"; the service consults allin1 first when configured.
    """
    backend = os.getenv("SECTION_DETECTOR", "librosa")
    if backend in ("librosa", "librosa_laplacian", "allin1"):
        return LibrosaLaplacianDetector()
    raise ValueError(f"unknown SECTION_DETECTOR: {backend!r}")
