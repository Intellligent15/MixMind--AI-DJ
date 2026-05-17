import os
from functools import lru_cache

from app.services.analysis.sections.base import SectionDetector
from app.services.analysis.sections.librosa_laplacian import LibrosaLaplacianDetector


@lru_cache(maxsize=1)
def get_section_detector() -> SectionDetector:
    backend = os.getenv("SECTION_DETECTOR", "librosa")
    if backend == "librosa":
        return LibrosaLaplacianDetector()
    raise ValueError(f"unknown SECTION_DETECTOR: {backend!r}")
