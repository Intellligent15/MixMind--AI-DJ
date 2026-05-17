"""Section detection protocol.

A section is a `{start, end, label}` JSON-serializable record. `label` is an
opaque string. For the librosa Laplacian detector, labels are "section_N"
cluster IDs where the same N across the song means "structurally similar."
A future detector (e.g. SongFormer) may use semantic labels like "chorus"
or "intro". Callers must not parse labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Section:
    start: float
    end: float
    label: str

    def to_dict(self) -> dict[str, float | str]:
        return {"start": self.start, "end": self.end, "label": self.label}


class SectionDetector(Protocol):
    def detect(self, audio: np.ndarray, sr: int) -> list[Section]:
        """Segment a mono audio buffer into contiguous sections."""
        ...

    def detect_file(self, path: Path) -> list[Section]:
        """Convenience: load `path` and segment it."""
        ...
