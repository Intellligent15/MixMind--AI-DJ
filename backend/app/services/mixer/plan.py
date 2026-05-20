"""Hand-built mix-plan generator — Phase 7's stand-in for the Phase 9 LLM.

Takes two AnalysisBundles and emits a list of tool-call dicts matching
the spec's Mix Plan Schema. Pure function; no I/O, no DB.

Strategy (full design: the design notes):
1. Seam in A = last section start, clamped to "no more than 16 bars
   before A.duration", snapped to nearest downbeat ≥ that point.
2. Seam in B = first downbeat ≥ end of B's first section (skip silent
   intros / count-ins).
3. duration_bars = min(16, available outro / bar, available stretched
   intro / bar), floored at 4 with a warning when shorter.

Phase 7 explicitly does NOT pitch-shift. pyrubberband artifacts at
shifts > 2 semitones can sound worse than the harmonic dissonance
they're trying to fix, and small shifts are subtle enough that most
listeners won't notice the mismatch. Phase 9's LLM has the
`pitch_shift` and `temporary_pitch_shift` tools available and can
decide per-pair whether to use them; the hand-built generator stays
neutral and leaves keys as-is.

`compute_pitch_shift` stays in the module as a public utility so
Phase 9 (and anything else that wants to reason about smallest-shift
targets) doesn't have to duplicate the key-parsing logic.
"""

from __future__ import annotations

import logging

from app.services.mixer.types import AnalysisBundle, MixPlanJSON

logger = logging.getLogger(__name__)

# Pitch classes use sharps (matches services/analysis/key.py).
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

DEFAULT_DURATION_BARS = 16
MIN_DURATION_BARS = 4
LARGE_SHIFT_THRESHOLD = 2  # |δ| > 2 → WARN (locked: apply anyway)


def _parse_key(key: str) -> tuple[int, bool]:
    """Return (pitch_class_index, is_minor). Accepts 'C', 'F#', 'Cm', 'F#m',
    plus flat aliases like 'Db' for forward compatibility (the analyzer
    only emits sharps, but the LLM in Phase 9 might emit flats)."""
    is_minor = key.endswith("m")
    root = key[:-1] if is_minor else key
    root = _FLAT_TO_SHARP.get(root, root)
    if root not in _PITCH_CLASSES:
        raise ValueError(f"unknown key: {key!r}")
    return _PITCH_CLASSES.index(root), is_minor


def compute_pitch_shift(a_key: str, b_key: str) -> int:
    """Smallest signed semitone shift to move B's tonic to A's tonic.

    Relative major/minor pairs (e.g. C major ↔ A minor, same Camelot
    position) are treated as compatible: δ=0.
    """
    a_root, a_minor = _parse_key(a_key)
    b_root, b_minor = _parse_key(b_key)

    # Smallest signed shift in [-6, 6].
    raw = a_root - b_root
    delta = ((raw + 6) % 12) - 6

    # Relative-major↔minor: same Camelot position, treat as compatible.
    # Relative minor of a major key is 3 semitones below (e.g. C major's
    # relative minor is Am, which is 9 above or -3 below). If mode flips
    # AND |delta| == 3, no shift needed.
    if a_minor != b_minor and abs(delta) == 3:
        return 0
    return delta


def _snap_to_downbeat(t: float, downbeats: list[float]) -> float:
    """First downbeat ≥ t, or the last downbeat if none qualifies."""
    if not downbeats:
        return t
    for d in downbeats:
        if d >= t:
            return d
    return downbeats[-1]


def build_pair_plan(a: AnalysisBundle, b: AnalysisBundle) -> MixPlanJSON:
    """Build the tool-call list for the A → B transition."""
    sec_per_bar_a = (60.0 / a.bpm) * a.time_signature
    max_outro_seconds = DEFAULT_DURATION_BARS * sec_per_bar_a

    # A seam: last section start, clamped to "no more than 16 bars before end".
    last_section_start = a.sections[-1]["start"] if a.sections else 0.0
    seam_a_raw = max(last_section_start, a.duration - max_outro_seconds)
    seam_a = _snap_to_downbeat(seam_a_raw, a.downbeats)

    # B seam: first downbeat ≥ end of first section.
    first_section_end = b.sections[0]["end"] if b.sections else 0.0
    seam_b = _snap_to_downbeat(first_section_end, b.downbeats)

    # Crossfade length: clamp to whatever's available in A's outro and
    # B's post-stretch intro. Stretch factor = b.bpm / a.bpm.
    stretch_factor = b.bpm / a.bpm
    available_a = a.duration - seam_a
    available_b_stretched = (b.duration - seam_b) * stretch_factor
    duration_bars = min(
        DEFAULT_DURATION_BARS,
        int(available_a / sec_per_bar_a),
        int(available_b_stretched / sec_per_bar_a),
    )
    if duration_bars < MIN_DURATION_BARS:
        logger.warning(
            "build_pair_plan: insufficient overlap (available_a=%.2fs, "
            "available_b_stretched=%.2fs); flooring duration_bars at %d",
            available_a, available_b_stretched, MIN_DURATION_BARS,
        )
        duration_bars = MIN_DURATION_BARS

    plan: MixPlanJSON = [
        {
            "tool": "set_transition_window",
            "from_song_time_start": seam_a,
            "to_song_time_start": seam_b,
            "duration_bars": duration_bars,
        }
    ]

    # No pitch_shift in Phase 7. The LARGE_SHIFT_THRESHOLD constant
    # below is retained for Phase 9's LLM-side reasoning; the threshold
    # is "what counts as a large shift" in the broader system, not just
    # a parameter to this generator.

    for stem in ("vocals", "drums", "bass", "other"):
        plan.append({
            "tool": "crossfade_stem",
            "stem": stem,
            "from_song": "A",
            "to_song": "B",
            "start_bar": 0,
            "duration_bars": duration_bars,
            "curve": "linear",
        })
    return plan
