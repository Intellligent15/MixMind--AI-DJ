"""Seam-candidate generation for planner v2.

The single biggest failure mode of the free-form (v1) LLM planner was
asking the model to *derive* seam timestamps from a sections array —
LLMs are unreliable at that kind of arithmetic, so plans routinely
violated headroom/vocal-safety rules and were silently replaced by the
deterministic fallback.

Planner v2 inverts the responsibility: this module pre-computes a small
menu of *valid-by-construction* seam candidates (downbeat-snapped,
inside the headroom budget, annotated with section role / energy /
vocal-safety), and the LLM only has to pick from the menu — e.g.
"A2 → B1". A wrong number becomes impossible; a wrong *choice* is at
worst a matter of taste.

Pure module: no DB, no storage, no settings. Everything arrives via
arguments so it can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.mixer.types import AnalysisBundle

# Maximum crossfade length the planner may use; headroom is reserved for
# it when computing max_seam_time. Mirrors the deterministic planner.
MAX_CROSSFADE_BARS = 16
# Safety buffer (seconds) past the crossfade end. Absorbs stem-WAV drift
# vs `Song.duration_seconds` metadata (Demucs trims/pads, yt-dlp rounds).
SEAM_SAFETY_SECONDS = 5.0
# Cap the menu so the prompt stays small and the choice stays easy.
MAX_CANDIDATES = 5
# Two candidates closer than this are duplicates for mixing purposes.
DEDUP_SECONDS = 2.0
# B entry points past this fraction of the song are pointless — the
# listener would barely hear B before the *next* transition starts.
B_ENTRY_MAX_FRACTION = 0.5
# A vocal-safe entry must stay clear of vocals for this many bars after
# the candidate point for a hard cut to land cleanly.
VOCAL_SAFE_LOOKAHEAD_BARS = 2.0
# Sections with normalized energy at/above this read as a drop/chorus.
HIGH_ENERGY_THRESHOLD = 0.8


def max_seam_time(duration: float, bpm: float, time_signature: int) -> float:
    """Latest seam time that leaves room for a full-length crossfade.

    Returned in original-song seconds. Clamped to 0 if the song is
    shorter than the reserved tail.
    """
    if not bpm or not duration:
        return 0.0
    sec_per_bar = (60.0 / bpm) * time_signature
    return max(0.0, duration - MAX_CROSSFADE_BARS * sec_per_bar - SEAM_SAFETY_SECONDS)


def enrich_sections(sections: list[dict], energy_curve: list[float]) -> list[dict]:
    """Annotate each section with mean energy, normalized 0..1 over the song.

    The librosa analyzer's section `label`s are opaque cluster IDs
    (`section_1`…) that tell a planner nothing; a structure-aware
    detector (e.g. allin1) emits real roles (`chorus`, `verse`). We keep
    meaningful labels and drop the opaque ones. The energy curve is
    sampled at 1 Hz (index ≈ second); we average it per section and
    normalize to the song's hottest section so ~1.0 reads as a
    drop/chorus and low values as intro/breakdown/outro.
    """
    if not sections:
        return []
    n = len(energy_curve)
    raw: list[float] = []
    for s in sections:
        lo = int(s["start"])
        hi = max(lo + 1, int(round(s["end"])))
        window = energy_curve[lo:hi] if n else []
        raw.append(sum(window) / len(window) if window else 0.0)
    peak = max(raw) or 1.0
    out = []
    for s, e in zip(sections, raw):
        item = {
            "start": round(s["start"], 1),
            "end": round(s["end"], 1),
            "energy": round(e / peak, 2),
        }
        label = s.get("label")
        if label and not str(label).startswith("section_"):
            item["label"] = label
        out.append(item)
    return out


@dataclass(frozen=True)
class SeamCandidate:
    """One pre-validated seam point the LLM may choose."""

    id: str                # "A1", "B3", ...
    time: float            # seconds, original-song time, downbeat-snapped
    description: str       # human/LLM-readable role, e.g. "start of final section"
    energy: float          # normalized 0..1 section energy at this point
    vocal_safe: bool       # True if a hard cut here avoids chopping a word

    def to_llm_dict(self) -> dict:
        return {
            "id": self.id,
            "time": round(self.time, 2),
            "description": self.description,
            "energy": self.energy,
            "vocal_safe": self.vocal_safe,
        }


@dataclass(frozen=True)
class PairCandidates:
    out_candidates: list[SeamCandidate] = field(default_factory=list)
    in_candidates: list[SeamCandidate] = field(default_factory=list)

    def find(self, candidate_id: str) -> SeamCandidate | None:
        for c in (*self.out_candidates, *self.in_candidates):
            if c.id == candidate_id:
                return c
        return None


def _snap_to_downbeat(t: float, downbeats: list[float]) -> float:
    """First downbeat ≥ t, or the last downbeat if none qualifies."""
    if not downbeats:
        return t
    for d in downbeats:
        if d >= t:
            return d
    return downbeats[-1]


def _snap_to_downbeat_at_or_before(t: float, downbeats: list[float]) -> float:
    """Latest downbeat ≤ t, or the first downbeat if none qualifies."""
    if not downbeats:
        return t
    best = None
    for d in downbeats:
        if d <= t:
            best = d
        else:
            break
    return best if best is not None else downbeats[0]


def _is_vocal_safe(
    t: float,
    safe_regions: list[dict],
    lookahead_seconds: float,
) -> bool:
    """True when [t, t + lookahead] sits inside one no-vocal span.

    `safe_regions` rows are `{"start", "end"}` (optionally with a
    `safe`/`reason` field from the vocal-safety service — when a `safe`
    key is present, only rows with safe=True count). With no region data
    at all we conservatively return False: "unknown" must not read as
    "safe" or hard cuts would chop words on un-transcribed songs.
    """
    if not safe_regions:
        return False
    for r in safe_regions:
        if "safe" in r and not r.get("safe"):
            continue
        if r["start"] <= t and (t + lookahead_seconds) <= r["end"]:
            return True
    return False


def _dedup_and_cap(cands: list[SeamCandidate]) -> list[SeamCandidate]:
    cands = sorted(cands, key=lambda c: c.time)
    kept: list[SeamCandidate] = []
    for c in cands:
        if kept and abs(c.time - kept[-1].time) < DEDUP_SECONDS:
            continue
        kept.append(c)
    return kept[:MAX_CANDIDATES]


def _reid(cands: list[SeamCandidate], prefix: str) -> list[SeamCandidate]:
    return [
        SeamCandidate(
            id=f"{prefix}{i + 1}",
            time=c.time,
            description=c.description,
            energy=c.energy,
            vocal_safe=c.vocal_safe,
        )
        for i, c in enumerate(cands)
    ]


def build_out_candidates(
    a: AnalysisBundle,
    energy_curve: list[float],
    safe_regions: list[dict],
) -> list[SeamCandidate]:
    """OUT points for the outgoing song A: late-song section starts that
    leave a full crossfade of headroom, downbeat-snapped."""
    ceiling = max_seam_time(a.duration, a.bpm, a.time_signature)
    if ceiling <= 0:
        return []
    sec_per_bar = (60.0 / a.bpm) * a.time_signature if a.bpm else 0.0
    lookahead = VOCAL_SAFE_LOOKAHEAD_BARS * sec_per_bar
    sections = enrich_sections(a.sections, energy_curve)

    raw: list[SeamCandidate] = []
    n = len(sections)
    for i, s in enumerate(sections):
        t = _snap_to_downbeat(s["start"], a.downbeats)
        if t > ceiling:
            # Section starts too late to fit a crossfade — skip; the
            # "late as possible" fallback below covers the tail.
            continue
        # Only late-song sections are musical OUT points.
        if n >= 3 and i < n - 3:
            continue
        role = "final section" if i == n - 1 else (
            "second-to-last section" if i == n - 2 else "late section"
        )
        label = s.get("label")
        desc = f"start of {role}" + (f" ({label})" if label else "")
        raw.append(
            SeamCandidate(
                id="A?", time=t, description=desc, energy=s["energy"],
                vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
            )
        )

    # Always offer "as late as the headroom allows" — the v1 default.
    late = _snap_to_downbeat_at_or_before(ceiling, a.downbeats)
    if late <= ceiling:
        energy = _energy_at(sections, late)
        raw.append(
            SeamCandidate(
                id="A?", time=late,
                description="latest possible out point (16 bars before the end)",
                energy=energy,
                vocal_safe=_is_vocal_safe(late, safe_regions, lookahead),
            )
        )

    return _reid(_dedup_and_cap(raw), "A")


def build_in_candidates(
    b: AnalysisBundle,
    energy_curve: list[float],
    safe_regions: list[dict],
) -> list[SeamCandidate]:
    """IN points for the incoming song B: early-song moments — post-intro
    downbeat, first energy rise, first couple of section starts."""
    ceiling = min(
        max_seam_time(b.duration, b.bpm, b.time_signature),
        b.duration * B_ENTRY_MAX_FRACTION,
    )
    if ceiling <= 0:
        return []
    sec_per_bar = (60.0 / b.bpm) * b.time_signature if b.bpm else 0.0
    lookahead = VOCAL_SAFE_LOOKAHEAD_BARS * sec_per_bar
    sections = enrich_sections(b.sections, energy_curve)

    raw: list[SeamCandidate] = []

    # v1's default: first downbeat after the first section (skips silent
    # intros / count-ins).
    if sections:
        t = _snap_to_downbeat(sections[0]["end"], b.downbeats)
        if t <= ceiling:
            raw.append(
                SeamCandidate(
                    id="B?", time=t, description="end of intro / first section",
                    energy=_energy_at(sections, t),
                    vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
                )
            )
    else:
        t = _snap_to_downbeat(0.0, b.downbeats)
        if t <= ceiling:
            raw.append(
                SeamCandidate(
                    id="B?", time=t, description="start of the song",
                    energy=0.5, vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
                )
            )

    # First high-energy section start = "the first drop / chorus".
    for i, s in enumerate(sections):
        if s["energy"] >= HIGH_ENERGY_THRESHOLD:
            t = _snap_to_downbeat(s["start"], b.downbeats)
            if t <= ceiling:
                label = s.get("label")
                desc = "first high-energy section (drop/chorus)" + (
                    f" ({label})" if label else ""
                )
                raw.append(
                    SeamCandidate(
                        id="B?", time=t, description=desc, energy=s["energy"],
                        vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
                    )
                )
            break

    # Starts of sections 2 and 3 round out the early-song menu.
    for i, s in enumerate(sections[1:3], start=2):
        t = _snap_to_downbeat(s["start"], b.downbeats)
        if t <= ceiling:
            label = s.get("label")
            desc = f"start of section {i}" + (f" ({label})" if label else "")
            raw.append(
                SeamCandidate(
                    id="B?", time=t, description=desc, energy=s["energy"],
                    vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
                )
            )

    if not raw:
        # Degenerate analysis — offer the first usable downbeat.
        t = _snap_to_downbeat(0.0, b.downbeats)
        if t <= ceiling:
            raw.append(
                SeamCandidate(
                    id="B?", time=t, description="start of the song",
                    energy=0.5, vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
                )
            )

    return _reid(_dedup_and_cap(raw), "B")


def _energy_at(enriched_sections: list[dict], t: float) -> float:
    for s in enriched_sections:
        if s["start"] <= t < s["end"]:
            return s["energy"]
    return enriched_sections[-1]["energy"] if enriched_sections else 0.5


def build_pair_candidates(
    a: AnalysisBundle,
    b: AnalysisBundle,
    a_energy_curve: list[float],
    b_energy_curve: list[float],
    a_safe_regions: list[dict],
    b_safe_regions: list[dict],
) -> PairCandidates:
    return PairCandidates(
        out_candidates=build_out_candidates(a, a_energy_curve, a_safe_regions),
        in_candidates=build_in_candidates(b, b_energy_curve, b_safe_regions),
    )
