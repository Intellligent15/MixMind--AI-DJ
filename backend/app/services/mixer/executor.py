"""Mixer executor — walks a MixPlanJSON and produces a stereo WAV.

Pure-ish: reads stem WAVs from disk through the StorageBackend, calls
pyrubberband for time-stretch / pitch-shift, returns the rendered WAV as
bytes. No DB.

Coordinate convention: tool-call times are in *original-song* time
(seconds in A, seconds in B before any stretch). The executor handles
the original→post-stretch translation internally. This matches what
the Phase 9 LLM will produce — it sees analyses in original time.

Full algorithm + math: see
the design notes
→ Layer 2 — services/mixer/executor.py.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pyrubberband.pyrb as pyrb
import soundfile as sf

from app.services.mixer.types import (
    AnalysisBundle,
    MixerPreconditionError,
    MixPlanJSON,
    RenderedTransition,
    SongRenderInputs,
)

logger = logging.getLogger(__name__)

REQUIRED_SAMPLE_RATE = 44100
REQUIRED_CHANNELS = 2
SOFT_CLIP_CEILING = 0.999
LARGE_SHIFT_THRESHOLD = 2


def _load_and_sum_stems(inputs: SongRenderInputs, storage) -> np.ndarray:
    """Read all 4 stems via storage, decode to float32 stereo, sum.

    Returns (samples, 2) float32. Raises MixerPreconditionError if any
    stem disagrees on sample rate or channel count.
    """
    arrays = []
    for stem in ("vocals", "drums", "bass", "other"):
        key = inputs.stem_paths.get(stem)
        if key is None:
            raise MixerPreconditionError(f"missing stem path: {stem!r}")
        path = storage.path(key)
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        if sr != REQUIRED_SAMPLE_RATE:
            raise MixerPreconditionError(
                f"stem {stem} sample rate {sr} != required {REQUIRED_SAMPLE_RATE}"
            )
        if data.shape[1] != REQUIRED_CHANNELS:
            raise MixerPreconditionError(
                f"stem {stem} channels {data.shape[1]} != required {REQUIRED_CHANNELS}"
            )
        arrays.append(data)
    # Pad to the max length in case stems happen to differ by a sample.
    n = max(a.shape[0] for a in arrays)
    summed = np.zeros((n, REQUIRED_CHANNELS), dtype=np.float32)
    for a in arrays:
        summed[: a.shape[0]] += a
    return summed


def _snap_downbeat(t: float, downbeats: list[float]) -> float:
    if not downbeats:
        return t
    for d in downbeats:
        if d >= t:
            return d
    return downbeats[-1]


def _all_stems_share_envelope(stem_calls: list[dict]) -> bool:
    """True when 4 crossfade_stem calls (one per canonical stem) share
    every envelope parameter except the stem name."""
    stems_seen = {c["stem"] for c in stem_calls}
    if stems_seen != {"vocals", "drums", "bass", "other"}:
        return False
    keys = ("from_song", "to_song", "start_bar", "duration_bars", "curve")
    first = stem_calls[0]
    return all(
        all(c[k] == first[k] for k in keys) for c in stem_calls[1:]
    )


def render(
    plan: MixPlanJSON,
    a: SongRenderInputs,
    b: SongRenderInputs,
    storage,
) -> RenderedTransition:
    """Walk `plan` in order, produce a 44.1k stereo WAV.

    Supported tools (Phase 7): set_transition_window, pitch_shift,
    crossfade_stem. Any other tool raises NotImplementedError — Phase 9
    will grow the dispatch table.
    """
    # 1. Load + sum stems per song. Validates sr/channels.
    a_mix = _load_and_sum_stems(a, storage)
    b_mix = _load_and_sum_stems(b, storage)

    # 2. Time-stretch B to A's BPM (entire B). pyrubberband rate convention:
    #    rate > 1 = faster/shorter. To go from b.bpm to a.bpm we want
    #    rate = a.bpm / b.bpm. When a < b, rate < 1 → B is slowed (longer).
    stretch_factor = b.analysis.bpm / a.analysis.bpm  # how much longer B becomes
    if abs(1.0 - stretch_factor) > 1e-6:
        rate = a.analysis.bpm / b.analysis.bpm
        b_mix = np.asarray(
            pyrb.time_stretch(b_mix, REQUIRED_SAMPLE_RATE, rate),
            dtype=np.float32,
        )

    # 3. Walk the plan, collecting state.
    window: dict | None = None
    pitch_shift_warning = False
    stem_calls: list[dict] = []

    for call in plan:
        tool = call["tool"]
        if tool == "set_transition_window":
            window = call
        elif tool == "pitch_shift":
            if call["song"] != "B":
                raise NotImplementedError(
                    f"pitch_shift on song {call['song']!r} not supported in Phase 7"
                )
            n_steps = int(call["semitones"])
            if n_steps != 0:
                if abs(n_steps) > LARGE_SHIFT_THRESHOLD:
                    logger.warning(
                        "render: large pitch shift applied (n_steps=%d); "
                        "expect pyrubberband artifacts",
                        n_steps,
                    )
                    pitch_shift_warning = True
                b_mix = np.asarray(
                    pyrb.pitch_shift(b_mix, REQUIRED_SAMPLE_RATE, n_steps),
                    dtype=np.float32,
                )
        elif tool == "crossfade_stem":
            stem_calls.append(call)
        else:
            raise NotImplementedError(f"tool {tool!r} not supported in Phase 7")

    if window is None:
        raise MixerPreconditionError("plan is missing set_transition_window")
    if len(stem_calls) != 4:
        raise MixerPreconditionError(
            f"plan must have 4 crossfade_stem calls, got {len(stem_calls)}"
        )
    if not _all_stems_share_envelope(stem_calls):
        # Phase 9 territory: per-stem envelopes. Not supported in Phase 7.
        raise NotImplementedError(
            "per-stem envelope variation requires Phase 9 mixer support"
        )

    duration_bars = int(stem_calls[0]["duration_bars"])

    # 4. Resolve seam downbeats. A's downbeats are unchanged (A isn't
    #    stretched); B's seam in original time → post-stretch by 1/rate.
    a_seam = _snap_downbeat(
        float(window["from_song_time_start"]), a.analysis.downbeats
    )
    b_seam_orig = _snap_downbeat(
        float(window["to_song_time_start"]), b.analysis.downbeats
    )
    b_seam_post = b_seam_orig * stretch_factor

    a_seam_sample = int(round(a_seam * REQUIRED_SAMPLE_RATE))
    b_seam_sample = int(round(b_seam_post * REQUIRED_SAMPLE_RATE))

    # 5. Crossfade length in samples.
    crossfade_seconds = (
        duration_bars * (60.0 / a.analysis.bpm) * a.analysis.time_signature
    )
    crossfade_samples = int(round(crossfade_seconds * REQUIRED_SAMPLE_RATE))

    # Bounds checks (the plan generator should have clamped, but assert).
    if a_seam_sample + crossfade_samples > a_mix.shape[0]:
        raise MixerPreconditionError(
            f"crossfade extends past A's end "
            f"(a_seam={a_seam_sample}, crossfade={crossfade_samples}, "
            f"len_a={a_mix.shape[0]})"
        )
    if b_seam_sample + crossfade_samples > b_mix.shape[0]:
        raise MixerPreconditionError(
            f"crossfade extends past B's end "
            f"(b_seam={b_seam_sample}, crossfade={crossfade_samples}, "
            f"len_b={b_mix.shape[0]})"
        )

    # 6. Build output buffer.
    out_len = a_seam_sample + (b_mix.shape[0] - b_seam_sample)
    out = np.zeros((out_len, REQUIRED_CHANNELS), dtype=np.float32)

    # Pre-seam: pure A.
    out[:a_seam_sample] = a_mix[:a_seam_sample]

    # Crossfade region. endpoint=False is load-bearing: t runs
    # [0, 1/N, 2/N, ..., (N-1)/N], so the last crossfade sample is still
    # slightly A-weighted and the next sample (post-seam, pure B from
    # b_mix[b_seam_sample + crossfade_samples:]) is the natural continuation.
    # Switching to endpoint=True would put a pure-B sample at the end of
    # the crossfade region AND a pure-B sample as the first post-seam
    # sample — a one-frame discontinuity right where the listener is
    # paying attention.
    t = np.linspace(0.0, 1.0, crossfade_samples, endpoint=False, dtype=np.float32)
    t_stereo = t[:, None]  # broadcast to (samples, 1) for stereo math
    a_region = a_mix[a_seam_sample : a_seam_sample + crossfade_samples]
    b_region = b_mix[b_seam_sample : b_seam_sample + crossfade_samples]
    out[a_seam_sample : a_seam_sample + crossfade_samples] = (
        (1.0 - t_stereo) * a_region + t_stereo * b_region
    )

    # Post-seam: pure B (already in post-stretch coords).
    out[a_seam_sample + crossfade_samples :] = b_mix[b_seam_sample + crossfade_samples :]

    # 7. Soft-clip to SOFT_CLIP_CEILING.
    peak = float(np.max(np.abs(out)))
    if peak > SOFT_CLIP_CEILING:
        attenuation = SOFT_CLIP_CEILING / peak
        logger.info(
            "render: soft-clipping output (peak=%.4f, attenuation=%.4f)",
            peak, attenuation,
        )
        out *= attenuation

    # 8. Encode to 16-bit PCM WAV in memory.
    buf = io.BytesIO()
    sf.write(buf, out, REQUIRED_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    duration_seconds = out.shape[0] / REQUIRED_SAMPLE_RATE
    return RenderedTransition(
        wav_bytes=buf.getvalue(),
        sample_rate=REQUIRED_SAMPLE_RATE,
        duration_seconds=duration_seconds,
        pitch_shift_warning=pitch_shift_warning,
    )
