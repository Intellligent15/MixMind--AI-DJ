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


def _load_and_sum_stems(inputs: SongRenderInputs) -> np.ndarray:
    """Read all 4 stems from local paths, decode to float32 stereo, sum.

    Returns (samples, 2) float32. Raises MixerPreconditionError if any
    stem disagrees on sample rate or channel count.
    """
    arrays = []
    for stem in ("vocals", "drums", "bass", "other"):
        path = inputs.stem_paths.get(stem)
        if path is None:
            raise MixerPreconditionError(f"missing stem path: {stem!r}")
        data, sr = sf.read(path, always_2d=True, dtype="float32")
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


def _curve_envelopes(curve: str, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (a_gain, b_gain) envelopes for the given crossfade curve.

    `t` is in [0, 1). `a_gain` ramps down (1 → 0); `b_gain` ramps up (0 → 1).

    Curves:
    - "equal_power": cos(π/2·t) for A, sin(π/2·t) for B. Gains-squared sum
      to 1.0 at every t → perceived loudness flat. Industry standard for
      crossfading uncorrelated tracks. **Default for Phase 7.**
    - "linear": (1-t) for A, t for B. Gains sum to 1.0 but gains-squared
      sum to 0.5 at midpoint → audible -3 dB dip. Correct for correlated
      signals only.
    - "exponential", "s_curve": reserved for Phase 9 — raise
      NotImplementedError until the LLM has reason to emit them.
    """
    if curve == "equal_power":
        a_gain = np.cos(t * (np.pi / 2.0)).astype(np.float32)
        b_gain = np.sin(t * (np.pi / 2.0)).astype(np.float32)
        return a_gain, b_gain
    if curve == "linear":
        return (1.0 - t).astype(np.float32), t.astype(np.float32)
    raise NotImplementedError(
        f"crossfade curve {curve!r} not supported in Phase 7 "
        f"(supported: equal_power, linear)"
    )


def render(
    plan: MixPlanJSON,
    a: SongRenderInputs,
    b: SongRenderInputs,
) -> RenderedTransition:
    """Walk `plan` in order, produce a 44.1k stereo WAV.

    Supported tools (Phase 7): set_transition_window, pitch_shift,
    crossfade_stem. Any other tool raises NotImplementedError — Phase 9
    will grow the dispatch table.
    """
    # 1. Load + sum stems per song. Validates sr/channels.
    a_mix = _load_and_sum_stems(a)
    b_mix = _load_and_sum_stems(b)

    # 2. Walk the plan to extract tools
    window: dict | None = None
    stem_calls: list[dict] = []
    perm_pitch = None
    temp_pitch = None
    tempo_ramp = None

    for call in plan:
        tool = call["tool"]
        if tool == "set_transition_window":
            window = call
        elif tool == "pitch_shift":
            perm_pitch = call
        elif tool == "temporary_pitch_shift":
            temp_pitch = call
        elif tool == "set_tempo_ramp":
            tempo_ramp = call
        elif tool == "crossfade_stem":
            stem_calls.append(call)
        else:
            raise NotImplementedError(f"tool {tool!r} not supported in Phase 7")

    if window is None:
        raise MixerPreconditionError("plan is missing set_transition_window")

    # 3. Time-stretch B
    rate_A = a.analysis.bpm / b.analysis.bpm  # rubberband rate convention
    stretch_factor = 1.0 / rate_A             # how much longer B becomes
    
    if tempo_ramp:
        # Variable time stretch
        start_orig = tempo_ramp["start_time"]
        end_orig = tempo_ramp["end_time"]
        start_samp = int(start_orig * REQUIRED_SAMPLE_RATE)
        end_samp = int(end_orig * REQUIRED_SAMPLE_RATE)
        
        time_map = [(0, 0)]
        if start_samp > 0:
            time_map.append((start_samp, int(start_samp / rate_A)))
            
        ramp_len = end_samp - start_samp
        if ramp_len > 0:
            num_points = 10
            t_source = np.linspace(0, ramp_len, num_points)
            rates = np.linspace(rate_A, 1.0, num_points)
            t_target = np.zeros_like(t_source)
            for i in range(1, num_points):
                dt = t_source[i] - t_source[i-1]
                avg_rate = (rates[i] + rates[i-1]) / 2.0
                t_target[i] = t_target[i-1] + dt / avg_rate
                
            for s, t in zip(t_source[1:], t_target[1:]):
                time_map.append((int(start_samp + s), int(start_samp / rate_A + t)))
                
        last_source = time_map[-1][0]
        last_target = time_map[-1][1]
        remaining = b_mix.shape[0] - last_source
        if remaining > 0:
            time_map.append((b_mix.shape[0], int(last_target + remaining)))
            
        b_mix_stretched = np.asarray(
            pyrb.timemap_stretch(b_mix, REQUIRED_SAMPLE_RATE, time_map),
            dtype=np.float32
        )
        fade_out_target_len = int(t_target[-1]) if ramp_len > 0 else 0
    else:
        # Constant time stretch
        if abs(1.0 - stretch_factor) > 1e-6:
            b_mix_stretched = np.asarray(
                pyrb.time_stretch(b_mix, REQUIRED_SAMPLE_RATE, rate_A),
                dtype=np.float32,
            )
        else:
            b_mix_stretched = b_mix
        fade_out_target_len = 0

    # 4. Pitch shift B
    pitch_shift_warning = False
    
    def apply_pitch(audio, semitones):
        nonlocal pitch_shift_warning
        if abs(semitones) > LARGE_SHIFT_THRESHOLD:
            logger.warning(
                "render: large pitch shift applied (n_steps=%d); expect artifacts",
                semitones,
            )
            pitch_shift_warning = True
        return np.asarray(
            pyrb.pitch_shift(audio, REQUIRED_SAMPLE_RATE, semitones),
            dtype=np.float32,
        )

    if perm_pitch and int(perm_pitch["semitones"]) != 0:
        b_mix = apply_pitch(b_mix_stretched, int(perm_pitch["semitones"]))
    elif temp_pitch and int(temp_pitch["semitones"]) != 0:
        semitones = int(temp_pitch["semitones"])
        b_shifted = apply_pitch(b_mix_stretched, semitones)
        
        # Crossfade from shifted back to original
        fade_start_samp = int(temp_pitch["start_time"] * REQUIRED_SAMPLE_RATE / rate_A)
        # Using the same target length calculated during the tempo ramp
        fade_end_samp = fade_start_samp + fade_out_target_len
        
        # Clamp fade_end_samp to avoid out of bounds
        fade_start_samp = min(fade_start_samp, b_mix_stretched.shape[0])
        fade_end_samp = min(fade_end_samp, b_mix_stretched.shape[0])
        crossfade_len = fade_end_samp - fade_start_samp
        
        b_mix = np.copy(b_shifted)
        if crossfade_len > 0:
            t = np.linspace(0.0, 1.0, crossfade_len, endpoint=False, dtype=np.float32)
            gain_shifted = np.cos(t * (np.pi / 2.0))
            gain_unshifted = np.sin(t * (np.pi / 2.0))
            b_mix[fade_start_samp:fade_end_samp] = (
                gain_shifted[:, None] * b_shifted[fade_start_samp:fade_end_samp] +
                gain_unshifted[:, None] * b_mix_stretched[fade_start_samp:fade_end_samp]
            )
        b_mix[fade_end_samp:] = b_mix_stretched[fade_end_samp:]
    else:
        b_mix = b_mix_stretched

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

    # Bounds: the plan generator clamps based on Song.duration_seconds
    # (from yt-dlp metadata), but the *actual* stem WAV length can drift
    # by a few hundred ms — Demucs pads/trims, pyrubberband's stretch
    # adds/removes a handful of samples, yt-dlp rounds. Clamp gracefully
    # to whatever room actually exists rather than refusing the render.
    # Only raise when there's literally no overlap to work with — that
    # case represents a real bug (plan picked a seam past end-of-audio).
    max_crossfade_a = a_mix.shape[0] - a_seam_sample
    max_crossfade_b = b_mix.shape[0] - b_seam_sample
    if max_crossfade_a <= 0 or max_crossfade_b <= 0:
        raise MixerPreconditionError(
            f"no overlap available for crossfade "
            f"(a_seam={a_seam_sample}/len_a={a_mix.shape[0]}, "
            f"b_seam={b_seam_sample}/len_b={b_mix.shape[0]})"
        )
    available = min(max_crossfade_a, max_crossfade_b)
    if crossfade_samples > available:
        logger.warning(
            "render: clamping crossfade from %d to %d samples "
            "(%.3fs -> %.3fs) — stem WAV length drifted from "
            "Song.duration_seconds by ~%d samples",
            crossfade_samples,
            available,
            crossfade_samples / REQUIRED_SAMPLE_RATE,
            available / REQUIRED_SAMPLE_RATE,
            crossfade_samples - available,
        )
        crossfade_samples = available

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
    curve = stem_calls[0]["curve"]
    a_gain, b_gain = _curve_envelopes(curve, t)
    a_region = a_mix[a_seam_sample : a_seam_sample + crossfade_samples]
    b_region = b_mix[b_seam_sample : b_seam_sample + crossfade_samples]
    out[a_seam_sample : a_seam_sample + crossfade_samples] = (
        a_gain[:, None] * a_region + b_gain[:, None] * b_region
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
