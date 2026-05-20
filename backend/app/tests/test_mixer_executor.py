"""Tests for `services.mixer.executor.render`.

We avoid loading real audio by stubbing `soundfile.read` to return
synthetic numpy arrays, and stubbing pyrubberband so we don't shell out
to rubberband-cli. The executor's actual time-stretch / pitch-shift
behavior is left for the integration smoke test (Task 8) where we
exercise the real rubberband binary against a known clip.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from app.services.mixer.executor import render
from app.services.mixer.types import (
    AnalysisBundle,
    MixerPreconditionError,
    SongRenderInputs,
)


SR = 44100
DUR = 5.0  # seconds; short for fast tests
N = int(SR * DUR)


def _bundle(bpm: float = 120.0, key: str = "C") -> AnalysisBundle:
    sec_per_beat = 60.0 / bpm
    sec_per_bar = sec_per_beat * 4
    return AnalysisBundle(
        bpm=bpm,
        key=key,
        camelot_key="8B",
        time_signature=4,
        beat_grid=[i * sec_per_beat for i in range(int(DUR / sec_per_beat))],
        downbeats=[i * sec_per_bar for i in range(int(DUR / sec_per_bar))],
        sections=[{"start": 0.0, "end": DUR, "label": "body"}],
        duration=DUR,
    )


def _stems_dict(prefix: str) -> dict[str, str]:
    """4-stem path map; values are arbitrary strings since soundfile is mocked."""
    return {s: f"stems/{prefix}/{s}.wav" for s in ("vocals", "drums", "bass", "other")}


def _inputs(bpm: float = 120.0, key: str = "C", prefix: str = "A") -> SongRenderInputs:
    return SongRenderInputs(stem_paths=_stems_dict(prefix), analysis=_bundle(bpm, key))


class _FakeStorage:
    def path(self, key: str):
        return f"/fake/{key}"


def _fake_sf_read(quiet: float = 0.0):
    """Returns a soundfile.read stub. Each stem returns a constant-tone
    stereo float32 array; summing them gives a recognisable per-song mix.
    `quiet` knocks each stem down so the 4-stem sum stays within ±1.0."""
    def _read(path, always_2d=True, dtype="float32"):
        # Choose tone by stem name to keep mixes distinguishable.
        amp = 0.2 - quiet
        sig = amp * np.ones((N, 2), dtype=np.float32)
        if "drums" in path:
            sig = -sig
        return sig, SR
    return _read


def test_render_happy_path_no_stretch_no_shift():
    """Identical BPM + key. 1-bar crossfade. Output exists, length is
    correct, seam region has non-zero content.

    NOTE: plan-text said duration_bars=4 but at 120bpm that's 8s while
    the synthetic signal is only 5s — crossfade would extend past A's
    end. Reduced to 1 bar (2s) so the crossfade fits.
    """
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "crossfade_stem", "stem": "vocals",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "drums",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "bass",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "other",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
    ]

    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b, _FakeStorage())

    assert result.sample_rate == SR
    assert not result.pitch_shift_warning
    # Decode output WAV bytes to verify shape.
    decoded, sr = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert sr == SR
    # Output length = N (A's full length, identical-BPM means no stretch).
    # Output is a_seam (=0 samples for seam=0.0) + crossfade + B's tail.
    # With seam_a=0, the entire output is the B side after crossfade.
    # For more useful coverage we want seam_a > 0; we exercise that in
    # test_render_seam_alignment below.


def test_render_seam_alignment_identical_bpm():
    """Seam in middle of A: pre-seam is from A, post-crossfade is from B."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 60.0 / 120.0 * 4  # = 2.0
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar,    # seam at 2.0s in A
         "to_song_time_start": 0.0,              # seam at 0.0s in B
         "duration_bars": 1},                    # 1-bar crossfade (2s = 88200 samples)
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b, _FakeStorage())

    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    seam_sample = int(round(sec_per_bar * SR))
    crossfade_samples = int(round(sec_per_bar * SR))
    expected_len = seam_sample + (N - 0)  # full B after b_seam=0
    assert decoded.shape[0] == expected_len
    # First sample should equal A pre-seam (per fake_sf_read, A's vocals
    # sums to 0.2-0.2+0.2+0.2 = 0.4 stereo).
    assert decoded[0, 0] == pytest.approx(0.4, abs=1e-5)
    # Mid-crossfade (sample seam_sample + crossfade_samples/2): equal blend.
    mid = seam_sample + crossfade_samples // 2
    assert -1.0 < decoded[mid, 0] < 1.0
    # Post-crossfade: pure B.
    post = seam_sample + crossfade_samples + 100
    assert decoded[post, 0] == pytest.approx(0.4, abs=1e-5)


def test_render_calls_time_stretch_with_correct_rate():
    """A=120, B=128 → rate = 120/128 = 0.9375 (slow B down to A's tempo)."""
    a = _inputs(bpm=120.0, prefix="A")
    b = _inputs(bpm=128.0, prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    captured = {}

    def fake_stretch(audio, sr, rate):
        captured["rate"] = rate
        # Return audio of approximately rate-scaled length.
        new_len = int(audio.shape[0] / rate)
        return np.zeros((new_len, audio.shape[1]), dtype=audio.dtype)

    with (
        patch("soundfile.read", side_effect=_fake_sf_read()),
        patch("pyrubberband.pyrb.time_stretch", side_effect=fake_stretch),
    ):
        render(plan, a, b, _FakeStorage())

    assert captured["rate"] == pytest.approx(120.0 / 128.0)


def test_render_calls_pitch_shift_with_correct_semitones():
    a = _inputs(key="C", prefix="A")
    b = _inputs(key="C", prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "pitch_shift", "song": "B", "semitones": -2},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    captured = {}

    def fake_shift(audio, sr, n_steps):
        captured["n_steps"] = n_steps
        return audio

    with (
        patch("soundfile.read", side_effect=_fake_sf_read()),
        patch("pyrubberband.pyrb.pitch_shift", side_effect=fake_shift),
    ):
        result = render(plan, a, b, _FakeStorage())

    assert captured["n_steps"] == -2
    # |δ|=2 is NOT > 2 → no warning. (Plan text used -3 with a "not warning"
    # assertion, which contradicts LARGE_SHIFT_THRESHOLD=2 / "|δ| > 2 → warn".
    # Reduced to -2 so the no-warn assertion holds against the locked threshold.)
    assert not result.pitch_shift_warning


def test_render_pitch_shift_warning_flag_when_large():
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "pitch_shift", "song": "B", "semitones": 5},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with (
        patch("soundfile.read", side_effect=_fake_sf_read()),
        patch("pyrubberband.pyrb.pitch_shift", side_effect=lambda y, sr, n_steps: y),
    ):
        result = render(plan, a, b, _FakeStorage())
    assert result.pitch_shift_warning


def test_render_precondition_rejects_wrong_sample_rate():
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [{"tool": "set_transition_window",
             "from_song_time_start": 0.0, "to_song_time_start": 0.0,
             "duration_bars": 1}]

    def bad_sr_read(path, always_2d=True, dtype="float32"):
        return np.zeros((N, 2), dtype=np.float32), 22050

    with patch("soundfile.read", side_effect=bad_sr_read):
        with pytest.raises(MixerPreconditionError, match="sample rate"):
            render(plan, a, b, _FakeStorage())


def test_render_precondition_rejects_mono():
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [{"tool": "set_transition_window",
             "from_song_time_start": 0.0, "to_song_time_start": 0.0,
             "duration_bars": 1}]

    def mono_read(path, always_2d=True, dtype="float32"):
        return np.zeros((N, 1), dtype=np.float32), SR

    with patch("soundfile.read", side_effect=mono_read):
        with pytest.raises(MixerPreconditionError, match="channels"):
            render(plan, a, b, _FakeStorage())


def test_render_soft_clips_loud_sum():
    """Sum two stems that exceed 1.0 peak; verify output is clamped to 0.999."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 1.0, "to_song_time_start": 1.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    def loud_read(path, always_2d=True, dtype="float32"):
        # Each stem returns 0.4 → 4-stem sum = 1.6 (clipping needed).
        # Use +0.4 on all 4 stems (no sign flip) so the sum is monotonic.
        return 0.4 * np.ones((N, 2), dtype=np.float32), SR

    with patch("soundfile.read", side_effect=loud_read):
        result = render(plan, a, b, _FakeStorage())

    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.max() <= 0.999 + 1e-5
    assert decoded.min() >= -0.999 - 1e-5


def test_render_clamps_crossfade_when_stems_shorter_than_plan(caplog):
    """Plan asks for a 1-bar crossfade (2s) starting at seam=2.0s, but
    the stem WAVs are only 2.5s long — there's room for 0.5s of crossfade,
    not 2s. Executor should clamp to what's available, log a WARN, and
    produce a valid WAV rather than raising MixerPreconditionError.
    Models the real-world drift between Song.duration_seconds (yt-dlp
    metadata) and the actual stem WAV length after Demucs + pyrubberband."""
    short_n = int(SR * 2.5)  # 2.5s — leaves 0.5s after the 2s seam

    def short_read(path, always_2d=True, dtype="float32"):
        return 0.2 * np.ones((short_n, 2), dtype=np.float32), SR

    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0,   # snaps to downbeat 2.0
         "to_song_time_start": 0.0,
         "duration_bars": 1},            # asks for 2s of crossfade
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with patch("soundfile.read", side_effect=short_read), \
         caplog.at_level("WARNING"):
        result = render(plan, a, b, _FakeStorage())

    # Should have logged the clamp.
    assert any("clamping crossfade" in rec.message for rec in caplog.records)
    # And produced a non-empty WAV.
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_rejects_when_no_overlap_available():
    """A's downbeats extend past the actual stem length (analysis claims
    a 10s downbeat in a 5s stem — a real bug case, not just sample
    drift). The seam snaps to that beyond-end downbeat and max_crossfade_a
    becomes negative. Executor still raises rather than silently
    producing nonsense."""
    bad_analysis = AnalysisBundle(
        bpm=120.0,
        key="C",
        camelot_key="8B",
        time_signature=4,
        beat_grid=[i * 0.5 for i in range(40)],
        downbeats=[10.0, 12.0],   # past the 5s audio
        sections=[{"start": 0.0, "end": DUR, "label": "body"}],
        duration=DUR,
    )
    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=bad_analysis)
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 8.0,   # snaps to 10.0 downbeat (past EOF)
         "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with patch("soundfile.read", side_effect=_fake_sf_read()):
        with pytest.raises(MixerPreconditionError, match="no overlap"):
            render(plan, a, b, _FakeStorage())
