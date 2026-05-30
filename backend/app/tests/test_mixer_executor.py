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
        result = render(plan, a, b)

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
        result = render(plan, a, b)

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
        render(plan, a, b)

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
        result = render(plan, a, b)

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
        result = render(plan, a, b)
    assert result.pitch_shift_warning


def test_render_temporary_pitch_mutes_vocal_then_fades_it_in():
    """Temporary-pitch path: B's vocal is silent through the ENTIRE pitch
    return (including the crossfade-back window) and only fades in once the
    key has fully settled. We make only the vocal stem non-zero so we can
    read its presence directly out of the rendered B body."""
    long_dur = 16.0
    long_n = int(SR * long_dur)
    sec_per_bar = 2.0  # 120 bpm, 4/4

    def _bundle_long(prefix_bpm: float = 120.0) -> AnalysisBundle:
        return AnalysisBundle(
            bpm=prefix_bpm, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * sec_per_bar for i in range(int(long_dur / sec_per_bar))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    def _read_vocal_only(path, always_2d=True, dtype="float32"):
        amp = 0.5 if "vocals" in path else 0.0
        return amp * np.ones((long_n, 2), dtype=np.float32), SR

    # Pitch shift starts at 4 s and returns over a 2-bar (4 s) window → native
    # by 8 s. The vocal stays silent through that whole return and only fades
    # in afterwards, over [8 s, 12 s], full from 12 s on. The tempo ramp is
    # longer (4–10 s) and intentionally decoupled.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "set_tempo_ramp", "song": "B",
         "start_time": 4.0, "end_time": 10.0, "start_bpm": 120.0, "end_bpm": 120.0},
        {"tool": "temporary_pitch_shift", "song": "B", "start_time": 4.0,
         "semitones": -4, "fade_in_bars": 0, "hold_bars": 0, "fade_out_bars": 2},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with (
        patch("soundfile.read", side_effect=_read_vocal_only),
        patch("pyrubberband.pyrb.pitch_shift", side_effect=lambda y, sr, n: y),
        patch("pyrubberband.pyrb.timemap_stretch", side_effect=lambda y, sr, tm: y),
    ):
        result = render(plan, a, b)

    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Before the pitch return (~3 s): vocal silent.
    pre = np.max(np.abs(out[int(3.0 * SR): int(3.1 * SR)]))
    assert pre < 0.05, f"vocal should be muted before pitch return, got {pre}"
    # DURING the pitch return (~6 s, inside [4 s, 8 s]): still silent — this is
    # the behavior we want (no vocal over the detuned/blended instrumental).
    during = np.max(np.abs(out[int(6.0 * SR): int(6.1 * SR)]))
    assert during < 0.05, f"vocal should be muted during pitch return, got {during}"
    # After the key has fully settled (~14 s): vocal at full level.
    present = np.mean(np.abs(out[int(14.0 * SR): int(14.1 * SR)]))
    assert present > 0.4, f"vocal should be present once key settles, got {present}"
    assert result.pitch_shift_warning  # |-4| > 2


def test_render_tempo_ramp_alone_mutes_vocal_then_fades_it_in():
    """Tempo-only path (no temp pitch shift): B's vocal must be silent
    while the tempo ramp is active and fade in only once B has settled at
    native tempo. Same fade window as the temp-pitch path so the two
    behave consistently."""
    long_dur = 16.0
    long_n = int(SR * long_dur)
    sec_per_bar = 2.0  # 120 bpm, 4/4

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * sec_per_bar for i in range(int(long_dur / sec_per_bar))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    def _read_vocal_only(path, always_2d=True, dtype="float32"):
        amp = 0.5 if "vocals" in path else 0.0
        return amp * np.ones((long_n, 2), dtype=np.float32), SR

    # Tempo ramp 4 s → 10 s, no temp pitch. Same BPM end-to-end so the
    # mocked timemap_stretch is a passthrough. Vocal stays silent through
    # the entire ramp, then fades in over [10 s, 14 s], full from 14 s on.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "set_tempo_ramp", "song": "B",
         "start_time": 4.0, "end_time": 10.0, "start_bpm": 120.0, "end_bpm": 120.0},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with (
        patch("soundfile.read", side_effect=_read_vocal_only),
        patch("pyrubberband.pyrb.timemap_stretch", side_effect=lambda y, sr, tm: y),
    ):
        result = render(plan, a, b)

    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Mid-ramp (~7 s): vocal must be muted.
    during = np.max(np.abs(out[int(7.0 * SR): int(7.1 * SR)]))
    assert during < 0.05, f"vocal should be muted during tempo ramp, got {during}"
    # Just before settle (~9 s): still muted.
    just_before = np.max(np.abs(out[int(9.0 * SR): int(9.1 * SR)]))
    assert just_before < 0.05, f"vocal should still be muted at 9s, got {just_before}"
    # After the vocal fade-in completes (~15 s): vocal at full level.
    present = np.mean(np.abs(out[int(15.0 * SR): int(15.1 * SR)]))
    assert present > 0.4, f"vocal should be present once tempo settles, got {present}"


def test_render_uses_original_master_for_a_head():
    """A's body before the seam comes from the untouched master, not the
    stem sum. A master reads 0.3; the 4-stem sum reads 0.4."""
    a = SongRenderInputs(
        stem_paths=_stems_dict("A"), analysis=_bundle(),
        original_audio_path="orig/A_master.wav",
    )
    b = _inputs(prefix="B")  # no original → B stays stem-based

    def _read(path, always_2d=True, dtype="float32"):
        if "master" in path:
            return 0.3 * np.ones((N, 2), dtype=np.float32), SR
        sig = 0.2 * np.ones((N, 2), dtype=np.float32)
        if "drums" in path:
            sig = -sig
        return sig, SR  # 4-stem sum = 0.2-0.2+0.2+0.2 = 0.4

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Pre-seam (~0.5 s, before the 2 s seam): the untouched master (0.3),
    # not the stem sum (0.4).
    head = np.mean(np.abs(out[int(0.5 * SR): int(0.6 * SR)]))
    assert abs(head - 0.3) < 0.02, f"A head should be master 0.3, got {head}"


def test_render_splices_original_master_into_settled_b_tail():
    """Once B has fully settled (tempo+pitch+vocal native), its tail is the
    untouched master, not the stem reconstruction. Same 16 s temp-pitch
    setup as the mute test, but B carries an original master reading 0.9."""
    long_dur = 16.0
    long_n = int(SR * long_dur)
    sec_per_bar = 2.0

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * sec_per_bar for i in range(int(long_dur / sec_per_bar))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(
        stem_paths=_stems_dict("B"), analysis=_bundle_long(),
        original_audio_path="orig/B_master.wav",
    )

    def _read(path, always_2d=True, dtype="float32"):
        if "master" in path:
            return 0.9 * np.ones((long_n, 2), dtype=np.float32), SR
        amp = 0.5 if "vocals" in path else 0.0
        return amp * np.ones((long_n, 2), dtype=np.float32), SR

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0, "duration_bars": 1},
        {"tool": "set_tempo_ramp", "song": "B",
         "start_time": 4.0, "end_time": 8.0, "start_bpm": 120.0, "end_bpm": 120.0},
        {"tool": "temporary_pitch_shift", "song": "B", "start_time": 4.0,
         "semitones": -4, "fade_in_bars": 0, "hold_bars": 0, "fade_out_bars": 2},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with (
        patch("soundfile.read", side_effect=_read),
        patch("pyrubberband.pyrb.pitch_shift", side_effect=lambda y, sr, n: y),
        patch("pyrubberband.pyrb.timemap_stretch", side_effect=lambda y, sr, tm: y),
    ):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Settled tail (~14 s): the master (0.9), not the stem vocal level (0.5).
    tail = np.mean(np.abs(out[int(14.0 * SR): int(14.1 * SR)]))
    assert abs(tail - 0.9) < 0.05, f"settled tail should be master 0.9, got {tail}"
    # During the pitch return (~6 s): still the stem path, master NOT yet
    # spliced (vocal muted → ~0).
    during = np.max(np.abs(out[int(6.0 * SR): int(6.1 * SR)]))
    assert during < 0.05, f"master must not appear during transition, got {during}"


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
            render(plan, a, b)


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
            render(plan, a, b)


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
        result = render(plan, a, b)

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
        result = render(plan, a, b)

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
            render(plan, a, b)


def test_render_equal_power_curve_keeps_loudness_flat_at_midpoint():
    """Equal-power crossfade: at t=0.5 both gains = cos(π/4) = sin(π/4) ≈
    0.707. Mid-crossfade output of two identical signals (both at amp 0.4)
    should be 0.4 × (0.707 + 0.707) = 0.566, NOT 0.4 (which linear would
    produce). This is what 'no midpoint dip' means in practice: when both
    inputs are at +0.4, the output is louder by sqrt(2) than either input
    alone, which is the correct sum-of-uncorrelated-powers behavior."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 60.0 / 120.0 * 4  # = 2.0

    def steady_read(path, always_2d=True, dtype="float32"):
        # Identical positive constant for both songs → A and B at +0.4
        # after the 4-stem sum (4 × 0.1).
        return 0.1 * np.ones((N, 2), dtype=np.float32), SR

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=steady_read):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    seam_sample = int(round(sec_per_bar * SR))
    crossfade_samples = int(round(sec_per_bar * SR))
    mid = seam_sample + crossfade_samples // 2
    # Equal-power midpoint: cos(π/4)*0.4 + sin(π/4)*0.4 ≈ 0.5657
    assert decoded[mid, 0] == pytest.approx(0.4 * np.sqrt(2.0), abs=1e-3)
    # Compare against the linear case at the same midpoint: 0.4 (no boost).
    plan_linear = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=steady_read):
        result_lin = render(plan_linear, a, b)
    decoded_lin, _ = sf.read(io.BytesIO(result_lin.wav_bytes), always_2d=True)
    assert decoded_lin[mid, 0] == pytest.approx(0.4, abs=1e-3)


def test_render_rejects_unsupported_curve():
    """s_curve and exponential are reserved for Phase 9; the executor
    refuses them rather than silently falling back to linear."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "s_curve"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        with pytest.raises(NotImplementedError, match="s_curve"):
            render(plan, a, b)
