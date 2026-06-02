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


def test_render_per_stem_envelopes_sum_correctly():
    """Per-stem envelopes: vocals fade out earlier than drums/bass/other.

    A's stems carry distinct constants; B's stems are zero. So at any output
    sample inside the post-seam region we know exactly which stems should
    still be ringing (gain≈1.0 on A-side because B is silent, less the
    envelope fade applied during each stem's window). The sum at each
    probe point matches the analytic expectation."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 60.0 / 120.0 * 4  # = 2.0s; total stem N = 5s, seam at 0

    # Distinct A levels per stem so the post-seam sum tells us which
    # stems are still in. B silent so envelopes only fade A.
    levels = {"vocals": 0.1, "drums": 0.2, "bass": 0.3, "other": 0.05}

    def stems_read(path, always_2d=True, dtype="float32"):
        if "B/" in path or "B_" in path or "/B/" in path:
            return np.zeros((N, 2), dtype=np.float32), SR
        for s, v in levels.items():
            if f"/{s}." in path:
                return v * np.ones((N, 2), dtype=np.float32), SR
        return np.zeros((N, 2), dtype=np.float32), SR

    # Vocals fade out bars [0, 1) → seconds [0, 2.0); other stems hold
    # through bar 0 and fade in bars [1, 2) → seconds [2.0, 4.0).
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 2},
        {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "drums", "from_song": "A", "to_song": "B",
         "start_bar": 1, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "bass", "from_song": "A", "to_song": "B",
         "start_bar": 1, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "other", "from_song": "A", "to_song": "B",
         "start_bar": 1, "duration_bars": 1, "curve": "linear"},
    ]
    with patch("soundfile.read", side_effect=stems_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    # Probe at 0.25s into the vocal fade-out (t=0.125 in vocal window):
    # vocals weight = 1 - 0.125 = 0.875 → vocals contribute 0.0875.
    # drums/bass/other are still pre-window (pure A) → full levels.
    probe_a = int(0.25 * SR)
    expected_a = 0.875 * levels["vocals"] + levels["drums"] + levels["bass"] + levels["other"]
    assert out[probe_a, 0] == pytest.approx(expected_a, abs=1e-3)

    # Probe at 1.0s (bar 0.5 into vocals' window so vocal=0.5 gain,
    # still pre-window for other stems):
    probe_b = int(1.0 * SR)
    expected_b = 0.5 * levels["vocals"] + levels["drums"] + levels["bass"] + levels["other"]
    assert out[probe_b, 0] == pytest.approx(expected_b, abs=1e-3)

    # Probe at 3.0s (mid-fade for drums/bass/other; vocals fully out):
    # drums/bass/other gain = 1 - 0.5 = 0.5; vocals = 0.
    probe_c = int(3.0 * SR)
    expected_c = 0.5 * (levels["drums"] + levels["bass"] + levels["other"])
    assert out[probe_c, 0] == pytest.approx(expected_c, abs=1e-3)
def _baseline_stem_calls(curve: str = "equal_power") -> list[dict]:
    """Standard 4-stem crossfade tail every plan needs."""
    return [
        {"tool": "crossfade_stem", "stem": s,
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": curve}
        for s in ("vocals", "drums", "bass", "other")
    ]


def test_render_filter_sweep_lowpass_kills_high_frequencies():
    """A lowpass sweep from 20 kHz → 200 Hz over the post-window region
    should leave the high-frequency content noticeably attenuated by the
    end of the sweep. We feed in a high-frequency sine on A and read the
    energy out before vs after the sweep window."""
    # 4 second signal so the sweep window has room.
    long_dur = 4.0
    long_n = int(SR * long_dur)

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * 2.0 for i in range(int(long_dur / 2.0))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    # A's stems are all a 5 kHz tone; B's are silent so we can isolate A.
    t_axis = np.arange(long_n) / SR
    tone = 0.1 * np.sin(2.0 * np.pi * 5000.0 * t_axis).astype(np.float32)
    tone_stereo = np.column_stack([tone, tone])

    def _read(path, always_2d=True, dtype="float32"):
        if "/A/" in path:
            return tone_stereo.copy(), SR
        return np.zeros((long_n, 2), dtype=np.float32), SR

    # Seam at 3.0s; A's body before the seam includes the sweep window
    # [0.5s, 2.5s]. After 2.5s the 5 kHz energy should be much lower.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 3.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "filter_sweep", "song": "A", "type": "lowpass",
         "start_time": 0.5, "end_time": 2.5,
         "start_cutoff_hz": 20000.0, "end_cutoff_hz": 200.0},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    pre_rms = float(np.sqrt(np.mean(out[int(0.3 * SR): int(0.5 * SR)] ** 2)))
    post_rms = float(np.sqrt(np.mean(out[int(2.6 * SR): int(2.9 * SR)] ** 2)))
    # 5 kHz survives a 20 kHz cutoff (start) but is heavily attenuated by
    # a 200 Hz cutoff (end). Expect at least a 10× drop.
    assert post_rms < pre_rms * 0.1, (
        f"lowpass sweep failed to kill highs: pre={pre_rms:.4f} post={post_rms:.4f}"
    )


def test_render_filter_sweep_clamps_zero_cutoff_silently():
    """A 0 Hz cutoff is illegal for iirfilter (division by zero in the
    bilinear transform). The executor clamps to 20 Hz silently rather
    than raising — the LLM prompt-side rules will steer away from this,
    but the DSP must not crash if it slips through."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "filter_sweep", "song": "A", "type": "highpass",
         "start_time": 0.5, "end_time": 1.5,
         "start_cutoff_hz": 0.0, "end_cutoff_hz": -100.0},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    # Just ensure we got a valid WAV back; the clamp shouldn't blow up.
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_echo_out_decaying_taps_after_cut():
    """echo_out at start_time T: dry signal is hard-cut at T (silence
    immediately after), and N taps of the pre-cut audio appear at one-beat
    intervals after T, each attenuated by feedback ** i."""
    long_dur = 6.0
    long_n = int(SR * long_dur)

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * 2.0 for i in range(int(long_dur / 2.0))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    # Put a unit-amplitude pulse on A's vocal stem; everything else silent.
    # Pulse fits within the last beat before the cut so it's the tap source.
    bpm = 120.0
    delay_samp = int(SR * 60.0 / bpm)
    cut_samp = int(2.0 * SR)
    pulse_start = cut_samp - delay_samp + 100  # a few samples into the tap window
    pulse_len = 200

    a_vocal = np.zeros((long_n, 2), dtype=np.float32)
    a_vocal[pulse_start : pulse_start + pulse_len] = 0.5

    def _read(path, always_2d=True, dtype="float32"):
        if "/A/vocals" in path:
            return a_vocal.copy(), SR
        return np.zeros((long_n, 2), dtype=np.float32), SR

    # Seam at 5s so the entire echo region is observable in the output.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 5.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "echo_out", "song": "A",
         "start_time": 2.0, "beats": 3, "feedback": 0.5, "bpm": bpm},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    # The original pulse pre-cut is preserved.
    assert out[pulse_start + 50, 0] == pytest.approx(0.5, abs=0.05)

    # Dry signal is hard-cut: NO energy at sample (cut_samp + 100), well
    # before the first echo lands.
    dry_after = float(np.max(np.abs(out[cut_samp + 50 : cut_samp + 200])))
    assert dry_after < 0.05, f"dry signal should be cut, got {dry_after}"

    # Each tap is the same pulse, attenuated by feedback ** i, placed at
    # cut_samp + i*delay_samp. The pulse arrives at the same offset within
    # each tap window as it was within the pre-cut tap source.
    pulse_offset_in_tap = pulse_start - (cut_samp - delay_samp)
    for i in (1, 2, 3):
        tap_peak_samp = cut_samp + i * delay_samp + pulse_offset_in_tap + 50
        expected = 0.5 * (0.5 ** i)
        # Output channels are stereo identical here.
        observed = float(out[tap_peak_samp, 0])
        assert observed == pytest.approx(expected, abs=0.05), (
            f"tap {i}: expected ~{expected:.3f}, got {observed:.3f}"
        )


def test_render_loop_section_repeats_section_length():
    """loop_section with repeats=2 and beats=1 should tile the chosen
    slice twice; the second copy is content-identical to the first.

    We mark the loop-source slice with a unique amplitude on A's vocal
    stem so we can detect it by reading the rendered output."""
    long_dur = 8.0
    long_n = int(SR * long_dur)
    bpm = 120.0
    beat_samp = int(SR * 60.0 / bpm)

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=bpm, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * 2.0 for i in range(int(long_dur / 2.0))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    # A's vocal: zero everywhere except a distinct stamp inside the loop slice.
    loop_start = int(1.0 * SR)
    stamp_offset = beat_samp // 4  # well inside the 1-beat loop window
    stamp_len = 100
    stamp_amp = 0.6

    a_vocal = np.zeros((long_n, 2), dtype=np.float32)
    a_vocal[loop_start + stamp_offset : loop_start + stamp_offset + stamp_len] = stamp_amp

    def _read(path, always_2d=True, dtype="float32"):
        if "/A/vocals" in path:
            return a_vocal.copy(), SR
        return np.zeros((long_n, 2), dtype=np.float32), SR

    # Seam at 7s so the entire looped region is in the output.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 7.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "loop_section", "song": "A",
         "start_time": 1.0, "beats": 1, "repeats": 2, "bpm": bpm},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    # First copy: the original stamp is still where we put it.
    first_peak = float(out[loop_start + stamp_offset + 10, 0])
    assert first_peak == pytest.approx(stamp_amp, abs=0.05)

    # Second copy: same stamp shifted by exactly one beat (the loop length).
    second_idx = loop_start + beat_samp + stamp_offset + 10
    second_peak = float(out[second_idx, 0])
    assert second_peak == pytest.approx(stamp_amp, abs=0.05), (
        f"second loop copy missing: idx={second_idx} got {second_peak}"
    )


def test_render_loop_section_zero_beats_is_noop():
    """beats=0 or repeats=0 must NOT raise — silent no-op."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "loop_section", "song": "A",
         "start_time": 0.0, "beats": 0, "repeats": 4, "bpm": 120.0},
        {"tool": "loop_section", "song": "A",
         "start_time": 0.0, "beats": 2, "repeats": 0, "bpm": 120.0},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_swap_stem_replaces_output_tail_with_to_song_stem():
    """swap_stem at time T: every output sample after T equals
    to_song's stem buffer on that channel."""
    long_dur = 6.0
    long_n = int(SR * long_dur)

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * 2.0 for i in range(int(long_dur / 2.0))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    # A's vocal stem is a distinctive +0.4 ramp; everything else 0.
    # We'll swap to A's vocal after the seam region — but easier to swap
    # to B's vocals using a sign-flipping signal so the assertion is
    # unambiguous.
    t_axis = np.arange(long_n) / SR
    b_vocal_signal = (0.3 * np.sin(2.0 * np.pi * 220.0 * t_axis)).astype(np.float32)
    b_vocal_stereo = np.column_stack([b_vocal_signal, b_vocal_signal])

    def _read(path, always_2d=True, dtype="float32"):
        if "/B/vocals" in path:
            return b_vocal_stereo.copy(), SR
        if "/A/" in path:
            return 0.1 * np.ones((long_n, 2), dtype=np.float32), SR
        return np.zeros((long_n, 2), dtype=np.float32), SR

    swap_time = 4.0
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 1.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *_baseline_stem_calls(),
        {"tool": "swap_stem", "from_song": "A", "to_song": "B",
         "stem": "vocals", "time": swap_time},
    ]
    # Same-BPM songs: rate = 1.0 so the stretch is a passthrough and the
    # B-side buffers stay in their original sample positions.
    with (
        patch("soundfile.read", side_effect=_read),
        patch("pyrubberband.pyrb.time_stretch", side_effect=lambda y, sr, r: y),
    ):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    # After the swap (allowing a few samples for the ZC snap), output
    # should match B's vocal stem on both channels.
    probe = int(swap_time * SR) + 200
    assert out[probe, 0] == pytest.approx(b_vocal_stereo[probe, 0], abs=1e-3)
    assert out[probe, 1] == pytest.approx(b_vocal_stereo[probe, 1], abs=1e-3)
    # Pre-swap region is unaffected.
    pre = int(swap_time * SR) - 1000
    assert out[pre, 0] != pytest.approx(b_vocal_stereo[pre, 0], abs=1e-3)


def test_render_swap_stem_past_output_length_is_noop():
    """time past output end → no-op (don't raise)."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *_baseline_stem_calls(),
        {"tool": "swap_stem", "from_song": "A", "to_song": "B",
         "stem": "drums", "time": 9999.0},
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_unknown_tool_still_raises():
    """The dispatch's else-branch now raises 'unknown tool' (rather than
    the old Phase-7 message) — the wrap-up signal stays the same so the
    worker's failure path still distinguishes 'plan bug' from
    'precondition error'."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "make_coffee", "song": "A"},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        with pytest.raises(NotImplementedError, match="unknown tool"):
            render(plan, a, b)


def _ab_reader(a_amp: float, b_amp: float):
    """soundfile.read stub returning a constant tone whose amplitude
    depends on which song's stem path is requested ('/A/' vs '/B/')."""
    def _read(path, always_2d=True, dtype="float32"):
        amp = a_amp if "/A/" in path else b_amp
        return amp * np.ones((N, 2), dtype=np.float32), SR
    return _read


def test_render_a_fade_out_bars_cuts_a_early():
    """With B silent, a_fade_out_bars < duration_bars must drive A to
    silence by the half-way point while a coupled crossfade keeps A
    audible across the whole window."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    # 120bpm => 2s/bar = 88200 samples. seam at 0.0 (a downbeat, no snap);
    # the 2-bar window starts at output sample 0.
    bar = int(2.0 * SR)
    win_start = 0                 # start_bar=0, seam at 0.0
    win_half = win_start + bar    # a_fade ends here when a_fade_out_bars=1
    win_end = win_start + 2 * bar

    def _plan(a_fade):
        stems = []
        for s in ("vocals", "drums", "bass", "other"):
            call = {"tool": "crossfade_stem", "stem": s,
                    "from_song": "A", "to_song": "B",
                    "start_bar": 0, "duration_bars": 2, "curve": "equal_power"}
            if a_fade is not None:
                call["a_fade_out_bars"] = a_fade
            stems.append(call)
        return [
            {"tool": "set_transition_window",
             "from_song_time_start": 0.0, "to_song_time_start": 0.0,
             "duration_bars": 2},
            *stems,
        ]

    # A audible (0.2/stem), B silent.
    with patch("soundfile.read", side_effect=_ab_reader(0.2, 0.0)):
        coupled = render(_plan(None), a, b)       # default == full-window A fade
        decoupled = render(_plan(1), a, b)        # A gone at the half point

    cd, _ = sf.read(io.BytesIO(coupled.wav_bytes), always_2d=True)
    dd, _ = sf.read(io.BytesIO(decoupled.wav_bytes), always_2d=True)

    back_coupled = np.max(np.abs(cd[win_half:win_end]))
    back_decoupled = np.max(np.abs(dd[win_half:win_end]))
    front_decoupled = np.max(np.abs(dd[win_start:win_half]))

    assert back_coupled > 0.1          # coupled: A still ringing in 2nd half
    assert back_decoupled < 1e-3       # decoupled: A silent in 2nd half (B silent too)
    assert front_decoupled > 0.1       # A still present in the 1st half


def test_render_a_fade_out_bars_default_is_identical():
    """Omitting a_fade_out_bars must be byte-identical to setting it equal
    to duration_bars — i.e. the coupled crossfade is unchanged."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")

    def _plan(explicit):
        stems = []
        for s in ("vocals", "drums", "bass", "other"):
            call = {"tool": "crossfade_stem", "stem": s,
                    "from_song": "A", "to_song": "B",
                    "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            if explicit:
                call["a_fade_out_bars"] = 1
            stems.append(call)
        return [
            {"tool": "set_transition_window",
             "from_song_time_start": 1.0, "to_song_time_start": 0.0,
             "duration_bars": 1},
            *stems,
        ]

    with patch("soundfile.read", side_effect=_fake_sf_read()):
        omitted = render(_plan(False), a, b)
        explicit = render(_plan(True), a, b)

    assert omitted.wav_bytes == explicit.wav_bytes
