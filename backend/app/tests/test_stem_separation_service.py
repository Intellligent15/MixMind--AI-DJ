"""Unit tests for StemSeparationService.

We don't actually load demucs weights — that downloads ~300 MB. The two
things the service-layer logic owns are (a) mapping demucs's source order
into our canonical STEM_NAMES dict, and (b) computing vocal_rms from the
vocal stem tensor. Both are testable against a stubbed model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from app.services.stems import STEM_NAMES, StemSeparationService
from app.services.stems.service import (
    VOCAL_ENVELOPE_FRAME_HZ,
    _compute_vocal_envelope,
)


def _make_stub_model(sample_rate: int = 44100, channels: int = 2):
    """Return a fake demucs model + the (sources, channels, samples) tensor
    that apply_model should hand back, ready to wire into patches."""
    # demucs canonical 4-stem order — this is what the service should
    # re-map into our STEM_NAMES dict.
    sources = ["drums", "bass", "other", "vocals"]
    model = MagicMock()
    model.samplerate = sample_rate
    model.audio_channels = channels
    model.sources = sources
    # apply_model returns (batch, sources, channels, samples); we'll have
    # the patch return [0]-indexed (sources, channels, samples).
    return model, sources


def test_separate_remaps_demucs_order_into_stem_names_dict():
    model, sources = _make_stub_model()
    # Each stem gets a distinguishable constant tensor so we can verify
    # the remapping by inspection.
    stems_tensor = torch.stack(
        [torch.full((2, 100), fill_value=float(i)) for i, _ in enumerate(sources)]
    )

    svc = StemSeparationService()
    with (
        patch("app.services.stems.service.get_model", return_value=model),
        patch(
            "app.services.stems.service.sf.read",
            return_value=(np.zeros((100, 2), dtype=np.float32), 44100),
        ),
        patch(
            "app.services.stems.service.convert_audio",
            side_effect=lambda wav, sr, msr, ch: wav,
        ),
        patch(
            "app.services.stems.service.apply_model",
            return_value=stems_tensor.unsqueeze(0),
        ),
    ):
        result = svc.separate(Path("/fake/audio.wav"))

    assert set(result.stems) == set(STEM_NAMES)
    # demucs index of "vocals" is 3 — its constant fill_value is 3.0.
    assert torch.equal(result.stems["vocals"], torch.full((2, 100), 3.0))
    # drums is index 0.
    assert torch.equal(result.stems["drums"], torch.full((2, 100), 0.0))
    assert result.sample_rate == 44100


def test_separate_computes_vocal_rms():
    model, _ = _make_stub_model()
    # Vocals = constant 0.5 amplitude — RMS = 0.5. Other stems irrelevant.
    sources_tensor = torch.zeros(4, 2, 1000)
    sources_tensor[3] = 0.5
    svc = StemSeparationService()
    with (
        patch("app.services.stems.service.get_model", return_value=model),
        patch(
            "app.services.stems.service.sf.read",
            return_value=(np.zeros((1000, 2), dtype=np.float32), 44100),
        ),
        patch(
            "app.services.stems.service.convert_audio",
            side_effect=lambda wav, sr, msr, ch: wav,
        ),
        patch(
            "app.services.stems.service.apply_model",
            return_value=sources_tensor.unsqueeze(0),
        ),
    ):
        result = svc.separate(Path("/fake/audio.wav"))

    assert abs(result.vocal_rms - 0.5) < 1e-6


def test_separate_silent_vocals_has_zero_rms():
    model, _ = _make_stub_model()
    sources_tensor = torch.zeros(4, 2, 500)
    svc = StemSeparationService()
    with (
        patch("app.services.stems.service.get_model", return_value=model),
        patch(
            "app.services.stems.service.sf.read",
            return_value=(np.zeros((500, 2), dtype=np.float32), 44100),
        ),
        patch(
            "app.services.stems.service.convert_audio",
            side_effect=lambda wav, sr, msr, ch: wav,
        ),
        patch(
            "app.services.stems.service.apply_model",
            return_value=sources_tensor.unsqueeze(0),
        ),
    ):
        result = svc.separate(Path("/fake/audio.wav"))

    assert result.vocal_rms == 0.0


def test_separate_emits_vocal_envelope_shape():
    """End-to-end: separate() returns an envelope with matching rms/peak lengths
    and a frame_hz tag at the module-level rate."""
    model, _ = _make_stub_model(sample_rate=4410)  # 4410/10 = 441-sample frames
    # 1.2 s of audio at 4410 Hz = 5292 samples → 12 full frames.
    n_samples = 5292
    sources_tensor = torch.zeros(4, 2, n_samples)
    sources_tensor[3] = 0.1  # vocals constant 0.1 → rms == peak == 0.1
    svc = StemSeparationService()
    with (
        patch("app.services.stems.service.get_model", return_value=model),
        patch(
            "app.services.stems.service.sf.read",
            return_value=(np.zeros((n_samples, 2), dtype=np.float32), 4410),
        ),
        patch(
            "app.services.stems.service.convert_audio",
            side_effect=lambda wav, sr, msr, ch: wav,
        ),
        patch(
            "app.services.stems.service.apply_model",
            return_value=sources_tensor.unsqueeze(0),
        ),
    ):
        result = svc.separate(Path("/fake/audio.wav"))

    env = result.vocal_envelope
    assert env["frame_hz"] == VOCAL_ENVELOPE_FRAME_HZ
    assert len(env["rms"]) == len(env["peak"]) == 12
    # Constant 0.1 amplitude → rms == peak == 0.1 per frame.
    assert all(abs(v - 0.1) < 1e-6 for v in env["rms"])
    assert all(abs(v - 0.1) < 1e-6 for v in env["peak"])


def test_compute_vocal_envelope_drops_trailing_partial_frame():
    sample_rate = 100
    frame_hz = 10  # frame_size = 10 samples
    # 25 samples → 2 full frames, 5-sample tail that should be dropped.
    vocals = torch.ones(2, 25) * 0.5
    env = _compute_vocal_envelope(vocals, sample_rate, frame_hz)
    assert env["frame_hz"] == 10
    assert len(env["rms"]) == 2
    assert len(env["peak"]) == 2


def test_compute_vocal_envelope_empty_when_shorter_than_one_frame():
    # 5 samples at sample_rate=100, frame_hz=10 → frame_size=10 → no full frames.
    env = _compute_vocal_envelope(torch.ones(1, 5), 100, 10)
    assert env == {"frame_hz": 10, "rms": [], "peak": []}


def test_write_stem_persists_wav(tmp_path: Path):
    tensor = torch.zeros(2, 1000)
    dest = tmp_path / "nested" / "vocals.wav"
    StemSeparationService.write_stem(tensor, 44100, dest)
    assert dest.exists()
    assert dest.stat().st_size > 0
