"""Unit tests for StemSeparationService.

We don't actually load demucs weights — that downloads ~300 MB. The two
things the service-layer logic owns are (a) mapping demucs's source order
into our canonical STEM_NAMES dict, and (b) computing vocal_rms from the
vocal stem tensor. Both are testable against a stubbed model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from app.services.stems import STEM_NAMES, StemSeparationService


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
            "app.services.stems.service.torchaudio.load",
            return_value=(torch.zeros(2, 100), 44100),
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
            "app.services.stems.service.torchaudio.load",
            return_value=(torch.zeros(2, 1000), 44100),
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
            "app.services.stems.service.torchaudio.load",
            return_value=(torch.zeros(2, 500), 44100),
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


def test_write_stem_persists_wav(tmp_path: Path):
    tensor = torch.zeros(2, 1000)
    dest = tmp_path / "nested" / "vocals.wav"
    StemSeparationService.write_stem(tensor, 44100, dest)
    assert dest.exists()
    assert dest.stat().st_size > 0
