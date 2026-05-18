from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model

logger = logging.getLogger(__name__)


# htdemucs_ft is the locked V1 choice (spec → Locked Decisions Summary).
# 4 stems, fine-tuned variant — produces these source labels in this order.
STEM_NAMES = ("drums", "bass", "other", "vocals")


@dataclass
class SeparationResult:
    """In-memory output of Demucs separation.

    `stems` maps each canonical stem name to a (channels, samples) tensor
    at the original audio's sample rate. `vocal_rms` is the linear RMS of
    the vocal stem (averaged across channels) and is what Phase 6 will
    read to decide whether to skip Whisper.
    """

    sample_rate: int
    stems: dict[str, torch.Tensor]
    vocal_rms: float


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class StemSeparationService:
    """Wraps Demucs htdemucs_ft for 4-stem source separation.

    Pure function: no DB, no storage I/O. Loads the model lazily so cold
    Celery workers don't pay the ~300 MB weight download on import.
    """

    def __init__(self, model_name: str = "htdemucs_ft") -> None:
        self.model_name = model_name
        self._model = None
        self._device = _select_device()

    def _load_model(self):
        if self._model is None:
            logger.info(
                "loading demucs model %s onto %s", self.model_name, self._device
            )
            model = get_model(self.model_name)
            model.to(self._device)
            model.eval()
            self._model = model
        return self._model

    def separate(self, audio_path: Path) -> SeparationResult:
        model = self._load_model()

        # soundfile gives (samples, channels); demucs convert_audio wants
        # (channels, samples). torchaudio.load would work but torchaudio
        # 2.11+ routes through TorchCodec which isn't on the dep list.
        audio, sample_rate = sf.read(str(audio_path), always_2d=True)
        wav = torch.from_numpy(audio.T).float()
        wav = convert_audio(wav, sample_rate, model.samplerate, model.audio_channels)

        # apply_model wants a (batch, channels, samples) tensor and returns
        # (batch, sources, channels, samples). split=True chunks long inputs
        # to keep peak memory bounded; overlap=0.25 is the demucs default.
        with torch.no_grad():
            sources = apply_model(
                model,
                wav[None].to(self._device),
                split=True,
                overlap=0.25,
                progress=False,
            )[0]

        sources = sources.cpu()

        # model.sources is the canonical ordering; STEM_NAMES is our wire
        # ordering. Build a dict to make the caller insensitive to that.
        stems = {name: sources[model.sources.index(name)] for name in STEM_NAMES}

        vocal_rms = float(torch.sqrt(torch.mean(stems["vocals"] ** 2)).item())

        return SeparationResult(
            sample_rate=model.samplerate,
            stems=stems,
            vocal_rms=vocal_rms,
        )

    @staticmethod
    def write_stem(tensor: torch.Tensor, sample_rate: int, dest: Path) -> None:
        """Persist a single stem tensor as a 16-bit PCM WAV.

        Demucs gives us (channels, samples); soundfile wants (samples,
        channels). torchaudio.save in 2.11+ requires TorchCodec, which we
        don't want to pull in just to write WAVs — soundfile is already
        on the dep list via librosa.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        audio = tensor.detach().cpu().numpy().T  # (samples, channels)
        sf.write(str(dest), audio, sample_rate, subtype="PCM_16")
