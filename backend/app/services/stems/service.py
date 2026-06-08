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


# htdemucs (single-model, not the _ft 4-bag ensemble) — same 4-stem
# output, ~4× faster on MPS since it runs one forward pass instead of
# averaging four checkpoints. Quality loss is inaudible for DJ mixing.
STEM_NAMES = ("drums", "bass", "other", "vocals")


# Frame rate for the vocal envelope sidecar. 10 Hz = 100ms per frame, which
# is finer than any DJ-relevant decision needs (transition windows are
# beat-sized, ~0.4–1.0s at typical tempos) but cheap to store: a 4-minute
# song yields ~2400 floats per channel ≈ 20 KB of JSON. Bumping this
# would buy nothing the LLM mix-planner can use.
VOCAL_ENVELOPE_FRAME_HZ = 10


@dataclass
class SeparationResult:
    """In-memory output of Demucs separation.

    `stems` maps each canonical stem name to a (channels, samples) tensor
    at the original audio's sample rate. `vocal_rms` is the linear RMS of
    the vocal stem (averaged across channels) and is what Phase 6 reads
    to decide whether to skip Whisper. `vocal_envelope` is the frame-wise
    RMS+peak sidecar payload, computed here while the vocal tensor is
    still in memory so the Vocal Safety service never re-loads the WAV.
    """

    sample_rate: int
    stems: dict[str, torch.Tensor]
    vocal_rms: float
    vocal_envelope: dict


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_vocal_envelope(
    vocals: torch.Tensor, sample_rate: int, frame_hz: int
) -> dict:
    """Frame-wise RMS + peak envelope of the vocal stem.

    Input is the (channels, samples) vocal tensor demucs hands back. We
    average to mono first — the LLM never cares about stereo width when
    deciding "is the vocal hot here" — then fold into non-overlapping
    `sample_rate // frame_hz`-sample frames and reduce each frame to
    (rms, peak). Trailing samples shorter than one frame are dropped:
    they can't carry meaningful energy at this resolution and keeping
    them would just leave a single noisy tail frame.

    Output shape matches the sidecar schema:
        {"frame_hz": int, "rms": [float, ...], "peak": [float, ...]}
    rms and peak have the same length. Floats are unrounded so a future
    reader can apply its own quantisation policy.
    """
    if vocals.ndim != 2:
        raise ValueError(
            f"expected (channels, samples) tensor, got shape {tuple(vocals.shape)}"
        )

    mono = vocals.mean(dim=0)
    frame_size = max(1, sample_rate // frame_hz)
    n_frames = mono.shape[0] // frame_size
    if n_frames == 0:
        return {"frame_hz": frame_hz, "rms": [], "peak": []}

    trimmed = mono[: n_frames * frame_size]
    frames = trimmed.view(n_frames, frame_size)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=1))
    peak = frames.abs().amax(dim=1)
    return {
        "frame_hz": frame_hz,
        "rms": rms.tolist(),
        "peak": peak.tolist(),
    }


class StemSeparationService:
    """Wraps Demucs htdemucs for 4-stem source separation.

    Pure function: no DB, no storage I/O. Loads the model lazily so cold
    Celery workers don't pay the ~80 MB weight download on import.
    """

    def __init__(self, model_name: str = "htdemucs") -> None:
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

    def _apply_with_oom_fallback(
        self, model, wav: torch.Tensor
    ) -> torch.Tensor:
        """Call demucs apply_model with a graceful OOM fallback chain.

        Tries in order:
          1. Original device (MPS / CUDA) with shifts=2  — best quality.
          2. Original device with shifts=0               — saves ~half the
             peak working set (no shifted-output accumulation).
          3. CPU with shifts=0                           — slow but unlimited
             memory; the safety net for very long songs.

        Returns the (sources, channels, samples) tensor from apply_model.

        Why this exists: MPS heap caps around 9 GiB on M-series. With
        shifts=2 the accumulated shifted_out tensor roughly doubles peak
        memory during apply_model. Long songs (~10+ minutes) can push past
        the cap. Rather than fail the whole separation, we degrade
        gracefully — first by dropping the quality bump, then by giving
        up on GPU acceleration entirely.
        """
        def _attempt(device: torch.device, shifts: int) -> torch.Tensor:
            if device.type == "mps":
                torch.mps.empty_cache()
            with torch.no_grad():
                return apply_model(
                    model,
                    wav[None].to(device),
                    split=True,
                    overlap=0.25,
                    shifts=shifts,
                    progress=False,
                )[0]

        try:
            return _attempt(self._device, shifts=2)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                "stem separation OOM at shifts=2 on %s; retrying with "
                "shifts=0 (vocal-bleed artifacts may be more audible)",
                self._device,
            )

        try:
            return _attempt(self._device, shifts=0)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                "stem separation OOM at shifts=0 on %s; falling back to "
                "CPU (this will be ~5-10× slower)",
                self._device,
            )

        # CPU fallback. nn.Module.to() is in-place, so we move the cached
        # model to CPU for this run, then restore to the original device so
        # subsequent (smaller) songs still get GPU acceleration.
        model.to("cpu")
        if self._device.type == "mps":
            torch.mps.empty_cache()
        try:
            return _attempt(torch.device("cpu"), shifts=0)
        finally:
            model.to(self._device)

    def separate(self, audio_path: Path) -> SeparationResult:
        model = self._load_model()

        # soundfile gives (samples, channels); demucs convert_audio wants
        # (channels, samples). torchaudio.load would work but torchaudio
        # 2.11+ routes through TorchCodec which isn't on the dep list.
        audio, sample_rate = sf.read(str(audio_path), always_2d=True)
        wav = torch.from_numpy(audio.T).float()
        wav = convert_audio(wav, sample_rate, model.samplerate, model.audio_channels)

        sources = self._apply_with_oom_fallback(model, wav).cpu()

        # model.sources is the canonical ordering; STEM_NAMES is our wire
        # ordering. Build a dict to make the caller insensitive to that.
        stems = {name: sources[model.sources.index(name)] for name in STEM_NAMES}

        vocal_rms = float(torch.sqrt(torch.mean(stems["vocals"] ** 2)).item())
        vocal_envelope = _compute_vocal_envelope(
            stems["vocals"], model.samplerate, VOCAL_ENVELOPE_FRAME_HZ
        )

        return SeparationResult(
            sample_rate=model.samplerate,
            stems=stems,
            vocal_rms=vocal_rms,
            vocal_envelope=vocal_envelope,
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
