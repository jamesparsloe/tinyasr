import torch
import torch.nn as nn
import torchaudio
from torch import Tensor


class WhisperMelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        norm: str = "slaney",
        mel_scale: str = "slaney",
    ):
        super().__init__()

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_stft=self.n_fft // 2 + 1,
            norm=norm,
            mel_scale=mel_scale,
        )

        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, waveform: Tensor):
        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self.mel_scale(magnitudes)

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
