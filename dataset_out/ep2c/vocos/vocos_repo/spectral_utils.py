## spectral_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import scipy.fftpack as fftpack

class SpectralUtils:
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        window_type: str = 'hann',
        spectral_representation: str = 'stft',  # or 'mdct' if extended
        phase_representation: str = 'phase_logits'  # or 'sign_logmag', etc.
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window_type = window_type
        self.spectral_representation = spectral_representation
        self.phase_representation = phase_representation
        self.window = self._get_window()
        # Generate mel filter bank once
        self.mel_filterbank = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=self.sample_rate / 2
        )
        # Small epsilon for numerical stability
        self.eps = 1e-7

    def _get_window(self):
        if self.window_type == 'hann':
            return torch.hann_window(self.n_fft, periodic=True)
        else:
            # Default to hann
            return torch.hann_window(self.n_fft, periodic=True)

    def compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram from waveform.
        Args:
            waveform: (batch, samples) or (samples,)
        Returns:
            mel_spec: (batch, n_mels, time_frames)
        """
        # Ensure batch
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        elif waveform.dim() == 2:
            pass
        else:
            raise ValueError("Waveform tensor must be 1D or 2D.")

        # Compute STFT
        # Use librosa's stft or torchaudio's functions
        # Using torch's stft
        stft_real, stft_imag = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(waveform.device),
            return_complex=False
        )  # shape: (batch, freq, time, 2)
        # Convert to complex tensor
        complex_stft = torch.complex(stft_real, stft_imag)

        # Compute magnitude spectrogram
        mag = torch.abs(complex_stft)  # shape: (batch, freq, time)

        # Convert to power spectrogram, then mel
        power_spec = mag ** 2  # (batch, freq, time)
        # Apply mel filterbank
        # mel_filterbank shape: (n_mels, freq)
        mel_spec = torch.einsum('mf,bftf->bmt', self.mel_filterbank, mag)
        # Convert to log scale for stability
        mel_spec = torch.log1p(torch.clamp(mel_spec, min=self.eps))
        return mel_spec

    def log_mag(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Convert magnitude spectrum to log scale.
        Args:
            magnitude: (batch, freq, time)
        Returns:
            log_magnitude: same shape
        """
        return torch.log1p(torch.clamp(magnitude, min=self.eps))

    def spectral_coeffs_to_complex(self, m_log: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Convert network outputs (m_log and p) into complex spectrogram.
        Args:
            m_log: tensor of shape (batch, freq, time) - log magnitude logits
            p: tensor of shape (batch, freq, time) - phase logits or parameters
        Returns:
            complex_spec: (batch, freq, time) complex tensor
        """
        # Convert magnitude logits to magnitude
        M = torch.exp(m_log)  # shape: (batch, freq, time)

        # Derive phase angles depending on representation
        if self.phase_representation == 'phase_logits':
            # p are logits that represent phase parameters
            # Map p to (-pi, pi): assuming p passed through tanh scaled by pi
            phase_angles = torch.atan2(torch.sin(p), torch.cos(p))
        elif self.phase_representation == 'phase_params':
            # p directly represents phase angles
            phase_angles = p
        elif self.phase_representation == 'sign_logmag':
            # p encode sign and magnitude alternatively (not used here)
            # For simplicity, assume p are phase logits
            phase_angles = torch.atan2(torch.sin(p), torch.cos(p))
        else:
            # Default fallback
            phase_angles = torch.atan2(torch.sin(p), torch.cos(p))
        
        # Wrap phase into (-pi, pi], atan2 does this inherently
        # Compute real and imaginary parts
        real = M * torch.cos(phase_angles)
        imag = M * torch.sin(phase_angles)
        complex_spec = torch.complex(real, imag)
        return complex_spec

    def derivative_phase(self, p: torch.Tensor) -> torch.Tensor:
        """
        Derive phase angle from phase parameters p.
        Args:
            p: tensor of shape (batch, freq, time)
        Returns:
            phase: tensor wrapped in (-pi, pi]
        """
        phase = torch.atan2(torch.sin(p), torch.cos(p))
        return phase

    def inverse_spectrogram(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from complex spectrogram using ISTFT.
        Args:
            complex_spec: (batch, freq, time)
        Returns:
            waveform: (batch, samples)
        """
        # complex_spec: (batch, freq, time)
        # Use torch.istft
        # Parameters must match analysis
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(complex_spec.device),
            center=True,
            length=None,  # optional: specify if known
        )  # shape: (batch, samples)
        return waveform

    def symlog(self, x: torch.Tensor) -> torch.Tensor:
        """
        Symmetric log compression.
        Args:
            x: tensor
        Returns:
            compressed tensor
        """
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
    def symexp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of symlog
        Args:
            x: tensor
        Returns:
            exponentiated tensor
        """
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def prepare_spectral_outputs(self, m_log: torch.Tensor, p: torch.Tensor):
        """
        Utility to produce complex spectrogram from raw network outputs.
        Args:
            m_log: magnitude logits
            p: phase logits or parameters
        Returns:
            complex_spec: (batch, freq, time)
        """
        phase_angles = self.derivative_phase(p)
        return self.spectral_coeffs_to_complex(m_log, phase_angles)

# Example usage (not to be included directly in the module):
# spectral_processor = SpectralUtils()
# mel_spec = spectral_processor.compute_mel_spectrogram(waveform)
# m_log = some_network_output_magnitude_logits
# p = some_network_output_phase_params
# complex_spec = spectral_processor.prepare_spectral_outputs(m_log, p)
# waveform_recon = spectral_processor.inverse_spectrogram(complex_spec)
