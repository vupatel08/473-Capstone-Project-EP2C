# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import random
from typing import List, Tuple, Optional

import torch
import torchaudio
import librosa

import spectral_utils  # ensure spectral_utils.py is in your project directory

class DatasetLoader:
    def __init__(self, config: dict):
        """
        Initializes the dataset loader with dataset paths and parameters.

        Args:
            config (dict): Configuration dictionary following 'config.yaml' structure.
        """
        # Extract dataset configuration
        dataset_cfg = config.get('dataset', {})
        self.sample_rate: int = dataset_cfg.get('sample_rate', 24000)
        self.segment_size: int = dataset_cfg.get('segment_size', 16384)

        # Mel spectrogram parameters
        mel_params = dataset_cfg.get('mel_params', {})
        self.n_fft: int = mel_params.get('n_fft', 1024)
        self.hop_length: int = mel_params.get('hop_length', 256)
        self.n_mels: int = mel_params.get('n_mels', 100)
        self.window_type: str = 'hann'  # assuming Hann window
        self.window: torch.Tensor = torch.hann_window(self.n_fft)

        # Data augmentation
        augmentation_cfg = dataset_cfg.get('data_augmentation', {})
        gain_range = augmentation_cfg.get('random_gain_db', [-6, -1])
        self.random_gain_db: Tuple[float, float] = (gain_range[0], gain_range[1])

        # List of audio files
        self.file_paths: List[str] = []
        dataset_dir: str = dataset_cfg.get('dataset_dir', '')
        self._load_file_list(dataset_dir)

        # Internal pointer for batching
        self._current_idx: int = 0
        self._shuffle_dataset()

        # For reproducibility
        self._rng = random.Random()

    def _load_file_list(self, dataset_dir: str) -> None:
        """
        Populates self.file_paths with paths to all audio files found in dataset_dir.
        """
        supported_exts = ['.wav', '.flac', '.mp3', '.m4a', '.ogg']
        for root, _, files in os.walk(dataset_dir):
            for fname in files:
                if any(fname.lower().endswith(ext) for ext in supported_exts):
                    self.file_paths.append(os.path.join(root, fname))
        self._dataset_size = len(self.file_paths)

    def _shuffle_dataset(self) -> None:
        """
        Shuffles the dataset file list.
        """
        self._rng.shuffle(self.file_paths)
        self._current_idx = 0

    def _load_audio(self, filepath: str) -> torch.Tensor:
        """
        Loads an audio waveform at the target sample rate.

        Args:
            filepath (str): Path to audio file.
        Returns:
            waveform (Tensor): 1D tensor of shape (samples,)
        """
        waveform, sr = torchaudio.load(filepath)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # shape: (samples,)
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform

    def _get_segment(self, waveform: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Crops or pads the waveform to the fixed segment size. Applies augmentation if train=True.

        Args:
            waveform (Tensor): 1D tensor
            train (bool): whether in training mode
        Returns:
            segment (Tensor): 1D tensor of size self.segment_size
        """
        length = waveform.shape[0]

        if length >= self.segment_size:
            max_start = length - self.segment_size
            start_idx = self._rng.randint(0, max_start) if train else 0
            segment = waveform[start_idx:start_idx + self.segment_size]
        else:
            # pad with zeros if shorter
            pad_amount = self.segment_size - length
            segment = torch.cat([waveform, torch.zeros(pad_amount, device=waveform.device)])

        # Data augmentation: random gain
        if train and self.random_gain_db is not None:
            gain_db = self._rng.uniform(self.random_gain_db[0], self.random_gain_db[1])
            gain = 10 ** (gain_db / 20)
            segment = segment * gain

        return segment

    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Computes mel spectrogram for the waveform.

        Args:
            waveform (Tensor): 1D tensor
        Returns:
            mel_spec (Tensor): shape (n_mels, frames)
        """
        mel_spec = spectral_utils.compute_mel_spectrogram(
            waveform=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window=self.window
        )
        return mel_spec

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches a sample: mel spectrogram and waveform segment.

        Args:
            index (int): sample index
        Returns:
            mel_spec (Tensor): (n_mels, frames)
            waveform (Tensor): (segment_size,)
        """
        # Wrap index if out-of-bounds
        idx = index % self._dataset_size
        filepath = self.file_paths[idx]
        waveform = self._load_audio(filepath)
        segment = self._get_segment(waveform, train=True)

        mel_spec = self._compute_mel_spectrogram(segment)

        return mel_spec, segment

    def __len__(self) -> int:
        return self._dataset_size

    def get_batch(self, batch_size: int, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a batch of data samples for training.

        Args:
            batch_size (int): size of batch
            train (bool): whether to apply data augmentation
        Returns:
            batch_mel (Tensor): (batch_size, n_mels, frames)
            batch_wave (Tensor): (batch_size, segment_size)
        """
        mel_list = []
        wave_list = []
        for _ in range(batch_size):
            mel, wave = self.__getitem__(self._current_idx)
            mel_list.append(mel)
            wave_list.append(wave)
            self._current_idx += 1
            if self._current_idx >= self._dataset_size:
                self._shuffle_dataset()

        batch_mel = torch.stack(mel_list, dim=0)  # (batch, n_mels, frames)
        batch_wave = torch.stack(wave_list, dim=0)  # (batch, segment_size)
        return batch_mel, batch_wave
```

## discriminator.py

```python
## discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD)
    Designed to analyze periodic structures in waveform at specified periods.
    """
    def __init__(self, periods=None, in_channels=1, base_channels=32, max_channels=256):
        """
        Args:
            periods (list[int]): List of periods to analyze, e.g., [2,3,5,7,11]
            in_channels (int): Number of input channels, typically 1 (mono audio)
            base_channels (int): Base number of channels for convolutions
            max_channels (int): Max channels to cap
        """
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.periods = periods
        self.discriminators = nn.ModuleList()
        for p in self.periods:
            self.discriminators.append(_PeriodDiscriminator(p, in_channels, base_channels, max_channels))
    
    def forward(self, waveform):
        """
        Args:
            waveform (Tensor): shape [B, 1, T]
        Returns:
            list of scores and feature maps from each period discriminator
        """
        feat_scores = []
        feat_maps = []
        for disc in self.discriminators:
            features, score = disc(waveform)
            feat_scores.append(score)
            feat_maps.append(features)
        return feat_scores, feat_maps

class _PeriodDiscriminator(nn.Module):
    """
    Discriminator analyzing waveform at a specific period.
    """
    def __init__(self, period, in_channels=1, base_channels=32, max_channels=256):
        super().__init__()
        self.period = period
        self.conv_layers = nn.ModuleList()

        # Convolutional layers with spectral normalization
        chs = [in_channels, 64, 128, 256, 512]
        chs = [min(c, max_channels) for c in chs]
        for i in range(len(chs) - 1):
            self.conv_layers.append(
                spectral_norm(nn.Conv1d(
                    chs[i],
                    chs[i+1],
                    kernel_size=5,
                    stride=2,
                    padding=2
                ))
            )
        self.final_conv = spectral_norm(nn.Conv1d(
            chs[-1],
            1,
            kernel_size=3,
            stride=1,
            padding=1
        ))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, waveform):
        """
        Args:
            waveform (Tensor): shape [B, 1, T]
        Returns:
            feature maps (list of Tensor), score (Tensor)
        """
        B, C, T = waveform.shape
        # Pad waveform to ensure divisibility by period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            waveform = F.pad(waveform, (0, pad_len))
        # Reshape to (B, C, T // period, period)
        T_new = waveform.shape[2]
        waveform_periodic = waveform.view(B, C, T_new // self.period, self.period)
        # Collapse last dimension: concatenate periodically
        waveform_periodic = waveform_periodic.reshape(B, C, T_new)
        features = []
        x = waveform_periodic
        for layer in self.conv_layers:
            x = layer(x)
            x = self.leaky_relu(x)
            features.append(x)
        score = self.final_conv(x)
        # Aggregate feature maps into list
        return features, score

class MultiResolutionDiscriminator(nn.Module):
    """
    Multi-Resolution Discriminator (MRD)
    Analyzes spectral features at multiple resolutions
    """
    def __init__(self, resolutions=None, in_channels=1, base_channels=32, max_channels=256):
        """
        Args:
            resolutions (list[int]): List of different spectral resolutions or kernel sizes.
            For simplicity, here we treat resolutions as different kernel sizes or downsampling configs.
        """
        super().__init__()
        if resolutions is None:
            resolutions = [512, 1024, 2048]
        self.resolutions = resolutions
        self.discriminators = nn.ModuleList()
        for r in resolutions:
            self.discriminators.append(_MultiResDiscriminator(r, in_channels, base_channels, max_channels))
    
    def forward(self, waveform):
        """
        Args:
            waveform (Tensor): shape [B, 1, T]
        Returns:
            list of scores and feature maps from each resolution discriminator
        """
        feat_scores = []
        feat_maps = []
        for disc in self.discriminators:
            features, score = disc(waveform)
            feat_scores.append(score)
            feat_maps.append(features)
        return feat_scores, feat_maps

class _MultiResDiscriminator(nn.Module):
    """
    Discriminator analyzing waveform at specific spectral resolution
    """
    def __init__(self, resolution, in_channels=1, base_channels=32, max_channels=256):
        super().__init__()
        # Use a custom kernel size or downsampling to produce different resolutions
        self.resolution = resolution
        self.conv_layers = nn.ModuleList()
        chs = [in_channels, 64, 128, 256, 512]
        chs = [min(c, max_channels) for c in chs]
        for i in range(len(chs) - 1):
            self.conv_layers.append(
                spectral_norm(nn.Conv1d(
                    chs[i],
                    chs[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ))
            )
        self.final_conv = spectral_norm(nn.Conv1d(
            chs[-1],
            1,
            kernel_size=3,
            stride=1,
            padding=1
        ))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, waveform):
        """
        Args:
            waveform (Tensor): shape [B, 1, T]
        Returns:
            feature maps list, score tensor
        """
        x = waveform
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            x = self.leaky_relu(x)
            features.append(x)
        score = self.final_conv(x)
        return features, score
```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import yaml

# Import spectral_utils functions (for inverse spectrogram)
from spectral_utils import SpectralUtils

# Assume pre-trained utilities for PESQ, VISQOL, and UTMOS
# For illustration, placeholder implementations or interfaces will be used.
# In practice, replace with actual libraries or APIs.
try:
    import pypesq  # for PESQ
except ImportError:
    pypesq = None

try:
    import visqol  # For VISQOL, hypothetical wrapper
except ImportError:
    visqol = None

# Placeholder class for UTMOS
class UTMOSModel:
    def __init__(self, device='cpu'):
        pass
    def predict(self, waveform: np.ndarray) -> float:
        # Placeholder: returns a dummy MOS score
        return 3.5

# Load config if exists
import sys
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
else:
    # Default minimal configuration
    config = {
        'dataset': {
            'sample_rate': 24000,
            'max_eval_samples': 100,  # Number of samples to evaluate
            'eval_mel_files': [],  # List of mel spectrogram file paths
            'ground_truth_waveforms': []  # List of waveform file paths
        },
        'model': {
            'checkpoint_path': './checkpoints/G_final.pt',  # Path to trained generator
            'fft_size': 1024,
            'hop_length': 256,
            'mel_bins': 100,
            'phase_representation': 'phase_logits'
        },
        'evaluation': {
            'save_dir': './evaluation_outputs'
        }
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize spectral_utils with parameters from config
spectral_utils = SpectralUtils(
    sample_rate=config['dataset'].get('sample_rate', 24000),
    n_fft=config['model'].get('fft_size', 1024),
    hop_length=config['model'].get('hop_length', 256),
    n_mels=config['model'].get('mel_bins', 100)
)

# Define the generator class (imported from model.py, but redefined here for completeness)
from model import SpectralPredictor

# Load trained generator model
net_G = SpectralPredictor({
    'fft_size': config['model'].get('fft_size', 1024),
    'hop_length': config['model'].get('hop_length', 256),
    'mel_bins': config['model'].get('mel_bins', 100),
    'spectral_heads': 2,  # magnitude and phase
    'phase_representation': config['model'].get('phase_representation', 'phase_logits')
})
net_G.load_state_dict(torch.load(config['model']['checkpoint_path'], map_location='cpu'))
net_G.to(device)
net_G.eval()

# Initialize UTMOS predictor (placeholder or replace with actual)
utmos = UTMOSModel(device=device)

# Optionally define PESQ, VISQOL, VAD utilities here
def compute_pesq_ref(ref_waveform, deg_waveform, sr):
    if pypesq is None:
        return np.nan
    try:
        score = pypesq.pesq(sr, ref_waveform, deg_waveform, 'wb')
        return score
    except:
        return np.nan

def compute_visqol(ref_waveform, deg_waveform):
    if visqol is None:
        return np.nan
    try:
        # Placeholder call
        score = visqol.evaluate(ref_waveform, deg_waveform)
        return score
    except:
        return np.nan

def compute_mos_waveform(waveform):
    # Replace with actual UTMOS prediction
    score = utmos.predict(waveform)
    return score

def compute_vuv_f1(gt_waveform, gen_waveform, sr, threshold=0.01):
    # Basic energy-based VAD
    def get_voiced_unvoiced(waveform):
        energy = np.abs(waveform) ** 2
        median_energy = np.median(energy)
        voiced = energy > (median_energy * threshold)
        return voiced

    gt_voiced = get_voiced_unvoiced(gt_waveform)
    gen_voiced = get_voiced_unvoiced(gen_waveform)
    # Compute F1
    from sklearn.metrics import f1_score
    min_len = min(len(gt_voiced), len(gen_voiced))
    f1 = f1_score(gt_voiced[:min_len], gen_voiced[:min_len])
    return f1

# Ensure evaluation output directory exists
os.makedirs(config['evaluation'].get('save_dir', './evaluation_outputs'), exist_ok=True)

# Helper to process a batch of mel spectrograms
def run_inference_and_evaluate(mel_files: List[str], gt_files: List[str]):
    results = {
        'PESQ': [],
        'VISQOL': [],
        'MOS': [],
        'VUV_F1': []
    }
    # Loop through files
    for idx, mel_file in enumerate(mel_files):
        mel_spec = np.load(mel_file)  # assuming mel saved as .npy
        mel_spec_tensor = torch.from_numpy(mel_spec.T).unsqueeze(0).to(device)  # shape: (1, T, n_mels)
        # For safety, ensure shape: (B, T, n_mels)
        # inference
        with torch.no_grad():
            m_logits, p_logits = net_G(mel_spec_tensor)  # output: (B, F, T)
        m_logits = m_logits.squeeze(0).cpu()
        p_logits = p_logits.squeeze(0).cpu()

        # Convert to spectral complex
        complex_spec = spectral_utils.prepare_spectral_outputs(m_logits, p_logits)
        complex_spec = complex_spec.to(torch.device('cpu'))

        # Reconstruct waveform via ISTFT
        waveform_recon = spectral_utils.inverse_spectrogram(complex_spec)
        waveform_recon = waveform_recon.numpy()

        # Save reconstructed waveform
        save_path = os.path.join(config['evaluation'].get('save_dir', './evaluation_outputs'),
                                 f'recon_{idx}.wav')
        wavfile.write(save_path, spectral_utils.sample_rate, waveform_recon.astype(np.float32))

        # Load ground truth if available
        gt_waveform = None
        if gt_files and len(gt_files) > idx:
            gt_waveform, _ = librosa.load(gt_files[idx], sr=spectral_utils.sample_rate)

        # Compute metrics if gt available
        # PESQ
        pesq_score = compute_pesq_ref(gt_waveform, waveform_recon, spect_utils.sample_rate) if gt_waveform is not None else np.nan
        # VISQOL
        visqol_score = compute_visqol(gt_waveform, waveform_recon) if gt_waveform is not None else np.nan
        # UTMOS
        mos_score = compute_mos_waveform(waveform_recon)
        # V/UV F1
        if gt_waveform is not None:
            vuv_f1 = compute_vuv_f1(gt_waveform, waveform_recon, spect_utils.sample_rate)
        else:
            vuv_f1 = np.nan

        results['PESQ'].append(pesq_score)
        results['VISQOL'].append(visqol_score)
        results['MOS'].append(mos_score)
        results['VUV_F1'].append(vuv_f1)

        # Optional: Plot spectrograms
        plt.figure(figsize=(12, 6))
        plt.subplot(2,1,1)
        plt.title('Ground Truth Magnitude Spectrogram')
        if gt_waveform is not None:
            spec_gt = np.abs(librosa.stft(gt_waveform, n_fft=spectral_utils.n_fft, hop_length=spectral_utils.hop_length))
            librosa.display.specshow(librosa.amplitude_to_db(spec_gt, ref=np.max), y_axis='log', x_axis='time')
        plt.subplot(2,1,2)
        plt.title('Reconstructed Magnitude Spectrogram')
        spec_recon = np.abs(librosa.stft(waveform_recon, n_fft=spectral_utils.n_fft, hop_length=spectral_utils.hop_length))
        librosa.display.specshow(librosa.amplitude_to_db(spec_recon, ref=np.max), y_axis='log', x_axis='time')
        plt.tight_layout()
        plt.savefig(os.path.join(config['evaluation'].get('save_dir', './evaluation_outputs'), f'spec_{idx}.png'))
        plt.close()

    # Aggregate results
    def mean_std_list(lst):
        lst_clean = [x for x in lst if not np.isnan(x)]
        if len(lst_clean) == 0:
            return (np.nan, np.nan)
        return np.mean(lst_clean), np.std(lst_clean)

    print('Evaluation Metrics:')
    for key in results:
        mean_val, std_val = mean_std_list(results[key])
        print(f"{key}: {mean_val:.3f} ± {std_val:.3f}")
    # Save overall results
    import json
    with open(os.path.join(config['evaluation'].get('save_dir', './evaluation_outputs'), 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

# Load dataset/mel files
mel_files = config['dataset'].get('eval_mel_files', [])
gt_files = config['dataset'].get('ground_truth_waveforms', [])
max_eval_samples = min(len(mel_files), len(gt_files), config['dataset'].get('max_eval_samples', 100))
mel_files = mel_files[:max_eval_samples]
gt_files = gt_files[:max_eval_samples]

# Run evaluation
run_inference_and_evaluate(mel_files, gt_files)
```

---

**Note:**  
- This code assumes that:
  - The mel spectrograms for evaluation are saved as `.npy` files (with shape [Time, Mel]), which are loaded with `np.load`.
  - Ground truth waveforms are available as file paths.
  - The trained generator checkpoint is at `'checkpoint_path'`.
  - You replace placeholder metric functions with actual implementations as available.
- The inference process uses spectral_utils' `prepare_spectral_outputs` to decode model outputs into complex spectrograms before inverse SFT.
- The spectrogram visualization uses librosa's `specshow` for clarity.
- The script prints overall scores and saves both plots and a JSON summary.
- For large-scale validation, batching two samples or more can be implemented, but here it processes samples one-by-one for clarity.

This implementation aligns tightly with the described design, assuming all modules and functions from the project are properly imported and available.

## main.py

```python
## main.py
import os
import yaml
import torch
import random
import numpy as np

from spectral_utils import SpectralUtils
from dataset_loader import DatasetLoader
from model import SpectralPredictor
from discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from trainer import Trainer
from evaluation import run_inference_and_evaluate

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create necessary directories for checkpoints and eval outputs
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    eval_output_dir = config.get('evaluation', {}).get('save_dir', './evaluation_outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    # Initialize SpectralUtils with parameters from config
    spect_utils = SpectralUtils(
        sample_rate = config['dataset'].get('sample_rate', 24000),
        n_fft = config['model'].get('fft_size', 1024),
        hop_length = config['model'].get('hop_length', 256),
        n_mels = config['dataset']['mel_params'].get('n_mels', 100),
        window_type='hann'
    )

    # Initialize DatasetLoader
    dataset = DatasetLoader(config)

    # Initialize generator (SpectralPredictor)
    gen_params = config['model']
    generator = SpectralPredictor(gen_params).to(device)

    # Initialize discriminators
    D_mpd = MultiPeriodDiscriminator().to(device)
    D_mrd = MultiResolutionDiscriminator().to(device)

    # Setup optimizers
    lr = config['training'].get('learning_rate', 2e-4)
    betas = tuple(config['training'].get('optimizer_betas', [0.9, 0.999]))
    weight_decay = config['training'].get('AdamW_weight_decay', 0.01)

    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    d_optimizer_mpd = torch.optim.AdamW(D_mpd.parameters(), lr=lr, betas=betas)
    d_optimizer_mrd = torch.optim.AdamW(D_mrd.parameters(), lr=lr, betas=betas)

    # Training parameters
    total_iterations = int(config['training'].get('total_iterations', 2000000))
    save_interval = int(config['training'].get('save_interval', 100000))
    log_interval = int(config['training'].get('log_interval', 1000))
    # Loss weights
    lambda_adv = 1.0
    lambda_mel = 1.0
    lambda_feat = 10.0

    # Initialize Trainer
    trainer = Trainer({
        'generator': generator,
        'D_mpd': D_mpd,
        'D_mrd': D_mrd,
        'g_optimizer': g_optimizer,
        'd_optimizer_mpd': d_optimizer_mpd,
        'd_optimizer_mrd': d_optimizer_mrd,
        'spect_utils': spect_utils,
        'dataset': dataset,
        'config': config,
        'checkpoint_dir': checkpoint_dir,
        'total_iterations': total_iterations,
        'save_interval': save_interval,
        'log_interval': log_interval,
        'lambda_adv': lambda_adv,
        'lambda_mel': lambda_mel,
        'lambda_feat': lambda_feat,
        'device': device
    })

    # Check for existing checkpoints (optional, here start fresh)
    # If desired, add code to load latest checkpoint

    # Run training
    trainer.train()

    # After training, run evaluation (optional, here at the end)
    # We assume validation mel spectrograms and GT waveforms are prepared
    run_inference_and_evaluate(
        config.get('evaluation', {}).get('eval_mel_files', []),
        config.get('dataset', {}).get('ground_truth_waveforms', [])
    )

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock1D(nn.Module):
    """A ConvNeXt block adapted for 1D spectral feature processing."""
    def __init__(self, in_channels: int, drop_path: float = 0.0):
        super().__init__()
        self.dw_conv = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.pw_conv1 = nn.Linear(in_channels, 4 * in_channels)
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * in_channels, in_channels)
        self.dropout = nn.Identity()  # Can be replaced with nn.Dropout if needed
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x shape: [B, T, C]
        shortcut = x
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = self.norm(x)
        # Invert the norm dimension to match nn.LayerNorm (which applies on last dim)
        # ConvNeXt standard normalization on the channel dimension
        # but here after permute, last dim is C; so norm applies to that
        x = self.norm(x)
        # MLP
        x_mlp = self.pw_conv1(x)
        x_mlp = self.gelu(x_mlp)
        x_mlp = self.pw_conv2(x_mlp)
        x = shortcut + self.drop_path(x_mlp)
        return x

class DropPath(nn.Module):
    """Stochastic Depth as DropPath."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor + keep_prob
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask

class SpectralHead(nn.Module):
    """Output spectral coefficients (magnitude logits and phase logits)."""
    def __init__(self, in_dim: int, spectral_dim: int, phase_dim: int):
        super().__init__()
        # Final linear layers to produce magnitude logits and phase logits separately
        self.magnitude_head = nn.Linear(in_dim, spectral_dim)
        self.phase_head = nn.Linear(in_dim, phase_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            m_logits: [B, T, spectral_dim]
            p_logits: [B, T, phase_dim]
        """
        m_logits = self.magnitude_head(x)
        p_logits = self.phase_head(x)
        return m_logits, p_logits

class SpectralPredictor(nn.Module):
    """
    Spectral predictor network based on ConvNeXt backbone.
    Converts mel spectrogram features into spectral coefficients (magnitude + phase).
    """
    def __init__(self, config: dict):
        super().__init__()
        # Extract configuration parameters
        self.fft_size = config.get('fft_size', 1024)
        self.hop_length = config.get('hop_length', 256)
        self.n_mels = config.get('mel_bins', 100)
        self.hidden_dim = config.get('hidden_dim', 768)
        self.spectral_heads = config.get('spectral_heads', 2)  # e.g., magnitude and phase
        self.phase_representation = config.get('phase_representation', 'phase_logits')  # or 'sign_logmag'

        # Embedding layer: project mel features to hidden_dim
        self.embedding = nn.Linear(self.n_mels, self.hidden_dim)

        # Decide number of ConvNeXt blocks, e.g., 12 for a deep stack
        self.num_blocks = config.get('num_blocks', 12)

        # Build ConvNeXt blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock1D(self.hidden_dim) for _ in range(self.num_blocks)
        ])

        # Spectral head: output logits for magnitude and phase
        spectral_dim = (self.fft_size // 2) + 1  # 513 for fft_size=1024
        phase_dim = self.spectral_heads  # e.g., 2 for sine/cosine or phase params
        self.head = SpectralHead(self.hidden_dim, spectral_dim, phase_dim)

        # Initialize weak weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel_spec):
        """
        Args:
            mel_spec: [B, T, n_mels]
        Returns:
            m_logits: [B, T, spectral_dim]
            p_logits: [B, T, phase_dim]
        """
        x = self.embedding(mel_spec)  # [B, T, hidden_dim]
        # Pass through ConvNeXt blocks
        for block in self.blocks:
            x = block(x)
        # Output spectral logits
        m_logits, p_logits = self.head(x)
        return m_logits, p_logits

    def get_spectral_coeffs(self, m_logits, p_logits):
        """
        Convert network output logits to complex spectral coefficients.
        Args:
            m_logits: [B, T, spectral_dim]
            p_logits: [B, T, phase_dim]
        Returns:
            complex_spec: [B, T, spectral_dim] complex tensor
        """
        # Convert magnitude logits to magnitude
        M = torch.exp(m_logits)  # ensure positivity

        # Map phase logits to phase angles
        if self.phase_representation == 'phase_logits':
            # Map p_logits to phase angles via atan2(sin, cos)
            # Assume p_logits are raw, could be scaled or passed through tanh
            # For stability, take sin and cos from logits
            phase_sin = torch.sin(p_logits)
            phase_cos = torch.cos(p_logits)
            phi = torch.atan2(phase_sin, phase_cos)  # shape: [B, T, phase_dim]
        elif self.phase_representation == 'phase_params':
            phi = p_logits  # Assuming direct phase angles, shape: [B, T, phase_dim]
        elif self.phase_representation == 'sign_logmag':
            # Not implemented here; placeholder if needed
            phi = p_logits
        else:
            # Default to 'phase_logits'
            phase_sin = torch.sin(p_logits)
            phase_cos = torch.cos(p_logits)
            phi = torch.atan2(phase_sin, phase_cos)

        # Compute complex spectrum
        real = M * torch.cos(phi)
        imag = M * torch.sin(phi)
        complex_spec = torch.complex(real, imag)
        return complex_spec
```

## spectral_utils.py

```python
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
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from spectral_utils import SpectralUtils
from dataset_loader import DatasetLoader
from model import SpectralPredictor
from discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator
import yaml
import os

class Trainer:
    def __init__(self, config: dict):
        # Extract configs with defaults
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Spectral utils initialization
        spectral_cfg = self.config['training']
        self.spectral_utils = SpectralUtils(
            sample_rate=self.config['dataset']['sample_rate'],
            n_fft=self.config['model']['fft_size'],
            hop_length=self.config['model']['hop_length'],
            n_mels=self.config['dataset']['mel_params']['n_mels']
        )
        # Instantiate dataset loader
        self.dataset_loader = DatasetLoader(self.config)
        # Instantiate generator
        self.net_G = SpectralPredictor(self.config['model']).to(self.device)
        # Instantiate discriminators
        self.D_list = []
        self.D_list.append(MultiPeriodDiscriminator().to(self.device))
        self.D_list.append(MultiResolutionDiscriminator().to(self.device))
        # Optimizers
        self.g_optimizer = torch.optim.AdamW(
            self.net_G.parameters(), lr=self.config['training']['learning_rate'],
            betas=tuple(self.config['training'].get('optimizer_betas', [0.9, 0.999])))
        )
        self.d_optimizer_list = []
        for D in self.D_list:
            self.d_optimizer_list.append(
                torch.optim.AdamW(D.parameters(), lr=self.config['training']['learning_rate'],
                                  betas=tuple(self.config['training'].get('optimizer_betas', [0.9, 0.999])))
            )
        # Training schedule
        self.total_iterations = self.config['training']['total_iterations']
        self.save_interval = self.config['training'].get('save_interval', 100000)
        self.log_interval = self.config['training'].get('log_interval', 1000)
        # Loss weights
        self.lambda_adv = 1.0
        self.lambda_mel = 1.0
        self.lambda_feat = 10.0
        # For simplicity, assume spectral regularization is off
        # Initialize logs and path
        self.checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Iteration counter
        self.global_step = 0

    def compute_adversarial_loss_D(self, real_scores, fake_scores):
        """Hinge loss for discriminator."""
        loss = 0
        for rs, fs in zip(real_scores, fake_scores):
            loss += torch.mean(F.relu(1.0 - rs)) + torch.mean(F.relu(1.0 + fs))
        return loss / len(real_scores)

    def compute_adversarial_loss_G(self, fake_scores):
        """Hinge loss for generator."""
        loss = 0
        for fs in fake_scores:
            loss += -torch.mean(fs)
        return loss / len(fake_scores)

    def run(self):
        for iteration in range(1, self.total_iterations + 1):
            self.global_step = iteration
            # Fetch batch
            mel_spec, real_waveform = self.dataset_loader.get_batch(self.config['training']['batch_size'])
            mel_spec = mel_spec.to(self.device)  # [B, N_mels, T]
            real_waveform = real_waveform.to(self.device)  # [B, samples]

            # Data augmentation (e.g., gain)
            # Not explicitly in code; assume dataset loader handles

            # Forward: generator
            m_logits, p_logits = self.net_G(mel_spec)  # [B, F, T], [B, F, T]

            # Convert network output to complex spectral coefficients
            spectral_coeffs = self.spectral_utils.prepare_spectral_outputs(m_logits, p_logits)
            # spectral_coeffs: [B, F, T], complex tensor

            # Waveform synthesis from spectral coefficients
            fake_waveform = self.spectral_utils.inverse_spectrogram(spectral_coeffs)
            # Clamp / normalize waveform if needed
            fake_waveform = fake_waveform.clamp_(-1.0, 1.0)

            # Discriminator steps
            real_scores_list = []
            fake_scores_list = []
            real_feats_list = []
            fake_feats_list = []

            for D, d_optimizer in zip(self.D_list, self.d_optimizer_list):
                # Real
                real_scores, real_feats = D(real_waveform)
                # Fake (detach to avoid grad through G when training D)
                fake_scores, fake_feats = D(fake_waveform.detach())

                real_scores_list.append(real_scores)
                real_feats_list.append(real_feats)
                fake_scores_list.append(fake_scores)
                fake_feats_list.append(fake_feats)

                # Discriminator optimizer step
                d_loss = self.compute_adversarial_loss_D([real_scores], [fake_scores])
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # --- Generator forward again for gradient --- #
            # Recompute discriminator scores for generator update
            gen_scores_list = []
            gen_feats_list = []
            for D in self.D_list:
                scores, feats = D(fake_waveform)
                gen_scores_list.append(scores)
                gen_feats_list.append(feats)

            # --- Compute Generator losses --- #
            # Adversarial loss
            g_adv_loss = self.compute_adversarial_loss_G(gen_scores_list)

            # Mel Spectrogram Loss between real and fake waveforms
            pred_mel = self.spectral_utils.compute_mel_spectrogram(fake_waveform)
            mel_true = self.spectral_utils.compute_mel_spectrogram(real_waveform)
            mel_loss = F.l1_loss(pred_mel, mel_true)

            # Feature matching loss
            feat_loss = 0
            num_Ds = len(self.D_list)
            for real_feats, fake_feats in zip(real_feats_list, gen_feats_list):
                for rf, ff in zip(real_feats, fake_feats):
                    feat_loss += F.l1_loss(ff, rf)
            feat_loss = feat_loss / (num_Ds * len(real_feats))

            # Total generator loss
            g_loss = self.lambda_adv * g_adv_loss + self.lambda_mel * mel_loss + self.lambda_feat * feat_loss

            # Optimize generator
            self.net_G.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging
            if iteration % self.log_interval == 0:
                print(f"Iter {iteration}: G_loss={g_loss.item():.4f}, adv={g_adv_loss.item():.4f}, mel={mel_loss.item():.4f}, feat={feat_loss.item():.4f}")

            # Save checkpoint
            if iteration % self.save_interval == 0:
                torch.save(self.net_G.state_dict(), os.path.join(self.checkpoint_dir, f"G_{iteration}.pt"))
                for idx, D in enumerate(self.D_list):
                    torch.save(D.state_dict(), os.path.join(self.checkpoint_dir, f"D{idx}_{iteration}.pt"))

            # Optionally: perform validation, evaluation metrics computation
            # For brevity, not included here.

        print("Training complete.")

```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\vocos\vocos_repo`
