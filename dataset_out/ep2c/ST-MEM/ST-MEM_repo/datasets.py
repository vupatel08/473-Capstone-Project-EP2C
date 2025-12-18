## datasets.py

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
import scipy.io
import random

class ECGDataset(Dataset):
    """
    ECGDataset manages loading, preprocessing, segmenting, patchifying,
    augmenting, and masking ECG signals for pretraining or downstream tasks.

    Attributes:
        data_list (list): List of dicts with keys: 'signal' (np.ndarray), 'label' (optional), 'lead_idx' (int).
        mode (str): 'pretrain', 'train', 'val', or 'test'.
        config (dict): Configuration parameters.
        lead_count (int): Number of leads in data (fixed to 12).
        signal_length (int): Samples per segment (e.g., 250 samples for 10 sec @250Hz).
        patch_size (int): Number of samples per patch.
        max_patches (int): Number of patches (usually 128).
        mask_ratio (float): Fraction of patches to mask in pretraining.
        augmentations (list): List of augmentation types.
        augmentation_params (dict): Parameters for augmentations.
        data_indices (list): Indices for shuffling/train split, etc.
        dataset_size (int): Total number of samples.
    """

    def __init__(self, file_paths, label_paths=None, mode='train', config=None):
        """
        Initializes the ECGDataset.
        Args:
            file_paths (list): List of paths to raw ECG files (e.g. .mat, .npy).
            label_paths (list): List of label files or labels; optional.
            mode (str): 'pretrain', 'train', 'val', or 'test'.
            config (dict): Configuration dictionary with hyperparameters.
        """
        self.file_paths = file_paths
        self.label_paths = label_paths
        self.mode = mode
        self.config = config if config is not None else {}
        self.lead_count = self.config.get('lead_count', 12)
        self.signal_length = self.config.get('segment_duration', 10) * self.config.get('sampling_rate', 250)  # e.g., 2500 samples
        self.patch_size = self.config.get('patch_size', 32)
        self.max_patches = self.config.get('num_patches', 128)
        self.mask_ratio = self.config.get('mask_ratio', 0.15) if mode=='pretrain' else 0.0
        self.augmentations = self.config.get('augmentation', {}).get('types', [])
        self.augmentation_params = self.config.get('augmentation', {}).get('parameters', {})
        # Load raw data
        self.data_list = []
        self._load_data()
        # Generate dataset indices with shuffle for train, no shuffle for val/test
        self.data_indices = list(range(len(self.data_list)))
        if self.mode=='train':
            random.shuffle(self.data_indices)
        # Compute total size
        self.dataset_size = len(self.data_indices)

    def _load_data(self):
        """
        Loads and preprocesses data from file_paths and labels.
        Supports .mat, .npy, or raw signals.
        Assumes data is stored in a format accessible via scipy/io or numpy.
        """
        for idx, fp in enumerate(self.file_paths):
            # Load raw signal
            signal = self._load_signal_from_path(fp)
            # Resample to target sampling rate
            signal = self._resample_signal(signal, target_rate=self.config.get('sampling_rate', 250))
            # Segment into 10s segments, discard if shorter
            segments = self._segment_signal(signal)
            for seg in segments:
                # Normalize signal (Z-score)
                if self.config.get('normalization', True):
                    seg = self._normalize(seg)
                # Store with optional label info
                sample_dict = {
                    'signal': seg,  # shape: (L, T)
                    'lead_idx': np.arange(self.lead_count),  # e.g., 0..11
                }
                # Handle label if provided
                if self.label_paths is not None:
                    label = self._load_label_for_signal(idx)
                    sample_dict['label'] = label
                self.data_list.append(sample_dict)

    def _load_signal_from_path(self, path):
        """
        Load raw ECG signal from disk.
        Support .mat, .npy, or other formats.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == '.mat':
            mat = scipy.io.loadmat(path)
            # Assuming the main variable is 'val' or similar
            # Adjust as per dataset
            signal = mat.get('val', None)
            if signal is None:
                raise ValueError(f"Cannot find variable 'val' in {path}")
            # Ensure shape: (L, T)
            signal = np.array(signal)
        elif ext == '.npy':
            signal = np.load(path)
        else:
            # Placeholder: other formats can be added
            raise NotImplementedError(f"Unsupported file extension {ext}")
        return signal

    def _resample_signal(self, signal, target_rate=250):
        """
        Resamples signals to target sampling rate.
        """
        original_rate = self.config.get('original_sampling_rate', 500)  # default
        if original_rate != target_rate:
            new_length = int(signal.shape[1] * target_rate / original_rate)
            signal = resample(signal, new_length, axis=1)
        return signal

    def _segment_signal(self, signal):
        """
        Segment signals into 10s windows; discard short segments.
        Supports multi-lead signals of shape (L, T).
        """
        total_samples = signal.shape[1]
        segment_samples = self.config.get('segment_duration', 10) * 250  # e.g., 250 samples/sec
        segments = []

        if total_samples < segment_samples:
            # Too short, discard or pad
            return []
        else:
            # Crop starting at 0
            num_segments = total_samples // segment_samples
            for i in range(num_segments):
                start = i * segment_samples
                end = start + segment_samples
                seg = signal[:, start:end]
                if seg.shape[1] == segment_samples:
                    segments.append(seg)
        return segments

    def _load_label_for_signal(self, index):
        """
        Load label associated with given index.
        Assumes labels are stored per-signal, supports multi-label.
        """
        # Placeholder: Implement actual label loading
        # For now, return a dummy label (scalar or array)
        return 0

    def _normalize(self, segments):
        """
        Z-normalize input signal (per segment, per lead).
        """
        mean = segments.mean(axis=1, keepdims=True)
        std = segments.std(axis=1, keepdims=True)
        std[std == 0] = 1e-8
        return (segments - mean) / std

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        """
        Retrieve processed sample:
        - Patchify
        - Apply augmentations (if training)
        - Mask patches (if pretraining)
        - Return dict with patches, mask, label
        """
        real_idx = self.data_indices[index]
        sample = self.data_list[real_idx]
        signal = sample['signal']  # shape: (L, T)

        # Patchify
        patches = self._patchify(signal)

        # For pretraining, apply augmentations and masking
        if self.mode=='pretrain':
            patches_aug = self._apply_augmentations(patches)
            mask = self._generate_mask(patches_aug.shape)
            # Mask patches as boolean mask
            patches_masked, patches_orig, mask_tensor = self._apply_mask(patches_aug, patches)
            return {
                'patches': torch.tensor(patches_masked, dtype=torch.float),  # shape: (L, n_patches, p)
                'mask': torch.tensor(mask_tensor, dtype=torch.bool),
                'original_patches': torch.tensor(patches_orig, dtype=torch.float),
                'lead_idx': torch.tensor(sample['lead_idx'], dtype=torch.long),
            }
        else:
            # Downstream classification mode: no mask
            return {
                'patches': torch.tensor(patches, dtype=torch.float),
                'lead_idx': torch.tensor(sample['lead_idx'], dtype=torch.long),
                'label': torch.tensor(sample['label']) if 'label' in sample else None,
            }

    def _load_signal_from_path(self, path):
        # Implemented above
        pass

    def _resample_signal(self, signal, target_rate=250):
        # Implemented above
        pass

    def _segment_signal(self, signal):
        # Implemented above
        pass

    def _normalize(self, segments):
        # Implemented above
        pass

    def _patchify(self, signal):
        """
        Divides the multi-lead signal into patches.
        Args:
            signal (np.ndarray): shape (L, T)
        Returns:
            patches (np.ndarray): shape (L, n_patches, p)
        """
        L, T = signal.shape
        patch_size = self.patch_size
        n_patches = self.max_patches

        # Compute total required length
        total_length = n_patches * patch_size
        if T < total_length:
            # Pad with zeros
            pad_width = total_length - T
            signal_padded = np.pad(signal, ((0,0),(0,pad_width)), mode='constant')
        else:
            # Truncate
            signal_padded = signal[:, :total_length]

        # Reshape into patches
        # shape: (L, n_patches, patch_size)
        patches = signal_padded.reshape(L, n_patches, patch_size)
        return patches

    def _generate_mask(self, shape):
        """
        Generate a boolean mask for patches to mask.
        shape: (L, n_patches)
        """
        mask = np.zeros(shape, dtype=bool)
        total_patches = shape[0] * shape[1]
        num_mask = int(self.mask_ratio * total_patches)
        # Randomly select patches across all leads
        indices = [(l, p) for l in range(shape[0]) for p in range(shape[1])]
        masked_indices = random.sample(indices, num_mask)
        for l, p in masked_indices:
            mask[l, p] = True
        return mask

    def _apply_mask(self, patches, original_patches):
        """
        Apply the mask to patches:
        - masked patches: replace with shared mask embedding during training
        - Keep original in original_patches for loss computation
        """
        # For simplicity, return patches with masked patches zeroed (or replaced externally)
        # During training, the model should know which patches are masked via mask tensor
        patches_masked = patches.copy()
        mask_tensor = np.zeros(patches.shape[:2], dtype=bool)
        return patches_masked, original_patches, mask_tensor

    def _apply_augmentations(self, patches):
        """
        Apply random augmentations to patches during pretraining.
        """
        augmented_patches = np.copy(patches)
        for lead_idx in range(augmented_patches.shape[0]):
            for patch_idx in range(augmented_patches.shape[1]):
                patch = augmented_patches[lead_idx, patch_idx]
                for aug_type in self.augmentations:
                    params = self.augmentation_params.get(aug_type, {})
                    patch = utils.apply_augmentation(patch, aug_type, params)
                augmented_patches[lead_idx, patch_idx] = patch
        return augmented_patches

    # Additional helper functions as needed...
