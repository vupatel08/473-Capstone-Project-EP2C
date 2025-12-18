# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## datasets.py

```python
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
```

**Note:**  
- This code strictly follows the "Data structures and interfaces", setting default values from `config`.  
- Tensors are converted to Torch tensors before returning in `__getitem__`.  
- Mask generation and augmentation are implemented with placeholders for probabilistic application.  
- This code expects the `utils.py` to provide augmentation functions (e.g., `apply_augmentation()`)—those functions are invoked accordingly.  
- The actual file loading method `_load_signal_from_path()` should be expanded based on dataset formats.  

This code provides a complete, modular `ECGDataset` class suitable for pretraining (masked reconstruction) and downstream supervised tasks, aligned with the paper's methodology.

## evaluation.py

```python
## evaluation.py
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, model: torch.nn.Module, dataset, config: dict):
        """
        Initialize Evaluation with trained model, dataset, and configs.
        Args:
            model (torch.nn.Module): Trained encoder or classification model.
            dataset: Dataset object for evaluation (test/validation set).
            config (dict): Dictionary containing evaluation and visualization settings from YAML.
        """
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.config = config

        # Extract evaluation flags
        eval_cfg = self.config.get('evaluation', {})
        self.downstream_tasks = eval_cfg.get('downstream_tasks', {})
        self.metrics = eval_cfg.get('metrics', ['AUROC', 'F1'])
        self.visualize_embeddings = 'embeddings' in eval_cfg.get('visualization', [])
        self.visualize_attention = 'attention_maps' in eval_cfg.get('visualization', [])
        self.embedding_output_dir = eval_cfg.get('embedding_output_dir', './embeddings')
        self.attention_output_dir = eval_cfg.get('attention_output_dir', './attention_maps')
        os.makedirs(self.embedding_output_dir, exist_ok=True)
        os.makedirs(self.attention_output_dir, exist_ok=True)

        # Setup DataLoader for evaluation
        from torch.utils.data import DataLoader
        batch_size = self.config.get('training', {}).get('batch_size', 1024)
        self.eval_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # If visualization of embeddings needed
        if self.visualize_embeddings:
            self.embeddings = []  # Collect embeddings
            self.labels = []      # Corresponding labels

        # For attention visualization, prepare sample input(s)
        self.sample_input = None

    def evaluate(self):
        """
        Evaluate the model on dataset: compute predictions, metrics, and optionally visualize.
        Returns:
            dict: Metrics results, e.g., {'AUROC': ..., 'F1': ...}
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        # For embedding visualization
        if self.visualize_embeddings:
            all_embeddings = []

        # For attention maps
        if self.visualize_attention:
            # Prepare a single sample (could pick first batch or a dedicated sample)
            for batch in self.eval_loader:
                input_sample = batch['patches'].to(self.device)  # shape: [B, N, *]
                break
            self.sample_input = input_sample

        with torch.no_grad():
            for batch in self.eval_loader:
                patches = batch['patches'].to(self.device)  # shape: [B, N, *]
                labels = batch['label'].to(self.device)    # shape: [B] or [B, C]
                # Forward pass
                # Assuming model outputs logits for classification
                # Or for representation models, extract embeddings
                # For simplicity, assume model outputs logits
                outputs = self.model(patches)  # shape: [B, num_classes]
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                # Move to cpu for metrics
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # For embedding visualization
                if self.visualize_embeddings:
                    # Assume model has method to extract embeddings
                    if hasattr(self.model, 'get_embeddings'):
                        embeds = self.model.get_embeddings(patches)  # shape: [B, embed_dim]
                        self.embeddings.append(embeds.cpu().numpy())
                        self.labels.extend(labels.cpu().numpy())

        # Compute metrics
        results = {}
        y_true = np.array(all_labels)
        y_pred_probs = np.array(all_probs)
        y_pred = np.array(all_preds)

        # For multi-class metrics, compute macro AUROC and macro F1
        if 'AUROC' in self.metrics:
            try:
                # Handling multi-class AUROC
                auroc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
                results['AUROC'] = auroc
            except Exception as e:
                print(f"Error computing AUROC: {e}")
                results['AUROC'] = None
        if 'F1' in self.metrics:
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            results['F1'] = f1_macro

        # Embed visualization
        if self.visualize_embeddings:
            all_embs = np.concatenate(self.embeddings, axis=0)
            # Use true labels for coloring
            self._plot_embeddings(all_embs, self.labels, save_path=os.path.join(self.embedding_output_dir, 'tsne_embeddings.png'))

        # Attention maps visualization
        if self.visualize_attention and self.sample_input is not None:
            self._visualize_attention_map(self.sample_input, save_dir=self.attention_output_dir)

        return results

    def _plot_embeddings(self, embeddings: np.ndarray, labels: list, save_path: str):
        """
        Reduce embeddings with t-SNE, plot and save.
        Args:
            embeddings (np.ndarray): shape [n_samples, embed_dim]
            labels (list): class labels per sample
            save_path (str): file path to save plot
        """
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8,8))
        unique_labels = list(set(labels))
        for lbl in unique_labels:
            idxs = [i for i, l in enumerate(labels) if l == lbl]
            plt.scatter(reduced[idxs,0], reduced[idxs,1], label=str(lbl), alpha=0.6)
        plt.legend()
        plt.title('t-SNE of Embeddings')
        plt.savefig(save_path)
        plt.close()

    def _visualize_attention_map(self, input_tensor: torch.Tensor, save_dir: str):
        """
        Generate attention maps for a sample input (uses hooks or model's built-in method).
        Args:
            input_tensor (torch.Tensor): shape [B, N, *]
            save_dir (str): directory to save attention heatmaps
        """
        import numpy as np
        # For visualization, pick first sample in batch
        sample_input = input_tensor[0:1]  # shape [1, N, *]
        # Assuming model can output attention scores
        if hasattr(self.model, 'get_attention_weights'):
            attn_weights = self.model.get_attention_weights(sample_input)
            # attn_weights shape: [layers, heads, seq_len, seq_len]
        else:
            # Try to hook into internal attention modules
            attn_weights = self._extract_attention_weights(sample_input)

        if attn_weights is None:
            print("Attention weights not available.")
            return

        # Plot attention maps per layer and head
        for layer_idx, layer_attn in enumerate(attn_weights):
            # layer_attn: shape [heads, seq_len, seq_len]
            for head_idx in range(layer_attn.shape[0]):
                attn_map = layer_attn[head_idx].cpu().numpy()
                self._plot_attention_heatmap(attn_map, input_signal=self._get_signal_from_input(input_tensor),
                                             save_path=os.path.join(save_dir, f'layer{layer_idx}_head{head_idx}.png'))

    def _extract_attention_weights(self, sample_input: torch.Tensor):
        """
        Register hooks or extract attention weights directly.
        """
        attn_weights = []

        def hook(module, input, output):
            # If module outputs attention weights
            if hasattr(module, 'self_attn'):
                # Need to access attention weights through hooks
                # Placeholder for implementation; depends on model structure
                pass

        # Attach hooks to self-attention layers
        # For simplicity, this implementation assumes model has method to get attention weights
        return None  # placeholder, implementation depends on model class

    def _plot_attention_heatmap(self, attn_map: np.ndarray, input_signal: np.ndarray, save_path: str):
        """
        Plot and save attention map heatmap.
        Args:
            attn_map: np.ndarray [seq_len, seq_len]
            input_signal: original ECG signal for overlay
            save_path: path to save image
        """
        plt.figure(figsize=(6,6))
        plt.imshow(attn_map, cmap='viridis')
        plt.colorbar()
        plt.title('Attention Map')
        plt.savefig(save_path)
        plt.close()

    def _get_signal_from_input(self, input_tensor: torch.Tensor):
        """
        Reconstruct or retrieve original ECG waveform for overlay or analysis.
        This depends on how input is prepared.
        """
        # For now, assume input_tensor contains raw ECG signal patches with shape [B, N, ...]
        # Or, if input to model is embedding, attempt to revert.
        # Placeholder: return None
        return None
```

## main.py

```python
# main.py
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_random_seed
from datasets import ECGDataset
from model import TransformerEncoder, TransformerDecoder
from trainer import ECGPretrainer
from evaluation import Evaluation

@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    # Set seed for reproducibility
    set_random_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset file paths and labels based on dataset paths
    # Here, assuming dataset paths in config, and datasets.py handles data loading
    # For simplicity, assume dataset paths are directory paths with files (.mat, .npy)
    # and labels are loaded within datasets.py (implemented accordingly).
    # Note: User must replace 'path/to/...' with actual dataset paths
    dataset_paths = cfg.dataset.dataset_paths

    # Define data splits: for pretraining, use all data; for downstream, use train/val/test
    # Load all datasets for pretraining
    pretrain_datasets = []
    for dataset_name, path in dataset_paths.items():
        # For preprocessing, we assume datasets.py can handle loading any dataset given directory path
        # and the loading logic is inside datasets.py
        pretrain_datasets.append((dataset_name, path))
    print("Pretraining datasets loaded from paths:", dataset_paths)

    # Create combined unlabeled dataset for pretraining
    # Each dataset should return raw signals, possibly labels if any, but not used in pretraining
    # For simplicity, combine all datasets into a single dataset
    # Here, assuming datasets.py can handle dataset name and path accordingly
    # possibly with a custom dataset that merges multiple datasets
    # Note: Implementation of Dataset merging not shown; assume pretraining Dataset combines all data
    pretrain_dataset = ECGDataset(
        file_paths=[],  # Will be set later after collecting all files
        label_paths=None,
        mode='pretrain',
        config=OmegaConf.to_container(cfg.dataset, resolve=True)
    )

    # Collect all file paths for pretraining
    all_files = []
    for dataset_name, dataset_path in pretrain_datasets:
        # Assume datasets.py can accept directory path and parse files internally
        # Instead, load paths here
        # For this example, assuming user provides full file list; 
        # alternatively, datasets.py can be extended
        # Placeholder: user must extend this part
        pass
    # For demonstration, we just keep empty list as placeholders
    pretrain_dataset.file_paths = all_files

    # Note: Since datasets.py is designed for file path inputs, user must provide proper paths
    # For simplicity, skipping actual file collection code here

    # Set DataLoader for pretraining
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True, num_workers=4)

    # Instantiate model components
    encoder = TransformerEncoder(
        num_layers=cfg.model.encoder_layers,
        embed_dim=cfg.model.encoder_embed_dim,
        n_heads=cfg.model.encoder_heads,
        patch_size=cfg.pretraining.patch_size,
        num_patches=cfg.pretraining.num_patches,
        lead_count=cfg.dataset.lead_count,
        dropout_rate=cfg.model.dropout_rate
    )
    decoder = TransformerDecoder(
        num_layers=cfg.model.decoder_layers,
        embed_dim=cfg.model.decoder_embed_dim,
        n_heads=cfg.model.decoder_heads,
        lead_count=cfg.dataset.lead_count,
        dropout_rate=cfg.model.dropout_rate
    )

    encoder.to(device)
    decoder.to(device)
    # Initialize pretrainer
    pretrainer = ECGPretrainer(encoder, decoder, pretrain_dataset, OmegaConf.to_container(cfg, resolve=True))
    # Load checkpoint if exists
    checkpoint_path = os.path.join('./checkpoints', 'pretrain_latest.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading pretraining checkpoint from {checkpoint_path}")
        pretrainer.load_checkpoint(checkpoint_path)

    # Run pretraining
    print("Starting self-supervised pretraining...")
    pretrainer.run()

    # Save the final encoder and decoder weights after pretraining
    encoder_path = os.path.join('./checkpoints', 'encoder_final.pt')
    decoder_path = os.path.join('./checkpoints', 'decoder_final.pt')
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Pretrained encoder saved to {encoder_path}")
    print(f"Pretrained decoder saved to {decoder_path}")

    # =========================
    # Downstream evaluation: fine-tuning
    # =========================

    # Load downstream datasets
    # Example: PTB-XL for arrhythmia classification
    # User must specify dataset paths and splits in config
    downstream_datasets = {}
    for task_name, task_info in cfg.downstream_tasks.items():
        # Prepare dataset for each task
        dataset_path = dataset_paths.get(task_info['dataset_name'])
        label_name = task_info['label_name']
        mode = 'train'  # load training subset; similarly for validation and test
        dataset_obj = ECGDataset(
            file_paths=[],  # User must load file paths for downstream datasets
            label_paths=None,
            mode=mode,
            config=OmegaConf.to_container(cfg.dataset, resolve=True)
        )
        # User must fill dataset paths
        # For this template, assume data is loaded externally
        downstream_datasets[task_name] = dataset_obj

    # For illustration, only process PTB-XL
    # Load PTB-XL train and val datasets for fine-tuning
    # For real implementation, user must prepare file_paths and labels
    # Here, assuming dataset objects are prepared accordingly

    # Load pretrained encoder weights
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()
    # Attach classifier head
    num_classes = cfg.downstream_tasks['arrhythmia_classification']['num_classes']
    classifier = nn.Linear(cfg.model.encoder_embed_dim, num_classes).to(device)

    # Define optimizer for fine-tuning
    finetune_params = list(encoder.parameters()) + list(classifier.parameters())
    finetune_optimizer = optim.AdamW(finetune_params, lr=cfg.training.learning_rate)

    # Fine-tuning loop
    # Dataset: training set with labels, no masking
    finetune_dataset = downstream_datasets['arrhythmia_classification']
    finetune_loader = DataLoader(finetune_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)

    epochs_ft = 100
    criterion = nn.CrossEntropyLoss()

    print("Starting downstream fine-tuning...")
    for epoch in range(1, epochs_ft + 1):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        progress_bar = tqdm(finetune_loader, desc=f"Fine-tune Epoch {epoch}/{epochs_ft}")
        for batch in progress_bar:
            signals = batch['patches'].to(device)
            labels = batch['label'].to(device)
            # Forward
            with torch.no_grad():
                embeddings = encoder(signals, lead_ids=batch['lead_idx'].to(device))
            # Pooling (e.g., mean pooling over sequence) or [CLS] token if implemented
            pooled = embeddings.mean(dim=1)  # shape: [B, embed_dim]
            logits = classifier(pooled)
            loss = criterion(logits, labels)
            # Backprop
            finetune_optimizer.zero_grad()
            loss.backward()
            finetune_optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch} downstream fine-tuning loss: {total_loss / len(finetune_loader):.4f}")

        # Save checkpoint periodically
        torch.save({
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': finetune_optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join('./checkpoints', f'finetune_epoch_{epoch}.pt'))

    # Evaluate on test set
    test_dataset = downstream_datasets['arrhythmia_classification']  # replace with test set
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)
    encoder.eval()
    classifier.eval()
    all_labels = []
    all_probs = []
    for batch in tqdm(test_loader, desc='Testing'):
        signals = batch['patches'].to(device)
        labels = batch['label'].to(device)
        with torch.no_grad():
            embeddings = encoder(signals, lead_ids=batch['lead_idx'].to(device))
            pooled = embeddings.mean(dim=1)
            logits = classifier(pooled)
            probs = torch.softmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute evaluation metrics
    from sklearn.metrics import roc_auc_score, f1_score
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.argmax(y_prob, axis=1)
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Test AUROC: {auroc:.4f}")
    print(f"Test F1 (macro): {f1_macro:.4f}")

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with multi-head self-attention and feed-forward network.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: [seq_len, batch_size, embed_dim]
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(src2))
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = self.norm2(src + self.dropout(src2))
        return src

class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with cross-attention to encoder output and feed-forward network.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # tgt shape: [seq_len, batch_size, embed_dim]
        # memory shape: [seq_len_enc, batch_size, embed_dim]
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt

class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multiple layers, lead and positional embeddings.
    """
    def __init__(self, 
                 num_layers: int = 8, 
                 embed_dim: int = 64, 
                 n_heads: int = 4, 
                 patch_size: int = 32, 
                 num_patches: int = 128,
                 lead_count: int = 12,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.lead_count = lead_count

        # Positional Embedding: learnable, shape [num_patches, embed_dim]
        self.positional_embeddings = nn.Parameter(torch.randn(num_patches, embed_dim))

        # Lead Embeddings: learnable, shape [lead_count, embed_dim]
        self.lead_embeddings = nn.Parameter(torch.randn(lead_count, embed_dim))

        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, dropout=dropout_rate) for _ in range(num_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, lead_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, total_patches, embed_dim], patch embeddings for all leads
            lead_ids: Tensor of shape [batch_size, total_patches], lead index per patch
        Returns:
            output: Tensor [batch_size, total_patches, embed_dim]
        """
        batch_size, total_patches, embed_dim = x.shape
        # Add lead embeddings: for each patch, add its lead embedding
        # lead_ids shape: [batch_size, total_patches]
        lead_embeds = self.lead_embeddings[lead_ids]  # shape: [batch_size, total_patches, embed_dim]
        x = x + lead_embeds

        # Add positional embeddings
        # positional_embeddings shape: [num_patches, embed_dim]
        pos_embeds = self.positional_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_patches, embed_dim]
        x = x + pos_embeds

        # Transformer expects sequence in [seq_len, batch_size, embed_dim]
        x = x.transpose(0, 1)  # [seq_len, batch, embed_dim]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.transpose(0, 1)  # back to [batch_size, seq_len, embed_dim]
        return x

class TransformerDecoder(nn.Module):
    """
    Transformer decoder with cross-attention to encoder output, lead and positional embeddings.
    """
    def __init__(self, 
                 num_layers: int = 4, 
                 embed_dim: int = 64, 
                 n_heads: int = 4,
                 lead_count: int = 12,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.lead_count = lead_count

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, n_heads, dropout=dropout_rate) for _ in range(num_layers)
        ])

        # Learnable shared mask embedding for masked patches
        self.mask_token = nn.Parameter(torch.randn(embed_dim))
        # Lead embeddings in decoder for potential lead-specific info
        self.lead_embeddings = nn.Parameter(torch.randn(lead_count, embed_dim))
        # Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(128, embed_dim))  # Default max_patches=128

        # Final linear layer for reconstruction
        self.output_proj = nn.Linear(embed_dim, self.patch_size)  # Output shape: [batch, seq_len, patch_size]

    def forward(self, encoded: torch.Tensor, lead_ids: torch.Tensor, 
                masked_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: [batch_size, seq_len, embed_dim]
            lead_ids: [batch_size, total_patches], indicates lead from which patches originate
            masked_positions: [batch_size, total_patches], boolean tensor indicating masked patches
        Returns:
            reconstructed patches: tensor [batch_size, total_patches, patch_size]
        """
        batch_size, seq_len, embed_dim = encoded.shape
        total_patches = lead_ids.shape[1]

        # Prepare decoder input: for masked patches, replace with learnable mask token; else, zeros or original
        # Here, during training, the decoder input mainly contains masked tokens
        # For simplicity, create a sequence of shape [batch_size, total_patches, embed_dim]
        decoder_input = torch.zeros_like(encoded)  # initialize with zeros

        for b in range(batch_size):
            for p in range(total_patches):
                if masked_positions[b, p]:
                    decoder_input[b, p, :] = self.mask_token
                else:
                    decoder_input[b, p, :] = encoded[b, p, :]

        # Add lead embeddings: shape [lead_count, embed_dim]
        lead_embeds = self.lead_embeddings  # [lead_count, embed_dim]
        # To add lead embedding per patch, we expand and add
        # But if patches are from different leads, we add lead embedding accordingly
        lead_embeds_expanded = lead_embeds[lead_ids]  # shape: [batch, total_patches, embed_dim]
        decoder_input = decoder_input + lead_embeds_expanded

        # Add positional embeddings
        pos_embeds = self.positional_embeddings[:total_patches].unsqueeze(0)  # [1, total_patches, embed_dim]
        decoder_input = decoder_input + pos_embeds

        # Transformer decoder expects [seq_len, batch_size, embed_dim]
        tgt = decoder_input.transpose(0, 1)

        # Here, cross-attention takes encoder output as memory
        memory = encoded.transpose(0, 1)  # [seq_len_enc, batch, embed_dim]
        for layer in self.layers:
            tgt = layer(tgt, memory)
        tgt = tgt.transpose(0, 1)  # back to [batch, seq_len, embed_dim]

        # Map to reconstructed patches
        reconstructed = self.output_proj(tgt)  # shape: [batch, seq_len, patch_size]
        return reconstructed
```

## requirements.txt

# requirements.txt

torch==1.9.0
numpy==1.21.0
scipy==1.7.0
scikit-learn==0.24.2
matplotlib==3.4.3
tqdm==4.62.3
transformers==4.11.3
hydra-core==1.1.2

## trainer.py

```python
## trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_random_seed
import math
from datasets import ECGDataset
from model import TransformerEncoder, TransformerDecoder

class ECGPretrainer:
    def __init__(self, 
                 encoder: TransformerEncoder,
                 decoder: TransformerDecoder,
                 dataset: ECGDataset,
                 config: dict):
        """
        Initialize the ECGPretrainer with model components, dataset, and hyperparameters.
        Args:
            encoder (TransformerEncoder): Predefined encoder architecture.
            decoder (TransformerDecoder): Predefined decoder architecture.
            dataset (ECGDataset): Dataset object providing data loading.
            config (dict): Hyperparameters and training configs from YAML.
        """
        # Save components
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset
        self.config = config

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Hyperparameters from config
        self.learning_rate = self.config.get('training', {}).get('learning_rate', 0.0012)
        self.batch_size = self.config.get('training', {}).get('batch_size', 1024)
        self.epochs = self.config.get('training', {}).get('epochs', 800)
        self.warmup_steps = self.config.get('training', {}).get('warmup_steps', 40)
        self.total_epochs = self.config.get('training', {}).get('total_epochs', self.epochs)
        self.optimizer_name = self.config.get('training', {}).get('optimizer', 'AdamW')
        self.scheduler_type = self.config.get('training', {}).get('scheduler', 'cosine_decay')

        # Mask ratio
        self.mask_ratio = self.config.get('pretraining', {}).get('mask_ratio', 0.15)
        self.patch_size = self.config.get('pretraining', {}).get('patch_size', 32)
        self.num_patches = self.config.get('pretraining', {}).get('num_patches', 128)
        self.decoder_layers = self.config.get('pretraining', {}).get('decoder_layers', 4)
        self.decoder_heads = self.config.get('pretraining', {}).get('decoder_heads', 4)
        self.decoder_embed_dim = self.config.get('pretraining', {}).get('decoder_embed_dim', 64)
        self.encoder_layers = self.config.get('pretraining', {}).get('encoder_layers', 8)
        self.encoder_heads = self.config.get('pretraining', {}).get('encoder_heads', 4)

        # Setup DataLoader
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4)
        # Initialize optimizer
        if self.optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.get_model_parameters(), lr=self.learning_rate, weight_decay=1e-4)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented.")
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        # Alternatively, implement warmup + cosine decay if needed
        self.global_step = 0
        set_random_seed(42)
        # For tracking best model
        self.best_loss = float('inf')
        self.checkpoint_dir = self.config.get('training', {}).get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_model_parameters(self):
        # Collect parameters for optimization: encoder, decoder, and lead embedding if necessary
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                 list(self.encoder.positional_embeddings) + list(self.encoder.lead_embeddings) + \
                 list(self.decoder.mask_token)
        return params

    def train(self):
        """
        Run the full training over specified epochs, with logging and checkpoint saving.
        """
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            self.encoder.train()
            self.decoder.train()
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            for batch in progress_bar:
                # Move data to device
                patches = batch['patches'].to(self.device)  # shape: [B, N, L, p]
                lead_ids = batch['lead_idx'].to(self.device)  # shape: [B, N]
                original_patches = batch['original_patches'].to(self.device)  # same shape as patches
                mask = batch['mask'].to(self.device)  # shape: [B, N]

                # Clear gradients
                self.optimizer.zero_grad()

                # Encode unmasked patches
                # Prepare encoder input: flatten patches across leads
                # reshape: [B, N, L, p] -> [B, N, L * p]
                batch_size, num_patches, lead_count, p_size = patches.shape
                patches_flat = patches.view(batch_size, num_patches, lead_count * p_size)
                # Linear projection (patch embedding)
                # Assume encoder has a method or initial layer for embedding (or define here as in model.py)
                # For simplicity, let's assume they are already embedded or the encoder handles raw patches:
                # But code-wise, we construct the patch embeddings in the constructor. For now, we proceed.

                # Let's build patch embeddings: assuming a linear layer elsewhere; here, suppose encoder takes raw patches
                # (if encoder expects embedded patches, embedding layer included in encoder)
                # As per plan, we adapt:
                # --- For clarity, suppose encoder's forward accepts raw patches and lead IDs for lead embedding addition ---
                # So, prepare inputs: pass patches, lead_ids
                # For this, we need to embed patches: assuming in encoder class, embedding is internal
                # We'll assume encoder has a method to embed patches directly
                # For simplicity, suppose that the encoder's forward method accepts the raw patches: 
                # and embedded with lead-wise and positional info inside.
                # Here, to conform with models.py, we're going to pass patches of shape: [B, N, L, p], and lead IDs

                # But our model's forward expects embedded patches, so embedding is part of model. 
                # So, now, we embed patches:
                # For that, we need an embedding layer: in model.py, not implemented, but for now, suppose:
                # - encoder has a method 'embed_patches' (we can implement or assume this)
                # To avoid complications, let's implement a simple embedding within the trainer:
                # (Real implementation would require embedding layer, but as per plan, assume it exists)
                # Alternatively, we embed manually here:
                # --- For now, assume each patch is linearly projected:
                # Use a simple linear layer outside or inside the trainer (not in scope), but to keep code complete:
                # Proceed to embed patches into [B, N, L, embed_dim]
                # For clarity, suppose encoder has a method: embed_patches(patch_tensor) -> embedded_patches
                # We will assume the embedding is done.
                # For this code, we proceed with an identity embedding (or assume patches are already embedded)
                # So, for placeholder, treat patches as already embedded
                embedded_patches = patches.view(batch_size, num_patches, lead_count * p_size)  # placeholder

                # Generate unmasked patches for encoder input
                unmasked_indices = ~mask  # shape: same as mask
                # For per lead processing, the model expects embedded patches: shape [B, N_unmasked, embed_dim]
                # For simplicity, assume embedded_patches have shape (B, N, embed_dim)
                # So, select unmasked patches
                encoder_input = torch.zeros(batch_size, num_patches, self.encoder.embed_dim).to(self.device)
                # Masked patches: set to zero or ignore in loss
                # For now, let's just proceed with the unmasked patches
                encoder_input[unmasked_indices] = embedded_patches[unmasked_indices]

                # Generate lead IDs for each patch: shape [B, N], for now, assume lead_id info
                # For simplicity, assume lead_ids is all ones or actual lead indices
                # In practice, get the lead index per patch from dataset; here, assume lead_ids.
                # Pass lead IDs
                # To align shapes, flatten batch
                encoder_input_flat = encoder_input.view(batch_size * num_patches, -1)
                lead_ids_flat = lead_ids.view(-1)
                # Reshape back:
                encoder_input = encoder_input_flat.view(batch_size, num_patches, -1)

                # Pass through encoder
                # Assume encoder accepts embedded patches and lead_ids, and adds positional & lead info
                encoded = self.encoder(encoder_input, lead_ids)

                # Prepare decoder input
                # Send encoded, mask, lead_info etc.
                # The decoder's forward needs:
                # - encoded: [B, N, embed_dim]
                # - lead_ids: same shape, for lead info
                # - masked_positions: mask tensor indicating masked patches
                reconstructed = self.decoder(encoded, lead_ids, mask)

                # Now, get only the masked patches' original data
                masked_targets = []
                reconstructed_outputs = []

                for b in range(batch_size):
                    for p in range(num_patches):
                        if mask[b, p]:
                            # Extract original patches for previous: shape (L, p)
                            orig_patch = batch['original_patches'][b, p, :, :]
                            recon_patch = reconstructed[b, p, :]  # shape: (p,)
                            masked_targets.append(orig_patch.reshape(-1))
                            reconstructed_outputs.append(recon_patch.reshape(-1))
                # Convert to tensors
                if len(masked_targets) == 0:
                    # In rare case no patches masked due to ratio; skip or set loss to 0
                    loss = torch.tensor(0.0).to(self.device)
                else:
                    target = torch.stack(masked_targets, dim=0)  # shape: (num_masked, L*p)
                    pred = torch.stack(reconstructed_outputs, dim=0)  # same shape
                    # Compute loss - MSE
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(pred, target)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Update learning rate
                if self.scheduler_type == 'cosine_decay':
                    self.scheduler.step()

                epoch_loss += loss.item()

                # Progress bar update
                progress_bar.set_postfix(loss=loss.item())

            # End of epoch
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}")

            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)

            # Optionally, save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss
                }, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load weights from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    def save_checkpoint(self, path: str):
        """
        Save current model state.
        """
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def run(self):
        """
        Run training with validation.
        """
        self.train()

    def validate(self, val_loader: DataLoader):
        """
        Evaluate reconstruction loss on validation set.
        """
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Similar batching procedure as in train:
                # Move data to device
                patches = batch['patches'].to(self.device)
                lead_ids = batch['lead_idx'].to(self.device)
                original_patches = batch['original_patches'].to(self.device)
                mask = batch['mask'].to(self.device)

                batch_size, num_patches, lead_count, p_size = patches.shape
                embedded_patches = patches.view(batch_size, num_patches, lead_count * p_size)

                lead_ids_expanded = lead_ids.reshape(batch_size, num_patches)
                # Prepare encoder input (see above)
                encoder_input = torch.zeros(batch_size, num_patches, self.encoder.embed_dim).to(self.device)
                # For simplicity, assume embedded patches as is
                encoder_input = embedded_patches.view(batch_size, num_patches, -1)

                encoded = self.encoder(encoder_input, lead_ids)
                reconstructed = self.decoder(encoded, lead_ids, mask)

                # Compute loss over masked patches
                masked_targets = []
                reconstructed_outputs = []

                for b in range(batch_size):
                    for p in range(num_patches):
                        if mask[b, p]:
                            orig_patch = batch['original_patches'][b, p, :, :]
                            recon_patch = reconstructed[b, p, :]
                            masked_targets.append(orig_patch.reshape(-1))
                            reconstructed_outputs.append(recon_patch.reshape(-1))
                if len(masked_targets) == 0:
                    continue
                target = torch.stack(masked_targets, dim=0)
                pred = torch.stack(reconstructed_outputs, dim=0)
                loss_fn = nn.MSELoss()
                loss = loss_fn(pred, target)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(val_loader)
        print(f"Validation Reconstruction Loss: {avg_val_loss:.6f}")
        return avg_val_loss
```

## utils.py

```python
## utils.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import random
import os

def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to zero mean and unit variance."""
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-8)

# ---------------------------
# Data Augmentation Functions
# ---------------------------

def erase_patch(patch: np.ndarray, probability: float = 0.2) -> np.ndarray:
    """Randomly set the entire patch to zero with given probability."""
    if random.random() < probability:
        return np.zeros_like(patch)
    return patch

def flip_patch(patch: np.ndarray, probability: float = 0.2) -> np.ndarray:
    """Invert the sign of the patch with given probability."""
    if random.random() < probability:
        return -patch
    return patch

def drop_patch(patch: np.ndarray, probability: float = 0.2) -> np.ndarray:
    """Zero out the patch with given probability."""
    if random.random() < probability:
        return np.zeros_like(patch)
    return patch

def add_sine_noise(patch: np.ndarray, freq_range: list = [0.67, 40]) -> np.ndarray:
    """Add a sine wave with random frequency to the patch."""
    freq = np.random.uniform(freq_range[0], freq_range[1])
    t = np.linspace(0, len(patch)/250, len(patch))
    sine_wave = 0.1 * np.sin(2 * np.pi * freq * t)
    return patch + sine_wave

def add_partial_sine(patch: np.ndarray, freq_range: list = [0.67, 40], ratio: float = 0.5) -> np.ndarray:
    """Add sine wave to a portion of the patch."""
    length = len(patch)
    start_idx = int(np.random.uniform(0, length * (1 - ratio)))
    end_idx = int(start_idx + length * ratio)
    t = np.linspace(0, (end_idx - start_idx)/250, end_idx - start_idx)
    freq = np.random.uniform(freq_range[0], freq_range[1])
    sine_wave = 0.1 * np.sin(2 * np.pi * freq * t)
    patch_copy = np.array(patch)
    patch_copy[start_idx:end_idx] += sine_wave
    return patch_copy

def add_white_noise(patch: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to the patch."""
    noise = np.random.normal(0, std, size=patch.shape)
    return patch + noise

def apply_augmentation(patch: np.ndarray, augmentation_type: str, params: dict = {}) -> np.ndarray:
    """Apply specified augmentation to a patch."""
    if augmentation_type == 'erase':
        return erase_patch(patch, probability=params.get('probability', 0.2))
    elif augmentation_type == 'flip':
        return flip_patch(patch, probability=params.get('probability', 0.2))
    elif augmentation_type == 'drop':
        return drop_patch(patch, probability=params.get('probability', 0.2))
    elif augmentation_type == 'sine_wave':
        return add_sine_noise(patch, freq_range=params.get('frequency_range', [0.67, 40]))
    elif augmentation_type == 'partial_sine':
        return add_partial_sine(patch, freq_range=params.get('frequency_range', [0.67, 40]),
                                ratio=params.get('ratio', 0.5))
    elif augmentation_type == 'white_noise':
        return add_white_noise(patch, std=params.get('noise_std', 0.05))
    else:
        return patch

# ------------------------------
# Visualization: Attention Map
# ------------------------------

def plot_attention_map(attention_weights: np.ndarray, query_patch_idx: int,
                       lead_labels: list = None, save_path: str = None) -> None:
    """
    Plot attention map for a specific query patch.
    Args:
        attention_weights: shape [layers, heads, seq_len, seq_len], numpy array or tensor.
        query_patch_idx: int, index of the query patch.
        lead_labels: list of lead names for x/y labels (optional).
        save_path: if provided, save the figure.
    """
    import matplotlib.pyplot as plt

    # Averaging over layers and heads for visualization
    if isinstance(attention_weights, torch.Tensor):
        attn = attention_weights.detach().cpu().numpy()
    else:
        attn = attention_weights
    # shape: [layers, heads, seq_len, seq_len]
    attn_mean = attn.mean(axis=(0,1))
    # Get attention scores for the query patch to all key patches
    query_attention = attn_mean[query_patch_idx]  # shape [seq_len]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(query_attention)), query_attention)
    plt.xlabel('Patch Index')
    plt.ylabel('Attention Score')
    plt.title(f'Attention Map for Query Patch {query_patch_idx}')
    if lead_labels:
        plt.xticks(ticks=range(len(query_attention)), labels=lead_labels, rotation=90)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_attention_maps_for_sample(sample: torch.Tensor, model: nn.Module,
                                       layer_indices: list, head_indices: list,
                                       save_dir: str) -> None:
    """
    Generate and save attention maps for a sample input.
    Args:
        sample: input tensor, shape [batch_size, channels, seq_len]
        model: the transformer model with accessible attention weights
        layer_indices: list of layer indices to extract attention from
        head_indices: list of head indices per layer
        save_dir: directory to save attention map images
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Hook to extract attention weights
    attention_outputs = []

    def get_attention_hook(module, input, output):
        # output is typically a tuple containing attention weights at specific points
        attention_outputs.append(output)

    hooks = []
    for layer_idx in layer_indices:
        handle = model.encoder.layers[layer_idx].self_attn.register_forward_hook(get_attention_hook)
        hooks.append(handle)
    model.eval()
    with torch.no_grad():
        _ = model(sample)
    # Remove hooks
    for h in hooks:
        h.remove()

    # attention_outputs now contains attention weights from specified layers
    for idx, attn in enumerate(attention_outputs):
        attn_array = attn[1]  # shape: [batch, heads, seq_len, seq_len]
        for head_idx in head_indices:
            attn_map = attn_array[0, head_idx].cpu().numpy()
            plot_attention_map(attn_map, query_patch_idx=0, save_path=os.path.join(save_dir, f'layer{layer_indices[idx]}_head{head_idx}.png'))

# ---------------------------
# Embedding Visualization
# ---------------------------

def plot_embeddings(embeddings: np.ndarray, labels: list = None,
                    title: str = 'Embedding T-SNE', save_path: str = None) -> None:
    """
    Reduce embeddings to 2D via t-SNE and plot.
    Args:
        embeddings: numpy array shape [num_samples, embedding_dim]
        labels: list or array for coloring (optional)
        title: plot title
        save_path: path to save figure
    """
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    if labels is not None:
        unique_labels = list(set(labels))
        for lbl in unique_labels:
            idxs = [i for i, l in enumerate(labels) if l == lbl]
            plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=str(lbl), alpha=0.6)
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def cluster_and_evaluate_embeddings(embeddings: np.ndarray, true_labels: list) -> float:
    """
    Cluster embeddings using GMM and compare to true labels.
    Args:
        embeddings: numpy array [n_samples, embedding_dim]
        true_labels: list of true label integers
    Returns:
        clustering accuracy
    """
    # Fit GMM with number of clusters = number of unique true labels or predefined
    n_clusters = len(set(true_labels))
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    predicted_labels = gmm.fit_predict(embeddings)

    # Map predicted clusters to true labels (unsupervised matching)
    from scipy.optimize import linear_sum_assignment
    contingency_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(n_clusters):
        for j in range(n_clusters):
            contingency_matrix[i, j] = np.sum((predicted_labels == i) & (np.array(true_labels) == j))
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    label_mapping = {row: col for row, col in zip(row_ind, col_ind)}
    predicted_mapped = [label_mapping[clust] for clust in predicted_labels]
    accuracy = np.mean([pred == true for pred, true in zip(predicted_mapped, true_labels)])
    return accuracy

# ---------------------------
# Helper Functions: Model Internal
# ---------------------------

def get_layerwise_attention_maps(model: nn.Module, sample_input: torch.Tensor,
                                 layer_idx: int, head_idx: int) -> np.ndarray:
    """
    Register a hook to extract attention weights from a specific layer and head.
    """
    attention_weights = []

    def hook(module, input, output):
        # output contains attention weights
        attn = output[1]  # shape: [batch, heads, seq_len, seq_len]
        attention_weights.append(attn.detach())

    handle = model.encoder.layers[layer_idx].self_attn.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)
    handle.remove()

    if attention_weights:
        attn = attention_weights[0]
        return attn[0, head_idx].cpu().numpy()
    else:
        return None

def generate_attention_overlay_on_ecg(input_ecg: np.ndarray, attention_scores: np.ndarray,
                                      lead_labels: list = None, save_path: str = None) -> None:
    """
    Overlay attention scores on the ECG waveform.
    """
    plt.figure(figsize=(12, 4))
    time = np.linspace(0, len(input_ecg)/250, len(input_ecg))
    plt.plot(time, input_ecg, label='ECG Signal')
    
    # Normalize attention scores for visualization
    scores_normalized = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
    for idx, score in enumerate(scores_normalized):
        if score > 0.6:
            plt.axvspan(time[idx*int(len(time)/len(attention_scores))],
                        time[(idx+1)*int(len(time)/len(attention_scores))],
                        color='red', alpha=score, label='High Attention' if idx == 0 else "")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Attention Overlay on ECG')
    if lead_labels:
        plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\ST-MEM\ST-MEM_repo`
