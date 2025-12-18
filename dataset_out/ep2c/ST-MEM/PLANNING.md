# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Below is a comprehensive, detailed plan structured around the key aspects of the methodology, experimental setup, datasets, hyperparameters, and evaluation protocols described in the paper. This roadmap will serve as a solid foundation for the subsequent coding phase.

---

# Reproducibility Plan for "Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram"

## 1. Understanding the Core Methodology

### 1.1. Goal
- Self-supervised pretraining of ECG representations, specifically designed to capture spatio-temporal relationships inherent in multi-lead ECG signals.
- Key innovation: Masked electrocardiogram modeling (ST-MEM), with explicit consideration of spatial (lead-wise) and temporal relationships.

### 1.2. Model Architecture
- **Encoder:**
  - Based on Vision Transformer (ViT) style, adapted for 12-lead ECG data.
  - Inputs: Patchified ECG signals with positional embeddings.
  - Patch size: 32 samples (~128 ms at 250Hz).
  - Number of patches: 128 per example.
  - Embeddings:
    --Patch embeddings: Each patch is linearly projected into embedding space.
    -Lead embeddings: Learned per lead to incorporate lead-specific info.
    -Positional embeddings: Learned positional encodings for temporal patches.
  - Multiple layers (variable depth tested): 1, 4, 8, 11 transformer blocks.
  - Attention heads: 4, with a total of 64 embedding dimensions per head.
- **Shared Decoder:**
  - Based on a transformer decoder architecture, with lead-wise shared decoder and lead embedding modules.
  - Takes encoded representations plus lead info to reconstruct masked patches.
  - Masks certain patches during training dynamically.
- **Outcome:**
  - Self-supervised learning by reconstructing masked patches.
  - Resulting representations are then used for downstream classification tasks.

### 1.3. Masking Strategy & Data Augmentation
- Masking ratio: 15%, 30%, 45%, 60%, 75%, 90%.
- Masking types:
  1. Erase: Randomly zero out certain patches.
  2. Flip: Invert signs of selected patches.
  3. Drop: Zero out patches completely.
  4. Cutout: Zeroing out a continuous interval.
  5. Sine wave: Add sinusoidal noise/artifacts.
  6. Partial sine: Add sine wave to part of a patch.
  7. White noise: Add Gaussian noise to patches.
- Data augmentation during pretraining should be applied to increase robustness.
- In experiments, masking ratios are critical hyperparameters.

### 1.4. Loss Function
- Reconstruction loss: Mean squared error or similar (not explicitly stated, but typical in masked autoencoding).
- Distance between unmasked patches and original patches.
- Possibly include positional and lead embedding consistency terms.

## 2. Dataset and Data Preparation

### 2.1. Training Data
- **Pretraining:**
  - Combine PTB-XL, CPSC2018, PhysioNet2017, and HRV data collections.
  - Use all available ECG signals (total combined: over 172,000 ECGs).
  - Sample rate: 250Hz.
  - Data partitioning:
    - For pretraining, use a large, unlabeled set (e.g., entire PTB-XL + others).
    - No labels used in pretraining.
- **Downstream tasks:**
  - Arrhythmia classification (e.g., MACE, MI, arrhythmia types, etc.).
  - Lead-wise classification (single-lead, 6-lead, 12-lead subsets).
  - Human activity classification via HRV dataset.

### 2.2. Data Processing
- Resample all ECG signals to 250Hz.
- Segment signals into 10s windows (or appropriate duration as in the paper).
- For training (pretraining):
  - Normalize signals (amplitude scaling).
  - Patchify: Divide into non-overlapping patches of 32 samples.
  - During training, randomly mask patches based on mask ratio.

### 2.3. Data Augmentation
- Apply the specified augmentation methods randomly during self-supervised training:
  - Erase: Zero patches.
  - Flip: Multiply patch by -1.
  - Drop: Zero portions.
  - Sine wave: Add sine noise.
  - Partial sine: Add to part.
  - White noise: Gaussian noise.
- For downstream evaluation, do not apply augmentation during inference.

## 3. Model Hyperparameters & Architecture Details
- **Transformer Encoder:**
  - Number of layers (depth): test 1, 4, 8, 11.
  - Number of heads: 4.
  - Embedding dimension per head: 64.
  - Patch embedding dimension: 64 or 128 (not explicitly specified, assume 64).
  - Positional embeddings: Learned per patch (up to 128 patches).
  - Lead embeddings: Learned, one per lead (size consistent with number of leads, e.g., 12, 6, 1).
- **Decoder:**
  - Layer depth (consistent with encoder for ablation): 1, 4, 8, 11.
  - Number of heads: 4 (or matching encoder).
  - Embedding dims: 64.
  - Cross-attention with encoder outputs.
- **Training:**
  - Batch size: 1024.
  - Epochs: 800.
  - Optimizer: AdamW.
  - Learning rate: cosine decay, initial 0.0012.
  - Warm-up steps: 40.
- **Fine-tuning:**
  - Learning rate decay, small (0.001 → 0.0001).
  - Epochs: 100.
  - Batch size: 1024.
  - Input: Encoded features from pretrained model, possibly with small classifier head.

## 4. Training Procedure

### 4.1. Pretraining Step
- Initialize model components (encoder, decoder, embeddings).
- For each batch:
  - Sample ECG signals.
  - Apply random augmentation and masking.
  - Patchify.
  - Add lead and positional embeddings.
  - Forward pass through encoder.
  - Mask patches (simulate missing segments).
  - Decoder reconstructs masked patches.
  - Compute reconstruction loss.
- Backpropagate and optimize for 800 epochs.
- Save the pretrained encoder weights.

### 4.2. Fine-tuning & Evaluation
- Load pretrained encoder.
- Add classification head(s) (fully-connected layers) suitable for downstream tasks.
- Fine-tune on labeled datasets:
  - Use standard class labels (e.g., arrhythmia types, MI, HRV activity classes).
- Hyperparameters for fine-tuning:
  - Smaller learning rate, fewer epochs (~100).
  - Dropout, weight decay as needed.
- Evaluate:
  - AUROC for downstream classification.
  - Report on validation and test sets.
  - Experiments with different lead combinations and data regimes.

## 5. Evaluation Metrics & Analysis

### 5.1. Metrics
- Primary: AUROC (Area Under ROC Curve).
- Additional: F1 score, accuracy (for balanced classes).
- For representation analysis:
  - t-SNE visualization (as shown in paper).
  - Clustering accuracy (e.g., distinguishing lead types, rhythms).
  - Attention map visualizations.

### 5.2. Ablation and Hyperparameter Analysis
- Vary decoder depth.
- Vary masking ratios.
- Compare lead-wise vs lead-agnostic decoder.
- Use different augmentation strategies.
- Test with various lead subsets (full 12-lead, 6-lead, single-lead).

## 6. Implementation Considerations & Gaps
- **Details not explicitly provided:**
  - Exact loss function (assumed mean squared error).
  - Positional embeddings initialization.
  - Decoder architecture specifics: number of layers, whether causal masking etc.
  - Exact sizes of embeddings (assumed 64).
  - How lead embeddings are integrated: addition, concatenation, or another method (assumed addition).
- **Assumption:**
  - Use standard transformer blocks (multi-head attention + MLP + layer norm, residuals).
  - Reconstruction targets are the original patches before masking.
- **Code modularity:**
  - Encapsulate patchify, mask, lead and positional embedding modules.
  - Implement flexible transformer encoder/decoder with variable depth.
  - Data pipeline accommodating multiple datasets with standard preprocessing.

---

# Summary of Action Items

1. Data collection and preprocessing:
   - Gather datasets, resample, segment, normalize.
   - Implement patchify, masking, and augmentations.
2. Model architecture:
   - Build ViT-like encoder with lead and positional embeddings.
   - Build transformer decoder with cross-attention.
3. Training:
   - Implement masking strategies, loss calculation, optimizer setup.
   - Run pretraining for many epochs, save checkpoints.
4. Fine-tuning:
   - Attach classification heads, run hyperparameter search.
   - Evaluate on downstream classification tasks.
5. Analysis:
   - Visualize embeddings, attention maps.
   - Perform ablation studies on depth, masking ratio, lead modules.
   
This plan aims to replicate the methodology faithfully while highlighting critical hyperparameters, architectural details, and experimental strategies. As you proceed, focus on iteratively refining the implementation, especially where the paper lacks explicit details, possibly informed by standard transformer and masked autoencoder practices.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular deep learning system using PyTorch and Hugging Face's Transformers library. The system will include data loaders, a ViT-inspired ECG encoder, a transformer-based decoder, data augmentation modules, and training routines for self-supervised masked ECG modeling. The design emphasizes flexibility in hyperparameters such as decoder depth, number of transformer layers, masking ratios, and augmentation methods, enabling thorough experimentation as described in the paper.",
    "File list": [
        "main.py",
        "datasets.py",
        "model.py",
        "trainer.py",
        "utils.py",
        "evaluation.py",
        "visualization.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class ECGDataset {\n        +__init__(file_paths: List[str], label_paths: Optional[List[str]], mode: str)\n        +__len__() -> int\n        +__getitem__(index: int) -> dict\n        +apply_augmentations(data: Tensor) -> Tensor\n    }\n    class ECGPretrainer {\n        +__init__(encoder: TransformerEncoder, decoder: TransformerDecoder, dataset: ECGDataset, config: dict)\n        +train() -> None\n        +save_checkpoint(path: str) -> None\n        +load_checkpoint(path: str) -> None\n    }\n    class TransformerEncoder (torch.nn.Module) {\n        +__init__(num_layers: int, embed_dim: int, n_heads: int, patch_size: int, num_patches: int)\n        +forward(x: Tensor, lead_embeddings: Tensor, position_embeddings: Tensor) -> Tensor\n    }\n    class TransformerDecoder (torch.nn.Module) {\n        +__init__(num_layers: int, embed_dim: int, n_heads: int)\n        +forward(encoded: Tensor, mask: Tensor, lead_embeddings: Tensor, position_embeddings: Tensor) -> Tensor\n    }\n    class MaskingStrategy {\n        +apply_mask(patches: Tensor, mask_ratio: float) -> Tensor\n        +apply_augmentation(patches: Tensor, method: str) -> Tensor\n    }\n    class Evaluation {\n        +__init__(model: torch.nn.Module, dataset: ECGDataset, config: dict)\n        +evaluate() -> dict\n    }\n    Main --> ECGDataset\n    Main --> ECGPretrainer\n    ECGPretrainer --> TransformerEncoder\n    ECGPretrainer --> TransformerDecoder\n    ECGPretrainer --> MaskingStrategy\n    ECGPretrainer --> utils\n    Main --> Evaluation\n    Evaluation --> TransformerEncoder\n    Evaluation --> TransformerDecoder\n    Utility modules include data augmentations, plotting, and metric calculation; interfaces are defined via functions in utils.py.",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant D as Dataset\n    participant P as Pretrainer\n    participant E as Evaluation\n    M->>D: Load datasets (train, val, test)\n    activate D\n    D-->>M: Return dataset objects\n    deactivate D\n    M->>P: Initialize with encoder, decoder, dataset, configs\n    P->>P: Load checkpoint if exists\n    P->>P: For epoch in range(max_epochs):\n        P-->D: Sample batch\n        D->>D: Apply augmentations and masks\n        D->>P: Return batch patches\n        P-->P: Forward pass: encoder + decoder\n        P-->P: Compute reconstruction loss\n        P->>P: Backpropagate and optimize\n    P->>P: Save checkpoint after training\n    M->>E: Load trained encoder model\n    E->>E: Run evaluation on test set\n    E-->>M: Return metrics (AUROC, F1)\n    Note over M, D: Repeat with different seed/splits for reproducibility.",
    "Anything UNCLEAR": "Details on the exact loss function (assumed to be mean squared error), precise dimension of lead embeddings, decoder architecture specifics (number of layers, layer types), and initializations (e.g., positional embeddings, lead embeddings). Also, clarification on the exact augmentation pipeline (sequence, probabilities, and parameters). Need confirmation whether the decoder shares parameters across leads or is lead-specific, and on the input normalization/preprocessing specifics for datasets."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.9.0",
        "numpy==1.21.0",
        "scipy==1.7.0",
        "scikit-learn==0.24.2",
        "matplotlib==3.4.3",
        "tqdm==4.62.3",
        "transformers==4.11.3",
        "hydra-core==1.1.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "datasets.py",
            "Define ECGDataset class for loading, preprocessing, segmentation, and augmentation. Handles dataset partitioning, normalization, patchify, masking, and augmentation strategies. Depends on numpy, scipy, and scikit-learn."
        ],
        [
            "model.py",
            "Define TransformerEncoder and TransformerDecoder classes, inspired by ViT and Masked Autoencoder architectures. Includes lead embeddings, positional embeddings, and flexible transformer layers. Uses torch.nn modules, torch.nn.functional, and transformers."
        ],
        [
            "trainer.py",
            "Implement ECGPretrainer class for self-supervised training. Handles batching, masking, forward passes through encoder and decoder, loss calculation, optimizer steps, checkpointing, and epoch control. Depends on datasets.py and model.py."
        ],
        [
            "evaluation.py",
            "Define Evaluation class for downstream task evaluation including AUROC, F1, and embedding visualization. Uses sklearn metrics, matplotlib for plotting, and loads trained models. Depends on trained encoder and decoder."
        ],
        [
            "utils.py",
            "Utility functions for data augmentation (erase, flip, sine noise, etc.), plotting attention maps, visualization of embeddings, and tensor operations. Shared across dataset and training modules."
        ],
        [
            "main.py",
            "Coordinate the full experiment pipeline: dataset loading, model initialization, training, checkpoint management, and evaluation. Configurable via Hydra or argparse, depends on all modules."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "utils.py",
        "datasets.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Hyperparameters such as masking ratios, transformer layer counts, embedding sizes, augmentation strategies, and dataset preprocessing steps are shared across modules. Dataset-specific label info used in downstream classification tasks.",
    "Anything UNCLEAR": "Clarification needed on the exact loss function used for reconstruction, initializers for positional and lead embeddings, and whether the decoder shares parameters across leads or is lead-specific."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0012
  batch_size: 1024
  epochs: 800
  optimizer: AdamW
  warmup_steps: 40
  scheduler: cosine_decay
  total_epochs: 800

pretraining:
  mask_ratio: 0.15  # Default masking ratio; can experiment with 0.3, 0.45, etc.
  patch_size: 32
  patch_count: 128
  decoder_layers: 4  # Default; tested also as 1, 8, 11
  decoder_heads: 4
  decoder_embed_dim: 64
  encoder_layers: 8  # Default; test 1, 4, 8, 11 for ablation
  encoder_heads: 4
  encoder_embed_dim: 64
  max_epochs: 800

dataset:
  sampling_rate: 250  # Hz
  segment_duration: 10  # seconds
  normalization: true
  dataset_paths:
    PTBXL: path/to/PTB-XL
    CPSC2018: path/to/CPSC2018
    PhysioNet2017: path/to/PhysioNet2017
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1

augmentation:
  types: ["erase", "flip", "drop", "sine_wave", "partial_sine", "white_noise"]
  ratios: [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
  parameters:
    erase:
      probability: 0.2
    flip:
      probability: 0.2
    drop:
      probability: 0.2
    sine_wave:
      frequency_range: [0.67, 40]  # Hz
    partial_sine:
      frequency_range: [0.67, 40]
      ratio: 0.5  # Part of patch
    white_noise:
      noise_std: 0.05

model:
  lead_embedding_dim: 64
  positional_embedding_dim: 64
  lead_count: 12
  patch_size: 32
  num_patches: 128
  encoder_layers: 8
  encoder_heads: 4
  encoder_embed_dim: 64
  decoder_layers: 4
  decoder_heads: 4
  decoder_embed_dim: 64
  dropout_rate: 0.1

evaluation:
  downstream_tasks:
    arrhythmia_classification: true
    myocardial_infarction: true
    rhythm_detection: true
  metrics: [AUROC, F1]
  visualization: attention_maps, embeddings
```

---

**Note:** Full configuration is available in `planning_config.yaml`
