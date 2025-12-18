# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## datasets.py

**Logic Analysis for datasets.py — Definition of ECGDataset class**

**Objective:**  
Implement a comprehensive `ECGDataset` class that manages loading, preprocessing, partitioning, patchifying, augmentation, masking, and sample retrieval for ECG data, according to the methodology described in the paper. This class supports both pretraining (self-supervised masked reconstruction) and downstream (classification) purposes, with flexibility to handle multiple datasets and label types.

---

### 1. **Class Structure & Initialization**

- **Inputs:**
  - `file_paths`: List of paths to ECG recordings or a directory housing such files.
  - `label_paths` (optional): Paths for labels corresponding to each ECG, used for downstream supervised tasks.
  - `mode`: `'train'`, `'val'`, `'test'`, or `'pretrain'`, indicating dataset purpose.
  - `config`: Configuration dictionary for hyperparameters and processing options.

- **Attributes:**
  - Store loaded ECG data, labels.
  - Copy of hyperparameters for processing (e.g., sampling rate, segment duration, patch size, masking ratios).
  - Preprocessing options: normalization, patch parameters, augmentation flags.

- **Procedures:**
  - Load raw ECG data from paths.
  - Resample to 250 Hz (per the dataset specification).
  - Segment signals into 10-second windows (if longer than 10s; discard shorter).
  - Normalize signals (Z-normalization), if specified.
  - Associate labels for downstream; for pretraining, labels are optional or not used.

---

### 2. **Data Loading & Preprocessing**

- **Loading records:**
  - Read raw signals (e.g., via `scipy.io.loadmat`, `mne`, or raw numpy arrays).
  - Standardize units (e.g., convert all to microvolts or millivolts as needed).
  - Resample signals to 250 Hz (`scipy.signal.resample` or `resample_poly`).

- **Segmenting:**
  - Extract non-overlapping 10s segments:
    - Duration in samples = `segment_duration * sampling_rate` (e.g., 2500 samples).
    - For signals longer than 10s, crop; discard signals shorter than 10s.
    - For longer signals, crop into multiple segments if needed.
  
- **Label filtering:**
  - For classification, discard signals with multiple labels (if multi-label filtering is required).
  - For other tasks, retain multiple labels as needed.

- **Normalization:**
  - Apply Z-normalization: subtract mean, divide by standard deviation, per signal.

- **Partitioning:**
  - For training, validation, testing:
    - Split accordingly (`train_split`, `val_split`, `test_split`).
    - Support random shuffling for training.
  
---

### 3. **Patchification**

- **Patch configuration:**
  - Input length: signal samples per segment (~2500 samples).
  - Patch size: 32 samples (~128ms).
  - Number of patches per segment:
    - `T` in samples (`T=2500`)
    - Patches: `T / patch_size` = 78.125 → round down or truncate to 78 patches.
  - For the paper, `max_patches` is set to 128 (support padding if needed).
  - Zero-pad or truncate signals to fit exactly `128 * 32 = 4096` samples if necessary (or just use truncated signals) to match the `num_patches` parameter.

- **Patch extraction:**
  - Use striding or reshaping to divide the signal into non-overlapping patches.
  - For multi-lead:
    - Arrange data as `(L, T)`.
    - For each lead, divide into patches, resulting in shape `(L, n_patches, p)`.
  - Store patches in a tensor: shape `(L, n_patches, p)`.

---

### 4. **Augmentation & Masking (for pretraining)**

- **When in pretraining mode:**
  - Randomly select masking ratio (from config or supplied).
  - Apply augmentations:
    - For each patch or lead, probabilistically apply augmentations (`erase`, `flip`, `drop`, `sine_wave`, `partial_sine`, `white_noise`).
    - Augmentations can be sequential or randomly chosen.
  - Generate a mask:
    - Randomly select percentage of patches (`mask_ratio`) to mask.
    - Mark selected patches as masked; leave others unmasked.
    - Create a boolean mask tensor, shape `(L, n_patches)`.
  - Return:
    - Patches (with augmentations applied if needed),
    - Mask tensor,
    - Original patches (for loss computation).

- **In downstream classification mode:**
  - Skip masking and augmentation.
  - Use full patches for feature extraction/classification.

---

### 5. **Sample Retrieval (`__getitem__`)**

- For each sample:
  - Load the corresponding preprocessed ECG signal.
  - Segment, normalize, and patchify.
  - If in pretraining mode:
    - Apply augmentations.
    - Apply masking based on the mask ratio.
    - Package:
      - Patches,
      - Mask pattern,
      - Lead info (lead index or embedding indicator).
  - If in downstream mode:
    - Return patches as-is (without mask).
    - Include label(s) for supervised training/evaluation.
  - For data augmentation or mode-specific processing, apply procedures accordingly.

- Output data as a dictionary:
  - `'patches'`: tensor `(L, n_patches, p)` (or `(n_patches, p)` per lead or combined, depending on implementation).
  - `'mask'`: boolean tensor `(L, n_patches)` (only in pretraining).
  - `'label'`: tensor (classification labels) if available.
  - `'lead_idx'`: lead identifier for lead embedding purposes.

---

### 6. **Additional Considerations & Edge Cases**

- **Multiple leads and labels:**
  - Handle arbitrary number of leads; supports subsets (e.g., limb leads, single lead).
- **Zero padding:**
  - For signals shorter than segment duration, discard or pad with zeros.
- **Efficiency:**
  - Cache processed signals if possible.
  - Use numpy/scipy for fast slicing and resampling.
- **Parallelization:** 
  - Data loading via `__getitem__` should be efficient with possible multi-threaded DataLoader.

---

### 7. **Implementation Output**

- Fully implement the class with methods:
  - `__init__()`: load data, initialize parameters.
  - `__len__()`: dataset size.
  - `__getitem__(index)`: return appropriately processed sample.
  - Internal methods:
    - `_load_data()`: load raw data.
    - `_segment_signal()`: segment and truncate/pad.
    - `_normalize()`: apply Z-normalization.
    - `_patchify()`: divide signals into patches.
    - `_apply_mask()`: mask patches randomly.
    - `_apply_augmentations()`: based on config, apply augmentations.
- Make sure to align the returned data format with the model's expected input.  
- Document assumptions explicitly, especially regarding loss functions, input dimensions, and augmentation.

---

This comprehensive logic analysis ensures a systematic, transparent, and reproducible implementation aligned with the paper's methodology and experimental design, forming a solid foundation for the coding phase.

## evaluation.py

# Evaluation.py Logic Analysis

## Purpose
Implement the `Evaluation` class responsible for evaluating downstream ECG classification models and analyzing learned representations. This includes calculating metrics like AUROC and F1, visualizing embeddings via t-SNE, and examining attention maps. The class should be flexible to handle different downstream tasks specified in configuration, utilizing pretrained encoder (and decoder if needed), with appropriate metric computation and visualization functions.

---

## Core Components

### 1. Initialization (`__init__`)
- **Arguments:**
  - `model`: trained Encoder model for downstream classification.
  - `dataset`: Dataset object for test/validation data.
  - `config`: hyperparameters and evaluation settings loaded from the config file.
- **Tasks:**
  - Load models (encoder, optionally decoder, if visualization of attention is required).
  - Store dataset, evaluation metrics, visualization flags.
  - Prepare data loader(s) for evaluation.
  - Initialize metrics and visualization directories.

### 2. Metrics Calculation
- **Metrics:**
  - AUROC (Area Under ROC Curve): suitable for multi-class, compute macro or mean AUROC.
  - F1 Score: macro average, considering class imbalance.
- **Implementation details:**
  - Use `sklearn.metrics.roc_auc_score` with `multi_class='ovr'` or `'macro'`.
  - Use `sklearn.metrics.f1_score`.

### 3. Evaluation Procedure (`evaluate()`)
- **Steps:**
  - Loop over the evaluation DataLoader:
    - Get input data `X`.
    - Forward pass through `model` (likely just the encoder for feature extraction, or classifier head).
    - Collect predicted logits/probabilities.
    - Collect true labels.
  - After iteration:
    - Compute metrics (AUROC, F1) for the entire evaluation set.
    - Log or print results with mean ± std if multiple runs.
- **Output:**
  - Dictionary with metrics: AUROC, F1.
  - Possibly return predicted probabilities for further analysis.

### 4. Embedding Visualization (`visualize_embeddings()`)
- **Purpose:**
  - Use t-SNE to reduce high-dimensional embeddings into 2D.
  - Plot embeddings colored by class labels or lead types.
  - Save plots to output directory.
- **Steps:**
  - Extract embeddings:
    - Pass the entire dataset (or subset) through the encoder.
    - Collect embedding vectors.
  - Use `sklearn.manifold.TSNE` to reduce to 2D.
  - Plot with `matplotlib`, label points by class or lead.
  - Save figures.

### 5. Attention Map Visualization (`visualize_attention_maps()`)
- **Purpose:**
  - Visualize self-attention weights for interpretability, similar to Figure 5 or 8 in the paper.
- **Steps:**
  - Load a sample ECG signal.
  - Forward it through the Transformer encoder (possibly extracting attention weights):
    - Requires encoder to output attention scores (modify the encoder to store attention layers).
  - Map attention weights onto patches:
    - Determine which patches correspond to which parts of the signal.
  - Generate heatmaps overlayed on the ECG or as matrices.
  - Save attention maps.

### 6. Handling Multiple Tasks/Configurations
- **Flexibility for Tasks:**
  - Use `config.evaluation.downstream_tasks` flags:
    - e.g., `arrhythmia_classification`, `myocardial_infarction`, `rhythm_detection`.
  - Load relevant labels and metrics accordingly.
  - Possibly have separate functions per task for specific processing.

### 7. Model & Data Loading
- **Model:**
  - Load pretrained encoder.
- **Dataset:**
  - For evaluation, datasets leading to test/validation sets.
  - Include ground-truth labels as available.
- **DataLoader:**
  - Batch data with appropriate batch size (`config.training.batch_size`).

### 8. Metrics and Visualizations Output
- **Metrics Report:**
  - Return dict with all relevant metrics.
- **Figures:**
  - Save visualizations with descriptive filenames.
  - Visualizations include:
    - Embedding scatter plots.
    - Attention heatmaps.

### 9. Utility Functions
- Use existing utility functions from `utils.py` for:
  - Computing metrics.
  - t-SNE visualization (`plot_embeddings()`).
  - Attention map visualization (`plot_attention()`).
  
---

## Additional Considerations

### 1. Logging and Reproducibility
- Log metrics, parameters, and random seeds.
- Save the plot figures.
- Ensure deterministic behavior if needed.

### 2. Attention to Model Size & Resources
- Loading the model:
  - Use `torch.load()` with device map.
- Embedding extraction:
  - Pass sample inputs, extract last hidden states.
  - For attention maps, access attention weights if model supports.

### 3. Handling Variability
- For multi-class evaluation, handle class imbalance.
- For multi-task evaluation, integrate labels and metrics per task.

### 4. Default Behavior
- If no visualization requested, skip.
- If attention map extraction isn't possible, skip that step.

---

## Summary
The class `Evaluation` in `evaluation.py` should embody the following workflow:

- Initialization: set up model, dataset, parameters.
- Run evaluation: calculate predictions, metrics.
- Visualization:
  - Embeddings via t-SNE.
  - Attention maps for interpretability.
- Output: dictionary with metrics, save plots.
- Operations are conditioned on config flags for flexible execution.

By following this structured analysis, the implementation will ensure faithful, reproducible evaluation aligned with the paper's methods and experimental framework.

---

**End of Logic Analysis**

## main.py

# Logic Analysis for main.py

This script serves as the central coordinator to execute the ECG self-supervised pretraining and downstream evaluation pipeline based on the provided configuration. The main.py's responsibilities are to orchestrate dataset loading, model initialization, pretraining, checkpoint management, and evaluation, ensuring flexibility and reproducibility.

## 1. Framework and Dependencies
- Use PyTorch for model training and inference.
- Use Hydra or argparse for configuration management (selected Hydra based on reference configuration).
- Use existing modules:
  - datasets.py for dataset loading and preprocessing.
  - model.py for model architectures (TransformerEncoder and TransformerDecoder).
  - trainer.py for pretraining routines.
  - evaluation.py for downstream evaluation.
  - utils.py for augmentation, visualization, and auxiliary functions.

## 2. Configuration Handling
- Load the YAML config.yaml (or similar via Hydra).
- Extract key parameters:
  - Pretraining parameters (mask ratio, patch size, number of patches, layers, heads, embedding dims).
  - Training hyperparameters (learning rate, batch size, epochs, warmup steps, scheduler).
  - Dataset paths and split ratios.
  - Augmentation strategies and probabilities.
  - Model hyperparameters (lead embeddings, dropout).
  - Evaluation flags (which tasks, metrics, visualization options).

## 3. Dataset Preparation
- Instantiate datasets using datasets.py:
  - Load datasets: PTB-XL, CPSC2018, PhysioNet2017.
  - Perform preprocessing:
    - Resample signals to 250Hz.
    - Segment signals into 10s windows.
    - Normalize signals if enabled.
  - Implement train/validation/test splits:
    - Use dataset paths and splits from config.
    - Shuffle and subset as required.
- For pretraining:
  - Prepare an unlabeled dataset combining all datasets.
  - Ensure dataset returns raw signals as tensors for patchify.
- For downstream evaluation:
  - Load appropriate labeled datasets with annotations.
  - Instantiate dataset objects for train/validation/test with labels.

## 4. Model Initialization
- Instantiate the Transformer encoder:
  - Set number of encoder layers, heads, embed_dim as per config.
  - Include lead embeddings and positional embeddings.
- Instantiate the Transformer decoder:
  - Set number of decoder layers, heads, embed_dim.
  - Probably share the decoder across leads but process lead-specific patches.
- Initialize lead-specific embedding modules and positional embeddings.
- For self-supervised training:
  - Initialize the masking module with mask_ratio.
  - (Optional) implement augmentation pipeline within utils.
- For downstream classification:
  - Define classifier head (fully connected layers, softmax for multi-class or sigmoid for multi-label tasks).
  - Freeze or fine-tune encoder based on training phase.

## 5. Pretraining Routine
- If checkpoint exists, load pretrained weights.
- Loop over epochs (max epochs from config):
  - For each batch:
    - Fetch raw ECG signals.
    - Apply data augmentations (from utils.py).
    - Patchify signals into non-overlapping patches of size 32.
    - Randomly select patches to mask according to mask_ratio.
    - Prepare lead IDs, positional IDs.
    - Forward pass:
      - Encoder processes unmasked patches.
      - Aggregate encoded patches.
      - Decoder reconstructs masked patches from encoder output plus lead and positional embeddings.
    - Calculate the reconstruction loss (likely MSE).
    - Backpropagate and update parameters.
  - Adjust learning rate scheduler.
  - Save checkpoint periodically or at the end.

## 6. Checkpoint Management
- During training, periodically save model weights (encoder + decoder).
- On resume, load latest checkpoint.

## 7. Downstream Finetuning & Evaluation
- After pretraining:
  - Extract the encoder weights.
  - Attach the classification head.
  - For each downstream task:
    - Use labeled dataset (train/val/test).
    - Fine-tune encoder + classifier:
      - Set appropriate learning rate.
      - Configure training epochs (~100).
      - Possibly freeze encoder initially.
    - Log metrics: AUROC, F1-score.
    - Save best models based on validation metrics.
- Perform evaluation on the test set.
- Generate visualization:
  - Attention maps.
  - Embedding distributions (t-SNE plots).
  - Clustering accuracy (for spatial embedding analysis).

## 8. Repeatability & Experiment Variations
- Vary key hyperparameters (layers, heads, masking ratio).
- Save logs and models for each configuration.
- Document different runs for ablation studies.

## 9. Additional Considerations
- Use seed for reproducibility.
- Utilize torch.device() to support GPU acceleration.
- Handle exceptions gracefully (e.g., dataset errors, file not found).
- Prepare command-line interface to accept different configs or experiment parameters.
- Log progress, metrics, and parameters with print statements or logging.

## 10. Summary Tasks
- [ ] Load configuration.
- [ ] Initialize datasets.
- [ ] Instantiate models.
- [ ] Run pretrained model training loop.
- [ ] Save/load checkpoints.
- [ ] Conduct downstream training/evaluation.
- [ ] Visualize results.
- [ ] Save models and logs reproducibly.

---

This detailed analysis ensures that main.py can be designed to systematically and reproducibly execute the entire experimental pipeline as described in the paper, following the given design and configuration instructions.

## model.py

{
  "model.py": [
    "Overview:\n- Implement two main classes: TransformerEncoder and TransformerDecoder, both subclassing torch.nn.Module.\n- They are designed to process ECG patches in a self-attention transformer manner.\n- Incorporate lead embeddings, positional embeddings, and flexible transformer layers.\n- Support variable depth (number of layers) and heads for experimental ablation.\n\n1. TransformerEncoder:\n- Initialization:\n  - Accepts parameters: num_layers (e.g., 8), embed_dim (e.g., 64), n_heads (e.g., 4), patch_size (e.g., 32), num_patches (e.g., 128), dropout_rate.\n  - Initialize positional embeddings: a learnable tensor of shape [num_patches, embed_dim].\n  - Initialize lead embeddings: a learnable tensor of shape [lead_count, embed_dim].\n  - Instantiate a list of transformer encoder layers (e.g., nn.TransformerEncoderLayer) or build a custom torch.nn.Module list, each with multi-head self-attention and feed-forward.\n- Forward Pass:\n  - Inputs:\n    - x: tensor of shape [batch_size, total_patches, embed_dim], representing patch embeddings concatenated for all leads.\n    - lead_ids: tensor of shape [batch_size, total_patches], indicating from which lead each patch originates.\n    - position_ids: optional, if not passed, use positional embeddings.\n  - Add lead embeddings: for each patch, add the corresponding lead embedding.\n  - Add positional embeddings: add positional encodings based on patch position.\n  - Pass through transformer layers sequentially, with residual connections and layer norm.\n  - Output: encoded representations, shape [batch_size, total_patches, embed_dim].\n\n2. TransformerDecoder:\n- Initialization:\n  - Accepts parameters: num_layers (e.g., 4), embed_dim, n_heads.\n  - Build transformer decoder layers, possibly nn.TransformerDecoderLayer, with similar structure to encoder layers.\n- Forward Pass:\n  - Inputs:\n    - encoded: encoder output tensor [batch_size, unmasked_patches, embed_dim].\n    - masked_positions: tensor indicating positions of masked patches.\n    - lead_ids: as above, needed to inject lead-specific info.\n    - position_ids: to add positional info.\n    - (Optional) masks for cross-attention if needed.\n  - Incorporate lead embeddings again, adding to the input embeddings.\n  - Add positional embeddings.\n  - Pass through decoder layers sequentially.\n  - Output: reconstructions of the patches, shape [batch_size, total_patches, embed_dim].\n\n3. Embeddings:\n- Lead embeddings: learn per lead, shape [lead_count, embed_dim].\n- Positional embeddings: learnable, shape [num_patches, embed_dim].\n- Patch embeddings: obtained from linear projection of each patch—shape [batch_size, num_patches, embed_dim].\n- Mask embeddings: a shared learnable embedding, shape [embed_dim], used for masked patches in decoder.\n\n4. Transformer Layer Details:\n- Use torch.nn.TransformerEncoderLayer and nn.TransformerDecoderLayer or custom layers.\n- MultiHeadAttention with n_heads.\n- Feedforward network with hidden size 4× or 8× embed_dim for capacity.\n- Dropout applied as per dropout_rate.\n- Layer normalization after each sublayer.\n\n5. Layer Flexibility:\n- Allow options to instantiate transformer with variable number of layers and heads.\n- Use ModuleList to assemble layers dynamically based on input parameters.\n\n6. Forward Pass Flow Summary:\n- For Encoder:\n  - Receive patch embeddings, lead IDs.\n  - Add lead and positional embeddings.\n  - Sequentially process through encoder layers.\n  - Output last encoder layer’s output.\n- For Decoder:\n  - Use the encoder outputs and masked patches.\n  - Add lead and positional embeddings.\n  - Process through decoder layers.\n  - Reconstruct the masked patches.\n\n7. Additional Considerations:\n- Implement appropriate masking: during training, some patches are zeroed or replaced by a learnable mask embedding.\n- Require to handle batches of ECG data, with batch size dimension.\n- Ensure the output shape aligns with the original patches shape for loss calculation.\n- Include utility functions for initializing positional embeddings and lead embeddings.\n"
  ],
  "Note": "This detailed logic analysis guides the implementation of the transformer-based classes with attention to the modularity, parameterization, and data flow described in the paper and plan. Explicit support for variable depths, lead embeddings, positional encodings, and masking strategies is essential for fidelity and reproducibility."
}

## requirements.txt

# requirements.txt

# Core deep learning framework for model definition, training, and evaluation
torch==1.9.0

# Numerical operations, tensor manipulation, normalization
numpy==1.21.0

# Signal processing, filtering, signal normalization, and segmentation
scipy==1.7.0

# Metrics for evaluation: AUROC, F1 score, accuracy
scikit-learn==0.24.2

# Visualization for embedding analysis, attention maps, and training curves
matplotlib==3.4.3

# Progress bar for training loops
tqdm==4.62.3

# Transformer architecture implementation, including multi-head attention, layer norm, positional encoding
transformers==4.11.3

# Flexible configuration management, experiment control, hyperparameter setting
hydra-core==1.1.2

# Additional notes on implementation logic:

1. **Dataset Loading & Preprocessing:**
   - Implement `datasets.py` with an `ECGDataset` class that:
     - Loads raw ECG data from specified paths.
     - Resamples all signals to 250 Hz.
     - Crops signals to 10 seconds segments.
     - Normalizes signals via Z-normalization (subtract mean, divide by std).
     - Segments signals into non-overlapping patches of size 32 samples (~128 ms).
     - Handles data augmentation as specified (erase, flip, drop, sine wave, partial sine, white noise) with configurable probabilities.
     - Applies masking of patches dynamically during training with a specified ratio (e.g., 15%).

2. **Model Architecture:**
   - In `model.py`, define:
     - `TransformerEncoder`: a ViT-inspired module with adjustable number of layers (1, 4, 8, 11), each including multi-head self-attention, MLP, residual connections, layer normalization.
     - Embeddings:
       - Patch embeddings: linear projection of flattened patches (size: 32 samples).
       - Lead embeddings: learnable per lead (size: lead count x embedding_dim).
       - Positional embeddings: learnable per patch position (size: max number of patches).
     - `TransformerDecoder`: a transformer with configurable number of layers (default 4), sharing parameters across leads.
       - Cross-attention with encoder output.
       - Reconstruction head to output patches matching original signals.
   - Incorporate lead-wise shared decoder, lead embeddings, SEP embeddings.
   - Integrate positional embeddings as fixed or learned parameters.
   - Use dropout rates and layer normalization as indicated.

3. **Pretraining Routine (`trainer.py`):**
   - Initialize the encoder and decoder with specified hyperparameters.
   - For each epoch:
     - Loop over dataset batches.
     - For each batch:
       - Apply augmentation transforms as per configuration.
       - Segment signals into patches.
       - Apply random masking based on the ratio (default 15%, can vary).
       - Add lead embedding + positional embedding to each patch embedding.
       - Pass unmasked patches through the encoder.
       - Feed encoder output and masked patches into the decoder.
       - Compute reconstruction loss (mean squared error) between original and reconstructed patches of masked tokens.
       - Backpropagate and update parameters.
   - Save checkpoint after training.

4. **Fine-tuning & Downstream Evaluation (`evaluation.py`):**
   - Load encoder weights loaded from pretraining.
   - Freeze encoder or allow fine-tuning based on experiment.
   - Attach a linear classifier head (fully connected layer matching the number of classes).
   - Train classifier on labeled downstream dataset:
     - Use appropriate loss (cross-entropy).
     - Use standard optimizer (AdamW) with small learning rate.
   - Evaluate on validation/test sets:
     - Compute AUROC, F1-score.
     - Perform considerations for class imbalance (macro-averaged metrics).
   - For reduced lead sets:
     - Manipulate lead embeddings so that the model can process fewer leads.
     - Maintain input size by patching signals accordingly.
   
5. **Attention Map Analysis & Visualization (`visualization.py`):**
   - Extract self-attention weights from the encoder.
   - Generate attention maps for specific query patches.
   - Visualize embedding spaces with t-SNE.
   - Use GMM clustering for spatial relationship validation.
   
6. **Augmentation Logic:**
   - Implement augmentation functions with probability controls:
     - Erase: zero out entire patches.
     - Flip: invert signal polarity.
     - Drop: zero out segments within patches.
     - Sine noise addition: add sine wave noise into patches.
     - Partial sine: add sine wave to segments.
     - White noise: Gaussian noise addition.
   
7. **Configuration Management (`config.yaml`):**
   - Define all hyperparameters, dataset paths, and augmentation probabilities.
   - Use Hydra to load configurations and pass to modules.

8. **Execution flow:**
   - Main script (`main.py`) orchestrates:
     - Dataset loading and splitting
     - Model initialization
     - Pretraining loop
     - Saving/checkpointing
     - Loading weights for downstream fine-tuning
     - Evaluation and visualization
   
9. **Hyperparameters & Defaults:**
   - Mask ratio: 0.15 (experiments with variations)
   - Decoder layers: 4 (testing 1, 8, 11)
   - Embedding size: 64
   - Number of patches: 128
   - Learning rate: 0.0012 with cosine decay
   - Batch size: 1024
   - Epochs: 800 for pretraining, 100 for fine-tuning
   - Dropout rate: 0.1
   - Lead count: 12 (configurable for reduced lead scenarios)

---

**Summary of Logical Components:**

- Data loading with augmentation and patchifying.
- Masking and reconstruction as the self-supervised task.
- Transformer-based encoder capturing spatio-temporal dependencies.
- Lead-specific embeddings and SEP modules to incorporate spatial information.
- Decoder designed to reconstruct masked patches.
- Fine-tuning on downstream classification with standard metrics.
- Visualization and analysis for interpretability.
- Modular design supporting ablation studies (layer number, masking ratio, lead modules).

This thorough logical decomposition provides the foundation for detailed implementation aligned with the paper's methodology and experimental setup.

## trainer.py

Logic Analysis for trainer.py
---------------------------

**Objective:**
Implement the `ECGPretrainer` class to facilitate self-supervised masked ECG modeling training based on the ST-MEM methodology. The class handles dataset batching, patch masking and augmentation, model forward passes, loss calculation, optimizer updates, checkpoint management, and training epochs.

---

### 1. Class Initialization (`__init__`)
- **Inputs:**
  - `encoder` (TransformerEncoder): initialized with specified layers, heads, embed dims.
  - `decoder` (TransformerDecoder): initialized similarly.
  - `dataset` (ECGDataset): includes data paths, preprocessing, augmentation info.
  - `config` (dict): hyperparameters, schedules, Save/Load paths.

- **Setup:**
  - Store the model components.
  - Set up training hyperparameters:
    - Optimizer (AdamW) with specified learning rate.
    - Scheduler (cosine decay).
    - Total epochs, warm-up steps.
  - Initialize:
    - Data loaders for training.
    - Checkpoint paths and logging.
  - Set model to training mode.

---

### 2. Data Loading & Batching
- **Dataset handling:**
  - Use `datasets.py` loaded dataset, assume already split into train/validation/test.
- **Data loader:**
  - Use `torch.utils.data.DataLoader` with batch size (1024).
  - Shuffle enabled for training.
- **Batch composition:**
  - Each batch yields multiple ECG signals:
    - Raw signals: shape `(batch_size, L, T)`.

---

### 3. Patchify & Augmentation
- For each batch:
  - **Preprocessing:**
    - Normalize signals if needed (assumed done in dataset class).
    - For each ECG:
      - Segment into patches of size `patch_size=32`.
      - Result: `(batch_size, num_patches=128, L, patch_size)` (assuming patching applies per lead).
  - **Augmentations:**
    - Randomly apply augmentations based on defined probabilities:
      - Erase, flip, drop, sine_wave, partial_sine, white_noise.
    - These are implemented via functions from `utils.py`.
  - **Masking:**
    - For each batch:
      - Select a masking ratio from the `mask_ratio` hyperparameter, e.g., 0.15.
      - Randomly select `mask_ratio * num_patches` patches to mask.
      - Generate a mask tensor `(batch_size, num_patches)` with True/False.
    - Apply mask: set masked patches to a mask token or zero tensor.
    - Generate masked index set: `mathcal{M}`.

---

### 4. Model Input Preparation
- **Embed patches:**
  - Linear projection on patches: shape `(batch_size, num_patches, L, embed_dim)`.
- **Add embeddings:**
  - Lead embeddings: learned, shape `(L, embed_dim)`.
  - Positional embeddings: learned, shape `(num_patches, embed_dim)`.
  - Add lead embeddings per patch based on lead of each patch.
  - Add positional embeddings to patch embeddings.
- **Encoder input:**
  - For unmasked patches:
    - Select only unmasked patches.
    - Shape: `(batch_size, num_unmasked_patches, L, embed_dim)`.
  - Concatenate lead and positional info as needed.
- **Pass through encoder:**
  - Output: encoded representation tensor with shape `(batch_size, total_patches, embed_dim)`.

---

### 5. Masked Patch Reconstruction
- **Decoder input:**
  - Concatenate:
    - Encoded unmasked patches.
    - Mask tokens for masked patches (learned embedding).
  - **Note:**
    - Following the paper, decoder is lead-wise shared and processes each lead independently.
    - Adjust input shape for decoder accordingly.

- **Decoder pass:**
  - For each lead:
    - Input the corresponding encoded patches with mask tokens and lead info.
  - Use cross-attention with encoder outputs if decoder is designed to do so.

---

### 6. Loss Computation
- **Target:**
  - Original raw patch data from the dataset for the masked patches.
  - Shape: `(batch_size, num_masked_patches, L, patch_size)`.
- **Output:**
  - Reconstructed patches from decoder.
- **Calculate loss:**
  - Use mean squared error (or L1, per paper's assumption).
  - Average over all masked patches: per batch.
- **Backward pass:**
  - Compute gradients.
  - Optimizer step.

---

### 7. Optimization & Learning Rate Schedule
- **Optimizer:**
  - AdamW with initial LR 0.0012, weight decay as specified.
- **Scheduler:**
  - Cosine decay with warm-up steps.
  - Adjust learning rate per iteration or epoch.
- **Logging:**
  - Track loss, step numbers for monitoring.
- **Checkpointing:**
  - Save model weights periodically (e.g., after every epoch or best validation).
  - Save best state based on validation loss or other metrics.

---

### 8. Epoch Loop
- Loop over `epochs=800`:
  - For each batch:
    - Data loading.
    - Augmentation & masking.
    - Embedding.
    - Forward pass (encoder & decoder).
    - Compute loss.
    - Backpropagation.
    - Optimizer update.
    - Scheduler step.
  - Validation:
    - Optionally, evaluate reconstruction loss on val set.
    - Save checkpoint if improved.
  - Implement early stopping or epoch-based stopping.

---

### 9. Additional Considerations
- **Model devices:**
  - Use GPU(s) if available.
  - Move tensors/models to device (`cuda`/`cpu`).
- **Reproducibility:**
  - Set manual seed.
  - Log hyperparameters.
- **Evaluation:**
  - After training, provide methods in class for:
    - Saving/loading model checkpoints.
    - Validation loss.
    - Monitoring convergence.

---

### 10. Summary of Main Methods
- `__init__`: initialize all components.
- `train()`: main training loop.
- `_load_batch()`: fetch batch, apply augmentations and masking.
- `_forward()`: run forward pass through encoder and decoder.
- `_compute_loss()`: calculate reconstruction loss.
- `_update()`: perform optimizer step.
- `_save_checkpoint()`, `_load_checkpoint()`: checkpoint management.
- `validate()`: Evaluate on validation data.
- `run()`: wrapper to execute for full training.

---

### 11. Edge Cases & Debugging
- Mismatched tensor shapes (patchify, embed, mask).
- Empty patches after masking (bad masking ratios).
- Gradient explosion (clip gradients).
- Model convergence issues (adjust hyperparameters).
- Handling GPU memory constraints (batch size tuning).

---

This comprehensive logic analysis sets the foundation required to implement the `ECGPretrainer` class in `trainer.py`, ensuring fidelity to the described methodology of ST-MEM. The code should modularly assemble dataset management, model forward passes, masking/augmentation, loss computation, optimization, and checkpointing for effective self-supervised training of ECG representations.

## utils.py

{
  "utils.py": "The utils.py module will contain a collection of helper functions essential for data augmentation, visualization, and tensor manipulations to facilitate both pretraining and downstream evaluation, ensuring fidelity to the paper's methodology.\n\n**1. Data Augmentation Functions:**\n- erase_patch(patch, probability): Randomly zero out entire patches with a certain probability.\n- flip_patch(patch, probability): Randomly invert the sign of the patch, simulating flip augmentation.\n- drop_patch(patch, probability): Zero out patches, similar to erase but may be used differently contextually.\n- add_sine_noise(patch, freq_range): Add a sinusoidal wave with frequency selected randomly within the specified range; to simulate sine waveform augmentation.\n- add_partial_sine(patch, freq_range, ratio): Add sine noise to a portion of the patch, with ratio controlling the size of affected region.\n- add_white_noise(patch, std): Add Gaussian noise with specified standard deviation.\nThese augmentations should include probabilities to be applied randomly during training.\n\n**2. Visualization of Attention Maps:**\n- plot_attention_map(attention_weights, query_patch_index, save_path=None):\n  * Accepts attention weights (e.g., numpy array or tensor with shape [layers, heads, seq_len, seq_len]), a specific query patch index, and optional save path.\n  * Visualizes the attention scores for a specific head/layer, highlighting which temporal and lead patches are focused on.\n  * Uses matplotlib heatmaps, with appropriate labels representing lead and temporal positions.\n- generate_attention_maps_for_sample(sample, model, layer_indices, head_indices, save_dir):\n  * Pass a sample through the model, extract attention maps from specified layers/heads.\n  * Save or plot the heatmaps for interpretability.\n\n**3. Embedding Visualization:**\n- plot_embeddings(embeddings, labels=None, title='Embedding T-SNE', save_path=None):\n  * Input: high-dimensional embeddings (e.g., from the encoder output), optional labels for color coding.\n  * Use t-SNE (from sklearn.manifold import TSNE) to reduce dimensionality to 2D.\n  * Plot with matplotlib; color-code by lead type or rhythm if labels are provided.\n  * Save plot if save_path is provided.\n- cluster_and_evaluate_embeddings(embeddings, true_labels):\n  * Apply Gaussian Mixture Model (GMM) clustering (from sklearn.mixture import GaussianMixture).\n  * Measure clustering accuracy vs. ground truth attending to lead groupings (limb vs precordial). Return accuracy score.\n\n**4. Tensor Operations and Utilities:**\n- create_lead_embeddings(lead_count, embed_dim):\n  * Initialize lead-specific learnable embeddings, possibly as nn.Embedding, or torch.Tensor with requires_grad.\n- create_positional_embeddings(num_patches, embed_dim):\n  * Generate sinusoidal or learnable positional encodings for temporal patches.\n- visualize_patch_reconstruction(original_patches, reconstructed_patches, save_path=None):\n  * Plot original and reconstructed waveforms side by side for qualitative assessment.\n- convert_attention_weights_to_heatmap(attention_weights):\n  * Normalize and plot heatmaps to interpret what portions of the input are attended by specific queries.\n\n**5. Helper Functions for Model and Data Interaction:**\n- get_layerwise_attention_maps(model, sample_input, layer_idx, head_idx):\n  * Use model's hooks or directly access attention weights if stored during forward pass.\n- generate_attention_overlay_on_ecg(input_ecg, attention_maps):\n  * Overlay attention scores on temporal ECG waveform, possibly highlighting patches with high attention.\n\n**6. Consistency and Utility Functions:**\n- set_random_seed(seed):\n  * Ensure reproducibility.\n- normalize_tensor(tensor):\n  * Normalize input signals for consistent visualization.\n- save_figure(fig, filename):\n  * Save matplotlib figures for plots.\n\n**Implementation notes:**\n- All plotting functions will leverage matplotlib.\n- Augmentation functions should operate on numpy arrays or tensors, converting as necessary.\n- For attention visualization, extract attention weights from the model during forward passes, possibly via hooks or during a dedicated 'forward_with_attention' method.\n- Design all functions modularly with clear inputs and outputs, enabling reuse across training, evaluation, and interpretability steps.\n\n**Overall goal:**\nEnsure utility functions enable faithful reproduction of the paper’s qualitative analyses (attention visualization, embedding clustering) and data preprocessing (augmentation), aligning tightly with the described methodology without modifying the core design and interfaces."
}

