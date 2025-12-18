# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset.py

**Logic Analysis for `dataset.py`**

---

### 1. Purpose and Responsibilities
- Implement a `Dataset` class capable of:
  - Loading univariate or multivariate time series datasets.
  - Handling training and testing splits.
  - Preprocessing steps including normalization.
  - Managing data batching for training and evaluation.
  - Supporting synthetic dataset generation if specified.
- Interface with other modules (model training, evaluation, interpretability scripts).

### 2. Input Parameters and Initialization
- Constructor input:
  - `dataset_name`: string identifier for dataset to load, or special indicator for synthetic datasets (`'WebTraffic'`).
  - `split`: `'train'` or `'test'`.
  - `dataset_dir`: root directory where datasets are stored.
  - Optional configuration (e.g., normalization method).

- Internal steps:
  - Based on `dataset_name`:
    - Load dataset files (e.g., CSV/TSV files, or numpy files).
    - For UCR datasets:
      - Use standardized folder/file naming or a predefined list.
      - Load train/test splits according to original splits.
    - For synthetic `WebTraffic`:
      - Generate data programmatically following detailed procedures (see paper or Supplemental Info below).

### 3. Data Loading Logic
- **For real datasets (UCR):**
  - Read dataset files:
    - Extract time series data, shape `(samples, T)` (for univariate) or `(samples, T, c)` if multivariate.
  - Load labels:
    - Shape `(samples,)`.
    - Map string labels or integer labels as needed.
  - Store data internally:
    - `self.X`: numpy array of shape `(samples, T)` or `(samples, T, 1)`.
    - `self.y`: numpy array of shape `(samples,)`.

- **For synthetic `WebTraffic`:**
  - Generate `n_samples`:
    - Sample parameters from specified distributions.
    - Construct time series with seasonality, signatures, and noise.
    - Assign corresponding class labels (0-9).
  - Store generated data in numpy arrays with shape `(n_samples, T)`, labels `(n_samples,)`.

### 4. Data Normalization
- Method:
  - Check if normalization is specified (via config).
  - Applied to each time series:
    - **z-score normalization:** subtract mean, divide by std dev.
    - Or min-max scaling to [0,1].
  - For synthetic datasets, components like signatures may require specific normalization procedures or normalization applied after injection.

### 5. Data Preprocessing
- If datasets are variable length:
  - Pad or truncate to `T` (fixed length) as per dataset.
- For the `WebTraffic` synthetic data:
  - Already generated with fixed length (`t=1008`).
- Add positional encodings integration:
  - Not directly in dataset, but prepared as part of the data pipeline post loading.
- Support for batching:
  - Internal attributes:
    - `self.batched_data`: a DataLoader or custom batch generator.
  - Methods:
    - `__getitem__`: return a batch (or individual sample).
    - Possibly implement explicit batching method returning batches as `(batch_X, batch_y)`.

### 6. Support for Data Augmentation or Synthetic Generation
- For synthetic datasets:
  - Generate on-the-fly during class instantiation.
  - Follow the generation process detailed in Appendix C1:
    - Sample parameters for each signature.
    - Inject into base seasonality signals.
    - Assign class labels.
  - Store as numpy arrays.

### 7. Reproducibility
- Use a fixed seed (`seed=123` by default in config) for stochastic processes:
  - Random sampling of parameters.
  - Signature injection locations.
- Ensure reproducibility by setting numpy’s random seed before dataset creation (`np.random.seed(...)`).

### 8. Dataset Class API
- **Attributes:**
  - `X`: numpy array `(number_of_samples, T, c)` or `(number_of_samples, T)` for univariate.
  - `y`: numpy array `(number_of_samples,)`.
  - `dataset_name`
  - `split` (`'train'` or `'test'`)
- **Methods:**
  - `load()`: loads raw data.
  - `preprocess()`: normalization, optional rescaling.
  - `get_batch(batch_size)`: yields mini-batches of `(X_batch, y_batch)`.
  - `get_dataset()`: returns full data for evaluation.
  - For synthetic data: additional methods for regeneration and signature injection.

### 9. Error Handling
- Check if dataset files exist; raise descriptive errors if not.
- Validate dataset shapes and label formats.
- Validate dataset parameters (e.g., consistent length within a dataset).
- Handle edge cases:
  - Empty datasets.
  - Mixed or inconsistent data formats.
  - Mismatched labels/null labels.

### 10. Integration and Extensibility
- Modular design to allow adding new datasets with minimal code change.
- For synthetic data:
  - Parameterize the generation function.
  - Save generated datasets if needed for reproducibility.

---

### Summary of Core Logic Flow:
```plaintext
- Initialize Dataset with name, split, dir, config
- If dataset is real:
    - Load data files.
    - Map labels.
    - Validate shapes.
- Else if synthetic:
    - Generate data with specified signatures and parameters.
- Normalize data as per config.
- Store data internally (X, y).
- Provide methods for train/test splits, batching, data retrieval.
- Implement fixed seed setting for reproducibility.
- Handle potential data anomalies and validation.
```

---

This thorough analysis provides a detailed blueprint to implement `dataset.py`, ensuring support for both real-world and synthetic datasets, proper data handling, normalization, reproducibility, and ready integration with the rest of the experimental system.

## evaluation.py

evaluation.py - Logic Analysis

Overview:
This module is responsible for evaluating the trained time series classification models in terms of predictive performance metrics (accuracy, AUROC, loss) and interpretability measures (AOPCR, perturbation analysis, and ND CG). It depends on outputs generated by the interpretability.py module, the trained model, and the dataset. It must support flexible evaluation, including multiple repeats for interpretability scores, and should be compatible with the structure of the model, datasets, and interpretability classes.

Key Responsibilities:
- Load or receive the trained model.
- Load or receive dataset splits (test set).
- Generate predictions on the test set.
- Compute performance metrics:
  - Accuracy
  - AUROC
  - Cross-entropy Loss
- Compute interpretability scores:
  - AOPCR
  - Perturbation curves
  - ND CG (if ground-truth signature locations are available)
- Handle multiple repeats for interpretability evaluation, especially for metrics like AOPCR.
- Output summarized results for performance and interpretability.

Input Dependencies:
- Model instance (trained model).
- Dataset object (test data).
- interpretability.py functions/objects:
  - compute_scores(): to get interpretability scores per sample.
  - compute_perturbation(): to generate and evaluate perturbation curves.
  - compute_ndcgc(): optional, for datasets where true signature locations are known.
- Configuration parameters or arguments to specify:
  - Which metrics to compute.
  - Number of repeats for interpretability scores.
  - Save or display options for results.

Workflow & Detailed Steps:

1. Initialization:
   - Accept as input:
     - `model`: trained PyTorch model.
     - `dataset`: dataset object with test data.
     - `interpretability`: interpretability class instance, initialized with `model` and pooling modules.
     - `metrics_list`: list of metrics to evaluate (e.g., ['accuracy', 'AUROC', 'loss', 'AOPCR', 'ND_CG', 'perturbation']).
     - `num_repeats`: number of times to perform interpretability metric evaluation (e.g., 3).
   - Prepare storage dictionaries or lists for results.

2. Data Loading:
   - Access test split data:
     - Inputs: `X_test` (shape: [samples, channels, timesteps])
     - Labels: `Y_test` (shape: [samples])
     - For synthetic datasets with known signatures: `signature_indices` per sample, if available.

3. Model Evaluation (Performance Metrics):
   - Switch model to evaluation mode (`model.eval()`).
   
   - For each sample in `X_test`:
     - Generate model output logits or probabilities.
     - Predicted class: via softmax or argmax.
   
   - Calculate overall metrics:
     - Accuracy: proportion correct.
     - AUROC: use scikit-learn's roc_auc_score with true labels and predicted probabilities (multi-class).
     - Loss: cross-entropy loss averaged across samples.
   
   - Store these metrics.

4. Interpretability Metrics:
   
   - For each sample:
     - Generate interpretability scores:
       - Call `interpretability.compute_scores(series)` which returns importance scores per time point.
       
     - For multiple repeats (if applicable, e.g., for AOPCR and ND CG):
       - Repeat interpretability score computations (e.g., for 3 repetitions) to average out stochastic effects.
       - Accumulate interpretability scores per sample.
   
   - Compute the overall interpretability metrics:
     - **AOPCR**:
       - For datasets without ground-truth signatures: evaluate rank ordering against random baselines.
       - For datasets with known signatures: evaluate based on the true signature indices.
     - **Perturbation curves**:
       - For each sample, create a perturbation curve:
         - Rank importance scores.
         - Sequentially remove top-ranked time points (or blocks) and record the decay in model's predicted class probability.
       - Aggregate over samples for an average curve.
     - **ND CG**:
       - For synthetic datasets where label truth for signature positions is known:
         - Calculate the `r_e_l` value for positions of true discriminative points.
         - Normalize as per formula.
       - Aggregate over samples.

5. Results Summarization:
   - Compile:
     - Mean and std/max for performance metrics.
     - Mean and std for interpretability scores (AOPCR, ND CG).
     - Distribution plots for interpretability heatmaps or curves if needed.
     - Comparative plots (e.g., perturbation decay vs. random, vs. different methods).
   
6. Output & Logging:
   - Return dictionaries or structured objects containing all metrics.
   - Optionally, save figures, heatmaps, or results to files.
   - Log summary results for further analysis.

7. Optional:
   - Support evaluation for different models or pooling methods.
   - Support batch evaluation for large datasets.
   - Handle datasets with or without ground-truth signature annotations.

Supporting Details:
- Use `torch.no_grad()` context during prediction to reduce memory usage.
- Use scikit-learn metrics: `accuracy_score`, `roc_auc_score`.
- For AUROC, handle multi-class with `multi_class='ovr'` or `multi_class='ovo'`.
- Ensure importance scores are normalized for interpretability analyses.
- Confidence intervals or multiple repeats are used to estimate stability of interpretability scores.

Edge Cases:
- Datasets with very few samples: handle with caution (e.g., avoid over-interpretation).
- Datasets with very long time series: consider sampling blocks if needed.
- Missing ground-truth signature info: skip ND CG evaluation.
- Imbalanced classes: note in performance metrics.

Reproducibility:
- Use fixed seeds for randomness.
- Maintain consistent data splits for test/train.
- Save model states, interpretability outputs for validation.

Summary:
- Clear separation between performance evaluation and interpretability evaluation.
- Modular functions for each metric.
- Repeats to obtain statistical robustness on interpretability.
- Flexibility for synthetic or real datasets.
  
This detailed logic analysis guides the implementation of `evaluation.py`, ensuring the code will faithfully execute all necessary steps for reproducing the experiments from the paper, integrating the interpretability metrics correctly, and producing comprehensive evaluation results.

## interpretability.py

# interpretability.py - Logic Analysis

This module provides functionality for computing interpretability metrics such as AOPCR, perturbation analysis, and ND CG, based on model predictions and attribution scores. It leverages the trained MIL models with different pooling methods to assess how well the model's explanations align with known discriminative time points (for synthetic data) or evaluate the quality of explanations in a general setting.

---

## 1. Core Components and Responsibilities
- **Interpretability class**:
  - Initialize with:
    - Trained model (PyTorch nn.Module).
    - Pooling method used (to interpret pooled outputs).
    - Metrics configuration (e.g., evaluation repeats).
  - Compute scores for individual time series:
    - `compute_scores(series)`: produces per-time-point scores indicating importance/discriminative strength.
  - Compute perturbation-based metrics:
    - Sequential removal of top importance points.
    - Measure decay in model's output (say, class probability).
  - Compute ND CG metrics:
    - Use known labels for ground-truth discriminative points (if available).
    - Evaluate the ordering of importance scores.
- **Dependence on model outputs**:
  - Time point importance scores are derived either directly (e.g., from attention weights or class-specific predictions) or indirectly (via attribution methods like CAM).
  - The extracted scores should match the interpretability method:
    - For attention-based methods: attention weights directly serve as importance scores.
    - For class prediction-based methods: class-specific time point scores are obtained via per-time-point predictions or saliency maps.

---

## 2. Inputs and Outputs
### Inputs:
- Raw time series data: `series` (tensor `t` x `c=1` channels, or shape `(t,)`).
- Model-specific intermediate outputs:
  - Per-time-point class predictions.
  - Attention weights (if applicable).
  - Embeddings (for attribution methods like CAM).
- Known signature locations (for synthetic data) – optional, for ND CG evaluation.

### Outputs:
- **Scores per time point**:
  - Numeric importance scores (`float`) across sequence length `t`.
- **Perturbation evaluation result**:
  - Decay curve (list of probabilities or class scores).
  - Quantitative measure: Area over perturbation curve.
- **ND CG**:
  - Numeric score indicating ranking quality.

---

## 3. Workflow & Implementation Details

### Initialization
- Initialize with:
  - Model instance.
  - Pooling method name: e.g., `'GAP'`, `'Attention'`, `'Instance'`, `'Additive'`, `'Conjunctive'`.
  - Any model-specific parameters for attribution extraction.
  - Optional: reference signatures for synthetic datasets.

### compute_scores(series)
- **Step 1**: Forward pass:
  - Convert input series to tensor shape (`1, t, c=1`) (assuming batch dimension).
  - Pass through backbone + pooling:
    - For models with MIL pooling:
      - Obtain per-time-point class predictions if applicable.
      - Obtain attention weights if available.
      - Obtain combined importance scores (depending on method).
- **Step 2**: Extract importance scores:
  - For Attention: use the attention weights directly (shape `t`).
  - For class-specific predictions (Instance, Additive, Conjunctive): 
    - Use the per-time-point class probabilities for the predicted class.
  
- **Step 3**: Return a 1D numpy array or tensor of importance scores for the sequence.

### compute_perturbation(series)
- **Input**: importance scores from `compute_scores`.
- **Process**:
  - Rank points by importance descending.
  - Sequentially remove points in importance order:
    - Create perturbed sequences (e.g., replace points with mean or zero).
    - Forward each perturbed sequence through the model.
    - Record predicted class probability (or logits).
  - Generate decay curve: how the model's confidence drops as more important points are removed.
  - Compute area over the curve (AOPCR):
    - Normalize scores by comparing with random orderings (average over 3 runs).
    - Score is the difference in decay curves relative to random.

### compute_ndcgc(series, true_signature_indices)
- **Input**: annotated true signature points, ground-truth important indices.
- **Process**:
  - Obtain importance scores.
  - Rank points.
  - Compute the score based on the formula (Eqn. A.9):
    - Sum over the positions of the true relevant points in the ranked importance list.
    - Assign higher weights if true points are earlier.
  - Normalize by the total weight (sum of weights across top `n` points).
- **Output**: numeric score between 0 and 1 indicating ranking alignment.

---

## 4. Handling Different Interpretability Methods
- **Attention-based**:
  - Use the attention weights directly as importance scores.
  - Can be averaged over multiple heads if multiple heads are used.
- **Class-specific predictions (Instance, Additive, Conjunctive)**:
  - Use the per-time-point class probabilities or class-specific activation maps to derive importance.
  - For Conjunctive pooling:
    - Use the scaled per-time-point class prediction (`scaled scores`) to determine importance.
  - For Instance:
    - Use per-time-point class probabilities directly.
- **CAM-based explanations**:
  - Extract class activation maps from the model's last convolution layer.
  - Aggregate to importance scores for each time point.

### Additional considerations:
- **Sparsity & Thresholding**:
  - Optional: filter importance scores below a threshold for visualization.
- **Aggregation of Multi-component scores**:
  - For methods combining attention and predictions, combine importance measures appropriately.

---

## 5. Reproducibility & Configurability
- Allow control over:
  - Number of perturbation repeats.
  - Type of importance scores used.
  - Whether to use class-specific or class-agnostic importance.
  - Which significance metric to evaluate (AOPCR, ND CG).
- Use fixed seeds for perturbation sampling for consistency.
- Save interpretability scores and perturbation data for further analysis.

---

## 6. Summary of Key Functions and Parameters
| Function | Purpose | Inputs | Outputs | Notes |
|------------|-----------|-----------|-----------|---------|
| `__init__()` | Initialize interpretability class | model, method, config | None | Prepare for subsequent evaluations |
| `compute_scores(series)` | Generate importance scores | Time series tensor | 1D array of importance scores | Uses model forward + method-specific extraction |
| `compute_perturbation(series)` | Evaluate decay curve and AOPCR | Time series tensor | decay curve, AOPCR score | Sequentially remove points based on scores |
| `compute_ndcgc(series, true_sig_indices)` | Evaluate rank accuracy | Time series tensor, true signature indices | float score | Based on ranking positions |

---

## 7. Unclear Aspects / Clarifications Needed
- Exact method for extracting class activation maps (if CAM is used).
- How to handle multi-head attention: aggregate by average or select specific head.
- Input data shapes and normalization details during evaluation.
- Whether multiple class options are evaluated or only the predicted class.
- Specific scoring thresholds or filtering for score visualization.

---

# Final Notes
- This module depends critically on the outputs of the trained MIL models, especially the per-time-point class predictions or attention weights.
- To ensure consistency, develop a clear interface between:
  - Model forward pass.
  - Extraction of importance scores based on chosen interpretability method.
  - Perturbation process for decay measurement.
- All implementations should be vectorized where possible for efficiency.
- Maintain reproducibility by fixing random seeds during importance score sampling or perturbation steps.

This thorough breakdown will facilitate accurate, faithful implementation aligned with the paper’s methodology and experiment setup.

## main.py

# Logic Analysis for main.py: Experiment Orchestration Script

This document details the logical flow, functional components, data handling, and control structure required for implementing `main.py`, the central orchestrator for reproducing the MILLET experiments as described in the paper.

---

## 1. Initialization and Setup

### 1.1 Import Modules
- Import standard libraries: `os`, `sys`, `yaml`, `logging`, `random`.
- Import scientific libraries: `numpy as np`, `torch`.
- Import custom modules:
  - `datasets.py`: Dataset class (load, preprocess).
  - `model.py`: Class to instantiate backbone networks.
  - `modules.py`: MIL pooling classes.
  - `interpretability.py`: For interpretability evaluation.
  - `trainer.py`: For training loop.
  - `evaluation.py`: For performance and interpretability metrics.
  - `utils.py`: Positional encoding generator, seed setting.

### 1.2 Parse Configuration
- Load `config.yaml` via `yaml.safe_load()`.
- Extract parameters into variables:
  - Training parameters: `learning_rate`, `batch_size`, `epochs`, `seed`, `early_stopping`.
  - Model parameters: `backbone`, `embedding_dim`, `dropout_rate`, `architecture_params`, `pooling_method`, `pooling_params`.
  - Interpretability: method, signature injection, evaluation repeats.
  - Dataset: name, split ratio, normalization, synthetic flag, dataset directory.

### 1.3 Set Reproducibility
- Call `set_seed()` with `seed` value to fix randomness across `numpy`, `torch`, `random`.
  
## 2. Dataset Loading and Preparation

### 2.1 Dataset Instantiation
- Use the `Dataset` class:
  - Pass dataset name, dataset directory.
  - For synthetic data (`synthetic=true`), generate data as per signature injection:
    - Set parameters such as classes, signatures, window lengths.
    - Generate the synthetic time series and labels accordingly.
  - For real datasets:
    - Load data splits from the archive.
    - Apply normalization (`z-score` or other) as per config.
- Split into training and test sets based on ratio provided (or load original splits if provided).

### 2.2 Data Batching
- Instantiate data loaders for train/test:
  - Use `torch.utils.data.DataLoader` with `batch_size`, shuffling, worker params.
  - Dataset class should output `(series, label)` pairs, where series shape: `(t,)` or `(1, t)`.

### 2.3 Data Validation
- Validate batch shapes, ensure data is float tensors.
- Confirm labels are integers for class index.

## 3. Model Architecture Initialization

### 3.1 Instantiate Backbone Network
- Use `model.py` to create backbone:
  - Select backbone architecture (`FCN`, `ResNet`, `InceptionTime`) based on config.
  - Pass architecture-specific parameters.
  - Output:
    - An initialized `torch.nn.Module`.
    - Confirm output feature embeddings shape: `(batch_size, t, embedding_dim)` (or compatible shape).
- Optional: verify model is moved to GPU if available.

### 3.2 Instantiate Positional Encoding
- If positional encoding enabled:
  - Create positional encoding tensor with size `(max_sequence_length, embedding_dim)`.
  - Fixed sinusoidal form as per Vaswani et al. (2017).
  - These encodings are added to feature embeddings after backbone processing.

### 3.3 Instantiate Pooling Module
- Instantiate pooling class (`GAP`, `Attention`, `Instance`, `Additive`, `Conjunctive`) from `modules.py`.
  - Pass pooling-specific parameters (attention heads, size).
  - Confirm the pooling produces both:
    - Series-level class logits.
    - Per-time-point interpretability scores or attention weights, as needed.

---

## 4. Model Composition & Forward Pass Logic

### 4.1 Forward Data
- For each batch:
  - Extract series `(batch_size, t, c)`.
  - Pass through backbone to obtain `(batch_size, t, embedding_dim)`.
  - If positional encoding active, generate encoding for sequence length and add to embeddings.
  - Pass embeddings (and attention weights if relevant) to the pooling module.
  - Pooling produces:
    - Series-level class logits `(batch_size, c)`.
    - Per-time-point scores for interpretability if applicable.

### 4.2 Loss Calculation
- Use cross-entropy loss on predicted class logits.
- Optionally, incorporate class weights if dataset imbalance requires.
- Compute:
  - `loss = criterion(outputs, labels)`.

### 4.3 Backpropagation
- Zero optimizer gradients.
- Backward pass: `loss.backward()`.
- Step optimizer: `optimizer.step()`.

---

## 5. Training Loop

### 5.1 Epochs
- Loop from `1` to `epochs`:
  - For each batch:
    - Perform the forward pass.
    - Compute loss.
    - Backpropagate.
  - Optionally compute validation performance.
  - Optional early stopping based on validation loss/accuracy.
  - Log training progress.
- After training:
  - Save the best model (by validation accuracy or training loss).

### 5.2 Replicated Models (Ensemble)
- If ensemble:
  - Repeat training `n` times with different seeds.
  - Save each model checkpoint.
  - For evaluation, ensemble their output logits by averaging.

---

## 6. Evaluation of Model Performance

### 6.1 Prediction
- Run the trained model on the test set:
  - Collect logits, compute predicted classes.
  - Calculate metrics:
    - Accuracy
    - AUROC (if applicable, softmax scores).
    - Loss.
- Save predictions and true labels for further analysis.

### 6.2 Interpretability Calculation
- For interpretability evaluation:
  - For datasets with signature ground truth:
    - Select a subset of samples.
    - Generate interpretability scores for each test series:
      - Use model and pooling outputs.
      - Compute interpretability metrics as per AOPCR, ND CG.
  - For datasets without signature labels:
    - Use AOPCR based on perturbation evaluation.
  - Repeat interpretability evaluations as configured (`evaluation_repeat` times).

### 6.3 Plot & Save Interpretations
- Generate heatmaps or importance scores for sample test series.
- Save plots/images for inspection.

## 7. Results Aggregation and Reporting

### 7.1 Performance Metrics
- Compute and compile metrics across all datasets:
  - Mean, variance, worst/best scores.
- Save metrics in structured formats (CSV, JSON).

### 7.2 Interpretability Metrics
- Summarize interpretability scores:
  - Average AOPCR, NDCG, perturbation curve decay.
  - Cross-compare pooling methods, backbone models.

### 7.3 Comparative Analysis
- Generate critical difference diagrams.
- Plot performance vs dataset properties (length, imbalance).

---

## 8. Finalization & Cleanup

- Save final models.
- Save experiment logs, metrics, and plots.
- Clear GPU cache if needed.
- Exit gracefully with status code 0.

---

## 9. Additional Remarks
- **Error Handling:** Add try-except blocks around major steps for fault tolerance.
- **Logging:** Use Python's `logging` module for status updates.
- **Experiment Reproducibility:** Save random seeds, model configs, dataset info.
- **Computational Efficiency:**
  - Use GPU if available.
  - Use `torch.no_grad()` for evaluation.
  - Batch evaluation to save time.
  
---

This comprehensive logic analysis provides a step-by-step blueprint to implement `main.py`, ensuring reproducibility, clarity, and adherence to the methodologies described in the paper.

## model.py

# Logic Analysis for model.py

This script defines the core backbone neural network classes supporting FCN, ResNet, and InceptionTime architectures, which are used as feature extractors in the MILLET framework. It should provide a unified interface for initializing, forward passing, and extracting feature embeddings from each architecture, based on provided architecture parameters.

---

## Core Responsibilities
- Implement three backbone models:
  - FCN (Fully Convolutional Network)
  - ResNet (Residual Network)
  - InceptionTime
- Support configurable hyperparameters, e.g., number of residual blocks, kernel sizes.
- Output feature embeddings before final classification layer, to be used by MIL pooling modules.
- Maintain compatibility with the rest of the pipeline, especially output shape requirements and differentiability.
- Include functions for:
  - Model construction (initialization)
  - Forward pass (returning embeddings or class logits)
  - Extraction of feature embeddings
- Support optional parameter tuning via `architecture_params` from the configuration.
- Maintain consistency with the "embedding_dim" parameter (set to 128 as default from config).

---

## Input Specifications
- Input tensor shape: `(batch_size, 1, t)` (for univariate series).
- `batch_size`: dynamic, depends on current data loader.
- `t`: sequence length, fixed per dataset.
- Data normalization handled externally (by the dataset loader or preprocessing).

## Output Specifications
- For MIL integration, the backbone should output:
  - Feature embeddings per time point: shape `(batch_size, 1, t, embedding_dim)` or `(batch_size, t, embedding_dim)` -- choose the format consistent with downstream layer requirements.
  - Or, alternatively, the final pooled features before classification; but more likely, the embedding from the last convolutional layer for MIL pooling modules.
  - Class logits for final prediction: shape `(batch_size, c)` (where `c` is number of classes).

## Structure & Implementation Details
- **Class `BackboneNetwork` (or similar)**:
  - Constructor parameters:
    - `architecture`: string ('FCN', 'ResNet', 'InceptionTime')
    - `embedding_dim`: int (default 128)
    - `architecture_params`: dict (custom hyperparameters)
  - Internal logic:
    - Select the network construction method based on `architecture`.
    - Build network layers accordingly.
  - `forward(x)`:
    - Process input `x` through layers.
    - Return:
      - `embeddings`: tensor of shape `(batch_size, t, embedding_dim)` (preferable for MIL modules).
      - Or class logits: as final output for training and evaluation.
- **Additional functionalities**:
  - Methods to extract feature embeddings explicitly if needed for interpretability.
  - Optionally, methods to load/save weights.

---

## Architecture-specific details:

### 1. FCN
- Simple stack of convolutional layers:
  - Usually 4 residual or convolutional blocks with residual connections.
  - Kernel sizes consistent with the original paper (e.g., [8, 5, 3] as in the config).
  - Activation: ReLU.
  - Batch normalization after each conv.
- Final feature extractor: last convolutional layer outputs `embedding_dim` channels.
- Output shape: `(batch_size, t, embedding_dim)`.

### 2. ResNet
- ResNet for time series (adapted from sequence labeling architectures):
  - Use residual blocks with convolutional layers.
  - Number of residual blocks: configurable (`residual_blocks`: 4).
  - Each block:
    - Conv1D layers with the kernels specified.
    - Batch norm, ReLU, residual skip connections.
- Final output: feature map with length reduced or same as input depending on stride/padding.
- Embedding extraction: last residual block output.
- Output shape similar to FCN for consistency.

### 3. InceptionTime
- Based on Inception modules:
  - Multiple parallel convolutional branches with multiple kernel sizes.
  - Concatenate their outputs.
  - Residual connections.
- Number of residual blocks: 4, as per config.
- Use of multiple kernel sizes as specified (e.g., [8, 5, 3]).
- Final output: feature map shape `(batch_size, t, embedding_dim)`.

---

## Hyperparameters and Config Integration
- Use `architecture_params` (from config):
  - `'residual_blocks'`: number of residual blocks (default 4).
  - `'kernel_sizes'`: list of kernel sizes.
  - Additional hyperparameters (e.g., number of filters) may be fixed or configurable.
- Always ensure that the output embedding size matches `embedding_dim` (128).
- Add dropout layers if specified (e.g., after convolutions).
- For consistency, adopt the activation functions, batch normalization, and residual connection strategies aligned with preceding literature.

---

## Reproducibility & Testing
- Initialize all layers with reproducible random seeds.
- Modular design: each network as a subclass, with the same interface.
- Connect with the rest of the pipeline for training/testing.
- Include a factory function `build_backbone(architecture, params)` to instantiate models dynamically.

---

## Summary & Key Points
- **Design conformant to the high-level API:**
  - Should accept input tensor of shape `(batch_size, 1, t)`.
  - Return feature embeddings `(batch_size, t, embedding_dim)` for MIL pooling.
  - Provide logits for training purposes.
- **Implementation must be modular, readable, and flexible for hyperparameter tuning or architectural adjustments.**
- **No external dependencies apart from PyTorch are needed.**

---

This logic analysis provides a thorough, step-by-step plan for implementing `model.py`, respecting the architectural design, input/output specifications, hyperparameter flexibility, and integration points with other modules (like `modules.py` for pooling). Actual code should follow these guidelines for consistency, correctness, and reproducibility.

## modules.py

{
  "modules.py": [
    {
      "component": "PositionalEncoding",
      "description": "Implement a class for fixed sinusoidal positional encodings as described by Vaswani et al. (2017).",
      "details": [
        "Input parameters: maximum sequence length (max_length), embedding dimension (d_model).",
        "Method generate(): produces a tensor of shape (max_length, d_model) with positional encodings using sine and cosine functions.",
        "Implementation: For position p in [1, max_length], compute PE using formulas:",
        "  PE(p, 2i) = sin(p / 10000^(2i / d_model)),",
        "  PE(p, 2i+1) = cos(p / 10000^(2i / d_model)).",
        "Add the positional encodings to the embeddings after feature extraction."
      ],
      "considerations": [
        "Handle variable sequence lengths at runtime: positions for in-batch sequences should be sliced accordingly, or generate fixed PE tensor and slice as needed.",
        "Ensure tensor shape compatibility: shape (t, d_model).",
        "Add encoding in the forward method or as a utility function."
      ]
    },
    {
      "component": "GAPPooling",
      "description": "Implement a class for global average pooling (mean of embeddings over time).",
      "details": [
        "Input: embeddings tensor of shape (batch_size, 1, t, d), where t is sequence length.",
        "Operation: take mean over sequence dimension (dim=2).",
        "Output: pooled tensor of shape (batch_size, 1, 1, d).",
        "Followed by a classifier (fully connected layer) for label prediction."
      ],
      "considerations": [
        "Ensure support for different batch sizes.",
        "The output shape matches the expected input to classifier modules."
      ]
    },
    {
      "component": "AttentionPooling",
      "description": "Implement the attention MIL pooling method: computes attention scores for each time point, scales embeddings accordingly, then pools by weighted sum.",
      "details": [
        "Input: embeddings tensor (batch, 1, t, d).",
        "Attention head: a two-layer neural network with tanh + sigmoid activations:",
        "  - Attention linear layer: input d, output a small hidden size (e.g., 8).",
        "  - Apply tanh activation.",
        "  - Attention linear layer: from hidden size to 1 (attention score per time point).",
        "  - Apply sigmoid to produce attention weights a_i_j in [0,1].",
        "Scale embeddings by attention scores: for each time point j, embed -> a_i_j * embed.",
        "Perform pooling: weighted sum over time points to produce a single embedding.",
        "Output: the pooled embedding fed into classifier for class prediction."
      ],
      "considerations": [
        "Ensure attention scores are properly normalized if softmax is needed; here, sigmoid is used per description.",
        "Support multi-head or single-head attention based on attention_heads parameter.",
        "Maintain differentiability for end-to-end training."
      ]
    },
    {
      "component": "InstancePooling",
      "description": "Each time point gets a class prediction, then average over all time points to get the series prediction.",
      "details": [
        "Input: embeddings tensor (batch, 1, t, d).",
        "Operation: apply a classifier (linear layer) independently to each time point to generate predictions (batch, 1, t, c).",
        "Pool: average over time dimension (dim=2) to produce (batch, 1, 1, c).",
        "This directly yields class-specific time point scores, interpretable as motifs supporting/refuting classes."
      ],
      "considerations": [
        "Ensure the classifier is applied in a convolutional manner or with proper reshaping.",
        "Support batch operations efficiently."
      ]
    },
    {
      "component": "AdditivePooling",
      "description": "Combine attention and instance predictions: compute attention weights, produce time point class predictions, then scale by attention, and pool.",
      "details": [
        "Input: embeddings (batch, 1, t, d).",
        "Attention head: same as AttentionPooling, producing attention scores a_i_j.",
        "Classifier head: independently produce class scores y_i_j for each time point.",
        "Combine: element-wise multiply each class prediction y_i_j by attention score a_i_j, per time point.",
        "Pool: average over time points for each class to obtain final series predictions."
      ],
      "considerations": [
        "Ensure that the scaling and pooling preserve class-specific interpretability.",
        "Manage tensor shapes for batch processing."
      ]
    },
    {
      "component": "ConjunctivePooling",
      "description": "Separate attention head and classifier for each time point, then combine via element-wise product of attention score and class prediction per time point, followed by average over time.",
      "details": [
        "Input: embeddings (batch, 1, t, d).",
        "Attention head: as before, produce attention scores a_i_j (batch, 1, t, 1).",
        "Classifier head: produce class predictions y_i_j per time point (batch, 1, t, c).",
        "Combine: create scaled predictions: y_i_j_scaled = a_i_j * y_i_j.",
        "Pool: average y_i_j_scaled over time to produce final class prediction per series."
      ],
      "considerations": [
        "Class-specific interpretability: attention highlights important time points supporting particular classes.",
        "Ensure model stability: scale predictions carefully, possibly normalize attention scores."
      ]
    },
    {
      "common considerations": "For all pooling modules:",
      "details": [
        "Make sure they accept input tensor of shape (batch, 1, t, d).",
        "Return class prediction logits (batch, 1, 1, c).",
        "Incorporate dropout layers after positional encoding or within attention/classification heads if specified in the configuration.",
        "Support for different attention head sizes and attention heads count via parameters.",
        "Ensure all modules are differentiable for training."
      ]
    }
  ],
  "considerations": [
    "Tensor device management: ensure all tensors are on the same device (CPU/GPU).",
    "Initialization of parameters: apply Xavier/Glorot uniform for linear layers.",
    "In the context of the full system, ensure the pooling modules can be swapped seamlessly for experimentation.",
    "Design each class with a forward() method accepting input tensors and returning outputs.",
    "Include optional support for positional encoding addition inside each pooling module if needed."
  ],
  "unclear_points": [
    "Handling of variable sequence length at runtime: whether to support dynamic length or fixed length only.",
    "Whether attention heads are multi-head or single-head in configuration.",
    "If softmax or sigmoid is preferred for attention at inference time. The paper indicates sigmoid is used.",
    "Additional normalization steps for attention weights or class predictions in Conjunctive pooling."
  ]
}

## trainer.py

# Logic Analysis for trainer.py

This module manages the training, validation, and optionally early stopping of the proposed MILLET models, integrating dataset loading, model instantiation, loss calculation, optimizer steps, and saving model checkpoints. It should be designed to be modular, flexible, and reproducible, respecting the configuration provided in config.yaml.

---

## 1. Initialization
- **Input parameters:**
  - `dataset`: dataset object (`Dataset`) providing training and validation data.
  - `model_params`: a dictionary containing hyperparameters for model (e.g., backbone, pooling method, embedding_dim, dropout_rate, architecture_params).
  - `training_params`: dictionary with training hyperparameters (learning_rate, batch_size, epochs, early_stopping, seed).
  - `interpretability`: optional, details if interpretability metrics/evaluation are to be run.
  
- **Actions:**
  - Set the random seed for reproducibility (`torch.manual_seed`).
  - Initialize the model:
    - Instantiate the backbone network (FCN, ResNet, or InceptionTime) using parameters.
    - Instantiate the pooling module (`MILPooling`) according to `pooling_method` and associated params.
    - Construct the overall model that includes backbone + pooling + classifier, ensuring the forward method outputs class logits, and per-time-point scores when needed.
  - Move the model to GPU if available (`cuda()`), else CPU.
  - Initialize the optimizer (`torch.optim.Adam`) with `learning_rate`.
  - Create a loss function: `torch.nn.CrossEntropyLoss()`.
  - Initialize variables for early stopping criteria if enabled (e.g., patience, best validation loss).

---

## 2. Data Loading & Batching
- **Inputs:**
  - Training data: provided by dataset object, yields `(series, label)` pairs.
  - Validation data: similar format, used for validation.
  
- **Actions:**
  - Wrap datasets in DataLoader objects with `batch_size=training_params['batch_size']`, shuffle enabled for training.
  - For reproducibility, set worker seed if needed.
  
## 3. Training Loop
- **For each epoch (up to `training_params['epochs']`):**
  - Set model in train mode (`model.train()`).
  - Initialize accumulators: total loss, correct predictions, total samples.
  - For each batch:
    - Retrieve batch data: series (`(batch, 1, t)`), labels (`(batch,)`).
    - Zero optimizer gradients (`optimizer.zero_grad()`).
    - Forward pass:
      - Pass series through model.
      - The model returns:
        - Logits: shape `(batch, num_classes)`.
        - Optional: per-time-point scores if interpretability calls for it.
    - Compute loss:
      - Use `CrossEntropyLoss` comparing logits and labels.
    - Backpropagation:
      - Call `loss.backward()`.
      - Step optimizer (`optimizer.step()`).
    - Update metrics:
      - Accumulate loss.
      - Count correct predictions (argmax of logits vs labels).
  - Compute average training loss and accuracy.
  
- **Optional validation:**
  - Set model in eval mode (`model.eval()`).
  - Loop through validation DataLoader:
    - No gradient computation (`torch.no_grad()`).
    - Forward pass with validation data.
    - Compute validation loss.
    - Count predictions for accuracy.
  - Compute validation metrics.
  
## 4. Early Stopping & Checkpointing
- **If early stopping enabled (`early_stopping=True`):**
  - Track the validation loss/accuracy per epoch.
  - Save the model state dict if validation metric improves.
  - Stop training if validation performance does not improve for a defined patience number.
  
- **Save best model:**
  - After all epochs or early stopping trigger, load the best model weights.

## 5. Post-training
- Save the final trained model.
- Return the trained model, training logs (loss/accuracy histories), and best model checkpoint if applicable.

## 6. Additional Considerations
- **Reproducibility:**
  - Use fixed random seed as per `training_params['seed']`.
  - Ensure DataLoader and model initializations are reproducible.
- **Logging and Tracking:**
  - Print or log epochs, training loss, training accuracy, validation loss, validation accuracy.
  - Optionally, integrate with tensorboard or other logging tools for detailed tracking.
- **Handling multiple seeds for variance estimation:**
  - Encapsulate training in a loop over multiple seeds if specified outside this script, or via a wrapper.

## 7. Implementation Checklist
- [ ] Set seed
- [ ] Initialize dataset loaders
- [ ] Instantiate model with specified backbone + pooling + classification head
- [ ] Set to GPU/CPU
- [ ] Define optimizer and loss
- [ ] Set early stopping variables if enabled
- [ ] Loop over epochs:
  - train: batch-wise forward + backward
  - validate: batch-wise forward
  - update/track metrics
  - save checkpoint if improved
- [ ] Finalize training
- [ ] Return model, metrics data

---

## Summary
This logical structure ensures a clean, reproducible training process aligned with the paper's methodology. The module will be designed to allow easy extension for different backbone models, pooling methods, interpretability evaluations, and experiment configurations, using the provided config.yaml.

This detailed analysis should guide the implementation of `trainer.py` with clarity, completeness, and reproducibility in mind.

## utils.py

# Logic Analysis for utils.py

**Objective:**  
Implement shared utility functions that support core modules in dataset processing, positional encoding generation, normalization, and dataset statistics calculations, facilitating the main training and evaluation pipelines.

---

## 1. Positional Encoding Generation

### Functionality:
- Generate sinusoidal positional encodings as per Vaswani et al. (2017).
- Encode positional information for sequences up to a maximum length.
- Add positional encodings to feature embeddings obtained from backbone models.
- Handle cases where sequences are truncated or have missing positions.

### Inputs:
- `max_length`: An integer representing the maximum expected sequence length (from configuration; e.g., 1008 for WebTraffic, or dataset-specific maximum length).
- `embedding_dim`: An integer (e.g., 128), the size of the embedding vectors (matches feature extractor output dimension).
- Optional: `device` for tensor placement (GPU/CPU).

### Outputs:
- Torch tensor of shape `(max_length, embedding_dim)` containing positional encoding vectors for each position.

### Implementation Details:
- Use the sinusoidal functions:
  - `PE(pos, 2i) = sin(pos / 10000^(2i/embedding_dim))`
  - `PE(pos, 2i+1) = cos(pos / 10000^(2i/embedding_dim))`
- Generate all positions at once to optimize computation.
- Store or cache this matrix if performance constraints or reuse exists.
- During data processing, add the encoding to the feature embeddings of sequences. If the sequence length is less than `max_length`, only use the relevant slice.

---

## 2. Dataset Preprocessing & Normalization

### Functions:
- **normalize_series(series, method)**:
  - Normalize a given time series array.
  - Methods:
    - `'z-score'`: subtract mean, divide by std.
    - `'min-max'`: scale data to [0, 1].
  - Inputs:
    - `series`: numpy array, shape `(timesteps,)`.
    - `method`: string indicating normalization method.
  - Outputs:
    - normalized series (numpy array).

- **compute_dataset_statistics(datasets)**:
  - Compute global dataset statistics like mean, std, min, max, for normalization.
  - Input:
    - List of datasets (each as array or pandas DataFrame).
  - Outputs:
    - A dictionary with aggregate measures for normalization or analysis.

### Usage:
- Standardize all datasets prior to training.
- Allow data augmentation or specific dataset training strategies.

---

## 3. Dataset Handling & Metadata

### Functions:
- **load_dataset(name, dataset_dir)**:
  - Load specific dataset (e.g., from UCR archive) as numpy arrays or pandas.
  - Inputs:
    - `name`: dataset name string.
    - `dataset_dir`: path to dataset location.
  - Outputs:
    - `(X_train, y_train), (X_test, y_test)`:
      - `X`: numpy array, shape `(samples, timesteps)` for univariate.
      - `y`: labels, shape `(samples,)`.
  - Handle normalizations if specified.
  - For synthetic datasets:
    - Generate data dynamically, inject signatures, assign labels.

- **pad_sequence(series, target_length, mode='replicate')**:
  - Pad time series to target length if needed.
  - Use `'replicate'` mode: fill padding with boundary value.
  - Use `'zero'` if specified, but default is `'replicate'`.

---

## 4. Utility Functions for Dataset Statistics & Metrics

### Functions:
- **calculate_class_distribution(y)**:
  - Count samples per class.
  - Input: labels array `y`.
  - Output: dictionary or numpy array with counts.

- **compute_imbalance_metric(y)**:
  - Calculate normalized Shannon entropy (as per App. D.2).
  - Formula:
    - `- (1 / log(c)) * sum_{i=1}^c (p_i * log p_i)`
    - where `p_i = class_count_i / total_samples`.
  - Use `scipy.stats.entropy` or implementation directly.

- **evaluate_accuracy(model, dataset)**:
  - Run inference over dataset and compute accuracy.
  - Inputs:
    - `model`: trained PyTorch model.
    - `dataset`: data loader or dataset object.
  - Outputs:
    - accuracy float.

- **evaluate_AUROC(model, dataset)**:
  - Calculate AUROC if dataset has binary or multi-class labels.
  - Use `sklearn.metrics.roc_auc_score`.
  - For multi-class, do one-vs-rest per class.

---

## 5. Implement Safe & Efficient Computation

- Batched tensor processing:
  - Support batch operations for positional encoding addition.
  - Support batch inference during evaluation.
- Caching:
  - Store generated positional encodings when multiple calls are made.
- Compatibility:
  - Maintain tensor device (CPU or GPU).

---

## 6. Additional Considerations

### Error Handling:
- Check for matching dimensions.
- Handle missing or malformed data gracefully.

### Extensibility:
- Allow functions to accommodate variable sequence lengths if datasets vary.
- Modular design: functions should be composable and interchangeable.

### Reproducibility:
- Include options for setting RNG seeds in normalization and dataset sampling functions (if used).
- Logging or returning summary statistics.

---

## Summary:
The `utils.py` file will contain:
- `generate_positional_encoding(max_length, embedding_dim, device='cpu')`  
- `normalize_series(series, method='z-score')`  
- `compute_dataset_statistics(datasets)`  
- `load_dataset(name, dataset_dir)`  
- `pad_sequence(series, target_length, mode='replicate')`  
- `calculate_class_distribution(y)`  
- `compute_imbalance_metric(y)`  
- `evaluate_accuracy(model, dataloader)`  
- `evaluate_AUROC(model, dataloader)`  

All functions should be optimized for efficiency, clearly documented, and designed for reuse across training, inference, and interpretability evaluations.

---

**Clarifications Needed:**
- Exact dataset format and paths.
- Whether additional normalization or transformations are required beyond standard z-score or min-max.
- Whether to include dataset splitting and shuffling logic.
- Confirm if caching positional encodings is necessary or optional.

This comprehensive logic analysis will guide implementation to ensure correctness, efficiency, and conformity with the paper’s methodology.

