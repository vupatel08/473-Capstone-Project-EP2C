# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

**Logic Analysis for `dataset_loader.py`: Implementation of DatasetLoader class**

---

### **Overview of `DatasetLoader` class purpose**

The `DatasetLoader` class's primary role is to encapsulate all data handling tasks necessary for training and evaluation. This includes:

- Loading datasets (e.g., GLUE tasks for NLP, ImageNet or CIFAR for vision)
- Performing tokenization (for NLP) or image preprocessing (for vision)
- Shuffling datasets and partitioning into batches
- Providing data iterators compatible with the training pipeline
- Maintaining consistency of data splits, batching, and preprocessing

Given the experimental context from the paper, the class should be flexible enough to support different dataset types (text and vision), modular enough to plug into models, and capable of providing batches suitable for importance sampling routines.

---

### **Core functional requirements**

1. **Dataset selection and loading**

   - Based on provided configuration (`dataset.name`, `dataset.task`)
   - Support common datasets such as:
     - NLP: GLUE tasks (e.g., SST-2)
     - Vision: ImageNet, CIFAR-10/100
   - Use the `datasets` library for NLP datasets and possibly torchvision for vision (though for uniformity, datasets library can be used with appropriate datasets; torchvision transforms can be applied post-loading).

2. **Tokenization and pre-processing**

   - For NLP:
     - Use the `transformers` library's tokenizer (specified by `tokenizer_name`)
     - Truncate/pad sequences to `max_tokens` (e.g., 128)
     - Handle dataset splits: train, validation, test
     - Convert data into tokenized input tensors, including attention masks, token IDs, labels
   - For vision:
     - Load images and apply normalization, resizing, cropping as needed
     - Ensure input tensors are consistent with model expectations
     - Support data splits similarly

3. **Batch preparation**

   - Implement **batch shuffling**—for training, data should be randomly shuffled per epoch (except possibly for evaluation).
   - Support **batch iteration**:
     - Return a generator or iterable over batches
     - Batches should be formatted as tensors compatible with models (e.g., input IDs, attention masks for NLP; images tensors for vision)
   - Batch size should be configurable (`batch_size` from `config.yaml`)

4. **Handling multiple datasets / tasks & splits**

   - For training, use datasets['train']
   - For evaluation, use datasets['validation'] or datasets['test']
   - Support mixing if necessary, but primarily focus on the specified split

5. **Data shuffling and reproducibility**

   - Shuffling should be controlled via a seed for reproducibility
   - During each epoch, data should be reshuffled

6. **Reproducibility & consistency**

   - Use explicit random seeds for shuffling
   - Consistent tokenization and processing pipeline

---

### **Implementation details & considerations**

- **Initialization parameters** (passed or set via config):

  - `dataset_name`, `task_name`, `split`
  - `tokenizer_name`
  - `max_tokens`
  - `batch_size`
  - `shuffle` (boolean for training)
  - `seed` (for reproducibility)

- **Normalization & Transform pipelines**:

  - For vision:
    - Resize, CenterCrop, Normalize transforms from torchvision
  - For NLP:
    - Tokenization at dataset loading
    - Pad/truncate sequences
  - Use `transformers`' Tokenizer's `__call__()` method to process text.

- **Dataset object**:

  - Loader from `datasets` library, e.g., `load_dataset()`
  - For vision, possibly use `datasets`' image datasets or custom loader using `torchvision.datasets`

- **DataLoader / batching**:

  - Use `torch.utils.data.DataLoader` with custom Dataset class wrapping the above logic
  - For importance sampling, the dataset should be flexible enough to return importance scores alongside data (not necessarily in this class, but if needed, the loader should be compatible)

- **Iteration API**:

  - Provide `__iter__` or `get_batch_iterator()` method
  - Support `batch_size`, `drop_last`, `shuffle`,
  - Should return batches in format expected for training loop: dictionary or tuple containing inputs and labels

---

### **Handling different dataset types**

- **NLP datasets (e.g., SST-2)**:
  - Use `datasets.load_dataset()`
  - Apply tokenizer
  - Return token ids, attention masks, labels
- **Vision datasets**:
  - Use `torchvision.datasets` or `datasets.load_dataset()` with image datasets
  - Apply torchvision transforms (resize, normalization)
  - Return image tensors, labels

### **Errors and edge cases**

- Dataset not found or incompatible task
- Data with missing labels or corrupt images
- Tokenizer mismatch
- Dataset size smaller than batch size
- Dataset split missing or mismatched orderings

Ensure the class gracefully handles these scenarios, perhaps with assertions or fallbacks.

---

### **Summary of key points for implementation**

| Aspect | Details |
|---------|---------|
| Dataset sources | `datasets` (NLP, small datasets), `torchvision` (vision) |
| Tokenization | via HuggingFace `transformers` tokenizer for NLP |
| Preprocessing | Pad, truncate sequences; Resize, normalize images |
| Batching | Use DataLoader, with shuffling during training |
| Data split | Load train/validation/test separately based on config |
| Reproducibility | Set seed; deterministic shuffling |
| Output format | Batches as dicts or tuples with input IDs, masks, labels |

---

### **Next steps**

- Design the DatasetLoader class interface with methods:
  - `__init__(config, split='train', seed=42)`
  - `get_loader() -> DataLoader`
  - List attributes to access dataset size, data type
- Implement load routines for datasets (GLUE/NLP and vision)
- Implement tokenization and transforms
- Make batching configurable and modular
- Prepare for potential importance sampling integration (by exposing importance scores if needed)

---

This detailed analysis should inform your implementation of `dataset_loader.py` so that it reliably loads data, applies necessary preprocessing, and generates batches compatible with the VCAS training pipeline described in the paper.

## evaluation.py

### Logic Analysis for `evaluation.py`

**Purpose:**  
Implement the `Evaluation` class that loads a trained model and dataset, runs inference, and computes relevant evaluation metrics. The class will be used post-training to assess the final performance of the model, ensuring the reproducibility and correctness of the training process, especially when validating the impact of the Variance-Controlled Adaptive Sampling (VCAS) method.

---

### Key Responsibilities:

1. **Model Loading:**  
   - Load the trained model, matching the architecture specified in the configuration (`model.type`), e.g., `'bert-base-uncased'`.
   - Ensure the model is loaded with the trained weights saved after the training process, potentially from a checkpoint.
   - Set the model into evaluation mode (`model.eval()`) to disable dropout, batchnorm updates, and other training-specific behaviors.

2. **Dataset and Tokenizer Initialization:**  
   - Load the dataset specified in the config (`dataset.name`, `dataset.task`, etc.).
   - Initialize the tokenizer (`dataset.tokenizer_name`) and apply it to the dataset.
   - Use consistent data pre-processing to match training conditions (e.g., max sequence length, truncation, padding).

3. **DataLoader Preparation:**  
   - Prepare a DataLoader for the dataset (validation or test split, as appropriate).
   - Use batch size as specified in the configuration or a small size suitable for inference.
   - Set `shuffle=False` to ensure deterministic evaluation order.

4. **Inference Loop:**  
   - Iterate over the evaluation DataLoader:
     - Forward pass the data through the model.
     - Collect outputs (logits, predictions).
   - Use `torch.no_grad()` context to avoid gradient computations, ensuring efficiency.

5. **Metrics Calculation:**  
   - Based on task-specific metrics listed under `evaluation.metrics` (e.g., `'accuracy'`, `'loss'`, possibly `'F1'` if applicable).
   - For classification tasks:
     - Convert logits to predicted labels (`argmax` for multi-class classification).
     - Compute accuracy: number of correct predictions / total samples.
     - Compute loss using the same criterion as in training (e.g., `CrossEntropyLoss`).
     - (Optional) Compute other metrics like F1, if relevant, following task specifics.
   - Store per-batch metrics and aggregate over entire dataset.

6. **Result Reporting:**  
   - Return a dictionary containing computed metrics.
   - Optionally, log the metrics to console or a logging system.
   - Ensure reproducibility: fix random seeds if needed, use deterministic inference.

---

### Implementation Details & Considerations:

- **Model Handling:**
  - Use HuggingFace transformers (`transformers.AutoModelForSequenceClassification`, etc.) for loading the models.
  - Load model weights from the specified checkpoint (e.g., `model_path`); ensure file exists or manage default loading.
  - Device: Move model and data to GPU if available for faster evaluation.
  
- **Dataset Handling:**
  - For GLUE tasks:
    - Use datasets library for loading datasets (`datasets.load_dataset`).
    - Use the same tokenizer and preprocessing pipeline as training.
    - For validation/test split: ensure correct split name, e.g., `'validation'` or `'test'`.

- **Evaluation Metrics Implementation:**
  - Use `scikit-learn`'s `accuracy_score`, `f1_score`, or custom implementations.
  - Compute loss with the same criterion as during training.
  - If `metrics` includes `'loss'`, perform per-batch loss calculation.
  - For classification accuracy:
    ```python
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    ```
- **Hyperparameters:**
  - Use the same max sequence length (`max_seq_length`) from the config.
  - Batch size during evaluation: small (e.g., 16 or 32).
  
- **Reproducibility:**
  - Set fixed seeds for random, numpy, torch.
  - Use deterministic evaluation mode if possible.

- **Output:**
  - Final metrics (accuracy, loss, others) in a dictionary.
  - Optional: print detailed per-epoch results.

---

### Logical Flow Summary:

```plaintext
Initialize:
  - Load model from checkpoint based on `model.type` and config.
  - Set device (GPU/CPU).
  - Load dataset with `datasets` library.
  - Initialize tokenizer with `dataset.tokenizer_name`.
  - Create DataLoader for validation/test data.

Evaluation:
  - Switch model to `eval` mode.
  - Initialize metric accumulators (total correct, total samples, total loss).
  - For each batch in DataLoader:
    - Tokenize and process batch samples.
    - Send data to device.
    - Run `with torch.no_grad()`:
      - Get logits = model(inputs)
    - Compute loss (if configured):
      - Use the same loss criterion used during training.
    - Derive predictions.
    - Compute batch metrics (accuracy, potentially F1).
    - Accumulate metrics.

Post-processing:
  - Calculate overall metrics (accuracy, average loss).
  - Return as dictionary for reporting.
```

---

### Additional Points:
- Ensure that the code handles the possibility of multiple metrics.
- Keep the evaluation deterministic for reproducibility.
- Maintain compatibility with the model and dataset used during training for faithful comparison.

---

This logical analysis aims to guide the implementation of `evaluation.py`, ensuring the class produces accurate, reproducible, and consistent performance assessments aligned with the experimental procedures described in the paper and plan.

## importance_sampler.py

## Logic Analysis for importance_sampler.py

### Objective:
Implement the `ImportanceSampler` class to manage importance ratios, adaptively update these ratios based on gradient variance estimates, and provide sampling functions for data points (e.g., samples in a batch or dataset) and tokens (sequence elements). The class facilitates importance sampling in VCAS, ensuring unbiased estimates and controlled variance.

---

## 1. Class responsibilities:
- Maintain current importance ratios:
  - Activation importance ratios (`rho_l`) for each layer `l`.
  - Weight importance ratios (`nu_l`) for each layer `l`.
- Support adaptive updating of these ratios based on variance estimates:
  - Use variance estimates from the `VarianceEstimator`.
  - Update ratios to keep additional variance within specified thresholds.
- Provide sampling functions:
  - For data points (per batch): `sample_indices()` based on importance scores with current ratios.
  - For tokens (per sequence): `sample_tokens()` based on importance scores with current ratios.
- Initialize with default ratios (`initial_ratio`, e.g., 1.0).
- Store importance scores or importance measures (gradients or leverage scores) to support sampling.

---

## 2. Key inputs and hyperparameters:
- `initial_ratio`
- Importance measurement method (e.g., `gradient_norm`)
- Importance scores for data points and tokens, supplied externally during `sample_indices()`.
- Current ratios `rho_l` and `nu_l` (layer-specific importance ratios).

---

## 3. Components and methods:

### 3.1. Initialization (`__init__`)
- Store initial ratios:
  - Set `self.rho = [initial_ratio for _ in range(L)]` for activation importance ratios.
  - Set `self.nu = [initial_ratio for _ in range(L)]` for weight importance ratios.
- Store method for importance measurement (e.g., `gradient_norm`) as configuration.
- Initialize any internal state for ratio updates, e.g., thresholds, moving averages if needed.
- Optional: prepare data structures for storing importance scores/update history if required.

---

### 3.2. `update_ratios(variance_estimates: dict) -> None`
Input:
- `variance_estimates`: a dictionary containing variance estimates for each layer's gradient.
  - e.g., keys `'activation'`, `'weight'` for overall corresponding variance bounds.
  - Could be further layered into individual layer variances if needed.
- Use thresholds `tau_act` and `tau_w` (from hyperparameters).

Process:
- For each layer `l`:
  - Calculate the desired ratio `rho_l` (activation) and `nu_l` (weight) based on the estimated variance for that layer.
  - The goal:
    - If variance exceeds the target (threshold), decrease ratio `>1` to reduce variance.
    - If variance is below target, possibly increase ratio to allow for larger importance sampling scope.
  - Employ rules, e.g.:
    - `rho_l_new = min(1, rho_l * (variance_estimate / tau_act))`
    - Similarly for `nu_l`.
  - Possibly clamp ratios to `[0, 1]`.

Output:
- Update internal ratios `self.rho` and `self.nu` accordingly.

### 3.3. `sample_indices(importance_scores: Tensor, ratio: float) -> Tensor`
Input:
- `importance_scores`: a 1D tensor of importance scores for data points (`N` samples in current batch).
- `ratio`: current importance ratio for data points, `ρ` (e.g., 0.8).

Process:
- Convert importance scores into sampling probabilities:
  - Normalize importance scores to sum to 1 or to a total importance budget.
  - For example:
    - `p_i = importance_scores[i] / sum(importance_scores)`
  - Scale probabilities so that expected number of samples is `N * ratio`:
    - e.g., `p_i = min(1, importance_scores[i] / threshold)` or normalized to sum to `ratio * N`.
- Use a sampling mechanism:
  - Bernoulli sampling or multinomial:
    - Bernoulli per element with probability `p_i`: construct a mask.
  - Alternatively, select top-k aligned with importance scores:
    - For unbiased sampling, Bernoulli with probabilities per importance.
- Return:
  - A boolean mask tensor (size `N`) indicating which data points are sampled.
  - Sampled indices for gradient computation.

### 3.4. `sample_tokens(token_importance_scores: Tensor, ratio: float) -> Tensor`
Input:
- `token_importance_scores`: 2D tensor of importance scores per sample and token, shape: `[N, T]`.
- `ratio`: importance sampling ratio for tokens, `ν`.

Process:
- Flatten importance scores:
  - For each data point `i` and token `t`, assign importance.
- Convert importance scores to probabilities similarly:
  - Normalize across all tokens to sum to `ratio * N * T`.
  - Use Bernoulli sampling to select tokens:
    - For each token, sample with probability proportional to its importance.
- Create a mask tensor `[N, T]` indicating which tokens are retained.
- Use this mask in backward computations:
  - Scale gradients with `1 / p_{i,t}` for unbiasedness.
- Return:
  - Boolean mask or importance weighting factors.

---

## 4. Additional considerations:
- **Variance bounds enforcement:**
  - Use empirical variance estimates from `variance_estimator.py`.
  - Ratios `rho_l`, `nu_l` controlled so that additional variance remains within tolerances `τ_{act}`, `τ_w`.
- **Multiple importance measures:**
  - Can implement multiple options (`gradient_norm`, leverage scores).
  - For this plan, assume `gradient_norm` as default.
- **Updating schedule:**
  - Ratios are updated periodically, every `F` steps, with `update_ratios()` receiving new variance estimates.
- **Numerical stability:**
  - Ensure the importance probabilities are bounded, e.g., minimum epsilon, maxima of 1.
  - For small importance scores, prevent probabilities of 0.

---

## 5. Data structures:
- Internal parameters:
  - `self.rho` (list or tensor): per-layer activation ratios.
  - `self.nu` (list or tensor): per-layer weight ratios.
- Input importance scores:
  - For data points, a 1D tensor (length `N`).
  - For tokens, a 2D tensor `[N, T]`.
- Sampling functions output:
  - Boolean masks or indices for selected data/tokens.
- Variance estimates stored externally or passed during update.

---

## 6. Summary:
- The `ImportanceSampler` class will:
  - Initialize ratios to 1.
  - Provide `sample_indices()` and `sample_tokens()` that generate importance-weighted Bernoulli samples based on importance scores and current ratios.
  - Update ratios via `update_ratios()` according to estimated variances to control additional variance.
  - Support different importance measurement strategies (mainly gradient norm).

This detailed plan ensures the implementation captures the theory, maintains unbiasedness, controls variance, and supports dynamic importance ratio management during training, aligned with the paper's methodology.

## main.py

# Logic Analysis for main.py

**Objective**: Develop the main entry-script `main.py` that orchestrates the entire experimentation pipeline for reproducing the VCAS method as described. This script should load configuration parameters, initialize all necessary components, execute training, dynamically adapt sample ratios, and evaluate results, ensuring fidelity with the described methodology.

---

## 1. Overview of Responsibilities

- **Load Configuration**: Parse `config.yaml` to extract all hyperparameters, dataset, model, importance sampling, sampling, and evaluation parameters.
- **Initialize Components**:
  - DatasetLoader: load dataset, prepare tokenization, batching.
  - Model: load pre-trained or initialize architecture (BERT, ViT as per config).
  - ImportanceSampler: instantiate with importance method, initial ratio.
  - VarianceEstimator: instantiate with Monte Carlo sample setting.
  - Trainer: set up with model, dataset, importance sampler, variance estimator, hyperparameters.
- **Training Loop**:
  - Execute multiple epochs until total steps or convergence criteria.
  - At intervals (dictated by `update_ratio_frequency`):
    - Compute importance scores via current model and data.
    - Use VarianceEstimator to estimate variance and determine if ratios need updating.
    - Run importance sampling to compute approximate, unbiased gradients with variance control.
    - Update sample ratios (`ρ_l`, `ν_l`) via the `update_ratios()` method based on variance feedback and hyperparameters (`α`, `β`).
  - Log progress, track estimator variance, gradients, loss trajectories.
- **Evaluation**:
  - After training completion, run evaluation on validation/test set.
  - Report metrics: accuracy, loss, and potentially FLOPs reduction as per experiments.
- **Reproducibility & Logging**:
  - Set random seed for reproducibility.
  - Log hyperparameters, ratios, variance estimates, training progress, and evaluation metrics.
- **Optional**: Save model checkpoints/versioning for analysis.

---

## 2. Important Implementation Details & Steps

### a. Load Config Parameters
- Use a YAML parser (`PyYAML`) to parse `config.yaml`.
- Extract:
  - Training hyperparameters: `learning_rate`, `batch_size`, `epochs`, total steps, warmup.
  - Variance threshold hyperparameters: `activation`, `weight`.
  - Importance sampling method: e.g., `'gradient_norm'`.
  - Sampling ratios initial value: `1.0` for both activation and weight.
  - Update frequency for ratios and variance calculations.
  - Monte Carlo sample count (`M`).
  - Model specifications: type, pretraining flag, max sequence length.
  - Dataset specifics: name, task, tokenizer, max tokens.
  - Evaluation intervals and metrics.

### b. DatasetLoader Initialization
- Instantiate DatasetLoader with dataset name, task info, tokenizer name.
- Load:
  - Dataset splits (train, validation) via `datasets` library.
  - Tokenize, encode, batch data with `DataLoader`.
- Prepare data for both training (for importance sampling) and evaluation.

### c. Model Initialization
- Load pre-trained or initialize models based on `model.type`.
- For BERT, use `transformers` library (`AutoModelForSequenceClassification`) with given `pretrained` flag.
- For ViT or others, switch accordingly.
- Ensure model is on appropriate device (`cuda` if available).

### d. ImportanceSampler and VarianceEstimator Initialization
- Instantiate `ImportanceSampler` with:
  - importance method (`'gradient_norm'`).
  - initial ratio (`1.0`).
- Instantiate `VarianceEstimator` with:
  - M (e.g., 4 as default from config).
  - Implement variance estimation methods, which sample importance scores M times per update.

### e. Setup Hyperparameters & Trainer
- Extract from config:
  - Variance thresholds (`τ_act`, `τ_w`).
  - Update step size (`α`).
  - Ratio scaling multiplier (`β`).
  - Variance update frequency (`F`).
  - Total training steps (`total_steps`).
- Instantiate `Trainer` with all components and hyperparameters.
- Initialize current ratios: `ρ_l` and `ν_l` to 1.0 for all layers.

### f. Training Execution
- For each step `t` in total steps:
  - Fetch batch data.
  - If `t % update_ratio_frequency == 0`:
    - Perform variance estimation:
      - Run Monte Carlo sampling of importance scores (with `M` samples).
      - Compute importance scores (gradient norms or leverage scores).
    - Compute current variance estimates for activation and weight sampling.
    - Use `update_ratios()` to adjust `ρ_l, ν_l` per variance estimates and thresholds:
      - If variance > threshold, increase ratios (drop more samples).
      - Else, decrease (sample more data).
  - For each batch:
    - Use importance sampling functions to select data points/tokens for activation and weight gradients.
    - Apply importance sampling masks/scales during backward passes:
      - Modify the backward functions to incorporate masks and importance weights, ensuring unbiasedness.
    - Propagate scaled, masked gradients.
  - Track variances and other metrics for logging.
  - Perform optimizer step, scheduler update (warmup, decay).
- Log training metrics: loss, variance estimates, importance ratios.

### g. Final Evaluation
- After completing all epochs/steps:
  - Run inference on validation/test dataset.
  - Compute accuracy, loss, and other metrics specified.
  - Log and save the final results.
  - Also record FLOPs reduction and wall-clock time if monitored.

### h. Additional Considerations
- Use `torch.cuda.amp` if desired for mixed precision.
- Enable multi-GPU training if applicable, based on hardware and framework.
- Save model checkpoints periodically for analysis.
- Optional: Save sample ratios and importance scores over time for further analysis.

---

## 3. Implementation Notes & Best Practices
- **Determinism**: Fix seed for reproducibility.
- **Efficiency**:
  - Use in-place operations where possible.
  - Implement importance sampling masks as buffers or hooks.
- **Logging**:
  - Use `tqdm` for progress bars.
  - Save logs in structured format (CSV, JSON).
  - Track hyperparameters, ratios, variances, variance thresholds, and gradient norms.
- **Hyperparameter Sensitivity & Validation**:
  - Run small preliminary experiments adjusting `α`, `β`, `τ_act`, `τ_w`.
  - Confirm variance bounds are respected.
  - Ensure importance sampling remains unbiased in implementation.

---

## 4. Handling Edge Cases & Limitations
- Ensuring unbiasedness: carefully scale importance masks.
- Variance estimation: keep M small but sufficient.
- Variance threshold tuning: as per ablation, choose conservative values (e.g., 0.025).
- Large model/dataset considerations: optimize data pipeline, batch size, and multi-GPU settings.

---

## 5. Summary
- The `main.py` acts as the central orchestrator.
- It must perform:
  - Config parsing.
  - Component initialization.
  - Looping over steps/epochs.
  - Variance estimation and ratio updating.
  - Importance sampling with control.
  - Final evaluation and logging.
- Strictly follow the hyperparameters and logic outlined in the paper, ensuring importance sampling, variance control, and adaptation are correctly implemented and integrated.

This thorough analysis provides a clear, actionable blueprint for implementing `main.py` aligned with the paper's methodology and experimental protocol.

## model.py

# Logic Analysis for model.py: Neural Network Architectures & Importance Sampling Integration

This analysis details the design and implementation considerations for the `model.py` module, which handles the definition of neural network architectures (namely BERT and ViT), and incorporates importance sampling mechanisms into the forward and backward passes as specified by the VCAS approach. It relies on HuggingFace transformers for BERT and torchvision or timm for ViT, ensuring the models are instantiated correctly, have hooks or custom modules for importance sampling, and are compatible with the adaptive variance control in the training loop.

---

## I. High-Level Requirements and Objectives

- **Model Definition & Initialization**:
  - Support loading pretrained models (`bert-base-uncased`, `vit-base`) via HuggingFace.
  - Support custom configurations (e.g., `max_seq_length=128`).
  - Enable flexibility for both full models and potential fine-tuning.

- **Integration of Importance Sampling**:
  - Implement importance sampling masks and importance scales on relevant layers.
  - Ensure the sampling procedure remains unbiased (scaling contributions appropriately).
  - Support importance sampling for:
    - Activation gradients (via importance masks on layer outputs/inputs).
    - Weight gradients (via importance masks/scales on weight matrices).

- **Layer and Module Customization**:
  - Use hooks or custom modules to insert importance sampling logic.
  - For BERT: attention layers, feed-forward layers.
  - For ViT: transformer encoder layers, attention, and MLP blocks.

- **Support for Variance Control & Importance Scaling**:
  - Provide methods or hooks to:
    - Scale importance weights during forward/backward passes.
    - Mask out unimportant neurons/token elements dynamically.
  - Maintain unbiasedness while enabling importance-based importance sampling.

---

## II. Architectural Components & Implementation Strategy

### 1. Model Loading & Initialization
- Use transformers library:
  - `AutoModelForSequenceClassification` for BERT (with supporting loading `bert-base-uncased`, pretrained).
  - `AutoModel` or custom ViT class for Vision Transformer.
- Load configurations from parameters:
  - Model type.
  - Pretrained flags.
  - Max sequence length (`max_seq_length=128`).
- Wrap models with additional modules or hooks for importance sampling.

### 2. Importance Sampling Module (Layer Wrappers / Hooks)
- **Design of importance sampling**:
  - Modules should accept importance masks/scales during the forward pass.
  - Masks should zero out less important parts and scale the contributions of sampled parts by inverse probability (to maintain unbiasedness).
- **Implementation options**:
  - Use `register_forward_hook()` or `register_backward_hook()` for each relevant layer.
  - Alternatively, substitute target layers with custom modules (e.g., `ImportanceConvLinearLayer`) that internally handle importance masks/scaling.
- **Layers of interest**:
  - Fully connected layers (Linear modules).
  - Attention layers (self-attention, query, key, value, output projections).
  - LayerNorm and other auxiliary modules should remain untouched; importance sampling applies mainly to gradients, especially activations and weights.

### 3. Forward Pass & Importance Masking
- During `forward()`:
  - Inputs are processed normally.
  - When importance sampling ratios are computed externally, masks are passed down to the network.
  - Masks/scaling factors are applied to:
    - Activation outputs (for activation gradient importance).
    - Layer inputs/outputs (to promote importance-aware sampling).
    - Weight matrices (for weight importance sampling).
- Such importance masks/scales are stored as `buffers` or `register_buffer()` in the layer modules for seamless management during training.

### 4. Unbiasedness & Scaling
- Ensure that importance masks/scales are scaled by the inverse of sampling probability:
  - For a mask `m` (binary), scale by `1/p` where `p` is the probability of sampling that neuron/token/data point.
  - For importance scores, derive `p` proportionally (e.g., norm-based importance).
- During backpropagation:
  - The importance-scaled activation derivatives propagate unbiased estimates.
  - Derivatives of weights accumulate scaled contributions, preserving unbiasedness per the paper’s proof.
- Implement a generalized interface:
  - `apply_importance_mask(layer, mask, scale)`.

### 5. Handling Diverse Layer Types
- **Linear layers**:
  - Inject importance masking/scaling in the weight matrices or during gradient calculation.
- **Attention layers**:
  - Apply importance sampling on attention weights, attention values, and outputs.
  - Token importance masks may zero out certain tokens and scale accordingly.
- **Other layers (e.g., Dropout, LayerNorm)**:
  - Remain standard; importance sampling mainly affects the core linear, attention, and feed-forward layers.

### 6. Forward Hooks / Custom Modules
- Define wrappers for linear and attention modules:
  - These wrappers accept importance masks/scales externally during forward passes.
  - Forward functions multiply masks/scales element-wise on activation outputs or pre-activations.
  - For weights, modify gradient computation via hooks or custom modules to include importance scaling.

### 7. Maintaining Flexibility & Compatibility
- Model modules should:
  - Support injection of importance masks/scales at runtime.
  - Be compatible with training routines and gradient accumulation.
  - Allow toggling importance sampling on/off (for ablation or debugging).

---

## III. Practical Implementation Details & Best Practices

- **Initialization**:
  - Load models with `from_pretrained()` from transformers.
  - Override target layers with custom modules or hooks for importance sampling.
- **Importance Mask / Scale Passing**:
  - During training, importance sampling ratios are computed externally.
  - Pass corresponding importance masks/scales as additional arguments or via buffer tensors.
- **Unbiased Scaling**:
  - Implement importance scales based on probability `p_i`: scale derivatives by `1/p_i`.
  - For data points, generate importance masks during sampling.
- **Layer Hooks**:
  - Use `register_forward_hook()` to modify activations.
  - Use `register_backward-hook()` or hooks on weights for importance-scaled gradients.
- **Handling Non-Linearities**:
  - For ReLU or other non-linear activations, importance sampling applies on derivatives, handled via hooks or custom modules.
- **Efficiency & Compatibility**:
  - Minimize overhead by pre-computing importance masks and only applying during backward passes.
  - Store importance masks efficiently.
  - Use batch operations to scale masks.

---

## IV. Testing & Validation
- Confirm unbiasedness:
  - Validate by checking that the expected gradient estimate matches the full backprop.
  - Use small models/datasets initially.
- Variance control:
  - Monitor gradient variance during training.
- Dynamic importance:
  - Ensure importance masks update logically with the training state.
- Compatibility with training:
  - Ensure no conflicts with optimizer steps, gradient clipping, etc.

---

## V. Summary of Key Implementation Aspects
| Component | Details |
| --------- | --------|
| Loading models | Load pretrained BERT/ViT via transformers, support configuration options |
| Importance masks | Implement via hooks or custom layers, pass dynamically, scaled by importance probability |
| Unbiasedness | Scale gradients with importance sampling weights (`1/p_i`) |
| Variance control | Provide interfaces to update importance sampling ratios dynamically, based on variance estimates |
| Compatibility | Leave core models unchanged, integrate importance sampling transparently for backward pass |
| Testing | Validate unbiasedness, variance bounds, training convergence, and final metrics |

---

This thorough analysis provides a comprehensive blueprint for implementing `model.py` aligned with the paper’s methodology, ensuring importance sampling with unbiased gradient estimates and variance control integrated into modern transformer-based models.

## trainer.py

# Logic Analysis for trainer.py

This document systematically dissects the implementation logic necessary for the `Trainer` class within `trainer.py`, following the research paper’s methodology, the provided plan, and the given configuration. It provides an in-depth conceptual map to guide faithful coding aligned with the VCAS approach, ensuring the process is unbiased, variance-controlled, and adaptive.

---

## 1. **Class Overview & Responsibilities**

- **Primary role:** Orchestrate the training loop for a deep neural network (e.g., BERT or ViT) applying Variance-Controlled Adaptive Sampling (VCAS).
- **Sub-tasks:**
  - Initialize with model, dataset, importance sampler, variance estimator, and relevant hyperparameters.
  - Execute multiple training epochs, implementing:
    - Data loading with batch formation.
    - Importance sampling to select informative data points and tokens.
    - Application of importance masks/scales to gradients.
    - Variance estimation at scheduled intervals.
    - Adaptive adjustment of sample ratios (`rho_l` for activation sampling, `nu_l` for weight sampling).
    - Logging and evaluation at specified intervals.

---

## 2. **Initialization**

- **Inputs:**
  - **Model instance:** e.g., `BertForSequenceClassification` or Vision model.
  - **Dataset:** e.g., GLUE SST-2, with tokenization.
  - **Importance Sampler:** a class to manage importance ratios and provide sampling indices.
  - **Variance Estimator:** a class to estimate gradient variance via Monte Carlo sampling.
  - **Hyperparameters:** thresholds (`tau_act`, `tau_w`), update steps (`alpha`, `beta`), update frequency (`F`), number of MC samples (`M`), etc.

- **Setup:**
  - Initialize data loaders for training and evaluation splits.
  - Set default sample ratios (`rho_l`, `nu_l`) as per configuration; these will be dynamically updated.
  - Store variance thresholds for adaptive control.
  - Set up logging, metrics, optimizer, and scheduler.

---

## 3. **Main Training Loop (`train()`)**

1. **Loop over epochs/steps:**
   - For each epoch, iterate over data loader batches.
   - For each batch:
     1. **Preprocessing:**
        - Tokenize data (if not pre-tokenized).
        - Attach batch-specific metadata if needed.

     2. **Importance Sampling for Current Batch:**
        - Determine importance scores for data points and tokens:
          - For activation gradients, importance scores are based on current gradient norm estimates.
          - For weight gradients, importance scores based on leverage scores (via SVD or similar).
        - Use `importance_sampler.sample_indices()` with current `rho_l`, `nu_l` to get sampling indices:
          - For data points: select a subset based on importance probabilities.
          - For tokens: select token indices within data points accordingly.
        - *Note:* These importance scores need to be computed *before* the backward pass, possibly by estimating the current importance scores based on previous iteration or a quick approximation.

     3. **Forward pass:**
        - Run `model.forward()` with the batch to get outputs/logits.
        - Compute loss with labels.

     4. **Backward pass with importance masks:**
        - For each layer:
          - **Apply importance masks/scaling:**
            - Using importance weights derived from sampling probabilities.
            - For activation gradients:
              - Modify the output gradients to zero out unimportant samples _(via importance masks)_.
              - Scale retained gradients by `1/p_i` to maintain unbiasedness.
            - For weight gradients:
              - Similarly, mask and scale appropriately based on importance scores.
          - **Implementation note:**
            - Use hooks or custom modules to inject importance masks/scales into the backward passes.
            - For layers like Linear:
              - Mask gradients w.r.t. activations and weights.
              - For convolutional layers, similar but with adapted importance scores.
            - **Note:** For some layers (e.g., CNN convolution), only activation sampling may be feasible, as per limitations.

        - Invoke `loss.backward()` respecting importance masks/scales.

     5. **Variance estimation and ratio adaptation (every `F` steps):**
        - If current step % `F` == 0:
          - **Variance estimation:**
            - Use `variance_estimator.estimate_variance()`:
              - Perform `M` MC samples of importance sampling.
              - For each, measure variance contributions of activation and weight gradient approximations.
          - **Adjust sample ratios (`rho_l`, `nu_l`):**
            - Based on variance estimates:
              - If variance exceeds thresholds (`tau_act`, `tau_w`):
                - Increase sample ratios to reduce variance.
              - Else:
                - Decrease or maintain ratios to encourage higher speed.
            - **Update rules:**
              - For activation ratios (`rho_l`):
                - Use proportional updates via `s` (gradient norm preserving ratio) and thresholds.
                - Implement as per Eq. 4 and Eq. 5 in the paper.
              - For weight ratios (`nu_l`):
                - Update based on variance feedback following Eq. 7.
              - Use hyperparameters `alpha` (step size) and `beta` (scaling factor).

          - **Update importance ratios:**
            - Invoke `importance_sampler.update_ratios()` with new ratios (`rho_l`, `nu_l`).

     6. **Optimization step:**
        - Call `optimizer.step()` for parameter update.
        - Zero gradients before next iteration.

     7. **Logging:**
        - Track training loss, variances, importance ratios, and variances.
        - Store metrics for convergence analysis.

2. **Repeat for all steps or until convergence criteria (e.g., total steps, validation thresholds)**:

   - **Optional early stopping:** based on validation accuracy or loss plateau.

---

## 4. **Variance Control and Ratio Adjustment (`adjust_ratios()`)**

- **Purpose:** Implemented during every `F` steps, this function will read variance estimates, compare with thresholds, and adjust `rho_l` and `nu_l`.
- **Implementation:**
  - Collect variance estimates for activation (`V_act`) and weight (`V_w`) gradients.
  - Apply the formulas from Sec. 5 for adaptation:
    - For activation: update ratio `s` using gradient sparsity and thresholds.
    - For each layer: update `nu_l` to control variance in weight gradients.
  - Enforce monotonicity or prespecified bounds (e.g., [0.1, 1.0]) to prevent unstable ratios.

---

## 5. **Handling Importance Sampling During Backward Pass**

- **Implementation options:**
  - Use backward hooks in PyTorch (`register_backward_hook`) or custom layer modules to:
    - Zero out importance-insignificant gradient entries.
    - Scale retained entries by importance weights.
  - **Key idea:** the actual gradient tensors are scaled element-wise by importance weights. 
  - **Ensure:** the importance sampling remains **unbiased**, i.e., expectations are exact, by scaling inversely proportional to sampling probability.

- **Layer-specific considerations:**
  - For linear layers:
    - Mask and scale weight gradients.
  - For activation gradients:
    - Mask gradients before propagation.
  - For non-linear layers:
    - Ensure that importance masks are applied consistently (gradients are multiplied element-wise by importance masks and scaled).

---

## 6. **Post-Training and Evaluation**

- **Finalize model:**
  - Save checkpoint after training.
  - Restore the full model parameters (unbiased estimates).
- **Validation:**
  - Run inference on validation set.
  - Compute accuracy and loss.
- **Metrics logging:**
  - Compare with baseline (full exact) training.

---

## 7. **Additional Notes**

- **Hyperparameters management:**
  - Load from config.yaml.
  - Implement sensible defaults with options for tuning.
- **Logging & Monitoring:**
  - Use `tqdm` for progress.
  - Log variances, importance ratios, and convergence metrics periodically.
- **Parallelization & Hardware:**
  - Ensure training compatibility with multi-GPU/DistributedDataParallel if needed.
  - The importance sampling and variance estimation logic should be optimized for GPU efficiency.

---

## Summary

The `Trainer` class will operate in a structured loop comprised of forward, importance-based masking, loss calculation, scaled backward, variance estimation, and ratio adjustment. It will leverage the importance to perform unbiased sampling, maintain the variance within bounds, and adaptively control sampling ratios over training.

This detailed logic serves as the blueprint for faithful, efficient implementation aligned with the methodological advances presented in the paper.

## variance_estimator.py

{
  "variant": "variance_estimator.py",
  "purpose": "Implement the VarianceEstimator class, which performs Monte Carlo sampling to estimate the variance of gradient estimates for the importance sampling importance. This variance estimate informs the adaptive sampling ratios, ensuring the additional variance introduced by approximations remains controlled and within thresholds as per the VCAS method.",
  "core responsibilities": [
    "Receive a batch of data and importance scores (or importance metrics) for activation and weight gradients.",
    "Sample importance importance indices (data points and tokens) according to importance scores, using importance sampling based on importance scores (e.g., importance weights proportional to gradient norms or leverage scores).",
    "Compute multiple importance samples (specified by parameter M) to empirically estimate the variance of the gradient estimators.",
    "Estimate the variance of activation gradients and weight gradients separately, as these are used in the importance ratio update rules.",
    "Return the estimated variances, which are used to adjust importance sampling ratios and hyperparameters in the main training loop."
  ],
  "inputs": {
    "dataset_batch": "A batch of data (X, y) containing input features and labels, used as the basis for importance sampling. For NLP, corresponds to tokenized sequences; for vision, raw images. The batch provides the data size N and dimensional structure.",
    "importance_scores": {
      "activation": "Tensor representing importance scores per data point, often derived from gradient norms (e.g., \(\| G_i \|_F\)) or other importance measures.",
      "weight": "Tensor representing importance scores per token or per connection, possibly derived from leverage scores or importance estimates."
    },
    "parameters": {
      "M": "Number of importance samples (Monte Carlo repetitions) to perform for variance estimation; small (e.g., 2-10) to limit overhead.",
      "importance_method": "Method to determine importance scores; for this context, 'gradient_norm' as specified in the config. or 'leverage_score'."
    }
  },
  "outputs": {
    "activation_variance_estimate": "Scalar estimate of the variance of the activation gradient estimator based on importance sampling. Calculated as empirical variance over M samples.",
    "weight_variance_estimate": "Scalar estimate of the variance of the weight gradient estimator, similarly derived.",
    "additional_info": "Optionally, can return variance components for individual layers if needed for layer-wise variance control."
  },
  "step-by-step process": [
    "Initialize storage vectors: - For each importance sample (i from 1 to M):\n    - Perform importance sampling to select a subset of data points (or tokens for weight gradients) based on importance scores. Use the current importance ratios as sampling probabilities.\n    - When sampling, generate importance weights \(w_i\) = 1 / probability for each selected index to ensure unbiasedness.\n    - Run an importance sampling iteration: compute the importance-sampled gradient estimates (activation and weight) for that sample.\n    - Store the importance-sampled gradient estimates and the importance weights.\n",
    "Calculate the empirical mean of the importance-sampled gradient estimates over M samples.\n",
    "Compute the empirical variance for each set of gradient estimates:\n    - For activation gradients: \(\text{Var}_act = \frac{1}{M-1} \sum_{i=1}^M (\hat{G}_i - \bar{G})^2\)\n    - For weight gradients: \(\text{Var}_w = \frac{1}{M-1} \sum_{i=1}^M (\hat{\nabla}_{\theta_i} - \overline{\nabla})^2\)\n",
    "Return these variances as the variance estimates.\n",
    "Ensure that the sampling procedure maintains unbiasedness by scaling importance weights as \(w_i = 1/p_i\), where \(p_i\) is the importance sampling probability for index \(i\).\n",
    "Optionally, parallelize the process: perform importance sampling iterations in parallel, then aggregate results.\n",
    "Implement validation/consistency checks, such as confirming that importance weights sum to approximately one or that variance estimates are stable across successive calls.\n"
  ],
  "noteworthy considerations": [
    "Ensure that importance scores are computed consistently with the importance method specified ('gradient_norm' or 'leverage_score').",
    "When importance scores are derived from gradient norms, compute the norms using the current importance sampling masks if present or from the full gradients during a reference pass.",
    "To reduce overhead, M should be small (e.g., 2–4), balancing estimation accuracy and computational cost.",
    "Variance estimates guide the adapted sample ratios in the main training loop; therefore, the estimates must be as precise as possible but not too costly.",
    "Maintain the unbiasedness of gradient estimates by proper importance weight scaling."
  ],
  "additional notes": [
    "This class should be called periodically (e.g., every 'importance_sampling' update step) in the training loop to provide updated variance estimates for adaptive ratio adjustments.",
    "It must align with the structure and data flow of the existing training pipeline, accepting batch data, importance scores, and current sampling ratios as inputs."
  ],
  "summary": "Design the VarianceEstimator class with a method 'estimate_variance' that performs M importance sampling iterations over the current batch, computes importance-weighted gradient estimates for activation and weight gradients, and calculates empirical variances. These variances are then used to adapt importance sampling ratios in VCAS, ensuring the additional variance remains within prescribed thresholds, thus stabilizing training dynamics."
}

