# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset.py

**Logic Analysis for dataset.py (WeatherDataset class)**

---

### Purpose:
Implement a robust WeatherDataset class responsible for loading, preprocessing, and providing data loaders for training, validation, and testing ERA5 reanalysis data, conforming to the details in the paper and configuration.

---

### Core Responsibilities:
1. Data loading:
   - Read raw ERA5 data from the specified netCDF file (`data_path`).
   - Select variables grouped as surface and atmospheric, based on configuration.
   - Extract data for three datasets:
     - Training: 1979–2018
     - Validation: 2019
     - Testing: 2020
2. Data preprocessing:
   - Regrid data to target grid resolution (128×256).
   - Normalize variables:
     - Calculate mean and std over training data for each variable.
     - Standardize all data (subtract mean, divide by std).
   - Store normalized tensors for efficient retrieval.
3. Data sampling:
   - Generate input-output pairs:
     - Initial state `X_0` at a timestamp.
     - Target future state `X_T` at lead time `T` (1-14 days), with step size 6 hours.
     - Compute the *difference* (delta) `Δ_T = X_T - X_0`, necessary for training with predictive delta modeling.
   - Support sampling sequences with variable lead times, possibly with random sampling, respecting the ranges specified in the config.
4. Data augmentation:
   - When needed, perform random cropping for patches based on the patch size (2 or 4).
   - Use or implement pressure-level weights as specified for weighted loss computation.
5. Data loader:
   - Provide PyTorch DataLoader objects for train, validation, and test datasets with batching, shuffling, and possible data augmentation.
6. Additional considerations:
   - Ensure data is aligned temporally.
   - Handle multi-variable, multi-pressure level data effectively.
   - Store preprocessed data efficiently (e.g., torch tensors).
   - Handle normalization consistently (calculate and save normalization stats during data initialization).

---

### Key Steps and Details:
#### 1. Data Loading
- Use xarray/netCDF4 to read the ERA5 netCDF file specified by `data_path`.
- Parse time dimension; load data between specified date ranges.
- Variables:
  - For surface: 'T2m', 'U10', 'V10', 'MSLP'
  - For atmosphere: 'Z', 'T', 'U', 'V', 'Q' at pressure levels `[50, 100, ..., 1000]`.
- Read all required variables at all pressure levels.
- Possibly subset spatially if needed or perform regridding.

#### 2. Regridding / Spatial Resampling
- Use a regridding method (bilinear interpolation) to match target grid size (128×256).
- Can be done via `xarray` interpolation or other spatial interpolation methods.
- Store regridded data for fast access.

#### 3. Normalization
- During data preparation:
  - Compute means and standard deviations for each variable over the training set:
    - For each variable, accumulate sum and sum of squares over all training data points.
  - Normalize:
    - For each dataset (train/val/test), subtract mean and divide by std.
- Store normalization stats within the dataset object for consistent application.

#### 4. Sequence Generation for Model Input:
- For each starting timestamp `t`, generate:
  - `X_0`: state at `t`.
  - `X_{T}`: state at `t + T * 6 hours`.
  - Compute delta `Δ_T = X_{T} - X_{0}`.
- Handling lead times:
  - For each lead time T (1-14 days → 1-14×24 hours / 6 hours steps), compute corresponding index offset.
- Limit sequences to avoid crossing data boundaries.

#### 5. Sampling Strategy:
- During training:
  - Randomly sample initial timestamps and lead times.
  - Potentially randomize `delta t` among `[6, 12, 24]` hours if configured.
- During validation/testing:
  - Use fixed sequences from the dataset to evaluate performance at specified lead times.
- Store index mappings for quick sampling during training.

#### 6. Data Augmentation:
- Extract data patches with size `patch_size` to increase robustness.
- Might implement random cropping or sliding windows.

#### 7. Data Structures:
- Organize data as tensors:
  - `X`: (time_steps, V, H, W)
  - Store separate tensors or dictionaries per variable, pressure level, and dataset split.
- Maintain normalization statistics:
  - per variable mean and std.
- Implement helper functions for delta computation, normalization, and indexing.

---

### Implementation Details:
- **Initialization:**
  - Load raw data.
  - Compute normalization stats on training data.
  - Regrid data.
  - Normalize and store in memory for fast access.
- **Data retrieval:**
  - For a given index:
    - Retrieve `X_0`.
    - Retrieve `X_T` at the specified lead time.
    - Compute delta `Δ_T`.
    - Return `X_0`, `Δ_T` (normalized), plus auxiliary info if needed.
- **Batching:**
  - collate function gathers individual samples into batch tensors.
  - Ensure proper batching dimensions: `(batch_size, V, H, W)`.

### Additional Considerations:
- Consistency of normalization between train/val/test.
- Efficient storage (possibly using memory-mapped files or reduced precision).
- Optionally support variable pressure weighting during loss calculation, which should be implemented in the training loop, not the dataset module per se.

---

### Summary:
The dataset.py file implementing `WeatherDataset` should:
- Load and pre-process ERA5 data on initialization.
- Store normalization stats based on training data.
- Provide methods to sample paired sequences with specified lead times.
- Support data augmentation with patches if required.
- Efficiently return tensors ready for model input and training.
- Be flexible enough to produce input-target pairs for multiple lead times and δt configurations during training and evaluation.

This detailed plan ensures reproducibility, fidelity to the paper's methodology, and efficiency in data handling.

## evaluation.py

# Logic Analysis for `evaluation.py`

This analysis delineates the precise logic and implementation considerations necessary to develop the `EvaluationMetrics` class, which computes evaluation metrics—RMSE, ACC, SSR—over forecast horizons with support for regridding and pressure-based weighting, relying on torch and numpy libraries.

---

# 1. Purpose & Scope
- **Goal**: Develop a class `EvaluationMetrics` to evaluate weather forecast model outputs against ground truth, supporting:
  - Computation of key metrics:
    - **RMSE** (Root Mean Square Error)
    - **ACC** (Anomaly Correlation Coefficient)
    - **SSR** (Spatial Similarity Score; to be defined)
  - Handling multi-variable, multi-lead-time evaluation
  - Regridding forecast and ground truth data to a common spatial grid
  - Applying pressure-level weighting based on the variable's pressure level
  - Support for latitude-weighted metrics
- **Context**: Applied post-inference, on model predictions stored as tensors, compared to ground truth tensors, over the forecast lead times.

# 2. Input Data & Preprocessing
- **Inputs for metrics calculation**:
  - `preds`: list of forecast tensors, shape `(N_samples, V, H, W)` (or batched) or a single forecast tensor.
  - `targets`: ground truth tensor `(V, H, W)` (or `(N_samples, V, H, W)` if batch)
- **Ground truth and forecast tensors**:
  - Variables (`V`) include multiple atmospheric variables defined in the config (e.g., T2m, U10, V10, etc.)
  - Spatial grid resolution may differ from forecast to evaluation reference; **regridding** needed before metric calculation.
- **Regridding**:
  - If regridding is enabled (`regrid_data==True`), resample both forecasts and ground truths onto the common grid `(128, 256)` (from config).
  - Use a suitable method: `xarray`, `scipy.interpolate`, or similar; likely, nearest-neighbor or bilinear interpolation.
  - **Note**: Regridding is crucial for spatially consistent evaluation metrics.

# 3. Handling Pressure Variables
- **Pressure weighting**:
  - For pressure-level variables, weights are assigned proportional to pressure levels (e.g., `w=1` at 50hPa, `w=0.1` at surface variables).
  - **Implementation**:
    - Maintain a mapping/dictionary: `pressure_weights = {50:1, 100:1, ... , 1000:0.1}`
    - For each pressure level variable, multiply the squared error or covariance metrics by the corresponding weight.
  - **Variable weights**:
    - For surface variables (T2m, U10, V10, MSLP), apply default weights (e.g., 1.0 for T2m, 0.1 for others).

# 4. Latitudinal weighting
- **Purpose**: To account for the non-uniformity of grid cells in spherical coordinates.
- **Implementation**:
  - Use latitude grid (`lat`) as a vector of size `H` (height index/latitude index).
  - Calculate latitude weights as `cos(lat)` or other common scheme.
  - Apply these weights to metrics over the latitudes, normalizing so weights sum to unity for unbiased estimate.

# 5. Metric Computation
Definitions:
- **RMSE**:
  \[
  RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{X}_i - X_i)^2}
  \]
  applied per variable, possibly over all spatial points, weighted by latitude.
  
- **ACC (Anomaly Correlation Coefficient)**:
  \[
  ACC = \frac{\sum_{i} (\hat{X}_i - \bar{\hat{X}})(X_i - \bar{X})}{\sqrt{\sum_{i} (\hat{X}_i - \bar{\hat{X}})^2 \sum_{i} (X_i - \bar{X})^2}}
  \]
  where \(\bar{\hat{X}}\) and \(\bar{X}\) are means over relevant spatial points, weighted appropriately.
  
- **SSR (Spatial Similarity Score)**:
  - **Definition needed**, but assuming it measures spatial pattern similarity—possibly a normalized spatial correlation or a measure like Structural Similarity Index (SSIM).
  - **Implementation**:
    - For each variable, compute a normalized spatial correlation between forecast and truth.
    - Averaged over samples/ensemble members if needed.
  
**Note**: Due to the lack of explicit SSR from the paper, assume a placeholder metric using spatial Pearson correlation or a simple index.

---

# 6. Implementation Details & Constraints
- **Batch processing**:
  - Support batch inputs for `preds` and `targets` for efficiency.
  - Loop over samples if ensemble forecast; average metrics for ensemble output.
- **Weighted masking**:
  - Compose a mask array if some regions or variables need exclusion.
  - Apply latitude and pressure weights during metric calculation for each variable.
  
- **Multiple lead times**:
  - Input forecast data should be structured accordingly (e.g., separate evaluation for 1-day, 3-day, etc.).
  - Provide functions accepting lead time as a parameter.

- **Variable index mapping**:
  - Map variable names to index positions for clarity.
  - Use config or input parameters for variable list and pressure levels.

---

# 7. Output Specification
- **Return values**:
  - For each metric, produce a dictionary:
    ```python
    {
      'variable_name': {
        'rmse': value,
        'acc': value,
        'ssr': value # assumed metric
      },
      ...
    }
    ```
- **Overall scores**:
  - Optionally, aggregate over all variables with weights for a global metric.

---

# 8. Additional Considerations
- **Diagnostics & validation**:
  - Plot sample error maps.
  - Log per-variable vs. overall metrics.
- **Robustness**:
  - Handle missing data or NaNs cautiously (skip or mask).
  - Ensure no divide-by-zero errors in correlation calculations.

---

# Summary:
- Utilize provided utilities for pressure weighting and positional encoding.
- Regrid forecast and target if necessary.
- Compute mean squared errors, spatial correlations, and anomalies.
- Apply pressure and latitude weights during calculations.
- Output metrics as structured dictionaries, supporting batch/ensemble inputs.
- Design the class to be adaptable for different lead times and variables, ensuring compatibility with the evaluation protocol as outlined.

This structured outline provides the complete, detailed logical flow necessary for accurate implementation of `EvaluationMetrics`. Follow this plan to develop a robust, reproducible evaluation module aligned with the paper’s experimental protocol.

## inference.py

# Logic Analysis for inference.py

This document details the logical foundations, functional components, and implementation steps necessary to develop `inference.py`, which provides the `InferencePipeline` class. This class handles the generation of multiple weather forecast trajectories based on variable interval combinations, performs iterative model rollouts, and combines the individual forecasts into an ensembled prediction. It also relies on auxiliary functions from `utils.py` for interval generation and forecast aggregation, and on methods from `model.py` for inference.

---

# 1. Overall Purpose and Responsibilities

- Facilitate efficient, flexible forecast generation for a specified lead time `T`.
- Generate diverse sequences of forecast steps (`\delta t` intervals) that sum to `T`.
- Execute iterative rollouts of the trained model, conditioned on these intervals.
- Aggregate multiple forecast trajectories into a single ensemble forecast.
- Support evaluation procedures comparing ensemble outputs against true validation/test data.

---

# 2. Key Inputs and Outputs

## Inputs:
- **Initial condition tensor** (`initial_condition`): The atmospheric state tensor with shape `(V, H, W)`, representing variables at initial time.
- **Target lead time** (`lead_time`): The forecast horizon period, in days or hours.
- **Mode** (`heterogeneous` or `homogeneous`): Defines whether to generate diverse interval sequences (heterogeneous) or same-interval sequences (homogeneous).
- **Optional parameters**:
  - Number of ensemble candidates (`n`): total number of interval combinations evaluated.
  - Number of top combinations (`m`): number of selected sequences to produce forecast ensemble.
  - Combining method (`'mean'` or `'ensemble'`): how to aggregate individual forecasts.

## Outputs:
- **Forecast tensor** (`forecast`): The predicted weather state after `T`, shape `(V, H, W)` (or batch if batching is supported).
- **Optional**: also returns forecasts for each interval combination, for analysis of uncertainty or ensemble diversity.

---

# 3. Core Functional Components

### 3.1. Interval Set Generation
- **Functionality**:
  - Generate multiple sequences of `\delta t` intervals summing approximately to the lead time `T`. 
  - Support both:
    - **Homogeneous combinations**: sequences with identical `\delta t` (e.g., all 6-hour steps).
    - **Heterogeneous combinations**: diverse, mixed-interval sequences.
  - Use utility functions from `utils.py` for combinatorics, e.g., `generate_intervals()`.

- **Implementation details**:
  - Given `T` (hours), generate all sequences of `\delta t` values where sum ≈ `T`.
  - For `n` combinations, randomly generate or select top-sequences based on validation loss (if available).
  - Incorporate validation if the interface supports it, to select the best sequences.

### 3.2. Forecast Rollout for a Sequence
- **Functionality**:
  - Given an initial state and a sequence of intervals `[d1, d2, ..., dk]` that sum roughly to `T`, produce a forecast by:
    - Iteratively feeding the model’s output as the input state.
    - Conditioning each step on the corresponding `\delta t` via model inputs.
  - The model's forward function takes in the current state and `\delta t`, outputs the predicted change (`\Delta \hat`) for that interval.
  - Update state: add predicted change to current state or incorporate in other domain-appropriate manner.

- **Implementation details**:
  - Loop over sequence of intervals.
  - For each interval:
    - Input current state and interval `\delta t` into the model.
    - Obtain predicted `\Delta \hat`.
    - Update current state: `X_{new} = X_{old} + \Delta \hat`.
  - After completing all steps in the sequence, output final forecast state.

### 3.3. Forecast Aggregation and Ensembling
- **Functionality**:
  - Perform the above forecast for each sequence in the selected combination set (top `m`).
  - Aggregate forecasts:
    - If `combine_method` is `'mean'`, average the forecast tensors.
    - If using more advanced ensemble methods, incorporate variability (e.g., variance, quantiles) if desired.

- **Implementation details**:
  - Use utility functions from `utils.py`, e.g., `ensemble_average(forecasts_list)`.
  - Return the ensembled forecast for downstream evaluation or visualization.

---

# 4. Logical Flow via Class Methods

### 4.1. Initialization (`__init__`)
- Instantiate the trained model object.
- Store configuration (e.g., `lead_time`, `n`, `m`, `mode`, combining method).
- Load necessary utility functions.

### 4.2. Interval Generation (`generate_intervals()`)
- Call `utils.py.generate_interval_combinations()` with parameters:
  - Target lead time `T` (in hours).
  - Mode (`heterogeneous` or `homogeneous`).
  - Number of combinations `n`.
- Obtain a list of sequences `[d1, d2, ..., dk]`.

### 4.3. Forecast Generation (`predict_one_sequence()`)
- For each sequence:
  - Initialize current state to initial_condition.
  - Loop over delta t intervals:
    - Normalize or prepare input tokens, including delta t embedding.
    - Call model's inference method, e.g., `model.forward()` or `model.predict_delta()`.
    - Convert model output into forecast update.
    - Update current state.
  - Save final forecast for this sequence.
- Collect all forecasts in a list.

### 4.4. Ensemble Forecasting (`ensemble_forecasts()`)
- Aggregate all sequence forecasts via averaging.
- Return final ensembled forecast tensor.

### 4.5. Main Interface: `generate_forecast()`
- Accept initial condition and target lead time.
- Generate interval sequences.
- For each, produce forecast via iterative rollout.
- Combine forecasts into final output.
- Return forecast tensor and optionally, individual forecasts.

---

# 5. Additional Considerations

## 5.1. Handling Variable Normalization
- Use stored normalization statistics from `dataset.py` or preprocessed data.
- Ensure consistent scale between training and inference.

## 5.2. Supporting Different Exercise Modes
- Support only the primary mode (`heterogeneous`) as default.
- Optionally support homogeneous combinations for efficiency.
- Parameterize the number of combinations and ensemble size.

## 5.3. Performance Optimization
- Batch multiple sequences if parallelization is feasible.
- Use mixed precision inference if supported.
- Leverage model's `.eval()` mode for inference reliability.

## 5.4. Error Handling
- Check that the sum of intervals approximates `T`.
- Handle edge cases where no valid sequences are generated.
- Log and monitor forecast ensemble diversity and confidence estimates.

---

# 6. Summary of Key Functions / Methods

| Function / Method | Purpose | Inputs | Outputs | Dependences |
|---------------------|---------|---------|-----------|--------------|
| `generate_intervals(T, mode, n)` | Create multiple interval sequences summing to `T` | `T`, mode (`hetero/homog`), `n` | List of sequences | `utils.py` |
| `predict_for_sequence(initial_state, sequence)` | Rollout forecast along a sequence | `initial_state`, sequence of `δt` | Forecast tensor | `model.py` outputs |
| `ensemble_predictions(forecast_list)` | Combine multiple forecasts | List of forecast tensors | Single ensembled forecast | `utils.py` |
| `generate_forecast(initial_condition, T)` | Main method for full forecast | initial condition, lead time | Final forecast tensor | Combines above functions |

---

# 7. Clarifications Needed for Implementation
- Exact normalization and variable preprocessing steps.
- Specific delta_t embedding strategy in `model.py`.
- Implementation details of `model.forward()` regarding conditioning.
- Handling of extremely long lead times (e.g., beyond the max combination length).
- Whether to support deterministic forecasts (single sequence) or stochastic samples (multiple sequences and added noise).

---

# Conclusion
This logic analysis establishes a comprehensive foundation for implementing `inference.py`. It defines procedures for interval set generation, iterative forecast rollouts conditioned on variable-length intervals, ensemble aggregation, and flexibility for different inference modes, all aligned with the Stormer methodology depicted in the paper. Implementation should strictly follow this structure, utilizing the designated utility functions, model interfaces, and configuration parameters to ensure reproducibility and fidelity.

## main.py

# Main.py Logic Analysis for Reproducing Stormer Weather Forecasting Framework

This analysis breaks down the necessary steps, logic, and flow to implement the `main.py` script, ensuring faithful reproduction of the Stormer methodology, as described in the paper and aligned with the provided configurations. The goal is to organize the experiment pipeline cohesively, from data loading to final evaluation, while referencing other modules and ensuring consistency with the detailed design and experimental protocol.

---

# 1. Imports & Dependencies

- Parse configuration: Use `PyYAML` to load `config.yaml`.
- Set up logging (e.g., `logging` module) for tracking experiment progress.
- Initialize hardware/distributed environment:
  - Use `torch.distributed` or `torch.nn.parallel` as per `cluster_strategy`.
  - Set device count: `128 GPUs` with `mixed_precision`.
- Import all modules:
  - `dataset.py`: (`WeatherDataset`)
  - `model.py`: (`TransformerModel`)
  - `trainer.py`: (`TrainerLightning`)
  - `inference.py`: (`InferencePipeline`)
  - `evaluation.py`: (`EvaluationMetrics`)
  - `utils.py`: (utility functions such as generating interval combinations)
- Any additional helpers (`os`, `time`, `numpy`, etc.).

---

# 2. Load Config and Set Up Environment

- Read `config.yaml` and store in a variable (e.g., `cfg`).
- Set device configuration (e.g., `torch.device`) based on available hardware.
- Initialize distributed training if specified:
  - Use `torch.distributed.launch` or `torch.nn.parallel.DistributedDataParallel`.
- Configure mixed precision:
  - Enable `torch.cuda.amp` autocast if `use_mixed_precision` is True.
- Set random seed for reproducibility if specified in config.

---

# 3. Data Preparation

- Instantiate `WeatherDataset`:
  - Pass parameters from `cfg['dataset']`:
    - `data_path`
    - Variables list (surface + atmospheric as specified)
    - Pressure levels
    - Downsampling hours
    - Grid resolution
    - Data splits (train, validation, test date ranges)
- Obtain PyTorch DataLoaders for train, validation, test:
  - Use standard batch size from `cfg['training']['batch_size']`.
  - Enable sharding/shuffle for training loader.
  - Apply any collate functions, normalization, or regridding as per design.
- Log dataset sizes and normalization stats.

---

# 4. Model Instantiation

- Create `TransformerModel`:
  - Pass model parameters (`patch_size`, `hidden_dim`, `num_layers`, `num_heads`, `mlp_ratio`, dropout rates, positional encoding configs).
- Include weather-specific embedding:
  - Variable tokenization
  - Variable aggregation (cross-attention)
- Initialize positional encodings appropriately.
- Define AdaLN modules conditioned on delta_t within transformer blocks:
  - MLP sizes for AdaLN parameters, consistent with the paper (e.g., 2-layer MLP).
- Load or initialize model weights.
- Log model configuration and parameter count.

---

# 5. Trainer Setup

- Instantiate `TrainerLightning`:
  - Pass model instance.
  - Pass optimizer configurations:
    - AdamW optimizer
    - Learning rates for phases
    - Weight decay
  - Scheduler:
    - Cosine schedule
    - Warmup epochs (10 for phase 1, 5 for finetuning)
  - Enable gradient checkpointing, mixed precision.
  - Set maximum epochs: 100 for phase 1; 20 for each finetuning phase.
- Set early stopping criteria:
  - Based on validation loss (aggregated over loss components and select lead times).
- Save checkpoint directory and checkpoint interval from config.
- Implement training loop or call `Trainer.fit()`.

---

# 6. Training Procedure

- **Phase 1**:
  - Train for 100 epochs on single-step forecasting objective.
  - For each batch:
    - Randomize $\delta t$ from `randomized_delta_t` list (`6,12,24` hr).
    - Compute target $\Delta_{\delta t}$.
    - Forward pass through model with input data and conditioning $\delta t$.
    - Compute pressure-weighted MSE loss.
    - Backpropagation and optimizer step.
  - Use `early stopping`.
  - Save best checkpoint.

- **Phase 2 (finetune)**:
  - Load phase 1 checkpoint.
  - Finetune for 20 epochs with $K=4$ rollout steps.
  - Same data processing, with fixed $\delta t$ sampling.
  - Save checkpoint, monitor validation metrics.

- **Phase 3 (finetune)**:
  - Load phase 2 checkpoint.
  - Finetune for 20 epochs with $K=8$ rollout steps.

- Log training metrics and loss curves.

---

# 7. Inference & Forecast Generation

- Instantiate `InferencePipeline` with trained model.
- For each desired forecast (e.g., each lead time $T$ in evaluation set):
  - Generate multiple interval combinations:
    - Use utility functions from `utils.py`:
      - For *homogeneous* mode: all $\delta t$ same
      - For *heterogeneous* mode:
        - Generate `n = 128` combinations of intervals summing to $T$
        - Select top `m = 3` based on validation loss (if `best m in n`)
  - For each interval sequence:
    - Initialize with initial condition tensor $X_0$.
    - Iteratively perform rollouts:
      - Input current state and $\delta t$ to generate $\Delta_{\delta t}$.
      - Add $\Delta_{\delta t}$ to previous state.
      - Feed updated state for next interval.
    - Collect forecast tensor.
  - Average forecasts over ensemble of interval sequences.

- Save forecasts for each lead time and combination.

---

# 8. Evaluation & Metrics

- Load ground truth data at test times.
- Regrid forecasts using `xarray` or `torch` utilities.
- Compute metrics:
  - Latitude-weighted RMSE for each variable at each lead time.
  - Latitude-weighted ACC and SSR.
  - Use `EvaluationMetrics` class, passing forecast and ground truth.
- Summarize scores, plot error maps, and compare with baselines.

---

# 9. Ablations & Additional Analyses

- Generate ablation results:
  - Remove/alter components:
    - Without randomized $\delta t$
    - Without pressure weighting
    - Absolute vs. dynamics forecasting
  - Adjust model size/patch size.
- Run scaling experiments:
  - Larger models (e.g., ViT-B, ViT-L).
- Perform ensemble vs non-ensemble comparisons.
- Use validation loss for best combination selection.

---

# 10. Save & Log Results

- Save best model checkpoints periodically.
- Save evaluation metrics, figures, and forecasts.
- Log all hyperparameters, training/validation curves.
- Organize outputs for reproducibility: save configs, scripts, checkpoints, and evaluation scripts.

---

# 11. Final Steps & Cleanup

- Gracefully terminate all processes.
- Consolidate logs and results.
- Generate summary reports (tables/figures).
- Wrap up, ensuring reproducibility instructions are documented.

---

# Summary

`main.py` serves as the experiment orchestrator that:
- Loads data and configuration
- Sets up hardware/distributed environment
- Instantiates dataset, model, and trainer
- Executes staged training with multi-phase finetuning
- Executes inference with diverse interval combinations
- Performs evaluation and logs comprehensive results

All steps follow detailed methods and hyperparameters from the paper, with strict reference to the provided configuration and modular code structure.

---

This structured, detailed logic plan ensures a clear pathway to implementation that faithfully reproduces Stormer's methodology and experimental results.

## model.py

{
  "overview": "The core goal of 'model.py' is to implement the TransformerModel class that encapsulates the entire neural network architecture proposed in the paper, including specialized variable embedding, transformer stack with adaptive layer normalization conditioned on the forecast interval delta_t, and output heads for predicting weather dynamics (differences) for multiple variables. The implementation must reflect the detailed architecture, dependencies, and interfaces described in the paper and design documentation, ensuring seamless integration with data preprocessing, training, inference, and evaluation modules.",
  "step-by-step reasoning": [
    "1. Dependencies & Imports:",
    "   - Import PyTorch modules: torch, torch.nn as nn, functional as F.",
    "   - Import 'timm' library modules (e.g., for transformer blocks if used from 'timm', or implement custom transformer blocks if needed).",
    "   - Import 'utils.py' functions: positional encoding, pressure weighting, any custom normalization modules / functions.",
    "   - Basic Python modules as needed: math, typing (Optional, Tuple).",
    "",
    "2. Class Definition:",
    "   - Define class 'TransformerModel' inheriting from 'nn.Module'.",
    "   - Within __init__, initialize:",
    "       a. Weather-specific variable embedding components:",
    "           - Variable tokenization layers: linear layers for each variable (or a shared layer with variable embedding masks).",
    "           - Variable aggregation: a cross-attention module (single-layer multi-head cross-attention) with learnable query vector.",
    "       b. Positional encodings:",
    "           - Use sinusoidal or learned encodings, possibly precomputed or dynamically generated.",
    "       c. Transformer stack:",
    "           - Use 'timm' or custom implementation for Transformer blocks with multi-head attention, feedforward, AdaLN (conditioned on delta_t).",
    "           - Ensure AdaLN can accept scale and shift parameters provided during forward pass.",
    "       d. Output head:",
    "           - Final linear layer(s) to project transformer output tokens to variable difference predictions (\Delta_t).",
    "       e. Conditioning mechanism:",
    "           - MLP that maps delta_t scalar to AdaLN parameters (gamma, beta) for each transformer block.",
    "           - Optionally, scale parameters alpha1, alpha2 for attention and FF layers, if used.",
    "   - Store hyperparameters such as patch size, D (hidden_dim), num_layers, number of heads, etc., consistent with config.yaml.",
    "",
    "3. Forward Method:",
    "   - Inputs:",
    "       a. input: tensor X of shape (batch_size, V, H, W), the initial weather state data.",
    "       b. delta_t: scalar tensor or batch tensor representing the forecast interval in hours.",
    "   - Steps:",
    "       a. Generate delta_t embedding via the MLP conditioned on the scalar delta_t.",
    "       b. Perform variable tokenization:",
    "           - For each variable v: pass V channels through a linear layer to embed into shape (H/p, W/p, D).",
    "           - Stack or concatenate all variable embeddings to shape (H/p, W/p, V, D).",
    "       c. Variable aggregation:",
    "           - Apply cross-attention over variable dimension with learnable query vector, resulting in shape (H/p, W/p, D).",
    "       d. Add positional encodings to tokens.",
    "       e. Flatten spatial tokens: shape (batch_size, num_tokens, D).",
    "       f. For each transformer block:",
    "           - Apply multi-head attention, incorporating AdaLN conditioned on delta_t embedding:",
    "               * Use gamma, beta parameters in AdaLN generated by the conditioning MLP.",
    "               * Pass tokens through attention + feedforward layers, each with AdaLN applied before or after residual connection as per implementation.",
    "       g. Final layer:",
    "           - Pass the output tokens through a linear layer to predict delta variables per token, reshape as needed.",
    "       h. Reshape or map predictions back to variable-dependent grid shape if necessary.",
    "   - Output:",
    "       - A tensor of shape (batch_size, V, H/p, W/p) representing the predicted delta (difference) for each variable at each spatial location.",
    "       - These can be upsampled or mapped onto the original grid if needed.",
    "",
    "4. Auxiliary Components:",
    "   - Implement the 'adaptive layer normalization' (adaLN): custom nn.Module accepting scale and shift from conditioning MLP.",
    "   - Implement the variable tokenization: per-variable linear embedding layers, possibly stored in a dict or ModuleList.",
    "   - Implement the cross-attention module for variable aggregation: one layer of multi-head attention with learnable query vector, possibly from 'utils.py' or from 'timm'.",
    "   - Implement positional encoding functions (e.g., sinusoidal or learned).",
    "",
    "5. Attention and Transformer Block Design:",
    "   - Choose or implement a transformer block that supports parameterized AdaLN and conditioning.",
    "   - If using 'timm' transformer implementations, subclass or adapt to include AdaLN as the normalization step.",
    "   - Otherwise, implement custom Transformer blocks with support for AdaLN.",
    "",
    "6. Consistency & Integration:",
    "   - Ensure the forward pass proper handles batched data and variable inputs.",
    "   - Confirm that the conditioning of AdaLN is correctly vectorized over batch dimension.",
    "   - Confirm that the output predictions are scaled appropriately, matching the variable units (e.g., U, V, T, geopotential).",
    "   - Map the output differences to the actual delta in the full data domain during training, applying inverse normalization if needed.",
    "",
    "7. Additional Considerations:",
    "   - Support for multi-GPU training with mixed precision: ensure model and tensors are compatible.",
    "   - Model should be compatible with multi-phase training: initialization, checkpoint loading, and fine-tuning should be straightforward.",
    "   - The model should expose methods if necessary for extracting intermediate features or for inference timing adjustments.",
    "",
    "8. Implementation Details Based on the Paper & Design:",
    "   - The model’s size: e.g., hidden_dim=1024, num_layers=24, num_heads=16 default per config.",
    "   - Variables: process all pressure levels and surface variables, embedding each pressure level as a separate variable or along with the surface variables as they are at their specific levels.",
    "   - The variable aggregation is critical; use a single-layer cross-attention with learnable queries as per the paper.",
    "   - AdaLN conditioning layer: implement as an nn.Module with parameters generated dynamically from delta_t embedding; ensure it modifies the standard LayerNorm behavior.",
    "   - Use position encodings in token sequence for both spatial and variable dimensions as in Vision Transformer or as specified.",
    "",
    "9. Testing & Validation:",
    "   - Confirm that the forward pass outputs match expected tensor shapes.",
    "   - Verify that conditioning adjusts the normalization parameters appropriately.",
    "   - Ensure that the variable embedding and aggregation effectively encode variable interactions.",
    "   - Test the module on dummy data, plotting the predicted delta fields for sanity checks.",
    ],
    "notes": "Clarify whether additional residual connections or skip pathways are used, whether normalization occurs before or after attention (pre/post-norm), and if specific weight initialization for AdaLN parameters is required. Confirm whether positional encodings are fixed sinusoidal or learned. Keep implementation modular for easier testing and future scaling.",
    "final remark": "Ensure adherence to the design without introducing external or unauthorized architecture modifications, and make the model ready for integration in full training/inference pipeline as specified."
  ]
}

## trainer.py

# Logic Analysis for trainer.py — Stormer Deep Learning Training Module

This document provides a comprehensive, step-by-step logical framework for implementing `trainer.py`, which manages all training-related processes for the Stormer weather forecasting model. The implementation is designed to utilize PyTorch Lightning to facilitate scalable, modular, and reproducible training, including multi-phase training, multi-step fine-tuning, randomized sampling, pressure-weighted loss, checkpointing, and evaluation. The following detailed analysis aligns strictly with the provided paper, plan, design, JSON specifications, and config.yaml.

---

# 1. Core Responsibilities of trainer.py

- **Data Loading and Preparation**
  - Communicate with dataset.py to load preprocessed ERA5 tensors.
  - Ensure normalization (mean/std), regridding, and batch sampling are correctly set up.
- **Model Instantiation**
  - Instantiate `TransformerModel` with architecture parameters.
  - Load existing checkpoints during fine-tuning phases.
- **Training Workflow**
  - Implement three-phase training:
    - Phase 1: single-step training (K=1)
    - Phase 2: fine-tuning with rollout K=4
    - Phase 3: fine-tuning with rollout K=8
  - During each epoch:
    - Sample batch data: initial conditions, true deltas (`Δ_δt`)
    - Sample random `δt` values according to configuration (uniform over {6, 12, 24} hours)
    - Generate model predictions for each batch
    - Compute pressure-weighted MSE loss
    - Conduct backpropagation with optimizer
    - Apply gradient checkpointing (if enabled)
  - Early stopping based on validation loss to prevent overfitting.
  - Save top-performing model checkpoints at intervals or based on validation metrics.
- **Multi-Phase Finetuning**
  - Load previous best checkpoint before subsequent phase.
  - Continue training for more epochs with more rollout steps.
  - Use consistent data splits and hyperparameters.
- **Learning Rate Scheduling**
  - Linear warm-up followed by cosine decay schedule.
  - Implement per-phase schedules as specified.
- **Enabling Mixed-Precision and Parallelism**
  - Use PyTorch Lightning's support for FP16/bfloat16.
  - Enable gradient checkpointing and efficient multi-GPU training.
  - Distribute using DDP across all allocated GPUs.
- **Monitoring and Logging**
  - Track primary metrics: validation loss, RMSE, etc.
  - Log progress, losses, and hyperparameters.
- **Checkpointing & Early Stopping**
  - Save checkpoints at specified intervals.
  - Stop training when no improvement after patience epochs.

---

# 2. Initialization & Setup

- **Hyperparameters & Configs**
  - Extract parameters from config.yaml:
    - Architecture: layers, heads, dims
    - Training: epochs, LR, warmup epochs, batch size
    - Phases: phase durations and K-values
    - Loss weights and variables
    - Checkpoint save intervals
- **Data Loaders**
  - Instantiate dataset.py's dataset object:
    - Provide data path, variables, pressure levels, normalization stats
  - Get train, validation, and test DataLoaders.
- **Model and Optimization**
  - Instantiate `TransformerModel` with architecture params and conditional AdaLN.
  - Setup optimizer (AdamW) with phase-specific LR.
  - Setup LR scheduler: cosine decay after warmup.
- **Lightning Trainer**
  - Wrap model in PyTorch Lightning `Trainer`.
  - Configure precision (fp16), gradient checkpointing.
  - Configure multiple GPUs (DDP).

---

# 3. Training Loop

- **Phase-Dependent Behavior**
  - Determine the current phase based on epoch count:
    1. Phase 1: single-step (K=1), train for 100 epochs.
    2. Phase 2: fine-tune with K=4, for 20 epochs.
    3. Phase 3: fine-tune with K=8, for 20 epochs.
  - Load checkpoint before each phase (except phase 1).

- **Batch Processing**
  - For Each Batch:
    - Retrieve batch data: initial conditions, true future data at full horizon.
    - Sample `δt` for the batch:
      - Random uniform over {6, 12, 24} hours.
    - Compute true deltas (`Δ_δt`) for the batch.
    - Pass initial conditions and `δt` to the model:
      - The model's `forward()` method should condition on `δt` via AdaLN.
      - Generate predicted `Δ̂_δt`.
    - Compute pressure-weighted loss:
      - Multiply squared errors by variable weights, latitude weights, and pressure weights.
      - Sum and normalize over batch.
    - Backpropagate, clip gradients if needed.
    - Update optimizer.
  - Accumulate running loss for metrics and log.

- **Multi-step Rollouts (Finite-Agency)**
  - For K>1 phases:
    - During training, generate K-step rollouts:
      - Iteratively feed model's previous output as new initial condition.
      - Compute loss at each step.
      - Average the loss over K steps.
    - This mitigates error accumulation during rollouts.

- **Validation & Early Stopping**
  - Periodically evaluate validation loss across lead times of interest (e.g., 1d, 3d, 5d).
  - After each epoch, compare validation metrics.
  - Save checkpoint if improvement.
  - Stop when early stopping patience exceeded.

- **Checkpointing**
  - Save at intervals (e.g., every 10 epochs).
  - Save best models based on combined validation loss.

---

# 4. Multi-Phase Fine-tuning

- **Checkpoint Loading**
  - Load checkpoint from previous phase before starting finetuning.
- **Training Settings**
  - Use smaller LR and epochs.
  - Keep the same data splits and normalization.
- **Use consistent `δt` sampling**
  - Same uniform distribution for sampling `δt` during fine-tuning.
- **Batch and K-values**
  - Fix rollout steps (`K=4` or `K=8`) in fine-tuning phases.
- **Loss & Optimization**
  - Continue pressure-weighted loss.
  - Possibly reduce learning rate to ensure stable fine-tuning.

---

# 5. Lightning Module Skeleton

- Implement `LightningModule` class:
  - Define `configure_optimizers()`: optimizer + scheduler.
  - Define `training_step()`: process batch, compute loss.
  - Define `validation_step()`: evaluate validation metrics.
  - Implement `on_epoch_end()`: log metrics, save checkpoints.
  - Support loading from checkpoint.
  - Handle phase adjustments: `K`, LR, epochs.
  - Incorporate logging via Lightning logger.

---

# 6. Loss Function Details

- **Pressure-Weighted MSE:**
  - Read pressure levels per variable.
  - Assign weights based on pressure (provided in config).
  - For each sample:
    - Compute per-variable squared error.
    - Weight error by pressure weight (distance-based).
    - Sum over variables, space; divide total by total weight.

- **Multi-step Loss**
  - For `K > 1`:
    - Perform iterative rollouts within `training_step()`.
    - Sum and average the losses at each rollout step.
  
- **Loss Aggregation**
  - Log total loss, per-variable loss, and validation loss.

---

# 7. Hyperparameter & Schedule Management

- Use config.yaml parameters for:
  - Learning rates
  - Warmup epochs
  - Epochs per phase
  - Loss weights
  - Batch size
- Implement dynamic adjustment if needed for different phases.
- Utilize torch.optim.lr_scheduler.CosineAnnealingLR with warmup scheduler.

---

# 8. Checkpointing & Early Stopping

- Use PyTorch Lightning's built-in callbacks:
  - `ModelCheckpoint` to save best models.
  - `EarlyStopping` with patience=15 epochs, monitor validation loss.
- Save models at configured intervals.
- Log metrics and hyperparameters to facilitate reproducibility.

---

# 9. Additional Considerations

- **Gradient Checkpointing**
  - Enable in model initialization for large models.
- **Mixed Precision**
  - Use Lightning's `precision=16` or `bf16`.
- **Distributed Training**
  - Use DDP across all devices.
- **Logging**
  - Use standard logging frameworks (TensorBoard, WandB).
  - Record epoch losses, validation metrics.

---

# 10. Summary

This logic analysis ensures that `trainer.py` will:

- Initiate with the correct data and model configurations.
- Follow the three-phase training schedule with respective `K` values.
- Incorporate randomized `δt` sampling and pressure-weighted loss.
- Support multi-step fine-tuning.
- Efficiently leverage multi-GPU training with gradient checkpointing and mixed precision.
- Save checkpoints regularly.
- Stop early if validation degrades.
- Fine-tune and resume seamlessly.
- Execute inference with ensemble combination appropriately, though inference code resides primarily in `inference.py`.

All these steps are aligned with the paper’s methods, design, and experimental protocols, ensuring faithful reproduction and operational scalability.

---

This concludes the thorough logic analysis for implementing `trainer.py` within the Stormer weather forecasting framework according to the detailed specifications and constraints provided.

## utils.py

# utils.py - Logic Analysis

This file provides core utility functions to support data processing, model architecture, training, inference, and evaluation as outlined by the Stormer methodology. Each function must be designed for clarity, flexibility, and adherence to the specifications from the paper and the configuration parameters.

---

## 1. Pressure Weighting Scheme

### Purpose:
Implement a flexible pressure weighting function that assigns weights to variables based on their pressure levels, prioritizing near-surface variables. The weights should be computed per pressure level and applied during loss calculation to emphasize variables with important physical significance (e.g., near-surface temperature and pressure).

### Key points:
- Use pressure levels provided in configuration (`pressure_levels` array).
- Assign weight of 1.0 for surface variables (e.g., T2m).
- Assign smaller weights (e.g., 0.1) for atmospheric variables at pressure levels, possibly distinguishing between surface and upper levels.
- Support potential customization for different pressure levels.

### Implementation plan:
- Create a function, e.g., `get_pressure_weights(variables: list, pressure_levels: list) -> Dict[str, float]`.
- Input:
    - `variables` (list of variable names involved in loss)
    - `pressure_levels` (list of pressure levels)
- Output:
    - Dictionary mapping variable name to weight.
- Logic:
    - For surface variables (e.g., T2m, MSLP, U10, V10), assign weight 1.0.
    - For atmospheric variables (Z, T, U, V, Q) at pressure levels:
        - Map their respective pressure levels.
        - Assign weight 0.1 to pressure levels not closest to surface.
        - Optionally, give higher weights to variables at lower pressures if justified.
    - Return combined dict for use in loss calculation.
    
### Additional considerations:
- Keep the function flexible for future modifications.
- Make sure the output mapping aligns with variable names in the dataset.

---

## 2. Positional Encoding

### Purpose:
Generate positional encodings to add spatial information to the patch tokens, either sinusoidal or learnable. These are necessary for transformer models to understand the spatial structure of the weather grid.

### Key points:
- The input shape: (H/p, W/p, D) after tokenization.
- Utilize either sinusoidal or learned embeddings.
- Support maximum position embedding size (`max_position_embeddings`) from config.

### Implementation plan:
- Create function `get_2d_positional_encoding(H_p, W_p, D, method='sinusoid', max_embeddings=1024) -> Tensor`.
- Logic:
    - If method='sinusoid':
        - Generate sinusoidal positional embeddings for H and W axes, sum or concatenate as needed.
    - If method='learned':
        - Use learnable parameters (nn.Embedding) for each position index.
- Output:
    - Tensor of shape (H_p, W_p, D)
- Usage:
    - Add positional encoding to patch tokens before input to transformer.

### Additional considerations:
- Store the encoding tensor if static, or generate dynamically during batch processing.
- Ensure compatibility with batching.

---

## 3. Normalization Routines

### Purpose:
Implement normalization functions to standardize input features and output differences, based on dataset-wide statistics. Also, handle normalization during training and inference.

### Functions:
- `normalize_input(data: Tensor, mean: Tensor, std: Tensor) -> Tensor`
- `denormalize_output(data: Tensor, mean: Tensor, std: Tensor) -> Tensor`

### Logic:
- For inputs:
    - Subtract the mean computed over training set.
    - Divide by std computed over training set.
- For outputs (model predicts delta variables), similarly normalize using respective delta statistics.
- During inference, denormalize predictions for evaluation.

### Additional:
- Store normalization means/stats in a structured way, perhaps in a dictionary.
- Support variable-specific normalization.

---

## 4. Generate Interval Combinations for Inference

### Purpose:
Create functions to generate different sequences of $\delta t$ intervals that sum to a target lead time `T`. This supports the heterogeneous combination inference method.

### Implementation:
- `generate_combinations(T: int, intervals: List[int], mode='heterogeneous') -> List[List[int]]`

### Logic:
- For homogeneous:
    - Return [ [T] ] only.
- For heterogeneous:
    - Use recursive or dynamic programming approach to generate all compositions of T with elements from `intervals`.
    - Optionally, limit to top `n` combinations based on some criterion or random sampling.
- Use these sequences during inference to perform rolling forecast steps.

### Additional:
- Sequence of intervals: e.g., for T=24, possible sequences:
    - [6,6,6,6], [12,12], [24]
    - or heterogeneous: [6,12,6], etc.

---

## 5. Ensemble Averaging Functions

### Purpose:
Combine multiple forecast predictions from different interval configurations into a single ensemble forecast.

### Functions:
- `ensemble_average(forecasts: List[Tensor]) -> Tensor`

### Logic:
- Element-wise mean across list tensors.
- Support arbitrary weights if needed (for weighted ensembling).
- Ensure all tensors are aligned spatially and variable-wise.

---

## 6. Additional Utility Functions (Optional but Recommended)

- **Clipping or masking masks** for physical constraints.
- **Error metrics** functions that accept predictions and ground truth:
    - `compute_rmse(pred: Tensor, target: Tensor, weights=None) -> float`
    - `compute_acc(pred: Tensor, target: Tensor, mask=None) -> float`
    - `compute_ssr(pred: Tensor, target: Tensor) -> float`
- **Logging helpers** to record loss, metrics, and hyperparameters during training and evaluation.

---

## Summary of Functions to Implement:
- `get_pressure_weights(variables, pressure_levels) -> dict`
- `get_2d_positional_encoding(H_p, W_p, D, method='sinusoid') -> Tensor`
- `normalize_input(data, mean, std) -> Tensor`
- `denormalize_output(data, mean, std) -> Tensor`
- `generate_combinations(T, intervals, mode='heterogeneous') -> List[List[int]]`
- `ensemble_average(forecasts: List[Tensor]) -> Tensor`

---

## Implementation Notes:
- Use PyTorch tensors for all calculations to enable GPU acceleration.
- All functions should be designed for batch processing.
- Keep functions stateless unless caching or learnable parameters are required (e.g., positional encoding).
- Ensure flexibility for future extensions, like variable-specific or time-dependent weights.

---

This detailed logic analysis will guide precise, well-structured, and modular implementation of `utils.py`, crucial for reconstructing the Stormer weather forecasting pipeline faithfully.

