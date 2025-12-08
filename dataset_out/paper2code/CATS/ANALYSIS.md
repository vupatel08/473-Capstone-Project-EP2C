# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

**Logic Analysis for `dataset_loader.py`**

---

### **Purpose and Responsibilities**

- Implement the `DatasetLoader` class that handles:
  - Loading datasets from specified paths.
  - Applying normalization (e.g., standardization).
  - Segmenting time series data into patches for model input.
  - Creating train, validation, and test splits.
  - Providing interfaces to supply data batches for training and evaluation.

---

### **Inputs and Parameters**

- **Dataset selection:**
  - Use the dataset name (e.g., "ETTm1", "Weather", "M4-Quarterly") specified in the configuration (`dataset.name`) and data path (`dataset.data_path`).
  - Dataset files formatted according to dataset specifics (likely CSV or similar).

- **Hyperparameters (from `config.yaml`):**
  - `input_sequence_length` (`L`): The length of the sequence used as input to the model.
  - `patch_size`: Patch length for segmentation; e.g., 24.
  - Data split ratios: train/val/test fractions (default 70/15/15).

- **Synthetic Data Options:**
  - For synthetic datasets (like in Section 5.5), incorporate logic to generate signals with periodic shocks, if relevant.
  - Else, load real data.

---

### **Design and Data Handling**

- **Data storage:**
  - Raw data stored as pandas DataFrame or numpy array.
  - Parsed into numpy array or tensors for further processing.

- **Normalization:**
  - Implement standardization (subtract mean, divide by std) across training data only.
  - Store normalization parameters (mean/std) for applying to validation/test sets.
  - Possibly support other normalizers (e.g., MinMaxScaler), chosen via `normalizer` parameter.

- **Segmentation into patches:**
  - Divide sequence into overlapping or non-overlapping patches:
    - For each sequence, create patches of size `patch_size`.
    - Use strides (e.g., equal to patch size for non-overlapping).
  - Store patches as tensors of shape `(number_of_patches, patch_size, features)` or `(number_of_patches, patch_size)` (if univariate).

- **Dataset splits:**
  - Use the specified ratios to split data indices into training, validation, and testing sets.
  - For synthetic signals, define fixed train/test splits aligned with sequence generation.
  - For real data: split based on time indices, preserve sequence order.

- **Batch preparation:**
  - Generate batches of patches and corresponding target sequences.
  - Implement data loaders that yield batches for training/evaluation.
  - Support batching multiple sequences if multivariate datasets.
  
---

### **Output and Interface**

- Provide methods such as:
  - `load_data()`: reads datasets, applies normalization, performs segmentation.
  - `get_train_test_split()`: outputs datasets suitable for DataLoader or explicit batch generation.
  - Possibly, methods for creating torch datasets/dataloaders for easy iteration.

- Return data in torch-compatible format (`Tensor`), ready for input into model:
  - Separate inputs (`X`) and targets (`Y`), shaped according to the model's expectations.
  - For patch-based approach: inputs as `(batch_size, number_of_patches, patch_size, features)`.
  - Targets as sequence of future points (`T` steps ahead), aligned correspondingly.

---

### **Implementation Details & Considerations**

- **Ensuring reproducibility:**
  - Use fixed seeds at data splitting stage if applicable.
  - Maintain consistent ordering.

- **Handling multivariate data:**
  - Each feature handled independently or jointly based on normalization and model design.
  - For the implementation described, assume separate univariate series or multivariate with features.

- **Synthetic Data Generation:**
  - For synthetic experiments (e.g., with periodic shocks), include a function to generate signals:
    - Sample from normal distribution.
    - Add periodic shocks at fixed intervals.
    - Use parameters (\(\tau=24, S=8, k=5\)).

- **Data Format Compatibility:**
  - Support CSV, HDF5, or numpy formats.
  - Include error handling and file existence checks.

- **Preprocessing pipeline:**
  - Loading → normalization → segmentation → splitting → batching.

---

### **API and Usage**

- **Main class constructor:**
  ```python
  class DatasetLoader:
      def __init__(self, dataset_names: List[str], config: dict):
          ...
  ```
- **Methods:**
  - `load_data()`: loads all datasets, initializes internal variables.
  - `get_train_test_split()`: returns train/test datasets as tensors.
  - Optionally, `get_dataloaders()`: returns PyTorch dataloaders for batch iteration.
  - Internal methods for normalization, patch segmentation, synthetic data creation.

---

### **Summary of Core Logic**
1. **Dataset selection & loading:**
   - Read dataset file(s) based on the provided name and path.
2. **Data normalization:**
   - Fit on training data; apply same normalization on val/test.
3. **Patch segmentation:**
   - Segment time series data into patches of specified size.
4. **Splitting:**
   - Split sequences into train/validation/test based on indices and ratios.
5. **Synthetic data generation (optional):**
   - Generate signals with periodic shocks for testing model interpretability.
6. **Output:**
   - Return datasets prepared as tensors, ready for data loader or direct batching.

---

### **Notes and Clarifications Needed**
- Exact dataset formats and structure.
- Specific sample rate and sequence lengths for synthetic data.
- Implementation of different normalization methods.
- Handling of multivariate vs univariate datasets: assume univariate unless otherwise specified.
- Whether synthetic data setup is part of core or optional.

---

This structured logic provides a comprehensive guide for implementing `DatasetLoader`, ensuring data is loaded, processed, and split appropriately, facilitating seamless integration with training and evaluation pipelines.

## evaluation.py

### Logic Analysis for `evaluation.py`

**Purpose:**
- Implement the `Evaluation` class responsible for:
  - Loading the trained CATS model and necessary components.
  - Running inference on the test dataset.
  - Computing evaluation metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE).
  - Visualizing attention maps (cross-attention heatmaps) and forecasted vs true signals.
  - Handling evaluation across multiple forecasting horizons with consistent methodology.

---

### Core Components & Responsibilities

#### 1. Initialization (`__init__`)
- **Inputs:**
  - Trained model checkpoint path (`model_path`), dataset information, configuration (hyperparameters), experiment parameters.
- **Actions:**
  - Load the saved model state (via `torch.load`) ensuring architecture matches the saved state.
  - Load or prepare the test dataset:
    - Use `dataset_loader.py` or dataset interface to load the test split.
    - Apply identical preprocessing (normalization, patching) as during training.
  - Instantiate the model:
    - Initialize `CATSModel` with model hyperparameters from the config.
    - Load weights into the model.
  - Initialize device (GPU/CPU) and move model to device.
  - Prepare evaluation metrics accumulators.
  - Prepare visualization utilities for attention heatmaps.

---

#### 2. Data Preparation and Batching
- **Dataset Input:**
  - The test dataset should be provided as a DataLoader or batched tensors.
  - Ensure test data is in the same format as training data:
    - Segmented into patches.
    - Normalized identically to training.
  - For synthetic data (if used), recreate the signals with known parameters:
    - For example, periodic signals with shocks, so that attention patterns and forecast accuracy can be interpreted.
- **Batching:**
  - Batch size should match the experimental setup or as specified (e.g., 32).
  - For multi-horizon forecasting, prepare input tensors for each horizon.
- **Attention Map Extraction:**
  - The model should store or allow extraction of attention weights during inference.
  - If not, modify the model architecture to return attention weights alongside predictions.

---

#### 3. Inference (`predict`)
- **Input:**
  - Batched input sequences `X_test` (patch-embedded).
- **Process:**
  - Run the model in evaluation mode (`model.eval()`).
  - For each batch:
    - Input the sequence into the model.
    - The model should output:
      - Forecasted sequence for each horizon
      - Cross-attention weights (or heatmaps)
    - Store predictions and attention maps.
- **Output:**
  - Predicted sequences (`Y_pred`)
  - True sequences (`Y_true`) for evaluation
  - Attention maps (if accessible), e.g., `attention_weights` per layer/head/horizon

---

#### 4. Metrics Calculation
- **Metrics:**
  - Calculate the per-horizon MSE and MAE over the entire test set.
  - Use batch-wise accumulation:
    - For each batch:
      - Compute squared errors (for MSE)
      - Absolute errors (for MAE)
    - Aggregate over all batches for final metrics.
- **Implementation:**
  - Use `scipy` or `numpy` to compute metrics.
  - For per-horizon analysis, store errors for each horizon separately.
  - Report overall average and per-horizon scores.

---

#### 5. Visualization (`visualize_attention_maps`)
- **Heatmaps:**
  - Generate attention score maps:
    - Typically 2D matrices representing attention weights between input patches and output patches.
  - Overlay attention scores on temporal positions to illustrate periodic or shock-related patterns:
    - Use matplotlib heatmaps (`imshow`)
    - Space axes to match input and output steps (e.g., input patches along x-axis, forecast horizon along y-axis)
  - Focus on interpretability:
    - Highlight patches with highest attention scores.
    - Show cross-attention maps as per Figures 10-15.
- **Forecast Comparisons:**
  - Plot the true future sequence and forecasted sequence.
  - Use line plots for temporal signals.
  - Add titles, axis labels, and legends for clarity.

---

#### 6. Additional Functionalities
- **Multi-horizon evaluation:**
  - Loop over multiple forecast horizons.
  - For each horizon:
    - Extract the corresponding predictions.
    - Compute individual MSE, MAE.
    - Visualize individual attention maps, if needed.
- **Result Logging & Export:**
  - Save metrics to a dictionary or structured report.
  - Save attention heatmaps as images or overlays.
  - Save forecast plots for qualitative analysis.
- **Reproducibility:**
  - Use fixed random seeds (if any stochasticity in evaluation).
  - Consistent data normalization and patching applied.

---

### Implementation Details & Caveats

- **Access to Attention Weights:**
  - The core model (`CATSModel`) must be designed or modified to output or store attention weights during inference.
  - If not, must be re-implemented with hooks (`register_hook`) to extract attention scores.
  
- **Synthetic Data & Synthetic Signal Experiments:**
  - For synthetic signals described (with shocks, periodicity), generate test signals within evaluation code, to verify interpretability and attention correctness.
  - Plot synthetic signals and overlay attention scores to interpret specific attention-elicited periodicity or shock detection.

- **Attention Map Visualization:**
  - Use the same axes scale as in the paper figures for comparability.
  - Normalize attention weights per head/layer if needed, for clearer visualization.
  
- **Attention Pattern Analysis:**
  - For synthetic signals such as Fig. 6-9, verify that attention maps reflect shock locations or periodicities as expected.
  - For real datasets, focus on observing periodicity and important input patches influencing predictions.

---

### Summary

By following this plan, the `evaluation.py` script will ensure:
- Proper loading and consistent preprocessing of test datasets.
- Correct inference using the trained cross-attention-only model.
- Accurate metric calculation for each forecast horizon.
- Clear visualization of cross-attention heatmaps and forecast results.
- Proper handling of synthetic and real datasets.
- Outputting structured, interpretable results aligning with the paper's figures and analyses.

This comprehensive approach guarantees that the experiment evaluation faithfully replicates the core methodology, analysis, and interpretability insights from the paper.

## main.py

# Logic Analysis for `main.py`

The primary purpose of `main.py` is to serve as the entry point for the entire experimentation pipeline, managing configuration parsing, data loading, model initialization, training, evaluation, and logging. Based on the provided paper, design, and configuration, the core logical flow can be broken down into distinct, well-ordered steps.

---

## 1. Parse Configuration

- **Input:** `config.yaml`
- **Objective:** Read and parse all experiment parameters, including:
  - Training hyperparameters: `learning_rate`, `batch_size`, `epochs`, `dropout_rate`, `mask_probability`, `patience`, `optimizer`, `weight_decay`.
  - Model architecture: `input_sequence_length`, `forecast_horizon`, `patch_size`, `num_layers`, `num_heads`, `embed_dim`, etc.
  - Dataset specifics: `name`, `data_path`, normalization type, data splits.
  - Miscellaneous: seed, masking strategy, number of patches.
  - Hardware setup: `gpus`.

- **Implementation details:**
  - Use `PyYAML` to load `config.yaml`.
  - Convert to a dictionary object (`config`) accessible for subsequent steps.

---

## 2. Set Random Seed for Reproducibility

- **Input:** `config['misc']['seed']` (e.g., 42)
- **Objective:** Fix the random seed for:
  - `torch.manual_seed`
  - `numpy.random.seed`
  - `random.seed` (if necessary)
  - Additional to ensure deterministic behavior (`torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`).
- **Justification:** Ensures reproducibility across runs, as emphasized in the paper.

---

## 3. Hardware Setup

- **Input:** `config['hardware']['gpus']`
- **Objective:**
  - Detect available GPU devices.
  - Initialize device(s) accordingly.
  - Use `torch.device('cuda')` or `torch.device('cpu')` if no GPU.
  - If multiple GPUs: initialize `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel` (if distributed setup intended).
- **Implementation:**
  - Use `torch.cuda.is_available()` and `torch.cuda.device_count()`.
  - Set `device` accordingly.
  - Log the number of GPUs for user confirmation.

---

## 4. Data Loading & Preprocessing

- **Input:** `config['dataset']` entries
- **Steps:**
  - Instantiate `DatasetLoader`.
  - Load dataset from `data_path`.
  - Apply normalization (`standard`, as in config).
  - Segment time series into patches of size `patch_size`:
    - For real datasets, use fixed or dynamic windowing.
    - For synthetic or special datasets, generate synthetic signals with described properties (shocks, periodicity).
  - Apply train/validation/test splits based on fractions (`train_split`, `val_split`, `test_split`).
  - Convert data to `torch.Tensor` objects.
- **Output:** Dictionary with `train_dataset`, `val_dataset`, `test_dataset`.

---

## 5. Model Initialization

- **Input:** Loaded dataset info, `config['model']`
- **Steps:**
  - Instantiate the `CATSModel` class.
  - Set hyperparameters:
    - `input_sequence_length`
    - `forecast_horizon`
    - `patch_size`
    - `num_layers`, `num_heads`, `embed_dim`, etc.
    - Use `parameter_sharing=True`.
    - Initialize learnable horizon-dependent query embeddings.
  - Incorporate masking parameters (`mask_probability`), if applicable.
  - If GPU available, move model to device.
  - Log model architecture and parameter count for transparency.
  
---

## 6. Optimizer & Learning Rate Scheduler Setup

- **Input:** Hyperparameters from config
- **Steps:**
  - Choose optimizer: e.g., `torch.optim.Adam` or tied to config.
  - Set optimizer parameters:
    - Learning rate as per `config['training']['learning_rate']`.
    - Weight decay.
  - Decide on learning rate scheduler:
    - Optional: cyclic, step decay, or constant.
- **Output:** Optimizer object, scheduler object (if used).

---

## 7. Define Loss Function

- **Based on evaluation metrics in the paper:**
  - Use `torch.nn.MSELoss()` for training.
  - Optionally, MAE could also be logged.
- **Implementation:** Instantiate once for use during training.

---

## 8. Training Loop

- **Conditions:**
  - Loop from epoch 1 to `epochs`.
  - Use early stopping with `patience`.
- **Per Epoch:**
  - **Training phase:**
    - Shuffle training data (if dataset is large).
    - For each batch:
      - Load batch data: `input_sequence`, `target_sequence`.
      - Apply query-adaptive masking:
        - Generate or decide mask based on `mask_probability`.
        - Mask the queries or their embeddings accordingly.
      - Forward pass:
        - Invoke model with input patches, horizon queries, and mask.
        - Collect predictions.
      - Compute loss (e.g., MSELoss between predictions and targets).
      - Backpropagation:
        - Zero gradients.
        - `loss.backward()`.
        - Optimizer step.
    - Record average loss for the epoch.
  - **Validation phase:**
    - Run inference on validation set with model in eval mode.
    - Compute validation metrics (MSE, MAE).
    - Store metrics for early stopping.
- **Checkpointing:**
  - Save model weights whenever validation improves.
  - Use patience to determine early stopping.

---

## 9. Evaluation & Testing

- **Post-training:**
  - Load best checkpoint model.
  - Run inference on the test set.
  - Generate forecasting results for all horizons.
  - Collect errors (MSE, MAE), attention maps, and attention scores.
- **Visualization:**
  - Use `matplotlib` integrated with `utils.py` to:
    - Plot forecast vs ground truth.
    - Visualize attention score maps (especially for interpretability, as shown in Sections 5.5 and Figures 6–15).
- **Metrics:**
  - Report comprehensive metrics matching what is presented in Tables 4–6, 11–12, etc.

---

## 10. Logging & Results Summary

- **Implementation:**
  - Record metrics per epoch and overall.
  - Log hyperparameters, training duration, resource consumption.
  - Save visualizations and attention maps for interpretability analysis.
- **Output:**
  - Console logs.
  - Saved plots/images and model checkpoint files.
  - Summary report or log file with final metrics.

---

## 11. Optional: Hyperparameter Search or Ablation Studies

- Repeat training with varied `mask_probability`, layer counts, or patch sizes.
- Record and compare performance and interpretability.
- Conduct ablation studies similar to the ones in the paper (e.g., removing masking, replacing cross-attention with self-attention).

---

## Summary of Dependencies & Sequence:

- Parse config → Set seeds → Set device → Load data → Initialize model → Setup optimizer/scheduler → Loop over epochs:
  - Train batches → Apply masking → Forward → Loss → Backward → Save best models.
- After training, run inference on test set → Generate metrics → Visualize attention patterns → Save final results.

This detailed logical flow ensures that every component, as described in the paper and plan, is properly incorporated, maintaining the original methodology, hyperparameters, and evaluation protocols for faithful experiment reproduction and interpretability.

## model.py

{
  "model.py": "This module implements the core components of the Cross-Attention-Only Time Series Transformer (CATS). It includes classes for patch embedding, horizon-dependent query embeddings, the stacked cross-attention blocks, and the overall model architecture. The design emphasizes parameter sharing across horizons and incorporates masking mechanisms to enhance robustness.\n\n**1. Patch Embedding Layer (`PatchEmbedder`)**:\n - Accepts raw input sequences (either time series or patches of size P). \n - Uses a linear layer or convolution to project patches into a dense embedding space of dimension `embed_dim`.\n - Supports batch input, returning embedded patches of shape `(batch_size, num_patches, embed_dim)`.\n\n**2. Horizon-Dependent Query Embeddings (`HorizonQueries`)**:\n - For each forecasting horizon (e.g., `T = 72`), define a learnable query vector or set of vectors per patch.\n - If `horizon_embeddings` is true, initialize trainable parameters of shape `(forecast_horizon, num_queries_per_patch, embed_dim)`.\n - These are repeated or broadcasted to match batch size during each forward pass.\n - These queries serve as the target for the cross-attention layers, representing the specific future time steps.\n\n**3. Cross-Attention Block (`CrossAttentionLayer`)**:\n - Implements multi-head cross-attention: queries from horizon embeddings, keys and values from embedded input patches.\n - Incorporates parameter sharing: the same attention weights are used across all horizons and layers, which reduces memory and promotes consistent attention patterns.\n - During forward pass, accepts `query_embeddings`, `key_value_embeddings`, and optionally a `mask` tensor for query-specific masking.\n - Outputs attended features, possibly augmented with residual connections and layer normalization.\n - Supports multi-head attention with standard scaled dot-product attention.\n\n**4. Masking Mechanism (`MaskingStrategy`)**:\n - During training, applies probabilistic masks to the attention scores or outputs to prevent overfitting and encourage query-specific focus.\n - Masking probability (`mask_probability`) can be horizon-dependent, possibly increasing linearly over the forecast horizon.\n - Masks are applied by setting certain attention scores or outputs to zero, simulating stochastic depth or dropout.\n\n**5. Complete Model Architecture (`CATSModel`)**:\n - Consists of:\n   - Patch embedding layer for input sequences.\n   - A set of `num_layers` cross-attention modules, each with shared parameters.\n   - Horizon-specific query embeddings, broadcasted per horizon.\n   - Optional query-adaptive masking during training.\n   - Final linear projection layer to produce forecasted output of dimension matching the dataset (e.g., `M` variables per time step).\n - During inference, forecasts are generated horizon-by-horizon, each with its learnable query.\n - Model supports parameter sharing across all layers and horizons, configurable via flags.\n\n**6. Initialization and Configurations**:\n - All learnable parameters (query embeddings, patch embeddings, attention weights) are initialized according to best practices (e.g., Xavier initialization).\n - The number of layers, heads, embed_dim, and other hyperparameters are driven by the config file.\n - Supports flexible batch sizes and variable sequence lengths compatible with input data.\n\n**7. Functional Flow in `forward()` Method**:\n - Receive raw input sequences.\n - Segment into patches based on the patch size.\n - Embed input patches via `PatchEmbedder`.\n - Generate or retrieve horizon-dependent query embeddings.\n - For each forecast horizon:\n    - Use horizon query as the multi-head query input.\n    - Execute cross-attention with embedded patches as keys and values.\n    - Apply masking if enabled.\n    - Aggregate the attended features, possibly passing through feed-forward networks.\n    - Project to output dimension.\n - Return hierarchical outputs: one per horizon, or combined as needed.\n\n**8. Additional Implementation Details**:\n - Maintain the modularity for easy stacking of attention layers.\n - Ensure parameter sharing across layers and horizons:\n    - Attention weights are shared, only queries and possibly the layer normalization parameters vary per layer/horizon.\n - In training, implement flexible masking policies, including fixed probability and query-dependent probabilistic masking.\n - Provide functions for visualization of attention maps for interpretability.\n\n**Summary**: \nThis `model.py` file encapsulates the core architecture: embedding raw sequences into patches, creating horizon-specific queries, stacking shared cross-attention layers, applying probabilistic masking, and producing multi-horizon forecasts. Design choices are aligned with experimental configurations for efficient, interpretable, and flexible forecasting, emphasizing parameter sharing to optimize resource use and facilitate analysis of attention patterns across horizons."
}

## trainer.py

# Logic Analysis for trainer.py in the CATS Time Series Forecasting Model

This document provides a comprehensive, step-by-step logical framework for implementing the `Trainer` class within `trainer.py`. The purpose of this class is to orchestrate the entire training process for the Cross-Attention-Only Time Series Transformer (CATS) model, ensuring it follows the experimental setup, loss computation, masking strategies, and evaluation protocols specified in the paper, plan, and configuration.

---

## 1. Overall Responsibilities of the Trainer Class
- Initialize with model, datasets, hyperparameters.
- Manage the training loop: loading batches, forward passes, loss computation, backpropagation, optimizer steps.
- Implement the query-adaptive masking strategy during training.
- Schedule learning rates and early stopping based on validation performance.
- Checkpoint saving: save the best model weights and final weights.
- Log training progress: losses, metrics, time per epoch.
- Evaluate the model periodically (e.g., at epoch end) and/or after training completes.
- Handle multiple datasets or hyperparameter configurations as needed.

---

## 2. Inputs & Initialization
- **Arguments:**
  - `model`: instance of `CATSModel`; contains the core transformer with horizon-dependent queries and cross-attention modules.
  - `dataset`: the dataset object or dict holding training and validation data, possibly test data.
  - `config`: dictionary containing hyperparameters, including:
    - learning rate, batch size, number of epochs, dropout rate, masking probability, patience.
    - dataset split info, input sequence length, forecast horizon.
    - optimizer type, early stopping criteria.
  - **Additional parameters:**
    - `device`: CUDA or CPU based on availability.
    - `logger`: optional, for logging training metrics.
    - `checkpoint_path`: path to save checkpoints.
- **Setup:**
  - Initialize the optimizer (Adam or specified in config) with model parameters, including weight decay.
  - Set up learning rate scheduler if used.
  - Prepare data loaders for training and validation sets, with batch size from config.
  - Initialize early stopping variables: best validation loss, patience count.
  - Set random seed for reproducibility.

---

## 3. Data Feeding & Batch Processing
- For each epoch:
  - Iterate over `train_loader`:
    - Load a batch of sequences: input sequences (`X`) and target sequences (`Y`).
    - Inputs are tensors of shape `[batch_size, sequence_length, features]`.
    - Targets have shape `[batch_size, forecast_horizon, features]`.
- **Preprocessing:**
  - Extract patches from input sequences according to patch size, or feed entire sequences if patching is done inside model.
  - Prepare horizon-dependent queries for each batch:
    - These are trainable embeddings (if horizon embeddings are used) or fixed queries for each forecast horizon.
    - These should be generated once per batch (or precomputed), matching the batch size and horizon count.
- **Masking:**
  - Apply query-adaptive masking strategy with probability `p` during training:
    - Mask the output queries' attention features with some probability, as per the method described.
    - This can be implemented via masking matrices applied to attention weights or the queries themselves before passing into the attention modules.
  - Masking is stochastic: sample from Bernoulli(p) for each sample/horizon.
  - Optionally, incorporate dropout regularization with rate specified.

---

## 4. Model Forward Pass
- Pass the batch input sequences into the `model`:
  - Inputs: embedded patches, position embeddings.
  - Queries: horizon-dependent learnable query vectors, embedded or directly used.
  - Pass both input patches and queries into the cross-attention modules:
    - The model's forward method should:
      - Directly pass input patches as the memory/key/value.
      - Use horizon-specific queries as the query input.
      - Return predictions (`\hat{Y}`), attention scores, and optionally, intermediate layer outputs for interpretability.
  - When applying masking:
    - Ensure the masking operation affects the attention outputs (or the queries) as per the probabilistic masking strategy.
- Obtain model output predictions for each horizon.

---

## 5. Loss Computation
- Compare the predicted sequences (`\hat{Y}`) with ground truth (`Y`):
  - Use the specified loss function:
    - MSE (mean squared error) for primary evaluation, or MAE based on experiment.
  - Compute the loss per batch:
    - Loss shape: `[batch_size]`.
    - Aggregate with mean over batch.

- **Additional considerations:**
  - The loss should be scaled appropriately if using multiple horizons.
  - Optionally, compute auxiliary losses or regularization terms (not specified but common).

---

## 6. Backpropagation & Optimization
- Zero optimizer gradients.
- Backward pass: `loss.backward()`.
- Apply gradient clipping if needed (to stabilize training).
- Update optimizer: `optimizer.step()`.
- Step the learning rate scheduler if used: `scheduler.step()`.

---

## 7. Early Stopping & Checkpointing
- Record current validation loss after each epoch:
  - Run validation over validation data loader:
    - Same as training but without gradient updates.
    - Compute average validation loss over dataset.
- If current validation loss improves upon the best so far:
  - Save model checkpoint to `checkpoint_path`.
  - Reset patience counter.
- Else, increment patience counter:
  - If patience exceeds threshold (e.g., `config['training']['patience']`), stop training early.

---

## 8. Logging & Monitoring
- Log training loss, validation loss, and relevant metrics per epoch.
- Record attention maps or interpretability outputs if needed.
- Track resource usage (time, GPU memory) if desired.
- Visualize learning curves as needed.

---

## 9. Post-Training & Evaluation
- Load the best saved model weights.
- Run inference on test dataset:
  - Generate forecasted sequences using the `model.predict()` method.
  - Ensure the same masking strategy is applied during inference if necessary.
- Compute and report metrics (MSE, MAE, etc.).
- Visualize attention maps, forecasting results, and interpretability plots as per the experiment.

---

## 10. Implementation Details & Best Practices
- **Reproducibility:**
  - Use fixed random seed everywhere.
- **Device Management:**
  - Move model and data tensors to the specified device.
- **Batch Size & Memory:**
  - Adjust batch size as per resource constraints; incorporate gradient accumulation if necessary.
- **Modularity:**
  - Separate training, validation, and testing functions within the class.
- **Hyperparameter Variability:**
  - Allow configuration injection so hyperparameters (e.g., masking probability, dropout, learning rates) can be varied systematically.
- **Attention Map Saving:**
  - Optionally save attention weights after forward pass for interpretability.

---

## 11. Handling Synthetic or Specialized Data
- For synthetic datasets (e.g., with shocks and periodic signals):
  - Generate data within the dataset loader or externally and load as regular datasets.
  - Ensure input sequence length \( L \), patch size, and masking strategies align with the experimental design described.
  - Use the same synthetic data parameters (\( \tau, S, k \)) for data generation during reproducibility.

---

## Summary of Key Variables & Methods:
| Variable / Function | Purpose |
|------------------------|---------|
| `self.optimizer`     | Adam optimizer for model updates |
| `self.scheduler`     | Learning rate scheduler (if used) |
| `_train_epoch()`     | Loop over training batches, apply masking, run forward, backpropagate |
| `_validate()`        | Run on validation data, compute loss, early stopping criteria |
| `apply_mask()`       | Utility: create masking tensor based on probability `p` for query attention |
| `save_checkpoint()`  | Save model weights if validation improves |
| `load_best_model()`  | Load the best model after training completes |
| `visualize_attention()` | Optional: for interpretability analysis of attention maps |

---

This detailed logic should guide the precise and faithful implementation of the `Trainer` class, ensuring all core aspects of the experimental methodology, masking strategies, and model training described in the paper are integrated and reproducible.

## utils.py

# Logic Analysis for utils.py

This file will implement utility functions that facilitate the primary components of the CATS model and training pipeline, including positional encoding, masking strategies, visualization helpers, and core tensor operations. The functions must be designed for reusability, consistency, and compatibility with PyTorch tensors.

---

## 1. Positional Encoding

### Purpose:
- Enable the model to incorporate temporal order information into the patch embeddings.
- Use learnable positional embeddings as specified in the configuration.

### Requirements:
- Create a function to generate positional embedding matrices of size `[sequence_length, embed_dim]`.
- Provide both sinusoidal (non-learnable) and learned positional encodings, with preferences based on the config.

### Implementation Notes:
- As per the experimental setup, learnable positional embedding parameters are used.
- Create a function: `get_positional_encoding(seq_len: int, d_model: int, learnable: bool = True)` that returns a tensor.
- If `learnable=True`, initialize a torch.nn.Parameter of shape `[seq_len, d_model]`.
- If not learnable, generate sinusoidal positional encodings for more traditional embedding.

---

## 2. Masking Functions

### Purpose:
- Support the query-adaptive masking strategy during training, as specified.
- Implement probabilistic masks on the attention outputs or input patches.

### Requirements:
- A function: `generate_mask(sequence_shape: Tuple[int, ...], probability: float)` that returns a binary mask tensor `[sequence_shape]` with elements sampled from Bernoulli distribution with `p=probability`.
- Implement a function: `apply_mask(tensor: torch.Tensor, mask: torch.Tensor)` that applies the mask (element-wise) to the tensor. Typically, set masked positions to zero or detach.

### Additional:
- A specialized function for horizon-dependent masking: for each horizon query, generate a mask with probability `p` (from config), and potentially extend to masking entire attention outputs or input features.

---

## 3. Visualization Helpers

### Purpose:
- Show attention maps, forecast results, and attention score heatmaps.
- Provide functions to visualize cross-attention maps over patches, highlighting periodic/pattern properties.

### Requirements:
- `plot_attention_map(attention_scores: torch.Tensor, title: str = "")`
  - Plot heatmaps of attention matrices over patches.
  - Accept attention scores of shape `[n_heads, seq_len_q, seq_len_k]`.
- `plot_forecast_and_attention(forecast: np.ndarray, input_sequence: np.ndarray, attention_map: np.ndarray, title: str = "")`
  - Plot the forecasted time series alongside true signals.
  - Overlay attention heatmaps if appropriate.

### Implementation:
- Use `matplotlib.pyplot` for plotting.
- For high-dimensional attention maps, normalize scores, use `imshow()` with colorbars.
- For synthetic data visualization, overlay shocks or periodicity annotations.

---

## 4. Common Tensor Operations

### Purpose:
- Standardize operations such as normalization, splitting sequences into patches, and concatenation.
- Ensure consistency and reduce redundant code in main modules.

### Requirements:
- `normalize_tensor(tensor: torch.Tensor, method: str = "standard") -> torch.Tensor`
  - Perform z-score normalization (`(x - mean) / std`)
  - Or min-max scaling as an alternative.
- `split_into_patches(sequence: torch.Tensor, patch_size: int, overlap: int = 0) -> torch.Tensor`
  - Divide sequence with optional overlap into patches.
- `combine_patches(patches: torch.Tensor, overlap: int = 0) -> torch.Tensor`
  - Reconstruct original sequence (may only be used for evaluation/visualization).
  
### Additional:
- Functions to handle batch operations efficiently, e.g., batch embedding, masking.

---

## 5. Miscellaneous Utilities

### 5.1. Horizon-Dependent Query Embeddings:
- A function: `create_horizon_queries(num_horizons: int, embed_dim: int, learnable: bool = True)` 
  - Return a tensor of shape `[num_horizons, embed_dim]` with learnable parameters, initialized accordingly.

### 5.2. Parameter Initialization:
- `init_parameters(tensor: torch.Tensor, method: str = "xavier")`
  - For consistent initialization.

### 5.3. Attention Map Visualization:
- Specific function to generate overlays based on attention scores, for interpretability analysis as shown in figures.

---

## 6. Implementation Compatibility & Style
- Use PyTorch tensors throughout.
- All functions should be compatible with GPU tensors for seamless integration.
- Clear docstrings and type annotations.
- Modular design to facilitate testing and debugging.

---

## 7. Summary: Function List in utils.py

| Function Name | Purpose | Inputs | Outputs | Notes |
|----------------|---------|---------|---------|--------|
| `get_positional_encoding(seq_len, d_model, learnable=True)` | Generate positional encodings | sequence length, embedding size | torch.Tensor | learnable or sinusoidal |
| `generate_mask(shape, probability)` | Generate binary Bernoulli mask | shape, probability | torch.Tensor | batch-wise or sequence-wise |
| `apply_mask(tensor, mask)` | Apply mask to tensor | tensor, mask | masked tensor | mask to zero positions|
| `plot_attention_map(attention_scores, title="")` | Plot attention heatmap | attention tensor | None | visualize attention per head |
| `plot_forecast_and_attention(forecast, input_seq, attention_map, title="")` | Plot forecast and attention | forecast array, input array, attention scores | None | comparative visualization |
| `normalize_tensor(tensor, method="standard")` | Normalize tensor | tensor | normalized tensor | standardization or min-max |
| `split_into_patches(sequence, patch_size, overlap=0)` | Segmentation into patches | sequence tensor | patches tensor | shape: [num_patches, patch_size] |
| `combine_patches(patches, overlap=0)` | Reconstruct original sequence | patches tensor | sequence tensor | for visualization |
| `create_horizon_queries(num_horizons, embed_dim, learnable=True)` | Generate horizon queries | num horizons, embed dim | tensor | for cross-attention queries |
| `init_parameters(tensor, method="xavier")` | Initialize tensors | tensor | None | for consistent init |

---

## 8. Final Remarks:
- All functions should be designed to be as general as possible yet optimized for the specific use case in the cross-attention-only time series transformer.
- Emphasis on reproducibility: random seeds for initialization if relevant, consistent use of tensor types.
- Attention visualization functions will be crucial for interpretability analysis as in Figures 6–15.
- Ensure clarity in API and flexible parameters for different datasets/scenarios.

This completes the logic analysis for `utils.py`. Each function must be implemented with thorough testing and documentation, aligned with the outlined requirements, facilitating end-to-end experiment reproduction and interpretability analyses.

