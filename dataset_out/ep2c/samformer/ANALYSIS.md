# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

# Logic Analysis for dataset_loader.py

## Overview
The DatasetLoader class serves as the primary data handling component, responsible for loading multiple datasets, preprocessing them, computing normalization statistics, generating input-target sequence pairs suitable for training, and preparing data loaders for batching. Its correct implementation ensures that the data fed into the model accurately reflects the experimental setup described in the paper, including normalization, sequence windows, and dataset-specific properties.

---

## Core Responsibilities

### 1. Load Datasets
- Support multiple dataset formats and paths, referring to dataset specifications from the YAML configuration.
- Expected datasets come in CSV format with time series data arranged as features (columns) and time steps (rows).
- Implement a flexible data loader that can handle:
  - Different granularities (hourly, 15min, daily, etc.).
  - Different feature dimensions and sequence lengths.
  - Specific dataset properties as per the configuration.

### 2. Preprocessing & Normalization
- Compute per-feature statistics (mean, standard deviation) on the training set:
  - For each feature `k`, calculate:
    - \(\hat{\mu}_k = \frac{1}{L_{train}} \sum_{t=1}^{L_{train}} x_{k,t}\)
    - \(\hat{\sigma}_k^2 = \frac{1}{L_{train}} \sum_{t=1}^{L_{train}} (x_{k,t} - \hat{\mu}_k)^2\)
  - These statistics are essential for RevIN normalization both at train and inference.
- Store the computed mean and std for each feature for later denormalization if needed.
- Apply normalization:
  - Use the formula from RevIN:
    \[
    \tilde{x}_{k,t}^{(i)} = \gamma_k \frac{x_{k,t}^{(i)} - \hat{\mu}_k}{\sqrt{\hat{\sigma}_k^2 + \varepsilon}} + \beta_k
    \]
  - For train set, \(\gamma_k\) and \(\beta_k\) are learnable parameters; for normalization purposes, during preprocessing, initial values can be set to 1 and 0; during inference, these are fixed based on training statistics.

### 3. Sequence Generation
- Given full dataset time series, generate overlapping sequences for training/validation/test:
  - Input sequence: window of length `L`.
  - Target sequence: immediately following window of length `H` (prediction horizon).
  - Shift windows with stride 1 across the data to produce many samples.
- For each dataset, split data into:
  - Training set: initial \( \sim 60\% \) (e.g., 70% in total, split into 60/20/20 or as per description).
  - Validation set: next 20%.
  - Test set: remaining 20%.
- Maintain dataset-specific splits, ensuring that sequences are generated properly over the data segments.

### 4. Data Handling & Batching
- Store sequences as numpy arrays or tensors:
  - `X`: shape `(num_samples, D, L)` where:
    - `num_samples` varies with dataset size and stride,
    - `D` is feature dimension (number of variables),
    - `L` is sequence length.
  - `Y`: shape `(num_samples, D, H)` for future horizon.
- Implement a data generator or use PyTorch's Dataset class to:
  - Yield batches of size `batch_size`.
  - Shuffle training data.
  - Support multiple datasets through parameterization.

### 5. Dataset Support & Flexibility
- Support datasets specified with paths, feature counts, sequence lengths, and granularities.
- Handle missing or variable sampling intervals if present (ideally datasets are already preprocessed into regular intervals).
- Maintain consistent feature ordering across datasets.

### 6. Dataset Statistics Storage
- For each dataset:
  - Save computed per-feature mean and std for normalization.
  - These statistics are crucial for the normalization layers during training and inference.
- Persist statistics if needed in a separate file or via class attributes for consistent normalization during subsequent runs.

### 7. Additional Dataset-specific Handling
- For datasets with unique properties:
  - Different granularity units (hours, minutes, days).
  - Special missing value handling (if any).
  - Custom scaling if normalization isn't sufficient.
- Implement dataset-specific preprocessing logic as needed but keep a uniform interface.

---

## Implementation Details & Workflow

### Initialization:
- Instantiate DatasetLoader with configuration dict containing dataset info:
  ```python
  dataset_config = {
    'name': 'ETTh1',
    'path': 'data/ETTh1.csv',
    'features': 7,
    'sequence_length': 17420,
    'prediction_horizon': 96,
    'granularity': 'hourly'
  }
  ```
- Load dataset:
  - Read CSV (using pandas).
  - Convert to numpy array of shape `(total_time_steps, features)`.
  - Transpose or organize as `(features, total_time_steps)` internally for ease.
  
### Computation of Statistics:
- On training data only:
  - For each feature:
    - Calculate mean and std.
  - Store these as class attributes for normalization.
- Log or output these statistics for reproducibility.

### Data Segmentation:
- Segment full time series into:
  - Training sequences: first `L + H` + overlapping windows.
  - Validation sequences: next continuous block.
  - Test sequences: last part.
- For each segment, generate sequences with moving window:
  ```python
  for i in range(start_idx, end_idx - L - H + 1):
      x = data[:, i:i+L]
      y = data[:, i+L:i+L+H]
  ```
- Store these as lists or arrays.

### Data Loader Construction:
- Convert sequences into torch datasets:
  - Use `torch.utils.data.Dataset` subclass.
  - Implement `__getitem__` returning input tensor `(D, L)` and target `(D, H)`.
- Use DataLoader for batching and shuffling:
  ```python
  DataLoader(dataset, batch_size=batch_size, shuffle=True)
  ```

### Final Remarks:
- The loader should be flexible to handle different datasets with configuration provided.
- Carefully ensure reproducibility of normalization:
  - Fix normalization parameters during inference.
  - Use train stats for validation/test normalization.
- The entire dataset handling pipeline must align with the experimental setup detailed in the paper.

---

This detailed analysis of dataset_loader.py ensures a comprehensive and correct implementation aligned with the experimental methodology, datasets, and hyperparameters described in the SAMformer paper and configuration file.

## evaluation.py

{
  "evaluation.py": [
    "Purpose & Scope:",
    "The Evaluation class is designed to perform inference on trained models over test datasets, compute relevant metrics (Mean Squared Error (MSE) and Mean Absolute Error (MAE)), compare different models, and generate visualization plots such as attention heatmaps and loss landscape visualizations. It acts as the final step in the reproducibility pipeline for assessing model performance and interpretability.",
    
    "Main Components & Methodology:",
    "1. Initialization:",
    "   - Instantiate with: trained model instance, test dataset loader, normalization layer (RevIN), device (CPU/GPU), and configuration of metrics and visualization options.",
    "   - Load and set the trained model in evaluation mode.",
    "   - Ensure all normalization statistics (mean, std) used during training are accessible for denormalization.",
    
    "2. Test Set Inference:",
    "   - Loop over the test dataset loader (which yields batches of input sequences and targets).",
    "   - For each batch:",
    "       a. Apply normalization (RevIN normalization parameters) to input sequences before passing to the model.",
    "       b. Forward pass the normalized inputs through the model to get predictions.",
    "       c. Denormalize the model outputs using stored normalization stats and parameters to align with original data scale.",
    "       d. Collect the denormalized predictions and ground truth labels.",
    "   - After processing all batches, concatenate all predictions and ground truths for global metric calculation.",
    
    "3. Metric Computation:",
    "   - Calculate MSE and MAE globally over the entire test set predictions vs. targets.",
    "   - Given that the datasets are segmented into windows, compute metrics across all samples for each horizon and dataset.",
    "   - Store these metrics for reporting.",
    "   - Support multiple horizons H: for each prediction horizon, slice the predicted output and true target sequences accordingly before metric calculation.",
    "   - Use numpy or torch to efficiently compute squared errors and absolute errors.",
    
    "4. Model Comparison:",
    "   - With multiple models (e.g., SAMformer, transformer, other baselines), load their corresponding test predictions if saved, or perform inference with each.",
    "   - Collect metrics for each model and dataset/horizon combination.",
    "   - Implement statistical tests (e.g., t-test) to assess significance of performance differences, based on mean and standard deviations over multiple runs.",
    "   - Organize comparison results into tables or arrays for reporting.",
    
    "5. Visualization & Plots:",
    "   - Attention Heatmaps:",
    "     - Extract attention matrices from the trained model (available via model's internal state or stored during inference).",
    "     - For each dataset, plot the attention matrices as heatmaps using matplotlib or seaborn to visualize self-correlation among features.",
    "     - If multiple batches or epochs are involved, average over them for a smoother visualization.",
    "   - Loss Landscape visualization:",
    "     - Accept stored loss history or precomputed sharpness metrics across training epochs or parameter perturbations.",
    "     - Plot the loss landscape (e.g., using 2D slices of the loss around the trained parameters).",
    "     - Use utility functions such as plot_loss_landscape, ensuring axes represent parameter perturbations and color indicates loss value.",
    "   - Forecast Predictions & Residuals:",
    "     - Plot actual vs predicted sequences for a selection of samples, horizons, and datasets.",
    "     - Visualize residuals or errors to assess model fit quality.",
    "   - Additional plots may include attention entropy distributions, nuclear norms of matrices, or other interpretability measures as guided by the paper.",
    
    "6. Denormalization & Data Handling:",
    "   - Use stored normalization parameters (means, stds, beta, gamma) from RevIN for each feature to denormalize outputs.",
    "   - Confirm consistent normalization during training and inference.",
    "   - Manage multiple datasets with their specific feature counts and normalization stats, ensuring correctness in denormalization.",
    "   - Maintain data structures for predictions and ground truths, e.g., numpy arrays of shape (number of samples, feature, horizon).",
    
    "7. Output & Reporting:",
    "   - Return or save test metrics (MSE, MAE) for each dataset, horizon, and model into structured files (CSV, JSON).",
    "   - Log additional diagnostics: attention matrix distributions, entropy measures, nuclear norms, and training loss landscapes.",
    "   - Generate comparative tables and plots for publication or analysis.",
    
    "8. Implementation Considerations:",
    "   - Leverage the shared knowledge: use utility functions for plotting, denormalization, statistical testing.",
    "   - Make sure that the code accommodates multi-horizon evaluation, slicing sequences accordingly.",
    "   - Ensure compatibility with models trained using different seeds and hyperparameters, possibly aggregating results.",
    "   - Modular design: separate inference, metrics, visualization, and statistical analysis for maintainability.",
    "   - Maintain reproducibility by fixing random seeds and consistent normalization procedures.",
    
    "Summary: The logic encompasses:",
    "- Sequential inference over test data with proper normalization/denormalization.",
    "- Metric calculation with numpy/torch optimized computations.",
    "- Stimulating interpretability and diagnostic plots (attention heatmaps, loss landscape).",
    "- Statistical comparison and robust reporting with clear visualization.",
    "- Reproducibility, modularity, and dataset-specific handling based on the configuration YAML.",
    
    "This comprehensive plan directly aligns with the requirements in the paper, ensuring that the implementation faithfully reproduces the experimental results and visualizations described therein."
  ]
}

## main.py

# Logic Analysis for main.py

This script is the main orchestration point for the approach described in the SAMformer paper. Its responsibilities include dataset loading, normalization, model instantiation, optimizer setup (including SAM), training and validation loops with early stopping, and final evaluation and visualization. It should follow the outlined plan and ensure fidelity to the paper's methodology, datasets, hyperparameters, and experimental setup.

---

## 1. **Import necessary modules and packages**

- Import standard packages: numpy, torch, pandas, matplotlib, and any utilities.
- Import custom modules:
  - DatasetLoader from dataset_loader.py
  - RevIN class from model.py
  - TransformerModel class (or specific model class) from model.py
  - SAM optimizer wrapper class from trainer.py or utils.py
  - Trainer class from trainer.py
  - Evaluation class from evaluation.py
  - Config parser to load settings from `config.yaml`

## 2. **Reproducibility & Seed Setting**

- Read seed value from config.yaml (`seed: 42`)
- Set random seed for:
  - Python `random`
  - numpy
  - torch (`torch.manual_seed`)
  - torch.cuda (`torch.cuda.manual_seed_all`)
- Enable deterministic behavior for PyTorch if possible:
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

## 3. **Load configuration from `config.yaml`**

- Use pyyaml or `yaml` package to parse `config.yaml`.
- Extract hyperparameters:
  - Learning rate, batch size, total epochs, weight decay, rho (SAM neighborhood)
  - List of datasets with their paths, features, prediction horizons, etc.
  - Random seed
- Store configurations in a dictionary or namespace for easy access.

## 4. **Dataset Loading & Preprocessing**

- Instantiate DatasetLoader with dataset-specific parameters, such as:
  - Dataset name (ETTh1, etc.)
  - Path to data file
  - Sequence length (`L`)
  - Prediction horizon (`H`)
  - Granularity (for time normalization if needed)
- Call `load_data()` (or similar method) to load train, validation, test datasets:
  - It returns data as numpy arrays or pandas DataFrames.
- Generate sequences:
  - Use sliding window method with stride=1 as per paper.
  - Separate into inputs (`X`) and targets (`Y`).
- Perform normalization:
  - During dataset loading, compute feature-wise mean and variance for train data.
  - Pass training, validation, test data through RevIN normalization layers (or compute and store stats for normalization).
  - Make sure normalization is invertible (for denormalization during evaluation).

## 5. **Initialize RevIN normalization layers**

- Instantiate RevIN object with feature dimension `D`.
- During data preprocessing:
  - For each dataset split, apply `RevIN.fit_transform()` on input sequences.
  - Save normalization parameters (`mean`, `std`, `beta`, `gamma`) for denormalization in evaluation.

## 6. **Build the model**

- Instantiate `TransformerModel` with parameters:
  - Input dimension `D`
  - Model dimension (`d_m`) as specified (e.g., 16)
  - Attention parameters (e.g., feature-wise attention as per paper)
  - Whether to use spectral normalization
  - Number of attention heads (likely 1, as per paper)
  - Normalization layers (RevIN inside the model, if incorporated)
- The model should implement:
  - Attention module (channel-wise attention)
  - Residual connections (input + attention output)
  - Linear output layer
  - All modules as per paper's Eq. 3, 4, 11

## 7. **Setup optimizer with SAM**

- Use `torch.optim.AdamW` with:
  - Learning rate as per config (e.g., 0.001)
  - Weight decay as per config (e.g., 1e-4)
- Wrap optimizer with SAM:
  - Implement or import SAM wrapper class.
  - Pass model parameters, base optimizer, and `rho` from config.

## 8. **Training loop**

- For each epoch:
  - Set model to train mode.
  - Loop over training batches:
    - Load batch data `X_batch`, `Y_batch`.
    - Apply RevIN normalization:
      - Normalize input sequences with stored train stats.
    - Forward pass:
      - Pass normalized `X_batch` through model, get predictions.
    - Compute loss (e.g., MSE) over batch.
    - Zero out optimizer gradients.
    - Backpropagate:
      - Compute gradient of loss.
    - SAM step:
      - Perturb weights in the direction of the gradient (via the SAM wrapper method):
        - Approximate inner maximization by epsilon in the neighborhood.
        - Perform ascent step.
        - Compute loss at perturbed weights.
        - Perform descent step (update weights).
    - Store or log training loss (for reporting and visualization).
  - Validation:
    - Switch model to eval mode.
    - Run on validation set without gradient updates.
    - Compute validation metrics (MSE, MAE).
    - Save best model if validation improves.
  - Early stopping:
    - Monitor validation loss for early stop patience (e.g., 5 epochs).

## 9. **Evaluation**

- Load the best model checkpoint.
- Run inference on test set:
  - Normalize inputs with RevIN.
  - Forward pass for each sample.
  - Denormalize outputs with stored RevIN parameters.
  - Compute metrics: MSE, MAE.
- Save predictions and metrics.
- Optionally, generate plots:
  - Attention matrices heatmaps.
  - Loss landscape visualizations (if applicable).
  - Predicted vs ground truth series.

## 10. **Visualization & Save Results**

- Use `utils.py` functions for plotting:
  - Attention matrices (heatmaps).
  - Loss landscape.
  - Prediction overlays.
- Save figures to disk.
- Save final model and configurations for reproducibility.

## 11. **Repeat for Multiple Datasets & Prediction Horizons**

- Loop over dataset list from configuration:
  - For each dataset, perform the entire process.
- Repeat experiments for multiple seeds (e.g., 5) for statistical robustness.
- Aggregate results:
  - Compute mean and standard deviation of errors.
  - Perform significance testing if needed.

## 12. **Logging & Reproducibility**

- Log hyperparameters, dataset info, seed, and results.
- Save experiment logs, checkpoints, predictions, and plots.
- Use consistent file naming for comparison.

---

# Summary of main steps in main.py:
- Parse config.
- Set seed.
- For each dataset:
  - Load data.
  - Compute normalization stats.
  - Initialize RevIN.
  - Generate sequences.
  - Instantiate model.
  - Wrap optimizer with SAM.
  - Train with early stopping.
  - Validate and save best.
  - Final evaluation:
    - Inference.
    - Denormalization.
    - Error computation.
    - Visualization.
- Save overall results.

This detailed logic ensures the implementation aligns with the methodology and experimental setup described in the paper, considering all hyperparameters, datasets, and models’ specifics.

## model.py

{
  "dataset": "In model.py, the primary focus is on implementing the TransformerModel class, which encapsulates the core architecture outlined in the paper. This includes the channel-wise attention mechanism, residual connections, spectral normalization, and normalization layers (RevIN). The dataset is already loaded and processed in dataset_loader.py, and normalization stats (mean, std) are shared and passed to the model. The attention is applied to feature channels, not temporal sequence length; thus, attention matrix has shape D x D, where D is the number of features/dimensions. Modeling choices must align with the paper's description of a shallow, single-layer transformer without feedforward layers, optimized with SAM. The implementation must support extracting attention matrices during forward passes, to analyze entropy collapse and spectral properties, and to be compatible with the training routine that applies spectral normalization and normalization layers. All parameters W_Q, W_K, W_V, W_O, and W are trainable, with spectral normalization applied. Residual connections add the input to attention outputs before a final linear projection. This residual connection enhances trainability and stability, as indicated in the paper. The model should output the prediction tensor of shape B x L x D or D x H, matching the data shape, with the latter if batch is reshaped accordingly. The attention module is computed via scaled dot-product, with softmax row-wise, and then applied to the input sequence. During the forward pass, the attention matrix should be retained and stored for analysis (e.g., for entropy and nuclear norm). The normalization layer RevIN is integrated into the class, with its parameters (mean, std, beta, gamma) given during initialization or passed in. The spectral normalization is applied to the weight matrices during parameter definition, using torch.nn.utils.spectral_norm. The model's operation must be compatible with the training pipeline—accepting input tensors, applying normalization, computing attention, residual addition, linear output, then returning predictions along with attention matrices for analysis. Hyperparameters like model dimension (d_m), the attention projections' dimensions (d_qk), and regularization parameters must be configurable. The overall class should expose a forward() method returning predictions and attention matrices, facilitating training with SAM. Additional care should be taken to ensure the matrices are properly scaled, normalized, and that gradient flow is maintained. The implementation code should avoid altering the high-level design; only definitions matching the paper's equations and structure are permitted. The attention calculation (Eq. 4) is key and must follow the softmax normalization, with attention matrix stored for entropy analysis. Residual connections preserve feature information, aiding generalization and avoiding rank collapse issues. The layer should be simple, providing only the attention-aware residual path, followed by a linear projection using matrix W. The entire architecture aims for minimal depth—single-layer transformer—ensuring computational efficiency and interpretability. Finally, the class should be compatible with the overall training scheme, supporting gradient-based updates, spectral normalization, normalization, and the extraction of intermediate matrices for the analysis of loss landscape and attention properties. All parameter initializations should follow standard procedures (e.g., Xavier/Glorot)."
}

## trainer.py

# Logic Analysis for trainer.py

This file implements the training, validation, and testing routines for the SAMformer model, ensuring that all procedures are consistent with the methodology described in the paper. It manages data flow, optimizer steps (including SAM), loss computation, early stopping, and result logging. The following detailed steps and components provide a comprehensive plan for implementing this module.

---

## 1. Imports and Dependencies
- Import essential Python modules:
  - `torch`, `torch.nn`, `torch.optim`
  - Hyperparameter / configuration access
  - Utility functions: for loss computation, plotting, logging
- Custom classes:
  - Import or define the `SAMOptimizer` wrapper for SAM
  - Import the model (`TransformerModel`)
  - Import dataset loader functions (from dataset_loader.py)

---

## 2. Initialization
- **Inputs**:
  - `model`: an instance of `TransformerModel`
  - `optimizer`: an instance of `SAMOptimizer` or a standard optimizer (e.g., AdamW)
  - `train_loader`: DataLoader for training data
  - `val_loader`: DataLoader for validation data
  - `device`: computation device (GPU/CPU)
  - `config`: dict containing hyperparameters (learning rate, rho, epochs, early stopping criteria, seed, etc.)
  - Additional args: loss type (MSE, MAE), flags for model evaluation, logging

- **Setup**:
  - Set seed across torch, numpy, python to guarantee reproducibility.
  - Initialize logs, history trackers for loss, metrics, etc.
  - Track `best_val_loss` for early stopping.
  - Create directories for saving models and logs if needed.

---

## 3. Data Processing in Each Epoch
- Loop over epochs:
  - **Training Phase**:
    - Set model in training mode: `model.train()`.
    - Loop over batches from `train_loader`:
      - Zero optimizer gradients: `optimizer.zero_grad()`.
      - Retrieve batch inputs (`batch_x`) and targets (`batch_y`).
      - Normalize inputs with RevIN if applicable; ensure normalization stats are stored.
      - Move `batch_x` and `batch_y` to device.
      - **Forward pass**:
        - Call `outputs = model(batch_x)`.
        - (Outputs shape: batch_size x D x H)
      - **Loss calculation**:
        - Compute `loss = criterion(outputs, batch_y)`.
        - `criterion`: MSELoss or MAELoss as configured.
      - **SAM step**:
        - If using SAM:
          1. Compute gradients normally: `loss.backward()`.
          2. Call `optimizer.first_step()` with `model`.
          3. Recompute loss at perturbed parameters: `loss_sam = criterion(model(batch_x), batch_y)`.
          4. Zero gradients, compute gradients again: `optimizer.zero_grad()`.
          5. Call `optimizer.second_step()` with `model`.
        - Else:
          - Call `loss.backward()`.
          - Step optimizer: `optimizer.step()`.
      - **Logging**:
        - Accumulate batch loss for epoch avg.
        - Optionally, compute attribute metrics on batch.
  - End batch loop:
    - Compute epoch training loss as average over batches.
    - Save loss history for potential plotting.
      
- **Validation phase** (every epoch or at set intervals):
  - Set model in eval mode: `model.eval()`.
  - No gradient calculations: `with torch.no_grad()`.
  - Loop over `val_loader`:
    - Forward pass.
    - Compute validation loss (and metrics).
  - Average validation loss over validation set.
  - Save `best_val_loss` if current validation loss improves:
    - Save model checkpoint at best validation performance.
    - Log epoch metrics.
    - If no improvement over `patience` epochs, trigger early stopping.

---

## 4. Early Stopping
- Track validation loss after each epoch.
- Implement a patience counter:
  - If validation loss does not improve for `patience` epochs:
    - Stop training early.
    - Load best model checkpoint.
  - Else, continue training until epochs exhausted.

---

## 5. Final Testing
- After training completion (early stopping or epoch limit):
  - Load best model checkpoint.
  - Set in eval mode.
  - Use `test_loader`:
    - Loop over test batches:
      - Forward pass.
      - Store outputs and targets.
  - Compute final test metrics: MSE, MAE (per dataset, horizon).
  - Denormalize outputs using RevIN inverse transform, if normalization was used.
  - Save/Plot test predictions compared to ground truth.

---

## 6. Logging and Visualization
- During training:
  - Plot training/validation loss curves.
  - Record metrics per epoch.
- After testing:
  - Visualize attention matrices:
    - Retrieve attention weights from model (if accessible).
    - Use `utils.py` plot functions.
  - Visualize loss landscape if needed (using stored loss history or pre-computed data).
  - Plot predicted vs true sequences.
- Save logs, model checkpoints, and metrics to files.

---

## 7. Hyperparameters and Configs
- Use values from `config.yaml`:
  - `learning_rate`, `weight_decay`, `rho`, `epochs`, `batch_size`, `seed`, `patience`.
- Adjust model training schedules:
  - Learning rate decay or warmup, if specified.
  - Early stopping parameters.
- Be consistent with the hyperparameters used in the paper’s experiments.

---

## 8. Additional Considerations
- Maintain clear separation between training and evaluation modes.
- Handle device transfers and data normalization properly.
- Implement a modular design so individual functions (train_batch, validate, test) can be reused and tested separately.
- Ensure reproducibility via seed setting.

---

## Summary
This logic analysis thoroughly plans the implementation of `trainer.py`. It emphasizes correct data flow, SAM application, loss computation, early stopping, and detailed logging consistent with the paper's methodology. The key is to respect the training sequence of:

- Data loading → normalization with RevIN → model forward → loss + SAM step → optimizer step → validation checks → test evaluation.

By following this, the reconstructed code will faithfully reproduce the model training and evaluation process outlined in the SAMformer paper.

## utils.py

{
  "utils.py": [
    {
      "component": "Plot Attention Heatmaps",
      "description": "Implement a function to visualize attention matrices as heatmaps, aiding qualitative analysis of attention patterns post-training. The function should accept attention matrices collected during evaluation or validation phases, as numpy arrays or torch tensors. It should generate heatmaps (e.g., via matplotlib's imshow() or seaborn's heatmap()) for each attention matrix, possibly averaged over batches or selected samples for clarity.",
      "details": [
        "Input: attention_matrices - list or array of matrices, shape (num_samples, D, D), where D is feature dimension.",
        "Normalization: Optionally normalize attention matrices (e.g., divide by max for clarity).",
        "Plot customization: axis labels, color maps (e.g., 'viridis'), titles indicating dataset and horizon.",
        "Output: Save plots to files or display inline (for Jupyter).",
        "Use case: Visual difference between SAMformer and other variants like sigma reparam, or to analyze diagonal dominance."
      ]
    },
    {
      "component": "Loss Landscape Visualization",
      "description": "Create a function to visualize the loss landscape (e.g., via 2D surface plots or contour plots) along directions in parameter space to interpret sharpness/flatness of minima. Based on the paper, the 'loss landscape analysis' involves plotting loss over axes aligned with certain directions (e.g., the gradient or random directions).",
      "details": [
        "Input: loss_history - list of loss values across perturbations or steps; or matrices defining perturbation directions.",
        "Generate a grid of values over two directions, compute loss at each grid point by perturbing model parameters accordingly.",
        "Use meshgrid + surface or contour plots in matplotlib.",
        "The function should be flexible to handle different directional axes, based on stored eigenvectors or random directions.",
        "Display the smoothed landscape illustrating the curvature/sharpness."
      ]
    },
    {
      "component": "Denormalization Functions for RevIN",
      "description": "Implement functions to reverse the RevIN normalization on sequences and predictions, reproducing the denormalization step described in Appendix D.1 and D.2. These functions should take normalized outputs, stored per-feature means (μ), variances (σ²), and learned parameters β, γ, and restore the original scale.",
      "details": [
        "Input: normalized sequence/tensor, stats (μ, σ²), and normalization parameters (β, γ).",
        "Operation: For each feature k, denormalized_value = ((normalized_value - β_k) / γ_k) * sqrt(σ²_k + ε) + μ_k.",
        "Ensure batch processing is supported.",
        "This function ensures the evaluation and visualizations reflect true data distribution."
      ]
    },
    {
      "component": "Statistical Testing Routine (Student's t-test)",
      "description": "Provide a function to perform statistical significance testing (e.g., paired t-test) between different model performance metrics (MSE or MAE), across multiple runs. Implement the test to accept two arrays: metrics (shape: number_of_runs).",
      "details": [
        "Input: performance arrays (e.g., test MAE or MSE from multiple seeds).",
        "Output: t-statistic, p-value, and whether the difference is statistically significant (e.g., p < 0.05).",
        "Use scipy.stats.ttest_rel or ttest_ind depending on experimental design.",
        "Optional: output confidence intervals for mean differences.",
        "This routine aligns with the significance test analysis in Appendix B.2."
      ]
    },
    {
      "component": "Additional Utility Functions",
      "description": "Include optional helpers to support dataset handling, like plotting overall metrics across datasets and horizons, or aggregating attention matrices (e.g., mean over layers or samples). These may include plotting bar charts or line plots for performance comparison or parameter impact visualization.",
      "details": [
        "Input: data arrays, labels, optional configuration for plot aesthetics.",
        "Purpose: facilitate comprehensive analysis and presentation of results aligned with the paper's figures and tables."
      ]
    }
  ],
  "Notes": [
    "All visualization functions should accept the relevant data in numpy or torch tensor format, and internally convert to numpy if needed.",
    "Ensure plotting functions include labels, titles, legends, and color bars for clarity.",
    "All functions should be modular, self-contained, and well-documented to integrate smoothly with main training/evaluation scripts.",
    "Implement default parameters for figures (size, dpi, color map) consistent with best practices for scientific plots."
  ]
}

