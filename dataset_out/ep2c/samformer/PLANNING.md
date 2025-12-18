# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, detailed plan to reproduce the SAMformer methodology and experiments from the paper, ensuring fidelity to the described techniques, datasets, hyperparameters, and evaluation strategies.

---

## 1. Understanding and Extracting the Methodology

### A. Model Architecture Overview
- **Core Idea**: A shallow, lightweight transformer-based model enhanced with Sharpness-Aware Minimization (SAM) and channel-wise attention mechanisms for multivariate long-term forecasting.
- **Main Components**:
  - **Attention Module**: Channel-wise attention computed via a softmax over features, applied to the sequence.
  - **Residual Connection**: Adds input sequence directly to attention outputs to improve generalization.
  - **Linear Output Layer**: A simple linear transformation on the attention outputs.
  - **Normalization Layers**: Use of RevIN (Reversible Instance Normalization) to handle non-stationarity.
  - **Training Procedure**:
    - Adam optimizer with carefully tuned hyperparameters.
    - Use of SAM for robustness.
    - Regularization of attention via spectral normalization (or similar regularizer).
    - Explicit handling of non-stationarity (mean and variance normalization based on input features).

### B. Key Algorithmic and Mathematical Details
- **Attention Mechanism**:
  - Channel-wise attention matrix `A(X)` (Eq. 3, 4).
  - Application of softmax row-wise for attention distribution.
  - Attention matrix reshaped to dimensions consistent with input.
- **Loss Function & Optimization**:
  - Use of MSE and MAE metrics.
  - SAM regularization: maximization over a neighborhood in parameter space, approximated via first-order Taylor expansion.
  - Regular optimizer: AdamW preferred over Adam for stability.
- **Generalization & Stability Enhancements**:
  - Channel-specific normalization by mean/variance (RevIN).
  - Spectral normalization of weights.
  - Attention regularization (entropy and nuclear norm monitoring).

### C. Theoretical & Practical Aspects
- **Identity initialization & weight decay careful tuning**.
- **Sensitivity to neighborhood radius `ρ`** in SAM.
- **Model size**: lightweight (~few thousand parameters).
- **Training stability**: Robust to different seeds, hyperparameters tuned with grid searches.

---

## 2. Dataset Preparation & Requirements

### A. Datasets
- **ETTh1, ETTh2, ETTm1, ETTm2**:
  - Multivariate time series from electricity consumption, weather, trading, and traffic datasets.
  - Published publicly with detailed metadata.
  - **Features**:
    - Number of features (`D`): e.g., 7, 21, or larger.
    - Sequence length (lookback window): 17420, 69680, etc.
    - Granularity: 1 hour, 15 minutes, 10 minutes, etc.
- **Other datasets** (Electricity, Exchange, Traffic, Weather):
  - Can use publicly available versions or equivalent datasets with similar properties.
  - Data splits: 60% train, 20% validation, 20% test (or as per original).
  - Input normalization needed (per feature mean and variance).

### B. Data Processing
- Normalize each feature per the statistics during train.
- Generate sequences with window size `L`.
- For each sample:
  - Input: a sequence of length `L`.
  - Target: the forecast horizon `H` (predict subsequent `H` steps).
- Use window shifting for batching.

---

## 3. Model Implementation Details

### A. Normalization: RevIN
- Implement RevIN:
  - Compute feature-wise mean and variance per batch during training.
  - Normalize input sequences: 
    \[
    \tilde{x}_{k t}^{(i)} = \gamma_k \frac{x_{k t}^{(i)} - \hat{\mu}_k}{\sqrt{\hat{\sigma}^2_k + \epsilon}} + \beta_k
    \]
  - During inference, denormalize predicted outputs using stored train-set stats.

### B. Attention Module
- **Per Feature (Channel-wise)**:
  - Compute attention scores: softmax over features (Eq. 3, 4).
  - Attention matrix `A`: shape \( D \times D \).
  - Use scaled dot-product: attention scores based on learned projections \( W_Q, W_K \), possibly simplified to diagonal or feature-wise attention.
  - Apply a softmax row-wise to get stochastic attention.

### C. Residual Connections
- Add input sequence directly: input + attention-based residual.
- Followed by linear layer with weights \( W \) (Eq. 11).

### D. Output Layer
- Linear layer from the combined features → prediction sequence of size \( D \times H \).

### E. Regularization
- Spectral normalization on weights \( W_Q, W_K, W_V, W_O, W \).
- Attention entropy regularization: monitor during training (Section 2.2).

### F. Loss Function & Optimization
- **Loss**:
  - Mean Squared Error for training \(\mathcal{L}_{train}\).
  - Use SAM:
    - Compute loss at parameters perturbed by adversarial epsilon in the direction of gradient.
    - Approximate using first-order Taylor expansion (Eq. 13–14).
- **Optimizer**:
  - AdamW (recommended) with hyperparameters:
    - Learning rate ~ 1e-3, 1e-4, or tuned per dataset.
    - Weight decay: 1e-3 or 1e-4.
    - Implement warm-up schedule, if necessary.
- **SAM parameters**:
  - Neighborhood radius \(\rho\): tune between 1e-5 to 1e-2 based on dataset.
  - Perform two forward-backward passes per step:
    1. Gradient ascent step to find neighborhood max.
    2. Mini-batch gradient descent.

### G. Hyperparameter Tuning
- Learning rate and decay via grid search.
- Neighborhood size \(\rho\) based on validation.
- Number of epochs (~300), early stopping based on validation MSE/MAE.
- Batch size (e.g., 32 or 64, constrained by GPU memory).

---

## 4. Experimental Procedure

### A. Experimental Setup
- Use GPUs with sufficient memory (preferably ≥16 GB).
- Set seed for reproducibility.
- Run multiple seeds (at least 5) for statistical reporting.

### B. Baselines & Comparisons
- Implement or reimplement baselines like TSMixer, Transformer, and other listed models.
- Ensure comparable hyperparameters or use published settings.

### C. Training
- For each dataset:
  - Fix a validation set.
  - Train models with:
    - AdamW + SAM.
    - Default optimizers (SGD/Adam) for baseline comparison.
- Save checkpoints covering:
  - Best validation performance.
  - End of training.

### D. Evaluation
- Compute test MSE and MAE (per dataset, horizon).
- Use a standard test split.
- Record mean and std over multiple runs.

---

## 5. Metrics & Reporting
- Main metrics: MSE, MAE.
- Report:
  - Average over 5 runs with standard deviation.
  - Significance testing (e.g., t-test) for differences.
  - Model size (parameters).
  - Training time (GPU hours).
- Additional diagnostics:
  - Loss landscape visualization.
  - Attention matrix analysis (entropy, nuclear norm).
  - Sensitivity to hyperparameters (\(\rho, lr, weight\_decay\)).

---

## 6. Additional Practical Considerations
- Implement attention regularizers (entropy, nuclear norm).
- Record training dynamics (loss, validation performance over epochs).
- Visualize attention matrices, loss landscapes, and denormalized predictions.
- Follow paper-specified data splits, normalization, and hyperparameter tuning strategies.

---

## 7. Documentation & Reproducibility
- Use clear, version-controlled code (e.g., Git).
- Log hyperparameters and random seeds.
- Save intermediate outputs for analysis.
- Provide scripts for:
  - Data processing.
  - Model training.
  - Hyperparameter sweeps.
  - Evaluation and visualization.

---

## Summary Roadmap:
1. Prepare datasets matching the described properties.
2. Implement RevIN normalization.
3. Code the shallow transformer with channel-wise attention, residuals, and linear output.
4. Incorporate spectral normalization and attention regularization.
5. Set up AdamW optimizer with SAM-based training loop.
6. Tune hyperparameters (learning rate, \(\rho\), weight decay, epochs).
7. Conduct multi-seed experiments with proper train-val-test splits.
8. Evaluate and visualize results.
9. Compare models statistically and report comprehensive metrics.

---

This plan ensures you can systematically implement the exact architecture, training process, and experiments as in the SAMformer paper. Once executed, this roadmap will facilitate faithfulness to the original work and reproducibility across diverse datasets.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular, lightweight system using PyTorch for core model implementation, including custom modules for RevIN normalization, channel-wise attention, and spectral normalization. We will implement the training pipeline with SAM as a custom optimizer wrapper, leveraging PyTorch's autograd for the gradient ascent step. Data handling will use pandas and numpy for preprocessing, normalization, sequence generation, and batching. Visualization and analysis will be supported through matplotlib and seaborn. Every component will be organized into separate files for clarity, with a main script orchestrating data loading, model instantiation, training, evaluation, and visualization.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]\n        +generate_sequences(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]\n    }\n\n    class RevIN {\n        +__init__(feature_dim: int)\n        +fit_transform(sequence: np.ndarray) -> np.ndarray\n        +inverse_transform(sequence: np.ndarray) -> np.ndarray\n        +compute_stats(sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]\n    }\n\n    class TransformerModel {\n        +__init__(params: dict)\n        +forward(x: Tensor) -> Tensor\n        +predict(x: Tensor) -> Tensor\n    }\n\n    class SAMOptimizer {\n        +__init__(parameters: Iterable[Tensor], base_optimizer: torch.optim.Optimizer, rho: float)\n        +step(loss: Tensor, model: nn.Module) -> None\n        +zero_grad() -> None\n    }\n\n    class Trainer {\n        +__init__(model: TransformerModel, optimizer: SAMOptimizer, dataloader: DataLoader, device: torch.device)\n        +train() -> None\n        +validate() -> dict\n    }\n\n    class Evaluation {\n        +__init__(model: TransformerModel, dataloader: DataLoader, device: torch.device)\n        +evaluate() -> dict\n    }\n\n    class Visualization {\n        +plot_attention_matrices(attention_matrices: List[Tensor]) -> None\n        +plot_loss_landscape(loss_history: List[float]) -> None\n        +plot_predictions(predictions: np.ndarray, targets: np.ndarray) -> None\n    }\n\n    DatasetLoader --> RevIN\n    main.py --> DatasetLoader\n    main.py --> TransformerModel\n    main.py --> SAMOptimizer\n    main.py --> Trainer\n    main.py --> Evaluation\n    main.py --> Visualization\n    Trainer --> TransformerModel\n    Trainer --> SAMOptimizer\n    Evaluation --> TransformerModel\n    Visualization --> TransformerModel",
    "Program call flow": "sequenceDiagram\n    participant M as main.py\n    participant D as DatasetLoader\n    participant R as RevIN\n    participant Mdl as TransformerModel\n    participant Opt as SAMOptimizer\n    participant Tr as Trainer\n    participant Ev as Evaluation\n    participant Vis as Visualization\n    M->>D: load_data()\n    D-->>M: train_data, val_data, test_data\n    M->>R: init(feature_dim)\n    M->>D: generate_sequences()\n    D-->>M: train_seq, target_seq\n    M->>Mdl: initialize model\n    M->>Opt: initialize optimizer with model parameters\n    loop training epochs\n        M->>D: get batch\n        D-->>M: batch_x, batch_y\n        M->>R: normalize(batch_x)\n        R-->>M: normalized_x\n        M->>Mdl: forward(normalized_x)\n        M->>Mdl: compute loss\n        M->>Opt: backward(loss)\n        Opt->>Opt: perform SAM step with approximate gradient ascent\n        Opt-->>M: update_parameters\n        alt validation condition met\n            M->>Ev: evaluate(model, val_data)\n            Ev-->>M: metrics\n        end\n    end\n    M->>Ev: evaluate(model, test_data)\n    Ev-->>M: test_metrics\n    M->>Vis: plot_attention\n    M->>Vis: plot_loss_landscape\n    M->>Vis: plot_predictions\n    activate visualization\n    end",
    "Anything UNCLEAR": "Need clarification on the exact details of the attention implementation (per feature or feature-wise), normalization statistics management (RevIN parameters), hyperparameter ranges for \(\rho\), learning rate schedule, and whether custom spectral normalization and entropy regularization should be integrated explicitly. Also, confirm datasets format and desired output evaluation scripts."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "pandas==1.3.0",
        "matplotlib==3.4.2",
        "scikit-learn==0.24.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class which loads, preprocesses, and sequencifies datasets. Implements normalization stats computation (mean, std) for RevIN. Depends on pandas and numpy. Loads datasets like ETTh1/2, ETTm1/2, Weather, Traffic, etc. Handles batching and sequence generation."
        ],
        [
            "model.py",
            "Contains TransformerModel class implementing the lightweight channel-wise attention module (per Eq. 3, 4), residual connections, spectral normalization, and normalization layers (RevIN). Uses torch.nn modules with custom layers for attention and normalization. Reports attention matrices, supports denoising behavior."
        ],
        [
            "trainer.py",
            "Contains Trainer class managing training loop with AdamW optimizer, SAM regularization (via wrapper), loss computation (MSE/MAE), training oracle, validation, early stopping, and logging. Calls model.forward(), updates weights, and handles optimizer and SAM-specific steps."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class to run inference on test set, compute metrics (MSE, MAE), compare models, and generate plots."
        ],
        [
            "utils.py",
            "Utility functions for plotting attention heatmaps, loss landscape visualization, denormalization functions matching RevIN, and statistical testing routines."
        ],
        [
            "main.py",
            "Entry point script that loads datasets via DatasetLoader, initializes RevIN, builds model, sets up optimizer and SAM, orchestrates training epochs, performs validation, and runs final evaluation. Manages hyperparameters, seeds, and results saving."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Normalization parameters (mean, std per feature) computed during dataset loading are shared with the normalization layers. The optimizer wrapper for SAM uses the same model parameters across training and validation. Plotting functions in utils.py are shared for visualization tasks. Hyperparameters such as \rho, learning rate, weight decay, and batch size are shared configurations."
    ,
    "Anything UNCLEAR": "Clarification needed on the exact hyperparameter ranges for \(\rho\), whether to perform grid search or fixed values, and dataset normalization details for custom datasets. Also, confirm if the attention matrices visualization should be per batch or aggregated across validation/test sets."
}

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001  # Starting learning rate; tuned around 1e-3 as per typical recommendations
  batch_size: 32        # Standard batch size for training; adjust based on hardware capabilities
  epochs: 300           # Total training epochs; use early stopping as needed
  weight_decay: 1e-4    # Regularization parameter; tuned around 1e-4
  rho: 1e-4             # Neighborhood radius for SAM; small value, tune around 1e-4 to 1e-3
  optimizer: AdamW      # Chosen for stability, particularly with SAM
  seed: 42              # Set seed for reproducibility
  warmup_epochs: 10     # Optional: number of warm-up epochs if scheduler used

dataset:
  datasets:
    - name: ETTh1
      path: data/ETTh1.csv
      features: 7
      sequence_length: 17420
      prediction_horizon: 96
      granularity: hourly
    - name: ETTh2
      path: data/ETTh2.csv
      features: 7
      sequence_length: 69680
      prediction_horizon: 192
      granularity: hourly
    - name: ETTm1
      path: data/ETTm1.csv
      features: 321
      sequence_length: 26304
      prediction_horizon: 336
      granularity: 15min
    - name: ETTm2
      path: data/ETTm2.csv
      features: 8
      sequence_length: 7588
      prediction_horizon: 96
      granularity: 15min
    - name: Electricity
      path: data/electricity.csv
      features: 96
      sequence_length: 17544
      prediction_horizon: 96
      granularity: hourly
    - name: Exchange
      path: data/exchange.csv
      features: 21
      sequence_length: 52696
      prediction_horizon: 720
      granularity: daily
    - name: Traffic
      path: data/traffic.csv
      features: 8
      sequence_length: 17544
      prediction_horizon: 96
      granularity: hourly
    - name: Weather
      path: data/weather.csv
      features: 21
      sequence_length: 52696
      prediction_horizon: 96
      granularity: hourly
```

---

**Note:** Full configuration is available in `planning_config.yaml`
