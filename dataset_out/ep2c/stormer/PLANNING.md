# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, step-by-step plan to reproduce the Stormer methodology and experiments as described in the paper. This outline covers the key scientific concepts, implementation details, and experimental procedures necessary for faithful reproduction.

---

# 1. Overall Objective
Reimplement Stormer, a scalable transformer-based deep learning weather forecasting model that employs:
- Weather-specific variable embeddings
- Randomized iterative forecasting
- Pressure-weighted loss
- Multi-phase training with fine-tuning
- Inference via diverse interval combinations and ensembling
- Robust evaluation metrics

---

# 2. Data Requirements & Processing
### Datasets
- **ERA5 reanalysis data** (curated WeatherBench 2 version)
  - Variables:
    - Surface: 2-meter temperature (T2m), 10m U/V wind components (U10, V10), Mean Sea Level Pressure (MSLP)
    - Atmospheric: Geopotential height (Z), temperature (T), wind (U, V), humidity (Q) at 13 pressure levels:
      {50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000} hPa
- Downsample to 6-hourly, 128×256 grid (or as specified in main experiments)
- Data splits: train (1979–2018), validation (2019), test (2020)
- Data preprocessing:
  - Regrid all models to common resolution (e.g., 128×256)
  - Normalize variables:
    - For each variable, compute mean and std over training set
    - Standardize (subtract mean, divide by std)
  - Store variables as tensors: shape (time_steps, V, H, W)

### Data Storage
- Use netCDF/xarray for raw data reading
- Save processed data as tensors for efficient loading during training

---

# 3. Model Architecture & Embedding
### Inputs
- Initial condition tensors $X_0$ (variables × spatial grid)
- Conditioning scalar $\delta t$ (lead time interval)

### Embedding Module
- **Weather-specific embedding:**
  - Tokenization:
    - Embed each variable independently with a linear layer into a “patch sequence”
    - Patch size $p$ (e.g., 2 or 4) determines the spatial tokenization granularity
    - Output shape: (H/p, W/p, V, D)
  - Variable aggregation:
    - Cross-attention with a learnable query vector over the variable dimension
    - Output shape: (H/p, W/p, D)
- **Positional Encoding:**
  - 2D sinusoidal or learned positional embeddings added to tokens

### Transformer Stack
- Use a stack of $N$ identical transformer blocks (e.g., 24 blocks)
- **Within each block:**
  - Multi-head self-attention
  - Feedforward network
  - Adaptive Layer Normalization (AdaLN), conditioned on $\delta t$ via a small MLP
- **Input:**
  - Tokens + positional embeddings + conditioned $\delta t$ (via AdaLN)
- **Output:**
  - Sequence of tokens, shape (H/p, W/p, D)

### Output Prediction
- For each variable, predict the *difference in variable* over interval $\delta t$:
  - $\widehat{\Delta}_{\delta t}^{v i j}$
- Head: linear layer projecting to variable domain shape, matching variable units

---

# 4. Randomized Iterative Forecasting
### Training
- **Objective:**
  - Minimize pressure-weighted MSE between predicted $\widehat{\Delta}_{\delta t}^{v i j}$ and true
  - For randomized $\delta t$ drawn uniformly from $\{\bar{6}, 12, 24\}$ hours (or other sets)
  - Use multi-phase training:
    1. Phase 1: train for 1-step
    2. Phases 2/3: fine-tune with $K=4$, then $K=8$ rollout steps
- **Implementation:**
  - For each batch:
    - Sample random $\delta t$ (or set)
    - Compute true $\Delta_{\delta t}$
    - Compute loss between predicted $\widehat{\Delta}$ and true
  - During inference simulation:
    - Generate multi-interval sequences by combining different $\delta t$ intervals summing to target lead time $T$.
    - Use iterative feeding: output from one step re-input as initial for next, allowing multiple treatment of the same model

---

# 5. Loss Function Components
### Pressure-Weighted Loss
- Variables are at pressure levels; weight each variable according to pressure (e.g., more near surface)
- Total loss:
  \[
  \mathcal{L}(\theta) = \mathbb{E}\left[\frac{1}{V H W} \sum_{v, i, j} w(v) L(i) (\widehat{\Delta}^{v i j} - \Delta^{v i j})^2 \right]
  \]
- $w(v)$: pressure-proportional weights
- $L(i)$: latitude weight

### Multi-step Fine-tuning
- Roll-out $K=4$, then $K=8$ steps, fine-tune checkpoint
- Loss averaged over steps
- Use same $\delta t$ during fine-tuning

---

# 6. Inference Strategy
### Generating Forecasts
- For target lead time $T$:
  - Compose multiple combinations of $\delta t$ intervals summing to $T$
    - Homogeneous: same $\delta t$ (e.g., all 6-h)
    - Heterogeneous: different $\delta t$s
  - Generate forecasts for each sequence using iterative rollouts
  - Average the predictions (ensembling)
- **Heterogeneous combination selection:**
  - Generate $n$ different combinations
  - Evaluate validation loss for each
  - Select top $m$ to produce final forecast
- **Ensembling:**
  - Average over top $m$ forecasts

---

# 7. Hyperparameters & Optimization
### Training
- Optimizer: AdamW
- Learning Rate:
  - Phase 1: 5e-4, warmup 10 epochs, cosine schedule to epoch 100
  - Phases 2/3: 5e-6, 5e-7, warmup 5 epochs, schedule 20 epochs
- Batch size: based on GPU memory (e.g., 128 devices with mixed precision; memory reduced with gradient checkpointing)
- Epochs:
  - Phase 1: 100 epochs
  - Finetuning phases: 20 epochs each
- Early stopping based on validation loss (aggregated on selected variables and lead times)
- Loss weights, variable embedding dimensions, patch size, model size (depth, width), number of transformer layers
- Model sizes tested:
  - Baseline: 1024 dims, 24 layers, patch size 2
  - Scaling: larger dims, patch size 4, more layers

### Hardware
- Use multiple A100 (or similar) GPUs
- Mixed precision training (fp16 or bfloat16)
- Utilize Distributed Data Parallel and gradient checkpointing

---

# 8. Evaluation Metrics & Protocol
### Metrics
- RMSE, ACC, SSR (per variable and lead time)
- Use latitude-weighted evaluation
- Regrid forecasts to common grid

### Protocol
- Generate forecasts at 00UTC initial conditions
- Lead times: 1-14 days
- Compare:
  - Homogeneous (single interval)
  - Heterogeneous (ensemble of combinations)
- Report mean and confidence metrics
- Visualize predictions vs. ground truth, error maps at specific lead times
- Perform ablation studies (components, model size, patch size, ensemble strategies)

---

# 9. Reproducibility & Transparency
- Document hyperparameters
- Include code for data processing, variable embedding, training loop, loss functions, inference tensor combination
- Release trained checkpoints, evaluation scripts
- Make clear data licensing, code licensing, and safe data handling practices

---

# Summary of Key Implementation Steps:
1. Process ERA5 data, standardize, store as tensors
2. Implement weather-specific embedding + positional encodings
3. Build transformer stack with AdaLN conditioned on $\delta t$
4. Train with randomized $\delta t$ and pressure-weighted loss
5. Fine-tune for multi-step rollout
6. Develop inference via diverse composition, ensembling
7. Evaluate with standardized metrics and protocols
8. Scale model in size, patch, sequence length for scaling studies
9. Document and manage licenses, safety, reproducibility details

---

This roadmap synthesizes all relevant details for a faithful, efficient reproduction of Stormer’s methodology and experiments. Adjust hyperparameters and model scale as available computational resources permit.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular deep learning system using PyTorch and PyTorch Lightning for scalable training. The core components include a custom transformer model with weather-specific variable embeddings and adaptive layer normalization conditioned on $\delta t$, along with a flexible inference pipeline implementing combination and ensembling strategies. The dataset loader will preprocess ERA5 reanalysis data into standardized tensors, applying variable normalization and regridding. We will implement random $\delta t$ sampling within the training loop, optimize with AdamW, and include multi-phase training with fine-tuning steps. The inference module will generate multiple interval combinations, perform iterative rollouts, and compute ensembled forecasts. Evaluation code will compute RMSE, ACC, SSR, at specified lead times, on the regridded data. Hyperparameters, model sizes, and input configurations will follow the paper's settings to ensure fidelity.",
    "File list": [
        "main.py",
        "dataset.py",
        "model.py",
        "trainer.py",
        "inference.py",
        "evaluation.py",
        "utils.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "
classDiagram
    class WeatherDataset {
        +__init__(config: dict)
        +load_data() -> Tuple[Tensor, Tensor]
        +get_train_loader() -> DataLoader
        +get_val_loader() -> DataLoader
        +get_test_loader() -> DataLoader
    }
    class TransformerModel {
        +__init__(model_params: dict)
        +embed_variables(input: Tensor, delta_t: Tensor) -> Tensor
        +forward(tokens: Tensor, delta_t: Tensor) -> Tensor
        +predict_delta(): Tensor
    }
    class ConditioningMLP {
        +__init__(input_dim: int, output_dim: int)
        +forward(x: Tensor) -> Tuple[Tensor, Tensor]
    }
    class TrainerLightning {
        +__init__(model: TransformerModel, optimizer_params: dict, scheduler_params: dict)
        +train(dataloader: DataLoader, epochs: int) -> None
        +finetune(dataloader: DataLoader, epochs: int) -> None
        +save_checkpoint(path: str) -> None
        +load_checkpoint(path: str) -> TransformerModel
    }
    class InferencePipeline {
        +__init__(model: TransformerModel)
        +generate_intervals(master_lead_time: int, mode: str = 'hetero') -> List[List[int]]
        +rollout_forecast(initial_condition: Tensor, sequence: List[int]) -> Tensor
        +combine_ensemble(forecasts: List[Tensor]) -> Tensor
    }
    class EvaluationMetrics {
        +compute_rmse(pred: Tensor, target: Tensor) -> float
        +compute_acc(pred: Tensor, target: Tensor) -> float
        +compute_ssr(pred: Tensor, target: Tensor) -> float
        +compute_metrics(preds: List[Tensor], targets: Tensor) -> dict
    }
    WeatherDataset --> DataLoader
    TransformerModel --> ConditioningMLP
    TrainerLightning --> TransformerModel
    InferencePipeline --> TransformerModel
    EvaluationMetrics --> Tensor
    Main workflow: Main -> WeatherDataset -> TrainerLightning -> InferencePipeline -> EvaluationMetrics
    "
    ,
    "Program call flow": "
sequenceDiagram
    participant App as main.py
    participant Data as dataset.py
    participant Model as model.py
    participant Train as trainer.py
    participant Infer as inference.py
    participant Eval as evaluation.py
    App->>Data: instantiate datasets with config
    Data-->>App: DataLoaders for train, val, test
    App->>Model: instantiate Transformer with params
    App->>Train: train(model, DataLoader, epochs=100)
    Note over Train: Phase 1 training (single-step)
    Train-->>App: saved checkpoint
    App->>Train: finetune(model, DataLoader, epochs=20) for phase 2
    Note over Train: For rollout K=4
    Train-->>App: save fine-tuned checkpoint (phase 2)
    App->>Train: finetune(model, DataLoader, epochs=20) for phase 3 (K=8)
    Note over App: During inference, generate interval combinations
    App->>Infer: generate_intervals(lead_time=$target_day, mode='hetero')
    Infer->>Model: iterative rollout for each interval sequence
    Model-->>Infer: forecast tensor
    Infer->>Infer: combine forecasts via ensemble
    Infer-->>App: ensemble forecast
    App->>Eval: compute_metrics(forecast, ground_truth)
    Eval-->>App: return metrics
    "
    ,
    "Anything UNCLEAR": "Need clarification on the exact variable normalization ranges and whether custom positional embeddings are preferred. Also, details on evaluation variable selections at each lead time are needed."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.0",
        "torchvision==0.13.0",
        "pytorch_lightning==1.9.0",
        "numpy==1.21.0",
        "xarray==0.20.2",
        "netCDF4==1.5.8",
        "timm==0.4.12",
        "PyYAML==6.0",
        "hydra-core==1.3.2",
        "scipy==1.7.3",
        "scikit-learn==1.0.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset.py",
            "Defines WeatherDataset class that handles ERA5 data loading, preprocessing (normalization, regridding), and provides data loaders for train/validation/test. Depends on xarray, netCDF4; uses utils.py for normalization routines."
        ],
        [
            "model.py",
            "Implements TransformerModel class that constructs the custom transformer architecture with weather-specific embeddings, variable aggregation via cross-attention, AdaLN conditioned on delta_t, and output heads for delta prediction. Depends on PyTorch, timm (for transformer blocks), and utils.py for positional encoding."
        ],
        [
            "trainer.py",
            "Implements Lightning-based Trainer class that manages training, multi-phase finetuning, checkpointing, and loading. Uses pytorch_lightning, defines optimizer and scheduler configs, and handles training loop with randomized delta_t sampling, pressure-weighted loss, and multi-step rollout fine-tuning."
        ],
        [
            "inference.py",
            "Provides InferencePipeline class that generates multiple delta_t interval combinations, performs iterative forecast rollouts using trained model, and combines forecasts via averaging. Also includes functions for generating interval sets and evaluating ensembles. Depends on model.py for model inference and utils.py for combining."
        ],
        [
            "evaluation.py",
            "Contains EvaluationMetrics class that calculates RMSE, ACC, SSR over the forecast horizon, supporting regridding and weighted metrics. Depends on torch and numpy."
        ],
        [
            "main.py",
            "Entry point script to coordinate data loading, model instantiation, training phases, inference, and evaluation. Parses configs, manages experiment workflow, and logs results. Depends on all other modules."
        ],
        [
            "utils.py",
            "Provides utility functions for pressure weighting, positional encoding, normalization routines, interval combination generation, and ensemble averaging. Shared across dataset, model, inference, and evaluation modules."
        ]
    ],
    "Task list": [
        "utils.py",
        "dataset.py",
        "model.py",
        "trainer.py",
        "inference.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "utils.py contains pressure weighting schemes, positional encodings, interval combinatorics, and ensemble averaging functions. Model.py relies on the specific variable embedding architecture. The training code shares normalization routines and data loader logic. Evaluation.py and inference.py depend on it for combining forecasts."
    ,
    "Anything UNCLEAR": "Clarify the exact range of hyperparameters for the adaptive layer norm conditioning (e.g., sizes, initialization). Confirm data exact preprocessing pipeline (regridding method, normalization statistics). Clarify whether to support heterogeneous interval combinations or only homogeneous ones."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## Config file for Stormer weather forecasting model

dataset:
  data_path: "path/to/ERA5_processed_data.nc"
  variables:
    surface:
      - T2m
      - U10
      - V10
      - MSLP
    atmospheric:
      - Z
      - T
      - U
      - V
      - Q
  pressure_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  grid_resolution: 128  # Target grid resolution
  downsample_hours: 6  # 6-hourly data
  train_start: "1979-01-01"
  train_end: "2018-12-31"
  valid_start: "2019-01-01"
  valid_end: "2019-12-31"
  test_start: "2020-01-01"
  test_end: "2020-12-31"

model:
  # Transformer architecture parameters
  patch_size: 2
  hidden_dim: 1024
  num_layers: 24
  num_heads: 16
  mlp_ratio: 4
  dropout_rate: 0.1
  attention_dropout: 0.1
  max_position_embeddings: 1024

training:
  batch_size: 128  # Number of samples per device
  epochs_phase1: 100
  epochs_finetune_2: 20
  epochs_finetune_3: 20
  learning_rate_phase1: 5e-4
  learning_rate_finetune_2: 5e-6
  learning_rate_finetune_3: 5e-7
  warmup_epochs: 10
  warmup_epochs_finetune: 5
  weight_decay: 1e-5
  optimizer: AdamW
  scheduler: cosine
  gradient_checkpointing: true
  mixed_precision: true
  device_count: 128

training_strategies:
  randomized_delta_t: [6, 12, 24]  # Hours
  rollout_steps: [1, 4, 8]
  interval_combinations: "generate using utils.py"  # Use utility functions for combinations

loss:
  pressure_weighting: true
  pressure_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  variable_weights:
    surface: 1.0
    U10: 0.1
    V10: 0.1
    MSLP: 0.1
  loss_function: MSE

inference:
  lead_time_intervals: [6, 12, 24]  # Hours
  combination_mode: "heterogeneous"
  ensemble_top_m: 3
  ensemble_n: 128  # Number of random combinations to evaluate
  combine_method: "mean"  # Or "ensemble"

evaluation:
  metrics:
    - RMSE
    - ACC
    - SSR
  lead_times: [1, 3, 5, 7, 10, 14]  # Days
  regrid_data: true  # Ensure all forecasts are on the same grid

logging:
  save_dir: "checkpoints/stormer/"
  save_checkpoint_interval: 10  # Save every 10 epochs
  early_stopping_patience: 15

dataset_statistics:
  normalization_stats: "computed over training set"
  pressure_weighting_scheme: "as per paper; pressure levels used for weighting"

hardware:
  devices: 128 A100 40GB GPUs
  use_mixed_precision: true
  enable_gradient_checkpointing: true
  cluster_strategy: "DistributedDataParallel"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
