# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a comprehensive, detailed plan to reproduce the experiments and methodologies described in the paper "Are Self-Attentions Effective for Time Series Forecasting?" The plan addresses key aspects of the architecture design, training procedures, dataset requirements, experimental configurations, and evaluation protocols, based on the information provided.

---

## 1. Overall Research Question & Objective
- **Primary goal:** Empirically and analytically evaluate whether self-attention is necessary or beneficial for time series forecasting.
- **Hypotheses to test:** 
  - Removing self-attention and relying solely on cross-attention mechanisms can outperform existing Transformer-based models.
  - The proposed Cross-Attention-Only Time Series Transformer (CATS) is more parameter-efficient, computationally efficient, and achieves superior forecasting performance.

---

## 2. Model Architecture & Methodology

### Core Architectural Innovations:
- **Elimination of Self-Attention Modules:** 
  - Traditional Transformer modules (self-attention + cross-attention) are replaced with a purely cross-attention-based architecture.
- **Cross-Attention Mechanism:**
  - Use the future horizon (target time steps) as **queries**.
  - Use the past observed series as **keys and values**.
- **Parameter Sharing:**
  - Establish horizon-dependent parameters as **learnable queries** (via neural embedding layers).
  - Share parameters **across horizons** for efficiency.
- **Input/Output Patches:**
  - Segment the time series into patches (non-overlapping or moderately overlapping) of fixed length (e.g., 24 steps).
  - Encode each patch into embeddings, applied to both inputs and outputs.
- **Query-Adaptive Masking:** 
  - Apply masking strategies during training, probabilistically mask output queries or their embeddings, as controlled by parameter \( p \), to enhance robustness and generalization.
- **Model Components & Settings (as per experimental setup):**
  - Number of layers: 3 (encoder and decoder modules).
  - Number of attention heads: 2.
  - Embedding dimension: 256 (or as specified).
  - Patch size: 24 (or as specified for different datasets).
  - Input sequence length (\( L \)): e.g., 96, 192, 336, 512, etc., depending on dataset or experiment.
  - Forecast horizon (\( T \)): e.g., 72, 96, 144, 288, etc.
  
### Additional design notes:
- **No embedding layer for self-attention:** 
  - Inputs are directly segment patches, embedded into learnable vectors.
- **Multi-head cross-attention:**
  - Compute attention between each output query patch and all input patches.
  - Use learnable query vectors corresponding to each horizon.
- **Parameter sharing:**
  - Across all horizons, share the same attention parameters for efficiency and interpretability.

### Implementation Details (based on text):
- Map the future horizon as a sequence of queries, each specific to a forecast step.
- Treat the entire process as a sequence-to-sequence model but with only cross-attention modules.
- Embedding sizes, number of layers, heads, and masking hyperparameters should be tuned as per the experimental settings.

---

## 3. Dataset & Data Processing
### Datasets:
- **Primary datasets:**
  - **ETT (ETTm1, ETTm2):** Hourly data with multiple variables (like temperature, electricity, traffic).
  - **Weather:** 10-minute interval series with temperature or similar signals.
  - **Traffic:** Hourly and/or 15-minute series.
  - **M4 competitions:** Quarterly, monthly, yearly datasets, with standardized splits and evaluation.
  - **Electricity:** Hourly or multi-variate data.
- **Additional datasets:**
  - Any other datasets with similar multivariate and univariate characteristics.
  
### Data Preprocessing:
- Segment time series into patches with defined patch size (e.g., 24 for hourly data, or 48 for 15-min data).
- Normalize or standardize each variable independently or jointly (based on dataset).
- For synthetic experiments (e.g., the one with shocks and periodicity), generate signals as described:
  - Random walk with additive periodic shocks (phase difference of 4).
  - Use the specific parameters (\( \tau=24, S=8, k=5 \)).
- Apply optional masking or dropout perturbations during training for robustness evaluation.

### Data Splits:
- Use standard training/validation/test splits provided by the datasets or as per the paper:
  - For M4 datasets, use hierarchical/standard temporal split (e.g., last month for testing).
  - For synthetic datasets, use the designed sequence length splits.
- Ensure consistent random seed for reproducibility.

---

## 4. Experimental Setup & Hyperparameters (as given)
- **Training epochs:** e.g., 30 for some datasets, 100 for others.
- **Batch sizes:** e.g., 32 or 128, based on hardware constraints.
- **Learning rates:** 
  - Starting from \( 1 \times 10^{-3} \) scaled by number of layers or as per dataset.
  - Use Adam optimizer or an adaptive variant.
- **Dropout & Masking probability \( p \):** 
  - Vary \( p \) between 0.1 and 0.7 in linear increments during experiments.
  - Dropout rate: e.g., \( 10^{-3} \) as per table.
  
### Training procedure:
- Optimize the model to minimize the Mean Squared Error (MSE) or MAE on validation set.
- Use early stopping based on validation performance.
- Implement cyclical learning rate or fixed schedule as suitable.
- Consistently apply parameter sharing across horizons and layers.

### Masking & Regularization:
- Apply the query-adaptive masking strategy for robustness.
- Mask based on horizon-dependent probability \( p \).
- Incorporate masking during training and evaluate with/without masking at test.

---

## 5. Evaluation Metrics & Benchmarking
- Primary metrics:
  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)**
  - **Overall error (e.g., average over horizons)**
- Evaluation for each dataset: 
  - For synthetic data, compare forecasted vs ground-truth signal.
  - For real datasets, use standard metrics as per dataset (e.g., MSE, MAE).
- Comparison benchmarks:
  - Existing models (PatchTST, TimeMixer, DLinear, etc.) as per Tables 4–6.
  - Variants of your model with/without self-attention, parameter sharing, masking.
- Hyperparameter tuning based on validation performance across the hyperparameter grid (layers, heads, embedding size, masking \( p \), etc.).

---

## 6. Reproducibility & Additional Considerations
- Maintain code and experiments under fixed seeds.
- Document the code structure:
  - Encoder: embedding layer + cross-attention modules (no self-attention).
  - Query embeddings: learn horizon-dependent, trainable parameters.
  - Patch splitting: uniform patch size.
  - Masking: probabilistic, horizon-dependent.
- Log training details:
  - Epoch number, loss curves, hyperparameters.
  - Resource usage (GPU memory, runtime).
  - Attention score maps for interpretability analysis.
- Implement tests/ablation studies:
  - Removing masking, varying mask probability.
  - Replacing cross-attention with self-attention (like in the ablation study).
  - Varying patch size and number of layers.
  - Multi-horizon vs. single-horizon forecast.

---

## 7. Summary & Execution Roadmap
- **Step 1:** Data acquisition, preprocessing, and synthetic data generation (if needed).
- **Step 2:** Model implementation:
  - Define patch embedding modules.
  - Build the cross-attention-only transformer blocks.
  - Integrate horizon-specific query parameters and shared attention weights.
  - Incorporate query-adaptive masking.
- **Step 3:** Training setup:
  - Hyperparameter grid search (layers, heads, \( p \), learning rate).
  - Training/validation splits.
  - Logging and checkpointing.
- **Step 4:** Evaluation:
  - Validate on held-out sets.
  - Generate forecasting results.
  - Visualize attention patterns.
- **Step 5:** Compare with baselines and conduct ablation studies as per the paper.

---

This roadmap captures each component needed to faithfully reproduce the core contribution, experimental results, and interpretability analyses presented in the paper. Each step is critical and should be executed with precision and thorough documentation for successful replication.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a cross-attention-only time series transformer (CATS) model that segments input sequences into patches, embeds them, and applies a stack of cross-attention layers with horizon-dependent queries. The model will share parameters across horizons for efficiency. Masking techniques will be incorporated to improve robustness. The system will use PyTorch for model implementation, data loading, training, and evaluation, with modular scripts for clarity and flexibility.",
    "File list": [
        "main.py",  
        "dataset_loader.py", 
        "model.py",  
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "config.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_names: List[str], config: dict)\n        +load_data() -> dict\n        +get_train_test_split() -> Tuple[Tensor, Tensor]\n    }\n    class PatchEmbedder {\n        +__init__(patch_size: int, embed_dim: int)\n        +embed(sequence: Tensor) -> Tensor\n    }\n    class CrossAttentionBlock {\n        +__init__(num_heads: int, embed_dim: int)\n        +forward(input_patches: Tensor, memory_patches: Tensor, query_embeddings: Tensor, mask: Optional[Tensor]) -> Tensor\n    }\n    class CATSModel {\n        +__init__(params: dict)\n        +train_on_batch(input_sequence: Tensor, target_sequence: Tensor, mask: Optional[Tensor]) -> Tensor\n        +predict(input_sequence: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: CATSModel, dataset: dict, config: dict)\n        +train() -> None\n        +save_checkpoint(path: str) -> None\n    }\n    class Evaluator {\n        +__init__(model: CATSModel, dataset: dict, config: dict)\n        +evaluate() -> dict\n    }\n    class MaskingStrategy {\n        +__init__(mask_prob: float)\n        +apply_mask(target: Tensor) -> Tuple[Tensor, Tensor]\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluator\n    Trainer --> CATSModel\n    CATSModel --> CrossAttentionBlock\n    CrossAttentionBlock --> PatchEmbedder\n    MaskingStrategy --> Masking Implementation\n    DatasetLoader --> Dataset return data/load functions\n    Main --> Config\n    Config --> Hyperparameters and dataset paths",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as CATSModel\n    participant TR as Trainer\n    participant EV as Evaluator\n    participant US as utils.py\n    M->>DL: load_data()\n    DL-->>M: datasets {train, test}\n    M->>MD: initialize with params from config\n    Note right of MD: includes patch size, num layers, heads, horizon, mask probability\n    M->>TR: prepare training\n    TR->>US: create horizon-dependent query embeddings\n    TR->>MD: train_on_batch(input_sequence, target_sequence, mask)\n    MD-->>TR: output predictions\n    TR->>Main: training complete, save model\n    Main->>EV: evaluate on test set\n    EV->>MD: predict(test_input)\n    MD-->>EV: predictions\n    EV-->>Main: report metrics (MSE, MAE)\n    Main->>US: visualize attention maps and predictions\n    Note over US: loop for hyperparameter tuning, mask probability variation, ablation studies\n",
    "Anything UNCLEAR": "Need clarification on specific dataset formats, exact hyperparameter ranges for extensive grid search, and whether synthetic data experiments (with shocks and periodicity) are included in final code or optional."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.9.0",
        "numpy==1.21.0",
        "scipy==1.7.0",
        "matplotlib==3.4.3",
        "tqdm==4.62.0",
        "pandas==1.3.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class responsible for loading datasets, applying normalization, segmenting time series into patches, and splitting into training, validation, and test sets. It interfaces with dataset files, applies normalization, and outputs tensors ready for training."
        ],
        [
            "model.py",
            "Defines the core CATS model: embedding layers for input patches, horizon-dependent query embeddings, multiple cross-attention blocks (with shared parameters across horizons), and output projection layers. Includes classes for PatchEmbedder, CrossAttentionBlock, and the full model."
        ],
        [
            "trainer.py",
            "Implements the Trainer class: orchestrates training loop, batch data feeding, masking strategy application, optimizer step, learning rate schedule, early stopping, and checkpoint saving. Calls the model's forward method with masked inputs and computes loss."
        ],
        [
            "evaluation.py",
            "Provides Evaluation class: loads trained model, runs inference on test data, computes metrics (MSE, MAE), visualizes attention maps, and forecasts for multiple horizons. Handles batch evaluation and visualization of scores and attention heatmaps."
        ],
        [
            "utils.py",
            "Includes utility functions: positional encoding, masking functions, visualization helpers, and common tensor operations for consistency and code reuse."
        ],
        [
            "main.py",
            "The entry script: parses configs, initializes dataset loader, model, trainer, and evaluator. Manages training/evaluation cycles, hyperparameters, and logs into console or files. Calls functions sequentially based on dependency order."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "utils.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Shared functions for masking, positional encoding, and visualization are implemented in utils.py. The model code with parameter sharing and horizon-dependent queries is heavily based on the described architecture, requiring careful integration of query embeddings, patches, and attention sharing mechanisms.",
    "Anything UNCLEAR": "Clarification needed on dataset download locations, specific hyperparameter ranges for tuning, and whether synthetic datasets (like with shocks and periodic signals) are optional or core experiments."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
# Training parameters based on the paper's experimental setup
training:
  learning_rate: 1e-3             # Initial learning rate; commonly used default, can be tuned
  batch_size: 32                  # Typical batch size for training, as per resource constraints
  epochs: 30                      # Number of training epochs; standard starting point
  dropout_rate: 0.1               # Dropout probability for regularization
  mask_probability: 0.2           # Probability for query-adaptive masking during training
  patience: 10                    # Validation patience for early stopping
  optimizer: Adam                 # Optimizer used; Adam is standard
  weight_decay: 1e-4              # Regularization weight decay

# Model hyperparameters
model:
  input_sequence_length: 96       # Input sequence length; can vary as per dataset or experiment
  forecast_horizon: 72            # Forecast horizon T; as per dataset and setup
  patch_size: 24                  # Patch length; from the synthetic and real datasets
  num_layers: 3                   # Number of transformer layers with cross-attention
  num_heads: 2                    # Number of attention heads
  embed_dim: 256                  # Embedding dimension for patches and queries
  num_attention_heads: 2          # Number of heads in attention modules
  horizon_embeddings: 1           # Whether to use horizon-dependent query embeddings
  parameter_sharing: true         # Use shared parameters across horizons for efficiency

# Dataset specific parameters (default, can be overridden per dataset)
dataset:
  name: "ETTm1"                    # Dataset name
  data_path: "./datasets/ETTm1"    # Path to dataset
  normalizer: "standard"           # Normalization technique
  train_split: 0.7                 # Fraction for training
  val_split: 0.15                  # Fraction for validation
  test_split: 0.15                 # Fraction for testing

# Miscellaneous parameters
misc:
  num_patches: 4                   # Number of patches per sequence, e.g., 4 for synthetic
  masking_strategy: "query_adaptive" # Strategy for masking; as per paper
  seed: 42                        # Random seed for reproducibility

# Hardware configuration (reference only; actual hardware may vary)
hardware:
  gpus: 4                         # Number of GPUs to use; set to 1 if training single GPU
```

---

**Note:** Full configuration is available in `planning_config.yaml`
