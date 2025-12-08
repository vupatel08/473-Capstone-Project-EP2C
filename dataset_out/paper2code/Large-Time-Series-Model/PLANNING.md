# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, structured plan to reproduce the methodology, experiments, and evaluation of the "Timer: Generative Pre-trained Transformers Are Large Time Series Models" paper. This plan draws directly from the paper’s content, emphasizing key details, hyperparameters, dataset preparations, and experiment configurations.

---

## 1. Pre-Training Large Time Series Transformer ("Timer")

### 1.1 Overview
- **Objective**: Pre-train a GPT-style Transformer on large-scale, heterogeneous time series datasets using next-token prediction.
- **Core idea**: Transform multivariate heterogeneous time series into a unified "single-series sequence" (S3) format via hierarchical, class-specific tokenization, then perform autoregressive language modeling.

### 1.2 Dataset Construction for Pretraining
- **Datasets**: Curate large heterogeneous time series datasets totaling up to 1 billion data points (examples: UTSD-12G, LOT, LTS**, etc.).
- **Hierarchical grouping**: Organize datasets into hierarchies based on domain complexity, source typology, and data variability.
- For each dataset:
  - **Samples**: Use the entire dataset (excluding test sets) for pretraining.
  - **Tokenization**:
    - Segment multivariate series into tokens:
      - **Single-series tokens**: Consecutive points of length `$S=96$` (or specified, e.g., 96).
      - **Hierarchical tokens** (depending on dataset): e.g., 672 to 1440 points for large datasets.
    - Convert each segment into a token: a vector of length `$S$` with continuous values.

- **Handling heterogeneity**:
  - **Unified format**: Concatenate all series, preserving hierarchical patterns, into a "unified single-series sequence" (S3).
  - **Diverse variate types and durations**:
    - Use adaptive resizing, and incorporate dimensional/feature embeddings.
    - Include a timestamp or auxiliary embedding if timestamps are irregular or multivariate.

### 1.3 Model Architecture
- **Transformer**:
  - Use a **decoder-only GPT-like architecture**.
  - **Hyperparameters**:
    - Model sizes: e.g., 29M, 50M, 91M, 311M, 385M, 710M, 768M, 1024M parameters.
    - Number of layers (`L`): e.g., 6 layers for smaller models, up to 8 layers for large ones.
    - Hidden size (`D`): e.g., from 256 to 1024.
    - Attention heads: fixed at 8.
    - Embedding dimension: equal to `D`.
    - Feedforward dimension: 2x `D`.
  - **Positional & Timestamp Embeddings**:
    - Learn positional embeddings for tokens.
    - Optionally, learn embeddings for timestamps or series-specific context.

### 1.4 Pretraining Objective
- **Task**: Next-token autoregressive prediction fitting the GPT-style causal language modeling.
- **Loss**: Standard cross-entropy loss (`L2`-equivalent to `MSE` on decoder outputs, as per the paper).
- **Training Details**:
  - Optimizer: AdamW or similar.
  - Learning rate schedule:
    - Initial LR: `3e-5` (or dataset-dependent).
    - Warm-up: e.g., first few thousand steps.
    - Decay: exponential decay with base 0.5.
  - Batch size: e.g., 2048 tokens.
  - Epochs: 10 for datasets of similar size; scale up for larger datasets.
  - Hardware: Multiple GPUs/TPUs with distributed training.

### 1.5 Model Scaling & Checkpointing
- Train multiple scaled models (e.g., 29M, 50M, 91M, up to 700M+).
- Save checkpoints periodically.
- Log training loss and validation loss (using held-out data in the pretraining datasets).

---

## 2. Fine-Tuning for Downstream Tasks

### 2.1 Tasks & Datasets
- **Forecasting**: e.g., ETTh1, ETTh2, ECL, Traffic, PEMS datasets.
- **Imputation**: missing value completion, e.g., PEMS and Energy datasets, with variable missing ratios.
- **Anomaly Detection**: e.g., UCAR Anomaly Archive, ECG, etc.
- **Additional Tasks**: classification, regression.

### 2.2 Data Preparation
- For each downstream dataset:
  - Convert multivariate time series into S3 format (using same segmentation as pretraining but possibly with task-specific adjustments).
  - For imputation, mask some points at various mask ratios (e.g., 12.5%, 25%, 37.5%, 50%)—simulate missingness.
  - For forecasting, select input sequence length (lookback/histogram) and forecast length (e.g., 96 for 1- or 5-step ahead).

### 2.3 Model Adaptation & Fine-tuning
- **Initialization**: Load pre-trained Timer checkpoint.
- **Architectural adjustments**:
  - For forecasting: append task-specific linear head (predicting numerical values).
  - For classification/detection: final layer (classification head, sigmoid, or binary decision).
- **Hyperparameters**:
  - Use the same architecture scaled for the task (e.g., 6 layers, 256 D, 8 heads).
  - Fine-tuning epochs: 10-20.
  - Learning rate: smaller LR (e.g., 1e-5 to 3e-5).
  - Batch size: as per GPU memory constraints—e.g., 512.

### 2.4 Training procedure
- Use sequence-to-sequence or autoregressive generation, depending on task.
- Use task-relevant loss:
  - Forecast: MSE (mean squared error).
  - Imputation: MSE + mask tokens.
  - Anomaly detection: classification or likelihood scores.
- Incorporate early stopping based on validation metrics.

---

## 3. Evaluation Metrics & Protocols

### 3.1 Forecasting
- **Metrics**: MSE, MAE.
- **Setup**: Evaluate on held-out test split.
- **Scalability**: Test with both from-scratch and pre-trained initialization.

### 3.2 Imputation
- **Metrics**: MSE, relative improvement over scratch-trained models.
- **Setup**: Mask ratio varies, compare error reduction when using pre-trained models versus scratch.

### 3.3 Anomaly Detection
- **Metrics**: Number of anomalies detected, precision, recall, F1-score at given quantiles.
- **Visual inspection**: overlay predicted anomalies vs ground-truth experts.

### 3.4 Zero-Shot & Transfer Evaluation
- Use pre-trained large models directly (without task-specific training) on unseen datasets.
- Measure error metrics and rank relative to baseline models.

---

## 4. Additional Implementation Details
- **Embeddings**:
  - Use learned position + timestamp embeddings.
  - For heterogeneity: include series-specific embeddings.
- **Tokenization**:
  - Segment series into overlapping windows if needed.
  - Cap maximum context length (`1440` or according to dataset size).
- **Stochasticity & Random seeds**:
  - Set seeds for reproducibility.
  - Use the same random seed for dataset shuffling, sampling, and model initialization.

---

## Summary Roadmap
- **Pre-train** Timer on large heterogenous datasets using hierarchical, class-specific tokenization, with GPT-style autoregressive training.
- **Fine-tune** or **adapt** Timer to downstream tasks with task-specific heads, using small data ratios (5%, 20%, 100%), with mask-based imputation, forecasting, or anomaly detection.
- **Evaluate** using error metrics, zero-shot capability, and qualitative visualizations matching the paper’s figures, to ensure fidelity.

---

This comprehensive plan provides detailed implementation steps, dataset handling, hyperparameter choices, and evaluation procedures aligned exactly with the paper’s methodology and experiments. Once approved, transforming this into code will be straightforward, guided by these specifications.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular system leveraging open-source libraries such as PyTorch for model development, Hugging Face Transformers for scalable Transformer APIs, and custom scripts for dataset processing and tokenization. The system will include components for large-scale dataset loading and hierarchical tokenization, a GPT-style decoder-only Transformer architecture, a training routine with scheduled learning rate decay, and evaluation modules for forecasting, imputation, and anomaly detection. Pretraining datasets will be converted into a unified sequence format, and the model will support hierarchical, class-specific tokenization. Fine-tuning will be task-specific, and zero-shot evaluation on new datasets will be possible by directly applying the pretrained model.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "tokenization.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(dataset_paths: list, config: dict)
        +load_data() -> List[Dict]
        +preprocess() -> list of tokenized sequences
    }
    class Tokenizer {
        +__init__(token_config: dict)
        +tokenize_series(series: List[float]) -> List[Token]
        +convert_to_unified_sequence(series_list: List[List[float]]) -> Sequence
        +hierarchical_tokenization(series: List[float], level: str) -> List[Token]
    }
    class Model {
        +__init__(model_params: dict)
        +forward(input_ids: torch.Tensor) -> torch.Tensor
        +generate(inputs: torch.Tensor, max_length: int) -> torch.Tensor
    }
    class Trainer {
        +__init__(model: Model, dataset: List[Sequence], config: dict)
        +train() -> None
        +save_checkpoint(path: str) -> None
        +load_checkpoint(path: str) -> None
    }
    class Evaluation {
        +__init__(model: Model, dataset: List[Sequence], task: str)
        +forecast() -> Dict
        +impute() -> Dict
        +detect_anomalies() -> Dict
        +compute_metrics() -> Dict
    }
    Main --> DatasetLoader
    Main --> Tokenizer
    Main --> Model
    Main --> Trainer
    Main --> Evaluation
    DatasetLoader --> Tokenizer
    Trainer --> Model
    Evaluation --> Model
    Model --> torch.nn.Module
    Tokenizer --> Token
    class Token {
        +values: List[float]
        +token_id: int
        +timestamp: Optional[float]
    }
    class Sequence {
        +tokens: List[Token]
        +convert_to_ids() -> torch.Tensor
    }
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant TK as Tokenizer
    participant MD as Model
    participant TR as Trainer
    participant EV as Evaluation
    
    M->>DL: initialize(dataset_paths, config)
    DL->>DL: load_data()
    DL-->>M: raw_data
    DL->>TK: preprocess(raw_data)
    TK-->>DL: tokenized_sequences
    DL->>MD: initialize(model_params)
    MD-->>MD: build model architecture
    ALT pretrained
        Main->>MD: load checkpoint
    end
    TR->>MD: train(tokenized_sequences)
    TR-->>Main: training complete
    Main->>EV: evaluate(model, test_data, task_type)
    EV->>MD: forecast() / impute() / detect_anomalies()
    MD-->>EV: predictions
    EV-->>Main: metrics
    Main->>TR: save checkpoint if needed
    "
    ,
    "Anything UNCLEAR": "Clarification needed on the exact dataset input format, including tokenization details and hierarchical segmenting parameters. Also, specify if there are any domain-specific embeddings or additional features required for heterogeneity beyond timing and value embeddings."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.9.0",
        "numpy==1.21.0",
        "pandas==1.3.0",
        "scipy==1.7.0",
        "transformers==4.12.0",
        "tqdm==4.62.0",
        "pyyaml==6.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class responsible for reading datasets, extracting raw data, and converting them to internal data formats. It depends on dataset paths, dataset format, and preprocessing configs. Needs to interact with 'tokenization.py' for sequence tokenization. Load large datasets (e.g., UTSD, PEMS, Energy datasets)."
        ],
        [
            "tokenization.py",
            "Implements Tokenizer class responsible for hierarchically segmenting multivariate time series into tokens, converting series into contiguous sequences, applying hierarchical tokenization levels, and integrating timestamps or positional information. Uses input from DatasetLoader. Output is sequences of token IDs for model ingestion."
        ],
        [
            "model.py",
            "Defines the GPT-style decoder-only Transformer architecture. Receives hyperparameters such as number of layers, hidden size, number of attention heads, and feedforward dimensions. Implements forward pass, generative token prediction, and optional sequence generation."
        ],
        [
            "trainer.py",
            "Implements Trainer class managing training loop, including data batching, optimizer, learning rate schedule (with exponential decay), loss computation (e.g., cross-entropy over token IDs), checkpoint saving/loading, and logging. Depends on model.py for the network, dataset_tokenized sequences, and training configs."
        ],
        [
            "evaluation.py",
            "Provides Evaluation class that performs inference for forecasting, imputation, anomaly detection, and zero-shot evaluation. Uses trained model, task-specific input data, and generates predictions, then calculates metrics like MSE, MAE, or detection metrics. Interacts with 'model.py' for forward passes."
        ],
        [
            "main.py",
            "Main entry point script that initializes dataset loader, tokenizer, model, trainer, and evaluation modules based on configuration. Orchestrates the overall experiment process: data loading, pretraining, fine-tuning, and evaluation. Calls each component in sequence, managing dependencies and file execution order."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "tokenization.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "None needed; system runs offline for pretraining and fine-tuning without external API calls.",
    "Shared Knowledge": "Configuration variables such as dataset paths, hierarchical tokenization levels, hyperparameters for model, training, and evaluation, are shared among main.py, dataset_loader.py, trainer.py, and evaluation.py. Utility functions for metric computations are shared between evaluation.py and main.py, if any.",
    "Anything UNCLEAR": "Clarification needed on the exact dataset input formats, especially hierarchical segmentation parameters, and whether additional embeddings (timing, series category) are required for heterogeneity handling."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 3e-5
  batch_size: 2048
  epochs: 10
  warmup_steps: 1000
  decay_strategy: exponential
  decay_rate: 0.5
model:
  size_m: 50
  num_layers: 6
  hidden_size: 512
  num_heads: 8
  ff_dim_multiplier: 2
  max_position_embeddings: 1024
  dropout_rate: 0.1
  input_token_length: 96
dataset:
  pretraining_datasets:
    - name: UTSD-12G
      path: path/to/UTSD-12G
    - name: LOT
      path: path/to/LOT
  dataset_species_levels:
    - name: SeriesHierarchies
      hierarchy_levels: [small, medium, large]
      segment_lengths: [96, 672, 1440]
  dataset_split_ratio: 0.8  # 80% train, 20% validation
  max_sequence_length: 1440
optimization:
  optimizer: AdamW
  weight_decay: 0.01
  warmup_steps: 1000
  lr_schedule: exponential_decay
evaluation:
  forecast_metrics: [MSE, MAE]
  imputation_metrics: [MSE]
  anomaly_detection_metrics: [precision, recall, F1]
logging:
  save_dir: checkpoints/
  log_interval: 100
  save_interval: 1 epoch
```

---

**Note:** Full configuration is available in `planning_config.yaml`
