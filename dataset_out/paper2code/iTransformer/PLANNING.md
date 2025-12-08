# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, step-by-step plan to reproduce the methodology and experiments from the paper "iTransformer: Inverted Transformers are Effective for Time Series Forecasting," based solely on the provided content and key details highlighted therein.

---

### 1. **Understanding and Implementing the Core Methodology**

#### A. **Overall Architecture and Conceptual Shifts**

- **Inverted Transformer Framework:**
  - **Key idea:** Instead of applying attention on temporal tokens (standard approach), embed each multivariate series independently into variate tokens.
  - **Main components:**
    - Series of *variates* (channels) are encoded as independent tokens.
    - Attention operates **over variate tokens** to model **multivariate correlations**.
    - Series of each variate are encoded **independently** into tokens.
    - Feed-forward networks (FFNs) are applied **per variate token** to learn nonlinear representations.

- **Component reinterpretation (without modifying basic modules):**
  - **Attention** on *variates* instead of *time points*.
  - **Feed-forward networks** process each variate token **independently**.
  - **Layer normalization** is applied **per variate series** to normalize the series tokens for robustness.

- **Key to implementation:** 
  - **Inverse attention dimension:** The attention mechanism operates **over variate tokens** (distinct series), capturing **multivariate correlations**.
  - **Embed** entire series of each variate (channel) independently as **variates tokens**.
  - **Stack multiple layers** (Transformer blocks) with standard attention, FFN, layer normalization.

#### B. **Component Details per the Paper**
  
- **Embedding:** 
  - Use a multi-layer perceptron (MLP) to embed raw series data **per variate**:
    \[
    \mathbf{h}_n^0 = \text{MLP}(\mathbf{X}_{:, n})
    \]
    where \(\mathbf{X}\) is of shape \(T \times N\) (time steps \(\times\) variates).
  - The number of variates \(N\), and embedding dimension \(D\), are set by experiment.

- **Transformer Stack:**
  - Multiple layers (say \(L\)), where each contains:
    - **LayerNorm** applied **per variate series tokens**;
    - **Scaled dot-product self-attention** operates **over variate tokens** (sequence length = number of variates);
    - **Feed-forward network (FFN):** MLP applied **per variate token**.
  - **Attention modules:**
    - Create query, key, value \(\mathbf{Q, K, V}\) from the variate tokens.
    - Compute attention scores \(\mathbf{A}\) over variate tokens to learn correlations.
  - **Position embedding:** Not used explicitly since sequence order is stored inherently in series tokens (per the paper note).
  - **LayerNorm normalization** applies **on each variate series**; normalize the series data before attention and FFN at each layer.

- **Output decoding:**
  - After \(L\) layers, apply an **MLP projection** to the final variate tokens to produce the forecasted series:
    \[
    \hat{\mathbf{Y}}_{:, n} = \text{MLP}(\mathbf{h}_n^L)
    \]
  - \(\hat{\mathbf{Y}}\) shape: \(N \times S\), where \(S\) is the forecast horizon.

#### C. **Model Hyperparameters & Architecture Details**
- **Number of transformer layers \(L\)**.
- **Embedding dimension \(D\)**.
- **Number of attention heads**: Typically, a standard value (e.g., 4 or 8).
- **Attention dimension \(d_k\)**: Usually set to \(D / \text{num_heads}\).
- **MLP dimensions**: E.g., 4x embed dimension.
- **Normalization:** LayerNorm on each variate series token.
- **Attention type:** Scaled dot-product attention with optional efficient attention mechanisms (the paper suggests plug-ins for large variates).
- **Training strategy:** Feed entire series, normalize variate series independently, no explicit position embedding.
- **Inversion process:** Embed variates independently, operate attention **on variate tokens**, learn multivariate correlations.

---

### 2. **Experimental Setup**

#### A. **Datasets and Data Preprocessing**

- **Datasets:**
  - ETT (ETTh1, ETTh2), with 7 variates, hourly data.
  - Exchange, Weather, Solar-Energy, Traffic, PEMS: multivariate with variates ranging approximately 12-862.
  - Market datasets with multiple subsets (varies from 12 to 1100+ variates).
  - **Input length \(T = 96\) time steps;** forecasts of lengths \(S \in \{12, 24, 36, 48\}\).
  - **Normalization:** Normalize each variate Series independently to Gaussian (zero mean, unit variance).

- **Data preparation:**
  - For each series:
    - Extract training/validation/test splits as per dataset.
    - Normalize variates separately.
    - Segment into input sequences of length \(T\), target sequences of length \(S\).
  - For training, **sample batches** consisting of multiple such sequences.

#### B. **Hyperparameters & Model Configurations**
- **Input lookback window \(T = 96\)**.
- **Forecast lengths \(S\)** as per dataset.
- **Variate embedding dimension \(D\)** (e.g., 64 or 128).
- **Number of transformer layers \(L\)** (e.g., 4-8).
- **Attention:**
  - Multi-head (e.g., 4 or 8 heads).
  - Attention dimension \(d_k = D / \text{num_heads}\).
- **MLP for embedding & projection:**
  - Use 2-3 layers, e.g., [D, 2D, D].
- **Normalization:** LayerNorm before attention and FFN in each block, over variate series.
- **Training details:**
  - Optimizer: AdamW or Adam.
  - Learning rate schedule: Cosine decay or step decay.
  - Batch size: 32-128 depending on GPU memory.
  - Loss: MSE (or MAE as secondary metric).

---

### 3. **Training Strategy & Important Details**

- **Training:**
  - Sample batches with multiple sequences.
  - For each sequence:
    - Embed each variate independently using the initial MLP.
    - Pass through \(L\) stacked inverted transformer layers.
    - Final variate tokens projected via MLP to produce forecasts.
  - Loss: Compute MSE with ground truth series.

- **Data augmentation & robustness:**
  - Apply independent variate normalization.
  - Use random seed variability to test robustness.

- **Efficiency considerations:**
  - For large variate numbers, consider plug-in efficient attention mechanisms.
  - Train on arbitrary variate subsets: randomly sample \(20\%\) of variates during training and inference to evaluate generalization.

---

### 4. **Evaluation Metrics**

- **Forecasting accuracy:**
  - MSE and MAE on test sequences.
  - Average over multiple horizons.
- **Model robustness:**
  - Variations over multiple random seeds.
  - Performance with varying variate subset sizes (e.g., 20%, 50%, 100%).

---

### 5. **Additional Implementation Notes & Details**

- **Code structure:**
  - Data loader:
    - Efficiently read datasets, perform normalization, create input-target pairs.
  - Model:
    - Embedding layer: MLP per variate.
    - Transformer encoder stacks with attention on variates.
    - Final MLP projection.
  - Training loop:
    - Batch sampling.
    - Loss computation.
    - Validation.
    - Checkpoints.
- **Visualization & analysis:**
  - Extract and visualize attention maps (correlation matrices) from the variate attention.
  - Visualize forecasted series vs ground truth.
  - Visualize learned correlation matrices to interpret multivariate correlations.

---

### 6. **Remaining Unclear or Informational Gaps to Clarify/Confirm**

- Exact hyperparameter ranges (embedding size, number of layers, attention heads) per dataset.
- Whether the attention mechanism incorporates any efficiency plug-ins for large variate sets.
- Precise normalization procedures (mean-variance across sequence or over variate features).
- Whether positional embeddings are completely omitted or experimented with in some variants.

---

**Summary:**

This plan highlights the core architectural design (embed each variate independently, operate attention over variate tokens, process series via FFN per variate, stack multiple layers) aligned with the paper’s “inverted” concept. It details data preprocessing, hyperparameters, training methodology, evaluation metrics, and analysis techniques. Once the above is validated or refined with further details, it will direct a precise implementation aligned with the original paper's approach and experiments.

---

Would you like me to proceed to draft a detailed code template based on this roadmap?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a concise, modular system in Python leveraging open-source libraries like PyTorch for deep learning, Pandas and NumPy for data handling, and Datasets for data loading. The core idea is to implement an 'Inverted Transformer' architecture that embeds each variate independently, applies self-attention over variate tokens, and processes series representations via FFNs. The system will include a dataset loader that performs normalization and generates input-target pairs, a model class implementing the inverted transformer stack with configurable hyperparameters, a trainer class managing training loops, and an evaluator class for metrics computation. The design emphasizes simplicity: data flows from loader to model, training involves iterative updates, and evaluation calculates MSE/MAE, with visualization options for attention maps.",
    "File list": [
        "main.py",
        "dataset_loader.py",
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
        +__init__(data_paths: dict, seq_len: int, pred_len: int)
        +load_data() -> Dataset
        +get_train_batches(batch_size: int) -> Iterator[Tuple[Tensor, Tensor]]
        +get_val_batches(batch_size: int) -> Iterator[Tuple[Tensor, Tensor]]
        +get_test_batches(batch_size: int) -> Iterator[Tuple[Tensor, Tensor]]
    }
    class InvertedTransformer {
        +__init__(hyperparams: dict)
        +forward(series_input: Tensor) -> Tensor
        +predict(series_input: Tensor, forecast_length: int) -> Tensor
    }
    class Trainer {
        +__init__(model: InvertedTransformer, optimizer: Optimizer, loss_fn: callable, device: str)
        +train_epoch(data_loader: DatasetLoader, batch_size: int) -> float
        +train(data_loader: DatasetLoader, epochs: int) -> None
    }
    class Evaluator {
        +__init__(model: InvertedTransformer, data_loader: DatasetLoader, device: str)
        +evaluate(batch_size: int) -> dict
        +visualize_attention_layers(layer_outputs: list) -> None
    }
    class VisualizationTools {
        +plot_series(series: np.ndarray, predictions: np.ndarray, title: str) -> None
        +plot_attention_matrix(matrix: np.ndarray, title: str) -> None
    }
    Main --> DatasetLoader
    Main --> InvertedTransformer
    Main --> Trainer
    Main --> Evaluator
    Trainer --> InvertedTransformer
    Evaluator --> InvertedTransformer
    VisualizationTools --> Evaluator
    "
    ,
    "Program call flow": "
sequenceDiagram
    participant C as Main
    participant DL as DatasetLoader
    participant M as InvertedTransformer
    participant TR as Trainer
    participant EV as Evaluator
    participant VT as VisualizationTools
    C->>DL: Initialize with data paths, seq_len, pred_len
    DL-->>C: load_data()
    C->>M: instantiate with hyperparameters
    C->>TR: train(model, optimizer, loss_fn, device)
    TR->>DL: get_train_batches(batch_size)
    TR->>M: forward(series_input)
    M->>TR: predictions
    TR-->>C: trained model
    C->>EV: evaluate model on validation/test
    EV->>M: forward(series_input)
    M-->>EV: metrics
    EV->>VT: visualize attention at different layers
    "
    ,
    "Anything UNCLEAR": "Clarify exact dataset formats and normalization procedures; clarify whether to include positional embeddings or not; specify if efficient attention plug-ins are required for large variate sets; confirm hyperparameter ranges and validation procedures."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "pandas==1.3.0",
        "matplotlib==3.4.3",
        "scikit-learn==0.24.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DataLoader class that loads raw datasets, normalizes each variate independently, slices sequences into input-target pairs, and yields batches. It handles different dataset formats and implements methods: __init__(), load_data(), get_train_batches(), get_val_batches(), get_test_batches(). Dependency: depends on datasets stored locally or accessible via provided paths."
        ],
        [
            "model.py",
            "Defines the InvertedTransformer class, including embedding modules, multiple stacked attention+FFN transformer blocks, layer normalization per variate, and projection layers. Implements __init__(), forward(), and predict() methods. Relies on helper modules: MultiHeadAttention, PositionwiseFFN, LayerNorm."
        ],
        [
            "trainer.py",
            "Implements Trainer class that manages training loop: initialization (__init__), train_epoch(), train(), with loss functions, optimizer, gradient clipping if necessary. Calls model.forward() on batches, computes loss, backpropagates, updates parameters."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class that loads the trained model, runs inference on validation/test datasets, computes metrics (MSE, MAE), and visualizes attention matrices and example forecasts. Includes evaluate() and visualization methods."
        ],
        [
            "main.py",
            "Main script initializing dataset loader with dataset paths, hyperparameters, instantiating model, trainer, and evaluator. Handles argument parsing, configuration setup, orchestrates training and evaluation over epochs, saves models, generates plots."
        ],
        [
            "utils.py",
            "Utility functions for normalization, plotting series and attention matrices, saving/loading models, and configuration handling. Shared functions used in dataset_loader.py, evaluation.py."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "model.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n\n# No API needed for core training code; optional endpoints if deploying inference server could include:\n# POST /predict with input sequence and variate subset, returning forecasted series.\n# GET /metrics?dataset=validation/testing\n",
    "Shared Knowledge": "Common utility functions for normalization, visualization, and model checkpointing are shared between utils.py, evaluation.py, main.py, and dataset_loader.py. The normalization applies variate-wise scaling to Gaussian distribution. Model parameters and hyperparameters are centrally managed in a config dictionary or YAML file loaded in main.py.",
    "Anything UNCLEAR": "Clarify dataset source formats (e.g., CSV, HDF5, custom) and exact normalization details. Confirm whether positional encodings should be incorporated. Clarify if large variate datasets require efficient attention modules, and specify available hardware resources for scaling experiments."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: "ETTh1"  # Example dataset, replace as needed
  data_paths:
    train: "path/to/train.csv"
    val: "path/to/val.csv"
    test: "path/to/test.csv"
  variate_normalization: true
  sequence_length: 96
  forecast_length: 96

model:
  embedding_dim: 128
  num_layers: 4
  num_heads: 4
  dropout_rate: 0.1
  feedforward_dim: 512

training:
  learning_rate: 0.001  # Typical starting value; confirm with hyperparameter tuning
  batch_size: 64
  epochs: 50
  optimizer: "adamw"
  weight_decay: 0.01

evaluation:
  metrics: ["MSE", "MAE"]
  evaluation_interval: 1  # Evaluate after each epoch

saving:
  save_dir: "checkpoints/"
  save_frequency: 10  # Save model every 10 epochs

logging:
  log_dir: "logs/"
  log_frequency: 10

visualization:
  attention_maps: true
  forecast_plots: true
```

---

**Note:** Full configuration is available in `planning_config.yaml`
