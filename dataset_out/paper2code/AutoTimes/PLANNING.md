# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a comprehensive, detailed plan to reproduce the methodology and experiments of the AutoTimes paper, following the structure you requested.

---

## 1. Understanding the Core Methodology of AutoTimes

### #a. Model Architecture & Key Components
- **Foundation**: AutoTimes repurposes large decoder-only language models (LLMs) (e.g., LLaMA, GPT-2, OPT) as autoregressive time series forecasters.
- **Embedding & Tokenization:**
  - *Time Series Segmentation*: Divide each univariate/multivariate time series into non-overlapping segments of length *S*. Each segment represents a token.
  - *Series Embedding*: Use a dedicated segmentation embedding layer (e.g., a small MLP) to embed each segment into a fixed latent space, matching LLM embedding dimensions.
  - *Textual Timestamp Embedding*: Convert timestamps to textual format ("YYYY/MM/DD HH:MM:SS") and embed using the LLM’s embedded representations of tokens in the positional prompt.
  
- **Prompt Construction:**
  - *Time Series Prompts*: The lookback series (context window) plus optional prompt variants (e.g., ahead period, recent series) are embedded and included as input tokens.
  - *Textual Timestamps*: Convert timestamps of segments into textual prompts embedded in the model’s input, providing chronological alignment.
  - *Sequence Formatting*: Concatenate prompt tokens with the segment tokens of the series to form the full input sequence.
  - *In-Context Forecasting*: Extend the context with prompt series beyond the normal lookback window, allowing flexible prediction horizons.

- **Autoregressive Prediction:**
  - Use the frozen LLM to predict the next tokens (future segment embeddings) iteratively, conditioned on previous tokens.
  - *Prediction Target*: Next segment tokens of length *F* (forecast horizon). Generate iteratively, feeding back predicted tokens as input for subsequent steps.

- **Prediction & Output:**
  - Project predicted tokens back to series space via a Segment Projection layer (MLP).
  - Reconstruct the predicted series by concatenating the predicted segments.

### #b. Explicit Timeline & Positional Embedding
- Textual timestamp embedding is used as position embedding to better encode chronological info.
- The timestamp embedding (precomputed) is concatenated with segmentation embedding as the input token embeddings.
- The model is trained to predict tokens conditioned on these embeddings, adhering to autoregressive behavior.

### #c. Training & Loss
- **Objective**: Minimize mean squared error (MSE) of predicted segments against ground truth.
- **Segmentation**: For training, series are truncated to ensure the prompt (lookback + lookahead) does not overlap, avoids data leakage.
- **Multi-step Forecasting**: Generate multiple future segments iteratively, updating inputs accordingly.

---

## 2. Experimental Setup & Datasets

### #a. Datasets & Data Preparation
- **Datasets Required:**
  - Long-term: ETTh1, Weather, Traffic, Solar-Energy, ECL (multiple granularities)
  - Short-term: M4 competition datasets (e.g., M3, Hourly, Daily)
- **Preprocessing Steps:**
  - Normalize each series (e.g., min-max scaling, z-score) as per the original setting.
  - Segmentation into non-overlapping segments of length *S* (e.g., 96, 192, 336, 720 time steps).
  - Convert timestamps to textual prompts, for each segment, based on the series timestamps.
  - Generate textual prompts covering the whole series or specific lookback periods, depending on prompt variant.

### #b. Dataset Formatting
- For each series:
  - Extract lookback window (context window of length *L*, e.g., 672) for training.
  - Generate prompt series (if in-context learning): select initial segments or metadata as prompt.
  - For test, generate series starting from the lookback length, extending prediction horizon *F* (e.g., 96, 192, 336).

### #c. Data Storage & Batching
- Store embedded segments and textual prompt tokens, aligned by timestamps.
- Organize data into training, validation, and test splits, respecting chronological order (no data leakage).

---

## 3. Hyperparameters & Model Details

### #a. Model & Embeddings
- **Base LLM**: Use available pre-trained decoder-only models (e.g., LLaMA-7B, GPT-2, OPT). 
- **Embedding dimension**: Match the LLM’s internal token embedding size (~768, 1024, 2048, depending on choice).
- **Segmentation Embedding (MLP)**:
  - Input: raw segment data of size *S*.
  - Output: embedding vector matching LLM’s embedding size.
- **Timestamp Embeddings**:
  - Represent timestamps textually ("YYYY/MM/DD HH:MM:SS").
  - Embed using the embedding layer of the LLM (text tokens).

### #b. Hyperparameters
- **Segmentation size (S)**: e.g., 96, 192, 336, 720 based on dataset granularity.
- **Lookback length (L)**: set to model input window (e.g., 672).
- **Forecast horizon (F)**: multiple (e.g., 96, 192, 336).
- **Batch size**: as in experiments (e.g., 224).
- **Learning rate**: start with small (e.g., 5e-5 to 1e-4); use Adam optimizer.
- **Training epochs**: sufficient for convergence (~50-100, depending on dataset size).
- **Gradient clipping**: to stabilize training (e.g., 1.0).
- **Prompt length**: extended prompt tokens (e.g., first 48 hours, last 48 hours, or custom).

### #c. Training Procedure
- Freeze the LLM backbone (except embedding/project layers).
- Optimize only segmentation embedding layer + prompt-related parameters.
- Use supervised MSE loss between predicted and ground truth segments.

---

## 4. Training & Inference Protocols

### #a. Model Training
- Input: sequence of embedded prompt + segmented series tokens (lookback window).
- Objective: predict next *F* segments.
- Auto-regressive: generate tokens iteratively, feeding predicted tokens as input for next step.
- Save model checkpoints, especially the embedding + projection layers.

### #b. Prediction & Multi-step Forecasting
- Initiate from last lookback window.
- Generate predicted segments step-by-step.
- For multi-horizon prediction, iterate until total horizon *F* is covered.
- During inference, extend prompt with additional prompt series if in-context approach.

### #c. Prompt Variants & Testing
- Use textual prompt variants (e.g., prompt from first *F* lookback, last *F*, recent series, random).
- Evaluate prediction accuracy (e.g., SMAPE, MSE, MAE).
- Test on multiple datasets and forecast horizons.

---

## 5. Evaluation Metrics & Analysis

### #a. Metrics
- **Primary**: SMAPE, MSE, MAE.
- **Zero-shot & In-context**: relative improvements over baseline.
- **Efficiency**: inference and training speed (seconds per epoch, GPU memory).

### #b. Baselines & Comparisons
- Reimplement or use official implementations (if available) for baselines (TimeLLM, UniTime, FPT, TimesNet, etc.).
- Compare like-for-like: same datasets, input lengths, forecast horizons.

### #c. Ablations & Hyperparameter Sensitivity
- Vary segmentation size *S*, prompt length, lookback *L*, forecast horizon *F*.
- Evaluate robustness with different model sizes (GPT-2, OPT, LLaMA).
- Visualize predictions, attention/activation maps, prompt influence.

---

## 6. Special Considerations & Potential Issues
- **Data Leakage Prevention**: truncate series to ensure prompts do not contain future info.
- **Prompt Design**: textual prompts should be natural and consistent; test variants.
- **Model Scaling**: larger models (e.g., LLaMA-13B) may improve but require significant resources.
- **Prompt Retrieval Strategy**: experiment with periodicity-based prompts, random prompts, or recent series.
- **Reproducibility**: fix random seeds, document hyperparameter choices, and report standard deviations.

---

## **Summary Roadmap**
1. Prepare datasets with segmentation, normalization, timestamp to textual prompt conversion.
2. Implement segmentation embedding layer + tokenization for series segments and textual timestamps.
3. Load pre-trained decoder-only models; freeze backbone, only train embedding + project layers.
4. Construct input sequences with prompt series (various strategies), including textual timestamp embeddings.
5. Train autoregressive model to predict future segments iteratively.
6. Evaluate on multiple datasets and forecast horizons; compare predictions against ground truth.
7. Research hyperparameter sensitivity, prompt variants, and model size effects.
8. Document all settings, hyperparameters, and code for reproducibility.

---

Let me know if you'd like me to further elaborate on specific implementation steps, hyperparameter choices, or details for particular datasets!

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Develop a modular, PyTorch-based framework that loads time series datasets, constructs tokenized segment embeddings and textual timestamp embeddings, loads a pre-trained decoder-only language model (e.g., LLaMA, GPT-2), freezes the backbone, and trains only the segmentation embedding layer and projection head. The system constructs prompts using lookback series plus textual time prompts, embeds them, and performs autoregressive prediction of future segments. Inference involves iterative token generation conditioned on past predictions, converting tokens back into series for evaluation. The overall pipeline includes data loading, model initialization, prompt building, training loop, and evaluation, with hyperparameters configurable via a config file.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "prompt_builder.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run(): None
    }
    class DatasetLoader {
        +__init__(dataset_paths: dict)
        +load_data() -> list[TimeSeriesSample]
    }
    class TimeSeriesSample {
        +series: np.ndarray
        +timestamps: np.ndarray
        +series_name: str
        +series_metadata: dict
    }
    class Model {
        +__init__(pretrained_model_path: str)
        +embed_segments(segments: torch.Tensor) -> torch.Tensor
        +embed_timestamps(timestamps: list[str]) -> torch.Tensor
        +predict_next_tokens(inputs: torch.Tensor) -> torch.Tensor
        +decode_tokens(tokens: torch.Tensor) -> torch.Tensor
    }
    class PromptBuilder {
        +build_prompt(series: np.ndarray, timestamps: np.ndarray, prompt_strategy: str, prompt_length: int) -> list[str]
        +convert_series_to_tokens(series: np.ndarray, timestamps: np.ndarray) -> torch.Tensor
        +convert_texts_to_embeddings(texts: list[str]) -> torch.Tensor
    }
    class Trainer {
        +__init__(model: Model, dataset: list[TimeSeriesSample], prompt_builder: PromptBuilder, config: dict)
        +train() -> None
        +save_checkpoint(path: str) -> None
    }
    class Evaluation {
        +__init__(model: Model, dataset: list[TimeSeriesSample], prompt_builder: PromptBuilder, config: dict)
        +evaluate(metrics_list: list[str]) -> dict
        +predict_and_convert(series: np.ndarray, timestamps: np.ndarray, horizon: int) -> np.ndarray
    }
    class Hyperparameters {
        +lookback_length: int
        +forecast_horizon: int
        +segment_size: int
        +prompt_strategy: str
        +training_epochs: int
        +batch_size: int
        +learning_rate: float
        +text_prompt_length: int
    }
    Main --> DatasetLoader
    Main --> Model
    Main --> PromptBuilder
    Main --> Trainer
    Main --> Evaluation
    Trainer --> Model
    Evaluation --> Model
    Trainer --> DatasetLoader
    Evaluation --> DatasetLoader
    PromptBuilder --> TimeSeriesSample
    Model <|-- PretrainedLanguageModel
    class PretrainedLanguageModel {
        +load_model(path: str)
        +freeze_parameters()
        +forward(inputs: torch.Tensor) -> torch.Tensor
    }
    Model o-- PretrainedLanguageModel
    Main: "-- uses YAML config file to initialize all components --"
    DatasetLoader: "-- loads raw series data, timestamps, and metadata --"
    PromptBuilder: "-- constructs prompt prompts as lists of strings, converts series and timestamps to token tensors --"
    Model: "-- segments embedding layer, textual prompt embedding, autoregressive prediction head --"
    Trainer: "-- manages training loop, optimizes only trainable layers --"
    Evaluation: "-- computes metrics, runs iterative prediction --"
    Hyperparameters: "-- consolidates hyperparameter values accessible for all modules --"
    Note: "Explicit relations between dataset samples, model input/output, prompt construction, and training are clearly defined for comprehensive implementation."
    ",
    "Program call flow": "
sequenceDiagram
    participant MC as Main Controller
    participant DL as DatasetLoader
    participant PB as PromptBuilder
    participant MD as Model
    participant TR as Trainer
    participant EV as Evaluation
    MC->>DL: load_data()
    DL-->>MC: dataset list
    MC->>MD: initialize model(pretrained_path)
    MD-->>MC: model object (parameters frozen except trainable layers)
    MC->>PB: build_prompt(series, timestamps, strategy, prompt_length)
    PB-->>MC: prompt_texts
    MC->>TR: train(model, dataset, prompt_builder, hyperparameters)
    TR-->>MC: training logs
    MC->>EV: evaluate(model, dataset, prompt_builder, metrics)
    EV-->>MC: metrics report
    MC->>MC: save_model(checkpoint_path)
    "
    ,
    "Anything UNCLEAR": "Exact dataset formats (e.g., time series files, timestamp formats), prompt construction strategies, and their parameters are not fully specified; hyperparameter ranges for training (epochs, learning rate) need confirmation; details on tokenization of textual prompts for the chosen LLM (e.g., GPT-2, LLaMA) should be clarified."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "pyyaml==5.4.1",
        "transformers==4.12.0",
        "tqdm==4.62.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "DataLoader class: handles loading, normalization, segmentation, and timestamps conversion for multiple datasets. Responsible for returning structured time series samples with series, timestamps, and metadata."
        ],
        [
            "prompt_builder.py",
            "PromptBuilder class: provides methods to construct textual prompts from series and timestamps based on strategies (e.g., lookback window + recent series). Converts series data into token tensors, and timestamps into textual prompts, using tokenizer from the chosen LLM."
        ],
        [
            "model.py",
            "Model class: loads pre-trained decoder-only language models, freezes backbone parameters (except trainable embedding layers), implements embedding functions for series segments and timestamps, autoregressive next-token prediction, and decoding back to series space."
        ],
        [
            "trainer.py",
            "Trainer class: initializes model and dataset, manages training loop. Only train the segmentation embedding layer and projection head. Performs iterative autoregressive predictions on batch data. Saves checkpoints."
        ],
        [
            "evaluation.py",
            "Evaluation class: loads trained model, constructs prompt sequences, performs iterative predictions for multiple horizons, computes metrics (SMAPE, MSE, MAE), and generates visualizations."
        ],
        [
            "main.py",
            "Entry point: reads configuration (hyperparameters, dataset paths, model path), orchestrates dataset loading, prompt construction, model initialization, training, and evaluation. Sets seeds, handles command-line args."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "prompt_builder.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Use tokenizer from the open-source language model (e.g., transformers library) for converting prompts. Maintain consistent tokenization between prompt construction and model input. Use configuration files for hyperparameters and dataset paths. Keep the tokens and embeddings aligned with the pre-trained model's dimensions."
    ,
    "Anything UNCLEAR": "Details about the format of textual timestamps, dataset file formats, and specific prompt strategies need clarification. Also, confirm preferred dataset normalization method and how to handle special tokens (e.g., start/end tokens). Hardware requirements for training large models should be specified."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  etth1_path: "path/to/ETTh1.csv"
  weather_path: "path/to/Weather.csv"
  ecl_path: "path/to/ECL.csv"
  traffic_path: "path/to/Traffic.csv"
  solar_energy_path: "path/to/SolarEnergy.csv"
  m4_yearly_path: "path/to/M4_Yearly.csv"
  m4_quarterly_path: "path/to/M4_Quarterly.csv"
  m4_monthly_path: "path/to/M4_Monthly.csv"
  m4_weekly_path: "path/to/M4_Weekly.csv"
  m4_daily_path: "path/to/M4_Daily.csv"
  m4_hourly_path: "path/to/M4_Hourly.csv"
  m3_yearly_path: "path/to/M3_Yearly.csv"
  m3_quarterly_path: "path/to/M3_Quarterly.csv"
  m3_monthly_path: "path/to/M3_Monthly.csv"
  m3_others_path: "path/to/M3_Others.csv"

hyperparameters:
  lookback_length: 672  # default as per paper (can be adjusted)
  forecast_horizon: 96  # or other horizons like 192, 336
  segment_size: 96      # e.g., 96, 192, 336, 720
  prompt_strategy: "firstF"  # options: firstF, lastF, recentSeries, random, etc.
  training_epochs: 50  # default around typical training cycles
  batch_size: 224     # based on experiments for training speed
  learning_rate: 5e-5  # based on general LLM fine-tuning practices
  text_prompt_length: 48  # number of textual prompt tokens for timestamp encoding

model:
  pretrained_model_name: "LLaMA-7B"  # or "GPT2", "OPT-1.3B"
  freeze_backbone: true
  embedding_dim: 768  # match with LLaMA-7B, GPT-2 base
  trainable_layers: ["embeddings", "projection_head"]
  max_input_length: 672  # same as lookback_length
  max_generate_length: 96  # forecast horizon

training:
  optimizer: "AdamW"
  learning_rate: 5e-5
  weight_decay: 0.01
  epochs: 50
  gradient_clip_norm: 1.0

evaluation:
  metrics: ["SMAPE", "MAE", "MSE"]
  eval_batch_size: 32
```

---

**Note:** Full configuration is available in `planning_config.yaml`
