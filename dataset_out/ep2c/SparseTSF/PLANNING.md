# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, step-by-step plan for reproducing the SparseTSF methodology, experiments, and evaluation based on the paper's details. This outline includes core theoretical components, data handling, implementation considerations, and experimental protocols needed for faithful replication.

---

## 1. Core Methodology and Model Architecture

### a. Theoretical Foundations
- **Decomposition Assumption:** Time series \( X(t) \) decomposes into a periodic component \( P(t) \) and a trend component \( T(t) \):  
  \[
  X(t) = P(t) + T(t)
  \]
- **Periodic components** are (approximately) invariant over a known period \( w \) (e.g., 24 hours for daily cycles).
- **Forecasting task** focuses primarily on predicting trend variability \( T(t) \), with periodicity separated out.

### b. Cross-Period Sparse Forecasting Technique
- **Downsampling & Decomposition:**
  - Given a historical sequence \( x_{t-L+1:t} \) (length \( L \)) and forecast horizon \( H \), downsample via fixed period \( w \):
    \[
    x'_{t-L+1:t} = \text{segment of } x_{t-L+1:t} \text{ sampled every } w \text{ steps}
    \]
  - Downsampled sequence length:
    \[
    n = \left\lfloor \frac{L}{w} \right\rfloor
    \]
- **Prediction on Downsampled Sequences:**
  - Formulate at each position \( t \):
    \[
    x'_{t-L+1:t} \text{ (length } n)
    \]
  - Input: sequence \( x'_{t-L+1:t} \)
  - Output: forecasted downsampled sequence \( \hat{x'}_{t+1:t+H} \), same length \( m \).  
- **Reconstruction:**
  - Upsample the predicted downsampled sequence back to length \( H \) (e.g., via repeating or interpolation).  
  - Add back estimated periodic component (constant over the horizon \( w \)), to recover actual forecast:
    \[
    \hat{X}_{t+1:t+H} = \text{upsampled forecast} + P(t+1:t+H)
    \]

### c. Model Architecture
- Use a **single 1D convolutional layer** to process the downsampled input sequence:
  - Kernel size related to the period \( w \) (possibly \( 2w \))
  - Number of parameters \( < I \times k \) (small scale)
- Follow the convolution with linear projection to produce the forecasted downsampled sequence.
- Upsampling: simple repeating or linear interpolation.
- Add constant periodic component \( P(t) \) which is pre-estimated or learned.

---

## 2. Implementation Details & Parameters
- **Hyperparameters:**
  - **Period \( w \):** For datasets with known periodicity, set \( w \) accordingly (e.g., 24 for daily; 7*24 for weekly). Also test various \( w \).
  - **Downsampling factor:** \( \left\lfloor \frac{L}{w} \right\rfloor \).
  - **Forecast horizon \( H \):** e.g., 720 steps as per datasets.
  - **Input sequence length \( L \):** encourage large, e.g., 336 or 720, depending on dataset.
- **Normalization Strategy:**
  - Before feeding sequences into the model: subtract mean of the input sequence (\( x_{t-L+1:t} \))
  - After prediction: add the mean back
- **Model Parameters:**
  - Only a few parameters: a small convolutional layer, possibly with shared weights across channels (channel-independent).
  - Use proper weight initialization (e.g., Xavier).

---

## 3. Datasets and Data Handling
- **Datasets Needed:**
  - Four datasets with daily periodicity:
    - ETTh1 / ETTh2 — electrical data (energy consumption)
    - Weather — weather data (10-minute freq)
    - Traffic — traffic flow data (10-minute freq)
  - Additional ultra-long datasets: ETTm1, ETTm2, Weather, with cycles > 7 days.
- **Preprocessing:**
  - Rescale / normalize data: min-max or standardization.
  - Segment data into sequences of length \( L \) for training.
  - For downsampling:
    - Use known period \( w \) (e.g., 24 for hourly, 1440 for minutes/large periods).
    - Generate downsampled sequence:
      \[
      x'_{t-L+1:t} = \text{sequence sampled every } w \text{ with length } n.
      \]
- **Train/Validation/Test Split:**
  - Use the splits as specified in datasets or as used in the paper (training on earlier, testing on later).
  - Maintain consistent periods for train/val/test splits for fair comparison.

---

## 4. Training Protocol
- **Loss Function:**
  - Mean Squared Error (MSE) on the entire predicted horizon.
  - Apply sequence normalization (subtract mean before, add after).
- **Optimizer:**
  - Adam optimizer with learning rate, e.g., 0.001, early stopping based on validation.
- **Batch Size:**
  - Use large batch sizes if GPU memory allows; e.g., 128 or 256.
- **Epochs:**
  - At least 30 epochs, with early stopping if validation error plateaus.
- **Regularization:**
  - Weight decay or dropout unnecessary due to minimal parameter size.
- **Learning Rate Decay:**
  - Cosine annealing or step decay for stabilization.

---

## 5. Evaluation Metrics and Protocol
- **Primary Metric:**
  - MSE on forecast horizon \( H \).
- **Additional Metrics:**
  - MAE: absolute error.
  - During hyperparameter tuning, measure performance for various \( w \) and sequence lengths.
- **Baselines for Comparison:**
  - Mainstream models (Informer, Autoformer, FEDformer, DLinear, FTS).
  - Implement as per original papers or reproduce their methods for fair comparison.
- **Generalization Experiments:**
  - Train on one dataset, evaluate on another (e.g., ETTh1 → ETTh2) to test transferability.

---

## 6. Additional Implementation Notes
- **Periodic Component \( P(t) \):**  
  - Can be estimated via simple mean over each period using train data.
  - Or learned as a sequence parameter (e.g., through a small embedding or a fixed periodic pattern).
- **Upsampling Methods:**
  - Repeat each point \( w \) times.
  - Or interpolate linearly.
- **Hyperparameter \( w \):**
  - Tune around known dominant periods.
  - Test sensitivity to deviations.
- **Handling Multiple Periods (e.g., weekly + daily cycles):**
  - Downsampling for each period, then ensemble predictions.
  - Or design multi-period models (though paper mainly discusses single period in ablations).

---

## 7. Final Checklist for Reproduction
- Implement data loading pipelines consistent with datasets.
- Normalize data before training and invert normalization for evaluation.
- Extract periodic component \( P(t) \) based on known or estimated period.
- Downsample input sequences with fixed period \( w \).
- Use a lightweight CNN (single conv layer) on downsampled sequence.
- Forecast downsampled sequence; then upsample to horizon length.
- Add back periodic component for final forecast.
- Evaluate with MSE, MAE, with multiple \( w \) settings.
- Perform ablation studies: impact of \( w \), normalization, sequence length.
- Validate on multiple datasets, including ultra-long cycles.

---

## Summary
This plan frames the key theoretical principles (decomposition, cross-period forecasting, lightweight conv-based architecture), details data preparation, hyperparameters, training procedures, and evaluation protocols aligned with the paper. It provides a platform for precise implementation while highlighting critical choices (period \( w \), normalization, dataset specifics) to ensure faithful reproduction.

---

Would you like me to proceed with translating this detailed plan into actual Python pseudocode or code snippets next?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a lightweight Time Series forecasting system implementing the SparseTSF methodology. The system will include modules for dataset loading and preprocessing, a model that downsamples the input series with a fixed period, processes it through a simple convolutional neural network, reconstructs the forecasted series, and adds a periodic component. We will use PyTorch for model implementation and training, pandas and numpy for data handling, and rely on standard training procedures with normalization and evaluation utilizing the provided metrics.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(dataset_paths: dict, period: int)\n        +load_data() -> Tuple[np.ndarray, np.ndarray]\n        +get_train_val_test_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray]\n    }\n    class SparseTSFModel {\n        +__init__(params: dict)\n        +predict(x: np.ndarray, period: int, horizon: int) -> np.ndarray\n        +get_periodic_component() -> np.ndarray\n    }\n    class Trainer {\n        +__init__(model: SparseTSFModel, train_data: Tuple[np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray], config: dict)\n        +train() -> None\n        +save_model(filepath: str) -> None\n    }\n    class Evaluator {\n        +__init__(model: SparseTSFModel, test_data: Tuple[np.ndarray, np.ndarray], horizon: int)\n        +evaluate() -> dict\n        +calculate_metrics(pred: np.ndarray, true: np.ndarray) -> dict\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluator\n    Trainer --> SparseTSFModel\n    Evaluator --> SparseTSFModel\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MT as SparseTSFModel\n    participant TR as Trainer\n    participant EV as Evaluator\n    M->>DL: load dataset paths and parameters\n    DL-->>M: return training, validation, test data + period\n    M->>MT: initialize model with hyperparameters\n    M->>TR: create trainer with model and data\n    TR->>MT: train (input sequences, target sequences)\n    TR-->>M: training complete, saved model\n    M->>EV: load saved model and test data\n    EV->>MT: predict on test sequences\n    EV->>EV: evaluate predictions with true values, compute metrics\n    EV-->>M: output evaluation results\n",
    "Anything UNCLEAR": "Clarify the expected format of input datasets (e.g., CSV, numpy arrays), the specifics of periodic component estimation (fixed or learned), and how to handle multiple periodicities (single period vs. multiple periods). Also, confirm whether to include hyperparameter tuning or fixed hyperparameters for w and sequence lengths."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "pandas==1.3.0",
        "scikit-learn==0.24.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: Responsible for reading raw data files, normalization, train/validation/test splitting, and producing numpy arrays. Must handle datasets like ETTh1, ETTh2, Weather, Traffic, with parameters for data paths and periodicities. Implements methods for downsampling sequences based on a fixed period w. Dependency: numpy, pandas."
        ],
        [
            "model.py",
            "Implements SparseTSFModel class: Creates a lightweight 1D convolutional neural network. Receives downsampled input sequences, processes via conv layer(s), predicts downsampled future sequences, and provides methods for reconstructing full horizon predictions. Also includes a method to estimate or store the periodic component P(t). Dependency: torch, numpy."
        ],
        [
            "trainer.py",
            "Implements Trainer class: Coordinates training loop, including data batching, normalization (mean subtraction, addition), forward pass, loss calculation (MSE), backpropagation, and validation. Combines the dataset loader and model, manages checkpoints. Dependencies: torch, model.py, dataset_loader.py."
        ],
        [
            "evaluation.py",
            "Evaluation class: Loads the trained model, runs predictions on test data, computes metrics (MSE, MAE), and handles result logging. Interfaces with model.py and torch. Dependency: torch."
        ],
        [
            "main.py",
            "Entry point script: Parses configs, initializes dataset loader, model, trainer, and evaluation objects. Coordinates the experiment: data loading, training, saving models, and evaluation. Runs hyperparameter experiments for different w and sequence lengths. Dependency: all above modules."
        ],
        [
            "utils.py",
            "Utility functions: For normalization/inversion, upsampling, fixed periodic component extraction (via mean, frequency analysis), and plotting if needed. Supports data handling and visualization. Dependency: numpy, matplotlib (optional)."
        ]
    ],
    "Task list": [
        "dataset_loader.py",    # Load and process raw datasets; implement downsampling logic.
        "utils.py",             # Utility functions: normalization, periodic component estimation, upsampling.
        "model.py",             # Design lightweight CNN for downsampled sequence prediction.
        "trainer.py",           # Framework for training procedure.
        "evaluation.py",        # Prediction and metric calculation.
        "main.py"               # Orchestrate full experiment pipeline.
    ],
    "Full API spec": ""
      // No API communication, focus is on internal module interactions.
    ,
    "Shared Knowledge": "Normalization functions (mean subtraction, mean addition), fixed period downsampling logic, window slicing utilities, and model checkpoint paths are shared across modules.",
    "Anything UNCLEAR": "Clarify whether the periodic component P(t) should be estimated dynamically or fixed, and if training on multiple periodicities (daily + weekly) needs to be integrated."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  energy_path: "path/to/energy_dataset.csv"
  weather_path: "path/to/weather_dataset.csv"
  traffic_path: "path/to/traffic_dataset.csv"
  period_etth: 24    # Daily cycle period for ETTh datasets
  period_weather: 1440 # Approximate period for weather (10 min freq, 24 hours)
  period_traffic: 144 # 10 min frequency, so approx 144 for daily cycle
training:
  learning_rate: 0.001
  batch_size: 128
  epochs: 30
  optimizer: "adam"
  loss_function: "mse"
  seed: 42
model:
  period_w: 24           # Default period w: set to data's periodicity
  input_length: 336       # Sequence length for training
  forecast_horizon: 720   # Prediction horizon
  kernel_size: 3          # For convolutional layer
  conv_channels: 8        # Number of convolution filters
  use_shared_params: true # Whether to share convolution weights across channels
normalization:
  method: "mean_subtract" # Normalize by subtracting mean
  inverse_method: "add_mean" # Add mean back after prediction
evaluation:
  metrics: ["mse", "mae"]
  validation_split: 0.2
  test_split: 0.2
seed:
  value: 42
```

---

**Note:** Full configuration is available in `planning_config.yaml`
