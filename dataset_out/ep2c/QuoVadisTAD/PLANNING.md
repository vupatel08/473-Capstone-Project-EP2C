# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Below is a detailed, structured plan to reproduce the methodology described in the paper **"Quo Vadis, Unsupervised Time Series Anomaly Detection?"** by Saquib Sarfraz et al. It covers all key experimental and methodological details, with explicit focus on implementation-relevant elements, dataset handling, hyperparameters, and evaluation.

---

# 1. Overview and Objectives
- **Goal:** Replicate the authors’ critical analysis demonstrating that simple linear models or basic heuristics can match or outperform complex deep models for unsupervised time series anomaly detection.
- **Approach:**  
  - Implement baseline methods: range-based, L2-norm, nearest neighbor, PCA error, and simple neural network (MLP).  
  - Replicate complex models: Deep autoencoders, GDN, TranAD, AnomalyTransformer, etc., for comparison (if computationally feasible).  
  - Conduct ablations with linearized versions of complex models  
  - Evaluate using point-wise F1, precision, recall, and range-based metrics (APRC).

---

# 2. Dataset Collection & Processing
### 2.1 Dataset Requirements
- **Datasets used in the paper:**  
  - UCR/IB datasets: IB-16 to IB-19, with precise numbers of sensors and anomaly patterns.  
  - Wadi-127, Wadi-112, SWaT, and SMD for multivariate data.  
  - Additional datasets: SMAP, MSL, etc. (if needed, for generalization).  

- **Access:**  
  - Download datasets from publicly available repositories or from author-provided links (e.g., UCR/IB, SWaT, WADI, SMD).  
  -Important: For datasets with labels, need both training (normal only) and test (normal + anomalies).

- **Preprocessing:**  
  - Convert raw data into time series format, ensuring alignment of timestamps.  
  - For multivariate datasets, assemble features per timestamp into feature vectors.  
  - For univariate datasets, handle as single series.  
  - Standardize or normalize data if needed, but be consistent with the paper's experimental setting:  
    - Use raw data or avoid normalization for baseline methods.  
    - For models requiring normalization (e.g., PCA, neural networks), normalize train data (e.g., robust scaler or z-score).

### 2.2 Data Transformation
- **Sliding window representation for univariate data:**  
  - Use window size **4** as optimal from the paper's ablation study (Section 4.3).  
  - Generate windowed subsequences per series:  
    - For a series \( \mathbf{x} \), collect samples \(\mathbf{x}_{t - w + 1: t}\) for \( t = w, ..., T \).  
    - Each subsequence becomes a feature vector (dimension \(w\)).  
    - Label each window as anomalous if any timestamp in it is anomalous (if evaluating range-based metrics).  
- **For multivariate data:**  
  - Use each timestamp as a feature vector.  
  - No windowing unless specified.

---

# 3. Implementation of Baseline Methods
### 3.1 Range-based Heuristic
- For each feature in the training set, record min and max values.  
- Detection: Test point is anomalous if any feature exceeds training min/max bounds.  
- Threshold: Strict bounds from train data.  
- Use as minimal baseline (failure if it underperforms).

### 3.2 L2-norm
- Compute \(\|\hat{\mathbf{x}}_t\|\) for each timestamp in the test set.  
- Cutoff threshold:  
  - Use a fixed percentile or threshold, or use the training distribution’s median/std (as in the paper).  
  - For reproducibility, select a threshold (e.g., 95th percentile on training norms).

### 3.3 Nearest Neighbor (k-NN) based
- **Method:**  
  - Fit a k-NN (e.g., scikit-learn's `NearestNeighbors`) on the training data (or windowed versions for univariate).  
  - Compute the distance of each test point to the nearest training point.  
  - Anomaly score: distance.  
- **Hyperparameters:**  
  - \(k=1\) or a small value (e.g., 3).  
  - Distance metric: Euclidean (\(\ell_2\)).  
- **Thresholds:**  
  - As with the norm, select percentile thresholds from the train distribution for anomaly detection.

### 3.4 PCA Reconstruction Error
- Compute PCA on train data (using `sklearn.decomposition.PCA`).  
  - Variance explained: use number of components to explain 90-95% variance (typical).  
  - Number of components: start with 30 (from paper) or tune based on explained variance.  
- Transform test data: project onto principal components, then reconstruct.  
- Error: \( \mathbf{E}_t = \hat{\mathbf{x}}_t - \text{reconstructed} \).  
- Anomaly score: \(\| \mathbf{E}_t \|\) (e.g., Frobenius norm).  
- Threshold: set based on training error distribution (95 percentile or other).

### 3.5 Neural network baseline: Linear MLP
- Architecture:  
  - Input: feature vectors (for point-wise data) or windowed vectors (for univariate).  
  - Single hidden layer: size 32, with ReLU activation.  
  - Output: scalar value (reconstruction / prediction).  
- Training:  
  - Loss: Mean squared error (MSE).  
  - Data: train data (normal only).  
  - Optimizer: Adam, learning rate 0.001 (or as used).  
  - Batch size: 512.  
  - Epochs: until convergence or max epochs (e.g., 50).  
- For each test sample, predict using the trained model.  
- Anomaly score: prediction error (e.g., MSE), or direct difference.

### 3.6 More Complex Deep Models (Optional, if resources permit)
- Implement or re-implement GDN, TranAD, Transformer-based models following open literature descriptions (if code/parameters aren’t available, use representative configs).  
- Use their reported hyperparameters from the paper or related repositories.  
- Focus on reproducing their point-based or range-based detection metrics.

### 3.7 Linear Approximation of Deep Models
- Linearize trained deep models (approximate neural networks by linear models).  
- Methods:  
  - Use first-order Taylor expansion (gradients) at training data points.  
  - Supervised linear regression (for reconstruction) based on deep model output features—if feasible.  
  - Alternatively, train a linear model (e.g., a linear regression or ridge) on the same input-output pairs as the deep network’s training.  
- Evaluate the linearized models’ performance on test data, as in the paper.

---

# 4. Evaluation Protocols
### 4.1 Metrics
- **Point-wise F1 score:**  
  - Use true labels and predicted anomaly points, applying thresholds (e.g., via point-adjust or directly).  
  - Thresholds:  
    - Optimize threshold on validation or train set for best F1 (using `precision_recall_curve` or similar).  
    - Use fixed thresholds for comparison, aligning with the paper's thresholds.
- **Range-based metrics:**  
  - **APRC (Average Precision of Range Coverage):**  
    - Compute overlapping intervals between predicted and true anomalies.  
    - Use formulae from the paper, including the intersection over union (IOU) thresholds introduced (Section 4.4).  
- **Additional metrics:**  
  - Precision, recall, F1 at various thresholds.

### 4.2 Threshold selection
- For each method:  
  - Determine thresholds on the train set using validation splits or cross-validation, optimizing F1 or range metrics.  
  - Evaluate detection on the test set with these fixed thresholds.

### 4.3 Evaluation details
- **Point-level detection:**  
  - Convert continuous anomaly scores into binary labels via thresholds.  
  - For range-based metrics, group consecutive true positives into anomaly segments.  
- **Interval-based detection:**  
  - For each predicted anomaly interval, compare against true ground-truth anomaly intervals to compute IOU, precision, recall, and F1.

---

# 5. Experimental Procedures
### 5.1 Data Handling
- Implement data loaders that can:
  - Load raw time series from files.
  - Generate windowed features if needed.
  - Provide train/test splits with labels.

### 5.2 Hyperparameter Tuning
- Use a validation subset or cross-validation to tune parameters:  
  - Thresholds for each method.  
  - PCA component count.  
  - k-NN parameter \(k\).  
  - Neural network hyperparameters (learning rate, epochs, batch size).  
  - Window size (consider 4 as optimal). 

### 5.3 Numerical Stability & Reproducibility
- Fix random seeds (`np.random.seed()`, `torch.manual_seed()`).
- Use consistent data normalization (or none if specified).
- Log all hyperparameters and thresholds for reproducibility.

### 5.4 Repetition & Robustness
- Run multiple runs with different seeds if feasible.
- Report mean and variance of performance metrics.

---

# 6. Summary of Key Implementation Elements
| Aspect                        | Details                                                          |
|------------------------------|------------------------------------------------------------------|
| Datasets                     | Download from repositories; format as series or windowed data.  |
| Sliding Window               | Size 4 (univariate) or 1 (multivariate) as baseline.             |
| Baseline Methods              | Range bounds, Norm, 1-NN, PCA, Linear NN, simple MLP.            |
| Deep Models                    | GDN, TranAD, Transformer, or approximations (if resources allow). |
| Threshold Selection            | Use train/validation sets; optimize F1 and range metrics.      |
| Evaluation Metrics             | Point-wise F1, precision, recall, range IOU, APRC.               |
| Hyperparameter Tuning        | Based on validation; record thresholds and model configs.        |
| Reproducibility              | Fixed seeds, documented parameters, consistent data splits.     |

---

This roadmap emphasizes a comprehensive understanding of the methods, rigorous experiment design, and faithful reproduction of evaluation protocols. Using this plan, you can systematically implement and compare baselines and models, respecting the authors’ focus on simplicity and interpretability.

Would you like me to now proceed with detailed code snippets for each component?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular but straightforward system that includes data loading, baseline methods (range heuristic, L2-norm, k-NN, PCA error, simple neural network), evaluation, and experiment orchestration. The core will be in using PyTorch for neural method components, scikit-learn for classical models, and numpy/pandas for data handling. The architecture is designed to be easily reproducible, with a main script coordinating data loading, training, and evaluation, and separate classes for dataset management, models, and evaluation procedures.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "models.py",
        "baseline_methods.py",
        "train.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run_experiment()
    }
    class DatasetLoader {
        +__init__(dataset_path: str, window_size: int=4)
        +load_data() -> dict
        +get_train_test() -> tuple
    }
    class RangeHeuristic {
        +__init__(train_data: np.ndarray)
        +detect(test_data: np.ndarray) -> np.ndarray
        +get_thresholds() -> dict
    }
    class NormBasedMethod {
        +__init__(train_data: np.ndarray)
        +detect(test_data: np.ndarray) -> np.ndarray
        +compute_thresholds(percentile: float=95) -> float
    }
    class KNNMethod {
        +__init__(train_data: np.ndarray, k: int=1)
        +detect(test_data: np.ndarray) -> np.ndarray
        +fit() -> None
    }
    class PCAReconstruction {
        +__init__(train_data: np.ndarray, n_components: int=30)
        +detect(test_data: np.ndarray) -> np.ndarray
        +fit() -> None
        +get_reconstruction_error() -> np.ndarray
    }
    class SimpleNeuralNet {
        +__init__(input_dim: int)
        +train(train_data: np.ndarray, epochs: int=50, batch_size: int=512, learning_rate: float=0.001) -> None
        +predict(test_data: np.ndarray) -> np.ndarray
        +compute_error(test_data: np.ndarray) -> np.ndarray
    }
    class Evaluator {
        +__init__(pred_scores: np.ndarray, labels: np.ndarray, thresholds: list)
        +compute_pointwise_metrics() -> dict
        +compute_range_metrics() -> dict
    }
    class Utility {
        +load_dataset(path: str) -> dict
        +normalize_data(data: np.ndarray) -> np.ndarray
        +generate_windows(data: np.ndarray, window_size: int=4) -> np.ndarray
    }
    Main --> DatasetLoader
    Main --> RangeHeuristic
    Main --> NormBasedMethod
    Main --> KNNMethod
    Main --> PCAReconstruction
    Main --> SimpleNeuralNet
    Main --> Evaluator
    DatasetLoader --> Utility
    SimpleNeuralNet --> utils.py
    Evaluator --> utils.py
    RangeHeuristic --> utils.py
    NormBasedMethod --> utils.py
    KNNMethod --> sklearn.neighbors
    PCAReconstruction --> sklearn.decomposition
    Main --> config (dict) to define data paths, hyperparameters, thresholds
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant D as DatasetLoader
    participant R as RangeHeuristic
    participant N as NormBasedMethod
    participant K as KNNMethod
    participant P as PCAReconstruction
    participant S as SimpleNeuralNet
    participant E as Evaluator
    M->>D: load_data()
    D-->>M: train/test datasets
    M->>R: new(train_data)
    R->>R: compute thresholds()
    R-->>M: thresholds info
    M->>N: new(train_data)
    N->>N: compute_thresholds()
    N-->>M: threshold info
    M->>K: new(train_data, k=1)
    K->>K: fit()
    K-->>M: ready
    M->>P: new(train_data, n_components=30)
    P->>P: fit()
    P-->>M: errors
    M->>S: new(input_dim)
    S->>S: train(train_data)
    S-->>M: trained model
    M->>variants: detect anomalies on test_data
        RangeHeuristic.detect(test_data)
        NormBasedMethod.detect(test_data)
        KNNMethod.detect(test_data)
        PCAReconstruction.detect(test_data)
        nn_errors = SimpleNeuralNet.compute_error(test_data)
        thresholds are applied as needed (from thresholds)
    M->>E: evaluate(predictions, labels, thresholds)
    E-->E: compute metrics
    M-->end
    Note over M: Thresholds determined on train data, applied to test scores.
    Note over E: Metrics include F1 pointwise, range IOU, APRC.
",
    "Anything UNCLEAR": "Clarify dataset formats, e.g., label formats, timestamp indices, and how to generate initial train/test splits. Confirm whether anomalies are segment-based or point-based, and how to handle multi-sensor data (flattening or windowed schemes). Confirm if extra normalization steps are required before baseline detection or neural training."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "scikit-learn==0.24",
        "torch==1.9.0",
        "pandas==1.3.0",
        "scipy==1.7.0",
        "matplotlib==3.4.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class that manages loading datasets, parsing raw data, generating windowed features, and splitting into train/test. Depends on utils.py for file I/O and normalization. Responsible for standardizing data or applying sliding windows."
        ],
        [
            "utils.py",
            "Provides utility functions for loading datasets, normalizing data, generating windows for univariate time series, and handling normalization thresholds. Shared by data loading, models, and evaluation."
        ],
        [
            "model.py",
            "Defines neural network models: a simple MLP class (e.g., SingleLayerMLP). Responsible for constructing, training, and inference. No external dependencies beyond torch."
        ],
        [
            "baseline_methods.py",
            "Implements range heuristic, L2-norm, k-NN, PCA error, and simple neural network (training and detection). Utilizes sklearn for PCA, neighbors, and statistical thresholds. Shared parameters include window size and thresholds from the training set."
        ],
        [
            "trainer.py",
            "Handles training of neural models such as the simple MLP. Uses torch optimizers, handles batching, epoch management, and checkpointing if needed. Complements model.py with training loop logic."
        ],
        [
            "evaluation.py",
            "Implements metrics such as point-wise F1, precision, recall, range IoU, and APRC. Receives scores and labels, computes thresholds, applies range intersection metrics, and outputs detailed performance reports."
        ],
        [
            "main.py",
            "Main orchestration script that coordinates dataset loading, training of neural models, applying all baseline methods, performing detection, and evaluating results. Assembles data, calls methods from other modules, manages flow control, and logs outputs."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "utils.py",
        "dataset_loader.py",
        "model.py",
        "baseline_methods.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ...  # No front-end/backend APIs needed for core experiments, only data and metrics collection if applicable.",
    "Shared Knowledge": "The utils.py module provides common functions for dataset I/O, normalization, window generation, and threshold calculation shared across dataset_loader.py, models.py, baseline_methods.py, and evaluation.py.",
    "Anything UNCLEAR": "Clarify dataset format details: exact label encoding, timestamp indexing, and anomaly interval representations for the range metrics. Confirm whether normalization or standardization is needed before applying thresholds."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  path: "/path/to/dataset"  # Placeholder path; replace with actual dataset directory
  window_size: 4            # Optimal window size from paper's ablation (Section 4.3)
  train_split_ratio: 0.8    # Using 80% for training, 20% for testing by default

model:
  type: "SimpleMLP"           # Model type; can be expanded to support others
  hidden_size: 32             # Single hidden layer size
  input_dim: null             # To be set dynamically based on data (e.g., window size)

training:
  optimizer: "Adam"
  learning_rate: 0.001       # Standard learning rate as used in paper's hyperparameters
  batch_size: 512            # Batch size from paper's description
  epochs: 50                 # Max epochs, or until convergence
  early_stopping_patience: 10  # Optional: stop if validation score doesn't improve

evaluation:
  thresholds:
    method: "percentile"     # Method to determine anomaly threshold
    percentile: 95           # Cutoff at 95th percentile of training error scores
  metrics:
    pointwise: true
    range: true            # Enable both point-wise F1 and range-based metrics
```

---

**Note:** Full configuration is available in `planning_config.yaml`
