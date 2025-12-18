# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## baseline_methods.py

# Logic Analysis for baseline_methods.py

The `baseline_methods.py` module is designed to implement and encapsulate several classical, heuristic, and simple model-based anomaly detection methods, following the methodology outlined in the paper. These methods serve as robust, interpretable, and computationally efficient baselines against which more complex deep models are compared.

The core functionalities to implement include:

1. **Range Heuristic**
2. **L2-norm Based Detector**
3. **k-Nearest Neighbors (k-NN) Distance Detector**
4. **PCA Reconstruction Error Detector**
5. **Simple Neural Network (MLP) Autoencoder)**

All methods are to be trained solely on the training data (comprising only normal instances), and then used to detect anomalies in the test data by applying method-specific thresholds derived from training scores.

---

# Common Design Considerations

- **Inputs and Outputs:**
  - Training data: `train_data` as a NumPy array shape `(T_train, F)` or `(num_samples, feature_dim)`
  - Test data: `test_data` shape `(T_test, F)`
  - Method-specific parameters, e.g., thresholds, PCA components, k-NN neighbors, window size for univariate, etc.
  - Outputs: anomaly scores (float array of length T_test), and potentially binary anomaly labels (thresholded).

- **Shared Parameters:**
  - `window_size` (int): for univariate or window-based methods.
  - Threshold values: determined on train data via a `percentile` (e.g., 95th percentile) or explicit value.
  
- **Thresholding:**
  - Based on training scores: thresholds are set to a specific percentile to emulate typical evaluation practices.
  - Detection: label test points as anomalies if their scores exceed the threshold.

- **Model Storage:**
  - For models requiring training (e.g., PCA, k-NN, neural network), store fitted models for test inference.

---

# 1. Range Heuristic
- **Logic:**  
  - During training, compute min and max values for each feature across all training samples.
  - During detection, compare each feature of test samples with training min/max.
  - Anomaly if any feature falls outside [min, max].

- **Implementation details:**  
  - Store `train_min` and `train_max` arrays of shape `(F,)`.
  - Detection scores: for each test sample, if any feature is outside range, assign a high anomaly score (e.g., overflow indicator) or binary label (1) based on threshold.
  - For scores, a simple indicator (binary) or count of out-of-range features can be used.

---

# 2. L2-norm Detector
- **Logic:**  
  - Compute the Euclidean norm of each training sample.
  - Threshold: set at a percentile (e.g., 95th) of training norms.
  - During detection, compute the norm for test samples, label anomalies if norms > threshold.

- **Implementation details:**  
  - Store training norms; compute threshold during training.
  - Similar to range heuristic, anomaly score per test sample: the norm itself.

---

# 3. k-Nearest Neighbors (k-NN) Distance
- **Logic:**  
  - Fit a k-NN model on training data features.
  - For each test sample, find the distance to the nearest training point (or the `k`-th neighbor if preferred).
  - Threshold: specified percentile of training distances.
  - Anomaly if test distance exceeds threshold.

- **Implementation details:**  
  - Use `sklearn.neighbors.NearestNeighbors`.
  - Fit on training data.
  - Test: compute `dist = nn.kneighbors(test_point).min()`.
  - Store training distances during fitting to set a threshold.
  
---

# 4. PCA Reconstruction Error
- **Logic:**
  - Fit PCA on training data: select number of components to capture ~90-95% variance or as specified.
  - Transform train and test data to PCA space.
  - Reconstruct test data via inverse transform.
  - Compute per-sample reconstruction error: usually Euclidean (Frobenius) norm.
  - Threshold: set at specified percentile (e.g., 95%) of training reconstruction errors.
  - Anomaly if reconstruction error > threshold.

- **Implementation details:**  
  - Use `sklearn.decomposition.PCA`.
  - Fit PCA on training data.
  - Store components and mean.
  - During detection, compute error vectors and their norms.

---

# 5. Simple Neural Network (MLP autoencoder)
- **Logic:**
  - Build a single hidden-layer MLP (size 32, as per config).
  - Input dimension `F` (or windowed feature dimension).
  - Train in self-supervised fashion to reconstruct inputs.
  - Loss: MSE between input and output.
  - Use training data (normal only).
  - During inference, compute reconstruction error per sample.
  - Threshold as above based on training errors (percentile).
  
- **Implementation details:**
  - Build network with `torch.nn.Module`.
  - Training: standard loop, optimizer `Adam`, early stopping optional.
  - After training, compute sample-wide errors.
  - Threshold defined on training errors.
  - During inference: output per-sample errors.

---

# Additional Notes:
- **Batching and efficiency:** methods should handle batched data properly during training and inference.
- **Reproducibility:** set random seeds for numpy, torch, sklearn (if applicable).
- **Output:**  
  - Final anomaly scores for test data.
  - Threshold values used (static or dynamic/per percentile).
  - Optional binary labels based on thresholding for evaluation.

---

# Summary
The implementation in `baseline_methods.py` must provide:

- Classes or functions for each baseline, capable of training (fitting) when needed and detection.
- Consistent interface: methods such as `.fit(train_data)` and `.detect(test_data)`, returning anomaly scores.
- Thresholding configuration: either as a user-set value or computed percentile on training scores.
- Use scikit-learn for PCA and k-NN, torch or numpy for neural network.
- Enforce consistent data handling: normalization, windowing, and feature extraction aligned with the described settings.

This detailed plan ensures faithful, reusable, and clear implementation aligned with the paper’s experimental approach.

## dataset_loader.py

### Logic Analysis for `dataset_loader.py`

The `dataset_loader.py` module is a crucial component in the experimental pipeline. Its purpose is to provide a flexible, reliable interface for loading, parsing, and preprocessing datasets, preparing them for subsequent modeling and evaluation. The class `DatasetLoader` should adhere to the following logical flow and design principles, directly informed by the paper’s methodology, dataset descriptions, and configuration directives.

---

## Core Responsibilities

1. **Initialization and Configuration:**
   - Accept dataset path(s) and parameters (e.g., window size, train/test split ratio).
   - Initialize internal variables to hold raw data, processed data, labels, and metadata.

2. **Data Loading:**
   - Read raw dataset files from specified directory.
   - Support multiple datasets formats, likely CSV, TSV, or specialized formats depending on the dataset.
   - Extract timestamps, sensor measurements, and labels.
   - For datasets with multiple traces or sensors, organize data into a consistent structure, e.g., a pandas DataFrame or a NumPy array of shape `(T, F)` or list thereof.
   - Ensure extraction of anomaly labels aligned as intervals or point labels (`0` for normal, `1` for anomalies).

3. **Parsing and Preprocessing:**
   - Convert raw data into a structured format suitable for feature extraction.
   - If necessary, convert timestamps to indices or handle irregular time series (assumed regular here unless specified otherwise).
   - Normalize or standardize data as needed for models:
     - Use external `utils.py` functions to normalize data (e.g., min-max scaling, z-score, median-IQR), typically applied on training data only, with normalization parameters stored for test data.
   - For multi-sensor data, ensure proper alignment: one feature vector at each timestamp.
   - For univariate data, handle as a single series; optionally, generate windowed features.

4. **Generating Windowed Features (for Univariate Time Series):**
   - When `window_size` specified (default 4):
     - Create overlapping windows of data points, i.e., `x_{t-w+1:t}` as feature vectors.
     - Each window corresponds to a new sample with dimension `(w, )` (flattened or preserved as array).
     - Labels for windows: assign label `1` if any point in the window is labeled as anomalous; otherwise `0`.
   - For multivariate datasets, typically, timestamps serve as features directly unless windowing is explicitly requested.

5. **Splitting into Training and Test Sets:**
   - Use `train_split_ratio` (default 0.8):
     - Randomly split the dataset or sequentially split if the datasets are time-series.
     - Maintain temporal order if necessary (generally, train on early data, test on later).
     - Only the training set must contain normal data (no anomalies), as per the paper description.
   - Store training and test data, labels, and associated metadata like timestamps, sensor labels, and true anomaly intervals.

6. **Output Structure:**
   - Provide methods:
     - `load_data()`: loads raw datasets and processes them.
     - `get_train_test()`: returns processed datasets, labels, and optionally additional info like anomaly intervals.
   - Deliver data in formats compatible with models:
     - NumPy arrays (`shape: (T, F)`) for multivariate data.
     - For univariate, after windowing, shape `(num_windows, window_size)`.

---

## Detailed Logical Sequence

### Initialization:
- Store parameters.
- Prepare placeholders for raw data and processed data.

### Data Loading:
- Invoke data reading functions:
  - For each dataset, read files into pandas DataFrame or NumPy array.
  - Extract features and labels:
    - Features: sensor measures or univariate series.
    - Labels: point-wise labels or intervals indicating anomalies.
- Parse metadata: timestamps, sensor info, anomaly intervals.

### Data Preprocessing:
- Call `utils.py` functions:
  - Normalize data: fit on train data, apply to both train and test.
  - Generate windows if univariate and `window_size` > 1.
- Generate feature matrices:
  - For multivariate: use raw features.
  - For univariate: generate sliding windows.

### Data Splitting:
- Split data into train and test:
  - Based on ratio (`train_split_ratio`).
  - Ensure train set contains only normal data (filter out anomaly points from training set).
  - Maintain sequence integrity (prefer sequential split unless data is known to be IID).

### Labels:
- Generate label arrays for train/test:
  - Binary labels (`0`, `1`).
  - For windowed data, label windows via logical OR of labels within the window.
- Map label indices to timestamps as needed.

### Output:
- Packaged data:
  - `train_data`, `train_labels`, `test_data`, `test_labels`.
  - Additional info: original timestamps, anomaly intervals, normalization parameters (if stored).

---

## Implementation Details and Assumptions

- **Dataset-specific parsing:**
  - Implement dataset-specific methods if needed, or generalize to formats with known schema.
- **Normalization Scheme:**
  - Use `utils.py` functions for normalization; store normalization parameters from training data.
- **Handling Anomaly Labels:**
  - If labels are interval-based, convert to point labels for pointwise evaluation and generate window labels for range metrics.
- **Time Series Nature:**
  - For evaluation, keep track of true anomaly intervals for range-based metrics, and point labels for pointwise.

---

## Summary

- The `DatasetLoader` class should:

  1. Load and parse raw datasets from files.
  2. Normalize data using `utils.py`.
  3. Generate sliding windows if needed.
  4. Partition data into training and test sets according to ratio.
  5. Ensure training data contains only normal data points.
  6. Provide interfaces to retrieve:
     - Raw train/test data.
     - Windowed features and labels.
     - Normalization parameters.
     - Metadata like true anomaly intervals for evaluation.
  7. Be modular enough to support multiple datasets with minimal change, primarily via parameters.

This detailed logical flow ensures fidelity to the paper’s experimental setup, supports isotropic evaluation, and facilitates reproducibility.

## evaluation.py

# Evaluation Logic Analysis for `evaluation.py`

## Purpose:
Implement functions to quantify the performance of time series anomaly detection models, aligning with the paper's evaluation protocols, particularly the point-wise F1, Precision, Recall, the range-wise F1 (using APPRC), and supporting detailed reporting.

---

## Core Responsibilities:
1. **Threshold determination:**  
   - Derive suitable thresholds (e.g., at the 95th percentile) on training/error scores to discriminate anomalies in test data.
   - Implement flexible methods (percentile-based, fixed thresholds) as per configuration.

2. **Binary Classification Generation:**  
   - Convert continuous anomaly scores to binary predictions using thresholds.
   - Support both points (per-timestamp labels) and range-based (contiguous interval overlaps).

3. **Point-wise Metrics Calculation:**  
   - Compute TP, FP, FN based on predicted and true point labels.
   - Calculate precision, recall, F1-score at point level.

4. **Range-wise Metrics Calculation:**  
   - Identify ground-truth anomaly intervals and predicted anomaly intervals.

   - Compute overlap metrics using IOU between these intervals over various hit ratio thresholds (from 0.2 to 0.95, step 0.05).
   
   - Use the formulae as specified:
     - Precision and recall as weighted sums over ground truth/predicted interval overlaps.
     - Calculate the F1 score as the harmonic mean of range-based precision and recall.

5. **APRC / Range-based F1 score:**  
   - Use the Intersection over Union (IOU) of detected intervals and true anomaly intervals with various hit ratio thresholds.
   - Average across thresholds for the final range-based F1.

6. **Evaluation output:**  
   - Produce a comprehensive dictionary of metrics:
     - Point-wise: precision, recall, F1, threshold.
     - Range-wise: precision, recall, F1, threshold, APRC.
   - Logging or printing detailed report for analysis.

7. **Input/Output Data:**
   - Inputs:
     - predicted scores (continuous): `scores`
     - ground truth labels: `labels`
     - optional thresholds (computed or fixed): `thresholds`
   - Outputs:
     - Metrics dictionary / report, possibly saved or printed.

---

## Detailed Step-by-Step Logic:

### 1. Threshold Determination
- **Input:** Array of scores from model predictions and validation data.
- **Process:**  
  - If method is `'percentile'`, compute the threshold as the percentile (e.g., 95th) of training scores.
  - If a fixed value is provided, use that.
- **Output:** Scalar threshold value to binarize scores.

### 2. Binarization
- **Input:** Continuous scores, threshold.
- **Process:**  
  - Predicted labels:  
    ```python
    y_pred = scores > threshold
    ```  
    (Boolean array, then convert to int: 0 or 1).
- **Output:** Binary label array (per timestamp).

### 3. Point-wise Metrics
- **Input:** `y_pred`, `labels`.
- **Process:**  
  - TP: count of timestamps where both predicted and true label are 1.
  - FP: count where predicted label is 1 but true label is 0.
  - FN: count where predicted label is 0 but true label is 1.
  - Precision = TP / (TP + FP), if denominator > 0.
  - Recall = TP / (TP + FN), if denominator > 0.
  - F1 = 2 * Precision * Recall / (Precision + Recall), if sum > 0.
- **Threshold optimization:**  
  - Iterate over thresholds or use ROC/PR curve to pick the best as per F1.
  - Select the threshold that yields the maximum F1 during validation or test (as per the paper’s protocol).

### 4. Range-based (Interval) Extraction
- **Identify true anomaly intervals:**  
  ```python
  true_intervals = get_anomaly_intervals(labels)
  ```
- **Identify predicted anomaly intervals:**  
  ```python
  pred_intervals = get_anomaly_intervals(y_pred)
  ```
  - These are lists of contiguous segments where labels are 1.

### 5. Intersection over Union (IOU) Calculation
- **For each pair of predicted interval and ground truth interval:**
  - Calculate IOU as:
    ```python
    IOU = intersect_length / union_length
    ```
- **Hit ratio for *A* in [a_s, b_s]:**  
  - Fraction of timestamps in \([a_s, b_s]\) with predicted labels=1.
  - For *covering* threshold (the hit ratio \(r\)), an interval is considered detected if its maximum predicted label overlap exceeds \(r\).
- **Interval matches:**  
  - An interval in ground truth is "detected" if matched by any predicted interval with IOU \(\geq r\).

### 6. Range metrics for various hit ratios:
- **For each hit ratio threshold \(r\) in [0.2:0.95:0.05]**:
  - Compute detection overlaps for each ground truth interval.
  - Calculate true positives (correctly detected) and total predicted/actual intervals.
  - Use the formula:
    \[
    \text{Precision}_T = \frac{1}{|\mathcal{P}|} \sum_{P} \gamma(|A_P|, P) \frac{|\cup \mathcal{A} \cap P| }{|P|}
    \]
    and similarly for recall.
  - \(\gamma( n, A )\) as defined: \(\left(\frac{|A|-1}{|A|}\right)^{n-1}\).  
  - Compute the F1 as usual:  
    \[
    F1_T = 2 * (Prec_T * Rec_T)/(Prec_T + Rec_T)
    \]
- **Finally,** average F1 over all thresholds to obtain the overall range-wise F1 (`F1_T`).

### 7. Final Output
- Return a dictionary with:
  - Point-wise metrics: `precision`, `recall`, `F1`, and the chosen threshold.
  - Range-based metrics: same, plus `APRC` (computed as average over all IOU thresholds).
  - All computed for specified thresholds but report the maximum F1s and corresponding thresholds as per setting.

---

## Additional Considerations
- Ensure that `labels` and `scores` arrays are aligned in time.
- For multi-sensor data, the logic applies per timestamp; for range metrics, proper interval detection is vital.
- Normalization of scores might be necessary before thresholding.
- Tight control over random seeds for reproducibility if randomness involved in detection (e.g., if thresholds are stochastic).

---

## Summary:
The `evaluation.py` module will:
- Accept raw continuous scores and labels.
- Determine thresholds using the specified method (percentile or fixed).
- Calculate point-wise detection metrics.
- Convert labels and scores into intervals, compute IOU, and subsequently range-wise detection metrics with various hit ratio thresholds.
- Aggregate metrics like F1, precision, recall, and APRC.
- Return and/or print detailed performance summaries, suitable for comparative analysis.

---

This detailed logic plan should guide precise implementation of `evaluation.py` aligned with the paper’s evaluation methodology.

## main.py

**Logic Analysis for `main.py` – The Main Orchestration Script**

---

### **Purpose & Responsibilities**
- Coordinate the overall experiment flow:
  - Load datasets
  - Generate necessary features (e.g., windowed data)
  - Initialize models and baseline methods
  - Train neural network models
  - Apply baseline detection methods
  - Perform anomaly detection
  - Evaluate results using specified metrics
  - Log and output performance metrics and diagnostics

---

### **Step-by-Step Logical Flow**

---

### **1. Load Configuration Settings**

- Read the `config.yaml` file to extract parameters:
  - Dataset path, window size, train/test split ratio
  - Model type and hyperparameters
  - Thresholding method and percentile
  - Metrics (point-wise, range-wise)

**Rationale:**  
Ensures experiments are customizable and reproducible with parameters read dynamically.

---

### **2. Initialize Data Handling**

- Instantiate `DatasetLoader` with:
  - `dataset_path` from config
  - `window_size` (from config) 
- Call `load_data()` to load raw dataset files:
  - DatasetLoader manages data reading from files, parsing timestamps, labels, and features.
  
- Retrieve training and testing data:
  - `train_data`, `test_data`, `train_labels`, `test_labels`, possibly as numpy arrays.

**Notes:**  
- Maintain consistent train/test split (e.g., using `train_split_ratio`)
- Apply normalization to `train_data`. For methods requiring normalized features (e.g., PCA, neural nets), normalize train data and apply same transformation to test data.

---

### **3. Generate Features for Baselines**

- For baseline methods:
  - Use raw or normalized data directly if point anomaly detection via scores.
  - For univariate datasets:
    - Generate windowed representations: For each univariate series, generate windows of size `window_size` (4) using sliding windows.
    - Label windows as anomalous if any point in that window is anomalous.
- Pass data structures (arrays) to baseline initialization.

**Key Details:**  
- For univariate series, implement `generate_windows()` function that returns features of shape `(num_windows, window_size)`.

---

### **4. Initialize and Train Models**

- **Neural Model:**  
  - Instantiate the model based on `model.type` – default here is `"SimpleMLP"`.  
  - Set `input_dim`: if data is windowed, `input_dim = window_size`; if point-wise, shape equals feature dimension.
  - Use `training` configuration for hyperparameters.
  - Call the `train()` method:
    - Data: training data features
    - Hyperparameters: epochs, batch size, learning rate
    - Use early stopping with `early_stopping_patience` if defined.
  - Save trained model object/reference for later inference.

- **Note:**  
  - For other models (e.g., autoencoders, GDN, etc.), instantiate according to their classes and hyperparameters.
  - For this main script, implement flexibility to support multiple models by config.

---

### **5. Apply Baseline Detection Methods**

- For each baseline:
  - **Range Heuristic:**  
    - Compute train data feature-wise min and max.
    - Detection: test points outside train min/max bounds.
  - **L2-norm:**  
    - Compute the L2-norm of each test point.
    - Determine a threshold based on `percentile` (e.g., 95th) of train norms.
  - **k-NN distance:**  
    - Fit `NearestNeighbors` on training data.
    - Compute distances for test points.
    - Threshold at train percentile (e.g., 95th percentile).
  - **PCA Reconstruction:**  
    - Fit PCA on train data (using `n_components` from config or default 30).
    - For test data, project onto PCA space, reconstruct, and compute errors.
    - Use training error distribution to set threshold (e.g., 95th percentile).

- For neural network baseline:
  - Run inference on test data:
    - For forecasting mode:
      - Use trained model to predict future points based on windowed past.
    - Compute prediction errors or scores.
    - Select threshold based on `percentile` on training errors or validation part.

---

### **6. Anomaly Scoring & Thresholding**

- For each method:
  - Calculate anomaly scores (e.g., errors, norms, distances).
  - Threshold scores to generate binary anomaly labels:
    - Use thresholds from training distribution derived via percentile method.
    - Or select thresholds that optimize validation performance (if validation set used).

**Special note:**  
- For range-based metrics, group predictions into anomaly intervals.
- For point-wise, compare predicted labels to ground-truth labels.

---

### **7. Performance Evaluation**

- For each detection method:
  - **Point-wise F1 score:**  
    - Using continuous scores and selected threshold, binarize scores.
    - Calculate precision, recall, F1-point using standard formulae.
  - **Range metrics (F1_T, APRC):**  
    - Convert binary labels into intervals.
    - Calculate segment overlap metrics based on IOU thresholds as in the paper.
    - Use the provided `evaluation.py` functions for computation.

- Store all metrics:
  - For each dataset, method, and metric.
  - Log results to console or save in output files for analysis.

---

### **8. Optional: Repeat with Different Thresholds or Hyperparameters**

- To replicate paper's protocol, vary thresholds (e.g., through percentile scans) to find optimal F1.
- Record detection thresholds and associated scores for reproducibility.

---

### **9. Final Report & Logging**

- Print summaries:
  - Method performance metrics (point-wise F1, range-based F1, AUPRC)
  - Thresholds used
  - Performance comparisons
- Save detailed metrics to files (JSON, CSV) for record-keeping.
- Include plots for ROC/PR curves if relevant.

---

### **10. Additional Considerations**

- **Seed Fixing:**  
  - Set fixed seeds before data loading, training, and inference to ensure reproducibility.

- **Handling Datasets with Labels & Anomalies:**  
  - Confirm label format:  
    - For segment-based anomalies: binary vector with 1 for anomalous points, 0 else.
  - For range metrics, group consecutive anomalies.
  - For point metrics, compare per timestamp.

- **Attention to Dataset Variants:**  
  - For multivariate data, handle missing data/preprocessing as per dataset specifics.
  - Confirm whether datasets are preprocessed or raw.

---

### **Summary**

`main.py` will orchestrate:

- Dataset loading & processing via `DatasetLoader`.
- Model instantiation & training with `model.py`.
- Baseline methods (range, norm, k-NN, PCA, neural) via `baseline_methods.py`.
- Detection scoring with thresholds based on training data distributions.
- Evaluation of detection scores using metrics defined in `evaluation.py`.
- Logging and reporting of final results.

This ensures modularity, reproducibility, and coherent implementation aligned with the research objectives and experimental setup described in the paper.

---

This comprehensive logic analysis should guide the development of a fully functional `main.py` tailored for replicating the experiments and analysis from the paper.

## model.py

**Logic Analysis for `model.py` – Design and Implementation of Neural Network Models**

---

### Purpose
The goal of `model.py` is to define neural network architectures used as baselines for unsupervised time series anomaly detection, primarily:
- A simple 1-layer feedforward Multilayer Perceptron (MLP) autoencoder (with no activation for reconstruction)
- Potential extension points for other neural network blocks (e.g., Transformer, GCN-LSTM), but the main focus remains on a simple linear MLP.

The module must support:
- Construction of the model object based on dynamic input dimensions.
- Training the model with specified hyperparameters.
- Performing inference on test data.
- Computing and returning error scores used for anomaly detection.

**Note:** The code should rely solely on `torch` (PyTorch), as per requirement.

---

### Key Components & Responsibilities
1. **Model class (SimpleMLP)**
   - Constructor (`__init__`) that initializes layers:
     - Input layer: size matching data dimensionality (`input_dim`)
     - Hidden layer: size 32 (per config)
     - Output layer: single neuron for reconstruction/prediction
   - No activation in the last layer (linear output), as per the paper’s description.
   - Optional: method for resetting parameters for reproducibility.
   
2. **Training Method (`train()`)**
   - Takes training data (features only: `train_data`)
   - Loss function: Mean Squared Error (MSE) – standard for reconstruction tasks.
   - Optimizer: Adam (with learning rate from config)
   - Batch-wise training with batch size 512.
   - Early stopping mechanism (optional, based on patience parameter).
   - Records best model state (if validation is used) or last model state.
  
3. **Inference Method (`predict()`)**
   - Given test data, returns model outputs.
   
4. **Error Computation (`compute_error()`)**
   - Computes per-sample error, e.g., MSE or absolute difference between input and output.
   - Used as anomaly scores in detection phase.
   
5. **Parameter management**
   - Store model hyperparameters (input_dim, hidden_size) as instance variables.
   - Allow saving/loading model state for reproducibility.

---

### Implementation Details & Considerations

**1. Constructor (`__init__`)**
- Receives `input_dim` (dataset feature dimension, possibly dynamic).
- Creates a sequential model:
  - Fully connected layer: `input_dim` → `hidden_size` (e.g., 32)
  - Activation (ReLU): optional but commonly used for nonlinear capacity (not specified in paper, but recommended; note: paper mentions "without any activation" but later describes a simple setting—clarify accordingly).
  - Final layer: `hidden_size` → 1 (for scalar prediction, or- for reconstruction of input features).
- Since the paper suggests "without any activation" for autoencoder, consider just linear layers.

**2. Training (`train()`)**
- Input:
  - `train_data`: numpy array or torch tensor.
  - `epochs`, `batch_size`, `learning_rate`, `early_stopping_patience` (from config).
- Process:
  - Convert train_data to tensor if needed.
  - Use DataLoader for batch processing.
  - Loss: MSELoss.
  - Optimizer: Adam.
  - Loop over epochs:
    - For each batch:
      - Forward pass: model(input).
      - Compute loss with ground truth (autoencoder aims to reconstruct input).
      - Backpropagate, optimizer step.
  - Early stopping:
    - Track validation loss (if validation set is extracted; if not, use training loss).
    - Stop after patience epochs without improvement.
- Save best model state.

**3. Prediction (`predict()`)**
- Input:
  - `test_data` as numpy array.
- Process:
  - Convert to tensor, run model in eval mode.
  - Return output predictions as numpy array (to match subsequent error computation).

**4. Error calculation (`compute_error()`)**
- Input:
  - `test_data` (original features)
- Process:
  - Run through `predict()` to get reconstructed output.
  - Compute per-sample error:
    - For reconstruction: Frobenius norm (or MSE) per sample.
    - For point-wise detection, use per-sample squared error.
- Output:
  - Error scores array, shape `(num_samples,)`.

---

### Additional Notes
- Model should be flexible to handle different `input_dim`.
- Maintain reproducibility by setting seeds globally outside, but ideally allow for reproducibility through fixed initialization or seed parameter.
- Modular design: separate `__init__`, `train()`, `predict()`, `compute_error()`.
- No dropout, batch normalization, or advanced regularizations unless specified.
- Proper device handling (`cpu`/`cuda`) is recommended but outside core logic; typically, the calling script manages device.

---

### Example (pseudocode for `SimpleMLP`)
```python
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, hidden_size)
        self.decoder = torch.nn.Linear(hidden_size, input_dim)
        # No activation, pure linear layers as per paper unless otherwise instructed

    def forward(self, x):
        z = self.encoder(x)
        # Optional: z = torch.nn.functional.relu(z)
        output = self.decoder(z)
        return output
```

### Summary Table

| Method                     | Description                                               |
|----------------------------|-----------------------------------------------------------|
| `__init__(self, input_dim, hidden_size)` | Build the linear autoencoder with specified input dimension and hidden size. |
| `train(self, train_data, ...)` | Train with MSE loss, Adam optimizer, early stopping.      |
| `predict(self, test_data)` | Generate reconstructions or predictions for test data.    |
| `compute_error(self, test_data)` | Calculate per-sample reconstruction/error for anomaly scoring. |

---

This detailed logic analysis ensures the `model.py` is aligned with the paper's methodology, providing clear guidance for an implementation that supports reproducibility and faithful comparison of baselines.

## requirements.txt

# requirements.txt

# Data handling and numerical computations
numpy==1.21.0                   # For array operations, normalization, and sliding window generation
pandas==1.3.0                   # For dataset I/O, data frame manipulations, and timestamp handling
scipy==1.7.0                     # For additional statistical functions if needed

# Machine learning and classical models
scikit-learn==0.24             # For PCA, k-Nearest Neighbors, and threshold calculations based on training data

# Deep learning framework
torch==1.9.0                   # For implementing, training, and inference of neural network models such as SimpleMLP

# Visualization (optional but helpful for debugging and reporting)
matplotlib==3.4.2

# Additional utility (if needed)
# You may include more packages like seaborn or tqdm for progress bars, but core dependencies are above

# Notes:
# - Dataset loading: Implement custom logic within dataset_loader.py based on dataset formats (CSV, binary, etc.).
# - Data normalization: Normalize training data features (e.g., min-max to [0,1]) before training models; apply same normalization to test data.
# - Sliding window: Generate windowed inputs for univariate data; for multivariate data, use per-timestamp feature vectors.
# - Range heuristics: Compute min and max per feature on training data; detect anomalies if test feature values fall outside these bounds.
# - PCA Reconstruction:
#     - Fit PCA on training data, select number of components (default 30 or as per dataset size).
#     - Store PCA model for transforming test data and computing reconstruction error.
# - k-NN:
#     - Fit on training data features (or windowed features for univariate).
#     - Use `sklearn.neighbors` to compute nearest neighbor distances for test points.
# - Neural network:
#     - Define a simple 1-layer MLP with input size equal to feature dimension.
#     - Use ReLU activation, hidden size 32.
#     - Train with MSE loss, Adam optimizer, batch size 512.
#     - Train for 50 epochs with early stopping if validation metrics stagnate.
# - Threshold setting:
#     - Calculate anomaly score threshold on training data (e.g., 95th percentile) by analyzing the distribution of errors (e.g., MSE or error indicator).
# - Evaluation:
#     - Execute on test data:
#         - Generate anomaly scores using each method.
#         - For range-based metrics, determine predicted anomaly segments from binary thresholded predictions.
#     - Compute metrics:
#         - Point-wise F1, Precision, Recall by optimizing threshold (e.g., using sklearn's `precision_recall_curve`).
#         - Range-based F1: calculate intersecting intervals between predicted and true anomalies, then compute precision and recall as per specified formulas.
#         - Use provided evaluation.py functions or custom implementations aligned with the paper's definitions.
# - Reproducibility:
#     - Set random seeds for numpy and torch.
#     - Log hyperparameters, thresholds, and data splits.
#     - Keep consistent data processing pipelines across runs.
# - Additional notes:
#     - Implement modular classes/functions with clear interfaces as per the design plan.
#     - Keep code flexible for dataset variations; thresholds and window sizes are configurable via the YAML.
#     - Ensure all evaluation metrics are computed exactly as described in the paper, including both point-wise and range-wise metrics with proper thresholding and interval calculations.

# End of requirements.txt with dependency notes and implementation considerations.

## trainer.py

{
  "trainer.py": {
    "Purpose": "This module manages the training process for neural network models (e.g., SimpleMLP), including data batching, epoch iteration, optimization, early stopping, and logging. It interacts with the model class to perform forward passes, compute loss, and update model parameters, aligning with hyperparameters and protocols described in the paper and configuration.",
    "Workflow steps": [
      "Initialize trainer with model instance, training data, validation data, and configuration parameters.",
      "Prepare data loaders or inline batching for training and validation datasets.",
      "Define optimizer (e.g., Adam) with specified learning rate from configuration.",
      "Optionally, set a scheduler for learning rate decay if needed.",
      "Set up early stopping criteria based on validation performance and patience parameter.",
      "Run the training loop over epochs:",
      "  - Shuffle and batch training data.",
      "  - For each batch:",
      "    - Forward pass: compute model predictions.",
      "    - Loss calculation: compute MSE between predictions and ground-truth inputs (autoencoder) or targets.",
      "    - Backpropagation: compute gradients via loss.backward().",
      "    - Optimizer step: update weights.",
      "  - After epoch, evaluate on validation data:",
      "    - Compute validation loss or other relevant metrics.",
      "    - Check early stopping criteria.",
      "    - Save model checkpoint if validation improves.",
    ],
    "Key components": [
      "Model: accepts input features, outputs predictions.",
      "Training data: features preprocessed into suitable shape (batch_size, input_dim).",
      "Loss function: MSELoss for reconstruction/update of error errors.",
      "Optimizer: Adam with learning_rate from config.",
      "Early stopping: monitor validation loss or accuracy, stop after 'early_stopping_patience' epochs without improvement.",
      "Logging: track training/validation loss per epoch, possibly print progress.",
      "Checkpointing: save best model weights based on validation metric (optional but recommended)."
    ],
    "Implementation details": [
      "Data batching: implement using torch DataLoader or manual batching with numpy slices. Inputs should be the pre-processed training features.",
      "Data normalization: ensure data is scaled already, or implement normalization in utils.py and pass normalized data.",
      "Model training loop:",
      "  - Set model to train mode.",
      "  - Clear optimizer gradients each batch.",
      "  - Forward pass: predictions = model(batch_input).",
      "  - Calculate loss: loss = criterion(predictions, batch_target).",
      "  - Backward: loss.backward().",
      "  - Optimizer step: optimizer.step().",
      "  - Track average loss per epoch.",
      "Validation:",
      "  - Set model to eval mode.",
      "  - Run predictions on validation set without gradients.",
      "  - Compute validation loss or other metrics.",
      "Early stopping:",
      "  - Compare current validation metric with the best so far.",
      "  - If no improvement for 'early_stopping_patience' epochs, stop training.",
      "Checkpointing:",
      "  - Save model weights when validation improves.",
      "Reproducibility:",
      "  - Set random seeds for numpy, torch, etc., at start.",
      "  - Maintain consistent data shuffling and batching.",
      "Logging:",
      "  - Log loss and, if desired, other metrics per epoch.",
      "  - Report training progress and best epoch."
    ],
    "Edge cases and considerations": [
      "Handling small datasets where batch size equals dataset size.",
      "Early stopping criteria aligning with model convergence behavior.",
      "Potential overfitting prevention where relevant.",
      "Device compatibility: moving data and model to GPU if available.",
      "Handling potential NaNs or instabilities during training."
    ],
    "Summary": "Create a class or function in trainer.py that: initializes with model, training data, validation data, and config; manages batch iteration; performs training steps per epoch; evaluates on validation set; manages early stopping and checkpointing; logs training metrics. The process reinforces best practices for model training, ensures reproducibility, and matches the experimental setup in the paper."
  }
}

## utils.py

# Logic Analysis for utils.py

This utility module serves as a central repository for common functions essential to data handling, preprocessing, normalization, and metric computations in the experimental pipeline for time series anomaly detection as described in the paper. Its design ensures reusability, consistency, and clarity across dataset loading, model training, and evaluation procedures. Below is a detailed breakdown of the core functions, their responsibilities, inputs, outputs, and implementation considerations.

---

# 1. Dataset Loading Functions

## 1.1 load_dataset(path: str) -> dict
- **Purpose:** Load raw dataset files from specified directory.
- **Inputs:**
  - `path`: Path to the dataset directory.
- **Outputs:**
  - Dictionary containing:
    - `'train_data'`: numpy array of training data (shape: [num_train_points, num_features or 1 for univariate])
    - `'test_data'`: numpy array of test data
    - `'train_labels'`: array of labels (only normal in training set, for validation purposes)
    - `'test_labels'`: array of labels (normal=0, anomaly=1) for evaluation
    - `'train_intervals'`, `'test_intervals'`: optional, used for range metrics (list of [(start,end), ...]) if available.
- **Implementation details:**
  - Depending on dataset format, read CSV, TSV, or custom formats.
  - Parse timestamp and feature columns.
  - For datasets with intervals, extract contiguous segments for range metrics.
  - Maintain raw data as numpy arrays or pandas DataFrames for ease.

## 1.2 parse_labels(file_path: str) -> np.ndarray
- **Purpose:** Parse label files if provided separately.
- **Inputs:** Path to label file.
- **Outputs:** numpy array (or list) of per-timestamp labels aligned with data.
- **Usage:** To ensure labels are correctly attached to data samples.

---

# 2. Data Normalization and Standardization

## 2.1 normalize_data(data: np.ndarray, method: str='zscore') -> np.ndarray
- **Purpose:** Normalize or standardize data.
- **Inputs:**
  - `data`: np.ndarray of shape [samples, features].
  - `method`: `'zscore'`, `'median_iqr'`, or `'none'`.
- **Outputs:**
  - Normalized data array.
- **Implementation:**
  - `'zscore'`: subtract mean, divide by std, computed on train data.
  - `'median_iqr'`: subtract median, divide by interquartile range (IQR).
  - `'none'`: return data unchanged.
- **Notes:** For reproducibility, store normalization parameters (mean/std or median/IQR) to apply consistently on test data.

## 2.2 get_normalization_thresholds(scores: np.ndarray, method: str='percentile', percentile: float=95) -> float
- **Purpose:** Compute anomaly detection threshold based on training scores.
- **Inputs:**
  - `scores`: An array of error scores computed on the training data.
  - `method`: `'percentile'`, `'fixed'`, etc.
  - `percentile`: e.g., 95.
- **Outputs:**
  - Threshold value for anomaly detection.
- **Implementation:**
  - `'percentile'`: threshold at the 95th percentile.
  - `'fixed'`: a fixed number (e.g., zero or a domain-specific cutoff).
  - The threshold can be stored externally for consistency.

---

# 3. Windowed Data Generation

## 3.1 generate_windows(data: np.ndarray, window_size: int=4) -> np.ndarray
- **Purpose:** Create windowed feature vectors for univariate or multivariate series, following the paper's ablation (Section 4.3).
- **Inputs:**
  - `data`: 1D or 2D numpy array (shape: [timesteps, features])
  - `window_size`: Integer (default 4)
- **Outputs:**
  - 2D array of shape [num_windows, window_size * features]
- **Implementation:**
  - For each index `t` in `[w, T-1]`, extract data from `[t - window_size + 1, t]`.
  - For multivariate data, flatten into a vector of size `window_size * features`.
  - For univariate, shape becomes [num_windows, window_size].
  - Handle edge cases carefully (start index ≥ window size).

## 3.2 reconstruct_from_windows(windows: np.ndarray, original_length: int, window_size: int=4) -> np.ndarray
- **Purpose:** Reassemble time series from overlapping windows, if needed for certain evaluation steps.
- **Inputs:**
  - `windows`: array generated by generate_windows.
  - `original_length`: length of original series.
- **Outputs:**
  - Reconstructed 1D array aligning with original time series length.
- **Notes:**
  - Overlap averaging or selection can be employed.

---

# 4. Error Score Computation

## 4.1 compute_point_error(model, data: np.ndarray) -> np.ndarray
- **Purpose:** Compute point-wise prediction errors or reconstruction errors for test data.
- **Inputs:**
  - `model`: neural network model with `predict()` method.
  - `data`: input series for inference.
- **Outputs:**
  - Array of error scores (e.g., absolute difference between predicted and true).
- **Implementation:**
  - For autoencoder/forecast model, compute reconstruction or prediction.
  - Error metric: L2-norm or MSE per point.
  - For models outputting vector predictions, compute maximal absolute difference \( \|\cdot\|_\infty \).

## 4.2 compute_range_scores(scores: np.ndarray, intervals: list) -> list
- **Purpose:** Aggregate point scores into range-based anomaly scores.
- **Inputs:**
  - `scores`: sequence of point scores
  - `intervals`: list of true anomaly segments (start, end)
- **Outputs:**
  - List of predicted anomaly intervals based on score thresholding.
- **Implementation:**
  - Threshold score array at the computed threshold.
  - Identify contiguous segments of scores above threshold.
  - Generate list of predicted intervals consistent with ground-truth segments.

---

# 5. Metrics and Evaluation

## 5.1 compute_pointwise_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict
- **Purpose:** Compute precision, recall, F1 for point-wise predictions.
- **Inputs:**
  - `y_true`: ground truth labels (0/1).
  - `y_pred`: predicted binary labels.
- **Outputs:**
  - Dictionary: `{ 'precision': ..., 'recall': ..., 'f1': ... }`.

## 5.2 compute_range_f1(y_true_intervals: list, y_pred_intervals: list) -> dict
- **Purpose:** Compute range-based precision, recall, F1 (following Wagner et al. 2023).
- **Inputs:**
  - Ground truth intervals list.
  - Predicted intervals list.
- **Outputs:**
  - Dictionary with { 'precision': ..., 'recall': ..., 'f1': ... }.
- **Implementation:**  
  - Based on overlap and IOU, following definitions provided.
  - Use thresholds around 0.2 to 0.95 for hit ratio.
  - Calculate intersection over union for overlapping intervals.
  - Use the gamma function (equation from paper) for scoring.

## 5.3 compute_AUPRC(scores: np.ndarray, labels: np.ndarray) -> float
- **Purpose:** Compute area under the precision-recall curve.
- **Implementation:**
  - Use `sklearn.metrics.average_precision_score`.
  - Scores normalized or scaled as needed; thresholds varied automatically.

---

# 6. Storage of Results

## 6.1 save_metrics(results: dict, filename: str)
- **Purpose:** Store evaluation results for reporting.
- **Implementation:**
  - Save as JSON or CSV, include all metrics and thresholds.
  
## 6.2 log_report()
- **Purpose:** Summarize and output all results after experiments.

---

# 7. Additional Implementation Details & Considerations

- **Reproducibility:**  
  - Fix random seed at the start (`np.random.seed()`, `torch.manual_seed()`) for consistent model behavior.
- **Thresholds:**  
  - Compute thresholds from training error scores or validation set using the percentile method.
  - Store thresholds for reproducible inference.
- **GPU/CPU:**  
  - Use CUDA if available; ensure consistent device placement.
- **Data Types:**  
  - Use float32 for data tensors to conserve memory and match deep model expectations.
- **Error Handling:**  
  - Validate data shapes and labels.
  - Handle cases where no anomaly is detected (avoid division by zero in metrics).

---

# Summary

This `utils.py` module provides flexible, modular, and robust utility functions for:
- Loading diverse datasets with labels.
- Normalizing and standardizing data for comparability.
- Generating sliding window features for univariate series, or handling raw series for multivariate.
- Computing error scores from models.
- Aggregating scores into range and point-based anomaly predictions.
- Calculating comprehensive evaluation metrics for both pointwise and range-based detection tasks.
- Ensuring experiment reproducibility and consistent thresholding.

It forms the backbone of the experimental codebase, enabling faithful reproduction and comparison aligned precisely with the methodology and evaluation protocols described in the paper.

---

*Note:* Actual implementations should be designed to allow easy configuration via parameters, ensuring they adhere to experimental choices made in the paper (e.g., window size=4, percentile threshold=95).

