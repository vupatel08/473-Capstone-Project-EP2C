# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

{
  "dataset_loader.py": "Contains DatasetLoader class responsible for loading standard datasets (CIFAR-10, CIFAR-100, ImageNet-100) with appropriate data augmentation pipelines, returning datasets compatible with training and validation phases as tensor pairs. Implementation depends on torchvision.datasets and torchvision.transforms. The class should support dataset selection via configuration, apply configured augmentation parameters, and provide methods to access train and validation datasets.\n\n**Implementation Details:**\n\n1. **Initialization:**\n   - Accept parameters: dataset name (e.g., 'CIFAR10', 'CIFAR100', 'ImageNet-100'), dataset_params (including crop size, number of crops, crop scales, augmentation parameters).\n   - Store these parameters as class attributes.\n\n2. **Dataset Loading Method (`load_data()`):**\n   - Based on the dataset name, instantiate the appropriate torchvision dataset class:\n     - CIFAR-10: `torchvision.datasets.CIFAR10(root, train=True/False, transform, download=True)`\n     - CIFAR-100: `torchvision.datasets.CIFAR100(...)`\n     - ImageNet-100: Custom dataset loader or subset loading from `torchvision.datasets.ImageFolder` if images are organized in folders.\n   - Configure data transforms according to `dataset_params`:\n     - For each phase (training and validation), define transform pipelines.\n     - Use `torchvision.transforms` for data augmentation, including:\n       - RandomResizedCrop with crop_scale parameters (min=0.08, max=1.0)\n       - RandomHorizontalFlip\n       - ColorJitter with brightness, contrast, saturation, hue\n       - Convert to tensor\n       - Normalize if necessary (e.g., ImageNet mean/std for ImageNet datasets)\n       - Additional augmentations like Gaussian noise, solarization, if specified.\n     - For multi-view generation, during training, generate two augmented views per sample.\n   - Return train and validation datasets wrapped with DataLoader objects, supporting batching.\n\n3. **Support for Multi-view Data Augmentation:**\n   - When the dataset is used for SSL training, ensure each sample yields multiple augmented views per sample.\n   - Possibly implement a custom Dataset class that, given an image, applies augmentation twice to produce two views.\n   - This class should be compatible with DataLoader for batch sampling.\n\n4. **Compatibility & Flexibility:**\n   - DatasetLoader should support dynamic dataset selection based on configuration.\n   - Implement method signatures that allow easy retriggering with different parameters if needed.\n   - Ensure datasets are returned as tuple pairs: (view1, view2) per sample for SSL training.\n\n5. **Download & Caching:**\n   - Use default cache directories for datasets but can be configured.\n   - Support download flag in dataset classes.\n\n6. **Implementation Notes:**\n   - Use parameterized transforms to match configuration:\n     - crop size, crop scales, augmentation probabilities.\n   - For ImageNet-100, load from ImageFolder, optionally filtering to a subset of 100 classes.\n   - Ensure reproducibility by setting seed where applicable.\n\n7. **Output:**\n   - Return datasets ready for DataLoader, ensuring each sample produces multiple views as needed during training.\n\n**Summary:**\n- The class encapsulates dataset selection, augmentation pipeline setup, data downloading, and dataset returning mechanisms.\n- Designed to be flexible for various datasets, with support for standard augmentation parameters.\n- Supports multiple views for SSL training scenarios.\n\nThis structure aligns with the data loading requirements of the overall training pipeline, including generating appropriately augmented pairs for self-supervised SSL, adhering to the dataset and augmentation parameters specified in the configuration file."
}

## evaluation.py

### Logic Analysis for evaluation.py

#### Overall Purpose
- Implement an `Evaluation` class to assess the quality of learned representations (embeddings) obtained from a trained SSL model.
- Support two core evaluation types:
  - **Linear Classification:** train a linear classifier (e.g., logistic regression or linear softmax layer) on frozen features for a fixed number of epochs and evaluate accuracy.
  - **k-Nearest Neighbor (k-NN) Evaluation:** perform classification using k-NN directly on the frozen features without training a dedicated classifier, measuring neighborhood-based accuracy.
- Employ **scikit-learn metrics** for accuracy calculations.
- Extract features via the encoder (backbone + projection head) from the trained model.

---

### Core Components & Steps

#### 1. Initialization (`__init__`)
- Input:
  - `model`: an instance of `SSLModel`, containing the trained encoder (backbone + projection head).
  - `data`: validation/test dataset using the structure provided by `dataset_loader.py` (likely TensorDatasets or DataLoader instances for features and labels).
  - `config`: optional, for specifying evaluation parameters like dataset split, evaluation type, and metrics.
- Tasks:
  - Store the model.
  - Prepare data loader or data tensors for feature extraction and evaluation.
  - Read configuration parameters relevant for evaluation, e.g.:
    - which dataset split to use (validation/test).
    - evaluation type (linear, k-NN).
    - metric (accuracy).
  - Possibly initialize scikit-learn estimators or metrics.

#### 2. Feature Extraction (`extract_features`)
- Input:
  - DataLoader or dataset tensors.
- Operation:
  - Set model to evaluation mode (`model.eval()`).
  - For each batch:
    - Pass input images through the model's encoder (via `model.encode()`), obtaining feature vectors.
    - Store features and ground-truth labels.
- Output:
  - Features tensor/array of shape `[num_samples, feature_dim]`.
  - Corresponding labels tensor/array.

#### 3. Linear Classification (`linear_classification`)
- Preprocessing:
  - Extract features on validation set (or test set).
- Fitting:
  - Use `sklearn.linear_model.LogisticRegression` or `sklearn.linear_model.SGDClassifier`.
  - Train on training features (if available) or directly on validation/test features split.
  - Use specified epochs if alignment with training or prefer a fixed training over features (e.g., logistic regression with default hyperparameters).
- Evaluation:
  - Predict labels on validation/test features.
  - Calculate accuracy (`sklearn.metrics.accuracy_score`).

#### 4. k-Nearest Neighbors Evaluation (`knn_evaluation`)
- Operation:
  - Use the training features as reference and validation/test features as query.
  - Instantiate `sklearn.neighbors.KNeighborsClassifier` with `k=5`.
  - Fit on the reference features.
  - Predict on the query features.
- Evaluation:
  - Compute accuracy similar to above.

#### 5. Metrics and Reporting
- For both evaluations, generate metrics:
  - Top-1 Accuracy.
  - Top-5 accuracy, if applicable.
- Summarize in a dictionary or structured output for logging.

#### 6. Additional Considerations
- Move model to CPU for inference if needed.
- Implement batch processing for large datasets to avoid memory overflow.
- Make the process reproducible by setting seeds if necessary.
- Log spectrum-related diagnostics if requested:
  - Eigenvalues of features or covariance matrices.
  - Condition numbers if relevant, for debugging spectral transformation impacts during evaluation (optional but useful in analysis).

---

### Input/Output Summary
| Input | Details |
|---------|----------|
| `model` | Trained `SSLModel` object with its encoder component. |
| `data` | Validation/Test dataset, with images and labels; could be:
  - DataLoader object, or
  - Tuple of tensors `(images, labels)`. |
| `config` | Evaluation parameters from config.yaml, e.g.,
  - `dataset` (which split: validation/test),
  - `eval_type` ('linear_classification', 'knn'),
  - `metric` ('accuracy').

| Output | Details |
|---------|----------|
| Evaluation metrics | Dictionary with keys like `'accuracy'`, `'top5_accuracy'`, etc. |

---

### Implementation details considerations:
- Ensure `extract_features()` can accept either DataLoader or tensors.
- Use `torch.no_grad()` context during feature extraction.
- For linear evaluation: use `sklearn.linear_model.LogisticRegression` with appropriate parameters (default, or fixed epochs if needed).
- For kNN: use `sklearn.neighbors.KNeighborsClassifier(k=5)`.
- Make sure to load the latest model weights (if not passed directly, assume model is already trained/frozen).
- Keep code modular: separate feature extraction, evaluation routines, and metrics computation.

---

### Handling Uncertainties & Clarifications
- **Evaluation Dataset:** Clarify if evaluation is on validation or test split; assume default to test if unspecified.
- **Evaluation Frequency:** Only for final trained model; no need for intermediate logs.
- **Spectrum Logging:** For spectral diagnostics, create optional parameters to log eigenvalues and condition numbers (if needed). Otherwise, focus solely on accuracy metrics.
- **Device Compatibility:** Default inference on CPU; optionally, enable CUDA inference if available.

---

### Summary
In `evaluation.py`, define `Evaluation` with:
- Initialization accepting model, data, and config.
- `extract_features()` method to run the backbone (and projection) on data.
- `linear_classification()` function fitting a simple logistic/regression on features and returning accuracy.
- `knn_evaluation()` fitting k-NN and returning accuracy.
- Main method `run()` orchestrates feature extraction and calls each evaluation type based on configuration, then returns metrics.

This thorough analysis ensures the module accurately reproduces the evaluation procedures described, supporting faithful reproduction and diagnostic insight.

## main.py

# Logic Analysis for main.py

This script serves as the entry point of the entire training and evaluation pipeline for reproducing the experiments in "MODULATE YOUR SPECTRUM IN SELF-SUPERVISED LEARNING". Its primary responsibilities include:

1. Parsing configurations
2. Initializing dataset loaders
3. Building model architectures
4. Setting up spectral transformation modules
5. Configuring regularizers (trace loss)
6. Setting up optimizer, scheduler, and training strategies
7. Running the pretraining loop with spectral regularization
8. Conducting evaluations (linear classifier, k-NN)
9. Logging and outputting results

Below is a step-by-step logical flow with detailed considerations:

---

## 1. Load and Parse Configurations
- **Input:** `config.yaml` converted into a Python dictionary.
- **Actions:**
  - Extract training hyperparameters: learning rate, batch size, epochs, warmup epochs, schedule, weight decay, momentum, spectral iteration `T`, spectral method, modulation parameter `p`, trace loss weight, epsilon for numerical stability.
  - Extract model parameters: backbone type (`ResNet50`), projection dimension, number of layers, hidden dimension.
  - Extract dataset settings: dataset name (`ImageNet-100`), augmentation parameters, crop sizes, number of crops, etc.
  - Extraction of evaluation type and metrics.

---

## 2. Initialize Dataset Loader
- **Objective:** Load training and validation datasets with the specified data augmentation.
- **Inputs:**
  - Dataset type and parameters from configuration:
    - e.g., `dataset: ImageNet-100`
    - augmentation parameters such as crop scales, brightness, contrast, etc.
- **Actions:**
  - Instantiate `DatasetLoader` class with dataset name and dataset_params.
  - Load datasets: training set for SSL pretraining, validation set for evaluation.
  - Use multi-view augmentation:
    - For training: produce 2 or more views (as specified).
    - For validation: standard resized/cropped images.
- **Output:**
  - Datasets ready for DataLoader, with batched data.

---

## 3. Build Backbone and Projection Head
- **Objective:** Construct the model components.
- **Actions:**
  - Instantiate backbone:
    - e.g., `torchvision.models.resnet50()` or `timm.create_model('resnet50')`
  - Build projection head:
    - Multi-layer MLP with:
      - Input: backbone output feature dim (e.g., 2048)
      - Hidden layers: specified in config (e.g., 4096 units, 2 layers)
      - Output layer: projection_dim (e.g., 8192)
  - Wrap into `SSLModel` class with encode() (for backbone) and project() methods.

---

## 4. Set Up Spectral Transformation Module
- **Objective:** Implement spectral modulation (`g(λ)`) based on config.
- **Actions:**
  - Instantiate `SpectralTransformer`:
    - Pass hyperparameters: `T=4`, method (`Newton`), `p=0.5`, epsilon for numerical stability.
  - Note: Spectral transform includes eigendecomposition, spectral modulation, and reconstruction functions.
  - For `p ≈ 0.5`, this performs whitening.
  - For IterNorm: configure accordingly with Newton iteration.
  - Ensure all functions (spectral modulation) are prepared to accept current eigenvalues, eigenvectors, and input embeddings.

## 5. Set Up Regularizer
- **Objective:** Implement trace loss regularizer.
- **Actions:**
  - Instantiate `Regularizer`:
    - Using trace_loss_weight from config (`0.01`).
  - The regularizer computes the difference of covariances’ eigenvalues from 1 (if needed).

## 6. Initialize Training Components
- **Optimizer:**
  - Use SGD with momentum, weight decay.
  - Learning rate as per config.
- **Learning Rate Scheduler:**
  - Implement cosine decay schedule with warm-up epochs.
  - Compute total steps/epochs.
- **Optional:**
  - EMA (Exponential Moving Average):
    - Initialize `torch.optim` with or without EMA based on config.
    - If used, instantiate a `Shadow` copy of the encoder.
    
## 7. Training Loop
- For each epoch:
  - Iterate over DataLoader:
    - For each batch:
      - **Data Preparation:**
        - Retrieve multiple views per sample (`X1`, `X2`).
        - Transfer to device (GPU).
      - **Encoding & Projection:**
        - Pass views through backbone + projection to get embeddings `Z1`, `Z2`.
      - **Spectral Transformation:**
        - Compute covariance matrix of embeddings (`Sigma = Z Z^T / m`).
        - Eigendecompose: `U, Lambda`.
        - Modulate spectrum via spectral function `g(λ)`:
          - For whitening: `λ^{-0.5}`
          - For INTL: iterative Newton function.
        - Reconstruct transformed embeddings: `Z_hat = U g(Λ) U^T Z`.
      - **Compute Losses:**
        - Main alignment loss: cosine similarity or normalized MSE between views.
        - Regularizer: compute trace loss on covariance of `Z_hat`.
        - Combine losses: `total_loss = alignment_loss + trace_loss_weight * trace_loss`.
      - **Backward Pass:**
        - Compute gradients.
        - Optionally clip gradients to improve numerical stability.
        - Update model parameters.
      - **Logging Eigenvalues / Condition:**
        - Optional: log eigenvalues and condition number for debugging or spectrum monitoring.
- **Learning Rate Adjustment:**
  - Step scheduler at epoch boundaries.

## 8. Validation & Evaluation
- **At specified intervals (e.g., after each epoch or every few epochs):**
  - **Linear Evaluation:**
    - Freeze encoder.
    - Train a linear classifier on the validation set.
    - Measure Top-1, Top-5 accuracy.
  - **K-Nearest Neighbors:**
    - Extract features.
    - Use sklearn's `KNeighborsClassifier` with k=5.
    - Measure accuracy.
  - **Spectrum Logging:**
    - Extract eigenvalues of embedding covariance.
    - Log condition number, eigenvalue distribution.
    - Save for analysis.

## 9. Post-Training Evaluation
- Run final evaluation:
  - Transfer learning on downstream datasets like COCO detection and segmentation if applicable.
  - Log AP scores and other metrics.
- Save model checkpoints, spectrum logs.

---

## 10. Result Reporting and Save Outputs
- Save training logs: loss curves, spectrum evolution.
- Save models: backbone + projection + spectral modules.
- Save evaluation results: classification accuracy, transfer metrics.
- Generate spectrum plots for eigenvalues during training.

---

# Additional Notes & Considerations:
- **Numerical stability:** Regularize covariance eigenvalues with small \(\epsilon\).
- **Eigen-decomposition:** Use `torch.linalg.eigh`; check for ill-conditioning.
- **Newton iterations:** For iterative whitening, implement the fixed T iterations carefully.
- **Hyperparameters Tuning:** Based on empirical results, use the specified `T=4`, regularization coefficient, and spectral method.
- **Logging:** Add spectrum, condition number, and loss metrics periodically for debugging and analysis.

This logic analysis ensures that main.py effectively orchestrates all components, manages hyperparameters, performs training/evaluation, and logs comprehensive diagnostics for reproduction fidelity.

## model.py

**Logic Analysis for model.py — Defining SSLModel class (Backbone + Projection Head)**

---

### 1. **Objective and Role of SSLModel Class**

The `SSLModel` class is designed to encapsulate the core neural network components used in the self-supervised learning framework:

- **Backbone Encoder:** A deep convolutional neural network (ResNet-50) or a Vision Transformer (ViT-tiny, ViT-small) architecture that extracts high-level features from input images.
- **Projection Head:** An MLP (multi-layer perceptron), typically consisting of 2-3 layers, that maps backbone features into a vector space suitable for spectral transformation, similarity calculation, and regularization.
- **Methods:**
  - `encode(x)`: processes raw input images through backbone to produce embeddings.
  - `project(emb)`: passes embeddings through the projection head to generate the final feature vectors for SSL training.

---

### 2. **Inputs and outputs**

- **Input to `encode(x)`:** tensors representing augmented images, shape `[batch_size, channels, height, width]`.
- **Output of `encode(x)`:** feature tensor `[batch_size, feature_dim]`, e.g., 2048 for ResNet-50.
- **Input to `project(emb)`:** features from backbone `[batch_size, feature_dim]`.
- **Output of `project(emb)`:** projection vectors `[batch_size, projection_dim]`, e.g., 8192.

---

### 3. **Design Details**

- **Model Components:**
  - Backbone: Instantiate from torchvision.models or external libraries like `timm`. Needs configuration for selecting model, e.g., ResNet50.
  - Projection Head: Sequential `nn.Linear`, `nn.BatchNorm1d`, `nn.ReLU`, and final linear layer.
- **Encapsulation:**
  - Methods `encode()` and `project()` provide interface for external modules and training loops.
  - The class maintains references to backbone and projection head modules.

- **Implementation Steps:**
  1. **Initialization:**
     - Accept backbone type, projection dimension, number of projection layers, and hidden dimension as parameters.
     - Build backbone architecture based on configuration.
     - Build projection head with variable layers based on `projection_layers`, `hidden_dim`, and `projection_dim`.
  2. **Method `encode(x)`:**
     - Forward pass `x` through backbone.
     - Return feature vectors.
  3. **Method `project(emb)`:**
     - Forward `emb` through the projection head.
     - Return projected features for spectral transform and spectral regularization.

- **Dependency:**
  - Use `torch.nn.Module` as base class.
  - Import necessary backbone architectures:
    - From `torchvision.models` for ResNet50.
    - Optionally, `timm` library if ViT models are used.
  - Use flexible architecture parameters for backbones (e.g., model name).

---

### 4. **Additional Considerations**

- **Preprocessing / Normalization:**
  - Normalization of images handled in data loader.
  - Potential normalization of features before projection (if specified), but typically not required.
- **Device placement:**
  - Model instantiation and method calls should support `.to(device)` for GPU compatibility.
- **Weights Initialization:**
  - Use default PyTorch initializations for backbone and projection layers, unless specified otherwise.
- **Modularity:**
  - Keep backbone and projection head as class attributes.
  - Ability to replace backbone easily (e.g., switching from ResNet to ViT) via configuration.

---

### 5. **Sample class structure**

```python
class SSLModel(torch.nn.Module):
    def __init__(self, backbone_name='ResNet50', projection_dim=8192, projection_layers=2, hidden_dim=4096):
        super().__init__()
        # Initialize backbone according to backbone_name
        # Initialize projection head with (projection_layers - 1) hidden layers and one output layer
    def encode(self, x):
        # Forward pass through backbone
        return backbone_features
    def project(self, emb):
        # Forward pass through projection head
        return projection_vectors
```

---

### 6. **Summary**

- Build a flexible, modular SSLModel class.
- Dynamic backbone instantiation based on configuration.
- A configurable projection head with specified layers and dimensions.
- Simple interface with `encode()` and `project()`.
- Support for multi-device training (GPU/CPU).
- Compatibility with spectral transformation modules (inputs/output feature vectors).

This logic forms the foundation for implementation of `model.py`, ensuring clarity, flexibility, and readiness for integration into the overall training pipeline.

## regularizer.py

{
  "regularizer.py": [
    "Purpose and Responsibility:",
    "The Regularizer class encapsulates the logic for computing the trace loss regularization term used in the INTL training scheme. Its primary role is to compute a scalar loss from the covariance matrix of transformed embeddings, encouraging the eigenvalues to be close to 1 for all spectral dimensions, thereby promoting spectral spread and avoiding collapse.",
    "Inputs:",
    "- A covariance matrix, typically denoted as Sigma, computed from the spectral-transformed embeddings (i.e., output of spectral transformation modules).",
    "- The trace loss weight hyperparameter (trace_loss_weight), controlling the strength of the regularization term relative to the main alignment loss.",
    "Outputs:",
    "- A scalar tensor representing the trace loss value.",
    "Key Operations and Logic:",
    1. **Input Handling:**
       - The class receives the covariance matrix \(\Sigma\) as a torch tensor, expected to be of shape \((d, d)\), where \(d\) is the embedding dimension.
       - The class is initialized with a fixed or adaptive weight for the trace loss, e.g., 0.01, as per configuration.
    2. **Trace Computation:**
       - Using `torch.trace()` (or equivalent), compute the trace of \(\Sigma\), which is the sum of its eigenvalues.
       - The trace regularization encourages the eigenvalues of \(\Sigma\) to approach 1, i.e., the spectrum tends toward an identity matrix scaled by 1.
    3. **Loss Calculation:**
       - For stability and simplicity, the loss is computed as the sum over all spectral dimensions:  
         \(\mathcal{L}_{trace} = \sum_{j=1}^{d} (1 - \Sigma_{jj})^2\).  
       - Alternatively, since \(\Sigma\) is symmetric, the diagonal elements are eigenvalues (they are the eigenvalues iff \(\Sigma\) is diagonalized), but the loss is computed directly on the diagonal elements (not eigenvalues), which simplifies implementation.
    4. **Implementation Details:**
       - No eigen-decomposition is necessary here; instead, use the diagonal elements of \(\Sigma\) directly (`torch.diag()` or `torch.diagonal()`), which is more efficient and numerically stable.
       - The computation involves subtracting 1 from each diagonal element, squaring, and summing.
    5. **Scaling and Regularization Strength:**
       - Multiply the computed sum by the `trace_loss_weight` hyperparameter to control the influence of the regularizer during training.
       - The regularization term is added to the overall loss function, typically combined with the similarity loss (cosine, MSE) in training.py or trainer.py.
    6. **Batch Handling:**
       - The covariance matrix Sigma is assumed to be precomputed or provided for the current batch of embeddings.
       - Ensure that the covariance matrix is regularized for numerical stability:
         - Possibly add a small epsilon (like 1e-5) to the diagonal if not already regularized.  
         - The epsilon can be incorporated within this class, either during initialization or before calling `compute_trace_loss()`.
    7. **Design and Usage Integration:**
       - The class should provide a method, e.g., `compute()` or `__call__()`, accepting the covariance matrix and returning a scalar.  
       - The class should be easily integrated into the training loop, called after spectral transformation module outputs the covariance matrix.
       - The computed loss from Regularizer is scaled by the hyperparameter and added to the main alignment loss during backpropagation.
    8. **Additional Details:**
       - Optionally, implement validation or checks on the covariance matrix for positive definiteness or regularization.
       - For logging or debugging, may expose intermediate values such as the diagonal elements, eigenvalues, or the trace value.
  ],
  "Implementation notes": [
    "The class will have an __init__ method accepting `trace_loss_weight`, and possibly an `epsilon` value for numerical stability.",
    "The main method, say `compute()`, takes as input the covariance matrix (`Sigma`) and returns the scalar regularization loss.",
    "Use `torch.diagonal()` or equivalent to extract the diagonal elements of Sigma.",
    "Subtract 1, square, and sum over all diagonals. Scale by `trace_loss_weight`."
  ],
  "Error handling and stability": [
    "Ensure the covariance matrix input is symmetric and positive semi-definite, though in practice \(\Sigma\) constructed from embeddings should be symmetric.",
    "Add epsilon (if needed) to the diagonal to prevent potential numerical issues in the main training loop, especially when \(\Sigma\) has small eigenvalues."
  ],
  "Summary": [
    "The core logic encapsulated: compute diagonal of covariance matrix, measure deviation from unit eigenvalues via \((1 - diag(\Sigma))^2\), sum, scale by hyperparameter, output scalar loss.",
    "This scalar loss is then minimized jointly with the spectral transformation alignment loss, enforcing the spectrum to spread and avoid collapse."
  ]
}

## spectral_transform.py

{
  "file": "spectral_transform.py",
  "overview": "This module implements the core spectral transformation logic, providing methods for covariance matrix calculation, eigendecomposition, spectral modulation (whitening, power functions, Newton iteration), eigenvalue reconstruction, and numerical stability handling. It is designed to be used during training to transform embeddings, regulate the spectrum, and support the trace loss regularization. The implementation must align precisely with the paper's methods and the specified configuration parameters, ensuring compatibility with the overall SSL framework.",
  "classes": {
    "SpectralTransformer": {
      "attributes": {
        "T": "Number of iterations T, default 4, as per config, for iterative Newton methods.",
        "method": "Spectral modulation method: 'Newton' (for IterNorm) or 'Power' (for power law functions).",
        "p": "Power modulation parameter in [0.45, 0.55], typically around 0.5, for spectrum modulation.",
        "epsilon": "Small positive value for numerical stability, default 1e-5.",
        "eig_decomposition_method": "Choosen eigendecomposition routine, e.g., torch.linalg.eigh for symmetric matrices."
      },
      "methods": {
        "compute_covariance": {
          "inputs": "embeddings tensor of shape (d, m).",
          "outputs": "covariance matrix Sigma (d, d).",
          "logic": "Calculate empirical covariance as: Sigma = (1/m) * Z Z^T. For stability, add epsilon * I if necessary."
        },
        "eig_decompose": {
          "inputs": "covariance matrix Sigma.",
          "outputs": "U (eigenvectors), Lambda (eigenvalues).",
          "logic": "Perform eigen-decomposition ensuring numerical stability. Use torch.linalg.eigh for symmetric matrices."
        },
        "spectral_modulate": {
          "inputs": "Lambda eigenvalues tensor, the modulation method: 'whitening', 'power', 'Newton'.",
          "outputs": "gLambda, the modulated eigenvalues after applying g(λ).",
          "logic": "Apply chosen spectral function: \n- For whitening: g(λ) = λ^{-0.5}\;\n- For power law: g(λ) = λ^{-p} with p close to 0.5.\n- For Newton iteration: g(λ) approximated via iterative functions (see below)."
        },
        "apply_iter_norm": {
          "inputs": "covariance matrix Sigma.",
          "outputs": "transformed embedding \(\hat{Z}\).",
          "logic": "Use Newton's method for iterative normalization: \n- Initialize P_0=I.\n- For k in [1,T], compute: \n  P_k = (3/2)*P_{k-1} - (1/2)*P_{k-1}^3 * Sigma_N, where Sigma_N = Sigma / tr(Sigma).\n- Final approximate whitening matrix: \(\Phi_T = P_T / \sqrt{tr(Sigma)}\)."
        },
        "compute_eigenvalues_transform": {
          "inputs": "eigenvalues Lambda, method='power' or 'whitening', parameter p if needed.",
          "outputs": "gLambda.",
          "logic": "For 'power' method, g(λ)=λ^{-p}. For 'whitening', g(λ)=λ^{-0.5}. For 'Newton', use iterative function f_T(λ / tr(Sigma))/√tr(Sigma)."
        },
        "reconstruct_embeddings": {
          "inputs": "U (eigenvectors), gLambda (spectrally modulated eigenvalues), Z (original embeddings).",
          "outputs": "transformed embeddings: \(\hat{Z} = U g(\Lambda) U^T Z\).",
          "logic": "Matrix multiplication: \(\hat{Z} = U (diag(g(\Lambda))) U^T Z\), with g(Λ) as diagonal matrix."
        },
        "log_eigenvalues": {
          "inputs": "current eigenvalues.",
          "logic": "Optional: log or return spectrum for debugging, spectrum regularity checks."
        }
      }
    }
  },
  "functions": {
    "NewtonIteration": {
      "inputs": {
        "Sigma": "Covariance matrix.",
        "T": "Number of iterations.",
        "epsilon": "Stability constant."
      },
      "logic": "Initialize P_0=I. For k in [1,T], compute:\n P_k = (3/2) * P_{k-1} - (1/2) * P_{k-1}^3 * Sigma_N,\n where Sigma_N = Sigma / tr(Sigma). \nReturn P_T / sqrt(tr(Sigma)). This approximates Sigma^{-0.5}."
    },
    "spectral_function_power": {
      "inputs": {
        "Lambda": "Eigenvalues vector.",
        "p": "Power parameter, e.g., 0.5.",
        "epsilon": "for numerical stability"
      },
      "logic": "Apply element-wise power: λ^{-p}, with optional stability adjustments."
    },
    "spectral_function_whitening": {
      "inputs": "Eigenvalues Lambda.",
      "logic": "g(λ)=λ^{-0.5}, with numerical stability enhancement if needed."
    }
  },
  "flow": [
    "Given input embeddings Z, enforce covariance matrix Sigma via compute_covariance.",
    "Perform eigendecomposition: U, Lambda = eig_decompose(Sigma).",
    "Depending on selected method, compute g(λ) via spectral_modulate: whitening, power law, Newton iteration.",
    "Reconstruct transformed embeddings \(\hat{Z}\) using apply_iter_norm (for Newton) or eigenvectors and eigenvalues.",
    "Output the spectral-transformed embeddings for downstream loss calculations.",
    "Optional: log spectrum metrics or eigenvalues for monitoring.",
    "Ensure numerical stability via epsilon adjustments at eigendecomposition and iteration steps."
  ],
  "special notes": [
    "Eigen-decomposition should handle near-zero eigenvalues by adding epsilon * I to Sigma if needed.",
    "Newton iteration method is a key part of approximating whitening without explicit eigen-decomposition in iterative normalization.",
    "For p close to 0.5, expect spectrum to be modulated towards a condition number close to 1.",
    "Eigenvalue regularization or eigenvalue clipping may be necessary to avoid NaN or instability.",
    "Design the methods to be compatible with batched inputs during training."
  ],
  "uncertainties": [
    "Exact choice of eigenvalue regularization parameters (epsilon).",
    "When to log spectrum for debugging purposes.",
    "Handling ill-conditioned matrices that cause eigen-decomposition failures.",
    "Implementation of iterative function f_T in Newton iteration — whether to include adaptive T or fixed T."
  ],
  "final considerations": [
    "Careful implementation of eigen-decomposition and Newton iterations for efficiency and stability.",
    "Shared utility functions for eigenvalue modulation, eigen-decomposition, and matrix synthesis.",
    "Compatibility with the training pipeline: input embeddings, output transformed embeddings, and spectrum regularization.",
    "Flexibility to switch between whitening, power law, and Newton methods via configuration parameters."
  ]
}

## trainer.py

{
  "trainer.py": [
    {
      "component": "Initialization",
      "description": "Define the Trainer class which will orchestrate the self-supervised learning training process, integrating model, spectral transformation, regularizer, optimizer, and evaluation components.",
      "details": [
        "Constructor (__init__):",
        "  - Accepts instances of SSLModel, SpectralTransformer, Regularizer, datasets, and hyperparameters (from config.yaml).",
        "  - Initializes optimizer (e.g., SGD or Adam) with model parameters and training hyperparameters (learning rate, weight decay, momentum).",
        "  - Sets iteration count T, spectral method, modulation parameter p, epsilon for numerical stability, and trace loss weight.",
        "  - Prepares data loaders for training data with multi-view batches, as per DatasetLoader."
      ]
    },
    {
      "component": "Training Loop",
      "description": "Implement a method (e.g., train()) which runs over all epochs; within each epoch, execute batch-by-batch training, including forward pass, spectral transformation, regularization, and backpropagation.",
      "details": [
        "For each epoch:",
        "  - Set model to training mode.",
        "  - For each batch (multi-view):",
        "    - Obtain two augmented views:Z1, Z2 from DatasetLoader (already preprocessed).",
        "    - Forward pass: pass each view through encoder and projection head to generate embeddings Z1, Z2.",
        "    - Inject embeddings into spectral_transform to compute spectral modulated embeddings:" ,
        "      - Compute covariance matrices (\Sigma1, \Sigma2) for Z1 and Z2.",
        "      - Perform eigendecomposition: U1, Lambda1; U2, Lambda2.",
        "      - Apply spectral modulation g(\lambda) = \lambda^{-p} or via Newton T-iteration (depending on method).",
        "      - Reconstruct the whitened (or spectrum-modulated) embeddings: \\hat{Z}_1, \\hat{Z}_2.",
        "    - Compute the spectral regularizer (trace loss):",
        "      - Calculate covariance matrices of \\hat{Z}_1, \\hat{Z}_2.",
        "      - Compute trace loss: sum over (1 - diagonal elements)^2, to encourage eigenvalues to equal 1.",
        "    - Compute the main SSL similarity loss:",
        "      - Typically negative cosine similarity or normalized MSE between the normalized \\hat{Z}_1 and \\hat{Z}_2.",
        "    - Combine total loss:",
        "      - Total loss = similarity loss + beta * trace loss (beta derived from batch size or fixed).",
        "    - Backpropagation:",
        "      - Zero optimizer gradients.",
        "      - Backward through combined loss.",
        "      - optimizer.step().",
        "    - Optional: update any momentum/EMA target network if used.",
        "  - Log spectrum metrics, eigenvalues, condition number for debugging if needed."
      ]
    },
    {
      "component": "Spectral Transformation Integration",
      "description": "Within each batch, for each view:",
      "details": [
        "Implement spectral transformation steps in a dedicated method:",
        "- Compute covariance matrix of embeddings.",
        "- Eigen-decompose the covariance matrix: U, Lambda.",
        "- Apply spectral modulation g(\\lambda):",
        "    - For whitening: \\lambda^{-0.5}",
        "    - For INTL approximation: iterative Newton method with T iterations.",
        "    - For power law modulation: \\lambda^{-p} with p ~ 0.5, with stability checks.",
        "- Reconstruct the modulated embeddings as \\hat{Z} = U g(\\Lambda) U^T Z.",
        "Ensure numerical stability by:",
        "- Regularizing covariance matrices with epsilon if eigenvalues are near zero.",
        "- Using stable eigen-decomposition routines.",
        "- Clipping or thresholding eigenvalues if needed.",
        "For Newton iteration, implement the iterative functions precisely, according to the T specified.",
        "Store or log eigenvalues and condition number periodically to verify spectral properties."
      ]
    },
    {
      "component": "Computing Losses and Regularization",
      "description": "Throughout each batch:",
      "details": [
        "Similarity Loss:",
        "- Compute normalized embeddings (L2 norm).",
        "- Use cosine similarity or normalized MSE between \\hat{Z}_1 and \\hat{Z}_2.",
        "- Compute the negative cosine similarity loss (or equivalent) and average over batch.",
        "Trace Loss:",
        "- Compute covariance matrices of \\hat{Z}_1 and \\hat{Z}_2 after spectral modulation.",
        "- Calculate the diagonal elements (variances/eigenvalues).",
        "- Formulate trace loss as sum over (1 - diagonal elements)^2.",
        "- Weight trace loss with parameter (trace_loss_weight / beta).",
        "Total Loss:",
        "- Sum of similarity loss + (trace_loss_weight * trace loss).",
        "- Use this total loss for backpropagation."
      ]
    },
    {
      "component": "Backpropagation and Optimization",
      "description": "Perform standard PyTorch training step:",
      "details": [
        "Zero optimizer gradients.",
        "Loss.backward().",
        "optimizer.step().",
        "Optional: Update target network (EMA) if used with decay coefficient.",
        "Compute and log metrics (loss values, spectrum measures) for diagnostics."
      ]
    },
    {
      "component": "Validation and Evaluation",
      "description": "Periodically evaluate encoder representations:",
      "details": [
        "Freeze encoder network weights.",
        "Extract features for validation set.",
        "Train linear classifier for a fixed number of epochs and measure accuracy.",
        "Compute k-NN classification accuracy on features.",
        "Perform downstream transfer tasks if specified.",
        "Log spectrum statistics and eigenvalues to monitor spectrum modulation effectiveness.",
        "Optionally, save models/checkpoints."
      ]
    },
    {
      "component": "Additional Details",
      "description": "Ensure robustness and numerical stability:",
      "details": [
        "Handle small eigenvalues in eigendecomposition, e.g., add epsilon or apply regularization.",
        "Limit maximum eigenvalues during Newton iterations to prevent divergence.",
        "Adjust T (number of Newton iterations) based on stability observations.",
        "Set eigenvalue clipping thresholds if needed.",
        "Log spectrum diagnostics regularly to ensure the embedding spectrum approaches the target (eigenvalues close to 1 and well-conditioned)."
      ]
    }
  ],
  "summary": "The trainer.py logic involves initializing components, executing a per-batch procedure that includes encoding, spectral modulation via eigendecomposition (with options for whitening, power laws, or Newton iteration), reconstructing embeddings, calculating both similarity and spectrum regularization losses, backpropagating, and updating parameters. Throughout training, spectrum diagnostics are logged for stability and effectiveness validation, with provisions for handling numerical issues encountered during eigen-decomposition or Newton iterations."
}

