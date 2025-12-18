# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a detailed, structured plan to faithfully reproduce the experimental methodology and core concepts from the paper “Reconsidering the Uniformity Metric in Self-supervised Learning.” It covers all key aspects from the methodology to experimental setup, hyperparameters, and evaluation metrics, designed to serve as a blueprint for subsequent implementation.

---

## 1. **Core Methodology Overview**

**Objectives:**
- Replace the classical uniformity metric \(-\mathcal{L}_U\) with a Wasserstein-distance-based uniformity metric \( -\mathcal{W}_2 \).
- The new metric measures the divergence between learned representation distributions and a specified *approximate* uniform spherical distribution.
- Integrate this metric as an auxiliary loss term into existing self-supervised methods (e.g., MoCo v2, BYOL, Barlow Twins) to improve representation quality and reduce collapse phenomena.

**Main Components:**
- **Uniformity Metric:**
  - Measure the Wasserstein distance between the empirical distribution of features and a reference Gaussian (or sphere) distribution.
  - For practical purposes, estimate the mean \(\hat{\mu}\) and covariance \(\hat{\Sigma}\) over representations in a batch, then compute \( -\mathcal{W}_2 \) based on the closed-form formula.
- **Training Objectives:**
  - Combine the existing contrastive/self-supervised loss (e.g., InfoNCE for MoCo, MSE for BYOL, covariance decorrelation for Barlow Twins) with the auxiliary uniformity loss \( -\mathcal{W}_2 \).
  - Use a hyperparameter \(\lambda\) to weight the uniformity loss.
- **Loss Design:**
  \[
  \text{Total Loss} = \text{Existing SSL Loss} + \lambda \times \big( - \mathcal{W}_2(\text{representations}) \big)
  \]
- **Representation Extraction:**
  - Use the encoder network to produce features from augmented views.
  - Normalize features to unit sphere, if specified, to ensure the distribution approximates a uniform spherical distribution.

---

## 2. **Implementation Details**

### 2.1. **Model Architecture**
- **Encoder backbone:** Use ResNet-18 or ResNet-50.
- **Projection head:** MLP (1-2 layers) with specified embedding size (e.g., 128), matching the settings in Table 3.
- **Predictor (for BYOL):** An MLP similar to the projection head.
- **Additional components:** Momentum encoder (for MoCo, BYOL), and batch normalization as needed.

### 2.2. **Data & Datasets**
- **Datasets:**
  - CIFAR-10 and CIFAR-100, for consistency with paper experiments.
  - Download via standard torchvision.datasets or similar.
- **Augmentations:**
  - Random crop, resize, color jitter, flip, Gaussian noise, grayscale, as these are standard in self-supervised learning.
  - Ensure augmentation pipeline is replicated accurately.

### 2.3. **Training Configuration & Hyperparameters**
- **Batch size:** 256 (or 512 as per experiments, depending on hardware).
- **Epochs:** 500 or 1000; refer to the paper’s training schedule.
- **Learning Rate:** Use cosine decay starting from 0.03 or 0.1, with warm-up if needed.
- **Optimizer:** SGD or Adam with momentum 0.9.
- **Temperature for contrastive losses:** \( t=0.2 \) (MoCo), or 0.5-0.7 depending on the specific method.
- **Projection/Prediction dimensions:** 128.
- **Uniformity loss weight \(\lambda\):** tune as hyperparameter; initial value could be 0.1–1.
- **Regularization:** Weight decay 1e-4, BatchNorm parameters as in standard SSL.

### 2.4. **Implementation of the Uniformity Metric \( -\mathcal{W}_2 \)**
- **Step 1:** Collect the features (say, after normalization) from a batch.
- **Step 2:** Compute empirical mean \(\hat{\mu}\) and covariance \(\hat{\Sigma}\).
- **Step 3:** Use the closed-form formula:
  \[
  - \mathcal{W}_2(\hat{\mu}, \hat{\Sigma}) = - \sqrt{\|\hat{\mu}\|_2^2 + 1 + \operatorname{tr}(\hat{\Sigma}) - \frac{2}{\sqrt{m}} \operatorname{tr}(\hat{\Sigma}^{1/2})}
  \]
  *Note:* Approximate \(\operatorname{tr}(\hat{\Sigma}^{1/2})\) via eigen-decomposition or SVD.
- **Step 4:** Implement efficient batching so that doing this per iteration on GPU is possible.

### 2.5. **Auxiliary Loss Integration**
- For each training iteration:
  - Compute features for both views.
  - Normalize features if necessary.
  - Calculate the batch distribution statistics.
  - Compute the \( -\mathcal{W}_2 \) loss.
  - Add weighted term to the main SSL loss.

---

## 3. **Experiments & Evaluation**

### 3.1. **Baseline Methods**
- Use MoCo v2, BYOL, and Barlow Twins as base models.
- For each method:
  - Regular training with the original loss.
  - Training with the added uniformity loss \( -\mathcal{W}_2 \) with a set \(\lambda\).

### 3.2. **Data & Schedule**
- Train for 500 epochs or as in the paper (e.g., 1000 epochs).
- Both CIFAR-10 and CIFAR-100 datasets.
- Fine-tune or evaluate directly on downstream tasks as described:
  - Linear classification (Top-1, Top-5 accuracy).
  - Representation analysis (e.g., singular value spectrum, eigenvalues).
  - Dimensional collapse assessment via singular values.
  - Uniformity metrics over training epochs.

### 3.3. **Key Metrics**
- **Downstream accuracy:** Top-1 accuracy for linear classifiers trained on frozen features.
- **Representation Uniformity:**
  - Compute the KL divergence and Wasserstein distances between the feature distributions and the theoretical Gaussian distribution at multiple training steps.
- **Collapse metrics:**
  - Singular value spectrum of features at different training epochs.
  - Visualizations (e.g., 2D plots) of feature distributions.

### 3.4. **Additional Analyses**
- Ablate uniformity loss weight \(\lambda\).
- Study correlation between \( -\mathcal{L}_U \) and \( -\mathcal{W}_2 \).
- Compare convergence behavior of uniformity during training.
- Visualize features and distribution (Figures 12–17).

---

## 4. **Implementation Checklist**
- [ ] Implement data loading pipeline with CIFAR-10/100 and data augmentations.
- [ ] Define encoder + projection head (plus predictor for BYOL).
- [ ] Implement the main SSL loss (contrastive for MoCo, BYOL loss, covariance decorrelation for Tw).
- [ ] Implement the empirical \(\hat{\mu}\), \(\hat{\Sigma}\) calculation per batch.
- [ ] Implement the closed-form \( -\mathcal{W}_2 \) loss.
- [ ] Integrate the auxiliary \( -\mathcal{W}_2 \) loss with a tunable weight \(\lambda\).
- [ ] Set up training schedule, optimizer, learning rate decay, and evaluation protocols.
- [ ] Save features and metrics during training to analyze uniformity and spectral properties.
- [ ] Visualize singular values, spectra, and distribution overlaps as in the figures.

---

## 5. **Remarks & Clarifications Needed**
- Confirm whether features should be normalized (e.g., \( \ell_2 \) norm to 1) before distribution computation.
- Clarify whether to compute the uniformity loss at every iteration or every certain number of epochs.
- Exact hyperparameters for each SS method (e.g., momentum coefficient for BYOL, queue size for MoCo) as per Table 3.
- Implementation details for the eigen-decomposition of \(\hat{\Sigma}\) or approximations.

---

This comprehensive plan will serve as a roadmap for coding the experiments, ensuring fidelity to the original paper's design, and enabling subsequent analytical and visualization tasks.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a self-supervised learning framework incorporating the Wasserstein-distance-based uniformity loss into existing SSL methods (MoCo v2, BYOL, Barlow Twins). The system will extract feature representations, compute the empirical mean and covariance, and then calculate the uniformity loss using the closed-form formula. This loss, weighted by a hyperparameter, will be added to the standard contrastive or decorrelation losses. We will leverage PyTorch for model implementation, data loading, and training, using torchvision for CIFAR datasets. Eigen-decomposition or SVD will be used for covariance square root approximation. Visualization and metrics computation will be included for analysis, with training loops iterating over epochs, periodically calculating uniformity and downstream accuracy.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "losses.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_name: str, batch_size: int, augmentations: list)\n        +load_data() -> DataLoader\n    }\n    class Model {\n        +__init__(arch: str, projection_dim: int)\n        +encoder: nn.Module\n        +projection_head: nn.Module\n        +predictor: nn.Module (optional, for BYOL)\n        +forward(x: Tensor) -> Tensor\n        +extract_features(x: Tensor) -> Tensor\n    }\n    class RepresentationStatistics {\n        +compute(batch_features: Tensor) -> (Tensor, Tensor)\n            # returns mean and covariance\n        +compute_uniformity_metrics(mean: Tensor, cov: Tensor) -> float\n    }\n    class LossFunction {\n        +__init__(loss_type: str, weight: float)\n        +compute(features: Tensor)\n        +combine_with_base_loss(base_loss: Tensor, features: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, dataloader: DataLoader, loss_fn: LossFunction, optimizer: Optimizer, epochs: int, lambda: float)\n        +train()\n    }\n    class Evaluation {\n        +__init__(model: Model, val_loader: DataLoader)\n        +compute_downstream_accuracy() -> dict\n        +compute_singular_value_spectrum(features: Tensor) -> list\n        +visualize_feature_distribution(features: Tensor) -> None\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Trainer --> Model\n    Trainer --> LossFunction\n    Trainer --> utils\n    Model --> encoder: nn.Module\n    Model --> projection_head: nn.Module\n    Model --> predictor: nn.Module (optional)\n    Trainer --> RepresentationStatistics\n    RepresentationStatistics --> utils\n    Evaluation --> Model\n    Evaluation --> utils",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant Md as Model\n    participant Tr as Trainer\n    participant Ev as Evaluation\n    M->>DL: load_data()\n    DL-->>M: dataset loader\n    M->>Md: initialize(arch, projection_dim)\n    Md-->>M: model object\n    M->>Tr: train(model, dataloader, base_loss, loss_weight, lambda, epochs)\n    Tr->>Md: forward(x)\n    Md-->>Tr: features\n    Tr->>LossFns: compute losses(base_loss, features, lambda)\n    LossFns-->>Tr: total loss\n    Tr-->>M: training loop with periodic uniformity & accuracy eval\n    M->>Ev: evaluate(model, val_loader)\n    Ev->>Md: forward(x)\n    Md-->>Ev: features\n    Ev->>utils: compute_singular_values(features)\n    Ev->>utils: visualize_distribution(features)\n    Ev-->>M: metrics report\n    M-->>End: conclude\n",
    "Anything UNCLEAR": "Clarification needed on the precise normalization step for features—should features be normalized to the sphere before computing mean/covariance? Hyperparameter tuning details for lambda and learning rate schedules are approximate; confirm preferred values. Also, specifics of the augmentation pipeline and eigen-decomposition method for covariance square root approximation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "torchvision==0.12.0",
        "numpy==1.21.0",
        "scipy==1.7.3",
        "matplotlib==3.4.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Class DatasetLoader: Responsible for loading CIFAR-10 and CIFAR-100 datasets with proper augmentations. Handles batching and shuffling. Receives dataset name, batch size, and augmentation list. Outputs PyTorch DataLoader objects."
        ],
        [
            "model.py",
            "Class Model: Defines backbone encoder (ResNet-18 or ResNet-50), projection head, and predictor if needed. Methods include __init__, forward, and extract_features. Encapsulates all neural network components for the SSL models."
        ],
        [
            "losses.py",
            "Class UniformityLoss: Implements computation of the \(-\mathcal{W}_2\) uniformity metric based on empirical mean and covariance. Also defines base SSL loss (e.g., InfoNCE, MSE). Method for computing total loss combining SSL and auxiliary uniformity loss."
        ],
        [
            "trainer.py",
            "Class Trainer: Manages training loop. Initializes with model, data loader, loss function, optimizer, epochs, lambda. Implements train() method, handles per-epoch or per-iteration loss calculation, gradient steps, and periodic evaluation."
        ],
        [
            "evaluation.py",
            "Class Evaluation: Handles downstream linear classification, singular value spectrum analysis, and feature distribution visualization. Methods include evaluate_downstream, compute_spectrum, visualize_distribution."
        ],
        [
            "utils.py",
            "Utility functions: eigen decomposition for covariance square root (e.g., scipy.linalg.eigh), normalization functions, plotting helpers, and metrics computation (e.g., accuracy)."
        ],
        [
            "main.py",
            "Entry point script: Parses configurations, initializes DataLoader, Model, Loss, Trainer, runs training, saves logs, and runs evaluation procedures, including visualization and metrics recording."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "losses.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "N/A",
    "Shared Knowledge": "Feature normalization to unit sphere before covariance calculation. Eigen-decomposition for covariance matrix sqrt. Hyperparameters for models and losses as per the paper (e.g., learning rate, lambda). Consistent data augmentation pipelines for CIFAR datasets across training and evaluation.",
    "Anything UNCLEAR": "Clarification needed on whether to normalize features to sphere before covariance (assumed yes). Exact hyperparameter values recommended for lambda and schedule. Eigen-decomposition method (full eigen or SVD) for covariance square root computation should be confirmed."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.03  # following cosine schedule, initial LR
  batch_size: 256    # standard batch size used in experiments
  epochs: 500        # number of epochs for CIFAR experiments
  optimizer: SGD     # optimizer type
  momentum: 0.9      # momentum value
  weight_decay: 1e-4 # weight decay
  warmup_epochs: 10  # optional, typical in SSL but not specified
  lambda_uniformity: 0.1  # weight for uniformity loss

model:
  backbone: ResNet-18     # or ResNet-50
  projection_dim: 128     # dimension of projection head
  predictor_dim: 128      # used in BYOL
  predictor: true         # whether to include predictor (BYOL)
  use_mlp: true           # whether to use MLP for projection/predictor

dataset:
  name: CIFAR-10          # or CIFAR-100
  augmentations:        # list of data augmentation techniques
    - RandomCrop: [32, 32, 4]
    - HorizontalFlip: true
    - ColorJitter: [0.4, 0.4, 0.4, 0.1]
    - GaussianBlur: 0.5

loss:
  base_loss: InfoNCE    # or MSE for BYOL, Covariance loss for Barlow
  uniformity_lambda: 0.1  # weight for Wasserstein uniformity loss
  tau: 0.2               # temperature for contrastive loss

training_schedule:
  schedule_type: cosine
  warmup_epochs: 10
  min_lr_ratio: 0.001

evaluation:
  downstream_epochs: 100 # epochs for linear evaluation
  evaluation_metric: accuracy
  evaluation_split: test

logging:
  log_dir: ./logs
  save_model_every: 50  # save checkpoint every 50 epochs
```

---

**Note:** Full configuration is available in `planning_config.yaml`
