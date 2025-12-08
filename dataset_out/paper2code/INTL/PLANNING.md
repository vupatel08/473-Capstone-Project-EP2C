# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. To reproduce the experiments and methodologies proposed in the paper "MODULATE YOUR SPECTRUM IN SELF-SUPERVISED LEARNING", the plan should cover three core components: (a) the precise methodological steps for implementing the spectral transformation techniques (notably INTL with IterNorm and trace loss), (b) the experimental setup details including datasets, data augmentation, hyperparameters, and training procedures, and (c) the evaluation protocols and metrics. Here is a comprehensive, detailed plan:

---

## 1. Methodology Reproduction Plan

### A. Implementation of Spectral Transformation Framework (ST)
- **Goal:** Modulate the embedding spectrum beyond whitening, seeking functions that avoid collapse.
- **Key Abstract:** For an embedding covariance matrix \(\Sigma\), spectral transformation \(\Phi_{ST} = g(\Sigma) = U g(\Lambda) U^T\), where:
  - \(U\) and \(\Lambda\) are eigenvectors and eigenvalues of \(\Sigma\).
  - \(g(\lambda)\) is a spectral modulation function.
- **Implementation Steps:**
  1. **Compute the covariance matrix \(\Sigma\) of mini-batch embeddings \(\mathbf{Z}\).**
     - Normalize embeddings (centered, if necessary).
     - Use either the empirical covariance \(\Sigma = \frac{1}{m} \mathbf{Z} \mathbf{Z}^T\) (if \(\mathbf{Z}\) is \(d \times m\)) or \(\Sigma = \frac{1}{m} \mathbf{Z} \mathbf{Z}^T\).
  2. **Eigen-decomposition \(\Sigma = U \Lambda U^T\):**
     - Use an efficient eigendecomposition (e.g., `torch.symeig` or `torch.linalg.eigh`).
     - Ensure numerical stability; for small eigenvalues, consider adding \(\epsilon I\).
  3. **Apply spectral modulation \(g(\lambda)\):**
     - For whitening, \(g(\lambda) = \lambda^{-0.5}\).
     - For INTL, use \(g(\lambda) = \lambda^{-p}\) where \(p\) adaptively varies around 0.5.
     - For IterNorm, approximate \(g(\lambda)\) through iterative functions \(f_T(\lambda / tr(\Sigma)) / \sqrt{tr(\Sigma)}\).
     - For the Power functions, implement as per equations, with iterative stability considerations.
  4. **Reconstruct transformed embeddings: \(\widehat{\mathbf{Z}} = U g(\Lambda) U^T \mathbf{Z}\).**
     - Use matrix multiplication order to optimize efficiency.
- **Notes:**
  - Implement iterative functions \(f_T(\cdot)\) (e.g., Newton iterations for IterNorm) with the specified number of iterations \(T\).
  - For trace loss, compute \(\operatorname{trace}(\Sigma_{\widehat{\mathbf{Z}}})\) and add as regularizer, encouraging the spectrum to be uniform.

### B. Implementation of Trace Loss (INTL specific)
- **Objective:** Encourage the covariance eigenvalues to become equal, preventing collapse.
- **Steps:**
  1. After spectral modulation and embedding transformation, compute the covariance matrix \(\Sigma_{\widehat{\mathbf{Z}}}\).
  2. Calculate \(\operatorname{trace}(\Sigma_{\widehat{\mathbf{Z}}})\).
  3. Define trace loss: \(\mathcal{L}_{trace} = \sum_{j=1}^{d} (1 - \Sigma_{\widehat{\mathbf{Z}}}_{j,j})^2\).
     - This pushes the eigenvalues to 1.
  4. Combine with the main alignment (e.g., cosine similarity or normalized MSE) loss.
- **Note:** Use a fixed \(\beta\) coefficient, regress it on batch size as per experimental results, or set empirically.

### C. Integrate with Self-Supervised Framework
- Use a Siamese architecture with two views per sample, with online and target networks.
- The encoder (e.g., ResNet-50 or ViT backbone) outputs embeddings \(\mathbf{Z}_1, \mathbf{Z}_2\).
- Apply spectral transformation to each embedding before calculating the similarity loss.
- Use the proposed spectral modulation plus trace loss as an additional regularizer in the overall SSL objective.

---

## 2. Experimental Setup Details

### A. Dataset Requirements
- **Primary Datasets:**
  - **ImageNet (full or subsets):** For large-scale pretraining.
  - **CIFAR-10 and CIFAR-100:** For low-resource evaluation and ablations.
  - **Optional:** ImageNet-100 for faster experiments.
  - **Additional:** COCO for downstream transfer and detection tasks.
- **Data splits:**
  - For pretraining: standard SSL splits.
  - For evaluation: linear probes and k-NN on validation/test sets.

### B. Data Augmentation
- **Transformations:**
  - Random cropping with scale parameters (see Table 4 parameters).
  - Random flip, color jitter, Gaussian noise, solarization, contrast, saturation, hue adjustments.
  - Multi-crop augmentation: produce multiple views with different crop sizes and the same augmentation parameters (e.g., 4 views: 2 local crops, 2 global crops).
- **Implementation:**
  - Use `torchvision.transforms` and `fast-multicrop` paradigms.
  - Keep augmentation parameters consistent as per tables for each dataset.

### C. Architecture & Model Details
- **Backbones:**
  - ResNet-50, ResNet-18 for baseline.
  - Vision Transformers (ViT-tiny, ViT-small) for ablation experiments.
- **Output Dimensions:**
  - Embedding dimension \(d \approx 4096\) or 8192 for experiments.
  - For ViT, typically 768 or 1024, scaled if needed.
- **Projection Head:**
  - 2-layer MLP with hidden size matching embedding size.
  - Activation: ReLU.
- **Spectral Transformation:**
  - Use Newton’s iteration for IterNorm (T iterations, e.g., T=4).
  - Alternatively, apply power functions \(g(\lambda) = \lambda^{-p}\), with \(p \in [0.45, 0.55]\).
- **Trace Loss:**
  - Add as regularization with coefficient \(\beta\) depending on batch size.

### D. Training Hyperparameters
- **Optimizer:** SGD with momentum (0.9) or Adam.
- **Learning Rate & Schedule:**
  - Base learning rate scaled per batch size (see tables).
  - Warm-up 2 epochs, cosine decay schedule.
  - Initial learning rate ~0.3 for ResNet, adjusted for backbone size.
- **Batch Size:**
  - 256 or larger (up to 4096 for large-scale).
  - For ViT, use smaller batch sizes (e.g., 128, 256).
- **Number of Epochs:**
  - 1000 epochs for CIFAR-scale experiments.
  - 200–400 epochs for ImageNet training.
  - Confirm with table parameters.
- **Regularization:**
  - Weight decay 1e-4.
  - Dropout if used.
- **Spectral iteration T:**
  - 4 (recommended as per paper).

### E. Evaluation Metrics
- **Linear Evaluation:**
  - Train linear classifier on frozen encoder using logistic regression.
  - Measure Top-1 accuracy, Top-5 accuracy.
- **K-NN Classification:**
  - 5-NN classifier on validation embeddings.
- **Transfer Tasks:**
  - Fine-tune on downstream datasets (COCO detection).
  - Measure AP (Average Precision) at different IOU thresholds.
- **Convergence & Stability:**
  - Monitor eigenvalues spectrum.
  - Condition number estimates.
  - Log condition number during training to assess numerical stability.

---

## 3. Practical Details and Clarifications
- **Eigen-decomposition stability:** Handle small eigenvalues carefully—possibly apply epsilon regularization.
- **Newton iteration Tuning:** Use T=4–5 for IterNorm; consider T=1 or 2 for approximations.
- **Trace regularization \(\beta\):** Regress \(\beta\) on batch size as per the empirical relation, or experiment with fixed values (e.g., 0.01).
- **Training Stability:**
  - Be aware that higher iteration T or power functions with \(p\) away from 0.5 might cause numerical instability, especially for small batch sizes.
  - Use intermediate eigenvalue thresholding or eigenvalue clipping if necessary.
  
---

## Summary
- Develop spectral transformation modules that take as input embeddings, compute \(\Sigma\), eigendecompose, modulate eigenvalues via chosen functions (whitening, power law, Newton iteration), and reconstruct embeddings.
- Integrate regularizer (trace loss) to promote spectrum uniformity.
- Use consistent data augmentation as per provided parameters.
- Train with standard SSL framework (Siamese, contrastive-like losses), adding the spectral regularization with empirically guided \(\beta\).
- Perform evaluations on linear probes, k-NN, transfer tasks, and eigenvalue spectrum dynamics.
- Validate robustness by varying batch size, embedding size, and iteration count.

This detailed plan provides a solid blueprint to implement and reproduce the methods and experiments described in the paper with fidelity.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Develop a modular PyTorch-based framework that encapsulates spectral transformation modules—including covariance computation, eigendecomposition, spectral modulation functions (whitening, power laws with iterative Newton methods)—and integrates them seamlessly into a Siamese SSL training pipeline. Use efficient eigendecomposition routines, vectorized operations, and GPU acceleration. Implement the trace loss regularizer as an auxiliary term. Build the training loop with support for multi-view augmentations, batch size variation, and spectral iteration T as hyperparameters. Final code will include a main script that orchestrates data loading, model initialization, spectral processing, training, and evaluation.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "spectral_transform.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(dataset_name: str, params: dict)
        +load_data() -> Tuple[torch.Tensor, torch.Tensor]
    }
    class SSLModel {
        +__init__(backbone: nn.Module, projection_head: nn.Module)
        +encode(x: torch.Tensor) -> torch.Tensor
        +project(emb: torch.Tensor) -> torch.Tensor
    }
    class SpectralTransformer {
        +__init__(T: int, method: str, p: float, epsilon: float)
        +transform(emb: torch.Tensor) -> torch.Tensor
        +compute_covariance(emb: torch.Tensor) -> torch.Tensor
        +eig_decompose(Sigma: torch.Tensor) -> Tuple[Tensor, Tensor]
        +spectral_modulate(Lambda: torch.Tensor) -> torch.Tensor
        +reconstruct_embeddings(U: torch.Tensor, gLambda: torch.Tensor, Z: torch.Tensor) -> torch.Tensor
    }
    class Regularizer {
        +__init__(trace_loss_weight: float)
        +compute_trace_loss(cov: torch.Tensor) -> torch.Tensor
    }
    class Trainer {
        +__init__(model: SSLModel, spectral_transformer: SpectralTransformer, reg: Regularizer, data: Tuple[Data, Data], hyperparams: dict)
        +train_epoch() -> None
        +train() -> None
    }
    class Evaluation {
        +__init__(model: SSLModel, data: Tuple[Data, Data])
        +linear_classification() -> dict
        +knn_evaluation() -> dict
    }
    Main --> DatasetLoader
    Main --> SSLModel
    Main --> SpectralTransformer
    Main --> Regularizer
    Main --> Trainer
    Main --> Evaluation
    Trainer --> SSLModel
    Trainer --> SpectralTransformer
    Trainer --> Regularizer
    Evaluation --> SSLModel
    "
    ,
    "Program call flow": "
sequenceDiagram
    participant C as Main
    participant DL as DatasetLoader
    participant M as SSLModel
    participant ST as SpectralTransformer
    participant R as Regularizer
    participant T as Trainer
    participant E as Evaluation
    C->>DL: instantiate(dataset_name, params)
    DL-->>C: load_data()
    C->>M: instantiate(backbone, head)
    C->>ST: initialize(T, method, p, epsilon)
    C->>R: initialize(trace_loss_weight)
    C->>T: initialize(M, ST, R, data, hyperparams)
    T-->>C: start training loop
    T->>M: encode()
    T->>ST: covariance, eigendecomposition, modulate spectrum
    ST-->>T: transformed embeddings
    T->>R: compute trace loss
    R-->>T: regularization term
    T-->>C: training metrics
    C->>E: evaluate(model)
    E-->>C: report metrics
    "
    ,
    "Anything UNCLEAR": "Clarification needed on the exact spectral modulation function implementations, whether Newton iterations or simple power laws are preferred, and on handling numerical stability for eigen-decomposition on batch covariance matrices with small eigenvalues."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch>=1.10",
        "torchvision>=0.11",
        "numpy==1.21.0",
        "scipy>=1.7",
        "matplotlib",
        "scikit-learn"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class that loads standard datasets (CIFAR-10, CIFAR-100, ImageNet-100) with data augmentation pipelines. It outputs train and validation datasets as tensor pairs. It depends on torchvision.datasets and transforms."
        ],
        [
            "model.py",
            "Defines SSLModel class encapsulating backbone (ResNet or ViT) and projection head (MLP). Includes encode() and project() methods. Depends on torch.nn. Imports backbone architectures from torchvision or timm."
        ],
        [
            "spectral_transform.py",
            "Implements SpectralTransformer class, with methods for covariance computation, eigendecomposition, spectral modulation functions (whitening, power law, Newton iteration), eigenvalue reconstruction, and numerical stability controls. Uses torch.linalg.eigh, custom Newton iteration functions."
        ],
        [
            "regularizer.py",
            "Provides Regularizer class that computes trace loss for covariance matrices to enforce spectrum regularization. Depends on torch functions for trace and tensor operations."
        ],
        [
            "trainer.py",
            "Contains Trainer class that orchestrates training loop: for each batch, encode, spectral transform, combine regularization, compute similarity loss, combine losses, backpropagate, and update models. Depends on model.py, spectral_transform.py, regularizer.py, and torch.optim. Uses Multi-view batches from DatasetLoader."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class performing linear classification and k-nearest neighbor evaluation on features extracted from trained encoder. Uses sklearn metrics and torch for feature extraction. Depends on model.py."
        ],
        [
            "main.py",
            "Entry point. Sets configs, initializes DatasetLoader, models, spectral transformer, regularizer, trainer, and evaluation objects. Manages training epochs, validation, and metrics reporting. Calls all components sequentially. Coordinates hyperparameter setup, dataset loading, training iterations, evaluations, and spectrum logging."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "spectral_transform.py",
        "regularizer.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n\n# No API needed — all execution occurs locally within scripts, configuration via dicts or JSON files.",
    "Shared Knowledge": "The spectral transformation module provides common eigendecomposition, spectrum modulation, and Newton iteration routines used across training and evaluation. Hyperparameters like T, method, p, epsilon are shared by main.py and spectral_transform.py.",
    "Anything UNCLEAR": "Clarify whether to integrate with existing deep learning training frameworks like Lightning or purely custom PyTorch code. Confirm at what frequency eigen-spectrum logs should be saved for debugging."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.3
  batch_size: 256
  epochs: 1000
  warmup_epochs: 2
  schedule: cosine_decay
  weight_decay: 1e-4
  momentum: 0.9
  spe_iteration_T: 4
  spe_method: Newton
  spe_p: 0.5
  trace_loss_weight: 0.01
  spe_epsilon: 1e-5
model:
  backbone: ResNet50
  projection_dim: 8192
  projection_layers: 2
  hidden_dim: 4096
data:
  dataset: ImageNet-100
  dataset_type: image_classification
  dataset_params:
    crop_size: 224
    total_crops: 2
    local_crops: 2
    global_crops: 2
    crop_scale_max: 1.0
    crop_scale_min: 0.08
    augmentation_params:
      brightness: 0.4
      contrast: 0.4
      saturation: 0.2
      hue: 0.1
      color_jitter_prob: 0.8
      horizontal_flip_prob: 0.5
      gaussian_prob: 0.0
      solarization_prob: 0.1
training_strategy:
  use_Ema: true
  use_MultiCrop: true
  spectr_enable: true
  spectr_type: Newton
  T_iter: 4
  p_modulation: 0.5
  trace_loss_weight: 0.01
evaluation:
  dataset: ImageNet-100
  eval_type: linear_classification
  metric: accuracy
```

---

**Note:** Full configuration is available in `planning_config.yaml`
