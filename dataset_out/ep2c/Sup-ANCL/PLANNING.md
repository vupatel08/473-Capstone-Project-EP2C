# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a comprehensive, step-by-step plan for reproducing the methods, experiments, and setup described in the paper "On the Effectiveness of Supervision in Asymmetric Non-Contrastive Learning." The plan is organized into logical sections to clarify the implementation pathway.

---

# 1. **Understanding and Detailing the Methodology**

**Core Objective:**
Implement the proposed supervised Asymmetric Non-Contrastive Learning (ANCL) frameworks, SUPSIAM and SUPBYOL, which incorporate supervised signals into the asymmetric, non-contrastive paradigm, improving representations while avoiding collapse.

### a. Architecture Components
- **Encoder:**
  - For most experiments: Use ResNet-50 (or ViT-Small for specific experiments).
  - For synthetic toy dataset: Use a linear encoder (one linear layer).
- **Projection Head:**
  - MLP with 2 or 3 layers depending on the baseline (e.g., SIMSIAM uses 3 layers, others 2).
  - Output dimension: 128 (typical) or 256 (for BYOL).
- **Predictor (for SIMSIAM and SUPSIAM):**
  - 2-layer MLP, hidden dimension: 4096 (or as specified).
  - No batch norm, activation functions: ReLU.
- **Target branch:**
  - Same as online branch but with optional:
    - Shared parameters (as in SUPSIAM), or
    - Momentum network (as in SUPBYOL, with EMA updates).
  - Model parameters are either tied or momentum updated (EMA).

### b. Loss Functions
- **Standard ANCL (SUPSIAM / SUPBYOL):**
  - **Supervised loss:** Minimize distance between online projection (with predictor) and supervised target features (averaged from class-specific pool).
  - **Self-supervised loss:** Similar to BYOL/SIMSIAM, attraction between two augmented views.
  - **Combined loss:** Convex combination with weight \(\alpha\), balancing supervised and self-supervised components.
- **Supervised component:**
  - For each anchor, select positive samples sharing the same label from the target pool.
  - Use stop-gradient operation to prevent collapse (simulate asymmetry).
  - Sampling: From class-specific queues or prototypes, with hyperparameter \(M\) (number of positives sampled per anchor—e.g., all positives or a fixed subset).
- **Collapse Avoidance:**
  - Use target pools (queues) with fixed size (e.g., 8192) storing features per class.
  - Employ exponential moving average (EMA) for target network parameters in SUPBYOL.
  - Enforce feature normalization (L2).
  - For supervised autoencoding: Incorporate constraints on feature covariance as shown in the paper; this guides the theoretical understanding but implementation may focus on empirical stability via stop-gradient and pooling.

### c. Training Strategy
- **Data Augmentation:**
  - For images: standard augmentations following SimCLR / BYOL: cropping, color jitter, Gaussian blur, flips.
  - For toy data: replace up to 60% of dimensions with the data mean vector.
  - For synthetic toy: custom augmentation as described.
- **Training Protocol:**
  - Minimize combined loss \(\mathcal{L} = \alpha \ell_{ssl} + (1 - \alpha) \ell_{sup}\).
  - \(\alpha\): Tuning parameter affecting intra-class variance.
  - Use optimizer: SGD with momentum 0.9.
  - Use cosine scheduling for encoder and projection head learning rate.
  - Predictor: fixed learning rate (e.g., 0.05), no decay.
  - Number of epochs: 
    - Toy: 200 epochs.
    - ImageNet-100: 200 epochs.
- **EMA parameters (for SUPBYOL):**
  - Initialize EMA at 0.99, increase to 1.0 linearly over epochs.

### d. Theoretical Components
- Implement the supervised loss component as per their formulations:
  - Sample positives from class-specific pools.
  - Normalize features.
  - Compute distances (e.g., squared Euclidean).
  - Combine losses with \(\alpha\) to control intra-class variance.
- Optional: Implement covariance-based regularization if desired for analysis.

---

# 2. **Datasets and Experimental Settings**

### a. Datasets
- **Toy Dataset:**
  - 3 classes, 2048-dim features, means with orthogonal vectors.
  - Generate synthetic data following Gaussian distributions.
  - Label each sample; apply augmentation.
  - Train/test split: 3000 training, 1500 testing.
  - Augmentation: replace ~60% features with data mean vector.
- **ImageNet-100:**
  - Subsample of ImageNet-1K: 100 classes.
  - Use images resized to 224x224.
  - Use official train/validation splits.
- **Downstream Evaluation:**
  - Transfer: linear classifier training on frozen features.
  - Few-shot: N-way K-shot classification with 5 samples per class, multiple episodes.

### b. Target Pool Design
- Implement class-specific queues:
  - Fixed size (e.g., 8192).
  - Store features (projected, normalized).
  - Update via EMA (for SUPBYOL) or via replacement (for SUPSIAM with exact sampling).
- Sampling positives:
  - From pool (all positives, or subset \(M\)).
  - Random or uniform sampling per batch.

### c. Hyperparameters
- Learning rates: 0.03 (ResNet), 0.2 (ViT), 0.15, 0.2, 0.1 for different methods.
- \(\alpha\): tune over \([0,1]\) (e.g., 0, 0.2, 0.5, 0.8, 1).
- Temperature \(\tau\): 0.1 (SimCLR, SUPCON), 0.2 (MoCo), 0.07 (SupMoCo).
- Batch size:
  - 128 (ImageNet), 256 (toy).
- Number of epochs:
  - 200 epochs for ImageNet pretrain.
  - 200 epochs for toy.
- EMA momentum:
  - 0.99 initially, increase linearly to 1.0.
- Dropout / BatchNorm:
  - As per baseline; typically none for projection head predictor.

### d. Evaluation Metrics
- **Transfer Linear Evaluation:**
  - Train a linear classifier on frozen features.
  - Measure Top-1 accuracy.
- **Few-shot classification:**
  - 5-way 5-shot accuracy over multiple episodes.
- **Representation quality:**
  - t-SNE visualizations for intra/inter class variance.
  - Intra-class variance measures as in paper (e.g., \(\tilde{S}_W\)).

---

# 3. **Implementation Details & Practical Considerations**

- **Framework:**
  - Use PyTorch or TensorFlow.
  - Modularize encoder, projection head, predictor.
  - Implement target pool class with support for sampling positives from class-wise queues.
  - Implement EMA update routines.
- **Data Augmentation:**
  - For images: follow standard augmentation pipelines.
  - For toy: dimension masking, adding Gaussian noise.
- **Training Loop:**
  - For each batch:
    - Generate two augmented views.
    - Forward through online branch:
      - Extract features, project, predictor.
    - Forward through target branch:
      - Use EMA parameters for target network.
      - Obtain features, normalize.
      - For supervised loss, sample positives from pool.
    - Compute combined supervised and self-supervised loss.
    - Backpropagate, update online parameters.
    - Update target pools.
    - Update EMA for target network (if applicable).
- **Reproducibility:**
  - Fix random seeds.
  - Run multiple seeds for averages.
  - Log training losses, intra-class variance, and downstream accuracies.

---

# 4. **Experiments and Ablations**

- **Ablation 1:** Effect of \(\alpha\). Train with \(\alpha\) in \(\{0, 0.2, 0.5, 0.8, 1\}\).
- **Ablation 2:** Pool design:
  - All positives vs class-specific vs prototypes.
- **Ablation 3:** Pool size \( |Q| \): 4096, 8192.
- **Ablation 4:** Effect of EMA in SUPBYOL.
- **Hyperparameter tuning:** run grid or random search around the specified values.

---

# 5. **Summary of the Implementation Roadmap**

- **Start with the toy dataset:**
  - Implement synthetic data generation.
  - Basic supervised ANCL with simple linear encoders and MLPs.
  - Verify intra-class variance reduction controls learning.
  - Tune \(\alpha\) for best downstream transfer.
- **Scale to ImageNet-100:**
  - Use ResNet-50 backbone.
  - Implement the full SUPSIAM / SUPBYOL pipelines.
  - Incorporate target pools, EMA, and combining losses.
  - Conduct linear evaluation and few-shot tasks.
- **Visualization and analysis**
  - t-SNE plots of feature space for different \(\alpha\).
  - Metrics for intra/inter class variance.
  - Check collapse avoidance via feature distribution.

---

This plan provides the necessary detailed roadmap to implement, reproduce, and validate the core contributions of the paper, ensuring thoroughness for subsequent coding and experimentation phases.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a simplified yet comprehensive training pipeline leveraging PyTorch and torchvision for dataset loading, augmentation, model definition, and training loop. The core model will include an encoder (ResNet-50 or ViT), projection head, and optional predictor. We will implement the supervised ANCL loss combining self-supervised and supervised components with stop-gradient operations, class-specific feature pools, and EMA updates for the target network. The architecture will modularly separate dataset handling, model instantiation, training, and evaluation, enabling easy hyperparameter tuning and visualization.",
    "File list": [
        "main.py",
        "dataset.py",
        "model.py",
        "loss.py",
        "trainer.py",
        "pool.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "```mermaid\nclassDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class Dataset {\n        +__init__(data_path: str, split: str)\n        +get_train_loader() -> DataLoader\n        +get_val_loader() -> DataLoader\n    }\n    class Model {\n        +__init__(arch: str, out_dim: int)\n        +forward(x: Tensor) -> Tensor\n        +update_target()  # Optional EMA update\n    }\n    class Pool {\n        +__init__(size: int, num_classes: int)\n        +enqueue(features: Tensor, labels: Tensor) -> None\n        +sample_positive(labels: Tensor, M: int) -> Tensor\n            # returns a sampled positive features tensor\n    }\n    class Loss {\n        +__init__(config: dict)\n        +compute(online_proj: Tensor, online_pred: Tensor, supervised_targets: Tensor, alpha: float) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, pool: Pool, loss: Loss, dataloader: DataLoader, config: dict)\n        +train_one_epoch() -> None\n        +train() -> None\n    }\n    class Evaluator {\n        +__init__(model: Model, dataloader: DataLoader)\n        +compute_metrics() -> dict\n        +linear_probe() -> float\n    }\n    Main --> Dataset\n    Main --> Model\n    Main --> Pool\n    Main --> Loss\n    Main --> Trainer\n    Main --> Evaluator\n    Trainer --> Model\n    Loss --> Pool\n    Pool --> features: Tensor\n    features --> sample_positive(labels: Tensor, M: int) -> Tensor\n    Trainer --> Loss\n    Main --> utils\n```",
    "Program call flow": "```mermaid\nsequenceDiagram\n    participant M as Main\n    participant DS as Dataset\n    participant MD as Model\n    participant P as Pool\n    participant L as Loss\n    participant TR as Trainer\n    participant EV as Evaluator\n    M->>DS: instantiate with data paths\n    DS-->>M: train_loader, val_loader\n    M->>MD: instantiate(arch, out_dim)\n    MD-->>M: model instance\n    M->>P: instantiate(pool_size, num_classes)\n    P-->>M: pool instance\n    M->>L: instantiate(loss_config)\n    L-->>M: loss instance\n    M->>TR: instantiate(model, pool, loss, dataloader, config)\n    TR-->>M: trainer instance\n    Note over TR, M: Partially initialize, prepare training loops\n    M->>TR: call train()\n    TR->>TR: for each epoch:\n        - sample batch data\n        - generate two augmented views\n        - forward online branch: encoder + projection + predictor\n        - forward target branch: EMA encoder + projection\n        - sample positives for supervised loss from pool\n        - compute combined loss\n        - backprop, optimize online network\n        - update pool with features and labels\n        - update target network via EMA\n    end\n    M->>EV: instantiate(model, val_loader)\n    EV-->>M: metrics = compute_metrics()\n    EV-->>M: linear_accuracy = linear_probe() \n    M->>Main: output results\n```",
    "Anything UNCLEAR": "Clarification needed on whether to include covariance regularization or focus solely on stop-gradient pooling and EMA updates. Specific pool initialization details (queue sizes, update rules) need confirmation. Also, confirm if synthetic toy data or only image datasets are prioritized for implementation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.1",
        "torchvision==0.13.1",
        "numpy==1.21.0",
        "scipy==1.7.3",
        "matplotlib==3.5.1",
        "tqdm==4.64.1",
        "scikit-learn==0.24.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset.py",
            "Defines Dataset classes for loading synthetic toy data and image datasets, including transformations, train/validation/test splits, and data loaders. It interfaces with torchvision for image augmentation and numpy/scipy for synthetic data. Dependencies: torchvision, numpy, scipy."
        ],
        [
            "model.py",
            "Defines the core model architecture including Encoder (ResNet-50 or ViT for image data, or linear for toy dataset), projection head (MLP), predictor (MLP), and EMA target network if in SUPBYOL mode. Includes normalization layers. Dependencies: torch.nn, torch."
        ],
        [
            "loss.py",
            "Implements the combined supervised and self-supervised loss functions, including the sampling from class pools, stop-gradient operations, loss combination with parameter alpha, and covariance regularizations if applicable. Dependencies: torch, numpy."
        ],
        [
            "pool.py",
            "Implements class-specific feature pools (queues) with enqueue and sampling functions. Supports balanced sampling of positives per class, supports EMA updates if needed. Dependencies: numpy, torch."
        ],
        [
            "trainer.py",
            "Defines the Trainer class which handles one epoch training: data batching, augmentation, forward passes through online and target networks, loss calculation, backpropagation, optimizer step, pool update, and EMA updates for target network. Manages hyperparameters, logs metrics, optional visualization. Dependencies: torch, pool.py, loss.py, model.py, dataset.py."
        ],
        [
            "evaluation.py",
            "Defines evaluation class for linear probing and downstream metric calculation. Implements feature extraction from frozen encoder, linear classifier training, and inference. Dependencies: torch, sklearn."
        ],
        [
            "main.py",
            "Main script which initializes configuration, loads datasets, creates models, pools, losses, trainers, runs training loops, and performs evaluation. Sets hyperparameters, parses arguments, manages experiment flow. Dependencies: all above modules."
        ]
    ],
    "Task list": [
        "dataset.py",
        "model.py",
        "loss.py",
        "pool.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "N/A – This system runs locally for training and evaluation; no API endpoints required.",
    "Shared Knowledge": "Common utilities include normalization functions, EMA update functions, pooling management, and configuration parameters. Hyperparameters are centrally stored in a config dict or YAML file included in main.py.",
    "Anything UNCLEAR": "Clarification needed on whether to support multiple backbones (ResNet/Vit) interchangeably, and whether to implement covariance regularization as an option. Also, confirm if synthetic toy dataset implementation should be included or only image datasets."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.05               # For toy dataset, SGD with cosine schedule
  batch_size: 256                   # Toy dataset batch size
  epochs: 200                       # Number of epochs for toy training and ImageNet-100 pretraining

pretraining:
  dataset: ImageNet-100            # Dataset for pretraining (ResNet-50 backbone)
  optimizer: SGD                   # Optimization algorithm
  weight_decay: 1e-4               # Weight decay regularization
  momentum: 0.9                    # Momentum term
  base_lr: 0.03                    # Learning rate for ResNet-50
  scheduler: cosine                # Cosine learning rate schedule
  total_epochs: 200                # Pretraining epochs

model:
  backbone: ResNet50               # Encoder architecture
  projection_dim: 128              # Dimension of projection head output
  predictor_dim: 4096              # Hidden dimension for predictor
  predictor_layers: 2              # Number of layers in predictor

loss:
  alpha: 0.5                       # Balance between ssl and supervised loss
  temperature: 0.1                 # For contrastive/distances in losses
  pool_size: 8192                  # Size of class feature pools
  sampling_pos: all                # Sample all positives or M positives per class
  covariance_regularization: false # Optional covariance regularization (default false)

pool:
  pool_type: class_specific        # Use class-specific queues for positives
  update_with_ema: true            # Use EMA update for target pool in SUPBYOL
  pool_size: 8192                  # Pool vector size

training_details:
  dataset_loader: torchvision       # Load datasets via torchvision
  augmentations: standard           # Standard augmentation pipeline
  seed: 42                          # Random seed for reproducibility

evaluation:
  downstream_tasks: linear_probe    # Linear classifier on frozen features
  few_shot_tasks: true               # Few-shot classification settings
  metrics: accuracy                 # Metrics to evaluate

dependencies:
  requirements: ["torch==1.12.1", "torchvision==0.13.1", "numpy==1.21.0", "scipy==1.7.3", "matplotlib==3.5.1", "tqdm==4.64.1", "scikit-learn==0.24.2"]
```

---

**Note:** Full configuration is available in `planning_config.yaml`
