# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Below is a comprehensive, step-by-step plan to reproduce the CARE methodology, including the core theoretical principles, model design, training procedures, datasets, hyperparameters, and experimental evaluation. This plan emphasizes fidelity to the paper's methods while providing sufficient detail for implementation and reproducibility.

---

# Step 1: Understand and Formalize Core Theoretical Principles

**Main Ideas:**
- **Rotational Equivariance in Embedding Space:** Augmentations in input space (e.g., cropping, jittering, color shifts) should correspond to *orthogonal (rotation/reflection)* transformations of the embedding.
- **Loss Function Design:** Enforce *pairwise angle preservation* between augmented samples to promote equivariance, combined with non-collapse (uniformity) and (optionally) invariance terms.
- **Orthogonality and Group Action:** For the transformation of embeddings, the map \( T_a \) for augmentation \( a \) is constrained to be a member of \( O(d) \) (the orthogonal group). The theoretical results show under ideal conditions, augmentation relates directly to a rotation matrix \( R_a \).

**Key Formal Tasks:**
- Derive and implement the **equivariance loss** \( \mathcal{L}_{equiv}(f) \) that aligns cosine similarities of pairs \( (x, a(x)) \).
- Ensure the **group structure** is respected: compositions of augmentations correspond to compositions of orthogonal transformations.
- Extend to approximate equivariance when the loss isn't exactly zero.

---

# Step 2: Model Architecture and Representation Space

**Model backbone (*f*):**
- Use a deep neural network encoder—preferably ResNet-50 for images (as used in experiments), but adaptable.
- Final feature layer should output vectors in \( \mathbb{R}^d \) where \( d \approx 128 \) or similar.
- Normalize final embeddings to unit norm (sphere constraint), facilitating angle-based loss and theoretical guarantees.

**Optional components:**
- A 2-layer MLP projection head for contrastive learning (matching standard approaches like SimCLR, MoCo).
- For the protein examples, use DeepSet architectures to ensure permutation invariance per data modality.

---

# Step 3: Loss Design

**3.1: Contrastive Loss (InfoNCE) \( \mathcal{L}_{InfoNCE}(f) \):**
- Implement as standard, with batch-wise negative sampling.
- Hyperparameters:
  - Temperature \( \tau \) (e.g., 0.1 to 0.5).
  - Batch size (e.g., 256 for images, adjusted if resources are limited).

**3.2: Equivariance Term \( \mathcal{L}_{equiv}(f) \):**
- For input pair \( (x, a(x)) \):
  - Compute embeddings \( z_x = f(x) \), \( z_{a} = f(a(x)) \).
  - Enforce that \( z_{a} \) is a rotation/projection of \( z_x \):
    \- Cosine similarity-based loss:
    \[
    \mathcal{L}_{equiv} = \mathbb{E}_{a, x} \left[ f(a(x'))^\top f(a(x)) - f(x)^\top f(x') \right]^2
    \]
  - To approximate orthogonal transformations, follow the paper's derivation, which minimizes the difference between the inner products of pairs before and after augmentation.
  - Use multiple augmentations to ensure the learned maps correspond to consistent \( R_a \in O(d) \).

**3.3: Non-collapse (Uniformity) \( \mathcal{L}_{unif}(f) \):**
- Penalizes collapsing embeddings:
  \[
  \mathcal{L}_{unif} = - \log \mathbb{E}_{x, x'} \exp(f(x)^\top f(x'))
  \]
- Ensures embeddings are spread over the sphere.

**3.4: Total Loss:**
\[
\mathcal{L}_{CARE} = \mathcal{L}_{inv} + \mathcal{L}_{unif} + \lambda \mathcal{L}_{equiv}
\]
- \( \mathcal{L}_{inv} \) can be approximated via the invariance loss (e.g., encouraging representations of \( x \) and \( a(x) \) to be close if small rotations are desired). Alternatively, omit or adjust depending on the experimental goal.
- Hyperparameter \( \lambda \): small (e.g., 0.001 to 0.01), tuned empirically.

---

# Step 4: Data Augmentation and Group Action Modeling

**Input Augmentations:**
- For images: cropping, jittering, color shifts, Gaussian blur—matching typical contrastive learning pipelines.
- For protein structures: random 3D rotations (SO(3) group actions).

**Augmentation Sampling Strategy:**
- Sample \( a \sim \mathcal{A} \) (e.g., a set of random crops, jitterings).
- For equivariance loss, select a small subset of augmentation functions per batch (e.g., 4, 8, or 16 splits) to reduce variance and stabilize training.

**Implementation Tip:**
- Use consistent random seeds or augmentation parameters to ensure reproducibility.
- For small rotations, generate rotation matrices close to the identity; for large rotations, sample uniformly from SO(3).

---

# Step 5: Model Training Pipeline

**5.1: Data Loading**
- Use datasets: CIFAR-10, CIFAR-100, STL10, ImageNet100 for images.
- For proteins: use Protein Data Bank (PDB) structures dataset.
- Implement custom data loaders that yield:
  - Original samples
  - Augmented pairs (x, a(x))
  - Mini-batches split into chunks for equivariance sampling.

**5.2: Optimization**
- Optimizer:
  - Adam (e.g., learning rate \(1e^{-3}\)) for small datasets.
  - SGD with cosine scheduling for larger datasets.
- Learning rate, weight decay, number of epochs as per the paper.
- Use gradient clipping if needed to stabilize equivariance loss.

**5.3: Batch Management**
- For invariant contrastive loss, large batches (e.g., 256–512).
- For equivariance loss, smaller batch splits (e.g., 4–16) as suggested.
- Use multiple augmentation samples per batch (via batching or repeated sampling).

**5.4: Training Loop**
- For each batch:
  - Compute contrastive loss \( \mathcal{L}_{InfoNCE} \).
  - Compute equivariance loss \( \mathcal{L}_{equiv} \) over augmented pairs.
  - Compute non-collapse \( \mathcal{L}_{unif} \).
  - Combine into total loss, backpropagate, and update parameters.
- Periodically evaluate the embeddings on downstream tasks.

---

# Step 6: Experimental Protocols and Hyperparameters

**6.1: Datasets**
- Image datasets:
  - CIFAR10/100, STL10: standard splits.
  - ImageNet100 (subset of ImageNet): for bigger scale.
- Protein structures:
  - Use the PDB dataset, structured as point clouds.

**6.2: Hyperparameters:**
- Temperature \( \tau \): e.g., 0.1–0.5.
- Equivariance weight \( \lambda \): 0.001 – 0.01.
- Batch splits for equivariance: 4, 8, 16 based on dataset.
- Augmentation parameters (crop size, jitter magnitude, rotation angles): align with paper's choices.
- Training epochs: 200 to 400 depending on dataset.
- Embedding dimension \( d \): 128 or 2048 internally, projected to 128.

**6.3: Evaluation**
- Linear probes: Train a linear classifier on frozen features afterwards.
- Metrics:
  - Top-1 accuracy on linear classification.
  - Measure of equivariance: cosine similarity distribution and rotation trajectory plots.
  - Downstream tasks (e.g., protein property prediction, image classification).

---

# Step 7: Reproducibility and Validation

- Save checkpoints, hyperparameter settings, augmentation parameters.
- Plot trajectories of embeddings during training for qualitative validation.
- Record metrics over training epochs, particularly the evolution of equivariance loss and invariance.

---

# Summary of the Plan

| Step | Focus | Details/Notes |
|---|---|---|
| 1 | Theoretical Foundation | Implement \( \mathcal{L}_{equiv} \), confirm group properties. |
| 2 | Model Architecture | Use a normalized embedding space, scalable backbone. |
| 3 | Loss Functions | Combine contrastive + equivariance + uniformity, tune \( \lambda \). |
| 4 | Augmentation | Input-dependent, small/large rotations, consistent sampling. |
| 5 | Training | Large batch for contrast, split batch for equivariance, optimizer choices. |
| 6 | Hyperparameters | Temperature, batch splits, epochs, learning rates, validation metrics. |
| 7 | Evaluation | Linear probe accuracy, embedding trajectory visualization, downstream tasks. |
| 8 | Reproducibility | Save configs, provide visualization codes, document training details. |

---

This roadmap captures the core insights, the practical implementation steps, and the experimental procedures required for reproducing the CARE methodology as described in the paper. It provides the detailed blueprint needed before translating into actual code.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular Python system based on PyTorch and related open-source libraries. The core components include data loading modules for images and proteins, a model architecture (residual network or DeepSet depending on modality), specialized loss functions implementing the contrastive, equivariance, and uniformity terms, and a training loop that efficiently manages batch splits for equivariance. We will leverage torchvision.transforms or Albumentations for data augmentations, PyTorch's nn.Module for model definitions, and torch.optim for optimizers. Visualization of embedding trajectories and cosine similarity histograms will be included as evaluation utilities. The system will be configured via a JSON or YAML config file to specify datasets, hyperparameters, augmentation parameters, and training settings. Checkpointing and logging will be incorporated for reproducibility.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "loss.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config_path: str)
        +run_experiment() -> None
    }
    class DatasetLoader {
        +__init__(dataset_name: str, config: dict)
        +load_data() -> Dataset
    }
    class Model {
        +__init__(model_type: str, hyperparams: dict)
        +forward(x: Tensor) -> Tensor
        +load_checkpoint(path: str) -> None
        +save_checkpoint(path: str) -> None
    }
    class LossFunctions {
        +__init__(config: dict)
        +contrastive_loss(z1: Tensor, z2: Tensor) -> Tensor
        +equivariance_loss(z_x: Tensor, z_a_x: Tensor) -> Tensor
        +uniformity_loss(z: Tensor) -> Tensor
        +total_loss(...) -> Tensor
    }
    class Trainer {
        +__init__(model: Model, optimizer: Optimizer, loss_fn: LossFunctions, dataloader: Dataset)
        +train() -> None
        +adjust_hyperparams(hp: dict) -> None
    }
    class Evaluation {
        +__init__(model: Model, dataloader: Dataset)
        +compute_linear_probe() -> dict
        +visualize_embeddings() -> None
        +compute_trajectory(x: Tensor, a_list: List[Callable]) -> List[Tensor]
    }
    Main --> DatasetLoader
    Main --> Trainer
    Main --> Evaluation
    Trainer --> Model
    Trainer --> LossFunctions
    Evaluation --> Model
    DatasetLoader --> Dataset
    Dataset <|-- ImageDataset
    Dataset <|-- ProteinDataset
    Model <|-- ResNetEncoder
    Model <|-- DeepSetEncoder
    LossFunctions --> ContrastiveLoss
    LossFunctions --> EquivarianceLoss
    LossFunctions --> UniformityLoss
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant MD as Model
    participant LT as LossFunctions
    participant TR as Trainer
    participant EV as Evaluation
    M->>+DL: load_data()
    DL-->>-M: dataset object
    M->>+MD: initialize(model_type, hyperparams)
    MD-->>-M: model instance
    M->>+TR: setup_trainer(model, optimizer, loss_fn, dataset)
    TR-->>-M: trainer ready
    loop for epoch
        TR->>TR: train_on_batch()
        activate data augmentation for equivariance batch splits
        compute losses: contrastive, equivariance, uniformity
        backpropagate total loss
    end
    M->>+EV: evaluate_performance()
    EV-->>-M: metrics, embeddings visualization
    Note right of EV: trajectories of rotated inputs plotted\ncosine similarities histograms generated
    M->>+MD: save_checkpoint()
    M->>main: finish
",
    "Anything UNCLEAR": "Clarify if protein data structures deviate from standard point cloud formats, or if specialized augmentation parameters are required. Also, confirm if there are preferred frameworks for visualization tools and evaluation metrics."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch>=1.9",
        "torchvision>=0.10",
        "numpy>=1.21",
        "PyYAML>=6.0",
        "tqdm>=4.62",
        "scikit-learn>=0.24",
        "matplotlib>=3.4",
        "albumentations>=1.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains Dataset and DataLoader classes; handles loading datasets (CIFAR, STL, ImageNet subset, proteins). Implements augmentations; dependencies: torch.utils.data, torchvision.transforms, custom protein loader."
        ],
        [
            "model.py",
            "Defines backbone Encoder class based on ResNet or DeepSet. Includes normalization and optional projection head. Dependencies: torch.nn, torchvision.models."
        ],
        [
            "loss.py",
            "Implements contrastive loss (InfoNCE), equivariance loss (angle preservation), uniformity loss, combined as CARE loss. Needs differentiability and compatibility with batch processing."
        ],
        [
            "trainer.py",
            "Manages training loop: initializing model, optimizer, loss functions; manages batch splits for equivariance; executes forward pass, computes total loss, backpropagation, logging. Dependencies: torch.optim, torch.nn, dataset_loader.py, loss.py."
        ],
        [
            "evaluation.py",
            "Provides functions for evaluation: linear probing, embedding trajectory visualization, cosine similarity histograms. Uses matplotlib, sklearn metrics."
        ],
        [
            "main.py",
            "Main entry point: loads config, initializes DatasetLoader, Model, Loss, Trainer, Evaluation; orchestrates training epochs and evaluation routines. Coordinates dependencies among other files."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "loss.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "None: No API required. All modules run locally with configuration files controlling parameters.",
    "Shared Knowledge": "Hyperparameters, augmentation parameters, and training configuration are shared across dataset_loader.py, loss.py, and trainer.py. Dataset class interfaces are consistent for both image and protein data. Embedding normalization is used for angle-based losses.",
    "Anything UNCLEAR": "Clarify whether to support only specific datasets or make it extensible. Confirm if protein data requires specialized augmentation or pre-processing not detailed in the paper. Also, specify if visualization tools should be integrated for trajectories or just saved as images."
}

## 4. Configuration

## Code: config.yaml
```yaml
## Configuration for CARE training pipeline

dataset:
  name: CIFAR10  # options: CIFAR10, CIFAR100, STL10, ImageNet100, Proteins
  path: "/path/to/dataset"  # Update to actual dataset location

model:
  type: resnet50  # options: resnet50, deepset for proteins
  embedding_dim: 128
  projection_head: True
  projection_dim: 128

training:
  epochs: 400
  batch_size: 256
  learning_rate: 0.001  # 1e-3 for CIFAR/standard datasets; adjust if needed
  weight_decay: 1e-6
  optimizer: Adam  # alternatively, SGD for larger scale
  temperature_infonce: 0.5
  temperature_equiv: 0.1
  batch_splits: 16  # number of splits for equivariance batch
  lambda_equiv: 0.001  # weight for equivariance loss
  augmentations:
    crop_size: 32  # for CIFAR
    jitter_std: 0.1
    color_jitter: true
    blur: true
    rotation_degrees: 15  # for image datasets
    rotation_small: 5  # for protein or small rotation augmentation

loss:
  contrastive: enabled
  equivariance: enabled
  invariance: optional (unless used, skipping)
  uniformity: enabled

evaluation:
  linear_probe_epochs: 100
  linear_probe_lr: 0.001
  evaluation_metrics:
    accuracy: true
    embedding_trajectory: true
    cosine_similarity_histogram: true

save:
  checkpoint_path: "./checkpoints"
  logs_path: "./logs"

notes: |
  Adjust dataset path and augmentation parameters as needed.
  For protein data, set model type to 'deepset' and modify augmentation parameters accordingly.
```

---

**Note:** Full configuration is available in `planning_config.yaml`
