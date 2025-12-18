# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a detailed, structured plan for reproducing the methodology and experiments described in the paper "Knowledge Distillation Based on Transformed Teacher Matching (TTM & WTTM)" by Zheng & Yang. This plan emphasizes **methodological fidelity**, **experimental configuration**, and **hyperparameter details** to guide subsequent implementation.

---

## 1. Overview and Key Objectives

- **Main goal:** Reproduce the proposed distillation methods: **Transformed Teacher Matching (TTM)** and **Weighted TTM (WTTM)**.
- **Innovation:** Dropping temperature scaling on the student's logits, interpreting temperature as a power transformation, and incorporating inherent Renyi entropy regularization.
- **Additional objective:** Demonstrate that WTTM, with sample-adaptive weighting, improves generalization over KD and TTM.
- **Experimental validation:** On CIFAR-100 and ImageNet datasets with various teacher-student pairs.

---

## 2. Core Methodology Details

### 2.1. Fundamental Loss Functions

- **Standard Knowledge Distillation (KD):**
  
  \[
  \mathcal{L}_{KD} = (1-\lambda) H(y, q) + \lambda T^2 D_{KL}(p_T^t || q_T),
  \]
  where:
  - \( H(y, q) \): cross-entropy with ground-truth label.
  - \( D_{KL}(p_T^t || q_T) \): KL divergence between temperature-scaled teacher \( p_T^t \) and student \( q_T \) distributions.
  - \( T \): temperature parameter, applied to logits of teacher and student equally.

- **Transformed Teacher Matching (TTM):**
  
  Drop student temperature scaling; interpret temperature as a **power transform**:
  
  \[
  p_T^t = \sigma(v/T) \quad \Rightarrow \quad \hat{p}_i = \frac{ p_i^\gamma }{ \sum_j p_j^\gamma },
  \]
  where \(\gamma = 1/T\) (temperature as a power exponent).
  
  The TTM loss:
  \[
  \mathcal{L}_{TTM} = H(y, q) + \beta D_{KL}(p_T^t || q),
  \]
  with \(\beta\) balanced to match the original KD loss ratio.

- **Weighted TTM (WTTM):**
  
  Introduce a sample-adaptive weight:
  
  \[
  \mathcal{L}_{WTTM} = H(y, q) + \beta U_{1/T}(p^t) \cdot D_{KL}(p_T^t || q),
  \]
  
  where
  \[
  U_\alpha(p) = \sum_j p_j^\alpha,
  \]
  and importance is given to softer teacher outputs among samples.

### 2.2. Theoretical Foundations
- Temperature as **power transform** (equivalence shown analytically).
- **Renyi entropy regularization** inherently appears in TTM, adding regularization compared to KD.
- Gradients encourage the student to match the **power-transformed teacher distribution**.

---

## 3. Dataset & Experimental Settings

### 3.1. Datasets
- **CIFAR-100**
  - 50K training, 10K test images of size 32x32.
  - 100 classes.
- **ImageNet**
  - Over 1.2M images, 50K validation.
  - 1000 classes.

### 3.2. Architectures
- **CIFAR-100:**
  - Teachers: WRN, ResNet, WideResNet, MobileNetV2, ShuffleNet.
  - Students: same or different architectures.
- **ImageNet:**
  - Teacher/Student pairs involving ResNet, MobileNet, etc.
  - Use torchvision or PyTorch model zoo for architecture definitions.
  
### 3.3. Data preprocessing
- **CIFAR-100:** standard normalization, data augmentations (random crop, flip). Replicate standard training strategies (e.g., SGD, learning rate schedule).
- **ImageNet:** torchvision transforms, standard training schedules.

---

## 4. Implementation Details & Hyperparameters

### 4.1. Model Training & Distillation Procedures
- **Teacher models:**
  - Pre-trained models (either from training from scratch or pre-trained weights from official sources).
  - For CIFAR: re-train teachers if needed (preferably from pretrained or trained the same way as in the paper).
  - For ImageNet: use models from official repositories or trained per the paper's settings.

- **Student training:**
  - Use standard SGD, recall the paper's optimizer and learning rate schedules.
  - Use a batch size and number of epochs matching the original experiments (e.g., 200 epochs for CIFAR, 100 epochs for ImageNet).

### 4.2. Hyperparameters
- **T (Temperature exponent):**
  - Set as per the paper (e.g., \(T=4\) for CIFAR, \(T=4\) for ImageNet).
- **\(\beta\):** set proportional to \(\lambda\), typically:
  \[
  \beta = \frac{\lambda T}{1-\lambda} \quad (\text{based on Eq. (19)}),
  \]
  with details in the hyperparameter tables (A.4). For WTTM, compute \(\hat{U}_{\frac{1}{T}}\) as per dataset distribution.
- **\(\lambda\):** balancing between cross entropy and distillation (commonly 0.9).
- **Sample weights in WTTM:**
  - Use \( U_{1/T}(p^t) \), normalized over dataset or per-batch.
- **Additional hyperparameters:**
  - \(\mu\) for combining with other distillation losses (e.g., CRD, ITRD); generally, set \(\mu = 0.8\) for CRD and 1 for ITRD.
  - For other losses (CRD, ITRD): follow original paper's hyperparameters (A.4, A.5).

### 4.3. Loss Function
- For TTM:
  - Compute teacher's power-transformed output \( p_T^t \).
  - Calculate KL divergence with student softmax output \(\hat{q}_i = \frac{q_i^\gamma}{\sum_j q_j^\gamma}\).
  - Use cross entropy for ground-truth.
- For WTTM:
  - Incorporate per-sample weight \(U_{1/T}(p^t)\).
- For combined losses (e.g., with CRD): include extra distillation loss with \(\mu\) coefficient.

---

## 5. Implementation Steps (High-Level)

1. **Load datasets** and define data augmentation, normalization.
2. **Prepare teacher models**:
   - Load pretrained weights, set to eval mode.
3. **Define student model architecture**s.
4. **Implement the transformation:**
   - Convert teacher softmax logits into \( p_i^\gamma \) probability distributions.
   - Normalize to obtain \(\hat{p}_i\).
5. **Implement loss functions:**
   - Cross-entropy with hard labels.
   - KL divergence between transformed teacher \( p_T^t \) and student distribution \( q \).
   - For WTTM, multiply KL by the sample-specific weight.
6. **Set hyperparameters** as per Tables 8-10.
7. **Optimizers & schedules:**
   - Stochastic Gradient Descent or Adam.
   - Learning rate plans matching original (e.g., cosine annealing, step decay).
8. **Training loop:**
   - For each batch, compute teacher outputs.
   - Generate teacher's transformed distribution.
   - Compute loss and backpropagate.
9. **Evaluation:**
   - Calculate top-1 accuracy on validation/test sets.
   - Record entropy, KL divergence over epochs for analysis.

---

## 6. Additional Notes for Reproducibility and Analysis

- **Hyperparameter tuning:** start with the provided tables; perform grid search if necessary.
- **Multiple runs:** run 3-5 for CIFAR (to report mean ± std).
- **Comparison baselines:** implement KD, feature-based methods, and other recent methods as references.
- **Entropy histograms:** reproduce Figures 3 and 4 for analysis of distribution smoothness.
- **Gradient analysis:** optional, can verify that gradients push student distribution toward the power-transformed teacher.

---

## 7. Summary of Critical Details

| Aspect | Details / Values |
|---------|------------------|
| Datasets | CIFAR-100, ImageNet |
| Architectures | ResNet variants, WideResNet, MobileNetV2, ShuffleNet |
| Temperature \(T\) | 4 (for CIFAR/ImageNet, per tables) |
| \(\beta\) | Derived via Eq. (19), Tables 8-10 |
| \(\lambda\) | 0.9 (typical) |
| Loss components | Cross entropy + \(\beta\) \(\times\) KL divergence |
| Sample weights | \(U_{1/T}(p^t)\) for WTTM |

---

## 8. Future Confirmations & Clarifications Needed

- Exact pretrained teacher weights if available.
- Final optimal hyperparameters if different from tables.
- Specific choices for optimizer, batch size, epochs, data augmentation exact details.
- Whether to use learning rate warm-up or cosine schedules.

---

This comprehensive roadmap ensures replicability of the core methodology and experimental results of the paper, emphasizing fidelity to the original design while leaving warranted flexibility for hyperparameter tuning and implementation choices.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the proposed distillation methods (TTM & WTTM) using PyTorch, leveraging torchvision for datasets and models. The system will load pre-trained teacher models, define student models, compute the power transform of teacher outputs, and compute the combined loss functions. The code will modularly include data loading, model definitions, a training routine with loss computation, and evaluation. Hyperparameters like T, beta, lambda, and sample weights will be configurable via a config file or command line arguments. We will incorporate open-source implementations of common distillation bases (e.g., cross entropy, KL divergence) and define custom functions for power transform, entropy, and sample weighting based on the formulas described in the paper. The training loop will perform per-batch transformation of teacher outputs, compute losses, and update student parameters. The evaluation will measure top-1 accuracy, entropy distribution, and loss metrics for analysis and reproducibility.",
    "File list": [
        "main.py",
        "datasets.py",
        "models.py",
        "losses.py",
        "train.py",
        "evaluation.py",
        "utils.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "```mermaid\nclassDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +load_data() -> tuple\n    }\n    class Model {\n        +__init__(architecture: str, pretrained: bool)\n        +forward(x: Tensor) -> Tensor\n        +load_weights(filepath: str) -> None\n    }\n    class DistillationLoss {\n        +compute(student_logits: Tensor, teacher_probs: Tensor, ground_truth: Tensor, sample_weight: Optional[Tensor]=None) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, teacher_model: Model, loss_fn: DistillationLoss, optimizer: torch.optim.Optimizer, config: dict)\n        +train_epoch() -> None\n        +train() -> None\n    }\n    class Evaluator {\n        +__init__(model: Model, dataset: tuple)\n        +evaluate() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluator\n    Trainer --> Model\n    Trainer --> DistillationLoss\n    class PowerTransform {\n        +compute_probs(teacher_logits: Tensor, gamma: float) -> Tensor\n    }\n    class SampleWeight {\n        +compute_probs(teacher_probs: Tensor, gamma: float) -> Tensor\n    }\n    PowerTransform --> PowerTransform : uses\n    SampleWeight --> SampleWeight : uses\n    Main --> "config.yaml": loads configs for hyperparameters and paths\n```",
    "Anything UNCLEAR": "Clarify the exact data format for models and datasets, and specify the desired hyperparameters (T, beta, lambda, etc.) for each experiment. Also, confirm if pre-trained teacher models should be loaded from specific URLs or checkpoints and whether custom data augmentation is required."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "torchvision==0.14.1",
        "numpy==1.21.0",
        "PyYAML==6.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Main entry point: initializes configuration, coordinates dataset loading, model instantiation, training, and evaluation. It depends on configuration data, instantiates DatasetLoader, Model, Trainer, and Evaluator objects, and calls their methods in sequence."
        ],
        [
            "datasets.py",
            "Handles dataset loading and preprocessing. Implements functions/classes for CIFAR-100 and ImageNet, including transformations (augmentations for training, normalization). Provides data loaders for training and validation sets."
        ],
        [
            "models.py",
            "Defines model architectures for teacher and student networks using torchvision.models or custom implementations, with functions to load pre-trained weights. Supports different architectures as per configuration."
        ],
        [
            "losses.py",
            "Implements core loss functions: cross-entropy, KL divergence, Renyi entropy, and custom functions for computing the power transform of teacher outputs, sample weights, and combined distillation losses. Ensures modularity for TTM/WTTM implementation."
        ],
        [
            "train.py",
            "Contains the training loop: for each batch, loads teacher logits, computes teacher's power-transformed distribution, sample weights, applies custom loss functions, backpropagates, and updates model weights. Uses optimizer and scheduler."
        ],
        [
            "evaluation.py",
            "Performs model evaluation: computes top-1 accuracy, entropy distributions, and logs metrics. Supports validation on test datasets after training."
        ],
        [
            "utils.py",
            "Provides utility functions: loading configurations, saving/loading checkpoints, computing the power transform, entropy, and normalization functions. May also implement dataset statistics or common helper functions."
        ],
        [
            "config.yaml",
            "Stores hyperparameters (T, beta, lambda, sample weights), paths for teacher weights, dataset options, model architecture choices, and training schedules."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "datasets.py",
        "models.py",
        "losses.py",
        "utils.py",
        "main.py",
        "train.py",
        "evaluation.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Configuration parameters like T, beta, lambda, optimizer settings, and data augmentation strategies are shared across dataset loading, loss computation, training, and evaluation modules. Utility functions for power transform and entropy calculations are shared between losses.py and train.py for consistent implementation.",
    "Anything UNCLEAR": "Clarification needed on whether pretrained teacher models should be loaded from specific URLs or checkpoints, and the exact hyperparameters (T, beta, lambda) used in different experiments for reproducing results."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: CIFAR100
  data_dir: ./data/cifar100
  image_size: 32

training:
  batch_size: 128
  epochs: 200
  learning_rate: 0.1
  weight_decay: 5e-4
  momentum: 0.9

distillation:
  T: 4  # Temperature as power exponent
  lambda: 0.9  # Cross-entropy vs distillation balance
  beta: 4.5  # Regularization weight
  TTM_ratio: 1.0  # For scaling beta in TTM

model:
  teacher_architecture: ResNet50
  student_architecture: ResNet18
  pretrained_teacher_weights_path: ./pretrained/ResNet50.pth

optimization:
  optimizer: SGD
  optimizer_params:
    lr: 0.1
    momentum: 0.9
    weight_decay: 5e-4

logging:
  save_checkpoint_dir: ./checkpoints/
  save_summary_every: 10  # save logs every N epochs
  verify_every: 10     # evaluate every N epochs
```

---

**Note:** Full configuration is available in `planning_config.yaml`
