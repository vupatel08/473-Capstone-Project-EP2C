# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, step-by-step plan for reproducing the experiments and methodologies described in the paper, with explicit attention to core details, hyperparameters, datasets, and evaluation metrics.

---

## 1. Core Methodology & Algorithmic Principles

### 1.1 The Fundamental Idea
- The proposed **Schedule-Free** approach introduces no schedules or hyperparameter dependencies on the training duration \( T \).
- It relies on a novel *interpolated iterate averaging* involving a momentum parameter \(\beta\) (typically around 0.9), enabling:
  - Fast convergence akin to Polyak-Ruppert averaging.
  - Enhanced stability due to coupling with gradient evaluation locations.
- It is grounded on a *new theoretical framework* unifying iterate averaging, online-to-batch conversion, and adaptive step sizes.

### 1.2 Key Algorithmic Components
- Introduce an auxiliary sequence involving **interpolated averages**:
  - The *base iterate sequence* (\(z_t\)), performed via standard optimizer steps (SGD, AdamW).
  - The *evaluation sequence* (\(x_t\)), which is a weighted average of the \( z_t \) sequence with interpolated coupling controlled by \(\beta \) (momentum).
- Use *massively larger learning rates* (e.g., \(\gamma = D / G\)) than classical theory suggests, justified by the theory's guarantees under the developed framework.
- No explicit schedule or stop step \( T \) needed; hyperparameters are set to prevent overly aggressive or divergent behavior.

### 1.3 Instantiation Details
- For a stochastic convex setting, perform updates:
  \[
  y_t = (1 - \beta) x_t + \beta z_t,
  \]
  with \( z_{t+1} = z_t - \eta_t \nabla f(y_t, \zeta_t) \),
  where \(\eta_t\) is the learning rate sequence (e.g., \(\eta_t \sim 1 / \sqrt{t}\) scaled by the bound \( D / G \)).
- The sequence \(x_t\) is computed as:
  \[
  x_{t} = (1 - c_{t}) x_{t-1} + c_t z_t,
  \]
  with weights \( c_t \sim 1/t \), ensuring last-iterate coupling akin to linear decay schedule.

---

## 2. Implementation Plan

### 2.1 Software & Environment
- Use **PyTorch** (recommended) for flexible optimizer implementations, gradient computations, and general ML tooling.
- Version control: Use latest stable PyTorch, Python (>=3.8), and relevant ML packages.
- Ensure reproducibility: set random seeds (e.g., `torch.manual_seed`) and initialize data loaders accordingly.

---

## 3. Dataset & Benchmarks

### 3.1 Convex/Logistic Regression Benchmarks
- Datasets placeholder: use publicly available datasets **from LIBSVM** repository or similar (e.g., `a1a`, `w1a`, `ijcnn1`, `rcv1`).
- For each dataset:
  - Normalize features (center around 0, scale to unit variance per feature).
  - Use the **default train/test split** provided.
  - Convert labels to \(\pm 1\) for logistic regression problems.

### 3.2 Deep Learning Benchmarks
- CIFAR10 / CIFAR100:
  - Architecture: Wide ResNet 16-8 (CIFAR10), DenseNet (CIFAR100).
- SVHN:
  - Deep ResNet 3-96.
- ImageNet:
  - ResNet-50.
- Other tasks for the self-tuning/MLCommons:
  - Use provided official data and codebase from the MLCommons challenge (if available).
  - Data augmentation, normalization consistent with official configs.

### 3.3 Additional Tasks
- Language, MRI, and large-scale models:
  - Use the specified architectures and datasets as per the paper.  
  - Follow relevant preprocessing (text tokenization, image normalization, etc.).

---

## 4. Hyperparameters & Settings

### 4.1 General Hyperparameters (from paper)
- **Step size (learning rate):** \(\sim D / G\), where \(D\) bounds initial \(x_1\) or the domain, \(G\) bounds gradient norm.
- **Interpolated weights:** \( c_t \sim 1/t \), e.g., \( c_t = 1/t \).
- **Momentum hyperparameter:** \(\beta \approx 0.9\) (default), with candidate \(\beta \approx 0.98\) for stability in some large models.
- **Large step size:** Use \(\eta_t\) that scales like \( D / G \), not schedule-dependent, e.g., \(\eta_t = D / (G \sqrt{t})\).

### 4.2 Optimizer choice
- For convex problems: custom "Schedule-Free" SGD with momentum (via interpolation).
- For deep models: "Schedule-Free AdamW" with hyperparameters matching the paper configs.

### 4.3 Implementation specifics
- Use **clipping or normalization** if necessary, but most models can train as-is with the specified hyperparameters.
- Weight decay:
  - For convex: specified (e.g., 0.0001 or 0.0002).
  - For deep models: match the paper configs, often very small (e.g., 0.0005).
- Batch sizes:
  - 16–128 depending on hardware (local GPU/TPU), matching experiments.

### 4.4 No schedule or warm-up
- Do **not** implement any learning rate schedule.
- Maintain constant large learning rate scaled as per estimates (\(D / G\)).

---

## 5. Experimental Procedure & Run Strategy

### 5.1 Convex/Logistic Regression
- Run multiple seeds (e.g., 10).
- For each seed:
  - Initialize model parameters randomly.
  - Run the "Schedule-Free" optimizer until approximately the desired number of iterations (about \(\sim T\) where \(T\) corresponds to a target performance).
  - No stopping schedule: run continuously, record the last iterate \(x_T\).

### 5.2 Deep Learning Models
- Run **full training**:
  - Fix hyperparameters as per paper configurations.
  - Run for the number of epochs until convergence curves plateau (~100-300 epochs).
  - Record metrics at the final epoch (or best validation performance for early stopping if used).
  - For self-tuning benchmarks, report the number of steps to reach target accuracy.

### 5.3 Evaluation Metrics
- Classification:
  - Test accuracy (top-1).
- Regression/MRI:
  - Define loss (e.g., SSIM, MSE, or PLM loss).
- Natural language:
  - BLEU, negative log-likelihood, depending on task.
- For the self-tuning benchmarks:
  - Time to target (steps to achieve specified validation/test accuracy).

### 5.4 Data & Gradient Norm Bounding
- Estimate \(D\), \(G\):
  - \(D\): initial parameter domain (e.g., norm of initial weights).
  - \(G\): monitor gradient norms during initial steps, set hyperparameters accordingly.

---

## 6. Additional Implementation Notes
- Carefully implement **interpolated iterate averaging**:
  - Maintain \(\{ x_t \}\) and \(\{ z_t \}\) sequences.
  - For each step:
    - Update \(z_t\): standard optimizer step using \(\nabla f\) at \(y_t\).
    - Compute \(x_t\) as weighted average with decay \( \sim 1/t \).
    - Compute \( y_t \) as \(\beta x_t + (1 - \beta) z_t\).

- For large models:
  - Use the **large learning rate** \(\sim D/G\).
  - Use "momentum interpolation" as described (with \(\beta \approx 0.9\)).

- Log detailed metrics at each epoch/step:
  - Losses, validation accuracy.
  - Gradient norms.
  - Final double-check that no schedule is employed.

---

## 7. Validation & Reproducibility
- Run multiple seeds to quantify variance.
- Cross-validate hyperparameters where possible (though many are set as per theoretical guidance).
- Compare final results directly with baseline schedules (cosine decay, step decay).

---

## Summary
By following this detailed plan:
- You will implement the core "Schedule-Free" methodology precisely.
- You can replicate reported results across convex and deep benchmarks.
- Hyperparameters are systematically derived from theory and paper recommendations, avoiding schedule dependence.
- You will carefully observe the theoretical advantages (e.g., large rate stability, last-iterate convergence) empirically.

Once this roadmap is executed, transitioning to code implementation becomes straightforward, leveraging each detailed step.

---

Would you like me to now prepare sample code snippets based on this plan?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the Schedule-Free optimizer as a custom PyTorch optimizer class that maintains both z_t and x_t sequences, updates z_t using standard optimizer steps at evaluation points y_t, and computes weighted averages for x_t to produce the final model. The optimizer will be designed to operate without any schedule-dependent hyperparameters, set large fixed learning rates based on input bounds, and include hyperparameters such as \beta, initial D, and G estimates. The training loop will initialize model and optimizer, generate initial estimates for D and G, run the optimizer for a fixed number of steps, and record the final iterate x_T for evaluation.",
    "File list": [
        "main.py",
        "optimizer.py",
        "dataset.py",
        "model.py",
        "train.py",
        "evaluate.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run() -> None
    }
    class DatasetLoader {
        +__init__(dataset_path: str, batch_size: int)
        +load_data() -> Tuple[DataLoader, DataLoader]
    }
    class Model {
        +__init__(model_class: str, hyperparams: dict)
        +forward(x: Tensor) -> Tensor
        +to(device: str) -> None
        +initialize_weights() -> None
    }
    class ScheduleFreeOptimizer {
        +__init__(model: Model, hyperparams: dict)
        +step() -> None
        +zero_grad() -> None
        +load_state_dict(state) -> None
        +state_dict() -> dict
    }
    class TrainLoop {
        +__init__(optimizer: ScheduleFreeOptimizer, train_loader: DataLoader, val_loader: DataLoader, config: dict)
        +train() -> None
        +log_metrics() -> None
        +save_checkpoint() -> None
    }
    class Evaluation {
        +__init__(model: Model, data_loader: DataLoader)
        +evaluate() -> dict
    }
    class Hyperparams {
        +learning_rate: float
        +beta: float
        +initial_D: float
        +G_estimate: float
        +num_epochs: int
        +batch_size: int
        +large_learning_rate: float
    }
    Main --> DatasetLoader
    Main --> TrainLoop
    TrainLoop --> ScheduleFreeOptimizer
    ScheduleFreeOptimizer --> Model
    Main --> Evaluation
    Model --> Model
    TrainLoop --> Model
",
    "Program call flow": "
sequenceDiagram
    participant C as Main
    participant DL as DatasetLoader
    participant M as Model
    participant OPT as ScheduleFreeOptimizer
    participant TL as TrainLoop
    participant EV as Evaluation
    C->>DL: load_data()
    DL-->>C: dataset_train, dataset_val
    C->>M: initialize(model_class, hyperparams)
    M-->>C: model instance
    C->>OPT: init(model, hyperparams)
    OPT-->>C: optimizer instance
    C->>TL: train(optimizer, dataset_train, dataset_val)
    TL->>OPT: step(), zero_grad()
    OPT-->>TL: update z_t, compute weighted x_t
    TL-->>C: training complete, save final model/x_T
    C->>EV: evaluate(model, dataset_val)
    EV-->>C: metrics
",
    "Anything UNCLEAR": "Clarify the exact hyperparameter initialization process, especially D and G estimates, and whether any particular optimizer (SGD/AdamW) should be used internally for z_t updates."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.0",
        "torchvision==0.14.0",
        "numpy==1.21.0",
        "matplotlib==3.5.2",
        "tqdm==4.64.0",
        "scikit-learn==1.0.2",
        "Pillow==9.2.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset.py",
            "Defines DatasetLoader class which manages data loading for training and validation sets, normalizes data, and prepares data loaders with batch size. Depends on torchvision and scikit-learn for dataset preprocessing and loading."
        ],
        [
            "model.py",
            "Implements Model class representing the neural network architectures used (e.g., ResNet, DenseNet). Includes methods for initialization, forward pass, weight setup. Depends on torch.nn modules."
        ],
        [
            "optimizer.py",
            "Implements ScheduleFreeOptimizer class that manages both z_t and x_t sequences, updates z_t via optimizer steps at evaluation points y_t, maintains interpolated x_t averages, and supports hyperparameters such as \beta, \eta, D, G. Depends on torch.optim and torch.nn."
        ],
        [
            "train.py",
            "Defines TrainLoop class that initializes training with dataset loaders, model, optimizer; manages training procedure over fixed steps, records hyperparameters, performs optimizer steps, updates x_t, and logs training metrics. Depend on model.py and optimizer.py."
        ],
        [
            "evaluation.py",
            "Provides Evaluation class to run model inference on validation/test datasets, computes accuracy or loss metrics. Depends on model.py and datasets."
        ],
        [
            "main.py",
            "Entry point script executing overall workflow: load config, initialize DatasetLoader, Model, optimizer, run training, evaluation, save final model state, and output performance metrics. Coordinates all modules."
        ],
        [
            "utils.py",
            "Includes utility functions for setting seeds, estimating bounds D and G, plotting results, and saving/loading checkpoints. Shared across training, evaluation, and main scripts."
        ]
    ],
    "Task list": [
        "utils.py",
        "dataset.py",
        "model.py",
        "optimizer.py",
        "train.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "N/A -- operation is controlled via main.py, no external API interface required.",
    "Shared Knowledge": "Config parameters such as large learning rate, \beta, D, G estimates, and fixed hyperparameters are defined and shared across main.py, train.py, and optimizer.py. Utility functions for seed setting, checkpointing, and plotting are reused.",
    "Anything UNCLEAR": "Clarify exact method for estimating initial D and G in the code, and whether specific architectures or datasets require customized normalization/preprocessing steps beyond standard torchvision transforms."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0025  # Large fixed learning rate scaled as D / G, as justified by theory, approximately 2.5e-3.
  batch_size: 32        # Common batch size for experiments involving ResNet, DenseNet, and ImageNet.
  epochs: 100           # Typical number of epochs for self-contained experiments on datasets like CIFAR and ImageNet.
  warmup_steps: 4000    # Fixed warmup steps (as used in some architectures), no schedule-based decay.
  large_learning_rate: true  # Confirmed to be used for scale-insensitive experiments.
  hyperparams:
    beta: 0.9           # Momentum hyperparameter, default as used in paper.
    initial_D: 1.0      # Set based on dataset normalization (to be estimated or provided).
    G_estimate: 1.0     # Estimated gradient norm bound; for initial runs, can be set conservatively.
    weight_decay: 0.0001 # Default weight decay for convex problems, adjustable based on dataset.
    optimizer: AdamW    # Optimizer type, set as default.
  # Additional hyperparameters related to optimizer or training can be added here.
dataset:
  name: CIFAR10        # Dataset name; can be replaced with CIFAR100, SVHN, or ImageNet as needed.
  data_path: ./data  # Path to dataset directory.
  normalize:
    mean: [0.4914, 0.4822, 0.4465]  # Dataset-specific mean for normalization.
    std: [0.2023, 0.1994, 0.2010]   # Dataset-specific std for normalization.
model:
  architecture: WideResNet  # Specify architecture; replace with DenseNet, ResNet50, etc., as needed.
  depth: 16
  width_multiplier: 8
  dropout: 0.3          # Dropout rate for regularization.
optimizer:
  type: AdamW
  weight_decay: 0.0001
training:
  num_epochs: 100
  batch_size: 32
  seed: 42
logging:
  save_checkpoints: true
  checkpoint_dir: ./checkpoints
  log_interval: 50
evaluation:
  metrics: accuracy
  evaluation_interval: 100  # Log metrics every 100 steps or epochs.
```

---

**Note:** Full configuration is available in `planning_config.yaml`
