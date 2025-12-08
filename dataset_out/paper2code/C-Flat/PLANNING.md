# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, detailed plan for reproducing the methodology presented in the paper "Make Continual Learning Stronger via C-Flat." This outline covers all critical aspects, including core algorithmic ideas, experimental setups, hyperparameters, datasets, and evaluation strategies, structured in a way that facilitates subsequent implementation.

---

### **I. Overview and Core Algorithmic Ideas**

#### **Main Contribution**
- The paper introduces **C-Flat**, a plug-and-play regularization-based optimization method that emphasizes model flatness in the loss landscape during continual learning (CL). The goal is to improve stability (retaining old knowledge) and generalization in CL tasks by encouraging convergence to flatter minima at each task step.
  
#### **Mathematical Foundations**
- The method combines **zeroth-order** (landscape sharpness) and **first-order** (gradient smoothness) regularizations into a single objective:
  
  \[
  \ell^{C}_{ST}(f^T(\theta^T)) = \ell^{R_\rho^{0}}_{S^T}(f^T(\theta^T)) + \lambda \cdot R_\rho^{1}(f^T(\theta^T))
  \]
  
- The key components:
  - **\( R_\rho^{0}(\theta) \)**: Zeroth-order sharpness regularization based on neighborhood loss difference.
  - **\( R_\rho^{1}(\theta) \)**: First-order flatness regularization based on neighborhood gradient norms.
  - **Hyperparameters** \(\rho, \lambda\): controls neighborhood radius and regularization strength.

#### **Optimization Strategy**
- Regularizers involve neighborhood evaluations of loss and gradient, approximated via:
  - Perturbed parameters within a radius \(\rho\).
  - Gradient-based approximations avoiding expensive Hessian calculations, e.g., Hessian-vector products.
- Update rule involves projected parameter updates ensuring flatness.

---

### **II. Experimental Setup and Dataset Requirements**

1. **Datasets**
   - Main benchmark datasets:
     - **CIFAR-100** (small image classification, 100 classes).
     - **ImageNet-100** (subset of ImageNet, 100 classes).
     - **Tiny-ImageNet** (small image dataset, 200 classes).
   - For reproducibility:
     - Standard splits (e.g., class-incremental with equal or half of classes initial, incremental phases).
     - Fixed random seed for class order shuffling (e.g., seed=1993).
  
2. **Continual Learning Protocols**
   - Class-incremental learning (CIL) with:
     - Fixed class splits.
     - Number of phases (e.g., 5–20 phases).
     - Classes learned per phase (e.g., 10 classes per task, or variable).
   - Metrics:
     - Overall accuracy after each phase.
     - Final (last phase) accuracy.
     - Forgetting measure: difference between initial and final performance.
     - Relative/maximum boost (as in their tables).

3. **Baselines & Comparisons**
   - Plain SGD (or Adam).
   - Existing methods like Replay, iCaRL, WA, MEMO, DER, FOSTER.
   - Ablations of C-Flat with different hyperparameter settings (\(\rho, \lambda\)).

---

### **III. Hyperparameters and Algorithmic Details**

- **Neighborhood radius \(\rho\)**
  - Values to test: e.g., \(\rho \in \{0.1, 0.2, 0.5, 1.0\}\). 
  - \(\rho\) may be scheduled or fixed.
- **Regularization weight \(\lambda\)**
  - Typical range: \(\lambda \in [0.1, 1.0]\).
  - Schedule or tune based on validation (if available).
- **Learning rate schedule**
  - Use optimizer-specific schedules (e.g., cosine decay, step decay).
  - Based on the paper: epoch-dependent decay, e.g., \(\eta_i^T = \bar{\eta} / \sqrt{i}\).
- **Batch size**
  - Standard choices: e.g., 64 or 128.
- **Neighborhood iteration count**
  - Number of evaluations for each regularization term per task:
    - For zeroth-order: perturb parameters within radius \(\rho\), compute loss differences.
    - For first-order: approximate gradient norm via gradient evaluations at perturbed points.
- **Optimizer**
  - Use SGD with momentum or Adam as per baseline experiments.
- **Number of epochs per task**
  - Fixed epochs: e.g., 50–150 epochs.
  - Possibly schedule shorter epochs for C-Flat regularized optimization if initial convergence is slow.
- **Task order randomness**
  - Fixed seed (e.g., 1993) for class order.
- **Repetitions**
  - Multiple runs (e.g., 3–10) for statistical significance.

---

### **IV. Implementation Plan**

#### **Step 1: Data Loading & Preparation**
- Load datasets as per protocols.
- Partition classes into incremental phases.
- For each phase:
  - Combine current phase's data with exemplars/memory samples from previous phases (if rehearsal used).
  - Prepare evaluation datasets.

#### **Step 2: Model Architecture**
- Use standard CNN architectures (ResNet-18/ResNet-32 or MobileNet) for consistency.
- Implement a flexible model that supports:
  - Fully shared parameters for regular CL.
  - Expansion modules for baseline comparison (MEMO-style).

#### **Step 3: Regularizer Computation**
- **Zeroth-order sharpness \( R_\rho^{0} \)**
  - For each batch:
    - Perturb parameters \(\theta\) within \(\rho\):
      \[
      \theta' = \theta + \rho \cdot \frac{\nabla \ell(\theta)}{\|\nabla \ell(\theta)\|_2}
      \]
    - Compute loss at \(\theta'\).
    - Loss difference: \(\ell(\theta') - \ell(\theta)\).
- **First-order flatness \( R_\rho^{1} \)**
  - For each batch:
    - Approximate gradient norms around \(\theta\):
      \[
      g_\text{approx} = \|\nabla \ell(\theta + \rho \cdot \frac{\nabla \ell(\theta)}{\|\nabla \ell(\theta)\|}) \|_2
      \]
    - Use Hessian-vector product for efficient approximation if necessary (via autograd).
- Aggregate regularizations over the neighborhood evaluations.

#### **Step 4: Loss Function Construction**
- Combine task-specific loss (e.g., cross-entropy on current data).
- Add regularization terms:
  \[
  \ell^{C}(\theta) = \ell_\text{task}(\theta) + R_\rho^{0}(\theta) + \lambda R_\rho^{1}(\theta)
  \]
- Implement the approximation schemes for regularizers as per the paper's formulas.

#### **Step 5: Optimization Loop for Each Task**
- Initialize \(\theta^T\) (from previous task or random).
- For epochs in [1, max_epochs]:
  - Sample minibatch.
  - Compute task loss + regularizers.
  - Compute gradient and update parameters:
    \[
    \theta^{T}_{\text{new}} = \theta^{T} - \eta \cdot \nabla \ell^{C}(\theta^T)
    \]
  - Project \(\theta^T\) into neighborhood if necessary (e.g., via clipping or explicit projection).
  - If schedule or adaptive scheme is used, update \(\rho, \lambda\).
- After training:
  - Save model \(\theta^T\) for the next task.
  - Store exemplars/memory as per baseline (if rehearsal is used).

#### **Step 6: Continual Learning Phases**
- Repeat Step 5 for each incremental phase.
- Employ rehearsal methods if needed (e.g., experience replay) with stored exemplars.

#### **Step 7: Evaluation**
- After each phase:
  - Evaluate on all seen classes.
  - Record accuracy, forgetting, landscape flatness metrics (Hessian eigenvalues, landscape plots).
- Aggregate results over multiple runs for statistical reliability.

---

### **V. Additional Implementation Details & Considerations**

- **Loss landscape visualization**
  - Use PyHessian or similar tools to visualize the loss landscape stability.
  - Capture Hessian eigenvalues and landscape plots at different phases/tasks.
- **Regularization evaluation**
  - For debugging: compare landscape measures (Hessian eigenvalues, trace) with and without C-Flat regularization.
- **Scheduling & Ablation**
  - Vary \(\rho\) over training epochs/tasks.
  - Test different \(\lambda\) values and schedules.
  - Ablate regularizers to measure individual contributions.

---

### **VI. Summary Roadmap**

| Step | Task | Subtasks | Notes |
|--------|---------|--------------|--------------|
| 1 | Data prep | Datasets, splits, class orders | Fixed seed, standard splits |
| 2 | Model setup | Architecture, modularity | Support expansion modules |
| 3 | Regularizer implementation | Neighborhood loss, gradient norms | Approximate using autograd & perturbations |
| 4 | Construct loss | Combine task + regularizers | Use efficient approximations |
| 5 | Training | Loop per task | Schedule \(\rho, \lambda\), decay learning rate |
| 6 | Repeat over phases | Incremental process | Store exemplars/memory |
| 7 | Evaluation | Metrics, landscapes | Visualize Hessian eigenvalues, landscape shape |
| 8 | Ablations & hyperparameter tuning | \(\rho, \lambda, schedule) | Multiple runs |

---

This plan provides a robust roadmap to implement "C-Flat" from scratch, enabling precise reproduction and thorough analysis of the results. When ready, detailed code modules can be systematically built following this outline.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular training system using PyTorch. The core components include a dataset loader for incremental class splits, a flexible CNN model (e.g., ResNet-18), and a Trainer class that applies the C-Flat regularization during optimization. The training loop will compute the regularizers (zeroth- and first-order) via neighborhood parameter perturbation and gradient approximations, and update the model parameters accordingly. We will visualize the loss landscape periodically and evaluate on all seen classes after each phase. The system will support multiple hyperparameter scheduling schemes and logging for results comparison.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "```mermaid\nclassDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(split_config: dict)\n        +load_data() -> Dataset\n        +get_task_dataloader(task_index: int) -> DataLoader\n    }\n    class Model {\n        +__init__(params: dict)\n        +forward(x: Tensor) -> Tensor\n        +perturb_params(rho: float) -> None\n        +apply_gradients(gradients: dict, lr: float) -> None\n        +save_checkpoint(path: str) -> None\n        +load_checkpoint(path: str) -> None\n    }\n    class Trainer {\n        +__init__(model: Model, dataset_loader: DatasetLoader, config: dict)\n        +train_phase(task_idx: int) -> dict\n        +compute_regularizers() -> dict\n        +update_model() -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, dataset_loader: DatasetLoader)\n        +evaluate() -> dict\n        +visualize_landscape() -> None\n    }\n    class Utils {\n        +compute_hessian_eigenvalues(model: Model) -> list\n        +visualize_landscape(loss_surface: np.ndarray) -> None\n        +schedule_params(current_step: int, total_steps: int, schedules: dict) -> dict\n    }\n\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> Model\n    Trainer --> Utils\n    Evaluation --> Model\n    Evaluation --> Utils\n```",
    "Program call flow": "```sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    participant U as Utils\n    M->>DL: __init__(split_config)\n    DL-->>M: load_data()\n    M->>MD: __init__(model_params)\n    M->>TR: __init__(model, dataset_loader, configs)\n    loop for each task in dataset\n        TR->>TR: train_phase(task_index)\n        TR-->MD: update parameters with regularizers\n        EV->>MD: evaluate()\n        EV->>U: visualize_landscape() for last epoch loss surface\n        EV-->>M: report metrics\n    end\n    M-->>Main: all phases completed\n```",
    "Anything UNCLEAR": "Clarification needed on the specific neighborhood evaluation frequency (per batch or epoch), exact scheduling methods for \(\rho\) and \(\lambda\), and whether to include exemplars or rehearsal during incremental phases. Also, preferred visualization tools or specific Hessian eigenvalue thresholds for landscape analysis would help refine the system."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "numpy==1.21.0",
        "matplotlib==3.5.0",
        "scikit-learn==0.24.2",
        "h5py==3.1.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains 'DatasetLoader' class that handles loading datasets, creating incremental class splits, and provides methods to retrieve data loaders for each phase. It depends on torch.utils.data and torchvision.datasets for dataset handling. It requires configuration inputs for split schemes and class order seeds."
        ],
        [
            "model.py",
            "Defines the CNN model architecture, e.g., ResNet-18, with support for model expansion modules if needed. It should include methods for forward pass, parameter perturbation within radius rho, and saving/loading checkpoints. Import torch.nn and torch.optim."
        ],
        [
            "trainer.py",
            "Implements 'Trainer' class managing the training loop for each phase, including computation of the regularizers based on neighborhood evaluations, and applying model updates. It requires methods for neighborhood parameter perturbation, gradient approximation, and optimizer steps. It depends on model.py and torch.autograd."
        ],
        [
            "evaluation.py",
            "Provides 'Evaluation' class that handles model evaluation on all seen classes, visualization of the loss landscape (using PyHessian or custom routines), Hessian eigenvalue calculations, and generates logs/plots. It depends on matplotlib and possibly PyHessian for landscape visualization."
        ],
        [
            "utils.py",
            "Includes utility functions for scheduling hyperparameters (\(\rho, \lambda, learning rate)\), Hessian eigenvalue calculation via Hessian-vector products, and regularization computations. It standardizes neighborhood evaluations and contains plotting helpers."
        ],
        [
            "main.py",
            "Acts as the entry point orchestrating dataset loading, model initialization, training over multiple phases, evaluation, logging, and visualization. It depends on dataset_loader.py, model.py, trainer.py, and evaluation.py."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "model.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Common configurations such as dataset split schemes, hyperparameters (\(\rho, \lambda, learning rate\)), and random seeds are shared across dataset_loader.py, trainer.py, and evaluation.py. Utility functions for landscape visualization, Hessian estimation, and scheduling are shared via utils.py.",
    "Anything UNCLEAR": "Clarification needed on the precise neighborhood evaluation frequency (per batch or per epoch) and whether to incorporate rehearsal exemplars for incremental phases. Additional info on visualization tools preferred (PyHessian vs custom) would be helpful."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.1          # Initial learning rate, scheduled to decay over epochs
  batch_size: 64              # Batch size for training
  epochs: 150                 # Total epochs per task phase
  schedule:
    decay_type: exponential   # Learning rate schedule type
    decay_rate: 0.1           # Decay factor for learning rate
  regularization:
    rho: 0.2                  # Neighborhood radius for regularizers
    lambda: 0.5               # Regularization weight for flatness penalties
    neighborhood_eval_per_epoch: 1  # Evaluate neighborhood regularizers per epoch

model:
  architecture: ResNet18      # Model backbone; can be changed to other architectures
  optimizer: SGD              # Optimizer type
  optimizer_params:
    momentum: 0.9             # Momentum parameter
    weight_decay: 1e-4        # Weight decay for regularization
  scheduler:
    type: MultiStepLR        # Learning rate scheduler type
    milestones: [50, 100]    # Epochs at which to decay LR
    decay_factor: 0.1        # Decay factor at each milestone

dataset:
  name: CIFAR-100             # Main dataset for experiments
  split_scheme: class_incremental      # or 'half_classes_then_increment'
  classes_per_task: 10        # Number of classes learned per task
  total_tasks: 10             # Total number of incremental tasks
  seed: 1993                  # Random seed for class order shuffling

regularization:
  neighborhood_radius: 0.2    # Neighborhood radius (\(\rho\))
  lambda: 0.5                # Regularization strength (\(\lambda\))
  eval_frequency: 1          # Evaluation/landscape visualization frequency (per epoch)

evaluation:
  metrics:
    accuracy: true            # Track average accuracy per phase
    forgetting: true          # Track forgetting
    landscape_visualization: true  # Visualize loss landscape periodically

logging:
  output_dir: logs/experiment1  # Directory to store logs, checkpoints, and plots
  save_frequency: 10             # Save model/checkpoints every 10 epochs

hardware:
  gpu: true                   # Enable GPU acceleration
  multi_gpu: false            # Use multiple GPUs if available

misc:
  random_seed: 1993            # Seed for all randomness in training
```

---

**Note:** Full configuration is available in `planning_config.yaml`
