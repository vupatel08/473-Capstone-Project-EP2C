# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Below is a comprehensive, step-by-step plan capturing the core methodologies, experimental setups, and implementation considerations based on the provided paper. It is organized into logical sections to serve as a detailed roadmap for reproducing the work. This will facilitate later translation into code.

---

## 1. Overall Objective and Approach
- **Main Goal:** Train diffusion-structured off-policy samplers (including GFlowNet variants) for sampling from unnormalized target distributions (e.g., Boltzmann energy functions), possibly with complex high-dimensional structures.
- **Core Innovation:** Incorporate exploration strategies (local search with replay buffers), hyperparameter smoothing (e.g., adaptive step sizes), and analysis of various training objectives, including trajectory-based and risk-sensitive methods.
- **Key Components to Implement:**
  - Diffusion process parameterization (forward and reverse SDEs).
  - Off-policy training via trajectory and importance sampling (GFlowNet / TB objectives).
  - Exploration enhancements: local search (parallel MALA), replay buffers, adaptive step sizes.
  - Evaluation metrics: partition function estimates, Wasserstein distances, visualization.

---

## 2. Model and Methodology Details

### 2.1 Diffusion Process Parameterization
- **Forward SDE:**
  - Use the specified SDE with parameters:
    - Diffusion coefficient: \(\sigma(t) := \nu \sqrt{2\beta(t)}\), with \(\beta(t) := (1 - t)\beta_{min} + t \beta_{max}\), typically \(\beta_{min} = 0.01\), \(\beta_{max} = 4\).
    - For the baseline, set \(\nu=1\).
  - Discretization:
    - Euler-Maruyama with \(\Delta t = 1/T\), \(T\) (e.g., 100, 200, 300).
    - Transition kernel: Gaussian with mean and variance derived from \(\beta(t)\).

- **Reverse Process (Sampling):**
  - Parameterize as a neural SDE with learnable drifts \(u(x, t; \theta)\) and noise \(g(x,t;\theta)\).
  - Design network architectures for \(u\) and \(g\):
    - Fully connected MLPs with input \((x, t)\).
    - Consider techniques for incorporating conditioning on the target (e.g., energy function).
  - For the experiments, fix the diffusion coefficients or learn them (see hyperparameters).

### 2.2 Energy / Target Distribution
- \(\mathcal{E}(x)\): Energy function defining the unnormalized density \(R(x) = \exp(-\mathcal{E}(x))\).
- For high-dimensional tasks (Manywell, Funnel, MNIST latent), implement energy functions with the specified parameters.
- For static tasks, avoid data dependence; for conditional (VAE), include condition \(x\) explicitly as input to the drift network.

---

## 3. Training Objectives and Off-Policy Exploration

### 3.1 Main Objectives
- **Trajectory-Based KL / TB Loss:**
  - Enforce equality between forward and backward trajectory densities.
  - Use the trajectory balance loss:
    \[
    \mathcal{L}_{TB} = \left(\log \frac{Z_\theta P_F(\tau)}{R(x_1) P_B(\tau|x_1)}\right)^2
    \]
  - Train \(u,g\) networks and estimate \(\log Z_\theta\).

- **Alternative Objectives:**
  - Variance-based estimators (VarGrad).
  - Subtrajectory balance (less effective in experiments but supported).

### 3.2 Off-Policy Sampling & Exploration Enhancements
- **Replay Buffer:**
  - Maintain buffers of low-energy (or promising) samples.
  - Implement FIFO buffer with large capacity (e.g., 600k or 900k slots).
  - Sample sub-trajectories or states from buffer for off-policy updates.

- **Local Search / Parallel MALA:**
  - Implement the detailed parallel MALA algorithm:
    - Dynamic step size \(\eta\), targeting acceptance rate ~0.574.
    - Adapt \(\eta\) based on acceptance feedback.
    - Use the gradient of energy \(\nabla \mathcal{E}(x)\) to propose new samples.
    - Keep a buffer \(\mathcal{D}_{LS}\) of low-energy samples for targeted exploration.

- **Sampling Strategy:**
  - Alternate training:
    - Trajectory sampling (on-policy) 50% of the time.
    - Buffer sampling + local search (off-policy exploration) the remaining 50%.
  - Use importance weights and posterior weighting strategies as per appendix.

- **Hyperparameters for Local Search:**
  - Number of steps \(K\) (e.g., 200 for unconditional; 500 for conditional).
  - Burn-in: first 100 steps (or 200 for conditional).
  - Initial step size \(\eta_0 = 0.01\).
  - Adapt step size \(\eta\) to target acceptance rate (~0.574) with factors 1.1/0.9.
  - Target acceptance rate: 0.574.

### 3.3 Hyperparameter Settings
- **Learning Rates:**
  - Use \(1e-3\) for neural SDE networks.
  - During local search MH steps, adapt \(\eta\) dynamically.
- **Training steps:**
  - Total iterations: ~25,000.
  - Batch size: 300 (or per resource constraints).
- **Objective weights:**
  - Prioritize trajectory balance loss, plus exploration loss, separate LFs, or variational estimators.
- **Model parameters:**
  - Neural networks: 2-4 layers, 400 neurons per layer, ReLU activations.
  - Log-partition \(\log Z_\theta\): train as a scalar or small MLP from data.

---

## 4. Experiments and Benchmark Details

### 4.1 Tasks / Datasets
- **Unconditional tasks:**
  - Manywell high-dimensional mixture (\(d=32, 128, 512\)), energy function specified.
  - Funnel: 10-dimensional with multi-modal structure.
  - 25GMM: 2D mixture with 25 Gaussians.
- **Conditional Task:**
  - MNIST latent space inference using pretrained VAE decoder.
  - Condition: an image \(x\), target: the latent \(z\) with density proportional to \(p(z|x) \propto p(z)p(x|z)\).
  - Energy (negative log-likelihood) given by the decoder.

### 4.2 Evaluation Metrics
- **Partition Function Estimate:**
  - Variational lower bounds (\( \log Z \)) using trajectory importance sampling and importance-weighted estimates.
- **Sample Quality:**
  - 2-Wasserstein distance between generated samples and baseline ground truth (if available) or true energy model.
- **Exploration and Mode Coverage:**
  - Visualization in 2D projections or energy contour plots.
- **Computational Cost:**
  - Record runtime per iteration and per batch on GPUs.

### 4.3 Hyperparameters & Settings Summary
- Diffusion steps \(T\): 100 for baseline, larger (e.g., 200, 300) for better quality.
- Step size \(\eta\): adaptive with target acceptance ~0.574.
- Buffer sizes: 600k-900k for large-scale experiments.
- Number of trajectory samples per gradient step: 20.
- Learning rate: \(1e-3\), with schedule if needed.

---

## 5. Implementation Details (Later Code)
Flag points:
- Modular neural network design: separate modules for drift \(u\), diffusion \(g\), and partition estimate.
- Trajectory sampling: Euler-Maruyama with denoising target.
- Off-policy updates: use buffer and local search periodically.
- Parallelization: batch compute MH proposals.
- Dynamic step size: update after each batch based on acceptance.
- Visualization tools: energy contours, sample scatter plots, and energy histograms.

---

## 6. Summary of Reproducibility Checklist
- Provide code snippets for:
  - Neural SDE architecture.
  - Euler-Maruyama discretization.
  - Trajectory and buffer sampling.
  - Local search / MH proposal with adaptive step size.
  - Objective computation (TB, VarGrad, etc.).
  - Data handling: synthetic energies, MNIST decoder.
- Explicit hyperparameters and schedule.
- Evaluation routines for partition function, Wasserstein, and visualizations.
- Documentation and pseudocode for buffer management and local search.

---

## Final Remarks
- This plan summarizes all critical parts needed to faithfully reproduce the method described.
- Attention should be paid to the robustness of training: multiple runs, monitoring acceptance rates, buffer health, and energy stability.
- Carefully implement and validate each component before integrating into large experiments.

---

Would you like me to now proceed with transforming this outline into explicit code snippets or detailed pseudocode for each component?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Build a modular system using PyTorch for neural network definitions and training, with NumPy for numerical computations. The core will consist of neural SDE architectures parameterized by simple MLPs, Euler-Maruyama discretization for forward sampling, and customizable reverse models. Implement buffers and local search via GPU-accelerated MH proposals. Encapsulate training objectives (trajectory-based, variance, etc.) as loss functions. Use optimized routines for importance sampling and partition function estimation. Visualization and evaluation scripts will generate energy contours, sample scatter plots, and compute metrics. Maintain separation of concerns: 'model.py' for networks, 'sampler.py' for SDE/ML implementations, 'buffer.py' for replay buffer, 'train.py' for training loops, 'eval.py' for metrics, and 'main.py' as orchestrator.",
    "File list": [
        "main.py",
        "model.py",
        "sampler.py",
        "buffer.py",
        "train.py",
        "evaluation.py",
        "visualization.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class NeuralSDE {\n        +__init__(input_dim: int, hidden_dim: int, network_type: str='MLP')\n        +forward(x: Tensor, t: float) -> Tensor\n        +get_parameters() -> dict\n    }\n    class EulerSampler {\n        +__init__(sde: NeuralSDE, T: int)\n        +sample(x0: Tensor, energy_fn: Callable, steps: int) -> List[Tensor]\n        +discretize(x: Tensor, t: float, delta_t: float, noise: Tensor) -> Tensor\n    }\n    class ReverseModel {\n        +__init__(input_dim: int, hidden_dim: int)\n        +forward(x: Tensor, t: float) -> Tensor\n        +get_parameters() -> dict\n    }\n    class ReplayBuffer {\n        +__init__(capacity: int)\n        +add(sample: Tensor, energy: float)\n        +sample(batch_size: int) -> List[Tuple[Tensor, float]]\n        +maintain() -> None\n    }\n    class Trainer {\n        +__init__(sde: NeuralSDE, reverse: ReverseModel, buffer: ReplayBuffer, config: dict)\n        +train() -> None\n        +compute_loss(trajectories: List[List[Tensor]], energies: List[float]) -> Tensor\n        +update_parameters(loss: Tensor) -> None\n    }\n    class Evaluation {\n        +__init__(model: NeuralSDE, energy_fn: Callable)\n        +estimate_logZ() -> float\n        +compute_wasserstein(samples: Tensor, target_samples: Tensor) -> float\n        +visualize_samples(samples: Tensor) -> None\n    }\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> NeuralSDE\n    Trainer --> ReverseModel\n    Trainer --> ReplayBuffer\n    Main --> Model (via NeuralSDE, ReverseModel)\n    Main --> Buffer (ReplayBuffer)\n    NeuralSDE --> DiscretizationMethod\n    Evaluation --> Sampler\n    Sampler --> NeuralSDE\n    Buffer --> SampleData\n    ",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant C as Config\n    participant T as Trainer\n    participant E as Evaluation\n    participant B as Buffer\n    participant S as Sampler\n    participant R as NeuralSDE\n    participant Rm as ReverseModel\n    M->>C: load configuration\n    C-->>M: return dict\n    M->>R: initialize with architecture\n    M->>Rm: initialize with architecture\n    M->>B: initialize buffer\n    M->>S: initialize sampler with neural SDE, T, reverse model, buffer\n    Note over M, S: main training loop begins\n    loop each iteration\n        S->>S: sample trajectories using Euler integrator, adapt step size if needed\n        S-->>T: provide trajectories, energies\n        T->>T: compute training loss (trajectory, TB, VarGrad, exploration)\n        T-->>T: calculate gradients\n        T->>R: update neural SDE parameters\n        T->>Rm: update reverse model params\n        T->>B: add promising samples to buffer, maintain size\n    end\n    M->>E: periodically evaluate on test set / target distribution\n    E-->>M: report logZ, Wasserstein, sample visuals\n    M->>Visualization: generate energy contour plots and sample scatterplots\n    Note over M: end of training, finalize models\n",
    "Anything UNCLEAR": "Details on the energy function implementations, the precise neural network sizes, and whether to fix diffusion coefficients or learn them. Also, exact schedule and hyperparameters for the adaptive step size in MH local search need confirmation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.1",
        "numpy==1.21.0",
        "scipy==1.7.3",
        "matplotlib==3.4.3",
        "tqdm==4.62.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Initializes configuration, orchestrates overall execution. Depends on 'trainer.py', 'evaluation.py', 'model.py', 'buffer.py', 'sampler.py'."
        ],
        [
            "dataset_loader.py",
            "Loads and preprocesses datasets such as MNIST or synthetic energies. Supplies data to trainer and evaluation. Depends on data format specifics and energy functions."
        ],
        [
            "model.py",
            "Defines neural network architectures for neural SDE drifts, diffusion, and reverse processes. Implements parameter classes and forward methods. Used by 'sampler.py' and 'trainer.py'."
        ],
        [
            "sampler.py",
            "Implements the forward SDE discretization (Euler-Maruyama), diffusion process, and sampling. Includes neural network calls for drift/g. Provides samples for training and evaluation. Depends on 'model.py'."
        ],
        [
            "buffer.py",
            "Manages replay buffers for off-policy exploration and low-energy sample storage. Supports sampling from buffer for training. Depends on sample outputs and buffer management logic."
        ],
        [
            "train.py",
            "Contains training logic, including loss computation for trajectory TB, VarGrad, and exploration objectives. Updates neural networks, manages step size adaptations, interacts with buffers. Depends on models, sampler, buffer, configuration."
        ],
        [
            "evaluation.py",
            "Performs estimation of \(\log Z\), Wasserstein, and generates visualization outputs. Uses samples from 'sampler.py', evaluation metrics, and energy functions. Dependent on trained models."
        ],
        [
            "visualization.py",
            "Generates energy contour plots, sample scatter plots for visualization and analysis. Uses generated samples and energy functions. Depends on 'evaluation.py'."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "sampler.py",
        "buffer.py",
        "train.py",
        "evaluation.py",
        "visualization.py",
        "main.py"
    ],
    "Full API spec": "Minimal, object-oriented internal APIs: \n\nclass NeuralSDE:\n    def __init__(self, input_dim, hidden_dim, network_type='MLP')\n    def forward(self, x: Tensor, t: float) -> Tensor\n    def get_parameters() -> dict\n\nclass EulerSampler:\n    def __init__(self, sde: NeuralSDE, steps: int)\n    def sample(self, x0: Tensor, energy_fn: Callable, steps: int) -> List[Tensor]\n\nclass ReverseModel:\n    def __init__(self, input_dim, hidden_dim)\n    def forward(self, x: Tensor, t: float) -> Tensor\n    def get_parameters() -> dict\n\nclass ReplayBuffer:\n    def __init__(self, capacity: int)\n    def add(self, sample: Tensor, energy: float) -> None\n    def sample(self, batch_size: int) -> List[Tuple[Tensor, float]]\n    def maintain(self) -> None\n\nClass Trainer:\n    def __init__(self, model: NeuralSDE, reverse: ReverseModel, buffer: ReplayBuffer, config: dict)\n    def train() -> None\n    def compute_loss(trajectories: List[List[Tensor]], energies: List[float]) -> Tensor\n    def update_parameters(loss: Tensor) -> None\n\nClass Evaluation:\n    def __init__(self, model: NeuralSDE, energy_fn: Callable)\n    def estimate_logZ() -> float\n    def compute_wasserstein(samples: Tensor, target_samples: Tensor) -> float\n    def visualize_samples(samples: Tensor) -> None",
    "Shared Knowledge": "Common utilities include neural network modules for MLPs, energy functions, initialization routines, adaptive step size logic, importance sampling functions, and visualization tools. Buffer logic is shared between 'buffer.py' and exploration strategies in 'train.py', facilitating off-policy sampling and buffer maintenance.",
    "Anything UNCLEAR": "Clarify specification for the energy functions' exact implementation (synthetic, real energy, conditional energies), detailed hyperparameters for network sizes, step size adaptation schedule, and dataset input/output formats. Confirm whether diffusion coefficients should be learned or fixed."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001                  # Use a learning rate of 1e-3 for neural networks
  batch_size: 300                       # Batch size of 300 samples per training iteration
  total_iterations: 25000               # Total training iterations (e.g., 25,000 as used in experiments)
  gradient_accumulation_steps: 1        # No gradient accumulation unless memory constrained
  optimizer: Adam                       # Use Adam optimizer
  optimizer_params:
    betas: [0.9, 0.999]
    eps: 1e-8
  gradient_clip_norm: 1.0              # Optional: clip gradients to stabilize training

model:
  neural_sde:
    input_dim: 2                        # Energy functions in examples have 2 or 32 dimensions
    hidden_dim: 400                     # Hidden layer size
    network_type: 'MLP'                 # MLP default
  diffusion_coefficient: fixed          # Set to 'fixed' or 'learned' based on experiment
  diffusion_value: 1.0                  # Fixed diffusion coefficient for baseline; can be tuned or learned

diffusion_process:
  T: 100                                # Number of discrete steps in Euler-Maruyama discretization
  delta_t: 0.01                         # \(\Delta t\), with T=100
  beta_min: 0.01                        # Minimum \(\beta(t)\)
  beta_max: 4.0                         # Maximum \(\beta(t)\) to cover the scheduled noise scale

training_objectives:
  trajectory_balance_loss: true         # Use trajectory balance objectives for diffusion
  var_grad_loss: true                   # Also evaluate variance-based estimators
  exploration_loss_weight: 1.0          # Relative weight of exploration or buffer-based loss
  buffer_capacity: 600000               # Buffer size (~600k samples in experiments)
  buffer_sampling_strategy: 'FIFO'      # FIFO buffer for samples
  buffer_priority_k: 0.01               # Prioritization hyperparameter \(k=0.01\)

exploration:
  local_search:
    steps_per_update: 200               # K=200 steps for local MH searches
    burn_in_steps: 100                  # First 100 steps are burn-in
    initial_step_size: 0.01             # \(\eta_0=0.01\)
    target_acceptance: 0.574            # Acceptance rate schedule target
    step_size_increase_factor: 1.1     # \(\mathcal{I}=1.1\)
    step_size_decrease_factor: 0.9     # \(\mathrm{f}_d=0.9\)
    max_steps: 200                      # Max MH steps per local search
  adaptive_step_size:
    schedule: 'feedback'                # Adapt \(\eta\) based on acceptance feedback for each batch
    target_acceptance: 0.574            # Target Holt acceptance rate
  exploration_strategy:
    buffer_sampling_ratio: 0.5          # 50% on-policy, 50% off-policy (buffer + local search)

evaluation:
  logZ_estimation_methods:
    importance_sampling: true           # Use importance sampling to estimate \(\log Z\)
    variance_estimator: true              # Use variance-based estimator (VarGrad)
    num_samples: 2000                     # Number of Monte Carlo samples
  metrics:
    Wasserstein: true                     # Compute 2-Wasserstein distance
    visualization: true                   # Generate plots for contours and samples
  evaluation_freq: 1000                 # Perform evaluation every 1000 iterations

hyperparameters:
  seed: 42                              # Random seed for reproducibility
  device: cuda:0                        # Use GPU if available
  save_checkpoint_every: 5000         # Save model checkpoints periodically
  resume_training: false                # Start fresh or resume from checkpoint

dataset:
  type: synthetic_energy                 # Default; switch to MNIST if conditional experiments
  dataset_name: 'Manywell'               # Use 'Manywell' for high-dimensional, synthetic energies
  input_dim: 32                          # Dimensionality (e.g., 32 for experiments)
  energy_function: 'specified'           # Use the energy function described in the paper
```

---

**Note:** Full configuration is available in `planning_config.yaml`
