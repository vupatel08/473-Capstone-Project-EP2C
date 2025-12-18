# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. To develop a comprehensive plan for reproducing the described methodology, I will organize it into clear sections aligned with the paper’s core methodological components, experimental setups, datasets, hyperparameters, and evaluation procedures. This will serve as a detailed blueprint for subsequent coding.

---

## 1. Overview of Core Methodology and Key Innovations

**Goal:** Build a hybrid neural-physics driven Lagrangian fluid simulation framework augmented with a GNN surrogate.

### Main Components:
- **Base GNN Surrogates:** GNS (Sanchez-Gonzalez et al., 2020) and SEGNN (Brandstetter et al., 2022a) with configurable equivariance properties.
- **Physics-Informed Enhancements:**
  - **External Force Treatment:** During training and inference, explicitly model external forces as separate terms, enforcing a forced acceleration model (Eq. 2).
  - **External Force Smoothing:** Convolve the external force field with a Gaussian kernel based on velocity-driven standard deviations to handle spatially varying external forces.
  - **SPH Relaxation:** Correct particle distributions post- (or during) inference with relaxation steps, grounded in classical SPH formulations with tuned hyperparameters ($\alpha$, $\beta$, number of steps/loops).
  - **Density and Boundary Handling:** Use density and pressure clamping, tensile instability control (TIC), and wall boundary conditions to stabilize free-surface particles and remove artifacts like clustering.

### Important Modeling assumptions:
- Equation of motion (Eq. 2) is split into learned gravity-driven accelerations plus explicit external forces.
- The hybrid approach explicitly models the force term, enforcing disentanglement between external influences and internal dynamics.
- Relaxation and force smoothing techniques are designed to stabilize long rollouts, especially critical near free surfaces and shock regions.

---

## 2. Dataset and Experimental Setup

**Goal:** Reproduce experiments on coarse-grained Lagrangian particle datasets:

### Dataset Types:
- **Dam Break (DAM):** 2D/3D, real-world-like free surface flow with violent displacements.
- **Reverse Poiseuille Flow (RPF):** 2D/3D shear-driven flow with laminar behavior.
- **Lid-Driven Cavity (LDC):** classical benchmark, vortex flows, boundary layer only.

### Dataset Requirements:
- Particle states over many timesteps (~1000+ steps).
- For each timestep: positions $\mathbf{p}_n^t$, velocities $\mathbf{u}_n^t$, particle types, external force features if applicable.
- The datasets provided in Toshev & Adams (2024), or code that can generate similar synthetic datasets by running classical SPH/CFD simulators with different initial/boundary conditions.

### Dataset Access:
- Official datasets are in Toshev et al. GitHub repositories or supplementary materials, particularly referenced in the paper.

### Data Processing:
- Load sequences of particle states.
- For training, select subsequences (e.g., every 100th step) for data efficiency.
- For evaluation, use longer trajectories (~400+ steps) to assess long-term stability.

---

## 3. Model Architectures

### GNN Surrogates:
- **GNS:** encoder-processor-decoder with graph network blocks (Battaglia et al. 2018)
- **SEGNN:** E(3)-equivariant message passing with steerable MLPs, accessible via existing deep learning frameworks (e.g., e3nn or a custom module)

### Implementation Notes:
- Use JAX/Flax or PyTorch Geometric for efficient graph neural network modeling.
- Incorporate positional encodings, particle types, and optional features (external forces).

### Input Features:
- Particle positions in history (e.g., last 5 timesteps) (for autoregressive modeling)
- Particle velocities and types
- External force vector $g$ as a feature (if available)
- Past accelerations inferred from finite differences

---

## 4. External Force Treatment and Smoothing

### Explicit Force Model: 
- Implement Eq. 2 split:
  - **Learned internal dynamics:** GNN predicts accelerations (including pressure, viscosity, external forces)
  - **Explicit external force term:** Add a precomputed external force $\mathbf{g}$ (or smoothed version)
- **Smoothing Force Function:**
  - Compute velocity std deviation $\sigma_u$ over the dataset.
  - Convolve $\mathbf{g}$ with Gaussian $\mathcal{N}(0,\sigma_u^2)$ or use the analytical erf-based approximation.
  - Replace $\mathbf{g}$ in Eq. 2 with the smoothed force.

### Implementation:
- During dataset preprocessing, generate the smoothed external force maps for each particle.
- During training and inference, pass these as features or include in the model input, additionally modeling the force as a separate term (Eq. 2).

---

## 5. SPH Relaxation: Concept, Implementation, and Hyperparameters

### Purpose:
- Correct pathological clustering artifacts and stabilize long rollouts.

### Formulation:
- Perform *relaxation steps* (up to 5) between model predictions.
- SPH relaxation in a zero-velocity limit with pressure and viscosity terms, using hyperparameters $(\alpha, \beta, l)$ as in Eq. 4 and Appendix G.

### Implementation:
- After each GNN step, for several steps:
  - Compute density using kernel sum.
  - Clamp density deviation (see section on density correction).
  - Calculate pressure and viscosity contributions.
  - Update particle positions (no velocities) iteratively per hyperparameters.
- Hyperparameters:
  - $\alpha$, $\beta$ for the strength of pressure and viscosity in relaxation.
  - Number of relaxation steps $l$, or loops/iterations.
  - Kernel cutoff radius (larger for relaxation: supported by the paper, e.g., 3x average particle distance).
- Use existing SPH functions or custom code for neighbor search and smoothing (e.g., quintic spline kernels).

### Hyperparameter Tuning:
- Hyperparameters tuned on subset (Appendix G.2):
  - $\alpha \in [0.005, 0.05]$ for pressure.
  - $\beta \in [0, 0.2]$ for viscosity.
  - Relaxation steps $l$ in [1-5].

---

## 6. Boundary and Free Surface Handling

### Boundary Conditions:
- Use Adami et al.’s generalized wall boundary condition (Appendix B).
- Volume/pressure violations at free surfaces managed by:
  - Clipping densities to $[0.98, 1.02] \times \rho_{ref}$.
  - Correct tensile instabilities (TIC) by modifying pressure formulation at free surfaces.
- Implement layering of boundary particles if needed for walls.

### Free Surface Correction:
- Density summation is handled with clipping.
- Use of the method (Eq. 7, Appendix G.3) to mitigate surface artifacts: set density to reference if too low, clip high outliers.

---

## 7. Long-Horizon Rollouts & Stability Techniques

- **Iterative correction:**
  - Between steps, perform 1-5 relaxation SPH steps to handle distortions.
- **Hyperparameters:**
  - Relaxation parameters $\alpha, \beta$, and number of loops/steps (see Appendices G.3–G.4).
- **Long-rollout checks:**
  - Monitor average error metrics ($\text{MSF}_{400}$, Sinkhorn divergence) over multiple trajectories.
  - Use ensemble averaging and quantile shading to assess stability.

---

## 8. Training Procedure

- Use **autoreg** training:
  - Inputs: history of positions, velocities, external force features.
  - Targets: accelerations, or position differences.
  - Include external force as explicit model component (Eq. 2).
  - Loss: Mean squared error (position, velocity, energy) plus optional regularizers for density stability.
- **Gradient Accumulation:** 
  - Used in the paper for larger models; support in code.
- **Augmentations:**
  - Add random walk noise (Pfaff et al. 2020).
  - Apply pushforward -> data augmentation.

---

## 9. Evaluation and Ablation Protocols

- **Metrics:**
  - $\mathrm{MSE}_{400}$, Sinkhorn divergence, $\mathrm{MSE}_\mathrm{E kin}$, MAE of density, Dirichlet energy, Chamfer distance.
- **Experiments:**
  - Ablate external force treatment (+ smoothing).
  - Ablate relaxation hyperparameters ($\alpha, \beta, l$).
  - Evaluate long-term stability (step 80, 240, 400, 1000+).
  - Visualize particle distribution (density maps, scatterplots).
- **Repetitions:**
  - Run multiple test trajectories (~12–25) for robust statistics.
  - Quantile error bands as in figures (e.g., Figs. 14-31).

---

## 10. Implementation Notes and Caveats

- Use **efficient neighbor search** kernels (e.g., cell-based, kd-tree) supporting large particle counts.
- Implement GNN message passing with **equivariance (SEGNN)** or standard MLP (GNS).
- Handle **position and velocity buffers** properly over time windows for AR modeling.
- Ensuring **stability of long rollouts**:
  - Monitor density deviations, velocity clustering.
  - Use relaxation steps adaptively if artifacts occur.
- **Code structure:**
  - Modularize into dataset loaders, network architectures, physics correction modules, inference routines.
- Use JAX or PyTorch for GPU acceleration, especially for kernel sums and neighbor searches.

---

## Summary of the Roadmap:

- **Data:** Load or generate particle trajectories, ensure features include external forces (smoothed as per standard deviations).
- **Network:** Implement baseline GNS/SEGNN with appropriate features; add explicit external force terms.
- **Physics Correction:**
  - Smoothing external forces.
  - Density correction and clipping.
  - Boundary conditions.
- **Relaxation:** Implement multi-step SPH relaxation with hyperparameters tuned per dataset.
- **Training:** Supervised, autoregressive, with data augmentation; include external force terms explicitly.
- **Inference:** GNN + external force + optional relaxation steps; apply boundary and CL corrections for free surfaces.
- **Evaluation:** Track long-term stability with metrics, visualize particle distributions, and perform ablations to verify impact.

---

This detailed plan allows you to systematically implement each component, tune hyperparameters, and evaluate thoroughly, aiming for faithful reproduction and extension of the paper's results.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular simulation pipeline using JAX for high-performance differentiation and GPU acceleration, leveraging existing open-source libraries. The core components include a GNN backbone (GNS and SEGNN), a force smoothing module based on velocity statistics, an explicit force addition in the dynamics, and an SPH relaxation module with configurable hyperparameters. We will structure the code into separate files for data loading, model architecture, training, inference, and evaluation, with clear interfaces. The main script will coordinate dataset loading, model instantiation, training loop with optional relaxation steps, and evaluation. We will utilize libraries such as JAX (or PyTorch if preferred), e3nn for equivariant GNNs, SciPy for kernel operations, and standard scientific Python libraries for data processing and visualization.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "eval.py",
        "utils.py",
        "sph_relaxation.py",
        "force_smoothing.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str, config: dict)\n        +load_data() -> Dataset\n    }\n    class Dataset {\n        +positions: np.ndarray  # shape: (num_sequences, seq_length, num_particles, dim)\n        +velocities: np.ndarray  # shape: (num_sequences, seq_length, num_particles, dim)\n        +types: np.ndarray       # shape: (num_sequences, num_particles)\n        +external_forces: np.ndarray  # shape: (num_sequences, seq_length, num_particles, dim) or None\n    }\n    class GNNModel {\n        +__init__(params: dict)\n        +predict(accelerations_and_forces: dict, features: dict) -> dict\n        +get_params() -> dict\n    }\n    class ForceSmoothedField {\n        +__init__(velocity_stats: np.ndarray)\n        +apply_force_smoothing(force_field: np.ndarray) -> np.ndarray\n    }\n    class SPHRelaxation {\n        +__init__(hyperparams: dict)\n        +relax.positions: np.ndarray  # shape: (num_particles, dim)\n        +relax.step() -> np.ndarray  # updates positions\n    }\n    class SimulationEngine {\n        +__init__(model: GNNModel, dataset: Dataset, config: dict)\n        +run_rollout(initial_state: dict, steps: int) -> list of dict  # each includes positions, velocities, densities, etc.\n        +apply_relaxation(positions: np.ndarray, steps: int) -> np.ndarray\n        +compute_forces(positions, velocities, external_force_map) -> dict\n    }\n    Main --> DatasetLoader\n    Main --> GNNModel\n    Main --> ForceSmoothedField\n    Main --> SPHRelaxation\n    Main --> SimulationEngine\n    SimulationEngine --> GNNModel\n    SimulationEngine --> SPHRelaxation\n    SimulationEngine --> ForceSmoothedField\n    classMetrics: evaluate() -> dict  # outputs error metrics\n    Main --> classMetrics\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant DB as Dataset\n    participant Mo as GNNModel\n    participant FS as ForceSmoothedField\n    participant S as SimulationEngine\n    participant R as SPHRelaxation\n    participant E as classMetrics\n    M->>DL: instantiate with dataset path and config\n    DL-->>M: load_data() -> Dataset\n    M->>Mo: init with params\n    M->>FS: init with velocity stats\n    M->>S: init with model, dataset, config\n    loop for each initial state\n        S->>S: apply initial conditions\n        S->>S: rollout for N steps, with optional relaxation after each step\n        S->>E: evaluate long-term errors (e.g., MSE, sinkhorn, MAE, etc.)\n    end\n    M-->>main: output evaluation results and visualizations\n    Note over S: During rollout, for each step:\n        - Compute forces via dataset or smoothed external force map\n        - Predict accelerations using GNN + external force\n        - Update positions via semi-implicit Euler\n        - Optional: run SPH relaxation steps\n        - Repeat\n    Note over FS: Force smoothing is applied before force prediction, using velocity standard deviations.\n    Note over R: Relaxation hyperparameters (alpha, beta, loops) are configurable and tuned based on dataset and artifact indicators."
    ,
    "Anything UNCLEAR": "Clarification needed on the exact data format for particle features (positions, velocities, external forces), whether to include any boundary particles explicitly, and detailed hyperparameter ranges for relaxation tuning. Also, confirmation on whether to implement the entire pipeline in JAX for efficiency or allow fallback with PyTorch+PyG."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "jax==0.3.25",
        "jaxlib==0.3.25",
        "numpy==1.21.0",
        "scipy==1.7.3",
        "e3nn==0.3.20",
        "matplotlib==3.4.3",
        "optax==0.1.1",
        "flax==0.6.0",
        "scikit-learn==0.24.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: loads particle sequences, including positions, velocities, particle types, and external force fields if applicable. It handles parsing raw data formats or generated datasets, ensuring proper shape and feature extraction. Dependency: numpy, scipy."
        ],
        [
            "utils.py",
            "Provides common utility functions: velocity statistic calculations, Gaussian convolution operations, neighbor search routines (could use SciPy KDTree), kernel functions for SPH (e.g., quintic spline), and boundary condition utilities."
        ],
        [
            "model.py",
            "Implements GNN model classes: GNS and SEGNN based on e3nn and Flax. Defines forward pass interfaces accepting particle features, including position, velocity, external force features, and particle type encoding. It must support loading pretrained parameters and the explicit force split logic as in Eq. 2."
        ],
        [
            "force_smoothing.py",
            "Contains functions to compute smoothed external force fields: calculate velocity std deviations, convolve with Gaussian kernels, and apply the smoothing to force maps. Uses SciPy stats for erf approximation and numpy for standard deviations."
        ],
        [
            "sph_relaxation.py",
            "Defines SPHRelaxation class: performs position-only corrections via pressure and viscosity terms. Supports configurable hyperparameters ($\alpha$, $\beta$, relaxation steps, kernel radii). Implements neighbor search, density computation, pressure correction, and position update steps with optional iterative loops."
        ],
        [
            "trainer.py",
            "Handles training loop: loads dataset, initializes model, optimizer (Optax), and hyperparameters. Implements autoregressive training with explicit external force features, density/pressure regularization, and optional relaxation step application during training. Supports checkpointing and hyperparameter tuning."
        ],
        [
            "evaluation.py",
            "Provides long-term rollout evaluation: runs simulation with learned model, applies optional relaxation after each step, computes metrics (MSE, Sinkhorn, MAE, Dirichlet, Chamfer). Supports plotting and result summaries."
        ],
        [
            "main.py",
            "Main entry point: initializes dataset loader, models, and hyperparameters. Runs training, then performs long-horizon rollouts with optional relaxation. Calls evaluation functions and generates visualizations. Manages workflow dependencies: dataset -> model -> training -> inference -> evaluation."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "utils.py",
        "model.py",
        "force_smoothing.py",
        "sph_relaxation.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "None required. All components are internal or data pipeline routines, no REST API needed.",
    "Shared Knowledge": "Common utilities such as neighbor search, kernel functions, velocity statistics calculations, and configuration-driven hyperparameters for relaxation and smoothing. Data consistency checks and boundary correction routines are shared utilities.",
    "Anything UNCLEAR": "Clarify details about the dataset formats, including whether external force fields are provided per particle or globally, and whether boundary particles are explicitly modeled. Confirm whether the entire pipeline should be implemented in JAX or if PyTorch with PyG is acceptable for models and neighbor search routines."
}

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  path: datasets/lagrangian_fluid
  sequence_length: 400  # Length of the test trajectories to evaluate long-term stability
  training_subsequence_interval: 100  # Frequency of subsequence sampling for training

model:
  type: GNS  # Options: GNS, SEGNN
  params:
    hidden_dim: 128
    num_layers: 10
    equivariance: false  # true for SEGNN, false for GNS
    particle_type_embedding: true

training:
  optimizer: adam
  learning_rate: 0.001
  batch_size: 64
  epochs: 100  # Total training epochs (tuning may be required based on validation performance)
  weight_decay: 1e-6
  gradient_clip_norm: 1.0
  data_augmentation: true  # Adds walk noise and pushforward tricks (as in Toshev et al., 2024)
  loss_weights:
    position_mse: 1.0
    velocity_mse: 0.1
    density_mae: 0.1

hyperparameters:
  external_force_smoothing:
    sigma_scale: 0.025  # Standard deviation scale based on dataset velocity stats
  relaxation:
    alpha: 0.03  # Pressure strength hyperparameter
    beta: 0.0    # Viscosity hyperparameter (set to zero if no viscosity correction)
    relaxation_steps: 3  # Number of relaxation iterations during inference
  neighbor_search:
    cutoff_radius: 1.5  # Typical radial cutoff for neighbor search
    relaxation_cutoff_radius: 3.0  # Larger cutoff for SPH relaxation
  force_field:
    external_force_field: true  # Whether external forces are present and should be modeled
    force_smoothing_method: gaussian  # Method: gaussian or erf approximation

evaluation:
  rollout_steps: 400
  metrics:
    position_mse: true
    sinkhorn_divergence: true
    kinetic_energy_mse: true
    density_mae: true
    dirichlet_energy: true
    chamfer_distance: true
  evaluation_trials: 12  # Number of trajectories to assess long-term stability
  visualization: true  # Generate particle distribution plots and density maps

checkpoint:
  save_dir: checkpoints/
  save_frequency: 10  # Save model every 10 epochs
  resume: false      # Whether to resume training from checkpoint

misc:
  debug: false
  random_seed: 42
```

---

**Note:** Full configuration is available in `planning_config.yaml`
