# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a comprehensive, step-by-step plan to reproduce the methodology and experiments described in the paper "Image Restoration Through Generalized Ornstein-Uhlenbeck Bridge (GOUB)." The plan is organized into key sections aligning with methodology, experimental setup, and implementation details to ensure fidelity and clarity.

---

## 1. **Understanding and Implementing the Core Method (GOUB and Mean-ODE Models)**

### 1.1. Mathematical Foundations
- **Models**: Focus on implementing two models:
  - The *Forward-GOU* diffusion process (from high-quality (HQ) to low-quality (LQ) images).
  - The *Reverse-GOUB* (point-to-point) process for restoration.
  - The *Mean-ODE* variant for efficient sampling.
  
- **Key SDEs**:
  - GOU process (Equation 4): \( dx_t = \theta_t(\mu - x_t) dt + g_t dw_t \)
  - Doob's h-transform modifies the process (Equation 6) to connect fixed points.
  - The stationary (steady) distribution can be eliminated via the h-transform, enabling image-to-image mapping.
  - Derive closed-form transition densities (Equations 8, 9).

### 1.2. Parameterizing the Diffusion Process
- **Parameters needed**:
  - \(\theta_t\):  Schedule (cosine or similar; discretized over T steps).
  - \(g_t\): diffusion coefficient, scaled with \(\theta_t\). Controlled via a schedule.
  - \(\mu\): embedded as a fixed reference vector or input (depends on image). For restoration, assign as **low-quality image (x_T)** or a learned embedding.
- **Discretization**:
  - Split [0, T] into N steps (e.g., N=100).
  - Precompute \(\bar{\theta}_{t}\), \(\bar{\sigma}_t^2\), \(\bar{\sigma}_{t:T}^2\) at each step (using Equations 8, 9, with schedule).
- **Training**:
  - Use maximum likelihood (ML) approach by deriving the negative log-likelihood (Section 1.2), leading to direct parameter estimation.
  - Implement ELBO (Section 3.3, Equation 9), optimizing the neural network \(\epsilon_\theta(\mathbf{x}_t, \mathbf{x}_T, t)\) which predicts the scaled noise.

### 1.3. Neural Network Architecture
- **Network design**:
  - U-Net-like encoder-decoder (without group norm/self-attention as per paper, to match training stability).
  - Inputs:
    - \(\mathbf{x}_t\): current noised image.
    - \(\mathbf{x}_T\): target low-quality image.
    - \(t\): scalar (normalized, schedule parameter).
  - Output:
    - \(\epsilon_\theta\): scaled noise prediction.
- **Training loss**:
  - L1 loss between predicted \(\epsilon_\theta\) and true scaled noise (Section 3.2).
  - Alternatively, combine with ELBO for better probabilistic learning.

### 1.4. Sampling and Reverse Process
- **Reverse SDE / ODE**:
  - Implement solve for the reverse SDE (Equation 9). Use Euler-Maruyama for SDE or deterministic ODE for Mean-ODE (Section 3.3).
  - Starting from low-quality image \(\mathbf{x}_T\), run the reverse process to restore.
  - Parameterize \(\nabla_{x_t}\log p(x_t|\mathbf{x}_T)\) via the trained neural network \(\epsilon_\theta\).

- **Mean-ODE**:
  - Use Equation 13 (Section 3.3) for efficient deterministic sampling:
    \[
    dx_t = \left[\left(\theta_t + g_t^2 \frac{e^{-2\bar{\theta}_{t:T}}}{\bar{\sigma}_{t:T}^2}\right)(x_T - x_t) - g_t^2 \nabla_{x_t}\log p(x_t|x_T)\right] dt
    \]
  - Run for T steps with adaptive step size, integrating from \(\mathbf{x}_T\) back to \(\mathbf{x}_0\).

### 1.5. Training Procedure
- For each training pair:
  - Sample schedule step \(t\).
  - Generate noisy \(x_t\) conditioned on \(\mathbf{x}_0\) (HQ image).
  - Use the ELBO-derived loss to update \(\epsilon_\theta\).
- Use Adam optimizer, initial LR as specified.
- Train for sufficient steps (e.g., 900K) to convergence.

---

## 2. **Experimental Setup and Dataset Preparation**

### 2.1. Datasets
- **Image inpainting, deraining, super-resolution**:
  - Use datasets specified: CelebA-HQ, Rain100H, DIV2K.
- **Inpainting**:
  - Generate masks (thin/fat) to simulate missing areas.
  - Use provided masks or generate simulated masks programmatically.
- **Deraining**:
  - Rain100H and Rain100L, provide degraded images.
- **Super-resolution**:
  - DIV2K downsampled at 4x and 8x (blur + decimation).
  
### 2.2. Data Loading and Preprocessing
- Resize or crop images to 128x128 or relevant size.
- Normalize to [0,1] or [-1,1].
- For inpainting:
  - Mask images with binary masks.
  - Masked regions replaced with zeros or masked out; provide as conditioning input.

### 2.3. Input Conditioning
- For restoration:
  - \(\mathbf{x}_T\): low-quality image (e.g., blurred, noisy, or masked).
  - For inpainting: mask applied to \(\mathbf{x}_0\).
- During training, provide both \(\mathbf{x}_0\) (clean HQ) and \(\mathbf{x}_T\) (degraded).

### 2.4. Metrics
- **PSNR**, **SSIM**, **LPIPS**, **FID**:
  - Compute for quantitative comparison.
  - Implement using existing libraries (e.g., scikit-image for PSNR/SSIM, LPIPS open implementation).

---

## 3. **Implementation Details and Hyperparameters**

### 3.1. Schedules and Parameter Controls
- \(\theta_t\): 
  - Use the cosine schedule or similar (Section 3.2).
- \(g_t\): diffusion scale, proportional to \(\theta_t\), with maximum \(\lambda^2=30\).
- Discretization:
  - N=100-200 steps.
- \(\delta\): small constant (\(0.0005\)) to avoid degeneration at \(t=1\).

### 3.2. Training Hyperparameters
- Batch size: 8–16.
- Learning rate: \(10^{-4}\), decay schedule at 300K/500K steps.
- Optimizer: Adam with \(\beta_1=0.9, \beta_2=0.999\).
- Total steps: ~900,000.
- Snapshot/validation checkpoints: after every 50K steps.

### 3.3. Model Architecture
- Wide U-Net:
  - Multi-scale features.
  - No group norm/self-attention layers initially.
- Input normalization:
  - Images scaled to [-1, 1] or [0, 1].
- Noise prediction:
  - Predict \(\epsilon_\theta\); optionally expand to predict variance scaling if necessary.

### 3.4. Sampling
- For high-fidelity results:
  - Use 100 or more steps.
  - Use deterministic Mean-ODE for fast inference.
- For diverse outputs:
  - Use stochastic SDE, with proper noise during sampling.

---

## 4. **Experiments and Validation**

### 4.1. Reproduction
- Run training as per hyperparameters.
- Evaluate on test datasets:
  - For inpainting: missing masks.
  - For deraining: rainy images.
  - For super-resolution: downsampled images.
- Compute metrics and compare with baseline methods (PromptIR, DDRM, etc.).
- Generate qualitative results: side-by-side images.

### 4.2. Ablation Studies
- Variations:
  - Use score-based vs. OD E sampling.
  - Vary number of steps.
  - Test Mean-ODE vs. stochastic SDE.
  - Different schedules (\(\theta_t\)).

---

## 5. **Additional Notes & Implementation Tips**
- Carefully precompute all schedule-related parameters for efficiency.
- Store \(\mathbf{x}_T\) (the conditioned low-quality or target image) as input at all steps.
- Ensure the optimizer/oracle's learning rate schedule aligns with training steps.
- Use gradient clipping if necessary.
- Use existing codebases for diffusion models as scaffolds (e.g., DDPM, DDIM).

---

**Summary**:  
This roadmap provides mathematical derivations, neural network architecture, training pipelines, dataset preparation, hyperparameters, and evaluation strategies aligned with the paper's methodology and experimental results. It ensures that each aspect—model parameterization, conditioning, schedule, training, and testing—is covered thoroughly for a successful reproduction effort.

---

Would you like me to start drafting code snippets based on this roadmap next?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Build a modular system with core components: a schedule and diffusion parameter setup, a neural network-based noise predictor (U-Net style), and a trainer that handles data loading, training, and inference. Use PyTorch for neural nets and training, Torchvision for datasets, and NumPy for schedule calculations. Leverage existing diffusion model utilities (e.g., from the 'diffusers' library) for SDE/ODE solvers. The system will include functions for schedule generation, ELBO optimization, and deterministic reverse sampling with the Mean-ODE. The training will process image pairs; the sampler will perform image restoration conditioned on low-quality inputs. Evaluation metrics will be integrated for validation.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "sampling.py",
        "evaluation.py",
        "schedule_utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run_experiment() -> None
    }
    class DatasetLoader {
        +__init__(dataset_path: str, batch_size: int, mode: str)
        +load_data() -> Dataset
    }
    class DiffusionSchedule {
        +__init__(steps: int, schedule_type: str)
        +get_schedule() -> dict
    }
    class NoisePredictorNet {
        +__init__(params: dict)
        +forward(x: Tensor, x_T: Tensor, t: float) -> Tensor
    }
    class DiffusionTrainer {
        +__init__(model: NoisePredictorNet, data: Dataset, schedule: dict, hyperparams: dict)
        +train() -> None
    }
    class Sampler {
        +__init__(model: NoisePredictorNet, schedule: dict, use_mean_ode: bool)
        +restore(x_T: Tensor) -> Tensor
    }
    class Evaluation {
        +__init__(model: NoisePredictorNet, dataset: Dataset)
        +compute_metrics(predictions: List[Tensor]) -> dict
    }
    Main --> DatasetLoader
    Main --> DiffusionSchedule
    Main --> NoisePredictorNet
    Main --> DiffusionTrainer
    Main --> Sampler
    Main --> Evaluation
    DiffusionTrainer --> NoisePredictorNet
    Sampler --> NoisePredictorNet
    Evaluation --> NoisePredictorNet
    classDiagram end
    ",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant S as DiffusionSchedule
    participant N as NoisePredictorNet
    participant T as DiffusionTrainer
    participant R as Sampler
    participant E as Evaluation
    M->>S: create schedule
    S-->>M: schedule dict
    M->>DL: load dataset
    DL-->>M: dataset
    M->>N: initialize noise predictor
    N-->>M: model
    M->>T: start training with dataset, schedule
    T->>N: forward(x, x_T, t)
    N-->>T: predicted epsilon
    T-->>M: training completed
    M->>R: instantiate sampler
    R->>N: perform reverse sampling (with or without mean-ODE)
    R-->>M: restored images
    M->>E: evaluate restored images
    E->>N: compute predictions
    N-->>E: metrics
    ",
    "Anything UNCLEAR": "Clarify dataset formats, whether paired images are in standard formats, and how the low/high quality pairs are generated or provided. Also, specify whether inference should always use deterministic Mean-ODE or stochastic SDE."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.10.0",
        "torchvision==0.11.1",
        "numpy==1.21.0",
        "scikit-image==0.18.3",
        "lpips==0.1.3",
        "flax==0.3.6", 
        "scipy==1.7.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "schedule_utils.py",
            "Contains functions to generate the time schedule, compute \(\bar{\theta}_t\), \(\bar{\sigma}_t^2\), and handle schedule discretization (for N steps). Dependencies: numpy, math."
        ],
        [
            "dataset_loader.py",
            "Defines DatasetLoader class. Loads images, applies masks for inpainting, downsampling for super-resolution, adds rain/noise for deraining. Uses torchvision.datasets, torchvision.transforms."
        ],
        [
            "model.py",
            "Defines NoisePredictorNet class, a U-Net style neural network with constructor parameters for depth, channels, no norm/self-attention layers. Uses torch.nn and torch.nn.functional."
        ],
        [
            "trainer.py",
            "Defines DiffusionTrainer class. Handles training loop: loading dataset, sampling schedules, computing ELBO loss (Section 3.2 & 3.3), optimizer updates (Adam). Depends on schedule_utils.py and model.py."
        ],
        [
            "sampling.py",
            "Provides Sampler class with methods for reverse SDE and Mean-ODE sampling, implementing Euler method (for SDE) and deterministic integration (for ODE). Depends on schedule_utils.py and model.py."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class that takes a trained model and dataset, computes PSNR, SSIM, LPIPS, FID using scikit-image and LPIPS library, for validation and comparison."
        ],
        [
            "main.py",
            "Entry point script. Parses config, initializes schedule, loads dataset, initializes model, runs training, and performs inference with the Sampler. Coordinates execution order."
        ]
    ],
    "Task list": [
        "schedule_utils.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "sampling.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (not applicable, system is local and non-API-based)",
    "Shared Knowledge": "schedule_utils.py provides schedule computations; dataset_loader.py handles dataset variations; model.py defines neural network; trainer.py uses ELBO loss and schedule; sampling.py handles reverse process with deterministic and stochastic solvers; evaluation.py computes metrics for results.",
    "Anything UNCLEAR": "Clarification needed on the expected image sizes and data augmentation preferences, as well as whether inference always uses deterministic Mean-ODE or stochastic sampling."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0001  # As per typical diffusion training, the paper uses an initial LR of 10^-4
  batch_size: 8        # Standard batch size; adjust based on memory constraints
  total_steps: 900000  # About 900K training steps as indicated
  lr_decay_steps: [300000, 500000, 600000, 700000]  # Decay schedule at these steps
  
model:
  type: UNet
  channels: 64         # Default base channels; modify if needed
  depth: 4             # Depth of U-Net encoder/decoder
  use_self_attention: false  # The paper suggests ignoring self-attention layers
  use_group_norm: false  # No group norm layers as per their architecture
  
schedule:
  schedule_type: cosine  # Use cosine schedule for \theta_t as suggested
  steps: 100             # Number of discretized steps for schedule
  
dataset:
  name: CelebA-HQ
  size: 128x128
  mode: supervised    # Expecting paired data for training
  training_sampling: true  # Whether to augment data with masks or noise
  
restoration:
  lambda_sq: 30         # The steady variance \(\lambda^2\) over 255 range
  schedule_steps: 100   # Number of steps for T in reverse process
  
evaluation:
  metrics: [PSNR, SSIM, LPIPS, FID]
  dataset: CelebA-HQ test set
  
inference:
  steps: 100            # Number of sampling steps for reverse process
  use_mean_ode: true    # Use the deterministic Mean-ODE for fast inference
```

---

**Note:** Full configuration is available in `planning_config.yaml`
