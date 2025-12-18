# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Based on the provided content, I will create a detailed, structured plan outlining the core methodology, experimental setup, hyperparameters, and theoretical foundation necessary to reproduce the work described in the paper on "Advancing Training Efficiency of Deep Spiking Neural Networks through Rate-based Backpropagation." The plan will include specific steps, critical details, and considerations for implementing both the training algorithms and evaluation protocols.

---

## 1. Methodology Outline

### A. Core Concept
- The paper proposes *rate-based backpropagation (rate-BP)* as an approximation to BPTT, leveraging *rate coding* (spike frequency averages) to reduce the temporal complexity and memory overhead.
- The rate-based approach focuses on *averaged spike activity*—i.e., firing rates—rather than detailed temporal spike sequences.
- It decouples temporal dependencies, enabling *single spatial backpropagation* over the averaged activity, significantly improving scalability.
  
### B. Neuronal Dynamics
- The neuron model used is a *Leaky Integrate-and-Fire (LIF)* with continuous membrane potential dynamics:
  
  \[
  u_t^l = \lambda (u_{t-1}^l - V_{th} s_{t-1}^l) + W^l s_t^{l-1}
  \]
  where:
  - \( u_t^l \): membrane potential
  - \( s_t^l \): spike (binary)
  - \( \lambda \): decay factor
  - \( V_{th} \): threshold
  - \( W^l \): synaptic weights
- Surrogate gradients approximate the non-differentiable spike function \( H(\cdot) \) (Heaviside step).

### C. Rate Coding Approximation
- Define *firing rate* at layer \(l\) as:
  \[
  r^l = \frac{1}{T} \sum_{t=1}^T s_t^l
  \]
- The forward pass in the rate approximation:
  \[
  c^l \approx \mathbb{E}[I_t^l] = W^l r^{l-1}
  \]
- The neural dynamics are approximated using the *average rates* instead of detailed spike trains.

### D. Gradient Computation
- Derive the gradient of loss \(\mathcal{L}\) with respect to weights \(W^l\) using *rate approximations*:
  
  \[
  \left( \nabla_{W^l} \mathcal{L} \right)_{rate} \approx \frac{\partial \mathcal{L}}{\partial c^l} r^{l-1}
  \]
  
- The key innovation is the *gradient approximation* for the neuron dynamics, involving *bandwidth-limited* or *surrogate gradients* that depend on the firing rate \(r\).

### E. Handling Temporal Dependencies
- The method approximates neuron activity over the entire sequence duration by *mean estimates*, *ignoring detailed temporal dependencies*.
- Empirically justified by theorem 1 and 2, asserting the *independence or bounded error* of the approximations.
- For *error backpropagation*, calculate auxiliary variables (\(e_t^l, g_t^l, \rho_t^l\)) during forward simulation, which serve as eligibility traces for the *rate-based gradient*.

### F. Implementation of the Algorithm
- Two modes:
  - **Rate \(^{; M}\)** (multi-step): backward pass per epoch uses *full sequence*; eligibility traces and gradients are accumulated over all T.
  - **Rate \(^{; S}\)** (single-step): backward occurs timestep-by-timestep, with eligibility computed at each step.
- The *computational graph* is simplified to *spatial backpropagation* of rate-based gradients, avoiding full temporal unfolding.

### G. Neural Network Architecture
- Utilize *spiking neuron models* (e.g., LIF with surrogate gradient).
- Integrate *batch normalization (BN)* with rate approximations, including **spatial BN** for single-step mode (\(\mathbf{\hat{BN}}\)) and **temporal BN** for multi-step mode (tdBN).

### H. Surrogate Gradient and Safety
- Surrogate functions (e.g., sigmoid approximations) ensure smooth gradient flow.
- Epsilon (\(\alpha\)) or clipping are advised to prevent unstable gradients.
- Implement *safety margins* for numerical stability (e.g., normalization, clipping).

---

## 2. Experimental Setup and Procedure

### A. Datasets & Data Preprocessing
- **CIFAR-10 & CIFAR-100**:
  - Normalize images: zero mean, unit variance.
  - Data augmentation: AutoAugment + Cutout.
  - Input encoding: direct spike encoding (e.g., Poisson or Bernoulli) at each timestep.
- **ImageNet-1K**:
  - Resize and crop images:
    - Training: Random resized crop (224x224), horizontal flip.
    - Validation: Resize to 256, center crop.
  - Use direct spike encoding over \(T\) timesteps.
- **CIFAR10-DVS**:
  - Event-based neuromorphic version.
  - Preprocessing: temporal pooling, normalization.
  - Data augmentation: random flip, jitter.

### B. Network Architectures
- Use *standard deep CNNs or ResNets*:
  - ResNet-18 for CIFAR-10/100.
  - VGG-11 for CIFAR-100.
  - SEW-ResNet-34 for ImageNet.
  - VGG-11-based models for CIFAR10-DVS.
- Adapt neuron models to spike form with suitable surrogate gradient functions.

### C. Hyperparameters
- **Number of Timesteps (T):**
  - Evaluate at various (2, 4, 6, 8, 16, 10 for DVS).
- **Learning Rate:**
  - 0.1 for CIFAR (tuned with warm-up).
  - 0.2 for ImageNet.
  - Use learning rate schedulers (e.g., cosine decay, step decay).
- **Batch Size:**
  - 128 (CIFAR-10/100, DVS).
  - 512 (ImageNet).
- **Weight Decay:**
  - \(5 \times 10^{-4}\) (CIFAR).
  - \(2 \times 10^{-5}\) (ImageNet).
- **Decay/Normalization:**
  - Exponential decay per epoch as per Table 3.
- **Surrogate Gradient Parameters:**
  - Sigmoid with \(\alpha=4\) or similar.
- **Optimizer:**
  - Adam with momentum 0.9; possibly SGD with momentum for comparison.

### D. Training Protocols
- Initialize weights (e.g., Xavier, Kaiming).
- Use *rate \(^{; M}\)* and *rate \(^{; S}\)* modes adaptively.
- Integrate *batch normalization* as specified (tdBN or spatial BN).
- Regularly record:
  - Training cost (time, memory).
  - Accuracy (Top-1, Top-5).
  - Spike rate statistics.
- Validate after each epoch; test at multiple T and evaluate robustness to temporal shuffle.

### E. Evaluation Metrics
- Accuracy (Top-1, Top-5).
- Training time (seconds per epoch).
- Memory consumption (GB).
- Eligibility trace accuracy (correlation to BPTT).
- Efficiency gains (speedup factors).
- Robustness to temporal shuffling (see Fig. 6, Table 2).

---

## 3. Implementation Details & Considerations

### A. Core Components
- Modified neuron module with surrogate gradient support.
- Declaration of variables for eligibility traces (\(e_t^l, g_t^l, \rho_t^l\)).
- Functions for:
  - Rate approximation forward pass.
  - Spatial backprop of rate gradients.
  - Eligibility trace updates.
  - Batch normalization variants.
  - Surrogate gradient calculation.

### B. Algorithmic Workflow
- **Forward pass:**
  - Compute membrane potentials.
  - Generate spikes with surrogate step.
  - Accumulate \(s_t^l\) for rate \(r^l\).
  - Update eligibility traces (\(e_t^l, g_t^l, \rho_t^l\)), per the equations.
- **Backward pass:**
  - Use precomputed eligibility variables.
  - Compute gradients \(\frac{\partial \mathcal{L}}{\partial W^l}\).
  - Update weights.
- **Retries for various T:**
  - Check the effect of increasing T on accuracy, memory, timestep consistency.

### C. Additional Implementation Tips
- Use PyTorch custom autograd functions to implement surrogate gradient.
- For memory efficiency:
  - Use offline computation of eligibility/traces.
  - Employ batch normalization strategies that support rate approximation (spatial or temporal BN).
- Validate theoretical assumptions empirically, e.g., the independence of \(\delta_t^{(s^l)}\) and \(\kappa_t^l\).

---

## 4. Additional Notes and Challenges
- Carefully tune surrogate gradient parameters (\(\alpha\), etc.).
- Validate the approximation bounds via empirical correlation (Figures 3, 5).
- Investigate effects of temporal scrambling as shown in tables.
- Consider additional regularization or stabilization techniques (clipping, normalization) to improve convergence.
- To match the paper’s robustness claims, include experiments for various T and with noisy conditions.

---

## 5. Summary of the Roadmap
- **Implement neuron models with surrogate gradients.**
- **Develop the rate-based approximation pipeline, including forward and simplified spatial backpropagation.**
- **Incorporate eligibility trace computation for both online (single timestep) and offline (full sequence) modes.**
- **Set up datasets with standard normalization, augmentation, and encoding procedures.**
- **Tune hyperparameters guided by the paper’s tables, especially T, learning rates, decay, and optimizer settings.**
- **Validate via accuracy, memory, time, and robustness metrics.**

This roadmap should facilitate accurate re-implementation and thorough evaluation of the proposed rate-based backpropagation method for deep spiking neural networks.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Implement a modular deep spiking neural network training system using PyTorch. Develop core components: neuron model with surrogate gradient support, rate code approximation modules, eligibility trace computation, and batch normalization aligned with rate coding. Use a simple training loop that switches between multi-step (rate M) and single-step (rate S) modes, with support for dataset handling, model definition, and evaluation. Libraries: PyTorch for core neural computation; torchvision for dataset loading and augmentation; custom modules for neural dynamics, eligibility, and BN. Maintain straightforward architecture: separate modules for data loading, model definition, training, and evaluation, focusing on clarity and reproducibility.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "surrogate_gradients.py",
        "neuron.py",
        "bn_layers.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(dataset_name: str, batch_size: int, timesteps: int)
        +load_data() -> Tuple[DataLoader, DataLoader]
    }
    class Model {
        +__init__(architecture: str, num_layers: int, neuron_model: str)
        +forward(x: Tensor, mode: str, T: int) -> Tuple[Tensor, Tensor]
        +get_stats() -> dict
    }
    class Trainer {
        +__init__(model: Model, optimizer: Optimizer, train_loader: DataLoader, val_loader: DataLoader)
        +train(num_epochs: int)
        +save_checkpoint(filepath: str)
        +load_checkpoint(filepath: str)
    }
    class Evaluation {
        +__init__(model: Model, test_loader: DataLoader)
        +evaluate(T: int) -> dict
    }
    class SurrogateGradients {
        +sigmoid(x: Tensor, alpha: float=4.0) -> Tensor
        +apply_surrogate(x: Tensor, mode: str) -> Tensor
    }
    class Neuron {
        +lif_neuron(u: Tensor, s_prev: Tensor, V_th: float, lambda_: float) -> Tensor
        +apply_surrogate_gradient(s: Tensor, u: Tensor, mode: str) -> Tensor
    }
    class BNLayer {
        +SpatialBN(I: Tensor) -> Tensor
        +TemporalBN(I_seq: Tensor) -> Tensor
        +backward_update(...) -> dict
    }
    Main --> DatasetLoader
    Main --> Model
    Main --> Trainer
    Main --> Evaluation
    Trainer --> Model
    Evaluation --> Model
    SurrogateGradients --> Neuron
    Neuron --> (lif_neuron, apply_surrogate)
    BNLayer --> (SpatialBN, TemporalBN)
    Mode switch ('rate_M', 'rate_S') influences forward and backward implementations
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant MD as Model
    participant TR as Trainer
    participant EV as Evaluation
    participant SG as SurrogateGradients
    participant NE as Neuron
    participant BN as BNLayer
    M->>DL: load_data()
    DL-->>M: datasets
    M->>MD: initialize_model(architecture, layers)
    M->>TR: start_training()
    TR->>MD: forward(x, mode, T)
    MD-->>TR: predictions, stats
    TR-->>M: checkpoint saved
    M->>EV: evaluate_model(T)
    EV->>MD: forward(x, mode, T)
    MD-->>EV: metrics
    EV-->>M: report
    Main->>SG: initialize_surrogate()
    SG-->>Neuron: apply_surrogate()
    Mode 'rate_M' triggers full sequence backward updates; 'rate_S' triggers timestep-wise updates.
    Model's forward pass uses neuron and BN modules, switching between spatial or temporal BN.
",
    "Anything UNCLEAR": "Clarify whether the dataset format includes pre-defined spike sequences, how to handle the surrogate functions during training/inference concretely, and specific evaluation hardware constraints that might affect batch size or memory assumptions."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "torchvision==0.14.1",
        "numpy==1.21.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "neuron.py",
            "Contains the implementation of the LIF neuron model, including membrane potential update, surrogate gradient approximation, and spike generation. Dependencies: torch, surrogate_gradients.py"
        ],
        [
            "surrogate_gradients.py",
            "Defines the surrogate gradient functions such as sigmoid approximation, supporting backward calculations for neurons. Dependencies: torch"
        ],
        [
            "bn_layers.py",
            "Implements spatial BN, temporal BN, and their backward update functions according to rate approximations. Dependencies: torch"
        ],
        [
            "dataset_loader.py",
            "Loads datasets (CIFAR, ImageNet, CIFAR-DVS), applies preprocessing, encoding, augmentation, and provides DataLoader objects. Dependencies: torchvision, torch, numpy"
        ],
        [
            "model.py",
            "Defines deep network architectures (ResNet, VGG, SEW-ResNet), integrating neuron modules and BN layers. Dependencies: torch, neuron.py, bn_layers.py"
        ],
        [
            "trainer.py",
            "Handles training loop, including forward with rate approximation, eligibility trace updates, surrogate gradient application, loss computation, and optimizer step. Dependencies: torch, model.py, neuron.py, surrogate_gradients.py"
        ],
        [
            "evaluation.py",
            "Performs inference with the trained model, computes accuracy and spike statistic metrics. Dependencies: torch, model.py"
        ],
        [
            "main.py",
            "Coordinates execution: initializes dataset loader, model, trainer, runs training, and evaluation for different modes and T, manages checkpoints, logs results. Dependencies: dataset_loader.py, model.py, trainer.py, evaluation.py"
        ],
        [
            "utils.py",
            "Provides common utility functions: argument parsing, configuration setup, timing, logging. Dependencies: None"
        ]
    ],
    "Task list": [
        "utils.py",
        "dataset_loader.py",
        "neuron.py",
        "surrogate_gradients.py",
        "bn_layers.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "",
    "Shared Knowledge": "Shared modules for surrogate gradient functions, BN layers, and neuron dynamics; consistent configuration management; timing and logging utilities.",
    "Anything UNCLEAR": "Clarify the exact spike encoding scheme for dynamic datasets like CIFAR-DVS and ImageNet (e.g., Poisson, Bernoulli), and confirm whether the rate approximation should directly replace spike sequences in both training and evaluation phases."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.1        # Initial learning rate, as per Table 3; tuned with decay
  batch_size: 128           # Standard batch size for CIFAR-10/100
  epochs: 300               # Number of training epochs, as used in the experiments
  weight_decay: 5e-4        # Typical weight decay, following common training settings
  decay_schedule: exponential # Decay schedule; use exponential decay per epoch
  decay_rate: 0.95          # Decay factor, as per Table 3
  momentum: 0.9             # Momentum for optimizer
  optimizer: Adam           # Optimizer choice
  surrogate_alpha: 4.0      # Surrogate gradient parameter (sigmoid inverse temperature)
  T: 4                      # Number of timesteps for static datasets (e.g., CIFAR-10/100)
  mode: rate_M              # Default mode, can switch to rate_S for single-step
model:
  architecture: ResNet-18   # Main architecture as per experimental settings
  neuron_type: LIF          # Leaky Integrate-and-Fire neuron model
  # Hyperparameters for neuron:
  V_th: 1.0                 # Threshold voltage
  decay_lambda: 0.95        # Membrane potential decay factor
  surrogate_grad: sigmoid   # Surrogate gradient function
  alpha: 4.0                # Surrogate gradient steepness parameter
dataset:
  name: CIFAR-10            # Default dataset; can be changed to CIFAR-100, ImageNet, CIFAR10-DVS
  input_encoding: direct    # Direct spike encoding
  augmentation: AutoAugment + Cutout
  normalization_mean: [0.4914, 0.4822, 0.4465] # CIFAR normalization
  normalization_std: [0.2023, 0.1994, 0.2010]
  train_split_ratio: 0.8  # 80% training, 20% validation
  test_split_ratio: 0.2
  size: 32x32             # CIFAR image size
training_mode:
  mode: rate_M             # Can switch to rate_S for single-step mode
  sequence_length: 4      # T=4 as default; can be set to other values (2, 6, 8, 16)
  online: false           # Default setting: offline training; change for online modes
evaluation:
  metrics: accuracy       # Top-1 accuracy as primary metric
  evaluation_script: eval.py
```

---

**Note:** Full configuration is available in `planning_config.yaml`
