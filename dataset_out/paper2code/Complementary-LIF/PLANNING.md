# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here's a detailed, comprehensive plan for reproducing the methodology, experiments, and evaluation described in the paper on CLIF neurons for SNNs.

---

### 1. Overall Objective
- Implement the CLIF neuron model with the precise dynamics described.
- Use surrogate gradient-based training (BPTT) with the specific gradient approximations.
- Conduct experiments on various datasets to compare CLIF with LIF and other models.
- Evaluate energy consumption via synaptic operations (ACs and MACs).
- Extend to inference by converting trained CLIF/SNNS to standard LIF for evaluation.

---

### 2. Implementation Strategy

**(A) Neuron Model Implementation**

- **Base models**:
  - Implement vanilla Leaky Integrate-and-Fire (LIF) neurons with soft reset.
  - Implement CLIF neurons according to Algorithm 1 with no learnable parameters.
  - For both, implement mechanisms to compute membrane potential (`u[t]`), complementary potential (`m[t]`), and spike output (`s[t]`) each timestep.
  - Use a binary spike generation via Heaviside (threshold) function.

- **Parameter Settings**:
  - Set threshold `V_th` (default 1), membrane time constant `τ` (hyperparameter swept from 1.1 to 2.0 for ablation).
  - `V_th` is fixed at 1 for simplicity as per paper; `τ` varies.

- **CLIF-specifics**:
  - Maintain both `u[t]` and `m[t]`.
  - Reset process:
    ```python
    u[t] = (1 - 1/τ) * (u[t-1] - V_th * s[t-1]) + W * s[t]
    m[t] = m[t-1] * sigmoid((1/τ) * u[t]) + s[t]
    ```
  - When spike occurs (`s[t]`), subtract `(V_th + sigmoid(m[t]))` from `u[t]`.

**(B) Surrogate Gradient Setup**

- Use the rectangle surrogate function:
  ```python
  dh/du ≈ (1/α) * indicator(|u - V_th| < α/2)
  ```
- Set `α = V_th` (as per the paper).
- For backpropagation:
  - Compute pseudo-gradients according to Eq. (6) and (14).
  - Implement the recursive gradient equations in code, paying attention to the decaying factor `γ = (1 - 1/τ)` for temporal gradients.
  - Use custom autograd functions (if using PyTorch), or manually compute these gradients with a backward pass that maintains the recursive structures.

**(C) Temporal Gradient Vanishing Address**

- Implement the detailed recursive equations for the gradient (`epsilon`, `xi`, `ψ`, etc.) as per the Appendix.
- Use these equations to guide gradient calculations:
  - Ensure that the additional terms contributed by `m[t]` (complementary potential) are included.
  - Confirm that the gradient products `∏ ρ[l][t'-t]` decay slowly, avoiding the vanishing problem depicted.

**(D) Training Loop & BPTT**

- For each input batch:
  - Simulate the network through all timesteps.
  - Store `u[t]`, `m[t]`, `s[t]` for each timestep.
  - Calculate loss at final timestep or combined over time (depending on task).
- For gradient calculation:
  - Backpropagate using the surrogate derivatives.
  - Implement the recursive gradient equations explicitly to prevent gradient vanishing.
- Use optimizer: SGD with momentum or Adam (hyperparameters from the paper: e.g., lr=0.01).
- Use exponential decay or general schedule (if needed).

---

### 3. Datasets & Data Preparation

- **Static image datasets** (CIFAR10, CIFAR100, TinyImageNet):
  - Standard preprocessing: normalization (mean/ std matching DataLoader).
  - Augmentation:
    - AutoAugment (`Cifar10` and `Cifar100`).
    - CutMix (if using).
  - Convert images into pixel streams:
    - For static datasets, replicate each image into a sequence of fixed length (e.g., 6 timesteps).
    - Normalize pixel values to [0,1], then convert to spikes via some encoding (e.g., rate coding or direct thresholding, the paper suggests a spiking input layer).
- **Neuromorphic datasets**:
  - Use event-based datasets (DVS Gesture, DVS-Gesture, etc.).
  - For these:
    - Use provided event streams directly.
    - Preprocessing might include frame windowing, normalization.
- **Simulate input encoding**:
  - If required, implement encoding to generate spike trains from pixel intensities or event streams.
  - Ensure consistent input presentation between models.

---

### 4. Network Architecture & Hyperparameters
- Use the exact architectures as described:
  - ResNet-18 for CIFAR datasets with 6 timesteps.
  - VGG11 for neuromorphic datasets with 20 timesteps.
- For all models:
  - Initialize weights uniformly (e.g., Xavier).
  - Batch size: 128 (CIFAR), 16 (DVS).
  - Learning rate: 0.1, decay over epochs as per schedule.
  - Train for 200 epochs for CIFAR and 300 for neuromorphic datasets.
  - Optimizer: SGD with momentum 0.9.
- Loss:
  - Cross-entropy at final timestep.
  - Possibly sum losses over several timesteps if stability benefits are observed.

**Note:** Perform ablation studies stressed in the paper:
- Vary `τ` from 1.1 to 2.0 in steps.
- Change surrogate derivative parameters (`α`).
- Vary training epoch to observe loss convergence.

---

### 5. Evaluation & Metrics
- **Accuracy**:
  - Measure final accuracy after training.
  - For dynamic experiments, record accuracy at increasing number of timesteps.
- **Loss**:
  - Plot training loss vs epoch, compare CLIF vs LIF.
- **Temporal Gradient Analysis**:
  - Compute gradient norms over timesteps to assess vanishing.
  - Compute autocorrelation of membrane potential for individual neurons.
- **Energy Estimation**:
  - Calculate AC (spiking) and MAC (weights * activations) based on network activity:
    - For each sample:
      - Count spikes (ACs) per neuron / layer.
      - Count weight * activation operations (MACs).
    - Use the formulae in the Appendix (Tables 6 and 7).
- **Inference Converters**:
  - After training, convert CLIF neurons to LIF with fixed bias (e.g., bias corresponding to mem potential offset).
  - Evaluate the accuracy with the fixed parameters for comparison (Table 8).

---

### 6. Experiments & Ablation Studies

- **Accuracy Comparison**:
  - Reproduce Figures 2, 3, 4, 6, 11.
  - Use identical seed setups across models.
  - Plot convergence curves, accuracy vs timesteps.
- **Energy Cost Analysis**:
  - Compute ACs and MACs for each task/model.
  - Confirm the energy savings provided by CLIF’s lower firing rate despite additional computations.
- **Loss and Gradient Analysis**:
  - Track training loss curves.
  - Plot gradient norms over timesteps.
  - Confirm training stability and absence of gradient vanishing in CLIF.
- **Sensitivity Tests**:
  - Vary reset biases (see hyperparameters in Table 8).
  - Vary the threshold or `τ`.

---

### 7. Final Verification & Validation
- Validate that:
  - CLIF produces binary spikes.
  - The recursive gradient equations are correctly implemented.
  - Gradient corrections prevent vanishing.
  - Trained CLIF models match or surpass paper's reported accuracies.
  - Energy consumption estimations align with paper (tables 6-8).
- Cross-verify with simple synthetic inputs to ensure dynamic simulation correctness.

---

### 8. Summary
- Implement class-based neuron models (LIF and CLIF).
- Set up dataset loaders with input encoding.
- Use BPTT with surrogate gradients; explicitly implement the recursive gradient equations for temporal gradients.
- Train models with identical hyperparameters to the paper.
- Collect metrics for accuracy, energy, and gradient analysis.
- Perform ablation over `τ`, reset bias, and surrogate parameters.
- Conclude with inference edge conversion.

---

This plan details every step, equation, network setting, and experiment needed for faithful reproduction of the paper’s methodology and results.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a PyTorch-based modular framework where neurons, their dynamics, and training are encapsulated in classes. The core will implement the CLIF neuron with recursive gradient computation, using custom autograd functions or explicit manual gradient updates as needed. The network will be constructed using nn.Module, using standard CNN architectures adapted for temporal inputs. Data loaders will process datasets into sequential spike trains, with optional augmentation. Training will employ BPTT with surrogate gradient approximations, incorporating the detailed recursive gradient equations to prevent vanishing. Evaluation will include accuracy, autocorrelation, and energy estimations, with conversion routines for inference. Hyperparameters and experimental configs will be stored in JSON/YAML files for reproducibility.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "neuron.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run() -> None
    }
    class DatasetLoader {
        +__init__(config: dict)
        +load_data() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
    }
    class SpikeEncoding {
        +__init__(params: dict)
        +encode(images: torch.Tensor) -> torch.Tensor
    }
    class NeuronBase {
        +__init__(params: dict)
        +forward(inputs: torch.Tensor, time_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        +compute_gradients(loss: torch.Tensor) -> None
    }
    class CLIFNeuron (NeuronBase) {
        +__init__(params: dict)
        +reset_state() -> None
        +update_state(input_spikes: torch.Tensor) -> torch.Tensor
        +compute_gradients(loss: torch.Tensor) -> None
    }
    class CNNModel (nn.Module) {
        +__init__(params: dict)
        +forward(x: torch.Tensor) -> torch.Tensor
    }
    class SpikingResNet (CNNModel) {
        +__init__(params: dict)
        +forward_spike_sequence(inputs: torch.Tensor, time_steps: int) -> torch.Tensor
    }
    class Trainer {
        +__init__(model: NeuronBase, data_loader: DatasetLoader, optimizer: torch.optim.Optimizer, device: torch.device)
        +train() -> None
        +save_checkpoint(path: str) -> None
        +load_checkpoint(path: str) -> None
    }
    class Evaluation {
        +__init__(model: nn.Module, dataset: Union[torch.utils.data.Dataset, DataLoader], device: torch.device)
        +evaluate() -> dict
        +energy_estimate() -> dict
        +convert_to_LIF() -> nn.Module
    }
    class GradientCalculator {
        +__init__(model: NeuronBase)
        +compute_recursive_gradients(loss: torch.Tensor) -> None
        +apply_surrogate_gradient(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor
    }
    Main --> DatasetLoader
    Main --> NeuronBase
    Main --> CNNModel
    Main --> Trainer
    Main --> Evaluation
    NeuronBase <|-- CLIFNeuron
    Trainer --> model: NeuronBase
    Evaluation --> model: nn.Module
    GradientCalculator --> NeuronBase
    CNNModel <|-- SpikingResNet
    CNNModel <|-- VGG11 (if implemented)
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant EN as Encoder
    participant NM as NeuronModel
    participant TR as Trainer
    participant EV as Evaluation
    M->>DL: load_data()
    DL-->>M: train_dataset, test_dataset
    M->>EN: initialize encoding
    EN-->>M: encoded_train, encoded_test
    M->>NM: instantiate neuron model (CLIF or LIF)
    NM-->>M: neuron_instance
    M->>TR: initialize(model=neuron_instance, data_loader=train_loader, optimizer=optimizer, device=device)
    TR->>NM: forward(inputs, time_steps)
    NM-->>TR: spike_sequence
    TR->>TR: compute loss
    TR->>GradientCalculator: compute_gradients(loss)
    GradientCalculator-->>TR: gradients
    TR->>optimizer: step()
    loop epochs
        TR->>TR: train iteration
        TR->>NM: update state
        ...
    end
    M->>EV: evaluate(model=test_model, dataset=test_dataset)
    EV->>NM: forward_spike_sequence()
    EV-->>M: report metrics, energy estimates
    EV->>EV: convert model for inference
    ",
    "Anything UNCLEAR": "Clarify whether the datasets are preprocessed with specific input encoding schemes. Confirm hyperparameter sweeps for τ and surrogate parameters. Clarify whether to include mechanisms for energy calculation based on activity logs or underlying hardware simulation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "numpy==1.21.0",
        "pyyaml==6.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Class DatasetLoader: loads datasets, applies preprocessing and encoding into spiking sequences; dependent on dataset specifications and encoding schemes."
        ],
        [
            "neuron.py",
            "Class CLIFNeuron: implements CLIF neuron dynamics, state updates, and custom backward functions for recursive gradient computation; depends on neural dynamics equations from the paper."
        ],
        [
            "model.py",
            "Class SpikingResNet or CNNModel: constructs the network architecture with CLIF/CALIF neurons; depends on dataset shape and hyperparameters."
        ],
        [
            "trainer.py",
            "Class Trainer: manages training loop, performs BPTT, applies surrogate gradients and recursive gradient equations, depends on model and neuron modules."
        ],
        [
            "evaluation.py",
            "Class Evaluation: runs inference, calculates accuracy, autocorrelation, energy estimates, and converts models for inference, depends on trained model and logging utilities."
        ],
        [
            "main.py",
            "Main script: initializes dataset loader, encoding, model, trainer, and evaluation modules; sequences the tasks and triggers training, validation, and conversion steps."
        ],
        [
            "utils.py",
            "Utility functions: include activation functions, surrogate gradient implementation, energy calculation functions, and helper routines, shared by trainer and evaluation modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "neuron.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "main.py"
    ],
    "Full API spec": "N/A - All modules are internally called; no external API is required.",
    "Shared Knowledge": "Utilities: surrogate gradient functions, energy calculation formulas, dataset encoding schemes; configuration parameters for hyperparameters, dataset paths, and training settings are shared across modules.",
    "Anything UNCLEAR": "Clarify dataset-specific encoding details (e.g., pixel-to-spikes, event streams), and confirm the exact hyperparameter sweep values for τ and surrogate parameters. Also, confirm whether energy calculation logs are available or need to be simulated."
}

## 4. Configuration

```yaml
## config.yaml
dataset:
  name: CIFAR10
  dataset_path: ./data/cifar10
  batch_size: 128
  num_workers: 4
  normalization_mean: [0.4914, 0.4822, 0.4465]
  normalization_std: [0.2023, 0.1994, 0.2010]
  encoding_scheme: direct_spike_encoding # placeholder, specify actual scheme in code
  
training:
  optimizer: SGD
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 5e-5
  epochs: 200
  scheduler: step  # placeholder, set schedule as per paper if specified
  step_size: 50
  gamma: 0.1
  surrogate_alpha: 1.0 # set to 1 (threshold V_th), modify if sweep needed
  time_constant_tau: 1.5 # default, sweep over 1.1 to 2.0
  seed: 2022

model:
  architecture: ResNet18
  timesteps: 6
  input_channels: 3
  num_classes: 10

neuron:
  type: CLIF
  threshold: 1.0
  tau: 1.5
  reset_bias_base: 0.0  # for initial, tune for ablation if needed
  
evaluation:
  metrics: accuracy
  energy_estimation: true
  convert_for_inference: true

logging:
  save_dir: ./checkpoints
  log_interval: 10
  verbose: true
```

---

**Note:** Full configuration is available in `planning_config.yaml`
