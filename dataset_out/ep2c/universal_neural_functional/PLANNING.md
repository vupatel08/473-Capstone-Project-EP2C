# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a detailed, structured plan to reproduce the methodology, experiments, and evaluation described in the paper “Universal Neural Functionals” (UNF), based on the provided content. This roadmap is organized into key sections: understanding the core methodology, implementation details, experimental setup, and evaluation metrics. It emphasizes comprehensive coverage of all critical components to facilitate later code development.

---

## 1. Understand the Core Methodology

### A. Objective:
- Construct permutation-equivariant neural functionals (UNFs) that operate on the weights of *any* neural network architecture, capturing symmetry structures (permutation groups).
- Enable building deep, expressive models that process network weights, gradients, or related features, invariant under neuron permutations.

### B. Key Concepts:
- **Permutation groups (S):** Defined per architecture, acting on weight tensors (e.g., permuting neurons for MLPs or features in CNNs).
- **Weight space (\(\mathcal{W}\)) decomposition:** As direct sums of tensor subspaces (each associated with a layer or component).
- **Equivariant linear layers:** Constructed via a basis derived from the symmetry group (permutations), ensuring layers are maximally expressive and equivariant.
- **Basis construction (Algorithm 1):** 
  - Enumerates valid partitions of tensor indices based on symmetry constraints.
  - Uses index-remapping to generate array operations representing the basis functions.
- **Stacked layers for deep models:** Combine multiple equivariant layers, with nonlinearities preserving equivariance, to form deep UNFs.

### C. Formal guarantees:
- Any linear equivariant map can be expressed as a linear combination over the basis generated.
- The basis spans the entire space of equivariant linear maps.

---

## 2. Implementation Strategy

### A. Representing Weight Spaces:
- **Specification of architecture’s weights:**
  - Use nested dictionaries or data classes describing each layer’s tensors.
  - For each tensor, label dimensions with meaningful identifiers (e.g., “n_in”, “n_out”, “k” for kernels).
- **Symmetry groups:**
  - For each tensor, specify active permutation groups (e.g., for neurons, channels, features).
  - For CNNs, specify which dimensions permute (e.g., input channels, output channels, spatial features).
  - For RNNs, include recurrent weights’ permutation actions.

### B. Basis Construction (Algorithm 1 Steps):
- For each weight tensor or set of tensors:
  1. Identify the index dimensions and their associated permutation groups.
  2. Generate all *valid partitions* of tensor indices, respecting the symmetry constraints.
     - Use combinatorial enumeration over index subsets.
     - For each partition:
       - Label subsets with characters for remapping.
       - Generate array operations (sums over shared indices) to implement the basis functions.
  3. Store basis functions as array operations that produce a tensor of the same shape, representing the equivariant linear maps.

### C. Building Layers:
- Use the basis functions (from 2.) to parameterize the layers:
  - Each layer gets learned coefficients (\(\lambda^{\ell m}_b\)). 
  - Linear combination of basis array operations yields the final equivariant layer.
- Stack multiple layers to form deep, permutation-equivariant (or invariant) models:
  - Use nonlinearities (ReLU, GELU, softplus) that preserve equivariance.
  - Add global pooling for invariant functions.

### D. Handling Multiple Features (Channels):
- Extend the basis construction for features stacked along an extra dimension.
- Modify array operations to sum over or linearly combine across channels with learned matrices, respecting equivariance.

---

## 3. Experimental Setup

### A. Tasks and Architectures:
- **Prediction of network generalization (e.g., RNN success rate):**
  - Dataset: Tiny RNN trained on arithmetic tasks.
  - Inputs: Raw weights of trained models, represented as tensors.
  - Labels: Test success rate (regression target).
  - Model: UNF (or prior methods like STATNN) processing raw weight tensors.
  - Output: Scalar or vector predictions, optionally invariant via pooling.

- **Learned optimizer training (MLP, CNN, RNN, Transformer):**
  - Dataset:
    - For MLP: FashionMNIST.
    - For CNN: CIFAR-10.
    - For RNN/Transformer: LM1B.
  - Network to optimize: Exact architecture as in paper (MLP, CNN, RNN, Transformer).
  - Variables: Weights, gradients, momentum terms.
- **Meta-optimization:**
  - Outer loop: Optimize the parameters of the UNF-based optimizer \(\cdot\) via evolutionary strategies.
  - Inner loop: Standard gradient updates of target models with learned optimizer.
  - Hyperparameters:
    - Learning rates, momentum coefficients, hyperparameter embedding dimensions, batch sizes, inner steps, number of meta-iterations.

### B. Model details:
- Construct the neural functional \(f(\textbf{weights, gradients, momentum terms})\):
  - Use the basis construction to generate equivariant linear layers.
  - Stack layers with ReLU nonlinearities.
  - Use global pooling (sum or mean) to get invariant outputs for some tasks.
  - For the optimizer tasks, define \(f\) as an architecture with:
    - 4 layers, 32 hidden channels, ReLU.
    - For deep models, adapt depth as per experimental design.

### C. Hyperparameters:
- Use the specified or similar hyperparameters:
  - \(\alpha\) and \(\beta\): initial values (e.g., \(\alpha = 0.001\), \(\beta=0.\)1).
  - Learning rates for meta-optimizer: e.g., 0.001.
  - Evolutionary strategies:
    - Population size, noise level (standard deviation), number of steps.
    - Iterations: 50,000 for small tasks; longer for Transformer.

### D. Evaluation Metrics:
- **Prediction tasks:** Correlation (rank correlation), mean squared error, test R-squared.
- **Optimizer tasks:** 
  - Training loss curves (negative log likelihood).
  - Convergence speed (e.g., steps to reach certain loss).
  - Final accuracy or validation success rate.
  - Variance over multiple runs for statistical significance.

---

## 4. Additional Practical Considerations

- **Library dependencies:**
  - Use JAX (recommended), with NumPy, Optax for optimizers, and custom array operations.
  - Leverage JAX's JIT and vmap features for efficiency.

- **Data handling:**
  - Reproduce datasets using numpy or PyTorch equivalents; download and preprocess according to described sizes.
  - For synthetic datasets or models (Tiny RNN arithmetic), generate programmatically.

- **Automatic basis generation:**
  - Implement enumeration of valid partitions robust to tensor shape and symmetry details.
  - Provide options for basis size reduction if needed.

- **Code modularization:**
  - Create modules for:
    - Specification loading (architecture description, symmetry groups).
    - Basis generation algorithms.
    - Layer construction using basis.
    - Stacking and nonlinearities.
    - Meta-optimization loop with evolutionary strategies.

- **Documentation & reproducibility:**
  - Log hyperparameters, seed values.
  - Save model checkpoints during meta-training.
  - Record basis sizes and complexity.

---

## 5. Summary of the Roadmap:
- **Step 1:** Encode architecture specifications and symmetry groups.
- **Step 2:** Implement Algorithm 1 for basis construction:
  - Enumerate valid partitions per tensor.
  - Generate array operations for basis functions.
- **Step 3:** Build equivariant layers with learnable linear combinations of bases.
- **Step 4:** Stack layers with nonlinearities and pooling for deep models or invariance.
- **Step 5:** Prepare datasets:
  - Synthetic for weight-based prediction.
  - Standard datasets (FashionMNIST, CIFAR-10, LM1B).
- **Step 6:** Set hyperparameters and initialize meta-optimizer.
- **Step 7:** Run meta-training using evolutionary strategies:
  - Inner loop: target model training with learned optimizer.
  - Outer loop: meta-update of UNF parameters.
- **Step 8:** Evaluate on held-out datasets, record training curves, and statistical metrics.
- **Step 9:** Analyze basis size, computational cost, and expressiveness.

---

This roadmap ensures a clear, comprehensive pathway to develop an implementation that faithfully reproduces the paper's detailed methodology and experimental results. Let me know if you'd like a more granular pseudocode outline or specific implementation tips next!

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will construct a modular system utilizing JAX for tensor operations and automatic differentiation, with NumPy for basic data handling. The core involves creating a flexible basis generator for permutation equivariant linear layers (Algorithm 1), which will be encapsulated in a class responsible for specifying weight tensors and their symmetries. The basis functions (array operations) will be implemented via JAX primitives, enabling efficient computation and JIT compilation. Stacking these basis-based layers with nonlinearities (ReLU, GELU) will form deep neural functionals (UNFs). The system will include dataset loaders for synthetic weight prediction tasks, and for standard benchmarks (FashionMNIST, CIFAR-10, LM1B), with a focus on easy configuration. Meta-optimization will be handled via evolutionary strategies (ES), implemented with Optax or custom code, to optimize the coefficients of basis combinations for the neural functional layers. The overall architecture is a pipeline: define architecture specs, generate basis layers, compose deep UNFs, define datasets, run meta-training, evaluate performance. The code will be structured in a few key scripts and classes to ensure simplicity and reusability.",
    "File list": [
        "main.py",
        "basis_generator.py",
        "model.py",
        "dataset_loader.py",
        "meta_trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class ArchitectureSpec {
        +layers: List[LayerSpec]
        +symmetry_groups: Dict[str, PermutationGroup]
        +get_tensor_spec(layer_name: str) -> TensorSpec
    }
    class LayerSpec {
        +name: str
        +tensor_shape: Tuple[int, ...]
        +symmetry: str  # e.g., 'permutation', 'recurrent_permutation'
    }
    class TensorSpec {
        +shape: Tuple[int]
        +symmetry_type: str  # e.g., 'permutation'
        +permutation_group: PermutationGroup
    }
    class PermutationGroup {
        +group_type: str  # e.g., 'S_n'
        +size: int
        +act(tensor: Tensor, permutation: Permutation) -> Tensor
    }
    class BasisGenerator {
        +__init__(architecture_spec: ArchitectureSpec)
        +generate_basis(tensor_spec: TensorSpec) -> List[ArrayOperation]
        # ArrayOperation is a callable implementing array sums, index remapping
    }
    class ArrayOperation {
        +apply(tensor: Tensor) -> Tensor
    }
    class NeuralLayer {
        +basis_functions: List[ArrayOperation]
        +coefficients: Tensor  # learnable parameters
        +apply(input_tensors: List[Tensor]) -> Tensor
    }
    class DeepUNF {
        +layers: List[NeuralLayer]
        +nonlinearity: Callable
        +pooling: Callable
        +apply(wt_input: List[Tensor]) -> Tensor
    }
    class DatasetLoader {
        +load_training_data() -> Dataset
        +load_validation_data() -> Dataset
        +load_test_data() -> Dataset
    }
    class Dataset {
        +inputs: List[Tensor]
        +labels: List[Tensor]
    }
    class MetaTrainer {
        +meta_parameters: Dict[str, float]
        +train(optimizer: Optimizer, model: DeepUNF, dataset: Dataset) -> None
        +evaluate(model: DeepUNF, dataset: Dataset) -> Metrics
    }
    class Evaluation {
        +compute_metrics(predictions: Tensor, targets: Tensor) -> Dict[str, float]
    }
    Main --> DatasetLoader
    Main --> ArchitectureSpec
    Main --> BasisGenerator
    Main --> DeepUNF
    Main --> MetaTrainer
    Main --> Evaluation
    DeepUNF --> NeuralLayer
    NeuralLayer --> ArrayOperation
    MetaTrainer --> DeepUNF
    MetaTrainer --> Dataset
    Evaluation --> Dataset
    DatasetLoader --> Dataset
    Dataset --> (inputs, labels)
    ArchitectureSpec --> LayerSpec
    LayerSpec --> TensorSpec
    TensorSpec --> PermutationGroup
    PermutationGroup --> act()
",
    "Program call flow": "
sequenceDiagram
    participant App as Main
    participant Arch as ArchitectureSpec
    participant BG as BasisGenerator
    participant DL as DatasetLoader
    participant Model as DeepUNF
    participant MT as MetaTrainer
    participant Eval as Evaluation
    App->>Arch: load architecture specs & symmetry groups
    Arch-->>App: return specs
    App->>BG: initialize with architecture specs
    BG-->>App: generate basis functions for each tensor
    App->>Model: build deep UNF layers using basis + coefficients
    Model-->>App: initialize model with layers
    App->>DL: load training, validation, test datasets
    DL-->>App: return datasets
    App->>MT: start meta-training with optimizer, model, datasets
    MT->>Model: apply layers to weights (forward pass)
    Model-->>MT: predicted metrics (e.g., success rate)
    MT->>Metrics: record performance
    MT-->>App: return best coefficients and trained UNF
    App->>Eval: evaluate trained UNF on test set
    Eval->>Model: compute predictions
    Model-->>Eval: output
    Eval-->>App: report metrics
",
    "Anything UNCLEAR": "Clarify if the basis functions need specific symmetries for residual or recurrent weights, and whether the array operations should be fully symbolic or optimized for runtime."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "jax==0.4.0",
        "numpy==1.21.0",
        "optax==0.1.7",
        "dm-haiku==0.0.7",
        "mlcollections==0.1.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "basis_generator.py",
            "Contains class BasisGenerator which takes an ArchitectureSpec and generates permutation-equivariant basis functions for each tensor spec based on the index partition enumeration. Implements Algorithm 1, relying on combinatorial enumeration and index remapping to produce array operations. Depends on data structures for architecture specs and tensor shapes."
        ],
        [
            "model.py",
            "Defines class DeepUNF which stacks multiple NeuralLayer instances, each consisting of basis functions (ArrayOperations) with learned coefficients. Applies nonlinearities and pooling; interacts with basis_generator to receive basis functions. Uses hk.Module from Haiku for parameter management."
        ],
        [
            "dataset_loader.py",
            "Provides functions load_training_data(), load_validation_data(), load_test_data() that load datasets (FashionMNIST, CIFAR-10, LM1B) in tensor format compatible with JAX. Handles synthetic weight prediction datasets and real datasets, applying necessary preprocessing."
        ],
        [
            "trainer.py",
            "Implements class MetaTrainer which performs meta-optimization via evolutionary strategies. It initializes the model (DeepUNF), data, and optimizer states; performs inner-loop training for target networks; evaluates performance to compute metrics; updates the coefficients of basis functions using ES. Depends on dataset_loader and model modules."
        ],
        [
            "evaluation.py",
            "Contains class Evaluation or functions that takes model outputs and targets to compute metrics like success rate, correlation, etc. Used during meta-training and final evaluation."
        ],
        [
            "main.py",
            "Entry script that loads architecture specifications, creates basis generator, constructs deep neural functionals (UNFs), loads datasets, initializes meta-trainer, runs meta-optimization loop, and performs final evaluation. Coordinates data flow among modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "basis_generator.py",
        "model.py",
        "evaluation.py",
        "trainer.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "BasisGenerator requires architecture specs with symmetry group actions, tensor shapes, and index labels; model.py interacts with generated basis functions via coefficients; dataset_loader handles domain-specific data formats; trainer.py relies on the API for model application and evaluation metrics.",
    "Anything UNCLEAR": "Clarify if specific hyperparameters or basis sparsification methods are preferred, and whether the architecture specifications need automated parsing from model configs."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001  # Inner optimizer learning rate for meta-optimization of the UNF
  batch_size: 128  # Batch size used during training datasets (e.g., imagery or language datasets)
  inner_steps: 2000  # Number of inner-loop training steps for target models
  meta_iterations: 50000  # Total meta-optimization iterations for updating the UNF coefficients
  meta_population_size: 64  # Population size for evolutionary strategies
  noise_std: 0.01  # Standard deviation of noise in ES for coefficient updates
  optimizer:
    type: "evolutionary_strategy"
    step_size: 0.001
    gamma: 0.9  # Momentum update factor in ES
dataset:
  name: "FashionMNIST"  # Example dataset used for MLP experiment
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1
  image_size: 28
  flatten: true
architecture:
  model_type: "MLP"
  hidden_layers: [600, 600, 600]
  activation: "relu"
  weight_specification: "nested dictionary with shape labels and permutation groups"
sample:
  seed: 42  # Random seed for reproducibility
  experiment_name: "UNF_MLP_FashionMNIST"
  device: "gpu"  # Hardware to run experiments, e.g., "gpu" or "cpu"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
