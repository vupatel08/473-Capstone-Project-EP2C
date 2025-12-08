# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, structured plan for reproducing the methodology and experiments of the paper "Recurrent Distance Filtering for Graph Representation Learning" (GRED). The plan is broken into key sections to ensure clarity, fidelity to the paper, and practical guidance for implementation later.

---

## 1. Overview and Core Objectives

**Goal:** Reproduce GRED—a novel graph neural network architecture combining multiset neighborhood aggregation with a stable linear RNN to encode hierarchical neighborhood information without positional encodings.

**Key contributions to implement:**
- Neighborhood partitioning: for each node, compute shortest-distance neighborhoods (i.e., sets $\mathcal{N}_k(v)$).
- Hierarchical set representation: aggregate node features per neighborhood distance.
- Sequence encoding: pass sequence of neighborhood representations through a diagonal linear RNN (LRU) for each node.
- Final node embedding: non-linear transformation (MLP + GLU) of the RNN final state.
- Deep stacking: multi-layer architecture with skip connections and layer normalization.
- Efficient computation: parallel aggregation and RNN computations.
- Expressiveness & stability: tune eigenvalues of the transition matrix during training.
- Evaluation: node classification on benchmarks, sensitivity analysis on hyperparameters like K.

---

## 2. Data Preparation & Dataset Requirements

**Datasets:**  
- Use publicly available benchmarks mentioned: **CIFAR10, MNIST, PATTERN, CLUSTER, Peptides-func, Peptides-struct**. For initial implementation, focus on small to medium size datasets, or synthetic graphs for debugging.

**Preprocessing Steps:**
1. **Graph construction:**  
   - Load or build the undirected graph $G = (V, E)$.
   - Identify node features (initial features may be identity, degree, or dataset features).
2. **Compute shortest path neighborhoods $\mathcal{N}_k(v)$:**  
   - Use Floyd-Warshall or Dijkstra (if graph is sparse) to compute pairwise shortest distances.
   - For each node $v$, store masks of nodes in neighborhoods at each hop $k$ (from 0 up to $\text{diameter}$ or a pre-defined maximum $K$).
   - Save these neighborhood masks for efficient parallel retrieval.

**Note:**  
- For large graphs, precompute neighborhoods offline to avoid runtime costs.
- For graphs with zero features (like some benchmarks), initialize features as learnable embeddings or constant vectors.

---

## 3. Implementation of Key Components

### 3.1 Neighborhood Partitioning
- For each node $v$, produce sets $\mathcal{N}_k(v)$ using the shortest path matrix:
  - \(\mathcal{N}_k(v) = \{ u \mid d(v, u) = k \}\).
  - Generate a mask tensor $M_{v, k}$: a binary indicator of member nodes.

### 3.2 Node Features & Set Representations
- Input node features: shape $(|V|, d)$.
- For each layer and each node, and each neighborhood set:
  - Gather features of nodes in $\mathcal{N}_k(v)$.
  - Aggregate via a multiset aggregation function (see below).

### 3.3 Multiset Aggregation (AGG)
- Parameterization:
  - Use two MLPs: \(\mathrm{MLP}_1\) and \(\mathrm{MLP}_2\).
  - Aggregate node features in $\mathcal{N}_k(v)$ via summation:
    \[
    \mathbf{x}_{v,k}^{(\ell)} = \mathrm{MLP}_2 \left( \sum_{u \in \mathcal{N}_k(v)} \mathrm{MLP}_1( \mathbf{h}_u^{(\ell-1)} ) \right)
    \]
- **Implementation notes:**
  - Use batched summations over nodes; for each neighborhood set, index features using masks, then sum.
  - Treat neighborhoods with zero nodes as zero or skip.

### 3.4 Sequence Encoding: Linear RNN (LRU)
- Representation:
  \[
  \mathbf{s}_{v,k}^{(\ell)} = \boldsymbol{\Lambda} \mathbf{s}_{v,k-1}^{(\ell)} + \mathbf{W}_{\text{in}} \mathbf{x}_{v,K - k}^{(\ell)}
  \]
- Key features:
  - Diagonal eigenvalues \(\boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1, ..., \lambda_{d_s})\) (trainable).
  - Matrices \(\mathbf{W}_{\text{in}}\): trainable, initialized as a diagonal matrix in the simplified proof.
- Implementation:
  - For each node, process the sequence in parallel over the batch:
    - Generate the sequence \(\{\mathbf{x}_{v,K-k}^{(\ell)}\}\) in batch.
    - Compute the recurrence via diagonal matrix multiplication (element-wise on eigenvalues).
  - Use tensor operations for parallel scans.
  - Initialize \(\boldsymbol{\Lambda}\) with values inside the unit disk (via eigenvalue parameterization, e.g., log-polar).

### 3.5 Final Node Representation
- Extract the last hidden state \(\mathbf{s}_{v,K}^{(\ell)}\).
- Compute:
  \[
  \mathbf{h}_v^{(\ell)} = \mathrm{MLP}_3 \left( \Re [ \mathbf{W}_{\text{out}} \mathbf{s}_{v,K}^{(\ell)} ] \right)
  \]
  - \(\mathbf{W}_{\text{out}}\): trainable complex matrix, or real with specific initialization.
  - Use \(\Re\) (real part) operation after a complex multiplication or operate directly over real-valued parameters, as this may suffice.
- Use a gated linear unit (GLU) for added non-linearity.

### 3.6 Multi-layer Architecture, Skip Connections, and Layer Norm
- Stack multiple GRED layers.
- Employ residual connections: add input to output of each layer.
- Apply layer normalization before residual addition.
- Mix \(\mathrm{MLP}_1\) and output transformations for compactness.

---

## 4. Hyperparameters & Initialization

- **$K$ (neighborhood depth):**  
  - Precompute neighborhoods up to graph diameter or max $K$.
  - Tune based on dataset; larger $K$ for long-range, small $K$ for efficiency.

- **Eigenvalues \(\lambda_i\):**  
  - Initialize within the unit disk (complex with magnitude ≤ 1).
  - Use log-polar parameterization for stability.
  - Constrain during training (e.g., via reparameterization) to ensure eigenvalues stay bounded.

- **MLPs:**  
  - Use width similar to dataset features/dimensions (e.g., 64, 128).
  - One hidden layer with GLU activation as per paper.

- **Number of layers:**  
  - As per dataset: 4–16.
  
- **Dropout:**  
  - Use dropout in MLPs, as in Table 5.

- **Training schedule:**  
  - Use Adam optimizer with learning rate decay.
  - Train for 600–2000 epochs depending on dataset.

- **Batching & parallelization:**  
  - Batch over nodes, precomputed neighborhoods, and sequences.
  - Exploit parallel scan algorithms for recurrence.

---

## 5. Experimental Settings & Evaluation

### 5.1 Classification Tasks
- Node classification benchmarks: CIFAR10, MNIST, PATTERN, CLUSTER, Peptides-func/struct, TUDataset.
- Use standard train/validation/test splits as per original papers.
- Report accuracy or MAE as specified.

### 5.2 Hyperparameter Tuning
- Perform grid search or Bayesian optimization:
  - $K$ (neighborhood size).
  - Number of layers.
  - Hidden dimensions (\(d, d_s\)).
  - Eigenvalue initialization parameters.
  - Dropout rate.
  - Learning rate schedule.

### 5.3 Metrics
- Primary: classification accuracy or MAE.
- Efficiency: training time per epoch, GPU memory consumption.
- Embedding analysis: eigenvalues (see Fig 3, 5).

### 5.4 Sensitivity & Ablation
- Vary $K$ to study performance vs. receptive field.
- Replace RNN with different flavors (standard RNN, LSTM, attention) to verify the benefits.
- Test stability by changing or constraining eigenvalues.

---

## 6. Implementation Considerations

- Use a deep learning framework supporting complex numbers [PyTorch or JAX].
- Modularize:
  - Neighborhood computation.
  - Multiset aggregator.
  - RNN encoding.
  - Final MLP + skip connections.
- Optimize:  
  - Use batch tensor operations.
  - Leverage parallel scans or cumulative sums.
- Save precomputed neighborhoods for each dataset.
- Log eigenvalues periodically to monitor stability and expressiveness.

---

## 7. Summary of the Implementation Roadmap

| Step | Sub-tasks | Notes |
|---|---|---|
| 1 | Load datasets → graph construction | Use provided datasets or synthetic graphs |
| 2 | Compute shortest path neighborhoods → masks | Floyd-Warshall/Dijkstra; save neighborhood masks |
| 3 | Initialize node features | Dataset features or learnable embeddings |
| 4 | Implement AGG function with batching | Use MLPs + summed features |
| 5 | Implement sequence generation → RNN | Parallel scan; diagonal eigenvalues; stabilize eigenvalues |
| 6 | Implement final per-node embedding | GLU + MLP + residual + layer norm |
| 7 | Stack multiple GRED layers | Residual + norm + skip connections |
| 8 | Train & evaluate | Optimizer, hyperparameters, early stopping |
| 9 | Hyperparameter tuning & ablations | $K$, eigenvalues, layer count, width |

---

## 8. Unspecified or Clarification Points

- Exact initialization details for eigenvalues \(\lambda_i\).
- Handling graphs with isolated nodes or zero features.
- Precise normalization strategies in MLPs/GLU.
- Details of the datasets (train/test splits, labels).
- Final layer size for the output (classifier head).

---

This roadmap provides a thorough foundation for later code implementation, ensuring each component of the architecture, from neighborhood extraction to eigenvalue stabilization, is properly designed and integrated.

---

If you'd like, I can prepare a detailed pseudocode or code skeleton based on this outline next!

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We'll build a modular graph neural network system that reproduces the GRED architecture using PyTorch and PyTorch Geometric. The core components include neighborhood computation (preprocessed and stored as masks), multiset aggregation with MLPs, diagonal linear RNN encoding with trainable eigenvalues (enforced via spectral parameterization), and stacked residual layers with layer norm. The pipeline loads datasets, precomputes shortest distances, constructs neighborhood masks, initializes features, processes the graph through multiple GRED layers (each involving neighborhood aggregation, sequence encoding, and feature update), and trains with Adam optimizer. Parallel batch operations will exploit efficient tensor operations for neighborhood aggregation and RNN update sequences. Evaluation on classification benchmarks and ablation on hyperparameters are included for validation. Hyperparameters are configurable via a YAML/JSON config file for easy tuning.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config_path: str)
        +load_data() -> Dataset
        +precompute_neighborhoods()
        +build_model() -> GraphGRED
        +train() -> None
        +evaluate() -> dict
        +run()
    }
    class Dataset {
        +__init__(graph_data: dict, features: torch.Tensor, labels: torch.Tensor)
        +get_shortest_paths() -> Tuple[torch.Tensor, torch.Tensor]
        +get_neighborhood_masks(max_k: int) -> Dict[int, torch.Tensor]
        +get_train_test_split() -> Tuple[Dataset, Dataset]
    }
    class GraphGRED {
        +__init__(hyperparams: dict)
        +forward(node_features: torch.Tensor, neighborhood_masks: dict) -> torch.Tensor
        +stack_layers() -> None
    }
    class GREDLayer {
        +__init__(params: dict)
        +aggregate_neighbors(features: torch.Tensor, masks: dict) -> torch.Tensor
        +encode_sequence(x_seq: torch.Tensor) -> torch.Tensor
        +update_node(features: torch.Tensor, seq_outputs: torch.Tensor) -> torch.Tensor
    }
    class RNNEncoder {
        +__init__(d_model: int, spectral_params: dict)
        +initialize_eigenvalues() -> torch.Tensor
        +encode_sequence(x_seq: torch.Tensor) -> torch.Tensor
        +trainable_eigenvalues: torch.Tensor
    }
    class utils {
        +compute_shortest_paths(graph: dict) -> torch.Tensor
        +create_neighborhood_masks(shortest_paths: torch.Tensor, K: int) -> dict
        +spectral_param_initialize() -> torch.Tensor
        +parallel_scan(recursive_op, seq: torch.Tensor) -> torch.Tensor
    }
    Main --> Dataset
    Main --> GraphGRED
    GraphGRED --> GREDLayer
    GREDLayer --> RNNEncoder
    Main --> utils
    Dataset --> utils
    GraphGRED --> utils
    GREDLayer --> utils
",
    "Program call flow": "
sequenceDiagram
    participant C as main.py
    participant DL as dataset_loader.py
    participant D as utils.py
    participant M as main.py (class Main)
    participant Dst as Dataset
    participant G as GraphGRED
    participant L as GREDLayer
    participant R as RNNEncoder
    participant E as evaluation.py
    C->>DL: load datasets and config
    DL-->>C: dataset object with graph, features, labels
    C->>D: compute shortest paths
    D-->>C: shortest path matrix
    C->>D: generate neighborhood masks up to max K
    D-->>C: neighborhood masks dict
    C->>M: build model with hyperparameters
    M-->>C: model ready
    C->>M: train model with features and masks
    M->>L: forward(features, masks)
    L->>L: aggregate neighbors (parallel batched)
    L->>R: encode sequence (parallel)
    R-->>L: sequence output
    L->>L: update node features
    M-->>C: training complete
    C->>E: evaluate model on test set
    E-->>C: accuracy metrics
    C->>Main: run()
",
    "Anything UNCLEAR": "Clarify if complex number support is required in implementation or if real-valued eigenvalues suffice, and confirm dataset formats and feature initializations."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.1",
        "torch-geometric==2.0.4",
        "PyYAML==6.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Acts as the orchestrator; initializes configuration, calls data loading, precomputes neighborhoods, builds the model, manages training loop, and evaluation."
        ],
        [
            "dataset_loader.py",
            "Handles dataset loading, conversion to graph structures, precomputes shortest path matrices via Floyd-Warshall or Dijkstra, generates neighborhood masks up to max K, and splits data into train/test sets."
        ],
        [
            "utils.py",
            "Provides core utility functions such as shortest path computation, neighborhood mask creation, spectral parameterization (initializing eigenvalues within the spectral norm), and parallel scan algorithms for batching recurrence computations."
        ],
        [
            "model.py",
            "Defines the main GraphGRED architecture, stacking multiple GRED layers. Implements components: neighborhood aggregation (with MLP), sequence encoding (diagonal linear RNN with trainable eigenvalues), residual connections, layer normalization, and final node embedding pipeline."
        ],
        [
            "trainer.py",
            "Implements the training loop: managing batching, loss computation (classification or MAE), backpropagation, optimizer steps, parameter scheduling, and logging. Uses a flexible hyperparameter setup."
        ],
        [
            "evaluation.py",
            "Contains evaluation routines on validation/test sets, computing accuracy or MAE, and possibly visualizations of eigenvalues or learned filters."
        ]
    ],
    "Task list": [
        "requirements.txt update",
        "dataset_loader.py",
        "utils.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (no external API required; local modules only)",
    "Shared Knowledge": "Neighborhood computation results are precomputed and stored as masks; eigenvalues parameterized via spectral normalization, shared across layers. The aggregation function is parameterized by MLPs. Recurrence uses diagonal matrices with learnable eigenvalues for stability and expressiveness.",
    "Anything UNCLEAR": "Clarify preferred hardware setup for large datasets; whether to support real or complex eigenvalues for the diagonal recurrence; exact dataset features or need for synthetic data generation."
}

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: "CIFAR10"  # Or "MNIST", "PATTERN", "CLUSTER", "Peptides-func", "Peptides-struct", etc.
  path: "path/to/dataset"  # Replace with actual dataset path if needed

model:
  num_layers: 8  # Number of GRED layers
  neighborhood_K: 4  # Neighborhood depth K
  hidden_dim: 64  # Dimensionality of node features after first layer
  state_dim: 64  # Dimensionality of the linear RNN hidden state
  out_dim: 64  # Dimension of final node embeddings
  
training:
  learning_rate: 0.001  # Default starting learning rate
  batch_size: 32  # Batch size for node/graph minibatch
  epochs: 600  # Number of training epochs
  weight_decay: 0.0001  # Regularization
  dropout_rate: 0.2  # Dropout rate for MLPs
  weight_decay: 1e-4  # Weight decay for optimizer
  
optimization:
  optimizer: "Adam"  # Optimization algorithm
  scheduler: "ExponentialDecay"  # Optional, if scheduler is used
  lr_decay_rate: 0.99  # Decay rate per epoch, optional
  lr_decay_steps: 1000  # Decay steps, optional

initialization:
  eigenvalues:
    type: "spectral"  # Use spectral parameterization (log-polar or uniform within spectral radius)
    spectral_radius: 0.9  # Eigenvalues initialized within the unit disk for stability

evaluation:
  metrics: ["accuracy"]  # Or "MAE" depending on task
  eval_interval: 10  # Run evaluation every 10 epochs

logging:
  experiment_name: "GRED_CIFAR10"
  log_dir: "logs/"
  save_model: true
  save_dir: "checkpoints/"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
