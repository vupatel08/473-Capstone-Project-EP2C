# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Here's a comprehensive, step-by-step plan to reproduce the methodology and experiments described in the paper "Graph Distillation with Eigenbasis Matching" (GDEM). This plan covers all critical technical details, from understanding the core algorithms to preparing the experimental setup, hyperparameters, and evaluation metrics.

---

## 1. Understanding the Core Methodology

**Primary Goal:**  
Distill a small synthetic graph that preserves the spectral properties of the real large graph to enable cross-architecture, efficient GNN training without GNN-dependent spectrum bias.

**Main Components:**
- **Eigenbasis Matching:** Match the eigenvectors (basis) between real and synthetic graphs to preserve global structure.
- **Spectrum Replication:** Use real eigenvalues to construct the synthetic graph's Laplacian for spectrum preservation.
- **Node Features & Eigenbasis Optimization:** Alternately optimize synthetic node features and eigenbasis to minimize spectral discrepancy, subject to orthogonality constraints.
- **Discrimination Constraint:** Enhance class-aware distribution matching by aligning category-level representations.
- **Spectral Approximation Formalization:** Theoretically guarantee the synthetic graph approximates the real graph under restricted spectral similarity.

**Key mathematical elements:**
- Eigenvalues $\{\lambda_i\}$, eigenvectors $\{u_i\}$.
- Regularization terms $\mathcal{L}_e$ (eigenbasis matching), $\mathcal{L}_o$ (orthogonality), and $\mathcal{L}_d$ (discrimination constraint).
- Construction of the synthetic Laplacian/adjacency from spectrum and eigenbasis.
- Total loss: $\mathcal{L}_{total} = \alpha \mathcal{L}_e + \beta \mathcal{L}_d + \gamma \mathcal{L}_o$.

---

## 2. Preparation of the Datasets & Initial Data Processing

**Datasets:**
- **For each dataset:** Citeseer, Pubmed, Ogbn-arxiv, Flickr, Reddit, Squirrel, Gamers.
- **Requirements:**
  - Original graph $G = (\mathbf{A}, \mathbf{X}, \mathbf{Y})$ with node adjacency $\mathbf{A}$, node features $\mathbf{X}$, labels $\mathbf{Y}$.
  - Access to dataset splits: training, validation, test.
  - For reproducibility, document dataset licenses and download from official links provided.

**Processing:**
- Normalize node features as needed (e.g., row-normalization).
- Construct the normalized Laplacian $\hat{\mathbf{L}} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$.
- Compute eigenvalues and eigenvectors of $\hat{\mathbf{L}}$:
  - Extract the bottom $K$ eigenvalues/eigenvectors (eigenbasis) for spectral matching.
  - Store full spectrum for spectrum replication.
- For high efficiency, precompute and store:
  - Eigenvalues $\{\lambda_i\}$.
  - Eigenvectors $\{u_i\}$ (shape $N \times K$).
  - Node features.

**Note:**  
- Use an eigen-decomposition method suited for large sparse matrices (e.g., Lanczos).  
- For very large graphs, select an appropriate $K$ (e.g., 500–5000) based on dataset size and prior results.

---

## 3. Implementation of Eigenbasis Matching and Synthetic Graph Construction

**Eigenbasis Matching ($\mathcal{L}_e$):**
- Match the synthetic eigenvectors $\mathbf{U}_K' \in \mathbb{R}^{N' \times K}$ to the real eigenvectors $\mathbf{U}_K$ using the loss:
  \[
  \mathcal{L}_e = \sum_{k=1}^K \left\| \mathbf{u}_k \mathbf{u}_k^\top - \mathbf{u}_k' \mathbf{u}_k'^\top \right\|_F^2
  \]
- Enforce orthogonality regularization:
  \[
  \mathcal{L}_o = \left\| \mathbf{U}_K'^\top \mathbf{U}_K' - \mathbf{I}_K \right\|_F^2
  \]
- Optimization:
  - Alternately optimize $\mathbf{U}_K'$ (eigenbasis) with an orthogonality constraint; project onto the Stiefel manifold as needed.
  - Use gradient-based updates with a step size $\eta_1$.
- Handle the choice of eigenvectors (e.g., largest or smallest eigenvalues) based on spectral properties observed in the datasets.

**Spectrum Replication:**
- Use real spectrum $\{\lambda_i\}$ to construct synthetic Laplacian and adjacency:
  \[
  \mathbf{L}' = \sum_{k=1}^{K} \lambda_k \mathbf{u}_k' \mathbf{u}_k'^\top,
  \]
  \[
  \mathbf{A}' = \sum_{k=1}^{K} (1 - \lambda_k) \mathbf{u}_k' \mathbf{u}_k'^\top
  \]
- Use these to form the synthetic graph's adjacency and Laplacian matrices.

**Node Features:**
- Initialize $\mathbf{X}'$ (e.g., randomly, or via eigenbasis projection).
- Alternately optimize $\mathbf{X}'$ to minimize spectral discrepancy and feature distribution constraints.

---

## 4. Incorporating Discrimination Constraint ($\mathcal{L}_d$)

- Compute category-level representations:
  \[
  \mathbf{H} = \mathbf{Y}^\top \hat{\mathbf{A}} \mathbf{X}, \quad
  \mathbf{H}' = {\mathbf{Y}'}^\top \sum_{k} (1 - \lambda_k) \mathbf{u}_k' \mathbf{u}_k'^\top \mathbf{X}'
  \]
- Maximize cosine similarity between class representations:
  \[
  \mathcal{L}_d = \sum_{i=1}^C \left( 1 - \frac{ \mathbf{H}_i^\top \mathbf{H}_i' }{ \|\mathbf{H}_i\| \|\mathbf{H}_i'\| } \right) + \sum_{i \neq j} \frac{ \mathbf{H}_i^\top \mathbf{H}_j' }{ \|\mathbf{H}_i\| \|\mathbf{H}_j'\| }
  \]
- Regularize hyperparameters that balance spectral basis matching and class-wise distribution sharing.

---

## 5. Optimization Procedure

- **Loss function:**
  \[
  \mathcal{L}_{total} = \alpha \mathcal{L}_e + \beta \mathcal{L}_d + \gamma \mathcal{L}_o
  \]
- **Optimization steps:**
  - Use Adam optimizer with learning rates $\eta_1$, $\eta_2$ for different components (eigenbasis, node features).
  - Alternately update:
    - Eigenbasis $\mathbf{U}_K'$ and regularize with $\mathcal{L}_o$.
    - Node features $\mathbf{X}'$ with spectral discrepancy and distribution constraints.
  - Use scheduled updates (e.g., update eigenbasis for $\tau_1$ steps, then node features for $\tau_2$ steps, as in pseudocode).
- Continue until convergence or a predefined number of iterations.

---

## 6. Synthetic Graph Construction & Spectrum Preservation

- After optimization:
  - Construct the synthetic Laplacian/adjacency matrices using the final $\mathbf{U}_K'$ and real eigenvalues.
  - Reconstruct adjacency matrix:
    \[
    \mathbf{A}' = \sum_{k=1}^K (1 - \lambda_k) \mathbf{u}_k' \mathbf{u}_k'^\top
    \]
  - Save synthetic adjacency and node features for downstream evaluation.

---

## 7. Downstream GNN Training & Evaluation

**Training:**
- Train a 2-layer GNN (e.g., GCN, SGC, or others as per experiment) on the synthetic graph:
  - Use original training labels.
  - Use the official GNN architecture and hyperparameters (from Appendix tables).
- For cross-architecture evaluation:
  - Repeat training with different evaluation GNNs (e.g., ChebyNet, BernNet, GPR-GNN).

**Evaluation Metrics:**
- Node classification accuracy on real test sets.
- Spectrum similarity metrics (e.g., TV) comparing the spectrum of the synthetic graph to the real graph.
- Variance of GNN performances across architecture variants (for generalization).

**Repetition:**
- Run multiple distillation runs (e.g., 10) for statistically robust results.
- Record mean and std dev of accuracies and spectrum similarity.

---

## 8. Hyperparameters & Specific Implementation Details (to be set per dataset as per Appendix 11)
- Regularization weights: $\alpha$, $\beta$, $\gamma$.
- Spectrum truncation size $K$ (e.g., 500–5000).
- Eigenbasis matching parameters $r_k$: select based on dataset requirements.
- Optimization step sizes: $\eta_1$, $\eta_2$.
- Number of alternating optimization steps and schedule durations.
- Eigen decomposition method: Lanczos or similar.
- Node feature initialization: random or related to eigenbasis.
- Final spectrum matching ensures spectral approximation with theoretical guarantees.

---

## 9. Additional Notes & Considerations
- For large graphs, eigen decomposition and spectrum construction might be computational bottlenecks; consider approximate methods.
- To avoid spectrum bias, ensure mixing of eigenvectors corresponding to different spectral bands.
- The orthogonality constraint $\mathcal{L}_o$ prevents eigenbasis collapse.
- Discrimination loss $\mathcal{L}_d$ addresses class-wise structure preservation, improving node classification accuracy.

---

## Summary of the Roadmap:
- **Preprocessing:** Load datasets, normalize features, compute spectra.
- **Eigenbasis Initialization:** Use eigen-decomposition, pick eigenvectors.
- **Alternating Optimization:**
  - Match eigenbases ($\mathcal{L}_e$) with orthogonality ($\mathcal{L}_o$).
  - Match category-level representations ($\mathcal{L}_d$).
  - Update node features $\mathbf{X}'$ accordingly.
- **Spectrum Construction:** Use real spectrum to generate the synthetic adjacency.
- **Synthetic Graph Use:** Train downstream GNNs and evaluate on real test sets.
- **Repeat:** Multiple runs for statistical significance; document hyperparameters.

---

This detailed plan provides a clear assembly of all crucial steps, hyperparameter choices, and mathematical formulations, setting a solid foundation to later implement the code and ensure experimental reproducibility.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular pipeline leveraging PyTorch and PyTorch Geometric for graph operations, eigen-decomposition, and neural network training. The system will include data loading, spectral decomposition, eigenbasis and spectrum matching, synthetic graph construction, and downstream GNN training/evaluation. Alternating optimization of eigenbasis and node features will be implemented using custom gradient steps with orthogonality constraints. Spectrum replication from real eigenvalues will be used to build synthetic adjacency and Laplacian matrices. The entire process is encapsulated within an orchestrator class coordinating dataset preprocessing, synthetic graph generation, GNN training, and evaluation.",
    "File list": [
        "app.py",
        "dataset_loader.py",
        "spectral_decomposition.py",
        "synthetic_graph_generator.py",
        "eigenbasis_matcher.py",
        "discrimination_module.py",
        "gnn_model.py",
        "train.py",
        "evaluate.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class App {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(dataset_paths: dict)
        +load_data() -> Tuple[Data, Data, Data]
    }
    class SpectralDecomposition {
        +__init__(adj_matrix: torch.Tensor, K: int)
        +compute_eigenbasis() -> Tuple[torch.Tensor, torch.Tensor]  # eigenvalues, eigenvectors
    }
    class SyntheticGraphGenerator {
        +__init__(eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, features: torch.Tensor)
        +construct_synthetic_adjacency() -> torch.Tensor
        +construct_synthetic_laplacian() -> torch.Tensor
        +initialize_node_features() -> torch.Tensor
        +optimize_node_features(objective: torch.Tensor, steps: int) -> torch.Tensor
        +optimize_eigenbasis(real_eigenvectors: torch.Tensor, real_eigenvalues: torch.Tensor, steps: int) -> torch.Tensor
        +build_graph() -> Tuple[torch.Tensor, torch.Tensor]
    }
    class EigenbasisMatcher {
        +__init__(target: torch.Tensor, init_basis: torch.Tensor)
        +match_basis(real_basis: torch.Tensor, steps: int) -> torch.Tensor
        +orthogonality_loss(basis: torch.Tensor) -> torch.Tensor
    }
    class DiscriminationModule {
        +__init__(real_labels: torch.Tensor, real_features: torch.Tensor, synthetic_labels: torch.Tensor, synthetic_features: torch.Tensor)
        +compute_class_representations() -> Tuple[torch.Tensor, torch.Tensor]
        +discrimination_loss() -> torch.Tensor
    }
    class GNNModel {
        +__init__(architecture_type: str, params: dict)
        +train(data: Data, labels: torch.Tensor) -> None
        +predict(data: Data) -> torch.Tensor
    }
    class Trainer {
        +__init__(model: GNNModel, train_data: Data, val_data: Data, labels: torch.Tensor)
        +train(epochs: int) -> None
        +save_best_model() -> None
        +load_model(filepath: str) -> None
    }
    class Evaluator {
        +__init__(model: GNNModel, test_data: Data, test_labels: torch.Tensor)
        +evaluate() -> dict
    }
    class Utils {
        +normalize_features(features: torch.Tensor) -> torch.Tensor
        +compute_spectrum(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        +project_onto_stiefel(basis: torch.Tensor) -> torch.Tensor
    }
    App --> DatasetLoader
    App --> SpectralDecomposition
    App --> SyntheticGraphGenerator
    App --> EigenbasisMatcher
    App --> DiscriminationModule
    App --> GNNModel
    App --> Trainer
    App --> Evaluator
    SyntheticGraphGenerator --> EigenbasisMatcher
    SyntheticGraphGenerator --> Utils
    EigenbasisMatcher --> Utils
    GNNModel --> Utils
    Trainer --> GNNModel
    Evaluator --> GNNModel
    DiscriminationModule --> Utils
",
    "Program call flow": "
sequenceDiagram
    participant A as App
    participant DL as DatasetLoader
    participant SD as SpectralDecomposition
    participant SGM as SyntheticGraphGenerator
    participant EM as EigenbasisMatcher
    participant DM as DiscriminationModule
    participant MN as GNNModel
    participant TR as Trainer
    participant EV as Evaluator
    participant U as Utils
    
    A->>DL: load_data()
    DL-->>A: dataset tuple
    A->>SD: initialize(adj_matrix, K)
    SD->>SD: compute_eigenbasis()
    SD-->>A: eigenvalues, eigenvectors
    A->>SGM: initialize with eigenvalues, eigenvectors, features
    SGM->>SGM: optimize_eigenbasis(real_eigenvectors, real_eigenvalues, epochs)
    SGM->>SGM: initialize_node_features()
    loop alternating optimization
        SGM->>EM: match_basis(real_basis, steps)
        EM-->>SGM: matched_basis
        SGM->>SGM: orthogonality_loss(matched_basis)
        SGM->>SGM: optimize_node_features(objective, steps)
    end
    SGM->>S: build_graph()
    S-->>A: synthetic adjacency and features
    A->>DM: initialize with real and synthetic class features
    DM-->DM: compute_class_representations()
    DM->>SGM: update with discrimination loss
    loop for each GNN architecture
        A->>MN: initialize architecture (e.g., GCN, ChebyNet)
        TR->>MN: train on synthetic graph, epochs
        TR-->>MN: trained model
        A->>EV: evaluate model
        EV->>MN: predict on test data
        EV-->>A: metrics report
    end
    A-->>User: report accuracies, spectrum similarity, and visualization
    ",
    "Anything UNCLEAR": "Clarify the specific dataset formats expected by the loader, the exact hyperparameter ranges, and whether full eigen-decomposition or approximate methods should be used for large graphs."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "torch-geometric==2.2.0",
        "scipy==1.10.0",
        "numpy==1.21.0",
        "sklearn==0.24.2",
        "matplotlib==3.5.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains class DatasetLoader that handles fetching datasets from given URLs, converting raw data into PyTorch Geometric Data objects, normalizing features, and computing the spectrum (eigenvalues/eigenvectors). Depends on datasets, scipy for eigen-decomposition, and torch_geometric for data handling."
        ],
        [
            "spectral_decomposition.py",
            "Contains class SpectralDecomposition that takes adjacency matrix, computes the normalized Laplacian, and performs eigen-decomposition (using scipy.sparse.linalg.eigsh for large graphs). Outputs eigenvalues and eigenvectors. Dependency: scipy, torch."
        ],
        [
            "eigenbasis_matcher.py",
            "Contains class EigenbasisMatcher that takes real eigenvectors and synthetic eigenvectors, and performs basis matching via gradient descent while enforcing orthogonality constraints (via projection or regularization). Depends on torch and optimization routines."
        ],
        [
            "synthetic_graph_generator.py",
            "Contains class SyntheticGraphGenerator, which initializes synthetic features, constructs adjacency/laplacian from spectrum, and performs alternating optimization: spectrum-based construction, eigenbasis matching, and feature optimization. Depends on classes EigenbasisMatcher and utils, and on initial spectrum and eigenvectors."
        ],
        [
            "discrimination_module.py",
            "Contains class DiscriminationModule that computes class-level features for real and synthetic data and calculates the discrimination loss (cosine similarity). Depends on torch.linalg for cosine similarity and class labels."
        ],
        [
            "model.py",
            "Defines class GNNModel with methods for initialization of different architectures (GCN, ChebyNet, etc.), forward propagation, loss calculation, and optimizer setting. Utilizes torch_geometric.nn modules."
        ],
        [
            "train.py",
            "Contains class Trainer managing training loops for GNNs, with methods to train, validate, and save models. Depends on GNNModel class, dataset splits, and torch optimizer."
        ],
        [
            "evaluate.py",
            "Contains class Evaluator which loads trained models and computes accuracy on test data. Uses torch and dataset evaluation functions."
        ],
        [
            "app.py",
            "Main orchestrator that initializes dataset loader, computes spectra, performs eigenbasis matching, builds synthetic graph, trains multiple GNN architectures, evaluates, and plots spectrum similarity metrics. Coordinates across all modules. Top-level entry point."
        ],
        [
            "utils.py",
            "Utility functions: normalize features, project matrices onto Stiefel manifold for orthogonality, compute spectrum similarity metrics, and plot results."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "spectral_decomposition.py",
        "eigenbasis_matcher.py",
        "synthetic_graph_generator.py",
        "discrimination_module.py",
        "model.py",
        "train.py",
        "evaluate.py",
        "app.py",
        "utils.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Use consistent dataset URLs and hyperparameters across modules. Maintain a global config object class to store hyperparameters, dataset paths, and constants used during eigen-decomposition, optimization steps, and regularization weights."
    ,
    "Anything UNCLEAR": "Clarify the expected size of spectrum K for eigenbasis matching, preferred ranges for hyperparameters like alpha, beta, gamma, and specific dataset formats (node features, adjacency matrices) for the loader. Also, clarification on whether to implement full or approximate eigen-decomposition for large graphs."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  citeseer_url: "https://github.com/kimiyoung/planetoid"
  pubmed_url: "https://github.com/kimiyoung/planetoid"
  ogbn_arxiv_url: "https://github.com/snap-stanford/ogb"
  flickr_url: "https://github.com/GraphSAINT/GraphSAINT"
  reddit_url: "https://github.com/williamleif/GraphSAGE"
  squirrel_url: "https://github.com/benedekrozemberczki/MUSAE"
  gamers_url: "https://github.com/benedekrozemberczki/datasets"

training:
  ########################################
  # No explicit hyperparameters provided in the paper.
  # Use default values consistent with common practice:
  optimizer: "Adam"
  learning_rate: 0.001
  epochs: 2000
  batch_size: 128
  weight_decay: 5e-4
  dropout: 0.5
  train_val_test_split:
    train_ratio: 0.6
    val_ratio: 0.2
    test_ratio: 0.2

distillation:
  K: 500  # Use 500 eigenvectors (adjusted per dataset as needed)
  spectrum_truncate_size: 500
  eigenbasis_match_steps: 3000  # Number of optimization steps for eigenbasis
  feature_opt_steps: 3000  # Number of optimization steps for node features
  lambda_e: 1.0  # Regularization weight for eigenbasis matching
  lambda_d: 1.0  # Discrimination loss weight
  lambda_o: 1.0  # Orthogonality regularization weight
  alpha: 1.0  # Weight for eigenbasis matching loss
  beta: 1.0   # Weight for discrimination loss
  gamma: 1.0  # Weight for orthogonality loss
  eigenbasis_match_lr: 1e-3
  feature_update_lr: 1e-3

evaluation:
  gnn_architectures:
    spatial:
      GCN: {hidden_units: 256, layers: 2}
      SGC: {hidden_units: 256, layers: 2}
      PPNP: {hidden_units: 256, layers: 2}
    spectral:
      ChebyNet: {poly_order: 10}
      BernNet: {poly_order: 10}
      GPR-GNN: {poly_order: 10}
  train_epochs: 200
  validation_interval: 10
  test_batch_size: 128

reproducibility:
  random_seed: 42

misc:
  spectrum_eigenvectors: "Bottom 500 or dataset-specific eigenvectors for eigenbasis matching"
  spectrum_eigenvalues: "Real eigenvalues used for spectrum replication"
  visualization_metrics: "Total Variation (TV), accuracy"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
