# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, structured plan to reproduce the proposed methodology and experimental setup from the paper "Mitigating Oversmoothing Through Reverse Process of GNNs for Heterophilic Graphs". This plan emphasizes the core methodological innovations, experimental details, and hyperparameters, enabling precise implementation later.

---

# 1. Overview & Core Methodology

**Main Concept:**  
- Use an *inverse (reverse) diffusion process* of message passing to produce more distinguishable node representations, especially suited for heterophilic graphs where traditional GNNs over-smooth.
- Combine representations from both forward (standard diffusion) and reverse (inverse process) pathways at the prediction head.
- Implement three variants of GNNs with the reverse process:
  - GRAND (diffusion-based, with attention matrix)
  - GNNs with residual connections (GCN, GAT, GAT with attention)
- The reverse process is grounded in diffusion equations (heat equation) and is implemented with fixed-point iterative methods, ensuring invertibility and stability.

---

# 2. Key Technical Components & Implementation Strategy

### A. Forward Diffusion (Heat Diffusion Model)  
- **Node features:** \(\mathbf{X}^{(0)}\) (initial)
- **Diffusion dynamics:**  
  \[
  \frac{\partial \mathbf{X}(t)}{\partial t} = (\mathbf{A}(\mathbf{X}(t)) - \mathbf{I}) \mathbf{X}(t)
  \]
- **Solution at timestep \(T_F\):**  
  \[
  \mathbf{X}(T_F) = \mathbf{X}(0) + \int_0^{T_F} \frac{\partial \mathbf{X}(t)}{\partial t} dt
  \]
- **Numerical solution:** Euler method with step size \( \Delta t \).  
- **Attention matrix \(\mathbf{A}(\mathbf{X})\):**  
  \[
  [A(X)]_{ij} = \text{softmax} \left( \frac{(\mathbf{W}_K \mathbf{x}_i)^T \mathbf{W}_Q \mathbf{x}_j}{d'} \right)
  \]
  (Multiple heads averaged if used)

### B. Reverse Diffusion (Inverse process)  
- **Initial node features:** \(\mathbf{X}(T_F)\) (obtained from forward diffusion)
- **Backward process (for timestep \(T_R<0\))** iteratively approximates \(\mathbf{X}(T_R)\):
  \[
  \mathbf{X}(T_R) = \mathbf{X}(0) - \int_{T_R}^0 \frac{\partial \mathbf{X}(t)}{\partial t} dt
  \]
- **Implementation:** fixed-point iteration with \(M\) steps,  
  \[
  \mathbf{X}^{(m+1)} = \mathbf{X}^{(m)} - h(\mathbf{X}^{(m)})
  \]
  where \(h\) is a residual operator based on inverse (gathered via iterative fixed-point scheme).

- **Layer stacking:** multiple reverse layers \((L_R)\). Each layer applies the fixed-point iteration to approximate inverse.

### C. Invertibility in GNNs with Residual Connections  
- The residual GNN layers (GCN, GAT) are designed or regularized to have Lipschitz constant < 1 (contractive property).
- Regularization: normalize weights (\(\mathbf{W}\)) after each gradient step to keep \(\|\hat{\mathbf{A}}\|_2 \|\mathbf{W}\|_F < 1\).
- Use fixed point iteration (Algorithm 1) to invert each residual layer.

### D. Combining Forward & Reverse Representations  
- Compute forward pass: \(\mathbf{X}_f^{(L_F)}\)
- Compute reverse pass: \(\mathbf{X}_r^{(L_R)}\)
- Concatenate features: \(\mathbf{h} = \|_{1 \leq \ell \leq L_F} f^{(\ell)}(\mathbf{X}^{(0)}) \| \text{ and similarly for reverse}
- Prediction head: \(\phi\) (MLP or linear classifier) takes concatenation of both paths.

---

# 3. Experimental Dataset Setup

**Datasets:**  
- **Heterophilic:** Minesweeper, Roman-empire, Questions, etc.  
- **Homophilic:** Cora, Citeseer, PubMed, etc.

**Statistics:**  
- Use dataset statistics as per Table 5: number of nodes, edges, classes, average degree, homophily measures, adjusted homophily.

**Preprocessing:**  
- For heterophilic datasets, confirm labels are balanced or note class imbalance.
- For homophilic datasets, measure edge homophily and adjusted homophily scores.

### Data splits:  
- 10 random splits into train/validation/test with ratios 60/20/20.
- Record standard deviations across splits for reproducibility.

### Graph structure:  
- Load adjacency matrix (\(\mathbf{A}\)), with self-loops added unless specified (for GCN \(\tilde{\mathbf{A}}= \mathbf{A} + \mathbf{I}\)).  
- Ensure sparse matrix representation when implementing matrix multiplications for efficiency.

---

# 4. Hyperparameters & Training Details

### A. Model Hyperparameters (based on paper and Tables)  
- **Layer depths:**  
  - Forward layers: \(L_F \in \{1, 2, 4, 8, 16, 32, 64, 1024\}\) for ablation.
  - Reverse layers: \(L_R \in \{1, 2, 4, 8, 16, 32, 64\}\), with actual value or tuned per dataset.
- **Number of fixed point iterations \(M\):** \(\{8, 16, 32, 64\}\).  
  - For invertibility, often set \(M=8\) as a baseline.
- **Diffusion timestep:**
  - \(T_F\) (forward): e.g., 0.1, 1, 10, or tuned.
  - \(T_R\) (reverse): e.g., \(-0.1, -1, -10\).
- **Learning rate:** \([10^{-5}, 10^{-1}]\), tuned, with initial guesses around \(10^{-3}\).
- **Weight normalization for invertibility:**  
  - After each weight update, normalize \(\mathbf{W}\) so that \(\|\hat{\mathbf{A}}\|_2 \|\mathbf{W}\|_F < 1\). Use Frobenius norm for simplicity.
- **Activation functions:** ReLU, LeakyReLU (with slope 0.2), or ELU, ensuring contractiveness if residual (varies).
- **Optimization:** Adam optimizer with default or tuned hyperparameters.
- **Dropout:** Apply as in Table 2 (e.g., dropout 0.5 for generalization).

### B. Training & Evaluation  
- **Loss function:** Cross-entropy for node classification.
- **Metrics:**  
  - Accuracy on validation/test sets.
  - Standard deviation over multiple splits.
- **Early stopping:**  
  - Stop after 100 epochs or when no improvement for 10 epochs, based on validation accuracy.
- **Number of epochs:** 1000 max, but typically less.

---

# 5. Implementation Details for the Inverse & Diffusion Steps

### A. Forward diffusion implementation  
- Use Euler method:  
  \[
  \mathbf{X} (t + \Delta t) = \mathbf{X}(t) + \Delta t \times (\mathbf{A}(\mathbf{X}(t)) - \mathbf{I}) \mathbf{X}(t)
  \]
- Number of steps: enough to reach the target \(T_F\) (e.g., 10-100 steps).
- Adaptive step size or fixed \( \Delta t \) (e.g., 0.1).

### B. Reverse process (fixed point iteration)  
- Initialize \(\mathbf{X}(T_F)\) from forward diffusion.
- For each reverse layer \(\ell\):
  - Set \(\mathbf{X}^{(0)} = \mathbf{X}(T_F)\).
  - Run fixed point iteration \(M\) times:  
    \[
    \mathbf{X}^{(m+1)} = \mathbf{X}^{(m)} - h(\mathbf{X}^{(m)})
    \]
    where \(h\) involves re-application of inverse residuals.
- Store \(\mathbf{X}(T_R)\) after convergence for downstream use.

### C. Residual layer invertibility regularization  
- During training, normalize weights to satisfy the Lipschitz condition as described (spectral norm constraints).

---

# 6. Prediction & Combining Representations:

- **Final prediction head:**  
  \[
  \hat{\mathbf{Y}} = \phi(\text{concat}(\mathbf{X}_f^{(L_F)}, \mathbf{X}_r^{(L_R)}))
  \]
- Use a simple MLP with one or two layers, consistent with paper settings.
- Cross-entropy loss computed on labeled nodes.

---

# 7. Ablation & Sensitivity Analyses

- Vary **number of reverse layers** \(L_R\) and forward layers \(L_F\) to analyze over-smoothing.
- Explore different **diffusion timesteps** \(T_F, T_R\).
- Analyze **number of fixed point iterations** \(M\) for stability and accuracy.
- Measure **smoothness** via GSL or label distinguishability across depth.

---

# 8. Reproducibility & Logging

- Record detailed logs for:
  - Hyperparameters.
  - Number of layers, iterations, diffusion steps.
  - Time per epoch, total training time.
  - Final accuracy with standard deviations.
- Save model weights after best validation epoch.
- Visualize node representations as in Figures 1, 2, 3, 4.

---

# 9. Summary Checklist for Implementation

- [ ] Implement forward diffusion with attention-based \(\mathbf{A}(\mathbf{X})\).
- [ ] Implement fixed-point iteration for inverse in residual GNN layers.
- [ ] Implement multiple reverse layers with iterative fixed-point methods.
- [ ] Maintain weight normalization constraints.
- [ ] Concatenate forward and reverse representations for prediction.
- [ ] Use datasets as described, with splits and homophily measures.
- [ ] Tune hyperparameters via validation, report results like Table 2 and 3.
- [ ] Visualize representation distinguishability (Figures 1–4).

---

This roadmap provides a comprehensive, detailed guide for the implementation phase of this research. It covers both the mathematical core, practical numerical methods, and experimental configurations. You can now proceed to translate these components into code with clarity and confidence.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular deep learning system leveraging PyTorch and PyTorch Geometric to implement the diffusion-based GNN with reverse process. The architecture will include dataset loaders for heterophilic and homophilic graph datasets, a core model module that encapsulates the diffusion equations, inverse fixed-point iteration, and forward-backward combination, and training/evaluation scripts. The system will support customizable number of forward and reverse layers, diffusion timesteps, fixed point iteration M, and hyperparameters with weight normalization for invertibility, following the paper’s design. Numerical solvers will be based on Euler’s method for forward diffusion, and iterative fixed-point solvers will implement the inverse process, with attention matrices computed as described. The code will be encapsulated in clear, concise classes with well-defined APIs, and a main.py script to coordinate dataset loading, model instantiation, training, and evaluation.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> tuple(sparse_adj: torch.sparse.FloatTensor, features: torch.Tensor, labels: torch.LongTensor, train_mask: torch.BoolTensor, val_mask: torch.BoolTensor, test_mask: torch.BoolTensor)\n    }\n    class DiffusionModel {\n        +__init__(adj: torch.sparse.FloatTensor, in_features: int, hidden_dim: int, num_forward_layers: int, diffusion_time_TF: float, num_reverse_layers: int, diffusion_time_TR: float, max_iterations: int, device: torch.device)\n        +forward_diffusion(x0: torch.Tensor, M: int, dt: float) -> torch.Tensor\n        +inverse_process(xT: torch.Tensor, target_time: float, M: int) -> torch.Tensor\n        +compute_attention(x: torch.Tensor) -> torch.sparse.FloatTensor\n        +normalize_weights()\n        +forward(x0: torch.Tensor) -> torch.Tensor\n        +inverse_pass(xT: torch.Tensor) -> torch.Tensor\n    }\n    class PredictorHead {\n        +__init__(input_dim: int, num_classes: int)\n        +predict_representation(fwd_repr: torch.Tensor, rev_repr: torch.Tensor) -> torch.Tensor\n        +predict_logits(repr: torch.Tensor) -> torch.Tensor\n    }\n    class Trainer {\n        +__init__(model: DiffusionModel, predictor: PredictorHead, data: tuple, hyperparams: dict, device: torch.device)\n        +train(): None\n        +validate(): dict\n        +save_model(path: str): None\n        +load_model(path: str): None\n    }\n    class Evaluation {\n        +__init__(model: DiffusionModel, predictor: PredictorHead, data: tuple, device: torch.device)\n        +evaluate(): dict\n    }\n    DatasetLoader --> main\n    DiffusionModel --> PredictorHead\n    main --> Trainer\n    main --> Evaluation\n    Trainer --> DiffusionModel\n    Trainer --> PredictorHead\n    Evaluation --> DiffusionModel\n    Evaluation --> PredictorHead",
    "Program call flow": "sequenceDiagram\n    participant MD as main.py\n    participant DL as DatasetLoader\n    participant DM as DiffusionModel\n    participant PH as PredictorHead\n    participant TR as Trainer\n    participant EV as Evaluation\n    MD->>DL: load_data()\n    DL-->>MD: return dataset tuple\n    MD->>DM: initialize with adjacency, features, diffusion params\n    MD->>DM: compute attention matrix\n    MD->>DM: normalize weights for invertibility\n    MD->>DM: forward_diffusion(x0, M, dt)\n    MD->>DM: inverse_process(xT, target_time, M)\n    MD->>PH: prepare representation inputs\n    MD->>TR: start training with data\n    TR->>DM: call forward() during training\n    DM-->>TR: return node representations\n    TR->>PH: get predictions\n    PH-->>TR: logits\n    TR-->>MD: loss + backpropagation\n    MD->>TR: update weights\n    TR->>MD: update diffusion and weight normalization\n    MD->>EV: evaluate on test set\n    EV->>DM: inference\n    EV->>PH: generate predictions\n    EV-->>MD: return metrics\n    MD-->>main: training complete, evaluation results\n",
    "Anything UNCLEAR": "Clarification needed on the precise dataset download sources and structure, as well as default hyperparameters for diffusion timesteps and M. Also, whether any implementation-specific techniques (e.g., GPU memory constraints, batching) are required for large datasets."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.1",
        "torch-geometric==2.0.4",
        "scipy>=1.4.1",
        "numpy==1.21.0",
        "tqdm"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class that loads heterophilic and homophilic datasets from files or URLs, processes them into sparse adjacency matrices, features, labels, and masks. It depends on torch and scipy sparse matrices. It prepares data for training, validation, and testing."
        ],
        [
            "model.py",
            "Contains DiffusionModel class responsible for forward diffusion via Euler’s method, compute attention matrices, normalization for invertibility, and inverse process using fixed-point iteration. Implements classes for diffusion equations, attention, and weight normalization. Implements the GNN layers with residual/invertibility constraints."
        ],
        [
            "trainer.py",
            "Implements Trainer class that manages training loop, computes loss, backpropagation, and updates. Calls forward diffusion, inverse process, and combines forward/back representations. Uses optimizer (Adam), manages weight normalization, and checkpoint saving."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class to perform inference on test sets, computes accuracy and other metrics, produces representation visualizations, and reports statistical measures. Relies on saved model states and the diffusion/inverse functions."
        ],
        [
            "main.py",
            "Acts as entry point that initializes dataset loader, diffusion model, predictor head, trainer, and evaluation. Manages overall flow: dataset loading, model configuration, training, validation, and final testing. Imports supported modules, sets hyperparameters, and manages GPU/CPU device allocation."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "main.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (not applicable for core reproduction system)",
    "Shared Knowledge": "Dataset statistics, diffusion equations, attention calculation, invertibility constraints, and fixed point iteration algorithms are shared utilities/functions across model, training, and evaluation modules.",
    "Anything UNCLEAR": "Clarification needed on exact diffusion step sizes, the number of fixed point iterations to use, and handling of large-scale datasets fitting in memory or requiring batching."
}

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001  # Set around 1e-3 based on standard practice; hyperparameter to tune
  batch_size: 256  # Typical batch size for graph neural network training; may adjust for memory
  epochs: 100  # Max epochs; use early stopping based on validation if applicable
  weight_decay: 5e-4  # Regularization for optimizer
  optimizer: Adam  # Optimizer choice
  dropout: 0.5  # Dropout rate for training
  patience: 10  # Early stopping patience epochs

model:
  diffusion_time_TF: 1.0  # Forward diffusion timestep
  diffusion_time_TR: -1.0  # Reverse diffusion timestep (negative for inverse)
  num_forward_layers: 4  # Example depth of forward diffusion layers
  num_reverse_layers: 4  # Example reverse layers, tunable based on experiments
  fixed_point_iter: 16  # Number of fixed point iterations for inverse process
  max_diffusion_steps: 100  # Euler steps count for forward diffusion
  inverse_diffusion_steps: 16  # Fixed point iterations for inverse
  attention_heads: 1  # Multi-head attention count, default 1 (single head)
  hidden_dim: 64  # Hidden dimension size, set based on dataset complexity
  diffusion_step_size: 0.1  # Step size for Euler’s method during forward diffusion
  normalize_weights: true  # Enable weight normalization to ensure invertibility
  
dataset:
  dataset_name: "Minesweeper"  # Or other dataset identifiers used
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  splits: 10  # Number of random splits for validation and robustness
  seed: 42  # Random seed for reproducibility
  
evaluation:
  metrics: [accuracy]  # Evaluation metric for classification
  visualization: true  # Enable visualization of representations (e.g., for Figures 1–4)
  visualization_layers: [forward, reverse, concatenated]  # Layers/representations to visualize
```

---

**Note:** Full configuration is available in `planning_config.yaml`
