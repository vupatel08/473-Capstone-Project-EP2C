# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## app.py

# Logic Analysis for app.py

## Purpose Overview
`app.py` serves as the main orchestrator that integrates all modules: data loading, spectral decomposition, graph synthesis (with eigenbasis matching), down-stream GNN training/evaluation, and visualization. Its goal is to execute the entire pipeline, from raw dataset to final performance metrics, ensuring fidelity to the methodology described in the paper and reproducibility according to the provided configuration.

---

## High-Level Tasks & Workflow
1. **Configuration Loading**
   - Load hyperparameters, dataset URLs, and spectral parameters from a config object (or directly from `config.yaml`).
   - Ensure all dataset URLs and hyperparameters are accessible and correctly parsed.
   
2. **Dataset Initialization & Loading**
   - Use `DatasetLoader` class with dataset URLs.
   - Download datasets if necessary.
   - Load datasets into PyTorch Geometric format, including:
     - Node features (`X`)
     - Adjacency matrix or edge list (`A`)
     - Labels (`Y`)
   - Normalize features as needed.
   - Split dataset into train/validation/test sets.
   
3. **Spectral Decomposition**
   - Pass adjacency matrix (`A`) to `SpectralDecomposition`.
   - Compute normalized Laplacian `L`.
   - Perform eigen-decomposition to extract:
     - Eigenvalues (`lambda`) (real spectrum)
     - Eigenvectors (`U`)
   - Select the first `K` eigenvectors/eigenvalues for spectral matching.
   - Save or store these spectral components for downstream use.
   
4. **Initialize Synthetic Graph**
   - Instantiate `SyntheticGraphGenerator` with eigenvalues, eigenvectors, and dataset features.
   - Initialize synthetic eigenbasis (`U'`) using spectrum-informed methods (e.g., eigen-decomposition of initial adjacency).
   - Initialize synthetic node features (`X'`) possibly by:
     - Random sampling
     - Projection via trained MLP (from dataset)
   - Optionally, initialize adjacency matrix (`A'`) and Laplacian (`L'`) from spectrum:
     - Employ spectrum replication: build `A'` and `L'` using `A' = sum_k (1 - λ_k) u'_k u'_k^T` and similar for `L'`.
   
5. **Alternating Optimization for Eigenbasis and Features**
   - For a predefined number of steps (`epochs`):
     - Use `EigenbasisMatcher` to update `U'` to match real eigenbasis:
       - Minimize `L_e` loss
       - Enforce orthogonality via `L_o`
     - Regularly (every `τ_1` or `τ_2` steps):
       - Update node features `X'` to minimize spectral discrepancy and task relevance (for example, via backprop on the classification objective).
   - During optimization:
     - Keep track of iterative loss values.
     - Use gradient steps with specified learning rates (`eigenbasis_match_lr` for eigenbasis, `feature_update_lr` for features).
   
6. **Spectrum Construction & Synthetic Graph Finalization**
   - After optimization, generate the synthetic adjacency (`A'`) and Laplacian (`L'`) matrices directly from the final eigenbasis:
     - `A' = sum_k (1 - λ_k) u'_k u'_k^T`
     - `L' = sum_k λ_k u'_k u'_k^T`
   - Save or export the synthetic adjacency and node features.
   - Store final eigenbasis for analysis or visualization.
   
7. **Classification & Downstream GNN Training**
   - For each architecture specified in the config:
     - Instantiate `GNNModel` with hyperparameters.
     - Use `Trainer` class:
       - Load synthetic graph (`A'`, `X'`, `Y'`) as input data.
       - Train on synthetic data for specified epochs.
       - Optionally, perform validation during training.
   - For cross-architecture testing:
     - Load trained model.
     - Evaluate on real test set:
       - Record accuracy.
       - Record computational metrics.
   
8. **Evaluation & Metrics Collection**
   - Collect final classification accuracy for each architecture.
   - Compute spectral similarity between real and synthetic graphs (e.g., TV or spectrum distance).
   - Compute variance of performances across models/architectures.
   - Save all metrics for analysis.
   
9. **Visualization & Result Plotting**
   - Plot spectral metrics over epochs (e.g., TV at different optimization steps).
   - Plot or visualize synthetic graphs (e.g., spectral distribution, distribution of TV).
   - Optionally, plot node feature embeddings (via t-SNE or PCA) to visualize distribution similarities.
   
10. **Output & Reporting**
    - Summarize classification accuracy, spectrum similarity, and variance.
    - Save graphs (synthetic adjacency features) for record-keeping.
    - Generate plots for spectrum similarity evolution.
    - Print or log hyperparameters and results for documentation.
   
---

## Detailed Step-by-Step Breakdown

### Initialization
- Load `config.yaml`.
- Set random seed (`42` as per reproducibility setting).
- Instantiate dataset loader with dataset URLs.
- Load dataset and preprocess features, adjacency.
- Convert to torch tensors; normalize features.
- Instantiate `SpectralDecomposition` with `A`, select `K` eigenvectors.
- Save spectrum: eigenvalues (`lambda`), eigenvectors (`U`).

### Synthetic Graph Initialization
- Call `SyntheticGraphGenerator` with spectral info:
  - Initialize `U'` (eigenbasis):
    - Use eigen-decomposition of initial adjacency (e.g., SBM or random graph) to seed `U'`.
  - Initialize node features (`X'`):
    - Randomly or via trained MLP (if applicable).
- Construct initial adjacency and Laplacian from spectrum:
  - Use `A' = sum_k (1 - λ_k) u'_k u'_k^T`.
  - Use `L' = sum_k λ_k u'_k u'_k^T`.

### Alternating Optimization Loop
- Loop over `epochs`:
  - **Eigenbasis matching step:**
    - Update `U'` by minimizing `L_e`.
    - Enforce orthogonality with `L_o`.
  - **Node feature update step:**
    - Update `X'` by minimizing spectral discrepancies and task-oriented loss.
  - Log loss values and optional spectrum similarity metrics.

### Final Synthetic Graph Construction
- Use final `U'` and leverage spectrum to build:
  - Final `A'` and `L'`.
- Save `A'`, `X'`, `Y'`, `U'`, `λ`.

### Training & Evaluation
- For each GNN architecture:
  - Load the synthetic graph (`A'`, `X'`, `Y'`).
  - Instantiate model.
  - Use `Trainer` to train the GNN on synthetic data.
  - Record training curves, metrics.
  - Use `Evaluator` to compute test accuracy on the real data.
- Collect performance metrics.

### Visualization & Reporting
- Plot spectrum similarity trajectories over optimization steps.
- Plot distribution matching (e.g., TV comparison).
- Visualize node embeddings if needed.
- Summarize results: accuracy, spectral similarity, variance.
- Save logs, plots, and models for further analysis.

---

## Additional Considerations
- Ensure data splits and labels are correctly aligned between datasets.
- Handle large graphs efficiently—prefer approximate eigen-decomposition.
- Maintain modular calls, from dataset loading to graph generation to downstream training.
- Use consistent hyperparameters in accordance with the default or dataset-specific settings from `config.yaml`.

---

## Conclusion
This logic analysis provides a comprehensive, step-by-step guide for implementing `app.py`. It emphasizes the integration of modules, clarity in the training pipeline, and fidelity to methodology and hyperparameters, enabling reproducible and accurate execution of the GDEM framework described in the paper.

## dataset_loader.py

{
  "dataset_loader.py": [
    "**Purpose and Role**:  
    Implement a class DatasetLoader responsible for downloading, loading, and preprocessing datasets specified by URLs in config.yaml. It converts raw datasets into PyTorch Geometric Data objects, normalizes node features, and computes the spectral decomposition (eigenvalues and eigenvectors) of the graph Laplacian suitable for downstream eigenbasis matching in GDEM.",
    
    "**Dependencies**:  
    - Python packages: torch, torch_geometric, scipy, numpy, sklearn for normalization and spectral computation.  
    - Utilizes dataset download links from provided URLs, requiring standard HTTP requests or dataset-specific loaders.  
    - Assumes datasets are in a standard format compatible with torch_geometric datasets or in a parseable format; if not, custom parsing is necessary.",
    
    "**Class Design**:  
    class DatasetLoader  
    - **Attributes**:
        - dataset_paths (dict): Contains dataset URLs extracted from config.yaml.  
        - data (torch_geometric.data.Data): Loaded dataset object.  
        - spectrum (dict): Contains 'eigenvalues', 'eigenvectors' (both tensors).  
        - Additional: Normalized features, adjacency matrices, split indices.  
    - **Methods**:
        - __init__(dataset_paths: dict): Initialize with URL paths.  
        - download_and_load(): Fetch dataset, parse into torch_geometric Data objects, handle different dataset formats.  
        - preprocess(): Normalize features, construct adjacency matrices, compute normalized Laplacian.  
        - compute_spectrum(K: int): Compute eigenvalues and eigenvectors of the normalized Laplacian, returning top-K eigenvalues/vectors.  
        - get_data(): Return dataset object with all preprocessing done.  
        - get_spectrum(): Return eigenvalues and eigenvectors for spectral matching.  
        - save/load methods (optional): Save preprocessed data and spectra to disk for efficiency.**,

    "**Implementation Logic Steps**:
    1. **Initialization**:
        - Receive dataset URLs from configuration.  
        - Set default dataset download paths, check if datasets are already downloaded or need to be fetched.  
    2. **Downloading datasets**:
        - If datasets are stored locally, proceed; otherwise, download from URLs (e.g., using requests or git clone if necessary).  
        - Ensure datasets are unzipped/extracted if compressed.  
    3. **Loading datasets**:
        - For standard datasets (e.g., Planetoid, OGB), leverage existing torch_geometric dataset classes if compatible.  
        - For custom datasets, implement parsers to read edge lists, features, labels, and splits.  
        - Convert raw data into torch_geometric.data.Data objects, with attributes:
            - x (node features tensor)  
            - edge_index (edge list tensor)  
            - y (labels tensor)  
    4. **Normalization and preprocessing**:
        - Normalize node feature matrix (e.g., row normalization or standard scaling).  
        - Compute degree matrix D, then normalized Laplacian $\hat{\mathbf{L}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$.  
        - Store adjacency matrix in sparse format compatible with scipy or torch.  
    5. **Spectral decomposition**:
        - Use scipy.sparse.linalg.eigsh to compute the smallest or largest K eigenvalues/vectors of the normalized Laplacian.  
        - Since eigsh returns eigenvalues ordered ascending, select appropriate end(s):
            - For capturing global structure, typically use the smallest (or largest, depending on the structure).  
        - Store eigenvalues and eigenvectors as torch tensors for later matching.  
        - For large graphs, ensure eigen-decomposition is feasible; consider approximate methods if performance issues arise.  
    6. **Output**:
        - Provide methods to retrieve:
            - The preprocessed torch_geometric Data object with normalized features.  
            - Eigenvalues and eigenvectors for spectral matching.  
        - Optional: Save preprocessed data and spectral info to disk to avoid recomputation.  
    7. **Additional Functionalities**:
        - Support for different dataset formats if provided datasets deviate from standard torch_geometric datasets.  
        - Handling dataset splits explicitly for training/validation/testing.  
        - Method to re-compute spectrum if needed (e.g., after modifications).  
    8. **Error Handling & Logging**:
        - Log steps such as downloading, preprocessing, and eigen-decomposition.  
        - Handle errors robustly: network errors, incompatible dataset formats, eigen-decomposition failures.  
        - Provide informative exceptions or warnings to guide debugging.**

  "**Design Decisions and Notes**":
  - Use consistent data formats across datasets; implement adapters if datasets differ.  
  - Store datasets and spectral info persistently for efficiency if multiple runs are needed.  
  - Be mindful of large datasets: eigen-decomposition can be expensive; employ approximate methods or batch-wise eigen computation if necessary.  
  - Align feature normalization with the experimental setup described in the paper to ensure reproducibility.  
  - Maintain flexibility for dataset-specific parsing, but default to torch_geometric Dataset classes where possible.  
  - Ensure hyperparameters like 'K' (number of eigenvectors) are configurable and dataset-dependent if needed.  
  - Final output of this class should facilitate downstream spectral computations and synthetic graph initialization in GDEM pipeline."
  ]
}

## discrimination_module.py

{
  "discrimination_module.py": [
    "Purpose: This module implements the DiscriminationModule class, which is responsible for computing class-level representations of real and synthetic graphs and calculating the discrimination loss, encouraging the synthetic graph's features to preserve class-wise distinctions. The loss is based on cosine similarity measures.",
    "Dependencies: Uses torch and torch.linalg for tensor operations and cosine similarity computations. Also depends on class labels provided in tensor form.",
    "Inputs to the class:",
    "  - real_labels (torch.Tensor): Label tensor for nodes in the real graph, shape (N,), with class indices.",
    "  - real_features (torch.Tensor): Node features tensor for the real graph, shape (N, d).",
    "  - synthetic_labels (torch.Tensor): Label tensor for nodes in the synthetic graph, shape (N',), with class indices.",
    "  - synthetic_features (torch.Tensor): Node features tensor for the synthetic graph, shape (N', d).",
    "Methods:",
    "  - __init__: Initializes with stored labels and features for both graphs.",
    "  - compute_class_representations():",
    "      * Purpose: Produces class-wise feature vectors (‘H’ for real, ‘H_prime’ for synthetic).",
    "      * Implementation:",
    "        - For each class c in 1..C:",
    "          - Find node indices in real_labels matching class c.",
    "          - Compute class centroid vector H[c] as the mean of real_features of class c, after applying adjacency matrix to incorporate neighborhood information.",
    "          - Similarly, for synthetic labels and features, compute H_prime[c], but note that H_prime is calculated using the spectrum-based eigenvectors and features accordingly.",
    "        - The adjacency is approximated or preprocessed to be used in the computation, likely as part of normalization. Use the adjacency matrices or node features directly if adjacency is not separately stored.",
    "  - discrimination_loss():",
    "      * Purpose: Computes the cosine similarity-based loss to encourage class feature alignment.",
    "      * Implementation:",
    "        - Calculate cosine similarity between each class’s real and synthetic class representation vectors, using torch.linalg.cosine_similarity or manual computation.",
    "        - For same class pairs, compute 1 minus the cosine similarity (aiming to maximize similarity).",
    "        - For different class pairs, encourage dissimilarity by adding the cosine similarity term (or its negative).",
    "        - Sum or average over all classes and pairs to form total discrimination loss.",
    "  - The losses are designed such that lower discrimination loss indicates higher similarity in class representations, which should help preserve category-level information during graph distillation.",
    "Implementation specifics:",
    "  - The class should contain methods to compute these class-level representations dynamically whenever called.",
    "  - It should manage the class-wise vectors as internal tensors or compute them on-the-fly.",
    "  - Hyperparameters for weighting the overall discrimination loss can be passed or set prior to usage but are not explicitly provided here.",
    " Data flow:",
    "  - At each training iteration (or whenever called), the real labels and features along with synthetic labels and features are passed/set in the class.",
    "  - compute_class_representations() is called to update class vectors.",
    "  - discrimination_loss() is called to calculate the regularization term, which gets incorporated into the total loss during the optimization of the synthetic graph.",
    "Notes:",
    "  - Ensure proper handling of class indices and matching. If number of classes is large, implement efficient masking (e.g., boolean masks or index-based selection).",
    "  - For numerical stability and consistent similarity calculations, normalize vectors before cosine similarity, or use torch's built-in functions that handle this.",
    "  - The method of which the class vectors are used during optimization should be compatible with the overall training loop, likely as a placeholder or an auxiliary loss addition.",
    "Summary:",
    "This module encapsulates the class-wise feature similarity component to enhance the representational fidelity of the synthetic graph concerning class distributions, complementing the spectral and eigenbasis matching. Its correct implementation is crucial to improve the generalization of the synthetic graph, especially in preserving category-specific information."
  ]
}

## eigenbasis_matcher.py

### Logic Analysis for eigenbasis_matcher.py

**Objective:**  
Implement the class `EigenbasisMatcher` which aligns the synthetic eigenvectors (`U'`) to the real eigenvectors (`U`) of the graph Laplacian, ensuring spectral similarity and orthogonality constraints during optimization.

---

### Core Responsibilities of `EigenbasisMatcher`

1. **Initialization:**
   - Accept input:
     - `target_basis` (`U`): the real graph’s eigenvectors (shape `[N, K]`).
     - `init_basis` (`U'`): the synthetic eigenvectors (shape `[N', K]`), which can be randomly initialized or from a prior method.
   - Store these for subsequent basis matching.

2. **Basis Matching via Gradient Descent:**
   - Perform optimization to update `U'` so that the subspace it spans aligns with that of `U`.
   - Use a loss function:
     \[
     \mathcal{L}_e = \sum_{k=1}^{K} \left\| u_k u_k^\top - u_k' u_k'^\top \right\|_F^2
     \]
     - This encourages the outer product of basis vectors to match, aligning the subspaces spanned.
   - Since basis vectors are orthogonal, the optimization should incorporate orthogonality constraints, ensuring `U'` remains valid eigenbasis.

3. **Orthogonality Enforcement (`L_o`):**
   - Regularization or projection step to keep `U'` orthogonal:
     \[
     \mathcal{L}_o = \left\| U'^T U' - I_K \right\|_F^2
     \]
   - Implementation:
     - After each gradient step, project `U'` back onto the Stiefel manifold (`U'^T U' = I`):
       - Use methods such as QR decomposition (`torch.linalg.qr`) or singular value decomposition (SVD).
     - Gradient-based correction in the direction of maintaining orthogonality.

4. **Optimization Algorithm:**
   - Use a standard optimizer like Adam for `U'`.
   - Alternate between:
     - Gradient updates of `U'` based on the basis matching loss.
     - Projection onto the orthogonal group to enforce the orthogonality constraint.
   - Hyperparameters:
     - `match_steps`: number of total gradient steps (e.g., 3000 as in config) for basis matching.
     - Learning rate: e.g., `eigenbasis_match_lr=1e-3`.

5. **Matching Strategy:**
   - Start with the initial `U'`.
   - For each iteration:
     - Compute `\mathcal{L}_e`.
     - Compute `\mathcal{L}_o` and combine (possibly with a weight, if needed).
     - Perform a gradient step to update `U'`.
     - Apply orthogonal projection to `U'` to satisfy constraints.
   - After convergence or reaching maximum steps, output the optimized basis `U'`.

6. **Output:**
   - The optimized synthetic basis `U'`.
   - These basis vectors will be used in subsequent steps to construct the synthetic graph consistent with the real graph’s spectral properties.

---

### Additional Implementation Details:

- **Input validation:**
  - Ensure `target_basis` and `init_basis` are of matching shapes.
  - Confirm `U` is orthogonal; if not, normalize or project onto the orthogonal basis.

- **Loss computation:**
  - For each basis vector `u_k` and `u_k'`, compute `u_k u_k^T` and `u_k' u_k'^T`.
  - Use `torch.norm` with Frobenius (`'fro'`) to calculate the squared differences.

- **Orthogonality enforcement:**
  - After each gradient step, project `U'` onto the Stiefel manifold:
    - Method: QR decomposition:
      ```python
      Q, R = torch.linalg.qr(U', mode='reduced')
      U' = Q
      ```
    - Ensures `U'^T U' = I`.

- **Gradient steps:**
  - Use optimizer step to update `U'`.
  - Incorporate `\mathcal{L}_o` as an explicit regularization term or via projection.

- **Stopping criteria:**
  - Convergence based on loss stability.
  - Fixed max steps as per config (e.g., 3000).

- **Optional:**
  - Early stopping if the basis overlap is sufficiently aligned.
  - Logging of basis similarity metrics (e.g., principal angles).

---

### Summary

- The class `EigenbasisMatcher` performs spectral basis alignment between real (`U`) and synthetic (`U'`) eigenvectors.
- Optimization minimizes a basis similarity loss, with orthogonality constraints enforced by projection (QR or SVD).
- Use Adam optimizer with learning rate specified in config (`1e-3`).
- Conduct matching for `match_steps` iterations.
- After optimization, return the constructed `U'` which is aligned with `U`.
- These basis vectors serve as a key component for constructing the synthetic graph within the broader GDEM framework.

This detailed analysis provides a clear, step-by-step plan for implementing the class while respecting the methodological and theoretical motivations.

## evaluate.py

**Logic Analysis for `evaluate.py`**

---

### **Purpose & Responsibility**
The primary purpose of `evaluate.py` is to define the `Evaluator` class, which:
- Loads a trained GNN model,
- Prepares the test dataset,
- Performs inference to compute predictions,
- Calculates classification metrics—primarily accuracy,
- Can output additional evaluation metrics if needed.

These evaluations are essential to measure the downstream task performance of the synthetic graphs generated via GDEM, specifically on the test data of the original datasets.

---

### **Input & Dependencies**
- **Inputs:**
  - Trained model checkpoint (file path).
  - Test dataset (PyTorch Geometric `Data` object or similar structure).
  - Ground-truth labels for test nodes.

- **Dependencies:**
  - `torch` for tensor operations.
  - `torch_geometric` for data handling.
  - Dataset-specific utility functions (e.g., data normalization, batching, if any).
  - Standard Python libraries (`os`, `logging`, etc.).
  - (Optional) sklearn metrics like accuracy, or custom accuracy implementations.

---

### **Key Components & Steps**
1. **Model Loading:**
   - Accept model type (e.g., 'GCN', 'ChebyNet') for flexible evaluation of different architectures.
   - Load the saved model weights (`state_dict`) from disk.
   - Instantiate the model object with the architecture hyperparameters, matching those used during training.
   - Set the model in `eval()` mode to disable dropout and batch norm updates.

2. **Test Data Preparation:**
   - Load or receive the test dataset object.
   - Ensure the data includes:
     - Node features: `x`.
     - Edge index: `edge_index`.
     - Labels: `y`.
   - Move data tensors and model to the same device (CPU or CUDA).

3. **Inference:**
   - Use `with torch.no_grad()` context to disable gradient computation.
   - Pass test node features and edge info into the model to get predictions (`logits`).
   - For node classification, predictions are typically class logits, apply softmax if needed.

4. **Evaluation Metric Computation:**
   - Determine predicted class labels via `argmax`.
   - Compare predictions with ground-truth labels.
   - Compute accuracy:
     \[
     \text{accuracy} = \frac{\text{number of correct predictions}}{\text{total number of nodes}}
     \]
   - Optionally, compute other metrics, e.g., F1-score, precision-recall, if necessary - but per the provided plan, accuracy is primary.

5. **Results & Output:**
   - Return a dictionary containing evaluation metrics: Accuracy, and potentially additional info.
   - Log the results and save if needed for reporting.

---

### **Design Considerations & Ensuring Fidelity**
- **Model Compatibility:**
  - Support different architectures (“GNN types”) by accepting a model_type argument.
  - Maintain consistency of hyperparameters (hidden units, layers, etc.) during model instantiation.
  
- **Reproducibility:**
  - Use the same `random_seed` across all evaluation runs, if any randomness is involved.
  - Evaluate on the same test set as used during the training/validation for the given experiment.

- **Efficiency:**
  - Perform inference on the entire test set at once (batching can be used if necessary).
  - Use `torch.no_grad()` and move data/model to GPU if available.

- **Error Handling:**
  - Check for mismatched device types.
  - Validate that test dataset contains all necessary fields (`x`, `edge_index`, `y`).
  - Handle exceptions during model loading and inference gracefully.

---

### **Implementation Outline: Pseudocode Structure**
```python
class Evaluator:
    def __init__(self, model_type, model_path, test_data, device='cpu'):
        # Load model architecture with hyperparameters matching training
        # Load trained state_dict and set model to eval mode
        # Move model to device
        # Store test_data and labels
        
    def evaluate(self):
        # Move data to device
        # Inference with torch.no_grad()
        # Compute logits
        # Apply argmax to get predicted labels
        # Compare with ground truth labels
        # Calculate accuracy
        # Return metrics dictionary
```

---

### **Additional Details & Final Notes**
- **Dataset Handling:**
  - The test dataset is expected to be a `torch_geometric.data.Data` object.
  - Ensure the data splits are consistent with those used in training.
- **Model Compatibility:**
  - Assumes models are implemented according to the architecture classes described in `model.py`.
  - For each architecture, the `__init__` method should accept hyperparameters defining number of layers, hidden units, and possibly polynomial degree for spectral models.
- **Metrics:**
  - Use `torch.eq` or numpy for comparison, then compute mean accuracy as a float.
  - Might optionally print validation accuracy for quick debugging.

---

**This thorough logical plan provides a blueprint for implementing `evaluate.py`'s `Evaluator` class, ensuring accurate, efficient, and reproducible evaluation aligned with the experimental framework described in the paper.**

## model.py

# **Logic Analysis for `model.py`**

This component is responsible for defining the core GNN models used for downstream training and evaluation. It will contain a class `GNNModel` that can instantiate various GNN architectures (e.g., GCN, SGC, PPNP, ChebyNet, BernNet, GPR-GNN) according to specified parameters and provide interfaces for forward computation, loss calculation, optimizer setup, and model management. The implementation leverages `torch_geometric.nn` modules to ensure compatibility and efficiency on graph data.

---

## **1. Class Responsibilities and Design**

- **Main Class:** `GNNModel`
- **Attributes:**
  - `architecture_type`: str indicating model type, e.g., 'GCN', 'SGC', etc.
  - `params`: dict containing model-specific hyperparameters (e.g., hidden units, layers, polynomial order).
  - `model`: a `torch.nn.Module` instance encapsulating the actual GNN layers.
  - `device`: torch device (cpu or cuda).

- **Methods:**
  - `__init__`: Initialize parameters, build the model layers based on `architecture_type` and `params`.
  - `forward(data)`: Execute forward pass; input is a `torch_geometric.data.Data` object.
  - `compute_loss(output, labels)`: Calculate classification loss (e.g., cross-entropy).
  - `get_optimizer(learning_rate, weight_decay)`: Returns an optimizer (Adam) for training.
  - `save_model(filepath)`, `load_model(filepath)`: Save/load model state dictionary.
  - `to_device(device)`: Transfer model parameters to GPU or CPU.

---

## **2. Support for Multiple Architectures**

The class must support these architectures:

### a. Spatial GNNs
- **GCN**:
  - Use `torch_geometric.nn.GCNConv`.
  - Layers: Number specified by `params['layers']` (usually 2).
  - Hidden units: `params['hidden_units']`.

- **SGC**:
  - Replaces GCN layers with simplified `torch_geometric.nn.SGCConv`.
  - Usually involves polynomial filtering, can be stacked as linear layers.

- **PPNP**:
  - Based on Personalized PageRank filter approximations (e.g., `torch_geometric.nn.GCNConv` with an explicit alpha parameter or a dedicated PPNP implementation possibly mimicking its filtering behavior).
  - Feed-forward: standard GCN layers plus a propagation step, or a custom module if available.

### b. Spectral GNNs
- **ChebyNet**:
  - Use `torch_geometric.nn.ChebConv`.
  - `poly_order`: from `params['poly_order']`.
  - Supports spectral filtering using Chebyshev polynomials.

- **BernNet**:
  - Implemented via `torch_geometric.nn.BernNetConv` or custom if not available.
  - Polynomial order: `params['poly_order']`.

- **GPR-GNN**:
  - Use `torch_geometric.nn.GPRConv`.
  - Polynomial order: `params['poly_order']`.

### c. Architectural Details:
- Number of layers: 2 (per dataset setup).
- Hidden units: 256.
- Dropout and activation: ReLU (common practice).
- Regularization: dropouts may be applied after each layer as in standard practice.

---

## **3. Implementation Details**

### a. Initialization
- During `__init__`, based on `architecture_type`, instantiate the necessary module sequence.
- Consider creating a `torch.nn.ModuleList` of layers.
- For spectral poly-approximate models, include polynomial order as parameter.
- For models supporting multiple layers, stack accordingly, possibly with residual connections if needed.

### b. Forward Pass
- Input `data`: expected to have:
  - `data.x`: node features
  - `data.edge_index`: adjacency information
  - Possibly `data.edge_weight` if used

- Implement `forward(data)`:
  - Pass features through each GNN layer.
  - Apply activation functions (ReLU).
  - Apply dropout if necessary.
  - For output, produce class logits (size: number of nodes x number of classes).

### c. Loss and Optimization
- Loss: Typically cross-entropy (`torch.nn.functional.cross_entropy`) between model output and `data.y` (node labels).
- During training, the optimizer can be set externally and attached via `get_optimizer()`.
- Model saving/loading: store state_dict.

### d. Flexibility & Compatibility
- Ensure all models are compatible with batched graphs if needed (for node classification, usually one big graph).
- Support transfer to device (cpu or cuda).

### e. Hyperparameters
- Hidden units: 256.
- Layers: 2.
- Activation: ReLU.
- Dropout: optional; set to 0.5 as per config.

---

## **4. Model Architecture Details and Pseudocode**

```python
class GNNModel(torch.nn.Module):
    def __init__(self, architecture_type: str, params: dict):
        super().__init__()
        self.architecture_type = architecture_type
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build layers based on architecture_type
        if architecture_type == 'GCN':
            self.model = self.build_gcn()
        elif architecture_type == 'SGC':
            self.model = self.build_sgc()
        elif architecture_type == 'PPNP':
            self.model = self.build_ppnp()
        elif architecture_type == 'ChebyNet':
            self.model = self.build_chebynet()
        elif architecture_type == 'BernNet':
            self.model = self.build_bernnet()
        elif architecture_type == 'GPR-GNN':
            self.model = self.build_gpr()
        else:
            raise ValueError("Unsupported architecture type.")

    def build_gcn(self):
        layers = torch.nn.ModuleList()
        # First layer
        layers.append(GCNConv(in_channels, self.params['hidden_units']))
        # Optional: add more layers if params specify
        # Final classification layer
        # ...
        return torch.nn.Sequential(*layers, ...)

    def build_sgc(self):
        # Similar to GCN but with SGCConv
        ...

    def build_ppnp(self):
        # Based on the GCN structure with PPNP specifics
        ...

    def build_chebynet(self):
        conv = ChebConv(in_channels, self.params['hidden_units'], K=self.params['poly_order'])
        # Compose with activation, dropout, final layer
        ...

    def build_bernnet(self):
        conv = BernNetConv(in_channels, self.params['hidden_units'], K=self.params['poly_order'])
        ...

    def build_gpr(self):
        conv = GPRConv(in_channels, self.params['hidden_units'], K=self.params['poly_order'])
        ...

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        # Forward through layers
        # For multiple layers, iterate
        for layer in self.model:
            x = layer(x, edge_index)
            x = F.relu(x)
            # optional dropout
        return x  # logits

    def compute_loss(self, output, labels):
        return F.cross_entropy(output, labels)

    def get_optimizer(self, learning_rate, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
```

---

## **5. Additional Considerations**

- **Activation & Dropout:** 
  - Incorporate activation functions (`ReLU`) after each convolution.
  - Include dropout layers with probability 0.5 if needed.

- **Batching:**
  - Typically, for node classification on a large graph, can process entire graph or use mini-batches (implemented via `torch_geometric.data.DataLoader`) with care.

- **Device Management:**
  - Move model to device during init or via `to_device()` method.

- **Configurable Hyperparameters:**
  - Allow setting hidden units, number of layers, dropout probability via `params` and external configurations.

---

## **6. Summary of Implementation Plan**

- Define `GNNModel` class.
- Support multiple architectures with a modular build approach.
- Use `torch_geometric.nn` modules.
- Implement forward, loss, optimizer retrieval, save/load.
- Maintain flexibility for hyperparameters via `params`.
- Ensure compatibility with graph data format used in downstream scripts.

This logic thoroughly prepares the implementation details needed for `model.py` to be correctly aligned with the methodology and experimental setup described in the paper and overall plan.

## requirements.txt

# requirements.txt

# This file documents the detailed reasoning and dependencies necessary for implementing the "Graph Distillation with Eigenbasis Matching" (GDEM) framework, aligned with the provided plan, design, and configuration.

# 1. Programming Environment
# - Python 3.8+ recommended for compatibility with PyTorch, PyTorch Geometric, and scientific libraries.

# 2. Core Deep Learning Framework
torch==1.13.1
# - Provides tensor computations, neural network modules, and GPU acceleration.

# 3. Graph Data Handling
torch-geometric==2.2.0
# - Facilitates easy loading, batching, and processing of graph data.
# - Provides GNN layers and graph utility functions.
# - Supports sparse matrices and adjacency operations essential for eigen-decomposition.

# 4. Scientific Computing & Eigen-Decomposition
scipy==1.10.0
# - Used for:
#   - Eigen-decomposition of the normalized Laplacian (via scipy.sparse.linalg.eigsh).
#   - Handling large sparse matrices.
#   - Performing spectrum-based operations for spectrum reproduction and analysis.

# 5. Numerical Operations
numpy==1.21.0
# - Numerical array operations.
# - Used for data preprocessing, eigenvector/eigenvalue processing, and general calculations.

# 6. Model Evaluation and Visualization
sklearn==0.24.2
# - Provides metrics like accuracy score.
# - Might be used for train/test split utilities.

matplotlib==3.5.3
# - Visualization of spectral properties, convergence curves, and distribution plots.
# - Plotting Total Variation (TV) and accuracy metrics as described in the paper.

# 7. Optional Visualization Utilities
# - Additional optional libraries like seaborn can be included for advanced plotting, but are not mandatory.

# 8. Experimental Setup Dependencies
# - Ensure that all dependencies support CUDA (if GPU is available) for efficient eigen-decomposition and GNN training.

# 9. Additional Notes
# - The code should be compatible with the specified versions to ensure reproducibility.
# - No third-party dependencies outside of the above are required according to the plan.
# - For large graphs where eigen-decomposition is computationally intensive, consider using eigen-solver options that support approximate methods (e.g., Lanczos).

# 10. Validation and Testing
# - Use pytest or unittest (not listed here) to verify code correctness, especially for spectral calculations and basis orthogonality constraints.

# 11. Hyperparameter Control
# - All hyperparameters (e.g., number of eigenvectors, regularization weights, learning rates) will be set via configuration and do not necessitate additional packages.

# Summary:  
# The above dependencies cover all necessary scientific and library needs for implementing the GDEM framework, including data loading, spectral analysis, optimization, and visualization, ensuring alignment with the provided design and experimental details.

# End of requirements.txt


## spectral_decomposition.py

### Logic Analysis for spectral_decomposition.py

**Objective:**  
Implement a class `SpectralDecomposition` that, given adjacency matrices, computes the normalized graph Laplacian and performs eigen-decomposition to output eigenvalues and eigenvectors. It must support large graphs via efficient eigenvalue solvers, and output the spectrum components required for spectrum replication and eigenbasis matching as described in the GDEM methodology.

---

### Core Responsibilities:

1. **Input Handling:**  
   - Accept as input an adjacency matrix `A` in sparse or dense format, represented using `torch.Tensor` or compatible format for conversion.
   - Ensure that the adjacency matrix reflects the undirected, unweighted graph structure as per the paper’s assumption.

2. **Normalization of Adjacency Matrix:**  
   - Convert adjacency matrix to sparse format if needed.
   - Compute the degree matrix `D` as a diagonal matrix: `D_{ii} = sum_j A_{ij}`.
   - Compute the symmetric normalized Laplacian:  
     \[
     \hat{\mathbf{L}} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}
     \]
   - Handle numerical stability when degrees are zero (isolated nodes). Typically, isolated nodes have zero degrees; set their degrees to 1 during reciprocal square root computation or handle accordingly.

3. **Eigen-Decomposition:**
   - Use `scipy.sparse.linalg.eigsh()` for computational efficiency, especially important for large sparse matrices.  
   - Support parameters:
     - Number of eigenvalues/eigenvectors to compute `K` (from the config, e.g., 500).  
     - Support full spectrum when `K = N` (full eigen-decomposition).
     - For large graphs, often only the smallest/largest eigenvalues are needed (spectral slicing).  
   - Ensure orthogonal eigenvectors are returned, order the eigenvalues from smallest to largest if necessary, consistent with spectral theory.

4. **Spectral Output:**
   - Return:
     - `eigenvalues`: a 1D torch.Tensor of shape `(K,)`, ordered ascending.
     - `eigenvectors`: a 2D torch.Tensor of shape `(N, K)` with orthogonal eigenvectors.
   - Document that the eigenvectors are normalized and orthogonal, consistent with spectral decomposition properties.
   
5. **Handling Large Graphs:**
   - For very large graphs, rely on `eigsh()`'s parameters:
     - `which='SM'` for smallest magnitude eigenvalues.
     - `which='LA'` for largest eigenvalues, if needed.
     - Use `k` with spectral slices.
   - For approximate methods, such as truncated SVD or Lanczos, ensure the accuracy is sufficient.

6. **Conversion & Data Type:**
   - Ensure input adjacency matrices are converted to CPU sparse matrices for scipy eigen-decomposition.
   - Convert input `torch.Tensor` to `scipy.sparse.csr_matrix` as needed.
   - Eigenvalues and eigenvectors should be converted back to torch.Tensor for further use.
   - Use `dtype=torch.float32` for consistency, although eigenvalues may be `float64` from `scipy`. Cast to float32 if preferred.

7. **Output & Compatibility:**
   - Keep output types consistent with the overall codebase (preferably torch tensors).
   - Support optional caching if large graphs are processed repeatedly (not mandatory but optimal).

---

### Step-by-Step Processing:

**Initialization:**
- Accept adjacency matrix (`A`) in `torch.Tensor` format.
- Convert adjacency to sparse format suitable for scipy: use `scipy.sparse.csr_matrix(A.cpu().numpy())`.

**Normalize adjacency:**
- Compute `D`: 
  - degrees = `A.sum(dim=1)` or suitable sparse sum.
  - Handle zeros: if degree = 0, set to 1 to avoid division by zero.
  - Compute `D^{-1/2}` as a diagonal matrix:
    \[
    D^{-1/2} = \operatorname{diag}(\frac{1}{\sqrt{\text{degree}_i}})
    \]
- Compute normalized Laplacian:
  \[
  \hat{\mathbf{L}} = \mathbf{I}_N - D^{-1/2} A D^{-1/2}
  \]
  - Do this with sparse multiplication for efficiency.

**Eigen-decomposition:**
- Use `scipy.sparse.linalg.eigsh` on the matrix `L_hat`.
- Pass parameters for `k=K`, `which='SM'` (smallest eigenvalues) or `'LA'` (largest), as needed.
- Retrieve eigenvalues and eigenvectors.
- Convert results to torch tensors:
  - Eigenvalues: shape `(K,)`.
  - Eigenvectors: shape `(N, K)`.

**Return:**
- Eigenvalues tensor: ascending order.
- Eigenvectors tensor: normalized eigenvectors of shape `(N, K)`.

---

### Additional Notes:

- **Eigen-decomposition Parameters:**  
  - For spectral matching and spectrum replication, use smallest eigenvalues (`which='SM'`).
  - For eigenbasis matching (matching eigenvectors), the order should be consistent.
  - For full spectrum, set `k=N`; otherwise, choose `k=K`.

- **Numerical Stability & Edge Cases:**  
  - Handle isolated nodes (zero degrees).
  - Eigen-decomposition may not converge for very large graphs with too many eigenvalues; batch approaches or approximate spectral methods may be used.

- **Dependencies & Compatibility:**
  - Fully rely on `scipy.sparse.linalg.eigsh`.
  - Use torch to convert and pass data types.
  - Keep in mind that eigenvectors are orthogonal: verify orthogonality as a check if needed.

---

### Summary:

- Input: adjacency matrix `A` (`torch.Tensor`).
- Conversion: `A` to sparse scipy matrix for eigen-decomposition.
- Compute normalized Laplacian `L_hat`.
- Use `eigsh()` with appropriate `k` and `which` parameters.
- Output eigenvalues (`torch.Tensor`) and eigenvectors (`torch.Tensor`), both converted back to torch tensors.
- Designed for flexible K (number of eigenvalues/eigenvectors), supports large/small graphs, and maintains spectral properties needed for GDEM.

This comprehensive plan ensures the class `SpectralDecomposition` performs accurate spectral analysis aligned with the method's requirements, enabling the spectral matching and spectrum replication in the main GDEM pipeline.

## synthetic_graph_generator.py

**Logic Analysis for __synthetic_graph_generator.py__**

**Purpose and Role:**
This module encapsulates the core logic for generating a synthetic graph that approximates the spectral properties of a real graph via eigenbasis matching and spectrum replication. It performs alternating optimization steps on the synthetic node features and eigenbasis, constructs the synthetic adjacency and Laplacian matrices, and ensures the preservation of key structural and spectral characteristics.

---

### Class Initialization and Attributes

- **Input dependencies:**
  - `eigenvalues`: tensor of size `[K]` containing the real graph's eigenvalues.
  - `eigenvectors`: tensor of size `[N, K]` representing the eigenbasis (principal eigenvectors) of the real graph.
  - `features`: tensor of shape `[N', d]` initial node features for the synthetic graph (can be random or projected).

- **Blended attributes:**
  - `N'`: number of synthetic nodes.
  - `K`: number of eigenvectors/eigenvalues used.
  - `U_prime`: `[N', K]` tensor, the synthetic eigenbasis to be optimized.
  - `X_prime`: `[N', d]` tensor, synthetic node features to be optimized.
  - `lambda_k`: real eigenvalues, used for spectrum construction.
  
**Initialize:**
- `U_prime`: random orthonormal basis `[N', K]`. This will be optimized to match the real eigenbasis.
- `X_prime`: initialized for features, potentially via a related pretrained MLP or random initialization.

---

### Core Methods and Logical Workflow

1. **Construct Spectrum-based Adjacency and Laplacian:**
   - Using the real eigenvalues (`lambda_k`) and the optimized `U_prime`, construct:
     - **Synthetic Laplacian:**
       \[
       \mathbf{L'} = \sum_{k=1}^K \lambda_k \mathbf{u}_k' \mathbf{u}_k'^\top
       \]
     - **Synthetic adjacency:**
       \[
       \mathbf{A'} = \sum_{k=1}^K (1 - \lambda_k) \mathbf{u}_k' \mathbf{u}_k'^\top
       \]
   - These are central to preserving the spectral distribution; constructed after eigenbasis replications.

2. **Alternating Optimization Procedure:**
   - **Alternating steps (based on pseudocode and schedule):**
     - **Eigenbasis update steps:**
       - Use `EigenbasisMatcher` (EM) to match `U_prime` to `U` via gradient descent:
         \[
         \mathcal{L}_e = \sum_{k=1}^K \left\| \mathbf{u}_k \mathbf{u}_k^\top - \mathbf{u}_k' \mathbf{u}_k'^\top \right\|_F^2
         \]
       - Enforce orthogonality via regularization:
         \[
         \mathcal{L}_o = \left\| \mathbf{U}_K'^\top \mathbf{U}_K' - \mathbf{I}_K \right\|_F^2
         \]
       - Optimization involves gradient descent on `U_prime`:
         - Use projected gradients or re-orthogonalization (e.g., via SVD or QR) to maintain orthogonality.
       - Perform a fixed number of steps (`eigenbasis_match_steps`).

     - **Node features update steps:**
       - Given the current eigenbasis `U_prime`:
         \[
         \mathcal{L}_e, \mathcal{L}_o \text{ are fixed}
         \]
       - Minimize the spectral discrepancy with respect to `X_prime`:
         \[
         \mathcal{L}_X = \text{computed via spectral loss functions}
         \]
       - Update `X_prime` using gradient descent with learning rate `feature_update_lr`.
       - Perform `feature_opt_steps`.

3. **Implementation details during alternating schedule:**
   - Use schedule parameters:
     - `tau_1`: number of steps for eigenbasis update.
     - `tau_2`: number of steps for feature update.
   - Alternate between these updates until total epochs (`epochs`) are covered.

4. **Orthogonality Handling:**
   - After each eigenbasis update, project `U_prime` onto the Stiefel manifold:
     - Via SVD: `U_prime = U * V^T` where `U`, `V` are from SVD of current `U_prime`.
   - Ensures `U_prime^\top U_prime ≈ I`.
   - This maintains the orthogonality regularization and basis properties.

5. **Synthetic Graph Construction Finalization:**
   - After optimization:
     - Build the adjacency matrix and Laplacian using the last optimized `U_prime` and real `lambda_k`.
     - Alternatively, optionally add a step to refine `A'` via graph sparsification or normalization, aligning with downstream GNN requirements.
   - Save/return computed adjacency, Laplacian, eigenbasis, and node features.

6. **Distribute Regularization & Loss Weights:**
   - Use hyperparameters (e.g., `lambda_e`, `lambda_d`, `lambda_o`) to weight spectral matching, class-based regularization, and orthogonality constraints.
   - These weights are accumulated into a total loss function during each update (see the paper’s Equation 9).

7. **Process flow when called externally:**
   - Initialization with possible spectral and feature regularizations.
   - Perform scheduled alternating optimization.
   - Construct spectrum-preserving adjacency.
   - Output synthetic adjacency, features, and eigenbasis.

---

### Additional Implementation Notes

- **Eigen-decomposition:**
  - Use `scipy.sparse.linalg.eigsh` for efficient eigenvalue/eigenvector computation for large sparse graphs.
  - For spectral basis: only compute the bottom/top `K` eigenvectors/eigenvalues.
- **Gradient updates:**
  - Use autograd (PyTorch) to compute gradients for `U_prime` and `X_prime`.
  - Freeze eigenvalues during basis update; update eigenvectors.
  - Freeze eigenbasis during feature optimization; update features.
- **Projection step:**
  - After each eigenbasis gradient step, project `U_prime` onto the orthogonal group:
    - `U_new = U * V^T` with `U, V` from SVD of `U_prime`.
- **Hyperparameters affecting optimization:**
  - `eigenbasis_match_lr`, `feature_update_lr`.
  - `tau_1`, `tau_2` (schedule steps).
  - Regularization weights: `lambda_e`, `lambda_d`, `lambda_o`.
- **Convergence and stopping criterion:**
  - Number of total epochs (`epochs`).
  - Optional early stopping based on spectral loss or other criteria.
- **Post-optimization:**
  - Use the final eigenbasis and spectrum to re-construct the full synthetic adjacency matrix.
  - Save all relevant matrices and features for downstream GNN training/evaluation.

---

### Summary

The `SyntheticGraphGenerator` class:
- **Inputs:** spectrum (`lambda_k`), eigenbasis (`U`), initial features.
- **Operations:**
  - Construct spectrum-based adjacency/laplacian.
  - Alternately optimize eigenbasis (match and orthogonalize) and node features to minimize spectral mismatch.
  - Incorporate regularizations (`L_e`, `L_o`, `L_d`) during updates.
- **Outputs:** final adjacency, Laplacian, eigenbasis, features of synthetic graph.
- **Supporting steps:** schedule management, gradient steps, basis projection, spectral computations, and regularizations.

This comprehensive logic ensures the synthetic graph effectively approximates the spectral distribution of the real graph, maintaining key structural and global properties as dictated by the paper’s methodology.

## train.py

# Logic Analysis for train.py

## Purpose
Implement the `Trainer` class responsible for managing the training, validation, and saving/loading of GNN models on synthetic or real data. This class should handle model initialization, training loop management, early stopping or best model selection, and evaluation.

---

## Core Responsibilities
1. Initialize with a GNN model, training data, validation data, and labels.
2. Execute training over specified epochs, with progress logging.
3. Save the best performing model based on validation metrics.
4. Load saved models for evaluation.
5. Support hyperparameter specification (learning rate, epochs, batch size, early stopping, etc.).
6. Compatibility with different architectures (spatial and spectral GNNs).
7. Reproducibility via setting random seeds.

---

## Inputs
- **model**: An instance of `GNNModel` (configured for specific architecture: GCN, ChebyNet, etc.).
- **train_data**: Data object or tensor(s) containing node features, adjacency, labels, and train/validation split masks.
- **val_data**: Data object or tensor(s) for validation; used to monitor performance.
- **labels**: Ground truth labels for training and validation, to compute accuracy or other metrics.
- **hyperparameters**:
  - `epochs`: total number of training epochs.
  - `learning_rate`: optimizer step size.
  - `batch_size`: for batch training (important for large datasets).
  - `weight_decay`: regularization term.
  - `dropout`: dropout rate used in model (if applicable).
  - `validation_interval`: frequency of validation (every n epochs).
  - `early_stopping_rounds`: optional, to stop if validation performance doesn't improve.

---

## Key Components and Methods
### 1. Initialization (`__init__`)
- Store provided model, data, labels, hyperparameters.
- Initialize optimizer (Adam) with model parameters and provided learning rate.
- Initialize loss criterion (likely `torch.nn.CrossEntropyLoss`).
- Initialize variables for best validation performance and checkpoint paths.

### 2. Training Loop (`train`)
- Loop over `epochs`.
- Set model to train mode (`model.train()`).
- For each batch or full data:
  - Forward pass: compute output predictions.
  - Compute loss with ground truth labels.
  - Backpropagate loss (`loss.backward()`).
  - Update optimizer (`optimizer.step()`).
  - Zero gradients (`optimizer.zero_grad()`).
- If batch processing:
  - Use DataLoader for batching.
- Logging:
  - Periodically print training loss and accuracy.
  - Record training performance metrics for progress visualization.

### 3. Validation (`validate`)
- Set model to evaluation mode (`model.eval()`).
- Disable gradient calculations (`torch.no_grad()`).
- Forward pass on validation data.
- Compute metrics: accuracy; possibly loss if needed.
- Check if current validation accuracy is better than the previous best.
  - Save the model if performance improves (`save_checkpoint()`).
  - Update `best_score`.

### 4. Save and Load methods
- `save_checkpoint(filepath)`: Save current model state dict.
- `load_model(filepath)`: Load saved model state dict for inference/evaluation.

### 5. Evaluation (`evaluate`)
- Load best model.
- Test on the real test set.
- Compute test accuracy and other metrics.
- Return evaluation metrics as dictionary or tuple.

---

## Hyperparameter considerations and configuration
- Set optimizer learning rate (`learning_rate`) from config.
- Use standard `Adam` optimizer.
- Define the number of epochs (`epochs`) based on config (e.g., 2000 for training).
- Log validation performance every `validation_interval` epochs.
- Implement early stopping if applicable, based on validation performance plateau.

---

## Reproducibility
- Ensure reproducibility by setting random seeds:
  - `torch.manual_seed()`
  - `np.random.seed()`
  - Set seed at the start of the module/class for consistent training runs.

## Compatibility
- Accept models of different architectures with same interface (forward pass, parameters).
- Support batch or full-graph training depending on dataset size.
- Use `torch-geometric` data structures for handling large graphs efficiently if necessary.

---

## Additional Considerations
- During training:
  - Apply dropout if necessary.
  - Possibly implement gradient clipping for stability in large models.
- During validation:
  - Accumulate predictions for bulk metric calculation.
- Log metrics:
  - Use simple print statements or a logging framework for training progress.
  - Save logs for later analysis (e.g., validation accuracy per epoch).
- After training:
  - Ensure proper cleanup, save final models, and provide hooks for evaluation scripts.

---

## Summary of ImplementationPlan
- Implement class `Trainer`.
- **__init__()**: store parameters, setup optimizer, and loss vars.
- **train()**: main training loop with periodic validation, checkpointing.
- **validate()**: evaluate on validation set and update best model.
- **save_model()** / **load_model()**: save/load model states.
- **evaluate()**: gather final test results.
- Ensure seed setting for reproducibility.

---

This detailed reasoning encapsulates the core logic needed in `train.py` to manage GNN training, validation, checkpointing, and evaluation in strict accordance with the experimental plan and configuration specifications.

## utils.py

{
  "utils.py": [
    {
      "function": "normalize_features",
      "description": "Normalize node feature matrices, typically row-wise normalization to prevent scale issues during spectrum and feature matching.",
      "inputs": [
        "features: torch.Tensor"
      ],
      "outputs": [
        "normalized_features: torch.Tensor"
      ],
      "implementation_notes": [
        "Use torch's normalization functions, e.g., torch.nn.functional.normalize with p=2 along dimension=1.",
        "Ensures each node feature vector has unit norm, facilitating spectral alignment and spectral similarity metric calculations."
      ],
      "dependencies": [
        "torch"
      ],
      "additional": "Implement as torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-8) to avoid division by zero."
    },
    {
      "function": "project_onto_stiefel",
      "description": "Projects a matrix (eigenbasis candidate) onto the Stiefel manifold to enforce orthogonality (U^T U = I).",
      "inputs": [
        "matrix: torch.Tensor"
      ],
      "outputs": [
        "orthogonal_basis: torch.Tensor"
      ],
      "implementation_notes": [
        "Use the QR decomposition: torch.linalg.qr(matrix, mode='reduced') and return the Q component.",
        "Guarantees that the eigenbasis approximation remains orthogonal after each gradient update during optimization.",
        "Ensure numerical stability by handling potential small negative values or deviations from perfect orthogonality."
      ],
      "dependencies": [
        "torch"
      ],
      "additional": "Function signature: def project_onto_stiefel(matrix: torch.Tensor) -> torch.Tensor"
    },
    {
      "function": "compute_spectrum",
      "description": "Calculate the eigenvalues and eigenvectors of a graph's normalized Laplacian or adjacency matrix.",
      "inputs": [
        "adj_matrix: torch.Tensor",
        "num_eigenvectors: int (K)"
      ],
      "outputs": [
        "eigenvalues: torch.Tensor",
        "eigenvectors: torch.Tensor"
      ],
      "implementation_notes": [
        "Use scipy.sparse.linalg.eigsh for large sparse matrices, which is efficient and suitable for large graphs.",
        "If the adjacency matrix is dense or small, consider torch.linalg.eig or torch.linalg.eigh.",
        "Input must be symmetric positive semi-definite for eigen-decomposition of Laplacian.",
        "Eigenvalues should be sorted ascending; select the bottom K or top K depending on the spectral band of interest."
      ],
      "dependencies": [
        "scipy.sparse.linalg",
        "torch",
        "numpy"
      ],
      "additional": "Define a wrapper function: def compute_spectrum(adj_matrix: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]"
    },
    {
      "function": "compute_graph_tv",
      "description": "Compute the Total Variation (TV) of a node feature signal on the graph, reflecting the smoothness or frequency content.",
      "inputs": [
        "features: torch.Tensor",
        "laplacian: torch.Tensor"
      ],
      "outputs": [
        "tv_value: float"
      ],
      "implementation_notes": [
        "TV is computed as trace(x^T L x) which equals sum_{(i,j) in E} (x_i - x_j)^2.",
        "Normalize features and Laplacian before calculation to ensure scale-invariance.",
        "Implement as: torch.trace(features.T @ laplacian @ features).",
        "To get per-dimension TV, compute this for each feature dimension separately, then average."
      ],
      "dependencies": [
        "torch"
      ],
      "additional": "This metric helps compare the spectral similarity between real and synthetic graphs."
    },
    {
      "function": "plot_spectrum_comparison",
      "description": "Visualize the spectral distributions (eigenvalues) of real and synthetic graphs for qualitative spectral similarity assessment.",
      "inputs": [
        "real_eigenvalues: torch.Tensor",
        "synthetic_eigenvalues: torch.Tensor",
        "metrics: dict (e.g., TV values, spectral distances)"
      ],
      "outputs": [
        "plots or save figure"
      ],
      "implementation_notes": [
        "Use matplotlib.pyplot to plot histograms or eigenvalue spectra.",
        "Optionally plot cumulative distribution functions (CDFs) for spectral distributions.",
        "Annotate plots with spectrum similarity metrics, e.g., TV, to connect quantitative and visual assessments.",
        "Ensure consistent axis limits for fair comparison."
      ],
      "dependencies": [
        "matplotlib.pyplot",
        "torch"
      ],
      "additional": "This visualization supports understanding how well the spectrum is preserved during distillation."
    }
  ],
  "Additional notes": [
    "All functions should be robust to input size and data type, with proper device handling (CPU/GPU).",
    "For functions with stochastic elements (e.g., eigen-decomposition), fix random seeds or document stochasticity for reproducibility.",
    "Implement appropriate input validation and error handling for inputs to each function."
  ]
}

