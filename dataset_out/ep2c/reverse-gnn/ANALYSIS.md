# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

## Logic Analysis for dataset_loader.py

### 1. Purpose and Responsibilities
- Implement `DatasetLoader` class to load a variety of graph datasets (heterophilic and homophilic) from files or URLs.
- Process raw data into standardized PyTorch tensors:
  - Adjacency matrix in sparse format (preferably `torch.sparse.FloatTensor`).
  - Node features (`torch.FloatTensor`).
  - Labels (`torch.LongTensor` for classification tasks).
  - Masks for train, validation, and test splits (`torch.BoolTensor`).
- For heterophilic datasets, ensure filtering is applied as specified in the paper (e.g., to prevent train-test leakage in datasets like Squirrel and Chameleon).
- Cleanly separate support for datasets via dataset name identifiers, likely mapped to files or URLs.
- Compute or assign dataset statistics and splits, maintaining reproducibility via seed.
- Expose a method `load_data()` that returns all components needed for training/validation/testing.

---

### 2. Dataset Input Sources
- Datasets may be stored locally or downloaded from URLs.
- Key datasets include:
  - Heterophilic: Minesweeper, Roman-empire, Questions, Tolokers, AmazonRatings, etc.
  - Homophilic: Cora, CiteSeer, PubMed, etc.
- For datasets like Minesweeper, the standard input format is likely a text or adjacency list file (e.g., edge list; features in separate files).
- The datasets could be in common formats (e.g., `.npz`, `.mat`, or custom JSON/CSV files).
- Use `scipy.io.loadmat`, `numpy.load`, or manual parsing as necessary.

---

### 3. Data Structures
- **Adjacency:**
  - Build a sparse adjacency matrix \(A\) using `scipy.sparse` (e.g., `csr_matrix`)
  - Convert to PyTorch sparse tensor when needed (using `torch.sparse_coo_tensor`).
  - For GCN, add self-loops (`A + I`) before normalization.
- **Features:**
  - Load into `numpy` array or `scipy` sparse matrix, then convert to `torch.FloatTensor`.
- **Labels:**
  - Load as numpy array, convert to `torch.LongTensor`.
- **Masks:**
  - Create boolean masks for train/val/test splits.
  - Use a consistent seed when shuffling or splitting data for reproducibility.
  
---

### 4. Data Loading and Preprocessing
- **Loading:**
  - For each dataset, implement a method that reads files or downloads data.
  - The method should output adjacency, features, labels, and masks.
- **Filtering (for heterophilic datasets):**
  - For datasets like Squirrel and Chameleon, perform filtering to remove data leakage.
  - For example, ensure the train/val/test nodes are separated meaningfully, potentially removing edges that connect train and test nodes, or based on provided splits.
- **Normalization:**
  - For features: possibly normalize each node feature vector (e.g., row-wise L2 normalization) if required.
  - For adjacency: ensure the adjacency matrix is symmetric (if undirected), normalized as per GCN (`D^{-1/2} A D^{-1/2}`).
  - Ensure the adjacency matrix is in sparse format compatible with torch.

---

### 5. Data Splits & Randomness
- Use the seed (`seed=42`) for reproducibility.
- Maintain consistent splits across experiments:
  - Support multiple splits (`splits=10`).
  - For each split: randomly permute nodes, assign first 60% to train, next 20% val, remaining 20% test.
- Store masks accordingly:
  - `train_mask`, `val_mask`, `test_mask`, boolean arrays indicating which nodes belong to each split.
- Provide a method to generate these splits programmatically for datasets where they are not pre-defined.

---

### 6. Dataset-Specific Details
- For datasets with labels: ensure labels are aligned with node ordering.
- For datasets with node features: ensure feature dimensions are consistent.
- For datasets with known splits: load predefined splits if available; otherwise, generate splits using the seed.

---

### 7. Handling Dataset Variations
- Define dataset-specific loading functions if necessary due to format differences.
- Maintain a dataset-to-filepath or URL mapping.
- Support filtering or preprocessing steps specific to the dataset:
  - E.g., in case of Minesweeper, features could be provided as grid data; convert into graph form.
  - For other datasets, ensure data is consistently processed into adjacency, features, and labels.

---

### 8. Return Structure
- `load_data()` should return:
  - `adjacency`: sparse matrix as `torch.sparse.FloatTensor`
  - `features`: `torch.FloatTensor` (shape: [num_nodes, feature_dim])
  - `labels`: `torch.LongTensor` (shape: [num_nodes])
  - `train_mask`: `torch.BoolTensor`
  - `val_mask`: `torch.BoolTensor`
  - `test_mask`: `torch.BoolTensor`

### 9. Implementation Considerations
- Use `torch.device` to ensure tensors are on GPU if available.
- Process large datasets efficiently; convert sparse matrices appropriately.
- Add exception handling if files are missing or data formats are incompatible.
- Document data source assumptions and preprocessing steps thoroughly for reproducibility.

---

### Summary of Main Steps in dataset_loader.py

1. **Initialization:**
   - Accept config parameters: dataset name, seed, data directory/URL.
2. **Data Acquisition:**
   - Download or load datasets.
3. **Preprocessing:**
   - Build adjacency, add self-loops.
   - Normalize adjacency for GCN.
   - Load and process node features.
   - Load labels.
   - Generate or load train/validation/test splits with masks.
   - For heterophilic datasets, apply filtering to prevent leakage.
4. **Output:**
   - Return tensors: adjacency in sparse form, features, labels, masks.

---

This logical breakdown guides the code implementation for the dataset loader, ensuring datasets are loaded reliably, processed correctly, and suitable for subsequent diffusion and GNN modeling in the main codebase.

## evaluation.py

**Evaluation.py Logic Analysis**

Objective:  
Implement an `Evaluation` class responsible for performing inference on test data, calculating evaluation metrics (e.g., accuracy), visualizing node representations (if enabled), and reporting statistical measures such as mean and standard deviation over multiple dataset splits. The class relies on the trained model, diffusion functions, and dataset information, and it must produce reproducible, interpretable results in line with the paper's methodology.

---

### Core Responsibilities

1. **Data Preparation & Loading**
   - Receive dataset tuple: adjacency matrix, features, labels, train/val/test masks.
   - Support multiple splits (per configuration `splits: 10`).
   - Load or reconstruct the diffusion model instance used during training.
   
2. **Model Loading & Initialization**
   - Load saved model states, including diffusion parameters and network weights.
   - Instantiate the `DiffusionModel` with the same configuration used for training.
   - Ensure the inverse functions (forward diffusion, inverse process) of the model are available.
   
3. **Inference Procedure**
   - For each dataset split:
     - Extract the test nodes using test mask.
     - Compute node representations via:
       - Forward diffusion (`forward()`) at timestep \(T_F\)
       - Inverse diffusion (`inverse_process()`) at timestep \(T_R\) (with negative value)
     - Perform the *matching* of the representations (both forward and reverse) to the test nodes.
     - Utilize the trained predictor head (`predict_logits()`) to generate class probabilities or labels.
   
4. **Representation Visualization** (Optional)
   - If `visualization: true`:
     - Collect node representations from specific layers:
       - Forward process at last layer
       - Reverse process at last reverse layer
       - Concatenated representations (both directions)
     - Visualize the representations:
       - Use t-SNE or PCA for 2D embedding if needed.
       - Color nodes by true labels for interpretability.
     - Save or display plots, following the visualization layer set in the config.
   
5. **Metrics Computation**
   - Calculate classification accuracy (or other metrics like ROC-AUC for binary data) per split.
   - Aggregate accuracy over all splits to compute mean and standard deviation.
   - Similarly, for visualization, report qualitative analyses unless quantitative metrics are specified.

6. **Statistical Reports**
   - Output mean ± std deviation metrics.
   - Optional: report GSL or other oversmoothing measures if relevant (if datasets and models have GSL info stored or accessible).
   
7. **Reproducibility & Logging**
   - Log key info: dataset name, seed, model hyperparameters, diffusion steps, version of diffusion parameters.
   - Save output metrics to a report file or stdout.
   - For visualization, store images with consistent naming conventions.

---

### Implementation Details & Considerations

- **Data Handling & Masks**
  - Loop over all splits:
    - Use test mask to isolate test nodes.
  - For each split, provide seed consistency for reproducibility.
  
- **Computational Workflow per Split**
  - Given features \(\mathbf{X}\), adjacency \(\mathbf{A}\):
    - Compute \(\mathbf{X}_f\) via `forward()` at \(T_F\).
    - Compute \(\mathbf{X}_r\) via `inverse_process()` at \(T_R\), starting from \(\mathbf{X}_f\).
    - Extract the representations:
      - Forward last-layer node features
      - Reverse last-layer node features
      - Concatenated representations (if applicable)
    - Run the predictor head (MLP or linear classifier)
    - Compare predictions with true labels
  
- **Metrics and Statistics**
  - Collect accuracy for each split.
  - Calculate overall mean ± std deviation.
  
- **Visualization**
  - Use matplotlib, seaborn, or similar libraries.
  - Generate scatter plots:
    - Color points by true label.
    - Overlay representations from different layers or methods.
  - Save visualizations for comparison (e.g., in figures folder).

- **Handling Large Datasets**
  - Implement batching for large graphs if needed for visualization.
  - Use sparse matrix operations for efficiency during diffusion/inversion.
  
- **Error Handling & Validation**
  - Check if model files exist before inference.
  - Confirm diffusion parameters align with training setup.
  - Validate inverse process accuracy (small error expected, as per paper).

- **Hyperparameters & Configs**
  - Load parameters like `diffusion_time_TF`, `diffusion_time_TR`, `fixed_point_iter`, etc., directly from configuration for consistency.
  - Use the same device (CPU or CUDA) as training for inference.

---

### Data Structures & Internal Workflow

- **Input/Output Interfaces**
  - Constructor: Accept dataset tuple, model state path, and config.
  - Methods:
    - `evaluate()`: Runs inference over all splits.
    - `compute_metrics()`: Computes accuracy (or other metrics).
    - `visualize()`: Generates plots (if enabled).
  - Internal members:
    - `self.model`: DiffusionModel object (includes forward and inverse functions).
    - `self.predictor`: trained predictor head.
    - Dataset tensors/masks.
    - Diffusion hyperparameters.
  
- **Flow Summary**
  ```
  load dataset; load model states
  for each split:
      get test indices
      compute forward representations @ TF
      compute inverse representations @ TR
      run predictor on both representations
      compute accuracy
      if visualization:
          generate plots
  aggregate metrics:
      report mean ± std
  save/print results
  ```

---

### Final Notes

- Ensure that all diffusion and inverse computations are consistent with training parameters.
- Maintain reproducibility: set seeds, follow normalization protocols, and use identical hyperparameter settings.
- Leverage existing diffusion functions from `model.py`.
- Keep visualization interpretable — replicate Figures 1–4 style (e.g., colored scatter plots).

---

This detailed logic analysis ensures a robust, faithful implementation of the evaluation phase that directly supports the measurement of the described reverse diffusion GNN's effectiveness, representation quality, and over-smoothing mitigation capability.

## main.py

{
  "main.py": [
    {
      "step": "Import necessary modules",
      "details": "Import torch, dataset_loader, model, trainer, evaluation, and utils modules as per the project structure. Import numpy for any array manipulations, and tqdm for progress visualization. Also, import os and random for seed setting and environment management. Ensure to set the device (GPU if available, else CPU)."
    },
    {
      "step": "Set reproducibility seeds",
      "details": "Set torch.manual_seed, numpy.random.seed, and random.seed using the seed from config (e.g., 42). This ensures reproducibility across runs."
    },
    {
      "step": "Read the configuration parameters",
      "details": "Load 'config.yaml' via PyYAML or similar library. Parse the 'dataset', 'training', 'model', 'evaluation' sections to extract hyperparameters, dataset name, diffusion times, layers, fixed point iterations, and evaluation metrics."
    },
    {
      "step": "Initialize device",
      "details": "Create a torch.device object, e.g., device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'). Log or print the chosen device for confirmation."
    },
    {
      "step": "Dataset loading",
      "details": "Create an instance of DatasetLoader with dataset_name and other dataset-specific parameters. Call load_data() method to obtain the adjacency matrix (sparse tensor), features tensor, labels tensor, and train/validation/test masks. The input includes dataset name, split ratios, and seed for reproducibility."
    },
    {
      "step": "Build the diffusion model",
      "details": "Instantiate DiffusionModel class with adjacency, number of layers, diffusion times, fixed point iteration count, and device. Pass normalization flag as per config. The diffusion model is responsible for forward diffusion steps, attention matrix computation, and inverse process implementation. The diffusion object will be used during training and inference."
    },
    {
      "step": "Create the predictor head",
      "details": "Instantiate PredictorHead with input dimension (matching features dimension), number of classes (from dataset labels), and potentially hidden dims. This component maps the concatenated forward/reverse representations to logits or classification output."
    },
    {
      "step": "Initialize the trainer",
      "details": "Create a Trainer instance passing in diffusion model, predictor head, dataset tensors, hyperparameters (learning rate, dropout, epochs, patience, batch size, etc.), and device. The trainer manages training loop, validation, testing, and checkpointing."
    },
    {
      "step": "Start training",
      "details": "Invoke trainer.train(), passing in training data and masks. During training, ensure to handle batching if datasets are large (consider full-batch training if memory permits). Save best model parameters based on validation accuracy via early stopping."
    },
    {
      "step": "Validation & Hyperparameter Tuning",
      "details": "During training, monitor validation metrics. Hyperparameters (diffusion times, layers, M, learning rate, weight normalization) are tuned based on validation performance; in the main code, can use grid search or fixed hyperparams as per config."
    },
    {
      "step": "Load the best model",
      "details": "After training completes, load the best checkpoint (via trainer.load_model()) for final evaluation."
    },
    {
      "step": "Evaluation on test set",
      "details": "Create an Evaluation instance with the trained model, dataset, and device. Call evaluate() to compute metrics such as accuracy. If visualization is enabled, generate representations at specified layers and save plots/images of the learned node features according to visualization_layers (e.g., forward, reverse, concatenated)."
    },
    {
      "step": "Visualization and qualitative analysis",
      "details": "If visualization flags are set, save node representations, label predictions, and produce figures similar to Figures 1–4—saving images in an output directory, with the appropriate subplot or overlay detailing distinguishability of representations."
    },
    {
      "step": "Output and logging",
      "details": "Print or log final test metrics, training curves, and visualization results. Save model weights, logs, and any generated figures. This ensures reproducibility and detailed record-keeping."
    }
  ],
  "notes": [
    "- Ensure to handle dataset splits correctly according to the specified ratio and seed for reproducibility.",
    "- For large datasets or high reverse layers, consider batching or memory-efficient implementations, but in the main.py, keep it as direct as possible.",
    "- The diffusion parameters (T_F, T_R, M) should be passed directly from config and used in diffusion object during diffusion and inverse process calls.",
    "- The code should be modular: data loading, model instantiation, training, and evaluation should be clearly separated for clarity and debugging.",
    "- Confirm that all modules are correctly imported and that each class/method adheres to the described interface."
  ]
}

## model.py

**Logic Analysis for `model.py` — Core Diffusion Model & Inverse Process in GNNs**

---

### Overview:
The `model.py` module encapsulates the core logic for implementing the reverse diffusion process in GNNs as described in the paper. The main class, `DiffusionModel`, manages the forward diffusion via Euler integration, computes attention matrices for GRAND, implements normalization steps to ensure invertibility, and performs the inverse process through fixed-point iterations. It also provides methods for applying GNN layers with residual/invertibility constraints, stacking multiple reverse layers, and combining features for downstream classification.

---

### 1. **Class Structure & Initialization (`DiffusionModel`)**
- **Inputs (`__init__`):**  
  - Adjacency matrix (`adj`): a sparse tensor (torch.sparse.FloatTensor)  
  - Input feature dimension (`in_features`)  
  - Hidden dimension (`hidden_dim`)  
  - Number of `L_F` forward layers, number of `L_R` reverse layers  
  - Diffusion times (`T_F` for forward, `T_R` for reverse)  
  - Fixed-point iteration count (`M`)  
  - Attention heads, optional (default 1)  
  - Max diffusion steps for Euler (`max_diffusion_steps`)  
  - Attention dimensions and normalization flags (e.g., `normalize_weights`)  
  - Device (CPU/GPU)

- **Initialization tasks:**
  - Store adjacency as sparse tensor, ensure added self-loops if needed
  - Set hyperparameters
  - Initialize learnable weight matrices (`W`), attention parameters (`W_K`, `W_Q`, `a`), for attention-based adjacency matrix computation
  - Prepare weight normalization (spectral norm constraint) to ensure Lipschitz constant `< 1`
  - Possibly set initial node features placeholders

---

### 2. **Diffusion via Euler Method (`forward_diffusion`)**
- **Purpose:**  
  Simulate forward diffusion of node features over time \(T_F\).

- **Inputs:**  
  - Initial features `x0`  
  - Number of Euler steps (discretization) `M` (from config)  
  - Step size `dt` (diffusion step size, from config)

- **Implementation:**
  - Initialize `x` with `x0`  
  - Loop for `M` steps:
    1. **Compute diffusion matrix (`A(x)`)**:  
       - Use current node features `x` to compute attention matrix `A(X)` (using `compute_attention`)  
       - This matrix is stored for the step or used directly  
    2. **Update features:**  
       \[
       \mathbf{X}^{(t+\Delta t)} = \mathbf{X}^{(t)} + \Delta t \times (\mathbf{A}(\mathbf{X}^{(t)}) - \mathbf{I}) \mathbf{X}^{(t)}
       \]
    3. **Update `x`** for next iteration.

- **Output:**  
  - Node features at timestep `T_F`, stored as `x` after the last step.

---

### 3. **Compute Attention Matrix (`compute_attention`)**
- **Purpose:**  
  Generate the attention-based adjacency matrix `A(X)` for current features `X`.

- **Details:**
  - Use node features `X` to calculate queries and keys:
    - \(Q = \mathbf{W}_Q \mathbf{X}\), \(K = \mathbf{W}_K \mathbf{X}\)  
  - Compute scaled dot-product attention scores:
    \[
    [A(X)]_{ij} = \text{softmax}_j\left(\frac{(\mathbf{W}_K \mathbf{x}_i)^T (\mathbf{W}_Q \mathbf{x}_j)}{d'}\right)
    \]
  - If multi-head attention enabled, compute multiple attentions and average them.
  - **Masking:** restrict to observed edges (`\(\mathcal{E}\)`) — sparse matrix form.
  - **Output:** sparse attention matrix `A(X)` (torch sparse tensor).

---

### 4. **Weight normalization for invertibility (`normalize_weights`)**
- **Purpose:**  
  Ensure the residual layer's Lipschitz constant \(< 1\) by normalizing weights.

- **Procedure:**
  - Compute spectral norm of `\(\hat{\mathbf{A}}\)` (known or normalized to 1)
  - Normalize `\(\mathbf{W}\)`:
    \[
    \mathbf{W} \leftarrow \frac{c \mathbf{W}}{\|\mathbf{W}\|_F}
    \]
    where \(c \in (0,1)\), e.g., 0.9, or as specified in config
  - Implementation: after each weight update in training, apply normalization using `torch.nn.utils.spectral_norm` or manual normalization.

---

### 5. **GNN Layers with Residual / Invertibility Constraints**
- **Standard GNN layer (e.g., residual GCN gait):**
  \[
  \mathbf{X}^{(\ell+1)} = \mathbf{X}^{(\ell)} + \sigma(\hat{\mathbf{A}} \mathbf{X}^{(\ell)} \mathbf{W})
  \]
- **Inverse layer:**
  - Use fixed-point iteration (Algorithm 1):
    - Initialize `X^0` at input features of the layer.
    - For `M` iterations:
      \[
      \mathbf{X}^{(m+1)} = \mathbf{X}^{(m)} - h(\mathbf{X}^{(m)})
      \]
      where \(h(\mathbf{X})\) models the residual `\(\sigma(\hat{\mathbf{A}}\mathbf{X}\mathbf{W})\)` or linear parts, ensuring contraction.

- **Implementation of inverse:**
  - Repeat until convergence or fixed number `M`.
  - Use residual block `h` function that satisfies Lipschitz constraint.
  - During training, normalize weights periodically to maintain invertibility.

---

### 6. **Inverse process for reverse layers (`inverse_process`)**
- **Input:**  
  - Features `xT` at timestep \(T_F\) (from forward diffusion)
  - Target timestep \(T_R<0\)
  - Number of fixed point iterations \(M\)

- **Procedure:**  
  - Initialize `x` with `xT`
  - Run `M` fixed point iterations:
    \[
    \mathbf{X}^{(m+1)} = \mathbf{X}^{(m)} - h(\mathbf{X}^{(m)})
    \]
  - The `h` here models the negative diffusion dynamics:
    \[
    \frac{\partial \mathbf{X}(t)}{\partial t} \approx (\mathbf{A}(\mathbf{X}(t)) - \mathbf{I}) \mathbf{X}(t)
    \]
    but in inverse, steps are reversed and approximated with iterative solvers.

- **Output:**  
  - Estimated features at timestep \(T_R\), more distinguishable.

---

### 7. **Stacking Multiple Reverse Layers**
- For `L_R` reverse layers:
  - Sequentially apply `inverse_process` starting from the last diffusion point, producing progressively earlier/equivocal features.
  - Each layer's output acts as input to the next inverse layer.
  - Use the same fixed-point iteration with the specified `M`.

---

### 8. **Model outputs & combination**
- **Extract features:**  
  - From forward diffusion path: final node features `X_f^{(L_F)}`
  - From inverse (reverse) path: final node features `X_r^{(L_R)}`
- **Concatenate features:**  
  \[
  \text{representation} = \text{concat}(X_f^{(L_F)}, X_r^{(L_R)})
  \]
- **Prediction head:**  
  - Use an MLP or linear layer as per config to produce logits
  - Apply softmax or sigmoid depending on task (multi-class or binary)

---

### 9. **Additional Notes & Stability**
- Ensure invertibility via weight normalization, constraining spectral norms
- Use contractive nonlinear activation functions (ReLU, tanh, ELU)
- Small diffusion step size \(\Delta t \approx 0.1\) ensures stability
- For large datasets, implement sparse matrix computations
- Store attention matrices and normalized weights for reproducibility

---

### 10. **Summary of Key Methods & Functions in `model.py`**
| Function / Method | Purpose | Key Operations |
|---|---|---|
| `__init__()` | Initialize model, hyperparameters, and weights | Store adjacency, set hyperparameters, init weights |
| `compute_attention(X)` | Compute attention matrix | Attention score calculation, softmax, sparse tensor form |
| `forward_diffusion(x0, M, dt)` | Euler forward diffusion simulation | Loop, update features via `(A - I)X` |
| `normalize_weights()` | Normalize weights for invertibility | Spectral norm, Frobenius norm normalization |
| `inverse_process(xT, T_R, M)` | Approximate the inverse features | Fixed point iteration, residual operator `h` |
| `gnn_layer(X, W, A)` | Forward residual GNN layer | Residual connection + activation |
| `inverse_gnn_layer(X, W, A, M)` | Inverse residual layer | Fixed point iterations, contraction conditions |
| `combine_representations(Xf, Xr)` | Concatenate features for prediction | Final features for downstream head |

---

This detailed logic analysis should guide precise implementation, ensuring adherence to the paper’s theory, the mathematical basis for diffusion and inverse processes, and the experimental setup.

## trainer.py

# Logic Analysis for trainer.py

The `trainer.py` module implements the `Trainer` class responsible for managing the complete training and validation process of the diffusion-based reverse GNN models as described in the paper. This involves orchestrating data flow through the model, calculating losses, updating parameters, and periodically evaluating performance for early stopping and model saving.

Below is a detailed, step-by-step logical breakdown of the core functionalities needed to realize the `Trainer` class in conformance with the paper, plan, design, and configuration.

---

## 1. Initialization of the Trainer Class

**Inputs and Dependencies:**

- `model`: A `DiffusionModel` object that encapsulates forward diffusion, inverse process, attention calculation, and weight normalization functionalities.
- `predictor`: A `PredictorHead` object that combines the forward and reverse representations for classification.
- `data`: A tuple containing all datasets components—adjacency matrix, features, labels, train/validation/test masks. Extract as: `(adj, features, labels, train_mask, val_mask, test_mask)`.
- `hyperparams`: A dictionary of hyperparameters from `config.yaml`. Include diffusion times, layer counts, fixed point iteration steps, learning rate, dropout, etc.
- `device`: `torch.device` object (CPU or CUDA).

**Actions:**

- Store references for convenient access.
- Initialize optimizer (e.g., Adam) with `model` and possibly `predictor` parameters.
- Set up learning rate, weight decay, dropout rate.
- Initialize best validation performance metric for early stopping.
- Set internal counters, such as epoch number, patience counter.
- Prepare for model checkpointing (saving best models).

---

## 2. Data Preparation and Loading

- Transfer features, labels, masks, and adjacency to the correct device.
- Ensure features are tensors compatible with the model.
- Possibly normalize features if required (not specified, but generally beneficial).

## 3. Training Loop (`train()` method)

**Repeat for each epoch:**

### A. Set model to training mode:
```python
model.train()
predictor.train()
```

### B. Forward diffusion and dataset input preparation:

- For each graph, perform forward diffusion:
  - Call `model.forward_diffusion(x0=features, M=M, dt=diffusion_step_size)`:
    - `x0`: Initial features.
    - `M`: Number of Euler steps for forward diffusion.
    - Return `xT`: Diffused features at `T_F`.

- Perform inverse process:
  - Call `model.inverse_process(xT=xT, target_time=T_R, M=M)`:
    - Results in `xT_R`: Features after inverse diffusion, indicating more distinguishable representations.

**Note:** These are performed once per epoch (or per batch if batching, but per the document, likely full-graph).

---

### C. Forward and reverse representations:

- Compute forward representations:
  - Pass `features` (or possibly diffused features `xT`) through forward GNN layers:
    - For `L_F` layers, perform `f^{(\ell)}(x)` via `model` or `diffusion.py` methods.
    - Save concatenation (or last) `X_f^{(L_F)}` for the forward pathway.

- Compute reverse representations:
  - Starting from `xT_R`, pass through `L_R` inverse residual layers:
    - Use fixed point iteration with `M` steps for each inverse layer.
    - Save the output `X_r^{(L_R)}`.

**Implementation details:**
- The layers of forward and inverse process are modular; call respective methods.
- During the inverse process, ensure the fixed point iteration is run until convergence or for the specified steps.

---

### D. Combine representations:
- Concatenate `X_f^{(L_F)}` and `X_r^{(L_R)}` (along feature dimension).
- Pass concatenation into `predictor.predict_logits()`:
  - Likely an MLP producing class logits for each node.

### E. Loss Calculation:
- Based on the task (classification):
  - Use `torch.nn.CrossEntropyLoss()` for multi-class.
  - Use `torch.nn.BCEWithLogitsLoss()` for binary classification.
- Only compute loss on training nodes (`train_mask`).

### F. Backpropagation:
- Zero optimizer gradients.
- Backward pass on loss.
- Apply gradient clipping if necessary (not specified).
- Step optimizer.

---

### G. Weight Normalization for Invertibility:

- After optimizer step, normalize `model` weights:
  - For each `W` used in residual/invertible layers:
    - Compute spectral norm (`\|\hat{\mathbf{A}}\|_2`), which is known to be ≤ 1 for GCN and GAT (per paper).
    - Normalize `W` so that `\|\hat{\mathbf{A}}\|_2 * \|W\|_F < 1`.
  - This ensures the Lipschitz condition for invertibility (`Lemma 1` and `Algorithm 1`).

---

### H. Tracking performance:

- Compute validation metrics:
  - Run full forward pass with `features`, compute `X_f^{(L_F)}` and `X_r^{(L_R)}` (no gradient updates).
  - Calculate validation accuracy/loss.
- Save the model state if validation improves.

### I. Early stopping:
- If no improvement over `patience` epochs:
  - Stop training loop.
  - Save best model checkpoint.

---

## 4. Model Saving & Checkpointing

- Save model state dict (parameters) when validation performance improves.
- Save along with optimizer state to resume if needed.
- Log metrics for monitoring training stability.

---

## 5. Validation & Testing (`validate()` method)

- Set model to evaluation mode.
- Perform forward diffusion + inverse process similar to training but without backprop:
  - Obtain forward `X_f^{(L_F)}`.
  - Obtain inverse `X_r^{(L_R)}`.
  - Concatenate features, pass through predictor.
  - Compute metrics: accuracy, ROC-AUC if applicable.
- Return metrics as a dictionary for logging.

---

## 6. Additional considerations:

- Implement visualization routines:
  - Plot or save representations (optional based on `visualization` flag).
  - Visualize distinguishability (e.g., visualization layers like Figures 1–4).

- Set reproducibility:
  - Use seed for all random operations.
  - Use deterministic flags if necessary for reproducibility.

- Handle batching or full-graph processing:
  - For large datasets, implement batching if memory constrained (not explicitly required but beneficial).

---

## 7. Summary of core function skeletons:

```python
class Trainer:
    def __init__(self, model, predictor, data, hyperparams, device):
        # initialize attributes, optimizer, scheduler if used
    def train(self):
        for epoch in range(max_epochs):
            self._train_one_epoch()
            val_metrics = self._validate()
            if early_stop_condition:
                break
        # save final model
    def _train_one_epoch(self):
        # set train mode
        # diffusion and inverse process
        # compute representations
        # compute loss
        # backpropagate
        # normalize weights
        # update optimizer
        # log metrics
    def _validate(self):
        # set eval mode
        # diffusion and inverse
        # compute validation metrics
        # return metrics dict
    def save_model(self, path):
        # save model state_dict
    def load_model(self, path):
        # load model state_dict
```

---

## Final notes:

- The core of the trainer revolves around integrating the diffusion and inverse processes into the training loop.
- The process must respect the design of the models, especially the invertibility constraints and how representations are combined.
- Hyperparameters such as diffusion times, number of layers, fixed point iteration steps, and learning rate are crucial for stable training. These should be tuned carefully, but defaults are provided in the configuration.
- Use consistent data normalization, seed fixing, and logging for reproducibility.

This detailed logical roadmap ensures a faithful, precise implementation aligned with the paper’s methodology and experimental setup.

