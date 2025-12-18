# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. To systematically reproduce the methodology described in the paper on **SliceGPT**, I will craft a detailed, step-by-step plan, covering both the core techniques and the experimental setup. This plan is structured into logical phases: understanding core concepts, implementing key transformations, performing slicing procedures, and validation/evaluation. For each phase, I will specify the exact technical steps, dataset considerations, hyperparameters, and evaluation metrics.

---

## I. Core Methodology and Conceptual Foundations

### 1. Understanding the "Computational Invariance" Principle
- **Key insight:** Transforming transformer weights via orthogonal matrices ($\mathbf{Q}$) preserves the network's output. 
- **Practically:**  
  - For each pair of linear layers (attention, FFN, LayerNorm variants), define their weight matrices as $\mathbf{W}_\text{in}$ and $\mathbf{W}_\text{out}$.  
  - Apply transformations:
    \[
    \tilde{\mathbf{W}}_\text{emb} = \mathbf{W}_\text{emb} \mathbf{Q},
    \quad
    \tilde{\mathbf{W}}_\text{in}^\ell = \mathbf{Q}^\top \mathbf{W}_\text{in}^\ell,
    \quad
    \tilde{\mathbf{W}}_\text{out}^\ell = \mathbf{W}_\text{out}^\ell \mathbf{Q}.
    \]
  - Transform biases accordingly.
  - **Implementation note:** These transformations leave the network's output unchanged, enabling the reorganization of the embedding space and subsequent slicing.

### 2. Transforming to RMSNorm
- **Why:** The invariance applies only to RMSNorm-connected transformers.
- **Procedure:**  
  - Absorb LayerNorm linear and bias terms into the adjacent matrices (before and after the RMSNorm layers), as per Eq (6).  
  - **Implementation:**  
    - For each RMSNorm layer, adjust the input weight matrices by the respective scale factors from the original LayerNorm.
    - Adjust the embedding and head matrices to account for mean subtraction.
- **Outcome:** The entire transformer becomes RMSNorm-based, enabling the orthogonal invariance.

### 3. Eigen-spectrum Calculation and Orthogonal Matrices
- **Purpose:** Use PCA (eigen-decomposition) on signals from each layer to construct the orthogonal matrix $\mathbf{Q}_\ell$.
- **Steps:**  
  - Collect activation signals $\mathbf{X}_\ell$ (e.g., from a calibration dataset).  
  - Compute the covariance matrix:
    \[
    \mathbf{C}_\ell = \sum_i \mathbf{X}_{\ell,i}^\top \mathbf{X}_{\ell,i}
    \]
  - Extract eigenvectors and eigenvalues: $\mathbf{Q}_\ell$ as eigenvectors sorted by eigenvalues.  
  - **Implementation detail:** Use double precision (FP64) for eigen-decomposition (PyTorch `torch.linalg.eigh`) for numerical stability.

---

## II. Implementation Steps

### 1. Prepare Model Infrastructure
- Select supports: Hugging Face Transformers with PyTorch.
- Download and instantiate the models (LLAMA-2, OPT, Phi-2) from Hugging Face or from local checkpoints.
- Enable access to all weight matrices (attention, FFN, embeddings, etc.).

### 2. Convert the Model to RMSNorm (if applicable)
- For each RMSNorm layer:
  - Scale the weight matrices by the stored RMSNorm scale factors.
  - Pre-multiply the linear weights with these scales.
  - Adjust biases accordingly for mean subtraction.
- Modify the embedding and output matrices:
  - Subtract mean across embedding vectors, and re-scale as per the scale factors.

### 3. Compute orthogonal transformations $\mathbf{Q}_\ell$
- For each transformer layer:
  - Collect a subset of signals (from calibration dataset; e.g., WikiText-2, Alpaca).
  - Compute the covariance matrix of signals.
  - Extract eigenvectors (`torch.linalg.eigh` in FP64).
  - Save eigenvectors in sorted order as $\mathbf{Q}_\ell$ matrices.
  
### 4. Apply Orthogonal Transformations
- For each layer:
  - Transform the weights:
    \[
    \tilde{\mathbf{W}}_\text{emb} = \mathbf{W}_\text{emb} \mathbf{Q}_0, \quad
    \tilde{\mathbf{W}}_\text{in}^\ell = \mathbf{Q}_\ell^\top \mathbf{W}_\text{in}^\ell, \quad
    \tilde{\mathbf{W}}_\text{out}^\ell = \mathbf{W}_\text{out}^\ell \mathbf{Q}_\ell,
    \]
  - Transform biases if necessary.
- Inject these into the model parameters.

### 5. Slice the Model (Remove Bottom Rows / Columns)
- For each weight matrix in attention, FFN, embedding, head:
  - Decide slicing ratio (e.g., 25%, 50%, 75%).
  - Use PCA eigenvectors to identify which components/rows/columns are less significant.
  - Retain top eigenvectors corresponding to largest eigenvalues, truncate the rest.
- Update matrices accordingly, adjusting dimensions.
- For residual pathways, insert the $\mathbf{Q}_\ell$ matrices into skip connections as per the residual micro-architectural transformations (Figure 4).
- Ensure consistency:  
  - Row deletion in input weight matrices corresponds to skipping rows in signals; similarly for output matrices.

---

## III. Experimental Setup and Hyperparameters

### 1. Model Selection and Sizes
- Models: LLAMA-2 (7B, 13B, 70B), OPT (125M to 66B), Phi-2 (various sizes).
- Use the publicly available checkpoints; ensure they include all relevant weights.

### 2. Calibration Data
- Datasets: WikiText-2 (small, fast PCA) and Alpaca (larger, downstream performance).
- Batch sizes for collecting signals: 128–1024 sequences.
- Sequence length: 1024 or 2048 tokens.
- Number of samples: At least 1024 (per paper) for PCA stability.

### 3. PCA Precision
- Use FP64 eigen-decomposition (PyTorch `torch.linalg.eigh`) for covariance eigenvectors.
- Record the eigen-spectrum to identify commonality and spectral decay.

### 4. Slicing Ratios
- 20%, 25%, 30%, 50%, as per experiment.
- Decide in practice based on spectrum decay and performance tradeoffs.

### 5. Fine-tuning / Recovery
- Apply lightweight recovery fine-tuning:  
  - Use LoRA (with specified rank, alpha) on sliced models.
  - Calibration dataset: ~1k sequences for WikiText-2, ~5k sequences for Alpaca.
  - Number of steps: ~1–3 epochs; ensure minimal training to preserve performance.
- Evaluate pre- and post-fine-tuning.

### 6. Evaluation Metrics
- Zero-shot perplexity on WikiText-2.
- Zero-shot accuracy on standard NLP tasks (e.g., PiQA, WinoGrande, HellaSwag, ARC tasks).
- Generation speed (ms/token) on representative hardware (A100, H100, RTX6000).
- Model size after slicing (parameters, memory).

---

## IV. Validation and Evaluation

### 1. Baseline Comparisons
- Dense baseline (original models).
- SparseGPT (2:4 sparsity) after retraining.
- SliceGPT (varying ratios), with and without fine-tuning.

### 2. Spectrum Analysis
- For each layer, compute and plot eigen spectra before/after transformation.
- Confirm spectral decay and slice more aggressively in early layers.

### 3. Performance and Speed
- Measure inference latency for single-token generation.
- Measure throughput (tokens/sec) on GPUs with batch size scaling.
- Cross-validate that sliced models retain performance metrics within acceptable delta.

---

## V. Implementation Caveats and Clarifications
- Dependence on the eigen-decomposition: numerical stability suggests use of FP64 and convergence checks.
- Handling layers with biases: apply transformations consistently.
- Maintaining residual pathways: matrix insertions for skip connections.
- Noise sensitivity: PCA (covariance eigen decomposition) might be sensitive to calibration set size; test with different dataset samples.
- Fine-tuning: minimal retraining, possibly with PEFT/LoRA, to recover performance.

---

## VI. Summary Roadmap
**High-Level Tasks:**

1. Instantiate pre-trained models.
2. Convert models to RMSNorm.
3. Collect activation signals from calibration datasets.
4. Compute covariance matrices and eigenvectors for each layer.
5. Construct orthogonal matrices $\mathbf{Q}_\ell$.
6. Transform weights with $\mathbf{Q}_\ell$.
7. Slice weights by retaining top eigen components.
8. Fix residuals accordingly.
9. Optionally, perform recovery fine-tuning.
10. Evaluate perplexity, accuracy, and speed.

---

This plan provides an explicit blueprint for implementing the method and conducting experiments. It explicitly specifies data requirements, hyperparameters, mathematical operations, and evaluation metrics, enabling the subsequent coding phase to be precise and reproducible.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular system that loads pre-trained transformer models (LLAMA-2, OPT, Phi-2), converts them to RMSNorm if necessary, computes per-layer orthogonal matrices via PCA on activation signals, applies the transformations to weights, performs structured weight slicing based on eigen-spectrum, and optionally retrains via lightweight fine-tuning with LoRA. This will involve a pipeline of data collection, spectrum calculation, weight transformations, slicing, and evaluation, all using PyTorch, Hugging Face Transformers, and numerical libraries like NumPy. The system overall will be controlled via a main script that orchestrates these steps.",
    "File list": [
        "main.py",
        "model_utils.py",
        "pca_transform.py",
        "slicer.py",
        "fine_tuning.py",
        "evaluation.py",
        "dataset_loader.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class ModelWrapper {\n        +__init__(model_name: str, model_ckpt_path: str, use_rmsnorm: bool)\n        +load_model() -> None\n        +convert_to_rmsnorm() -> None\n        +apply_transformation(Q: torch.Tensor, layer_idx: int) -> None\n        +slice_weights(layer_idx: int, keep_ratio: float) -> None\n        +get_weights(layer_idx: int) -> Dict[str, torch.Tensor]\n        +set_weights(layer_idx: int, weights: Dict[str, torch.Tensor]) -> None\n        +forward(inputs: torch.Tensor) -> torch.Tensor\n    }\n    class PCAProcessor {\n        +__init__(dataset: Dataset, layer_count: int)\n        +collect_signals() -> torch.Tensor\n        +compute_eigenvectors() -> List[torch.Tensor]\n        +save_eigenvectors(path: str) -> None\n        +load_eigenvectors(path: str) -> List[torch.Tensor]\n    }\n    class Slicer {\n        +__init__(model: ModelWrapper, eigenvectors: List[torch.Tensor])\n        +apply_slicing(layer_idx: int, ratio: float) -> None\n        +apply_residual_transform(layer_idx: int, Q: torch.Tensor) -> None\n    }\n    class FineTuner {\n        +__init__(model: ModelWrapper, dataset: Dataset)\n        +train(steps: int, lora_rank: int) -> None\n        +save_checkpoint(path: str) -> None\n    }\n    class Evaluator {\n        +__init__(model: ModelWrapper, datasets: Dict[str, Dataset])\n        +evaluate_perplexity() -> Dict[str, float]\n        +evaluate_zero_shot() -> Dict[str, float]\n        +measure_throughput(batch_size: int, sequence_length: int, device: str) -> float\n    }\n    Main --> ModelWrapper\n    Main --> PCAProcessor\n    Main --> Slicer\n    Main --> FineTuner\n    Main --> Evaluator\n    PCAProcessor --> Dataset\n    Slicer --> ModelWrapper\n    FineTuner --> ModelWrapper\n    Evaluator --> ModelWrapper",
    "Program call flow": "sequenceDiagram\n    participant MP as main.py\n    participant MW as ModelWrapper\n    participant PCA as PCAProcessor\n    participant SL as Slicer\n    participant FT as FineTuner\n    participant EV as Evaluator\n    MP->>MW: init(model_name, checkpoint_path, use_rmsnorm)\n    MW-->>MP: load_model()\n    MP->>MW: convert_to_rmsnorm()\n    MP->>PCA: collect_signals(dataset)\n    PCA-->>MP: signals\n    MP->>PCA: compute_eigenvectors()\n    PCA-->>MP: eigenvectors list\n    MP->>SL: apply transformation with eigenvectors per layer\n    loop for each layer:\n        MP->>SL: apply_slicing(layer_idx, ratio)\n    end\n    MP->>FT: optional train with dataset for recovery\n    FT-->>MP: save_checkpoint()\n    MP->>EV: evaluate_perplexity()\n    EV-->>MP: perplexity scores\n    MP->>EV: evaluate_zero_shot()\n    EV-->>MP: accuracy scores\n    MP->>EV: measure throughput(batch_size, sequence_length, device)\n    EV-->>MP: throughput\n    MP-->>User: output results and saved models",
    "Anything UNCLEAR": "Need clarification on the exact dataset preparation steps, hyperparameters for PCA and fine-tuning (e.g., number of steps, LoRA configs), and whether the models are to be loaded from specific checkpoints or Hugging Face hub. Also, details on the threshold for eigen-spectrum pruning and how residual skip connections are precisely modified need confirmation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "transformers==4.29.2",
        "numpy==1.21.0",
        "scipy==1.9.3",
        "datasets==2.8.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: responsible for loading calibration datasets (WikiText-2, Alpaca) with batching, tokenization, and sampling. Input: dataset name, path; Output: signals for PCA."
        ],
        [
            "model.py",
            "Defines ModelWrapper class: loads Hugging Face transformer models, supports conversion to RMSNorm, applies orthogonal matrices to weights, and slicing operations. APIs: load_model(), convert_to_rmsnorm(), apply_transformation(Q, layer_idx), slice_weights(layer_idx, ratio), get_weights(), set_weights(), forward()"
        ],
        [
            "pca_transform.py",
            "Defines PCAProcessor class: receives signals, computes covariance matrices, eigenvectors via torch.linalg.eigh (FP64), saves/loads eigenvectors, constructs Q matrices, and maintains spectrum information. Inputs: signals; Outputs: eigenvectors."
        ],
        [
            "slicer.py",
            "Defines Slicer class: initialized with ModelWrapper and PCA eigenvectors; methods for per-layer slicing, residual matrix application, and structural pruning based on eigen-spectrum eigenvalues. Dependence: get_weights() & set_weights() from ModelWrapper."
        ],
        [
            "fine_tuning.py",
            "Defines FineTuner class: performs lightweight LoRA fine-tuning for recovery after slicing. Inputs: model, dataset; functions for training (steps, LoRA rank), saving checkpoint."
        ],
        [
            "evaluation.py",
            "Defines Evaluator class: evaluates perplexity, zero-shot task accuracy, and throughput on specified datasets and hardware configuration using ModelWrapper. Methods: evaluate_perplexity(), evaluate_zero_shot(), measure_throughput()."
        ],
        [
            "main.py",
            "Serves as orchestrator: coordinates loading models, converting to RMSNorm, dataset loading, PCA eigenvector calculation, transformations application, weight slicing, optional fine-tuning, and evaluation. Calls all other modules in proper order."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "pca_transform.py",
        "slicer.py",
        "fine_tuning.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "N/A",
    "Shared Knowledge": "The get_weights() and set_weights() methods in ModelWrapper are used repeatedly to access and modify model weights after transformations. Eigenvector matrices are stored and loaded from disk to allow reuse and consistency across steps.",
    "Anything UNCLEAR": "Clarification needed on the specific granularity of weight matrices (attention, FFN, embeddings) to be sliced and the eigen-spectrum pruning thresholds. Also, whether to prioritize speedup or accuracy preservation when choosing ratios."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 1e-4
  batch_size: 128
  epochs: 3
  optimizer: AdamW
  weight_decay: 0.01
  warmup_steps: 500
  total_steps: 10000
model:
  model_name: "llama2"
  model_checkpoint_path: "/path/to/llama2/checkpoints"
  use_rmsnorm: true
data:
  calibration_dataset:
    name: "WikiText-2"
    sample_size: 1024
    sequence_length: 2048
  fine_tuning_dataset:
    name: "Alpaca"
    sample_size: 5000
    sequence_length: 1024
slicing:
  ratio: 0.25  # 25% weight parameters removal
  spectrum_threshold: "auto"  # based on eigen-spectrum decay
evaluation:
  tasks:
    - perplexity
    - zero_shot_accuracy
  hardware:
    device: "cuda"  # or "cpu" for testing
    gpus: 1
    use_speed_measure: true
```


---

**Note:** Full configuration is available in `planning_config.yaml`
