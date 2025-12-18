# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a detailed, structured plan to reproduce the experiments and methodologies from the paper "Inherently Interpretable Time Series Classification Via Multiple Instance Learning (MILLET)":

---

## 1. Clarify Overall Aim & Approach
- **Goal:** Implement MILLET: a framework that enables inherent interpretability in deep learning TSC models by leveraging various MIL pooling operations, positional encodings, and interpretability measures.
- **Key Innovations:**
  - MIL pooling methods (Global Average Pooling, Attention, Instance, Additive, Conjunctive).
  - Positional encodings integrated into embeddings.
  - Interpretability metrics (AOPCR, perturbation curves, ND CG).
  - Use of multi-component models (base CNN/ResNet/InceptionTime + MIL pooling + interpretability modules).
  - Comparison against baselines (GAP, HC2, Hydra-MR).

---

## 2. Data & Datasets
### Dataset Collection
- **Primary dataset:** UCR Archive (85 univariate datasets).
- **Synthetic dataset:** WebTraffic (for interpretability evaluation).
- **Additional datasets for imbalanced scenario analysis:** Select subsets with dataset imbalance metric < 0.9.
- **Data format prerequisites:**
  - Time series: univariate, fixed length $t$.
  - Labels: class labels per series.

### Data Preparation:
- Load datasets into a standardized format (e.g., pandas DataFrames or numpy arrays).
- For each dataset:
  - Organize into `(samples, timesteps)` arrays.
  - Class labels into `(samples,)`.
- Perform train/test splits as per the original datasets or use the default splits provided.
- For synthetic data, generate via the described process: sample parameters, inject signatures, and produce class labels.

---

## 3. Data Preprocessing
- **Normalization:**
  - Apply min-max normalization or z-score normalization per time series.
  - For synthetic data, normalize signature injections when appropriate.
- **Positional encoding addition:**
  - Implement fixed sinusoidal positional encodings (Vaswani et al., 2017).
  - Add positional encodings to embeddings after extraction from base models.
- **Handling variable lengths:**
  - Given datasets are fixed length; otherwise, pad or truncate.
- **Train/test splits** should be kept consistent with original settings for comparison.

---

## 4. Model Architecture & Implementation Details
### 4.1 Base Backbone network
- **Choices:** FCN, ResNet, InceptionTime.
- **Implementation:**
  - Use existing network architectures (available from open-source TSC repositories or re-implement following references).
  - Ensure flexibility to switch backbone architectures.
  - Use the same hyperparameters: e.g., number of layers, kernel sizes, residual blocks as per original papers or the paper’s detailed layer specifications.
- **Input:**
  - Shape: `(batch_size, 1, t)` for univariate.
- **Output:**
  - Embeddings at each time point: shape `(batch, 1, t, embedding_dim)`.

### 4.2 Positional Encoding Module
- **Implement fixed sinusoidal encodings:**
  - For each position in sequence $p \in [1, t]$:
    - Compute PE using formulas provided.
  - Add PE to embeddings after backbone but before pooling.

### 4.3 MIL Pooling & Interpretability Module
- **Pooling methods:**
  - Implement each as a modular layer, as per Tables A.1-A.5, with parameters:
    - Embedding size (128)
    - Attention size (8)
  
- **Pooling specifics:**
  - Embedding pooling (global average).
  - Attention pooling: attention head (two-layer: tanh + sigmoid activations) producing scores scaled by sigmoid or softmax as specified.
  - Instance pooling: produce per-time-point class predictions, then average over time.
  - Additive pooling: attention-weighted class predictions over time, then average.
  - Conjunctive pooling: separate attention and classifier heads; multiply attention scores with class predictions (scaled/scaled scores as per Eqn. 3).

- **Interpretability outputs:**
  - For each approach, produce:
    - Discriminatory scores per time point.
    - Attention weights.
    - Final class prediction for the series.

- **Implementation notes:**
  - Use PyTorch (recommended): each pooling method as a separate nn.Module.
  - Ensure differentiability and compatibility with the backbone model.
  - Incorporate dropout if specified.

### 4.4 Training & Optimization
- **Loss function:** Cross-entropy on class predictions.
- **Hyperparameters:**
  - Learning rate: 0.001 (or as per the paper’s recommendation).
  - Optimizer: Adam.
  - Batch size: 16 or as per dataset size.
  - Dropout rate: 0.1 (if specified).
  - Number of epochs: 1500 (or until convergence).
  - Early stopping: monitor validation accuracy or loss (optional but recommended).

- **Training procedure:**
  - For each dataset:
    - Train models with:
      - Only backbone + GAP (baseline).
      - Backbone + each MIL pooling approach.
    - For ensemble:
      - Combine multiple models’ logits via averaging.

- **Repeat runs:**
  - Run each configuration with multiple random seeds (e.g., 3 seeds) to estimate variance.

---

## 5. Interpretability Measures & Evaluation
### 5.1 Metrics
- **AOPCR ($\chi'$-NDCG):**
  - Use the interpretability scoring method as per formula (Eqn. A.8) and description.
  - For datasets with known signature regions, evaluate how well the top-ranked time points capture the signatures.
  - Average over multiple random orderings (3 repeats).

- **Perturbation curves:**
  - Sequentially remove time points based on interpretability scores.
  - Measure decay of model confidence or predicted probability.
  - Use for visual and quantitative comparison.

- **ND CG ($\chi'$-NDCG):**
  - For datasets with known ground-truth discriminative time points.
  - Measure rank correlation.

### 5.2 Interpretability datasets:
- For synthetic data:
  - Use known signature regions as ground truth.
  - Generate interpretability heatmaps as per Fig. A.2.
- For real datasets:
  - Use existing interpretability heatmaps (from benchmarks).

---

## 6. Additional Analyses and Ablations
- **Model variance analysis:** compute deviation from median accuracy (Fig. A.7).
- **Scalability & complexity analysis:**
  - Implement and verify Eqn. A.15.
  - Generate plots like Fig. A.8: how parameter counts and computation scale with time series length and number of classes.
- **Dataset imbalance influence:** correlate dataset imbalance metric with accuracy differences as shown in Fig. A.5.

---

## 7. Experimental Protocol Summary
- **For each dataset:**
  - Prepare data.
  - For each backbone + pooling method (including baseline GAP):
    - Initialize model with the described parameters and add positional encodings.
    - Train for 1500 epochs, with early stopping if desired.
    - Save models, logs.
    - Evaluate accuracy, AUROC, loss.
    - Compute interpretability scores (AOPCR, perturbation).
- **For synthetic datasets:** inject signatures as described, maintain signatures for interpretability evaluation.
- **For real datasets:** follow original train/test splits.
- **Aggregate results:** report means, variances, ranks.
- **Compare methods:** ensemble vs individual; ablation variants.

---

## 8. Implementation Notes & PlantUML Workflow
- Modular design:
  - Dataset loader → Preprocessing.
  - Backbone models: FCN, ResNet, InceptionTime.
  - Positional encoding generator.
  - Pooling modules: GAP, Attention, Instance, Additive, Conjunctive.
  - Interpretability evaluation functions.
- Use parameter configs to switch experiments.
- Leverage open-source TSC codebases (e.g., from torchTS, TSC repository) where effective.
- Use experiment tracking (e.g., logging with WandB or TensorBoard).
- Version control code (e.g., GitHub) and experiment scripts.

---

By following this roadmap, you will be able to systematically implement, train, evaluate, and interpret the models from the paper with the fidelity described. This plan addresses data handling, model architecture, training, interpretability, and complexity analysis, aligning with each element detailed in the original work.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular Python system using PyTorch as the core deep learning framework. The system will include components for dataset loading and preprocessing, a flexible backbone neural network (supporting FCN, ResNet, or InceptionTime variants), modules for positional encoding, multiple MIL pooling approaches, interpretability scoring, and evaluation. The training pipeline will handle hyperparameter configuration, manage multiple seeds for reproducibility, and support ensemble predictions. We will leverage open-source TSC repositories where applicable, customize modules for position encoding and interpretability, and provide core scripts for experiment orchestration, hyperparameter tuning, and performance evaluation.",
    "File list": [
        "main.py",  
        "dataset.py", 
        "model.py",  
        "modules.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiments()\n    }\n    class Dataset {\n        +__init__(dataset_name: str, split: str)\n        +load() -> Tuple[numpy.ndarray, numpy.ndarray]\n        +preprocess() -> None\n    }\n    class PositionalEncoding {\n        +__init__(max_length: int, embedding_dim: int)\n        +generate() -> torch.Tensor\n    }\n    class BackboneNetwork {\n        +__init__(hidden_dim: int, architecture: str)\n        +forward(x: torch.Tensor) -> torch.Tensor\n    }\n    class MILPooling {\n        +__init__(method: str, params: dict)\n        +pool(embeddings: torch.Tensor, attention_scores: Optional[torch.Tensor]) -> torch.Tensor\n    }\n    class Interpretability {\n        +__init__(model: nn.Module, pooling: MILPooling)\n        +compute_scores(series: torch.Tensor) -> torch.Tensor\n        +compute_perturbation(series: torch.Tensor) -> List[Tuple[torch.Tensor, float]]\n        +compute_ndcgc(series: torch.Tensor, true_signature_indices: List[int]) -> float\n    }\n    class Trainer {\n        +__init__(model: nn.Module, dataset: Dataset, config: dict)\n        +train() -> None\n        +evaluate() -> dict\n    }\n    class Evaluation {\n        +__init__(model: nn.Module, dataset: Dataset, interpretability: Optional[Interpretability])\n        +evaluate_metric(metric_name: str) -> float\n        +compute_all_metrics() -> dict\n    }\n\nMain --> Dataset\nMain --> BackboneNetwork\nMain --> MILPooling\nMain --> Interpretability\nMain --> Trainer\nMain --> Evaluation\nTrainer --> BackboneNetwork\nTrainer --> MILPooling\nInterpretability --> Model\nInterpretability --> MILPooling\nEvaluation --> Model\nEvaluation --> Interpretability\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DS as Dataset\n    participant BN as BackboneNetwork\n    participant MP as MILPooling\n    participant I as Interpretability\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DS: load() + preprocess()\n    DS-->>M: dataset ready\n    M->>BN: initialize(architecture='InceptionTime')\n    BN-->>M: backbone model\n    M->>MP: initialize(method='Conjunctive', params={})\n    MP-->>M: pooling module\n    M->>I: initialize(model, pooling)\n    I-->>M: interpretability instance\n    M->>TR: start training(model, dataset, seed=123)\n    TR->>BN: forward(x)\n    BN-->>TR: feature embeddings\n    TR->>MP: pool(embeddings, attention_scores)\n    MP-->>TR: pooled features\n    TR->>BN: classifier(pooled_features)\n    BN-->>TR: class logits\n    TR->>TR: backpropagation & optimizer step\n    loop epochs\n        ...\n    end\n    M->>EV: evaluate(model, dataset, interpretability)\n    EV->>I: compute_scores(series)\n    I-->>EV: interpretability scores\n    EV->>M: report metrics (accuracy, AUROC, interpretability scores)\n",
    "Anything UNCLEAR": "Clarify if the exact backbone architecture details, optimizer hyperparameters, and the expected dataset format (e.g., shape, normalization) are fixed or flexible. Also, confirm if interpretability metrics need implementation from scratch or can be adapted from available libraries."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.9.0",
        "numpy==1.21.0",
        "scipy==1.7.0",
        "pandas==1.3.0",
        "scikit-learn==0.24.2",
        "matplotlib==3.4.3",
        "seaborn==0.11.2",
        "tqdm==4.62.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset.py",
            "Defines Dataset class for loading datasets, handling train/test splits, normalization, and data batching. Dependency: numpy, pandas, scikit-learn."
        ],
        [
            "utils.py",
            "Contains utility functions such as positional encoding generator, normalization functions, and dataset statistics calculators. These are shared across modules."
        ],
        [
            "model.py",
            "Defines BackboneNetwork class supporting FCN, ResNet, InceptionTime architectures. Includes method for constructing specific models, outputting feature embeddings. Depends on torch, torch.nn, torch.nn.functional."
        ],
        [
            "modules.py",
            "Implements MIL pooling classes: GAP, Attention, Instance, Additive, Conjunctive pooling. Also includes PositionalEncoding class. Uses torch, torch.nn."
        ],
        [
            "interpretability.py",
            "Contains Interpretability class for computing scores like AOPCR, perturbation curves, ND CG, using the trained model and pooling modules. Depends on torch, numpy."
        ],
        [
            "trainer.py",
            "Handles training and validation loops, including model instantiation, optimizer setup, loss computation, and checkpoint management. Uses torch, torch.optim, and dataset.py."
        ],
        [
            "evaluation.py",
            "Provides functions for evaluating performance metrics (accuracy, AUROC, loss) and interpretability, integrating interpretability.py. Depend on the trained model, interpretability modules."
        ],
        [
            "main.py",
            "Acts as experiment orchestrator: parses configs, loads datasets, initializes models, pools, interprets, runs training/evaluation, manages reproducibility seeds. Uses all previous modules."
        ]
    ],
    "Task list": [
        "dataset.py",
        "utils.py",
        "model.py",
        "modules.py",
        "interpretability.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "<none>  # No server-side API needed for this experiment system, only internal module APIs.",
    "Shared Knowledge": "Common functions for positional encoding, parameter setting, dataset statistics, and model saving/loading. Hyperparameters are consistent across modules.",
    "Anything UNCLEAR": "Clarify dataset input formats and whether hyperparameter tuning is required or fixed, especially for backbone model choices and interpretability metric details."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001       # Default learning rate used for optimizer
  batch_size: 16             # Batch size adapted to 1500 epochs and dataset size
  epochs: 1500               # Fixed number as per paper's standard training setup
  early_stopping: true       # Optional: Enable early stopping based on validation performance
  seed: 123                  # Default seed for reproducibility
model:
  backbone: 'InceptionTime'  # Choices: 'FCN', 'ResNet', 'InceptionTime'
  embedding_dim: 128         # Embedding size for features extracted from backbone
  dropout_rate: 0.1          # Dropout rate as in standard practice
  architecture_params:       # Specific hyperparameters per backbone, can be extended
    residual_blocks: 4
    kernel_sizes: [8, 5, 3]
  pooling_method: 'Conjunctive'  # Options: 'GAP', 'Attention', 'Instance', 'Additive', 'Conjunctive'
  pooling_params:
    attention_heads: 1
    attention_size: 8
interpretability:
  method: 'AOPCR'            # Metrics: AOPCR, Perturbation, ND CG
  signature_injection: true  # Synthetic dataset signature injection enabled
  evaluation_repeat: 3       # Number of repeats for interpretability metrics
dataset:
  name: 'UCR_all'            # Use full archive or subset
  train_split_ratio: 0.8     # Standard train/test split
  normalization: 'z-score'   # Normalize time series before training
  synthetic: false           # Set true for synthetic datasets like WebTraffic
  dataset_dir: 'datasets/'   # Path to datasets
# Additional parameters can be added for specific dataset handling or tuning
```

---

**Note:** Full configuration is available in `planning_config.yaml`
