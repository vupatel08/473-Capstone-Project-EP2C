# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Based on the provided excerpts from the paper "Unveiling and Harnessing Hidden Attention Sinks", here is a comprehensive, detailed plan to reproduce the methodology and experiments described:

---

# Reproduction Plan for "Unveiling and Harnessing Hidden Attention Sinks"

## 1. Core Objective & Method Overview
- **Main goal:** Develop a training-free attention calibration method (ACT) that automatically adjusts attention distributions **during inference** to improve large language model (LLM) accuracy without weight finetuning.
- **Key insight:** Identification and modulation of **attention sinks**, which are tokens that disproportionately attract attention, often at low semantic value, negatively impacting model performance.

---

# 2. Methodology Components

### A. Attention Sink Identification
- **Definition:** Tokens with abnormally high attention scores, particularly those receiving disproportionate attention (attention sink).
- **Operationalization:**
  - Analyze attention weights at each layer and head, focusing on the distribution of the pairwise relationship among tokens (\(\mathbf{A}_h^l[i, j]\)).
  - Compute **attention scores for each token** as the average attention received across heads and layers.
  - Quantify attention sinks via a metric like the ratio of a token’s attention to the overall attention distribution (e.g., sum over attention weights normalized per token).
  - Identify **attention sinks** as tokens in the top percentile (e.g., top 1-5%) of attention scores or those exceeding an adaptive threshold (related to \(\alpha\)). Also, pay special attention to tokens within the middle input context.

### B. Attention Calibration Technique (ACT)
- **Main idea:** During inference, dynamically **reduce the attention weight** of identified sinks before generating the output.
- **Implementation steps:**
  1. **Layer-wise Attention Map Extraction:**
     - For a given input, extract the attention maps (\(\mathbf{A}_h^l\)) across all layers \(l\) and heads \(h\).
  2. **Sink Detection per Input:**
     - For each layer \(l\) and head \(h\), compute aggregate attention scores for each token.
     - Identify tokens with high attention scores (using a threshold \(\alpha\) or percentile-based cutoffs).
  3. **Attention Adjustment:**
     - For each sink token \(i\), modify the attention weights in the relevant layers/heads:
       - Multiply the sink's attention weights by a factor (e.g., \(1 - \beta\)), where \(\beta\) is a hyperparameter controlling the reduction.
       - Alternatively, re-normalize the attention distribution to diminish sink influence.
     - Use a **filtering scheme**:
       - "Layer-wise attention sink removal" (by setting attention weights to zero or reducing them with an adaptive factor).
       - "Selective attention redistribution": the reduced attention "mass" is redistributed among high semantic tokens, e.g., by normalizing the remaining attention map or selectively boosting certain tokens according to contextual relevance.
  4. **Layer-wise and Head-wise Application:**
     - Apply the attention reduction both **globally** and **per-head**, with options to calibrate only specific layers or heads if deemed beneficial.
  5. **Inference with Adjusted Attention:**
     - Run the LLM inference process with these modified attention maps (pseudo-injected at each transformer layer before the softmax computation).

### C. Attention Sink Filtering and Thresholding
- Use hyperparameters:
  - \(\alpha\): a threshold (e.g., from the paper’s range 3, 5, 7) to select high attention tokens.
  - \(\beta\): the magnitude of attention reduction (tuning hyperparameter).
  - "Subset size" controls how many tokens are considered sinks based on attention score percentile (e.g., top 40%, 60%, 80%, 100% as in the ablation).
  - Different configurations enable adaptive calibration depending on input length, dataset, or desired robustness.

---

# 3. Experimental Setup & Datasets

### A. Models
- **Base models:**
  - Large pretrained LLMs such as Llama-7B, Llama2-7B-chat, Llama2-13B, GPT-J-7B, OPT-2.7B, Vicuna-7B, OPT-6B, VLuna-30B, and others as referenced.
  - Obtain the models via public repositories, e.g., HuggingFace or official sources.
  - Use models in inference mode (no further finetuning).

### B. Tasks & Datasets
- **Classification Tasks:**
  - SST2, SST5, AGNews, PIQA, ARC (Easy and Challenge), CQA, BoolQ, and other datasets used in the paper.
- **Open-ended Question-Answering Tasks:**
  - RTE, CommonsenseQA, etc., specifically: datasets with labeled answers for performance evaluation.
- **Multisample/Multilabel Tasks:**
  - MT-Bench for multi-turn reasoning or complex language tasks.
- **Few-shot/Zero-shot Settings:**
  - Prepare prompts with the baseline prompts as per the paper, including few-shot exemplars.

### C. Evaluation Metrics
- **Accuracy/F1 Score:**
  - Classification datasets: Exact match (EM), F1, or accuracy.
  - Multiple-choice datasets: accuracy.
  - Open-ended Q&A: exact match, or task-specific metrics.
- **Performance Gains:**
  - Measure the relative improvement (percentage points) over vanilla inference.
- **Additional metrics:**
  - Attention map statistics, e.g., distribution histograms, sink ratios.
  - Model robustness across different hyperparameters.

---

# 4. Hyperparameters and Tuning Strategy
- **Attention reduction factor \(\beta\):** 0.4 (according to the paper); explore around 0.3–0.7.
- **Threshold \(\alpha\):** experiment with the suggested set {3, 5, 7} (hyperparameters controlling sink selection sensitivity).
- **Subset size:** 0%, 40%, 60%, 80%, 100% to test effectiveness of partial sink suppression.
- **Number of heads/layers calibrated:** test calibrating only specific heads/layers (e.g., top attention sinks) versus all.
- **Attention normalization:** ensure sum of attention remains 1 post-adjustment.

---

# 5. Implementation Steps for Reproduction
1. **Extract Attention Maps:**
   - Use hooks / model internals to extract attention weight matrices \(\mathbf{A}_h^l\) during inference.
2. **Identify Attention Sinks:**
   - Compute per-token attention scores across heads and layers; detect outliers (top percentile or > threshold).
3. **Apply Attention Calibration (ACT):**
   - For each inference sample:
     - Detect sinks.
     - Adjust attention weights for sinks by multiplying with (1 - \(\beta\)).
     - Re-normalize attention distributions as needed.
4. **Modified Forward Pass:**
   - Use the adjusted attention weights in the model inference pipeline to generate output.
   - Implement a custom transformer wrapper or modify attention computations dynamically.
5. **Evaluation:**
   - Run inference with and without ACT.
   - Compute metrics, record improvements.
   - Generate attention maps for analysis & visualization.

---

# 6. Additional Considerations & Open Questions
- **Availability of Attention Maps:**
  - Confirm model code access to intercept internal attention matrices.
- **Sourcing Data and Prompts:**
  - Use the prompts supplied in the paper or replicate prompts from publicly available datasets.
- **Robust Hyperparameter Search:**
  - Conduct ablations over \(\alpha\), \(\beta\), subset sizes, layers, heads, etc.
- **Code modularity:**
  - Separate sink detection, attention adjustment, and inference steps for clarity and iterative testing.

---

# Summary
- The core of the implementation revolves around **extracting attention maps during inference**, **dynamically identifying and suppressing attention sinks**, and then **evaluating the downstream model’s performance**.
- The process is **input-adaptive, training-free**, and hinges on hyperparameter tuning for **thresholds and suppression magnitudes**.
- Extensive visualizations and metrics should be used to understand how attention calibration impacts model behavior.

---

This roadmap provides a detailed, step-by-step blueprint for reimplementing the ACT method as described, setting a firm foundation before proceeding to actual code development.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular system that extracts attention maps during inference, identifies high-attention sinks, and adjusts attention weights dynamically to calibrate the model's focus. We will leverage HuggingFace Transformers library for model loading and inference, extending the model to hook into internal attention matrices. The core component will be an AttentionCalibrator class that processes attention weights to detect sinks based on configurable thresholds and applies multiplicative suppression factors before passing the modified attention to subsequent layers. The system will run inference on each input, perform attention map analysis, calibrate attention weights on the fly, and then produce predictions. The system will also support hyperparameter tuning and visualization for debugging and analysis.",
    "File list": [
        "main.py",
        "attention_calibrator.py",
        "dataset_loader.py",
        "model_wrapper.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_paths: List[str], prompts: dict)\n        +load_data() -> List[dict]\n    }\n    class ModelWrapper {\n        +__init__(model_name: str, device: str)\n        +register_attention_hooks() -> None\n        +compute_attention() -> List[Dict[int, Tensor]]  # For each layer, dict of head index to attention matrix\n        +generate_output(input_ids: Tensor, attention_weights: Optional[List[Dict[int, Tensor]]]=None) -> str\n    }\n    class AttentionCalibrator {\n        +__init__(thresholds: dict, suppress_factor: float, subset_percent: float)\n        +detect_sinks(attention_maps: List[Dict[int, Tensor]]) -> List[Tuple[int, int, float]] # layer, head, token_score\n        +apply_suppression(attention_maps: List[Dict[int, Tensor]], sinks: List[Tuple[int, int, float]]) -> List[Dict[int, Tensor]]\n        +calibrate_attention(attention_maps: List[Dict[int, Tensor]], sinks: List[Tuple[int, int, float]]) -> List[Dict[int, Tensor]]\n    }\n    class Evaluation {\n        +__init__(model: ModelWrapper, dataset: List[dict])\n        +run_evaluation() -> dict\n    }\n\nMain --> DatasetLoader\nMain --> ModelWrapper\nMain --> AttentionCalibrator\nMain --> Evaluation\nDatasetLoader --> +load_data()\nModelWrapper --> +register_attention_hooks()\nModelWrapper --> +compute_attention()\nModelWrapper --> +generate_output()\nAttentionCalibrator --> +detect_sinks()\nAttentionCalibrator --> +apply_suppression()\nAttentionCalibrator --> +calibrate_attention()",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MW as ModelWrapper\n    participant AC as AttentionCalibrator\n    participant E as Evaluation\n    M->>DL: load_data()\n    DL-->>M: dataset list of dicts\n    M->>MW: initialize(model_name, device)\n    MW-->>MW: load and prepare model\n    MW->>MW: register_attention_hooks()\n    loop for each input in dataset\n        M->>MW: generate_attention_maps(input_ids)\n        MW-->>MW: retrieve attention matrices\n        AC->>AC: detect_sinks(attention_maps)\n        AC-->>AC: list of sinks (layer, head, score)\n        AC->>AC: apply_suppression(attention_maps, sinks)\n        AC-->>attention_maps: calibrated attention\n        MW->>output: generate_output(input_ids, attention_maps)\n        output-->>M: prediction\n    end\n    M->>E: evaluate(model, dataset)\n    E-->>M: metrics report\n",
    "Anything UNCLEAR": "Clarification needed on how to interface with internal attention matrices—whether to modify model code or hook external modules. Also, specify if the maximum number of layers and heads are fixed, and whether inference speed should be optimized or can be sacrificed for thorough analysis."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "transformers==4.28.0",
        "torch==1.13.0",
        "numpy==1.21.0",
        "scipy==1.7.3",
        "pandas==1.3.5",
        "matplotlib==3.5.1"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: responsible for loading datasets in specified formats (classification, QA). Implements load_data() returning list of dicts with input prompts and labels. Dependency: needs dataset paths and prompt templates."
        ],
        [
            "model_wrapper.py",
            "Defines ModelWrapper class: initializes the model from HuggingFace Transformers, registers hooks to extract attention weights during inference, provides method compute_attention() to get attention matrices, and generate_output() for inference with optional attention modifications. Dependency: requires model name, device, and access to internal transformers modules."
        ],
        [
            "attention_calibrator.py",
            "Defines AttentionCalibrator class: responsible for detecting high-attention sinks from attention maps, applying suppression factors based on configurable thresholds (\u03b1, \u03b2, subset cutoff), and performing attention calibration. Implements detect_sinks(), apply_suppression(), calibrate_attention() methods. Dependency: consumes attention maps from ModelWrapper, hyperparameters, and outputs adjusted attention weights."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class: runs predictions on datasets using the model wrapper, compares predictions to labels, calculates metrics (accuracy, F1), and supports running ACT modifications during inference. Dependency: needs dataset and model wrapper with calibrated attention."
        ],
        [
            "main.py",
            "Main execution script: orchestrates dataset loading, model initialization, attention hook registration, performs inference on each input with attention calibration, and runs evaluation. Calls classes/methods from dataset_loader.py, model_wrapper.py, attention_calibrator.py, evaluation.py. Dependency: all other modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model_wrapper.py",
        "attention_calibrator.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n  # Not required; no frontend-backend API needed for core reproduction.\n",
    "Shared Knowledge": "Attention map extraction and processing functions are shared between ModelWrapper and AttentionCalibrator. Hyperparameters (\u03b1 for sink detection, \u03b2 for suppression strength, subset size cutoff) are shared configuration variables. Dataset format specifications are consistent across DatasetLoader and Evaluation modules.",
    "Anything UNCLEAR": "Details on whether to modify the internal attention computations directly or implement external attention weight re-normalization during inference. Clarification needed on maximum input length, batch processing support, and optimization tradeoffs."
}

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 3e-5  # Not explicitly specified in the paper, typical for inference-focused finetuning
  batch_size: 32       # Common batch size, adjustable based on hardware resources
  epochs: 3            # Finetuning epochs; optional, as the method is inference-only, so set minimal
  weight_decay: 0.01   # Standard weight decay; optional, not explicitly specified
model:
  name: "Llama-7b"       # Example model, can be replaced with other models as needed
  device: "cuda"         # Run on GPU for performance
  model_path: "path_to_huggingface_model"  # Path or identifier to model weights
attention_calibration:
  alpha: 5               # Attention sink detection threshold (hyperparameter; try {3,5,7})
  suppress_factor: 0.4 # Suppression factor \u03b2 for reducing sink attention (hyperparameter)
  subset_percent: 0.4  # Percentage of top attention tokens considered sinks (e.g., 40%)
  calibrate_layers:
    - 1
    - 2
    - 3
    - 4              # Layers to calibrate; configurable
  calibrate_heads:
    - 0
    - 1
    - 2
    - 3              # Heads to calibrate; configurable
evaluation:
  datasets:
    classification:
      - name: "SST2"
        split: "test"
        metric: "accuracy"
      - name: "AGNews"
        split: "test"
        metric: "accuracy"
    qa:
      - name: "SQuAD"
        split: "dev"
        metric: "EM/F1"
    multi_turn:
      - name: "MT-Bench"
        split: "test"
        metric: "score"
hyperparameters:
  learning_rate: 3e-5
  batch_size: 32
  num_epochs: 3
  max_input_length: 1024
  attention_threshold: 5
  suppression_factor: 0.4
  subset_cutoff: 0.4
```

---

**Note:** Full configuration is available in `planning_config.yaml`
