# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here's a comprehensive, step-by-step plan to reproduce the core methodology, experiments, and evaluation of the paper *"TELL YOUR MODEL WHERE TO ATTEND: POST-HOC ATTENTION STEERING FOR LLMS"*. This plan will ensure fidelity to the described approach, provide guidance on dataset preparation and hyperparameters, and segment tasks for implementation.

---

## 1. **Understanding and Implementing the Core Methodology (PASTA)**

### 1.1. **Post-hoc Attention Steering (Inference-time operation)**

- **Objective:** Given a pretrained large language model (LLM), at inference time, modify its attention scores to emphasize user-specified input segments.  
- **Key Steps:**
  - **Identify User-Specified Input Regions (`𝒢`):** Mark spans of tokens in the input that a user wants the model to focus on (e.g., by adding emphasis markers \* or **).
  - **Compute Attention Scores (`A^{(l,h)}`):** For each layer `l` and head `h`, extract the raw attention scores matrix during inference.
  - **Apply Attention Steering (`T(A^{(l,h)})`):**
    - For each token pair `(i,j)`:
      - If `j` is in `𝒢` (tokens highlighted by user), **scale up** the attention scores by `1/α` (with `α=0.01` fixed for stability).
      - Else, **scale down** the attention scores by `α`.
    - Compute normalization constant `C_i` per head:
      \[
      C_i = \sum_{j \in 𝒢} A_{ij} + \alpha \sum_{j \notin 𝒢} A_{ij}
      \]
    - Set  
      \[
      [\mathcal{T}(A)]_{ij} = \begin{cases}
      \frac{\alpha A_{ij}}{C_i} & \text{if } j \notin 𝒢 \\
      \frac{A_{ij}}{C_i} & \text{if } j \in 𝒢
      \end{cases}
      \]
  - **Recompute Contextual Representation:** Use the reweighted attention scores to produce the modified attention output for each head, layer, and the subsequent decoder components.
  - **Generate the final output** with the model using these adjusted attention scores (without retraining).

### 1.2. **Selecting Effective Attention Heads (Model Profiling)**

- **Objective:** Identify which attention heads, when steered, produce the most benefit for downstream tasks.
- **Procedure:**
  - **Profiling on small datasets**: For a small subset of training data per task (`~1000 samples`), evaluate each attention head's steering performance:
    - For each head `(l,h)`, modify attention scores as above and measure task-specific performance (e.g., accuracy, JSON validity, pronoun correctness).
  - **Ranking Heads:** Based on performance, rank all heads for each task.
  - **Head Selection:**
    - Use strategies such as:
      - **Top-`k`** heads per task.
      - **Intersection** of top heads across multiple tasks.
      - **Union** of top heads for broader coverage.
    - Typically, select `k` in the range `[300, 500]` heads for LLAMA-7B, based on ablation results.

---

## 2. **Experimental Setup**

### 2.1. **Datasets for Evaluation**
- Reproduce the four main tasks:
  1. **JSON Formatting**
     - Use biographical datasets (e.g., BiasBios) with annotated instructions for output format.
     - Generate prompts that instruct the model to produce JSON-formatted responses.
  2. **Pronouns Changing**
     - Use datasets with biographical texts; prompts include instructions to change pronouns (`she`/`he`) to `they`.
  3. **BiasBios** (professional biographical data)
     - Input: Biographical paragraph.
     - Output: Predicted occupation.
  4. **CounterFact**
     - Test for knowledge updating/conflicting facts: include old vs new facts in prompts.
- **Dataset size:** 
  - Small profiling set: 1,000 samples per task.
  - Test sets: ~5000 per task to evaluate performance.

### 2.2. **Prompt Design & Markers**
- Use emphasis markers (e.g., markdown `**` or `__`) around user-highlighted parts during prompt construction.
- For task-specific prompts, follow the configurations from Appendices, ensuring instructions clearly highlight which part of input is emphasized.
- Example for JSON formatting:
  ```
  {instruction} {highlighted span with emphasis markers}
  ```
- Example for pronoun change:
  ```
  {instruction} {highlighted span}
  ```

### 2.3. **Hyperparameters**
- **α (attention scaling factor):** fix at `0.01` as per paper.
- **Number of heads `k` to steer:** Use cross-validation on the profiling dataset; typical values around `[300, 400, 500]`.
- **Number of heads per task profile:** e.g., `400` heads for LLAMA-7B, selected based on the maximum task performance.
- **Profiling samples:** 1000 samples per task.
- **Prompting strategy:**
  - Zero-shot (no example).
  - Few-shot (3 demonstration examples).

### 2.4. **Model Execution**
- Use `HuggingFace Transformers` or `OpenAI GPT`-like APIs for inference.
- For each prompt:
  - Tokenize input (including emphasis markers).
  - Extract underlying attention matrices at every layer/head.
  - Apply the post-hoc attention reweighting.
  - Generate output greedily or with beam search.
- Ensure attention scores are modified **during inference** before the forward pass proceeds.

---

## 3. **Evaluation Metrics and Analysis**

### 3.1. **Task-specific Metrics**
- **JSON Formatting:** Accuracy of valid JSON and value correctness.
- **Pronouns Changing:** Exact match accuracy, "all changed" accuracy.
- **BiasBios & CounterFact:** Classification accuracy, efficacy score (`ES`), paraphrase robustness.
- **Fluency & Consistency:**
  - Use the **entropies** of n-gram distributions.
  - Filter out low-fluency samples (<3.0).

### 3.2. **Sensitivity & Ablation Studies**
- Vary:
  - Number of heads steered (`k`).
  - Whether to steer all heads, layer-wise, or specific heads.
  - The reweighting coefficient `α`.
- Record performance variance and stability.
- Measure** sensitivity to prompt phrasing** by testing rephrased prompts, and compute the variance in output quality and task performance.

---

## 4. **Implementation Phases**

**Phase 1: Setup & Baseline**
- Load pretrained models (`LLAMA-7B`, `GPT-J`, `Vicuna-7B`) with accessible attention score extraction.
- Implement plain inference with prompt concatenation.

**Phase 2: Attention Score Modification**
- Implement function to extract attention scores `A^{(l,h)}`.
- Implement the `T( A )` operation to reweight attention scores according to selected highlighted tokens.
- Enable passing modified scores back into the forward pass for decoding.

**Phase 3: Model Profiling**
- Run small datasets to evaluate each attention head's influence.
- Generate rankings and select top heads per task.

**Phase 4: Attention Steering & Testing**
- Apply steering to selected heads during inference.
- Evaluate on validation data.
- Conduct ablation with different `k`, speakers of heads, and `α`.

**Phase 5: Full Evaluation**
- Run full test suite (JSON, pronouns, bias, counterfactuals).
- Record and analyze results, compare with baselines (`zero-shot`, `marked prompting`).

---

## 5. **Open Questions & Clarifications**
- **Attention Score Extraction:** Confirm extraction method per model (e.g., transformers library provides hooks for attention matrices).
- **Compatibility:** Ensure the model supports attention score manipulation (some models may have non-standard implementations).
- **Reweighting Implementation:** Should reweight attention scores before softmax? The paper suggests directly scaling and renormalizing attention logits.
- **Head Selection:** Clarify whether specific head indices are known from profiling or need to be computed anew.
- **Task evaluation details:** For full reproducibility, ensure dataset splits (train/test), prompt templates, and evaluation scripts align with the paper.

---

## Final Summary:
- The core reproducibility hinges on the **attention reweighting operation** during inference, guided by user emphasis markers.
- **Model profiling** is crucial to select which heads to steer; this must be done on small datasets with performance metrics relevant per task.
- Several **hyperparameters** (e.g., alpha, number of heads `k`) are fixed or tuned via cross-validation.
- Detailed **prompt templates** and **dataset handling** are essential, with emphasis markers integrated into input prompts.
- A thorough **evaluation** includes accuracy metrics, fluency, and variance analysis to verify robustness.

This roadmap provides a solid foundation to implement, tune, and later extend the experiments in the paper.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Develop a modular pipeline that loads pretrained LLMs with accessible attention matrices, identifies effective attention heads via profiling, and applies post-hoc score reweighting during inference to steer attention based on user-marked highlights. Utilize the Hugging Face Transformers library for model interoperability, hooks for attention extraction, and implement a lightweight reweighting function that modifies scores before softmax. Incorporate prompt templates with emphasis markers and a command-line interface for user input, prompt selection, and task-specific configurations.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "attention_module.py",
        "profiling.py",
        "steering.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
        +parse_args()
    }
    class DatasetLoader {
        +__init__(data_paths: dict)
        +load_dataset(task_name: str) -> Dataset
    }
    class Model {
        +__init__(model_name: str, device: str)
        +load()
        +extract_attention() -> List[Tensor]
        +forward(input_ids: Tensor, attention_mask: Tensor, modified_attentions: Optional[List[Tensor]] = None) -> Tensor
    }
    class AttentionReweighter {
        +__init__(head_indices: List[Tuple[int, int]], alpha: float)
        +apply_masking(attention_scores: List[Tensor], highlighted_tokens: List[int]) -> List[Tensor]
    }
    class ProfileAnalyzer {
        +__init__(model: Model, profile_dataset: Dataset, top_k: int)
        +profile_heads() -> List[Tuple[int, int]]
        +save_profile(head_indices: List[Tuple[int, int]])
    }
    class PromptBuilder {
        +__init__(prompt_template: str, emphasis_marker: str = "**")
        +create_prompt(input_text: str, highlighted_spans: List[str]) -> str
    }
    class InferenceEngine {
        +__init__(model: Model, attention_weighter: AttentionReweighter, profile_heads: List[Tuple[int,int]], max_heads: int)
        +prepare_attention(h_heads: List[Tuple[int,int]]) -> None
        +generate(prompt: str) -> str
    }
    class Evaluation {
        +__init__(model: Model, dataset: Dataset, task: str)
        +evaluate_task() -> dict
        +compute_metrics() -> dict
    }
    Main --> DatasetLoader
    Main --> Model
    Main --> ProfileAnalyzer
    Main --> PromptBuilder
    Main --> InferenceEngine
    Main --> Evaluation
    ProfileAnalyzer --> Model : profile_heads()
    AttentionReweighter --> List[Tensor] : apply_masking()
    InferenceEngine --> Model : forward()
    Main --> Utils : load_config()
",
    "Program call flow": "
sequenceDiagram
    participant C as Main
    participant DL as DatasetLoader
    participant M as Model
    participant P as ProfileAnalyzer
    participant A as AttentionReweighter
    participant PBT as PromptBuilder
    participant IE as InferenceEngine
    participant E as Evaluation

    C->>C: parse_args()
    C->>DL: load_dataset(task_name)
    DL-->>C: dataset
    C->>M: load()
    M-->>C: model_instance
    C->>P: profile_heads(profile_dataset, top_k)
    P-->>C: head_indices
    C->>A: initialize(head_indices, alpha=0.01)
    C->>PBT: create_prompt(template, highlighted_spans)
    PBT-->>C: prompt_string
    C->>IE: generate(prompt_string)
    IE-->>C: generated_text
    C->>E: evaluate_task()
    E-->>C: metrics
    Note over C: Repeat for all tasks and compare results

    note left of IE: During each generation, extract attention scores, apply apply_masking, and set modified attentions at each inference step before decoding.
    Note right of E: Metrics include accuracy, fluency, consistency, and robustness as per task specifics.
",
    "Anything UNCLEAR": "Clarify if attention scores are accessible directly via transformers hooks for all pretrained models. Need confirmation on the exact format and interface for attention extraction and whether modifications during inference are supported natively or via custom forward hooks. Also, request for example dataset format and templating standards to align prompt building."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.0",
        "transformers==4.26.0",
        "datasets==2.7.0",
        "numpy==1.21.0",
        "scipy==1.7.3",
        "mermaid==8.0.0 (for diagram representation if needed)"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Main entry point coordinating argument parsing, configuration loading, dataset initialization, profiling, inference, and evaluation. It depends on all other modules to orchestrate the workflow."
        ],
        [
            "dataset_loader.py",
            "Defines DatasetLoader class responsible for loading datasets for each task, parsing dataset files, preparing train/test splits, and providing datasets compatible with Hugging Face datasets or custom formats."
        ],
        [
            "model.py",
            "Defines Model class that loads pretrained models (LLAMA, GPT-J, Vicuna) using transformers. Implements methods for extracting attention scores, modifying attention weights during inference, and managing device placements."
        ],
        [
            "attention_module.py",
            "Provides functions or classes to hook into model's attention scores. Implements extraction of attention logs for each head/layer and applying reweighting logic based on highlighted tokens."
        ],
        [
            "profiling.py",
            "Contains ProfileAnalyzer class that profiles attention heads' contributions on small datasets, ranks the heads, and saves profiles for fast retrieval. It depends on model.py for head access and data_loader for dataset samples."
        ],
        [
            "steering.py",
            "Implements AttentionReweighter class which applies the post-hoc attention scaling as per the highlighted tokens, per head, layer during inference. Depends on attention_module.py for modifying attention scores."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class to compute metrics (accuracy, fluency, consistency) for each task, including task-specific evaluation functions. Relies on inference outputs and possibly external scripts or libraries."
        ],
        [
            "utils.py",
            "Provides utility functions such as config parsing, prompt template rendering, argument validation, and helper functions for file management and logging."
        ]
    ],
    "Task list": [
        "utils.py",
        "dataset_loader.py",
        "model.py",
        "attention_module.py",
        "profiling.py",
        "steering.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n- POST /load_dataset: load dataset configuration.\n- POST /profile_heads: trigger head profiling.\n- POST /generate: generate model output with attention reweighting.\n- GET /metrics: retrieve evaluation metrics.\n- POST /prepare_prompt: construct prompts with emphasis markers.\n- GET /model_info: retrieve model configuration and available attention layers/heads.",
    "Shared Knowledge": "Uniform prompt templates with emphasis markers; attention reweighting functions; dataset parsing and splitting logic; head profiling info. All modules depend on consistent data formats and config files.",
    "Anything UNCLEAR": "Clarify if the models' attention scores are accessible via transformers hooks in current versions; details on dataset format and the preferred interface for passing highlighted spans; whether to support multiple hardware configurations for large-scale profiling and inference."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
# Training hyperparameters: not explicitly specified in the paper; set defaults based on common practice.
training:
  learning_rate: 1e-5  # Typical for fine-tuning large language models
  batch_size: 16     # Standard batch size suitable for GPU memory constraints
  epochs: 3          # Few epochs for fine-tuning or profiling, since the method is inference-time
  max_sequence_length: 512  # Supporting typical prompt lengths
  inference_batch_size: 8   # For evaluation, small batch inference
  alpha: 0.01        # Attention score reweighting coefficient fixed as per paper
  num_profiling_samples: 1000  # Number of samples used during head profiling
  top_k_heads: 400   # Number of attention heads selected for steering after profiling
  model_name: llama-7b  # Default model for experiments, can be overridden
  device: cuda       # Use GPU for profiling and inference; fallback to cpu if unavailable

datasets:
  count_ratio: 1.0    # Use full datasets during evaluation; small subset for profiling
  dataset_paths:  # Placeholder paths for datasets (user to specify actual paths)
    bias_bios: path/to/biasbios_dataset
    counterfact: path/to/counterfact_dataset
    json_format: path/to/json_dataset
    pronouns_changing: path/to/pronouns_dataset

prompts:
  json_format_template: |
    {instruction} {highlighted_spans}
  pronouns_change_template: |
    {instruction} {highlighted_spans}

evaluation:
  metrics:
    json_format:
      format_accuracy: true
      prediction_accuracy: true
    pronouns_changing:
      accuracy: true
      all_changed_accuracy: true
    bias_bios:
      classification_accuracy: true
    counterfact:
      efficacy_score: true
      paraphrase_score: true
    fluency:
      min_entropy: 3.0
  evaluation_metrics: [accuracy, fluency, consistency]
```

---

**Note:** Full configuration is available in `planning_config.yaml`
