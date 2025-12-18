# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, step-by-step outline to reproduce the methodology and experiments from the paper "Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs":

---

## I. Overall Objective
Implement PASTA—Post-hoc Attention Steering Approach—on large pre-trained language models (LLMs) such as LLAMA-7B and GPT-J-6B, without model fine-tuning, by:
- Identifying effective attention heads per task via profiling.
- Applying inference-time attention reweighting to emphasize user-specified input spans.
- Evaluating on various tasks (JSON formatting, pronoun changing, BiasBios, CounterFact) using datasets of specified sizes.
- Comparing against baselines (zero-shot, marked prompting).

---

## II. Core Components & Implementation Strategy

### 1. **Identify and Extract Attention Heads (Model Profiling)**
- **Goal:** Determine which heads are most effective for each task.
- **Method:**
  - **Input:** Small labeled datasets (e.g., 1000 samples per task), for training the profiling, plus validation/test sets.
  - **Procedure:**
    - For each sample in the profiling set and for each layer \( l \) and head \( h \):
      - Use the model to generate outputs with *no* attention steering (standard inference).
      - Modify the internal attention scores at layer \( l, h \) by scaling or masking (see below).
      - Evaluate performance on the small validation subset using the relevant task metric (accuracy, ES, PSI, etc.).
      - Record performance metrics for each head.
  - **Selection:**
    - Select top R heads per task based on performance.
    - Aggregate across multiple tasks via intersection or union:
      - **Intersection:** only heads good for all tasks.
      - **Union:** all heads good for at least one task.
    - This process is performed once per model and task set (no retraining, only score evaluation).

### 2. **Attention Score Reweighting ("Post-hoc Attention Steering")**
- **Objective:** During inference, emphasize user-selected input spans.
- **Steps:**
  - **Input:** Text prompt, with user emphasis highlighted (e.g., via markdown `*` or custom markers).
  - **Span Extraction:**
    - Parse the prompt to identify emphasized tokens/spans (via delimiters or style markers—e.g., `*text*` in markdown).
  - **Attention Reweighting:**
    - For each selected head \( (l, h) \in \mathcal{H} \):
      - During each inference step, access the model's raw attention scores \( A^{(l,h)} \).
      - Apply the transformation:
        \[
        \tilde{A}^{(l,h)}_{i,j} = \begin{cases}
        \alpha \times A_{i,j} / C_i, & \text{if } j \notin \text{highlighted tokens} \\
        A_{i,j} / C_i, & \text{if } j \in \text{highlighted tokens} \\
        \end{cases}
        \]
      - Where:
        - \( \alpha \in [0,1) \) (default ~0.01) controls emphasis.
        - \( C_i \) is a normalization factor for token \( i \):
          \[
          C_i = \sum_{j \in \text{highlighted}} A_{i,j} + \alpha \sum_{j \notin \text{highlighted}} A_{i,j}
          \]
    - Followed by renormalization so scores sum to 1.
    - **Implementation detail:** Modify the attention scores **before** softmax, perhaps by intercepting the model's attention computation if API allows; otherwise, implement as a custom forward hook or via a framework that permits attention score customization (e.g., HuggingFace Transformers).

### 3. **Inference Procedure**
- For each task:
  - Use the **profiled** set of heads \( \mathcal{H} \).
  - For each user input, detect emphasized spans.
  - For each attention head \( (l, h) \in \mathcal{H} \):
    - Reweight attention scores as described.
  - Generate output tokens greedily or with beam search.
  - Collect generated output for evaluation.

### 4. **Handling Multiple Heads & Layers**
- Based on profiling:
  - **Single-head steering**:
    - Reweight only heads \( (l,h) \in \mathcal{H} \).
  - **Global approach**:
    - Steer all heads in selected layers or all heads in a specific layer, depending on empirical findings.
- The authors find multi-head selection (top-K heads per task) effective; pick \( K \) according to cross-validation, generally between 25 and 100 heads.

---

## III. Dataset & Experimental Design

### 1. **Datasets & Task Construction**
- **Datasets:**
  - JSON Formatting: 4,996 samples (training + validation/test, each with 1,000).
  - Pronouns Changing, BiasBios, CounterFact: 5,000 samples each for training, validation, testing.
- **Input Structure:**
  - For each task, prepare prompt templates from the paper.
  - For examples, append instruction, context, and user emphasis markers as needed.
- **Task-specific prompt templates:**
  - For zero-shot, just instruction + input.
  - For marked prompting, add emphasis markers (`*` etc.).
  - For PASTA, explicitly identify emphasizes input spans (e.g., the part to steer attention).

### 2. **Evaluation Metrics & Procedures**
- Use the metrics described:
  - JSON Format: Format accuracy + prediction accuracy.
  - Pronouns Changing: Accuracy + all-changed accuracy.
  - BiasBios: Classification accuracy.
  - CounterFact: Effectiveness score, paraphrase score.
  - Fluency: Bigrams, trigrams entropy, remove poor fluency generations.
- Collect performance with and without steering (zero-shot baseline, marked prompting).

### 3. **Hyperparameters & Settings**
- **Number of heads \( |\mathcal{H}| \):** Based on profiling results; typically 30–100 heads.
- **Scaling coefficient \( \alpha \):** Fixed at 0.01; test sensitivity as in paper.
- **Number of samples for profiling per task:** 1000 samples for small validation; 200 samples to test the effect of profile size.
- **Reweighting approach:**
  - Multiplicative attention reweighting as per formulation.
  - Implement during inference using model hooks/package APIs.
- **Profiling:**
  - Performance evaluation per head via small task dataset.
  - Select heads based on top performance (simple accuracy or F1, depending on task metric).

### 4. **Experimental Procedure Summary**
- Profiling phase:
  - For each task, evaluate individual heads on small validation set.
  - Select top heads based on task performance.
- Steering phase:
  - During inference, apply the attention reweighting only on selected heads.
- Evaluation phase:
  - Generate with steering and compare against baseline prompting.
  - Record scores, fluency, and consistency.
  - Analyze per-layer/head performance variance.

---

## IV. Implementation Notes & Potential Challenges
- **Access to attention scores:** Ensure the chosen frameworks (e.g., HuggingFace) support extracting and modifying attention scores during inference—possibly via `hooks`.
- **Token span detection:** Prompt parsing must reliably identify user emphasized tokens.
- **Efficiency:** Profiling can be parallelized; attention modification should be optimized (batched if possible).
- **Model Compatibility:**
  - LLAMA-7B and GPT-J-6B are both compatible with HuggingFace Transformers with custom attention hooks.
  - For GPT-J, use the appropriate library (e.g., EleutherAI) with support for attention score adjustment.
- **No fine-tuning:** All methods are inference-time only, leveraging attention score reweighting.

---

## V. Summary of Key Steps
1. **Profile** all attention heads per task using small datasets; select top-heads.
2. **Prepare** prompt templates with emphasized spans.
3. **Implement** attention score modification during inference:
   - Identify spans from prompts.
   - Reweight attention scores at selected heads.
4. **Generate** outputs under steering.
5. **Evaluate** performance using the prescribed metrics.
6. **Compare** with baselines to validate improvements.

---

## VI. Final notes
- Carefully document the selected heads and hyperparameters.
- Use cross-validation or validation datasets to choose the best \( |\mathcal{H}| \) and hyperparameters.
- Consider robustness measures when varying emphasis span markers or reweighting coefficient \( \alpha \).

---

This roadmap provides a detailed blueprint to develop, test, and evaluate PASTA on the specified models and tasks, facilitating later implementation.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Design a modular system using HuggingFace Transformers to load pre-trained models (LLAMA-7B, GPT-J-6B), perform profiling to select effective attention heads, and apply inference-time attention reweighting based on user-specified emphasized spans. The system will include data loaders for experiments, a core model wrapper that supports hooking into attention scores, and a steering module that modifies attention during inference without fine-tuning. The evaluation pipeline will handle multiple tasks and metrics as specified in the paper, with hyperparameter tuning based on small validation samples.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model_wrapper.py",
        "attention_steering.py",
        "profile.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(task_name: str, data_dir: str, split: str)\n        +load_dataset() -> Dataset\n    }\n    class ModelWrapper {\n        +__init__(model_name: str, model_path: str)\n        +register_attention_hook(heads: List[Tuple[int, int]]) -> None\n        +generate(input_ids: List[int], emphasis_spans: List[int], alpha: float=0.01) -> str\n        +get_attention_scores(layer: int, head: int, inputs: Tensor) -> Tensor\n    }\n    class AttentionSteering {\n        +__init__(model_wrapper: ModelWrapper, alpha: float=0.01)\n        +apply_reweighting(attention_scores: Tensor, emphasis_spans: List[int]) -> Tensor\n    }\n    class Profiler {\n        +__init__(model_wrapper: ModelWrapper, dataset: Dataset, top_heads_count: int)\n        +profile_heads() -> List[Tuple[int, int]]\n        +select_top_heads() -> List[Tuple[int, int]]\n    }\n    class Evaluation {\n        +__init__(model: ModelWrapper, dataset: Dataset, task_name: str)\n        +evaluate() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> ModelWrapper\n    Main --> Profiler\n    Main --> AttentionSteering\n    Main --> Evaluation\n    Profiler --> ModelWrapper\n    ModelWrapper --> AttentionSteering\n    ModelWrapper --> DatasetLoader\n    Evaluation --> DatasetLoader\n    Evaluation --> ModelWrapper",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MW as ModelWrapper\n    participant P as Profiler\n    participant A as AttentionSteering\n    participant E as Evaluation\n    M->>DL: load_dataset(task_name, data_dir, split)\n    DL-->>M: dataset object\n    M->>MW: initialize model(model_name, model_path)\n    MW-->>M: model instance\n    M->>P: profile_heads(dataset, top_heads_count)\n    P-->>M: selected_heads\n    M->>MW: register_attention_hook(selected_heads)\n    M->>A: initialize(model_wrapper=MW, alpha=0.01)\n    alt profiling\n        M->>MW: generate(input_ids, emphasis_spans, alpha)\n        MW-->>M: generated text\n    end\n    alt evaluation\n        M->>E: evaluate(dataset, task_name)\n        E-->>M: metrics dictionary\n    end\n    Note right of M: Loop over tasks, apply attention reweighting, and record metrics.\n",
    "Anything UNCLEAR": "Clarification needed on the exact internal API support for modifying attention scores during inference (e.g., whether custom hooks are feasible), and on the preferred span extraction method from styled prompts (e.g., markdown syntax). Also, confirmation on the dataset format and how emphasis spans are specified in practice." 
}

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "transformers==4.26.0",
        "torch==1.10.0",
        "datasets==1.18.0",
        "numpy==1.21.0",
        "scipy==1.7.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class responsible for loading, splitting, and preprocessing the datasets. It must handle dataset formats, tokenize inputs, and prepare emphasis span annotations. It is a dependency for the profiling, training, and evaluation modules."
        ],
        [
            "model_wrapper.py",
            "Implements the ModelWrapper class which loads the pre-trained model (LLAMA, GPT-J), registers hooks for attention extraction, and provides generate() method that applies inference-time attention reweighting. It depends on transformers library and interacts with attention_steering.py."
        ],
        [
            "attention_steering.py",
            "Defines AttentionSteering class that reweights attention scores based on emphasis spans and the specific heads selected from profiling. It manipulates attention scores during inference, acting as a hook or wrapper around model's forward pass."
        ],
        [
            "profile.py",
            "Implements the Profiler class that performs head performance evaluation across small validation sets. It calls model_wrapper.py to extract attention scores at each head, runs inference with different heads/heads subsets, and selects top heads based on metric evaluation. This profiling is used to determine which heads to steer during main experiments."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class to run generated outputs on test datasets, compute metrics (accuracy, ES, PSI, fluency). It depends on dataset_loader.py and model_wrapper.py for data and inference execution."
        ],
        [
            "main.py",
            "Acts as the orchestrator. Reads configs, initializes dataset loader, model wrapper, profiling, and evaluation modules. It performs profiling, sets up steering, runs inference with attention reweighting, performs evaluation, and logs results. It manages dependency flow and task execution order."
        ],
        [
            "utils.py",
            "Includes common utility functions: e.g., tokenization helpers, span extraction from styled prompts, normalization functions, etc. Shared by main, profiling, and inference modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",       # Load datasets, handle splits, tokenization, span annotation. Dependency for profiling, inference, evaluation.
        "model_wrapper.py",        # Load models, register hooks, generate text with optional attention reweighting.
        "attention_steering.py",   # Core logic for modifying attention scores during inference based on user emphasis.
        "profile.py",              # Profile attention heads, evaluate performance, select top heads.
        "evaluation.py",           # Compute metrics like accuracy, fluency, etc., on generated outputs.
        "main.py",                 # Overall flow: load data, perform profiling, set up steering, generate, evaluate, and log results.
        "utils.py"                 # Helper functions shared across modules.
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "All modules depend on standard APIs from transformers for model inference, and datasets for data loading. Head selection in profiling depends on a consistent metric evaluation framework. Span extraction from styled prompts must be standardized for robustness.",
    "Anything UNCLEAR": "Clarification needed on exact attention score access via hooks in the chosen transformer models (e.g., LLAMA/ GPT-J), and whether the interface supports manipulating attention scores at runtime. Also, how emphasis spans are annotated in input prompts (e.g., markdown style markers) needs to be confirmed for robust span extraction."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
model:
  name: "LLaMA-7B"
  model_path: "path/to/llama-7b"

dataset:
  name: "BiasBios"
  data_dir: "datasets/BiasBios"
  split: ["train", "validation", "test"]
  train_size: 1000
  val_size: 1000
  test_size: 5000

profiling:
  top_heads_count: 50 # number of heads to select based on profiling
  profile_samples: 1000 # number of samples for profiling

attention_steering:
  alpha: 0.01  # attention reweighting scale coefficient
  heads_selection_strategy: "top-per-task"  # options: "top-per-task", "union", "intersection"

evaluation:
  metrics:
    format_accuracy: true
    prediction_accuracy: true
    pronoun_accuracy: true
    fluency: true
    counterfact_effectiveness: true

training:
  batch_size: 16  # inference does not require training, but placeholder if needed later
  learning_rate: 5e-6  # placeholder; no fine tuning in described method
  epochs: 0  # no training; method is inference-time only
```

---

**Note:** Full configuration is available in `planning_config.yaml`
