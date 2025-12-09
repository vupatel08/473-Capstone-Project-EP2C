# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, detailed plan outlining the critical methodological and experimental aspects necessary for reproducing the "Make Your LLM Fully Utilize the Context (IN2)" approach, based solely on the information extracted from the provided excerpt of the paper. This plan will serve as a robust foundation for subsequent implementation.

---

## 1. Core Methodology Outline

### a. **Objective & Hypothesis**
- The main goal is to enhance the long-context utilization capabilities of large language models (LLMs) by explicitly training them on **information-intensive tasks** that emphasize understanding, integration, and reasoning over extended contexts.
- The hypothesis is that insufficient supervision during training causes "lost-in-the-middle" problems, which can be mitigated by **explicit supervision** involving **complex multi-segment questions** requiring information from various parts of long texts.

### b. **IN2 Training Strategy**
- IN2 (Information-INtensive) training is **data-driven**.
- It involves creating a synthetic, **long-context question-answer dataset** where answers:
  1. Require **fine-grained awareness** within a short segment (~128 tokens).
  2. Involve **integration** and **reasoning** across multiple short segments within a very long context (up to 32K tokens).

### c. **Dataset Construction**
- Long contexts are **composed from short segments** randomly or strategically placed within a very long original text (ranges from 4K to 32K tokens).
- Each short segment (e.g., 128 tokens) is associated with specific question-answer pairs designed to:
  - Emphasize local information recognition.
  - Require the model to **integrate multiple segments** for multi-hop reasoning.
- The dataset includes:
  - 1.1M samples for **fine-grained awareness**.
  - 300K samples for **information integration and reasoning**.
  - 150K for **short-context QA**.
  - 200K general instruction data for auxiliary training.

### d. **Synthetic Long-Context QA Data Generation**
- Generate a dataset by:
  1. Extracting short segments (~128 tokens) from large corpus (e.g., from scientific literature, Wikipedia, or other extensive texts).
  2. Randomly or strategically placing these segments into an extended context (up to 32K tokens).
  3. Creating QA pairs per segment:
     - **Type 1**: Answer based solely on one short segment.
     - **Type 2**: Answer requiring **multiple segments** (multi-hop).
     - **Type 3**: Other question types emphasizing integration.
  4. Use GPT-4-Turbo with prompts to automatically generate question-answer pairs, or craft them via heuristic/template-based methods.

### e. **Model Fine-tuning Protocol**
- Fine-tune an existing open-source instruction-tuned LLM (e.g., Mistral-7B-instruct-v0.2, GPT-4-Turbo) with the assembled dataset.
- Fine-tuning parameters:
  - Batch size: 128.
  - Schedule: Single epoch (~14K steps, or adjusted for dataset size).
  - Learning rate: Cosine decay, max LR ~1e-6.
  - Hardware: Distributed training on multiple GPUs (e.g., 16 nodes of A100 80G), with full sharding and CPU offloading.
- The model should be instructed explicitly to **maximize use of long contexts**, highlighting the importance of context-aware question-answering (e.g., explicitly prompt the model via instruction or system prompt).

### f. **Supervisions & Instructions in Data**
- Use explicit instructions in GPT prompts when generating questions (e.g., emphasizing multi-segment reasoning).
- Format data with:
  - Long contexts constructed from multiple segments.
  - QA pairs that demand multi-hop inference.
  - Clear demarcation of segments and context boundaries.
- Maintain template consistency (see figure 2 in the paper) with explicit mention of which segments are relevant for the QA.

### g. **Mitigating Bias & Length Bias**
- Evenly distribute long-context lengths (from 4K up to 32K tokens) during training to prevent length bias.
- Retain ~10% of original short texts to preserve basic short-context performance.
- Use special techniques like sliding window (SW) during training to evaluate and improve robustness across varied context lengths—possibly with different strategies (e.g., SW only in IN2 training, SW in both pre-training and IN2, or without SW).

---

## 2. Experimental Setup & Evaluation

### a. **Datasets Required**
- **Synthetic Long-Context QA Dataset:**
  - Derived from scientific articles, Wikipedia, or similar large corpus.
  - Each sample:
    - Long context (4K–32K tokens) assembled from multiple short segments (~128 tokens).
    - QA pairs requiring:
      - Local awareness (single segment).
      - Multi-segment reasoning (multi-hop).
  - Automated via prompts to GPT-4-Turbo or heuristic methods.
- **Instruction-Tuning Data:**
  - Open-source datasets like OpenOrca.
  - General instruction data (~9%), short QA (~9%), etc.

### b. **Model Fine-tuning Details**
- Model: Open instruction-tuned large language models (e.g., Mistral-7B, GPT-4-Turbo).
- Training hyperparameters:
  - Batch size: 128.
  - Epoch: 1 (with ~14K steps).
  - Learning rate decay: Cosine schedule, max LR 1e-6, warm-up 3%.
  - Hardware: Multi-node A100 80G, full sharding, CPU offload.
- Data handling:
  - For long contexts: generate datasets with lengths: 4K, 8K, 16K, 32K tokens.
  - Use sliding window strategies during evaluation to assess robustness at up to 64K tokens.
- RoPE (rotary positional embedding) base: 
  - Test multiple values (from 1e6 to 1e8) to evaluate impact on long context performance.

### c. **Evaluation Metrics & Tasks**
- **Probing / Retrieval Tasks**:
  - Document sentence retrieval (bi-directional), code function retrieval, database entity retrieval, structured data retrieval.
  - Measure: Accuracy, relative position performance (Figure 1, 4, 9).
- **Long-Context QA / Practical Tasks**:
  - NarrativeQA, GPT4-Turbo evaluation, real-world long-task benchmarks.
  - Metrics: F1 scores, accuracy (e.g., on NarrativeQA 23.6 F1 as in paper), performance on tasks requiring up to 32K tokens.
- **Few-Shot & Zero-Shot Tasks**:
  - Standard benchmarks like MMLU, BoolQ, Race-H, etc., to confirm short-context performance.
- **Robustness & Scalability**:
  - Sliding window evaluation at extended sequence lengths (e.g., 64K tokens, Figure 9).
  - Record performance drop or stabilization as context length increases.

### d. **Ablations & Hyperparameter Studies**
- Vary training data size (from 1% to 100%) and analyze impact (Table 6).
- Vary RoPE base during IN2 training (Table 5).
- Test different sliding window strategies (Figure 8).
- Evaluate at different context window extensions (4K, 8K, 16K, 32K, 64K).
- Examine effect of dataset length distribution (from 4K to 32K tokens) on performance.

### e. **Additional Considerations**
- Maintain strict separation between dataset generation (GPT prompts, heuristics) and fine-tuning code.
- Confirm model outputs are conditioned explicitly on long contexts and multiple segments.
- Document the detailed prompt templates, data formatting conventions, and evaluation scripts.

---

## 3. Implementation Details & Open Questions
- **Data Generation**:
  - How exactly to craft QA pairs that require multi-segment reasoning? (Template prompts or heuristics needed).
  - How to select and assemble segments (strategy for random placement, strategic placement, or stratified sampling)?
- **Prompt Engineering**:
  - Precise instructions to GPT-4 Turbo for synthetic data generation—how to emphasize multi-hop reasoning?
  - How to format contexts for training (e.g., clear delimiters of segments)?
- **Evaluation Protocols**:
  - How to measure the “lost-in-the-middle” problem? (As in Figures 1, 4, 9).
  - How to systematically evaluate long-context retrieval and reasoning at different lengths?
- **Reproducibility & Hyperparameters**:
  - Exact hyperparameter tuning schedules.
  - Handling of randomness: seed control, dataset shuffling, and sampling strategies.
- **Ethical & Licensing Considerations**:
  - Dataset licensing from sources.
  - Ensuring safety & bias mitigation (based on the guidelines).

---

## Summary Roadmap
1. **Data Construction**
   - Collect large corpus.
   - Generate QA pairs via GPT-4 with prompts emphasizing multi-seg reasoning.
   - Assemble synthetic long contexts with specified length distributions.
2. **Model Fine-tuning**
   - Use instruction-tuned models.
   - Configure hyperparameters as specified.
   - Implement sliding window enhancements.
   - Experiment with RoPE bases.
3. **Evaluation**
   - Probing long-context retrieval tasks.
   - Real-world long document QA.
   - Short-context benchmarks.
   - Long-context scaling (up to 64K tokens).
4. **Ablations & Variations**
   - Dataset size effect.
   - Context length impact.
   - Sliding window strategies.
   - Long-context capability analysis.

---

This roadmap captures key design choices, dataset creation strategies, experimental setups, and hyperparameter considerations necessary for faithful reproduction of the described methodologies. It also highlights aspects needing clarification or detailed implementation prompts when operationalizing later.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular system with clear separation between dataset creation, model fine-tuning, and evaluation. The core components will include dataset generation scripts that use GPT-4 for synthetic long-context QA pairs, a training pipeline utilizing Hugging Face Transformers and PEFT for instruction tuning, and evaluation scripts for probing and scaling assessments. The system will automate the assembly of long contexts with multiple segments, customize hyperparameters (learning rate, RoPE base, context length), and support sliding window inference for very long sequences, leveraging PyTorch and DeepSpeed/Accelerate for efficiency.",
    "File list": [
        "app.py",  
        "dataset_generator.py", 
        "model.py",  
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "config.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class App {
        +__init__(config: dict)
        +main()
    }
    class DatasetGenerator {
        +__init__(config: dict)
        +generate_dataset() -> List[Dict]
        +save_dataset(path: str) -> None
        +load_dataset(path: str) -> List[Dict]
    }
    class Model {
        +__init__(model_name: str, params: dict)
        +load_pretrained() -> None
        +train(dataset: List[Dict], training_args: dict) -> None
        +save_model(path: str) -> None
        +load_model(path: str) -> None
        +forward(input_ids: Tensor, attention_mask: Tensor) -> Tensor
    }
    class Trainer {
        +__init__(model: Model, dataset: List[Dict], training_args: dict)
        +train() -> None
        +evaluate(eval_dataset: List[Dict]) -> dict
    }
    class Evaluation {
        +__init__(model: Model)
        +probe_retrieval_tasks() -> dict
        +scale_long_context(model: Model, context_lengths: List[int]) -> dict
        +compute_metrics(on_tasks: List[str]) -> dict
    }
    class Config {
        +dataset_size: int
        +context_lengths: List[int]
        +learning_rate: float
        +batch_size: int
        +epochs: int
        +model_name: str
        +use_sliding_window: bool
        +sliding_window_size: int
        +rope_base: int
        +output_dir: str
    }
    App --> DatasetGenerator
    App --> Model
    App --> Trainer
    App --> Evaluation
    DatasetGenerator --> utils.py: generate_and_prompt
    Model --> utils.py: load_pretrained, save, forward
    Trainer --> Model
    Evaluation --> Model
    Evaluation --> utils.py: process_retrieval, compute_metrics
    Config <|-- App
    Config <|-- DatasetGenerator
    Config <|-- Trainer
    Config <|-- Evaluation
    Trainer --> utils.py: training_loop
    Evaluation --> utils.py: probe_tasks, scale_evaluate
",
    "Program call flow": "
sequenceDiagram
    participant C as Config
    participant D as DatasetGenerator
    participant M as Model
    participant T as Trainer
    participant E as Evaluation
    participant U as utils.py
    App->>C: load configuration
    App->>D: generate_dataset()
    D-->>App: dataset list
    App->>M: load_pretrained()
    M-->>App: model loaded
    App->>T: initialize training with dataset and training args
    T->>utils.py: training_loop()
    utils.py-->>T: training completed
    App->>M: load trained model
    M-->>App: model loaded
    App->>E: probe_retrieval_tasks()
    E->>utils.py: process_retrieval()
    utils.py-->>E: retrieval metrics
    App->>E: scale_long_context()
    E-->>App: performance at different lengths
    App->>E: compute_metrics()
    E-->>App: final evaluation report
    App-->>main: end
    Note over App: Dataset generation & training are modular, allowing hyperparameter tuning and ablations
    Note over M: Model supports extended positions with RoPE base adjustments
    Note over E: Evaluation includes retrieval, QA scaling, and robustness tests
",
    "Anything UNCLEAR": "Need clarification on the exact prompt templates for GPT-4 data synthesis, detailed training hyperparameters (learning rate schedules, optimizer), and whether to include multi-modal data sources. Details on how to implement sliding window inference (overlap size, handling of context boundaries) are also needed."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "transformers==4.30.0",
        "torch==1.13.1",
        "datasets==2.10.0",
        "peft==0.3.0",
        "accelerate==0.17.0",
        "protobuf==4.21.0",
        "numpy==1.21.0",
        "seqeval==1.2.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "utils.py",
            "Contains utility functions for prompt generation, dataset assembly, and text processing. Shared across data generation, training, and evaluation modules; handles prompt templating, segment assembly, and data serialization."
        ],
        [
            "dataset_generator.py",
            "Uses GPT-4 API via openai or similar SDKs to generate synthetic long-context QA data. Depends on prompt templates defined in utils.py. Generates dataset with varying context lengths, segment placements, and QA pairs requiring multi-hop reasoning."
        ],
        [
            "dataset_loader.py",
            "Loads generated datasets, supports data batching, and applies preprocessing such as tokenization, segment extraction, and window sampling. Uses Hugging Face datasets or custom dataset classes."
        ],
        [
            "model.py",
            "Defines the model class using Hugging Face transformers, supports loading pretrained weights, applying LoRA/PEFT for instruction tuning, and enables inference with extending positional embeddings (RoPE base)."
        ],
        [
            "trainer.py",
            "Handles training loop, optimizer setup, gradient accumulation over multiple steps, and incorporates sliding window inference support. Coordinates with model.py for forward and backward passes. Implements training for long-context data and supports hyperparameters (learning rate decay, batch size)."
        ],
        [
            "evaluation.py",
            "Includes scripts for probing long-context retrieval tasks, measuring F1, accuracy, and robustness metrics at various context lengths. Implements scaling evaluation with sliding window overlaps, multi-hop QA evaluation, and retrieval probing benchmarks."
        ],
        [
            "main.py",
            "Orchestrates the workflow: loads configs, generates datasets, initializes model and trainer, conducts training, runs evaluations including retrieval, scaling tests, and ablation experiments. Manages dependency flow and sequential execution."
        ]
    ],
    "Task list": [
        "utils.py",
        "dataset_generator.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n\n// For this project, API communication is minimal; the scripts primarily run sequentially with file I/O, so no REST API is strictly required. If needed, an API for dataset preview or remote evaluation can be defined but is not necessary for core reproduction.",
    "Shared Knowledge": "utils.py provides prompt templates and dataset serialization functions; config.py defines configurable hyperparameters, dataset sizes, context lengths, and training options. All files rely on torch, transformers, and huggingface datasets for consistent integration.",
    "Anything UNCLEAR": "Clarification needed on the precise GPT-4 API prompt formatting for synthetic data, detailed hyperparameter schedules (learning rate decay, number of steps), and whether to incorporate multi-modal data or only text. Also, confirm hardware resources or approximate training times for large-scale experiments."
}

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 1e-6
  batch_size: 128
  epochs: 1
  warmup_steps: 0.03  # 3% warm-up steps, relative to total steps
  total_steps: 14000  # approximate for one epoch based on dataset size (~14K steps)
model:
  name: "mistral-7b-instruct-v0.2"  # or GPT-4 Turbo, depending on experiment
  rope_base: 1e6  # default, can be tuned as needed
long_context:
  min_length: 4000  # 4K tokens
  max_length: 32000 # 32K tokens
  length_distribution: [4000, 8000, 16000, 32000]
dataset:
  size: 1.1e6  # total number of synthetic data samples for IN2 training
  batch_size: 128
  steps_per_epoch: 14000
generation:
  temperature: 0.7  # optional, if GPT-4 API used for dataset creation
  top_p: 0.95  # optional, for diversity
evaluation:
  tasks:
    retrieval: true
    scaling: true
    few_shot: true
  long_context_lengths: [4000, 8000, 16000, 32000]
  sliding_window_size: 4096  # 4K tokens for sliding window inference
  use_sliding_window: true
```

---

**Note:** Full configuration is available in `planning_config.yaml`
