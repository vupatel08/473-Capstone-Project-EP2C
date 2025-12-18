# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a detailed, structured plan to reproduce the methodology presented in the paper "OpenChat: Advancing Open-Source Language Models with Mixed-Quality Data." The plan covers the key experimental setup, model training, and evaluation procedures aligned with the paper's content.

---

### 1. Data Preparation and Datasets

**a. Base Datasets:**
- **Mixed-Quality Conversation Data:**
  - Use datasets similar to ShareGPT (~70K conversations), containing:
    - **Expert data** (e.g., GPT-4 generated, ~6K conversations).
    - **Sub-optimal data** (GPT-3.5 generated, remaining conversations).
  - For experiments, sample ~128 conversations uniformly from each source for training and evaluation.
- **High-Quality Expert Data:**
  - A small subset (~few thousand) of high-quality conversations, if available, for fine-tuning guidance.
- **Benchmark Datasets:**
  - AlpacaEval, MT-bench, Vicuna-bench, AGIEval (all standard instruction/QA/ reasoning evaluation sets).

**b. Data Processing:**
- **Class Labels:**
  - Assign 'expert' to GPT-4-like data, 'sub-optimal' to GPT-3.5-like data.
- **Construct Class-Conditioned Dataset:**
  - For each conversation, form a tuple: `(input_prompt, response, class_label)`.
  - Define implicit coarse rewards based on class:
    - Expert (`GPT-4`): reward `r_c=1.0`.
    - Sub-optimal (`GPT-3.5`): reward `r_c=α` (e.g., α=0.8).

**c. Prompt Templates:**
- Use the prompt styles illustrated in the paper (e.g., "GPT4 User:", "User:", "Assistant is GPT4<end_of_turn>|>") to standardize input formatting.
- For class-conditioned policy training, prepend context prompts to differentiate class (e.g., including class in the prompt or using specific templates).

---

### 2. Model Architecture and Initialization

**a. Model Selection:**
- Use open-source foundation models:
  - Examples: LLaMA-13B, LLaMA-2-13B, or variants like UltraLM-13B, Vicuna-13B.
- Base models should have a context window of 2,048 tokens.
- Initialize from pre-trained weights; do **not** retrain from scratch.

**b. Model Conditioning:**
- Fine-tune as a **class-conditioned language model**:
  - Incorporate class labels (`c`) as input tokens or conditioning prompts.
  - Example: For a conversation, include a marker (`<|expert|>`) or prompt template indicating class.

---

### 3. Fine-Tuning via C(onditioned)-RLFT

**a. Method Overview:**
- Treat the dataset as samples from class-conditioned policies \(\pi_c(y|x,c)\).
- The reward \(r_c(x,y)\) is coarse, derived from class label.
- Regularize the policy \(\pi_\theta(y|x,c)\) towards the behavior policy \(\pi_c(y|x,c)\) with a KL term, weighted by \(\beta\).

**b. Implementation Details:**
- **Policy parameterization:**
  - Fine-tune the model \(\pi_\theta(y|x,c)\), conditioning on class label.
- **Input Construction:**
  - Concatenate prompts with class-conditioning tokens/templates.
  - For example, "GPT4 User:" for expert data, "User:" for sub-optimal.
- **Objective Function:**
  
  \[
  J(\theta) = \mathbb{E}_{(x,c,y)\sim \mathcal{D}_c} \left[\exp\left(\frac{1}{\beta} r_c(x,y)\right) \log \pi_\theta(y|x,c) \right]
  \]
  
  - Approximate via supervised, reward-weighted regression.
- **Training Steps:**
  - For each batch:
    - Sample a batch of tuples \((x_i,c_i,y_i)\).
    - Compute weights \(w_i = \exp\left(\frac{1}{\beta} r_{c_i}(x_i,y_i)\right)\).
    - Update the model by maximizing weighted likelihood:
      
      \[
      \sum_{i} w_i \log \pi_\theta(y_i | x_i, c_i)
      \]
      
    - Use cross-entropy loss weighted by \(w_i\).
  - Learning rate schedule: Use a relatively small learning rate (e.g., 1e-5 to 3e-5), with warm-up and cosine decay.
  - Number of epochs: 3-5 epochs over the dataset to prevent overfitting.
  - Batch size: 200-400 (adjust depending on hardware).
- **Hyperparameters:**
  - \(\beta\): tune in range [0.1, 0.3], balancing reward influence.
  - \(\alpha\) (reward for sub-optimal data): e.g., 0.8.
  - KL regularization indirectly incorporated via the reward weights.

---

### 4. Model During Inference

- Use the same prompt template as during training, with the class-conditioning prefix.
- Generate responses using greedy search or temperature sampling (e.g., temperature=0.7).
- For class-conditioned responses, prompt as: "GPT4 User:" or "User:" depending on class.

---

### 5. Evaluation Strategy

**a. Instruction-following Benchmarks:**
- Use publicly available evaluation models:
  - AlpacaEval, MT-bench, Vicuna-bench, AGIEval.
- Set up:
  - Use top-rated GPT-4 or GPT-3.5 responses as references.
  - Evaluate via:
    - Win rate in pairwise comparisons.
    - Absolute score (e.g., GPT-4 judge scoring responses on a 1-10 scale).
  - For pairwise: compare model A vs model B on the same task and extract win/lose/tie.

**b. Metrics:**
- **Win Rate (%):** proportion of model responses rated better than baseline.
- **Score (Average):** mean score assigned by evaluator.
- **Correlation with Human Judgment:** optionally compute Pearson/Spearman correlations.

**c. Additional Analyses:**
- Conduct ablation studies:
  - Remove class-conditioning prompts.
  - Vary reward \(\alpha\).
  - Sample subset of data to assess robustness.
- Visualization:
  - Use t-SNE or UMAP on conversation embeddings (from models’ hidden states) to visualize distribution differences with and without class conditioning.
  - Assess representation separation of expert/sub-optimal data.

---

### 6. Reproducibility and Implementation Details

- **Libraries/Tools:**
  - Use Hugging Face transformers or similar framework supporting LLaMA / LLaMA-2 models.
  - Implement custom training loop for weighted supervised training.
  - Use mixed-precision (FP16 or bfloat16) for efficiency.
- **Hardware:**
  - GPU clusters with at least 16-24GB VRAM.
  - Distribute training across multiple GPUs if possible.
- **Code Management:**
  - Use seed control for reproducibility.
  - Log hyperparameters, training curves, and evaluation results systematically.

---

### 7. Open Questions & Clarifications Needed
- Exact value of reward parameter (\(\beta\)) and reward for sub-optimal data (\(\alpha\))—initially set as suggested, with hyperparameter tuning.
- Specific form of templates and tokenization approaches for class-conditioning (e.g., special tokens, prefix prompts).
- Whether to fine-tune as a single condition (e.g., "Expert" vs "Suboptimal") or individual templates per class.
- How to handle multi-turn conversations: input-output concatenation, dialogue history formatting.

---

This roadmap provides a comprehensive plan aligned with the paper's methodology, dataset choices, training procedure, and evaluation metrics, paving the way for subsequent implementation and experimentation.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Build a modular system utilizing Hugging Face Transformers for model loading, a custom supervised training loop for reward-weighted fine-tuning of a class-conditioned LLaMA-based model, and standard evaluation scripts for benchmark assessment. The core process involves loading a pre-trained model, preparing class-conditioned prompts, performing weighted supervised learning based on coarse rewards, and evaluating on instruction-following benchmarks. The architecture emphasizes simplicity with separate modules for data loading, model handling, training, and evaluation, all orchestrated by a main script.",
    "File list": [
        "app.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__()\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> List[Tuple[str, str, str, float]]  # (prompt, response, class_label, reward)\n        +get_train_dataset() -> Dataset\n        +get_eval_dataset() -> Dataset\n    }\n    class Model {\n        +__init__(pretrained_model_name: str, conditioning_token: str)\n        +forward(input_ids: Tensor, attention_mask: Tensor) -> Tensor\n        +generate(prompt: str, max_new_tokens: int, temperature: float) -> str\n        +set_conditioning(conditioning_token: str) -> None\n    }\n    class Trainer {\n        +__init__(model: Model, dataset: Dataset, beta: float, learning_rate: float, batch_size: int, epochs: int)\n        +train() -> None\n        +save_model(path: str) -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, eval_dataset: Dataset)\n        +evaluate(metrics: List[str]) -> dict\n        +score_response(prompt: str, response: str) -> float\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> Model\n    Evaluation --> Model\n    DatasetLoader --> Tuple[str, str, str, float]\n    Model --> Tensor\n",
    "Program call flow": "sequenceDiagram\n    participant C as Main\n    participant DL as DatasetLoader\n    participant M as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    C->>DL: load_data()\n    DL-->>C: dataset list (prompt, response, class_label, reward)\n    C->>M: initialize(pretrained_model_name, conditioning_token)\n    C->>TR: start training(model, dataset, beta, lr, batch_size, epochs)\n    TR->>M: set_conditioning(class_label)\n    TR->>M: forward(input_ids, attention_mask)\n    TR-->C: emit training metrics\n    C->>M: save_model(path)\n    C->>EV: evaluate(model, eval_dataset)\n    EV->>M: generate(prompt, max_tokens, temperature)\n    EV-->>C: evaluation metrics\n    C->>Main: Training and evaluation complete\n",
    "Anything UNCLEAR": "Clarify the exact input dataset format (prompt + response structure), preferred tokenizer handling, and specific hyperparameters for reward scaling (\(\beta\), \(\alpha\)). Also, confirm whether to incorporate class labels as special tokens or prompts and how to handle multi-turn conversations during training."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "transformers==4.31.0",
        "datasets==2.9.0",
        "sentencepiece==0.1.96",
        "scipy==1.10.0",
        "numpy==1.21.0",
        "tqdm==4.65.0",
        "ubelt==1.2.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class: responsible for loading mixed-quality conversation datasets, assigning class labels, and returning structured data (prompts, responses, class labels, reward weights). It depends on datasets library and handles data parsing and batching."
        ],
        [
            "model.py",
            "Defines Model class: loads a pre-trained LLaMA-based model from Hugging Face, implements prompt conditioning (via special tokens or prompt templates), and provides methods for forward pass and response generation. It depends on transformers library."
        ],
        [
            "trainer.py",
            "Contains Trainer class: initializes with model, dataset, hyperparameters (beta, lr, batch size, epochs), implements the weighted supervised training loop based on reward weights, includes setting conditioning prompts, optimizer, and loss calculation. Depends on torch API and model.py."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class: loaded with trained model and evaluation dataset, provides methods to generate responses, score responses according to benchmark criteria, and compute metrics like win rates and scores. Depends on model.py and datasets."
        ],
        [
            "main.py",
            "Entry script orchestrating the workflow: initializes dataset loader, model, trainer, runs training, saves model, then runs evaluation on benchmarks. Coordinates calling methods from other modules, manages hyperparameters, and handles logging."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "Not applicable – no front-end API, only training and evaluation scripts provided.",
    "Shared Knowledge": "Dataset loader shares dataset configurations and data parsing utilities with trainer (such as reward assignment logic). Model class provides standardized API for forward pass and generation. Hyperparameters like beta, alpha values need to be consistently used across modules.",
    "Anything UNCLEAR": "Clarify the exact format or schema of dataset files (e.g., JSON, CSV), preferred prompt templates for class conditioning, and any hardware constraints for large-scale experiments."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 3e-5
  batch_size: 200
  epochs: 3
  beta: 0.2                     # Reward scale parameter
  alpha: 0.8                    # Reward for sub-optimal data
  max_grad_norm: 1.0
  warmup_steps: 1000
  weight_decay: 0.01
evaluation:
  eval_interval: 1              # Evaluate every epoch
  eval_benchmarks:
    - AlpacaEval
    - MT-bench
    - Vicuna-bench
    - AGIEval
  metrics:
    - win_rate
    - score
  num_eval_samples: 128
model:
  pretrained_model_name: "huggingface/llama-13b"
  conditioning_token: "<|class|>"   # Token or prompt prefix for class conditioning
dataset:
  dataset_path: "data/sharegpt_mixed_quality.json"  # Path to the dataset
  train_sample_size: 128
  eval_sample_size: 128
optimization:
  lr_scheduler: cosine
  max_steps: 10000
seed: 42
output_dir: "output/openchat_finetuned"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
