# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, step-by-step reproduction plan aligned with the paper's methodology, experiments, datasets, hyperparameters, and evaluation metrics. This plan serves as a comprehensive blueprint for implementing the experiments, training reward models, and performing evaluations as described.

---

## **High-Level Reproduction Strategy for the Paper**

### 1. **Data Acquisition & Preparation**
- **Instruction Data:**
  - Collect and compile a dataset of ~5200 diverse instructions from sources such as Dolly, Super-NI, and User-oriented datasets.
  - Generate **candidate responses** for each instruction using **Alpaca-7B**, prompting it with the instruction to produce **5 responses per instruction** (max length 128).
- **Feedback Data:**
  - Acquire **large-scale human and AI feedback datasets**:
    - **AI feedback:** 71K instances of ratings and 46.5K instances of pairwise rankings.
    - **Human feedback:** 4K instances for ratings and 2K for pairwise rankings.
  - Pay particular attention to the **formats**:
    - Ratings: scores 1-7, possibly with explanations.
    - Rankings: pairwise preferences with optional explanations.
- **Data Quality Checks:**
  - Confirm that datasets include **both** ratings and rankings (collected independently from humans and AI).
  - Process and filter annotations to remove inconsistent or noisy annotations based on the paper’s inconsistency analysis.

---

### 2. **Reward Model (RM) Construction and Training**
- **Objective:**
  - Train reward models to reflect human/AI preferences.
- **Model Architectures:**
  - Use **Alpaca-7B** as the base model for initial responses.
  - For reward functions, implement:
    - **Regression Reward Model (RR):** trained on response scores (ratings).
    - **Pairwise Preference Models (RR + ranking):** trained on pairwise ordering data.
    - **Alternative: RoBERTa-Large** as an auxiliary reward model to compare robustness.
- **Data for Reward Model Training:**
  - **Ratings data:** For RR, use the scores directly.
  - **Preferences data:** Convert pairwise rankings into training data:
    - For each pair (response1, response2), assign:
      - 1 if response1 preferred,
      - 0 if response2 preferred,
      - 0.5 if tie, or use the “equally good/bad” labels with rare ties.
- **Training Details:**
  - Use **negative log-likelihood loss** (e.g., for pairwise data) to model preference probabilities.
  - For regression, use MSE or sigmoid-based score regression.
  - **Hyperparameters:**
    - Batch size: 16–64, depending on GPU memory.
    - Learning rate: e.g., 1e-5 to 3e-5 (tune based on validation performance).
    - Number of epochs: Sufficient to converge (~3–5), with early stopping based on performance on validation subset.
  - **Implementation notes:**
    - Use **standard HuggingFace Transformers** with fine-tuning.
    - Incorporate **custom loss functions** for preference learning (pairwise likelihood) and rating regression.

---

### 3. **Model Alignment & Policy Optimization**
- **Policy Approach:**
  - Use the **Rejection Sampling (Best-of-n) policy**:
    - Sample n responses from **Alpaca-7B** for each instruction.
    - Score each response via the trained reward model.
    - Select the response with the highest reward as the final output.
- **Hyperparameters & Sampling Strategy:**
  - n (number of responses per instruction): e.g., 64.
  - Use the reward model trained on the chosen feedback protocol.
  - Implement **sampling from the base model** (e.g., top-p sampling, temperature=0.0 or +0.5).
  - Use **early stopping** or **annealing** if incorporating reinforcement learning techniques; otherwise, rejection sampling suffices.
- **Fine-tuning (Optional):**
  - If desired, tune the base LLM with the reward signal obtained from these policies (similar to RLHF).

---

### 4. **Evaluation Protocols & Metrics**
- **Win-Rate Computation:**
  - For each instruction in the evaluation set (unseen instructions):
    - Generate:
      - Base response (e.g., Alpaca-7B).
      - **Response from the aligned policy** (Best-of-n).
      - **Reference responses** (e.g., from GPT-3.5-Turbo).
    - Use **pre-trained reward models** (aligned with training protocol) to evaluate:
      - Response vs. Reference responses.
      - Responses from different models.
  - For each comparison:
    - Compute the **preference** from the reward model scores.
    - Calculate **win-rate**: proportion where the aligned model is preferred.
- **Evaluation Schemes:**
  - **Ranking-based evaluation**: compare responses pairwise using the reward model’s preference probability.
  - **Rating-based evaluation**: have annotators or the reward model output a score for each response, then compare scores.
- **Metrics:**
  - **Win-rate** against a reference (e.g., GPT-3.5 Turbo) on the test set.
  - **Inconsistency analyses**:
    - Measure preference consistency (e.g., from multiple annotators or model replicates).
    - Use the provided formulas.

---

### 5. **Inconsistency & Bias Analyses**
- **Quantify inconsistency:**
  - Calculate the percentage of pairwise comparisons that do **not** align with the expected preference (e.g., based on human judgment or model preference).
  - Use metrics similar to Eq. (hl, h-m, etc.).
- **Bias Checks:**
  - Assess whether response length or vocabulary diversity influences preferences:
    - Implement statistical tests (e.g., compare average length/UNIQUE tokens between preferred and unpreferred responses).
  - Evaluate verbosity biases (from prior work references).
- **Sensitivity of evaluation:**
  - Verify how changing feedback protocols (ratings vs. rankings) affects model evaluation metrics (Table 2, 10).

---

### 6. **Reproducibility & Hyperparameters**
- **Random Seeds:**
  - Fix seeds across all experiments (e.g., Python, NumPy, PyTorch seeds).
- **Computational Resources:**
  - Use at least 1–2 GPUs with sufficient memory (e.g., 48GB VRAM) for large models.
  - For faster training, consider multi-GPU or distributed training if available.
- **Logging & Metrics:**
  - Track training loss, validation preferences, and consistency scores.
  - Save checkpoints per epoch or after early stopping criteria.
- **Validation:**
  - Use a subset of data (e.g., 5–10%) for validation of reward models and performance tuning.

---

## **Summary Timeline & Steps**

1. **Data Collection & Preprocessing (Weeks 1–2):**
   - Collect instruction datasets, generate responses, and gather feedback.
   - Process, filter, and format ranking and rating data.

2. **Reward Model Implementation & Training (Weeks 3–4):**
   - Implement models (Alpaca-7B fine-tuning, RoBERTa-based models).
   - Train on collected feedback with proper loss functions.

3. **Policy Construction & Fine-tuning (Weeks 5–6):**
   - Use the trained reward models in rejection sampling.
   - Generate responses and evaluate on unseen instructions.

4. **Evaluation & Analyses (Weeks 7–8):**
   - Compute win-rates, inconsistency metrics.
   - Cross-validate with human annotations where feasible.
   - Perform bias and variability checks.

5. **Reporting & Documentation (Week 9):**
   - Summarize results, compare protocols.
   - Document hyperparameters and experiment settings.

---

## **Notes & Clarifications Needed**
- Exact dataset splits (train/validation/test) for feedback data: Not explicitly provided – approximate via the paper.
- Thresholds for early stopping: Not specified – tune based on validation.
- Implementation details of the preference and regression loss functions: Use standard HuggingFace training routines, custom loss if needed.
- Response length and other biases: Confirmed minimal, but should verify in experiments.
- Protocol details (e.g., sampling temperature, top-p): Use values consistent with prior standards and prior work.

---

This plan provides a thorough roadmap, emphasizing reproducibility, key details, and experimental rigor aligned with the paper. Once executed, it will enable precise replication and further exploration.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular software system that loads datasets, trains reward models using feedback data, and performs response generation and evaluation. Open-source libraries such as Hugging Face Transformers will be used for language models, PyTorch for training, and standard scripting for data processing and evaluation. The architecture comprises clearly separated classes for data handling, model training, reward modeling, policy sampling, and evaluation, enabling straightforward reproduction and experimentation.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "reward_model.py",
        "policy.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(data_dir: str)\n        +load_instruction_data() -> Dataset\n        +load_feedback_data() -> FeedbackData\n        +generate_responses(model: LanguageModel, instructions: List[str], responses_per_instruction: int) -> List[ResponseSample]\n    }\n    class LanguageModel {\n        +__init__(model_name: str, device: str)\n        +generate(prompt: str, max_length: int) -> str\n    }\n    class FeedbackData {\n        +ratings: Dict[Tuple[str, str], float]\n        +pairwise_rankings: List[Tuple[str, str, int]]  # response1, response2, preference (1 or 2)\n    }\n    class RewardModel {\n        +__init__(model_name: str, feedback_data: FeedbackData, use_preferred: bool)\n        +train() -> None\n        +score_response(response: str, instruction: str) -> float\n        +score_pair(response1: str, response2: str, instruction: str) -> float\n    }\n    class PolicySampler {\n        +__init__(base_model: LanguageModel)\n        +sample_responses(instruction: str, n_samples: int, temperature: float) -> List[str]\n        +select_best_response(responses: List[str], reward_model: RewardModel, instruction: str) -> str\n    }\n    class Evaluation {\n        +__init__(model: LanguageModel, reward_models: List[RewardModel], dataset: Dataset)\n        +evaluate_response_quality() -> dict\n        +compute_win_rate() -> float\n        +assess_inconsistency() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> RewardModel\n    Main --> PolicySampler\n    Main --> Evaluation\n    DatasetLoader -- generate_responses: list of ResponseSample\n    RewardModel -- score_response(response, instruction): float\n    RewardModel -- score_pair(response1, response2, instruction): float\n    PolicySampler -- sample_responses(instruction,): list of str\n    PolicySampler -- select_best_response(responses, reward_model, instruction): str\n    Evaluation -- evaluate_response_quality(): dict\n    Evaluation -- compute_win_rate(): float\n    Evaluation -- assess_inconsistency(): dict",
    "Program call flow": "sequenceDiagram\n    actor Main\n    participant PS as PolicySampler\n    participant DM as DatasetLoader\n    participant LM as LanguageModel\n    participant RM as RewardModel\n    participant EV as Evaluation\n\n    Main->>DM: load_instruction_data()\n    DM-->>Main: instructions list\n    Main->>LM: initialize with model_name\n    Main->>DM: generate_responses(instructions, responses_per_instruction=5)\n    DM-->>Main: responses list\n    Main->>RM: initialize reward models (ratings, pairwise)\n    RM-->>Main: reward models\n    Main->>RM: train on feedback data\n    Main->>PS: instantiate with base_model\n    loop for each instruction in evaluation dataset\n        PS->>LM: sample_responses(instruction, n=64, temperature=0.0)\n        participant responses: list of strings\n        loop over responses\n            RM->>response: score_response(response, instruction)\n        end\n        best_response = PS.select_best_response(responses, RM, instruction)\n        # similarly, evaluate against reference responses or reference model responses\n    end\n    Main->>EV: evaluate_response_quality()\n    EV-->>Main: metrics data\n    Main->>EV: compute_win_rate()\n    EV-->>Main: win rate percentage\n    Main->>EV: assess_inconsistency()\n    EV-->>Main: inconsistency metrics\n"
    ,
    "Anything UNCLEAR": "Clarification needed on the exact dataset file formats, data paths, and hyperparameters such as number of training epochs, batch sizes, and temperature values for response sampling. Also, confirm if the feedback data is preprocessed or needs custom filtering, and whether to include human annotations in the reward model training."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "transformers==4.29.2",
        "torch==1.12.1",
        "datasets==2.4.0",
        "scipy==1.9.3",
        "numpy==1.21.0",
        "pyyaml==6.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains classes for loading instruction datasets, feedback datasets, and functions for generating responses using language models (e.g., Alpaca-7B). It depends on 'datasets' for data IO and 'transformers' for model inference."
        ],
        [
            "model.py",
            "Defines the LanguageModel class encapsulating loading and inference of models like Alpaca, GPT-3.5, or GPT-4. It uses 'transformers' models or API wrappers and must be importable by 'reward_model.py' and 'policy.py'."
        ],
        [
            "reward_model.py",
            "Implements RewardModel class for training and evaluating reward models. It uses outputs from 'model.py' for scoring responses (regression/reg preference). It requires torch, transformers, and handling of feedback datasets."
        ],
        [
            "trainer.py",
            "Contains classes/functions for training reward models (both regression and preference). Depends on 'torch', 'scipy' for optimization, and 'feedback' datasets produced by dataset_loader.py."
        ],
        [
            "policy.py",
            "Defines PolicySampler class implementing response sampling with reward model scoring for Best-of-n policies. Uses 'model.py' for response generation and 'reward_model.py' for scoring."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class to compute win rates, inconsistency metrics, and response quality evaluations. Depends on models, reward models (from 'reward_model.py'), and dataset_loader for test sets."
        ],
        [
            "main.py",
            "Entry point script orchestrating data loading, model training, response sampling, and evaluation. It instantiates classes from all above modules and manages execution flow."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "reward_model.py",
        "trainer.py",
        "policy.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n  # No API required; all operations are internal function calls and class methods for reproducibility.",
    "Shared Knowledge": "Common configuration such as model names, hyperparameters, dataset paths, and evaluation protocols should be stored in a config file or constants module accessed by all scripts.",
    "Anything UNCLEAR": "Clarify if response generation should be via API calls (OpenAI API) or local models, and the exact hardware setup intended for training large models. Also, confirm the format and storage location of feedback datasets."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 1e-5
  batch_size: 16
  epochs: 3
  weight_decay: 0.01
  warmup_steps: 500
  evaluation_steps: 1000
  save_steps: 2000
  max_grad_norm: 1.0
sampling:
  n_responses: 64
  temperature: 0.0
reward_model:
  model_name: "allenai/longformer-base-4096"
  training_epochs: 3
  learning_rate: 3e-5
  batch_size: 16
  weight_decay: 0.01
  early_stopping_patience: 2
policy:
  responses_per_instruction: 64
  sample_temperature: 0.0
evaluation:
  test_instructions_path: "data/test_instructions.json"
  reference_responses_path: "data/reference_responses.json"
  batch_size: 32
  evaluation_samples: 1000
feedback:
  feedback_data_path: "data/feedback_feedback.json"
  human_annotations_path: "data/human_annotations.json"
  feedback_protocol: "ranking"  # or "rating"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
