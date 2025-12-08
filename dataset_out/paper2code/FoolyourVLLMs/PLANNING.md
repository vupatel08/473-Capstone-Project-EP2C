# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, step-by-step plan outlining how to reproduce the experiments and methodologies described in "Fool Your (Vision and) Language Model with Embarrassingly Simple Permutations." This plan covers the core theoretical approach, datasets, experimental setups, hyperparameters, evaluation, and essential technical details based on the paper’s description.

---

## Overview and Core Objectives

**Main goal:**  
Demonstrate that large language models (LLMs) and vision-language models (VLLMs) are vulnerable to permutation-based attacks in multiple-choice question answering (MCQA) tasks. Show how, by systematically permuting answer choices, models' accuracy significantly degrades, often below chance, revealing insights about positional biases and robustness.

---

## 1. Extract Key Methodological Details

### **A. Permutation-Based Attack Strategy**
- **Core idea:**  
  For a given multiple-choice question, permute the order of answer options in a systematic way to generate adversarial answer sets, then evaluate the model on each permutation.
- **Implementation steps:**  
  - Generate all possible permutations of the answer options (or a subset) for each question.
  - For each permutation, rerank (or re-evaluate) the model’s predicted answer distribution.
  - Identify the permutation(s) that cause the model to produce an incorrect answer with high confidence, or to perform worst (below chance).
  - Record the minimum number of permutations needed to fool the model (i.e., cause a performance below chance).

### **B. Evaluation Metrics**
- Accuracy before and after the attack.
- Relative performance drop per question.
- Distribution of permutations needed to break the predictions (e.g., average number of permutations).

### **C. Additional attack variants**
- **Moving answers into specific positions** (e.g., position bias evaluation).
- **Answer set pruning:** reducing answer choices to evaluate robustness as answer options decrease.
- **Model confidence metrics:** maximum predicted probability per permutation.
- **Voting strategies:** majority votes over permutations, including in-context learning (ICL).

---

## 2. Dataset Requirements and Preparation
- **Datasets:**
  - **Text MCQA datasets:** MMLU, ARC-Challenge, BoolQ, SocialIQA, MedMCQA.
  - **Vision-language datasets:** e.g., ScienceQA, A-OKVQA, SEED-Bench, MMBench, among others.  
  - **Other datasets for generalization, e.g., answer set pruning experiments.**
- **Data needed:**
  - Questions with multiple-choice answer options.
  - Correct ground-truth answer labels.
  - For vision-language datasets, images with associated MCQA prompts.
- **Data preparation:**
  - Parse datasets into a canonical format: `question`, `answer options`, `correct answer`.
  - For datasets with rich context or images, prepare prompt templates (see below).
- **Answer permutations:**
  - Generate all permutations of the answer options (or sample a representative subset if combinatorially large).
  - For positional bias analysis, permute answer options systematically (e.g., all possible permutations or a subset).

---

## 3. Model Selection and Setup

### **A. Model choices:**
- **Language models (LLMs):**
  - Llama2-7B, Llama2-13B, Llama2-70B
  - InternLM-7B, InternLM-20B
  - MPT-7B
- **Vision-Language Models (VLLMs):**
  - Otter-Llama, Vicuna-V1.5, WizardLM-13B, etc.
  
### **B. Model access:**
- Use pre-trained open models available via HuggingFace, official repositories, or APIs when possible.
- For models requiring fine-tuning (e.g., robustness experiments), prepare a fine-tuning pipeline.
- For models with multicore parallelism, leverage multiple GPU devices, and consider batching multiple questions.

### **C. Prompting strategy:**
- Use a standard prompt template for multiple-choice questions, e.g.:

  ```
  Question: {question}
  Answer options:
  A. {option A}
  B. {option B}
  C. {option C}
  D. {option D}
  Please select the most appropriate answer: [Answer:]
  ```

- For vision models, also include image embedding or description prompts as per dataset specifics.

### **D. Input formatting:**
- For permutation attack, permute answer options in the prompt before feeding into the model.
- For alternative decoding strategies (confidence, vote), adjust the input prompt accordingly.

---

## 4. Experimental Procedures

### **A. Baseline evaluation:**
- Run the model on the original question-answer set.
- Extract:
  - Model’s predicted answer (e.g., highest probability or token sequence).
  - Predicted answer distribution probabilities (softmax scores).
- Record accuracy (correct if model’s top prediction matches ground truth).

### **B. Permutation attack:**
- Generate all (or a subset of) permutations of answer options.
- For each permutation:
  - Reformat the prompt with permuted answer options.
  - Run model inference—preferably with:
    - Logits output (if available), to measure confidence more directly.
    - Consistent decoding parameters.
  - Record predicted answer and confidence.
- Determine:
  - Whether the permutation leads to an incorrect answer.
  - The minimum number of permutations needed to fool the model below chance.
  - The permutation(s) with the lowest confidence score on the true answer.

### **C. Performance aggregation:**
- For each question:
  - Calculate the percentage of permutations leading to correct answers (per question).
  - Record the number of permutations needed to break the question.
- Aggregate across all questions:
  - Distribution histograms.
  - Mean and median permutation counts.
  - Compare overall performance before vs after attack.

### **D. Additional analyses:**
- Analyze positional bias by fixing the answer position and permuting other options.
- Implement answer set pruning by reducing the number of answer choices (e.g., from 4 to 2 or 3).
- Implement majority voting:
  - Run multiple permutations.
  - Take the model’s best (most confident) answer across permutations.
  - Study robustness.

---

## 5. Hyperparameters and Technical Details

- **Decoding parameters:**
  - Temperature (e.g., 0.5 or 1.0, or tested at 0.5, 1.5).
  - Top-k sampling: e.g., top-10 or top-k=0 for greedy.
  - Nucleus sampling (top-p): e.g., 0.9.
  - Max tokens or max generation length: enough to cover answer options.
  - Decoding strategy: greedy, beam search, or sampling depending on experiment.
- **Permutation specifics:**
  - Generate all permutations for small answer set sizes (max 4 options).
  - For larger sets, sample a fixed number (e.g., 1000).
- **Confidence metrics:**
  - Use softmax over logits for answer choices.
  - Record maximum probability for predictions.
- **Execution resources:**
  - GPUs with sufficient memory (e.g., 40+ GB VRAM for large models).
- **Reproducibility:**
  - Set random seeds.
  - Run multiple trials for stochastic decoding.
  - Log all prompt formats, permutations, and model outputs.

---

## 6. Implementation of the Attack and Evaluation

### **A. Permutation search:**
- For each question, iterate over generated permutations:
  - Feed permuted prompt into the model.
  - Extract answer predictions and confidence.
  - Log the permutation index and predicted answer.
- Check if the answer is correct or below chance (for binary or multi-choice).
- Record permutations needed to fool the model.

### **B. Summarizing results:**
- Create histograms of permutation counts.
- Compute average, median, and worst-case permutation number to break models.
- Calculate performance drops (accuracy ratios).

### **C. Visualization:**
- Histograms (like Figure 4).
- Accuracy before and after attack (tables).
- Correlation analysis (as per figures 2 and 3).

---

## 7. Additional Considerations
- **Model confidence calibration:**
  - Compute calibration metrics if feasible.
  - Use confidence scores to decide when the model is "fooled."
- **In-context learning:**
  - Insert few-shot examples, possibly permuted, to test robustness.
  - Use consistent prompt templates.
- **Reproducibility & Validation:**
  - Cross-validate on multiple datasets.
  - Repeat attacks with different permutation subsets.
  - Test models trained/fine-tuned with different strategies.

---

## Final Notes
- You’ll need to implement permutation generation, prompt formatting, inference routines, and evaluation scripts.
- Ensure modularity so different models, datasets, and attack variants can be swapped easily.
- Carefully match prompt formats and decoding settings to those used in the paper for credible reproduction.

---

This roadmap should guide a comprehensive implementation of the permutation-based adversarial evaluation as described. Once done, the actual code will involve systematic permutational evaluation, confidence extraction, and statistical analysis based on these outlined strategies.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular evaluation framework that loads multiple-choice QA datasets, generates systematic permutations of answer options, runs inference on each permutation using an open-source transformer-based language model (via HuggingFace's transformers library), extracts predictions and confidence scores, and analyzes model robustness. The system will support both text-only and vision-language models, utilizing prompt formatting and inference APIs. Core features include permutation generation, prompt management, inference, confidence assessment, permutation-based attack, performance metrics computation, and visualization outputs. We will implement multithreading or batching for efficiency and ensure reproducibility with seed control.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "prompt_formatter.py",
        "inference.py",
        "attack.py",
        "evaluation.py",
        "visualization.py",
        "config.py"
    ],
    "Data structures and interfaces": "```mermaid\nclassDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(dataset_name: str, split: str)\n        +load_data() -> List[Question]\n    }\n    class Question {\n        +question_text: str\n        +answer_options: List[str]\n        +correct_answer: str\n        +context: Optional[str]\n        +image_path: Optional[str]\n    }\n    class PromptFormatter {\n        +format_prompt(question: Question, permutation: List[str]) -> str\n    }\n    class Model {\n        +__init__(model_name: str, device: str, use_cuda: bool)\n        +predict(prompt: str) -> (str, float)\n    }\n    class Inference {\n        +run(models: List[Model], questions: List[Question], permutations: List[List[str]], prompt_formatter: PromptFormatter) -> List[InferenceResult]\n    }\n    class InferenceResult {\n        +question_id: int\n        +permutation: List[str]\n        +predicted_answer: str\n        +confidence_score: float\n        +correct: bool\n    }\n    class Attack {\n        +generate_permutations(question: Question) -> List[List[str]]\n        +execute_attack(models: List[Model], question: Question, permutations: List[List[str]], prompt_formatter: PromptFormatter) -> AttackOutcome\n    }\n    class AttackOutcome {\n        +question_id: int\n        +permutation_attempts: int\n        +worst_permutation: List[str]\n        +model_predictions: List[InferenceResult]\n    }\n    class Evaluation {\n        +calculate_accuracy(results: List[InferenceResult]) -> float\n        +calculate_performance_drop(base_acc: float, attacked_acc: float) -> float\n    }\n    class Visualization {\n        +plot_permutation_performance(histogram_data: Dict)\n        +plot_accuracy_comparison(pre_attack: float, post_attack: float, dataset_name: str) -> None\n    }\n    Main --> DatasetLoader\n    Main --> Model\n    Main --> PromptFormatter\n    Main --> Inference\n    Main --> Attack\n    Main --> Evaluation\n    Main --> Visualization\n    Attack --> Question\n    Attack --> PromptFormatter\n    Inference --> Model\n    Evaluation --> InferenceResult\n    Visualization --> Dict\n```",
    "Program call flow": "```sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant P as PromptFormatter\n    participant Mdl as Model\n    participant I as Inference\n    participant A as Attack\n    participant E as Evaluation\n    participant V as Visualization\n\n    M->>DL: load_data()\n    DL-->>M: dataset list\n    M->>Mdl: initialize model(config)\n    Note right of M: Loop over models, datasets\n    M->>MI: run inference on original questions\n    MI->>E: compute baseline accuracy\n    M->>A: generate permutations for each question\n    A->>MI: execute_attack for each question, permutations\n    MI-->>A: collect prediction results\n    A->>E: compute attacked accuracy\n    E->>V: plot performance drop and permutation distribution\n    M->>E: report final results\n    ```",
    "Anything UNCLEAR": "Clarification needed on: 1) Which specific datasets to support first; 2) Whether to support vision inputs or restrict to text-based questions; 3) Exact API details for inference, especially confidence extraction; 4) Hyperparameters for inference such as temperature, beam size, etc.; 5) The expected output formats and visualization preferences; 6) Whether to include multiple attack variants or focus solely on permutation-based attacks."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "transformers==4.26.0",
        "datasets==1.17.0",
        "matplotlib==3.5.1",
        "seaborn==0.11.2",
        "tqdm==4.63.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class: loads multiple-choice QA datasets, parsing question text, options, correct answer, and optional context/image paths; provides load_data() method returning question objects."
        ],
        [
            "prompt_formatter.py",
            "Contains PromptFormatter class: formats prompts given a question and a permutation of answer choices; supports both text-only and vision-language prompts; outputs formatted prompt string."
        ],
        [
            "model.py",
            "Contains Model class: initializes pre-trained models (e.g., Llama, GPT, vision-language models via HuggingFace transformers), supports predict(prompt: str) -> (predicted_answer: str, confidence: float). Handles both text-only and multimodal inference setups."
        ],
        [
            "inference.py",
            "Contains Inference class: manages batch inference over questions and permutations, calls Model.predict(), collects predictions, confidences, and correctness labels, supports parallel processing."
        ],
        [
            "attack.py",
            "Contains functions: generate_permutations(question), execute_attack(models, question, permutations, prompt_formatter); manages permutation generation and systematic attack process, records minimal permutations to fool model."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class: computes accuracy before/after attack, models performance drops, and statistics like permutation counts and distributions; provides functions for detailed metrics."
        ],
        [
            "visualization.py",
            "Contains plotting functions: visualize permutation success distribution, accuracy before/after attack, and permutation complexity analysis using matplotlib and seaborn."
        ],
        [
            "main.py",
            "Entry point script: manages configuration, initializes dataset loader, models, prompt formatter, orchestrates experiment flow: baseline eval, attack runs, evaluation, and visualization."
        ]
    ],
    "Task list": [
        "config.py",                   # Configurations including dataset choices, model parameters, batch sizes, hyperparameters, paths.
        "dataset_loader.py",           # Load datasets, parse questions, options, answers, and optional context/images.
        "prompt_formatter.py",         # Create formatted prompts, handle both text-only and vision-language contexts.
        "model.py",                    # Setup models with transformers, openai API, or custom inference pipelines.
        "inference.py",                # Run batch inference, handle multiple permutations per question, store predictions/confidence.
        "attack.py",                   # Generate permutations; run permutation-based attack, track minimal permutations needed.
        "evaluation.py",               # Calculate performance metrics, accuracy, performance drops, permutation statistics.
        "visualization.py",            # Generate plots to illustrate permutation effects, accuracy degradation, and permutation complexity.
        "main.py"                      # Coordinate overall flow: load dataset, evaluate baseline, perform attack, evaluate robustness, visualizations.
    ],
    "Full API spec": ""
    "Shared Knowledge": "Unified prompt formatting functions, dataset object structures (Question class), model wrapper classes, inference pipeline, and utilities for batching and logging results. All components should share configuration variables and logging functions for consistency.",
    "Anything UNCLEAR": "Clarify whether to include vision input handling and image embedding pipelines; confirm model API interface (logits output vs text generation); specify whether attack permutations cover all or sampled; detail output requirements for visualization formats."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
datasets:
  - name: "MMLU"
    split: "test"
  - name: "ARC-Challenge"
    split: "test"
  - name: "BoolQ"
    split: "validation"
  - name: "SocialIQA"
    split: "test"
  - name: "MedMCQA"
    split: "test"
  - name: "ScienceQA"
    split: "test"
  - name: "A-OKVQA"
    split: "test"
  - name: "SEED-Bench"
    split: "test"
  - name: "MMBench"
    split: "test"
  - name: "Answer Set Pruning"
    split: "validation"

models:
  - name: "Llama2-7B"
  - name: "Llama2-13B"
  - name: "Llama2-70B"
  - name: "Vicuna-v1.5-7B"
  - name: "Vicuna-v1.5-13B"
  - name: "WizardLM-13B"
  - name: "InternLM-7B"
  - name: "InternLM-20B"
  - name: "Falcon-7b"
  - name: "MPT-7B"

inference_parameters:
  temperature: 0.5
  top_k: 10
  max_tokens: 50
  num_return_sequences: 1

attack:
  permutation_sampling: 1000
  max_permutations: null  # means generate all or sample up to permutation_sampling
  confidence_threshold: 0.5

prompt_format:
  base_template: |
    Question: {question_text}
    Answer options:
    {answer_options}
    Please select the most appropriate answer: [Answer:]

model_settings:
  device: "cuda"
  use_cuda: true
  seed: 42
```

---

**Note:** Full configuration is available in `planning_config.yaml`
