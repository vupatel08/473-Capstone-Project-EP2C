# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, step-by-step plan that captures all critical methodology components, experimental setups, hyperparameters, and evaluation metrics necessary for reproducing the paper’s VDC framework and experiments.

---

# 1. Overview & Objectives
- **Primary goal:** Reproduce the "Versatile Data Cleanser (VDC)" for detecting various dirty samples (poisoned, noisy labels, hybrids) in image datasets.
- **Core innovation:** Use multimodal Large Language Models (MLLMs) for cross-modal semantic inconsistency measurement via a three-module pipeline:
  - Visual Question Generation (VQG)
  - Visual Question Answering (VQA)
  - Visual Answer Evaluation (VAE)
- **Universality & Generalization:** Method is designed to work without training on specific dirty samples; only relies on prompting and inference via pre-trained models.

---

# 2. Core Methodology Components & Design Strategy

### A. Dataset Preparation
- **Datasets:**
  - CIFAR-10 (size: 50k train, 10k test)
  - ImageNet-100 (subset of ImageNet, class imbalance: ~500/100 per class)
  - ImageNet-Dog (subset: 10 dog classes)
- **Dirty sample generation:**
  - Poisoned samples:
    - Using triggers from BadNets, Blended, SIG, TrojanNN, SSBA, WaNet.
    - For each attack, embed triggers in images and change labels.
  - Noisy labels:
    - Symmetric: Random class flip proportion.
    - Asymmetric: Flip to next class or similar.
  - Hybrid datasets: Combine poisoned + noisy labels.
- **Implementation notes:** 
  - For poison triggers, use predefined or approximate trigger images/masks as shown in Figures 3-4.
  - Use existing backdoor attack code or custom code to embed triggers:
    - Stick to specified trigger types and parameters.
    - For synthetic noise, flip labels randomly or asymmetrically according to ratios.
- **Data format:** Store as image files with labels; ensure dataset splits are maintained.

### B. Main Pipeline (VDC)

**Modules:**
1. **Visual Question Generation (VQG):**
   - Input: Image + label
   - Output: A set of questions:
     - General questions (G): e.g., "Describe the image briefly"
     - Label-specific questions (S): generated via prompts with the label as context.
   - Implementation detail:
     - Use ChatGPT (or GPT-3.5/4) with carefully designed prompts.
     - Prompts template for each question type:
       - For general: fixed template, e.g. "Describe the image."
       - For label-specific: prompt with label, e.g., "Is the object in the image a [label]? Explain."
   
2. **Visual Question Answering (VQA):**
   - Input: Images + questions
   - Process: Use a pre-trained multimodal large language model (MLLM) capable of cross-modal reasoning, e.g., Otter (oracles like GPT-4 with vision capabilities, or LLMs integrated with visual encoders).
   - Implementation detail:
     - Call the MLLM inference API (or local model) with prompt: "Answer the question based on the image (provided as input)."
     - Use the question prompt received above.
   - Output: Model response (string)

3. **Visual Answer Evaluation (VAE):**
   - Input: Expected answers + model responses
   - For label-specific questions:
     - Use string containment ("yes"/"no") for expected deterministic answers.
     - Direct string match or keyword matching for labels.
   - For general questions:
     - Use ChatGPT (or GPT-4) as an evaluator, with prompts like:
       - "Does the answer '{response}' correctly describe the image with label '{label}'? Answer 'true' or 'false'."
   - Calculate correctness indicator $e_i^j$.

4. **Semantic Consistency Scoring & Sample Detection:**
   - Compute consistency score $s_i$ for each sample:
     \[
          s_i = \frac{\sum_{j=1}^{N_q} \mathbb{1}(e_i^j = true)}{N_q}
     \]
     where $N_q$ is total questions per sample.
   - Threshold $\alpha$ (e.g., 0.2) to decide if sample is dirty:
     - If $s_i < \alpha$, classify as dirty.

### C. Implementation Details
- Use **pre-trained models**:
  - For Visual Encoder + Prompting:
    - OpenAI's GPT-4 with vision (if accessible)
    - CLIP (for vision-language similarity)
    - Other open-source multimodal models (e.g., Otter, VisualBERT, Unimodal + tokenizer prompts)
  - For question-answering, API call or local inference.
  - For prompt design, follow described formats and include image or visual embedding as needed.
- **Prompt engineering:**
  - For question generation prompts, provide examples and labels.
  - For evaluating correctness, prepare clear prompts to ChatGPT or GPT-4.

### D. Hyperparameters & Settings
- Number of questions per sample: 
  - General: 2-4, label-specific: 2-4.
  - For ablation: vary the number of questions (e.g., 1 to 8), as shown in Figures 2b.
- Threshold $\alpha$:
  - Set as 0.2 for detection (as per paper).
- Model parameters:
  - For training baseline classifiers (e.g., ResNet, SimiFeaT), follow the settings:
    - Optimizer: Adam or SGD
    - Learning rate: start at 0.1 or 0.01
    - Batch size: 64 or 128
    - Epochs: 40 (for CIFAR-10), 200 (for ImageNet), with decay if used.
- For prompt calls:
  - Use API keys or open-source models with APIs (OpenAI, Azure, or local models).
  - Set model temperature to low (~0.2) for deterministic output, high (~0.8) for diversity.
  
---

# 3. Experimental Procedures & Evaluation Metrics

### A. Dirty Sample Detection
- For each dataset:
  - Run the pipeline on the entire dataset.
  - Apply thresholding on $s_i$ to classify:
    - Clean samples
    - Dirty samples
  - Compute metrics:
    - True Positive Rate (TPR): proportion of dirty samples detected.
    - False Positive Rate (FPR): proportion of clean samples incorrectly flagged.
- For hybrid or mixed datasets:
  - Use combined detection strategy.
  - Compare with baseline (e.g., pure CLIP similarity, confidence-based methods).

### B. Retrieval & Similarity Checks
- Use CLIP for fast similarity scores:
  - Embedding the image and label text.
  - Threshold similarity score for detection.
- Cross-validate with prompt-based inference.

### C. (Optional) Model retraining after clean-up
- Retrain target classifiers:
  - CIFAR-10/100, ImageNet subsets.
  - Using purified data based on detection thresholds.
- Evaluate:
  - Clean accuracy (ACC).
  - Attack Success Rate (ASR) for poisoned samples.
- Validate the generalization and robustness of the cleaned dataset.

### D. Additional Evaluation
- Ablations on:
  - Number of questions per sample.
  - Type of questions (general vs label-specific).
  - Multimodal model choice.
- Sensitivity analysis over thresholds $\alpha$.
- Computational cost measurement (API inference time, model size).

---

# 4. Implementation Notes & Caveats
- **Model dependencies:**
  - Need access to GPT-4 (vision-enabled) or equivalent.
  - Use of OpenAI API or open-source models like BLIP2, LaVAE, or open CLIP.
- **Prompt design:**
  - Carefully craft prompts for consistency evaluations.
  - Example prompts provided in Appendices E1-E2.
- **Data Privacy & API tokens:**
  - Manage API keys securely.
  - For large scale, consider local models or GPU inference.

---

# 5. Closing Remarks
- Structure experiments systematically:
  - Generate poisoned/noisy data.
  - Run detection modules.
  - Vary questions/questions count.
  - Measure TPR, FPR, and retraining metrics.
- Document all hyperparameters, prompts, model versions, and evaluation procedures for reproducibility.

---

This roadmap serves as a solid foundation to implement, tune, and evaluate the VDC framework as described. Let me know if you'd like a detailed prompt template, code templates, or specific API configurations next.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Build a modular system with core components for dataset management, prompt generation, model inference, and evaluation. Use open-source libraries such as PyTorch for dataset handling, Hugging Face Transformers for language models, OpenAI API for GPT-4 with vision (if accessible), or open-source multimodal models like BLIP2 and CLIP. The system employs a straightforward pipeline: load data, generate questions via prompts, answer questions using an MLLM, evaluate responses for consistency, and classify samples based on the computed semantic scores. For large-scale experimentation, script batch processing with configurable parameters (questions count, thresholds). Detection results are used to filter datasets and retrain classifiers, re-evaluating detection accuracy and downstream task performance.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "prompt_generator.py",
        "model_inference.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +load_data() -> Dataset\n        +apply_poisoning(poison_params: dict) -> Dataset\n        +apply_noisy_labels(noise_params: dict) -> Dataset\n    }\n    class PromptGenerator {\n        +generate_general_questions(label: str, num_questions: int) -> List[str]\n        +generate_label_specific_questions(label: str, num_questions: int) -> List[str]\n    }\n    class ModelInference {\n        +answer_questions(image: Image, questions: List[str]) -> List[str]\n        +evaluate_response(question: str, answer: str, image: Image, label: str) -> bool\n        +get_semantic_similarity(image: Image, label: str) -> float\n    }\n    class Evaluator {\n        +calculate_score(answers: List[str], responses: List[str]) -> float\n        +detect_dirty_sample(score: float, threshold: float) -> bool\n    }\n    Main --> DatasetLoader\n    Main --> PromptGenerator\n    Main --> ModelInference\n    Main --> Evaluator\n    DatasetLoader --> Dataset\n    Main --> Dataset\n    PromptGenerator --> str: question\n    ModelInference --> List[str]: answers\n    ModelInference --> bool: correctness\n    ModelInference --> float: similarity\n    Evaluator --> float: score\n    Evaluator --> bool: detection\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant PG as PromptGenerator\n    participant MI as ModelInference\n    participant E as Evaluator\n    M->>DL: load_data()\n    DL-->>M: dataset\n    M->>PG: generate_general_questions(label, num_questions)\n    M->>PG: generate_label_specific_questions(label, num_questions)\n    PG-->>M: questions list\n    M->>MI: answer_questions(image, questions)\n    MI-->>M: answers list\n    loop for each sample\n        MI->>MI: evaluate_response(question, answer, image, label)\n        MI-->>M: correctness boolean\n    end\n    MI->>MI: get_semantic_similarity(image, label)\n    MI-->>M: similarity score\n    M->>E: calculate_score(answers, correctness)\n    E-->>M: sample score\n    M->>E: detect_dirty_sample(score, threshold)\n    E-->>M: classification (dirty/clean)\n    loop for dataset filtering and retraining\n    M->>DatasetLoader: apply_poisoning/noisy_labels as needed\n    \"",
    "Anything UNCLEAR": "Clarification needed on the preferred open-source multimodal inference library (e.g., Hugging Face models, OpenAI API, or open vision-language models). Also, confirmation on the processing workflow for batching samples—should the pipeline be per-sample or batched for efficiency?"
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch>=1.12",
        "numpy>=1.21",
        "transformers>=4.26",
        "requests",
        "Pillow",
        "tqdm"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class responsible for loading CIFAR-10, ImageNet subsets, applying poisoning or noisy label augmentation, and providing dataset splits for training and evaluation. This module is foundational and must be completed prior to model training and evaluation tasks."
        ],
        [
            "prompt_generator.py",
            "Implements PromptGenerator class responsible for creating question prompts for general and label-specific questions, utilizing fixed templates or template-based prompt construction, possibly interfacing with GPT models for question generation. It depends on the selection of class labels and dataset info, thus invoked early in the pipeline."
        ],
        [
            "model_inference.py",
            "Contains ModelInference class that wraps the multimodal large language model API calls, including answering questions about images, evaluating correctness responses, and computing semantic similarity scores. It requires the prompts from prompt_generator.py and the dataset images; thus, it depends on prior data loading."
        ],
        [
            "evaluation.py",
            "Implements Evaluator class responsible for scoring answers per question, aggregating to sample-level scores, applying thresholds, and classifying samples as dirty or clean. It consumes the correctness indicators and similarity scores from ModelInference and dataset labels for final detection logic."
        ],
        [
            "main.py",
            "The highest-level orchestrator that initializes configuration, loads datasets via DatasetLoader, generates questions via PromptGenerator, runs inference and evaluation through ModelInference and Evaluator, performs detection, and optionally retrains models. It depends on all other modules being ready in the specified sequence."
        ],
        [
            "utils.py",
            "Provides shared utility functions such as image processing, batching helpers, API wrappers, and configuration management. It supports all modules needing common functions, especially for API calls and data handling."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "prompt_generator.py",
        "model_inference.py",
        "evaluation.py",
        "utils.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n\n# The system involves the following API endpoints:\n# 1. POST /generate_questions\n#    - Description: Generate visual questions for given label.\n#    - Request body: { \"label\": str, \"question_type\": \"general\" | \"label-specific\", \"question_count\": int }\n#    - Response: { \"questions\": List[str] }\n# 2. POST /answer_questions\n#    - Description: Given an image and questions, get answers from MLLM.\n#    - Request body: { \"image_path\": str, \"questions\": List[str] }\n#    - Response: { \"answers\": List[str] }\n# 3. POST /evaluate_response\n#    - Description: Evaluate correctness of answers, score via semantic similarity.\n#    - Request body: { \"question\": str, \"answer\": str, \"label\": str }\n#    - Response: { \"is_correct\": bool, \"similarity\": float }\n# 4. POST /detect_sample\n#    - Description: Classify sample as dirty or clean based on aggregated score.\n#    - Request body: { \"sample_score\": float, \"threshold\": float }\n#    - Response: { \"is_dirty\": bool }",
    "Shared Knowledge": "All modules rely on a common configuration object that specifies dataset paths, API keys (for GPT), thresholds, number of questions, and hyperparameters. Utility functions provide image loading, API call wrappers (with rate limit handling), and prompt templates. Dataset loading and poisoning parameters are shared via dataset_loader.py and configuration.",
    "Anything UNCLEAR": "Clarification needed on the exact API access to GPT-4 (vision-enabled) or other multimodal models, and whether local inference is preferred. Also, confirmation on whether detection results lead directly to dataset filtering or require re-annotation."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  cifar10:
    image_size: 32
    num_classes: 10
    train_samples_per_class: 500
    test_samples_per_class: 100
  imagenet100:
    image_size: 224
    num_classes: 100
    train_samples_per_class: 500
    test_samples_per_class: 100
  imagenet_dog:
    image_size: 224
    num_classes: 10
    train_samples_per_class: 800
    test_samples_per_class: 200

attack:
  poisoned:
    triggers: [BadNets, Blended, SIG, TrojanNN, SSBA, WaNet]
    embed_method: "trigger embedding method details"
    label_change_strategy: "change label to target label"
  noisy_labels:
    type: symmetric
    ratio: 0.4
    type: asymmetric
    ratio: 0.4
  hybrid:
    poison_ratio: 0.09
    noise_ratio: 0.1

training:
  epochs: 40  # standard for CIFAR-10 as per paper
  learning_rate: 0.1  # typical default, adjust if needed
  batch_size: 64  # typical batch size, adjust per system memory
  optimizer: "SGD or Adam"  # specified in experiments
  learning_rate_decay: 0.1  # optional, based on usual conventions
  
detection:
  questions_per_sample: 4  # example from experiments
  question_type: "general and label-specific"
  threshold: 0.2
  use_multimodal_model: true

model:
  name: "ResNet"  # or other models used for retraining
  input_size: 32  # for CIFAR-10 baseline; change for ImageNet
  epochs: 40
  optimizer: "SGD"
  learning_rate: 0.01  # default, tune if necessary

evaluation:
  detection_threshold: 0.2
  retraining_epochs: 40
  metrics:
    detection: {TPR, FPR}
    downstream: {accuracy, attack success rate}
```

---

**Note:** Full configuration is available in `planning_config.yaml`
