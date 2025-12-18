# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive and detailed step-by-step plan to reproduce the methodology, experiments, and evaluations described in the paper "Language Model Detectors Are Easily Optimized Against". The plan covers all key aspects necessary for faithful reproduction while highlighting gaps or uncertainties where relevant.

---

# 1. Understand and Extract Core Methodology Components

## a. Main Objective
- Fine-tune a large language model (LLM) (e.g., Llama-2-7B) such that its outputs "fool" detection models into misclassifying AI-generated text as human-written, using reinforcement learning (RL) with preference-based reward signals, under a KL-divergence constraint to maintain language quality.

## b. Key Techniques
- RL optimization leveraging **Direct Preference Optimization (DPO)**.
- Use of **detectors** (OpenAI RoBERTa, DetectGPT, etc.) as black-box reward functions (human-ness scores, log probabilities).
- **Preference dataset**: Pairs of outputs (AI-generated texts) labeled according to which is more human-like, derived from detector scores.
- **Fine-tuning**: Start from a pre-trained LM and fine-tune via rewards obtained from detector preferences.
- **Evaluation**: AUROC and perplexity metrics, with additional human annotation and qualitative assessments.

---

# 2. Implementation Roadmap

## A. Data Collection & Preparation
1. **Base dataset for initial prompt-response pairs:**
   - Use existing open-source datasets, e.g., OpenWebText, or publicly available prompt-response datasets, such as the OpenAI API outputs.
   - For qualitative consistency, replicate the paper’s approach for text types: e.g. academic, social media, essays.
   
2. **Generate initial outputs:**
   - Use the base model (pre-trained Llama-2 7B) to generate outputs from the selected prompts.
   - Keep track of prompt-response pairs for subsequent experiments.

3. **Obtain human annotations for ground-truth labels (optional but beneficial):**
   - As per the paper, human annotators labeled pairs of texts (baseline vs. fine-tuned). For initial validation, rely on detector scores; human annotations enable more nuanced evaluations.

4. **Construct Preference Dataset:**
   - For each prompt, generate two outputs: one from the base model, one from the fine-tuned model.
   - Score both outputs with detectors (e.g., RoBERTa, DetectGPT).
   - Assign preference: label the output perceived as more human-like.
   - Repeat across many prompt-output pairs (target number ~10K pairs for training, as per the paper).

## B. Detector Models & Reward Signal
1. **Acquire or implement detector models:**
   - Obtain publicly available detectors such as:
     - OpenAI's RoBERTa-based classifier
     - DetectGPT (if available)
     - Commercial detectors (via APIs)
   - For black-box models (API-based), implement input-output interface.
   
2. **Determine detector scores as reward:**
   - Larger human-ness score (log probability, detector output probability) implies more "human-like".
   - For each generated text, obtain the detector score.

3. **Preference probability (reward) calculation:**
   - Use the Bradley-Terry model as in the paper:
     \[
     p(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))
     \]
   - Here, \( r(x, y) \) could be the detector score (or log probability). 
   - For simplicity, use the detector score directly as \( r \).

## C. Fine-tuning via Reinforcement Learning (DPO Method)
1. **Starting point**: Use the pre-trained Llama-2 7B model.
2. **Define loss function**:
   - Implement the DPO objective as in the paper:
     \[
     \mathcal{L}_{DPO} = - \mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma(\beta \Delta R) \right]
     \]
   - Where:
     \[
     \Delta R = \log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)
     \]
     (determine whether to use log probabilities or detector scores directly)
   - \(\beta\): hyperparameter controlling the reward strength (e.g., 0.5).

3. **KL-divergence constraint**:
   - Enforce divergence regularization via KL penalty, as in the RL setup.
   - Use a coefficient \(\beta\) (not to be confused with the reward scaling factor): tune between 0.05 to 5 as in paper.

4. **Optimization algorithm**:
   - Use PPO (Proximal Policy Optimization), adapted for language models with sequence sampling.
   - Incorporate KL penalty: either via constrained optimization or as penalty term.

5. **Training steps**:
   - Sample prompts.
   - Generate outputs from current model.
   - Compute detector preference scores.
   - Calculate \(\mathcal{L}_{DPO}\).
   - Update model parameters via PPO or a similar RL algorithm, maintaining the KL constraint.
   - Periodically evaluate on validation pairs.

6. **Training duration**:
   - Target similar steps: around 1-2 hours (~30k steps), depending on hardware.
   - Use limited compute (~a few hundred dollars of cloud resources).

## D. Evaluation & Metrics
1. **Automatic detector AUROC**:
   - Generate a set of texts before and after fine-tuning.
   - Get detector scores for each text.
   - Measure AUROC, expecting a drop after optimization.

2. **Perplexity**:
   - Evaluate the language model’s perplexity on OpenWebText or equivalent corpora.

3. **Human evaluation**:
   - Conduct annotations similar to Figure 4:
     - Present humans with pairs (baseline vs. fine-tuned) for the same prompt.
     - Record preference for human-likeness.

4. **Transferability tests**:
   - Evaluate the fine-tuned models on various detectors (not used during training) to analyze generalization.
   - Measure AUROC differences.

5. **Ablation/Hyperparameter sweep**:
   - Vary \(\beta\), dataset size, sequence length, and model size (if doing scale experiments).
   - Record resulting AUROC and perplexity.

---

# 3. Additional Details & Missing Information

- **Model fine-tuning setup**:
  - Do not have explicit hyperparameters (e.g., learning rate, batch size). Use suggested defaults:
    - Learning rate: 1e-5 to 5e-5
    - Batch size: 16-32 sequences
    - Optimizer: AdamW
    - Sequence length: 120-250 tokens
- **Reward calibration**:
  - When detector scores are used directly, check for score scaling. Possibly normalize detector scores for stability.
- **Sampling method**:
  - Use top-k sampling or nucleus sampling (e.g., top-p=0.9) during generation.
- **Number of RL steps**:
  - Approximate as in the paper: 30k steps (~1hr training on GPUs).
- **Constraint hyperparameter (\(\beta\))**:
  - Experiment with \(\beta \in \{0.05, 0.5, 5\}\).
- **Data logging**:
  - Record prompt, output (pre/post), detector scores, preference labels, loss, and hyperparameters.
- **Reproducibility**:
  - Set seed values for sampling and training.
  - Use consistent detokenizer, prompt templates, and model configs.

---

# 4. Summary Roadmap

| Stage | Tasks | Details/Notes |
|---------|--------|----------------|
| Data Preparation | Collect prompts → generate baseline outputs → compute detector scores | Replicate datasets, use similar prompts and generators |
| Preference Dataset | Pair outputs based on detector scores | Human annotation optional but recommended |
| Detector Setup | Load or proxy detectors | API-based or open-source models |
| Reward Computation | Calculate preference probabilities | Use log scores or detector probabilities |
| Fine-tuning | Implement DPO objective → run RL with KL constraints | Use PPO; tune hyperparameters |
| Evaluation | Measure AUROC, perplexity, human preferences | Use multiple detectors, human annotators |
| Transferability | Test models on unseen detectors | Confirm robustness |

---

# 5. Clarifications & Missing Data
- Exact hyperparameters for RL training (learning rate, batch size)
- Precise definition of reward \( r(x,y) \) (detector score, log prob, or normalized)
- Number of RL epochs or steps (target ~30k steps)
- Handling of random seeds for reproducibility
- Specific prompt templates or datasets used in the paper (can approximate with publicly available datasets)

---

This detailed plan provides a solid foundation for software implementation, ensuring the reproduction faithfully captures the research methodology. It also identifies areas where assumptions or further clarification may enhance fidelity.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Build a modular system comprising data collection, detector interface, RL fine-tuning, and evaluation. Use Hugging Face Transformers for model loading and generation, PyTorch for training, and a custom PPO implementation adapted for sequence sampling with KL constraints. Leverage existing detector models via API or open-source implementations. Use DPO loss for RL updates, with detector scores as reward signals. Keep the architecture straightforward with a main control script, dataloader, model wrapper, trainer, and evaluator.",
    "File list": [
        "main.py",  
        "dataset_loader.py", 
        "model.py",  
        "trainer.py",
        "evaluation.py",
        "detectors.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(prompts: List[str], responses: List[str])\n        +load_data() -> dict\n        +get_pairs() -> List[Tuple[str, str, str]]\n    }\n    class ModelWrapper {\n        +__init__(model_name: str, device: str)\n        +generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str\n        +log_prob(sequence: str, input_prompt: str) -> float\n        +get_score(sequence: str) -> float  # detector score\n    }\n    class PPOTrainer {\n        +__init__(model: ModelWrapper, dataset: List[Tuple[str, str, str]], detector: DetectorAPI, beta: float, kl_coeff: float, lr: float, batch_size: int, total_steps: int)\n        +train() -> None\n        +save_checkpoint(path: str) -> None\n        +load_checkpoint(path: str) -> None\n    }\n    class Evaluator {\n        +__init__(model: ModelWrapper, detectors: List[DetectorAPI])\n        +evaluate_texts(texts: List[str]) -> dict\n        +compute_auroc(scores_human: List[float], scores_ai: List[float]) -> float\n        +compute_perplexity(texts: List[str]) -> float\n    }\n    class DetectorAPI {\n        +__init__(model_name: str or api_endpoint: str)\n        +score(text: str) -> float\n    }\n    class Utilities {\n        +normalize_scores(scores: List[float]) -> List[float]\n        +compute_preference(p1_score: float, p2_score: float) -> float  # logistic\n        +sample_batch(dataset: List[Tuple], batch_size: int) -> List[Tuple]\n    }\n    Main --> DatasetLoader\n    Main --> PPOTrainer\n    Main --> Evaluator\n    Main --> Utilities\n    PPOTrainer --> ModelWrapper\n    PPOTrainer --> DetectorAPI\n    Evaluator --> DetectorAPI\n    DatasetLoader --> Utilities\n    ModelWrapper <|-- TransformersModelWrapper  # subclass implementing interface\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MW as ModelWrapper\n    participant PPO as PPOTrainer\n    participant EV as Evaluator\n    participant D as DetectorAPI\n\n    M->>DL: load_data() // Load prompts and responses\n    DL-->>M: data\n    M->>MW: initialize(model_name='llama2-7b')\n    M->>PPO: start training with data, detector, hyperparameters\n    PPO->>PPO: for step in total_steps apply PPO update using detector rewards\n    PPO->>MW: update parameters\n    alt save checkpoint\n        PPO->>Utilities: save_checkpoint()\n    end\n    M->>EV: evaluate_texts(pre_post_samples)\n    EV->>D: score(text)\n    D-->>EV: score value\n    EV-->>M: report metrics (AUROC, perplexity)\n    M->>Main: finish\n",
    "Anything UNCLEAR": "Clarify detector sources: Are they API-based (e.g., OpenAI) or open-source models? Exact hyperparameters for PPO (learning rate, KL penalty weight). Details on prompt formats and number of training steps. Confirm if human annotation integration is needed or optional for automated pipeline. Clarify dataset sources or prompt templates used in the experiments."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "transformers==4.29.1",
        "torch==2.0.0",
        "numpy==1.23.5",
        "scipy==1.10.1",
        "tqdm==4.65.0",
        "requests==2.31.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class to handle loading prompt-response pairs, generate outputs, compute detector scores, and assemble preference pairs. Depends on the data sources and detector APIs."
        ],
        [
            "detectors.py",
            "Provides DetectorAPI class to interface with detector models or APIs (e.g., API tokens, open-source models). Functions: score(text) returning scalar detector scores. Used in DatasetLoader and evaluation modules."
        ],
        [
            "model.py",
            "Defines ModelWrapper class wrapping Hugging Face models, with methods for sequence generation, computing log probabilities, and interfacing with model inference. Relies on transformers library."
        ],
        [
            "trainer.py",
            "Implements PPOTrainer class, responsible for RL fine-tuning using DPO loss, detector rewards, and KL regularization. Depends on ModelWrapper, DatasetLoader, and DetectorAPI."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class for computing metrics: AUROC (via detector scores), perplexity, and human preference evaluation. Uses DetectorAPI and model inference methods."
        ],
        [
            "main.py",
            "Main script initializing data, models, detector interfaces, training process, and evaluation. Coordinates the study, sequences the workflow, and manages hyperparameters."
        ],
        [
            "utils.py",
            "Contains utility functions such as score normalization, sampling helpers, preference probability calculations, and data batching. Provides common functions shared across modules."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "detectors.py",
        "utils.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "None (internal modules only; detector API calls for evaluation and preference generation).",
    "Shared Knowledge": "Shared hyperparameters for training (learning rates, batch sizes), detector score normalization, sequence sampling parameters (top-p, temperature), and KL regularization coefficients.",
    "Anything UNCLEAR": "Clarify whether detector models are API-based or local open-source; details on exact prompt templates; hyperparameter tuning specifics (learning rate, steps); desired hardware setup and expected runtime."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 1e-5  # Default starting learning rate for model fine-tuning
  batch_size: 16       # Batch size for PPO RL training
  total_steps: 30000   # Total RL training steps, approximated from paper (~1 hour on GPU)
  kl_coeff: 0.5        # Coefficient for KL divergence regularization
  beta: 0.5            # Reward scaling factor for preference loss in DPO
  model_name: "Llama-2-7b"  # Model identifier from Hugging Face
  sequence_length: 120  # Max tokens per prompt-response pair during generation
  temperature: 0.7      # Sampling temperature for text generation
  top_p: 0.9            # Nucleus sampling parameter
  save_interval: 5000  # Save model checkpoint every N steps
  device: "cuda"        # Device setting
dataset:
  prompt_source: "openwebtext"   # Source of prompts (or custom dataset)
  responses_sample_size: 10000   # Number of samples for preference dataset
detectors:
  api_endpoint: "https://detector.api"  # Placeholder URL for detector API
  api_token: "YOUR_API_TOKEN"            # Placeholder for API token if needed
evaluation:
  evaluate_steps: 10000  # Number of steps between evaluations
  human_annotation: false # Set true if human preference annotation is included
```

---

**Note:** Full configuration is available in `planning_config.yaml`
