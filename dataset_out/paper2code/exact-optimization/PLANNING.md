# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, structured plan outlining the key steps for reproducing the methodology, experiments, and evaluation detailed in the paper *"Towards Efficient Exact Optimization of Language Model Alignment"*. This plan covers the core theoretical derivations, the practical implementation details, dataset handling, hyperparameters, and evaluation procedures needed to faithfully reproduce the results.

---

## 1. Overall Objective and Theoretical Foundation
- **Goal:** Implement EXO (Efficient Exact Optimization) for language model alignment, which mathematically circumvents RL gradients by directly minimizing a distribution divergence (reverse KL) toward an optimal policy derived from preference data.
- **Core idea:** 
  - Theoretically, the optimal policy is a Boltzmann distribution (Eq. 4).
  - Exact optimization (via Eq. 13) aligns via minimizing the reverse KL between a reweighted policy derived from the current policy and the ideal Boltzmann distribution.
  - This method avoids high-variance RL gradient estimates, enabling stable, sample-efficient training.

---

## 2. Implementation Blueprint

### A. Data Preparation
- **Preference Datasets:** 
  - Collect human annotation pairs (response pairs) or use existing preference datasets (e.g., on summarization, dialogue, instructions).
  - For experiments mimicking paper settings:
    - **Controlled experiments:** Generate synthetic responses (e.g., from GPT-2 or GPT-3) under a fixed baseline policy; define a reward function (oracle or learned reward model).
    - **Real human preferences:** Use datasets like Reddit TL;DR, Anthropic helpful/harmful dialogues, or custom human annotations.
- **Test Prompts:** Curate prompts for evaluation, ensuring consistency with the paper's datasets; e.g., IMDb sentiment prompts, summarization prompts, or instruction-following prompts.

### B. Model Initialization and Architectures
- Base Models:
  - Use pretrained language models as initial policies:
    - GPT-2 large or Pythia-2.8B for summarization/dialogue.
    - ChatGLM2-6B for instruction following.
- **Reward Model:**
  - Either train a reward model (e.g., binary classifier or regression) on human-preference labels or use existing reward models (e.g., trained from RLHF).
- **Sequence Normalization:**
  - Use causal autoregressive models (e.g., GPT-family). Ensure models are loadable and allow token sampling.

### C. The Approximate Distribution Matching (Eq. 13)
- **Sampling:**
  - For each prompt \(x\), generate a batch of \(K\) responses from the current policy \(\pi_\theta\) (preferably \(K=4\) or \(8\) to balance variance and computation).
  - Use temperature scaling (e.g., \(\tau=0.8\)) consistent with the paper.
- **Compute weights:**
  - For each response \(y_i\), compute likelihood ratio:
    \[
    f_\theta(y_i) = \frac{1}{\beta_r} r_\phi(x, y_i) \quad \text{(if using learned reward)}.
    \]
  - Use the sample responses to estimate the distribution \(p_{f_\theta}\):
    \[
    p_{f_\theta}(i) \propto \exp(f_\theta(y_i))
    \]
  - Compute a similar distribution \(p_{r_\phi}\) from the reward model probabilities.
- **Divergence minimization:**
  - Minimize the empirical reverse KL between \(p_{f_\theta}\) and \(p_{r_\phi}\), which reduces to a form of importance-weighted maximum likelihood or a contrastive divergence (see Eq. 23). This involves:
    - For responses \(y_i\), compute importance weights \(\sim \exp(f_\theta(y_i))/\sum_j \exp(f_\theta(y_j))\).
    - Use these weights to update \(\pi_\theta\) via weighted maximum likelihood.

### D. Gradient Estimation and Parameter Updates
- **Gradient step (Eq. 13, 14):**
  - For each data batch, compute the weighted likelihood gradient:
    \[
    \nabla_\theta \mathcal{L}_{exo} \approx \sum_{i} w_i \nabla_\theta \log \pi_\theta(y_i|x),
    \]
    where weights \(w_i \sim p_{f_\theta}(i)/p_{r_\phi}(i)\).
  - Use autograd frameworks (PyTorch, TensorFlow) to compute these gradients efficiently.
- **Regularization:**
  - Optionally, add KL regularization to maintain proximity to the initial LM policy if needed.
- **Optimization hyperparameters:**
  - Learning rate (e.g., \(1e-5\) to \(3e-5\)), batch size (e.g., 64), number of epochs or gradient steps per prompt (e.g., 3000-5000 steps).
  - Use Adam with fix parameters \(\beta_1=0.9,\beta_2=0.999\).

### E. Training Workflow
1. For each prompt:
   - Sample \(K\) responses from \(\pi_\theta\).
   - Compute importance weights based on reward model \(r_\phi\).
2. Update \(\pi_\theta\) using the importance-weighted likelihood.
3. Periodically (every N steps), evaluate divergence (reverse KL estimate) and reward performance.
4. Optionally, tune \(\beta_r, \beta_\pi\) to balance diversity and reward adherence, as per the paper's suggested regime.

---

## 3. Experiment-Specific Details
- **Controlled experiment** (Fig. 2,3,7):
  - Generate responses from a fixed LM baseline.
  - Use an oracle reward (e.g., sentiment classifier).
  - Run multiple \(\beta_r, \beta_\pi\) settings.
  - Plot reward vs divergence (reverse KL).
- **Real human preference tuning** (Tables 1-2, Figures 4,6,8):
  - Use datasets with human annotations or preferences.
  - Fine-tune \(\pi_\theta\) by minimizing the EXO loss (Eq. 13, 23).
  - Evaluate with:
    - Human pairwise comparisons (if available).
    - Proxy reward models.
    - Zero-shot GPT-4 evaluation (via API), asking GPT-4 to judge "which response is better".

### Hyperparameter Tuning
- Augment hyperparameters:
  - \(\beta_\pi, \beta_r\): Explore magnitudes \(\{0.05, 0.1, 0.5, 1.0\}\).
  - Temperature in sampling: \(\tau=0.8\).
  - Batch size: 64 responses per prompt.
  - Number of training steps: 3000–10,000.
  
### Computational Aspects
- Parallelize over prompts and responses.
- Use GPU accelerators; for large models, distribute across multiple GPUs.
- Store model checkpoints periodically for analysis and ensemble evaluation.

---

## 4. Evaluation Procedures
- **Metrics:**
  - Reward model proxy scores (oracle or learned reward).
  - GPT-4 pairwise comparisons for human-aligned preferences.
  - Classification accuracy (e.g., sentiment ROC-AUC for summarization instruction prompts).
  - Divergence estimates: approximate reverse KL (using importance sampling or density ratio estimators).
  - Win rates in pairwise human or GPT-based judge comparisons.

- **Plots:** 
  - Reward vs reverse KL curves (Fig. 2, 6, 8).
  - Density ratio visualizations (Fig. 3, 9).
  - Accuracy vs divergence (Fig. 7).
  - Win rate over training steps (Fig. 8).

---

## 5. Summary and Additional Considerations
- Carefully match the probabilistic formulations in the appendix to implement the divergence minimization.
- Use stochastic gradient descent with variance reduction techniques:
  - Sampling multiple responses.
  - Importance weights normalization.
- Validate the optimization by checking divergence estimates and reward scores periodically.
- Cross-validate hyperparameters like \(\beta_r\), \(\beta_\pi\), and temperature.
- For reproducibility, record seed, model versions, and dataset specifics.

---

This roadmap provides a comprehensive, step-by-step guide to implementing, training, and evaluating the EXO approach as described. Each step can be further detailed during actual coding, but following this plan ensures fidelity to the paper's methodology and experimental protocols.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Design a modular system leveraging Hugging Face Transformers for language models, PyTorch for tensor operations and optimization, and optional gradient accumulation for large models. Implement separate modules for data loading, model initialization, response sampling, importance weight computation, divergence minimization, and evaluation. Use parallel processing for response generation and batch updates for efficiency. Implement the core loss as a custom PyTorch function incorporating the importance weights and divergence terms, following equations (13) and (23). The training loop alternates between sampling responses, computing weights from the reward model, updating the policy via weighted likelihood, and periodically evaluating reward metrics and divergence estimates. Use APIs from Hugging Face, PyTorch, and OpenAI for GPT-2/3, or similar models, and pre-trained reward models when available.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(dataset_path: str)
        +load_preference_pairs() -> List[Tuple[Prompt, ResponsePair]]
        +load_prompts() -> List[str]
    }
    class LanguageModel {
        +__init__(model_name_or_path: str, device: str)
        +generate_responses(prompt: str, num_responses: int, temperature: float) -> List[str]
        +log_probs(responses: List[str], prompt: str) -> List[float]
    }
    class RewardModel {
        +__init__(model_path: str)
        +score_response(prompt: str, response: str) -> float
        +score_responses(prompt: str, responses: List[str]) -> List[float]
    }
    class ResponseSampler {
        +sample_responses(prompt: str, num_responses: int, temperature: float) -> List[str]
        +compute_importance_weights(responses: List[str], reward_scores: List[float], beta: float) -> List[float]
    }
    class ModelTrainer {
        +__init__(lm: LanguageModel, reward_model: RewardModel, optimizer_params: dict)
        +train_step(prompts: List[str], batch_size: int, beta: float, reward_scale: float) -> None
        +save_checkpoint(path: str)
        +load_checkpoint(path: str)
    }
    class Evaluator {
        +__init__(lm: LanguageModel, reward_model: RewardModel)
        +evaluate_on_dataset(prompts: List[str], responses: List[str]) -> dict
        +estimate_reverse_kl(responses: List[str], prompt: str) -> float
    }
    class UtilityFunctions {
        +softmax(logits: Tensor) -> Tensor
        +importance_weighted_log_likelihood(responses: List[str], weights: List[float], lm: LanguageModel, prompt: str) -> torch.Tensor
    }
    Main --> DatasetLoader
    Main --> LanguageModel
    Main --> RewardModel
    Main --> ResponseSampler
    Main --> ModelTrainer
    Main --> Evaluator
    ResponseSampler --> LanguageModel
    ModelTrainer --> LanguageModel
    ModelTrainer --> RewardModel
    Evaluator --> LanguageModel
    Evaluator --> RewardModel
    UtilityFunctions --> LanguageModel
    UtilityFunctions --> torch
    responses: List[str]
    weights: List[float]
    logits: torch.Tensor
    \n// All classes implement type-annotated methods with clear input/output types, adhering to PEP8 and standard Python practices.",
    "Program call flow": "
sequenceDiagram
    Main->>DatasetLoader: load_preference_pairs()
    DatasetLoader-->>Main: data list with prompt-response pairs
    Main->>LanguageModel: init (model_name_or_path, device)
    Main->>RewardModel: load (model_path)
    Main->>ModelTrainer: init with lm, reward_model, optimizer_params
    loop Training steps
        Main->>Main: select batch of prompts
        Main->>ResponseSampler: sample_responses(prompt, num_responses, temperature)
        ResponseSampler-->>Main: responses list
        Main->>RewardModel: score_response(prompt, response) for each response
        RewardModel-->>Main: reward_scores list
        Main->>UtilityFunctions: compute importance weights with beta
        UtilityFunctions-->>Main: weights list
        Main->>ModelTrainer: train_step(prompts, responses, weights, beta, learning_rate)
        alt evaluation interval
            Main->>Evaluator: evaluate(prompts, responses)
            Evaluator-->>Main: metrics (reward, divergence)
        end
    end
    Main->>Main: save model checkpoint
    "
}

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.1",
        "numpy==1.21.0",
        "transformers==4.29.1",
        "tqdm==4.64.1",
        "scipy==1.10.1"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class: handles loading preference pairs, prompts; supports both synthetic and real data; depends on file I/O, dataset formats, and coordinates with main.py for data intake."
        ],
        [
            "model.py",
            "Defines LanguageModel class: initializes with pretrained transformer models from Huggingface, supports response generation and log probability extraction; depends on transformers library and torch."
        ],
        [
            "reward_model.py",
            "Implements RewardModel class: loads a pretrained reward model (e.g., classifier or regression head), scores individual responses and response batches; interfaces with main.py for scoring processes."
        ],
        [
            "trainer.py",
            "Implements ModelTrainer class: manages training loop, response sampling, importance weight computation, and gradient updates; depends on torch, model.py, reward_model.py, and utility functions."
        ],
        [
            "evaluation.py",
            "Contains Evaluator class: executes evaluation on datasets, estimates divergence (reverse KL), assesses reward scores, and supports metrics collection; interfaces with model.py and reward_model.py."
        ],
        [
            "utils.py",
            "Includes utility functions such as softmax calculation, importance weighting, and divergence estimation; shared by trainer.py for core training routines and evaluation.py for metrics."
        ],
        [
            "main.py",
            "Entry point script: initializes data loader, models, trainer, evaluator; orchestrates the training, evaluation loop; manages hyperparameters; depends on all other modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "reward_model.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "\"\"\" (defining training, evaluation, sampling APIs for modular access; e.g., generate responses, update model, evaluate divergence and reward; details depend on HTTP or CLI setup, if any. For CLI, minimal API or function calls suffice.)\"\"\"",
    "Shared Knowledge": "Common utility functions such as softmax, importance weighting, and divergence estimators in utils.py support training and evaluation modules. Configurations such as model paths, hyperparameters, and response batch sizes are maintained centrally and loaded by main.py.",
    "Anything UNCLEAR": "Clarification needed on whether the implementation targets multi-GPU distributed training for large models or single GPU/CPU; also, details on response batching size and synchronization policies during training."
}

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 1e-5  # Default learning rate; adjust based on validation
  batch_size: 64  # Number of prompt-response samples per update
  epochs: 100  # Total number of epochs to run; approximate, depending on convergence
  response_samples: 8  # Number of responses generated per prompt during sampling
  response_temperature: 0.8  # Controls diversity of generated responses
  importance_sampling_beta: 0.5  # Importance weight scaling for response likelihoods
  divergence_regularization_beta: 0.1  # Beta parameter for divergence regularization
  max_training_steps: 5000  # Total training steps or checkpoints
model:
  pretrained_model_path: "gpt2-large"  # Use HuggingFace identifier, e.g., "gpt2-large"
  device: "cuda"  # Use "cuda" if GPU is available, else "cpu"
reward_model:
  model_path: "path/to/reward/model"  # Path to the pretrained reward score model
dataset:
  preference_data_path: "path/to/preference/dataset"  # Path to human preference pairs dataset
  prompts_path: "path/to/prompts.txt"  # File containing prompts for evaluation
evaluation:
  eval_batch_size: 16  # Batch size for evaluation inference
  eval_steps: 3000  # Number of steps between evaluation checkpoints
  divergence_estimation_method: "importance_sampling"  # Can be "density_ratio" or "importance_sampling"
# Additional hyperparameters may be tuned as needed
```

---

**Note:** Full configuration is available in `planning_config.yaml`
