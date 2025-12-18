# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, step-by-step outline for reproducing the methodology described in the paper "Meaning Representations from Trajectories in Autoregressive Models." This plan covers all key technical details, experimental setups, hyperparameters, datasets, and evaluation metrics, structured to facilitate later implementation.

---

## 1. Core Concept & Method Overview

**Objective:**  
Represent the *meaning* of a prompt (sentence, phrase, or composite textual/modality input) as a *distribution over possible continuations* (trajectories) of that prompt, using pre-trained autoregressive models without fine-tuning or prompt engineering.

**Key innovation:**  
- Instead of static vector embeddings, derive *distributional meaning representations* grounded in model likelihoods over continuations starting from a given prompt.  
- These distributions can be manipulated algebraically to model asymmetric relations like entailment or hypernym/hyponym (via likelihood functions).

---

## 2. Step-by-step Methodology

### A. Data Inputs & Preprocessing

- **Primary input:**  
  - A *prompt string* \( s \), which could be an entire sentence, phrase, or modality-specific (image captioning prompts).  
  - For multimodal models, raw image inputs or object features (if applicable).  
- **Outputs & representations:**  
  - Distribution over trajectories extending \( s \).

- **Processing details:**  
  - Tokenize input prompts using the specific tokenizer of the chosen pre-trained model.  
  - For multimodal inputs, prepare modality-specific prompt formats (e.g., “Describe this image:...” for images).

---

### B. Model Selection & Setup

- **Model choice depends on task (text-only, multimodal):**  
  - **Text-only:** Use large autoregressive transformers like GPT-2, LLaMA, or GPT-2-XLa.  
  - **Multimodal:** Use vision-language models like LLaVA, with possible modifications.

- **Model interface:**  
  - Load with HuggingFace transformers or custom implementations.  
  - Ensure the model can output *token probabilities* (logits) for sequence continuation sampling.

---

### C. Trajectory Sampling & Distribution Approximation

**Goal:**  
Approximate the *distribution over all possible continuations* \( M_s \) (likelihood functions conditioned on prompt \( s \)).

**Steps:**  
1. **Sampling trajectories:**
   - For each prompt \( s \), generate \( n \) trajectories \(\{ t_i \}\), each up to length \( m \) tokens.
   - Use multinomial sampling (or temperature-based sampling) from the model conditioned on \( s \).  
   - **Hyperparameters:**  
     - Number of trajectories \( n \) (e.g., 20).  
     - Maximum trajectory length \( m \) (e.g., 20 tokens).  
     - Sampling temperature \( \lambda \) (e.g., 1.0).  
   - Store sequences of token probabilities or *log-likelihoods*.

2. **Likelihood estimation:**
   - For each sampled trajectory \( t_i = a_1, a_2, \dots, a_m \), compute the likelihood \( P(a_i | s, a_{<i}) \) using the model logits.  
   - Compute the *log-likelihood* sum:  
     \[
     \log P(t_i | s) = \sum_{i=1}^m \log p(a_i | s, a_{<i})
     \]

3. **Distribution over continuations:**
   - Derive an approximate *likelihood function* \( M_s(t) \) for each trajectory:  
     \[
     M_s(t_i) \propto \exp(\sum_{i=1}^m \log p(a_i | s, a_{<i}))
     \]
   - Use the normalized likelihoods across sampled trajectories for approximating the model's distribution over continuation space.

---

### D. Represent Meaning as Distribution Functions

- Construct a **distribution function \( M_s \)** over the set of trajectories as a *probability measure* derived from likelihood ratios or normalized sums.  
- **Automata perspective:**  
  - Construct *set of feasible continuation strings* \( T_s \) with likelihood thresholds (if modeling "set" vs. "distribution").  
  - For simplicity, approximate the *distribution over continuations* conditioned on prompt.

**Algebraic operations:**
- Use likelihood ratios to compare meanings:  
  \[
  d(M_u, M_v) \approx \mathbb{E}_{t \sim \frac{1}{2}(M_u + M_v)} |\log M_u(t) - \log M_v(t)|
  \]
- Approximate expectation via sampled trajectories, as in the method described.

---

## 3. Tasks and Experiments

### A. Semantic Similarity

- For pairs \( (u, v) \):
  - Sample trajectories conditioned on \( u \) and \( v \) as above.  
  - Approximate the divergence (e.g., KL, L1, or cosine-similarity in likelihood space).  
  - Use these divergence scores as measures of semantic similarity.

### B. Asymmetric Relation (Entailment, Hypernymy)

- Use likelihood ratio algebraic operations:  
  - Compute \( M_u \) and \( M_v \), then evaluate the asymmetry via conditional likelihoods or set containment measures:  
    \[
    d(M_u, M_v) \quad \text{vs.} \quad d(M_v, M_u)
    \]
  - Alternatively, manipulate likelihood functions \( M_{u}^{|t|} \), \( M_{v}^{|t|} \) with the likelihood algebra outlined.

### C. Multimodal Extension

- For images:
  - Process images with the multimodal model, generate trajectories by appending modality-specific prompts (e.g., "Describe this image...").  
  - Generate samples as above, compare distributions with text prompts or other images to assess cross-modal similarity.

---

## 4. Hyperparameters and Implementation Details

| Hyperparameter | Typical Range | Description |
|------------------|----------------|--------------|
| \( n \)          | 20-50          | Trajectories sampled per prompt. |
| \( m \)          | 10-20 tokens   | Max continuation length for sampling. |
| Sampling Temperature \( \lambda \) | 0.5-1.0 | Controls diversity of sampled trajectories. |
| Likelihood normalization parameter \( \tau \) | 0.5 | To smooth likelihoods (see likelihood normalization). |
| Distance metrics | various (KL, cosine, L1, L2, Hellinger) | To quantify divergence between meaning distributions. |

---

## 5. Evaluation Metrics

- **Semantic Textual Similarity (STS):**  
  - Compute correlations (Spearman's rank, Pearson) between similarity scores obtained from likelihood-based divergence and human annotations.  
- **Entailment & Containment:**  
  - Use algebraic likelihood relations to predict entailment direction.  
  - Measure accuracy or ranking correlation against known labels (e.g., WordNet hypernym/hyponym relations).  
- **Multimodal similarity:**  
  - Image-image, image-text similarity scores compared to human or dataset annotations.

---

## 6. Reproducibility & Implementation notes

- **Sampling procedures:** Random seed control for reproducibility.  
- **Likelihood computations:** Use efficient batch processing of model logits.  
- **Normalization:** Carefully normalize likelihood or likelihood ratios for stable comparison.  
- **Scaling:** Test with different model sizes (e.g., GPT-2, GPT-2-XL, LLaMA variants) to examine model scaling effects.

---

## 7. Next steps

- Finalize hyperparameter choices via validation (e.g., on STS datasets).  
- Develop utility functions for sampling, likelihood evaluation, and divergence computation.  
- Prepare datasets: prompt pairs, multimodal inputs, WordNet relations, and human annotations for evaluation.

---

This plan provides the detailed blueprint needed to implement the pipeline: from sampling trajectories, computing likelihood functions, representing meanings, to quantifying relations and similarity—fully aligned with the methodology and experiments of the paper.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular Python system leveraging transformers (HuggingFace), datasets (HuggingFace, NLTK, or custom loaders), and numerical libraries (NumPy, SciPy). The core components include: 1) Loading and wrapping pre-trained autoregressive models for likelihood evaluation and sampling, 2) Sampling trajectories conditioned on prompts, 3) Computing likelihood ratios and divergences for semantic similarity and relation inference, 4) Dataset handling for text pairs, WordNet relations, and multimodal inputs, 5) Evaluation modules for correlation, entailment, and hypernym predictions. The main script 'main.py' orchestrates configuration, model setup, sampling, inference, and evaluation. Hyperparameters are configurable via a config or argparse.",
    "File list": [
        "main.py",
        "model.py",
        "dataset_loader.py",
        "sampling.py",
        "likelihood.py",
        "distance.py",
        "evaluation.py",
        "utils.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "classDiagram\n    class ModelWrapper {\n        +__init__(model_name: str, model_type: str)\n        +sample_trajectories(prompt: str, n: int, max_length: int, temperature: float, seed: int) -> List[Trajectory]\n        +compute_log_likelihood(sequence: List[str], prompt: str) -> float\n        +get_token_probabilities(sequence: List[str], prompt: str) -> List[float]\n    }\n    class DatasetLoader {\n        +__init__(dataset_name: str, config: dict)\n        +load_pairs() -> List[Tuple[str, str]]\n        +load_wordnet_relations() -> List[Tuple[str, str, str]]\n        +load_multimodal_inputs() -> List[Dict]\n    }\n    class Trajectory {\n        +sequence: List[str]\n        +log_likelihood_score: float\n    }\n    class DivergenceCalculator {\n        +compute_divergence(dist_type: str, ml1: List[float], ml2: List[float]) -> float\n        +approximate_expectation(samples: List[Trajectory], model1: ModelWrapper, model2: ModelWrapper) -> float\n    }\n    class Evaluation {\n        +__init__(prediction_scores: List[float], labels: List[float])\n        +calculate_spearman() -> float\n        +calculate_accuracy(binary_predictions: List[int], labels: List[int]) -> float\n        +predict_entailment(model1: ModelWrapper, model2: ModelWrapper, pairs: List[Tuple[str, str]]) -> List[int]\n        +predict_hypernymy(model1: ModelWrapper, model2: ModelWrapper, word_pairs: List[Tuple[str, str]]) -> List[int]\n    }\n\nRelationships:\nModelWrapper --* DatasetLoader\nMain --* ModelWrapper\nMain --* DatasetLoader\nMain --* Sampling (calls sample_trajectories)\nMain --* DivergenceCalculator\nMain --* Evaluation\nSampling --* Trajectory\nLikelihood --* DivergenceCalculator\nEvaluation --* compute methods\n",
    "Program call flow": "sequenceDiagram\n    participant C as Config\n    participant M as ModelWrapper\n    participant D as DatasetLoader\n    participant S as Sampling Module\n    participant F as Likelihood Computation\n    participant Dist as DivergenceCalculator\n    participant E as Evaluation\n    Main->>C: load configuration\n    Main->>M: initialize with model_name, model_type\n    Main->>D: load datasets (pairwise prompts, WordNet relations, multimodal inputs)\n    D-->>Main: return datasets\n    Main->>M: sample trajectories for each prompt (n=n_samples, max_length=m, temperature=λ, seed=seed_value)\n    M-->>Main: trajectories list\n    Main->>F: compute likelihood scores for trajectories with optional normalization\n    F-->>Main: likelihood scores\n    Main->>Dist: choose divergence type (KL, L1, etc.)\n    Main->>Dist: approximate divergence between distributions (mean likelihoods, samples)\n    Dist-->>Main: divergence scores\n    Main->>E: evaluate similarity with human annotations (correlation metrics)\n    Main->>E: run entailment and hypernym prediction logic\n    E-->>Main: output metrics and predictions\n    Main-->>User: print/report the results\n",
    "Anything UNCLEAR": "Clarification needed on specific implementation details for the likelihood normalization step, e.g., the exact use of parameter τ, and whether to normalize likelihoods across all trajectories before divergence computation. Also, confirm the expected prompt formats for multimodal inputs and how to handle large models with batching and GPU resources."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "transformers==4.29.1",
        "datasets==2.11.0",
        "numpy==1.23.5",
        "scipy==1.10.1",
        "pandas==1.5.3",
        "tqdm==4.65.0",
        "torch==2.0.0",
        "PyYAML==6.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Implements the orchestrator, loading configurations, setting up models, datasets, and hyperparameters, and calling sampling and evaluation routines."
        ],
        [
            "dataset_loader.py",
            "Defines DatasetLoader with methods load_prompt_pairs(), load_wordnet_relations(), load_multimodal_inputs(), which load datasets and format them appropriately for prompt-based sampling and relation inference."
        ],
        [
            "model.py",
            "Provides ModelWrapper class: initializes autoregressive models (e.g., GPT-2, LLaMA, multimodal models), methods to sample trajectories (sample_trajectories()), compute sequence likelihoods (compute_log_likelihood()), and token probabilities (get_token_probabilities())."
        ],
        [
            "sampling.py",
            "Contains functions to generate trajectories via sampling, manage random seed control, temperature scaling, and maximum sequence length. Uses ModelWrapper's sampling capabilities."
        ],
        [
            "likelihood.py",
            "Implements likelihood computation functions, normalization (e.g., likelihood normalization with parameter τ), and likelihood ratio calculations for divergence measures between distributions derived from trajectories."
        ],
        [
            "distance.py",
            "Contains divergence calculation functions: KL, L1, L2, and other distance measures; functions to approximate model divergence using samples based on likelihood scores."
        ],
        [
            "evaluation.py",
            "Implements evaluation classes and functions for computing Spearman correlation, classification accuracy (entailment/hypernym prediction), and similarity metrics based on likelihood divergence scores."
        ],
        [
            "utils.py",
            "Provides utility functions: command-line argument parsing, hyperparameter setup, batching utilities, seed control, and data formatting."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "model.py",
        "sampling.py",
        "likelihood.py",
        "distance.py",
        "evaluation.py",
        "utils.py",
        "main.py"
    ],
    "Full API spec": "OpenAPI spec is not directly required, as this is an internal module system, but if any API is needed for dataset fetching or external evaluation, define GET endpoints for dataset info, POST for evaluation results, and sample generation. Detailed APIs should include:\n- SampleTrajectory(sample_params)\n- ComputeLikelihood(sequence, prompt)\n- ComputeDivergence(distribution1, distribution2)\n- LoadDataset(dataset_name)\n- EvaluateSimilarity(results)\n- PredictRelations(prompt_pairs, method)\n- LoadModel(model_name, model_type)\nAll APIs should follow REST conventions and include parameters for model name, hyperparameters, dataset identifiers, and evaluation metrics.",
    "Shared Knowledge": "Common utilities include seed control, batching functions, configuration management, and likelihood normalization methods. Dataset loader functions should be standardized for prompt pairs, WordNet relations, and multimodal inputs to ensure consistent input formatting.",
    "Anything UNCLEAR": "Clarification needed on the exact API endpoints for external dataset sources and evaluation result submissions. Also, confirm whether GPU acceleration is required and whether any specific hardware configuration is preferred for large models."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
model:
  name: "gpt2-large" # default model name; modify as needed
  type: "transformers" # model type for wrapper

sampling:
  num_trajectories: 20   # n: number of sampled trajectories per prompt
  max_length: 20         # m: maximum tokens per trajectory
  temperature: 1.0       # lambda: sampling temperature
  seed: 42               # random seed for reproducibility

divergence:
  type: "log_l1"         # divergence measure: log L1, KL, etc.
  likelihood_normalization_tau: 0.5  # tau: likelihood normalization parameter

evaluation:
  batch_size: 32         # batch size for evaluation
  metrics:
    - "spearman"       # compute Spearman correlation
    - "accuracy"       # for entailment/hypernym predictions

dataset:
  prompt_pairs_path: "data/prompt_pairs.json"       # path to prompt pair dataset
  wordnet_relations_path: "data/wordnet_relations.json" # path to WordNet relation data
  multimodal_data_path: "data/multimodal_inputs.json"   # path to multimodal inputs
  
hyperparameters:
  learning_rate: 5e-5     # default learning rate; no fine-tuning detailed in paper
  batch_size: 16          # default batch size for sampling if applicable
  epochs: 3               # epochs for fine-tuning scenarios; not detailed in paper, placeholder

misc:
  model_device: "cuda"   # device: 'cuda' or 'cpu', depending on hardware
  save_model_path: "models/"     # directory to save trained/fine-tuned models
  verbose: true                      # verbose logging flag
```

---

**Note:** Full configuration is available in `planning_config.yaml`
