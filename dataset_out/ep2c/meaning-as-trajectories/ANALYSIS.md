# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

# Logic Analysis for dataset_loader.py

## Purpose:
`dataset_loader.py` is responsible for defining the `DatasetLoader` class, which will facilitate loading and preprocessing datasets used across experiments in the paper. This includes datasets for prompt pairs (for semantic similarity evaluation), WordNet hyponym/hypernym relations, and multimodal inputs (images and captions). The loader should produce data in formats directly compatible with the sampling, inference, and evaluation modules.

---

## High-Level Design:
- **Class:** `DatasetLoader`
- **Initialization:**  
  - Accept dataset identifiers (via paths or dataset names in the configuration).  
  - Accept additional configuration parameters if needed (e.g., dataset splits, filtering options).  
- **Methods:**  
  - `load_prompt_pairs()`: Load question pairs or sentence pairs for semantic similarity tasks.  
  - `load_wordnet_relations()`: Load paired WordNet relations (hyponym/hypernym).  
  - `load_multimodal_inputs()`: Load image-caption pairs or data necessary for multimodal experiments.  
  
- **Outputs:**  
  - Data should be returned as lists (or iterable generators) of suitably formatted items, e.g., tuples or dictionaries, ready to be fed into sampling routines or evaluation modules.

---

## Dataset-specific details:

### 1. load_prompt_pairs()
- **Input:** Path to JSON or other structured dataset (from config).  
- **Content:** Pairs of texts or prompts designed for semantic similarity evaluation.  
- **Format:**  
  - List of tuples `(prompt1, prompt2)` or dictionaries with keys:  
    ```python
    [{"prompt1": str, "prompt2": str, "label": float (human similarity score)}]  
    ```
- **Loading process:**  
  - Parse JSON file.  
  - Ensure encoding is consistent.  
  - For each pair, extract the text inputs.  
  - Optionally, include the human similarity score as a label for evaluation.  
- **Preprocessing:**  
  - Clean whitespace if necessary.  
  - Standardize prompt formats (e.g., add punctuation if required by the methodology).  
- **Return:**  
  - List of `(prompt1, prompt2, label)` tuples or dicts.

### 2. load_wordnet_relations()
- **Input:** Path to JSON or structured list of relations.  
- **Content:**  
  - Word pairs with relations like hyponym or hypernym.  
  - Example: `[{"word1": str, "word2": str, "relation": "hyponym" | "hypernym"}]`  
- **Process:**  
  - Parse JSON/data.  
  - For each pair, extract words and relation label.  
  - (Optional) Filter pairs to a predefined set of interest or specific relation types.  
- **Preprocessing:**  
  - Ensure words are formatted consistently.  
  - Possibly map relation strings to binary labels (`1` for hyponym, `0` for hypernym).  
- **Return:**  
  - List of tuples `(word1, word2, relation_label)`.

### 3. load_multimodal_inputs()
- **Input:** Path to JSON or structured dataset (from config).  
- **Content:**  
  - List of dict objects containing image and caption data, e.g.:  
    ```python
    [{"image_path": str, "caption": str}, ...]
    ```
  - Or, if images are preprocessed, feature vectors.  
- **Process:**  
  - Read image paths or features.  
  - Load images via an image loader (e.g., PIL or OpenCV).  
  - Verify the caption text.  
- **Preprocessing:**  
  - Resize or normalize images if necessary, or leave to be handled downstream.  
  - Format prompt strings consistently: e.g., for images: `"Describe this image: <image>"`; for captions: no change or formatted as: `"This is a caption: <caption>"`.  
- **Return:**  
  - List of dictionaries with keys: `{"image": <image object/loaded data>, "caption": <text>, "prompt": <formatted prompt string>}`.

---

## Implementation considerations:

- **File formats:**  
  - JSON is standard; ensure parsing robustness.  
  - Support for various dataset schemas should be flexible.  
- **Error handling:**  
  - Missing data entries (e.g., missing images, captions, or relations).  
  - Invalid JSON or malformed entries should be caught with exceptions or logged warnings.  
- **Efficiency:**  
  - For large datasets, use lazy loaders or generators if possible.  
  - Use caching or memory-mapping for images to avoid memory overload.  
- **Compatibility:**  
  - Ensure that loaded prompts or data are compatible with the sampling routines: e.g., already tokenized or raw text?  
  - Presumably, leave tokenization to be handled later to maintain uniformity and flexibility.

---

## Connection with other modules:

- **Sampling module:**  
  - Provide prompt text (or image prompts) to generate trajectories.  
- **Evaluation module:**  
  - Retrieve labels to compare with predictions for correlation or classification accuracy.  
- **Multimodal:**  
  - Prepare prompts or input formats matching the prompts used in the main methodology (“Describe this image,” etc.)  
  - Possibly serve as input to `ModelWrapper` for trajectory sampling.

---

## Summary:

| Functionality | Implementation details | Expected data output format | Caveats / Notes |
|---|---|---|---|
| load_prompt_pairs() | Read JSON, extract prompt pairs, optional human scores | List[Tuple[str, str, float]] | Ensure normalization, punctuation consistency |
| load_wordnet_relations() | Read JSON, extract word pairs and relation types | List[Tuple[str, str, int]] | Map relation string to binary; filter as needed |
| load_multimodal_inputs() | Load image paths with image processing, extract captions, compose prompts | List[Dict] with keys: 'image', 'caption', 'prompt' | Maintain uniform prompt formats for model input |

---

## Final notes:

- The loader should be easy to extend if new datasets or formats are added later.  
- The returned data structures should be compatible with downstream functions: e.g., sampling trajectory functions expect prompts (prompt strings) and labels (for evaluation).  
- Consider adding optional parameters in class initialization for dataset filtering (e.g., only specific relations or splits).

This thorough analysis will guide precise implementation of `DatasetLoader` to meet the experimental and methodological requirements described in the paper and the system design.

## distance.py

# Logic Analysis for distance.py

## Purpose
The `distance.py` module will implement functions to compute divergence or distance measures between two *distributions* over text continuations (trajectories), which are approximated via sampled likelihood scores from autoregressive models. These measures include KL divergence, L1 distance, L2 distance, and potentially other metrics like Hellinger distance, depending on implementation needs.

Given the methodology, the core idea is to compare two *model distributions* \( M_u \) and \( M_v \) by using a set of sampled trajectories \( t_i \) (from the sampling module) and their *likelihood scores* \( M_{u}(t_i) \), \( M_{v}(t_i) \).

# 1. Core Inputs and Outputs

### Inputs:
- Two sets of likelihood scores:
  - \( \mathbf{ml}_u = [\log M_u(t_1), \log M_u(t_2), ..., \log M_u(t_n)] \)
  - \( \mathbf{ml}_v = [\log M_v(t_1), \log M_v(t_2), ..., \log M_v(t_n)] \)

These can be vectors of *log-likelihoods* corresponding to the same set of sampled trajectories \( t_i \).

- Alternatively, given the sampling process, you might have the likelihood values \( M_u(t_i) \) and \( M_v(t_i) \). However, working in log-space is numerically more stable.

- The *probability distributions* are implicitly represented by these likelihoods over the same sample set.

### Outputs:
- The divergence or distance score (float), e.g., scalar value representing how dissimilar the two distributions are, such as:
  - KL divergence
  - L1 distance
  - L2 distance
  - Other measures if implemented

## 2. Key Steps & Logic

### A. Normalization of likelihoods
- As likelihood scores can vary significantly, normalization is crucial.
- Based on the configuration parameter `likelihood_normalization_tau` (\(\tau\)), apply a *power normalization*:
  
  \[
  M_u^{n}(t) = \frac{ (M_u(t))^\tau }{\sum_{t'} (M_u(t'))^\tau}
  \]
  
- Similar for \( M_v \).
- Alternatively, since we often work with *log likelihoods*, the normalization can be done in log-space:
  
  \[
  \log M_u^{n}(t) = \tau \times \log M_u(t) - \log Z_u
  \]
  
  where
  
  \[
  Z_u = \sum_{t'} (M_u(t'))^\tau
  \]
  
  can be computed in the normalizing constant.

### B. Computing divergence
- **KL divergence \( D_{KL}(P || Q) \):**
  
  \[
  D_{KL}(M_u || M_v) = \sum_{i} p_u(t_i) \times \left( \log p_u(t_i) - \log p_v(t_i) \right)
  \]
  
  - Approximate via samples:  
    - Convert likelihoods to normalized probabilities \( p_u(t_i) \) and \( p_v(t_i) \).  
    - Estimate the divergence:
    
      \[
      \hat{D}_{KL} = \sum_{i} p_u(t_i) \times (\log p_u(t_i) - \log p_v(t_i))
      \]
      
- **L1 and L2 distances:**
  
  \[
  d_{L1} = \sum_{i} | p_u(t_i) - p_v(t_i) |
  \]
  
  \[
  d_{L2} = \sqrt{\sum_{i} (p_u(t_i) - p_v(t_i))^2}
  \]

- **Symmetric versions:**
  
  - For KL, often compute symmetrized form:
  
    \[
    D_{KL}^{sym}(P, Q) = \frac{1}{2}(D_{KL}(P || Q) + D_{KL}(Q || P))
    \]
    
- For other measures such as Hellinger:
  
  \[
  H^2(P, Q) = \frac{1}{2} \sum_{i} (\sqrt{p_u(t_i)} - \sqrt{p_v(t_i)})^2
  \]
  
  which can be computed if probabilities are normalized.

### C. Expectation Approximation
- The divergence functions approximate the expectation over the joint distribution by summing over the sampled trajectories.
- The sample-based approximation:
  
  \[
  \text{divergence} \approx \frac{1}{N} \sum_{i=1}^{N} | \log M_u(t_i) - \log M_v(t_i) |
  \]
  
  for L1, or similar for other divergences with the corresponding probability weights.

### D. Function signature
- Functions like:
  
  ```python
  def compute_kl_divergence(log_m_u: np.ndarray, log_m_v: np.ndarray, tau: float) -> float
  def compute_l1_distance(log_m_u: np.ndarray, log_m_v: np.ndarray, tau: float) -> float
  def compute_l2_distance(log_m_u: np.ndarray, log_m_v: np.ndarray, tau: float) -> float
  ```

- Inputs: arrays of log likelihood values, normalization parameter (\(\tau\)).

- Optional: the sample set's probability weights after normalization.

## 3. Detailed Considerations

### A. Log-space stability
- Compute all likelihoods in log-space for numerical stability.
- Use `np.logaddexp` when summing in log-space, or convert to probabilities after normalization.

### B. Normalization
- Implement a helper function to normalize likelihoods in log-space:
  
  ```python
  def normalize_log_likelihoods(log_likelihoods: np.ndarray, tau: float) -> np.ndarray:
      # Convert to normalized probabilities
      scaled = tau * log_likelihoods
      max_log = np.max(scaled)
      shifted = scaled - max_log
      probs = np.exp(shifted)
      probs /= np.sum(probs)
      return probs
  ```
  
- For divergence computations, use these normalized probabilities.

### C. Handling arbitrary sample sizes
- Ensure the functions handle arbitrary number of samples \( n \), consistent array sizes.

### D. Symmetry considerations
- For symmetric divergences, combine divergence of \( M_u || M_v \) and \( M_v || M_u \).

---

## 4. Error Handling & Edge Cases

- When likelihoods are extremely small or zero, work in log space.
- When the normalized probabilities sum to zero (e.g., all likelihoods very low), handle divide-by-zero: add small epsilon.
- For sample pairs where likelihood scores are noisy or inconsistent, consider smoothing or thresholding.

## 5. Summary
- Implement functions to:
  - Normalize sample likelihoods (via a temperature or τ).
  - Compute divergence metrics based on these probabilities.
- Ensure all calculations are numerically stable.
- Use vectorized operations (NumPy) for efficiency.
- Provide clear API functions with consistent input formats.

---

**This detailed logical plan ensures that the `distance.py` module can reliably compute projections of the divergence between the approximate distributions over trajectories generated by models, utilizing the likelihood scores and sampling methods core to the paper’s methodology.**

## evaluation.py

# Evaluation.py Logic Analysis

This module is responsible for implementing the evaluation classes and functions necessary to quantitatively assess the quality of the distribution-based meaning representations and their derived relations, as defined in the paper. The key evaluation tasks include: 

- Computing semantic similarity scores and correlating with human annotations (e.g., via Spearman correlation).
- Classifying entailment directions between sentence pairs.
- Predicting hyponym/hypernym relations between words.
- Evaluating multimodal similarity scores (images, captions, or cross-modal pairs).

This analysis outlines the core classes, their attributes, methods, flow, and their interactions with other module components, consistent with the paper’s methodology, task specifications, and provided configuration.

---

# 1. Core Classes & Functions

## 1.1. class Evaluation

Purpose:
- Orchestrate the overall evaluation process.
- Compute metrics such as Spearman correlation and classification accuracies.
- Load relevant data, process model outputs, and compare against human annotations or ground truth labels.

Attributes:
- prediction_scores: list or array, of divergence scores between pairs of prompts (e.g., similarity scores based on likelihood divergence).
- labels: list or array, of ground-truth similarity scores, entailment labels, or hyponym/hypernym indicators.

Methods:
- __init__(self, prediction_scores, labels): store inputs.
- calculate_spearman(self): compute Spearman coefficient between stored scores and labels.
- calculate_accuracy(self, binary_predictions, labels): compute classification accuracy.
- evaluate_similarity(self, prompt_pairs, models, divergence_params): 
   + for each pair, generate or retrieve their divergence score.
   + compute correlation with human annotations.
- evaluate_entailment(self, prompt_pairs, models, divergence_params): 
   + for each pair, compute divergence scores in both directions.
   + compare divergence-based entailment prediction against ground truth.
- evaluate_hypernymy(self, word_pairs, models, divergence_params): 
   + for each word pair, compute divergence scores.
   + determine directionality (hyponym vs. hypernym).

## 1.2. Supporting functions

- compute_divergence(dist_type, log_likelihoods1, log_likelihoods2):  
  + Compute the specified divergence measure (e.g., log L1, KL, L2) between two distributions represented as sequences of log-likelihood scores.  
  + Use pairwise likelihood scores sampled for each prompt/distribution.

- compute_spearman(scores, labels):  
  + Return Spearman correlation coefficient for arrays.

- compute_accuracy(predictions, labels):  
  + Return overall accuracy.

---

# 2. Inputs & Data Preparation

- The evaluation module receives as inputs:

  a) Divergence scores for prompt pairs, generated through the process in likelihood.py (or other sampling modules). This involves comparing sampled trajectories’ likelihoods between two prompts/distributions \( M_u \) and \( M_v \).

  b) Ground-truth labels:
     - Human similarity scores (0–5 scale or scaled) for STS datasets.
     - Entailment labels (entailment / not entailment).
     - Hyponym/hypernym relation indicators (binary).

- For dataset loading:
  - Prompt pairs for semantic similarity evaluation.
  - Word pairs with known hyponym/hypernym relations.
  - Multimodal data for cross-modal evaluation.

- The module uses a configuration or explicit parameters to determine dataset paths, batch sizes for evaluation, and divergence types.

---

# 3. Correlation Evaluation (Semantic Similarity)

Method:
- Given a list of divergence scores for pairs (e.g., from Algorithm 1), and ground-truth similarity scores, apply:
  - `scipy.stats.spearmanr()` to obtain the correlation coefficient.
- To facilitate multiple runs, wrap into `calculate_spearman()`.

Flow:
```
def evaluate_similarity_scores(self, scores, ground_truth_labels):
    return compute_spearman(scores, ground_truth_labels)
```

Notes:
- Preprocessing of human scores may include normalization or scaling.
- The divergence scores are derived from multiple trajectories sampled per pair.

---

# 4. Classification Tasks (Entailment & Hyponym-Hypernym Prediction)

### 4.1. Entailment Prediction (Binary Classification)

Algorithm:
- For each prompt pair \((u, v)\):
  1. Compute two divergence scores:
     - \( d(M_u, M_v) \) (e.g., model from prompt u to v).
     - \( d(M_v, M_u) \).

  2. Decide entailment direction by comparing these scores:
     - If \( d(M_u, M_v) < d(M_v, M_u) \), infer \( u \Rightarrow v \).
     - Else, \( v \Rightarrow u \).

  3. Collect predictions and compare to ground-truth labels.

Evaluation:
- Use `calculate_accuracy()` on predicted labels vs. true labels.
  
Implementation:
```
def evaluate_entailment(self, prompt_pairs, model_wrapper, divergence_params):
    predictions = []
    for u, v in prompt_pairs:
        divergence_uv = compute_divergence(..., model_wrapper, u, v, divergence_params)
        divergence_vu = compute_divergence(..., model_wrapper, v, u, divergence_params)
        pred = 1 if divergence_uv < divergence_vu else 0
        predictions.append(pred)
    return predictions
```

### 4.2. Hyponym/Hypernym Prediction

Algorithm:
- For word pairs \((u, v)\):
  1. Compute distributions \(\overline{M}_u\), \(\overline{M}_v\).
  2. Calculate the distances \( d(\overline{M}_u, \overline{M}_v) \), \( d(\overline{M}_v, \overline{M}_u) \).

  3. Use the outlined *Hyponym Test*:
     - If \( d(\overline{M}_v, \overline{M}_u) < d(\overline{M}_u, \overline{M}_v) \), predict \( v \) is hyponym of \( u \).
     - Else, \( u \) is hyponym of \( v \).

Evaluation:
- Use `calculate_accuracy()` with true labels indicating hyponym/hypernym relation.

---

# 5. Multimodal & Cross-Modal Similarity

- Use the same divergence computations between the distributions associated with images, captions, or combinations.
- For multimodal data, the same logic applies but with prompts including modality-specific context.

---

# 6. Implementation Details & Integration

- Instantiate `Evaluation` objects with lists of divergence scores and corresponding labels.
- Use hyperparameters (like divergence type) from `config.yaml`:
  - divergence.type (e.g., `"log_l1"`).
- Each method should:
  - Handle batch processing to improve efficiency.
  - Optionally accept other divergence settings such as normalization (via likelihood_normalization_tau).

- Output:
  - Metrics (Spearman coefficient, accuracy).
  - Predicted relation directions (entailment, hyponym/hypernym).

---

# 7. Summarized Pseudocode Snippet

```python
class Evaluation:
    def __init__(self, scores: List[float], labels: List[float]):
        self.scores = scores
        self.labels = labels

    def compute_spearman(self):
        return scipy.stats.spearmanr(self.scores, self.labels).correlation

    def compute_accuracy(self, preds: List[int], labels: List[int]):
        correct = sum(p == l for p, l in zip(preds, labels))
        return correct / len(labels)

    def evaluate_similarity(self):
        return self.compute_spearman()

    def evaluate_entailment(self, divergence_uvs, divergence_vus):
        preds = [1 if d_uv < d_vu else 0 for d_uv, d_vu in zip(divergence_uvs, divergence_vus)]
        return self.compute_accuracy(preds, true_labels)

    def evaluate_hypernymy(self, divergence_us, divergence_vs, true_labels):
        preds = [1 if d_v < d_u else 0 for d_u, d_v in zip(divergence_us, divergence_vs)]
        return self.compute_accuracy(preds, true_labels)
```

---

# 8. Additional Considerations

- Ensure divergence computations and likelihood normalization are consistent with the likelihood.py implementations.
- Batch process divergence calculations to handle large datasets efficiently.
- Track and report confidence intervals or statistical significance if needed.
- Possibly implement methods to aggregate divergence scores over multiple trajectory samples, as described.

---

**Summary:**  
The `evaluation.py` module encapsulates all metrics for assessing how well the distributional meaning representations (via likelihood-based divergences) correspond to human judgments and linguistic relations, following the methodology in the paper. It leverages divergence measures, statistical correlations, and classification metrics, all tightly integrated with the sampling and likelihood evaluation modules.

This logical blueprint ensures fidelity to the paper's approach, configuration parameters, and experimental design.

## likelihood.py

**Likelihood.py Logical Analysis**

---

### Purpose and Responsibilities:
The primary purpose of likelihood.py is to implement functions that enable the computation, normalization, and comparison of likelihood scores derived from sampled trajectories conditioned on prompts, according to the methodology outlined in the paper. Specifically, it should:

- Compute the likelihood of a sequence (trajectory) given a prompt, as estimated by a pre-trained autoregressive model.
- Normalize likelihood scores across multiple trajectories to produce comparable measures, incorporating the normalization parameter τ.
- Calculate divergence (distance) metrics between model distributions \( M_u \) and \( M_v \), using their sampled likelihood scores, to measure semantic similarity or asymmetric relations (entailment, hyponym/hypernym).
- Support approximate expectation calculations over trajectories for divergence estimation.

---

### Input Data:
- Sequences: List or batch of token sequences, each sequence being a trajectory (list of tokens or string).
- Prompts: The prompt string \( s \), which the model extends via sampling.
- Likelihood scores: Derived from model outputs, specifically the probabilities or log probabilities of the sampled sequences.
- Hyperparameters: τ (normalization parameter), trajectory scores, and divergence type options.

---

### Core Functions and Their Logic:

#### 1. `compute_log_likelihood(sequence, prompt)`
- **Input:** 
  - sequence: list of tokens (e.g., ['The', 'dog', 'barked'])
  - prompt: the prompt string \( s \)
- **Process:**
  - Tokenize the prompt \( s \).
  - Tokenize the sequence \( sequence \).
  - Use the model to get the logits for each position, conditioned on prompt + previous tokens.
  - Obtain the next-token probability distributions.
  - For each token in sequence, extract the probability of that token given previous tokens.
  - Compute the sum of logs of these probabilities.
- **Output:** 
  - Log-likelihood of the sequence conditioned on prompt, i.e.,
    \[
    \log P(sequence | prompt) = \sum_{i=1}^{|sequence|} \log p(a_i | s, a_{<i})
    \]
- **Notes:**
  - Implement efficient batching if needed.
  - Handle numerical stability.

#### 2. `calculate_likelihood(sequence, prompt)`
- **Input:** 
  - sequence: list of tokens
  - prompt: string \( s \)
- **Process:**
  - Invoke `compute_log_likelihood`.
  - Return likelihood as \(\exp(\text{log likelihood})\).

#### 3. `normalize_likelihoods(likelihoods, tau)`
- **Input:**
  - likelihoods: list or array of raw likelihoods
  - tau: normalization exponent hyperparameter (float, typically 0.5 per config)
- **Process:**
  - Apply normalization: 
    \[
    M(t)^{\text{norm}} = \frac{(M(t))^\tau}{\sum_{t'} (M(t'))^\tau}
    \]
  - Use the likelihood scores raised to the power τ.
  - Compute the sum over all trajectories' scores for normalization.
  - Return normalized scores which sum to 1.
- **Output:**
  - normalized likelihood scores for each trajectory.

#### 4. `compute_divergence(distribution1, distribution2, dist_type)`
- **Input:**
  - distribution1: array of likelihood scores (after normalization) for set in M_u
  - distribution2: array of likelihood scores for set in M_v
  - dist_type: string specifying divergence metric (e.g., 'log_l1', 'kl', 'log_l2')
- **Process:**
  - Depending on `dist_type`, compute the divergence.
  - For 'log_l1': 
    \[
    d = \sum | \log M_u(t) - \log M_v(t) |
    \]
  - For 'kl' (Kullback-Leibler): 
    - Ensure distributions are properly normalized.
    - Compute \(\sum p \log (p / q)\).
  - For 'log_l2': 
    \[
    \sqrt{\sum (\log M_u(t) - \log M_v(t))^2}
    \]
- **Notes:**
  - Handle cases where likelihoods are small or zero: add small epsilon if necessary.
  - Use logs or direct likelihoods according to the specific divergence.

#### 5. `approximate_expectation(samples, model1, model2, divergence_type)`
- **Input:**
  - samples: list of sampled trajectories (with likelihood scores) for both prompts
  - models: model wrappers for each prompt
  - divergence_type: string
- **Process:**
  - For each trajectory \( t_i \) in samples:
    - Compute likelihood scores \( M_u(t_i) \), \( M_v(t_i) \).
  - Normalize the likelihoods with the tau parameter.
  - Use the normalized likelihoods to approximate the integral in the divergence metrics.
  - Return the divergence score.
- **Output:**
  - Estimated divergence value between the two distributions.

---

### Implementation Specifics:
- **Likelihood score calculations:**  
  - Use model logits to obtain token probabilities at each step.
  - Compute per-sequence log-likelihood via sum of per-token log probabilities for each trajectory sample.
- **Numerical stability:**  
  - Apply small epsilon to probabilities before logs if needed.
  - Logarithmic computations to prevent underflow.
- **Efficiency considerations:**  
  - Batch processing multiple trajectories at once.
  - Use device (GPU) tensors for speed.
- **Parameter handling:**  
  - Read τ from config: likelihood_normalization_tau.
  - Support divergence types via string labels.

---

### Summary:
`likelihood.py` functions should be built to:

- Accept batch of sequences and prompts.
- Use model interfaces (from model.py) to compute probabilities.
- Calculate likelihoods (probability or log-likelihood).
- Normalize likelihood scores across sampled trajectories using τ.
- Compute divergence metrics suitable for distribution comparisons.
- Support approximate integration over the estimated distribution (sampling-based).

This design ensures faithful implementation of the method described in the paper, allowing measures of semantic similarity, entailment, and containment to be computed directly from likelihood distributions over trajectories.

---

### Final Notes:
- Confirm the structure of the `Trajectory` object or data container; include likelihood score as attribute.
- Carefully handle edge cases (e.g., zero probabilities).
- Provide clear interfaces for calling functions from main code or evaluation modules.
- Ensure functions are modular and documented for clarity and reusability.

This concludes the detailed logic analysis to guide the coding of likelihood.py aligned with the paper’s methodology.

## main.py

### Logic Analysis for `main.py`

**Purpose:**  
`main.py` serves as the central orchestrator orchestrating all steps: configuration loading, model initialization, dataset loading, trajectory sampling, divergence calculation, evaluation, and reporting of results. It coordinates the entire pipeline to reproduce the experiments outlined in the paper aligned with the provided design, datasets, and hyperparameters.

---

### 1. Initialization and Setup

**a. Load configuration**  
- Parse `config.yaml` using `PyYAML` to extract all settings:
  - Model details (`model.name`, `model.type`)
  - Sampling hyperparameters (`sampling.num_trajectories`, `sampling.max_length`, `sampling.temperature`, `sampling.seed`)
  - Divergence measure (`divergence.type`, `divergence.likelihood_normalization_tau`)
  - Evaluation settings (`evaluation.batch_size`, `evaluation.metrics`)
  - Dataset paths (`dataset.prompt_pairs_path`, `dataset.wordnet_relations_path`, `dataset.multimodal_data_path`)
  - Misc options: device selection, verbosity

**b. Set random seed**  
Use torch, numpy, and python `random` to ensure reproducibility via `sampling.seed`.

---

### 2. Model Initialization

**a. Instantiate ModelWrapper**  
- Use `model.name` and `model.type` to initialize the specific model (`transformers` or other frameworks).  
- Load pre-trained weights via HuggingFace or custom interface.  
- Set device (from `misc.model_device`), move model to GPU/CPU accordingly.

**b. Verify tokenizer**  
- Ensure proper tokenization consistent across scripts, wrapping tokenizer for batch processing if needed.

---

### 3. Dataset Loading

**a. Load datasets via DatasetLoader**  
- *Prompt pairs*: for semantic similarity experiments (e.g., sentence pairs from STS datasets).  
- *WordNet relations*: for hyponym/hypernym relation tests.  
- *Multimodal inputs*: images or image descriptions for multimodal relation experiments.

**b. Data format handling**  
- Read JSON, CSV, or other formats specified in paths.  
- Pre-process data into suitable prompt formats.  
- For multimodal data, prepare sample prompts combining text and modality-specific instructions.

---

### 4. Sampling Trajectories from the Model

**a. For each prompt pair or individual prompt:**  
- Call `ModelWrapper.sample_trajectories()` with parameters:  
  - `prompt`: string prompt (from dataset)  
  - `n`: number of trajectories (`sampling.num_trajectories`)  
  - `m`: max tokens length (`sampling.max_length`)  
  - `λ`: temperature (`sampling.temperature`)  
  - `seed`: for reproducibility.

**b. Parallelization**  
- Use batching or multiprocessing if available to expedite sampling, respecting hardware (GPU memory, batch size constraints).

**c. Store trajectories**  
- Keep list of sequences with their associated log-likelihoods or likelihood scores (`Trajectory` objects).  
- If needed, cache samples for repeated computations, particularly in relation tasks.

---

### 5. Likelihood and Divergence Calculation

**a. Compute likelihood scores**  
- For each trajectory, compute  
  \[
  \log P(t | s) = \sum_{i=1}^m \log p(a_i | s, a_{<i})
  \]  
  via `Likelihood.compute_log_likelihood(sequence, prompt)`.

**b. Likelihood normalization** (if used):  
- Normalize likelihoods using the \( \tau \) parameter (e.g., via `likelihood_normalization_tau`) for stable divergence calculation per the paper.  
- Possible normalization: \( M_{\text{norm}}(t) = \frac{M(t)^\tau}{\sum_{t'} M(t')^\tau} \).

**c. Divergence approximation**  
- For each pair, pass the samples and likelihood scores to `DivergenceCalculator.compute_divergence()` with the specified divergence type (`log_l1`, `kl`, etc.).  
- Implement expectation approximation as per the paper's Monte Carlo method, aggregating across pairs of sampled trajectories.

---

### 6. Semantic Similarity and Relation Tests

**a. Similarity computation:**  
- For each input pair (text or multimodal),  
  - sample trajectories for \( u \) and \( v \).  
  - compute divergence between their distributions (`distance.py`).  
  - Store the divergence score.

**b. Entailment and hypernym/hyponym inference:**  
- For pairs \( (u,v) \), compute the divergence in both directions: \( d(M_u, M_v) \) and \( d(M_v, M_u) \).  
- Use empirical criteria or thresholds (e.g., as in the paper) to classify entailment or hyponym relations.

---

### 7. Evaluation and Metrics

**a. Semantic textual similarity (STS):**  
- Collect divergence scores for all test pairs.  
- Convert divergence to similarity score (for SS, lower divergence = higher similarity).  
- Compute Spearman and Pearson correlations with ground truth annotations using `evaluation.py`.  
- Report metrics.

**b. Entailment accuracy:**  
- Use the divergence scores to predict entailment directions.  
- Calculate classification metrics (accuracy, F1) for entailment/directionality.

**c. WordNet hyponym/hypernym relations:**  
- Apply the same divergence-based containment check to word pairs.  
- Measure accuracy against ground-truth WordNet relations.

**d. Multimodal similarity:**  
- Similar process: sample trajectories for images, captions, or combined inputs.  
- Compute distances and compare with dataset labels or human annotations.

---

### 8. Reporting & Logging

- Log all hyperparameters, runtime diagnostics, intermediate divergence scores, and evaluation metrics.  
- Optionally visualize hierarchies (WordNet) or embedding spaces if desired, using `matplotlib` or similar libraries.

**a. Verbose option:**  
- When `verbose` is True, print progress updates, intermediate scores, and logs.

**b. Final report:**  
- Summary statistics: average similarity with human scores, ensemble over datasets.  
- Model performance scaling with size, datasets, and task type.

---

### 9. Cleanup & Saving Results

- Save models (if fine-tuning occurs, e.g., future extensions).  
- Save sampled trajectories, divergence matrices, and evaluation metrics in structured files (`csv`, `json`).  
- Handle exception cases gracefully: missing data, API issues, sampling failures.

---

### 10. Additional Considerations & Clarifications

- Ensure all components follow the interface definitions: ModelWrapper, DatasetLoader, etc.  
- Confirm modularity to facilitate testing of components independently.  
- Make the code flexible for different models, datasets, and evaluation metrics based on config.yaml.

---

**Summary:**  
`main.py` acts as the central orchestrator, performing: configuration loading → model setup → dataset loading → trajectory sampling → likelihood evaluation → divergence computation → metrics evaluation → result reporting. All steps are integral to faithfully reproducing the experiments and analyses described in the paper, with strict adherence to model and data interfaces, hyperparameters, and evaluation protocols.

---

**End of Logic Analysis**

## model.py

# Logic Analysis for model.py

This module is responsible for providing an abstraction layer over the pre-trained autoregressive language models (including multimodal models), enabling sampling of trajectories, likelihood computation, and access to token probabilities. It should be designed with flexibility, modularity, and efficiency in mind to support the experimental procedures outlined in the paper.

Below is a comprehensive, step-by-step analysis of the key components, logic flow, required methods, and interactions necessary for implementing 'model.py' and defining the 'ModelWrapper' class.

---

## 1. Objectives of 'model.py'

- **Initialization:**  
  Load and wrap the specified pre-trained model, given parameters from configuration.  
  Handle both standard (e.g., GPT-2, LLaMA) and multimodal models, possibly needing different tokenizers and processing pipelines.

- **Trajectory Sampling:**  
  Generate multiple continuation sequences ('trajectories') conditioned on a prompt, using specified sampling hyperparameters (number of trajectories, maxTokens, temperature, seed).

- **Likelihood Computation:**  
  Calculate the likelihood or log-likelihood of a given sequence (trajectory) conditioned on a prompt (or unconditioned). This includes handling of normalization, especially with the 'likelihood_normalization_tau' parameter.

- **Token Probabilities Retrieval:**  
  Return per-token probabilities for a sequence continuation, which may be useful for divergence calculations or further analysis.

---

## 2. Key Components and Methods

### A. Initialization and Loading Models

- Based on the `'model.name'` and `'model.type'`, load the appropriate pre-trained model architecture and tokenizer:
  - For 'transformers'-based models, use `AutoModelForCausalLM` and `AutoTokenizer` from HuggingFace.
  - Support different models such as GPT-2, LLaMA, Falcon, multimodal (LLaVA) as per configuration.
  
- **Device setup:**  
  Load the model onto the specified `'model_device'`, typically `'cuda'` for GPU or `'cpu'`.

- **Tokenizer handling:**  
  Use the model-specific tokenizer, ensuring that the special tokens (e.g., EOS, BOS) are correctly handled or added if absent.

- **Model configuration:**  
  - Set model to evaluation mode (`model.eval()`) for inference-only use.
  - Enable `torch.no_grad()` during inference to improve efficiency.

- **Multimodal support:**  
  If supported, handle multimodal inputs (images, captions), possibly by special prompt formatting or additional embedding strategies.

### B. Sampling Trajectories: `sample_trajectories()`

- **Input:**  
  - `prompt`: string prompt to condition generation
  - `n`: number of trajectories to sample (from config)
  - `max_length`: maximum number of tokens in each trajectory (from config)
  - `temperature`: sampling diversity (from config)
  - `seed`: for reproducibility

- **Process:**  
  1. Tokenize prompt; prepare input tensor.
  2. Set random seed for reproducibility.
  3. For each trajectory:
     - Generate tokens iteratively until `max_length` or EOS token is produced.
     - Use top-k or nucleus sampling if preferred, but typically just temperature sampling.
     - For each step:
       - Run model to get logits.
       - Apply temperature scaling.
       - Sample next token according to the adjusted probability distribution.
     - Store the resulting token sequence and compute its log-likelihood via `compute_log_likelihood()` or incremental log probability calculation.
  4. Collect all `Trajectory` objects containing sequences and likelihood scores.

- **Output:**  
  - List of `Trajectory` objects: each with sequence (list of tokens/strings) and score (float).

- **Performance note:**  
  - Use batching when sampling multiple trajectories to optimize GPU utilization.
  - Control randomness with seed; may reset the generator/ RNG seed before sampling.

### C. Likelihood Computation: `compute_log_likelihood()`

- **Input:**  
  - sequence: list of tokens or string
  - prompt: string prompt conditioned upon

- **Process:**  
  1. Tokenize sequence; prepare input tensors including prompt tokens.
  2. Run model to get logits for each token position.
  3. Calculate token probabilities \( p(a_i | s, a_{<i}) \).
  4. Logarithmically accumulate likelihoods for the entire sequence.
  5. Return average likelihood (or unnormalized total, depending on use case).

- **Note:**  
  - This should be optimized to handle batch computation if multiple sequences are processed simultaneously.
  - Maintain numerical stability; use log probabilities.

### D. Token Probabilities: `get_token_probabilities()`

- **Input:**  
  - sequence: list of tokens
  - prompt: conditioning prompt

- **Output:**  
  - List of probabilities (or log probs) for each token, conditioned on prompt and previous tokens.

- **Use case:**  
  - For divergence/similarity computations where per-token likelihood is needed.

---

## 3. Hyperparameters & Configuration Usage

- **`sampling.num_trajectories` (`n`)**  
  Controls the number of trajectories generated per prompt, influencing estimate accuracy.

- **`sampling.max_length` (`m`)**  
  Limits the length of each sampled continuation, balancing fidelity and computational cost.

- **`sampling.temperature` (`λ`)**  
  Controls diversity of generated tokens:  
  - Low values (e.g., 0.5) favor more deterministic sampling.  
  - Higher values (e.g., 1.0) promote diversity.

- **Seed for randomness:**  
  Important for reproducibility; be sure to set `torch.manual_seed` and, if using Python's `random`, set seed accordingly.

- **Likelihood normalization tau (`τ`)** in divergence calculations:  
  - Use as a power/exponent to likelihoods during normalization, as specified in the paper (eq. 3).  
  - For example, normalize likelihoods as \( \tilde{M}(t) \propto M(t)^\tau \).

- **Batching and parallelization:**  
  To improve efficiency, sample multiple trajectories in batches, process their likelihoods together using model batch inference.

---

## 4. Implementation details and considerations

- **Model wrappers:**  
  - Encapsulate all model-specific calls within the wrapper.
  - Provide uniform interface for sampling, likelihood, token probability extraction, abstracting away model specifics.

- **Data handling:**  
  - For multimodal inputs, integrate modality-specific prompts into the prompt string.
  
- **Memory management:**  
  - Keep models in evaluation mode; free GPU memory when done.
  - Use `with torch.no_grad()` during inference to optimize.

- **Error handling:**  
  - Handle missing EOS tokens gracefully.
  - Validate input sequences; re-tokenize if necessary.

- **Reproducibility:**  
  - Fix random seed at the start of each sampling run.
  - Record seed in logs for replication.

---

## 5. Summary of Main Methods & Their Signature

```python
class ModelWrapper:
    def __init__(self, model_name: str, model_type: str, device: str):
        # Load model, tokenizer, set device, prepare for inference
        pass
    
    def sample_trajectories(self, prompt: str, n: int, max_length: int, temperature: float, seed: int) -> List[Trajectory]:
        # Generate n trajectories conditioned on prompt
        pass

    def compute_log_likelihood(self, sequence: List[str], prompt: str) -> float:
        # Compute total log-likelihood of sequence conditioned on prompt
        pass
    
    def get_token_probabilities(self, sequence: List[str], prompt: str) -> List[float]:
        # Return per-token probabilities conditioned on prompt
        pass
```

---

## Conclusion

This logic analysis outlines the necessary components, flow, and considerations for implementing 'model.py'. Focus should be placed on:

- Efficient and modular loading of models and tokenizers.
- Robust sampling routines respecting hyperparameters.
- Precise likelihood calculations, including normalization.
- Clear abstraction for downstream computations like divergence, similarity, and relation inference.

Adhering to these principles will ensure the implementation aligns closely with the scientific methodology presented in the paper and facilitates reproducibility and comparability of results.

## requirements.txt

# requirements.txt

# Core dependencies for model loading, sampling, likelihood evaluation, and computations:
transformers==4.29.1        # For loading pre-trained autoregressive models (GPT-2, LLaMA via custom wrappers, multimodal models)
datasets==2.11.0            # For dataset handling (prompt pairs, multimodal inputs)
numpy==1.23.5               # For numerical operations, array manipulations
scipy==1.10.1               # For statistical divergence computations and advanced math functions
pandas==1.5.3               # For data management and dataset manipulation
tqdm==4.65.0                # For progress bars during sampling and evaluation
torch==2.0.0                # Underlying deep learning framework for model inference, GPU acceleration
PyYAML==6.0                 # For configuration parsing from YAML files

# Additional utilities (if used):
# (Note: Strictly none specified for third-party packages beyond core dependencies)

# Ensure reproducibility and seed control:
# (This will be handled programmatically via Python's random, numpy, torch.manual_seed)

# Note:
# - The above packages cover the detailed model, data, and numerical function libraries 
# - No specialized or proprietary external API clients are required as per plan.
# - For multimodal models, ensure the respective model repositories (like LLaVA) are compatible via transformers or custom wrappers.

# Summary:
# The code will set up a modular system fully based on these standard packages, following the interfaces and logic described, enabling sampling, likelihood evaluation, diversity computation, and evaluation metrics as specified in the paper's methodology.

# Additional Remarks:
# - Do not include dependencies unrelated to core logic (e.g., visualization libraries), unless desired.
# - Hardware dependencies (like CUDA) are managed via torch, not a third-party package.
# - For reproducibility, seed management is programmatically handled in the code base, not via external packages.

# End of requirements.txt


## sampling.py

**Logic Analysis for sampling.py**

---

### 1. Purpose and Responsibilities
The `sampling.py` module’s primary role is to provide flexible and robust functions for generating multiple trajectories (sequences of tokens) conditioned on a given prompt, while controlling critical sampling parameters such as the number of trajectories, maximum length, temperature, and seed. It interacts closely with the `ModelWrapper` class to invoke model-specific sampling routines and to ensure reproducibility and consistency across experiments.

---

### 2. Core Inputs and Hyperparameters
- **Inputs:**
  - `prompt` (str): The string (prompt) from which to generate continuations.
  - `n` (int): Number of trajectories to sample (`num_trajectories` from config).
  - `max_length` (int): Max tokens per trajectory (`sampling.max_length`).
  - `temperature` (float): Sampling temperature (`sampling.temperature`).
  - `seed` (int): Random seed for reproducibility (`sampling.seed`).

- **Configuration (from `config.yaml`):**
  - Use the provided hyperparameters unless explicitly overridden in function parameters.
  - Ensure consistency across sampling runs by fixing random seed.

---

### 3. Key Functionalities and Steps

#### 3.1. Seed Control
Establish deterministic sampling:
- Use `torch.manual_seed(seed)` for CPU/GPU randomness.
- If necessary, seed NumPy (`np.random.seed(seed)`) and Python’s built-in `random.seed()` for scope-wide consistency.
- Set seed prior to sampling the trajectories.

#### 3.2. Sampling Trajectories
Implementation choices:
- **Method 1:** Use `ModelWrapper.sample_trajectories()` (recommended), which:
  - Accepts prompt, number of trajectories, max length, temperature, seed.
  - Returns a list of `Trajectory` objects with their generated sequences and likelihood scores.
- **Method 2 (if avoiding direct model calls):** Implement sampling loop:
  - For `i` in range(n):
    - Call model’s generate method with prompt, max_length, temperature, seed=some seed incremented or varied for each trajectory.
    - Compute likelihood score of the generated sequence.
  
**Decision:** Prefer calling `ModelWrapper.sample_trajectories()` to ensure modularity, model abstraction, and consistency.

#### 3.3. Trajectory Representation
- Store each trajectory as:
  - List of tokens (strings), or combined text if necessary.
  - Associated likelihood score (float), computed during sampling.
- Return a list of `Trajectory` objects or a structured dictionary/list containing the sequence and score for each trajectory.

#### 3.4. Likelihood Calculation (if not provided directly by sampling)
- Use `ModelWrapper.compute_log_likelihood(sequence, prompt)` for each trajectory after generation:
  - Compute the total log-likelihood for the trajectory.
  - Store as `log_likelihood_score`.
- Convert to normalized likelihood if needed (for divergence computations).  
  - Possible normalization step: apply `likelihood_normalization_tau`, e.g., smoothing or scaling.

**Note:**  
Sampling with model’s max tokens may introduce bias if the model’s next-token sampling is not stored during generation. Prefer the model’s sampling method to return likelihoods directly.

---

### 4. Implementation Details and Best Practices

- **Parallelization:**  
  - To handle multiple trajectories efficiently, dispatch sampling calls in parallel (threads or processes).  
  - Use batching if the model and API allow (e.g., process multiple prompts together).

- **Sampling temperature:**  
  - Use the `temperature` parameter in the model’s sampling function to control diversity.  
  - Temperatures less than 1.0 produce less diverse, sharper samples; above 1.0 produce more diversity.

- **Sampling randomness:**  
  - To ensure reproducibility, the seed must be fixed at the start, and all model calls must be deterministic given the seed.

- **Trajectory data structure:**  
  - For each trajectory, store:
    ```python
    class Trajectory:
        sequence: List[str]
        log_likelihood_score: float
    ```
  - This simplifies downstream likelihood ratio calculations and divergence measures.

- **Handling maximum length:**  
  - Clip generated sequences to `max_length`.  
  - Stop early if EOS token encountered within the length constraint.

- **Edge Cases:**
  - If the model generates EOS early, truncate trajectory at EOS.
  - Handle cases where no tokens are generated (rare but possible).

---

### 5. Integration with `main.py` and other modules
- `sampling.py` functions will be called within the main control flow:
  - Sampling trajectories for each prompt before divergence or similarity computation.
  - Reused across different tasks (similarity, entailment, hierarchy) as needed.
- Output format must align with `Trajectory` objects or similar.

---

### 6. Pseudocode Skeleton

```python
def sample_trajectories(prompt: str, n: int, max_length: int, temperature: float, seed: int) -> List[Trajectory]:
    # Set seed for reproducibility
    set_seed(seed)
    
    trajectories = []
    for _ in range(n):
        sequence, log_likelihood = model_wrapper.sample_sequence(prompt, max_length, temperature)
        trajectories.append(Trajectory(sequence=sequence, log_likelihood_score=log_likelihood))
    return trajectories
```

Or using a batch/parallel approach:

```python
def sample_trajectories_batch(prompt: str, n: int, max_length: int, temperature: float, seed: int) -> List[Trajectory]:
    # Set seed
    set_seed(seed)
    # Generate n trajectories in batch if model supports
    sequences, log_likelihoods = model_wrapper.sample_batch(prompt, n, max_length, temperature)
    return [Trajectory(seq, log_likelihood) for seq, log_likelihood in zip(sequences, log_likelihoods)]
```

---

### 7. Summary
The `sampling.py` functions must:

- Establish deterministic behavior via seed control.
- Generate multiple continuations per prompt, respecting max length and temperature.
- Store sequences and their likelihood scores precisely, for use in divergence and similarity computation.
- Be flexible enough to handle various prompts and modalities.
- Provide clear and reproducible interfaces for downstream divergence and relation inference modules.

This thorough logic foundation ensures consistent, scalable, and faithful sampling aligned with the methodology described in the paper.

## utils.py

{
  "utils.py": "The utils.py module serves as the foundational utility layer facilitating reproducibility, configuration, data handling, sampling setup, and general helper functions needed across the project to implement the methodology from the paper.\n\nKey utility functions and considerations include:\n\n1. **Command-line Argument Parsing & Configuration Loading**:\n   - Parse CLI arguments to override or specify configuration parameters.\n   - Load the 'config.yaml' file for experiment setup, including model, sampling, divergence, evaluation, dataset, and misc parameters.\n   - Provide a function `load_config()` returning a structured dictionary or a namespace for easy referencing.\n   - Support optional command-line overrides for specific parameters such as seed, model name, or dataset paths.\n\n2. **Random Seed Control**:\n   - Implement `set_seed(seed_value)` to set seeds for `random`, `numpy`, and `torch` (if GPU-based) to ensure reproducibility.\n   - This guarantees sampling consistency across runs.\n\n3. **Hyperparameter and Parameter Management**:\n   - Extract hyperparameters from loaded config: `num_trajectories`, `max_length`, `temperature`, etc.\n   - Provide utility functions to parse/convert these into appropriate types and defaults.\n\n4. **Batching & Data Handling**:\n   - Implement functions to batch datasets for prompt pairs, relation data, and multimodal inputs.\n   - Facilitate reading JSON files (`prompt_pairs.json`, `wordnet_relations.json`, `multimodal_inputs.json`) into Python data structures, such as lists of dicts or tuples.\n   - Helper for batching large datasets to avoid memory issues during evaluation.\n\n5. **Sampling Utilities**:\n   - Create `prepare_sampling()` to initialize random seeds, set up batch processing, and define loop parameters.\n   - Provide `generate_trajectories(model, prompt, n, max_length, temperature, seed)` which calls the model wrapper for sampling trajectories.\n   - Ensure sampling is reproducible with seed control.\n\n6. **Likelihood & Log-Likelihood Computations**:\n   - Define `compute_sequence_log_likelihood(sequence, model, prompt)` which wraps model's token probability outputs to compute total log-likelihood.\n   - Normalize likelihoods using the parameter `tau` (e.g., 0.5):\n     \n     \[\n     M^{norm}(t) = \frac{M(t)^\tau}{\sum_{t'} M(t')^\tau}\n     \]\n   - These normalized likelihoods are used for divergence and similarity computations.\n\n7. **Likelihood Normalization and Divergence Prep**:\n   - Provide `normalize_likelihoods(likelihoods, tau)` to normalize sampled likelihood scores across trajectories.\n   - Prepare for divergence or distance calculations (L1, L2, KL, etc.) using `scipy` or custom functions.\n\n8. **Seed & Reproducibility Support**:\n   - Ensure that all stochastic processes—sampling trajectories, dataset shuffling—are seeded using the seed from config.\n   - Consistent seed setting across all functions called during an experiment.\n\n9. **Optional Utility Functions**:\n   - Formatting inputs: joining tokens into strings, adding punctuation if missing (e.g., full stops, as discussed in the paper).\n   - Functions to handle textual prompts/inputs, e.g., `prepare_prompt(prompt_str)`.\n   - Timing/decorators for profiling, if needed.\n\n10. **Configuration Validation**:\n    - Verify the presence and correctness of critical parameters (e.g., model name, dataset paths, hyperparameters).\n    - Set default values where parameters are missing if needed.\n\n**Summary**: \nDevelop `utils.py` to provide functions that load and parse configs, control randomness for reproducibility, handle dataset loading and batching, perform likelihood computation with normalization, facilitate trajectory sampling with seed control, and support general helper functions for formatting and validation. \n\nThese utilities will enable the main scripts (like main.py) to seamlessly orchestrate model inference, sampling, evaluation, and experiment management aligned precisely with the methodology, hyperparameters, and experimental protocols from the paper."
}

