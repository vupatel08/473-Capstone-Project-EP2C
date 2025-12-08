# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Below is a comprehensive, step-by-step plan to reproduce the Hierarchical Context Merging (HOMER) methodology as described in the paper, covering both the core algorithmic details and experimental setup.

---

# Overall Reproduction Plan for HOMER

## 1. Core Methodology Outline

### A. Objective
Implement an efficient, training-free method to extend the effective context length of pretrained large language models (LLMs) like Llama-2, enabling processing of sequences up to 64K tokens with maintained fluency.

---

### B. Key Components

#### i. Hierarchical Divide-and-Conquer Context Merging
- **Chunking:** Divide long input sequences into manageable, fixed-length chunks.
- **Independent Forward Passes:** Run each chunk through the initial transformer layers independently.
- **Hierarchical Merging:**
  - **Intermediate Layers:** Merge adjacent chunks by concatenation after applying token reduction (pruning) based on attention significance.
  - **Propagation:** Process merged chunks at higher layers, repeating recursively until a single chunk spans the entire input.
  - **Layer-wise Embeddings:** Obtain fixed-length, layerwise embeddings that summarize the entire sequence efficiently.

#### ii. Token Reduction
- **Significance Scoring:** Compute a score for each token based on the attention logit difference:  
  \[
  s_{i}^{\text{sig}} = l_{\text{att}}^{i} - l_{\text{bias}}^{\text{dist}(i)}
  \]
- **Pruning Criterion:** Remove tokens with low significance scores to keep chunk sizes manageable during merging phases.
- **Implementation:** Use attention-based importance, potentially calibrated, to prune tokens within each chunk before merging.

#### iii. Propagative Refinement
- **Upper-Layer Pruning Decisions:** Pruning of tokens in the top layers propagates downward, pruning the same tokens in lower-layer embeddings.
- **Outcome:** Uniform, compact embeddings at each layer, reducing memory and computation during inference time.

#### iv. Memory & Computation Optimization
- **Binary Tree Traversal:** Use DFS strategy to process nodes (chunks) recursively, reducing peak memory from linear to logarithmic.
- **Implementation of Algorithm 1:** Structure the merging and refinement to follow the binary tree order, apply propagative pruning after processing each node.

---

### C. Implementation Steps in Detail

1. **Input Processing:**
   - Tokenize input long texts.
   - Truncate if necessary or pad if shorter than target lengths.

2. **Chunk Creation & Initial Embeddings:**
   - Split input into chunks of maximum size \( C \) (e.g., 4K tokens).
   - Run each chunk independently through the base pretrained model's early layers (up to certain depth) to obtain initial hidden states or embeddings.
   - Store chunk embeddings for each such chunk.

3. **Hierarchical Merging Loop (Binary Tree Traversal):**
   - Represent chunks as leaf nodes.
   - Recursively process parent nodes:
     - **Process children first:** obtain their embeddings.
     - **Token Pruning:**
       - Compute attention importance scores.
       - Remove least important tokens to prune chunk sizes to a target (e.g., 2K tokens).
     - **Concatenate Adjacent Chunks:**
       - Merge pruned chunks via concatenation.
     - **Pass through higher transformer layers:**
       - Run the merged chunk's embeddings through subsequent transformer layers.
       - Apply propagative refinement to prune tokens at all lower levels.
     - **Continue upward** until a single merged chunk remains, representing the entire input sequence.

4. **Layer-wise Embedding Extraction:**
   - After each merging and refinement step, extract fixed-length embeddings representing the entire sequence.
   - These can serve as the context cache during autoregressive decoding.

---

### D. Additional Procedural Details

- **Position IDs management:** Reuse position IDs across chunks; assign same IDs to corresponding tokens in different chunks and affixes.
- **Token pruning calibration:** Use calibration datasets (e.g., WikiText-103) to set thresholds for importance scores, ensuring the effectiveness across different input lengths.
- **Memory & Time Optimization:** Implement the DFS-based processing order for large sequences to reduce peak memory to \( O(\log n) \) where \( n \) is input tokens.

---

## 2. Experimental Setup and Evaluation Strategy

### A. Datasets
- **Passkey Retrieval:**
  - Use or simulate the dataset described (500 samples per context length) for evaluation.
  - Long texts (up to 32K tokens) with passkeys embedded for downstream retrieval accuracy.
- **Question Answering:**
  - Use the QuALITY validation set (or similar QA benchmarks) with long documents.
- **Long Document Language Modeling:**
  - PG-19 dataset: 25 long documents (~25000-64K tokens each) to measure perplexity.
- **Additional Data:**
  - For calibration, use validation segments from WikiText-103 to tune token importance thresholds and positional encoding calibration.

### B. Model & Baselines
- **Base Model:** Llama-2 pretrained weights (both 7b and 13b variants).
- **Baselines:** 
  - Plain Llama (with original context limit, e.g., 4K tokens).
  - Position interpolation methods: PI, NTK, YaRN (apply their code with available configs).
- **Implementation note:** Since code is not provided, replicate their setup—installing or re-implementing positional encoding scaling methods.

### C. Hyperparameters
- **Chunk size \( C \):** default 4K tokens, with experiments scaling up to 8K, 16K, 32K, 64K (limited by hardware).
- **Token reduction policy:** prune to roughly 50-75% size at merging stages, calibrated on validation.
- **Transformer layers:** Use the same early layers as used in baseline models (e.g., 8 or 12 layers).
- **Number of hierarchical levels:** determined by \( \log_2(n / C) \), process recursively.
- **Number of merge and prune cycles:** match the depth of the binary tree.
- **Calibration datasets:** 100 segments for attention calibration/importance thresholds.

### D. Evaluation Metrics
- **Retrieval accuracy:** passkey retrieval task accuracy across varying input lengths.
- **Question answering accuracy:** correctness rate on the QA task with long context prompts.
- **Perplexity:** on long documents (PG-19) to measure fluency.
- **Inference speed:** measured in seconds per 20/50/100 tokens.
- **Memory usage:** peak GPU memory during processing (with FlashAttention).

### E. Implementation Details & Reproducibility
- **Tokenization:** consistent with Llama tokenizer, maximum sequence length.
- **Model execution:** implement early-layer inference via HF Transformers or custom code; for higher layers, run the merged chunk embeddings through the remaining layers.
- **Parallelization:** follow the DFS memory-efficient order for large sequences.
- **Calibration:** fine-tune significance pruning thresholds on validation data.
- **Code base:** adapt or re-implement evaluation scripts from the paper's tables (accuracy, perplexity, memory, speed).

---

## 3. Unclear Aspects or Assumptions
- Exact thresholds for token pruning (attention significance cutoff).
- Calibration procedures (parameter tuning details).
- Precise transformer layer parameters for early vs. late layers.
- How affix handling and positional IDs are managed during merging.
- Implementation specifics for the recursive hierarchical merging (pseudo-code only provided).
- Details about model fine-tuning—authors specify training-free, but calibration or slight tuning may improve performance.

---

## 4. Summary: Roadmap for Implementation
1. Build or adapt a script to tokenize and chunk input texts.
2. Implement early-layer inference for individual chunks.
3. Design a recursive binary tree traversal:
   - Merge chunks after token pruning based on attention importance.
   - Run merged chunks through subsequent layers.
   - Apply propagative token pruning at each node.
4. Collect fixed-length, layerwise embeddings after each merge until a single sequence embedding is obtained.
5. Integrate these embeddings into an autoregressive generation pipeline with kv-cache replacement.
6. Calibrate token importance thresholds/properties using validation data.
7. Evaluate on downstream tasks (passkey retrieval, QA) and perplexity.
8. Measure memory and inference time, compare with baselines.

---

This plan provides a detailed, reproducible strategy aligned with the paper’s central ideas, experimental protocols, and evaluation metrics. The next step involves translating this conceptual framework into code, adhering to the designed procedures.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular system that enables hierarchical long-context processing by chunking, independent transformer layer inference, recursive merging with token pruning, and fixed-length embedding extraction. We leverage Hugging Face Transformers for model loading, tokenization, and layer access. The core will recursively process input sequences via a binary tree traversal, pruning tokens at each merge step based on attention importance scores, and propagate pruning decisions downward. The system will include calibration steps for importance thresholds, efficient memory management via DFS order, and integration with autoregressive decoding using kv-cache replacements. For evaluation, scripts will handle downstream tasks such as passkey retrieval and long-document perplexity, measuring accuracy, perplexity, speed, and memory.",
    "File list": [
        "app.py",
        "dataset_loader.py",
        "context_merger.py",
        "model.py",
        "utils.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Tokenizer {\n        +__init__(model_name: str)\n        +tokenize(text: str) -> List[int]\n        +convert_tokens_to_ids(tokens: List[str]) -> List[int]\n        +pad_or_truncate(tokens: List[int], max_length: int) -> List[int]\n        +decode(tokens: List[int]) -> str\n    }\n    class TransformerModel {\n        +__init__(model_weights_path: str)\n        +get_hidden_states(input_ids: List[int], layers: List[int]) -> List[Tensor]\n        +get_layer_outputs(input_ids: List[int], layer_idx: int) -> Tensor\n        +run_segment(segment_ids: List[int], start_layer: int, end_layer: int) -> Tensor\n        +access_layer(layer_idx: int) -> Layer\n    }\n    class Layer {\n        +forward(hidden_state: Tensor) -> Tensor\n        +get_attention_weights() -> Tensor\n    }\n    class InputChunk {\n        +tokens: List[int]\n        +embedding: Tensor\n        +attention_scores: List[float]\n        +pruned_tokens: List[int]\n    }\n    class HierarchicalMerger {\n        +merge_chunks(left: InputChunk, right: InputChunk) -> InputChunk\n        +prune_tokens(chunk: InputChunk, importance_threshold: float) -> InputChunk\n        +recursive_merge(chunks: List[InputChunk], current_layer: int, max_layer: int) -> InputChunk\n    }\n    class EvaluationMetrics {\n        +compute_passkey_accuracy(predictions: List[int], ground_truth: int) -> float\n        +compute_perplexity(scores: List[float], references: List[float]) -> float\n        +measure_inference_time() -> float\n        +measure_memory_usage() -> float\n    }\n\nRelationships:\nTokenizer <|-- main app\nTransformerModel -- Layer\nHierarchicalMerger -- InputChunk\nHierarchicalMerger -- recursive_merge()\napp.py -- uses --> dataset_loader.py\napp.py -- uses --> model.py\napp.py -- uses --> context_merger.py\napp.py -- uses --> evaluation.py\n",
    "Program call flow": "sequenceDiagram\n    participant A as app.py\n    participant D as dataset_loader.py\n    participant M as model.py\n    participant H as context_merger.py\n    participant E as evaluation.py\n    A->>D: initialize dataset with config\n    D-->>A: load input texts\n    A->>M: load model weights and tokenizer\n    M-->>A: model ready\n    A->>H: initiate recursive_merge for input sequence\n    H-->>A: obtain long context embedding\n    A->>E: evaluate on downstream tasks (passkey retrieval, QA, perplexity)\n    E-->>A: metrics and reports\n    Note over A: For generation, replace model kv-cache with the embeddings.\n",
    "Anything UNCLEAR": "Clarification needed on the exact importance threshold calibration process, detailed merging parameters (like layer assignments), and any specific utilities for managing position IDs and affixes during chunk merging."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "transformers==4.31.0",
        "numpy==1.21.0",
        "tqdm==4.65.0",
        "scipy==1.10.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class for loading and tokenizing input texts, handling long sequences (via truncation/padding), and supporting calibration datasets. Depends on transformers for tokenization."
        ],
        [
            "model.py",
            "Defines TransformerModel class for loading pretrained Llama weights, accessing specific layers, and running inference on input segments. Uses Hugging Face Transformers, depends on model's layer access and output functions."
        ],
        [
            "context_merger.py",
            "Implements HierarchicalMerger class for recursive merging of chunk embeddings, token importance scoring, and pruning. Depends on TransformerModel for running layers and getting attention scores, and on utility functions for sequence segmentation."
        ],
        [
            "evaluation.py",
            "Includes classes for measuring downstream task accuracy (passkey, QA), perplexity, inference time, and memory usage. Depends on data outputs from other modules."
        ],
        [
            "app.py",
            "Main entry point orchestrating dataset loading, model initialization, recursive context merging, and evaluation. Uses dataset_loader.py, model.py, context_merger.py, and evaluation.py."
        ],
        [
            "utils.py",
            "Contains supporting functions such as importance threshold calibration, position ID management, DFS traversal for memory efficiency, and batching logic. Shared by context_merger.py and app.py."
        ]
    ],
    "Task list": [
        "requirements.txt (define dependencies for environment setup)",
        "utils.py (implement significance scoring, memory optimization, position ID handling)",
        "dataset_loader.py (implement dataset loading, tokenization, calibration dataset preparation)",
        "model.py (implement model loading, layer access, inference on segments)",
        "context_merger.py (recursive binary tree merge, token pruning, propagative refinement)",
        "evaluation.py (implement accuracy, perplexity, memory/time measurement)",
        "app.py (drive pipeline: load data, initialize model, run hierarchical merging, evaluate)"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "The system relies on Hugging Face Transformers for model and tokenizer loading. Utilities for attention importance scoring and recursive binary tree processing are shared between context_merger.py and app.py. Calibration datasets (WikiText-103) are used for importance threshold calibration. The project assumes availability of GPU with sufficient memory and access to pretrained Llama models."
    ,
    "Anything UNCLEAR": "Details needed on specific importance threshold calibration procedure, precise layer ranges for early vs. late transformer layers, and handling of special tokens/affixes during chunk merging."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  # No explicit training parameters provided in paper; focus on inference and hierarchical merging.
  # If fine-tuning is later applied, typical parameters such as the following could be used:
  # learning_rate: 1e-5  # Placeholder for potential finetuning
  # batch_size: 4  # Placeholder batch size for finetuning
  # epochs: 3  # Placeholder number of epochs
inference:
  max_input_length: 64000  # Target maximum sequence length in tokens, based on paper's experiments (up to 64K)
  chunk_size: 4000  # Chunk size for initial chunking (e.g., 4K tokens)
  prune_ratio: 0.5  # Keep roughly 50% tokens during pruning, adjustable based on calibration
  max_layer: 12  # Number of transformer layers used in model, e.g., for Llama-2-13B
  merge_layers: 12  # Number of layers in the model to run during merging
  importance_threshold_calibration_samples: 100  # Number of samples for importance calibration
  importance_calibration_dataset: "wikitext-103" # Dataset used for importance threshold calibration
memory:
  enable_memory_optimization: true  # Use DFS traversal for memory efficient processing
  memory_goal_percentage: 70  # Aim to reduce peak memory to 70% of raw usage
evaluation:
  passkey_retrieval_tasks:
    input_lengths: [4000, 8000, 16000, 32000]
    metric: accuracy (percentage of correctly retrieved passkeys)
  question_answering:
    dataset: "Qualify"
    prompt_format: "prompt_format_qa.txt"  # Template file for QA prompt
    metric: "accuracy"
  long_document_perplexity:
    dataset: "PG-19"
    max_eval_length: 64000  # Evaluation on the full length
    metric: "perplexity"
  inference_speed:
    measure: "seconds per 20 tokens"
    hardware: "single A100 GPU"
calibration:
  dataset: "WikiText-103"
  samples: 100  # For importance calibration
```

---

**Note:** Full configuration is available in `planning_config.yaml`
