# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, structured plan for reproducing the experiments and implementing the methodology described in the paper "InfLLM: Training-Free Long-Context Extrapolation for LLMs" by Xiao et al. 

---

# Overall Objectives
- Implement a **training-free, memory-enhanced extension** of pre-trained LLMs (e.g., Llama-3, Mistral-7B).
- Enable processing of extremely long sequences (up to 1,024K tokens).
- Validate the effectiveness via experiments on benchmark long-document datasets (e.g., ∞-Bench, LongBench).
- Match the evaluation protocols and hyperparameters as specified, to reproduce reported results and ablation studies.

---

# 1. Methodology Implementation Plan

## 1.1 Core Components
- **Sliding Window Attention**:
  - Local attention within a fixed window size (e.g., 4K tokens).
  - Limit attention to local tokens; ignore distant tokens unless relevance info supplied via memory.
- **External Memory Module ("Context Memory")**:
  - Organized into **block-level units** (each block containing `l_b s` tokens).
  - For each block:
    - Compute an **"importance score"** for each token:
      \[
      r_m = \frac{1}{l_L} \sum_{j=1}^{l_L} \mathbf{q}_{m+j} \cdot \mathbf{k}_m
      \]
      where \(\mathbf{q}_{m+j}\) are queries for tokens in local window; \(\mathbf{k}_m\) are keys.
    - Select **top \( r_k \)** tokens with highest importance scores per block as **unit representations**.
    - Store **block key-value pairs** (block-level summaries).
- **Memory Lookup Mechanism**:
  - For each input token during inference:
    - Compute **relevance scores** with stored memory units:
      \[
      \text{Relevance}(X, B) = \sum_{i=1}^{l_X}\sum_{j=1}^{r_k} \mathbf{q}_i \cdot \mathbf{k}_b^{(j)}
      \]
    - Select top \(k_b\) highly relevant units, load their key-value pairs (offloading less relevant units to CPU).
- **Positional Encoding**:
  - Assign **shared positional embeddings** for distant tokens (beyond local window).
  - For current tokens, use standard positional encoding; for memory units, assign a fixed large offset (e.g., l_L).

## 1.2 Memory Management & Cache System
- Use **an LRU cache** on GPU to store **most frequently accessed memory units**.
- Offload seldom-used units to CPU memory (avoid GPU memory overload).
- Update relevance scores after each attention step; dynamically update cache.
- Evaluate the **miss rate** of cache (see empirical data and formulas in the paper).

## 1.3 Integration with Pre-trained LLMs
- **Wrap pre-trained models (e.g., Llama-3, Mistral-7B)**:
  - Replace the attention module with the **sliding window + memory lookup**.
  - Keep the original model parameters fixed (no training).
- **Inference-only Mode**:
  - Implement **chunk-by-chunk encoding**:
    - Encode segments of sequence with the augmented attention.
    - Update memory cache after each chunk.
  - For sequences up to 1,024K tokens, process in smaller chunks (e.g., 4-8K local window + relevant memory).

## 1.4 No Fine-Tuning
- The entire approach relies solely on **attention modifications**; no additional training.
- Memory units are **constructed dynamically** based on importance scores.
- For block selection, assume **importance scores are computed on the fly** without extra auxiliary models.

---

# 2. Experimental Design & Protocols

## 2.1 Datasets & Benchmarks
- **Primary Datasets**:
  - **∗-Bench** (Zhang et al., 2023a):
    - Long sequences, broad task coverage.
    - Average sequence length: ~145K; 95% quantile: 214K tokens.
  - **LongBench** (Bai et al., 2023):
    - Sequence lengths often below 32K, up to 1024K in some experiments.
- **Additional Datasets**:
  - GovReport, QMSum, MultiNews, TREC, TQA, SAMSum (for evaluation of sequence length scaling and diversity).

## 2.2 Experimental Settings
- **Model bases**:
  - Llama-3-8B (or smaller variants for certain tests).
  - Mistral-7B.
- **Sequence Lengths**:
  - Short (e.g., 8K tokens) as baseline.
  - Long (up to 1024K tokens) for stress testing the method.
- **Chunking Strategy**:
  - Chunk input sequences into **local window size** (e.g., 4K tokens) plus relevant memory.
  - For very long sequences, **stream processing** with context memory updating.
- **Hyperparameters**:
  - Local window size (`l_x`): 4K.
  - Memory block size (`l_bs`): 512.
  - Top tokens per block (`r_k`): 4.
  - Decay coefficient (`d`): 0.1 (per paper).
  - Number of relevant units loaded (`k_b`): varies (e.g., 5, 10, 20).
  - Cache size: e.g., 64 units, or based on empirical cache miss rate.

## 2.3 Metrics & Evaluation
- **Per-task performance metrics**:
  - QA: Accuracy.
  - Summarization: ROUGE scores.
  - Math and logic tasks: accuracy.
  - Retrieval tasks: Precision, Recall.
- **Long-sequence performance**:
  - Sequence-level accuracy.
  - Ability to capture long-distance dependencies.
  - Computational resource usage:
    - VRAM, memory footprint.
    - Processing time per sequence (see paper for timings).
- **Ablations**:
  - Effect of memory block size.
  - Cache size and management strategy.
  - Without memory (local attention only).
  - Without cache.

## 2.4 Reproduction of Results & Ablations
- Reproduce key results such as:
  - Performance vs. sequence length (Fig. 2, 3).
  - Effectiveness of block importance scoring.
  - Cache miss rates.
  - Effect of increasing number of memory units.
- Confirm that no training is performed; only inference modifications.
- Compare with baseline models (e.g., original Llama, Mistral without memory extension).

---

# 3. Implementation Details & Practical Tips

## 3.1 Data Preparation
- For long sequences:
  - Generate or obtain datasets with sequence lengths from 10K to 1,024K tokens.
  - Chunk sequences for streaming inference.
- For relevance computation, store **evicted tokens' key-value pairs** in a structured format (block-wise).

## 3.2 Model Modification
- Use HuggingFace transformers **modify attention modules**:
  - Insert custom attention logic:
    - Local attention (windowed).
    - Memory relevance lookup.
  - Ensure no retraining:
    - Freeze all original model parameters.
    - Incorporate cache and relevance logic only.

## 3.3 Code Structure
- Modularize in:
  - Base model wrapper.
  - Memory management:
    - Build block organization.
    - Compute importance scores on chunk encoding.
    - Implement cache system with LRU policy.
  - Inference engine:
    - Chunk stream, update memory after each chunk.
    - Concatenate outputs.
- Logging:
  - Record sequence length, resource consumption, cache hit/miss rates.
  - Save outputs for performance evaluation.

## 3.4 Hyperparameter Tuning
- Reproduce the default values:
  - Local window size: 4K tokens.
  - Memory block size: 512 tokens.
  - Top representative tokens: 4.
  - Relevance decay factor: 0.1.
- Then, vary parameters systematically to observe impact:
  - Memory size (`k_b`): 5, 10, 20.
  - Validity threshold for relevance score.
  - Cache size.

---

# 4. Additional Considerations

- Confirm that the model's **attention API** allows custom attention masks and custom relevance computations.
- Handle **positional encoding** carefully:
  - For distant memory units, assign large offset embeddings.
  - Confirm that positional embeddings do not confuse the model with the original ones.
- Optimize for **VRAM efficiency**:
  - Offload infrequently used memory units to CPU.
  - Use batch processing across chunks where possible.
- Validate the **cache update and relevance scoring** methodology on a small subset first.

---

# Summary
This roadmap thoroughly covers the methodology, experiment protocols, hyperparameters, resource management, and evaluation strategies needed to reproduce and implement the paper's core ideas. Next, each step can be translated into specific code modules, data loaders, and training/inference scripts aligned with these specifications.

---

Let me know if you'd like me to prepare a detailed code skeleton or specific pseudo-code for critical components based on this plan.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular yet simple system using HuggingFace Transformers for pre-trained models, adding custom attention wrappers that incorporate local sliding window attention combined with a memory management system. The memory module will dynamically select relevant block-level key-value units based on importance scores computed on-the-fly. Cache management will employ an LRU mechanism using Python data structures, with offloading to CPU memory for less frequently accessed units. The entire system will stream process long sequences in chunks, updating external memory after each step, and will be wrapped in a flexible inference pipeline to evaluate on long document datasets without retraining models.",
    "File list": [
        "main.py",
        "model.py",
        "attention_wrapper.py",
        "memory_manager.py",
        "dataset_loader.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run_experiment()
    }
    class DatasetLoader {
        +__init__(dataset_paths: List[str])
        +load_dataset() -> Iterable[dict]
    }
    class ModelWrapper {
        +__init__(model_name: str, config: dict)
        +generate(prompt: str, long_input: str) -> str
    }
    class AttentionWrapper {
        +__init__(local_window_size: int, memory_manager: MemoryManager)
        +call(inputs: Tensor, memory: ExternalMemory) -> Tensor
        +compute_importance_scores(block_tokens: Tensor, queries: Tensor, keys: Tensor) -> Tensor
        +adjust_attention_mask() -> Tensor
    }
    class MemoryManager {
        +__init__(block_size: int, top_k: int, cache_size: int, decay: float)
        +update_memory(evicted_blocks: List[Block]) -> None
        +select_relevant_units(current_tokens: Tensor, top_k: int) -> List[KVPair]
        +offload_to_cpu() -> None
        +load_from_cpu() -> None
        +get_most_frequent_units() -> List[KVPair]
    }
    class ExternalMemory {
        +__init__()
        +store_blocks(blocks: List[Block]) -> None
        +retrieve_relevant(current_block: Tensor, top_k: int) -> List[KVPair]
        +manage_cache() -> None
    }
    class ChunkedInput {
        +__init__(long_sequence: str, chunk_size: int)
        +stream_chunks() -> Iterable[str]
    }
    class Evaluation {
        +__init__(model: ModelWrapper, dataset: Iterable[dict])
        +evaluate_performance() -> dict
        +log_metrics()
    }
    Main --> DatasetLoader
    Main --> ModelWrapper
    Main --> AttentionWrapper
    Main --> MemoryManager
    Main --> Evaluation
    ModelWrapper --> AttentionWrapper
    AttentionWrapper --> ExternalMemory
    ExternalMemory --> MemoryManager
    ExternalMemory --* Block
    MemoryManager --> ExternalMemory
    ChunkedInput --> DatasetLoader
    Main --> ChunkedInput
    Main --> Evaluation
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant MD as ModelWrapper
    participant AT as AttentionWrapper
    participant EM as ExternalMemory
    participant CM as CacheManager
    participant EV as Evaluation

    M->>DL: load_dataset()
    DL-->>M: dataset iterable
    M->>MD: initialize(model_name, config)
    MD-->>M: ready
    alt for each long sequence in dataset
        M->>ChunkedInput: stream_chunks()
        ChunkedInput->>M: get chunk
        loop for each chunk
            M->>EM: select relevant units
            EM-->EM: manage cache, load new units
            M->>AT: call(inputs, EM)
            AT-->MD: generate output with custom attention + memory
        end
        M->>Evaluation: evaluate performance
        Evaluation-->M: metrics
        M->>Evaluation: log_metrics()
    end
    end
    note over M: stream process all sequences, updating memory dynamically, computing importance scores, and evaluating metrics without training
",
    "Anything UNCLEAR": "Clarify the precise format of the input datasets, especially how sequences are provided and chunked. Also, confirm if custom attention mask logic needs to be integrated into the HuggingFace Transformer attention modules. Details about evaluation benchmarks and resource constraints would help optimize cache sizes and timing."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch>=1.12",
        "transformers>=4.26",
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=0.24",
        "tqdm>=4.62",
        "pandas>=1.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class that loads datasets (∗-Bench, LongBench, etc.), supports stream chunking, and returns sequence dicts with prompts, labels, and metadata."
        ],
        [
            "model.py",
            "Defines ModelWrapper class that loads pre-trained models (e.g., Llama-3, Mistral-7B) via HuggingFace transformers, and wraps inference with hooks for attention, cache management, and custom attention mask application."
        ],
        [
            "attention_wrapper.py",
            "Implements AttentionWrapper class that re-implements attention calls, applying local sliding window attention, computing importance scores, and integrating external memory relevance retrieval."
        ],
        [
            "memory_manager.py",
            "Implements MemoryManager class which manages block-level memory units: inserting, selecting top relevance units based on importance scores, cache update via LRU, and offloading to CPU memory—relies on Block data structures."
        ],
        [
            "main.py",
            "Serves as entry point, initializes dataset loader, model, memory manager, and orchestrates streaming of each dataset sequence, chunk-by-chunk, updating memory after each chunk, invoking custom attention, and collecting outputs for evaluation."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class that takes model and dataset, runs sequences through streaming inference pipeline, collects metrics (accuracy, ROUGE, etc.), and logs performance including resource usage and cache hit/miss statistics."
        ],
        [
            "utils.py",
            "Contains utility functions for positional encoding adjustments, importance score calculation, cache management helpers, and sequence chunking helpers."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "attention_wrapper.py",
        "memory_manager.py",
        "main.py",
        "evaluation.py",
        "utils.py"
    ],
    "Full API spec": "No API endpoints required; internal class APIs for model inference, memory management, data loading, and evaluation are sufficient.",
    "Shared Knowledge": "Shared knowledge includes the importance scoring formulas, cache management (LRU), chunking methods, and dataset format expectations. Model wrapper and attention wrapper depend on HuggingFace models and custom attention calls.",
    "Anything UNCLEAR": "Clarify the expected dataset format (sequence chunks, prompts, labels). Confirm if any model-specific positional encoding modifications are necessary. Resource constraints for large sequences and hardware options for offloading also need clarification."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: "∗-Bench"
  sequence_length: 214000  # 95% quantile for sequence length
  batch_size: 1  # process sequences individually

model:
  name: "Llama-3"  # or "Mistral-7B" depending on choice
  load_in_8bit: false
  freeze_parameters: true  # inference-only, no training
  max_sequence_length: 1024000  # scale to handle 1,024K tokens

attention:
  local_window_size: 4096  # local sliding window size
  memory_block_size: 512  # tokens per memory block
  top_k_representative_tokens: 4  # per block
  relevance_decay: 0.1  # relevance score decay coefficient

memory_manager:
  cache_size_gpu: 64  # number of memory units maintained on GPU
  cache_offload_strategy: "LRU"  # least recently used
  memory_block_size: 512  # tokens per unit
  relevance_top_k: 4
  offload_to_cpu: true  # offload less relevant units
  update_frequency: 1  # update memory after each chunk

inference:
  chunk_size: 4096  # local window chunk size
  max_sequence_length: 1024000  # process sequences up to 1,024K tokens
  device: "cuda"  # using GPU
  offload_memory_to_cpu: true

evaluation:
  metrics:
    - accuracy
    - ROUGE
    - retrieval precision
  evaluation_interval: 1  # evaluate after each sequence

resources:
  gpus: 1  # minimum GPU needed
  memory_gb: 16  # available GPU memory
  timing:
    total_time_per_sequence: null  # to be measured during run
```

---

**Note:** Full configuration is available in `planning_config.yaml`
