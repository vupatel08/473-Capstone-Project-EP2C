# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## attention_steering.py

{
  "attention_steering.py - Logic Analysis": [
    {
      "module purpose": "Implement the 'AttentionSteering' class responsible for adjusting attention scores during inference based on user-identified emphasized spans and pre-selected attention heads. This class acts as a core component for real-time attention reweighting, enabling post-hoc steering without retraining the model."
    },
    {
      "high-level design": "The class should initialize with model reference, selected head set, and hyperparameters such as alpha (scaling coefficient). It must contain a method that, given raw attention scores and emphasis spans, outputs reweighted attention scores to bias the model’s focus toward specified tokens."
    },
    {
      "core logic steps": [
        "Upon inference, for each head in the selected set, intercept the attention score matrices (prior to softmax) using hooks or wrapper functions.",
        "For each attention head (layer l, head h), retrieve the corresponding attention score matrix A^(l,h).",
        "Given a set of emphasis spans (represented as token indices), identify tokens to emphasize and tokens to downweight in that attention head.",
        "Apply the reweighting transformation: scale down attention scores for non-emphasized tokens by multiplying by alpha, normalize across each query token (row-wise) to produce a valid probability distribution for the attention mechanism.",
        "Inject the adjusted attention scores back into the model's computation flow, replacing the original attention scores at the specific head(s)."
      ],
      "note": "Implementation should support per-head reweighting during inference, compatible with executing via hooks, capturing the attention scores just before softmax or within the attention computation function."
    },
    {
      "detailed steps": [
        "Initialize AttentionSteering class with parameters:",
        " - model reference (or model wrapper), to register hooks or manipulate attention internally.",
        " - selected_heads: List of (layer, head) tuples to steer.",
        " - alpha: scaling factor (default 0.01).",
        " - emphasis_spans: list of token indices corresponding to the user-highlighted spans.",
        "Implement a method 'apply_reweighting(attention_scores, emphasis_spans)' that:",
        " - For each (layer, head) in selected_heads:",
        "   - Access corresponding attention score matrix (shape: [batch_size, num_query_tokens, num_key_tokens])",
        "   - Create a mask for tokens outside emphasis spans: a boolean mask where tokens in emphasis spans are True, others False.",
        "   - For each attention score matrix:",
        "     - Downweight non-emphasized token scores: multiply by alpha at positions where tokens are outside emphasis spans.",
        "     - Keep emphasized token scores unchanged.",
        "     - Normalize each row (query token): divide by the sum of all scaled scores to ensure the distribution remains valid.",
        " - Return the reweighted attention scores.",
        "The class should contain methods to:",
        " - Register hooks into model's attention layers that call 'apply_reweighting' during attention computation.",
        " - Accept emphasis spans dynamically at inference time, derived from prompt parsing.",
        " - Potentially handle batching and multiple heads efficiently."
      ],
      "note": "Careful attention is required to access internal attention scores within the model architecture—preferably via predefined hooks or by modifying the forward pass if hook API is insufficient. All reweighting must preserve differentiability and valid probability distributions, conforming to the original attention computation process."
    },
    {
      "considerations": [
        "Ensure attention reweighting is only applied to the selected heads as profiled (per task).",
        "Design the reweighting process to be modular and stateless, so it can be invoked during inference per input instance.",
        "Handle variable sequence lengths and batch sizes properly.",
        "Maintain numerical stability: normalization after reweighting is critical.",
        "Validate that the attention manipulation preserves overall model functioning and does not introduce artifacts or inconsistencies."
      ]
    },
    {
      "validation": "Unit test the reweighting:
        - Create mock attention matrices and emphasis spans.
        - Verify that after reweighting:
        - (a) emphasis tokens have increased attention weights.
        - (b) non-emphasized tokens are scaled down by alpha.
        - (c) each row sums to 1 (probability distribution preserved)."
    }
  ]
}

## dataset_loader.py

**Logic Analysis for `dataset_loader.py` — DatasetLoader Class**

---

### **Purpose & Role**
The `DatasetLoader` class is designed to facilitate loading, preprocessing, and formatting datasets for use in profiling, inference, and evaluation of the PASTA method. It must support:
- Loading datasets from specified directories.
- Splitting datasets into train, validation, and test sets.
- Tokenizing input texts compatible with the target LLM.
- Annotating emphasis spans from user-highlighted input markers.
- Structuring data samples appropriately for downstream modules.

---

### **Core Responsibilities**
1. **Loading Data**
   - Read raw datasets from the specified directory, which likely contain textual prompts and labels.
   - For the evaluation datasets (BiasBios, CounterFact, JSON Formatting, Pronouns Changing), load the data splits: `train`, `validation`, and `test`.
   - Support dataset formats such as JSONL, CSV, or plain text, depending on dataset specifics provided.

2. **Dataset Splitting & Sampling**
   - Use dataset sizes specified in `config.yaml`:
     - `train_size`: e.g., 1000 samples for training.
     - `val_size`: e.g., 1000 samples for validation.
     - `test_size`: e.g., 5000 samples for testing.
   - Draw **random samples** (preferably via reproducible seed) to create subsets for profiling and evaluation.
   - Ensure balanced sample selection when necessary, especially for profiling (e.g., diverse samples to evaluate attention head performance).

3. **Preprocessing & Tokenization**
   - **Tokenize** prompts and contexts using the model’s tokenizer (`transformers` library).
   - Convert texts into token ID sequences compatible with model inference.
   - Maintain offsets of tokens to align with emphasis span annotations.
   - Handle input length constraints (truncate or pad) as necessary to match model input limits.

4. **Annotating Emphasis Spans**
   - Detect and extract emphasized segments within inputs:
     - Presumably enclosed with markers like `*` or markdown syntax, e.g., `*emphasized text*`.
   - For each sample, identify the start and end token indices corresponding to emphasized spans:
     - Use tokenizer offsets to map character span positions to token indices.
     - Support multiple emphasized spans (not necessarily contiguous).
     - Generate a list of emphasized token indices or spans for downstream reweighting.
   - Store emphasis annotations within each sample, e.g., as `emphasis_spans: List[List[int]]`.

5. **Data Structuring**
   - Store each sample as a structured object, e.g., a dictionary or custom class, with fields such as:
     - `prompt`: raw prompt input text.
     - `tokenized_input`: token IDs after tokenization.
     - `attention_mask`: for padding.
     - `emphasis_spans`: list of token indices or start/end pairs.
     - `labels`: target output (for evaluation and profiling).
   - Support retrieval of raw text, tokenized input, and emphasis annotations.

6. **Output & Compatibility**
   - Provide an API to retrieve datasets as ready-to-batch forms, e.g., via `__getitem__` and `__len__`, enabling integration with training, profiling, or evaluation pipelines.
   - Enable batching and shuffling for validation and training sets.
   - Ensure datasets are compatible with models, i.e., potential padding/truncation logic.

---

### **Implementation Details & Considerations**

- **Dataset Format Handling**
  - Detect dataset file formats (JSON, CSV, etc.) based on file extension or metadata.
  - Parse dataset entries, assuming consistent schema per dataset:
    - For example: `{ "context": "...", "question": "...", "label": "...", "emphasis_marker": "*" }`.
  - For datasets with pre-labeled emphasis regions, load span indices directly.
  - For datasets with emphasis markers in raw text, implement span extraction logic.

- **Emphasis Span Extraction Logic**
  - Use utility functions to locate emphasis markers.
  - Extract character start/end positions of emphasized sections.
  - Map character spans to token indices:
    - Tokenizer outputs include offsets mapping tokens to original string positions.
    - Use these offsets to assign tokens to emphasis spans.

- **Tokenization & Offset Handling**
  - For each raw text, tokenize with `return_offsets_mapping=True`.
  - For emphasis spans specified as character ranges:
    - Match these to token index ranges via offsets.
  - For multiple emphasis spans, accumulate token indices.

- **Reproducibility & Sampling**
  - Fix random seed for samples selection to ensure reproducible results.
  - Use `np.random.choice()` or similar for sampling with replacement or without, as needed.

- **Edge Cases**
  - Empty emphasis spans or missing markers: treat as no emphasis.
  - Overlapping emphasis spans: merge or treat distinctly as per intent.
  - Very long inputs: truncate to max token length aligned with model limits.

---

### **Outputs & Interface**
- **`DatasetLoader` Interface**
  - Initialization parameters: task name, data directory, split (`train`/`validation`/`test`).
  - Method `load_dataset()`:
    - Loads raw data.
    - Performs sampling.
    - Tokenizes texts.
    - Extracts emphasis spans.
    - Returns dataset object optimized for iterating during profiling/evaluation.
  - Support optional parameters:
    - Reproducibility seed.
    - Maximum sequence length.
    - Whether to extract emphasis span information or use pre-labeled data.

---

### **Summary of Critical Logic Steps**

| Step | Description                                                                                      | Implementation Notes                                                                                       |
|-------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 1     | **Load dataset files** from `data_dir`, supporting multiple formats.                            | Use `json.load()`, `csv.reader()`, etc., with format detection.                                            |
| 2     | **Sample subsets** for profiling or evaluation, based on `train_size`, `val_size`, `test_size`. | Use fixed seed; sample without replacement; ensure dataset diversity.                                  |
| 3     | **Parse prompts/context** and **detect emphasis markers** (`*` or other).                      | Use regex or string functions; handle multiple/mixed spans.                                               |
| 4     | **Map character spans** of emphasis to token indices via `return_offsets_mapping`.             | Match span start/end to token offset ranges; handle multiple spans.                                     |
| 5     | **Store tokenized inputs**, emphasizing span info, along with labels and raw texts.            | Prepare data dictionaries for dataset objects with all relevant fields.                                |
| 6     | **Provide dataset objects** with APIs or attributes for downstream use.                       | Support indexing, batching, shuffling.                                                                    |

---

### **Final Points**
- Ensure reproducibility and robustness in span extraction.
- Maintain alignment between character-based emphasis marks and token indices.
- Confirm dataset format compatibility.
- This class must serve as a foundational component, feeding data to profiling, inference, and evaluation modules with the correct emphasis annotations to facilitate attention steering in PASTA.

---

This detailed analysis guides the implementation of `DatasetLoader` class, ensuring it reliably loads, preprocesses, and annotates datasets for effective profiling, inference, and evaluation aligned with the PASTA methodology.

## evaluation.py

# Logic Analysis for `evaluation.py`

## Purpose
The `evaluation.py` module encapsulates the `Evaluation` class, responsible for executing model inference on test datasets and calculating performance metrics as specified in the configuration. It interacts with data loading (`dataset_loader.py`) for preparing datasets and with model inference (`model_wrapper.py`) for generating outputs. It also computes various evaluation metrics (accuracy, success rates, fluency, etc.) based on generated and reference texts.

---

## Core Responsibilities
1. **Loading Datasets:** Use `dataset_loader.py` to fetch test datasets, structured appropriately with tokenized inputs, emphasis span annotations, and reference outputs.
2. **Model Inference:** Use `model_wrapper.py` to generate model outputs given input prompts, with optional inference-time attention reweighting for specified heads and emphasis spans.
3. **Metric Computation:** Apply evaluation metrics as specified:
    - **Format accuracy (F. Acc.):** Valid JSON output.
    - **Prediction accuracy (P. Acc.):** Correctness against ground truth.
    - **Pronoun accuracy and all-changed accuracy (A. Acc.):** For pronoun handling tasks.
    - **CounterFact efficacy (ES/PS):** Effectiveness and paraphrase scores.
    - **BiasBios accuracy:** Correct classification of occupation.
    - **Fluency metrics:** Bigram and trigram entropy.
4. **Reporting Results:** Aggregate metrics across datasets, produce per-task and overall summaries, and handle task-specific evaluations.
5. **Handling Inputs & Generated Outputs:**
    - For each sample:
      - Input prompt with emphasis spans.
      - Model inference with attention steering if applicable.
      - Parse and extract generated output for metrics.
6. **Robustness & Reproducibility:**
    - Ensure consistent evaluation by fixed seed, batch processing.
    - Handle potential invalid JSON or malformed outputs gracefully.
    - Implement caching or logging as needed for traceability.

---

## High-Level Workflow
### Step 1: Initialization
- Read configuration (if needed from arguments or environment).
- Set up evaluation parameters (metrics to compute).
- Prepare datasets for the test split: load via `dataset_loader.py`.

### Step 2: Data Loading
- Use `DatasetLoader` class to retrieve test data.
- Each test example should include:
  - Input prompt text.
  - Emphasis markers span indices or tokens.
  - Reference output(s).

### Step 3: Inference Loop
- For each test example:
  - Parse emphasis spans from the prompt text (via `utils.py` helper functions).
  - Feed the prompt to `model_wrapper.py`'s `generate()`:
    - Pass emphasis span indices.
    - Indicate whether attention steering is active.
  - Gather generated output string.

### Step 4: Post-processing
- Parse generated output for correctness:
  - For JSON tasks, attempt to parse JSON; handle errors.
  - For classification tasks, extract predicted label.
- For pronoun changing, compare generated pronouns vs. reference.
- For counterfact tasks, compare with referenced facts.
- For bias tasks, determine the predicted occupation vs. ground truth.

### Step 5: Metric Calculation
- Based on task:
  - Compute specific accuracy metrics.
  - Count the number of valid JSON outputs and correct JSON values.
  - Compute success rates for counterfact.
  - Calculate fluency:
    - Tokenize generated text.
    - Compute bigram & trigram entropy.
  - For consistency or other auxiliary metrics, compare generated to reference texts.

### Step 6: Aggregate Results
- Maintain counters/statistics:
  - Total examples per task.
  - Counts of correct/valid outputs.
  - Sum of fluency scores.
- After iteration:
  - Compute final metrics (percentages, averages).
  - Prepare task-specific report.

### Step 7: Final Reporting
- Organize results into a structured dict or printout.
- Include per-metric and overall scores.
- Optionally, save detailed logs or result summaries.

---

## Implementation Details & Nuances
- **Metrics Specifics:**
  - **Format accuracy:** Check JSON validity using `json.loads()`.
  - **Prediction accuracy:** Compare generated values with reference (exact match).
  - **Pronoun tasks:** Use regex or `utils.py` span extractor to find and replace pronouns; compare.
  - **CounterFact:** Parse input question, compare model output with old/new facts.
  - **BiasBios:** Implement classifier or matching logic to see if predicted occupation matches the label.
- **Fluency:**
  - Use n-gram entropy calculations.
  - For each generated text, compute entropy scores for bigram/trigram.
- **Error Handling:**
  - Implement try-except blocks for JSON parsing, metric calculations.
  - Log errors or skip invalid examples.

---

## Inputs & Outputs
*Inputs:*
- Dataset examples (text, emphasis spans, labels).
- Model inference function (`model_wrapper.py`).
- Evaluation configuration flags (`evaluation.metrics`).

*Outputs:*
- A dictionary (or serialized JSON) with all metrics per task.
- Optional detailed logs of outputs and errors.
- Final performance summary (overall accuracy, fluency, etc.) for reporting.

---

## Final Notes
- Make the evaluation process reproducible:
  - Use fixed random seeds.
  - Ensure consistent tokenization.
- Modular structure:
  - `run()` function orchestrates dataset loading, inference, metrics.
  - Dedicated functions for each metric to keep code organized.
- Flexibility:
  - Allow different tasks to specify which metrics to evaluate.
  - Enable switching metric calculation modes via config.

This detailed logic analysis guides the implementation of `evaluation.py` to faithfully emulate the evaluation process described in the paper, ensuring consistent, accurate, and comprehensive performance assessment of PASTA-enhanced LLMs.

## main.py

**Logic Analysis for main.py**

---

### **Purpose & Role of main.py**

- Serve as the **orchestration script**, controlling the overall workflow:
  - Load configuration parameters.
  - Initialize and coordinate datasets, models, profiling, attention steering, inference, and evaluation.
  - Manage the sequence: data loading → profiling → head selection → inference with attention reweighting → evaluation → logging results.

### **Step-by-Step Logical Flow & Components**

---

### **1. Load Configuration**

- Parse `config.yaml`:
  - Model details: name and path.
  - Dataset info: dataset name, directory, splits, sizes.
  - Profiling setup: number of heads to select, number of samples.
  - Attention steering hyperparameters (alpha, strategy).
  - Evaluation metrics flags.

---

### **2. Initialize Dataset Loader**

- Instantiate `DatasetLoader`:
  - Provide dataset name, directory, and splits.
  - Load raw datasets for train, validation, and test splits.
- Use `load_dataset()` method for each split.
- Apply tokenization and preprocessing steps (possibly via utilities), especially:
  - Identifying emphasis spans in samples (based on markdown markers or custom tags).
  - Ensuring datasets are ready for profiling, inference, and evaluation.

---

### **3. Initialize Model Wrapper**

- Instantiate `ModelWrapper`:
  - Load pre-trained language model according to `model_path`.
  - Configure device (GPU/CPU).
- Confirm support for attention score access:
  - Register hooks or callbacks to intercept attention scores.
  - Verify that registered hooks allow modification of attention scores during inference without retraining.

---

### **4. Perform Profile of Attention Heads**

- Instantiate `Profiler` with:
  - `model_wrapper` reference.
  - Small datasets: training subsets for profiling.
  - Specified number of samples (`profile_samples`).
  - Top heads to select (`top_heads_count`).

- Call `profile_heads()` method:
  - For each sample in profiling set:
    - For each layer \( l \) and head \( h \):
      - Temporarily modify attention (via registered hooks) to steer only head \( (l,h) \).
      - Run inference on sample.
      - Evaluate performance relevant to task metrics.
      - Record results.
  - Aggregate results per head across samples.
  - Select *top-k heads* based on performance metric (accuracy, effectiveness).
  - Store the selected head set (`heads_selection_strategy` determines intersection/union/top-k).

- Store the resulting `selected_heads` (list of `(layer, head)` tuples).

---

### **5. Set Up Attention Steering for Inference**

- Instantiate `AttentionSteering`:
  - Pass `model_wrapper`.
  - Set `alpha` as per config.
  - Assign the `selected_heads`.
  
- This class internally registers hooks for the selected heads to modify attention scores during inference.

---

### **6. Processing Test Set / Tasks for Inference**

- Loop over each test sample:
  - **Extract emphasis spans**:
    - Use utility functions to parse prompt text.
    - Identification could be based on markdown markers (`*`, quotes, etc.).
  - **Set emphasis information**:
    - Convert token span indices into token IDs aligned with tokenizer.
  - **Generate output with attention steering**:
    - Invoke `model_wrapper.generate()`:
      - Pass the input tokens.
      - Provide emphasis spans.
      - Set `alpha`.
    - During generation:
      - The registered hooks in `model_wrapper` will reweight attention scores at targeted heads to emphasize spans.
  - **Optional:** store raw generated output for evaluation.

---

### **7. Evaluate Generated Outputs**

- Instantiate `Evaluation`:
  - Pass `model_wrapper`-based inference and datasets.
  - Enable relevant metrics based on flags (format accuracy, pronoun accuracy, fluency, effectiveness).
- Run `evaluate()`:
  - For each test sample:
    - Compare generated output against ground truth.
    - Calculate metrics accordingly (e.g., JSON validity, accuracy, fluency entropy).

---

### **8. Log Results & Summary**

- Collect all metrics into a results dictionary.
- Print/log detailed results:
  - Per task metrics.
  - Overall average performance.
- Save logs/output files (e.g., JSON, CSV) if needed.

---

### **9. Optional Hyperparameter and Ablation Exploration**

- Vary: number of steering heads, alpha.
- Can rerun inference on the same test set with different configurations.
- Log all variants for comparison.

---

### **10. Final Clean-up**

- Close hooks gracefully if required.
- Release model resources.
- Summarize performance and provide final report in console or logs.

---

### **Additional Considerations**

- **Error handling**:
  - Validate dataset loading.
  - Confirm attention hook registration.
  - Handle tokenization and span extraction errors.
- **Compatibility**:
  - Ensure that all modules (`dataset_loader.py`, `model_wrapper.py`, etc.) are imported and called with correct APIs.
- **Performance efficiency**:
  - Possibly cache attention scores or profiling results.
  - Batch inference where feasible.
- **Reproducibility**:
  - Set random seeds for datasets and model inference (if applicable).
  - Log hyperparameters and profiling results.

---

### **Summary of main() logic outline**

```plaintext
- Parse config.yaml
- Initialize dataset loader
- Load datasets (train, validation, test)
- Initialize model wrapper
- Run profiling:
    * Select small profiles datasets
    * Instantiate Profiler
    * profile_heads() -> selected heads
- Setup attention steering:
    * Instantiate AttentionSteering with selected heads and alpha
- For each test sample:
    * Extract emphasis span(s)
    * Generate output with reweighted attention
- Run evaluation with metrics flags:
    * Compute accuracy, fluency, effectiveness
- Log and save results
- Finish
```

---

This detailed analysis should serve as a blueprint for implementing `main.py` in line with the methodology described in the paper, ensuring all dependencies, sequences, and hyperparameters are correctly coordinated.

## model_wrapper.py

### Logic Analysis for `model_wrapper.py`

This module implements the `ModelWrapper` class, providing core functionalities for loading pre-trained models (LLAMA-7B, GPT-J-6B), access to individual attention scores, registration of hooks for attention score manipulation, and a `generate()` method that performs inference with optional attention reweighting during decoding.

**Key Responsibilities:**
- Load the specified pre-trained model and tokenizer.
- Register hooks on attention modules to intercept and modify attention scores dynamically during inference.
- Enable extract and modification of attention scores at specified layers and heads.
- Perform text generation, incorporating attention reweighting if steering is enabled.
- Interface with `attention_steering.py` for the reweighting logic.

---

### 1. **Model Initialization**

- **Input:** `model_name`, `model_path` (from config).
- **Process:**
  - Load the model and tokenizer (from `transformers`) based on `model_name`.
  - Identify attention modules within the model architecture. For LLAMA or GPT-J, this typically involves accessing internal `model.transformer.h` (for LLAMA) or `model.transformer.blocks` (for GPT-J).
  - **Register hooks:** The hooks need to target attention modules at the layer level to allow score manipulation. 
  - The hooks will enable access to raw attention scores before softmax during execution.
- **Output:** A `ModelWrapper` object with attached hooks.

---

### 2. **Attention Score Hook Registration**

- **Implementation Details:**
  - Use `model.register_forward_hook()` or using custom attention implementation.
  - For each layer and head, hooks will:
    - Capture the attention scores `A^{(l,h)}` during forward passes.
    - Store or modify these scores for reweighting.
- **Approach:**
  - When registering hooks, specify which layers/heads to target, based on the profile.
  - The hook should expose a way to replace or modify the attention scores dynamically during inference.

---

### 3. **Accessing Attention Scores**

- **Method:** `get_attention_scores(layer, head, inputs)`
  - During inference, simulate or run forward pass with hooks active.
  - At layer `l`, head `h`, extract the raw attention score matrix `A^{(l,h)}`.
  - These scores are usually computed inside the attention module; hooks inject code to access these matrices.

- **Important:**
  - The stored attention matrices must be accessible for reweighting.
  - Possibly store references or cache the scores during the forward pass for reweighting.

---

### 4. **Inference with Attention Reweighting**

- **Method:** `generate(input_ids, emphasis_spans, alpha=0.01)`
  - **Input:**
    - `input_ids`: tokenized prompt with emphasis spans marked.
    - `emphasis_spans`: list of token indices that the user highlighted.
    - `alpha`: reweighting scalar (from config).

  - **Process:**
    - During each decoding step:
      - For each targeted layer `l` and head `h \in \mathcal{H}`:
        - Access the attention scores for current inputs.
        - Apply the reweighting transformation as per equation (2):
          \[
          \tilde{A}^{(l,h)}_{i,j} = \begin{cases}
          \alpha A_{i,j} / C_i, & j \notin \text{highlighted} \\
          A_{i,j} / C_i, & j \in \text{highlighted}
          \end{cases}
          \]
        - Recompute or override the attention scores with these reweighted scores.
        - Ensure normalization so scores sum to 1 (via `C_i`).
      - Proceed with the model's decoder step, now with modified attention scores.
    - When the last token is generated or the sequence limit is reached, output the generated text.

  - **Implementation Detail:**
    - The actual attention scores are set before the softmax operation.
    - Use hooks in the attention modules to intercept and modify scores on-the-fly.
    - It requires the model to support such hooks, or a modified attention implementation.

---

### 5. **Dynamic Head and Layer Handling**

- The list of heads to steer (`heads`) comes from the profiling step:
  - A list of tuples `(layer_idx, head_idx)`.
  - During generation, only those heads are reweighted, while others remain unaffected.
  - The design should accommodate:
    - Steering all heads in certain layers.
    - Steering only selected heads.
 
- **Optional:** Provide flexibility to steer entire layers or specific subgroupings, if needed.

---

### 6. **Handling Model Type Differences**

- **LLAMA:**
  - Internal architecture exposes attention modules typically accessed via attributes like `model.layers`.
- **GPT-J:**
  - Similar approach but may involve modules within `model.transformer.h`.
- Take care to generalize hook registration:
  - Detect the model architecture type.
  - Obtain reference to the attention module(s) for hooking.

---

### 7. **Error Handling and Robustness**

- Ensure hooks are registered only once.
- Save references to original attention matrices and hooks so they can be unregistered or reset if needed.
- Include fallback mechanisms if the attention scores are not accessible.

---

### 8. **Summary of Methods/Functions:**

| Method | Purpose | Inputs | Outputs | Remarks |
|---------|------------|--------|----------|---------|
| `__init__(model_name, model_path)` | Load model, register hooks | Model identifiers | Initialized object | Setup attention hooks here |
| `register_attention_hook(heads)` | Attach hooks to attention layers | List of `(layer, head)` | None | Hooks modify attention scores during inference |
| `get_attention_scores(layer, head, inputs)` | Retrieve attention at runtime | layer/h, inputs | attention tensor | Used during steering |
| `generate(input_ids, emphasis_spans, alpha)` | Perform inference with steering | Token ids, span indices, alpha | Generated text | Applies score reweighting during inference |

---

### 9. **Final Notes:**

- The success depends on the model architecture supporting access/modification of attention scores.
- Attention reweighting via hooks must be thread-safe and efficient.
- The approach aligns with the paper's inference-time, parameter-free adjustment framework, leveraging model internals without retraining.

---

This comprehensive analysis provides the basis for implementing `model_wrapper.py` to enable precise, efficient, and flexible attention reweighting during inference, as required by PASTA.

## profile.py

**Logic Analysis for `profile.py`: Implementation of the Profiler Class**

---

### **Objective:**

Develop a `Profiler` class that:

- Evaluates the importance or effectiveness of each attention head within the specified model for given tasks.
- Uses a small random subset of data (profiling samples) to perform this assessment.
- Modifies the model's attention heads (via `model_wrapper.py`) to measure their impact on task-specific performance.
- Aggregates results across multiple datasets/tasks if necessary.
- Determines a set of top-performing heads (`top_heads`) to be used later during inference for attention reweighting, according to a specified selection strategy.

---

### **Key Components & Workflow:**

1. **Initialization:**

- Inputs:
  - `model_wrapper`: an instance of `ModelWrapper` that provides access to the model and hooks.
  - `dataset`: dataset object containing small profile samples (`profile_samples`) from the dataset for evaluation.
  - `top_heads_count`: number of top heads to select based on profiling.
  - `profile_samples`: number of samples for profiling (from `config.yaml`).

- Parameters:
  - Strategy for head selection: default is `"top-per-task"` (but may support `"union"` or `"intersection"` if extended).

2. **Acquiring Candidate Attention Heads:**

- Retrieve the total number of layers `L` and heads `H` from `model_wrapper`.
- Generate the list of all possible `(layer, head)` pairs for the model.

3. **Profile Each Head Independently:**

- For each `(layer, head)`:

  a. **Modify Model Attention:**
  
   - Use `model_wrapper` to register hooks that can modify the attention scores at head `(layer, head)` during inference.
   
  b. **Evaluation:**

   - Run inference over all `profile_samples` from the dataset:
     - For each sample:
       - Generate output using the original prompt.
       - During generation, reweight the attention scores of only the targeted head `(layer, head)`—this may involve:
         - Activating a hook that applies a steering function (see `attention_steering.py`).
         - If multiple heads are being tested separately, ensure only the current head is modified.
       - Obtain the generated output.

  c. **Performance Measurement:**

   - Compute a task-specific metric for each sample:
     - For classification tasks: accuracy (correct label vs. model output).
     - For format correctness: validity of generated JSON.
     - For other tasks (e.g., bias detection): relevant metrics as described.
   - Aggregate the metrics across all samples:
     - Use mean accuracy or other relevant scoring aggregation.

4. **Collect Head-wise Performance Results:**

- Store the performance score for each `(layer, head)`.

5. **Rank Heads and Select Top Heads:**

- Based on performance scores, rank all heads.
- According to the `attention_steering.head_selection_strategy`:
  - `"top-per-task"`: select the top `top_heads_count` heads across all evaluated heads (possibly per task, if multi-task profiling).
  - `"union"`: union of top heads across multiple tasks (if applicable).
  - `"intersection"`: intersection of top heads across tasks.

- Store the final selected `(layer, head)` list as `self.selected_heads`.

6. **Return:**

- Output the list of selected head `(layer, head)` pairs for use during inference steering.

---

### **Implementation Details & Considerations:**

- **Model Hook Registration:**
  - Use `model_wrapper.register_attention_hook` method to modify attention scores dynamically.
  - Hooks should be set during the independent evaluation of each head and removed/updated for next head.

- **Evaluation Strategy:**
  - Keep predictions deterministic (e.g., greedy decoding).
  - For consistency, use the same dataset subset for each head.
  - Implement a timing mechanism to avoid overly long profiling (parallelization recommended).

- **Performance Metric:**
  - Select a metric aligned with the task:
    - Classification accuracy (e.g., BiasBios, CounterFact).
    - JSON validity and correctness (for JSON-formatted tasks).
    - Custom accuracy metrics as needed.
  - Log both per-head performance and overall summary.

- **Efficiency & Robustness:**
  - Profiling over 1000 samples should be optimized for batch processing.
  - Use DataLoader for batching samples.
  - For each head:
    - Set hooks, evaluate tasks.
    - Remove or reset hooks after each head profiling.
  - Store results in a data structure (e.g., list of dicts or a pandas DataFrame).

- **Handling multi-task evaluation:**
  - If multiple tasks are used for profiling:
    - Run the evaluation for each task separately.
    - Aggregate results for each head across tasks.
    - For `"intersection"` strategy: compute intersection across task sets.
    - For `"union"`: combine union of top-heads per task.

- **Resilience:**
  - Log head performance scores.
  - Provide fallback options or warnings if no heads perform well.
  
---

### **Pseudocode Outline:**

```python
class Profiler:
    def __init__(self, model_wrapper, dataset, top_heads_count=50, profile_samples=1000, strategy='top-per-task'):
        self.model_wrapper = model_wrapper
        self.dataset = dataset
        self.top_heads_count = top_heads_count
        self.profile_samples = profile_samples
        self.strategy = strategy
        self.selected_heads = []

    def profile_heads(self):
        all_heads = self._get_all_heads()
        head_performance = []

        for (layer, head) in all_heads:
            # Register hook to modify attention scores at this head
            self.model_wrapper.register_attention_hook(layer, head)
            # Evaluate performance
            perf_score = self._evaluate_head_performance(layer, head)
            head_performance.append({'layer': layer, 'head': head, 'score': perf_score})
            # Unregister hooks after evaluation
            self.model_wrapper.remove_attention_hook(layer, head)

        # Rank heads
        sorted_heads = sorted(head_performance, key=lambda x: x['score'], reverse=True)

        # Select top N heads according to strategy
        if self.strategy == 'top-per-task':
            self.selected_heads = [(h['layer'], h['head']) for h in sorted_heads[:self.top_heads_count]]
        elif self.strategy == 'union':
            # implement union for multi-task scenario, if applicable
            self.selected_heads = self._compute_union_heads(sorted_heads)
        elif self.strategy == 'intersection':
            # implement intersection
            self.selected_heads = self._compute_intersection_heads(sorted_heads)
        else:
            # default fallback
            self.selected_heads = [(h['layer'], h['head']) for h in sorted_heads[:self.top_heads_count]]

        return self.selected_heads

    def _get_all_heads(self):
        # Retrieve total layers and heads from model_wrapper
        L, H = self.model_wrapper.get_num_layers_heads()
        all_heads = []
        for l in range(L):
            for h in range(H):
                all_heads.append((l, h))
        return all_heads

    def _evaluate_head_performance(self, layer, head):
        # Run inference over limited samples, modify attention scores via hooks to only steer head
        # Return aggregated metric
        scores = []
        for sample in self.dataset.get_samples(self.profile_samples):
            # Generate output with attention modification
            output = self.model_wrapper.generate(sample['input'], emphasis_spans=None, head=(layer, head))
            score = self._compute_task_metric(output, sample['label'])
            scores.append(score)
        return sum(scores)/len(scores)

    def _compute_task_metric(self, output, label):
        # Implement task-specific evaluation, e.g., accuracy
        pass

    def _compute_union_heads(self, sorted_heads):
        # Based on task-specific logic
        pass

    def _compute_intersection_heads(self, sorted_heads):
        # Based on task-specific logic
        pass
```

---

### **Final Remarks:**

- The `Profiler` class serves as an autonomous, repeatable, one-time profiling step to identify the best attention heads for steering.
- It depends heavily on:
  - The `model_wrapper`'s ability to register hooks for individual attention heads for modification.
  - Dataset interface to provide a manageable subset for profiling.
  - Proper task-specific metric evaluation functions.
- Logging and saving profiling results are crucial for reproducibility and later analysis.
- The profile step can be parallelized to reduce total runtime, given that head evaluations are independent.

This detailed logic ensures a robust, flexible implementation aligning with the paper's methodology, dataset, and experimental design.

## utils.py

**Logic Analysis for utils.py**

---

### Purpose:
The `utils.py` module provides shared utility functions crucial for supporting the core modules in data processing, prompt span extraction, attention score manipulation, and normalization. These functions are invoked across the main orchestrator, profiling, inference, and evaluation phases to ensure consistency, robustness, and modularity.

---

### Core Utility Functions & Logic:

#### 1. **Tokenization Helpers**
- **Objective:** Convert raw text inputs into token IDs suitable for transformer models, ensuring proper handling of task-specific tokenization nuances.
- **Key Methods:**
  - `tokenize(text: str, tokenizer) -> List[int]`
    - Uses the tokenizer (from `transformers`) to encode the string.
    - Handles conversion to token IDs.
    - Supports batching if needed.
  - `detokenize(token_ids: List[int], tokenizer) -> str`
    - Converts token IDs back into the human-readable string.
- **Edge cases:**
  - Handling special tokens (e.g., `[CLS]`, `[SEP]`) if relevant.
  - Dealing with unknown tokens (fallbacks or replacements).
  - Ensuring the tokenization preserves emphasis span boundaries nicely for span detection.

---

#### 2. **Span Extraction from Styled Prompts**
- **Objective:** Parse input prompts with emphasis markers (e.g., markdown `*` or custom tags) to identify token spans for attention reweighting.
- **Functions:**
  - `extract_emphasis_spans(prompt: str, marker: str='*') -> Tuple[str, List[Tuple[int, int]]]`
    - Scans the prompt for emphasis markers.
    - Determines the start and end indices of emphasized segments in raw text.
    - **Implementation steps:**
      - Use regex to locate all marker pairs (e.g., `*...*`).
      - Record the character positions of each emphasis.
      - Map character positions to token indices via tokenization.
    - **Output:**
      - The cleaned prompt (if markers are to be removed for model input).
      - List of token index spans `(start_idx, end_idx)` representing emphasized tokens.
  - `get_token_indices_for_span(span_text: str, tokens: List[str]) -> Tuple[int, int]`
    - Matches the span text against tokenized tokens to find start and end token indices.
- **Edge cases:**
  - Multiple emphasized spans.
  - Overlapping or nested emphasis markers.
  - Ambiguous spans (e.g., multi-word emphasis).

---

#### 3. **Normalization Functions**
- **Objective:** Normalize attention scores for reweighting purposes to ensure they sum to 1 per query token (row-wise softmax compatibility).
- **Method:**
  - `normalize_attention_scores(scores: Tensor) -> Tensor`
    - Implements row normalization by dividing each row by its sum.
    - Ensures numerical stability (e.g., adding a small epsilon).
  - `scale_attention_scores(scores: Tensor, alpha: float, mask: Tensor) -> Tensor`
    - Applies reweighting:
      - For tokens outside emphasized spans: multiply scores by `alpha`.
      - For emphasized tokens: keep scores as-is or multiply by 1.
    - Follow up with normalization.
- **Edge cases:**
  - Zero scores (avoid division by zero).
  - Very small or large values; handle with clamping or epsilon adjustments.

---

#### 4. **Emphasis Span Mappings**
- **Objective:** Map span character indices or token positions from prompt annotation to token IDs, ensuring precise targeting.
- **Methods:**
  - `map_char_span_to_token_indices(char_start: int, char_end: int, token_offsets: List[Tuple[int, int]]) -> Tuple[int, int]`
    - `token_offsets` map tokens to character positions.
    - Finds the token indices corresponding to character span.
- **Implementation notes:**
  - Precompute token offsets during tokenization.
  - Ensure the span matches selected tokens exactly, or use heuristic overlap if approximate.

---

### Additional Considerations:
- **Handling Variations in Emphasis Markers:**
  - The utility functions should be flexible to adapt to various emphasis markup styles (e.g., `*`, `"``, custom tags).
  - Standardize the marker to prevent errors.

- **Use of Tokenizer API:**
  - Must work with the tokenizer compatible with the model (e.g., HuggingFace tokenizer).
  - Use encoding with `return_offsets_mapping=True` to facilitate span mapping.

- **Efficiency and Robustness:**
  - Minimize repeated tokenization by caching token offsets.
  - Handle malformed inputs gracefully (missing markers, unmatched emphasis pairs).

---

### Summary:
`utils.py` encapsulates:
- Tokenization and detokenization helpers.
- Span extraction and mapping based on style markers.
- Functions to convert emphasis spans into token indices for precise attention manipulation.
- Normalization routines to ensure attention scores are valid probability distributions.
- These functions underpin the core logic of emphasis span detection and attention score reweighting, enabling the main modules to perform inference with user-guided attention steering reliably and efficiently.

---

This detailed analysis informs the implementation of robust, efficient, and flexible utility functions central to executing PASTA as described in the paper.

