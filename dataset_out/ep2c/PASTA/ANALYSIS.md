# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## attention_module.py

{
  "attention_module.py": "Logic Analysis\n\nObjective:\n\nThe purpose of this module is twofold:\n1. To facilitate the extraction of per-layer, per-head attention scores during inference for a given model.\n2. To provide the functionality to modify (reweight) these attention scores dynamically during inference, based on user-highlighted tokens, in accordance with the PASTA approach.\n\nKey Tasks:\n\n1. Attention Score Extraction\n2. Attention Reweighting Application\n\n---\n\n1. Attention Score Extraction:\n\n- Need to hook into or access the model's attention outputs during inference.\n- For models built with Hugging Face's Transformers library, this is typically achieved via forward hooks or by requesting the model to output attentions explicitly.\n- Assume models are loaded with `output_attentions=True` to obtain attention matrices.\n- Attention matrices are usually in the shape:\n  `[batch_size, num_heads, seq_length, seq_length]`\n- We focus on a single batch at a time during inference.\n- For each forward pass, extract and store attention scores for each layer and head.\n- Map these scores to the corresponding tokens: need to handle tokenization (input IDs to tokens). The attention matrices are indexed over tokens; the tokens are ordered as per the tokenizer.\n- These attention scores are raw logits before softmax on attention scores.\n\nImplementation considerations:\n- To minimize performance overhead, collect attentions during inference via model's built-in output if available, or via hooks.\n- Provide a method: `extract_attention_scores(input_ids, attention_mask)` that returns a structured container `attention_scores`:\n  ```python\n  attention_scores: List[Dict[Tuple[int,int], Tensor]]\n  # List over layers.\n  # Each element: dict with keys tuple(layer, head), value: attention matrix of shape [batch_size, seq_len, seq_len]\n  ```\n- Maintain consistency: e.g., store attention scores for the current inference, not accumulate over multiple steps.\n\n2. Attention Reweighting Application:\n\n- Input: list of attention scores (per layer, per head), the highlighted tokens set `𝒢`, and coefficient `α`.\n- For each layer and head:\n  - For each token `i` in the sequence (corresponding to input token position):\n    - For each attention target token `j`, modify the attention score `A_{ij}`.\n    - The reweighting follows the formula:\n      \n      c_i = sum_{j} [\textbf{A}_{ij}] where:\n        - For tokens `j` in 𝒢 (highlighted): scale by 1/α.\n        - For tokens `j` not in 𝒢: scale by α.\n      \n      Then normalize: \n      \n      \[\n      \widetilde{A}_{ij} = \frac{A_{ij} \times s_j}{C_i}\,\text{, where}\,\ s_j = 1\,\text{if}\, j \in 𝒢,\ \text{else}\,\ \alpha\n      \]\n\n- Implementation details:\n  - For each layer, head, and batch:\n    - Compute the scaled attention score matrix accordingly.\n    - Perform normalization row-wise (across `j`) to ensure each attention distribution sums to 1.\n  - Replace the original attention scores with the reweighted ones for the inference step.\n- To integrate reweighted attention into the model:\n  - Either modify model's internal attention scores during the forward pass via hooks.\n  - Or, modify the attention logits before softmax in the custom attention computation.\n  - This might involve defining a wrapper or a custom attention layer, or using hooks to patch the attention before softmax.\n\n3. Supporting Functionality:\n\n- Provide clear interfaces:\n  - `get_attention_scores(input_ids, attention_mask)` - returns raw attention scores.\n  - `apply_attention_reweighting(attention_scores, highlighted_tokens, alpha)` - applies the reweighting and returns modified attention scores.\n- Ensure that during inference, the attention reweighting is applied just before the softmax step in the attention module.\n- Implementation should be compatible with models in the transformers library:\n  - Use `register_forward_hook` for attention score capture.\n  - Or, if the model supports `output_attentions=True`, simply access the `attentions` attribute after forwarding.\n\n4. Additional Considerations:\n\n- Efficiently handle batching: attention scores are per batch.\n- Correct token alignment: map user-highlighted spans (which may be at token or character level) to token indices, assuming the user inputs are already marked with emphasis markers that align with token positions.\n- Compatibility: ensure methods do not break existing model behavior.\n- Data structures:\n  - Use dictionaries or nested lists indexed by layer and head.\n  - Store attention matrices as tensors for easy manipulation.\n  - Maintain a set or list of highlighted token indices (`𝒢`), derived from user input.\n\n5. Summary:\n\n- The module must provide:\n  - A method to extract attention scores during model inference.\n  - A method to reweight these scores based on highlighted tokens with coefficient `α`.\n  - The reweighted scores are fed into the model's subsequent next-state computations during inference.\n- Integration should be seamless, supporting batch inference, and only modifying attention scores at the specified heads/layers as profiled.\n\n---\n\nIn conclusion, this module facilitates the critical steps of extracting attention matrices, applying precise reweighting based on user emphasis, and ensuring the modified attention is used during inference, thus enabling PASTA's post-hoc attention steering as described in the paper."

## dataset_loader.py

**Logic Analysis for `dataset_loader.py`**

---

### Objective:

Implement a `DatasetLoader` class to load datasets required by the experiments, parse raw data files, prepare train/validation/test splits, and provide datasets compatible with downstream modules. The class must support multiple tasks (`BiasBios`, `CounterFact`, `JSON Formatting`, `Pronouns Changing`) with datasets formatted as per the appendix and experimental setup.

---

### Core Responsibilities:

1. **Dataset Initialization:**
   - Take as input dataset paths and task identifiers.
   - Load raw data files from specified paths.
   - Support custom formats aligned with the dataset descriptions and prompt templates.

2. **Data Parsing and Formatting:**
   - For each task, parse dataset entries, extracting:
     - Input context (biographical info, context paragraph, etc.)
     - Highlighted spans (`𝒢`) marked explicitly (e.g., via emphasis tokens or annotations).
     - Target outputs (occupation, pronoun change, JSON output, correctness labels, etc.)
   - Ensure data is cleaned and tokenized appropriately (or at minimum, kept raw for later tokenization by tokenizer module).

3. **Task-specific Dataset Construction:**
   - Construct data samples as dictionaries or custom objects with fields:
     - `input_text`: raw input prompt with embedded emphasis markers where applicable.
     - `target_text`: expected output or label.
     - `highlighted_spans`: info on which tokens/parts are emphasized for attention steering.
     - `task_type`: identifier for downstream evaluation and metric selection.
   - For datasets with special formatting (e.g., JSON), validate JSON correctness if needed or prepare examples for accuracy evaluation.

4. **Splitting Data into Train/Validation/Test:**
   - Load the full dataset.
   - Split into train, validation, test sets following the ratios specified (e.g., train=1000, val=1000, test=5000).
   - Ensure reproducibility—set a fixed seed for splits.

5. **Data Compatibility:**
   - Optionally convert datasets into Hugging Face `datasets.Dataset` objects for seamless integration.
   - Support batch retrieval and iteration suitable for profiling, training, and evaluation.

6. **Handling Task-specific Details:**
   - **JSON Formatting:** Parse dataset entries with `{"name": ..., "occupation": ...}`, store as raw data, prepare prompts accordingly.
   - **Pronouns Changing:** Include original and changed pronouns, store both input version and change instructions.
   - **BiasBios:** Load biographical text, annotate target occupation, preserve context.
   - **CounterFact:** Load tuples with old and new facts, mask relevant input spans if necessary.

7. **Effectively support emphasis markers:**
   - Incorporate emphasis markers into prompts during data loading if dataset annotations are not already in emphasized form.
   - Maintain consistent formatting to facilitate attention reweighting.

8. **Maintain Data Integrity & Efficiency:**
   - Cache datasets after loading to avoid repeated I/O.
   - Output datasets in structures that simplify downstream processing (list of dicts or `datasets.Dataset`).

9. **Additional Considerations:**
   - Provide utility functions to:
     - Load raw files (e.g., JSON, CSV, or custom).
     - Apply necessary parsing rules (e.g., identify emphasis regions).
     - Convert raw data into model-ready samples.
   
10. **Exception Handling & Defaults:**
    - Handle missing files gracefully.
    - Validate dataset completeness and format integrity.
    - Use reasonable default splits if dataset is already pre-split.

---

### Implementation Outline:

1. **Constructor (`__init__`)**
   - Accepts parameters:
     - `dataset_paths`: dictionary with keys (`bias_bios`, `counterfact`, `json_format`, `pronouns_changing`).
     - `task_name`: string to specify which dataset to load.
     - Optional: `split_ratios`, `seed`.

2. **Methods:**

- `load_dataset()`:
  - Selects the appropriate internal parsing method based on `task_name`.
  - Loads raw data file(s).

- `parse_bias_bios()`
  - Loads biographical data, extracts context and occupation, identifies emphasis markers if any.

- `parse_counterfact()`
  - Loads tuples of (old fact, new fact), process prompt accordingly.

- `parse_json_format()`
  - Loads biographical info, associates with instruction prompts, validates or structures JSON examples.

- `parse_pronouns_changing()`
  - Loads biographical data with pronouns, generates annotated samples with pronoun change instructions.

- `split_dataset()`
  - Performs train/val/test splits.
  - Uses fixed seed for reproducibility.
  - Returns three datasets (lists or `datasets.Dataset`).

3. **Data Representation:**
   - Use dictionaries for each sample:
     ```python
     {
        'input_text': str,
        'target_text': str,
        'highlighted_spans': list of token indices or span annotations,
        'task_type': str
     }
     ```
   - For dataset processing, convert to `datasets.Dataset` if desired.

4. **Validation/Utility:**
   - Implement functions to verify JSON validity, leverage `json.loads()` for JSON outputs.
   - Logic for token-level emphasis annotation (if dataset annotations are in raw text).

5. **Caching & Loading:**
   - Support saving preprocessed datasets.
   - Load from cache if available.

---

### Additional Notes:

- Make sure to handle the diverse dataset annotations and formats outlined in Appendix A.
- For datasets requiring emphasis markers, assume dataset includes these markers explicitly, or insert during parsing if raw text is available.
- Maintain consistent interface: after loading, datasets should be ready for batch tokenization and feeding into models.
- Since the `DatasetLoader` will serve multiple tasks, integrate a flexible task parser dispatch mechanism.

---

### Summary:

- The `DatasetLoader` class will centrally load and prepare datasets as required.
- Load datasets according to paths specified in the config.
- Parse raw data carefully, respecting task-specific formats.
- Create structured samples with emphasis markers.
- Split datasets into train/validation/test with reproducibility.
- Convert to compatible formats for downstream processing.
- Ensure robustness and modularity for easy extension and debugging.

This thorough logic facilitates accurate, efficient, and reproducible data handling consistent with the paper's methodology and experimental design.

## evaluation.py

### Evaluation.py Logic Analysis

The `evaluation.py` module is responsible for implementing the `Evaluation` class and associated functions to measure the performance of the PASTA inference pipeline on the four key tasks: JSON Formatting, Pronouns Changing, BiasBios, and CounterFact. The class will compute multiple metrics including accuracy, task-specific quality scores (e.g., format accuracy, efficacy score), fluency, and consistency.

Below is a structured, detailed breakdown of the logic, methods, inputs, outputs, and design considerations for `evaluation.py`.

---

## 1. **Overall Design and Class Responsibilities**

- **Main class:** `Evaluation`
  
  - Initialized with:
    - The trained model instance or inference outputs (generated texts).
    - The dataset (test splits).
    - Task-specific configuration parameters.
    - Relevant task metrics to compute.
  
  - Core methods:
    - `evaluate_task()` — Runs all required metrics for the task.
    - `compute_metrics()` — Returns a dictionary of all metrics for reporting.

- **Supporting functions/functions** for:
  - Computing accuracy for classification tasks.
  - Validating JSON format output.
  - Calculating fluency (entropy of n-grams).
  - Measuring content similarity for `confidence/consistency`.
  - Computing efficacy and paraphrase scores for knowledge change tasks.
  - Handling dataset predicates, such as extracting relevant input spans, ground truths, and reference texts.

---

## 2. **Inputs for Evaluation**

- **Generated texts (`generation_output`)**: The model outputs obtained after applying attention reweighting. These are typically strings generated by the model, possibly stored as logs, or directly captured during inference.
  
- **Ground-truth labels/annotations (`ground_truth`)**: For each task, this includes:
  - For JSON formatting: correct JSON objects and validation key.
  - For pronouns changing: known correct pronoun replacements.
  - For BiasBios: correct occupation labels.
  - For CounterFact: old and new target facts, reference answers.

- **Task-specific parameters/configurations**: e.g., JSON format parsing rules, span extraction info, reference references for content similarity.

---

## 3. **Metrics Calculation Breakdown**

### 3.1. Accuracy (classification tasks)

- **JSON Formatting:**
  - **Format accuracy:** check if the output is a valid JSON object.
  - **Prediction accuracy:** compare parsed JSON fields to ground truth, e.g.:
    - Extract expected occupation.
    - Parse model output JSON (with error handling).
    - Compute number of correct matches.
  
- **Pronouns Changing:**
  - **Accuracy in pronoun substitution:**
    - Use regex or NLP tokenization to detect whether pronouns `she/he` have been correctly replaced with `they`.
    - Measure exact match, e.g., counting correct replacements.
  - **All-changed accuracy:**
    - Verify if all relevant pronouns are replaced correctly.
    - Count as correct only if all replacements match expected.

- **BiasBios Classification:**
  - Use predicted probabilities or labels (possibly from model output or directly from prompt). If the task involves classification scores, check whether the highest probability label matches ground-truth occupation.

- **CounterFact:**
  - **Efficacy score (ES):**
    - For each example, compare model probabilities (or answer correctness) when prompted about the new versus old facts.
    - Count as success if the model prefers the new target.
  - **Paraphrase score (PS):**
    - Same as ES but with rephrased questions, assessing the model's robustness across questions.

### 3.2. Fluency

- **Entropy-based Fluency Score:**
  - **Input:** generated text.
  - For each generation:
    - Tokenize the text into words or subwords.
    - Compute bigram and trigram probabilities.
    - Calculate entropy over these n-grams.
  - **Output:** average entropy score.
  - **Filtering:** exclude outputs with entropy score < 3.0.

### 3.3. Content/Confidence/Consistency

- **Consistency Score:**
  - Use TF-IDF vectorization:
    - Vectorize the generated text.
    - Vectorize reference dataset texts (e.g., all references for the task).
    - Compute cosine similarity or TF-IDF similarity.
  - **Outcome:** Higher similarity indicates better content retention.

---

## 4. **Implementation Details & Steps**

### 4.1. Data Preprocessing

- Load generated outputs (strings).
- Parse outputs where necessary (e.g., JSON format validation).
- For classification or attribute matching, extract relevant fields.

### 4.2. Metric functions

- **validate_json(output: str) -> bool:**
  - Tries to parse output as JSON.
  - Checks for validity (syntax, required fields).
  - Returns True/False.

- **compute_accuracy_json(generated: str, ground_truth: dict):**
  - Uses `validate_json()`.
  - Extracts fields and compares with ground truth.

- **compute_pronoun_change(generated: str, target_pronouns: list, expected: dict):**
  - Counts correct pronoun replacements.
  - Checks all/partial correctness.

- **compute_biasbios_accuracy(generated: str, target_label: str):**
  - Implement simple classification correctness.
  - Could use probabilities if available, or string matching.

- **compute_counterfactual_scores(generated: str, old_fact, new_fact, question):**
  - Determine whether model prefers new fact.
  - Count as success if model's output assigns higher probability to new fact compared to old.

- **compute_fluency(generated: str):**
  - Generate n-grams.
  - Compute their entropy.
  - Average over dataset.

- **compute_consistency(generated_texts: list, reference_texts: list):**
  - Vectorize via TF-IDF.
  - Compute cosine similarity or tf-idf similarity.

### 4.3. Handling Benchmark Conditions

- Implement mechanisms to **skip low-fluency outputs**.
- For **complex answer validation** (like JSON), handle parsing errors gracefully.

### 4.4. Results Compilation

- Store individual metric scores in a results dictionary.
- Present results per task, with overall averages if needed.
- Ensure metrics are compatible with the paragraph's example scores.

---

## 5. **Edge Cases and Robustness**

- Invalid JSON outputs (parse failures): count for format accuracy and set prediction accuracy to zero.
- Rephrasing or paraphrased prompts: evaluate robustness by multiple runs if needed.
- Repeated or degenerate generations (e.g., low entropy): mark as invalid or discard.
- Attention effects may influence output variability; consider multiple inference runs for statistical stability.

---

## 6. **Final Notes**

- The implementation should be modular:
  - Separate functions for each metric.
  - Clear interface for passing generated texts and relevant references.
- Use existing NLP libraries for tokenization, JSON parsing, TF-IDF, and entropy calculations.
- Focus on robustness, error handling, and flexibility for extending other tasks or metrics.
- Report detailed logs or intermediate results for debugging and variance analysis.

---

This detailed logic analysis guides the development of `evaluation.py` by clarifying the required operations, data flow, and evaluation standards, ensuring accurate reproduction of the paper's experimental assessment.

## main.py

# Logic Analysis for main.py

This script serves as the primary entry point to coordinate the entire process—from loading configurations, datasets, and models; profiling attention heads; performing inference with attention steering; and evaluating results—by orchestrating all modules and workflows specified in the plan. The logic must follow the detailed experimental pipeline described in the paper and adhere to the design interface.

---

## 1. **Parsing Command-Line Arguments and Configuration**

- **Objective:** Enable flexible execution modes, e.g., profiling, inference, evaluation, or combined execution.
- **Implementation:**
  - Use `argparse` or a similar CLI parser.
  - Define arguments such as:
    - `--config`: Path to the YAML configuration file (`config.yaml`).
    - `--tasks`: Optional list of tasks to run; default includes all.
    - `--do_profiling`: Boolean flag to indicate whether to perform head profiling.
    - `--do_inference`: Whether to run inference with attention steering.
    - `--do_evaluation`: Whether to evaluate model outputs.
    - `--profile_tasks`: Which tasks to profile on.
    - `--test_tasks`: Which tasks to evaluate on.
    - `--load_profile`: Path to precomputed profiling results.
    - `--k_heads`: Number of heads to select for steering (if not specified, use config).
  - Load and validate the arguments and the configuration file.

---

## 2. **Loading Configuration**

- Read `config.yaml`.
- Extract major parameters:
  - Model info (`model_name`, `device`, hyperparameters).
  - Dataset paths.
  - Hyperparameters (`alpha`, `top_k_heads`, `max_sequence_length`, `profiling_samples`).
  - Evaluation settings.
  - Prompts templates for each task.
- Validate consistency:
  - Ensure model path or name is correct.
  - Dataset paths exist.
  - Hyperparameters are within reasonable ranges.

---

## 3. **Initializing Modules**

- **DatasetLoader:**
  - Instantiate with dataset paths and split ratios.
  - Load datasets for each task:
    - BiasBios, CounterFact, JSON Formatting, Pronouns Changing.
  - Prepare datasets for profiling (small subset, e.g., 1000 samples) and test (full or subset).
- **Model:**
  - Instantiate with model name, device.
  - Load pretrained weights.
  - Set model to evaluation mode (`model.eval()`).
  - Ensure hooks are attached to extract attention scores during inference.
- **PromptBuilder:**
  - For each task, prepare prompt templates (from config or appendix).
  - Generate prompts given input texts and highlighted spans.
- **ProfileAnalyzer:**
  - If profiling is required:
    - Use the small profiling dataset.
    - Run profiling: evaluate the impact of steering each head on each task.
    - Obtain rankings, select top `k`.
    - Save head profile for future inference.
- **AttentionReweighter:**
  - Initialized with selected head indices and `α=0.01`.
  - Provides method to reweight attention scores during inference based on highlighted tokens.

---

## 4. **Profiling Attention Heads (if requested)**

- **Objective:** Identify which attention heads to steer per task.
- **Procedure:**
  - Loop through the profiling dataset.
  - For each example:
    - Prepare input (prompt + highlighted spans).
    - Run a forward pass with the model:
      - Extract raw attention scores (`A^{(l,h)}`).
    - For each head:
      - Apply the performed `T(A)` reweighting according to the highlighted tokens.
      - Measure performance impact (task-dependent metric).
  - Aggregate results and rank heads.
  - Choose the top `k` heads that consistently improve performance.
- **Output:**
  - Save selected head indices to file (`profile_heads.json` or similar).

---

## 5. **Inference with Attention Steering**

- **Input:** New prompt (with emphasis markers) and highlighted spans.
- **Process:**
  - Generate the prompt string for each example using `PromptBuilder`.
  - Pass prompt through the model:
    - During each model call:
      - Hook into the attention scores.
      - For each head `(l,h)` in the preselected set:
        - Reweight the attention scores using `apply_masking()`:
          - Scale scores for tokens outside `𝒢` by `α`.
          - Normalize per-head.
        - Replace the attention scores in the model's internal state (using hooks or by passing in modified scores if the API allows).
    - Run inference (greedy or beam search).
  - Collect generated outputs.

---

## 6. **Evaluation and Metrics Calculation**

- For each task:
  - Use the generated outputs.
  - Compute accuracy, prediction correctness, JSON validity, pronoun change correctness, etc.
  - Fluency and consistency:
    - Calculate n-gram entropy for fluency.
    - Calculate tf-idf similarity for consistency.
  - Filter out samples with fluency < 3.0 (if required).
- Summarize results across dataset splits.
- Store and log detailed results, including variances and robustness measures.

---

## 7. **Handling Multiple Modes & Reusability**

- **Profiling and Inference:**
  - If `do_profiling`:
    - Run profiling, save head rankings.
  - Else if `load_profile` is provided:
    - Load precomputed head indices.
  - Else:
    - Use default or previously saved heads.
- **Inference & Evaluation:**
  - Run inference for test tasks.
  - This can be parallelized per task or example.
  - Save outputs for analysis.

---

## 8. **Output and Logging**

- Print summaries of:
  - Selected attention heads per task.
  - Performance metrics.
  - Variance analysis.
- Save detailed logs and results in structured format (CSV, JSON).

---

## 9. **Special Considerations**

- **Attention score extraction:**
  - Confirm that the model supports hooks to access attention during inference.
  - Support for models from Huggingface transformers.
- **Interfacing with model forward:**
  - May require custom wrapper to replace attention scores during inference.
  - Use hooks or modify the forward pass if supported.
- **Reweighting operation:**
  - Implemented per the `T(A)` formula in the paper, ensuring proper normalization.
- **Memory management:**
  - For large models, ensure device allocation and batch inference are optimized.
- **Hyperparameter consistency:**
  - Always refer to `config.yaml` for parameters like `α`, `top_k_heads`, etc.

---

## 10. **Summary**

This main.py must act as the orchestrator:

- Parse args and load config.
- Load datasets and models.
- Perform profiling if required.
- Select relevant heads (via profile or load).
- For each task:
  - Generate prompts with emphasis markup.
  - Run model inference with attention reweighting applied to selected heads at every step.
  - Collect outputs.
  - Evaluate with task-specific metrics.
- Summarize and report results.

All steps should be implemented to closely follow the experimental design and methodology outlined in the paper, ensuring reproducibility and fidelity to the original approach.

---

This comprehensive logic analysis guides the implementation of main.py to meet the experimental and methodological standards of the paper.

## model.py

{
  "file": "model.py",
  "purpose": "Implement the Model class responsible for loading pre-trained language models (LLAMA, GPT-J, Vicuna), extracting attention scores, applying attention score modifications (reweighting/steering), and managing device placement. This class will serve as the main interface for inference and attention manipulation during the experimental pipeline.",
  "core functionalities": [
    "Loading pretrained models with transformers, ensuring that attention outputs are accessible during inference.",
    "Providing methods for extracting raw attention scores for any given input batch, for each layer and attention head.",
    "Implementing mechanisms for modifying attention scores during inference based on user-highlighted spans, using the configured reweighting scheme.",
    "Supporting flexibility for different models (LLAMA, GPT-J, Vicuna) with potential differences in API or attention output structure.",
    "Device management: moving models to GPU/CPU as specified in the configuration.",
    "Providing a clean interface to generate outputs given input tokens, with optional attention score modifications.",
    "Ensuring compatibility with the attention reweighting and profiling modules."
  ],
  "details": [
    "Model Initialization:",
    "  - Input: model_name (e.g., 'llama-7b', 'gpt-j-6b', 'vicuna-7b') and device (e.g., 'cuda' or 'cpu') from the config file.",
    "  - Load pretrained transformers models: check whether models are available in Hugging Face or require custom loading.",
    "  - Load tokenizer along with the model to handle tokenization consistent with training or fine-tuning.",
    "  - Set the model to evaluation mode (`model.eval()`) to disable dropout.",
    "  - Implement device placement: `model.to(device)`.",
    "  - Implement access to attention scores: use model hooks or transformer config options",
    "    to extract attention matrices during the forward pass.",
    "Extraction of Attention Scores:",
    "  - During inference, register hooks on the model's attention modules (e.g., `model.transformer.h` or similar) that save attention scores (logits or probability matrices).",
    "  - The method `extract_attention()` should return a nested structure: a list (per layer) of attention matrices for each head (layer H, heads H).",
    "  - Attention matrices have shape `[batch_size, seq_len, seq_len]`.",
    "Modification of Attention Scores (during inference):",
    "  - Provide a method to receive raw attention scores and apply the reweighting scheme (as specified in the paper), based on the highlighted token indices (`𝒢`).",
    "  - For each head, modify the attention scores by scaling non-highlighted tokens by `α` (from config) and normalize using the `C_i` constant per token.",
    "  - The modified scores are then used to produce the attention output once reweighted, ideally during the forward process.",
    "  - This can be achieved either by:",
    "    a) Intercepting the attention score computation (via hooks or custom forward functions), and modifying the scores before softmax,",
    "    or",
    "    b) Overriding the attention computation to use custom reweighted scores, ensuring only the targeted heads/layers are affected.",
    "  - Note: Care must be taken to apply reweighting per head; the structure of stored attention scores might be a list or dict with layer and head indices.",
    "Handling Different Models:",
    "  - Since model APIs vary, implement model-specific handling if necessary (e.g., LLAMA models might have different attention output formats than GPT-J).",
    "  - Possibly implement model backend-specific functions to standardize extraction and injection of attention scores.",
    "Inference method:",
    "  - `generate()` method that takes input tokens, optional highlighted spans, applies the attention reweighting if needed, and produces output sequence.",
    "  - It should handle batching (if supported), attention score extraction, reweighting, and decoding with greedy or beam search.",
    "  - Maintain a clean separation between raw model inference and attention modulation to facilitate debugging and profiling.",
    "Device management:",
    "  - The model class should provide methods `to(device)` to handle moving the model to GPU or CPU.",
    "  - When reweighting attention, ensure the tensor operations happen on the same device.",
    "Error handling and compatibility:",
    "  - Check for model-specific modules or attributes; raise informative errors if features are unavailable.",
    "  - If the model does not support returning attention scores natively, the class should either:",
    "    a) Use transformers hooks to capture attention during the forward pass, or",
    "    b) Wrap the model to inject attention reweighting at the correct step.",
    "  - Consider optional flags or parameters for debugging, e.g., `return_attention=True` for debugging or profiling.",
    "Output:",
    "  - The class should return generated text, possibly also returning raw and modified attention matrices if requested for debugging and profiling.",
    "Validation:",
    "  - Verify the attention reweighting by inspecting attention matrices pre- and post-application to ensure correct scaling.",
    "Testing:",
    "  - Include unit tests that check if given inputs, attention matrices are appropriately extracted and modified, ensuring the logic matches the reweighting formulas (especially normalization)."
  ],
  "additional considerations": [
    "Support for batch inference: if batch size > 1, attention reweighting must be applied per sample, considering each sample's highlighted tokens.",
    "Efficiency: minimize overhead during inference; attention hooks should be registered once, and reweighting should happen in-place.",
    "Compatibility: models may have built-in support for attention extraction or may require custom hooks; handle both cases.",
    "Hyperparameters (like alpha): fixed at 0.01, but provide interface to override if needed for experimentation.",
    "Logging: include optional verbose/debug mode to output attention scores before and after reweighting for debugging.",
    "Memory management: ensure attention score tensors are freed or detached when no longer needed.",
    "Extensibility: design with clear interfaces so that future models or different attention computation methods can be integrated easily."
  ],
  "summary": "The `model.py` file defines a `Model` class that loads pre-trained models, manages attention extraction via hooks, provides methods for attention reweighting according to highlighted user spans, and supports inference with optional attention modifications. The class should keep device placement robust, handle model-specific differences, and provide interfaces for integration with profiling and steering modules, adhering to the outlined design and hyperparameters."
}

## profiling.py

**Logic Analysis of `profiling.py` for Implementing the `ProfileAnalyzer` Class**

---

### **Purpose & Role**
The `ProfileAnalyzer` class is designed to identify the most effective attention heads within a pre-trained LLM that contribute positively to task performance when steered. This is achieved through a systematic profiling process that:
- Uses a small subset of training data per task.
- Evaluates each attention head's impact on performance when its attention scores are manipulated.
- Ranks heads based on their contribution.
- Stores the resulting head profiles for subsequent use in attention steering during inference.

---

### **Key Functional Responsibilities**
1. **Initialization & Setup**
   - Accepts a pretrained `Model` instance capable of extracting attention scores.
   - Receives a designated profiling dataset (`Dataset` object).
   - Uses hyperparameters such as `top_k` (number of heads to select), which influence the profiling process.
   - Loads or prepares a structure for storing profiling results—likely a dictionary or JSON file for persistence.

2. **Attention Head Profiling Procedure**
   - **Iterate over each sample** in the small profiling dataset.
   - For **each layer `l`** and **each attention head `h`**:
     - **Inject a modification** into the attention scores:
       - Apply attention reweighting as per the described method:
         - Downscale non-highlighted tokens by `α`.
         - Keep highlighted tokens unscaled or scaled up.
     - **Evaluate the performance** after steering:
       - This could be accuracy (classification), JSON validity, or other relevant metrics per task.
       - The evaluation must be **task-dependent**; the dataset and metric functions must be compatible.
     - **Record the performance metric** associated with steering head `(l,h)`.

3. **Head Ranking and Selection**
   - After profiling each head `(l,h)` across all samples:
     - **Aggregate performance** scores:
       - For example, compute the average score across all samples for each head.
     - **Rank heads** within each layer based on their average performance.
   - For **multi-task profiling**, aggregate across all tasks:
     - Perform cross-task head ranking, e.g., by taking the intersection or union of top-performing heads across all tasks.
     - As per the paper, generally, the **intersection of top-`k` heads** across multiple tasks results in robustness.
   - **Select the top `k` heads** based on the ranking:
     - Store the indices (`(l,h)` pairs) of selected heads.

4. **Persistence & Reuse**
   - Save the profiling results (list of selected heads) to disk (e.g., a JSON file).
   - Allow re-loading for subsequent inference tasks or evaluations.
   - The profiling process is **performed once per model** and **not** during each inference, making it akin to a one-time setup or calibration.

---

### **Implementation Details & Considerations**

- **Extracting Attention Scores:**
  - Use hooks or wrappers provided by `model.py` or `attention_module.py` to access raw attention matrices `A^{(l,h)}`.
  - Hooks should be designed to:
    - Capture attention weights for each forward pass during profiling.
    - Store attention scores temporarily for evaluation.

- **Applying Attention Reweighting during Profiling:**
  - For each sample:
    - Use the user-emphasized tokens `𝒢`; relevant if the sample is annotated for highlights.
    - For each `(l,h)`:
      - Generate reweighted attention scores `T(A^{(l,h)})` as per the formula:
        \[
        [\mathcal{T}(A)]_{ij} = 
        \begin{cases}
        \frac{A_{ij}}{C_i}, & j \in 𝒢 \\
        \frac{\alpha A_{ij}}{C_i}, & j \notin 𝒢
        \end{cases}
        \]
        with normalization constant:
        \[
        C_i = \sum_{j \in 𝒢} A_{ij} + \alpha \sum_{j \notin 𝒢} A_{ij}
        \]
    - Set these scaled scores in the model for the current inference step to evaluate performance.

- **Evaluation of Steering Effect:**
  - For each `(l,h)` and each dataset/sample:
    - Perform a model inference with the attention scores modified.
    - Compute task-specific metrics (accuracy, JSON correctness, etc.).
    - Record performance results per head and sample.

- **Aggregation & Ranking Metrics:**
  - Compute mean or median performance per head across all samples.
  - Possibly normalize if needed to compare heads.

- **Head Selection Strategy:**
  - For multiple tasks, determine whether to:
    - Take the *intersection* of top-`k` heads per task, ensuring heads are generally effective across tasks.
    - Or use a *union* for broader coverage.
  - Store the final set of heads in an accessible format (`profiles.json`).

- **Handling Multiple Layers & Heads:**
  - Loop through layers `l=1..L`.
  - Loop through heads `h=1..H`.
  - Efficiently store the performance metrics in a multi-dimensional structure or a flat list with `(l,h)` keys.
  - Use ranking algorithms (sort by performance) to pick top heads.

- **Scalability & Efficiency:**
  - Profiling over hundreds of heads and thousands of samples is computationally intensive.
  - Use batching and parallelization where possible.
  - Optionally, cache attention scores during the profiling run to avoid repeated extraction.

---

### **Error Handling & Validation**
- Validate that extracted attention scores are correctly shaped (e.g., `[batch_size, seq_len, seq_len]`).
- Ensure that attention reweighting maintains valid probability distributions post-normalization.
- Confirm that performance metrics are correctly computed and that steering improves task-specific performance metrics.

---

### **Summary of Core Logic in `profiling.py`**

| Step | Description | Implementation Notes |
|-------|-----------------|----------------------|
| Initialization | Load model, dataset, hyperparameters | Accept hyperparameters such as top_k, alpha |
| Data Processing | Load small profiling dataset | Subsample 1000 samples per task |
| For each sample | Extract attention matrices | Use model hooks for attention extraction |
| For each head `(l,h)` | Apply reweighting `T(A)` | Recompute attention with scaled scores |
| Inference + Evaluation | Generate output, compute metrics | Use task-specific evaluation functions |
| Record performance | Store head `(l,h)` score | Aggregate performance scores |
| After all samples | Calculate average scores | Rank heads by mean performance |
| Cross-task aggregation | Compute intersection or union of top heads | Final head set `H` for steering |
| Save profile | Persist selected heads | Save as JSON or similar |

---

### **Additional Notes & Clarifications Needed**
- Confirm accessing attention scores: Is it via forward hooks in current Huggingface models?
- How are highlighted spans specified during profiling? (e.g., for tasks like JSON formatting, highlighting the instruction or key terms)
- How to align the evaluation pipeline to ensure precise measurement of steering impact?
- Are there existing utility functions in `attention_module.py` for computing the scaled matrices?

---

This detailed logic analysis forms the foundation for implementing `ProfileAnalyzer` to systematically identify the critical attention heads for post-hoc steering, ensuring the pipeline's fidelity with the described methodology.

## steering.py

{
  "class": "AttentionReweighter",
  "purpose": "This class is responsible for applying post-hoc attention score reweighting during inference, per specified attention heads, based on user-highlighted tokens in the input. It implements the method described in Section 3.1 of the paper, which scales attention scores at targeted heads to emphasize highlighted tokens and suppress others, thus steering the model's attention dynamically without any model parameter updates.",
  "inputs": [
    {
      "name": "head_indices",
      "type": "List[Tuple[int, int]]",
      "description": "A list of tuples indicating the selected attention heads for steering, where each tuple is (layer_idx, head_idx). These should be determined from the profiling process, which ranks attention heads based on their utility for the target task(s)."
    },
    {
      "name": "alpha",
      "type": "float",
      "description": "A fixed scalar coefficient (default 0.01), which scales the non-highlighted tokens' attention scores to downweight them. This ensures emphasized tokens gain relatively higher importance after normalization."
    }
  ],
  "methods": [
    {
      "name": "__init__",
      "parameters": [
        "head_indices",
        "alpha (default 0.01)"
      ],
      "logic": "Initializes the AttentionReweighter instance, storing the specified head indices and the scaling coefficient for use during inference."
    },
    {
      "name": "apply_masking",
      "parameters": [
        "attention_scores",
        "highlighted_token_indices"
      ],
      "logic": "Given the attention scores for a particular head (a tensor of shape [batch_size, seq_length, seq_length]) and a list of token indices designated as highlighted by the user in the input, this method scales the attention logits appropriately:"
    },
      "process": [
        "For each attention score matrix in the input list:",
        "Iterate over each sample in the batch:",
        "For each token position i in the sequence:",
        "Identify tokens in the input sequence that are highlighted (indices in 'highlighted_token_indices').",
        "Create a mask to distinguish highlighted vs. non-highlighted tokens.",
        "Apply the scaling: multiply scores targeting non-highlighted tokens (columns j not in highlighted set) by alpha.",
        "For highlighted tokens j, leave scores unchanged (multiplier 1).",
        "Compute the normalization constant C_i for each token i, summing over j:",
        "    C_i = sum_j (A_{ij} * scaling_factor_j) where scaling_factor_j = 1 if j in highlighted, else alpha.",
        "Normalize the scores row-wise (per token i):"
        "   [A_tilde]_{ij} = (A_{ij} * scaling_factor_j) / C_i",
        "Return the set of reweighted attention score matrices for the targeted layers and heads."
      ],
    {
      "name": "integrate_with_transformers",
      "parameters": [
        "model",
        "layer_idx",
        "head_idx",
        "attention_scores",
        "highlighted_token_indices"
      ],
      "logic": "This helper function hooks into the model's attention mechanism at the specified layer and head, retrieves the attention scores during inference:"
    },
      "process": [
        "Register a forward hook in the transformer attention module at the specified layer.",
        "In the hook function, before softmax, call 'apply_masking' with the current attention scores and highlighted token indices.",
        "Replace the attention scores (logits) in-place with the reweighted scores.",
        "After modifying the scores, proceed with the rest of the forward pass to generate outputs."
      ],
    {
      "name": "reset_hooks",
      "parameters": [
        "model"
      ],
      "logic": "To cleanly manage hooks, implement a method to deregister hooks after inference, preventing interference with subsequent calls."
    }
  ],
  "additional_notes": [
    "The class depends on attention_module.py to provide a means to hook into the model's attention computations, e.g., via forward hooks or custom attention modules.",
    "During inference, the process involves: for each tokenized input, determine the highlighted token indices (from user input), then for each targeted head, activate the hook to perform masking and reweighting of attention scores.",
    "The 'apply_masking' method should be designed to work efficiently over batches and sequence lengths, perhaps vectorized with PyTorch operations.",
    "Hyperparameters like the 'alpha' are fixed from configs (e.g., 0.01), but could be made adjustable for hyperparameter tuning.",
    "This class does not modify model weights; it operates solely on attention scores during the inference forward pass."
  ],
  "uncertainties": [
    "Exact format of attention scores received from the transformer model: ensure compatibility with the implementation (tensor shape, device).",
    "Mechanism for attaching hooks: depends on the specific transformer implementation; must confirm support for in-place modification of attention logits during the forward pass.",
    "Order of attention scores access: whether from a standard self-attention implementation or a custom attention module.",
    "Potential need to handle multi-batch inference efficiently, ensuring batch dimensions are preserved.",
    "Details on handling variable sequence lengths and padding: ensure the reweighting is applied only to valid tokens."
  ]
}

## utils.py

{
  "title": "Logic Analysis for utils.py",
  "purpose": "Implement utility functions to support overall pipeline, including configuration parsing, prompt generation with emphasis markers, dataset handling, logging, and any helper methods needed for model inference and profiling.",
  "core functionalities": [
    {
      "name": "Configuration Parsing",
      "description": "Load and validate the YAML configuration file (`config.yaml`).",
      "details": [
        "Use PyYAML to parse `config.yaml`.",
        "Provide functions to access nested config fields with default fallbacks.",
        "Ensure parameters like alpha, batch sizes, model names, dataset paths, etc., are properly loaded and accessible."
      ],
      "considerations": [
        "Implement error handling for missing or malformed entries.",
        "Add validation to check data types (e.g., float for alpha, int for number of heads)."
      ]
    },
    {
      "name": "Prompt Template Rendering",
      "description": "Generate prompts dynamically given input text, highlighted spans, and task-specific templates.",
      "details": [
        "Provide functions such as `create_prompt(template_str, input_text, highlighted_spans, instruction)`.",
        "Handle emphasis markers (e.g., `**` or custom delimiters), replacing placeholders in the templates.",
        "Support multi-span highlights; spans can be continuous or discontinuous, with multiple emphasis parts.",
        "Ensure prompts are well-formatted and compatible with the models' expectations."
      ],
      "considerations": [
        "Make marker customization (e.g., emoji, markdown syntax) configurable via arguments or config.",
        "Escape or sanitize input if necessary to prevent formatting issues."
      ]
    },
    {
      "name": "Dataset Loading & Preprocessing",
      "description": "Provide functions to load datasets for each task, split into train/validation/test sets, and prepare samples for profiling and evaluation.",
      "details": [
        "Function like `load_dataset(task_name, dataset_paths)` to read datasets in expected formats (JSON, TSV, CSV).",
        "Support parsing emphasis markers to determine highlighted token indices.",
        "Tokenize datasets' input texts using the model tokenizer, ensuring alignment between tokens and original spans.",
        "Implement a helper for creating small profiling datasets (`~1000 samples`) and larger test sets (`~5000`)."
      ],
      "considerations": [
        "Ensure dataset files have consistent format, with tokens and labels aligned.",
        "For datasets with emphasis annotations, extract token indices for `𝒢` (highlighted tokens)."
      ]
    },
    {
      "name": "Attention Score Extraction & Modification",
      "description": "Provide functions to hook into model's internal attention scores, extract them, and modify them based on highlighted tokens.",
      "details": [
        "Implement functions such as `register_attention_hooks(model)` that attach hooks to model layers/heads for capturing attention matrices during forward pass.",
        "Create `modify_attention_scores(attention_scores, highlighted_token_indices, alpha)` that applies the scaling and normalization as per the paper.",
        "Support reweighting only heads specified during profiling; store head indices for applying modifications.",
        "Ensure the reweighting is done **before** the softmax to avoid affecting downstream calculations.",
        "Provide an interface to pass in the raw attention matrices and get back reweighted attention matrices, keeping the original data structure intact."
      ],
      "considerations": [
        "Attention scores are typically tensors of shape `(batch_size, num_heads, seq_len, seq_len)`.",
        "Modify attention scores head-wise, layer-wise, preserving model integrity."
      ]
    },
    {
      "name": "Logging & Helper Methods",
      "description": "Implement logging functions to monitor progress, debug information, and evaluation results.",
      "details": [
        "Log steps such as dataset loading, profiling results, hyperparameter settings, inference progress, and evaluation metrics.",
        "Use Python's `logging` module for flexible logging configurations."
      ],
      "considerations": [
        "Include timestamps, task identifiers, and hyperparameter summaries in logs for reproducibility."
      ]
    },
    {
      "name": "Miscellaneous Helpers",
      "description": "Additional functions for input validation, argument parsing, file management, and validation.",
      "details": [
        "Functions like `validate_params()` to ensure inputs are within expected ranges.",
        "Helpers for saving/loading profiling results and attention head selections (e.g., serialize as JSON).",
        "Functions to check if files exist, create directories, and manage temporary files."
      ],
      "considerations": [
        "Implement robust error handling for file I/O."
      ]
    }
  ],
  "general considerations": [
    "Ensure all functions are modular, stateless where possible, and clearly documented.",
    "Design for flexibility: allow marker tokens and templates to be customizable via parameters or config entries.",
    "Maintain compatibility: all functions should operate on data structures compatible with model inference (e.g., tokenized input, attention matrices).",
    "Keep side effects minimal; only functions related to inference, dataset, or file I/O should modify external state."
  ],
  "uncertainties / clarifications needed": [
    "Clarify the precise format of attention matrices from different models (e.g., Hugging Face transformers vs. custom models).",
    "Confirm the best method for intercepting attention scores during inference—whether hooks are sufficient or model modifications are needed.",
    "Determine if handling multiple models with different attention implementations requires additional abstraction."
  ],
  "final note": "All utility functions should serve to streamline dataset handling, prompt generation, attention modification, and logging, thus supporting the core experimental pipeline described in the plan while adhering closely to the paper's methodology."
}

