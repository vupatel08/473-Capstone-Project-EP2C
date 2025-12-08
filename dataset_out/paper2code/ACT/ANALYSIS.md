# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## attention_calibrator.py

# Logic Analysis for `attention_calibrator.py`

## Purpose
`attention_calibrator.py` defines the core class `AttentionCalibrator`, which 
- detects **attention sinks** in the attention maps during inference,
- applies attention weight suppression or redistribution strategies,
- outputs calibrated attention weights for subsequent inference steps.

Its goal is to **dynamically modify attention distributions** on-the-fly, to improve model performance by reducing excessive focus on low-semantic tokens (attention sinks), as inspired by the paper's findings.

---

## Core Responsibilities
1. **Sink Detection (`detect_sinks`)**: identify tokens with abnormally high attention scores.
2. **Attention Suppression (`apply_suppression`)**: modify attention weights to reduce the influence of detected sinks.
3. **Calibration (`calibrate_attention`)**: orchestrate detection and suppression, producing adjusted attention matrices for inference.

---

## Inputs & Outputs
- **Input**:
  - `attention_maps`: A list of attention matrices for each layer, each containing multiple heads.
    - Each attention matrix: shape `(batch_size, num_heads, seq_len, seq_len)`.
  - **Hyperparameters**:
    - `alpha`: float threshold to detect high attention tokens (e.g., 5).
    - `subset_percent`: float, e.g., 0.4 to consider top 40% tokens as sinks.
    - `suppress_factor`: float, e.g., 0.4, amount to reduce sink attention weights.
- **Output**:
  - Modified attention matrices with suppressed or redistributed attention weights.

---

## Step-by-step Functional Breakdown

### 1. Sink Detection (`detect_sinks`)
- **Objective**: For each layer and head, identify tokens that are **attention sinks**.
- **Method**:
  - Compute per-token **attention scores**:
    - For each attention matrix `(batch_size, num_heads, seq_len, seq_len)`:
      - Sum along source tokens for each target token (column sums across source dimension).
      - Average across heads to get a **layer-level sink score per token**.
  - Aggregate across batch (if batch > 1) to get a **dataset-level distribution**—or process per sample during inference.
  - Determine the **attention sink threshold**:
    - For each layer, head, or the entire sequence, select tokens whose attention scores exceed `\(\alpha\)` times the mean or top `subset_percent` percentile.
    - Use `subset_percent` to select top tokens:
      - For each layer-head, get attention scores per token.
      - Sort tokens by attention scores.
      - Select the top `subset_percent` (e.g., top 40%) tokens as potential sinks.
- **Outputs**:
  - A list of sink tokens per layer and head: 
    - Tuple: `(layer_index, head_index, token_index, attention_score)`.

### 2. Attention Suppression (`apply_suppression`)
- **Objective**: For each detected sink token, reduce its attention weights.
- **Method**:
  - For each attention matrix `(batch, heads, seq_len, seq_len)`:
    - Identify sink tokens (from detected list).
    - For each sink token `s`, and each source token `k`:
      - Multiply `A_h^l[k, s]` by `suppress_factor` (`\(\beta\)`).
    - Re-normalize the modified attention map so rows sum to 1:
      - After suppression, attention weights in each source token row are normalized:
        \[
        \hat{A}_h^l[k, :] = \frac{A_h^l[k, :]}{\sum_j A_h^l[k, j]}
        \]
        possibly with the suppressed weights replacing the original ones.
  - **Note**:
    - Ensure numerical stability.
    - For tokens that are not sinks, keep original weights.
  - **Optionally**, redistribute the suppressed attention mass across other tokens to preserve total sum=1.

### 3. Entire Calibration Process (`calibrate_attention`)
- **Workflow**:
  1. Receive raw attention maps for a forward pass.
  2. Invoke `detect_sinks()` to find high attention tokens:
     - Configurable: per layer, per head, based on `alpha` and `subset_percent`.
  3. Call `apply_suppression()` to reduce the influence of sinks.
  4. Output the calibrated attention matrices to be used in the inference step.
  - **Additional Options**:
    - Prioritization of certain layers/heads based on experimental setup.
    - Vary suppression strength (`\(\beta\)`) per layer or head, if needed.
- **Return**: Calibrated attention maps.

---

## Implementation Details & Considerations

### Data Structures:
- Attention maps stored as nested dictionaries/lists or tensors:
  ```
  attention_maps: List[Tensor]
    - each element corresponds to a layer: shape `(batch_size, num_heads, seq_len, seq_len)`
  ```
- Sink info stored as list of tuples:
  ```
  sinks: List[Tuple[int, int, int, float]]
    - layer_idx, head_idx, token_idx, attention_score
  ```

### Hyperparameters:
- These should be configurable:
  - `alpha`: threshold multiplier for high attention detection.
  - `subset_percent`: e.g., 0.4 for top 40%.
  - `suppress_factor (beta)`: e.g., 0.4 to significantly reduce sink attention.
- Sensitivity analysis through ablations should be performed for robustness.

### Efficiency:
- For large models and multiple layers/heads, optimization may be needed:
  - Use batch operations (vectorized).
  - Cache normalization factors.
- Use attention weights **only during inference**, no model weight updates.

### Integration:
- This class will be invoked **before generating output**, possibly inside the inference loop.
- During inference:
  - Extract attention maps via hooks.
  - Apply `calibrate_attention()` per input sample.
  - Pass modified attention weights to the model's forward function.

---

## Summary of Key Methods
| Method | Purpose | Core Logic | Hyperparameters |
|----------|---------|--------------|----------------|
| `detect_sinks(attention_maps)` | Identify attention sinks based on attention scores | Sum/average attention across heads/layers, threshold/percentile-based selection | `alpha`, `subset_percent` |
| `apply_suppression(attention_maps, sinks)` | Reduce sink attention weights | Multiply sink columns row-wise, renormalize rows | `suppress_factor` (`beta`) |
| `calibrate_attention(attention_maps)` | Full process: detection + suppression | Calls above two, returns calibrated maps | hyperparameters |

---

## Conclusion
The `AttentionCalibrator` class will operationalize the paper's core findings: excessive attention on sinks can be detected via **attention scores**, and their **suppression** (or redistribution) based on **hyperparameter thresholds** can **improve LLM inference** performance. The implementation requires careful attention to **attention map extraction**, **threshold-based sink detection**, and **attention normalization** post-calibration, all designed to be **input-adaptive and efficient**.

## dataset_loader.py

### Logic Analysis for `dataset_loader.py`

**Purpose**:  
Implement a `DatasetLoader` class that loads multiple datasets across different task types (classification, multiple-choice, question answering) from specified paths or identifiers, and returns them in a standardized format suitable for inference and evaluation. The loaded data should include input prompts (text inputs formatted with task-specific prompts) and their corresponding labels (if available), enabling straightforward processing in the inference pipeline.

---

### Core Responsibilities

1. **Dataset Specification & Flexibility**  
   - Support multiple dataset formats (classification, multiple-choice, QA).
   - Accept dataset paths or identifiers, along with corresponding prompts/templates.
   - Allow flexible configuration for dataset splits (test, dev, train), depending on task type.

2. **Data Loading & Preprocessing**  
   - Load raw data from files or datasets (e.g., JSON, CSV, HuggingFace datasets).  
   - Generate formatted prompts for each data sample based on template prompts provided.
   - Handle label retrieval (classification labels, correct answers, or reference answers) for evaluation.

3. **Data Structuring & Return Format**  
   - Return a list of dictionaries, each representing one sample with keys:
     - `'prompt'` : formatted input string ready for model input.
     - `'label'` : true label or answer for evaluation (optional).
     - `'metadata'` : any auxiliary info needed (e.g., original data, IDs).

4. **Dataset Compatibility & Storage**  
   - Support datasets from file paths or datasets specified by name in a registry (e.g., HuggingFace datasets).
   - Apply consistent tokenization requirements (enforce maximum input length).
   - Optionally support batching (if needed), but primarily for inference.

5. **Error Handling & Validation**  
   - Validate that datasets are correctly loaded.
   - Check that prompt templates are compatible with data features.
   - Properly handle missing data or labels.

6. **Implementation Details (Per the Design & Task Specifications)**  
   - Use standard Python libraries (`json`, `csv`) for local files.
   - Use HuggingFace datasets library for larger or standardized datasets.
   - Incorporate prompts provided in the original paper for each dataset type.

---

### Step-by-Step Logic

#### 1. Initialization (`__init__`)
- Accept configuration parameters:
  - `dataset_paths`: dict or list, specifying dataset sources (paths, datasets names).
  - `prompts`: dict of task-specific prompt templates.
  - `split`: which subset to load (e.g., 'test', 'dev,' 'train').
  - `max_input_length`: limit for tokenized input length; truncate or pad as needed.
  - `task_type`: optional, to distinguish processing logic.

#### 2. Data Loading (`load_data`)
- For each dataset specified:
  - Determine the source type:
    - **Local file**:
      - Infer format (JSON, CSV) based on extension or explicit type.
      - Load data entries into memory.
    - **HuggingFace dataset**:
      - Use `datasets.load_dataset()` with the dataset name.
      - Select the specified split.
  - For each sample:
    - Extract features (question, context, choices, labels) as needed.
    - Format the prompt:
      - Use provided prompt templates.
      - E.g., for multiple choice:
        `'Complete the following sentence with an appropriate ending. <Question> <choice 1> <choice 2> ... Answer:'`
      - For QA:
        `'Answer question using information in the preceding background paragraph. Title: [title] Background: [background] Q: [question] A:'`
      - Fill in the placeholders with dataset-specific data.
    - Save formatted sample (prompt string).
    - Extract label (if available), e.g., correct answer index or text.
    - Store auxiliary info if needed.
- After loading all samples:
  - If needed, truncate or pad inputs to `max_input_length`.
  - Store the resulting list of dictionaries.

#### 3. Data Output Format
- Return a list of dictionaries:
  ```python
  [
    {
      "prompt": "<formatted prompt>",
      "label": <label, e.g., answer index or text>,
      "metadata": {...}
    },
    ...
  ]
  ```
- This allows downstream code to iterate over samples easily.

---

### Key Considerations

- **Prompt Formatting**:  
  Use the exact prompts as specified in the appendix, replacing placeholders with dataset features.

- **Labels and Evaluation**:  
  Retain labels for evaluation purposes, especially for accuracy or F1 calculations.

- **Dataset Diversity**:  
  Match the expected dataset format:
  - Multiple choice (questions + choices + answer)
  - Classification (sentence + label)
  - QA (question + context + answer)
  
- **Handling Missing Data**:  
  Gracefully skip or report missing features or labels.

- **Performance & Scalability**:  
  For very large datasets, consider streaming or batch loading, but for initial reproduction, simple in-memory loading suffices.

---

### Example Pseudocode (High-Level)

```python
class DatasetLoader:
    def __init__(self, dataset_paths, prompts, split='test', max_input_length=1024):
        self.dataset_paths = dataset_paths
        self.prompts = prompts
        self.split = split
        self.max_input_length = max_input_length

    def load_data(self):
        all_samples = []
        for dataset_name, path in self.dataset_paths.items():
            # Load dataset depending on source
            if path.startswith('hf:'):
                dataset = load_hf_dataset(name=path[3:], split=self.split)
            else:
                dataset = load_local_dataset(path, format=determine_format(path))
            # Process each dataset sample
            for sample in dataset:
                prompt = self.format_prompt(sample, dataset_name)
                label = self.extract_label(sample, dataset_name)
                metadata = {'original_sample': sample}
                all_samples.append({'prompt': prompt, 'label': label, 'metadata': metadata})
        return all_samples
    
    def format_prompt(self, sample, dataset_name):
        if dataset_name in classification_prompts:
            template = self.prompts['classification']
        elif dataset_name in qa_prompts:
            template = self.prompts['qa']
        elif dataset_name in mc_prompts:
            template = self.prompts['multiple_choice']
        else:
            # Default or error
            template = default_template
        # Fill template based on dataset features
        prompt = template.format(...)  # with features from sample
        return prompt

    def extract_label(self, sample, dataset_name):
        # Depending on dataset type
        return derived_label
```

---

### Final Remarks
This logic ensures comprehensive support for multiple datasets tailored for diverse NLP tasks, consistent prompt formatting according to the paper's methodology, and structured data output supporting downstream inference and evaluation modules. The implementation will need to ensure alignment with dataset formats, prompt templates, and evaluation metrics as outlined throughout the paper and plan.

## evaluation.py

### Logic Analysis for `evaluation.py`

The `evaluation.py` module is responsible for orchestrating the evaluation of the models with and without Attention Calibration Technique (ACT) on various datasets, calculating relevant performance metrics, and integrating the attention calibration process during inference. This module's core functions include data handling, model inference, optional attention calibration, and metric computation.

Below is a detailed breakdown of the logic and necessary components for implementing this class:

---

### 1. Initialization (`__init__`)
- **Inputs:**
  - `model`: an instance of `ModelWrapper`, capable of producing model predictions and extracting attention maps, with optional ability to perform attention calibration.
  - `datasets`: a list or dictionary defining datasets for evaluation (classification, QA, multi-turn, etc.). Each dataset includes:
    - 'name'
    - 'split' (e.g., 'test', 'dev')
    - 'metric' (accuracy, EM, F1, score, etc.)
- **Purpose:**
  - Store references to the model and datasets.
  - Load dataset splits (e.g., via DatasetLoader or direct loading routines).
  - Prepare storage for evaluation results, e.g., a dictionary for metrics.

---

### 2. Data Loading
- **Function:**
  - For each dataset in the provided list:
    - Use dataset loader functions (assumed to be outside this class) to load data according to 'name' and 'split'.
    - For classification: list of dicts with keys like `prompt`, `label`.
    - For QA: texts with questions and context.
- **Considerations:**
  - Each data sample may contain:
    - `input_text`: string prompt or question.
    - `label`: correct answer or label.
  - For datasets requiring special formatting, prepare the prompts accordingly (e.g., using prompts from `C.` section).

---

### 3. Inference Loop
- **Core logic:**
  - Loop over each loaded sample in datasets.
  - For each sample:
    - Generate input IDs (tokenized prompt) with the model.
    - If attention calibration is enabled:
      - Activate attention extraction during inference.
      - Extract attention maps (`compute_attention()`).
      - Passage attention maps to the `AttentionCalibrator`:
        - `detect_sinks()` identified high-attention tokens.
        - `apply_suppression()` adjusts attention weights.
        - Potentially, modify the attention weights or the internal state temporarily for that inference run.
    - Generate output (prediction) by passing adjusted attention maps if calibrated attention is applied, or regular inference otherwise.
      - Use `generate_output()` method, which may accept attention modifications if internally supported.
    - Collect predicted output.

- **Special considerations:**
  - Attention modifications should be performed **per sample**, **at inference time**.
  - For efficiency, ensure the process of extracting and adjusting attention is optimized:
    - Hook mechanisms should allow passing modified attention weights into the forward pass.
    - Maintain consistency when modifying attention weights and subsequent normalization.

---

### 4. Prediction and Output Handling
- **Classification Tasks:**
  - Extract model output logits or predicted label/token.
  - Map output tokens to class labels.
  - For example, for multiple-choice:
    - Cast predicted tokens to the selected class label.
- **QA / Open-ended Tasks:**
  - Extract generated text.
  - Use string matching or metric functions for exact match (EM), F1 (for SQuAD), or other measures.
  - For multi-turn, multi-sample evaluation, aggregate results.

---

### 5. Metric Computation
- Post-process predictions and compare with ground truth:
  - For classification:
    - Count correct predictions.
    - Compute accuracy = correct_predictions / total_samples.
  - For QA (SQuAD):
    - Compute EM and F1 scores.
  - For multiple datasets:
    - Store individual dataset metrics, then aggregate (average accuracy, EM, F1, etc.).
- **Implementation:**
  - Maintain counters: total samples, number of correct predictions, sum of EM/F1 scores.
  - Use standard metrics from `scipy` or `sklearn.metrics`, or custom implementation if needed.

---

### 6. Performing Multiple Evaluation Rounds
- To compare the effect of ACT:
  - Run inference **twice or more**:
    - Without attention calibration (vanilla)
    - With calibration (ACT)
  - Store results separately for analysis.

### 7. Data Structuring and Results
- Prepare evaluation report:
  - For each dataset:
    - `name`, `split`, `accuracy/em/f1`, and **delta** compared to vanilla.
  - Possibly a summary dictionary:
    ```python
    results = {
        'dataset_name': {
            'metric': value,
            'delta': improvement over baseline,
            'details': {...}
        },
        ...
    }
    ```
- Summarize overall performance across datasets:
  - Mean accuracy or F1 improvement.
  - Plots or visualizations if needed.

---

### 8. Error Handling & Edge Cases
- Handle datasets with different sample lengths and formats.
- Ensure attention calibration applies **only during inference**.
- Correctly handle samples where attention sinks are not present or weak.
- Confirm model outputs are consistent given attention modifications (e.g., no shape mismatches).

---

### 9. Hyperparameters & Configurations
- Use values from `config.yaml`, e.g.:
  - Threshold \(\alpha\) for sink detection.
  - Suppression factor \(\beta\).
  - Subset percent of tokens considered as sinks.
- Allow passing these as arguments or class parameters.

---

### 10. Summary of Key Steps
- Load datasets for evaluation.
- For each sample:
  - Tokenize input.
  - Extract attention maps.
  - If ACT enabled:
    - Detect high-attention tokens.
    - Suppress/adjust attention weights.
  - Generate model output.
  - Store predictions.
- After inference:
  - Compute metrics.
  - Compare with vanilla baseline.
  - Summarize performance improvements.

---

**This comprehensive logic analysis ensures a clear, methodical implementation plan for `evaluation.py` that adheres to the paper's experimental design and the provided configuration, enabling accurate reproduction of the experiment results and detailed analysis of ACT's impact.**

## main.py

# Logic Analysis for main.py

**Objective:**  
Implement a main script that orchestrates dataset loading, model initialization, attention map extraction and modification, inference with attention calibration (ACT), and evaluation, following the described experimental methodology.

---

# 1. Initialization & Configuration Load

- **Load configuration:**  
  - Parse 'config.yaml' to extract all hyperparameters, model info, attention calibration settings, evaluation datasets, and other parameters.
- **Set device:**  
  - Use `torch.device()` based on 'device' parameter from the config.
- **Prepare logging and tracking:**  
  - Set up logging, metrics storage, and visualization directories if needed.

---

# 2. Dataset Loading

- **Instantiate DatasetLoader:**  
  - Call `DatasetLoader` class with dataset paths and prompts as per config.
- **Load datasets:**  
  - Invoke `load_data()` method; get a list of data entries (dicts) with prompt input text and labels.
- **Preprocessing:**  
  - For classification, QA, or other tasks: tokenization, considering max input length (e.g., 1024).
  - For zero-shot/few-shot prompts, assemble the input string accordingly.

---

# 3. Model Initialization

- **Instantiate ModelWrapper:**  
  - Provide model name/identifier (e.g., "Llama-7b") and device.
- **Load pre-trained weights:**  
  - Use Hugging Face transformers AutoModelForCausalLM or similar.
- **Register attention hooks:**  
  - Call `register_attention_hooks()` to enable extraction of attention matrices during inference.
- **Verification:**  
  - Confirm hook registration (inside ModelWrapper).

---

# 4. Attention Map Extraction & Calibration Preparation

- **Define AttentionCalibrator object:**  
  - Pass hyperparameters: `alpha`, `suppress_factor`, `subset_percent`, and optionally specific layers/heads to calibrate.
- **Identify attention sink thresholds:**  
  - Use attention scores to detect tokens with attention > threshold (\(\alpha\)).  
  - For each input, the actuator will detect sinks based on attention maps.

---

# 5. Inference Loop with Attention Calibration

- **For each data sample:**  
  - **Prepare input prompt:**  
    - Format prompt with the input text, according to the dataset type (e.g., classification, multiple-choice, QA).
  - **Tokenize input:**  
    - Use tokenizer, truncate/pad as necessary.
  - **Extract attention maps:**  
    - Run `model_wrapper` forward pass with hooks enabled.  
    - During inference, capture attention matrices (\(\mathbf{A}_h^l\)) for all layers and heads.
  - **Identify attention sinks:**  
    - Pass attention maps to `AttentionCalibrator.detect_sinks()`.  
    - Use threshold \(\alpha\) and subset percentage from config.
  - **Apply attention calibration:**  
    - Call `calibrate_attention()` with detected sinks.  
    - For each sink token, reduce attention weights by factor \(\beta\); re-normalize to sum to 1.
  - **Modified inference:**  
    - Run the model's generate method with the **adjusted attention weights**:
      - *Important:* This may involve temporary overriding the attention matrices before the softmax, which could entail modifying the forward pass or passing modified attention matrices if the model supports it.
    - **Note:** If direct modification isn't feasible, implement a wrapper or a method to "simulate" attention suppression during inference via hooks or custom attention function.
  - **Obtain output text:**  
    - Decode the generated tokens into output string.
  - **Store predictions:**  
    - Save output predictions and attention maps for analysis.

---

# 6. Evaluation & Metrics Calculation

- **Compare predictions with ground truth labels:**
  - For classification and multiple-choice tasks, compute accuracy.
  - For QA, compute EM and F1 scores.
  - For open-ended tasks, compute appropriate metrics (e.g., exact match, scores).
- **Record results:**
  - Store performance metrics with and without attention calibration.
  - Track improvements (\( \Delta \) accuracy, F1, etc.).

---

# 7. Aggregation & Reporting

- **Aggregate results:**
  - Compute mean accuracy/F1 scores, improvements over baseline.
  - Possibly plot histograms of attention sinks and attention maps before/after ACT.
- **Optional visualization:**
  - Save attention heatmaps for selected examples.
  - Generate diagnostics plots to verify sink detection and suppression effects.
  
---

# 8. Final steps

- **Summarize overall performance:**
  - Print/Log overall improvements.
  - Save detailed logs and results in structured format (JSON or CSV).
- **Clean up:**
  - Remove hooks from model.
  - Free resources, close files.

---

# 9. Key Implementation Details & Considerations

- **Attention Hook Registration:**
  - Use `register_forward_hook` or equivalent in HuggingFace/transformers to get attention maps during inference.
- **Handling attention matrices:**
  - Ensure the attention matrices are accessible for modification.
  - Appropriately normalize after modification to maintain valid probability distributions.
- **Hyperparameters:**
  - Use set values from 'config.yaml':
    - `alpha=5`
    - `suppression_factor=0.4`
    - `subset_percent=0.4`
- **Batching:**
  - Depending on hardware, process samples individually or in batches.
  - Make sure attention modifications are per-sample.

---

# 10. Summary of Key Logical Steps

```
Load config → Load dataset → Initialize model → Register attention hooks
For each sample:
    Tokenize input prompt
    Run inference → capture attention maps
    Detect attention sinks using thresholds
    Apply ACT → reduce attention in sinks, normalize
    Generate output with adjusted attention
    Collect predictions and attention maps
Evaluate predictions against ground truth
Report metrics and improvements
```

---

This comprehensive logic analysis provides all necessary detailed steps and considerations to implement 'main.py' effectively, ensuring alignment with the methodologies and experiments described in the paper.

## model_wrapper.py

**Logic Analysis for `model_wrapper.py`**

---

### **Objective**

Implement the `ModelWrapper` class to facilitate:
- Loading and managing a pretrained LLM from HuggingFace Transformers.
- Registering hooks to extract attention weights during inference.
- Providing a method to retrieve attention matrices (`compute_attention()`).
- Generating model outputs with the possibility of applying attention modifications (for ACT).
- Ensuring modularity for seamless integration with calibration and evaluation modules.

---

### **Core Functional Requirements**

1. **Model Initialization**
   - Load specified model and tokenizer from HuggingFace using the provided `model_name` and `model_path`.
   - Set device (CPU or CUDA) for inference.
   - Ensure model is in evaluation mode (`model.eval()`).
   - Optionally, configure the model for fast inference (e.g., disable gradients).

2. **Attention Map Extraction**
   - Register hooks on the model’s transformer layers that capture `attention weights` during the forward pass.
   - Store these attention weights in a structure accessible for analysis and modification.
   - The hooks should be able to handle multiple layers and heads, as specified by configuration.

3. **Data Structures for Attention Storage**
   - Maintain a list (or dictionary) of attention maps per layer, per head.
   - Each attention map should be a tensor with shape `[batch_size, num_heads, seq_len, seq_len]` if batching is supported.
   - For inference, support batch processing if needed, but primarily focus on per-sample inference.

4. **Inference with Optional Attention Modification**
   - Implement `generate_output()`:
     - Accepts tokenized input IDs.
     - Optionally, an external attention map (modified after calibration) can be used.
     - If attention is modified:
       - Inject the calibrated attention weights into the model before softmax.
       - This may require overriding the attention computation in the forward pass or directly manipulating the stored attention matrices prior to softmax computation.
   - Output:
     - Generate the textual output (prompt + response).
     - Return the output string for downstream processing.

5. **Accessibility and Modifiability of Attention Maps**
   - Provide `compute_attention()`:
     - Return the stored attention matrices from the latest inference.
     - Useful for analysis by `AttentionCalibrator`.
   - Include a method to reset/hook removal if necessary.

6. **Hook Management**
   - Register hooks at model load.
   - Ensure hooks are properly removed or disconnected at the end of evaluation or when reloading.
   - Hooks should minimally interfere with inference performance.

7. **Handling Model Architecture Details**
   - Transformers from HuggingFace (e.g., GPT, Llama) have a consistent internal structure but may vary.
   - Hook into `SelfAttention` modules within the transformer layers.
   - Access attention weights via defined hook functions that capture `attn_outputs` (usually stored in model outputs or internal buffers).

8. **Device Compatibility and Performance**
   - Move model and tensors to the specified device.
   - During attention extraction and modifications, maintain tensor device consistency.
   - Use torch.no_grad() during inference to improve speed.

---

### **Step-by-Step Logical Design**

#### **A. Model Loading**
- Load model:
  ```python
  self.model = AutoModelForCausalLM.from_pretrained(model_path)
  self.tokenizer = AutoTokenizer.from_pretrained(model_path)
  ```
- Set device:
  ```python
  self.device = torch.device(device)
  self.model.to(self.device)
  self.model.eval()
  ```
- Support configurations:
  - Enable/disable torch gradients (`torch.no_grad()` context during inference).
  - Support batching if needed (primarily focus on single inference samples for detailed attention extraction).

#### **B. Registering Attention Hooks**
- Identify all `SelfAttention` modules or equivalent in the model.
- Register a hook function:
  ```python
  def attention_hook(module, input, output):
      # output: attention probabilities, shape [batch_size, num_heads, seq_len, seq_len]
      # Store in self.attention_maps per layer
  ```
- Attach hook to each relevant layer:
  ```python
  handle = module.register_forward_hook(attention_hook)
  ```
- Store hook handles for cleanup.

#### **C. Extracting Attention Maps**
- During inference:
  - Forward pass your input IDs.
  - Attention maps are captured by hooks.
  - Store attention maps for analysis/modification.
- The stored attention maps should be accessible via:
  ```python
  self.attention_maps
  ```
  structured as:
  ```python
  {
    layer_index: {
      head_index: attention_tensor
    }
  }
  ```

#### **D. Generating Outputs**
- Use `model.generate()` with `input_ids`.
- To incorporate attention modifications:
  - Either:
    - Override the attention computation via hooks or model internals.
    - Or, after detection/modification, replace the attention weights directly before the softmax computation.
- Ensure that the modified attention weights are used in the subsequent layer computations, maintaining consistency.

#### **E. Attention Map Access**
- Implement `compute_attention()`:
  - Return the latest stored attention maps for analysis.
  - Possibly smoothing or processing attention maps for visualization or calibration.

#### **F. Hook Management**
- Provide methods:
  - `register_attention_hooks()`:
    - Register all hooks during model initialization.
  - `remove_attention_hooks()`:
    - Remove hooks after inference or calibration is complete.

---

### **Handling Hyperparameters and Configurations**
- Use the provided `config.yaml` parameters:
  - Layer indices for calibration (`calibrate_layers`).
  - Head indices (`calibrate_heads`).
  - Attention sink detection threshold (`alpha`).
- These parameters influence how attention maps are captured and modified:
  - Only modify attention in specified layers/heads.
  - Detect sinks based on thresholding attention scores.

---

### **Error Handling & Edge Cases**
- No attention maps captured if hooks not properly registered.
- Attention maps may be empty if model architecture varies.
- Input size exceeds maximum sequence length: truncate or handle appropriately.
- GPU memory constraints: avoid memory leaks, free hooks after use.

---

### **Summary**

The `ModelWrapper` class serves as the interface between the model, attention analysis, and inference. The essential capabilities include:
- Loading and configuring the model.
- Registering hooks to capture attention matrices dynamically.
- Extracting and storing these matrices in a structured format.
- Generating outputs with optional input attention modifications, enabling the ACT to calibrate attention during inference.
- Putting in place mechanisms for cleanup, flexibility, and hyperparameter-driven customization.

This logical framework ensures faithful implementation aligned with the paper’s methodology while supporting effective attention sink detection and calibration crucial for the experimental validation.

