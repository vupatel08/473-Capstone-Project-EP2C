# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

**Logic Analysis for dataset_loader.py**

---

### Purpose:
Implement a `DatasetLoader` class that:
- Loads the mixed-quality conversation dataset from a specified file.
- Parses the dataset entries into structured data suitable for training.
- Assigns class labels ('expert' or 'sub-optimal') based on source or content.
- Calculates or assigns coarse reward weights (`r_c`) according to class labels, with `r_c=1.0` for expert data and `r_c=α` (e.g., 0.8) for sub-optimal data.
- Prepares and returns datasets (training and evaluation splits) for subsequent training and evaluation workflows.

---

### Inputs:
- The dataset path from `config.yaml` (`dataset_path`).
- Sample size parameters (`train_sample_size` and `eval_sample_size`).
- The data format, likely JSON (per dataset_path filename). For safety, assume dataset entries are JSON objects with fields such as:
  - `prompt`: input instruction prompt.
  - `response`: model-generated or reference response.
  - `source`: optional, indicates dataset source (e.g., 'GPT-4', 'GPT-3.5').

### Dataset Characteristics:
- The dataset contains conversations labeled by source quality:
  - GPT-4 or GPT-4-like data: high quality, class label `'expert'`.
  - GPT-3.5 or other lower-quality data: class label `'sub-optimal'`.

### Step-by-step Implementation Details:

#### 1. Data Loading:
- Use `json` or `datasets` library (`datasets.load_dataset`) if the dataset is in JSON format.
- Read the dataset file (`sharegpt_mixed_quality.json`).
- Load all entries into a list or a `Dataset` object (from the `datasets` library), each with:
  - `prompt`: string containing user instruction.
  - `response`: string containing model reply.
  - Optional fields: source, quality indicator.

#### 2. Data Parsing:
- For each conversation:
  - Create a structured data tuple:
    - `input_prompt`: formatted string concatenating prompt and class-conditioning prefix as needed (for training) — e.g., "[<|class|>] GPT-4 User: ..." or "User: ..." + conversation history.
    - `response`: the target response string.
    - `class_label`: `'expert'` or `'sub_optimal'`.
    - `reward_weight`: float, assigned as:
      - `1.0` if class='expert' (GPT-4 source).
      - `α` (default 0.8) if class='sub_optimal' (GPT-3.5 source).
- If the dataset contains a `source` field, map source strings accordingly.
- For datasets lacking source info, possibly infer from other indicators or assign defaults.

#### 3. Sampling:
- Randomly sample up to `train_sample_size` conversations for training, and `eval_sample_size` for evaluation.
- Use `random.seed` for reproducibility (seed=42 as per config).

#### 4. Dataset Splitting:
- Divide loaded data into training and evaluation datasets according to sampling sizes.
- Maintain separate lists or datasets for train/eval.

#### 5. Data Structure:
- Define each entry as a dictionary (or tuple) with keys:
  - `'prompt'`: string
  - `'response'`: string
  - `'class_label'`: string (`'expert'` or `'sub_optimal'`)
  - `'reward'`: float (`1.0` or `α`)
- For compatibility, create a class or namedtuple for encapsulating data entries.

#### 6. Handling Special Tokens / Prompt Conditioning:
- Incorporate the `conditioning_token` (from config) or class label indicator into the prompt.
- For example, prepend:
  - `"<|class|> GPT4"` for expert data.
  - `"<|class|> GPT3"` for sub-optimal data.
- The exact template should match the training prompt template specified in the paper.

#### 7. Output:
- Provide methods:
  - `load_data()`: loads and returns a list of all data entries (dictionaries).
  - `get_train_dataset()`: returns a dataset (list or Dataset object) for training.
  - `get_eval_dataset()`: returns a dataset for evaluation.
- Optional: Implement batching or collating functions for DataLoader based on framework used downstream.

---

### Additional Considerations:
- **Data Validation:**
  - Ensure each data point is complete with prompt and response.
  - Fail gracefully or warn if missing fields.
- **Class Label Consistency:**
  - Maintain consistent class labels to match reward assignment logic.
- **Reproducibility:**
  - Use a fixed seed for sampling.
- **Performance:**
  - For large datasets, consider memory efficiency, e.g., reading line-by-line or streaming.
- **Extensibility:**
  - Allow for customizable label-to-reward mappings, e.g., extend to more classes if needed.

---

### Summary:
The `DatasetLoader` class will:
- Read and parse JSON data entries.
- Assign class labels based on source information.
- Calculate reward weights (`r_c`) based on class labels.
- Pre-process prompts with class-conditioning prefixes.
- Sample fixed-size datasets for training and evaluation.
- Expose methods to access prepared datasets for `trainer.py`.

This logic ensures dataset consistency, proper reward assignment, and alignment with the described experimental setup in the paper, facilitating effective training via the reward-weighted supervised objective.

---

## evaluation.py

# Evaluation.py: Logic Analysis

This module implements the `Evaluation` class, which facilitates model inference, response scoring, and metric computation during evaluation on benchmark datasets. It depends on the `model.py` module for handling the language model and `datasets` library (or equivalent preprocessing tools) for managing evaluation datasets. The core purposes are to generate model responses for given prompts, score these responses against benchmark standards, and compute overall metrics such as win rate and average score.

---

## Core Components & Responsibilities

### 1. Initialization (`__init__`)
- **Inputs:**
  - Trained `model` object: an instance of the `Model` class from `model.py`.
  - `eval_dataset`: structured dataset containing evaluation prompts, responses, class labels, and possibly reference responses or scoring info.
- **Setup:**
  - Store references to the model and dataset.
  - Prepare a scoring evaluator or plugin: e.g., initialize GPT-4 or GPT-3.5 API for automatic scoring if scoring involves external LLMs, or load an internal scoring function.
  - Configure parameters such as number of samples per prompt, whether to perform pairwise comparisons or single-response scoring.

---

### 2. Response Generation (`generate_response`)
- **Inputs:**
  - A `prompt` string, formatted according to the benchmark or evaluation setting.
  - Optional parameters: `max_new_tokens`, `temperature`, `top_k`, etc.
- **Workflow:**
  - Call the `model.generate()` method, passing the prompt and generation parameters.
  - Return the generated text as the model's response.
- **Notes:**
  - During evaluation, responses must adhere to the benchmark format.
  - Use consistent prompts matching training format, e.g., including class-conditioned prompts.

---

### 3. Response Scoring (`score_response`)
- **Inputs:**
  - `prompt`: the input prompt given to model.
  - `response`: the model response to evaluate.
- **Scoring Approaches:**
  - **Automated Scoring (preferred):**
    - Use GPT-4 or GPT-3.5 API with a prompt template (as in Table 6) to evaluate response quality objectively.
    - The prompt includes instructions to act as an impartial judge, considering helpfulness, relevance, accuracy, etc.
    - Parse the returned score (e.g., a number 1-10).
  - **Alternative:** 
    - Use string matching, reference-based scores, or pre-trained quality classifiers (less preferred).
- **Output:**
  - Numeric score (e.g., float or int from 1 to 10).

---

### 4. Benchmark Evaluation (`evaluate`)
- **Inputs:**
  - List of benchmark names (e.g., ["AlpacaEval", "MT-bench", "Vicuna-bench", "AGIEval"]).
  - Number of samples per prompt (e.g., `num_eval_samples=128`).
  - Evaluation mode: single response scoring or pairwise comparison.
- **Workflow:**
  - For each dataset:
    - Load the prompts/tasks.
    - For each prompt:
      - Generate `num_eval_samples` responses by calling `generate_response()`.
      - For scoring:
        - Use `score_response()` for each response.
        - If pairwise, compare responses from two models and determine which is better according to the evaluator.
    - Record scores for each prompt.
  - Aggregate results:
    - Compute win rate: proportion of responses where model response outperforms baseline or others, using pairwise comparison.
    - Compute average score per prompt.
    - Collect overall metrics per benchmark.

---

### 5. Metrics Calculation
- **Win Rate:**
  - For pairwise comparisons, define:
    
    \[
    \text{Win}_{A,B} = \frac{\text{Number of times } response_A \text{ outperforms response_B}}{\text{Total comparisons}}
    \]
  - For single-response metrics:
    - Use the scoring function to assign a score to each model's response.
    - Derive a win/lose/tie based on the comparison of scores.
- **Average Score:**
  - Sum all response scores and divide by the number of responses evaluated.
- **Correlation or consistency checks:**
  - Optional: compute Pearson/Spearman correlation between different automatic evaluators if multiple scoring models are used.

---

### 6. Handling Benchmark-specific Details
- Implement separate prompt templates or formatting for each benchmark, ensuring responses adhere to their expected styles.
- For AGIEval or multiple-choice tasks:
  - Responses may require parsing the model's answer string to extract selected options.
  - Compute correctness based on answer matching against the known answer.
- For pairwise evaluation:
  - Sample responses for two models on the same prompt.
  - Use the scoring function to compare pair responses, and record the winner.

---

## Practical Implementation Considerations

- **Response Generation:**
  - Use batching where possible for multiple prompts to improve efficiency.
  - Control randomness with fixed seed and temperature.
- **Scoring:**
  - To reduce API costs, consider caching responses or batching scoring prompts.
  - Ensure robust prompt templates (see Table 6) for consistent evaluation.
- **Evaluation Data:**
  - Dataset loading involves reading JSON/CSV files with prompt-response pairs, possibly including class labels.
  - Sensitive to formatting; ensure prompts and responses match the format used in training (e.g., class-conditioned prompts).
- **Logging & Result Storage:**
  - Store per-prompt responses, scores, and overall metrics.
  - Save evaluation metrics and comparison results for reproducibility.

---

## Summary
The `Evaluation` class should provide:
- Initialization to load a trained model and benchmarks.
- A generation method for obtaining responses on prompts.
- A scoring method to rate responses, typically via external LLMs like GPT-4.
- An evaluation method to systematically assess performance across multiple tasks, compute win rates, average scores, and generate comprehensive reports.
- Modular design allowing easy extension to new benchmarks or evaluation metrics.

All these components should operate coherently, leveraging existing modules (`model.py`, datasets) and ensuring consistent prompt formatting aligned with training procedures. The class should facilitate robust, objective, and reproducible assessment of the fine-tuned model's capabilities.

## main.py

{
  "main.py": {
    "Purpose": "Coordinate the entire training and evaluation pipeline for the OpenChat C-RLFT fine-tuning process, ensuring datasets are loaded, the model is initialized with class-conditioned prompts, training is performed according to specified hyperparameters, outputs are saved, and model evaluation on performance benchmarks is conducted systematically.",
    "Step-by-step Breakdown": [
      {
        "1. Import necessary modules": "Import classes and functions from dataset_loader.py (DatasetLoader), model.py (Model), trainer.py (Trainer), evaluation.py (Evaluation), and any utility libraries such as torch, os, and logging."
      },
      {
        "2. Set configurations and hyperparameters": "Read the configuration parameters from config.yaml using a YAML parser (e.g., PyYAML). Assign values for dataset paths, training hyperparameters (learning rate, batch size, epochs, beta, alpha, etc.), and model saving paths.",
        "Note": "Ensure the configuration aligns with the provided spec, including 'model', 'training', 'evaluation', 'dataset', and 'output_dir'."
      },
      {
        "3. Set random seed for reproducibility": "Initialize the random seed (using seed=42) across libraries such as torch, numpy, and Python's random for experiment reproducibility."
      },
      {
        "4. Initialize DatasetLoader": "Create an instance of DatasetLoader, passing dataset_path, train_sample_size, eval_sample_size, and any dataset-specific configurations. Call load_data() to load and process the dataset:",
        "Details": "The dataset loader should parse the data file (e.g., JSON). Each sample is a tuple containing: prompt text with class-conditioning prefix, response text, class label (e.g., 'GPT-4' or 'GPT-3.5'), and reward weight (based on alpha).",
        "Outcome": "Obtain structured datasets for training and evaluation."
      },
      {
        "5. Initialize Model": "Create a Model instance with the pretrained_model_name and conditioning_token. This model should be capable of conditioning prompts with class tokens or templates as specified. Call any initialization routines required (e.g., load weights, prepare tokenization)."
      },
      {
        "6. Initialize Trainer": "Create a Trainer instance with the model, training dataset, and hyperparameters:",
        "Arguments": {
          "model": "the Model instance",
          "dataset": "training dataset with samples (prompt, response, class_label, reward_weight)",
          "beta": "from config, e.g., 0.2",
          "learning_rate": "from config, e.g., 3e-5",
          "batch_size": "from config, e.g., 200",
          "epochs": "from config, e.g., 3",
          "max_grad_norm": "from config",
          "warmup_steps": "from config",
          "weight_decay": "from config"
        },
        "Operation": "The Trainer should be responsible for setting up the optimizer (AdamW), learning rate scheduler, and loss function. It should internally handle the class-conditioned prompt conditioning during training, applying the weighted supervised learning based on reward weights."
      },
      {
        "7. Run Training": "Call trainer.train() method:",
        "Details": "During training, the process iterates over epochs and batches, for each batch:",
        "Workflow": [
          "Sample a batch of data samples: (prompt, response, class_label, reward_weight).",
          "Construct input models inputs with class conditioning prompts, e.g., prepend conditioning tokens or specialized prompt strings according to class.",
          "Forward pass through the model to compute predicted responses.",
          "Calculate cross-entropy loss weighted by the reward-based weights (exponential of reward divided by beta).",
          "Backpropagate, clip gradients if necessary (max_grad_norm), and update parameters."
        ],
        "Result": "The model parameters are optimized with reward-weighted supervised regression, effectively implementing the C-RLFT method."
      },
      {
        "8. Save trained model": "Post-training, invoke trainer.save_model(output_dir) to save the fine-tuned model weights, tokenizer, and configuration for inference and future reproducibility."
      },
      {
        "9. Initialize Evaluation": "Create an instance of Evaluation with the trained model and evaluation dataset (either the same dataset for quick validation or distinct benchmark datasets), and specify metrics (win_rate, score).",
        "Operation": "evaluation.py handles the generation of responses using the trained model, the application of benchmark-specific scoring functions (e.g., comparing to reference answers, scoring via GPT-4 evaluator), and metric computation."
      },
      {
        "10. Run Evaluation on benchmarks": "For each benchmark listed in 'eval_benchmarks', invoke evaluation.evaluate(metrics) with parameters such as 'win_rate' and 'score'.",
        "Details": "Loop over benchmarks; generate model outputs for benchmark questions, compare responses with reference or baseline responses, and log metrics. The evaluation should follow the methodology in the paper—pairwise comparisons and automated scoring via GPT models."
      },
      {
        "11. Log and output results": "Print or log the evaluation metrics per benchmark, overall average, and specific observations about instruction-following performance, generalization, and robustness."
      },
      {
        "12. Optional – Visualization and debugging": "In case needed, extract conversation embeddings for visualization (UMAP/t-SNE), or check the distribution of responses, monitor training loss curves, or generate sample conversations for qualitative assessment."
      }
    ],
    "Additional Considerations": [
      "Ensure consistent tokenization, particularly when incorporating class-conditioned prompts/templates.",
      "Handle multi-turn conversation formatting appropriately during data loading, model input construction, and inference.",
      "Make sure hyperparameters like beta, alpha, learning_rate, batch_size are consistent across modules.",
      "Provide clear logging for each stage to support reproducibility and debugging.",
      "Confirm that all paths (dataset_path, output_dir) are correctly set relative to the project structure."
    ],
    "Uncertainties": [
      "Clarify the exact data file format (expected JSON schema).",
      "Determine specific prompt templates for class conditioning (e.g., prefix tokens, special markers).",
      "Decide on the length of context window and truncation strategy for long conversations at inference."
    ]
  }
}

## model.py

### Logic Analysis for `model.py`

**Objective:**  
Implement a `Model` class that encapsulates loading a pre-trained LLaMA-based language model, supports class-conditioning via prompt or token prompts, and provides methods for forward inference and response generation, consistent with the methodology in the paper.

---

### 1. Responsibilities of `Model` class:
- Load the pre-trained backbone language model and tokenizer from Hugging Face.
- Support prompt conditioning to facilitate class-conditioning, likely via special tokens or prompt templates.
- Provide a method to execute forward passes for training (calculating token-level logits and loss).
- Provide a generation method to produce responses given prompts, conditioning on class labels.

---

### 2. Key Components & Methods:
#### a. Initialization (`__init__`)
- Load the pre-trained model and tokenizer:
  - Use `transformers.AutoModelForCausalLM` and `AutoTokenizer`.
  - Specify the `pretrained_model_name` from config (`"huggingface/llama-13b"`).
  - Load model in half-precision (FP16) if supported, for memory efficiency.
- Initialize class conditioning:
  - Store `conditioning_token` (e.g., `<|class|>`).
  - Decide how to incorporate class info into prompts:
    - As a prefix token (e.g., `<|class|> expert`).
    - As a prompt prefix template with class label embedded.
- Prepare any special tokens (adding to tokenizer if necessary):
  - Check if `conditioning_token` exists in tokenizer; add if not.
  - Save token IDs for easy insertion during prompt creation.
- Set device (e.g., CUDA).

#### b. Forward Pass (`forward`)
- Inputs:
  - `input_ids`: tensor of token IDs representing the input prompt + conversation history.
  - `attention_mask`: corresponding mask.
- Output:
  - Logits for each position (from the model output).
  - Needed for loss calculation during training.
- Logic:
  - Run model with `return_dict=True`.
  - Extract `logits`.
  - Return logits for loss or further processing.

#### c. Response Generation (`generate`)
- Inputs:
  - `prompt`: string (including class conditioning prompt).
  - optional: `max_new_tokens`, `temperature`.
- Process:
  - Prepare prompt by inserting conditioning prefix as per class.
  - Encode prompt with tokenizer.
  - Call `model.generate()` with parameters:
    - `max_new_tokens`
    - `temperature`
    - Possibly `do_sample=True` for diversity.
  - Decode generated tokens into string.
- Output:
  - Generated response string.
  
--- 

### 3. Prompt Conditioning:
- To implement prompt conditioning as per paper:
  - Append the `conditioning_token` and associated class label (e.g., "Expert", "Suboptimal") at the start of each input prompt.
  - Use a template like:
    - `"<|class|> expert\nUser: {query}\nAssistant:"`
    - `"<|class|> suboptimal\nUser: {query}\nAssistant:"`
  - Alternatively, prepend special tokens:
    - Add `<|class|>` as a special token, then include class label token during prompt creation.
- For multi-turn conversations:
  - For both training and generation:
    - Include class-conditioning tokens at the start of each turn.
    - Maintain conversation history with turns:
      - e.g.,
        ```
        <|class|> expert
        User: {instruction}
        Assistant: {response}
        ```
  - During training, ensure the prompt sequence properly concatenates tokens and turns.

---

### 4. Tokenizer Handling
- Ensure `conditioning_token` is in the tokenizer vocabulary:
  - Use `tokenizer.add_special_tokens()` if needed.
- The class label can be appended as a token or string that is tokenized and inserted.
- Maintain `conditioning_token_id` and the class label token IDs for prompt construction.

---

### 5. Model Save/Load
- Include a method for saving model weights if needed.
- During inference, load the model from saved weights.

---

### 6. Additional Details
- Use `torch.device()` for device management.
- Consider gradient checkpointing if memory is limited.
- Make sure to set model to evaluation mode (`model.eval()`) during inference.

---

### 7. Hyperparameters & Configs
- Pretrained model name and conditioning token are passed during initialization.
- Encoding prompts with class info should be consistent with training prompts used.

---

### **Summary of Implementation Plan for `model.py`:**

- **`__init__(self, pretrained_model_name, conditioning_token)`**:
  - Load pre-trained model and tokenizer.
  - Check/add `conditioning_token` in tokenizer.
  - Save token IDs for conditioning.
  - Set to device.
  
- **`set_conditioning(self, class_label)`**:
  - Store class label info.
  - Prepare prompt prefix with `<|class|>` + class label string.

- **`prepare_prompt(self, prompt:str, class_label:str)`**:
  - Concatenate conditioning prefix with user prompt.
  - Encode into input IDs.

- **`forward(self, input_ids, attention_mask)`**:
  - Run `model(input_ids, attention_mask)`.
  - Return logits for loss.

- **`generate(self, prompt:str, class_label:str=None, max_new_tokens=256, temperature=0.7)`**:
  - Prepend class-conditioned prefix to prompt if needed.
  - Encode prompt + class info.
  - Generate response.
  - Decode output tokens.
  - Return response string.

This logic ensures that the `model.py` supports the class-conditioning paradigm, inference, and training needs detailed in the paper, aligned with the experimental design and methodology.

## trainer.py

**Logic Analysis for `trainer.py`**

---

### Overview

The `trainer.py` module implements the `Trainer` class responsible for training the class-conditioned language model using a reward-weighted supervised learning approach, as described in the paper's C-RLFT method. The core functionality includes:

- Initialization with model, dataset, and hyperparameters.
- Setting up the optimizer and learning rate scheduler.
- Preparing inputs with class-conditioning prompts.
- Computing weighted loss based on coarse rewards.
- Performing training steps with gradient clipping and regularization.
- Saving the trained model.

### Key Components and Responsibilities

1. **Initialization (`__init__`)**
   - Receive `model`, `dataset`, and training hyperparameters.
   - Instantiate optimizer (`AdamW`) with specified learning rate, weight decay, and gradient clipping norm.
   - Set up learning rate scheduler (`cosine`) with `max_steps`.
   - Store hyperparameters (`beta`, `alpha`, etc.) for use during training.
   - Prepare conditional prompt template or tokenization strategy compatible with `model.py`.
   
2. **Data Preparation**
   - Dataset provides pairs: `(prompt, response, class_label, reward_weight)`.
   - For each batch:
     - Batch size matching configured `batch_size`.
     - Input prompts are constructed using the conditional prompt template (e.g., prepend with `<|class|> GPT4 User:` for `expert` class, or corresponding sub-optimal).
     - Tokenize `prompt` and `response` separately, respecting maximum sequence length (likely 4,096 tokens).
     - Prepare input IDs and attention masks for the model.
   - Input features must incorporate class-conditioned context.

3. **Training Loop (`train`)**
   - Loop over epochs (`epochs`) and steps (`max_steps` or dataset size).
   - For each batch:
     - Load batch data: `(prompts, responses, class_labels, reward_weights)`.
     - Tokenize prompts and responses.
     - Set model inputs with conditional prompts according to class label, via `model.py`.
     - Forward pass:
       - Obtain model logits.
       - Compute the negative log-likelihood (cross-entropy loss) for each token.
     - Compute the **weighted loss**:
       - Loss for each sample \(i\): \(w_i \times \text{cross-entropy}\)
       - \(w_i = \exp\left(\frac{1}{\beta} r_{c_i}(x_i, y_i)\right)\)
     - Sum or average over tokens and batch.
     - Apply gradient clipping (`max_grad_norm`).
     - Perform optimizer step.
     - Step scheduler.
   - Optional: Gradient accumulation if memory constraints exist.

4. **Reward Weight Computation**
   - Coarse reward based on class:
     - For `'expert'` class (GPT-4): \(r_c=1.0\).
     - For `'sub-optimal'` class (GPT-3.5): \(r_c=\alpha=0.8\).
   - Compute \(w_i = \exp\left(\frac{1}{\beta} r_c\right)\):
     - Use `torch.exp` on rewards scaled by `1/beta`.
   - These weights influence the loss to emphasize high-reward (better quality) examples.

5. **Model Conditioning**
   - Use a specific conditioning token (e.g., `<|class|>`) or prompt template.
   - During batch construction:
     - For class `'expert'`: prepend or insert "GPT4 User:" style prompt.
     - For `'sub-optimal'`: prepend "User:" style prompt.
     - Ensure tokenization aligns with model's tokenizer.
   
6. **Gradient Update and Regularization**
   - Loss is purely supervised (cross-entropy), weighted by reward-based weights.
   - Gradient clipping to prevent instability (`max_grad_norm`).
   - Maintain exponential moving average or other training stabilization if desired.

7. **Logging and Progress Tracking**
   - Log training loss, average reward, gradient norms, and learning rate.
   - Track epoch, step, and cumulative metrics.

8. **Model Saving (`save_model`)**
   - Save final model weights and optimizer states after training.
   - Save to the output directory specified (e.g., `output/openchat_finetuned`).

---

### Additional Considerations

- **Batch Sampling**
  - Use shuffling to ensure dataset variability.
  - Possibly implement random sampling or curriculum over dataset segments.

- **Loss Function**
  - Standard cross-entropy over the tokenized response.
  - Incorporate padding masks to ignore loss on pad tokens.

- **Handling Multi-GPU / Distributed Training**
  - If on multiple GPUs, wrap optimizer/ model in `DistributedDataParallel` or `DataParallel`.
  - Synchronize gradients.

- **Hyperparameters and Scheduling**
  - Use `warmup_steps` for learning rate warm-up.
  - Cosine decay schedule:
    \[
    \eta_t = \eta_{max} \times 0.5 \left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
    \]
  - Pass the `beta` value into the loss weighting formula consistently.

- **Stopping Criteria**
  - Train for specified `epochs` or until reaching `max_steps`.
  - Possible early stopping based on validation loss or evaluation metrics.

---

### Summary of Main Workflow

```plaintext
initialize Trainer with model, dataset, hyperparameters
for epoch in epochs:
    for each batch in dataset:
        prepare prompts and responses with class-conditioning
        tokenization -> input_ids, attention_mask
        forward pass to get logits
        compute token-wise cross-entropy loss
        apply weights w_i to each sample based on their reward r_c
        sum over batch to get total loss
        backpropagation with gradient clipping
        optimizer step
        scheduler step
    if epoch % eval_interval == 0:
        run evaluation
save model once training completes
```

---

### Clarifications Needed Before Implementation

- Exact format of dataset samples—are they JSON objects, CSV lines, or another schema?
- Specific implementation of class-conditioning prompts or special tokens.
- Choice of tokenizer handling, especially if special tokens are used.
- Whether to use mixed-precision training and any hardware constraints.
- Precise reward calculation and scaling (though initial plan uses \(\beta=0.2\), \(\alpha=0.8\)).

This comprehensive logic guide ensures correct alignment with the paper's methodology while being ready for coding.

