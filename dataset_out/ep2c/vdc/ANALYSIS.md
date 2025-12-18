# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

**Logic Analysis for `dataset_loader.py`**

---

### 1. Purpose and Responsibilities
- Implement a `DatasetLoader` class responsible for:
  - Loading datasets: CIFAR-10, ImageNet-100, ImageNet-Dog.
  - Applying data augmentations:
    - Poisoning: embed triggers, modify images and labels.
    - Noisy labels: corrupt labels according to symmetric or asymmetric noise ratios.
  - Providing train/test datasets ready for subsequent training or evaluation.
- Ensuring datasets adhere to the structure and standards used in experiments, including class imbalance and data splits.

---

### 2. Input and Configuration
- Receive a configuration object (`config`) conforming to the provided `config.yaml`.
- Option parameters:
  - Dataset name (`dataset_name`): e.g., `'cifar10'`, `'imagenet100'`, `'imagenet_dog'`.
  - Poisoning parameters:
    - `poisoned`: boolean (apply poisoning if true).
    - Trigger strategies (`triggers`): list of trigger types.
    - Embedding method: string describing trigger embedding.
    - Label changing strategy: target label.
    - Poison ratio (if applicable).
  - Noisy label parameters:
    - Type: symmetric or asymmetric.
    - Ratio: float (e.g., 0.4).
  - Dataset-specific parameters: images per class, total samples, etc.

### 3. Dataset Loading
- **For CIFAR-10:**
  - Use `torchvision.datasets.CIFAR10` for both training and test.
  - Apply transformations to match dataset specifications (`image_size`).
  - Load full dataset into memory or on-the-fly.
- **For ImageNet-100 / ImageNet-Dog:**
  - Load datasets from local paths or datasets API.
  - Use image folder structure or a custom loader with labels.
  - Resize images to `image_size`.
  - Manage class label indexing (labels 0 to K-1).

### 4. Poisoned Data Augmentation
- When `apply_poisoning`:
  - For each selected sample (based on `poison_ratio`), embed trigger:
    - Use the specified `embed_method`:
      - For visible triggers: overlay patch at fixed position.
      - For invisible triggers: apply blending or warping.
      - For others: follow detailed methods in appendix C.2.
    - Change labels:
      - Replace original label with `target_label`.
  - Keep track of poisoned indices for evaluation.
- Trigger embedding should be robust but straightforward; e.g., overlay a small patch or pattern at a predefined location, consistent across samples.

### 5. Noisy Labels Augmentation
- When `apply_noisy_labels`:
  - For each sample (or a subset based on ratio):
    - Generate noisy label:
      - Symmetric: randomly flip label to any class ≠ original.
      - Asymmetric: flip to 'next' class or similar.
    - Keep images unchanged.
  - Maintain indices of noisy labels for evaluation.

### 6. Hybrid Dataset Construction
- Combine poisoned and noisy label augmentation:
  - For datasets where both are configured:
    - Randomly select `poison_ratio` samples to poison.
    - Randomly select `noise_ratio` samples to have noisy labels.
    - Some samples may have both modifications.
  - Ensure the total number of dirty samples matches targets.

### 7. Data Structure and Output
- Return dataset objects:
  - Typically PyTorch datasets (`torch.utils.data.Dataset`) with:
    - `__getitem__` to provide `(image, label, index)` if needed.
    - Keep track of poisoned/noisy status via metadata or separate lists.
- Provide train/test splits as separate dataset objects.
- Possibly provide mask indices for training or validation.

### 8. Data Preprocessing and Transformations
- Apply consistent transformations:
  - Resize to `image_size`.
  - Normalize, convert to tensor.
- For poisoning, embed triggers **before** transformations if needed.
- For datasets with triggers, ensure images are correctly wrapped.

### 9. Considerations for Reproducibility
- Use fixed seed for randomness if deterministic poisoning/noise is desired.
- Make seed configurable or set explicitly.
- Consistent trigger placement and parameters across runs.

### 10. Interface and Usage
- `DatasetLoader` class:
  - Should have `__init__(self, config)` to initialize parameters.
  - `load_data()` method:
    - Loads raw dataset.
    - Applies poisoning/noise if requested.
    - Returns train/test datasets.
  - Optional methods:
    - `apply_poisoning()`
    - `apply_noisy_labels()`
- Maintain internal state for dataset, indices, etc.

---

### 11. Summary of Key Implementation Steps
1. Parse configuration parameters.
2. Load datasets using appropriate library functions.
3. For each dataset:
   - Resize and normalize images.
   - Based on config, apply poisoning:
     - Overlay trigger patches or patterns.
     - Change labels accordingly.
   - Apply noisy label corruption:
     - Flip labels with specified probability.
4. Create train/test dataset objects.
5. Return datasets with associated metadata for evaluation.

---

### 12. Additional Notes
- Ensure dataset indices for poisoned/noisy labels are stored for detection analysis.
- The trigger embedding should follow the specifics in appendix C.2, targeting trigger size, position, and method.
- Keep the dataset structure compatible with downstream processes, e.g., DataLoader.

---

This comprehensive logic analysis forms a detailed roadmap for implementing the `DatasetLoader` class aligning with the paper's methodology and experimental setup.

## evaluation.py

# Logic Analysis for `evaluation.py`  
**Purpose:**  
Implement the `Evaluator` class responsible for scoring individual question responses, aggregating these into a sample-level score, applying detection thresholds, and classifying samples as clean or dirty. It consumes correctness indicators (`e_i^j`) and similarity scores from `ModelInference`, and outputs final detection results for datasets.

---

## 1. Input Data and Dependencies  
- **Inputs per sample:**  
  - Correctness indicators for each question: `e_i^j` (Boolean: True if answer is correct, False otherwise).  
  - Similarity score `similarity_i`: a float that measures semantic closeness between image and label (optional, if used in combined scoring).  
  - Dataset label info (e.g., ground-truth label `y_i`, suspicious label `\tilde{y}_i`, etc.) — for optional consistency checks or reporting.

- **Configuration parameters:**  
  - Detection threshold `alpha` (e.g., 0.2).  
  - Possibly weighting coefficients if combining correctness and similarity scores.

- **Outputs:**  
  - For each sample:  
    - **Sample score:** scalar value indicating likelihood of being clean or dirty.  
    - **Classification:** binary label (dirty or clean), based on threshold comparison.  
  - Aggregated metrics:  
    - TPR (True Positive Rate): proportion of dirty samples correctly detected.  
    - FPR (False Positive Rate): proportion of clean samples incorrectly flagged.

---

## 2. Core Functionalities

### 2.1. Scoring Individual Questions  
- **Correctness evaluation `e_i^j`:**  
  - Boolean: determined by `ModelInference` based on comparing model response with expected answer.  
  - For deterministic label-specific questions: string matching (e.g., presence of "yes" or "no").  
  - For general questions: a more complex evaluation, e.g., language model based response classification via prompt.  

- **Similarity score `similarity_i`:**  
  - Optional metric provided by `ModelInference` that measures semantic alignment between the image and label.  
  - Can be directly used as another confidence signal or combined with correctness.

---

### 2.2. Aggregating per-sample score `s_i`  
- **Method:**  
  ```python
  s_i = (Number of correct answers) / (Total questions)  
  ```  
  - For each sample, count how many questions are answered correctly (i.e., sum of `e_i^j` True values).  
  - Normalize by total number of questions `N_q`.  
  - Alternatively, if similarity score is incorporated, combine them as:
    ```python
    s_i = w_c * (correct_answers_ratio) + w_s * similarity_score
    ```
    where `w_c` and `w_s` are weights (e.g., both 0.5 or as per hyperparameter).  

- **Decision rule:**  
  ```python
  is_dirty = (s_i < alpha)
  ```  
  - If the aggregated score `s_i` falls below the threshold `alpha`, classify the sample as dirty.  

### 2.3. Thresholding and Classification
- Apply the detection threshold `alpha`.  
- Samples with `s_i < alpha` are flagged as dirty, otherwise clean.  
- Ensure robust handling for edge cases:  
  - All answers correct: `s_i` close to 1.  
  - All answers wrong: `s_i` close to 0.  
  - Handle potential tie or borderline cases with sensitivity analysis if needed.

### 2.4. Metrics Calculation (TPR, FPR)  
- After classifying all samples, compare with ground truth labels:  
  - **True Positives (TP):** dirty samples correctly detected as dirty.  
  - **False Positives (FP):** clean samples incorrectly flagged as dirty.  
- **Compute:**  
  ```python
  TPR = TP / (Total actual dirty samples)  
  FPR = FP / (Total actual clean samples)
  ```  
- These metrics can be accumulated over the dataset after classification, for reporting.

---

## 3. Additional Aspects and Implementation Details

### 3.1. Handling Diverse Questions Types
- For label-specific deterministic answers:  
  - String matching for confirmation ("yes"/"no").  
- For general questions:  
  - Use a language model (e.g., GPT) to evaluate answer correctness through a dedicated prompt.  
  - Implement as a separate evaluation function, accepting answer strings and expected semantics, returning a Boolean correctness indicator.

### 3.2. Multiple Question Responses and Voting
- **Ensemble:**  
  - The final sample score `s_i` is calculated as the proportion of questions answered correctly.  
  - Can incorporate confidence from similarity scores (e.g., average or weighted sum).  
- **Voting ensemble:**  
  - The Boolean correctness of each question since the paper uses a voting scheme based on correctness votes.

### 3.3. Threshold Sensitivity and Tuning
- Threshold `alpha` is set to 0.2 as per experiments, but should be customizable for experiments or hyperparameter tuning.

### 3.4. Output and Logging
- Store per-sample scores and classification results for downstream analysis.  
- Log the distribution of scores, true/false detection labels, and ground truth labels for computing and plotting metrics (TPR, FPR).

---

## 4. Data Structures
- For each dataset batch:  
  - List of correctness indicators: `e_list` (Booleans).  
  - List of similarity scores: `similarity_list` (floats).  
  - List of ground-truth labels: `label_list`.  
  - Aggregated sample scores: `sample_scores` (floats).  
  - Final detection labels: `detection_labels` (Booleans).  

- For overall evaluation:  
  - Counters: `TP`, `FP`, `TN`, `FN`.  
  - Metrics: calculated post processing.

---

## 5. Summary of Processing Steps in `evaluation.py`:
1. Receive batch or dataset of samples:  
   - Foreach sample: obtain per-question correctness (`e_i^j`) and similarity score (`similarity_i`).  
2. For each sample:  
   - Count the number of correct answers.  
   - Compute `sample_score = (correct_answers) / N_q` (or combined with similarity).  
3. Classify as dirty or clean based on threshold `alpha`.  
4. Collect true labels (ground truth) and detection labels.  
5. After all samples are processed, compute overall metrics (TPR, FPR).  
6. Return per-sample scores, labels, and metrics.

---

## 6. Implementation Constraints
- Ensure compatibility with `ModelInference` outputs.  
- Make code modular to allow easy tuning of weights, thresholds, and question count.  
- Include exception handling for API failures or ambiguous responses.

---

This thorough analysis provides a clear blueprint for implementing `Evaluator` in `evaluation.py`. It emphasizes modularity, parameterization, and metrics calculation aligned with the experimental procedures, facilitating precise reproduction of the experiments.

## main.py

# Logic Analysis for `main.py` — High-Level Orchestrator

This file functions as the central controller that manages the entire process flow for dataset loading, question generation, inference, detection, and model retraining, relying on the modular design previously outlined. The following analysis provides a step-by-step understanding of the required logic, flow, data dependencies, and control structures necessary to realize the script according to the paper’s methodology and design specifications.

---

## 1. Initialization and Configuration Loading

- **Load configuration (`config.yaml`):** Use a configuration parser to load all hyperparameters, dataset paths, model parameters, API keys, thresholds, and flags controlling process steps (e.g., whether to retrain).
- **Set up environment:**
  - Verify required packages (`torch`, `transformers`, `requests`, etc.) are installed.
  - Confirm API access tokens for OpenAI GPT are available if using GPT-4 via API.
- **Instantiate core component classes:**
  - `DatasetLoader`
  - `PromptGenerator`
  - `ModelInference`
  - `Evaluator`

*Purpose:* Centralize parameter-driven behavior and prepare modules for execution.

---

## 2. Dataset Loading and Preparation

- **Select dataset(s) to process:** Based on configuration, e.g.,
  - CIFAR-10
  - ImageNet-100
  - ImageNet-Dog
- **Load datasets:**
  - Call `DatasetLoader.load_data()` with dataset-specific parameters.
  - Data includes images, true labels, training/test splits.
  
- **Apply poisoning/noise if specified:**
  - For poisoned: invoke `DatasetLoader.apply_poisoning(poison_params)` with triggers and embedding info.
  - For noisy labels: invoke `DatasetLoader.apply_noisy_labels(noise_params)` with noise type and ratio.
- **Result:** A dataset (`dataset`) with mixed clean and dirty samples, annotated internally via labels.

*Purpose:* Obtain the initial dataset, possibly contaminated, for further analysis.

---

## 3. Question Generation for Each Sample

- **Iterate over dataset samples:**

  For each sample `(image, label)`:
  
  - **Generate general questions:**
    - Call `PromptGenerator.generate_general_questions(label, questions_per_sample)`
  - **Generate label-specific questions:**
    - Call `PromptGenerator.generate_label_specific_questions(label, questions_per_sample)`
  - **Collect questions:**
    - Maintain per-sample list: `questions_list` (e.g., concatenating both question types)

*Design Consideration:*
- Batch question prompts if API call overhead is a concern.
- For large datasets, consider batching or sampling.

*Purpose:* Create prompts that will elicit semantic responses about the visual content aligned with the label.

---

## 4. Answering Questions Using Multimodal LLM

- **Per sample, per question:**

  - Call `ModelInference.answer_questions(image, questions)`.
  
- **Aggregate answers:**
  - Generate a list of answers `answers_list`.
  
- **Answer response:**
  - Responses are strings, processed for correctness evaluation and similarity scoring.

*Design Consideration:*
- Use API calls for each sample or batch if supported.
- Handle API rate limits and errors gracefully.

*Purpose:* Obtain semantic responses from the MLLM for each visual question.

---

## 5. Evaluation of Answers and Semantic Consistency

- **Per question:**

  - Evaluate correctness: 
    - For deterministic answers (label-specific), use string matching (“yes”/“no”) directly.
    - For free-form (general questions), invoke `ModelInference.evaluate_response(question, answer, label)` which internally calls GPT to determine correctness.
  - Obtain a boolean `is_correct` and optional similarity score `similarity`.
  
- **Per sample:**

  - Calculate `sample_score = (Number of correct answers) / (Total questions)`
  
- **Comparison with threshold (`detection_threshold`):**

  - Classify each sample:
    - If `sample_score < threshold`, label as **dirty**
    - Else, label as **clean**

*Design considerations:*
- Store classifications for dataset filtering.
- Record per-sample scores and correctness details for further analysis.

*Purpose:* Quantify the semantic inconsistency that indicates whether a sample is dirty or clean.

---

## 6. Aggregate Detection Results and Generate Final Sample Labels

- **For each sample:**
  - Confirm if dirty (threshold-based detection)
  - Accumulate statistics: number/ratio of samples flagged as dirty.
  
- **Output:**
  - Saved dataset with clean/dirty labels.
  - Optional: generate reports or logs of the detection process, per-sample scores.

*Purpose:* Identification of dirty samples for dataset purification.

---

## 7. Optional: Retrain Classifier on Purified Dataset

- **Filtering:**
  - Remove identified dirty samples from the dataset.
- **Retraining:**
  - Initialize a classifier (e.g., ResNet-18 as per experiment).
  - Train for specified epochs (`training.epochs`)
  - Use hyperparameters:
    - Optimizer: `SGD` with `learning_rate`
    - Batch size
    - Decay schedule if applicable
- **Evaluate:**
  - On the test set, record:
    - Clean accuracy (`ACC`)
    - Attack Success Rate (`ASR`) if backdoor attack present
  
*Purpose:* Verify that dataset purification improves downstream model performance and robustness.

---

## 8. Final Reports and Output

- **Write detection metrics:**
  - TPR, FPR for detection performance.
- **Results after retraining:**
  - Comparison of accuracy, attack success rate.
- **Optional Visualizations or logs:**
  - Plot detection thresholds vs. TPR/FPR
  - Log per-sample scores and detection states
- **Save all relevant outputs:** detection results, models, logs.

---

## 9. Error Handling & Robustness Measures

- Wrap API calls in try/except blocks.
- Log and handle missing models, API failures.
- Confirm dataset dimensions and labels match assumptions.

---

## 10. Summary of Dependencies & Sequence

- **Sequence overview:**
  1. Load config
  2. Initialize modules
  3. Load and prepare dataset
  4. For each sample:
     - Generate questions
     - Answer questions
     - Evaluate correctness and similarity
     - Compute sample score
     - Classify sample
  5. Aggregate detection results
  6. (Optional) Retrain classifier
  7. Save results and logs

---

This detailed logical blueprint ensures that `main.py` will orchestrate all components correctly, following the methodology's intent and design constraints, regardless of the dataset or attack type. It emphasizes modularity, extensibility, and clarity, aligning perfectly with the experimental framework described in the paper.

## model_inference.py

**Logic Analysis for model_inference.py**

---

### **Overall Objective:**
Implement the `ModelInference` class that provides the core functionalities for processing visual and textual data via multimodal large language models (MLLMs). Specifically, it should:
- Generate answers to questions about images (VQA)
- Evaluate the correctness of answers (e.g., matching responses with expectations)
- Compute semantic similarity between the image content and labels
- Wrap API calls effectively, manage prompts, and handle responses reliably

### **Dependencies & Inputs:**
- Prompts generated externally by `prompt_generator.py`
- Dataset images loaded (via `dataset_loader.py`)
- External or local multimodal inference models such as OpenAI's GPT-4 with vision, BLIP2, or other open-source models  
- Utility functions in `utils.py` for loading images, API calls, and processing responses

---

### **Key Functionalities & Implementation Details**

#### 1. **Answer Questions about Images (VQA)**
- **Input:**
  - `image_path` (or image object)
  - List of questions (`questions`)
- **Process:**
  - For each question:
    - Compose a prompt that includes the visual information (may involve passing the image as an input, either as a URL/path or embedding)
    - Submit the prompt to the MLLM API or inference engine
    - Obtain textual response for each question
- **Output:**
  - List of answers (`answers`)
- **Implementation notes:**
  - Use API functions (e.g., `call_api()` or library inference)
  - Handle batch processing if supported
  - Ensure prompt format aligns with model's expected input
  - Manage potential API errors or timeouts

#### 2. **Evaluate Correctness of Responses**
- **Input:**
  - question string
  - answer string (generated by MLLM)
  - `label`: Ground-truth label (string)
- **Process:**
  - Use a prompt to GPT-4 or other LLM for semantic evaluation:
    - e.g., "Does the answer '{answer}' correctly describe '{label}'? Answer 'yes' or 'no'."
  - Parse the response:
    - Interpret "yes" as `True`, "no" as `False`
    - Use string matching with keywords for consistency ("yes", "no")
  - Alternatively, for deterministic questions (label-specific with known answer), apply simple string matching:
    - Check if answer contains "yes" or "no" as expected
- **Output:**
  - Boolean: `is_correct`
  - Float: similarity score from the API (if available)
- **Implementation notes:**
  - For general questions, rely on GPT evaluation
  - For label-specific questions, string matching suffices

#### 3. **Compute Semantic Similarity between Image Content and Labels**
- **Input:**
  - `image_path` or image object
  - `label`: string
- **Process:**
  - Use a pre-trained vision-language model (e.g., CLIP) to encode:
    - Image: via image encoder, produce vector `I`
    - Label text: via text encoder, produce vector `T`
  - Compute cosine similarity:
    \[
      similarity = \frac{I \cdot T}{\|I\| \|T\|}
    \]
  - This score indicates how well the image matches the label's semantics.
- **Output:**
  - Float similarity score (range probably between -1 and 1)
- **Implementation notes:**
  - Utilize `transformers` library models like CLIP
  - Handle image resizing, normalization per model requirements
  - Implement caching if multiple evaluations are performed

---

### **Supporting Implementation Considerations**

- **API Management:**
  - If using OpenAI GPT-4 (with vision), create functions to send requests with appropriate prompts
  - Handle token limits, retries, rate limits
  - Configure response tempeture and stopping tokens if needed

- **Prompt Engineering:**
  - Use or generate prompt templates for:
    - Question answering
    - Correctness evaluation
  - Incorporate question/image info into prompts clearly

- **Data Handling:**
  - Load images via `utils.py` with consistent preprocessing (resizing, normalization)
  - For API inputs, convert images to base64 or URL as required by the API
  - Ensure question responses are processed (e.g., strip whitespace, parse YES/NO)

- **Error Handling and Robustness:**
  - If API call fails, implement retries
  - Validate API responses match expected format
  - Log errors for debugging

- **Batch Processing & Efficiency:**
  - Process multiple questions in batches if supported
  - Minimize API calls duration
  - Cache repeated responses where applicable

---

### **Class and Method Structure:**

```python
class ModelInference:
    def __init__(self, api_keys, model_name, max_tokens=..., temperature=...):
        # Initialize API credentials, select models, set parameters
        pass
    
    def answer_questions(self, image_path, questions):
        # Generate answers for each question
        # Return list of answers
        pass

    def evaluate_response(self, question, answer, label):
        # Use GPT prompt (or heuristic) to evaluate correctness
        # Return boolean and optional similarity score
        pass

    def get_semantic_similarity(self, image_path, label):
        # Encode image and label through CLIP or similar
        # Return cosine similarity score
        pass
```

---

### **Sequence of Calls in Main Workflow:**

1. Load an image
2. For each sample:
   - Generate questions (via external `prompt_generator`)
   - Call `answer_questions(image, questions)` to get answers
   - For each answer:
     - Call `evaluate_response()` with label to check correctness
   - Calculate sample score as ratio of correct answers
   - Use `get_semantic_similarity()` for an auxiliary similarity score
3. Use scores to decide if sample is dirty based on threshold `α`
   
---

### **Summary:**
- Integrate and wrap multimodal inference models
- Manage prompt-based queries for question answering and evaluation
- Encode visual and text modalities for similarity computations
- Maintain robustness, error handling, and efficiency
- Ensure adherence to modular design and compatibility with entire pipeline

This detailed logic analysis should guide the development of `model_inference.py` accurately aligned with the VDC methodology and experimental design.

## prompt_generator.py

{
  "prompt_generator.py": [
    {
      "Objective": "Implement the PromptGenerator class that generates contextual prompts for different types of questions used in the VDC pipeline, specifically for question generation modules which ask a multimodal large language model (MLLM) to produce insightful questions about images given labels.",
      "Core Responsibilities": [
        "Generate general questions applicable across images and labels for holistic understanding.",
        "Generate label-specific questions tailored to a particular label to extract localized semantic details.",
        "Use fixed templates or prompt templates that are flexible for different dataset labels (e.g., CIFAR-10 classes, ImageNet labels, dog breeds).",
        "Interfacing with GPT-based models (via prompts) or defining prompt strings to guide the question generation process."
      ],
      "Input Data": [
        "List of class labels for the dataset (e.g., ['cat', 'dog', 'airplane']).",
        "Number of questions to generate per label (e.g., 4 questions).",
        "Question type indicator ('general' or 'label-specific'), which guides prompt selection.",
        "Optional: Additional context or dataset-specific instruction (e.g., dataset name, label descriptions)."
      ],
      "Output Data": [
        "A list or array of generated questions per label, each being a string prompt suitable for passing to the GPT/MLLM for further processing.",
        "Structured data, such as a list of tuples: (question, expected answer), where applicable."
      ],
      "Design Details and Constraints": [
        "Use predefined fixed prompts for general questions (e.g., 'Describe the image briefly.'), as per Appendix E1.",
        "Generate label-specific questions based on dataset labels, using prompts that encourage questions about object attributes, functions, or characteristics relevant to that label, as in Appendix E2.",
        "For label-specific questions, dynamically insert the label name into the prompt template.",
        "Design prompts with enough contextual information to produce meaningful, insightful questions that would ideally lead the answer to 'yes' if the label matches.\n- For example: 'Generate questions to verify the object in the image corresponds to the label {label}. The questions should be answerable with 'yes' or 'no' and focus on attributes or defining features.'",
        "Ensure prompts are compatible with GPT-like APIs; questions should be unambiguous and natural language.",
        "In case of multi-question generation, maintain consistent question styles and prompts to avoid model confusion."
      ],
      "Implementation Approach": [
        "Create two main methods within PromptGenerator: generate_general_questions(label, num_questions) and generate_label_specific_questions(label, num_questions).",
        "For generate_general_questions():",
        "  - Select fixed question templates related to image description, summarization, or overall content. Use an internal list of templates or standard prompts as in Table 10.",
        "  - For each requested question, pick a template (randomly or sequentially) and format it accordingly.",
        "For generate_label_specific_questions():",
        "  - Use dataset label and generate prompts as per Appendix E2.",
        "  - Incorporate the label name into prompts to ask attribute-based questions, breed-specific features, or function-related questions.",
        "  - If interfacing with GPT, send the prompt with label info and receive questions; otherwise, generate prompts directly in code.",
        "Implement a standardized prompt format to ensure consistency.",
        "Maintain internal templates or prompt strings that can be dynamically filled with label names.",
        "Assign unique IDs or manage the order if multiple questions are generated for each label.",
        "Provide an interface to fetch questions as a list to downstream modules."
      ],
      "Additional Considerations": [
        "Prompt engineering is critical: prompts should avoid ambiguity to produce informative questions.",
        "Design prompts so that the generated questions are diverse yet relevant.",
        "Optionally, incorporate dataset-specific vocabulary, facts, or context if available.",
        "Ensure the output questions are appropriately formatted for the inference pipeline, i.e., plain text strings."
      ],
      "Summary": "The PromptGenerator acts as an interface for crafting domain-specific, instructive prompts for question generation modules. It leverages fixed templates and dataset-derived labels to produce questions that facilitate cross-modal semantic verification, critical to detecting dirty samples via the VDC pipeline."
    }
  ]
}

## utils.py

# Logic Analysis for utils.py

The `utils.py` module serves as a foundational support component providing essential shared functionalities across all core modules (`dataset_loader.py`, `prompt_generator.py`, `model_inference.py`, `evaluation.py`, `main.py`). Its responsibilities include data handling, image processing, API wrapper functions for LLM/MLLM calls, configuration helpers, and general utility functions to streamline and standardize operations required during data poisoning/noisy label application, question generation, inference, and evaluation procedures. Below is a detailed breakdown of required functionalities aligned with the paper methodology, experimental setup, and the provided configuration.

---

## 1. Configuration Management

- **Purpose:**
  - Load and provide access to the configuration parameters from the `config.yaml` file.
- **Approach:**
  - Implement a singleton or module-level cache to load `config.yaml` at initialization.
  - Use `PyYAML` or `ruamel.yaml` to parse YAML.
  - Provide functions such as `get_dataset_config()`, `get_attack_config()`, `get_training_config()`, etc.
- **Usage:**
  - To ensure consistency across modules, all configuration parameters (dataset sizes, thresholds, model hyperparameters, attack ratios, etc.) should be accessible via utility functions.

---

## 2. Data Handling & Image Processing

- **Loading Images:**
  - Functions to read images from disk or from dataset structures.
  - Support converting images to specified input sizes (as per dataset configs: 32x32 or 224x224).
- **Preprocessing:**
  - Normalization (mean/std) compatible with the models used (e.g., ResNet).
  - Resize images if necessary.
  - Convert images to tensors suitable for model inputs.
- **Batching:**
  - Generate batches for inference or training.
  - Support for DataLoader wrappers if needed.
- **Poisoning & Noise Application:**
  - Given that poisoning is done during dataset creation, the actual embedding of triggers (e.g., overlay, blending) must be handled before dataset loading.
  - For noisy labels, functions to flip labels (symmetric/asymmetric).

## 3. API Wrappers for Model Inference & Evaluation

- **LLM API Wrapper:**
  - Function to send prompt strings to GPT-family APIs (OpenAI or other providers).
  - Ensure robust handling:
    - Rate limiting
    - Retry logic
    - Timeout handling
  - Functions:
    - `call_llm(prompt: str) -> str`
    - Should be generic, supporting different prompt templates.
- **Multimodal Model API Wrapper:**
  - Interface to run inference on visual-question-answering models (e.g., BLIP2, Otter).
  - Due to the variability in models, implement:
    - `answer_question(image: PIL.Image, question: str) -> str`
  - For local inference, use Hugging Face Transformers pipeline when possible.
  - For API-based inference (e.g., OpenAI vision), wrap API calls.

## 4. Prompt Construction Functions

- **Question Generation:**
  - Functions to generate prompts for:
    - General questions
    - Label-specific questions
  - E.g., `generate_general_question(label: str, count: int) -> List[str]`
  - E.g., `generate_label_specific_questions(label: str, count: int) -> List[str]`
  - These functions should:
    - Use fixed templates from Appendix E1.
    - Possibly automatically generate label-specific questions via GPT (if desired), wrapping GPT prompt calls.
- **Response Evaluation:**
  - Function to generate prompts for answer correctness:
    - E.g., `generate_evaluation_prompt(response: str, label: str) -> str`

## 5. Inference & Response Processing

- **Answering Questions:**
  - Use API wrappers or local models:
    - Input: image (loaded PIL or tensor)
    - Question: string
    - Output: string response
- **Evaluating Response Correctness:**
  - String matching:
    - For deterministic answers ("yes"/"no"), check presence of "yes" or "no".
  - Use GPT or GPT-based prompt (from the evaluation prompt function) for free-text answers, especially for general questions.
  - Parse responses with consistent heuristics:
    - Lowercase comparison
    - Keyword detection
  
- **Semantic Similarity:**
  - Implement a function to compute cosine similarity between:
    - Image embedding from CLIP
    - Label text embedding from CLIP or similar model
  - Use `transformers` to load CLIP model and tokenizer:
    - `encode_image(image)` -> embedding
    - `encode_text(text)` -> embedding
  - Return a float similarity score (range: [-1, 1], normalized to [0, 1]).

## 6. Scoring and Detection

- **Sample Score Calculation:**
  - Collate correctness indicators (`e_i^j`) for each question.
  - Compute confidence score `s_i`:
    \[
      s_i = \frac{\sum_{j=1}^{N_q} \mathbb{1}(e_i^j = true)}{N_q}
    \]
- **Detection Decision:**
  - Based on threshold (e.g., 0.2), classify sample as:
    - Dirty if `s_i < threshold`
    - Clean if `s_i >= threshold`

## 7. Utility Functions & Miscellaneous

- **Image Conversion:**
  - From dataset format (numpy arrays, PIL images) to model input tensors.
  - For visual models that require specific size, perform resize.
- **Progress & Logging:**
  - Use `tqdm` for progress bars during batch processings.
  - Logging of API call success; error handling.
- **Random Seeds:**
  - To ensure reproducibility, set seeds for random, numpy, and torch operations.
- **Timer/Performance Metrics:**
  - Measure inference times.
  - Record number of questions generated per sample and inference calls.

---

## 8. Cross-module Integration & Usage Guidelines

- All functions should accept parameters from central configuration.
- Functions like API calls should be parameterized:
  - API keys
  - model settings
  - prompt templates
- Provide optional verbose logging.
- Ensure batch processing minimizes API calls (if supported) or handle per-sample calls efficiently.
- Output should include:
  - Per-sample scores
  - Detected labels (dirty or clean)
  - Optional: confidence metrics or anomaly scores for post-hoc analysis.

---

## Summary

The `utils.py` must encompass:
- Configuration loading utilities
- Image processing helpers (load, resize, tensor conversion)
- Poison/noise augmentation functions
- API wrappers for LLMs and multimodal models
- Prompt construction functions for question generation and evaluation
- Response parsing functions
- Score calculation functions
- Detection decision logic
- Batch/stream processing support
- Logging and error handling

All functions should be modular, well-documented, and designed for efficiency, robustness, and flexibility, enabling the core modules to implement VDC fully aligned with the paper methodology.

---

This comprehensive analysis guides the development of `utils.py` to support an end-to-end, reproducible implementation of VDC, ensuring adherence to the paper's methodology and experimental design.

