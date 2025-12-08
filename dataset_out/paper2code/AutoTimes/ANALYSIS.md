# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

# Logic Analysis for dataset_loader.py

**Objective**: Implement a `DataLoader` class that efficiently loads, preprocesses, and structures multiple time series datasets, providing an interface to retrieve segmented, normalized series with associated timestamps and metadata for subsequent training and evaluation.

---

## 1. Inputs and Dataset Specifications

- The class should accept `dataset_paths` as a dictionary containing file paths for each dataset (per the plan).
- Dataset files are CSVs with time series data, including:
  - *Series data*: multivariate (C variates) or univariate (1 variate).
  - *Timestamps*: associated date/time info, possibly in textual form (e.g., "2016/07/05 00:00:00").
- Other relevant info:
  - Sampling frequency (hourly, 10-min, daily, etc.) should be in dataset description.
  - No explicit file format given: assume CSV with columns for time and variates.
  - Provide support for the datasets listed in the configuration yaml.

---

## 2. Data Loading

- Read each dataset file using `pandas.read_csv`.
- For each dataset:
  - Identify appropriate columns: timestamp column and variate columns. (Assume common convention or specify column names, e.g., "timestamp" plus variates.)
  - Load data as a DataFrame, converting timestamps to `datetime` objects (using `pd.to_datetime`).
- Save raw series: numpy array of shape `(T, C)` with T time points and C variates.
- Save timestamps as `np.ndarray` of datetime objects for easier manipulation.

---

## 3. Timestamp Conversion and Textual Prompts

- For each timestamp:
  - Convert to textual format suitable for prompt (e.g., `"YYYY/MM/DD HH:MM:SS"`).
  - Use `strftime("%Y/%m/%d %H:%M:%S")` on pandas datetime object.
- Store these textual representations alongside series data for prompt construction.

---

## 4. Data Normalization

- To match paper procedures:
  - Normalize each series using a standard method, e.g., min-max scaling or z-score.
  - Since times series vary by dataset, choose normalization per dataset:
    - **Training/validation/test split**: given as data is split chronologically, normalization should be:
      - Fit normalization on training set only (for realism and avoiding data leakage).
      - Apply same normalization parameters to validation and test sets.
  - Save normalization parameters for consistent processing during evaluation.

---

## 5. Series Segmentation

- Based on the `segment_size` (e.g., 96, 192, etc.):
  - Divide the entire series into non-overlapping segments:
    - For a series of length `T`, number of segments `N = floor(T / S)`.
    - Segment i: `series[(i-1)*S : i*S]`.
  - For multivariate series:
    - Each segment is shape `(S, C)`.
  - Store each segment as a numpy array of shape `(S, C)`.

- Result:
  - For each series: a list/array of segments:
    ```python
    segments = [series[(i-1)*S : i*S] for i in range(1, N+1)]
    ```
  - Store index of segments for time alignment with timestamps.

---

## 6. Metadata and Sample Structuring

- For each series:
  - Store:
    - *Series name/ID*
    - *Normalized series data* or list of segments
    - *Timestamps*: for each segment, represent start, end times, or per segment timestamps (e.g., starting timestamp + S time points).
    - *Series metadata*: any additional info (e.g., station ID, location, frequency).
- Encapsulate as a structured object, e.g., `TimeSeriesSample`:
  ```python
  class TimeSeriesSample:
      series: np.ndarray           # Full series data (T, C)
      timestamps: np.ndarray       # Corresponding datetime objects
      series_name: str
      series_metadata: dict
  ```
- For dataset loading, return a list of such samples, each representing one time series.

---

## 7. Dataset Standardization & Handling Multiple Datasets

- For code generality:
  - The loader can accept multiple dataset paths.
  - Load all datasets into a list of `TimeSeriesSample`.
  - For datasets with missing or inconsistent timestamp formats, provide options or preprocessing steps.
  - Use consistent normalization across all datasets or dataset-specific normalization.

---

## 8. Data Splits & Indexing

- Maintain the chronological order:
  - For training, validation, testing: split based on indices to prevent data leakage.
- For each series, store:
  - *Training segment indices*: first part of series.
  - *Validation & test indices*: later parts.

- For datasets with fixed train/validation/test splits, pre-calculated indices or date ranges should be used.

---

## 9. Supporting Datasets for Experimental Protocols

- Support the following datasets (specified in config):
  - ETTh1
  - Weather
  - ECL
  - Traffic
  - Solar-Energy
  - M4 (yearly, quarterly, monthly, weekly, daily, hourly)
  - M3 (yearly, quarterly, monthly, others)
- Each dataset may vary:
  - Different total length
  - Different number of variates
  - Different timestamp formats and granularity

**Implementation tip**: define dataset-specific loaders if needed, or generalize via settings.

---

## 10. Reproducibility & Output

- Provide methods:
  - `get_dataset()` or `load_all()` to load and process all datasets.
  - For each dataset, return structured objects (list of samples).
- Include options for:
  - Saving preprocessed datasets into cache for faster repeated loadings.
  - Accessing raw data, normalized data, segments, and timestamps separately.

---

## Summary

- **Main Tasks**:
  - Read CSVs, convert timestamps to `datetime` and textual formats.
  - Normalize each series using training set parameters.
  - Segment series into fixed-length, non-overlapping segments.
  - Store series, segments, textual timestamps, and metadata.
  - Return a structured list of `TimeSeriesSample` objects for downstream use.
  
- **Key Points**:
  - Support multiple datasets with flexible formats.
  - Maintain chronological integrity.
  - Prepare data for prompt-based models: textual prompt construction will happen downstream, using timestamp strings.
  - Ensure normalization is only fit on training data to avoid leakage.
  - Modular functions: load CSV, process timestamps, normalize, segment, package.

This thorough plan ensures the dataset loader serves as a reliable, flexible interface that seamlessly integrates with the training and prompt construction modules, aligning with the paper's methodological specifications.

## evaluation.py

Evaluation.py Logic Analysis

Overview:
The Evaluation module's primary goal is to load a trained AutoTimes model, construct appropriate prompts and input sequences for inference (including in-context prompt variations), perform autoregressive multi-step forecasting, compute relevant metrics (SMAPE, MAE, MSE), optionally generate visualizations, and report results in a structured manner. It must support different datasets, prediction horizons, and prompt strategies. All steps should adhere strictly to the protocol and experimental setup described in the paper.

---

1. Importing Required Libraries & Modules
- numpy as np
- torch
- transformers (for tokenizer and model loading)
- metrics functions (SMAPE, MAE, MSE)
- visualization tools (matplotlib or seaborn, optional)
- logging for tracking
- paths handling (os)

2. Initialization & Loading
- Inputs:
  - Path to saved model checkpoint
  - Dataset configuration (dataset name, dataset sample size, etc.)
  - Prediction horizon(s)
  - Metrics to evaluate
  - Any prompt strategy parameters
- Load the pre-trained language model backbone:
  - Use transformers library to load model (e.g., AutoModelForCausalLM)
  - Load the tokenizer matching the backbone
  - Load the saved embedding/projection layers (if separately saved)
- Set the model to evaluation mode
- Freeze backbone parameters if not already frozen during training

3. Data Preparation
- Load datasets:
  - Use dataset loader utility; load raw series, timestamps, metadata
  - For each time series sample (univariate/multivariate):
    - Normalize if required
    - Identify the last lookback segment: length = lookback_length (L)
    - Identify the forecast horizon F
- For zero-shot evaluation:
  - Use the last L time points as input
- For in-context evaluation:
  - Use prompt series (e.g., initial series segments from the earlier data) as prompts
  - Select prompt according to prompt_strategy (e.g., firstF, recent series, periodic, out-of-series)
- Convert series segments into tokens:
  - Use prompt_builder.convert_series_to_tokens() to create token sequence
  - Convert textual timestamps into tokenized prompt embeddings
  - Concatenate prompt tokens with the lookback series tokens

4. Building Input Sequences
- For each sample:
  - Construct prompt sequence:
    a. Prompt tokens (relevant series + timestamps, if in-context)
    b. Lookback window tokens
  - Include position embeddings:
    - Assign timestamp-based textual prompt embeddings
    - Embed timestamps textually ("YYYY/MM/DD HH:MM:SS")
  - Concatenate prompt + series tokens into input tensor 
- For prediction:
  - Prepare an initial input tensor for the model:
    - shape: (sequence length, embedding dimension)
    - with proper attention masks

5. Autoregressive Multi-step Prediction
- Loop over forecast horizon N:
  a. Feed current input sequence into the model
  b. Compute the logits or embeddings for the next token conditioning on previous tokens
  c. Extract the predicted next token embedding
  d. Decode the predicted token into series segment space using SegmentProjection
  e. Append predicted segment tokens (series values) to the output
  f. For the next step:
     - Update the input sequence by appending the newly predicted tokens (for iterative forecasting)
- This process produces predicted series segments step-by-step until reaching the total horizon (F).
  
6. Post-Processing Predictions
- Concatenate all forecasted segments to reconstruct the forecasted series
- Convert predicted token sequences back to numerical series
- If multiple samples:
  - Collect predictions per sample
  - Store predicted series in array form

7. Metrics Computation
- For each forecasted series:
  - Compare with ground truth series (from dataset split)
  - Compute SMAPE:
    \[
    \text{SMAPE} = \frac{1}{F} \sum_{t=L+1}^{L+F} \frac{| \hat{x}_t - x_t |}{( |x_t| + |\hat{x}_t|)/2}
    \]
  - Compute MAE:
    \[
    \text{MAE} = \frac{1}{F} \sum_{t=L+1}^{L+F} |\hat{x}_t - x_t|
    \]
  - Compute MSE:
    \[
    \text{MSE} = \frac{1}{F} \sum_{t=L+1}^{L+F} (\hat{x}_t - x_t)^2
    \]
- Aggregate metrics across all samples:
  - Compute mean and standard deviation for SMAPE, MAE, MSE
  - Optionally compute confidence intervals or p-values if desired

8. Visualization & Showcases
- Optional:
  - Plot actual vs. predicted series for selected samples to illustrate prediction quality
  - Visualize attention weights or token transition activations if applicable
  - Plot error distributions
  
9. Result Summarization & Reporting
- Print or log detailed metrics for each dataset and horizon
- Store results in structured format (dict, JSON)
- Save predictions if needed for further analysis or submission
- Summarize average errors and deviations for multiple runs (if multiple seeds)

10. Robustness & Error Handling
- Handle cases:
  - Series shorter than needed for prompt; skip or pad as necessary
  - Prompts not relevant; fallback to zero or generic prompts
  - Model inference errors
  - Inconsistent sequence lengths or tokenization mismatches

11. Additional Notes
- Ensure reproducibility:
  - Use consistent random seed
  - Set torch.no_grad() context
- Efficient batching:
  - Process multiple series in parallel
  - Use padding masks
- Handle multiple experiment settings:
  - Different datasets, forecast horizons, prompt strategies

---

Summary:
The Evaluation.py module performs the following key tasks in strict adherence to the methodology:
- Loads the trained model and tokenizer
- Loads and preprocesses datasets into series, timestamps, and tokens
- Constructs prompt sequences with flexible strategies, including textual timestamp embeddings
- Performs autoregressive inference in a step-by-step manner, using the frozen LLM backbone
- Converts predicted tokens into numerical series
- Calculates error metrics (SMAPE, MAE, MSE)
- Reports and visualizes results comprehensively
- Supports in-context, zero-shot, and traditional evaluation paradigms

This detailed logic ensures faithful reproduction of results and supports rigorous comparison across different datasets, prompt strategies, and model sizes.

---

If you need further clarification on implementation details or specific computational steps, please ask!

## main.py

# Logic Analysis for main.py — Entry Point of AutoTimes Reproduction Pipeline

This script serves as the orchestration hub to execute the entire reproduction process, from loading datasets and configuration, through prompt construction, model initialization, training, inference, to evaluation. Its core tasks involve setting up environment parameters, managing data flows, and coordinating modules for a reproducible pipeline aligned exactly with the AutoTimes methodology.

---

## 1. Initialization and Setup
- **Import dependencies**: modules for dataset loading, prompt building, model handling, training, evaluation, config parsing, and utility functions.
- **Parse command-line arguments**: accept optional config file path, logging options, or run modes.
- **Load configuration**: read `config.yaml` for all parameters.
- **Set seeds**: ensure reproducibility across NumPy, PyTorch, and any other randomness sources.
- **Device setup**: determine whether to run on CPU, GPU, or multiple GPUs, based on system availability (`torch.device`).

---

## 2. Dataset Loading
- **Paths from config**: fetch dataset file paths (`etth1_path`, `weather_path`, etc.).
- **Instantiate DatasetLoader**:
  - Call its `load_data()` method with each dataset path.
  - **In DatasetLoader**:
    - Read CSV files into pandas DataFrame or numpy array.
    - Normalize series data (e.g., min-max or z-score) as in the original paper.
    - Convert timestamp columns to textual format syntactically aligned with the model’s expectations.
    - Segment the series into non-overlapping windows of size `segment_size` (default 96, 192, etc.).
    - Store series, timestamps, and optional metadata (series name, frequency, etc.).
  - **Output**: structured list/dictionary of `TimeSeriesSample` objects.

- **Dataset split management**:
  - For evaluation, ensure proper training-validation-test splits respecting chronological order.
  - For in-context or zero-shot experiments, label datasets to include prompts or prompts plus series as needed.

---

## 3. Prompt Construction
- **Initialize PromptBuilder**:
  - Pass necessary model tokenizer (from transformers library) based on `pretrained_model_name`.
- **For each series sample**:
  - Use the `build_prompt()` method with:
    - Input series (e.g., last `L` steps for lookback)
    - Timestamps as array of textual date-times.
    - Selection strategy (`prompt_strategy`, e.g., `firstF`) and prompt length (`text_prompt_length`).
  - **In PromptBuilder**:
    - Convert series segments into tokens:
      - Use segmentation layer (`segment_size`) to split series into tokens.
      - Use the model tokenizer for textual prompts (convert textual timestamps into tokens).
    - Return prompt texts: list of prompt strings (e.g., “Historical series from date X...”).
    - Also, prepare combined token tensors: for embedded series and timestamps, ready for input.

- **Batch prompt tensors**:
  - For training/evaluation, batch prompts with appropriate padding/truncation.
  - Store prompt tensors separately or as part of a data batch object for efficient loading.

---

## 4. Model Initialization
- **Instantiate the Model**:
  - Load pre-trained language model (`pretrained_model_name`) using `transformers` library.
  - Set `freeze_backbone=True` to keep the backbone parameters frozen as per AutoTimes.
  - Initialize embedding layers:
    - Segment embedding layer (MLP) to project raw series segments into LLM's embedding space.
    - Textual timestamp embedding (if not precomputed, compute embedding of textual timestamp prompts).
  - Initialize projection head for decoding predicted tokens back into series segments.
- **Parameter Freezing**:
  - Freeze all backbone layers except `segment_embedding` and `projection_head` layers needed for training.

---

## 5. Training Module
- **Instantiate Trainer**:
  - Input: model object, training dataset samples, prompt builder, hyperparameters.
- **Training procedure**:
  - For each epoch:
    - For each batch:
      - Generate prompts:
        - Call `build_prompt()` for each series in batch.
        - Prepare input token sequences combining prompts + series tokens.
      - Forward pass:
        - Input token tensor into the model (with frozen backbone).
        - Compute output predicted tokens.
      - Loss calculation:
        - Use supervised MSE between predicted series segments and ground-truth segments.
        - Backpropagate only through trainable layers (`embedding` + `projection head`).
        - Apply gradient clipping (`gradient_clip_norm`) as needed.
    - Periodically save checkpoints (`save_checkpoint()`).

---

## 6. Inference and Prediction
- **Prepare for inference**:
  - Load trained model checkpoint.
  - Set model to evaluation mode.
- **Construct prompt sequences**:
  - For test series (long-term or in-context), generate prompt sequences:
    - Use the `build_prompt()` similar to training.
    - For in-context forecast, include the prompt series as extended context.
- **Generate predictions**:
  - Initialize with the last `L` observed series segment.
  - Iteratively:
    - Convert current input sequence to token tensor, including textual timestamp embeddings.
    - Forward through model: predict next segment tokens.
    - Decode tokens back to series (`decode_tokens()`).
    - Append predicted segment to input for next step.
    - Continue until total forecast horizon `F` is complete.
- **In-context prompts**:
  - For in-context experiments, incorporate selected prompt series (e.g., same series, historical series with periodicity) into the input sequence.
  - Ensure no data leakage: prompts come only from past or relevant prompt datasets.
  
---

## 7. Evaluation and Metrics
- **Compute metrics**:
  - SMAPE, MSE, MAE between predicted series and ground truths.
- **Additional metrics or visualization**:
  - Save predicted series plots for a subset for comparison.
  - Save metrics summaries into logs or report files.
- **Statistical robustness**:
  - Compute standard deviations over multiple random seed runs if applicable.
- **Record inference time and resource usage**:
  - Log GPU memory, runtime per batch/series.

---

## 8. Save and Finalize
- Save final trained model state, embedding layers, and projection head.
- Save generated predictions and metrics for reporting.
- Collect logs for reproducibility:
  - Paths, hyperparameters, seeds, model versions, dataset stats.
- **Optional**:
  - Generate visualizations (example series, prediction overlays).

---

## 9. Additional Recommendations
- **Error handling**:
  - Confirm dataset file availability.
  - Validate that tokenization and prompt construction matches the pretrained model's tokenizer.
- **Logging and monitoring**:
  - Use `tqdm` for progress bars.
  - Log hyperparameters, epoch metrics.
- **Reproducibility**:
  - Fix random seeds.
  - Save all configs, dataset splits.
  - Record hardware info (GPU type, memory).

---

# Summary
`main.py` performs a high-level orchestration:
- Parses configs, sets seeds.
- Loads datasets & splits.
- Builds prompts (series + timestamps).
- Initializes and loads the frozen LLM backbone + task-specific embedding layers.
- Conducts training (with only added layers).
- Performs iterative autoregressive inference with autoregressive generation.
- Evaluates predictions.
- Saves models, checkpoints, logs, plots.

This structured flow ensures exact replication of AutoTimes methodology and experimental results aligned with the paper's design, datasets, and hyperparameters.

---

If any detail remains ambiguous, further clarification on prompt strategies, dataset formats, or tokenization procedures should be obtained before detailed coding.

## model.py

**Logic Analysis for `model.py` — Implementation of the `Model` Class**

---

### **Objective of the Model Class**
The `Model` class encapsulates the core neural architecture required to adapt a pre-trained decoder-only language model (LLM) (e.g., LLaMA, GPT-2, OPT) for autoregressive time series forecasting. It must enable:
- Loading and using a pre-trained tokenizer and language model.
- Freezing the backbone (all parameters except specific trainable layers).
- Embedding segmented time series data and textual timestamps into the model's input space.
- Performing autoregressive next-token prediction for arbitrary length forecasting.
- Decoding predicted tokens into time series segments.

---

### **Major Components & Responsibilities**

1. **Pretrained Language Model Loading**
   - Load the decoder-only LM using the `transformers` library (`AutoModelForCausalLM` or similar).
   - Load the tokenizer compatible with the LM.
   - Maintain the model in evaluation mode (`model.eval()`) during inference.
   - Freeze backbone parameters (`requires_grad=False` for all except trainable layers).

2. **Embedding Layers**
   - **Series Segment Embedding (`SegmentEmbedding`)**:
     - Implemented as an MLP (two-layer), mapping raw segment data (`size = S`) into embedding space (`D`, e.g., 768).
     - Responsible for capturing the variation dynamics of series segments.
   - **Textual Timestamp Embedding (`TimestampEmbedding`)**:
     - Convert timestamps to a textual description (e.g., "YYYY/MM/DD HH:MM:SS").
     - Tokenize the timestamp text with the tokenizer.
     - Use the tokenizer's embedding layer to convert text tokens into embeddings.
     - Extract the embedding of the special token (e.g., `<EOS>`) to represent the entire timestamp.
   - **Input Embedding Composition (`E`)**:
     - For each segment token, concatenate/add the segment embedding and timestamp embedding.
     - The combined embedding (`E`) is used as input for the LM.

3. **Handling Special Tokens and Positional Embeddings**
   - Incorporate timestamp embeddings as position embeddings to encode temporal info explicitly.
   - Maintain alignment between timestamps and series segments.
   - Possibly add special tokens (e.g., `<EOS>`) at the end of prompt sequences if necessary.

4. **Autoregressive Prediction**
   - Implement a function (`predict_next_tokens`) that:
     - Takes current sequence of input token embeddings.
     - Feeds them to the LM in causal fashion.
     - Obtains logits for the next token.
     - Applies softmax (via `lm` outputs) to get token probabilities.
     - Selects top token probabilities, or uses sampling if stochasticity is needed.
     - Iteratively generate tokens for `F / S` steps to simulate multi-segment forecasting.
   - During inference:
     - Use the last generated tokens as input for subsequent prediction steps.
     - Convert generated tokens back to series segments.

5. **Decoding Tokens into Series Segments**
   - Implement `decode_tokens()`:
     - Map predicted token embeddings back through the projection head (MLP).
     - Obtain series segment prediction (`size = S`).
     - Compose full predicted series by concatenating these segments.

6. **Trainability & Parameter Freezing**
   - Load the pre-trained LM with all parameters frozen (`requires_grad=False`).
   - Initialize and train only:
     - The SegmentEmbedding layer.
     - The SegmentProjection layer (MLP).
   - During training, only these layers are updated:
     - Use standard backprop with MSE loss.
   
7. **Model Initialization & Setup**
    - Constructor (`__init__`) should:
      - Load the pre-trained model and tokenizer.
      - Instantiate embedding layers.
      - Set `requires_grad` appropriately.
      - Determine input dimension (`D`), segment size (`S`), and model config.
    - Provide methods:
      - `embed_segments(segments: torch.Tensor) -> torch.Tensor`
        - Input: raw segment data (`batch_size`, `S`)
        - Output: embedded segment vectors
      - `embed_timestamps(timestamps: list[str]) -> torch.Tensor`
        - Input: list of timestamp strings
        - Output: timestamp embeddings matching those in the input sequence
      - `predict_next_tokens(inputs: torch.Tensor) -> torch.Tensor`
        - Input: input sequence embeddings
        - Output: predicted next token embeddings or logits
      - `decode_tokens(tokens: torch.Tensor) -> torch.Tensor`
        - Converts predicted tokens back into time series segments.

---

### **Implementation Details & Considerations**

- **Model Loading**
   - Use `AutoModelForCausalLM.from_pretrained(pretrained_model_name)` with `torch.device` setup.
   - Use the tokenizer corresponding to the pre-trained LM.
   
- **Embedding Layers**
   - `SegmentEmbedding`: a `nn.Sequential` with two linear layers and activation (e.g., ReLU).
   - `TimestampEmbedding`: extract the last hidden state of the timestamp prompt, possibly only the `<EOS>` token embedding.
   
- **Embedding Combination**
   - Combine `SeriesSegmentEmbedding` and `TimestampEmbedding` via addition or concatenation (preferred is addition for simplicity).

- **Parameter Freezing**
   - After loading the LM, iterate over `model.parameters()` to set `requires_grad=False`.
   - Only `SegmentEmbedding` and `SegmentProjection` layers are trainable.

- **Prediction Routine**
   - For each iteration:
     - Input the current sequence embeddings.
     - Forward through the LM.
     - Extract logits for the last token position.
     - Obtain the next token embedding.
     - Project via `SegmentProjection`.
     - Append to sequence for iterative generation.
   
- **Decoding & Series Reconstruction**
   - Use the `decode_tokens()` method to convert token embeddings back into raw series data.
   - Handle padding or truncation according to model's max input length.
   
- **Handling Variable Lengths**
   - Input sequence length (`max_input_length`) is constrained by the pre-trained model.
   - Padding or truncation may be required.
   
- **Efficiency & Stability**
   - Use `with torch.no_grad()` during inference.
   - Precompute timestamp embeddings for known timestamps if possible.
   
- **Maintain compatibility with training and inference modes**
   - Switch between mode: `train()` and `eval()` as needed.
   
---

### **Summary Checklist for `model.py`**

- [ ] Load pre-trained decoder-only LM and tokenizer.
- [ ] Freeze the LM parameters.
- [ ] Implement an embedding layer for series segments (`SegmentEmbedding`).
- [ ] Implement text-to-embedding for timestamps utilizing the tokenizer.
- [ ] Combine series and timestamp embeddings for LM input.
- [ ] Implement autoregressive prediction of tokens:
  - Input current sequence.
  - Generate next token embedding/logits.
  - Select next token (e.g., argmax or sampling).
  - Append token for next iteration.
- [ ] Implement decoding of predicted tokens to raw series segments.
- [ ] Provide interface for:
  - Forward pass for training (given embedded inputs).
  - Inference routine for autoregressive generation.
- [ ] Support arbitrary forecast horizons via iterative token prediction.
- [ ] Properly document handling of special tokens and position embeddings.

---

This detailed analysis provides a complete blueprint for implementing the `model.py` **`Model` class**, ensuring that all stages—from model loading, embedding, autoregressive prediction, to decoding—are coherently designed to replicate AutoTimes methodology in the NeurIPS paper.



## prompt_builder.py

**Logic Analysis for `prompt_builder.py` — PromptBuilder Class**

---

### Purpose:
The `PromptBuilder` class encapsulates all functions necessary to construct textual prompts for the AutoTimes model, facilitating the conversion of numerical time series data and timestamps into sequences of tokens suitable as input to a frozen, pre-trained decoder-only language model (LLM). It supports multiple prompt strategies, handles tokenization, and ensures proper alignment between series segments and textual timestamp prompts.

---

### Core Responsibilities:
1. **Prompt Construction Methods:**
   - Assemble prompts based on different strategies (e.g., initial segments, recent data, random segments, or out-of-series prompts).
   - Incorporate textual representations of timestamps as position embeddings.
   - Maintain preparation of both series tokens (for numerical data) and textual timestamp tokens, aligning them correctly.

2. **Series-to-Token Conversion:**
   - Convert numerical series segments into token IDs through embedding or tokenization.
   - Use the tokenizer of the specific LLM (e.g., from the `transformers` library) to convert numerical prompts or textual timestamp prompts into token sequences compatible with the model.

3. **Timestamp Textualization and Embedding:**
   - Convert numerical timestamps (e.g., float seconds, datetime objects) into textual form ("YYYY/MM/DD HH:MM:SS").
   - For each timestamp, produce a textual prompt that can be tokenized.
   - Optionally, incorporate textual timestamp prompts as additional tokens or position embeddings.

4. **Prompt Strategy Selection & Dynamics:**
   - Implement different strategies such as:
     - **P.1:** Using the first F time points as prompt.
     - **P.2:** Using the first 2F points plus the lookback window (most relevant for in-context learning).
     - **P.3:** Using the last 2F points of the series.
     - **P.4:** Using out-of-series prompts (from unrelated series), to analyze prompt relevance.
   - Allow flexible configuration of prompt length, lookback length, and prompt strategy.

5. **Conversion of Series Data and Timestamps to Tokens:**
   - For each segment, generate a sequence of tokens representing the series data.
   - For each timestamp, generate a sequence of textual tokens representing date/time info.
   - Use the tokenizer's `encode` function with proper settings (e.g., add special tokens, padding, truncation).

6. **Input Interface & Output:**
   - Main method: `build_prompt(series, timestamps, strategy, prompt_length)`.
   - Inputs:
     - `series`: numpy array of raw series data.
     - `timestamps`: numpy array of datetime objects or numerical timestamps.
     - `strategy`: string corresponding to prompt selection method.
     - `prompt_length`: integer defining the number of tokens or segments in the prompt.
   - Output:
     - A list of prompt strings or a tensor of token IDs, depending on implementation stage.

7. **Tokenization & Embedding:**
   - Convert textual prompts to token IDs using the tokenizer.
   - Optionally, produce prompt embedding tensors for model input (if prompts are embedded separately), but typically, token IDs are passed through the tokenizer/embedding layer.

8. **Compatibility & Flexibility:**
   - Accept different LLM tokenizer configurations (e.g., GPT-2, LLaMA, OPT). The tokenizer instance should be passed or initialized during class instantiation.
   - Support dynamic prompt sizes, accommodate variable prompt strategies.

9. **Handling Special Tokens:**
   - Manage start/end tokens, such as `<EOS>`, `<PAD>`, `<BOS>`, as appropriate.
   - Define or assume special tokens for indicating prompt start/end if needed.

10. **Efficiency & Reusability:**
    - Precompute textual representations of timestamps where feasible.
    - Minimize repeated conversions by caching tokenized timestamp prompts.
    - Modularize methods for easy extension of new prompt strategies.

---

### Design Considerations:
- **Tokenizer Initialization:**
  - Require a tokenizer object (from `transformers`) upon class instantiation.
  - Ensure consistent tokenization between prompt construction and model input.
- **Prompt Strategy Methods:**
  - Implement as separate private methods `_get_prompt_strategy_X()` to modularize logic.
- **Series and Timestamp Processing:**
  - **Series Conversion:** Map each data point or segment into string representations (e.g., rounded floats).
  - **Timestamp Conversion:** Convert datetime objects or float timestamps into string form based on the template ("YYYY/MM/DD HH:MM:SS").
- **Prompt Construction Workflow:**
  - Based on the selected strategy, extract relevant segment(s) or series prompt(s) from the input.
  - Convert each into textual prompts.
  - Image or log-based series can be represented as sequence of strings or numerical tokens (future extension).
  - Include timestamp text prompts aligned with series segments.
- **Final Output Formatting:**
  - Assemble all prompts in sequential order.
  - Tokenize entire prompt into token IDs.
  - Return token sequences that can be fed into the language model as input.

---

### Implementation Outline:
```python
class PromptBuilder:
    def __init__(self, tokenizer, segment_size, prompt_strategy, prompt_length, timestamp_format=None):
        self.tokenizer = tokenizer
        self.segment_size = segment_size
        self.prompt_strategy = prompt_strategy
        self.prompt_length = prompt_length
        self.timestamp_format = timestamp_format or "%Y/%m/%d %H:%M:%S"  # default

    def build_prompt(self, series, timestamps, strategy=None, prompt_length=None):
        # Select the strategy
        strat = strategy or self.prompt_strategy
        prompt_len = prompt_length or self.prompt_length

        # Retrieve prompt segments based on strategy
        prompt_texts = self._select_prompt_segments(series, timestamps, strat, prompt_len)

        # Convert series segments to text/prompts
        series_prompts = [self._series_segment_to_text(s) for s in prompt_texts['series_segments']]
        # Convert timestamps to textual prompts
        timestamp_prompts = [self._timestamp_to_text(ts) for ts in prompt_texts['timestamps']]

        # Concatenate prompt strings
        full_prompt_text = self._assemble_prompt(series_prompts, timestamp_prompts)

        # Tokenize entire prompt
        token_ids = self.tokenizer.encode(full_prompt_text, add_special_tokens=True)

        return token_ids

    def _select_prompt_segments(self, series, timestamps, strategy, prompt_length):
        # Implement strategies:
        # e.g., 'firstF', 'lastF', 'recentSeries', 'random', 'out_series'
        # Returns dict with 'series_segments' and 'timestamps'
        pass

    def _series_segment_to_text(self, segment):
        # Convert numerical segment data to string (e.g., space-separated floats)
        return " ".join([f"{val:.3f}" for val in segment])

    def _timestamp_to_text(self, timestamp):
        # Format timestamp as string, e.g., '2016/07/05 00:00:00'
        if isinstance(timestamp, (float, int)):
            dt = datetime.datetime.fromtimestamp(timestamp)
        else:
            dt = timestamp
        return dt.strftime(self.timestamp_format)

    def _assemble_prompt(self, series_prompts, timestamp_prompts):
        # Assembles prompt string, e.g., with spacers or special tokens
        prompt = ""
        for s_text, t_text in zip(series_prompts, timestamp_prompts):
            prompt += f"Series: {s_text} Timestamp: {t_text}\n"
        return prompt
```

---

### Summary:
The `PromptBuilder` class performs the critical task of translating raw series data and timestamps into well-formed, tokenized prompts that respect the selected sampling strategy. It supports multiple strategies, ensures this data is properly textualized, synchronized, and tokenized, ready for input to a frozen LLM to enable autoregressive training or inference in AutoTimes.

---

Please specify any particular point you'd like to explore further, such as exact prompt strategies, timestamp formatting details, or tokenization techniques.

## trainer.py

# Logic Analysis for `trainer.py`

This file implements the `Trainer` class, which orchestrates the training, validation, testing, and inference phases for the AutoTimes model as specified in the paper and design documentation. The core responsibilities include model initialization, loading datasets, managing training iterations (only training the added embedding and projection layers while freezing the backbone), performing autoregressive prediction during inference, evaluating with specified metrics, and checkpoint management.

---

## 1. Initialization of Trainer

### Inputs:
- `model`: An instance of the `Model` class, which contains:
  - The pre-trained decoder-only LLM backbone (frozen).
  - Additional trainable layers: segmentation embedding (`SegmentEmbedding`) and segment projection head (`SegmentProjection`).
- `dataset`: List/iterator of `TimeSeriesSample` objects containing series data, timestamps, and metadata.
- `prompt_builder`: An object to create prompts (series + timestamps) suited to tasks.
- `hyperparameters`: Hyperparameters object/dict containing training config (`lookback_length`, `forecast_horizon`, etc.).
- `evaluation_metrics`: List of metrics (e.g., SMAPE, MAE, MSE).

### Process:
- Set device (GPU/CPU).
- Initialize optimizer (AdamW) to only update trainable layers (`SegmentEmbedding` and `SegmentProjection` parameters). The LLM backbone remains frozen.
- Set learning rate, weight decay, gradient clipping norm from config.
- Prepare data loaders for training, validation, testing.

---

## 2. Dataset Preparation

- Use `dataset_loader.py` to load all samples with proper preprocessing:
  - Normalize series if needed (consistent with paper).
  - Segment series into fixed-sized segments (`segment_size` = 96 or as specified).
  - Convert timestamps to textual prompts, then embed them using the LLM’s tokenizer (via `prompt_builder`).
  - Organize into batches for training and evaluation.

- For training:
  - Use a sliding window approach with lookback length (e.g., 672).
  - For each training sample, prepare:
    - Input prompt: series segments + textual timestamp prompts.
    - Target: next segment of size equal to the forecast horizon or smaller steps (iterative prediction).

---

## 3. Prompt Construction & Data Flow

- For each batch:
  - Use `prompt_builder.build_prompt()` to generate prompt strings:
    - Concatenate lookback series segments and, optionally, textual timestamp prompts according to strategy.
    - Convert series segments to token tensors using `convert_series_to_tokens()`.
    - Convert textual prompts (timestamps) to embeddings/tokens in alignment with the model.

- Data batch structure:
  - Series segments tensor: shape `(batch_size, lookback_length / segment_size, segment_size)`.
  - Timestamps textual prompts: list of strings, embedded into input tokens.
  - Combine embeddings with timestamp positional information to produce input tensor for LLM.

---

## 4. Forward Pass

### During training:
- Feed input sequence into the model:
  - The model’s forward pass propagates through frozen backbone, with only the trainable embedding and projection layers active.
- Obtain predicted next segment tokens:
  - `Model.embed_segments()` embeds lookback segments.
  - Positional embeddings from timestamp prompts are added.
  - Pass through frozen LLM layers to get output embeddings.
  - Plugins (segment projection head) decode embeddings to series space.
- Loss computation:
  - Calculate MSE loss between predicted segments and actual next segments.
  - This is a token-wise supervision for next token prediction.

### During inference:
- Autoregressively generate the forecast horizon:
  - Start from last known series segments (lookback sequence).
  - For each iteration:
    - Pass current sequence plus previous predictions into the model.
    - Predict next segment tokens.
    - Append predicted segment tokens to sequence.
  - Repeat until entire forecast horizon is generated.
- Convert tokens back into series data (via `decode_tokens()`).

---

## 5. Optimization Loop

- For each epoch:
  - Loop over batches:
    - Zero out gradients.
    - Forward pass:
      - Embed input series and prompts.
      - Compute predicted tokens.
    - Compute loss (supervised MSE for training).
    - Backpropagation:
      - Clip gradients (norm 1.0).
      - Update only trainable parameters (`SegmentEmbedding`, `SegmentProjection`).
  - Validation:
    - Run on validation set (no gradient).
    - Record metrics (e.g., SMAPE, MAE, MSE).
    - Save best checkpoint based on validation performance.

---

## 6. Validation & Testing

- Evaluate the trained model on validation data periodically.
- Use `evaluation.py` or internal functions:
  - Generate predictions via autoregressive inference.
  - Compute metrics.
  - Store metrics and optionally generate visualization.

---

## 7. Checkpoint Management

- Save model state dict, optimizer state, training iteration steps when validation improves.
- Save model at the end of best epoch.

---

## 8. Inference and Multi-horizon Prediction

- **Autoregressive Forecasting:**
  - Load last lookback window.
  - Iteratively generate segments until full horizon is reached.
  - Use previous predicted segments as inputs for subsequent prediction.
- **In-Context or Prompt-Enhanced Inference:**
  - Incorporate additional series prompts (if in-context forecasting), building prompts with relevant prior series.
  - Extend the input sequence accordingly during inference.

---

## 9. Additional Considerations

- **Handling Variable Input Lengths:**
  - The model can handle varying lookback lengths within the maximum (`max_input_length` = 672).
  - During training, pad or truncate sequences as needed.
- **Efficiency:**
  - Perform gradient accumulation if batch size is constrained.
  - Use `torch.cuda.amp` for mixed precision if supported.
- **Reproducibility:**
  - Set random seed at initialization.
  - Log hyperparameters, training and validation metrics.
- **Error Handling:**
  - Check for dataset anomalies (e.g., missing data).
  - Ensure sequence lengths are compatible with segmentation parameters.

---

## 10. Summary of Key Functions in `trainer.py`

- `__init__(…)`: set up model, datasets, optimizer, metrics.
- `_train_epoch()`: process one epoch, batching, forward pass, loss, backward, optimizer step.
- `_validate()`: run inference on validation set, compute metrics.
- `train()`: loop through epochs, call `_train_epoch()`, perform validation, manage checkpoints.
- `predict()`: perform autoregressive inference for the test set or specific horizon.
- `save_checkpoint()`, `load_checkpoint()`: manage model states.
- `evaluate()`: comprehensive evaluation, visualizations if needed.

---

This detailed analysis provides a clear roadmap to implement `trainer.py` aligned strictly with the AutoTimes methodology, ensuring fidelity to the experimental design and reproducibility based on the provided configuration and design design.

