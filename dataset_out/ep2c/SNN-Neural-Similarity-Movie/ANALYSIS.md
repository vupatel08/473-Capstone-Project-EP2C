# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

### Logic Analysis: dataset_loader.py

This module is responsible for loading, preprocessing, and organizing the neural response data from the Allen Brain Observatory dataset as well as stimuli movies used for both neural data comparison and model input. It lays the foundation for simulating neural responses and generating data for evaluation metrics such as TSRSA and regression analysis.

---

#### 1. **Primary Functionality Overview**

- Load neural responses (spike trains or PSTH data) for the 6 cortical regions.
- Normalize or preprocess neural data for consistency with the experimental measures.
- Load visual stimulus movies (Movie1 and Movie2) and manipulate them for experiments.
- Provide interfaces to extract neural responses aligned with stimulus frames.
- Support data splits for calculating neural ceilings and variability estimates.
- Enable data synchronization between neural responses and stimulus presentation.

---

#### 2. **Datasets and Data Sources**

- **Neural responses**:
  - From the Allen Brain Observatory Visual Coding dataset.
  - Recorded responses from six cortical regions with electrode (Neuropixels) data.
  - Each response per neuron per trial, per frame.
  - Responses: raw spike counts over time, to be converted to PSTH (per frame average spikes).

- **Stimuli movies**:
  - Movie1: 30s, 30Hz frame rate (900 frames, approx.).
  - Movie2: 120s, 30Hz (3600 frames, approx.).

---

#### 3. **Data Loading Procedures**

- **Neural data**:
  - Load from storage: assumed to be in structured formats (e.g., HDF5, NPY, CSV, or MATLAB files).
  - Load neural spike times or per-trial responses.
  - Convert spike times to PSTH:
    - Sum spikes within each frame (per frame bin).
    - Average over trials to get mean response for each neuron.
  - Filter neurons:
    - Exclude neurons with average firing rate < 0.5 spikes/sec.
    - Implementation:
      - Calculate mean spike rate per neuron: (sum of spikes / total duration).
      - Drop neurons below threshold.

- **Stimuli movies**:
  - Load from video files or image sequences stored locally.
  - Resize to specified dimensions (e.g., 224x224) for model input.
  - Ensure temporal order is preserved.
  - Store as numpy arrays or tensors.

---

#### 4. **Functionality Details**

**A. Neural Response Loading**

- Implement `load_neural_responses()`:
  - Parameter: dataset path, region identifiers.
  - Load neural spike data.
  - Compute PSTH:
    - Sum spikes across each frame interval.
    - Average across all trials per neuron.
  - Return:
    - A numpy ndarray of shape `(number of neurons, number of frames)` for each cortical region.
    - Also return a list of region labels or names.

**B. Data Preprocessing**

- Function to exclude neurons with low firing rate:
  - For each neuron: compute mean response over all frames.
  - Remove neurons with response below 0.5 spikes/sec.
  - Output filtered response matrix.

**C. Stimuli Movie Loading**

- Implement `load_stimuli()`:
  - Read movie files (`.mp4`, `.avi`, or image sequences).
  - Resize frames to `[224, 224]`.
  - For Movie1:
    - Convert to numpy array of shape `(num_frames, height, width, channels)`.
  - For Movie2:
    - Same as above.
- Maintain chronological order.
- Normalize pixel values as needed (e.g., [0,1] or mean/std).

**D. Stimuli Manipulation Functions**

- Implement functions for experimental conditions:
  - `shuffle_frames(movie_array, window_size)`:
    - Divide the sequence into windows of `window_size`.
    - Shuffle frames within each window.
    - Return the modified movie.
  - `replace_with_noise(movie_array, ratio)`:
    - For each proportion of total frames (ratio), replace selected frames with Gaussian noise images.
    - To generate noise images:
      - Use numpy with mean=0.5, std=0.5 (or as per the experiments).
      - Consistent size `[224, 224, 3]`.
  - Additional:
    - Load static noise images from a specified source if needed for further experiments.

**E. Data Splitting and Neural Ceiling Calculation**

- For neural ceiling estimates:
  - Split the neural response data into two halves (by trials or neurons).
  - Store the responses separately.
  - Compute the TSRSA between the two halves to estimate maximum achievable similarity.
  - Implement `compute_neural_ceiling()`:
    - Input: neural responses split into two subsets.
    - Output: scalar ceiling value.

---

#### 5. **File Format and Data Access**

- Neural responses:
  - Likely stored as `.npy`, `.mat`, or `.csv`.
  - Load with `numpy.load()`, `scipy.io.loadmat()`, or `pandas.read_csv()`.
- Stimuli movies:
  - Load via OpenCV (`cv2.VideoCapture`) or `imageio`.
  - Convert frames to numpy arrays.

---

#### 6. **Interfaces and Usage Pattern**

- Instantiate the DatasetLoader class with dataset path.
- On initialization:
  - Load neural data for all cortical regions.
  - Load stimulus movies and preprocessed responses.
- Methods:
  - `get_neural_responses(region_name)` returns responses for a specific cortical region.
  - `get_movie(condition)` returns movie array(s).
  - `get_responses_aligned_with_movie()` returns neural responses aligned with stimulus frames for TSRSA.
  - `apply_manipulations()` functions return manipulated movies for experiments.
  - `compute_ceiling()` returns neural ceiling scores for reference.

---

#### 7. **Summary & Remarks**

- Ensure that data loading functions are flexible with file paths or dataset versions.
- Maintain synchronous timing: neural responses aligned to frames at 30Hz.
- Consistent handling of data shapes: `(neurons, frames)` for responses, `(frames, H, W, C)` for stimulus.
- Implement robust filtering and normalization steps.
- Modular functions for manipulation to facilitate repeatability and testing.
- Prepare for experiments involving multiple stimulus versions, so parameterize window sizes, ratios, and noise types.

---

This comprehensive logic outline will guide the implementation of `dataset_loader.py`, ensuring accurate reproduction of the neural and stimulus data processing as per the paper’s experimental paradigm.

## evaluation.py

# evaluation.py - Logic Analysis

This module serves as the core component for computing the representational similarity metrics between model responses and neural data, performing manipulations of stimuli, estimating neural response ceilings, and performing regression analyses. It must provide all functions and classes needed for reproducible evaluation consistent with the methodology described in the paper.

## 1. Core Functionalities Overview

- **TSRSA computation**:
  - Input: Model responses and neural responses, both as time-series responses.
  - Outputs: Similarity score (Spearman correlation) reflecting how closely the model captures the cortical responses during movie stimuli.
- **Neural ceiling estimation**:
  - Input: Neural response data (population responses) across trials.
  - Outputs: Ceiling RSA score, used as an upper bound for model performance.
- **Regression analysis**:
  - Fits model responses to neural responses of individual neurons, computes \( R^2 \) for neuronal-level similarity.
- **Stimuli manipulations**:
  - Shuffle frames within windows (dynamic manipulation).
  - Replace frames with Gaussian noise or static natural images (static manipulation).
- **Visualization functions**:
  - Plot scores, curves over conditions, error bars.

## 2. Data Input and Output Formats

### a. Responses:

- Model responses and neural responses are represented as matrices:
  - Shape: \( N \times T \), where:
    - \( N \): Number of units/population neurons (model units or cortical neurons).
    - \( T \): Number of stimulus frames/time points.
- Each column: Population response for a specific stimulus frame (or time-point).
- Responses are numerical arrays, preferably numpy arrays for efficient computation.

### b. Neural Data:
- Extracted from PSTH: averages over trials, per neuron, per frame.
- Neural responses are stored in a numpy array format, with neuron index and frame indices.

### c. Stimuli:
- Movie stimuli: numpy arrays or images.
- Manipulated stimuli: same format, with functions to shuffle or replace frames.

## 3. Functions and Classes

### a. `compute_TSRSA`:
- Input:
  - `responses_model`: list or dict of numpy arrays, responses from different layers/regions.
  - `responses_neural`: numpy array of neural responses.
  - `layer_name`: optional, specify which layer's responses are used for comparison.
- Output:
  - Float value: The Spearman correlation score.
- Logic:
  - For both responses:
    - Create a concatenated vector of correlation similarities `s_t`:
      - For each time \( t \):
        - Calculate Pearson correlation between response \( r_t \) and subsequent responses \( r_{t+p} \).
      - Store all \( s_{t,p} \) into a large vector.
  - After constructing model and neural concatenated similarity vectors, compute Spearman correlation coefficient.

### b. `estimate_neural_ceiling`:
- Input:
  - Neural response data across trials for each neuron.
- Output:
  - Scalar ceiling score.
- Logic:
  - Split data into two halves.
  - For each half, generate a response matrix.
  - Compute TSRSA between halves.
  - Return average similarity; this reflects maximum achievable correlation.

### c. `regression_score`:
- Input:
  - Neural responses of individual neurons.
  - Model responses (layer-wise or concatenated).
- Output:
  - \( R^2 \)-score for each neuron.
- Logic:
  - For each neuron:
    - Fit a linear regression model using model responses as predictor variables.
    - Predict neural response.
    - Compute \( R^2 \).
  - Aggregate results (mean \( R^2 \), or distributions).

### d. `shuffle_frames_within_windows`:
- Input:
  - Movie array: shape `[frames, height, width, channels]`.
  - Window size in frames.
- Output:
  - Shuffled movie array, with frames permuted within each window.
- Logic:
  - For each window:
    - Randomly permute frames within window.
    - Keep the window collection intact.

### e. `replace_frames_with_noise`:
- Input:
  - Movie array: shape `[frames, height, width, channels]`.
  - Replacement ratio.
  - Noise type: Gaussian.
- Output:
  - Modified movie array with certain frames replaced.
- Logic:
  - Select frames at random based on ratio.
  - Generate Gaussian noise images.
  - Replace selected frames.

### f. `visualize_scores`:
- Plots scores across manipulation conditions.
- Plots error bars with confidence intervals.
- Uses `matplotlib`.

## 4. Implementation Details

### a. Correlation measures:
- Use `scikit-learn`:
  - `PearsonCorrcoef` for similarity vectors.
  - `spearmanr` for comparing similarity vectors.

### b. Handling time series:
- For each `t`:
  - Compute Pearson correlation across model/neural response vectors.
  - Use vectorized `np.corrcoef()` or `scipy.stats.pearsonr` for small pairs.
- Process response vectors efficiently:

```python
for t in range(T):
    for p in range(1, T - t):
        s_t_p = pearson_correlation(response_t, response_{t+p})
        append to similarity vector
```

### c. Error Bars:
- Confidence intervals over repeated experiments (trials).
- Use bootstrap or standard error estimation with multiple runs (if applicable).

### d. Data management:
- Responses stored in numpy arrays.
- Responses and stimuli manipulations stored similarly, ensuring traceability.
- All functions should accept parameters for seed control for reproducibility.

## 5. Reproducibility and Robustness

- Maintain consistent random seed across functions.
- Document the number of repetitions/trials used in neural ceiling and statistical analysis.
- Save intermediate data where necessary for debugging.

## 6. Summary of Critical Functions

| Function name | Purpose | Input/Output | Notes |
|-----------------|---------|--------------|--------|
| `calculate_similarity_vector` | Generate similarity vector for a given response matrix | responses: np.ndarray; output: np.ndarray | Computes Pearson correlations over time |
| `compute_spearman_score` | Obtain TSRSA score | similarity vectors: np.ndarray | computes scipy.stats.spearmanr |
| `estimate_ceiling_score` | Compute neural ceiling | neural data matrix | split neural data, compute pairwise correlations |
| `fit_neuron_regression` | Fit responses per neuron | neural response vector, model response vector | uses sklearn linear regression |
| `manipulate_movie_shuffle` | Shuffle frames within windows | movie array | produce dynamic variation experiments |
| `manipulate_movie_noise` | Replace frames with Gaussian noise | movie array | produce static variation experiments |

## 7. Final Notes

- All functions must include detailed docstrings and error handling.
- Ensure parameter inputs are flexible (e.g., window sizes, ratios).
- Provide options for verbose logging for debugging.
- Design with modularity so responses from different layers/regions can be processed seamlessly.
- Maintain consistent units and normalization, e.g., response normalization, for fair comparisons.
- The functions should be compatible with batch processing if needed, though typically responses are processed per stimulus.

This comprehensive logic analysis guides the detailed implementation of `evaluation.py`, ensuring it supports precise, reproducible, and biologically faithful analysis as required.

## feedback_module.py

**Feedback Module Logic Analysis (feedback_module.py)**

---

### Purpose:

Implement the feedback recurrent module that models long-range corticocortical feedback in the LoRaFB-SNet. This module processes higher-region responses, and projects feedback signals back to lower-level layers (or regions), contributing recurrently to the network's dynamics. It should be compatible with the overall model pipeline, integrate seamlessly with the backbone residual convolutional spiking network, and support training via surrogate gradient methods.

---

### Core Requirements & Principles:

1. **Input & Output:**
   - Receives the responses (features) from a higher cortical region or layer, represented as a tensor.
   - Produces feedback signals to be integrated into lower or preceding layers for subsequent processing.
   
2. **Processing:**
   - The feedback response is processed via learned or fixed weights.
   - The feedback can be recurrent (i.e., the output at time t can loop back at subsequent time steps, modeling long-range corticocortical feedback).
   - The feedback signal may be delayed by a certain number of milliseconds (configured as `feedback_delay`).

3. **Recurrent Architecture:**
   - The module will contain a learnable weight matrix (or matrices) for projecting higher-level responses back.
   - It may include a feedback recurrence, where the feedback at time t depends on responses at previous steps.

4. **Parameters & Configuration:**
   - Feedback connection weights: derived from the main system configuration (`feedback_connection_type: recurrent`).
   - Feedback delay: specified in the configuration (`feedback_delay: 2` ms or steps).
   - Feedback strength: the scalar multiplicative factor (`feedback_strength: 1.0`).
   - The module should account for variable batch sizes and feature dimensions.

5. **Implementation:**
   - Should be a `torch.nn.Module`.
   - Supports batch processing and variable feature dimensions.
   - Uses differentiable operations compatible with surrogate gradient training.
   
6. **Integration:**
   - The module is invoked at each time step within the main network's forward pass.
   - It provides feedback signals that modify or influence the membrane potential or input current of lower or earlier layers.
   - Maintains internal state if recurrent feedback over multiple time steps.

7. **Design Constraints & Details:**
   - It needs to initialize learnable weights (e.g., linear layer or matrix).
   - The `feedback_delay` may implement a buffer or hold previous responses for delay.
   - The module should be compatible with the overall system, supporting backpropagation through time (BPTT).

---

### Detailed Functional Breakdown:

#### 1. Initialization:

- Define parameters:
  - `input_dim`: dimension of higher region response features.
  - `feedback_dim`: dimension of feedback projection (likely same as input_dim for simplicity).
  - `weight`: trainable weight matrix of size `(feedback_dim, input_dim)`.
  - `bias` (optional): bias term for projection.
  - Other parameters dictated by configuration:
    - `feedback_strength` (scaling factor).
    - `feedback_delay` (buffer size for delaying responses).

- If the feedback weights are to be learned, initialize as `torch.nn.Linear`.

#### 2. Forward Pass:

- Inputs:
  - `higher_response`: tensor of shape `[batch_size, response_dim]`, representing the current higher-region response at time t.
  - `stored_responses`: buffer or queue of previous responses, to handle delay.

- Processing:
  - If `feedback_delay` > 0:
    - Append current `higher_response` to a buffer.
    - Retrieve the response from `feedback_delay` steps ago.
    - If buffer is not full, use zeros or hold initial states.
  - Project higher response:
    - `feedback_signal = weight(higher_response)` (or linear layer).
  - Apply feedback strength:
    - `feedback_signal = feedback_strength * feedback_signal`.
  - Return `feedback_signal` as the feedback input to lower layers.

- Recurrent Dynamics:
  - If feedback is recurrent:
    - Optionally, incorporate feedback from previous feedback signals.
    - Support multiple time steps or iterations if needed.

#### 3. State Maintenance:

- During training, the internal buffers (for delay handling) should be stored as buffers or persistent states.
- Reset states at the beginning of each new stimulus/fixed-trial segment.

#### 4. Utility Functions & Additional Features:

- Method to reset buffers/state at stimulus onset.
- Support for different feedback connection types, e.g., fixed weights or learned weights.
- Optionally, include regularization or normalization if specified.

---

### Summary of Key Implementation Aspects:

| Aspect | Details |
| -------- | -------- |
| Class Name | `FeedbackModule` |
| Parent | `torch.nn.Module` |
| Constructor inputs | config parameters (e.g., `feedback_strength`, `feedback_delay`, `input_dim`, `feedback_dim`, `weight`, etc.) |
| Parameters | trainable weight matrix (e.g., `self.fc`), buffering for delay, optional bias |
| Forward | takes higher-region response, returns feedback signal considering delay and strength |
| Buffering | implement queue or buffer for delayed responses |
| Integration | designed to be called per time step in main network loop |

---

### Final Remarks:

This module must be carefully integrated within the main model execution flow, maintaining proper temporal alignment, handling batch processing, and supporting end-to-end training via surrogate gradients. Proper state-resetting and buffering ensure that feedback signals reflect appropriately delayed and recurrent responses, enabling the model to emulate cortical feedback dynamics as described in the paper.

## main.py

# Logic Analysis for main.py

## Purpose
`main.py` serves as the central orchestration script for the entire experimental pipeline. Its function is to initialize datasets, models, trainers, and evaluation modules based on the configurations, execute training/pretraining, load neural response data, simulate stimuli responses through the models, perform manipulations and analyses, and finally, save and visualize results to quantify the model's alignment with neural data in terms of static and dynamic representations.

---

## Major Components and Workflow

### 1. Setup and Configuration
- **Import Modules and Dependencies**
  - Load core libraries: `numpy`, `torch`, `spikingjelly`, `scikit-learn`, `matplotlib`, and custom modules (`dataset_loader.py`, `model.py`, etc.).
- **Read Configuration**
  - Parse `config.yaml` to extract parameters, including training hyperparameters, dataset paths, model specs, evaluation types, and manipulation parameters.
- **Set Random Seeds**
  - Use `system['seed']` for reproducibility across numpy, torch, and other relevant processes.
- **Initialize Device**
  - Set `CUDA` or `cpu`, as per `system['device']`.

---

### 2. Data Loading
- **Neural Data**
  - Instantiate `DatasetLoader` to load neural responses:
    - Paths to Allen Brain Observatory data.
    - Extract PSTH-processed responses aligned to stimulus frames.
    - Obtain neural responses for Movie1 and Movie2 (responses per cortical region).
- **Stimuli Movies**
  - Load stimulus movies (Movie1 and Movie2) aligned with neural stimuli:
    - Store as list of numpy arrays or tensors with shape `[num_frames, height, width, channels]`.
    - Preprocessing: resize to `[224, 224]`, normalize if needed.
- **Response/Stimuli Data Structures**
  - Responses: numpy arrays or tensors, stored per region/layer, per stimulus.
  - Stimuli: sequences of images.

---

### 3. Model Initialization
- **Construct the Model Instance (`SpikingResNet`)**
  - Instantiate with parameters:
    - Input channels: 3
    - Number of regions: 6
    - Residual blocks: 3
    - Layers per region.
    - Feedback mode: recurrent.
    - Feedback strength and delay: extract from `training` and `system`.
- **Load Pretrained Weights**
  - Load trained weights for UCF101 and/or ImageNet models:
    - Use saved checkpoint files (paths can be specified or default).
- **Feedback Modules**
  - Instantiate feedback modules (`FeedbackModule`) with specified parameters.
  - Integrate into the main network architecture.
- **Loss Function and Optimizer**
  - Set optimizer: Adam as per `training['optimizer']`.
  - Learning rate, weight decay, etc., as per config.
  - Surrogate gradient method: inverse tangent, with `surrogate_alpha`.
- **Surrogate Gradient Setup**
  - Use inverse tangent approximation for backprop through spikes.

---

### 4. Training/Pretraining
- **Decision Node**:
  - Determine training target:
    - On UCF101: action recognition task.
    - On ImageNet: object recognition task.
- **Training Loop**:
  - For each epoch up to `system['max_epochs']` or specified:
    - Load training batches.
    - For each batch:
      - Feed sequences (video frames or repeated images).
      - Run forward pass with temporal simulation (T=16 or T=4).
      - Obtain spike responses; generate output predictions.
      - Calculate loss (classification) with surrogate gradient.
      - Backpropagate and update parameters.
    - Schedule learning rate decay.
    - Save model checkpoints periodically.
- **Model Saving**:
  - Save the trained weights/backbone with feedback modules.

---

### 5. Response Extraction from Stimuli
- **Simulation with Movie Stimuli**
  - For each stimulus (Movie1, Movie2):
    - Prepare input sequence as per training type:
      - Continuous frame sequence (for UCF101 pretraining)
      - Static repeated images (for ImageNet pretraining)
    - Run forward pass through the trained model:
      - Over time steps matching stimulus (e.g., 16 frames for UCF101).
      - Collect population responses/activations per cortical region/layer.
    - Store responses in structured format:
      - Responses per region/layer as arrays: shape `[neurons, time]`.

---

### 6. Stimuli Manipulation Experiments
- **Dynamic (Temporal) Manipulation**
  - Use `manipulations.shuffle_frames()`:
    - Input the original movie.
    - For each window size (e.g., 5, 10, 20, 50 frames):
      - Shuffle frames within windows.
      - Generate manipulated stimulus sequence.
- **Static (Texture) Manipulation**
  - Use `manipulations.replace_with_noise()`
    - Randomly select frames to replace with Gaussian noise images.
    - Ratios: 0.25, 0.5, 0.75, 1.0.
  - Also, replace some frames with static natural images if applicable.
- **Implications**
  - Run the same simulation pipeline:
    - Feed manipulated movies into the trained model.
    - Extract responses as before.
- **Store manipulated responses**:
  - Responses per manipulation condition for subsequent similarity analysis.

---

### 7. Similarity Computations
- **Compute TSRSA**
  - For each model layer and cortical region:
    - Extract response matrices (`response_layer_region`) and response matrices from neural data.
    - Implement `evaluation.compute_TSRSA()`:
      - Calculate similarity vectors:
        - For each time t, correlate (Pearson) responses to responses at t + p.
        - Concatenate vectors into a comprehensive similarity vector.
      - Compute Spearman rank correlation between model and brain similarity vectors.
  - Save similarity scores for:
    - Original stimuli.
    - Manipulated stimuli (disrupted temporal/static info).
- **Neural Ceiling Estimation**
  - Split neural data over trials.
  - Compute pairwise similarity between splits.
  - Use as reference to evaluate model performance.

---

### 8. Regression Analysis
- **Fit model responses to individual neuron responses**:
  - Use linear regression between model population responses and neural responses.
  - Calculate \( R^{2} \) as measure of fit.
- **Compare across models** for static/dynamic conditions.

---

### 9. Results Visualization and Storage
- Generate plots:
  - TSRSA scores (original and manipulated).
  - Drop rates vs chaos/replace ratio.
  - Model vs neural ceiling comparison.
  - Response examples of neural firing profiles.
- Save figures and numerical scores.
- Optionally, save model weights and responses for further inspection.

---

## Additional Considerations
- **Ablation experiments**:
  - Initialize models without feedback or without spike neurons.
  - Run identical simulations.
- **Systematic variation of parameters**:
  - Feedback delay, feedback strength.
  - Number of residual blocks.
  - Surrogate gradient parameters.
- **Output Storage**:
  - Save all response matrices, TSRSA scores, and figures to designated directories.
- **Error handling**:
  - Confirm data paths, sample sizes, and model convergence.
  - Log progress for each major step for reproducibility.

---

## Summary
`main.py` will implement a comprehensive pipeline:
- Import dependencies and configurations.
- Load neural and stimulus data.
- Instantiate, train, and load pretrained models with feedback and spike mechanisms.
- Perform stimulus simulations to generate neural-like responses.
- Apply stimulus manipulations.
- Calculate visual similarity metrics, analyzing the influence of static/dynamic stimulus features.
- Visualize and save results.

All operations are governed by parameters in `config.yaml`. The script should modularize components for clarity and reproducibility, emphasizing precise control over each step.

---

This detailed logic analysis ensures that all critical steps and design details from the paper are respected, forming a robust foundation for subsequent programming.

## manipulations.py

{
  "manipulations.py": "The manipulations.py module is responsible for implementing functions that modify stimuli movies to assess the sensitivity of model and neural representations to dynamic and static information changes, following the experimental procedures outlined in the paper.\n\n### Functions to Implement:\n\n1. shuffle_frames(movie: np.ndarray, window_size: int) -> np.ndarray\n   - Input:\n     - movie: 3D numpy array [num_frames, height, width, channels], representing the original stimulus movie.\n     - window_size: integer, number of contiguous frames per window within which frames will be shuffled.\n   - Output:\n     - shuffled_movie: same shape as input, with frames within each window randomly permuted.\   - Process:\n     - Divide the total frames into segments/windows of size 'window_size'.\n     - For each window, generate a random permutation of frames within that window.\n     - Reassemble the movie with frames rearranged accordingly.\n     - Special care for the last window if total frames are not divisible by window_size: handle partial window.\n   - Details:\n     - Use numpy slicing for segment extraction.\n     - Use numpy.random.permutation to generate a permutation order.\n     - Maintain the original order for frames outside the window.\n\n\n2. replace_frames_with_noise(movie: np.ndarray, ratio: float, noise_type: str='Gaussian') -> np.ndarray\n   - Input:\n     - movie: 3D numpy array as above.\n     - ratio: float, proportion of total frames to replace with noise, e.g., 0.25, 0.5, etc.\n     - noise_type: string, specify 'Gaussian' or potentially other types.\n   - Output:\n     - noisy_movie: same shape as input, with a subset of frames replaced.\n   - Process:\n     - Determine number of frames to replace: num_replace = int(ratio * total_frames).\n     - Divide the movie into 'windows' similar to the paper (e.g., equal-sized segments).\n     - For each selected window, randomly pick one frame to replace.\n     - Generate a noise image per replaced frame:\n       - For 'Gaussian', create an array of shape [height, width, channels] with values sampled from a Gaussian distribution (mean=0, std=1 or scaled appropriately to match input range). \n       - Ensure the noise is visually sufficiently distinct.\n     - Replace the selected frames with generated noise images.\n     - Maintain the overall temporal order of the remaining frames.\n   - Details:\n     - Use numpy for noise generation.\n     - To avoid excessive replacement, enforce that only one frame per window is replaced.\n     - The number of windows can match the number of replacements or be proportional, based on the ratio.\n\n3. generate_static_noise_image(size: Tuple[int, int], channels: int) -> np.ndarray\n   - Input:\n     - size: e.g., [224, 224]\n     - channels: e.g., 3 for RGB.\n   - Output:\n     - noise_image: 2D or 3D array with static noise.\n   - Process:\n     - For Gaussian noise, generate values with numpy.random.normal with desired mean and std, normalized to typical input range [0, 1] or [0, 255], based on input format.\n     - For more visual diversity, consider using static natural images if available.\n   - Usage:\n     - Used in replace_frames_with_noise for creating noise images.\n\n### Additional Considerations:\n- Data Types:\n  - Ensure that input movies are numpy arrays with appropriate dtype, e.g., float32 or uint8.\n  - Outputs should match input data type, scaled appropriately.\n- Validations:\n  - Check window sizes do not exceed total number of frames.\n  - Clip or normalize noise images if required to match input range.\n- Randomness:\n  - For reproducibility, seed numpy random generator with a fixed seed if necessary.\n- Compatibility:\n  - Functions should accept and return numpy arrays with consistent dimensions.\n- Integration:\n  - These functions will be called within main experiment scripts to modify stimuli before passing to the model.\n\n### Summary:\nThe module manipulates the stimulus movie along two axes:\n- Temporal structure: through frame shuffling within windows.\n- Static texture: via replacing selected frames with noise images.\nImplementations must adhere to the parameters specified in 'config.yaml' for window sizes, ratios, and noise types, to ensure experimental consistency and reproducibility."
}

## model.py

# Logic Analysis for model.py: Spiking ResNet with Feedback Modules and Surrogate Gradient

---

## Purpose:
Implement the core neural network architecture—**SpikingResNet**—that models cortical visual regions with biological plausibility, incorporating residual convolutional blocks, spiking neuron dynamics, feedback modules, and recurrent feedback connections. This module facilitates forward inference, feature extraction, and supports backpropagation via surrogate gradients.

---

## 1. Class and Function Definitions

### a. **SurrogateGradient (inverse tangent):**
- Implements the inverse tangent to approximate the derivative of the neuron activation.
- Ensure differentiability for backpropagation.
- Used during the backward pass in training.

### b. **LIFNeuron (or integrated within spiking neuron layers):**
- Encapsulates the Leaky Integrate-and-Fire model:
  - State variables: membrane potential `V_t`.
  - Dynamics:
    - Compute `H_t = V_{t-1} + (1/τ) * (X_t - (V_{t-1} - V_reset))`
    - Generate spike: `S_t = Theta(H_t - V_thresh)`
    - Reset `V_t = V_reset` if spike occurs.
- Parameters:
  - `τ` (membrane time constant) from config.
  - Threshold (`V_thresh`) and reset voltage (`V_reset`) from config.
- Note: Implement as a custom torch module or function with surrogate gradient for the non-differentiable threshold.

### c. **ResidualBlock (Residual convolution with spiking neurons):**
- Input: Feature map tensor.
- Components:
  - Conv layer (2D conv with specific channels and kernel size).
  - Batch Normalization.
  - Spiking neuron (LIF or surrogate).
  - Skip connection: adds input to output.
- Function:
  - Forward through convolution → BN → spiking neuron.
  - Add skip connection.
  - Maintain temporal states if needed for recurrent connections.

### d. **FeedbackModule (Recurrent Feedback Module):**
- Purpose:
  - Process higher-region responses (from later cortical areas).
  - Project feedback signals back to lower layers or regions.
- Implementation:
  - Use learnable weight matrices (e.g., linear or convolutional).
  - Can include delay (feedback delay from config).
  - Recurrent connections modeled explicitly as part of forward pass.
- Inputs:
  - Higher-level response tensor.
- Outputs:
  - Feedback signal tensor (to modulate lower layers).

### e. **ResidualConvSpikingNet (Main Class):**
- Composition:
  - Multiple residual blocks organized into layers per cortical region.
  - Framework for multiple regions: store per-region layers.
  - Feedback modules integrated at appropriate layers.
  - Recurrent feedback pathways connecting regions.
- Initialization:
  - Based on configuration parameters.
  - Initialize weights for convolutional residual blocks.
  - Instantiate feedback modules.

---

## 2. Data Flow and Recurrent Dynamics

### a. **Input Processing:**
- Input: sequence of movie frames (`[batch_size, channels, height, width]`) across time steps or a single frame repeated `T` times.
- Action:
  - For each time step `t`:
    - Feed input frame into the backbone network.
    - Pass through residual blocks with spiking neurons.
    - Integrate feedback signals (from feedback modules).
  - Maintain membrane potential states across time.

### b. **Temporal Simulation:**
- For each time step in sequence:
  - Update membrane potentials `V_t` based on prior states.
  - Generate spikes `S_t`.
  - Store spike outputs and membrane states for feature extraction.
- For the feedback:
  - At each time step, retrieve higher-level responses.
  - Pass through feedback modules.
  - Inject feedback into lower layers' membrane potentials or inputs.

### c. **Feedback Integration:**
- Feedback signals are generated from responses of cortical regions (higher layers) at time t.
- Recurrently routed to earlier layers or regions in subsequent time steps.
- Feedback strength modulated as per `config.system.feedback_strength`.
- Recurrent cycle over `feedback_delay` steps:
  - Delayed feedback is incorporated by temporal buffers or explicit delay parameters.
- Overall:
  - This process creates a loop of information flow mimicking top-down cortical feedback.

---

## 3. Surrogate Gradient and Backpropagation

- Given the non-differentiability of spike function `Theta()`, surrogate gradient is used:
  - During forward: binary spike (0/1).
  - During backward: approximate derivative using inverse tangent function.
- Implementation points:
  - Wrap the spike activation with surrogate gradient function.
  - Ensure the computational graph is correctly constructed to propagate errors backward through the membrane potential dynamics.
- The surrogate function is parameterized by `alpha` (from config).

---

## 4. Handling Residual Connections

- Residual modules:
  - Main pathway: convolution → BN → spiking neuron.
  - Skip connection: input added to output post-activation.
- Consistency:
  - Maintain membrane states consistently across residual layers.
  - Optionally, reset membrane voltages at layer boundaries per batch/sequence.

---

## 5. Feedback Modules

- **Design specifics:**
  - Feedback weights are learned or fixed (configurable).
  - Use convolutional or linear layers, depending on spatial dimensions.
  - Incorporate delay (feedback_delay) in the feedback pathway.
- **Handling recurrent feedback:**
  - For each epoch, store previous higher-stage responses.
  - In forward pass:
    - Compute feedback at current time `t`.
    - Inject feedback signal (weighted and delayed) into the lower layers.
- **Recurrent feedback cycle:**
  - Can be implemented as an explicit RNN-like loop or an iterative process within each time step.

---

## 6. Implementation Details

### a. Initialization:
- Weights:
  - Convolutional residual blocks: Xavier or He initialization.
  - Feedback modules: randomly initialized, or set to identity if fixed.
- Membrane potentials:
  - Initialize to `V_reset`.
  - Track membrane potentials as buffers during forward pass.
  
### b. Forward pass:
- Loop over time steps:
  - For each region:
    - Compute features with residual blocks.
    - Update membrane potential with LIF equations.
    - Generate spikes.
    - Send responses to feedback modules.
  - Feedback responses are processed and injected into lower regions or earlier layers.
- Store layer-wise responses per time step for TSRSA.

### c. Outputs:
- Response matrices: per region/layer, across time steps, containing neuron responses.
- Spike sequences:
  - For visual similarity analysis and pretraining, save spike trains or spike counts.

---

## 7. Integration with the Rest of the Pipeline
- Ensure that:
  - Model loading from pretrained weights.
  - Compatibility with the data loader.
  - Facilitates gradient-based training with surrogate gradients.
  - Responses can be extracted and formatted for TSRSA and regression.
  - Feedback modules are properly trained/end-to-end.

---

## 8. Summary of Key Parameters (from config.yaml):
- `membrane_tau=2`
- `neuron_threshold=1.0`
- `reset_voltage=0.0`
- Feedback:
  - `feedback_strength=1.0`
  - `feedback_delay=2`
- Surrogate gradient `alpha=1.0`
- Residual block counts and channels:
  - 3 residual stages with 2 blocks each.
- Input channels=3.
- Layers with spiking neuron modules.

---

This detailed logic covers all core components, data flows, and training considerations necessary for implementing `model.py` consistent with the described architecture, methodology, and biological plausibility criteria.

## trainer.py

# Logic Analysis: trainer.py

This script implements the core training logic for pretraining the models (LoRaFB-SNet, SEW-ResNet, and CORnet) on UCF101 and ImageNet datasets using surrogate gradient-based learning. It also manages hyperparameters, learning schedule, checkpointing, and the use of appropriate data loaders. Below is a detailed breakdown of the envisioned implementation components and flow, aligned with the paper's methodology, configuration, and experimental setup.

---

## 1. Initialization and Setup

**a. Imports and Dependencies:**
- Import necessary modules: torch, torch.nn, torch.optim, appropriate data loading libraries.
- Import surrogate gradient functions (inverse tangent) from utility modules or define inline.
- Import model definitions for the backbone (ResidualConvSPikeNet), feedback modules, and any auxiliary classes.
- Set random seed for reproducibility (`system.seed` from YAML).

**b. Configuration Parameters:**
- Read parameters from the `config.yaml`:
  - Training: learning rate, batch size, epochs, optimizer, surrogate gradient settings, weight decay, initial LR, decay steps & rate, simulation time.
  - Model: backbone type, residual blocks, feedback structure, input channels, feature channels.
  - Dataset: dataset paths, frame length (for UCF101), repeats (for ImageNet), split ratios.
  - Feedback: feedback strength, delay.
  - System: device, max epochs, seed.

**c. Device Setup:**
- Initialize device (cuda or cpu) accordingly.
- Set seed for PyTorch and CUDA.

**d. Data Loaders:**
- Instantiate data loaders for UCF101 and ImageNet:
  - Use torchvision or custom loaders to load appropriately resized images/frames.
  - For both datasets, implement temporal windowing and batching.
  - Implement shuffling and augmentation as needed.
- Implement train/validation split based on thresholds (80% train, 20% validation).

**e. Model Instantiation:**
- Create the residual convolutional spike network (`ResidualConvSPikeNet`), configured with residual blocks, input/output channels, and normalization layers.
- For feedback:
  - Instantiate feedback modules (`FeedbackModule`) with learned weights.
  - Integrate feedback as recurrent connections within the model, ensuring proper end-to-end differentiability.
- Initialize model weights with appropriate schemes.

**f. Surrogate Gradient Function:**
- Implement or import the inverse tangent surrogate gradient:
  - For each backprop step, compute the approximate gradient of the non-differentiable spike function.
  - Use a custom backward function or hook with PyTorch's autograd mechanism.

**g. Optimizer and Scheduler:**
- Set up optimizer (Adam) with the specified learning rate.
- Define learning rate scheduler:
  - Step decay at `lr_decay_steps`, decaying by `lr_decay_rate` (0.1).
  - Or a cosine annealing schedule if preferred.
- Apply weight decay in optimizer.

---

## 2. Training Loop

**a. Loop across epochs (up to 320):**
- For each epoch:
  - Set model to training mode.
  - Initialize running metrics: total loss, accuracy metrics if applicable.
  - Loop over batches from the data loader:
    - Prepare batch data:
      - For UCF101: batch of sequences, shape `[batch_size, T, C, H, W]`.
      - For ImageNet: static images repeated T=4 times, shape `[batch_size, T, C, H, W]`.
    - Reset hidden states/membrane potentials of spiking neurons at batch start.
    - For each time step in sequence:
      - Extract frames/tensors slice.
      - Forward pass through the model:
        - Inputs: current frame (or repeated frame for ImageNet).
        - Feed feedback signals if applicable.
        - Capture spike outputs and membrane potentials.
    - Aggregate the population responses over the sequence or select responses from last layer.
    - Compute output logits/probabilities for classification tasks.
    - Loss calculation:
      - Use cross-entropy against class labels.
      - Backpropagation using surrogate gradient:
        - Implement custom backward functions if necessary.
        - Compute surrogate gradient for the spike nonlinearity.
    - Perform optimizer step.
    - Track loss and accuracy metrics.

**b. Learning Rate Decay:**
- Update LR according to scheduler schedule per epoch or iteration.

**c. Checkpointing:**
- Save model weights periodically (e.g., every N epochs) or when validation loss improves.
- Save training logs.

---

## 3. Validation and Logging

- After each epoch:
  - Evaluate on validation set:
    - Forward pass without gradient calculation.
    - Record validation loss and accuracy.
- Optionally, compute and record the surrogate gradient-specific metrics or neuron firing statistics.
- Log progress: epoch, loss, accuracy, learning rate.

---

## 4. Saving and Loading

- Save final model state_dict at end of training.
- Save optimizer states for potential resume.
- Save training logs and hyperparameters for reproducibility.

---

## 5. Additional Considerations

**a. Surrogate Gradient Settings:**
- Surrogate alpha: 1.0 (per YAML).
- Use inverse tangent surrogate:
  \(\Theta(x) \approx \arctan(\alpha x)\),
  with custom backward to approximate derivative:
  \(\frac{d}{dx} \arctan(\alpha x) = \frac{\alpha}{1 + (\alpha x)^2}\).

**b. Feedback Module:**
- During training, ensure feedback connections are learned jointly:
  - Initialize feedback weights.
  - Backpropagate errors through feedback paths (recurrent loops).
  
**c. Computational Efficiency:**
- Use batching and GPU acceleration.
- Use efficient data loaders.
- Limit simulation time per batch to balance computational cost.

**d. Reproducibility:**
- Fix random seed.
- Log hyperparameters, data splits, and training details.

---

## 6. Hyperparameters and Experiment Variants
- Use the YAML-provided settings unless tuning:
  - Learning rate: 0.1.
  - Epochs: 320.
  - Batch size: 32.
  - Surrogate gradient: inverse tangent.
  - Feedback strength: 1.0.
  - Membrane time constant: 2.
  - Threshold: 1.0.
  - Reset voltage: 0.0.
  - Learning rate decay steps: 100; decay rate: 0.1.

---

## Summary
The trainer.py script orchestrates the entire pretraining pipeline:
- Data loading & preprocessing (including manipulation when needed).
- Model creation with feedback and spiking neurons.
- Surrogate gradient-based training for action/object recognition.
- Learning rate scheduling and checkpointing.
- Logging training progress.
This setup closely follows the paper's experimental protocols and the specified configuration, ensuring reproducibility and biological plausibility.

---

This detailed logical workflow forms a comprehensive blueprint for implementing 'trainer.py' aligned with the paper and provided specifications.

