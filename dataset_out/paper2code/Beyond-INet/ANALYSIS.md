# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## app.py

# Logic Analysis for `app.py`

This script serves as the central orchestrator for the entire experimental pipeline outlined in the plan. Its purpose is to coordinate data loading, model initialization, evaluation, visualization, and result management in a logical, modular, and reproducible manner. The following detailed analysis specifies the flow, components, dependencies, and key functions to implement in `app.py`.

---

## 1. Initialization & Configuration Loading

- **Load Configuration:**
  - Read and parse the `config.yaml` file (or equivalent configuration dictionary).
  - Extract parameters such as dataset info, model variants, evaluation settings, and miscellaneous options.
  - Use `yaml.safe_load()` to parse configuration at startup.
  - Establish random seed (from `misc.seed`) for reproducibility using `torch.manual_seed()` and `np.random.seed()`.

- **Set up Environment:**
  - Import necessary libraries: `torch`, `torchvision`, `torchmetrics`, `matplotlib`, `numpy`, `scipy`.
  - Set device: GPU if available (`torch.device('cuda')`), else CPU.
  - Configure logging or print statements for traceability.

---

## 2. Data Loading

- **Create Dataset Loader Instance:**
  - Instantiate `DatasetLoader` class with dataset parameters from config (`dataset.name`, `dataset.validation_split`, `dataset.image_size`).
  - Call `load_data(split='validation')` to:
    - Load ImageNet-1K validation dataset.
    - Apply standard preprocessing transforms (resize, center crop, normalization).
    - Return a PyTorch DataLoader with configured `batch_size` and `shuffle=False`.
  - For out-of-distribution, robustness, or synthetic datasets, instantiate additional DataLoaders as needed (e.g., ImageNet-R, cue-conflict, PUG-ImageNet).

---

## 3. Model Initialization

- **Instantiate Model Instances:**
  - For each model in `models` in config:
    - Read architecture type, pretrained flag, dataset, and source.
    - Use `Model.load_pretrained()` method to load pretrained weights:
      - For ResNet/ConvNeXt/Vit: Load from torchvision, HuggingFace, or open repositories.
      - For CLIP: Load from OpenCLIP or compatible sources.
    - Instantiate model objects, set to eval mode (`model.eval()`).
    - Move models to device (GPU/CPU).

- **Store Model Instances:**
  - Keep in a list or dict for iterative access during evaluation.

---

## 4. Evaluation Process

- For each model:
  
  #### 4.1. Accuracy & Mistake Analysis
  - Run inference on the dataset DataLoader:
    - Collect predictions and confidence scores.
    - Calculate top-1 accuracy via `torchmetrics.Accuracy()`.
  - Generate confusion matrices to analyze specific mistake patterns.
  - Store accuracy and mistake stats in a results dictionary.

  #### 4.2. Calibration
  - For predictions, compute:
    - Max softmax probability as confidence.
    - Use torchmetrics or custom functions to compute ECE.
    - Generate reliability diagrams and confidence histograms.
    - Save calibration metrics and plots.

  #### 4.3. Shape vs Texture Bias
  - Evaluate on cue-conflict images:
    - For each image, get model prediction.
    - Determine if prediction favors shape or texture cues.
    - Calculate shape bias (fraction leaning toward shape).
    - Store results and generate bias bar plots.

  #### 4.4. Invariance Tests (Scale, Shift, Resolution)
  - For each transformation type:
    - Apply transformations to validation images:
      - Scale: resize images with factors (1, 1.25, 1.5, 2, 3).
      - Shift: crop images shifted by pixels specified.
      - Resolution: resize images (e.g., 112, 224, 336, 512, 640).
    - Run inference on each transformed set.
    - Record accuracy degradation or consistency.
    - Store results for later visualization.

  #### 4.5. Transferability & Robustness
  - Run evaluation on VTAB datasets:
    - For transferability, perform linear probing on frozen features.
    - Collect accuracy scores across datasets, matrices of transfer results.
  - Run robustness tests on variants:
    - Image noise, distortions, domain shifts.
    - Record accuracy for each variant.

---

## 5. Visualization & Results

- Use `visualization.py` functions:
  - **Calibration plots:** Reliability diagrams, confidence histograms.
  - **Confusion matrices:** Visualize common mistake patterns.
  - **Bias charts:** Bar plots for shape vs texture bias.
  - **Invariance plots:** Accuracy curves vs transformation magnitude.
  - **Transfer & robustness:** Bar charts, comparison tables.

- Save all plots and key metrics to files or display directly.

---

## 6. Results Management

- **Results Collection:**
  - Organize all metrics, figures, and statistics in a nested dictionary or JSON structure.
  - Include per-model performance on each metric and dataset, consistency metrics, and hyperparameters used.

- **Results Output:**
  - Save results to a file (`results.json`, `.csv`, or `.yaml`) for reproducibility and comparison.
  - Optionally, generate summarized report (console print or LaTeX table) for publication.

---

## 7. Error Handling & Reproducibility

- Wrap critical steps in try-except blocks, log errors.
- Ensure models are loaded with fixed seed, deterministic settings if needed.
- Record versions of packages, environment info for reproducibility.
- Use consistent transforms and data splits.

---

## 8. Finalization & Cleanup

- Close data loaders, save plots, and output final report.
- Optionally, transfer results to cloud or download directory.

---

## Summary of Logical Flow in `app.py`

- Load config → Set environment (seed, device)  
- Load datasets → Validation + synthetic + cue-conflict + out-of-distribution datasets  
- Load pretrained models → store in dict/list  
- For each model:
  - Run inference on datasets
  - Compute accuracy, mistake stats, calibration
  - Evaluate bias, invariance
  - Collect metrics and generate plots
- Organize results in structured form
- Save and display results

---

This detailed logical analysis ensures all components are integrated coherently, dependencies are maintained, and the overall process is reproducible, modular, and aligned with the scientific objectives of the paper.

## dataset_loader.py

**Logic Analysis for dataset_loader.py**

---

### Overview:
The purpose of `dataset_loader.py` is to define the `DatasetLoader` class, which manages loading various datasets—such as ImageNet-1K, ImageNet-R, synthetic datasets like PUG-ImageNet, and cue-conflict images—applying appropriate transformations, and providing PyTorch DataLoader objects for evaluation or analysis purposes.

---

### Core Responsibilities:
- Initialize with configuration parameters.
- Load datasets according to specified names and subsets.
- Apply dataset-specific data transformations (resize, crop, augmentation).
- Prepare DataLoader objects with given batch sizes, shuffling policies, etc.
- Support special datasets/testing conditions for invariance, bias, mistake analysis, etc.
- Handle synthetic and out-of-distribution datasets distinct from standard ImageNet.

---

### Step-by-step Details:

---

#### 1. Initialization

- Accept a configuration object or dictionary, which specifies:
  - Dataset name and parameters.
  - Image size, transformations, and dataset paths.
- Set dataset directory paths based on the dataset name (assuming datasets stored locally or via torchvision/datasets).

#### 2. Dataset Loading

- **Standard datasets (e.g., ImageNet):**
  - If the dataset is ImageNet-1K:
    - Use torchvision's `ImageFolder` or a dedicated ImageNet loader.
    - Apply standard transformations (resize to `image_size`, center crop, normalization).
    - Split into validation set (since validation_split is true).
  - Load validation images for performance evaluation.
- **Other datasets:**
  - **ImageNet-R** (out-of-distribution dataset):
    - Load custom dataset, possibly from a specified directory or archive.
    - Use similar transforms as validation for consistency.
  - **Synthetic datasets (PUG-ImageNet):**
    - Load images generated with factors of variation.
    - Likely stored as a folder structure or a structured dataset format.
    - Support batching and apply same preprocessing as evaluation images.
  - **Cue-conflict images:**
    - Load images specifically prepared with conflicting shape and texture cues.
    - Provide labels indicating shape/texture bias evaluation.
    - These images may be stored separately or generated on-the-fly (if generation code provided elsewhere).

#### 3. Transformations

- Define transforms dynamically based on the use case:
  - **Evaluation:**
    - Resize to the dataset's `image_size`.
    - Center crop.
    - Convert to tensor.
    - Normalize mean/std.
  - **Invariance testing (scale, shift, resolution):**
    - Additional or different transforms will be applied during the evaluation phase, possibly within data generator functions.
- Handle dataset-specific transforms, e.g.:
  - For shape/texture bias: apply cue-conflict image trees.
  - For synthetic datasets: apply factors like pose, texture, size variations as per dataset generation methods.

#### 4. DataLoader Preparation

- Wrap datasets with `torch.utils.data.DataLoader`:
  - Use specified batch size (`evaluation.batch_size`).
  - Set `shuffle=False` for validation datasets.
  - Set `num_workers` appropriately (say 4 or 8).
  - Enable pin_memory for GPU speedups.

#### 5. Dataset Interface

- Return DataLoader objects for:
  - Standard validation set.
  - Synthetic datasets.
  - Cue-conflict images.
  - Variations (for invariance tests).

---

### Dataset Modules/Classes:

- Consider defining multiple internal Dataset classes or functions:
  - `ImageNetValidationDataset`
  - `ImageNetRDataset`
  - `SyntheticDataset` (PUG-ImageNet)
  - `CueConflictDataset`
  - Each supports:
    - Initialization with image list and labels.
    - Transformation application.

---

### Dataset Paths & Data Sources:

- Assume dataset paths are configured via environment variables or predefined constants.
- For datasets like ImageNet, rely on torchvision datasets or custom loading if datasets are not prepackaged.
- For synthetic and cue-conflict images, specify local directories or load from structured data files.

---

### Handling Specific Dataset Configurations:
- For synthetic data:
  - Load images corresponding to each factor (pose, size, texture, etc.)
  - Provide mechanism to evaluate performance across these factors.
- For cue-conflict images:
  - Load images designed with conflicting cues, possibly in a dedicated directory.
  - Assign labels for shape-based vs texture-based classification.

---

### Error Handling & Data Validation:
- Check existence and validity of dataset paths.
- Validate number of images loaded matches expectations.
- Ensure transformations do not alter labels or data integrity.

---

### Summary:

**Inputs:**
- Configuration dictionary (dataset name, image size, dataset paths, etc.)

**Outputs:**
- One or more `DataLoader` objects for evaluation or analysis, dynamically configured for each dataset type.

**Key Points:**
- Support multiple dataset types with minimal code duplication.
- Efficient batch processing.
- Dataset-specific transformations for accurate property analysis.
- Modular design to facilitate adding new datasets or testing conditions.

---

This detailed logic analysis ensures a comprehensive, flexible implementation aligned with the experimental needs described in the paper.

## evaluation.py

**Evaluation.py: Logic Analysis**

---

### **Overview**

The `evaluation.py` module is responsible for implementing all evaluation metrics and analyses required for the comprehensive model assessment as outlined in the paper and plan. It will define a dedicated class `Evaluation` that contains methods for computing accuracy, calibration metrics (including ECE and reliability diagrams), mistake analysis, shape/texture bias evaluation, invariance testing, and visualization routines. All methods accept model predictions and dataset information, and produce quantitative metrics and plots, which are stored and/or displayed.

---

### **Primary Components**

1. **Initialization & Inputs**
   - `Evaluation` class is instantiated with configurations such as 
     - dataset name (to determine data specifics),
     - batch size,
     - evaluation transformations (scale, shift, resolution),
     - number of inference steps.
   - It may hold references to utility functions or external libraries (torchmetrics, matplotlib).

2. **Core Methods**
   - `compute_accuracy()`: return top-1 accuracy, over datasets like ImageNet validation, ImageNet-R.
   - `compute_calibration()`: computes ECE, produces reliability diagrams and confidence histograms.
   - `mistake_analysis()`: analyzes errors by class, mistake types, and generates confusion matrices.
   - `bias_evaluation()`: measures shape vs texture bias on cue-conflict images.
   - `invariance_tests()`: evaluates model robustness to scale, shift, resolution transformations.
   - `transferability()`: assess on datasets like VTAB; measure accuracy and calibration.
   - `synthetic_data_evaluation()`: evaluate model performance on synthetic datasets such as PUG-ImageNet.
   
3. **Supporting Functions**
   - `plot_reliability_diagram()`, `plot_confusion_matrix()`, `plot_bias_bars()`, `plot_invariance()`.
   - Possibly helper functions for ECE calculation, binning, and statistical summaries.

---

### **Detailed Functional Breakdown**

#### **1. Accuracy Metrics**

- Input: `predictions`, `labels`.
- Method:
  - Use `torchmetrics`'s `Accuracy` class with `top-1` parameter.
  - For datasets, evaluate in batches, accumulate correct counts, and compute overall accuracy.
- Output:
  - Scalar accuracy value.
  - Alternatively, per-class or confusion matrix for misclassification breakdown.

#### **2. Calibration & Reliability Diagrams**

- Input: `predicted probabilities` and `ground truth labels`.
- Method:
  - Calculate confidence scores: maximum softmax probabilities.
  - Bin predictions into M equal-width bins (e.g., 15 bins as per config).
  - For each bin:
    - Compute mean confidence and accuracy.
    - Compute absolute difference for calibration (for ECE).
  - Use torchmetrics' `ExpectedCalibrationError` if applicable.
- Visualization:
  - Plot reliability diagram:
    - x-axis: confidence.
    - y-axis: accuracy.
    - Points/trends close to diagonal indicate better calibration.
  - Confidence histograms:
    - Distribution plot of predicted confidences.

- Output:
  - Calibration metrics: ECE value.
  - Plots as figures or image files.

#### **3. Mistake Analysis**

- Input:
  - Predictions, true labels.
  - Dataset metadata with annotations for mistake factors (pose, style, texture, occlusion, etc.).
- Method:
  - Identify false predictions.
  - Match each misclassified example to its annotated factors.
  - Calculate error ratios per factor:
    \[
    \text{error_ratio}(factor) = \frac{1 - \text{accuracy}_{\text{factor}}}{1 - \text{overall accuracy}}
    \]
  - Aggregate to see which factors contribute most.
  - Generate confusion matrices.
- Output:
  - Error ratios per factor.
  - Confusion matrix visualization.

#### **4. Shape/Text Bias**

- Input:
  - Predictions, labels, cue-conflict images.
  - Dataset with front-end images having conflicting shape and texture cues.
- Method:
  - For each cue-conflict image:
    - Determine predicted class.
    - Record whether the answer aligns with shape or texture cue.
  - Compute shape bias as the fraction leaning toward shape.
  - Compare but also quantify the texture bias.
- Visualization:
  - Bar plots (bias bars) indicating proportions.

#### **5. Invariance Testing (Scale, Shift, Resolution)**

- Input:
  - Transformed images under different scales, shift pixels, resolutions.
  - Corresponding predictions.
- Method:
  - For each transformation type:
    - Evaluate accuracy compared to baseline (original image).
    - Measure the degradation or invariance (e.g., ratio or difference).
  - Possibly fit curves or compute metrics like area under curve (AUC) for robustness.
- Visualization:
  - Accuracy vs. transformation magnitude plots.

#### **6. Transferability & Robustness**

- Transferability:
  - Load model representations on VTAB datasets.
  - Conduct linear probing; measure accuracy per dataset.
  - Calculate calibration metrics per dataset.
- Robustness:
  - Evaluate on datasets with domain shifts, distortions, and corruptions.
  - Compute accuracy and compare across models.
  - Plot robustness curves if applicable.

#### **7. Synthetic Data Evaluation**

- Load synthetic datasets like PUG-ImageNet.
- Evaluate model accuracy across various factors.
- Highlight differences in synthetic-data robustness.

---

### **Important Implementation Details**

- **Metrics & Visuals:**
  - Use torchmetrics for accuracy and ECE.
  - Use matplotlib for diagrams; save figures to files.
- **Batch Processing:**
  - Leverage DataLoader objects for efficiency.
  - Process images in batches for inference.
- **Reproducibility:**
  - Use fixed seeds (`seed=42` from config).
  - Consistent binning and thresholds for metrics.
- **Model Outputs:**
  - For CLIP models: use `logits` or `probabilities` from the openCLIP interface.
  - For ConvNeXt/ViT: softmax over class logits.
- **Handling Dataset Annotations:**
  - Load annotations for mistake factors and cue-conflict labelings from local files or provided datasets.
  - Map images to their annotated factors.

---

### **Edge Cases & Additional Notes**

- **Imbalanced datasets vs. binning for ECE**: Ensure bins are properly populated; prevent empty bins.
- **Out-of-distribution or contrast datasets** may require normalization or specific handling.
- **Synthetic & cue-conflict images**: must be preloaded or generated for proper analysis.
- **Model loading**: Use pretrained weights as per the configuration; handle missing weights gracefully.
- **Dataset splits**: strictly use validation set for the evaluations.

---

### **Summary**

`evaluation.py` will be a comprehensive module providing:

- Functions for computing accuracy, calibration, mistake statistics.
- Specialized analysis functions for shape/texture bias.
- Invariance testing functions with input transformations.
- Visualization routines for all metrics.
- Integration points for dataset-specific annotations.
- A clean interface allowing the main script to orchestrate experiments easily.

All methods will produce quantitative results and visual outputs, aligning tightly with the experimental descriptions in the paper.

---

This detailed logic plan ensures correct, reproducible, and organized implementation for all evaluation aspects as per the paper’s methodology.

## model.py

**Logic Analysis for 'model.py' — Implementation of the `Model` class**

---

### Purpose:
Design and implement a class `Model` that encapsulates the loading, inference, and output processing of pretrained vision models (ConvNeXt, ViT, CLIP variants). The class must support:
- Loading specific models with configurations based on architecture, size, and pretrained weights/source
- Running inference on input images
- Extracting class probabilities
- Computing confidence scores
- Handling models across multiple architectures and training paradigms (Supervised, CLIP)

This class is a core component for evaluation pipelines, supporting accuracy, calibration, mistake analysis, and other metrics.

---

### Functional Requirements:

1. **Model Loading (`load_pretrained`)**:
   - Inputs: model name, size (if applicable), pretrained indicator, source (for CLIP models)
   - Outputs: Initialized model with pretrained weights loaded, ready for inference
   - Behavior:
     - For **ConvNeXt** and **ViT** models:
       - Load from torchvision or HuggingFace/official repositories
       - Use model zoo weights trained on ImageNet-21K, as per configuration
     - For **CLIP** models:
       - Use OpenCLIP/pretrained weights from 'OpenCLIP' source
       - Load via open-source CLIP library or HuggingFace
   - Checkpoint loading:
     - Load exact weights specified in configuration
     - Set model to evaluation mode (`model.eval()`)
     - Handle device placement (CPU/GPU)

2. **Inference (`predict`)**:
   - Input: A tensor batch of preprocessed images
   - Output: Raw model logits
   - Behavior:
     - Perform forward pass in `torch.no_grad()` context
     - Ensure input tensors are on the correct device
     - Process batch inputs efficiently
   - For CLIP models:
     - Output logits are for zero-shot classification or similarity scores
     - For consistency, convert to class probabilities (softmax)

3. **Class Probabilities (`get_probabilities`)**:
   - Input: logits from `predict()`
   - Output: normalized probabilities via softmax
   - Implementation:
     - Use `torch.nn.functional.softmax(logits, dim=1)`

4. **Confidence Extraction (`get_confidence`)**:
   - Input: probabilities
   - Output: maximum probability per sample (overall model confidence for each prediction)
   - Implementation:
     - Use `probabilities.max(dim=1)` to get confidence score per sample
  
5. **Model Variants & Handling**:
   - Support multiple architectures with predefined names in configuration.
   - Support for different sizes:
     - For ViT: S, L, H
     - For ConvNeXt: Tiny, Small, Base, Large, Huge
     - For CLIP: Large, XLarge
   - For each, ensure correct input preprocessing, output layer configuration, and weight loading.
  
6. **Device Management**:
   - Allow for model to be loaded onto CPU or CUDA
   - Default to GPU if available, else CPU
   - When loading weights, ensure matching device placement
   - Add optional parameter for device specification during object creation

7. **Error Handling & Validation**:
   - Raise meaningful errors if:
     - Model name or size not recognized
     - Pretrained weights not found
     - Model loading fails
   - Validate input image tensor shape and data range (should be preprocessed: normalized, size aspect ratio maintained)

---

### Implementation Details:

- **Class Attributes:**
  - `self.model`: Holds the actual model instance
  - `self.architecture`: String identifier for architecture (e.g., 'ConvNeXt-Huge')
  - `self.device`: Device string ('cuda' or 'cpu')
  - `self.model_name`: Full identifier (e.g., 'ConvNeXt-Huge')
  - `self.pretrained_source`: For CLIP models; default to 'OpenCLIP' per config
  - `self.weights_path`: Path to the checkpoint if needed (or URL)

- **Methods:**
  - `__init__(self, architecture, pretrained=True, dataset='ImageNet-21K', pretrained_source=None, device=None)`
  - `load_pretrained(self)`:
    - Internal logic to select and instantiate model, load weights
    - Use conditional logic for architecture types
  - `predict(self, images: Tensor) -> Tensor`:
    - Run inference, output logits
  - `get_probabilities(self, logits: Tensor) -> Tensor`:
    - Apply softmax
  - `get_confidence(self, probabilities: Tensor) -> Tensor`:
    - Max probability per sample
      
---

### Special Cases & Considerations:

- **CLIP models:**
  - These often contain separate text and image encoders
  - For zero-shot evaluation:
    - Build class text features
    - Compute cosine similarity
    - Convert similarity scores into probabilities
  - Alternatively, for consistency with the rest:
    - Use the image encoder to produce features
    - Use `predict()` to output similarity scores as logits
- **Input size expectations:**
  - Confirm 'image_size' from config
  - For CLIP: typical size is 224x224, but may vary
- **Normalization:**
  - Ensure image tensors are normalized as per model's expected input
  - May need to provide a `preprocess()` method or rely on external preprocessing

---

### Summary of Implementation Expectations:

- The `Model` class:
  - Encapsulates model selection, loading, and inference
  - Supports diverse architectures and training paradigms
- Provides clear API:
  - `load_pretrained()`
  - `predict(images)`
  - `get_probabilities(logits)`
  - `get_confidence(probabilities)`
- Handles device management internally
- Errors are informative and validation is performed

---

This logic analysis ensures that the `model.py` file provides robust, flexible, and clear functionality to load, run, and interpret pretrained vision models, in line with experimental needs described in the paper and design plan.

## trainer.py

{
  "trainer.py - Logic Analysis": [
    {
      "Purpose": "Implement the core class `Trainer` responsible for orchestrating model evaluation across multiple properties as described in the paper. It will handle data iteration, inference, and metrics computation for accuracy, mistake analysis, calibration, bias, invariance testing, and result visualization.",
      "Main Responsibilities": [
        "Initialize with a model, dataset loader, and configuration parameters.",
        "Run inference over datasets (validation, test sets, or synthetic/transformed versions).",
        "Compute and aggregate metrics: accuracy, confusion matrices, ECE, bias measures, invariance scores.",
        "Manage multiple evaluation modes: standard accuracy, mistake analysis, calibration, bias, invariance.",
        "Call visualization functions to produce plots (reliability diagrams, confusion matrices, bias bars, invariance curves).",
        "Store results in structured formats for comparison and reporting."
      ],
      "Key Components/Functions": [
        {
          "Initialization (__init__)": "Accepts parameters such as model, dataset loader, config dict, and evaluation mode flags. Sets up internal state and metrics objects."
        },
        {
          "run_evaluation()": "Main function to run the full suite of evaluations. It calls specific methods for accuracy, mistake analysis, calibration, bias, invariance, each in turn."
        },
        {
          "compute_accuracy()": "Iterate over dataset DataLoader, perform inference, compare predictions with labels, accumulate correct counts, compute top-1 accuracy."
        },
        {
          "compute_mistake_stats()": "Identify misclassified examples, analyze their class confusion patterns, and possibly compute error ratios with respect to factors (e.g., pose, style) if such info available."
        },
        {
          "compute_calibration()": "Collect confidences and correctness labels across samples, construct confidence bins, compute per-bin accuracy and confidence, then compute ECE. Generate reliability diagrams and histograms."
        },
        {
          "compute_bias()": "Using cue-conflict images, classify decisions as shape-based or texture-based. Calculate the bias fractions for each image/set. Possibly generate bias bar plots."
        },
        {
          "compute_invariance()": "Apply transformations (scaling, shifting, resolution change) to images. Measure model accuracy under these transformations. Collect accuracy vs transformation parameter data, and prepare for plotting invariance curves."
        },
        {
          "evaluate_transferability()": "Run inference on VTAB datasets (or similar). Use linear probes if required or directly measure accuracy of frozen features."
        },
        {
          "evaluate_robustness()": "Test model on datasets with natural corruptions, distortions, synthetic modifications, and record accuracy and error patterns."
        },
        {
          "visualize_results()": "Call plotting functions for reliability diagrams, confusion matrices, bias bar charts, invariance curves, error ratio plots. Save or display plots."
        }
      ],
      "Data Handling & Inputs": [
        "Dataset Loader: Provides DataLoader objects for different datasets and transformations.",
        "Model: Accepts input images, performs inference, outputs class probabilities or logits.",
        "Configuration parameters: Sets evaluation batch sizes, number of bins, specific transformation parameters."
      ],
      "Metrics & Computation": [
        "Accuracy": Correct predictions / total in dataset.",
        "Confusion Matrix": For mistake analysis and class-wise errors.",
        "ECE": Bin confidence scores, measure calibration error as per formula. Implement using torchmetrics or custom code.",
        "Bias Measures": Calculate shape vs texture bias from cue-conflict images based on model predictions.",
        "Invariance Results": Accuracy variation as a function of transformation parameters.",
        "Transfer & Robustness": Accuracy over different unseen datasets or corruptions."
      ],
      "Visualization & Output": [
        "Reliability Diagrams": Plot predicted confidence vs empirical accuracy per bin.",
        "Confidence Histograms": Plot frequency of predictions per confidence bin.",
        "Confusion Matrices": Heatmaps for class errors.",
        "Bias Bar Plots": Bar charts of shape vs texture bias fractions.",
        "Invariance Plots": Line plots showing accuracy across scales, shifts, and resolutions."
      ],
      "Error Handling & Flexibility": [
        "Support evaluation on datasets with or without appended or factor annotations.",
        "Use configuration flags to enable/disable specific evaluations.",
        "Ensure data iteration is robust, with progress tracking (via tqdm) if needed.",
        "Design for extension to incorporate additional metrics or datasets."
      ],
      "Output Data Structures": [
        "Results Dict: Structured dictionary with keys like 'accuracy', 'confusion', 'calibration', 'bias', 'invariance', each containing respective metrics and plots.",
        "Visualization outputs saved as images; metrics saved in JSON or CSV format for reporting."
      ],
      "Notes & Clarifications": [
        "Ensure evaluation is done in evaluation mode (no gradient updates).",
        "Use the batch size from config, defaulting to 128.",
        "For synthetic & cue-conflict images, assume datasets are preloaded or generated prior, and passed to the evaluation routines.",
        "Support multiple factors (pose, style, occlusion, texture, etc.), provided via dataset annotations or metadata; if missing, focus on available metrics.",
        "Maintain a consistent API with clear separation of concerns: data loading, inference, metrics, visualization."
      ],
      "Summary": "This class centralizes all evaluation logic: loading datasets, running inference, calculating detailed metrics, producing visualizations, and storing results. It interacts with data loader, model, and visualization modules, and config-driven parameters ensure reproducibility and flexibility. Explicitly, methods are designed modularly for each property analyzed in the paper, following the outlined plan and maintaining clarity and extendability."
    }
  ]
}

## visualization.py

**Logic Analysis for `visualization.py`**

---

### **Purpose & Scope**

This module contains functions responsible for visualizing various evaluation results, including calibration analysis, mistake confusion matrices, shape/texture bias bars, and invariance test accuracy trends. These visualizations assist in understanding model behaviors beyond raw accuracy metrics, aligning with the paper’s comprehensive analysis framework.

### **Inputs & Outputs**

- **Inputs:**  
  Depending on the specific function, inputs are typically dictionaries, arrays, or tensors that contain processed metrics, data points, or results obtained from evaluation modules.
  
- **Outputs:**  
  Visual plots (matplotlib figures) either displayed directly or saved to disk. No return values are expected from visualization functions; they are primarily for rendering.

---

### **Key Visualization Functions & Considerations**

#### 1. **Plot Reliability Diagrams and Confidence Histograms**

**Function Name Suggestion:** `plot_reliability_diagram_and_histogram`

**Inputs:**  
- `rel_diag`: Dictionary or data structure containing bin-wise data for a reliability diagram (bin confidence, accuracy).  
- `conf_hist`: Histogram data for confidence levels (probability distribution per bin).

**Process & Logic:**  
- *Reliability Diagram:*  
  - Plot bin centers vs. (predicted confidence, true accuracy).  
  - Plot the diagonal line y=x for perfect calibration reference.  
  - Overlay model’s reliability curve.  
  - Use consistent axis ranges (0-1 for confidence and accuracy).  
- *Confidence Histogram:*  
  - Plot histogram bars for the distribution of predicted confidences across bins.  
  - Emphasize high-confidence bins to detect over- or under-confidence patterns.

**Implementation Details:**  
- Use `matplotlib.pyplot` to generate subplots (e.g., 2x1 layout).  
- Add labels, titles, legends.  
- Optionally, save figures with descriptive filenames (e.g., `calibration_reliability.png`).  

---

#### 2. **Plot Confusion Matrices**

**Function Name Suggestion:** `plot_confusion_matrix`

**Inputs:**  
- 2D confusion matrix array (N x N where N is number of classes).  

**Process & Logic:**  
- Use `matplotlib.pyplot.imshow()` with a colormap (`Blues`, `hot`, or `viridis`) for visual clarity.  
- Add color bar for scale reference.  
- Annotate cells with numeric counts or normalized rates, if value display is required.  
- Set class labels on axes if available.  
- Format axes (e.g., rotate labels) for readability.  

---

#### 3. **Plot Bias Bar Charts (Shape vs Texture Bias)**

**Function Name Suggestion:** `plot_bias_bars`

**Inputs:**  
- Dictionary or DataFrame containing bias metrics for each model (e.g., shape bias, texture bias percentages).

**Process & Logic:**  
- Plot grouped bar charts, with models on the x-axis and bias percentages on the y-axis.  
- Use different colors for shape bias and texture bias bars for clear comparison.  
- Add legend, axis labels, and title.  
- Align bars for easy comparison across models.

---

#### 4. **Plot Invariance Test Results (Accuracy Over Transformations)**

**Function Name Suggestion:** `plot_invariance_results`

**Inputs:**  
- Dictionary mapping transformation types (scale, shift, resolution) to lists/arrays of accuracy values over specific levels (e.g., scale factors, shift pixels, resolution sizes).  

**Process & Logic:**  
- For each transformation type, generate line plots showing accuracy versus the transformation magnitude.  
- Use distinct line styles/colors to distinguish between models or datasets.  
- Include axes labels (`Transformation level`, `Accuracy`), legend, grid.  
- Optionally, highlight points of key interest (e.g., when accuracy drops below a threshold).  
- Save plot for analysis.

---

### **Common Details & Best Practices**

- **Figure Management:**  
  - Create figures with `plt.figure()` or `plt.subplots()`.  
  - Close figures after saving or displaying with `plt.close()` to prevent memory leaks.  

- **Consistency:**  
  - Maintain consistent axis scales and labels across plots for comparability.  
  - Use normalized axes where appropriate (e.g., 0-1 confidence).  

- **Customization & Flexibility:**  
  - Allow optional parameters for titles, save paths, figure size.  
  - Include plot annotations for clarity when needed.

- **Performance & Efficiency:**  
  - For multiple subplots, organize with `subplots()` for efficiency.  
  - Avoid redundant figure creation.  

---

### **Summary of Implementation Tasks**

| Function                                | Inputs                                          | Core Actions                                                     | Expected Output                        |
|-----------------------------------------|------------------------------------------------|------------------------------------------------------------------|----------------------------------------|
| `plot_reliability_diagram_and_histogram` | rel_diag dict, conf_hist array               | Plot reliability curve + histogram                                | Reliability diagrams + histograms   |
| `plot_confusion_matrix`                  | confusion matrix array                        | Visualize with `imshow()`, add labels and color bar              | Confusion matrix figure             |
| `plot_bias_bars`                         | bias metrics dict/list                        | Bar chart comparison of shape vs. texture bias                   | Bias bar chart                      |
| `plot_invariance_results`                | accuracy lists per transformation             | Line plots per transformation type                                | Invariance accuracy trend plots     |

---

### **Potential Data Structures**
- Use dictionaries or pandas DataFrames for structured data (bias, invariance metrics).  
- Use numpy arrays or tensors for raw model predictions, confidence, and confusion matrices.

---

### **Note on Data Privacy & Storage**
- Save plots in designated output folders with meaningful filenames (e.g., `calibration_diagram.png`, `bias_comparison.png`).  
- Provide options to display or save automatically, governed by function parameters.

---

**This logic analysis guides the functional implementation, ensuring that the visualizations reliably and clearly communicate the model behaviors as outlined in the paper.**

