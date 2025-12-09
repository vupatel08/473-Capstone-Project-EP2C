# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a **comprehensive, detailed plan** to reproduce the methodology and experiments described in the paper "ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy". This plan captures the key aspects of the experimental setup, datasets, models, hyperparameters, metrics, and procedural steps to enable precise implementation later.

---

# 1. Overview & Goals

- **Primary Objective:** Conduct an **in-depth comparative analysis** of various models (ConvNets, Vision Transformers, supervised, CLIP-based) beyond traditional ImageNet accuracy.
- **Focus Areas:** Model robustness, calibration, feature invariance, mistake types, transferability, shape vs texture bias, and effect of model size.
- **Key Insight:** Emphasize behavior analysis without relying solely on additional training or fine-tuning, using pre-trained models with fixed parameters.

---

# 2. Model Selection & Configuration

**Models to reproduce:**
- **ConvNeXt models**:
  - Variants: Tiny, Small, Base, Large, Huge
  - Training: Supervised pre-training on ImageNet-21K
  - Use publicly available weights or reproduce training on ImageNet-21K.
- **Vision Transformer (ViT) models:**
  - Variants: ViT-S/16, ViT-L/16, ViT-H/14
  - Training: Supervised on ImageNet-21K
- **CLIP-based models:**
  - Variants: ConvNeXt-L/14, ViT-L/14, ViT-H/14
  - Pretraining: OpenCLIP public weights
  - Use the model weights derived from the LAION-2B pretraining.

**Important details:**
- All models should be obtained via official repositories when possible or pretrained on comparable datasets.
- Keep models fixed for evaluation (no further training or fine-tuning).

---

# 3. Datasets & Data Preparation

### 3.1. Base Training Datasets:
- **ImageNet-21K:**  
  - Use pretrained weights (e.g., from Deit, DeiT, ConvNeXt, or open CLIP weights) trained on this dataset.
  - If training from scratch, replicate architecture details, training schedule, and data augmentations as per original papers (not explicitly provided but can be inferred).

### 3.2. Evaluation Datasets:
- **ImageNet-1K:**  
  - Standard validation set for accuracy, calibration, mistake analysis.
- **ImageNet-R (R-ImageNet):**  
  - For robustness tests; evaluate model accuracy, calibration, and invariances.
- **Synthetic Data (PUG-ImageNet):**  
  - For transferability: Evaluate model matching/mistakes on a large set of synthetically transformed images with associated labels.
- **Out-of-distribution & perturbation datasets:**
  - **Shape vs Texture dataset (Geirhos et al., 2018)**:  
    - To evaluate shape vs texture bias, generate or use provided cue-conflict images (shape vs texture conflict).
- **Calibration datasets:**
  - Use the validation images of ImageNet-1K and ImageNet-R for confidence/histogram calibration analyses.

### 3.3. Data Augmentation & Transformation
- Use standard training augmentation (for training models), such as random resized crops, horizontal flips, color jitter (if training images from scratch).
- For evaluation, only resize and center crop images to model input size.

---

# 4. Model Input & Preprocessing

- Resize images to the model's expected input size:
  - ConvNeXt models: e.g., 224x224
  - ViT models: e.g., 224x224 or as per the size in their configuration.
  - CLIP models: typically 224x224 pixel inputs.
- Normalize images with ImageNet mean and std:  
  \[
  \text{mean}=[0.485, 0.456, 0.406], \quad \text{std}=[0.229, 0.224, 0.225]
  \]
- For synthetic and robustness assessments, apply transformations like scale, shift, resolution variations, and shape-texture confounds.

---

# 5. Evaluation Procedure

### 5.1. Accuracy & Mistake Analysis
- Compute top-1 accuracy on:
  - ImageNet-1K validation
  - ImageNet-R
  - Synthetic datasets (PUG-ImageNet)
- Generate mistake/misclassification statistics:
  - Identify most frequent misclassified classes per model.
  - Use confusion matrices as needed.

### 5.2. Calibration & Uncertainty
- Using model outputs (softmax probabilities):
  - Construct reliability diagrams.
  - Compute ECE (Expected Calibration Error).
  - Analyze confidence histograms.
- Calibration evaluation:
  - Apply binning (e.g., 10-20 bins).
  - Compute ECE on ImageNet-1K and ImageNet-R.

### 5.3. Shape vs Texture Bias
- Use cue-conflict images with controlled shape and texture cues.
- Calculate the fraction of decisions leaning toward shape or texture.
- Measure bias as the proportion toward shape-based or texture-based classification.

### 5.4. Invariance Tests
- **Scale, Shift, Resolution Invariance:**
  - Apply controlled transformations to images:
    - Scale factors (e.g., 1x, 1.25x, 1.5x, 2x, 3x).
    - Shift in pixels.
    - Resolution changes (e.g., 112px, 224px, 336px, etc.).
  - Measure accuracy under each transformation.
  - Quantify invariant behavior through accuracy degradation or consistency over transformations.

### 5.5. Transferability & Robustness
- Use datasets like VTAB-1k (~19 benchmark subsets):
  - Evaluate:
    - Transfer accuracy
    - Calibration (ECE)
    - Mistake types
  - Compare models: CLIP vs supervised ConvNeXt/ViT.
- Robustness evaluation:
  - Use domain shifts, synthetic distortions, to analyze performance stability.
  - Generate robustness curves.

### 5.6. Additional Metrics:
- Model size (parameters)
- Calibration metrics (confidence vs accuracy)
- Mistake typology (shape vs texture)
- Error ratios on ImageNet-X (out-of-distribution vs in-distribution)

---

# 6. Implementation Notes & Tips

- **Hardware:** Use hardware compatible with large models (GPUs with sufficient VRAM).
- **Frameworks:** Prefer PyTorch or TensorFlow based on models available.
- **Pretrained weights:** Use publicly available checkpoints:
  - CLIP via OpenAI/Runway or HuggingFace
  - ConvNeXt, ViT from HuggingFace/torchvision, or official implementations.
- **Custom datasets:** For cue-conflict images and synthetic transformations, generate or acquire datasets matching the paper descriptions and example images.

---

# 7. Hyperparameters & Settings

- **Batch size:** Match original training assumptions for evaluation speed.
- **Evaluation:**
  - For calibration: sufficient samples (e.g., 10k images).
  - For invariance: generate multiple transformed images per original.
- **Metrics calculations:** Use standard implementations of ECE, confusion matrices, accuracy, etc.

---

# 8. Summary of the Step-by-step Experimental Pipeline

1. **Obtain & prepare models**: Load pretrained weights, set evaluation mode.
2. **Prepare datasets**: Download ImageNet validation, ImageNet-R, synthetic datasets, cue-conflict images.
3. **Preprocess input images**: Resize, normalize.
4. **Run evaluations**:
   - Standard accuracy on ImageNet-1K.
   - Mistake & confusion analysis.
   - Calibration: reliability diagrams, ECE.
   - Shape-texture bias on cue-conflict images.
   - Invariance tests under scale, shift, resolution.
   - Transferability on VTAB subsets.
   - Robustness tests on synthetic/noisy/distorted data.
5. **Collect & analyze metrics**: Compare model behaviors by size, calibration, invariance, mistake types.

---

# 9. Potential Missing Details & Clarifications

- Exact training schedule and data augmentation parameters for models trained from scratch (if reproduction needed).
- Specific image transformation parameters (e.g., shift pixel amounts, scale ranges) — can be inferred from typical invariance studies.
- How to generate or access cue-conflict images consistent with the paper's examples.
- Precise binning schemes for calibration metrics.
- Exact subset selections for transferability & robustness evaluations.

---

This roadmap provides a structured outline to implement all experiments thoroughly. It balances fidelity to the original methodology with flexibility for synthetic and out-of-distribution evaluations, which are crucial for beyond-accuracy analysis. Once confirmed, this plan forms a solid foundation to develop the detailed codebase.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will design a modular Python system leveraging PyTorch, torchvision, OpenCLIP, and additional relevant libraries. The core architecture includes dataset loading and pre-processing, model loading (pretrained from open repositories), evaluation routines for accuracy, calibration, invariance, and mistake analysis, encapsulated in dedicated classes. The main script initializes configurations, loads datasets, models, and runs evaluations sequentially, storing and presenting results. Use simple class-based interfaces for models, datasets, and evaluators to keep complexity low, and utilize existing open-source tools for metrics (e.g., torchmetrics).",
    "File list": [
        "app.py",
        "dataset_loader.py",
        "model.py",
        "evaluation.py",
        "invariance_tests.py",
        "visualization.py",
        "utils.py"
    ],
    "Data structures and interfaces": "```mermaid\nclassDiagram\n    class App {\n        +__init__(config: dict)\n        +run(): None\n    }\n    class DatasetLoader {\n        +load_data(split: str) -> Dataset\n        +apply_transformations(image: PIL.Image) -> Tensor\n    }\n    class Model {\n        +load_pretrained(name: str, size: str, weights_source: str) -> None\n        +predict(inputs: Tensor) -> Tensor\n        +get_confidence(probabilities: Tensor) -> Tensor\n    }\n    class Evaluator {\n        +evaluate(model: Model, dataloader: DataLoader) -> dict\n        +compute_calibration(model: Model, dataloader: DataLoader) -> dict\n        +compute_mistake_stats(preds: Tensor, labels: Tensor) -> dict\n        +compute_bias(preds: Tensor, labels: Tensor, dataset: str) -> dict\n        +generate_invariance_tests(model: Model, dataset: Dataset, tests: list) -> dict\n    }\n    class Visualization {\n        +plot_reliability(rel_diag: dict) -> None\n        +plot_confusion(conf_mat: Tensor) -> None\n        +plot_bias_bars(bias_results: dict) -> None\n        +plot_invariance_accuracy(data: dict) -> None\n    }\n    App --> DatasetLoader\n    App --> Model\n    App --> Evaluator\n    Evaluator --> Visualization\n    DatasetLoader --> Dataset\n    Model --> ModelWeights\n    Dataset --> Images\n    ```",
    "Program call flow": "```mermaid\nsequenceDiagram\n    participant C as App\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant EV as Evaluator\n    participant VIZ as Visualization\n    C->>DL: load_data('ImageNet-1K')\n    DL-->>C: dataset\n    C->>MD: load_pretrained('ConvNeXt-Huge', 'Large', 'OpenCLIP')\n    MD-->>C: model instance\n    C->>EV: evaluate(model, dataset) [accuracy, mistake stats]\n    EV->>MD: model.predict(images)\n    MD-->>EV: predictions\n    EV->>VIZ: plot_reliability(calibration_results)\n    EV->>VIZ: plot_confusion(confusion_matrix)\n    EV->>VIZ: plot_bias(bias_results)\n    EV->>VIZ: plot_invariance(invariance_results)\n    C->>EV: evaluate on ImageNet-R, synthetic distortions, cue-conflict images\n    EV-->>C: metrics & plots\n    Note over C: Results stored, summarized, and displayed.\n    ```",
    "Anything UNCLEAR": "Clarify dataset formats for cue-conflict images and synthetic transformations, as well as exact hyperparameters for synthetic invariance tests (scale factors, shift pixels, resolution parameters). Confirm preferred tools for plotting results and whether models should be re-trained or only evaluated using publicly available pretrained weights."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "torchvision==0.14.1",
        "tqdm",
        "matplotlib",
        "numpy==1.21.0",
        "scipy",
        "scikit-learn",
        "torchmetrics"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Class DatasetLoader: Handles data loading, applying necessary transformations for various datasets, including standard, synthetic, and cue-conflict images. Loads datasets like ImageNet, ImageNet-R, and synthetic datasets, applying specific transformations (resize, crop, augmentation) and outputs PyTorch DataLoader objects."
        ],
        [
            "model.py",
            "Class Model: Defines methods to load pretrained models (ConvNeXt, ViT, CLIP variants) with specified configurations (size, weights source). Provides 'predict' method to generate class probabilities and 'get_confidence' to extract confidence scores."
        ],
        [
            "trainer.py",
            "Class Trainer: Manages evaluation process, taking a model and dataset loader, runs inference, and computes metrics such as accuracy, mistake statistics, calibration, bias, and invariance. Separates logic for various evaluation aspects (accuracy, calibration, shape-texture bias, invariance tests)."
        ],
        [
            "evaluation.py",
            "Class Evaluation: Implements methods for computing various metrics, including accuracy, calibration (ECE, reliability diagrams), mistake analysis, bias evaluation, and invariance testing (scale, shift, resolution). Uses torchmetrics and custom code for visualization and analysis."
        ],
        [
            "visualization.py",
            "Functions to visualize calibration (reliability diagrams, confidence histograms), confusion matrices, bias bar charts, and invariance results. Uses matplotlib for plotting and saves or displays results."
        ],
        [
            "app.py",
            "Main script providing orchestrated flow: loads dataset, initializes models, performs evaluations, generates plots, and saves results. Manages dependency between data loader, model, evaluator, visualization, for comprehensive experiments."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "evaluation.py",
        "visualization.py",
        "trainer.py",
        "app.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Define common configuration variables such as dataset paths, model configs, evaluation settings, and helper functions for metrics and plotting. Use a config dictionary or YAML file loaded at startup to keep parameters consistent across modules.",
    "Anything UNCLEAR": "Clarify specific hyperparameters for synthetic transformations (e.g., scale factors, shift pixel ranges) and dataset details for cue-conflict images. Confirm expected input sizes and whether models require image-specific preprocessing steps beyond standard normalization."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: ImageNet-1K
  validation_split: true
  image_size: 224

models:
  convnext_tiny:
    architecture: ConvNeXt-Tiny
    pretrained: true
    dataset: ImageNet-21K
  convnext_small:
    architecture: ConvNeXt-Small
    pretrained: true
    dataset: ImageNet-21K
  convnext_base:
    architecture: ConvNeXt-Base
    pretrained: true
    dataset: ImageNet-21K
  convnext_large:
    architecture: ConvNeXt-Large
    pretrained: true
    dataset: ImageNet-21K
  convnext_huge:
    architecture: ConvNeXt-Huge
    pretrained: true
    dataset: ImageNet-21K
  vit_s_16:
    architecture: ViT-S/16
    pretrained: true
    dataset: ImageNet-21K
  vit_l_16:
    architecture: ViT-L/16
    pretrained: true
    dataset: ImageNet-21K
  vit_h_14:
    architecture: ViT-H/14
    pretrained: true
    dataset: ImageNet-21K
  clip_large:
    architecture: CLIP-Large
    pretrained_source: OpenCLIP
    dataset: LAION-2B
  clip_xl:
    architecture: CLIP-XLarge
    pretrained_source: OpenCLIP
    dataset: LAION-2B

evaluation:
  batch_size: 128
  transformations:
    scale_factors: [1, 1.25, 1.5, 2, 3]
    shift_pixels: [0, 10, 20]
    resolution_sizes: [112, 224, 336, 512, 640]
  inference_steps: 10000

training:
  learning_rate: 0.001  # Default to typical training; actual training is pretrained, no fine-tuning from scratch
  batch_size: 256
  epochs: 100
  optimizer: AdamW
  weight_decay: 0.01
  scheduler: cosine_annealing

misc:
  seed: 42
  evaluation_metrics:
    accuracy: top-1
    calibration: ECE (Expected Calibration Error)
    mistake_analysis: class confusion
    bias: shape vs texture bias
    invariance_tests: scale, shift, resolution
```

---

**Note:** Full configuration is available in `planning_config.yaml`
