# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, detailed plan for reproducing the methodology and experiments presented in the paper **AutoVP: An Automated Visual Prompting Framework and Benchmark**. This roadmap is organized into distinct sections covering core methodology, experimental design, hyperparameter management, and evaluation metrics. It will serve as a solid guide for subsequent implementation.

---

## 1. Core Methodology of AutoVP

### 1.1 Overview and Key Components
AutoVP is an end-to-end, modular framework for designing and optimizing visual prompting (VP) strategies on pre-trained models for various image classification tasks. The primary design choices involve:
- Input scaling (resizing factors).
- Visual prompts (padding and prompts added around images).
- Selection of pre-trained backbone models.
- Label mapping strategies at the output.

AutoVP automates the search over hyperparameters to find the optimal configuration, optimizing for accuracy.

### 1.2 Workflow Steps
- **Input Processing**: Resize images with a configurable scale factor.
- **Prompt Application**:
  - Generate visual prompts (padding images) of given size $p$.
  - Prompts are learned prompts, either directly at pixel level or in frequency domain.
- **Feature Extraction**:
  - Feed prompted images into a pre-trained backbone (e.g., ResNet, Swin, CLIP).
  - Backbone weights are frozen during prompting optimization.
- **Output Mapping & Label Transfer**:
  - Use one of 4 strategies:
    - Frequency Map (FreqMap): Map target class based on the most frequent source class predictions.
    - Iterative Map (IterMap): Repeat frequency mapping at each epoch.
    - Semantic Map (SemanticMap): Use CLIP encodings and cosine similarity to map source to target classes.
    - Fully Connected Layer (FullyMap): Train a linear layer on source predictions to align with target classes.
- **Training and Optimization**:
  - Prompts (pixel-level prompts or frequency domain prompts) are trainable parameters.
  - Hyperparameters such as learning rate, prompt size, prompt type, optimizer settings, and label mapping parameters are tuned via a hyperparameter search.
- **Auto-tuning**:
  - The framework uses grid search/early stopping for hyperparameters (prompt size, input scale, prompt type, model, label mapping).
  - The best configuration, according to validation/test accuracy, is selected.
- **Evaluation**:
  - Metrics: Accuracy/IoU for classification/segmentation, robustness scores, etc.
  - Analysis of hyperparameter impacts and ablations.

---

## 2. Dataset and Experimental Setup

### 2.1 Datasets
- Use the 12 classification datasets listed:
  - Examples: EuroSAT, CIFAR10/100, Flowers102, Food101, UCF101, OxfordIITPet, ISIC, FMoW, GTSRB, Food, Pets.
- For segmentation: ISIC and DTD.
- For detection: VOC, ISIC.
- Additional datasets for robustness/other experiments: VOC, SUN, DTD, etc.

**Implementation note**: 
- Download datasets following their official sources.
- Split datasets into train/test sets as per original papers (if available), or use standard splits.
- For segmentation/detection datasets, ensure annotations are available (e.g., masks, bounding boxes).

### 2.2 Data Preprocessing
- **Images**: Resize according to input scale hyperparameter, maintaining aspect ratios or resizing to fixed dimensions.
- **Segmentation/Detection**: Need masks or bounding box labels.
- **Batching**: Use consistent batch sizes (e.g., 32 or 64) matching original experiments.

### 2.3 Model Selection
- Implement pre-trained backbones:
  - Vision-only models: ResNet-18, ResNeXt-101-IG, Swin-T.
  - Vision-language model: CLIP (ViT-B/32).
- Freeze backbone weights during prompt training.
- For CLIP, extract text embeddings for class names for semantic mapping.

### 2.4 Evaluation Metrics
- **Classification**: Top-1 accuracy.
- **Segmentation**: Intersection over Union (IoU), pixel accuracy.
- **Detection**: IoU, AP.
- **Robustness Testing**: Accuracy drop under corruptions (e.g., CIFAR-C, ImageNet-C).

### 2.5 Experimental Settings
- For initial experiments, replicate AutoVP’s choices:
  - Prompt size: e.g., 16 or 48 pixels.
  - Input scaling: factors like 0.5, 1.0, 1.5.
  - Prompts: learned pixel prompts or frequency domain.
  - Label mapping: FreqMap, IterMap, SemanticMap, FullyMap.
- Use standard training/validation splits.
- Implement early stopping for hyperparameter search (e.g., 2-5 epochs per configuration).

---

## 3. Hyperparameter Search & Tuning Details

### 3.1 Hyperparameters to Search
**Prompt Size**:
- Values: 16, 48 (selected by paper).
  
**Input Scale Factor**:
- Values: 0.5, 1.0, 1.5.
  
**Prompt Type**:
- Pixel prompts (learn pixel tokens of size $p$).
- Frequency domain prompts (via FFT).

**Model Choice**:
- ResNet18, ResNeXt-101-IG, Swin-T, CLIP.

**Label Mapping Strategy**:
- Choose among FreqMap, IterMap, SemanticMap, FullyMap.

**Prompt Size $p$**:
- E.g., 16 or 48 pixels.
  
**Number of source classes mapped (if applicable)**:
- E.g., 1, 5, 10 source classes per target.

### 3.2 Optimization and Tuning Strategy
- Use grid search over the hyperparameter combinations.
- Implement early-stopping (e.g., 2-5 epochs max per configuration).
- Use validation/test accuracy for selection.
- Tuning of learning rate:
  - Values: 1e-5, 1e-4, 1e-3.
- Weight decay:
  - Values: 0, 1e-5, 1e-4, 1e-3.
- For prompt learning:
  - Optimizer: Adam or SGD.
  - Learning rate: as above.

### 3.3 Hyperparameter Tuning Procedure
- Automate grid search with early stopping (e.g., via Ray Tune or manual loops).
- Record validation accuracy per configuration.
- Pick best performing configuration.
- Retrain full prompt with final configuration.

---

## 4. Training Details
- Use a small number of epochs for tuning (2-5 epochs).
- Final training:
  - Full dataset training for convergence (~50 epochs).
  - Use the best hyperparameters.
- Batch size: 32 or 64.
- Learning rate scheduling: cosine decay or step decay.
- Optimization:
  - Prompts are trainable parameters.
  - Backbone remains frozen.

### 4.1 Prompts
- Pixel prompts: initialized as zeros/binary masks.
- Frequency prompts: initialized as zeros in spatial domain, or FFT components.
- Regularization: optional (e.g., sparsity or smoothness).

### 4.2 Implementation Details
- For prompt optimization, maintain a separate learnable tensor.
- When in frequency domain, FFT prompts are parameterized in the frequency space.
- Norm and regularization may be applied to prompts.

---

## 5. Evaluation & Analysis
- **Accuracy vs. Hyperparameters**:
  - Plot accuracy against prompt size, input scaling, mapping method.
- **Robustness Tests**:
  - Evaluate on datasets like CIFAR-C or ImageNet-C.
  - Measure accuracy drops.
- **Prompt Visualization**:
  - Visualize learned prompts in pixel and frequency domain.
- **Ablations**:
  - Compare with models trained without prompts, with fixed prompts, or with different prompts.

---

## 6. Additional Implementation Details & Caveats
- **Dataset adaptation**:
  - For segmentation/detection, generate prompts per image or per batch.
  - For CLIP, use text encoder for semantic mapping.
- **Prompt initialization**:
  - Zero or random.
- **Fine-tuning vs static prompts**:
  - Default to prompted parameters trained only for a few epochs or full training.
- **Evaluation protocol**:
  - Use the same test datasets and splits.
- **Reproducibility**:
  - Set random seeds.
  - Document experiment configurations.

---

## Summary Roadmap (High-Level)

1. **Dataset Preparation**: Download datasets, prepare splits, annotations.
2. **Model Loading**: Implement or load pre-trained backbones, freeze weights.
3. **Prompt Module Implementation**:
   - Pixel prompts: trainable tensor of size $p \times c \times 1 \times 1$.
   - Frequency prompts: trainable in FFT domain.
4. **Input Scaling & Prompt Application**:
   - Resize images according to scale hyperparameters.
   - Add prompts (padding pixels or FFT-based).
5. **Feature Extraction**: Forward prompted images through backbone.
6. **Output Mapping & Class Prediction**:
   - Implement each of FreqMap, IterMap, SemanticMap, FullyMap.
   - For semantic mapping, encode class names with CLIP text encoder.
7. **Hyperparameter Search**:
   - Grid over prompt size, input scale, model type, mapping strategy.
   - Utilize early stopping.
8. **Training & Fine-tuning**:
   - Train prompts' parameters for a small number of epochs during tuning.
   - Final full training with the best configuration.
9. **Evaluation**:
   - Metrics: accuracy, IoU, robustness metrics, etc.
   - Visualizations of prompts and correspondence mappings.
10. **Analysis & Ablations**:
    - Plot accuracy gains, robustness, prompt visualizations.

---

This plan offers a comprehensive, step-by-step approach to implement AutoVP from scratch, covering all methodological and experimental nuances described in the paper. It emphasizes clarity, reproducibility, and detailed hyperparameter management, providing a solid foundation for subsequent code development.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Design a modular system with clear separation between data loading, prompt generation, model handling, and hyperparameter tuning. Use PyTorch for model implementation, HuggingFace's transformers for CLIP, and torchvision for datasets and image transforms. Leverage numpy for FFT frequency prompts. Automate hyperparameter search with Ray Tune or a custom grid search with early stopping. Prompts are trainable parameters; backbone models are frozen. Label mapping strategies are implemented as separate functions/classes, with semantic mapping utilizing CLIP text embeddings. Final training retrains prompts with optimal hyperparameters, then performs evaluation and visualization. Keep core components as classes with clear APIs.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "prompt_module.py",
        "label_mapping.py",
        "trainer.py",
        "tuner.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(datasets: List[str], batch_size: int, transforms: callable)\n        +load_data() -> Dict[str, Any]\n    }\n    class PretrainedModel {\n        +__init__(model_name: str, freeze: bool=True)\n        +forward(x: Tensor) -> Tensor\n        +extract_features(x: Tensor) -> Tensor\n    }\n    class PromptGenerator {\n        +__init__(prompt_size: int, prompt_type: str, init_type: str)\n        +get_prompt() -> Tensor\n        +update(prompt_grads: Tensor) -> None\n        +visualize() -> image\n    }\n    class LabelMapper {\n        +__init__(strategy: str, class_names_source: List[str], class_names_target: List[str], map_params: dict)\n        +map(predictions: Tensor) -> Tensor\n        +update_mapping() -> None\n        +visualize_mapping() -> image\n    }\n    class Trainer {\n        +__init__(model: PretrainedModel, prompts: PromptGenerator, dataset: Dict[str, Any], label_mapper: LabelMapper, optimizer: torch.optim.Optimizer, lr_scheduler: optional)\n        +train(epoch: int) -> None\n        +save_checkpoint(path: str) -> None\n        +load_checkpoint(path: str) -> None\n    }\n    class HyperparameterTuner {\n        +__init__(search_space: dict, max_trials: int, evaluation_metric: str)\n        +run_tuning() -> dict\n        +get_best_config() -> dict\n    }\n    class Evaluator {\n        +__init__(model: PretrainedModel, prompts: PromptGenerator, dataset: Dict[str, Any], label_mapper: LabelMapper)\n        +evaluate() -> dict\n        +visualize_results() -> None\n    }\n\nMain --> DatasetLoader\nMain --> HyperparameterTuner\nMain --> Trainer\nTrainer --> PretrainedModel\nTrainer --> PromptGenerator\nTrainer --> LabelMapper\nMain --> Evaluator\nPretrainedModel <|-- CLIPModel\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant P as PromptGenerator\n    participant Mdl as PretrainedModel\n    participant LM as LabelMapper\n    participant TR as Trainer\n    participant EV as Evaluator\n    M->>DL: load_data()\n    DL-->>M: datasets\n    M->>Mdl: init(model_name, freeze=True)\n    M->>P: init(prompt_size, prompt_type, init_type)\n    M->>LM: init(strategy, class_names_source, class_names_target, params)\n    M->>TR: start_training()\n    TR->>MDL: forward(x)\n    MDL-->>TR: features/predictions\n    TR->>P: get_prompt()\n    P-->>TR: prompt tensor\n    TR-->>Main: training complete\n    Main->>EV: evaluate()\n    EV->>MDL: forward(x)\n    MDL-->>EV: preds\n    EV-->>Main: metrics, visualizations\n    Main->>P: visualize prompts\n    Main->>LM: visualize mapping\n    Note over P,LM: Hyperparameter tuning loop runs here with grid search, updating prompts and mapping as needed.\n",
    "Anything UNCLEAR": "Clarify dataset formats (especially segmentation/detection annotations), specify the exact hyperparameter ranges for prompts, prompt sizes, and input scales, and confirm the preferred frameworks (PyTorch, HuggingFace). Also, confirm whether prompt prompts are pixel-based or in frequency domain, and how to handle multi-class label mapping in actual code."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "torchvision==0.14.1",
        "numpy==1.23.1",
        "pillow==8.4.0",
        "scipy==1.10.1",
        "transformers==4.26.1",
        "timm==0.6.13",
        "ray[tune]==1.13.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Class DatasetLoader: responsible for downloading, loading, and batching datasets for classification, segmentation, and detection tasks; handles dataset-specific transformations and annotations. Implements load_data() returning datasets as dicts with images and labels or masks."
        ],
        [
            "model.py",
            "Class PretrainedModel: initializes and manages pre-trained backbones (ResNet, Swin, CLIP). Methods include forward() for feature extraction, extract_features() for embeddings, with frozen weights. Supports CLIP text embedding extraction."
        ],
        [
            "prompt_module.py",
            "Class PromptGenerator: manages pixel prompt tensors or frequency domain prompts; provides methods get_prompt() for current prompt, update() for gradient updates, and visualize() for visualizing prompts. Handles both pixel-based and FFT prompts."
        ],
        [
            "label_mapping.py",
            "Class LabelMapper: implements different label mapping strategies (FreqMap, IterMap, SemanticMap, FullyMap). Methods include map() for predictions, update_mapping() for iterative strategies, and visualize_mapping(). Supports semantic mapping via CLIP text embeddings."
        ],
        [
            "trainer.py",
            "Class Trainer: coordinates training of prompts and models. Methods include train() for prompt optimization over epochs, supports early stopping via validation accuracy. Manages optimizer, learning rate scheduling, and checkpoints."
        ],
        [
            "tuner.py",
            "Class HyperparameterTuner: performs grid search or Bayesian search with early stopping over hyperparameters (prompt size, scale, model, label map strategy). Runs multiple experiments, records results, and retrieves best hyperparameters."
        ],
        [
            "evaluation.py",
            "Class Evaluator: evaluates trained model on test set, computes accuracy, IoU, robustness metrics; supports visualization of prompts, mappings. Implements evaluate() and visualize_results()."
        ],
        [
            "main.py",
            "Main application entry point. Initializes configuration, loads datasets, models, prompts. Runs hyperparameter tuning, trains the best configuration, then performs evaluation and visualization."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "prompt_module.py",
        "label_mapping.py",
        "trainer.py",
        "tuner.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (leave empty if no client-server API needed, otherwise describe endpoints for dataset upload, training configuration, results retrieval and visualization)",
    "Shared Knowledge": "Shared components include data augmentation transforms, normalization configs, and a common configuration dictionary. Prompts are trainable tensors; hyperparameters are managed via configuration objects passed across modules.",
    "Anything UNCLEAR": "Clarify if prompts are pixel-based or frequency domain; confirm dataset versions and splits; specify hardware constraints for large datasets; clarify if multi-GPU training is required."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  # The number of total training epochs (auto prompts training runs for 2-5 epochs, final training may be longer)
  epochs: 50
  
  # Learning rate for prompt optimizer (guided by hyperparameter tuning, e.g., 1e-5, 1e-4, etc.)
  learning_rate: 1e-4

  # Batch size for dataset loader
  batch_size: 32

  # Learning rate scheduler type (optional: 'cosine' or 'step')
  lr_scheduler: cosine

  # Early stopping patience in epochs during hyperparameter tuning (if validation accuracy does not improve for this many epochs)
  early_stop_patience: 3

  # Weight decay for optimizer (tuning parameter)
  weight_decay: 1e-5

  # Total number of training iterations (can be determined by dataset size and epochs)
  total_iterations: null  # will be computed based on dataset size and batch size

model:
  # Pre-trained backbone model name (e.g., 'resnet18', 'resnext101-ig', 'swin-t', 'clip')
  backbone: clip

  # Backbone frozen during prompt training
  freeze: true

  # Input scale factor options (for resizing images)
  input_scale_options: [0.5, 1.0, 1.5]

  # Prompt size in pixels
  prompt_size: 16

  # Prompt initialization type ('zeros', 'random', 'learned')
  prompt_init_type: zeros

  # Prompt type ('pixel' or 'frequency') -- pixel prompts are trainable pixel tensors; frequency prompts are FFT components
  prompt_type: pixel

prompt:
  # Prompt size (pixels)
  size: 16

  # Number of source classes to map to each target class (depends on label mapping strategy)
  num_source_classes_per_target: 5

  # Prompt learning iterations during initial tuning
  tuning_epochs: 5

hyperparameters:
  # Hyperparameter search space for tuning, including prompt size, input scale, model, label map strategy
  prompt_size_options: [16, 48]
  input_scale_options: [0.5, 1.0, 1.5]
  model_choices: ['resnet18', 'resnext101-ig', 'swin-t', 'clip']
  label_mapping_strategies: ['FreqMap', 'IterMap', 'SemanticMap', 'FullyMap']

dataset:
  # Dataset name (e.g., 'Flowers102', 'ISIC', etc.)
  name: Flowers102

  # Dataset root directory (to be specified at runtime)
  root_dir: null  # to be filled at runtime

  # Train/test split ratios or specific indices
  train_split_ratio: 0.8
  test_split_ratio: 0.2

  # Image resize scale factor (to be selected by hyperparameter tuning)
  scale_factor: 1.0

  # Number of classes in dataset (e.g., 16 for Flowers102)
  num_classes: 16

evaluation:
  # Metrics to compute
  metrics:
    classification_accuracy: true
    iou_score: true
    robustness_score: true

  # Number of test samples to evaluate
  test_samples: null  # default: evaluate full test set

logging:
  # Log directory for saving model checkpoints, visualizations, logs
  log_dir: './logs'

misc:
  # Random seed for reproducibility
  seed: 42

  # Verbosity level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
  verbosity: INFO
```

---

**Note:** Full configuration is available in `planning_config.yaml`
