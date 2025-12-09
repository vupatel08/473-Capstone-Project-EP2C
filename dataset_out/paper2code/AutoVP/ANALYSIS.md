# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

**Logic Analysis: for dataset_loader.py**

---

### **Purpose & responsibilities**

- Implement the `DatasetLoader` class, responsible for:
  - Downloading and loading multiple datasets (classification, segmentation, detection).
  - Handling dataset-specific preprocessing, annotations, and transformations.
  - Providing a method `load_data()` that returns a dictionary containing data suitable for training and evaluation, including images, labels, and optionally masks or bounding boxes.

### **Inputs & outputs**

- **Inputs (via class constructor):**
  - `datasets`: List of dataset names (from a predefined set or configuration), e.g., ['Flowers102', 'ISIC', 'UCF101'].
  - `batch_size`: integer specifying batch size for data loaders.
  - `transforms`: callable or list to apply data augmentations, normalization, resizing, etc.

- **Outputs (via load_data()):**
  - Dictionary of dataset splits (`train`, `val`, `test`).
  - Each split contains datasets possibly wrapped in PyTorch `DataLoader` objects.
  - Each sample: dictionary with `image`, `label`, and possibly `mask` or `bbox`.
  - Should be compatible with training and evaluation roles.

### **Logical steps & structure**

1. **Define dataset mapping:**
   - Map dataset name strings to their dataset class, download links, and any specific configurations/handlings.
   - Use torchvision datasets (e.g., CIFAR10, CIFAR100, Flowers102, StanfordCars, GTSRB, etc.).
   - For custom datasets (e.g., ISIC), define dataset classes or use existing sources; may require custom download and annotation handling.

2. **Dataset download & initialization:**
   - For each dataset in `datasets`, check if dataset exists locally (`root_dir`).
   - If not, download and prepare datasets.
   - For each dataset:
     - Instantiate dataset object with proper transforms.
     - For segmentation/detection datasets, load masks or bounding box annotations.
     - Store train, val, test splits accordingly.

3. **Data preprocessing & transformations:**
   - Resize images according to the `scale_factor` parameter (from config/yaml).
   - Apply data augmentation transforms:
     - Resize, normalize (mean/std) matching backbone normalization.
     - Optional augmentations (horizontal flip, crops, etc.).
   - For segmentation: resize masks accordingly.
   - For detection: prepare bounding boxes, resize coordinates if needed.

4. **DataLoader creation:**
   - Wrap datasets into `torch.utils.data.DataLoader`.
   - Use `batch_size` from config.
   - Configure shuffling for training, no shuffling for test/validation.
   - Use multiple workers if desired to speed up loading.

5. **Return value:**
   - Return a dictionary, e.g.,
     ```python
     {
       'train': train_loader,
       'val': val_loader,
       'test': test_loader,
       'datasets': {dataset_name: dataset_object}
     }
     ```
     or possibly raw datasets if loaders are to be created externally.

6. **Dataset-specific handling:**
   - Use different classes or functions for datasets with:
     - **Classification** (standard image datasets).
     - **Segmentation** (with masks).
     - **Detection** (with bounding boxes).
   - Handle dataset-specific annotation formats.

### **Implementation notes**

- **Dataset classes**:
  - Use `torchvision.datasets` for standard datasets.
  - For datasets like ISIC or specialized segmentation datasets, define custom `Dataset` subclasses:
    - Load images and masks.
    - Resize images and masks to the appropriate scale.
    - Return samples as dicts.

- **Transforms**:
  - Compose transforms for images:
    - Resize to scaled size (using `scale_factor`).
    - Data augmentations if applicable.
    - Convert to tensor.
    - Normalize with ImageNet mean/std if backbone expects that.

- **Regularization**:
  - Ensure data loading is reproducible: set `torch.manual_seed` and dataset seed if needed.

- **Error handling**:
  - Gracefully handle missing datasets or download failures.
  - Confirm dataset paths and existence.

- **Optional**:
  - Allow passing dataset-specific parameters via constructor (e.g., annotation files, split ratios).
  - Provide options for validation split (e.g., 80/20) or use predefined splits.

---

### **Summary**

- **Main Functionality**:
  - For each dataset:
    - Download/load dataset.
    - Apply consistent transformations, resizing based on scale factor.
    - Handle labels, masks, boxes as needed.
    - Wrap datasets into DataLoaders.
- **Return**:
  - Dict containing train/val/test DataLoaders or dataset objects.
  - Dataset-specific handling for images, annotations, masks.
- **Design**:
  - Modular, with dataset-specific methods/functions.
  - Dataset construction should be flexible for classification, detection, segmentation.
  - Incorporate dataset-specific transforms and annotation processing.

This logic analysis provides a comprehensive blueprint for implementing `DatasetLoader` class, ensuring that subsequent coding aligns openly with dataset diversity, experimental consistency, and code modularity.

## evaluation.py

**Evaluation.py Logic Analysis for AutoVP Framework**

---

### 1. Purpose & Role
- This module is responsible for:
  - Loading a trained model (including prompt parameters and label mappings).
  - Running inference on the test datasets.
  - Computing various evaluation metrics (classification accuracy, IoU, robustness).
  - Generating visualizations of prompts and label mappings for interpretability.
  - Returning evaluation results for further analysis or reporting.

---

### 2. Inputs & Dependencies
- **Inputs:**
  - *Model*: The trained `PretrainedModel` instance (with prompts).
  - *Prompt Generator*: The trained prompt tensor(s) (pixel-based or frequency domain).
  - *Dataset*: The test dataset (images, labels, masks, or bounding boxes as applicable).
  - *Label Mapper*: The label mapping object (to convert model source predictions to target labels).
  - *Evaluation Config*: Metrics to compute, number of samples, possibly data splits.
- **Dependencies:**
  - PyTorch for model inference.
  - Datasets for data loader.
  - Metrics calculation functions (accuracy, IoU).
  - Visualization tools (matplotlib, PIL).

---

### 3. Main Components & Workflow

**A. Initialization**
- Load dataset: create a DataLoader for the test split.
- Prepare model:
  - Freeze or set to eval mode.
  - Load prompts and label mapper.
- Initialize metrics accumulators: counters for accuracy, IoU, damages, robustness, etc.
- Optional: preload class names, class embeddings, or masks if needed for visualization.

---

**B. Evaluation Procedure**
- Set model in evaluation mode (`model.eval()`).
- Loop over test DataLoader:
  - For each batch:
    - If dataset is segmentation/detection:
      - Extract images, labels, masks, bounding boxes as relevant.
      - Generate prompts if prompts are not fixed (if adaptively trained).
    - Else:
      - For classification tasks:
        - Input images undergo resizing/scaling as per configuration.
        - Apply prompts (pixel tensors or FFT prompts) to images.
        - Forward through backbone:
          - Obtain raw outputs (predictions/logits).
          - If `label_mapping` strategy is non-parametric (`FreqMap`, `SemanticMap`), apply the static mapping.
          - If trainable (`FullyMap`, `IterMap`), ensure the model has been trained with trained label mapping at this point.
        - For CLIP, use image encoder and label text embeddings to generate final predictions.
    - Collect predicted labels and ground truths.
    - Compute per-batch metrics:
      - Accuracy: count correct predictions.
      - IoU (if segmentation): compute intersection over union.
      - Robustness metrics (if applicable): e.g., accuracy on corrupted images.
  - Accumulate metrics:
    - Total correct / total samples for classification accuracy.
    - Sum of IoU scores across data.
    - Confidence scores for robustness/uncertainty.

**C. Post-processing & Metrics Calculation**
- Finalize metrics:
  - Compute overall accuracy = correct / total samples.
  - Average IoU = sum IoU scores / number of samples.
  - Robustness: perhaps calculate average confidence, or difference between clean and corrupted data.
- Store detailed results:
  - Per-class accuracy.
  - Per-dataset metrics.
  - Ambiguity/confidence distributions.

**D. Visualizations**
- Prompts visualization:
  - Pixel prompts: reshape and display as images.
  - Frequency prompts: inverse FFT and display.
- Label maps visualization:
  - Map class names to semantic embeddings if semantic mapping used.
  - Show predicted vs. true labels, confusion matrices, or class similarity maps.
- Save visualizations optionally to configured directories.

---

### 4. Implementation Details & Considerations

**A. Data Loop**
- Use DataLoader for batch iteration.
- For segmentation/detection, adapt the inference accordingly.
- Use `torch.no_grad()` context to disable gradient computations.

**B. Model Inference**
- For CLIP:
  - Encode images via CLIP image encoder.
  - Compute cosine similarity with class text embeddings.
  - Derive predictions from maximum similarity.
- For ResNet/Swin:
  - Forward images, get logits.
  - Apply label mapping if necessary.
  - Decide final class labels.

**C. Metrics Calculations**
- Accuracy:
  \#correct / total.
- IoU:
  - For segmentation:
    - Obtain predicted masks (via argmax or threshold if needed).
    - Calculate intersection and union with ground truth masks.
    - Accumulate per batch.
- Robustness:
  - Test on corrupted images if provided.
  - Compute accuracy on corrupted datasets.
  - Record confidence scores (max softmax) as needed.

**D. Visualization & Reports**
- Visualize prompts: 
  - Plot pixel prompts as images using `matplotlib`.
  - Plot FFT prompts in magnitude spectrum.
- Mapping visualization:
  - For semantic maps, display cosine similarity heatmaps.
  - For FullyMap, plot weight distributions or confusion matrices.

**E. Final Output**
- Return a dictionary or data class with:
  - All metrics summary.
  - Visualization images (or save file paths).
  - Additional logs/config info.

---

### 5. Edge Cases & Robustness
- Handle missing data or corrupt images gracefully.
- For datasets with masks or bounding boxes, adapt IoU/detection metrics accordingly.
- If label mapping is not applied, default to raw model outputs.
- Ensure that prompts and label mappings are applied consistently during inference.

---

### 6. Hyperparameters & Configuration
- Use the `config.yaml` to:
  - Select dataset, model, metrics.
  - Set batch size, number of samples.
  - Control visualization options.
- Allow overrides via function parameters for flexibility.

---

### 7. Summary
- The `evaluate()` method runs inference with a trained model:
  - Loads dataset.
  - Processes each batch:
    - Applies prompts (pixel or FFT).
    - Scales input images.
    - Runs through backbone.
    - Applies label mapping (if needed).
    - Computes metrics.
  - Collects and averages metrics.
  - Generates visualizations.
- The `visualize_results()` method:
  - Visualizes prompts, label mappings, confusion matrices, etc.
  - Saves or displays plots.

---

With this detailed logic analysis, you are now equipped to implement a comprehensive, modular, and reproducible `evaluation.py` module aligned with AutoVP methodology, datasets, and experimental design.

## label_mapping.py

# Logic Analysis for label_mapping.py

## Overview
This module implements multiple strategies for mapping source model predictions to target dataset labels within the AutoVP framework. The primary class, **LabelMapper**, encapsulates all strategies, with specific subclasses or implemented options for each. This module provides methods for:

- **map()**: convert raw source model predictions into mapped labels based on the selected strategy.
- **update_mapping()**: refine the mapping dynamically during training for strategies like IterMap and FullyMap with weight updates.
- **visualize_mapping()**: produce visual explanations or representations of mapping relationships, such as class correspondences or learned weights.

The strategies include:
- Frequency Mapping (FreqMap)
- Iterative Mapping (IterMap)
- Semantic Mapping (SemanticMap)
- Fully Connected Layer Mapping (FullyMap)

---

## Inputs and Outputs
### Inputs
- **Predictions**: source model outputs, typically logits or class scores, possibly raw or post-softmax.
- **Class Names**:
  - Source class names (list or dict)
  - Target class names (list or dict)
- **Mapping parameters**:
  - For FreqMap: number of classes to map per target class
  - For SemanticMap: class name lists, or class label embeddings
  - For FullyMap: initial weights and biases
  - For IterMap: current mapping, used to update iteratively
- **Embeddings**:
  - For SemanticMap: text embeddings of class names, computed via CLIP text encoder

### Outputs
- **Mapped labels**: tensor or list of predictions aligned to target classes.
- **Updated mapping matrices or parameters**: e.g., the weight matrix for FullyMap.
- **Visualizations**: images or diagrams illustrating class correspondences or weight distributions.

---

## Internal Structure and Control Flow

### 1. Initialization
- The LabelMapper class is instantiated with configuration parameters:
  - *strategy* (string): 'FreqMap', 'IterMap', 'SemanticMap', 'FullyMap'
  - *source_class_names* (list): class names in the source model
  - *target_class_names* (list): label names in target dataset
  - *map_params* (dict): additional parameters such as n_classes_per_target, initial weights, number of source classes, embeddings, etc.

- For **SemanticMap**:
  - Compute embeddings for source class names using CLIP text encoder
  - Compute embeddings for target class names similarly
  - Compute similarities between source and target class embeddings
  - Establish initial class correspondence

- For **FullyMap**:
  - Initialize a linear layer (weights and biases)
  - If weight initialization via semantic info is enabled, assign weights accordingly

### 2. map(predictions)
- **Input**: raw predictions (logits or scores)
- Based on the strategy:
  - *FreqMap*: 
    - Aggregate predictions over training data for each class
    - For each target class, assign the source class that most frequently predicts that target class
    - For each new prediction, assign label according to the established frequency mapping
  - *IterMap*:
    - Call **update_mapping()** at each epoch or iteration to refresh the mapping depending on current model predictions
  - *SemanticMap*:
    - For each predicted class (from source model), find the target class with the highest cosine similarity between embeddings
    - Map source predictions to target labels accordingly
  - *FullyMap*:
    - Pass source logits through the learned linear layer
    - Convert final logits to predicted class labels in target space

### 3. update_mapping()
- Only applicable for **IterMap** and **FullyMap**:
  - *IterMap*:
    - Recompute frequency-based mapping using current model predictions on training target data
    - Update internal mapping matrix or class correspondence
  - *FullyMap*:
    - Update linear layer weights, possibly with a special initialization based on semantic similarity or learned via training
  - This process usually involves:
    - Forward passing input images (or training set) through the source model + current prompts
    - Gathering prediction statistics
    - Re-establishing or refining class correspondences

### 4. visualize_mapping()
- For **FreqMap**:
  - Show class counts or confusion matrices
- For **SemanticMap**:
  - Plot class embeddings, similarities, or class naming correspondence
- For **FullyMap**:
  - Visualize weight matrices as heatmaps or per-class weight distributions
- For **IterMap**:
  - Show evolution of class mappings over iterations

---

## Implementation Details and Edge Cases
- **Semantic Map**:
  - Requires explicit class name lists and CLIP text encoder
  - Robust to semantic mismatch — might require thresholding
- **FreqMap**:
  - Needs access to training data predictions; must store prediction counts for each class
  - Handle classes with limited data or heavily skewed data
- **IterMap**:
  - Needs to be called at each epoch or iteration
  - Mapping might fluctuate; limit the number of source classes mapped per target
- **FullyMap**:
  - Parameters W and b are trainable; initialize carefully (e.g., identity or semantic-based)
- **General**:
  - All strategies should return labels in a consistent format (tensor or list of indices)
  - Properly handle unmatched classes (e.g., in semantic mapping, if cosine similarity is below threshold)
  
---

## Summary
- **Class**: LabelMapper
- **Main attributes**:
  - strategy
  - class name lists
  - mapping matrices or weights
  - embeddings (for semantic mapping)
- **Methods**:
  - **map(predictions)**: returns target class predictions
  - **update_mapping()**: updates internal mapping based on latest model output
  - **visualize_mapping()**: produces visual explanations

This structure ensures modularity, flexibility, and alignment with the paper's approach. Properly implementing these will support AutoVP's joint optimization and automated strategy selection.

---

**Note**: This logic analysis assumes access to class name info, trained CLIP embeddings, and training data predictions, which are standard for this module.

## main.py

# Main.py: Logical Analysis and Design for AutoVP Reproduction

This document provides a detailed, step-by-step logic analysis for implementing `main.py`, the entry point of the AutoVP framework, guided by the provided paper, design scheme, task list, and configuration file (`config.yaml`). The goal is to establish a clear, structured flow that initializes the necessary components, manages hyperparameter tuning, conducts training, and finally performs evaluation and visualization — all in a modular, reproducible manner.

---

## 1. High-Level Objectives and Workflow

- **Read and parse configuration parameters**.
- **Load and prepare datasets** based on dataset name and hyperparameters.
- **Initialize pre-trained backbone models** with frozen weights.
- **Set up prompts**: trainable pixel or frequency prompts, with specified size and init strategy.
- **Define label mapping mechanism**: FreqMap, IterMap, SemanticMap, or FullyMap.
- **Conduct hyperparameter search**:
  - Iterate over all combinations in the defined search space.
  - For each configuration:
    - Initialize prompts, models, label maps as per settings.
    - Run a short training (2-5 epochs) with early stopping.
    - Record validation/test accuracy.
- **Identify the best hyperparameter configuration** based on validation/test scores.
- **Retrain prompts and models** using the optimal configuration (full epochs).
- **Perform final evaluation** using the best trained model:
  - Compute accuracy, IoU, robustness metrics.
  - Visualize prompts, label mappings, and results.
- **Log, save checkpoints and results** for reproducibility and analysis.

---

## 2. Step-by-Step Logical Breakdown

### 2.1 Import dependencies and set reproducibility
- Import standard libraries: `torch`, `numpy`, `os`, `random`.
- Import custom modules: dataset loader, model handler, prompt manager, label mapping, trainer, tuner, evaluator.
- Set seed (from `config.misc.seed`) for reproducibility.
- Configure logging verbosity with `logging` module, set to `config.logging.verbosity`.
  
### 2.2 Load dataset
- Parse dataset name (`config.dataset.name`), directory (`config.dataset.root_dir`).
- Use `DatasetLoader` class:
  - Pass dataset name, root directory, split ratios, resize scale (`config.dataset.scale_factor`).
  - Load training, validation, test datasets.
  - Apply appropriate transforms (resizing, normalization), depending on dataset type (classification, segmentation, detection).
- Save datasets into variables for subsequent use.

### 2.3 Initialize the pre-trained backbone model
- Retrieve model name (`config.model.backbone`) from config.
- Initialize `PretrainedModel` instance:
  - Load correct pre-trained weights.
  - Set `freeze=True`.
- Check if model supports extracting features and text embeddings (for CLIP semantic map).

### 2.4 Prepare prompts
- Read prompt size (`config.model.prompt_size`) and type (`config.model.prompt_type`).
- Initialize contrastive or pixel prompts:
  - For `prompt_type == 'pixel'`:
    - Create a trainable tensor of shape `[prompt_size, channels, 1, 1]`.
    - Initialize based on `prompt_init_type` (zeros, random, learned).
  - For `prompt_type == 'frequency'`:
    - Initialize FFT coefficients (complex tensor), zero or random.
- Encapsulate in `PromptGenerator` class:
  - Provide `get_prompt()`, `update()`, `visualize()` methods.

### 2.5 Set up label mapping strategy
- Select mapping strategy (`FreqMap`, `IterMap`, `SemanticMap`, `FullyMap`) based on hyperparameters.
- Instantiate corresponding `LabelMapper`:
  - For `SemanticMap`:
    - Use CLIP text encoder for class name embeddings.
  - For `FullyMap`:
    - Initialize as linear layer (possibly with weight init based on class names).
  - For `FreqMap` and `IterMap`:
    - Initialize count matrices, mapping dictionaries.
- Keep in mind the number of source classes per target class (`num_source_classes_per_target`).

### 2.6 Define hyperparameter search space
- Enumerate all combinations over:
  - Prompt size options (`16`, `48`)
  - Input scale options (`0.5`, `1.0`, `1.5`)
  - Model choices (`resnet18`, `resnext101-ig`, `swin-t`, `clip`)
  - Label map strategies (`FreqMap`, `IterMap`, `SemanticMap`, `FullyMap`)
  - Number of source classes mapped (if applicable)
- Use grid search with early stopping:
  - For each configuration:
    - During initial phase, train prompts for 2-5 epochs.
    - Record validation/test accuracy.
  - Retain top configurations (e.g., top 2) for full training.

### 2.7 Conduct hyperparameter search
- Loop over configurations.
- For each:
  - Initialize model, prompt, label mapper with current hyperparameters.
  - Run a small number of training epochs:
    - For each epoch:
      - For each batch:
        - Resize images according to scale.
        - Generate prompts.
        - Forward pass through backbone.
        - Map outputs via selected label map.
        - Compute loss, backpropagate prompts.
        - Update prompts.
      - Evaluate on validation set; apply early stopping if no improvement.
  - Save the best validation/test score for configuration.

### 2.8 Select and retrain with the best hyperparameters
- Pick configuration with the highest validation/test accuracy.
- Re-initialize prompts, models, label map with the selected hyperparameters.
- Train fully over all epochs specified (`config.training.epochs`), with learning rate, weight decay, optimizer settings.
- Save the final model checkpoints.

### 2.9 Final evaluation
- Load the fully trained best model and prompt.
- Evaluate on the test set:
  - Compute classification accuracy.
  - If segmentation/detection, compute IoU.
  - Compute robustness metrics with corrupted datasets if applicable.
- Generate visualizations:
  - Visual prompts overlay.
  - Label mapping diagrams.
  - Frequency analysis if needed.
- Store results in logs, save models, plots, and detailed metrics.

### 2.10 Log, Save & Cleanup
- Log all hyperparameter configurations, metrics.
- Save the best checkpoint.
- Save logs / visualization outputs to `log_dir`.
- Finish with a summary report.

---

## 3. Additional Considerations & Checks
- **Dataset splits**: Ensure reproducibility, consistent splits.
- **Model compatibility**: Make sure models support feature extraction and embeddings.
- **Prompt updates**: Implement gradient updates with appropriate optimizer (Adam/SGD).
- **Check hardware constraints**: batch size, memory management for large datasets/models.
- **Reproducibility**: Set random seeds across `torch`, `numpy`, and Python's `random`.

---

## 4. Summary of Modular Components in `main.py`
- **Configuration**:
  - Load and override defaults.
- **Dataset Loader**:
  - Class instance to load datasets per setting.
- **Model Handler**:
  - Instantiate pre-trained models.
- **Prompt Module**:
  - Instantiate prompts, optional FFT prompts.
- **Label Mapper Module**:
  - Instantiate label mapping strategy.
- **Hyperparameter Tuner**:
  - Run grid search with early stopping.
- **Training Loop**:
  - For each configuration: prompt optimization training.
- **Evaluation and Visualization**:
  - Final metrics, plotting.

---

This detailed logic analysis ensures that `main.py` will orchestrate all necessary steps following the AutoVP methodology, enabling systematic, reproducible, and extendable implementation consistent with the paper's experimental design.

## model.py

**Logic Analysis for `model.py` — Implementation of `PretrainedModel` class**

---

### **Objective:**

Implement a `PretrainedModel` class that:
- Initializes pre-trained vision models (ResNet, Swin Transformer) or vision-language model (CLIP).
- Provides methods for forwarding images to obtain predictions.
- Provides methods for extracting feature embeddings (images, text).
- Keeps model weights frozen during prompts training.
- Supports extraction of CLIP text embeddings for semantic label mapping.

---

### **Key Components & Requirements:**

1. **Model Initialization:**
   - Load pre-trained models based on configuration (`backbone` parameter).
   - For `resnet18`, `resnext101-ig`, `swin-t`:
     - Use `torchvision.models` or relevant pretrained weights.
   - For `clip`:
     - Use HuggingFace `transformers` library (e.g., `CLIPModel`, `CLIPProcessor`).
2. **Freeze Model Weights:**
   - Set `requires_grad = False` for all parameters during initialization.
3. **Methods to Implement:**
   - `forward(x: Tensor) -> Tensor`:
     - Given input images, output the raw logits (predictions).
   - `extract_features(x: Tensor) -> Tensor`:
     - Output feature embeddings (e.g., penultimate layer features).
   - `extract_text_embeddings(class_names: List[str]) -> Tensor`:
     - For CLIP: tokenize class names, encode text, output normalized embeddings.
4. **Supporting Capabilities:**
   - For CLIP:
     - Multiple modes:
       - Use image encoder for features.
       - Use text encoder for class label semantics.
     - Use appropriate tokenizer (`CLIPProcessor`) for text.
   - For vision-only models:
     - Use model’s forward pass, optionally obtaining features from intermediate layers.
     - Possibly modify the model to output features (depending on model architecture).

---

### **Implementation Details:**

#### **1. Initialization**

- **Input Parameters:**
  - `model_name`: name string, e.g., `'resnet18'`, `'resnext101-ig'`, `'swin-t'`, `'clip'`.
  - `freeze`: boolean, default `True`.
  
- **Logic:**
  - Based on `model_name`, load the respective pre-trained model:
    - For torchvision models: use `torchvision.models`.
    - For CLIP: use `transformers` library.
  - For CLIP:
    - Load both `CLIPModel` and tokenizer (`CLIPProcessor`).
    - Keep text encoder and image encoder separately accessible.
  - Set all model parameters `requires_grad = False` to freeze weights.
  - Save model components as attributes (`self.model`, `self.text_encoder`, `self.tokenizer`, etc.).

---

#### **2. Methods**

- **`forward(x)`**:
  - For vision models:
    - Just pass input tensor `x` through `self.model`.
    - Obtain logits or prediction output.
  - For CLIP:
    - Pass `x` through image encoder.
    - Obtain similarity scores or logits.
    - (Note: The main prompt training uses features, so may need to output raw features before softmax.)
  
- **`extract_features(x)`**:
  - For vision models:
    - Forward `x` up to the penultimate layer (e.g., features before final classification layer).
        - This might require sub-classing the models or hooks if using torchvision models.
    - For CLIP:
      - Use `image_encoder` and output normalized embeddings.
  - Return feature tensor suitable for further processing or similarity computations.

- **`extract_text_embeddings(class_names)`**:
  - Tokenize class names with CLIP tokenizer.
  - Encode text using `text_encoder`.
  - Normalize embeddings (unit vectors).
  - Store and return tensor of class embeddings.
  - **Note:** Only applicable if backbone is CLIP.

---

#### **3. Handling Different Models**

- *ResNet / Swin:*
  - Use standard torchvision models with pretrained weights (`torchvision.models.ResNet18(pretrained=True)`).
  - Identify how to extract features:
    - ResNet: output features from `avgpool` or penultimate layer.
    - Swin: output features from appropriate transformer tokenization layer; may require custom hooks or subclassing.
- *CLIP:*
  - Load model: `transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")`.
  - Tokenizer: `CLIPProcessor` for class label text.
  - Generating text embeddings involves tokenizing class names and encoding.

---

### **Additional Considerations:**

- **Model Device:**
  - Use GPU if available; ensure model and tensors are on the same device.
- **Normalization:**
  - For similarity computations, normalize embeddings (unit vectors).
- **Parameter Freezing:**
  - Fully set `requires_grad=False` for all parameters after loading.
- **Supporting multiple models:**
  - Ensure modularity: use dictionary mapping model names to loading procedures.

---

### **Pseudo-logic for class `PretrainedModel`:**

```python
class PretrainedModel:
    def __init__(self, model_name, freeze=True):
        # Load model
        if model_name == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.eval()
            # Setup hooks if needed for feature extraction
        elif model_name == 'resnext101-ig':
            # Load torchvision model if available, or via timm
            self.model = timm.create_model('resnext101_32x8d', pretrained=True)
            self.model.eval()
        elif model_name == 'swin-t':
            self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
            self.model.eval()
        elif model_name == 'clip':
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
        else:
            raise ValueError('Unsupported model name')

        # Freeze parameters if `freeze` is True
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Additional setup for extracting features if needed
        # For vision models:
        # - Use hooks or subclass to get intermediate features
        # For CLIP:
        # - Use `self.model.get_image_features()` for image features
        # - Use `self.model.get_text_features()` for text features

    def forward(self, x):
        # input: tensor of shape (batch_size, C, H, W)
        if hasattr(self, 'processor'):
            # For CLIP, wrap in processor
            return self.model.get_image_features(pixel_values=x)
        else:
            # For ResNet/Swin, directly forward
            return self.model(x)

    def extract_features(self, x):
        # Return features as needed
        if hasattr(self, 'processor'):
            # CLIP
            features = self.model.get_image_features(pixel_values=x)
            return features / features.norm(dim=-1, keepdim=True)
        else:
            # Vision models: use feature extractor
            # Might require hooks or custom sub-models
            return self._extract_vision_features(x)
        
    def extract_text_embeddings(self, class_names):
        # For CLIP only
        if not hasattr(self, 'processor'):
            raise RuntimeError('Text embeddings are only supported for CLIP')
        # Tokenize class names
        inputs = self.processor(text=class_names, return_tensors='pt', padding=True)
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Encode
        text_features = self.model.get_text_features(**inputs)
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
```

**Note:** 
- Actual extraction of features from vision models may require hooks or subclassing.
- For CLIP, `get_image_features()` and `get_text_features()` are standard APIs.
- For custom architectures, implement feature extraction accordingly.

---

### **Summary:**

- Initialize backbone based on `model_name`.
- Keep model frozen (set `requires_grad=False`).
- Provide `forward()` for predictions.
- Provide `extract_features()` for embedding extraction.
- Provide `extract_text_embeddings()` for CLIP semantics.
- Ensure correct device placement.
- Modular and extensible for additional models.

This detailed logic forms the basis for implementing `model.py` ensuring it aligns with the paper's description and the broader AutoVP pipeline.

## prompt_module.py

## Logic Analysis for `prompt_module.py`

### Overview:
`prompt_module.py` contains the implementation of the `PromptGenerator` class, responsible for managing the prompts used in AutoVP framework. Prompts can be pixel-based (direct pixel tensors) or frequency-based (FFT components). The class must provide methods to generate current prompts (`get_prompt()`), update prompts based on gradients (`update()`), and visualize prompts (`visualize()`).

---

### 1. Initialization (`__init__`)

- **Inputs**:
  - `prompt_size` (int): size of the prompt, e.g., 16 pixels.
  - `prompt_type` (str): `'pixel'` or `'frequency'`.
  - `prompt_init_type` (str): `'zeros'`, `'random'`, or `'learned'`.
  - `input_channels` (int): typically 3 for RGB images.
  - `initial_prompt` (optional): a tensor to initialize prompts; if None, initialize based on `prompt_init_type`.

- **Behavior**:
  - **Mode: `'pixel'`**
    - Create a trainable tensor of shape `[channels, prompt_size, prompt_size]`.
    - Initialization:
      - `'zeros'`: all zeros.
      - `'random'`: initialize with small random values.
      - `'learned'`: same as `'random'` or using specific schemes, but likely treated as trainable pixel tensor.
  - **Mode: `'frequency'`**
    - Initialize FFT coefficients:
      - A real-valued tensor of shape `[channels, prompt_size, prompt_size]` or `[channels, T]` in FFT domain, where `T` is total FFT components.
    - Initialization:
      - `'zeros'`: coefficients are zero.
      - `'random'`: coefficients are random small values.
    - Note: in FFT prompts, actual prompts are reconstructed only when needed (via inverse FFT).

- **Attributes to store**:
  - `self.prompt_type`: `'pixel'` or `'frequency'`.
  - `self.prompt_tensor`: torch parameter for prompts.
  - Additional attributes for FFT domain: store real and imaginary parts or FFT coefficients.

---

### 2. Method: `get_prompt()`

- **Purpose**:
  - Return the current prompt tensor that can be added to images during forward pass.

- **Behavior**:
  - If `'pixel'`:
    - Return `self.prompt_tensor` directly.
  - If `'frequency'`:
    - Perform inverse FFT (`torch.ifft` or `torch.fft.ifft2`) on current FFT coefficients.
    - Output the reconstructed spatial prompt tensor with shape `[channels, prompt_size, prompt_size]`.
    - Ensure the output is real-valued (or take real part).
  
- **Implementation notes**:
  - For frequency prompts, store FFT coefficients separately; reconstruct each time `get_prompt()` is called.
  - For efficiency, consider caching the FFT output if prompts do not update frequently.

---

### 3. Method: `update(grads)`

- **Purpose**:
  - Update the prompts’ parameters using gradients obtained from backpropagation.

- **Input**:
  - `grads`: gradient tensor(s) corresponding to the prompts.
  
- **Behavior**:
  - Use an optimizer step:
    - Use `optimizer.step()` on `self.prompt_tensor`.
    - Alternatively, apply manual gradient updates:
      - `self.prompt_tensor.data -= learning_rate * grads`.
    - For `'frequency'` prompts:
      - If `self.prompt_tensor` holds real FFT coefficients:
        - Update real and imaginary parts separately if needed.
    - Zero gradients after update if manual.

- **Notes**:
  - The update step occurs after a backward pass during training.
  - The `PromptGenerator` class should store its own optimizer or allow external update.

---

### 4. Method: `visualize()`

- **Purpose**:
  - Visualize current prompt tensor as an image for interpretability.

- **Behavior for `'pixel'` prompts**:
  - Convert `self.prompt_tensor` (shape `[channels, p, p]`) into a displayable image:
    - Normalize or clamp pixel values to [0, 1] or [0, 255].
    - Use matplotlib or Pillow to generate visualization.
    - Return the image object or save it.

- **Behavior for `'frequency'` prompts**:
  - Reconstruct spatial prompt:
    - Perform inverse FFT.
    - Normalize as above.
  - Visualize the same way as `'pixel'`.

- **Implementation notes**:
  - For visual interpretation, optionally plot the prompt as it appears (e.g., heatmaps or raw images).
  - Visualize in `matplotlib.pyplot.imshow()` or similar.

---

### 5. Additional Considerations:

- **Device handling**:
  - Ensure prompts are on the same device (cpu or cuda).
  - Initialize prompts as `torch.nn.Parameter`.
- **Gradient management**:
  - Prompts are trainable parameters, so `requires_grad=True`.
  - During training, prompt optimizer updates prompts based on computed gradients.
  
- **Prompt size and shape**:
  - For `'pixel'`, prompt tensor shape: `[channels, p, p]`.
  - For `'frequency'`, shape might be `[channels, p, p]`, representing FFT coefficients.
  
- **FFT coordinate system**:
  - Use `torch.fft.fft2` / `torch.fft.ifft2` (PyTorch 1.13.1+ syntax).
  - Real coefficients stored separately, or complex as a single tensor.

- **Prompt regularization** (optional): smoothness, sparsity, etc., can be implemented, but not essential in core logic.

### 6. Usage flow:
- During training:
  1. Call `get_prompt()` to retrieve prompt tensor.
  2. Add it to the image input before passing into the backbone.
  3. After backward pass, call `update(grads)` to update prompts.
- During visualization:
  - Call `visualize()` to obtain prompt as an image.

---

### Summary:
- The class must support flexible prompts in pixel or frequency domain.
- Efficient forward reconstruction for frequency prompts via inverse FFT.
- Proper parameter management for gradient updates.
- Visualization routines to interpret learned prompts.
- Device management and regularization options for prompts (if extended).

This comprehensive analysis guides the development of `PromptGenerator` class, ensuring it aligns with the described framework and supports all necessary functionalities for AutoVP.

## trainer.py

# Logic Analysis for trainer.py

The `trainer.py` module implements the core training process responsible for optimizing the visual prompts, possibly alongside updating label mappings during limited tuning epochs. Its main class, `Trainer`, coordinates all components—dataset loading, feature extraction via the frozen backbone, prompt updates, label mapping, and hyperparameter adjustment—under the supervision of a training loop with early stopping based on validation performance.

Below is a detailed step-by-step logical breakdown and design considerations for implementing the `Trainer` class and its methods:

---

### 1. **Class Initialization (`__init__`)**
- **Inputs:**
  - `model`: an instance of `PretrainedModel`, initialized with the backbone name, with weights frozen, and capable of extracting features or predictions.
  - `prompts`: an instance of `PromptGenerator`, providing methods to retrieve current trainable prompts and visualize them.
  - `dataset`: a dictionary or list of datasets split into training and validation sets, possibly including data loaders or raw data. Should include images, labels, and any annotations (for segmentation/detection).
  - `label_mapper`: an instance of `LabelMapper`, facilitating label type conversion (e.g., via FreqMap, SemanticMap, etc.).
  - `optimizer`: a PyTorch optimizer (e.g., Adam) to update only the prompts (and possibly label mapping parameters).
  - `lr_scheduler` (optional): a scheduler for learning rate decay, such as cosine or step decay.
  - `early_stop_patience`: integer defining how many epochs to wait without improvement before stopping early.
  - `training hyperparameters`: total epochs, batch size, learning rate, etc.

- **Responsibilities:**
  - Store all inputs as attributes.
  - Prepare data loaders for training and validation datasets based on batch size.
  - Initialize training state: epoch counter, best validation accuracy, early stopping counter.
  - Ensure model weights are frozen; only prompts and potentially label mapper parameters are trainable.
  - Set up any additional training utilities (e.g., logging, checkpoint paths).

---

### 2. **Method: `train()`**
- **Purpose:**
  - Execute the main training loop for the specified number of epochs or until early stopping criterion is met.
  
- **Workflow per epoch:**
  1. **Set model/train mode** (`model.eval()` for backbone, prompts in train mode if prompts are explicitly marked trainable).
  2. **Iterate over batches** from the training data loader:
     - Fetch batch images and labels.
     - **Resizing:**
       - Resize images according to the current scale factor (from hyperparameters or learned if enabled).
       - Use differentiable resizing if input scale is a learnable parameter.
     - **Prompt Application:**
       - Generate prompts (`prompt.get_prompt()`).
       - Apply prompts:
         - For pixel prompts: add prompt tensors (e.g., padding or overlay) to images as per Eq. 1.
         - For frequency prompts: modify in frequency domain appropriately (if applicable).
       - Generate `prompted_images`.
     - **Feature Extraction:**
       - Forward `prompted_images` through `model`:
         - For classification: get logits or prediction scores.
         - For segmentation/detection: get pixel-wise outputs or bounding boxes (not primary here).
     - **Predict labels:**
       - For classification:
         - `preds = model.forward(prompted_images)` — output logits.
       - For segmentation/detection:
         - Handle accordingly.
     - **Output Label Mapping:**
       - Use `label_mapper.map()` to convert model’s source labels (or logits) to target domain labels.
       - For trainable label mappings:
         - If `FullyMap`, compute output logits via trained linear layer.
         - If `FreqMap`, `IterMap`, or `SemanticMap`, apply fixed or iterative mapping rules.
     - **Loss Computation:**
       - Compute cross-entropy loss between mapped predictions and ground truth labels.
     3. **Backward pass:**
       - Zero gradients.
       - Backpropagate loss only for prompts, label mapper parameters (excluding backbone).
       - Update optimizer (`optimizer.step()`).
       - Clip or constrain prompt outputs if necessary.
     4. **Update metrics** for training (accuracy, etc).
  3. **Validation Evaluation:**
     - Run model on validation dataset without gradient updates.
     - Compute validation accuracy, IoU, robustness measures.
  4. **Early stopping check:**
     - Compare current validation accuracy to previous best.
     - If improvement:
       - Save current prompts, label mapping, and model checkpoint.
       - Reset early stop counter.
     - Else:
       - Increment early stop counter.
       - If counter exceeds `early_stop_patience`, break training early.
  5. **Learning Rate Scheduling:**
     - Update LR scheduler if provided.
  6. **Logging:**
     - Log epoch metrics (train/val accuracy, loss).
     - Save model checkpoints periodically or when improvement is observed.

---

### 3. **Method: `save_checkpoint()` and `load_checkpoint()`**
- Handle persistence of:
  - Prompt tensors.
  - Label mapping parameters.
  - Optimizer state.
  - Validation metrics.
- **Purpose:**
  - Enable restart, early stopping, or final evaluation with the best parameters.

---

### 4. **Supporting Utility Methods**
- **`_compute_loss()`**:
  - Calculates cross-entropy between predictions and true labels after label mapping.
- **`_evaluate()`**:
  - Runs inference on validation/test set, returning metrics like accuracy, IoU, robustness scores.
- **`_update_early_stop()`**:
  - Manages early stopping logic based on validation accuracy.
- **`_resize_images()`**:
  - Differentiable resizing function—if input scale is learned—using `kornia.geometry.transform()` or similar.
- **`_apply_prompt()`**:
  - Handle applying pixel prompts or frequency prompts.
- **`_log_metrics()`**:
  - Output progress information to console, logs, or visualization tools.

---

### 5. **Hyperparameter and Epoch Management**
- **Initialization:**
  - Load configuration parameters from YAML or input dict.
  - Compute total training steps based on dataset size and batch size.
- **Epoch loop:**
  - For each epoch, perform training, validation, early stopping checks.
- **Post-training:**
  - Retrieve and load the best prompt, label mapping, and model state.
  - Optionally, re-train prompts on full dataset with final hyperparameters.

---

### 6. **Key Implementation Notes**
- **Prompt management:**
  - `PromptGenerator.get_prompt()` returns current trainable tensor.
  - Prompts are updated via gradients; only prompts and label mapping parameters are optimized.
- **Gradient masking:**
  - Backbone frozen: ensure only prompts and label mapping parameters are in optimizer.
- **Differentiable resizing:**
  - Use `kornia.geometry.transform()` for input scale and prompt size.
- **Strategies to handle different label maps:**
  - FreqMap and IterMap are fixed during training.
  - FullyMap’s linear layer is trained; initialize weights according to semantic info if WI used.
- **Training stability:**
  - Use gradient clipping if necessary.
  - Occasionally visualize prompts for debugging.

---

### 7. **Handling Multiple Datasets & Multi-task Settings**
- Adapt `train()` to handle dataset dicts with multiple data modalities.
- Possibly maintain separate Stats for each dataset if needed.
- For segmentation/detection tasks, modify data loader and loss functions accordingly.

---

### 8. **Workflow Integration & Final Notes**
- Integrate `trainer.py` with `main.py`, passing configuration and datasets.
- Ensure all assets (e.g., checkpoints) are saved appropriately.
- Encapsulate training loop with flexibility for different prompt and label mapping strategies.
- Keep code compatible with GPU acceleration and to facilitate debugging/visualization.

---

# Summary
The `Trainer` class orchestrates the optimization of prompts and label mappings while keeping the backbone frozen, guided by hyperparameters that are tuned via an external tuner. It supports early stopping based on validation accuracy, manages data loading, performs differentiable input resizing, and applies the selected representation learning strategy. It logs progress, checkpoints the best models, and can evaluate multiple metrics including robustness.

This detailed logical framework provides a clear and structured basis for implementing the `trainer.py` module aligned with the paper’s methodology and the overall code architecture.



## tuner.py

# Logic Analysis for `tuner.py`  
*Purpose:*  
Implement the `HyperparameterTuner` class to perform systematic hyperparameter search (grid or Bayesian) over specified parameter spaces for AutoVP, leveraging early stopping, and selecting the best configuration based on evaluation metrics.

---

## 1. **Inputs and Initializations**  
- **Configuration Parameters:**  
  - Define search space for hyperparameters: `prompt_size_options`, `input_scale_options`, `model_choices`, `label_mapping_strategies`  
  - Total `max_trials` or iterations (e.g., grid search over the cartesian product or Bayesian approach).  
- **Evaluation Metric:**  
  - Use validation/test accuracy as the primary metric to compare configurations.  
- **Resources:**  
  - Use Ray Tune (or equivalent) for distributed hyperparameter tuning, early stopping, trial management.  
- **Initial State:**  
  - Keep track of best configuration and best validation accuracy achieved.

---

## 2. **Hyperparameter Search Space Definition**  
- Construct a composite search space as the cartesian product of all options, or define an explicit search space dict suitable for Ray Tune.  
- For each hyperparameter:  
  - `prompt_size`: integers such as [16, 48]  
  - `input_scale`: continuous or discrete list, e.g., [0.5, 1.0, 1.5]  
  - `model_choice`: categorical in ['resnet18', 'resnext101-ig', 'swin-t', 'clip']  
  - `label_mapping_strategy`: categorical in ['FreqMap', 'IterMap', 'SemanticMap', 'FullyMap']

---

## 3. **Trial Function (Objective Function Implementation)**  
- For each trial:  
  - **Sample hyperparameter values**: from the defined search space.  
  - **Configure the model and prompts** based on these values.  
  - **Initialize model & prompts**: load pre-trained backbone consistent with selection (`model_choice`), initialize prompts as per `prompt_init_type`.  
  - **Dataset Preparation**:  
    - Load dataset specified in configuration.  
    - Resize images according to sampled `input_scale` and `prompt_size`.  
  - **Set up Prompt Module**: create prompts (pixel or frequency domain).  
  - **Set up Label Mapper**: instantiate chosen mapping strategy (FreqMap, IterMap, etc.).  
  - **Train**:  
    - Run initial prompt tuning for `tuning_epochs` (e.g., 2-5).  
    - Use a validation set or test set for evaluation at the end of tuning.  
  - **Validation & Early Stopping**:  
    - During tuning, monitor validation accuracy after each epoch.  
    - If validation accuracy hasn't improved over `early_stop_patience` epochs, terminate early.  
  - **Result Recording**:  
    - Save the validation accuracy, final hyperparameters, model state if needed, and logs.

---

## 4. **Early Stopping & Iterative Trials**  
- Use Ray Tune's `ASHAScheduler` or similar to implement early stopping:  
  - Assign each trial a `score_attr` for validation accuracy.  
  - Retain top `n` trials (e.g., top 2).  
  - Stop bad-performing trials early to save compute.

---

## 5. **Hyperparameter Optimization Strategy**  
- **Grid Search**:  
  - Iterate over all combinations systematically; feasible if search space is small.  
- **Bayesian/Population-Based Search**:  
  - Use Ray Tune's advanced schedulers (`BayesOptSearch`, `HyperOpt`) for larger space; less likely needed if the paper specifies grid search.  
- **Sampling & Parallelization**:  
  - Launch multiple trials in parallel for efficiency.  
  - Use worker pools if needed.

---

## 6. **Result Aggregation & Best Config Retrieval**  
- After the search completes, extract the hyperparameters with the highest validation/test accuracy.  
- Record the best configuration details (prompt size, scale, model, mapping, etc.).

---

## 7. **Final Training with Best Hyperparameters**  
- Load dataset and model again.  
- Configure prompts and label mapping strategy with the best hyperparameters.  
- Perform full training on the dataset (not just tuning epochs) for accuracy convergence (~50 epochs).  
- Save or return the fully trained model state if needed.

---

## 8. **Logging & Reporting**  
- Log results of all trials: hyperparameters, validation metrics, early stop info.  
- Generate a summary report of best configuration and corresponding metrics.

---

## 9. **Edge Cases & Special Considerations**  
- Handle invalid combinations (e.g., semantic map without class labels).  
- Account for resource constraints (limit trials).  
- Make sure to seed the randomness for reproducibility when sampling hyperparameters.  
- Confirm the modules `main.py` will invoke the `run()` method, which internally calls the tuning process.

---

## 10. **Summary of Critical Steps in `tuner.py`**  
**a. Setup search space and scheduler**  
**b. Define trial function** (model setup, dataset loading, prompt & mapping initialization, training, validation)  
**c. Launch search process** (e.g., Ray Tune `tune.run()`) with defined trial function, scheduler, and stopping criteria  
**d. Collect best configuration and metrics**  
**e. Return or save the best hyperparameters for subsequent full training**

---

This detailed logic analysis guides precise implementation of the `HyperparameterTuner` class, ensuring an efficient and robust exploration of the AutoVP hyperparameter space aligned with the experimental framework described in the paper.

