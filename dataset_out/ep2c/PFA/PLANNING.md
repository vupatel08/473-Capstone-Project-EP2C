# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! To create a comprehensive reproduction plan for the proposed scribble-supervised semantic segmentation method, I will organize it into key sections that address the methodology, datasets, experimental setup, and hyperparameters. 

---

## 1. Overall Approach and High-level Workflow
Before diving into details, understand the entire pipeline:
- **Backbone encoder** (Mix Transformer, Segformer backbone—ResNet101 is discussed for comparison).
- **Decoder** (multi-level decoder producing segmentation maps).
- **Prototype extraction** (local and global prototypes).
- **Prototype memory bank management** (local prototypes per batch, global prototypes across training).
- **Prototype-based feature augmentation** (using local and global prototypes separately or combined).
- **Loss functions** (partial cross entropy, consistency loss) and training schedule that incorporates prototype utilization phases (warm-up, partial, full prototype use).
- **Inference** (with or without prototype augmentation, with class label guidance or not).

---

## 2. Data Requirements & Dataset Preparation
- **Dataset**: Use PASCAL VOC 2012 with scribble annotations.
  - The original images (12,031 total), with training set (10,582 images) and validation set (1,449 images).
  - Annotations: Scribble labels at pixel level, indicating partial labels for some regions.
- **Preprocessing & Data Augmentation**:
  - Random scaling (0.5–2.0).
  - Random rotation (-10° to 10°).
  - Random horizontal flipping.
  - Gaussian blur (probability-based or uniform).
  - Cropping to 512×512.
- **Implementation notes**:
  - Convert scribble labels into pixel-wise label images (categorical class IDs for supervised pixels, ignored regions for unlabelled pixels).
  - Maintain a separate list or dictionary of scribble annotations for training.

---

## 3. Model Architecture
### 3.1 Encoder (Backbone)
- Use Segformer's Mix Transformer (e.g., MiT-B1 or other variants).
- Initialize with ImageNet pre-trained weights.
- Extract features from multiple levels (likely four levels corresponding to the transformer layers).

### 3.2 Decoder
- A multi-level, transformer-based decoder:
  - Input features from encoder's different levels.
  - Upsample progressively to match input image resolution.
  - Final output: segmentation prediction map (size 512×512, number of classes=21).

### 3.3 Losses
- Partial Cross Entropy (L_pce): only computed over labeled pixels.
- Consistency loss (L_con): MSE between initial prediction and prototype-augmented predictions.
- Additional auxiliary losses (if implemented, e.g., regularization, edge/contour supervision).

---

## 4. Prototype Extraction & Update Strategy
### 4.1 Definitions
- **Local prototypes**: extracted from the current batch's features and prediction maps.
- **Global prototypes**: maintained in a memory bank, updated during training.
- **Prototype sets**: For each category, a set of K prototypes (e.g., K=5 in experiments).

### 4.2 Extraction Procedure
- From the current batch:
  - For each category c, gather feature vectors from features corresponding to labeled pixels (or high-confidence pixels).
  - Compute the weighted mean (via softmax or top-k selection) as the local prototype for that category.
- For the initial phase (warm-up):
  - Extract prototypes only from high-confidence, well-recognized regions.
- For the global prototypes:
  - Maintain a memory bank for each class.
  - Update using the described cosine similarity-based replacement:
    - If not full, directly add the new prototype.
    - If full, replace the most similar prototype (via cosine similarity) to the current local prototype with an interpolation (using α).

### 4.3 Prototype Management
- Use a memory bank data structure (e.g., PyTorch tensor buffers or NumPy arrays per class).
- Maintain an indicator for whether the global set is full.
- Hyperparameters:
  - Number of prototypes per class: K=5.
  - Momentum α (0.99).

---

## 5. Prototype-based Feature Augmentation
### 5.1 Augmentation based on Local Prototypes
- Input:
  - Feature map f (from encoder).
  - Extracted local prototype f_{p,local}.
- Process:
  - Compute attention weights by measuring similarity between each feature vector in f and the prototype (e.g., dot product, softmax).
  - Use weighted prototypes (attention-weighted features).
  - Concatenate or combine with original features.
  - Transform via a linear layer with ReLU.
  - Add residual connection: augmented_feature = ReL u(f + transformed features).
- Output:
  - Augmented feature map for subsequent decoding.

### 5.2 Augmentation based on Global Prototypes
- Similar, but prototypes are from the global prototype set:
  - Merge prototypes across classes as needed.
  - Use for augmentation only after the global prototype set is fully filled.
  - Might involve merging prototypes for the same class via clustering or mean pooling.

### 5.3 Combined Augmentation
- Mix both local and global prototypes:
  - When both are active, perform augmentation with each separately and combine predictions via ensemble/voting.
  - Use the combined augmented features for the final prediction.

---

## 6. Training Procedure & Loss Scheduling
- **Warm-up phase**: no prototype augmentation (use only partial cross entropy); global prototypes are empty.
- **Prototype extraction start**: after warm-up (e.g., first few epochs).
- **Prototype update**: during each batch, extract local prototypes; update global prototypes accordingly.
- **Use of prototypes**:
  - Phase 1: only local prototypes + partial CE.
  - Phase 2: local prototypes + global prototypes + partial CE + consistency loss.
  - Phase 3: both prototype types + full loss (partial CE + λ_l * L_con-l + λ_g * L_con-g).
- **Loss weights** (hyperparameters):
  - λ_l: e.g., 0.02.
  - λ_g: e.g., 0.05.
- **Training epochs**:
  - Use the multi-step scheduler with an initial lr=3e-5, decay by a factor of 0.01 at specified epochs (e.g., after 80% of total epochs).

---

## 7. Inference & Evaluation
### 7.1 Without Prototype Augmentation
- Use the trained decoder directly.
- Optionally, apply post-processing like CRF/Dilation if desired.

### 7.2 With Prototype Guidance
- During inference, in addition to the regular prediction:
  - Use retrieved prototypes to augment features further.
  - Use class label guidance if available (based on the setting).

### 7.3 Evaluation Metrics
- Mean IoU over 21 categories.
- Per-class IoU for detailed analysis.
- Use the original val set of VOC2012 for benchmark comparison.

---

## 8. Implementation Details & Hyperparameter Settings
- **Backbone**: Segformer MiT-B1, pretrained on ImageNet.
- **Optimizer**: AdamW, initial lr=3e-5, weight decay as in transformer settings.
- **Batch size**: 16.
- **Prototype set size**: 5 prototypes per category.
- **Momentum for prototypes**: 0.99.
- **Loss weights**: λ_l=0.02, λ_g=0.05 (tune if needed).
- **Prototypes extraction**:
  - Top-k selection, e.g., top 50% (k=0.5 in experiments).
- **Drop and shrink ratios** for robustness testing.
- **Training duration**: 100 epochs or as in the original paper, with learning rate decay schedule.

---

## 9. Additional Considerations
- **Code Structure**:
  - Modular design: dataset loader, model builder, prototype bank manager, training loop, validation.
- **Debugging & Validation**:
  - Validate prototype extraction separately.
  - Visualize prototype influence and augmented features.
  - Check the convergence of global prototype set.
- **Reproducibility**:
  - Fix random seeds.
  - Log training metrics (loss terms, IoU).

---

This detailed roadmap should serve as a solid foundation for the subsequent implementation phase. Once ready, we can translate these strategies into specific PyTorch (or other framework) code, respecting the architecture, data pipeline, prototype management, augmentation procedures, and training schedule described here.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the system using PyTorch and PyTorch Lightning for modularity and easy training management. The architecture will include a backbone encoder based on Segformer (implemented with timm or a custom implementation), a transformer decoder, prototype extraction modules, and feature augmentation modules. We will maintain prototype memory banks as PyTorch tensors, updating them via cosine similarity. Loss functions such as partial cross entropy and consistency loss will be implemented with torch.nn modules. The training loop will handle phases: warm-up, local prototype only, and full prototype utilization, scheduling losses accordingly. Data augmentation and dataset loading will be handled with albumentations and PyTorch datasets. Visualization and evaluation metrics (mIoU, per-class IoU) will be provided using standard tools. The system will be organized into main.py, dataset.py, model.py, trainer.py, and evaluation.py, with a configuration system for hyperparameters.",
    "File list": [
        "main.py",
        "dataset.py",
        "model.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class Dataset {\n        +__init__(data_dir: str, split: str, transform: Callable)\n        +__len__() -> int\n        +__getitem__(index: int) -> Tuple[Tensor, Tensor]\n    }\n    class Model {\n        +__init__(backbone_name: str, num_classes: int)\n        +forward(x: Tensor) -> Tensor\n        +extract_features(x: Tensor) -> list of Tensors\n        +augment_with_prototypes(features: Tensor, prototypes: Dict[str, Tensor]) -> Tensor\n    }\n    class PrototypeBank {\n        +__init__(num_classes: int, prototypes_per_class: int)\n        +update_prototypes(new_prototypes: Dict[str, Tensor]) -> None\n        +get_prototypes() -> Dict[str, Tensor]\n    }\n    class Trainer {\n        +__init__(model: Model, dataset: Dataset, config: dict)\n        +train() -> None\n        +schedule_losses(epoch: int) -> None\n        +extract_local_prototypes(features: list of Tensors, labels: Tensor) -> Dict[str, Tensor]\n        +update_global_prototypes(local_prototypes: Dict[str, Tensor]) -> None\n        +compute_loss(preds: Tensor, labels: Tensor, prototypes: Dict[str, Tensor], phase: str) -> Tensor\n    }\n    class Evaluation {\n        +__init__(model: Model, dataset: Dataset)\n        +evaluate() -> dict\n        +calculate_mIoU(preds: Tensor, labels: Tensor) -> float\n    }\n\nMain --> Dataset\nMain --> Trainer\nTrainer --> Model\nTrainer --> PrototypeBank\nTrainer --> Evaluation\nModel --> PrototypeBank\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DS as Dataset\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DS: __init__(data_dir, split, transform)\n    DS-->>M: dataset object\n    M->>MD: __init__(backbone, num_classes)\n    M->>TR: new(model, dataset, config)\n    TR->>TR: Initialize prototype bank\n    TR->>TR: For epoch in epochs:\n    loop\n        TR->>DS: get batch\n        DS-->>TR: batch images and labels\n        TR->>MD: extract features (from images)\n        TR->>TR: compute predictions\n        TR->>TR: schedule loss (phase based on epoch)\n        TR->>TR: extract local prototypes\n        TR->>TR: update global prototypes\n        TR->>TR: compute total loss\n        TR->>MD: backward() & step()\n    end\n    M->>EV: evaluate(model, val_dataset)\n    EV->>MD: forward(val.images)\n    MD-->>EV: predictions\n    EV-->>M: metrics (mIoU, per-class IoU)\n",
    "Anything UNCLEAR": "Clarification needed on the exact format of scribble annotations and whether additional post-processing (e.g., CRF) is included during inference."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.0",
        "torchvision==0.14.0",
        "timm==0.6.7",
        "albumentations==1.0.3",
        "scikit-learn==0.24.2",
        "numpy==1.21.0",
        "matplotlib==3.5.1",
        "pycocotools==2.0.2",
        "pyaml==21.12.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset.py",
            "Implement Dataset class: handles loading Pascal VOC, parsing scribble annotations, applying augmentations with albumentations, and returning image-label pairs compatible with torch Dataset standards. Depends on annotations in VOC format and scribble masks. Relies on torchvision.transforms for data augmentation, and a custom collate function if necessary."
        ],
        [
            "model.py",
            "Implement backbone encoder based on Segformer (timms or custom). Define the Mix Transformer encoder, which extracts multi-level features. Build the transformer decoder that takes features and prototypes, performs feature enhancement through prototype-based augmentation, and outputs segmentation logits. Includes the prototype extraction modules and feature augmentation layers with flexible interface for local/global prototypes."
        ],
        [
            "trainer.py",
            "Implement Trainer class: manages the training loop, including initial prototype extraction, updating prototype memory banks, scheduling the training phases (warm-up, partial, full), computing losses (partial cross entropy and consistency loss), and integrating prototype-based augmentation into forward passes. Depends on model.py, dataset.py, and prototype management code."
        ],
        [
            "evaluation.py",
            "Implement evaluate() function: loads the trained model, performs inference on validation set, optionally with prototype guidance, computes per-class IoU and mean IoU, and records the results. Utilizes utils from torchvision or scikit-learn for metric calculation."
        ],
        [
            "main.py",
            "Main entry point: initializes dataset loaders (train/val), model, trainer, and evaluation objects, runs training with proper schedule, manages checkpoints, and runs final evaluation, saving metrics and visualizations. Coordinates dependencies among data, model, training, and evaluation modules."
        ]
    ],
    "Task list": [
        "dataset.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "None (internal API for training and evaluation functions).",
    "Shared Knowledge": "Shared parameters like number of prototypes per class, prototype update momentum, loss weights (λ_l, λ_g), and data augmentation configs are set in a shared configuration or constants module, imported across all scripts.",
    "Anything UNCLEAR": "Clarification needed on the exact format of scribble annotations and whether the model should support inference with/without prototypes during testing, as well as the specific evaluation protocol (e.g., whether test time augmentation is expected)."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 3e-5  # Initial learning rate as used in experiments
  batch_size: 16     # Number of images per batch
  epochs: 100        # Total number of training epochs
  lr_decay_epoch: 80 # Epoch at which to decay the learning rate
  lr_decay_factor: 0.01 # Factor by which to decay learning rate
model:
  backbone: "mit-b1"  # Segformer MiT-B1 backbone
  num_classes: 21   # 20 object categories + 1 background
  proto_num_per_class: 5 # Number of prototypes per class (global prototypes)
  proto_momentum: 0.99 # Momentum for prototype updates
  prototype_extraction_topk: 0.5 # Top-k percentage during prototype extraction
loss:
  partial_ce_scale: 1.0  # Weight for partial cross entropy loss
  lambda_local: 0.02   # Loss weight for local prototype consistency
  lambda_global: 0.05  # Loss weight for global prototype consistency
data:
  image_size: 512     # Cropped size for training images
  augmentation:
    random_scale: [0.5, 2.0]
    rotation_degrees: [-10, 10]
    horizontal_flip: True
    gaussian_blur: True
  dataset_path: "path/to/VOC2012" # Path to Pascal VOC 2012 dataset
  scribble_annotations_path: "path/to/ScribbleAnnotations" # Path to scribble masks
prototypes:
  interaction_topk: 0.5 # Top-k percentage from feature vectors
  global_proto_count: 5 # Number of prototypes per class in memory bank
  update_momentum: 0.99 # Momentum for updating global prototypes
  warmup_epochs: 10 # Epochs before prototypes are used
inference:
  use_prototypes: true # Whether to use prototypes during inference
  class_guidance: false # Whether to use class label guidance during inference
```

---

**Note:** Full configuration is available in `planning_config.yaml`
