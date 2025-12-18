# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a detailed, structured plan to reproduce the methodology and experiments described in the paper "You Only Look Around: Learning Illumination Invariant Feature for Low-light Object Detection." This plan covers crucial design choices, dataset requirements, training procedures, hyperparameters, and evaluation protocols, based on information directly from the paper. It is intended to serve as a comprehensive guide for subsequent implementation.

---

# 1. Understanding the Core Methodology

### **Main Contribution**
- The paper proposes **YOLA**, a framework for low-light object detection promising to learn illumination-invariant features via a novel module called **Illumination Invariant Module (IIM)**.
- The key idea is to approximate illumination-invariant features based on the Lambertian image formation model, exploiting relationships between neighboring color channels and adjacent pixels.
- The module learns convolutional kernels with a zero-mean constraint and a task-driven loss (II Loss) encouraging the kernels to generate features invariant to illumination variations.

### **Architecture Overview**:
- Input image (standard RGB) → IIM (extracts illumination-invariant features) → fusion with original images → object detector (YOLOv3 or TOOD).
- The IIM can be integrated into existing detection frameworks.

### **Core components to implement**:
- IIM: with learnable kernels satisfying symmetry constraints (zero-mean) and trained with II Loss.
- Fusion block: combines IIM features with original image features.
- Detection framework: YOLOv3 or TOOD, fine-tuned with the modified features.
- Optional: progressive kernel size exploration, with kernel sizes in {3, 5} as per experiments.

---

# 2. Dataset and Data Preparation

### **Datasets used in the study:**
- **ExDark**: Low-light detection dataset with 12 object categories, in both training and testing splits.
- **UG^2 + DARK FACE**: Combination of UG^2 (with night images) and DarkFace datasets. 
- **LIS**: Low-light Instance Segmentation dataset for segmentation tasks.
- **Coco, VOC, or other general detection/segmentation datasets** for additional experiments outside low-light.

### **Data requirements/questions:**
- Obtain the *ExDark* dataset from its official source.
- For UG^2+Dark Face, compile images from UG^2 and DarkFace datasets, ensuring labels are aligned for detection.
- For segmentation experiments, compile LIS dataset.
- All datasets should be resized to **608×608** resolution for consistency with the paper (as per their experimental setup). 
- Annotate with bounding boxes and masks as needed.
- Create train/validation/test splits consistent with the paper for reproducibility.

### **Data augmentation:**
- For low-light scenarios, augment images with synthetic lighting changes when needed.
- Basic augmentations: flip, scale, crop, possibly additional low-light simulation (gamma correction, UV smoothing) if helpful.

---

# 3. Implementation Details — Methodology

### **3.1 Illumination Invariant Module (IIM)**
- **Design:**
  - Input: RGB image.
  - The module applies **learnable convolutional kernels** (size k×k, with k ∈ {3, 5}) to approximate the fixed physics-inspired equations.
  - Constraints:
    - Enforce **zero-mean** on kernel weights via channel-wise normalization.
    - Kernel weights are learned during training.
  - **Output:** features that approximate the illumination-invariant features.
  
- **Kernel Parameterization:**
  - Initialize kernels according to the fixed equation (related to the Lambertian model's cross-color ratios).
  - During training, optimize kernels with a custom **II Loss** to keep features illumination-invariant.
  - Both **fixed weight kernels** (edge-based as IIM-Edge) and **learnable kernels** (full IIM) are explored.

- **Training constraints:**
  - Zero-mean constraint: after each gradient update, project kernels to zero mean: `W = W - mean(W)`.
  - Use of **weight normalization** for stability.

### **3.2 II Loss (Illumination Invariant Loss)**
- **Purpose:** Enforce consistency of features extracted across images with different illumination conditions.
- **Implementation:**
  - Generate pairs of images: original light and artificially-illuminated (gamma correction, or other brightness adjustments).
  - For each pair:
    - Extract features via IIM.
    - Compute the **feature difference**.
    - Compute the **L2 loss** of the differences, scaled by a factor β=1.
  - **Loss term:** 
    \[
    \text{II Loss} = \sum_{i} \left\| f_{\mathcal{W}_i}(I) - f_{\mathcal{W}_i}(\sigma(I)) \right\|^2
    \]
  - Regularly apply **kernel smoothing** or **local mean constraints** to ensure kernels don't trivially satisfy zero-mean due to degeneracy (see Fig. 5 and discussion).
  
- **Hyperparameters:**
  - Balance II Loss with detection/classification loss, scaled by 0.01.

### **3.3 Model Integration**
- Insert IIM after the backbone (e.g., Darknet for YOLOv3 or backbone for TOOD).
- Merge features from IIM with original features via concatenation or addition (fuse convolution block).
- Fine-tune the entire detector with combined supervision.

### **3.4 Detection Frameworks**
- **YOLOv3**:
  - Use official or widely available PyTorch/TensorFlow implementations.
  - Fine-tune on the prepared datasets, with full detection and classification supervised.
- **TOOD**:
  - Use the open-source HH Mask RCNN or its detection modules, modified with IIM as above.
  - Fine-tune for 12-24 epochs, learning rate starting at ~1e-3, with decay.

---

# 4. Hyperparameters & Training Details

| Parameter | Default / Recommended | Notes |
|-------------|------------------------|--------|
| Input image size | 608×608 | consistent with experimental results |
| Kernel size | {3, 5} | investigate both; choose based on ablation results |
| Number of kernels | 4–8 (experimentally optimized) | start small, tune further |
| Batch size | 16–32 | depends on GPU memory |
| Learning rate | 1e-3 → 1e-4 | with decay after 10 epochs |
| Optimizer | Adam or SGD + momentum | Adam recommended for stability |
| II Loss scale | 0.01 | relative to detection loss |
| Zero-mean constraint | Projection after each update | implement as a kernel projection step |

### **Training protocol:**
- Pre-train detection backbone (e.g., Darknet-53 for YOLOv3) on COCO or VOC.
- Insert IIM; jointly train detection + IIM with II Loss.
- For low-light datasets:
  - Begin with detection pretraining (if possible).
  - Fine-tune with low-light images, synthetic illumination variations, and II Loss.
- Use validation set to monitor detection mAP (on small and large objects) at IoU thresholds (e.g., 0.5, 0.75).

### **Evaluation metrics:**
- Detection: mAP@0.5 and 0.75.
- Segmentation (if applicable): APseg.
- Additional: qualitative visualization of feature maps and detection boxes.

---

# 5. Implementation Constraints & Considerations

- **Physics-based assumptions:** The Lambertian model is an approximation, so the learnable kernels are critical for robustness.
- **Kernel size and number:** Use results from ablation in Table 6 to select kernel size (preferably size 5 for best performance).
- **Kernel initialization:** Start with physics-inspired kernel weights, but allow learnability.
- **Training pairs for II Loss:** Generate image pairs with controlled illumination changes during training.
- **Handling of biases:** Avoid trivial solutions by applying the II Loss and local kernel regularization.

---

# 6. Additional Experiments & Ablation Studies

- Explore kernel sizes {3, 5}.
- Vary number of kernels.
- Evaluate effect of II Loss with different weights.
- Test integration with YOLOv3 and TOOD.
- Evaluate on multiple datasets (ExDark, DarkFace, LIS).

---

# 7. Summary Checklist (for implementation):
- [ ] Prepare datasets at 608×608 resolution.
- [ ] Implement IIM with learnable kernels, zero-mean projection.
- [ ] Set up detection backbone + fusion + IIM + detection head.
- [ ] Implement II Loss with paired images under different illumination.
- [ ] Choose optimizer, learning rate schedule, and training epochs.
- [ ] Run ablation for kernel size and number.
- [ ] Evaluate detection/masks, visualize features, and compare against baselines.

---

This roadmap incorporates the key technical details and experimental design elements described in the paper, embedding their physics-based derivation, constraints, and optimization strategies. It respects the authors’ parameter choices while leaving room for hyperparameter tuning based on ablation results. Once this plan is executed, detailed coding and experimental validation will be streamlined and well-informed.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular detection framework using PyTorch, integrating an Illumination Invariant Module (IIM) with learnable kernels constrained via zero-mean projection and trained with an illumination-invariance loss. The detection backbone will be YOLOv3 or TOOD, with the IIM inserted after the backbone features. Synthetic illumination variations will be generated on-the-fly during training for II Loss. The system will load datasets (ExDark, DarkFace, LIS) resized to 608x608, with dual branches for detection and II Loss. We will optimize all components jointly, employing Adam optimizer with learning rate decay, and evaluate with mAP metrics. The code will be structured into main.py, dataset_loader.py, model.py, trainer.py, and evaluation.py, with clear class interfaces and training loops.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> Dataset\n    }\n    class Dataset {\n        +images: List[Tensor]\n        +annotations: List[dict]\n        +__init__(image_paths: List[str], label_paths: List[str], size: Tuple[int, int])\n    }\n    class Model {\n        +__init__(params: dict)\n        +features: torch.nn.Module  # backbone + IIM + fusion\n        +detect(x: Tensor) -> List[Detection]\n        +extract_features(x: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, dataset: Dataset, optimizer: torch.optim.Optimizer, loss_params: dict)\n        +train(num_epochs: int) -> None\n        +calculate_loss(batch: Batch) -> torch.Tensor\n    }\n    class Evaluation {\n        +__init__(model: Model, dataset: Dataset)\n        +evaluate() -> dict  # returns detection metrics like mAP\n    }\n    class Detection {\n        +boxes: Tensor\n        +scores: Tensor\n        +labels: Tensor\n        +masks: Optional[Tensor]\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> Model\n    Model --> torch.nn.Module\n    Dataset --> List[Tensor]\n    Detection --> Tensor\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: load_data()\n    DL-->>M: dataset\n    M->>MD: initialize() with params\n    M->>TR: train(model, dataset, optimizer, epochs)\n    TR->>TR: for each epoch: batch -> compute loss, backpropagate\n    TR-->>M: training complete\n    M->>EV: evaluate(model, dataset)\n    EV->>EV: run detection on dataset, compute mAP\n    EV-->>M: metrics\n    Note over M: System runs training then evaluation, iterating with hyperparameter tuning as needed.\n",
    "Anything UNCLEAR": "Clarify the exact dataset input formats and any preferred hyperparameters or augmentation specifics. Confirm whether to use YOLOv3 or TOOD, and detail any custom evaluation scripts needed."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.9.0",
        "torchvision==0.10.0",
        "numpy==1.21.0",
        "matplotlib==3.4.3",
        "opencv-python==4.5.3",
        "scipy==1.7.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: loads images and annotations, applies resizing to 608x608, implements train/test splits. Loads pairs for II Loss and synthetic illumination augmentation. Handles dataset format compatibility and preprocessing pipeline."
        ],
        [
            "model.py",
            "Defines DetectionModel class: builds backbone + IIM (learnable kernels with zero-mean constraint), fusion module, and detection head (YOLOv3 or TOOD). Implements feature extraction, kernel initialization with physics-based priors, and the convolutional kernels with projection step for zero-mean constraint."
        ],
        [
            "trainer.py",
            "Defines Trainer class: initializes model, optimizer, and training loop. Implements joint training procedure, including detection loss and II Loss. Handles synthetic illumination variation generation on-the-fly, applies II loss, and projects kernels after each update."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class: performs inference on validation/test datasets, computes detection metrics (mAP), includes visualization routines for feature maps and detection boxes, and outputs evaluation reports."
        ],
        [
            "main.py",
            "Script entry point: creates dataset loader, initializes model, trainer, and evaluation objects, manages overall training + evaluation pipeline, plots performance metrics, and saves final models/visualizations."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "",
    "Shared Knowledge": "Common utility functions include image resizing, data augmentation, kernel zero-mean projection, and synthetic illumination variation generation. Shared data structures include Dataset objects, detection outputs, and configuration dicts for hyperparameters.",
    "Anything UNCLEAR": "Exact dataset format details (annotation structure, labeling conventions), preferred hyperparameters (learning rate schedule, kernel size options), and whether to use YOLOv3 or TOOD are not fully specified. Clarification needed for hardware constraints and training environment (GPU specs)."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001  # initial learning rate, recommended starting point
  batch_size: 16        # batch size, adjust according to hardware capacity
  epochs: 24            # total number of training epochs
  weight_decay: 0.0005  # weight decay for optimizer regularization
  lr_scheduler: step    # learning rate scheduler type
  step_size: 10        # number of epochs before learning rate decay
  gamma: 0.1           # decay factor for LR scheduler
  detection_input_size: 608  # input image resolution for detection network

model:
  backbone: darknet53     # or 'tood' if using TOOD framework
  backbone_pretrain: True # whether to pretrain backbone on COCO or ImageNet
  kernel_size_options: [3, 5]  # explore kernel sizes 3 and 5
  num_kernels: 4         # number of learnable kernels in IIM
  kernel_constraints:
    zero_mean_constraint: true  # enforce zero-mean kernels post-optimization
  fusion_method: concat   # how to fuse IIM features with backbone features

dataset:
  dataset_name: ExDark   # primary dataset for low-light detection
  input_size: 608        # resize images to this size
  train_split_ratio: 0.8 # 80% training, 20% validation
  val_split_ratio: 0.2
  synthetic_illumination: true  # generate image pairs with gamma transforms for II loss
  augmentation:
    flip: true
    scale: [0.8, 1.2]
    crop: true
    brightness_adjustment: [0.5, 1.5]

loss:
  detection_loss_weight: 1.0  # weight for detection loss
  ii_loss_weight: 0.01      # weight for illumination invariance (II Loss)
  ii_loss_scale: 1.0        # scale factor applied to II Loss during joint optimization

evaluation:
  metrics: [mAP@0.5, mAP@0.75]
  evaluation_epochs: 1  # evaluate after every epoch
  save_model_every: 5  # save model checkpoints every 5 epochs
  evaluate_on_test: true
  visualization:
    feature_maps: true
    detection_boxes: true

hardware:
  gpus: 1  # number of GPUs to use; adjust based on hardware
  mixed_precision: false  # use mixed precision training if supported
```

---

**Note:** Full configuration is available in `planning_config.yaml`
