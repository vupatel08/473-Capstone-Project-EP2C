# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. To create a comprehensive reproduction plan for the proposed Sample-specific Multi-channel Mask (SMM) framework as described, I will structure the outline into several clear sections, emphasizing the key details from the methodology, experimental setup, hyperparameters, datasets, and evaluation metrics. This will serve as an actionable blueprint for later implementation. 

---

## 1. **Understanding the Core Methodology**

### 1.1. Architecture Overview
- **Objective:** Re-purpose a pre-trained classifier (ResNet, ViT) for target datasets without full finetuning.
- **Main Innovation:** Generate *sample-specific* three-channel masks ($f_{\text{mask}}$) via a lightweight CNN, interpolated with patch-wise upsampling, to adapt the prompt patterns at a sample level.
- **Pipeline:**
  - Input image undergoes resizing (bilinear interpolation) to target size.
  - A CNN-based mask generator produces a mask of reduced resolution.
  - Patch-wise upsampling (interpolation) enlarges this mask to match the image size.
  - The learnable pattern $\delta$ (small parameter set) is pixel-wise multiplied with this mask.
  - This masked pattern is added onto the resized image.
  - The resulting image is fed into a *fixed*, pre-trained classifier.
  - Only the pattern $\delta$ and mask generator parameters $\phi$ are learned.

### 1.2. Objective Functions & Optimization
- Minimize empirical loss (e.g., cross-entropy) between model output on reprogrammed images and target labels.
- Alternately update:
  - Pattern $\delta$ (shared across all samples)
  - Mask generator parameters $\phi$ (sample-specific mask generation)
- Use an Adam or SGD optimizer with carefully tuned hyperparameters for these.

### 1.3. Sample-specific Mask Generation ($f_{\text{mask}}$)
- **Input:** Resized input image.
- **Architecture:** Lightweight CNN with 5–6 layers, including:
  - Convolutional layers with small kernel sizes (3×3)
  - Batch normalization + ReLU
  - Pooling layers (max pooling, 2×2) for resolution reduction
  - Final convolutional layer with 3 channels
- **Output size:** Reduced resolution (e.g., 1/4 or 1/8 of input size), then patch-wise upsample via repeating pixels (no learned interpolation).
- **Parameters:** Very few (~10^4), e.g., as in Table 4.

### 1.4. Reprogramming Pattern ($\delta$)
- Initialize as zero, shared for all images.
- Update via gradient descent.
- Adds an adaptive pattern into the image, masked by $f_{\text{mask}}$.

---

## 2. **Implementation Details & Key Aspects**

### 2.1. Data Preprocessing
- **Resize images** from dataset to the target size (e.g., 224×224 or 384×384).
- Generate target train/test splits matching original datasets:
  - CIFAR-10/100
  - SVHN
  - GTSRB
  - Flowers102
  - DTD
  - UCF101
  - Food101
  - EuroSAT
  - OxfordPets
  - SUN397
- Normalize images as per the pre-trained model's requirements (ImageNet mean/std for ViT or ResNet).

### 2.2. Mask Generator & Pattern Initialization
- **Mask generator ($f_{\text{mask}}$):**
  - Use a small CNN with 5–6 layers, with configurable depth based on dataset complexity.
  - Input: Resized image (H×W×3).
  - Output: Mask in reduced resolution (e.g., H/8×W/8×3).
  - Initialize weights randomly.
- **Pattern $\delta$**:
  - Initialize as zeros of shape [H×W×3].
  - Treat as a learnable parameter.

### 2.3. Patch-wise Interpolation
- After CNN produces the low-res mask:
  - Upsample by repeating each pixel in a grid-like manner (patch-wise).
  - No bilinear or bicubic interpolation; simply tile each pixel within patches.
- The resolution of the final mask should match the input image size.

### 2.4. Feature Masking & Pattern Application
- Pixel-wise multiply $\delta$ with the padded/upscaled mask.
- Add the result to the resized image, forming the prompt-enhanced input.

### 1.5. Loss & Optimization
- Use cross-entropy loss between classifier's output on reprogrammed images and ground truth labels.
- Only $\phi$ and $\delta$ are optimized; pre-trained model is frozen.
- Learning rates (e.g., 0.01 for $\phi$, 0.001 for $\delta$) are based on hyperparameter tuning.
- Use epoch-based training (e.g., 200 epochs).

---

## 3. **Experimental Setup & Hyperparameters**

### 3.1. Datasets & Input Sizes
- CIFAR-10/100, SVHN, GTSRB, Flowers102, DTD, UCF101, Food101, EuroSAT, OxfordPets, SUN397.
- Input sizes: 128×128 for most, 384×384 for ViT-B32.
- For small datasets, resize images appropriately.

### 3.2. Hyperparameters
- **Optimizer:** Adam or SGD with momentum.
- **Learning Rates/Schedules:**
  - For $\phi$: starting at 0.01 with step decay (e.g., decay by 0.1 at 100 epochs).
  - For $\delta$: smaller learning rate (e.g., 0.001).
- **Batch size:** 64 or 128 depending on dataset size and GPU memory.
- **Epochs:** 200 with step decay or cosine schedule.
- **Regularizations:** Optional weight decay or pattern norm constraints to prevent overfitting.

### 3.3. Mask Generator Hyperparameters
- Number of max pooling layers: 2–3.
- Kernel sizes: 3×3.
- Number of filters: small, e.g., 64 or 128.
- Final output channels: 3.
- Size of masked area (mask resolution): e.g., 1/8 or 1/4 of input.

---

## 4. **Training Procedure**
1. Initialize $\phi$ and $\delta$.
2. For each epoch:
   - For all samples:
     - Resize images.
     - Generate mask via CNN.
     - Perform patch-wise upsampling.
     - Pixel-wise multiply $\delta$.
     - Add to resized images.
   - Feed into pre-trained fixed classifier (ResNet, ViT).
   - Compute loss.
   - Backpropagate to updates:
     - $\phi$ (mask generator params).
     - $\delta$ (prompt pattern).
3. Optionally update label mappings (e.g., Flm, Ilm) if targeting label mismatches.
4. Track training and validation accuracy.

---

## 5. **Evaluation Metrics and Reporting**
- Classification accuracy on target test sets.
- Comparison with baseline VR (shared masks, watermarking).
- Effect of mask resolution (patch size), pattern size.
- Visualized results: reprogrammed images (see Figures 13–23) with and without noise patterns.
- Ablation of:
  - Mask generator depths.
  - Patch sizes.
  - Pattern initializations.
  - Hyperparameters.

---

## 6. **Additional Considerations**
- For large models or high resolutions, you may utilize mixed-precision training.
- Validate the sample-specific masks visually.
- Repeat experiments over multiple seeds to assess stability.
- Use GPU acceleration, especially for CNN mask generator, pattern updates, and large datasets.

---

## 7. **Summary Checklist for Implementation**
- Prepare datasets with correct resizing.
- Implement CNN-based mask generator with configurable depth.
- Implement upsampling via patch repetition.
- Initialize pattern $\delta$; treat as a gradient variable.
- Freeze backbone classifier, optimize $\phi$ and $\delta$.
- Use a standard optimizer and schedule.
- Log training curves and report final accuracy.
- Visualize patterns and reprogrammed images for inspection.

---

This roadmap should enable you to implement the SMM framework systematically, ensuring all critical methodological nuances, experimental configurations, and hyperparameters are addressed. Once the plan is validated, you can proceed to coding each component in Python (PyTorch/TensorFlow).

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular system using PyTorch. The core components include a dataset loader, a lightweight CNN mask generator, the main training loop, and evaluation routines. We will design a MaskGenerator class to produce sample-specific masks, a Pattern parameter tensor to optimize across all samples, and a reprogramming function to add masked patterns to resized images. The classifier backbone (ResNet or ViT) will be loaded pre-trained and frozen. During training, only the mask generator parameters and pattern will be updated. Patch-wise upsampling will be achieved by repeating pixel values. The system will support multiple datasets and hyperparameter configurations for optimizer, learning rate, and schedule. Visualization functions will display reprogrammed images and masks. We will maintain simplicity by having a main.py script orchestrate dataset loading, model initialization, training, and evaluation, with clear configuration management.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "visualization.py",
        "config.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: Config)
        +run_experiment() -> None
    }
    class DatasetLoader {
        +__init__(dataset_name: str, split: str, resize_size: Tuple[int, int])
        +load_data() -> Tuple[DataLoader, DataLoader]
    }
    class Model {
        +__init__(model_name: str, architecture_params: dict)
        +load_pretrained() -> None
        +freeze() -> None
        +forward(x: Tensor) -> Tensor
    }
    class MaskGenerator {
        +__init__(config: dict)
        +generate_mask(image: Tensor) -> Tensor
        +parameters: torch.nn.Parameter
        +forward(image: Tensor) -> Tensor
    }
    class Pattern {
        +__init__(shape: Tuple[int, int, int])
        +pattern: torch.nn.Parameter
    }
    class Trainer {
        +__init__(model: Model, mask_generator: MaskGenerator, pattern: Pattern, optimizer: torch.optim.Optimizer, loss_fn: callable)
        +train(train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None
        +update_parameters() -> None
    }
    class Evaluation {
        +__init__(model: Model)
        +evaluate(test_loader: DataLoader) -> dict
    }
    class Visualizer {
        +show_reprogrammed_images(images: List[Tensor], masks: List[Tensor], patterns: Tensor) -> None
        +display_mask(mask: Tensor) -> None
    }
    Main -- DatasetLoader
    Main -- Model
    Main -- MaskGenerator
    Main -- Pattern
    Main -- Trainer
    Main -- Evaluation
    Main -- Visualization
    Trainer --> Model
    Trainer --> MaskGenerator
    Trainer --> Pattern
    Pattern --> torch.nn.Parameter
    MaskGenerator --> torch.nn.Module
    Pattern --> torch.nn.Parameter
    Evaluation --> Model
    Visualization --> Pattern
    Visualization --> MaskGenerator
",
    "Program call flow": "
sequenceDiagram
    participant MC as Main
    participant DL as DatasetLoader
    participant Mdl as Model
    participant MG as MaskGenerator
    participant Pt as Pattern
    participant Tr as Trainer
    participant Ev as Evaluation
    participant Vis as Visualization
    MC->>DL: load_data()
    DL-->>MC: dataset loaders
    MC->>Mdl: load_pretrained()
    Mdl-->>MC: model loaded, frozen
    MC->>MG: initialize with config
    MC->>Pt: initialize pattern
    MC->>Tr: create trainer with model, mask generator, pattern, optimizer, loss
    loop for epochs
        Tr->>DL: get train_loader, val_loader
        Tr->>Tr: train one epoch
        Tr-->>MC: epoch complete
    end
    MC->>Ev: evaluate(test_loader)
    Ev->>Model: forward()
    Model-->>Ev: predictions
    Ev-->>MC: metrics
    MC->>Vis: visualize reprogrammed images and masks
    Vis-->>MC: display results
",
    "Anything UNCLEAR": "Clarify dataset formats and expected image resolutions. Confirm if the mask generator should support varying architecture depths and patch sizes as described, and clarify the hyperparameter settings for optimizer schedules."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.10.0",
        "torchvision==0.11.1",
        "numpy==1.21.0",
        "matplotlib==3.4.3",
        "Pillow==8.4.0",
        "scipy==1.7.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: handles loading datasets, applies dataset splits, resizing images to target input size, and returns DataLoader objects. It must be modular to support multiple datasets with their specific configurations."
        ],
        [
            "model.py",
            "Defines Model class: loads pre-trained models such as ResNet or ViT, initializes the model, loads pre-trained weights, and sets model to frozen mode; exposes a forward() method for inference."
        ],
        [
            "mask_generator.py",
            "Defines MaskGenerator class: implements a lightweight CNN with configurable depth, number of filters, pooling layers, and output size. Generates mask tensors from input images, supports different patch sizes for upsampling via pixel repetition."
        ],
        [
            "pattern.py",
            "Defines Pattern class: manages a torch.nn.Parameter pattern tensor, initialized as zeros, shared across all images, adjustable during training."
        ],
        [
            "trainer.py",
            "Defines Trainer class: manages the training loop, updates mask generator parameters ($\phi$) and pattern ($\delta$), computes loss (cross-entropy), applies mask to pattern, adds to resized image, and optimizes only sample-specific mask generator and pattern parameters. Supports LR schedules and multiple epochs."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class: runs the fixed (pre-trained + frozen) classifier on reprogrammed images, computes accuracy and metrics, supports visualization."
        ],
        [
            "visualization.py",
            "Provides functions to display original images, reprogrammed images, generated masks, and overlay visualizations of the mask application, using matplotlib."
        ],
        [
            "main.py",
            "Main script: configures hyperparameters, sets up datasets via DatasetLoader, initializes model, mask generator, pattern, trainer, and evaluation modules. Orchestrates training epochs, periodic validation, and final evaluation. Handles command-line arguments for flexible configuration."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "config.py",
        "dataset_loader.py",
        "model.py",
        "mask_generator.py",
        "pattern.py",
        "trainer.py",
        "evaluation.py",
        "visualization.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n\n# No front-end API needed; core modules interact internally. If future extension needed, define endpoints for dataset configuration, training status, and visualization, e.g.,\n# POST /train, GET /status, GET /results.\n",
    "Shared Knowledge": "Configuration variables such as dataset paths, input sizes, optimizer settings, hyperparameters, and model types are shared across modules. Utility functions for image resizing, tensor normalization, and tensor tiling are placed in helpers or utility functions, imported where needed.",
    "Anything UNCLEAR": "Clarify specific dataset formats and normalization schemes, hyperparameter tuning ranges for learning rates and patch sizes, and whether to support multiple dataset configurations simultaneously or sequentially."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: CIFAR10
  input_size: [32, 32]
  train_split: 50000
  test_split: 10000
  batch_size: 128

model:
  name: ResNet50
  pretrained: true

mask_generator:
  architecture_depth: 5
  kernel_size: 3
  filters: 64
  pooling_layers: 2
  output_ratio: 1/8

training:
  optimizer: Adam
  learning_rate: 0.01
  lr_decay_epochs: [100]
  lr_decay_factor: 0.1
  epochs: 200
  pattern_lr: 0.001
  pattern_lr_decay_epochs: [100]
  pattern_lr_decay_factor: 0.1
  weight_decay: 1e-4
  pattern_init: zeros

sampling:
  image_resize: [32, 32]
  patch_size: 8

evaluation:
  metrics: accuracy
  visualize: true

reproducibility:
  seed: 42
```

---

**Note:** Full configuration is available in `planning_config.yaml`
