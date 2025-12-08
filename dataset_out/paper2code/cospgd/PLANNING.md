# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, detailed plan to reproduce the CosPGD algorithm and the associated experiments as described in the paper. This plan is organized into clear sections covering core methodological components, dataset setup, hyperparameters, experimental procedures, and evaluation metrics, with explicit details extracted from the paper, including the supplementary material and figures.

---

## 1. Core Methodology of CosPGD

**Goal:** Implement the Cosine-AlignmentScaled PGD attack for pixel-wise prediction tasks (semantic segmentation, optical flow, image restoration) in a white-box setting. The attack encourages balanced errors across the entire pixel domain, leveraging a smooth, differentiable measure of pixel-wise prediction-target alignment.

**Key points:**

- The attack involves iteratively updating the input sample \(X^{adv}\) to increase loss scaled by a normalized cosine similarity between model predictions and targets.
- The scaling function: \(\sum_{i} \cos( \psi(f_\theta(X^{adv})_i), Y_i) \cdot \bar{L}(f_\theta(X^{adv})_i, Y_i)\).
- \(\psi\): a differentiable, monotonically increasing function, in practice the softmax function applied pixel-wise, normalized to produce a distribution over classes or outputs.
- Cosine similarity: \(\cos(P, Y) = \frac{P \cdot Y}{\|P\| \|Y\|}\), with \(\|Y\|=1\) in one-hot encoding.
- The gradient update: scaled by the sign of gradient of the scaled loss, with step size \(\alpha\).
- Loss \(\bar{L}\): pixel-wise loss (e.g., cross-entropy or regression loss).

### Implementation outline:
- Initialize: \(X^{adv}_0 = X^{clean} + \text{uniform noise in } [-\epsilon, +\epsilon]\).
- For each iteration \(t\):
  - Compute model prediction \(P = f_\theta(X^{adv}_t)\).
  - Compute \(\psi(P)\); typically the softmax normalization over model outputs.
  - Compute cosine similarity (alignment score): \(\cos(\psi(P)_i, Y_i)\) for each pixel \(i\).
  - Multiply the pixel-wise loss \(\bar{L}(P_i, Y_i)\) by the alignment score to get scaled pixel loss.
  - Backpropagate this scaled pixel loss to compute gradient w.r.t. \(X^{adv}_t\).
  - Update \(X^{adv}_{t+1} = X^{adv}_t + \alpha \cdot \text{sign}(\nabla_{X^{adv}_t} \text{scaled loss})\).
  - Clip \(X^{adv}_{t+1}\) to \(\left[\max(0, X^{clean} - \epsilon), \min(1, X^{clean} + \epsilon)\right]\) (or \(\ell_\infty\) ball).
- Return \(X^{adv}_T\) after \(T\) iterations.

**Implementation notes:**
- Use PyTorch or TensorFlow, with automatic differentiation.
- Pixel-wise classification: model outputs logits (before softmax). \(\psi(\cdot)\) applies softmax.
- For regression tasks, \(\psi\) might be the identity or other normalization as justified.
- The scaling emphasizes pixels with predictions close to target (high cosine similarity), promoting balanced error induction across all pixels.
  
---

## 2. Dataset Requirements & Preprocessing

- **Semantic Segmentation:**
  - Dataset: PASCAL VOC 2012 (standard splits, augment to ~10,582 images as per authors).
  - Input size: standard sizes (e.g., 224x224, 512x512), as used in experiments (also see sections for resolution).
  - Labels: one-hot or class indices; softmax used to compute \(\psi\).

- **Optical Flow:**
  - Dataset: KITTI 2015, Sintel (train/validation splits).
  - Input: image pairs, ground-truth optical flow (vector fields).
  - Preprocessing: normalize inputs; model predicted outputs are flow vectors, targets are the flow ground truth.
  - For attack input: no special preprocessing other than normalization, same as training.

- **Image De-Noise / Restoration:**
  - Dataset: GoPro (for deblurring), SSID (for denoising).
  - Inputs: degraded images, per dataset. Obtain corresponding ground truths.
  - Input size: e.g., 1280x720; model: Restormer, NAFNet.
  - Prepare data loaders: ensure that batch loading, normalization, optional augmentations match training setup.

**Additional setup:**
- Normalize inputs to \([0,1]\) range.
- For pixel-wise loss, use typical loss functions (cross-entropy for segmentation, MSE for optical flow, L2 or SSIM for restoration).
- Construct target tensors accordingly:
  - Segmentation: one-hot or class index labels.
  - Optical flow: flow vector fields.
  - Restoration: ground truth images.

---

## 3. Hyperparameters

- **Epsilon (\(\epsilon\)):** maximum perturbation. Use values consistent with experiments, e.g., \(\epsilon = \frac{8}{255}\) for \(\ell_\infty\), or as specified.
- **Step size (\(\alpha\)):** typically set proportionally to \(\epsilon\), e.g., \(\alpha=0.01\). Figures suggest \(\alpha=0.01, 0.001\) for ablation.
- **Number of Iterations (T):** range from 3 to 100; key results show effectiveness at small iteration counts (e.g., 3-10).
- **Scaling parameters:**
  - \(\psi\) softmax: use temperature variants (e.g., default 1).
  - For targeted attacks, scale cosine similarity as \(1 - \cos(\cdots)\); for untargeted, as \(\cos(\cdots)\).
- **Loss function \(\bar{L}\):**
  - Semantic segmentation: cross-entropy
  - Optical flow/regression: Euclidean loss or endpoint error
  - Image restoration: MSE or SSIM loss.

### Additional notes for stability:
- Use small \(\alpha\) to ensure convergence (see ablation in figures).
- Clip the perturbed \(X^{adv}\) after each update.
- For stability, optionally include a small decay or momentum term.

---

## 4. Algorithmic Details & Implementation Aspects

- The cosine similarity computation:
  - \(\psi(P)_i = \text{softmax}(P_i)\) for each pixel \(i\).
  - \(\cos(\psi(P)_i, Y_i) = \frac{\psi(P)_i \cdot Y_i}{\|\psi(P)_i\| \|Y_i\|}\).
  - Since \(Y_i\) is one-hot, \(\|Y_i\|=1\).
  - Normalize \(\psi(P)_i\): softmax ensures sum-to-one; the norm can be computed over the class dimension.

- Loss scaling:
  - Compute pixel-wise scaled loss: \(\cos(\psi(P)_i, Y_i) \cdot \bar{L}(P_i, Y_i)\).
  - Sum or mean over all pixels.

- Differentiability:
  - Use differentiable operations only; softmax, dot products, norms, and pixel-wise losses are all differentiable.
  - Use autograd to backpropagate aggregated scaled loss.

- For targeted attacks:
  - Replace \(\cos(\psi(P)_i, Y_i)\) with \(1 - \cos(\psi(P)_i, Y_i)\).
  - Scale loss accordingly to push predictions towards the target.

- Clip adversarial perturbations:
  - Use element-wise clipping to \([- \epsilon, + \epsilon]\) around the original clean sample.
  - For images: clip in [0,1].

---

## 5. Experimental Procedure & Evaluation

- **Generate adversarial samples:**
  - For each test image, perform the iterative CosPGD or baseline PGD/SegPGD attack.
  - Set number of iterations as per experimental plan (e.g., 3, 5, 10, 20, 40).

- **Metrics:**
  - For segmentation: mean IoU, pixel accuracy (as in paper), average pixel-specific loss.
  - For optical flow: endpoint error (EPE) averaged per image.
  - For restoration: PSNR, SSIM, averaged over test set.

- **Baseline & comparisons:**
  - Implement standard PGD attacks for comparison.
  - Implement SegPGD as in the paper for ablation.
  - Implement PCFA (if computing transfer robustness).

- **Results interpretation:**
  - Can include per-iteration plots (Figures 14-17), distribution plots (Figures 20-21), and tables (Tables 3-15).
  - Ablate \(\alpha\), \(\epsilon\), iteration Count, and loss functions as in the figures.

---

## 6. Additional Considerations

- **Code implementation details:**
  - Use PyTorch/TensorFlow for automatic differentiation.
  - Modularize: separate functions for:
    - Computing \(\psi(\cdot)\),
    - Cosine similarity,
    - Pixel-wise loss,
    - Attack iteration step,
    - Clipping.
  - Ensure numerical stability (use \(\epsilon\) inside norms to avoid division by zero).

- **Reproducibility:**
  - Fix random seeds.
  - Use consistent data splits and augmentation.
  - Use the same hyperparameters as in the paper/figures for comparable results.

- **Logging & plotting:**
  - Capture per-iteration metrics.
  - Save adversarial examples and predicted masks for qualitative assessment.

---

## 7. Summary of Implementation Roadmap:

1. Prepare datasets for semantic segmentation, optical flow, and image restoration tasks as per sections.
2. Implement the CosPGD attack:
   - Initialize with uniform noise.
   - For each iteration:
     - Compute model predictions.
     - Calculate \(\psi\) (softmax).
     - Calculate pixel-wise cosine similarity.
     - Compute scaled pixel loss.
     - Backpropagate to get gradient.
     - Update \(X^{adv}\) with sign of gradient scaled by \(\alpha\).
     - Clip to \(\ell_\infty\) (or \(\ell_2\), as needed).
3. Conduct attacks with varying iterations (\(3,5,10,20,40,100\)).
4. Evaluate attack effectiveness using the metrics specified.
5. Repeat for different models (DeepLabV3, PSPNet, UNet, NAFNet) and datasets.
6. Perform ablations and compare with baseline PGD/SegPGD/PCFA.
7. Generate figures, tables, and qualitative visuals as in paper supplementary.

---

This detailed plan should serve as a solid roadmap for implementing CosPGD and reproducing the experiments, even in the absence of official code. Further steps would involve translating each step into code, paying attention to the specific dataset preprocessing, normalization, and model inference details aligned with the original experiments.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Design a modular system using PyTorch. The core component is a CosPGD attack class that performs iterative pixel-wise scaling of the loss based on cosine similarity. Utilize torchvision datasets for segmentation and optical flow or custom loaders for image restoration datasets. Models (DeepLabV3, PSPNet, UNet, NAFNet) are loaded from pre-trained weights or checkpoints. The attack integrates the scaling operation into the loss computation, computing softmax predictions, pixel-wise cosine similarity, and scaled pixel loss, then updating inputs via sign gradient steps clipped to an epsilon ball. The system includes separate dataset loaders, model interfaces, and an attack runner to handle multiple configurations, hyperparameters, and evaluation metrics (IoU, pixel accuracy, EPE, PSNR, SSIM). Results are saved and plotted after attack iterations for analysis.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "attack.py",
        "utils.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +load_models()
        +load_datasets()
        +run_attacks()
        +evaluate_results()
    }
    class DatasetLoader {
        +__init__(dataset_name: str, data_path: str, task: str, split: str)
        +load_dataset() -> Dataset
    }
    class Model {
        +__init__(model_name: str, checkpoint_path: str)
        +predict(x: torch.Tensor) -> torch.Tensor
        +load_weights() -> None
    }
    class Attack {
        +__init__(model: Model, epsilon: float, alpha: float, max_iters: int, task: str)
        +attack(x: torch.Tensor, y: torch.Tensor, targeted: bool = False, target_label: torch.Tensor = None) -> torch.Tensor
        +compute_scaled_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor
        +update_input(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor
    }
    class Evaluation {
        +__init__(model: Model, dataset: Dataset)
        +compute_metrics() -> dict
        +save_results()
    }
    Main --> DatasetLoader
    Main --> Model
    Main --> Attack
    Main --> Evaluation
    Attack --> Model
    DatasetLoader --> Dataset
    Model --> TorchModel  # Underlying torch model object
    Attack --> torch
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant MD as Model
    participant AT as Attack
    participant EV as Evaluation
    Main->>DL: initialize()
    DL-->>Main: Dataset object
    Main->>MD: load_weights()
    MD-->>Main: pre-trained model
    Main->>AT: create(epsilon, alpha, max_iters, task)
    loop for each dataset image
        Main->>MD: predict(image)
        MD-->>Main: prediction
        Main->>AT: attack(image, label, targeted=false)
        AT-->>Main: adversarial image
        Main->>EV: evaluate(adversarial image, label)
        EV-->>Main: metrics
    end
    Main->>EV: save all results
    Main->>plot_results(): generate graphs
",
    "Anything UNCLEAR": "Clarify dataset formats: segmentation labels as class indices or one-hot? For optical flow, specific vector formats and normalization? Also specify exact hyperparameters used for attack (step size, iterations, epsilon). Confirm if models are custom or from torchvision, and checkpoint paths. Clarify if evaluation metrics implementations are standard or require custom code."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.9.0",
        "torchvision==0.10.0",
        "numpy==1.21.0",
        "scipy==1.7.0",
        "matplotlib==3.4.3",
        "PILLOW==8.2.0",
        "tqdm==4.62.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class, responsible for loading datasets (Pascal VOC, KITTI, GoPro, SSID) with correct formatting. Handles image reading, resizing, normalization, and label processing, providing an iterator over dataset samples. Handles train, validation, and test splits per experiment, supporting augmentation if needed."
        ],
        [
            "model.py",
            "Provides Model class, which loads model architectures (DeepLabV3, PSPNet, UNet, NAFNet) from given checkpoints. Implements predict(x: Tensor) method that runs inference, returning prediction tensors (logits for classification, flow vectors, images for restoration). Handles model initialization, loading weights, and any necessary device/parallelization setup."
        ],
        [
            "attack.py",
            "Implements Attack class, including methods for initializing input perturbation, performing iterative CosPGD updates. Key functions: compute_scaled_loss(pred, target) calculating cosine-scaled pixel loss, update_input(x, grad) applying gradient sign step with clipping. Supports targeted and untargeted modes, with parameters for epsilon, alpha, max iterations."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class with methods to compute metrics (IoU, pixel accuracy, EPE, PSNR, SSIM), possibly saving visual outputs. Supports evaluation on multiple datasets, saving results, and plotting metrics over iterations."
        ],
        [
            "main.py",
            "Entry point that orchestrates experiment: loads datasets via DatasetLoader, models via Model, sets attack parameters, runs attack via Attack class over dataset images, evaluates via Evaluation class, and saves/plots results. Supports configuration for models, datasets, attack hyperparameters. Controls the flow of training, attack, and evaluation."
        ],
        [
            "utils.py",
            "Contains utility functions: clipping tensors, computing cosine similarity, logging, plotting graphs, handling normalization, and possibly reproducibility helpers (seeds)."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "attack.py",
        "evaluation.py",
        "utils.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\nresponses:\n  200:\n    description: Success\n    content:\n      application/json:\n        schema:\n          type: object\n          properties:\n            metrics:\n              type: object\n              description: Dictionary of evaluation metrics\n            adversarial_examples:\n              type: array\n              description: List of adversarial images and predictions\n            attack_info:\n              type: object\n              description: Attack parameters and details",
    "Shared Knowledge": "Shared hyperparameters such as epsilon, step size, number of iterations, results storage paths. Utility functions for normalization, clipping, cosine similarity calculations, and progress logging used across attack and evaluation modules.",
    "Anything UNCLEAR": "Clarification needed on dataset label formats (class indices or one-hot), exact model checkpoint paths, and whether the evaluation metrics (IoU, PSNR, SSIM, EPE) need custom implementations or standard libraries. Also, hardware constraints or specific computational resources recommended."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
datasets:
  semantic_segmentation:
    name: PascalVOC2012
    root_dir: ./data/PascalVOC2012
    split: train
    augment: true
    input_size: 512  # Common size used in experiments
  optical_flow:
    name: KITTI2015
    root_dir: ./data/KITTI2015
    split: validation
  image_restoration:
    name: GoPro
    root_dir: ./data/GoPro
    split: train
    input_size: 1280x720

models:
  deeplabv3:
    backbone: resnet50
    checkpoint_path: ./checkpoints/deeplabv3_resnet50.pth
  psnet:
    backbone: resnet50
    checkpoint_path: ./checkpoints/psnet_resnet50.pth
  unet:
    encoder: ConvNeXt-tiny
    checkpoint_path: ./checkpoints/unet_convnext_tiny.pth
  nafnette:
    checkpoint_path: ./checkpoints/nafnette.pth

attack_parameters:
  epsilon: 8/255        # Max perturbation for l_infinity norm
  step_size: 2/255    # Small step size (e.g., 0.0078)
  attack_iters:
    - 3
    - 5
    - 10
    - 20
    - 40
    - 100
  targeted: false
  targeted_target: null

training:
  learning_rate: 0.01  # Default learning rate, can be adjusted
  batch_size: 16        # Typical batch size used in experiments
  epochs: 50            # Number of training epochs
  optimizer: Adam       # Optimizer used during training
  weight_decay: 1e-4    # Weight decay
  scheduler: StepLR     # Learning rate scheduler

evaluation:
  metrics:
    segmentation:
      IoU: true
      pixel_accuracy: true
    optical_flow:
      EPE: true
    restoration:
      PSNR: true
      SSIM: true

hardware:
  device: cuda:0
  num_gpus: 1
  seed: 42
```

---

**Note:** Full configuration is available in `planning_config.yaml`
