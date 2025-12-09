# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. To reproduce the experiments and methodologies in the paper "Binarized Diffusion Model for Image Super-Resolution," the plan must cover all critical details necessary for faithful replication, including model architecture, training procedures, hyperparameters, datasets, and evaluation procedures. Here is a comprehensive, structured outline:

---

## 1. Overall Approach and Key Objectives
- **Goal:** Implement BI-DiffSR, a binarized diffusion-based super-resolution model, with architecture, activation redistribution, and fusion strategies described.
- **Main Experimental Tasks:**
  - Build the optimized UNet-based noise estimation network.
  - Precisely binarize components, including weights and activations.
  - Implement the timestep-aware redistribution (TaR) and activation (TaA) modules.
  - Ensure dimension consistency, especially through CP-Down, CP-Up, and CS-Fusion modules.
  - Train and evaluate on standard super-resolution datasets with the diffusion process.

---

## 2. Model Architecture Details

### 2.1. UNet Structure for Noise Estimation
- **Input:**
  - Low-resolution (LR) image (conditioning input).
  - Noisy high-resolution (HR) image (diffusion process).
  - Timestep $t$ (encoded as embedding).
- **Encoder:**
  - 4 levels, each with 2 residual blocks.
  - Use consistent channels (initial: 64).
  - Downsampling via CP-Down modules:
    - For each level: apply convolution, CP-Down (see 2.1.2 for details).
  - Sharesholding: Use BinConv (binarized convolutions).
- **Decoder:**
  - 4 levels, each with 3 residual blocks.
  - Upsampling via CP-Up modules.
  - Skip connections with CS-Fusion.

### 2.2. Core Modules
- **CP-Down / CP-Up:**
  - Designed to maintain dimension consistency of full-precision features.
  - CP-Down:
    - Split input features into two (via $\text{PS}^{-1}$ or Pixel-UnShuffle).
    - Process each half with binarized convolutions.
    - Recombine for output.
  - CP-Up:
    - Concatenate two features processed with binarized convolutions.
    - Pixel-Shuffle to increase resolution.
- **CS-Fusion:**
  - For skip concatenations:
    - "Channel-shuffle" operation: split features into odd/even channels.
    - Concatenate odd channels from one feature with even channels from another.
    - Apply two binarized convolutions to fuse.
    - Ensures matching of value ranges and stable fusion.

### 2.3. Activation Redistribution and Activation Functions
- **TaR:**
  - Multiple bias-activation pairs for each $\text{RPReLU}$.
  - Timestep-based selection of bias/activation pair via grouping.
- **TaA:**
  - Similarly, multiple RPReLU functions per group.
  - Selects based on timestep group divisions.
- **Implementation:**
  - Discrete set of biases and RPReLU modules (e.g., $K=5$).
  - Use "indicator function" (e.g., $\lfloor \frac{K \times t}{T} \rfloor$) to select appropriate pair.
  
### 2.4. Binarized Convolution Block (Basic BI-Conv)
- Use binarized weights:
  - Full-precision weights ($\mathbf{w}^f$) scaled by mean absolute value.
  - Binary weights ($\mathbf{w}^b$): sign of scaled full weights.
- Binarized activations:
  - Sign function with learnable bias ($\mathbf{b}$).
  - Straight-through estimator (STE) for backprop.
- Operations:
  - Convolution via XNOR + bit-count (efficient binarized convolution).
  - Activation layer: RPReLU or sign-based thresholding.
- **Full-precision residual & shortcut links** are maintained to stabilize training.

---

## 3. Diffusion Process & Conditioning
- **Diffusion steps:**
  - 2000 total steps (as per experimental setup).
  - During inference, DDIM sampling with 50 steps.
- **Conditioning:**
  - LR images as conditions.
  - Input LR images are bicubically downsampled HR images for training.
- **Training Loss:**
  - L1 loss between predicted noise and true noise (standard for diffusion models).

---

## 4. Dataset and Data Preparation
- **Training datasets:**
  - DIV2K (~800 images), Flickr2K.
  - Data augmentation:
    - Random cropping (64×64 patches).
    - Rotation (90°, 180°, 270°).
    - Horizontal flips.
- **Testing datasets:**
  - Set5, B100, Urban100, Manga109.
- **Resolution settings:**
  - Scale×2 and ×4.
- **Generation of low-res images:**
  - Bicubic downsampling from HR ground truth.

---

## 5. Hyperparameters & Training Settings
- **Model hyperparameters:**
  - Channels: 64.
  - Encoder/decoder levels: 4.
  - ResBlocks per level: 2 (encoder), 3 (decoder).
  - Total timesteps $T=2000$.
  - DDIM sampling with 50 steps during inference.
- **Training Hyperparameters:**
  - Optimizer: Adam.
  - Learning rate: $1\times10^{-4}$.
  - Batch size: 16.
  - Total iterations: ~1,000,000.
- **Binarization:**
  - Weights scaled using mean absolute value.
  - Activation binarization via sign function + learnable bias.
- **Loss:**
  - L1 between predicted and true noise.
- **Training environment:**
  - PyTorch framework.
  - Hardware: NVIDIA A100 (or similar GPU).

---

## 6. Evaluation Metrics
- **Quantitative:**
  - PSNR and SSIM on the luminance channel (Y channel in YCbCr).
  - LPIPS (perceptual metric).
- **Qualitative:**
  - Visual comparison for challenge cases (Fig. 8, Fig. 1).
  - Visualize activation distributions across timesteps.
  - Report parameter count (Params) and OPs (operations per sampling step).

---

## 7. Reproducibility & Additional Details
- **Implementation:**
  - Provide detailed code structure for:
    - Data loader with augmentation.
    - Binarized convolution module (with precision scaling).
    - UNet with CP-Down/CP-Up and CS-Fusion.
    - Timestep encoding, TaR, and TaA modules.
    - Diffusion sampling with DDIM.
- **Hyperparameter tuning:**
  - Use the same $K=5$ bias/activation pairs.
  - Keep channels, level count, and blocks as specified.
- **Sanity Checks:**
  - Validate binarization only (e.g., compare to real-valued baseline).
  - Confirm matching of dimensions through CP modules.
  - Evaluate on small images first.

---

## 8. Additional Considerations & Clarifications
- The exact form of the "Mask" in Eq. (2.4) (distribution comparison) can be inferred as the difference in activation distributions before and after binarization, used primarily for analysis, not training.
- For the "consistent" CP modules, ensure the Pixel-UnShuffle/Shuffle operations are correctly scaled/implemented.
- When implementing the biases in TaR/TaA modules, initialize biases sensibly (e.g., zeros or small random).
- Carefully implement STE for sign gradient approximation.
- During evaluation, ensure the activation distributions are visualized as in Fig. 4, and compare with baseline models quantitatively as shown in Tables 1 and 2.

---

## Final Summary:
This plan provides a detailed blueprint:
- **Model architecture** grounded in the UNet with CP and CS modules.
- **Binarization scheme** with scaling and STE.
- **Activation modules** (TaR and TaA) for timestep adaptability.
- **Experimental procedure** from dataset preparation through training hyperparameters to evaluation.
- **Reproducibility mechanisms** via explicit mention of modules, hyperparameters, and evaluation protocols.

Once this roadmap is validated and fully documented, it will serve as a robust foundation for implementing the code and recovering the experimental results described in the paper.

---

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular Python system leveraging PyTorch and supporting libraries. The system will encompass components for dataset handling, model construction including binarized UNet with CP-Down/Up and CS-Fusion modules, training with diffusion and timestep modules, and evaluation. The core components include datasets with augmentation, binarized convolution operations, the UNet with residual, CP, and CS modules, timestep encoding with TaR and TaA, full training and inference workflows with DDIM sampling, and metrics calculation for PSNR, SSIM, and LPIPS. All code will be organized into clear, separated modules with a main script to coordinate execution.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "sampling.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "```mermaid\ngraph TD\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_paths: dict, batch_size: int)\n        +load_data() -> DataLoader\n    }\n    class UNetModel {\n        +__init__(channels: int, levels: int, res_blocks: int)\n        +forward(input: Tensor, timestep_embedding: Tensor, condition: Tensor) -> Tensor\n    }\n    class BinarizedConv {\n        +__init__(in_channels: int, out_channels: int)\n        +binarize_weights(weights: Tensor) -> Tensor\n        +forward(input: Tensor) -> Tensor\n    }\n    class CPDownModule {\n        +__init__()\n        +apply(input: Tensor) -> Tensor\n    }\n    class CPUpModule {\n        +__init__()\n        +apply(input: Tensor) -> Tensor\n    }\n    class CSSFusion {\n        +__init__()\n        +fuse(x1: Tensor, x2: Tensor) -> Tensor\n    }\n    class TimestepEncoder {\n        +__init__(timesteps: int, K: int)\n        +get_embedding(t: int) -> Tensor\n    }\n    class TaR {\n        +__init__(K: int)\n        +apply(x: Tensor, t: int) -> Tensor\n    }\n    class TaA {\n        +__init__(K: int)\n        +apply(x: Tensor, t: int) -> Tensor\n    }\n    class DiffusionSampler {\n        +__init__(model: UNetModel, total_steps: int, sampler_type: str)\n        +sample(condition: Tensor, low_res: Tensor, seed: int) -> Tensor\n    }\n    class Evaluation {\n        +__init__(model: UNetModel, datasets: dict, metrics: list)\n        +compute_metrics(pred: Tensor, target: Tensor) -> dict\n        +visualize_results(images: list)\n    }\n\n    Main --> DatasetLoader\n    Main --> UNetModel\n    Main --> DiffusionSampler\n    Main --> Evaluation\n    DatasetLoader --> DataLoader\n    UNetModel --> BinarizedConv\n    UNetModel --> CPDownModule\n    UNetModel --> CPUpModule\n    UNetModel --> CSSFusion\n    DiffusionSampler --> UNetModel\n    DiffusionSampler --> TimestepEncoder\n    TimestepEncoder --> TaR\n    TimestepEncoder --> TaA\n    Evaluation --> MetricCalculations\n```\n",
    "Program call flow": "```mermaid\sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant US as UNetModel\n    participant SD as DiffusionSampler\n    participant EV as Evaluation\n    M->>DL: initialize(dataset_paths, batch_size)\n    DL->>DL: load_data()\n    DL-->>M: return DataLoader\n    M->>US: build_model(channels=64, levels=4, res_blocks=2)\n    US-->>M: model instance\n    M->>SD: initialize(model, total_steps=2000, sampler_type='DDIM')\n    Note right of SD: set diffusion parameters\n    M->>SD: sample(condition=LR_image, low_res=LR_condition, seed=123)\n    SD-->>M: generated_high_res_image\n    M->>EV: initialize(model, datasets, metrics=['PSNR','SSIM','LPIPS'])\n    EV->>model: run inference on test sets\n    model->>EV: produce predictions\n    EV-->>M: report metrics and visualizations\n    Note right of EV: evaluate models; plot distributions\n```",
    "Anything UNCLEAR": "Clarification needed on the precise implementation details of CP modules, exact architecture depths, and the integration of the timestep encodings within the training API. Also, confirmation on dataset path formats and the exact diffusion sampling parameters (steps, DDIM settings) is required."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pillow>=8.0.0",
        "lpips",
        "tqdm"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class that handles loading datasets, applying augmentation (random crop, flips, rotations), and returning DataLoader objects. It depends on dataset paths, batch size, and transforms. Requires torchvision or custom transformation functions."
        ],
        [
            "model.py",
            "Defines UNet with residual blocks, CP-Down/Up modules, CS-Fusion, binarized convolution layers, and the timestep encoding network. Implements BinarizedConv class with weight scaling, STE gradient approximation, and binarization logic. Needs to connect to dataset_loader.py for data shape info."
        ],
        [
            "trainer.py",
            "Handles training loop, including model initialization, optimizer setup, data loading, diffusion process (adding noise, predicting noise), updating TaR and TaA modules, applying loss functions, and managing checkpoints. Depends on model.py for architecture and dataset_loader.py for data."
        ],
        [
            "sampling.py",
            "Implements diffusion sampling with DDIM, sampling steps, and integrating the trained model and timestep encoding modules. Uses model.py and the diffusion schedule parameters."
        ],
        [
            "evaluation.py",
            "Contains functions/classes to run inference on test datasets, compute PSNR, SSIM, LPIPS metrics, and generate visualizations. Depends on dataset_loader.py for test data, and model.py for inference."
        ],
        [
            "main.py",
            "Main script to coordinate dataset loading, model training, checkpoint management, inference, and evaluation orchestration. Instantiates dataset_loader, model, trainer, sampler, and evaluation classes; manages control flow."
        ],
        [
            "utils.py",
            "Utility functions including binarization helpers, activation distribution visualization, dataset augmentation methods, and metric calculation wrappers. Shared across modules."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "utils.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "sampling.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "",
    "Shared Knowledge": "Common functions for binarization (sign, STE), dataset augmentation routines, positional encoding for timesteps, and metric calculation functions (PSNR, SSIM, LPIPS) are shared across modules.",
    "Anything UNCLEAR": "Clarification needed on the exact implementation of CP modules, detailed architecture depth, and the precise form of the timestep grouping and bias/activation pair selection logic for TaR/TaA modules."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0001
  batch_size: 16
  total_iterations: 1000000
  iterations_per_epoch: 5000
  image_crop_size: 64
  augmentation:
    flip: true
    rotations: [90, 180, 270]
  optimizer:
    type: Adam
    betas: [0.9, 0.99]
evaluation:
  scale: 2  # upscaling factor
  datasets:
    train:
      - DIV2K
      - Flickr2K
    test:
      - Set5
      - B100
      - Urban100
      - Manga109
  metrics:
    - PSNR
    - SSIM
    - LPIPS
diffusion:
  total_timesteps: 2000
  sampler_type: DDIM
  inference_timesteps: 50
model:
  channels: 64
  encoder_levels: 4
  res_blocks_per_level: 2
  decoder_res_blocks: 3
  timestep_encoding_K: 5
  binarization:
    scale_weights: true
    bias_pairs: 5
optimizer:
  type: Adam
  lr: 0.0001
  betas: [0.9, 0.99]
```

---

**Note:** Full configuration is available in `planning_config.yaml`
