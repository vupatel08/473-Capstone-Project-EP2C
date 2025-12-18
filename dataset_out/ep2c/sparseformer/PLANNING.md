# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Below is a comprehensive, structured outline for reproducing "SPARSEFORMER: SPARSE VISUAL RECOGNITION VIA LIMITED LATENT TOKENS," covering both core methodology and experimental setups based on the paper's content and figures.

---

## 1. OVERALL OBJECTIVE AND HIGH-LEVEL DESIGN
- Implement SparseFormer with a **sparse architecture** that learns to represent images with **few latent tokens** (e.g., 9, 16, 25, ...).
- The model should process images efficiently by:
  - Performing sparse feature sampling in the image space guided by token RoIs.
  - Using a combination of **focusing transformer** (for region sampling and RoI refinement) and **cortex transformer** (for deep feature processing).
  - Maintaining a fixed number of tokens (e.g., 49) across stages, with a single shared pixel embedding map.
- Train models from scratch on ImageNet-1K and fine-tune pre-trained variants.
- Extend the framework for video inputs (optional, if resources permit).

---

## 2. KEY METHODOLOGY COMPONENTS & DETAILED STEPS

### 2.1. Data Preparation
- **Dataset**: ImageNet-1K for classification. Also consider ImageNet-21K pretraining if replicating scaling experiments.
- **Image pre-processing**:
  - Resize images such that their short edge is 224 pixels as default.
  - Normalize images with ImageNet mean/std.
- **Data augmentations**:
  - Random crop, resize, horizontal flip for training.
  - Validation: Center crop/testing with fixed resolution.
- **For video extension**: sample 8 frames per clip with temporal sampling strategy provided.

### 2.2. Model Architecture
**2.2.1. Backbone: SparseFormer**
- **Input Layer**:
  - Lightweight initial convolution to produce a shared feature map at resolution \( H/4 \times W/4 \).
- **Latent Tokens**:
  - Initialize with learnable embeddings \(\mathbf{t}_i\), and associated RoIs \(\mathbf{b}_i\).
  - Number of tokens \(N\) varies across variants: 9, 16, 25, 49, 64, 81, 144, 196.
  - RoIs are parameterized by 4 numbers: center \((x,y)\), width \(w\), height \(h\), normalized to [0,1].
- **Focusing Transformer (for each stage)**:
  - Use a small number of repetitions (e.g., 1 to 4 times).
  - For each token:
    - Generate \(P\) sampling points via a learnable linear layer conditioned on token embedding:
      - Offsets \(\{ \Delta x_i, \Delta y_i \}\); then absolute locations relative to RoI.
    - Bilinearly sample features at these points from the shared feature map.
  - Update token RoIs via an iterative regression-like mechanism:
    - Generate adjustments \(\Delta t_x, \Delta t_y, \Delta t_w, \Delta t_h\) via a linear layer on token embedding.
    - Update RoIs using equations (x', y', w', h') with exponential scaling for size.
- **Adaptive Feature Decoding**:
  - For each token:
    - Use a lightweight network \(\check{\mathcal{F}}\) to produce spatial and channel weights.
    - Decode sampled features with GELU-activated linear transformations.
- **Cortex Transformer (standard encoder)**:
  - Consist of multiple layers (e.g., 12-24).
  - Take token embeddings as input after focusing.
- **Output**:
  - Average token embeddings after final stage.
  - Pass through a linear classifier (fully connected layer) for classification.

### 2.3. Sparse Feature Sampling Details
- Generate a **fixed grid of sampling points** per token during training (e.g., 36 points).
- Generate sampling points dynamically via the token-dependent offsets.
- Use bilinear interpolation on the shared feature map to extract regional features efficiently.
- Use **normalized RoI parameters** and convert to absolute pixel locations for sampling.
- Implement multiple stages with recursive RoI adjustment and feature sampling (stage 1-4 in Fig. 2).

### 2.4. RoI Refinement and Sampling
- **Initialization**:
  - RoIs initialized to cover the entire image or uniform regions.
- **Refinement loop**:
  - For each focusing iteration:
    - Generate local sampling points conditioned on token embedding.
    - Extract features via bilinear interpolation.
    - Compute RoI adjustments.
    - Update RoIs to focus on salient regions.
- **Final stage**:
  - Use the refined RoIs closely aligned with foreground objects.
   
### 2.5. Model Hyperparameters & Sizes
- Variants (tiny, small, base) settings per Table 1:
  - Token count \(N\)
  - Token embedding dimension \(d_f\), \(d_c\)
  - Number of cortex transformer layers \(L_c\)
  - Number of focusing transformer repetitions \(L_f\)
  - Sampling points \(P\) (e.g., 16, 36, 64, 81)
  - Use certain hyperparameters for early convolution, decoders, etc.
- Use the provided FLOP and parameter estimates as reference.
  
### 2.6. Losses & Optimization
- **Classification loss**: Cross-entropy on the average token embedding.
- **RoI refinement guidance**:
  - No explicit supervision, trained end-to-end.
- **Training settings**:
  - Pretrain on ImageNet-21K if needed (e.g., 300 epochs).
  - Fine-tune on ImageNet-1K (e.g., 50 epochs).
  - Use AdamW, cosine scheduler, weight decay of 0.05.
  - Learning rate warmup for initial epochs (e.g., 5 epochs, start LR ~1e-3 reduce to 1e-5).
  - Batch size: 128 or more, distributed training.
  - Gradient clipping may be beneficial.
  
---

## 3. EXPERIMENTAL SETUPS AND EVALUATION METRICS
- **Dataset**:
  - ImageNet-1K: 1.28M train, 50K val.
  - Optional pretraining on ImageNet-21K.
- **Training**:
  - Use standard data augmentation (random resized crop, flip).
  - For video applications: frames sampled uniformly, same model architecture with extended temporal dimension.
- **Evaluation**:
  - Top-1 accuracy on the validation set.
  - FLOPs (measured in Giga flops with batch size=1) for efficiency comparison.
  - Throughput: images/sec on a standard GPU (e.g., V100, A5000).
  - Extra metrics: mIOU for segmentation if extended.

---

## 4. ADDITIONAL DETAILS AND UNCERTAINTIES
- **Initializing token embeddings**: learnable parameters; possibly initialize to uniform or Kaiming normal, as suggested in implementation notes.
- **RoI parameters**:
  - Normalized coordinates for training; convert to pixel locations dynamically.
  - RoIs refined iteratively, starting wide.
- **Number of emphasis samples** per token (P): 36, 49, 64, or 81, depending on variant.
- **Number of transformer layers**:
  - Focusing transformer: 1-4 repetitions.
  - Cortex transformer: 12-24 layers.
- **Data augmentation**:
  - Training: standard ImageNet augmentations.
  - For extended experiments, consider cropping, cropping size, and temporal sampling as per the paper.
- **Implementation notes**:
  - Bilinear sampling from the shared feature map.
  - Use position normalization and learnable RoI offsets.
  - Adaptive decoder with GELU and linear layers.
  - Carefully implement RoI update equations for stability.
- **Training stability**:
  - Use small learning rate warmup and weight decay.
  - Dropout or stochastic depth optional.
- **Scaling experiments**:
  - Replicate Table 4 by pretraining the base model on ImageNet21K and fine-tuning on ImageNet1K.

---

## 5. SUMMARY OF THE ROADMAP
- Implement the core components: shared conv, token initialization, sparse feature sampling, RoI updating, decoders, focusing transformer, cortex transformer, classifier.
- Follow the hyperparameters, sample points, and number of tokens as per variant.
- Train on ImageNet accordingly, verify FLOPs and accuracy.
- Visualize token RoIs and sampling density for interpretability.
- Extend to video by adding temporal sampling, possibly multi-frame features.

---

This roadmap ensures a precise, detailed comprehension of the methodologies and experimental configurations. It guides the subsequent implementation phase and allows tracing from the original paper to your code with fidelity.

Would you like me to proceed with a specific detailed module plan next?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement SparseFormer using PyTorch, leveraging existing modules for transformer blocks, bilinear sampling, and data handling. The system will consist of a shared backbone convolution, a set of learnable token embeddings, focusing transformer modules for sparse feature sampling and RoI refinement, a cortex transformer encoder, and a final classification head. The key modules will be: (1) Token initialization with learnable embeddings and RoIs, (2) Sparse feature sampling via bilinear interpolation from the shared feature map based on dynamically generated sampling points conditioned on token embeddings, (3) RoI refinement using iterative adjustments, (4) adaptive feature decoding using lightweight MLPs, (5) focusing and cortex transformers combining cross and self-attention, and (6) training scripts for end-to-end optimization and evaluation.",
    "File list": [
        "main.py",
        "dataset.py",
        "model.py",
        "train.py",
        "utils.py",
        "visualize.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class Dataset {\n        +__init__(config: dict)\n        +load_data() -> Dataset\n    }\n    class SparseFormer {\n        +__init__(params: dict)\n        +forward(images: Tensor) -> Tensor\n        +sample_features(tokens: Tensor, feature_map: Tensor) -> Tensor\n        +refine_rois(tokens: Tensor) -> Tensor\n        +decode_features(sampled: Tensor, tokens: Tensor) -> Tensor\n        +get_token_embeddings() -> Tensor\n    }\n    class TokenEmbedding {\n        +__init__(num_tokens: int, embed_dim: int)\n        +initialize() -> None\n        +get_embeddings() -> Tensor\n        +update_rois(rois: Tensor) -> Tensor\n        +refine_rois(tokens: Tensor) -> Tensor\n    }\n    class FocusingTransformer {\n        +__init__(layers: int)\n        +apply(tokens: Tensor, feature_map: Tensor, rois: Tensor) -> Tensor\n        +iterate_sampling(tokens: Tensor, feature_map: Tensor, rois: Tensor, stages: int) -> Tensor\n    }\n    class CortexTransformer {\n        +__init__(layers: int)\n        +encode(tokens: Tensor) -> Tensor\n    }\n    class ClassifierHead {\n        +__init__(input_dim: int, num_classes: int)\n        +forward(tokens: Tensor) -> Tensor\n        +predict() -> Tensor\n    }\n    Main --> Dataset\n    Main --> SparseFormer\n    SparseFormer --> TokenEmbedding\n    SparseFormer --> FocusingTransformer\n    SparseFormer --> CortexTransformer\n    SparseFormer --> ClassifierHead\n    TokenEmbedding --> RoIs\n    FocusingTransformer --> TokenEmbedding\n    CortexTransformer --> Tokens\n    ClassifierHead --> Tokens\n    Sampled features --> BilinearSampler\n    RoIs --> TokenRoIs\n",
    "Program call flow": "sequenceDiagram\n    participant C as Main\n    participant D as Dataset\n    participant M as SparseFormer\n    participant T as TokenEmbedding\n    participant F as FocusingTransformer\n    participant R as RoRefinement\n    participant C2 as CortexTransformer\n    participant H as ClassifierHead\n    C->>D: load data\n    D-->>C: dataset object\n    C->>T: initialize tokens & RoIs\n    T-->>C: token embeddings & RoIs\n    C->>M: forward(images)\n    M-->>C: logits or embeddings\n    C->>H: predict labels\n    H-->>C: predicted class\n    C->>C2: encode tokens\n    C2-->>C: encoded tokens\n    C->>F: focus sampling & refinement\n    F-->>T: update RoIs\n    F-->>M: sample sparse features based on RoIs\n    M-->>T: decode features\n    loop for multiple stages\n        F->>R: refine RoIs\n        R-->>T: updated RoIs\n        M-->>T: sample features again\n    end\n    Note right of C: Train end-to-end with cross-entropy loss.\n",
    "Anything UNCLEAR": "Clarify precise configurations for the number of attention layers (focusing and cortex), sampling points P, and the depth of the adaptive decoding network. Need confirmation on initialization strategies for token RoIs and embeddings. Also, confirm dataset pipeline specifics, e.g., augmentations and batching details for maximum fidelity."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "torchvision==0.12.0",
        "numpy==1.21.0",
        "tensorboard",
        "matplotlib"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset.py",
            "Handles dataset loading, preprocessing, augmentation, and batching for ImageNet-1K. Implements a Dataset class that returns images and labels, with support for validation/test splits and optional data augmentation pipeline."
        ],
        [
            "model.py",
            "Defines SparseFormer class implementing the entire architecture: shared conv backbone, token embeddings, focusing transformer (sampling points generation, RoI refinement), adaptive decoder, cortex transformer encoder, and classification head. Includes all submodules with clear interfaces."
        ],
        [
            "train.py",
            "Orchestrates training loop, including data loading, model forward pass, loss computation, backpropagation, optimizer step, and scheduler update. Implements multi-stage training with pretraining on ImageNet21K, and fine-tuning on ImageNet1K. Uses mixed precision and gradient clipping as needed."
        ],
        [
            "evaluation.py",
            "Provides evaluation functions to compute top-1 accuracy, FLOPs, throughput measurement, and visualization of token RoIs and feature sampling density. Includes functions to run validation periodically and produce interpretability visualizations."
        ],
        [
            "utils.py",
            "Contains utility functions: bilinear sampling from feature maps, RoI update calculations, learning rate schedules, model weight initialization, and logging."
        ],
        [
            "main.py",
            "Main entry point that parses config, initializes dataset, model, optimizer, scheduler, and trainer. Runs training loop, evaluation, and visualization. Coordinates pretraining and fine-tuning phases based on config flags."
        ]
    ],
    "Task list": [
        "dataset.py",
        "model.py",
        "utils.py",
        "train.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "None (standalone training and evaluation scripts without external API dependencies).",
    "Shared Knowledge": "Shared modules include bilinear sampling, RoI parameterization and update functions, linear and MLP decoders, transformer blocks, normalization layers, and token embedding management. Hyperparameters are configurable via a dict or YAML config.",
    "Anything UNCLEAR": "Clarify the exact configuration details for the number of focus iterations, sampling points P per token, and the number of transformer blocks (focusing and cortex). Confirm initialization schemes for token embeddings and RoIs. Need confirmation on the evaluation protocol for FLOPs and throughput measurement."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: ImageNet1K
  data_dir: /path/to/imagenet/
  image_size: 224
  train_split: train
  val_split: val
  num_workers: 8
  augmentation:
    - random_resized_crop: true
    - horizontal_flip: true
    - normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]

model:
  variant: base        # options: tiny, small, base
  num_tokens: 81       # number of fixed tokens
  token_dim: 768       # dimension of token embeddings (d_c)
  focusing_layers: 1   # repetitions of focusing transformer
  cortex_layers: 12    # number of cortex transformer encoder layers
  sampling_points: 36  # number of sampling points P
  stage_repeats: 1     # repetitions of focusing stage
  roi_initialization: 'grid' # or 'full', default is 'grid' covering the image

training:
  epochs: 50
  batch_size: 128
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.05
  scheduler:
    name: cosine
    warmup_epochs: 5
    warmup_lr: 0.001
  gradient_clip_norm: 1.0
  mixed_precision: true
  gradient_accumulation_steps: 1
  total_steps: 30000    # approximate, scaled with dataset size

pretraining:
  dataset: ImageNet21K
  epochs: 300
  lr: 0.001
  batch_size: 512
  optimizer: AdamW
  weight_decay: 0.05
  scheduler: cosine
  warmup_epochs: 5

evaluation:
  metrics:
    - top1_accuracy
    - FLOPs (measure on batch size 1)
    - throughput (images/sec on single GPU)
  visualize_token_rois: true
  visualize_sampling_density: true

hardware:
  gpus: 1  # or multi-GPU setup if available

saving:
  checkpoint_dir: './checkpoints/'
  save_every_epochs: 10
  resume_from_checkpoint: false
```

---

**Note:** Full configuration is available in `planning_config.yaml`
