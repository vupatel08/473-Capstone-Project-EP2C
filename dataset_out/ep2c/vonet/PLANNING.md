# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

### Reproduction Plan for VONet (Unsupervised Video Object Learning)

---

## 1. Overview of the Methodology

**Goal:** Reproduce VONet's unsupervised object learning from videos, adhering to the detailed architecture, training, and evaluation strategies outlined by the authors.

**Core Components:**

- **Feature Extraction Backbone:** CNN shared for all components.
- **Parallel Attention U-Net:** Efficiently produces attention masks for all object slots simultaneously.
- **Slot Encoder & Context:** Encode image-region features, keep track of slot history via RNN, derive context vectors for temporal consistency.
- **Object-wise Sequential VAE:** Enforces temporal consistency via slot trajectory modeling.
- **Transformer-Based Decoder:** Decodes scene composition, leveraging object representations, and capturing complex interactions.
- **Training & Optimization:** Use variational inference with KL regularization, with curriculum KL annealing and replay buffers.
- **Inference & Evaluation:** Generate segmentation masks, derive FG-ARI and mIoU, visualize results.

---

## 2. Dataset Details and Setup

- **Datasets:** MOVI-A, MOVI-B, MOVI-C, MOVI-D, MOVI-E
  - Video frames at 128×128 resolution.
  - 24 frames per video.
  - Object count up to 10 or 23.
  - Use official splits, with validation split as test set.
  
- **Preprocessing:**
  - Resize to 128×128.
  - No data augmentation unless explicitly decided (authors did not perform augmentation).
  - Standard normalization (not specified, but safe to normalize pixel values to [0,1]).

- **Data Loader:**
  - Load videos as sequences of frames.
  - For training, sample short segments (length=3 frames) with shuffling.
  - For evaluation, process full videos, extracting masks and metrics over all frames.

---

## 3. Architectural Components & Implementation Details

### (a) Backbone CNN for Image Features

- Use a CNN similar to ResNet or a custom shallow CNN that outputs a feature map of shape `[batch, channels, 128, 128]`.
- Share backbone with attention module and slot encoder (via weight sharing).

### (b) Parallel Attention U-Net

- **Input:** Backbone feature map + context vectors for each slot.
- **Process:**
  - For each slot `k`, generate an estimated attention mask via a slot attention-like procedure:
    - Convolve context vectors with a small kernel (e.g., 3×3, stride=1, padding=1).
    - Initial attention logits per slot, shape `[batch, K, 128, 128]`.
  - U-Net architecture:
    - Downsample path: 5 residual blocks with conv layers, each followed by layer norm + ReLU.
    - Communication among slots at the bottleneck via a transformer decoder:
      - Flatten downsampled features `[batch * K, channels, H', W']` into sequence `[batch * K, (H'×W'), channels]`.
      - Transformer decoder: 3 layers, 3 heads, with no position encodings to ensure slot permutation invariance.
      - Output: sequence of mask embeddings `[batch * K, (H'×W'), embed_dim]`.
    - Upsample paths: decode each slot embedding to produce `[batch, K, 128, 128]` mask logits.
  - Apply pixel-wise softmax across [K+1] (K masks + null class, threshold at 0.3).

### (c) Slot Encoder & Context Vectors

- **Slot Encoder:** For each slot:
  - Element-wise multiply feature map with the soft attention mask.
  - Apply an MLP + BatchNorm + LayerNorm to get per-slot feature vector `y_{t,k}` (dimension 128).
- **Context:**
  - For each slot, maintain a per-trajectory latent `r_{t,k}`.
  - Initialize: sample from Gaussian N(0, I) for all slots at t=0.
  - Update after each frame via an RNN (LSTM or GRU), taking slot features and previous `r_{t-1,k}`.
  - Context vector `c_{t,k}` is obtained from a slot encoder (MLP + Masked input).

### (d) Sequential Object VAE (Temporal Slot Modeling)

- **Per object slot:**
  - Posterior: encode `z_{t,k}` from `r_{t,k}` via an MLP to produce Gaussian mean and log-variance.
  - Prior: predict from previous occupancy using a transformer (Eq.7), producing `r'_{t,k}` (predictive slot state).
  - Use reparameterization trick for `z_{t,k}` during training.
- **Loss:** ELBO with KL divergence, scaled by a coefficient `beta` (annealed).

### (e) Transformer Decoder for Scene Decoding

- **Input:** All slot latents `z_{t,k}`.
- **Architecture:**
  - Transformer decoder attending over all slots simultaneously.
  - Autoregressive decoding of pixel patches or image grid (as in Singh et al. STEVE).
  - Produce reconstructed image tensor (e.g., via a multi-head self-attention decoder).
- **Output:** Reconstructed scene, matching input frames (for loss calculation).

---

## 4. Loss Function & Optimization Strategy

### (a) Variational Loss

- **Reconstruction loss:** Log likelihood of input image given slot representations, modeled with a transformer decoder as in Singh et al.
- **KL Divergence:** Between posterior `q(z_{t,k}|r_{t,k})` and prior `p(z_{t,k}|r'_{t,k})`.
  - KL scaled with a coefficient `beta`, annealed over training.
- **Total loss:** Sum over timesteps and batch:
  - `Loss = sum_t [-log P_dec(x_t | z_{t,1:K}) + beta * KL(q(z_{t,k}|r_{t,k}) || p(z_{t,k}|r'_{t,k}))]`

### (b) Curriculum KL Annealing

- Start with `beta=0` at early iterations, increase linearly to `0.7` over first 50k steps.
- Keeps the model focusing initially on reconstruction, then enforcing temporal consistency.

### (c) Replay Buffer for Long-term Consistency

- Store slot states (`r_{t,k}` and associated variables) for 10k frames (~16 video-lengths).
- During training:
  - For each mini-batch:
    - Sample 16 segments from buffer.
    - Run VONet with initialized states from buffer.
    - Update states based on current step.
  - Replaces buffer segments periodically.
- This ensures i.i.d. training and better long-term consistency.

### (d) Optimizer & Hyperparameters

- Use Adam (lr schedule: warm-up to 1e-4, decay).  
- Batch size: 32 for MOVI-A/B/C, 24 for MOVI-D/E.
- Update: 150k total steps, batch of 3 segments (size depends on dataset).
- Gradient clipping: norm to 0.1.
- Loss weights: Diagrammed in Appendix; keep consistent.

---

## 5. Training Strategy

- **Initialization:**
  - Random seed for slot initial latents (128-D Gaussian noise).
  - Initialize KL coefficient `w` from 0 to 1 over first 50k steps.
  
- **Unroll:**
  - On each segment, run VONet forward for 2 frames, update states, collect losses.
- **Replay:**
  - Draw from buffer to initialize slot states.
- **Buffer Management:**
  - Maintain a 10k frame buffer.
  - Replace sampled segments periodically.
  
- **Monitoring:**
  - Track FG-ARI & mIoU on validation.
  - Monitor KL divergence, reconstruction loss.
  - Visualize attention masks and masks over frames periodically.

---

## 6. Evaluation & Metrics

- **Segmentation Masks:**
  - Threshold attention maps at 0.3.
  - Assign pixels to nearest slot or null (background).
- **Metrics:**
  - FG-ARI (foregorund clustering similarity).
  - mIoU (Hungarian matching of predicted masks to groundtruth).
  - Compute over entire video or per frame, averaging results.
- **Post-processing:**
  - No masks smoothing or smoothing post-processes if following authors' protocol.
- **Visualization:**
  - Overlay masks on frames.
  - Plot KLD over time.
  - Show attention masks, reconstructed scenes, and segmentation contours.

---

## 7. Additional Implementation Details & Hyperparameters

- **Number of Slots:**
  - 11 slots for MOVI-A/B/C; 16 for MOVI-D/E.
- **Slot Embedding Size:** 128 dimensions.
- **Transformer Details:**
  - Attention heads: 3.
  - Layers: 3 for slot transformer, 2 for prior transformer.
  - No position encodings (slot permutation invariance).
- **Decoder:**
  - Use transformer decoder akin to Singh et al. STEVE.
  - Autoregressive pixel modeling.
- **Slot Initialization:**
  - Random Gaussian (`epsilon_k` size=128) for each slot.
- **Training:**
  - 3 frames per segment.
  - Run at 4× GPU (e.g., 4×3090 or similar).
  - Approximate training time: ~36 hours per run (authors).

---

## 8. Summary of Key Hyperparameters & Settings (from Appendix & Main text)

| Parameter                  | Value / Range                                   |
|----------------------------|-------------------------------------------------|
| Input resolution           | 128×128 pixels                                |
| Number of slots            | 11 (A/B/C), 16 (D/E)                            |
| Segment length during train | 3 frames                                      |
| Batch size                 | 32 (A/B/C), 24 (D/E)                            |
| Learning rate schedule     | Linear warm-up to 1e-4, plateau, decay        |
| KL annealing              | 0→0.7 over 50k steps                         |
| Slot embedding size        | 128 dimensions                                |
| Transformer layers         | 3 for mask, 2 for prior                      |
| Attention heads            | 3 per transformer layer                     |
| Replay buffer length       | 10,000 frames (~16 videos)                   |
| Number of training steps   | 150,000 (approximate)                        |

---

## 9. Key Challenges & Clarifications

- **Attention Masks & U-Net:** Precise U-Net architecture details are inferred; should replicate MONet’s and Singh et al.’s of similar size.
- **Transformer Communication among Slots:** No positional encodings; ensure permutation invariance.
- **Temporal Consistency:** Slot trajectory RNNs and KL regularization as described.
- **No Official Code:** Prototype implementation should mirror described architecture closely; hyperparameters should match those in Appendix.

---

## 10. Final notes

- Once architecture prototypes are set, implement the training loop with replay buffer, KL annealing, and multi-object positional modeling.
- Focus on modularity: feature extractor, attention module, slot encoder, prior & posterior transformers, decoder.
- Progressively train, monitor, and visualize intermediate outputs: attention masks, scene reconstructions, KL trends.
- Use the reported metrics and visualization scripts from the appendices for validation.

---

**This plan provides a detailed, step-by-step outline to implement VONet faithfully, covering all major components, training strategies, hyperparameters, and evaluation protocols.**

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement VONet using PyTorch for core model components, including CNN backbone, U-Net attention module, slot encoders, transformers, and decoder. Training will be modular: a DataLoader handles sequence sampling, the model class encapsulates all subnetworks, the trainer manages forward passes, loss calculation, KL annealing, replay buffer updates, and optimizer steps. Evaluation modules generate masks, compute metrics, and visualize masks. Hyperparameters are configured in a config file, ensuring easy adjustments. The system architecture maintains simplicity by modularizing the key components, and effectively reuses transformers and CNNs from open-source implementations where applicable.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "config.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_paths: dict)\n        +load_data(split: str) -> Dataset\n    }\n    class Dataset {\n        +__init__(data: list)\n        +__getitem__(index: int) -> Tuple[Tensor, Tensor]\n        +__len__() -> int\n    }\n    class Model {\n        +__init__(params: dict)\n        +extract_features(x: Tensor) -> Tensor\n        +generate_attention(features: Tensor, context: Tensor) -> Tensor\n        +encode_slots(features: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]\n        +predict_slot_prior(r_prev: Tensor) -> Tensor\n        +decode_scene(z_slots: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, optimizer: Optimizer, data_loader: DataLoader, config: dict)\n        +train_epoch() -> None\n        +run_training() -> None\n        +update_replay_buffer(states: List[Tuple]) -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, dataset: Dataset, config: dict)\n        +generate_masks(x: Tensor) -> Tensor\n        +compute_metrics(pred_masks: Tensor, gt_masks: Tensor) -> dict\n        +visualize_masks(x: Tensor, masks: Tensor) -> None\n    }\n\nRelations:\nMain -->> DatasetLoader\nMain -->> Trainer\nTrainer -->> Model\nEvaluation -->> Model\nDataset -->> Data loader via __getitem__ and __len__\nModel contains feature extractor, attention U-Net, slot encoders, transformer prior/posterior, and decoder components\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant D as Dataset\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: load_dataset(split='train')\n    DL-->>M: return dataset\n    M->>MD: initialize model + hyperparameters\n    M->>TR: start training()\n    TR->>TR: for epoch in epochs:\n        activate replay buffer\n        for batch in data loader:\n            extract batch data (x, gt_masks)\n            forward through model:\n                features = MD.extract_features(x)\n                context_vecs = get(past slot states)\n                attention_masks = MD.generate_attention(features, context_vecs)\n                masks = softmax(attention_masks)\n                slot_feats, slot_states = MD.encode_slots(features, masks)\n                z_slots = encode_posteriors(slot_states)\n                r_prior = MD.predict_slot_prior(previous_r)\n                kl_loss = compute_KL(q(z_slots), p(r_prior))\n                recon = MD.decode_scene(z_slots)\n                loss = reconstruction_loss(x, recon) + kl_loss * Beta\n            optimizer.zero_grad()\n            loss.backward()\n            optimizer.step()\n            update replay buffer with current slot states and r states\n    M->>EV: evaluate()\n    EV->>EV: generate masks for validation data\n    EV->>EV: compute FG-ARI, mIoU, visualize masks\n    EV-->>M: report metrics\n",
    "Anything UNCLEAR": "Need clarification on the precise network hyperparameters, especially transformer layer sizes, attention heads, and the architecture details of the U-Net (depth, channels), as well as dataset formats for the loader."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.0",
        "torchvision==0.13.0",
        "numpy==1.21.0",
        "scipy==1.7.3",
        "jsonlines==3.0.1",
        "tqdm==4.62.3",
        "einops==0.3.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains Dataset and DataLoader classes that load video sequences, handle batching, sequence sampling, and preprocess images to 128x128. It also manages dataset splits, handles the replay buffer states for temporal consistency, and yields tuples of tensor images and masks."
        ],
        [
            "model.py",
            "Defines core neural network modules including:\n- CNN backbone: a shared feature extractor.\n- Parallel U-Net attention module: with downsampling, transformer communication, and upsampling to produce K+1 masks.\n- Slot encoder: multiplies masks with features, encodes into slot features.\n- Slot trajectory RNN: updates prior slot states with GRU.\n- Variational posteriors: compute from slot states.\n- Prior transformers: predict slot trajectories.\n- Transformer decoder: autoregressively reconstructs scene from slot embeddings.\nImplementation emphasizes modularity and reuse of open-transformer libraries."
        ],
        [
            "trainer.py",
            "Manages training loop: initializes model, optimizer, and scheduler, runs epochs. Uses replay buffer for slot states, computes ELBO loss with reconstruction and KL terms, applies curriculum KL annealing, performs backpropagation, and updates model parameters. Handles the update of replay buffer with current slot states after each batch."
        ],
        [
            "evaluation.py",
            "Provides functions to generate attention masks, reconstruct scenes, and compute metrics FG-ARI and mIoU. Handles mask thresholding, Hungarian matching, and visualization overlays. Supports evaluation over entire videos by aggregating per-frame predictions."
        ],
        [
            "main.py",
            "Entry point: reads config, initializes dataset loader, model, trainer, and evaluation. Runs full training loop with periodic validation, saves models, plots metrics and attention masks, and produces final visualizations of segmentation masks on validation/test videos."
        ],
        [
            "utils.py",
            "Provides auxiliary functions: KL annealing schedule, visualization helpers, metrics computation, and utility functions for tensor shape management, data normalization, and mask post-processing."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "N/A – No backend API needed, the system runs end-to-end in Python for training/validation.",
    "Shared Knowledge": "Shared hyperparameters (e.g., slot count, embedding size), common modules (transformer blocks, residual CNNs), and the replay buffer logic are used across trainer.py and evaluation.py. Configurations are centralized in config.py and imported by main.py to ensure consistency.",
    "Anything UNCLEAR": "Clarification needed on dataset format specifics, especially how ground truth masks are stored for evaluation, and whether domain-specific visualization preferences are required."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: MOVI
  resolution: 128x128
  frames_per_video: 24
  dataset_split: official_split
  object_max_count: 10 # for MOVI-A/B/C, set to 16 for D/E
  is_realistic_scene: true

training:
  total_steps: 150000
  batch_size: 32  # for MOVI-A/B/C; use 24 for D/E
  segment_length: 3
  learning_rate_schedule:
    warmup_steps: 5000
    max_lr: 1e-4
    plateau_steps: 100000
    decay_steps: 50_000
  initial_lr: 1e-5
  final_lr: 1e-5
  warmup_lr: 1e-5
  warmup_start_lr: 1e-6
  kl_anneal_start_step: 0
  kl_anneal_end_step: 50000
  kl_final_weight: 0.7
  kl_initial_weight: 0.0
  use_replay_buffer: true
  replay_buffer_size: 10000  # in frames
  
model:
  slot_number: 11  # for MOVI-A/B/C; set to 16 for D/E
  slot_embedding_dim: 128
  transformer_mask_layers: 3
  transformer_mask_heads: 3
  transformer_prior_layers: 2
  transformer_prior_heads: 3
  decoder_layers: 3
  decoder_heads: 3
  feature_extractor: ResNet-like architecture
  attention_unet_depth: 5 residual blocks
  attention_unet_channels: 64 (initial channels, can be scaled)
  slot_latent_dim: 128

optimization:
  optimizer: Adam
  max_gradient_norm: 0.1
  gradient_clipping: true
  clip_norm_value: 0.1
  slot_prior_weight: 1.0
  result_save_path: ./results/
  model_save_path: ./checkpoints/

evaluation:
  metrics:
    fg_ari: true
    mIoU: true
  mask_threshold: 0.3
  evaluation_interval: every 10k steps
  visualization: true

misc:
  seed: 42
  device: cuda
  visualization_frames: 3 (select key frames for visualization)
```

---

**Note:** Full configuration is available in `planning_config.yaml`
