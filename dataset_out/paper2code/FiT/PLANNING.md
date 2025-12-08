# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, step-by-step plan for reproducing the methodology, experiments, and evaluation as described in the paper "FiT: Flexible Vision Transformer for Diffusion Model" — focusing on the technical implementation details, experimental setup, hyperparameters, and evaluation strategies.

---

# 1. Understanding the Core Contributions and Methodology
**Main innovations:**
- A flexible, training-free extrapolation technique based on 2D Rotary Positional Embeddings (RoPE) (VisionNTK/YaRN).
- A novel architecture "FiT" that can generate images at arbitrary resolutions/aspect ratios without additional training or finetuning.
- Replacement of traditional fixed-resolution positional encodings with scalable, resolution-agnostic mechanisms.
- Use of transformer-based diffusion models, combining components like ViT backbones, SwiGLU, and various extrapolation methods.

---

# 2. Model Architecture & Components
**a. Base model structure:**
- A Vision Transformer (ViT)-style encoder-decoder architecture, aligned with DiT-B/XL.
- Use of a patch size of 2 (pixel or latent patch size).
- Hidden size, number of layers, attention heads: same as DiT (e.g., 12 layers, 768 dims, 12 heads for DiT-B/2, 28 layers, 1152 dims for DI XL/2).
- Pretrained VAE (from Stable Diffusion), with downsampling ratio 1/8, latent feature dimension = 4.
- The total token length is 200 (from 20×20 patches).

**b. Tokenization:**
- Encode images into latent codes via pre-trained VAE.
- Patchify the latent code map into a sequence of tokens.
- Average token length in experiments: 200 tokens.

**c. Backbone:**
- Transformer with modifications:
  - Replace MLP with SwiGLU in feed-forward layers.
  - Use a multi-head self-attention with scaled rotary positional embeddings (2D RoPE).
  - Incorporate flexible, resolution-extrapolation-friendly positional encodings (VisionNTK or YaRN).
  
**d. Positional Embedding:**
- Replace absolute positional embeddings with scalable 2D RoPE.
- RoPE: Separately encode height and width position info via decoupled rotary matrices.
- Both NTK-aware and YaRN interpolation methods are employed for resolution extrapolation.

---

# 3. Explicit Methods for Resolution & Aspect-Ratio Scalability
**a. 2D RoPE construction:**
- For each token, encode its 2D position (w, h).
- For each dimension, generate rotary matrices with dimension-specific angles \( \theta_d \).
- Concatenate the 2D RoPE encodings as:
  \[
  f_q(\mathbf{q}_{m}, h_m, w_m) = [e^{i h_m \Theta_h} \mathbf{q}_m \parallel e^{i w_m \Theta_w} \mathbf{q}_m]
  \]

**b. Resolution extrapolation during inference:**
- Use scaled rotary bases. For a desired out-of-distribution resolution \((H_{test}, W_{test})\), compute scale factors \(s_h, s_w\):
  \[
  s_h = \max\left(\frac{H_{test}}{L_{train}}, 1.0\right), \quad s_w = \max\left(\frac{W_{test}}{L_{train}}, 1.0\right)
  \]
- Modify rotary bases accordingly:
  \[
  b'_h = b \cdot s_h^{\frac{|D|}{|D|-2}}, \quad b'_w = b \cdot s_w^{\frac{|D|}{|D|-2}}
  \]
- Replace original rotary frequencies with scaled ones for inference.

**c. Resolution & aspect ratio interpolation methods:**
- NTK-aware: Adjust rotary bases directly with scale factors.
- YaRN: Use γ(r(d)) ramp function to interpolate between trained and extrapolated rotary frequencies per dimension.
- Use separate scale factors and rotary frequencies for height and width.

---

# 4. Diffusion Process & Sampling
- Use a pretrained DDPM or similar diffusion process consistent with the paper (250 steps, guidance scales).
- For sampling, implement the diffusion iterative denoising as per the diffusion model's schedule.
- Guidance used with guidance scale 4.0 during inference.
  
# 5. Training & Data
**a. Dataset:**
- Use ImageNet (e.g., 1.28M training images) at resolution ≤ 256×256.
- For reproducibility, any subset or augmentation should match the experimental setup.

**b. Preprocessing:**
- Resize images so their shortest side matches a fixed size before random cropping.
- During training, use random crop, flip, etc.
- Tokenize images using pre-trained VAE encoder.

**c. Hyperparameters:**
- Training steps: 400K (approx 1.8 million in terms of total updates).
- Batch size: 256.
- Learning rate: 1e-4 (AdamW optimizer; schedule as in the paper).
- Optimizer: AdamW.
- EMA: Use a pre-trained EMA with decay 0.9999.
- Loss function: Standard diffusion loss, combined with classification-free guidance.

---

# 6. Experiments & Evaluation Strategy
**a. In-distribution performance:**
- Train the model on fixed resolution (≤ 256×256).
- Evaluate on images at same resolution, measuring FID, IS, Prec, Rec.

**b. Out-of-distribution & resolution extrapolation:**
- Test the ability to generate images at resolutions up to 1024×1024 and aspect ratios 1:2, 1:3, etc.
- Use resolution extrapolation methods (NTK, YaRN) with the pre-trained model, no extra training.
- Evaluate FID, IS, Prec, Rec at these unseen scales.
- Use guidance scales similar to paper (e.g., 4.0).

**c. Resolution/Aspect Ratio Ablations:**
- Train models with varying token lengths (e.g., 256, 512, 1024 tokens).
- Compare performance with and without FT, NTK, YaRN, visualizing extrapolation strengths.

**d. Qualitative evaluation:**
- Generate samples at various resolutions/aspect ratios.
- Show qualitative sample diversity and artifacts.
- Compare to baselines (e.g., DiT with fixed encodings).

**e. Quantitative evaluation:**
- Compute FID, IS, Prec/Rec metrics.
- Use standard evaluation datasets (e.g., validation splits of ImageNet).

---

# 7. Additional Implementation Details
- Use PyTorch as framework.
- Leverage existing pretrained ViT and diffusion components if possible.
- Implement custom rotary positional encoding modules (decoupled 2D RoPE).
- For flexible inference, implement resolution interpolation modules (NTK, YaRN).
- Use no label conditioning or class guidance unless specified.
- Guidance: implement classifier-free guidance with scale 4.0.
- For extrapolation, modify rotary bases dynamically according to test resolution/aspect ratio, without retraining.

---

### Summary of the Roadmap:
| Step | Description |
|---------|----------------|
| 1. Architectural setup | ViT backbone, patch size=2, attention heads, layers, hidden dims, pretrained VAE. |
| 2. Tokenization | Encode images with VAE; create sequence of latent tokens. |
| 3. Positional encoding | Implement decoupled 2D RoPE with NTK/YaRN interpolation for resolution extrapolation. |
| 4. Diffusion training | Trained on fixed resolution (≤256), using standard diffusion schedule, EMA, guidance. |
| 5. Resolution extrapolation | During inference, Scale rotary bases based on test resolution/aspect ratio. |
| 6. Evaluation | Measure FID, IS, Prec, Rec on in-distribution and extrapolated scales; qualitative samples. |
| 7. Ablation and hyperparameter tuning | Vary token length, guidance, and resolution to analyze performance. |

---

# Final notes:
**Uncertainties or missing details:**
- Exact scheduler (learning rate warmup, decay).
- The number of attention heads and layers for base models.
- Specific optimizer parameters or training schedule.
- Exact guidance scale during each sampling scenario.
- Implementation of guidance "freezing" or classifier-free guidance steps.
- Dataset preprocessing details (e.g., augmentation specifics).

*These can be aligned with the DiT training code or set based on defaults from similar diffusion transformer models.*

---

This roadmap provides a detailed plan enabling systematic implementation and thorough experimentation later. Would you like me to proceed with a step-by-step code outline based on this?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular Python system using PyTorch and supporting libraries (like torchvision, transformers, and diffusers) to reproduce the paper's method. The core components include a ViT backbone with flexible 2D rotary positional embeddings for resolution extrapolation, integrated with a pretrained VAE and a diffusion sampling pipeline. The system will support arbitrary resolution inference via dynamic rotary base scaling and positional interpolation (NTK/YaRN). Training involves standard diffusion loss on ImageNet, using EMA and guidance scale, followed by inference at extrapolated resolutions, generating and evaluating samples.",
    
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "diffusion_pipeline.py",
        "positional_encoding.py",
        "trainer.py",
        "extrapolation.py",
        "evaluation.py",
        "utils.py"
    ],
    
    "Data structures and interfaces": "```mermaid\nclassDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str, resolution: Tuple[int, int])\n        +load_data() -> Dataset\n    }\n    class FiTModel {\n        +__init__(model_params: dict)\n        +forward(tokens: Tensor, positions: Positions) -> Tensor\n        +set_resolution_scale(h_scale: float, w_scale: float) -> None\n        +inject_rotary_bases(scale_h: float, scale_w: float) -> None\n    }\n    class DiffusionSampler {\n        +__init__(model: FiTModel, diffusion_steps: int, guidance_scale: float)\n        +sample(prompt: str, resolution: Tuple[int, int], aspect_ratio: float, guidance_scale: float) -> Image\n    }\n    class PositionalEncoding {\n        +__init__(dim: int, max_resolution: Tuple[int, int], mode: str)\n        +get_rotary_bases(scale_h: float, scale_w: float) -> Tuple[Tensor, Tensor]\n        +encode_position(w: int, h: int, scale_h: float, scale_w: float) -> PositionTensor\n    }\n    class Evaluation {\n        +compute_fid(generated_images: List[Image], reference_images: List[Image]) -> float\n        +compute_is(generated_images: List[Image]) -> float\n    }\n    Main --> DatasetLoader\n    Main --> FiTModel\n    Main --> DiffusionSampler\n    FiTModel --> PositionalEncoding\n    DiffusionSampler --> FiTModel\n    Main --> Evaluation\n```",
    
    "Program call flow": "```mermaid\nsequenceDiagram\n    participant C as main.py\n    participant DL as dataset_loader.py\n    participant M as model.py\n    participant P as positional_encoding.py\n    participant D as diffusion_pipeline.py\n    participant E as evaluation.py\n    C->>DL: load_data(dataset_path, resolution)\n    DL-->>C: dataset\n    C->>M: initialize_model(model_params)\n    M->>P: initialize(max_resolution, mode='NTK' or 'YaRN')\n    M-->>C: model instance\n    C->>D: set diffusion parameters (steps=250, guidance=4.0)\n    loop for resolution extrapolation\n       C->>M: set_resolution_scale(h_scale, w_scale)\n       C->>D: generate image(prompt, resolution, aspect_ratio, guidance_scale)\n       D-->>C: generated Image\n       C->>E: evaluate/fid_score(generated_image, reference_set)\n    end\n    C->>Main: end\n```",
    
    "Anything UNCLEAR": "Clarify the exact format for input prompts and reference images, the specific diffusion sampler configuration (scheduler, guidance schedule), and if pretrained models (ViT backbone, VAE, diffusion) are available or need reproduction. Also, confirm the hyperparameters like total diffusion steps, guidance scale, and maximum training duration for comprehensive fidelity."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "torchvision==0.14.1",
        "numpy==1.21.0",
        "matplotlib==3.5.1",
        "tqdm==4.64.0",
        "dlib",
        "scipy",
        "Pillow==9.2.0",
        "transformers==4.29.2",
        "diffusers==0.14.0",
        "accelerate==0.20.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: loads ImageNet dataset (or subset if needed), applies resizing and cropping pre-processing, outputs datasets for training/evaluation. Depends on torchvision. Provides dataset objects for training and evaluation scripts."
        ],
        [
            "positional_encoding.py",
            "Implements classes for decoupled 2D RoPE, NTK/YaRN interpolation, rotary bases scaling functions. Used by model.py during model initialization and inference for dynamic positional embeddings. Should be independent but required by model.py."
        ],
        [
            "model.py",
            "Defines FiTTransformer backbone class: constructs ViT with attention, SwiGLU ffn, incorporates flexible positional encoding (via positional_encoding.py). Has methods for setting resolution scale factors and rotary bases scaling for inference-time extrapolation."
        ],
        [
            "diffusion_pipeline.py",
            "Implements the diffusion process: defines a DiffusionSampler class, loads pretrained diffusion model, handles sampling procedure with guidance, supports adjusting guidance scale, and generates images at arbitrary resolutions with provided prompt. Depends on model.py, transformers, diffusers."
        ],
        [
            "trainer.py",
            "Handles training loop: instantiates dataset, model, optimizer, and EMA, manages training steps, loss calculation, checkpoint saving. Depends on dataset_loader.py and model.py."
        ],
        [
            "evaluation.py",
            "Provides functions for computing FID, IS, and Precision/Recall metrics. Requires evaluation dataset and generated images (from diffusion_pipeline.py). Used after inference."
        ],
        [
            "main.py",
            "Entry point: parses args or configs, initializes dataset loader, model, diffusion sampler, runs training for set steps, performs resolution extrapolation tests via diffusion pipeline, and evaluates results. Manages overall flow, dependencies among modules, and logging."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "positional_encoding.py",
        "model.py",
        "diffusion_pipeline.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n\n// No formal REST APIs required; focus on internal module APIs and functions.\n// Methods include dataset loading, positional encoding calculation, model forward, setting resolution scale, and sampling images.\n// For example:\n// - DatasetLoader.load_data() -> Dataset\n// - FiTTransformer.forward(tokens, positions) -> Tensor\n// - set_resolution_scale(h_scale, w_scale)\n// - DiffusionSampler.sample(prompt, resolution, aspect_ratio, guidance_scale) -> Image\n// - compute_fid(generated_images, reference_images) -> float\n// - evaluate() in main.py to trigger evaluation pipeline.",
    "Shared Knowledge": "Shared variables include maximum training token length (e.g., 256 tokens), guidance scale (4.0), choice of positional interpolation method (NTK or YaRN), and pretrained models: ViT backbone, pretrained VAE, diffusion model. Utility functions for scaling rotary bases based on resolution ratio, and for applying positional interpolation techniques in inference are common to multiple modules.",
    "Anything UNCLEAR": "Clarify exact pretrained diffusion model used (e.g., which checkpoint), whether specific hyperparameters such as training steps, learning rate schedule, or sampler schedule (e.g., DDIM, DDPM) are fixed; also, details on precise evaluation dataset splits and metrics calculation methodology are needed."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  total_steps: 400000
  batch_size: 256
  learning_rate: 1e-4
  guidance_scale: 4.0
  ema_decay: 0.9999

dataset:
  name: ImageNet
  resolution: 256
  subset_size: full

model:
  architecture: FiT
  patch_size: 2
  hidden_dims: 768  # for FiT-B; for FiT-XL, set accordingly
  layers: 12
  attention_heads: 12
  ffn_type: SwiGLU
  pretrained_vae_path: "path/to/pretrained/vae"
  pretrained_diffusion_path: "path/to/pretrained/diffusion"

optimization:
  optimizer: AdamW
  weight_decay: 0.01
  scheduler: linear_warmup_and_decay
  warmup_steps: 10000
  learning_rate: 1e-4

extrapolation:
  method: NTK  # or 'YaRN'
  max_resolution: [1024, 1024]
  resolution_inference: [512, 512]
  aspect_ratio_test: [1:2, 1:3]

generation:
  inference_steps: 250
  guidance_scale: 4.0

evaluation:
  fid_samples: 10,000
  is_samples: 2,000
  reference_dataset: ImageNet validation split

logging:
  output_dir: "./outputs"
  save_checkpoints: true
  save_interval: 10000
  log_interval: 500
```

---

**Note:** Full configuration is available in `planning_config.yaml`
