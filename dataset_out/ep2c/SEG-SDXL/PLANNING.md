# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. To produce a reproducible implementation of the **Smoothed Energy Guidance (SEG)** method described in the paper, the plan must precisely capture the methodology, experimental setup, and practical considerations. Here's a structured and detailed roadmap:

---

## 1. Understand and Extract Core Methodology of SEG

### A. Theoretical Foundations
- **Energy-based view of attention**:
  - Attention weights \( \mathbf{A} = \mathbf{Q} \mathbf{K}^\top \)
  - Defined an energy function \( E(\mathbf{A}) \) linked to the Hopfield energy.
- **Key insight**:
  - Applying Gaussian blur \( G \) to attention weights reduces the underlying energy curvature, leading to a smoother energy landscape.
  - Blurring attention weights modulates the second derivatives (the Hessian), attenuating sharpness and fostering stable guidance.

### B. Practical Implementation of Gaussian Blur on Attention
- **Gaussian kernel**:
  - Standard deviation \( \sigma \) controls smoothing intensity.
  - Keep guidance scale \( \gamma_{seg} \) fixed, control curvature via \( \sigma \).
- **Blurring procedure**:
  - Instead of directly blurring attention matrices (which is \( O(n^2) \)), use a linear operation:
    - **Blur queries** \( \mathbf{Q} \) with Gaussian \( G \): \( \mathbf{Q}_{blur} = G * \mathbf{Q} \)
    - Compute attention as usual with \( \mathbf{Q}_{blur} \):
      \[ \mathbf{A}_{seg} = \operatorname{softmax}( \mathbf{Q}_{blur} \mathbf{K}^\top / \sqrt{d} ) \]
- **Efficiency note**:
  - Blurring queries (or keys) is equivalent to convolving the matrices with a Gaussian kernel, which avoids quadratic complexity in tokens.

### C. Modified Denoising/Generation Equation
- Supplant the original attention-based prediction with the *smoothed* version:
  \[
  \mathbf{s}_{\theta, seg}(x, t) = \text{Attention}_G(\mathbf{Q}, \mathbf{K}, V)
  \]
  - where attention is computed using Gaussian-blurred queries or keys.
- Incorporate guidance:
  \[
  d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 (\gamma_{seg}\mathbf{s}_\theta - (\gamma_{seg} - 1) \tilde{\mathbf{s}}_\theta)] dt + g(t) d\bar{\mathbf{w}}
  \]
  - \( \tilde{\mathbf{s}}_\theta \) is the unconditional prediction with blurred attention.

---

## 2. Implementation Details for Reproduction

### A. Data Preparation
- **Datasets**: 
  - Use large-scale, open datasets compatible with training diffusion models:
    - *Unconditional*: FFHQ, LSUN, or LAION-400M (depending on resource).
    - *Conditional*: Large-scale text-image datasets such as LAION-2B or as per Pretrained SDXL training.
- **Preprocessing**:
  - Resize images (e.g., 512×512).
  - Normalize pixel values to \([-1, 1]\) or \([0,1]\), as per the backbone model.

### B. Model Architecture
- **Base Diffusion Model**:
  - Use an architecture similar to SDXL or similar large-scale diffusion models:
    - U-Net backbone with self-attention modules.
  - **Attention modules**:
    - Extract queries \( \mathbf{Q} \) and keys \( \mathbf{K} \) at each attention layer.
- **Attention Operation Modifications**:
  - Implement blurring on queries or keys:
    - Use a Gaussian kernel convolution (preferably separable for efficiency).
    - This can be implemented with a depthwise convolution or fast Gaussian filtering routines.

### C. Hyperparameters
- **Gaussian blur (\( G \))**:
  - Standard deviation \( \sigma \); tune over \(\{1, 2, 5, 10, \infty\}\).
  - Kernel size = \( 2 \times \lceil 3\sigma \rceil + 1 \).
- **Guidance parameters**:
  - Guidance scale \( \gamma_{seg} \): fixed at 3.0 (per experiments).
  - Guidance strength \( \gamma_{cfg} \): tune over \(\{1,3,5,7,9\}\) during experiments.
- **Training hyperparameters**:
  - Batch size: depend on GPU memory—ideally 16+.
  - Learning rate: start from 1e-5 to 3e-5.
  - Optimizer: AdamW with appropriate weight decay.
  - Number of training steps: 100k+ for stable results.

### D. Model Training & Fine-tuning
- **Pretrained Backbone**:
  - Leverage existing pretrained SDXL or similar models.
- **Fine-tuning strategy**:
  - Freeze backbone except for attention modules if possible; or fine-tune whole model.
  - Incorporate Gaussian blur module in attention layers:
    - During training, expose the model to the *blurred* attention weights.
    - Alternately train with and without Gaussian blur to stabilize.
- **Guided sampling**:
  - For unconditional: generate without conditioning.
  - For conditional: include text prompts, optionally fine-tune text encoders alongside.

### E. Sampling & Generation
- **Sampling procedure**:
  - Use DDIM or other ODE-based solvers.
  - During each step:
    - Compute attention with blurred queries/keys.
    - Calculate unconditional and conditional predictions.
    - Apply guidance using fixed guidance scale.
    - Specially control \( \sigma \): interpolate guidance between original and smoothed versions.
- **Adaptive control of \( \sigma \)**:
  - Experiment with fixed \( \sigma \) and schedule increasing \( \sigma \) during sampling, inspired by experiments.

---

## 3. Experimental Setup & Evaluation Metrics

### A. Ablation Studies
- **Vary \( \sigma \)**: \(\{0, 1, 2, 5, 10, \infty\}\).
- **Guidance scale (\( \gamma_{cfg} \))**: \(\{1,3,5,7,9\}\).
- **Check for**:
  - Fidelity (FID scores).
  - Sample quality (CLIP score, LPIPS).
  - Side effects (structure distortion, saturation).

### B. Reproduction of Results
- **Unconditional**:
  - Generate images with various \( \sigma \) and guidance.
  - Compare image quality against vanilla diffusion.
- **Conditional**:
  - Use text prompts; evaluate realism, adherence, and diversity.
  - Use ControlNet or similar conditioning architectures with segmentation maps, sketches, etc.
- **Visual Evaluation**:
  - Reproduce figures similar to Figures 1-20.
  - Qualitative assessment of detail preservation, smoothness, and structure.

### C. Additional Metrics
- Quantitative:
  - FID using official implementation.
  - CLIP similarity.
  - LPIPS for diversity.
- Societal impact:
  - Check for biases and safety, report adherence to guidelines.

---

## 4. Implementation Notes & Clarifications
- Use a flexible codebase (e.g., adapted from Stable Diffusion or SDXL open source) with modules:
  - Attention layer with Gaussian blur (separable 2D convolution).
  - Guidance module to implement the modified guidance equation.
- Incorporate a evaluation/effective hyperparameter search over \( \sigma \) and guidance scales.
- Ensure reproducibility:
  - Seed all randomness.
  - Log all hyperparameters.
  - Save intermediate models if possible.

---

## Summary of Critical Next Steps:
- Implement Gaussian-blurred attention *inside* the core attention modules.
- Fix guidance scale; control smoothness via \( \sigma \).
- Validate with unconditional and conditional generation.
- Compare quantitative metrics and qualitative results.

This roadmap should serve as a comprehensive guide for code development, training, and experimental validation of the SEG method as described in the paper.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will extend a pretrained diffusion model backbone (e.g., Stable Diffusion / SDXL) by incorporating an attention module with Gaussian blur functionality. The core idea is to blur the query or key tensors in each self-attention layer using a fast separable Gaussian filter, then perform attention as usual, modifying the prediction according to guidance equations. We'll implement the entire process in PyTorch, leverage existing diffusion codebases, and add modules to apply Gaussian blur in attention computations dynamically during sampling. The sampling procedure will be adapted to include the smoothed attention weights, compute the 'smoothed' unconditional and conditional predictions, and guide the reverse denoising accordingly. The system will support varying \(\sigma\) and guidance scale, with easy parameter input for experiments. We’ll ensure efficiency by convolving only the queries or keys, avoiding quadratic complexity over tokens.",
    "File list": [
        "main.py",  
        "dataset_loader.py", 
        "model.py",  
        "attention.py", 
        "diffusion_sampler.py", 
        "utils.py", 
        "evaluation.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DiffusionModel {\n        +__init__(architecture: str, pretrained_path: str)\n        +load_model() -> None\n        +forward(x: Tensor, cond: Optional[Tensor], guidance_scale: float, sigma: float) -> Tensor\n        +get_attention_layers() -> List[AttentionLayer]\n    }\n    class AttentionLayer {\n        +__init__()\n        +compute_attention(q: Tensor, k: Tensor, v: Tensor, sigma: float) -> Tensor\n        +apply_gaussian_blur(tensor: Tensor, sigma: float) -> Tensor\n    }\n    class DiffusionSampler {\n        +__init__(model: DiffusionModel, guidance_scale: float, sigma_schedule: List[float])\n        +sample(cond: Optional[Tensor]) -> Tensor\n        +run_reverse_process(initial_noise: Tensor, cond: Optional[Tensor]) -> Tensor\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str, image_size: Tuple[int, int])\n        +load_unconditional_dataset() -> Dataset\n        +load_conditional_dataset() -> Dataset\n    }\n    class Evaluation {\n        +calculate_fid(samples: List[Tensor], real_data_path: str) -> float\n        +calculate_clip_score(images: List[Tensor], prompts: List[str]) -> float\n    }\n\nMain --> DatasetLoader\nMain --> DiffusionModel\nMain --> DiffusionSampler\nDiffusionModel --> AttentionLayer\nDiffusionSampler --> DiffusionModel\nAttentionLayer --|> utils.apply_gaussian_filter\n\n// places where Gaussian filtering is integrated into attention call",
    "Program call flow": "sequenceDiagram\n    participant C as Main\n    participant D as DatasetLoader\n    participant M as DiffusionModel\n    participant S as DiffusionSampler\n    participant E as Evaluation\n    C->>D: load datasets\n    D-->>C: dataset objects\n    C->>M: initialize with architecture and checkpoint\n    M-->>C: model ready\n    C->>S: initialize sampler with guidance parameters\n    S-->>C: sampler ready\n    C->>S: call sample with conditioning (unconditional/conditional)\n    S->>M: run reverse process at each step, compute attention with Gaussian blur (vary sigma)\n    M-->>S: predict noise/direction\n    S-->>C: generated images\n    C->>E: evaluate images\n    E-->>C: metrics\n\n// Loop for varying \sigma and guidance\n    loop over sigma in sigma_schedule\n        loop over guidance in guidance_list\n            C->>S: generate images\ndirectly with current parameters\n    end\n  end\nend",
    "Anything UNCLEAR": "Clarification needed on the exact interface of the pretrained diffusion core, whether to modify the codebase directly or wrap it, and on specific dataset paths and formats used in training. Also, confirm if a specific diffusion codebase (e.g., public SDXL implementation) is preferred or if a custom minimal implementation is needed."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch>=1.11",
        "torchvision>=0.12",
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib",
        "tqdm",
        "transformers",
        "pillow"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Define DatasetLoader class: responsible for loading and preprocessing image datasets for training and evaluation. It should support datasets like LAION, CIFAR, FFHQ, with options for normalized tensor output and optional conditioned prompts. Dependency: torchvision, PIL, dataset definitions."
        ],
        [
            "model.py",
            "Define DiffusionModel class: loads a pretrained diffusion backbone (SDXL or similar), with the ability to extract attention layers. Implement a custom attention module with gaussian blur functionality integrated, supporting a parameter for \u03c3 (sigma). It should implement forward() that takes input tensor, guidance scale, and sigma, returning predicted noise or label conditioned outputs."
        ],
        [
            "attention.py",
            "Implement AttentionLayer class: encapsulates self-attention with Gaussian Blur application on queries/keys dynamically via apply_gaussian_blur() method. The method uses scipy or custom separable convolution for efficiency to blur the attention tensors. Must support different sigma values supplied at inference."
        ],
        [
            "trainer.py",
            "Define Trainer class: manages training or fine-tuning. For this task, mainly responsible for constructing the training loop with optional attention kernel modifications. Implement functions to handle dataset batching, optimizer steps, and save/load model checkpoints."
        ],
        [
            "diffusion_sampler.py",
            "Define DiffusionSampler class: handles the reverse process sampling. Implements sample() and run_reverse_process() functions, which perform iterative denoising. During each step, it calls model.forward() with guidance, passing sigma and guidance scale as parameters, including the attention module with Gaussian blur applied as per current \u03c3. Supports varying \u03c3 over sampling steps."
        ],
        [
            "evaluation.py",
            "Define Evaluation class: computes metrics such as FID, CLIP score, LPIPS. Uses precomputed real dataset statistics. Consumes generated images, runs inference, and outputs metrics for comparison. Additionally, may include utilities for side-effect analysis."
        ],
        [
            "main.py",
            "Entry point: initializes dataset loader, model, and sampler. Reads configuration (hyperparameters, dataset paths, guidance parameters). Executes training or loading pretrained model. Supports user inputs for guidance scale, \u03c3 schedule, number of steps, evaluation triggers. Calls sampler for image generation and evaluation modules for metrics. Coordinates experiment variation parameters."
        ]
    ],
    "Task list": [
        "dataset_loader.py", 
        "attention.py", 
        "model.py", 
        "trainer.py", 
        "diffusion_sampler.py", 
        "evaluation.py", 
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "The attention module's Gaussian blur functionality is shared between model.py and diffusion_sampler.py for inference. The dataset loader provides dataset objects with normalization and prompts. Evaluation functions require image tensors and prompt text for metrics like CLIP and FID. Guidance parameters (\u03b3_{seg}, \u03c3) are configurable at runtime and influence sampling instructions.",
    "Anything UNCLEAR": "Clarification needed on the pretrained diffusion backbone architecture specifics, dataset formats (e.g., image resolution, conditioning inputs), and exact hyperparameter ranges used in experiments for comprehensive replication."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0003
  batch_size: 8
  epochs: 100000
  fp16: true
  gradient_accumulation_steps: 1

dataset:
  dataset_path: "/path/to/dataset"
  image_size: [512, 512]
  dataset_type: "unconditional"  # or 'conditional'
  dataset_name: "laion"  # Example, replace as needed

model:
  architecture: "SDXL"
  pretrained_checkpoint: "/path/to/pretrained/model.ckpt"
  freeze_backbone: false  # Fine-tune or freeze
  attention_blur: true
  sigma_values: [0, 1, 2, 5, 10, 20, 50, 100]  # \u03c3 values to experiment, 0 means no blurring

guidance:
  guidance_scale: 3.0  # Fixed guidance scale during training
  guidance_variants:
    - "none"
    - "segmented_attention"  # Using Gaussian-blurred queries or keys
  sigma_schedule: [0, 1, 2, 5, 10, 20, 50, 100]  # Sigma changing steps for sampling

sampling:
  steps: 1000
  sampler_type: "ddim"   # or "ddpm", replace as needed
  guidance: "segmented_attention"
  sigma: 10  # default sigma for inference unless experimenting
```

---

**Note:** Full configuration is available in `planning_config.yaml`
