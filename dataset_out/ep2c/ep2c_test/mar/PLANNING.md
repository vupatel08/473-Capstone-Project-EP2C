# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Based on the detailed content of the paper, here is a comprehensive, step-by-step plan for reproducing their experiments and methodologies. This outline covers core concepts, model architectures, training procedures, datasets, hyperparameters, evaluation, and practical considerations necessary to faithfully implement their approach.

---

## 1. Core Methodology Overview

### Key Innovative Idea:
- **Elimination of discrete tokenizers**: The model predicts continuous-valued tokens for images, removing the need for vector quantization.
- **Diffusion modeling for tokens**: Per-token probability distribution is modeled via a diffusion process, specifically a noise schedule applied to tokens, with a learned denoising network (`ε_θ`) predicting the added noise.
- **Autoregressive modeling in continuous space**:
  - Generate tokens sequentially, with a context of previous tokens.
  - Use a Transformer to produce conditioning vectors `z^i` for each token, representing the prior sequence.
  - Model `p(x^i | context)` via the diffusion process conditioned on `z^i`.
- **Diffusion Loss**:
  - Predicted token `x` is obtained through a reverse diffusion process using the network’s noise estimate.
  - Loss is computed as an expected MSE between true noise and estimated noise, conditioned on `z`.
  
### Sampling (Inference):
- Run reverse diffusion conditioned on autoregressively predicted `z^i`.
- Sample tokens with optional temperature scaling.
- Generate multiple tokens simultaneously via masked autoregressive formulation.
  
---

## 2. Architecture Details

### 2.1. Tokenizer & Tokens:
- Use pre-existing tokenizers to get initial tokens:
  - **VQ-GAN (e.g., VQ-16 or KL-16)**: Discrete tokens (e.g., 16×16 codebook, each token in `[0, K)` with `K=256`).
  - **Continuous tokens**:
    - Use the continuous latent representations directly (e.g., KL-regularized or no tokenizer).
    - Model tokens as vectors `x ∈ ℝ^D`.
    
### 2.2. Transformer Model:
- **Input**: Sequence of previous tokens (or masked tokens).
- **Architecture**:
  - ~32 transformer blocks.
  - Width: 1024 channels.
  - Positional embeddings for token positions.
  - Causal masking for autoregressive predictions, or bidirectional attention if modeling masked prediction.
- **Outputs**:
  - Conditioning vector `z^i` at each position.
  - Hidden states used to produce the diffusion conditioning vector.

### 2.3. Diffusion Denoising Network (`ε_θ`):
- Small MLP (~3 residual blocks, 1024 width, layer normalization, SiLU activations).
- Conditioned on:
  - Noisy token `x_t`.
  - Time step `t` (via learned embeddings).
  - Condition vector `z^i`.
- Outputs: Predicted noise vector `ε_θ(x_t, t, z)`.

---

## 3. Training Procedures

### 3.1. Data:
- Dataset: **ImageNet 256×256** images.
- Use preexisting tokenizers (VQ-GAN/KL-16) to encode images into token sequences.
- For continuous tokens: Use latent representations directly (e.g., KL-regularized or VQ embeddings).

### 3.2. Autoregressive Model Training:
- **Objective**:
  - Predict next token `x^i` conditioned on previous tokens.
  - Produce conditioning vectors `z^i` for each token sequence.
- **Input**:
  - Sequence of tokens (with causal masking).
- **Loss**:
  - `Diffusion Loss`: 
    \[
    \mathcal{L}(z^i, x^i) = \mathbb{E}_{t, \varepsilon}\left[ \|\varepsilon - \varepsilon_θ(x_t, t, z^i)\|^2 \right]
    \]
  - Sample `t` uniformly during training.
  - Corrupt `x^i` with Gaussian noise at step `t` during training.
  
### 3.3. Diffusion Process Details:
- Noise schedule: Cosine schedule with 1000 steps, but training can involve fewer (e.g., 100 steps for efficiency).
- `x_t`: noise-corrupted token vector at step `t`:
  \[
  x_t = \sqrt{\bar{\alpha}_t} x + \sqrt{1 - \bar{\alpha}_t} \varepsilon
  \]
- `ε_θ` estimates the added noise given `x_t`, `t`, and `z`.

### 3.4. Optimizer & Hyperparameters:
- Optimizer: AdamW.
- Learning rate: Approximately 8×10^-4 with cosine warm-up.
- Batch size: ~2048 tokens per step (distributed over multiple GPUs).
- Number of epochs: ~400 epochs.
- Model size: Transformer (~40M params for AR backbone).
- Diffusion MLP (denoiser): 3 residual blocks, 1024 channels.
- Loss weighting: Include optional variational lower bound if needed.

### 3.5. Additional Techniques:
- Teacher-forcing via conditioning on previous true tokens during training.
- Use a `μ`-scaling for temperature during sampling (scale `ε_θ` to control diversity).

---

## 4. Inference & Sampling Strategy

### 4.1. Autoregressive Sampling:
- Generate tokens sequentially from left to right.
- Conditioned on previous tokens sum with learned `z^i`.
- For each position:
  - Compute conditioning vector `z^i`.
  - Run reverse diffusion conditioned on `z^i`.
- Decoding:
  - For discrete tokens, sample via Gumbel-max, inverse transform sampling, or temperature-adjusted sampling.
  - For continuous tokens, run the diffusion process directly.

### 4.2. Masked Autoregressive:
- Can predict multiple tokens simultaneously using bidirectional attention with masking.
- Reduce number of steps adaptively (cosine schedule from 1.0 to 0).
- Simultaneous prediction with progressively decreasing mask ratio.

### 4.3. Diffusion Sampling:
- Reverse diffusion starting from Gaussian noise.
- Use a smaller number of steps (e.g., 100 steps) for efficiency, with results comparable to more steps.
- Scale the noise (`τ`) as a hyperparameter to tune diversity and quality.

---

## 5. Experiment Setup & Hyperparameters

| Aspect                     | Details                                                                                                                                                          |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dataset                    | ImageNet 256×256 images encoded into tokens via VQ-GAN (VQ-16) or KL-16.                                                                                      |
| Batch size                 | ~2048 tokens per step across multiple GPUs.                                                                                                                    |
| Models                     | - Transformer autoregressive (AR) backbone (~40M params).<br>- Diffusion denoising network: small MLP (~3 residual blocks).<br>- Variants: Masked AR, bidirectional. |
| Training epochs            | ~400 epochs.                                                                                                                                                     |
| Learning rate              | About 8×10^-4 with cosine warm-up.                                                                                                                               |
| Noise schedule             | Cosine schedule with 1000 steps; for inference, use fewer steps (100 default).                                                                              |
| Diffusion steps during train | 100 steps, sampling multiple `t`s per batch for efficiency.                                                                                                |
| Sampling steps during inference | 100-256 steps, tuned for quality-speed trade-offs.                                                                                                      |
| Temperature scaling        | Apply temperature `τ` by scaling noise level or sampling probabilities.                                                                                      |
| Evaluation metrics          | FID, Inception Score, Precision/Recall, with attention to reproducibility and variability.                                                                   |

---

## 6. Implementation Considerations

### Code & Data:
- Use preexisting tokenizers for image encoding.
- Implement Transformer with causal or bidirectional attention as needed.
- Implement small MLP denoiser, conditioned on `z`, `t`, and noisy `x_t`.
- Utilize diffusion schedule (cosine) with flexible steps.
- Ensure accurate sampling routines (reverse diffusion) with temperature scaling.
- Save model checkpoints periodically; record hyperparameters and training logs for reproducibility.

### Additional:
- Careful numerical stability in noise schedule and reverse diffusion.
- Accurate conditioning extraction from autoregressive Transformer.
- Efficient batching to handle large datasets with multiple GPUs.
- Use evaluation protocols consistent with original work for fair comparison.

---

## 7. Summary & Next Steps

- **Reproduce data encoding:** Use existing image tokenizers or latent representations.
- **Build Transformer & denoiser:** Follow architecture details.
- **Train autoregressive model with diffusion loss:** As described.
- **Implement reverse diffusion sampling:** Using the trained denoising network.
- **Experiment with temperature scaling and step count:** To find optimal quality-speed trade-offs.
- **Evaluate with metrics:** FID, IS, Precision/Recall on ImageNet 256×256.

This outline provides a thorough roadmap to develop a full implementation, allowing later detailed coding and hyperparameter tuning aligned with the paper's methodology and experimental results. If any specific component requires clarification or additional technical detail, I can expand further.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular system in Python leveraging open-source libraries such as PyTorch, Hugging Face Transformers, and existing tokenizers. The architecture consists of data loading, a transformer-based autoregressive tokenizer, a diffusion denoising network, training routines, and sampling modules. The system trains an autoregressive transformer to produce conditioning vectors for each token sequence, which are then used in a diffusion-based denoising model to predict and generate tokens sequentially or in parallel. The code will be organized into distinct modules: data processing, model definitions, training loop, and evaluation, with a simple command-line interface for running experiments and sampling.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "sampling.py",
        "evaluation.py",
        "utils.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str)\n        +load_data() -> Dataset\n    }\n    class Tokenizer {\n        +__init__(model_name: str)\n        +encode(image: Tensor) -> List[int]\n        +decode(tokens: List[int]) -> Tensor\n    }\n    class TransformerAutoregressive {\n        +__init__(params: dict)\n        +generate_conditioning(seq: List[int]) -> Tensor\n        +forward(seq: Tensor) -> Tensor\n    }\n    class DiffusionDenoiser {\n        +__init__(params: dict)\n        +predict_noise(x_t: Tensor, t: int, z: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Union[TransformerAutoregressive, DiffusionDenoiser], data: Dataset, config: dict)\n        +train() -> None\n    }\n    class Sampler {\n        +__init__(denoiser: DiffusionDenoiser, model: TransformerAutoregressive, config: dict)\n        +sample(sequence_length: int, y_cond: Optional[Tensor]=None, temperature: float=1.0) -> List[int]\n    }\n    class Evaluation {\n        +__init__(model: Union[TransformerAutoregressive, DiffusionDenoiser], data: Dataset)\n        +compute_metrics() -> dict\n    }\n\n    Main --> DatasetLoader\n    Main --> Training\n    Main --> Sampling\n    Main --> Evaluation\n    Training --> TransformerAutoregressive\n    Training --> DiffusionDenoiser\n    Sampling --> DiffusionDenoiser\n    Sampling --> TransformerAutoregressive\n    Evaluation --> TransformerAutoregressive\n    Evaluation --> DiffusionDenoiser",
    "Program call flow": "sequenceDiagram\n    participant C as Main\n    participant DL as DatasetLoader\n    participant TA as TransformerAutoregressive\n    participant DD as DiffusionDenoiser\n    participant TR as Trainer\n    participant SA as Sampler\n    participant EV as Evaluation\n    C->>DL: Load dataset from path\n    DL-->>C: Return dataset object\n    C->>TA: Initialize with hyperparameters\n    C->>DD: Initialize diffusion denoiser\n    C->>TR: Create trainer with models and dataset\n    TR->>TR: train()\n    TR-->>C: training complete\n    C->>SA: Start sampling, conditioned on prompt or previous tokens\n    SA->>TA: generate conditioning vectors\n    SA->>DD: run reverse diffusion process\n    SA-->>C: return generated token sequence\n    C->>EV: Load dataset for evaluation\n    EV->>TA: evaluate()\n    EV->>DD: evaluate()\n    EV-->>C: report metrics\n    Note over C: Repeat sampling/evaluation as needed\n    End"
    ,
    "Anything UNCLEAR": "Clarify the expected input/output formats for images and tokens, specify whether continuous or discrete tokens are preferred for implementation, and detail the exact evaluation protocol (specific FID implementation, seed settings, sampling steps)."

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "torchvision==0.12.0",
        "numpy==1.21.0",
        "PyYAML==6.0",
        "tqdm==4.62.3",
        "faiss-cpu==1.7.2",
        "transformers==4.25.1",
        "scipy==1.7.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Handles loading ImageNet tokens, implements Dataset class with methods load_data() and optional dataset splitting. Depends on the tokenizer outputs and raw images if used. Loads data, applies padding and normalization as needed."
        ],
        [
            "model.py",
            "Defines all neural network modules: Transformer autoregressive backbone, denoising diffusion model (small MLP), conditioning vectors, and the full pipeline. Key classes include: TransformerAutoRegressive, DiffusionDenoiser, conditioning generator. Imports from torch.nn and transformers."
        ],
        [
            "trainer.py",
            "Implements training routines for both the autoregressive transformer and the diffusion denoising network. Depends on models in model.py and dataset from dataset_loader.py. Manages the optimization of both components, applies diffusion loss, includes scheduling, logging, checkpointing."
        ],
        [
            "sampling.py",
            "Provides functions for reverse diffusion sampling conditioned on autoregressive outputs or partial sequences. Uses trained models from model.py and trainer.py. Implements step-by-step reverse process, handles temperature scaling, and parallel token sampling."
        ],
        [
            "evaluation.py",
            "Calculates evaluation metrics like FID, Inception Score, and Precision/Recall. Loads trained models, runs sampling on a held-out dataset, and computes metrics. Depends on sampling.py and dataset_loader.py."
        ],
        [
            "main.py",
            "Entry point script. Parses command-line arguments or config.yaml, initializes dataset_loader, models, trainer, and evaluation modules. Coordinates training, sampling, and evaluation workflows sequentially. Manages dependencies to ensure models are loaded before training/evaluation."
        ],
        [
            "utils.py",
            "Contains utility functions: diffusion schedule, noise corruption, inverse sampling, normalization, logging, checkpoint saving/loading. Shared across modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "sampling.py",
        "evaluation.py",
        "main.py",
        "utils.py"
    ],
    "Full API spec": "",
    "Shared Knowledge": "The code shares configuration parameters such as dataset paths, model hyperparameters, diffusion schedule, optimizer settings, and evaluation metric parameters. All modules use consistent configuration management via YAML or JSON files.",
    "Anything UNCLEAR": "Clarify the exact structure and format of the tokenized dataset (discrete/vectors), preferred seed and evaluation protocols for FID, and whether to use discrete tokens, continuous latent vectors, or both during training and inference."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 8e-4
  batch_size: 2048
  epochs: 400
  warmup_epochs: 10
  optimizer: AdamW
  optimizer_params:
    weight_decay: 0.02
    betas: [0.9, 0.999]
  gradient_clip_norm: 1.0

model:
  transformer:
    num_layers: 32
    hidden_dim: 1024
    num_heads: 16
    dropout_rate: 0.1
    max_sequence_length: 1024
  diffusion_denoiser:
    residual_blocks: 3
    residual_width: 1024

diffusion:
  schedule_type: cosine
  total_steps: 1000
  inference_steps: 100
  noise_schedule_params:
    s: 0.008
  temperature: 1.0
  temperature_scale_method: scale_noise

dataset:
  name: ImageNet256
  path: /path/to/imagenet256/encoded_tokens
  tokenizer_type: vq-gan
  tokenizer_name: vq-16
  sequence_length: 1024
  shuffle_buffer_size: 65536
  normalization: true

training_schedule:
  warmup_epochs: 10
  total_epochs: 400
  steps_per_epoch: 2500  # approximate, depends on dataset size
  total_training_steps: 1,000,000

evaluation:
  fid_scoring_lib: "torch-fid"
  fid_real_dataset_path: /path/to/imagenet/validation_tokens
  eval_batch_size: 512
  eval_steps: 10000
  metrics:
    - FID
    - InceptionScore
    - Precision
    - Recall

output_paths:
  checkpoints_dir: "./checkpoints/"
  sample_results_dir: "./samples/"
  evaluation_results_dir: "./eval/"

# Additional parameters related to reproducibility
seed: 42
```

---

**Note:** Full configuration is available in `planning_config.yaml`
