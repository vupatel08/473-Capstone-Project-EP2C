# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, detailed plan outlining how to reproduce the ControlVideo methodology, structured into key sections for clarity:

---

### 1. **Reproduction Overview & Core Methodology**
- **Objective:** Generate controllable, high-quality videos from text prompts, conditioned on motion sequences (e.g., depth maps, edges, pose sequences) **without training**.
- **Main idea:** Adapt a pre-trained text-to-image diffusion model (ControlNet + Stable Diffusion) for video generation by:
  - Inflating the U-Net for temporal modeling.
  - Introducing cross-frame attention mechanisms.
  - Applying an interleaved-frame smoother to enhance temporal consistency and reduce flickering.
  - Employing hierarchical sampling for efficient long-video generation.
- **Key components to implement:**
  - A ControlNet-finetuned SD model (or a compatible control-conditioned diffusion model).
  - Modified U-Net architecture with 3D convolutions and cross-frame attention.
  - Interleaved-frame smoother algorithm (Alg. 1) operating during denoising.
  - Hierarchical sampler for multi-step generation.

---

### 2. **Model Components & Architecture**
#### 2.1. Base Diffusion Model
- Start with a **pre-trained SD (Stable Diffusion) model**, ideally the vanilla SD or a publicly available variant.
- Adapt ControlNet:
  - Use the ControlNet weights or replicate architecture based on ControlNet (Zhang & Agrawala, 2023).
  - ControlNet should accept additional conditions like depth, edges, or pose maps.
  - Finetune ControlNet (if weights are available) on image datasets with corresponding structural maps or, alternatively, use them directly.

#### 2.2. Inflating the U-Net for Video
- Convert 2D Conv layers to 3D:
  - Replace kernel size `3x3` with `1x3x3` to treat the temporal axis.
  - Adjust the number of channels accordingly.
- Extend the self-attention modules:
  - Implement cross-frame attention by querying from neighboring frames (see Eq. in Sec. 3.1).
  - For example, modify the self-attention to attend across frames, querying from `[z_t^i]` and other reference frames.
  - Use `W^Q`, `W^K`, `W^V` with the same logic as 2D but across the temporal dimension.
- Incorporate auxiliary U-Net from ControlNet for structural guidance, kept fixed or finetuned.

#### 2.3. Cross-Frame Attention Mechanism
- Implement fully cross-frame attention as the primary mechanism.
- Consider reference frame selection:
  - For experience, use all previous frames or recent frames within a window.
  - Experiment with more sparse attention for efficiency.

#### 2.4. Hierarchical Sampling
- Implement a coarse-to-fine sampling approach:
  - Downscale input conditions (e.g., pose maps, edges, etc.).
  - Generate videos initially at low resolution or fewer steps.
  - Increase detail and resolution in subsequent passes.
- Long video generation:
  - Use a hierarchical sampler: generate segments sequentially, then stitch smoothly.
  - Potentially apply the hierarchical sampling in a sliding or chunked manner.

---

### 3. **Interleaved-Frame Smoother Algorithm (Alg. 1)**
#### 3.1. Operational steps:
- **Input:** Latent representations `z_t`, structural conditions `c`, text prompt `τ`.
- **Process:**
  - Convert `z_t` to RGB frames via DDM (`𝔇`).
  - At specific timesteps (large noise levels), smooth the video via interpolation:
    - Divide all three-frame clips into even and odd middle frames sets.
    - For each clip:
      - Interpolate middle frames from neighboring frames in pixel space or latent space.
      - Re-project interpolated RGB frames back into latent space (`𝔈`).
      - Use these smoothed latent sequences to update `z_{t-1}`.
- **Implementation details:**
  - Select timesteps for smoothing based on the paper's ablation goals: e.g., {48, 49}, {30, 31}, {0, 1}.
  - The smoothing step involves reconstructing RGB frames, interpolating, and then denoising.

#### 3.2. Integration during denoising:
- During each denoising step, at specified timesteps:
  - Apply the smoothing procedure.
  - Proceed with the usual DDIM update as in Eq. 4.
- At intermediate steps, only perform smoothing at the selected timesteps, preserving quality.

---

### 4. **Experimental Setup & Dataset Preparation**
#### 4.1. Dataset
- Prepare a dataset of video clips with structural annotations:
  - Use DAVIS or a similar video dataset.
  - Extract structure maps:
    - Depth maps: Using pretrained depth estimation (e.g., MiDaS).
    - Edges: Using Canny edge detector.
    - Human pose sequences: Using OpenPose or HRNet for keypoints.
  - Manually annotate or use existing labels for the source caption.
  - For diversity, select various scenes: urban, nature, indoor, etc.

#### 4.2. Structure condition annotations
- For each video:
  - Generate edge maps, depth maps, or pose sequences.
  - Save as conditioning maps aligned with frames.
- For prompts:
  - Use descriptive text prompts as per the paper, e.g., "A swan moving in a lake" or "A robot dancing on the road".

#### 4.3. Text prompts
- Gather prompts matching the source videos.
- For user study or evaluation:
  - Generate multiple prompts per video with slight variations.

---

### 5. **Training/Finetuning (if applicable)**
- The authors emphasize training-free approach, but:
  - To improve ControlNet, finetune it on the collected dataset or use it as-is if weights are publicly available.
- Fine-tune prompt-conditioning or the cross-frame attention modules if needed, using a small dataset or self-supervised consistency.

---

### 6. **Generation Pipeline**
- **Input:** Text prompt + motion condition (edge, depth, pose map).
- **Steps:**
  1. Initialize latent `z_T` from Gaussian noise.
  2. For `t` in decreasing order:
     - Compute `z_{t-1}` via DDIM update, optionally applying the hierarchical smoother at selected steps.
     - During each step, incorporate control conditions via ControlNet, guiding the diffusion process.
  3. Apply interleaved-frame smoother periodically (per ablation) during denoising.
  4. If generating long videos:
     - Use hierarchical sampling:
       - Generate segments with low res or at key frames.
       - Sequence segments with consistency.
     - Use the hierarchical sampler to reduce computational cost.

---

### 7. **Metrics & Evaluation**
- **Quantitative:**
  - FC (Frame consistency): cosine similarity between consecutive frames’ features (e.g., CLIP features or pixel-based).
  - PC (Prompt consistency): cosine similarity between text embedding (CLIP text encoder) and each frame's image embedding.
  - WE (Video quality): perceptual metrics (LPIPS, FID).
- **Qualitative:**
  - Visual inspection.
  - User preference studies as described, involving raters comparing videos.

---

### 8. **Implementation Details & Hyperparameters**
- **Diffusion parameters:**
  - Number of denoising steps: 50 (per paper), but experiment with fewer.
- **Model scale:**
  - Inflation factor for temporal extension: scale factor of 0.3 for the cross-frame attention scope.
- **Smoothing steps:**
  - Temporal smoothing at steps {48, 49}, {30, 31}, or {0, 1}.
- **Number of smoothing iterations:**
  - Typically 2 steps.
- **Hierarchical sampling resolution:**
  - Generate at 512x512 pixel resolution.
  - Use low-res versions (e.g., 256x256) for initial segments if needed.
- **Control conditions:**
  - Use control maps with structural features (edges, depth, pose).
- **Computational resources:**
  - Target hardware: One GPU (like NVIDIA 2080Ti) for near real-time; optimize code for memory efficiency.

---

### 9. **Potential Uncertainties / Clarifications Needed**
- Exact architecture of the inflated U-Net: kernel sizes, number of attention heads, layer depths.
- Specific control condition types supported (depth, edges, pose, etc.).
- Denoising schedule: step-by-step noise schedule (`β_t`), and whether to adapt or use standard DDIM.
- Exact interpolation method: in latent space or pixel space? Equation suggests pixel space, but implementation detail might vary.
- Is the ControlNet strictly finetuned on the dataset, or is a pre-trained ControlNet used directly?

---

### 10. **Summary of Key Implementation Steps for Code Later**
- Set up environment with PyTorch, diffusers or custom diffusion implementation.
- Load pre-trained SD + ControlNet weights.
- Implement 3D U-Net with cross-frame attention.
- Incorporate the Edge/Depth/pose condition encoding.
- Modify the denoising loop:
  - Integrate the smoothing procedure at selected steps.
  - Use hierarchical sampling for large sequences.
- Final output: convert latent to RGB frames, compile into video.
- Evaluate via metrics or user study.

---

This roadmap should serve as a solid foundation for a reproducible implementation aligned with the described paper. Would you like me to proceed with a detailed step-by-step code plan next?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular pipeline based on open-source diffusion libraries (like diffusers or a custom PyTorch implementation). Key components include a pre-trained Stable Diffusion backbone, a ControlNet module conditioned on structural maps (edges, depth, pose), a 3D U-Net with cross-frame attention for temporal modeling, and an algorithm for interleaved-frame smoothing integrated into the denoising loop. The system will generate videos by iteratively denoising latent representations, applying cross-frame attention for temporal coherence, and smoothing selected steps (e.g., large noise levels) to reduce flickering. Hierarchical sampling will be implemented by progressively increasing detail across multiple denoising passes and cropping or segmenting long sequences. Utility modules for feature extraction (edge, depth, pose), condition encoding, and prompt embedding will be included. The configuration will be hyperparameter-driven, allowing flexibility for timestep schedules, smoothing frequency, and model scales.",
    
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "diffusion_utils.py",
        "trainer.py",
        "video_utils.py",
        "evaluation.py",
        "config.py"
    ],
    
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(dataset_path: str, annotations: dict)
        +load_structures() -> List[Tuple[frames: list, condition_maps: dict]]
    }
    class ConditioningEncoder {
        +__init__(condition_type: str)
        +encode(condition_map: np.ndarray) -> torch.Tensor
    }
    class TextPromptEmbedder {
        +__init__(model_name: str)
        +embed(prompt: str) -> torch.Tensor
    }
    class ControlNet {
        +__init__(control_type: str)
        +load_weights(weights_path: str)
        +apply_control(x: torch.Tensor, condition: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor
    }
    class DiffusionModel {
        +__init__(model_path: str)
        +inflate_for_video(): None
        +denoise(z_t: torch.Tensor, c: torch.Tensor, prompt_emb: torch.Tensor, cross_frame_attn: bool) -> torch.Tensor
        +convert_latent_to_rgb(z: torch.Tensor) -> np.ndarray
    }
    class CrossFrameAttention {
        +__init__(num_heads: int)
        +apply(z_frames: torch.Tensor) -> torch.Tensor
    }
    class InterleavedFrameSmoother {
        +smooth(z_sequence: torch.Tensor, t_steps: list) -> torch.Tensor
        +interpolate_frames(frames: List[np.ndarray]) -> np.ndarray
    }
    class HierarchicalSampler {
        +generate_segment(initial_latent: torch.Tensor, num_steps: int) -> torch.Tensor
        +stitch_segments(segments: List[torch.Tensor]) -> torch.Tensor
    }
    class VideoUtils {
        +save_frames_as_video(frames: List[np.ndarray], filename: str): None
        +extract_features(image: np.ndarray, feature_type: str) -> np.ndarray
    }
    class EvaluationMetrics {
        +compute_fid(videos: List[np.ndarray]) -> float
        +compute_clip_similarity(videos: List[np.ndarray], text_embeddings: torch.Tensor) -> float
        +compute_temporal_consistency(videos: List[np.ndarray]) -> float
    }
    
    Main --> DatasetLoader
    Main --> ConditioningEncoder
    Main --> TextPromptEmbedder
    Main --> ControlNet
    Main --> DiffusionModel
    DiffusionModel --> CrossFrameAttention
    DiffusionModel --> InterleavedFrameSmoother
    DiffusionModel --> HierarchicalSampler
    DiffusionModel --> VideoUtils
    Main --> EvaluationMetrics
    ControlNet --> DiffusionModel
    CrossFrameAttention --> DiffusionModel
    InterleavedFrameSmoother --> DiffusionModel
    HierarchicalSampler --> DiffusionModel
    VideoUtils --> Main
    EvaluationMetrics --> Main
    DatasetLoader --> Main
    ConditioningEncoder --> Main
    TextPromptEmbedder --> Main,
    
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant CE as ConditioningEncoder
    participant TPE as TextPromptEmbedder
    participant CN as ControlNet
    participant DM as DiffusionModel
    participant CFA as CrossFrameAttention
    participant IFS as InterleavedFrameSmoother
    participant HS as HierarchicalSampler
    participant VU as VideoUtils
    participant EM as EvaluationMetrics
    
    M->>DL: initialize()
    DL-->>M: load data
    M->>CE: encode condition maps
    M->>TPE: embed prompt
    M->>CN: load weights
    CN-->>DM: prepare control application
    M->>DM: inflate for video
    loop denoising steps t
        DM->>CFA: apply cross-frame if enabled
        DM->>DM: denoise latent z_t with control
        alt timestep in smoothing schedule
            DM->>IFS: smooth(z_sequence, t)
        end
    end
    DM->>VU: convert latent to rgb
    VU->>Main: save frames as video
    M->>EM: evaluate generated videos
    EM-->>Main: report metrics
    Main--> end
    "
    
    "Anything UNCLEAR": "Details on the exact architecture configuration (layer sizes, number of attention heads), and whether finetuning ControlNet is needed or pre-trained weights are used directly. Clarification on how to handle structural map generation or availability of pose extraction tools."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "torchvision==0.10.0",
        "transformers==4.12.0",
        "diffusers==0.3.0",
        "opencv-python==4.5.3",
        "scikit-image==0.18.3",
        "scikit-learn==0.24.2",
        "tqdm==4.62.3",
        "PILLOW==8.3.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class that handles loading videos, extracting structural maps (edges, depth, pose sequences) using external tools (e.g., MiDaS, OpenPose), and prepares data for training or inference. It depends on data preprocessing modules, and outputs structured data for the training pipeline."
        ],
        [
            "conditioning.py",
            "Defines ConditioningEncoder class that encodes various control maps into tensor conditioning inputs suitable for ControlNet. It relies on image processing libraries (OpenCV, PIL) and model loading (e.g., pretrained encoders)."
        ],
        [
            "text_prompt.py",
            "Implements TextPromptEmbedder class that uses CLIP or similar models to embed given textual prompts into latent vectors for conditioning in the diffusion process."
        ],
        [
            "model.py",
            "Contains the core model definitions: a modified 3D U-Net with cross-frame attention modules, the ControlNet architecture, and support for inflating 2D convolutions to 3D. This file depends on torch and transformers. It provides functionalities to load weights, inflate the architecture, and perform denoising steps."
        ],
        [
            "diffusion_utils.py",
            "Provides utility functions for diffusion process steps: sampling, interleaved smoothing, latent-to-image conversion, and interpolation in latent/visual space. It includes definitions for DDIM, DDPM sampling, and smoothers, depending on torch and numpy."
        ],
        [
            "video_utils.py",
            "Handles saving generated frame sequences into video files, extracting features for metrics, and applying post-processing. Uses OpenCV and OpenCV-Video libraries."
        ],
        [
            "trainer.py",
            "Implements the main training or inference loop, controlling the denoising process: initializing latents, applying cross-frame attention, integrating the smoother at scheduled timesteps, and managing hierarchical sampling when needed. Calls diffusion_utils and model modules."
        ],
        [
            "evaluation.py",
            "Contains classes/functions to calculate metrics like FID, CLIP similarity, and temporal consistency from generated videos for evaluation. Depends on pretrained feature extractors (e.g., CLIP, video feature extractors)."
        ],
        [
            "main.py",
            "Main entry script that integrates all modules: loads datasets, encodes conditions & prompts, initializes the model, runs the denoising with Smoothing & Cross-Frame attention, saves outputs, and conducts evaluation. Coordinates the task flow in sequence, controlling dependencies between modules."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "conditioning.py",
        "text_prompt.py",
        "model.py",
        "diffusion_utils.py",
        "video_utils.py",
        "evaluation.py",
        "trainer.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Utilize pretrained CLIP for text embeddings, pretrained MiDaS or HRNet for depth/pose extraction, and ControlNet weights for control maps. The code for latent smoothing, hierarchical sampling, and attention modifications should be compatible with standard PyTorch diffusion pipelines. Common parameters include denoising steps, smoothing schedule, control map types, and model inflation/configuration.",
    "Anything UNCLEAR": "Clarification needed on the exact configurations for the inflated 3D U-Net architecture—layer sizes, number of attention heads, and whether ControlNet weights are to be finetuned or used directly. Also, guidance on preferred hardware constraints and whether real-time performance is a goal."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  # Number of denoising diffusion steps; standard setting in the paper is 50.
  denoising_steps: 50
  # Batch size for inference and smoothing procedures.
  batch_size: 1
  # Learning rate is not explicitly specified for training, as the method is training-free.
  learning_rate: null
  # Number of smoothing steps applied during the interleaved-frame smoothing at selected timesteps.
  smoothing_steps: 2
  # Timestep schedule for applying the smoother; default at large noise levels (e.g., steps 48, 49).
  smoothing_timesteps: [48, 49]
  # Hierarchical sampling: default resolution for initial low-res generation.
  low_res_size: 256
  # Final high-res size for the generated video frames.
  high_res_size: 512
  # Total number of frames in the generated video.
  total_frames: 125
  # The number of hierarchical segments for long video generation.
  hierarchical_segments: 4

model:
  # Inflation scale for extending 2D unet to handle temporal dimension.
  inflation_scale: 0.3
  # Number of attention heads in the cross-frame attention modules.
  num_heads: 8
  # Use full cross-frame attention, as indicated as best performance.
  use_full_cross_attention: true

conditioning:
  # Types of control conditions supported; e.g., edges, depth, pose.
  control_type: ["edges", "depth", "pose"]
  # ControlNet weights path (can be pre-loaded or finetuned; specify accordingly).
  controlnet_weights_path: "weights/controlnet.pth"

prompt:
  # Text prompt to condition the video generation.
  prompt: "A dynamic scene matching the structural condition"

evaluation:
  # Metrics to evaluate quality and consistency.
  metrics: ["FID", "CLIP similarity", "Temporal consistency"]
  # Path to save evaluation outputs.
  evaluation_output_path: "outputs/evaluation/"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
