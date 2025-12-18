# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed and comprehensive plan outlining the methodology and experiments based on the paper "Exploring the Low-Pass Filtering Behavior in Image Super-Resolution" by Deng et al., suitable for guiding implementation:

---

## Overall Approach and Objectives
- **Goal:** Reproduce and verify the core findings, especially that modern neural network-based ISR models (like SwinIR, RDN, etc.) act as low-pass filters with sinc-like impulse responses, and that this behavior is connected to their internal architecture.
- **Key innovation:** Use the hybrid response analysis (HyRA) framework to decompose neural networks into linear (low-pass filter) + non-linear components using impulse responses, spectral analysis, and frequency spectrum similarity metrics (FSDS).
- **Outcome:** Generate impulse responses, visualize their sinc-like nature, and quantify the frequency response behavior, linking it to the low-pass filtering hypotheses.

---

## A. Methodology Breakdown

### 1. Decomposition of Neural Network as a Linear + Nonlinear System

- **Core Concept:**  
  The network output `N(I)` can be decomposed into a linear part `H(I)` and a nonlinear part `G(I)`:
  \[
  N(I) = H(I) + G(I)
  \]
  
- **Linear System `H(I)` Response (Impulse Response Derivation):**  
  - Feed an impulse (Dirac delta) input `\(\delta\)` (converted to a 2D impulse image with a single pixel at the center) into the network.
  - Extract the network's *linear response* `H(I)` to this impulse, which approximates the impulse response of the linear low-pass filter represented by the network.
  - Use convolution (or cross-correlation, since the filter is symmetric) with known impulse response `H(δ)` to evaluate the response to other inputs.

- **Nonlinear System `G(I)`:**  
  - For a sampled input `I`, compute the residual `G(I) = N(I) - H(I)` (where `H(I)` is obtained by convolving the input with the impulse response from the impulse network).
  - Investigate the spectral characteristics of `G(I)` and confirm injection of high-frequency details.

---

### 2. Impulse Response Estimation

- **Create Impulse Image:**
  - Generate a high-resolution impulse image `I^{cont}` with a single pixel at the center set to 1 and others 0.
  - Downsample (via convolution or binning) to obtain low-resolution input `I^{LR}`.
- **Feed impulse to network:**
  - Use the neural network to get output `N(I^{LR})`.
  - Compute `H(I^{LR})` by convolving `I^{LR}` with the impulse response `H(δ)`.
- **Determine `H(δ)`:**
  - Directly obtained from the network response to the impulse input.
  - Visualize the impulse response for analyses (e.g., 2D sinc shape).
  - Validate sinc shape by comparing with ideal sinc functions visually and via spectral analysis.

---

### 3. Spectral Analysis

- **Fourier Transform:**
  - Compute FFTs of `N(I)`, `H(I)`, and `G(I)`:
    - Use 2D FFT on the images.
    - For impulse response `H(δ)`, FFT gives the transfer function `H(jω)`.
- **Spectral metrics:**
  - `FSDS`: Measure spectral similarity between the spectral responses of the network's output and the ideal low-pass filter.
  - Spectral representations of responses:
    - Visualize the spectral magnitude and phase.
    - Confirm sinc-like patterns in impulse response FFTs.

### 4. Spectral Characteristics and Low-pass Filter Behavior
- Confirm that `H(δ)` (impulse response) approximates sinc functions.
- Analyze how the convolution response `H(I)` behaves over various inputs, observing the low-pass filtering behavior.
- Confirm that the residual `G(I)` injects high-frequency details (visualize spectrum).

---

## B. Experiments & Implementation Details

### 1. Dataset & Input Generation
- **Dataset for training (if needed):**
  - Use publicly available datasets such as DIV2K for fidelity.
  - For impulse response estimation, synthetic impulse images are generated as described.
- **Impulse Image:**
  - Create a 2D matrix (e.g., 64×64 or 128×128) with a central pixel = 1, others = 0.
- **Downsample:**
  - To mimic the low-res input, apply a typical downsampling process (bilinear, bicubic, or the kernel as per paper) to the impulse image.
  - Alternatively, directly use the impulse as the "input" for the network.

### 2. Network Architectures to Reconstruct
- **Select models:**  
  Implement or load models equivalent to those tested: SwinIR, RDN, ESRGAN, etc.
- **Training/Evaluation Mode:**
  - Load pretrained weights or train models on datasets matching the paper (e.g., DIV2K, Urban 100).
  - For the spectral analysis, models *must* be trained/ready for inference.
 
### 3. Impulse Response Calculation
- **Feed impulse LR images into the network** to get super-resolved output.
- **Convolve LR input with impulse response**:
  - Use FFT-based convolution for efficiency.
  - Confirm sinc shape visually by plotting the impulse response in spatial domain.
- **Extract `H(δ)`**:
  - Use the network response to pure impulse to get the impulse response.

### 4. Response to Arbitrary Inputs
- Generate custom textures, textures with behaviors susceptible to spectral analysis.
- Compute `H(I)` via convolution with impulse response.
- Calculate `G(I)` as residual.

### 5. Spectral Analysis (FFT-based)
- Use `np.fft.fft2` or `torch.fft.fft2` on images.
- Compute log-magnitude spectra.
- Visualize with colormaps, highlight sinc characteristics.

### 6. Spectral Similarity via FSDS
- Implement the spectral similarity metric based on the paper's formula.
- Compare different models' responses.
- Validate if foreshadowed behavior: sinc impulse → low pass; residual → high frequencies.

---

## C. Additional Experimental Considerations

- **Impulse Response Visualization:**
  - 2D plots of the impulse response.
  - Cross-section plots comparing response shapes vs. ideal sinc.
- **Frequency Spectrum Analysis:**
  - Use log-magnitude and phase plots.
  - Focus on central peak and sinc-like side lobes.
- **Spectral Diversity:**
  - Explore responses at different layers using feature map visualizations.
  - Confirm the existence of sinc-like textures in later layers or post-activation maps.

---

## D. Hyperparameters & Technical Details

- **Impulse Image Size:**  
  At least 64×64 or 128×128 for clear spectral analysis.
- **Downsampling Kernel:**  
  Use bicubic interpolation or the natural kernel used in training.
- **FFT Size:**  
  Zero-pad FFT to avoid wrap-around artifacts.
- **Fourier Smoothing:**  
  Apply Gaussian smoothing to spectra if needed for clearer visualization.
- **Spectral Metrics:**
  - Use the formulas provided for FSDS.
  - Integrate over the entire spectrum or select specific bands.

---

## E. Validation & Verification

- **Compare impulse responses with ideal sinc functions**.
- **Spectral similarity:**  
  Compute FSDS between the impulse response's spectrum and the ideal ideal low pass.
- **Layer-wise analysis:**  
  Visualize feature maps to observe sinc-like textures emerging.

---

## F. Summary of Necessary Resources & Code

- **Codebase:**
  - Implement spectral analysis, convolution via FFT.
  - Load or reimplement models (we can adapt from Pytorch models).
  - Generate impulse images and perform the HyRA decomposition.
- **Datasets:**
  - DIV2K or other super-resolution datasets for training/fine-tuning.
  - Synthetic impulse images for response analysis.
  - Optionally, test on real images for residual high-frequency analysis.

---

This roadmap provides a detailed plan to replicate the core experiments and validate the sinc-filter behavior, spectral properties, and the HyRA framework proposed. Each step is modular, allowing incremental validation before moving to the next analysis, and should serve as an effective foundation for later code implementation.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Design a modular Python system utilizing PyTorch for models and FFT operations, NumPy and SciPy for spectral analysis, and Matplotlib for visualization. The core focus is to implement the HyRA decomposition by generating impulse responses, spectral responses, and visualizations, avoiding complex training pipelines. The system will load pre-trained models, generate impulse input images, compute responses (convolutional and spectral), and visualize sinc-like impulse responses as well as spectral spectra. Use torch.nn.functional for FFT-based convolutions and spectral analyses, and organize code into clear modules for data handling, model inference, spectral analysis, and visualization.",
    "File list": [
        "main.py",
        "impulse_generation.py",
        "model_loader.py",
        "response_analysis.py",
        "spectral_analysis.py",
        "visualization.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class ModelLoader {\n        +__init__(model_path: str)\n        +load_model() -> torch.nn.Module\n    }\n    class ImpulseGenerator {\n        +create_impulse_image(size: tuple) -> torch.Tensor\n    }\n    class ResponseAnalyzer {\n        +compute_impulse_response(model: torch.nn.Module, impulse_img: torch.Tensor) -> torch.Tensor\n        +visualize_impulse_response(h_resp: torch.Tensor) -> None\n        +compute_response_to_input(model: torch.nn.Module, input_img: torch.Tensor, h_resp: torch.Tensor) -> torch.Tensor\n        +extract_linear_response(input_img: torch.Tensor, impulse_response: torch.Tensor) -> torch.Tensor\n    }\n    class SpectralAnalysis {\n        +fft_response(image: torch.Tensor) -> complex ndarray\n        +calculate_fsds(resp1: np.ndarray, resp2: np.ndarray) -> float\n        +visualize_spectra(resp1: np.ndarray, resp2: np.ndarray) -> None\n    }\n    class Visualization {\n        +plot_impulse_response(h_resp: torch.Tensor) -> None\n        +plot_spectra(magnitude1: np.ndarray, phase1: np.ndarray, magnitude2: np.ndarray, phase2: np.ndarray) -> None\n        +plot_responses(input_img: torch.Tensor, linear_response: torch.Tensor, nonlinear_response: torch.Tensor) -> None\n    }\n\nMain --> ModelLoader\nMain --> ImpulseGenerator\nMain --> ResponseAnalyzer\nMain --> SpectralAnalysis\nMain --> Visualization\nResponseAnalyzer --> ModelLoader\nSpectralAnalysis --> ResponseAnalyzer\nVisualization --> ResponseAnalyzer\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant ML as ModelLoader\n    participant IG as ImpulseGenerator\n    participant RA as ResponseAnalyzer\n    participant SA as SpectralAnalysis\n    participant V as Visualization\n    M->>ML: new(model_path)\n    ML-->>M: load_model()\n    M->>IG: create_impulse_image(size=(128,128))\n    IG-->>M: impulse_image\n    M->>RA: compute_impulse_response(model, impulse_image)\n    RA->>RA: compute impulse response\n    RA-->>V: visualize_impulse_response(h_resp)\n    M->>RA: compute_response_to_input(model, impulse_image, h_resp)\n    RA-->>V: plot responses (input, linear, nonlinear)\n    M->>SA: fft_response(impulse_response)\n    SA-->>SA: spectral response\n    M->>SA: fft_response(network_output)\n    SA-->>SA: spectral response of output\n    M->>SA: calculate_fsds(resp1, resp2)\n    SA-->>V: visualize_spectra\n    V-->>M: complete\n    Note over M,SA,V: Repeat for various models or responses as needed.\n",
    "Anything UNCLEAR": "Clarify if there are pretrained models available, and confirm whether specific models (e.g., SwinIR, RDN) need to be implemented entirely or just loaded. Also clarify if the impulse image size should be flexible and if visualization code should support interactive displays or save static images."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "scipy==1.7.0",
        "matplotlib==3.4.3",
        "Pillow==8.3.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Main script initializes configuration, loads models via ModelLoader, creates impulse images with ImpulseGenerator, invokes ResponseAnalyzer for impulse response estimation and visualizations, calls SpectralAnalysis for spectral computations and FSDS metrics, and coordinates visualization functions. Depends on classes from response_analysis.py, spectral_analysis.py, and visualization.py."
        ],
        [
            "dataset_loader.py",
            "Provides data loading utilities if needed for training or inference, primarily for loading pre-trained models and datasets; may be minimal or placeholder depending on model input formats."
        ],
        [
            "model_loader.py",
            "Defines ModelLoader to handle loading of pretrained super-resolution models from disk (e.g., SwinIR, RDN). Provides load_model() returning a torch.nn.Module. Dependency: torch."
        ],
        [
            "impulse_generation.py",
            "Implements create_impulse_image(size) to generate a tensor with a single impulse pixel at the center. Used in main.py before response analysis."
        ],
        [
            "response_analysis.py",
            "Provides ResponseAnalyzer class with methods to compute impulse response (by feeding impulse image and extracting response), convolve with impulse response to find linear response, and visualize the impulse response as images. Relies on torch, torch.nn.functional, and utils for FFT-based convolution."
        ],
        [
            "spectral_analysis.py",
            "Contains SpectralAnalysis class with methods for FFT response extraction, FSDS spectral similarity calculations, and spectrum visualization functions. Uses numpy, scipy.fft, and matplotlib."
        ],
        [
            "visualization.py",
            "Provides plotting functions for impulse responses, spectral spectra, FSDS metrics, and response comparison images. Uses matplotlib.pyplot."
        ],
        [
            "utils.py",
            "Contains utility functions for FFT, image normalization, and possibly data conversion if needed. Supports consistency across modules."
        ]
    ],
    "Task list": [
        "model_loader.py",
        "impulse_generation.py",
        "response_analysis.py",
        "spectral_analysis.py",
        "visualization.py",
        "dataset_loader.py",
        "main.py"
    ],
    "Full API spec": "",
    "Shared Knowledge": "ResponseAnalyzer depends on the impulse response response extraction, convolution, and visualization functions. SpectralAnalysis shares FFT and spectrum visualization utilities with ResponseAnalyzer. All modules require consistent tensor/image normalization and FFT conventions. The models should be loaded from validated pretrained weights matching paper models.",
    "Anything UNCLEAR": "Clarification needed on whether to implement multiple models for comparison, or focus on a specific model (e.g., SwinIR). Also, confirm the expected impulse image size and whether any fine-tuning is feasible or the use of pretrained models only."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001  # Default starting learning rate for model training if needed
  batch_size: 16       # Typical batch size for training or inference (depends on GPU memory)
  epochs: 100          # The number of training epochs (if training is required)
  optimizer: Adam      # Optimizer choice; Adam used commonly for super-resolution models
  weight_decay: 0.0001 # Regularization parameter
  scheduler: CosineAnnealingLR  # Learning rate scheduler
  save_dir: "./checkpoints"     # Directory for saving model checkpoints
  log_interval: 100              # Steps between logging during training

model:
  name: SwinIR           # Target model architecture, e.g., SwinIR
  pretrained_path: "models/swinir_pretrained.pth"  # Path to pretrained weights if available
  input_size: 128        # Input size of impulse image (square)
  output_size: 128       # Output size after super-resolution
  scale_factor: 2        # Upscaling factor (e.g., 2x SR)

dataset:
  name: DIV2K           # Dataset name, if training/fine-tuning is needed
  train_path: "./datasets/DIV2K/train"  # Path to training dataset
  val_path: "./datasets/DIV2K/val"      # Path to validation dataset
  image_size: 128       # Size of patches used for training
  crop: True            # Whether to crop patches for training
  crop_size: 128        # Cropped patch size

evaluation:
  spectral_metrics:
    fsds_threshold: 26.0        # Threshold for FSDS to compare models' responses
  viz_save_dir: "./visualizations"  # Directory to save visualization images
  metrics_save_path: "./metrics/response_metrics.json"  # Save response spectral metrics

impulse_image:
  size: [128, 128]      # Size of impulse image to estimate impulse response

general:
  model_framework: PyTorch  # Framework used
  device: cuda            # Device to run experiments ('cuda' or 'cpu')
```

---

**Note:** Full configuration is available in `planning_config.yaml`
