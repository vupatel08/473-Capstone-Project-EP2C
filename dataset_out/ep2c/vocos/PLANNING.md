# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive plan outlining how to reproduce the VOCOS model as described in the paper, including its methodology, experimental setup, and evaluation framework. This plan will serve as a blueprint for implementing the code in subsequent steps.

---

## 1. Core Objective and Approach
- **Goal:** Develop a neural vocoder that directly generates Fourier spectral coefficients (magnitude + phase) of audio signals, aiming to achieve high audio quality and efficiency.
- **Main Innovation:** Use a GAN-based architecture that models spectral coefficients in the Fourier domain, bypassing complex phase recovery issues in traditional time-domain models.
- **Key features:**
  - Spectral domain modeling with Fourier coefficients.
  - A generator based on ConvNeXt adapted for spectral features.
  - Spectral coefficients represented using polar form: magnitude and phase (or equivalent phase parametrization).
  - Adversarial training with multi-period and multi-resolution discriminators.

---

## 2. Methodology Details (Model Architecture & Inputs/Outputs)
### 2.1 Input & Output
- **Input:** Mel-spectrogram features derived from audio, at a specified sampling rate (24 kHz).
- **Output:** Complex spectral coefficients (Fourier spectrum) per frame, represented as:
  - Magnitude (`M`)
  - Phase (`ϕ`), derived via normalized phase representation or via phase parametrization (`p`)
- **Representation of spectral coefficients:**
  - **Polar form:** \( S = M \cdot e^{jϕ} \) with \( M = \exp(m) \), \( ϕ \) in range \((-π, π]\)
  - Alternatively, phase can be parametrized using \(\cos(p)\) and \(\sin(p)\) or via a sigmoided sign + log-magnitude approach.
- **Spectrogram parameters:**  
  - Fourier window size \(n_{fft} \approx 1024\)
  - Hop length = 256 samples
  - Number of mel bins = 100
  - Sampling rate = 24 kHz

### 2.2 Spectral Representation & Phase Handling
- Use only one sideband spectrum due to conjugate symmetry.
- To compute phase:  
  - From phase parametrization \( p \), derive phase as \( \varphi = \text{atan2}(\sin(p), \cos(p)) \).
  - Wrap phase into \((-π, π]\).
- The generator outputs:  
  - Magnitude logits \( m \) (which become \( M = \exp(m) \))
  - Phase logits or parameters \( p \) (which become phase angles).

### 2.3 Generator Architecture
- Use **ConvNeXt** backbone adapted for 1D spectral features.
- Embedding layer: projects mel-spectrogram features into a hidden dimension.
- Convolutional stack:
  - 1D depthwise convolution + inverted bottleneck (pointwise convolution + GELU + LayerNorm).
  - Maintain constant temporal resolution across layers.
  - Use residual connections and possibly dilated convolutions for receptive field expansion.
- Output layer: projects features to \(n_{fft}/2 + 1\) magnitude logits + phase logits.
- Spectral coefficient reconstruction:  
  \[
  \mathbf{S} = \exp(\mathbf{m}) \cdot (\cos(\mathbf{p}) + j \sin(\mathbf{p}))
  \]
- Optional: phase and magnitude are modeled separately, or sign in the MDCT variation.

### 2.4 Discriminators
- Use **multi-period discriminator (MPD)**: analyzes periodic structures in waveform (or spectral features).
- Use **multi-resolution discriminator (MRD)**: analyzes spectral features at various resolutions.
- These discriminators process the reconstructed waveform obtained via inverse Fourier transform (or IMDCT in spectral domain).

### 2.5 Loss Functions
- **Adversarial loss:** hinge GAN loss for generator and discriminators.
- **Spectrogram (mel-spectrogram) reconstruction loss:** L1 distance between ground-truth mel-spectrograms and spectral reconstructions.
- **Feature matching loss:** based on discriminator feature maps.
- **Optional phase regularization:** to stabilize phase predictions.
- **Cycle consistency or phase wrapping constraints:** to ensure phase continuity.

### 2.6 Spectrogram & Spectral Coefficient Computation
- Compute mel spectrograms from raw audio:
  - Use `librosa` or equivalent:  
    - `n_fft=1024`, `hop_length=256`, `mel_bins=100`.
- Use STLFFT for spectral analysis.
- Spectra are modeled as real-valued vectors, with only one side (magnitude + phase).

### 2.7 Spectrogram & Spectral Coefficient Reconstruction
- Inverse Fourier:
  - Compute complex spectrogram from model output.
  - Apply inverse Fourier method (e.g., ISTFT, IMDCT, or FFT-based reconstruction).
  - Generate waveform for discriminator and loss computation.

---

## 3. Experiments & Dataset
### 3.1 Dataset
- **LibriTTS:** full training set (both train-clean and train-other).
- Sampling rate: 24 kHz.
- Mel spectrogram parameters as above.
- Data augmentation:
  - Random gain between -1 and -6 dBFS.
  - Cropping to 16,384 samples (~683 ms at 24 kHz).
- Batch size: 16.
- Number of training iterations: ideally up to 2 million, with checkpoints for evaluation.

### 3.2 Hyperparameters
- Optimizer: AdamW.
- Learning rate: initial 2e-4 with cosine decay.
- Betas: (0.9, 0.999).
- Discriminator updates per generator.
- Model parameters:
  - `n_fft=1024`
  - `hop_length=256`
  - Spectral output dimension: \(n_{fft}/2 + 1 = 513\)
  - Number of mel bins: 100
  - Hidden dimension: ~768 or as per ConvNeXt default
  - Batch size: 16
- Spectral coefficient regularization: optional, but should include spectral smoothing or phase wrapping constraints.

### 3.3 Evaluation Metrics
- **Objective:**
  - UTMOS (source-dependent metric)
  - VISQOL
  - PESQ
  - V/UV F1 score
  - Periodicity measure (smoothness of phase)
- **Subjective:**
  - 5-point MOS (Mean Opinion Score)
  - SMOS for perceptual similarity
- **Sample-based:**
  - Spectrograms visualization (ground truth vs generated)
  - Waveform listening tests
  
### 3.4 Additional Testing Scenarios
- Cross-dataset: test on out-of-distribution singing voice samples.
- Bandwidth variation experiments:
  - Model performance at different spectral bandwidths (e.g., 1.5 kHz, 3 kHz, 6 kHz) for data efficiency analysis.
- Out-of-distribution audio: evaluate robustness.

---

## 4. Implementation Considerations
- **Spectral operations:**
  - Implement spectral parametrization functions (`exp`, `atan2`, `cos`, `sin`) carefully.
  - Use FFT and inverse FFT (or IMDCT) for waveform synthesis.
- **Spectrogram processing:**
  - Standardize mel-spectrogram calculation as per paper.
- **Loss setup:**
  - Implement hinge losses for adversarial training.
  - Use feature matching on discriminator feature maps.
- **Training loop:**
  - Alternate generator/discriminator updates.
  - Log objective metrics periodically.
- **Model checkpoints:**
  - Save models every few hours.
  - Validate on a hold-out set periodically.

---

## 5. Additional Open Questions & Clarifications
- **Phase representation:** Is the phase output parameterized as \(p\) (angles), or sign + magnitude? The paper discusses both schemes—choose phase parametric approach (logit \(p\)) for training stability.
- **Spectral smoothing:** Should spectral smoothing or phase continuity regularization be explicitly used?
- **Waveform synthesis:** Use ISTFT or IMDCT? Paper favors inverse spectral transforms; detail implementation choices for real-world code.
- **Discriminator specifications:** Exact network architectures? (Likely based on prior work but should replicate MPD and MRD from referenced papers.)
- **Training stability:** Any need for gradient penalty or spectral regularization? The paper emphasizes phase robustness – may include spectral consistency constraints.

---

## Summary of the Roadmap
- Extract mel spectrograms from raw audio using specified parameters.
- Design generator based on ConvNeXt with spectral (frequency) output head predicting magnitude logits and phase logits.
- Incorporate spectral coefficient parametrization (exponential + phase parametrization).
- Use inverse Fourier transform for waveform reconstruction.
- Implement adversarial discriminators (MPD and MRD).
- Define combined loss functions: adversarial hinge loss, spectral (mel) L1 loss, feature matching.
- Train with dataset augmentations, proper optimizer, and decay.
- Evaluate comprehensively with objective metrics and perceptual MOS.
- Run extensive out-of-distribution tests and bandwidth experiments.

---

This detailed plan ensures that every component—from spectral processing to architecture, training, and evaluation—is well defined before any code is written.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular training pipeline for the Fourier-based neural vocoder 'Vocos' using open-source libraries like PyTorch for modeling, torchaudio for spectral processing, and librosa for spectrogram computation. The system consists of a spectral feature extractor, a generator network based on ConvNeXt with spectral prediction heads, discriminators (multi-period and multi-resolution), and a training loop combining adversarial, spectral L1, and feature matching losses. For waveform synthesis, the inverse FFT will be used to reconstruct audio from spectral coefficients. The code will accept mel spectrograms as input, generate spectral coefficients, synthesize waveforms, and evaluate via both objective metrics and perceptual MOS. Overall, the design emphasizes simplicity, clarity, and quality, leveraging existing deep learning and audio libraries, with the key module designed for high flexibility and experimental control.",
    "File list": [
        "app.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "spectral_utils.py",
        "losses.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class App {
        +__init__(config: dict)
        +run() -> None
    }
    class DatasetLoader {
        +__init__(config: dict)
        +load_data() -> Dataset
        +get_batch() -> Tuple[Tensor, Tensor]  # mel spectrograms, waveforms
    }
    class SpectrogramProcessor {
        +compute_mel_spectrogram(waveform: Tensor) -> Tensor
        +compute_inverse_spectrogram(coefficients: Tensor) -> Tensor  # ISTFT
    }
    class SpectralPredictor (Generator) {
        +__init__(params: dict)
        +predict_spectral_coeffs(mel_spec: Tensor) -> SpectralCoefficients
    }
    class SpectralCoefficients {
        +magnitude_logits: Tensor
        +phase_logits: Tensor  # or phase parameters p
        +to_complex() -> ComplexSpectrogram
    }
    class ComplexSpectrogram {
        +magnitude: Tensor
        +phase: Tensor
        +to_complex() -> Tensor  # complex tensor for inverse FFT
    }
    class Discriminator {
        +__init__(params: dict)
        +forward(waveform: Tensor) -> TensorScore
    }
    class Trainer {
        +__init__(model: SpectralPredictor, discriminators: list, dataset: Dataset, config: dict)
        +train() -> None
    }
    class Evaluator {
        +__init__(model: SpectralPredictor, dataset: Dataset)
        +evaluate() -> dict
    }
    class Losses {
        +adversarial_loss(pred_score_real: Tensor, pred_score_fake: Tensor) -> Tensor
        +spectral_l1_loss(mel_true: Tensor, mel_pred: Tensor) -> Tensor
        +feature_matching_loss(feat_real: list, feat_fake: list) -> Tensor
    }
    App --> DatasetLoader
    App --> SpectrogramProcessor
    App --> SpectralPredictor
    App --> Discriminator
    App --> Trainer
    App --> Evaluator
    Trainer --> SpectralPredictor
    Trainer --> Discriminator
    SpectralPredictor --> SpectrogramProcessor
    SpectralPredictor --> SpectralCoefficients
    SpectralCoefficients --> ComplexSpectrogram
    ComplexSpectrogram --> SpectralCoefficients
    Discriminator --> SpectralPredictor
    Discriminator --> SpectrogramProcessor
    Losses --> Trainer
",
    "Program call flow": "
sequenceDiagram
    participant C as App
    participant DL as DatasetLoader
    participant SP as SpectralPredictor
    participant SPH as SpectralCoefficients
    participant SC as SpectrogramProcessor
    participant D as Discriminator
    participant TR as Trainer
    participant EV as Evaluator
    C->>DL: load_data()
    DL-->>C: dataset
    C->>SP: initialize model()
    C->>TR: start training(model, dataset)
    Note right of TR: Loop over epochs
    TR->>DL: get_batch()
    DL-->>TR: batch data (mel_spec, waveform)
    TR->>SC: compute_mel_spectrogram(waveform)
    SC-->>TR: mel_spectrogram
    TR->>SP: predict spectral coefficients(mel_spectrogram)
    SP-->>TR: spectral coefficients (m_logits, p_logits)
    TR->>->>SC: reconstruct complex spectrogram from spectral coefficients
    SC-->>TR: complex spectrogram
    TR->>D: discriminate waveform (via inverse FFT)
    D-->>TR: real/fake scores
    TR->>Losses: compute adversarial loss, spectral L1, feature matching
    Losses-->>TR: total loss
    TR-->>C: update generator/ discriminator
    Note right of C: Repeat until convergence
    C->>EV: evaluate() on validation set
    EV->>SP: generate waveforms per validation mel
    EV->>SpectrogramProcessor: compute objective metrics (e.g., PESQ, MOS)
    EV-->>C: report metrics
",
    "Anything UNCLEAR": "Clarify exact spectral head design (phase parametrization vs phase logits), discriminator network architectures, and specific spectral post-processing steps (e.g., windowing functions, inverse FFT method). Need confirmation on whether to use ISTFT or IMDCT for waveform reconstruction, and details of spectral normalization or phase regularization used during training."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "librosa==0.8.0",
        "torchaudio==0.11.0",
        "numpy==1.21.0",
        "scipy==1.7.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "spectral_utils.py",
            "Contains functions for mel spectrogram extraction, inverse FFT, IMDCT, and spectral param conversions (e.g., log-magnitude, phase parametrization, symlog). Supports spectral transform, inverse transforms, and spectral coefficient handling. Essential for decoding and encoding spectral representations."
        ],
        [
            "dataset_loader.py",
            "Handles loading raw audio, computes mel spectrograms, and prepares batches. Depends on spectral_utils to extract features. Supplies data to training and evaluation modules."
        ],
        [
            "model.py",
            "Defines the generator architecture based on ConvNeXt adapted for spectral output, spectral coefficient heads (magnitude logits and phase logits or parameters). Implements forward step to produce spectral coefficients from mel spectrograms."
        ],
        [
            "discriminator.py",
            "Implements MPD and MRD discriminators for waveform or spectral input. Receives reconstructed audio waveform, outputs real/fake scores and feature maps for feature matching."
        ],
        [
            "trainer.py",
            "Coordinates training process: loading batches, running generator, spectral synthesis via inverse FFT, discriminator steps, computing all losses (adversarial, spectral L1, feature matching), updating networks. Manages training loops, checkpoint saves, logging."
        ],
        [
            "evaluation.py",
            "Performs inference: generates spectral coefficients from mel spectrograms, reconstructs waveforms, computes objective metrics (PESQ, VISQOL, MOS) and visualizes spectrograms for qualitative analysis. Depends on spectral_utils for inverse transforms."
        ],
        [
            "main.py",
            "Program entry point: initializes configuration, loader, models, trainers; runs training loop; executes periodic evaluation; handles experiment setup and checkpoint management. Integrates all modules coherently."
        ]
    ],
    "Task list": [
        "spectral_utils.py",
        "dataset_loader.py",
        "model.py",
        "discriminator.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Common spectral processing functions (spectrogram extraction, inverse FFT, spectral param conversions) shared between spectral_utils.py, evaluation.py, and trainer.py. Consistent configuration variables for spectral sizes, FFT parameters, phase prediction schemes, and spectral normalization techniques.",
    "Anything UNCLEAR": "Clarify whether to use IMDCT or ISTFT for waveform reconstruction, and confirm spectral representation detail (phase parametrization vs phase logits). Also, specify if any spectral smoothing or regularization (e.g., phase wrapping) is required during training."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0002  # AdamW optimizer initial learning rate
  batch_size: 16        # Batch size for training
  total_iterations: 2000000  # Total training iterations (based on paper: 2M)
  decay_strategy: cosine  # Learning rate decay schedule
  optimizer_betas: [0.9, 0.999]
  AdamW_weight_decay: 0.01  # Default weight decay

model:
  fft_size: 1024            # FFT window size
  hop_length: 256           # Hop length between frames
  mel_bins: 100             # Number of mel spectrogram bins
  spectral_dim: 513         # Number of spectral coefficients (fft_size/2 + 1)
  hidden_dim: 768           # Hidden dimension inside generator network
  spectral_heads: 2         # Number of spectral output heads: magnitude logits and phase (or phase parameters)
  phase_representation: 'phase_logits'  # Options: 'phase_logits', 'phase_params', 'sign_logmag'

dataset:
  sample_rate: 24000        # Sampling rate in Hz
  segment_size: 16384       # Number of samples per training segment (~683 ms)
  mel_params:
    n_fft: 1024
    hop_length: 256
    n_mels: 100
  data_augmentation:
    random_gain_db: [-6, -1] # Random gain range in dBFS

training:
  total_iterations: 2000000
  save_interval: 100000     # Save model checkpoints every 100k iterations
  log_interval: 1000        # Log training metrics every 1k iterations

evaluation:
  metrics:
    pesq: true
   .visqol: true
    mos: true
  eval_batches: 16          # Number of batches for evaluation
  eval_samples: 1000        # Number of samples for objective metrics

loss:
  adversarial:
    type: hinge
  spectral_l1: true
  feature_matching: true
  spectral_regularization: true  # Optional, if spectral smoothing or phase regularization is used
```

---

**Note:** Full configuration is available in `planning_config.yaml`
