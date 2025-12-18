# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, step-by-step plan to reproduce the RECOMBINER methodology and experiments as described in the paper. The plan covers key technical details, experimental setup, data requirements, hyperparameters, and evaluation protocols so that an implementation can be developed systematically later.

---

## 1. Core Methodology Outline

### a. Objective & Framework
- **Goal**: Learn a Bayesian implicit neural representation (INR) that can be optimized for joint rate-distortion performance, then compress data via inference and Bayesian coding.
- **Approach**:
  - Use a neural network g(·, ·, ·) mapping coordinates to values (pixels, audio samples, voxels, etc.)
  - Place a Gaussian mean-field variational posterior over INR weights w.
  - Train by maximizing a β-ELBO (variational free energy) balancing rate (KL between q(w) and prior p(w)) and distortion (mean squared/error).
  - Use a hierarchical Bayesian model for patches (for high-res images, video, 3D data), sharing information via global and local variables.
  - Incorporate learnable positional encodings to capture local details.
  - Employ a linear reparameterization A to enrich expressiveness of the variational family.
  - During compression, infer q(w), then encode it with REC (a Bayesian coding scheme).

### b. Key Components
- **Network architecture**:
  - INR defined as a small network: several layers (e.g., 4), hidden units (e.g., 32), with SIREN activations.
  - Positional Encoding: Fourier features concatenated with learnable positional encodings h_z, generated via an upsampling CNN (Figure 6).
  - Hierarchical patches: for high-res data, split into overlapping patches and model dependencies hierarchically (Appendix B.2, Figure 2).
  
- **Reparameterizations & Priors**:
  - Linear reparameterization: w = h_w A, where h_w has factorized Gaussian posterior, A is learned.
  - Variational posterior for all parameters (including A, positional encodings, hierarchical weights).
  - Hierarchical prior models for patches: global, group, patch levels, with Gaussian assumptions per Appendix B.2.

- **Training Objective (Equation 1)**:
  - Maximize β-ELBO:
    \[
    \mathcal{L} = \beta D_{KL}(q(w) \| p(w)) + \frac{1}{D} \sum_{i=1}^D \mathbb{E}_{q(w)}[\Delta(y_i, g(x_i, \phi(x_i), w))]
    \]
  - Distortion Δ: MSE or perceptual loss depending on data modality.
  - β: adaptively tuned to fit target bitrate C (per Algorithm 1).

- **Bayesian coding**:
  - Encode q(w) using a REC scheme suitable for Gaussian-valued parameters.
  - For high-res/patch data, split into blocks and encode as per Appendix B.2, permuting/blocking the weights, to spread KL uniformly.

---

## 2. Data Requirements & Experimental Setup

### a. Data Modalities
- **Images**: CIFAR-10, Kodak, with different resolutions and patch-based approaches for high-res.
- **Audio**: LibriSpeech (or similar), with raw waveforms sampled at 16KHz.
- **Video**: Clips from datasets like YouTube, or synthetic sequences.
- **3D Protein Structures**: e.g., Protein Data Bank datasets, with point cloud or mesh representations.

### b. Data Processing & Preparation
- Normalize data to zero mean and unit variance if required.
- For images:
  - Extract patches (e.g., 64×64, 16×16, 128×128) depending on experiment.
  - Generate coordinate grids for pixels.
- For audio:
  - Segment into fixed-length chunks.
  - Generate time indices.
- For 3D data:
  - Convert atomic coordinates into voxel grids or point clouds.
  - Use spatial coordinates as input features.

### c. Data for Patch-based training
- Split high-res images/video frames into overlapping patches as on Kodak, with specified patch sizes.
- For protein, treat each structure as a single point set or voxel grid, possibly decomposing into subregions if beneficial.

---

## 3. Model Architecture & Hyperparameters

### a. INR Network
- 4 layers, each with 32 hidden units.
- Activation: Sine (SIREN) with custom bias if needed.
- Output dimension: data-specific (e.g., 3 for RGB, 1 for grayscale, 1 for audio, 3 for 3D).
- Fourier feature encoding: dimension 16.
- Positional encodings:
  - Reshape h_z (learnable, size depending on resolution: e.g., 128×50 for CIFAR-10).
  - Upsampling CNN: small architecture (Figure 6, Appendix B.1), kernel sizes, e.g., 5×5, 3×3, with layer widths 16→32→64, etc.

### b. Variational Distributions
- q(h_w): mean-field Gaussian, initialized with small variance (e.g., 10^{-6}).
- Variational parameters (μ, ρ) per weight, a total of ~3K parameters for all weights.
- A: learned linear transformation matrices (per layer or global, e.g., block-diagonal for efficiency).
- Hierarchical variables: global, group, and patch level means and variances, modeled as Gaussians (Appendix B.2).

### c. Hyperparameters
- β initial: e.g., 0.3 bpp equivalent (per Table 3,4,6).
- Step size for β adjustment: τ_C = 0.5.
- Prior variances: same across data.
- Training epochs: sufficient for convergence; e.g., 550 epochs for CIFAR-10.
- Sample size for rate estimation: 1 (or a small number, e.g., 5, for stable estimate).
- Batch size:
  - Images: ~50 patches.
  - Audio/video: as per resource constraints.

---

## 4. Training Procedure & Implementation Details

### a. Initialization
- Initialize network weights: SIREN |ϕ|~U(−1/a, 1/a), with a = d_in or as per Appendix B.
- A matrices: start with identity or small random matrices.
- Positional encodings: small Gaussian noise + CNN upsampling from h_z.
- β: start from small (e.g., 0.3), adaptively tuned.

### b. Optimization Loop (Algorithm 1 + Appendices)
- For each iteration:
  1. Fix prior p(w), A, and φ (upsampling CNN).
  2. Infer q(w) (μ, ρ by gradient descent) over the dataset, minimizing Equation (6).
  3. Update prior parameters (μ, σ) per Equation (7).
  4. Adjust β based on target code rate using described heuristic.
- Use Adam optimizer with learning rate 1e-4, momentum as appropriate.
- For patches, parallelize inference across patches with permutation strategies.
- For hierarchical models, update global and group variables as per Appendix B.2.

### c. Early Stopping & Tuning
- Monitor β-ELBO, distortion, and estimated rate via KL.
- Adjust β to match target bits per pixel/atom.
- Use validation patches or structures for hyperparameter tuning.

---

## 5. Compression (Inference and Bayesian Coding)
- After training:
  - Infer q(w) on new data (images, audio, structures).
  - Encode the posterior sample with REC:
    - Permutation to spread KL evenly.
    - Use depth-limited A* or approximate entropy coding (as in Guo et al. 2023, Flamich et al. 2020).
  - Reconstruct by passing coordinate data through the network with sampled weights.

---

## 6. Evaluation Metrics & Validation
- **Rate**: bits per pixel/atom/sample, calculated from expected KL of posterior over prior.
- **Distortion**:
  - Images & video: PSNR, SSIM.
  - Audio: PSNR, SI-SDR.
  - Proteins: RMSD, bits per atom.
- **Plot curves**: rate-distortion (RD) curves compare with baselines (VCINR, VC, JPEG, MP3, etc.).
- **Qualitative**:
  - Visual inspection of reconstructions and residuals.
  - 3D structure overlays (ground truth vs decoded).

---

## 7. Additional Considerations & Clarifications
- **Implementation gaps or uncertainties**:
  - Exact CNN architecture for positional encodings (Appendix B.1).
  - Exact permutation strategies (Appendix B.2).
  - How hierarchical patch models are structured when stacking global, group, patch variables.
  - Choice of distortion functions per modality (likely MSE or perceptual loss).
  - REC coding details: use of Gaussian coding scheme possibly with approximate entropy codes.
- **Extensions**:
  - Whether to include hierarchical levels for other data modalities (audio, protein).
  - Variance tuning heuristics for β.

---

## Summary of Implementation Roadmap
- Build small SIREN models with Fourier features and learnable positional encodings.
- Implement linear reparametrization of weights during training.
- Define β-ELBO loss with an adaptive β schedule.
- For data: prepare patches, coordinate grids, and labels.
- Train with a schedule that adjusts β to target rate C.
- During inference: sample q(w), encode via Bayesian coder, reconstruct.
- Evaluate RD performance on datasets matching the paper's experimental setup.

---

This plan provides a detailed, modular blueprint to reproduce the RECOMBINER framework and experiments. Each component aligns with the paper's methodology and can be incrementally implemented, debugged, and optimized later.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular system using PyTorch for neural network components, NumPy for data handling, and a Bayesian coding library (e.g., bits-back with a custom or open-source implementation) for entropy coding. The system will consist of data loading modules, model definitions for the INR network with positional encodings, variational parameters including A matrices, and hierarchical patch modeling. A training loop will optimize the β-ELBO with an adaptive schedule, inferring the posterior q(w) and updating priors. Post-training, we will sample from q(w) and encode using Bayesian coding schemes with permutations and block structures. Decoding exploits the same networks and Bayesian codes to reconstruct data. The system will include core classes: DatasetLoader, INRModel, VariationalDistribution, HierarchicalPatchModel, Trainer, and Evaluator, with clear methods for inference, training, encoding, and decoding.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "variational.py",
        "hierarchical_patch.py",
        "trainer.py",
        "coding.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run_experiment()
    }
    class DatasetLoader {
        +__init__(config: dict)
        +load_data() -> Dataset
    }
    class Dataset {
        +get_sample(index: int) -> dict
        +get_patch(index: int) -> Tensor
        +__len__() -> int
    }
    class INRModel {
        +__init__(args: dict)
        +forward(coords: Tensor, pos_encodings: Tensor, params: dict) -> Tensor
        +sample_weights() -> Tensor
    }
    class VariationalDistribution {
        +__init__(shape: list, init_params: dict)
        +sample() -> Tensor
        +kl_divergence(prior: VariationalDistribution) -> float
        +update_params(new_params: dict) -> None
    }
    class HierarchicalPatchModel {
        +__init__(global_params: dict, patch_params_list: list)
        +infer_patch_weights(patch_indices: list) -> list
        +update_global_params() -> None
    }
    class Trainer {
        +__init__(model: INRModel, hier_model: HierarchicalPatchModel, variational: VariationalDistribution, dataset: Dataset, target_rate: float)
        +train() -> None
        +adjust_beta()
    }
    class Encoder {
        +encode_weights(weights: Tensor) -> Bitstream
        +encode_data(data: dict, weights: Tensor) -> Bitstream
    }
    class Decoder {
        +decode_weights(bitstream: Bitstream) -> Tensor
        +decode_data(bitstream: Bitstream, coords: Tensor, pos_encodings: Tensor) -> Tensor
    }
    class Evaluation {
        +evaluate_reconstruction(original: Tensor, reconstructed: Tensor) -> dict
    }
    Main --> DatasetLoader
    Main --> Trainer
    Trainer --> INRModel
    Trainer --> HierarchicalPatchModel
    Trainer --> VariationalDistribution
    Encoder --> VariationalDistribution
    Decoder --> VariationalDistribution
    Evaluation --> INRModel
    Evaluation --> Dataset
    HierarchicalPatchModel --> INRModel
    HierarchicalPatchModel --> VariationalDistribution
    """,
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant TR as Trainer
    participant ENC as Encoder
    participant DEC as Decoder
    participant EVAL as Evaluation
    M->>DL: initialize(config); load_data()
    DL-->>M: dataset
    M->>TR: initialize with dataset, target_rate
    TR->>TR: train()
    loop inference on new data
        TR->>TR: infer posterior q(w)
        TR->>ENC: encode_weights(sample from q(w))
        ENC-->>M: bitstream_weights
        ENC->>ENC: encode data samples
        -- later --
        DEC-->>M: decode weights
        DEC->>TR.model: set_weights
        DEC->>TR.model: forward(coords, pos_encodings)
        TR.model-->>DEC: predictions
        DEC-->>EVAL: form reconstructed data
        EVAL->>original data: compare
        EVAL-->>M: report metrics
    end
",
    "Anything UNCLEAR": "Details needed on the exact structure of the hierarchical prior (number of groups, dependencies), and the precise Bayesian coding implementation (e.g., the specific entropy coding algorithm or library used). Clarification is also needed on the upsampling CNN architecture parameters and the fixed hyperparameters for the β schedule."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "scipy==1.7.0",
        "pybinding==1.0",
        "bitstream=0.1"  // assuming a custom or open-source Bayesian coding library
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class; responsible for loading and batching datasets across image, audio, video, and 3D protein data; manages dataset formats and patch extraction, coordinate and label generation. Implements load_data() method to return dataset objects required downstream."
        ],
        [
            "model.py",
            "Defines INR network class (INRModel) based on SIREN architecture with Fourier features, positional encodings, and patch handling. Contains forward() method accepting coordinates, positional encodings, and model parameters, producing output values. Also includes methods to initialize and configure the network with hyperparameters specified by config."
        ],
        [
            "variational.py",
            "Implements VariationalDistribution class for mean-field Gaussian q(w); supports initialization, sampling, KL divergence calculation, and parameter updating. Manages variational parameters (μ, ρ) for the weights, including A matrices for linear reparametrization and hierarchical variables when enabled."
        ],
        [
            "hierarchical_patch.py",
            "Contains HierarchicalPatchModel class; manages global, group, and patch-level parameters, including methods for hierarchical inference, updating global parameters, and deriving local patch weights. Supports dependency modeling for patch correlation and dependency spreading in the hierarchical prior."
        ],
        [
            "trainer.py",
            "Defines Trainer class; orchestrates training procedure including gradient steps on the β-ELBO loss, beta adjustment heuristic, A matrices and positional encoding updates, and hierarchical prior updates for patch data. Uses datasets, model, variational, and hierarchical models. Implements train() and adjust_beta() methods."
        ],
        [
            "coding.py",
            "Provides Encoder and Decoder classes; implement Bayesian coding schemes—encoding q(w) posterior, encoding data residuals, applying permutations/block structures, and reconstructing weights. Interfaces to entropy coding libraries or custom implementations, wrapping the coding process with permutation strategies and block encoding based on the approximation described."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class; runs reconstruction on test data, computes rate-distortion metrics such as PSNR and RMSD, and generates plots for RD curves. Implements evaluate_reconstruction() to compare original versus reconstructed data."
        ],
        [
            "main.py",
            "Entry script that initializes configuration, invokes DatasetLoader, builds INR, variational, and hierarchical models, runs training, performs inference and compression, then evaluates and plots results. Coordinates process flow, handles argument parsing, and manages experiment lifecycle."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "variational.py",
        "hierarchical_patch.py",
        "trainer.py",
        "coding.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... // Placeholders, as communication APIs are minimal. Core functions are internal methods; no front-end API needed unless extending to interface with a user dashboard.",
    "Shared Knowledge": "Consistent definitions of dataset formats, Fourier features, positionalEncoding CNN parameters, hyperparameters for β tuning, and Bayesian coding schemes. Utilities for permutation, patch construction, and prior/posterior updates are shared among classes.",
    "Anything UNCLEAR": "Clarification needed on the specific entropy coding implementation details and whether to support online learning or batch inference. Also, more detail on the exact CNN architecture for positional encodings (Appendix B.1) is desired."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0001        # Initial learning rate for optimizer
  batch_size: 50               # For image patches: number of patches per batch
  epochs: 550                  # Total training epochs for CIFAR-10
  beta_initial: 0.3            # Starting value for beta in β-ELBO, adaptively tuned
  beta_adjust_step: 0.5        # Step size for adjusting beta during training
  beta_min: 0.0                # Minimum beta value
  beta_max: 1.0                # Maximum beta value
  beta_target_bpp: 0.3         # Target bits per pixel for target rate control
  optimizer: Adam              # Optimizer type
  optimizer_params:
    lr: 0.0001                 # Learning rate, match initial in training
    betas: [0.9, 0.999]       # Adam beta parameters
    weight_decay: 1e-5        # Regularization weight
  seed: 42                     # Random seed for reproducibility

datasets:
  cifar10:
    data_dir: "path/to/cifar10"  # Path to CIFAR-10 dataset
    image_size: 32,32            # Full image size
    patch_size: 64,64            # Patch size when using high-res patches
    patches_in_group: 6          # Number of patches in the middle level group
  kodak:
    data_dir: "path/to/kodak"    # Path to Kodak images
    image_size: 768, 512        # Full resolution images
    patch_size: 128, 128        # Target patch size
    patches_in_group: 16        # Number of patches at middle level
  libri_speech:
    data_dir: "path/to/librispeech"  # Path to LibriSpeech dataset
    sample_rate: 16000            # Audio sample rate
    chunk_duration_sec: 1.0       # Duration of each audio chunk
    patches_in_group: 12          # Number of patches from audio chunks
  proteins:
    data_dir: "path/to/proteins"  # Path to protein structure data
    structure_format: "voxel"     # Data format (voxels or point cloud)
    voxel_resolution: 64            # Resolution for voxel grid
    patches_in_group: 4             # Number of patches or segments

model:
  type: INR_SIREN
  layers: 4                     # Number of hidden layers
  hidden_units: 32              # Units per layer
  activation: sine              # Activation function (SIREN)
  fourier_features_dim: 16      # Fourier features for coordinate encoding
  positional_encoding_dim: 128  # Dimensionality for positional encodings (h_z)
  patch_encoding_dim: 50        # Dimensionality of patch positional encodings
  input_coord_dim: 2            # 2D for images; 1D or 3D for audio, proteins
  output_dim: 3                 # For RGB; 1 for grayscale/audio; 3 for proteins rendered
  hierarchy_levels: 3           # For high-res data: global, group, patch levels

optimization:
  learning_rate: 0.0001
  optimizer: Adam
  optimizer_params:
    betas: [0.9, 0.999]
    weight_decay: 1e-5
  total_epochs: 550
  beta_initial: 0.3
  beta_adjust_step: 0.5

bayesian_coding:
  coding_scheme: BitsBackAStar  # Approximate Bayesian coding implementation
  permutation_strategy: patch-wise or block-wise  # For spreading KL evenly
  bits_per_posterior_sample: as in paper (approximate KL times log2(e))
  fudge_factor: 1e-6            # Small regularization for numerical stability

experiment:
  seed: 42                     # Random seed for reproducibility
  experiment_name: "reproduce_RECOMBINER"

unknowns:
  - Exact CNN architecture parameters for positional encoding CNN (Appendix B.1)
  - Precise permutation schemes for hierarchical patch modeling
  - Specific implementation of Bayesian entropy coding (e.g., codebook, bits-back details)
```

---

**Note:** Full configuration is available in `planning_config.yaml`
