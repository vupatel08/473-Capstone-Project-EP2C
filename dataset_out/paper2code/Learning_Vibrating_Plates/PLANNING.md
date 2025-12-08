# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a **comprehensive, step-by-step plan** for reproducing the experiments and methodologies described in the paper "Learning to Predict Structural Vibrations." This roadmap covers all critical aspects, including architecture design, data preparation, training procedures, hyperparameters, evaluation metrics, and experimental setups, based solely on the detailed information from the provided extract.

---

## 1. Understanding the Core Methodology

### Goal:
Predict the frequency response (vibration patterns) of flexible plate geometries under harmonic excitation, for arbitrary excitation frequencies, using neural operator architectures.

### Key Components:
- **Inputs**:
  - Geometry: embedding shape via shape encoding (e.g., implicit functions, mesh-based encodings).
  - Scalar properties: material (~ Young's modulus, Poisson ratio, density, damping loss factor), boundary conditions (rotational stiffness, boundary shape), and loading parameters (load position, magnitude).
  - Frequency query: scalar excitation frequency (f) at which the response is predicted.
  
- **Outputs**:
  - Spatial vibration response pattern: velocity field over the plate surface at frequency `f`.

### Main Approach:
- **Encoder-decoder architecture**:
  - Geometry encoded via a dedicated geometry encoder (`Φ`).
  - Scalar material and boundary properties incorporated into the encoding.
  - The frequency query `f` is embedded or concatenated as input.
  - The decoder (`Ψ`) produces:
    - Either a velocity field for the given `f`.
    - Or directly the frequency response function (e.g., power spectrum, vibration amplitude).

- **Operator learning**: trained to approximate the nonlinear operator mapping `(geometry, properties, frequency) → response`.

### Additional innovations:
- Use of **implicit shape representations** (e.g., signed distance functions or mesh-based encodings).
- The model supports *querying* responses at arbitrary frequencies, not only the frequencies seen during training.

---

## 2. Data Generation and Dataset

### Datasets:
- **V-5000** and **G-5000**:
  - 6000 samples each.
  - Variations in geometry (lines, ellipses, complex shapes).
  - Material parameters: Young's modulus, Poisson ratio, density, damping, shear modulus, boundary damping stiffness.
  - Geometry parameters: length, width, thickness.
  - Boundary and loading parameters: boundary stiffness, load location.
  - Velocity fields `V(f)` and response `F(f)` computed over frequencies `[1, 300] Hz`.
  - Discretization:
    - Mesh: e.g., `181 × 121` for G-5000, `121 × 81` for V-5000.
    - Finite element solution based on Mindlin plate theory.
    - Resolution sufficient for bending wave length.
  - Computational cost per sample ≈ 2 min 19 sec on a high-performance CPU.

### Steps:
- **Finite element simulation**:
  - Implement or adapt a FEM solver based on the shell formulation (see equations).
  - Use triangular linear elements with 3 nodes.
  - Discretize the geometry to capture relevant wavelengths (using the provided formula for λ_B).
- **Parameter variation**:
  - Generate geometry variations (lines, ellipses, complex shapes) with different beading patterns.
  - Vary material properties within specified ranges.
  - Vary boundary conditions (boundary stiffness, rotational stiffness).
  - Vary load positions and magnitudes.
- **Frequency responses**:
  - For each sample, simulate the response over `f ∈ [1, 300] Hz`.
- **Dataset split**:
  - 1000 samples reserved for testing.
  - Remaining 5000 for training.
- **Noise or variability**:
  - Optional: add small variations or noise to parameters for robustness testing.
- **Data storage**:
  - Save shape encodings (implicit functions or meshes), properties, load conditions, velocity fields, response functions.
  
---

## 3. Model Architectures

### 3.1. Geometry Encoder (`Φ`)
- **Options**:
  - Implicit SDF-based encoder (`φ`): process shape encoded as a signed distance function or occupancy grid.
  - Mesh-based encoder: process mesh vertices and elements via MLPs or graph neural networks.
  - Vision backbone (CNN or Vision Transformer) applied to rasterized shape images or pixel-based shape encodings.
- **Implementation details**:
  - Use 3D signed distance functions (SDF) or mesh discretization.
  - Map raw shape data into latent shape embedding vectors (`z_shape`).

### 3.2. Scalar Property Encoder (`m`)
- Embeddings or simple normalization of material and boundary parameters.
- Concatenate or embed into spatial features as per design.

### 3.3. Frequency Embedding
- Embed frequency `f` using a positional encoding (e.g., sinusoidal embedding or Fourier features), consistent with implicit neural operators.

### 3.4. Decoder (`Ψ`)
- **Type 1: Velocity field prediction (`V(f)`)**:
  - Input: latent shape code, scalar properties, frequency embedding.
  - Architecture: a U-Net or similar CNN-based implicit decoder that outputs a velocity map over the discretized spatial domain.
- **Type 2: Response function (`F(f)`)**:
  - Input: latent codes + frequency encoding.
  - Architecture: an MLP operating on the embedded features to directly output scalar response (e.g., power spectrum).

### 3.5. Variants
- **FQO-UNet**: CNN-based with multi-scale features.
- **FQO-ViT**: Transformer-based spatial encoder.
- **Grid-RN18**: ResNet-based encoder on grid representations.
- **FQO-RN18**: CNN + Fourier neural operators and residual connections.

---

## 4. Training Procedures

### Loss functions:
- **Velocity response (if predicting `V(f)`):**
  - Use combined loss:
    - Velocity field loss: e.g., MSE over the predicted velocity map.
    - Response function loss: MSE or spectral distance over `F(f)`.
  - Balance with hyperparameters, e.g., a weighted sum:
    \[
    \text{Loss} = \alpha \cdot \text{MSE}_V + (1-\alpha) \cdot \text{MSE}_F
    \]
  - The paper suggests experimenting with `α`, e.g., 0.25, 0.5, 0.9.

### Hyperparameters:
- Learning rate: e.g., 1e-3 with scheduler.
- Batch size: e.g., 16 samples.
- Number of epochs: until convergence (~ hundreds of epochs).
- Optimizer: Adam or AdamW.

### Training details:
- Use a validation set (e.g., 500 samples from training) for early stopping.
- Implement gradient clipping if necessary.
- Support training with:
  - Varying number of frequency points per sample (see Table 11).
  - Fewer frequencies per shape with dataset augmentation to simulate generalization.

### Training strategies:
- Meta-learning or curriculum over frequencies (`f`).
- Ablation: train with only velocity loss vs. response function loss.
- Vary model "width" (channels) for architecture tuning (see Table 9).

---

## 5. Evaluation Metrics

### Primary metrics:
- **Evaluation on response functions**:
  - **EMD** (Energy Distance): with normalized power spectra (see Eq. in Appendix B).
  - **Peak detection accuracy**:
    - Compare the number and location of peaks.
    - Use Hungarian matching to evaluate shift errors.
- **Per-frequency response error**:
  - MSE or spectral Wasserstein distance (`$\mathcal{E}_{F}$`).
- **Peak error**:
  - Measure shift or misspecification of resonance peaks (`$\mathcal{E}_{PEAKS}$`).

### Additional metrics:
- **Prediction quality at trained frequencies** vs. **arbitrary unseen frequencies**.
- **Speed of inference**:
  - Benchmark models on GPU/CPU with batch sizes (see Table 7).
- **Data efficiency**:
  - Training size ablation to assess how models perform with less data (see Table 10).

---

## 6. Experimental Setup and Hyperparameter Tuning

- **Dataset splits**:
  - Use the pre-defined 5000 training and 1000 test samples.
  - For ablation studies:
    - Vary `α` in the combined loss.
    - Vary data size (e.g., 10%, 25%, 50%, 75%, full).
    - Vary number of frequency points per shape (Table 11).
    - Vary model width (channels) (Table 9).

- **Model evaluation**:
  - Measure metrics on held-out test samples.
  - Perform visualizations as in Figures 11 and 12.

- **Reproducibility measures**:
  - Report seed initialization.
  - Use identical hardware/software (e.g., same GPU, FEM solver settings).
  - Use the same discretizations and FEM parameters for data generation.

---

## 7. Additional Considerations
- **Physics-informed constraints**:
  - Consider integrating physics-based regularization or enforcing boundary conditions during training.
- **Data augmentation**:
  - Shape, boundary, and material variation can improve generalization.
- **Code modularity**:
  - Separate shape encoding, property encoding, frequency embedding, and decoder into reusable modules.
- **Documentation**:
  - Log hyperparameters, training curves, and failure modes for transparency.

---

## Summary Roadmap:
1. **Data Simulation**:
   - Implement the finite element solver based on plate theory equations.
   - Generate 6000 samples with shape, material, boundary, load, and frequency variations.
2. **Preprocessing**:
   - Encode geometries into implicit functions or mesh-based data.
   - Normalize scalar properties.
   - Prepare frequency embeddings.
3. **Model Architecture**:
   - Design geometry encoder (`Φ`) (e.g., implicit CNN, mesh graph network).
   - Incorporate scalar properties.
   - Embed frequency query (`f`) via Fourier features.
   - Build decoder (`Ψ`) (e.g., UNet, Fourier neural operator, residual CNN).
4. **Training**:
   - Use combined loss (`α` weight on velocity/response vs. spectral response).
   - Train for hundreds of epochs with early stopping.
   - Support variable frequency sampling and dataset sizes.
5. **Evaluation & Visualization**:
   - Use Wasserstein distance, peak matching/error metrics.
   - Visualize velocity fields and peak correspondences.
6. **Ablation & Hyperparameter Studies**:
   - Vary dataset size, model width, frequency sampling, loss weights.
7. **Reporting & Reproducibility**:
   - Document hyperparameters, code architecture, data generation specifics.
   - Use consistent FEM and neural network implementations.

---

This detailed plan should serve as a solid foundation for coding the architecture, generating the data, training models, and reproducing the experiments as described in the paper.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular, open-source software system for data generation, model training, and evaluation of neural operator models for structural vibration prediction. The system will include components for FEM-based data simulation (using a FEM library, e.g., FEniCS or PyMesh), geometry encoding (implicit function representation via signed distance functions), a neural operator architecture (e.g., UNet or Fourier Neural Operator implemented in PyTorch or JAX), and training scripts. The main workflow involves generating the dataset with the FEM solver, encoding geometries, training the neural network with combined velocity and response losses, and evaluating using spectral Wasserstein and peak-error metrics.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "visualization.py",
        "utils.py"
    ],
    "Data structures and interfaces": "```mermaid\ngraph TD\n    classShapeEncodings ::= class Diagram\n    class GeometryEncoder {\n        +encode_shape(shape: ShapeData) -> Tensor\n    }\n    class MaterialProperties {\n        +to_tensor() -> Tensor\n    }\n    class FrequencyEmbedding {\n        +embed(f: float) -> Tensor\n    }\n    class ResponseDecoder {\n        +predict_velocity(geometry_embedding: Tensor, properties: Tensor, freq: float) -> VelocityField\n        +predict_response(geometry_embedding: Tensor, properties: Tensor, freq: float) -> float\n    }\n    class Dataset {\n        +load() -> Tuple[ShapeData, MaterialProperties, List[Freq], List[VelocityField], List[Response]]\n    }\n    class MainWorkflow {\n        +run() -> None\n    }\n\n    MainWorkflow --> Dataset\n    MainWorkflow --> Model\n    MainWorkflow --> Trainer\n    MainWorkflow --> Evaluation\n    Dataset --> GeometryEncoder\n    Dataset --> MaterialProperties\n    Dataset --> FrequencyEmbedding\n    Model --> ResponseDecoder\n    GeometryEncoder --> ShapeData\n    ResponseDecoder --> VelocityField\n    ResponseDecoder --> float\n```\n\n- `ShapeData`: class containing raw shape info (signed distance function or mesh vertices)\n- `VelocityField`: class or tensor array representing spatial velocity at a frequency\n- `Response`: class or float representing scalar frequency response\n- `Dataset`: loads and preprocesses all data; returns tensors for geometry, material, response over frequencies\n- encoders and decoders: convert raw data into latent representations and produce the responses\n- `MainWorkflow`: orchestrates dataset loading, model training, validation, and testing with explicit method signatures\n",
    "Program call flow": "```sequenceDiagram\n    participant MP as Main.py\n    participant DL as dataset_loader.py\n    participant MD as model.py\n    participant TR as trainer.py\n    participant EV as evaluation.py\n    MP->>DL: load dataset\n    DL-->>MP: returns dataset\n    MP->>MD: initialize model with hyperparameters\n    MP->>TR: train(model, dataset)\n    TR->>MD: forward(x)\n    MD-->>TR: velocity prediction / response\n    TR-->>MP: training complete\n    MP->>EV: evaluate(model, test_dataset)\n    EV->>MD: forward response at sample frequencies\n    MD-->>EV: metrics (EMD, peak shifts)\n    EV-->>MP: report metrics\n    MP->>visualization.py: plot responses and peaks\n```",
    "Anything UNCLEAR": "Clarification needed on the exact mesh/shape input format (implicit SDF or mesh vertices), and any domain-specific hyperparameters, especially FEM solver details or specific kernel parameters for Fourier neural operators."
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
        "fenics==2019.1.0",  # FEM simulation library
        "pyvista==3.0.4"    # Mesh and geometry processing
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Handles loading raw shape data, material properties, boundary conditions, FEM simulation outputs, and prepares them for training. Depends on FEM solver implementation details; interfaces with mesh/shape encoding classes. Requires functions to load shape, material, load parameters, run FEM simulations, and store response over frequencies."
        ],
        [
            "model.py",
            "Defines neural operator models (e.g., UNet, Fourier Neural Operator) with modules for geometry encoding (cell modules for implicit shape representation), frequency embedding, and response prediction decoders. Imports PyTorch modules."
        ],
        [
            "train.py",
            "Orchestrates training of models using combined velocity and response function loss. Uses datasets from dataset_loader.py. Depends on model.py for model architecture and evaluation.py for metrics. Implements training loop, loss calculation, validation, and checkpointing."
        ],
        [
            "evaluation.py",
            "Contains functions/classes to calculate spectral Wasserstein metrics, peak detection errors, and visualizations. Needs response predictions, ground-truth responses, and peak matches; computes metrics for response similarity."
        ],
        [
            "visualization.py",
            "Provides plotting functions for response functions, velocity fields, peak alignments, and response comparisons, used post-training for qualitative assessment."
        ],
        [
            "main.py",
            "Main entry point to run dataset generation (if needed), initialize models, call training, and evaluate results. Coordinates overall flow by importing classes from other modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",  # dataset loading and FEM simulation integration
        "model.py",           # network architecture definition, shape encoding modules
        "train.py",           # training loop, loss functions, model optimization
        "evaluation.py",      # metrics and response/peak analysis
        "visualization.py",   # plotting and qualitative results
        "main.py"             # orchestrator: data loading, model training, eval, visualize
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Geometry encoding methods (implicit functions, mesh processing), FEM response calculation routines, frequency embedding techniques, combined loss function design, library APIs (PyTorch, SciPy, fenics, pyvista)."
    ,
    "Anything UNCLEAR": "Exact details of the implicit shape representation method (signed distance function or mesh voxel grid), FEM solver configuration (discretization parameters, solver tolerances), and the structure of response datasets (response tensor structures). Clarify if simplified or approximate FEM outputs are acceptable for initial prototypes."
}
[/CONTENT]

## 4. Configuration

```yaml
## config.yaml
training:
  learning_rate: 0.001  # Default, can be tuned
  batch_size: 16  # Typical batch size used for training
  epochs: 300  # Number of epochs until convergence, based on standard training runs
  early_stopping_patience: 20  # Optional early stopping to prevent overfitting
  loss_weights:
    velocity_loss_weight: 0.25  # Alpha: weight for velocity field loss
    response_loss_weight: 0.75  # (1 - alpha): weight for response function loss
  optimizer:
    type: Adam
    betas: [0.9, 0.999]
    weight_decay: 1e-4
  learning_rate_scheduler:
    type: ReduceLROnPlateau
    factor: 0.5
    patience: 10
    min_lr: 1e-6

## Data:
dataset:
  dataset_name: VibratingPlateResponses
  train_size: 5000
  test_size: 1000
  validation_size: 500
  discretization_mesh: 181x121  # for G-5000; change as needed for V-5000
  frequency_range:
    min: 1 Hz
    max: 300 Hz
  frequency_points: 300  # full response over all frequencies in the dataset

## Model:
model:
  architecture: UNet  # or Fourier Neural Operator, can be flexible
  encoder:
    type: implicit_shape_encoder
    shape_representation: signed_distance_function
  scalar_properties_dim: 7  # material, boundary, load params
  frequency_embedding: FourierFeatures
  response_decoder:
    type: velocity_field_decoder  # or direct response prediction
    output_type: velocity_map
  channels: 64  # model width (channels); can be tuned
  depth: 4  # number of levels in the UNet

## Misc:
training_strategy:
  frequency_sampling: uniformly across [1, 300] Hz
  data_variation_batches: 3 (for robustness experiments)
  dataset_augmentation: include shape and property variation
  save_checkpoints: True
  save_best_only: True
  evaluation_metrics:
    - Wasserstein_EMD
    - Peak_shift
```

---

**Note:** Full configuration is available in `planning_config.yaml`
