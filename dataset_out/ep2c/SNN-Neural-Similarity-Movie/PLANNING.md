# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Below is a comprehensive, detailed plan for reproducing the experiments and methodologies described in the paper "Long-Range Feedback Spiking Network Captures Dynamic and Static Representations of the Visual Cortex under Movie Stimuli" by Huang et al. The plan covers architectural design, training procedures, evaluation metrics, datasets, experimental manipulations, hyperparameters, and implementation considerations, ensuring fidelity to the original work and facilitating later coding.

---

## 1. Model Architecture and Biological Plausibility Principles

**Primary Goal:** Develop a biologically inspired deep spiking neural network (SNN) with long-range feedback connections, mimicking cortical hierarchies and feedback mechanisms.

### a. Core Principles:
- Incorporate *top-down feedback* between cortical region simulations.
- Embed *spiking neuron models*, specifically Leaky Integrate-and-Fire (LIF) neurons, with surrogate gradient mechanisms for training.
- Mimic known cortical architecture with multiple regions—specifically, six regions as per the paper.

### b. Cortical Regions and Connection Topology:
- Areas: VISp, VISl, VISrl, VISal, VISpm, VISam.
- Connectivities:
  - Feedforward: hierarchy from lower to higher visual areas.
  - Feedback: long-range top-down connections spanning wide cortical regions, mimicking corticocortical feedback pathways.
  - Lateral connections are less emphasized here but can be included within regions if needed.
- Construction:
  - Each cortical region is modeled as a layered module comprising convolutional layers with residual blocks.
  - Long-range feedback includes recurrent loops that span several regions, with delays or transfer times modeled appropriately.

### c. Layer Structure:
- Use residual blocks based on SEW-ResNet architecture, extended to spiking neurons.
- Each stage: convolution → batch normalization → spiking neurons (SN).
- Incorporate feedback connections into the residual/conv chain as recurrent feedback modules.

### d. Spike Model:
- Use LIF neurons:
  - State variables: membrane potential V.
  - Dynamics governed by the provided equation:
    - \( H_t = V_{t-1} + \frac{1}{\tau} (X_t - (V_{t-1} - V_{reset})) \)
    - Spike output \( S_t = \Theta(H_t - V_{thresh}) \)
    - Reset to \( V_{reset} \) after spike.
- Surrogate gradient: use inverse tangent (arctangent) approximation for backpropagation through the non-differentiable threshold.
- Spiking neurons process dynamic (time-dependent) spikes; their architecture should allow temporal integration.

### e. Feedback:
- Implement a *long-range feedback module*: recurrent connections that feed higher-level representations back into earlier layers/regions with appropriate delays.
- These modules should process the *spike sequences over temporal windows*.
- Feedback connections modulate the membrane potential or synaptic inputs of lower layers/neuron populations periodically.

---

## 2. Data Handling and Preprocessing

### a. Dataset:
- Use the Allen Brain Observatory Visual Coding dataset:
  - Neural responses (spike trains) recorded from 6 cortical regions.
- Movie stimuli:
  - Movie1: 30s, repeated 20 trials, frame rate 30Hz.
  - Movie2: 120s, repeated 10 trials.
- Preprocessing neural data:
  - Compute PSTH: sum spikes per frame → average over trials.
  - Exclude neurons with firing rate < 0.5 spikes/sec.
  - Result: population response matrices per cortical area, with responses aligned to movie frames.

### b. Visual Stimuli for Model Input:
- For pretraining:
  - **UCF101**: clip of 16 frames, with each frame input at one time step; total simulation time: 16 steps.
  - **ImageNet**: static images; each image input 4 times (as in training) to match the temporal input pattern.
- For neural similarity:
  - Feed the same movie frames into the network at the same frame rate and order; ensure sequence continuity.
- Data augmentation/manipulation:
  - For experiments: shuffle frames, replace static frames with noise images, or permute temporally.

### c. Static/Natural Noise Images:
- For static image manipulation: generate Gaussian noise images matching the noise characteristics used in the paper.
- For shuffled frames: divide movies into windows, shuffle within windows.
- For noise replacement: replace selected frames with noise images, select proportion based on experiments.

---

## 3. Training Procedures

### a. Pretraining:
- Tasks:
  - Action recognition on UCF101.
  - Object recognition on ImageNet.
- Architecture:
  - Use the residual-based spiking backbone as the feature extractor.
  - Set simulation time T:
    - UCF101: T=16 steps, matching frame count.
    - ImageNet: T=4 steps, replicating images 4 times.
- Objective:
  - Cross-entropy loss on class labels.
- Optimization:
  - Use surrogate gradient-based optimizer (e.g., Adam) on spike-based activations.
  - Initialize weights, train for 320 epochs (or as needed for convergence).
  - Learning rate, decay, and momentum:
    - Initial learning rate: 0.1.
    - Use step decay or cyclic schedule.
    - Surrogate gradient parameters as specified.
    
### b. Feedback Training:
- No explicit training for feedback modules—these are integrated recurrent mechanisms, trained together with the backbone via end-to-end surrogate gradient optimization.
- Feedback modules:
  - Process representations from higher regions, projecting back via learned weights.
  - During training, compute loss on final classification, propagate errors through feedback connections.
  
### c. Fine-tuning:
- Once pretrained on large datasets, optionally fine-tune on the neural data:
  - Freeze early layers or both, depending on experimental design.
  - Use neural responses as targets, optimizing the model to maximize representational similarity.

### d. Hyperparameters:
- Membrane time constant \(\tau=2\).
- Threshold \(V_{thresh}=1\), reset \(V_{reset}=0\).
- Surrogate gradient "inverse tangent".
- Learning rate schedule, weight decay, batch size matching hardware capability.
- Feedback delay/strength tuning (if needed).

---

## 4. Evaluation Metrics and Analysis

### a. Representational Similarity:
- Use **Time-Series Representational Similarity Analysis (TSRSA)**:
  - Extract population response matrices over time.
  - For each layer and cortical region:
    - Compute similarity vectors:
      - Calculate Pearson correlation between responses at time t and t+p (\(s_{t,p}\)).
    - Concatenate across t, obtain a similarity vector.
  - Use Spearman correlation between model and neural similarity vectors as the similarity score.
  
### b. Neural Ceiling:
- Calculate neural ceilings:
  - Split the neural responses into two halves over trials.
  - Compute RSA similarity between halves as ceiling.
  - Use the ceiling for normalization of model scores.

### c. Static and Dynamic Manipulations:
- Dynamic:
  - Shuffle movie frames within windows; evaluate how similarity declines.
- Static:
  - Replace some frames with noise; evaluate static texture encoding.
- Measure effect of manipulation on similarity scores across models and conditions.

### d. Regression Analysis:
- Fit linear models:
  - Response: neural similarity responses.
  - Predictors: model responses.
- Use \(R^2\) or alternative goodness-of-fit metrics.
- Perform cross-validation for robustness.

---

## 5. Experimental Manipulations and Physiological Consistency Tests

### a. Long-Range Feedback:
- Implement feedback as recurrent modulatory connections.
- Vary feedback window delays, strengths, or connection patterns to test robustness.
  
### b. Ablations:
- Use different network variants as baselines:
  - No feedback: pure feedforward.
  - Feedback only: without spike dynamics.
  - Spiking only: static feedback.
- Evaluate each variant with TSRSA to measure their cortical similarity.

### c. Noise and Disruption:
- Add controlled noise to the network inputs or internal states.
- Disrupt temporal order (shuffle frames).
- Replace frames with noise images.
- Quantify effect on similarity scores and model robustness.

---

## 6. Reproducibility and Hyperparameter Search

### a. Random Seeds:
- Run multiple (e.g., 10) independent runs to assess variability.
- Record mean and confidence intervals.

### b. Hyperparameters:
- Fine-tune parameters like learning rate, feedback strength, neuron thresholds, and temporal constants based on validation performance (similarity metrics).

### c. Computing Resources:
- Confirm GPU/TPU availability:
  - 8-GPU (NVIDIA V100) is mentioned; use batch sizes accordingly.
  - Use efficient surrogate gradient implementations (e.g., SpikingJelly or custom PyTorch extensions).

### d. Software Environment:
- Use PyTorch or compatible deep learning framework supporting custom neuron models.
- Implement surrogate gradient method as per the descriptions.

---

## 7. Summary Roadmap

**Phase 1: Data Preparation**
- Extract neural response matrices.
- Prepare stimuli inputs for the model (movie frames).
- Generate manipulated stimuli: shuffled, noise replacement.

**Phase 2: Model Construction**
- Implement residual conv blocks with SNN layers.
- Embed feedback mechanisms.
- Design feedback modules as recurrent loops.

**Phase 3: Pretraining**
- Train on UCF101 and ImageNet with respective protocols.
- Save pretrained weights.

**Phase 4: Neural Response Simulation**
- Feed the moving stimuli into pretrained model.
- Extract population responses over time layers/regions.

**Phase 5: Similarity Analysis**
- Compute TSRSA vectors.
- Calculate Spearman similarity scores to cortical data.
- Perform manipulations and ablations.

**Phase 6: Results and Interpretation**
- Compare models.
- Visualize dynamic/static response effects.
- Cross-validate with neural ceiling benchmarks.

---

This outline provides a full, actionable blueprint respecting the paper's details, experimental setups, and biological motivations. It guides the implementation of model architecture, training, data handling, and evaluation, even if some details (e.g., exact residual block parameters) require adjustments from pilot experiments, which should follow the general principles here.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular deep spiking neural network with feedback connections using PyTorch and SpikingJelly. The model will have separate classes for the backbone residual modules with spiking neurons, feedback modules, and recurrent dynamics. Pretraining on UCF101 and ImageNet will be performed with surrogate gradient training, then the neural responses will be simulated and responses extracted. Similarity scores will be calculated using TSRSA and regression analysis. Data manipulations (shuffling frames, noise replacements) will be implemented as preprocessing functions. All components will be integrated into a main script that orchestrates data loading, model setup, training, manipulation experiments, and evaluation.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "train.py",
        "evaluation.py",
        "manipulations.py",
        "feedback_module.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(path: str, mode: str)\n        +load_data() -> Tuple[Tensor, Tensor]\n        +get_neural_responses() -> np.ndarray\n    }\n    class SpikingResNet {\n        +__init__(hyperparams: dict)\n        +forward(inputs: Tensor, feedback_input: Optional[Tensor]=None) -> Tensor\n        +extract_features() -> list[Tensor]\n    }\n    class FeedbackModule {\n        +__init__(params: dict)\n        +process(higher_level_response: Tensor) -> Tensor\n    }\n    class SurrogateGradient {\n        +arc Tangent surrogate(x: Tensor) -> Tensor\n    }\n    class Evaluation {\n        +compute_TSRSA(neural_data: np.ndarray, model_responses: np.ndarray) -> float\n        +compute_neural_ceiling() -> float\n        +regression_score(neural_data: np.ndarray, model_responses: np.ndarray) -> float\n    }\n    class Manipulation {\n        +shuffle_frames(movie: np.ndarray, window_size: int) -> np.ndarray\n        +replace_frames_with_noise(movie: np.ndarray, ratio: float) -> np.ndarray\n    }\n    class DataManager {\n        +load_neural_data(path: str) -> Tuple[np.ndarray, List[str]]\n        +load_stimuli() -> List[np.ndarray]\n        +save_responses() -> None\n    }\n    Main --> DatasetLoader\n    Main --> Train\n    Main --> Evaluation\n    Main --> Manipulation\n    Main --> Model\n    Model --> FeedbackModule\n    Model --> SurrogateGradient\n    main.py orchestrates calls among these classes following the process flow: load data, initialize model, train on datasets, simulate stimuli, manipulate input as needed, evaluate similarity scores, and save/visualize results.\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as SpikingResNet\n    participant FB as FeedbackModule\n    participant SG as SurrogateGradient\n    participant EV as Evaluation\n    participant MP as Manipulation\n    M->>DL: load_neural_data()\n    DL-->>M: neural_responses, cortical_names\n    M->>MD: __init__(hyperparams)\n    Note right of MD: Initialize backbone with residual conv and LIF neurons\n    MD->>MD: train on UCF101/ImageNet with surrogate gradient\n    MD-->>M: pretrained_weights\n    M->>DL: load stimuli movies\n    DL-->>M: stimuli_list\n    loop For each stimulus condition\n        M->>MD: forward(inputs over time)\n        MD-->>M: feature responses\n        M->>EV: evaluate responses vs neural responses using TSRSA\n        EV-->>M: similarity scores\n        M->>Manipulation: apply shuffle/replacement\n        Manipulation-->>DL: manipulated_movies\n        M->>MD: forward(manipulated movies)\n        MD-->>M: responses to manipulated movies\n        M->>EV: evaluate manipulated responses\n        EV-->>M: manipulated similarity scores\n    end\n    M->>Evaluation: compute neural ceiling\n    Evaluation-->>M: ceiling values\n    M->>Main: save and plot results\n"
    ,
    "Anything UNCLEAR": "Details about specific hyperparameters (e.g., neuron thresholds, feedback connection weights), exact network residual block configurations, surrogate gradient parameters, and the preprocessing pipeline for neural data need clarification. Additionally, confirmation on whether code modules like feedback connections are to be integrated end-to-end or fine-tuned separately."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "spikingjelly==0.2.1",
        "scikit-learn==0.24.2",
        "matplotlib==3.4.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Handles loading neural responses and stimuli data, implements functions to load Allen Brain Observatory responses, preprocess PSTH, and load stimuli movies. It depends on numpy, pandas, and cv2 for image processing."
        ],
        [
            "model.py",
            "Defines the residual conv-based spiking neural network architecture with feedback modules. Implements classes for SpikingResNet, feedback modules, surrogate gradient functions, and the recurrent dynamics. Uses torch and spikingjelly for neuron models."
        ],
        [
            "trainer.py",
            "Contains logic for training the pretrained models on UCF101 and ImageNet datasets using surrogate gradients. Implements training loops, loss functions, learning rate scheduling, and model checkpointing."
        ],
        [
            "evaluation.py",
            "Provides TSRSA similarity computation, neural ceiling estimation, regression analyses, and static/dynamic stimulus manipulation functions. Uses scikit-learn for correlation measures and matplotlib for visualization."
        ],
        [
            "manipulations.py",
            "Implements functions to shuffle frames within windows, replace frames with noise, and apply static noise images. Depends on numpy and OpenCV for image processing."
        ],
        [
            "feedback_module.py",
            "Implements feedback recurrent modules that process higher-region responses and project back to lower layers. Uses torch modules with flexible weight matrices; integrated into the main model pipeline."
        ],
        [
            "main.py",
            "The orchestrator script that initializes datasets, models, trainers, performs pretraining, loads neural responses, runs stimulus manipulations, carries out evaluations, and saves/visualizes results."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "manipulations.py",
        "feedback_module.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0  # No API needed; system is core experimental pipeline and data processing.",
    "Shared Knowledge": "Common utility functions include surrogate gradient implementation, response extraction, and response normalization. Neural response data is shared across evaluation modules for TSRSA and regression.",
    "Anything UNCLEAR": "Exact parameters for feedback connection weights, delay times, and surrogate gradient settings require clarification. Also, details on handling the neural ceiling estimation during evaluation are needed."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.1
  batch_size: 32
  epochs: 320
  optimizer: Adam
  surrogate_gradient: inverse_tangent
  weight_decay: 1e-5
  initial_lr: 0.1
  lr_decay_steps: 100
  lr_decay_rate: 0.1
  simulation_time: 16  # for UCF101 training
  feedback_strength: 1.0
  feedback_delay: 2  # arbitrary delay in feedback module
  membrane_tau: 2
  neuron_threshold: 1.0
  reset_voltage: 0.0
  surrogate_alpha: 1.0  # for inverse tangent surrogate
model:
  backbone: ResidualConvSPikeNet
  residual_blocks: 3  # number of residual stages
  feedback_module: FeedbackModule
  feedback_connection_type: recurrent
  input_channels: 3
  feature_channels: 64
  num_regions: 6  # cortical regions
  layers_per_region: [2, 2, 2]
dataset:
  ucf101:
    video_length: 16  # num frames per clip
    frame_rate: 30  # Hz
    input_size: [224, 224]  # image resize
    train_split: 0.8
  imagenet:
    image_size: [224, 224]
    input_repeats: 4
    train_split: 0.8
evaluation:
  consultation_methods:
    - TSRSA
    - NeuralCeiling
    - RegressionAnalysis
manipulations:
  shuffle_frames:
    window_sizes: [5, 10, 20, 50]  # frames per window
  replace_with_noise:
    ratios: [0.25, 0.5, 0.75, 1.0]
    noise_type: Gaussian
  static_images:
    source: "Gaussian noise images"
    image_size: [224, 224]
feedback:
  connection_weights: learned
  feedback_delay: 2  # ms
  feedback_strength: 1.0
  feedback_module: FeedbackModule
system:
  max_epochs: 320
  device: cuda  # or 'cpu'
  seed: 42
```

---

**Note:** Full configuration is available in `planning_config.yaml`
