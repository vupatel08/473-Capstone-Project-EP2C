# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, step-by-step plan that captures the methodology, experimental setup, hyperparameters, and validation steps described in the paper "Adaptive Randomized Smoothing: Certified Adversarial Robustness for Multi-Step Defences." This plan will serve as a blueprint for faithful implementation.

---

# Overall Objective:
Implement **Adaptive Randomized Smoothing (ARS)**, extending standard randomized smoothing (RS) with input-dependent, high-dimensional, test-time adaptivity grounded in $f$-Differential Privacy theory, with certified robustness guarantees. Evaluate on image classification datasets—CIFAR-10, CelebA, ImageNet—using models (ResNet variants) and masks (U-Net) trained with the described protocols.

---

# I. Core Methodology Components

### 1. The ARS Architecture:
- **Two-step adaptive process:**
  - **Step 1:** Generate an input-dependent *mask* $w(m_1)$ via a U-Net, capturing task-relevant features spatially, post noisy input $X + z_1$.
  - **Step 2:** Apply mask $w(m_1)$ to input $X$ (element-wise $w(m_1) \odot X$), add noise $z_2$, and produce a *second noisy image*.
- **Final prediction:**
  - A base classifier $g$ (ResNet) is trained to output class predictions.
  - The outputs from Step 1 ($m_1$) and Step 2 ($m_2$) are linearly combined into an unbiased estimate $\hat{X}$, which is input to $g$.
  - The smoothed classifier predicts the class as the most probable over noisy draws.

### 2. Theoretical Guarantees:
- Use the $f$-DP formulation of Gaussian mechanisms to quantify robustness radius.
- **Adaptive composition:** leverage differential privacy composition guarantees to combine multiple steps, with no radius penalty for adaptivity as shown in Theorem 2.3.

### 3. Certification:
- Using the smoothed predictions, certify input robustness within an $L_\infty$ radius based on class probabilities ($p_+$, $p_-$) and formulas involving $\Phi^{-1}$ and the privacy parameters ($\sigma_i$, $\mu_i$).

---

# II. Implementation Details

### 1. Data Preparation:
- Datasets: CIFAR-10, CelebA, ImageNet.
- For CIFAR-10 & CelebA:
  - **Input size**:
    - CIFAR-10: $32 \times 32 \times 3$
    - CelebA: Crop & resize to $128 \times 128 \times 3$
  - **Creates larger "background" images** (e.g., 640x640, 960x960 pixels) with CIFAR/celeba images embedded along edges at random positions for scaling evaluations.
- For ImageNet:
  - Use standard 224x224 images, possibly amino scaled backgrounds (e.g., 1, 4, 16 times larger).

### 2. Model Architectures:
- **Mask Model (Step 1):** U-Net, pixel-wise output to generate mask $w(m_1)$.
  - Input: noisy image $X + z_1$, with $z_1 \sim \mathcal{N}(0, \sigma_1^2 I)$.
  - Output: single-channel mask (values in [0,1]) via sigmoid activation.
  - Hyperparameters:
    - Base channels: 32, channels_mult: [1,2,4,8].
    - Step size: 40, gamma: 0.5, momentum: 0.9.
- **Base Classifier (Step 2):** ResNet-110 (CIFAR-10), ResNet-50 (CelebA, ImageNet).
  - Input: masked and noisy images, combined via linear weights for unbiased estimate.
  - Hyperparameters:
    - Optimizer: AdamW with specified learning rate decay.
    - Loss: cross-entropy for classification.
  
### 3. Noise Parameters & Budget Splitting:
- Total privacy budget per layer: $\sigma$.
- **Variance split**:
  - $\sigma_1 = \sigma_2 = \sqrt{2} \sigma$.
  - This per-step splitting enables input-dependent adaptivity with no radius penalty.
- Hyperparameters for $\sigma$:
  - For CIFAR-10 / CelebA: 0.25, 0.5, 1.0, 1.5.
  - For ImageNet: scale accordingly.
- **Hyperparameters tuning**: 
  - End-to-end via validation, optimizing $\beta$ (noise distribution parameters) for RS.
  - Use grid search for $\beta$ around Gaussian ($\beta=2$) or near-Gaussian ($\beta=2.25$).
- **Mask weight linear combination:** for each pixel $i$, compute $c_{1,i}, c_{2,i}$ to minimize variance under the unbiased constraint.

### 4. Training:
- **Step 1 (Mask Model):**
  - End-to-end training on classification task.
  - Loss: standard cross-entropy.
  - Input: noisy images ($X + z_1$), with mask $w(m_1)$ learned end-to-end.
- **Step 2 (Classifier):**
  - Trained on the combined (masked+noise) images; additionally include the mask prediction pipeline during training.
- **Regularization & Hyperparameter tuning:**
  - Use grid search for $\sigma_i$, gamma, momentum, learning rate.
  - Stabilize mask training via early stopping or regularization (weight decay).

### 5. Test-time Procedure:
- **Input noise $z_1$, $z_2$:** drawn fresh at each sample.
- **Mask inference:**
  - Compute $w(m_1)$ with the trained UNet on noised image.
- **Adaptive image construction:**
  - $\hat{X}$ = linear combination of $m_1, m_2$ predictions, ensuring unbiased estimate.
- **Classifier prediction:**
  - Feed $\hat{X}$ to $g$ over multiple Monte Carlo noise draws.
- **Certification:**
  - Calculate class probabilities ($p_+$, $p_-$).
  - Use the formulas (e.g., Eq. 2.2, 2.4, 2.5) to compute radius bounds for $L_\infty$ robustness.

---

# III. Experimental Protocols

### Datasets:
- CIFAR-10 (standard, with 20kBG backgrounds)
- CelebA (aligned & unaligned; large resolution backgrounds change the difficulty)
- ImageNet (standard size; scale backgrounds as a multiple of original image size)

### Hyperparameter Tuning:
- Noise level $\sigma$ (e.g., 0.25, 0.5, 1.0, 1.5)
- Variance split: $\sigma_1 = \sigma_2 = \sqrt{2} \sigma$.
- Mask network hyperparameters: as above.
- $\beta$ (consumer noise distribution parameter): tuned via validation for best certified accuracy.
- Learning rates & optimizer schedule: as per appendix.

### Evaluation:
- Standard accuracy ($r=0$).
- Certified robustness: Percentage of samples with $L_\infty$ radius greater than various thresholds.
- Certification bounds: derived using class probability estimates, $f$-DP algebra, and Gaussian mechanism formulas.
- **Scalability** check: vary background size, $\sigma$, $k$, $d$.
- **Adaptivity** ablation: compare fixed variance RS vs ARS with input-dependent masking.
- **Robustness**: test on adversarially perturbed samples within bounded $L_\infty$ norms.

---

# IV. Implementation Notes & Considerations

- **Reproducibility:**
  - Fix random seeds.
  - Use established deep learning libraries (PyTorch recommended).
  - Log hyperparameters, training/validation, and certification results systematically.
- **Code structure:**
  - Data loaders for CIFAR-10, CelebA, ImageNet.
  - Model modules: Mask UNet, ResNet classifier.
  - Noise injection modules aligned with hyperparameters.
  - Certification functions based on formulas (e.g., Eq. 2.2, 2.4, 2.5).
  - Monte Carlo sampling for prediction averaging.

---

# V. Potential Missing Details & Clarifications Needed
- Exact architecture hyperparameters (number of layers, channels) for the UNet—though provided in appendix.
- Precise training schedule, early stopping criteria, batch sizes.
- How to compute class probability estimates ($p_+$, $p_-$) exactly in certification:
  - More details in algorithm Appendix B, involving Monte Carlo or analytical bounds.
- Handling of randomness at test-time: number of samples for confidence, legal thresholds.
- Distribution parameters ($\beta$) specifics for noise distribution -- whether Gaussian or near-Gaussian variants.

---

# Summary:
This plan elaborates on:
- Building the two-step mask + classifier pipeline with noise and adaptive privacy-based variance control.
- Training and hyperparameter tuning strategies.
- Monte Carlo procedures for robust certification.
- Evaluation metrics and dataset strategies.
- Theoretical underpinnings to guide hyperparameter choices and ensure correctness.

Following this roadmap will facilitate a faithful, detailed implementation aligned with the paper's claims and protocol.

---

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular system using PyTorch for model implementation, training, and evaluation. The core components include data loaders for CIFAR-10, CelebA, and ImageNet; a U-Net-based mask generator; ResNet-based classifiers; and modules for adding Gaussian noise, combining mask-weighted images, and performing Monte Carlo prediction and certification. The training pipeline will optimize the mask and classifier end-to-end, with hyperparameters like noise levels and privacy budgets tuned on validation sets. During inference, the system will generate input-dependent masks, produce noisy masked images, aggregate predictions over Monte Carlo samples, and compute robustness certificates using formulas from the paper based on class probabilities and $f$-DP bounds. Open-source libraries like PyTorch, torchvision, and NumPy will be used for deep learning, data processing, and mathematical functions. The architecture will be simple: main script orchestrates loading data, training models, and evaluating robustness; helper modules handle noise injection, mask generation, and certification calculations; hyperparameter configs are flexible via a JSON or command-line interface.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "models.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "config.json"
    ],
    "Data structures and interfaces": "```mermaid\nclassDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment() -> None\n    }\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> Tuple[train_dataset: Dataset, val_dataset: Dataset]\n    }\n    class MaskUNet {\n        +__init__(params: dict)\n        +predict_mask(noisy_input: Tensor) -> Tensor[mask_value in [0,1]]\n    }\n    class ResNetClassifier {\n        +__init__(params: dict)\n        +predict(image: Tensor) -> Tensor[class_logits]\n        +train(train_data: Dataset)\n        +evaluate(eval_data: Dataset) -> dict\n    }\n    class NoiseInjector {\n        +__init__(sigma: float)\n        +add_noise(input: Tensor) -> Tensor\n    }\n    class Predictor {\n        +__init__(classifier: ResNetClassifier, mask_model: MaskUNet, sigma1: float, sigma2: float)\n        +generate_mask(noisy_input: Tensor) -> Tensor\n        +combine_predictions(m1_preds: List[Tensor], m2_preds: List[Tensor]) -> Tensor\n        +average_over_samples(images: List[Tensor]) -> Tensor\n        +predict_and_certify(input: Tensor, n_samples: int) -> dict {\n          -- class_probabilities: dict (class: probability)\n          -- radius: float\n        }\n    }\n    class Certification {\n        +compute_radius(p_plus: float, p_minus: float, sigma: float, info: dict) -> float\n    }\n    Main --> DatasetLoader\n    Main --> MaskUNet\n    Main --> ResNetClassifier\n    Main --> NoiseInjector\n    Main --> Predictor\n    Predictor --> ResNetClassifier\n    Predictor --> MaskUNet\n    Predictor --> NoiseInjector\n    Certification --> Predictor\n```",
    "Program call flow": "```mermaid\nsequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant Msk as MaskUNet\n    participant Cls as ResNetClassifier\n    participant N as NoiseInjector\n    participant Pred as Predictor\n    participant Cert as Certification\n    M->>DL: load_data()\n    DL-->>M: datasets\n    M->>Msk: init(params from config)\n    M->>Cls: init(params from config)\n    M->>NoiseInjector: init(sigma levels)\n    alt training\n        M->>Cls: train(train_dataset)\n        M->>Msk: train(train_dataset)\n    end\n    loop inference & certification\n        M->>DL: select input batch\n        Note over M,DL: For each input X\n        M->>NoiseInjector: add_noise Z1\n        NoiseInjector-->>M: noisy_input z1\n        M->>Msk: predict_mask(noisy_input)\n        M-->>Msk: mask\n        M->>NoiseInjector: add_noise Z2\n        NoiseInjector-->>M: noisy_masked_x\n        M->>Cls: predict(noisy_masked_x)\n        Cls-->>Pred: class_logits\n        Note over Pred: Multiple samples\n        Pred->>Pred: aggregate predictions (m1, m2)\n        Pred->>Cert: compute class probabilities p_plus, p_minus\n        Cert->>Cert: calculate radius (Eq. 2.2 / 2.4)\n        Cert-->>M: radius, class prediction, certified decision\n    end\n    M->>Evaluation: evaluate metrics (accuracy, certification rates)\n```",
    "Anything UNCLEAR": "Exact dataset pre-processing steps, and the number of Monte Carlo samples needed for stable certification, require confirmation. Clarification on how to obtain class probabilities—whether via explicit Monte Carlo sampling or analytical bounds—would be helpful. Details of hyperparameter tuning procedures and early stopping criteria are not specified and should be clarified."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "torchvision==0.14.1",
        "numpy==1.21.0",
        "scipy==1.7.3",
        "matplotlib==3.5.1"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class that loads datasets (CIFAR-10, CelebA, ImageNet). It handles data splits, resizing backgrounds, and dataset-specific operations."
        ],
        [
            "model.py",
            "Defines the MaskUNet class (U-Net) and ResNetClassifier class. MaskUNet predicts pixel-wise mask; classifiers perform end-to-end classification."
        ],
        [
            "trainer.py",
            "Implements training routines: trains mask model and classifier end-to-end. Handles optimizer setup, loss functions, hyperparameters, checkpointing, and validation."
        ],
        [
            "evaluation.py",
            "Handles Monte Carlo prediction--generates noisy masked images, aggregates predictions, computes class probabilities, and performs robustness certification formulas (Eq. 2.2, 2.4)."
        ],
        [
            "main.py",
            "Main entry point orchestrates argument parsing, configuration loading, dataset initialization, model creation, training invocation, and evaluation including certification bounds."
        ],
        [
            "utils.py",
            "Provides utility functions: Gaussian noise injection, linear combination of predictions, calculation of certified radius (Eq. 2.2 / 2.4), class probability estimation, and hyperparameter tuning helpers."
        ]
    ],
    "Task list": [
        "requirements.txt (static, not part of code but necessary for environment setup)",
        "dataset_loader.py",
        "model.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "PENDING (no front-end/backend communication, only script and function calls).",
    "Shared Knowledge": "Shared functions in utils.py for noise addition, aggregation, radius calculation. Models share configuration: noise levels, hyperparameters tuning method, data augmentation strategies."
    ,
    "Anything UNCLEAR": "Clarify hyperparameter tuning procedure specifics, Monte Carlo sample count for stable certification, class probability estimation method, and desired hardware environment for large-scale experiments."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: CIFAR10  # Options: CIFAR10, CelebA, ImageNet
  background_scale: 640  # Resize background images to this scale for CIFAR-10/CelebA; for ImageNet, set accordingly
  data_split: train  # Options: train, val, test
training:
  epochs: 200
  batch_size: 256
  learning_rate: 0.001  # Initial learning rate, tune as needed
  weight_decay: 1e-4
  optimizer: AdamW
  momentum: 0.9
  lr_decay: step 30  # Step decay schedule
  lr_gamma: 0.1
  total_noise_budget_sigma: 1.0  # Total privacy budget sigma
  sigma_split_factor: 2.0  # The split: sigma1 = sigma2 = sqrt(2)*sigma, explicitly set
  hyperparameter_tuning: grid_search  # Method: grid_search, bayesian, etc.
  hyperparameter_search_space:
    sigma: [0.25, 0.5, 1.0, 1.5]
    beta: [2.0, 2.25]
model:
  mask_unet:
    base_channels: 32
    channel_mult: [1, 2, 4, 8]
    step_size: 40
    gamma: 0.5
    momentum: 0.9
    learning_rate: 1e-3
    weight_decay: 1e-4
    optimizer: AdamW
  classifier:
    architecture: ResNet50  # Options: ResNet50, ResNet110, etc.
    learning_rate: 1e-3
    weight_decay: 1e-4
    optimizer: AdamW
  training:
    epochs: 200
evaluation:
  monte_carlo_samples: 1000
  certification_confidence: 0.99
  certification_error_tolerance: 0.01
  background_scale_for_certification: 1  # Used for $L_\infty$ certification formulas
hyperparameters:
  mask_weight_loss: cross_entropy  # End-to-end training of mask predictor
  classifier_loss: cross_entropy
  noise_addition: gaussian  # Gaussian noise, sigma parameters controlled via hyperparameter
  privacy_budget_sigma: 1.0  # Total privacy budget sigma for Gaussian mechanisms
  privacy_split: sqrt(2)  # The split for sigma1, sigma2
```


---

**Note:** Full configuration is available in `planning_config.yaml`
