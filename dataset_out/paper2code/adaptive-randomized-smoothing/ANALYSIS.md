# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

# Logic Analysis for dataset_loader.py

## Purpose:
Implement the `DatasetLoader` class that encapsulates dataset loading, preprocessing, data splits, background resizing, and dataset-specific operations necessary for training and evaluation of the ARS pipeline. This class must support CIFAR-10, CelebA, and ImageNet datasets, with configurable options for training, validation, and test splits, as well as background augmentation for high-dimensional background placement.

## Responsibilities:
- Load datasets according to the configuration.
- Perform dataset-specific preprocessing (e.g., cropping, resizing, normalization).
- For CIFAR-10 and CelebA:
  - Generate background images and embed CIFAR images along edges at random positions for high-dimensional scenarios.
  - Resize backgrounds to a specified scale (`background_scale` in config).
  - Split datasets into train/validation/test as specified.
- For ImageNet:
  - Load the dataset with standard transformations, optionally with background augmentation if specified.
- Support data augmentation and transformations aligning with model training standards.
- Provide easy access to data loaders for training, validation, and testing with batch size, shuffle settings, and workers.

## Key Inputs:
- **Configuration parameters** (from the provided `config.yaml`):
  - `dataset.name`: 'CIFAR10', 'CelebA', or 'ImageNet'
  - `dataset.background_scale`: target size scale for background images (for CIFAR-10 and CelebA)
  - `dataset.data_split`: 'train', 'val', or 'test'
  - Additional dataset-specific options (if any)
  
## Dataset-Specific Logic:

### 1. CIFAR-10:
- Standard CIFAR-10 images of size 32x32x3.
- For robustness evaluation with backgrounds:
  - Generate or load background images (large images, e.g., 640x640).
  - Embed CIFAR images along edges of these backgrounds at random positions.
  - Resize backgrounds to the scale specified (`background_scale`: e.g., 640).
  - Data split: use train/test split; possibly create a validation set.
- Normalization: normalize images per standard (mean/std).

### 2. CelebA:
- Load unaligned images.
- Resize and crop images to 160x160 (or specified size).
- Generate spatial variants by randomly cropping images so that the mouth features are offset (simulate spatial variation).
- For evaluation, possibly generate data with mouth position variability.
- Possibly generate masks or only provide raw images with annotated features for custom processing.

### 3. ImageNet:
- Load images with standard 224x224 (or configured size).
- No background augmentation by default; optionally, backgrounds can be scaled (e.g., to 224x W x H).
- Implement appropriate data augmentations for training.
- Support train/validation/test splits.

## Implementation Details:
- Use `torchvision.datasets` subclasses (`CIFAR10`, `CelebA`, `ImageNet`) for standard loading.
- For datasets requiring background placement:
  - Preload or generate background images.
  - Use `torchvision.transforms` to resize backgrounds to `background_scale`.
  - Randomly place CIFAR/celeba images on the background along edges:
    - Random edge selection (top, bottom, left, right).
    - Random position along selected edge.
  - Composite CIFAR or CelebA images onto backgrounds:
    - Overlay the CIFAR image at the chosen position.
    - Maintain transparency or overwrite.
- For CelebA:
  - Apply cropping to extract face regions.
  - Perform spatial variation by randomly cropping during data loading.
- For all datasets:
  - Perform normalization and data augmentation during training.
  - Return data loaders with appropriate batch sizes and shuffling.

## Data Loader API:
- Method `get_data_loader(split: str, batch_size: int, shuffle: bool)` returning `torch.utils.data.DataLoader`.
- Dataset attribute providing access to dataset-specific data (e.g., `(images, labels)`).
- Support optional transformations within dataset for flexible preprocessing.

## Error Handling & Robustness:
- Validate dataset loading options.
- Gracefully handle missing files or corrupted datasets.
- Ensure datasets are downloaded or cached as needed.
- Support reproducibility by setting seed parameters if necessary.

## Summary:
The `DatasetLoader` class must be flexible to load datasets in either standard or augmented modes (with backgrounds, crops, embeddings). It must handle dataset splits, apply correct preprocessing, support large backgrounds and high-dimensional embeddings for CIFAR-10 and CelebA, and support scalable background resizing. The class provides data loaders explicitly configured for training, validation, or testing, integrating dataset-specific augmentations aligned with the experimental protocols. Proper validation and error handling are essential to ensure the class works reliably across multiple datasets and experimental settings.

---

This completes the logic analysis for implementing `dataset_loader.py`.

## evaluation.py

# Evaluation.py Logic Analysis

This evaluation.py module orchestrates the core inference and certification procedures for the ARS framework, following the methodology, formulas, and data flow specified in the paper. It leverages trained models (mask unet, classifier) and utility functions to generate robust class predictions with provable guarantees under input perturbations.

The essential functions within evaluation.py include:

- **Monte Carlo Prediction Generation**: Sampling noise, mask inference, masked images, prediction aggregation.
- **Class Probability Estimation**: Estimating the probability of the predicted class ($p_+$) and its adversarial class ($p_{-}$).
- **Robust Radius Calculation**: Applying formulas (e.g., Eq. 2.2, 2.4) based on class probabilities and noise parameters for certified $L_\infty$ radius bounds.
  
Below is a detailed, step-by-step **logic flow** and **function structure** to support implementation:

---

# 1. Initialization and Inputs

- **Input Data**:
  - `X`: Original input image tensor (normalized, batch consideration if applicable).
  - `n_samples`: Number of Monte Carlo samples for prediction (from config hyperparameters).
  - `conf_level`: Confidence level for probability bounds (e.g., 0.99).
  - `error_tolerance`: Tolerance for the statistical estimation error in probability bounds.
- **Models & Utilities**:
  - `mask_unet`: Trained mask module (`w`), provides pixel-wise masks with values in [0, 1].
  - `classifier`: Trained classifier model `g`.
  - `noise_params`: Dict or objects with `sigma1`, `sigma2`.
- **Other Configurations**:
  - `background_scale`: For certification radius calculations under $L_\infty$ bounds.
  - `norm_type`: Norm bounds considered (e.g., `L_inf`).
  - `certification_formulas`: Reference to formula code (Eq. 2.2, 2.4).

---

# 2. Step 1: Mask Prediction (Input-Dependent Mask)

**Function: `predict_mask(noisy_input)`**
- Sample noise `z1` ~ `N(0, sigma1^2 I)`.
- Generate noisy image: `X_noisy = X + z1`.
- Pass `X_noisy` through `mask_unet`: output value in [0, 1]).
- Return the mask tensor `w(m1)`.

**Note**: This step is designed to focus the subsequent noise addition on relevant parts, reducing noise variance in step 2.

---

# 3. Step 2: Generate Noisy Masked Image

- **Compute the Mask**: `w = predict_mask(X + z1)` (per sample or batch).
- **Element-wise multiply**: `masked_X = w * X`.
- **Add noise**:
  - Sample `z2` ~ `N(0, (||w||_2)^2 / d * sigma2^2 * I)` (per the formula).
  - Generate `m2 = masked_X + z2`.
- The noise in this step depends on the mask norm (`||w||_2`) and the privacy parameters.

**Function: `generate_masked_noisy_image(w, X)`**

---

# 4. Prediction via Classifier

- **Input**:
  - Multiple Monte Carlo samples (n_samples).
- **Process**:
  - For each sample:
    1. Generate `z1`, `z2`.
    2. Predict mask: `w`.
    3. Generate `m2`.
    4. Compute weighted average estimate:
       $$ \hat{X} = c_{1} m_1 + c_{2} m_2 $$
       where `c_1` and `c_2` are derived to produce an unbiased estimate with minimal variance, per the formulas:
       $$ c_{1,i} = \frac{||w||_2 \sigma_2^2}{\sigma_1^2 w_i^2 + ||w||_2^2 \sigma_2^2} $$
       $$ c_{2,i} = \frac{\sigma_1^2 w_i}{\sigma_1^2 w_i^2 + ||w||_2^2 \sigma_2^2} $$
  - Pass `\hat{X}` through classifier `g`.
- **Output**:
  - Collect class logits or probabilities for each sample.

---

# 5. Class Probability Estimation and Confidence Bounds

- **Aggregate Predictions**:
  - Compute softmax over logits or directly estimate class probabilities over all `n_samples`.
- **Estimate class probabilities**:
  - Extract `p_{+}`: probability of the predicted class (`y+`) as the fraction of samples predicted as class `y+`.
  - Extract `p_{-}`: the maximum probability over other classes.
- **Confidence bounds**:
  - Use statistical bounds (Clopper-Pearson or normal approximation) at confidence level `conf_level` with error tolerance `error_tolerance` to construct lower bounds (`p_{+}^{lower}`) and upper bounds (`p_{-}^{upper}`).

**Function: `estimate_class_probabilities(predictions, conf_level, error_tolerance)`**

---

# 6. Certification Radius Calculation

- Based on class probability bounds:
  - Apply Eq. 2.2 or Eq. 2.4:
  
  $$ r_{X} = \frac{\sigma}{2} \left( \Phi^{-1}(\underline{p_{+}}) - \Phi^{-1}(\overline{p_{-}}) \right) $$

- For $L_\infty$:
  
  $$ r_{X}^{\infty} = \frac{\sigma}{2\sqrt{d}} \left( \Phi^{-1}(\underline{p_{+}}) - \Phi^{-1}(\overline{p_{-}}) \right) $$

- Adjust the radius calculation formula as per the norm constraints, background scale, and dimension.
- Result: a scalar bounding radius where the predicted class remains robust with probability at least `conf_level`.

**Function: `compute_certified_radius(p_plus_lower, p_minus_upper, sigma, dimension, norm_type='L_infinity', background_scale=1)`**

---

# 7. Return Results

- **Predicted Class**:
  - Class with highest estimated probability.
- **Certified Radius**:
  - Calculated robustness radius.
- **Class probabilities with confidence bounds**.
- **Additional info**:
  - Estimated class probabilities.
  - Variance bounds.
  - Confidence level.

---

# 8. Additional Considerations

- **Monte Carlo sampling size (`n_samples`)**:
  - Pick sufficiently large (e.g., 1000) for statistical accuracy, as per configuration.
- **Ensemble over multiple inputs**:
  - When evaluating batches, process sample-wise, then aggregate class probabilities.
- **Efficiency Optimization**:
  - Reuse noise samples where possible.
  - Log detailed sampling stats and class probability estimates.
Parse and handle multiple images per batch.

---

# Summary

`evaluation.py` will:

- Call the mask predictor with noisy inputs.
- Generate masked, noisy images with input-dependent variance.
- Aggregate multiple predictions over Monte Carlo samples for class probabilities.
- Adjust class probabilities estimates based on confidence bounds.
- Compute the certified robustness radius following formulas (Eq. 2.2, 2.4).
- Return class label, radius, and probability confidence bounds.

This structured approach adheres to the framework, ensuring transparent, reproducible, and theoretically sound robustness certifications consistent with the paper’s methodology.

## main.py

# Logic Analysis for main.py

This script serves as the orchestrator for the entire experimental pipeline described in the paper. Its core responsibilities are: parsing configurations, initializing datasets and models, conducting training, and performing evaluation and certification. The flow must be aligned with the methodology, hyperparameters, and evaluation protocols as specified, ensuring reproducibility and fidelity to the paper's experiments.

---

# 1. Initialization and Argument Parsing
- Parse command-line arguments specifying:
  - Mode: 'train', 'evaluate', or 'certify'
  - Dataset subset (if applicable)
  - Hyperparameter tuning mode (grid search, etc.)
  - Debug/verbose flags
- Load configuration from **config.yaml** to ensure all parameters are set as per the paper:
  - Dataset parameters
  - Model hyperparameters (mask model, classifier)
  - Noise levels (sigma, split factor)
  - Training parameters (epochs, batch size, learning rate, optimizer)
  - Evaluation parameters (Monte Carlo samples, confidence levels)
  - Hyperparameter search spaces

---

# 2. Dataset Initialization
- Instantiate DatasetLoader class:
  - Use dataset parameters: name (CIFAR10, CelebA, ImageNet), background scale, data split.
  - For CIFAR-10/CelebA:
    - Resize background images to `background_scale` (e.g., 640).
    - Place CIFAR images or CelebA cropped images randomly on backgrounds for the BG experiments.
    - Apply relevant data augmentations if specified.
  - For ImageNet:
    - Load standard validation or test fold images.
- Prepare datasets for:
  - Training (with augmentation if specified)
  - Validation (for hyperparameter selection)
  - Testing (for final evaluation and certification)

---

# 3. Model Construction
- **Mask Model (w)**:
  - Instantiate MaskUNet with parameters:
    - base_channels=32
    - channel_mult=[1,2,4,8]
    - step_size=40
    - gamma=0.5
  - Initialize weights with standard methods (He, Xavier).
  - Setup optimizer (AdamW) with learning rate 1e-3, weight decay 1e-4.
  - Prepare for end-to-end training.
    
- **Base Classifier (g)**:
  - Instantiate with architecture specified (ResNet50, ResNet110).
  - Initialize weights.
  - Setup optimizer with learning rate 1e-3, weight decay 1e-4.
  - Load pretrained weights if doing pretraining, or initialize randomly for joint training.

---

# 4. Training Procedures
- **Hyperparameter Selection (if needed)**:
  - Depending on mode, run grid search over the following:
    - $\sigma$ (0.25, 0.5, 1.0, 1.5)
    - $\beta$ (2.0, 2.25)
    - Mask model hyperparameters (e.g., learning rate schedule)
  - For each candidate hyperparameter set:
    - Train mask model and classifier jointly end-to-end.
    - Use the training dataset:
      - For each batch:
        - Generate Gaussian noise $z_1 \sim \mathcal{N}(0, \sigma_1^2 I)$ with $\sigma_1 = \sqrt{2}\sigma$ (per config).
        - Pass noisy input $X + z_1$ through mask network to produce $w(m_1)$.
        - Apply mask to input: $w(m_1) \odot X$.
        - Add second Gaussian noise $z_2 \sim \mathcal{N}(0, \sigma_2^2 I)$ with $\sigma_2 = \sqrt{2}\sigma$.
        - Build combined input $\hat{X}$ via linear weights calculated to ensure unbiased estimate.
        - Train classifier with cross-entropy loss on prediction output.
        - Train mask network with, e.g., masked-cross entropy or an auxiliary loss if specified.
    - Validate on validation subset, record metrics.
- **Final model training**:
  - Select best hyperparameters per validation performance.
  - Train the final model(s) with the chosen hyperparameters and full training data.

---

# 5. Inference Workflow
- For each test sample:
  - Generate multiple Monte Carlo samples ($\sim$1000 as per config):
    - For each sample:
      - Draw noise $z_1$, compute mask $w(m_1)$.
      - Generate $m_1$ (from noisy input + $z_1$).
      - Compute masked input: $w(m_1) \odot X$.
      - Draw noise $z_2$, compute $m_2$.
      - Compute linear combination $\hat{X}$:
        - Coefficients are calculated as per formulas to minimize variance while maintaining unbiasedness.
      - Feed $\hat{X}$ into classifier $g$ to get class logits.
  - Average class logits/probabilities over all samples.
  - Determine class prediction based on maximum average probability.
  - Compute class probabilities ($p_+$, $p_-$) from the sampled predictions.
  
---

# 6. Certification Calculation
- For each sample prediction:
  - Acquire class probabilities: $\underline{p_+}$, $\overline{p_-}$ using statistical estimation (Clopper-Pearson, Monte Carlo bounds).
  - Use formulas:
    - Eq. 2.2 / 2.4 / 2.5:
      - Calculate the radius $r_X^\infty$ using class probabilities, noise levels, and the inverse Gaussian CDF $\Phi^{-1}$.
      - For $L_\infty$, include the dimension scaling (dividing the radius estimate accordingly).
    - For adaptive steps, incorporate the $f$-DP composition result (Equation 3/4).
- Determine the certified robustness radius per sample.
- Collect and aggregate results:
  - Certified accuracy: fraction of samples with radius ≥ threshold.
  - Standard accuracy: fraction of correct predictions without robustness bound filter.

---

# 7. Logging & Output
- Save raw class probabilities, predicted classes, and radius bounds.
- Record per-sample and overall metrics:
  - Standard accuracy.
  - Certified robustness at various radii thresholds.
- Save model checkpoints with hyperparameters and training logs.
- Generate plots for:
  - Certified accuracy vs. radius.
  - Standard accuracy vs. $\sigma$.
  - Visualizations of masks (especially significant for interpretability).

---

# 8. Additional Considerations
- Use fixed random seeds for reproducibility.
- Manage GPU resources per environment: e.g., allocate sufficient memory, clear cache after each run.
- Handle large background images efficiently in data pipeline.
- Document hyperparameters, training time, and experimental details for reproducibility and alignment with paper.

---

# 9. Unclear or Optional Enhancements
- Exact number of Monte Carlo samples needed for stability: ensure at least 1000 samples.
- Confidence intervals for class probabilities: typically, use the Clopper-Pearson method or Bayesian bounds.
- Hyperparameter tuning details: adopt grid search or Bayesian optimization based on validation performance.
- For large-scale experiments, consider distributed training or evaluation if resources permit.

---

This thorough analysis ensures that every step reflects the paper’s methodology, experimental protocols, and evaluation criteria, enabling faithful implementation and reproducibility.

## model.py

# Logic Analysis for model.py

This file contains the core model definitions: the MaskUNet (the mask generator) and the ResNetClassifier (the classifier backbone). The implementation must align exactly with the architecture and interfaces prescribed in the paper, following the design constraints, hyperparameters, and procedural steps described.

---

# 1. **Class: MaskUNet**

### Purpose:
- Generates an adaptive pixel-wise mask $w(m_1)$ from a noisy input image during test-time.
- Used as part of the adaptive randomized smoothing pipeline, where the mask reduces the effective input dimension and noise variance in the second step.

### Inputs:
- Noisy image tensor: shape `(batch_size, 3, H, W)` (RGB image with added Gaussian noise).
- Hyperparameters:
  - `base_channels` (int): initial number of filters in UNet's first layer, e.g., 32.
  - `channel_mult` (list): multiplier per UNet level, e.g. `[1, 2, 4, 8]`.
  - `step_size` (int): number of training steps; for architecture, this is primarily training hyperparameter.
  - `gamma` (float): growth factor or learning rate decay (if applicable, or used for numeric stability).
  - `momentum` (float): momenta, often for optimizer (but in this class, primarily for training, not inference).

### Outputs:
- Mask tensor: shape `(batch_size, 1, H, W)`, with values in [0, 1], after passing through sigmoid activation.

### Architecture:
- Based on a U-Net structure, with the following key points:
  - Encoder: sequence of conv + downsampling (max pool or strided conv), progressively increasing filters: starting from `base_channels`, scaled by `channel_mult`.
  - Bottleneck: last layer of encoder.
  - Decoder: upsampling + skip connections from encoder, with the same `channel_mult` scale.
  - Final layer: 1 filter (per-pixel), sigmoid activation.
  
### Implementation details:
- Use `torch.nn` modules: Conv2d, BatchNorm2d, ReLU, MaxPool2d, ConvTranspose2d (or Upsample + conv).
- Initialize parameters as per hyperparameters.
- During training:
  - Loss: pixel-wise cross-entropy or mean squared error (depends on training target—most likely binary cross-entropy per pixel).
  - Optimizer: AdamW (per config).
  - Learning rate schedule: step with gamma, as per hyperparameter.
- During inference:
  - Forward pass produces a mask in [0, 1].

### Notes:
- Mask prediction acts as a post-processing of the noisy input.
- Make sure the spatial dimensions (`H, W`) are preserved or correctly scaled as per architecture.
- Use dropouts or normalizations as needed for training stability.

---

# 2. **Class: ResNetClassifier**

### Purpose:
- Complete classification with the combined, masked, and noise-perturbed input.
- Consists of:
  - A backbone (ResNet-50 or ResNet-110).
  - Input: median- or Monte Carlo-sampled images, which are a linear combination of $m_1$, $m_2$, and the mask.
- Produces class logits or probabilities for prediction and certification.

### Inputs:
- Image tensor: shape `(batch_size, 3, H, W)` after the combination step.
- Hyperparameters:
  - Architecture: ResNet variant (ResNet50 or ResNet110).
  - Learning rate, weight decay, optimizer: configured during training.
  
### Outputs:
- Logits tensor: shape `(batch_size, num_classes)` (e.g., 10 for CIFAR-10).

### Implementation:
- Use pre-existing ResNet implementations (from torchvision or custom).
- Initialize model weights; load from checkpoint if resuming training.
- Forward pass:
  - Process images through ResNet layers.
  - Final fully connected layer output logits.

### Training:
- Loss: cross-entropy.
- Include data augmentation, normalization as per standard.
- During testing:
  - Generate multiple noisy samples.
  - Average logits or class probabilities.
  - Use class probabilities for certification formulas.

### Notes:
- The classifier must be flexible to accept the combined, adaptive, masked image input.
- For training stability, standard techniques apply (learning rate scheduling, early stopping, regularization).

---

# 3. **Additional considerations**

### a. Interfaces:
- The classes should provide:
  - `forward()` methods returning either raw masks, logits, or softmax probabilities.
  - For `MaskUNet`, a method for inference to generate masks given noisy inputs.
  - For `ResNetClassifier`, a method to predict class probabilities given input images.
- Encapsulation:
  - Full separation of mask generator and classifier.
  - Methods for training, evaluation, and inference on new noisy inputs.

### b. Parameter passing:
- Model parameters (hyperparameters) supplied during instantiation (`__init__`).
- During training:
  - Save and restore states via `.state_dict()`.
- During inference:
  - Forward pass with provided noisy inputs.
  
### c. Computational considerations:
- Use reusable modules; avoid redundant computation.
- Ensure that model training and inference are clearly distinguished (train/eval modes).

### d. Reproducibility:
- Initialize weights with fixed seed.
- Consistent hyperparameters.
- Activation functions and normalization layers carefully chosen for training stability.

---

# 4. **Implementation outline (not code):**

```python
class MaskUNet(nn.Module):
    def __init__(self, base_channels, channel_mult, step_size, gamma, momentum):
        super().__init__()
        # Define encoder blocks: Conv2d + BatchNorm + ReLU + Downsampling
        # Define decoder blocks: upsampling + Conv2d + BatchNorm + ReLU, skip connections
        # Final layer: 1 convolution + sigmoid
        # Store hyperparameters
    def forward(self, noisy_input):
        # Process through encoder, bottleneck, decoder
        # Skip connections
        # Output shape (batch_size, 1, H, W)
        # Apply sigmoid
        return mask

class ResNetClassifier(nn.Module):
    def __init__(self, architecture, num_classes, learning_rate, weight_decay, optimizer_params):
        super().__init__()
        # Instantiate ResNet backbone (from torchvision or custom)
        # Fully connected layer adjusted for num_classes
        # Store hyperparameters
    def forward(self, images):
        # Return logits
        # Or softmax probabilities if needed
        return logits
    def predict(self, images, n_samples):
        # Generate multiple noisy samples if needed
        # Aggregate class probabilities, compute mean
        # Return probabilities
```

---

# 5. **Summary**

This logic analysis specifies:
- Clear input-output interfaces.
- Architecture choices reflective of the paper (U-Net for mask, ResNet for classifier).
- Training and inference behaviors aligned with the described method.
- Hyperparameters to be tuned or set as constants.
- Reproducibility and modularity for integration into the main pipeline.

This thorough understanding ensures that the code implementation faithfully reproduces the methodology and experimental setup detailed in the paper.

## requirements.txt (static, not part of code but necessary for environment setup)

# requirements.txt

This file specifies the core Python packages required to implement, train, evaluate, and certify the Adaptive Randomized Smoothing (ARS) method as described in the paper, following the provided design and configuration details. The packages include deep learning frameworks, mathematical and scientific libraries, and visualization tools, ensuring a stable and reproducible environment compliant with the paper's experimental setup.

## Core Deep Learning Framework
- **torch==1.13.1**  
  Provides the PyTorch deep learning library, fundamental for implementing the ResNet classifiers, U-Net mask model, noise injection modules, and training routines.

- **torchvision==0.14.1**  
  Facilitates dataset handling, image transformations, data augmentation, and pretrained model access (if needed). Supports dataset loading for CIFAR-10, CelebA, and ImageNet.

## Numerical and Scientific Computing
- **numpy==1.21.0**  
  Essential for array manipulations, mathematical operations, and numerical computations necessary for constructing the certificates, combining model outputs, and implementing formulas (e.g., Gaussian quantiles).

- **scipy==1.7.3**  
  Provides statistical functions (such as inverse CDF via `scipy.stats.norm.ppf`), optimization routines, and support for numerical derivations relevant to certification bounds.

## Visualization and Debugging
- **matplotlib==3.5.1**  
  For plotting experimental results, certification accuracy curves, masks, and example images during training and evaluation, as outlined in the paper.

## Additional Considerations
- **Optional / Environment-related:**  
  - **torchsummary** (for model visualization and debugging).  
  - **tensorboard** (for logging training progress, hyperparameter sweeps, and visualization).  
  - **seaborn** (for advanced statistical plotting, if needed).

---

# Summary
This static set of dependencies ensures that implementation aligns with the experimental, theoretical, and infrastructural requirements specified in the plan and design, supporting reproducibility of results under the configuration provided.  
Ensure compatible CUDA/cuDNN versions if GPU acceleration is utilized, matching the paper's experimental hardware (NVIDIA A100, RTX 4090).

## trainer.py

# Logic Analysis for trainer.py in Adaptive Randomized Smoothing Implementation

This file implements the training routines for both the mask network (MaskUNet) and the base classifier (ResNet), following the methodology outlined in the paper "Adaptive Randomized Smoothing," with particular emphasis on the hyperparameters, training procedures, and data processing detailed in the paper and the provided configuration (config.yaml).

---

## Key Responsibilities

1. **Model Initialization**
   - Instantiate the mask network (`MaskUNet`) with parameters specified in `model.mask_unet`.
   - Instantiate the classifier (`ResNet`) with architecture specified (e.g., ResNet50, ResNet110).
   - Set up optimizers for both models (AdamW) with their respective hyperparameters.
   
2. **Hyperparameters & Training Schedule**
   - Use the number of epochs (`training.epochs`) for full training.
   - Roll out the learning rate decay schedule (`lr_decay`, `lr_gamma`) at specified step epochs.
   - Use batch size (`training.batch_size`), initially 256, configurable.
   - Learning rate (`training.learning_rate`), e.g., 1e-3, with decay as scheduled.
   - Weight decay for regularization.

3. **Data Handling**
   - Utilize data loaders based on dataset (`dataset.name`) with possible background scaling (`dataset.background_scale`).
   - Implement data augmentation consistent with the methodology (standard normalization, possibly random cropping, flipping if specified).
   - For the training set, use the dataset split indicated (`train` in data_split).

4. **Noise Injection for Differential Privacy / Robustness**
   - For each training batch, generate Gaussian noise `z_1` for the mask network input.
   - Noise parameters: 
     - `sigma_1` chosen as per hyperparameters, e.g., `training.total_noise_budget_sigma / sqrt(2)` (since sigma split evenly).
     - Noise addition should be implemented via a utility function (e.g., `add_gaussian_noise(tensor, sigma)`).
   - Data augmentation (e.g., random shifts, flips) should be applied in the data loader pipeline.

5. **Training a Two-Component Model**
   - **Step 1:** Forward pass:
     - Input noisy images (`X + z_1`) into mask network (`MaskUNet.predict_mask()`), producing mask `w(m_1)`.
   - **Step 2:** Use the mask to generate masked input:
     - Elementwise multiply mask with input image: `masked_input = w(m_1) * X`.
     - Add second noise `z_2` with variance `sigma_2` (computed as per split, e.g., `sqrt(2)*sigma`).
     - Implement noise addition (`add_gaussian_noise()`).
   - **Step 3:** Use the combined image:
     - Compute the unbiased estimate `hat_X`, linearly combining `m_1` and `m_2`:
       - Calculate weights `c_{1,i}` and `c_{2,i}` per pixel to minimize variance, enforcing unbiased estimate.
       - These weights depend on mask and noise variances. Use derived formulas from the paper, or precompute within training iteration.
   - **Step 4:** Classification:
     - Pass `hat_X` through classifier `g`.
     - Compute classification loss (cross-entropy) with the true labels.
   - **Step 5:** Mask loss (if training mask jointly):
     - If incorporating an auxiliary loss (`mask_weight_loss`), include a loss term on the mask predictions (e.g., if using supervised mask learning, or regularization).
     - Otherwise, only optimize the classifier end-to-end.
   
6. **Loss Function & Optimization**
   - Primary loss: cross-entropy on predictions.
   - If joint mask training: include mask loss.
   - Backpropagate combined loss (if mask and classifier trained jointly).
   - Step optimizers:
     - Update mask network weights.
     - Update classifier weights.
   - Use gradient clipping or regularization as needed to stabilize training.

7. **Learning Rate & Scheduler**
   - Implement step decay at specified epochs (e.g., at step 30), decaying by factor `lr_gamma`.
   - Use PyTorch's `lr_scheduler.StepLR` or custom scheduler.

8. **Checkpointing & Validation**
   - Save model weights periodically (e.g., best validation accuracy).
   - Evaluate on validation set after each epoch:
     - Compute classification accuracy.
     - Evaluate validation of robustness parameters (if possible).
   
9. **Hyperparameter Tuning & Grid Search**
   - For hyperparameters like `sigma`, `beta`, learning rate, perform grid or Bayesian search over validation set.
   - Record resulting performance metrics.

10. **Training Loop Skeleton (Outline)**
    ```
    for epoch in range(training_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 1. Add Gaussian noise for mask generator
            z1 = generate_noise(sigma_1, batch_size)
            noisy_inputs = inputs + z1
            
            # 2. Compute mask
            mask = mask_net.predict_mask(noisy_inputs)
            
            # 3. Generate masked input
            masked_input = mask * inputs
            
            # 4. Add second noise for the classifier input
            z2 = generate_noise(sigma_2, batch_size)
            noisy_masked_input = masked_input + z2
            
            # 5. Compute unbiased estimate (hat_X) via variance minimization (per pixel)
            hat_X = compute_unbiased_estimate(weights, m_1, m_2)
            
            # 6. Forward pass classifier with hat_X
            logits = classifier(hat_X)
            loss = cross_entropy(logits, labels)
            
            # 7. Backpropagation
            optimizer_mask.zero_grad()
            optimizer_classifier.zero_grad()
            loss.backward()
            optimizer_mask.step()
            optimizer_classifier.step()
            
        # Learning rate decay
        scheduler.step()
        
        # Validation and checkpointing at epoch end
        evaluate_and_save_if_improved()
    ```
    - Implement the above with appropriate pyTorch code, data augmentation, and error handling.

---

## Assumptions & Clarifications Needed
- Exact number of Monte Carlo samples used during training (e.g., whether to simulate multiple noise instances per batch).
- Whether the mask is trained fully supervised or via self-supervised regularization.
- How frequently to perform validation/early stopping.
- Specification of the data augmentation pipeline.
- Precise calculation of per-pixel weights (`c_{1,i}`, `c_{2,i}`), which depend on mask outputs and noise variances.
- Open questions on the auxiliary mask loss – whether to include a supervised component or not.

---

## Summary
This logic analysis guides the development of a comprehensive training pipeline capable of jointly training and optimizing:
- The mask generator (mask network): adds test-time adaptivity.
- The classifier: end-to-end with the mask and noise layers.
- The entire architecture is trained to optimize classification accuracy, robustness, and the probabilistic guarantees as per the paper.
  
The training process aligns with the detailed methodology, hyperparameters, and theoretical framework, ensuring fidelity and reproducibility.

## utils.py

{
  "utils.py": [
    {
      "component": "Gaussian Noise Injection Function",
      "description": "Implement a function to add Gaussian noise to input tensors, supporting both training and testing phases. The function will take an input tensor and a sigma value as parameters, and return the noisy tensor. It should handle batching of inputs and be compatible with gradient computation if used during training. The function will be called `add_gaussian_noise(input: torch.Tensor, sigma: float) -> torch.Tensor`.",
      "details": "Use torch.randn to generate noise with the same shape as input, scaled by sigma. During evaluation, the noise is sampled once per test sample; during training, multiple samples might be used to approximate expectation. Ensure that the noise is sampled from a normal distribution: `noise = torch.randn_like(input) * sigma`."
    },
    {
      "component": "Linear Combination of Predictions",
      "description": "Create a function to perform an unbiased linear combination of two noisy predictions, `m1` and `m2`, given their respective unnormalized outputs or estimates. The function computes coefficients `c1` and `c2` per pixel to minimize variance under the constraint `c1 + w_i c2 = 1`, where `w_i` are mask weights from the mask model, and outputs the combined estimate `hat_x` for each pixel. The resulting `hat_x` should be an unbiased estimator of the original input `X`. The function will be `combine_predictions(m1: torch.Tensor, m2: torch.Tensor, w: torch.Tensor, sigma1: float, sigma2: float) -> torch.Tensor`.",
      "details": "Compute per-pixel variance-minimizing coefficients based on the formulas provided in section 3, involving computing `c1,i = (w_i^2 σ2^2 + ||w||_2^2 σ1^2)^{-1} * σ1^2 * w_i` and `c2,i = (σ1^2 w_i^2 + ||w||_2^2 σ2^2)^{-1} * σ1^2 w_i`, then generate `hat_x = c1 * m1 + c2 * w * m2`. Make sure to prevent division by zero or numerical instability, e.g., add epsilon as needed. The mask `w` is pixel-wise, shape matching the images, with values in [0, 1]."
    },
    {
      "component": "Compute Certified Radius (Eq. 2.2 / 2.4)",
      "description": "Implement functions to compute the certification radius based on class probabilities and privacy parameters, following formulas (Eq. 2.2 for standard setup, Eq. 2.4 for $L_\infty$ bounds). Functions will take class probability estimates `p_plus`, `p_minus`, the noise level `sigma`, and auxiliary info (such as the total number of Monte Carlo samples). The functions will be `compute_cert_radius(p_plus: float, p_minus: float, sigma: float) -> float`.",
      "details": "Use the inverse Gaussian CDF function from `scipy.stats.norm.ppf`. For standard case (Eq. 2.2): `radius = (sigma / 2) * (scipy.stats.norm.ppf(p_plus) - scipy.stats.norm.ppf(p_minus))`. For $L_\infty$, adapt the formula accordingly, dividing by `2 * sqrt(d)`, where `d` is input dimension. These functions will be called during evaluation to produce the certification bounds."
    },
    {
      "component": "Class Probability Estimation via Monte Carlo",
      "description": "Implement a function to estimate class probabilities from multiple stochastic predictions over Monte Carlo samples. This function will generate multiple noisy inputs (using `add_gaussian_noise`), pass them through the classifier `g`, and tally class predictions to estimate probabilities. The function signature: `estimate_class_probabilities(input: torch.Tensor, classifier: callable, num_samples: int) -> dict`.",
      "details": "For each sample, add noise, pass through `g`, and record the class. After all samples, compute the proportion of times each class is predicted. The returned dict maps class labels to their estimated probabilities. These class probabilities are essential for the certification process and for selecting the top class (`p_plus`) and competitors (`p_minus`)."
    },
    {
      "component": "Hyperparameter Tuning Helpers",
      "description": "Provide utilities to perform grid search or Bayesian optimization over key hyperparameters such as `sigma` and `beta`. Functions include `search_sigma()` and `search_beta()`, which evaluate model performance (validation accuracy or certified accuracy) across a specified space and select the best hyperparameters.",
      "details": "Implement a function to iterate over the search space, train or evaluate models accordingly, and store results. Use the validation dataset (or a subset, e.g., 200 samples as per the paper) for tuning. Record hyperparameters and corresponding accuracy metrics for analysis."
    },
    {
      "component": "Additional Utility Functions",
      "description": "Implement helper functions such as `load_background_images()`, `resize_backgrounds()`, and `sample_batch()` to support data handling and batch processing. For example, resizing backgrounds to a given `background_scale`, cropping, and batching for training/inference.",
      "details": "Ensure these functions are compatible with the dataset loader pipeline, and support multiple datasets with dataset-specific pre-processing steps."
    }
  ],
  "Remarks": "All functions should be well-documented, with clear inputs and outputs, supporting batch processing. Use `torch` and `numpy` for tensor operations, vectorization, and numerical stability. Ensure reproducibility by fixing random seeds for noise sampling and Monte Carlo estimates during evaluation. Verify formula correctness with small test cases before integration."
}

