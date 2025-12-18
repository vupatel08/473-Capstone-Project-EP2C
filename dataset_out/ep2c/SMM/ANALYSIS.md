# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## config.py

# Logic Analysis for config.py

This configuration file encapsulates and formalizes all essential parameters, settings, and hyperparameters required for implementing the Sample-specific Multi-channel Mask (SMM) framework as described in the paper. Its purpose is to serve as a single source of truth, ensuring consistency, reproducibility, and clarity during the implementation process.

Below is a comprehensive breakdown of each component and parameter, aligned with the paper and the provided plan, to facilitate systematic coding:

---

## 1. Dataset Configuration

**Purpose:** Defines dataset specifics and ensures proper loading, resizing, and batching.

- `name`: String identifier of the dataset, e.g., `'CIFAR10'`. Controls which dataset loader to instantiate.
- `input_size`: List `[height, width]` of images after resizing. Used for dataset preprocessing and target model input sizing.
- `train_split` & `test_split`: Sizes of training and testing sets for training and validation.
- `batch_size`: Number of samples per batch for training and evaluation, e.g., 128.

*Need to ensure the dataset loader reads images according to `input_size`, normalizes properly, and supports batching.*

---

## 2. Model Configuration

**Purpose:** Specifies pre-trained backbone model type and whether to load it.

- `name`: String, e.g., `'ResNet50'`, dictates model selection (ResNet or ViT-Base as per paper).
- `pretrained`: Boolean, e.g., `true`, to load pre-trained weights (ImageNet-1k trained).
  
*Implementation note:* Load the model in evaluation mode with parameters frozen except for the mask generator and pattern.

---

## 3. Mask Generator Architecture

**Purpose:** Controls the structure and size of the lightweight CNN used to generate sample-specific masks.

- `architecture_depth`: Integer (e.g., 5), specifies CNN depth (number of convolutional + pooling layers).
- `kernel_size`: Integer (e.g., 3), kernel size for convolutional layers.
- `filters`: Integer (e.g., 64), number of filters in convolutional layers.
- `pooling_layers`: Integer (e.g., 2), number of max pooling layers, determines reduction in spatial resolution.
- `output_ratio`: String `'1/8'`, determines the downsampling factor of the output mask relative to input image size.

*In implementation,* these parameters guide the construction of the `MaskGenerator` class. Ensure that the output size after CNN and pooling aligns with the specified ratio, and support flexible design for different depths.

---

## 4. Training Hyperparameters

**Purpose:** Define the optimizer, learning rates, decay schedules, and regularization.

- `optimizer`: String, `'Adam'`, specifies optimization algorithm.
- `learning_rate`: Float, `0.01`, initial LR for mask generator (`phi`).
- `lr_decay_epochs`: List of integers, `[100]`, epochs at which learning rate decay occurs.
- `lr_decay_factor`: Float, `0.1`, multiplicative factor for LR decay.
- `epochs`: Integer, 200, total training epochs.
- `pattern_lr`: Float, e.g., `0.001`, learning rate for the pattern (`delta`).
- `pattern_lr_decay_epochs`: List `[100]`, decay schedule for pattern LR.
- `pattern_lr_decay_factor`: Float, similarly `0.1`.
- `weight_decay`: Float, e.g., `1e-4`, L2 regularization for optimizer.
- `pattern_init`: String, `'zeros'`, pattern pattern initialization method.

*Implementation:* Create optimizer instances for $\phi$ and $\delta$ separately with these LR settings and decay schedules.

---

## 5. Sampling & Masking Configuration

**Purpose:** Determines image resizing, patch size for interpolation, and mask resolution.

- `image_resize`: List `[32, 32]`, size to resize input images for uniform processing.
- `patch_size`: Integer `8`, used in patch-wise upsampling (e.g., a patch of 8×8 pixels).
  
*Implementation:* The resized images serve as input to the CNN mask generator; the patch size determines the level of mask granularity during upsampling. Ensure the generated mask size matches the input size after tile-upscaling.

---

## 6. Evaluation & Visualization Settings

**Purpose:** Confirm metrics, visualization preferences, and output handling.

- `metrics`: String `'accuracy'`, type of evaluation metric.
- `visualize`: Boolean, `true`, controls whether to generate and display illustrative images/masks after training.
  
*Implication:* During evaluation, generate visualizations of reprogrammed images with masks overlaid, following the figures provided in the paper.

---

## 7. Reproducibility

**Purpose:** Set static seed for deterministic operations.

- `seed`: Integer `42`, ensures seed setting for torch, numpy, and other relevant libraries.

*Use for reproducible experiments; set seeds at program start.*

---

## 8. Additional Notes / Future Extensions

- Compatibility notes: The configuration facilitates switching between models (ResNet, ViT) and datasets.
- Support hyperparameter sweeps for patch size, LR schedules, etc.
- Could include toggles for different experimental variants (e.g., pattern initialization, mask resolutions).

---

## Summary

To implement the code effectively, **each section of this configuration must be read carefully**:

- Instantiate dataset loaders with `dataset` parameters.
- Initialize model backbone with `model` parameters.
- Construct `MaskGenerator` using `mask_generator` parameters.
- Configure optimizer and scheduler with `training` parameters.
- Maintain pattern and mask generation according to stated hyperparameters.
- Use seed for reproducibility.
- Enable visualization if specified.

**This well-structured `config.py` serves as the backbone for modular, flexible, and reproducible implementation of the SMM framework as described in the paper.**

## dataset_loader.py

**Logic Analysis for dataset_loader.py**

---

### Purpose:
Design and implement a `DatasetLoader` class capable of supporting multiple datasets, ensuring flexible data loading, dataset splitting, dataset normalization, and image resizing, all conforming to the input size specified in the configuration. The class should produce `DataLoader` instances suitable for training, validation, and testing phases, facilitating batch processing and shuffling.

---

### Core Responsibilities:
1. **Dataset Configuration**:
   - Map dataset names (e.g., CIFAR10, CIFAR100, SVHN, GTSRB, Flowers102, DTD, UCF101, Food101, EuroSAT, OxfordPets, SUN397) to their specific dataset classes in `torchvision.datasets` or customDataset classes.
   - Support labels and splits as per the `config.yaml`.

2. **Data Loading**:
   - Load datasets with appropriate root directories.
   - Apply splits for training and testing (or validation).
   - Apply dataset-specific transformations: normalization, conversion to tensors, and resizing to target input size.

3. **Resizing Images**:
   - Resize images from their original dimensions to the specified `input_size` in config.
   - Use bilinear interpolation for resizing to match the SMM input pipeline.

4. **Transform Pipelines**:
   - Compose dataset transforms:
     - Resize to target size.
     - Convert images to tensors.
     - Normalize according to pretrained model expectations (e.g., ImageNet mean/std).
   - Ensure consistent normalization across datasets matching the pre-trained model's normalization.

5. **DataLoader Creation**:
   - Wrap datasets into `DataLoader`s.
   - Use batch size and shuffle settings from config.
   - Enable shuffling for training datasets, disable for evaluation datasets.

6. **Supporting Multiple Datasets & Flexibility**:
   - Support selection of datasets via constructor argument.
   - Support dynamic input size parameters.
   - Incorporate dataset-specific considerations (e.g., for UCF101 and SUN397 which are video datasets, treat frames as images or support video loading if required).
   
7. **Reproducibility**:
   - Set random seed (e.g., seed=42) for dataset shuffling and other stochastic processes.

---

### Implementation Steps:
1. **Initialization**:
   - Accept dataset name, split parameter, resize size, batch size, and seed.
   - Map dataset name to dataset class and relevant parameters.
   - Store transformations; include resizing, normalization, and conversion to tensors.

2. **Dataset Loading**:
   - Instantiate dataset object (e.g., `torchvision.datasets.CIFAR10`) with proper root directory, train/test flag, transform pipeline.
   - Apply splitting if necessary (most datasets are pre-split; if not, implement splitting logic).

3. **Transform Pipeline**:
   - Use `transforms.Compose()` including:
     - `transforms.Resize(target_size, interpolation=Image.BILINEAR)`
     - `transforms.ToTensor()`
     - Dataset-specific normalization; e.g., mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

4. **DataLoader Construction**:
   - Instantiate `DataLoader` with:
     - dataset object.
     - batch_size (from config).
     - shuffle=True for training, False for test.
     - drop_last=True or False depending on preference.
     - num_workers (set to a fixed number, e.g., 4, for efficient loading).

5. **Output**:
   - Return training and testing `DataLoader` objects suitable for downstream training/validation.

---

### Special Considerations:
- For datasets like UCF101 and SUN397 (video datasets), consider just using key frames or a representative static image during dataset loading unless specified otherwise.
- Ensure consistent normalization across datasets to align with pre-trained classifier expectations.
- Support potential future extension to datasets requiring custom loading procedures.

---

### Validation:
- Confirm dataset loading supports the specified `train_split` and `test_split`.
- Verify resizing preserves image aspect and resolution.
- Make sure `DataLoader` shuffles only during training.
- Test on multiple datasets with different input sizes as per `config.yaml`.
- Ensure reproducibility with fixed seed.

---

### Summary:
`DatasetLoader`'s key functions:
- Initialization with dataset name and parameters.
- Dataset class selection.
- Transformation setup including resizing.
- Data loader instantiation with batch size, shuffling, and seed control.
- Modular code structure enabling easy extension to new datasets.

This detailed plan will guide the implementation of a robust, flexible, and reproducible data handler compatible with the experiment setup detailed in the paper.

## evaluation.py

Evaluation.py Logic Analysis
==============================

Purpose:
--------
The `evaluation.py` module defines the `Evaluation` class, responsible for fully evaluating the performance of the reprogrammed dataset images on the fixed, pre-trained classifier. It manages inference, calculates accuracy, and optionally provides visualizations of reprogrammed images and masks.

Main Responsibilities:
----------------------
- Load or receive a trained classifier model (which is fixed/frozen).
- Accept a DataLoader for the test dataset (or validation set).
- Run inference with images processed via the reprogramming pipeline.
- Compute evaluation metrics, primarily classification accuracy.
- Generate visualizations, such as original images, reprogrammed images, applied masks, overlays, etc.

Key Components:
---------------
1. **Initialization (`__init__`)**:
   - Must receive the classifier model (`model`).
   - Should set up for evaluation metrics.
   - Possibly accept visualization configs (whether to visualize, number of images).

2. **Evaluation Method (`evaluate`)**:
   - Accepts a DataLoader (`test_loader`) with test images and labels.
   - Runs inference:
     - Supports batch processing.
     - Inputs images must undergo the reprogramming pipeline (resize, mask application, add pattern, etc.).
       - This pipeline was constructed earlier in training and is part of the overall system.
   - Collect predicted labels and true labels.
   - Calculate the chosen metric (`accuracy`).
   - Return performance metrics in a dictionary.

3. **Visualization (`visualize_results`)**:
   - If enabled, generates images for display:
     - Original input images.
     - Reprogrammed images (images plus pattern in masked regions).
     - Mask visualization (displaying the generated mask).
     - Overlay of mask over images.
   - Uses `matplotlib` for plotting.
   - Supports displaying a subset (e.g., first N images).

4. **Supporting Functions**:
   - Helper to run inference on a batch:
     - Input images are pre-processed according to model expectations.
     - Can include optional visualization.
   - Helper to compute metrics:
     - Accuracy is primary; can be extended.

Input Data:
-----------
- `test_loader`: DataLoader object containing dataset with images and labels.
- `model`: Pre-trained, fixed classification model (ResNet, ViT), already loaded and frozen.

Output:
-------
- Dictionary with evaluation metrics (accuracy).
- Optional images for visualization.

Implementation Details:
-----------------------
- Use `torch.no_grad()` during inference for efficiency.
- For each batch:
  - Move inputs and labels to appropriate device.
  - Run the classifier:
    - Inputs should be images processed by reprogramming pipeline (this step may be handled outside evaluation class). If not, reapply here.
  - Collect predicted labels.
- Calculate accuracy: (Number of correct predictions) / (total samples).
- Visuals:
  - For selected images, generate visualizations:
    - Plot original vs reprogrammed images.
    - Plot masks and overlays.
  - Can limit visualization to first few images for clarity.
- Return evaluation results.

Workflow:
---------
1. Instantiate `Evaluation` with fixed classifier model.
2. Call `evaluate(test_loader)`:
   - Perform inference, compute accuracy.
   - Trigger visualization if configured.
3. Present or log the performance metrics.

Edge Cases & Considerations:
---------------------------
- Ensure model is eval mode (`model.eval()`).
- Handle CUDA/CPU device compatibility.
- Provide a clear interface for passing images if the reprogramming pipeline is integrated.
- Support batch processing efficiently.
- Handle no-visualization cases gracefully.
- Maintain reproducibility if randomness affects visualization (e.g., select first N images).

Configuration:
--------------
- Visualization toggle (e.g., `visualize=True`).
- Number of images for visualization (default to small subset).
- Device configuration (automatic detection or specified in system).

In Summary:
-----------
The `Evaluation` class will be a simple, efficient wrapper that performs inference with the fixed classifier on images generated via the reprogramming pipeline, computes accuracy, and optionally visualizes key results. It should be designed for modularity, clarity, and efficiency, accommodating batch processing, device compatibility, and visualization requirements.

---

## main.py

# Logic Analysis for main.py

This module serves as the central orchestrator of the entire SMM-based VR experiment pipeline. Its core responsibilities include loading configurations, initializing datasets, models, modules for mask generation, pattern management, training, evaluation, and visualization, and coordinating the training and evaluation loops. The key points below outline the detailed logical flow and interactions necessary for correct and reproducible implementation:

---

## 1. **Configuration and Argument Handling**
- **Load Configuration:** Parse and load `config.yaml` using a YAML parser (`PyYAML` or similar).
- **Command-line Arguments:** Optional: allow overriding some parameters for flexibility, e.g., dataset selection, seed, or model architecture.
- **Set Random Seeds:** Use the seed value (`42`) from config for:
  - `torch.manual_seed()`
  - `np.random.seed()`
  - `torch.backends.cudnn.manual_seed_all()`
  
This ensures reproducibility across runs.

---

## 2. **Dataset Setup via DatasetLoader**
- **Instantiation:**
  - Extract dataset parameters from config:
    - `dataset.name`, `dataset.input_size`, `dataset.train_split`, `dataset.test_split`, `dataset.batch_size`.
  - Instantiate the `DatasetLoader` class, passing in these parameters.
- **Load Data:**
  - Call `load_data()` method to obtain:
    - `train_loader`: DataLoader for training data.
    - `val_loader`: Validation set DataLoader (or use test set if specified).
- **Dataset-specific preprocessing:**
  - Ensure images are resized to `sample_resize` (from config) before feeding into the pipeline.
  - Normalization: apply ImageNet mean/std normalization if necessary, matching pre-trained model requirements.
- **Data shuffling and batching**:
  - Set to shuffle training data, no shuffle for validation/test.
- **Logging:** Print dataset info, sizes, and some sample batch shapes.

---

## 3. **Model Initialization via Model Class**
- **Load Pretrained Model:**
  - Instantiate the Model class with:
    - `model.name` (e.g., ResNet50)
    - `pretrained=True` (from config).
- **Freeze Model:**
  - Set model.train(False) or freeze parameters explicitly:
    ```python
    for param in model.parameters():
        param.requires_grad = False
    ```
- **Validation of Setup:**
  - Confirm model outputs expected shape (batch_size × num_classes).
  - Confirm model is set to evaluation mode with `model.eval()`.

---

## 4. **Mask Generator Setup**
- **Instantiation:**
  - Read architecture parameters:
    - `mask_generator.architecture_depth`
    - `kernel_size`
    - `filters`
    - `pooling_layers`
  - Construct an instance of `MaskGenerator` class with these parameters.
- **Initialize weights:** according to `pattern_init` (zeros).
- **Parameter requiring gradients:**
  - Enable `phi` parameters for optimization.
- **Note:** The mask generation module should be in `train()` mode, with `requires_grad=True`.

---

## 5. **Pattern Initialization (Learnable Pattern $\delta$)**
- **Shape:**
  - Shape must match input image size:
    ```python
    shape = (channels, height, width)
    ```
  - From config:
    - `sampling.image_resize` defines size, e.g., [32, 32]
    - For channels: 3 (from dataset/model)
  - Initialize the pattern tensor:
    ```python
    delta = torch.zeros(shape, requires_grad=True)
    ```
- **Register as `torch.nn.Parameter`.**

---

## 6. **Optimizer Setup**
- **Optimize only** `{phi, delta}`:
  - Use an optimizer like Adam:
    ```python
    optimizer = torch.optim.Adam([
        {'params': mask_generator.parameters(), 'lr': config.training.pattern_lr},
        {'params': delta, 'lr': config.training.pattern_lr}
    ], weight_decay=1e-4)
    ```
- **Learning rate scheduling:**
  - Implement step decay based on `lr_decay_epochs`:
    - For mask generator (`phi`)
    - For pattern (`delta`)
  - Or use `torch.optim.lr_scheduler.MultiStepLR` with scheduled milestones.

---

## 7. **Training Loop (Epochs)**
- For each epoch (total `config.training.epochs`):
  - **Set model.eval()** (fixed pre-trained backbone).
  - **Iterate over train_loader:**
    - For each batch:
      1. Transfer images and labels to device (GPU/CPU).
      2. Resize images to `sampling.image_resize`.
      3. Generate masks:
         - Pass resized images through `MaskGenerator`:
           ```python
           masks = mask_generator(images)
           ```
         - Upsample masks via patch-wise repetition:
           - For patch size `patch_size`:
             ```python
             masks_upsampled = masks.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
             ```
      4. Apply pattern:
         - Pixel-wise multiply $\delta$ with the mask:
           ```python
           masked_pattern = delta * masks_upsampled
           ```
         - Add to resized images:
           ```python
           reprogrammed_inputs = resized_images + masked_pattern
           ```
         - (Optional) Clip/normalize if needed.
      5. Forward:
         - Input `reprogrammed_inputs` to the frozen model:
           ```python
           outputs = model(reprogrammed_inputs)
           ```
      6. Compute loss:
         - CrossEntropyLoss between `outputs` and labels.
      7. Backward:
         - Zero gradients.
         - Backpropagate loss.
         - Update only `phi` and `delta` parameters.
  - **Validation/Training metrics:**
    - Accumulate loss and accuracy.
    - Save best model parameters if required.
  - **Update learning schedule** at milestones.

---

## 8. **Evaluation**
- After training:
  - Switch model to evaluation mode.
  - Process full test dataset:
    - Resize, generate masks, apply pattern, run through the classifier.
    - Compute accuracy.
  - **Optional visualization:**
    - Use visualization.py to display original images, reprogrammed images, masks, and overlay masks.
    - Save sample figures for analysis.

---

## 9. **Logging and Saving**
- Save trained pattern `delta` and mask generator `phi` states via `torch.save`.
- Log training/validation/test accuracy, losses, and hyperparameters.
- Save visualization outputs if `visualize` flag is true.

---

## 10. **Reproducibility & Finalization**
- Save all hyperparameters, seed, and environment info.
- Document any deviations or experimental notes.
- Wait for completion; report final metrics.

---

## Summary:
- **Sequentially** initializes components.
- **Iteratively** updates mask generator and pattern (only).
- **Evaluates** with fixed pre-trained model.
- **Visualizes** key outputs.
- **Configurable**, flexible, with hyperparameters from YAML.

This detailed logic ensures a faithful, reproducible implementation aligned with the methodology in the paper, integrating all design components and experimental configurations.

## mask_generator.py

# Logic Analysis for `mask_generator.py`

The purpose of this module is to define a `MaskGenerator` class responsible for creating sample-specific masks from images, which serve as part of the visual reprogramming framework. The masks are generated via a lightweight CNN, which should be configurable in terms of depth, number of filters, pooling layers, and output resolution, aligning with the settings in the provided configuration.

Below is a detailed, step-by-step logical breakdown of the requirements, design considerations, and implementation details for this class:

---

### 1. **Inputs and Initialization**

- **Input Parameters:**
  - `architecture_depth` (int): Number of convolutional layers in the CNN. Typically, 5 or 6 layers based on dataset complexity.
  - `kernel_size` (int): Size of the convolution kernels, e.g., 3.
  - `filters` (int): Number of filters in each convolutional layer, e.g., 64.
  - `pooling_layers` (int): Number of MaxPooling layers to reduce spatial resolution. For example, 2.
  - `output_ratio` (float): Ratio of the output mask size to input image size, e.g., 1/8 (or 0.125). This determines the size of the generated mask before patch-wise upsampling.
- **Model Components:**
  - A sequence of convolutional layers with ReLU activations and batch normalization (or instance normalization if desired for stability).
  - Transition layers: After certain convolutional layers, optional normalization + ReLU + MaxPooling, respecting `pooling_layers`.
  - Final convolution layer with 3 output channels, corresponding to the three-channel mask.
- **Device Handling:**
  - Ensure model is moved to GPU if available.

---

### 2. **Design of the CNN Architecture**

- **Layer Construction:**
  - Starting with an input image tensor of shape `(batch_size, 3, H, W)`.
  - Build a sequential stack of `architecture_depth` convolutional layers:
    - Each convolution: `nn.Conv2d(in_channels, filters, kernel_size, padding=1)` to maintain size, with `ReLU` activation.
    - Optional normalization layer: `nn.BatchNorm2d(filters)` for stability.
  - After each convolutional layer, optionally apply max pooling:
    - Use `nn.MaxPool2d(kernel_size=2, stride=2)` for spatial downsampling.
    - Limit the number of pooling layers to prevent excessive size reduction.
  - Final layer: `nn.Conv2d(filters, 3, kernel_size=3, padding=1)` for generating 3-channel masks.

- **Size Calculations:**
  - Input size: (H, W), e.g., 32×32 or 224×224.
  - After `l` pooling layers, spatial dimensions reduce approximately by `2^l`. For example, with 2 pooling layers: `(H / 4, W / 4)`.
  - Output size of final CNN: `(batch_size, 3, H', W')`, where `(H', W')` = `(floor(H / 2^l), floor(W / 2^l))`.
  - These dimensions are determined by the input image size and `pooling_layers`.

- **Parameter Initialization:**
  - Use Kaiming Uniform or Normal initialization for convolution weights.
  - Biases initialized to zero.
  - BatchNorm layers initialized with weight=1 and bias=0 for stable training.

---

### 3. **Sample-specific Mask Generation**

- **Forward pass:**
  - Input: `images` tensor (batch of resized images).
  - Output: Mask tensor of shape `(batch_size, 3, H', W')`.
 
- **Patch-wise Upsampling (Interpolate):**
  - Since the CNN produces a smaller mask, perform upsampling via pixel repetition:
    - For each spatial dimension `(H', W')`, repeat each pixel `patch_size` times both vertically and horizontally.
    - Example: `tile` operation in NumPy or `repeat_interleave` in PyTorch.
  - `patch_size` = `input_size / output_mask_size`, e.g., 8 if mask size is 1/8 of input size.
  - No gradients are propagated through the patch-wise interpolation step; it is a simple operation to resize the mask.

---

### 4. **Implementation Details**

- **Class Structure:**
  - Constructor (`__init__`): builds the CNN module according to provided parameters.
  - `forward(image: Tensor) -> Tensor`: 
    - Accepts a batch of images.
    - Passes images through CNN to get low-res masks.
    - Performs patch-wise upsampling via repeating pixels.
    - Returns the final masks aligned with image size.

- **Parameter Sharing:**
  - The CNN's weights are shared and learned over the dataset.
  - Each call to `forward` processes individual images uniformly.

- **Training Integration:**
  - The CNN parameters receive gradients during training.
  - Model is trained jointly with the pattern $\delta$ for the entire dataset.

---

### 5. **Customization & Hyperparameters**

- **Flexibility:**
  - Support variable `architecture_depth` to tailor complexity.
  - Support variable `pooling_layers` affecting output size.
  - Support variable `filters` for model capacity.
  - Support variable `patch_size` (or compute based on `output_ratio`) for the resolution of masks.

- **Size Consistency:**
  - Use input image size info to compute output mask size.
  - Enforce size constraints: `H' = floor(H / 2^l)`, `W' = floor(W / 2^l)`.

---

### 6. **Testing & Validation**

- Validate size output for different input image sizes.
- Confirm the upsampling step generates masks that match the input images exactly.
- Visualize generated masks to ensure they are meaningful and sample-specific.
- Verify that the CNN overfits on small datasets, indicating proper training.

---

### Summary

The `MaskGenerator` class must:
- Be a lightweight, configurable CNN.
- Generate masks at reduced resolution, then upsample via pixel-repetition.
- Support different CNN depths, filter sizes, and pooling layers.
- Produce masks that match the input images' spatial dimensions when combined with the patch-wise upsampling.
- Be integrated into the training process for end-to-end jointly learned mask generation.

This detailed logic serves as a comprehensive foundation for precise implementation, ensuring fidelity to the original design and effectiveness within the SMM framework.

## model.py

# Logic Analysis for model.py

This module is responsible for defining a `Model` class that encapsulates the loading, initialization, and inference functionalities of pre-trained backbone models such as ResNet or ViT. It will serve as a core component in the pipeline, providing a standardized interface for model inference while keeping the backbone frozen during training of the prompt pattern and mask generator.

---

## 1. Objectives and Responsibilities

- **Model Initialization:**
  - Load a pre-trained model specified in configuration.
  - Support for multiple architectures (e.g., ResNet50, ViT-B32).
  - Use pretrained weights, as indicated by `pretrained: true`.
  - Adjust input layer or configurations if necessary, to match dataset input size (`input_size` from config).

- **Freezing the Backbone:**
  - Set all parameters of the loaded model to `requires_grad=False`.
  - Ensure no gradient updates occur within the pre-trained backbone during training.

- **Inference Interface:**
  - Implement a `forward()` method:
    - Accept an input tensor `x` (batch of images).
    - Return model predictions (logits or softmax outputs as needed).
  - Facilitate evaluation or testing phases where backbone remains untrainable, but outputs are needed.

---

## 2. Input and Output Specifications

- **Input:**
  - `x`: torch.Tensor, shape `[batch_size, channels, height, width]`.
  - Data normalization: For models trained on ImageNet, apply normalization (mean/std) consistent with pretraining. This normalization step can be handled externally for flexibility, but if integrated here, include normalization.
  - Input images are expected to be resized appropriately before passing in.

- **Output:**
  - `logits`: torch.Tensor, shape `[batch_size, num_classes]`.
  - The raw output of the model’s final linear layer, not passed through softmax internally.
  - The inference pipeline will handle interpreting outputs (e.g., argmax for predicted class).

---

## 3. Model Loading & Compatibility

- **Framework:**
  - Use PyTorch's `torchvision.models` for ResNet architectures.
  - Use `timm` library or `torchvision` for ViT models if necessary; but since only `torchvision` is assumed, verify model availability.
  
- **Supported Models:**
  - Identify supported model names from the config (e.g., `"ResNet50"`, `"ViT-B32"`).

- **Loading Weights:**
  - Utilize `torchvision.models.<model>(pretrained=True)` if supported.
  - For ViT, may need to use `torchvision` or third-party model zoo; if not supported natively, document that custom loading might be needed.
  
- **Input Size Adjustments:**
  - Confirm if preprocessing or input resizing occurs outside the model class; if not, include normalization as part of the model’s preprocessing pipeline or assume input is preprocessed.

---

## 4. Implementation Details

- **Initialization:**
  - Instantiate the specified backbone model with `pretrained=True`.
  - Possibly include a `model_name` or `architecture` parameter for flexibility.
  - If the dataset input size differs from default (e.g., 224×224 for ResNet), assume input images are already resized.
  - Save the model as an attribute for future inference.

- **Freezing Parameters:**
  - Loop through `model.parameters()`:
    - Set `param.requires_grad = False` to prevent updates during optimizer steps.
  - This ensures only the prompt pattern and mask generator are trained.

- **Forward Method:**
  - Receive input tensor `x`.
  - Forward `x` through the model.
  - Return output logits directly.
  - No modification of outputs is required here.

- **Additional Considerations:**
  - Support for models' `.eval()` mode for inference.
  - Optionally, include `.to(device)` handling, but training script typically manages device placement.
  - Could include attribute for number of output classes.

---

## 5. Error Handling and Validation

- **Model support validation:**
  - Check if the model name in config is among supported architectures.
  - Raise informative errors if unsupported.

- **Input shape validation:**
  - Ensure inputs are 4D tensors `[batch_size, channels, height, width]`.
  - Confirm that input channels match expected (e.g., 3 channels). 

---

## 6. Summary of Implementation Logic

- Parse configuration parameters (`model.name`, `model.pretrained`).
- Load the specified pre-trained model.
- Set all model parameters to frozen (`requires_grad=False`).
- Define `forward()` to perform inference.
- Provide interface for the trained object to be used externally.

---

## 7. Example Pseudocode for Implementation:

```python
import torch
import torchvision.models as models

class Model:
    def __init__(self, model_name: str, pretrained: bool=True):
        # Load pre-trained model
        if model_name == "ResNet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == "ResNet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "ViT-B32":
            # Load ViT-B32 model appropriately (assuming availability)
            # Possibly from torchvision.models or another source
            self.model = ... # custom load or from timm
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Freeze backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Optional: set to eval mode
        self.model.eval()

        # Save device info if needed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: preprocessed tensor
        with torch.no_grad():
            outputs = self.model(x)
        return outputs
```

---

## 8. Final Notes

- Ensure dataset input images are resized to match expected input size.
- Implement normalization consistent with pretraining.
- Possibly extend the class to support switching between training/test modes, although for inference only, `.eval()` mode is suitable.
- Document the supported architectures explicitly for maintainability and extension.

---

This comprehensive logic analysis provides the blueprint to implement the `Model` class per the paper's specifications, ensuring compatibility, correctness, and clarity.

## pattern.py

# Logic Analysis for `pattern.py` — Definition of the Pattern Class

The `Pattern` class is responsible for representing the learnable prompt pattern, denoted as $\delta$, which is a trainable tensor shared across all samples. Its core functionality involves initializing this tensor, providing access for training updates, and ensuring it is correctly integrated into the overall training pipeline for visual reprogramming.

Below is a detailed, step-by-step logical breakdown of its design considerations, interface, attributes, and behavior, aligned with the approach and specifications described in the paper and the provided configuration.

---

## 1. **Purpose & Role**

- **Main Role:** Manage the `delta` pattern tensor, which:
  - Is a trainable parameter (`torch.nn.Parameter`).
  - Has the same spatial dimensions as the input images ($H \times W \times C$).
  - Is shared across all samples, i.e., it is a single tensor continuously optimized during training.
  - Is initialized according to the configuration (zeros, in this case).

- **Usage within training:** During each training iteration, this pattern will be retrieved and applied pixel-wise in the mask application step, where it is multiplied with the generated sample-specific masks and added to resized images.

## 2. **Design Considerations**

- **Initialization:**
  - As per the configuration (`pattern_init: zeros`), initialize the pattern tensor as zeros.
  - The tensor should have shape `[C, H, W]` or `[H, W, C]`. Considering PyTorch conventions, best to choose `[C, H, W]`.
  - Pattern shape matches the input image size after resizing; the dimensions are specified as `[height, width, channels]`, so for PyTorch, convert as needed.

- **Parameters and Gradient:**
  - The pattern tensor must be a `torch.nn.Parameter`, enabling gradient updates during training.
  - Keep it as a module attribute for accessibility.

- **Encapsulation:**
  - The class will be a `torch.nn.Module` subclass, providing:
    - Initialization method (`__init__`)
    - Method to return the current pattern tensor (`get_pattern`)
    - (Optionally) a method to reset or reinitialize (not explicitly required, but useful for experimentation)

- **Seamless Integration:**
  - The class should be compatible with optimizer, meaning the pattern tensor is directly passed into an optimizer.
  - Designed to be flexible if, in future, pattern shape needs adjustment.

## 3. **Interface Specification**

- **Constructor (`__init__`)**:
  - Input parameters:
    - `shape`: tuple specifying `[channels, height, width]`.
    - Initialization type: defaults to zeros, configurable if needed.
  - Actions:
    - Instantiate a `torch.nn.Parameter` tensor with zeros.
    - Register it as `self.pattern`.

- **Attributes**:
  - `self.pattern`: `torch.nn.Parameter` of shape `[C, H, W]`.

- **Methods**:
  - `get_pattern()`:
    - Returns `self.pattern`.
  - (Optional) `reset()`:
    - Reinitializes pattern data to zeros or other schemes.

## 4. **Implementation Details & Best Practices**

- Use `torch.zeros()` to initialize pattern data.
- Wrap in `torch.nn.Parameter()` to ensure correct registration with optimizer.
- Consider device placement: pattern tensor should follow device (CPU/GPU) of the model.
- Ensure the pattern is a leaf tensor (which `torch.nn.Parameter` guarantees).

## 5. **Usage in Training Pipeline**

- In the main training script:
  - Instantiate the `Pattern` class with size `[channels, height, width]`.
  - Pass `pattern.pattern` to the optimizer.
  - During each iteration:
    - Retrieve `pattern.pattern`.
    - Use it in `apply_mask_to_image()` operation:
      ```
      masked_pattern = pattern.pattern
      ...
      reprogrammed_image = resized_image + (masked_pattern * generated_mask)
      ```
    - Compute loss on the classifier output.
  - During backpropagation:
    - Only updates `pattern.pattern` (via optimizer).

## 6. **Alignment with Configuration & Paper Details**

- The shape of the pattern tensor is determined by:
  - Resize target size (`H`, `W`), which depend on dataset (`dataset.input_size`) — e.g., `[32, 32]`.
  - Number of channels (`C`) — e.g., 3 for RGB, as per dataset.
- The pattern is initialized to zeros (`pattern_init: zeros`).
- It is a shared parameter across the dataset in training.

---

## 7. **Summary of `Pattern` Class**

| Aspect | Description |
|---------|----------------------------------------------|
| Purpose | Maintain the trainable prompt pattern (`delta`) for prompt-based reprogramming. |
| Initialization | Zero tensor of shape `[channels, height, width]`. |
| Data Type | `torch.nn.Parameter`. |
| Methods | `__init__()`, `get_pattern()` (returns pattern tensor), optional `reset()`. |
| Usage | Passed to optimizer; used in constructing reprogrammed inputs during each iteration. |

---

This comprehensive analysis ensures that the `Pattern` class adheres to the conceptual and technical specifications laid out in the paper and plan, providing clear guidance for precise, faithful implementation.

## requirements.txt

# requirements.txt

# The following delineates the core logic and structural components necessary for implementing the SMM framework  
# per the paper's methodology, experimental setup, and your provided configuration. This analysis emphasizes the dependencies, 
# class interactions, data flows, key functions, hyperparameters, and procedural steps imperative to achieve faithful reproduction.

---

# 1. Dataset Loading and Preprocessing
- Load datasets such as CIFAR-10, with proper train/test splits.
- Resize input images to target input size specified in `config.yaml` (`sample/image_resize`).
- Normalize images consistent with pre-trained model expectations (e.g., ImageNet mean/std for ResNet/VIT).
- Generate DataLoader objects with batch size from `config.yaml`.
- Handle dataset-specific peculiarities (e.g., color space, data augmentation if applicable).

# 2. Model Initialization
- Load pre-trained backbone model (`ResNet50`) with `pretrained=True`.
- Confirmation that model weights are frozen (disable training on backbone), per the methodology.
- Ensure the model outputs logits, compatible with cross-entropy loss.
- Maintain a clear interface for forwarding the images.

# 3. Pattern and Mask Generation Modules
- Implement `MaskGenerator` class:
  - Use configurable depth (e.g., 5 layers) specified by `mask_generator.architecture_depth`.
  - Constructed with 3×3 convolutions, filters as per `mask_generator.filters`.
  - Include 2 max-pooling layers (`mask_generator.pooling_layers`) for resolution reduction.
  - Final layer outputs 3-channel mask tensor at reduced resolution (`output_ratio`).
  - Support batch processing of resized images.
- Implement patch-wise upsampling:
  - Upsample generated low-res masks to original input size by pixel repetition.
  - No interpolated weights; merely tile each pixel into an `patch_size×patch_size`.
  - This is code-level logic rather than a third-party library.

- Mask parameter tensor:
  - Managed as trainable `torch.nn.Parameter` (size matching input resolution).
  - Initialization: zeros, as per `Pattern pattern_init`.
  - Shared across all samples.

# 4. Training Loop
- Iteratively optimize:
  - Mask generator parameters `phi`.
  - Pattern tensor `delta`.
- For each batch:
  - Resize images as per configuration.
  - Generate sample-specific masks:
    - Pass each resized image through `MaskGenerator`.
    - Upsample via pixel repetition (no gradient backpropagation through interpolation).
  - Pixel-wise multiply `delta` (learnable pattern) with generated mask.
  - Add resulting pattern-mask to resized input image, producing reprogrammed input.
- Forward reprogrammed input through frozen pre-trained model:
  - Capture model output logits.
- Compute cross-entropy loss with target labels.
- Backpropagation:
  - Update `phi` and `delta` using optimizer (Adam), with learning rate schedules.
- Learning rate schedules:
  - Decay at predefined epochs (`lr_decay_epochs` and `pattern_lr_decay_epochs`).
  - Pattern pattern updates with lower learning rate (`pattern_lr`).
- Regularization and stability:
  - Apply weight decay.
  - Clip or normalize pattern if necessary to prevent overfitting.

# 5. Optimization & Hyperparameters
- Use `torch.optim.Adam` or as specified.
- Set initial learning rate from `training.learning_rate`.
- Schedule decay factor at specified epochs.
- Train for total epochs (`training.epochs`, e.g., 200).
- Use seed `reproducibility.seed` for deterministic runs.

# 6. Evaluation & Visualization
- After training:
  - Run inference with the fixed, pre-trained classifier on test set.
  - Compute accuracy metrics.
  - Store and possibly visualize reprogrammed images and masks:
    - Display original vs. reprogrammed images.
    - Visualize generated masks (heatmaps, overlays).
- Visualization enabled if `evaluation.visualize` is true.
- Use `matplotlib`, `Pillow` for plotting and image saving.

# 7. Supporting Utilities
- Image resizing functions:
  - Bilinear interpolation during initial resize; patch-wise upsampling via pixel tiling.
- Tensor normalization functions.
- Save and load state dictionaries of `MaskGenerator` and pattern tensor.
- Track and log training curves and accuracy scores.

# 8. Reproducibility & Configuration
- Set random seed throughout the codebase.
- Follow configuration parameters strictly:
  - Input size
  - Model architecture
  - Hyperparameters for optimizer and schedule
  - Dataset parameters
- Maintain consistent environment with specified package versions.

# 9. Additional Modules (Optional Extensions)
- Implement `evaluation.py` for rigorous testing.
- Implement visualization routines.
- Build `main.py` as orchestrator:
  - Parse config.
  - Instantiate dataset, model, generator, pattern.
  - Run training.
  - Run evaluation.
  - Visualize results.

---

# Summary of Dependencies and Libraries
- **PyTorch (`torch`)**: core tensor operations, module definition, gradient updates.
- **Torchvision (`torchvision`)**: datasets, transforms, pre-trained models.
- **NumPy**: numerical array manipulations.
- **Matplotlib**: visualization of images, masks, feature spaces.
- **Pillow**: image I/O.
- **SciPy**: optional for advanced image processing if needed.

---

# Note
The code must rigorously follow the "Data structures and interfaces" to ensure modularity and clarity, encapsulating each module's responsibility, especially regarding mask generation, image resizing, and training. No external interpolation libraries (e.g., OpenCV, skimage) necessary beyond built-in PyTorch or NumPy functions for pixel repetitions.

---

This comprehensive requirements.txt-style logic analysis provides a thorough blueprint for implementation, ensuring the fidelity of the experimental methodology, sample-specific mask generation, training regimes, hyperparameters, and evaluation protocols as per the paper description and configuration.

## trainer.py

# Logic Analysis for trainer.py

## Overview
The goal of `trainer.py` is to define a `Trainer` class responsible for orchestrating the training loop for the proposed SMM (Sample-specific Multi-channel Mask) framework. It manages the update of two core trainable components:

- The mask generator parameters, denoted as $\phi$ (implemented as the parameters of the `MaskGenerator` class).
- The shared learnable pattern, $\delta$, which is a `torch.nn.Parameter`.

The `Trainer` class leverages a fixed, pre-trained classifier (e.g., ResNet50 or ViT), which remains frozen during training.

The objective function is to minimize over the dataset the empirical cross-entropy loss between the classifier's outputs on the reprogrammed images and the target labels, updating only $\phi$ and $\delta$.

Furthermore, the `Trainer` supports specifying learning rate schedules via decay epochs, weight decay regularization, as well as optional visualization of reprogrammed images and masks.

---

## Core Components & Data Flow
1. **Initialization Inputs:**
   - Model: pre-trained, frozen classifier.
   - Mask generator: an instance of `MaskGenerator`.
   - Pattern: a tensor (e.g., shape `[H, W, 3]`), initialized as zeros (`pattern_init` from config).
   - Data loaders: `train_loader` (training data), optional `val_loader` for validation.
   - Hyperparameters: learning rates, decay schedules, epochs, weight decay, visualization flags.

2. **Per-epoch Loop:**
   - For each epoch:
     - Possibly adjust learning rates according to the schedule.
     - Loop over training data:
       - For each batch:
         - Resize images to input size (from config) if necessary.
         - Generate masks for each image via the mask generator.
         - Upsample masks via the patch-wise interpolation method.
         - Element-wise multiply patterns $\delta$ with the generated masks.
         - Add the masked pattern to the resized images.
         - Forward pass the resulting images through the fixed classifier.
         - Compute the cross-entropy loss against labels.
         - Backpropagate loss to update only $\phi$ (mask generator params) and $\delta$.
     - Compute and record training metrics.

3. **Parameter Updates:**
   - Since only $\phi$ and $\delta$ are trainable, the optimizer is configured to only optimize these.
   - Implement learning rate decay at specified epochs, if applicable.
   - Regularization: weight decay may be employed (e.g., 1e-4).

4. **Additional Considerations:**
   - Initialization of $\delta$ prior to training (zeros).
   - Support for multiple optimizer steps per epoch, with gradient accumulation if desired.
   - Save best model parameters based on validation metrics, if validation set is used.
   - Visualization routines (optional) invoked at logging intervals, showing:
     - Original images.
     - Reprogrammed images.
     - Mask images.
     - Overlay visualizations.

---

## Implementation Details & Step-by-Step

### 1. Class Constructor (`__init__`)
- Store references to the fixed model, mask generator, pattern tensor.
- Initialize optimizer:
  - Use `torch.optim.Adam` (as per config) on `[mask_generator.parameters(), pattern]`.
  - Set learning rates considered for $\phi$ and $\delta$ (`pattern_lr`, `phi_lr`).
- Setup LR scheduler if decay steps specified.
- Store hyperparameters:
  - Total epochs.
  - Decay epochs.
  - Regularization parameters.
  - Visualization flag.

### 2. Training Method (`train`)
- Loop over epochs:
  - Adjust optimizer LR if scheduled.
  - For each batch from `train_loader`:
    - Extract images, labels.
    - Resize images to the model's expected input size (via bilinear resize).
    - Clear gradients.
    
    - **Mask Generation & Upsampling:**
      - For each image in batch:
        - Generate mask patch using `MaskGenerator.generate_mask(image)`.
        - Upsample mask via patch repetition to match input size.
        - (Optional) convert mask to desired shape `[H, W, 3]`.
    
    - **Pattern Application:**
      - Expand pattern $\delta$ to batch size (broadcast).
      - Element-wise multiply masks with pattern.
      - Add resulting structured pattern onto resized images.
    
    - **Forward Pass & Loss:**
      - Pass the reprogrammed images through the fixed classifier (no gradient update).
      - Compute cross-entropy loss with the labels.
    
    - **Backward & Optimization:**
      - Backpropagate loss.
      - Update only $\phi$ and $\delta$ parameters.
    
    - Accumulate loss and accuracy metrics.
  
  - **LR Decay & Logging:**
    - Decay learning rate if epoch matches decay schedule.
    - Log training metrics (loss, accuracy).
    - If validation set used, evaluate and save best model.

### 3. Additional Methods
- **Learning rate adjustment**:
  - Check epochs against decay schedule to multiply LR.
- **Visualization**:
  - Generate and display images/masks as per flags.
  - Save images for further inspection if needed.

### 4. Finalization:
- After training completes:
  - Save final $\phi$ and $\delta$.
  - (Optionally) produce final reprogrammed images for reporting.
  - Return or store training metrics.

---

## Hyperparameter & Implementation Notes:
- **Learning Rates:** as specified (pattern_lr=0.001, model LR=0.01).
- **Decay Strategy:** decay at epochs (e.g., 100) by factor (e.g., 0.1).
- **Batch Size:** match from dataset loader.
- **Pattern Initialization:** zeros.
- **Optimizer:** Adam with weight decay.
- **Gradient Masking:** ensure only $\phi$ and $\delta$ are included in optimizer params.

---

## Summary
The `Trainer` class:
- Manages a training loop over many epochs.
- For each batch:
  - Resize images.
  - Generate sample-specific masks with the lightweight CNN.
  - Upsample masks pixel-wise (patch repetition).
  - Mask the learnable pattern $\delta$.
  - Add to images.
  - Forward, compute loss, backpropagate.
- Update only mask generator params and pattern.
- Apply LR schedules.
- Record metrics, perform visualizations if configured.
- Save best models and final parameters for downstream evaluation.

This mixed process ensures a sample-adaptive prompt mechanism, leveraging the theoretical guarantees of reduced approximation error from the paper.

---

This completes the thorough logic analysis for `trainer.py`.

## visualization.py

{
  "Visualization.py - Logic Analysis": [
    {
      "Component": "Purpose",
      "Description": "Implement functions to visualize original input images, reprogrammed images, generated masks, and overlay visualizations showing how masks and patterns modify images. These visualizations aid qualitative assessment of the reprogramming process, mask quality, and learned patterns."
    },
    {
      "Component": "Input Data Types",
      "Description": "Functions will primarily handle torch.Tensors and numpy arrays representing images and masks:"
    },
      {
        "Images": "Tensor of shape (C, H, W), typically normalized (e.g., ImageNet stats); support conversion to displayable format."
      },
      {
        "Masks": "Tensor of shape (H, W, 3) or (C, H, W), depending on generation; representing per-pixel mask values, possibly with values (0 to 1) or raw pattern values."
      }
    ],
    {
      "Component": "Main Functions",
      "Description": [
        {
          "show_original_image": {
            "Purpose": "Display original input image to compare baseline inputs against reprogrammed outputs.",
            "Inputs": ["image_tensor: torch.Tensor", "title: str (optional)"],
            "Operations": [
              "Convert tensor to numpy array and move channels to last dimension if needed.",
              "De-normalize if normalization was applied during preprocessing (if applicable).",
              "Use matplotlib.pyplot.imshow() to display image.",
              "Add title if provided.",
              "Disable axis for clarity."
            ],
            "Output": "Display window with original image."
          }
        },
        {
          "show_reprogrammed_image": {
            "Purpose": "Display image after adding mask-guided pattern, to visualize the effect of reprogramming.",
            "Inputs": ["reprogrammed_image: torch.Tensor", "title: str (optional)"],
            "Operations": [
              "Convert tensor to numpy array as above.",
              "Ensure pixel values are in displayable range [0,1] or [0,255].",
              "Use plt.imshow() and plt.axis('off').",
              "Add optional title."
            ],
            "Output": "Display window with reprogrammed image."
          }
        },
        {
          "show_mask": {
            "Purpose": "Display the generated mask tensor, highlighting which regions are active or masked.",
            "Inputs": ["mask_tensor: torch.Tensor", "title: str (optional)"],
            "Operations": [
              "Convert to numpy array if needed.",
              "Normalize or clip mask values to [0,1] for visualization.",
              "Use plt.imshow(), possibly with a colormap ('gray' or 'viridis') for better contrast.",
              "Remove axis for clarity.",
              "Add title if provided."
            ],
            "Output": "Display window with mask heatmap."
          }
        },
        {
          "show_mask_overlay": {
            "Purpose": "Visualize the mask overlayed on the original image to see impact regions of masking or pattern addition.",
            "Inputs": ["original_image: torch.Tensor", "mask: torch.Tensor", "alpha: float (transparency of overlay, e.g., 0.5)", "title: str (optional)"],
            "Operations": [
              "Convert both images and mask to numpy.",
              "Normalize images as needed.",
              "Create an overlay by blending original image and mask heatmap (e.g., using cv2.addWeighted or matplotlib blending).",
              "Display with plt.imshow(), axis off.",
              "Optional: add colorbar or legend if mask indicates specific regions."
            ],
            "Output": "Display window showing original image with overlay mask highlighting pattern regions."
          }
        }
      ]
    },
    {
      "Component": "Supporting Utilities",
      "Description": [
        {
          "convert_tensor_to_np": {
            "Function": "Helper function to standardize conversion of torch.Tensor to numpy array suitable for matplotlib.",
            "Operations": [
              "Handle potential normalization (mean/std) removal.",
              "Transpose channels from (C, H, W) to (H, W, C) if needed.",
              "Clip or scale values to [0,1] for display."
            ]
          }
        }
      ]
    },
    {
      "Component": "Additional Considerations",
      "Description": [
        {
          "Color Map Choice": "Use grayscale ('gray') for masks; consider 'viridis' or 'hot' for heatmaps.",
          "Range Handling": "Ensure all tensors are scaled between 0 and 1 before display.",
          "Batch Handling": "Design functions to handle both single-image and batch inputs (can loop over batch).",
          "Visualization Layout": "Optionally allow side-by-side comparison of original, mask, overlay images."
        }
      ]
    },
    {
      "Component": "Integration",
      "Description": "Ensure visualization functions are compatible with the training pipeline, invoked to analyze masks and images at different training stages, and support quick iteration for debugging and qualitative assessment."
    },
    {
      "Component": "Summary",
      "Description": "Implement modular, generalized visualization functions that accept tensors and display images/masks overlayed with minimal assumptions about data range or shape, facilitating visual inspection of masks, images, and their interactions during sample reprogramming."
    }
  ]
}

