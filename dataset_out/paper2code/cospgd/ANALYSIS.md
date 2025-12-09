# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## attack.py

{
  "attack.py": [
    "Class Purpose and Overview",
    "The Attack class encapsulates the entire attack process, managing initialization, iteration, loss computation, gradient calculation, input updating, and clipping. It must support both targeted and untargeted attack modes, pixel-wise tasks, and different norms (primarily l_infinity, but structure allows extension to l_2).",
    "Key methods include:",
    "  - __init__: sets hyperparameters, model, task type, attack mode, and scaling functions.",
    "  - attack: executes the iterative attack process, returning the adversarial example.",
    "  - initialize: creates initial adversarial perturbation within epsilon bounds.",
    "  - compute_scaled_loss: computes pixel-wise loss scaled by cosine similarity (alignment score) with the target prediction or ground truth.",
    "  - update_input: takes the sign of the gradient, scales with step size, and applies to current input, then clips the resulting adversarial example to within the epsilon ball and valid data range.",
    "  - clip: enforces pixel value constraints ([0,1] for images) and epsilon norm bounds.",
    "General flow within attack():",
    "1. Initialize \(X^{adv}\) with small random noise within epsilon bounds.",
    "2. For each iteration up to max_iters:",
    "   a. Enable gradients on \(\mathbf{X}^{adv}\).",
    "   b. Forward pass: compute prediction P = f_θ(X^{adv}).",
    "   c. Compute \(\psi(P)\): softmax normalization per pixel (for classification) or identity for regression.",
    "   d. Compute cosine similarity score for each pixel: \(\cos( \psi(P)_i , Y_i )\).",
    "   e. Compute scaled pixel-wise loss using compute_scaled_loss().",
    "   f. Backward pass: compute gradient of the scaled loss w.r.t. \(\mathbf{X}^{adv}\).",
    "   g. Update \(\mathbf{X}^{adv}\) by taking a step in the direction of the sign of the gradient: \(\mathbf{X}^{adv} \leftarrow \mathbf{X}^{adv} + \alpha \cdot \text{sign}(\nabla_{\mathbf{X}^{adv}})\).",
    "   h. Clip \(\mathbf{X}^{adv}\) to stay within the epsilon ball around \(X^{clean}\), and clip pixel values to valid range (e.g. [0,1]), ensuring the adversarial example remains valid.",
    "3. Return the final \(\mathbf{X}^{adv}\).",
    "Key Details and Implementation Aspects:",
    "- Use PyTorch autograd functions, with torch enables gradients on \(\mathbf{X}^{adv}\).",
    "- At each iteration, the gradient should be zeroed before backward pass.",
    "- The sign operation ensures a fast gradient sign method (FGSM-like update).",
    "- Loss scaling involves the per-pixel cosine similarity multiplied by the pixel-wise loss (e.g., cross-entropy).",
    "- For numerical stability: ensure norm calculations have a small epsilon (e.g., 1e-8) to prevent division by zero.",
    "- For targeted attacks, modify the cosine similarity from \(\cos(\cdots)\) to \(1 - \cos(\cdots)\) to invert the scaling effect, pulling predictions towards the target.",
    "- For clipping operations, use torch.clamp() for pixel values ([0,1]) and element-wise clipping relative to the original clean image for epsilon bounds.",
    "- To implement the epsilon constraint for \(\ell_\infty\), keep track of the initial clean input and clip differences for each pixel.",
    "- For future extension to \(\ell_2\) bounds, adapt the clip method to project to the \(\ell_2\) ball (requires computing the Euclidean norm of the perturbation).",
    "Additional considerations:",
    "- Incorporate optional verbose logging or plotting per iteration if needed for debugging.",
    "- Model outputs: logits before softmax; target labels: class indices, use one-hot for softmax and cosine calculation.",
    "- For regression tasks (optical flow, images), consider identity or other suitable normalization for \(\psi(\cdot)\).",
    "- Ensure attack is compatible with batch inputs (batched predictions and labels).",
    "- Save intermediate adversarial examples if needed for analysis.",
    "The implementation must conform to the specified 'Data structures and interfaces' to ensure integration with the overall pipeline.",
    "Summary:",
    "- Initialize adversarial sample with random noise within epsilon.",
    "- Loop for max iterations: forward, compute \(\psi\), cosine similarity, scaled loss, backward, update with sign, clip.",
    "- Return the final adversarial sample."
  ]
}

## dataset_loader.py

**Logic Analysis for dataset_loader.py – DatasetLoader Class**

---

### 1. Purpose and Responsibilities of DatasetLoader

- To load datasets for different pixel-wise prediction tasks:
  - Semantic segmentation (Pascal VOC 2012)
  - Optical flow estimation (KITTI 2015)
  - Image restoration (GoPro)
  - Image denoising (SSID)
- To handle dataset-specific formats, preprocessing, and augmentation.
- To provide an iterator/generator over samples for batch or single-sample processing.
- To ensure data is normalized correctly, and labels are in a format compatible with model inputs.
- To offer flexibility for train, validation, and test splits, as specified in the configuration.

---

### 2. Inputs and Initialization Parameters

- **dataset_name** (e.g., "PascalVOC2012", "KITTI2015", "GoPro", "SSID")
- **data_path**: root directory containing dataset images and annotations.
- **task** (e.g., "semantic_segmentation", "optical_flow", "image_restoration", "image_denoising")
- **split**: "train", "validation", or "test"
- **augment**: boolean, whether to apply data augmentation (cropping, flipping, scaling, color jitter, etc.), especially during training.
- **input_size**: size to resize images (height, width); default sizes based on dataset (e.g., 512x512, 1280x720).

### 3. Dataset-specific Details and Loading

---

#### **Semantic Segmentation (Pascal VOC 2012)**

- **Data components**:
  - Images: RGB images in formats like PNG or JPEG.
  - Labels: pixel-wise class labels as 2D arrays or images.
- **Location**:
  - Images: usually in a 'JPEGImages' folder.
  - Labels: in 'SegmentationClass' folder (or similar).
- **Preprocessing**:
  - Resize images and labels to `input_size` (512x512 in config).
  - Normalize images: pixel values scaled to [0,1] (divide by 255).
- **Labels**:
  - Load as integer class indices (0..N-1).
  - For loss computation: convert to one-hot encoding if needed, or keep as class indices for cross-entropy.
- **Supporting augmentations**:
  - Random crop, flip, color jitter applied during training.

---

#### **Optical Flow (KITTI 2015)**

- **Data components**:
  - Image pairs: two consecutive frames (e.g., 3D RGB images).
  - Ground truth flow: 2D vector field per pixel.
- **Location**:
  - Images: in a folder, e.g., "training/image_2/".
  - Flow labels: in a separate folder, e.g., "training/flow_occ/"; stored as .png or .flo files.
- **Preprocessing**:
  - Resize images and flow to `input_size`.
  - Normalize images to [0,1].
  - Flow: load as float tensors; apply resizing while keeping vector directions consistent.
- **Labels**:
  - Flow fields: 2-channel tensors (horizontal and vertical components).
  - Missing or occluded areas: handled via a mask or ignore label.
- **Supporting augmentations**:
  - Random cropping, flipping, slight color adjustments during training.

---

#### **Image Restoration (GoPro)**

- **Data components**:
  - Degraded images (blurred, etc.).
  - Ground-truth sharp images.
- **Location**:
  - Images in 'train' or 'test' folders.
- **Preprocessing**:
  - Resize images to input_size (1280x720).
  - Normalize images to [0,1].
- **Labels**:
  - Ground-truth images: floating-point tensors.
- **Supporting augmentations**:
  - Random cropping, flipping, brightness, contrast adjustments during training.

---

#### **Image Denoising (SSID)**

- **Data components**:
  - Noisy images and clean ground truths.
- **Location**:
  - Same structure as GoPro.
- **Preprocessing**:
  - Resize, normalize similarly.

---

### 4. Data Loading and Formatting Strategies

- Use a Python dataset class (inherits `torch.utils.data.Dataset`):
  - Implement `__len__` and `__getitem__`.
- Inside `__getitem__`:
  - Load image(s) and label(s):
    - For segmentation: load image and label mask.
    - For optical flow: load image pair and flow vectors.
    - For restoration/denoising: load degraded and ground truth images.
  - Resize to `input_size`.
  - Normalize images.
  - Convert labels into appropriate format:
    - Integer class labels (for cross-entropy).
    - Float flow vectors.
    - Raw images (for restoration).
  - Apply augmentation if `augment` is true.
  - Return a dictionary or tuple: `(inputs, labels)`.

---

### 5. Handling Dataset-specific Edge Cases & Details

- **Missing labels or occlusion masks**:
  - For optical flow: mask invalid regions.
- **Data augmentation**:
  - Random crops: aligned for image and label.
  - Flips: same flip consistent for image and label.
  - Color jitter: only for images.
- **Data types**:
  - Convert to `torch.FloatTensor` or `torch.LongTensor` as appropriate.
  - Keep labels as integers for cross-entropy.
- **One-hot encoding**:
  - Optional, for a batch of labels, convert as needed when computing loss.

---

### 6. Dataset Split Handling

- Based on `split` parameter:
  - Load data from train/validation/test directories.
  - Use separate data subsets for training, validation, attack, and evaluation.
- Assume dataset paths are correctly structured; implement directory parsing accordingly.

---

### 7. Implementation Considerations

- Lazy loading: load images on demand in `__getitem__`.
- Multi-threaded data loading via `DataLoader`.
- Consistent normalization: define a utility function or method for normalization.
- Flexible input sizing: resize images to specified `input_size`.
- Compatibility with model input expectations.

---

### 8. Output and Usage

- The class provides an iterator over dataset samples.
- Each sample is a tuple/dict:
  
  ```python
  {
    'image': torch.Tensor,             # normalized image tensor
    'label' or 'flow' or 'target':  # depending on task
  }
  ```
  
- DatasetLoader can be instantiated multiple times with different parameters for different experiments.

---

### 9. Summary of Key Components

- Constructor:
  - Initializes dataset paths, split, augmentation, input size.
- `load_dataset()`:
  - Reads file lists, creates dataset object.
- `__getitem__`:
  - Loads image(s), label(s).
  - Resizes.
  - Normalizes.
  - Applies augmentations if needed.
  - Returns sample dict.
- Support functions:
  - Image reading (`PIL`, `cv2`).
  - Resizing (`cv2.resize`, `transforms`).
  - Normalization.
  - Label processing (categorical conversion, flow normalization).

---

**This comprehensive logic analysis should guide the implementation of `dataset_loader.py`, ensuring accurate, dataset-specific, and task-specific data loading aligned with the experimental setup and the descriptions provided by the paper and supplementary material.**

## evaluation.py

Evaluation.py: Logic Analysis for 'Evaluation' Class

Overview:
The Evaluation class is responsible for computing key performance metrics for the model's predictions on various pixel-wise prediction tasks (semantic segmentation, optical flow, image restoration). These metrics include IoU and pixel accuracy for segmentation, EPE (endpoint error) for optical flow, and PSNR and SSIM for image restoration. The class should support assessing model performance over datasets, saving quantitative results, and optionally visual outputs, as well as plotting metrics over attack iterations or different models/datasets.

Key Responsibilities:
- Initializing with model, dataset, attack parameters, and configuration.
- Running inference on dataset samples (images or sequences).
- Comparing predictions with ground truth labels (or flow fields, images).
- Computing metrics per sample and aggregating across dataset.
- Saving results and producing visualizations (e.g., adversarial examples, segmentation overlays).
- Supporting evaluation at multiple attack iterations, capturing metric trends over iterations.
- Saving logs and providing structured metrics output (dict).

**Core elements to implement:**

1. **Initialization:**
   - Store model, dataset, task type, and evaluation flags.
   - Load ground truth labels in formats appropriate for each task.
   - Initialize data structures for metrics (dicts, lists).
   - Set model to evaluation mode and select device.

2. **Prediction on Dataset:**
   - For each dataset sample:
     - Load input data (image, flow, or image pairs for denoising).
     - Run model inference: `pred = model.predict(x)`.
     - Store predictions for metrics computation.
     - Optional: save adversarial examples or overlay images for qualitative analysis.

3. **Metrics Computation:**
   
   - **Semantic segmentation:**
     - Use predicted logits (or probabilities) and ground truth labels.
     - Compute IoU:
       - For each class: intersection and union over true/ predicted pixels.
       - Aggregate mean IoU over classes.
     - Compute pixel accuracy:
       - Correct predicted pixels / total pixels.
     - Store per-sample metrics, update overall averages.
   
   - **Optical Flow:**
     - Use predicted flow vectors and flow ground truth.
     - Compute EPE:
       - For each pixel: Euclidean distance between predicted and true flow vectors.
     - Compute EPE-f1-all:
       - Count pixels with EPE > 3.0 or normalized EPE > 0.05, then average over all images.
     - Store metrics per sample and aggregate.
   
   - **Image Restoration (Denoising / Deblurring):**
     - Use ground truth images and reconstructed outputs.
     - Compute PSNR:
       - Peak Signal-to-Noise Ratio = \(20 \cdot \log_{10} ( \max{I_{true}} / \text{MSE} )\).
     - Compute SSIM:
       - Use standard implementation (from `skimage.metrics` or custom).
     - Collect per-image and average.

4. **Results Saving:**
   - Save the computed metrics in a structured dictionary.
   - Save per-sample predicted images, overlays, and adversarial examples (if required).
   - Save logs to disk (JSON, CSV, or pickle) for analysis.

5. **Plotting:**
   - Generate graphs over attack iterations:
     - For segmentation: IoU trend, accuracy trend.
     - For optical flow: EPE vs iteration.
     - For restoration: PSNR and SSIM vs iteration.
   - Use matplotlib to produce plots with labels, legends, and save figures.

6. **Supporting Multiple Datasets & Tasks:**
   - Use task-specific branches for metrics:
     - Maintain separate functions per metric type.
     - Detect dataset type and select appropriate metrics.
   
7. **Handling Results over Attack Iterations:**
   - If evaluation over multiple attack steps is required (e.g., after 3, 5, 10, 20, 40, 100 iterations), the class should:
     - Accept list of adversarial images per iteration.
     - Compute metrics for each set.
     - Store sequentially.
     - Plot metric curves over iterations.

8. **Additional Details:**
   - For multi-class segmentations, ensure ignore indices are handled.
   - For flow fields, handle different vector formats.
   - For images, ensure correct normalization for PSNR and SSIM.
   - Maintain reproducibility and consistent device utilization.
   - Use dedicated functions for metric computation to keep code clean.

**Edge Cases & Extra Considerations:**
- Discrepancy between the size of model output and ground truth: resize or interpolate as needed.
- Missing or corrupted labels: skip or assign default performance.
- Handling sparse or partial flow annotations (if applicable).
- Handling batch evaluation: sum metrics over batch size, average at the end.
- Manage inter-iteration evaluation: total average and per-iteration trends.

---

**In summary, the 'Evaluation' class:**

- Loads dataset and ground-truth labels.
- Runs model predictions on all samples.
- Computes task-specific metrics (IoU, pixel accuracy, EPE, PSNR, SSIM).
- Collects metrics per sample and averages over dataset.
- Saves metrics and visual results.
- Supports iterative evaluation (per attack step) and plotting of metric curves.
- Modularizes per-task functions for clarity and maintainability.

This comprehensive logic allows development of robust, flexible evaluation code that reproduces all reported results accurately in line with the research paper’s methodology.

## main.py

# Logic Analysis for main.py

This document provides a detailed, step-by-step logical breakdown for implementing `main.py`, which orchestrates the execution of experiments to reproduce CosPGD attacks and evaluations as described in the paper. The goal is to ensure fidelity with the experimental design, dataset handling, model inference, attack procedure, evaluation metrics, and configuration parameters, as outlined in the previous plan.

---

## 1. **Imports and Setup**

- **Import Required Modules:**
  - Load core libraries: `torch`, `torchvision`, `numpy`, `matplotlib`, `os`, `logging`.
  - Import custom modules:
    - `DatasetLoader` from `dataset_loader.py`
    - `Model` from `model.py`
    - `Attack` from `attack.py`
    - `Evaluation` from `evaluation.py`
    - Utility functions from `utils.py` (such as setting seed, plotting, etc.).

- **Set Device & Random Seed:**
  - Use `torch.device()` based on configuration (`cuda:0`).
  - Set random seed for reproducibility across PyTorch, numpy, and any other libraries.

- **Configure Logging:**
  - Set verbosity level (INFO/WARNING).
  - Log start of experiment, dataset, models, hyperparameters.

---

## 2. **Load Dataset(s) via DatasetLoader**

- **Initialize DatasetLoader instances:**
  - For **semantic segmentation**:
    - Pass parameters:
      - `dataset_name='PascalVOC2012'`
      - `root_dir=./data/PascalVOC2012`
      - `split='train'`
      - `augment=True`
      - `input_size=512`
  - For **optical flow**:
    - Pass parameters:
      - `dataset_name='KITTI2015'`
      - `root_dir=./data/KITTI2015`
      - `split='validation'`
  - For **image restoration**:
    - Pass parameters:
      - `dataset_name='GoPro'`
      - `root_dir=./data/GoPro`
      - `split='train'`
      - `input_size=1280x720`

- **Load datasets:**
  - Call dataset loader `.load_dataset()` method to get iterable datasets/dataloaders.
  - For batch processing, create DataLoader objects with the specified `batch_size` (from config).

- **Dataset splitting:**
  - Confirm train/validation/test splits are used as per experiment setup.
  - Implement any augmentation if specified.

---

## 3. **Load & Initialize Models via Model.py**

- **Instantiate models as per configuration:**
  - For **DeepLabV3**, **PSNet**:
    - Use model name, e.g., `'deeplabv3'` or `'psnet'`.
    - Load specified checkpoint path.
  - For **UNet** and **NAFN**:
    - Use their specific setup (e.g., encoder `'ConvNeXt-tiny'`).
- **Load weights:**
  - Call `load_weights()` method.
  - Ensure model is loaded to GPU as per device setting.
- **Set mode:**
  - Switch models to evaluation mode with `.eval()`.
- **Optional:**
  - Wrap models with `torch.nn.DataParallel` if multi-GPU (not specified here, single GPU assumed).

---

## 4. **Set Up Attack Parameters from Config**

- Parse `attack_parameters:` section:
  - `epsilon` (float, e.g., `8/255`)
  - `step_size` (float, e.g., `2/255`)
  - `attack_iters` list (e.g., `[3, 5, 10, 20, 40, 100]`)
  - `targeted` (boolean)
  - `target_label` (nullable; used if targeted attack enabled)

- Instantiate attack object:
  - Pass models, device, and attack hyperparameters.
  - For each attack run, specify:
    - Number of iterations (per each run).
    - Targeted or untargeted mode.
    - Whether to attack in \(\ell_\infty\) or \(\ell_2\) norm (assumed from context).
    
---

## 5. **Run Attacks Over Dataset Images**

- **Iterate over dataset:**
  - For each sample:
    - Extract input image tensor `x` and label `y`.
      - Ensure correct normalization (values in [0,1]).
      - For segmentation: labels as class indices or one-hot (per code in dataset loader).
      - For optical flow/regression: flow field tensor.
    - **Initialize adversarial input:**
      - Call attack's `attack()` method with inputs:
        - `x`, `y`
        - `targeted=attack_params['targeted']`
        - `target_label=attack_params['target_label']` (if targeted)
      - Inside `attack()`, handle:
        - Initial perturbation addition within \(\epsilon\) bound.
        - Number of iterations as per the current value in `attack_iters`.
        - Attack mode (targeted/non-targeted).
        - Uses `Attack` class's `attack()` method, which internally:
          - Performs iterative CosPGD updates with the scaled loss.
          - Clips perturbed input within \(\epsilon\)-ball.
    - Collect adversarial example.
    
- **Optional:**
  - Store adversarial examples, model predictions, and metrics at each iteration for later analysis.

---

## 6. **Evaluate Adversarial Examples Using Evaluation.py**

- For each adversarial sample:
  - Run the **model inference**:
    - `model.predict(x_adv)` to get predictions.
    - For segmentation: class logits or probabilities (to evaluate IoU, accuracy).
    - For optical flow: flow vectors to evaluate EPE.
    - For restoration: reconstructed images for PSNR/SSIM.
  - Use the `Evaluation` class:
    - Initialize with model, dataset, and inference mode.
    - Call metric functions:
      - **Semantic segmentation:** compute IoU, pixel accuracy.
      - **Optical flow:** compute endpoint error (EPE).
      - **Image restoration:** compute PSNR and SSIM.
  - Save metrics per sample.
  - Aggregate metrics over dataset for overall assessment.

- Store evaluation results in dictionaries/logs.

---

## 7. **Save Results & Generate Plots**

- **Store all results:**
  - Save metrics per attack iteration and dataset.
  - Save adversarial images, predictions, and relevant visualizations (e.g., attacked segmentation masks, flow maps).

- **Generate plots:**
  - Plot attack effectiveness vs. number of iterations, similar to Figures 2, 3, 14-21.
  - Plot metric decay (e.g., IoU, EPE) over attack steps.
  - Plot qualitative example comparisons after attacks.

- **Save plots and metrics:**
  - Use `matplotlib` for plotting.
  - Save as images or figures with descriptive filenames.

---

## 8. **Optional: Attack & Evaluation Variations**

- Repeat for:
  - **Targeted vs. Untargeted** attacks.
  - Different \(\ell_\infty\) / \(\ell_2\) bounds.
  - Multiple initializations (random noise).
  - Transfer attacks (using different models as attack and evaluate).

---

## 9. **Final Remarks & Cleanup**

- **Ensure all paths** for saving data and results are created.
- **Clear cache** if needed for memory management.
- **Log summary:**
  - Final metrics achieved.
  - Attack success rates.
  - Comparative effectiveness.

---

## 10. **Summary**

The main.py orchestrates the overall workflow:
1. Load datasets & models based on configs.
2. For each dataset sample:
   - Generate adversarial inputs via CosPGD with specified iterations.
   - Evaluate the model predictions on adversarial inputs.
   - Record metrics and visualizations.
3. Aggregate and save all results.
4. Plot the evolution of metrics over attack steps.
5. Support multiple configurations for hyperparameters and experimental setups as per paper.

This structured logic ensures the code implementation remains consistent with the paper's methodology, experimental design, and reproducibility goals.

## model.py

**Logic Analysis for `model.py`**

---

### Overview:
The `model.py` module provides a `Model` class designed to load, initialize, and run inference with various pixel-wise prediction models, including semantic segmentation architectures (DeepLabV3, PSPNet, UNet) and image restoration models (NAFNet). It should support loading from checkpoint files, handling different architectures, and providing a `predict(x: torch.Tensor)` method which performs inference and outputs predictions suited for downstream attack and evaluation.

---

### Core Responsibilities:

1. **Model Selection and Initialization:**
   - Based on configuration input parameters, determine the model to instantiate. For example:
     - `'deeplabv3'` with ResNet50 backbone
     - `'psnet'` with ResNet50 backbone
     - `'unet'` with ConvNeXt-tiny encoder
     - `'nafnette'` (assumed custom or from external code)
   - Call appropriate model constructor, possibly from torchvision (for DeepLabV3, PSPNet) or custom model definitions for UNet, NAFNet.
   
2. **Weight Loading:**
   - Load pre-trained weights from checkpoint files specified in `checkpoint_path`.
   - Possibly handle device compatibility (load to GPU if available, else CPU).
   - Ensure model is in evaluation mode (`model.eval()`).

3. **Device Management:**
   - Use the `torch.device` specified in the configuration (`cuda:0`, or CPU).
   - Move model to device accordingly.
   - Optionally, handle multi-GPU via `DataParallel` if needed (not specified, assume single GPU).

4. **Inference (`predict(x: torch.Tensor)`):**
   - Accept input tensor `x` (properly normalized and preprocessed).
   - Forward pass through the model.
   - Return prediction tensor:
     - For segmentation: logits or class probabilities per pixel.
     - For optical flow: flow vector field predictions.
     - For image restoration: reconstructed images.
   - The output tensor structure should match dataset task:
     - Classification (segmentation): shape `[batch_size, num_classes, H, W]`.
     - Regression (flow): shape `[batch_size, 2, H, W]`.
     - Restoration: shape `[batch_size, C, H, W]` (C=3 for RGB).

5. **Model Encapsulation:**
   - Keep internal reference to the actual torch model object.
   - Provide an interface (predict) for downstream attack and evaluation code.

---

### Implementation details and considerations:

- **Model Architecture Loading:**
  - For DeepLabV3 / PSPNet:
    - Use torchvision models (`torchvision.models.segmentation.deeplabv3_resnet50`, `PSPNet` if available, or custom code).
    - Load weights via `torch.load(checkpoint_path)` into the model’s state_dict.
  - For UNet:
    - Use a custom UNet implementation with ConvNeXt-tiny encoder.
    - Initialize the `UNet` class, then load weights.
  - For NAFNet:
    - Import custom class.
    - Load weights with `torch.load()`.
  
- **Checkpoint Loading:**
  - Implement error handling for missing files or mismatched architectures.
  - Use `map_location='cuda:0'` or CPU depending on device availability.
  
- **Evaluation Mode:**
  - Both during inference and against attacks, models should be set to `model.eval()`.
  - For models like UNet or NAFNet, ensure no dropout or batch norm training behaviors are active (set in eval mode).

- **Device Management:**
  - Support moving models to GPU (`cuda:0`) if available and specified.
  - Confirm that input tensors are on the same device during prediction.

- **Prediction Method (`predict(x)`):**
  - Wrap in `torch.no_grad()` context for efficiency.
  - Accept batched tensor input.
  - Return raw prediction tensor (logits before softmax for segmentation, vector fields for optical flow, images for restoration).
  - Possibly, for segmentation, raw logits are preferable; softmax can be applied outside if needed.

- **Handling Model Variants:**
  - For segmentation:
    - Optionally, return class probabilities or raw logits.
    - The attacker uses logits for gradient calculation, so do not apply softmax in `predict` (unless explicit normalization is needed).
  - For regression:
    - Return raw predicted flow/disparity.
  
---

### Code structure and tasks:
- **Class Definition:**
  ```python
  class Model:
      def __init__(self, model_name: str, checkpoint_path: str, device: torch.device):
          # Load model based on `model_name`.
          # Load weights from checkpoint_path.
          # Move to device.
          pass
      
      def predict(self, x: torch.Tensor) -> torch.Tensor:
          # Forward pass in torch.no_grad()
          # Return prediction tensor.
          pass
  ```

- **Supporting functions:**
  - Model initialization functions per architecture.
  - Weight loading helpers.

- **Configuration compatibility:**
  - Use `config['models'][model_name]['checkpoint_path']` for checkpoint path.
  - Select device based on `config['hardware']['device']`.

---

### Summary:
- The `Model` class encapsulates model architecture, loading, and inference.
- It takes model specifications and checkpoint paths.
- It handles device allocation.
- Its core method `predict(x)` performs inference, returning the model output tensor.
- The output tensor structure is aligned with the dataset/task specifics, critical for attack generation.

---

### Additional notes:
- Make sure that the input `x` to `predict()` is preprocessed similarly to training: normalized to [0,1], resized appropriately.
- For datasets that output class indices, convert them to tensors compatible with model input.
- The class should be flexible, supporting new architectures if needed, using a modular approach (e.g., factory pattern or dict-based model constructors).

This detailed logic ensures that the implementation will correctly load models, handle different architectures, and produce inference outputs suitable for the CosPGD attack and subsequent evaluation.

## utils.py

# Utility Functions for CosPGD Attack Implementation

This module provides essential utility functions to support the implementation of the CosPGD adversarial attack framework, dataset handling, model inference, and evaluation metrics. The functions are designed to be modular, efficient, and compatible with the deep learning framework (PyTorch). The core functionalities include tensor clipping, cosine similarity computation, normalization, reproducibility helpers, and plotting utilities.

---

## 1. Reproducibility Helper

### Function: `set_seed(seed: int) -> None`
- **Purpose:** Ensure deterministic behavior across experiments by setting random seeds.
- **Inputs:** Integer seed value (from config.yaml `training.seed` or general).
- **Implementation:**
  - Set Python's `random.seed`.
  - Set `torch.manual_seed`.
  - Set `torch.cuda.manual_seed_all`.
  - Enable deterministic algorithms via `torch.backends.cudnn.deterministic = True`.
  - Disable benchmark to ensure reproducibility: `torch.backends.cudnn.benchmark = False`.
- **Usage:** Call at the start of main script.

---

## 2. Tensor Operations

### Function: `clip_tensor(x: torch.Tensor, epsilon: float, x_clean: torch.Tensor, norm_type: str='l_infinity') -> torch.Tensor`
- **Purpose:** Clip adversarial tensor `x` to be within the \(\ell_\infty\) or \(\ell_2\) ball centered at `x_clean`.
- **Inputs:**
  - `x`: current tensor (adv example).
  - `epsilon`: maximum allowed perturbation.
  - `x_clean`: original clean input tensor.
  - `norm_type`: `'l_infinity'` (default) or `'l_2'`.
- **Implementation:**
  - For `l_infinity`, clip each element: `torch.clamp(x, x_clean - epsilon, x_clean + epsilon)`.
  - For `l_2`, normalize the difference norm, project onto \(\epsilon\)-sphere, then add back to `x_clean`.
- **Output:** Tensor clipped within the specified norm ball.

### Function: `normalize_tensor(x: torch.Tensor, method: str='softmax') -> torch.Tensor`
- **Purpose:** Apply normalization to a tensor, typically the model outputs, to produce probability distributions.
- **Inputs:**
  - `x`: raw predictions (logits).
  - `method`: `'softmax'` (default), could include `'identity'` or other methods if necessary.
- **Implementation:**
  - For `'softmax'`: apply `torch.nn.functional.softmax(x, dim=1)` across class/channel dimension.
  - For `'identity'`: return `x` directly.
  - Extendable for other normalization functions if needed.

---

## 3. Cosine Similarity Computation

### Function: `compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor`
- **Purpose:** Calculate per-pixel cosine similarity between normalized prediction and target vectors.
- **Inputs:**
  - `pred`: normalized model prediction tensor (\(\psi(f_\theta(X))\)), shape `[batch, classes, H, W]`.
  - `target`: target tensor (`Y`), shape `[batch, classes, H, W]`.
- **Implementation:**
  - Compute dot product: `torch.sum(pred * target, dim=1)` over class dimension.
  - Compute norms: `torch.norm(pred, p=2, dim=1)` and `torch.norm(target, p=2, dim=1)`.
  - Avoid division by zero with small epsilon inside norms.
  - Compute cosine similarity: `(dot_product + eps) / (norm_pred * norm_target + eps)`.
- **Output:** Tensor of shape `[batch, H, W]`, per pixel cosine similarity in range [-1,1].

### Function: `cosine_similarity_per_pixel(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor`
- **Purpose:** Integrated function that handles softmax normalization and computes per-pixel cosine similarity over whole batch.
- **Implementation:**
  - Apply `normalize_tensor` with `'softmax'` on `pred_logits`.
  - Use `compute_cosine_similarity`.
- **Note:** Can be used within the attack to scale pixel-wise losses.

---

## 4. Loss Functions

### Function: `pixelwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str='cross_entropy') -> torch.Tensor`
- **Purpose:** Compute pixel-wise loss suitable for pixel classification or regression.
- **Inputs:**
  - `pred`: raw logits or predicted outputs, shape `[batch, classes, H, W]` or `[batch, 1, H, W]`.
  - `target`: ground truth labels, shape `[batch, H, W]` for class labels or `[batch, 1, H, W]` for regression.
  - `loss_type`: `'cross_entropy'` or `'mse'`.
- **Implementation:**
  - For `'cross_entropy'`: use `torch.nn.functional.cross_entropy`, with `reduction='none'` to get per-pixel losses.
  - For regression: `torch.nn.functional.mse_loss` with `reduction='none'`.
- **Output:** Tensor `[batch, H, W]` of pixel-wise losses.

### Function: `scaling_loss(pred: torch.Tensor, target: torch.Tensor, ...) -> torch.Tensor`
- **Purpose:** Calculate the scaled pixel loss using cosine similarity as per the CosPGD formulation.
- **Implementation:**
  - Compute normalization (`psi`) with softmax.
  - Compute cosine similarity per pixel.
  - Calculate pixelwise loss (`cross_entropy`, `MSE`, etc.).
  - Weight the pixelwise loss by the cosine similarity (or \(1 - \cos\) for targeted case).
  - Return the scaled loss tensor for gradient backpropagation.

---

## 5. Visualization & Plotting

### Function: `plot_metrics(metrics: dict, title: str='Metrics over Attack Iterations') -> None`
- **Purpose:** Plot evaluation metrics (IoU, accuracy, EPE, PSNR, SSIM) over attack iterations.
- **Implementation:**
  - Use `matplotlib`.
  - For each metric, plot mean with shaded region of standard deviation if available.
  - Label axes, add legend, title.

### Function: `save_image_batch(images: torch.Tensor, filename: str) -> None`
- **Purpose:** Save a batch of images or adversarial examples for qualitative analysis.
- **Implementation:**
  - Convert tensors to numpy arrays.
  - Use `PIL.Image` or `matplotlib.pyplot` to save images.
- **Note:** Ensure pixel range normalization to [0, 255] before saving.

---

## 6. Miscellaneous Helpers

### Function: `compute_pixelwise_angle(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor`
- **Purpose:** Helper to compute the angle between prediction vectors and targets if needed for diagnosis.
- **Implementation:**
  - Use `torch.acos` of cosine similarity; optional for debugging.

### Function: `log_progress(current_step: int, total_steps: int, logs: dict) -> None`
- **Purpose:** Log progress and current metrics to console or file during attack or evaluation.

---

## 7. Summary of Functions and Usage

| Function | Purpose | Inputs | Outputs | Notes |
|------------|--------------|------------------|--------------|--------------|
| `set_seed()` | Ensure reproducibility | seed: int | None | Set random seeds across frameworks |
| `clip_tensor()` | Clip to epsilon ball | x, epsilon, x_clean, norm_type | clipped tensor | Essential in each attack iteration |
| `normalize_tensor()` | Normalize model output | x, method | normalized tensor | Used for \(\psi\) function |
| `compute_cosine_similarity()` | Compute per-pixel cosine similarity | pred, target | [batch, H, W] tensor | Core for scaling loss |
| `pixelwise_loss()` | Calculate pixel loss | pred, target, loss_type | [batch, H, W] tensor | Cross entropy or MSE |
| `scaling_loss()` | Scale pixel loss with cosine similarity | pred, target, ... | scaled loss tensor | Used as attack objective |
| `plot_metrics()` | Plot attack progress | metrics dict | None | For analysis |
| `save_image_batch()` | Save images for analysis | images tensor | image files | Visualize attack effects |

---

## Final Notes:

- Ensure compatibility with the dataset formats:
  - Semantic segmentation labels as class indices or one-hot.
  - Optical flow fields as 2D vector maps.
  - Restored images normalized to [0,1].

- Handle batch sizes and device placement appropriately.
- Consistent naming conventions and parameter units (e.g., epsilon, step size).

This logic analysis provides a comprehensive specification of utility functions needed to implement CosPGD and associated evaluation, structured for clear, modular, and scalable code development.

