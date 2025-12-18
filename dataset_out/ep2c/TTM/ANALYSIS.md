# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## datasets.py

**Logic Analysis for datasets.py**

The purpose of this module is to handle data loading and preprocessing for the experimental setup, specifically for CIFAR-100 and ImageNet datasets, ensuring consistency with the methodology of the paper and the provided configuration. It must facilitate datasets' preparation for training and evaluation, including data augmentation, normalization, and batching, aligned with standard practices and the experimental protocols outlined.

---

### 1. Modules and Imports
- Use `torchvision.datasets` for dataset objects.
- Use `torchvision.transforms` for data augmentation, normalization, and preprocessing.
- Use `torch.utils.data.DataLoader` to create data loaders.
- Optionally, import `os` for directory management.
- Import `yaml` or directly load config parameters; but since config YAML is external, its relevant parts should be passed as function parameters or imported globally.

---

### 2. Dataset Handling

#### 2.1. Dataset Types Supported
- **CIFAR-100**:
  - Training and validation datasets.
  - Data augmentations: random crop, random horizontal flip, normalization.
  - Common normalization mean/std: per official cifar10/cifar100 standards or as in the original paper.
- **ImageNet**:
  - Training and validation datasets.
  - Data augmentations: random resized crop, random horizontal flip, normalization.
  - Use ImageNet mean/std for normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

#### 2.2. Dataset Path & Directory Management
- Dataset root directory (`data_dir`) is provided via config.
- Validate directory existence or create if needed (for training data).

#### 2.3. Dataset Transformations
- Define separate `transform_train` and `transform_eval` pipelines for each dataset, complying with typical training and testing protocols:
  - **CIFAR-100:**
    - Training: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, convert to tensor, normalize.
    - Evaluation: convert to tensor, normalize.
  - **ImageNet:**
    - Training: `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, convert to tensor, normalize.
    - Evaluation: `Resize(256)`, `CenterCrop(224)`, convert to tensor, normalize.

#### 2.4. Dataset Instances
- For dataset loading, instantiate datasets using torchvision:
  - `datasets.CIFAR100(root=..., train=True/False, download=False, transform=...)`
  - `datasets.ImageNet(root=..., split='train'/'val', transform=...)`
- Assume datasets are already downloaded or provide `download=True` if appropriate.

### 3. DataLoader Construction
- Use DataLoader with specified batch size (from config), shuffle=True for training, shuffle=False for evaluation.
- Set `num_workers`, e.g., 4 or 8, for parallel data loading.
- Use pin_memory=True for efficient transfer to GPU.

### 4. Function Interface
- Provide a function (or class) `load_data` that:
  - Takes as input the dataset name, data directory, batch size (from config), and data split (train/test).
  - Returns DataLoader objects for training and validation datasets.

### 5. Integration with Existing Pipelines
- Structure code to allow easy dataset selection based on the configuration.
- Call `load_data` with parameters from `config.yaml`.

### 6. Ensuring Reproducibility / Consistency
- Consistently set the data augmentations and normalization as per the experimental protocol.
- Include options for shuffling, with random seed setting external or optional.

### 7. Error Handling & Edge Cases
- Check if provided data directory exists.
- Handle unsupported dataset names with meaningful exceptions.
- Optionally, include a download flag if datasets are not present.

---

### 8. Summary
- **Inputs:** dataset name, data_dir, batch_size, split_type (train/val/test), optional download flag.
- **Outputs:** DataLoader for specified dataset split.
- **Transform pipelines:** dependent on dataset and training/evaluation stage.
- **Configurations:** derived from `config.yaml` (e.g., `dataset.name`, `dataset.data_dir`, `training.batch_size`).

---

### 9. Example (Pseudocode)
```python
def load_data(dataset_name, data_dir, batch_size, is_train=True):
    if dataset_name == 'CIFAR100':
        if is_train:
            transform = transform_train_cifar
        else:
            transform = transform_eval_cifar
        dataset = datasets.CIFAR100(root=data_dir, train=is_train, download=False, transform=transform)
    elif dataset_name == 'ImageNet':
        split = 'train' if is_train else 'val'
        if is_train:
            transform = transform_train_imagenet
        else:
            transform = transform_eval_imagenet
        dataset = datasets.ImageNet(root=data_dir, split=split, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True)
    return dataloader
```

### 10. Additional Considerations
- Make sure to align normalization constants to the experimental setup.
- For reproducibility, set `torch.manual_seed()` externally before data loading if needed.
- Ensure that the data transformations match those used in the experiments to facilitate comparability.

---

**Note:**  
The above logic analysis provides a detailed, stepwise conceptual blueprint facilitating the implementation of `datasets.py`. Actual code should instantiate and implement the specific transformations, dataset objects, and data loaders accordingly, with proper parameters from the configuration file.

## evaluation.py

### Logic Analysis for `evaluation.py`

This module is responsible for assessing the trained student model (and optionally, the teacher model) on validation/test datasets. Its main functions are to compute the top-1 accuracy, analyze the entropy distribution of the model's output probabilities, and log relevant metrics for reproducibility and analysis.

---

### Core Responsibilities & Functionality

1. **Data Loading & Preparation**:
   - Load validation/test dataset using data loader utilities (from `datasets.py`).
   - Apply necessary transformations (normalization, resizing). These should be consistent with training, ensuring a fair evaluation.
   
2. **Model Loading & Setup**:
   - Load the trained student model from saved checkpoints.
   - Load pre-trained teacher model if comparisons or additional evaluations are needed.
   - Set the models to evaluation mode (`model.eval()`).
   - Use `torch.no_grad()` context to avoid gradient computation during evaluation.

3. **Inference and Metrics Calculation**:
   - For each dataset batch:
     - Forward pass through the model to get logits (not probabilities).
     - Compute probability distribution \( q \) (or \( p \)) via softmax.
     - Derive top-1 predictions by identifying the class with maximum probability.
     - Calculate per-sample entropy of output probability distribution:
       \[
       H(q) = - \sum_{i=1}^{K} q_i \log q_i
       \]
   - Aggregate across all batches:
     - Count total correct predictions for accuracy.
     - Collect all output probabilities \( q \) to compute entropy distribution.

4. **Accuracy Calculation**:
   - Use true labels and predicted labels:
     \[
     \text{Top-1 accuracy} = \frac{\text{correct predictions}}{\text{total samples}} \times 100\%
     \]
     
5. **Entropy Distribution & Mean Entropy**:
   - Calculate the Shannon entropy for each sample's output distribution.
   - Compute the mean entropy over all samples.
   - Optionally, generate histograms or distribution plots (if visualization is required outside this scope).

6. **Logging and Output**:
   - Log key metrics such as:
     - Top-1 accuracy.
     - Mean entropy.
     - Distribution summaries (max, min, quartiles) if additional analysis.
   - Save metrics to console, logfile, or external files as needed for reproducibility.

7. **Additional Analyses (Optional)**:
   - Could include tracking of entropy during the evaluation epoch.
   - If multiple models are evaluated together, collect results for comparison.

---

### Implementation Details and Considerations

- **Inputs & Parameters**:
  - Path to dataset (from configuration).
  - Path to model checkpoint.
  - Whether to evaluate the teacher model or only the student.
  - Batch size (consistent with training config).

- **Outputs**:
  - Accuracy (float percentage).
  - Average entropy (float).
  - Optional: detailed entropy distribution data.

- **Code Structure & Pseudocode**:
  
  ```python
  def evaluate(model, dataloader, device):
      model.eval()
      total_samples = 0
      correct = 0
      entropy_list = []

      with torch.no_grad():
          for batch in dataloader:
              inputs, labels = batch
              inputs, labels = inputs.to(device), labels.to(device)

              logits = model(inputs)
              probs = torch.softmax(logits, dim=1)

              # Compute predictions
              _, predicted = probs.max(dim=1)

              # Count correct predictions
              correct += (predicted == labels).sum().item()
              total_samples += labels.size(0)

              # Compute per-sample entropy
              entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # small epsilon
              entropy_list.extend(entropies.cpu().numpy())

      accuracy = 100.0 * correct / total_samples
      mean_entropy = np.mean(entropy_list)

      # Log metrics
      print(f"Evaluation results: Accuracy: {accuracy:.2f}%")
      print(f"Average Entropy: {mean_entropy:.4f}")
      
      # Optionally output histograms or save results
      return {
          'accuracy': accuracy,
          'mean_entropy': mean_entropy,
          'entropy_distribution': entropy_list
      }
  ```

- **Integration with Main Training or Testing Script**:
  - Call `evaluate()` after training is completed.
  - Load model weights for student (and teacher if needed) beforehand.
  
- **Reproducibility**:
  - Use fixed seed for dataset splitting if applicable.
  - Save the evaluation results (hyperparameters, metrics) along with logs.

---

### Additional Notes:

- **Model Loading**: Must be compatible with the architecture defined in `models.py`. Ensure to load the exact checkpoint after training.
- **Device Management**: Support evaluation on GPU (`cuda`) or CPU.
- **Handling of Distribution Data**: Keep all probabilities and entropy calculations in CPU memory if memory-limited, or process per batch.
- **Performance**: No backpropagation needed; focus on efficient inference.
- **Optional**: Add visualization (histograms) outside this module, or as part of a logging pipeline.

---

### Summary

The `evaluation.py` module will:

- Load the test/validation dataset.
- Load the trained model.
- Perform inference without gradient calculation.
- Compute top-1 accuracy.
- Calculate the entropy of output distributions for each sample.
- Aggregate and log these metrics for analysis.
- Support multiple models (teacher, student) evaluation for comprehensive analysis.

This rigorous evaluation enables analysis of model accuracy and the regularization effects signified by the entropy distributions, complementing the main experimental results of the paper.

## losses.py

# Logic Analysis for losses.py

This file is responsible for implementing all core loss functions used in the TTM/WTTM distillation framework, alongside related utility functions such as computing the power transform of teacher outputs, Renyi entropy, and sample weights. The goal is to modularize and clearly organize all mathematical operations related to loss computations, ensuring they align exactly with the theoretical derivations and formulas presented in the paper.

---

## 1. Core Loss Functions and Utility Operations

### 1.1. Cross-Entropy Loss

- **Purpose:** Standard cross-entropy (CE) between ground-truth labels \( y \) and model predictions \( q \).

- **Implementation details:**
  - Input:
    - `predictions` (student logits): `Tensor` shape `[batch_size, num_classes]`.
    - `targets` (ground-truth labels): `Tensor` shape `[batch_size]` (integer class labels).
  - Use `torch.nn.functional.cross_entropy` for numeric stability.
  - No softmax needed; the loss applies directly to logits.

- **Notes:**
  - The operation is used for \( H(y, q) \) in the total loss.
  - Should be flexible enough to handle batch operations.

---

### 1.2. KL Divergence

- **Purpose:** Compute \( D_{KL}(p_T^t || q) \), i.e., the divergence between the teacher's transformed distribution \( p_T^t \) and the student's distribution \( q \).

- **Implementation details:**
  - Input:
    - Teacher's probability distribution: `Tensor` `[batch_size, num_classes]`.
    - Student's distribution: `Tensor` `[batch_size, num_classes]`.
  - Use `torch.nn.functional.kl_div` with `reduction='batchmean'` or `reduction='none'` (then average) as needed.
  - Ensure both `p_T^t` and `q` are log-probabilities or probabilities:
    - For the KL divergence, convert to log probabilities via `torch.log()` or `F.log_softmax()` as needed.
    - The implementation can directly take log-probabilities for stability.

- **Notes:**
  - The divergence will be computed between the teacher's transformed distribution \( p_T^t \) and the student distribution \( q \).
  - For the WTTM loss, multiply the divergence by the sample weight.

---

### 1.3. Power Transform of Teacher Outputs

- **Purpose:** Given teacher logits \( v \), compute the transformed distribution \( p_T^t \) according to the power transform:

  \[
  \hat{p}_i = \frac{p_i^\gamma}{\sum_j p_j^\gamma}
  \]

- **Implementation:**
  - Inputs:
    - Teacher logits: `Tensor` `[batch_size, num_classes]`.
    - \( \gamma \) (gamma): scalar float, \( 0 < \gamma \leq 1 \).
  - Steps:
    1. Compute teacher probabilities:
       ```python
       p = F.softmax(teacher_logits, dim=1)
       ```
    2. Apply power transform:
       ```python
       p_pow = torch.pow(p, gamma)
       ```
    3. Normalize:
       ```python
       p_transformed = p_pow / torch.sum(p_pow, dim=1, keepdim=True)
       ```
  - Output:
    - Transformed distribution: `Tensor` `[batch_size, num_classes]`.

- **Notes:**
  - This function encapsulates the core transform aligned with the theoretical equivalence to temperature scaling (\(\gamma = 1/T\)).

---

### 1.4. Renyi Entropy Calculation

- **Purpose:** Compute the Renyi entropy of a distribution \( p \):

  \[
  H_\alpha(p) = \frac{1}{1-\alpha} \log \sum_j p_j^\alpha
  \]

- **Implementation:**
  - Inputs:
    - Distribution: `Tensor` `[batch_size, num_classes]`.
    - \(\alpha\): order of the entropy, scalar float, typically \( 0 < \alpha < 1 \).
  - Steps:
    1. Compute power sum:
       ```python
       sum_pow = torch.sum(torch.pow(p, alpha), dim=1)
       ```
    2. Take logarithm:
       ```python
       log_sum = torch.log(sum_pow)
       ```
    3. Compute entropy:
       ```python
       entropy = log_sum / (1 - alpha)
       ```
  - Outputs:
    - Entropy per sample: `Tensor` `[batch_size]`.

- **Notes:**
  - For \(\alpha \to 1\), handle with limit (approach Shannon entropy).
  - Useful for defining additional regularizers or sample weights.

---

### 1.5. Sample Adaptive Weight \( U_{1/T}(p^t) \)

- **Purpose:** Quantify the smoothness of teacher output \( p^t \):

  \[
  U_{1/T}(p^t) = \sum_j (p_j^t)^{1/T}
  \]
- **Implementation:**
  - Input:
    - Teacher distribution: `Tensor` `[batch_size, num_classes]`.
    - \( T \): temperature/scaling factor.
  - Similar to Renyi entropy, but with \(\alpha = 1/T\), which is \( <1 \) for typical T>1.
  - Compute:
    ```python
    alpha = 1 / T
    U = torch.sum(torch.pow(p_teacher, alpha), dim=1)
    ```
- **Outputs:**
  - Per-sample weight: `Tensor` `[batch_size]`.

---

## 2. Loss for TTM & WTTM

### 2.1. TTM Loss

Total loss:
\[
\mathcal{L}_{TTM} = H(y, q) + \beta D_{KL}(p_T^t || q)
\]

- Implementation:
  - Given:
    - predictions \( q \) (student logits or probabilities),
    - teacher's transformed distribution \( p_T^t \),
    - label \( y \).

  - Compute:
    - cross_entropy
    - KL divergence

### 2.2. WTTM Loss

Total loss:
\[
\mathcal{L}_{WTTM} = H(y, q) + \beta U_{1/T}(p^t) \times D_{KL}(p_T^t || q)
\]

- Implementation:
  - Compute sample weights \( U_{1/T}(p^t) \) for each sample.
  - Multiply divergence term by these weights element-wise.

### 2.3. Combined Losses with Other Distillation Losses (Optional)

- If combining WTTM with other losses (e.g., CRD, ITRD), add their corresponding losses scaled by \(\mu\).

- Ensure overall loss is a sum of:
  - standard CE
  - weighted divergence
  - other regularizers if specified.

---

## 3. Additional Regularizers

- **Entropy Regularizers:**
  - Optional: implement function to compute Shannon entropy for the teacher or student distributions for analysis.

---

## 4. Implementation Structure

- **Functions to Implement:**
  - `cross_entropy(pred_logits, targets)`
  - `kl_divergence(p, q)`
  - `power_transform_probs(teacher_logits, gamma)`
  - `renyi_entropy(p, alpha)`
  - `compute_sample_weight(p_teacher, T)` or directly `U_{1/T}(p)`
  - `compute_total_loss(student_logits, teacher_logits, targets, hyperparameters)`

- **Constants:**
  - Use hyperparameters from configuration (`config.yaml`) for T (`gamma`), `beta`, and `lambda`.

- **Naming conventions:**
  - Use clear, descriptive names per the mathematical terminology.

---

## 5. Summary

This losses.py module must:
- Provide flexible functions for transforms and entropy calculations.
- Implement core divergence and CE losses.
- Support batch processing.
- Be modular for plugging into training routines.
- Guarantee numerical stability and correctness based on the outlined formulas.

This detailed logical framework ensures all loss-related computations are faithfully implemented, aligned with the mathematical derivations and experimental setup described in the paper.

## main.py

# Logic Analysis for main.py

The role of `main.py` is to serve as the orchestrator for the entire training and evaluation pipeline, following the design plan and ensuring fidelity to the methodology described in the paper. Its primary responsibilities include initializing configurations, setting up datasets, loading models, executing training routines, and performing evaluation. Below is a detailed step-by-step logical flow and reasoning to implement `main.py`:

---

## 1. Load Configuration

- **Objective:** Read all experiment settings, hyperparameters, paths, and dataset options from `config.yaml`.
- **Approach:** Use `PyYAML` to parse the YAML file into a Python dictionary or structured config object.
- **Details to extract:**
  - Dataset info (`name`, `data_dir`, `image_size`)
  - Training hyperparameters (`batch_size`, `epochs`, `learning_rate`, `weight_decay`, `momentum`)
  - Distillation hyperparameters (`T`, `lambda`, `beta`, `TTM_ratio`) 
  - Model specifications (`teacher_architecture`, `student_architecture`, `pretrained_teacher_weights_path`)
  - Optimization parameters (optimizer type and parameters)
  - Logging paths and frequency (`save_checkpoint_dir`, `save_summary_every`, `verify_every`)

---

## 2. Instantiate DatasetLoader

- **Objective:** Prepare data loaders for training, validation, and testing.
- **Approach:**
  - Use a class or function (e.g., in `datasets.py`) to load datasets based on `dataset.name`.
  - Apply appropriate transformations:
    - **Training:** data augmentation (random crop, flip), normalization.
    - **Validation/test:** only normalization.
  - Return PyTorch `DataLoader` objects for:
    - `train_loader`
    - `val_loader`
    - `test_loader`
- **Important considerations:**
  - Ensure batch size matches configuration.
  - Set `shuffle=True` for training.

---

## 3. Load Models

- **Teacher Model:**
  - Instantiate architecture specified by `teacher_architecture`.
  - Load pretrained weights from `pretrained_teacher_weights_path`.
  - Set to evaluation mode (`model.eval()`) during training, but keep for inference.
- **Student Model:**
  - Instantiate architecture specified by `student_architecture`.
  - Initialize randomly or with a specified initialization strategy.
- **Frameworks/libraries:**
  - Use torchvision or custom model definitions as per `models.py`.
- **Additional:**
  - Optionally, move models to GPU (if available) for acceleration.

---

## 4. Define Loss Function(s)

- **Core Distillation Loss (WTTM):**
  - For each batch:
    - Pass input images through teacher and student models.
    - Obtain teacher logits.
    - Compute teacher probabilities: `p = softmax(teacher_logits)`.
    - Compute power transform: \(\hat{p}_i = \frac{p_i^\gamma}{\sum_j p_j^\gamma}\) with \(\gamma = 1/T\).
    - Generate sample-specific weight \(U_{1/T}(p)\).
    - Compute student probabilities: `q = softmax(student_logits)`.
    - Compute student transformed distribution if necessary (`q_T`).
    - Calculate the KL divergence: `D_{KL}(\hat{p}_i || q_i)`.
    - Incorporate sample weight: multiply divergence by \(U_{1/T}(p)\).
    - Combine with cross entropy loss: `H(y, q)`.
  - For WTTM, combine losses as:
    \[
    \mathcal{L} = H(y, q) + \beta U_{1/T}(p) D_{KL}(\hat{p}_T^t || q).
    \]
- **Additional losses (optional):**
  - If the experiment involves other loss terms (e.g., contrastive, feature alignment), prepare as needed with their hyperparameters.

---

## 5. Set Up Optimizer and Scheduler

- Instantiate optimizer, such as SGD, with parameters from config.
- Set learning rate scheduler if specified:
  - E.g., CosineAnnealingLR, StepLR, etc.
- Make sure optimizer updates only student model parameters.

---

## 6. Prepare Training Loop

For each epoch (from 1 to `epochs`):

- **Training phase:**
  - Set student model to train (`model.train()`).
  - For each batch:
    1. Load input images and labels.
    2. Forward pass:
       - Compute teacher logits via teacher model.
       - Compute student logits.
    3. Generate teacher's power-transformed distribution:
       - Calculate teacher probabilities: `p`.
       - Compute \(\hat{p}\) with \(\gamma=1/T\): softmax of scaled logits raised to power.
       - Calculate sample weight: `U_{1/T}(p)`.
    4. Compute loss:
       - Cross entropy with ground truth.
       - KL divergence between teacher's transformed distribution and student.
       - Multiply divergence by sample weight for WTTM.
       - Sum/average over batch.
    5. Backpropagation:
       - Zero gradients.
       - Compute gradients.
       - Step optimizer.
  - Log training metrics (loss, entropy, etc.).

- **Evaluation phase (every `verify_every` epochs or at end):**
  - Set student model to eval (`model.eval()`).
  - Compute top-1 accuracy over validation/test set.
  - Record distribution entropy, divergence, and other metrics.

- **Logging:**
  - Save checkpoints periodically.
  - Save metrics for analysis.

---

## 7. Final Evaluation and Results Saving

- After training:
  - Perform a final evaluation on the test set.
  - Save the best model (according to validation accuracy or other criteria).
  - Log final metrics:
    - Top-1 accuracy
    - Distribution entropy history
    - Divergence progress
- Save logs and checkpoints into specified directory.

---

## 8. Additional Considerations

- Ensure reproducibility:
  - Fix random seeds.
  - Use deterministic algorithms if necessary.
- Use GPU acceleration if available:
  - `cuda`/`cpu` device management.
- Modularize code:
  - Call functions from `datasets.py`, `models.py`, `losses.py`, etc.
  - Keep configuration-driven hyperparameters.

---

## Summary of the Logical Steps in main.py

```plaintext
1. Load config.yaml into a structured config object/dict.
2. Initialize dataset loaders with transformations based on dataset and augmentation configurations.
3. Instantiate teacher and student models with appropriate architectures.
4. Load teacher weights; initialize student.
5. Define optimizer, scheduler, and loss functions, including the WTTM-specific loss with hyperparameters from config.
6. For each epoch:
   - For each batch:
     a. Forward pass through teacher and student.
     b. Compute teacher's power-transformed distribution.
     c. Compute sample weight \(U_{1/T}(p)\).
     d. Calculate WTTM loss (including cross entropy, KL divergence, weighted).
     e. Backpropagate and optimize.
   - If evaluation step:
     - Perform validation.
     - Log accuracy, entropy, divergence.
     - Save checkpoints.
7. Final evaluation, report final metrics.
```

---

This structured logic plan ensures a comprehensive and faithful implementation of `main.py`, aligned with the paper's methodology and the provided plan and configuration. It prepares for detailed coding and integration across all modules.

## models.py

**Logic Analysis for models.py — Model Architectures and Weight Loading**

---

### Purpose & Role:

The `models.py` module is responsible for **defining the neural network architectures** for both teacher and student models, and for **loading pretrained weights** into the teacher model. It must support flexible architecture selection based on configuration, enable consistent model initialization, and manage pretrained weight loading. This setup is essential for reproducibility, ensuring that the teacher models can produce outputs aligned with the experimental setup, and for the student, which is trained from scratch or initialized as needed.

---

### Core Functionalities Required:

1. **Model Architecture Definitions:**
   - **Support multiple architectures:** ResNet50, ResNet18, WideResNet, MobileNetV2, ShuffleNet, optionally other models.
   - **Use torchvision.models when possible** for straightforward models.
   - **Implement or import custom implementations** if architectures are not available in `torchvision`.

2. **Parameterization & Configuration:**
   - Architectures and pretrained weight paths are provided via configuration (`config.yaml`).
   - The code must accept architecture identifiers (e.g., 'ResNet50') and instantiate the correct model with default or ImageNet-pretrained weights.
   - **Ensure architecture string identifiers match across the code and config.**

3. **Pretrained Weights Loading:**
   - For the **teacher model**, load pretrained weights from specified path or url.
   - For **the student model**, initialize randomly unless specified otherwise.

4. **Model Instantiation API:**
   - Provide **functions/classes** that:
     - Given an architecture string, return a model object.
     - Optionally load weights if a path is provided.
   - Example signature:
     ```python
     def get_model(architecture: str, pretrained: bool=False, weights_path: str=None):
         ...
     ```

5. **Compatibility & Consistency:**
   - Ensure output shapes/functions match those expected in the training script.
   - Consistent model evaluation mode (`model.eval()`) handled outside but ensure models are ready for inference.

6. **Device Support:**
   - Accept a `device` argument or handle manually outside.
   - Models should be moved to the correct device (`cpu` or `cuda`) after instantiation.

---

### Implementation Logic:

#### Step 1: Define a model selection function based on architecture string

- Use if-elif-else or dictionary mapping:
  - Example:
    ```python
    model_dict = {
        'ResNet50': torchvision.models.resnet50,
        'ResNet18': torchvision.models.resnet18,
        'WideResNet': custom_wideresnet,
        'MobileNetV2': torchvision.models.mobilenet_v2,
        'ShuffleNetV2': torchvision.models.shufflenet_v2_x1_0,
        # add others as needed
    }
    ```
- For architectures not directly in torchvision, plan to import or implement custom models within the same file or module.

#### Step 2: Instantiate models

- For models available in torchvision:
  - Use corresponding constructor, e.g.,
    ```python
    model = torchvision.models.resnet50(pretrained=False)
    ```
- For custom architectures:
  - Import or define the class, then instantiate, e.g.,
    ```python
    model = CustomWideResNet(depth=28, width=10)
    ```

#### Step 3: Load pretrained weights (for teacher exclusively)

- **Check if pretrained weights path is provided in config:**
  - If yes, load weights via `model.load_state_dict(torch.load(weights_path))`.
  - Ensure weights are compatible; may require model-specific adjustments.
  - If no path provided, assume randomly initialized.

- For pretraining weights:
  - Use `model.eval()` after loading to prevent accidental training.
  - Properly handle models that are pretrained on ImageNet (standard for ResNet, MobileNetV2, etc.).

#### Step 4: Return models

- Return the instantiated model object.
- The calling script (e.g., `train.py`) will handle moving the model to device, setting `train()` or `eval()` modes.

---

### Additional Considerations:

- **ResNet Variants**:
  - For ResNet, use `torchvision.models.resnet50`, `resnet18`.
  - Ensure that `pretrained=False` when loading custom weights to avoid conflicts.
  - For ImageNet training, load ImageNet-pretrained weights if desired.

- **WideResNet**:
  - May not be in `torchvision`. Require an in-project implementation.
  - Assume such a class is available or provide placeholder for user to implement.

- **MobileNetV2 / ShuffleNet**:
  - Use torchvision: `mobilenet_v2`, `shufflenet_v2_x1_0`.
  - For custom variants, extend similarly.

- **Name consistency and expandability**:
  - Make the architecture string mapping easy to extend for future models.

---

### Outline of function in `models.py`:

```python
def get_model(architecture: str, pretrained: bool = False, weights_path: str = None):
    # Define a dict mapping architecture names to model constructors
    model_map = {
        'ResNet50': torchvision.models.resnet50,
        'ResNet18': torchvision.models.resnet18,
        'MobileNetV2': torchvision.models.mobilenet_v2,
        'ShuffleNetV2': torchvision.models.shufflenet_v2_x1_0,
        'WideResNet': custom_wideresnet,  # custom-defined class/function
        # add more architectures as needed
    }
    # Instantiate model
    if architecture not in model_map:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model_fn = model_map[architecture]
    model = model_fn(pretrained=pretrained) if 'pretrained' in model_fn.__code__.co_varnames else model_fn()

    # Load custom weights if provided
    if weights_path:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint)

    return model
```

---

### Summary:

- The `models.py` module must:
  - Provide a flexible factory function/class to instantiate desired models based on string identifiers.
  - Load pretrained weights for the teacher if specified.
  - Support custom architectures for wider applicability.
- The design must facilitate consistent reproduction:
  - Same model configurations for teacher and student.
  - Reproducible weight initialization.
  - Easy extension for new models.

---

**End of Logical Analysis for models.py.**

## requirements.txt

# requirements.txt

# This file summarizes the detailed logical dependencies and structured flow needed to implement the methodology and experiments described in the paper "Knowledge Distillation Based on Transformed Teacher Matching (TTM & WTTM)". It aligns with the plan, design, dataset handling, model architectures, loss functions, hyperparameters, and experimental procedures as specified.

---

## Essential Package Dependencies

- **PyTorch (torch)**: core deep learning framework for defining models, computing losses, gradients, etc.

- **torchvision**: provides model architectures (ResNet, MobileNet, ShuffleNet variants), datasets, and image transforms.

- **NumPy**: numerical operations, especially for probability normalization, dataset statistics, and auxiliary calculations.

- **PyYAML**: loads configuration parameters from `config.yaml`, ensuring hyperparameters and dataset paths are flexible and reproducible.

---

## Additional Logical Dependencies

- **Dataset Loading & Preprocessing**:
  - Use `torchvision.datasets` for CIFAR-100 and ImageNet.
  - Implement image augmentation strategies (random crop, flip, normalization) in data loaders.
  - Provide training and validation iterators with batch size and shuffling controlled via config.

- **Model Definition & Loading**:
  - Instantiate teacher and student models as per architecture names in the configuration.
  - Load pre-trained weights for the teacher from specified paths.
  - Initialize student models randomly for training.

- **Loss Components & Computation**:
  - Implement cross-entropy loss (`nn.CrossEntropyLoss`) for ground-truth supervision.
  - Implement KL divergence loss, with temperature/post-transformation applied:
    - Compute teacher's output probabilities: `p_i = softmax(logits)`.
    - Apply **power transform**: `p_i^\gamma` normalized to sum to 1.
    - Compute student output probabilities `q_i` similarly.
  - Incorporate sample weights in WTTM:
    - Calculate `U_{1/T}(p^t)` for each sample.
    - Normalize or scale as per formula to weight the KL divergence loss.
  - Implement the combined loss:
    \[
    \mathcal{L}_{total} = H(y,q) + \beta \cdot U_{1/T}(p^t) \cdot D_{KL}(p_T^t || q) + \mu \cdot \mathcal{L}_{dist}
    \]
  - Support additional distillation methods like CRD or ITRD (if combined) with their respective hyperparameters.

- **Power Transform & Renyi Entropy**:
  - Implement function to perform the power transform of teacher's probabilities with exponent \(\gamma = 1/T\).
  - Compute sample-wise weight `U_{1/T}` used in WTTM.
  - Calculate the Renyi entropy or related regularization terms as needed (for analysis).

- **Training Loop & Optimization**:
  - For each batch:
    - Forward pass through teacher and student models.
    - Generate teacher probabilities, perform power transform.
    - Compute loss components with appropriate weights.
    - Backpropagate and handle optimizer steps.
  - Schedule learning rate updates (e.g., step decay, cosine annealing).

- **Logging & Evaluation**:
  - Track metrics: top-1 accuracy on validation/test set.
  - Record entropy distribution of student outputs per epoch to visualize regularization effects.
  - Save model checkpoints at intervals set in config.
  - Log loss components and sample-wise metrics for analysis and reproduction.

---

## Reproducibility & Hyperparameter Control

- All hyperparameters (T, beta, lambda, mu, etc.) are to be read from `config.yaml`.
- Hyperparameters such as the dimension of the models, optimizer parameters, and data augmentation are fixed to match experimental setups.
- Multiple runs (3–5) for CIFAR-100 to report mean and standard deviation.
- For ImageNet, follow canonical training schedules and reported settings.

---

## Implementation Logic Summary

- Load configuration parameters.
- Instantiate datasets and data loaders with augmentation.
- Initialize models (teacher from checkpoint, student randomly).
- Prepare loss functions:
  - Custom power transform function.
  - KL divergence with sample weights for WTTM.
- During training:
  - Forward: get teacher logits, compute teacher probabilities, apply power transform.
  - Forward: student logits, compute predictions.
  - Compute cross entropy with labels.
  - Compute distillation loss (with weights if WTTM).
  - Combine losses with hyperparameters.
  - Backpropagate, update student.
- After each epoch:
  - Check validation accuracy.
  - Save checkpoints.
- Final evaluation:
  - Calculation of top-1 accuracy.
  - Entropy and divergence analyses.
  - Optional visualization of output distribution entropy histograms to match Figures 3 and 4.

---

## Final Notes

- Ensure all computations involving probability distributions are numerical stable and normalized.
- Code modules should be modular to allow switching between TTM, WTTM, and other distillation methods.
- Maintain strict alignment with pseudocode and hyperparameter settings provided in the appendices for exact reproducibility.
- Add proper logging, seed control, and resource management to ensure consistency across runs.

---

*This 'requirements.txt' encapsulates the logical dependencies, critical steps, and structured flow necessary for an accurate, faithful implementation aligned with the paper's methodology and experimental protocol.*

## train.py

**Logic Analysis for `train.py` — Training Loop for WTTM (and TTM) with Distillation**

---

### **Overall Purpose of train.py**

- To execute the core training iterations for the student model under the proposed distillation framework:
  - Implement the **transformed teacher matching** loss (TTM or WTTM).
  - Integrate standard supervised learning (**cross-entropy loss** with true labels).
  - Support optional combination with other distillation strategies (e.g., CRD, ITRD).
  - Handle data loading, forward passes, loss computations, backpropagation, and parameter updates.
  - Log training progress, loss values, and primary metrics (accuracy).

---

### **Key Inputs & Data**

- **Model objects:**
  - `student_model` — the trainable neural network (ResNet, MobileNet, etc.).
  - `teacher_model` — pre-trained, fixed teacher, set to evaluation mode.
  
- **Loss hyperparameters:**
  - `T` (temperature, as power exponent) — from config.
  - `beta` (distillation regularization weight).
  - `lambda` (balance for cross-entropy vs distillation — possibly for combined loss).
  - For WTTM, an additional **sample weight scaling factor** derived dynamically.
  
- **Dataset loader:**
  - Batches of input images (`images`) and labels (`targets`).

- **Optional:**
  - Precomputed teacher logits (if stored), or computed on-the-fly.
  - For efficiency, prefer to compute teacher logits per batch during training rather than preloading entire set.

---

### **Main Step-by-step Logic Flow**

#### **1. Initialization for Each Batch**
- Load the input batch:
  ```python
  images, targets = next(batch_iterator)
  ```
- Ensure models are in training mode for `student_model`:
  ```python
  student_model.train()
  teacher_model.eval()
  ```
- Moving data to device (GPU/CPU):
  ```python
  images = images.to(device)
  targets = targets.to(device)
  ```

#### **2. Forward Pass of Teacher and Student**
- Compute both models’ output logits:
  ```python
  with torch.no_grad():
      teacher_logits = teacher_model(images)  # shape: [batch_size, num_classes]
  student_logits = student_model(images)   # shape: [batch_size, num_classes]
  ```
- Set teacher output to `torch.no_grad()` to prevent gradients.

#### **3. Compute Teacher’s Power-Transformed Distribution (`p_T^t`)**
- Apply softmax to teacher logits:
  ```python
  teacher_probs = F.softmax(teacher_logits, dim=1)
  ```
- Compute the `p_T^t` distribution:
  - Use the `gamma = 1/T` value from config.
  ```python
  gamma = 1.0 / T
  p_t = torch.pow(teacher_probs, gamma)
  ```
- Normalize `p_t` to form a probability distribution:
  ```python
  p_t_sum = p_t.sum(dim=1, keepdim=True)
  p_t_norm = p_t / p_t_sum
  ```
- This is the **power transformed teacher distribution**.

#### **4. Compute Sample-Adaptive Weight (`U_{1/T}(p^t)` for WTTM)**
- For WTTM, compute the **power sum** for each sample:
  ```python
  U = p_t_norm.sum(dim=1)  # For each sample in batch
  ```
- For the general scenario:
  - The sample weight is proportional to `U`, which emphasizes smoother teacher soft targets.
  
- If normalization of weights over batch is needed (as per paper/protocol), do so accordingly:
  ```python
  weight_scaling_factor = U.mean()  # or keep as vector U for per-sample weights
  ```

#### **5. Compute Student Distribution & Distillation Loss**
- Compute student softmax outputs:
  - Using the raw logits:
  ```python
  student_probs = F.softmax(student_logits, dim=1)
  ```
- **Compute the power-transformed student distribution (\(\hat{q}\)) as in the paper:**
  ```python
  q_transform = torch.pow(student_probs, gamma)
  q_norm = q_transform / q_transform.sum(dim=1, keepdim=True)
  ```
- **Calculate the KL divergence**:
  - Use `F.kl_div` with `log_target=False`, or prefer manually:
  
    ```python
    kl_loss = torch.sum(q_norm * torch.log((q_norm + epsilon) / (p_t_norm + epsilon)), dim=1)
    ```
  - Use `epsilon` to ensure numerical stability.

- For **WTTM**, multiply the per-sample `kl_loss` by the sample weights:
  ```python
  distillation_loss = torch.mean(U * kl_loss)
  ```
- For **TTM (without weight)**:
  ```python
  distillation_loss = torch.mean(kl_loss)
  ```

#### **6. Compute Cross Entropy with Ground Truth**
- Standard supervised cross entropy:
  ```python
  ce_loss = F.cross_entropy(student_logits, targets)
  ```

#### **7. Compute Total Loss**
- Combine losses:
  - **Basic TTM/WTTM:**
    \[
    \mathcal{L} = (1 - \lambda) \times ce + \lambda \times \beta \times distillation\_loss
    \]
  - **Including other distillation components** (e.g., CRD, ITRD):
    \[
    \mathcal{L}_{total} = ce\_loss + \mu \times \text{additional\_loss}
    \]
- Hyperparameters are obtained from config.yaml for \(\lambda, \beta, \mu\).

```python
loss = (1 - lambda_) * ce_loss + mu * (beta * distillation_loss + other_dist_loss)
```

#### **8. Backpropagation and Optimization**
- Reset gradients:
  ```python
  optimizer.zero_grad()
  ```
- Backward:
  ```python
  loss.backward()
  ```
- Update parameters:
  ```python
  optimizer.step()
  ```

---

### **Additional Details & Considerations**

- **Training stability:**
  - Use gradient clipping if necessary.
  - Clip logits or probabilities if numerical issues occur.
  
- **Logging & Monitoring:**
  - Record:
    - Loss components (`ce_loss`, `distillation_loss`)
    - Total loss
    - Model accuracy on validation subset.
    - Entropy of output distributions as in the paper figures.
    - Divergences and sample weights (optional).
        
- **Scheduler:**
  - Implement learning rate scheduler to follow original training schedule.

- **Multiple epochs:**
  - Loop over dataset for a fixed number of epochs, as specified.

- **Checkpoints:**
  - Save model states periodically, especially when achieving better validation performance.

---

### **Summary in Pseudocode**

```python
for epoch in range(total_epochs):
    for batch in data_loader:
        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        # Forward pass for teacher and student
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        student_logits = student_model(images)

        # Softmax probabilities
        teacher_probs = F.softmax(teacher_logits, dim=1)

        # Compute power-transformed teacher distribution p_T^t
        gamma = 1.0 / T
        p_t = torch.pow(teacher_probs, gamma)
        p_t_norm = p_t / p_t.sum(dim=1, keepdim=True)

        # Compute sample-dependent weight (WTTM)
        U = p_t_norm.sum(dim=1)

        # Student softmax and power transform
        student_probs = F.softmax(student_logits, dim=1)
        q_transform = torch.pow(student_probs, gamma)
        q_norm = q_transform / q_transform.sum(dim=1, keepdim=True)

        # KL divergence per sample
        epsilon = 1e-8
        kl_per_sample = torch.sum(q_norm * torch.log((q_norm + epsilon) / (p_t_norm + epsilon)), dim=1)

        # Apply sample weighting for WTTM
        if WTTM:
            sample_weight = U
            distillation_loss = torch.mean(sample_weight * kl_per_sample)
        else:
            distillation_loss = torch.mean(kl_per_sample)

        # Cross entropy with true labels
        ce_loss = F.cross_entropy(student_logits, targets)

        # Total loss
        total_loss = (1 - lambda_) * ce_loss + mu * beta * distillation_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Logging
        update_metrics()
```

---

### **Final Notes**

- The loss computation strictly follows the formulas:
  - Power transform for teacher probabilities.
  - KL divergence between student \(\hat{q}\) and teacher \(\hat{p}\).
  - Dynamic sample weights based on the teacher output smoothness.
- Ensure all operations support batch processing and are numerically stable.
- Consistently use hyperparameter values from the `config.yaml`.
- Modularize components (computation of `p_T^t`, sample weights, loss functions) for clarity, reuse, and easier hyperparameter tuning.

This detailed logic analysis ensures the implementation aligns with the paper's theoretical insights and experimental procedures.

## utils.py

# Logic Analysis for utils.py

The `utils.py` module will serve as a collection of utility functions to support core processes in dataset handling, model operations, mathematical transformations, and logging. The functions should be modular, reusable, and aligned with the paper's methodology and experimental setup.

Below is a detailed, step-by-step logical breakdown of the functions and components to implement, including their purpose, inputs, outputs, and any relevant considerations derived from the paper and plan.

---

## 1. Configuration Loading

**Purpose:**  
Provide a function to load hyperparameters and experimental settings from the `config.yaml` file, ensuring that all modules can access consistent configuration values.

**Design Considerations:**  
- Use `PyYAML` to parse `config.yaml`.
- Return a dictionary or a structured object (e.g., namespace or custom class) containing all hyperparameters.
- Handle missing keys gracefully; set defaults if necessary.

**Implementation Highlights:**  
- Function Signature: `def load_config(path: str) -> dict`
- Example: `config = load_config('./config.yaml')`

---

## 2. Probability Transformation Functions

### 2.1 Power Transform of Teacher Outputs

**Purpose:**  
Implement the core transformation \( p_i^{\gamma} \) normalized over all classes, corresponding to the interpretation of temperature as a power `γ`.

**Input:**  
- `teacher_logits`: Tensor of shape `(batch_size, num_classes)` (logits from the teacher model)  
- `gamma`: float, the exponent for the power transform, derived from `T` as `γ=1/T`.

**Output:**  
- Probability tensor `p_power`: `(batch_size, num_classes)` with each class probability after power transform and normalization.

**Logic:**  
- Apply softmax to teacher logits to obtain `p_teacher`.
- Compute `p_i^{\gamma}` for each sample.
- Normalize `p_i^{\gamma}` to sum to 1 per sample.

**Mathematical details:**  
\[
\hat p_i = \frac{ p_i^\gamma }{ \sum_j p_j^\gamma }
\]

**Additional notes:**  
- Use `torch.clamp()` if numerical stability issues occur (e.g., zero probabilities).
- Ensure no NaNs or Infs during computations.

### 2.2 Calculation of U\(_{\alpha}\) (Sample-Adaptive Weight)

**Purpose:**  
Compute the power sum \( U_\alpha(p) = \sum_j p_j^\alpha \), which acts as a measure of the teacher distribution’s smoothness.

**Input:**  
- `teacher_probs`: `(batch_size, num_classes)`

- `alpha`: float, e.g., \( \alpha = 1/T \)

**Output:**  
- Tensor of shape `(batch_size, 1)` or `(batch_size,)` for the sum per sample.

**Logic:**  
- Raise probabilities to the power `alpha`.
- Sum over classes for each sample.

---

## 3. Entropy Calculations

### 3.1 Shannon Entropy

**Purpose:**  
Compute the Shannon entropy \( H(p) = - \sum_j p_j \log p_j \) of a probability distribution.

**Input:**  
- `probs`: `(batch_size, num_classes)`

**Output:**  
- `entropy`: `(batch_size,)`

**Implementation Details:**  
- Use `torch.clamp()` to avoid `log(0)` issues (`p_j > 0`).
- Sum over classes with negative sign.

### 3.2 Renyi Entropy

**Purpose:**  
Compute the Renyi entropy of order \( \alpha \):

\[
H_\alpha(p) = \frac{1}{1 - \alpha} \log \sum_j p_j^\alpha
\]

- This generalizes Shannon entropy (limit as \( \alpha \to 1 \)).

**Input:**  
- `probs`: `(batch_size, num_classes)`  
- `alpha`: float, typically \( 1/T \).

**Output:**  
- `renyi_entropy`: `(batch_size,)`

**Considerations:**  
- Handle `alpha` near 1 carefully (use numerical stability or approximate limit).
- Use `torch.clamp()` to prevent `log(0)`.

---

## 4. Save & Load Checkpoints

**Purpose:**  
Facilitate saving and loading model states during training.

**Functions:**  
- `def save_checkpoint(model, optimizer, epoch, path: str) -> None`
- `def load_checkpoint(model, optimizer, path: str) -> None`

**Logic:**  
- Save state dictionaries for model and optimizer.
- Load into model, optimizer as needed.

---

## 5. Miscellaneous Helper Functions

### 5.1 Model Checkpoints Path Management

**Purpose:**  
Create paths based on experiment names, model architectures, and epochs for saving/loading.

### 5.2 Data Normalization (if required)

**Purpose:**  
Provide standard normalization functions matching dataset statistics.

- For CIFAR-100: mean and std normalization as per typical values.
- For ImageNet: similar.

---

## 6. Model & Dataset Utilities

**Note:**  
Although mainly in other modules, include minimal helper functions if needed (e.g., dynamically instantiate models).

---

## 7. Practical Considerations

- Use `torch.no_grad()` during inference for efficiency.
- Ensure numerical stability across transformations.
- Add comments and docstrings for clarity, aligning with the coding standards.

---

## Summary of Required Functions in `utils.py`

| Function Name                                | Purpose                                              | Inputs                                    | Outputs                      | Notes                                               |
|----------------------------------------------|------------------------------------------------------|------------------------------------------|------------------------------|-----------------------------------------------------|
| `load_config(path: str) -> dict`             | Load configuration hyperparameters                  | `path`: str                              | `dict`                       | Use `PyYAML`                                        |
| `compute_power_probs(teacher_logits, gamma)`  | Power transform of teacher logits                     | `teacher_logits`, `gamma` (float)        | `(batch_size, num_classes)`  | Normalize after power transform                     |
| `compute_U(teacher_probs, alpha)`             | Compute U\(_{\alpha}\) (sample weight)               | `teacher_probs`, `alpha` (float)         | `(batch_size,)` or `(batch_size,1)` | Sum over classes                                    |
| `compute_shannon_entropy(probs)`              | Compute Shannon entropy                              | `probs`: `(batch_size, num_classes)`    | `(batch_size,)`              | Use `-sum(p*log(p))`                                |
| `compute_renyi_entropy(probs, alpha)`         | Compute Renyi entropy                                | `probs`, `alpha`                        | `(batch_size,)`              | Handle `alpha` near 1 carefully                     |
| `save_checkpoint(model, optimizer, epoch, path)` | Save model checkpoint                               | `model`, `optimizer`, `epoch`, `path`   | None                         | Save `state_dict`s                                 |
| `load_checkpoint(model, optimizer, path)`     | Load model checkpoint                                | `model`, `optimizer`, `path`            | None                         | Load `state_dict`s                                |

---

## 8. Final Notes

- All functions should be compatible with batch processing.
- Use torch functions for differentiable parts, numpy if needed for auxiliary computations.
- Incorporate configuration parameters (e.g., T, beta) from loaded config where applicable.
- Document functions adequately for readability and traceability with respect to the paper’s methodology.

---

This detailed logic analysis ensures `utils.py` will provide robust, correct, and efficient support functions aligned with the theoretical and experimental contributions of the paper, supporting a faithful reproduction of results.

