# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

### Logic Analysis for `dataset_loader.py` — DatasetLoader Class

---

#### Purpose:
Implement a `DatasetLoader` class responsible for loading datasets, applying class-incremental splits according to specified schemes, shuffling class order with a fixed seed, and providing data loaders for each incremental phase in continual learning experiments.

---

### Core Responsibilities:
1. **Dataset Initialization:**
   - Load datasets (e.g., CIFAR-100, ImageNet-100, Tiny-ImageNet).
   - Store full dataset and class labels.

2. **Class Split Scheme Handling:**
   - Support different split schemes:
     - `'class_incremental'`: Equal number of classes per phase.
     - `'half_then_increment'`: First half of classes in initial phase, remaining classes split into incremental phases.
   - Determine number of classes per phase based on total classes and `classes_per_task`.

3. **Class Order Shuffling:**
   - Shuffle class order using a fixed seed (`seed` parameter).
   - Ensures reproducibility of class sequences across runs.

4. **Create Phase-wise Class-to-Data Mappings:**
   - For each phase, identify which classes are included.
   - Store class-to-sample mappings to facilitate retrieval of data for each phase.

5. **Build Data Loaders for Each Phase:**
   - For training:
     - Filter dataset samples belonging to classes in current phase.
     - Create DataLoader with specified batch size.
     - Apply necessary transforms (standard normalization, data augmentation if needed).
   - For evaluation:
     - Optionally prepare a combined test set or class-wise test set for overall evaluation.

6. **Reproducibility & Configurability:**
   - Use configuration parameters:
     - `dataset_name`: e.g., `'CIFAR-100'`
     - `split_scheme`: `'class_incremental'` or `'half_then_increment'`
     - `classes_per_task`: number of classes per incremental step
     - `total_tasks`: total number of phases
     - `seed`: fixed for class order shuffling

7. **Additional Utilities:**
   - Methods:
     - `get_task_dataloader(task_index)`: returns DataLoader for the specified task phase.
     - `get_full_dataset()`: possibly return the entire dataset with class labels.
     - Optionally, `get_test_dataset()`: entire or class-specific test datasets for evaluation purposes.

---

### Implementation Details:

#### 1. Data Loading:
- Use `torchvision.datasets`:
  - `'CIFAR-100'`: `datasets.CIFAR100(root, train=True/False, transform, download=True)`.
  - `'ImageNet-100'`: Custom loader or subset from ImageNet; may require specific handling.
  - `'Tiny-ImageNet'`: Load from local path or subset if provided.
  
- Load training and test datasets separately.
- Store samples and labels in internal data structures (e.g., list or dataset object).

#### 2. Class Order Shuffling:
- Generate a list of class labels `[0, 1, ..., N-1]`.
- Shuffle with `np.random.seed(seed)` and `np.random.shuffle`.
- Save the class order to be used for consistent splitting.

#### 3. Class-to-Samples Mapping:
- After loading dataset, map class labels to sample indices or sample objects.
- Create a dictionary:
  ```python
  class_to_samples = {
    class_label: list of (sample, label)
  }
  ```
  
#### 4. Class Incremental Splits:
- Based on the chosen scheme:
  - `'class_incremental'`:
    - Divide classes into `total_tasks`, each with `classes_per_task`.
  - `'half_then_increment'`:
    - First task: first half of classes.
    - Subsequent phases: remaining classes divided evenly.
- Record class sets for each task index.

#### 5. Data Loader Construction:
- For each task:
  - Collect all samples from current class set.
  - Create a subset of the dataset containing only these samples.
  - Instantiate DataLoader with defined `batch_size`.
  - Use consistent transformations.
  - Ensure shuffling within each DataLoader for stochasticity.

#### 6. Reproducibility:
- Dependence on fixed seed for:
  - Class order shuffling.
  - Data shuffling in DataLoader (if shuffling enabled).
- Consistent data splits.

#### 7. Output:
- Provide methods:
  - `get_task_dataloader(task_index)`: yields DataLoader for the specified incremental phase.
  - Possibly keep an internal mapping: `task_index -> DataLoader`.
  
- Maintain a list of class splits per task for reference.

---

### Edge Cases & Additional Considerations
- Dataset download failures or missing data: handle with try-except.
- If dataset requires special handling (e.g., ImageNet subset and folder structure), implement flexible loading.
- For reproducibility, set `torch.manual_seed()` and `np.random.seed()` at loader initialization.
- Can cache class-to-samples mapping for efficiency if used repeatedly.
- Since the plan suggests one dataset at a time, implement a flexible initialization to accommodate multiple dataset types.

---

### Summary:
The `DatasetLoader` class must be able to:
- Load datasets according to configuration.
- Shuffle class order with a fixed seed for reproducibility.
- Split classes per the selected scheme into phases.
- Map each class to dataset samples.
- Provide phase-specific DataLoaders with appropriate data filters.
- Ensure data shuffling and transformations are consistent.
- Expose utility methods for retrieving data loaders and data sets relevant to training/testing phases.

This structured approach will enable reproducible, flexible, and efficient data provisioning aligned with the experimental design of the paper.

## evaluation.py

# Logic Analysis for `evaluation.py`

This module provides the `Evaluation` class responsible for evaluating the model's performance after each incremental phase in continual learning, visualizing the loss landscape, and analyzing the model's flatness via Hessian spectrum estimations. The class interfaces with the model, dataset loader, visualization tools (e.g., PyHessian), and utility functions from `utils.py`. The key objectives are:

- **Accurate performance measurement** on all seen classes.
- **Landscape visualization** at selected epochs to illustrate the effect of C-Flat on the loss surface.
- **Hessian eigenvalue computation** to quantify landscape flatness.
- **Logging and plotting** for analysis and comparison during experiments.

---

## 1. Initialization (`__init__`)

Inputs:
- `model`: the neural network to be evaluated.
- `dataset_loader`: object managing data splits and task data.
- `config`: dictionary containing evaluation parameters and settings.

Responsibilities:
- Store references to the dataset and model.
- Determine evaluation frequency and visualization options based on `config`.
- Initialize logging tools (e.g., directories, files) for saving figures, logs, Hessian eigenvalues.
- If visualization tools like `PyHessian` are to be used, initialize them with the current model parameters.

---

## 2. Core Methods

### `evaluate()`

- **Purpose**: run evaluation over all **seen classes** after each phase.
- **Procedure**:
  - For each task data/dataset:
    - Run inference on the test set.
    - Compute accuracy metrics.
  - Aggregate accuracy over all classes seen so far.
  - Store or log these metrics.

- **Implementation details**:
  - Loop over all data loaders or datasets representing previous and current tasks.
  - Accumulate accuracies and compute average accuracy.
  - Handle multiple runs if needed for averaging.

### `visualize_landscape()`

- **Purpose**: plot the loss landscape around the current model parameters.
- **Inputs**:
  - Current model state.
  - Model loss surface (computed via PyHessian or custom routines).
  - Eigenvectors/eigenvalues for the surface.
- **Procedure**:
  - Use `PyHessian` or custom code to generate 2D cross-sections of the loss surface.
  - Plot loss surface contours and model parameter perturbations.
  - Save figures to specified directory.
- **Frequency**:
  - Based on epochs or phases as scheduled (`config.evaluation.landscape_visualization`).

### `compute_hessian_eigenvalues()`

- **Purpose**: calculate the top eigenvalues/eigenvectors of the Hessian matrix of the loss at model parameters.
- **Method**:
  - Use Hessian-vector product approximation via `torch.autograd.functional.hvp` or `torch.autograd.grad`.
  - Implement spectral algorithms like Lanczos or power iteration to estimate leading eigenvalues.
  - Possibly utilize `PyHessian` if permitted.
- **Output**:
  - List or array of top eigenvalues/eigenvectors.
  - Store or log these values for correlation with landscape flatness.

### `record_metrics()`

- **Purpose**: log or save metrics after each evaluation.
- **Metrics**:
  - Accuracy on all seen classes.
  - Forgetting measure (initial vs last performance).
  - Hessian eigenvalues, landscape plots.
- **Implementation**:
  - Append to CSV, JSON, or in-memory logs.
  - Optionally generate plots (accuracy over phases).

---

## 3. Utility Functionality

- **Landscape visualization**:
  - Generate 2D slices by perturbing parameters along top Hessian eigenvectors.
  - Plot the loss surface with annotated minima or perturbation directions.
- **Hessian calculation**:
  - Use efficient approximations to avoid full Hessian.
  - For each eigenvalue:
    - Set a random vector or eigenvector estimate.
    - Compute Hessian-vector product.
- **Metrics and summaries**:
  - Compute and interpret Hessian spectral measures.
  - Correlate flatness (via eigenvalues) with accuracy and forgetting.

---

## 4. Data Handling & Implementation Details

- **Data inputs**:
  - Access data via `dataset_loader.get_task_dataloader()` for current and past tasks.
- **Model state**:
  - Use `model.eval()` to switch to evaluation mode.
  - Save/restore model state dicts as needed for visualizations.
- **Visualization tools**:
  - If using `PyHessian`:
    - Initialize with current model parameters.
    - Generate surface plots along eigenvectors.

---

## 5. Implementation Scheduling & Evaluation Frequency

- Run evaluations at specified intervals:
  - After each epoch or fixed epoch intervals (from `config.evaluation.metrics`).
  - During landscape visualization: at selected epochs (~every few epochs or only at end).
- Call `visualize_landscape()` periodically, e.g., every 10 epochs.
- Compute Hessian eigenvalues less frequently due to cost, e.g., every 50 epochs.

---

## 6. Output and Logging

- Store detailed logs:
  - Accuracy curves.
  - Flatness measures (eigenvalues, traces).
  - Loss landscape figures.
- Save figures to disk in a structured directory (e.g., `logs/experiment1/landscape_phaseX.png`).
- Export numerical data for further analysis.

---

## 7. Additional Notes

- **Consistency and reproducibility**:
  - Use the same random seed and configuration for Hessian estimation.
  - Consistently evaluate on all seen data up to current phase.
- **Handling large models and datasets**:
  - Use approximation methods to compute the top few Hessian eigenvalues.
  - Limit the number of perturbation points for visualization for efficiency.
- **Extension**:
  - Potential to support different landscape visualization methods (e.g., Random Directions, Eigenvector directions).

---

### Summary:
The `Evaluation` class is a comprehensive analysis module that, at specified intervals, evaluates model accuracy over all seen classes, visualizes the loss landscape in a perturbation space, estimates landscape flatness through Hessian eigenvalues, and logs all findings for comparison. It should be designed with modularity, efficiency (approximate Hessian computations), and clear logging/visualization pipelines in mind, supporting hyperparameter configurations from `config.yaml`.

---

## main.py

# Logic Analysis: main.py

This script serves as the main orchestrator for executing the continual learning experiment as outlined in the paper's methodology, the provided plan, and the configuration file. Its core responsibilities include initialization, dataset loading, handling incremental training phases, invoking training routines with C-Flat regularization, evaluating performance, visualizing landscapes, and managing logs and checkpoints.

Below is a detailed breakdown of the logic and structure required for main.py, aligned with the specified dependencies and the approach advocated by the paper:

---

## 1. Import Dependencies
- Import necessary libraries: torch, numpy, os, and modules from dataset_loader.py, model.py, trainer.py, evaluation.py, and utils.py.
- Import configuration loader (e.g., yaml) to read config.yaml.
- Import logging or visualization libraries (matplotlib, possibly PyHessian).

## 2. Load Configuration
- Read and parse 'config.yaml'.
- Extract all relevant parameters:
  - training parameters (learning_rate, batch_size, epochs, schedule, regularization \(\rho, \lambda\), neighborhood_eval_per_epoch)
  - model parameters (architecture, optimizer, scheduler Type, milestones, decay)
  - dataset details (name, split_scheme, classes_per_task, total_tasks, seed)
  - evaluation and logging settings (metrics, output directory, save frequency)
  - hardware settings (GPU, multi-GPU)
  - miscellaneous (random seed for reproducibility)

## 3. Set Random Seeds & Device
- Set random seed for numpy, torch, and possibly cuda.
- Determine device availability:
  - If GPU enabled and available, set device to cuda; else cpu.
- Configure multi-GPU settings if applicable, using torch.nn.DataParallel or DistributedDataParallel.

## 4. Initialize Dataset Loader
- Instantiate `DatasetLoader` object with split scheme, seed, and dataset name.
- Call load_data() to obtain:
  - training data split per task: list of datasets or data loaders
  - total number of tasks (T_total)
- For each task, the loader should provide data specific to that phase (e.g., train and test loaders).

## 5. Initialize Model
- Instantiate the model architecture specified (ResNet-18 or as needed).
- Initialize optimizer with model parameters and optimizer_params from config.
- Configure the learning rate scheduler per the schedule specified.
- Load checkpoint if resuming from previous state (optional).

## 6. Initialize Evaluation & Logging
- Prepare output directories for logs, checkpoints, and landscape plots.
- Set up logging (e.g., print, tensorboard, or custom logger).
- Prepare lists/dicts to store accuracy, forgetting, landscape data, Hessian eigenvalues.

## 7. Loop Over Tasks (Incremental Phases)
For each task index from 1 to total_tasks:
  
### 7.1. Data Preparation
- Obtain current task's data loader via dataset_loader.get_task_dataloader(task_idx).
- For rehearsal or exemplar-based approaches:
  - Prepare rehearsal data if applicable (from memory buffer or exemplars stored in previous tasks).

### 7.2. Initialize or Update Model State
- For first task: initialize model parameters randomly.
- For subsequent tasks: load previous task's model parameters, possibly expand if using expansion-based methods.

### 7.3. Training with C-Flat Regularization
- Instantiate `trainer.Trainer` object with current model, data loader, and config parameters.
- Invoke `train_phase(task_idx)` method:
  - Loop for each epoch (bounded by config's epochs):
    - For each batch:
      - Compute gradient of task data loss.
      - Evaluate regularizers \( R_\rho^{0} \) and \( R_\rho^{1} \):
        - Perturb parameters based on the gradient and radius \(\rho\),
        - Approximate loss difference and gradient norms,
        - Use autograd to compute Hessian-vector products or approximate Hessians,
        - Sum regularization contributions scaled by \(\lambda\).
      - Compute total loss as task loss + regularizers.
      - Perform optimizer step.
    - Schedule learning rate decay as per schedule configuration.
    - Optionally evaluate neighborhood regularizers per epoch if configured.
  
### 7.4. Save Model Checkpoints
- Save the model state_dict periodically per save frequency.
- Save the final model after completing all epochs for the task.

### 7.5. Store Exemplar Data (if rehearsal-based)
- Update memory buffer/exemplar set based on the current task's data.
- Ensure that rehearsal data is available for subsequent tasks.

### 7.6. Evaluation & Landscape Visualization
- Evaluate current model on:
  - All seen classes (or test set) using evaluation.py.
  - Compute accuracy, forgetting metrics.
- Visualize the loss landscape around current minima:
  - Generate loss surfaces via PyHessian or custom routines.
  - Save landscape plots to output directory.
- Compute Hessian eigenvalues:
  - Use Hessian-vector products for eigenvalue estimation.
  - Record eigenvalues and trace metrics.
- Log evaluation metrics and landscape metrics.

### 7.7. Logging & Visualization
- Log training/validation accuracy, forgetting, landscape plots, Hessian metrics.
- Possibly generate early snapshots or summaries for reporting.

## 8. Finalization
- After all phases, produce comprehensive reports:
  - Plot accuracy over phases.
  - Plot Hessian eigenvalue distribution and landscape summaries.
  - Summarize improvements (boosts) compared to baselines if applicable.
- Save all logs, plots, and final model.

## 9. Clean Up
- Close log files or visualization tools.
- Save final checkpoint models.
- Optionally, release resources (GPU memory cleanup).

---

## 10. Additional Considerations
- The evaluation of landscape metrics should be performed periodically (e.g., every epoch or every N epochs), controlled via `evaluation.evaluation_frequency`.
- Ensure reproducibility by fixing random seed across all components.
- Include exception handling for robustness.
- Keep the code modular: main.py should orchestrate, delegating dataset handling, training, evaluation, and visualization to respective modules.

---

This detailed logic analysis provides a comprehensive plan for main.py, ensuring the proper orchestration of all steps needed to accurately implement the methodology and experimental protocol described in the paper, aligned with all dependencies, hyperparameters, and configuration directives.

## model.py

**Logic Analysis for `model.py`**

---

### **Purpose and Responsibilities**

- Define the neural network architecture for continual learning experiments, specifically ResNet-18 (or other architectures if specified).
- Support methods for:
  - Forward pass.
  - Parameter perturbation within radius \(\rho\) (needed for regularizer computation).
  - Save and load model checkpoints.
- Support potential model expansion (for expansion-based CL methods), if applicable.
- Ensure compatibility with the training and regularization routines in the overall pipeline.

---

### **Key Components and Functions**

#### 1. **Model Architecture**

- **Design:**
  - Implement `ResNet-18` architecture (or configurable via parameters).
  - Use standard PyTorch modules (`torch.nn.Module`).
  - Support optional expansion modules or additional components if specified (e.g., new branches, dynamic modules). For now, focus on the baseline architecture.

- **Implementation:**
  - Initialize the entire network with `torch.nn.Sequential` or subclass `nn.Module`.
  - Consistent with reproducibility, ensure initialization is deterministic or configurable.
  - Support for flexibility: allow passing in a `config` object (dict) that can specify architecture variants, such as `ResNet18` or `ResNet34`, if future extension is desired.

#### 2. **Forward Method**

- **Input:**
  - Input images/tensors, e.g., batch of images.
- **Output:**
  - Class logits or features, depending on the model configuration.
- **Implementation:**
  - Build the `forward()` method that passes input through the ResNet backbone.
  - Return final output logits for classification.

#### 3. **Parameter Perturbation Method**

- **Purpose:**
  - Generate a perturbed version of the model parameters, moving within the neighborhood radius \(\rho\), for regularizer calculations.
  
- **Design:**
  - Define a method `perturb_params(rho: float)`:
    - Compute the gradient (already available during training).
    - Calculate the normalized gradient direction:
      \[
      \delta = \rho \cdot \frac{\nabla_\theta \ell}{\|\nabla_\theta \ell\| + \epsilon}
      \]
    - Add the perturbation to each model parameter:
      \[
      \theta' = \theta + \delta
      \]
    - Store the perturbed parameters separately or modify in-place carefully (preferably in a non-destructive manner).

- **Implementation details:**
  - Use `torch.no_grad()` for in-place perturbation.
  - Instead of adding directly, maintain an auxiliary copy or temporarily modify parameters.
  - Incorporate a small \(\epsilon\) (e.g., 1e-8) to prevent division by zero.

- **Note:**
  - For efficiency, only perturb parameters that are trainable (excluding buffers or non-learnable states).
  - The perturbation is for evaluation of regularizers, not for model training directly.

#### 4. **Save and Load Checkpoints**

- **Methods:**
  - `save_checkpoint(filepath: str)`:
    - Save the model state_dict.
  - `load_checkpoint(filepath: str)`:
    - Load state_dict into the model.
  
- **Implementation:**
  - Use `torch.save()` and `torch.load()`.
  - For compatibility, ensure strict loading.
  - Save additional info, such as optimizer state if needed, but primarily model weights.

#### 5. **Model Expansion Support (Optional/Extended)**

- **Purpose:**
  - Support expansion modules, e.g., new branches, additional layers.
  
- **Design:**
  - Implement optional submodules like `expand_module` that can be activated or replaced.
  - Support methods to freeze or unfreeze certain parts during training.
  - Placeholder methods for adding new components, if applicable for expansion-based CL methods.

---

### **Implementation Details & Constraints**

- **Imports:**
  - `torch.nn` for model layers.
  - `torch.nn.functional` if needed.
  - `torch.optim` only if model-specific optimizers are attached here (though typically outside).
  
- **Determinism:**
  - Fix seed globally (from configuration), ensure deterministic init if needed.
  
- **Modularity & Extensibility:**
  - Design `class ResNet18(nn.Module)` that allows future modifications.
  
- **Additional Notes:**
  - If hyperparameters for the architecture are provided in a config (e.g., number of layers, block types), parse and support them.
  - Support returning features if needed for some CL methods, especially for feature regularization.

---

### **Outline of Core Class and Methods**

```python
class ResNet18(nn.Module):
    def __init__(self, num_classes=100, expand_modules=None, config=None):
        super().__init__()
        # Build backbone: standard ResNet-18 or configurable
        # TODO: support expansion modules if specified
        pass

    def forward(self, x):
        # Implement forward pass
        return logits

    def perturb_params(self, rho):
        # Perturb trainable parameters within neighborhood radius rho
        # Based on gradient norm
        pass

    def save_checkpoint(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.load_state_dict(torch.load(filepath))
```

### **Additional Considerations**

- **Consistency with training loop:**
  - The perturbation method will be invoked during regularizer computation, which should be coordinated with gradient computations.
- **Handling of GPU/Device:**
  - Make sure model and parameters are on the correct device (GPU/CPU).
- **Flexibility for other architectures:**
  - For experimental purposes, support passing a custom architecture or an existing model constructor.

---

### **Summary**

- The `model.py` will define a `ResNet18` class (or configurable model).
- It will provide:
  - Standard forward pass.
  - `perturb_params(rho)` method to generate neighborhood perturbations based on current gradients.
  - Methods for saving and loading checkpoints.
- Designed to facilitate the `C-Flat` regularization routine, with minimal coupling, and extensibility for expansion modules in future CL variants.

This thorough understanding ensures precise implementation aligned with the paper's methodology, experimental setup, and hyperparameters.

## requirements.txt

# requirements.txt

# Core deep learning framework
torch==1.11.0

# Numerical computing and array operations
numpy==1.21.0

# Visualization tools for landscape plots and Hessian eigenvalues
matplotlib==3.5.0

# Scikit-learn for additional utilities, possibly for evaluation metrics or class splitting
scikit-learn==0.24.2

# HDF5 support if saving/loading large datasets or models
h5py==3.1.0

# Auxiliary packages for reproducibility, optional but recommended
# (e.g., for dataset handling, progress bars, or logging)
# No additional third-party dependencies specified beyond core packages

# Additional notes:
# - The code will depend heavily on PyTorch’s autograd for neighborhood regularizer approximations,
#   including gradient computation and Hessian-vector products.
# - For visualization of loss landscapes and Hessian eigenvalues, PyHessian or custom routines
#   will be used; ensure compatible versions if explicitly required.
# - No dependencies on special packages like GPytorch or Hessian approximation libraries are specified, 
#   but implementing efficient Hessian-vector products may require them if desired.
# - Ensure reproducibility with seed settings in code, as the seed is specified as 1993.
# - No other language dependencies are expected; all code should be written in Python.
# - For hardware considerations, use GPU acceleration if available, but code must support CPU fallback.
# - If implementing advanced neighborhood evaluation (like multiple steps or iterations), plan for batching and memory management accordingly.

# Additional notes for implementation:
# - The regularizers involve neighborhood evaluations that can be computationally intensive.
# - Design functions to approximate loss and gradient in a neighborhood using efficient autograd operations.
# - Schedule hyperparameters (\(\rho\), \(\lambda\), learning rate) according to the config or schedule.
# - For landscape visualization, prepare utility functions compatible with matplotlib or PyHessian.

# No extra third-party packages are explicitly required as per the plan, but optional packages for Hessian computations or visualization may be included if needed.

# End of requirements.txt

## trainer.py

# Logic Analysis for `trainer.py`

This file is responsible for implementing the `Trainer` class that manages the training process for each incremental learning phase, including:

- Data loading and batching.
- Regularizer computation (C-Flat regularization involving neighborhood evaluations).
- Gradient-based model updates.
- Maintaining the incremental learning protocol.
- Evaluation, visualization, and logging.

Below is a detailed, component-wise breakdown of the logical implementation and flow.

---

## 1. **Class `Trainer` Initialization**

### Inputs:
- `model`: an instance of the model class (`model.py`), which supports perturbation, forward pass, save/load.
- `dataset_loader`: an instance of `DatasetLoader`, provides access to data loaders per task phase.
- `config`: configuration dictionary containing hyperparameters (`learning_rate`, `batch_size`, `epochs`, `rho`, `lambda`, schedule info, etc.).
- `device`: CPU or GPU device for computation.

### Responsibilities:
- Store references.
- Initialize optimizer (e.g., SGD with parameters per config).
- Initialize learning rate scheduler.
- Initialize regularization hyperparameters (`rho`, `lambda`) — potentially schedulable.
- Set up structures for logging metrics (accuracy, landscape visualization).
- Prepare for neighborhood evaluation scheduling (per epoch or per batch).

---

## 2. **Main Training Loop (`train_phase`)**

For each task phase \( T \):

### a. **Data Preparation**
- Retrieve data loader for current phase: `dataset_loader.get_task_dataloader(task_index)`.
- (Optional) Prepare exemplars/memory buffer if rehearsal is incorporated.
- Initialize or reset model parameters (`theta^T`).

### b. **Epoch Loop**
- For each epoch in `[1, epochs]`:
  - Adjust learning rate via scheduler.
  - (If scheduled) adjust \(\rho\) and \(\lambda\) periodically.
  - For each minibatch:
    - Perform forward pass.
    - Compute standard cross-entropy loss (`task_loss`).
    - **Compute Regularizers**:
      
      **i. Neighborhood Evaluation Decision**
      - Check if epoch or batch aligns with evaluation scheduling (`neighborhood_eval_per_epoch` or per batch).
      
      **ii. Zeroth-order regularizer \( R_\rho^0 \)**
      - Perturb model parameters:
        \[
        \theta' = \theta + \rho \cdot \frac{\nabla \ell (\theta)}{\|\nabla \ell (\theta)\|_2 + \epsilon}
        \]
        - Compute the loss at \(\theta'\): \(\ell(\theta')\).
        - Compute the loss difference: \(\ell(\theta') - \ell(\theta)\).
      - Store this as \( R_\rho^0 \).

      **iii. First-order regularizer \( R_\rho^1 \)**
      - Approximate the gradient norm at the perturbed point:
        - Use autograd to compute \(\nabla \ell (\theta + \rho \cdot \frac{\nabla \ell (\theta)}{\|\nabla \ell (\theta)\|})\).
        - Calculate the gradient norm \(\| \nabla \ell (\theta + \rho \cdot \frac{\nabla \ell (\theta)}{\|\nabla \ell (\theta)\|}) \|_2 \).
      - Regularization term:
        \[
        R_\rho^1(\theta) = \rho \cdot \max_{ \theta' \in B(\theta,\rho)} \| \nabla \ell (\theta') \|_2
        \]
        approximated by the current evaluation.

    **iv. Total Regularization Loss**
    - Combine:
      \[
      \ell^{reg} = \ell_{task} + R_\rho^0 + \lambda R_\rho^1
      \]
      
    **Note:** Regularizations are computed via approximate neighborhood evaluations using differentiable autograd operations, Hessian-vector products, or finite differences as per the formulae. Use the approximations:
    - \(\theta' = \theta + \rho \cdot \frac{\nabla \ell (\theta)}{\|\nabla \ell (\theta)\|_2 + \epsilon}\).
    - For \( R_\rho^1 \), approximate \(\nabla \| \nabla \ell (\theta) \|_2 \) via Hessian-vector product:
      \[
      \nabla R_\rho^{1} \approx \rho \cdot \nabla^2 \ell (\theta) \cdot \frac{\nabla \ell (\theta)}{\|\nabla \ell (\theta)\|_2}
      \]
      
    **v. Gradient Step**
    - Backpropagate `\(\ell^{reg}\)` and update model parameters with optimizer.
    - Optionally apply projection/constraints to ensure \(\theta\) remains within a neighborhood of \(\theta^T\).

### c. **Post Epoch Updates**
- Update or decay hyperparameters \(\rho_t\), \(\lambda_t\).
- Save model checkpoints.
- Perform periodic evaluation and landscape visualization:
  - Compute Hessian eigenvalues via autograd Hessian-vector methods.
  - Visualize loss landscape using PyHessian or custom 2D plots by parameter interpolation.
- Log metrics: accuracy, loss, Hessian metrics.

---

## 3. **Neighborhood Regularizer Computation Details**

### Zeroth-order Regularizer (\( R_\rho^0 \))
- For each batch or evaluation:
  - Compute gradient \(\nabla \ell (\theta)\).
  - Generate perturbed parameters:
    \[
    \theta' = \theta + \rho \cdot \frac{\nabla \ell (\theta)}{\|\nabla \ell (\theta)\|_2 + \epsilon}
    \]
  - Forward at \(\theta'\) to compute \(\ell(\theta')\).
  - Difference: \(\ell(\theta') - \ell(\theta)\).

### First-order Regularizer (\( R_\rho^{1} \))
- Approximate \(\| \nabla \ell (\theta + \rho \cdot \frac{\nabla \ell (\theta)}{\|\nabla \ell (\theta)\|}) \|_2 \) using Hessian-vector products.
- Use `torch.autograd.functional.hessian` or `autograd.grad` with `create_graph=True` to get Hessian-vector product.
- Compute \(\nabla \| \nabla \ell (\theta) \|_2 \approx \text{Hessian-vector} \times \text{gradient}\).

### Approximation notes:
- To reduce computational complexity, only evaluate neighborhood regularizers periodically (per epoch or per certain batch count).
- Use efficient Hessian-vector product routines (`torch.autograd` methods).

---

## 4. **Regularization Regularization Schedule**
- \(\rho\) and \(\lambda\) can be scheduled via decay (e.g., exponential decay or inverse square root schedule as per the configuration).
- During training, update these hyperparameters as per schedule:
  \[
  \rho_t = \text{schedule}(\text{initial } \rho, t)
  \]
  \[
  \lambda_t = \text{schedule}(\text{initial } \lambda, t)
  \]
  
### Implementation:
- Use functions from `utils.py` for scheduling.
- Likely exponential decay after each epoch or based on epoch index.

---

## 5. **Evaluation and Logging**
- After each task:
  - Run evaluation on all seen classes.
  - Compute accuracy, forgetting.
  - Save model checkpoints.
  - Visualize landscapes: flatness measures, loss surface plots.
- Keep logs in the specified directory, with periodic snapshots.

---

## 6. **Additional Considerations**
- For models supporting expansion modules (e.g., in MEMO), ensure only relevant parts are trained or regularized as per the approach.
- Rehearsal of exemplars: load previous class samples for incremental phases if applicable.
- Use GPU acceleration; ensure tensors are on correct device.
- Implement additional debugging logs for regularizer magnitudes, eigenvalues, and gradient norms.

---

# Summary of the `trainer.py` logical flow:

1. Initialize trainer with model, data, and hyperparameters.
2. For each incremental phase:
   - Load data, set model state.
   - For each epoch:
     - Adjust hyperparameters if scheduled.
     - For each minibatch:
       - Forward pass.
       - Compute \(\nabla \ell (\theta)\).
       - Evaluate neighborhood regularizers:
         - Perturb parameters for zeroth-order.
         - Compute neighborhood gradient norm for first-order.
       - Calculate total loss and update parameters.
     - Periodically evaluate and visualize.
3. Post-training:
   - Save model.
   - Run full evaluation.
   - Store landscape metrics for analysis.

---

This detailed logic analysis provides a comprehensive guide to implement the training process supporting the C-Flat regularization in a modular, efficient, and reproducible manner.

## utils.py

# utils.py - Logic Analysis

This utility module supports functions for hyperparameter scheduling, loss landscape analysis (Hessian eigenvalues, traces), regularization computation, and visualization, to facilitate implementing the C-Flat continual learning approach. Below is a detailed, structured breakdown of functions, data flows, and core logic that should be incorporated into utils.py according to the provided plan, paper, and configuration.

---

## 1. Hyperparameter Scheduling Functions

### a. Learning Rate Scheduler

**Purpose:**  
Adjust the learning rate per epoch during training, following the specified schedule (e.g., exponential decay).

**Inputs:**  
- `initial_lr`: float, initial learning rate (from config)  
- `current_epoch`: int, current epoch in training  
- `schedule_config`: dict, containing `'decay_type'`, `'decay_rate'`, `'milestones'` (optional)  

**Logic:**  
- If `decay_type` is `'exponential'`, compute:  
  \[
  lr = initial\_lr \times decay\_rate^{current\_epoch / total\_epochs}
  \]  
  or a simpler exponential decay:  
  \[
  lr = initial\_lr \times decay\_rate^{current\_epoch}
  \]
- If `schedule_config` specifies milestones (multi-step), reduce LR at each milestone:  
  \[
  \text{for each milestone } m: \text{if } current\_epoch \ge m: \text{lr} \leftarrow \text{lr} \times decay\_factor
  \]
- Implement a function:  
```python
def schedule_learning_rate(initial_lr, current_epoch, schedule_config):
```

**Outputs:**  
- `lr`: float, scheduled learning rate for current epoch

---

### b. Neighborhood Radius and Lambda Scheduler

**Purpose:**  
Optionally decay or schedule hyperparameters \(\rho\) and \(\lambda\) during training, following paper's proposed decay schemes.

**Inputs:**  
- `initial_rho`: float  
- `initial_lambda`: float  
- `current_epoch`: int  
- `total_epochs`: int  
- `schedule_params`: dict (e.g., linear, exponential decay schemes)  

**Logic:**  
- For simplicity and according to the paper, set schemes such as:  
  \[
  \rho_i = \rho_{-} + \frac{\rho_{+} - \rho_{-}}{\eta_{+} - \eta_{-}} (\eta_i - \eta_{-})
  \]
  where \(\eta_i\) decays with epochs, e.g., \(\eta_i = \bar{\eta} / \sqrt{i}\).

- Similarly for \(\lambda\) if scheduling.

**Implement a function:**  
```python
def schedule_hyperparameters(initial_rho, initial_lambda, epoch, total_epochs, schedule_type='decay', params=None):
```

**Outputs:**  
- `rho`, `lambda`: floats for current epoch

---

## 2. Hessian Eigenvalue Computation via Hessian-Vector Product

### **a. Hessian-Vector Product (HvP)**

**Purpose:**  
Efficiently estimate the top eigenvalues of the Hessian matrix (landscape curvature), crucial for landscape analysis and regularizer bounds.

**Inputs:**  
- `model`: nn.Module (PyTorch model)  
- `loss_fn`: callable, computes loss given model output and labels  
- `inputs`: tensors, input batch  
- `labels`: tensors, labels  
- `vector`: torch tensor, the vector for Hessian-vector product  
- `damping`: float, optional damping parameter for stabilization (usually small, e.g., 1e-5)

**Logic:**  
- Compute the gradient of loss w.r.t. parameters: `grad`
- Use `torch.autograd.grad` with `create_graph=True`, `allow_unused=True` to get `Hv`  
- Approximate computation:  
```python
grad_grad = torch.autograd.grad(grad_outputs=grad, inputs=model.parameters(), grad_outputs=vector, retain_graph=True, allow_unused=True)
```

- **Implementation note:**  
  Wrap the above in a function:  
```python
def hessian_vector_product(model, loss_fn, inputs, labels, vector, damping=1e-5):
```  
- Sum over all model parameters, handle parameter shapes patiently.

**Output:**  
- `Hv` tensor matching model parameter shapes, representing the Hessian-vector product.

---

### **b. Top Eigenvalue Estimation (Power iteration)**

**Purpose:**  
Estimate the maximal eigenvalue (spectral norm) of the Hessian via power iteration using HvP.

**Inputs:**  
- `model`, `loss_fn`, `inputs`, `labels` (as above)  
- `num_iterations`: int, e.g., 20 for convergence stability  
- `damping`: float  

**Logic:**  
- Initialize a random vector with the same size as total parameters: `vector`  
- For each iteration:  
  - Compute `Hv` = `hessian_vector_product(model, loss_fn, inputs, labels, vector, damping)`  
  - Normalize: `vector = Hv / ||Hv||_2`  
- After `num_iterations`, compute Rayleigh quotient:  
  \[
  \lambda_{\max} \approx \frac{\langle v, Hv \rangle}{\langle v, v \rangle}
  \]
  
**Implementation:**  
```python
def estimate_hessian_eigenvalue(model, loss_fn, inputs, labels, num_iterations=20):
```

**Outputs:**  
- `lambda_max`: float, estimated largest eigenvalue

---

## 3. Regularizer Computation

### a. Zeroth-order Sharpness \( R_\rho^{0} \)

**Logic:**  
- Given `model`, compute gradient `g` w.r.t. current parameters:  
```python
g = torch.autograd.grad(loss, params, create_graph=True)
```
- Perturb parameters along `g`:  
```python
direction = rho * g / (torch.norm(g) + epsilon)
perturbed_params = [p + d for p, d in zip(params, direction)]
```
- Forward pass at `perturbed_params`, compute `loss_perturbed`.  
- Compute sharpness: difference `loss_perturbed - loss`.

**Implementation tip:**  
- Wrap neighborhood evaluation in a function:  
```python
def compute_zeroth_order_sharpness(model, loss_fn, data, labels, rho):
```

### b. First-order Flatness \( R_\rho^{1} \)

**Logic:**  
- Approximate gradient norm in neighborhood via:  
```python
g_neighbor = torch.autograd.grad(loss_at_perturbed, params, create_graph=True)
grad_norm = torch.norm(torch.cat([g.view(-1) for g in g_neighbor]))
```
- To avoid Hessian, approximate Hessian-vector product as in previous section.

- Alternatively, directly compute gradient norm at the perturbed point.

**Implementation:**  
Returns the maximum gradient norm in neighborhood, or an approximation thereof.

```python
def compute_first_order_flatness(model, loss_fn, data, labels, rho):
```

---

## 4. Visualization Helpers

### a. Loss Landscape Visualization

**Purpose:**  
Plot 2D loss surface along selected eigenvectors or random directions.

**Logic:**  
- Select two directions (eigenvectors or random) in parameter space: `v1`, `v2`  
- Generate a grid in 2D space around current weights:  
```python
for alpha in linspace(-depth, depth, grid_size):
    for beta in linspace(-depth, depth, grid_size):
        perturbed_params = base_params + alpha*v1 + beta*v2
        evaluate loss at perturbed_params
```
- Plot contours or surface.

**Implementation:**  
```python
def plot_loss_landscape(model, loss_fn, directions, center_params, grid_size=50, depth=1.0):
```

### b. Hessian Eigenvalues and Trace Plotting

- Use `matplotlib` to plot eigenvalues distribution, trace over epochs for landscape flatness tracking.

---

## 5. Miscellaneous Utilities

### a. Default Parameter Initialization & Noise

- Precise perturbation schemes consistent with the paper's formulas:
  \[
  \theta' = \theta + \rho \frac{\nabla \ell(\theta)}{\|\nabla \ell(\theta)\|_2}
  \]
- Ensure epsilon for numerical stability.

### b. Random Seed Management

- Centralize setting seed for reproducibility across functions.

---

## 6. Summary

- As per configuration and paper:
  - Handle hyperparameters \(\rho, \lambda, LR\) with scheduling functions.
  - Compute landscape regularizers—zeroth-order via loss difference, first-order via gradient norms—using parameter perturbation in neighborhood.
  - Estimate Hessian spectral properties via power iteration with Hessian-vector products.
  - Visualize loss landscapes along eigenvectors/directions.
- All functions should accept model, data, and hyperparameters as inputs and return scalars or visualizations, supporting integration into main training/evaluation loop.

---

This detailed logic analysis provides a comprehensive blueprint for implementing functions in utils.py, ensuring the practical realization of the C-Flat method, landscape analysis, hyperparameter scheduling, and visualization, fully aligned with the paper and configuration instructions.

