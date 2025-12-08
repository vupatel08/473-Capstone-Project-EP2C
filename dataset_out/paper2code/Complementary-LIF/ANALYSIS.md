# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

**Logic Analysis for dataset_loader.py — Class DatasetLoader**

---

### Purpose and Responsibilities:
The `DatasetLoader` class is responsible for:
- Loading selected datasets (e.g., CIFAR10, CIFAR100, TinyImageNet, DVS-Gesture, DVS-CIFAR10) from disk.
- Preprocessing data according to dataset specifications (normalization, augmentation).
- Encoding static images or events into sequences suitable for spiking neural network training (e.g., direct pixel-to-spike encoding, event-based frame encoding).
- Returning dataset objects or dataloaders prepared for training and testing, compatible with PyTorch's DataLoader.

---

### Inputs and Configuration Dependencies:
- **Configuration dict**: Receives dataset details:
  - `dataset.name` (str): e.g., 'CIFAR10', 'DVS-Gesture', etc.
  - `dataset.dataset_path` (str): path to dataset location.
  - `dataset.batch_size` (int), `dataset.num_workers` (int): for DataLoader.
  - `dataset.normalization_mean`, `dataset.normalization_std`: for normalization.
  - `dataset.encoding_scheme`: specifies how to convert raw data into spike sequences.

- **Hyperparameters**:
  - `training.seed`: for reproducibility.
- **Dataset-specific details**:
  - Resize, cropping, augmentation strategies for static datasets.
  - Event-to-frame encoding parameters for neuromorphic datasets.

---

### Dataset Loading Strategy:
1. **Identify dataset type**:
   - Read `dataset.name` to determine which dataset to load.
   - Use appropriate `torchvision.datasets` or custom loaders for neuromorphic data.

2. **Dataset preprocessing pipeline**:
   - **For static datasets (CIFAR-10, CIFAR-100, TinyImageNet)**:
     - Normalize pixel values with mean/std.
     - Apply data augmentation:
       - Random crop + padding.
       - Random horizontal flip.
       - AutoAugment if specified.
       - Cutout if necessary.
     - Convert images into sequences of spikes:
       - For direct spike encoding:
         - Normalize pixel values to [0,1].
         - Use a threshold (e.g., threshold = 1) to generate spikes:
           - For each pixel value, generate a binary spike per timestep based on rate or threshold crossing.
         - Alternatively, replicate the static image across `T` timesteps, using rate-based encoding or direct thresholding.
   - **For neuromorphic datasets (DVS-Gesture, DVS-CIFAR10)**:
     - Load event streams.
     - Convert event streams into frame sequences:
       - Use event accumulation within temporal windows.
       - Normalize as needed.
     - Convert to spike trains:
       - These are usually already event-based; may require minimal processing.

3. **Encoding scheme**:
   - Use a flag or string in `dataset.encoding_scheme` to select encoding:
     - For static images: `"direct_spike_encoding"` means converting pixel intensities into spike trains—implemented via rate coding or threshold crossing.
     - For event-based data: process raw event streams into frames.
   - Implementation detail:
     - For rate coding:
       - Generate Poisson spike trains with rate proportional to pixel intensity.
     - For direct threshold-based:
       - Binarize pixel intensities based on threshold.
     - For neuromorphic:
       - Use existing event data formats.

---

### Dataset Initialization:
- Load datasets using torchvision for static images:
  - Compose transforms:
    - `transforms.RandomCrop`, `transforms.RandomHorizontalFlip`, normalization.
  - Instantiate dataset objects.

- For neuromorphic datasets:
  - Use existing datasets (or write custom loader if necessary):
    - Load event streams from files.
    - Convert events into tensors (e.g., 2D frames or sequences).
    - Normalize preprocessed data appropriately.

---

### Data Encoding Implementation:
- After dataset loading:
  - Implement a post-processing step:
    1. For each image or event sequence:
       - Convert to pixel tensor (for static datasets).
       - Normalize pixel values.
       - Encode into spike train over `T` timesteps:
         - For rate coding:
           - For each pixel:
             - Assign spike probability proportional to pixel value.
             - Generate binarized spike sequence with Bernoulli sampling.
         - For direct threshold:
           - For each pixel, create a binary spike vector:
             - Spike at each timestep if pixel value exceeds a threshold.
- Store these sequences as the dataset samples.

---

### Dataset Output:
- Return:
  - `(train_dataset, test_dataset)` objects for use with DataLoader:
    - Each dataset should yield a tensor of shape `(T, C, H, W)` or similar, representing the spiking input sequence.
- Use:
  - `torch.utils.data.DataLoader` with `batch_size` and `num_workers` as specified.
  - Set `shuffle=True` for training, `False` for testing.

---

### Additional Considerations:
- **Reproducibility**:
  - Fix seeds (`training.seed`) early during dataset and data augmentation setup.
- **Dataset splits**:
  - Ensure training/test splits match dataset (standard CIFAR10 splits).
- **Data augmentation parameters**:
  - Follow specified augmentation protocol.
- **Encoding parameters**:
  - Hyperparameters like spike rate, firing threshold, and number of timesteps (`T`) influence encoding.
  - Keep consistent with training configuration, especially `timesteps`.

---

### Summary:
- **Load datasets** based on name.
- **Apply dataset-specific preprocessing** (normalization, augmentation).
- **Encode static images into spike sequences** suitable for SNN training:
  - Use rate-based or threshold crossing encoding.
- **Handle neuromorphic datasets** by converting event streams into frame sequences.
- **Return PyTorch datasets** compatible with DataLoader.
- **Ensure reproducibility** via seed control.

---

This detailed logic guides the implementation of the `DatasetLoader` class, ensuring dataset integrity, accurate encoding, and consistency with the experimental setup described in the paper.

## evaluation.py

{
  "evaluation.py": [
    {
      "purpose": "The evaluation.py module is responsible for assessing the trained SNN model's performance through inference, accuracy measurement, and additional analyses such as autocorrelation and energy estimation. It also facilitates model conversion for inference, aligning with the methodologies in the paper.",
      "core functionalities": [
        "Loading the trained model and preparing it for evaluation, including optional conversion to LIF for inference if specified.",
        "Performing inference on validation/test datasets to compute accuracy, including multiple runs if needed for statistical stability.",
        "Calculating the autocorrelation of neuron membrane potentials or complementary potential during inference to analyze temporal dynamics and gradient representations.",
        "Estimating energy consumption based on spike activity logs and synaptic operations (ACs) and weight * activation (MACs), following formulas in the paper’s appendix.",
        "Predicting outputs for the test dataset and computing accuracy metrics.",
        "Optionally, converting the trained CLIF model into a standard LIF model with fixed bias to compare inference performance, as per the model conversion protocol in Table 8.",
        "Logging and saving evaluation results, metrics, and energy estimates for analysis and reproducibility."
      ],
      "inputs": [
        "The trained model, which could be a CLIF or LIF network, in its current trained state.",
        "The dataset (test split), i.e., CIFAR10/CIFAR100 or neuromorphic datasets, prepared with the same encoding scheme used during training.",
        "Evaluation configurations, e.g., whether to convert to LIF, whether to compute energy, whether to evaluate autocorrelation.",
        "Optional: Spike activity logs or recording mechanisms to gather neuron spikes during inference (for energy estimation)."
      ],
      "outputs": [
        "Evaluation metrics such as overall accuracy, per-class accuracy (if needed).",
        "Autocorrelation metrics of membrane potential or complementary potential over temporal lags.",
        "Estimated energy consumption figures based on spike rates and synaptic operations.",
        "Converted models for inference (if model conversion enabled)."
      ],
      "steps and logic flow": [
        "Initialize and load the trained model; check if a conversion to LIF is required based on config. If yes, perform conversion by replacing neuron modules or adjusting reset parameters, as specified in Table 8.",
        "Set evaluation mode (`model.eval()` in PyTorch) to disable dropout, batch normalization updates, etc.",
        "Prepare the dataset loader for inference, ensuring the input encoding matches training (direct spike encoding or event streams).",
        "For each sample in the test dataset or batch:",
        "  - Forward pass through the network, collecting output logits/predictions.",
        "  - During the forward pass, record neuron membrane potentials (`u[t]`), complementary potentials (`m[t]`) (if accessible), and spike logs necessary for energy calculation.",
        "Calculate the predicted label by selecting the class with the highest output (for classification tasks).",
        "Compare predictions against ground truth labels to update accuracy counts.",
        "Optionally, compute and store autocorrelation of the membrane potentials or complementary potentials to analyze temporal activity patterns.",
        "Once all samples are processed, compute overall accuracy, per-class accuracy if needed, and other statistics.",
        "Calculate average neuron firing rates over the test set or batch for energy estimation, utilizing stored spike logs.",
        "Estimate energy consumption based on spike activity and synaptic operations as per the formulas provided, e.g., ACs and MACs calculations, considering sparsity.",
        "If model conversion enabled, create a version of the model with fixed thresholds (or biases) according to Table 8. Run inference again to evaluate any loss/gain in performance.",
        "Log and return all evaluation metrics: accuracy, autocorrelation, energy estimates, and conversion results.",
        "Save these metrics and logs to disk, if logging is enabled, for reproducibility and further analysis."
      ],
      "considerations": [
        "Ensure the autocorrelation calculation is performed over the correct variable (`u[t]` or `m[t]`) across the temporal dimension, utilizing numpy or torch autocorrelation functions.",
        "In energy estimation, Spike rates are derived from spike logs collected during inference, ensuring synchronization between activity logs and energy calculations.",
        "Conversion routines should accurately replace neuron modules while preserving trained weights and states, following the described procedure, e.g., fixing biases, resetting membrane potentials.",
        "The code should be compatible with the model class structures in model.py, especially the forward methods, possibly leveraging hooks or direct attribute access.",
        "Debugging traceability is important: verify that stored variables (`u[t]`, `m[t]`, spikes) match the actual computations during inference.",
        "Logging hyperparameters, results, and timing for reproducibility, aligned with the config.yaml settings."
      ],
      "error handling": [
        "Validate model state before evaluation; ensure proper device transfer (`cpu()` or `cuda()`) and state loading.",
        "Handle missing spike logs gracefully, i.e., warnings if energy or autocorrelation cannot be computed due to lack of data.",
        "Check for dimensional consistency when doing autocorrelation and energy calculations, especially across batches and time steps.",
        "Ensure that conversion routines do not inadvertently alter accuracy logs or parameter states during inference."
      ],
      "Conclusion": "The evaluation.py module must be meticulously designed to assess the performance of the CLIF neuron-based SNNs accurately, replicating the experimental analysis, and providing detailed metrics for comparison. It should be tightly integrated with the model architecture, data loading, and logging utilities to facilitate comprehensive, reproducible results following the methodology outlined in the paper."
    }
  ]
}

## main.py

# Logic Analysis for `main.py`

This file serves as the orchestrator of the entire experiment pipeline, responsible for initializing all components, managing the workflow, and executing the training, validation, testing, and model conversion processes. The logic flow must closely follow the experimental setup described in the paper and align with the provided configuration parameters.

---

### 1. **Import Necessary Modules and Libraries**

- **Standard Libraries**: `os`, `sys`, `random`
- **Deep Learning Framework**: `torch`, `torch.nn`, `torch.optim`
- **Configuration**: `yaml` to parse `config.yaml`
- **Custom Modules**:
  - `dataset_loader.py`: `DatasetLoader` (handles dataset loading and preprocessing)
  - `model.py`: `build_model()` (constructs the SNN architecture with CLIF neurons)
  - `trainer.py`: `Trainer` (performs training with BPTT and surrogate gradients)
  - `evaluation.py`: `Evaluation` (handles validation, testing, energy estimation, model conversion)
  - `utils.py`: shared utility functions (e.g., seed setting, logging)

---

### 2. **Load Configuration**

- Read `config.yaml` using `yaml.safe_load()`.
- Extract key parameters:
  - Dataset parameters: dataset name, path, batch size, normalization, encoding scheme
  - Training hyperparameters: optimizer type, learning rate, epochs, weight decay, scheduler settings, seed
  - Model parameters: architecture name, number of timesteps, input channels, number of classes
  - Neuron parameters: neuron type (CLIF), threshold, tau
  - Evaluation flags: whether to perform energy estimation, model conversion
  - Logging options: save directory, intervals, verbosity

---

### 3. **Set Random Seeds for Reproducibility**

- Call seed setting utility with `seed=2022` or as per config.
- Set seeds for `random`, `numpy`, and `torch` (including `torch.cuda` if GPU is used).
- Set deterministic options for CUDA/cuDNN if using GPU.

---

### 4. **Initialize Dataset Loader and Data Preprocessing**

- Instantiate `DatasetLoader` with dataset path, batch size, normalization parameters, encoding scheme.
- Call `load_data()`:
  - For static datasets (e.g., CIFAR10):
    - Load train and test datasets.
    - Apply normalization.
    - Apply augmentations (AutoAugment, CutMix) if specified.
    - Convert images into spike sequences:
      - Use direct pixel-level spike encoding.
      - Generate sequential data with `timesteps=6` (per config).
  - For neuromorphic datasets:
    - Load event streams directly (preprocessed).
    - Encode into spike trains as available.
- Create respective `DataLoader` objects for training and testing.

---

### 5. **Initialize the Model Architecture**

- Use `build_model()` or equivalent function:
  - Pass model architecture (`ResNet18`), input channels=3, number of classes=10, and `timesteps=6`.
  - The model should instantiate a `SpikingResNet` or similar, incorporating CLIF neurons.
  - Set the neuron parameters: threshold=1.0, tau=1.5.
- Move model to device (`cuda` if available, else CPU).

---

### 6. **Setup the Optimizer and Learning Schedule**

- Instantiate optimizer (e.g., `torch.optim.SGD`) with model parameters, learning rate=0.01, momentum=0.9, weight_decay=5e-5.
- If a scheduler is specified (e.g., step decay), initialize it accordingly:
  - Use `StepLR` with `step_size=50`, `gamma=0.1`.
- For reproducibility and consistent training, set optimizer hyperparameters as per config.

---

### 7. **Initialize Trainer**

- Instantiate `Trainer` object with:
  - `model`: the neural network
  - `optimizer`: as set
  - `train_loader`: DataLoader for training data
  - `device`: device used
  - Hyperparameters like epochs, `timesteps=6`, neuron parameters
- The trainer should implement:
  - Forward pass over multiple timesteps
  - BPTT with surrogate gradients
  - Recursive gradient computations to prevent vanishing (per paper and design)
  - Loss computation (e.g., cross-entropy) at the final timestep or sum

---

### 8. **Training Loop**

- For each epoch:
  - Call `trainer.train_one_epoch()`
  - Log training loss periodically (every `log_interval=10`)
  - Step learning rate scheduler if used
  - Save model checkpoint if performance improves or at regular intervals
  - Optionally, evaluate on validation set (if validation data available), or use test set periodically for intermediate validation

---

### 9. **Post-Training Evaluation**

- Instantiate `Evaluation` object with:
  - Trained model
  - Test dataset
  - Device
  - Flags for energy estimation and model conversion
- Perform `evaluate()`:
  - Compute accuracy over test set
  - (Optional) compute energy estimates (number of spikes, synaptic operations)
- If conversion flag is true:
  - Call `convert_to_LIF()` or equivalent:
    - For inference, replace CLIF neurons with equivalent LIF parameters (biases, thresholds)
    - Use the method as per Table 8; likely involves applying fixed biases, possibly with hard resets
- Run inference with the converted model and record accuracy for comparison.

---

### 10. **Logging & Results**

- Throughout the workflow:
  - Log key metrics: training loss, test accuracy, energy consumption
  - Save model checkpoints at appropriate epochs or based on best validation
  - Save final models in `save_dir`
- Generate plots:
  - Loss curves
  - Accuracy vs epochs
  - Accuracy vs number of timesteps
  - Autocorrelation or gradient decay plots (if applicable)
  - Energy analysis charts (fire rate, SOP)

---

### 11. **Cleanup & Exit**

- Finalize logs.
- Save configuration and results for reproducibility.
- Clear GPU cache if applicable.
- Exit gracefully with status code 0.

---

### Summary:

- **Initialization**:
  - Load configs, set seeds
  - Prepare dataset with encoding
  - Instantiate SNN architecture with CLIF neurons
  - Set optimizer, scheduler

- **Training**:
  - Run epochs with BPTT, surrogate gradients, recursive gradient calculation
  - Log and checkpoint periodically

- **Evaluation**:
  - Measure accuracy, energy
  - Convert for inference (if enabled)
  - Final performance logs

- **Output**:
  - Models, plots, metrics, energy estimates

This logical flow ensures high fidelity in reproducing the experiments in the paper, following the precise experimental setup, hyperparameters, and evaluation criteria.

## model.py

**Logic Analysis for `model.py`: Construction of Spiking Neural Network Architecture Incorporating CLIF Neurons**

---

### **1. Overview & Purpose**

The `model.py` module defines the neural network architecture, specifically constructing a deep SNN (e.g., ResNet-18 or VGG) that employs CLIF neurons as its fundamental computational units (layers). The architecture must be designed to accept time-sequence spike inputs, process them across multiple timesteps, and output predictions (e.g., class logits).

### **2. Key Elements & Requirements**

- **Input Shape Handling:**
  - Input tensors correspond to batches of data with shape `[batch_size, channels, height, width]`.
  - Because the emphasis is on temporal processing, the input during training and inference will be expanded over time dimension, e.g., `[batch_size, time_steps, channels, height, width]`.
  - This temporal dimension can be handled either inside the `forward()` method (by looping over timesteps) or by processing the entire sequence at once (via tensor reshaping).

- **Architecture Components:**
  - Built using `torch.nn.Module`, consistent with PyTorch standards.
  - For `ResNet-18`:  
    - Use `torchvision.models.resnet.ResNet` or a custom implementation.
    - Replace all standard residual blocks’ neuron activations with CLIF neuron layers.
  - For `VGG` or other backbones:
    - Adapt convolutional layers, batch normalization, pooling, and classifier parts.
    - Replace activation functions and neurons with CLIF layers.

- **Neuron Layers:**
  - The CLIF neuron includes:
    - State variables: membrane potential `u[t]`, complementary potential `m[t]`.
    - Dynamics update per timestep:
      ```python
      u[t] = (1 - 1/τ) * (u[t-1] - V_th * s[t-1]) + W * s[t]
      m[t] = m[t-1] * sigmoid((1/τ) * u[t]) + s[t]
      ```
    - Spike generation: `s[t] = Θ(u[t] - V_th)`
    - Reset condition: subtract `(V_th + sigmoid(m[t]))` when neuron fires.
  - These layers will be embedded in residual blocks or convolutional blocks.

- **Layer Integration:**
  - Each convolutional or linear layer will be followed by a CLIF neuron activation.
  - For residual blocks:
    - Implement custom residual blocks where after convolution, CLIF neurons process the output.
    - Maintain internal state variables per sample (batch-wise).
  - For the classifier head:
    - Usually fully connected; the neurons here can be considered as static functions or progressive spike accumulators, depending on the implementation.
    - At the final layer, output logits for all classes.

- **Implementation Details & Consistency:**
  - All layers should be compatible with batched input and recurrent for multiple time steps.
  - During training:
    - Sequence over `time_steps` need to be propagated fully.
    - States (`u`, `m`) need to be stored and updated per sample, per layer, per timestep.
  - During inference:
    - Same forward pass, but potentially with fixed states or no gradient calculations.

- **Parameter Settings:**
  - Threshold `V_th`: from `config.yaml`, default 1.0.
  - Timestep: from `config.yaml`, e.g., 6 for CIFAR.
  - Time constant `τ`: from `config.yaml`, e.g., 1.5.
  - Input channels: 3 (RGB images).
  - Number of classes: 10 (for CIFAR10).
  - Interaction with `neuron.py`:
    - The CLIF neuron class will be instantiated with the model’s layers.
    - Layers will invoke forward passes on CLIF neurons, with appropriate state passing.

### **3. Construction Logic & Strategy**

- **Step 1: Create a class `SpikingResNet` or `VGG` inheriting from `nn.Module`.**
  - **Init method:**
    - Initialize standard layers, e.g., Conv2d, BatchNorm2d, Pool, etc.
    - Replace activation layers with CLIF neuron instances; for residual blocks, define custom blocks with CLIF units.
    - Define the classifier block at the end.
  - **Forward method:**
    - Accept input `[batch_size, time_steps, channels, height, width]`.
    - For each timestep `t` in `[0, T-1]`:
      - Extract the `t`-th frame: shape `[batch_size, channels, height, width]`.
      - Pass through initial conv and residual blocks.
      - Each convolutional layer outputs are passed through CLIF neuron activation.
      - Update `u`, `m` states internally (per layer) on each timestep.
    - **States management:**
      - Maintain per-layer state variables for `u` and `m`, reinitialized at the start of each sequence.
      - They must be updated in sequence over the timesteps.
    - **Aggregate outputs:**
      - During the last timestep, or sum over timesteps, generate logits.
      - Optionally, average the logits over timesteps for stability (if per-timestep output is reported).

- **Step 2: State Initialization & Resetting**
  - Implement a method `reset_state()` that initializes all `u`, `m` states (e.g., zeros) at start of each sequence.
  - For batch processing, ignore per-sample state persistence beyond individual sequence passing.

- **Step 3: Integration with CLIF Neuron Module**
  - Instantiate the CLIF neuron layer inside the model.
  - Forward pass in composite layers:
    - Receive the current input (after convolution or pooling).
    - Call CLIF neuron layer with current input and previous state.
    - Obtain output spikes, update internal states.

- **Step 4: Handling Residual Connections**
  - For residual blocks:
    - Forward the main branch input through convolution + CLIF activation.
    - Shortcut bypasses.
    - Element-wise add.
    - Continue processing with CLIF layers.
    - The internal states must be correctly managed to ensure proper gradient flow and state updates.

- **Step 5: Final Layer & Output**
  - After feature extraction:
    - Flatten spatial dims.
    - Fully connected (linear) layer (may be standard, not spiking), or a spiking linear layer with CLIF if desired.
    - Generate logits per class.
    - During training, compute loss; during inference, produce class predictions.

### **4. Additional Considerations**

- **Batch-wise and Layer-wise State Management:**
  - For compactness and clarity, implement states as `torch.nn.Parameter` or buffers if states need to be persistent (for recurrent behavior), or assign in the instance and reset per sequence.
  - Automate state resetting at sequence start.

- **Compatibility with Training & Backprop:**
  - Ensure that the gradient computation leverages the custom gradient equations for the CLIF neuron.
  - The call to neuron modules may involve custom autograd functions, integrated in `neuron.py`.

- **Hyperparameters:**
  - Pass `τ`, `V_th`, and other parameters from configuration.
  - Allow parameter tuning for ablation studies.

---

### **5. Summary of Key Data Flows and Logic**

- **Input handling:**
  - Receive `[batch, time, channels, H, W]`.
  - Loop through time steps, process each frame sequentially.
- **Layer flow per timestep:**
  - For each convolutional layer:
    - Input: feature maps from previous layer (or initial input for first layer).
    - Pass through CLIF neuron:
      - Update `u`, `m` states.
      - Generate spike output.
  - For residuals:
    - Add residual connections maintaining the same state management.
- **State updates:**
  - For each layer, store `u`, `m` at each timestep.
  - Use the equations in the paper to update these states.
- **Output & Loss:**
  - Collect output logits at the last timestep (or average over timesteps).
  - Compute loss during training.
- **During inference:**
  - Use the same forward pass, no gradient calculation, possibly with fixed states or direct input encoding.

---

### **6. Final Validation & Testing Strategy**

- Confirm dynamic behaviors by:
  - Validating state updates via synthetic inputs.
  - Checking spike rates, membrane potential statistics.
- Verify gradient flow through recursive structure.
- Ensure that the model’s output matches the accuracy and loss profiles reported.

---

**End of Logic Analysis.**  
This detailed plan guides the implementation of `model.py` to construct the ResNet or VGG with integrated CLIF neurons, respecting the architecture, equations, and design discussed in the paper.

## neuron.py

**Logic Analysis for neuron.py – Implementing the CLIFNeuron Class**

---

### Overview

The `CLIFNeuron` class encapsulates the dynamics and gradient computations of the Complementary Leaky Integrate-and-Fire (CLIF) neuron model outlined in the paper. The implementation must faithfully reproduce the neuron’s forward and backward behavior as derived from the paper equations, ensuring the proper handling of the membrane potential `u(t)`, the complementary potential `m(t)`, and the spike output `s(t)` for each timestep. Crucially, recursive gradient propagation formulas, which include additional gradient paths to avoid vanishing gradients, are central to the model and must be carefully incorporated via custom autograd functions or explicit backpropagation routines.

---

### Core Components

1. **State Variables**:
   - `u[t]`: Membrane potential at time `t`.
   - `m[t]`: Complementary membrane potential at time `t`.
   - `s[t]`: Binary spike output at time `t`.

2. **Neuron Parameters**:
   - Threshold `V_th` (from config; default = 1.0).
   - Time constant `τ` (from config; default = 1.5). Used to compute `γ = 1 - 1/τ`.
   - Reset bias and other optional parameters (such as initial states).

3. **Forward Dynamics**:
   - **Membrane potential update**:
     \[
     u(t) = \gamma (u(t-1) - V_{th} s(t-1)) + W \cdot s_{prev}
     \]
     - `W \cdot s_{prev}` is the input current or synaptic input at the current timestep.
     - The decay constant `γ = 1 - 1/τ`.
   - **Spike generation**:
     \[
     s(t) = \Theta(u(t) - V_{th})
     \]
     - Implement using a differentiable approximation for backprop, e.g., rectangle surrogate, with derivative \(\frac{1}{\alpha} \mathbf{1}(|u - V_{th}| < \frac{\alpha}{2})\).
   - **Complementary potential update**:
     \[
     m(t) = m(t-1) \odot \sigma \left(\frac{1}{\tau} u(t)\right) + s(t)
     \]
     - Sigmoid acts as decay compensation, non-learnable.
   - **Reset process on spike**:
     \[
     u(t) = u(t) - s(t) \left(V_{th} + \sigma(m(t))\right)
     \]
     - Subtracts scaled threshold and complementary potential.

4. **Implementation Detail**:
   - Use a class inheriting from `torch.autograd.Function` or define a custom backward function that:
     - Implements forward pass per the equations.
     - Implements backward pass according to the recursive gradient equations (see paper and appendix), explicitly calculating the temporal derivatives, including additional gradient paths via `m(t)`.

5. **Gradient Computation**:
   - Follow Eq. (45)–(52).
   - The key is to:
     - Compute the local gradients:
       - \(\frac{\partial s(t)}{\partial u(t)}\) (via surrogate).
       - \(\frac{\partial u(t)}{\partial u(t-1)} = \gamma\) (decay).
       - \(\frac{\partial u(t+1)}{\partial m(t)} \) and related terms.
     - For each timestep:
       - Calculate recursive terms for the gradient of loss with respect to `u(t)` and `m(t)`.
     - Store intermediate terms (`\epsilon`, `\psi`, `\xi`, etc.) during forward for use in backprop.

6. **Implementation Strategy**:
   - The `CLIFNeuron` class should:
     - Initialize with parameters (`threshold`, `τ`).
     - Have an internal method for state update per timestep.
     - Use a custom `autograd.Function` (`CLIFFunction`) that controls forward and backward passes.
     - During the forward:
       - Update `u` and `m` according to equations.
       - Generate `s` spike (with surrogate gradient approximation).
     - During backward:
       - Use the stored forward activations and auxiliary variables.
       - Implement the explicit recursive gradient calculation, including the extra gradient terms from `m`.

7. **State Handling**:
   - Maintain `u`, `m`, `s` as buffers or state tensors.
   - Reset states at the start of each sequence (batch).  
   - During training, accumulate states across time steps within a sequence.

8. **Additional Details**:
   - Handle the special case where `u(t)` is close to `V_th` for surrogate derivative.
   - Clamp or threshold as needed to prevent numerical instability.
   - Ensure differentiability where necessary, including the surrogate for spike function.

---

### Implementation Steps Summary

- **Step 1**: Define `CLIFFunction` inheriting from `torch.autograd.Function`.
  - Implement `forward()`:
    - Compute `u[t]`, `m[t]`, `s[t]` as described.
    - Save variables for backward (e.g., `u[t]`, `m[t]`, `s[t]`, input currents).
  - Implement `backward()`:
    - Retrieve saved variables.
    - Compute surrogate derivatives for `s[t]`.
    - Apply the recursive equations from Appendix to calculate:
      - `∂L/∂u(t)`
      - `∂L/∂m(t)`
    - Use the explicit formulas to compute `grad_input`.
- **Step 2**: Encapsulate in `CLIFNeuron` class:
  - Initialize with parameters (`threshold`, `τ`, etc.).
  - Maintain internal state tensors for `u`, `m`.
  - On each forward call:
    - Pass inputs (from previous layer or external).
    - Call `CLIFFunction.apply()` with current states and input.
    - Update stored states accordingly.
- **Step 3**: Expose interface:
  - `forward(inputs, time_steps)` method:
    - Iterate over timesteps.
    - Call the `CLIFFunction`.
    - Return spike sequence for loss computation.
- **Step 4**: Additional utilities:
  - Surrogate function (rectangle).
  - Initialization routines.
  - Reset functions for states per sequence.

---

### Final Notes

- The implementation must match the equations exactly; particularly, the terms involving recursive derivatives and gradient paths from `m(t)`.
- Care must be taken to avoid numerical issues with the sigmoid and indicator functions.
- Ensure gradients flow properly through the recursive structure, likely requiring explicit control over the backward pass.
- The approach aligns with the detailed derivations in Appendices G–I.

---

This comprehensive logic analysis ensures that the code will implement the CLIF neuron accurately, facilitating correct forward dynamics and recursive gradient calculations, which are foundational for reproducing the experiments and results in the paper.

## trainer.py

# Logic Analysis for `trainer.py`

This document provides a detailed, step-by-step logical flow and design considerations for implementing the core **Trainer** class within `trainer.py` based on the paper, design, and configuration parameters provided. The goal is to ensure the training process faithfully reproduces the methodology, including recursive gradient calculations, surrogate gradient application, and proper data handling, particularly for the CLIF neuron model with BPTT.

---

## 1. Class Purpose & Responsibilities

**`Trainer` Class**:
- Encapsulates the training loop for the SNN with CLIF neurons.
- Handles data input, forward pass (over multiple timesteps).
- Implements the Backpropagation Through Time (BPTT) procedure with surrogate gradients.
- Uses explicit recursive gradient equations to compute accurate temporal gradients, addressing vanishing issues.
- Logs training progress, saves checkpoints, and supports model conversion post-training.
- Incorporates hyperparameters from the configuration file, including learning rates, epochs, surrogate alpha (`α`), and time constant `τ`.

---

## 2. Key Modules & Dependencies

**Inputs & Dependencies**:
- **Model**: An instance of the SNN, e.g., ResNet-18 with CLIF neurons.
- **Dataset Loader**: Provides training data arranged as sequences over `T` timesteps.
- **Optimizer**: `torch.optim.SGD` with specified `lr`, momentum, `weight_decay`.
- **Criterion**: Usually cross-entropy loss, applied after collapse of temporal outputs.
- **Device**: `cuda` or `cpu`, determined at runtime.
- **Hyperparameters**: From `config.yaml`, including `epochs`, `batch_size`, `surrogate_alpha` (`α`), `tau`, etc.

---

## 3. Data Preparation

- Load datasets (`CIFAR10`) via `dataset_loader.py`, yielding batches with shape `(batch_size, channels, height, width)`.
- Apply preprocessing: normalization (mean/std), as per config.
- Encoding:
  - Use **direct_spike_encoding** (or as specified).
  - For static datasets, replicate raw images into sequences of length `T=6` (or as configured).
  - Each input sample becomes a sequence of pixel intensities/values per timestep, forming an input spike train.
- Data loaders should ensure shuffling, appropriate batching, and multithreading (`num_workers`).

---

## 4. Forward Pass Logic

- **Initialization**:
  - Reset the neuron states (`u[t]`, `m[t]`, `s[t]`) for the batch at start of each epoch.
  - Ensure model's internal states are cleared/reset (manual or via built-in reset method).
- **Temporal Simulation Loop**:
  - For each timestep `t` in `1` to `T`:
    - Feed input spikes for the `t`-th timestep.
    - Call model’s `forward()` method:
      - For CLIF, the model should internally update `u[t]`, `m[t]`, generate spikes `s[t]`.
      - Return output spikes or logits for that timestep, or final output after entire sequence (depending on task setup).
  - Collect output spikes or logits over all steps, then produce a combined output (e.g., sum or mean) for loss calculation.

---

## 5. Loss Computation

- Aggregate model outputs:
  - For classification, typically use the output at the last timestep or average over time.
  - For simplicity, assume final timestep output is used for loss.
- Compute cross-entropy loss across classes, comparing against labels.
- If multiple outputs are used (e.g., sum over steps), sum losses accordingly.

---

## 6. Backward Pass & Gradient Computation

### 6.1. Surrogate Gradient Application:
- The surrogate gradient (e.g., rectangle function) is applied during backward pass.
- Implement `torch.autograd.Function` or manual backward functions:
  - For the native `U` and `M` state variables, store their values during forward.
  - During backward, compute gradients via explicit recursive equations (derived from equations in the paper and appendix):
    - **Temporal gradient**: Use equations for `epsilon`, `xi`, `ψ`, etc.
    - For CLIF, incorporate extra terms (e.g., gradient contributions from `m[t]`) as per Appendix G.
    - For gradient `∂L/∂u[t]`, include the sum over future steps weighted by recursive products (`∏ ρ[l][t'-t]` or equivalent).
  - Handle the decay factor `γ` (`1 - 1/τ`) carefully within each recursive step for accurate gradient flow.
  - For each parameter `W`:
    - Calculate `∂L/∂W` as sum over `t` of `∂L/∂u[t] * ∂u[t]/∂W` (which is `s[t-1]` for this network).

### 6.2. Explicit Recursive Gradient Equations:
- Use the explicit recursive relations modeled as:
  - `∂L/∂u[t]` += sum of terms across `t+1` to `T` involving the product of decay factors (`γ` or equivalents).
  - For CLIF, include gradients from complementary potential `m[t]`.
  - Equations adapted from Eqs. (6), (14), (20), (35).

### 6.3. Handling vanishing gradients:
- The recursive equations for `epsilon`, `xi`, and `ψ` explicitly incorporate the additional temporal paths introduced by CLIF, preventing vanishing.
- These are implemented as auxiliary tensors stored and propagated during backward.

---

## 7. Optimization Step

- Zero the optimizer gradients.
- Perform `loss.backward()`, ensuring the custom backward functions (or manual gradient updates) are utilized.
- Call `optimizer.step()` to update parameters.
- Optionally, implement gradient clipping if necessary.

---

## 8. Logging and Checkpointing

- Log current epoch, batch loss, accuracy, possibly gradient norms or energy estimates.
- Save model checkpoints (`torch.save()`) at regular intervals (`log_interval`).
- Save best model based on validation accuracy if applicable.

---

## 9. Epoch Loop & Evaluation

- Loop over `epochs`.
- For each epoch:
  - Reset state variables.
  - Loop over data loader:
    - Forward pass over each batch.
    - Compute loss.
    - Backward pass with recursive gradient equations.
    - Update weights.
  - After epoch, evaluate on validation/test set:
    - Forward pass without gradient computation.
    - Compute accuracy.
    - Record metrics.
- Repeat for entire training schedule.

---

## 10. Post-Training, Model Conversion & Inference

- Use `model.convert_to_LIF()` (or similar routine) if specified:
  - Replace `CLIF` with `LIF` reset biases.
  - Use stored weights for inference.
- Run inference:
  - Input the test data as spike sequences.
  - Collect output spike responses.
  - Compute final accuracy.
- For calibration, apply a fixed reset bias as per Table 8 in the paper, then evaluate.

---

## 11. Additional Considerations

- **Seed stability**:
  - Set the seed (`2022`) for reproducibility.
- **Hyperparameter sweep**:
  - For τ and surrogate α, implement loops or command-line args to vary these, noting their effect on gradient flow and accuracy.
- **Energy estimation**:
  - During evaluation, log firing rates (spikes per neuron per sample) for energy calculations based on the formulas.
- **Gradient debugging**:
  - Optionally, include routines to check gradient norms to verify non-vanishing.

---

## 12. Summary Flow (Pseudocode Sketch):

```python
for epoch in range(1, epochs+1):
    model.reset_state()
    total_loss = 0
    total_correct = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward_sequence(inputs, T)
        loss = criterion(outputs, labels)
        loss.backward()  # via custom recursive gradient logic
        optimizer.step()
        total_loss += loss.item()
        # Compute accuracy
        pred = outputs.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        # Logging per batch if needed
    # Validation step
    val_acc = evaluate(model, val_loader)
    # Save checkpoint if improved
```

---

## 13. Critical Points for Implementation

- Ensure the backward extension with explicit recursive relations, particularly for `epsilon`, `xi`, and `ψ`, matches the equations for CLIF detailed in the appendix.
- Validate the gradient flow: check no `∂L/∂u[t]` vanishes prematurely; the extra terms from `m[t]` should sustain gradient propagation.
- Confirm the initialized states are consistent (zeros or as specified).
- Keep all computations vectorized for efficiency.
- Precisely match the surrogate gradient parameter `α` and decay factor `γ` to the paper's values.

---

This detailed logical flow and hierarchical breakdown should guide the implementation of `trainer.py`, ensuring correctness and reproducibility aligned with the paper's methodology and experimental design.

## utils.py

# Logic Analysis for utils.py

This utility module will contain functions and classes used broadly across the training and evaluation pipeline to support the core logic, calculation, and operations based on the provided paper, design, and configuration. The functions will be implemented to facilitate surrogate gradient calculation, energy estimation, neuron input encoding, and general helpers needed for neural computations.

---

### 1. Surrogate Gradient Function

**Purpose:**  
Implement the surrogate gradient function for backpropagation of the spiking neuron model, specifically the rectangle function as per Eq. (6) and Eq. (14). The surrogate derivative is used during the backward pass to approximate the non-differentiable Heaviside function.

**Details:**  
- Input: membrane potential `u`, threshold `V_th`, `alpha` (surrogate slope, default 1.0).  
- Output: the surrogate gradient value (float tensor).  
- Implementation:  
  - Use an indicator function that outputs `1/alpha` if `|u - V_th| < alpha/2`, else 0.  
  - This mimics the rectangle surrogate described in the paper.

**Function Signature:**  
```python
def surrogate_gradient(u: torch.Tensor, V_th: float, alpha: float = 1.0) -> torch.Tensor
```

---

### 2. Energy Estimation Functions

**Purpose:**  
Estimate energy consumption based on the activity logs of neurons and synapses as per the detailed analytic formulas in the paper (Tables 6 and 7). This includes calculating the number of synaptic operations (SOPs), considering activity-based (ACs) and weight-based (MACs) operations, and aggregating energy costs.

**Details:**  
- Inputs:  
  - Activity logs: spike counts per neuron or layer (per sample or batch).  
  - Network parameters: total number of neurons, layers, and parameters in the model (can be derived from the model architecture).  
  - Per-operation energy costs: in pJ or similar units (constants).  
- Output:  
  - Total estimated energy in physics units (μJ or pJ).  
  - Breakdown: SOP, ACs, MACs.

- Implementation:  
  - For ACs: sum of spike counts (from logs).  
  - For MACs: sum over weights multiplied by activations or spike counts.  
  - Use constants for energy per operation.

**Function Signature:**  
```python
def estimate_energy(spike_counts: list, model_params: dict) -> dict
```

---

### 3. Dataset Input Encoding

**Purpose:**  
Transform raw dataset inputs into appropriate spike trains. Considering the configuration `direct_spike_encoding`.  
- For static datasets (CIFAR10/100, TinyImageNet):  
  - Convert images to a sequence of binary spikes per pixel, possibly using rate encoding or thresholding.  
- For neuromorphic datasets (DVS Gesture, DVS-CIFAR10):  
  - Use raw event streams directly or perform temporal/frame binning if needed.

**Details:**  
- For simplicity and consistency with the paper, assume input pixels are converted into Poisson spike trains or direct thresholding.  
- Input: raw images or events, dataset-specific; pixel intensity normalized, then encoded.  
- For static images: replicate the pixel intensities across timesteps, thresholded at V_th, or sampled as spike probabilities if appropriate.

**Implementation:**  
- Function to accept image tensor, output spike tensor of shape `[batch_size, channels, height, width, time_steps]`.  
- For neuromorphic data, load streams without additional encoding if preprocessed.

**Function Signature:**  
```python
def encode_input(images: torch.Tensor, T: int, encoding_scheme: str = 'direct_spike_encoding') -> torch.Tensor
```

---

### 4. Helper Functions for Neuron Dynamics

**Purpose:**  
Provide common operations such as decay calculations, reset operations, and spike generation for neuron models. These will be utilized within neuron classes but can be useful as utility functions—for example, to handle membrane potential decay, complementary potential update, and spike detection logic.

**Functions:**

- `decay_potential(u: torch.Tensor, tau: float) -> torch.Tensor`  
  Implements `u[t] = (1 - 1/τ)*u[t-1] + c[t]` or simple exponential decay.

- `generate_spikes(u: torch.Tensor, V_th: float) -> torch.Tensor`  
  Implements spike generation with Heaviside function, i.e., `s = 1 if u >= V_th else 0`.

- `apply_soft_reset(u: torch.Tensor, s: torch.Tensor, V_th: float) -> torch.Tensor`  
  Subtract the reset amount, for soft reset: `u -= V_th * s`.

- `apply_hard_reset(u: torch.Tensor, s: torch.Tensor, V_th: float, bias: float=0.0) -> torch.Tensor`  
  Subtract full reset value (with optional bias).

**Implementation Notes:**  
- These functions should be differentiable where applicable or used in conjunction with custom autograd if needed (e.g., for gradient flow through membrane potential updates).

**Function Signatures:**  
```python
def decay_potential(u: torch.Tensor, tau: float) -> torch.Tensor

def generate_spikes(u: torch.Tensor, V_th: float) -> torch.Tensor

def soft_reset(u: torch.Tensor, s: torch.Tensor, V_th: float) -> torch.Tensor

def hard_reset(u: torch.Tensor, s: torch.Tensor, V_th: float, bias: float=0.0) -> torch.Tensor
```

---

### 5. Utility Functions for Recording and Logging

**Purpose:**  
Track activity such as spike counts, membrane potentials, and gradient norms for analysis and visualization.

**Functions:**

- `count_spikes(s: torch.Tensor) -> int` or array of counts per neuron/layer.
- `compute_autocorrelation(signal: torch.Tensor) -> torch.Tensor`  
  Based on the method in the paper, to compute auto-correlation over the temporal sequence.
- `log_status(epoch: int, loss: float, accuracy: float, energy: dict)`  
  To log training/validation status periodically.

---

### 6. Random Seeds and Initialization

**Purpose:**  
Ensure reproducibility by setting seeds across all relevant modules (`torch`, `numpy`) with the seed value `2022`.  
Implement functions to initialize model parameters explicitly if needed.

**Function:**  
```python
def set_seed(seed: int = 2022) -> None
```

---

### 7. Model Conversion Utility

**Purpose:**  
Convert trained CLIF neurons into LIF neurons for inference, with fixed reset biases as per Table 8.

**Function signature:**  
```python
def convert_clif_to_lif(model: nn.Module, reset_bias: float = 0.0) -> nn.Module
```

---

### 8. Additional Considerations

- **Constants:**  
  - Energy per AC operation: `0.9 pJ`  
  - Energy per MAC operation: `4.6 pJ`  
  Use consistent units throughout calculations.
- **Device handling:**  
  - Ensure functions accept device info and transfer tensors accordingly.
- **Compatibility:**  
  - Use PyTorch-compatible functions, avoid deprecated elements, and write efficient tensor operations.

---

### Summary of Key Functions to Implement:

| Function Name | Purpose | Inputs | Outputs | Remarks |
|----------------|---------|--------|---------|---------|
| `surrogate_gradient` | Compute surrogate derivative for `s = Θ(u - V_th)` | `u`, `V_th`, `alpha` | Tensor | Uses indicator function |
| `estimate_energy` | Estimate energy consumption from activity logs | spike counts, params | dict of energy breakdown | Based on tabulated formulas |
| `encode_input` | Convert images/events to spike trains | images, T, scheme | spike tensor | Fixed or configurable encoding scheme |
| `decay_potential` | Potential decay over time | `u`, `tau` | decayed `u` | Exponential decay |
| `generate_spikes` | Spike generation based on `u` | `u`, `V_th` | spike tensor | Heaviside implementation |
| `soft_reset` | Soft reset of membrane potential | `u`, `s`, `V_th` | reset `u` | \( u \leftarrow u - V_{th} s \) |
| `hard_reset` | Hard reset with bias | `u`, `s`, `V_th`, `bias` | reset `u` | \( u \leftarrow u - (V_{th} + bias) s \) |
| `set_seed` | Set random seed for reproducibility | seed | None | For deterministic behavior |
| `convert_clif_to_lif` | Convert trained CLIFs to LIFs | model, bias | converted model | For inference transfer |

---

This comprehensive logic analysis ensures that every core operation, calculation, and utility needed is considered, properly aligned with the equations, algorithms, and experimental setup detailed in the paper and plan.

