# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## curriculum_strategies.py

### Logic Analysis for `curriculum_strategies.py`

This module will define the core classes and interfaces to implement various curriculum learning strategies, facilitating flexible integration into the overall training framework. The design primarily hinges on a base class `CurriculumStrategy`, with multiple subclasses implementing specific curricula (e.g., difficulty-based, confidence-based, self-paced).

---

## 1. Core Class: `CurriculumStrategy`

**Purpose:**  
Define a unified interface for all curriculum methods, enabling the training loop to obtain sample weights or ordering per epoch and optionally update internal parameters for adaptive curricula.

### 1.1. Attributes:
- **Parameters:**  
  - `start_ratio`: float (0.0 to 1.0), starting proportion/threshold to select easy samples.
  - `grow_epochs`: int, number of epochs over which curriculum gradually progresses.
  - `grow_fn`: str ('linear', 'exponential', etc.), dictates how curriculum difficulty increases.
  - `warm_epochs`: int, number of initial epochs for warm-up (no curriculum modification).
  - `parameters`: dict, for storing strategy-specific hyperparameters.

### 1.2. Methods:
- **`get_sample_weights(dataset, model, epoch) -> Tensor`:**  
  - Inputs:
    - `dataset`: dataset object containing data samples (e.g., images, texts, graph data).
    - `model`: current model instance, to evaluate or infer difficulty.
    - `epoch`: current epoch number (int).
  - Outputs:
    - `weights`: Tensor of size `[dataset_size]`, giving importance weight or sampling probability for each sample.
  - Logic:
    - Compute difficulty scores for samples based on strategy.
    - Transform difficulty scores into weights based on current curriculum progress (how far along `grow_epochs` we are, grow_fn).
    - For fixed curricula, may return static weights.
    - For adaptive curricula, may update internal parameters based on model feedback.

- **`update_strategy(epoch, model, dataset) -> None` (optional):**  
  - Used for strategies that adapt over epochs, e.g., updating difficulty estimations based on current model performance.
  - Default implementation: pass (no update).

### 1.3. Initialization:
- Store hyperparameters (start_ratio, grow_epochs, grow_fn, warm_epochs).
- Prepare any internal data structures (e.g., difficulty scores cache).

---

## 2. Subclasses of `CurriculumStrategy`:

Each subclass implements the `get_sample_weights()` method with distinct logic:

### 2.1. Difficulty-based Curriculum (DifficultyBased)

**Approach:**  
- Use an explicit difficulty measure (e.g., loss, entropy, image noise level).
- Techniques:
  - Precompute difficulty scores or compute dynamically via model predictions.
  - E.g., for images, higher noise or model confidence on samples could define difficulty.

**Implementation notes:**  
- Difficulty scores could be stored in `dataset.difficulty_scores` or computed on-the-fly.
- During each epoch:
  - Calculate a threshold corresponding to the current curriculum progress.
  - Assign higher weights to easier samples (lower difficulty scores).

**Parameters:**  
- `difficulty_metric`: function or string identifier ("loss", "confidence", "entropy", "noise_level").
- `curriculum_progress`: function of `epoch`, `start_ratio`, `grow_fn`, `grow_epochs`, `warm_epochs`.

**Sample weights calculation:**  
- Normalize difficulty scores to [0,1].
- Use a decay function (e.g., `grow_fn`) to accelerate or slow curriculum progression.
- Assign weights accordingly, emphasizing or de-emphasizing samples.

### 2.2. Confidence-based Curriculum (ConfidenceBased)

**Approach:**  
- Use model's prediction confidence or entropy as difficulty indicators.
- During training:
  - Compute confidence scores (e.g., softmax probability for predicted class).
  - For samples with low confidence (less certain), assign higher difficulty.

**Implementation:**  
- On each epoch:
  - Forward pass on dataset to evaluate sample confidence scores.
  - Normalize scores.
  - Adjust weights based on curriculum schedule.

**Optional:**  
- Can update confidence metrics during training (if true online adaptation).

### 2.3. Self-Paced Learning (SelfPaced)

**Approach:**  
- Start with easy samples (e.g., with low loss or high confidence).
- Gradually include harder samples over epochs.
- Incorporate warm-up phase.

**Implementation:**  
- Use model loss or uncertainty as difficulty.
- Implement schedule:
  - Increase `start_ratio` or threshold over `grow_epochs`.
  - Use `grow_fn` to control progression shape.

### 2.4. Noise-Robust Curriculum (Noise-based)

**Approach:**  
- Use label noise estimates or sample loss to detect noisy samples.
- Discard or down-weight samples suspected to be noisy, especially in early epochs.
- Focus on clean samples first.

---

## 3. Parameters & Hyperparameters:
- `start_ratio`: controls initial selection of samples.
- `grow_epochs`: number of epochs over which curriculum progresses.
- `grow_fn`: determines shape of progression:
  - `'linear'`: progress linearly from start_ratio to full dataset.
  - `'exponential'`, `'logarithmic'`: alternative shaping functions.
- `warm_epochs`: initial epochs where curriculum parameters are kept static, possibly no filtering/exclusion.
- Strategy-specific hyperparameters:
  - e.g., noise level, difficulty scoring function, confidence thresholds.

## 4. Integration with Training Loop
- During each epoch:
  - Call `get_sample_weights(dataset, model, epoch)` to obtain sampling weights.
  - Use these weights for:
    - Weighted sampling of mini-batches.
    - Sample filtering (excluding high difficulty).
  - Use `update_strategy()` if needed, e.g., recalibrating difficulty scores.

---

## 5. Summary
- **Design pattern:** Abstract base class with shared interface, subclasses implementing specific difficulty or learning criteria.
- **Parameters management:** all strategy parameters passed via constructor or `hyperparameters`.
- **Epoch-wise operation:** dynamically compute sample weights based on the current epoch, model feedback, and curriculum progression.
- **Optional updating:** strategies that adapt over time may override `update_strategy()` for difficulty recalibration.

This logic ensures a flexible, extensible, and coherent implementation of curriculum strategies, consistent with the experimental procedures in the paper and compatible with the overall framework described in your design plan.

## dataset_loader.py

**Logic Analysis for `dataset_loader.py` — DatasetLoader class**

---

### Goal:
Implement a robust `DatasetLoader` class that handles dataset loading, preprocessing, transformation, and data augmentation, adhering to the experimental protocols described in *CurBench*. The class must support various datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet, MNIST, Graph datasets, etc.), apply label noise and class imbalance transformations, and provide an interface for downstream training and evaluation.

---

### 1. Class Overview:
- **Class Name:** `DatasetLoader`
- **Primary Responsibilities:**
  - Initialize with configuration parameters (dataset name, split ratios, noise ratio, imbalance factor).
  - Load datasets in their native formats using appropriate APIs.
  - Apply label noise for the *noise setting* with specified ratio `p`.
  - Generate class imbalance for the *imbalance setting* using specified ratio `r`.
  - Support data augmentation where applicable.
  - Provide clean and transformed datasets via `load_data()` and `apply_noise_or_imbalance()`.

### 2. Inputs & Configuration:
- **Constructor (`__init__`) Parameters:**  
  Input options may include:
  - `dataset_name` (string): Which dataset to load (`CIFAR10`, `CIFAR100`, `Tiny-ImageNet`, `MNIST`, etc.)
  - `split_ratios` (dict): For train/val/test splits (default 80%/10%/10%).
  - `noise_ratio` (`p` float in [0,1]): Probability to flip labels for label noise.
  - `imbalance_factor` (`r` float): Multiplier for class imbalance.
  - Additional options: `apply_augmentation`, `difficulty_scores`, etc.

- **Constants & Hyperparameters:**
  - For datasets like CIFAR, use standard normalization transforms.
  - For noise, generate random label corruption.
  - For imbalance, select a sampling strategy to reduce samples per class.

---

### 3. Dataset API:
- **`load_data()` Method:**
  - Executes dataset loading using:
    - `torchvision.datasets.CIFAR10` or `CIFAR100`
    - `torchvision.datasets.ImageFolder` (for Tiny-ImageNet)
    - `torchvision.datasets.MNIST`  
    - External APIs for graph datasets (e.g., OGB, TUDataset)
  - Returns datasets in a common format (e.g., PyTorch `Dataset`, or custom wrapper).

- **Dataset Preparation:**
  - After loading raw data, perform:
    - Split into `train`, `val`, `test` sets based on `split_ratios`.
    - Store datasets with associated labels, features, or graph attributes.
    - Possibly store or generate sample difficulty scores (for curriculum).

---

### 4. Noise and Imbalance Application:
- **`apply_noise_or_imbalance()` Method:**
  - Intended to modify the dataset in-place or return a new dataset with the specified transformations.
  - **Noise**:
    - For each sample in training set, with probability `p`, assign a random label uniformly among other classes.
    - Ensure reproducibility by seeding the randomness.
  - **Imbalance**:
    - For class imbalance, reduce the number of samples per class to follow an exponential or geometric distribution:
      - Use the formula: `n_c = n_0 * d^c`, where `d` is a decay factor.
    - Implement via a stratified sampling subset that retains the desired imbalance ratio.
    - Ensure that the reduction is consistent with the specified `r`.
  - **Implementation notes**:
    - For label noise:
      - Loop or vectorized operation on labels.
      - Use `np.random` or `torch` for randomization.
    - For class imbalance:
      - Calculate number of samples per class.
      - Randomly select the required number of samples for each class.
      - Handle cases where the number exceeds available data.

---

### 5. Data Transformation & Augmentation:
- Utilize torchvision transforms:
  - Normalize: mean/std normalization for CV datasets.
  - Augmentation: random crops, flips, color jitters, etc.
- For graph datasets:
  - Use appropriate graph transforms (if applicable), e.g., node feature augmentation, subgraph sampling.
- These transformations are to be applied during data loading or preprocessing pipeline.

---

### 6. API Behavior:
- **`load_data()`**:
  - Loads raw datasets.
  - Applies initial preprocessing.
  - Returns dataset objects (e.g., `torch.utils.data.Dataset` or datasets compatible with DataLoader).
- **`apply_noise_or_imbalance()`**:
  - Transforms the datasets according to specified noise/imbalance ratios.
  - Modifies labels or sample distributions.
  - Returns a transformed dataset, possibly a new dataset object, ready for training.
- **Clarity & Reproducibility**:
  - Use fixed random seed (e.g., from the configuration) during noise/imbalance generation.
  - Log the applied transformations for reproducibility.

### 7. Edge Cases & Special Considerations:
- Support for datasets without labels (or with complex labels, e.g., in graphs).
- Ensure no data leakage: do not apply noisy/imbalanced transformations to validation/test sets.
- For datasets like Tiny-ImageNet lacking official test labels, load validation set as test, following the paper.
- Handle the format conversions needed for datasets, e.g., images to tensors, graphs to data objects.
- Document assumptions about dataset structures.

---

### 8. Implementation Details & Final Notes:
- Should be compatible with main training pipeline.
- Should implement debugging/logging outputs (e.g., info logs on data sizes, noise, imbalance info).
- Provide clear and simple API for external code to invoke data loading and transformations:
  - Example:
    ```python
    loader = DatasetLoader(config_params)
    dataset = loader.load_data()
    transformed_dataset = loader.apply_noise_or_imbalance(dataset)
    ```
- Modular design to extend to other datasets or transformations easily.

---

### Summary:
- The `DatasetLoader` class systematically manages dataset fetching, splitting, and transformations.
- Apply label noise and class imbalance rigorously, respecting reproducibility.
- Integrate data augmentation seamlessly during data loading.
- Provide consistent datasets for the training pipeline, aligning with the experimental protocols described in the paper.

This logic will guide the precise coding of the dataset handling module aligned with *CurBench*'s requirements.

## evaluation.py

**Logic Analysis for `evaluation.py` — Implementation of the Evaluation Class**

---

### Purpose:
The `Evaluation` class is responsible for:
- Taking a trained model and a dataset,
- Computing quantitative performance metrics (accuracy, F1-score, Spearman correlation, etc.) according to dataset/task type,
- Logging or returning resource usage metrics such as training time and maximum GPU memory consumption,
- Producing a structured metrics dictionary as output.

---

### Core Responsibilities:
1. **Input Handling:**
   - Accepts:
     - `model`: A trained PyTorch model.
     - `dataset`: Dataset object consisting of test (and optionally validation) data.
   - Ensure model is loaded onto the correct device (`cuda` or `cpu`) matching the training device.

2. **Evaluation on Dataset:**
   - Iterate over the dataset (test set).
   - Perform inference (forward pass).
   - Collect predictions and true labels.
   
3. **Metrics Computation:**
   - Determine task type based on dataset info:
     - **Classification Tasks:** Use accuracy, F1-score, Matthews, etc.
     - **Regression Tasks:** Use Spearman correlation, Pearson correlation, etc.
     - **Graph datasets:** Use accuracy, ROC-AUC.
     - **Natural Language tasks:** Use accuracy, sometimes F1 or other task-specific metrics.
   - For each metric:
     - Aggregate over all predictions (e.g., via `sklearn.metrics`).
     - Compute mean and standard deviation if multiple runs are supported (though in this context, metrics are for a single run).

4. **Resource Usage Logging:**
   - Use `torch.cuda` API for:
     - GPU memory peak usage (`torch.cuda.max_memory_allocated()` or `torch.cuda.get_device_properties()`).
     - Time measurement possibly via `time.time()` or more precise timers.
   - Record total evaluation time.

5. **Output Formation:**
   - Return a dictionary with:
     - Performance metrics keyed by metric names.
     - Resource usage info:
       - Total evaluation time.
       - Max GPU memory consumption.
   - Support inclusion of multiple metrics, depending on dataset/task.

---

### Step-by-Step Implementation Details:

**1. Setup and Initialization:**
- The `__init__` method initializes with:
  - `model` (pre-trained and loaded onto appropriate device).
  - `dataset` (test data loader object, e.g., `torch.utils.data.DataLoader`).
- Optional: other parameters like task type, or evaluation flags.

**2. Data Loading and Inference:**
- Use a DataLoader for batch-wise processing.
- For each batch:
  - Transfer input data to device.
  - Perform inference: `outputs = model(inputs)`.
  - Store predictions and true labels.
- Use `torch.no_grad()` context to disable gradients.

**3. Metric Calculation:**
- After inference:
  - Convert predictions and labels to CPU (if on GPU).
  - Use `sklearn.metrics`:
    - `accuracy_score`
    - `f1_score`
    - `mathews_corrcoef`
    - `spearmanr`
    - `roc_auc_score` (for graph datasets or binary classification)
- Handle dataset/task-specific metrics.

**4. Resource Monitoring:**
- **Timing:**
  - Record start time before inference.
  - Record end time after.
  - Evaluate total evaluation duration.
- **Memory:**
  - Before evaluation, record initial memory (`torch.cuda.max_memory_allocated()`).
  - After, record maximum memory usage.
  - Reset or update peak memory usage.

**5. Final Metrics Assembly:**
- Structure output dict:
  ```python
  metrics_dict = {
      "accuracy": value,
      "f1_score": value,
      "spearman": value,
      "mathews": value,
      "roc_auc": value,
      "evaluation_time_sec": total_time,
      "max_gpu_memory_MB": max_memory_in_MB
  }
  ```
- Allow optional inclusion of only relevant metrics per dataset type.

---

### Additional Considerations:
- **Multiple metrics per dataset:** select or configure metrics accordingly.
- **Multiple runs support:** in this code, metrics are computed for a single run; if multiple runs are required, this logic can be outside Evaluation or aggregate over runs.
- **Handling missing metrics:**
  - For datasets/tasks lacking certain metrics, ensure safe checks.
- **Compatibility & Device Safety:**
  - Make sure computations happen on the same device.
  - Use `torch.cuda.empty_cache()` if memory management is needed.

---

### Summary:
The `Evaluation` class in `evaluation.py` focuses on inference, task-specific metric computation, resource usage measurement, and results packaging into a dictionary. It operates on the assumption that the model is correctly trained and valid on the dataset, and resource tracking is enabled as per configurations. The implementation is designed for flexibility to support datasets from various domains, with metrics being modular and extendable.

---

*This detailed logic provides a clear plan to implement a robust evaluation module aligned with the paper's experimental protocols, ensuring reproducibility and comprehensive reporting.*

## main.py

### Logic Analysis for `main.py` (Entry Point)

This script acts as the central orchestrator for the entire experiment pipeline, initializing all components, executing training, evaluation, logging, and result saving, respecting the experimental configurations. The goal is to organize a clear, reproducible, and flexible flow that adheres to the benchmark’s systematic evaluation framework.

---

#### 1. **Configuration Parsing**

- **Input:**  
  - Load the configuration dictionary from `config.yaml`, containing dataset, model, train, curriculum, hyperparameters, and resource logging settings.
- **Process:**  
  - Use `yaml.safe_load()` to parse `config.yaml`.  
  - Optionally, include command-line argument parsing (via `argparse`) for overriding or specifying a different config file or parameters.
- **Output:**  
  - A comprehensive `config` object/dict for subsequent component initialization.

---

#### 2. **Set Random Seeds & Environment Setup**

- **Purpose:**  
  - Ensure reproducibility across runs.
- **Process:**  
  - Use `utils.set_seed(config['hyperparameters']['seed'])`.  
  - If specified, enable deterministic behavior in `torch.backends.cudnn`.
  - Seed all relevant libraries (`numpy`, `torch`, etc.).
- **Note:**  
  - Consistency is crucial for experimental fairness.

---

#### 3. **Initialize Resource Logging (Optional)**

- **Conditional:**  
  - If `resource_logging['enable_time_tracking']` or `enable_memory_tracking` is true, instantiate resource monitor.
- **Process:**  
  - Use `resources.Resources()` object to start tracking; prepare to log per-epoch data.
- **Outcome:**  
  - Time and memory metrics recorded for later reporting.

---

#### 4. **Dataset Initialization**

- **Input:**  
  - Read dataset name, split ratios, noise ratio, imbalance factor from config.
- **Process:**  
  - Instantiate `DatasetLoader` with parameters (dataset name, split ratios, noise_ratio, imbalance_factor).
  - Call `load_data()` to:
    - Download/load dataset.
    - Apply data transformations (normalize, augment).
    - Apply noise (label corruptions) if `noise_ratio > 0`.
    - Apply class imbalance adjustments if `imbalance_factor > 1`.
  - Obtain datasets: `train_dataset`, `val_dataset`, `test_dataset`.
- **Output:**  
  - Properly prepared datasets with testing/validation splits adhered to experimental protocols.

---

#### 5. **Model Initialization**

- **Input:**
  - Model type (e.g., `ResNet-18`) and hyperparameters.
- **Process:**
  - Call `models.ModelFactory(model_type, hyperparameters)` or similar factory pattern.
  - If pretrained is specified (e.g., BERT, GPT2), load pretrained weights.
  - Modify input/output layers based on dataset specifics:
    - Number of classes.
    - Input shape (images, tokens, graphs).
- **Outcome:**
  - A model instance ready for training.

---

#### 6. **Optimizer & Scheduler Setup**

- **Input:**  
  - Hyperparameters: learning rate, weight decay, optimizer type, scheduler parameters.
- **Process:**  
  - Initialize optimizer: e.g., `optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)`.
  - Initialize scheduler if specified (e.g., `StepLR`, `CosineAnnealingLR`).
- **Outcome:**  
  - Optimizer and scheduler objects for the training loop.

---

#### 7. **Curriculum Strategy Initialization**

- **Input:**  
  - Curriculum strategy from config (e.g., `DifficultyBased`) plus parameters (`start_ratio`, `grow_epochs`, etc.)
- **Process:**  
  - Instantiate `curriculum_strategies.CurriculumStrategy` class corresponding to the specified strategy.
  - Pass necessary parameters such as `start_ratio`, `grow_fn`, `warm_epochs`.
- **Outcome:**  
  - Curriculum object capable of generating sample weights or data filters per epoch.

---

#### 8. **Trainer Initialization**

- **Input:**  
  - Model, datasets, curriculum, optimizer, scheduler, hyperparameters.
- **Process:**  
  - Instantiate `Trainer(model, dataset, curriculum, hyperparameters, optimizer, scheduler, resource_logger)`.
- **Outcome:**  
  - Prepared trainer object that manages the epoch loop, data sampling (= curriculum schedule), model updates, and resource metrics.

---

#### 9. **Training Loop Execution**

- **For each epoch in total epochs:**
  - **Resource Recording:**  
    - If enabled, record start time and memory usage before epoch.
  - **Curriculum Update:**  
    - Call `curriculum.get_sample_weights(dataset, model, epoch)`  
    - Possibly, `curriculum.update_strategy(epoch, model, dataset)` if adaptive or schedule-based.
  - **Data Preparation for Epoch:**  
    - Use sample weights or ordering to select a subset or reweight samples.
    - Construct data loader accordingly (weighted sampling, filtering, or reweighting in loss).
  - **Model Training:**  
    - Run `trainer.train_one_epoch()` or full epoch method:
      - Forward pass with sample weights if applicable.
      - Backward pass and optimizer step.
  - **Resource Logging:**
    - Record epoch duration.
    - Record maximum GPU memory if enabled.
  - **Optional:**  
    - Log intermediate metrics (loss, accuracy, resource usage).
- **End of Loop:**  
  - Save training logs, hyperparameters, per-epoch resource data.

---

#### 10. **Model Evaluation**

- **Process:**
  - Instantiate `Evaluation(model, dataset)` for test dataset.
  - Call `evaluate()` to compute metrics:
    - Accuracy, F1, Spearman, etc., depending on task.
  - Record evaluation metrics with associated dataset and model info.
- **Optional:**  
  - Save details into a results file or logs for analysis.

---

#### 11. **Result Saving & Visualization**

- **Metrics & Resources:**
  - Save final metrics, resource logs (training time, memory).
  - Generate plots: performance vs. noise ratios, resource usage across methods.
- **Output:**
  - CSV or JSON files for logs.
  - Plots saved in output directory.

---

#### 12. **Cleanup & Reproducibility**

- Finalize logs, close resource monitors.
- Save a comprehensive experiment report:
  - Config used.
  - Seed.
  - Hyperparameters.
  - Results with std deviations.
  - Resource metrics.

---

### Summary:
- This `main.py` ensures a **full, reproducible experimental pipeline**:
  - Load config → setup environment.
  - Prepare datasets → apply noise/imbalance.
  - Initialize models → prepare optimizers and schedulers.
  - Instantiate curriculum strategies.
  - Run epoch-wise training with integrated curriculum.
  - Record resource usage.
  - Evaluate on test set.
  - Save logs and results.

### Unclear points (for clarification, if needed):
- Exact format of difficulty annotations or how to incorporate them beyond noise/imbalance.
- Specific resource logging library or API preferences.
- Handling of curriculum update strategies within each epoch when applicable.
- Visualization or detailed logging expectations.

This comprehensive plan will guide the implementation of `main.py` to match the described methodology, experimental setups, and evaluation protocol from the paper and their supplementary materials.

## models.py

**Logic Analysis for models.py**

This module is intended to define all model classes used within the benchmark, conforming to the specific architectures for each domain (CV, NLP, graph) as described in the paper and requirements. Each model class should be implemented with an __init__() constructor that accepts configuration parameters, and a forward() method that performs the forward pass. The goal is to create flexible, standardized classes that can be instantiated with hyperparameters, optionally load pretrained weights, and be used uniformly within training and evaluation pipelines.

---

### 1. General Design Principles

- **Modularity & Extensibility:**  
  Each model should be encapsulated in a class with a consistent interface (`__init__()` and `forward()`). This enables easy swapping and addition of new models.

- **Hyperparameters Utilization:**  
  All models should accept a `params` dictionary containing relevant hyperparameters (e.g., number of layers, dropout rates, pretrained flags), which should be passed during instantiation.

- **Pretrained Weights:**
  - For some models (e.g., BERT, GPT2, ResNet), support an option to load pretrained weights based on configuration.
  - For others (LeNet, GCN, GAT, GIN, LSTM), typically initialize randomly unless specified otherwise.

- **Device Compatibility:**  
  - All models are instantiated on CPU or CUDA (as set in the main pipeline). No explicit device placement should be hardcoded here; place model on CPU initially, and move to device outside this module.
  - Forward pass takes input tensor `x` (N, C, H, W for images; sequence tensors for NLP; graph data objects for GNNs).

- **Implementation Details:**
  - Use `torchvision.models` for CV architectures (LeNet, ResNet, ViT).
  - Use `transformers` library for BERT, GPT2.
  - Use `torch_geometric.nn` for GNN models (GCN, GAT, GIN).

### 2. CV Models

#### (a) LeNet
- Build a custom class implementing LeNet architecture manually or adapt from references (e.g., a simple sequence of Conv, Pool, FC layers).
- Hyperparameters may include input shape and number of classes.
- Since LeNet is simple, implement a class inheriting `torch.nn.Module`, define convolutional layers, pooling, and classifiers explicitly.
- No pretrained weights are typical; initialize weights randomly.

#### (b) ResNet-18
- Use torchvision.models resnet18 with `pretrained` option.
- Customize the final layer (`fc`) to match the number of classes for CIFAR datasets.
- Support loading pretrained on ImageNet if `params['pretrained'] == True`.
- Modify `fc` layer to output the number of classes (10 for CIFAR-10, 100 for CIFAR-100, etc.).

#### (c) ViT (Vision Transformer)
- Use `vit-pytorch`'s implementation or `transformers` library.
- Load pretrained weights if `pretrained=True`.
- Adjust the `num_classes` parameter for dataset (e.g., 10 for CIFAR-10, 200 for Tiny-ImageNet).
- Handle image patch size and other hyperparameters via constructor parameters if needed.

### 3. NLP Models

#### (a) LSTM
- Implement a standard `torch.nn.Module` with an `nn.LSTM` layer.
- Hyperparameters include input size, hidden size, number of layers, bidirectionality.
- Support pretrained embeddings if possible, but not specified; otherwise randomly initialized.
- Use standard optimizer setting outside; focus on sequence modeling.
- Forward should support input tensors of shape `(seq_len, batch_size, input_dim)` or `(batch_size, seq_len, input_dim)` based on data pipeline.

#### (b) BERT
- Load from `transformers`, e.g., `from transformers import BertModel`.
- Supports loading pretrained weights (`pretrained=True`) or initializing from scratch based on params.
- Final classifier head (if needed) can be added externally; for this purpose, implement a wrapper that returns the BERT backbone.
- Model should return embeddings or classification logits depending on usage.
- Adjust number of labels/output dims for specific tasks.

#### (c) GPT-2
- Similar approach as BERT, use `transformers` library.
- Load pretrained weights if specified.
- For downstream tasks, can add classification head; for pure feature extraction, keep GPT2 model only.
- The output in forward() should match requirements (e.g., last hidden state).

### 4. Graph Models

#### (a) GCN
- Use `torch_geometric.nn.GCNConv` modules.
- Hyperparameters include number of layers, hidden units, activation functions.
- Able to initialize with random weights, or load from checkpoint if necessary.
- Forward accepts a `torch_geometric.data.Data` object (`x`, `edge_index`, `edge_attr`, etc.).

#### (b) GAT
- Similar to GCN but use `torch_geometric.nn.GATConv`.
- Hyperparameters: number of heads, hidden units, dropout etc.

#### (c) GIN
- Use `torch_geometric.nn.GINConv`.
- Hyperparameters similar to GCN/GAT, with possibly different readout layers.

### 5. Pretrained & Initialization Strategy
- For models with available pretrained weights, load with `pretrained=True`.
- For models pretrained on different datasets (e.g., BERT, GPT2), load via transformers with `from_pretrained`.
- For others, initialize weights randomly.
- Make the model ready for training (no further modifications needed here).

### 6. Return Types & Usage
- Each class should inherit `torch.nn.Module`.
- The `forward(x)` method performs the data flow, returning outputs suitable for loss functions.
- For GNNs, forward input will usually be a `Data` object; models should handle this internally.
- For NLP models, consider tokenized inputs.
- For vision, raw image tensors.

### 7. Additional Considerations
- **Weight Initialization:** Optionally, define a method to reinitialize weights if needed.
- **Preprocessing:** The models.py should not handle data preprocessing but need to set up the expected input shape.
- **Hyperparameters passing:** All model-specific parameters should be extracted from `params` dict during `__init__()`.
- **Device placement:** leave model on CPU initially; move to GPU outside this module.

---

### **Summary of Key Actions**

- Implement individual classes for each model, with constructor accepting `params`.
- Use `torchvision/models` (ResNet, LeNet if custom) for CV models.
- Load pretrained weights when available, customize final layers (number of classes).
- Use `transformers` for BERT, GPT2, support pretrained loading.
- Use PyTorch Geometric for GNN models; initialize weights or load pre-trained if applicable.
- Provide clean API: `__init__(params)` and `forward(x)`.

### 8. Pseudocode Snippet (for guidance)
```python
class ResNet18(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=params.get('pretrained', False))
        num_classes = params.get('num_classes', 10)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Similar for other models...
```

---

This detailed logic ensures consistent, flexible, and extendable model implementations aligned with the experimental design outlined in the paper.

## resources.py

**Logic Analysis for `resources.py`**

---

### **Purpose & Responsibilities**

The `resources.py` module is designed to facilitate resource monitoring during the training and evaluation processes within the CurBench framework. Its core functions include:

- Tracking **training time** (per epoch and overall).
- Monitoring **GPU memory consumption** at specified points.
- Logging resource usage metrics systematically for subsequent analysis.
- Providing easy-to-use interfaces that integrate smoothly with the training pipeline.

### **Design Principles & Requirements**

1. **Simplicity & Flexibility**  
   - The module should support optional resource tracking based on configuration flags (`enable_time_tracking`, `enable_memory_tracking`) — as specified in `config.yaml`.
   - It must be compatible with multi-GPU setups but primarily designed for single-GPU tracking to simplify implementation (can be extended for multi-GPU).

2. **Accuracy & Precision**  
   - Use high-resolution timers for measuring epoch durations (e.g., `time.time()` or `time.perf_counter()`).
   - Use `torch.cuda.max_memory_allocated()` or `torch.cuda.get_max_memory_allocated()` to obtain GPU memory peaks during training.

3. **Modular & Reusable Interface**  
   - Implement a class `ResourceLogger` with methods:
     - `start_epoch()` and `end_epoch()` to mark resource measurement boundaries.
     - Internal variables/attributes to store cumulative time, maximum memory, etc.
   - Log data can be stored in dictionaries or pandas DataFrame for exporting (CSV, JSON).

4. **Integration & Usage Workflow**

   - Before each epoch begins, `start_epoch()` is called:
     - Record start time (`start_time`).
     - Record initial GPU memory (if enabled).
   - After each epoch ends, `end_epoch()` is called:
     - Compute elapsed time.
     - Record peak GPU memory (if enabled).
     - Append resource metrics to logs.
   - At training completion, collate all logs for reporting/visualization.

5. **Handling Multi-GPU and Multi-device contexts**
   
   - For multi-GPU, index accordingly or sum resources; but for initial design, assume single GPU.
   - Wrap CUDA calls inside try-except to handle environments without GPU.

---

### **Specific Implementation Details**

- **Initialization:**
  - Accept a configuration object or environment variables for enabling time and memory tracking.
  - Initialize storage variables, e.g., `self.epoch_times = []`, `self.peak_memory = 0`.

- **Memory Tracking:**
  - Use `torch.cuda.reset_peak_memory_stats()` at start of training or epoch.
  - Use `torch.cuda.max_memory_allocated()` after epoch to get maximum memory used during epoch.
  - Store values in MB (bytes / 1e6).

- **Time Tracking:**
  - Use `time.perf_counter()` for precise interval measurement.
  - Record start timestamp at beginning of epoch.
  - Record end timestamp at completion; difference yields epoch duration.

- **Logging & Export:**
  - Store per-epoch resource data (epoch index, time, peak memory).
  - Support exporting logs to CSV or JSON upon training end, or on demand.

- **Optional Extensibility:**
  - Add methods for reset, multiple device support, or real-time updates.

---

### **Sample Pseudocode Structure**

```python
class ResourceLogger:
    def __init__(self, enable_time_tracking=False, enable_memory_tracking=False):
        self.enable_time_tracking = enable_time_tracking
        self.enable_memory_tracking = enable_memory_tracking
        self.epoch_logs = []  # List to hold per-epoch resource data
        self.start_time = None
        self.peak_memory = 0

    def start_epoch(self):
        if self.enable_time_tracking:
            self.start_time = time.perf_counter()
        if self.enable_memory_tracking:
            torch.cuda.reset_peak_memory_stats()

    def end_epoch(self, epoch_idx):
        epoch_time = None
        max_mem = None
        if self.enable_time_tracking:
            epoch_time = time.perf_counter() - self.start_time
        if self.enable_memory_tracking:
            max_mem_bytes = torch.cuda.max_memory_allocated()
            max_mem = max_mem_bytes / 1e6  # convert to MB
            self.peak_memory = max(self.peak_memory, max_mem_bytes)
        # Save per-epoch data
        self.epoch_logs.append({
            "epoch": epoch_idx,
            "time_sec": epoch_time if self.enable_time_tracking else None,
            "max_memory_MB": max_mem if self.enable_memory_tracking else None
        })

    def export_logs(self, filename):
        # Export the logs as CSV or JSON
        pass
```

---

### **Integration points in the main training loop**

- At the **start of each epoch**: call `resource_logger.start_epoch()`.
- After completing an epoch: call `resource_logger.end_epoch(epoch_idx)`.
- After training conclusion: call `resource_logger.export_logs()`.

---

### **Additional Considerations**

- **Resource limitations:** Incorporate checks for scenarios where CUDA is unavailable, fall back to dummy logs or skip resource measurement.
- **Synchronization:** For accurate timing and memory measurement, ensure no overlapping asynchronous GPU operations interfere.
- **Portability:** Optionally, extend to support CPU (via `psutil` or `memory_profiler`), but initial design focuses on GPU.

---

## **Summary**

The `resources.py` will contain a `ResourceLogger` class, designed to:

- **Optionally** measure training time **per epoch**.
- **Optionally** monitor maximum GPU memory **per epoch**.
- **Collect and** save resource utilization data systematically.
- **Support hooks** into the training loop for seamless operation.
  
This design aligns with the experimental benchmarks, ensuring resource data is captured accurately, systematically, and efficiently for all curriculum methods and datasets.

## trainer.py

{
  "file": "trainer.py",
  "Logic Analysis": [
    "Class 'Trainer' serves as the orchestration core for training models with curriculum learning.",
    "Initialization (__init__):",
    "  - Accepts inputs: model (instance of a Model class), dataset (loaded Dataset object), curriculum (instance of a CurriculumStrategy), hyperparameters (dict).",
    "  - Sets internal attributes for optimizer, scheduler, hyperparameters, resource logging, and other configurations based on inputs.",
    "  - Initializes training data loader(s), potentially including weighted sampling based on the curriculum's sample weights.",
    "Training method ('train'):",
    "  - Loop over epochs (range from 1 to total epochs):",
    "    - Before each epoch:",
    "      - If curriculum is adaptive, invoke curriculum's 'update_strategy' with current epoch, model, dataset.",
    "      - Fetch sample weights or selection criteria by calling curriculum's 'get_sample_weights' method, passing current dataset, model, epoch.",
    "      - If the curriculum provides sample weights:",
    "        - Use these weights to perform weighted sampling or filtering of dataset batches or samples, ensuring that the data fed into each epoch reflects the curriculum strategy.",
    "    - For each batch in the DataLoader:",
    "      - Move input data (images, sequences, graphs) and labels to the device ('cuda' or 'cpu').",
    "      - Compose batch inputs considering sample weights or selections from curriculum if applicable.",
    "      - Perform model forward pass:",
    "        - Compute the predictions: predictions = model(inputs).",
    "      - Compute loss:",
    "        - Use criterion (loss function) appropriate to dataset/task (e.g., cross-entropy for classification).",
    "        - If sample weights are available, incorporate them into the loss calculation, e.g., loss = (sample_weight * criterion(predictions, labels)).mean().",
    "      - Backpropagation:",
    "        - Zero optimizer gradients: optimizer.zero_grad().",
    "        - Backward pass: loss.backward().",
    "        - Optimizer step: optimizer.step().",
    "      - Optional: scheduler.step() if a learning rate scheduler is used, possibly per epoch or per batch.",
    "    - Resource Logging:",
    "      - Record training time for the epoch if resource tracking is enabled.",
    "      - Record maximum GPU memory usage during the epoch if enabled, using torch.cuda.max_memory_allocated() or similar API.",
    "    - Iteration completes the epoch; log metrics and resources as needed.",
    "Post-epoch operations:",
    "  - At the end of all epochs or upon early stopping criteria:",
    "    - Save model checkpoints if required.",
    "    - Record final training metrics.",
    "  - Return or store logs for performance and resource usage.",
    "Handling of Curriculum Strategies:",
    "  - General interface expected:",
    "    - 'get_sample_weights' returns a tensor of weights or a subset of data indices.",
    "    - 'update_strategy' applies if the method is adaptive and updates internal parameters based on training progress.",
    "  - Examples:",
    "    - For data selection strategies: generate a sample weight vector or index list for sampling.",
    "    - For loss reweighting strategies: provide sample weights used in loss calculation.",
    "Resource Management:",
    "  - Use 'resources.py' or integrated functions to time each epoch accurately.",
    "  - Use torch.cuda.memory_allocated() before and after epochs to compute peak memory if enabled.",
    "  - Log all resource metrics with appropriate labels for later analysis.",
    "Error Handling & Reproducibility:",
    "  - Set fixed seed before training starts for reproducibility across runs.",
    "  - Wrap training loop with try-except to manage interruptions and log errors gracefully.",
    "Hyperparameters & Configs:",
    "  - Read from the passed 'hyperparameters' dict: learning rate, batch size, epochs, log interval, device, etc.",
    "  - Initialize optimizer and scheduler according to configurations.",
    "  - Log hyperparameters and setup for reproducibility.",
    "Summary:",
    "  - The 'train' method aligns with the experimental protocols described in the paper: fixed hyperparameters, resource tracking, multiple runs, and curriculum integration.",
    "  - Modular design: curriculum strategies are invoked per epoch; resource logging is separated for clarity."
  ]
}

## utils.py

# Logic Analysis for `utils.py`

This file provides essential utility functions to support reproducibility, configuration handling, resource logging, random seed setting, and data visualization. It ensures consistent experimental conditions and convenient logging of experiment metrics, resource usage, and hyperparameters. Below is a detailed breakdown of its core functions and their design considerations:

---

## 1. Configuration Parsing and Handling

**Objective:**  
- Read and process the experimental configuration from the `config.yaml` file.
- Provide an easy interface to access nested parameters.
- Enable consistency across modules by centralized configuration management.

**Design Decisions:**  
- Use `PyYAML` to load `config.yaml`.
- Parse nested structures into a dictionary.
- Implement utility functions:
  - `load_config(filepath: str) -> dict`  
    - Reads the YAML file.
  - `get_config_value(config: dict, key_path: str, default=None)`  
    - Access nested parameters (e.g., `config['dataset']['noise_ratio']`) using dot-separated key paths.
- Set default values if certain parameters are missing, ensuring robustness.

**Implementation considerations:**  
- Lazy loading or caching — load config once at startup.
- Validation: ensure that essential keys are present (`dataset`, `model`, `hyperparameters`).

---

## 2. Reproducibility & Random Seed Setting

**Objective:**  
- Guarantee reproducible experiments by fixing randomness across libraries and frameworks.

**Key Libraries:**  
- `torch`: seed via `torch.manual_seed()` for CPU and CUDA.
- `numpy`: seed via `np.random.seed()`.
- Python built-in `random`: seed via `random.seed()`.

**Design of `set_seed(seed: int) -> None`:**  
- Accepts an integer seed (from config or experiment argument).
- Sets:
  - `random.seed(seed)`
  - `np.random.seed(seed)`
  - `torch.manual_seed(seed)`
  - If CUDA is used: `torch.cuda.manual_seed_all(seed)`
- Additional configurations:
  - `torch.backends.cudnn.deterministic = True` (trade-off with performance)
  - `torch.backends.cudnn.benchmark = False`

**Impact:**  
- Ensures deterministic behavior of data shuffling, weight initialization, and other stochastic processes.
- Facilitates reproducibility across experiments.

---

## 3. Resource Logging (Time and Memory)

**Objective:**  
- Track training time per epoch and maximum GPU memory consumption.
- Store logs for analysis, visualization, or debugging.

**Design of resource monitoring functions:**  
- `log_time_start()`, `log_time_end()`:  
  - Record timestamps at the start and end of training/epoch.
- `get_elapsed_time() -> float`:  
  - Calculate elapsed time.
- `log_gpu_memory() -> float` or `max_gpu_memory() -> float`:  
  - Use `torch.cuda.max_memory_allocated()` to get maximum memory during training.
  - Or `torch.cuda.memory_summary()` for detailed info.

**Implementation:**  
- Use `time.time()` or `datetime.datetime.now()` for timestamps.
- Implement handlers or context managers for automatic logging.
- Store logs in a file or in-memory structure for later export.

**Notes:**  
- Timing should measure only training phases, excluding setup.
- Memory logs should be reset at the start of each epoch, and maximum values tracked during the epoch.

---

## 4. Data Augmentation & Preprocessing Utilities

**Objective:**  
- Provide functions to apply standard data transformations consistently across datasets.
- Support dataset-specific augmentations, e.g., random crop, flip for CV datasets.

**Design considerations:**  
- Functions like `get_data_transform(dataset_name: str, phase: str)` returning transformation pipelines compatible with `torchvision.transforms`.
- Support additional transformations if specified (e.g., noise injection, class imbalance handling).

---

## 5. Miscellaneous Utilities

**Logging Setup:**  
- Initialize logging with `logging` module.
- Configure log format, level, and output file if necessary.

**Seed & experiment reproducibility:**  
- Functions to set global seeds upon startup + reproducibility flags.

**Plotting & Visualization:**  
- `plot_metrics(metrics_dict: dict, save_path: str)`  
  - Plot training/validation metrics over epochs.
  - Save plots to disk with clear labels/right format.

---

## 6. Summary of Key Functions & Their Signatures

| Function Name                     | Purpose                                              | Input Parameters                                              | Output/Notes                                              |
|----------------------------------|------------------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------|
| `load_config(filepath: str) -> dict` | Load `config.yaml` configuration into a dictionary     | Path to config file                                         | Dict with nested configurations                            |
| `get_config_value(config: dict, key_path: str, default=None)` | Access nested config param via dot notation             | Config dict, key path (`dataset.noise_ratio`)                | Corresponding value or default                            |
| `set_seed(seed: int) -> None`     | Set random seeds for reproducibility                  | Seed integer                                                | None                                                      |
| `log_time_start() -> float`       | Record start time for an experiment/epoch           | None                                                        | Start timestamp                                              |
| `log_time_end(start_time: float) -> float` | Compute elapsed time since start                    | Start timestamp                                             | Duration in seconds                                         |
| `log_gpu_memory()`               | Log current maximum GPU memory usage during training | None                                                        | Returns current max memory in bytes                         |
| `save_metrics(metrics: dict, filepath: str)` | Save metrics dict into a file (JSON, CSV)          | Metrics dict, file path                                    | None                                                      |
| `plot_metrics(metrics: dict, save_path: str)` | Plot metrics curves over epochs                     | Dict of metrics, path to save figure                     | Generates and saves plots                                   |
| `setup_logger()`                  | Initialize logging configuration                      | Optional: file name, level, format                        | Logging configured                                           |

---

## 7. Additional Notes & Clarifications

- **Dataset annotations:**  
  - For difficulty annotations (e.g., noise labels), these should be generated prior to training or embedded in dataset loaders; utility can support their loading and validation.
- **Resource logging APIs:**  
  - Use `torch.cuda.max_memory_allocated()` and `torch.cuda.reset_max_memory_allocated()` for accurate tracking.
  - Timing functions should use high-resolution timers (`time.perf_counter()`).
- **Hyperparameter access:**  
  - Leverage config parsing functions to ensure hyperparameters are set consistently and passed into other modules.
- **Reproducibility Controls:**  
  - Enforce seed setting at the start of each run.
  - Disable nondeterministic behaviors as feasible.

---

## Final Summary

The `utils.py` module will serve as a foundational component, enabling experiment reproducibility, resource tracking, configuration management, and visualization. It requires careful implementation of functions to support seamless integration with the main training, dataset, model, and curriculum modules. Consistent interface design and extensive use of the configuration file will ensure that all experiments are controlled, comparable, and traceable.

---

This completes the thorough logic analysis for `utils.py`, aligning with the paper, design framework, and experimental protocol requirements.

