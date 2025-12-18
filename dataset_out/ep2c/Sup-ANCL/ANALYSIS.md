# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset.py

# Logic Analysis for `dataset.py`

The purpose of `dataset.py` is to define reusable dataset classes and data loaders that facilitate both synthetic toy data generation and real image datasets (such as ImageNet-100 and downstream datasets). The implementation must support training, validation, and test splits, handle appropriate data augmentation, and interface seamlessly with PyTorch’s `torch.utils.data.Dataset` and `DataLoader`.

---

## Core Components to Implement:

### 1. **ToyDataset Class (Synthetic Data)**
- **Objective:** Generate a synthetic dataset of 3 classes, each following a Gaussian distribution with orthogonal means, shared isotropic covariance.
- **Key Details:**
  - Number of classes: 3
  - Dimensionality: 2048
  - Number of samples:
    - Training: 1000 samples per class (3000 total)
    - Testing: 500 samples per class (1500 total)
  - Means: orthogonal vectors derived from the left singular vectors of a Gaussian matrix.
  - Covariance: identity scaled by 0.35.
  - Data augmentation:
    - For each sample, replace approximately 60% of features with the overall data mean vector (to mimic the described augmentation).
  - Responses:
    - Return data samples (`x`) and associated labels (`y`).

- **Implementation™:**
  - Generate means during dataset initialization, store as class attributes.
  - For each sample:
    - Draw from the class Gaussian distribution.
    - Perform augmentation (feature masking/ replacement as specified).
  - Support indexing (`__getitem__`) returning a tensor and label.
  - Implement a `__len__()` method.

---

### 2. **Image Dataset Class (Using torchvision)**
- **Objective:** Wrap torchvision dataset loading for ImageNet-100 (or other datasets) with train/test splits and proper augmentations.
- **Key Details:**
  - Load datasets via torchvision datasets or custom loader if needed.
  - Support train, val, test splits.
  - Apply well-defined data augmentations:
    - Standard augmentations: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayScale, GaussianBlur.
  - Ensure reproducibility by fixing the random seed.
  - Support optional transforms for each phase.

---

### 3. **Augmentation Handling**
- **For Synthetic Data:**
  - Augmentation: Replace ~60% of features in the vector with the mean vector (or apply a similar masking).
  - Possibly simulate augmentation with a custom function operating on the synthesized features.
- **For Image Data:**
  - Use torchvision.transforms with standard augmentations.
  - Compose transforms conditioned on dataset phase (train/test).

---

### 4. **Dataset Management**
- **Supervision:**
  - Each dataset should return `(x, y)` pairs, where `x` is a tensor (image or vector) and `y` is the label.
- **Splits:**
  - `train_dataset`: Use training data.
  - `val_dataset`: Use validation subset of dataset (if granular split exists).
  - `test_dataset`: Use test subset.
  - Implement proper splitting for validation, either through torchvision datasets split, or custom splitting if needed.

---

### 5. **DataLoader Integration**
- Instantiate PyTorch DataLoader objects for each dataset split, with configurable batch sizes, shuffling, and num_workers.
- For training datasets, shuffling should be enabled.
- For validation/testing, shuffling typically disabled.
- Support setting the seed for deterministic reproducibility.

---

## Implementation Steps:

### A. **Synthetic Toy Dataset (`ToyDataset`)**
- **During `__init__`:**  
  - Generate class means (orthogonal vectors):  
    - Sample a random Gaussian matrix  
    - Perform SVD, extract left singular vectors as class means  
    - Rescale to match covariance specifications (or directly use the scaled identity)
  - Generate samples:
    - For each class:
      - Sample from `N(μ_y, 0.35 * I)`
    - Store all data and labels.
- **Augmentation:**
  - Implement a function to simulate the augmentation:
    - Randomly replace 60% of features with the overall mean (computed during init).
- **`__getitem__`:**
  - Return augmented sample and label.

### B. **Image Dataset Wrapper (`ImageDataset`)**
- **Uses torchvision.datasets.ImageFolder or custom dataset.**
- **Transforms:**
  - Compose data augmentation transformations.
- **Splitting:**
  - Handle validation split, if datasets lack an explicit val set.
- **`__getitem__`:**
  - Return image tensor and label.

---

### C. **Common Interfaces**
- All dataset classes should:
  - Inherit from `torch.utils.data.Dataset`.
  - Define `__len__()` and `__getitem__()`.
- Use consistent data types (e.g., float32 tensors).
- Retain labels as integers.
- Accept configuration parameters:
  - `split` type: train/test/validation.
  - `augmentations`: apply as needed.
  - `dataset_path`: for image datasets.

---

## Additional Considerations
- Ensure deterministic behavior:
  - Use a fixed seed.
  - Consistent shuffling if needed.
- Data normalization:
  - For image datasets: standard ImageNet mean/std normalization.
  - For toy data: normalization may be optional; features are post-augmentation.
- Support DataLoader batching naturally.

---

## Summary

- **`ToyDataset`:**
  - Generate class means with SVD for orthogonal vectors.
  - Sample from Gaussian distributions per class.
  - Implement augmentation by feature masking/replacement.
  - Return `(x, y)` as tensors for each index.

- **`ImageDataset`:**
  - Load datasets via torchvision.
  - Apply augmentations with torchvision.transforms.
  - Support train, val, test splits.
  - Return `(image_tensor, label)`.

- **Pool & DataLoader:**
  - Instantiate DataLoader with batch size, shuffling, seed setting.
  - Maintain flexibility for two datasets (toy and image).

---

This detailed explanation provides a concrete, step-by-step plan for coding `dataset.py` in line with the research paper, plan, and configuration, ensuring compatibility with subsequent modules such as training, loss, and evaluation.

## evaluation.py

# Evaluation.py Logic Analysis

This module is responsible for evaluating the quality of learned representations in two key downstream tasks:
1. Linear probing: assessing linear separability of frozen features.
2. Possibly additional downstream metrics (e.g., object detection), but based on current description, mainly linear evaluation and feature extraction.

The structure primarily involves:
- Loading pretrained models (encoder backbone) with frozen weights.
- Extracting features from datasets.
- Training a linear classifier (e.g., logistic regression) on these features.
- Evaluating classifier accuracy on held-out test/validation data.
- (Optional) computing other downstream metrics; in current scope, focus on accuracy.

---

# Core Components & Implementation Details

### 1. **Class: Evaluation**

**Responsibilities:**
- Initialize with:
  - a model (encoder backbone, pre-trained and frozen).
  - dataset loader for evaluation dataset (test set or validation set).
  - configuration parameters for decide evaluation protocol.
- Methods:
  - **extract_features()**:
    - Pass dataset samples through the encoder (without gradient computation).
    - Output features of dimension given by the projection/output layer.
    - Store these features for downstream training/evaluation.
  - **train_linear_classifier()**:
    - Use sk-learn (scikit-learn) logistic regression or linear classifier.
    - Fit classifier using training features and labels.
  - **evaluate()**:
    - Call extract_features() on test set.
    - Use trained classifier to predict labels.
    - Calculate accuracy metrics.
    - Return metrics (in dict form).
- **Optional:**
  - Support for multiple datasets (for multiple downstream tasks).
  - Visualization or additional metrics (not currently specified).

---

### 2. **Feature Extraction Process**

- For each dataset:
  - Loop over data loader (batch-wise).
  - For each batch:
    - Forward pass using encoder:
      - Ensure `torch.no_grad()` context to prevent gradient tracking.
      - Feed images through the encoder.
    - Extract features:
      - Typically, use the output just before the projection layer; or use the final feature layer.
      - Ensure features are normalized (L2 normalization) to match training loss consistency.
  - Collect all features and labels (for full dataset).

- Store features and labels in numpy arrays for sklearn compatibility.

### 3. **Linear Classifier Training**

- Use `sklearn.linear_model.LogisticRegression`:
  - Parameters:
    - Regularization: default or tune with cross-validation.
    - Solver: ‘lbfgs’ or ‘saga’ for large datasets.
  - Input:
    - Features: shape `(N_samples, feature_dim)`.
    - Labels: shape `(N_samples,)`.
- Fit classifier solely on the training features (frozen encoder).

### 4. **Evaluation & Metrics**

- After training classifier:
  - Predict on test features.
  - Compute accuracy:
    - Use `accuracy_score()` from sklearn.metrics.
  - Return accuracy as main metric.
- For comprehensive report:
  - Could extend to include confusion matrices, class-wise accuracy, etc.

### 5. **Design Specifics**

- Use Python classes.
- Encapsulate the model loading and freezing:
  - Assumes the encoder (backbone) is already pre-trained.
  - During feature extraction:
    - Set model to `eval()` mode.
    - Use `torch.no_grad()` context.
- Handle dataset:
  - Input dataset loader objects, compatible across datasets.
  - Use `torch.utils.data.DataLoader`.
- Device management:
  - Move models and data to device (GPU/CPU).
- Log progress:
  - Optional: use tqdm for progress bar if desired.
  - Log number of batches processed.

### 6. **Dependencies**

- `torch`: for model inference, dataset iteration.
- `sklearn`: for classification and accuracy metrics.
- `numpy`: for array processing.
- `matplotlib`: optional, if visualization incorporated later.

---

# Pseudocode Outline

```python
class Evaluation:
    def __init__(self, model, dataset_loader, device='cuda'):
        self.model = model
        self.dataset_loader = dataset_loader
        self.device = device
        self.features = None
        self.labels = None

    def extract_features(self):
        self.model.eval()
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.dataset_loader:
                images = images.to(self.device)
                # Pass images through encoder
                feats = self.model.encoder(images)
                feats = normalize(feats)  # L2 normalization
                all_features.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())
        self.features = np.concatenate(all_features, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)

    def train_linear_classifier(self):
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.clf.fit(self.features_train, self.labels_train)

    def evaluate(self):
        # Extract test features
        self.extract_features()
        # Predict
        preds = self.clf.predict(self.features_test)
        accuracy = sklearn.metrics.accuracy_score(self.labels_test, preds)
        return {'accuracy': accuracy}

    def run_evaluation(self):
        # Load train/test features
        self.extract_features()  # for train or test as needed
        self.train_linear_classifier()
        metrics = self.evaluate()
        print(f"Linear probing accuracy: {metrics['accuracy']*100:.2f}%")
        return metrics
```

---

### 7. **Summary of Key Implementation Points**
- **Feature normalization**: ensure features are normalized before classifier training.
- **No gradient update during evaluation**.
- **Use correct dataset splits**: load separate datasets for feature extraction and training classifier.
- **Batch handling**:
  - Concatenate features for full dataset.
  - Ensures efficient sklearn training.
- **Device management**:
  - Keep model and data on GPU if available.
- **Extensibility**:
  - Support multiple datasets.
  - Add visualization for feature distributions if needed.

---

# 8. **Remaining Clarifications**
- Confirm that features are extracted from the encoder's final output before any projection, or from the projection layer.
- Confirm if normalization should be applied to features before or after the classifier (typically before).
- Confirm dataset loading details, e.g., normalization transforms, batch sizes for inference.

---

# Conclusion
This detailed logic spec prepares the implementation of `evaluation.py`, focusing on:
- Efficient feature extraction.
- Accurate classifier training.
- Precise metric evaluation.
- Clear interface encapsulation.

It adheres strictly to the described evaluation protocol, ensuring reproducibility and scientific rigor.

## loss.py

# Logic Analysis for `loss.py`

This module is responsible for implementing the core loss functions used in the supervised asymmetric non-contrastive learning (ANCL) framework, specifically for SUPSIAM and SUPBYOL. It must handle multiple aspects: the calculation of self-supervised and supervised losses, their combination with the hyperparameter \(\alpha\), as well as supporting stop-gradient operations, sampling positives from a class-specific pool, and optionally, covariance regularization. 

Below is a detailed, step-by-step analysis of the components and required logic:

---

# 1. **Inputs to Loss Functions**

- **Online projections and predictions:**
  - `online_proj`: The output of the online branch projector, shape: `[batch_size, projection_dim]`.
  - `online_pred`: The output of the predictor network (for Siam/Siam-like methods), shape: `[batch_size, projection_dim]`.

- **Supervised targets:**
  - `supervised_targets`: Features sampled from the class-specific feature pool (queue), shape: `[batch_size, projection_dim]`.
  - The supervised targets are obtained for each anchor by sampling `M` positives from the feature pool associated with the anchor’s class (or all positives).

- **Hyperparameters:**
  - `alpha`: balancing coefficient (float in [0,1]) that adjusts the contribution of supervised vs. self-supervised loss.
  - `temperature`: used in contrastive-like distance normalization; for distance-based losses, might influence scaling or softmax temperature conversions (if applicable).
  - `pool_size`: size of the class-specific feature pools, used for sampling.
  - `sampling_pos`: indicates whether to sample all positives ("all") or `M` positives (integer).

- **Additional optional parameters:**
  - `covariance_regularization`: boolean flag to enable covariance regularization components.

- **Stop-gradient operations:**
  - For features `z`, ensure the correct tensors are processed with or without gradient flow, as per the asymmetric design.

---

# 2. **Key Operations and Logic**

### A. **Normalization of Features**

- **Why:** To ensure features are on a unit hypersphere, facilitating Distance/Similarity calculations.
- **Implementation:**
  - Apply `F.normalize(tensor, p=2, dim=1)` on:
    - `online_proj` (before passing to predictor)
    - `supervised_targets`: features sampled from pool
  - This is crucial for consistency with the described loss formulations.

### B. **Sampling Positive Features from Pool (`supervised_targets`)**

- **Sampling mode options:**
  - `"all"`: use all stored positives for the given class in the pool, i.e., the entire class queue.
  - `M` positives: randomly sample `M` features from the class queue.
  
- **Implementation:**
  - For each sample in batch:
    - Identify class label or auxiliary info.
    - Access corresponding stored positives.
    - Sample `M` features uniformly at random.
    - Average if `z_sup = (1/M) * sum(z')`.
  - This step assumes access to a class-indexed structure or a way to retrieve these features efficiently.

- **Important:**
  - Ensure the sampled features `z_sup` are normalized.
  - Maintain batch alignment: the batch anchors and supervised targets need to be matched.

### C. **Losses Definition**

The total loss \(\ell\) is a convex combination:

\[
\ell = \alpha \cdot \ell_{ssl} + (1 - \alpha) \cdot \ell_{sup}
\]

- **Self-supervised loss (\(\ell_{ssl}\)):**
  - Based on BYOL / SIMSIAM:
  - \( \ell_{ssl} = \|\text{predicted}_1 - \text{stopGradient}(z_2)\|_2^2 \)
  - Both vectors are normalized.
  - `stopGradient(z2)` is achieved with `torch.no_grad()` in PyTorch or `detach()` on the tensor.
  - No symmetry conventions are explicitly needed here, but optionally, the symmetry can be symmetrical, i.e., also compare the other view (as per experiments).

- **Supervised loss (\(\ell_{sup}\)):**
  - \( \ell_{sup} = \|\text{predicted}_1 - \text{stopGradient}(z_{sup})\|_2^2 \)
  - `z_{sup}` is the average of sampled positive features for the class.
  - Stop-gradient on `z_{sup}` to avoid collapse.
  
- **Note:** Both `predicted_1` and `z_2`, `z_{sup}` are L2 normalized to ensure the distance is on the hypersphere.

### D. **Loss Computation**

- Compute both losses:
  - For each batch element:
    - Calculate `ssl_loss_i`.
    - Calculate `sup_loss_i`.
  - Aggregate over the batch:
    - Possibly by mean, sum, or other appropriate schemes.

- Final loss:
  \[
  \text{total_loss} = \alpha \times \text{ssl_loss} + (1-\alpha) \times \text{sup_loss}
  \]

Adjust based on experimental setting, typically averaging over the batch dimension.

### E. **Covariance Regularization (Optional)**

- **Purpose:** To prevent trivial solutions and encourage feature decorrelation, as per the theoretical motivation.
- **Implementation:**
  - Compute feature covariance matrix:
  
    \[
    C = \frac{1}{N} \sum_{i=1}^{N} (z_i - \bar{z})(z_i - \bar{z})^T
    \]
  
    or directly, for features `Z`:
  
    \[
    C = \frac{1}{N} (Z^T Z)
    \]
  
  - Regularize covariance matrix entries:
    - Penalize off-diagonal entries: encourage orthogonality.
    - Penalize deviation of diagonal entries from 1: encourage unit variance.
  - This is optional and should be incorporated if enabled.
  - The covariance regularization loss can be summed with standard loss, scaled by a hyperparameter if needed.

---

# 3. **Implementation Flow and Pseudocode (Conceptual)**

```python
def compute_loss(online_proj, online_pred, class_labels, target_pool, alpha, temperature, sampling_mode, covariance_reg=False):
    # Normalize features
    online_pred_norm = F.normalize(online_pred, p=2, dim=1)
    
    # Sample supervised targets
    supervised_targets = []
    for label in class_labels:
        if sampling_mode == 'all':
            z_samples = target_pool.get_all_positives(label)  # shape: [N_class, projection_dim]
        else:
            z_samples = target_pool.sample_positives(label, M)  # shape: [M, projection_dim]
        z_avg = torch.mean(z_samples, dim=0)  # shape: [projection_dim]
        z_avg = F.normalize(z_avg, p=2, dim=0)
        supervised_targets.append(z_avg)
    supervised_targets = torch.stack(supervised_targets, dim=0)  # shape: [batch_size, projection_dim]
    
    # Compute self-supervised loss
    z2 = stop_gradient(target_proj) # in BYOL, target projection
    ssl_loss = torch.sum((online_pred_norm - stop_gradient(z2))**2, dim=1).mean()
    
    # Compute supervised loss
    z_sup = supervised_targets
    sup_loss = torch.sum((online_pred_norm - stop_gradient(z_sup))**2, dim=1).mean()
    
    # Combine
    total_loss = alpha * ssl_loss + (1 - alpha) * sup_loss
    
    # Optional covariance regularization
    if covariance_reg:
        cov_loss = compute_covariance_loss(online_proj)
        total_loss += cov_lambda * cov_loss

    

    return total_loss
```

*Note:* The above is a conceptual pseudocode highlighting the main steps. Actual code must handle batching, device transfers, and efficient sampling.

---

# 4. **Additional Considerations**

- **Stop-gradient:**
  - Use `z2.detach()` for stop-gradient features.
  - For the pools, ensure the features are stored and retrieved correctly; sampling should be random and uniform unless specified otherwise.
  
- **Feature Pool Updates:**
  - Tasks involve hard updates with an EMA or replacement as per the pool design.
  - During loss computation, features from the pool are fetched and normalized.
  - Consistency between the pool features and the features in the model is essential for the pipeline.

- **Batch Normalization / Activation:**
  - Typically, in self-supervised settings, avoid batch norm in projection/predictor layers or keep consistent with original architecture.

- **Hyperparameters:**
  - Ensure `alpha`, `temperature`, `pool_size`, and `M` are adjustable via configuration.
  - Use fixed seeds for reproducibility if needed.

---

# 5. **Summary**

- The primary task of `loss.py` is to compute a joint loss combining:
  - Self-supervised (e.g., BYOL-type) prediction distance with stop-gradient.
  - Supervised prediction distance with stop-gradient, sampling positives from feature pools.
- Incorporate normalization, stop-gradient, and pooling logic carefully.
- Support optional covariance regularization.
- Return a scalar tensor loss suitable for backpropagation.

This structured analysis guides implementing correct, robust, and efficient loss functions respecting the theoretical and experimental design of the paper.

## main.py

# Logic Analysis for main.py – Central Orchestrator for Data Loading, Model Initialization, Training, and Evaluation

This module serves as the entry point for the entire experimental pipeline, aligning with the paper's methodology, the described architecture, and the configuration settings outlined in `config.yaml`. Its purpose is to initialize all components, coordinate their interactions, and execute training and evaluation workflows systematically.

---

# Step 1: Parse Configuration and Environment Setup

- **Load Configuration:**
  - Read and parse `config.yaml` into a dictionary `config`.
  - Extract key parameters:
    - `training`: learning rate, batch size, epochs, seed.
    - `pretraining`: dataset, optimizer, weight decay, total epochs, base learning rate, scheduler.
    - `model`: backbone type, projection/predictor dimension and layers.
    - `loss`: alpha, temperature, pool size, sampling mode.
    - `pool`: type, update method, size.
    - `evaluation`: task types, metrics.
- **Set Random Seed:**
  - Set torch, numpy, and other libraries' random seeds to ensure reproducibility.

---

# Step 2: Initialize Dataset and DataLoader

- **Dataset Loading:**
  - Use torchvision datasets (e.g., ImageFolder or custom as needed).
  - Apply the standard augmentation pipeline for training:
    - Random crop, horizontal flip, color jitter, grayscale, Gaussian blur.
  - For the toy dataset:
    - Generate synthetic Gaussian data with class means orthogonal, as specified.
    - Implement a custom `ToyDataset` class that creates data on-the-fly or loads precomputed features.
- **DataLoader:**
  - Instantiate DataLoader with the specified batch size.
  - Shuffle training data.
  - Set `seed` for reproducibility.
  - Prepare validation/test loaders if needed for evaluation.

---

# Step 3: Instantiate Models

- **Encoder Backbone:**
  - Based on `backbone` parameter:
    - For ResNet50:
      - Use torchvision models, possibly removing final classification layer.
    - For ViT:
      - Use a suitable implementation, if required.
  - For toy data:
    - Build a simple 1-layer linear encoder (or an identity if focusing solely on features).
- **Projection Head:**
  - Implement an MLP with specified layers and output dim (128 or 256).
  - For SUPSIAM/SUPBYOL:
    - Use 2 or 3 layers in the projection head.
    - Initialize parameters.
- **Predictor:**
  - For SIMSIAM/SUPSIAM:
    - 2-layer MLP with hidden size as per config (4096), no batch norm.
  - For BYOL/SUPBYOL:
    - Same as above, or as specified.
- **Target Network:**
  - **For SUPSIAM:**
    - Share parameters with online network or create an independent copy.
  - **For SUPBYOL:**
    - Create a target network that is an exponential moving average (EMA) of online network parameters.
    - Initialize with online network parameters.
    - Set EMA decay (`momentum`).
- **Feature Normalization:**
  - Ensure features are normalized (L2) after projection.
- **Additional modules:**
  - Implement energy regularizers or covariance regularizers if specified.

---

# Step 4: Instantiate Class-Specific Feature Pools

- **Pool Management:**
  - Instantiate a class `Pool` with:
    - `pool_type`: class-specific queues.
    - `size`: e.g., 8192.
    - `number of classes`: as per dataset.
  - Initialize per-class feature buffers.
  - Set `update_with_ema` as per `config`.
- **Pooling Strategy:**
  - For each class:
    - Maintain a queue (e.g., FIFO).
    - Store features (projected, normalized).
  - For sampling positives:
    - Use `sample_positive(labels, M)` to select class positives (all or subset).

---

# Step 5: Instantiate Loss Function

- **Combine Losses:**
  - Create an object `Loss` configured with:
    - `alpha` to weight supervised vs self-supervised components.
    - `temperature` for distance scaling.
    - Enable covariance regularization if specified.
  - Inside:
    - Define functions for:
      - SSL loss (e.g., similar to BYOL, SimSiam).
      - Supervised loss: sampling from pool, calculating squared distances.
- **Sampling Logic for supervision:**
  - For each batch, for each label:
    - Retrieve features (z) from the pool – using `sample_positive`.
    - Compute mean or subset as supervised target `z_sup`.

---

# Step 6: Instantiate Trainer Object

- **Trainer Class:**
  - Initialize with:
    - Encoder, projection, predictor.
    - Target network (EMA or shared).
    - Pool object.
    - Loss object.
    - DataLoader for training.
    - Hyperparameters: `alpha`, learning rate schedule, total epochs.
- **Optimizer:**
  - Use `torch.optim.SGD` with:
    - Learning rate: `training.learning_rate`.
    - Momentum: 0.9.
    - Weight decay: 1e-4.
  - Optionally, parameter groups: encoder, predictor, projection head.
- **Learning Rate Scheduler:**
  - Cosine scheduler over total epochs.
  - For the predictor: fix learning rate or tune as per setup.

---

# Step 7: Prepare for EMA Updates

- **For SUPBYOL:**
  - Implement EMA update rule:
    ```python
    target_params = target_network.parameters()
    online_params = online_network.parameters()
    for t_param, o_param in zip(target_params, online_params):
        t_param.data = decay * t_param.data + (1 - decay) * o_param.data
    ```
  - Update after each batch or epoch.

---

# Step 8: Main Training Loop

- For each epoch in total epochs:
  - Loop over data batches:
    - For each batch:
      - Generate two augmented views (`x1`, `x2`).
      - Forward pass online branch:
        - Compute features, projection, predictor.
        - Normalize output (`p1`).
      - Forward pass target branch:
        - Using EMA network:
          - Compute features from `x2` (or `x2_sup` for supervised target).
          - Normalize features (`z2` or `z2_sup`).
      - Sample supervised positives:
        - From feature pool for the current labels.
        - Average features if multiple positives are used (parameter `M`).
        - Ensure features are normalized.
      - Compute the combined loss:
        - SSL loss: attraction between `p1` and `z2`.
        - Supervised loss: attraction between `p1` and `z2_sup`.
        - Weight by `alpha`.
      - Backpropagate:
        - Zero gradients.
        - Compute loss.
        - optimizer.step().
      - Pool updates:
        - Add current features (`z2`) and labels to the pool.
        - For class-specific pools, enqueue features at the appropriate class slot.
      - EMA update:
        - Update target network parameters if SUPBYOL.
  - (Optional) Log metrics:
    - Losss, intra-class variance estimates, feature norms.
  - (Optional) Save checkpoints periodically.

---

# Step 9: Post-Training Evaluation and Visualization

- **Linear Probe Evaluation:**
  - Freeze encoder.
  - Train a linear classifier (e.g., logistic regression) on training subset.
  - Measure top-1 accuracy on validation/test set.
- **Few-Shot Evaluation:**
  - Sample episodes.
  - Use frozen encoder.
  - Fit logistic classifier per episode, report accuracy.
- **Feature Visualization:**
  - Extract features from test set.
  - Apply t-SNE visualization.
  - Save plots for intra/inter class variance analysis.
- **Compute metrics:**
  - Intra-class variance \(\tilde{S}_W\), inter-class variance.
  - Collapsing tendencies, class overlaps.

---

# Step 10: Finalize Results and Save Artifacts

- Save the trained model weights, target pools, feature embeddings.
- Save logs, metrics, and visualization figures.
- Optionally, perform hyperparameter sweeping over alpha, pool size, M, learning rate, etc., for analysis.

---

# Additional Considerations and Assumptions:
- **Modular Design:**
  - Ensure each module (dataset, model, loss, pool, train, eval) is independently testable.
- **Default Settings:**
  - Default hyperparameters in `config.yaml` are used unless overridden.
- **Reproducibility:**
  - Fix random seed.
  - Save the configuration and results consistently.
- **Compute Environment:**
  - Use GPU if available.
  - Run multiple seeds for robustness.

---

This detailed logic analysis guides the construction of `main.py`, ensuring steps mirror the paper's methodology, fit the specified configuration, and facilitate accurate reproductions of the experiments.

## model.py

# Logic Analysis for model.py

This module defines the core neural network architectures used in supervised ANCL, including encoders (ResNet-50, ViT, or linear for toy datasets), projection heads (MLPs), predictor networks (MLPs), and the EMA target network for SUPBYOL. The design must be modular, flexible, and adhere strictly to the specifications from the paper and the overall plan.

---

## Primary Components and their Responsibilities

### 1. Encoder (Backbone)
- **Function:** Extract high-level features from input images or features for toy datasets.
- **Supported Architectures:**
  - **ResNet-50:**
    - Use torchvision.models.resnet50, remove final classification layer, output feature vector.
  - **ViT-Small:**
    - If specified, implement via existing vision transformer classes or from a third-party library; ensure it outputs feature vectors.
  - **Linear for toy dataset:**
    - A single linear layer: input raw features (e.g., 2048-D from synthetic data), output feature vector.
- **Implementation Details:**
  - Should be initialized based on configuration parameter `backbone`.
  - Should output feature tensor of shape `(batch_size, feature_dim)`.
  - Feature normalization to be handled downstream or within architecture (e.g., L2 normalization).
- **Dependencies:** torch.nn.Module, torchvision.models, or custom ViT module.

### 2. Projection Head (MLP)
- **Function:** Map encoder features into a latent space suitable for contrastive/predictive learning.
- **Architecture:**
  - **Number of layers:** Either 2 or 3 layers, per config (`projection_layers` for shared code).
  - **Layer sizes:** 
    - Input: feature size from encoder (2048 or 768 for ViT).
    - Hidden: configurable (default 128 for self-supervised; 256 for some SUPBYOL).
  - **Activation:** ReLU after hidden layers.
  - **Output:** of dimension `projection_dim` (from config).
  - **Normalization:** L2 normalization (applied externally or inside the network’s forward pass).
- **Implementation detail:** 
  - Make it flexible for different input/output sizes.
  - Consistent with paper’s design.

### 3. Predictor (MLP, optional for SIMSIAM / SUPSIAM)
- **Function:** Additional MLP following the projection head to prevent collapse.
- **Architecture:**
  - 2 layers, hidden size (e.g., 4096).
  - Activation functions: ReLU.
  - No batch normalization unless explicitly required.
- **Constraints:**
  - Only instantiated in models that employ predictor (SIMSIAM/SUPSIAM).
- **Implementation detail:**
  - Universal class but conditionally used.
  - The predictor should be initialized with the specified dimensions.

### 4. EMA Target Network (for SUPBYOL)
- **Function:** Maintain a separate network with exponential moving average parameters for the target branch.
- **Implementation details:**
  - Clone the encoder and projection head (if needed).
  - At each training step, update parameters:
    - `target_params = m * target_params + (1 - m) * online_params`, where `m` is EMA (initial 0.99 → 1.0).
  - Support shared parameters (tie weights) or EMA updating.
- **Checkpoint:**
  - Should support loading initial parameters from online network.
  - Should be lightweight and efficiently update.

### 5. Normalize Features
- **Function:** L2 normalization is crucial for loss stability due to similarity metrics.
- **Implementation:**
  - Either as part of each network's forward pass or externally in training code.
  - Ensure features are normalized before loss calculation and sampling positives.

---

## Architecture Class Design

### Class: `Encoder`
- **Constructor:**
  - Accepts `arch` ('ResNet50', 'ViT', 'Linear').
  - Loads corresponding model.
  - For `Linear`: initialize with input features (e.g., 2048 or 768).
- **Forward:**
  - Input: raw images or features.
  - Output: feature vector.
  - Optional: feature norm.

### Class: `ProjectionHead`
- **Constructor:**
  - Input dimension (from encoder output).
  - Layers: 2 or 3, sizes: e.g., 2048->128->128 or as specified.
- **Forward:**
  - Output normalized feature vector of shape `(batch_size, projection_dim)`.

### Class: `Predictor`
- **Constructor:**
  - Input: projection_dim.
  - Layers: 2 layers, hidden size (could be 4096).
- **Forward:**
  - Output normalized predicted vector.

### Class: `EMAEncoder`
- **Constructor:**
  - Clone `Encoder` and `ProjectionHead`.
  - EMA factor `m`.
- **Methods:**
  - `update()` for EMA update.
  - `load_state_dict()` from online network.
  - `forward()` same as encoder.
  
### Combined Architecture Class (e.g., `SimSiamModel`, `SupSiamModel`)
- **Contain:**
  - `online_encoder` + `projection_head` + `predictor`
  - `target_encoder` + `target_projection_head` (EMA version)
- **Functionality:**
  - Forward for online branch: pass input, output features, apply predictor.
  - Forward for target branch: pass input, output features.
- **Note:**
  - For supervised variants, the backbone is shared, or with an independent target network for EMA.

---

## Operational Logic and Design Constraints

- **Parameter Sharing:**
  - Online encoder and target encoder can share weights in SUPSIAM; or target is EMA.
- **Normalization:**
  - Features and projections are L2-normalized before distance calculations.
- **EMA updates:**
  - Maintain exact EMA parameter update routine.
- **Flexibility:**
  - The class design should allow easy switching among:
    - Backbone architectures (ResNet, ViT, linear).
    - With/without predictor.
    - With/without EMA target network.
  - Modular to support experimental variations.

---

## Additional Considerations
- **GPU Compatibility:**
  - Model parameters on GPU; EMA targets must be moved accordingly.
- **Weight Initialization:**
  - Proper initialization for non-pretrained models.
- **Saving/Loading:**
  - Save all components’ state_dicts for reproducibility.
- **Extensibility:**
  - Make it easy to add new backbone architectures or alteration of predictor size.

---

## Summary of Implementation steps in `model.py`:

1. Define backbone encoder class supporting ResNet, ViT, or linear.
2. Define projection head class with flexible depth.
3. Define predictor class.
4. Define `EMAEncoder` class that maintains a copy of online encoder with EMA weights.
5. The main model class brings together online encoder + projection + predictor, and optionally the target encoder.
6. Provide methods for:
   - Forward pass through online branch.
   - Forward pass through target branch.
   - EMA weight update routine.
7. Ensure all features are normalized (L2) before loss calculation.
8. Confirm compatibility with the rest of the codebase, especially for pooled features and supervised targets.

---

This thorough analysis provides a detailed blueprint for implementing the architecture module (`model.py`), aligning with the paper's description and the overall experimental design.

## pool.py

# Logic Analysis for pool.py

This module implements the core data structure for class-based feature pools (queues) used in supervised asymmetric non-contrastive learning (ANCL). Its primary responsibilities are managing per-class feature buffers, enabling efficient positive sampling, and supporting optional updates via exponential moving average (EMA) for stabilized feature representations. The key functionalities include enqueueing features, sampling positive features for supervised loss, and managing pool sizes to prevent memory overflow.

---

## 1. Purpose and Design Goals

- **Manage class-specific feature pools**: Store features associated with each class label, supporting balanced sampling.
- **Support large pool sizes**: The total memory footprint should be manageable, with configurable total pool size (`pool_size`) distributed across classes.
- **Sampling positives**:
  - Either sample *all* positives per class (“all positives” setting).
  - Or sample a fixed number `M` (hyperparameter) positives per class.
- **Pool updates**:
  - When features are produced during training, enqueue/update them in their respective class-specific pools.
  - Optionally, for SUPBYOL, update features via EMA (exponential weighted moving average).
- **Efficiency**:
  - Use numpy arrays or torch tensors for storage.
  - Ensure enqueue and sampling operations are efficient.
- **Flexibility**:
  - Support different pool management strategies: class-specific queues with fixed size, or potentially prototypes (not required here).
- **Reproducibility**:
  - Develop deterministic updates where possible (e.g., random sampling with fixed seed).

---

## 2. Core Data Structures

- **Per-class feature buffer**: 
  - Implement as a dictionary `class_feature_buffers`:
    - Key: class label (int)
    - Value: a fixed-length deque or tensor buffer with `maxlen`=per-class buffer size, for storing features.
  - Alternatively, initialize as a list of numpy arrays or torch tensors with shape `(buffer_size_per_class, feature_dim)` to optimize batch operations.
- **Class sample counts**:
  - Keep track of the number of features for each class accordingly.
- **Global parameters**:
  - Total pool size, number of classes (`num_classes`), per-class buffer size.
  
---

## 3. Initialization and Configuration

- Initialize buffers:
  - Determine per-class buffer size:
    \[
    buffer\_size\_per\_class = \left\lfloor \frac{pool\_size}{num\_classes} \right\rfloor
    \]
  - For simplicity, assign same size to all classes; last class may get remaining slots if `pool_size` isn't divisible.
  - Alternatively, use a fixed number per class, with total size boundary checked.
- Choice of data container:
  - Use `torch.Tensor` buffer initialized with zeros.
  - Or pre-allocate numpy arrays for each class.

---

## 4. Enqueue Functionality

- **Input**:
  - `features`: tensor of shape `(batch_size, feature_dim)` for features to enqueue.
  - `labels`: tensor of shape `(batch_size,)`, class labels corresponding to each feature.
- **Operations**:
  - For each sample:
    - Identify class label `y`.
    - Append or overwrite a feature in the buffer of class `y`.
    - When buffer is full, implement a FIFO replacement policy (if using deques) or cyclic overwrite.
  - **Optional EMA update**:
    - When `update_with_ema` is true:
      - Maintain current feature buffer as an exponential moving average of incoming features.
      - Update feature as:
        \[
        z_{buffer}^{new} = m \times z_{buffer}^{old} + (1 - m) \times z_{current}
        \]
      - Clamp or normalize as needed.
- **Implementation detail**:
  - Use in-place operations for efficiency.
  - Maintain a pointer or index for each class to identify where to insert next (cyclic overwrite).

---

## 5. Sampling Functionality

- **Input**:
  - `labels`: tensor (batch labels) for each query feature (anchor).
  - `M`: number of positives to sample per class.
- **Operation**:
  - For each label in batch:
    - Retrieve the corresponding class buffer.
    - If `sampling_pos` is `"all"`, select all stored features.
    - Else, randomly sample `M` features from class buffer.
  - Stack sampled features into a tensor of shape `(batch_size, M, feature_dim)` (or `(batch_size, feature_dim)` if `M=1`).
  - Ensure the features are L2 normalized — normalization applied after sampling.
- **Handling empty buffers**:
  - When a class buffer is empty, handle gracefully:
    - Sample with replacement (or ignore with warning).
    - Ensure during early training, enough features are accumulated before sampling.

---

## 6. Memory Management and Edge Cases

- When the number of stored features exceeds `maxlen`, remove oldest entries (FIFO).
- For classes with fewer than `M` features:
  - Sample with replacement.
  - Or, if in 'all' mode, return all available features.
- Manage consistent random seed for sampling, to ensure reproducibility if needed.

---

## 7. Additional Considerations

- **EMA update**:
  - For SUPBYOL, features stored in the buffer are updated via EMA:
    \[
    z^{new} = m \times z^{old} + (1 - m) \times z^{current}
    \]
  - Normalize afterward.
- **Normalization**:
  - Store features as normalized vectors for distance calculations in loss.
  - Can normalize upon enqueue or sampling.
- **Thread safety / concurrency**:
  - If multi-threaded data loading, use locks or ensure single-thread updates.
- **Device handling**:
  - Support storing buffers either as numpy arrays or torch tensors on CPU or GPU.
- **Logging and debugging**:
  - Provide functions to inspect buffer sizes, features per class.
  - Log age/updates for debugging.

---

## 8. Summary of Functions

| Function Name | Inputs | Outputs | Description |
|----------------|---------|---------|--------------|
| `__init__(size: int, num_classes: int, feature_dim: int, update_with_ema: bool, ema_m: float)` | configuration params | none | Initialize class buffers, pointers, and parameters. |
| `enqueue(features: Tensor, labels: Tensor) -> None` | features `(batch_size, feature_dim)`, labels `(batch_size,)` | None | Insert features into class buffers, update with EMA if enabled. |
| `sample_positive(labels: Tensor, M: int) -> Tensor` | labels `(batch_size,)`, M | `(batch_size, M, feature_dim)` | For each label, sample M features from corresponding class buffer. |
| `get_buffer_for_class(y: int) -> Tensor` | class label | tensor of shape `(current_size, feature_dim)` | Return current features stored. |
| `update_buffer_class(y: int, feature: Tensor) -> None` | class label, feature vector | None | Update buffer for class y (enqueue, possibly with EMA). |

---

## 9. Final Notes

- Ensure code is modular, allowing toggling between different pool strategies.
- Use clear and consistent normalization.
- Document assumptions about buffer sizes and sampling behavior.
- For extensive datasets or class imbalance, extend design to accommodate variable class buffer sizes distribution.

---

This thorough analysis provides a structured blueprint for implementing `pool.py`. The key points include data structure design, enqueue/sampling operations, EMA updates, normalization, efficiency considerations, and edge case handling, all aligned with the paper's methodology and your system design.

## trainer.py

# Logic Analysis for `trainer.py`

This module implements the core training loop for the supervised asymmetric non-contrastive learning (ANCL) framework, supporting both experiments on synthetic toy data and large-scale image datasets like ImageNet-100. The primary class is `Trainer`, which manages per-epoch training, including data batching, data augmentation, network forward passes, loss computation, optimizer steps, update of the target pool, and optional EMA updates. The design ensures fidelity to the methodology described in the paper, especially regarding the combination of supervised and self-supervised losses, the use of stop-gradient operations, and the maintenance of a class-specific target pool for positive samples.

---

## 1. Initialization

**Inputs:**
- `model`: encapsulates online encoder, projector, predictor, and the target encoder (with EMA update mechanism).
- `pool`: manages the class-specific feature queues, supports sampling positives.
- `loss_fn`: computes the combined loss, encapsulating `alpha`, temperature, and potentially covariance regularization.
- `dataloader`: provides batches with augmented views.
- `config`: contains hyperparameters like learning rate, batch size, `alpha`, EMA momentum, etc.

**Setup Tasks:**
- Assign model components to training mode (`model.train()`).
- Instantiate optimizer (according to `config`) with appropriate parameters.
- Establish EMA update mechanism for target network if in support (e.g., SUPBYOL).
- Initialize variables for tracking metrics (loss, intra-class variance, etc.).
- Seed reproducibility if required.

---

## 2. Per-Batch Processing

Each iteration handles one batch of data, with key steps as follows:

**a. Load and Augment Data:**
- Extract batch data: original inputs and labels.
- Apply data augmentation to generate two views per sample:
  - **View1**: input1, processed through online encoder.
  - **View2**: input2, processed through target encoder.
- Use torchvision transformations (from `augmentation` config): random crop, flip, jitter, Gaussian blur, etc.
- For synthetic toy data: use custom augmentation (dimension masking, mean vector replacement).

**b. Forward Pass through Online Branch:**
- Pass the first view through online encoder (`f`).
- Pass through the projector (`g`) to obtain `z1`.
- Pass `z1` through predictor (`h`) to produce `p1`.
- Normalize `p1` (L2 normalization).

**c. Forward Pass through Target Branch:**
- Pass second view through the EMA target encoder (`\(\tilde{f}\)`).
- Pass the output through target projector (`\(\tilde{g}\)`).
- For supervised positives:
  - Sample class-specific positives from the pool (using their labels).
  - Average `M` positives for `z2_sup`.
- For self-supervised positives:
  - Use `z2` directly from the target encoder output (with stop-gradient as per paper).
- Normalize the features: `z2`, `z2_sup`.

**d. Loss Computation:**
- Compute the self-supervised loss component (`\(\ell_{ssl}\)`):
  - Distance between `p1` and `z2` (from augmented view 2), with stop-gradient on `z2`.
- Compute the supervised loss component (`\(\ell_{sup}\)`):
  - Distance between `p1` and `z2_sup` (averaged positives), with stop-gradient.
- Combine losses via `\(\alpha\)`:
  \[
  \ell = \alpha \ell_{ssl} + (1 - \alpha) \ell_{sup}
  \]
- If covariance regularization is implemented, incorporate as additional regularizer (optional).

**e. Backpropagation:**
- Zero optimizer gradients.
- Backward pass on total `\(\ell\)`.
- Clip gradients if necessary.
- Optimizer step to update online encoder, projector, predictor.

**f. Update Target Pool:**
- Extract features `z2` (from target encoder, normalized).
- Store in the class-specific queues along with labels.
- For large datasets: maintain class-balanced queues.
- When using `enqueue`, replace oldest features or sample positives as per pool update strategy.

**g. Update Target Encoder with EMA (for SUPBYOL):**
- For each target parameter:
  \[
  \tilde{\theta} \leftarrow m \tilde{\theta} + (1 - m) \theta
  \]
- Update parameters with momentum `m` (from `config`) after each batch.

---

## 3. Metrics and Logging
- Accumulate per-batch loss and intra-class variance metrics.
- Measure and log:
  - Total loss.
  - SSL component loss.
  - Supervised component loss.
  - Intra-class variance estimates (`\(\tilde{S}_W\)`).
  - Inter-class variance (`\(\tilde{S}_B\)`).
- Optional: visualize feature distributions (e.g., t-SNE periodic check).

---

## 4. End of Epoch
- Aggregate metrics over all batches.
- Record metrics for analysis.
- Save model checkpoints if performance improves.
- Adjust learning rate schedule if using cosine scheduler.

---

## 5. Teacher/Target Network EMA
- After each batch:
  - Update the target encoder parameters using EMA.
  - Implementation:
    ```python
    for param, target_param in zip(model.encoder.parameters(), model.target_encoder.parameters()):
        target_param.data.mul_(momentum).add_(param.data, alpha=1 - momentum)
    ```
- Ensure no gradient propagation through the target encoder during loss computation.

---

## 6. Key Implementation Details

- **Stop-gradient operations:** Use `torch.no_grad()` when passing features to loss functions involving `z2 *` targets to prevent gradient backpropagation into the target network.
- **Pooling logic:**
  - Implement class-buffers as per your class specific queue design.
  - Handle missing class data gracefully if class pools are empty, e.g., default to using available positives or skip.
- **Hyperparameters:**
  - Use `alpha` (from config; e.g., 0.5).
  - Use temperature `τ` (e.g., 0.1).
  - Pool size: 8192.
  - EMA momentum: initial 0.99, increase to 1.0.
  - Learning rate: from config, with cosine decay schedule.
- **Reproducibility:**
  - Set random seed (`torch.manual_seed`) for deterministic behavior.
  - Log random seed for experiments.

---

## 7. Additional Considerations
- For synthetic toy data, ensure data generation pipeline aligns with assumptions made in the paper.
- For large datasets like ImageNet, consider efficient data loaders and memory management.
- Implement model saving/loading if manipulating trained checkpoints.
- For visualization tools and debugging, embed plotting routines for t-SNE or intra-class variance traces.

---

## Final notes:
- Ensure all operations respect `\(\text{stop-gradient}\)` as per the paper's asymmetry.
- Validate the target pool update consistency and proper sampling.
- Keep the code modular for easy experimentation with `alpha`, pool size, and EMA.
- Follow the design constraints: no code multipliers outside methods, encapsulate functionalities cleanly, and document hooks for hyperparameter tuning.

This thorough logic analysis aims to specify the core operations, flow, and implementation nuances necessary for faithfully reproducing the experimental framework outlined in the paper.

