# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

**Logic Analysis for `dataset_loader.py` — Class `DatasetLoader`**

---

### **Purpose & Responsibility:**

The `DatasetLoader` class is tasked with:
- Loading CIFAR-10 or CIFAR-100 datasets based on configuration.
- Applying specified data augmentations consistent with the experimental setup.
- Wrapping datasets into PyTorch `DataLoader` objects configured for training and evaluation.
- Managing batching, shuffling, and data iteration during training.

---

### **Inputs & Parameters:**

- `dataset_name` (str): `"CIFAR-10"` or `"CIFAR-100"` (from `dataset.name`).
- `batch_size` (int): Number of samples per batch (`training.batch_size`).
- `augmentations` (list): List of augmentation transforms to be applied, e.g.,
  ```python
  [
      {'RandomCrop': [32, 32, 4]}, 
      {'HorizontalFlip': True},
      {'ColorJitter': [0.4, 0.4, 0.4, 0.1]},
      {'GaussianBlur': 0.5}
  ]
  ```
- `is_train` (bool): Flag indicating whether loading training or testing data (to determine augmentation and shuffling).

---

### **Outputs:**

- PyTorch `DataLoader` object for training or testing, ready for iteration during training/evaluation.

---

### **Core Logic & Steps:**

#### **1. Dataset Selection & Loading:**
- Use `torchvision.datasets.CIFAR10` or `CIFAR100` based on `dataset_name`.
- Confirm the dataset path and download if not already present.
- Depending on the `is_train` flag:
  - For training data:
    - Load full training split (`train=True`).
  - For evaluation:
    - Load test split (`train=False`).

#### **2. Data Augmentation & Preprocessing:**
- Compose a series of transformations, always including:
  - ToTensor: convert images to tensors.
  - Normalize: standard CIFAR normalization (`mean` and `std` for CIFAR-10 or CIFAR-100).
- Append additional transformations based on `augmentations` list:
  - For `RandomCrop`: apply `transforms.RandomCrop(size=32, padding=4)`.
  - For `HorizontalFlip`: apply `transforms.RandomHorizontalFlip()`.
  - For `ColorJitter`: apply `transforms.ColorJitter()` with specified parameters.
  - For `GaussianBlur`: implement a custom Gaussian blur transform or mock-up if not available.
- Compose the transformations into a single `transforms.Compose()` pipeline.

#### **3. Dataset Wrapping:**
- Instantiate the dataset (`CIFAR10` or `CIFAR100`) with the composed transforms.
- If data augmentation involves multiple views (e.g., for contrastive learning), ensure that the dataset returns either:
  - Paired augmented images, OR
  - A single image and the augmentation pipeline is applied twice during data loader batch collation.
- For the current class, likely return a single image; augmentation will be applied during data loading.

#### **4. Data Loader Configuration:**
- Use `torch.utils.data.DataLoader`.
- Set:
  - `batch_size` to input parameter.
  - `shuffle=True` during training; `False` during evaluation.
  - `num_workers`: typically, 2-4 for CIFAR (or more depending on hardware).
  - `drop_last`: True if consistent batch size is critical.
  - `pin_memory=True`: recommended for CUDA training.

#### **5. Return the DataLoader:**
- Return the configured DataLoader object ready for training or validation loop.

---

### **Additional Considerations:**

- Verify augmentation compatibility:
  - Custom `GaussianBlur` may need a helper class.
  - Data augmentations should match those in the experimental setup for reproducibility.
- Consistency: ensure that the preprocessing (normalization means and stds) matches the paper settings, e.g., CIFAR normalization (`mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`).
- Shuffling:
  - Enabled for training to ensure randomness.
  - Disabled for evaluation to keep consistency.

### **Edge Cases & Robustness:**
- Handle unrecognized augmentation names gracefully, possibly with warnings.
- Confirm dataset download and caching behavior.
- Validate augmentation parameters for validity/consistency (e.g., no negative crop size).
- Modularize: allow for future extension if additional augmentations are added.

---

### **Summary of Key Logic Flow:**

```
- Input parameters: dataset_name, batch_size, augmentations, is_train
- Determine dataset class (CIFAR10 or CIFAR100)
- Compose transformation pipeline:
    - Always include: ToTensor, Normalize
    - Append augmentations based on configuration
- Instantiate dataset with transformation
- Create DataLoader with batch size, shuffle (True for train), num_workers, pin_memory
- Return DataLoader
```

---

### **Final Notes:**

This class acts as a backbone for data feeding, so its design must be:
- Clear & flexible for different augmentations.
- Reproducible with consistent normalization.
- Optimized with batch size and number of workers matching hardware constraints.

It forms the foundation for the training loop to sample representations, which are then fed into the uniformity loss computation.

---

This completes the detailed logic analysis for implementing `dataset_loader.py`.

## evaluation.py

# Evaluation.py Logic Analysis

This file defines the `Evaluation` class responsible for assessing learned representations via multiple downstream and analytical procedures, based on methods described in the paper. The class provides interfaces for evaluating linear classification accuracy, analyzing the singular value spectrum of features, and visualizing feature distributions, both for qualitative and quantitative understanding of uniformity and dimensional collapse. The detailed logic below aims to ensure fidelity to the paper, clarity, efficiency, and modularity.

---

## 1. **Class Structure and Initialization**

**Purpose:**
- Store the trained SSL model.
- Keep references to dataset statistics if needed.
- Set parameters for evaluation methods, such as whether to normalize features, number of classes for downstream, and plotting configurations.

**Key variables:**
- `model`: The trained `Model` instance, with neural networks and projection (and predictor for BYOL).
- `device`: To run tensors on GPU or CPU.
- `downstream_epochs`: Number of epochs to configure for linear evaluation (e.g., 100).
- `evaluation_split`: "test" as per config; used for evaluation datasets.
- `logger`: Optional, for logging metrics.
- `save_dir`: Path to save visualizations and spectra.

---

## 2. **Method: evaluate_downstream**

**Objective:**
- Perform linear evaluation: freeze the backbone encoder.
- Train a linear classifier (fully connected layer) on the frozen features.
- Calculate accuracy metrics (Top-1, Top-5).
- Dataset split: use test set for evaluation.
- Protobuf-like step:
  - For each batch:
    - Extract features using `model.extract_features()`.
    - Optional normalization to unit sphere (if consistent with training).
    - Pass features through a simple linear classifier.
    - Track predictions and compute accuracy over entire set.
  - After training classifier, evaluate on entire test set.
- Return accuracy metrics as dictionary.

**Implementation specifics:**
- Use sklearn's `LinearClassifier` or manually implement a linear model in PyTorch.
- Use a fixed, small learning rate (e.g., 0.1) and number of epochs (`downstream_epochs`).
- Batch features predict labels; compute confusion matrix; derive accuracy.
- Can use `torchmetrics` or custom code; consistent with codebase.

---

## 3. **Method: compute_spectrum**

**Objective:**
- Given a feature tensor (batch or entire dataset):
  - Compute the covariance matrix of features.
  - Perform eigen-decomposition (via `torch.linalg.eigh` in PyTorch).
  - Extract singular values (square roots of eigenvalues).
  - Logarithmically plot or analyze spectrum.
- Purpose:
  - Assess the degree of dimensional collapse.
  - Compare singular value distributions to evaluate how well the features utilize the ambient space.
- Visualization:
  - Plot the spectrum in log scale.
  - Save plot if specified or visualize inline.

**Implementation specifics:**
- Input: features tensor shape `(N, m)` after normalization.
- Output: list or array of singular values.
- Optional: write helper to plot spectrum, optionally in a separate figure.

---

## 4. **Method: visualize_distribution**

**Objective:**
- For qualitative intuition, visualize the embedding distribution.
- Types:
  - 2D scatter plots for pairs of features.
  - Density contours or heatmaps over the feature space.
- Procedure:
  - Select a subset of features (or all features if 2D).
  - For high dimension, reduce via PCA if needed.
  - Plot points on scatter plot.
  - Color-code points if labels known (not applicable here) or use density.

**Implementation:**
- Generate 2D projections.
- Use `matplotlib.pyplot.scatter`.
- Save figures to `save_dir`.
- Label axes (feature dimensions), format plots clearly.

---

## 5. **Additional Analysis Procedures**

- Can implement auxiliary functions:
  - `compute_uniformity_metric()`:
    - Use the current representation features to compute the Wasserstein distance (or KL divergence) as per the paper.
    - Useful for tracking the evolution of uniformity during training.
  - `visualize_feature_overlap()`:
    - Overlap distributions of \(Y_i\) and \(\hat{Y}_i\) as in appendix figures.
    - Generate 1D and 2D distribution plots for selected features to compare distributions.

---

## 6. **General Implementation Considerations**

- **Data Pipeline:**
  - Features are extracted via `model.extract_features()` after a forward pass.
  - Ensure features are normalized before analysis if consistent with training (e.g., \(\ell_2\) normalization).
- **Computational Efficiency:**
  - Perform spectrum calculations over a subset or the entire dataset if feasible.
  - Use batchwise covariance computation to handle large data.
  - Eigen-decomposition can be costly; consider SVD if needed.
- **Visualization:**
  - Save figures periodically.
  - Clear plotting states to prevent memory leaks.
- **Logging & Results Storage:**
  - Save accuracy metrics, uniformity scores, and spectrum plots.
  - Optionally, log epoch-wise metrics for convergence analysis.

---

## 7. **Usage Flow in Main Script**

**Typical sequence:**
- After training concludes:
  - Instantiate `Evaluation`.
  - Call ` evaluate_downstream()` for linear classification.
  - Call `compute_spectrum()` for spectral analysis.
  - Call `visualize_distribution()` for distribution visualization.
- Save or display results as needed.
- Use results for comparing different models or training configurations.

---

## 8. **Potential Customizations or Remarks**

- For large datasets, consider subsampling for spectrum and visualization.
- If multi-GPU, gather features via distributed pooling.
- Normalize features using the same scheme as during training.
- Adjust visualization parameters (bin size, range) as needed to replicate figures.

---

## 9. **Uncertainties & Clarifications**

- Confirm whether features should be normalized (`\ell_2`) before spectral and distribution analyses—assumed yes based on the paper.
- Clarify whether to evaluate on the same features used for training or add further data augmentation.
- Verify eigen decomposition method; full or partial eigen decomposition could be used.
- Whether to perform spectral analysis on a subset or the entire saved dataset.

---

# Summary
The `Evaluation` class comprises methods for:
- Running a linear classifier to assess downstream accuracy.
- Computing the spectrum via covariance eigen-decomposition to evaluate dimensional collapse.
- Visualizing feature distributions for qualitative insights.
- Tracking uniformity metrics over training epochs for ongoing monitoring.

This design aligns with the experimental and analytical procedures described in the paper, emphasizing the core metrics, visualization, and spectral analysis as tools to assess uniformity and collapse phenomena, ultimately supporting the paper's claims and analyses.

This detailed logic informs subsequent implementation, ensuring that every step faithfully reflects the methodology and analysis standards outlined in the original work.

## losses.py

{
  "losses.py": [
    "Class: UniformityLoss\n  Purpose: Responsible for computing the uniformity metrics based on Wasserstein distance (-\\mathcal{W}_2) and for combining this with base SSL losses such as InfoNCE, MSE, or covariance-based losses.\n\nOperational Steps:\n\n1. Initialization:\n   - Accept parameters: loss_type (e.g., 'InfoNCE', 'MSE', 'Covariance'), uniformity_lambda (weight for the uniformity loss), tau (temperature for contrastive loss if applicable), and any hyperparameters needed.\n\n2. Feature Extraction and Preprocessing:\n   - Input: Batch of features Z (Tensor, shape [batch_size, feature_dim]), typically output from the model's extract_features() method.\n   - Normalization:\n     - Decide whether to normalize features to the unit sphere; based on the paper, features should be normalized to approximate uniform spherical distribution. Implement z = F.normalize(z, p=2, dim=1).\n\n3. Computing Means and Covariance:\n   - For each batch:\n     - Compute empirical mean: \\(\hat{\\mu} = \\text{mean}(Z, dim=0)\\).\n     - Center features: Z_centered = Z - \\hat{\\mu}.\n     - Compute covariance matrix: \\(\\hat{\\Sigma} = \\frac{1}{n-1} Z_{centered}^T Z_{centered}\\).\n     - For numerical stability, may add small epsilon to diagonals if needed, but generally not required for the empirical covariance.\n\n4. Computation of the \\( -\\mathcal{W}_2 \\) Uniformity Metric:\n   - Eigen-decompose \\(\\hat{\\Sigma}\\) (e.g., eigenvalues and eigenvectors):\n     - eigenvalues \\(\\{\\lambda_i\\}\\), with eigenvalues >= 0.\n     - Compute trace: \\(\\operatorname{tr}(\\hat{\\Sigma}) = \\sum_i \\lambda_i\\)\n     - Compute square root of covariance: eigenvalues: take square roots \\(\\sqrt{\\lambda_i}\\) for all eigenvalues.\n       - Reconstruct \\(\\hat{\\Sigma}^{1/2} = V \\operatorname{diag}(\\sqrt{\\lambda_i}) V^T\\) (V eigenvectors).\n       - Compute trace of the square root: sum of the square roots of the eigenvalues.\n\n   - Calculate the Wasserstein distance:\n     \n     \\[\n     \\text{W}_{2} = \\sqrt{ \\| \\hat{\\mu} \\|_2^2 + 1 + \\operatorname{tr}(\\hat{\\Sigma}) - \\frac{2}{\\sqrt{m}} \\operatorname{tr}(\\hat{\\Sigma}^{1/2}) } \\]\n\n   - Negative uniformity loss: \\( - \\mathcal{W}_2 = - \\text{W}_{2} \\).\n\n   - Implementation notes:\n     - For eigen-decomposition, use torch.linalg.eigh for symmetric matrices.\n     - Ensure eigenvalues are non-negative; clamp if necessary.\n     - Handle potential numerical instability when eigenvalues are very small or negative due to floating-point errors.\n\n5. Choice of Loss Type:\n   - 'InfoNCE':\n     - Standard contrastive loss over positive/negative pairs (use temperature tau).\n     - Input: features from two augmented views.\n     - Compute similarity matrix, apply cross-entropy loss.\n   - 'MSE':\n     - Mean squared error between normalized features of two views.\n   - 'Covariance':\n     - Use decorrelation losses such as in Barlow Twins or VICReg.\n     - Possibly add cross-correlation loss; here, treat as a separate loss component.\n\n6. Total Loss Computation:\n   - Compute base SSL loss according to \"loss_type\" on the provided features.\n   - Compute uniformity loss via Wasserstein distance as above.\n   - Total loss: sum base loss + (uniformity_lambda * (-\\mathcal{W}_2)).\n\n7. Return:\n   - The combined total loss (Tensor) for backpropagation.\n   - Optional: Return individual components for logging (e.g., base_loss, uniformity_loss, total_loss).\n\n8. Additional implementation notes:\n   - For the eigen-decomposition, batch processing can be done batch-wise if features are large, but typically eigen-decomposition is per batch.\n   - To improve stability: eigenvalues should be clamped above zero.\n   - To make it efficient: cache eigenvalues, avoid recomputing if features do not change substantially.\n   - Keep the code compatible with GPU tensors.\n\nSummary:\n- This class centralizes the uniformity metric, ensuring it is easily integrated into training.\n- It provides the flexibility to switch between different base SSL losses.\n- It allows easy adjustment of the loss weight for flexible training schedules.\n- The core calculation leverages the spectral decomposition of the covariance matrix to compute \\( \\operatorname{tr}(\\hat{\\Sigma}^{1/2}) \\), crucial for the Wasserstein distance.\n- Finally, during training, invoke this class to compute uniformity component and sum into the total loss."
  ]
}

## main.py

# Main.py Logic Analysis

This script serves as the entry point for the experiment pipeline, orchestrating data loading, model initialization, loss setup, training, evaluation, and visualization based on provided configurations. The goal is to create a modular, flexible, and reproducible workflow faithfully aligned with the paper's methodology, experimental setup, and hyperparameters.

---

## 1. **Configuration Parsing & Setup**

- Read and load the configuration from the YAML file (`config.yaml`).  
- Properly parse all sections:
  - `training`: learning rate, batch size, epochs, optimizer, momentum, weight decay, warmup epochs, uniformity loss weight (\(\lambda\))
  - `model`: backbone architecture (ResNet-18), projection dimensions, predictor usage, MLP use
  - `dataset`: dataset name (`CIFAR-10`/`CIFAR-100`), augmentation list
  - `loss`: base SSL loss type, uniformity lambda, temperature (\(\tau\))
  - `training_schedule`: schedule type (cosine), warmup epochs, min LR ratio
  - `evaluation`: number of epochs for linear probing, evaluation split
  - `logging`: directory for logs, save frequency

- Initialize experiment parameters, set a seed if desired (not specified but recommended for reproducibility).

---

## 2. **Data Loading**

- Use dataset_loader.py or similar logic:
  - Instantiate CIFAR-10 or CIFAR-100 dataset.
  - Apply augmentation pipeline specified:
    - RandomCrop with padding—likely 4 pixels.
    - HorizontalFlip.
    - ColorJitter with specified strengths.
    - GaussianBlur with a probability or kernel size.
  - Wrap in DataLoader with batch size (from config), shuffling enabled.
- Ensure support for dual augmentations (for contrastive methods) during training where needed.

## 3. **Model Initialization**

- Instantiate `Model`:
  - Encoder: ResNet-18 or ResNet-50 as per config.
  - Projection head: MLP with `projection_dim`.
  - For BYOL, include predictor MLP of size `predictor_dim`.
  - Use `use_mlp` flag to determine whether to include MLP layers.
- Initialize model weights—preferably randomly unless transfer learning is desired (not indicated).

## 4. **Loss Function Setup**

- Instantiate a `Losses` class or similar:
  - **Base SSL Loss:**
    - For MoCo v2/NNS, implement InfoNCE loss with temperature \(\tau\).
    - For BYOL, implement MSE loss or similar.
    - For Barlow Twins, covariance orthogonal regularization.
  - **Uniformity Loss:**
    - Implement `UniformityLoss` class with method to compute the \( -\mathcal{W}_2 \) loss:
      - Input: Batch features (or normalized features).
      - Compute empirical mean \(\hat{\mu}\).
      - Compute covariance matrix \(\hat{\Sigma}\).
      - Eigen-decompose \(\hat{\Sigma}\) or use SVD.
      - Calculate:
        \[
        - \mathcal{W}_2 = - \sqrt{ \|\hat{\mu}\|_2^2 + 1 + \operatorname{tr}(\hat{\Sigma}) - \frac{2}{\sqrt{m}} \operatorname{tr}(\hat{\Sigma}^{1/2}) }
        \]
    - The loss function class should combine the base SSL loss with the uniformity term, weighted by \(\lambda\) (from config).
- Set up optimizer (SGD with momentum, weight decay).

## 5. **Training Loop**

- For epoch in [1..`training.epochs`]:
  - **Warm-up & Schedule:**
    - Update learning rate according to cosine schedule; optional warmup with `warmup_epochs`.
    - Adjust the weight \(\lambda\) for uniformity loss during training if dynamic weighting scheme (e.g., linear decay) is used.
  - For each batch:
    - Generate augmented pairs: obtain two views per sample.
    - Forward pass:
      - Compute features using the encoder.
      - For methods like BYOL or MoCo, process through momentum encoders as specified.
    - Normalize features (L2 normalization) to approximate uniform distribution.
    - Compute the base SSL loss.
    - Compute batch statistics (mean, covariance).
    - Calculate \( -\mathcal{W}_2 \) loss.
    - Sum total loss:
      \[
      \text{loss} = \text{base loss} + \lambda \times (- \mathcal{W}_2)
      \]
    - Backpropagation:
      - Zero gradients.
      - Backward pass.
      - Optimizer step.
  - **Periodic evaluation:**
    - After each epoch or defined interval, log training metrics.
    - Save model checkpoints periodically (`save_model_every` configuration).

## 6. **Post-Training Evaluation**

- Freeze the encoder.
- Perform linear evaluation:
  - Attach a linear classifier (fully connected layer) to the frozen encoder.
  - Train linear classifier on training set (CIFAR’s train split) for `downstream_epochs`.
  - Record Top-1 and Top-5 accuracy.
- Compute representation properties:
  - Singular value spectrum of features (visualize for spectral collapse).
  - Distribution overlaps (via KL and Wasserstein distances) between batch features and Gaussian approximation.
  - Visualization:
    - Plot features in 2D (principle components or PCA).
    - Singular value spectra plots (log scale).
- Collect uniformity metrics during training:
  - Record at each epoch or fixed intervals for convergence analysis.
  - Visualize how the uniformity improves with training and with the additional loss.

## 7. **Visualization & Logging**

- Generate visualizations similar to Figures 12–17:
  - Distribution overlap plots.
  - Singular value spectra.
  - 2D scatter plots of features.
- Log metrics:
  - Accuracy (Top-1, Top-5).
  - Uniformity metrics (\(-\mathcal{L}_U\), \(-\mathcal{W}_2\)).
  - Spectral collapse indicators.
- Save logs and model checkpoints.

## 8. **Additional Considerations**

- For the eigenvalue-based covariance square root, implement eigen-decomposition such that numerical stability is handled, e.g., regularize small eigenvalues.
- For each computation of the Wasserstein distance, handle batch covariance \(\hat{\Sigma}\) shape correctly (eigen-decomposition of \(m \times m\) matrix).
- Ensure all random seed for reproducibility.
- Confirm normalization procedure—features should be normalized to the sphere before calculating the uniformity loss unless stated otherwise.

---

## 9. **Unspecified Details & Clarifications**

- Exact decay schedule for \(\lambda\): use linear decay or fixed value? *(The config allows for dynamic weighting but details are unspecified; initial proposal is either fixed or linearly decayed)*
- Eigen-decomposition method: full eigen-decomposition preferred for accuracy; approximate methods acceptable if computationally limited.
- Data augmentation pipeline: confirm order and parameters to match the paper.

---

# Summary

The `main.py` script will:
- Load and parse configurations.
- Set up datasets, models, losses, and optimizer.
- Run a training loop with integrated uniformity regularization via the Wasserstein distance.
- Perform periodic evaluations, including downstream accuracy and spectral analysis.
- Log results and produce visualizations to analyze the effectiveness of the uniformity metric.
- Save model checkpoints for further analysis or fine-tuning.

This logic ensures the experiment pipeline is aligned with the paper’s methodology, storage, and analytical goals, fostering reproducibility and rigorous evaluation.

## model.py

**Logic Analysis for `model.py` — Class `Model`**

---

### 1. **Purpose & Responsibilities**

The `Model` class is responsible for encapsulating all neural network components necessary for the self-supervised learning framework, including:

- The backbone encoder (either ResNet-18 or ResNet-50).
- The projection head (MLP) that maps encoder features to the learned embedding space.
- An optional predictor network (MLP), used in methods like BYOL.
- Methods to perform forward passes overall, extract features for analysis, and handle model initialization based on the configuration.

---

### 2. **Core Components**

#### 2.1. **Encoder (Backbone)**
- **Implementation:**
  - Instantiate a ResNet-18 or ResNet-50, based on the configuration parameter `backbone`.
  - Use torchvision's implementation (`torchvision.models.resnet18` / `resnet50`).
- **Output:**
  - Encoder outputs feature maps; typically, the output before the final fully connected layer.
  - For SSL, often features are taken after the global average pooling layer (`avgpool`).
- **Note:**
  - May need to exclude the final classification layer (`fc`) to get feature embeddings.
  - Optional: modify the last layer to output features of desired dimension if applicable.

#### 2.2. **Projection Head**
- **Implementation:**
  - An MLP with one or two fully-connected layers.
  - Input size: the encoder's feature dimension (e.g., 512 for ResNet-18/50 after global pooling).
  - Output size: the `projection_dim` (e.g., 128).
- **Activation:**
  - Use non-linear activation (ReLU) between layers.
  - Optional normalization (e.g., BatchNorm) after each linear layer.
- **Design considerations:**
  - Maintain consistent architecture with experimental setup.
  - Use dropout or other regularizers if needed (not specified but optional).

#### 2.3. **Predictor Network (Optional, Mainly in BYOL)**
- **Implementation:**
  - Same as the projection head but only instantiated if `predictor` is True.
  - Input: features from the projection head.
  - Output: same dimension as projection, typically.
- **Purpose:**
  - To predict the transformed representation, as in BYOL.
- **Parameters:**
  - Similar architecture to the projection MLP or simplified.

---

### 3. **Methods and Functionalities**

#### 3.1. **`__init__()`**
- **Inputs:**
  - The configuration parameters:
    - `backbone`: string, "ResNet-18" or "ResNet-50".
    - `projection_dim`: int, e.g., 128.
    - `predictor`: bool, whether predictor is used (BYOL).
    - `use_mlp`: bool, whether to include MLP layers.
  - Initialize:
    - `encoder`: the ResNet model without the final FC layer.
    - `projection_head`: MLP for feature embedding.
    - `predictor`: optional MLP.
  - Additional:
    - Possibly initialize weights, set device (GPU/CPU).
- **Implementation hints:**
  - Use `torchvision.models.resnet18` / `resnet50`.
  - Remove or replace the final FC layer, keeping features before classification.
  - Implement the MLPs with `nn.Sequential` for simplicity.
  
#### 3.2. **`forward(x)`**
- **Input:**
  - Input images or augmented views (batch).
- **Process:**
  - Pass input through encoder.
  - Apply the projection head to produce embedding features.
  - If `predictor` exists, pass the embedding through predictor.
- **Output:**
  - The predicted features (for the model used in SSL training).
  - Or, if needed, just the embedding features.
- **Note:**
  - Can be designed to return different outputs: e.g., features before predictor, final prediction, depending on the use case.

#### 3.3. **`extract_features(x)`**
- **Purpose:**
  - Extract raw features from the encoder (excluding projection head).
  - Useful for evaluating spectral properties, visualization, or auxiliary experiments.
- **Implementation:**
  - Pass input through encoder only.
  - Return the features immediately before the classification layer (global avg pooling output).

---

### 4. **Implementation Details & Tips**

- **Dynamic Initialization:**
  - Allow flexible selection of backbone (`resnet18`, `resnet50`).
- **Pre-trained weights:**
  - Default to training from scratch; optionally, support preloaded weights for fine-tuning.
- **Feature dimension determination:**
  - Need to determine the output size of encoder's pooled features (commonly 512 or 2048).
  - Usually accessible via inspecting the pretrained model or by forward pass with a dummy batch.
- **Normalization:**
  - Typically, features are normalized (L2) prior to computing the uniformity loss, but this can be handled outside `Model`.

---

### 5. **Summary & Pseudocode Outline**

- **Initialization:**

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Select backbone
        if config.backbone == 'ResNet-18':
            self.encoder = torchvision.models.resnet18(pretrained=False)
        elif config.backbone == 'ResNet-50':
            self.encoder = torchvision.models.resnet50(pretrained=False)
        else:
            raise ValueError('Unsupported backbone')
        
        # Remove final fc layer
        self.encoder = nn.Sequential(*(list(self.encoder.children())[:-1])) # Exclude last FC
        
        # Determine feature dimension (e.g., 512 for ResNet-18, 2048 for ResNet-50)
        self.feature_dim = ... # get from model or forward dummy input
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config.projection_dim)
        )
        
        # Predictor (if BYOL)
        if config.predictor:
            self.predictor = nn.Sequential(
                nn.Linear(config.projection_dim, 512),
                nn.ReLU(),
                nn.Linear(512, config.projection_dim)
            )
        else:
            self.predictor = None
```

- **Forward pass:**

```python
def forward(self, x):
    features = self.extract_features(x)  # raw features
    projections = self.projection_head(features)
    if self.predictor:
        predictions = self.predictor(projections)
        return predictions
    return projections
```

- **Feature extraction:**

```python
def extract_features(self, x):
    # Pass through encoder
    feat_map = self.encoder(x)
    # Global average pooling
    pooled_feat = torch.flatten(feat_map, start_dim=1)
    return pooled_feat
```

- **Notes:**
  - During training, features from `extract_features()` are used for the distribution-based uniformity computation.
  - Features may be normalized (L2) as a preprocessing step before calculating statistics.
  - Model parameters such as dropout, BatchNorm, etc., follow the original ResNet implementation or as specified.

---

### 6. **Edge Cases & Additional Considerations**
- **Size of features:**
  - Confirm the output size based on the input image size and architecture.
- **Batch Normalization:**
  - Use train mode; ensure BatchNorm is updated properly during training.
- **Device:**
  - Move all modules and tensors onto the same device (GPU/CPU).
- **Reproducibility:**
  - Set random seeds, initialize weights if needed.

---

This logic analysis provides a detailed, step-by-step blueprint for implementing the `Model` class aligned with the paper and experimental plan. It ensures modularity, flexibility, and correctness for subsequent coding.

## trainer.py

# Logic Analysis for `trainer.py`

This module is responsible for orchestrating the entire training procedure. It manages data feeding, loss computation, backpropagation, model updates, and periodic evaluation and logging. The implementation must be tightly aligned with the specifications derived from the paper, design plan, and configuration parameters.

---

## 1. **Initialization and Input Dependencies**
- **Inputs:**
  - `model`: An object encapsulating the encoder, projection head, and predictor (if applicable).
  - `dataloader`: A PyTorch DataLoader providing batches of augmented data.
  - `loss_fn`: An object implementing the combined loss, including the base SSL loss (e.g., InfoNCE) and the auxiliary Wasserstein uniformity loss.
  - `optimizer`: For updating model parameters; e.g., SGD with momentum.
  - `epochs`: Total number of training epochs.
  - `lambda_uniformity`: Weight for the uniformity loss; retrieved from configuration.

- **Attributes to Maintain:**
  - Current epoch count.
  - Step counter for iteration-based operations.
  - Checkpoint/save management for periodic logging.

---

## 2. **Training Loop Structure**
The core `train()` method involves iterating through epochs and, within each epoch, iterating over data batches.

### Epoch Loop:
- For each epoch in the range `[1, epochs]`:
  - Reset epoch-specific metrics.
  - Optionally, adjust the uniformity loss weight \(\lambda_t\) if a decay schedule is used (e.g., linear decay between `lambda_max` and `lambda_min` over epochs).   
    *Note:* The config specifies a linear decay, so compute \(\lambda_t\) at each epoch accordingly.

### Batch Loop:
- For each batch (inputs: views, labels if available) in `dataloader`:
  
  1. **Data Processing:**
     - Extract the floating point tensors of the mini-batch: e.g., `x1`, `x2` (for contrastive) or just `x` (for BYOL).
     - Move tensors to GPU if CUDA available.
     - Optionally, perform normalization of features (e.g., `L2` normalization) after encoding, as per design. This is essential since the uniformity metric assumes features are on the sphere.

  2. **Feature Extraction:**
     - Pass inputs through `model.encoder` (and optional predictor) to obtain features:
       ```
       z_a = model.extract_features(x1)
       z_b = model.extract_features(x2)  # or just z for non-contrastive methods
       ```
     - Normalize features:
       ```
       z_a_norm = normalize(z_a)
       z_b_norm = normalize(z_b)
       ```
     - For batch comments: For BYOL, also generate the predictor output, if applicable.

  3. **Compute the Base SSL Loss:**
     - Depending on the method:
       - For contrastive (e.g., MoCo): compute InfoNCE loss based on `z_a_norm`, `z_b_norm`.
       - For BYOL: compute MSE between predictor and target features.
       - For BarlowTwins: compute covariance-based loss.
     - Store this `base_loss`.

  4. **Compute the Uniformity Loss:**
     - Concatenate features (or treat both views together).
     - Compute batch means and covariance:
       ```
       mean, cov = compute_statistics([z_a_norm, z_b_norm])
       ```
     - Optional: re-scale features or ensure features are scaled to fit the assumption.
     - Calculate \( -\mathcal{W}_2 \) uniformity loss:
       - Using the batch mean and covariance:
         ```
         mu_hat = mean
         Sigma_hat = cov
         ```
       - Eigen-decompose \(\hat{\Sigma}\):
         ```
         eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_hat)
         ```
       - Compute trace terms:
         ```
         trace_Sigma = sum of eigenvalues
         sqrt_Sigma = eigenvectors @ diag(sqrt(eigenvalues)) @ eigenvectors.T
         trace_sqrt_Sigma = sum of sqrt eigenvalues
         ```
       - Compute the Wasserstein distance:
         ```
         W2_val = sqrt( ||mu_hat||^2 + 1 + trace_Sigma - 2 / sqrt(m) * trace_sqrt_Sigma )
         ```
       - Set uniformity loss \( = - W2_\text{val} \).

  5. **Aggregate Total Loss:**
     - Calculate total:
       ```
       total_loss = base_loss + lambda_t * (- W2_val)
       ```
     - Note: The sign indicates that lower Wasserstein distance corresponds to higher uniformity, so negative of the metric is used.

  6. **Backpropagation and Optimization:**
     - Zero out gradients:
       ```
       optimizer.zero_grad()
       ```
     - Backward pass:
       ```
       total_loss.backward()
       ```
     - Step optimizer:
       ```
       optimizer.step()
       ```

  7. **Metrics Logging (Optional per iteration):**
     - Track batch loss, uniformity loss, and other metrics.

### End of Batch:
- Update batch-level metrics (average loss, uniformity, etc.)

---

## 3. **Epoch End Operations**
- **Learning rate scheduling:** update according to a cosine schedule or warmup schedule implemented via PyTorch's scheduler.
- **Decay uniformity coefficient \(\lambda_t\):**
  - Compute this based on linear schedule between `lambda_max` and `lambda_min` over epochs:
    \[
    \lambda_t = \lambda_{max} - \frac{( \lambda_{max} - \lambda_{min} )}{T} \times t
    \]
  - Update the weight for loss accordingly for subsequent epoch.

- **Evaluation:**
  - At specified intervals (every `save_model_every` epochs):
    - Save model checkpoints.
    - Run validation:
      - Extract features with frozen encoder.
      - Compute downstream linear classifier accuracy.
      - Record uniformity metrics over the epoch.
      - Compute spectral analysis if needed.

- **Logging:**
  - Log training loss, uniformity, learning rate, \(\lambda_t\).
  - Store training histories for plotting convergence figures.

---

## 4. **Post-training**
- Final evaluations on test set.
- Save model checkpoint.
- Generate spectral plots, spectral singular values, and distribution visualizations.
- Export metrics for comparison.

---

## 5. **Error Handling & Additional Considerations**
- Ensure numerical stability during covariance eigen-decomposition:
  - Possibly add small epsilon to the eigenvalues if necessary.
- Handle potential batch size issues with covariance matrices.
- For large covariance matrix eigen-decomposition, optimize via eigendecomposition only on the batch.
- Confirm the use of float32 precision for eigen calculations.

---

## Summarized Key Points:
- The primary logic flow: Data → Feature Extraction → Statistics → Uniformity Metric → Loss aggregation → Backpropagation → Repeat.
- The Wasserstein distance is central; requires eigen-decomposition or SVD.
- Dynamic \(\lambda\) scheduling is used.
- Periodic evaluation of downstream performance and spectral properties.
- Emphasis on accurate feature normalization and batch statistics.

---

This detailed logical outline will guide the implementation of the `train()` method to precisely reflect the experimental design and replicate the key contributions of the paper.

## utils.py

{
  "utils.py": [
    {
      "Overview": "This module supplies essential utility functions supporting the main training and evaluation pipelines. Its core functions include feature normalization, covariance eigen-decomposition, spectral analysis, visualization, and metrics computation. These functions facilitate the implementation of the Wasserstein uniformity loss, spectral analysis of representations, and accuracy assessments.",
      "Detailed Functions and Logic": [
        {
          "FunctionName": "normalize_features",
          "Purpose": "Normalize feature vectors to lie on the unit hypersphere, ensuring all features have unit \(\ell_2\) norm. This standardization aligns with the theoretical assumptions about distributions being on the sphere, which is crucial for stable covariance estimates and meaningful uniformity metrics.",
          "Input": "features: Tensor of shape (batch_size, feature_dim)",
          "Output": "normalized_features: Tensor of same shape",
          "Logic": [
            "For each sample (row) in 'features', compute the \(\ell_2\) norm.",
            "Divide each feature vector by its \(\ell_2\) norm to obtain unit norm vectors.",
            "Handle case where norm is zero to prevent NaNs: add epsilon or use masking."
          ],
          "Implementation Note": "Use PyTorch operations for GPU efficiency, e.g., 'features = features / (features.norm(dim=1, keepdim=True) + epsilon)'."
        },
        {
          "FunctionName": "compute_empirical_statistics",
          "Purpose": "Calculate the empirical mean vector (\(\hat{\mu}\)) and covariance matrix (\(\hat{\Sigma}\)) of the features. These statistics underpin the quadratic Wasserstein loss computation.",
          "Input": "features: Tensor of shape (batch_size, feature_dim)",
          "Output": "mean: Tensor of shape (feature_dim, ), covariance: Tensor of shape (feature_dim, feature_dim)",
          "Logic": [
            "Compute the mean: mean = features.mean(dim=0).",
            "Center data: centered = features - mean.unsqueeze(0).",
            "Calculate covariance as: cov = (centered^T @ centered) / (batch_size - 1).",
            "Add small regularization if needed for numerical stability (e.g., epsilon scaled identity)."
          ],
          "Implementation Note": "Use PyTorch's batch ops for efficiency."
        },
        {
          "FunctionName": "covariance_sqrt_eigen",
          "Purpose": "Compute the square root of the covariance matrix (\(\hat{\Sigma}^{1/2}\)) via eigen-decomposition. This is needed for the closed-form Wasserstein distance calculation, where \(\operatorname{tr}(\hat{\Sigma}^{1/2})\) appears.",
          "Input": "covariance: Tensor of shape (feature_dim, feature_dim)",
          "Output": "cov_sqrt: Tensor of same shape, eigenvalues, eigenvectors",
          "Logic": [
            "Perform eigen-decomposition: eigvals, eigvecs = torch.linalg.eigh(covariance).",
            "Ensure eigenvalues are non-negative, clip if necessary for numerical stability.",
            "Compute sqrt_eigenvalues = torch.sqrt(eigvals).",
            "Reconstruct square root: cov_sqrt = eigvecs @ diag(sqrt_eigenvalues) @ eigvecs^T.",
            "Return the covariance square root matrix."
          ],
          "Implementation Note": "Use torch.linalg.eigh for symmetric matrices; handle potential negative eigenvalues due to numerical errors by clamping eigenvalues at zero."
        },
        {
          "FunctionName": "compute_uniformity_metric_W2",
          "Purpose": "Compute the \( -\mathcal{W}_2 \) uniformity score according to the paper's closed-form formula, based on statistical estimates.",
          "Input": "mean: (feature_dim,), cov: (feature_dim, feature_dim)",
          "Output": "scalar value representing negative Wasserstein distance",
          "Logic": [
            "Calculate the trace of the covariance matrix: trace_cov = trace(cov).",
            "Obtain the covariance square root: cov_sqrt = covariance_sqrt_eigen(cov).",
            "Compute trace of the covariance square root: trace_sqrt = trace(cov_sqrt).",
            "Calculate squared norm of mean: mu_norm_sq = torch.sum(mean ** 2).",
            "Compute the objective: W2 = sqrt(mu_norm_sq + 1 + trace_cov - (2 / sqrt(feature_dim)) * trace_sqrt).",
            "Return negative of W2 for loss minimization via gradient descent."
          ],
          "Implementation Note": "Wrap computations with detach and avoid gradients propagating through eigen-decomposition if only used for loss term."
        },
        {
          "FunctionName": "compute_accuracy",
          "Purpose": "Calculate classification accuracy for evaluation of learned representations in downstream linear classification tasks.",
          "Input": "preds: Tensor of predicted labels, labels: True labels, both size (num_samples)",
          "Output": "accuracy: float",
          "Logic": [
            "Compare predicted labels with true labels using (preds == labels).sum() / total samples.",
            "Return accuracy as percentage or decimal fraction."
          ],
          "Implementation Note": "Ensure preds and labels are on the same device; handle data type conversions if needed."
        },
        {
          "FunctionName": "plot_spectrum",
          "Purpose": "Visualize the singular values (spectral decay) of the covariance matrix of features, to analyze dimensional collapse.",
          "Input": "singular_values: array-like of shape (feature_dim,)",
          "Output": "Plot: log-spectral decay (can save as PNG or display)",
          "Logic": [
            "Convert singular values to log scale.",
            "Plot log(singular_values) vs index (sampled eigenvalue index).",
            "Label axes: 'Component index' and 'Log singular value'.",
            "Display or save plot for analysis. "
          ],
          "Implementation Note": "Use matplotlib.pyplot.ux or plt."
        },
        {
          "FunctionName": "visualize_distribution",
          "Purpose": "Visualize 2D scatter or density plots of features projected onto top 2 eigenvectors or randomly sampled features for distribution analysis.",
          "Input": "features: Tensor or ndarray (batch_size, feature_dim)",
          "Output": "Generated visualization plot",
          "Logic": [
            "Optional: Project features onto first two principal components or eigenvectors.",
            "Use scatter or density plotting (e.g., plt.scatter or plt.kdeplot).",
            "Color-code points for clarity.",
            "Save or display detailed distribution plots for inspection of uniformity vs collapse."
          ],
          "Implementation Note": "Use functions from matplotlib or seaborn."
        }
      ],
      "Implementation considerations": [
        "Eigen-decomposition: prefer 'torch.linalg.eigh' for symmetric matrices, handle eigenvalues carefully to avoid numerical instability.",
        "Feature normalization: ensure consistent normalization (e.g., all features normalized before covariance computation).",
        "Performance: batch all computations, avoid unnecessary copying, and leverage GPU acceleration.",
        "Numerical Stability: add small epsilon when taking eigenvalues and eigenvectors, clip negative eigenvalues, and regularize covariance matrices if necessary.",
        "Robustness: handle cases with low variance or near-zero feature vectors to avoid NaN or Inf errors."
      ],
      "Remarks": "These utility functions will underpin the core loss computations, spectral analyses, and visualization tasks necessary for diagnosing and demonstrating the properties of the proposed uniformity metric."
    }
  ],
  "Summary": "This detailed logic analysis for 'utils.py' provides a blueprint for implementing critical helper functions needed across the training and evaluation pipeline. It emphasizes numerical stability, efficiency, and correctness aligned with the theoretical principles in the paper, ensuring the functions support experiments for uniformity assessment, spectral analysis, accuracy measurement, and visualization of the learned representations."
}

