# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. To develop a comprehensive plan for reproducing the proposed "Sm" enhanced MIL method from the paper, I will organize the approach into two main sections: **Methodology** and **Experiments**. Each subsection will steadily detail the key components, hyperparameters, technical decisions, and procedural steps to follow. I will also highlight any information that is **not fully specified** in the paper, so you can decide on approximations or further detail collection.

---

## I. Methodology

### 1. Overall Objective and Architecture
- The goal is a **modular framework** for MIL that can incorporate the novel "smooth" operator "Sm" into existing attention-based MIL models, notably building on ABMIL.
- The architecture comprises:
  - An **instance feature extractor** (pre-trained CNN/VIT, e.g., ResNet18, ResNet50, ViT-B-32, with optional self-supervision).
  - A **global dependency module** (transformer encoder, optional).
  - The **"Sm" operator** applied at specific points:
    - On instance embeddings, prior to attention pooling ("early" variant).
    - Inside the transformer encoder to impose local smoothness ("late" variant).
    - As a combination ("both") in transformer and pooling.
  - Final **attention pooling** and **classifier** to produce bag and instance predictions.

### 2. Implementation Details
- **"Sm" Operator**:
  - Based on graph Laplacian regularization, approximated via an iterative process with T=10 steps (as per paper).
  - **Normalized Laplacian** used to ensure spectral properties.
  - Hyperparameters:
    - **α**: trade-off controlling smoothness versus feature fidelity, initialized at 0.5 and trainable.
    - **Number of iterations T=10**.
  - Implementation:
    - Construct the **adjacency matrix A** from instance embeddings or a predefined neighborhood system.
    - Compute degree matrix D and normalized Laplacian \( \tilde{L} \).
    - Update rule: \( G(t) = \alpha (I - \tilde{L}) G(t-1) + (1 - \alpha) U \).
    - Final output: \( \text{Sm}(U) = G(T) \).
- **Graph Construction**:
  - Use a fixed k-NN graph based on Euclidean distance in feature space, with \( k \in \{4, 8\} \) (or as hyperparameter).
  - For efficiency, precompute neighborhoods or dynamically compute at each forward pass.
  - Shared adjacency matrix for the batch, or compute per-bag.

### 3. Network Components
- **Feature Extractor**:
  - Use pre-trained ResNet18 / ResNet50 / ViT-B-32.
  - Extract features for each instance (patch or slice):
    - For WSI: extract patches of size 512x512 at 20x magnification.
    - For CT: extract slices as instances or patches.
  - Input features: 512-D (ResNet18), 2048-D (ResNet50), 768-D (ViT-B-32).
  - Optional use of feature embedding layer to reduce dimension if necessary.
- **Global Dependencies**:
  - Transformer encoder with:
    - Number of layers: 2-4 (hyperparameter).
    - Hidden size: same as feature dimension.
    - Multi-head self-attention: 4-8 heads.
  - Apply "Sm" either before, after, or in both points (early/mid/late).
- **Attention Pooling & Classifier**:
  - Attention module as in ABMIL but with "Sm" applied to instance features or attention scores.
  - Use softmax attention weights.
  - Final linear classifier (fully connected layer) for bag-level prediction.
  - Instance-level predictions via the attention-weighted features.

### 4. Training Details
- **Loss Function**:
  - Binary cross-entropy for bag-level label.
  - Optional auxiliary instance supervision if available (though not mandatory here).
- **Hyperparameters**:
  - Learning rate: 1e-4 (Adam optimizer).
  - Batch size: dependent on GPU memory (e.g., 8-16 bags).
  - Regularization: weight decay 1e-4.
  - "α" parameter:
    - Initialized at 0.5.
    - Trainable with bounds [0,1], using sigmoid or softplus.
- **Training schedule**:
  - 50 epochs (based on paper).
  - Use early stopping based on validation bag AUROC.
  - Use five-fold cross-validation splits per dataset (see dataset section).
- **Gradient considerations**:
  - Ensure "Sm" is differentiable w.r.t. α and the instance features.
  - Possibly constrain weights/scaling to prevent instability.

### 5. Variants to Implement
- **"early"**: Apply "Sm" only before attention pooling.
- **"mid"**: Apply "Sm" before transformer, after embeddings.
- **"late"**: Apply "Sm" after transformer encoder output.
- **"both"**: Apply "Sm" in both points.
- Implement "Sm" as a custom PyTorch module/functions, integrated in the forward pass with batch processing.

### 6. Additional notes
- Initialize "α" as 0.5, train end-to-end.
- For spectral normalization (in "Sm"), enforce spectral norm constraints on W matrices if needed.
- When computing adjacency matrices, consider:
  - **Instance-wise**: based on feature similarity.
  - **Fully connected**: with learned edge weights.
  - Regularize or sparsify the adjacency for stability.

---

## II. Experiments

### 1. Datasets and Preprocessing
- **RSNA**:
  - 1150 scans, 39750 slices.
  - Slices resized/cropped to patches of size 512x512.
  - Use the Kaggle split (train/test).
  - Features: ResNet18 pre-trained on ImageNet.
  - Bag: entire scan; instances: patches.
  - Labels: binary (hemorrhage presence/absence).
- **PANDA**:
  - 10503 WSIs, 1,107,931 patches.
  - Patches of size 512x512 at 10x magnification.
  - Features: ResNet50 (possibly BT trained).
  - Bag: WSI; instances: patches.
  - Labels: severity (binary).
- **CAMELYON16**:
  - 270 WSI train, 130 WSI test.
  - Patches of size 512x512 at 20x magnification.
  - Features: ResNet50-BT (self-supervised).
  - Bag: WSI; instances: patches.
  - Labels: metastatic or not.
- Dataset split: following original or the paper’s splits.
- For consistency, split training data into five folds for cross-validation.

### 2. Feature Extraction
- Use provided pre-trained models.
- Extract features **offline** (on large GPU or TPU clusters) to save memory:
  - Save features per instance for each bag.
- Store features for all datasets with associated bag labels.

### 3. Model Training
- For each dataset:
  - Use a five-fold cross-validation setup.
  - For each fold:
    - Initialize the model with:
      - Pre-trained feature extractor frozen or finetuned (based on resource).
      - Attention-based pooling with/without "Sm".
      - Transformer encoder (optional, in variants).
    - Hyperparameters:
      - Learning rate: 1e-4.
      - Batch size: as memory permits (e.g., 8 bags).
      - Use Adam optimizer.
      - Train 50 epochs, early stop based on validation AUROC.
      - Hyperparameters α, number of T steps, k-NN adjacency, number of transformer layers: as per paper defaults or through validation.
- Measure bag AUROC and instance AUROC (via attention scores or instance predictions).
- For localization, analyze attention maps and instance scores.

### 4. Evaluation Metrics
- **Bag-level**:
  - AUROC: main metric.
  - F1 score at optimal threshold.
- **Instance-level**:
  - AUROC of attention scores with respect to ground truth instance labels (if available).
  - Visualization of attention maps; qualitative assessment.
- **Additional**:
  - Dirichlet energy of attention scores.
  - Bag or instance rank summaries.
  
### 5. Ablation Studies & Hyperparameter Sweeps
- Vary "α" in [0, 1] (e.g., 0.1, 0.5, 0.9).
- Vary number of "Sm" steps T (e.g., 1, 5, 10, 20).
- Explore different adjacency construction strategies.
- Compare variants ("early", "mid", "late", "both").
- Evaluate different feature extractors.

### 6. Visualization & Interpretation
- Attention maps overlaid on images (WSI slices, CT slices).
- Attention histograms for instance scores.
- Highlight local smoothness effect by different α values.

---

## Summary of Critical Missing Details and Assumptions:
- Exact graph construction (k-NN, fully connected, learned edges): default to k-NN based on feature space.
- Hyperparameters for transformer (layers, heads): default to 2-3 layers, 4-8 heads.
- Training epochs: 50 with early stopping.
- Regularization and normalization: weight decay 1e-4, spectral normalization on W if implemented.
- Final code will need to programmatically implement:
  - The iterative "Sm" approximation.
  - Dynamic adjacency or fixed pre-computed.
  - Weight constraints/initialization for "α".
  - Cross-validation and evaluation metrics.

---

With this detailed roadmap, you are well-positioned to proceed with code development, ensuring alignment with the paper’s methodology and experimental setup.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular MIL pipeline using PyTorch that integrates a pre-trained feature extractor, an optional transformer encoder, and the 'Sm' smoothing operator applied at configurable points. The core components include data loading with feature extraction, a model class encapsulating the attention mechanism with potential 'Sm' application, and a training loop that optimizes on bag labels with cross-validation. The 'Sm' operator is implemented as a differentiable PyTorch module that approximates the iterative graph smoothing process, using the normalized Laplacian constructed from instance features via a k-NN graph. Hyperparameters such as alpha, number of 'Sm' steps, and graph parameters are trainable or configurable. The system includes data preprocessing scripts, model classes, training and validation routines, and evaluation metrics focusing on AUROC for both bag and instance predictions, along with visualization utilities for attention maps.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "sm_operator.py",
        "config.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(data_paths: dict, transforms: callable)\n        +load_data() -> Dataset\n    }\n    class CustomDataset {\n        +__init__(bags: List[dict])  # Each dict: {'features': Tensor, 'label': int, 'instance_labels': Optional[Tensor]}\n        +__getitem__(index: int) -> dict\n        +__len__() -> int\n    }\n    class Model {\n        +__init__(feature_dim: int, use_transformer: bool, sm_points: str, hyperparams: dict)\n        +forward(instance_features: Tensor, adjacency: Tensor) -> dict  # outputs: bag_pred, instance_scores, attention_weights\n    }\n    class SmOperator (\n        +__init__(num_steps: int, alpha: float, feature_dim: int)\n        +forward(instance_embeddings: Tensor) -> Tensor  # returns smoothed embeddings\n    }\n    class Trainer {\n        +__init__(model: Model, dataset: Dataset, optimizer: torch.optim.Optimizer, criterion: callable, config: dict)\n        +train_epoch() -> float  # returns loss\n        +validate() -> dict  # returns metrics like AUROC\n        +train() -> None\n    }\n    class Evaluation {\n        +compute_metrics(predictions: dict, ground_truths: dict) -> dict\n        +plot_attention_maps(attention_weights: Tensor, images: Tensor) -> None\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Trainer --> Model\n    Model --> SmOperator\n    Trainer --> evaluation.py\n    dataset_loader.py <|-- CustomDataset\n    utils.py <|-- (utility functions: graph construction, feature normalization, metric calculation)\n    sm_operator.py --|> SmOperator\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant DS as Dataset\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: initialize and load_data()\n    DL-->>M: dataset\n    M->>MD: initialize with hyperparameters\n    loop cross-validation split\n        MD->>TR: initialize optimizer, criterion\n        TR->>TR: train_epoch() for each epoch\n        TR->>EV: validate() -> metrics\n        EV->>TR: return metrics\n    end\n    TR->>EV: evaluate final model on test set\n    EV->>EV: plot attention maps, compute AUROC\n    Main-->END: Save results, visualize, cleanup\n",
    "Anything UNCLEAR": "Clarification needed on the exact input dataset format, especially how instance labels are provided or inferred, and whether specific neighborhoods are needed for adjacency matrix construction. Also, details on how to handle large BAGs (memory constraints) and whether multiple train runs or hyperparameter searches are expected."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "torchvision==0.12.0",
        "numpy==1.21.0",
        "scikit-learn==0.24.2",
        "pandas==1.3.5",
        "tqdm==4.62.3",
        "scipy==1.7.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Acts as the entry point, initializes configs, loads dataset, calls training and evaluation modules, manages cross-validation loops, and orchestrates overall experiment flow."
        ],
        [
            "dataset_loader.py",
            "Implements DatasetLoader class: handles raw data paths, performs feature extraction offline (or during initialization), constructs dataset objects (CustomDataset), manages dataset splits, and provides data loaders for training, validation, testing. Ensures adjacency matrix creation based on feature similarity for the 'Sm' operator."
        ],
        [
            "model.py",
            "Defines Model class: encapsulates feature extractor (pre-trained CNN or ViT), optional transformer encoder, attention pooling, and application points for Sm (early/mid/late). Implements the 'Sm' operator as a sub-module, ensuring differentiability and trainable alpha; constructs adjacency matrix within Sm or accepts externally computed adjacency. Outputs bag predictions, attention maps, and instance scores."
        ],
        [
            "sm_operator.py",
            "Implements SmOperator class: performs graph Laplacian smoothing, constructs normalized Laplacian from instance features (provided or computed in DatasetLoader), uses the iterative process with T=10 steps, hyperparameters alpha (trainable) and steps can be optimized. Provides forward() method returning smoothed embeddings."
        ],
        [
            "trainer.py",
            "Defines Trainer class: initializes optimizer, loss criterion; handles training loops, validation, early stopping. Coordinates model forward pass, applies Sm at specified points, computes metrics (AUROC, F1). Manages batching, model checkpoints if needed."
        ],
        [
            "evaluation.py",
            "Provides evaluation routines: compute metrics (AUROC, F1), generate attention maps for localization, visualize attention overlays, and statistical summaries of instance predictions. Uses sklearn metrics and matplotlib for plots."
        ],
        [
            "utils.py",
            "Contains utility functions: construct adjacency (k-NN graph), normalize features, compute AUROC, plot attention maps, handle cross-validation splits, and log results."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "sm_operator.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "None. Internal module APIs: \n\nclass SmOperator:\n    def __init__(self, num_steps: int = 10, alpha: float = 0.5):\n        ... \n    def forward(self, embeddings: Tensor, adjacency: Tensor) -> Tensor:\n        ...\n\nclass Model:\n    def __init__(self, feature_dim: int, use_transformer: bool = False, sm_points: str = 'early'):\n        ... \n    def forward(self, instance_features: Tensor, adjacency: Tensor) -> dict:\n        ...\n\nclass Trainer:\n    def __init__(self, model: Model, dataset: Dataset, optimizer: Optimizer, criterion: callable, config: dict):\n        ...\n    def train_epoch(self):\n        ...\n    def validate(self):\n        ...\n    def train(self):\n        ...\n\nclass Evaluation:\n    def __init__(self, model: Model, dataset: Dataset):\n        ... \n    def evaluate(self):\n        ... \n    def plot_attention(self, attention_weights: Tensor, images: Tensor):\n        ...\n\nData pipeline: DatasetLoader loads features, labels, and precomputes adjacency matrices. DataLoaders for train, val, test. Model API for prediction. Trainer manages epoch cycles and validation. Evaluation handles metrics and visualization.",
    "Shared Knowledge": "Feature extractor configurations (ResNet, ViT), hyperparameter defaults for alpha and T, adjacency matrix construction method (k-NN based on features), normalization functions, PyTorch modules for attention pooling and transformer layers, metric calculation functions (AUROC, F1). The 'Sm' operator is implemented as a separable class with differentiable forward(). Cross-validation splitting utilities and logging framework for reproducibility.",
    "Anything UNCLEAR": "Exact process for adjacency matrix construction (fixed k-NN or dynamic), whether instance labels are available or inferred, and the preferred way to integrate 'Sm' (points of application). Clarification needed on handling large bags and hardware considerations for memory management, as well as validation set hyperparameter tuning procedures."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0001       # Based on typical deep learning practice in paper; paper states "train 50 epochs" but does not specify LR, so default to 1e-4
  batch_size: 8               # Depends on GPU/memory; value selected as a reasonable default
  epochs: 50                  # As per paper training schedule
  early_stopping_patience: 10 # To prevent overfitting; not explicitly in paper but recommended
  weight_decay: 0.0001        # Common regularization
  optimizer: Adam             # Recommended optimizer
  alpha_init: 0.5             # Initial alpha for Sm (trainable parameter)
  num_smoothing_steps: 10     # Number of T steps in Sm approximation
  use_spectral_norm: true     # To stabilize training with Laplacian matrices
  device: cuda                # or 'cpu' depending on hardware
  seed: 42                    # For reproducibility

dataset:
  name: RSNA                  # Change as needed (PANDA, CAMELYON16)
  data_paths:
    features: path/to/features   # To be set by user
    labels: path/to/labels       # To be set by user
  image_size: 512              # Patch/image size in pixels
  magnification: 20            # For WSIs
  split_seed: 42               # For cross-validation splits

model:
  feature_extractor: resnet18  # Options: resnet18, resnet50, vit-b-32
  freeze_feature_extractor: true # Freeze during training; set false for finetuning
  use_transformer: true        # Optional transformer encoder after features
  transformer_layers: 2        # Number of transformer layers
  transformer_heads: 4         # Number of attention heads in transformer
  attention_points: early      # Options: early, mid, late, both
  sm_enabled: true             # Enable the 'Sm' operator
  sm_points: early             # Points where 'Sm' is applied: early, mid, late, or both
  sm_num_steps: 10             # Number of steps in 'Sm' iterative approximation
  sm_trainable_alpha: true     # Alpha is trainable, initialized at 0.5

evaluation:
  metrics: [AUROC, F1]          # Metrics for evaluation
  visualization: true          # Enable attention map overlays
  attention_map_color: jet     # Colormap for attention heatmaps

misc:
  cross_validation_folds: 5   # Number of CV splits
  num_runs: 5                 # For averaging results
  logging_dir: logs/           # Directory for logs
  output_dir: outputs/       # Directory for checkpoints and final results
```

---

**Note:** Full configuration is available in `planning_config.yaml`
