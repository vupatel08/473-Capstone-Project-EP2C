# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here's a comprehensive, step-by-step plan to reproduce the methodology, experiments, and evaluation of the "Balancing Alignment and Uniformity (BAU)" framework for Domain Generalizable Person Re-Identification as described in the paper. The plan covers methodological details, dataset configurations, hyperparameters, and experimental protocols, enabling precise implementation later.

---

## 1. Methodology Implementation Plan

### **Overview & Core Objectives**
- The goal is to learn a discriminative yet domain-invariant feature space that balances intra-class alignment and distribution uniformity.
- Key components:
  - Data augmentations applied to input images.
  - Feature extraction via a backbone network.
  - Contrastive losses: alignment, uniformity, and domain-specific uniformity.
  - Additional regularizers: cross-entropy loss (classification), triplet loss.
  - Optional weighting strategy for alignment loss based on sample reliability.

---

### **a. Data Preparation**
- Datasets entail multiple labeled source domains and possibly target/test domains for evaluation.
- For training:
  - Prepare batches that contain images from multiple identities and domains.
  - For datasets with multiple identities per domain, ensure balanced sampling.
  - For datasets with disjoint/split identities across domains, maintain domain labels for domain-specific operations.

### **b. Data Augmentation**
- Implement the augmentations described:
  - **Random Erasing**: Randomly erase patches in an input image with a specified probability.
  - **RandAugment**: Apply a sequence of random transformations with controlled probability and magnitude.
  - **Color Jitter**: Adjust brightness, contrast, saturation, hue with probabilistic control.
- Each augmentation can be parameterized by probability `p`:
  - Generate augmented views `~x` for each input `x`.
  - The augmentation probability `p` is a hyperparameter; typical range: `[0, 1]`.
- Remember to generate augmentation parameters dynamically per batch during training.

### **c. Feature Encoder Network**
- Backbone:
  - Use a standard ResNet-50 or equivalent as in the backbone experiments for initial validation.
  - For ablation, consider VIT-B/16 or MobileNetV2.
- Embedding layer:
  - Output feature vectors normalized to the unit hypersphere (l2 normalization).
  - Separate the feature extractor from the classifier heads.

### **d. Loss Functions & Strategy**
- **Contrastive Alignment Loss (`L_align`)**:
  - Applied between features of original and augmented images.
  - Sample positive pairs are features from images with the same identity.
  - Use a weighting scheme:
    - Calculate reciprocal k-NN sets for each augmented feature and its original.
    - Compute Jaccard similarity `w_{ij}`.
    - Normalize weights over the positive pairs.
    - Loss:
      \[
      \mathcal{L}_{align} = \sum_{(i,j)} \bar{w}_{ij} \left\| \tilde{\mathbf{f}}_i - \mathbf{f}_j \right\|_2^2
      \]
- **Uniformity Loss (`L_uniform`)**:
  - Calculated on features from all samples in a batch:
    \[
    \mathcal{L}_{uniform} = \log \left( \frac{1}{|\mathcal{T}_{data}|} \sum_{(i,j)} e^{-2 \|\mathbf{f}_i - \mathbf{f}_j\|_2^2} \right) + \log \left( \frac{1}{|\mathcal{T}_{data}|} \sum_{(i,j)} e^{-2 \|\bar{\mathbf{f}}_i - \bar{\mathbf{f}}_j\|_2^2} \right)
    \]
  - `$\mathbf{f}_i$` are features; `$\bar{\mathbf{f}}_i$` are their (possibly) normalized versions or embedded features.
- **Domain-specific Uniformity Loss (`L_domain`)**:
  - Using a memory bank of class prototypes:
    - Maintain class prototypes updated via a momentum strategy:
      \[
      \mathbf{c}_j \leftarrow \mu \mathbf{c}_j + (1 - \mu) \mathbf{f}_i
      \]
    - For each domain, compute the distribution of features relative to domain prototypes and maximize uniformity within each domain.
    - Loss encourages features of the same domain to be dispersed around their prototypes.
- **Classification Loss (`L_ctr`)**:
  - Cross-entropy over identity labels.
- **Triplet Loss (`L_triplet`)**:
  - Standard batch-hard triplet with proper sampling.
- **Total Loss**:
  \[
  \mathcal{L}_{BAU} = \mathcal{L}_{ctr} + \mathcal{L}_{triplet} + \lambda \mathcal{L}_{align} + \mathcal{L}_{uniform} + \mathcal{L}_{domain}
  \]
  - Hyperparameters: `$\lambda$`, weight decay, learning rate, margin for triplet, etc.

### **e. Weighting Strategy for Alignment**
- For each augmented-original pair:
  - Compute reciprocal  k-NN for features within the mini-batch.
  - Compute Jaccard similarity `w_{ij}`.
  - Normalize weights.
- Apply weights in the alignment loss summation to focus on reliable pairs, reducing noisy alignments.

### **f. Training Details**
- Optimizer:
  - Use SGD or Adam with momentum.
  - Learning rate schedule with warm-up (if needed) and cosine decay.
- Batch size:
  - Select based on hardware (~64 or 128 images per batch).
- Epochs:
  - Sufficient to converge (from paper: about 60–120 epochs).
- Regularization:
  - Label smoothing, dropout as needed.
- Checkpointing:
  - Save early and best models based on validation mAP or CMC accuracy.

---

## 2. Experimental Setup & Dataset Details

### **a. Datasets**
- **Source Domains** (training): 
  - Market-1501, MSMT17, CUHK02, CUHK03, CUHKSYSU, PRID, GRID, VIPeR, iLIDS.
- **Test Domains** (evaluation):
  - Held-out datasets or protocols (e.g., Market-to-Duke, Multi-source, or cross-dataset evaluations for DG).
- Data specifics:
  - For each dataset:
    - Number of identities.
    - Number of images and cameras.
    - Dataset splits (train/test/protocols).
  - For training:
    - Use identities from source datasets.
    - Keep domain labels for domain-specific losses.

### **b. Data Processing & Augmentation**
- Resize images (e.g., 256×128 as standard).
- Normalize inputs (mean and std based on backbone).
- Apply augmentations probabilistically per batch.
- For dataset-specific:
  - Implement careful sampling to include multiple instances per identity.
  - For multi-view/sequence datasets, sample multiple images per identity.

### **c. Evaluation Protocols**
- Protocol-1 (single source): evaluate on the same dataset.
- Protocol-2 & 3 (cross-domain):
  - Train on source(s). 
  - Test across hold-out datasets.
- Metrics:
  - Mean Average Precision (mAP).
  - Rank-1 accuracy.
  - Cumulative Match Characteristic (CMC top-K).
- Use standard evaluation scripts (e.g. from ReID baselines).

---

## 3. Hyperparameter & Training Strategy
- **Hyperparameters**:
  - `λ` (alignment loss weight): e.g., 1.0.
  - `k` in reciprocal nearest neighbors: e.g., 10.
  - Augmentation probability `p`: tuned between 0.25–0.75.
  - Learning rate: e.g., 0.001 with decay.
  - Margin for triplet: e.g., 0.3.
  - Momentum `μ` for prototype update: 0.999.
  - Weight decay: 1e-4.
- **Training Schedule**:
  - Epochs: 60–120.
  - Learning rate warm-up for first 5 epochs, then cosine decay.
  - Regular evaluation on validation split or a held-out subset.
- **Ablation & Sensitivity Analysis**:
  - Vary augmentation probability.
  - Vary `k` and `λ` to check robustness.
  - Evaluate loss components individually to observe their impact.

---

## 4. Additional Implementation Notes
- **Memory Bank**:
  - Store class prototypes initialized by class means.
  - Update after each batch with momentum.
  - For classes not in the batch, keep previous prototype.
- **Sample Reliability Weights**:
  - Ensure weights are clipped/normalized for stability.
- **Code Structure**:
  - Modular: data loader with augmentations, backbone with normalization, loss module, optimizer, evaluation scripts.
- **Logging & Monitoring**:
  - Track mAP and Rank-1 during training.
  - Log uniformity, alignment metrics (per batch) for analysis.

---

## 5. Summary & Potential Implementation Challenges
- Precise implementation of the weighting scheme for alignment.
- Effective batch sampling ensuring identities and domains are well represented.
- Efficient nearest neighbor computation within each mini-batch.
- Proper handling of domain prototypes, especially with large datasets.
- Careful design of augmentations and their probabilities.
- Balancing training stability with multiple regularizers and losses.

---

## 6. Final Remarks
- Start with the backbone and standard classification + triplet losses.
- Incorporate data augmentations with probabilistic application.
- Implement the uniformity and domain uniformity losses with memory bank.
- Use the described hyperparameters as initial defaults; tune based on validation.
- Evaluate per protocols for robustness.
- Document each step extensively before coding.

---

This roadmap offers a detailed blueprint to reproduce the described BAU framework and results faithfully. Adjustments can be made based on dataset availability, hardware constraints, or desired ablation depth.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular training pipeline using PyTorch. The system will comprise a dataset loader with augmentation, a backbone model with normalization, a combined loss module including contrastive (alignment, uniformity, domain-specific uniformity), classification, and triplet losses, a training loop handling augmentation, neighbor lookup for alignment weights, and prototype updates. The system will also include an evaluation module to compute ReID metrics, and configuration scripts for hyperparameters and dataset paths.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "losses.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "config.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_paths: dict, augmentation_probs: dict)\n        +load_data(split: str) -> Dataset\n        +get_batch() -> Batch\n    }\n    class Model {\n        +__init__(params: dict)\n        +forward(x: Tensor) -> Tensor\n        +extract_features(x: Tensor) -> Tensor\n    }\n    class LossFunctions {\n        +__init__(config: dict)\n        +calculate_alignment(features: Tensor, neighbor_indices: torch.Tensor) -> Tensor\n        +calculate_uniformity(features: Tensor) -> Tensor\n        +calculate_domain_uniformity(features: Tensor, prototypes: Tensor, domain_labels: List[int]) -> Tensor\n        +classification_loss(preds: Tensor, labels: Tensor) -> Tensor\n        +triplet_loss(embeddings: Tensor, labels: Tensor) -> Tensor\n    }\n    class PrototypeBank {\n        +__init__(num_classes: int, feature_dim: int, momentum: float)\n        +update(prototypes: Tensor, class_indices: List[int]) -> None\n        +get_prototypes() -> Tensor\n        +assign_closest_prototypes(features: Tensor, domain_labels: List[int]) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, losses: LossFunctions, prototypes: PrototypeBank, data_loader: DatasetLoader, config: dict)\n        +train_epoch() -> None\n        +compute_weights(features: Tensor, neighbors: Tensor) -> Tensor\n        +update_prototypes(features: Tensor, labels: List[int], domain_labels: List[int]) -> None\n        +train() -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, dataset: Dataset, config: dict)\n        +evaluate() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> Model\n    Main --> LossFunctions\n    Main --> PrototypeBank\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> Model\n    Trainer --> LossFunctions\n    Trainer --> PrototypeBank\n    Trainer --> DatasetLoader\n    Evaluation --> Model\n    DatasetLoader ..> Dataset\n    Model --> Tensor\n    LossFunctions --> Tensor\n    PrototypeBank --> Tensor\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant LF as LossFunctions\n    participant PB as PrototypeBank\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>+DL: load_data(training_split)\n    DL-->>-M: dataset object\n    M->>+MD: initialize(model_params)\n    MD-->>-M: model instance\n    M->>+LF: initialize(loss configs)\n    LF-->>-M: loss object\n    M->>+PB: initialize(num_classes, feat_dim, momentum)\n    PB-->>-M: prototypes\n    M->>+TR: start_training(model, loss, prototypes, data_loader, configs)\n    TR->>TR: for epoch in epochs:\n    TR->>DL: get_batch()\n    DL-->>TR: batch data\n    TR->>MD: features = model.extract_features(batch images)\n    TR->>TR: compute neighbor sets for weights\n    TR->>TR: weights = compute_weights(features, neighbors)\n    TR->>LF: total_loss = sum of all losses with proper weights and prototypes\n    TR->>MD: backward and update\n    TR->>PB: update prototypes from features and labels\n    TR->>EV: evaluate(model, test_dataset)\n    EV-->>-TR: metrics\n    TR-->>-M: report metrics and save best model\n    M->>EV: evaluate(model, test_dataset)\n    EV-->>-M: final metrics\n",
    "Anything UNCLEAR": "Exact dataset formats, neighbor search hyperparameters like search radius and k for reciprocal neighbors, and how to initialize and update domain labels for domain-specific uniformity are not explicitly detailed; clarification is needed for these implementation specifics, especially the neighbor search method and prototype initialization details."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.10.0",
        "torchvision==0.11.1",
        "numpy==1.21.0",
        "scipy==1.7.0",
        "scikit-learn==0.24.2",
        "opencv-python==4.5.3",
        "pyyaml==6.0",
        "tqdm==4.62.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Main entry point: initializes configurations, instantiates DatasetLoader, Model, Loss modules, PrototypeBank, Trainer, and Evaluation. Controls overall execution flow: dataset preparation, training epochs, evaluation, and model saving."
        ],
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: handles dataset paths, dataset splitting, sampling strategies, and data augmentation pipeline. Loads datasets (source & target), applies augmentations probabilistically, and provides batching logic flexible for multi-domain datasets. Interfaces: load_data(), get_batch()."
        ],
        [
            "model.py",
            "Defines Model class: encapsulates backbone network (ResNet, ViT, etc.), feature extraction (with normalization), and optional classifier heads. Methods: __init__, forward, extract_features. Ensures output features are l2-normalized for contrastive losses."
        ],
        [
            "losses.py",
            "Contains loss functions: alignment loss with neighbor weights, uniformity loss, domain uniformity loss, cross-entropy, triplet. Exposes functions/classes: compute_alignment(), compute_uniformity(), compute_domain_uniformity(), get_cross_entropy_loss(), get_triplet_loss()."
        ],
        [
            "prototype.py",
            "Defines PrototypeBank class: maintains class prototype vectors, updates them via momentum, assigns features to closest prototypes for domain-specific uniformity, and retrieves prototypes for losses. Methods: __init__, update(), get_prototypes(), assign_closest()."
        ],
        [
            "trainer.py",
            "Implements Trainer class: orchestrates training loop—loads batches, computes neighbor sets for alignment weights, applies augmentations, calculates total loss, performs backprop, updates prototypes, logs metrics. Methods: __init__, train_epoch, train, compute_weights, update_prototypes."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class: loads evaluation dataset, runs inference, computes metrics (mAP, Rank-1, CMC), supports multi-domain evaluation, provides evaluate() method returning metrics."
        ],
        [
            "utils.py",
            "Utility functions: neighbor search (kNN, reciprocal neighbors), augmentation implementations, prototype update helpers, metric calculators, configuration parsers."
        ],
        [
            "config.py",
            "Configuration parameters: dataset paths, hyperparameters (loss weights, augmentation probabilities, k-NN neighbors), training schedule, model options (backbone type), evaluation protocols."
        ]
    ],
    "Task list": [
        "config.py",
        "utils.py",
        "dataset_loader.py",
        "model.py",
        "losses.py",
        "prototype.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "All modules rely on common utility functions; dataset loader and trainer share configuration parameters like augmentation probabilities, neighbor search parameters, and prototype update strategies; model and losses modules share feature normalization conventions and loss calculation interfaces.",
    "Anything UNCLEAR": "Details on neighbor search parameters, exact feature normalization methods, dataset sampling schemes for multi-domain data, and protocol-specific dataset splits need clarification to ensure faithful replication."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 60
  warmup_epochs: 5
  weight_decay: 1e-4
  triplet_margin: 0.3
  lambda_alignment: 1.0
  g_hard_triplet_loss: true
  augmentation_probability: 0.5
  neighbor_k: 10
  prototype_momentum: 0.999

model:
  backbone: resnet50
  feature_dim: 512
  normalize_features: true

dataset:
  datasets_paths:
    market1501: "/path/to/market1501"
    msmt17: "/path/to/msmt17"
    cuhk02: "/path/to/cuhk02"
    cuhk03: "/path/to/cuhk03"
    cuhksysu: "/path/to/cuhksysu"
    prid: "/path/to/prid"
    grid: "/path/to/grid"
    viper: "/path/to/viper"
    ilids: "/path/to/ilids"
  image_size: [256, 128]
  training_split: train
  testing_split: test

augmentation:
  random_erasing: true
  random_erasing_prob: 0.25
  rand_augment: true
  rand_augment_prob: 0.5
  color_jitter_prob: 0.3
  color_jitter_params:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1

loss_weights:
  classification: 1.0
  triplet: 1.0
  alignment: 1.0
  uniformity: 1.0
  domain_uniformity: 1.0

evaluation:
  protocol: Protocol-3
  metrics:
    - mAP
    - Rank-1
    - CMC@1-5
  evaluation_batch_size: 64

misc:
  neighbor_search:
    k: 10
  prototype_update:
    momentum: 0.999
  save_model_path: "./results/model.pth"
  log_interval: 50
  seed: 42
```

---

**Note:** Full configuration is available in `planning_config.yaml`
