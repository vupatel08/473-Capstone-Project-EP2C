# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! To prepare a comprehensive plan for reproducing the xMIL framework detailed in the paper, I will organize the outline into two main sections: **Methodology** and **Experiments**. Each section will include specific sub-sections covering datasets, model architecture, training procedures, explanation techniques, evaluation metrics, hyperparameters, and implementation considerations.

---

# 1. Methodology

## 1.1. Core Concept and Overall Architecture of xMIL
- **Goal:** Provide instance-level explanations for MIL models, particularly in histopathology.
- **Main idea:** Combine a base MIL model (attention-based or transformer-based) with layer-wise relevance propagation (LRP), adapted as xMIL-LRP, to obtain faithful, fine-grained explanations.
- **Key features:**
  - Hierarchical relevance propagation, considering instance interactions.
  - Explanation scores discriminate between supporting or refuting evidence.
  - Compatibility with various MIL architectures (attention, transformer, additive MIL).
  
## 1.2. Base MIL Models
- **Models covered:**
  - Attention MIL (AttnMIL): Uses learned attention weights for instance aggregation.
  - TransMIL (Transformer MIL): Uses self-attention mechanisms, e.g., via Transformer layers.
  - Additive MIL: Sum of instance logits, inherently interpretable.
  
- **Implementation notes:**
  - **Foundation feature extractor:** Likely a pre-trained CNN (e.g., ResNet18 trained on ImageNet) for feature extraction from image patches.
  - **Bag-level classifier:** Linear or MLP, incorporating attention (attention scores as weights) or transformer self-attention weights.
  - **Input:** For each slide, extract patches, convert to features, and process via the model.

## 1.3. Explanation via xMIL-LRP
- **Layer-wise Relevance Propagation:**
  - Adapted to MIL: propagate relevance from the bag output back to individual instances and features.
  - **Relevance decomposition:**
    - Use specific propagation rules (e.g., epsilon-rule for linear layers, A- rule for attention modules, LN-rule for layer norms).
    - Handle the attention mechanism explicitly (via the AH-rule) to attribute relevance to instances based on attention weights.
  - **Hierarchy of relevance flow:**
    - From model output → attention weights → instance features → input features.
    - For transformers: relevance flow involves self-attention matrices (via the attention rollout method).
  
- **Positive/Negative Evidence:**
  - Relevance scores can be real-valued, supporting or opposing the class.
  - Instance relevance scores can be aggregated per instance's features to interpret supporting/refuting evidence.

## 1.4. Explanation Implementation Details
- **Relevance rules:**
  - Implement the epsilon rule for linear layers with ReLU.
  - Implement the AH-rule for attention modules, as detailed.
  - Implement the LN-rule for layer norms.
- **Instance relevance scores:**
  - Compute per-feature relevance and aggregate (sum over features) as instance evidence scores.
  - These scores form the basis for heatmaps/masks over input patches.

## 1.5. Generalization to Multiple Instance Learning
- **Assumptions:**
  - Each bag is a set of instances with features (or image patches).
  - The model’s prediction decomposes via relevance propagation into instance-level contributions.
  - Explanation scores are comparable across instances for identifying positive/negative evidence support.

## 1.6. Additional Explanation Techniques
- **Comparison methods:**
  - Gradient-based (e.g., IG, G×I)
  - Attention scores (raw attention, attention rollout)
  - Perturbation-based (single, one-removed, coalitions like MILLI)
  - Additive MIL insights (instance logits as explanations)
- **Implementation approach:**
  - Use existing frameworks like Captum for gradient methods.
  - Implement custom relevance rules for LRP adaptation.
  - For attentions, directly extract attention weights or rollouts.

---

# 2. Experiments

## 2.1. Toy Experiments (controlled synthetic data)
- **Datasets:**
  - Generate synthetic bags of MNIST instances, with specified class interactions:
    - 4-Bags: presence of specific digits (e.g., 8 supporting class1 etc.).
    - Pos-Neg: class labels based on counts of positive/negative instances.
    - Adjacent Pairs: class based on adjacency of digits.
- **Procedure:**
  - Create synthetic bags with known ground-truth evidence (support/refute/neutral).
  - Train MIL models (attn MIL, TransMIL, Additive MIL) to predict classes.
  - Explanation methods:
    - Apply xMIL-LRP, gradient methods, attention scores, perturbation.
  - **Evaluation:**
    - Use AUPRC-2, comparing explanation scores with ground-truth evidence.
    - Visualize relevance scores for instances.
  
- **Hyperparameters:**
  - Number of bags, instances, features.
  - Training epochs (300-1000), learning rate (~1e-3).
  - Repetitions for statistical stability.

## 2.2. Real Data – Histopathology
- **Datasets:**
  - CAMELYON16 (lymph node metastasis detection).
  - NSCLC (non-small cell lung carcinoma).
  - HNSC HPV status.
  - LUAD TP53 mutation.
- **Preprocessing:**
  - Extract patches (~256×256 px).
  - Use Otsu’s thresholding to exclude background.
  - Feature extraction:
    - Use a pre-trained CNN (ResNet18 or similar), frozen or fine-tuned.
    - Obtain a fixed feature vector per patch.
- **Bag construction:**
  - Sample a fixed number (e.g., 2048 patches); for large slides, subsample.
  - Validation: use all patches for inference to ensure stability.
- **Model training:**
  - For attention MIL: batch size 32, epochs 1000+, learning rate ~2e-3.
  - For TransMIL: batch size 5, epochs 200+, learning rate ~2e-4.
  - Use early stopping based on validation AUROC.
- **Explanation:**
  - Calculate relevance scores with xMIL-LRP.
  - Generate heatmaps over patches.
  - Compare with attention scores or other explanation methods.
- **Evaluation:**
  - Quantitative:
    - Use area under the perturbation curve (AUPC).
    - Use the AGREE metric across multiple baseline explanations.
  - Qualitative:
    - Visualize heatmaps on slide images.
    - Identify whether explanations align with known histological features.

## 2.3. Hyperparameters & Optimization
- **For training:**
  - Learning rate (grid search 1e-4 to 2e-3).
  - Dropout rates (0, 0.2, 0.5).
  - Number of feature vectors per slide (2048) or modified.
- **For explanations:**
  - Relevance propagation variants.
  - Thresholds for patch relevance significance.
  - Number of perturbation steps (up to 100) for AUPC.

## 2.4. Hardware & Resources
- Pretrained CNN feature extractor (e.g., ResNet18).
- MIL model implementations (attention, transformer, additive).
- GPU: NVIDIA A100 40GB or 80GB preferred for histopathology.
- CPU: adequate for large-scale inference (transMIL).

## 2.5. Reproducibility & Code Structure
- Use reproducible random seeds for data splits and training.
- Modular code:
  - Dataset preparation.
  - Feature extraction pipeline.
  - MIL model architectures.
  - Relevance propagation (`xMIL-LRP`) implementation.
  - Explanation metrics and visualization.
- Documentation:
  - Log hyperparameters.
  - Save checkpoints for best models.
  - Output explanations, heatmaps, and quantitative scores.

---

# Summary of the Roadmap

| Step | Description | Notes |
|---------|--------------|--------|
| 1. Data | Collect datasets, preprocess patches, extract features | Use sliding window, Otsu's threshold, pre-trained CNN |
| 2. Model | Implement MIL architectures (attention, transformer, additive) | Based on paper's descriptions, adopt existing frameworks for attention/transformers |
| 3. Training | Optimize models with early stopping, hyperparameter tuning | Based on dataset specifics, validation AUROC for best checkpoint |
| 4. Explanation | Implement xMIL-LRP with tailored relevance rules | Attention rule (AH-rule), epsilon-rule, LN-rule, relevance hierarchy |
| 5. Evaluation | Use AUPRC-2, AUPC, qualitative heatmap visualization | Conduct toy and real experiments, repeated runs for stability |
| 6. Documentation | Record hyperparameters, code dependencies, hardware | Make code modular and reproducible, provide detailed comments |

---

This outline provides a detailed, step-by-step blueprint for implementing everything—model, explanation, evaluation, and experimental protocols—faithful to the paper’s methodology. You can extend or modify specifics (like hyperparams or dataset splits) based on resources and project constraints during subsequent coding phases.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will construct a modular Python software system leveraging open-source libraries such as PyTorch for model implementation, TorchVision for feature extraction, Captum for relevance attribution, and standard data handling libraries (e.g., OpenSlide or tifffile) for histopathology data preprocessing. The system will include dataset loading, feature extraction, MIL model training (attention, transformer, or additive), relevance-based explanation via adapted xMIL-LRP, and evaluation with perturbation and heatmaps. We will use configurations for hyperparameters and model choices, and utilize PyTorch Lightning for clear training orchestration and reproducibility.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "feature_extractor.py",
        "model.py",
        "explanation.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "config.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Config {\n        +dataset_paths: dict\n        +hyperparameters: dict\n        +model_type: str\n        +explanation_method: str\n        +device: str\n    }\n    class DatasetLoader {\n        +__init__(config: Config)\n        +load_data() -> Dataset\n    }\n    class FeatureExtractor {\n        +__init__(model_name: str)\n        +extract_features(image_path: str) -> Tensor\n    }\n    class MILModel {\n        +__init__(model_type: str, params: dict)\n        +forward(features: Tensor) -> Tensor\n        +predict(bag: List[Tensor]) -> float\n    }\n    class Explanation {\n        +__init__(model: MILModel, explanation_method: str)\n        +compute_relevance(bag: List[Tensor]) -> List[float]\n        +generate_heatmap(bag: List[Tensor], relevance_scores: List[float]) -> Image\n    }\n    class Trainer {\n        +__init__(model: MILModel, dataset: Dataset, config: Config)\n        +train() -> None\n        +save_checkpoint(path: str) -> None\n    }\n    class Evaluation {\n        +__init__(model: MILModel, dataset: Dataset, explanation: Explanation, config: Config)\n        +evaluate() -> dict\n        +compute_AUPC() -> float\n        +visualize_heatmaps() -> None\n    }\n    Main --> DatasetLoader\n    Main --> FeatureExtractor\n    Main --> MILModel\n    Main --> Explanation\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> MILModel\n    Evaluation --> Explanation\n    Explanation --> MILModel\n    DatasetLoader --> Dataset\n    FeaturesExtractor --> Tensor\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant FE as FeatureExtractor\n    participant M as Model\n    participant E as Explanation\n    participant T as Trainer\n    participant EV as Evaluation\n    M->>DL: load_data()\n    DL-->>M: dataset object\n    M->>FE: initialize with model_name\n    Note over FE: load images, extract features\n    M->>FE: extract_features(image_path)\n    FE-->>M: feature Tensor\n    M->>Model: initialize with type and hyperparams\n    Note over Model: set model architecture\n    M->>T: train(model, dataset)\n    T->>Model: forward(features)\n    Model-->>T: predictions\n    T-->>M: training complete\n    M->>E: initialize explanation\n    Note over E: select explanation method (LRP, IG, attention rollout)\n    M->>E: compute_relevance(bag)\n    E-->>M: relevance scores\n    M->>E: generate_heatmap given relevance\n    E-->>M: heatmap image\n    M->>EV: evaluate(model, dataset, explanation)\n    EV->>Model: inference on test data\n    EV->>E: relevance and heatmap generation\n    EV-->>M: metrics (AUPRC, AUPC)\n    EV-->>Main: output results\n    Main: end\n",
    "Anything UNCLEAR": "Clarify whether the implementation should fully support multiple MIL model types (attention, transformer, additive) interchangeably, or if focus should be on one specific architecture. Also, confirm whether pre-trained feature extractors and explanation methods are to be pre-selected or configurable. Clarify if any custom visualization or interface components are needed beyond heatmaps. Lastly, specify if the data handling (slides, patches) will be provided or needs to be automated."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.0",
        "torchvision==0.13.0",
        "captum==0.4.0",
        "numpy==1.21.0",
        "scikit-learn==1.0.2",
        "openslide-python==1.1.2",
        "matplotlib==3.4.3",
        "tifffile==2021.11.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Acts as the entry point coordinating initialization of configs, data loading, model training, explanation, and evaluation. Depends on all other modules."
        ],
        [
            "config.py",
            "Contains all configuration parameters for data paths, hyperparameters, model selection, explanation method, and hardware options. Shared by all modules."
        ],
        [
            "dataset_loader.py",
            "Implements DatasetLoader class to load raw slide data, perform patch extraction (using OpenSlide), apply background filtering, and prepare data in a structured format. Dependent on data paths and preprocessing parameters from config.py."
        ],
        [
            "feature_extractor.py",
            "Implements FeatureExtractor class that loads a pre-trained CNN (ResNet18 from torchvision, frozen or fine-tuned) and extracts feature vectors from image patches. Depends on input image paths from dataset_loader and config for model choice."
        ],
        [
            "model.py",
            "Defines multiple model classes: AttentionMIL, TransMIL, AdditiveMIL depending on model_type in config. These models process patch features and output bag predictions. Depends on config parameters for architecture and hyperparameters."
        ],
        [
            "explanation.py",
            "Implements Explanation class that performs relevance attribution via xMIL-LRP, adapting the relevance rules for linear, attention, and norm layers. Depends on the trained model and model architecture, as well as explanation method setting in config."
        ],
        [
            "trainer.py",
            "Implements Trainer class that manages training loop, loss computation, optimizer setup, early stopping, and checkpoint saving. Depends on model.py and dataset_loader.py output features and dataset structures."
        ],
        [
            "evaluation.py",
            "Implements Evaluation class that computes metrics such as AUPRC-2, AUPC, and visualizes heatmaps from relevance scores. Relies on trained model, explanations, and test dataset. Uses matplotlib for visualization."
        ],
        [
            "utils.py",
            "Provides utility functions such as evaluation metric calculations, dataset split helpers, and plotting routines. Shared by trainer.py and evaluation.py."
        ]
    ],
    "Task list": [
        "config.py",
        "dataset_loader.py",
        "feature_extractor.py",
        "model.py",
        "explanation.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "",
    "Shared Knowledge": "Config.py centralizes all hyperparameters and paths; utils.py offers reusable functions for metrics and visualizations; the model.py uses standardized PyTorch modules. Explanation relies heavily on adapted LRP rules and explanation utilities in explanation.py.",
    "Anything UNCLEAR": "Clarification needed on the expected input dataset format—whether slides are pre-split into patches or need to be processed automatically. Also, confirm if pre-trained CNN should be frozen or fine-tuned. Hardware constraints for very large slide data handling (such as exceeding memory or patch count) also require clarification."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset_paths:
  histopathology: "/path/to/histopathology/data"
  toy_data: "/path/to/toy/dataset"

model_parameters:
  model_type: "attention"    # options: "attention", "transformer", "additive"
  hidden_dim: 512
  feature_extractor: "resnet18"  # pretrained CNN for feature extraction
  freeze_feature_extractor: true

training:
  learning_rate: 0.002
  batch_size: 32
  epochs: 1000
  optimizer: "Adam"
  dropout: 0.0

explanation_method:
  method: "xMIL-LRP"      # options: "xMIL-LRP", "IG", "G×I", "attention_rollout"
  relevance_rules:
    linear: "LRP-epsilon"
    attention: "AH-rule"
    layer_norm: "LN-rule"

hardware:
  device: "cuda"   # options: "cuda" or "cpu"
  gpus: 1

evaluation:
  perturbation_steps: 100
  metrics:
    AUPRC2: true
    AUPC: true
  visualization:
    heatmaps: true

save:
  model_checkpoint_path: "./checkpoints/"
  explanation_heatmaps_path: "./heatmaps/"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
