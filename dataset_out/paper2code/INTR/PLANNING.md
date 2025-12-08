# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, detailed plan to reproduce the INTR method from the paper. The plan is structured into key components that align with the methodology, experimental setup, hyperparameters, and evaluation, providing clear actions for implementation.

---

## I. Overview and Core Concept
- **Goal:** Implement the Interpretable Transformer (INTR), which uses class-specific queries and cross-attention to produce both accurate and interpretable image classification.
- **Key Innovation:** Unlike traditional classifiers, INTR extracts class-specific features directly via learned queries, with the cross-attention maps serving as faithful explanations.
- **Main Components in Implementation:**
  - Backbone feature extractor (pre-trained and possibly fine-tuned)
  - Class-specific learnable queries
  - Multi-head cross-attention modules
  - Classifier (inner product with class-wise weights)
  - Extraction of attention maps for interpretability
  - Training with standard cross-entropy loss

---

## II. Data and Datasets
- **Datasets Required:**
  - Fine-grained datasets (e.g., CUB-200-2011, Birds-525, Oxford Pets, Stanford Dogs, FGV-Aircraft, etc.)
  - Dataset statistics for hyperparameter setup and batch sizing:
    - Number of classes
    - Number of images (train/test)
    - Specific class labels
- **Preparation Steps:**
  - Download and preprocess datasets (resize/crop as necessary)
  - Organize data into train/test splits
  - For interpretability: ensure annotated attributes (if available) to evaluate faithfulness, but optional for training

---

## III. Model Architecture & Components
### 1. Backbone Feature Extractor
- Use a **pre-trained Vision Transformer (ViT)**, e.g., ViT-H-21K, ViT-Base, or ResNet-based backbone as in experiments.
- **Implementation details:**
  - For the ViT, extract feature maps (spatial token embeddings): shape `[H, W, D]` (or flattened `[N, D]`)
  - For ResNet, extract features from the last convolutional layer, possibly with some adaptation (resizing to match ViT feature shapes).

### 2. Class-specific Queries
- Learn `C` class-specific query vectors `Z_in` in `R^D`.
- Initialization:
  - Random or Xavier initialization.
  - Shape: `[D, C]`.
- These are the input to the decoder cross-attention layers.

### 3. Multi-Head Cross-Attention Modules
- Implement `M` (e.g., 4 or 8) parallel multi-head cross-attention blocks, following the Transformer decoder structure.
  - Each block:
    - Inputs: Class query vectors (`[D, C]`) and feature map tokens (`[D, N]`).
    - Outputs: Encoded class-specific feature vectors (`[D, C]`).
    - Each head within the attention:
      - Learn projection matrices `W_q`, `W_k`, `W_v` (`[D, D]` each).
      - Compute scaled dot-product attention:
        \[
        \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{K^\top Q}{\sqrt{D}}\right) V
        \]
  - Multi-heads:
    - For each head, generate separate projections.
    - Concatenate head outputs for each class.
- Stack multiple decoder layers, allowing class queries to refine their focus with self-attention (if implemented).

### 4. Final Classification Layer
- Use a *shared* learnable class weight matrix `W_w` (`[D, C]`).
- Compute logits as:
  \[
  \hat{y}_c = \arg \max_{c} ( W_w[:, c]^\top Z_{out}^{(c)} )
  \]
- Loss: standard cross-entropy over the predicted class scores.

### 5. Attention Map Extraction for Interpretability
- For each class:
  - Use the softmax of the attention weights to obtain the importance of each spatial token.
  - Visualize these maps over the input image (upsampled to image size).

### 6. Multi-head Attention & Layer Configurations
- Use multiple heads (e.g., 4 or 8) per layer.
- Use at least 2-4 decoder layers for refinement.
- For each experiment, tune:
  - Number of heads
  - Number of decoder layers
  - Embedding dimension `D`

---

## IV. Implementation Details & Hyperparameters
- **Feature extractor:**
  - Pre-trained ViT or ResNet, optionally fine-tuned during training
- **Input resolution:**
  - For ViT: 224x224 or 384x384 depending on the backbone
- **Batch size:**
  - 16 or 32 (depending on resources)
- **Learning rate:**
  - Start with 1e-4 to 3e-4
  - Use Adam or AdamW optimizer
- **Number of epochs:**
  - 50–100 epochs with early stopping as needed
- **Optimizer settings:**
  - Weight decay (e.g., 0.05)
  - Learning rate decay or scheduler (e.g., cosine annealing)
- **Loss:**
  - Cross-entropy over class logits
- **Regularization:**
  - Dropout in attention layers if needed
  - Weight decay
- **Evaluation metrics:**
  - Top-1 accuracy
  - Faithfulness (contextually, insertion/deletion scores)
  - Attribute localization quality (if attribute annotations available)
  
---

## V. Training Procedure
1. Initialize all network components:
   - Backbone (pre-trained, fixed or fine-tuned)
   - Class-specific queries (`Z_in`)
   - Projection matrices (`W_q`, `W_k`, `W_v`)
   - Class weight matrix (`W_w`)
2. For each epoch:
   - For each batch:
     1. Extract features from images via backbone/trunk
     2. For each class, pass `Z_in` as queries to cross-attention layers
     3. Multiple decoder layers refine class-specific features
     4. Compute class logits via inner product with `W_w`
     5. Calculate cross-entropy loss with ground truth
     6. Backpropagate and update parameters
3. Validation:
   - Measure accuracy
   - Save best models based on validation accuracy
   - Save attention maps for interpretability

---

## VI. Evaluation and Interpretability
- **Faithfulness:**
  - Use insertion/deletion metrics comparing attention maps to attribute locations.
  - Perturb regions of the image based on attention map relevance.
- **Qualitative analysis:**
  - Overlay attention maps on images.
  - Verify whether the attention heads focus on meaningful object parts or attributes.
- **Quantitative analysis:**
  - Attribute-based accuracy (e.g., attribute localization accuracy if attribute annotation available).
  - Compare with post-hoc methods like Grad-CAM, RISE.
  
---

## VII. Additional Notes & Considerations
- **Configuring Queries:**
  - Random initialization is fine, but consider label-initialized or attribute-informed queries if data allows.
- **Number of Decoder Layers and Heads:**
  - More heads improve attribute disentanglement but increase complexity.
  - Typically 4 layers with 4–8 heads as per experiments.
- **Implement Multi-headed Attention:**
  - Standard in PyTorch (`nn.MultiheadAttention`) or custom.
- **Scaling to Larger Datasets:**
  - With larger datasets, increase epochs or fine-tune backbone
  - Consider transfer learning strategies
  
---

## VIII. Summary of Key Implementation Steps
- Set up backbone feature extractor (pre-trained ViT or ResNet)
- Define learnable class queries (`Z_in`)
- Build multi-head cross-attention decoder layers
- Incorporate residual/self-attention if refined training
- Use class-specific weights (`W_w`) for classification
- Train end-to-end with standard cross-entropy
- Extract and visualize attention maps for interpretability
- Conduct thorough evaluation (accuracy and faithfulness)

---

This roadmap should enable you to design, implement, and reproduce the INTR model as described. Each step can be expanded into detailed code modules during actual coding, ensuring fidelity to the original paper’s methodology and experimental settings.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular deep learning system utilizing PyTorch and torchvision. The system will include a backbone feature extractor (pre-trained Vision Transformer), a multi-head cross-attention decoder with class-specific queries, and a classifier layer. The core components will be encapsulated in classes: one for dataset loading, one for the model architecture, one for training, and one for evaluation with interpretability. Attention maps will be extracted directly from the cross-attention weights for interpretability. The system is designed to be end-to-end trainable with standard optimizers and losses, and all components are configurable via a simple YAML or JSON configuration file.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str, batch_size: int)\n        +load_data() -> Dict[str, Any]\n    }\n    class INTRModel {\n        +__init__(config: dict)\n        +forward(images: Tensor) -> Tuple[Tensor, Tensor]\n        +get_attention_maps() -> List[Tensor]\n        +predict() -> Tensor\n    }\n    class Trainer {\n        +__init__(model: INTRModel, data_loader: DatasetLoader, optimizer: torch.optim.Optimizer, loss_fn: callable)\n        +train_epoch() -> None\n        +train() -> None\n    }\n    class Evaluator {\n        +__init__(model: INTRModel, data_loader: DatasetLoader)\n        +evaluate() -> dict\n        +visualize_attention_maps(attention_maps: List[Tensor], images: Tensor) -> None\n    }\n    Main --> DatasetLoader\n    Main --> INTRModel\n    Main --> Trainer\n    Main --> Evaluator\n    Trainer --> INTRModel\n    Evaluator --> INTRModel\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant Mdl as INTRModel\n    participant Tr as Trainer\n    participant Ev as Evaluator\n    M->>DL: load_data()\n    DL-->>M: dataset_dict\n    M->>Mdl: initialize(config)\n    M->>Tr: start training(model, dataset, optimizer, loss_fn)\n    Tr->>Mdl: forward(images)\n    Mdl-->>Tr: class_logits, attention_maps\n    Tr-->>M: training complete\n    M->>Ev: evaluate_model(model, dataset)\n    Ev->>Mdl: forward(images)\n    Ev-->>M: metrics, attention_maps\n    Ev->>M: visualize attention maps (attention_maps, images)\n    Note over M, Ev: Loop over epochs and datasets as configured\n",
    "Anything UNCLEAR": "Clarify whether the feature extractor backbone is fixed or fine-tuned, and specify the desired number of decoder layers and attention heads. Confirm dataset input formats, attribute annotations, and whether specific attribute supervision is available for interpretability evaluation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "torchvision==0.12.0",
        "numpy==1.21.0",
        "yaml==5.4.1"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class that handles loading datasets, applying necessary transformations, batching, and providing data in dicts. Depends on torchvision.datasets and torch.utils.data for dataset and DataLoader. Responsible for preparing training/testing splits and returning datasets with labels and images."
        ],
        [
            "model.py",
            "Encapsulates INTRModel class which integrates backbone feature extractor (pre-trained ViT or ResNet), class-specific queries, multi-head cross-attention modules, residual/self-attention blocks, and final classifier. Implements the forward pass, extraction of attention maps, and prediction functions. Depends on torch.nn, torch.nn.functional, and appropriate transformer modules."
        ],
        [
            "trainer.py",
            "Defines Trainer class responsible for training loop. Handles optimizer setup, loss computation, backpropagation, and learning rate scheduling. Uses dataset_loader output for batching; communicates with INTRModel for forward passes and attention map extraction. Focused on optimizing model parameters end-to-end."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class with methods for assessing model performance on validation/test data. Computes accuracy, generates attention maps, and overlays these on images for interpretability. Depends on model.py for predictions and attention maps, and matplotlib or similar for visualization."
        ],
        [
            "main.py",
            "Application entry point that reads configuration files, initializes DatasetLoader, INTRModel, Trainer, and Evaluation instances, orchestrates training across epochs, performs validation, saves best models, and calls visualization functions for interpretability analysis. Coordinates overall flow and handles command-line interface."
        ],
        [
            "utils.py",
            "Provides shared utility functions such as configuration file parsing, seed setting for reproducibility, visualization helpers (overlaying attention maps), and any common tensor manipulation functions. Used throughout other modules."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "utils.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\ncomponents:\n  schemas:\n    Dataset:\n      type: object\n      properties:\n        train_loader:\n          type: object\n          description: DataLoader for training data\n        test_loader:\n          type: object\n          description: DataLoader for test data\n        class_labels:\n          type: array\n          items: string\n          description: List of class labels\n    Prediction:\n      type: object\n      properties:\n        logits:\n          type: array\n          items: number\n        attention_maps:\n          type: array\n          items: object\n    # Further endpoints would handle model training, evaluation, and visualization as needed.",
    "Shared Knowledge": "Shared utility functions include configuration parsing, tensor visualization overlays, attention map normalization, and reproducibility setup. The model training pipeline depends on standard PyTorch API with clear separation of feature extraction and classifier modules. Dataset loader depends on torchvision datasets with standard transforms.",
    "Anything UNCLEAR": "Clarify whether the backbone should be fixed or fine-tuned during training. Confirm exact dataset formats and attribute annotations if available for interpretability evaluation. Specify preferred ground-truth attribute labels or evaluation metrics for evaluating faithfulness of attention maps."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: "CUB-200-2011"
  train_split: "path/to/train_split"
  test_split: "path/to/test_split"
  image_size: 224  # Match the dataset's standard resize
  batch_size: 16    # As suggested for typical training with GPU resources
  
model:
  backbone: "vit"
  pretrained_weights: "path/to/pretrained/vision_transformer.pth"
  embed_dim: 768        # Typical for ViT-Base
  num_heads: 4
  num_layers: 4
  class_queries: 200    # Number of classes, from dataset
  query_dim: 768        # Same as embed_dim
  
training:
  learning_rate: 0.0003    # Based on common transformer training practices
  batch_size: 16
  epochs: 50              # As used in experimental setup
  weight_decay: 0.05
  optimizer: "AdamW"
  scheduler: "cosine_annealing"
  save_dir: "outputs/checkpoints"
  device: "cuda"        # Use 'cpu' if no GPU available
  
loss:
  type: "cross_entropy"

attention:
  heads: 4
  decoder_layers: 4
  attention_type: "multi-head cross-attention"
  attention_map_size: "depends on feature map resolution; typically 14x14 or 16x16 depending on backbone"
  
interpretability:
  save_attention_maps: true
  visualization_overlay: true
  evaluation_metrics: ["faithfulness", "accuracy"]
  
misc:
  seed: 42
  random_crop: true
  use_fully_finetune_backbone: true   # finetune backbone for performance
```

---

**Note:** Full configuration is available in `planning_config.yaml`
