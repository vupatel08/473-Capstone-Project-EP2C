# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, detailed plan outlining how to reproduce the methodology and experiments described in the paper “CONTEXT-AWARE META-LEARNING (CAML).” The plan is structured into key sections: **Methodology (Model Architecture and Training), Dataset and Experimental Setup, Implementation Details, and Evaluation & Analysis. Each section details specifics based on the provided paper content, highlighting assumptions, clarifications needed, and potential code considerations.**

---

## 1. **Understanding & Reproducing the Methodology**

### 1.1 **Core Concept & Reformulation**
- The methodology emulates LLM in-context learning for vision: treat n-way-K-shot classification as sequence modeling over support examples + a query.
- The sequence consists of support images (support set) and a query image, encoded via a frozen pre-trained feature extractor.
- The sequence is fed into a **non-causal transformer** (our core architecture) that predicts the label of the query image.

### 1.2 **Model Architecture**
- **Feature Extractor (Encoder):**  
  - **Frozen pre-trained vision encoding network**—specifically, CLIP’s image encoder (e.g., ViT-B/16, ViT-H/14, or ViT-Huge).  
  - **Implementation note**: download or instantiate a CLIP backbone (e.g., via OpenAI’s CLIP or HuggingFace) with weights frozen during training.
  - **Output dimension**: 768 (ViT-B/16), 1024 (ViT-H/14), or larger for ViT-Huge.
- **Label Encoder (ELMES):**  
  - Learnable embedding for each class label.  
  - Dimension: 256 (per paper).  
  - Initialize as either learnable embeddings or initializations aligned with the class label (one-hot + embedding).  
  - Symmetries: structured to respect label symmetry (e.g., embeddings of classes are designed to be equiangular/tight frames, but initial implementation can start with randomly initialized learnable vectors, refined to be equiangular).
  
- **Sequence Model (Transformer):**  
  - Non-causal transformer (e.g., ViT-Large as in the paper).  
  - Input sequence: concatenated image embedding + label embedding sequence (support labels and possibly support images).  
  - Special tokens: positional embeddings (learnable) applied to sequence, and class label tokens aligned with support examples and query.
  - Output: the query token embedding’s prediction of class via inner-product with label embeddings.

### 1.3 **Logical flow of inference**
- Encode support images (support set) with frozen image encoder.
- Encode support labels with the ELMES (learnable label embeddings) based on class.
- Encode query image.
- Form sequence: support image embedding + support label embedding, with positional encodings.
- Append query image embedding as the last token.
- Feed entire sequence into transformer.
- Extract transformer output for the query token.
- Compute similarity (inner product) with each class label embedding (\(\phi_j\)) — the class with the highest similarity is predicted.

### 1.4 **Training objectives & properties**
- Loss: typically cross-entropy based on softmax over inner products of query token with label embeddings.
- No fine-tuning of the encoder; only train the transformer and label embeddings (ELMES).
- The key theoretical properties: permutation invariance, maximum angular separation (equiangular tight frame property), using the structure of label vectors.

### 1.5 **Additional details for implementation**
- Use support set images and labels to create a sequence of support image + label embedding pairs.
- For zero-shot inference, encode support set and query image, then predict label for the query.
- For few-shot, support images are limited (\(k \le 5\)), which is manageable in sequence length.

---

## 2. **Dataset & Experimental Setup**

### 2.1 **Datasets**
- **Universal Meta-Learning Benchmarks** include:
  - MiniImageNet, CIFAR-fs, tiered-ImageNet, Pascal VOC, Paintings, CUB, Aircraft, ChestX, etc.
  - These cover diverse domains, some natural images, some medical or art.
- **Special Dataset Requirements:**
  - Few-shot classification: for each task, sample support set and query set.
  - For experiments, implement episodic sampling:
    - 5-way, 1-shot or 5-shot per dataset.
    - Sample classes, then support images per class, then query images.
- **Preprocessing:**
  - Resize/crop images as required by CLIP or other encoders.
  - Normalize images according to the pre-trained encoder’s expectations.

### 2.2 **Experimental Protocols**
- **Universal setting:** evaluate models cross-dataset:
  - No meta-training on the target dataset tasks; zero-shot or in-context style inference.
  - Use large-scale pretraining on ImageNet, COCO, WikiArt, fungi, etc.
- **Training:**
  - Large-scale pretraining on multiple datasets for the sequence model (transformer + label embeddings).
  - No meta-training on specific few-shot tasks; just pretrain on datasets in a supervised, episodic manner.
  - Loss: cross-entropy over support set + query support sequence.
- **Evaluation:**
  - Test on unseen classes/tasks (from the same dataset distribution or transfer).
  - Metrics: mean accuracy over multiple episodes, with standard error.
- **Hyperparameters:**
  - Learning rate schedule: cosine decay with warmup (see paper: 9,600 steps, 1e-5 -> 1e-6).
  - Batch size: large (around 525 per epoch; support + query).
  - Optimizer: Adam or AdamW.
  - No hyperparameter tuning—use defaults to replicate claims.

---

## 3. **Implementation Details & Practical Considerations**

### 3.1 **Model Components**
- **Encoders:**
  - Use pre-trained CLIP models (via HuggingFace or OpenAI, with freeze=True).
  - Support image embedding: tensor shape `[batch_size, embedding_dim]`.
- **Label embeddings (ELMES):**
  - Implement as a learnable embedding matrix `[number_of_classes, 256]`.
  - Initialize randomly, train via gradient descent.
- **Sequence construction:**
  - For each task:
    - Support set: concatenate support image embedding and class label embedding.
    - Append query image embedding as the last token.
    - Embed positionally (learned positional embeddings).
  - Sequence shape: `[sequence_length, embedding_dim]`.
- **Transformer:**
  - Use existing Vision Transformer implementations, adapt for sequence modeling (not causal).
  - Inputs: sequence of embeddings + positional encodings.

### 3.2 **Training Loop**
- Sample a batch of tasks (e.g., episodes):
  - For each task:
    - Encode support images.
    - Form sequence with support image + class label.
    - Encode query image.
    - Append query to sequence.
  - Forward through transformer.
  - Compute similarity with label embeddings for query token.
  - Cross-entropy loss.
- Backpropagate only transformer weights + label embeddings.
- Repeat over large epochs with dataset composition as described.

### 3.3 **Permutation & Symmetry**
- Sequence is formed in arbitrary order; verify permutation invariance by data augmentation (shuffling support set order).
- Special care in label encoder to ensure label symmetry—initially with learnable vectors, then possibly impose the equiangular structure.

### 3.4 **Optimization & Regularization**
- Use AdamW with default or schedule as per paper.
- No dropout, attention dropout, or weight decay.
- Batch size: large (~525) for stable training, distributed training recommended.
- Learning rate schedule: warmup + cosine decay over ~400k steps.

### 3.5 **Other Implementation Notes**
- The total number of parameters (~300 million) is large; use efficient hardware (GPUs or TPUs).
- If necessary, start with smaller transformer size and scale up.
- Freeze encoders during training.
- Save label embedding state separately to analyze class structure (equiangular properties).

---

## 4. **Evaluation & Analysis**

### 4.1 **Metrics**
- **Accuracy:** mean over episodes.
- **Standard error:** across episodes.
- **Universal generalization:** evaluate model on unseen datasets/classes without fine-tuning.
- **Permutation invariance check:** shuffle support set order, verify performance stability.
- **Label symmetry & angular separation:** compute inner products/angles among label embeddings post-training.

### 4.2 **Theoretical Validation**
- Verify the properties of learned label embeddings agree with the paper:
  - Are they close to equiangular tight frames?
  - Do they maximize angular separation?
- Use t-SNE visualizations of class embeddings (e.g., Fig 4) to interpret support set structure.

### 4.3 **Ablations & Sensitivity Tests**
- Vary support set size (1-shot vs 5-shot).
- Evaluate with different pretrained encoders (CLIP ViT-B/16, ViT-H/14, ViT-Huge, ViT-Base, CLIP from different datasets).
- Test permutation invariance by shuffling support sequences.
- Test features extracted from encoders trained/finetuned on different datasets.

---

## 5. **Open Questions & Assumptions**
- Exact architecture details (number of transformer blocks, heads, dimensions) are inferred from the paper and appendix; choices may vary.
- The paper emphasizes theoretical properties (equiangular tight frames); initial implementation may relax these, then analyze embedding structure.
- Hyperparameters such as support set size, sequence length, and optimizer specifics should follow the paper’s guided parameters or be tuned if resources permit.
- Implementation of the permutation invariance and label symmetry constraints: initial random label embeddings, then enforce via regularization or structured initialization.

---

## 6. **Summary Roadmap**
- **Step 1:** Download or implement CLIP (frozen), process datasets, and sample episodic tasks.
- **Step 2:** Design the sequence: support image + label embedding, append query image.
- **Step 3:** Implement non-causal transformer with positional embeddings.
- **Step 4:** Initialize and train label embeddings and transformer on large-scale pretraining (using support image + label sequences).
- **Step 5:** Evaluate on unseen datasets/tasks, analyze label embeddings' structure, perform permutation tests.
- **Step 6:** Perform ablations to confirm properties (support size, encoder choice).

---

This outline provides a clear, detailed blueprint for reproducing the CAML approach, covering all major components, data protocols, and theoretical considerations for implementation, evaluation, and analysis. You can now translate this plan into specific code blocks, starting with data loading, model instantiation, training scripts, and evaluation routines.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "The system will leverage open-source libraries such as HuggingFace Transformers for models, OpenAI's CLIP model implementation for image encoding, and PyTorch for core training and inference routines. The main architecture includes a dataset loader for episodic tasks, a model class implementing the non-causal transformer with support for support query sequence formation, label (ELMES) embedding management, and a training class that orchestrates large-scale pretraining and in-context evaluation. The approach involves freezing CLIP encoders, training only the transformer and label embeddings, and evaluating on unseen few-shot tasks with permutation invariance testing.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__()\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> Dataset\n        +sample_task(way: int, shot: int) -> Task\n    }\n    class Model {\n        +__init__(model_params: dict)\n        +encode_image(image: Image) -> Tensor\n        +predict_support_sequence(sequence: Tensor) -> Tensor\n        +predict_query(query_token: Tensor, support_labels: Tensor) -> int\n    }\n    class LabelEmbeddings {\n        +__init__(num_classes: int, embedding_dim: int)\n        +get_embeddings() -> Tensor\n        +update_embeddings() -> None\n    }\n    class Trainer {\n        +__init__(model: Model, dataset_loader: DatasetLoader, config: dict)\n        +pretrain() -> None\n        +train_one_episode(episode: Task) -> float\n        +train() -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, dataset_loader: DatasetLoader, config: dict)\n        +evaluate_on_dataset() -> dict\n        +test_permutation_invariance() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> Model\n    Model --> LabelEmbeddings\n    Evaluation --> Model\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: initialize(config)\n    DL-->>M: return dataset loader\n    M->>TR: create(model, dataset_loader, config)\n    TR->>TR: pretrain() on large-scale dataset\n    Note right of TR: large-scale pretraining, no finetuning on support sets\n    loop for each episode in evaluation\n        DL->>DL: sample_task(way, shot)\n        DL-->>M: return task with support & query images\n        TR->>TR: prepare sequence support images + label embeddings\n        TR->>TR: forward sequence and predict query label\n        TR-->>M: loss, accuracy\n    end\n    M->>EV: initialize(model, dataset_loader, config)\n    EV->>EV: evaluate on new unseen tasks\n    EV-->>M: report metrics (accuracy, permutation invariance)\n",
    "Anything UNCLEAR": "Clarify the expected dataset input formats (support/query support sets structure, label encoding), and confirm if pretraining should be on multiple datasets or solely on ImageNet variants. Also, specify environment constraints (hardware, compute budget)."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.0",
        "torchvision",
        "transformers==4.26.0",
        "timm",
        "pillow",
        "numpy",
        "scipy"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Entry point that initializes configuration, loads dataset, sets up training and evaluation, and manages the overall workflow. It depends on dataset_loader.py for data, trainer.py for training, and evaluation.py for evaluation. Coordinates sequence flow: load data -> train model -> evaluate."
        ],
        [
            "dataset_loader.py",
            "Defines DatasetLoader class with methods: __init__(config), load_data(), sample_task(way, shot). Responsible for loading datasets, providing episodic tasks, and dataset-specific preprocessing. Supports multi-dataset setup and sampling support/query sets for few-shot tasks."
        ],
        [
            "model.py",
            "Defines Model class with methods: __init__(parameters), encode_image(image), prepare_sequence(support_images, support_labels, query_image), predict(query_token, support_labels). Implements frozen CLIP encoder, label embeddings (ELMES), and transformer model. Handles sequence construction according to support/query setup, including positional encodings."
        ],
        [
            "trainer.py",
            "Contains Trainer class with methods: __init__(model, dataset_loader, config), pretrain(), train_episode(task), train(). Handles large-scale pretraining on multiple datasets, manages optimizer, learning rate schedule, and gradient updates. Calls model.encode_image and manages training data batching."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class with methods: __init__(model, dataset_loader, config), evaluate_on_dataset(), test_permutation_invariance(). Responsible for zero-shot or few-shot evaluation across datasets, including permutation tests for invariance, collects metrics, and generates detailed reports."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "None required, the system is designed as a command-line pretraining and evaluation pipeline with class-based interfaces.",
    "Shared Knowledge": "The Model class and DatasetLoader class both share the support/query sample handling. The LabelEmbeddings (ELMES) are part of model.py and are also initialized in main.py. DatasetLoader shares dataset paths and sampling strategies with the trainer.",
    "Anything UNCLEAR": "Clarify specific dataset format requirements, whether support/query images are preprocessed or raw, and details on hyperparameters such as transformer size, dropout, and training length for large-scale pretraining experiments."
}

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 1e-5
  warmup_steps: 9600
  total_steps: 400000
  cosine_decay: true
  batch_size: 525
  support_shot: 5
  way: 5
  evaluation_episodes: 1000
  eval_support_shot: 5
  eval_way: 5
model:
  image_encoder: "openai/clip-vit-base-patch32"
  image_encoder_trainable: false
  label_embedding_dim: 256
  transformer_model_name: "facebook/vit-large-patch16-224"
  transformer_params:
    num_layers: 24
    num_heads: 16
    hidden_dim: 1024
    dropout: 0.0
dataset:
  datasets:
    - name: "mini-ImageNet"
      split: "train"
    - name: "CIFAR-fs"
      split: "train"
    - name: "tiered-ImageNet"
      split: "train"
    - name: "Pascal VOC"
      split: "train"
    - name: "Paintings"
      split: "train"
evaluation:
  episodes: 1000
  support_shot: 5
  way: 5
  permutation_test_episodes: 1000
  datasets:
    - name: "mini-ImageNet"
    - name: "CIFAR-fs"
    - name: "tiered-ImageNet"
    - name: "Pascal VOC"
    - name: "Paintings"
```


---

**Note:** Full configuration is available in `planning_config.yaml`
