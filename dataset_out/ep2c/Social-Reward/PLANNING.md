# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Below is a detailed, structured plan that synthesizes the key methodological, experimental, and evaluation aspects from the paper. This roadmap is designed to ensure fidelity to the original work while highlighting critical implementation details. You can use this as a blueprint for incremental code development.

---

## 1. Data Collection & Dataset Construction

### a. Underlying Data Sources
- **Primary Data Source:** Picsart platform user interactions involving image editing and creative activities.
- **Secondary Data Sources for Prompt Generation:**
  - Prompts from publicly available datasets (e.g., COCO Captions, DiffusionDB, etc.).
  - Additional prompts crafted by platform creators or derived from prompt generation models (e.g., Prompt Engineering).
- **Target Data:** Implicit user preferences from over one million users, with annotations on which images are favored (e.g., via user engagement signals).

### b. Dataset Curation Steps
- **Implicit Feedback Extraction:**
  - Identify user actions such as likes, shares, comment counts, and image re-mixes.
  - Assign preference labels: e.g., images with positive engagement signals are labeled as “positive”, others as “negative”.
  - For robust annotations, normalize engagement signals (e.g., normalize likes/views).
- **Prompt-Image Pairing:**
  - Use platform prompts (user-authored, system-generated, or derived prompts from datasets).
  - Limit prompts to top 5 words.
  - Exclude prompts with no associated images or low engagement.
  - Ensure prompt-image pairs are diverse and cover both popular and non-popular categories.
- **Filtering & Debiasing:**
  - Remove pairs with very low view counts or ambiguous signals.
  - Filter out images with no clear signal of community preference.
  - Remove noise by filtering out images that are heavily distorted or irrelevant.

### c. Prompt Engineering & Clustering
- **Prompt Clustering Techniques:**
  - Use hierarchical clustering (Ward’s method) on prompt embeddings (from CLIP text encoder).
  - Determine optimal number of clusters (~173) based on silhouette scores to capture thematic differences.
  - Assign each prompt to a cluster to analyze distribution shifts.
  
### d. Dataset Stats & Visualization
- **Sample Visualizations:**
  - Distribution of prompt clusters (e.g., via t-SNE visualizations).
  - Example images ranked by Social Reward (from best to worst).
- **Summary Stats:**
  - Number of total pairs, positive vs negative, cluster purity measures.
  - Distribution analysis of user engagement signals.

---

## 2. Model Architecture & Training Pipeline

### a. Base Models & Encoders
- **Select Foundation Models:**
  - CLIP (OpenAI) or BLIP as backbone for text and image encoding.
  - Use both visual (image encoder) and textual (prompt encoder) components.
- **Implementation Approach:**
  - Use pre-trained CLIP/BLIP models from HuggingFace or OpenAI.
  - Frame as a metric-learning scenario with cosine similarity.

### b. Social Reward Model
- **Design:**
  - Input: Prompt and Image pair (each encoded via CLIP or BLIP encoders).
  - Output: Scalar Social Reward score estimating community preference.
- **Loss Function:**
  - Fine-tune using a **pairwise ranking loss**—e.g.,
    \[
    \mathcal{L}_{triplet} = \max(0, \|\textbf{a} - \textbf{p}\|^2 - \|\textbf{a} - \textbf{n}\|^2 + \alpha)
    \]
  where:
    - \(\textbf{a}\): prompt embedding,
    - \(\textbf{p}\): positive image embedding,
    - \(\textbf{n}\): negative image embedding,
    - \(\alpha\): margin (e.g., 0.2–0.5).
- **Training Details:**
  - Optimizer: AdamW
  - Learning rate: 3e-4
  - Batch size: 32
  - Distributed setup: 8 A100 GPUs
- **Sample Construction:**
  - For each prompt, select the top-ranked (by Social Reward) positive image and a negative image from the same cluster or randomly.
  - Data augmentation may be applied to images to improve generalization.

### c. Hyperparameters & Fine-tuning
- **Learning Rate & Scheduling:**
  - Use a fixed learning rate or cosine annealing.
- **Model Components:**
  - Fine-tune only the final layers of CLIP or BLIP encoders, or employ LoRA/adapter modules for efficiency.
- **Validation:**
  - Use a held-out validation set to monitor pairwise accuracy and avoid overfitting.

---

## 3. Evaluation & Metrics

### a. Dataset for Evaluation
- **Test Set Construction:**
  - Use a subset of prompt-image pairs with known preference labels (from implicit feedback evaluation or manual annotation).
  - Ensure diversity across prompt clusters and engagement levels.

### b. Quantitative Metrics
- **Pairwise Accuracy:**
  - Based on cosine similarity rankings between prompt and image embeddings.
  - Calculate percentage of pairs where the model’s preferred image is ranked higher.
- **Comparison Benchmarks:**
  - Evaluate against baseline metrics:
    - CLIP score
    - HPS v2
    - ImageReward
    - PickScore
  - Record accuracy scores (e.g., Social Reward achieves ~69.7%).

### c. Qualitative & Visual Assessment
- **Ranking Visualization:**
  - Reproduce Figures 6 & 9: display top-ranked vs bottom-ranked images according to Social Reward.
  - Conduct human judgment on whether the ranking aligns with community preferences.

### d. Additional Analyses
- **Cluster-wise Validation:**
  - Use t-SNE to verify prompt-embedding separability across models.
- **Ablation Studies:**
  - Compare model performance with different backbone encoders.
  - Evaluate the effect of prompt clustering granularity.

---

## 4. Fine-tuning Existing Generative Models

### a. Model Fine-tuning
- **Target Models:**
  - Stable Diffusion or other customizable text-to-image models.
- **Approach:**
  - Use Social Reward as a reward signal in a reinforcement learning framework (e.g., RLHF).
  - Fine-tune the generative model to maximize Social Reward scores.
  - Use prompt-image pairs with high Social Reward to retrain or adapt the generator (see Section 4 of paper).

### b. Training Setup
- **Method:**
  - Reinforcement learning with the Social Reward as the reward function.
  - Optimizer: AdamW or custom RL algorithms (e.g., PPO).
  - Hyperparameters adapted per model size.
- **Evaluation:**
  - Generate images for prompt clusters.
  - Rank images by Social Reward vs baseline to observe improvements.

### c. Prompt-Image Sampling & Ranking
- Generate paired images:
  - For a set of prompts, produce multiple images via the fine-tuned model.
- Score ranking:
  - Use Social Reward to rank images.
  - Compare against baseline model in visual quality and preference correlation.

---

## 5. Implementation & Validation

### a. Development Environment
- Frameworks: PyTorch, HuggingFace Transformers, OpenAI CLIP, diffusers, TensorFlow (if needed).
- Hardware: 8x A100 GPUs for training; multi-GPU setup for distributed training.

### b. Reproducibility & Logging
- Use consistent seed initialization.
- Log training metrics, validation accuracy, and sample rankings (via tensorboard or WandB).
- Save checkpoints of fine-tuned models.

---

## 6. Summation of Critical Points
- **Datasets:** Assemble large-scale implicit feedback dataset with prompt-image pairs, filtering, and clustering.
- **Model:** Fine-tune cross-modal encoders with pairwise triplet loss, emphasizing social signals.
- **Evaluation:** Pairwise accuracy, comparison with existing metrics, and human validation.
- **Fine-tuning:** Use Social Reward in a reinforcement loop to improve generative models aligned with community preferences.
- **Visualization & Analysis:** t-SNE, ranking figures, and implicit feedback distribution plots.

---

This roadmap provides a comprehensive guide to implement the Social Reward framework, replicate core experiments, evaluate the models, and prepare for fine-tuning generative models. It ensures methodological fidelity and paves the way for practical coding phases.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Build a streamlined pipeline leveraging pre-trained CLIP models; fine-tune using triplet loss on a large curated dataset of prompt-image pairs with implicit social preference feedback. The system handles data loading, preprocessing, model training, and evaluation with open-source libraries like HuggingFace Transformers, PyTorch, and Datasets. Use a simple orchestrator script to manage data flow, training, and validation, with a focus on pairwise ranking accuracy. Integrate visualization tools for prompt clustering (t-SNE) and ranking visualization to analyze model behavior.",
    "File list": [
        "app.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment() -> None\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str, prompt_cluster_data: str)\n        +load_data() -> Tuple[Dataset, Dataset]\n        +get_prompt_pairs() -> List[Tuple[dict, Tensor, Tensor]]\n    }\n    class Model {\n        +__init__(pretrained_model_name: str)\n        +encode_prompt(prompt: str) -> Tensor\n        +encode_image(image_path: str) -> Tensor\n        +compute_score(prompt_embedding: Tensor, image_embedding: Tensor) -> float\n        +load_weights(path: str) -> None\n        +save_weights(path: str) -> None\n    }\n    class Trainer {\n        +__init__(model: Model, train_data: List[Tuple[dict, Tensor, Tensor]], val_data: List[Tuple[dict, Tensor, Tensor]], config: dict)\n        +train() -> None\n        +save_checkpoint(path: str) -> None\n        +load_checkpoint(path: str) -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, test_data: List[Tuple[dict, Tensor, Tensor]])\n        +evaluate() -> dict\n        +calculate_pairwise_accuracy() -> float\n        +visualize_ranking() -> None\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    DatasetLoader --> Dataset\n    Trainer --> Model\n    Evaluation --> Model\n    Model --> pre-trained encoders\n    Trainer --> utils for logging and visualization\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: Initialize dataset loader with dataset path and prompts\n    DL-->>M: Load dataset + prompt clusters\n    M->>MD: Initialize model with pre-trained CLIP/BLIP\n    MD-->>M: Loaded model\n    M->>TR: Instantiate trainer with model and training dataset\n    TR->>TR: train() on triplet loss using social preference signals\n    TR-->>M: Save trained model checkpoint\n    M->>EV: Load trained model\n    EV->>EV: evaluate() on test set; compute metrics\n    EV-->>M: Output metrics and rankings\n    M->>EV: visualize_ranking() for prompt/image rankings\n",
    "Anything UNCLEAR": "Clarify the exact format and structure of the curated dataset, especially how social preference signals are embedded, and confirm if manual annotation or additional filters are needed. Additionally, specify if there is a preferred open-source visualization library for clustering and ranking dashboards."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.1",
        "transformers==4.28.0",
        "datasets==2.4.0",
        "scikit-learn==1.2.2",
        "matplotlib==3.5.3",
        "t-SNE==0.2.1",
        "sentence-transformers==2.2.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class responsible for: \n- Loading prompt-image preference data from curated files\n- Handling prompt clustering with hierarchical clustering (e.g., Ward's method)\n- Splitting data into train, validation, and test sets\n- Providing methods to sample prompt-image pairs formatted for triplet training\n- Outputting prompt and image embeddings as tensors\n(if needed, helper functions for data filtering and prompt filtering are included)"
        ],
        [
            "model.py",
            "Defines Model class responsible for:\n- Initializing with pre-trained joint text-image models (CLIP, BLIP)\n- Providing encode_prompt(prompt: str) -> Tensor\n- Providing encode_image(image_path: str) -> Tensor\n- Computing similarity score using cosine or other metrics\n- Loading and saving fine-tuned model weights"
        ],
        [
            "trainer.py",
            "Defines Trainer class responsible for:\n- Accepting model instance and dataset of triplet samples\n- Implementing the training loop with triplet loss function\n- Handling optimization (AdamW), learning rate scheduling, and batch sampling\n- Logging training metrics and saving checkpoints\n- Supporting evaluation on validation data\n- Incorporating visualization of clustering and ranking results"
        ],
        [
            "evaluation.py",
            "Defines Evaluation class responsible for:\n- Loading the trained model\n- Computing pairwise accuracy for test prompt-image pairs\n- Visualizing rankings (ranking images by Social Reward scores)\n- Generating t-SNE plots of prompt embeddings to verify clustering\n- Implementing analysis such as per-cluster accuracy comparison"
        ],
        [
            "main.py",
            "Orchestrates experiment execution:\n- Initialize dataset_loader with dataset path and clustering info\n- Load datasets, prompts, and clusters\n- Initialize model with pre-trained weights\n- Set up trainer with dataset and model\n- Run training loop\n- Save model checkpoints\n- Run evaluation on test set\n- Visualize rankings and clustering\n- Handle command-line arguments for different phases (train/evaluate/fine-tune)"
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "No external API endpoints needed; all operations are handled internally in scripts. If exposing services, document APIs for dataset upload, training start, and evaluation results visualization, following OpenAPI 3.0 standards.",
    "Shared Knowledge": "Shared utility functions for embedding normalization, cosine similarity calculation, prompt clustering, and visualization plotting; configuration management (paths, hyperparameters) should be consistent across files.",
    "Anything UNCLEAR": "Clarification needed on dataset format specifics (e.g., data annotation structure, exact filter criteria), and whether the clustering granularity (number of clusters) is fixed or dynamically determined based on silhouette scores."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0003
  batch_size: 32
  epochs: 10  # Number of epochs not explicitly given; set a reasonable default for large-scale triplet training
  optimizer: AdamW
  device: cuda # Assuming GPU execution
  seed: 42

dataset:
  dataset_path: "path/to/curated/dataset"
  train_split_ratio: 0.7
  val_split_ratio: 0.1
  test_split_ratio: 0.2
  max_prompt_words: 5
  min_view_count: 10
  max_view_count: 1000
  filter_negative_signals: true

model:
  pretrained_model_name: "openai/clip-vit-base-patch32"
  save_dir: "models/checkpoints/"
  load_weights: null  # Set if resuming training

loss:
  margin: 0.2  # Typical triplet margin, not explicitly given but a standard choice
  loss_function: triplet  # Custom triplet loss implementation

clustering:
  method: ward  # Hierarchical clustering method
  num_clusters: 173  # Based on suggested clustering analysis
  linkage_distance_threshold: null  # Will be set based on silhouette score or kept as variable

evaluation:
  rank_visualization: true
  cluster_dimensionality: 2  # For t-SNE visualization
  evaluation_metrics:
    pairwise_accuracy: true
    ranking_visualization: true

fine_tuning:
  enabled: false  # Enable if planning fine-tuning generative model later
  learning_rate: 1e-4
  epochs: 5  # Optional, not specified in paper => default to 5
  reward_model_checkpoint: "models/your_trained_social_reward.pt"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
