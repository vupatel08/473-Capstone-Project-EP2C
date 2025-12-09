# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a **comprehensive, step-by-step plan** that synthesizes the methodology, experimental setup, and evaluation details from the paper "CurBench: Curriculum Learning Benchmark." This outline is designed to guide an implementation from scratch, clarifying key decisions, dataset preparations, hyperparameter settings, and evaluation protocols.

---

## I. Overall Objective and Approach

**Goal:**  
Develop an implementation of a comprehensive curriculum learning benchmark platform, *CurBench*, that facilitates the systematic evaluation of multiple curriculum learning (CL) methods across various datasets, models, and experimental settings.

**Core Concept:**  
- Encapsulate *existing curriculum strategies* as modular "curriculum modules."
- Provide a *unified pipeline* for data preprocessing, model training, hyperparameter configuration, and evaluation.
- Enable *fair comparison* by standardizing experimental protocols across methods, datasets, and models.

---

## II. Methodology Components

### 1. Modular Design of Curriculum Methods
- **Curriculum modules:** Each representing a specific curriculum scheduling strategy (e.g., curriculum based on difficulty, self-paced learning, progressive augmentation).
- **Implementation basis:** Abstract each method with a class or function that:
  - Accepts raw training data, model, hyperparameters.
  - Outputs a *curriculum schedule* (e.g., ordering of samples, weighting schema, difficulty scaling).
- **Common interfaces:**
  - `initialize(dataset, model, hyperparameters)`
  - `update(epoch, data, model)` — optional, for adaptive methods.
  - `get_sample_weights()` or `get_sample_order()`

### 2. Data Augmentation & Difficulty Measures
- **Difficulty metrics:** Implement various difficulty criteria:
  - Data complexity (e.g., image noise level, class imbalance, or handcrafted difficulty scores).
  - Model-based difficulty (e.g., confidence-based, loss-based, or sample hardness).
- **Data processing pipeline:**  
  - For datasets like CIFAR, MNIST, IMAGENET, etc., prepare augmentations or difficulty annotations (e.g., sample difficulty scores).
  - For noisy or imbalanced datasets, modify sample labels or data distributions accordingly.

### 3. Data Handling & Dataset Preparation
- Support multiple datasets with different characteristics (classification, sequences, graphs):
  - CIFAR-10/100, Tiny-ImageNet, IMAGENET for CV.
  - MNIST, Glove-derived features for NLP.
  - Graph datasets like ogb-molhiv, MUTAG.
- **Dataset splits:**  
  - Follow the protocol: train/validation/test splits, with explicit validation and test sets.
  - For noisy/imbalanced data:
    - Generate noisy labels with specified noise ratio (`p`).
    - Create imbalance factors (`r`) for class imbalance.
  - For graph datasets, adhere to existing splits.

### 4. Model Selection & Backbone Architectures
- Implement or leverage existing implementations:
  - CV: LeNet, ResNet-18, ViT.
  - NLP: LSTM, BERT, GPT2.
  - Graph: GCN, GAT, GIN.
- **Model loading:**
  - Use pre-trained models where specified (e.g., BERT, ResNet pretrained on ImageNet).
  - For scratch training, initialize from random.
  - Modular architecture loader (based on a model string or class).

### 5. Training Procedure & Hyperparameters
- **General training loop:**
  - For each epoch:
    - Obtain sample weights or sample order from the curriculum module.
    - Pass data accordingly to the model (e.g., weighted sampling, sample difficulty filtering).
    - Standard optimizer steps.
  - At scheduled intervals (e.g., every epoch or every fixed number of epochs), update curriculum if adaptive.
- **Hyperparameters:**
  - Implement key hyperparameters as configurable variables, including:
    - Learning rate.
    - Batch size.
    - Number of epochs or growth epochs.
    - Data augmentation settings if applicable.
    - Curriculum-specific parameters (e.g., growth_fn type, start_ratio, warm-up epochs).
- **Schedule of hyperparameters:**
  - Keep them consistent across methods for fair comparison, per the paper.

### 6. Evaluation & Metrics
- Objective metrics:
  - Accuracy, F1 score, Spearman correlation on validation/test.
- Performance & complexity:
  - Record training time per epoch.
  - Record maximum GPU memory consumption.
- Repeat each experiment *multiple times* (e.g., 5-7 runs) with different random seeds to compute mean and std dev.
- **Evaluation procedure per method/dataset/model:**
  - Train from scratch; record logs.
  - Run on validation and test sets; compute objective metrics.
  - Save training logs, hyperparameters, and resource usages.

---

## III. Experimental Setup & Protocols

### 1. Datasets & Preprocessing
- **Datasets:** Download and prepare datasets listed in the benchmark:
  - CV: CIFAR-10, CIFAR-100, Tiny-ImageNet.
  - NLP: MNIST, GLUE tasks.
  - Graph: ogb-molhiv, MUTAG.
- **Preprocessing:**
  - Normalize images.
  - Convert text datasets to token IDs.
  - Build graph data objects (nodes, edges, labels).
- **Difficulty annotations:**
  - For datasets like noisy CIFAR, generate label noise with noise ratio `p`.
  - For class imbalance, assign imbalance ratio `r` over dataset.
  - For graph datasets, that may involve subgraph or node difficulty scores if applicable.

### 2. Experimental Settings
- **Settings types:**
  - Standard: balanced, clean data.
  - Noise: add synthetic label noise (`p`).
  - Imbalance: create class imbalance (`r`).
- **Training configurations:**
  - Batch sizes, epochs, learning rate schedules per dataset/model.
  - Use fixed, standard hyperparameters or adopt ones from authors' reported best performances.
- **Number of runs:**  
  - At least 5 independent runs per scenario with different seeds.

### 3. Hyperparameters & Schedule Configurations
- Establish hyperparameter configurations as per the paper:
  - For curriculum strategies: start ratio, growth epochs, growth_fn type (linear, exponential), warm epochs.
  - For model training: learning rate, weight decay, optimizer type.
- For adaptive strategies: set criteria or decay rates as described.
- Maintain uniform training hyperparameters across methods, vary only curriculum parameters if testing curriculum effects.

### 4. Implementation & Resource Recording
- Use GPU (V100 or equivalent).
- Record:
  - Training time per epoch.
  - Peak GPU memory per run.
- Use standardized logging framework (e.g., TensorBoard, CSV logs).

---

## IV. Evaluation & Analysis

- After training, evaluate models on:
  - Objective metrics (accuracy, F1, correlation).
  - Resource metrics (training time, memory).
- Aggregate results:
  - Compute mean±std over multiple runs.
- Visualize:
  - Performance vs. noise ratio `p`.
  - Performance vs. imbalance ratio `r`.
  - Resource consumption comparisons.
- Use the detailed results in Tables 8–11 for benchmarking and validation.

---

## V. Summary & Implementation Roadmap

**1. Framework Core:**
- Dataset loaders + data augmenters + difficulty annotators.
- Model loader + architecture definitions.
- Curriculum method classes/functions.
- Training loop that takes curriculum schedule outputs.
- Resource monitor (training time, GPU memory).
- Evaluation plug-ins for various metrics.

**2. Reproducibility & Fairness:**
- Fixed hyperparameter scripts.
- Multiple seed runs.
- Log hyperparameters, seed, resource usage, and final metrics.

**3. Extensibility:**
- Modular design allows adding new curriculum strategies, datasets, or models.

---

This comprehensive plan provides the blueprint to implement *CurBench*, including all relevant details derived from the paper’s methodology, datasets, experiments, and evaluation protocols, ensuring alignment with the authors’ experimental rigor and evaluation fairness.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will design a modular pipeline using PyTorch for models, datasets, training, and evaluation. The core architecture includes separate modules for dataset loading, curriculum strategies, model definition, training, and evaluation. Curriculum modules will be classes implementing various strategies (e.g., difficulty-based, confidence-based). Hyperparameters and configurations will be managed via YAML or JSON files. The system will handle standard and noisy/imbalanced datasets by transforming data during loading and provide interfaces for resource logging (time and memory). All components interact through clearly defined APIs, enabling flexible addition of new curriculum methods.",
    "File list": [
        "main.py",  
        "dataset_loader.py", 
        "models.py",  
        "curriculum_strategies.py",
        "trainer.py",
        "evaluation.py",
        "resources.py",
        "utils.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "classDiagram\nclass Main {\n  +__init__(config: dict)\n  +run_experiment() -> None\n}\nclass DatasetLoader {\n  +__init__(config: dict)\n  +load_data() -> Dataset\n  +apply_noise_or_imbalance() -> Dataset\n}\nclass Model {\n  +__init__(model_type: str, hyperparams: dict)\n  +forward(x: Tensor) -> Tensor\n  +configure_optimizer() -> torch.optim.Optimizer\n}\nclass CurriculumStrategy {\n  +__init__(params: dict)\n  +get_sample_weights(dataset: Dataset, model: Model, epoch: int) -> Tensor\n  +update_strategy(epoch: int, model: Model, dataset: Dataset) -> None\n}\nclass DataAugmentation {\n  +apply_transform(x: Dataset) -> Dataset\n}\nclass Trainer {\n  +__init__(model: Model, dataset: Dataset, curriculum: CurriculumStrategy, hyperparams: dict)\n  +train() -> None\n  +log_resources(epoch: int) -> None\n}\nclass Evaluation {\n  +__init__(model: Model, dataset: Dataset)\n  +evaluate() -> dict\n}\n\nMain --> DatasetLoader\nMain --> Model\nMain --> CurriculumStrategy\nMain --> Trainer\nTrainer --> Model\nTrainer --> Dataset\nTrainer --> Resources\nMain --> Evaluation\nEvaluation --> Model\ncycle Resources\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant CS as CurriculumStrategy\n    participant TR as Trainer\n    participant EV as Evaluation\n    participant R as Resources\n    M->>+DL: load_data()\n    DL-->>M: dataset\n    M->>+MD: initialize(model_type, hyperparams)\n    MD-->>M: model\n    M->>+CS: initialize(params)\n    CS-->>M: strategy\n    M->>TR: train(model, dataset, strategy, hyperparams)\n    loop for each epoch\n        TR->>CS: get_sample_weights(dataset, model, epoch)\n        CS-->>TR: sample_weights\n        TR->>model: forward(batch with sample weights)\n        model-->>TR: predictions\n        TR->>model.optimizer: step()\n        TR->>CS: update_strategy(epoch, model, dataset) # if adaptive\n        TR->>R: log_time_memory(epoch)\n    end\n    M->>EV: evaluate(model, dataset)\n    EV->>model: forward(test_batch)\n    model-->>EV: predictions\n    EV-->>M: metrics\n    Resources-->>Resources: log per epoch\n",
    "Anything UNCLEAR": "Clarification needed on the exact format and structure of datasets, especially their difficulty annotations, and how to integrate resource logging (specific tools or APIs preferred). Also, confirm if any particular open-source model repository preferences are mandated."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.11.0",
        "torchvision==0.12.0",
        "numpy==1.21.0",
        "pyyaml==6.0",
        "scikit-learn==0.24.2",
        "matplotlib==3.4.3",
        "pillow==8.4.0",
        "tqdm==4.62.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "DatasetLoader class: loads datasets (CIFAR, MNIST, ImageNet, etc.), applies noise/imbalance transformations, supports data augmentation, provides API load_data() -> Dataset, apply_noise_or_imbalance() -> Dataset."
        ],
        [
            "models.py",
            "Model classes: e.g., LeNet, ResNet, ViT, LSTM, BERT, GPT2, GCN, GAT, GIN. Each class with __init__(params), forward(x). Using torchvision.models or transformers as base, with configuration based on hyperparameters."
        ],
        [
            "curriculum_strategies.py",
            "CurriculumStrategy base class and subclasses (DifficultyBased, ConfidenceBased, SelfPaced, etc.): implement get_sample_weights(dataset, model, epoch), optional update_strategy(), hold parameters like growth_fn, start_ratio, warm_epochs."
        ],
        [
            "trainer.py",
            "Trainer class: initializes with model, dataset, curriculum object, hyperparameters; contains train() method with epoch loop: fetch sample weights, perform weighted sampling or filtering, run model forward/backward, update optimizer, log resources, call curriculum update if adaptive."
        ],
        [
            "evaluation.py",
            "Evaluation class: input trained model and dataset; evaluate() computes metrics like accuracy, F1, SRCC, and resource usage; outputs dict of metrics."
        ],
        [
            "resources.py",
            "Resources logging: functions/classes to record training time per epoch, GPU memory usage via torch.cuda API, save logs for resource analysis."
        ],
        [
            "utils.py",
            "Utility functions: configuration parsing, data plotting, logging setup, seed setting, reproducibility controls."
        ],
        [
            "main.py",
            "Entry point: argument parsing (via argparse or config file), initialize dataset loader, model, curriculum method, trainer, and evaluator; orchestrate the entire experiment cycle; save results and logs."
        ]
    ],
    "Task list": [
        "utils.py",
        "config.yaml",
        "dataset_loader.py",
        "models.py",
        "curriculum_strategies.py",
        "trainer.py",
        "evaluation.py",
        "resources.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\ncomponents:\n  schemas:\n    Dataset:\n      type: object\n      properties:\n        name:\n          type: string\n        split:\n          type: object\n          properties:\n            train:\n              type: array\n            val:\n              type: array\n            test:\n              type: array\n        difficulty_annotations:\n          type: object\n    Model:\n      type: object\n      properties:\n        type:\n          type: string\n        params:\n          type: object\n    CurriculumStrategy:\n      type: object\n      properties:\n        name:\n          type: string\n        parameters:\n          type: object\n        get_sample_weights:\n          type: operation\n    Trainer:\n      type: object\n      properties:\n        model:\n          $ref: '#/components/schemas/Model'\n        dataset:\n          $ref: '#/components/schemas/Dataset'\n        curriculum:\n          $ref: '#/components/schemas/CurriculumStrategy'\n        hyperparameters:\n          type: object\n    Evaluation:\n      type: object\n      properties:\n        metrics:\n          type: object\n        resource_usage:\n          type: object\n    Resources:\n      type: object\n      properties:\n        training_time:\n          type: number\n        gpu_memory_MB:\n          type: number\n"
        ,
    "Shared Knowledge": "Common utility functions for reproducibility: seed setting, resource logging, data augmentation. Curriculum modules share an interface get_sample_weights() and optional update_strategy(). Dataset loader applies transformations based on noise rate (p) and imbalance factor (r). Model modules are compatible with torchvision and transformers. Experiment hyperparameters are managed via config.yaml, accessible throughout modules."
        ,
    "Anything UNCLEAR": "Clarification needed on the exact file formats for dataset annotations (difficulty scores, noise labels, imbalance distribution). Unsure whether to implement resource logging via specific tools (e.g., tensorboard, custom logs). Need confirmation if the system should support progressive curriculum adaptation (e.g., growth_fn) and the expected logging or visualization outputs."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  name: CIFAR10
  split_ratios:
    train: 0.8
    validation: 0.1
    test: 0.1
  noise_ratio: 0.4  # For noise setting, as used in experiments
  imbalance_factor: 50  # For imbalance setting, as used in experiments
model:
  type: ResNet-18
  hyperparameters:
    learning_rate: 0.01  # Not specified, using common default
    batch_size: 128     # Not specified, pick standard for ResNet
    epochs: 100         # Not specified, common training length
train:
  optimizer: Adam       # Not specified, assume Adam for experiments
  weight_decay: 1e-4  # Standard value for regularization
  scheduler:
    type: StepLR
    step_size: 30
    gamma: 0.1
curriculum:
  strategy: DifficultyBased  # Example placeholder; actual strategies detailed in code
  start_ratio: 0.0
  grow_epochs: 20
  grow_fn: linear
  warm_epochs: 5
hyperparameters:
  seed: 42
  log_interval: 10
  device: cuda  # Use GPU if available
resource_logging:
  enable_time_tracking: true
  enable_memory_tracking: true
```

---

**Note:** Full configuration is available in `planning_config.yaml`
