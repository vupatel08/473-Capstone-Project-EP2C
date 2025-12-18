# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, detailed plan for reproducing the method presented in the paper "PACE: Marrying generalization in Parameter-efficient fine-tuning with Consistency regularization." This plan emphasizes key theoretical motivations, implementation specifics, experiment design, hyperparameters, and evaluation, to enable faithful reproduction and future code development.

---

### 1. Methodology Overview & Implementation Strategy

**a. Core Concepts & Goals:**  
- PACE couples **gradient regularization** (via implicit reduction of weight and Hessian norms) and **model output alignment** to retain pretraining knowledge, thereby improving generalization in PEFT frameworks.
- It perturbs adapter features multiplicatively, ensuring consistency across perturbations, which regularizes gradients implicitly.
- It explicitly constrains output difference between the fine-tuned model and the pre-trained "original" via a consistency loss.
- The approach is designed to be easy to integrate into existing PEFT methods (like LoRA, Adapter, VPT), by adding a regularization *plus* a consistency regularization component.

**b. Implementation Roadmap:**  
- Start with a base pre-trained transformer model (e.g., ViT-B/16 or ViT-B/16 trained with PEFT).
- Fine-tune with the **PACE method** by supplementing standard PEFT loss with the following components:
  - **Multiplicative feature perturbation** on learned adapter features during each forward pass, generating multiple noisy models.
  - **Consistency loss** between outputs of models with different perturbations.
  - **Regularization of gradients** should be implicitly achieved through the perturbation + consistency scheme, but can be monitored via gradient norm tracking.
- For the *fast* or *lazy* variants:
  - **PACE_fast:** Save previous epoch outputs and compare with current epoch outputs.
  - **PACE_lazy:** Apply perturbations intermittently, e.g., every N steps with less frequent updates (every N steps, only partially perturb).

---

### 2. Key Experimental Details & Dataset Requirements

**a. Datasets & Tasks:**
- Follow the paper's benchmarks:
  - **Visual Tasks:** VTAB-1K, FGVC (OxfordPets, Flowers102, StanfordCars, etc.), Domain adaptation (ImageNet-1K W/ domain shifts like CUB, NABirds, etc.).
  - **Natural Language Tasks:** GLUE, GSM-8K (mathematical reasoning).
- Access:  
  - For vision: Use pretrained ViT-B/16 models (e.g., from torchvision, HuggingFace, or official checkpoints), especially ones trained with or compatible with PEFT.
  - For NLP: Use models like LLaMA, T5, or BERT, trained with PEFT approaches.
- Fine-tuning datasets: Use the same splits, dataset sizes, or sampling strategies specified (e.g., VTAB-1K: 800 training samples per dataset).

**b. Data Handling & Preprocessing:**
- Follow the respective benchmarks’ preprocessing pipelines (e.g., augmentations for vision, tokenization for NLP).
- For reproducibility, fix data splits or use the same splits/augmentation schemes described.

---

### 3. Model & Adapter Architecture Setup

**a. Base Pretrained Models:**
- For vision: ViT-B/16 pretrained on ImageNet-21K or similar large-scale datasets.
- For NLP: LLaMA, BERT, or T5, pretrained on large corpora.

**b. Adapter & PEFT Modules:**
- Implement or use existing PEFT modules:
  - **LoRA:** low-rank adapters, modifying key linear layers.
  - **Adapter:** residual adapters inserted into Transformer blocks.
  - **VPT (Visual Prompt Tuning):** learnable prompts added to input tokens.
- Ensure the modules can be perturbed:  
  - For adapter-based modules, perturb features multiplicatively (element-wise).
  - Structure features to allow feature perturbation: e.g., after projection layers or within attention mechanisms.

**c. Adapter Feature Perturbation:**
- During forward pass:
  - For each sample, perturb the adapter features with multiplicative noise:
    - Noise $z \sim \mathcal{N}(1, \sigma^2 I)$, where $\sigma$ is a hyperparameter.
    - Element-wise: adapter_feature *= z.
  - Use independent noises $z_1$, $z_2$ for different perturbation samples or epochs or model instances.

---

### 4. Loss Functions & Regularization

**a. Main Task Loss:**
- Classification: standard cross-entropy.
- Regression: MSE or similar.

**b. Consistency Regularization Loss (Per Sample):**  
- Compute predictions for multiple noisy perturbed models.
- Enforce output invariance by L2 distance:
  \[
  D_{\text{pace}}(\mathbf{x}) = \|f(\mathbf{x}; \theta + z_1 \odot \Delta \theta) - f(\mathbf{x}; \theta + z_2 \odot \Delta \theta)\|_2^2
  \]
- During training:
  - Sample $z_1$, $z_2$ from Gaussian for each batch.
  - Accumulate the consistency loss weighted by $\lambda$.

**c. Implicit Gradient Regularization:**
- While not explicitly regularizing gradients, the multiplicative noise plus output consistency regularizes the model:
  - Regularization hyperparameter $\lambda$ controls strength.
  - For the *fast* variant, use previous epoch outputs for consistency instead of the sampled noise.

---

### 5. Hyperparameters & Optimization Details

**a. Hyperparameters to set & tune:**
- **AdamW** or similar optimizer:
  - Learning rate: follow the paper’s schemes, e.g., 2e-5 to 5e-5.
  - Weight decay: e.g. 1e-4 (see tables 16–19).
- **Dropout / Stochastic parameters:**
  - Dropout is optional; multiplicative noise is the regularizer.
- **Perturbation noise $\sigma$:**
  - Hyperparameter, e.g., 0.1, 0.2, or 0.5.
  - Range tested in the paper: 0.1–1.5.
- **Regularization weight $\lambda$:**  
  - Use a range of values, e.g. 0.001, 0.005, 0.01, 0.05, 0.1, as in Figures 8–9.
  - Tune based on validation performance or follow the paper’s recommendations.
- **Number of epochs/training steps:**
  - For visual datasets: e.g., 300 epochs as in the tables, with early stopping or fixed epochs.
  - For NLP tasks: align with existing fine-tuning schedules.

**b. Variants:**
- **Baseline:** standard PEFT + output regularization (if applicable).
- **PACE:** adds perturbation + consistency.
- **PACE_fast:** store previous epoch outputs, compute consistency between epochs.
- **PACE_lazy:** intermittently perturb, e.g., every N steps, with varying batch sizes.

---

### 6. Implementation Details & Code Elements

- Implement a perturbation module:
  - Element-wise Gaussian noise applied to adapter features.
- Integrate into training:
  - During each forward:
    - Perturb adapter features.
    - For each sample, compute multiple noisy outputs.
    - Compute consistency loss.
  - Sum with main task loss.
- For *lazy* or *fast* variants:
  - Store previous epoch outputs or selectively perturb.
- Monitoring:
  - Track gradient norms (via `torch.nn.utils.clip_grad_norm_` or manual hooks).
  - Track consistency loss value across epochs.
  - Log gradient norms, output differences, and accuracy.

---

### 7. Evaluation & Metrics

- **Main metric:**
  - Classification accuracy on test datasets.
  - For regression tasks, MSE or correlation.
- **Gradient regularization:**
  - Average gradient norm $\|\nabla_\theta \mathcal{L}\|_2$ during training.
- **Output consistency:**
  - Average $D_{\text{pace}}(\mathbf{x})$ on validation/test sets.
- **Generalization tests:**
  - Evaluate on unseen datasets/modalities.
  - Variability across random seeds.
- **Ablations:**
  - Hyperparameter sweeps on $\sigma$, $\lambda$.
  - Effect of perturbation frequency (lazy variants).
  - Memory/time overhead analysis.

---

### 8. Additional Considerations & Notes

- **Reproducibility:**
  - Fix seeds for randomness (PyTorch, NumPy).
  - Use fixed data splits.
  - Save adapter weights, previous outputs as needed for lazy variant.
- **Code interface:**
  - Modular design: base model + PEFT + perturbation + regularization.
- **Experimental plan:**
  - Start with standard PEFT + consistency.
  - Validate gradient norms reduction & output alignment.
  - Gradually incorporate lazy/fast schemes.
  - Conduct ablations on $\sigma$, $\lambda$, number of perturbations.
  - Compare with baselines: LoRA, Adapter, VPT, and existing SOTA.

---

### Summary of Key Parameters to Control & Details:

| Parameter | Range / Default | Purpose/Notes |
|-------------|------------------|--------------|
| $\sigma$ | 0.1–1.5 (tune) | Noise level for perturbation; controls regularization strength |
| $\lambda$ | 0.001–0.1 | Weight for consistency loss |
| epochs | 100–300 | Usually as per benchmarks, early stopping if needed |
| batch size | 16–64 | As per hardware; larger batch for stability |
| learning rate | 2e-5–5e-5 | As standard for fine-tuning large transformers |
| optimizer | AdamW | Standard choice |
| frequency of lazy updates | N=2–10 steps | For lazy variant; balance between resource saving and stability |
| previous epoch storage | yes/no | For PACE_fast |

---

**In essence**, this plan covers: selecting the pretrained model and PEFT module, implementing feature perturbation with Gaussian noise, integrating a consistency loss comparing perturbed outputs, maintaining hyperparameter tuning strategies, handling experimental setups and datasets, and monitoring regularization effects with analysis of gradient norms and generalization.

Once these steps are executed, follow-up with rigorous hyperparameter sweeps, ablations, and comparisons to baselines will enable faithful reproduction of results and insights.

---

Would you like me to now produce the pseudocode or template functions based on this plan?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the PACE method by extending an existing PEFT fine-tuning framework with modules for feature perturbation, consistency loss, and optional lazy/fast updates. Using PyTorch and Huggingface transformers, the system will load a pretrained transformer (vision or NLP), integrate a PEFT adapter (LoRA, Adapter, or VPT), and during training, apply multiplicative Gaussian noise to the adapter features. We will add a regularization term based on the L2 difference of model outputs with different perturbations, using either current or previous epoch outputs for the lazy variant. The optimizer will update only adapter and perturbation parameters, while tracking gradient norms and model outputs. Hyperparameters such as noise level, lambda for regularization, learning rate, and number of epochs will be configurable. After training, evaluate on validation/test sets and compare accuracy, gradient norms, and output divergence; all components will be encapsulated for modularity.",
    "File list": [
        "main.py",
        "dataset.py",
        "model.py",
        "trainer.py",
        "utils.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
    }
    class Dataset {
        +__init__(dataset_paths: dict)
        +load_train()
        +load_validation()
        +load_test()
        +preprocess(data: Dataset) -> Dataset
    }
    class TransformerModel {
        +__init__(pretrained_name: str, config: dict)
        +load_pretrained()
        +forward(inputs: Tensor) -> Tensor
        +get_adapter() -> Adapter
    }
    class Adapter {
        +__init__(type: str, num_params: int)
        +apply(input: Tensor) -> Tensor
        +perturb_features(noise_level: float) -> Tensor
            # Modifies internal feature representations with multiplicative Gaussian noise
    }
    class Trainer {
        +__init__(model: TransformerModel, dataset: Dataset, hyperparams: dict)
        +train(epochs: int)
        +apply_perturbation()
        +compute_consistency_loss(output1: Tensor, output2: Tensor) -> Tensor
        +track_gradient_norms() -> float
        +save_previous_outputs()
        +load_previous_outputs()
    }
    class Evaluation {
        +__init__(model: TransformerModel)
        +evaluate(dataset: Dataset) -> dict
        +compute_metrics(outputs: Tensor, labels: Tensor) -> dict
    }
    Main --> Dataset
    Main --> TransformerModel
    Main --> Trainer
    Main --> Evaluation
    Trainer --> TransformerModel
    Adapter --> TransformerModel
    Trainer --> Adapter
    Trainer --> Dataset
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DS as Dataset
    participant Mdl as TransformerModel
    participant Dr as Adapter
    participant Tr as Trainer
    participant Ev as Evaluation
    M->>DS: load datasets with configs
    DS-->>M: datasets loaded and preprocessed
    M->>Mdl: initialize with pretrained model
    M->>Tr: start training
    Tr->>Mdl: forward(inputs)
    Mdl-->>Tr: predictions
    Tr->>Tr: perturbFeatures(noise_level)
    Tr-->>Tr: output for each perturbation
    alt lazy update
        Tr->>Tr: save previous epoch's outputs
        Tr->>Tr: load previous epoch's outputs
        Tr-->>Tr: compute consistency loss w/ previous outputs
    else normal
        Tr-->>Tr: compute consistency loss between perturbations
    end
    Tr-->>M: training complete
    M->>Ev: evaluate(dataset)
    Ev->>Mdl: forward(inputs)
    Mdl-->>Ev: predictions
    Ev-->>M: metrics reporting
    Main->>Console/Logger: log metrics, gradient norms
    loop hyperparameter sweeps, validation
}
",
    "Anything UNCLEAR": "Clarify whether the multiplicative noise is strictly Gaussian and whether the number of perturbations per batch should be fixed or variable. Also, specify if the previous epoch outputs for the lazy variant should be stored in memory or disk, and what frequency of lazy updates is preferred. Confirm the set of datasets and whether to reproduce all benchmarks or focus on select tasks."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "numpy==1.21.2",
        "tqdm==4.64.0",
        "scikit-learn==0.24.2",
        "transformers==4.26.1",
        "datasets==1.18.0",
        "upload==0.0.25"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset.py",
            "Defines Dataset class responsible for loading datasets, running preprocessing pipelines, and providing train/validation/test splits. It depends on datasets library for loading standard benchmark datasets, and may include custom loaders for VTAB split, FGVC, GLUE, GSM-8K, etc."
        ],
        [
            "model.py",
            "Defines TransformerModel class that loads pre-trained transformer backbone (e.g., ViT-B/16, BERT, LLaMA), initializes PEFT modules such as LoRA, Adapter, or VPT, and incorporates perturbation mechanisms for features. Provides methods for forward pass, perturb features, and extract adapter components."
        ],
        [
            "trainer.py",
            "Implements the Trainer class managing the training loop, including applying multiplicative Gaussian feature perturbations, computing main task loss (classification/regression), computing consistency loss between multiple perturbations or previous epoch outputs (lazy variant), handling optimizer step, gradient clipping, and gradient norm tracking. It also manages saving/loading previous outputs for lazy update options."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class or functions that evaluate the model on validation/test datasets, generating metrics (accuracy, MSE, correlation), also tracking gradient norms if needed."
        ],
        [
            "utils.py",
            "Utility functions for Gaussian noise generation, regularization scheduling, hyperparameter management, gradient norm calculation, logging, and checkpointing."
        ],
        [
            "main.py",
            "Main execution script that parses configuration, initializes datasets, models, trainer, and evaluator; runs training epochs with specified parameters and performs final evaluation. It coordinates hyperparameter sweeps, logs results, and handles experiments for varying dataset splits, noise levels, lambda values, and lazy/fast variants."
        ]
    ],
    "Task list": [
        "dataset.py",
        "model.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "The code shares core hyperparameters such as noise level, regularization coefficient lambda, learning rate, number of epochs, and dataset configurations. Utility functions are shared across modules for noise application, gradient norm calculation, and logging. Model modules share the adapter extraction and perturbation interfaces.",
    "Anything UNCLEAR": "Clarify the desired level of precision for gradient norm calculations and whether to include augmentations in the dataset loader. Also, confirm if the experiments should be on vision, NLP, or multi-modal models, and whether to support multiple model architectures interchangeably."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 2e-5
  batch_size: 16
  epochs: 300
  optimizer: AdamW
  weight_decay: 1e-4
  lambda_consistency: 0.01
  sigma_noise: 0.2
  regularization_type: "standard" # options: 'standard', 'lazy', 'fast'
  lazy_update_interval: 10  # applicable if using lazy variant
  use_previous_epoch_outputs: true # applicable for lazy variant
dataset:
  dataset_name: "VTAB-1K"
  train_split: "train"
  validation_split: "validation"
  test_split: "test"
  # dataset configs might include dataset-specific details
model:
  pretrained_model_name: "google/vit-base-patch16-224-in21k"  # for vision; change accordingly for NLP
  peft_method: "LoRA"  # options: 'LoRA', 'Adapter', 'VPT'
  peft_rank: 16
  adapter_params: 0.5  # scale for adapter size, if applicable
  perturbation_sigma: 0.2
  adapter_perturbation: true
  output_regularization: true
evaluation:
  metrics: "accuracy" # or "MSE", "correlation"
  metric_target: "validation"
  save_best_model: true
  evaluation_interval: 10 # in epochs
misc:
  seed: 42
  device: "cuda:0"  # or 'cpu' or other device string
  verbose_logging: true
```

---

**Note:** Full configuration is available in `planning_config.yaml`
