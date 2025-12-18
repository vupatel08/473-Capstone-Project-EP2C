# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, structured plan to guide the implementation and experimental reproduction of the Variance-Controlled Adaptive Sampling (VCAS) method described in the paper. This plan focuses on decoding the methodology, understanding experimental setups, and identifying critical hyperparameters to ensure faithful reproduction.

---

## I. Understanding and Reproducing the Methodology

### 1. Core Objective & Principle
- **Goal:** Accelerate backpropagation in neural network training by sampling data points and tokens in a fine-grained, importance-aware manner, while controlling the additional variance introduced by approximation.
- **Key Insight:** The importance of data points and tokens in gradients varies across layers and training steps. VCAS adaptively samples based on importance, maintaining unbiasedness while controlling variance.

### 2. Unbiased Gradient Estimation via Importance Sampling
- **Activation Gradient Sampling**
  - **Unbiasedness:** Proved via induction over layers. For linear layers, the importance sampling is straightforward; for non-linear layers (e.g., ReLU), the unbiasedness is maintained because the importance sampling applies to the gradients multiplied by the saved Jacobian.
  - **Implementation:**
    - Sample a subset of data points for the activation gradient by probability $p_i$ proportional to the gradient norm $\|G_i\|_F$ (or an approximation).
    - Use a stochastic mask $m(\epsilon, \rho)$ to zero out less important samples, ensuring unbiasedness.
    - Scale retained contributions by $1/p_i$ to keep unbiased estimates.
- **Weight Gradient Sampling**
  - **Unbiasedness:** Based on leverage-score sampling applied at the token and data point level.
  - Implement similar importance sampling with probabilities proportional to the importance measure derived from the approximate gradient and activation activity.
  - For linear layers, a similar unbiased importance sampling applies, scaled appropriately.

### 3. Variance Control
- **Variance bounds** are enforced by adaptively tuning sampling probabilities:
  - For activation gradients: keep variance of the estimator below a threshold $\tau_{act}$ proportional to the total variance.
  - For weight gradients: similarly bound variance with threshold $\tau_w$.
- **Adaptive Algorithm:** 
  - Update sample ratios $\{\rho_l\}$ for activations and $\{\nu_l\}$ for weights per layer based on variance estimates.
  - Variance estimates are obtained via Monte Carlo sampling with $M$ samples per iteration.
  - Hyperparameters:
    - $\tau_{act}, \tau_{w}$: thresholds controlling variance.
    - $\alpha$, $\beta$: step size and ratio scaling for the hyperparameters.
    - Frequency $F$ for update steps of $\{\rho_l\}$, $\{\nu_l\}$, and hyperparameters.

### 4. Fine-grained Importance Sampling Strategy
- **Activation Gradients:**
  - For each layer, compute importance scores for data points:
    - Proportional to current gradient norm $\|G_i\|_F$.
  - Sample data points with probabilities $p_i \propto \|G_i\|_F$.
  - Scale contributions by $1/p_i$.
- **Token Importance:**
  - For tokens (sequence elements), importance can be estimated via leverage scores (eigenvalue-based importance in linear algebra).
  - Importance sampling probabilities $\propto$ importance measure.
  - Applies in layers with sequence data (e.g., Transformer layers).
- **Layer-specific importance ratios:** 
  - $\rho_l$ for activation sampling.
  - $\nu_l$ for token sampling in weight gradient computation.
- **Sampling Schedule:**
  - Initialize ratios to 1.
  - Update every $F$ steps using variance estimates as feedback.
  - Use thresholds $\tau_{act}$ and $\tau_w$ to control variance inflation.

### 5. Implementation Details
- **Unbiased estimators**: Use scaled sampling weights to ensure the expected value aligns with full gradient.
- **Batching:**
  - Implement data parallelism; sample importance ratios per batch.
- **Layer handling:**
  - For linear layers, importance sampling over weights is straightforward.
  - For non-linear layers, importance sampling applies to the activation gradients, applying Jacobian-based importance (or approximate).
- **Approximation mechanics:**
  - Maintain the original gradient trajectory; only approximate where importance sampling is applied.
- **Variance estimation:**
  - Use Monte Carlo estimates with small $M$ per iteration (e.g., 2–10) for variance bounds.
  - Step size $\alpha$, update frequency $F$, and thresholds $\tau_{act}$, $\tau_w$ are hyperparameters.

---

## II. Experimental Setup & Implementation Considerations

### 1. Dataset and Tasks
- **Natural Language:**
  - Fine-tuning BERT-base on:
    - GLUE tasks (e.g., SST-2, MNLI, QNLI, CoLA, SST, MRPC, RTE).
  - Pretraining BERT (masked language modeling) with crammed data.
- **Vision:**
  - Fine-tuning Vision Transformers (ViT) on ImageNet-1K.

### 2. Model Architectures
- **Transformer-based models:** BERT-base, BERT-large, ViT-base, ViT-large.
- **Layer structure:** Use layer definitions and sizes from official pretrained models for reproducibility.

### 3. Hyperparameters
- **Sampling importance:**
  - Initialize importance ratios ($\rho_l$, $\nu_l$) to 1.
  - Hyperparameters for variance thresholds: e.g., $\tau_{act} = 0.025$, $\tau_w = 0.025$, but tune based on dataset/model.
  - Update hyperparameters with small step sizes $\alpha$, using the heuristic rules in Sec. 5.
- **Batch size, learning rate:**
  - For BERT: batch size around 16–64, learning rate in [2e-5, 5e-5].
  - For vision: e.g., batch of 256, learning rate as per original. Use cosine decay or linear warm-up.
- **Update frequency $F$:** e.g., every 100–1000 steps.
- **Monte Carlo samples $M$:** e.g., 2–10 for variance estimation.

### 4. Implementation Details
- **Frameworks:** PyTorch or TensorFlow (preferably PyTorch for flexibility).
- **Data loading:** Parallel data loaders; implement importance sampling in the backward pass.
- **Layer modifications:** Integrate sampling masks into backward routines at each layer.
- **Variance estimation:** Use small $M$ per iteration to tune ratios dynamically.
- **Hyperparameter tuning**: Conduct grid search over $\alpha$, $\beta$, $\tau$ for best variance control.

### 5. Evaluation Metrics
- **Training cost:** FLOPs reduction, wall-clock time.
- **Convergence curve:** Loss trajectory, validation accuracy.
- **Variance monitoring:** Track estimator variance to ensure control.
- **Final performance:** Downstream task metrics (accuracy, F1, etc.).

### 6. Baseline Comparisons
- Exact backpropagation.
- Previous importance-sampling methods (SB, UB).
- Standard training without sampling.

---

## III. Summary and Next Steps

- **Implement Unbiased Importance Sampling:**
  - For activation gradients: importance based on norms, with unbiased scaling.
  - For weight gradients: leverage-score importance via SVD/ eigenvalue estimates.
- **Variance Control Loop:**
  - During training, estimate variance via Monte Carlo.
  - Update sample ratios $\{\rho_l, \nu_l\}$ every $F$ steps to keep variance below thresholds.
- **Hyperparameter Sensitivity:**
  - Conduct grid search on $\alpha$, $\beta$, $\tau_{act}$, $\tau_w$, $\rho$.
- **Experimental validation:**
  - Reproduce all results (e.g., FLOPs reduction, accuracy, loss trajectories) on chosen tasks.
  - Use small-scale experiments first to validate correctness.

---

This plan provides a comprehensive roadmap for faithful implementation of VCAS, covering the core unbiased sampling techniques, variance control measures, and experimental design parameters. You can now proceed with code development following this structured blueprint.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the VCAS method within a PyTorch training loop, integrating importance sampling for activation gradients and weight updates, and an adaptive variance control mechanism. The system will modularize data loading, model definition, importance sampling logic, variance estimator, training routines, and evaluation, with hyperparameters for variance thresholds, update frequency, and sample ratios. We will use existing Torch modules for models (e.g., BERT, ViT), modify backprop routines to include importance masks and scaling, and dynamically tune sample ratios based on variance estimates during training.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "importance_sampler.py",
        "variance_estimator.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "```mermaid\nclassDiagram\nclass Main {\n    +__init__(config: dict)\n    +run()\n}\nclass DatasetLoader {\n    +__init__(dataset_config: dict)\n    +load_data() -> Dataset\n}\nclass Model {\n    +__init__(model_config: dict)\n    +initialize()\n    +forward(x: Tensor) -> Tensor\n    +apply_mask_and_scale(mask: Tensor, scale: Tensor) -> None\n}\nclass ImportanceSampler {\n    +__init__(initial_ratio: float)\n    +update_ratios(variance_estimates: dict) -> None\n    +sample_indices(probabilities: Tensor, ratio: float) -> Tensor\n}\nclass VarianceEstimator {\n    +__init__(num_samples: int)\n    +estimate_variance(data: Tensor, importance_scores: Tensor) -> float\n}\nclass Trainer {\n    +__init__(model: Model, dataset: Dataset, importance_sampler: ImportanceSampler, variance_estimator: VarianceEstimator, hyperparameters: dict)\n    +train_loop() -> None\n    +adjust_sampling_ratios() -> None\n}\nclass Evaluation {\n    +__init__(model: Model, dataset: Dataset)\n    +evaluate() -> dict\n}\nMain --> DatasetLoader\nMain --> Trainer\nTrainer --> Model\nTrainer --> ImportanceSampler\nTrainer --> VarianceEstimator\nMain --> Evaluation\nModel --> importance_sampler: use masking and importance scaling\n```\n",
    "Program call flow": "```sequenceDiagram\nparticipant M as Main\nparticipant DL as DatasetLoader\nparticipant MD as Model\nparticipant TR as Trainer\nparticipant EV as Evaluation\n\nM->>DL: __init__(dataset_config)\nDL-->>M: load_data()\nM->>MD: __init__(model_config)\nMD-->>M: model ready\nM->>TR: __init__(model, dataset, importance_sampler, variance_estimator, hyperparameters)\nTR->>TR: initialize sampling ratios and variance control\nTR->>TR: train_loop()\nloop each epoch\n   TR->>TR: adjust_sampling_ratios() based on variance estimates\n   TR->>MD: forward() with importance masks\n   MD-->>TR: gradients\n   TR->>importance_sampler: get sampling indices based on importance scores and ratios\n   importance_sampler-->>TR: indices\n   TR->>MD: apply mask, scale gradients\n   MD-->>TR: scaled gradients\n   TR->>variance_estimator: estimate variance\n   note over TR: update ratios if needed\nend\nTR->>EV: evaluate()\nEV->>MD: forward()\nMD-->>EV: predictions\nEV-->>M: metrics\n```",
    "Anything UNCLEAR": "Need clarification on how to integrate importance sampling into existing backward functions for each layer type (linear, non-linear). Also, details on initial hyperparameter tuning ranges for variance thresholds, update frequency, and importance ratios would help finalize the implementation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.12.1",
        "numpy==1.21.0",
        "tqdm==4.62.3",
        "transformers==4.26.1",
        "datasets==2.5.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Creates DatasetLoader class that loads datasets (e.g., ImageNet, GLUE tasks) and handles batching, tokenization (for NLP), and data pre-processing. Implements dataset splitting, shuffling, and batch formation compatible with transformers and vision models. Relies on datasets and transformers libraries."
        ],
        [
            "model.py",
            "Defines neural network architectures, including BERT and ViT models. Wraps HuggingFace transformers or torchvision models. Implements model instantiation, forward pass, and integrating importance sampling masks/scales into specific layers (via hooks or custom modules)."
        ],
        [
            "importance_sampler.py",
            "Implements ImportanceSampler class, which maintains state of importance ratios, dynamically updates them based on variance estimates, and provides sampling functions for data points and tokens. Contains methods `update_ratios()` and `sample_indices()` with input of importance scores and current ratios."
        ],
        [
            "variance_estimator.py",
            "Provides VarianceEstimator class to estimate variance of gradients via Monte Carlo sampling. Offers `estimate_variance()` method which performs M importance samples, computes variances for activation and weight gradients, and returns estimates used for hyperparameter tuning."
        ],
        [
            "trainer.py",
            "Implements Trainer class orchestrating the training loop. It initializes model, datasets, importance sampler, variance estimator, and hyperparameters. Manages training epochs, calls importance sampler to determine sample masks, applies scaling, updates sampling ratios periodically, calculates losses, backward passes, and logs variance estimates. Implements `train()` and `adjust_ratios()` methods following the adaptive variance control algorithm."
        ],
        [
            "evaluation.py",
            "Contains Evaluation class that loads the trained model and dataset, runs inference, and computes evaluation metrics relevant to the task (accuracy, F1, loss). Used post-training to assess final performance."
        ],
        [
            "main.py",
            "Main script as entry point that loads config, initializes DatasetLoader, model, importance_sampler, variance_estimator, and trainer, then runs training and evaluation sequentially. Manages hyperparameter settings, experiment repetitions, and logging."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "importance_sampler.py",
        "variance_estimator.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n\n# No front-end API; interaction mostly via direct script invocations and configuration files.",
    "Shared Knowledge": "Common utilities are shared across modules for data loading, logging, and configuration parsing. Importance sampling functions, variance calculation, and hyperparameter update rules are consistently used and documented for reproducibility.",
    "Anything UNCLEAR": "Clarification needed on the expected hardware configuration for large-scale experiments, and whether multi-GPU training should be incorporated into the training pipeline."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 2e-5
  batch_size: 32
  epochs: 3
  warmup_steps: 500
  total_steps: 10000
  update_ratio_frequency: 100
  variance_thresholds:
    activation: 0.025
    weight: 0.025
importance_sampling:
  initial_ratio: 1.0
  importance_method: gradient_norm
sampling:
  activation_ratio: 1.0
  weight_ratio: 1.0
hyperparameters:
  variance_control_thresholds:
    activation: 0.025
    weight: 0.025
  update_step_alpha: 0.01
  ratio_scaling_beta: 0.95
  variance_update_frequency: 100
  monte_carlo_samples: 4
model:
  type: bert-base-uncased
  pretrained: true
  max_seq_length: 128
dataset:
  name: glue
  task: SST-2
  split: train
  tokenizer_name: bert-base-uncased
  max_tokens: 128
evaluation:
  step_interval: 100
  metrics:
    - accuracy
    - loss
```

---

**Note:** Full configuration is available in `planning_config.yaml`
