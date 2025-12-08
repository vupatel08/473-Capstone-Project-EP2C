# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, structured plan to reproduce the methodologies and experiments outlined in the paper "Distributive Preference Learning: Understanding and Accounting for Hidden Context in RLHF". The plan is divided into key sections: (A) Core Methodology, (B) Experimental setup and datasets, (C) Hyperparameters and training procedures, (D) Evaluation metrics, and (E) Implementation notes and potential ambiguities.

---

### **A. Core Methodology: Preference Modeling with Hidden Context**

1. **Objective & Theoretical Foundations**
   - The main goal is to model preferences when hidden context (e.g., annotator identity, objectives, irrationality) influences human feedback, but is unobserved.
   - The paper introduces *distributional preference learning (DPL)* as an extension of standard preference learning, explicitly estimating a distribution over utility values for each alternative.

2. **Theoretical Results to Reproduce**
   - *Theorem 3.1*: Preference learning with BTL loss implicitly aggregates hidden context via the Borda count.
   - *Theorem 3.2*: Under i.i.d. symmetric noise, preference learning converges to the expected utility.
   - *Theorem 3.4*: Limitations of identifiability — because of noisy, hidden contexts, learned utilities may not match true expected utilities.

3. **Implementation of Preference Models**
   - **Standard Preference Learning**: Fit a utility function \( \hat{u}(a) \) using a Maximum Likelihood Estimation (MLE) based on pairwise comparison data.
   - **Distributional Preference Learning (DPL)**:
     - Implement two variants:
       - **Mean & Variance DPL**: Model \( u(a,z) \sim \mathcal{N}(\hat{\mu}(a), \hat{\sigma}^2(a)) \).
       - **Categorical DPL**: Model \( u(a,z) \) as discrete points (e.g., 10 evenly spaced values) with softmax probabilities.
     - **Neural Network Architecture**:
       - Input: prompt-response pair \( a \).
       - Outputs:
         - For mean & variance: two scalars (\( \hat{\mu}(a) \), \( \hat{\sigma}(a) \)).
         - For categorical: length-10 vector \( \hat{p}(a) \) via softmax.
     - Regularization: Use an entropy bonus for categorical DPL.
   
4. **Loss Function & Optimization**
   - **Standard Preference Loss (BTL Loss)**: Based on pairwise comparison likelihoods, as in the Bradley-Terry or Thurstone models.
   - **Regularization**: Add \( \frac{\lambda}{2} \sum_{a} \hat{u}(a)^2 \) with \( \lambda > 0 \) to ensure convexity and stability.
   - **Optimization Algorithm**:
     - Use AdamW optimizer.
     - Learning rate schedule: cosine decay from \( \sim3 \times 10^{-6} \) to \( 3 \times 10^{-7} \).
     - Batch Size: 2 comparisons, 4 responses total.
     - Epochs:
       - On datasets with pure helpfulness or harmfulness labels: 2 epochs.
       - On combined datasets: 1 epoch (to balance total gradient steps).

5. **Theoretical Implementation Details**
   - Implement the convex loss functions described:
     - Logistic loss for pairwise preferences.
     - Cross-entropy loss for the categorical likelihood.
   - Ensure the loss computation correctly incorporates the noisy context variable via the models described.

---

### **B. Experimental Setup and Datasets**

1. **Datasets**
   - **Synthetic Data**:
     - Alternatives \( \mathcal{A} = [0,1] \).
     - Hidden context \( z \sim \operatorname{Bernoulli}(0.5) \).
     - True utility functions \( u(a,z) \) specified explicitly (e.g., piecewise).
     - Generate comparison outcomes \( O_u(a,b,z) \) based on true utility.
   - **Real Data**:
     - Use the HH-RLHF dataset (from Bai et al., 2022a) with human annotations for helpfulness and harmlessness.
     - Additional relabeling via GPT-3.5: For each prompt, generate comparison pairs with labels reflecting the true label, then invert for harmfulness to simulate hidden context.

2. **Data Collection & Preprocessing for Experiments**
   - For synthetic:
     - Generate large comparison datasets with known hidden context per *Section 4*.
   - For real data:
     - Use original pairwise comparisons.
     - Perform relabeling:
       - Prompt GPT-3.5 with the two prompts provided.
       - Collect or simulate preference responses, flipping labels for hidden objectives.

3. **Synthetic Experiment Structure**
   - Train models (standard, mean-and-variance DPL, categorical DPL) on datasets with:
     - Just helpfulness labels.
     - Just harmless labels.
     - Combined labels with relabeling to introduce hidden context.
   - Compare models’ ability to recover true utility signals or detect hidden context.

---

### **C. Hyperparameters & Training Procedures**

- **Model Architecture**
  - Base: LLAMA-2-7B with LoRA parameters (dimension ~8).
  - Final linear output layer:
    - 1 scalar (standard preference).
    - 2 scalars (mean & variance DPL).
    - 10 logits (categorical DPL).
  
- **Training Settings**
  - Optimizer: AdamW.
  - Learning rate: start at \( 3 \times 10^{-6} \), decay to \( 3 \times 10^{-7} \).
  - Weight decay: 0.0001.
  - Batch size: 2 comparisons per batch (4 responses).
  - Epochs:
    - 2 epochs for single-objective (helpful or harmful).
    - 1 epoch for combined data (to balance total updates).
  - Regularization: \(\lambda\) tuned to balance convexity and overfitting (initial guess: \( \sim 1 \times 10^{-4} \), but needs validation).

- **Training Losses**
  - **BTL Loss**: As in the derivation, approximate pairwise preference probabilities.
  - **Categorical Loss**: Cross-entropy over the softmaxed output.
  - **Regularization**: L2 on utility output, entropy bonus for categorical distribution.
  
---

### **D. Evaluation Metrics & Analyses**

1. **Preference Accuracy**
   - Compare learned utility vs. true utility in synthetic experiments.
   - Use rank correlation metrics (e.g., Spearman’s \(\rho\), Kendall’s \(\tau\)).

2. **Borda Count & Preference Aggregation**
   - Compute Borda counts of learned utilities.
   - Verify Theorem 3.1: the ordering of \(\hat{u}(a)\) aligns with the Borda count.

3. **Hidden Context Detection**
   - Use the variance (\(\hat{\sigma}^2(a)\)) in DPL to identify alternatives with larger hidden context influence.
   - Compute \( r^2 \)-like metrics to quantify how well the model captures missing feature effects.
   
4. **Jailbreak & Safety Evaluation**
   - Collect response pairs to jailbreak prompts.
   - Measure the frequency of harmful responses, comparing standard RLHF models with DPL models.
   - Use the *jailbreak rate* and *helpfulness accuracy* as primary metrics.

5. **Theoretic Validations**
   - Confirm convexity of loss functions through plotting.
   - Validate the convergence of \(\hat{u}\) to true or expected utilities in synthetic data.

---

### **E. Implementation Notes and Potential Ambiguities**

- **Sample Sizes**:
  - Synthetic datasets should be large enough (~10k comparisons) to observe convergence behaviors as in the paper.
  - For real dataset relabeling, replicate the prompts using OpenAI APIs.

- **Pairwise Comparison Generation**
  - Synthetic: manual simulation based on true utilities.
  - Real: GPT-3.5-based relabeling as detailed, possibly with multiple augmentations for robustness.

- **Model Code & Loss Functions**
  - Implement the loss functions explicitly:
    - Logistic for pairwise preferences.
    - Cross-entropy for categorical DPL.
  - Use automatic differentiation frameworks (PyTorch).

- **Handling Hidden Context**
  - The main experimental focus is to observe how models' estimates (\(\hat{\mu}, \hat{\sigma}\)) reflect hidden context influence.
  - For synthetic, control exactly \(z\).
  - For real, rely on relabeling strategy to embed hidden context.

- **Ambiguities / Missing Details**
  - Exact regularization coefficient \(\lambda\): initial value 0.0001; tune via validation.
  - Use of early stopping or validation set: recommended, especially for real data.
  - For the evaluation of causality in hidden context detection, further statistical tests may be needed but are beyond the scope here.

---

**Summary of the Roadmap:**

1. **Data Preparation**:
   - Generate synthetic data with known \(u(a,z)\).
   - Collect/relabeled real data as described.
2. **Model Construction**:
   - Fine-tune LLAMA-2-7B with LoRA.
   - Implement loss functions for standard, mean-variance DPL, and categorical DPL models.
3. **Training**:
   - Follow prescribed hyperparameters.
   - Regularize properly.
4. **Theoretical Validation**:
   - Validate convexity.
   - Confirm that learned utilities conform to the theory (e.g., Borda count ordering).
5. **Experiment Analysis**:
   - Compare with true utilities (synthetic).
   - Detect hidden context influence through variance.
   - Test robustness against jailbreak or unsafe prompts.
   
This plan provides a detailed, step-by-step strategy to accurately reproduce and validate the core claims and experimental results from the paper.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will build a modular system for training and evaluating preference models with hidden context. The core components include data loading with synthetic or relabeled datasets, a neural model with flexible output heads (scalar, mean-variance, categorical), a training pipeline implementing the described loss functions (BTL loss for preference pairs, cross-entropy for categorical), and evaluation modules for comparing learned utilities to true utilities or Borda counts. We will leverage HuggingFace Transformers for LLAMA-2-7B base, PyTorch for modeling and optimization, and custom scripts for synthetic data generation and relabeling via GPT. The system will be driven from a main script that initializes dataset, model, trainer, and evaluator, executing training and validation stages sequentially.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "synthetic_data.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_type: str, relabel: bool)\n        +load_data() -> Dataset\n        +generate_synthetic_data() -> Dataset\n        +relabel_data(dataset: Dataset) -> Dataset\n    }\n    class Dataset {\n        +data: List[ComparisonPair]\n        +labels: List[int]\n    }\n    class ComparisonPair {\n        +prompt_response_a: str\n        +response_b: str\n        +preference: int # 1 if a preferred, 0 if b preferred\n        +label_objective: str # helpful/harmful if annotated\n    }\n    class PreferenceModel {\n        +__init__(model_name: str, head_type: str, num_outputs: int)\n        +predict(a: str, b: str) -> PairwisePreference\n        +predict_distribution(a: str) -> Distribution\n    }\n    class Trainer {\n        +__init__(model: PreferenceModel, dataset: Dataset, lambda: float, learning_rate: float, epochs: int)\n        +train() -> None\n        +save() -> None\n    }\n    class Evaluation {\n        +__init__(model: PreferenceModel, dataset: Dataset)\n        +evaluate() -> EvaluationMetrics\n        +compute_borda_counts() -> Dict[str, float]\n        +detect_hidden_context() -> List[Alternative]\n    }\n    class Program {\n        +main()\n    }\n\n    Main --> DatasetLoader\n    Main --> PreferenceModel\n    Main --> Trainer\n    Main --> Evaluation\n    DatasetLoader -- loads --> Dataset\n    Dataset -- used by --> Trainer\n    PreferenceModel -- trained by --> Trainer\n    PreferenceModel -- evaluated by --> Evaluation\n    Trainer -- updates --> PreferenceModel\n    Evaluation -- reports --> Main\n",
    "Program call flow": "sequenceDiagram\n    participant MS as MainScript\n    participant DL as DatasetLoader\n    participant DM as PreferenceModel\n    participant TR as Trainer\n    participant EV as Evaluation\n    MS->>DL: __init__(dataset_type, relabel)\n    alt generate synthetic\n        DL->>DL: generate_synthetic_data()\n    else relabel dataset\n        DL->>DL: relabel_data()\n    end\n    MS->>DL: load_data()\n    DL-->>MS: dataset\n    MS->>DM: __init__(model_name='llama-2-7b', head_type='scalar/variance/categorical')\n    MS->>TR: __init__(model=DM, dataset, lambda, lr, epochs)\n    TR->>DM: train()\n    DM-->>TR: updated model\n    MS->>EV: __init__(model=DM, dataset)\n    EV->>DM: evaluate()\n    DM-->>EV: metrics\n    EV-->>MS: report\n    MS->>MS: save models\n    end\n",
    "Anything UNCLEAR": "Could you clarify if the synthetic dataset should mimic specific preference distributions (e.g.,hidden context effects as in the paper) or can be uniformly generated? Also, what are exact hyperparameters for LoRA tuning? Should we implement all variants of DPL or focus on the mean-and-variance model?"
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "transformers==4.28.1",
        "numpy==1.21.0",
        "scipy==1.10.1",
        "tqdm==4.65.0",
        "pyyaml==6.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "synthetic_data.py",
            "Defines functions to generate synthetic alternatives and comparison pairs with known hidden context functions. Implements data structures for comparisons and functions to simulate preferences based on true utilities, including noisy contexts. Depends on numpy and scipy for distributions."
        ],
        [
            "dataset_loader.py",
            "Implements DatasetLoader class with methods: load_data() for pre-existing datasets (e.g., relabeled HH-RLHF), generate_synthetic_data() for synthetic cases, relabel_data() for relabeling via GPT API. The class handles dataset creation, relabeling, and batch preparation. Depends on standard libraries and needs async API calls for relabeling."
        ],
        [
            "model.py",
            "Defines the neural network class PreferenceModel with flexible output heads: scalar, mean & variance, or categorical. Implements forward passes, outputs distribution parameters or logits. Supports loading pre-trained LLAMA-2-7B and integrating LoRA. Uses transformers library and PyTorch."
        ],
        [
            "trainer.py",
            "Defines Trainer class that handles model training. Uses loss functions: pairwise logistic (BTL), cross-entropy for categorical, and regularization terms. Implements train() method with learning rate schedule, regularization, and epoch control. Uses torch.optim and supports saving checkpoints."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class with methods evaluate(), compute_borda_counts(), and detect_hidden_context(). Supports calculating model performance metrics, correlation with true utilities, and variance analysis to identify hidden context influence. Uses numpy, scipy, and custom logic based on model outputs."
        ],
        [
            "main.py",
            "Top-level script initializing dataset loader, model, trainer, and evaluator. Parses config, handles command-line args. Executes data loading (synthetic or real), model instantiation, training, and evaluation. Coordinates API calls and logs results."
        ]
    ],
    "Task list": [
        "synthetic_data.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (no front-end API annotations needed; focus on internal modules)",
    "Shared Knowledge": "Models share utility functions for loss computation, data formatting, and regularization. Dataset loader and relabeler should share data batching and formatting routines. The model module must be compatible with both synthetic and real datasets, using the configurable head types.",
    "Anything UNCLEAR": "Clarify the expected size of synthetic datasets, hyperparameter settings for LoRA tuning, and whether to focus exclusively on mean-variance DPL or support all variants for experimental comparison."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 3e-6             # Starting learning rate as per paper
  min_learning_rate: 3e-7        # Final learning rate after schedule decay
  batch_size: 2                  # Comparisons per batch (4 responses)
  epochs: 2                      # For datasets with single labels
  lambda_reg: 0.0001             # Regularization coefficient for convexity
  regularization_type: l2        # Type of regularization
  optimizer: AdamW
  weight_decay: 0.0001
  lr_schedule: cosine            # Cosine decay schedule
  lr_decay_epochs: null           # Use default cosine decay over epochs
  total_steps: null               # To be computed based on dataset size and epochs

model:
  base_model: llama-2-7b-hf       # Using pre-trained LLAMA-2-7B model
  head_type: scalar                # Options: scalar, mean_var, categorical
  num_outputs: 1                   # 1 for standard, 2 for mean & variance, 10 for categorical
  lora_rank: 8                     # LoRA implementation detail (dimensionality)
  
dataset:
  dataset_type: relabeled_hh_rlhf  # Include synthetic, relabeled, and real datasets
  synthetic_size: 10000            # Number of synthetic comparisons (adjust as needed)
  relabel: false                   # Whether to perform GPT relabeling

loss:
  preference_loss: logistic        # Logistic Bradley-Terry pairwise loss
  regularization: l2               # L2 regularization on utility outputs
  entropy_bonus: 0.1               # For categorical DPL, entropy regularization weight
  use_regularization: true
  
optimization:
  optimizer: AdamW
  learning_rate: 3e-6             # Initial LR
  min_learning_rate: 3e-7        # Final LR
  weight_decay: 0.0001
  scheduler: cosine
  epochs: 2                        # for single-label datasets
  batch_size: 2
  lambda_reg: 0.0001
  total_steps: auto                # Calculated based on dataset size and epochs

evaluation:
  metrics:
    preference_accuracy: true      # Accuracy over pairwise preferences
    rank_correlation: true         # Spearman or Kendall correlation
    bias_detection: true           # Variance and hidden context detection
  compare_model: true              # Compare with true utilities (synthetic)
  save_best_model: true            # Save best performing checkpoint
```

---

**Note:** Full configuration is available in `planning_config.yaml`
