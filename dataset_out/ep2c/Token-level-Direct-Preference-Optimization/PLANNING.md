# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, structured plan designed to precisely reproduce the experiments and methodology described in the paper on Token-level Direct Preference Optimization (TDPO). This plan covers the core methodological steps, experimental setup, hyperparameters, datasets, and evaluation metrics, ensuring fidelity to the original work.

---

# 1. **Understanding and Implementing the Methodology**

## A. Core Conceptual Framework
- **Reformulate the preference optimization as token-level KL-constrained advantage maximization**:
  - **Token-level Markov Decision Process (MDP)**:
    - States: `[x, y^{<t}]`, prompt + generated tokens up to `t-1`.
    - Actions: next token `y^t`.
    - Reward: token-wise `$R_t$`.
  - **Policy Parameterization**:
    - Use a large language model (like GPT-4) with fine-tuning via TDPO objectives.
    - The policy $\pi_\theta$ predicts token distributions conditioned on input prompts and previous tokens.
  
- **Token-level reward and advantage functions**:
  - Derive the advantage `A_\pi` at each token using the sequence of rewards and value functions.
  - Use the formulations from Lemmas 4.1, 4.2, and Theorems 4.5, 4.6 to connect reward functions, advantage, and the optimal policy.

- **Preference Model Reformulation**:
  - Based on the Bradley-Terry (BT) or Regret Preference Model:
    - Express human preferences as probability of one response over another, parameterized by token-level reward and KL divergence terms.
    - The model assesses preferences via a sigmoid of the difference in token rewards minus the divergence penalty (`u(x, y_1, y_2) - δ(x, y_1, y_2)`).

## B. Loss Functions
- Implement **Token-level DPO**:
  - Use the derived token-wise formulations eq. 7 and the modifications eq. 15 & 16.
  - **Two versions**:
    1. `TDPo_1`: Using a straightforward difference in advantage functions.
    2. `TDPo_2`: Introducing a scaling parameter $\alpha$ and stop-gradient `sg()` to control divergence.
  - Implement the loss function as a **log-sigmoid** of the weighted advantage difference, incorporating KL constraints at token level, as in graphs.

- **Gradient Computation & Regularization**:
  - Carefully compute:
    - `∇θ u(x, y_w, y_l)` for the reward difference.
    - `∇θ δ(x, y_w, y_l)` for the divergence penalty, optionally with stop-gradient.
  - Include the hyperparameters: `beta` (KL trade-off), `alpha`, `a` (scaling factor).

## C. Mapping to the Optimal Policies
- For each pair of responses `(y_w, y_l)`, compute:
  - Token-wise reward differences from the current policy and reference model.
  - Token divergence penalties.
- Use the closed-form optimal policy from Lemma A.2:
  \[
  \pi_\theta^*(z | [x, y^{<t}]) \propto \pi_{ref}(z|[x, y^{<t}]) \exp\left(\frac{1}{\beta}Q_{ref}([x,y^{<t}], z)\right)
  \]
- To approximate, model the reward `Q_{ref}` via:

  - Estimating it as (from eq. 38): 
    \[
    Q_{ref}([x, y^{<t}], z) \approx \beta \log \frac{\pi_\theta(z|[x, y^{<t}])}{\pi_{ref}(z|[x, y^{<t}])} + \beta \log Z([x, y^{<t}]; \beta)
    \]
  - Since `Z()` is intractable, approximate it empirically via sampling or normalization.

## D. Implementing the BT Model & Preference Likelihood
- At each token step, compute the preference probability (Eq. 12, Theorem 4.5):
  \[
  P_{BT}(y_1 \succ y_2|x) = \sigma\left(u(x, y_1, y_2) - \delta(x, y_1, y_2)\right)
  \]
- These probabilities inform the loss, as in eq. 15 & 16.

---

# 2. **Experimental Setup**

## A. Datasets
- **IMDb sentiment classification dataset** (for single-turn dialogue preference experiments):
  - Use provided prompts (movie reviews or summaries).
  - Responses: human-labeled responses, or responses from GPT-4.
  - Human preferences: derived from GPT-4 evaluating response quality.
  - Generate pairwise response comparisons based on human annotations or GPT-4 preferences.

- **Antropic HH Dataset**:
  - Use the labeled dialogue dataset with preferences for safe, aligned responses.
  - Responses: GPT-4 output compared against model outputs or human labels.
  - Hyperparameters (like `α`) tuned on validation preferences.

- **MT-bench Dataset**:
  - Use the **scoring data** provided (responses compared via GPT-4).
  - Construct pairwise comparisons, and record "win/tie/lose" ratios.

## B. Data Processing
- For each dataset:
  - Prompt the model with the predefined prompt or question.
  - Generate multiple responses per prompt via a base model (e.g., GPT-4 or finetuned version).
  - Create response pairs `(y_w, y_l)` labeled according to human or GPT preferences.
  - Convert pairwise preferences into probability labels for the BT model in the loss.

## C. Response Generation & Sampling
- During training:
  - Generate responses token-by-token using the current policy (`π_θ`) conditioned on prompt + previous tokens.
  - Use top-K sampling or nucleus sampling for diversity.
  - Maintain maximum sequence length (`T`), e.g., 512 tokens.

## D. Hyperparameters and Controls
- **KL constraint parameters**:
  - `β` (KL temperature), e.g., values like 0.1, 0.5, 1.0.
  - `α` (divergence balancing coefficient), e.g., 0.1, 0.3, 0.5, 0.7.
  - `a` for divergence control in eq. 16.
- **Training**:
  - Batch size: e.g., 64 for mini-batches.
  - Number of steps: 150-200 steps with early stopping if convergence seen.
  - Learning rate: e.g., `5e-6` or as per GPT-4 fine-tuning standards.
  - Optimizer: AdamW.

- **Pretrained Base Model**:
  - Use GPT-4 **via API** or open-source equivalents (like LLama, GPT-J) with similar capacity.
  - Fine-tune with token-level objectives.

---

# 3. **Evaluation Metrics & Analyses**

## A. Main Metrics
- **Alignment**:
  - GPT-4 evaluated "win rate" on the test set response pairs (see Figures 4-6).
  - Use GPT-4 or human raters to compare model responses against preferred responses.
  - Calculate the percentage of wins, ties, losses.

- **Diversity**:
  - Entropy of generated responses (response token distribution).
  - Sequence KL divergence to reference or previous models.

- **KL Divergence**:
  - Sequential KL divergence at token level for preferred and dispreferred responses.
  - Monitor divergence growth over training steps for stability and control.

## B. Downstream Evaluations
- Plot frontier curves (reward vs. KL divergence).
- Track "margin" between preferred and dispreferred response divergences.
- Measure the rate (`growth`) of divergence for model diagnostics.

---

# 4. **Implementation and Debugging Checklist**
- Implement pairwise comparison sampling during training.
- Ensure proper calculation of token-level advantage, reward, and divergence:
  - Use a reference model (GPT-4 or Open-Source equivalent) for the baseline Q values.
- Implement stop-gradient (`sg()`) for `δ` computations in `TDPo_2`.
- Stabilize training with gradient clipping and learning rate warm-up.
- Regularly validate preference accuracy on validation sets.
- Reproduce Figures 3-6: reward frontier, divergence curves, win rates.

---

# 5. **Summary Timeline and Iteration Strategy**
- **Weeks 1-2**:
  - Gather datasets, prepared prompts, responses, and pairwise labels.
  - Implement base response generation with a pre-trained GPT model.
  - Set up the pairwise comparison framework and reference model.

- **Weeks 3-4**:
  - Implement the token-level TDPO loss functions (`TDPo_1`, `TDPo_2`).
  - Integrate the BT preference model into loss.
  - Establish the advantage & reward estimations.

- **Weeks 5-6**:
  - Conduct hyperparameter tuning on validation preference metrics.
  - Generate initial curves: reward frontier, KL divergence plots.

- **Weeks 7-8**:
  - Evaluate on test datasets, compare win rates.
  - Reproduce key figures (Figures 3-6).

---

# Final Notes:
- Exact replication of the intractable normalization/constants (`Z`) will likely require approximation via sampling or normalization over token probability distributions.
- Since the work heavily involves token-wise computations, ensure tensor shapes are consistent, and avoid detachment errors in gradient flow.
- Maintain logs and checkpoints enabling ablation studies (varying `α`, `β`, divergence control strategies).

---

This comprehensive roadmap enables later precise implementation, debugging, and replication of the original experiments. It resets all assumptions, emphasizes necessary derivations, and highlights the key aspects outlined in the paper.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular pipeline using PyTorch and Hugging Face transformers to implement token-level TDPO. The core components include data loading, response generation, pairwise comparison, and training with the custom loss functions. We'll utilize a pretrained language model (like GPT-2 or a similar open-source model) and wrap it with necessary functions for reward and advantage computation, divergence constraints, and preference modeling. The training will alternately update model parameters based on pairwise preferences, using the loss formulations from the paper, with divergence control via the specified alpha and beta hyperparameters. Evaluation will involve using the GPT-4 API for preference scoring and divergence measurement, with plotting tools for tracing reward-frontier and KL divergence curves.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "losses.py",
        "utils.py",
        "experiments_config.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(data_path: str)
        +load_data() -> Dataset
        +get_response_pair(prompt: str) -> Tuple[str, str, int]
    }
    class ResponseGenerator {
        +__init__(model_name: str, load_from_checkpoint: Optional[str])
        +generate_response(prompt: str, max_tokens: int) -> str
        +get_probability_distribution(tokens: List[str], context: str) -> Tensor
    }
    class PreferenceModel {
        +__init__(reference_model: ResponseGenerator, beta: float)
        +compute_preference(y1: str, y2: str, prompt: str) -> float
        +estimate_divergence(y1: str, y2: str, prompt: str) -> float
    }
    class CustomLoss {
        +__init__(beta: float, alpha: float, divergence_scale: float)
        +compute_loss(
            responses_w: List[str], 
            responses_l: List[str], 
            prompt: str, 
            response_probs_w: Tensor, 
            response_probs_l: Tensor, 
            divergence_penalty: float
          ) -> Tensor
    }
    class Trainer {
        +__init__(model: ResponseGenerator, loss_fn: CustomLoss, dataset: Dataset, preference_model: PreferenceModel)
        +train(steps: int) -> None
        +save_checkpoint(filepath: str) -> None
        +load_checkpoint(filepath: str) -> None
    }
    class Evaluation {
        +__init__(model: ResponseGenerator, dataset: Dataset, preference_model: PreferenceModel)
        +evaluate() -> dict
        +plot_divergence_curves() -> None
        +compute_win_rates() -> dict
    }
    Main --|> DatasetLoader
    Main --|> ResponseGenerator
    Main --|> PreferenceModel
    Main --|> CustomLoss
    Main --|> Trainer
    Main --|> Evaluation
    ResponseGenerator --> +get_probability_distribution()
    Trainer --> ResponseGenerator
    Trainer --> CustomLoss
    Evaluation --> ResponseGenerator
    Evaluation --> PreferenceModel
    CustomLoss --> ResponseGenerator
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant RG as ResponseGenerator
    participant PM as PreferenceModel
    participant L as CustomLoss
    participant T as Trainer
    participant E as Evaluation
    M->>DL: initialize and load_data()
    DL-->>M: dataset
    M->>RG: load pretrained model / checkpoint
    RG-->>M: model instance
    M->>PM: initialize reference model
    PM-->>M: preference handling
    M->>L: initialize with beta, alpha, divergence_scale
    M->>T: train(model, data, loss_fn, preference_model) for N steps
    T->>RG: generate response tokens
    T->>PM: compute preference y1 vs y2
    T->>L: compute token-level loss
    T-->>M: training complete
    M->>E: evaluate model on test set
    E->>RG: generate responses
    E->>PM: score preferences
    E-->>M: metrics & divergence plots
    M-->>Program: End
",
    "Anything UNCLEAR": "Clarification needed on the reference model used for the reward estimation, whether to use GPT-2 or a custom model; specifics of the reference Q-value estimation; dataset format details; and exact response generation constraints (e.g., max tokens, sampling strategy). Additionally, confirm the API access or open-source model availability for GPT-4 evaluation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.1",
        "transformers==4.26.1",
        "numpy==1.21.0",
        "matplotlib==3.5.3",
        "tqdm==4.64.1"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class. Loads datasets, processes pairwise response data, and provides APIs to get response pairs for a prompt. Depends on dataset file formats and preprocessing steps."
        ],
        [
            "model.py",
            "Defines ResponseGenerator class, encapsulating pretrained models (e.g., GPT-2 or open-source equivalent). Implements methods for token sampling, probability distribution retrieval, and model checkpoint management."
        ],
        [
            "losses.py",
            "Implements classes/functions for computing token-level contrastive loss, divergence constraints, and the specialized TDPO loss formulations per the paper (including TDPo_1 and TDPo_2)."
        ],
        [
            "utils.py",
            "Contains utility functions such as sampling wrappers, KL divergence calculators, and plotting functions for divergence curves and reward frontiers."
        ],
        [
            "trainer.py",
            "Defines Trainer class that takes the model, custom loss, dataset, and preference scorer, and manages per-epoch training, including response generation, pairwise response comparison, loss computation, gradient update, and checkpoint saving."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class for model validation, computing win/tie/lose rates with GPT-4 API or human feedback, plotting divergence and reward frontiers, and evaluating the final model on test datasets."
        ],
        [
            "main.py",
            "Entry point script that initializes configuration, loads data, models, preference scorer, runs training loops, and triggers evaluation steps. Coordinates overall experiment flow."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "losses.py",
        "utils.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (not applicable, reserved for potential future API communication, e.g., web interface or distributed training control)",
    "Shared Knowledge": "Both dataset_loader.py and trainer.py will share dataset formats, response tokenization procedures, and response pair extraction methods. utils.py provides divergence and plotting functions called by evaluation.py and trainer.py.",
    "Anything UNCLEAR": "Clarification needed on the exact reference model and Q-value estimation method—whether to use a fine-tuned open-source justification or GPT-4 API. Also, details about dataset schema and pairwise preference labeling for implicit human preference simulation."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 5e-6
  batch_size: 64
  epochs: 10
  max_response_tokens: 512
  divergence_beta: 0.1
  alpha: 0.5
  divergence_scale: 1.0
  train_steps: 200
  warmup_steps: 50
  gradient_clipping: 1.0
model:
  pretrained_model_name: gpt2-medium
  checkpoint_path: null  # specify if resuming from checkpoint
dataset:
  train_data_path: "path/to/train/dataset"
  validation_data_path: "path/to/validation/dataset"
  test_data_path: "path/to/test/dataset"
evaluation:
  use_gpt4: true
  gpt4_api_key: "YOUR_API_KEY"
  evaluation_interval: 50  # steps between evaluations
  save_checkpoint_interval: 100  # steps
hyperparameters:
  beta: 0.1  # KL divergence coefficient
  alpha: 0.5  # divergence balancing in TDPo_2
  divergence_offset: 1.0  # for divergence offset in loss
  stop_gradient: true  # whether to stop gradient for divergence term
```

---

**Note:** Full configuration is available in `planning_config.yaml`
