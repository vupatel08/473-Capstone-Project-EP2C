# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, structured plan to reproduce the proposed method (LiRE) and its experimental validation, based on the provided paper content. The plan is organized into major sections for clarity: (A) Methodology, (B) Dataset Construction, (C) Implementation details, (D) Experimental setup, (E) Evaluation metrics, and (F) Additional considerations.

---

## A. Methodology Overview and Implementation Steps

### 1. Constructing the Ranked List of Trajectories (RLT)
- **Input:** Offline preference feedbacks (ternary labels: preferred, equal, dispreferred) between trajectory segments.
- **Assumptions:**
  - Completeness: any pair of segments has feedback as preferred, dispreferred, or equal.
  - Transitivity: preference relations are transitive (robustness tested even if this doesn't strictly hold).
- **Core idea:**
  - Sequentially insert segments into an ordered list `L`.
  - Initialize `L` with a random segment.
  - For each new segment:
    - Select a candidate segment from the dataset.
    - Perform binary search over current list segments to find the correct insertion point, based on preference feedback.
    - Feedback is obtained by comparing the candidate with a segment in `L`, using the same ternary labels.
    - Insert the segment at the determined position.
- **Implementation details:**
  - Use the binary search algorithm:
    - For each compare:
      - Query feedback (preference) via a human or synthetic oracle.
      - The feedback informs whether the candidate is preferred, equal, or less preferred than the list segment at the mid.
  - *Note on scalability:* For very large datasets, approximate methods or heuristics (like merging partial RLTs) can be considered, but standard binary insertion suffices for the described experiments.

### 2. Constructing the Reward Model
- **Reward model architecture:**
  - Use a neural network with specified layers (e.g., 3 hidden layers, 128 dims, ReLU activations).
  - Final activation: Tanh (bounded output for reward in [0, R]).
- **Training objectives:**
  - **Listwise Loss (preferred):** Minimize the Kullback-Leibler divergence between list probability distributions derived from the modeled scores.
  - **Pairwise Loss (alternative):** Use preference pairs to compute contrastive or logistic loss.
  - Justify choice: The implementation prefers the listwise (ListNet) loss for stability.
- **Input features:**
  - Trajectory segments features (could be image embeddings, state-action sequences, or handcrafted features; not explicitly specified, so choose a method consistent with the environment).
- **Scoring functions:**
  - Exponential (`exp(x)`) or linear (`x`).
  - Use the hyperparameters and code snippets provided to implement both variants.
- **Training procedure:**
  - Sample `n=10` segments from the current RLT "list".
  - Compute scores for segments via the neural network.
  - Compute the probability distribution (`P_s(i)`) over the permutation according to ListNet.
  - Minimize the KL divergence loss over the sampled segments.
  - Use Adam optimizer with specified hyperparameters.
  - Learning rate: `1e-3`.
  - Batch size: 512 segments.
  - Number of epochs: 300 (or until convergence).
  - Regularization: if applicable, include weight decay or early stopping based on validation.

### 3. Policy Optimization with Learned Rewards
- **Offline RL setup:**
  - Use the learned reward model to assign reward signals to the dataset episodes.
  - Implement an off-policy RL algorithm (e.g., IQL, DDPG, or a similar Q-learning-based method).
  - Base reward input: model output from reward network.
- **Training:**
  - For each episode or trajectory:
    - Compute per-step reward using the reward network.
  - Use standard offline RL training with the dataset, tuning hyperparameters for stability.
  - Replay buffer: for exploration, use the constructed or existing dataset.
- **Evaluation of policy:**
  - Run policy in simulation.
  - Measure success rate on predefined tasks.

---

## B. Dataset Construction and Preference Feedback Collection

### 1. Offline Datasets
- **Medium-replay datasets:**
  - Generate or use existing datasets from environments like Meta-World:
    - Collect 200-500 trajectories per task.
    - For each trajectory, store sequences and associated features.
  - Tasks of interest: button-press-topdown, box-close, dial-turn, sweep, sweep-into, drawer-open, lever-pull.
  - Ensure dataset coverage over many states and actions for robustness.
- **Expert datasets:**
  - Collect 500-1000 preference labels from human or synthetic "expert" feedback:
    - Human feedback could be simulated by time-based heuristic (e.g., proximity to goal) or actual human labelers.
    - Synthetic feedback can be generated from ground-truth rewards if available.
  - Feedback types:
    - Binary preference: preferred vs dispreferred.
    - Prefer same/equal/different (ternary).
    - For listwise construction, generate preference lists over multiple segments.
- **Synthetic noise:**
  - Inject noise (e.g., flip preference labels with some probability) to simulate imperfect human feedback for robustness testing.

### 2. Preference Feedback Data
- **Preference labels per pair:**
  - For each pair, store labels:
    - Preferred segment ID.
    - Dispreferred segment ID.
    - Equal segments (if including second-order preference).
- **Second-order preference / listwise data:**
  - For each trajectory, generate a ranked list of segments based on their ground-truth reward or task-specific heuristics.
- **Data management:**
  - Store feedback in a structured format:
    - List of tuples `(segment_a, segment_b, label)`.
    - For listwise feedback: `list_of_segments, label` (ranking).

---

## C. Technical Implementation Details

### 1. Neural Network Model
- Hyperparameters:
  - 3 fully connected layers, each of 128 units.
  - Activation: ReLU.
  - Final activation: Tanh scaled to [0, R].
  - Dropout: optional (e.g., 0.1).
- Loss functions:
  - Listwise KL divergence using softmax over scores.
  - Implement both exponential and linear scoring variants, compare.
- Optimization:
  - Adam optimizer, learning rate `1e-3`.
  - Use early stopping based on validation if possible.

### 2. Preference Query Simulation
- For synthetic evaluation, use ground-truth rewards to generate preference labels:
  - Segment with higher ground-truth reward preferred.
  - Equal reward: label as equal.
- For human simulation, optionally add noise and second-order preferences.

### 3. Hyperparameters
- Number of segments sampled per iteration: 10.
- List length for listwise loss: 16 segments (per the paper).
- Number of epochs: 300.
- Batch size: 512.
- Regularization: weight decay `3e-4`.
- Noise levels: 0, 0.1, 0.3 preference noise.

---

## D. Experiments and Evaluation Protocol

### 1. Offline Reward Estimation
- Train reward models using the constructed preference datasets:
  - With listwise loss (preferred) and pairwise loss.
  - Use both scoring functions (`exp` and `linear`).
- Validation:
  - Use a held-out set of segments with ground-truth rewards for correlation analysis:
    - Calculate Pearson correlation in reward estimates vs GT rewards.
  - Visualize scatter plots, distribution of estimated vs ground-truth.

### 2. Offline Policy Training
- Use ground-truth rewards (GT) for baseline IQL.
- Use predicted rewards (LiRE, MR, others) as reward signals.
- Measure success rate in environment:
  - Success criteria as task-specific (e.g., button pressed, object close).
  - Report the average success over random seeds.

### 3. Ablation Studies
- Vary number of preference feedback (50, 200, 500, 1000).
- Vary list length (`Q` in listwise loss).
- Combine LiRE with other baselines (OPRL, PT, IPL), evaluate combined success.
- Test robustness to feedback noise (Gaussian, binary flip).

---

## E. Evaluation Metrics

1. **Reward Estimation Quality:**
   - Pearson correlation with ground-truth reward.
   - Scatter plots of estimated vs.
   - R-squared metrics or mean-squared error if ground rewards are available.

2. **Policy Success Rate:**
   - Percentage of episodes where task is considered successful.
   - Average episode return.

3. **Sample Efficiency:**
   - Success rate vs preference feedback budget.
   - Learning curves over training steps.

4. **Ablation Metrics:**
   - Effect of list length `Q`.
   - Effect of preference noise.
   - Success rate of combining multiple reward models and listwise vs pairwise loss.

---

## F. Additional Implementation and Experimental Considerations

- **Data Storage:**
  - Store datasets of trajectory segments, preference labels, and dataset metadata separately.
- **Synthetic Human Feedback:**
  - For initial experiments, generate synthetic labels based on ground rewards.
  - For robustness, add label noise as per noise schedule.
- **Benchmarking:**
  - Compare LiRE with MR, PT, IPL, SeqRank, and other baselines on success rate.
- **Code modularity:**
  - Separately implement:
    - RLT construction module.
    - Reward model training module.
    - Offline RL training module.
    - Evaluation and plotting scripts.
- **Reproducibility:**
  - Fix random seeds for all stochastic components.
  - Log hyperparameters, dataset details, and model checkpoints.

---

## Summary of the Roadmap

- **Step 1:** Collect or simulate preference-annotated datasets (preferably with ground truth rewards for validation).
- **Step 2:** Construct the RLT sequentially, using binary search with preference feedback.
- **Step 3:** Train the reward model using listwise loss (preferably with the exp scoring function, but compare linear).
- **Step 4:** Use the trained reward to evaluate reward correlation and to generate reward signals for offline RL.
- **Step 5:** Run offline RL algorithms (IQL or others) with GT and learned rewards.
- **Step 6:** Conduct ablation studies varying feedback amount, list length, noise, and reward functions.
- **Step 7:** Visualize reward estimate accuracy (scatter plots, correlation) and policy success rate/return.
- **Step 8:** Analyze robustness to feedback noise and scenario complexity, compare performance against baselines.

---

This comprehensive plan should prepare you well for an implementation phase, focusing on each core component step-by-step with detailed attention to the methodology, dataset creation, hyperparameters, and experimental validation.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Build a modular system utilizing open-source libraries: PyTorch for neural networks, NumPy and Pandas for data handling, Gym/Meta-World for environments, and standard Python utilities for dataset and experiment management. The core pipeline includes constructing the RLT using preference feedback, training the reward model with listwise loss, and offline RL policy training with the learned reward. Human or synthetic preference data is simulated for offline dataset creation. The system sequentially builds the RLT, trains the reward model with open-source loss implementations, evaluates reward correlations, and then performs offline policy training and success measurement. Hyperparameters are configurable via a config file.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "rlt_builder.py",
        "reward_model.py",
        "reward_trainer.py",
        "offline_rl.py",
        "evaluation.py",
        "utils.py",
        "config.yaml"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment() -> None\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str)\n        +load_preference_data() -> PreferenceDataset\n        +load_episodes() -> List[TrajectorySegment]\n    }\n    class PreferenceDataset {\n        +preference_pairs: List[Tuple[SegmentID, SegmentID, PreferenceLabel]]\n        +listwise_rankings: List[List[SegmentID]]\n    }\n    class Segment {\n        +segment_id: str\n        +features: np.ndarray\n    }\n    class RLTBuilder {\n        +__init__(preference_data: PreferenceDataset)\n        +construct_rlt() -> List[Segment]\n        +insert_segment(segment: Segment, compare_fn: Callable[[Segment, Segment], PreferenceLabel]) -> None\n    }\n    class RewardModel {\n        +__init__(config: dict)\n        +train(train_data: List[Segment], preference_data: PreferenceDataset) -> None\n        +predict(segment: Segment) -> float\n        +score_function: Callable[[float], float]\n    }\n    class RewardTrainer {\n        +__init__(reward_model: RewardModel, train_segments: List[Segment], preference_data: PreferenceDataset)\n        +train() -> None\n    }\n    class OfflineRLPolicy {\n        +__init__(reward_function: Callable[[Segment], float], environment: GymEnv or MetaWorldEnv)\n        +train() -> Policy\n        +evaluate() -> dict\n    }\n    class Evaluation {\n        +__init__(policy: Policy, environment: GymEnv or MetaWorldEnv)\n        +measure_success() -> float\n        +plot_reward_scatter(ground_truth_rewards: np.ndarray, estimated_rewards: np.ndarray) -> None\n    }\n\n    Main --> DatasetLoader\n    Main --> RLTBuilder\n    Main --> RewardModel\n    Main --> RewardTrainer\n    Main --> OfflineRLPolicy\n    Main --> Evaluation\n    RewardTrainer --> RewardModel\n    OfflineRLPolicy --> RewardModel\n    "
    ,
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant RB as RLTBuilder\n    participant RM as RewardModel\n    participant Rt as RewardTrainer\n    participant RL as OfflineRLPolicy\n    participant EV as Evaluation\n    M->>DL: load_preference_data()\n    activate DL\n    DL-->>M: preference_data\n    deactivate DL\n    M->>RB: construct_rlt(preference_data)\n    activate RB\n    RB-->>M: rlt_segments\n    deactivate RB\n    M->>RM: initialize(config)\n    activate RM\n    RM-->>M: reward_model\n    deactivate RM\n    M->>Rt: train_reward_model(reward_model, rlt_segments, preference_data)\n    activate Rt\n    Rt-->>RM: trained model\n    deactivate Rt\n    M->>RL: initialize(policy_params, reward_fn=reward_model.predict)\n    activate RL\n    RL-->>Policy: policy object\n    deactivate RL\n    M->>RL: train_policy(policy, environment)\n    activate RL\n    RL-->>EV: evaluate(policy, environment)\n    activate EV\n    EV-->>: success_rate\n    EV-->>: reward_scatter\n    deactivate EV\n    deactivate RL\n    "
    ,
    "Anything UNCLEAR": "Clarification needed on the exact format of preference labels and segments (are they images, states, low-dimensional features?) and whether ground-truth rewards are available for validation. Also, confirmation on environment types (Gym or Meta-World) and preferred RL algorithms (e.g., IQL or others). Hyperparameter optimization details and data collection procedures should be clarified for consistency."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "PyYAML==5.4",
        "scikit-learn==0.24",
        "pandas==1.3.3",
        "matplotlib==3.4.3",
        "gym==0.21.0",
        "metaworld==1.1.2"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines class DatasetLoader: responsible for loading preference data, trajectories, ground-truth rewards, and environment details. Depends on preference feedback files, environment APIs, and auxiliary data formats."
        ],
        [
            "rlt_builder.py",
            "Defines class RLTBuilder: constructs the Ranked List of Trajectories (RLT) by sequentially inserting segments using preference feedback. Depends on DatasetLoader, preference data, and comparison functions."
        ],
        [
            "reward_model.py",
            "Defines class RewardModel: neural network with specified architecture. Contains methods for initialization, training (using listwise or pairwise loss), and inference (score prediction). Depends on hyperparameters, preference data, and environment features."
        ],
        [
            "reward_trainer.py",
            "Defines class RewardTrainer: manages training of RewardModel with preference data via listwise loss, handling batch sampling from RLT, and optimization routines. Depends on RewardModel, preference data, and hyperparameters."
        ],
        [
            "offline_rl.py",
            "Defines class OfflineRLPolicy: manages offline policy training using the learned reward function. Depends on environment APIs, the reward prediction method, and the RL algorithm (e.g., IQL)."
        ],
        [
            "evaluation.py",
            "Defines class Evaluation: runs policy rollouts, measures success rate, and plots reward scatter. Depends on trained policy, environment, and ground-truth reward data."
        ],
        [
            "main.py",
            "Entry point script: initializes configuration, orchestrates dataset loading, RLT construction, reward model training, offline policy training, evaluation, and results plotting. Coordinates dependencies among other modules."
        ],
        [
            "utils.py",
            "Includes utility functions: data normalization, preference label simulation, environment setup, and hyperparameter loading. Shared among core modules."
        ]
    ],
    "Task list": [
        "utils.py",
        "dataset_loader.py",
        "rlt_builder.py",
        "reward_model.py",
        "reward_trainer.py",
        "offline_rl.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Preference data can be synthetic or human-labeled; environments include Gym and Meta-World. The reward model uses configurable scoring functions (exp, linear). Hyperparameters are stored in config.yaml and passed to modules.",
    "Anything UNCLEAR": "Clarification needed on whether ground-truth rewards are available for validation, and specific environment configurations (e.g., environment API endpoints, segment feature extraction). Also, whether to use human interaction or purely synthetic data for preferences."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
preference_model:
  # Use listwise loss with 'exp' as scoring function for stability and performance
  score_function: exp
  # Number of segments sampled from RLT per training step
  sample_size: 10
  # Number of segments in list for listwise loss
  list_size: 16
  # Regularization coefficient for model training
  regularization: 0.0003
  # Learning rate for the reward neural network
  learning_rate: 0.001
  # Batch size for training reward model
  batch_size: 512
  # Total training epochs for reward model
  epochs: 300
  # Neural network architecture details
  hidden_layers: [128, 128, 128]
  activation: relu
  final_activation: tanh
  # Loss type: 'listwise' (preferred) or 'pairwise'
  loss_type: listwise
  # Scoring function options: 'exp' or 'linear'
  score_function_type: exp

offline_rl:
  # Number of offline training epochs
  epochs: 300
  # Algorithm hyperparameters (e.g., learning rate, discount factor) – defaults to typical settings
  learning_rate: 0.001
  discount_factor: 0.99

environment:
  # Environment choice: 'gym' or 'metaworld'
  type: metaworld
  # Environment-specific parameters (if needed)
  env_name: 'reach-v2'

preference_feedback:
  # Number of preference feedbacks collected per experiment
  feedback_count: 500  # can also set as 200, 1000 for ablation
  # Feedback noise level: 0 (no noise), or higher percentages for robustness tests
  noise_level: 0

experiment:
  # Number of random seeds for robustness
  seeds: 6
  # Evaluation and validation hooks
  validation_interval: 50 epochs
```

---

**Note:** Full configuration is available in `planning_config.yaml`
