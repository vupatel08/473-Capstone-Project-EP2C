# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Below is a detailed, structured plan for reproducing SEABO as described in the paper, covering the core methodology, experiment configurations, datasets, hyperparameters, and evaluation protocols.

---

## 1. Understanding and Implementing the Core Methodology

### 1.1 Goal of SEABO
- To assign pseudo-rewards to unlabeled offline data based solely on proximity to expert demonstrations, enabling offline imitation learning without reward annotations or discriminator-based reward models.
- Operates in an **unsupervised manner**, leveraging only the structure of expert trajectories and unlabeled data.

### 1.2 Key Components
- **Expert Demonstration**: a small (single or multiple) trajectory(s) of expert behavior, possibly consisting of only observations or state-action pairs.
- **Unlabeled Offline Dataset**: diverse trajectories with observations and actions but no rewards.
- **Distance Metric**: Euclidean distance (or other metric) to quantify similarity between transitions in the dataset and expert demonstration.
- **Nearest Neighbor Search**: For each transition in unlabeled data, find the closest expert transition via spatial search (KD-tree or Ball-tree).
- **Reward Assignment**:
  - Calculate the distance *d* between the transition and its nearest expert transition.
  - Compute pseudo-reward using the exponential squashing function:
  
    \[
    r = \alpha \exp \left( - \frac{\beta \times d}{|\mathcal{A}|} \right),
    \]
  
  where \(|\mathcal{A}|\) is the action space dimension (for normalization).
- **Hyperparameters**:
  - \(\alpha\): reward scale factor.
  - \(\beta\): decay factor influencing how proximity affects reward.
  - Number of neighbors \(N\) (often 1, but can be varied).
- **Distinction from Previous Methods**:
  - No reward model training, discriminator, or domain-specific assumptions.
  - Works directly with individual transitions.

### 1.3 Implementation Steps
1. **Expert Trajectory Handling**
   - Load a small demonstration (observation-only or state-action).
   - Build an efficient spatial index (KD-tree or Ball-tree) for expert transitions.
2. **Dataset Preparation**
   - Collect unlabeled data: observations/actions without rewards.
   - Normalize features if necessary for stable KD/ball-tree search.
3. **Nearest Neighbor & Reward Labeling**
   - For each unlabeled transition:
     - Query the expert spatial index for nearest neighbor(s).
     - Compute distances \(d\).
     - Assign rewards based on the exponential function.
   - Optionally, tune \(N\) (neighbors) and hyperparameters (\(\alpha, \beta\)) via grid search; default \(\alpha=1\), \(\beta=0.5\).
4. **Dataset Augmentation**
   - Augment each transition with the pseudo-reward.
   - Save as a labeled dataset for policy training.

### 1.4 Integration with Offline RL Algorithms
- Feed the labeled dataset into any offline RL algorithm (e.g., IQL, TD3+BC).
- Use the pseudo-rewards for policy optimization.

---

## 2. Experiment Design & Dataset Acquisition

### 2.1 Datasets
- **Expert Demonstrations**:
  - Small (single trajectory) or multiple trajectories.
  - Consisting either of raw observations or state-action pairs.
  - Sources: can be synthetic (e.g., expert policy driven) or human demonstrations.
- **Unlabeled Offline Dataset**:
  - Large collection of trajectories with no reward annotations.
  - Should include diverse behaviors for robustness.

### 2.2 Tasks & Domains
- **MuJoCo (e.g., HalfCheetah, Hopper, Walker2d)**:
  - From D4RL or similar offline RL benchmarks.
  - Use provided datasets (e.g., medium, medium-replay, expert versions).
- **AntMaze / Adroit / Kitchen environments**:
  - For evaluation of long-horizon / high-dimensional tasks.
  - Optional: area of investigation for transferability.

### 2.3 Offline RL Algorithms
- **Base algorithms**:
  - IQL (recommended for initial implementation).
  - TD3-BC or others if desired.
- **Training procedure**:
  - Input the reward-labeled datasets intoes offline RL framework.
  - Follow standard training protocols:
    - Hyperparameters as in the paper or default settings.
    - Transitions sampled uniformly or with prioritized sampling.

### 2.4 Evaluation Metrics
- **Return / Normalized Score**:
  - Compute the total return on evaluation episodes.
  - Normalize based on the known expert and random policies:
  
    \[
    \mathrm{score} = \frac{J - J_r}{J_e - J_r}
    \]
    
  where:
  - \(J_r\): return of a random policy.
  - \(J_e\): return of the expert policy.
- **Trajectory-based Evaluation**:
  - Run the learned policy in the environment.
  - Average over multiple evaluation seeds/episodes.
- **Comparison Baselines**:
  - Offline RL with true reward labels.
  - Reward learning methods (e.g., DEMO-DICE, Value-DICE).
  - Other offline IL approaches (e.g., PWIL, OTR).

### 2.5 Hyperparameter Settings (from paper's Appendix)
- Rewards: exponential with \(\alpha=1\), \(\beta=0.5\).
- Number of neighbors \(N=1\), but evaluate sensitivity (see hyperparameter study).
- Action space normalization: divide distances by \(|\mathcal{A}|\).

---

## 3. Implementation Details & Practical Considerations

### 3.1 Data Preprocessing
- Normalize states and actions (standard practice).
- Use the same normalization in training and inference.
- If only observations are available, build the KD-tree on observations.

### 3.2 Search & Computation
- Use scikit-learn's `KDTree` or `BallTree` for nearest neighbor searches.
- Batch query for efficiency.
- Handle scattered/truncated trajectories:
  - Transitions are independent; process each individually.
  - No trajectory-level dependencies needed.

### 3.3 Hyperparameter Tuning
- Conduct a grid search over \(\alpha, \beta, N\)?
- Use validation performance or directly compare in final testing.

### 3.4 Algorithmic Procedure for the Code
1. Load expert demo data.
2. Build spatial index on expert transitions.
3. For each unlabeled transition:
   - Query nearest neighbors.
   - Compute distance \(d\).
   - Compute reward \(r\).
4. Attach rewards to transitions.
5. Save labeled dataset.
6. Train offline RL algorithm with this dataset.
7. Evaluate trained policy.

---

## 4. Summary of Critical Details & Potential Ambiguities
- **Expert demonstration format**: observations vs (states, actions). Need to align with the method's flexibility.
- **Distance metric**: Euclidean, normalized features recommended.
- **Hyperparameters**:
  - \(\alpha=1\),
  - \(\beta=0.5\),
  - number of neighbors \(N=1\).
  - Action space normalization: divide distances by \(|\mathcal{A}|\).
- **Hyperparameter sensitivity**:
  - Append hyperparameter studies to verify robustness.
  - Tuning \(\alpha, \beta\), neighbor count could improve performance.
- **Computational resources**:
  - Can handle large datasets efficiently with KD-trees.
  - For high-dimensional data, consider approximate nearest neighbor search.

---

## 5. Final Remarks
- This plan aligns with the paper's methodology and experimental setup.
- Implementation can proceed iteratively:
  - Validate the reward labeling process on a small dataset.
  - Verify offline RL training with pseudo-rewards.
  - Compare against baselines and tune hyperparameters.
- Keep detailed logs and record hyperparameters for reproducibility.

---

Would you like me to proceed with preparing a precise code skeleton following this plan?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular system where the core steps include loading expert demonstrations and unlabeled datasets, constructing spatial indices for nearest neighbor search using scikit-learn, computing pseudo-rewards with hyperparameters, and then training an offline RL policy (e.g., IQL) with the labeled dataset. The process involves defining classes for data loading, reward labeling, model, training, and evaluation, integrated through a main entry point that orchestrates dataset processing, reward assignment, policy training, and evaluation.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "reward_labeler.py",
        "model.py",
        "trainer.py",
        "evaluation.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run_experiment() -> None
    }
    class DatasetLoader {
        +__init__(dataset_path: str, expert_path: str, only_observations: bool = False)
        +load_data() -> Tuple[Any, Any]
            # Returns (unlabeled_dataset, expert_demonstration)
    }
    class RewardLabeler {
        +__init__(expert_samples: np.ndarray, alpha: float, beta: float, neighbor_count: int)
        +assign_rewards(unlabeled_data: np.ndarray) -> np.ndarray
    }
    class OfflineModel {
        +__init__(model_type: str, params: dict)
        +train(dataset: dict) -> None
        +predict(obs: np.ndarray) -> np.ndarray
    }
    class Trainer {
        +__init__(model: OfflineModel, dataset: dict, val_dataset: Optional[dict] = None)
        +train() -> None
    }
    class Evaluation {
        +__init__(model: OfflineModel)
        +evaluate(env: str, episodes: int = 10) -> dict
    }
    Main --> DatasetLoader
    Main --> RewardLabeler
    Main --> Trainer
    Main --> Evaluation
    RewardLabeler --> np.ndarray: uses expert samples
    Trainer --> OfflineModel: trains models
    Evaluation --> OfflineModel: runs policies
    Main --> DatasetLoader
    Main --> RewardLabeler
    Main --> Trainer
    Main --> Evaluation
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant RL as RewardLabeler
    participant Mdl as OfflineModel
    participant Tr as Trainer
    participant E as Evaluation
    M->>DL: load_data()
    DL-->>M: expert_data, unlabeled_data
    M->>RL: initialize expert_samples, hyperparams
    RL-->>M: reward_array
    M->>Mdl: initialize(model_type, params)
    M->>Tr: train(model, dataset_with_rewards)
    Tr->>Mdl: train on dataset with pseudo_rewards
    M->>E: evaluate(env_name, episodes=10)
    E->>Mdl: predict policies
    E-->>M: performance metrics
    Main ends
",
    "Anything UNCLEAR": "Clarify whether the expert demonstration consists of only observations or (state, action) pairs, and if the dataset format requires specific normalization or preprocessing steps for consistency."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "scikit-learn==0.24.2",
        "torch==1.9.0",
        "yaml==5.4.1"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Class DatasetLoader: Responsible for loading expert demonstration data and unlabeled datasets, providing interfaces load_data() -> (expert_data, unlabeled_data). It depends on data formats, handles normalization if needed, and prepares datasets for subsequent processing."
        ],
        [
            "reward_labeler.py",
            "Class RewardLabeler: Takes expert samples and hyperparameters (alpha, beta, neighbor count). Implements assign_rewards() -> np.ndarray, which performs KD-tree construction on expert data, queries nearest neighbors, computes distances, and outputs reward labels. Uses scikit-learn's KDTree."
        ],
        [
            "model.py",
            "Class OfflineModel: Defines the offline RL policy network (e.g., IQL). Handles initialization, training, and prediction. Depends on the dataset with rewards, training routines, and environment-specific architecture."
        ],
        [
            "trainer.py",
            "Class Trainer: Manages training the offline RL model with labeled data. Implements train() -> None, controlling the training loop, sampling mini-batches, and optimizer updates. Depends on OfflineModel and labeled dataset."
        ],
        [
            "evaluation.py",
            "Class Evaluation: Loads trained policy, runs in environment for specified episodes, and calculates metrics like average return. Implements evaluate(env_name, episodes) -> dict. Depends on environment setup and trained model."
        ],
        [
            "main.py",
            "Main entry point: Orchestrates loading datasets, calling RewardLabeler to generate rewards, training the offline RL policy via Trainer, and evaluating performance via Evaluation. Coordinates dependencies among other modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",      # Load expert demonstration and unlabeled data
        "reward_labeler.py",      # Build KD-tree, assign pseudo-rewards
        "model.py",               # Define offline RL policy network
        "trainer.py",             # Train policy with the reward-labeled dataset
        "evaluation.py",          # Evaluate policy performance in environment
        "main.py"                 # Orchestrate full experiment pipeline
    ],
    "Full API spec": "openapi: 3.0.0\n\n# No front-end API needed, all modules communicate via class interfaces.\n# DatasetLoader: load_data()\n# RewardLabeler: assign_rewards(unlabeled_data)\n# OfflineModel: train(dataset), predict(obs)\n# Trainer: train(), save_model()\n# Evaluation: evaluate(env_name, episodes)\n# main.py: coordinate calls",
    "Shared Knowledge": "Common utility functions include data normalization, feature scaling, and environment setup scripts. Dataset formats require consistent observation-action pairs in numpy array format. Hyperparameters are loaded from a config dictionary.",
    "Anything UNCLEAR": "Clarify whether expert demonstrations consist of only observations or (states, actions). Also, confirm if the environment setup for evaluation should be simulated or real, and whether any specific neural network architectures are recommended."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  expert_demo_path: "path/to/expert_demonstration.npy"  # Path to the expert demonstration data
  unlabeled_data_path: "path/to/unlabeled_dataset.npy"  # Path to the unlabeled offline dataset
  only_observations: false  # Set true if demonstration contains only observations
reward_labeling:
  alpha: 1.0          # Reward scale factor
  beta: 0.5           # Exponential decay factor
  neighbors: 1        # Number of nearest neighbors to consider
model:
  type: "iql"         # Offline RL algorithm type
  params:
    hidden_layers: [256, 256]  # Neural network architecture
    learning_rate: 3e-4       # Learning rate for optimizer
    batch_size: 256           # Batch size
    total_timesteps: 1e6      # Total training steps
training:
  epochs: 50                # Number of training epochs
  optimizer: "Adam"         # Optimizer type
  learning_rate: 3e-4       # Learning rate (consistent with model params)
  weight_decay: 0.0001      # Weight decay coefficient
evaluation:
  environment: "environment_name"  # Environment for evaluation
  eval_episodes: 10                # Number of episodes per evaluation
  eval_interval: 10000             # Steps between evaluation
hyperparameters:
  alpha: 1.0      # Reward scaling hyperparameter
  beta: 0.5       # Reward decay hyperparameter
  neighbor_count: 1  # Number of neighbors for KD-tree search
```

---

**Note:** Full configuration is available in `planning_config.yaml`
