# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. To systematically reproduce the work described in the paper, I propose the following step-by-step plan, organized by major methodological components, experiment design, datasets, hyperparameters, and evaluation metrics. Each section includes specific details, referencing the paper’s content, and clearly notes any assumptions or ambiguities to address.

---

## 1. Overall Objective & Core Approach
- Goal: Improve protein sequence optimization by smoothing the fitness landscape using graph-based regularization, then leveraging this smooth landscape with Bayesian sampling (Gibbs With Gradients, GWG) for sequence design.
- Core innovations:
  - Represent protein fitness as a graph signal and smooth it via Tikhonov regularization.
  - Train a neural model to predict smoothed fitness.
  - Use this model as an energy-based model in GWG sampling.
- Validation: On GFP and AAV benchmark datasets with different difficulty levels, measuring in-silico fitness improvement, extrapolation abilities, diversity, and novelty.

---

## 2. Methodological Steps

### 2.1. Preprocessing & Data Preparation
- **Start from initial datasets:**
  - Green Fluorescent Protein (GFP): sequences with reported fluorescence fitness.
  - Adeno-Associated Virus (AAV): sequences with infectivity or other relevant fitness measures.
- **Construct a training set:**
  - Use the sequences with known fitness.
  - The paper filters datasets into "easy," "medium," "hard" tasks based on fitness percentile, mutational gap, and sequence similarity (see Tables 1-2; Figures 3-4).
  - For example:
    - **Hard GFP task:** sequences with >30 mutational distance from top sequences, fitness below 30th percentile.
    - **Hard AAV task:** similar filters applied.
- **Note ambiguities:**
  - Exact filtering thresholds are given (e.g., "Gap = 7" for GFP Hard, "Range < 30%" etc.) — these should be systematically applied.
  - Sequence length constraints or amino acid vocabularies should be consistent with the original datasets (e.g., GFP sequences: 236 residues).
  - **Assumption:** The initial datasets (e.g., from design-bench, GFP benchmark, AAV datasets) are accessible; if not, simulate or use public datasets with similar properties.

### 2.2. Graph Construction & Smoothing of Fitness Landscape
- **Sequence similarity graph:**
  - Use Levenshtein distance between sequences or a Hamming distance if sequences have fixed length.
  - Create an undirected graph where nodes = sequences, edges connect sequences within certain similarity (distance <= k-NN or threshold).
- **Graph Signal:**
  - Fitness values as node attributes.
- **Graph Laplacian:**
  - Compute Graph Laplacian \(L\), normalized or unnormalized, based on adjacency matrix.
- **Smoothing via Tikhonov Regularization:**
  - Solve the convex quadratic problem:
    \[
    \hat{Y} = (I + \gamma L)^{-1} Y
    \]
  - Hyperparameter: \(\gamma\), set via hyperparameter search (see below).
- **Implementation notes:**
  - Efficient inversion via sparse matrix solvers.
  - Hyperparameter \(\gamma\): explore a grid (e.g., \(0.01, 0.1, 1, 10\)) as in the paper.
- **Validation of smoothing:**
  - Check total variation reduction (see paper Eq. (1) and results).
  - Confirm that smoothed labels have lower \(\mathsf{TV}_2\).

### 2.3. Model Training on Smoothed Data
- **Architecture:**
  - Use a neural network model consistent with existing sequence-to-fitness predictors, e.g., an equivariant or convolutional model for sequence data.
  - Input: sequence (one-hot encoding or embedding).
  - Output: continuous fitness score.
- **Training Data:**
  - Sequences with their smoothed fitness labels.
- **Hyperparameters:**
  - Learning rate, optimizer (Adam), epochs, batch size.
  - Regularization techniques (dropout, weight decay).
  - Dropout rates and hyperparameter \(\beta\) for training to be set based on hyperparameter grid or validation.
- **Assumption:**
  - Model complexity similar to prior work (e.g., Flax, Jax, PyTorch).
- **Evaluation:**
  - Performance in terms of in-sample root mean squared error (RMSE) or MAE on validation set.
  - Confirm the model's predictive performance on a held-out set with smoothed labels.

### 2.4. Energy-based Model & Probabilistic Interpretation
- **Energy function:**
  - Use trained predictor \(f_\theta(x)\) as negative energy.
  - The distribution: \(p(x) \propto \exp(f_\theta(x))\).
- **Gradient computation:**
  - For GWG, compute \(\nabla_x f_\theta(x)\) (batched, in sequence space, using differentiable one-hot or embedding layers).

### 2.5. Sequence Sampling with GWG
- **Sampling procedure:**
  - Initialize sequences from the training set.
  - For each sequence, run \(R\) rounds:
    - Generate candidate mutations within 1-Hamming ball (point mutations).
    - Compute proposal distribution \(q(x'|x)\) using the gradient-informed softmax (see eq. (2) & Algorithm 3).
    - Accept/reject via Metropolis-Hastings as in eq. (4).
  - Hierarchical clustering:
    - Cluster the sampled sequences in each round based on Levenshtein distance.
    - Select top-fitness sequence per cluster as the next starting point (\(\tilde{X}_r\)); this controls the explosion of sequences.
- **Hyperparameters:**
  - Number of GWG steps per sequence (\(N_{p}\)), temperature \(\tau\) (grid search as in Fig. 3), number of rounds \(R\), cluster count \(\mathcal{C}\).
- **Clustering:**
  - Use approximate algorithms for Levenshtein if sequences are long.
  - Number of clusters: e.g., 10, 20, 50; chosen based on compute budget and experiments.
- **Number of proposals:**
  - Per sequence per round \(N_p\), e.g., 100, as in the paper.
  
### 2.6. Sequence Proposal & Optimization Loop
- **Initial sequence pool:**
  - Use sequences in training data and their smoothed labels.
- **Iterate GWG sampling:**
  - For a fixed number of rounds (\(R=15\) as in paper).
  - In each round: generate new candidate sequences, cluster, select top sequences.
- **Output:**
  - Top sequences (per fitness or diversity measures) after all rounds.
  - Focus on top-K sequences for evaluation in the next step.

### 2.7. Evaluation & Metrics
- **In-silico evaluation:**
  - Predict fitness of top sequences using the **original predictor** \(f_\theta\) (not just smoothed labels).
  - Measure:
    - Mean fitness (average over top sequences)
    - Max fitness (best in the top set)
    - Fitness jump: \( \text{best sampled} - \text{best in training} \)
- **Diversity metrics:**
  - Variance in sequence space (Levenshtein, Hamming)
  - Number of unique mutations
- **Novelty:**
  - Sequences outside training set or beyond mutational gap filters.
- **Extrapolation:**
  - Use holdout or augmented test sets with sequences not used for training.
- **Comparison to baselines:**
  - The baseline methods: e.g., XP, Calibrated models, etc., as per literature.
  - Evaluate the same metrics: fitness jump, diversity, extrapolation accuracy.

---

## 3. Experimental Settings & Hyperparameter Tuning
- **Hyperparameters to explore:**
  - \(\gamma\) (smoothing regularization weight): e.g., \(\{0.01, 0.1, 1, 10\}\).
  - Model learning rate: e.g., \(\{1e-3, 5e-4, 1e-4\}\).
  - Batch size: e.g., 128 or 256.
  - Sequence embedding dimension: based on amino acid vocab size (\(20\) amino acids).
  - Number of epochs: 50-100, with early stopping based on validation error.
  - GWG temperature \(\tau\): grid over \(\{0.01, 0.1, 1, 10\}\).
  - Number of GWG rounds \(R\): e.g., 15.
  - Number of proposals per sequence \(N_p\): e.g., 100.
  - Clusters \(\mathcal{C}\): e.g., 10–50 depending on compute.
- **Validation:**
  - Use held-out sequences or cross-validation
  - Check the correlation of predicted fitness with actual fitness.
- **Reproducibility:**
  - Ensuring random seeds fixed.
  - Multiple independent runs to assess variance.

---

## 4. Addressing Assumptions & Uncertainties
- **Data Access & Filtration:**
  - Exact datasets are not provided; replicate using public datasets (designbench, GFP benchmark, or synthesize similar data).
- **Graph Construction:**
  - The cutoff for neighbors (e.g., \(k\)-NN) must be tuned; explore a few options.
- **Smoothing hyperparameter \(\gamma\):**
  - Use a validation subset or hyperparameter search grid.
- **Model architecture:**
  - Use existing sequence models known to perform well in protein fitness prediction (e.g., CNNs, transformers) but aligned with prior art.
- **Clustering method:**
  - Use hierarchical clustering with Levenshtein distance or approximate methods for scalability.
- **Metrics computation:**
  - For diversity and novelty, decide on thresholds (e.g., edit distance > 7 mutations).
  - For fitness prediction, rely on models trained on smoothed labels but evaluate with raw fitness measures if available.

---

## 5. Summary of Workflow
1. **Prepare datasets** with filtering for difficulty levels based on percentile, mutational gap, and similarity.
2. **Construct sequence similarity graph** and **smooth fitness labels** via Tikhonov regularization.
3. **Train a neural predictor** to map sequences to smoothed fitness.
4. **Apply GWG sampling:**
   - Generate mutations guided by gradients (using the smoothed model).
   - Perform MH acceptance.
   - Cluster sequences in each round.
   - Select top sequences for next round.
5. **Evaluate generated sequences** in terms of fitness jump, diversity, and extrapolation.
6. **Tune hyperparameters** using validation sets and replicate across difficulty levels for robustness.

---

## 6. Final Notes
- To **simulate the results**, carefully replicate the datasets and thresholds.
- Be ready to adapt hyperparameters based on experimental calibration.
- Keep detailed logs of all steps for reproducibility.
- Optionally, implement the algorithms in a modular way to facilitate hyperparameter sweeps and ablation studies (e.g., effect of smoothing strength).

---

This roadmap should guide implementation of all components, from data processing through smoothing, model training, sampling, and evaluation. Let me know if you'd like a specific module detailed further, such as pseudocode or specific library recommendations.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the method by composing a pipeline of four main modules: (1) Data Preparation with filtering and dataset loading, (2) Graph Construction and Label Smoothing via Tikhonov regularization, (3) Neural Model Training to predict smoothed fitness, and (4) Sequence Sampling using GWG with clustering. Leveraging open-source libraries such as Biopython for dataset handling, NetworkX and SciPy for graph operations, Jax/Flax or PyTorch for neural network modeling, and scikit-learn for clustering, we will keep the architecture modular and manageable. The overall flow will be orchestrated in a main.py script that loads data, applies smoothing, trains the predictor, performs sampling, and evaluates results. Hyperparameters will be configurable via a config dictionary, enabling grid search or manual tuning.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "graph_utils.py",
        "model.py",
        "trainer.py",
        "sampling.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str, filters: dict)\n        +load_filtered_data() -> List[SequenceSample]\n    }\n    class SequenceSample {\n        sequence: str\n        fitness: float\n    }\n    class GraphConstructor {\n        +__init__(sequences: List[str], fitness: List[float], k: int)\n        +build_graph() -> networkx.Graph\n        +compute_laplacian() -> scipy.sparse matrix\n    }\n    class LabelSmoother {\n        +__init__(laplacian: scipy.sparse matrix, gamma: float)\n        +smooth_labels(labels: np.ndarray) -> np.ndarray\n    }\n    class PredictorModel {\n        +__init__(hyperparameters: dict)\n        +train(data: List[SequenceSample], labels: np.ndarray) -> None\n        +predict(sequence: str) -> float\n        +predict_batch(sequences: List[str]) -> np.ndarray\n        +compute_gradients(sequence: str) -> np.ndarray\n    }\n    class Sampler {\n        +__init__(predictor: PredictorModel, sequences: List[str], fitness: np.ndarray, params: dict)\n        +run_sampling() -> List[str]\n        +mutate(sequence: str) -> List[str]\n        +cluster_sequences(sequences: List[str]) -> List[List[str]]\n        +select_top_per_cluster(sequences: List[str], fitnesses: List[float], clusters: List[List[str]]) -> List[str]\n    }\n    class Evaluation {\n        +__init__(predictor: PredictorModel, sequences: List[str], true_fitnesses: List[float])\n        +compute_metrics() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> GraphConstructor\n    Main --> LabelSmoother\n    Main --> PredictorModel\n    Main --> Sampler\n    Main --> Evaluation\n    GraphConstructor --> networkx.Graph\n    GraphConstructor --> scipy.sparse\n    LabelSmoother --> scipy.sparse\n    PredictorModel --> torch.nn.Module (or flax.nn.Module)\n    Sampler --> PredictorModel\n    SequenceSample -- sequence\n    SequenceSample -- fitness\n    classSequenceSample --> sequence\n    classSequenceSample --> fitness\n    main.py invokes methods in sequence handling data loading, smoothing, training, sampling, and evaluation.",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant GC as GraphConstructor\n    participant LS as LabelSmoother\n    participant P as PredictorModel\n    participant S as Sampler\n    participant E as Evaluation\n    Note over M: Initialize configuration\n    M->>DL: load_filtered_data()\n    DL-->>M: sequences, fitnesses\n    M->>GC: build_graph(sequences, fitnesses, k)\n    GC-->>M: graph\n    M->>LS: smooth_labels(np.array(fitnesses), gamma)\n    LS-->>M: smoothed_labels\n    M->>P: train(sequences, smoothed_labels)\n    P-->>M: trained model\n    M->>S: run_sampling() \n    S->>P: compute_gradients(sequence)\n    S->>S: mutate(sequence)\n    S->>S: cluster_sequences(sequences)\n    S->>S: select_top_per_cluster\n    loop for each round\n        M->>S: generate new sequence subset\n        S->>P: predict & gradients\n        S->>S: accept/reject mutations\n        S->>S: cluster & select\n    end\n    M->>E: evaluate(predictor, sampled_sequences, true_fitnesses)\n    E-->>M: metrics\n    Note over M: Results are logged and logged for analysis.",
    "Anything UNCLEAR": "Clarify if there are any dataset-specific processing steps not explicitly detailed, especially related to sequence length constraints and amino acid vocab handling. Also, confirm preferred deep learning framework (PyTorch or Jax/Flax) and clustering method (hierarchical vs approximate) for large datasets."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "scipy==1.7.0",
        "scikit-learn==0.24.2",
        "networkx==2.6.3",
        "matplotlib==3.4.3",
        "jax==0.3.13",
        "flax==0.3.6",
        "torch==1.9.0",
        "tqdm==4.62.3",
        "biopython==1.78"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class to load datasets, apply difficulty filters, and output lists of sequences and fitness scores. Responsible for dataset pre-processing, including filtering sequences by percentile, mutational gap, and other criteria based on input parameters."
        ],
        [
            "graph_utils.py",
            "Defines functions to construct sequence similarity graphs (via Levenshtein or Hamming distances), compute graph Laplacians, and support clustering. Requires NetworkX and SciPy for sparse matrix operations."
        ],
        [
            "label_smoothing.py",
            "Implements LabelSmoother class that takes the graph Laplacian and gamma hyperparameter to smooth labels using the Tikhonov regularization formula. Uses NumPy/Scipy sparse matrices."
        ],
        [
            "model.py",
            "Defines the neural network predictor architecture (e.g., CNN, transformer). Has methods for training, predicting, and computing gradients via autograd frameworks (PyTorch or Jax/Flax). Utilizes configurable hyperparameters."
        ],
        [
            "trainer.py",
            "Encapsulates training loop for predictor model including data batching, loss computation, validation, and checkpointing. Depends on model.py's Model class."
        ],
        [
            "sampling.py",
            "Implements GWG sampling protocol: mutation proposal generation based on gradient info, Metropolis-Hastings acceptance, clustering of sampled sequences, and selection of top sequences per cluster. Uses scikit-learn for clustering."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class that computes in-sample and ex-sample fitness metrics, diversity, fitness jump, and extrapolation ability. Uses the predictor's prediction functions and metrics calculations."
        ],
        [
            "main.py",
            "Orchestrates the overall process: loads datasets via DatasetLoader, constructs and smooths graph labels, trains predictor, runs GWG sampling with clustering, and evaluates. Coordinates the execution flow and hyperparameter sweeps."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "graph_utils.py",
        "label_smoothing.py",
        "model.py",
        "trainer.py",
        "sampling.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ...  # No external API communication; code structured as internal modules for batch processing.",
    "Shared Knowledge": "The dataset_loader and label_smoothing modules share sequence lists and raw labels; predictor.py provides API for training and inference; sampling depends on predictor gradients; clustering uses scikit-learn. Hyperparameters are configured centrally.",
    "Anything UNCLEAR": "Clarify if the specific choice of deep learning framework (PyTorch vs Jax/Flax) is preferred; dataset access details or formats; and whether the code should include hyperparameter search or fixed grids."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset_filters:
  GFP:
    easy: 
      percentile_range: [50, 60]
      mutational_gap: 0
    medium:
      percentile_range: [20, 40]
      mutational_gap: 6
    hard:
      percentile_range: [0, 30]
      mutational_gap: 7
  AAV:
    easy:
      percentile_range: [50, 60]
      mutational_gap: 0
    medium:
      percentile_range: [20, 40]
      mutational_gap: 13
    hard:
      percentile_range: [0, 30]
      mutational_gap: 13

graph_construction:
  knn_neighbors: 20  # Default, can be tuned based on dataset size
  similarity_threshold: null  # Use knn for edge criteria

label_smoothing:
  gamma_values: [0.01, 0.1, 1.0, 10.0]  # Hyperparameter for regularization strength

predictor_model:
  architecture: "cnn"  # Placeholder, can be replaced with "transformer" etc.
  learning_rate: 1e-3  # Default from common practice; tune as needed
  batch_size: 128
  epochs: 50
  dropout_rate: 0.1

training:
  optimizer: "Adam"
  optimizer_params:
    learning_rate: 0.001
  epochs: 50
  batch_size: 128
  seed: 42  # For reproducibility

sampling:
  gwg_rounds: 15
  proposals_per_sequence: 100
  temperature_grid: [0.01, 0.1, 1.0, 10.0]
  clustering_clusters: 20  # Number of clusters for Reduce step
  mutation_batch_size: 100

evaluation:
  fitness_metric: "Max Fitness"  # or "Average Fitness"
  diversity_metric: "Levenshtein"
  extrapolation_method: "Holdout"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
