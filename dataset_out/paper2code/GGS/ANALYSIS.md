# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

# Logic Analysis for dataset_loader.py

## Purpose
Implement the `DatasetLoader` class to load, filter, and preprocess protein sequence datasets for GFP and AAV. The class should produce sequences and associated fitness scores, filtered and processed according to criteria specified in configuration, especially for defining training subsets at different difficulty levels (easy, medium, hard).

---

## Core Responsibilities
1. **Dataset Loading**
   - Load raw datasets containing amino acid sequences with associated fitness measurements.
   - Support multiple datasets: GFP, AAV, or similar.
   - Handle data formats: typically, list of sequences with fitness scores, possibly in CSV, JSON, or custom formats.
   - Use Biopython or standard file I/O to parse data.

2. **Filtering Criteria (Per config.yaml)**
   - *Percentile Range Filter for Fitness*: Only include samples where  fitness measurements fall within specified percentile ranges.
   - *Mutational Gap Filter*: Ensure the sequences are a minimum (or at most) a certain mutational distance (e.g., Levenshtein or Hamming) from known top sequences, typically to create challenging tasks.
   - *Sequence Length Constraints (if any)*: Sequences should match expected length (e.g., GFP ~ 236 residues).
   - *Sequence Validity*: Sequences must only contain valid amino acids (20 standard residues).

3. **Filtering Implementation Detail**
   - Compute thresholds for fitness percentiles dynamically based on data.
   - For each sequence:
     - Determine if it falls within the specified percentile range.
     - Calculate mutational distance (Levenshtein or Hamming) to top sequences.
     - Filter based on the mutational gap criterion.
   - For the 'hard' difficulty, sequences are selected to be sufficiently mutation distant from top sequences, and fitness below a percentile.

4. **Difficulty-specific Dataset Generation**
   - Use the configuration’s percentile and mutational gap to produce datasets per difficulty:
     - **easy**: high fitness, possibly within top percentile, mutational gap ~0.
     - **medium**: moderate fitness percentile, larger mutational gap.
     - **hard**: low fitness percentile, larger gap, sequences far from top.

5. **Output**
   - Return two lists:
     - `sequences`: list of sequence strings (or one-hot matrices if needed later).
     - `fitnesses`: corresponding fitness scores.

6. **Reproducibility & Flexibility**
   - Parameterize filtering thresholds (percentile ranges, mutational gaps) via input arguments.
   - Support optional filtering or inclusion/exclusion criteria.

---

## Implementation Details

### Inputs
- `dataset_path`: path to raw dataset files.
- `filters`: dict containing filtering parameters:
  - `percentile_range`: list of two ints/e.g., `[0, 30]`.
  - `mutational_gap`: int, e.g., 7.
- `difficulty_level`: string, `'easy'`, `'medium'`, `'hard'`; determines the filtering criteria based on config.yaml.

### Internal Data Structures
- Raw data: list of tuples `(sequence, fitness)`.
- Filtered data: filtered list of `(sequence, fitness)`.

### Key Steps
1. **Load raw data**
   - Parse from file (CSV/JSON/structured dataset).
   - Extract sequences and fitness scores.

2. **Compute percentile thresholds**
   - Use numpy to determine percentile cutoffs for fitness scores.
   - For example, for percentile_range `[0,30]`, compute the 0th and 30th percentile values over all fitness scores.

3. **Filter by fitness percentile**
   - Include only samples with fitness within the given percentile bounds.

4. **Compute mutational distances**
   - Use Levenshtein distance (recommended for variable-length sequences or Hamming if fixed length) between each sequence and the known top sequences.
   - To define "top sequences":
     - Identify sequences in the dataset with highest fitness scores (e.g., top 1-5 sequences).
     - Calculate the mutational gap as the minimal distance between each candidate sequence and these top sequences.

5. **Apply mutational gap filter**
   - Filter sequences to only include those:
     - At least `mutational_gap` mutations away from top sequences (for difficult cases).
     - Or, exactly matching criteria based on the difficulty.

6. **Produce output lists**
   - Lists of sequences and fitness scores that pass filters.

---

## Additional Considerations
- **Handling Noisy Data:** Since the datasets may have multiple measurements per sequence, consider averaging or selecting the measurement with the highest fitness, or use raw data as is.
- **Efficiency**
  - Distance calculations could be costly for large datasets.
  - Optimize by caching distances or limiting computations to candidate sets.
- **Reproducibility**
  - Fix random seeds if sampling is involved.
  - Ensure consistent percentile thresholds across runs.

---

## Summary
The `DatasetLoader` class encapsulates dataset ingestion, filtering by percentile and mutational distance, and output formatting. It must dynamically compute percentile thresholds, perform sequence comparisons efficiently, and generate filtered datasets suited for evaluating protein optimization algorithms at different difficulty levels, as specified by configuration.

---

Would you like me to proceed to the next component, e.g., graph_utils.py, or prepare pseudocode for DatasetLoader?

## evaluation.py

**Evaluation.py - Logic Analysis**

---

### Overview:

The primary goal of `evaluation.py` is to define an `Evaluation` class capable of assessing the quality of generated or sampled protein sequences according to various metrics. These include in-sample fitness, extrapolation performance, diversity within the generated sequences, and novelty relative to the training dataset. The metrics should align with the paper’s evaluation setup, utilizing a predictor (fitness model) as a proxy for biological fitness and taking into account the real ground-truth (or in-silico) fitness when available.

---

### Core Inputs:

- **Predictor Model (g_φ):**
  - Provides sequence-to-fitness prediction.
  - Can predict for both training, test, or generated sequences.
  - Also capable of computing gradients if necessary for certain sampling behaviors or analyses (not directly needed here but important context).

- **Sequences for Evaluation:**
  - **Sampled sequences (`X_samples`)**: Sequences generated during a sampling or optimization run.
  - **Training sequences (`X_train`)**: Original sequences used to train the predictor.
  - **Ground-truth fitnesses (`Y_true`)**: If available, the actual or experimental fitness values (e.g., from datasets or experimental validation).
  - **Predicted fitnesses (`Y_pred`)**: Predicted via the predictor for each sequence as a proxy metric.

---

### Evaluation Metrics & Logic:

**A. In-sample or In-silico Fitness:**

- **Predicted fitness:**
  - Use `g_φ` to evaluate sequences.
  - For a batch of sequences, call a batch prediction function: `predict_batch`.
- **Real fitness:**
  - If dataset provides `Y_true`, compare predicted fitnesses to true fitness values (linear scaling or normalization may be used).
- **Selected metric:**
  - Max fitness: The highest predicted fitness among the sampled set.
  - Median fitness: The median of the predicted fitnesses.
  - Alternatively, the mean or percentile-based statistics could be added depending on desired analysis, but given the paper, focus on max and median.

**B. Extrapolation & Holdout Evaluation:**

- **Compare predicted fitnesses versus true fitnesses:**
  - For sequences not seen during training, or with holdout true fitness values.
  - Use MAE (Mean Absolute Error) between predicted and true fitness to measure how well the predictor generalizes to unseen sequences.
  - Compute for both training set (in-sample) and holdout/unseen sequences (out-of-sample).
- **Assess the model's extrapolation capability:**
  - Lower MAE indicates better generalization.
  - Larger MAE suggests the predictor struggles with unseen digressions, which could impact sampling efficacy.

**C. Diversity Metric:**

- **Sequence Diversity:**
  - Use Levenshtein distance or Hamming distance between pairs of sequences.
  - Compute pairwise distances within the sampled set.
  - Summary statistic often used: median of all pairwise distances or mean.
  - For computational efficiency, possibly sample a subset of pairs.
- **Implementation:**
  - Use scikit-learn or custom functions to compute distances.
  - Return the median of the pairwise distances as the diversity measure.

**D. Novelty Measurement:**

- **Definition:**
  - How different are the sampled sequences from the original training set?
  - Calculated as median (or mean) of the minimal distances of each sampled sequence to any sequence in the training set.
  - Distance: Typically Levenshtein or Hamming, same as above.
- **Implementation:**
  - For each sampled sequence, compute the distance to all training sequences.
  - Take the minimal distance.
  - Median over all sampled sequences gives the overall novelty measure.

**E. Fitness Jump / Improvement:**

- **Calculate the fitness jump:**
  - Find the maximum predicted fitness among sampled sequences.
  - Compare to the maximum fitness in the original training data.
  - `fitness_jump = max_sampled_fitness - max_training_fitness`
- **Interpretation:**
  - Indicates how much the sampling/optimization method has crossed the fitness landscape from initial points; an important result discussed in the paper.

---

### Additional Considerations:

- **Data Handling:**
  - Sequences might be stored as strings, integers, or one-hot encodings.
  - Predictor functions:
    - Expect `predict_batch(sequences)` to return a numpy array of fitness predictions.
    - If gradients are required, expect `compute_gradients(sequence)`.

- **Ground-truth data:**
  - If available, used for extrapolation MAE, fitness comparison; else, rely on predictor predictions alone.
  - In the absence of experimental data, the model’s own predictions are used as proxies.

- **Hyperparameters / Config:**
  - Metric choices should be flexible based on configuration but default to the most relevant (max fitness, median diversity).

- **Optional enhancements:**
  - Additional metrics like entropy of sequence distribution, or clustering quality, can be added for extended analysis but are out of scope here.

---

### Pseudocode Sketch:

```python
class Evaluation:
    def __init__(self, predictor, sequences, true_fitnesses=None, train_sequences=None, train_fitnesses=None, config=None):
        self.predictor = predictor
        self.sequences = sequences
        self.true_fitnesses = true_fitnesses  # Optional
        self.train_sequences = train_sequences  # For novelty/extrapolation
        self.train_fitnesses = train_fitnesses
        self.config = config

    def evaluate_fitness(self):
        # Predict fitness for sample sequences
        predicted_fitnesses = self.predictor.predict_batch(self.sequences)

        # Maximum and median predictions
        max_fitness = np.max(predicted_fitnesses)
        median_fitness = np.median(predicted_fitnesses)

        self.results['max_fitness'] = max_fitness
        self.results['median_fitness'] = median_fitness

        # If true fitness available, compare
        if self.true_fitnesses is not None:
            # Calculate MAE, etc.
            predicted_true = self.predictor.predict_batch(self.sequences)
            mae = np.mean(np.abs(predicted_true - self.true_fitnesses))
            self.results['mae'] = mae

    def evaluate_extrapolation(self):
        if self.true_fitnesses is None:
            return
        # Predict for train and holdout
        train_preds = self.predictor.predict_batch(self.train_sequences)
        holdout_preds = self.predictor.predict_batch(self.sequences)  # or holdout if available

        # Compute MAE
        train_mae = np.mean(np.abs(train_preds - self.train_fitnesses))
        holdout_mae = np.mean(np.abs(holdout_preds - self.true_fitnesses))

        self.results['train_mae'] = train_mae
        self.results['holdout_mae'] = holdout_mae

    def compute_diversity(self, sequences=None):
        if sequences is None:
            sequences = self.sequences
        # Compute pairwise distances
        distances = []
        for i in range(len(sequences)):
            for j in range(i+1, len(sequences)):
                dist = compute_distance(sequences[i], sequences[j])  # Levenshtein or Hamming
                distances.append(dist)
        median_distance = np.median(distances)
        self.results['diversity'] = median_distance

    def compute_novelty(self):
        # Distance of sampled sequences to training set
        if self.train_sequences is None:
            return
        min_distances = []
        for seq in self.sequences:
            dist_list = [compute_distance(seq, train_seq) for train_seq in self.train_sequences]
            min_distances.append(np.min(dist_list))
        median_novelty = np.median(min_distances)
        self.results['novelty'] = median_novelty

    def summarize(self):
        # Call all evaluation metrics
        self.evaluate_fitness()
        self.compute_diversity()
        self.compute_novelty()
        # Save or return results
        return self.results
```


---

### Summary:

`evaluation.py` will contain:

- **Class `Evaluation`**:
  - Initialization with predictor, sequences, optional ground-truth fitnesses, training sequences.
  - Methods to compute:
    - Predicted fitness statistics (max, median).
    - Extrapolation MAE and generalization performance.
    - Diversity within the sampled set.
    - Novelty relative to training data.
  - Final `summarize()` method aggregating the metrics as a dict, ready for reporting/plotting.

This structured approach provides clarity, fidelity to the paper’s metrics, and extensibility for additional measures if needed.

---

Anything unclear? If not, I can proceed to encode this logic into actual code templates.

## graph_utils.py

# Logic Analysis for graph_utils.py

This module is responsible for implementing core functions that construct the sequence similarity graph, compute its Laplacian, and support clustering of protein sequences based on similarity measures. These functions underpin the graph-based smoothing technique described in the methodology, which regularizes sequence fitness values over the similarity graph.

---

## 1. Inputs and Outputs

### Inputs:
- list of sequences (`sequences: List[str]`)
- hyperparameters: 
  - `knn_neighbors: int` (from config: default 20)
  - `similarity_threshold: float or None` (from config: default None, indicating knn-based graph)
- For clustering: sequences to cluster; similarity measures to compare sequences

### Outputs:
- Adjacency matrices (preferably sparse) representing the graph: 
  - For construction: NetworkX Graph object
  - For Laplacian: scipy sparse matrix (e.g., CSR)
- Clustering labels or groupings based on similarity (for reducing sequences)
  
---

## 2. Core Functions & Their Logic

### 2.1. create_sequence_graph(sequences: List[str], knn_neighbors: int, similarity_threshold: Optional[float]) -> nx.Graph

- **Purpose:** Construct an undirected similarity graph where nodes are sequences and edges represent similarity.
- **Process:**
  - For each sequence, compute pairwise similarity/distance to others.
  - Use Levenshtein distance or Hamming distance based on sequence properties:
    - **Levenshtein distance:** 
      - Suitable if sequences have insertions/deletions or vary in length.
      - Use in cases with varied sequence lengths.
      - Use an efficient implementation (e.g., `python-Levenshtein` or `pyfastx`).
    - **Hamming distance:**
      - Suitable for fixed-length sequences.
  - Construct edges:
    - If `similarity_threshold` is set: connect pairs with distance below threshold.
    - Else, use K-nearest neighbors: connect each sequence to its `knn_neighbors` closest sequences.
  - Use `networkx.Graph()` for node/edge management.
  - Store node attributes if needed (e.g., sequence string).

- **Special considerations:**
  - For large datasets, pairwise computation can be expensive; consider approximate or efficient methods.
  - Use sparse adjacency matrix representations for subsequent computations.

---

### 2.2. compute_graph_laplacian(graph: nx.Graph, normalized: bool=True) -> scipy.sparse.csr_matrix

- **Purpose:** Generate the graph Laplacian matrix (preferably sparse) for regularization.
- **Process:**
  - Extract adjacency matrix:
    - Use `nx.adjacency_matrix(graph)` which returns a sparse matrix.
  - Compute degrees:
    - `degree = adjacency.sum(axis=1)`
  - Construct Laplacian:
    - **Unnormalized:** \(L = D - A\)
    - **Normalized:** \(L_{sym} = I - D^{-1/2} A D^{-1/2}\)
  - Return sparse matrix (`scipy.sparse.csr_matrix`) to enable efficient operations.

- **Notes:**
  - The paper employs the unnormalized Laplacian in the equation \(\hat{Y} = (I + \gamma L)^{-1} Y\).

---

### 2.3. perform_label_smoothing(labels: np.ndarray, laplacian: scipy.sparse.csr_matrix, gamma: float) -> np.ndarray

- **Purpose:** Smooth raw fitness labels over the graph using Tikhonov regularization.
- **Process:**
  - Solve \(\hat{Y} = (I + \gamma L)^{-1} Y\):
    - Use `scipy.sparse.linalg.spsolve` or `factorized` approach for efficient solution.
  - Input: raw labels (vector `Y`), Laplacian `L`, hyperparameter `gamma`.
  - Output: smoothed labels (`np.ndarray`) for the sequences.
  
- **Additional:**
  - This operation regularizes fitness labels, reducing noise and local variability.
  - It effectively propagates label information across the graph.

---

### 2.4. cluster_sequences(sequences: List[str], num_clusters: int) -> List[List[str]]

- **Purpose:** Cluster similar sequences based on Levenshtein distance to reduce computational complexity of GWG sampling.
- **Process:**
  - For large datasets, avoid exhaustive pairwise clustering. Use hierarchical clustering.
  - Use scikit-learn's hierarchical clustering:
    - Input: sequences, compute a condensed distance matrix (Levenshtein).
    - Use linkage method (`average`, `ward`, or custom) suitable for sequence distances.
  - Assign cluster labels:
    - Use `scipy.cluster.hierarchy.fcluster` with a threshold that results in approximately `num_clusters` clusters or with desired number.
  - Output:
    - List of clusters (each a list of sequences).
  
- **Optimization note:**
  - For efficiency, approximate clustering methods may be used if dataset is large.
  - Ensure sequences within each cluster are similar.

---

### 2.5. select_top_per_cluster(sequences: List[str], fitnesses: np.ndarray, clusters: List[List[str]]) -> List[str]

- **Purpose:** From each cluster, select the sequence with the highest predicted fitness.
- **Process:**
  - For each cluster:
    - Map sequences to their predicted fitness scores.
    - Select the sequence with maximum fitness.
- **Output:**
  - List of top sequences (one per cluster) for the next GWG iteration.

---

## 3. Implementation Details & Assumptions

- **Distance Measures:**
  - Use Levenshtein distance for generality, especially if sequences vary in length.
  - For fixed-length sequences, Hamming distance is computationally cheaper.
- **Graph Construction:**
  - For datasets with thousands of sequences, leverage parallelism or approximate nearest neighbor algorithms.
  - Use NetworkX's functions combined with scipy for matrix representation.
- **Clustering:**
  - Complete linkage clustering with available distance matrices.
  - Since Levenshtein is slow, consider approximate methods or precompute distances.
- **Data Types:**
  - Sequences as strings.
  - Adjacency matrices as sparse scipy matrices.
  - Clusters as lists of sequence strings.
- **Hyperparameters:**
  - `knn_neighbors` set to 20 as default, adjustable.
  - `similarity_threshold` as None, indicating reliance on knn.
  
## 4. Edge Cases & Error Handling
- Empty sequence list: return empty graph and empty clusters.
- Sequences with invalid characters or inconsistent length: handle exceptions or filter out.
- Levenshtein computation failures: implement efficient fallback or approximate measures.
- Large datasets: warn about computational cost and advise approximate neighbor methods if needed.

---

## 5. Summary
This module will encapsulate the core graph operations integral to the regularization and sampling processes:

- **Graph construction** via nearest neighbor or threshold-based similarity.
- **Laplacian computation** for spectral smoothing.
- **Label smoothing** using the closed-form solution.
- **Clustering sequences** by similarity to facilitate efficient and diverse sampling.

By adhering to the specified inputs, outputs, and hyperparameter configurations, the functions will be modular, interpretable, and aligned with the methodology detailed in the paper, ensuring fidelity for the subsequent steps in model training and sequence sampling.

---

Would you like me to proceed with drafting the actual code implementation based on this analysis?

## label_smoothing.py

# Logic Analysis for label_smoothing.py

This module implements the `LabelSmoother` class responsible for smoothing the fitness labels associated with a graph of protein sequences. The core functionality involves applying Tikhonov regularization (graph-based smoothing) to obtain a set of smoothed fitness labels that are less noisy and more amenable for subsequent model training and optimization. The following analysis details the step-by-step logic and implementation considerations, referencing the paper and the provided configuration.

---

## 1. Inputs to LabelSmoother
- **Graph Laplacian, \(L\)**:
  - A sparse matrix representing the graph's Laplacian, constructed from sequence similarity (e.g., k-NN based on Levenshtein or Hamming distance).
  - Computed externally, likely in `graph_utils.py`, based on the dataset.
- **Hyperparameter \(\gamma\)**:
  - A scalar regularization weight controlling the degree of smoothing (see eq. (1) in the paper).
  - Selected from a grid: [0.01, 0.1, 1.0, 10.0].

- **Original labels \(Y\)**:
  - 1D NumPy array or SciPy sparse vector of size equal to the number of nodes, with initial observed fitness values for sequences.

---

## 2. Object Initialization
- The `LabelSmoother` class should be initialized with:
  - The graph Laplacian matrix \(L\).
  - The gamma hyperparameter.
- **Purpose**:
  - Store these inputs for use in smoothing method(s).
  - Possibly allow changing gamma hyperparameters if needed (for hyperparameter sweeps).

## 3. Main Functionality: Smoothing Labels
- **Method signature**:
  ```python
  def smooth_labels(self, labels: np.ndarray) -> np.ndarray:
  ```
- **Input**:
  - `labels`: array of size \(|V|\), with observed fitness values for nodes.
- **Output**:
  - `smoothed_labels`: array of size \(|V|\), with regularized, smooth fitness labels.

---

## 4. Mathematical Operation
- The core computation follows the Tikhonov regularization solution:
  
  \[
  \hat{Y} = (I + \gamma L)^{-1} Y
  \]
  
  where:
  - \(I\): Identity matrix of size \(|V| \times |V|\).
  - \(L\): Graph Laplacian (sparse matrix).
  - \(Y\): Original fitness label vector.
  - \(\hat{Y}\): Smoothed labels.

- **Implementation considerations**:

  - Use `scipy.sparse.linalg.splu` or `scipy.sparse.linalg.factorized` to efficiently solve the linear system without explicitly computing the inverse.
  
  - For larger graphs, direct matrix inversion is computationally expensive and unstable; prefer sparse factorizations.
  
  - Since \(L\) is symmetric positive semi-definite, \(I + \gamma L\) is positive definite, making it suitable for Cholesky or LU decomposition.

---

## 5. Implementation Steps
- **Step 1:** Construct the matrix:

  \[
  A = I + \gamma L
  \]
  
  - Ensure the matrix is in sparse format (e.g., `scipy.sparse.csr_matrix`).

- **Step 2:** Solve the linear system:

  \[
  A \hat{Y} = Y
  \]
  
- **Step 3:** Return the resulting vector \(\hat{Y}\).

- **Edge Cases:**
  - If some labels are missing (NaNs), decide whether to mask or set to default.
  - \(\gamma=0\) should return original labels.
  
- **Numerical Stability & Efficiency:**
  - Use sparse solvers.
  - Precompute factorization if multiple smoothing calls with different \(\gamma\) values are planned.

---

## 6. Hyperparameter Handling
- **Gamma grid search:**
  - During experiments, iterate over the `gamma_values` list.
  - Initialize `LabelSmoother` with each \(\gamma\), or provide a method to dynamically set a different \(\gamma\).
- **Possible extension:**
  - Cross-validate smoothing strength based on downstream model validation or in-silico performance metrics.

---

## 7. Class API and Usage
- **Constructor**:
  ```python
  class LabelSmoother:
      def __init__(self, laplacian: scipy.sparse matrix, gamma: float):
          ...
  ```
- **Method**:
  ```python
  def smooth_labels(self, labels: np.ndarray) -> np.ndarray:
      ...
  ```
  
- **Workflow**:
  - Instantiate with graph Laplacian and desired \(\gamma\).
  - Call `smooth_labels(Y)` to get smoothed labels.
  - Use smoothed labels to retrain the predictor model.

---

## 8. Summary
The core logic integrates:
- Accepting the graph Laplacian and gamma as parameters.
- Solving the regularized linear system efficiently.
- Returning smoothed labels that are less noisy and exhibit smoother variation over the graph.

This process aligns with the paper's description of Tikhonov regularization for smoothing the fitness landscape, enabling improved performance in subsequent protein sequence optimization tasks.

---

**End of the detailed logic analysis.**

## main.py

# Logic Analysis for `main.py` — Orchestrating the Protein Optimization Pipeline

This document provides a detailed, step-by-step logical blueprint for implementing `main.py`, aligning with the outlined plan, file interfaces, and hyperparameters defined in `config.yaml`. The goal of `main.py` is to coordinate all modules to perform data loading, graph construction, label smoothing, predictor training, sequence sampling, and evaluation in a reproducible, flexible, and hyperparameter-driven manner.

---

# Overall Execution Structure

1. **Initialization**:
   - Load configuration parameters from a YAML file or dictionary.
   - Set random seed for reproducibility across all modules, including dataset loader, graph construction, neural training, and sampling.

2. **Loop over task categories and difficulty levels** (if hyperparameter search or multiple experiments are desired):
   - For each dataset (GFP, AAV).
   - For each difficulty setting (`easy`, `medium`, `hard`) per dataset, based on filters in `dataset_filters`.

3. **Sequential Modules**:
   - **Data Loading & Filtering** (via `DatasetLoader`):
     - Load raw sequence data and associated fitness metrics.
     - Apply filtering: percentile range, mutational gap, and other criteria, consistent with hyperparameters from `config.yaml`.
     - Output: list of sequences (`X`) and fitness scores (`Y`).

   - **Graph Construction** (via `GraphUtils`):
     - Build a k-NN or similarity graph of sequences.
       - Use Levenshtein or Hamming distance.
       - Use `knn_neighbors=20` (or tune as needed).
     - Compute the graph Laplacian \(L\).
     - Output: graph object, adjacency matrix, Laplacian.

   - **Graph-based Label Smoothing** (via `LabelSmoother`):
     - For each smoothing hyperparameter \(\gamma\) in `gamma_values`:
       - Compute smoothed fitness labels: \(\hat{Y} = (I + \gamma L)^{-1} Y\).
       - Generate different smoothed label sets for hyperparameter tuning.
     - Optionally, select \(\gamma\) by validation or fixed hypergrid.

   - **Predictor Model Training** (via `Trainer` or `Model`):
     - Using the sequences and smoothed labels:
       - Encode sequences in one-hot format.
       - Train the neural network architecture specified (`architecture: "cnn"` or other).
       - Use optimizer parameters: learning rate = 1e-3, epochs=50, batch size=128.
     - Save model checkpoints, if needed.
     - Validate: compute model's MAE or RMSE on validation subset or held-out data.

   - **In-silico Evaluation**:
     - Predict fitness values for the sequences of interest:
       - Top sequences selected after sampling.
       - Use `predict_batch` method for efficiency.
     - Compute metrics:
       - Median fitness.
       - Diversity (using Levenshtein between sequences).
       - Novelty (comparison with initial training set).
       - Fitness jump from initial training sequences.
     - Store evaluation metrics for comparison.

   - **Sequence Sampling with GWG & Clustering** (via `Sampling`):
     - For each sequence in the current pool:
       - Run specified GWG rounds (`gwg_rounds=15`).
       - Proposal generation:
         - Use gradient info: `compute_gradients(sequence)`.
         - Form proposal distribution with temperature grid (`[0.01, 0.1, 1, 10]`).
         - Sample proposals considering mutations within 1 Hamming distance.
       - Acceptance step via Metropolis-Hastings.
     - **Clustering and Selection**:
       - Cluster sampled sequences:
         - Use scikit-learn `AgglomerativeClustering` or similar, based on Levenshtein distance.
         - Number of clusters: 20 (hyperparameter from `clustering_clusters`).
       - Select top-fitness sequence from each cluster (`select_top_per_cluster`).
     - **Repeat** for all rounds, updating the starting sequences for next round based on top sequences.
     - Accumulate all generated sequences for final evaluation.

4. **Post-Processing & Final Evaluation**:
   - Aggregate top sequences across rounds.
   - Compute final metrics:
     - Max fitness.
     - Average fitness.
     - Diversity, novelty.
     - Extrapolation capability if holdout or unseen sequences are available.
   - Record metrics for all hyperparameter settings if performing a grid search.

---

# Detailed Step-by-Step Logic

### Step 1: Load Configuration & Set Seeds
- Parse YAML config or provided dictionary.
- Set random seed (`seed=42`) for `numpy`, `scipy`, `torch`, `jax`, and any other stochastic modules.

### Step 2: Dataset Loading and Filtering
- Instantiate `DatasetLoader`:
  - Arguments: dataset path, filters for the specific dataset and difficulty.
  - Load sequences, fitness, applying percentile and mutational gap filters.
- For reproducibility, save the filtered datasets or logs of filtering parameters.

### Step 3: Graph Construction
- Call `graph_utils.build_graph(sequences, knn_neighbors)`:
  - Generate pairwise distances (Levenshtein or Hamming).
  - Build graph (via k-NN edges).
- Compute Laplacian `L` using `scipy.sparse.linalg` as per Algorithm 5.

### Step 4: Label Smoothing
- For each value in `gamma_values`:
  - Compute smoothed labels:
    - \(\hat{Y} = (I + \gamma L)^{-1} Y\).
    - Use sparse matrix solvers (`scipy.sparse.linalg.spsolve` or `factorized`).
  - Store or keep all smoothed label sets for potential hyperparameter selection.

### Step 5: Model Training
- Instantiate `model.py` predictor:
  - Configure based on `architecture` (CNN).
  - Set optimizer: ADAM with `learning_rate=1e-3`, `epochs=50`.
- Train predictor on sequences with each smoothed label set:
  - Use `trainer.train(list_of_sequences, smoothed_labels)`.
- Save trained model weights for inference.

### Step 6: Predictor Evaluation & Metrics
- Use `predictor.predict_batch(sequences)`:
  - Get estimated fitness.
- Compute:
  - Median predicted fitness.
  - Diversity: median of Levenshtein distances between all pairs.
  - Novelty: median distance to training data.
  - Fitness improvement (max) relative to starting data.
- Log or output these metrics for all hyperparameter configurations.

### Step 7: Sampling with GWG & Clustering
- Initialize sample pool:
  - Starting from the filtered training sequences.
- For each round in `gwg_rounds`:
  - For each sequence:
    - Compute gradients: `predictor.compute_gradients(sequence)`.
    - Generate proposals \((x^\prime)\) within 1 Hamming distance:
      - Use the softmax over gradient-based logits scaled by temperature \(\tau\).
    - Accept/reject proposals via MH criterion.
  - Cluster resulting sequences:
    - Use Levenshtein distance, with `clustering_clusters=20`.
  - From each cluster, select the highest predicted fitness sequence.
- Collect all sequences from each round for evaluation.

### Step 8: Final Evaluation
- Predict fitness of generated sequences.
- Calculate fitness jump:
  - Max fitness observed minus max in initial training set.
- Calculate average fitness, diversity, novelty.
- Record results.

---

# Additional Considerations & Assumptions
- **Hyperparameter extension**:
  - Iterate over `gamma_values`, `temperature_grid`, `clustering_clusters`.
  - Use validation metrics or in-silico fitness to select best hyperparameters.
- **Reproducibility**:
  - For multiple runs, fix random seeds.
  - Log all hyperparameters and seed states.
- **Data handling**:
  - Validation datasets may be constructed by holding out a subset or using explicit test splits.
- **Computational efficiency**:
  - Use batch processing where possible.
  - Leverage sparse matrices for graph calculations.
  - Clustering and sequence similarity computations may be optimized or approximated.

---

# Summary
`main.py` will perform the following core logical steps:
- Load and filter datasets.
- Construct sequence similarity graphs.
- Smooth fitness labels with hyperparameter tuning.
- Train a predictor model on smoothed labels.
- Run iterative GWG sampling with clustering.
- Evaluate the sequences generated at each stage.
- Output metrics for comparison/optiFromon.

This linear, modular approach ensures clarity, reproducibility, and systematic hyperparameter exploration aligned with the paper's methodology and experimental design.

## model.py

{
  "Purpose": "The model.py module defines a flexible neural network architecture for predicting protein sequence fitness. It should support both training (via backpropagation) and inference operations, including gradient computation with respect to input sequences. The architecture should be adaptable according to hyperparameters specified in the configuration, particularly the 'architecture' key (e.g., CNN, transformer).", 
  
  "Design Principles": [
    "Modularity: The core class should encapsulate architecture, training, prediction, and gradient computation.",
    "Configurable: Hyperparameters such as architecture type, layer sizes, dropout rates, learning rate, and batch size should be adjustable and parsed from a provided configuration.",
    "Differentiability: The prediction function must be differentiable with respect to the input sequence representation, enabling gradient-based sampling schemes like GWG.",
    "Compatibility: Support both 'train' and 'predict' modes, along with gradient computation.",
    "Reproducibility: Initialize with seed parameters, ensuring reproducibility across runs."
  ],
  
  "Input & Output Expectations": [
    "Input: Sequence data as either one-hot encoded tensors or embeddings.",
    "Outputs: For prediction, a scalar fitness score; for gradient computation, gradients with respect to sequence representations."
  ],
  
  "Assumptions & Clarifications": [
    "Framework choice: Select either PyTorch or Jax/Flax for neural models. Based on system requirements and ease of gradient computation, favor PyTorch for flexibility and maturity.",
    "Sequence encoding: Typically use one-hot encoding with shape [batch_size, sequence_length, vocab_size], or an embedding of shape [batch_size, sequence_length, embedding_dim].",
    "Hyperparameters such as layer sizes, dropout, etc., will be passed via a hyperparameter dictionary or a specialized config object.",
    "Model training: Loss function should be MSE (mean squared error) between predicted fitness and smoothed label, as per the paper.",
    "For gradient computation: Enable autograd to obtain \(\nabla_x f_\theta (x)\) with respect to sequence embeddings or input representations.",
    "Predictor supports batch inference for efficiency during sampling."
  ],
  
  "Implementation details": [
    "Class: SequenceFitnessPredictor",
    "Constructor: __init__(self, config: dict) - should initialize model layers based on architecture type and parameters.",
    "Attributes: network layers, optimizer, hyperparameters.",
    "Methods:",
    " - train(self, sequences: List[str], labels: np.ndarray): trains the model on provided data.",
    " - predict(self, sequence: str) -> float: predicts scalar fitness for a single sequence.",
    " - predict_batch(self, sequences: List[str]) -> np.ndarray: batch inference for multiple sequences.",
    " - compute_gradients(self, sequence: str) -> np.ndarray: computes and returns \(\nabla_x f_\theta(x)\). Implementation can utilize autograd (PyTorch's backward) on the input tensor.",
    "Serialization: Save and load model weights for reproducibility."
  ],
  
  "Layer-specific considerations": [
    "For CNN architecture:",
    " - Input: sequence one-hot encoding or embeddings.",
    " - Layers: 1D convolution with kernel size 5, number of channels (e.g., 256 as per paper).",
    " - Activation: ReLU.",
    " - Pooling: Max-pooling.",
    " - Fully connected (dense) layer to output a scalar prediction.",
    " - Initialize weights with standard schemes (e.g., Xavier or He).",
    "For transformer options:",
    " - Input embedding layer, positional encoding, transformer encoder layers, etc.",
    " - Output: scalar via final linear layer.",
    ""
  ],
  
  "Hyperparameter integration": [
    "Read from config['predictor_model']:",
    " - 'architecture': 'cnn' or 'transformer' (expandable).",
    " - 'learning_rate': float.",
    " - 'batch_size': int.",
    " - 'dropout_rate': float.",
    " - 'epochs': int.",
    "Optional: Allow overrides or modular extension for different architectures."
  ],
  
  "Gradient computation considerations": [
    "Ensure input tensor is set to require_grad=True.",
    "Process: input tensor -> forward pass -> compute prediction -> backward() -> get gradients w.r.t. input.",
    "Method should return the gradient tensor with respect to sequence input, matching the input shape.",
    "Ensure handling one-hot inputs or embeddings accordingly."
  ],
  
  "Reproducibility & debugging": [
    "Set random seeds for weight initialization and data shuffling.",
    "Log model configurations and hyperparameters.",
    "Return intermediate outputs for diagnostic purposes if needed."
  ],
  
  "Summary": "The model.py file defines a SequenceFitnessPredictor class, supporting configurable architecture (CNN vs transformer), training with MSE loss, inference, and differentiable gradient computation. It should facilitate training on smoothed labels and provide gradient information for GWG sampling, adhering to the specifications given in the paper and aligned with the provided configuration."

## sampling.py

**Logic Analysis for sampling.py**

---

### **Overview of module purpose**

The `sampling.py` module implements the core sampling routine for protein sequence optimization guided by a trained predictor model. The process involves generating mutations of input sequences, scoring these mutations using gradients, accepting or rejecting proposals via Metropolis-Hastings, clustering the accepted sequences to control diversity and computational load, and selecting top sequences for the next iteration. The procedure is based on the GWG (Gibbs With Gradients) algorithm described in the paper, with modifications for efficiency and modularity.

---

### **Core subcomponents and their roles**

1. **Proposal Generation (`mutate`)**:
    - Generates candidate sequences by mutating current sequences within a specified neighborhood.
    - Neighborhood is typically the set of sequences differing by a single amino acid (Hamming distance = 1).
    - Mutation involves selecting a position within the sequence and replacing the amino acid with a different one.
    - Uses the gradient info to inform mutation probabilities but, for proposal generation, typically samples uniformly from neighborhood, with optional influence from the gradient (softmax over gradient scores).

2. **Gradient Computation (`compute_gradients`)**:
    - Calculates \(\nabla_x f_\theta(x)\), the gradient of the predictor's output with respect to the input sequence.
    - Necessary for shaping proposal distribution (\(q(x'|x)\)) as per eq. (2).
    - Implemented with autograd frameworks (PyTorch or Jax).

3. **Proposal Distribution (`q`)**:
    - Defines the probability \(q(x'|x)\) for proposing mutations.
    - Uses a softmax over gradient-based scores scaled by temperature (\(\tau\)), favoring mutations that are predicted to increase fitness.
    - Implemented as per eq. (2) and the pseudocode: probabilities proportional to \(\exp\left(\frac{d_\theta(x)_i,j}{\tau}\right)\).

4. **Acceptance Step (Metropolis-Hastings test)**:
    - Compares the energies \(f_\theta(x')\) and \(f_\theta(x)\), adjusting for proposal probabilities.
    - Accepts move with probability:
      \[
      \alpha = \min \left( 1, \exp(f_\theta(x') - f_\theta(x)) \times \frac{q(x|x')}{q(x'|x)} \right)
      \]
    - Implementation involves generating a uniform random number and comparing to \(\alpha\).

5. **Clustering accepted sequences (`cluster_sequences`)**:
    - Clusters sequences based on sequence similarity (Levenshtein or Hamming distance).
    - Methods: hierarchical clustering (e.g., agglomerative clustering) with a pre-defined number of clusters, per task configuration (e.g., 20 clusters as the default).
    - Goal: reduce redundancy and select top sequences, which prevents exponential growth of sequence pool and maintains diversity.

6. **Selection of Top Sequences per Cluster (`select_top_per_cluster`)**:
    - For each cluster, choose the accepted sequence with the highest predicted fitness according to the predictor model.
    - Ensures that each cluster contributes a representative high-fitness sequence to the next iteration.
    - Results in a list of sequences that form the pool for the next round.

7. **Main sampling loop (`run_sampling`)**:
    - Initializes with either seed sequences or sequences accepted in previous round.
    - For each round:
      - Generates proposals for each sequence using `mutate`.
      - Computes gradients, evaluates proposal probabilities.
      - Performs acceptance test.
      - Collects accepted sequences.
      - Clusters accepted sequences.
      - Selects top sequences per cluster.
    - Repeats for a fixed number of rounds (`gwg_rounds` hyperparameter).

---

### **Input/Output specifications**

- **Inputs**:
  - Current sequence pool (`sequences: List[str]`).
  - Predictor model (`predictor`) with methods for prediction, gradient computation.
  - Hyperparameters:
    - `proposals_per_sequence` — number of proposals per sequence at each round.
    - `temperature` (`tau`) for proposal softmax.
    - `clustering_clusters` (`C`) — number of clusters for reduce step.
    - `mutation_batch_size` — total proposals generated per sequence per round.

- **Outputs**:
  - Lists of accepted sequences after all rounds: `List[str]`.
  - Possibly include the top sequences by fitness at end for evaluation.

---

### **Algorithm flow**

**Step 1: Initialization**
- Initialize an empty list for accepted sequences per input sequence.

**Step 2: For each round (`r in 0..R-1`)**
- For each sequence in current pool:
  - Generate `proposal_per_sequence` mutations:
    - For each proposal:
      - Compute \(\nabla_x f_\theta(x)\).
      - Compute proposal probabilities \(q(x'|x)\).
      - Sample candidate \(x'\).
      - Evaluate \(f_\theta(x')\).

  - For each proposal:
    - Accept or reject based on MH criterion.
    - If accepted, add to `accepted_sequences`.

- After processing all sequences:
  - **Cluster accepted sequences** into `clustering_clusters`.
  - **Select top sequence in each cluster** with highest predictor score.
  - Set these sequences as input for next round.

**Step 3: Termination**
- After \(R\) rounds, output the union of accepted sequences, or the top \(K\) by predicted fitness for evaluation.

---

### **Implementation details & considerations**

- **Gradient calculation**:
  - Sequence input is one-hot encoded or embedded.
  - For differentiability: sequences should be represented in a form suitable for autograd (e.g., differentiable embeddings).
  - Use the method provided in predictor model (e.g., `compute_gradients(sequence)`).

- **Mutation proposal**:
  - Efficiently generate mutations within 1-Hamming distance:
    - For each position, try substituting other amino acids (excluding current amino acid).
  - If the sequence is represented as a one-hot or embedding:
    - Mutate the one-hot vectors accordingly.
  - Store proposals along with their gradient scores for softmax calculation.

- **Proposal distribution**:
  - For each mutation candidate:
    - Compute \(d_\theta(x)_i,j\).
    - Compute softmax probabilities with temperature \(\tau\).

- **Proposal sampling**:
  - Use numpy or torch to sample from the categorical distribution dictated by softmax scores.
  
- **Acceptance probability**:
  - Efficiently compute \(f_\theta(x')\) and \(f_\theta(x)\).
  - Compute proposal ratios \(q(x|x')/q(x'|x)\).
  - Perform the MH step with uniform random number.

- **Clustering**:
  - Use scikit-learn's hierarchical clustering (`AgglomerativeClustering`) with distances computed via Levenshtein distance.
  - For large datasets, consider approximate clustering or precompute distances offline.
  
- **Selection**:
  - Evaluate each sequence's fitness prediction.
  - Pick top in each cluster.

- **Loop controls**:
  - Logging acceptance rates, diversity measures.
  - Early stopping if convergence behavior observed.

---

### **Hyperparameters and flexibility**

- `proposals_per_sequence`: e.g., 100 proposals per sequence.
- `gwg_rounds`: e.g., 15.
- `mutate`: mutation function parameters.
- `temperature` (`tau`): test values grid, e.g. 0.01–10.
- `clustering_clusters`: suggested around 20, tune as per dataset.
- Random seed for reproducibility.

---

### **Edge cases & robustness**

- No proposals accepted in a round: consider fallback or skip mutation.
- Sequences with invalid mutations: enforce amino acid constraints.
- Large datasets: optimize clustering and mutation generation for scalability.
- Sequence length variations: ensure input representations are consistent.

---

### **Summary**

The `sampling.py` module implements an iterative, gradient-informed proposal and MH acceptance routine, incorporating clustering to manage sequence pool size, aimed at efficiently exploring high-fitness regions in protein sequence space. It combines the theoretical components laid out in the paper with practical implementation considerations aligned with the provided package structure and configuration.

---

If further clarification or modular breakdowns are needed, I can prepare detailed pseudocode or specific interface definitions.

## trainer.py

{
  "trainer.py": "Logic Analysis for Implementing the Training Loop Module\n\nOverview:\nThis module manages the training of the neural network predictor \(f_\\theta\) that learns to map protein sequences to their fitness scores, specifically the smoothed labels obtained after graph-based label smoothing. It encompasses data batching, loss calculation, model parameter updates, validation, and checkpointing to ensure reproducibility and optimal performance.\n\nCore Responsibilities:\n- Load training and validation datasets.\n- Process batches: convert sequences to model inputs.\n- Compute model outputs and loss against target smoothed labels.\n- Perform backpropagation and optimizer steps.\n- Track training/validation metrics.\n- Save model checkpoints based on validation performance.\n- Allow hyperparameter configuration from an external or internal dict.\n\nAssumptions & Requirements:\n- Data:\n   - Sequences: list of strings of amino acid sequences, each of fixed length M.\n   - Labels: numpy array or list of smoothed fitness labels, continuous.\n   - Data is already processed and filtered according to the dataset_filters specified in config.yaml.\n- Model:\n   - A class (e.g., in model.py) implementing methods:\n       - predict(sequence): outputs scalar prediction.\n       - predict_batch(sequences): batch inference.\n       - compute_gradients(sequence): returns gradient w.r.t input, used if needed.\n- Framework:\n   - Preferably using PyTorch (recommended) with autograd for gradient computations.\n- Hyperparameters:\n   - Learning rate, batch size, epochs, dropout, optimizer are configurable.\n- Reproducibility:\n   - Fix random seed variables.\n- Metrics:\n   - Track loss (e.g., MSE or MAE), possibly MAE aligned with evaluation.\n\nStep-by-Step Logic:\n1. Initialization:\n   - Read hyperparameters from input config dict (learning rate, batch size, epochs, seed, dropout, optimizer). \n   - Initialize the model from model.py, e.g., Model(model_architecture, dropout_rate). \n   - Set up optimizer (Adam) with specified parameters.\n   - Initialize data loaders or batching strategies for train and validation sets.\n   - Load datasets: sequences and smoothed labels (np.ndarray).\n   - Set up logging, checkpoint paths, and seed for reproducibility.\n2. Data Preparation:\n   - Convert sequences into model input format:\n       - One-hot encoding of amino acids if required, shape (batch_size, M, vocab_size=20).\n       - Or embeddings if using an embedding layer.\n   - Ensure labels are aligned with sequences.\n3. Training Loop (per epoch):\n   For epoch in range(1, epochs+1):\n     - Shuffle training data indices (deterministically if seed fixed).\n     - Process data in batches:\n       - Extract batch sequences and labels.\n       - Convert sequences to model inputs.\n       - Forward pass: obtain predictions.\n       - Compute loss: for regression, typically MSE or MAE between predicted and target labels.\n       - Backward pass: compute gradients.\n       - Optimizer step: update model parameters.\n       - Log training metric (e.g., mean loss) for the batch.\n     - Validation step:\n       - Disable gradient computation.\n       - Process validation sequences in batches: predict and compute validation loss.\n       - Aggregate validation metrics.\n     - Checkpointing:\n       - Save model state if current validation loss is better than previous best.\n     - Logging:\n       - Record training/validation loss, other metrics.\n4. Post-training:\n   - Save final model.\n   - Return the trained model object.\n\nOptional Enhancements:\n- Early Stopping: stop training if validation loss does not improve over several epochs.\n- Learning Rate Scheduler: decay learning rate based on epochs or performance.\n- Logging via progress bar (tqdm) or tensorboard.\n\nImplementation details:\n- Utilize tqdm for progress bar over epochs.\n- Use DataLoader if the dataset is large, else process in native numpy slices.\n- Incorporate seed setting for reproducibility: numpy, torch random seed.\n\nThis logical structure aligns with standard ML training practices and is compatible with the dependencies and interfaces specified in the overall plan, ensuring modularity and clarity for sensory implementation in code."
}

