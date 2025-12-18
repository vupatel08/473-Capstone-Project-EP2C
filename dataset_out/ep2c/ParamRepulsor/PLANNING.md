# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, structured plan to reproduce the core experiments and methodology described in the paper. The plan emphasizes both the algorithmic implementation details and the experimental setup/cross-validation procedures, based on the paper's content and supplementary information.

---

## 1. Core Methodology: Implementation Strategy for ParamRepulsor

### **a. Foundation: The Loss Function & Parametric Architecture**

- **Loss Function**:
  - The core loss follows the _contrastive/rejective framework_ with a strong emphasis on negative sampling via hard negatives (mid-near pairs).
  - The final triplet-based loss (Eq. in Sec. 4, Appendix D) involves the sum over pairs: attraction for nn, repulsive force for false negatives (FP) and mid-near negatives (MN).
  - The **similarity functions** (`q_θ`) are explicitly defined in Sec. 2 & 4:
    - For NN pairs: `q_θ(y_i, y_j) = exp(-(d_2(i,j) + 10) / (d_2(i,j) + 10))` simplified as a scaled exponential.
    - For F P (farthest or random) pairs: similar functions based on `d_2(i,j)`.
  - The **loss E** combines these similarity-based forces, with dedicated weights (`w_NB`, `w_MN`, `w_FP`).

- **Neural Network Architecture**:
  - Use a shallow multi-layer perceptron with 3 hidden layers, each with **100 neurons**, ReLU activation.
  - Input: high-dimensional data `X` of shape `(N, D)`.
  - Output: 2D embeddings `Y` of shape `(N, 2)` (or other user-specified low dims).

- **ParamRepulsor Implementation**:
  - **Hard Negative Mining (HN)**:
    - For each anchor point:
      - Sample **6 points** uniformly (or as specified).
      - Select the **second closest point** as MN.
    - These MN points are designed to be challenging negatives with high gradients, reducing false negatives.
  - **Batch construction**:
    - During each epoch:
      - Sample an **entire batch** of size `b` (e.g., 1024).
      - For each point in batch:
        - Retrieve precomputed NN, MN, FP sets.
        - Sample respective pairs (`n_NB`, `n_MN`, `n_FP`) within the batch.
      - Compute the **pairwise distances** in the current embedding for these pairs.
  - **Loss calculation**:
    - From pairwise distances, compute the similarity scores (`q_θ`) as above.
    - Sum over all pairs within the batch with respective weights (`w_NB`, `w_MN`, `w_FP`).
  - **Gradient update**:
    - Use Adam optimizer with learning rate `η` (e.g., 0.001).
    - Update neural network parameters to minimize the loss.
  
### **b. Additional Implementation Details**

- **Initialization**:
  - Neural network parameters: Xavier initialization.
  - Embedding space: possibly scaled or normalized to aid training.
- **Distance metric**:
  - Use Euclidean (`L2`) distance in both high-dimensional and low-dimensional space.
  - Keep track of distance distributions for analysis.
- **Sampling during training**:
  - For each epoch:
    - For each point:
      - Precompute NN, MN, FP sets **once** before training (or update periodically if datasets are large).
      - Sample pairs dynamically within each batch to maximize diversity.
- **Black-box functions**:
  - Compute pairwise Euclidean distances in the low-dimensional space efficiently (batch-wise).
  - Calculate the similarity functions `q_θ` as per the equations.
- **Stopping criteria**:
  - Fixed number of epochs (`nepochs`, e.g., 100).
  - Early stopping based on local or global distance preservation metrics if desired.

---

## 2. Dataset Requirements & Experimental Setup

### **a. Datasets**

- Use the **MNIST** dataset (as in the paper) with variations:
  - 10,000 points for initial experimentation.
  - Larger datasets like COIL-20, COIL-100, USPS, and biological datasets (scRNA-seq, T-cell, etc.) for scalability validation.
- **Preprocessing**:
  - Standardize or normalize features.
  - Optional: PCA reduction to 50–100 dims to optimize computational efficiency.
- **Features**:
  - For image datasets (MNIST, COILs, USPS): raw pixels or PCA-reduced features.
  - For biological data: gene expression matrices; normalize using log or scaling if needed.

### **b. Constructing Pair Sets**

- **k-Nearest Neighbors (kNN)**:
  - Use scikit-learn's `neighbors.NearestNeighbors` for fitting kNN models.
  - Choose `k=15~30` (per paper's hyperparameter snacks).
- **Mid-near pairs (MN)**:
  - For each seed point:
    - Sample 6 points randomly within the dataset.
    - Determine their 2nd closest neighbor as MN (via precomputed kNN distances).
- **Farthest pairs (FP)**:
  - Sample uniformly from points not in the NN set, from the entire dataset.
  - Use uniform sampling for the negatives but limited to the sampled set size for efficiency.

### **c. Hyperparameters**

- Number of epochs (`nepochs`): 100–200.
- Batch size (`b`): choose 1024–2048 depending on memory.
- Learning rate (`η`): default 0.001 or tuned.
- Weights (`w_NB`, `w_MN`, `w_FP`): set based on dataset size, e.g., 1.0 for NN pairs, 0.5 for MN, 0.2 for FP.
- Number of negative samples per embedded point: 10–30.
- Number of MN points per anchor: 6 (as in Sec. 4).

### **d. Computational Resources**

- Use GPU (e.g., NVIDIA V100 or A100) for training, especially on large datasets (~100k+ points).
- Memory: sufficient to hold `X`, precomputed neighbor info, batch data, and neural network parameters.
  
---

## 3. Evaluation Metrics & Validation Strategies

### **a. Preservation of Structure**

- **Local structure**:
  - 10-NN accuracy in embedding space.
  - Triplet preservation (Sec. 4 & 5.3, Tables 3 & 5).
- **Global structure**:
  - Distance correlation between high- and low-dimensional embeddings (Tables 4 & 6).
  - Triplet preservation for distant triplets.

### **b. Visualization & Qualitative Checks**

- Plot 2D embeddings with cluster labels to visually assess boundary sharpness, cluster separation.
- Distance histograms for different pair types (NN, MN, FP) to verify effective hard negative mining.

### **c. Reproducibility**

- Run multiple random initializations (e.g., 10 runs per dataset).
- Report mean ± std for all metrics.
- Document hyperparameter settings clearly.

---

## 4. Additional Considerations & Troubleshooting

- **Parameter Sensitivity**:
  - Validate baseline hyperparameters (learning rate, batch size).
  - Tune weights (`w_NB`, `w_MN`, `w_FP`) via grid search if necessary.
- **Precomputing NN, MN, FP**:
  - Since the info remains fairly static during training, precompute once and keep fixed.
  - For large datasets, consider approximate kNN methods (FAISS or Annoy).
- **Scalability**:
  - For datasets > 10^5, consider mini-batch kNN or using approximate NN.
  - Leverage sparse data structures if features are sparse.

---

## 5. Summary: Step-by-Step Workflow

1. **Preprocessing**:
   - Normalize features.
   - Optional PCA reduction.

2. **Construct Graphs**:
   - Build kNN graph.
   - Generate MN and FP pair sets via sampling strategies.

3. **Model Initialization**:
   - Define 3-layer MLP with 100 neurons each, ReLU.
   - Initialize weights.

4. **Training Loop**:
   - For each epoch:
     - Shuffle data, sample batch of points.
     - Retrieve NN, MN, FP pairs within batch.
     - Compute low-D distances and similarity scores.
     - Calculate total loss with weights.
     - Backpropagate, update with Adam.
     - Periodically recompute or update pair sets if dataset is large.

5. **Validation & Evaluation**:
   - After training, compute validation metrics (NN accuracy, triplet accuracy, distance correlation).
   - Plot embeddings to visualize boundary sharpness.
   - Repeat with different seeds for statistical robustness.

6. **Comparison & Analysis**:
   - Reproduce baseline methods (e.g., UMAP, t-SNE) with similar hyperparameters.
   - Report structure metrics and computational times.

---

This roadmap should bridge the methodological details in the paper and produce a comprehensive, scalable implementation plan suitable for later coding. It covers architecture, training procedures, dataset handling, hyperparameters, evaluation, and safety checks, in line with the authors' techniques and reported results.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the ParamRepulsor method as a modular system using PyTorch for neural network modeling, scikit-learn for graph construction, and FAISS for efficient neighbor querying. The system will consist of a dataset loader, a model class implementing the shallow MLP, a trainer to handle pair sampling, loss computation with mid-near hard negatives, and training iterations, and an evaluation module. Precompute neighbor sets (kNN, MN, FP) outside the training loop, then perform batch-wise pair sampling and loss calculation. Use torch DataLoader for batching, and optimize with Adam optimizer. Save the trained model and generate evaluation metrics after training.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "utils.py",
        "pair_sampler.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> Tuple[np.ndarray, np.ndarray]\n        +preprocess() -> np.ndarray\n    }\n    class NeuralNetwork {\n        +__init__(params: dict)\n        +forward(x: torch.Tensor) -> torch.Tensor\n        +get_embedding(x: torch.Tensor) -> torch.Tensor\n    }\n    class PairSampler {\n        +__init__(knn_indices: np.ndarray, knn_distances: np.ndarray, n_points: int)\n        +generate_pairs(batch_indices: np.ndarray) -> List[Tuple[int, int, str]]  # pairs with labels: 'NN', 'MN', 'FP'\n    }\n    class Trainer {\n        +__init__(model: NeuralNetwork, data: Tuple[np.ndarray, np.ndarray], pair_sampler: PairSampler, config: dict)\n        +train() -> None\n        +save_model(file_path: str) -> None\n    }\n    class Evaluation {\n        +__init__(model: NeuralNetwork, data: Tuple[np.ndarray, np.ndarray])\n        +evaluate() -> dict\n        +visualize_embeddings() -> None\n    }\n\nmain --> DatasetLoader\nmain --> Trainer\nmain --> Evaluation\nTrainer --> NeuralNetwork\nPairSampler --> DatasetLoader (for neighbor info)\n",
    "Program call flow": "sequenceDiagram\n    participant M as main\n    participant DL as DatasetLoader\n    participant NS as PairSampler\n    participant NM as NeuralNetwork\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: load_data()\n    activate DL\n    DL-->>M: dataset\n    M->>DL: prepare neighbor graphs (kNN, MN, FP)\n    DL-->>M: neighbor info\n    M->>NM: initialize model()\n    deactivate DL\n    M->>NS: create with neighbor info\n    M->>TR: train(model, dataset, pair_sampler)\n    TR->>NM: get embeddings(batch_points)\n    NM-->>TR: embeddings\n    TR->>TR: compute loss over pairs\n    TR->>NM: backpropagate and optimize\n    loop over epochs\n        TR->>pair_sampler: generate pairs for batch\n        pair_sampler-->>TR: pairs list\n        TR->>NM: forward(batch data)\n        TR->>TR: compute loss\n        TR->>NM: update model\n    end\n    TR->>main: save_model()\n    main->>EV: evaluate(model, dataset)\n    EV->>NM: get final embeddings\n    EV->>EV: compute metrics (NN accuracy, triplet, correlation)\n    EV->>main: report results\n",
    "Anything UNCLEAR": "Clarify dataset input formats (e.g., raw pixel arrays, feature matrices), and any specific evaluation thresholds or metrics preferred from the paper. Confirm whether to reproduce the entire baseline comparisons or only specific main experiments with parameters."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "scikit-learn==0.24.2",
        "faiss-cpu==1.7.2",
        "tqdm==4.62.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Implements DatasetLoader class. Responsible for loading datasets (MNIST, COIL, USPS, biological data), normalization, PCA reduction, and precomputing neighbor graphs using scikit-learn's NearestNeighbors. It provides methods to load data, preprocess, and output data and neighbor info."
        ],
        [
            "pair_sampler.py",
            "Implements PairSampler class. Uses neighbor graphs to generate pairs for training: NN pairs (neighbor sets), MN (mid-near challenging negatives), and FP (far negatives). Handles pair sampling during training batches, guided by precomputed neighbor info for efficiency."
        ],
        [
            "model.py",
            "Defines NeuralNetwork class implementing a shallow MLP with 3 layers and 100 neurons each, ReLU activation. Handles initialization, forward pass, and getting embedding vectors."
        ],
        [
            "trainer.py",
            "Handles training loop: samples batches of points, constructs pair sets per batch, computes the loss based on pairwise distances and similarity functions (eq in Sec. 4), performs backpropagation with Adam optimizer, and periodically saves model checkpoints."
        ],
        [
            "evaluation.py",
            "Implements Evaluation class with methods for computing local accuracy (NN), triplet preservation, global distance correlation, and visualization of embeddings. Uses trained model for embedding new data and calculates metrics."
        ],
        [
            "main.py",
            "Main entry point orchestrates the entire process: loads dataset, prepares neighbor graphs, initializes model, pair sampler, trainer, runs training epochs, evaluates, and outputs results. Manages the sequence of data flow between modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "pair_sampler.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n\n# No front-end API needed. If needed, APIs would include:\n# GET /load_data\n# POST /train\n# GET /evaluate\n# POST /save_model\n# etc.\n\n# For the current scope, no API spec required.",
    "Shared Knowledge": "Shared functions for distance calculations, pairwise similarity computations, and neighbor graph data structures. Utility functions for serialization/deserialization of data and models. Dataset preprocessing routines shared between dataset_loader.py and trainer.py.",
    "Anything UNCLEAR": "Clarify dataset format specifics: feature dimensionality, file formats, and whether specific biological datasets need special normalization steps. Also, confirm how many experiments and hyperparameters should be reproduced, and whether to include baseline comparisons."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001  # Standard initial learning rate for Adam optimizer
  batch_size: 1024     # Practical size for GPU memory and performance
  epochs: 100          # Number of epochs, as commonly used in similar studies
  
model:
  hidden_layers: 3     # Number of hidden layers in MLP
  neurons_per_layer: 100  # Number of neurons per hidden layer
  activation: relu     # Activation function
  
pair_sampling:
  neighbor_count: 15   # k for k-NN neighbor graph
  mid_near_sample_count: 6  # Number of mid-near pairs per anchor
  negative_samples_per_point: 20  # Negatives per anchor during batch
  
loss_weights:
  weight_NN: 1.0       # Weight for neighbor pairs in loss
  weight_MN: 0.5       # Weight for mid-near pairs
  weight_FP: 0.2       # Weight for far negatives
  
optimization:
  optimizer: adam      # Use Adam optimizer
  learning_rate: 0.001 # Learning rate for optimizer
  betas: [0.9, 0.999] # Betas parameters for Adam
  
hyperparameters:
  num_epochs: 100       # Number of training epochs
  report_interval: 10  # Epoch interval for reporting metrics
  
dataset:
  name: MNIST          # Dataset name for initial experiments
  feature_dim: auto    # Use raw or PCA features (auto as placeholder)
  data_path: ./data/mnist.pkl  # Path to dataset if applicable

misc:
  random_seed: 42      # Seed for reproducibility
  save_model_path: ./models/paramreprulsor.pth  # Path to save trained model
```

---

**Note:** Full configuration is available in `planning_config.yaml`
