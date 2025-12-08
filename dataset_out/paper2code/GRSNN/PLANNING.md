# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here is a comprehensive, structured plan for reproducing the experiments and methodologies from the paper "Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning". The plan will cover foundational details, implementation specifics, datasets, hyperparameters, experimental settings, and evaluation strategies, all aligned with the paper's content.

---

## 1. Core Methodology and Theoretical Foundations

### a. Model Architecture: Graph Reasoning Spiking Neural Network (GRSNN)
- **Node Representation:**  
  - Assign *N* spiking neurons to each graph node (e.g., N=32 based on ablation analysis).  
  - Neurons are modeled as Leaky Integrate-and-Fire (LIF) neurons with membrane potentials, thresholds, and decay constants.  
- **Edge Characterization:**  
  - Each edge between nodes is represented via synaptic weight and delay \(d_{ij}\).  
  - Delays are task- and relation-dependent, parameterized as \(d_{ij}^{q} = \beta \sigma(W_{r} r^{q} + b_{r})\), quantized as integers for simulation, where \(\sigma\) is sigmoid, \(\beta\) is scale, and \(r^{q}\) is relation embedding.
  - For homogeneous graphs, delay reduces to constant or relation-independent values.

### b. Spike Propagation & Temporal Coding
- **Dynamics:**  
  - Use either the discrete-time LIF simulation equations (Eq. (7)) or SRM formulations for precise gradient computation.  
  - Excitatory inputs cause membrane potential increases; spikes occur when threshold is crossed, followed by reset.
- **Temporal Encoding:**  
  - Spike times encode relation and path information.  
  - Edge delays modulate signal timing, representing relation importance and path properties.
- **Path Representation:**  
  - Spike trains propagate along graph paths; their arrival times at target nodes encode path lengths or relevance.

### c. Graph Path Formulation & Relation Modeling
- **Path Representation (\(\mathbf{h}^{q}(x,y)\))**:  
  - Sum over paths \(P\) from \(x\) to \(y\), combining edge representations using generalized path formulation (product or sum over path edges).  
  - Specific relation models: Katz (multiplicative decay), Personalized PageRank, and Graph Distance encoded in delay and decoding functions.
- **Decoding of Path Importance:**  
  - Spike trains at target nodes are decoded to estimate the likelihood of a triplet (x, q, y).  
  - Decoding functions include:  
    - First spike time (shortest path).  
    - Summation of spike magnitudes with exponential weights (relation importance).  
    - Vector of spike counts across neurons (path diversity).

### d. Learning & Gradient Computation
- Use surrogate gradients for non-differentiable spikes (e.g., sigmoid derivative).  
- Implement backpropagation through discrete time steps (Eq. (8)-(9), Appendix E)  
- Gradients w.r.t. synaptic weights \(w_{ij}\), delays \(d_{ij}^{q}\), and relation embeddings \(r^{q}\) are computed via chain rule, leveraging the SRM formulations and trace variables \(tr_{ij}\).

### e. Loss Function
- **Link Prediction and Path Importance:**  
  - Use cross-entropy or logistic loss based on predicted likelihood \(p(y|x,q)\).  
  - Negative sampling: generate negative triplets by replacing head or tail entity, calculate their likelihood, and compute average negative loss.  
- **Objective:** Minimize negative log-likelihood plus negative sampling loss (Eq. (11)).

---

## 2. Implementation Strategy

### a. Data Handling & Dataset Preparation
- **Datasets:**  
  - Transductive: FB15k-237, WN18RR (knowledge graphs).  
  - Homogeneous: Cora, Citeseer, PubMed (citation graphs).  
  - Inductive splits: Prepare train/test graphs with disjoint entities (see Tables 3-5).  
- **Data Format:**  
  - Graph as adjacency list/matrix with edges, relation types, and features if needed.  
  - For large graphs, subgraph extraction may be required (e.g., for GNN-incorporated methods).  
- **Negative Sampling:**  
  - For each triplet, generate a fixed number (e.g., 50) of corrupted triplets by replacing head or tail entities not in the graph.

### b. Model Components
- **Neurons:**  
  - Implement custom spiking neuron simulating continuous or discrete membrane potential dynamics.  
  - Support surrogate gradients for backpropagation.  
- **Edge Delays Parameterization:**  
  - Initialize delays using sigmoid neural embedding \(d_{r}^{q}\) with hyperparameters \(\beta, b_{r}\).  
  - Quantize delays into integers for discrete event simulation; use straight-through estimator for gradients during training.  
- **Spike Propagation:**  
  - For each discrete time step (T=10, 20, 50, ...), propagate spikes according to delays.  
  - Track spike trains for each node and each neuron within the node (vectorized if possible).

### c. Training Procedure
- Use mini-batch training, batching triplets and their negatives.  
- For each batch:  
  - Inject constant current \(I^{q}\) into source node neurons (relation embedding).  
  - Propagate spike trains over \(T\) steps.  
  - Decode spike trains to obtain pair representations \(\mathbf{h}^{q}(x,y)\).  
  - Compute likelihood via logistic or multilayer perceptron (MLP) on the decoded spike features.  
  - Calculate loss with positives and negatives, backpropagate gradients.  
  - Update weights, relation embedding parameters, delays, and biases using Adam or SGD.

### d. Hyperparameters & Settings
- **Neuron count \(N\):** 16, 32, 64 based on ablation.  
- **Time steps \(T\):** Evaluate at 10, 20, 50, 100 (see Figures 8-9).  
- **Delay parameters:** \(\beta, b_{r}, \text{scaling factor}\) as per appendix.  
- **Dropout / Regularization:** Use to stabilize training (not explicitly mentioned but generally useful).  
- **Learning Rate:** Start from \(2 \times 10^{-3}\) or similar; tune via validation.  
- **Batch Size:** 32-128, depending on memory.  
- **Number of Epochs:** ~20-50, monitoring validation metric.

### e. Evaluation Metrics
- **Triplet Ranking Metrics:**  
  - Mean Rank (MR): lower is better.  
  - Mean Reciprocal Rank (MRR): higher is better.  
  - Hits@1, Hits@3, Hits@10: higher is better.  
- **Other Metrics:**  
  - AUROC, AP for classification perspective (see Appendix E).  
  - Path interpretability by analyzing spike timings (qualitative).

---

## 3. Key Implementation Details & Practical Tips

- **Discrete delays:** Implement quantization, employ straight-through estimator for backpropagation.  
- **Surrogate gradient:** Use sigmoid derivatives for \(\partial s/\partial u\).  
- **Async propagation:** For scalable code, vectorize spike propagation across neurons and edges, updating membrane potentials at each discrete time step.  
- **Edge delays:** Initialize as learnable parameters; constrain via sigmoid and quantization.  
- **Path encoding:** Maintain spike trains per node and path; decode based on first spike timing, spike counts, or weighted sums.

---

## 4. Experimentation & Ablation Protocols

- **Vary neuron count per node \(N\):** 8, 16, 32, 64 (see Table 10).  
- **Vary time step \(T\):** 10, 20, 50, 100 (see Figure 8).  
- **Delay initialization & scale \(\beta\):** e.g., 4, 8, 16, as in Appendix E.  
- **Relation embedding dimension:** e.g., 16-64, subject to tuning.  
- **Negative samples:** 50 per triplet as standard.  
- **Training epochs:** 20-50 with early stopping based on validation metrics.

### 5. Validation & Results Comparison
- Track metrics (MR, MRR, Hits@k) on validation/test sets.  
- Compare GRSNN with baseline models (Path ranking, GNNs, entity embedding).  
- Perform statistical significance tests if needed (error bars in Tables 11-12).  
- Analyze spike train decoding for interpretability (path importance).

---

## 6. Additional Notes & Unresolved Details
- Exact hyperparameter values (scale factors, thresholds) are provided in appendix sections; tuning may be required.  
- Implementation of the forward and backward passes for delays and spike timing is non-trivial; consider an automatic differentiation framework with custom autograd functions.  
- Graph subgraph extraction and batching pipeline, especially for large datasets, should follow standard practices in knowledge graph link prediction.

---

## Summary Roadmap
- **Data:** Load and preprocess datasets; create negative samples.
- **Model:** Implement spiking neuron dynamics, delay parameterization, spike propagation, decoding functions.
- **Training:** Set hyperparameters, initialize parameters, run epochs with backpropagation.
- **Evaluation:** Compute metrics, compare to baselines, visualize paths.
- **Ablations:** Vary neuron count, delay settings, time steps, etc.  
- **Analysis:** Interpret spike timings, importances, path representations.

---

This roadmap ensures a rigorous, detailed approach to faithfully emulate and reproduce the experiments described in the paper, laying a solid foundation for subsequent code development.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement a modular Python system leveraging PyTorch for automatic differentiation and simulation of spiking neuron dynamics. The core components include dataset loading, a custom spiking neuron layer with delay learning, a graph propagation module with temporal encoding, and an evaluation pipeline. The system will use sparse tensor operations and custom autograd functions to model delays, spike timing, and backward gradients. All modules are organized to facilitate flexible hyperparameter tuning, efficient batching, and straightforward experiment execution.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "spiking_layer.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment(): None\n    }\n    class DatasetLoader {\n        +__init__(dataset_path: str)\n        +load_data() -> dict\n    }\n    class SpikingNeuronLayer {\n        +__init__(num_neurons: int, delay_param_dim: int, other_params: dict)\n        +forward(spike_trains: Tensor, delays: Tensor, weights: Tensor) -> Tensor\n        +backward(grad_output: Tensor) -> dict\n        +get_delays() -> Tensor\n        +update_delays(new_delays: Tensor): None\n    }\n    class GraphPropagationModule {\n        +__init__(relation_embeddings: Tensor, delay_params: Tensor, neuron_layer: SpikingNeuronLayer, config: dict)\n        +inject_source_spikes(source_indices: Tensor, relation_index: int): None\n        +propagate_timestep(): List[Tensor]\n        +decode_spike_train(target_spikes: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: GraphPropagationModule, dataset: dict, config: dict)\n        +train(): None\n        +save_model(filepath: str): None\n    }\n    class Evaluation {\n        +__init__(model: GraphPropagationModule, dataset: dict)\n        +evaluate(): dict\n        +decode_paths(spike_trains: List[Tensor]) -> list\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> GraphPropagationModule\n    GraphPropagationModule --> SpikingNeuronLayer\n    Evaluation --> GraphPropagationModule\n",
    "Program call flow": "sequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant CD as Config\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: load_data()\n    DL-->>M: dataset dict\n    M->>+CD: load config\n    CD-->>M: config params\n    M->>+TR: initialize with model and dataset\n    TR->>TR: train() over epochs\n    TR-->M: model checkpoint saved\n    M->>+EV: load model for eval\n    EV->>EV: run evaluate()\n    EV->>M: output metrics\n    M->>Main: End\n    Note over TR: During training, in each timestep,\n    note right of TR: spikes injected, propagated, and delayed\n    note right of TR: backward passes compute gradients for delays and weights\n    Note over EV: During evaluation, decode spike trains to path scores\n",
    "Anything UNCLEAR": "Clarify dataset input formats: should entity and relation data be preprocessed into adjacency matrices or triplet lists? Also, specifics of the shape and embedding size for relation embeddings and delay parameters, and the exact interface expected by each class for batch processing require confirmation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch==1.13.0",
        "numpy==1.21.0",
        "scipy==1.7.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains class DatasetLoader: responsible for loading datasets, parsing triplets, and creating training/validation/test splits. Implements methods load_data() returning dataset dicts, handles negative sampling. Needs to output data in triplet list format for efficient batch loading."
        ],
        [
            "model.py",
            "Defines class SpikingGraphReasoningModel: encapsulates the core SNN with learnable delays, relation embeddings, and neuronal dynamics. Implements methods for initializing network parameters, forward propagation per discrete timestep, surrogate gradient-based backpropagation, and delay update functions."
        ],
        [
            "trainer.py",
            "Contains class Trainer: manages training loop, batching, loss computation, backpropagation, and optimizer updates. Implements train() method that runs multiple epochs, handles dynamic delay updates, manages checkpoints, and logs training metrics."
        ],
        [
            "evaluation.py",
            "Defines class Evaluator: for running inference on test data, decoding spike trains, computing metrics like MR, MRR, Hits@k, and visualizing paths. Implements evaluate() and decode_spike_trains() methods."
        ],
        [
            "main.py",
            "Entry point script that initializes configuration, loads datasets via DatasetLoader, instantiates the model, runs training via Trainer, performs evaluation via Evaluator, and manages the overall experiment flow. Responsible for orchestrating task dependencies and command sequence."
        ],
        [
            "utils.py",
            "Provides helper functions such as surrogate gradient functions, delay quantization utilities, spike train decoding functions, and plotting utilities for interpretability analysis. Shared by model, trainer, and evaluation modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "utils.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "No API endpoints required; internal function calls used for dataset loading, model training, and evaluation.",
    "Shared Knowledge": "Utilizes the surrogate gradient functions for non-differentiable spike handling, delay quantization methods, and spike train decoding mechanisms. Common hyperparameters like neuron count, delay scale, time steps, and learning rate are centrally defined for easy tuning.",
    "Anything UNCLEAR": "Clarify the precise data format expected from datasets, especially the normalization of relation embeddings, delay ranges, and whether the datasets are preprocessed or require preprocessing within loaders. Also, confirm the network's exact layer/parameter initialization procedures."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  path: "path/to/your/dataset"
  type: "knowledge_graph"  # options: 'knowledge_graph', 'citation'
  batch_size: 32
  negative_samples: 50

model:
  neuron_count_per_node: 32           # Number of neurons assigned to each graph node
  time_steps: 20                      # Discrete time steps for simulation
  delay_scale: 4.0                    # Scale factor for delay initialization
  delay_scale_relation: 4.0           # Scale for delays dependent on relation embedding
  delay_scale_homogeneous: 2.0        # Delay for homogeneous graphs
  delay_scale_inductive: 8.0          # Delay for inductive generalization
  delay_scale_edge_importance: 4.0    # Delay scale for edge importance weighting
  neuron_params:
    threshold: 1.0
    membrane_decay: 0.95
    reset_potential: 0.0

training:
  learning_rate: 0.002               # Learning rate for optimizer (Adam)
  epochs: 20                         # Number of training epochs
  optimizer: "Adam"
  weight_decay: 1e-5
  gradient_clip: 0.5                 # Gradient clipping norm
  dropout_rate: 0.3                  # Dropout probability, if used
  early_stopping_patience: 5         # Early stopping rounds based on validation metric

evaluation:
  metrics: ["MR", "MRR", "Hits@1", "Hits@3", "Hits@10"]
  max_eval_time_steps: 20            # Max timesteps for decoding during evaluation

misc:
  seed: 42                           # Random seed for reproducibility
  device: "cuda"                     # Device: 'cuda' or 'cpu'
```

---

**Note:** Full configuration is available in `planning_config.yaml`
