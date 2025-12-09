# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

{
  "dataset_loader.py": 
  "Class DatasetLoader: responsible for dataset ingestion, preprocessing, and data formatting suitable for training and evaluation.\n\nResponsibilities include:\n1. Initialization with dataset path, type (knowledge graph or citation), and optional configuration parameters.\n2. Loading raw dataset files (triplets, node and relation data) from specified path.\n3. Preprocessing data into structured formats:\n   - For knowledge graphs: triplet list [entity1, relation, entity2], including inverse triplets for training augmentation.\n   - For citation datasets: node features and adjacency matrices.\n4. Creating train, validation, and test splits based on dataset splits (e.g., 85/5/10 ratios) or provided splits.\n5. Implementing negative sampling: for each positive triplet, generate 'negative_samples' number of triplets by replacing head or tail with randomly selected entities, ensuring negative triplets are not in the graph.\n6. Returning datasets as python dictionaries or custom data structures:\n   - For triplet datasets: list of triplets, with labels indicating positive/negative.\n   - For efficient batching during training, store triplets as tensor arrays (entity and relation indices). Also, maintain a mapping from entity/relation name to index.\n7. Handle data normalization: normalize relation embeddings if applicable; confirm if datasets are preprocessed or need additional normalization.\n8. Support methods:\n   - load_data(): returns a dictionary with keys such as 'train', 'val', 'test', each containing structured datasets.\n   - get_batch(): generate batches for training, including negative samples, possibly with shuffling.\n9. If datasets are large, implement memory-efficient loading or subgraph extraction (not indicated explicitly but advisable for scalability). \n\nImplementation details:\n- Parsing raw data: read triplet files, map entities and relations to unique integer IDs.\n- For knowledge graphs: build adjacency lists or matrices for neighbor lookups, possibly pre-build inverse triplets.\n- Negative sampling: ensure negative triplets do not overlap with existing positive triplets.\n- Data output: structured as list or tensors of shape [batch_size, 3], with corresponding labels if applicable.\n- Consider seed for reproducibility in negative sampling and shuffling.\n\nSpecial notes:\n- Relation embeddings may need to be normalized or scaled according to config settings.\n- Dataset loading must be compatible with downstream batching in PyTorch, e.g., conversion to tensors.\n- Ensure methods are flexible to support different dataset types and experiment protocols.\n- Maintain a clean API with methods like load_data() emitting ready-to-use datasets.\n",
  "Anything UNCLEAR": "Clarify whether datasets are expected in a raw text format (e.g., triplet files) or preprocessed into structured files (e.g., pickled, JSON, HDF5). Also, verify if inverse triplets are auto-generated or pre-existing. Confirm the exact format for negative sampling: whether to generate on-the-fly during batch loading or precompute negative samples. Clarify if dataset splits are provided explicitly or need to be created based on ratios. Ensure about entity and relation ID mappings: should the loader build these mappings or assume pre-processed numerical IDs? Lastly, specify if any normalization or feature encoding is required beyond triplet parsing."
}

## evaluation.py

# Evaluation.py Logic Analysis

This evaluation.py module is designed to facilitate inference, metric computation, interpretability analysis, and visualization for the GRSNN-based graph reasoning experiments presented in the paper. It operates in conjunction with trained models, dataset loaders, and decoding functions, relying on the detailed methodology, datasets, and parameters outlined in the overall plan, code structure, and configuration.

---

## 1. Class Overview: Evaluator

**Purpose:**  
- To load trained GRSNN models and datasets, run inference on test trips or graphs, decode spike trains into similarity scores or relation predictions, and evaluate performance by relevant metrics.  
- To support interpretability via visualization and importance analysis of reasoning paths.

**Main Methods:**  
- `__init__()`: Set up evaluation environment, load datasets, model, hyperparameters, etc.  
- `evaluate()`: Perform inference on the test set, decode spike trains, compute evaluation metrics, and return detailed results.  
- `decode_spike_trains()`: Convert spike train outputs into continuous features/representations using the proposed decoding methods (e.g., weighted sum of spike magnitudes).  
- `visualize_paths()`: (Optional) Visualize reasoning paths based on spike importance scores, for interpretability analysis.

---

## 2. Initialization: __init__() Method

- **Inputs:**  
  - `model`: the trained SNN model object implementing the core propagation and decoding logic.  
  - `dataset`: dataset object or dict containing train/validation/test triplets and metadata.  
  - `config`: dictionary of evaluation settings (metrics, max decoding time steps, device).  
- **Goals:**  
  - Load or confirm dataset splits (test triplets).  
  - Transfer model to specified device (GPU/CPU).  
  - Prepare internal state for inference, e.g., cache relation embeddings if needed, load precomputed parameters.  
  - Set up metric trackers (e.g., list containers for per-triplet scores, predictions).

- **Dataset Input Format:**
  - Triplet list: `(head_entity, relation, tail_entity)` for transductive tasks.  
  - For inductive tasks, disjoint entity sets with relation types.  
  - Data may include adjacency structures or node indices; these are used within the model.

---

## 3. Decoding Spike Trains: decode_spike_trains()

- **Input:**  
  - `spike_trains`: List or tensor of spike train(s) associated with target node(s) after propagation, corresponding to a triplet `(x, q, y)` or candidate triplet `(x, q, y')`.  
- **Decoding methods based on paper:**
  - **First Spike Timing:**  
    - Find the earliest spike time—representing shortest path or most relevant path.  
    - Can be implemented as: `t_first = min(spike_times)` for each neuron or the neuron with the earliest spike.  
  - **Weighted Summation:**  
    - Compute a weighted sum across spike magnitudes with exponential decay:  
      \[
      D(\mathbf{s}) = \frac{\sum_\tau \lambda^\tau \cdot \mathbf{s}[\tau]}{\sum_\tau \lambda^\tau}
      \]  
    - Here, \(\lambda\) is the decay factor from config, and \(\mathbf{s}[\tau]\) is the spike count at time \(\tau\).  
    - This yields a vector feature corresponding to path relevance.  
  - **Path Diversity / Multiple Path Encoding:**  
    - Concatenate or aggregate spike counts over neurons and time, forming a feature vector (e.g., spike counts per neuron over time).  
- **Output:**  
  - Continuous feature vectors or scalar scores representing the pair or triplet similarity.

- **Note:**  
  - In implementation, ensure that the spike train data structure supports efficient minimum search and summation.  
  - Handle cases with no spikes (absence of activation)—e.g., assign default high or low scores as per the metric.

---

## 4. Metrics Computation: evaluate()

- **Inputs:**  
  - `test_triplets`: List of triplets `(x, q, y)` (and negatives).  
  - `model`: model inference method; for each triplet, run the propagation and decode spike trains.  
  - **Batching:**  
    - For efficiency, process triplets in batches if possible.  
    - For each triplet:  
      - Inject source spike train (source entity + relation embedding).  
      - Run `T` discrete timesteps of propagation.  
      - Decode spike trains at sink node `(y)` to get a feature vector.  
      - Compute likelihood score via `g` (MLP or linear layer) as per Appendix E.1.  
- **Performance Metrics:**
  - **MR (Mean Rank):**  
    - For each triplet, rank the true tail entity against all others based on likelihood scores.  
    - Aggregate the average inverse of rank.  
  - **MRR (Mean Reciprocal Rank):**  
    - Average reciprocal of the rank for each triplet.  
  - **Hits@N:**  
    - Calculate the proportion of true tail entities ranked in top N.  
  - **Additional:**  
    - For homogeneous graphs: compute AUROC, AP using predicted scores and ground truth edges.

- **Ranking Procedure:**
  - For each triplet `(x, q, y)` and candidate tails `\(\{ y' \}\)` (including negatives):  
    - Compute scores concurrently to speed up.  
    - Rank the entities by scores.  
- **Output:**  
  - Dictionary with metrics: MR, MRR, Hits@1, Hits@3, Hits@10, AUROC, AP.

---

## 5. Path Importance & Interpretability: visualize_paths()

- **Purpose:**  
  - To interpret the reasoning paths by analyzing edge importance scores derived from gradients.  
  - To identify high-importance paths using beam search strategies as per Appendix F.3 and F.2.  
- **Implementation considerations:**  
  - Use gradient-based edge importance:  
    \[
    \frac{\partial \text{prediction}}{\partial \text{edge\_parameter}} \approx \text{edge importance}
    \]  
    - Derived during backpropagation of the prediction score with respect to edge weights or spike contributions.  
  - Use beam search to select top paths based on cumulative importance.  
- **Visualization:**  
  - Map top paths to graph topology, annotate edge relevance by importance scores.  
  - Show sample reasoning paths with their importance scores, as exemplified in Table 9.

---

## 6. Additional Considerations

- **Handling Variable Time Steps:**  
  - During evaluation, adopt the maximum time step specified (`max_eval_time_steps`).  
  - Ensure that decoding aligns temporally—only consider spike train slices up to that step.  
- **Reproducibility & Robustness:**  
  - Set seed within evaluation for deterministic behavior if necessary.  
  - Provide error metrics or variance if multiple runs are conducted (see error bars in Table 12).  
- **Parallelization & Performance:**  
  - Leverage batch processing on GPU.  
  - For large graphs, consider subgraph extraction or batching heuristic techniques.

---

## 7. Summary of Key Inputs & Outputs

| Method | Inputs | Outputs | Notes |
|---------|---------|---------|--------|
| `__init__()` | Model, dataset, config | Internal setup | Preprocessing, device transfer |
| `evaluate()` | Test triplets | Metrics (MR, MRR, Hits@N, AUROC, AP) | Propagation, decoding, ranking |
| `decode_spike_trains()` | Spike trains | Continuous features, scores | Temporal sum, first spike, etc. |
| `visualize_paths()` | Spike importance data | Graph path visualization | Gradient-based importance |

---

**In summary:**  
The evaluation.py module implements the inference pipeline by propagating spike trains over the graph, decoding these spikes into features or scores, ranking entities for link prediction, and computing performance metrics as defined in the paper. It allows interpretability through edge importance analysis, aligning with the paper's methodology for path explanations. All implementations must operate seamlessly with the trained GRSNN model, leveraging the specified decoding functions and hyperparameters, ensuring reproducibility and consistency with the experimental setup.

## main.py

# Logic Analysis for main.py

This script serves as the primary orchestrator to execute the entire graph reasoning experiment utilizing the GRSNN model. It must be designed to:

- Initialize and parse the experimental configuration from `config.yaml`.
- Set random seed and device for reproducibility.
- Load datasets through DatasetLoader, supporting knowledge graphs or citation graphs as specified.
- Instantiate the Graph Reasoning SNN model with parameters from configuration.
- Instantiate the Trainer for training, including batching, loss, gradient calculations, and parameter updates.
- Run the training loop over specified epochs, including early stopping if configured.
- Save trained model checkpoints if needed.
- Instantiate the Evaluator for testing, decoding spike trains, and computing relevant metrics.
- Finally, output performance metrics and, optionally, interpretability visualizations for reasoning paths.

---

### 1. Initialization & Configuration

- **Load Configuration:**  
  Use a `yaml` parser (`yaml.safe_load`) to load parameters from `config.yaml`. Store the parameters into variables for easy access.

- **Set Reproducibility:**  
  - Set random seed (`torch.manual_seed`, `np.random.seed`) based on `misc.seed`.
  - Configure device (`cuda` if available and specified, else cpu).

- **Device Context:**  
  - Use `torch.device()` for device assignment.
  - Ensure that all tensors, models, and data are transferred to the correct device.

### 2. Dataset Loading

- **Instantiate DatasetLoader:**  
  - Pass dataset path, dataset type (`knowledge_graph`, `citation`), batch size, negative samples, and other dataset-specific parameters.
  - Call `load_data()` which returns a dataset dictionary, including:
    - Triplet lists for training, validation, test sets.
    - Entity and relation information, possibly in triplet list format.
    - Support for datasets that require splitting or postprocessing.
- **Data Preprocessing:**  
  - Ensure datasets are in triplet format.
  - For large graphs, consider batching and subgraph extraction if applicable.
  - Prepare negative samples as per training protocol.

### 3. Model Instantiation

- **Initialize Graph Reasoning SNN (GRSNN):**  
  - Pass key model parameters:
    - `neuron_count_per_node` (32 default).
    - `time_steps` (20 default).
    - Delay-related parameters: `delay_scale`, `delay_scale_relation`, etc.
    - Neuron parameters: thresholds, decay constants, reset potentials.
  - Include relation embeddings (initialization – random or pre-trained).
- **Parameter Initialization:**  
  - Weights, delays, relation embeddings should be initialized as per model specifications, possibly with Xavier or uniform initialization.
  - Delays are learnable, constrained via sigmoid and quantized techniques.
- **Assign model to device (GPU/CPU).**

### 4. Training Setup

- **Optimizer & Loss Function:**  
  - Instantiate Adam optimizer, with learning rate, weight decay, and clip norm.
  - Use surrogate gradient functions for spikes.
- **Training Loop:**  
  - For each epoch:
    - Shuffle dataset/triplets for stochasticity.
    - For each batch:
      - Reset neuronal states if needed.
      - Inject constant current \(I^{q}\) into source node neurons for the current triplet and relation \(q\).
      - Run the spike propagation over `time_steps`.  
        - At each timestep:
          - Update membrane potentials, apply leakage.
          - Propagate spikes with delays. Use buffer structures for delays.
          - Record spike trains per neuron/node.
      - After propagation:
        - Decode spike trains to get pair representation \(\mathbf{h}^{q}(x,y)\).  
          - Use early spike time, spike count, or weighted sum as described.
      - Compute likelihoods via the MLP head with relation embedding.
      - Calculate the loss with positive and negative triplets.
      - Perform backpropagation using surrogate gradients.
        - Gradients flow to delays, weights, relation embeddings, neuron parameters.
        - Use `torch.autograd` with custom autograd functions if implementing delay quantization.
      - Optimizer step.
    - Monitor validation metrics (MR, MRR, Hits@k).
    - Implement early stopping based on validation performance.

### 5. Evaluation & Path Visualization

- **Post-Training Evaluation:**  
  - Load the best checkpoint based on validation metrics.
  - For test triplets, repeat spike propagation process:
    - Inject source currents.
    - Propagate spikes for `max_eval_time_steps`.
    - Decode spike trains for pairs.
  - Use the trained head to compute likelihood scores.
  - Rank triplets to derive MR, MRR, Hits@k.

- **Interpretability:**  
  - Use gradient-based importance measures to identify key edges and paths.
  - Optionally, visualize reasoned paths based on spike timings and gradient contribution.

### 6. Results Output & Saving

- Output quantitative results (metrics) to stdout or save in a file.
- Save the trained model state_dict for reproducibility.
- Save any interpretability visualizations if performed.

### 7. Additional Considerations

- **Error Handling:**  
  - Wrap dataset loading, training, and evaluation in try-except blocks.
- **Logging:**  
  - Use logging or print statements for progress updates, hyperparameter info, and final results.
- **Hyperparameter Management:**  
  - Ensure hyperparameters from `config.yaml` are consistently used across modules.
- **Reproducibility:**  
  - Save seed, dataset splits, and hyperparameters for reproducibility.
- **Decouple Components:**  
  - Keep dataset loading, model instantiation, training, and evaluation separate to facilitate debugging and modularity.

---

### Summary

The `main.py` must follow this structure:

- Load config and set device.
- Load datasets.
- Instantiate the SNN model with learnable delays and relation embeddings.
- Set up optimizer and loss.
- Run training epochs, handling batching, spike propagation, decoding, gradient computation.
- Select best model via early stopping.
- Load best model and evaluate on test data.
- Collect and print metrics, plus optional interpretability visualizations.

This process provides detailed, reproducible, and scalable execution aligned with the paper's methodology, the design, and the configuration provided.

## model.py

{
  "dataset_structure": "Input data to the model includes triplets (head, relation, tail) along with their entity and relation indices. The dataset loader is responsible for creating batches consisting of positive triplets and negative samples, formatted likely as tensors of shape (batch_size, 3) for triplets. The entity and relation indices are mapped via embedding tables.",
  "core_model_object": "The class SpikingGraphReasoningModel will construct the core neural network architecture. It will include parameters for entity embeddings, relation embeddings, delay parameters, and the neuronal dynamics encapsulating spiking behavior. It will support methods for initialization, forward pass over a discrete number of time steps, and update functions for learnable delays.",
  "entity_embeddings": "Create an embedding matrix for entities, shape (num_entities, neuron_count_per_node). Similarly, relation embeddings of shape (num_relations, embedding_dim). These embeddings are learnable parameters initialized typically randomly or following the description in the appendix.",
  "relation_embeddings": "Relation embeddings will be used to generate input currents for source nodes and to modulate delays. They should be normalized or scaled as per the configuration. The relation embedding can be a vector (e.g., size 16-64), possibly scaled and mapped via linear layers.",
  "delay_parameters": "Each edge type (relation) or edge instance (in the case of heterogeneous graphs) will have associated learnable delay parameters. These will be parameterized via sigmoid functions, scaled by delay_scale, and quantized to integers for discrete simulation. For Generalization, include functions to produce delay from relation embeddings using parameters like W_r and b_r, applying sigmoid and scaling, with optional binarization or straight-through estimators.",
  "delay_function": "Implement a function that computes delays from relation embeddings, including sigmoid transformation, scaling, quantization, and possibly learnable biases. Delays should be stored in a tensor of shape (number_of_edges, 1), aligned with edge relations or instances.",
  "spiking_neurons_layer": "Implement a custom module for neurons with membrane potential dynamics, following equations (7) or the SRM approximation. It should simulate over T discrete steps, updating membrane potential, injecting currents, and generating spikes based on thresholds. The neuron layer should support surrogate gradients for the non-differentiable spiking operation.",
  "spike_propagation": "In the forward pass, at each timestep, for each node and its connected neighbors, propagate spikes considering the delay-induced arrival times. The input current for each neuron is a sum over all incoming spikes weighted by the synaptic weights and delayed by their respective delays (as integer indices). The dynamics follow the exponential decay equations with spike reset after threshold crossing.",
  "forward_method": "The main forward method will simulate T discrete steps. At each step: \n- Inject source current into node neurons based on relation embedding.\n- Aggregate incoming spikes with delay and weight considerations.\n- Update neuron membrane potentials using exponential decay and input current.\n- Generate spikes when membrane potential exceeds threshold.\n- Store spike trains for decoding.\nAfter T steps, output the spike trains for each node's neurons. These spike trains will be used for decoding the pair representation.",
  "decoding_spike_trains": "Implement a decoding function that takes spike trains of the target node's neurons, computes a weighted sum over time (exponential decay factor \(\lambda^{\tau}\)), or uses the time of the first spike, to generate the pair representation \(\mathbf{h}^{q}(x, y)\). This representation can be a vector (e.g., spike counts or weighted sums).",
  "loss_and_training": "The loss is the negative log likelihood based on the output of a predictor network applied to \(\mathbf{h}^{q}(x,y)\). The predictor could be a simple MLP. Surrogate derivatives are used for \(\partial s / \partial u\), often sigmoid-based. During backpropagation, the gradients flow through the temporal simulation, delay parameters, and entity/relation embeddings. Constraint functions for delays should support straight-through estimators for gradient flow.",
  "update_delays": "Implement a method to update delay parameters, converting sigmoid outputs to integer delays, possibly via straight-through estimator during backpropagation. The learned delays correspond to the edge properties in the graph, influencing spike timing during propagation.",
  "hyperparameters": "The neuron parameters such as threshold, membrane decay, and reset potential are initialized from configuration. The total number of neurons per node, number of time steps, delay scale, and relation embedding dimensions are dynamically set from the config. These parameters influence simulation accuracy and computational complexity.",
  "initialization": "Randomly initialize entity embeddings, relation embeddings, and delay parameters. Convert delay parameters into scaled sigmoid outputs, then quantize to integers for the simulation. Initialize neuron states (membrane potentials, currents) at rest (zero).",
  "training dynamics": "In each batch, process multiple triplets simultaneously. Inject relation-specific current at the start (relation embedding transformed to current). Propagate spikes for T steps, capturing the spike trains of the target nodes. Compute the pair representation, then loss, and backpropagate. Use optimizer step (Adam). Update delay parameters as part of learnable parameters, applying the straight-through estimator for delay quantization.",
  "model outputs": "The main output of forward() is the spike trains at the target node, which are decoded into the pair representation. Also, store the spike trains for interpretability or visualization if needed.",
  "additional considerations": "Implement clipping/clamping for delay parameters if needed. Encapsulate all simulation steps within a single forward() call for efficiency. Modularize components: dependency injection for neuron dynamics, delay functions, and decoding methods to facilitate experiments."
}

## trainer.py

# Logic Analysis for `trainer.py`

The `trainer.py` module is responsible for the high-level training loop, including data batching, spike-based forward pass, loss computation, backpropagation (including delay and weight updates), and evaluation. The core class `Trainer` interfaces with data loaders, models, and evaluators to structure the training process faithfully to the paper's methodology.

---

## 1. Responsibilities and Objectives

- Initialize and manage the training process.
- Perform forward propagation of spike trains over discrete time steps.
- Compute a loss based on predicted triplet likelihoods during training, incorporating positive and negative samples.
- Employ surrogate gradient-based backpropagation through time for parameters: synaptic weights \(w_{ij}\), delays \(d_{ij}^q\), relation embeddings \(r^q\), and neuronal parameters.
- During training, update the parameters using an optimizer (Adam) with gradient clipping, early stopping, and learning rate schedule if applicable.
- Persist model checkpoints for best validation performance.
- Log performance metrics for monitoring.
- Execute evaluation upon training completion, decoding spike trains and computing metrics like MR, MRR, Hits@k.

---

## 2. Inputs and Data Handling

- **Data Batches:**  
  - Each batch contains triplets: `(x, q, y)`, with associated negative samples `(x', q, y')`.  
  - Each triplet is mapped to node indices (`x`, `y`) and relation type `q`.  
  - For large datasets, batching should be efficient, potentially sampling negative triples dynamically or pre-generating negatives.
- **Data Pipeline Assumption:**  
  - Datasets are loaded in `dataset_loader.py` and returned as dictionaries: containing triplet lists and auxiliary data structures.  
  - Negative samples are generated within the batch for each triplet (as per training method in Appendix E.1).

---

## 3. Forward Propagation Steps (within each epoch/batch)

- **Source Injection:**  
  - For each triplet `(x, q, y)`, inject a constant current `I^q` into the `neuron_count_per_node` neurons of node `x`.  
  - The injection is based on the relation embedding `r^q`, projected via a parameterized linear transformation to `I^q`.

- **Spike Propagation Over Discrete Time:**  
  - For each of `T` time steps (from `1` to `model.config.model.time_steps`):  
    - Update neuron membrane potentials using the SRM equations or discrete LIF equations.  
    - For neurons receiving input from neighbors:  
      - Sum their spike contributions delayed by `d_{ij}^q`.  
      - The delays are integers derived from the learned or initialized delay parameters, using tanh/sigmoid transformation scaled by `delay_scale` and quantization via straight-through estimator.
    - Compute spikes using the threshold condition and reset mechanism.
    - Record spike trains (binary or weighted signals per neuron).

- **Propagation of Spike Trains:**  
  - Spike trains are propagated along edges, modulated by learned delays and weights, to compute post-synaptic currents and membrane potential updates.  
  - This process is recurrent for each discrete time step, accommodating the delays, to mimic the path accumulation described in the paper.

- **Decoding Spike Trains:**  
  - After `T` steps, obtain the spike train `s_y^q(t)` for the target node `y`.  
  - Decode via the specified decoding function:
    - *Early spike time:* find the first spike for each neuron, or  
    - *Weighted sum:* apply \(\lambda^\tau\) exponential decay weights on spikes, or  
    - *Spike count vector:* sum over `τ`.
  - Generate a path representation vector \(\mathbf{h}^{q}(x, y)\).

- **Likelihood and Prediction:**  
  - Feed the decoded pair representation into the classifier `g` (MLP), which takes as input the representation and possibly the relation embedding `r^q`.  
  - Obtain the triplet likelihood score `p(y|x,q)` by applying a sigmoid to `g`.

---

## 4. Loss Function and Optimization

- **Binary Cross-Entropy Loss:**  
  - For positive triplet `(x, q, y)`, target label = 1; for negatives, label = 0.  
  - Loss: `-log(p(y|x,q)) + sum over negatives of -log(1 - p(x', q, y'))`.

- **Backpropagation via Surrogate Gradients:**  
  - Gradients flow through spike trains, membrane potentials, weights, and delays.  
  - Use surrogate gradient for spikes: derivative of sigmoid function as in Eq. (13).  
  - For delays \(d_{ij}^q\), employ straight-through estimator: quantize delays during forward, treat gradient as if continuous during backward.  
  - Calculate gradients for all parameters:  
    - **Weights \(w_{ij}\):** from chain rule, sum over all time steps and neurons.  
    - **Delays \(d_{ij}^q\):** derived from the trace of their influence on spike timing, involve gradient of \(\kappa(\tau - d_{ij})\).  
    - **Relation embeddings \(r^q\):** update via chain rule through relation-dependent delay and current injection.

- **Gradient Clipping:**  
  - Clip gradients to `training.gradient_clip` norm to ensure stable training.

- **Parameter Updates:**  
  - Use Adam optimizer with specified learning rate, weight decay.  
  - Update parameters with accumulated gradients after each batch.

---

## 5. Managing Delays

- Delays are learned as continuous parameters during training, quantized into integers for simulation.  
- Delay parameters are constrained using the sigmoid function scaled by `delay_scale` (or other specified).  
- During backward pass: treat delays as real-valued with straight-through estimator for gradient flow, allowing effective learning.

## 6. Model Checkpointing and Early Stopping

- After each epoch, evaluate on validation set if specified or keep track of training metrics.  
- Save model parameters when performance improves on validation with the target metric (e.g., MRR).  
- Implement early stopping patience based on validation performance to prevent overfitting, as configured.

---

## 7. Evaluation Phase

- **Inference:**  
  - For each triplet `(x, q, y)`: inject current into source node, propagate `T` discrete steps.  
  - Obtain spike trains at target node `y`, decode using the chosen method.  
  - Compute likelihood score through the downstream classifier `g`.  
- **Ranking:**  
  - Perform filtered ranking against negative samples.  
  - Compute metrics: MR, MRR, Hits@1/3/10.

- **Interpretability:**  
  - Optionally visualize reasoning paths via gradient importance or path importance calculations per Appendix F.2.

---

## 8. Implementation Notes and Best Practices

- Maintain modular components:  
  - Dataset loader handles batching, negative sampling.  
  - Model encapsulates neuron dynamics, delay parameters, spike propagation.  
  - Trainer orchestrates the loop, logging, checkpointing.  
  - Evaluation conducts thresholding, decoding, and metric calculation.
- Use `torch.autograd.Function` for custom delay-based delay propagation and surrogate gradients if custom backward passes are needed.
- For large graphs, optimize sparse matrix operations and avoid redundant computations.
- Carefully handle the calibration of delay scales, and ensure consistent quantization during forward and backward passes.

---

## Summary

The `trainer.py` module will:

- Initialize the model, data, optimizer.
- Loop over epochs, batches:
  - Prepare inputs, inject current.
  - Run discrete-time spike propagation with delays.
  - Decode spike trains.
  - Compute loss with positives and negatives.
  - Backpropagate surrogate gradients, including delay parameters using straight-through estimators.
  - Update parameters.
- Validate periodically, save best models.
- Perform final testing with spike decoding, report metrics.

This detailed logic ensures faithful replication of the experimental methodology, respecting all the nuances of the paper’s approach on temporal SNNs with synaptic delay for graph reasoning.

## utils.py

{
  "utils.py": [
    {
      "Function/Component": "surrogate_gradient",
      "Purpose": "Implement the surrogate gradient function for non-differentiable spike functions, specifically using the sigmoid derivative to approximate the gradient of a spike activation.",
      "Details": "Return a smooth approximation of ∂s/∂u during backpropagation to enable gradient-based learning in the spiking neural network. Use the sigmoid derivative formula: (1/a1) * exp((V_th - u) / a1) / (1 + exp((V_th - u) / a1))^2, with a1 ~ 0.25 for stability.",
      "Inputs": ["u (Tensor): Membrane potential tensor, shape matching neurons or spike trains."],
      "Outputs": ["grad_s (Tensor): Surrogate gradient tensor matching the shape of u."]
    },
    {
      "Function/Component": "delay_quantize",
      "Purpose": "Quantize the real-valued delay parameters into integer discrete delays suitable for simulation and hardware implementation, employing straight-through estimator for backpropagation.",
      "Details": "Scale delays by a scale factor (e.g., delay_scale or delay_scale_relation), apply sigmoid to bound delays, multiply by max delay, and round to nearest integer. During backprop, use straight-through estimator to pass gradients unaltered through the quantization process.",
      "Inputs": ["d (Tensor): Delay parameters before quantization, shape: number of relations or edges."],
      "Outputs": ["d_int (Tensor): Quantized delay tensor with shape matching d."]
    },
    {
      "Function/Component": "decode_spike_train_first_spike",
      "Purpose": "Decode the spike train at a target node to estimate the shortest path or first spike time, which encodes path length information.",
      "Details": "For each neuron or the combined spike train, find the earliest time point at which a spike occurs. Return tensor of shape: batch size or number of nodes, containing the first spike time for each entity.",
      "Inputs": ["spike_train (Tensor): Spike train over T time steps, shape: (num_neurons, T) or batch size x neurons x T."],
      "Outputs": ["first_spike_times (Tensor): Earliest spike times, shape: matching number of entities."]
    },
    {
      "Function/Component": "decode_spike_train_weighted_sum",
      "Purpose": "Decode spike trains into a vector representation by applying exponential weighting over spike counts, emulating relation importance and relevance encoding.",
      "Details": "Sum spikes across neurons and time steps, weight with λ^τ (decay factor), and normalize over total weights. Used for relation-sensitive path encoding as in Eq. (B.2).",
      "Inputs": ["spike_train (Tensor): Shape: neurons x T, representing spike occurrences."],
      "Outputs": ["vector_rep (Tensor): Decoded vector for pair representation."]
    },
    {
      "Function/Component": "plot_paths_importance",
      "Purpose": "Visualize illustrative reasoning paths based on edge importance scores derived during backpropagation through spike trains, aiding interpretability.",
      "Details": "Accept edge importance scores, path sequences, and path importance weights. Generate plots (e.g., network graphs or path sequences) highlighting relevant paths and their contributions. Useful for path importance analysis in Appendix F.2.",
      "Inputs": ["edges (list or array): Edge sequence or graph structure.", "edge_importances (Tensor): Importance scores per edge.", "paths (list): Extracted paths for a given query."],
      "Outputs": ["None"]: Produces static plots or interactive visualizations for interpretability analysis."]
    },
    {
      "Function/Component": "initialize_relation_embeddings",
      "Purpose": "Create and initialize relation embedding vectors used for current injection and delay parameterization, according to configuration.",
      "Details": "Use uniform or Xavier initialization for relation embeddings, with dimension consistent with dataset relation count. These embeddings serve as input for delay and current calculation.",
      "Inputs": ["num_relations (int): Total number of relation types, from dataset."],
      "Outputs": ["relation_embeddings (Tensor): shape: num_relations x embedding_dim."]
    },
    {
      "Function/Component": "update_delays_from_params",
      "Purpose": "Apply the learned parameters (relation, edge importance) to compute the final delays, including sigmoid transformation and scaling, ensuring delay constraints are satisfied, typically in the range [0, delay_scale].",
      "Details": "Given delay parameters (e.g., delay_scale_relation * sigmoid(W_r * r^q + b_r)), compute continuous delay values, then quantize via delay_quantize().",
      "Inputs": ["relation_relation_params (Tensor): Learnable parameters for delay, shape: relation_count x param_dim."],
      "Outputs": ["final_delays (Tensor): Discrete delays for each edge or edge-relation pair."]
    },
    {
      "Function/Component": "spike_train_to_path_importance",
      "Purpose": "Estimate the importance of each path based on spike timing and importance weights used in decoding, especially for interpretability and analysis.",
      "Details": "Aggregate spike arrival times and importance scores (e.g., via gradient importance), sum contributions across paths, possibly threshold to identify most relevant paths as in beam search.",
      "Inputs": ["spike_trains (list): Spike trains at target nodes for multiple paths.", "path_weights (list): Importance scores for the paths."],
      "Outputs": ["path_importances (list): Importance value per path, for visualization."]
    }
  ],
  "Remarks": "Ensure all functions support batch processing for efficiency. Use torch.autograd.Function subclasses to implement custom backward passes where needed, especially for delay quantization and surrogate gradients. Maintain clarity and modularity to facilitate integration with the main training, modeling, and evaluation pipelines."
}

