# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here's a **comprehensive, step-by-step plan** to reproduce the methodology and experiments from the paper "End-to-End Ontology Learning with Large Language Models." This plan meticulously covers the key technical details, dataset assumptions, model training procedures, hyperparameter tuning, and evaluation strategies, based solely on the paper content.

---

# Comprehensive Reproduction Plan for "End-to-End Ontology Learning with Large Language Models"

## I. Overall Approach
- Develop an end-to-end ontology learning pipeline utilizing large language models (LLMs).
- The core pipeline involves:
  - **Pretraining/finetuning** an LLM (specifically, a Mistral 7B-based model) for ontology subgraph modeling.
  - **Generating subgraphs** for individual documents with prompts.
  - **Summing and pruning** subgraphs to produce the final ontology graph.
  - **Post-processing** with edge/node filtering based on hyperparameters.
- Evaluation using newly introduced graph similarity metrics (Graph F1, Motif Distance, etc.), enabling comparison across generated and ground truth ontologies.

---

## II. Data Requirements & Preparation
### 1. Datasets
- **Primary datasets:**
  - **Wikipedia:** Large corpus (~1.4M concepts and ~2.7M taxonomic relations, ~3.6M documents). For reproducing main experiments:
    - Extract concepts, relations, and documents in the form needed.
    - Use the Wikipedia API or existing dumps (e.g., from Wikifile) to collect document texts and associated concepts.
  - **arXiv:** (~161k concepts, ~1.2M taxonomic relations, ~3.6M documents)
    - Extract arXiv article texts, concepts, and possibly the original ontology for ground-truth comparison.
- **Additional datasets (if used):**
  - Name and gather datasets mentioned (e.g., WordNet for relation similarity thresholds).

### 2. Annotations for Training
- For Wikipedia:
  - Generate concept annotations per document (can be performed via keyword matching, existing labels, or external annotations consistent with the paper's approach).
- For arXiv:
  - Use a small set (~2048 document-concept subgraph pairs) for transfer finetuning, as done in the paper.
- Use the same concept and relation sets as described (e.g., only concepts and taxonomic relations).

### 3. Ground Truth Ontologies
- For evaluation purposes:
  - Use the ground truth ontologies provided in the datasets.
  - Convert full ontologies into subgraphs centered on selected topics for qualitative visualization and graph similarity metrics.

## III. Model & Training
### 1. Model Architecture
- Use **Mistral 7B v0.2** as the base large language model.
- Incorporate **Low-Rank Adaptation (LoRA)** modules:
  - Fine-tune LoRA parameters with rank = 32, as per paper.
  - Initialize with pretraining weights (not from scratch).

### 2. Fine-tuning Procedure
- **Objective:**
  - Finetune the model to generate subgraphs representing document-specific concept-taxonomy relations.
  - Use a masked loss to reduce overfitting on high-frequency relations.
- **Training Data:**
  - For Wikipedia:
    - Use annotations (concept-path pairs) for supervised finetuning.
  - For arXiv:
    - Use a small set (~2048 document-concept subgraph pairs); possibly derive these via concept extraction heuristic.
- **Loss Function & Masking:**
  - Implement the custom regularizer:
    - During training, mask the target tokens of relations/concepts with probability proportional to \( \max(1 - M / n, 0) \), where \(n\) is relation frequency.
    - Set \(M\) (average relation occurrences) based on dataset analysis (~see hyperparameters below).
- **Training parameters:**
  - Epochs: 2 epochs for Wikipedia, early stopping on validation.
  - Optimizer: Adam.
  - Learning Rate: As per hyperparameter sweep, starting around \(1 \times 10^{-5}\).
  - Batch size: 16 document examples per batch.
  - Gradient clipping or other stability methods as needed.
- **Hyperparameters tuning:**
  - Use grid search or Bayesian optimization over:
    - \( \alpha \) (relation pruning threshold)
    - \( \beta \) (relation relation pruning threshold)
    - Masking parameter \(M\)
- **Transfer learning:**
  - Finetune the model further on 2048 document-concept pairs from arXiv.

### 3. Prompt Design
- Use the provided templates:
  - **Chain-of-thought prompt:** to generate concept relation paths.
  - **Direct prompt:** for concept classification without chain-of-thought.
- For zero-shot inference: use instruction-tuned models with prompts similar to Figures 6–8.
- For end-to-end modeling: generate subgraphs for individual documents with prompts (see section II of the paper).

## IV. Inference Pipeline
### 1. Subgraph Generation per Document
- For each document:
  - Construct input prompt:
    - Use the chain-of-thought prompt or direct prompt as per the experiment.
    - Prompt includes document text, concept list, or concepts to generate relations.
  - Generate multiple samples (Top-p sampling, temperature 0.1) to account for variance.
  - Use the trained / instruction-tuned model.
- Parse the generated text:
  - Extract concept-to-concept relations, relations types, and relation paths.

### 2. Summation & Pruning
- Sum the subgraphs:
  - Aggregate all relation edges across document subgraphs.
- Prune edges:
  - Use hyperparameters (\(\alpha, \beta\)) to remove low-confidence edges based on edge scores.
  - Also, remove:
    - Self-loops.
    - Inverse edges (bidirectionality enforcement).
- Generate the final ontology graph:
  - Create nodes for concepts.
  - Create directed edges for relations.
  - Use edge weights (frequency or confidence scores).

## V. Post-component Processing & Graph Construction
- Convert the aggregated/subgraph edges into a graph object.
- Apply filtering:
  - Threshold edges by \(\alpha, \beta\) as sweeped via grid search.
  - Remove edges with weight below the threshold.
- Cyclic edge removal:
  - Approximate cycle removal by iterative, greedy removal of smallest edges in cycles.
- Compute the **final graph**:
  - Nodes: concepts.
  - Edges: directed, weighted, with type labels if inferred.
- **Optional:** Re-embedding nodes for visualization or similarity metrics.

## VI. Evaluation
### 1. Metrics
- **Graph F1 (node-based and edge-based):**
  - Generate embeddings:
    - Use pretrained sentence transformers (\( \text{MiniLM-L6-v2} \)) for concept texts.
    - Calculate cosine similarity between ground-truth and generated nodes.
  - Highest scoring node matching (Hungarian algorithm) for node match.
  - Edge matching based on similarity scores.
- **Motif Distance:**
  - Count 3-node subgraphs in both ground-truth and generated graphs.
  - Compute total variation distance between distributions.
- **Semantic Similarity:**
  - Use deep sentence embeddings for concepts to measure semantic correctness.
- **Qualitative visualization:**
  - Generate node-edge graphs centered on multiple topics (see Figures in Appendix).
- **Additional metrics:**
  - Use the existing Edge F1, Node F1, and the newly introduced metrics as per the paper.
  
### 2. Visualization
- Use graph visualization tools (e.g., NetworkX + Graphviz or Gephi):
  - Color edges by similarity score.
  - Visualize subgraphs centered on topics.
  - Visualize matching edges/nodes with opacity proportional to similarity.

---

## VII. Hyperparameters & Implementation Details (to replicate)
| Parameter | Description | Suggested Value / Sweep | Notes |
|-------------|--------------|-------------------------|-------|
| LoRA rank | LoRA module rank | 32 | As per paper for fine-tuning |
| Batch size | Number of documents per training iteration | 16 | Adjust based on GPU memory |
| Epochs | Fine-tuning epochs | 2 | Or early stop based on validation |
| Masked loss parameter \(M\) | Relation occurrence smoothing parameter | ~ dataset-dependent | Tune via grid search |
| Learning rate | Adam optimizer LR | \(1 \times 10^{-5}\) or as tuned | Sweep around initial value |
| Prompt temperature | Sampling temperature | 0.1 | For deterministic output |
| Top-p \( p \) | Nucleus sampling | 0.9 | Consistent across models |
| Relation path length \(N\) | Subgraph path length | 3 (arXiv), 4 (Wikipedia) | Dataset-dependent but fixed during experiment |
| Relation pruning thresholds \(\alpha, \beta\) | Post-processing thresholds | Sweep over geomspace (e.g., 0.1 to 1) | Maximize Graph F1 on validation |
| Graph similarity thresholds | For graph matching | Use thresholds based on cosine similarity (e.g., median 0.436) | For fuzzy matching |

---

## VIII. Implementation Caveats & Open Questions
- **Details about concept annotations and relation extraction heuristics**:
  - The paper mentions using concept/path annotations and relation extraction heuristics (see Appendix A for further clarification). You may need to develop or adapt your own concept annotation pipeline.
- **Ground-truth ontology format**:
  - Must be converted into concept-path graphs for visualization and metric calculations.
- **Cycle detection & removal**:
  - Implement cycle analysis, then greedy removal by lowest-weight edges.
- **Hyperparameter tuning**:
  - Use validation set and grid search over \(\alpha, \beta, M\), or similar.
- **Document tokenization & prompt formatting**:
  - Use T5 or GPT-friendly prompt templates (Figures 6-8). The described chain-of-thought and direct prompts guide the generation process.
- **Access & reproducibility**:
  - Given the datasets are large and proprietary (e.g., Wikipedia dump), plan to use publicly available datasets or prior datasets disclosed in the appendix.

---

# Final Remarks
This plan ensures:
- ** Fidelity to the paper's methodology** (model architecture, custom loss, prompting, post-processing).
- ** Reproducibility of experiments** (dataset partitioning, hyperparameter search).
- ** Comprehensive evaluation** (graph similarity metrics, qualitative visualizations).

Once you implement this pipeline, you can systematically reproduce the main results, visualize ontologies, and evaluate model performance according to the paper's standards.

---

Would you like me to now convert this plan into code structure summaries or specific pseudocode?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the ontology learning pipeline in Python using open-source libraries such as Huggingface Transformers for large language models, PEFT (LoRA) for model adaptation, NetworkX for graph operations, SciPy for the Hungarian matching algorithm, and Sentence Transformers for embedding-based similarity metrics. The system will be modular, comprising data loading, model finetuning with a custom masked loss, prompt-based inference for subgraph generation, graph aggregation and pruning, and evaluation with custom metrics. Hyperparameter tuning will be performed via grid search using validation metrics. Visualization of ontologies will be achieved using NetworkX and matplotlib.",
    "File list": [
        "app.py",
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "prompt_generator.py",
        "graph_utils.py",
        "evaluation.py",
        "visualization.py",
        "hyperparameter_search.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class App {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +load_wikipedia() -> Dataset
        +load_arxiv() -> Dataset
    }
    class Dataset {
        +documents: List[str]
        +concepts: List[str]
        +relations: List[Tuple[str, str, str]] # (concept1, relation_type, concept2)
        +annotations: dict # document_id: List[concepts]
    }
    class Model {
        +__init__(model_name: str, use_lora: bool, lora_rank: int=32)
        +finetune(train_data: Dataset, epochs: int, loss_masking: bool) -> None
        +load_pretrained() -> None
        +generate_subgraph(prompt: str, max_tokens: int, sampling_params: dict) -> str
    }
    class PromptGenerator {
        +get_chain_of_thought_prompt() -> str
        +get_direct_prompt() -> str
        +get_instruction_prompt() -> str
    }
    class Graph {
        +nodes: Set[str]
        +edges: Dict[Tuple[str, str], float]  # (concept1, concept2): weight
        +add_edge(concept1: str, concept2: str, weight: float=1.0) -> None
        +prune_edges(threshold: float) -> None
        +remove_cycles() -> None
        +visualize(output_path: str) -> None
    }
    class EvaluationMetrics {
        +graph_f1_score(model_graph: Graph, true_graph: Graph) -> float
        +motif_distance(model_graph: Graph, true_graph: Graph) -> float
        +semantic_similarity(concept1: str, concept2: str) -> float
        +compute_node_embedding(concept: str) -> np.ndarray
    }
    class HyperparameterTuner {
        +search(params_grid: dict, validation_func: callable) -> dict
    }
    class Visualization {
        +plot_graph(graph: Graph, title: str, output_path: str) -> None
    }
    App -- DatasetLoader
    App -- Model
    App -- Trainer
    App -- EvaluationMetrics
    App -- HyperparameterTuner
    Model -- PromptGenerator
    Model -- Graph
    Graph -- Visualization
    Model -- evaluation methods
    DatasetLoader -- Dataset
    Trainer -- Model
    Trainer -- Dataset
    EvaluationMetrics -- Graph
    HyperparameterTuner -- Model
    Visualization -- Graph
    ",
    "Program call flow": "
sequenceDiagram
    participant C as App
    participant DL as DatasetLoader
    participant M as Model
    participant P as PromptGenerator
    participant G as Graph
    participant E as EvaluationMetrics
    participant H as HyperparameterTuner
    participant V as Visualization

    C->>DL: load_wikipedia()
    DL-->>C: dataset
    C->>H: tune hyperparameters(params_grid, validation_func)
    H-->>C: best_params
    C->>M: load_pretrained()
    C->>M: finetune(train_data=dataset, epochs=2, loss_masking=True)
    Note over M, P: optional arXiv transfer finetuning step
    C->>P: get_chain_of_thought_prompt()
    C->>P: get_direct_prompt()
    loop over each document in dataset.documents
        P->>M: generate_subgraph(prompt, max_tokens, sampling_params)
        M-->>P: raw_output
        P-->>G: processed edges from raw output
        G-->>C: accumulate graph
    end
    G-->>C: aggregate edges
    G->>G: prune_edges(threshold
    G->>G: remove_cycles()
    V->>G: visualize_graph(output_path)
    C->>E: evaluate(graph, true_graph)
    E-->>C: metrics
    C->>V: visualize_graphs(subgraphs or final graph)
    "
    ,
    "Anything UNCLEAR": "Clarification needed on the exact input formats for the datasets, the detailed construction of concept annotations, and specific prompt structures with exact formatting. Also, details on hyperparameter ranges for grid search, and whether to include additional constraints like relation types or relation confidence scores in the primitive edges."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "torch>=1.13",
        "transformers>=4.26",
        "sentence-transformers",
        "numpy>=1.21",
        "scipy",
        "networkx",
        "matplotlib",
        "scikit-learn",
        "pyyaml"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class which includes load_wikipedia() and load_arxiv() methods. Responsible for importing dataset, parsing documents, concepts, relations, and annotations. Loads data into Dataset objects. Needs to handle large-scale data efficiently with possibly multi-processing or batching."
        ],
        [
            "model.py",
            "Defines the BingModel class that initializes the pretrained LLM (e.g., Huggingface Model), adds LoRA modules if use_lora=True. Implements finetune() method with the masked loss, which executes the custom training loop involving data batching, loss masking based on relation frequency, and LoRA parameter updates. Implements generate_subgraph() which takes a prompt and sampling parameters, calls the model, and parses generated text into graph relations."
        ],
        [
            "trainer.py",
            "Defines Trainer class that manages model training: calling model.finetune() with datasets, handling validation, validation metrics, and saving checkpoints. Responsible for hyperparameter validation and possibly using hyperparameter tuning logic. May also include utilities for early stopping."
        ],
        [
            "prompt_generator.py",
            "Provides functions to generate prompts: get_chain_of_thought_prompt(), get_direct_prompt(), and get_instruction_prompt(). Parameters include document context, concept lists, relation prompts. Ensures prompt consistency across training and inference."
        ],
        [
            "graph_utils.py",
            "Contains functions to create, merge, prune, and cycle-removal of graphs using NetworkX. Provides support for converting raw relations from model output to graph structures, thresholding edges, removing inverse/bidirectional edges, cycle removal algorithms, and visualization helpers."
        ],
        [
            "evaluation.py",
            "Implements the evaluation class with methods: graph_f1_score(), motif_distance(), semantic_similarity(), and node_embedding(). Uses sentence-transformers for semantic similarity of concepts. Uses Hungarian matching for node correspondence. Counts motifs and calculates distances to ground truth graphs."
        ],
        [
            "visualization.py",
            "Provides visualization functions to generate graph plots with edge confidence or similarity, used for qualitative analysis of generated ontologies."
        ],
        [
            "hyperparameter_search.py",
            "Implements grid search over alpha, beta, M, and other hyperparameters, executing validation metrics, and selecting best parameters based on validation Graph F1."
        ],
        [
            "main.py",
            "System entry point orchestrating data loading, hyperparameter tuning, model finetuning, subgraph generation, aggregation, pruning, evaluation, and visualization. Manages overall experiment pipeline, including data flow between modules."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "prompt_generator.py",
        "graph_utils.py",
        "evaluation.py",
        "visualization.py",
        "hyperparameter_search.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Uses sentence-transformers for semantic similarity embedding; NetworkX for graph management; Hungarian algorithm from SciPy for node matching; datasets stored with concept text, relations, and annotations; prompts stored as template strings. Hyperparameters include mask ratio, pruning thresholds, sampling top-p and temperature, LoRA rank, and relation path length.",
    "Anything UNCLEAR": "Exact dataset annotation process for Wikipedia and arXiv, including concept detection heuristics, relation types, and concept/path annotations, needs clarification. Also, details of the precise prompt templates and how relation extraction from text is parsed into edges are unspecified."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 1e-5  # Initial learning rate; to be tuned further
  batch_size: 16  # Based on GPU memory capacity
  epochs: 2  # Full finetuning epochs for Wikipedia dataset
  loss_masking: true  # Use custom masked relation loss regularizer
  relation_masking_M: 100  # Average relation occurrences (dataset-dependent, tune as needed)
  mask_mask_prob: 0.5  # Mask probability base, adjustable
  gradient_clipping_norm: 1.0  # Optional, for training stability

model:
  base_model_name: "mistral-7b-v0.2"  # Huggingface model identifier
  use_lora: true
  lora_rank: 32
  finetune_on_dataset: "wikipedia"  # or "arxiv" for transfer step
  loRA_init_from_pretrained: true

prompt:
  chain_of_thought_template: |
    <Insert chain-of-thought prompt template here, matching paper's style>
  direct_prompt_template: |
    <Insert direct prompt template here, matching paper's style>
  instruction_prompt_template: |
    <Insert instruction prompt template here, matching paper's style>

sampling:
  temperature: 0.1
  top_p: 0.9
  max_tokens: 512

post_processing:
  prune_alpha_threshold: 0.5  # Threshold for relation confidence pruning, tune via validation
  prune_beta_threshold: 0.5  # Threshold for relation similarity pruning
  cycle_removal_strategy: "greedy"  # Strategy for cycle removal

evaluation:
  semantic_similarity_threshold: 0.436  # Median cosine similarity from WordNet for concept matching
  motif_k: 3  # Motif size (3-node subgraphs)
  validation_metric: "Graph F1"  # Main validation metric, can be switched based on experiments

misc:
  hardware_requirements: "At least one GPU with 16GB+ memory for training and evaluation"
  seed: 42  # Random seed for reproducibility
```

---

**Note:** Full configuration is available in `planning_config.yaml`
