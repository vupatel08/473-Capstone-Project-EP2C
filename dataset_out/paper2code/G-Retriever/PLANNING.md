# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, step-by-step plan to reproduce the G-Retriever framework described in the paper, covering all major aspects of methodology, experiments, datasets, and hyperparameters.

---

# 1. Overall Objective & Approach Overview
- Develop a **retrieval-augmented generation (RAG)** pipeline capable of "chatting with graphs" by integrating graph neural networks (GNNs), large language models (LLMs), and retrieval techniques.
- The key novelty is formulating retrieval over textual graphs via a Prize-Collecting Steiner Tree (PCST) problem, to resist hallucinations and handle large graphs beyond LLM context windows.

---

# 2. Core Methodological Steps

### 2.1. Data Preparation
- **Input Data Types**:
  - **Textual Graph Descriptions**: Each graph is described in a JSON or CSV format with nodes, edges, attributes, and relationships.
  - **Questions/Prompts**: Textual input queries about the graph.
  - **Answers/Responses**: Generated or ground-truth answers, potentially with highlights or key subgraphs.

- **Data Conversion Tasks**:
  - Convert raw graphs into **textualized graph descriptions** (flattened attributes, relationships, node/edge labels).
  - For training/fine-tuning, prepare datasets with:
    - Textual graph descriptions
    - Questions
    - Ground-truth answers or labels
  - For evaluation, prepare separate validation/test sets with **ground truth** and **system outputs**.

### 2.2. Graph Embedding
- Use a **graph encoder** (e.g., Graph Attention Network - GAT) to embed subgraphs.
- Embedding process:
  - For each node: encode features (attributes, labels) into a vector.
  - For edges: encode relationship attributes or relation types.
  - Obtain node-centric embeddings, aggregate via mean pooling (or alternative methods) into a single **graph embedding vector**.
- Encode textual attributes (node labels, edge labels) with a **text embedder** (e.g., SentenceBERT or pretrained LLM layer).

### 2.3. Indexing over Graphs
- **Index all graphs** in the dataset:
  - Compute **textualized graph descriptions** for each graph.
  - Compute **graph embeddings**.
  - Build an **index** (e.g., FAISS, Annoy) for approximate nearest neighbor retrieval of nodes and edges based on cosine similarity.

### 2.4. Query Processing
- For each user question:
  - Encode the question with the same **text embedder** into a vector (`z_q`).
  - Retrieve top-K nodes (`V_k`) and edges (`E_k`) via similarity search.
  - Assign **prize values** based on similarity rank (descending prizes for top-K).

### 2.5. Subgraph Construction via PCST
- Formulate the **Prize-Collecting Steiner Tree** (PCST) over the subgraph:
  - Nodes and edges are assigned prizes based on their similarity scores.
  - The PCST aims to find a connected subgraph that maximizes total prize minus edge costs.
  - **Extend vertices with virtual nodes** if edge prizes exceed edge costs (to handle negative prizes).
  - Use a **near-linear time PCST solver** (e.g., from [9]) to get `S*`.

### 2.6. Textualization of Subgraph
- Convert the constructed subgraph `S*` into a **textual description** (`textualize(S*)`):
  - Flatten node attributes, relationships, and edge attributes into structured text.
  - Concatenate all relevant node/edge descriptions into a single prompt.

### 2.7. Answer Generation
- Concatenate:
  - **Textualized subgraph description**
  - **Question prompt** (`x_q`)
- Input into a **frozen or fine-tuned LLM**:
  - Use **prompt tuning** (learned soft prompts) or fine-tuning approaches (LoRA, full fine-tuning).
  - Incorporate the **graph embedding** as a soft prompt (`h_g`), handled via a **graph prompt layer** (e.g., GNN + Soft Prompt Layer).
  - Use **graph prompt tuning** or **graph token** embedding (`\hat{h}_g`) as a learned soft prompt.
- Generate **response text** with the LLM, conditioned on combined embeddings.

---

# 3. Implementation Details & Hyperparameters

### 3.1. Graph Embedding Encoder
- Use **GAT** with:
  - 4 layers
  - 4 attention heads per layer
  - Hidden dimension: 1024 (from the paper)
- Text embedding layer (SentenceBERT or pretrained LLM's embedding layer).

### 3.2. Indexing & Retrieval
- Use FAISS or similar approximate nearest neighbor (ANN):
  - Node/edge embeddings stored in Faiss index.
  - Retrieval top-`k`:
    - `k=5` for WebQSP
    - `k=3` for SceneGraphs (based on the paper's settings)
- Similarity metric: cosine similarity.

### 3.3. PCST Setup
- Prize assignment:
  -Prize for node `n`: `(k - i)` if in top-`k` nodes, 0 otherwise.
  -Prize for edge `e`: similar for top-`e` edges.
- Edge cost parameter: `C_e = 1` (or tuned per experiment).
- `k` (number of nodes) for the top-k selection during retrieval: experiment with `k=3,5,10,20`.
  
### 3.4. Textualization
- Flatten attributes into text like: `"name: banana; color: yellow; ..."`.
- Relationships as: `"NodeA is related to NodeB: [relation]"`.
- Use consistent templates aligned with dataset styles.

### 3.5. LLM Fine-tuning / Prompt Tuning
- Use **Llama2-7B** (or similar).
- Prompt tuning:
  - Initialize a soft prompt with `q` tokens (e.g., 10 tokens).
  - Use gradient descent on these tokens only.
  - Learning rate: 1e-5 to 1e-4.
  - Prompt length: 10 tokens.
- Full fine-tuning with LoRA:
  - Use rank=16-32 LoRA modules.
  - Fine-tune for 10 epochs on training data.

### 3.6. Evaluation Metrics
- Main metric: **Accuracy** for classification tasks, **Hit@1** for retrieval.
- Hallucination mitigation: measure **valid node/edge fraction**.
- Additional metrics for generative quality: BLEU, ROUGE, or human evaluation (if applicable).

### 3.7. Hardware & Resources
- Use at least **2x NVIDIA A100/80GB** or equivalent.
- Estimated GPU runtime:
  - Fine-tuning LoRA: ~6-12 minutes/epoch.
  - Inference: few seconds per query.

---

# 4. Experiment & Benchmark Protocol

| Step | Description | Details |
|---------|------------------------------|------------------------------|
| Dataset loading | Prepare dataset lists with node/edge info, questions, answers | ExplaGraphs, SceneGraphs, WebQSP |
| Embedding & Indexing | Compute textual and graph embeddings | Use SentenceBERT + GAT + FAISS |
| Retrieval | Implement cosine similarity retrieval | Top-`k` nodes/edges per query |
| PCST Construction | Formulate and solve PCST problem | Use [9]'s near-linear solver |
| Subgraph Textualization | Convert subgraph to prompt text | Templates aligned with dataset |
| Prompt Preparation | Concatenate textualized graph & question | Incorporate graph token embeddings |
| Model Inference | Generate answer via LLM (prompt tuning / full fine-tuned) | Use LoRA/Prompt tuning/Frozen LLama2-7B |
| Evaluation | Compute correctness, hallucination metrics | Accuracy, valid node/edge fractions |

---

# 5. Missing or Clarification Points
- Precise dataset format (JSON, CSV, or custom?): Need sample datasets for conversion.
- Exact hyperparameters for prompt tuning: prompt length, learning rate, epochs, batch size.
- Whether to implement the soft prompt mechanism explicitly or rely on existing prompt tuning approaches.
- How to handle multi-turn chat (if applicable), or focus only on single-turn questions.
- Computational resource specifications and expected runtime estimates.

---

# Summary
This roadmap captures every core component: from data processing, embeddings, retrieval, PCST optimization, textualization, prompt design, model training, and evaluation. Following this plan will set a strong foundation for implementation, enabling reproduction and further development of the G-Retriever framework.

---

Would you like me to prepare a detailed data schema, pseudocode, or specific hyperparameter suggestions next?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will design a modular pipeline with the following core components: 1) Graph Data Handler for loading, converting, and textualizing graph datasets; 2) Embedding Module to encode nodes, edges, and questions using pretrained language models and GAT; 3) Indexing and retrieval system built with FAISS for fast candidate search; 4) PCST Solver to construct relevant subgraphs from retrieved candidates; 5) Graph to Textual Prompt Converter for converting subgraphs into structured text prompts; 6) Prompt Tuning or Fine-tuning Module leveraging HuggingFace LoRA and Llama2-7B for answer generation; 7) Textual Input-Output Interface to manage question prompts and generate responses. This architecture uses open-source libraries: faiss for retrieval, HuggingFace transformers for models, and networkx for graph operations, all orchestrated in a simple main.py script to coordinate the flow.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "graph_processor.py",
        "embedding.py",
        "retrieval.py",
        "pcst_solver.py",
        "prompt_tuning.py",
        "model.py",
        "train.py",
        "evaluate.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(dataset_paths: dict)\n        +load_data() -> list\n        +convert_graph_to_text(graph: dict) -> str\n    }\n    class GraphProcessor {\n        +__init__()\n        +textualize_graph(graph: dict) -> str\n    }\n    class Embedding {\n        +__init__(model_name: str)\n        +encode_nodes(nodes: list) -> np.ndarray\n        +encode_edges(edges: list) -> np.ndarray\n        +encode_question(question: str) -> np.ndarray\n    }\n    class RetrievalSystem {\n        +__init__(embedding_dim: int)\n        +build_index(embeddings: np.ndarray) -> None\n        +search(query_vec: np.ndarray, top_k: int) -> (list, list)\n    }\n    class PCSTSolver {\n        +__init__(node_prizes: np.ndarray, edge_prizes: np.ndarray, edge_cost: float)\n        +solve(graph: networkx.Graph) -> networkx.Graph\n    }\n    class PromptConverter {\n        +convert_graph_to_prompt(graph: networkx.Graph) -> str\n    }\n    class LLMModel {\n        +__init__(model_name: str, prompt_length: int)\n        +prepare_prompt(prompt_text: str, graph_embedding: np.ndarray) -> str\n        +generate_response(prompt: str) -> str\n    }\n    class Trainer {\n        +__init__(model: LLMModel, train_data: list)\n        +train(epochs: int, learning_rate: float) -> None\n    }\n    class Evaluator {\n        +__init__(model: LLMModel, test_data: list)\n        +evaluate() -> dict\n    }\n\nMain --> DatasetLoader\nMain --> Embedding\nMain --> RetrievalSystem\nMain --> PCSTSolver\nMain --> PromptConverter\nMain --> Trainer\nMain --> Evaluator\nEmbedding --> models.transformers\nRetrievalSystem --> faiss\nPromptConverter --> utils\nTrainer --> model\nEvaluator --> model\n",
    "Program call flow": "sequenceDiagram\n    participant C as Main\n    participant D as DatasetLoader\n    participant E as Embedding\n    participant R as RetrievalSystem\n    participant P as PCSTSolver\n    participant PT as PromptConverter\n    participant M as LLMModel\n    participant T as Trainer\n    participant Evl as Evaluator\n    C->>D: initialize with dataset paths\n    D->>D: load_data()\n    D-->>C: dataset list\n    C->>E: initialize with model name\n    loop for each data point\n        D->>D: convert_graph_to_text(graph)\n        E->>E: encode question\n        E->>R: encode and build FAISS index\n        R->>R: build index\n        E->>E: get query vector\n        R->>R: search top-K node/edge vectors\n        R->>P: construct subgraph via PCST\n        P->>P: solve PCST\n        P-->>C: get subgraph\n        PT->>PT: textualize subgraph\n        PT-->M: prepare prompt with graph prompt + question\n        M->>M: generate response\n        T->>T: train if tuning\n        Evl->>Evl: evaluate model responses\n    end\n    C->>C: produce final metrics\n",
    "Anything UNCLEAR": "Clarify if dataset formats are JSON or CSV, and define whether prompt tuning is preferred over full fine-tuning for rapid prototyping. Also, specify if multi-turn dialogue is needed or only single-response generation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "faiss-cpu==1.7.2",
        "transformers==4.26.0",
        "sentence-transformers==2.2.2",
        "networkx==2.6.3",
        "scipy==1.7.1"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Contains DatasetLoader class with methods for loading datasets from JSON/CSV, converting graphs into textual descriptions, and split for train/validation/test sets."
        ],
        [
            "graph_processor.py",
            "Defines GraphProcessor with a method 'textualize_graph(graph: dict) -> str' to convert node/edge data into consistent structured text prompts."
        ],
        [
            "embedding.py",
            "Includes Embedding class that initializes text encoder models (e.g., SentenceTransformer) and encodes nodes, edges, and questions into vector representations."
        ],
        [
            "retrieval.py",
            "Builds FAISS index from node and edge embeddings, provides search(query_vec: np.ndarray, top_k: int) -> (list, list) to retrieve relevant candidates for given query."
        ],
        [
            "pcst_solver.py",
            "Implements solve_pcst(node_prizes: np.ndarray, edge_prizes: np.ndarray, edge_cost: float) -> networkx.Graph: solves PCST using a near-linear time algorithm, returns subgraph."
        ],
        [
            "prompt_tuning.py",
            "Defines PromptTuner class using Huggingface's LoRA or prompt tuning methods, with methods for prompt optimization and embedding of graph prompt tokens."
        ],
        [
            "model.py",
            "Creates LLMModel class that loads pretrained Llama2-7B, prepares input prompts with graph info, and generates responses; includes optional prompt tuning methods."
        ],
        [
            "train.py",
            "Contains Trainer class responsible for fine-tuning the LLM (LoRA or full fine-tuning) with training data involving prompts and responses."
        ],
        [
            "evaluation.py",
            "Provides Evaluation class for computing accuracy, hallucination metrics, and other relevant evaluation metrics on validation/test sets."
        ],
        [
            "main.py",
            "Serves as the orchestration script: coordinates dataset loading, encoding, retrieval, PCST construction, prompt preparation, inference, and evaluation; maintains control flow."
        ],
        [
            "utils.py",
            "Utility functions such as cosine similarity, data normalization, visualization helpers, and any configuration parsers needed."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "graph_processor.py",
        "embedding.py",
        "retrieval.py",
        "pcst_solver.py",
        "prompt_tuning.py",
        "model.py",
        "train.py",
        "evaluation.py",
        "main.py",
        "utils.py"
    ],
    "Full API spec": "",
    "Shared Knowledge": "Shared functions such as cosine similarity, dataset JSON schemas, interface for handling the text-to-graph conversion, and standard prompt optimization routines.",
    "Anything UNCLEAR": "Clarify dataset input formats (JSON or CSV), preferred model hyperparameters, and if multi-turn dialogue support is required or only single-shot question answering."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0001  # Based on prompt tuning / LoRA fine-tuning typical settings
  batch_size: 16        # As used in experimental setup to balance resource use
  epochs: 10            # Standard for convergence in small prompt tuning experiments
  prompt_length: 10     # Number of prompt tokens for prompt tuning
  prompt_learning_rate: 1e-5  # Typical small learning rate for prompt optimization
  edge_cost: 1.0        # Default edge cost parameter for PCST, adjustable if needed
  retrieval_top_k: 5    # Top-K candidates retrieved from FAISS, as per paper settings
  prompt_tuning_method: "LoRA"  # Can be "LoRA" or "prompt tuning"; default is LoRA
  model_name: "Llama2-7B"  # Main LLM backbone as per paper
  model_precision: "fp16" # Use FP16 for efficiency; adjust if full fine-tuning used
  max_input_tokens: 512 # Max token length for the textualized subgraph + question
  smoothing_epsilon: 1e-6 # For numerical stability in cosine similarity calculations

evaluation:
  metrics:
    accuracy: true
    hallucination_fraction: true
  eval_batch_size: 16  # Batch size during evaluation
  eval_steps: 100       # Number of evaluation steps (adjust to dataset size)
```

---

**Note:** Full configuration is available in `planning_config.yaml`
