# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

**Logic Analysis for dataset_loader.py**

---

### Overview:
The `dataset_loader.py` module is responsible for loading various datasets, converting their raw graph data formats into structured, textual representations suitable for downstream processing in G-Retriever, and preparing data splits for training, validation, and testing. This class will be implemented as `DatasetLoader`, with methods to support dataset reading, graph textualization, and dataset splitting.

---

### Core Responsibilities:
1. **Loading datasets from files**:
   - Datasets are stored in JSON or CSV formats.
   - Different datasets (ExplaGraphs, SceneGraphs, WebQSP) have distinct schemas and formats, which must be parsed into a common internal structure.

2. **Graph data extraction**:
   - Parse raw graph data that includes nodes, edges, and attributes.
   - Nodes and edges may be represented as:
     - JSON objects with `node_id`, `node_attr` (attributes), and possibly coordinates or other features.
     - CSV with columns such as `node_id`, `node_attr`, `src`, `dst`, `edge_attr`.
   - Extract nodes, edges, and their attributes into a standard internal format:
     - `nodes`: list of dictionaries with `node_id`, `attributes`.
     - `edges`: list of dictionaries with `src`, `dst`, `attributes`.

3. **Conversion to textual description**:
   - Implement a method (e.g., `convert_graph_to_text(graph: dict) -> str`) that converts the graph into a natural language prompt.
   - The textualization should be consistent across dataset types, and ideally follow a template such as:
     ```
     Node descriptions:
     0: name: banana; attribute: small, yellow; (x,y,w,h): (248, 55, 64, 34)
     ...
     Edge descriptions:
     src -> dst : relation attribute
     ```
   - This facilitates input into LLMs and consistency for the benchmark.

4. **Splitting datasets**:
   - Perform train/validation/test splits in a 6:2:2 ratio, or as per dataset specifics.
   - Store splits as lists of data entries, each containing:
     - Graph in internal format
     - Textual description
     - Associated questions
     - Ground-truth answers for evaluation

5. **Data structure**:
   - Return data as a list of dictionaries, e.g.:
     ```python
     {
       'graph_id': str,
       'nodes': List[{'node_id': int, 'attributes': dict}],
       'edges': List[{'src': int, 'dst': int, 'attributes': dict}],
       'text_description': str,
       'question': str,
       'answer': str
     }
     ```

---

### Detailed Step-by-Step Logic:

#### 1. Initialization (`__init__`)
- Accept a dataset path or list of dataset files.
- Accept dataset name or type (e.g., ExplaGraphs, SceneGraphs, WebQSP).
- Set internal variables for paths, dataset type, and configuration parameters (e.g., dataset splits).

#### 2. Loading Data (`load_data`)
- Based on dataset type:
  - Read JSON: Use `json.load()`.
  - Read CSV: Use `csv.DictReader()`.
- For ExplaGraphs:
  - Parse data into triplets, then convert triplets to graph format:
    - Each triplet: (head, relation, tail) → nodes and edges.
  - Store nodes with their attributes, edges with relations.
- For SceneGraphs:
  - Parse JSON structure containing objects, attributes, relations.
  - Convert JSON objects into nodes/edges format.
- For WebQSP:
  - Parse triplet or graph-like data, e.g., semantic triples.
  - Convert into nodes and edges.

#### 3. Converting raw data to internal graph representation
- For each dataset entry:
  - Create nodes: assign node_id, attributes.
  - Create edges: assign src, dst, relation attributes.
- Store nodes and edges in structured lists or dicts.

#### 4. Converting graph to textual description (`convert_graph_to_text`)
- For each node:
  - Concatenate attributes into a descriptive string.
- For edges:
  - Describe relationships explicitly, e.g., "NodeA is connected to NodeB via relation."
- Use a standardized template:
  ```
  Nodes:
  {node_id}: {attribute descriptions}
  Edges:
  {src} -> {dst}: {relation}
  ```
- Return as a string.

#### 5. Data splitting (`split_dataset`)
- Shuffle the dataset entries if needed.
- Split into:
  - Train: 60%
  - Validation: 20%
  - Test: 20%
- Use consistent randomness seed for reproducibility.
- Store splits as separate lists of data entries.

---

### Additional Considerations:
- **Handling missing data**:
  - Some nodes or edges might lack attributes; treat missing attributes as empty strings.
- **Dataset-specific transformations**:
  - For ExplaGraphs, handle triplet conversion explicitly.
  - For SceneGraphs, process JSON nested structure.
  - For WebQSP, align with semantic triples.
- **Error handling**:
  - Try/except blocks to handle malformed data.
  - Log errors for later inspection but continue processing if possible.
  
### Final Notes:
- Maintain modularity, with dedicated methods:
  - `load_json()`, `load_csv()`, `parse_expla()`, `parse_scene()`, `parse_webqsp()`.
  - `convert_graph_to_text()`.
  - `split_dataset()`.
- Ensure reproducibility by setting fixed seeds during data shuffling.

---

## Summary:
The `DatasetLoader` class will serve as a flexible, dataset-agnostic loader that reads raw graphs, standardizes them, converts to natural language descriptions, and prepares dataset splits—forming the foundation for subsequent retrieval, textualization, and question-answering tasks as described in the paper.

## embedding.py

# Logic Analysis for embedding.py

## Purpose and Role
The `embedding.py` module centers around defining the `Embedding` class, which is responsible for initializing text encoder models, encoding textual attributes of nodes and edges, as well as user questions, into fixed-length vector representations. These embeddings are crucial for downstream retrieval (via cosine similarity), graph encoding, and prompt conditioning in the G-Retriever pipeline, enabling semantic matching and relevance estimation between queries and graph components.

---

## Core Components and Functionality

### 1. Model Initialization
- **Objective**: Load a pre-trained text encoder suitable for generating semantically meaningful vector representations.
- **Implementation Considerations**:
  - Based on `model_name` from configuration (`"Llama2-7B"`), select an appropriate model architecture.
  - Use Huggingface transformers or sentence-transformers library to load a model that can encode textual data into dense vectors.
  - Ensure that the model supports inference in the specified precision (`"fp16"`), if applicable.
  - Keep the model in evaluation mode and freeze parameters if encoding is solely for retrieval (no training of embeddings needed).

### 2. Input Data Handling
- **Input Types**:
  - Nodes: Each node's attribute structured as a string or structured text, e.g., `"name: banana; color: yellow."`
  - Edges: Text labels or attributes representing the relationship, e.g., `"to the left of"`.
  - Questions: User queries, e.g., `"What is the color of the banana?"`
- **Preprocessing**:
  - For each input, ensure consistent formatting.
  - Tokenize inputs before passing to the encoder, respecting `max_input_tokens` (512 tokens).
  - Apply any special tokens or prompts needed to improve encoding consistency.

### 3. Encoding Methods
- **Nodes & Edges**:
  - Use a method `encode_nodes(nodes: list)`:
    - Iterate over node attributes.
    - Encode each attribute string into a fixed-size vector.
  - Use a method `encode_edges(edges: list)`:
    - Iterate over edge attributes.
    - Encode each into a vector.
  - Consider batching to improve efficiency.
  - Store embeddings in a numpy array or torch tensor for retrieval use.
  
- **Questions/Queries**:
  - Use a method `encode_question(question: str)`:
    - Process input question string.
    - Encode into a vector.
    - Useful for similarity search in retrieval.

### 4. Embedding Output Format
- Return embeddings as numpy arrays or PyTorch tensors:
  - For nodes and edges:
    - Shape: `(num_items, embedding_dim)`.
    - `embedding_dim` depends on the model’s hidden size (e.g., 768, 1024).
  - For questions:
    - Shape: `(1, embedding_dim)`.

### 5. Hyperparameters and Settings
- `model_name`: e.g., `"Llama2-7B"`:
  - Choose a compatible model (probably from Huggingface `transformers`).
  - Use `AutoModel` or custom class with `AutoTokenizer`.
- `max_input_tokens`: 512:
  - Use tokenizer’s `max_length=512`, truncate or pad as needed.
- `model_precision`: `"fp16"`:
  - Load model in half-precision for efficiency.
  - Ensure environment supports fp16 operations (e.g., GPU with FP16 support).

### 6. Technical Details & Best Practices
- **Model Loading**:
  - Initialize tokenizer and model once during class instantiation.
  - Use `from_pretrained()` with model paths or identifiers from transformers.
- **Device Handling**:
  - Allow flexible device assignment (CPU/ GPU). Default to GPU if available.
  - Convert models and tensors accordingly.
- **Evaluation Mode**:
  - Set `model.eval()`.
  - Disable gradient computations with `torch.no_grad()` during encoding.

### 7. Dependencies and External libraries
- Transformers for loading models and tokenizers.
- NumPy for storing or manipulating embeddings.
- Torch for tensor operations and inference.
- Optional: SentenceTransformers for more specialized models.

---

## Class Structure and Interfaces

### `class Embedding`
- **__init__(self, model_name: str)**
  - Loads the tokenizer and model.
  - Sets the device.
  - Stores embedding dimension (`d`) based on model config.
- **encode_nodes(self, nodes: list of str) -> np.ndarray**
  - Tokenize node attributes.
  - Encode via model.
  - Pool or select representation (e.g., [CLS] token or mean pooling).
- **encode_edges(self, edges: list of str) -> np.ndarray**
  - Same as encode_nodes, for edge attributes.
- **encode_question(self, question: str) -> np.ndarray**
  - Encode the user query into a vector.
- **(optional) save and load embeddings** for efficiency if needed.

---

## Summary and Key Points

- Load a pretrained text encoder aligned with the specified `model_name`.
- Implement batch encoding for efficiency.
- Maintain consistent tokenization and truncation at `max_input_tokens`.
- Return dense vectors suitable for cosine similarity search.
- Ensure embeddings are stored in numpy arrays or tensors for fast retrieval.
- Make provisions for device management and inference (e.g., no training needed).
- Support for embedding of nodes, edges, and question prompts to facilitate subsequent retrieval and prompt conditioning.

---

This thorough logical breakdown ensures that the implementation in `embedding.py` addresses all necessary components, handles external dependencies gracefully, optimizes performance through batching, and aligns with the overall pipeline to support the retrieval-driven G-Retriever architecture.

## evaluation.py

# Evaluation.py Logical Analysis for G-Retriever Experimental Framework

This document provides a detailed, step-by-step logical blueprint for implementing the `evaluation.py` module, specifically the `Evaluation` class, which is critical for assessing the performance and hallucination mitigation capabilities of the G-Retriever pipeline as described in the paper. The analysis aligns strictly with the paper’s methodology, datasets, experimental configurations, and the provided `config.yaml`, ensuring reproducibility and fidelity.

---

## 1. Objectives of evaluation.py

The core purpose of the `Evaluation` class is to:

- Compute **accuracy-based metrics** for downstream question-answering tasks (classification, retrieval, etc.).
- Measure **hallucination** levels, specifically:
  - Valid nodes fraction
  - Valid edges fraction
  - Fully valid graph (both nodes and edges are correct)
- Provide detailed diagnostics to understand the model’s faithfulness in referencing actual graph elements.
- Support batch/step-wise evaluation aligned with dataset splits.

---

## 2. Inputs to Evaluation Class

### 2.1. Data
- **Test or validation dataset**, structured as:
  - For each sample, the *question text*, *graph description or subgraph*, *ground truth answer(s)*.
  - *Model predictions* for each sample, typical as textual responses, plus the referenced nodes and edges (extracted from responses) for hallucination assessment.

### 2.2. Configuration Parameters
- Metrics to compute, from `config['evaluation']['metrics']`:
  - Accuracy (or task-specific metrics like Hit@1)
  - Hallucination metrics: fractions for valid nodes, valid edges, and fully valid graphs
- Evaluation batch size (`eval_batch_size`)
- Evaluation steps or total number of samples to evaluate (`eval_steps`)

### 2.3. Model & System
- Access to the **model inference outputs**:
  - Predicted answer text
  - Optional: Rationale or cited graph elements as part of the output (either extracted from responses or stored separately)
- Ground-truth labels and data for accuracy evaluation

---

## 3. Key Functional Components and Logical Flow

### 3.1. Data Loading & Preparation
- Input: The evaluation dataset split (test/validation).
- For hallucination metrics, additionally:
  - Maintain a **reference graph** for each sample, possibly in a structured format (list of nodes, edges, attributes).
  - Extract from the model's predicted response the **referenced nodes and edges**:
    - Use regex or structured parsing (e.g., if responses include explicit mention of node IDs or labels).
    - Or, if the model outputs citations explicitly, decode this linking.

### 3.2. Metric Computation

**(A) Accuracy Metrics**
- Based on ground truth and predicted answers:
  - For classification: match predicted answer to ground truth label(s).
  - For multiple-choice or top-k metrics, compare index or label accuracy.
- Use metrics such as:
  - Exact match
  - Top-1 accuracy (or specified by task)

**(B) Hallucination Metrics**
- **Valid Nodes Fraction**:
  - For each predicted node citation:
    - Check if the node ID/label exists in the reference graph.
    - Count how many predicted nodes are valid (true positives).
  - Compute fraction: (number of valid nodes) / (total predicted nodes).

- **Valid Edges Fraction**:
  - Repeat as above for edges:
    - Validate each referenced edge against the reference graph.
    - Count valid edges.

- **Fully Valid Graphs Fraction**:
  - Both node and edge sets are fully correct for the sample:
    - Count as valid only if **all** cited nodes and edges are correct.
    - Compute fraction over total evaluated samples.

**(C) Overall Faithfulness Measure**
- This encompasses the combined correctness of referenced nodes and edges per sample.

### 3.3. Implementation Strategy for Hallucination
- To assess cited graph elements:
  - Develop a **extraction method**:
    - Parse model responses to extract cited node IDs/labels and edge IDs/labels.
    - For example, responses might follow a template:
      - "The nodes supporting this are: node1, node2, node3; The edges supporting this are: edge1, edge2."
    - The code should accommodate flexible parsing or rely on model outputs formatted accordingly.
- Cross-reference these citations with the **ground truth graph** stored per sample.
- Derive validation flags for each citation, accumulate counts.

### 3.4. Automation of Metrics Calculation
- Perform batch evaluation for efficiency.
- Store metrics in a dictionary or structured object for reporting.

### 3.5. Result Aggregation & Reporting
- Aggregate metrics:
  - Mean and standard deviation across samples.
  - For hallucination metrics: overall fractions.
- Generate a summary report, including:
  - Accuracy statistics (mean/std).
  - Hallucination metrics (fractions per category).
  - Optional: per-sample breakdown for deeper diagnostics.

---

## 4. Specific Implementation Details Based on the Paper & YAML

- Use `numpy` for numerical computations (means, std).
- Use `scipy` if needed for statistical tests (not explicitly required, but optional).
- Follow strict definitions:
  - Valid nodes/edges: must match the reference graph exactly (string label or node ID).
  - Fully valid: both node set and edge set are completely accurate.
  - Use a `smoothing_epsilon` from config to prevent division errors if necessary.
- Use `pandas` or structured data access if datasets are CSV/JSON.

---

## 5. Additional Considerations

- Log intermediate statistics for debugging.
- Handle cases where citations are missing or malformed.
- Consider model responses that do not cite any graph elements; treat these as invalid.
- For multi-modal models, ensure the references are reliably parsed.

---

## 6. Example Pseudocode for the Evaluation Class

```python
class Evaluation:
    def __init__(self, dataset, model_outputs, ground_truths, config):
        self.dataset = dataset
        self.model_outputs = model_outputs
        self.ground_truths = ground_truths
        self.metrics_config = config['evaluation']['metrics']
        self.results = {}

    def compute_accuracy(self):
        # Compare predicted answers to ground truth labels.
        correct = 0
        for pred, true in zip(self.model_outputs, self.ground_truths):
            if self.match_answers(pred, true):
                correct += 1
        accuracy = correct / len(self.ground_truths)
        self.results['accuracy'] = accuracy

    def match_answers(self, pred_answer, true_answer):
        # Exact, top-k, or semantic matching as required.
        return pred_answer.strip().lower() == true_answer.strip().lower()

    def evaluate_hallucination(self):
        valid_nodes_counts = []
        valid_edges_counts = []
        fully_valid_counts = []

        for sample_idx, sample in enumerate(self.dataset):
            ref_graph = sample['graph']
            pred_response = self.model_outputs[sample_idx]
            cited_nodes, cited_edges = self.extract_citations(pred_response)

            # Count valid nodes
            valid_nodes = sum(1 for n in cited_nodes if n in ref_graph['nodes'])
            # Count valid edges
            valid_edges = sum(1 for e in cited_edges if e in ref_graph['edges'])

            # Check if ALL cited nodes and edges are valid
            all_nodes_valid = (valid_nodes == len(cited_nodes))
            all_edges_valid = (valid_edges == len(cited_edges))
            fully_valid = all([all_nodes_valid, all_edges_valid])

            valid_nodes_counts.append(valid_nodes / max(len(cited_nodes), 1))
            valid_edges_counts.append(valid_edges / max(len(cited_edges), 1))
            if fully_valid:
                fully_valid_counts.append(1)
            else:
                fully_valid_counts.append(0)

        # Aggregate
        self.results['valid_node_fraction'] = np.mean(valid_nodes_counts)
        self.results['valid_edge_fraction'] = np.mean(valid_edges_counts)
        self.results['full_graph_fraction'] = np.mean(fully_valid_counts)

    def extract_citations(self, response_text):
        # Parse the answer to find cited nodes and edges.
        # Implementation depends on response format, e.g.:
        # Use regex or structured extraction if responses are formatted.
        cited_nodes = []  # parsed from response
        cited_edges = []
        return cited_nodes, cited_edges

    def evaluate_all(self):
        self.compute_accuracy()
        if 'hallucination' in self.metrics_config:
            self.evaluate_hallucination()
        return self.results
```

---

## 7. Final notes

- Ensure the code is compatible with `numpy` and `scipy`.
- The class should support batch processing and cumulative metrics for multiple samples.
- Design to output metrics with mean and std if multiple runs/samples are involved.
- Incorporate detailed logs for debugging and per-sample diagnostics.

---

## Summary

This logic analysis provides a comprehensive, clear, and actionable plan for implementing `evaluation.py`. It emphasizes alignment with paper methodology, dataset formats, model outputs, and hyperparameters. Strict adherence to these points will enable the accurate reproduction of the experimental evaluation, especially the critical hallucination mitigation metrics.

---

Would you like me to proceed with the actual code implementation based on this analysis?

## graph_processor.py

### Logic Analysis for `graph_processor.py`: `GraphProcessor` and `textualize_graph()` method

---

#### **Purpose & Overall Functionality**

The core purpose of `GraphProcessor` is to transform a raw graph data structure (represented as a dictionary) into a well-structured, natural language **textual prompt** suitable for inclusion in LLM inputs. This textualized graph should preserve all relevant node and edge information in a human-readable, consistent format, facilitating effective prompt creation for retrieval and generation tasks.

---

#### **Input Structure**

- **Graph Format**: The input `graph` is expected as a `dict` (or JSON-like structure) with standard attributes:
  - `node_id`: Unique identifier for each node.
  - `node_attr`: Descriptive attributes (strings) for nodes.
  - `src` / `dst`: Source and destination node IDs for edges.
  - `edge_attr`: String attributes or relationship descriptions for edges.
  
- **Sample Form**:
```python
{
    "node_id": [0, 1, 2, ...],
    "node_attr": ["banana", "spotted apple", ...],
    "src": [0, 1, ...],
    "edge_attr": ["next to", "on", ...],
    "dst": [1, 2, ...]
}
```

- For datasets, the graph might be a dictionary containing lists of nodes and edges separately, or nested JSON objects that need to be flattened into this structure.

---

#### **Output Format**

- **Textual Description**:
  - Nodes should be described in a uniform, structured textual format, e.g.:
    ```
    node_id: [ID], name: [attributes]
    ```
  - Edges should be described similarly, e.g.:
    ```
    src, edge_attr, dst
    ```
- **Concatenation & Formatting**:
  - The entire graph will be serialized as multiple lines, each representing a node or edge.
  - For clarity, nodes and edges should be grouped and separated by appropriate headers or annotations.
  - For example, the final string could be:

```plaintext
Nodes:
node_id: 0, name: banana; other attributes...
node_id: 1, name: spots; other attributes...
Edges:
0, next to, 1
1, on, 2
...
```

or a flattened, paragraph-style prompt that maintains logical structure:
```plaintext
The graph contains the following nodes:
- Node 0: banana, small, yellow
- Node 1: spots, small, water
Edges:
- between Node 0 and Node 1: next to
- between Node 1 and Node 2: on
```

---

#### **Key Logical Steps & Transformations**

1. **Input Handling & Validation**
   - Ensure the input `graph` contains required fields: `node_id`, `node_attr`, `src`, `dst`, `edge_attr`.
   - Validate that list lengths are consistent; `len(node_id) == len(node_attr)` and `len(src) == len(dst) == len(edge_attr)` for edges.
   - If the dataset presents in alternate format, add a normalization step to convert it into the above structure.

2. **Node Textualization**
   - Iterate over the node list:
     - For each node, retrieve its ID (`node_id[i]`) and attributes (`node_attr[i]`).
     - Generate a well-formatted sentence or line:
       ```
       Node {id}: {attributes}
       ```
     - If additional attributes (e.g., color, shape) are available, include them uniformly.

3. **Edge Textualization**
   - Iterate over the edge list:
     - For each edge, get index `i`.
     - Retrieve `src[i]`, `edge_attr[i]`, `dst[i]`.
     - Generate a sentence/line of form:
       ```
       Node {src}: {edge_attr}: Node {dst}
       ```
       or a simplified version:
       ```
       {src} --{edge_attr}--> {dst}
       ```
   - Ensure the textual description clearly shows the relationship.

4. **Formatting & Structuring**
   - Compose the text sequentially:
     - Start with nodes descriptions, possibly under a "Nodes:" heading.
     - Follow with edges descriptions, under an "Edges:" heading.
   - Alternatively, combine into a single narrative paragraph for more natural language prompts, e.g.,
     ```
     The graph includes nodes: Node 0: banana; Node 1: apples; Edges: 0 --next to--> 1...
     ```
   - Maintain consistent language style across the entire textualization.

5. **Handling Special Cases**
   - Empty nodes or edges: produce minimal or "no nodes/edges" descriptions.
   - Graphs with multiple attributes per node/edge: concatenate attributes with commas.
   - Large graphs: ensure the serialization remains human-readable, possibly truncating or summarizing if needed.

6. **Output Quality & Consistency**
   - Apply uniform formatting rules:
     - Use lowercase or title case as per dataset style.
     - Maintain consistent delimiters and punctuation.
   - Guarantee tokens are not excessively verbose; keep prompts concise yet informative.

---

#### **Additional Considerations & Best Practices**

- **Customization & Extensibility**
  - Design `textualize_graph()` to accept optional parameters (e.g., include/exclude attributes, formatting styles).
  
- **Preprocessing & Dataset Compatibility**
  - Built-in functions to handle different input formats from various datasets (JSON, CSV).
  - Convert nested data into flattened, uniform structure for textualization.

- **Code Robustness & Error Handling**
  - Add try/except blocks to handle missing or malformed data.
  - Log warnings for missing attributes or inconsistent lengths.

---

#### **Summary**

The `textualize_graph()` function must:

- Take a graph dictionary with node and edge lists.
- Generate a multi-line, consistent textual description detailing nodes and their attributes, then edges and their relationships.
- Format descriptions for clarity, natural language suitability, and prompt effectiveness.
- Be flexible enough to handle different dataset formats and edge cases while producing compact, informative text describing the entire graph.

This structured textualized output will serve as a core part in the prompt for retrieval-augmented question answering, ensuring that relevant graph context is effectively communicated to the language model.

## main.py

# Main.py Logic Analysis for G-Retriever Framework

This document provides a comprehensive, step-by-step analysis of the main.py script, which serves as the central orchestrator of the entire G-Retriever pipeline. Its purpose is to coordinate dataset loading, graph encoding, retrieval, subgraph construction via PCST, prompt preparation, LLM-based response generation, and evaluation. All components and flow should adhere strictly to the design, data structures, and hyperparameters specified.

---

# 1. Initialization and Configuration Loading

- **Load configuration (`config.yaml`)**:
  - Parse hyperparameters such as learning rates, batch sizes, prompt length, top-K retrieval, PCST edge cost, etc.
  - Extract model names, evaluation settings, and other parameters.
- **Initialize Components**:
  - DatasetLoader instance: for loading datasets (`ExplaGraphs`, `SceneGraphs`, `WebQSP`)
  - Embedding Module: with text encoder (SentenceBERT or similar)
  - Retrieval System: FAISS index built from embedded nodes/edges
  - Graph Encoder: GAT network
  - PCST Solver: instance configured with edge cost and prizes
  - LLM Model: chat-based Llama2-7B with optional prompt tuning or LoRA
  - Prompt Tuning/LoRA: loaded if tuning is enabled
  - Evaluation module: set up for metrics calculation

---

# 2. Dataset Loading and Preprocessing

- For each dataset (or selected subset):
  - **Load raw data** using DatasetLoader:
    - Datasets are likely in JSON or CSV formats, converted into internal dict/list structures.
    - Structure expected: list of graphs, each with nodes, edges, attributes (`x_n`, `x_e`) and questions (`x_q`), plus answers.
  - **Convert raw graph data into textual descriptions**:
    - Use `graph_processor.py`'s `textualize_graph()`:
      - Expand node attributes, edge attributes, and relationships into a structured, concatenated string (e.g., `"node_id: attribute; relation: node_id"`).
  - **Split datasets** into training, validation, and test sets as per dataset specs (original, then possibly re-split with 6:2:2 ratio).

---

# 3. Precompute Embeddings and Build Index

- For every graph:
  - **Encode nodes and edges**:
    - Use the embedding module to generate vectors (`z_n`, `z_e`) for each node and edge based on their textual attributes.
  - **Compute graph embedding**:
    - Use the GAT network to encode the entire graph into a single vector (`h_g`).
  - **Store textual descriptions** for later prompt construction.
- **Build FAISS index**:
  - Collect all node/edge embeddings across datasets.
  - Insert into FAISS index for fast cosine similarity search.
- Keep references (e.g., IDs, texts, embeddings) for retrieval.

---

# 4. Query Processing and Retrieval

- **For each test/query example**:
  - **Encode question**:
    - Using text embedder (`SentenceTransformer` or LHLM's first layer) to get `z_q`.
  - **Retrieve candidate nodes and edges**:
    - Use FAISS `search()` with cosine similarity.
    - Collect top-`k` nodes and edges (k from config), e.g., `k=5`.
  - **Assign prizes**:
    - Sorted by similarity; highest gets `k-1`, next `k-2`, etc., zero beyond top-`k`.
  - **Construct a weighted subgraph for PCST**:
    - Use the indices of retrieved nodes and edges.
    - Include their respective prizes.
    - Set edge costs as per config (`C_e=1`, or other value).
    - Handle negative prizes via virtual nodes as described in the paper.
  - **Solve PCST**:
    - Run the PCST algorithm (`pcst_solver.solve_pcst()`), obtaining a connected subgraph `S*`.
  - **Store/prepare `S*`** for textualization.

---

# 5. Conversion of Subgraph to Textual Prompt

- Use `graph_processor.py`'s `textualize_graph(S*)`:
  - Convert nodes, relationships, and edges to a well-structured descriptive string.
  - Maintain consistent formatting similar to dataset style.
- **Concatenate** textualized subgraph with the query:
  - Final prompt: `[Textualized Subgraph] [Question]`.
  - Apply prompt templates if necessary to guide LLM responses.

---

# 6. Prepare Input for LLM

- **Graph prompt embedding**:
  - Use the `PromptTuning` or `LoRA` modules:
    - Generate or update graph token representations (`\hat{h}_g`) as soft prompts.
    - Concatenate `\hat{h}_g` and the textual prompt embedding (`h_t`) derived from the combined textual description + question.
- **Formulate prompt input**:
  - Input tokens: `[Graph prompt] [Textual question]`.
  - Input embeddings: merge soft prompt vectors with text embeddings.

---

# 7. Inference and Response Generation

- **Generate answer**:
  - Feed combined input into the frozen or fine-tuned LLM (`Llama2-7B`):
    - Use `generate()` method with max new tokens (`max_length=32`) and temperature settings as per config.
    - During prompt tuning only prompts are trained; if fine-tuning, adjust entire model accordingly.
- **Output**:
  - Capture generated response text.
  - Save responses for evaluation.

---

# 8. Evaluation and Metrics Computation

- **Metrics**:
  - Accuracy against ground truth answers (for classification datasets).
  - Hit@1 and hallucination metrics:
    - Compare referenced nodes/edges in responses against ground truth.
    - Compute the fraction of valid nodes/edges cited.
  - Log response quality, hallucination reduction, and parsing errors.
- **Optional**:
  - For qualitative assessment, store examples and subgraphs.

---

# 9. Iterative and Batch Processing

- Process data in mini-batches:
  - Batch size as per config (`batch_size=16`).
  - Use appropriate data loaders.
- For large datasets, run in a loop:
  - For each dataset example:
    - Load, encode, retrieve, construct subgraph, textualize, prompt, generate, and evaluate.
- Record metrics and logs periodically.

---

# 10. Final Output and Reporting

- **Produce main evaluation reports**:
  - Aggregate metrics over dataset splits.
  - Generate detailed logs for hallucination analysis.
- **Save**:
  - Embeddings, retrieved subgraphs, generated responses.
  - Model checkpoints (for fine-tuning).
  - Configurations and hyperparameters used.

---

# Additional Considerations
- Ensure all external dependencies (FAISS, transformers, PyTorch, etc.) are initialized once for efficiency.
- Use device management (`cuda`/`cpu`) for speed.
- Incorporate exception handling for missing data, failed retrieval, or PCST solutions.
- Modular structure: encapsulate steps into functions or classes while maintaining clear flow.

---

# Summary
The main.py script systematically manages each stage: dataset ingestion, embedding, retrieval, subgraph construction, textualization, prompt preparation, LLM inference, and evaluation. Its implementation should strictly respect the data interfaces, hyperparameters, and modular design outlined. The process fosters accurate, scalable, and explainable graph questioning aligned with the paper’s methodology.

---

Would you like me to produce pseudocode snippets, class diagrams, or a detailed step-by-step flow for implementation?

## model.py

**Logic Analysis: for model.py**

---

### Purpose:
Implement the `LLMModel` class responsible for:
- Loading the pretrained Llama2-7B model
- Preparing prompts that include graph embeddings and textual data
- Generating responses with the model
- Supporting optional prompt tuning via prompt embeddings or LoRA fine-tuning

---

### Core functionalities:

1. **Model Initialization**:
   - Load the pretrained Llama2-7B model and tokenizer
   - Set precision (FP16) if specified
   - Initialize prompt tuning components if specified:
     - Standard prompt tokens (virtual tokens)
     - Soft prompt embedding layer
     - LoRA modules for full fine-tuning
   - Ensure model is loaded with `requires_grad=False` for frozen base model

2. **Prompt Preparation**:
   - Inputs:
     - `graph_token_embedding` (`\hat{h}_g`): The learned soft prompt embedding for the subgraph, shape `[d_l]`
     - `textual_embedding` (`h_t`): The embedding of question + textualized graph prompt, shape `[L, d_l]`
     - Additional optional prompt tokens (for prompt tuning)
   - Combine embeddings:
     - Concatenate `graph_token_embedding` with the textual embedding (if applicable), with correct placement in input sequence
     - Format prompt:
       - Embedded prompt tokens + textual question + textualized graph prompt
   - Convert combined embeddings into input IDs via tokenizer if necessary, or directly into input embeddings, respecting the approach (prompt tuning often involves directly manipulating embeddings to avoid re-tokenizing)
   - Prepare attention mask, if working with token IDs

3. **Generation Process**:
   - Use the frozen Llama2-7B model with:
     - Special attention to whether prompt embeddings are used
     - Appropriate `generate()` method call:
       - Set `max_new_tokens=32`
       - Use `do_sample=False` for deterministic output unless sampling is desired
       - Pass in prepared input embeddings or input IDs
       - Pass attention masks as needed
   - Collect output tokens
   - Decode tokens to text

4. **Prompt Tuning / LoRA Handling**:
   - If using prompt tuning:
     - Maintain a learnable `prompt_embedding` (e.g., `[q, q, ..., q]` for 10 tokens)
     - During training:
       - Use backprop on these prompt embeddings
     - During inference:
       - Replace or prepend these prompt embeddings in the input
   - If using LoRA:
     - Load the LoRA modules (via huggingface adapter or PEFT)
     - Ensure the `requires_grad` is set accordingly
     - Model is loaded with the LoRA extension

5. **Inference Details**:
   - Ensure that the input prompt construction is consistent
   - The generated response should respect the max token limit
   - Output should be plain text generated by the model

6. **Additional considerations**:
   - Prompt embedding shape: shape `[prompt_length, d_l]`
   - Handle device (`cuda`/`cpu`) accordingly
   - Be prepared for model-specific tokenization and embedding handling (`inputs_embeds` vs. `input_ids`)

---

### Implementation notes:

- **Model loading**:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  ```
  - Load model: `AutoModelForCausalLM.from_pretrained()` with `revision='main'` or specific checkpoint
  - Set precision: convert model to FP16 if specified
- **Prompt embeddings**:
  - If soft prompt tuning:
    - Initialize learnable tensor of shape `[prompt_length, d_l]`
    - Register as a parameter for optimization
  - During inference:
    - Use the prompt embedding as `inputs_embeds` input to the model
- **Generation**:
  ```python
  outputs = model.generate(inputs_embeds=prepared_embeddings, max_new_tokens=32, ...)
  ```
- **Decoding**:
  ```python
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)
  ```

---

### Summary:

- The class will encapsulate the loading and configuration of the Llama2-7B model with options for prompt tuning / LoRA, managing prompt embedding parameters.
- Input preparation involves constructing combined prompt embeddings with the graph soft prompt and textual input.
- `generate()` is called with these embeddings to produce the answer.
- Output is decoded into human-readable strings.

This thorough, modular capacity will facilitate experimentation with prompt tuning, full fine-tuning, and different prompt configurations for evaluating the G-Retriever's questioning capabilities.

---

This completes the logical analysis for `model.py`. Would you like me to proceed with drafting the class skeleton or detailed pseudocode next?

## pcst_solver.py

{
  "file_name": "pcst_solver.py",
  "purpose": "Implement a near-linear time Prize-Collecting Steiner Tree (PCST) solver that, given node prizes, edge prizes, and a uniform edge cost, identifies an optimal connected subgraph maximizing total prize minus edge costs. The result is returned as a NetworkX graph representing the relevant subgraph for retrieval and explanation in the G-Retriever framework.",
  "Key functional requirements": [
    "Input parameters:",
    "  - node_prizes: np.ndarray of shape (N,) representing prizes for each node in the graph.",
    "  - edge_prizes: np.ndarray of shape (M,) representing prizes for each edge.",
    "  - edge_cost: float representing uniform cost assigned to each edge.",
    "Output:",
    "  - A networkx.Graph object containing a connected subgraph adhering to PCST criteria."
  ],
  "Assumptions": [
    "The input graph (nodes and edges) can be reconstructed or is implicitly represented via node_prizes and edge_prizes.",
    "The number of nodes and edges are compatible with the provided prize arrays.",
    "The algorithm is designed for efficiency, thus approximate solutions are acceptable; exact NP-hard solutions are infeasible at the scale (~near-linear time requirement).",
    "The prizes and costs are non-negative; handling negative prizes (via virtual nodes) is incorporated to avoid negative edge costs.",
    "The graph structure is sparse enough to allow near-linear time algorithms."
  ],
  "Implementation details / Approach": [
    "1. Graph Reconstruction:",
    "   - (Optional) If the graph structure isn't provided explicitly, assume the virtual graph can be reconstructed or is accessible; if not, the process is conceptualized as operating over abstract prizes and costs.",
    "   - For the purpose of this function, the internal graph structure for the PCST is an abstract or preconstructed graph with nodes and edges linked via indices.",
    "2. Prize Assignment and Virtual Nodes:",
    "   - Use node_prizes directly for top-k nodes (based on rank or similarity).",
    "   - Use edge_prizes similarly for top-k edges.",
    "   - For edges where the edge prize exceeds the edge cost, model negative edge costs via virtual nodes:"
    + "   - Create a virtual node per such edge, connecting it to both endpoints, and assign a prize equal to (edge prize - edge cost) to the virtual node.",
    "   - Connect the virtual node with zero-cost edges to original endpoints.",
    "3. Solving PCST:",
    "   - Use a near-linear time approximation algorithm as per [9].",
    "   - The algorithm should aim to maximize sum of node and edge prizes minus total edge costs, possibly adopting a primal-dual or greedy heuristic suitable for large graphs.",
    "4. Return:",
    "   - Output the resulting connected subgraph as a networkx.Graph object, containing nodes and edges selected by the PCST solution.",
    "   - Remove any virtual nodes and associated edges from the output to keep the subgraph structurally relevant.",
    "   - Ensure that the subgraph is connected and comprises the nodes and edges with the highest cumulative prize, subject to the cost constraints."
  ],
  "Step-by-step logic": [
    "A. Initialize a graph structure, possibly a NetworkX graph, with nodes and edges corresponding to the global graph or to a virtualized version where necessary.",
    "B. Assign initial node prizes directly from node_prizes array.",
    "C. Assign edge prizes directly from edge_prizes array.",
    "D. For each edge where prize > edge_cost, replace the edge with a virtual node:",
    "   - Create a new node (virtual node) with prize = (edge prize - edge cost).",
    "   - Connect this node to the original endpoints with zero-cost edges.",
    "E. Run the PCST approximation algorithm over this augmented graph:",
    "   - The algorithm selects a subset of nodes and edges that maximize total prize minus total edge costs.",
    "   - The solution should enforce connectivity and maximize prize collection.",
    "F. Extract the subgraph from the solution:",
    "   - Remove virtual nodes and their incident edges.",
    "   - Ensure only original nodes and edges remain in the output.",
    "G. Output the subgraph as a networkx.Graph object.",
    "H. Guarantee that the subgraph is connected, validated via connectivity checks.",
    "I. Return the subgraph or handle cases where no nodes/edges are selected (empty graph)."
  ],
  "Notes": [
    "This implementation relies on a suitable approximation algorithm detailed in [9], which is known for near-linear complexity and good practical performance.",
    "Handling of virtual nodes ensures that the possibility of negative edge costs is incorporated without violating algorithm constraints.",
    "The function should be designed modularly, allowing optional parameters (e.g., parameters for the approximation algorithm).",
    "Efficiency tips: Use sparse graph representations, avoid unnecessary copying, and leverage optimized libraries such as NetworkX and numpy."
  ],
  "Summary": "The core purpose of 'solve_pcst' is to mediate between the prizes (node and edge rewards that reflect relevance) and costs (edge penalty, including the virtual node mechanism to handle edge prize > cost scenarios), and to identify a subgraph that best balances relevance and compactness for downstream retrieval tasks. The approximate PCST algorithm must be scalable, fast, and capable of producing solutions consistent with the formulation, enabling effective subgraph retrieval in the large-scale and complex graph environment described."

}

## prompt_tuning.py

{
  "prompt_tuning.py": "The core purpose of this module is to define the class PromptTuner that manages prompt optimization (both prompt tuning and LoRA fine-tuning) for the large language model (LLM), specifically Llama2-7B, as described in the paper and configuration.\n\nKey functionalities include:\n- Initialization of the prompt tuning environment, including loading the pretrained LLM,\n  setting up prompt parameters (prompt tokens), and optional prompt tuning or LoRA modules.\n- Methods for forward pass: embedding prompts, incorporating graph information as soft prompts,\n  and generating responses.\n- Optimization routines: updating prompt tokens (prompt tuning) or LoRA parameters (full fine-tuning) based on loss.\n- Handling of prompt parameters: initialization, gradient updates, and possibly saving/loading.\n\nDetailed Step-by-Step Logical Breakdown:\n\n1. Initialization:\n- Load the pretrained Llama2-7B model, with specified precision (fp16).\n- Decide whether to use prompt tuning or LoRA based on configuration.\n- For prompt tuning:\n  - Initialize a sequence of trainable prompt tokens (embedded as learnable parameters) of length `prompt_length`.\n  - The prompt embedding might be directly defined as a torch.nn.Parameter of shape `(prompt_length, embedding_dim)`.\n- For LoRA fine-tuning:\n  - Wrap the model's weights with LoRA modules, which involve additional trainable matrices.\n- Set up optimizer including a small learning rate (`prompt_learning_rate`=1e-5)\n  and parameters (prompt tokens or LoRA modules).\n- Ensure that only prompt tokens or LoRA parameters are trainable, with all other model weights frozen.\n\n2. Embedding & Prompt Preparation:\n- When given a graph prompt, prepare the textual prompt including the textualized graph description.\n- Map the prompt tokens (or prompt text) into embeddings:\n  - If prompt tuning: use the trainable prompt embedding parameters.\n  - Concatenate prompt embeddings with the question tokens.\n- Pass through the model's embedding layer to get prompt embeddings compatible with the model.\n\n3. Soft Prompt & Graph Embedding:\n- Integrate the learned prompt embedding (graph prompt) `P_e` (prompt tokens) into the model input.\n- During training, update these prompt tokens based on generated loss.\n- For graph prompt embedding (`h_g`):\n  - Initialize as a torch.nn.Parameter of shape `(1, embedding_dim)` or `(prompt_length, embedding_dim)` depending on implementation.\n  - During training, include `h_g` as part of the prompt input, allowing gradients to update it.\n- The prompt tokens embed into the same dimension as model's hidden states (`d_l`) and are used as a soft prefix (prompt embedding) to influence generation.\n\n4. Optimization Routine:\n- For prompt tuning:\n  - Use optimizer (AdamW) to update only prompt token parameters (`P_e`).\n  - During each training step:\n    - Prepare combined prompt and question input.\n    - Perform forward pass through the frozen LLM.\n    - Calculate loss comparing generated output and ground truth.\n    - Backpropagate only through prompt parameters.\n- For LoRA fine-tuning:\n  - Update LoRA modules' weights (via standard backprop and optimizer), keeping other weights frozen.\n\n5. Generation:\n- During inference:\n  - Generate answer by concatenating textualized subgraph prompt and question.\n  - Incorporate the soft prompt embedding (`hat_h_g`) into the prompt.\n  - Run the model in generation mode (e.g., `model.generate()`), producing responses.\n- During training:\n  - Use label tokens (answers) to compute cross-entropy loss.\n\n6. Saving / Loading:\n- Save prompt parameters (`P_e`) and LoRA weights as needed.\n- Load pre-trained prompt parameters for inference or further fine-tuning.\n\n7. Integration with Main Pipeline:\n- During inference, receive the textualized graph, question, and optional ground truth.\n- Use `PromptTuner` to set up prompts, run the model to generate answers.\n- During training, update prompt/LoRA parameters as described.\n\nSpecial considerations: \n- Ensure the prompt tokens are initialized properly (e.g., random or specific initialization).\n- Keep other model weights frozen (except prompt or LoRA modules).\n- Maintain reproducibility: seed setting, clear parameter states.\n- For efficiency, limit prompt length (`prompt_length: 10`) and max tokens (`max_input_tokens: 512`).\n- Use mixed precision (fp16) as specified.\n- Provide interfaces for saving/loading prompt and LoRA weights.\n\nIn summary, the class prompts should implement:\n- `__init__(self, config: dict)`:\n  - Load the pretrained model.\n  - Initialize prompt/LoRA modules based on config.\n  - Set up optimizer.\n- `prepare_prompt(self, textual_graph: str, question: str)`:\n  - Embed prompt tokens.\n  - Concatenate prompt embeddings with question tokens.\n  - Return combined prompt tensor.\n- `train_step(self, input_prompt, ground_truth_answer)`:\n  - Forward pass.\n  - Compute loss.\n  - Backpropagate only prompt/LoRA parameters.\n- `generate_response(self, input_prompt)`:\n  - Run model in inference mode.\n  - Return generated text.\n\nThis detailed logic ensures alignment with the paper’s description of prompt tuning and LoRA methods, supporting efficient, effective prompt optimization, and consistent answer generation."}

## retrieval.py

# Logic Analysis for `retrieval.py` — FAISS Index Construction and Search

## Purpose Overview
The `retrieval.py` module is responsible for managing the retrieval system within the G-Retriever pipeline. Its core functions include:
- Building an efficient nearest neighbor index for all graph node and edge embeddings (indexing).
- Conducting rapid similarity searches to find the most relevant nodes and edges given a query embedding (retrieval).

## Main Components and Responsibilities
1. **Index Construction (`build_index`)**
   - Input: A set of embeddings for all nodes and edges across the dataset.
   - Function:
     - Initialize a FAISS index suitable for cosine similarity.
     - Store all node and edge embeddings into the index.
   - Output: A fully constructed FAISS index ready for fast queries.

2. **Search Function (`search`)**
   - Input:
     - Query vector (`query_vec`): Embedding of the current question, of shape `(d,)`.
     - Top-k (`top_k`): Number of candidates to retrieve.
   - Function:
     - Use the FAISS index to find the `top_k` nearest neighbors to the query vector based on cosine similarity.
     - Return:
       - Top relevant node IDs or indices.
       - Corresponding similarity scores.
   - Output:
     - Two lists: one with indices (or IDs) of retrieved nodes/edges, and the second with their similarity scores.

## Implementation Details and Key Considerations

### 1. Embedding Space and Data Structures
- Embeddings are stored as numpy arrays with shape `(N, d)`, where `N` is total number of nodes + edges, and `d` is embedding dimension (e.g., 1024).
- Embeddings for nodes and edges are concatenated or stored separately, but more flexible to store them separately for clarity.
- Each embedding corresponds to an individual node or edge, identified by an ID or index.

### 2. FAISS Index Choice
- To support cosine similarity:
  - Use `faiss.IndexFlatIP` (inner product similarity).
  - Normalize all embeddings to unit vectors during indexing and search to align inner product with cosine similarity.
- Alternatively:
  - Use `faiss.IndexFlatL2` with cosine conversion (by transforming vectors).
  - But since the paper emphasizes cosine similarity, normalization is preferable.

### 3. Index Building Steps
- Normalize embeddings: `embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)`
- Instantiate FAISS index:
  ```python
  index = faiss.IndexFlatIP(d)
  ```
- Add normalized embeddings to index:
  ```python
  index.add(embeddings)
  ```
- Store the `index` object for subsequent search.

### 4. Search Functionality
- Normalize the query vector:
  ```python
  query_vec /= np.linalg.norm(query_vec)
  ```
- Use `search()`:
  ```python
  distances, indices = index.search(query_vec.reshape(1, -1), top_k)
  ```
- `distances` are inner products for normalized vectors, reflecting cosine similarity.
- Convert `distances` and `indices` to Python lists, returning:
  - list of indices or IDs (based on dataset indexing)
  - list of similarity scores (`distances`)

### 5. Additional Functional Aspects
- Keep track of which embeddings correspond to which node and edge IDs for meaningful retrieval results.
- During retrieval, may want to retrieve separate top-k lists for nodes and edges:
  - Maintain two indices or combine into a single index with labels.
- Ensure reproducibility by setting FAISS seed if necessary (optional, FAISS does not seed RAM random generators directly).

### 6. Hyperparameters and Configuration
- `top_k`: Retrieve parameter configured in `config.yaml` (default 5 for WebQSP, 3 for SceneGraphs).
- Embedding dimension: `d=1024` (from the Graph Transformer or text embedder output layer).
- Alignment with the main pipeline:
  - Embeddings should be precomputed during indexing.
  - Retrieval should accept a query embedding aligned with the dataset's embedding space.

### 7. Error Handling and Optimization
- Check that index has been built before searching.
- Handle cases where the number of stored embeddings is less than `top_k`.
- Optionally, support incremental additions to the index (not required but useful).

## Summary
- The `build_index` method:
  - Takes an array of all node/edge embeddings.
  - Normalizes them.
  - Builds a FAISS index with inner product similarity.
  - Stores reference to the index.

- The `search` method:
  - Normalizes the query embedding.
  - Performs FAISS search.
  - Returns the top `k` indices and their cosine similarity scores.

This logic ensures efficient, scalable retrieval aligned with the method described in the paper, critical for constructing relevant subgraphs via PCST in the overall GraphQA framework.

---

Would you like me to prepare pseudocode snippets or a detailed class API structure for this module next?

## train.py

**Logic Analysis for `train.py` — Implementation of the Trainer Class for Fine-tuning LLM with LoRA or Prompt Tuning**

---

### Purpose:
The `train.py` module defines the `Trainer` class, responsible for fine-tuning the large language model (LLM) (e.g., Llama2-7B) on dataset samples, specifically tuning the parameters of prompt tokens or LoRA modules, while keeping the core model weights frozen. It manages training loop, optimization, and logging, ensuring that the model learns to generate accurate answers based on the input prompts, including graph information, in a reproducible and efficient manner.

---

### Core Responsibilities:
1. **Initialization & Setup**
   - Instantiate the `Trainer` with model, dataset, hyperparameters, and optimization settings.
   - Prepare data loaders for training, validation, and test splits.
   
2. **Prompt Preparation**
   - For prompt tuning:
     - Generate initial soft prompt tokens (e.g., 10 tokens).
     - Prompt tokens are learnable parameters initialized randomly or via specific strategy.
   - For LoRA fine-tuning:
     - Integrate LoRA modules into the frozen base model.
     - Freeze core model weights, allow only LoRA parameters to update.
   
3. **Training Loop**
   - Iterate over epochs:
     - For each batch:
       - Prepare input prompts: combine textual graph description, question, and soft prompt tokens.
       - Forward pass:
         - Input prompt into the LLM.
         - Generate output distribution.
       - Compute the loss:
         - Cross-entropy loss comparing generated tokens to ground-truth answer.
       - Backpropagation:
         - Only update prompt parameters (prompt tuning) or LoRA modules.
       - Optimization step:
         - Use AdamW optimizer for local updates.
     - Apply learning rate decay (cosine schedule).
     - Handle early stopping (based on validation performance or patience).
   
4. **Hyperparameters & Configurations**
   - Learning rate: 1e-5 for prompt tuning or LoRA modules.
   - Batch size: 16, as per config.
   - Epochs: 10.
   - Prompt length: 10 tokens.
   - Max input tokens: 512 (truncate or pad prompts).
   - Prompt learning rate: 1e-5.
   - Edge cost, top-k retrieval: as specified.
   
5. **Logging & Checkpointing**
   - Save model checkpoints periodically and at best validation performance.
   - Log training/validation loss, accuracy, and other metrics.
   
6. **Reproducibility**
   - Use deterministic settings where possible.
   - Initialize random seeds.
   - Record hyperparameters and environment details.

7. **Special Handling**
   - When using LoRA:
     - Apply LoRA modules to the LLM's attention layers identified in configuration.
     - Optimize only LoRA parameters.
   - When using prompt tuning:
     - Initialize prompt token embeddings as learnable parameters.
     - Optimize only prompt token embeddings, keeping the rest of the model frozen.
   - Loss calculation:
     - Use cross-entropy loss on the generated token sequence.
   - Learning rate scheduling:
     - Use cosine decay (or similar) as per config.

8. **Training Data Handling**
   - Dataset consists of input prompts (textualized graphs + question) with target answers.
   - Provide properly tokenized inputs (via the tokenizers from transformers).
   - Masking or padding input sequences to max length (512 tokens).
   
9. **Model Saving & Loading**
   - Save the trained prompt/token embeddings and LoRA modules separately.
   - Keep a record of best model state based on validation metrics.
   
10. **Implementation Details & Best Practices**
    - Use `torch.nn.Module` and `torch.optim.AdamW`.
    - Ensure prompt tokens are registered as parameters (`nn.Parameter`) for prompt tuning.
    - For LoRA:
      - Use a LoRA implementation compatible with HuggingFace transformers or custom modules.
    - Gradients:
      - Set `requires_grad` to False for core model weights.
      - Only enable gradients for prompt tokens or LoRA modules.
    - Use `torch.no_grad()` and gradient clipping if necessary.
    - Maintain reproducibility with manual seed setting at start.

---

### Methods & Class Structure:

```python
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        # Store model, datasets, hyperparameters
        # Initialize optimizer, scheduler
        # If prompt tuning: create prompt token embeddings
        # If LoRA: integrate LoRA modules into the model
        # Set device and seed

    def train(self):
        # Main training loop over epochs
            for epoch in range(self.config['training']['epochs']):
                self.train_one_epoch()
                self.validate()
                # Save checkpoints, early stopping

    def train_one_epoch(self):
        # Iterate over batches
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            inputs, targets = self.prepare_inputs(batch)
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            # Optional: scheduler step

    def prepare_inputs(self, batch):
        # Tokenize input prompt + context (graph + question)
        # Handle prompt tokens, padding
        return inputs, targets

    def compute_loss(self, outputs, targets):
        # Cross-entropy on next token prediction
        return loss

    def validate(self):
        # Run inference on validation data
        # Compute metrics (accuracy, hallucination metrics)
        # Update best model if improved

    def save_checkpoint(self, path):
        # Save model state dict, prompt tokens, LoRA parameters

    def load_checkpoint(self, path):
        # Load trained parameters
```

---

### Additional Considerations:
- Use `torch.device` appropriately.
- Maintain clear separation between model inference and training modes.
- Manage gradient accumulation if batch size is constrained.
- Implement logging of training progress and evaluation metrics.
- Support for both prompt tuning (learnable soft prompt) and LoRA (parameter-efficient fine-tuning).

---

### Summary:
The `train.py`'s `Trainer` class orchestrates all training activities, focusing on prompt or LoRA parameter updates, avoiding updating the entire model parameters. It manages the tokenization, data batching, loss computation, optimizer updates, and model saving. Hyperparameters align strictly with the provided `config.yaml`, with flexible support for different tuning modes. The implementation ensures reproducibility, efficiency, and modularity for effective fine-tuning aligned with the methodology in the paper.

---

Would you like me to proceed with drafting the class implementation now?

## utils.py

# Logic Analysis for utils.py

This file provides core utility functions necessary across multiple modules of the G-Retriever framework, including data processing, similarity calculations, normalization, visualization, and configuration management. The functions facilitate modularity, reusability, and clarity. Below is a comprehensive analysis of the required functions, their logic, input/output specifications, and implementation considerations, aligned with the overall system design.

---

## 1. Cosine Similarity Function

### Purpose:
Calculate the cosine similarity between two vectors or two sets of vectors, with support for batch processing. It is a core component for retrieval (via FAISS or manual similarity), subgraph prize assignment, and any other similarity-based scoring.

### Inputs:
- `vec1`: numpy.ndarray or torch.Tensor
- `vec2`: numpy.ndarray or torch.Tensor
- `epsilon`: float, small value added for numerical stability (default: 1e-6)

### Outputs:
- similarity: float (if both are 1D vectors)
- similarity vector: numpy.ndarray or torch.Tensor (if either input is batch and supports multiple vectors)

### Logic:
-Normalize `vec1` and `vec2` independently:
  - Divide each vector by its L2 norm + epsilon.
- Compute dot product of normalized vectors.
- Return the dot product as cosine similarity, which lies in [-1, 1].

### Implementation notes:
- Support both single vectors and batches.
- Use numpy or torch based on the common convention; in this code, prefer numpy, as it aligns with other parts.
- Ensure consistent handling of data types.

---

## 2. Data Normalization Function

### Purpose:
Normalize feature vectors (e.g., embeddings) to unit length to ensure cosine similarity is meaningful and consistent, especially after encoding.

### Inputs:
- `vectors`: numpy.ndarray
- `axis`: int (default: 1), axis along which to normalize
- `epsilon`: float, small value for numerical stability

### Outputs:
- normalized_vectors: numpy.ndarray

### Logic:
- Compute the L2 norm along the specified axis.
- Divide each vector by its norm + epsilon.
- Return normalized vectors.

---

## 3. Visualization Helpers (Optional)

### Purpose:
Provide functions for visualizing graphs, embeddings, or retrieved subgraphs. For example, plotting a graph with highlighted subgraph or embedding points.

### Functions:
- `plot_graph(graph: networkx.Graph, highlighted_nodes: list = None, title: str = "")`:
  - Render a networkx graph with optional highlighting.
  - Use matplotlib for visualization.

### Logic:
- Draw nodes and edges.
- Highlight specified nodes/edges in color.
- Add labels and title.
- Show figure.

### Implementation considerations:
- Support for saving plots.
- Support for different layouts.

---

## 4. Configuration Parsers

### Purpose:
Load and parse the provided YAML configuration file (`config.yaml`). This is essential for ensuring all training and evaluation hyperparameters are consistent and accessible.

### Functions:
- `load_config(filepath: str) -> dict`:
  - Reads YAML file.
  - Returns a dictionary of parameters.

### Logic:
- Use the `yaml` library (if used; else, implement minimal parser).
- Validate presence of key parameters.
- Provide access to configuration for other modules.

---

## 5. Additional Utility Functions (Optional)

Depending on system needs, include:
- Logging setup for debugging.
- Numerical stability functions.
- Text processing helpers for textual graph representations, e.g., cleaning, tokenization helpers.

---

## Summary of functions:

| Function Name | Purpose | Inputs | Outputs | Notes |
|-----------------|---------|--------|---------|--------|
| `cosine_similarity(vec1, vec2, epsilon=1e-6)` | Compute cosine similarity between vectors | vectors | float or numpy.ndarray | Support batch processing |
| `normalize(vectors, axis=1, epsilon=1e-6)` | Normalize vectors for cosine | vectors | normalized vectors | Batch-compatible |
| `plot_graph(graph, highlighted_nodes=None, title="")` | Visualize graphs | networkx.Graph, list of node ids | matplotlib figure | Optional for debugging/visualization |
| `load_config(filepath)` | Load YAML config | string filepath | dict | Parse and validate YAML config |

---

## Implementation considerations:
- Use numpy for numerical calculations for efficiency.
- Keep code clean, support batch operations.
- Ensure support for both CPU and GPU tensors if torch is involved.
- Encapsulate small, repeated tasks, e.g., norm calculation, for reuse.
- Design with default parameters aligned with experiment settings.

---

This thorough logic analysis ensures `utils.py` will contain robust, flexible, and clear utility functions that support the entire G-Retriever pipeline, from data processing to model evaluation.

