# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

{
  "dataset_loader.py": "The DatasetLoader class is responsible for importing and processing the datasets for Wikipedia and arXiv, as specified in the paper. Its primary functions are load_wikipedia() and load_arxiv(), each of which must load, parse, and structure large-scale textual and ontological data into Dataset objects. The Dataset object should contain 'documents', 'concepts', 'relations', and 'annotations' (document-to-concept mappings).\n\n**Key logical steps:**\n\n1. **Data Acquisition:**\n   - For Wikipedia:\n     - Download or access a Wiki dump (e.g., via the Wikipedia API or dump files). \n     - Extract relevant categories starting from the 'Main topic classifications' root, performing a BFS traversal up to depth 3, as in the paper.\n     - For each category (concept), retrieve up to 5000 page titles and summaries.\n   - For arXiv:\n     - Access the dataset via its provided URL or download (from arXiv's dataset archive).\n     - Filter papers from 2020–2022 with ≥10 citations.\n     - For each paper, extract the title, abstract, and assign concepts based on the arXiv taxonomy.\n\n2. **Data Parsing and Structuring:**\n   - Parse the raw data into structured format:\n     - Document: plain text (title + abstract/summary).\n     - Concepts: strings representing concept labels.\n     - Relations: taxonomic relations (e.g., parent-child, 'subclass of') between concepts.\n     - Document annotations: map each document to one or more concepts.\n   - Handle large datasets efficiently via batching or streaming to avoid memory overload.\n\n3. **Ontology and Concept Extraction:**\n   - For Wikipedia:\n     - Build concept list from categories visited in BFS.\n     - Store parent-child relations between categories.\n   - For arXiv:\n     - Use arXiv taxonomy or keywords to assign concepts.\n     - Map documents to concepts based on metadata.\n\n4. **Data Structures and Storage:**\n   - Store documents as a list of strings.\n   - Store concepts as a list of strings.\n   - Store relations as a list of triples (concept1, relation_type, concept2).\n   - Store annotations as a dictionary mapping document IDs to concept lists.\n\n5. **Efficiency and Scalability:**\n   - Use generators or streaming when loading large datasets.\n   - Utilize multi-processing if necessary for large-scale data pre-processing.\n   - Cache datasets to avoid re-loading.\n\n6. **Output:**\n   - Return Dataset objects encapsulating all loaded info.\n   - Ensure compatibility with downstream components: e.g., datasets should support easy access to documents, concepts, relations, annotations.\n\n7. **Validation and Integrity Checks:**\n   - Verify no duplicate concepts or relations.\n   - Check that document annotations contain only concepts present in the concept list.\n   - Validate that relations are correctly formatted and bidirectionally consistent where applicable.\n\n**Additional considerations:**\n- The structure should support usage of large datasets without excessive memory consumption.\n- Define clear data formats: e.g., JSON, CSV, or custom structured files.\n- Document and implement error handling for missing data, malformed entries, or API failures.\n- Maintain code modularity: separate functions for data extraction, parsing, and validation.\n\n**Summary:**\nThe load_wikipedia() and load_arxiv() methods will each perform dataset downloading or loading, BFS traversal for Wikipedia categories, text extraction for documents, concept and relation extraction, and creation of a Dataset object, optimized for handling large datasets efficiently with batching and streaming where necessary. Proper validation checks will be embedded to ensure data integrity, aligning with the experimental setup described in the paper."
}

## evaluation.py

{
  "evaluation.py": [
    {
      "name": "graph_f1_score",
      "purpose": "Compute the similarity score between the generated ontology graph and the ground truth ontology using a node and edge matching approach based on embeddings.",
      "logic": [
        "Input parameters: model_graph (Graph), true_graph (Graph); both contain nodes, edges, and weights.",
        "Embed concepts (nodes) from both graphs: for each concept node, extract its textual label and compute its embedding using the sentence-transformers model (e.g., MiniLM-L6-v2).",
        "Construct node embedding matrices for both graphs: node_embeddings_true, node_embeddings_pred.",
        "Calculate pairwise cosine similarity between all node embeddings in true and predicted graphs.",
        "Apply the Hungarian algorithm from scipy.optimize to find the maximum similarity matching between nodes in ground-truth and predicted graphs.",
        "Determine the node matching: for each matched pair, note the similarity score.",
        "Calculate total similarity scores for matched nodes: sum the similarity scores for the matched node pairs (s_node).",
        "Match edges: for each edge in predicted graph, find a corresponding edge in true graph based on the embeddings of source and target nodes; compute similarity accordingly.",
        "Edge matching: use the node matchings to compare edges. For each edge in the predicted graph, compare with edges in the true graph, compute similarity (e.g., via endpoint embeddings), and find the best matching edges again with Hungarian algorithm.",
        "Compute edge similarity score: sum of similarities over matched edges (s_edge).",
        "Calculate the overall precision and recall:"
        "   - Precision: s_node / total number of predicted nodes, or based on matched nodes.",
        "   - Recall: s_node / total number of true nodes, or based on matched nodes.",
        "Similarly for edges: compute a weighted F1 based on s_edge and total edges.",
        "Return the F1 score combining node and edge matchings, possibly as harmonic mean: F1 = 2 * precision * recall / (precision + recall).",
        "Ensure normalization of similarity scores between -1 and 1, and consider only positive similarity (e.g., threshold at 0.436)."
      ],
      "notes": "Use current best practices for node and edge matching: embeddings via SentenceTransformer, Hungarian for maximum bipartite matching, normalization to ensure comparability, and handling unmatched nodes/edges gracefully."
    },
    {
      "name": "motif_distance",
      "purpose": "Quantify the structural difference between the true and generated graphs by comparing the distribution of 3-node motifs.",
      "logic": [
        "Input parameters: model_graph (Graph), true_graph (Graph).",
        "Extract all 3-node subgraphs in true_graph and model_graph: enumerate all possible triples of nodes and count the specific motif types (e.g., feedforward, cycle, chain).",
        "Count occurrences of each motif type in each graph: motif_counts_true, motif_counts_pred.",
        "Normalize these counts by total number of 3-node subgraphs (or total counts per motif) to obtain distributions: p_true, p_pred.",
        "Calculate the total variation distance (TVD):"
        "   - For each motif type m, compute: |p_true[m] - p_pred[m]|.",
        "Sum over all motif types to get the total variation: motif_distance = 0.5 * sum(|p_true[m] - p_pred[m]|).",
        "Record the motif distance, which ranges from 0 (identical motif distributions) to 1 (completely different)."
      ],
      "notes": "Ensure motif enumeration is efficient; for large graphs, sample motif counts or use approximate motif counting algorithms. Focus on 3-node motifs for simplicity as per paper."
    },
    {
      "name": "semantic_similarity",
      "purpose": "Estimate the semantic similarity between two concepts based on their textual labels using sentence-transformers.",
      "logic": [
        "Input: concept label 1 and concept label 2 as strings.",
        "Pre-compute embeddings of these labels using a pretrained sentence transformer (e.g., MiniLM-L6-v2).",
        "Calculate cosine similarity between the two embeddings: cosine_sim = (A · B) / (||A|| * ||B||).",
        "Return the cosine similarity score, which ranges from -1 (opposite meanings) to 1 (identical).",
        "Compare each pair of concepts during node and edge matching procedures, using the threshold of 0.436 as a cutoff for considering a match as semantically similar enough.",
        "Use these similarity scores to determine matched pairs in node matching and edge matching via Hungarian algorithm for the best overall match."
      ],
      "notes": "Store concept embeddings to avoid recomputation. Use normalized cosine similarity for consistency."
    },
    {
      "name": "node_embedding",
      "purpose": "Generate a fixed-dimensional vector embedding for each concept node based on its label for use in similarity computations.",
      "logic": [
        "Input: concept label (string).",
        "Use pretrained sentence-transformer model (e.g., MiniLM-L6-v2).",
        "Tokenize and encode label to obtain dense vector embedding.",
        "Normalize embedding (e.g., L2 normalization) if needed for cosine similarity.",
        "Return the embedding as a numpy array.",
        "Optional caching of embeddings for efficiency, especially when using large graphs.",
        "Use these embeddings in graph matching algorithms to compare nodes and their neighborhoods."
      ],
      "notes": "Ensure embedding process matches the loading of the model; load the sentence-transformer once and reuse for all concepts."
    }
  ],
  "Additional considerations": [
    "For all methods, validate inputs for consistency (e.g., no missing concept labels).",
    "Implement batch processing for embedding similarity calculations to improve efficiency.",
    "Handle edge cases where graphs are empty or have no matching nodes/edges gracefully.",
    "Provide options for visualization and detailed reporting of matchings and motif distributions.",
    "Design the functions to be modular, so they can be called independently for different validation or analysis purposes."
  ],
  "Summary": "The 'evaluation.py' module will contain four main functions/methods, each following a logical flow as above, ensuring the use of embeddings for robustness, the Hungarian algorithm for optimal matching, and appropriate distance or similarity measures to quantify graph similarity, structural divergence, and concept semantic alignment."
}

## graph_utils.py

{
  "Implementation approach": "In 'graph_utils.py', implement utility functions to handle graph construction, manipulation, pruning, and visualization using NetworkX. These functions will be designed to support the end-to-end ontology learning pipeline, including converting model outputs into graph structures, merging multiple subgraphs, applying pruning thresholds, removing cycles, and visualizing graphs. Each function should be stateless and accept data as parameters, returning updated graph objects. The functions will be used sequentially in the main pipeline to process raw relations into validated ontology graphs.",
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
classGraphUtils:
    + create_graph_from_relations(relations: List[Tuple[str, str, str, float]]) -> nx.DiGraph
        # Converts a list of relations with optional weights into a directed graph.
    + merge_graphs(graph_list: List[nx.DiGraph]) -> nx.DiGraph
        # Merges multiple graphs into a single graph, summing weights of duplicate edges.
    + prune_edges(graph: nx.DiGraph, alpha: float, beta: float) -> nx.DiGraph
        # Applies threshold pruning based on alpha (global) and beta (local relative).
    + remove_self_loops(graph: nx.DiGraph) -> nx.DiGraph
        # Removes all edges where u == v.
    + remove_inverse_edges(graph: nx.DiGraph) -> nx.DiGraph
        # Enforces unidirectionality for bidirectional edges, keeping edges with higher weight.
    + remove_cycles(graph: nx.DiGraph, strategy: str='greedy') -> nx.DiGraph
        # Detects cycles and removes edges based on a strategy (default 'greedy') until acyclic.
    + visualize_graph(graph: nx.DiGraph, output_path: str, options: dict=None) -> None
        # Plot the graph with optional styling, saving to output_path.
",
  "Program call flow": "
sequenceDiagram
    participant G as GraphUtils
    participant D as DatasetLoader
    participant M as Model
    participant P as PromptGenerator
    participant E as Evaluation
    participant V as Visualization

    D->>G: create_graph_from_relations(raw_relations)
    G-->>V: visualize_graph(initial_raw_graph, 'initial.png')
    G->>G: merge_graphs(list_of_subgraphs)
    G-->>V: visualize_graph(merged_graph, 'merged.png')
    G->>G: prune_edges(merged_graph, alpha, beta)
    G-->>V: visualize_graph(pruned_graph, 'pruned.png')
    G->>G: remove_self_loops
    G-->>V: visualize_graph(no_self_loops_graph, 'noselfloops.png')
    G->>G: remove_inverse_edges
    G-->>V: visualize_graph(unidirectional_graph, 'unidirectional.png')
    G->>G: remove_cycles(strategy='greedy')
    G-->>V: visualize_graph(acyclic_graph, 'acyclic.png')
    G->>E: evaluate(acyclic_graph, ground_truth)
    E-->>V: output_metrics
    "
}

## hyperparameter_search.py

{
  "Implementation approach": "The script will perform systematic grid search over key hyperparameters: relation pruning thresholds (alpha, beta), the masking average relation count (M), and possibly other parameters like the number of samples or relation path length. For each combination, it will run the full validation pipeline: generate the ontology predictions on the validation set, compute the chosen evaluation metric (e.g., Graph F1), and record the results. The best hyperparameter set based on the maximum validation Graph F1 score will be selected and returned. The code will interface with the evaluation module to compute metrics, with the dataset loader to load validation data, and with the graph_utils to manage graph creation and pruning. It may also utilize the model module to generate subgraphs if needed during validation.",
  "File list": [
    "hyperparameter_search.py"
  ],
  "Data structures and interfaces": "The script will define a HyperparameterTuner class or function that takes as input a grid of hyperparameters (e.g., lists of candidate values for each). It will load validation datasets via DatasetLoader, generate ontology predictions using the model's generate_subgraph() function with current hyperparameters (sampling settings, prompt settings). It will then construct the assembled graphs, prune edges thresholds, remove cycles, and evaluate via evaluation.py's functions. Results will be stored as a list/dictionary: { 'alpha': value, 'beta': value, 'M': value, 'validation_score': float }. After all combinations are tested, the maximum validation_score will be identified, and the corresponding hyperparameters returned.",
  "Program call flow": "Start: Initialize hyperparameter grid and load validation data. For each combination in the grid:\n- Set hyperparameters: alpha, beta, M, etc.\n- Generate subgraphs for validation documents:\n  - For each document in validation set:\n    - Use model.generate_subgraph(prompt, sampling_params) with current hyperparameters.\n- Aggregate subgraphs into a full graph.\n- Apply post-processing: prune edges with alpha, beta thresholds; remove cycles.\n- Evaluate the resulting graph against ground-truth validation graph via graph_f1_score().\n- Record the metric score.\nAfter completing all grid points: identify hyperparameters with best validation score.\nReturn: best hyperparameter set.\nEnd."
}

## main.py

# Logic Analysis for main.py

The purpose of main.py is to serve as the system entry point that coordinates the entire ontology learning pipeline, based on the paper's methodology, the provided design, and configurations. It should implement the following high-level steps:

1. **Load configuration parameters** from the provided `config.yaml`.
2. **Load datasets** (Wikipedia and/or arXiv) via `dataset_loader.py`.
3. **Partition datasets** into training, validation, and test splits according to the protocol described (e.g., split top-level concepts, then assign subgraphs within maximum depth).
4. **Hyperparameter tuning**:
   - Use `hyperparameter_search.py` to perform grid search over thresholds (\(\alpha, \beta\)), masking parameter \(M\), and possibly other parameters, with validation via the `evaluation.py` metrics.
   - Select hyperparameters that optimize validation metrics, e.g., Graph F1.
5. **Initialize the Language Model**:
   - Instantiate `model.py`'s `BingModel` class with parameters:
     - Pretrained model: "mistral-7b-v0.2".
     - Use LoRA: True, rank: 32.
     - Load pretrained weights; optionally, further finetune on specific dataset (wikipedia/arXiv).
6. **Finetune the language model**:
   - Call `model.finetune()` with:
     - Training dataset (from step 2) generated with the annotated concept paths.
     - Epochs = 2.
     - Loss masking: True (due to the custom regularizer).
   - For transfer to arXiv:
     - Load checkpoint from prior finetuning.
     - Further finetune on 2048 document-subgraph pairs following the same approach.
7. **Subgraph generation**:
   - For each document in the dataset:
     - Generate the prompt via `prompt_generator.py`.
     - Call `model.generate_subgraph()` with:
       - The generated prompt.
       - Sampling parameters: temperature=0.1, top_p=0.9.
       - Max tokens: 512 (matching the paper's setting).
     - Parse generated output into graph edges and concepts.
8. **Aggregate all subgraphs**:
   - Use `graph_utils.py`:
     - Collect edges across all document outputs.
     - Sum weights: increment counts for edges appearing multiple times.
9. **Post-process the raw graph**:
   - Prune edges with thresholds \(\alpha, \beta\) as tuned:
     - Remove edges with weight below \(\alpha\).
     - Remove inverse edges with higher weight on mutual pairs.
     - Remove self-loops.
     - Remove nodes with no edges.
     - Remove cycles via the described greedy algorithm, if `cycle_removal_strategy` is "greedy."
10. **Evaluate the final graph**:
    - Load ground-truth ontology.
    - Use `evaluation.py`:
      - Compute semantic and structural similarities:
        - Graph F1
        - Motif Distance
        - Node embedding similarity.
      - Record evaluation metrics.
11. **Visualization**:
    - Use `visualization.py`:
      - Generate plots for the final ontology.
      - Create subgraph visualizations for qualitative analysis.
12. **Output results**:
    - Save final ontology graph to disk (e.g., as GraphML or adjacency list).
    - Save evaluation metrics to a log or output file.
    - Save visualizations.

---

# Detailed Step-by-step Logic

### Initialization
- Import necessary modules: `yaml`, dataset_loader, model, prompt generator, graph_utils, evaluation, visualization, hyperparameter_search.
- Load configuration from `config.yaml`:
  - Dataset details.
  - Model parameters.
  - Hyperparameters for pruning and training.
  - Sampling parameters.
  - Evaluation and visualization settings.

### Data Loading
- Instantiate `DatasetLoader`.
- Load datasets:
  - Call `load_wikipedia()` if working on Wikipedia experiments.
  - Call `load_arxiv()` for transfer experiments or arXiv domain adaptation.
- Split datasets into train/validation/test:
  - Follow the dataset split protocol:
    - Partition top-level concepts into train, validation, test splits.
    - Extract subgraphs within depth \( d \).
    - Generate document-concept annotations if not available.
- Store datasets in structured objects for downstream use.

### Hyperparameter Tuning
- Call `hyperparameter_search()`:
  - Input: parameter grid for \(\alpha, \beta, M, ...\).
  - Validation function: runs a small number of inference passes, then evaluates a validation subset.
  - Output: best hyperparameter set.

### Model Initialization
- Instantiate model:
  ```python
  model = BingModel(
      model_name=config['model']['base_model_name'],
      use_lora=config['model']['use_lora'],
      lora_rank=config['model']['lora_rank']
  )
  ```
- Load pretrained weights.

### Fine-tuning / Transfer Learning
- If training from scratch:
  - Call `model.finetune()` with training dataset:
    - Dataset: annotated concept graphs per document.
    - Epochs: 2.
    - Loss masking: true.
- If transfer learning:
  - Load previous checkpoint/model weights.
  - Fine-tune further on fewer arXiv document/subgraph pairs.

### Generating Subgraphs
- For each document:
  - Generate prompt (via prompt_generator.py).
  - Call `model.generate_subgraph()` with sampling parameters:
    - temperature=0.1, top_p=0.9, max_tokens=512.
  - Parse the text output into edges:
    - Extract concept pairs and relations via regex or explicit parser.
- Store generated subgraphs per document.

### Aggregation & Post-Processing
- Aggregate edges from all document subgraphs:
  - Increment edge counts/weights in `graph_utils`.
- Apply post-processing:
  - Prune edges with weight below \(\alpha\).
  - Remove inverse edges if bidirectional and with lower weight.
  - Remove self-loops.
  - Remove nodes without edges.
  - Remove cycles greedily if configured.
- Result: a cleaned, weighted ontology graph.

### Evaluation
- Load ground truth ontology graph.
- Compute:
  - Graph F1 score using node embeddings and Hungarian matching.
  - Motif distance for structural similarity.
  - Semantic similarity using concept embeddings.
  - Optional: visualize node correspondences.
- Save metrics to file.

### Visualization
- Generate graph plots:
  - For the final ontology.
  - For sample subgraphs around key concepts.
- Save figures.

### Final Output
- Save the ontology graph in a specified format.
- Save evaluation results.
- Save qualitative visualizations.

---

# Assumptions & Clarifications Needed
- Exact formats of dataset annotations for concepts and relations.
- Precise prompt template strings matching the paper's figures.
- Parsing logic for model output into graph edges.
- Range of hyperparameters for grid search.
- Thresholds for pruning (\(\alpha, \beta\)) based on validation.

This detailed logic analysis ensures that the implementation in main.py will effectively coordinate all components, follow the methodology, and produce reproducible, comparable results as per the paper.

## model.py

# Logic Analysis for `model.py`: `BingModel` Class

This module focuses on defining the core class `BingModel` responsible for initializing and fine-tuning the large language model (LLM), as well as generating subgraphs for ontology learning. Its design must follow the detailed methodology described in the paper, specifically:

- Using the pretrained LLM (e.g., Mistral 7B v0.2) from Huggingface.
- Incorporating LoRA modules if specified.
- Implementing a customized `finetune()` method that incorporates the masked loss regularizer described.
- Implementing `generate_subgraph()` for inference: prompting, sampling, decoding, and parsing output into graph relations.

This logic analysis breaks down each component, aligning with the paper’s methodology and the provided configuration.

---

# 1. Initialization (`__init__`)
- **Inputs**:
  - `model_name`: e.g., `"mistral-7b-v0.2"`.
  - `use_lora`: boolean indicating whether to add LoRA modules.
  - `lora_rank`: rank of LoRA modules (default=32).
  - `finetune_on_dataset`: indicates dataset (e.g., "wikipedia" or "arxiv") for dataset-specific configurations.
- **Actions**:
  - Load the pretrained model from Huggingface Transformers (`AutoModelForCausalLM.from_pretrained()`).
  - Optionally wrap with LoRA:
    - Use PEFT or similar library to add LoRA attention/ feed-forward modules with specified `lora_rank`.
  - Initialize tokenizer associated with the model.
  - Store hyperparameters for training, such as learning rate, loss masking parameters, sampling parameters (temperature, top_p), and maximum tokens.
- **Outputs**:
  - Fully initialized model object with necessary hooks for training and inference.

# 2. Finetuning (`finetune`)
- **Inputs**:
  - `train_data`: dataset object, containing:
    - documents,
    - concepts,
    - relations,
    - annotations.
  - `epochs`: e.g., 2.
  - `loss_masking`: True/False for custom regularizer.
- **Actions**:
  - Prepare DataLoader:
    - Batch data (size=16 per `config.yaml`).
    - Each batch consists of paired input prompt + target output sequence.
  - For each batch:
    - Generate input tokens with a prompt (using `prompt_generator.py` templates) and document text + concept info.
    - Forward the batch through the model:
      - Compute cross-entropy loss over output tokens.
    - If `loss_masking`:
      - For each relation in the batch, compute its frequency `n`.
      - Determine masking probability: `max(1 - M/n, 0)`, where `M` is provided in config.
      - Randomly mask (set to ignore in loss) tokens corresponding to relations with high frequency, following the masking probability.
      - This reduces overfitting on frequent relations.
    - Backpropagate:
      - Apply gradient clipping (`gradient_clipping_norm`) if specified.
      - Update LoRA parameters (if used).
      - Save best model checkpoints based on validation metrics.
  - Use Adam optimizer with the specified learning rate.
  - Implement early stopping if validation loss/metric does not improve.
- **Outputs**:
  - Finetuned model weights saved internally or externally.
  - State suitable for inference.

# 3. generate_subgraph(prompt, max_tokens, sampling_params)
- **Inputs**:
  - `prompt`: string, constructed from document context/template.
  - `max_tokens`: e.g., 512 tokens.
  - `sampling_params`: dict with `temperature`, `top_p`.
- **Actions**:
  - Tokenize prompt using the model's tokenizer with necessary padding/truncation.
  - Generate output tokens using `model.generate()` with:
    - Temperature: 0.1
    - Top-p sampling: 0.9
    - Max tokens: as specified
    - No repetition penalties or top-k sampling unless explicitly set.
  - Decode generated tokens to text.
  - Parse generated text:
    - Extract concept relation triplets or paths following a structured format (see appendix).
    - Handle potential parsing errors.
- **Outputs**:
  - Text or structured relation data representing a subgraph (e.g., edges like `(concept1, relation_type, concept2)`).

# 4. Parsing Generated Text into Graph Relations
- Use regex or structured heuristics to identify relation triplets, paths, or relation words following the prompt template.
- Convert parsed triplets into a graph data structure (nodes, directed edges).
- Store confidence weights if available (e.g., link probability, embedding similarity) for later pruning.

# 5. Loss Function with Masking
- **Core idea**:
  - For each relation `u v` in training:
    - Count its frequency `n`.
    - Mask the tokens corresponding to `v` with probability `max(1 - M/n, 0)` during loss computation.
  - Maintain the tokens’ original position as context for the model.
- **Implementation**:
  - Use a masked cross-entropy loss:
    - During batching, create a mask tensor marking tokens to ignore.
    - Adjust the loss accordingly.
- **Note**:
  - This regularizer encourages the model to learn low-frequency relations better, reducing high-level concept overfitting, as explained in Section 5.1.

# 6. Hyperparameters
- Use the config.yaml parameters:
  - `learning_rate=1e-5`
  - `batch_size=16`
  - `epochs=2`
  - `loss_masking=True`
  - `relation_masking_M=100`
  - `mask_mask_prob=0.5`
  - `gradient_clipping_norm=1.0`
- During training, hyperp tune `alpha`, `beta`, `M` based on validation `Graph F1`.

# 7. Utility functions
- **`save_model()` and `load_model()`**:
  - Use `torch.save()` and `torch.load()`.
- **`calculate_loss_mask()`**:
  - For relation `u v`, determine probability for token masking dynamically.
- **`parse_output()`**:
  - Use regex or simple heuristic parsers to extract structured graph information from raw generated text.

# 8. Model interfaces summary
- **Initialization**:
  - Load pretrained (`transformers.AutoModelForCausalLM`) + optional LoRA modules.
- **Training**:
  - Batch loader yields `(input_ids, attention_mask, labels)` with masking applied.
- **Generation**:
  - Calls `model.generate()` with sampling parameters.
  - Post-process output to extract relation edges.
- **Data flow**:
  - Input: document + concept info, prompt.
  - Output: structured subgraph (edges/nodes) for aggregation.

---

# Summary
This detailed logic ensures the `BingModel` class adheres to the methods, regularization strategies, and generation paradigms described in the paper, facilitates efficient training with the custom loss, and supports flexible inference with prompt-based subgraph generation. The implementation will leverage the specified packages, obey the hyperparameters, and ensure modularity and reproducibility aligned with the design plan.

---

Would you like me to proceed with drafting code snippets implementing these components?

## prompt_generator.py

{
  "Implementation approach": "The prompt_generator.py module will define three functions: get_chain_of_thought_prompt(), get_direct_prompt(), and get_instruction_prompt(). Each function constructs a prompt string based on input parameters, including document text, concept sets, and predefined prompt templates from the configuration. The prompts must match the styles used during training and inference to ensure model consistency. The function get_chain_of_thought_prompt() will incorporate a detailed multi-step reasoning style, guiding the model to produce concept relation paths or subgraphs with explicit reasoning steps. get_direct_prompt() will directly instruct the model to generate the concept relations or subgraphs without reasoning steps, suitable for zero-shot or inference without reasoning. get_instruction_prompt() can provide general instructions or prompts used to initialize the model, possibly including task description, dataset specifics, or formatting guidelines, aligning with the paper's templates. The overall goal is ensuring prompt uniformity, clarity, and fidelity to the templates shown in the paper figures, thus promoting reliable generation and reproducibility.",
  "File list": [
    "prompt_generator.py"
  ],
  "Data structures and interfaces": "function get_chain_of_thought_prompt(document_text: str, concepts: list, additional_parameters: dict) -> str\nfunction get_direct_prompt(document_text: str, concepts: list, additional_parameters: dict) -> str\nfunction get_instruction_prompt(instruction_type: str, additional_parameters: dict) -> str\n\nInput parameters include document_text (string), concepts (list of strings), and optional configuration parameters such as prompt templates or task-specific info in additional_parameters. Output is a formatted string prompt matching the style of the templates provided in Appendix figures 6-8.\n\nThe functions should handle the replacement or insertion of variables into template placeholders, ensure line breaks and formatting stay consistent, and support customization through parameters like document snippets, concept lists, prompt style variations, or extra instructions.",
  "Program call flow": "The prompt generator functions are called within the main pipeline before model inference:\n- get_chain_of_thought_prompt() is invoked when multi-step reasoning prompts are needed, possibly during training or inference when explicitly reasoning about subgraphs.\n- get_direct_prompt() is used for straightforward concept relation generation, suitable for zero-shot or baseline experiments.\n- get_instruction_prompt() supplies generic or task instructions at the start of an inference session or as a system prompt.\nThese prompts are passed as input to the model.finetune() or generate_subgraph() functions, ensuring consistent and standardized input formatting as per the templates shown in the paper figures.",
  "Anything UNCLEAR": "The exact prompt text templates (the literal strings with placeholders) are not provided in the paper or appendix. Clarification is needed concerning the specific template syntax, placeholders, and formatting conventions used in Figures 6-8. For example, the structure of the prompts, how to include the document text, concepts, and reasoning steps, and whether to insert labels like ‘Relevance,’ ‘Relation Paths,’ or ‘Explain your reasoning,’ require precise definition. Additional clarification is necessary to replicate the prompt style exactly as intended, ensuring consistent model behavior during training and inference."
}

## trainer.py

{
  "trainer.py": "The Trainer class is responsible for orchestrating the training process of the OLLM model, handling data batching, loss computation, validation, hyperparameter validation, and model checkpointing. Its main roles include:\n\n1. Initialization:\n   - Accept configuration options, especially hyperparameters such as learning rate, batch size, number of epochs, loss masking parameters, and cycle removal strategy.\n   - Initialize the model instance, optimizer (e.g., Adam), and optionally learning rate schedulers.\n   - Prepare validation dataset and metric setup, primarily focusing on 'Graph F1' or chosen main validation metric.\n\n2. Data Loading and Batching:\n   - Receive a Dataset object containing training data: documents, concept annotations, and relevant subgraphs.\n   - Implement efficient batching, possibly with DataLoader or custom batching, to load multiple document examples per iteration.\n   - Ensure batch shuffling for stochastic gradient descent.\n\n3. Training Loop:\n   - For each epoch:\n     - Iterate over training batches.\n     - For each batch:\n       - Prepare inputs (tokenized documents and concepts) suitable for feeding into the pretrained large language model (using transformers tokenization API).\n       - Generate model outputs via model.generate() with specified sampling parameters (temperature=0.1, top_p=0.9, max_tokens=512).\n       - Parse outputs to extract predicted subgraph relations (edges, paths). This involves text parsing, regex matching, or structured extraction consistent with the linearisation schema.\n       - Compute the sequence loss between the model output tokens and the target sequence, applying the custom masked loss:\n         - For relations \(u v\) seen \(n\) times in dataset:\n           - With probability \( \max(1 - M / n, 0) \), mask the loss contribution for target tokens of \(v\), to reduce overfitting on high-frequency relations.\n           - Implement this masking within loss calculation (e.g., setting loss for masked tokens to zero).\n       - Backpropagate the loss.\n       - Apply gradient clipping if specified (% e.g., norm=1.0).\n       - Optimizer step and zero gradients.\n     - Record epoch losses and metrics.\n     - Save model checkpoint periodically.\n\n4. Validation:\n   - At the end of each epoch, evaluate the model on the validation set:\n     - Generate subgraphs for validation documents using the same pipeline.\n     - Prune edges with hyperparameters (\(\alpha, \beta\)) that were tuned via grid search.\n     - Remove cycles if necessary.\n     - Compute validation metrics:\n       - Use graph-based metrics such as 'Graph F1', 'Motif Distance', and semantic similarity.\n     - Compare validation metrics to determine early stopping or model selection.\n\n5. Hyperparameter Tuning:\n   - If integrated, perform grid search over hyperparameters (\(\alpha, \beta, M, learning_rate\)):\n     - For each hyperparameter combination:\n       - Run a training session (or partial, for validation only).\n       - Record validation metrics.\n     - Select the hyperparameters yielding the best main validation metric (e.g., 'Graph F1').\n\n6. Model Saving and Loading:\n   - Save the best model when validation performance improves.\n   - Load model from checkpoint for inference or further transfer finetuning.\n\n7. Additional Utilities:\n   - Implement early stopping based on validation metric stagnation or degradation.\n   - Handle logging, error handling, and reproducibility (using fixed seed, deterministic operations).\n\n8. Implementation specifics:\n   - Use the transformers library for tokenization and generation.\n   - Use torch for training, with optimizer Adam, possibly with custom learning rate scheduler.\n   - Implement the loss masking as a custom PyTorch loss function or within the training loop.\n   - For cycle removal post-processing, invoke functions from graph_utils.py.\n\nThis logic ensures that the 'trainer.py' script fully supports the training lifecycle, aligns with the hyperparameters provided in 'config.yaml', and facilitates reproducible model finetuning with the custom regularization strategy described in the paper."
}

## visualization.py

{
  "visualization.py": [
    {
      "Purpose": "Provide functions to visualize ontologies (graphs) with edges colored or styled based on confidence or similarity scores, aiding qualitative analysis.",
      "Dependencies": [
        "networkx",
        "matplotlib.pyplot",
        "numpy"
      ],
      "Key Inputs": [
        "Graph object: an instance of the Graph class or networkx DiGraph with nodes and edges",
        "Optional parameters for visualization such as node positions, edge labels, node labels, title, output file path"
      ],
      "Core Functions": [
        {
          "function_name": "visualize_graph",
          "description": "Create a visual representation of the ontology graph, highlighting edge confidence or similarity scores.",
          "inputs": [
            "graph: Graph object containing nodes and edges",
            "output_path: str (path to save the figure, e.g., PNG or PDF)",
            "title: str (title for the plot)",
            "node_labels: dict (optional, mapping node to label)",
            "edge_labels: dict (optional, mapping edge to label or score)",
            "edge_colors: dict (optional, mapping edge to color, to override default coloring based on scores)"
          ],
          "outputs": [
            "A saved figure file at output_path"
          ],
          "process": [
            "Convert the Graph object into a networkx DiGraph if necessary.",
            "Determine positions for nodes (e.g., using spring layout or hierarchical layout).",
            "Draw nodes with labels.",
            "Compute edge colors based on confidence or similarity scores:",
            "   - Normalize scores to [0,1]",
            "   - Map scores to a colormap (e.g., for high confidence, color in blue; for low, in gray or red).",
            "Draw edges with styles or widths reflecting scores.",
            "Add edge labels if provided, representing confidence or similarity.",
            "Add title and display or save the plot."
          ],
          "Notes": [
            "The visualization should be flexible to handle directed graphs, emphasizing directionality if relevant.",
            "Color maps and thresholds should be configurable for clarity."
          ]
        },
        {
          "function_name": "plot_subgraph",
          "description": "Visualize a subgraph centered around specific seed nodes, optionally highlighting certain nodes or edges.",
          "inputs": [
            "graph: Graph object",
            "center_nodes: list of nodes to focus on",
            "radius: int (how many hops from center to include in subgraph)",
            "output_path: str",
            "title: str"
          ],
          "outputs": [
            "Saved visualization file"
          ],
          "process": [
            "Extract the subgraph within the specified radius from center_nodes.",
            "Use layout algorithms suitable for small subgraphs.",
            "Color nodes: highlight centers, other nodes differently.",
            "Color or style edges based on scores, as with visualize_graph.",
            "Render and save."
          ]
        }
      ],
      "Implementation considerations": [
        "Color mapping: use colormap (matplotlib colormap) to translate scores to colors, e.g., 'viridis' or 'coolwarm'.",
        "Edge width can reflect confidence: thicker for higher confidence scores.",
        "Handle directed edges with arrows.",
        "Add legends or colorbars if necessary for interpretability.",
        "Ensure layout stability for comparison across visualizations.",
        "Design should be modular enough to visualize different aspects: edge confidence, structural motifs, or node similarity."
      ],
      "Hyperparameters": {
        "layout_algorithm": "spring_layout" or "spectral_layout",
        "colormap": "'viridis' or custom",
        "edge_width_scale": "linear mapping from scores to width",
        "node_size": "fixed or based on degree”
      },
      "Additional notes": [
        "The function should support high-resolution output, e.g., dpi=300.",
        "Color mapping should handle score normalization carefully, considering any min/max or percentile-based thresholds."
      ],
      "In summary": "Visualization.py will offer at least one public function `visualize_graph()` that takes a Graph object and optional scoring info, renders a clear, informative graph with edges styled to reflect their confidence or similarity scores, and saves the figure for analysis. Additional helper functions may be implemented for specialized visualizations such as local subgraphs or labeled graphs."
    }
  ],
  "Anything UNCLEAR": "Exact details of the graph data structure (whether nodes are strings or objects, how edges with scores are stored), preferred visualization style (e.g., directed arrows, color schemes), and whether to visualize edge labels or only colors need clarification. Also, whether the graph is weighted and directed, as the paper discusses confidence scores and structural similarity, which could impact the rendering approach."
}

