# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## data_generator.py

{
  "data_generator.py": "The primary role of this module is to convert a given ReasoningGraph object into a natural language data point (question, reasoning, options, answer) using a prompt template, send it to an LLM API for generation, and return the generated data. The logic must ensure the process adheres to the specifications, supports multiple dataset types, and allows flexible prompt templates. The key components of this class are initialization, prompt preparation, API call, and response parsing.\n\n**1. Initialization (`__init__`):**\n- Accept configuration parameters, especially the prompt template and API key.\n- Load the prompt templates from external files as configured (`reasoning_graph_generation`).\n- Store API credentials and default parameters (temperature, max_tokens).\n- Validate or set defaults for parameters.\n\n**2. Prompt Preparation (`construct_prompt`):**\n- Input: a ReasoningGraph object.\n- Strategy:\n  - Convert the graph into a string representation suitable for prompt insertion.\n  - Use a structured template with placeholders, filling in the graph data.\n  - Ensure the prompt guides the LLM to generate a question, reasoning chain, options, and answer closely aligned with the graph.\n- Handle different dataset formats by passing in dataset-specific prompts if necessary.\n- The output should be a string prompt that guides LLM to generate the desired response.\n\n**3. Graph to Text Generation (`generate_text`):**\n- Input: ReasoningGraph object.\n- Process:\n  - Call `construct_prompt()` to generate the prompt string.\n  - Send an API request (`openai.Completion.create()` or similar), with configured hyperparameters.\n  - Handle API errors/exceptions (rate limits, timeouts).\n  - Use retries if needed, especially if the response format is not as expected.\n- Output: raw generated text from API, ideally structured in a consistent format (e.g., question, reasoning, options, answer).\n\n**4. Response Parsing (`parse_response`):**\n- Input: raw API response text.\n- Strategy:\n  - Extract key components using regex, delimiters, or parser specific to the prompt template.\n  - Validate the extracted question, reasoning, options, and answer.\n  - Normalize the data (e.g., removing extraneous text, whitespace).\n  - Handle cases where parsing fails (log error, possibly attempt re-generation).\n- Output: a structured data point, e.g., a dictionary or an instance of DataPoint with all fields filled.\n\n**5. End-to-End Generation (`generate_data_point`):**\n- Input: ReasoningGraph object.\n- Steps:\n  - Call `generate_text()`.\n  - Parse response with `parse_response()`.\n  - Return DataPoint object or similar structured object compatible with downstream processing.\n\n**6. Modularity & Flexibility:**\n- The design should accommodate multiple prompt templates for different datasets.\n- Support optional parameters for temperature, max_tokens, retries.\n- Implement logging for traceability.\n- Support for batch processing if needed, though likely single calls per graph for fidelity.\n\n**7. Integration points:**\n- Dependence on external prompt templates stored in files (`reasoning_graph_generation`).\n- API integration via openai SDK.\n- The class must handle API key securely.\n- Validation of generated output before acceptance (possibly via an external verification step). \n\n**Summary:**\n- Initialize with config and templates.\n- Convert graph to prompt.\n- Call API with prompt and parameters.\n- Parse API output into structured format.\n- Return data point for dataset assembly.\n\nThis logical flow ensures the DataGenerator class faithfully translates reasoning graphs into high-quality, contextually relevant natural language datasets, consistent with the methodology described in the paper, and supports dataset-specific adaptations and robustness."

## dataset_loader.py

# Logic Analysis for dataset_loader.py

## Overview and Purpose
The `dataset_loader.py` module implements the `DatasetLoader` class, which handles the loading of static datasets (e.g., GSM8K, BBQ, BBH Dyck) from predefined file paths, parsing raw data into structured `DataPoint` objects, and saving new or perturbed data points post-augmentation. It should support flexible data formats, ensure reproducibility, facilitate data validation, and be compatible with subsequent modules. This class is critical for establishing a reliable foundation for reasoning graph extraction and dataset augmentation within the overall DARG pipeline.

---

## Core Responsibilities
- Load datasets from specified paths, which are usually in JSON or similar structured formats.
- Parse raw data entries into `DataPoint` objects, ensuring correct mapping of each data component:
  - `question_text`: The textual question or prompt.
  - `answer`: The correct answer or label.
  - `reasoning_text`: Explanation or reasoning chain, if available.
  - `reasoning_graph`: Structured object representing the reasoning structure (may initially be empty or inferred).
- Save generated or perturbed data points back to disk, supporting JSON serialization.
- Validate dataset entries against schema or format standards (to prevent parsing errors).
- Support data sampling, subset creation, and ensuring reproducibility via consistent file I/O.

---

## Input Data Formats
1. **Original datasets**:
   - Commonly in JSON format, where each entry is a dictionary containing at least: `"question"`, `"answer"`, `"reasoning"`, `"graph"` (optional).
   - Example:
     ```json
     {
       "question": "...",
       "answer": "...",
       "reasoning": "...",
       "reasoning_graph": {...}
     }
     ```
   - Metadata such as dataset splits (train/test) may be embedded or managed externally.
2. **Perturbed/generated data**:
   - Saved in similar JSON format, possibly with additional attributes indicating perturbation level or complexity.
   - The class should support appending new data points with consistent schema.

3. **Auxiliary files**:
   - Supporting files (e.g., schema definitions, validation schemas, prompt templates) may be stored separately or embedded within code.

---

## Output Data Formats
- **Loaded Data**:
  - Represented internally as a list of `DataPoint` objects.
  - Each `DataPoint` object encapsulates the fields listed above.
- **Saved Data**:
  - Serialize list of `DataPoint` objects back into JSON files matching input structure.
  - Optionally include additional metadata (complexity levels, perturbation parameters).

---

## Functional Tasks & Implementation Details

### 1. Initialization (`__init__`)
- Accept dataset path (JSON or other format).
- Load dataset file into memory, parse entries into `DataPoint`.
- Maintain internal storage (e.g., `self.data_points: List[DataPoint]`).

### 2. `load_data()`
- Read dataset file.
- Deserialize JSON into dictionaries.
- Validate each entry:
  - Check required fields (`question`, `answer`, `reasoning`, `reasoning_graph`).
  - Ensure data types are correct (strings, dicts).
- Instantiate `DataPoint` objects with parsed data.
- Handle exceptions:
  - Missing fields: log warning/error, decide whether to skip or raise.
  - Malformed JSON: handle gracefully.
- Return list of `DataPoint` objects.

### 3. `save_dataset(output_path)`
- Serialize current `DataPoint` objects into JSON.
- For each `DataPoint`, use `to_json()` method.
- Write JSON array to file at `output_path`.
- Ensure proper formatting (indentation, encoding).

### 4. Data Validation & Schema Checking
- Optional: integrate JSON Schema validation to verify data consistency.
- Use `jsonschema` package (if applicable).
- Validation should confirm:
  - presence and correct format of all required fields.
  - type correctness (strings, objects).
  - schema compliance for `reasoning_graph`.

### 5. Data Sampling & Subset Creation
- Methods to select subsets (e.g., for evaluation).
- Support for random sampling or deterministic subset selection (using seed).

### 6. Data Point Class (`DataPoint`)
- Fields:
  - `question_text`, `answer`, `reasoning_text`, `reasoning_graph`.
- Methods:
  - `to_json()`: serialize into dict compatible with JSON.
  - `from_json()`: create an instance from dict.

### 7. Additional Support
- Log progress, errors.
- Optional: support de-duplication, filtering, and annotation checks.
- Compatibility with other modules (e.g., reasoning graph extraction).

---

## Data Handling & Error Management
- Use try-except blocks when reading/parsing.
- Validate each entry against schema before adding.
- In case of errors, decide whether to skip entry or raise exception.
- During save, ensure the output file structure matches original format.

---

## Reproducibility & Consistency
- Support consistent random seed for sampling.
- Support loading datasets from specific versions or splits.
- Maintain a versioning system or naming conventions for dataset versions (original, augmented, perturbed).

---

## Summary of Key Implementation Points
- **Input**: Dataset path (e.g., `path/to/gsm8k_dataset.json`)
- **Process**:
  - Load dataset
  - Parse entries into `DataPoint`
  - Validate data
- **Output**:
  - List of `DataPoint` objects
  - Save method for JSON/dictionary output
- **Edge Cases**:
  - Missing or malformed reasoning graphs
  - Inconsistent data types
  - Large datasets requiring efficient I/O
- **Validation**:
  - Use validation schemas
  - Run sample tests to confirm parsing correctness.

---

## Final notes:
- The class should be designed for extensibility, allowing easy addition of new datasets or formats.
- Modular functions for dataset parsing, validation, and saving enable maintainability.
- Refer to the shared data schemas in the code structure and configuration files to ensure compliance.

This thorough logic analysis provides a comprehensive blueprint for implementing `dataset_loader.py` aligned with the paper’s methodology and experimental design.

## graph_perturber.py

{
  "Logic Analysis": "
The core purpose of `graph_perturber.py` is to implement functions that modify existing reasoning graphs to alter their complexity along defined dimensions—namely, graph depth, width, and numerical (node value) complexity—while ensuring the resulting graphs remain logically consistent. This module directly supports the process described in Section 2.3 of the paper, allowing fine-grained control to generate datasets with varying levels of difficulty for debugging, robustness testing, and bias analysis.

**Key Inputs and Outputs:**
- Input: A ReasoningGraph object, along with parameters indicating how much to perturb each complexity dimension.
- Output: A new ReasoningGraph object, structurally and numerically perturbed per the specified deltas, maintaining internal consistency.

**Primary Functions and Tasks:**

1. **Perturbing Graph Depth (`perturb_depth() / mutate_depth()`):**
   - *Goal:* Increase or decrease the maximum reasoning steps (longest path) in the graph.
   - *Implementation:*
     - Identify the longest path in the graph.
     - To increase depth:
       - Insert new nodes into this path.
       - Generate intermediate reasoning steps, possibly by interpolation or replication, to extend the chain.
       - For each inserted node, assign plausible content (e.g., intermediate calculation, reasoning step).
     - To decrease depth:
       - Remove or condense intermediate nodes in the longest chain, merging steps where possible, or removing redundant nodes.
   - *Constraints:* Ensure the start and end nodes still match logical reasoning sequences and the overall correctness can be verified later.

2. **Perturbing Graph Width (`perturb_width() / mutate_width()`):**
   - *Goal:* Increase or decrease the number of nodes or attribute pairs at a given reasoning step.
   - *Implementation:*
     - To increase width:
       - For specific nodes, add attribute nodes or related entities connected via `to_attribute` edges.
       - For reasoning steps that involve multiple entities, add additional attribute nodes.
       - For bias attributes (social reasoning), selectively attach attributes with unbiasing or biasing features.
     - To decrease width:
       - Remove attribute nodes or relation branches, focusing on core reasoning paths.
     - *Note:* When increasing width, generate attributes or entities that are contextually plausible and semantically related.

3. **Perturbing Numerical Value Complexity (`perturb_numerical()`):**
   - *Goal:* Change the numerical values of nodes to increase calculation difficulty or scale (e.g., larger numbers).
   - *Implementation:*
     - For nodes representing numbers, adjust their 'value' attribute based on the complexity level.
     - For increases:
       - Scale up the numeric values by a factor that maintains semantic plausibility.
       - For math tasks, choose values that require more complex operations or larger calculations.
     - *Constraints:*
       - Maintain logical consistency: e.g., increasing operands should still lead to valid calculations.
       - When increasing numerical complexity, e.g., from 10 to 1000, update downstream calculations as necessary to reflect changes.
      
4. **Coordinated Perturbation:**
   - When perturbing multiple dimensions, orchestrate the modifications in a way that preserves logical order:
     - E.g., increasing depth (more reasoning steps) with larger numerical values in nodes to escalate difficulty.
   - Keep track of the original graph to allow relative modifications (from the baseline).

5. **Implementation Details & Constraints:**
   - Use `networkx.DiGraph` within ReasoningGraph:
     - Add or remove nodes based on delta values.
     - Insert nodes along existing paths for depth increases.
     - Add attribute nodes connected via `to_attribute` for width increases.
   - Maintain node IDs and ensure no duplication or dangling references.
   - For each perturbation, generate explainable reasoning steps in content that match the original dataset style.
   - After modifications, optionally, call verification modules (by external scripts) to ensure the resulting graph and text are valid, but this step is primarily outside this module.
- **Error Handling and Edge Cases:**
  - When decreasing depth or width, if the graph is already minimal, skip or return unaltered graph.
  - For numerical perturbations, ensure values stay within plausible bounds (e.g., non-negative, reasonable size).
  - When inserting nodes, generate content consistent with the task type (math, social, symbolic, spatial).

**Summary of Usage:**
- The main function will be similar to:
  ```
  def perturb_graph(original_graph: ReasoningGraph, depth_delta: int, width_delta: int, numerical_scale: float) -> ReasoningGraph:
      new_graph = copy.deepcopy(original_graph)
      new_graph = perturb_depth(new_graph, depth_delta)
      new_graph = perturb_width(new_graph, width_delta)
      new_graph = perturb_numerical(new_graph, numerical_scale)
      return new_graph
  ```
- These functions will modify the graph while preserving as much reasoning coherence as possible, aligned with the complexities defined in the paper's Section 2.3.

**Final Notes:**
- Modularity: Design each perturbation as an independent function for clarity and flexibility.
- Reproducibility: Set random seeds where randomness occurs, so modifications are deterministic across runs if needed.
- Testing: Include example cases with known graphs and validate their size, depth, and node/attribute properties after perturbations.

This detailed logic analysis ensures that the perturbation functions will be consistent with the paper's methodology, supporting controlled experiments on reasoning complexity, bias, and robustness."


## label_verifier.py

{
  "description": "The 'label_verifier.py' module implements the 'LabelVerifier' class, which verifies the correctness of a regenerated answer and reasoning for a data point using a structured prompt to an LLM with code execution capabilities (e.g., GPT-4 with code interpreter). The core function is 'verify_label(text: str) -> bool', which takes the generated text (containing reasoning, answer, and possibly intermediate steps) and determines whether the model's answer matches the logical and computational expectations derived from the reasoning.\n\nKey elements and logic:\n\n1. Input structure:\n   - The input to 'verify_label' is a text string generated by the data regeneration step, which should include the reasoning, the final answer, and potentially intermediate calculations.\n   - The generated text is expected to follow a structured and prompt-engineered format, as specified in the 'verification_prompt.txt' template.\n\n2. Prompting strategy:\n   - The class uses a prompt template that inserts the generated reasoning text into a structured prompt to elicit a verification response from the LLM.\n   - This template should instruct the LLM explicitly to analyze the reasoning, check the correctness of answer derivations, and determine consistency.\n   - The prompt will include instructions to parse the reasoning, identify calculations, and cross-verify the final answer with the reasoning steps.\n\n3. API call:\n   - The verification process makes a call to the OpenAI API (e.g., GPT-4) with the structured prompt.\n   - The model responds with a verification verdict—e.g., 'Correct' or 'Incorrect', or with a detailed explanation from which correctness can be inferred.\n   - Usually, the response is a short verification label ('Correct'/'Incorrect') for automation; optionally, detailed logs can be stored.\n\n4. Parsing and decision:\n   - The function processes the model's response:\n       - If the response explicitly states correctness (e.g., contains 'Correct'), return True.\n       - Otherwise, if it indicates incorrectness or ambiguity, return False.\n   - Optionally, implement heuristic rules to parse the response text, looking for keywords.\n\n5. Multiple retries & robustness:\n   - Implement logic to retry verification if the response is ambiguous or missing key information.\n   - Limit retries to avoid infinite loops.\n\n6. Integration with the main process:\n   - 'verify_label' is invoked immediately after data generation to filter out invalid samples.\n   - Only samples verified as correct are accepted into the augmented dataset.\n\n7. External dependencies:\n   - Relies on 'openai' SDK, with configured API key set in the class.\n   - Uses prompt template loaded from 'verification_prompt.txt'.\n\n8. Additional considerations:\n   - Ensure compliance with API limits (rate limiting, timeout settings).\n   - Handle possible API errors/exceptions gracefully (retry, fallback).  \n   - Maintain logs for debugging and auditing verification outcomes.\n\nIn summary, 'LabelVerifier' automates correctness checking by leveraging LLMs' reasoning and code execution capabilities via structured prompts. It feeds the generated answer and reasoning into the model, parses the verification response, and outputs a boolean indicating label correctness, which then influences whether a sample is retained or discarded for further training or evaluation.",
  "considerations": [
    "Ensure the prompt template is well-defined and matches the expected input/output format, as per 'verification_prompt.txt'.",
    "The 'verify_label' function must incorporate robust parsing to interpret the LLM's response reliably.",
    "Handle API errors, rate limits, and retries properly to ensure consistency in verification.",
    "Configure the API call parameters (temperature, max_tokens) in accordance with experiment settings, e.g., low temperature for deterministic responses.",
    "Optionally, include a short verification explanation in logs for debugging."
  ],
  "reference": "The process is based on the descriptions in the paper's Section 2.4 and Appendix F, and aligns with the overall experimental framework described."
}

## main.py

{
  "main.py": {
    "Purpose": "Coordinate the entire experimentation pipeline from dataset loading, reasoning graph extraction, systematic perturbation, data regeneration, label verification, and evaluation across different complexity levels and datasets. Ensure modularity, configurability, and reproducibility.",
    "Overall Flow": [
      "Load configuration parameters from 'config.yaml' and possibly prompts from external files.",
      "Initialize modules: DatasetLoader, ReasoningGraphHandler, DataGenerator, LabelVerifier, ModelEvaluator.",
      "For each dataset specified in configuration:",
      "    Load the dataset using DatasetLoader.",
      "    For each data point in the dataset:",
      "        Extract the reasoning graph via ReasoningGraph (using heuristics or rules).",
      "        For each complexity level in 'complexity_levels' (from 'config.yaml'):",
      "            Perturb the reasoning graph via GraphPerturber to achieve desired complexity profile.",
      "            Use DataGenerator to regenerate natural language data from the perturbed graph.",
      "            Verify the generated data's correctness with LabelVerifier; retry if verification fails (with possible multiple attempts).",
      "    Store all (original and perturbed) data with metadata (including complexity info).",
      "Evaluate all generated datasets with ModelEvaluator:",
      "    For each model specified:",
      "        Run inference with appropriate prompts (CoT, LtM, etc.).",
      "        Compute accuracy, bias scores, CIARRs (or other metrics).",
      "        Record performance metrics per complexity level and task.",
      "Perform analysis comparing model robustness across complexity levels, plot figures (performance decline, bias increase), and cross-validate robustness metrics.",
      "Facilitate output: Save processed datasets, evaluation metrics, and logs for reproducibility and detailed analysis."
    ],
    "Details and Considerations": [
      "Loading configuration: Use 'pyyaml' to parse 'config.yaml', store parameters in a Config object or dict for accessible parameters throughout.",
      "Prompt Templates: Load from external files, pass to respective modules (DataGenerator, LabelVerifier).",
      "Dataset Loading: Handle datasets in JSON or CSV format; ensure schema matches expectations (question, answer, reasoning, etc.); support dataset splits if needed.",
      "Graph Extraction: May involve regex parsing or NLP heuristics; ensure reproducibility by fixing random seeds where randomness occurs in graph perturbation.",
      "Perturbation Process: Adjust graph attributes to increase depth, width, and numerical complexity as per specified deltas; maintain logical consistency.",
      "Data Regeneration: Call openai API with prompts/completions structured with templates; set temperature ~0 for deterministic output; implement retries with backoff.",
      "Verification: Send generated explanations/answers to LabelVerifier, which uses structured prompt + code execution. Retry failed verifications up to predefined attempts.",
      "Data Storage: Save generated data points with associated graph and complexity metadata in structured directories or databases.",
      "Model Evaluation: Use API keys and model names specified, handle both API calls (GPT models) and local models if necessary. Apply prompt engineering for CoT, LtM methods.",
      "Metrics Calculation: Compute accuracy per complexity level; calculate CIARRs across increasing complexity levels to measure robustness; gather bias scores where applicable.",
      "Results Visualization: Generate plots similar to Figures 2, 3, 4, etc., to analyze performance trends.",
      "Logging & Reproducibility: Record all parameters, prompts, random seeds, API call details, API rate limits, and hardware info. Save logs."
    ],
    "Error Handling & Validation": [
      "Incorporate manual validation checkpoints, e.g., sample verification of extracted graphs, regenerated data, and model outputs.",
      "Implement retries for API calls and verification failures.",
      "Log failed cases separately for manual inspection and potential adjustment of rules/prompts.",
      "Validate the logical coherence of perturbed graphs before regeneration to avoid nonsensical data."
    ],
    "Resource Management": [
      "Batch processing for API calls to avoid rate limits.",
      "Use caching of API responses or intermediate data to prevent redundant requests.",
      "Support multiple models and hardware configurations by modularly encapsulating API vs. local inference."
    ],
    "Reproducibility": [
      "Set random seeds for graph perturbations and sampling.",
      "Document hyperparameters, prompt versions, dataset paths, API configurations.",
      "Save all intermediate and final datasets with clear schemas.",
      "Ensure scripts can be run sequentially with minimal external intervention."
    ],
    "Final Notes": [
      "Design the script to be configurable via command-line arguments or config files.",
      "Enable parallelization for large datasets where feasible.",
      "Include verbose logging and progress checkpoints for debugging and audit.",
      "Facilitate easy extension to additional datasets, models, or complexity dimensions."
    ]
  }
}

## model_evaluator.py

{
  "Implementation approach": "Design the ModelEvaluator class to facilitate flexible evaluation of multiple LLMs across generated datasets. It should handle both API-based models (e.g., GPT via openai SDK) and local models (if applicable). The class must load datasets, execute inference with appropriate prompting, parse model outputs to obtain answers, compare with ground truth labels, and compute metrics such as accuracy, bias scores, and CIARRs. Incorporate support for batch evaluation and handle different prompt templates for each model type and task. Include methods to summarize and store evaluation results for analysis. The system must be robust to different model input/output formats and configurable thresholds.",
  "Data structures and interfaces": "Class DataPoint: with properties for question, reasoning, answer, reasoning_graph. Class EvaluationMetrics: to hold accuracy, bias_score, ciarr values. Class ModelEvaluator: with methods load_dataset(), evaluate(), parse_output(), compute_metrics(), and save_results(). Internal data structures include lists of DataPoints, dictionaries for metrics, and structured storage for model responses and computed metrics. The evaluate() method iterates over dataset points, invokes model inference, parses responses, and updates metrics. Use standard JSON/dictionary formats for data interchange, following the schemas provided in the API spec.",
  "Logic analysis": [
    "Initialization: The ModelEvaluator constructor should accept a list of models (with their identifiers and API keys or local model info), relevant prompt templates, and evaluation hyperparameters. Set up API clients or local inference wrappers accordingly.",
    "Dataset loading: Implement load_dataset() to read dataset files (JSON, CSV) containing DataPoint objects, prepare data for batching if applicable.",
    "Evaluation flow: For each DataPoint in the dataset, generate model input prompts based on task-specific template (e.g., encouraging chain-of-thought reasoning). Handle model-specific prompt formatting and parameters (temperature, max_tokens).",
    "Model inference: For API models, implement API calls with appropriate parameters (temperature, max_tokens). For local models, execute inference routines, possibly with batching. Enforce timeout policies and manage retries if needed.",
    "Output parsing: Define parse_output() to extract the final answer from the model's text. This involves regex or NLP parsing to identify answer spans, especially in reasoning tasks with structured outputs. For chain-of-thought, locate the final answer within reasoning steps. For multiple-choice tasks, recognize options and select the best fit.",
    "Answer comparison: Compare parsed model answer with ground truth label. For GSM8K, likely exact numeric match or tolerance-based match. For multiple-choice data, direct string or label matching. Record correctness (binary accuracy).",
    "Bias and score calculations: For bias scores, implement additional parsing to identify bias indicators or attribute mentions. For CIARR, maintain a record of accuracy at each complexity level, then compute percentage retention as per formula: \(\mathrm{CIARR}_D = \frac{1}{n-1} \sum_{i=1}^{n-1} (A_{i+1}/A_i) \times 100\%\).",
    "Metrics storage: After evaluation, generate a summary dictionary/object containing overall accuracy, bias scores, and CIARR per model, per complexity level. Store per-point data for failure analysis if needed.",
    "Results reporting: Implement save_results() to output metrics and detailed logs to JSON or CSV files for further analysis. Possibly visualize metric trends across complexity levels.",
    "Error handling: Ensure the code gracefully handles failed API calls, parsing errors, and inconsistent outputs. Log such incidents for debugging.",
    "Configurable parameters: Use parameters from the configuration file for API keys, prompt templates, thresholds, and evaluation settings, enabling reproducible experiments.",
    "Modularity: Organize parsing, inference, and metrics computation into separate helper functions/methods for clarity, testing, and extensibility."
  ],
  "Anything UNCLEAR": "Clarify how the answers are expected to be outputted by models—are answers always in a specific format (e.g., answer: XX, or just the final number)? Confirm the parsing logic for different tasks and prompt styles. Clarify if bias scoring involves specific pre-defined metrics or requires custom analysis. Provide sample model outputs for different tasks to refine parsing heuristics. Confirm whether to perform multiple inference retries for unstable models or responses, and if so, the maximum retry count. Ensure understanding of the exact metric definitions and calculation procedures for bias scores and CIARR, including any normalization or weights. Clarify expected output formats and thresholds for correctness in different datasets (e.g., tolerance for numerical answers in GSM8K vs. exact string match for multiple-choice questions)."
}

## reasoning_graph.py

# Reasoning Analysis for reasoning_graph.py

### 1. Purpose and Responsibilities
- Implement the **ReasoningGraph** class that:
  - Extracts reasoning graphs from raw textual reasoning chains in data points.
  - Represents the graphs with nodes and edges, adhering to the data structure interfaces.
  - Supports manipulation (perturbation) of the graph structure to control complexity along dimensions like depth and width.
  - Supports serialization/deserialization to/from JSON compatible formats.
  
---

### 2. Core Components and Data Structures
- **Nodes**:
  - Should be instances of **GraphNode** with properties:
    - **id**: unique integer identifier.
    - **type**: string indicating category (`"initial"`, `"intermediate"`, `"final"` or dataset-specific types such as `"person"`, `"attribute"`, `"math_step"`).
    - **content**: descriptive text.
- **Edges**:
  - Instance of **GraphEdge** with:
    - **source_id**: source node ID.
    - **target_id**: target node ID.
    - **relation_type**: string description (e.g., `"leads_to"`, `"computes"`, `"has_attribute"`).
  
- **Internal Representation**:
  - Use `networkx.DiGraph` as the underlying structure:
    - Nodes: store **GraphNode** objects or just their properties.
    - Edges: store **GraphEdge** properties.
  - Provide mapping from node IDs to node objects for easy processing.
  
---

### 3. Extraction of Reasoning Graphs from Text
- **Input**:
  - Raw textual reasoning chain (from dataset).
- **Output**:
  - Constructed `ReasoningGraph` object.
- **Parsing Strategy**:
  - Use regex or NLP heuristics to identify key components:
    - **Nodes**:
      - Extract statements, equations, entities, or reasoning steps.
      - Detect initial facts, intermediate steps, and final answer steps.
    - **Edges**:
      - Infer relationships based on syntactic cues:
        - Sequential steps (e.g., "then", "next", "after").
        - Causal or logical connectives ("because", "since", "therefore").
        - Equation structure (e.g., "A = B + C") suggests edges from B, C to A.
  - **Dataset-specific heuristics**:
    - For math, parse equations to identify operands and operators.
    - For social/symbolic tasks, parse explanations for relation statements.
  - **Iterative validation**:
    - After parsing, check if the reasoning graph can produce the correct answer via forward evaluation or rule-based label computation.
- **Validation**:
  - Use a rule-based function (such as `f_l`) to compute the label from the graph.
  - Compare with the original label to validate correctness.
  
---

### 4. Building the Graph Structure
- **Method(s)**:
  - `__init__()`: initialize empty networkx.DiGraph
  - `add_node()`:
    - Assign unique ID.
    - Store type and content.
  - `add_edge()`:
    - Connect source and target node IDs.
    - Store relation_type.
- **Construction from parsed data**:
  - Create nodes with IDs, types, texts.
  - Add edges based on inferred relationships.
  
---

### 5. Perturbation and Complexity Control
- **Perturbation parameters**:
  - **Depth**:
    - Increase or decrease by inserting or removing intermediate nodes.
    - To increase depth:
      - Find longest path.
      - Insert nodes or split nodes along the path.
  - **Width**:
    - Increase or decrease the number of sibling nodes.
    - For increasing width:
      - Add sibling nodes at a certain level.
      - Connect new nodes via edges maintaining logical coherence.
  - **Numerical/Numeric Complexity**:
    - For math:
      - Modify node contents (numbers) to larger/smaller values.
      - For example, increase node values by a factor.
- **Implementation**:
  - `perturb()` method:
    - Accepts parameters like `depth_delta`, `width_delta`, `numerical_scale`.
    - Modifies the internal networkx graph accordingly.
    - Ensures logical consistency after perturbations.
- **Preserving semantics**:
  - When increasing complexity, make sure the reasoning chain remains valid.
  - When decreasing, prune or simplify while maintaining the original answer.
  
---

### 6. Serialization / Deserialization
- **to_json()**:
  - Convert internal networkx graph to JSON/dict format:
    - Nodes: list of dicts with id, type, content.
    - Edges: list of dicts with source, target, relation_type.
- **from_json()**:
  - Construct graph from JSON/dict for compatibility with other modules.
  - Rebuild networkx.DiGraph based on stored nodes and edges.
  
---

### 7. Implementation Details & Best Practices
- **ID management**:
  - Assign consecutive IDs to nodes.
  - When inserting nodes, avoid duplicate IDs.
- **Edge consistency**:
  - Enforce DAG structure.
  - Prevent cycles when perturbing.
- **Validation**:
  - After perturbation, optionally evaluate if the graph still yields the same label (via rule-based `f_l`).
- **Extensibility**:
  - Design `perturb()` function to support multiple types of perturbations:
    - Structural (depth, width).
    - Content (numbers, attributes).
    
### 8. Error Handling
- Ensure methods handle invalid inputs gracefully:
  - Non-existent node IDs.
  - Invalid relation types.
  - Broken graph coherence after perturbation.
- Raise informative exceptions or warnings.

---

### 9. Summary
- The `ReasoningGraph` class acts as:
  - The parser: converting textual explanations into formal graph representations.
  - The manipulator: adjusting complexity along defined axes.
  - The serializer: exporting/importing graph structures.
- Core functions (`extract_from_text()`, `perturb()`, `to_json()`, `from_json()`) should be well-encapsulated.
- Use the `networkx` library for graph operations, ensuring DAG constraints and consistency.
- Use heuristic parsing and rule-based validation, acknowledging some dataset-specific tailoring might be necessary.

---

This reasoning analysis provides a blueprint for implementing a robust, flexible, and dataset-agnostic `ReasoningGraph` class that adheres to the paper's methodology and interfaces, supporting both extraction and perturbation for dynamic reasoning dataset generation.

## utils.py

{
  "utils.py": "The utils.py module serves as a support library containing auxiliary functions essential for orchestrating the DARG framework's operations. It provides utilities such as configuration file handling, prompt template management, JSON serialization/deserialization, graphical debugging, and initialization helpers. Key functions and classes should include:\n\n1. **Configuration Handling:**\n   - `load_config(config_path: str) -> dict`\n     - Reads the provided YAML configuration file (`config.yaml`).\n     - Parses and loads hyperparameters, prompt templates, dataset paths, API keys, and experiment parameters into Python dictionaries.\n     - Ensures consistent access to configuration parameters across modules.\n\n2. **Prompt Management:**\n   - `get_prompt(template_name: str) -> str`\n     - Retrieves a specific prompt template by name (e.g., reasoning graph generation, data regeneration, label verification, evaluation).\n     - Prompts are stored as external text files (`prompts/*.txt`) for maintainability and easy updates.\n   - `load_prompt_template(file_path: str) -> str`\n     - Reads prompt template files into string variables.\n   - Optionally, functions to format prompts with parameters (e.g., fill in question text, reasoning graph) for structured API calls.\n\n3. **JSON Serialization Helpers:**\n   - `save_to_json(data: Any, file_path: str) -> None`\n     - Serializes Python objects (e.g., DataPoint, ReasoningGraph) into JSON format and saves to disk.\n   - `load_from_json(file_path: str) -> Any`\n     - Loads JSON data from disk and deserializes into Python objects.\n   - `serialize_datapoint(datapoint: DataPoint) -> dict`\n     - Converts a DataPoint object into a JSON-serializable dictionary.\n   - `deserialize_datapoint(data: dict) -> DataPoint`\n     - Converts dictionary data into a DataPoint object.\n\n4. **Graph Visualization & Debugging:**\n   - `plot_graph(graph: ReasoningGraph) -> None`\n     - Uses networkx/matplotlib to visualize the reasoning graph for debugging or analysis.\n     - Supports annotations such as node labels, edge labels, and complexity indicators.\n   - `print_graph_details(graph: ReasoningGraph) -> None`\n     - Outputs internal graph structure details for verification.\n\n5. **String and Text Processing Helpers:**\n   - `extract_reasoning_steps(text: str) -> List[str]`\n     - Parses reasoning explanation texts into step-by-step components.\n   - `match_answer_in_text(text: str, answer: str) -> bool`\n     - Checks if the generated answer matches the expected label.\n\n6. **API Utilities:**\n   - `initialize_openai_api(api_key: str) -> None`\n     - Sets API key and configures openai SDK parameters.\n   - `call_openai_api(prompt: str, model: str, temperature: float, max_tokens: int) -> str`\n     - Handles API calls, retries, and error handling.\n\n7. **Miscellaneous Helpers:**\n   - Functions for random seed control, e.g., `set_seed(seed: int)`\n   - Directory creation and path validation helpers.\n\n**Implementation notes:**\n- Ensure modular design: each function should perform a single, well-defined task.\n- Use type hints for clarity.\n- Handle exceptions gracefully and log errors for troubleshooting.\n- Integrate with the main pipeline to facilitate seamless orchestration.\n- Follow the code conventions in the project to facilitate consistency and readability.\n\nThis comprehensive utility suite will underpin core pipeline components such as dataset loading, prompt management, graph handling, and evaluation, enabling reproducibility and maintainability."}

