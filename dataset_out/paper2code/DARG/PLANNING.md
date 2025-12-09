# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a detailed, step-by-step plan that distills the methodology, experimental design, and key implementation considerations from the paper. This roadmap will guide subsequent code development and ensure fidelity to the original approach.

---

# Reproduction Plan for DARG Framework

## 1. Overview & Objectives
- **Goal:** Implement the DARG framework to dynamically generate, perturb, and evaluate reasoning datasets across multiple reasoning tasks and models.
- **Core idea:** Extract reasoning graphs from existing datasets, perturb these graphs along controllable complexity dimensions, regenerate data, and evaluate LLMs’ robustness and bias.
- **Key components:**
  - Reasoning graph extraction
  - Graph perturbation (complexity control)
  - Data regeneration/back-transformation
  - Label verification with code-augmented LLM
  - Evaluation under varying complexities

---

## 2. Data & Benchmark Selection
- **Datasets used in the paper:**
  - GSM8K (math reasoning)
  - BBQ (social reasoning)
  - BBH Dyck Language (symbolic reasoning)
- **Data needs:**
  - Original datasets in their text format
  - Access to dataset annotations and reasoning structures, or methods to extract reasoning graphs
- **Action:** For each dataset:
  - Acquire the static dataset with questions, options, and labels.
  - Obtain or implement reasoning graph extraction methods (see Section 2.1).
  - Ensure the dataset splits for training, testing, and perturbation.

---

## 3. Reasoning Graph Extraction (Section 2.1)
- **Purpose:** Represent each data point (question + answer) as a reasoning graph.
- **Approach:**
  - Parse existing reasoning chains (from explanations, if available) into graph structures:
    - Nodes: reasoning steps, facts, or entities
    - Edges: logical, temporal, or causal relations
  - Use rule-based heuristics or NLP processing:
    - For math, parse step-by-step solutions into sequences and convert into graph (e.g., nodes= calculation steps, edges= order)
    - For social or symbolic tasks, manually define rules or use heuristics to identify elements and relations.
    - For symbolic reasoning (Dyck language), construct tree-like graphs from nested brackets.
- **Implementation tips:**
  - Regular expressions or syntactic parsers for structured explanations.
  - Store graphs in JSON format compatible with the paper’s described graph schema.
- **Goal:** For each dataset, generate a graph representation encoded as JSON/dictionary.

---

## 4. Graph Perturbation / Complexity Control (Section 2.2 & 2.3)
- **Define complexity dimensions:**
  - Numerical complexity (number size, calculation depth)
  - Graph depth (longest chain of reasoning steps)
  - Graph width (number of entities or relations at a step)
- **Perturbation strategies:**
  - Numerical complexity:
    - Increase/decrease number sizes (e.g., multiply/divide operands)
    - Adjust calculation difficulty (e.g., nested operations)
  - Graph depth:
    - Inserting or removing reasoning steps
    - Extending or compressing chains (e.g., interpolating additional steps)
  - Graph width:
    - Adding/removing nodes or attribute pairs (e.g., bias attributes)
    - Increasing the number of branches or disjunctive options
- **Implementation:**
  - Develop rule-based graph manipulation functions:
    - For each dimension, specify how to alter nodes/edges while maintaining syntactic coherence.
    - Use random sampling within defined rules (e.g., for attribute addition/removal).
  - Ensure that the perturbation process respects constraints (e.g., logical consistency, linguistic diversity).

---

## 5. Data Regeneration / Back-Transformation (Section 2.4)
- **Goal:** Convert perturbed graphs back into natural language data points.
- **Method:**
  - Use structured prompt-based generation with a large language model (e.g., GPT-4 Turbo) trained or instructed to:
    - Map the graph structure to textual explanation or question formats.
    - Incorporate new attribute-value pairs, reasoning steps, and contextual info.
  - Design prompt templates (see Appendix F):
    - Provide the graph structure as input
    - Request generation of a clear, human-readable reasoning chain, question, and answer
- **Implementation:**
  - Automate prompt construction from the graph JSON.
  - Use OpenAI API (or similar) with specified temperature settings (close to zero for deterministic output).
  - Parse generated text (question, reasoning, answer) and verify consistency.
  
---

## 6. Label Verification via Code-augmented LLM (Section 2.4 & Appendix F)
- **Purpose:** Confirm correctness of generated labels.
- **Process:**
  - Prompt the LLM (e.g., GPT-4 Turbo with code interpreter or self-verification prompt) with:
    - The generated reasoning steps and outputs.
    - The expected answer and label.
  - Ask the LLM to verify:
    - The logic and calculation correctness.
    - The consistency of the answer with the reasoning graph.
    - To produce a token/label indicating "correct" or "incorrect."
- **Outcome:**
  - Accept data point if verified as correct.
  - Discard or regenerate if verification fails.
- **Details:**
  - Use structured code-based verification prompts as in the paper ("Code output" sections in Appendix F).
  - Implement automatic retries for failed verifications.

---

## 7. Dataset Construction & Multiple Complexity Levels
- **For each dataset:**
  - Create datasets at multiple levels:
    - Original (unperturbed)
    - +1, +2, +4, +8 complexity (per dimension)
  - Include multiple perturbation iterations/combinations:
    - For depth, width, and numerical perturbations
  - Store the perturbed data in a structured format (JSON or CSV) with:
    - Text: question, reasoning, options, answer
    - Graph: original and perturbed version
    - Complexity attributes
    - Verification status
    
- **Note:** To compare, also generate control datasets with no perturbation but consistent text.

---

## 8. Model Evaluation (Section 3)
- **Models:**
  - Use the same models as in the paper:
    - GPT-4 Turbo, GPT-3.5 Turbo, Llama models, Mixtral, WizardLM, DeepSEA, Gemini, etc.
  - Obtain access/authentication to APIs or local deployment.
- **Prompting strategies:**
  - Chain-of-Thought (CoT): Use prompts from Appendix F.
  - LtM (least-to-most) and other prompting: Design templates similar to the paper.
- **Evaluation metrics:**
  - Success rate (accuracy) per complexity level.
  - CIARR (Complexity-Induced Accuracy Reduction Rate):
    - Compute by comparing accuracy progression as complexity increases.
- **Implementation:**
  - Automate experiment runs with multiple models and dataset variants.
  - Record answers, compare against reference labels, and compute accuracy.
  - Collect error cases for qualitative analysis.

---

## 9. Bias & Bias Bias Evaluation (Sections on social reasoning)
- **Specific tasks:**
  - BBQ and its modifications—add attribute pairs, bias attributes.
- **Method:**
  - Follow prompts from Appendix E (and Figures 4, 6, 12, 13):
    - Generate data with attribute modifications.
    - Evaluate bias scores (e.g., bias scores, avoidance rates).
- **Analysis:**
  - Track bias score changes with attribute complexity.
  - Aim to replicate Figures 4 and 6.

---

## 10. Implementation Environment & Hyperparameters
- **OpenAI API:**
  - Use temperature near 0 for determinism; e.g., 0.0–0.1.
  - Set max tokens (e.g., 512–1024) for response length.
- **Local inference:**
  - Use GPT-3.5 or GPT-4 via API.
  - Use open-source models (Llama, Mixtral, WizardLM) with appropriate frameworks (HuggingFace, etc.).
  - Fine-tuning hyperparameters:
    - Learning rate, batch size, epochs as per models’ documentation.
    - For fine-tuning: follow paper’s hyperparameters (if specified), or adopt standard best practices.
- **Prompt design:**
  - Templates should follow the paper’s style.
  - Temperatures should be low.
  - Use few-shot examples as provided in Appendix F.
- **Evaluation:**
  - Automate accuracy calculations.
  - For CIARR, implement difference metrics as in the paper.

---

## 11. Summary & Additional Considerations
- **Key challenges/unknowns to resolve:**
  - Exact graph extraction heuristics for each dataset.
  - Fine-tuning or instruction prompts for reasoning graph to text generation.
  - Ensuring consistency and correctness with a code-augmented verifier.
  - Handling failure cases and consistency checks.
- **Iterative validation:**
  - Verify correctness of generated data points manually for a subset.
  - Adjust perturbation rules and prompts as needed.
- **Scalability:**
  - Use batching and parallel API calls.
  - Maintain version control for all datasets, prompts, and code.

---

# Final note:
This roadmap ensures the reproduction process is:
- **Systematic:** From data acquisition, graph extraction, perturbation, regeneration, to evaluation.
- **Flexible:** Modular components can be refined or replaced depending on dataset characteristics.
- **Detailed:** Critical for faithful replication and subsequent analysis.

---

Would you like me to prepare specific prompt templates, data schemas, or detailed pseudocode next?

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Design a modular system with clear separation of core steps: dataset processing, reasoning graph extraction, graph perturbation, data regeneration via open AI API, label verification, and evaluation. Use existing NLP libraries (e.g., spaCy, regex) for graph extraction, networkx for graph manipulation, and OpenAI's Python SDK for model interactions. Structure code in classes for each module to enable easy maintenance and extensibility. Include configuration files for hyperparameters and prompts. Implement a main pipeline in `main.py` to orchestrate data loading, graph extraction, perturbation, regeneration, verification, and evaluation, facilitating reproducibility of the paper's experiments.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "reasoning_graph.py",
        "graph_perturber.py",
        "data_generator.py",
        "label_verifier.py",
        "model_evaluator.py",
        "utils.py",
        "config.yaml",
        "prompts/regen_prompt.txt",
        "prompts/verification_prompt.txt"
    ],
    "Data structures and interfaces": "
classDiagram
    class MainPipeline {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(dataset_path: str)
        +load_data() -> List[DataPoint]
        +save_dataset(output_path: str) -> None
    }
    class DataPoint {
        -question_text: str
        -answer: str
        -reasoning_text: str
        -reasoning_graph: ReasoningGraph
        +to_json() -> dict
        +from_json(data: dict) -> DataPoint
    }
    class ReasoningGraph {
        -nodes: List[GraphNode]
        -edges: List[GraphEdge]
        +add_node(node: GraphNode) -> None
        +add_edge(edge: GraphEdge) -> None
        +perturb(depth_delta: int, width_delta: int, numerical_scale: float) -> ReasoningGraph
        +to_json() -> dict
        +from_json(data: dict) -> ReasoningGraph
    }
    class GraphNode {
        +id: int
        +type: str
        +content: str
    }
    class GraphEdge {
        +source_id: int
        +target_id: int
        +relation_type: str
    }
    class DataGenerator {
        +__init__(prompt_template: str, openai_api_key: str)
        +generate_text(graph: ReasoningGraph) -> str
    }
    class LabelVerifier {
        +__init__(prompt_template: str, openai_api_key: str)
        +verify_label(text: str) -> bool
    }
    class ModelEvaluator {
        +__init__(models: List[str], prompts: dict)
        +evaluate(dataset: List[DataPoint]) -> dict
    }
    class Config {
        +load(config_path: str)
        +save(config_path: str)
        +hyperparameters: dict
        +prompt_templates: dict
    }
    MainPipeline --> DatasetLoader
    MainPipeline --> ReasoningGraph
    MainPipeline --> DataGenerator
    MainPipeline --> LabelVerifier
    MainPipeline --> ModelEvaluator
    ReasoningGraph --> GraphNode
    ReasoningGraph --> GraphEdge
    DataGenerator --> openai.api
    LabelVerifier --> openai.api
    DatasetLoader --> DataPoint
    DatasetLoader --> DataPoint
    ModelEvaluator --> DataPoint
    ModelEvaluator --> DataPoint
    ModelEvaluator --> DataPoint
    ModelEvaluator --> DataPoint
    ModelEvaluator <-- dict
    MainPipeline --> Config
    Config --* hyperparameters: dict
    Config --* prompt_templates: dict
    ",
    "Program call flow": "
sequenceDiagram
    participant MC as MainPipeline
    participant DL as DatasetLoader
    participant DG as DataGenerator
    participant VL as LabelVerifier
    participant ME as ModelEvaluator
    participant DC as ReasoningGraph
    participant C as Config
    MC->>DL: load_data()
    DL-->>MC: dataset
    MC->>DC: extract_graphs(dataset)
    DC-->>MC: graph_list
    loop for each graph in graph_list
        MC->>DC: perturb_graph(graph, depth_delta, width_delta, numerical_scale)
        DC-->>MC: perturbed_graph
        MC->>DG: generate_text(perturbed_graph)
        DG-->>MC: generated_text
        MC->>VL: verify_label(generated_text)
        alt verified
            MC->>DL: save_new_point(generated_text, graph metadata)
        else not verified
            MC->>DC: discard or re-perturb
        end
    end
    MC->>ME: evaluate(dataset)
    ME-->>MC: metrics
    ",
    "Anything UNCLEAR": "Clarify the reasoning graph extraction rules for each dataset, details about dataset formats, and any existing instructions/prompts used during regeneration or verification steps that could influence outcomes."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "networkx==2.6",
        "pyyaml==5.4.1",
        "matplotlib==3.4.3",
        "openai==0.27.0",
        "regex==2022.3.15",
        "scipy==1.7.0",
        "jsonschema==3.2.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Class DatasetLoader: responsible for loading static datasets (GSM8K, BBQ, BBH), parsing raw data into DataPoint objects, and saving generated data points after perturbation. Depends on data format specifications, existing dataset files, and schemas. Uses standard file I/O and supports JSON serialization/deserialization."
        ],
        [
            "reasoning_graph.py",
            "Class ReasoningGraph: responsible for extracting reasoning graphs from DataPoint's reasoning chains, represented as nodes and edges. Implements methods for parsing explanations (using regex or NLP heuristics), building graph structures. Also supports perturbation functions to modify depth, width, and complexity along specified dimensions, leveraging networkx for internal graph operations."
        ],
        [
            "graph_perturber.py",
            "Functions for applying perturbations to ReasoningGraph objects based on specified complexity parameters. Implements rules for increasing/decreasing depth, width, and numerical complexity while maintaining logical coherence. Depends on ReasoningGraph to transform graphs and produce new modified graphs."
        ],
        [
            "data_generator.py",
            "Class DataGenerator: takes a ReasoningGraph, prompts a language model (e.g., GPT via openai API) structured with templates to regenerate the natural language question, reasoning, and options. Ensures the output syntactically and semantically matches the perturbed graph. Uses prompt templates stored externally."
        ],
        [
            "label_verifier.py",
            "Class LabelVerifier: responsible for verifying correctness of regenerated explanations/answers using a code-augmented API prompt. Sends the generated reasoning and answer for validation, parsing model response to determine correctness. Relies on structured prompts and openai API."
        ],
        [
            "model_evaluator.py",
            "Class ModelEvaluator: runs inference on the generated datasets at various complexity levels. Measures model performance (accuracy, bias scores). Supports multiple models (via API or local models). Includes functions to parse model outputs and compute metrics including CIARRs."
        ],
        [
            "main.py",
            "Main orchestration script: loads datasets, extracts reasoning graphs, applies perturbations, regenerates data, verifies correctness, and evaluates models. Manages configuration, prompt templates, and experiment loop. Coordinates execution flow between modules."
        ],
        [
            "utils.py",
            "Supporting utilities: prompt management, JSON serialization helpers, graph plotting/debugging functions, configuration file handling."
        ],
        [
            "config.yaml",
            "Hyperparameters, prompt templates, dataset paths, model API keys, experiment settings, complexity level parameters."
        ]
    ],
    "Task list": [
        "utils.py",
        "dataset_loader.py",
        "reasoning_graph.py",
        "graph_perturber.py",
        "data_generator.py",
        "label_verifier.py",
        "model_evaluator.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 \n  components:\n    schemas:\n      DataPoint: \n        type: object\n        properties:\n          question: {type: string}\n          reasoning: {type: string}\n          answer: {type: string}\n          reasoning_graph: {type: object}\n        required: [question, reasoning, answer, reasoning_graph]\n      EvaluationMetrics:\n        type: object\n        properties:\n          accuracy: {type: number}\n          bias_score: {type: number}\n          ciarr: {type: number}\n  paths:\n    /load:\n      get:\n        summary: Load datasets\n        responses:\n          200:\n            description: Dataset loaded\n      post:\n        summary: Save perturbation results\n        requestBody:\n          description: Data points to save\n          required: true\n          content:\n            application/json:\n              schema:\n                type: array\n                items:\n                  $ref: '#/components/schemas/DataPoint'\n    /evaluate:\n      post:\n        summary: Run model evaluation\n        requestBody:\n          description: Dataset for evaluation\n          required: true\n          content:\n            application/json:\n              schema:\n                type: array\n                items:\n                  $ref: '#/components/schemas/DataPoint'\n        responses:\n          200:\n            description: Evaluation metrics\n            content:\n              application/json:\n                schema:\n                  $ref: '#/components/schemas/EvaluationMetrics'",
    "Shared Knowledge": "Shared hyperparameters, prompt templates, and data schemas are stored in config.yaml and external prompt text files. Utility functions support JSON I/O, graph visualization, and prompt management. Depend on openai SDK for API calls, networkx for graph manipulation.",
    "Anything UNCLEAR": "Exact heuristics for parsing reasoning chains into graphs per dataset require clarification. Also, details on the numeric perturbation rules (e.g., specific transformations) and the prompt templates used for regeneration need confirmation. Clarify hardware constraints for large dataset processing and API rate limits if scale is large."
}
[/CONTENT]

## 4. Configuration

```yaml
## config.yaml
datasets:
  GSM8K:
    path: "path/to/gsm8k_dataset.json"
  BBQ:
    path: "path/to/bbq_dataset.json"
  BBH_Dyck:
    path: "path/to/bbh_dyck_dataset.json"

model:
  name: "gpt-4-turbo"  # options: gpt-4-turbo, gpt-3.5-turbo, llama, etc.
  api_key: "YOUR_OPENAI_API_KEY"
  temperature: 0.0
  max_tokens: 1024

experiment:
  complex_dim:
    numerical_scale: 1.0      # base scale, perturbations create increase
    graph_depth_delta: 1      # incremental change in reasoning graph depth
    graph_width_delta: 1      # incremental change in width (nodes/attribute pairs)
  complexity_levels: [0, 1, 2, 4, 8]  # multiplicative levels of complexity

prompt_templates:
  reasoning_graph_generation: "place reasoning graph prompt template here"
  data_regeneration: "place regeneration prompt template here"
  label_verification: "place verification prompt template here"
  evaluation_prompt: "place evaluation prompt template here"

hyperparameters:
  # Use hyperparameters consistent with paper, if specified
  # The paper mainly mentions using near-zero temperature (0-0.1)
  # and standard prompt lengths. Additional hyperparameters can be added as needed.
  learning_rate: 2e-5  # default, not explicitly specified but recommended
  batch_size: 8  # default, adjustable based on hardware
  epochs: 10  # placeholder; not stated explicitly, set for training-related steps
  fine_tune_epochs: 3  # sample, if fine-tuning models
  inference_timeout: 300  # seconds, configurable per API rate limits
```

---

**Note:** Full configuration is available in `planning_config.yaml`
