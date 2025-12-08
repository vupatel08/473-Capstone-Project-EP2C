# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a detailed, step-by-step plan aligning with the methodology and experimental setup described in the paper, designed to guide subsequent implementation:

---

# Reproduction Plan for Autoformalizing Euclidean Geometry in LeanEuclid

## 1. Understanding the Goal
- **Main Objective:** Develop an automated pipeline that takes human-readable Euclidean geometry informal proofs and produces **formally verified Lean proofs**.
- **Key Components:** 
  - Dataset collection (problems & formal proofs).
  - Human-in-the-loop formalization via LLMs (GPT-4, GPT-4V).
  - Automated verification of formalized proofs via SMT solving.
  - Performance evaluation (semantic similarity, correctness).

---

## 2. Data Collection and Construction
### a. Formalization Dataset
- **Euclid Elements & UniGeo:** 
  - Obtain formal proofs of Euclid elements (~173 proofs) and UniGeo (~125 problems).
  - These are used as ground truth formal proofs.
- **Data for Autoformalization Task:**
  - For each informal problem, extract the human-readable proof steps, diagram context, and problem statement.
  - Derive formal problem statements, hypotheses, and goals based on Euclid's axioms and previous formal proofs.
  - Generate “informal proofs” in natural language, either from Euclid texts or human annotations.

### b. Diagrammatic Contexts
- **Diagrammatic Data:**
  - Collect or generate diagram images corresponding to each problem, possibly from the UniGeo dataset or Euclid problem statements.
  - Simplify diagram inputs: use geometric primitives (points, lines, circles, angle relations).

### c. Human Formal Proofs
- **Create or Use a Corpus:**
  - Use the available formal proofs as the reference standard.
  - Organize proofs into a consistent structure (sequence of tactics aligned with Euclid's reasoning style).

---

## 3. Prompt Design for LLM Autoformalization
### a. Human-readable prompt steps (as per paper’s templates)
- **Template for informal proofs:** 
  - Include problem statement, diagram context, and the informal proof narrative.
  - Use the structured prompt format outlined, e.g.:
    - Input: "If a triangle has two angles equal..."
    - Output: “Let the triangle be...”
- **Explicitly list the proof tactics:**
  - Encourage the LLM to generate tactic sequences, e.g., `euclid_apply`, `euclid_intro`, `euclid_apply proposition_X`, etc.

### b. Formal proof style
- Follow the syntax of the `lean Euclid` formalization:
  - Each tactic should be clearly listed.
  - Diagrams should be referenced explicitly with commands that construct or reuse geometric primitives.
  - Use geometric reasoning tactics (apply propositions, establish congruences, angle equalities).

### c. Diagrams and symbolic reasoning
- Provide diagrammatic hints (e.g., "triangle ABC, with sides AB, BC, AC").
- When prompting the LLM, include instructions on diagram interpretation:
  - "Based on the diagram, identify points, lines, and angles, and their relations."

---

## 4. Deep Integration with Formal System (Lean)
### a. Formal axioms & definitions
- Implement or import axioms in the formal Lean library, e.g., points, lines, circles, angles, and their relations.
- Implement geometric primitives and their properties: e.g.,
  - Equal length, angle congruences, parallelism, congruence of triangles.
- Include axioms as in Euclid's Elements and UniGeo.

### b. Formal tactics
- Extend or emulate tactics such as:
  - `euclid_apply`, `euclid_intro`, `euclid_proposition`, `euclid_between`, `euclid_angle_equals`, etc.
- Automate the pattern: 
  - **Apply proposition → derive new relations→ verify geometric relations → finish proof.**

### c. Automate proof verification
- Use Lean's proof checker to validate each generated proof.
- In case of failure, analyze whether the failure is due to missing tactics, incomplete formalization, or errors in the automated tactic sequence.

---

## 5. Automated Geometric Reasoning via SMT
### a. Formalize geometric predicates
- Encode geometric relations (collinearity, segment lengths, angle measures, congruence) as SMT formulas.
- Use a theorem prover: e.g., `Z3` SMT solver, with a custom frontend that interprets geometric predicates and relations.

### b. Equivalence Checking for Validity
- Given the formalized proof and the ground truth, check:
  - **Logical equivalence** via SMT solver: Can the formal proof be proved to be equivalent to the ground truth?
  - Partial verification: If fully verified isn't possible, measure the ratio of verified propositions or tactics success.

### c. Proof Repair & Fine-tuning
- Use the approach like "repair" in the paper: edit or patch proofs automatically where minor mismatches occur.
- Evaluate partial correctness by inferences of SMT, especially for loose or underspecified steps.

---

## 6. Experiment Design & Hyperparameters
### a. Model Settings
- Use **GPT-4 & GPT-4V** as primary autoformalizers.
- Test various prompt formats:
  - Few-shot prompting (5 examples per problem).
  - Zero-shot prompting for comparison.
  - Chain-of-thought prompts encouraging tactic sequencing.
- For GPT-4V, include diagram images in prompts, converting images to accessible diagrams or text annotations.

### b. Hyperparameters
- Temperature: set to low (`0.1-0.2`) to improve deterministic output.
- Max tokens: limit based on prompt length and expected output.
- Few-shot examples: 5 exemplars per problem.

### c. Evaluation metrics
- **Semantic similarity:** Levenshtein ratio between the autoformalized proof text and the ground truth.
- **Logical equivalence:** SMT proof success rate for verifying proof correctness.
- **Proof completeness:** Number of tactics matches for ground truth.

---

## 7. Implementation Workflow
- **Step 1:** Collect and preprocess Euclid and UniGeo formal proofs & diagrams.
- **Step 2:** Generate human-readable informal proofs & problem statements.
- **Step 3:** Design prompts based on the templates; include diagram annotations.
- **Step 4:** Run prompts through GPT-4 / GPT-4V.
- **Step 5:** Parse model outputs to tactics formalization.
- **Step 6:** Use Lean to check proofs syntactically and semantically.
- **Step 7:** Use SMT solvers to verify logical equivalence, perform repair if needed.
- **Step 8:** Collect metrics, analyze success cases, and iterate prompt / formalization improvements.

---

## 8. Reporting & Validation
- Quantify performance across the dataset:
  - Formal proof correctness rate.
  - Semantic similarity scores.
  - SMT verification success ratio.
- Perform qualitative analysis on:
  - Correctly formalized proofs.
  - Failures and common error modes.
- Compare with baseline human formalization efforts or prior automated methods.

---

**Note:** Throughout this process, loop back to refine prompts, improve geometric predicate encodings, and expand the formal axiomatic base to cover missing Euclidean relations.

---

This roadmap aligns with the methods, datasets, and evaluation metrics described in the paper, providing a comprehensive foundation for subsequent code development and experiments.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular system that leverages open-source tools: OpenAI's API for GPT-4 and GPT-4V (via API calls), Lean's theorem prover for formal verification, Z3 SMT solver for logical equivalence checks, and a custom prompt engineering pipeline. The core workflow involves generating formal proofs from human-readable Euclidean geometry statements using GPT models, parsing the output into formal tactics, and verifying proofs within Lean. Diagram inputs will be processed as images (for GPT-4V) and translated into formal geometric primitives. The system includes a dataset loader, a prompt template engine, a proof parser, and verification modules, all orchestrated by a main experiment runner.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "prompt_engineer.py",
        "proof_parser.py",
        "lean_verifier.py",
        "smt_checker.py",
        "diagram_processor.py",
        "utils.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run()\n    }\n    class DatasetLoader {\n        +__init__(data_path: str)\n        +load_dataset() -> List[dict]\n    }\n    class PromptEngineer {\n        +__init__(template: str)\n        +generate_prompt(problem: dict, diagrams: Optional[Image]=None) -> str\n    }\n    class ProofParser {\n        +parse_gpt_output(output: str) -> List[Tactic]\n    }\n    class LeanVerifier {\n        +__init__(lean_path: str)\n        +verify_proof(tactic_list: List[Tactic], proof_goal: str) -> bool\n    }\n    class SMTChecker {\n        +__init__(solver_path: str)\n        +check_equivalence(formula1: str, formula2: str) -> bool\n    }\n    class DiagramProcessor {\n        +__init__(image_path: str)\n        +extract_geometric_primitives() -> Dict[str, Any]\n    }\n    class Tactic {\n        +name: str\n        +parameters: dict\n    }\n    Main --> DatasetLoader\n    Main --> PromptEngineer\n    Main --> ProofParser\n    Main --> LeanVerifier\n    Main --> SMTChecker\n    Main --> DiagramProcessor\n    ProofParser --> Tactic\n    LeanVerifier --> Tactic\n    SMTChecker --> str\n    DiagramProcessor --> dict\n",
    "Program call flow": "sequenceDiagram\n    participant C as Main\n    participant DL as DatasetLoader\n    participant PE as PromptEngineer\n    participant PRS as ProofParser\n    participant LV as LeanVerifier\n    participant SC as SMTChecker\n    participant DP as DiagramProcessor\n    C->>DL: load_dataset()\n    DL-->>C: dataset\n    C->>PE: generate_prompt(problem, diagrams)\n    PE-->>C: prompt_str\n    C->>GPT_API: call(prompt_str)\n    GPT_API-->>C: gpt_output\n    C->>PRS: parse_gpt_output(gpt_output)\n    PRS-->>C: tactic_list\n    C->>DP: process_diagrams(problem_images)\n    DP-->>C: geometric_primitives\n    C->>LV: verify_proof(tactic_list, goal)\n    LV-->>C: proof_valid\n    C->>SC: check_equivalence(formula1, formula2)\n    SC-->>C: is_equivalent\n    C->>Main: store_results()\n    Note over C: repeats for all dataset entries\n",
    "Anything UNCLEAR": "Clarification needed on specific prompt templates, particularly how to incorporate diagrams reliably for GPT-4V, and details of parsing GPT outputs into formal tactics accurately. Also, instructions on patching proofs when SMT verification fails are needed."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "openai",
        "torch==1.9.0",
        "transformers",
        "z3-solver",
        "Pillow",
        "scikit-image"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader: Loads Euclid/UniGeo formal proofs, human-readable problems, diagrams, and ground truth references. Depends on file I/O and data parsing modules."
        ],
        [
            "prompt_engineer.py",
            "Implements PromptEngineer: Generates NLP prompts based on templates, problem data, and optional diagram inputs. Depends on prompt templates and problem formatting logic."
        ],
        [
            "diagram_processor.py",
            "Implements DiagramProcessor: Extracts geometric primitives from diagram images using image processing libraries (e.g., scikit-image, Pillow). Outputs geometric feature dictionaries to be embedded in prompts."
        ],
        [
            "proof_parser.py",
            "Defines ProofParser: Parses GPT output into sequence of tactics and formal proof structures. Needs to understand the expected tactic syntax and output patterns."
        ],
        [
            "lean_verifier.py",
            "Defines LeanVerifier: Uses Lean's proof checker API or local installation to verify formal proofs generated from tactics. Ensures proof correctness."
        ],
        [
            "smt_checker.py",
            "Uses Z3 or an SMT solver API: Checks the logical equivalence of the formal proof against ground truth formulas, facilitating proof quality metrics."
        ],
        [
            "main.py",
            "Coordinates all modules: Loads dataset, generates prompts, calls GPT-4/4V APIs, parses output, runs Lean proof verification, performs SMT equivalence checks, and records metrics. Relies on the orchestrator pattern."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "diagram_processor.py",
        "prompt_engineer.py",
        "proof_parser.py",
        "lean_verifier.py",
        "smt_checker.py",
        "main.py"
    ],
    "Full API spec": ""
    ,
    "Shared Knowledge": "Common configurations like GPT API keys, Lean installation paths, and API endpoints are shared modules or environment variables. The prompt templates are standardized and version-controlled.",
    "Anything UNCLEAR": "Clarification needed on the specific prompt templates for various theorem types and the expected format of GPT output. Confirmation on the format and API for Lean proof verification, especially whether local Lean server or external API is used."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
dataset:
  euclid_proofs_path: "path/to/euclid_proofs"
  unigeo_dataset_path: "path/to/unigeo_dataset"
  diagram_image_dir: "path/to/diagram_images"

model:
  openai_api_key: "your-openai-api-key"
  gpt_model: "gpt-4"  # or "gpt-4V" for visual input
  few_shot_examples: 5
  temperature: 0.2
  max_tokens: 1500

prompt:
  template: |
    {problem_statement}
    Diagram context (if available): {diagram_description}
    Generate formal tactics sequence following the style:
    1. {prompt_template}
    ...
    8. Finish and verify in Lean.

verifier:
  lean_path: "/path/to/lean"  # path to local Lean installation or API endpoint
  z3_solver_path: "path/to/z3"  # optional if using API

training:
  epochs: 10  # Note: actual training epochs depend on the process; here, training mainly involves prompt testing
  batch_size: 4  # Batch size for batch inference (if applicable)
  learning_rate: 0.001  # Hyperparameter for optional tuning, not directly used in inference
  validation_split: 0.2

evaluation:
  metrics:
    semantic_similarity_threshold: 0.6
    verification_success_threshold: 0.8  # proportion of proofs verified correct
```

---

**Note:** Full configuration is available in `planning_config.yaml`
