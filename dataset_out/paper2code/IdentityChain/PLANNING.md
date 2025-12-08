# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly. Here's a comprehensive plan structured into key sections that reflect the paper's methodology, experimental design, and evaluation procedures, as derived from the provided content:

---

# 1. Understanding and Formalizing the Core Concepts

**a. Self-Consistency & Strong Self-Consistency**
- *Definitions*: Implement the formal definitions of self-consistency and strong self-consistency.
  - Use the interpretation function `sem` (semantics) that maps NL and PL (natural language specifications and programs) to a semantic space `D`.
  - Self-consistency requires that, for all iterations `i`, the semantics of the generated NL and PL are equal.
  - Strong self-consistency additionally requires that the initial NL and PL have the same semantics.

**b. Translation Functions**
- *NL→PL (`M_{n2p}`)* and *PL→NL (`M_{p2n}`)*:
  - These are model-invoked functions.
  - Implemented via prompting large language models (LLMs); in the code, simulated as model API calls.
- *Iterative Chain Construction*:
  - Given initial NL (`nl_0`), generate `pl_0`.
  - Then, repeatedly:
    - Generate `nl_{i+1}` from `pl_i`.
    - Generate `pl_{i+1}` from `nl_{i+1}`.
  - Continue for `n` iterations or until early stopping criteria.

---

# 2. Framework Design: IdentityChain

**a. Input & Initialization**
- Input initial natural language specification `nl_0`.
- Prompt the model (using `M_{n2p}`) to generate the initial program `pl_0`.

**b. Chain Execution & Evaluation**
- For each step `i` (from 0 to `n-1`):
  - Generate `nl_{i+1} = M_{p2n}(pl_i)` (PL→NL decoding).
  - Generate `pl_{i+1} = M_{n2p}(nl_{i+1})` (NL→PL decoding).
- Use *greedy decoding* (deterministic) at each step for efficiency.
- Implement early stopping:
  - If `pl_{i+1}` matches `pl_i` exactly, or `nl_{i+1}` matches `nl_i`, assume models are stable and stop further iterations for that input.

**c. Output & Metrics**
- For each input, record whether semantic semantics are preserved across all iterations:
  - Use automated metrics, primarily the *Test Output Match (TOM)* score, which compares model outputs/test results.
- Compute:
  - Overall self-consistency score `SC_n`.
  - Overall strong self-consistency score `SSC_n`.

---

# 3. Evaluation Metrics & Approximate Semantic Equivalence

**a. Test Output Match (TOM)**
- Generate test cases from the initial program.
- Run the program `pl_i` and `pl_{i+1}` on all test cases.
- Compute the percentage of exact output matches.
- Use this as an approximation for semantic equivalence of `sem(pl_i)` and `sem(pl_{i+1})`.

**b. Test Case Generation**
- For each `pl_i`, generate a set of representative and comprehensive test cases.
- For programs with multiple test cases, execute both `pl_i` and `pl_{i+1}`.
- Record whether the output matches exactly for each test case.

**c. Aggregate Scores**
- Compute per-input binary scores:
  - `sc_{n,j}`: all semantic equalities hold within `n` steps for test input `j`.
  - `ssc_{n,j}`: strictly, semantics are identical initially and for all `i`.
- Finally, compute macro scores:
  - `SC_n` = average over `m` inputs of `sc_{n,j}`.
  - `SSC_n` = average over `m` inputs of `ssc_{n,j}`.

---

# 4. Experimental Setup & Datasets

**a. Datasets**
- *HumanEval*: 164 hand-crafted Python problems.
- *HumanEvalPlus*: Use the Mini-v0.1 version with each problem ~16 test cases.
- *MBPP*: 974 crowd-sourced problems; focus on the sanitized 257-problem subset with 3 test cases.
- *Function Name Replacement*: Replacing meaningful function names with generic "func" for robustness, especially when evaluating semantics and self-consistency.

**b. Model Selection**
- *Baseline Models*: Open-source models (StarCoderBase, Code Llama).
- *Instruction Tuning*: Use instruction-tuned variants (e.g., CodeLlama-Inst, StarCoder-Inst, GPT-4, GPT-3.5 variants).
- *Model APIs*: Use API calls or open-source model interfaces (like Huggingface transformers, OpenAI API).

**c. Prompts & Decoding Settings**
- One-shot prompts for open-source models; chat-style prompts (system, user, assistant roles) for instruction-tuned models.
- Use *greedy decoding* for the chain to simulate deterministic model behavior.
- Chain length `n`: standard choice up to 5 steps; verify early-stopping conditions.

**d. Hyperparameters**
- Temperature: Close to zero (0) to ensure deterministic outputs.
- Max generation tokens:
  - For NL→PL and PL→NL, set sufficiently large values (e.g., 1024 tokens).
- Number of test cases:
  - Use all available/tested test cases or a representative subset if the set is large.

---

# 5. Specific Experiment Variants & Robustness Checks

**a. Temperature Sensitivity**
- Evaluate at multiple temperatures (e.g., 0, 0.2, 0.6) for greedy sampling, confirm the consistency of self-consistency scores.

**b. Error Analysis & Failure Modes**
- Implement functions to capture:
  - Syntax errors (detect via exception catching).
  - Runtime errors (collect error messages).
  - Mismatches in output or semantics.

**c. Replacing Function Names with a Generic Label**
- For each `nl_i` and `pl_i`, replace meaningful function names with a generic name like "func" before invoking the model.
- Purpose: control for semantic confusion caused by function names.

---

# 6. Implementation Details & Practical Considerations

**a. Automation & Scheduling**
- Automate multiple input evaluations.
- Use parallel execution where possible, especially for independent inputs.

**b. Human vs. Automated Semantic Verification**
- Automate primarily via test-based metrics (TOM).
- Optionally, sample a subset manually for validation.

**c. Performance & Resource Tracking**
- Record model response time, test execution time.
- Keep logs of early stopping points and mismatches for debugging.

---

# 7. Additional Tools & Extensions

**a. Identity Variations & Debugging**
- Use `IdentityChain` to explore weak spots of models:
  - e.g., implicit semantics, data types, code execution.
- Develop functions to analyze the chain for:
  - Fluctuations in `SC` and `SSC` scores.
  - Specific conclusions about model weaknesses.

**b. Extending Evaluation**
- Explore *multi-task* self-consistency by constructing multiple nl specifications per problem.
- Augment test coverage for robustness.

---

# Summary of the Roadmap
- Implement functions for model prompting (`M_{n2p}`, `M_{p2n}`).
- Automate chain execution with early stopping based on exact matches.
- Generate comprehensive test sets for each generated program.
- Use `TOM` (test output match) as the primary semantic similarity metric.
- Calculate macro self-consistency scores across multiple inputs.
- Extend evaluations over various temperatures, prompt styles, and model variants.
- Replace function names with generic tokens to stabilize semantics.
- Analyze results for stability, weaknesses, and insights into models' semantic understanding.

---

Let me know if you'd like me to prepare a detailed coding framework (functions/classes) based on this roadmap.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will create a modular system that prompts open-source language models (using APIs like OpenAI or local models via Huggingface transformers) to execute the IdentityChain methodology. The main pipeline will initialize with an input NL specification, generate a PL program, then iteratively produce NL and PL pairs, evaluating semantic consistency via test execution match scores. The system will include functions for chain dynamics, early stopping, and score aggregation. Automated test case execution will use the `subprocess` or `exec` functions for safe code execution, with test case inputs supplied and outputs captured. Results will be summarized with scores for self-consistency and strong self-consistency. Replacing function names with 'func' and controlling prompt templates ensures semantic stability. The entire workflow will be designed to run efficiently on multiple inputs, supporting parallelization where possible, with configuration parameters adjustable for prompts, decoding, and chain length.",
    "File list": [
        "main.py",
        "prompt_engineering.py",
        "model_api.py",
        "test_executor.py",
        "evaluation.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run() -> None
    }
    class PromptEngine {
        +__init__(prompt_template: str, model_api: ModelAPI)
        +generate_nl_from_pl(pl_code: str, prompt_params: dict) -> str
        +generate_pl_from_nl(nl_description: str, prompt_params: dict) -> str
    }
    class ModelAPI {
        +__init__(api_key: str, model_name: str, temperature: float)
        +call_model(prompt: str, max_tokens: int) -> str
    }
    class ChainExecutor {
        +__init__(prompt_engine: PromptEngine, max_chain_length: int, early_stop: bool)
        +execute_chain(nl_initial: str, test_cases: list) -> dict
            # Returns scores and chain details
    }
    class TestExecutor {
        +__init__()
        +execute_test(pl_code: str, test_inputs: list) -> list
    }
    class EvaluationMetrics {
        +compute_tom(test_outputs: list, expected_outputs: list) -> float
        +aggregate_scores(chain_results: dict) -> dict
    }
    Main --> PromptEngine
    Main --> ChainExecutor
    Main --> ModelAPI
    ChainExecutor --> PromptEngine
    ChainExecutor --> TestExecutor
    ChainExecutor --> EvaluationMetrics
    TestExecutor --> ModelAPI
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant PE as PromptEngine
    participant API as ModelAPI
    participant CE as ChainExecutor
    participant TE as TestExecutor
    participant EM as EvaluationMetrics

    M->>PE: initialize with prompt_template, model_api
    PE-->>M: prompt functions ready
    M->>CE: execute_chain(nl_0, test_cases)
    CE->>TE: execute_test(pl_code, test_cases)
    TE-->>CE: test_outputs
    CE->>EM: compute_tom(test_outputs, expected_outputs)
    EM-->>CE: scores
    CE-->>M: final scores, chain details
    M-->Evaluation: report and analyze
",
    "Anything UNCLEAR": "Clarify the exact format and source of test cases, whether they are generated automatically or predefined, and confirm if API keys or local models are preferred for prompt execution."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "transformers==4.29.0",
        "torch==1.12.1",
        "numpy==1.21.0",
        "pandas==1.3.5",
        "requests==2.28.1",
        "scipy==1.7.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "main.py",
            "Main entry point. Orchestrates the workflow: initializes configurations, invokes prompt engineering, runs chain execution, and aggregates results."
        ],
        [
            "prompt_engineering.py",
            "Contains classes and functions to build prompts, including prompt templates for NL→PL and PL→NL conversions, replacements of function names with 'func', and prompt parameter management."
        ],
        [
            "model_api.py",
            "Defines ModelAPI class, which manages API calls to large language models (e.g., OpenAI API or local Huggingface models) with explicit functions like call_model(prompt, max_tokens). Handles prompt sending and response capturing."
        ],
        [
            "test_executor.py",
            "Implements TestExecutor class, capable of executing generated programs with input parameters, capturing outputs, handling exceptions, and returning test results for TOM scoring."
        ],
        [
            "evaluation.py",
            "Provides EvaluationMetrics class to compute TOM scores, aggregate chain scores into self-consistency metrics, and manages output report generation."
        ],
        [
            "utils.py",
            "Includes utility functions, e.g., replacing function names in code strings with 'func', generating synthetic or extracted test cases, executing code safely (via subprocess or exec), and handling early stopping criteria based on exact matches."
        ]
    ],
    "Task list": [
        "utils.py",                  # Develop utility functions: replace function names, execute code safely, generate test cases, and handle early stopping.
        "model_api.py",              # Implement API interface to communicate with LLMs; support prompt construction, API calls, and response parsing.
        "prompt_engineering.py",     # Create prompt templates and functions for consistent prompt generation, including name replacement.
        "test_executor.py",          # Develop code execution functions, test case input generation, and output comparison for TOM scoring.
        "evaluation.py",             # Build scoring functions for TOM, aggregate scores, and provide result summaries.
        "main.py"                    # Final script to glue all components: load config, initialize classes, run chain iterations, perform evaluations, and output results.
    ],
    "Full API spec": "openapi: 3.0.0\n\nComponents:\n  schemas:\n    EvaluationResult:\n      type: object\n      properties:\n        self_consistency_score:\n          type: number\n          description: Average self-consistency score over all inputs.\n        strong_self_consistency_score:\n          type: number\n          description: Average strong self-consistency score.\n        details:\n          type: array\n          items:\n            type: object\n            properties:\n              input_id:\n                type: integer\n              scores:\n                type: object\n                properties:\n                  tom:\n                    type: number\n                  exact_match:\n                    type: boolean\n  paths:\n    /evaluate:\n      post:\n        summary: Evaluate the model chain for self-consistency\n        requestBody:\n          required: true\n          content:\n            application/json:\n              schema:\n                type: object\n                properties:\n                  nl_initial:\n                    type: string\n                  test_cases:\n                    type: array\n                    items:\n                      type: object\n                      properties:\n                        input:\n                          type: any\n                        expected_output:\n                          type: any\n        responses:\n          '200':\n            description: Evaluation results including self-consistency scores\n            content:\n              application/json:\n                schema:\n                  $ref: '#/components/schemas/EvaluationResult'",
    "Shared Knowledge": "Reuse prompt templates with placeholders for function names, test case input generation, and code execution. Implement a versioning system for models and test cases. Standardize response parsing to handle different model APIs. Modularize the utility functions for interchangeable components like dynamic code execution and name replacements.",
    "Anything UNCLEAR": "Clarify whether local or API-based model deployments are preferred, especially for large-scale experiments. Confirm the exact format for test case inputs and expected outputs, and whether synthetic or derived test cases are preferred. Clarify if the system should support only Python or multiple programming languages."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.0001  # (Assumed default for fine-tuning large models; paper doesn't specify)
  batch_size: 16        # (Typical small batch size for API-based models)
  epochs: 3             # (Standard minimal fine-tuning epoch count; paper does not specify)
  max_tokens: 1024      # Max tokens for model input/output during prompts
  temperature: 0.0      # Deterministic decoding to ensure reproducibility
  chain_length: 5       # Maximum chain iterations before early stopping
evaluation:
  test_case_count: 16   # Number of test cases per program (based on HumanEvalPlus)
  scoring_method: "TOM" # Test Output Match as primary semantic similarity metric
models:
  openai_api_key: "<your-api-key>"  # Placeholder for API key if using OpenAI models
  models_list:
    - name: "GPT-4"
      type: "API"
    - name: "GPT-3.5"
      type: "API"
    - name: "StarCoderBase"
      type: "local"
      model_path: "path/to/starcoderbase/model"
    - name: "CodeLlama"
      type: "local"
      model_path: "path/to/codellama/model"
prompt_templates:
  nl2pl_prompt: "Replace meaningful function names with 'func' and generate code for the specified task: {task_description}. Use only the provided test cases for validation."
  pl2nl_prompt: "Summarize the following program into an unambiguous natural language description, replacing function names with 'func'. Do not infer implicit behaviors."
  chain_prompt_params:
    chain_steps: 5
    early_stop_on_exact_match: true
```

---

**Note:** Full configuration is available in `planning_config.yaml`
