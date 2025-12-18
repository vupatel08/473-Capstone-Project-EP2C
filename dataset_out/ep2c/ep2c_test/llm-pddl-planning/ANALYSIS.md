# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## app.py

# Logic Analysis for `app.py`

This file serves as the central orchestrator, coordinating all components: environment interaction, initial NL-to-PDDL generation, iterative refinement guided by EW metrics, plan validation, logging, and final result compilation. The goal is to implement a comprehensive main workflow adhering strictly to the methodology outlined in the paper, the detailed plan, and the formal model, ensuring reproducibility and fidelity to the original approach.

---

### **High-Level Workflow**

1. **Initialization:**
   - Read configuration parameters from `config.yaml`.
   - Load or define environment specifics (object list, action interface, environment description).
   - Instantiate environment interaction interface (class `Environment`) with environment API signatures.
   - Instantiate LLM manager (`LLMManager`) for prompt generation and refinement.
   - Instantiate PDDL generator (`PDDLGenerator`) for initial generation and refinement.
   - Instantiate metrics calculator (`Metrics`) for EW scores, plan validation, and domain similarity.
   - Instantiate logger for results tracking.

2. **Problem and Domain Initialization:**
   - Receive the NL description of environment (`nl_desc_env`) and task (`nl_desc_task`).
   - Generate initial problem PDDL candidates via `LLMManager`, feeding the NL task description.
   - Generate initial domain PDDL candidates similarly, based on environment NL description.
   - Select top candidates based on preliminary EW scores or default choices.
   - Set chosen initial domain and problem files as current working versions.

3. **Iterative Refinement Loop:**
   - For each environment:
     - For each task:
       - **Plan Generation and Validation:**
         - Use classical planner with current PDDL files (`domain.pddl` and `problem.pddl`) to produce a plan.
         - Use environment sim/execution to run the plan (via `execute_plan()`):
           - If plan executes successfully and reaches goal, mark task as solved; log success.
           - Else, analyze failure.
         - If plan fails:
           - Compute EW score between current domain and environment model (`calculate_ew_score()`), sampling action sequences.
           - Provide EW feedback and environment signals to LLM prompt to generate domain refinement.
           - Update domain PDDL via refinement function (`refine_domain()` or similar).
           - Repeat plan generation with new domain, cycle until success or max iterations.
           
       - Record metrics: solve success, EW score, number of iterations, resource/time consumed.
       
   - For environments and tasks that do not succeed after max iterations:
     - Log failure, EW scores, and relevant metrics.

4. **Post-Processing and Output:**
   - Aggregate results across all environments/tasks.
   - Save logs, domains, problems, scores, environment states.
   - Produce summary statistics: success rate, average EW, iteration counts.
   
5. **Optional:**
   - Save environment states and PDDL files for reproducibility.
   - Generate visualizations of EW scores, success over iterations.
   - Generate detailed logs for debugging/analysis.

---

### **Critical Components & Logic Details**

**1. Configuration Loading**
- Parse `config.yaml`.
- Extract parameters:
  - Environment: max sampling length (`T_max`), number of samples (`num_samples`).
  - Refinement: max refinements (`max_refinement_iterations`), EW threshold (`ew_threshold`), plan success rate threshold (`plan_success_rate_threshold`).
  - Prompting templates: load or define inline.
  - Planning: select solver, time limits.
  - Logging: directory, level, resource saving flag.

**2. Environment Initialization**
- Instantiate environment object:
  ```python
  env = Environment(api_signatures=..., env_id=...)
  ```
- Environment must provide:
  - `execute_plan(domain_file, problem_file, plan) -> bool`
  - `check_feasibility(actions_list) -> List[bool]`
  - `get_state() -> dict`
  - `sample_action_sequences(T_max, num_samples) -> List[List[str]]`

**3. LLM Manager and PDDL Generator Setup**
- Instantiate `LLMManager` with API key.
- Instantiate `PDDLGenerator`.
- Implement prompt templates: use string formatting with environment description, NL task description, current domain/problem files, EW scores, environment state during refinement.

**4. Generate Initial PDDL Candidates**
- Use `LLMManager`:
  ```python
  problem_candidates = generate_initial_problem(nl_desc_task, object_list, api_signatures)
  domain_candidates = generate_initial_domain(nl_desc_env, api_signatures)
  ```
- For each candidate, save as working version, prepare for evaluation.

**5. Plan Generation & Evaluation Loop**
- For each task:
  - Initialize iteration counter.
  - For each iteration:
    - Call classical planner with current domain/problem files.
    - If plan found:
      - Run plan in environment:
        ```python
        success = env.execute_plan(domain_file, problem_file, plan)
        if success:
            mark success, break
        else:
            # Plan invalid or incomplete: proceed to refinement
        ```
      - If success: log, record, break.
    
    - Else:
      - Compute EW score:
        ```python
        ew_score = metrics.calculate_ew_score(current_domain, environment, seqs_T)
        ```
      - Generate refinement prompt with feedback signals:
        ```python
        new_domain = llm_manager.refine_prompt(current_domain, ew_score, env_state)
        ```
      
      - Update domain file:
        ```python
        current_domain = new_domain
        ```
      
    - Check if iteration count exceeds max; if so, stop and log failure.
  - Record success/failure, final EW score, iterations, resources.

**6. Final Result Compilation**
- Summarize stats:
  - Overall success rate per environment.
  - Mean/median EW scores.
  - Total time/resource consumption.
- Save logs to specified directory.
- Save final PDDL files and environment states.

---

### **Error Handling & Edge Cases**
- Syntax or semantic errors in generated PDDL:
  - Parse PDDL files; if errors, generate correction prompts via `llm_manager.py`.
- Non-executability of plans:
  - Detect via environment signals; trigger refinement.
- No plan found:
  - Use EW-based feedback to guide LLM.
  - Limit refinement iterations.
- Environment failure:
  - Record and proceed.

---

### **Evaluation & Reproducibility**
- Precisely capture prompt content, API signatures, hyperparameters.
- Record all environment states, generated files, scores.
- Log intermediate responses for debugging.
- Store random seeds, parameters, hardware info.

---

### **Summary**
The `app.py` main script should:
- Initialize all components based on configuration.
- Generate initial candidate PDDL files.
- Loop over environment/task pairs:
  - Generate plans.
  - Evaluate via environment signaling.
  - When failure occurs:
    - Calculate EW scores.
    - Prompt LLM for domain refinements.
    - Update domain/problem PDDL.
  - Repeat until success or max iterations.
- Log success, resource consumption, success rate, and final models.
- Ensure fully reproducible, traceable pipeline per methodology described in the paper.

This thorough analysis forms the blueprint for implementing `app.py`. It enforces fidelity to the experimental design, clarity in logic flow, and adherence to reproducibility principles.

## environment.py

# environment.py - Logic Analysis

This module implements the Environment class, which encapsulates all interactions with the environment/simulator necessary for the PDDL refinement system. The logic detailed below ensures that Environment provides the core functionalities: executing plans, verifying action sequences, sampling random action sequences, and querying state information. The implementation must adhere precisely to the paper's methodology, assumptions, and interfaces, while being flexible enough to support multiple environment types (e.g., virtual PDDL simulators, physics engines, or real-world systems).

---

## 1. Core Responsibilities & Required Methods

### a) Initialization: `__init__(api_signatures: dict, env_id: str)`
- Load environment-specific API signatures (action names, parameter structures).
- Set environment identifier (e.g., domain name) for logging, environment-specific logic, or API endpoints.
- Prepare or connect to the environment simulation engine or real system.
- Setup environment state representation and interaction functions.

### b) Plan execution: `execute_plan(plan: str) -> bool`
- Inputs:
  - `plan`: a list of actions in string format (probably in PDDL action notation, e.g., `(move robot1 room2 room3)`).
- Processing:
  - Sequentially parse actions.
  - For each action:
    - Decompose into action name and parameters.
    - Verify preconditions (via environment-specific check).
    - Apply effects to environment state.
    - Detect errors or infeasible actions.
  - Check if the final environment state satisfies the goal conditions (if relevant).
- Outputs:
  - Boolean: `True` if plan successfully executed to goal; `False` if an infeasible or error-triggering action occurs.
- Implementation notes:
  - Should catch environment-specific exceptions.
  - Must maintain the environment state internally; update after each action.
  - Should support rollback or simulation modes if needed for testing.

### c) Feasibility check: `check_feasibility(actions: List[str]) -> List[bool]`
- Inputs:
  - A sequence of actions (string format).
- Processing:
  - For each action:
    - Verify if it can be executed from the current state.
    - Use environment API (e.g., API call, logical check).
  - Usually designed to return an array of booleans corresponding to each action.
- Use:
  - To pre-validate sequences sampled from the environment or generated by the LLM during EW sampling.
- Notes:
  - Should be optimized for batch checking if possible.

### d) State querying: `get_state() -> dict`
- Outputs:
  - Current environment state representation as a dictionary.
  - State should encode all relevant info needed by the PDDL generator and for environment feedback.
- Contents:
  - Object positions, object statuses, environment variables.
  - Environment-specific state features, e.g., 'robot_positions', 'object_statuses', 'instrument_statuses', etc.
- Usage:
  - To inform the LLM prompt, provide state description, or verify goal achievement.

### e) Sampling action sequences: `sample_action_sequences(sequence_length: int, num_samples: int) -> List[List[str]]`
- Inputs:
  - `sequence_length`: number of actions per sample (`T_max` in EW).
  - `num_samples`: number of sequences to generate (e.g., 4).
- Processing:
  - Randomly generate sequences respecting the environment constraints:
    - Use the environment's action interface/API to determine possible actions at each step.
    - Randomly select actions uniformly from the set of possible actions.
    - Build sequences step-by-step, updating the simulated environment state or sampling based on known transition probabilities.
- Outputs:
  - List of action sequences (each as list of string actions).
- Notes:
  - Should avoid infeasible sequences (sampling only from executable actions).
  - For efficiency and consistency, may leverage the environment's own `check_feasibility()` internally during sampling.

---

## 2. Underlying Assumptions & Interface Constraints

- **Environment API**:
  - Access to a set of available actions, with parameters (via `api_signatures`).
  - Ability to verify preconditions and effects either through:
    - Environment state update functions, 
    - Or explicit feasibility/verification API calls.
- **Action strings**:
  - Assume a standard formalism `(action_name param1 param2 ...)`.
  - Parsing functions needed to extract action name and parameters reliably.
- **State Management**:
  - The environment must internally maintain an up-to-date state. It can initialize from a default or provided seed.
  - Each action modifies this state in accordance with environment rules.
- **Error Handling & Robustness**:
  - Infeasible actions or errors should be detected and handled gracefully.
  - Failures in execution or infeasibility should return `False`/`None` as appropriate.
- **Environment Feedback**:
  - The class must provide accurate execution results, including whether the plan was fully executable and whether goals were achieved (if relevant).

---

## 3. Implementation Details & Design Strategy

### a) Internal State Representation
- Use environment-specific data structures:
  - Object locations/containers.
  - Status flags (e.g., `holding`, `free`).
  - Environment variables (e.g., orientation, calibration states).

### b) Execution Engine
- For virtual environments:
  - Wrap environment simulation SDKs or code.
  - Implement a step-by-step execution of actions.
- For real environments:
  - Send API commands.
  - Wait for acknowledgment and error codes.
- Simulation mode (for testing EW sampling):
  - Use internal models or dummy data to mimic environment physics.

### c) Action Parsing
- Provide a helper function `parse_action(action_str: str) -> (action_name, parameters)`:
  - Parse PDDL notation string.
  - Return sorted, structured data for verification and effect application.

### d) Feasibility Verification
- For each action:
  - Check current state.
  - Verify preconditions explicitly.
- Precondition check:
  - Could be a direct logical check if environment supports (e.g., Python functions).
  - Or use environment API if it provides such checks.

### e) State Update & Effects
- When executing actions:
  - Update environment state according to action effects.
  - Enforce effects based on environment rules.

### f) Sampling Sequences
- Generate actions uniformly from the set of allowed actions at each step.
- Use `check_feasibility()` to filter sequences.
- For consistency, may use environment's own function to get available actions in current state (`Ad,p(s)`).

### g) Error & Exception Management
- Detect invalid actions, infeasible steps, environment errors.
- Return failure indicators or raise exceptions as needed.
- Log errors for debugging and refining.

### h) Interaction with External Process
- If simulating:
  - Maintain an in-memory environment model.
- If interacting with real environment:
  - Implement API calls with retries, timeouts, and error checks.

---

## 4. Integration Considerations

- `Environment` class must be compatible with the overall plan:
  - Invoke `execute_plan()` with plan string to run in the real or simulated environment.
  - Use `check_feasibility()` during EW sampling phase.
  - Provide `get_state()` for state descriptions in prompts.
  - Sample sequences that are likely executable to avoid unnecessary failures.
- For multiple environment types or domains:
  - Implement environment-specific subclasses or parameterized dispatch.
- Ensure observability:
  - After execution, environment state reliably reflects the real environment.
  - Feedback functions accurately indicate status.

---

## 5. Security, Robustness, and Testing

- Incorporate safeguards against invalid actions.
- Validate environment API calls with dummy tests.
- Use unit tests for parsing, effect application, and state updates.
- Confirm that sampling does not generate impossible action sequences.

---

## 6. Summary

The `Environment` class in `environment.py` must:

- Initialize from environment signatures and system identifiers.
- Provide an `execute_plan()` method that simulates (or controls) environment to execute an entire plan, returning success/failure.
- Support `check_feasibility()` for partial action sequence verification.
- Allow `sample_action_sequences()` to generate random, executable action sequences respecting environment constraints, leveraging API calls.
- Support `get_state()` for providing accurate environment descriptions.
- Handle errors gracefully, maintaining consistent internal state updates, to enable the higher-level PDDL refinement system to rely on this interface.

Following this detailed logical structure ensures faithful reproduction of the environment interaction component as described in the paper, with robustness and clarity suitable for large-scale experimental automation.

## llm_manager.py

{
  "llm_manager.py": "The llm_manager.py module is responsible for managing all interactions with the Large Language Model (LLM) API, such as GPT-4, to generate and refine PDDL files based on NL descriptions and environment feedback. Its core functions are: \n\n1. generate_initial_problem(nl_description, object_list, api_signatures):\n    - Purpose: To produce a preliminary, fully specified PDDL problem file given the environment's NL description, object list, and API signatures.\n    - Inputs:\n        - nl_description: String containing the natural language description of the environment and task.\n        - object_list: List or structured data of environment objects (objects types and counts), necessary to fill the PDDL object section.\n        - api_signatures: Dictionary detailing available environment actions (name, parameters, signatures).\n    - Process:\n        - Construct a prompt for the LLM using a predefined template for initial problem generation.\n        - Insert the passed NL description, object list, and API signatures into the prompt.\n        - Call the LLM API with this prompt, with parameters such as temperature=0 (greedy) to ensure deterministic, high-fidelity output.\n        - Parse the response: Expect a complete PDDL problem file text.\n        - Return the PDDL string.\n    - Output: String containing the generated problem PDDL.\n\n2. generate_initial_domain(nl_description, api_signatures):\n    - Purpose: To produce a preliminary PDDL domain file from the environment’s NL description and API signatures.\n    - Inputs:\n        - nl_description: Same as above.\n        - api_signatures: Environment action API signatures.\n    - Process:\n        - Use a prompt template tailored for domain creation, instructing the LLM to produce syntax-correct, action-centered PDDL domain files.\n        - Insert the NL description and signature info into the prompt.\n        - Call the API with temperature=0.\n        - Parse the response: Expect a complete domain PDDL string.\n        - Return the domain PDDL string.\n    - Output: String of the domain PDDL.\n\n3. refine_prompt(current_domain, environment_feedback):\n    - Purpose: To generate an improved version of the current domain PDDL by incorporating environment feedback, EW scores, or detected issues.\n    - Inputs:\n        - current_domain: The current PDDL domain string.\n        - environment_feedback: Quantitative feedback such as EW scores, plan success/failure status, or qualitative signals from environment interactions.\n    - Process:\n        - Construct a prompt for refinement using a dedicated template.\n        - Insert current domain PDDL into the prompt.\n        - Include environment feedback signals, EW scores, or problematic action sequences.\n        - Call the LLM API with a temperature > 0 (e.g., 0.7) to allow creative and nuanced corrections.\n        - Parse the returned text: Expect a refined, syntax-correct PDDL domain.\n        - Possibly repeat or validate the refinement, ensuring syntax correctness.\n        - Return the refined domain PDDL string.\n\nGeneral Requirements and Considerations:\n- Prompt Templates:\n    - These templates are stored as multi-line strings (e.g., in the config.yaml or internal constants).\n    - They should be designed to clearly instruct the LLM to generate syntactically valid, referenceable PDDL files, possibly with comments or section labels.\n- API Calls:\n    - Use openai.ChatCompletion.create() or a similar function.\n    - Set parameters: 'model' to 'gpt-4', temperature as specified, max_tokens sufficiently large to accommodate full files.\n    - Handle exceptions and API errors gracefully. Implement retries if needed.\n- Response Parsing:\n    - Extract PDDL code from the completion response, ensuring it is well-formed.\n    - Use delimiters or code fences if applicable.\n- Reproducibility:\n    - Responses should be deterministic for initial problem/domain generation (temperature=0). For refinement, allow variability.\n    - Log prompts and responses for debugging.\n- Data Handling:\n    - The functions should accept inputs in native Python data types and convert to appropriate prompt strings.\n    - For object lists and api_signatures, format as readable sections in prompts.\n- Interface: \n    - Functions should be accessible as methods of a class instance, perhaps initialized with API keys, prompt templates, and configuration parameters.\n\nIn conclusion, the logic ensures robust, deterministic initial generation, and flexible, feedback-driven refinement, maintaining broad compatibility with diverse domain/problem NL descriptions, object lists, and environment APIs, as mandated by the overall framework."

## metrics.py

{
  "metrics.py": "The metrics.py module is central to the quantitative evaluation and guiding of the PDDL refinement process. Its primary responsibilities are: (1) to compute the Exploration Walk (EW) similarity scores between candidate domain models and the ground-truth environment to assess domain alignment; (2) to evaluate the success of generated plans within the environment (plan validation); and (3) to quantify the similarity or divergence between two domain models, facilitating informed refinement decisions.\n\nThe module implements three key functions:\n\n1. calculate_ew_score(domain1, domain2, sequences1, sequences2, environment, max_length):\n   - Purpose: Compute a symmetric EW score between two domain models (e.g., the current candidate vs. the ground-truth or another candidate).\n   - Inputs:\n     - domain1, domain2: String representations of domain PDDL files.\n     - sequences1, sequences2: Pre-sampled lists of action sequences (each a list of action strings) from domain1 and domain2 respectively.\n     - environment: An environment instance for interaction and sampling.\n     - max_length: The maximum length of action sequences sampled for evaluation.\n   - Procedure:\n     - For each domain, sample multiple action sequences using environment.sample_action_sequences() with sequence length max_length.\n     - For each sampled sequence, verify executability in the environment given the domain model with check_feasibility().\n     - Compute the fraction of sequences from domain1 that are executable in domain2's model, and vice versa.\n     - Return the harmonic mean of these fractions, as per the symmetric EW metric definition.\n   - Output: A float EW score in [0, 1], reflecting domain similarity.\n\n2. evaluate_plan_success(plan, environment):\n   - Purpose: Check whether a particular plan is valid and successfully achieves the goal in the real environment.\n   - Inputs:\n     - plan: String representation of the plan, typically in a format compatible with environment.execute_plan().\n     - environment: Environment instance capable of executing the plan.\n   - Procedure:\n     - Call environment.execute_plan(domain, problem, plan).  \n     - The environment returns success=True if the plan executes without errors and achieves the goal; else False.\n   - Output: Boolean indicating plan success.\n\n3. compute_similarity(domain1, domain2):\n   - Purpose: A high-level function to quantify similarity between two domains purely based on their models, without sampling sequences.\n   - Inputs:\n     - domain1, domain2: String representations of domain PDDL files.\n   - Procedure:\n     - Use structural or predicate-based heuristics, such as counting predicate overlaps, action overlaps, or predicate parameter similarities (if structured parsing is enabled).  \n     - Alternatively, invoke the sample Sequence-based EW calculation using 'sample_action_sequences()' with environment feedback, as in calculate_ew_score(), but for a fixed set of sequences.\n     - The function supports the main EW calculation but may leverage domain parsing and comparison heuristics for efficiency.\n   - Output: A float score indicating similarity, scaled in [0, 1].\n\nAdditional considerations: \n- All functions must rely on the environment.py interfaces for sampling sequences, checking executability, and executing plans. \n- For the EW score, the sampling process must be aligned with the environment's API, ignoring the internal PDDL formalism. \n- The heuristic for compute_similarity() should be consistent with the EW definitions, facilitating comparison and guiding refinement.\n- The functions should handle exceptions gracefully, including invalid PDDL syntax, infeasible sequences, or environment execution errors, returning consistent scores (e.g., 0 for no similarity, False for plan failure) and logging errors.\n\nThis structure ensures the metrics.py module accurately measures domain-environment alignment, plan validity, and model similarity, aligning with the experimental framework and improving the iterative refinement guided by EW scores." 
}

## pddl_generator.py

# Logic Analysis for pddl_generator.py

This module is responsible for generating and refining PDDL domain and problem files from natural language descriptions, object lists, API signatures, and feedback signals, aligning with the methodology outlined in the paper. It implements three core functions:

- `generate_initial_problem()`: Creates the initial problem PDDL file.
- `generate_initial_domain()`: Creates the initial domain PDDL file.
- `refine_domain()`: Produces a refined domain based on prior domain, environment feedback, and evaluation metrics.

The design relies heavily on prompt-based input-output with large language models (LLMs) and on the use of feedback signals like EW scores to guide refinements. Below is a detailed logical breakdown to guide implementation.

---

## 1. `generate_initial_problem(nl_desc, object_list, api_signatures)`

### Purpose
Produce an initial, syntactically correct complete PDDL problem file based on NL description, object list, and API signatures, to set a baseline problem which can be evaluated and refined later.

### Inputs
- `nl_desc`: String containing natural language description of the environment and task.
- `object_list`: List of objects with types and identifiers, e.g., `['robot1', 'robot2', 'ball1', 'room1', ...]`.
- `api_signatures`: Dictionary describing action signatures, e.g.,
  ```python
  {
    'move': [('? r - robot', '? from - room', '? to - room')],
    'pick': [('? r - robot', '? o - obj', '? room - room', '? g - gripper')],
    'drop': [('? r - robot', '? o - obj', '? room - room', '? g - gripper')]
  }
  ```
### Output
- String representing a complete, syntactically valid PDDL problem file, wrapped in markdown code block with `pddl` syntax.

### Logic
- Use a prompt template that incorporates:
  - `nl_desc`: to provide contextual NL description.
  - `object_list`: to list objects with explicit types.
  - `api_signatures`: to specify available actions and their parameters.
- The prompt instructs the LLM to `generate a complete PDDL problem file`, emphasizing syntax, object constants, initial state, and goal specification.
- Post-processing:
  - Parse the LLM output to validate syntax correctness (may involve syntax validation using `pddlpy`).
  - Extract the `(:init ...)` and `(:goal ...)` sections.
  - Ensure object list is consistent with environment simulation (if discrepancies, akin to the placeholder "trivial mismatch handling").
- Handle variations where the LLM may omit or produce incomplete elements by re-prompting or correction loops.

---

## 2. `generate_initial_domain(nl_desc, api_signatures)`

### Purpose
Create an initial domain PDDL file based on NL description and API signatures, establishing types, predicates, actions, and example initial conditions.

### Inputs
- `nl_desc`: String NL description of environment.
- `api_signatures`: Dictionary with action parameter signatures.

### Output
- String that contains a complete syntactic domain PDDL file, wrapped in markdown code block with `pddl` syntax.

### Logic
- Use a prompt template that:
  - Summarizes the environment context, types, predicates, and actions.
  - Incorporates `api_signatures` to explicitly detail action parameters.
  - Explicitly states the action effects, preconditions, and parameter semantics.
- When generating:
  - Initialize with a template that includes:
    - Requirements (`:strips`, `:typing`, optionally others).
    - Types (derived from objects and typical environment knowledge).
    - Predicates (e.g., `at`, `free`, `carry`, `on`), possibly augmented with environment-specific predicates.
    - Actions:
      - Move actions, including `move`, `move-up`, `move-down` with preconditions/effects.
      - Pick/Drop actions with parameters and preconditions/effects.
  - Call LLM with explicit instructions to generate syntactically correct PDDL domain.
  - Post-process and validate output syntax, possibly correcting small syntax errors by re-prompt or minor patching.
  - Ensure the action signatures conform exactly to `api_signatures`.

### 3. `refine_domain(current_domain, environment, ew_feedback)`

### Purpose
Refine a given domain PDDL based on environment feedback signals and EW scores, helping the model correct previous inaccuracies or improve action modeling.

### Inputs
- `current_domain`: String of current domain PDDL.
- `environment`: Environment object to interact with environment functions.
- `ew_feedback`: Numeric EW score (e.g., between 0 and 1), indicating domain similarity; used to guide whether refinement is needed.

### Output
- String of the refined domain PDDL, syntactically valid, and improved based on feedback.

### Logic
- Use a prompt template:
  - Include current domain PDDL as context.
  - Summarize environment feedback signals:
    - Errors identified (e.g., syntax errors, semantic inconsistencies, invalid predicates).
    - Failures in plan execution or EW discrepancies.
  - Instruct the LLM to:
    - Correct identified issues.
    - Improve action preconditions/effects based on environment feedback.
    - Preserve the action signature (boundary conditions).
- Incorporate a `refinement prompt template`:
  - Example hints: "Check for missing predicates," "Ensure move actions include accurate preconditions for environment movement."
- Use the environment (via `environment.py`) to:
  - Sample action sequences or interact with environment for specific feedback signals (success/error logs).
  - Incorporate these signals into the prompt to guide the correction.
- Post-process the LLM output:
  - Syntax validation.
  - Possibly re-prompt with specific corrections if errors are detected.
- Limit iteration to avoid divergence; maximum cycles guided by configuration (e.g., 20 iterations).

---

## 4. Cross-Module Interactions & Constraints
- All functions must produce valid, deployable PDDL files that can be consumed by classical planners.
- Consistency between domain and problem files is crucial, especially regarding action signatures and object parameters.
- Feedback signals (`ew_feedback`, environment errors) must be translated into explicit instructions for the LLM (e.g., "correct move action precondition to include ...").
- Prompts must be structured with clarity, emphasizing syntactic correctness, action semantics, and environment alignment.
- Consider possible failure modes:
  - Syntax errors in generated PDDL.
  - Missing predicates or incorrect effects.
  - Misaligned object references.
- Re-try or re-prompt in case of failure, possibly adjusting prompts based on error signals.

---

## 5. Summary of Implementation Logic
- Leverage templated prompts for initial generation.
- Validate syntax and content after each LLM response.
- Use environment interaction (plan execution, EW sampling) to assess realism.
- Apply iterative correction and refinement, guided by EW scores and environment feedback.
- Maintain traceability and logs for reproducibility, capturing:
  - Generated PDDL files.
  - Feedback/error signals.
  - EW scores.
  - Plan execution outcomes.
- Ensure the entire process can be rerun for different environments with minimal adjustments, guided by configuration parameters.

---

This comprehensive logic guides systematic implementation of `pddl_generator.py` consistent with the described methodology, emphasizing robustness, validation, and iterative improvement aligned with the paper's approach.

## refinement.py

# Logic Analysis for refinement.py

## **Purpose and Role of refinement.py**

The `refinement.py` module is responsible for orchestrating the iterative refinement of the PDDL domain and problem files. It manages the reasoning loop that uses environment feedback and the Exploration Walk (EW) metric to guide the generation of more accurate PDDL models. Specifically, it:

- Interfaces with the `metrics.py` to compute domain similarity scores and plan success.
- Calls upon `llm_manager.py` to generate, refine, or improve domain models based on feedback.
- Evaluates generated models by testing their executability against environment feedback signals.
- Uses a controlled loop to iteratively improve the models until success criteria (EW threshold, plan success, iteration limits) are met.

---

## **Inputs and Inputs Handling**

- **Current Domain Model String**: The current best estimate of the domain PDDL (`current_domain`).
- **Environment Feedback Signals**: Quantitative measure, e.g., EW score (`ew_feedback`), or possibly detailed feedback from environment interaction.
- **Environment Interface / Environment Object**: To run simulated or real environment interactions, including executing plans and sampling action sequences.
- **Metrics Module**: For calculating similarity scores and plan success; depends on environment API for sampling sequences.
- **LLM Manager**: For generating or refining domain models, based on prompts and feedback.
- **Configuration Parameters**: Max refinement iterations, EW threshold, plan success rate threshold.

---

## **Main Logic and Workflow**

### Initialization:

- Set counter for the number of refinement iterations (`iteration = 0`).
- Initialize variables:
  - `best_domain`: Save the current best PDDL string.
  - `best_ew_score`: Track the EW similarity score of the best model.
  - `best_plan_success`: Track success in environment testing.
- Optionally, initialize logging structures for iteration data, success logs.

### Refinement Loop:

While `iteration < max_refinement_iterations`:

1. **Generate Candidate Refinement**:
   - Call `llm_manager.refine_prompt()` (or a similar function) to produce a refined domain PDDL.
   - Input prompt: current domain model (`best_domain`), environment feedback signals or previous EW scores, and possibly a description or prompt instructing to improve fidelity.
   - Store generated domain model string (`candidate_domain`).

2. **Validation and Consistency Checks**:
   - Check syntax validity of `candidate_domain` (parsing, PDDL grammar).
   - Optional: validate internal consistency (matching API signatures, predicate familiarity).

3. **Compute Similarity via EW Metric**:
   - Use `metrics.calculate_ew_score()`:
     - Input: current domain (`best_domain`), candidate domain.
     - Environment API functions for sampling random action sequences.
     - Max walk length specified in config (`max_sampling_length`).
   - Retrieve `ew_score`.

4. **Plan Evaluation in Environment**:
   - With the environment object, run the classical planner on the current best problem model + the candidate domain.
   - Obtain a plan string (`plan`).
   - Validate the plan via `environment.execute_plan()` or `metrics.evaluate_plan_success()`:
     - Execute in environment.
     - Check for success: plan leads to goal, execution completes without errors.

5. **Update Best Model if Improved**:
   - Compare `ew_score` and plan success:
     - If `ew_score >= ew_threshold` AND `plan_success`:
       - Accept candidate as new best domain.
       - Update `best_domain`, `best_ew_score`, `best_plan_success`.
       - Log success; optionally break if thresholds are met.
   - Else:
     - Keep the current best; continue refinement.

6. **Termination Conditions**:
   - Break if:
     - Plan successfully executes and reaches goal.
     - EW score exceeds threshold AND plan success.

7. **Increment Iteration Counter**:
   - `iteration += 1`

---

## **Handling of Environment Feedback and Metrics**

- The environment feedback during candidate generation is used to guide LLM refinement prompts.
- `metrics.py` functions include:
  - `calculate_ew_score(domain1, domain2, environment)`:
    - Samples sequences from environment, tests executability across domains.
    - Computes similarity score based on sampling and environment responses.
  - `evaluate_plan_success(plan, environment)`:
    - Executes the plan in the environment.
    - Checks goal attainment.
  - `compute_similarity()` can be used during initial assessment or for internal metrics tracking.

---

## **Error and Exception Handling**

- Syntax errors or invalid PDDL from LLM should be caught and reported.
- Environment execution errors (plan infeasible) will inform whether to discard candidate models or request further refinement.
- Environment failures should be logged, with potential fallback prompts or reinitialization.
- If the environment reports no improvements over several iterations, the loop may terminate early.
- Implement retry logic at the level of prompt generation, plan execution, or sampling if needed.

---

## **Output and Return Values**

- The refined, best domain PDDL string (`best_domain`) at the end of all iterations.
- Associated metrics (EW score, plan success, iteration count).
- Log of iterative results for analysis and reproducibility.

---

## **Summary of Key Functions to Implement and Use**

- `refine(domain_str, environment, ew_score) -> str`:
  - Calls LLM with prompt, environment feedback in context, produces refined domain.
- `calculate_ew_score(domain1, domain2, environment) -> float`:
  - Uses environment sampling and the EW metrics detailed in the paper.
- `evaluate_plan_success(plan, environment) -> bool`:
  - Executes plan, verifies goal attainment.
- Control Loop managing iterations, thresholds, and acceptance criteria.

---

## **Conclusion**

The `refinement.py` module encapsulates the core iterative refinement logic, heavily dependent on:

- Environment interaction and sampling for feedback signals.
- LLM prompt calls for generated refinement.
- Metrics calculation (EW similarity) to guide direction.
- Classical planner invocation for planning validation.
- Looping with limits and success thresholds.

This structured logic ensures systematic, transparent, and reproducible domain/model refinement aligned with the methodology described in the paper, facilitating high-quality PDDL model generation guided exclusively by environment feedback and LLM reasoning.

## utils.py

{
  "utility_functions": [
    {
      "function_name": "log_results",
      "description": "Logs the success status, EW score, number of refinement iterations, and resource usage details for each environment run to facilitate reproducibility and post-hoc analysis.",
      "inputs": [
        "env_id (str): Unique identifier for the environment/test case.",
        "success (bool): Whether the plan successfully achieved the goal in the environment.",
        "ew_score (float): The EW similarity score at the time of evaluation.",
        "iterations (int): Number of refinement iterations performed for this environment.",
        "resource_usage (dict): Dictionary containing relevant resource metrics, such as token counts and time."
      ],
      "outputs": ["None"],
      "notes": "This function should handle logging levels, possibly output to console and save to a log file, respecting 'log_dir' and 'log_level' from configuration. It maintains traceability for experiments."
    },
    {
      "function_name": "generate_prompt_initial_problem",
      "description": "Constructs a prompt string for the LLM to generate an initial problem PDDL based on NL description, object list, and API signatures, using a predefined template.",
      "inputs": [
        "nl_description (str): Natural language description of the environment's task.",
        "object_list (list): List of object identifiers and types involved in the environment.",
        "api_signatures (dict): Signatures of environment API actions, including parameters and expected responses."
      ],
      "outputs": ["prompt (str)"],
      "notes": "Uses the 'initial_prompt_template' from config.prompting; templates should be filled with the inputs. Ensures prompts are formatted consistently for repeatability."
    },
    {
      "function_name": "generate_prompt_initial_domain",
      "description": "Constructs a prompt string for the LLM to generate a domain PDDL given NL descriptions and API signatures, facilitating initial domain hypotheses.",
      "inputs": [
        "nl_description (str): NL domain description.",
        "api_signatures (dict): Signatures of environment API actions."
      ],
      "outputs": ["prompt (str)"],
      "notes": "Uses the 'domain_prompt_template' from config.prompting. Critical for generating a starting domain model for the refinement loop."
    },
    {
      "function_name": "refine_prompt",
      "description": "Constructs a refinement prompt to guide the LLM towards improving current domain model based on environment feedback and EW score.",
      "inputs": [
        "current_domain (str): The current PDDL domain model string.",
        "environment_feedback (str or float): Feedback or EW-related scalar indicating current model shortcomings.",
        "additional_context (dict, optional): Any supplementary info such as previous prompts or EW scores."
      ],
      "outputs": ["prompt (str)"],
      "notes": "Uses the 'refinement_prompt_template' from config.prompting. The output prompt incorporates previous domain details and the feedback metric to steer LLM improvements."
    },
    {
      "function_name": "execute_plan_in_environment",
      "description": "Submits a plan, along with domain and problem files, to the environment API for execution and success verification.",
      "inputs": [
        "domain_file (str): PDDL domain file path or content string.",
        "problem_file (str): PDDL problem file path or content string.",
        "plan (str): Action sequence in PDDL plan format."
      ],
      "outputs": ["success (bool)"],
      "notes": "The success indicates whether the plan achieves the environment's goal post-execution, as per environment API (see '/interact_environment'). In simulation, can be implemented by invoking the environment's execute or simulate method, capturing errors or success status."
    },
    {
      "function_name": "is_plan_executable_in_environment",
      "description": "Checks whether a given action sequence is feasible in the environment without full execution (e.g., using environment API or simulation).",
      "inputs": [
        "domain (str): Current PDDL domain string.",
        "problem (str): Current PDDL problem string.",
        "action_sequence (list): List of actions (strings) to verify."
      ],
      "outputs": ["feasible (bool)"],
      "notes": "Should query environment API or interpreter to verify feasibility; does not necessarily execute but tests action steps for feasibility."
    },
    {
      "function_name": "sample_action_sequences",
      "description": "Samples a set of diverse action sequences from the environment, constrained by the maximum sampling length T_max and number of samples.`,
      "inputs": [
        "domain (str): PDDL domain content or object representing model.",
        "problem (str): PDDL problem content or object.",
        "sequence_length (int): Length T_max of each sequence.",
        "num_samples (int): Number of sequences to sample."
      ],
      "outputs": ["List[List[str]]"],
      "notes": "Generates random but valid action sequences based on environment's API, perhaps guided by current domain model or sampling heuristics. Used for computing EW scores."
    },
    {
      "function_name": "compute_ew_score",
      "description": "Calculates the Exploration Walk (EW) similarity metric between two domain models over a set of sampled sequences, based on environment feedback.",
      "inputs": [
        "domain1 (str): PDDL domain string for environment D.",
        "domain2 (str): PDDL domain string for candidate or refined environment D_hat.",
        "sequences1 (list): List of sampled sequences from domain1.",
        "sequences2 (list): List of sampled sequences from domain2.",
        "environment (Environment): Environment object encapsulating API to verify sequence executability.",
        "max_length (int): The maximum length T_max for the EW walks."
      ],
      "outputs": ["ew_score (float)"],
      "notes": "Implements the described EW metric involving sampling sequences, executing in environments, measuring executability probabilities, and combining via harmonic mean. Ensures metrics are symmetric and robust to trivial similarities."
    },
    {
      "function_name": "evaluate_plan_success",
      "description": "Determines if a plan is successful in achieving the environment goal, by checking against environment state or task completion signals.",
      "inputs": [
        "plan (str): Action plan in PDDL format.",
        "environment (Environment): Environment object to verify goal achievement."
      ],
      "outputs": ["success (bool)"],
      "notes": "May involve executing in environment or analyzing environment state/assertions indicating goal achievement."
    },
    {
      "function_name": "parse_pddl_string",
      "description": "Utility to parse a PDDL string and verify syntax correctness, predicate definitions, and action signatures.",
      "inputs": ["pddl_str (str)"],
      "outputs": ["parsed_structure (dict) or bool"],
      "notes": "Support checking validity of generated PDDL files; potentially validate syntax before usage."
    },
    {
      "function_name": "load_resource_metrics",
      "description": "Extracts and formats resource usage stats (tokens, time, memory) from logs or API responses for reproducibility.",
      "inputs": ["raw_output (dict or str)"],
      "outputs": ["resource_metrics (dict)"],
      "notes": "Captures token counts, runtime duration, and memory to support computing resource analysis."
    },
    {
      "function_name": "save_to_logfile",
      "description": "Saves structured logs—could be JSON or structured CSV—containing environment id, success flag, EW score, iterations, resource metrics.",
      "inputs": [
        "log_path (str): Path to log file directory.",
        "log_data (dict): Data to persist, including environment id, metrics, resource usage."
      ],
      "outputs": ["None"],
      "notes": "Supports detailed record-keeping for experiment reproducibility and analysis."
    }
  ],
  "notes": [
    "All functions should use consistent formatting for prompts, PDDL actions, and resource metrics, as per the templates provided.",
    "Functions wrapped with resource management and exception handling to ensure robust execution.",
    "Logging functions should adhere to the 'log_level' setting, and ensure logs are timestamped and uniquely tagged by environment id.",
    "All functions should be designed with modularity, enabling easy testing and extension for new environments or feedback schemas.",
    "When implementing, ensure that the environment API functions (execute_plan, check_feasibility) are consistent with the formal contract described in the system design and API spec.",
    "Prioritize efficiency: for example, sample_action_sequences should minimize environment calls while providing representative samples for EW calculation.",
    "In the final code, include docstrings, type annotations, and comments explaining the logic steps for clarity and ease of reproducibility."
  ]
}

