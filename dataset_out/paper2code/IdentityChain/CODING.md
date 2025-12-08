# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## evaluation.py

```python
## evaluation.py
import math
from typing import List, Dict, Any, Tuple


class EvaluationMetrics:
    """
    Implements self-consistency evaluation logic based on the Test Output Match (TOM) score,
    as well as aggregation methods for overall scores across multiple inputs/problems.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize evaluation parameters from configuration.
        Args:
            config (dict): Configuration dictionary (from config.yaml).
        """
        # Configurable parameters with default values
        self.test_case_count: int = config.get('evaluation', {}).get('test_case_count', 16)
        self.scoring_method: str = config.get('evaluation', {}).get('scoring_method', 'TOM')
        # For debugging/diagnostics
        self.per_input_results: List[Dict[str, Any]] = []

    def compute_tom(self,
                    test_outputs_pl_i: List[Union[str, Any]],
                    test_outputs_pl_next: List[Union[str, Any]],
                    test_cases: List[Dict[str, Any]]) -> float:
        """
        Compute the Test Output Match (TOM) score between outputs of successive programs.
        Args:
            test_outputs_pl_i: outputs from current program (`pl_i`) for each test case.
            test_outputs_pl_next: outputs from next program (`pl_{i+1}`).
            test_cases: list of test case dicts, each with 'inputs' and optional 'expected_output'.
        Returns:
            float: TOM score in [0,1], representing percentage of matching test outputs.
        """
        total_tests = len(test_outputs_pl_i)
        if total_tests == 0:
            return 0.0

        match_count = 0
        detailed_results = []  # for diagnostics

        for idx in range(total_tests):
            out_i = test_outputs_pl_i[idx]
            out_next = test_outputs_pl_next[idx]

            # Determine if both outputs are errors or normal outputs
            is_error_i = self._detect_error(out_i)
            is_error_next = self._detect_error(out_next)

            # Convert outputs to strings for comparison
            str_i = str(out_i).strip()
            str_next = str(out_next).strip()

            # If both are errors, compare error messages exactly
            if is_error_i and is_error_next:
                match = (str_i == str_next)
            # If both are normal outputs, compare for exact equality
            elif not is_error_i and not is_error_next:
                match = (str_i == str_next)
            else:
                # One error, one normal output -> mismatch
                match = False

            match_count += int(match)

            # Save detailed per-test-case info
            detailed_results.append({
                'test_idx': idx,
                'output_current': out_i,
                'output_next': out_next,
                'both_errors': is_error_i and is_error_next,
                'match': match
            })

        # Save per-input detailed results
        self.per_input_results.append({
            'match_score': match_count / total_tests,
            'detailed': detailed_results
        })

        return match_count / total_tests

    def _detect_error(self, output: Union[str, Any]) -> bool:
        """
        Heuristically determines if the output indicates an error (syntax, runtime).
        Args:
            output: output string or object
        Returns:
            bool: True if output appears to be an error message, False otherwise.
        """
        if isinstance(output, str):
            lowered = output.lower()
            error_indicators = ['syntaxerror', 'indentationerror', 'error', 'exception', 'traceback', 'timeout']
            return any(ind in lowered for ind in error_indicators)
        # For non-string types, assume normal output
        return False

    def aggregate_scores(self, per_input_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate per-input results into overall self-consistency scores.
        Args:
            per_input_results (list): list of dicts with per-input 'match_score' and 'detailed'.
        Returns:
            dict: { 'SC_n': float, 'SSC_n': float }
        """
        if not per_input_results:
            return {'SC_n': 0.0, 'SSC_n': 0.0}

        total_inputs = len(per_input_results)
        sum_sc = 0.0
        sum_ssc = 0.0

        for res in per_input_results:
            match_score = res.get('match_score', 0.0)
            # For 'self-consistency' (SC_n), check if all tests matched
            sc_score = 1.0 if match_score == 1.0 else 0.0
            # For 'strong self-consistency' (SSC_n),
            # check additional criteria if desired. Here, we assume same as SC, but can extend.
            ssc_score = sc_score

            sum_sc += sc_score
            sum_ssc += ssc_score

        overall_sc = sum_sc / total_inputs
        overall_ssc = sum_ssc / total_inputs

        return {
            'SC_n': overall_sc,
            'SSC_n': overall_ssc
        }

    def reset(self):
        """
        Reset diagnostics/results storage for fresh evaluation if needed.
        """
        self.per_input_results = []
```

## main.py

```python
## main.py
import os
import yaml
import logging
from typing import List, Dict, Any
import random

# Import the modular components
from utils import replace_function_names
from model_api import ModelAPI
from prompt_engineering import PromptManager
from test_executor import TestExecutor
from evaluation import EvaluationMetrics

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """ Load the configuration from YAML file. """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def initialize_models(models_config: List[Dict[str, Any]], eval_config: Dict[str, Any]) -> Dict[str, Any]:
    """ Instantiate model objects based on config. """
    models = {}
    for model_cfg in models_config:
        name = model_cfg['name']
        m_type = model_cfg.get('type', 'local')  # default to local
        if m_type == 'api':
            api_key = model_cfg.get('api_key', '')
            model_obj = ModelAPI(
                model_name=name,
                api_key=api_key,
                temperature=0.0,  # deterministic
                max_new_tokens=eval_config.get('max_tokens', 512)
            )
        elif m_type == 'local':
            model_path = model_cfg.get('model_path', '')
            model_obj = ModelAPI(
                model_name=model_path,
                temperature=0.0,
                max_new_tokens=eval_config.get('max_tokens', 512),
                model_type='local'
            )
        else:
            raise ValueError(f"Unknown model type {m_type} for model {name}")
        models[name] = model_obj
    return models

def generate_test_cases(problem_metadata: Dict[str, Any], count: int = 16) -> List[Dict[str, Any]]:
    """ Generate synthetic or extract test cases. Placeholder implementation real dataset integration needed. """
    test_cases = []
    for _ in range(count):
        # Placeholder: generate dummy test cases; replace with dataset-specific extraction if available
        import random
        test_cases.append({'inputs': [random.randint(1, 100) for _ in range(1)]})
    return test_cases

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Load config
    config = load_config()
    # Extract configs
    training_cfg = config.get('training', {})
    eval_cfg = config.get('evaluation', {})
    models_cfg = config.get('models', {})
    prompt_cfg = config.get('prompt_templates', {})

    # Initialize models
    models = initialize_models(models_cfg, eval_cfg)

    # Initialize prompt manager
    prompt_manager = PromptManager(config)

    # Initialize test executor and evaluation metrics
    test_executor = TestExecutor(max_time=2.0)
    eval_metrics = EvaluationMetrics(config)

    # Placeholder: Load dataset / problems
    # For demonstration, we create a list with a single dummy problem
    problems = [{
        'id': 0,
        'task_description': 'Find the maximum even number between 1 and 100',
        'metadata': {'input_params': 2},  # For test case generation
        # 'initial_nl': could be dataset provided; here, generate from model
    }]

    # Results container
    all_results = []

    for prob in problems:
        problem_id = prob['id']
        task_desc = prob['task_description']
        problem_metadata = prob.get('metadata', {})

        # 1. Generate initial NL specification
        # For simplicity, assume NL can be taken as task description
        nl_0 = task_desc

        # 2. Generate initial program from NL
        pl_0 = None
        for model_name, model_obj in models.items():
            # Create prompt for NL2PL
            prompt_nl2pl = prompt_manager.create_nl2pl_prompt(task_desc, test_cases=[])
            pl_candidate = model_obj.call_model(prompt_nl2pl, max_tokens=training_cfg.get('max_tokens', 512))
            # Optional: Post-process pl_candidate if needed
            pl_0 = pl_candidate
            break  # For simplicity, pick the first model; or extend for multiple models

        # 3. Replace function names with 'func' for self-consistency stability
        pl_0 = prompt_manager.replace_function_names(pl_0)

        # 4. Generate test cases for the program
        test_cases = generate_test_cases(prob, eval_cfg.get('test_case_count', 16))
        # For real datasets, replace above with proper extraction

        # 5. Initialize chain iterative variables
        chain_length = training_cfg.get('chain_length', 5)
        chain_n = chain_length
        early_stop = True
        # Store per-problem results
        chain_generated_pls = [pl_0]
        chain_generated_nls = []
        chain_self_consistency_scores = []
        chain_strong_scores = []

        # 6. Iterative chain execution
        nl_prev = nl_0
        pl_prev = pl_0
        # For computing, keep track of semantics via test outputs
        per_input_test_outputs_prev = None

        for i in range(chain_n):
            # Generate NL_{i+1} from PL_{i}
            prompt_p2n = prompt_manager.create_pl2nl_prompt(pl_prev)
            nl_next = models[next(iter(models))].call_model(prompt_p2n, max_tokens=training_cfg.get('max_tokens', 512))
            nl_next = nl_next.strip()
            # Replace function names just in case
            nl_next = prompt_manager.replace_function_names(nl_next)

            # Generate PL_{i+1} from NL_{i+1}
            prompt_n2p = prompt_manager.create_nl2pl_prompt(nl_next)
            pl_next = models[next(iter(models))].call_model(prompt_n2p, max_tokens=training_cfg.get('max_tokens', 512))
            pl_next = pl_next.strip()
            pl_next = prompt_manager.replace_function_names(pl_next)

            # Check for early stopping: exact match
            if early_stop:
                if pl_next == pl_prev or nl_next == nl_prev:
                    print(f"Early stopping at step {i} for problem {problem_id}")
                    break

            # Store generated sequences
            chain_generated_nls.append(nl_next)
            chain_generated_pls.append(pl_next)

            # 7. Evaluate current pair with test cases
            # Execute pl_{i} / pl_{i+1} against test cases
            test_results_prev = test_executor.execute_test(pl_prev, test_cases)
            test_results_curr = test_executor.execute_test(pl_next, test_cases)

            # 8. Compute TOM score for self-consistency between pl_{i} and pl_{i+1}
            # For simplicity, average TOM over all test cases for entire chain
            # Alternatively, compute for each pair as a factor in overall score
            # Here, we just store per-input per-pair scores
            # Let's compute per input if all test outputs match
            # We consider the TOM score as average over test cases
            # For that, we need test outputs; simulate by filtering actual outputs
            # For now, just store test outputs for the last pair
            # This is a simplified approach: more detailed diagnostics can be added
            # For proper per-input TOM, maintain a list of outputs per test case
            # using previous execute_test calls
            pass  # To keep code clean, we'll handle scoring outside loop

            # Update previous nl and pl for next iteration
            nl_prev, pl_prev = nl_next, pl_next

        # 9. Final evaluation: For dataset input, get last test results
        # Execute initial PL and back-translate to NL to assess PL→NL accuracy
        # For simplicity, use initial pl_0 and last nl generated
        # Generate natural language from initial program (or final program)
        # Using the last nl generated from chain, or the initial one
        # For now, use the last nl for back-translation
        prompt_n2p_final = prompt_manager.create_nl2pl_prompt(nl_next)  # nl_next from last step
        pl_back = models[next(iter(models))].call_model(prompt_n2p_final, max_tokens=training_cfg.get('max_tokens', 512))
        pl_back = pl_back.strip()
        pl_back = prompt_manager.replace_function_names(pl_back)

        # Run back translation to get NL from the program
        prompt_pl2nl_final = prompt_manager.create_pl2nl_prompt(pl_back)
        nl_from_pl = models[next(iter(models))].call_model(prompt_pl2nl_final, max_tokens=training_cfg.get('max_tokens', 512))
        nl_from_pl = nl_from_pl.strip()
        nl_from_pl = prompt_manager.replace_function_names(nl_from_pl)

        # 10. Execute initial program (pl_0) on test cases to compute initial Pass@1
        initial_test_results = test_executor.execute_test(pl_0, test_cases)
        # Determine Pass@1
        pass_at_1 = 1.0 if all(res['passed'] for res in initial_test_results) else 0.0

        # 11. Compute TOM scores between pl_{i} and pl_{i+1}
        # For simplification, after the chain, compare pl_0 and pl_{last},
        # similarly for their back translation
        # For the main metric, let's compare pl_0 and pl_{last} outputs on test cases
        test_outputs_pl0 = test_executor.execute_test(pl_0, test_cases)
        test_outputs_pl_last = test_executor.execute_test(pl_prev, test_cases)

        # Calculate TOM score
        tom_score = eval_metrics.compute_tom(test_outputs_pl0, test_outputs_pl_last, test_cases)

        # Aggregate self-consistency metrics
        # For simplicity, assume last pair's outputs approximate overall self-consistency
        sc_n = tom_score
        # For strong self-consistency, check if initial PL semantics match with initial NL
        # For illustration, assume if exact match in test outputs on initial program
        # (This is simplified; real semantic analysis may require external tools)
        ssc_n = 1.0 if pass_at_1 == 1.0 else 0.0

        # Store result
        result_dict = {
            'problem_id': problem_id,
            'pass_at_1': pass_at_1,
            'sc_n': sc_n,
            'ssc_n': ssc_n,
            'tom_score': tom_score,
            'final_nl': nl_next,
            'initial_pl': pl_0,
            'final_pl': pl_prev,
            'test_results': {
                'initial': initial_test_results,
                'final': test_outputs_pl_last
            }
        }
        all_results.append(result_dict)

    # 12. Aggregate over all problems
    # Calculate overall metrics
    sc_scores = [res['sc_n'] for res in all_results]
    ssc_scores = [res['ssc_n'] for res in all_results]
    pass_scores = [res['pass_at_1'] for res in all_results]
    mean_sc = sum(sc_scores) / len(sc_scores) if sc_scores else 0
    mean_ssc = sum(ssc_scores) / len(ssc_scores) if ssc_scores else 0
    mean_pass = sum(pass_scores) / len(pass_scores) if pass_scores else 0

    # Prepare final report
    report = {
        'overall_self_consistency': mean_sc,
        'overall_strong_self_consistency': mean_ssc,
        'overall_pass_at_1': mean_pass,
        'per_problem_results': all_results
    }

    # Print or save report
    import json
    print(json.dumps(report, indent=2))
    # Optionally, save to file
    with open('evaluation_results.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()
```

## model_api.py

```python
## model_api.py
import os
import requests
import logging
from typing import Optional, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class ModelAPI:
    """
    This class provides an interface to interact with either an API-based large language model
    (e.g., OpenAI API) or a local HuggingFace transformers model.

    It supports deterministic decoding (temperature=0), handles prompt sending, and response parsing,
    following the design and configuration specifications.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        model_type: str = "local"
    ):
        """
        Initialize the ModelAPI instance.
        Args:
            model_name (str): Model identifier or local model path.
            api_key (Optional[str]): API key if using API-based models.
            temperature (float): Decoding temperature; default 0 for deterministic.
            max_new_tokens (int): Max tokens to generate.
            model_type (str): "api" for API-based, "local" for local models.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model_type = model_type.lower()

        if self.model_type == "api":
            # Assume using OpenAI API
            # Set API endpoint based on model name if necessary
            if self.model_name.lower().startswith("gpt-4") or self.model_name.lower().startswith("gpt-3.5"):
                self.api_url = "https://api.openai.com/v1/chat/completions"
            else:
                # If other API-based models, need to specify accordingly
                self.api_url = "https://api.openai.com/v1/completions"
            if self.api_key is None:
                raise ValueError("API key must be provided for API-based models.")
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        elif self.model_type == "local":
            # Initialize local model and tokenizer
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except ImportError:
                raise ImportError("Transformers library required for local models.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            # Enable fast generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0,  # assume CUDA device 0; adjust as needed
                trust_remote_code=False
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def call_model(self, prompt: str, max_tokens: int) -> str:
        """
        Send the prompt to the model and return the generated output.
        Args:
            prompt (str): The input prompt string.
            max_tokens (int): Number of tokens to generate.

        Returns:
            str: The model's generated text response.
        """
        if self.model_type == "api":
            return self._call_api(prompt, max_tokens)
        elif self.model_type == "local":
            return self._call_local(prompt, max_tokens)
        else:
            raise RuntimeError(f"Unsupported model_type: {self.model_type}")

    def _call_api(self, prompt: str, max_tokens: int) -> str:
        """
        Call the OpenAI API with the prompt.
        """
        payload = {}
        if self.api_url.endswith("/chat/completions"):
            # Chat completion format
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "n": 1
            }
        else:
            # Text completion format
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "n": 1
            }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30  # seconds
            )
            response.raise_for_status()
            result_json = response.json()

            # Parse response depending on API format
            if self.api_url.endswith("/chat/completions"):
                # Chat API response
                reply = result_json['choices'][0]['message']['content']
            else:
                # Text completion API response
                reply = result_json['choices'][0]['text']
            return reply.strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return f"Error: {str(e)}"
        except (KeyError, IndexError) as e:
            logging.error(f"Response parsing error: {e}")
            return f"Error: Response parsing error: {str(e)}"

    def _call_local(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text from a local transformer model using pipeline with greedy decoding.
        """
        try:
            # Generate with greedy decoding (temperature=0)
            output = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_tokens,
                do_sample=False,
                temperature=0.0,
                num_return_sequences=1,
                # Use top_k=50 if desired; here default suffice
            )
            # output is a list of dicts
            generated_text = output[0]['generated_text']
            # Remove the prompt from the generated text
            # Assumption: generated_text starts with prompt
            if generated_text.startswith(prompt):
                gen_response = generated_text[len(prompt):]
            else:
                gen_response = generated_text
            return gen_response.strip()
        except Exception as e:
            logging.error(f"Local model generation failed: {e}")
            return f"Error: {str(e)}"
```

## prompt_engineering.py

```python
## prompt_engineering.py
import re
from typing import Dict, Any

class PromptTemplate:
    """
    Represents a prompt template with placeholders for dynamic content.
    Provides methods to generate complete prompts for NL→PL and PL→NL tasks.
    """

    def __init__(self, template_str: str):
        """
        Initialize with a raw template string containing placeholders.
        """
        self.template_str = template_str

    def generate(self, **kwargs) -> str:
        """
        Fill in placeholders in the template with provided keyword arguments.
        """
        return self.template_str.format(**kwargs)


class PromptManager:
    """
    Manages prompt templates and provides methods to create specific prompts for
    NL→PL and PL→NL tasks, incorporating configuration parameters and name replacements.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration dictionary loaded from 'config.yaml'.
        """
        # Extract prompt templates from config
        self.nl2pl_template_str = config.get('prompt_templates', {}).get('nl2pl_prompt', '')
        self.pl2nl_template_str = config.get('prompt_templates', {}).get('pl2nl_prompt', '')
        
        # Initialize prompt templates
        self.nl2pl_template = PromptTemplate(self.nl2pl_template_str)
        self.pl2nl_template = PromptTemplate(self.pl2nl_template_str)
        
        # Store other relevant parameters
        self.chain_steps = config.get('prompt_templates', {}).get('chain_steps', 5)
        self.early_stop = config.get('prompt_templates', {}).get('early_stop_on_exact_match', True)
        self.placeholder_function_name = "func"

    def replace_function_names(self, code_str: str) -> str:
        """
        Replace all function names in the code string with the placeholder 'func'.
        Uses AST parsing for robustness.
        """
        import ast
        import astor

        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            # If code is invalid, fallback to regex replacement
            return self._replace_names_regex(code_str)

        class FuncNameReplacer(ast.NodeTransformer):
            def __init__(self):
                self.func_names = set()

            def visit_FunctionDef(self, node):
                self.func_names.add(node.name)
                node.name = self.placeholder_function_name
                self.generic_visit(node)
                return node

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.func_names:
                        node.func.id = self.placeholder_function_name
                elif isinstance(node.func, ast.Attribute):
                    # For method calls, optional: skip or replace attribute if needed
                    pass
                self.generic_visit(node)
                return node

        replacer = FuncNameReplacer()
        replacer.placeholder_function_name = self.placeholder_function_name
        tree = replacer.visit(tree)
        ast.fix_missing_locations(tree)
        replaced_code = astor.to_source(tree)
        return replaced_code

    def _replace_names_regex(self, code_str: str) -> str:
        """
        Fallback method: replace function definitions and calls with regex.
        """
        # Replace function definitions
        code_str = re.sub(r'def\s+(\w+)\s*\(', f'def {self.placeholder_function_name}(', code_str)
        # Replace function calls
        code_str = re.sub(r'(\w+)\s*\(', f'{self.placeholder_function_name}(', code_str)
        return code_str

    def create_nl2pl_prompt(self, task_description: str, test_cases: list = None) -> str:
        """
        Generate NL-to-PL prompt given a task description.
        Optionally include test cases information if provided.
        """
        prompt = self.nl2pl_template.generate(
            task_description=task_description,
            test_cases=self._format_test_cases_for_prompt(test_cases)
        )
        return prompt

    def create_pl2nl_prompt(self, code_snippet: str) -> str:
        """
        Generate PL-to-NL prompt given a code snippet.
        Replaces function names with 'func' to enhance semantic stability.
        """
        replaced_code = self.replace_function_names(code_snippet)
        prompt = self.pl2nl_template.generate(
            code=replaced_code
        )
        return prompt

    def _format_test_cases_for_prompt(self, test_cases: list) -> str:
        """
        Convert test cases list into a string suitable for inclusion in prompts.
        """
        if not test_cases:
            return "No test cases provided."
        formatted = ""
        for idx, test in enumerate(test_cases, 1):
            inputs = test.get('inputs', [])
            # Convert inputs list to comma-separated string
            inputs_str = ', '.join(str(i) for i in inputs)
            formatted += f"Test case {idx}: inputs = [{inputs_str}]\n"
        return formatted
```

## test_executor.py

```python
## test_executor.py
import os
import tempfile
import subprocess
import sys
from typing import List, Dict, Any, Union

class TestExecutor:
    """
    Executes given code snippets on provided test cases, captures outputs or errors,
    and returns detailed results suitable for TOM scoring.
    """

    def __init__(self, max_time: float = 2.0):
        """
        Initialize the executor with a maximum allowed execution time per test.
        Args:
            max_time (float): Timeout in seconds for each code execution.
        """
        self.max_time = max_time

    def execute_test(self, pl_code: str, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs the provided code against each test case, capturing output or errors.
        Args:
            pl_code (str): The code snippet to execute.
            test_cases (list): List of test cases, each with 'inputs' and optional 'expected_output'.
        Returns:
            List of dicts, each containing:
                - 'input': test input
                - 'expected_output': expected output if available
                - 'actual_output': output produced or error message
                - 'passed': boolean indicating correctness if expected_output provided
                - 'error': error message if any
        """
        results = []

        for test_idx, test_case in enumerate(test_cases):
            inputs = test_case.get('inputs', [])
            expected = test_case.get('expected_output', None)

            # Generate a wrapper Python script to execute the function with inputs
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_script:
                script_path = tmp_script.name
                # Build code to execute the function with provided inputs
                # We'll assume the code defines a function named 'func'
                # Inputs are passed as arguments
                call_code = self._generate_call_code(pl_code, inputs)
                tmp_script.write(call_code)
                tmp_script.flush()

            try:
                # Run the script via subprocess with timeout
                process = subprocess.Popen([sys.executable, script_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           universal_newlines=True)
                stdout, stderr = process.communicate(timeout=self.max_time)
                # Clean up script file
                os.remove(script_path)
            except subprocess.TimeoutExpired:
                stdout = ''
                stderr = 'TimeoutExpired'
                os.remove(script_path)
            except Exception as e:
                stdout = ''
                stderr = f'ExecutionError: {str(e)}'
                os.remove(script_path)

            actual_output = stdout.strip() if stdout else ''
            error_msg = stderr.strip() if stderr else ''

            # Determine pass/fail if expected output supplied
            passed = None
            if expected is not None:
                # Check for runtime or syntax errors
                if self._detect_error(error_msg):
                    passed = False
                else:
                    passed = self._compare_outputs(actual_output, expected)

            result = {
                'input': inputs,
                'expected_output': expected,
                'actual_output': actual_output if not error_msg else error_msg,
                'passed': passed,
                'error': error_msg if error_msg else None
            }
            results.append(result)

        return results

    def _generate_call_code(self, code_str: str, inputs: List[Any]) -> str:
        """
        Wraps the user code and generates a script to call 'func' with inputs.
        """
        # Prepare import statements, if needed, or assume 'func' is defined
        # Generate code to call function and print output
        input_strs = []
        for inp in inputs:
            if isinstance(inp, str):
                escaped = inp.replace('"', '\\"')
                input_strs.append(f'"{escaped}"')
            elif isinstance(inp, (int, float)):
                input_strs.append(str(inp))
            elif isinstance(inp, list):
                # For list inputs, build list representation
                input_strs.append(str(inp))
            elif isinstance(inp, dict):
                input_strs.append(str(inp))
            else:
                # fallback to str
                input_strs.append(str(inp))
        args_repr = ', '.join(input_strs)

        call_code = (
            f"{code_str}\n"
            f"try:\n"
            f"    result = func({args_repr})\n"
            f"    if isinstance(result, list) or isinstance(result, dict):\n"
            f"        print(repr(result))\n"
            f"    else:\n"
            f"        print(str(result))\n"
            f"except Exception as e:\n"
            f"    import traceback\n"
            f"    traceback.print_exc()\n"
        )
        return call_code

    def _detect_error(self, output_or_error: str) -> bool:
        """
        Heuristic to identify if the output indicates a syntax or runtime error.
        """
        error_indicators = ['Traceback', 'SyntaxError', 'IndentationError', 'Error', 'Exception', 'TimeoutExpired']
        return any(ind in output_or_error for ind in error_indicators)

    def _compare_outputs(self, actual: str, expected: Any) -> bool:
        """
        Compares actual output string to expected output, allowing for common representations.
        """
        if isinstance(expected, list):
            try:
                # Convert actual string to list
                actual_eval = eval(actual)
                return actual_eval == expected
            except:
                return False
        elif isinstance(expected, dict):
            try:
                actual_eval = eval(actual)
                return actual_eval == expected
            except:
                return False
        else:
            # For strings, numbers
            return str(expected) == actual
```

## utils.py

```python
## utils.py
import re
import ast
import subprocess
import tempfile
import os
from typing import List, Tuple, Union, Dict, Any

def replace_function_names(code_str: str, placeholder: str = 'func') -> str:
    """
    Replace all function names (definition and calls) in a Python code string with a placeholder.
    Uses the `ast` module for robust parsing.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        # If code is invalid, return as is
        return code_str

    class FuncNameReplacer(ast.NodeTransformer):
        def __init__(self):
            # Collect all function names
            self.func_names = set()

        def visit_FunctionDef(self, node):
            original_name = node.name
            self.func_names.add(original_name)
            node.name = placeholder
            self.generic_visit(node)
            return node

        def visit_Call(self, node):
            # Replace function call names if they are in our collected set
            if isinstance(node.func, ast.Name):
                if node.func.id in self.func_names:
                    node.func.id = placeholder
            elif isinstance(node.func, ast.Attribute):
                # For method calls, skip or handle if necessary
                pass
            self.generic_visit(node)
            return node

    replacer = FuncNameReplacer()
    transformed_tree = replacer.visit(tree)
    ast.fix_missing_locations(transformed_tree)

    # Convert AST back to code
    import astor
    replaced_code = astor.to_source(transformed_tree)

    # Alternatively, if astor not available, fallback to codegen or regex
    return replaced_code

def execute_code(code_str: str, test_cases: List[Any], timeout: float = 5.0) -> List[Union[str, Any]]:
    """
    Execute the given Python code with provided test cases.
    Each test case is a dictionary with 'inputs' and 'expected_output'.
    Returns a list of outputs or error messages corresponding to each test case.
    """
    results = []
    for test in test_cases:
        try:
            # Save code to a temporary file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp_file:
                code_file_path = tmp_file.name
                tmp_file.write(code_str)
                tmp_file.flush()

            # Prepare command to execute the code
            # Assuming the code defines a function `func` which takes inputs accordingly
            # Generate a harness to run func with test inputs
            input_args = test['inputs']
            # Build command: python temp_file.py and pass inputs via stdin or arguments
            # For simplicity, assume inputs are passed via stdin or set as variables
            # Here, we'll try to execute the code with arguments
            if isinstance(input_args, list):
                args = [str(arg) for arg in input_args]
            else:
                args = [str(input_args)]

            cmd = ['python', code_file_path]
            # Run the code with input arguments
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Send inputs if necessary
            stdout, stderr = proc.communicate(input=' '.join(args), timeout=timeout)

            if proc.returncode != 0:
                # Runtime or syntax error
                results.append(stderr.strip() if stderr else 'Error: Non-zero exit')
            else:
                results.append(stdout.strip())

            # Clean up
            os.remove(code_file_path)
        except subprocess.TimeoutExpired:
            results.append('Error: Timeout')
        except Exception as e:
            results.append(f'Error: {str(e)}')
            # Ensure cleanup
            if 'code_file_path' in locals() and os.path.exists(code_file_path):
                os.remove(code_file_path)
    return results

def generate_test_cases(problem_metadata: Dict[str, Any], count: int = 16) -> List[Dict[str, Any]]:
    """
    Generate synthetic test cases based on problem metadata.
    If explicit input types are available, create diverse inputs.
    Otherwise, generate generic ones.
    """
    test_cases = []
    # For this implementation, create simple synthetic test cases
    # that are general for numerical inputs or strings
    for _ in range(count):
        # Dummy example: if problem expects integers, generate random integers
        # Here, we assume inputs are integers between 1 and 100
        test_input = []
        # For example purposes, we generate one or two parameters
        # But actual implementation should parse problem metadata
        # For illustration, generate random integers
        import random
        num_params = problem_metadata.get('input_params', 1)
        params = [random.randint(1, 100) for _ in range(num_params)]
        test_case = {'inputs': params}
        # Expected output is unknown; in real scenario, use actual test data or rely on model's output
        test_cases.append(test_case)
    return test_cases

def check_exact_match(str1: str, str2: str) -> bool:
    """
    Check if two strings are exactly equal, ignoring leading/trailing whitespace.
    """
    return str1.strip() == str2.strip()

def detect_syntax_or_runtime_error(output: str) -> bool:
    """
    Simple heuristic to detect syntax or runtime errors based on output message.
    """
    error_indicators = ['SyntaxError', 'IndentationError', 'Error', 'Exception', 'Traceback']
    return any(indicator in output for indicator in error_indicators)

def compute_tom_score(model_outputs: List[Union[str, Any]], expected_outputs: List[Union[str, Any]]) -> float:
    """
    Compute Test Output Match (TOM) score: proportion of exact output matches.
    """
    assert len(model_outputs) == len(expected_outputs), "Outputs and expectations length mismatch"

    match_count = 0
    total = len(model_outputs)
    for out, expected in zip(model_outputs, expected_outputs):
        # Handle error messages
        if isinstance(out, str) and isinstance(expected, str):
            if detect_syntax_or_runtime_error(out) or detect_syntax_or_runtime_error(expected):
                if check_exact_match(out, expected):
                    match_count += 1
            else:
                if check_exact_match(out, expected):
                    match_count += 1
        else:
            # For non-str outputs, compare directly
            if out == expected:
                match_count += 1
    if total == 0:
        return 0.0
    return match_count / total
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\IdentityChain\IdentityChain_repo`
