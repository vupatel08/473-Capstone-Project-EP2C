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
