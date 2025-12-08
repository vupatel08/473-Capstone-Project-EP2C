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
