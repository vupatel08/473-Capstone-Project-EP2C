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
