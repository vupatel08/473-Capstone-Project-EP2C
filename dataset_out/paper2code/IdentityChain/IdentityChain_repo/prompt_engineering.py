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
