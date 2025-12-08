## proof_parser.py
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Tactic:
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)

class ProofParser:
    """
    Parses GPT-generated tactic sequences into structured Tactic objects.
    Assumes tactics follow specific command formats:
    - 'euclid_intros'
    - 'euclid_apply <rule> <args>'
    - 'euclid_assert <P>'
    - 'use <X>'
    - 'euclid_finish'
    """

    def __init__(self):
        # Define regex patterns for identifying tactics
        self.pattern_intros = re.compile(r'^\s*euclid_intros\s*$', re.IGNORECASE)
        self.pattern_apply = re.compile(
            r'^\s*euclid_apply\s+([\w\-]+)\s*(.*)$', re.IGNORECASE)
        self.pattern_assert = re.compile(r'^\s*euclid_assert\s+(.+)$', re.IGNORECASE)
        self.pattern_use = re.compile(r'^\s*use\s+([^\s]+)$', re.IGNORECASE)
        self.pattern_finish = re.compile(r'^\s*euclid_finish\s*$', re.IGNORECASE)

    def parse_gpt_output(self, output: str) -> List[Tactic]:
        """
        Parses the raw GPT output string into a list of Tactic objects.

        Args:
            output (str): Raw string output from GPT model.

        Returns:
            List[Tactic]: List of parsed Tactic objects.
        """
        tactics: List[Tactic] = []
        lines = output.splitlines()
        for line in lines:
            line_stripped = line.strip()

            # Skip empty lines or lines that don't match tactics
            if not line_stripped:
                continue

            # Match 'euclid_intros'
            if self.pattern_intros.match(line_stripped):
                tactics.append(Tactic(name='euclid_intros'))
                continue

            # Match 'euclid_apply <rule> <args>'
            match_apply = self.pattern_apply.match(line_stripped)
            if match_apply:
                rule_name = match_apply.group(1)
                args_str = match_apply.group(2).strip()
                # Parse arguments: split by whitespace, handle parentheses if needed
                args_list = self._parse_apply_args(args_str)
                tactics.append(Tactic(
                    name='euclid_apply',
                    parameters={
                        'rule': rule_name,
                        'args': args_list
                    }
                ))
                continue

            # Match 'euclid_assert <P>'
            match_assert = self.pattern_assert.match(line_stripped)
            if match_assert:
                assertion = match_assert.group(1).strip()
                tactics.append(Tactic(name='euclid_assert', parameters={'assertion': assertion}))
                continue

            # Match 'use <X>'
            match_use = self.pattern_use.match(line_stripped)
            if match_use:
                var_name = match_use.group(1)
                tactics.append(Tactic(name='use', parameters={'variable': var_name}))
                continue

            # Match 'euclid_finish'
            if self.pattern_finish.match(line_stripped):
                tactics.append(Tactic(name='euclid_finish'))
                continue

            # If no pattern matched, ignore or log warning
            # Optional: log or print warning for unrecognized line
            # print(f"Warning: Unrecognized tactic line: {line_stripped}")
        return tactics

    def _parse_apply_args(self, args_str: str) -> list:
        """
        Parses the arguments part of 'euclid_apply' command.
        Handles parentheses and multiple arguments.

        Args:
            args_str (str): String containing arguments.

        Returns:
            list: List of argument strings.
        """
        args_list = []

        # Remove surrounding parentheses if present
        args_str = args_str.strip()
        if args_str.startswith('(') and args_str.endswith(')'):
            args_str = args_str[1:-1].strip()

        # Split by commas or whitespace; assume whitespace separation
        # but handle parentheses grouping if needed
        # For simplicity, split on whitespace
        # Note: if arguments contain spaces (e.g., in multiple-word rule names), further parsing needed
        # Here, assume arguments are simple identifiers or parenthesized groups
        tokens = []
        current_token = ''
        paren_level = 0
        for ch in args_str:
            if ch == '(':
                paren_level += 1
                current_token += ch
            elif ch == ')':
                paren_level -= 1
                current_token += ch
            elif ch.isspace() and paren_level == 0:
                if current_token:
                    tokens.append(current_token.strip())
                    current_token = ''
            else:
                current_token += ch
        if current_token:
            tokens.append(current_token.strip())

        # Clean tokens if they are grouped
        for token in tokens:
            token = token.strip()
            if token:
                args_list.append(token)
        return args_list
