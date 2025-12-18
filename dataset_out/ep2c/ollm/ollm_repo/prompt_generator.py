## prompt_generator.py

"""
This module provides functions to generate prompts for training and inference
of the ontology learning model, following the styles specified in the paper.
It uses prompt templates defined based on the paper's appendix figures and description.
The functions include:
- get_chain_of_thought_prompt()
- get_direct_prompt()
- get_instruction_prompt()

They are designed to ensure prompt consistency and fidelity to the original paper's
prompt styles, facilitating reliable subgraph generation and reproducibility.
"""

from typing import List, Dict

# Define default prompt templates matching figures 6-8 from appendix
# These templates incorporate placeholders for document text, concept list, and reasoning steps.

def get_chain_of_thought_prompt(
    document_text: str,
    concepts: List[str],
    additional_parameters: Dict = None
) -> str:
    """
    Generates a chain-of-thought prompt for model to generate concept subgraphs with reasoning.
    This prompt guides the model through explicit reasoning steps.

    Args:
        document_text (str): The text of the document (e.g., summary, abstract).
        concepts (List[str]): List of concepts relevant to the document.
        additional_parameters (Dict, optional): Additional parameters for prompt styling.

    Returns:
        str: The formatted chain-of-thought prompt string.
    """
    # Retrieve additional parameters if provided
    params = additional_parameters if additional_parameters else {}

    # Use the predefined template or define a default one
    template = params.get(
        "template",
        """Given the document:
{doc}
and the list of concepts:
{concepts}

Explain your reasoning step by step. Based on your reasoning, 
list the relevant concept relation paths as a list of sequences where each sequence is a chain of concepts connected by '->'. 
For example:
- {concept1} -> {relation} -> {concept2} -> {relation} -> {concept3}
After reasoning, generate the concept subgraph in the form of paths listed above, each on a new line.

Please elucidate your reasoning clearly and produce the relation paths accordingly."""
    )

    # Format the prompt
    prompt = template.format(
        doc=document_text,
        concepts=', '.join(concepts)
    )
    return prompt


def get_direct_prompt(
    document_text: str,
    concepts: List[str],
    additional_parameters: Dict = None
) -> str:
    """
    Generates a direct, instruction-style prompt for the model to produce relevant concept relations,
    without explicit reasoning steps, suitable for zero-shot or inference.

    Args:
        document_text (str): The text of the document.
        concepts (List[str]): List of concepts relevant to the document.
        additional_parameters (Dict, optional): Additional parameters.

    Returns:
        str: The formatted direct prompt string.
    """
    params = additional_parameters if additional_parameters else {}

    template = params.get(
        "template",
        """Given the document:
{doc}
and the list of concepts:
{concepts}
Provide the concept relation subgraph associated with this document as a list of relation paths, each on a new line, in the form:
- {concept} -> {relation} -> {concept}
List only the relevant relations inferred from the document. Do not include explanations or reasoning."""
    )

    prompt = template.format(
        doc=document_text,
        concepts=', '.join(concepts)
    )
    return prompt


def get_instruction_prompt(
    instruction_type: str = "task_instructions",
    additional_parameters: Dict = None
) -> str:
    """
    Generates a general instruction prompt used as a system prompt or task description.
    This can include dataset details, task description, or formatting guidelines.

    Args:
        instruction_type (str): Type of instruction prompt, e.g., "task_instructions".
        additional_parameters (Dict, optional): Additional context or instructions.

    Returns:
        str: The instruction prompt string.
    """
    # For simplicity, we define a static template matching the paper style.
    # This can be extended based on 'instruction_type' or other parameters.
    template = additional_parameters.get(
        "template",
        """You are an AI assistant tasked with constructing part of an ontology. 
Given documents and concepts, generate subgraphs representing their hierarchical relations.
Use clear, concise language and produce relation paths in the format:
- {concept} -> {relation} -> {concept}
Ensure outputs are well-structured and adhere to the formatting examples provided in the documentation."""
    )

    return template
