# prompt_templates.py

"""
This module defines standardized prompt templates used throughout the MutaPLM framework
for pretraining, finetuning, and inference, facilitating consistent chain-of-thought reasoning,
mutation explanation, protein function description, and mutation proposal generation.

The templates incorporate placeholders for dynamic content such as protein sequences,
textual descriptions, mutation details, and special boundary tokens. This organization
supports seamless integration with model training and external API calls (e.g., GPT-4),
aligning with the specifications outlined in Appendix A6 and A7 of the paper.
"""

# Boundary tokens for dialog turn demarcation, as per Appendix A6
BOP = "<BOP>"  # Beginning Of Prompt for explanation
EOP = "<EOP>"  # End Of Prompt
BOM = "<BOM>"  # Beginning Of Mutation or Mutational Effects
EOM = "<EOM>"  # End Of Mutation or Mutational Effects

# ---------------------------------------------------------------------
# 1. Pretraining Prompt Templates
# Designed for masked language modeling and sequence-to-text generation,
# embedding the protein sequence and textual description for alignment.
# ---------------------------------------------------------------------
PRETRAIN_PROTEIN_DESC_TEMPLATE = (
    "Protein: {sequence}\n"
    "Description: {text}"
)

# Example usage during pretraining (for sequence-to-text):
# prompt = PRETRAIN_PROTEIN_DESC_TEMPLATE.format(sequence=protein_seq, text=description_text)

# ---------------------------------------------------------------------
# 2. Chain-of-Thought Prompt Templates for Fine-tuning
# These templates structure multi-round dialogs for protein function description,
# mutation explanation, and mutation engineering proposals, using boundary tokens.
# ---------------------------------------------------------------------

# 2.1 Round 1: Describe protein functions based on sequence
FUNCTION_DESCRIPTION_PROMPT = (
    "You are an expert in biology. Given this protein sequence:\n"
    "{protein_sequence}\n"
    "Please describe its functions in a few sentences."
)

# 2.2 Round 2: Explain mutational effects based on previous function description
MUTATION_EXPLANATION_PROMPT = (
    "{BOP} {function_description} {EOP}\n"
    "Based on the previous function description:\n"
    "{function_description}\n"
    "and the mutation at position {pos} ({original_AA} to {mutant_AA}), "
    "explain the effect of this mutation."
)

# 2.3 Round 2: Proposal of mutation given the mutational effect
MUTATION_PROPOSAL_PROMPT = (
    "{BOM} {effect_description} {EOM}\n"
    "Given the functional effect: {effect_description}\n"
    "propose a single amino acid mutation in sequence:\n"
    "{protein_sequence}"
)

# ---------------------------------------------------------------------
# 3. Inference Prompts for External APIs/Models like GPT-4
# These templates prompt models to generate explanations or mutation proposals.
# ---------------------------------------------------------------------

# 3.1 Explanation generation prompt
EXPLANATION_API_PROMPT = (
    "You are an expert in protein biochemistry. Given the mutation at position "
    "{pos} ({original_AA} to {mutant_AA}) in the protein sequence:\n"
    "{sequence}\n"
    "Explain the potential functional impact."
)

# 3.2 Mutation proposal suggestion prompt
PROPOSAL_API_PROMPT = (
    "Based on the following description of mutational effects:\n"
    "{effect_description}\n"
    "suggest a mutation proposal that could enhance or impair the function as described."
)

# ---------------------------------------------------------------------
# 4. Special Tokens and Placeholders
# These tokens mark parts of multi-turn dialogs and are used to parse or structure prompts.
# They are included in templates above and can be referenced if needed.
# ---------------------------------------------------------------------

# Usage notes:
# - Fill placeholders with `.format()` or f-strings during code implementation.
# - Use boundary tokens (`<BOP>`, `<EOP>`, `<BOM>`, `<EOM>`) to clearly delineate different dialog components.
# - Templates are designed for clarity, consistency, and effective prompting aligned with the paper's Appendix A6 and A7.
