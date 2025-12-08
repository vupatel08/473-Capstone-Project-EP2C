# main.py
import os
import json
import yaml
import openai
import logging
from typing import List, Dict, Any, Optional

# Import modules from local files
from dataset_loader import DatasetLoader, ProofRecord, Problem, Dataset
from diagram_processor import DiagramProcessor
from prompt_engineer import PromptEngineer
from proof_parser import ProofParser
from lean_verifier import LeanVerifier
from smt_checker import SMTChecker

# Load configuration
with open("config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize main components
dataset_loader = DatasetLoader(config)

# Load dataset
dataset = dataset_loader.load_all()

# Initialize prompt engineer with the template
prompt_template = config.get("prompt", {}).get("template",
    "Solve the geometry problem:\n{problem_statement}\nDiagram context: {diagram_description}\nGenerate formal tactics sequence...")
prompt_engineer = PromptEngineer(prompt_template)

# Initialize proof parser
proof_parser = ProofParser()

# Initialize GPT API parameters
gpt_model = config.get("model", {}).get("gpt_model", "gpt-4")
api_key = config.get("model", {}).get("openai_api_key", "")
temperature = float(config.get("model", {}).get("temperature", 0.2))
max_tokens = int(config.get("model", {}).get("max_tokens", 1500))

# Initialize Lean verifier
lean_path = config.get("verifier", {}).get("lean_path", "/usr/bin/lean")
lean_verifier = LeanVerifier(lean_path)

# Initialize SMT checker
z3_path = config.get("verifier", {}).get("z3_solver_path", "z3")
smt_checker = SMTChecker(z3_path)

# Evaluation thresholds
semantic_similarity_threshold = float(config.get("evaluation", {}).get("semantic_similarity_threshold", 0.6))
verification_success_threshold = float(config.get("evaluation", {}).get("verification_success_threshold", 0.8))

# Utility function to call OpenAI API
def call_gpt(prompt: str) -> str:
    responses = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[{"role": "system", "content": "You are a geometric autoformalization assistant."},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None
    )
    return responses.choices[0].message['content']

# Placeholder function for embedding diagram in prompt
def get_diagram_description(image_path: Optional[str]) -> str:
    if image_path and os.path.exists(image_path):
        # For GPT-4V, could include image; here, just describe or omit
        return f"Diagram image at {image_path}"
    else:
        return "No diagram provided."

# Function to process each problem
def process_problem(problem: Problem) -> Dict[str, Any]:
    result = {
        "problem_id": problem.problem_id,
        "formal_proof": None,
        "parsed_tactics": [],
        "verified": False,
        "ground_truth_formal": problem.ground_truth,
        "semantic_equivalent": False,
        "similarity_score": 0.0,
        "smt_equivalence": False,
        "error": None
    }
    try:
        # Generate diagram description (if applicable)
        diagram_description = get_diagram_description(problem.diagram_image_path)

        # Generate prompt for GPT
        prompt = prompt_engineer.generate_prompt(
            problem_statement=problem.problem_statement,
            diagram_description=diagram_description,
            # Additional context or notes can be added here
        )

        # Call GPT API
        gpt_response = call_gpt(prompt)
        logging.info(f"GPT Response for problem {problem.problem_id}:\n{gpt_response}")

        # Parse GPT output into tactics list
        tactics_list = proof_parser.parse_gpt_output(gpt_response)
        result["parsed_tactics"] = tactics_list

        # Verify in Lean
        proof_verified = lean_verifier.verify_proof(tactics_list, problem.problem_statement)
        result["verified"] = proof_verified

        # If proof is verified, we can attempt semantic equivalence check
        if proof_verified:
            # Construct formula strings representing the generated proof and ground truth
            # Here, assume we have functions to serialize proofs/formulas; simulated as strings
            pred_formula = "GeneratedFormulaPlaceholder"  # This should be replaced by actual formula extraction
            gt_formula = problem.ground_truth

            # Check semantic equivalence via SMT
            is_equivalent = smt_checker.check_equivalence(pred_formula, gt_formula)
            result["semantic_equivalent"] = is_equivalent
            result["smt_equivalence"] = is_equivalent

        # Compute similarity score (placeholder, as actual implementation requires string similarity)
        # For example, using Levenshtein or cosine similarity on tactic sequences
        # Here, set to 1.0 if verified, else 0.0 as placeholder
        if result["verified"]:
            result["similarity_score"] = 1.0
        else:
            result["similarity_score"] = 0.0

    except Exception as e:
        logging.exception(f"Error processing problem {problem.problem_id}")
        result["error"] = str(e)

    return result

# Main execution loop
def main():
    # Containers for overall metrics
    total_problems = len(dataset.euclid_proofs) + len(dataset.unigeo_problems)
    verified_count = 0
    semantically_correct_count = 0
    SMT_pass_count = 0

    results = []

    # Process Euclid proofs
    for proof_record in dataset.euclid_proofs:
        logging.info(f"Processing Euclid proof {proof_record.problem_id}")
        res = process_problem(
            Problem(
                problem_id=proof_record.problem_id,
                problem_statement=proof_record.theorem_statement_informal,
                informal_proof="",  # Not used here
                category=proof_record.category,
                diagram_image_path=proof_record.diagram_image_path,
                ground_truth=proof_record.ground_truth_formalization
            )
        )
        results.append(res)
        # Update counters
        if res['verified']:
            verified_count += 1
        if res['semantic_equivalent']:
            semantically_correct_count += 1
        if res['smt_equivalence']:
            SMT_pass_count += 1

    # Process UniGeo problems
    for problem in dataset.unigeo_problems:
        logging.info(f"Processing UniGeo problem {problem.problem_id}")
        res = process_problem(problem)
        results.append(res)
        # Update counters
        if res['verified']:
            verified_count += 1
        if res['semantic_equivalent']:
            semantically_correct_count += 1
        if res['smt_equivalence']:
            SMT_pass_count += 1

    # Output overall statistics
    print("=== Autoformalization Results Summary ===")
    print(f"Total problems processed: {total_problems}")
    print(f"Successfully verified in Lean: {verified_count} ({verified_count / total_problems * 100:.2f}%)")
    print(f"Semantically equivalent (SMT confirmed): {semantically_correct_count} ({semantically_correct_count / total_problems * 100:.2f}%)")
    print(f"SMT verification success: {SMT_pass_count} ({SMT_pass_count / total_problems * 100:.2f}%)")

    # Save detailed results to a file
    with open("autoformalization_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
