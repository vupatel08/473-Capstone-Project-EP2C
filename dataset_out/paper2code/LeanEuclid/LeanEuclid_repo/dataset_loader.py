## dataset_loader.py
import os
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ProofRecord:
    problem_id: str
    theorem_statement_informal: str
    formal_proof_steps: List[str]
    category: str
    diagram_image_path: Optional[str]
    ground_truth_formalization: str

@dataclass
class Problem:
    problem_id: str
    problem_statement: str
    informal_proof: str
    category: str
    diagram_image_path: Optional[str]
    ground_truth: str

@dataclass
class Dataset:
    euclid_proofs: List[ProofRecord] = field(default_factory=list)
    unigeo_problems: List[Problem] = field(default_factory=list)
    diagram_map: Dict[str, str] = field(default_factory=dict)

import yaml

class DatasetLoader:
    def __init__(self, config: dict):
        self.euclid_proofs_path: str = config.get('dataset', {}).get('euclid_proofs_path', '')
        self.unigeo_dataset_path: str = config.get('dataset', {}).get('unigeo_dataset_path', '')
        self.diagram_image_dir: str = config.get('dataset', {}).get('diagram_image_dir', '')
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def load_euclid_proofs(self) -> List[ProofRecord]:
        proofs: List[ProofRecord] = []
        path = self.euclid_proofs_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Euclid proofs path not found: {path}")
        files = [f for f in os.listdir(path) if f.endswith('.json')]
        for filename in files:
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Expect each JSON to contain fields: problem_id, theorem_statement_informal, formal_proof_steps, category, ground_truth_formalization
                proof = ProofRecord(
                    problem_id=data.get('problem_id', filename.rstrip('.json')),
                    theorem_statement_informal=data.get('theorem_statement_informal', ''),
                    formal_proof_steps=data.get('formal_proof_steps', []),
                    category=data.get('category', ''),
                    diagram_image_path=data.get('diagram_image_path', None),
                    ground_truth=data.get('ground_truth_formalization', '')
                )
                proofs.append(proof)
            except Exception as e:
                logging.warning(f"Failed to load proof from {file_path}: {e}")
        logging.info(f"Loaded {len(proofs)} Euclid proofs.")
        return proofs

    def load_unigeo_dataset(self) -> List[Problem]:
        problems: List[Problem] = []
        path = self.unigeo_dataset_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"UniGeo dataset path not found: {path}")
        # Assuming problems stored as JSON files per problem
        files = [f for f in os.listdir(path) if f.endswith('.json')]
        for filename in files:
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                problem = Problem(
                    problem_id=data.get('problem_id', filename.rstrip('.json')),
                    problem_statement=data.get('problem_statement', ''),
                    informal_proof=data.get('informal_proof', ''),
                    category=data.get('category', ''),
                    diagram_image_path=data.get('diagram_image_path', None),
                    ground_truth=data.get('ground_truth', '')
                )
                problems.append(problem)
            except Exception as e:
                logging.warning(f"Failed to load problem from {file_path}: {e}")
        logging.info(f"Loaded {len(problems)} UniGeo problems.")
        return problems

    def load_diagram_images(self) -> Dict[str, str]:
        diagram_map: Dict[str, str] = {}
        dir_path = self.diagram_image_dir
        if not os.path.exists(dir_path):
            logging.warning(f"Diagram image directory not found: {dir_path}")
            return diagram_map
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Assumes filename contains problem_id, e.g., 'problem123.png'
                problem_id = os.path.splitext(filename)[0]
                full_path = os.path.join(dir_path, filename)
                diagram_map[problem_id] = full_path
        logging.info(f"Loaded {len(diagram_map)} diagram image mappings.")
        return diagram_map

    def load_all(self) -> Dataset:
        # Load Euclid proofs
        euclid_proofs = self.load_euclid_proofs()
        # Load UniGeo problems
        unigeo_problems = self.load_unigeo_dataset()
        # Load diagram mappings
        diagram_map = self.load_diagram_images()

        # Map problem ids to diagram paths (if available)
        # override diagram_path in problems if in diagram_map
        for problem in unigeo_problems:
            pid = problem.problem_id
            if pid in diagram_map:
                problem.diagram_image_path = diagram_map[pid]

        return Dataset(
            euclid_proofs=euclid_proofs,
            unigeo_problems=unigeo_problems,
            diagram_map=diagram_map
        )
