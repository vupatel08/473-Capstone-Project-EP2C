# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
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
```

## diagram_processor.py

```python
## diagram_processor.py
import os
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
from skimage import feature, transform, color
import math

# Configuration parameters and constants (can be extended or loaded from config.yaml)
# For demonstration, using fixed parameters; in production, load from config if needed.
EDGE_DETECTION_SIGMA: float = 1.0
HOUGH_LINE_THRESHOLD: float = 0.3
HOUGH_MIN_LINE_LENGTH: int = 50
HOUGH_LINE_GAP: int = 10
CIRCLE_DETECTION_RADIUS_RANGE: tuple = (20, 100)  # Adjust based on expected circle size
TOLERANCE: float = 5.0  # Pixel tolerance for geometric relations

class DiagramProcessor:
    def __init__(self, image_path: str):
        self.image_path: str = image_path
        self.image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        self.lines: List[Dict[str, Any]] = []
        self.circles: List[Dict[str, Any]] = []
        self.points: List[Dict[str, Any]] = []
        # Relations
        self.collinearity: List[List[int]] = []
        self.on_line_relations: List[Dict[str, int]] = []
        self.on_circle_relations: List[Dict[str, int]] = []
        self.parallels: List[Dict[str, int]] = []
        self.intersections: List[Dict[str, int]] = []

    def load_image(self) -> None:
        """Loads the image file."""
        img = Image.open(self.image_path)
        self.image = np.array(img)
        print(f"Loaded image with shape {self.image.shape}")

    def preprocess_image(self) -> None:
        """Converts to grayscale and extracts edges."""
        if self.image is None:
            raise RuntimeError("Image not loaded.")
        # Convert to grayscale
        if len(self.image.shape) == 3:
            gray = color.rgb2gray(self.image)
        else:
            gray = self.image / 255.0
        self.gray_image = gray
        # Edge detection
        self.edges = feature.canny(
            gray, sigma=EDGE_DETECTION_SIGMA
        )

    def detect_lines(self) -> None:
        """Detects straight lines using probabilistic Hough transform."""
        if self.edges is None:
            raise RuntimeError("Edge image not processed.")
        lines = transform.probabilistic_hough_line(
            self.edges,
            threshold=int(HOUGH_LINE_THRESHOLD * np.max(self.edges)),
            line_length=HOUGH_MIN_LINE_LENGTH,
            line_gap=HOUGH_LINE_GAP
        )
        for idx, (p0, p1) in enumerate(lines):
            line_dict = {
                'id': f'line_{idx}',
                'start_point': p0,
                'end_point': p1,
                'length': np.linalg.norm(np.array(p0) - np.array(p1)),
                'direction_vector': np.array(p1) - np.array(p0)
            }
            self.lines.append(line_dict)
        print(f"Detected {len(self.lines)} lines.")

    def detect_circles(self) -> None:
        """Detect circles using Hough Circle Transform."""
        if self.gray_image is None:
            raise RuntimeError("Grayscale image not processed.")
        # Using Hough circle detection in skimage
        # Parameters may need tuning based on image resolution
        # Convert to uint8 if needed
        from skimage.transform import hough_circle, hough_circle_peaks
        # Estimate radius range based on image size
        min_radius, max_radius = CIRCLE_DETECTION_RADIUS_RANGE
        # Edge detection for circle detection
        edges = self.edges
        # For better detection, can adjust number of radii
        radii = np.arange(min_radius, max_radius, 2)
        hough_res = hough_circle(edges, radii)
        accums, cx, cy, radii_detected = hough_circle_peaks(
            hough_res, radii, total_num_peaks=10
        )
        for idx, (x, y, r) in enumerate(zip(cx, cy, radii_detected)):
            circle_dict = {
                'id': f'circle_{idx}',
                'center': (float(x), float(y)),
                'radius': float(r)
            }
            self.circles.append(circle_dict)
        print(f"Detected {len(self.circles)} circles.")

    def extract_points(self) -> None:
        """Extract points from detected line endpoints and circle centers."""
        point_coords = {}
        point_id_counter = 0
        # Collect endpoints of lines
        for line in self.lines:
            for pt in [line['start_point'], line['end_point']]:
                key = (round(pt[0]), round(pt[1]))
                if key not in point_coords:
                    point_id = f'point_{point_id_counter}'
                    point_coords[key] = {'id': point_id, 'coord': (float(pt[0]), float(pt[1]))}
                    point_id_counter += 1
        # Collect circle centers
        for circle in self.circles:
            center_unrounded = circle['center']
            key = (round(center_unrounded[0]), round(center_unrounded[1]))
            if key not in point_coords:
                point_id = f'point_{point_id_counter}'
                point_coords[key] = {'id': point_id, 'coord': (float(center_unrounded[0]), float(center_unrounded[1]))}
                point_id_counter += 1
        # Convert to list
        self.points = list(point_coords.values())

    def check_collinearity(self) -> None:
        """Determine collinear points by line detection."""
        # Build a list of point coordinates for distance computation
        coords = {p['id']: p['coord'] for p in self.points}
        point_ids = list(coords.keys())
        for i in range(len(point_ids)):
            for j in range(i + 1, len(point_ids)):
                p_id1 = point_ids[i]
                p_id2 = point_ids[j]
                p1 = np.array(coords[p_id1])
                p2 = np.array(coords[p_id2])
                for k in range(j + 1, len(point_ids)):
                    p_id3 = point_ids[k]
                    p3 = np.array(coords[p_id3])
                    if self._are_collinear(p1, p2, p3):
                        self.collinearity.append([p_id1, p_id2, p_id3])
        print(f"Found {len(self.collinearity)} collinear triplets.")

    def _are_collinear(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        """Check if three points are collinear within tolerance."""
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        return area < TOLERANCE

    def relate_points_on_lines(self) -> None:
        """Determine if points lie on detected lines."""
        for point in self.points:
            point_id = point['id']
            point_coord = np.array(point['coord'])
            for line in self.lines:
                start = np.array(line['start_point'])
                end = np.array(line['end_point'])
                if self._point_on_line(point_coord, start, end):
                    relation = {'point_id': point_id, 'line_id': line['id']}
                    self.on_line_relations.append(relation)
        print(f"Established {len(self.on_line_relations)} point-on-line relations.")

    def _point_on_line(self, pt: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
        """Determine if point is on the line segment within tolerance."""
        line_vec = end - start
        point_vec = pt - start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(pt - start) < TOLERANCE
        # Project point onto line
        projection = np.dot(point_vec, line_vec) / line_len
        if 0 - TOLERANCE <= projection <= line_len + TOLERANCE:
            # Check perpendicular distance
            closest_point = start + (projection / line_len) * line_vec
            dist = np.linalg.norm(pt - closest_point)
            return dist < TOLERANCE
        return False

    def relate_points_on_circles(self) -> None:
        """Determine points lying on circles."""
        for point in self.points:
            point_id = point['id']
            pt = np.array(point['coord'])
            for circle in self.circles:
                center = np.array(circle['center'])
                radius = circle['radius']
                dist = np.linalg.norm(pt - center)
                if abs(dist - radius) < TOLERANCE:
                    relation = {'point_id': point_id, 'circle_center': circle['id']}
                    self.on_circle_relations.append(relation)
        print(f"Established {len(self.on_circle_relations)} point-on-circle relations.")

    def compute_circle_intersections(self) -> None:
        """Compute intersection points between pairs of circles."""
        circle_pairs = []
        for i in range(len(self.circles)):
            for j in range(i + 1, len(self.circles)):
                circle_pairs.append((self.circles[i], self.circles[j]))
        for c1, c2 in circle_pairs:
            pts = self._circle_circle_intersections(c1['center'], c1['radius'], c2['center'], c2['radius'])
            for pt in pts:
                # Register as a point if not duplicated
                key = (round(pt[0]), round(pt[1]))
                point_id = f'intersect_{c1["id"]}_{c2["id"]}_{key}'
                self.points.append({'id': point_id, 'coord': (float(pt[0]), float(pt[1]))})
                self.intersections.append({'point_id': point_id, 'circles': (c1['id'], c2['id'])})
        print(f"Computed {len(self.intersections)} circle-circle intersection points.")

    def _circle_circle_intersections(self, c1: tuple, r1: float, c2: tuple, r2: float) -> List[np.ndarray]:
        """Calculate intersection points of two circles."""
        d = np.linalg.norm(np.array(c2) - np.array(c1))
        if d > r1 + r2 + TOLERANCE or d < abs(r1 - r2) - TOLERANCE:
            return []  # No intersection
        # Compute intersection points
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h_sq = r1**2 - a**2
        if h_sq < 0:
            return []
        h = math.sqrt(h_sq)
        mid_point = np.array(c1) + a / d * (np.array(c2) - np.array(c1))
        offset = h / d * np.array([-(c2[1] - c1[1]), c2[0] - c1[0]])
        intersection1 = mid_point + offset
        intersection2 = mid_point - offset
        if np.linalg.norm(intersection1 - intersection2) < TOLERANCE:
            return [intersection1]
        else:
            return [intersection1, intersection2]

    def determine_parallel_lines(self) -> None:
        """Estimate parallel lines based on their slopes."""
        for i in range(len(self.lines)):
            for j in range(i + 1, len(self.lines)):
                line1 = self.lines[i]
                line2 = self.lines[j]
                if self._are_lines_parallel(line1, line2):
                    self.parallels.append({'line1': line1['id'], 'line2': line2['id']})
        print(f"Found {len(self.parallels)} pairs of parallel lines.")

    def _are_lines_parallel(self, line1: Dict[str, Any], line2: Dict[str, Any]) -> bool:
        """Determine if two lines are parallel within a tolerance."""
        vec1 = line1['direction_vector']
        vec2 = line2['direction_vector']
        cross = np.cross(vec1, vec2)
        return abs(cross) < TOLERANCE

    def extract_relations(self) -> Dict[str, Any]:
        """Compile all extracted relations into structured dict."""
        relations = {
            'collinearity': self.collinearity,
            'on_line': self.on_line_relations,
            'on_circle': self.on_circle_relations,
            'parallel_lines': self.parallels,
            'intersections': self.intersections
        }
        return relations

    def process(self) -> Dict[str, Any]:
        """Complete pipeline: load, preprocess, detect primitives, and extract relations."""
        self.load_image()
        self.preprocess_image()
        self.detect_lines()
        self.detect_circles()
        self.extract_points()
        self.check_collinearity()
        self.relate_points_on_lines()
        self.relate_points_on_circles()
        self.compute_circle_intersections()
        self.determine_parallel_lines()
        relations = self.extract_relations()
        return {
            'points': self.points,
            'lines': self.lines,
            'circles': self.circles,
            'relations': relations
        }
```

## lean_verifier.py

```python
## lean_verifier.py
import subprocess
import tempfile
import os
import logging
from typing import List, Optional

class LeanVerifier:
    """
    Class to verify a given proof script in Lean environment.
    Uses subprocess to invoke local Lean compiler or checker.
    """
    def __init__(self, lean_path: str = "/usr/bin/lean", timeout: int = 10):
        """
        Initialize the verifier with path to Lean executable and timeout.
        """
        self.lean_path: str = lean_path
        self.timeout: int = timeout
        # Set up logging
        logging.basicConfig(level=logging.INFO)

    def verify_proof(self, tactics: List[str], theorem_statement: str = "") -> bool:
        """
        Verify the proof tactics sequence against the theorem statement.
        Returns True if proof is successfully verified, False otherwise.

        Args:
            tactics (List[str]): List of tactic commands in string form.
            theorem_statement (str): The theorem statement (not strictly needed, but for context).

        Returns:
            bool: True if the proof verifies successfully, False otherwise.
        """
        # Build the full proof script
        proof_script = self._construct_proof_script(tactics, theorem_statement)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as tf:
            filename = tf.name
            tf.write(proof_script)
        try:
            # Call Lean to check the proof
            cmd = [self.lean_path, '--check', filename]
            # Run the process
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                universal_newlines=True
            )
            # Debug logs
            logging.info(f"Running command: {' '.join(cmd)}")
            logging.info(f"Lean stdout: {result.stdout}")
            logging.info(f"Lean stderr: {result.stderr}")

            # Check success: lean returns 0 exit code if proof verified
            if result.returncode == 0:
                return True
            else:
                # Verification failed; log error message
                logging.warning(f"Proof verification failed for {filename}")
                return False
        except subprocess.TimeoutExpired:
            logging.error("Lean verification timed out.")
            return False
        except Exception as e:
            logging.exception(f"Error during Lean verification: {e}")
            return False
        finally:
            # Clean up the temp file
            try:
                os.remove(filename)
            except OSError:
                pass

    def _construct_proof_script(self, tactics: List[str], theorem_statement: str) -> str:
        """
        Compose the full Lean proof script with import statements, theorem statement,
        and tactics sequence.
        """
        # Basic Lean import and environment setup, assuming standard import
        header = "import Euclid\n\n"
        # Optional: include the theorem statement as a comment or as the theorem declaration
        # Here, we create a dummy theorem with the provided statement for verification
        # or assume the tactics are embedded within a theorem declaration
        # For this implementation, assume tactics are within a theorem named 'proven_theorem'
        theorem_declaration = f"theorem proven_theorem : \n"
        # For verifying tactics, wrap them into a definition/proof block
        proof_block = "begin\n"
        for t in tactics:
            # Each tactic line is appended
            proof_block += f"  {t.strip()}\n"
        proof_block += "end\n"

        # Alternatively, if tactics are provided as a sequence, place directly
        # For safety, wrap in a proof block
        script = header
        script += f"theorem temp_verification : {theorem_statement}\n"
        script += "begin\n"
        for t in tactics:
            script += f"  {t}\n"
        script += "end\n"
        return script
```

## main.py

```python
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
```

## prompt_engineer.py

```python
## prompt_engineer.py
import yaml
from typing import Optional, Dict, Any

class PromptEngineer:
    """
    Objective:
        Generate structured, problem-specific prompts for GPT-4 / GPT-4V based on provided problem data,
        to facilitate autoformalization of Euclidean geometry proofs.
    """

    def __init__(self, template_str: str, config_path: str = "config.yaml"):
        """
        Initialize with a prompt template string and optional configuration file path.
        """
        self.template_str: str = template_str
        self.config: dict = self._load_config(config_path)
        # Extract example prompts if present in config (for few-shot examples)
        self.examples: str = ""
        if "prompt" in self.config and "examples" in self.config["prompt"]:
            self.examples = self.config["prompt"]["examples"]
        # Store the main template
        self.main_template: str = self.config.get("prompt", {}).get("template", self.template_str)

    def _load_config(self, path: str) -> dict:
        """
        Loads configuration from a YAML file.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            # If file not found or error, return empty dict
            return {}

    def generate_prompt(self,
                        problem_statement: str,
                        diagram_description: Optional[str] = None,
                        problem_context: Optional[str] = None,
                        problem_notes: Optional[str] = None,
                        problem_examples: Optional[str] = None,
                        is_visual: bool = False) -> str:
        """
        Generate a GPT prompt based on problem data.

        Args:
            problem_statement (str): The human-readable natural language theorem or proof step description.
            diagram_description (Optional[str]): Textual description of the diagram or image info (for GPT-4V).
            problem_context (Optional[str]): Additional background or proof context.
            problem_notes (Optional[str]): Additional notes or instructions.
            problem_examples (Optional[str]): Few-shot examples in prompt style.
            is_visual (bool): Whether the problem involves diagrams (True for GPT-4V).

        Returns:
            str: The complete prompt string ready for GPT input.
        """

        # Prepare variables for the template
        prompt_vars: Dict[str, Any] = {
            "problem_statement": problem_statement,
            "diagram_description": diagram_description or "No diagram provided.",
            "context": problem_context or "",
            "notes": problem_notes or "",
            "examples": self.examples
        }

        # Compose prompt based on whether diagram info is provided
        if is_visual:
            # For GPT-4V, embed diagram reference and description
            prompt = self._assemble_prompt_visual(prompt_vars)
        else:
            # For GPT-4 textual input
            prompt = self._assemble_prompt_text(prompt_vars)

        return prompt

    def _assemble_prompt_visual(self, vars: Dict[str, Any]) -> str:
        """
        Assemble prompt for GPT-4V with image reference.
        """
        # The template might contain placeholders like {problem_statement} and {diagram_description}
        prompt_template = self.main_template
        # Replace placeholders in the template
        prompt_filled = prompt_template.format(
            problem_statement=vars["problem_statement"],
            diagram_description=vars["diagram_description"],
            context=vars["context"],
            notes=vars["notes"],
            examples=vars["examples"]
        )
        # Append instruction to include image guidance
        prompt_full = (
            f"{prompt_filled}\n\n"
            "Include the diagram image in your reasoning as per the context. "
            "Use references like 'Diagram Image' or specify image data accordingly. "
            "Generate a sequence of tactic commands following Euclidean style, numbered or ordered, ending with 'euclid_finish'."
        )
        return prompt_full

    def _assemble_prompt_text(self, vars: Dict[str, Any]) -> str:
        """
        Assemble prompt for GPT-4 with textual problem description.
        """
        prompt_template = self.main_template
        prompt_filled = prompt_template.format(
            problem_statement=vars["problem_statement"],
            diagram_description=vars["diagram_description"],
            context=vars["context"],
            notes=vars["notes"],
            examples=vars["examples"]
        )
        prompt_full = (
            f"{prompt_filled}\n\n"
            "Generate a sequence of tactic commands (one per line) for autoformalizing the proof in Lean Euclid style. "
            "Follow the guidelines closely and end with 'euclid_finish'."
        )
        return prompt_full
```

## proof_parser.py

```python
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
```

## smt_checker.py

```python
## smt_checker.py
import logging
from z3 import Solver, Bool, Real, And, Not, sat, unsat, parse_smt2_string, MemoryTimeoutError
from typing import Tuple

class SMTChecker:
    """
    Implements semantic equivalence checking between two geometric formulas 
    using Z3 SMT solver. Encodes formulas as logical assertions and verifies 
    if their logical equivalence holds.
    """
    def __init__(self, z3_solver_path: str = 'z3', timeout_ms: int = 3000):
        """
        Initialize the SMTChecker with path to Z3 solver and optional timeout.

        Args:
            z3_solver_path (str): Path to the Z3 executable or command accessible in PATH.
            timeout_ms (int): Timeout in milliseconds for Z3 solver (default 3000 ms).
        """
        self.z3_solver_path: str = z3_solver_path
        self.timeout_ms: int = timeout_ms

    def check_equivalence(self, formula1: str, formula2: str) -> bool:
        """
        Checks whether two geometric formulas are semantically equivalent
        by verifying if their logical assertions are mutually entailed.

        Args:
            formula1 (str): First formula expression as a string.
            formula2 (str): Second formula expression as a string.

        Returns:
            bool: True if formulas are equivalent (proof succeeds), False otherwise.
        """
        try:
            solver = Solver()
            # Set solver timeout
            solver.set("timeout", self.timeout_ms)

            # Encode formulas into SMT assertions
            # For simplicity, we assume formula1 and formula2 are already in SMT-LIB format
            # or in a string that can be directly parsed to assertions
            # Here, we treat formulas as logical strings to be asserted.

            # For equivalence check: assert (not (= formula1 formula2))
            # This universal negation tests if they are NOT equivalent.
            # If this is unsatisfiable, they are equivalent.
            combined_assertion = f"(assert (not (= {formula1} {formula2})))"
            solver.add(parse_smt2_string(combined_assertion))

            result = solver.check()
            if result == unsat:
                # Unsatisfiable negation => formulas are equivalent
                return True
            elif result == sat:
                # Satisfiable negation => formulas are not equivalent
                return False
            else:
                # Unknown or timeout
                logging.warning("SMT solver returned unknown status.")
                return False
        except MemoryTimeoutError:
            logging.error("SMT solver timeout expired.")
            return False
        except Exception as e:
            logging.exception(f"Exception during SMT verification: {e}")
            return False
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\LeanEuclid\LeanEuclid_repo`
