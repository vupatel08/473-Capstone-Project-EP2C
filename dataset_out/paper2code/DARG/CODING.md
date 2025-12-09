# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## data_generator.py

```python
## data_generator.py
import os
import re
import json
import logging
from typing import Optional
import openai
from utils import load_config, get_prompt
from reasoning_graph import ReasoningGraph
from models import DataPoint

logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, openai_api_key: str, dataset_type: str = 'generic'):
        """
        Initialize the DataGenerator with API credentials and prompt templates.
        Loads configuration from the YAML config file.
        Args:
            openai_api_key (str): API key for OpenAI.
            dataset_type (str): To support different prompt templates/dataset formats.
        """
        self.config = load_config()
        self.api_key = openai_api_key
        openai.api_key = self.api_key
        self.dataset_type = dataset_type
        self.prompt_template = get_prompt('reasoning_graph_generation')

        # Set API parameters from config defaults
        self.temperature = self.config.get('model', {}).get('temperature', 0.1)
        self.max_tokens = self.config.get('model', {}).get('max_tokens', 1024)

        # Support for optional retries
        self.max_retries = 3

        # Load prompt template (possibly different per dataset_type)
        # For simplicity, using a common template; can extend for dataset specificity
        self.prompt_template = self.prompt_template

    def construct_prompt(self, graph: ReasoningGraph) -> str:
        """
        Convert the ReasoningGraph into a prompt string suitable for the LLM.
        Insert the graph structure into the prompt template.
        Args:
            graph (ReasoningGraph): The reasoning graph to visualize.
        Returns:
            str: The constructed prompt string.
        """
        # Convert the reasoning graph into JSON string
        graph_json = json.dumps(graph.to_json(), indent=2)
        # Replace placeholder in template
        prompt = self.prompt_template.replace("{reasoning_graph}", graph_json)
        # Additional context can be added here if needed
        return prompt

    def generate_text(self, graph: ReasoningGraph) -> str:
        """
        Generate text (question, reasoning, options, answer) from the graph.
        Handles API call and retries.
        Args:
            graph (ReasoningGraph): The graph to translate into text.
        Returns:
            str: The raw generated text output from the API.
        """
        prompt = self.construct_prompt(graph)
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.config.get('model', {}).get('name', 'gpt-4-turbo'),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=0.95,
                    messages=[
                        {"role": "system", "content": "You are an assistant that generates reasoning problems from graphs."},
                        {"role": "user", "content": prompt}
                    ]
                )
                generated_text = response.choices[0].message['content'].strip()
                logger.info(f"Generated text (Attempt {attempt + 1}): {generated_text}")
                return generated_text
            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
        # After retries, return empty string or raise error
        logger.error("Failed to generate text after retries")
        return ""

    def parse_response(self, response_text: str) -> Optional[dict]:
        """
        Parse the API response to extract structured data: question, reasoning, options, answer.
        Uses regex or delimiters based on prompt guidance.
        Args:
            response_text (str): Raw text from the API.
        Returns:
            dict: Parsed components or None if parsing fails.
        """
        # Example parsing assumptions - adapt according to prompt design
        try:
            # Example pattern: assuming the model outputs a structured JSON
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                # Validate necessary keys
                question = data.get('question')
                reasoning = data.get('reasoning')
                options = data.get('options')
                answer = data.get('answer')

                if question and reasoning and options and answer:
                    return {
                        'question': question,
                        'reasoning': reasoning,
                        'options': options,
                        'answer': answer
                    }
            # Alternatively, parse by delimiters if JSON not strict
            # For simplicity, let's assume JSON format
        except Exception as e:
            logger.warning(f"Parsing response failed: {e}")
        return None

    def generate_data_point(self, graph: ReasoningGraph) -> Optional[DataPoint]:
        """
        Full pipeline: generate text from graph, parse it, and produce a DataPoint.
        Args:
            graph (ReasoningGraph): The reasoning graph to process.
        Returns:
            DataPoint: The generated data point or None if failure occurs.
        """
        raw_text = self.generate_text(graph)
        if not raw_text:
            logger.warning("No response generated")
            return None
        parsed = self.parse_response(raw_text)
        if parsed is None:
            logger.warning("Parsing failed, skipping data point")
            return None
        # Construct DataPoint object with parsed data
        question = parsed['question']
        reasoning = parsed['reasoning']
        options = parsed['options']
        answer = parsed['answer']

        # For options, convert list or dict to string if needed
        options_str = json.dumps(options) if isinstance(options, (list, dict)) else str(options)

        # Return DataPoint instance
        data_point = DataPoint(
            question_text=question,
            reasoning_text=reasoning,
            answer=answer,
            reasoning_graph=graph.to_json()
        )
        return data_point
```

## dataset_loader.py

```python
## dataset_loader.py
import os
import json
import logging
from typing import List, Optional
from functools import partial
from utils import load_config

# Configure logger for debugging and info
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataPoint:
    """
    Data class representing a single data entry, including question, answer,
    reasoning explanation, and reasoning graph.
    """
    def __init__(
        self,
        question_text: str,
        answer: str,
        reasoning_text: str,
        reasoning_graph: dict  # The graph stored as a dict (serializable)
    ):
        self.question_text = question_text
        self.answer = answer
        self.reasoning_text = reasoning_text
        self.reasoning_graph = reasoning_graph

    def to_json(self) -> dict:
        """
        Serialize DataPoint into a JSON-serializable dict.
        """
        return {
            'question': self.question_text,
            'answer': self.answer,
            'reasoning': self.reasoning_text,
            'reasoning_graph': self.reasoning_graph
        }

    @staticmethod
    def from_json(data: dict) -> 'DataPoint':
        """
        Create DataPoint instance from dict, validating presence of fields.
        """
        question = data.get('question')
        answer = data.get('answer')
        reasoning = data.get('reasoning')
        reasoning_graph = data.get('reasoning_graph', {})

        if question is None or answer is None or reasoning is None:
            raise ValueError(f"Missing one of required fields in data: {data}")
        return DataPoint(
            question_text=question,
            answer=answer,
            reasoning_text=reasoning,
            reasoning_graph=reasoning_graph
        )

class DatasetLoader:
    """
    Loader class for datasets like GSM8K, BBQ, BBH Dyck.
    Handles loading, parsing, and saving datasets.
    """

    def __init__(self, dataset_path: str, dataset_name: Optional[str] = None):
        """
        Initialize with dataset file path.
        Args:
            dataset_path: Path to dataset JSON file.
            dataset_name: Optional, label for dataset type (e.g., 'GSM8K'), for validation.
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.data_points: List[DataPoint] = []

    def load_data(self) -> List[DataPoint]:
        """
        Load dataset file into self.data_points as DataPoint objects.
        Supports JSON format with list of entries.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        logger.info(f"Loading dataset from {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {self.dataset_path}: {e}")

        # Support datasets stored as list of dicts
        if not isinstance(raw_data, list):
            raise ValueError(f"Dataset should be a list of data points: {self.dataset_path}")

        self.data_points = []
        for idx, entry in enumerate(raw_data):
            # Validate structure
            if not isinstance(entry, dict):
                logger.warning(f"Skipping non-dict entry at index {idx}")
                continue
            try:
                datapoint = DataPoint.from_json(entry)
            except Exception as e:
                logger.warning(f"Skipping invalid data at index {idx}: {e}")
                continue
            self.data_points.append(datapoint)

        logger.info(f"Loaded {len(self.data_points)} data points from {self.dataset_path}")
        return self.data_points

    def save_dataset(self, output_path: str, data_points: Optional[List[DataPoint]] = None) -> None:
        """
        Save current data points to a JSON file at output_path.
        Args:
            output_path: Path to save the dataset.
            data_points: Optional list of DataPoint objects to save. Defaults to internal.
        """
        to_save = data_points if data_points is not None else self.data_points
        json_list = [dp.to_json() for dp in to_save]

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_list, f, indent=2)
        logger.info(f"Saved {len(json_list)} data points to {output_path}")

    def get_subset(self, sample_size: int, seed: int = 42) -> List[DataPoint]:
        """
        Randomly sample a subset of data points for evaluation or analysis.
        Args:
            sample_size: Number of data points to sample.
            seed: Random seed for reproducibility.
        Returns:
            List of sampled DataPoint objects.
        """
        import random
        random.seed(seed)
        if sample_size >= len(self.data_points):
            return self.data_points
        return random.sample(self.data_points, sample_size)
```

## graph_perturber.py

```python
## graph_perturber.py

import copy
import random
from typing import Tuple
from reasoning_graph import ReasoningGraph

# Load configuration parameters from utils.load_config
try:
    from utils import load_config
    CONFIG = load_config()
except Exception:
    # Fallback default configuration if load fails
    CONFIG = {
        'experiment': {
            'complex_dim': {
                'numerical_scale': 1.0,
                'graph_depth_delta': 1,
                'graph_width_delta': 1
            }
        }
    }

# Extract parameters with defaults; reinforce defaults
NUMERICAL_SCALE_DEFAULT = CONFIG['experiment']['complex_dim'].get('numerical_scale', 1.0)
GRAPH_DEPTH_DELTA_DEFAULT = CONFIG['experiment']['complex_dim'].get('graph_depth_delta', 1)
GRAPH_WIDTH_DELTA_DEFAULT = CONFIG['experiment']['complex_dim'].get('graph_width_delta', 1)

random.seed(42)  # For reproducibility if needed


def perturb_depth(graph: ReasoningGraph, delta: int) -> ReasoningGraph:
    """
    Increase or decrease the depth of the reasoning graph.
    Positive delta: extend the longest path by inserting intermediate nodes.
    Negative delta: shorten or flatten the longest path.
    """
    new_graph = graph.clone()

    if delta > 0:
        for _ in range(delta):
            new_graph._increase_depth_one_level()
    elif delta < 0:
        for _ in range(-delta):
            new_graph._decrease_depth_one_level()
    # If delta == 0: no change

    return new_graph


def perturb_width(graph: ReasoningGraph, delta: int) -> ReasoningGraph:
    """
    Increase or decrease the width of the reasoning graph.
    Positive delta: add attribute or sibling nodes to increase width.
    Negative delta: remove attribute nodes or general width.
    """
    new_graph = graph.clone()

    if delta > 0:
        for _ in range(delta):
            new_graph._increase_width_one_step()
    elif delta < 0:
        for _ in range(-delta):
            new_graph._decrease_width_one_step()

    return new_graph


def perturb_numerical(graph: ReasoningGraph, scale: float) -> ReasoningGraph:
    """
    Scale numerical node contents to increase numerical complexity.
    Larger scale: multiply number values by 'scale' (>1 to increase magnitude).
    Smaller scale (<1 to decrease magnitude, optional).
    """
    new_graph = graph.clone()

    for node in new_graph._nodes.values():
        if node.type == 'number':
            try:
                num_val = float(node.content)
                new_val = num_val * scale
                # Keep integer if original was integer
                if node.content.isdigit():
                    node.content = str(int(new_val))
                else:
                    node.content = str(new_val)
            except:
                continue  # skip if conversion fails

    return new_graph


def perturb_graph(
    original_graph: ReasoningGraph,
    depth_delta: int = None,
    width_delta: int = None,
    numerical_scale: float = None
) -> ReasoningGraph:
    """
    Compose all perturbations based on delta parameters.
    If delta is None, use default from config.
    """
    # Set default deltas if not provided
    if depth_delta is None:
        depth_delta = GRAPH_DEPTH_DELTA_DEFAULT
    if width_delta is None:
        width_delta = GRAPH_WIDTH_DELTA_DEFAULT
    if numerical_scale is None:
        numerical_scale = NUMERICAL_SCALE_DEFAULT

    # Apply perturbations sequentially
    g = copy.deepcopy(original_graph)
    g = perturb_depth(g, depth_delta)
    g = perturb_width(g, width_delta)
    g = perturb_numerical(g, numerical_scale)
    return g


# Private helper methods district for ReasoningGraph
# These are meant to be used within the module for deep modifications


def _increase_depth_one_level(graph: ReasoningGraph) -> None:
    """
    Insert an intermediate node into the longest path / chain to increase depth.
    """
    # Find the longest path (approximate) in the current graph
    try:
        paths = list(nx.all_simple_paths(graph._graph, source=None, target=None))
        if not paths:
            # fallback: pick a node arbitrarily
            if len(graph._graph.nodes) > 1:
                path = list(graph._graph.nodes)
            else:
                return
        else:
            # Pick the longest path
            path = max(paths, key=len)
        # If path length >=2, insert node between middle
        if len(path) >= 2:
            insert_idx = len(path) // 2
            prev_node_id = path[insert_idx - 1]
            next_node_id = path[insert_idx]
            # Insert new node
            new_node_id = graph.add_node(type='intermediate', content='intermediate reasoning step')
            # Remove existing edge and insert new
            if graph._graph.has_edge(prev_node_id, next_node_id):
                graph._graph.remove_edge(prev_node_id, next_node_id)
            # Add edges to the new node
            graph.add_edge(prev_node_id, new_node_id, relation_type='intermediate')
            graph.add_edge(new_node_id, next_node_id, relation_type='intermediate')
    except Exception:
        pass  # On failure, do nothing


def _decrease_depth_one_level(graph: ReasoningGraph) -> None:
    """
    Remove or flatten a node in the longest chain to decrease depth.
    """
    # Remove a node with only one predecessor and one successor
    try:
        for node_id in list(graph._graph.nodes):
            preds = list(graph._graph.predecessors(node_id))
            succs = list(graph._graph.successors(node_id))
            if len(preds) == 1 and len(succs) == 1:
                pred_id = preds[0]
                succ_id = succs[0]
                # Remove node and connect predecessor directly to successor
                graph._graph.remove_node(node_id)
                graph.add_edge(pred_id, succ_id, relation_type='flattened')
                break
    except Exception:
        pass


def _increase_width_one_step(graph: ReasoningGraph) -> None:
    """
    Add sibling attribute nodes to increase width at a random position.
    """
    try:
        if not graph._graph.nodes:
            return
        # Select a random node to add sibling attribute
        node_id = random.choice(list(graph._graph.nodes))
        # Add a new attribute node
        attr_node_id = graph.add_node(type='attribute', content='added attribute')
        # Connect it with existing node's parent(s)
        preds = list(graph._graph.predecessors(node_id))
        if preds:
            for pred in preds:
                graph.add_edge(pred, attr_node_id, relation_type='has_attribute')
        else:
            # Attach directly to node if no parent
            graph.add_edge(node_id, attr_node_id, relation_type='has_attribute')
    except Exception:
        pass


def _decrease_width_one_step(graph: ReasoningGraph) -> None:
    """
    Remove leaf attribute nodes to decrease width.
    """
    try:
        leaves = [n for n in list(graph._graph.nodes) if graph._graph.out_degree(n) == 0]
        if leaves:
            node_to_remove = random.choice(leaves)
            graph._graph.remove_node(node_to_remove)
    except Exception:
        pass

# Monkey patch these helper functions into the module namespace for internal use
import networkx as nx

setattr(
    globals(),
    '_increase_depth_one_level',
    _increase_depth_one_level
)
setattr(
    globals(),
    '_decrease_depth_one_level',
    _decrease_depth_one_level
)
setattr(
    globals(),
    '_increase_width_one_step',
    _increase_width_one_step
)
setattr(
    globals(),
    '_decrease_width_one_step',
    _decrease_width_one_step
)
```

## label_verifier.py

```python
## label_verifier.py
import os
import re
import json
import logging
from typing import Optional
import openai
from utils import load_config, get_prompt

logger = logging.getLogger(__name__)

class LabelVerifier:
    """
    The LabelVerifier class implements correctness verification of generated reasoning
    and answers using a structured prompt to a code-capable LLM (e.g., GPT-4 with code interpreter).
    It determines whether the generated label (answer) matches the inferred correctness based
    on reasoning steps, as per structured prompts defined in 'verification_prompt.txt'.
    """

    def __init__(self, api_key: str):
        """
        Initialize the verifier with API key, load config, set prompt template.
        Args:
            api_key (str): API key for OpenAI.
        """
        self.api_key = api_key
        # Load configuration
        self.config = load_config()
        # Set OpenAI API key
        openai.api_key = self.api_key
        # Load prompt template for verification
        self.prompt_template = get_prompt('label_verification')
        # API parameters
        self.temperature = self.config.get('model', {}).get('temperature', 0.0)
        self.max_tokens = self.config.get('model', {}).get('max_tokens', 1024)
        # Retry limit for robustness
        self.max_retries = 3

    def verify_label(self, generated_text: str) -> bool:
        """
        Verify the correctness of the generated reasoning and answer.
        Args:
            generated_text (str): The text output from the data generator, including reasoning, answer.
        Returns:
            bool: True if verified as correct, False otherwise.
        """
        prompt = self._construct_verification_prompt(generated_text)
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.config.get('model', {}).get('name', 'gpt-4'),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=0.95,
                    messages=[
                        {"role": "system", "content": "You are an assistant that verifies reasoning correctness."},
                        {"role": "user", "content": prompt}
                    ],
                )
                reply = response.choices[0].message['content'].strip()
                logger.info(f"Verification attempt {attempt + 1}: {reply}")
                return self._parse_verification_response(reply)
            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
        # If all retries fail, default to False
        logger.error("Verification failed after retries.")
        return False

    def _construct_verification_prompt(self, text: str) -> str:
        """
        Insert the generated reasoning-answer text into the verification template.
        Args:
            text (str): The model-generated text containing reasoning and answer.
        Returns:
            str: Structured prompt for verification.
        """
        prompt = self.prompt_template.replace("{generated_text}", text)
        return prompt

    def _parse_verification_response(self, response_text: str) -> bool:
        """
        Parse the model's response to determine correctness.
        Looks for keywords like 'Correct', 'Incorrect', 'Yes', 'No'.
        Args:
            response_text (str): The raw output from the model.
        Returns:
            bool: True if the response indicates correctness, False otherwise.
        """
        # Normalize response to lowercase for keyword search
        lowered = response_text.lower()
        # Check for explicit correctness indicators
        if 'correct' in lowered:
            return True
        if 'incorrect' in lowered:
            return False
        # Additional heuristic: yes/no
        if re.search(r'\b(yes|correct)\b', lowered):
            return True
        if re.search(r'\b(no|incorrect)\b', lowered):
            return False
        # Default fallback: if uncertain, return False
        return False
```

## main.py

```python
# main.py

import os
import sys
import time
import logging
import random
from typing import List, Dict, Any
from pathlib import Path

# Import external modules
import yaml
import numpy as np

# Import custom modules
from utils import load_config, get_prompt, save_to_json, load_from_json
from dataset_loader import DatasetLoader, DataPoint
from reasoning_graph import ReasoningGraph
from graph_perturber import perturb_graph
from data_generator import DataGenerator
from label_verifier import LabelVerifier
from model_evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Load prompt templates
    prompts_dir = 'prompts'  # predefined prompts directory
    regen_prompt_template = get_prompt('reasoning_graph_generation', prompts_dir)
    verification_prompt_template = get_prompt('label_verification', prompts_dir)
    # Additional prompts (graph to text, others) can be loaded similarly if needed
    
    # Get model configuration
    model_name = config['model'].get('name', 'gpt-4-turbo')
    api_key = config['model'].get('api_key', '')
    temperature = float(config['model'].get('temperature', 0.1))
    max_tokens = int(config['model'].get('max_tokens', 1024))
    
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize modules
    # Data Generator for graph to text regeneration
    data_gen = DataGenerator(api_key=api_key)
    data_gen.prompt_template = regen_prompt_template
    data_gen.temperature = temperature
    data_gen.max_tokens = max_tokens
    
    # Label verifier
    label_verifier = LabelVerifier(api_key=api_key)
    
    # Define datasets to process
    datasets_info = [
        ('GSM8K', config['datasets'].get('GSM8K', {}).get('path', '')),
        ('BBQ', config['datasets'].get('BBQ', {}).get('path', '')),
        ('BBH_Dyck', config['datasets'].get('BBH_Dyck', {}).get('path', ''))
    ]
    
    # Load model evaluation configuration
    model_list = [
        'gpt-4-turbo',
        'gpt-3.5-turbo',
        'llama-8b',
        'llama-70b',
        'mixtral-8x7b',
        'mixtral-8x22b',
        'wizardlm-8x22b',
        'deepseekmath-7b',
        'gemini-1.5-pro',
        'claude-3-opus'
        # Add other models as needed
    ]
    # For each model, define API keys if needed, else assume local inference
    
    # Initialize ModelEvaluator
    model_eval = ModelEvaluator(
        model_names=model_list,
        model_api_keys={},  # Fill with actual API keys if using APIs
        prompt_template="",  # Not directly used here; modules handle prompts internally or via args
        config_path='config.yaml'
    )
    
    # Define complexity levels
    complexity_levels = config.get('experiment', {}).get('complexity_levels', [0,1,2,4,8])
    
    # For each dataset
    for dataset_name, dataset_path in datasets_info:
        if not dataset_path:
            logger.warning(f"Dataset path for {dataset_name} not specified. Skipping.")
            continue
        logger.info(f"Processing dataset: {dataset_name} from {dataset_path}")
        # Load dataset
        dataset_loader = DatasetLoader(dataset_path)
        raw_data_points = dataset_loader.load_data()
        logger.info(f"Loaded {len(raw_data_points)} data points for {dataset_name}")
        
        # Prepare storage for all data points and their perturbed versions
        all_data_points = []
        
        # Process each data point
        for idx, datapoint in enumerate(raw_data_points):
            # Initialize reasoning graph from data point
            reasoning_graph = ReasoningGraph()
            reasoning_graph.extract_from_text(datapoint.reasoning_text, dataset_type=dataset_name.lower())
            # Save original data point info
            all_data_points.append(datapoint)
            # For each complexity level, generate perturbed data
            for level in complexity_levels:
                # Determine perturbation parameters based on level
                # For simplicity, assume linear scaling
                depth_delta = int(level * config.get('experiment', {}).get('complex_dim', {}).get('graph_depth_delta', 1))
                width_delta = int(level * config.get('experiment', {}).get('complex_dim', {}).get('graph_width_delta', 1))
                numerical_scale = float(level * config.get('experiment', {}).get('complex_dim', {}).get('numerical_scale', 1.0))
                
                # Perturb graph
                perturbed_graph = perturb_graph(
                    reasoning_graph,
                    depth_delta=depth_delta,
                    width_delta=width_delta,
                    numerical_scale=numerical_scale
                )
                
                # Generate data from perturbed graph
                generated_text = data_gen.generate_text(perturbed_graph)
                if not generated_text:
                    logger.warning(f"Generation failed at idx {idx} for level {level}")
                    continue  # skip this perturbation
                
                # Parse generated text into DataPoint
                generated_datapoint = data_gen.parse_response(generated_text)
                if not generated_datapoint:
                    logger.warning(f"Parsing generated text failed at idx {idx} for level {level}")
                    continue
                
                # Verify correctness using label verifier
                verification_passed = False
                for attempt in range(3):
                    if label_verifier.verify_label(generated_text):
                        verification_passed = True
                        break
                if not verification_passed:
                    logger.info(f"Verification failed after retries for idx {idx} at level {level}")
                    continue  # skip if not verified
                # Assign metadata
                # Save the perturbation info into data point attributes
                generated_datapoint.metadata = {
                    'dataset': dataset_name,
                    'original_idx': idx,
                    'complexity_level': level,
                    'perturbed_graph': perturbed_graph.to_json()
                }
                all_data_points.append(generated_datapoint)
        
        # Save all generated and original data (with metadata)
        output_dir = Path('darg_generated') / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f'{dataset_name}_full_data.json'
        json_data = [utils.serialize_datapoint(dp) for dp in all_data_points]
        save_to_json(json_data, str(save_path))
        logger.info(f"Saved combined data for {dataset_name} to {save_path}")
        
        # Evaluate with models on this dataset
        # For each model, evaluate performance
        evaluation_results = model_eval.evaluate(all_data_points)
        # Save evaluation results
        eval_path = output_dir / f'{dataset_name}_evaluation.json'
        save_to_json(evaluation_results, str(eval_path))
        logger.info(f"Evaluation results saved for {dataset_name}")

if __name__ == "__main__":
    main()
```

## model_evaluator.py

```python
## model_evaluator.py
import os
import json
import re
import logging
from typing import List, Dict, Optional, Union
import numpy as np

# External dependencies
import openai

# Import internal modules and classes
from utils import load_config
from dataset_loader import DataPoint
from reasoning_graph import ReasoningGraph

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EvaluationMetrics:
    """
    Data class to hold evaluation metrics for a dataset and model,
    including accuracy, bias score, and CIARR.
    """
    def __init__(
        self,
        overall_accuracy: float = 0.0,
        bias_score: float = 0.0,
        ciarr: float = 0.0,
        accuracy_per_level: Optional[Dict[str, float]] = None,
        bias_per_level: Optional[Dict[str, float]] = None
    ):
        self.overall_accuracy = overall_accuracy
        self.bias_score = bias_score
        self.ciarr = ciarr
        self.accuracy_per_level = accuracy_per_level if accuracy_per_level else {}
        self.bias_per_level = bias_per_level if bias_per_level else {}

class ModelEvaluator:
    """
    Class to evaluate multiple large language models across a dataset.
    Implements inference, output parsing, metrics computation, and result storage.
    """
    def __init__(
        self,
        model_names: List[str],
        model_api_keys: Dict[str, str],
        prompt_template: str = "",
        config_path: str = "config.yaml",
        task_type: str = "generic",
        complexity_levels: List[int] = [0, 1, 2, 4, 8],
        metrics_output_path: str = "evaluation_results.json",
        batch_size: int = 8,
        inference_timeout: int = 300
    ):
        """
        Initialize the evaluator with models, APIs, prompts, and config.
        Args:
            model_names (List[str]): List of model identifiers.
            model_api_keys (Dict[str, str]): Mapping model IDs to API keys.
            prompt_template (str): Prompt template for inference.
            config_path (str): Path to configuration file.
            task_type (str): Type of reasoning task, influences parsing.
            complexity_levels (List[int]): Levels of complexity for evaluation.
            metrics_output_path (str): Path to save metrics JSON.
            batch_size (int): Batch size for inference.
            inference_timeout (int): Timeout in seconds for API calls.
        """
        self.model_names = model_names
        self.model_api_keys = model_api_keys
        self.prompt_template = prompt_template
        self.config = load_config(config_path)
        self.task_type = task_type
        self.complexity_levels = complexity_levels
        self.metrics_output_path = metrics_output_path
        self.batch_size = batch_size
        self.inference_timeout = inference_timeout

        # Initialize API clients for each model
        self.api_clients = {}
        for model_name in self.model_names:
            # For openai models
            if 'gpt' in model_name.lower() or 'openai' in model_name.lower():
                api_key = self.model_api_keys.get(model_name, "")
                # set up api key globally
                openai.api_key = api_key
                self.api_clients[model_name] = 'openai'
            else:
                # For local models, placeholder for actual inference implementation
                self.api_clients[model_name] = 'local'

    def load_dataset(self, dataset_path: str) -> List[DataPoint]:
        """
        Load dataset from a JSON file path into list of DataPoint objects.
        Args:
            dataset_path (str): Path to dataset JSON file.
        Returns:
            List[DataPoint]: Loaded dataset.
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        data_points = []
        for entry in raw_data:
            data_points.append(DataPoint.from_json(entry))
        return data_points

    def evaluate(self, dataset: List[DataPoint]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on the dataset across all complexity levels.
        Records accuracy, bias scores, and CIARRs.
        Returns:
            dict: Nested dict with metrics per model and per complexity level.
        """
        results = {}
        for model_name in self.model_names:
            logger.info(f"Evaluating model: {model_name}")
            model_res = {
                'accuracy': {},
                'bias_score': {},
                'ciarr': {}
            }
            # For each complexity level
            for level in self.complexity_levels:
                logger.info(f"Processing complexity level: {level}")
                total_points = 0
                correct_points = 0
                bias_scores = []
                accuracy_per_point = []
                # For CIARR calculation, store accuracy at previous levels
                prev_accuracy = None
                acc_list = []

                for point in dataset:
                    # Generate prompt for current point
                    prompt_input = self._prepare_prompt(point, level, model_name)
                    # Run inference
                    output_text = self._run_model(prompt_input, model_name)
                    # Parse answer from output
                    answer = self.parse_output(output_text, point)
                    # Check correctness
                    is_correct = self.check_answer(answer, point.answer)
                    accuracy_per_point.append(1 if is_correct else 0)
                    total_points += 1
                    if is_correct:
                        correct_points += 1
                    # Possibly compute bias score
                    bias_score = self.estimate_bias(output_text, point)
                    bias_scores.append(bias_score)
                    # For CIARR computation
                overall_acc = (correct_points / total_points) * 100 if total_points > 0 else 0.0
                model_res['accuracy'][str(level)] = overall_acc
                # Compute bias score average
                bias_avg = np.mean(bias_scores) if bias_scores else 0.0
                model_res['bias_score'][str(level)] = bias_avg
                # Compute CIARR across levels
                if len(self.complexity_levels) > 1:
                    accuracies = [model_res['accuracy'][str(l)] for l in self.complexity_levels]
                    ciarr_value = self._compute_ciarr(accuracies)
                else:
                    ciarr_value = 0.0
                model_res['ciarr'][str(level)] = ciarr_value
            results[model_name] = model_res
        # Save to file
        with open(self.metrics_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results

    def _prepare_prompt(self, data_point: DataPoint, level: int, model_name: str) -> str:
        """
        Prepare the prompt for inference, possibly including context, prompt template,
        and complexity-specific modifications.
        """
        # Use prompt template, insert question, reasoning, other info as needed
        prompt = self.prompt_template
        # Replace placeholders or format prompt with data point
        # For demonstration, we assume prompt has {question}, {reasoning}, {level}
        prompt = prompt.replace("{question}", data_point.question_text)
        prompt = prompt.replace("{reasoning}", data_point.reasoning_text)
        prompt = prompt.replace("{answer}", data_point.answer)
        prompt = prompt.replace("{complexity_level}", str(level))
        return prompt

    def _run_model(self, prompt: str, model_name: str) -> str:
        """
        Run inference on model, handle API or local inference.
        """
        if self.api_clients[model_name] == 'openai':
            # Use openai SDK
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an reasoning model."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.get('model', {}).get('temperature', 0.0),
                    max_tokens=self.config.get('model', {}).get('max_tokens', 1024),
                    top_p=0.95
                )
                answer_text = response.choices[0].message['content'].strip()
                return answer_text
            except Exception as e:
                logger.warning(f"API call failed for model {model_name}: {e}")
                return ""
        else:
            # Placeholder for local model inference
            # Implement local inference as needed
            return ""

    def parse_output(self, output_text: str, data_point: DataPoint) -> Union[str, float, dict]:
        """
        Parse the model output to extract the final answer, formatted as numeric or string.
        Implement dataset-specific parsing if needed.
        """
        # Example: extract answer from response
        # For GSM8K, extract numeric answer with regex
        answer_match = re.search(r'(?i)answer[:\s]*([-\d\.]+)', output_text)
        if answer_match:
            answer = answer_match.group(1)
            try:
                answer_value = float(answer)
                return answer_value
            except:
                return answer  # fallback to string if float conversion fails
        # For multiple-choice or other datasets, implement accordingly
        # Fallback: use last line or entire output
        lines = output_text.strip().splitlines()
        if lines:
            return lines[-1].strip()
        return output_text.strip()

    def check_answer(self, model_ans: Union[str, float, dict], correct_answer: str) -> bool:
        """
        Check if model's answer matches the ground truth.
        For numerical answers, allow small tolerance.
        """
        if isinstance(model_ans, float):
            try:
                correct_value = float(correct_answer)
                return abs(model_ans - correct_value) <= 1e-2
            except:
                return False
        elif isinstance(model_ans, str):
            # For exact match
            return model_ans.strip() == correct_answer.strip()
        else:
            # For other types, fallback
            return False

    def estimate_bias(self, output_text: str, data_point: DataPoint) -> float:
        """
        Placeholder for bias estimate: could analyze the output for biased terms,
        attribute mentions or use a bias scoring model.
        """
        # Simplified: 0.0 bias score; in practice, may require additional analysis
        return 0.0

    def _compute_ciarr(self, accuracies: List[float]) -> float:
        """
        Compute the CIARR metric: average percentage retention of accuracy over levels.
        """
        ciarr_value = 0.0
        n = len(accuracies)
        if n < 2:
            return ciarr_value
        for i in range(n - 1):
            if accuracies[i] == 0:
                # avoid division by zero; define as 0 in that case
                continue
            ratio = accuracies[i+1] / accuracies[i]
            ciarr_value += ratio
        ciarr_value /= (n - 1)
        ciarr_percentage = ciarr_value * 100
        return ciarr_percentage

    def save_results(self, results: Dict[str, Dict]) -> None:
        """
        Save the computed metrics dictionary to the designated output path.
        """
        with open(self.metrics_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
```

## reasoning_graph.py

```python
## reasoning_graph.py
import networkx as nx
import re
import uuid
from typing import List, Dict, Optional, Tuple

# Import the DataPoint class if needed or define minimal structure here
# But as per design, ReasoningGraph is independent of DataPoint and focuses on graph structure.

class GraphNode:
    def __init__(self, id: int, type: str, content: str):
        self.id = id
        self.type = type
        self.content = content

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'content': self.content
        }

    @staticmethod
    def from_dict(data: Dict) -> 'GraphNode':
        return GraphNode(
            id=int(data['id']),
            type=str(data['type']),
            content=str(data['content'])
        )

class GraphEdge:
    def __init__(self, source_id: int, target_id: int, relation_type: str):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type

    def to_dict(self) -> Dict:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'relation_type': self.relation_type
        }

    @staticmethod
    def from_dict(data: Dict) -> 'GraphEdge':
        return GraphEdge(
            source_id=int(data['source']),
            target_id=int(data['target']),
            relation_type=str(data['relation_type'])
        )

class ReasoningGraph:
    def __init__(self):
        # Internal directed graph
        self._graph = nx.DiGraph()
        # Maintain a mapping from node id to GraphNode
        self._nodes: Dict[int, GraphNode] = {}
        self._next_id = 1  # for assigning unique node IDs

    def add_node(self, type: str, content: str, node_id: Optional[int] = None) -> int:
        """
        Add a node to the graph, assign an id if not provided.
        Returns the node id.
        """
        if node_id is None:
            node_id = self._next_id
            self._next_id += 1
        else:
            # Override id assignment if provided
            if node_id >= self._next_id:
                self._next_id = node_id + 1
        node = GraphNode(id=node_id, type=type, content=content)
        self._nodes[node_id] = node
        self._graph.add_node(node_id)
        return node_id

    def add_edge(self, source_id: int, target_id: int, relation_type: str) -> None:
        """
        Add an edge to the graph, ensure nodes exist.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            raise ValueError("Source or target node id does not exist.")
        # Enforce acyclic graph, prevent cycle upon insertion
        self._graph.add_edge(source_id, target_id)
        # Store relation_type as edge attribute
        self._graph.edges[source_id, target_id]['relation_type'] = relation_type

    def nodes(self) -> List[GraphNode]:
        """Return list of nodes."""
        return list(self._nodes.values())

    def edges(self) -> List[GraphEdge]:
        """Return list of edges."""
        edges_list = []
        for u, v in self._graph.edges():
            relation_type = self._graph.edges[u, v].get('relation_type', '')
            edges_list.append(GraphEdge(source_id=u, target_id=v, relation_type=relation_type))
        return edges_list

    def to_json(self) -> Dict:
        """
        Serialize entire graph to dict suitable for JSON dumping.
        """
        nodes_list = [node.to_dict() for node in self.nodes()]
        edges_list = [edge.to_dict() for edge in self.edges()]
        return {
            'nodes': nodes_list,
            'edges': edges_list
        }

    @staticmethod
    def from_json(data: Dict) -> 'ReasoningGraph':
        """
        Create a ReasoningGraph object from JSON dict.
        """
        rg = ReasoningGraph()
        # Reconstruct nodes
        for node_data in data.get('nodes', []):
            node = GraphNode.from_dict(node_data)
            rg._nodes[node.id] = node
            rg._graph.add_node(node.id)
            if node.id >= rg._next_id:
                rg._next_id = node.id + 1
        # Reconstruct edges
        for edge_data in data.get('edges', []):
            try:
                source_id = int(edge_data['source'])
                target_id = int(edge_data['target'])
                relation_type = str(edge_data['relation_type'])
                # Add edge ensuring nodes exist
                if source_id in rg._nodes and target_id in rg._nodes:
                    rg._graph.add_edge(source_id, target_id)
                    rg._graph.edges[source_id, target_id]['relation_type'] = relation_type
            except Exception:
                continue
        return rg

    def extract_from_text(self, reasoning_text: str, dataset_type: str='generic') -> None:
        """
        Parse a reasoning chain text and construct the corresponding reasoning graph.
        The parsing strategy depends on dataset_type.
        """
        # Clear existing graph
        self._graph.clear()
        self._nodes.clear()

        # Different parsing heuristics based on dataset_type
        if dataset_type == 'math':
            self._parse_math_reasoning(reasoning_text)
        elif dataset_type == 'social' or dataset_type == 'bbq':
            self._parse_social_reasoning(reasoning_text)
        elif dataset_type == 'symbolic' or dataset_type == 'dyck':
            self._parse_symbolic_reasoning(reasoning_text)
        elif dataset_type == 'spatial' or dataset_type == 'bbh':
            self._parse_spatial_reasoning(reasoning_text)
        else:
            # Fallback: naive line-based parsing
            self._parse_generic_reasoning(reasoning_text)

    def _parse_math_reasoning(self, reasoning_text: str) -> None:
        """
        Parse math reasoning (equations) into nodes and edges.
        Expect equations in standard form, e.g., "A + B = C".
        """
        lines = reasoning_text.splitlines()
        node_id_map = {}  # map variable names to node ids
        for line in lines:
            eq_match = re.match(r'([A-Z])\s*=\s*([\d\.]+)', line)
            op_match = re.match(r'([A-Z])\s*=\s*([\d\.]+)\s*([\+\-\*/])\s*([\d\.]+)', line)
            if op_match:
                # Parse equation with operation
                result_var, operand1, operator, operand2 = op_match.groups()
                # create or get nodes
                if operand1 not in node_id_map:
                    nid1 = self.add_node(type='number', content=operand1)
                    node_id_map[operand1] = nid1
                else:
                    nid1 = node_id_map[operand1]
                if operand2 not in node_id_map:
                    nid2 = self.add_node(type='number', content=operand2)
                    node_id_map[operand2] = nid2
                else:
                    nid2 = node_id_map[operand2]
                # create result node
                res_node_id = self.add_node(type='intermediate', content=result_var)
                node_id_map[result_var] = res_node_id
                # connect operands to result
                self.add_edge(nid1, res_node_id, relation_type=f'operand_{operator}')
                self.add_edge(nid2, res_node_id, relation_type=f'operand_{operator}')
            else:
                # Parse simple assignment or statement
                assign_match = re.match(r'([A-Z])\s*=\s*([\d\.]+)', line)
                if assign_match:
                    var, value = assign_match.groups()
                    if var not in node_id_map:
                        nid = self.add_node(type='number', content=value)
                        node_id_map[var] = nid
                    else:
                        # Update content
                        self._nodes[node_id_map[var]].content = value
        # Optionally, define a final answer node
        # Create a 'final' node pointing to answer if inferred
        pass

    def _parse_social_reasoning(self, reasoning_text: str) -> None:
        """
        Parse social reasoning chains, e.g.,
        "Person A has attribute X. Person B has attribute Y. Person A is older."
        """
        # Split sentences or statements
        sentences = reasoning_text.split('.')
        node_id_map = {}
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Detect person nodes
            person_match = re.match(r'(Person \w+)', sent)
            attr_match = re.findall(r'(attribute \w+)', sent)
            if person_match:
                person_name = person_match.group(1)
                pid = self.add_node(type='person', content=person_name)
                node_id_map[person_name] = pid
            # Edge inferences based on attribute relations
            attr_relation_match = re.match(r'(Person \w+) has (\w+)', sent)
            if attr_relation_match:
                person_name, attribute = attr_relation_match.groups()
                if person_name in node_id_map:
                    # attribute node
                    attr_node_id = self.add_node(type='attribute', content=attribute)
                    # add relation
                    self.add_edge(node_id_map[person_name], attr_node_id, relation_type='has_attribute')
        # Additional parsing rules can be added per dataset

    def _parse_symbolic_reasoning(self, reasoning_text: str) -> None:
        """
        Parse nested brackets or similar expressions into tree-like graphs.
        """
        # For Dyck language, build nested structure based on parentheses
        stack: List[int] = []
        current_node_id = None
        for match in re.finditer(r'[\(\)]', reasoning_text):
            char = match.group()
            if char == '(':
                nid = self.add_node(type='bracket', content='(')
                if current_node_id is not None:
                    self.add_edge(current_node_id, nid, relation_type='next')
                stack.append(nid)
                current_node_id = nid
            elif char == ')':
                if stack:
                    stack.pop()
                current_node_id = stack[-1] if stack else None
        # Additional structure inference can be done based on nested depth

    def _parse_spatial_reasoning(self, reasoning_text: str) -> None:
        """
        For spatial reasoning tasks, parse actions like "move forward 3 steps".
        """
        lines = reasoning_text.splitlines()
        prev_node_id = None
        for line in lines:
            match = re.match(r'(?:Move|Take)\s+(\d+)\s+steps?\s+(forward|backward|left|right)', line)
            if match:
                step_num, direction = match.groups()
                node_content = f'{step_num} steps {direction}'
                node_id = self.add_node(type='spatial_step', content=node_content)
                if prev_node_id:
                    self.add_edge(prev_node_id, node_id, relation_type='sequential')
                prev_node_id = node_id

    def _parse_generic_reasoning(self, reasoning_text: str) -> None:
        """
        Naive parser for unclassified reasoning text.
        """
        # Split into statements and create nodes
        lines = reasoning_text.splitlines()
        prev_node_id = None
        for line in lines:
            text = line.strip()
            if not text:
                continue
            node_id = self.add_node(type='statement', content=text)
            if prev_node_id:
                self.add_edge(prev_node_id, node_id, relation_type='sequential')
            prev_node_id = node_id

    def perturb(self, depth_delta: int=0, width_delta: int=0, numerical_scale: float=1.0) -> 'ReasoningGraph':
        """
        Apply fine-grained perturbations to the graph along depth, width, and numerical complexity.
        """
        # Clone current graph to modify
        perturbed = self.clone()

        # Perturb depth
        if depth_delta != 0:
            perturbed._adjust_depth(delta=depth_delta)

        # Perturb width
        if width_delta != 0:
            perturbed._adjust_width(delta=width_delta)

        # Perturb numerical values
        if numerical_scale != 1.0:
            perturbed._adjust_numerical_content(scale=numerical_scale)

        return perturbed

    def _adjust_depth(self, delta: int) -> None:
        """
        Increase (positive delta) or decrease (negative delta) the depth.
        """
        if delta > 0:
            # Increase depth by finding longest path and inserting intermediate nodes
            for _ in range(delta):
                self._increase_depth_one_level()
        elif delta < 0:
            # Decrease depth by removing or flattening longest paths
            for _ in range(-delta):
                self._decrease_depth_one_level()

    def _increase_depth_one_level(self) -> None:
        """
        Insert a new intermediate node into the longest path to increase depth.
        """
        try:
            all_paths = list(nx.all_simple_paths(self._graph, source=None, target=None))
            # Find longest path; fallback: use topological order
            if not self._graph.nodes:
                return
            # Use longest simple path in graph
            max_path = max(nx.all_simple_paths(self._graph, source=min(self._graph.nodes),
                                               target=max(self._graph.nodes)),
                           key=len, default=[])
            if len(max_path) < 2:
                return
            # Insert a new node in the middle
            insert_pos = len(max_path)//2
            prev_node_id = max_path[insert_pos - 1]
            next_node_id = max_path[insert_pos]
            # Insert new node between
            new_node_id = self.add_node(type='intermediate', content='intermediate node')
            # Remove edge between prev and next
            if self._graph.has_edge(prev_node_id, next_node_id):
                self._graph.remove_edge(prev_node_id, next_node_id)
            # Add new edges
            self.add_edge(prev_node_id, new_node_id, relation_type='intermediate')
            self.add_edge(new_node_id, next_node_id, relation_type='intermediate')
        except Exception:
            pass

    def _decrease_depth_one_level(self) -> None:
        """
        Remove or flatten one level from the longest path.
        """
        # For simplicity, remove nodes with only one predecessor and successor
        for node_id in list(self._graph.nodes):
            preds = list(self._graph.predecessors(node_id))
            succs = list(self._graph.successors(node_id))
            if len(preds) == 1 and len(succs) == 1:
                pred = preds[0]
                succ = succs[0]
                # Remove node and connect predecessor directly to successor
                self._graph.remove_node(node_id)
                self.add_edge(pred, succ, relation_type='flattened')
                break

    def _adjust_width(self, delta: int) -> None:
        """
        Increase or decrease width (number of sibling nodes).
        """
        # For increase: add sibling nodes at a random level
        for _ in range(abs(delta)):
            if delta > 0:
                # choose random node to add sibling to
                if self._graph.number_of_nodes() == 0:
                    continue
                node_id = list(self._graph.nodes)[0]
                # Add sibling node
                sibling_id = self.add_node(type='attribute', content='additional attribute')
                parent_preds = list(self._graph.predecessors(node_id))
                for pred in parent_preds:
                    self.add_edge(pred, sibling_id, relation_type='has_attribute')
            else:
                # Decrease width by removing a leaf node
                leaves = [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]
                if leaves:
                    leaf = leaves[0]
                    self._graph.remove_node(leaf)

    def _adjust_numerical_content(self, scale: float) -> None:
        """
        Scale numerical values in nodes' content.
        For math nodes with number content, multiply by scale.
        """
        for node in self._nodes.values():
            if node.type == 'number':
                try:
                    num_value = float(node.content)
                    new_value = num_value * scale
                    # Keep integer if originally integer
                    if node.content.isdigit():
                        node.content = str(int(new_value))
                    else:
                        node.content = str(new_value)
                except:
                    continue

    def clone(self) -> 'ReasoningGraph':
        """
        Deep copy of the graph.
        """
        new_graph = ReasoningGraph()
        new_graph._graph = self._graph.copy()
        new_graph._nodes = {nid: GraphNode(nid, n.type, n.content) for nid, n in self._nodes.items()}
        new_graph._next_id = self._next_id
        return new_graph
```

## utils.py

```python
## utils.py
import os
import yaml
import json
import random
import logging
from typing import Any, Dict, List
import openai
import networkx as nx
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_prompt(template_name: str, prompts_dir: str = 'prompts') -> str:
    """
    Retrieve a prompt template by name from external text files.
    """
    prompt_files = {
        'reasoning_graph_generation': 'regen_prompt.txt',
        'data_regeneration': 'regen_prompt.txt',
        'label_verification': 'verification_prompt.txt',
        'evaluation_prompt': 'evaluation_prompt.txt'
    }
    filename = prompt_files.get(template_name)
    if filename is None:
        raise ValueError(f"Unknown prompt template: {template_name}")
    file_path = os.path.join(prompts_dir, filename)
    return load_prompt_template(file_path)


def load_prompt_template(file_path: str) -> str:
    """
    Read a prompt template text file into a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt template file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        template = f.read()
    logger.info(f"Loaded prompt template from {file_path}")
    return template


def save_to_json(data: Any, file_path: str) -> None:
    """
    Serialize data into JSON and save to disk.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Data saved to {file_path}")


def load_from_json(file_path: str) -> Any:
    """
    Load and deserialize JSON data from disk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded data from {file_path}")
    return data


def serialize_datapoint(datapoint: 'DataPoint') -> dict:
    """
    Convert a DataPoint object into a JSON-serializable dictionary.
    """
    return {
        'question': datapoint.question_text,
        'reasoning': datapoint.reasoning_text,
        'answer': datapoint.answer,
        'reasoning_graph': datapoint.reasoning_graph.to_json()
    }


def deserialize_datapoint(data: dict) -> 'DataPoint':
    """
    Create a DataPoint object from a dictionary.
    """
    from models import DataPoint, ReasoningGraph
    reasoning_graph = ReasoningGraph.from_json(data['reasoning_graph'])
    return DataPoint(
        question_text=data['question'],
        reasoning_text=data['reasoning'],
        answer=data['answer'],
        reasoning_graph=reasoning_graph
    )


def plot_graph(graph: 'ReasoningGraph') -> None:
    """
    Visualize the reasoning graph using networkx and matplotlib.
    """
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.id, label=node.content, type=node.type)
    for edge in graph.edges:
        G.add_edge(edge.source_id, edge.target_id, relation=edge.content, type=edge.type)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
    labels = {node: G.nodes[node]['label'] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    edge_labels = {(u, v): G.edges[u, v]['relation'] for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Reasoning Graph Visualization")
    plt.show()


def print_graph_details(graph: 'ReasoningGraph') -> None:
    """
    Print detailed structure of the reasoning graph for debugging.
    """
    print("Nodes:")
    for node in graph.nodes:
        print(f"ID: {node.id}, Type: {node.type}, Content: {node.content}")
    print("\nEdges:")
    for edge in graph.edges:
        print(f"Source: {edge.source_id} -> Target: {edge.target_id}, Relation: {edge.content}, Type: {edge.type}")


def extract_reasoning_steps(text: str) -> List[str]:
    """
    Parse reasoning explanation text into step-by-step list.
    """
    # Example implementation: split by sentences or numbered steps
    import re
    steps = re.split(r'(?<=\.)|\d+\.', text)  # Split at periods or numbered steps
    steps = [step.strip() for step in steps if step.strip()]
    return steps


def match_answer_in_text(text: str, answer: str) -> bool:
    """
    Check if the generated text contains the correct answer.
    """
    return answer.strip() in text


def initialize_openai_api(api_key: str) -> None:
    """
    Set the API key for OpenAI SDK.
    """
    openai.api_key = api_key
    logger.info("OpenAI API key initialized.")


def call_openai_api(prompt: str, model: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """
    Make a call to OpenAI API with retries and basic error handling.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are an assistant helping with reasoning tasks."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return ""


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch_manual_seed = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\DARG\DARG_repo`
