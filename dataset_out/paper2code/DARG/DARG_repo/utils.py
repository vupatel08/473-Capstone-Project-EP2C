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
