## dataset_loader.py
import json
import os
from typing import List, Tuple, Dict, Optional
import logging

class DatasetLoader:
    """
    Load datasets required for semantic similarity, WordNet relations,
    and multimodal experiments, formatted for downstream use.
    """
    def __init__(
        self,
        prompt_pairs_path: str = "data/prompt_pairs.json",
        wordnet_relations_path: str = "data/wordnet_relations.json",
        multimodal_data_path: str = "data/multimodal_inputs.json",
        verbose: bool = False
    ):
        """
        Initialize DatasetLoader with dataset file paths.
        Args:
            prompt_pairs_path (str): Path to JSON file with prompt pairs for semantic similarity.
            wordnet_relations_path (str): Path to JSON with WordNet hyponym/hypernym relation data.
            multimodal_data_path (str): Path to JSON with multimodal (image+caption) inputs.
            verbose (bool): Enable debug logging.
        """
        self.prompt_pairs_path = prompt_pairs_path
        self.wordnet_relations_path = wordnet_relations_path
        self.multimodal_data_path = multimodal_data_path
        self.verbose = verbose
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
        
    def load_prompt_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Load prompt pairs with optional human similarity scores.
        Returns:
            List of tuples: (prompt1, prompt2, label)
        """
        data = []
        try:
            with open(self.prompt_pairs_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for entry in raw:
                prompt1 = entry.get("prompt1", "").strip()
                prompt2 = entry.get("prompt2", "").strip()
                label = float(entry.get("label", 0.0))
                data.append((prompt1, prompt2, label))
            if self.verbose:
                print(f"Loaded {len(data)} prompt pairs from {self.prompt_pairs_path}")
        except Exception as e:
            print(f"Error loading prompt pairs: {e}")
        return data

    def load_wordnet_relations(self) -> List[Tuple[str, str, int]]:
        """
        Load WordNet hyponym/hypernym relations.
        Returns:
            List of tuples: (word1, word2, relation_label)
            relation_label: 1 for hyponym, 0 for hypernym
        """
        data = []
        try:
            with open(self.wordnet_relations_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for entry in raw:
                word1 = entry.get("word1", "").strip()
                word2 = entry.get("word2", "").strip()
                relation_str = entry.get("relation", "").strip().lower()
                if relation_str == "hyponym":
                    relation_label = 1
                elif relation_str == "hypernym":
                    relation_label = 0
                else:
                    # Skip unknown relation types
                    continue
                data.append((word1, word2, relation_label))
            if self.verbose:
                print(f"Loaded {len(data)} WordNet relations from {self.wordnet_relations_path}")
        except Exception as e:
            print(f"Error loading WordNet relations: {e}")
        return data

    def load_multimodal_inputs(self) -> List[Dict]:
        """
        Load multimodal data entries: images and captions.
        Returns:
            List of dicts: each with keys 'image' (loaded image object), 'caption', 'prompt'
        """
        import PIL.Image  # Import here to avoid requirements issues if images are not used
        data = []
        try:
            with open(self.multimodal_data_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for entry in raw:
                image_path = entry.get("image_path", "").strip()
                caption = entry.get("caption", "").strip()
                # Load image
                if os.path.isfile(image_path):
                    image = PIL.Image.open(image_path).convert("RGB")
                else:
                    # If image file not found, skip or set to None
                    image = None
                # Format prompt for model input (adjust as needed)
                prompt = f"Describe this image: "
                if image is not None:
                    prompt += "[IMAGE]"  # Placeholder; actual image handling depends on model
                else:
                    prompt += caption  # fallback
                # Store info
                data.append({"image": image, "caption": caption, "prompt": prompt})
            if self.verbose:
                print(f"Loaded {len(data)} multimodal inputs from {self.multimodal_data_path}")
        except Exception as e:
            print(f"Error loading multimodal inputs: {e}")
        return data
