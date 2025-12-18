## dataset_loader.py
import os
import json
import csv
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional

# Load configuration for dataset paths and settings
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Initialize tokenizer for protein sequences and texts
# From the config, get model names
PLM_NAME = CONFIG['model'].get('plm_name', 'facebook/esm2_t6_8a_14B')
TEXT_ENCODER_NAME = CONFIG['model'].get('text_encoder_name', 'allenai/scibert_scivocab_uncased')

tokenizer_sequence = AutoTokenizer.from_pretrained(PLM_NAME)
tokenizer_text = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
# Assume that the tokenizers have consistent behavior
# For protein sequences, we may customize tokenizer if needed
# Here, we'll use the same tokenizer for simplicity, or customize as needed
# For now, we assume the models' tokenizers handle amino acids as chars or tokens

# Utility function for enrichment via GPT API (placeholder/mock)
def gpt_text_enrichment(prompt: str, literature_refs: List[str]) -> str:
    """
    Given a prompt and literature references, use GPT API or placeholder to expand mutational text.
    """
    # Placeholder implementation: in practice, replace with API call
    # For testing, return a dummy enriched text or the original text
    return "Enriched description based on literature and prompt."

class ProteinTextDataset(Dataset):
    """
    Dataset class to handle loading, preprocessing, and serving data for different modes:
    pretraining, finetuning, testing.
    """
    def __init__(self,
                 data_path: str,
                 mode: str,
                 config: Dict[str, Any]):
        """
        Initialize dataset.
        Args:
            data_path (str): Path to the data file (JSONL or CSV).
            mode (str): One of {'pretraining', 'finetuning', 'test'}.
            config (dict): Configuration dictionary with dataset settings.
        """
        self.data_path = data_path
        self.mode = mode.lower()
        self.config = config
        self.samples = []  # type: List[Dict[str, Any]]
        self._load_data()
        # Data filtering based on mode
        if self.mode == 'test':
            # For test, load full data
            pass
        elif self.mode == 'finetuning':
            # For finetuning, load relevant samples
            pass
        elif self.mode == 'pretraining':
            # For pretraining, focus on protein-text pairs
            pass
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _load_data(self):
        """
        Load data from file into samples list.
        Supports JSONL or CSV based on extension.
        Each sample is a dict with keys depending on dataset.
        """
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        ext = os.path.splitext(self.data_path)[1].lower()
        raw_entries = []
        if ext == '.jsonl':
            with open(self.data_path, 'r') as f:
                for line in f:
                    raw_entries.append(json.loads(line))
        elif ext == '.csv':
            with open(self.data_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_entries.append(row)
        else:
            raise ValueError(f"Unsupported data file extension: {ext}")

        # Parse each raw entry
        for raw in raw_entries:
            sample = self._preprocess_sample(raw)
            # For modes, we can filter or include all samples, depending on mode
            # For simplicity, include all; filtering can be done externally or here
            self.samples.append(sample)

        # Optionally, apply filtering based on mode within _preprocess_sample
        # or here if needed.

    def _preprocess_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw data into model-ready format.
        Includes tokenization, enrichment, data augmentation.
        """
        # Parsing raw fields
        protein_seq = raw_sample.get('protein_sequence', '')
        mutation_info = raw_sample.get('mutation_info', {})  # dict: position, original_aa, mutated_aa
        mut_effect_text = raw_sample.get('mutational_effect_text', '')  # textual description
        literature_refs = raw_sample.get('literature_refs', [])
        species = raw_sample.get('species', '')
        seq_length = int(raw_sample.get('sequence_length', len(protein_seq)))

        # Tokenize protein sequence
        prot_encoding = tokenize_protein(protein_seq, tokenizer_sequence)
        sequence_ids = prot_encoding['input_ids'].squeeze(0)  # shape: [seq_len]
        attention_mask = prot_encoding['attention_mask'].squeeze(0)

        # Prepare textual annotation
        text_raw = mut_effect_text

        # Data Enrichment: if text is too short or low quality, expand
        text_enriched = self._enrich_text(text_raw, literature_refs)

        # Tokenize textual annotation
        text_encoding = tokenize_text(text_enriched, tokenizer_text)
        text_ids = text_encoding['input_ids'].squeeze(0)  # shape: [text_seq_len]
        text_attention_mask = text_encoding['attention_mask'].squeeze(0)

        # Mutation labels (for finetuning and test), optional in pretraining
        mutation_labels = None
        if self.mode in ['finetuning', 'test']:
            position = mutation_info.get('position', -1)
            original_aa = mutation_info.get('original_aa', '')
            mutated_aa = mutation_info.get('mutated_aa', '')
            mutation_labels = {
                'position': position,
                'original_aa': original_aa,
                'mutated_aa': mutated_aa
            }

        # Prepare sample dict
        sample = {
            'protein_sequence': protein_seq,
            'sequence_ids': sequence_ids,
            'attention_mask': attention_mask,
            'text': text_enriched,
            'text_ids': text_ids,
            'text_attention_mask': text_attention_mask,
            'mutation_labels': mutation_labels,
            'species': species,
            'sequence_length': seq_length,
            'literature_refs': literature_refs
        }

        # For modes involving reverse (benign/malignant), generate reversed samples
        if self.mode == 'finetuning' or self.mode == 'test':
            # Assume mutation label exists
            if mutation_labels:
                reversed_sample = self._create_reverse_sample(sample)
                return reversed_sample

        return sample

    def _enrich_text(self, text: str, literature_refs: List[str]) -> str:
        """
        Enrich low-quality or short textual annotations using literature abstracts.
        """
        MIN_WORD_COUNT = 10  # arbitrary threshold
        words = text.strip().split()
        if len(words) >= MIN_WORD_COUNT:
            return text
        else:
            # Use literature references to fetch abstracts
            # Use GPT or placeholder to expand
            enriched_text = gpt_text_enrichment(
                prompt=f"Expand the following mutation description: {text}",
                literature_refs=literature_refs
            )
            # Fallback if GPT API fails, return original
            if not enriched_text:
                return text
            return enriched_text

    def _create_reverse_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a reversed sample swapping wild-type and mutant info, for data balancing.
        """
        seq = sample['protein_sequence']
        labels = sample['mutation_labels']
        if labels is None:
            return sample  # no labels to reverse

        pos = labels['position']
        orig_a = labels['original_aa']
        mut_a = labels['mutated_aa']

        # Reverse sequence: swap original and mutated amino acids at position
        seq_list = list(seq)
        if 0 <= pos - 1 < len(seq_list):
            seq_list[pos - 1] = orig_a  # set back to original amino acid
        reversed_seq = ''.join(seq_list)

        # Create new mutation info
        reversed_labels = {
            'position': pos,
            'original_aa': mut_a,
            'mutated_aa': orig_a
        }

        # Generate new mutational effect text by prompting, or simply swap descriptions
        # For simplicity, assume textual description can be generated similarly
        reversed_text = self._enrich_text(
            f"Reversal of mutation: {orig_a} at {pos} to {mut_a}",
            literature_refs=sample.get('literature_refs', [])
        )

        # Tokenize reversed sequence and text
        prot_encoding = tokenize_protein(reversed_seq, tokenizer_sequence)
        sequence_ids = prot_encoding['input_ids'].squeeze(0)
        attention_mask = prot_encoding['attention_mask'].squeeze(0)
        text_encoding = tokenize_text(reversed_text, tokenizer_text)
        text_ids = text_encoding['input_ids'].squeeze(0)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0)

        reversed_sample = {
            'protein_sequence': reversed_seq,
            'sequence_ids': sequence_ids,
            'attention_mask': attention_mask,
            'text': reversed_text,
            'text_ids': text_ids,
            'text_attention_mask': text_attention_mask,
            'mutation_labels': reversed_labels,
            'species': sample.get('species', ''),
            'sequence_length': len(reversed_seq),
            'literature_refs': sample.get('literature_refs', [])
        }
        return reversed_sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Return the processed sample at index idx.
        """
        sample = self.samples[idx]
        return {
            'sequence_ids': sample['sequence_ids'],
            'attention_mask': sample['attention_mask'],
            'text_ids': sample['text_ids'],
            'text_attention_mask': sample['text_attention_mask'],
            'mutation_labels': sample['mutation_labels'],
            'species': sample['species']
        }
