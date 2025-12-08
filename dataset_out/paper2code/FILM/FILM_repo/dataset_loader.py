## dataset_loader.py

import os
import json
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
import random
from transformers import PreTrainedTokenizer
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Extract core config parameters with defaults
MODEL_NAME = CONFIG['model'].get('name', 'mistral-7b-instruct-v0.2')
ROPE_BASE = float(CONFIG['model'].get('rope_base', 1e6))
MAX_SEQ_LENGTH = 4096  # Standard max sequence length (can be model-specific)
LONG_CONTEXT_MIN = CONFIG['long_context'].get('min_length', 4000)
LONG_CONTEXT_MAX = CONFIG['long_context'].get('max_length', 32000)
LENGTH_DISTRIBUTION = CONFIG['long_context'].get('length_distribution', [4000, 8000, 16000, 32000])
BATCH_SIZE = CONFIG['dataset'].get('batch_size', 128)

# Sliding window configs
USE_SLIDING_WINDOW = CONFIG['evaluation'].get('use_sliding_window', True)
SLIDING_WINDOW_SIZE = CONFIG['evaluation'].get('sliding_window_size', 4096)

# Dataset file path (assumed to be pre-generated)
DATASET_PATH = 'full_dataset.json'  # Or any preferred dataset file

# Initialize tokenizer
from transformers import AutoTokenizer
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class QA_LongContextDataset(Dataset):
    """
    PyTorch Dataset for synthetic long-context QA data, supporting sliding window sampling.
    Each item yields input tensors suitable for model training.
    """
    def __init__(self, data: List[Dict], max_seq_length: int=MAX_SEQ_LENGTH, use_sliding: bool=USE_SLIDING_WINDOW, window_size: int=SLIDING_WINDOW_SIZE):
        """
        Args:
            data: list of dict with 'context', 'question', 'answer', optional metadata
            max_seq_length: maximum tokens per input (model-specific)
            use_sliding: whether to apply sliding window on long contexts
            window_size: size of sliding window in tokens
        """
        self.data = data
        self.max_seq_length = max_seq_length
        self.use_sliding = use_sliding
        self.window_size = window_size

        # Prepare a list of all expanded samples for sliding window sampling if needed
        self.sample_indices = []  # Will hold tuples: (data_idx, start_token_idx, end_token_idx)
        self.expanded_samples = []  # Will hold dicts with 'prompt', 'labels', etc.

        self._preprocess()

    def _preprocess(self):
        """
        For each data item, if context longer than max_seq_length, generate multiple windowed samples.
        Else, store as single sample.
        """
        for idx, item in enumerate(self.data):
            context_text = item['context']
            question_text = item['question']
            answer_text = item['answer']

            # Tokenize entire context once to get token length
            context_enc = tokenizer(context_text, add_special_tokens=False)
            ctx_token_ids = context_enc['input_ids']
            ctx_length = len(ctx_token_ids)

            # Determine number of windows needed
            if ctx_length <= self.max_seq_length:
                # Single sample
                sample_dict = self._build_input_example(item, context_text)
                self.expanded_samples.append(sample_dict)
            else:
                # Long context - create sliding window samples
                if self.use_sliding:
                    stride = int(self.window_size * 0.5)  # 50% overlap for robustness
                    start_idx = 0
                    while start_idx < ctx_length:
                        end_idx = start_idx + self.window_size
                        if end_idx > ctx_length:
                            end_idx = ctx_length
                        # Decode the window token IDs to string (or store IDs)
                        input_ids_window = ctx_token_ids[start_idx:end_idx]
                        # Reconstruct partial context string
                        context_window_text = tokenizer.decode(input_ids_window, clean_up_tokenization_spaces=True)
                        # Build input example
                        example = self._build_input_example(item, context_window_text, is_window=True,
                                                              start_idx=start_idx, end_idx=end_idx)
                        self.expanded_samples.append(example)
                        if end_idx == ctx_length:
                            break
                        start_idx += stride
                else:
                    # If not using sliding window, truncate or skip long contexts
                    input_ids_trunc = ctx_token_ids[:self.max_seq_length]
                    context_trunc_text = tokenizer.decode(input_ids_trunc, clean_up_tokenization_spaces=True)
                    example = self._build_input_example(item, context_trunc_text)
                    self.expanded_samples.append(example)

    def _build_input_example(self, item: Dict, context_str: str, is_window: bool=False, start_idx:int=0, end_idx:int=0) -> Dict:
        """
        Tokenize prompt + context + question, prepare labels (answer).
        """
        question_text = item['question']
        answer_text = item['answer']
        # Compose prompt; rely on model's instruction style if needed
        prompt = f"Context:\n{context_str}\nQuestion: {question_text}\nAnswer:"
        encoding = tokenizer(
            prompt,
            max_length=None,
            truncation=False,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        # For training, labels are usually set to the answer tokens following the prompt.
        # Here, for simplicity, set labels to the entire sequence (causal LM training).
        labels = input_ids.clone()

        # Optional: Mask tokens outside answer span if wanted; here, we keep full
        # For better instruction tuning, compute labels so that only answer tokens are used for loss if fine-tuning

        # Prepare metadata
        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'context_length': len(encoding['input_ids'][0]),
            'raw_context': context_str,
            'question': question_text,
            'answer': answer_text,
            'ctx_start_idx': start_idx if is_window else -1,
            'ctx_end_idx': end_idx if is_window else -1,
            'original_item_idx': id(item),
        }
        return sample

    def __len__(self):
        return len(self.expanded_samples)

    def __getitem__(self, index):
        return self.expanded_samples[index]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to pad batch samples into uniform tensors.
    """
    input_ids_list = [sample['input_ids'] for sample in batch]
    attention_mask_list = [sample['attention_mask'] for sample in batch]
    labels_list = [sample['labels'] for sample in batch]
    # Find max length in batch
    max_len = max([len(ids) for ids in input_ids_list])
    input_ids_padded = []
    attention_mask_padded = []
    labels_padded = []

    for ids, mask, lbl in zip(input_ids_list, attention_mask_list, labels_list):
        pad_len = max_len - len(ids)
        input_ids_padded.append(torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id)]))
        attention_mask_padded.append(torch.cat([mask, torch.zeros(pad_len)]))
        lbl_pad = torch.full((max_len,), -100)  # ignore index in loss
        lbl_pad[:len(lbl)] = lbl
        labels_padded.append(lbl_pad)

    batch_input_ids = torch.stack(input_ids_padded)
    batch_attention_mask = torch.stack(attention_mask_padded)
    batch_labels = torch.stack(labels_padded)

    return {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'labels': batch_labels
    }

def load_dataset_from_json(filepath: str) -> List[Dict]:
    """
    Load dataset JSON lines or list of dicts.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Usage example:
def get_dataloader(batch_size: int=BATCH_SIZE):
    # Load dataset
    dataset_list = load_dataset_from_json(DATASET_PATH)
    dataset = QA_LongContextDataset(dataset_list)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return dataloader

# For debugging or evaluation
if __name__ == "__main__":
    dataloader = get_dataloader()
    for batch in dataloader:
        print(batch['input_ids'].shape)
        print(batch['labels'].shape)
        break  # remove break for full iteration

