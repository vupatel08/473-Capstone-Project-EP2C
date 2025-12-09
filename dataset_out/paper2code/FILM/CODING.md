# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_generator.py

```python
## dataset_generator.py

import os
import json
import random
import time
from typing import List, Dict, Tuple, Optional
import openai
import numpy as np
from tqdm import tqdm
import re

from utils import (
    generate_prompt,
    assemble_context,
    tokenize_and_process,
    save_dataset,
    load_dataset,
    format_prompt_template,
    normalize_text,
    calculate_context_length,
)
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Configuration parameters
TOTAL_DATA_SIZE = int(CONFIG['training'].get('dataset_size', 1100000))
BATCH_SIZE = int(CONFIG['training'].get('batch_size', 128))
TOTAL_STEPS = int(CONFIG['training'].get('steps_per_epoch', 14000))
LONG_CONTEXT_MIN = int(CONFIG['long_context'].get('min_length', 4000))
LONG_CONTEXT_MAX = int(CONFIG['long_context'].get('max_length', 32000))
LENGTH_DIST = CONFIG['long_context'].get('length_distribution', [4000, 8000, 16000, 32000])
MODEL_NAME = CONFIG['model'].get('name', 'mistral-7b-instruct-v0.2')
ROPE_BASE_DEFAULT = float(CONFIG['model'].get('rope_base', 1e6))
USE_SLIDING_WINDOW = CONFIG['evaluation'].get('use_sliding_window', True)
SLIDING_WINDOW_SIZE = int(CONFIG['evaluation'].get('sliding_window_size', 4096))
API_TEMPERATURE = CONFIG['generation'].get('temperature', 0.7)
API_TOP_P = CONFIG['generation'].get('top_p', 0.95)

# OpenAI API key
API_KEY = os.getenv('OPENAI_API_KEY', None)
if API_KEY is None:
    raise ValueError("Please set OPENAI_API_KEY environment variable.")

# Helper: Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load tokenizer (assuming tokenization is consistent across utils.py and model)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Data source: Assume we have a large corpus of raw texts (strings).
# For simplicity, here we define a placeholder function to get raw texts.
# In practice, you should replace `get_raw_texts()` with your actual corpus loader.

def get_raw_texts() -> List[str]:
    """
    Placeholder: Load your large corpus here.
    For example, read texts from files, datasets, or preprocessed data.
    """
    # For demonstration, returning a small list
    return [
        ("This is a sample document about climate change. It discusses impacts and mitigation strategies." * 50),
        ("Detailed scientific findings on quantum computing. Includes algorithms and hardware details." * 40),
        # ... load actual large corpus in practical usage
    ]

# Optional: filter texts to avoid overlaps with evaluation sets
def filter_texts(texts: List[str], eval_overlap_set: set, min_overlap_grams=10) -> List[str]:
    """
    Filter texts to avoid overlap with evaluation datasets.
    """
    filtered_texts = []
    for txt in texts:
        # crude overlap check: count overlapping 10-grams
        grams = set(re.findall(r'\b\w+\b', txt))
        overlap = grams.intersection(eval_overlap_set)
        if len(overlap) < min_overlap_grams:
            filtered_texts.append(txt)
    return filtered_texts

# --- Main class for dataset generation
class DatasetGenerator:
    def __init__(self,
                 raw_texts: List[str],
                 max_examples: int = TOTAL_DATA_SIZE,
                 batch_size: int = BATCH_SIZE,
                 context_lengths: List[int] = LENGTH_DIST,
                 min_len: int = LONG_CONTEXT_MIN,
                 max_len: int = LONG_CONTEXT_MAX,
                 api_temperature: float = API_TEMPERATURE,
                 top_p: float = API_TOP_P,
                 use_sliding_window: bool = USE_SLIDING_WINDOW,
                 sliding_window_size: int = SLIDING_WINDOW_SIZE,
                 model_name: str = MODEL_NAME,
                 rope_base: float = ROPE_BASE_DEFAULT,
                 instruction_templates: Dict[str, str] = None,
                 multi_hop_ratio: float = 0.7,  # ratio of multi-hop QA samples
                 fine_grained_ratio: float = 0.3  # ratio of fine-grained
                ):
        self.raw_texts = raw_texts
        self.max_examples = max_examples
        self.batch_size = batch_size
        self.context_lengths = context_lengths
        self.min_len = min_len
        self.max_len = max_len
        self.api_temperature = api_temperature
        self.top_p = top_p
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.model_name = model_name
        self.rope_base = rope_base
        # instruction templates for prompt generation
        self.instruction_templates = instruction_templates or {
            'fine_grained': "Given a short segment of text, generate a highly specific question and its answer based solely on that segment: {segment}",
            'multi_hop': "Given multiple segments, produce a question that requires integrating and reasoning across these segments: {segments}. Provide the answer as well.",
        }
        # ratios for QA types
        self.multi_hop_ratio = multi_hop_ratio
        self.fine_grained_ratio = fine_grained_ratio

        # Placeholder for evaluation set overlap filtering
        self.eval_overlap_set = set()  # Should be populated with overlapping n-grams in practice

    def get_random_raw_text(self) -> str:
        return random.choice(self.raw_texts)

    def sample_length(self) -> int:
        return random.choice(self.context_lengths)

    def get_segments_from_text(self, text: str) -> List[str]:
        """
        Segment a raw text into 128-token chunks, using Algorithm 1.
        """
        tokens = tokenizer.tokenize(text)
        segments = []
        i = 0
        while i < len(tokens):
            chunk = tokens[i: i + 128]
            seg_text = tokenizer.convert_tokens_to_string(chunk)
            segments.append(seg_text)
            i += 128  # step size
        return segments

    def build_long_context(self, segments: List[str], target_token_count: int) -> str:
        """
        Assemble a long context from randomly shuffled segments to match target length.
        """
        # Shuffle segments
        segs = segments.copy()
        random.shuffle(segs)
        context_tokens = []
        for seg in segs:
            token_ids = tokenizer.encode(seg, add_special_tokens=False)
            if len(context_tokens) + len(token_ids) > target_token_count:
                break
            context_tokens.extend(token_ids)
        # Convert back to string
        context_text = tokenizer.decode(context_tokens, clean_up_tokenization_spaces=True)
        return context_text

    def generate_qa_for_segment(self, segment: str, instruction_type: str='fine_grained') -> Tuple[str, str]:
        """
        Generate QA pair from a single segment using GPT-4 API.
        """
        prompt = generate_prompt(
            self.instruction_templates[instruction_type],
            segment,
            instruction_type
        )
        # Call GPT-4 API
        response = self.call_gpt_api(prompt)
        question, answer = self.parse_qa_response(response)
        return question, answer

    def generate_qa_for_segments(self, segments: List[str]) -> Tuple[str, str]:
        """
        Generate QA that requires multiple segments (multi-hop).
        """
        segments_str = " | ".join(segments)
        prompt = generate_prompt(
            self.instruction_templates['multi_hop'],
            segments_str,
            'multi_hop'
        )
        response = self.call_gpt_api(prompt)
        question, answer = self.parse_qa_response(response)
        return question, answer

    def call_gpt_api(self, prompt: str) -> str:
        """
        Send prompt to GPT-4 API with retries and rate limiting handling.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[
                        {"role": "system", "content": "You are an assistant generating QA pairs for training."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.api_temperature,
                    top_p=self.top_p,
                    max_tokens=512,
                    api_key=API_KEY
                )
                return response.choices[0].message['content']
            except Exception as e:
                print(f"API call error: {e}")
                time.sleep(2 ** attempt)  # exponential backoff
        raise RuntimeError("GPT API call failed after retries.")

    def parse_qa_response(self, response_text: str) -> Tuple[str, str]:
        """
        Parse the GPT response to extract question and answer.
        Expect format: Q: ... \nA: ...
        """
        q_match = re.search(r'Q[:\s]*(.+)', response_text, re.IGNORECASE)
        a_match = re.search(r'A[:\s]*(.+)', response_text, re.IGNORECASE)
        question = q_match.group(1).strip() if q_match else "Question"
        answer = a_match.group(1).strip() if a_match else "Answer"
        return question, answer

    def generate_example(self, example_idx: int) -> Dict:
        """
        Generate a single dataset example: long context + QA pair.
        """
        # Sample raw text
        text = self.get_random_raw_text()

        # Segment the raw text
        segments = self.get_segments_from_text(text)
        total_tokens = calculate_context_length(tokenizer.encode(text))
        # Determine target length for long context
        target_length = self.sample_length()

        # Assemble long context
        long_context = assemble_context(segments, target_length)

        # Decide QA type: fine-grained or multi-hop based on ratio
        if random.random() < self.fine_grained_ratio:
            qa_type = 'fine_grained'
        else:
            qa_type = 'multi_hop'

        # Generate QA pair
        if qa_type == 'fine_grained':
            # Pick one segment to base question
            selected_seg = random.choice(segments)
            question, answer = self.generate_qa_for_segment(selected_seg, 'fine_grained')
        else:
            # For multi-hop, select 2-3 segments
            num_segments_for_q = random.randint(2, min(4, len(segments)))
            selected_segments = random.sample(segments, num_segments_for_q)
            question, answer = self.generate_qa_for_segments(selected_segments)

        # Build the sample dict
        sample = {
            'context': long_context,
            'question': question,
            'answer': answer,
            'context_length': calculate_context_length(tokenizer.encode(long_context)),
            'qa_type': qa_type,
            'segments_used': selected_seg if qa_type=='fine_grained' else selected_segments
        }
        return sample

    def generate_dataset(self) -> List[Dict]:
        """
        Generate the entire dataset up to max_examples samples.
        """
        dataset = []
        print("Starting dataset generation...")
        for i in tqdm(range(self.max_examples)):
            try:
                example = self.generate_example(i)
                dataset.append(example)
                # Optional: periodic saving per batch
                if (i+1) % 10000 == 0:
                    save_dataset(dataset, f'dataset_part_{i+1}.json')
            except Exception as e:
                print(f"Error generating example {i}: {e}")
                continue
        print("Finished dataset generation.")
        return dataset

# --- Main execution
def main():
    print("Loading raw texts...")
    raw_texts = get_raw_texts()

    print("Initializing dataset generator...")
    generator = DatasetGenerator(raw_texts=raw_texts)

    dataset = generator.generate_dataset()

    print("Saving full dataset to 'full_dataset.json'...")
    save_dataset(dataset, 'full_dataset.json')
    print("Dataset saved successfully.")

if __name__ == '__main__':
    main()
```

## dataset_loader.py

```python
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

```

## evaluation.py

```python
## evaluation.py
"""
Evaluation module for assessing long-context utilization of the trained language model.
Implements retrieval probing, scaling evaluation with sliding window, and real-world long
document tasks, based on configurations from 'config.yaml'.
"""

import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from utils import (
    load_dataset_from_json,
    generate_prompt,
    normalize_text,
    tokenize_and_process,
    format_prompt_template
)
import yaml

# Load configurations from 'config.yaml'
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Extract config parameters with defaults
MAX_SEQ_LENGTH = 4096  # Model-supported max sequence length
LONG_CONTEXT_LENGTHS = CONFIG.get('long_context', {}).get('length_distribution', [4000, 8000, 16000, 32000])
USE_SLIDING_WINDOW = CONFIG.get('evaluation', {}).get('use_sliding_window', True)
SLIDING_WINDOW_SIZE = CONFIG.get('evaluation', {}).get('sliding_window_size', 4096)
MODEL_NAME = CONFIG.get('model', {}).get('name', 'mistral-7b-instruct-v0.2')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_TASKS = CONFIG.get('evaluation', {}).get('tasks', {})
# For simplicity, evaluate all tasks by default
EVAL_RETRIEVAL = EVAL_TASKS.get('retrieval', True)
EVAL_SCALING = EVAL_TASKS.get('scaling', True)
EVAL_FEWSHOT = EVAL_TASKS.get('few_shot', True)

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Load tokenizer for decoding generated tokens
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Utility functions for metrics
def compute_exact_match(pred: str, ref: str) -> int:
    return int(normalize_text(pred) == normalize_text(ref))

def compute_recall(pred_tokens: list, target_tokens: set) -> float:
    """
    Word-level recall: fraction of target tokens appearing in prediction.
    """
    pred_token_set = set(pred_tokens)
    recall = len(target_tokens.intersection(pred_token_set)) / max(1, len(target_tokens))
    return recall

def extract_answer_from_response(response: str, answer_type='qa') -> str:
    """
    Extract answer text from model response, expecting formats like 'Q: ... A: ...'.
    """
    match = re.search(r'([Qq][:\s\S]+?)(?:\n|$)', response)
    question = ''
    answer = ''
    q_match = re.search(r'Q[:\s]*(.+)', response, re.IGNORECASE)
    a_match = re.search(r'A[:\s]*(.+)', response, re.IGNORECASE)
    if q_match:
        question = q_match.group(1).strip()
    if a_match:
        answer = a_match.group(1).strip()
    else:
        # fallback
        answer = response.strip()
    return answer

# Function to perform inference with optional sliding window
def generate_response(input_ids, attention_mask, max_new_tokens=512, do_sample=False, temperature=1.0, top_p=0.9):
    seq_len = input_ids.shape[1]
    if seq_len <= MAX_SEQ_LENGTH:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p
        )
        return outputs
    else:
        # Long sequence: sliding window approach
        return _long_sequence_generate(input_ids, attention_mask, max_new_tokens, do_sample, temperature, top_p)

def _long_sequence_generate(input_ids, attention_mask, max_new_tokens, do_sample, temperature, top_p):
    generated = input_ids
    attn_mask = attention_mask
    for _ in range(max_new_tokens):
        input_slice = generated[:, -MAX_SEQ_LENGTH:]
        attn_slice = attn_mask[:, -MAX_SEQ_LENGTH:]
        outputs = model.generate(
            input_ids=input_slice,
            attention_mask=attn_slice,
            max_new_tokens=1,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p
        )
        next_token = outputs[:, -1:].to(generated.device)
        generated = torch.cat([generated, next_token], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(next_token, dtype=torch.long).to(generated.device)], dim=1)
    return generated

# Load dataset (assumed processed and stored in JSON)
def load_eval_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Evaluation functions
def evaluate_retrieval(dataset, style='document', pattern='bi', position='middle'):
    """
    Evaluate retrieval probing:
        - style: 'document', 'code', 'structured'
        - pattern: 'forward', 'backward', 'bi'
        - position: 'start', 'middle', 'end'
    """
    metrics_results = {}
    total = 0
    correct = 0
    recall_scores = []
    for sample in tqdm(dataset, desc=f"Eval {style}-{pattern}-{position}"):
        context = sample['context']
        question = sample['question']
        answer = sample['answer']
        # Prepare prompt
        prompt = generate_prompt(
            template=format_prompt_template('qa'),
            segment=context,
            instruction_type='qa'
        )
        input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(DEVICE)
        attention_mask = torch.ones_like(input_ids)
        # Generate response
        output_ids = generate_response(input_ids, attention_mask)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # For retrieval, check if answer tokens are present and compute recall
        pred_answer = extract_answer_from_response(output_text)
        total += 1
        # Pointer to relevant answer tokens
        target_tokens = set(normalize_text(answer).split())
        pred_tokens = normalize_text(pred_answer).split()
        recall = compute_recall(pred_tokens, target_tokens)
        recall_scores.append(recall)
        if normalize_text(answer) == normalize_text(pred_answer):
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    recall_mean = np.mean(recall_scores) if recall_scores else 0.0
    return {'accuracy': accuracy, 'recall': recall_mean}

def evaluate_scaling(model, dataset, lengths=LONG_CONTEXT_LENGTHS):
    """
    Evaluate model performance at varying context lengths, utilizing sliding windows if necessary.
    """
    length_performance = {}
    for length in lengths:
        position = 'middle'  # Can vary position if data is segmented
        subset = [sample for sample in dataset if sample.get('context_length', 0) >= length*0.9]
        if not subset:
            continue
        perf = evaluate_retrieval(subset, style='document', pattern='bi', position=position)
        length_performance[length] = perf
    return length_performance

def evaluate_realworld_tasks():
    """
    Evaluate model on real-world long document tasks (e.g., NarrativeQA, Qasper, etc.).
    Assumed dataset per task is prepared similarly as in training, with prompt formatting.
    """
    tasks = {
        'NarrativeQA': 'Evaluate narrative question answering with long documents.',
        'Qasper': 'Evaluate QA on scientific datasets.',
        'MultiFQA': 'Evaluate multi-document QA.',
        'HotpotQA': 'Multihop QA.',
        '2WikiMQA': 'Multi-hop QA with wiki data.',
        'MuSiQue': 'Large-scale QA benchmark.',
        'GovReport': 'Summarization of long reports.',
        'QMSum': 'Long meeting summarization.',
        'MultiNews': 'Multinews summarization.'
    }
    results = {}
    for task_name, description in tasks.items():
        dataset_path = f'data/{task_name}.json'  # assume dataset files
        data = load_eval_dataset(dataset_path)
        # Depending on task, choose ops (classification, QA, summarization)
        # For simplicity, assume QA style
        scores = []
        for sample in tqdm(data, desc=f"{task_name}"):
            prompt = generate_prompt(
                template=format_prompt_template('qa'),
                segment=sample['context'],
                instruction_type='qa'
            )
            input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(DEVICE)
            attention_mask = torch.ones_like(input_ids)
            output_ids = generate_response(input_ids, attention_mask)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred_answer = extract_answer_from_response(output_text)
            ref_answer = sample['answer']
            # For F1 or accuracy, implement as needed. Here, example with exact match
            score = compute_exact_match(pred_answer, ref_answer)
            scores.append(score)
        mean_score = np.mean(scores) if scores else 0.0
        results[task_name] = mean_score
    return results

def evaluate_short_context_tasks():
    """
    Evaluate on standard short-context tasks: MMLU, BoolQ, RACE, etc.
    Assumes datasets are formatted and scripts are available.
    """
    # Placeholder: Implementation depends on specific evaluation scripts
    # For demo, we assume all are loaded and evaluated similarly
    short_tasks = ['MMLU', 'BoolQ', 'RACE', 'CommonsenseQA', 'ARC']
    results = {}
    for task in short_tasks:
        dataset_path = f'data/{task}.json'
        data = load_eval_dataset(dataset_path)
        scores = []
        for sample in tqdm(data, desc=task):
            prompt = generate_prompt(
                template=format_prompt_template('qa'),
                segment=sample['context'],
                instruction_type='qa'
            )
            input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(DEVICE)
            attention_mask = torch.ones_like(input_ids)
            output_ids = generate_response(input_ids, attention_mask)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred_answer = extract_answer_from_response(output_text)
            ref_answer = sample['answer']
            scores.append(compute_exact_match(pred_answer, ref_answer))
        results[task] = np.mean(scores)
    return results

# Main evaluation orchestrator
def main():
    dataset_path = 'full_dataset.json'
    dataset = load_eval_dataset(dataset_path)

    # Retrieval probing at various positions (e.g., start, middle, end)
    positions = ['start', 'middle', 'end']
    styles = ['document', 'code', 'structured']
    pattern = 'bi'  # bi-directional pattern

    results = {}
    if EVAL_RETRIEVAL:
        for style in styles:
            for pos in positions:
                res = evaluate_retrieval(dataset, style=style, pattern=pattern, position=pos)
                results[f'{style}-{pos}'] = res

    # Evaluation over context lengths (scaling)
    if EVAL_SCALING:
        length_perf = evaluate_scaling(model, dataset)
        results['scaling'] = length_perf

    # Evaluation on real-world long document tasks
    if EVAL_SCALING:
        real_task_results = evaluate_realworld_tasks()
        results['real_world_tasks'] = real_task_results

    # Short-context tasks
    if EVAL_FEWSHOT:
        short_results = evaluate_short_context_tasks()
        results['short_context'] = short_results

    # Save or print results
    output_path = 'evaluation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
```

## main.py

```python
## main.py
"""
Main script orchestrating the training and evaluation pipeline for REPRODUCING
the "Make Your LLM Fully Utilize the Context" (IN2) approach as described.
It sequentially:
- Loads configuration
- Generates synthetic long-context QA dataset
- Loads and possibly extends the model (RoPE, position embeddings)
- Fine-tunes the model on the dataset
- Performs probing and scaling evaluation
- Reports metrics and saves models
"""

import os
import sys
import yaml
import torch
from tqdm import tqdm

# Import custom modules
import utils
import dataset_generator
import dataset_loader
import model
import trainer
import evaluation

def main():
    # 1. Load configuration
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Extract core configs with defaults and strongly typed
    model_name = cfg.get('model', {}).get('name', 'mistral-7b-instruct-v0.2')
    rope_base = float(cfg.get('model', {}).get('rope_base', 1e6))
    max_position_embeddings = cfg.get('model', {}).get('max_position_embeddings', 0)
    use_sliding_window = cfg.get('evaluation', {}).get('use_sliding_window', True)
    sliding_window_size = int(cfg.get('evaluation', {}).get('sliding_window_size', 4096))
    training_cfg = cfg.get('training', {})
    dataset_cfg = cfg.get('dataset', {})
    long_context_cfg = cfg.get('long_context', {})
    generation_cfg = cfg.get('generation', {})
    eval_cfg = cfg.get('evaluation', {})

    learning_rate = float(training_cfg.get('learning_rate', 1e-6))
    batch_size = int(training_cfg.get('batch_size', 128))
    epochs = int(training_cfg.get('epochs', 1))
    total_steps = int(training_cfg.get('steps_per_epoch', 14000))
    warmup_ratio = float(training_cfg.get('warmup_steps', 0.03))
    warmup_steps = int(warmup_ratio * total_steps)

    dataset_size = int(dataset_cfg.get('size', 1_100_000))
    context_lengths = long_context_cfg.get('length_distribution', [4000, 8000, 16000, 32000])
    output_dir = cfg.get('output_dir', 'outputs')

    # 2. Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Generate dataset
    print("Loading raw texts for dataset generation...")
    raw_texts = utils.get_raw_texts()
    
    print("Initializing dataset generator...")
    gen = dataset_generator.DatasetGenerator(
        raw_texts=raw_texts,
        max_examples=dataset_size,
        batch_size=batch_size,
        context_lengths=context_lengths,
        model_name=model_name
    )
    print("Generating synthetic long-context QA dataset...")
    dataset = gen.generate_dataset()
    # Save the full dataset for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    full_dataset_path = os.path.join(output_dir, 'full_dataset.json')
    utils.save_dataset(dataset, full_dataset_path)
    print(f"Dataset saved to {full_dataset_path} with {len(dataset)} samples.")

    # 4. Initialize model
    print("Loading model...")
    # For simplicity, assume the model is a LongContextModel supporting extension
    model_obj = model.LongContextModel(
        model_name=model_name,
        rope_base=rope_base,
        extend_positional=(max_position_embeddings > 0),
        max_position_embeddings=max_position_embeddings,
    )
    # 5. Fine-tune model
    print("Loading dataset for training...")
    train_dataset = utils.load_dataset_from_json(full_dataset_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn,
    )
    print("Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model_obj.model.parameters(), lr=learning_rate)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup=int(0.03 * total_steps),
        num_training_steps=total_steps
    )

    print("Starting training...")
    model_obj.model.train()
    global_step = 0
    for epoch in range(epochs):
        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in epoch_iter:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to(input_ids.device)
            labels = batch['labels'].to(input_ids.device)
            outputs = model_obj.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            # gradient clip if desired
            torch.nn.utils.clip_grad_norm_(model_obj.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}/{total_steps}: loss={loss.item():.4f}")

            if global_step % 5000 == 0:
                ckpt_path = os.path.join(output_dir, f'checkpoint_step_{global_step}')
                os.makedirs(ckpt_path, exist_ok=True)
                model_obj.save_model(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break
    print("Training completed. Saving final model...")
    final_path = os.path.join(output_dir, 'final_model')
    model_obj.save_model(final_path)
    print(f"Model saved to {final_path}")

    # 6. Load model for evaluation
    print("Loading trained model for evaluation...")
    eval_model = model.LongContextModel(
        model_name=final_path,
        rope_base=rope_base,
        extend_positional=(max_position_embeddings > 0),
        max_position_embeddings=max_position_embeddings,
    )
    eval_model.model.eval()

    # 7. Run probing and evaluation
    print("Loading dataset for evaluation...")
    dataset_eval = utils.load_dataset_from_json(full_dataset_path)

    print("Starting probing tasks evaluation...")
    probing_results = evaluation.probe_long_context(eval_model, dataset_eval)

    print("Running scaling evaluation...")
    scaling_results = evaluation.scale_long_context(eval_model, dataset_eval)

    print("Evaluating real-world long document tasks...")
    real_tasks_results = evaluation.evaluate_real_world_tasks(eval_model)

    print("Evaluating short-context tasks...")
    short_results = evaluation.evaluate_short_tasks(eval_model)

    # 8. Save results
    results = {
        "probing": probing_results,
        "scaling": scaling_results,
        "real_tasks": real_tasks_results,
        "short_tasks": short_results,
    }
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
""" 
Model class supporting loading, extending position embeddings via RoPE scaling,
long-context inference (including sliding window), and model saving/loading.
Designed for compatibility with Hugging Face transformers and supporting
long-context training as described in the IN2 paper.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional
import os

# Load configuration from global config.yaml via utils.py (assumed loaded statically)
import yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

MODEL_NAME = CONFIG['model'].get('name', 'mistral-7b-instruct-v0.2')
ROPE_BASE = float(CONFIG['model'].get('rope_base', 1e6))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LongContextModel:
    """
    Wrapper class for large language models supporting:
    - Loading models with optional adapters
    - Extending positional embeddings for long contexts via RoPE scaling
    - Long context inference with sliding window
    - Saving and loading fine-tuned models
    """
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        rope_base: float = ROPE_BASE,
        device: torch.device = DEVICE,
        load_checkpoint_path: Optional[str]=None,
        load_adapter_path: Optional[str]=None,
        extend_positional: bool=False,
        max_position_embeddings: int=0,  # 0 means no extension
        verbose: bool=True
    ):
        """
        Args:
            model_name (str): Pretrained model name or path.
            rope_base (float): RoPE (rotary positional embedding) base (theta).
            device (torch.device): Device to run the model on.
            load_checkpoint_path (str): Optional path to a checkpoint to load.
            load_adapter_path (str): Optional path to PEFT adapter to load.
            extend_positional (bool): Whether to extend positional embeddings for longer contexts.
            max_position_embeddings (int): If >0, extension length to be added.
            verbose (bool): Whether to print detailed info.
        """
        self.model_name = model_name
        self.rope_base = rope_base
        self.device = device
        self.verbose = verbose
        self.max_position_embeddings = max_position_embeddings

        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)

        # Support loading adapter weights if available
        if load_adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, load_adapter_path)

        # Extend positional embeddings if requested
        if extend_positional and self.max_position_embeddings > 0:
            self._extend_position_embeddings(self.max_position_embeddings, self.rope_base)
        elif self.verbose:
            print(f"Loaded model {self.model_name} with {self.model.config.hidden_size} hidden size.")

        # Save original sinusoidal buffer (if applicable) for scaling
        self._create_or_store_rotary_cache()

    def _create_or_store_rotary_cache(self):
        """
        Cache sinusoidal position encodings for scaling,
        assuming model uses sinusoidal RoPE.
        """
        # Both models using rotary embeddings often have sinusoid buffers
        # For simplicity, we generate sinusoidal grid once for max position
        # and scale later. This code assumes the model uses sinusoidal RoPE.
        # For models with different positional encodings, adapt accordingly.
        max_pos = 2048  # default max for pretrained, can extend via extend_position_embeddings
        self._sinusoidal_cache = self._generate_sinusoidal_cache(max_pos)

    def _generate_sinusoidal_cache(self, max_pos: int):
        """
        Generate sinusoidal position embeddings for a range.
        Returns:
            position encodings: Tensor of shape (max_pos, hidden_dim)
        """
        dim = self.model.config.hidden_size
        position = torch.arange(0, max_pos, dtype=torch.float32)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        sinusoid_inp = position.unsqueeze(1) * div_term.unsqueeze(0)  # (max_pos, dim/2)
        sin_emb = torch.sin(sinusoid_inp)
        cos_emb = torch.cos(sinusoid_inp)
        # Expand to full size with shape (max_pos, hidden_dim)
        emb = torch.zeros((max_pos, dim))
        emb[:, 0::2] = sin_emb
        emb[:, 1::2] = cos_emb
        return emb  # shape: (max_pos, hidden_dim)

    def _extend_position_embeddings(self, new_length: int, new_theta: float):
        """
        Extend position embeddings sinusoidally to `new_length` with scaled θ.
        This modifies the model's rotary sinusoidal parameters accordingly.
        """
        if self.verbose:
            print(f"Extending position embeddings to length {new_length} with theta scaling {new_theta}.")

        # Generate new sinusoidal embeddings scaled by theta
        new_emb = self._generate_sinusoidal_cache(new_length).to(self.model.device)

        # Scale the sinusoidal embeddings' phase components to match new_theta
        # The original sinusoid wave is determined by sin(ω * pos), cos(ω * pos),
        # where ω relates to theta. Here, for scaling, we can interpolate or scale
        # the embeddings to approximate larger phase shifts.

        # For simplicity, rescale existing sinusoidal embeddings:
        # Actually, for true RoPE with scaled θ, the sin/cos functions depend on θ.
        # So, produce the sinusoid directly at the new length with scaled θ.
        # To do so, generate sinusoid with scaled phase ω' = ω * (new_theta / base_theta)
        # but since the sinusoid is generated with sinusoidal formulas, we can
        # generate directly with scaled θ.
        scaled_cache = self._generate_sinusoidal_cache(new_length).to(self.model.device)

        # Now, replace the model's rotary position embeddings with scaled ones
        self._apply_rotary_embedding_cache(scaled_cache)

        # Save the new max position length
        self.model.config.max_position_embeddings = new_length

    def _apply_rotary_embedding_cache(self, cache: torch.Tensor):
        """
        Overwrite the model's rotary sinusoidal cache with the scaled version.
        This method is model-specific:
        - For models with sinusoidal rotary embeddings, the cache is utilized in the forward.
        - For others, you may need to override attention modules.
        """
        # For simplicity, assume model uses RoPE with sin/cos buffer accessible
        # For models based on HuggingFace, this might require patching attention modules
        # or directly replacing internal sinusoid buffers if exposed.
        # This is model-specific; actual implementation varies.
        # Placeholder: No direct method to replace, so here we could override the positional encoding if possible.
        # Or, if using a model supporting [set_post_init], patch accordingly.
        # For the purpose here, assume in practice, this would be a function that patches or
        # replaces sinusoid buffers used during rotary attention.
        pass

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        Forward pass through the model.
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            outputs: Model outputs (logits, loss if labels provided)
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int=512,
        do_sample: bool=False,
        temperature: float=1.0,
        top_p: float=0.9
    ):
        """
        Generate text with long context support, optionally using sliding window.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # For sequences larger than max position, implement sliding window
        seq_len = input_ids.shape[1]
        if seq_len <= self.model.config.max_position_embeddings:
            # No need for sliding window
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )
            return output_ids
        else:
            # Use sliding window inference
            return self._generate_long_sequence(input_ids, attention_mask, max_new_tokens, do_sample, temperature, top_p)

    def _generate_long_sequence(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float
    ):
        """
        Generate long sequences by windowing over input_ids.
        """
        generated_ids = input_ids
        current_attention_mask = attention_mask

        for _ in range(max_new_tokens):
            # For last sequence, slice last max_position_embeddings tokens
            input_slice = generated_ids[:, -self.model.config.max_position_embeddings:]
            attn_slice = current_attention_mask[:, -self.model.config.max_position_embeddings:]

            outputs = self.model.generate(
                input_ids=input_slice,
                attention_mask=attn_slice,
                max_new_tokens=1,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )

            next_token = outputs[:, -1:].to(self.device)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Update attention_mask
            # Note: Normally, attention_mask is 1 for tokens and 0 for padding,
            # but here extend it accordingly.
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones_like(next_token, dtype=torch.long).to(self.device)],
                dim=1
            )

        return generated_ids

    def save_model(self, path: str):
        """
        Save the model weights and configuration.
        """
        # Save model weights
        self.model.save_pretrained(path)
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        if self.verbose:
            print(f"Model saved at {path}")

    def load_model(self, path: str):
        """
        Load a saved model weights.
        """
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.model.to(self.device)
        # Possibly reload tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.verbose:
            print(f"Model loaded from {path}")
        # Re-create sinusoidal cache
        self._create_or_store_rotary_cache()

    def extend_position_embeddings(self, new_length: int, new_theta: float):
        """
        External method to extend position embeddings after model initialization.
        """
        self._extend_position_embeddings(new_length, new_theta)

    def get_tokenizer(self):
        return self.tokenizer

```

## trainer.py

```python
## trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils import (
    load_dataset_from_json,
)
import yaml
import os
import math
from tqdm import tqdm
from model import LongContextModel
from evaluation import evaluate_retrieval_and_scaling

# Load configuration
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Hyperparameters & settings from config
learning_rate = float(CONFIG['training'].get('learning_rate', 1e-6))
batch_size = int(CONFIG['training'].get('batch_size', 128))
epochs = int(CONFIG['training'].get('epochs', 1))
total_steps = int(CONFIG['training'].get('steps_per_epoch', 14000))
warmup_ratio = float(CONFIG['training'].get('warmup_steps', 0.03))
warmup_steps = int(warmup_ratio * total_steps)

model_name = CONFIG['model'].get('name', 'mistral-7b-instruct-v0.2')
rope_base_value = float(CONFIG['model'].get('rope_base', 1e6))
use_sliding_window = CONFIG['evaluation'].get('use_sliding_window', True)
sliding_window_size = int(CONFIG['evaluation'].get('sliding_window_size', 4096))
max_position_embeddings = CONFIG['model'].get('max_position_embeddings', 0)  # if applicable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model
model = LongContextModel(
    model_name=model_name,
    rope_base=rope_base_value,
    extend_positional=(max_position_embeddings > 0),
    max_position_embeddings=max_position_embeddings
)
model.model.train()

# Prepare dataset
dataset_path = 'full_dataset.json'
dataset_list = load_dataset_from_json(dataset_path)
train_dataset = dataset_list  # Assuming dataset has been prepared already

# DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=utils.collate_fn  # Assumed imported or define in utils
)

# Optimizer
optimizer = optim.AdamW(model.model.parameters(), lr=learning_rate)

# Scheduler with cosine decay and warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Total training steps
current_step = 0
best_eval_loss = float('inf')

print("Starting training for {} steps...".format(total_steps))
for epoch in range(epochs):
    epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch in epoch_iterator:
        # batch: dict with 'input_ids', 'attention_mask', 'labels'
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward
        loss.backward()
        # Optional: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        current_step += 1

        # Logging
        if current_step % 100 == 0:
            print(f"Step {current_step}/{total_steps} - Loss: {loss.item():.4f} - LR: {scheduler.get_last_lr()[0]:.8f}")

        # Checkpointing
        if current_step % 5000 == 0:
            save_path = f"checkpoint_step_{current_step}"
            os.makedirs(save_path, exist_ok=True)
            model.save_model(save_path)
            print(f"Model checkpoint saved at {save_path}")

        # Evaluation during training
        if current_step % 1000 == 0:
            model.eval()
            eval_metrics = evaluate_retrieval_and_scaling(model, dataset_path, device=device)
            print(f"Evaluation at step {current_step}: {eval_metrics}")
            model.train()

        # Break after total steps
        if current_step >= total_steps:
            break
    # End of epoch check
    if current_step >= total_steps:
        break

# Save final model
model.save_model('final_model')
print("Training completed and model saved.")

# Optional: Final evaluation on validation set (long-context probing and real tasks)
# Can be called here or after training ends
model.eval()
final_metrics = evaluate_retrieval_and_scaling(model, dataset_path, device=device)
print("Final evaluation metrics:", final_metrics)
```

## utils.py

```python
## utils.py

import os
import json
import pickle
import random
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import openai

# Load configuration values from config.yaml
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Set default parameters from CONFIG
LONG_CONTEXT_MIN_LEN = CONFIG['long_context'].get('min_length', 4000)  # 4K tokens
LONG_CONTEXT_MAX_LEN = CONFIG['long_context'].get('max_length', 32000)  # 32K tokens
LENGTH_DISTRIBUTION = CONFIG['long_context'].get('length_distribution', [4000, 8000, 16000, 32000])
DATASET_SIZE = int(CONFIG['dataset'].get('size', 1_100_000))
BATCH_SIZE = CONFIG['dataset'].get('batch_size', 128)
TOTAL_STEPS = CONFIG['dataset'].get('steps_per_epoch', 14000)
MODEL_NAME = CONFIG['model'].get('name', 'mistral-7b-instruct-v0.2')
ROPE_BASE_DEFAULT = float(CONFIG['model'].get('rope_base', 1e6))
USE_SLIDING_WINDOW = CONFIG['evaluation'].get('use_sliding_window', True)
SLIDING_WINDOW_SIZE = CONFIG['evaluation'].get('sliding_window_size', 4096)

# Initialize tokenizer for the model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Ensure tokenizer has padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===================== Utility Functions =====================

def generate_prompt(template: str, segment: str, instruction_type: str, additional_params: Optional[dict]=None) -> str:
    """
    Generate a prompt string for GPT API based on template, segment, and instruction type.
    Args:
        template (str): Prompt template with placeholders.
        segment (str): Segment text to embed.
        instruction_type (str): Type of instruction e.g., 'fine_grained', 'multi_hop', 'reasoning'.
        additional_params (dict): Optional additional parameters for template formatting.
    Returns:
        prompt (str): Fully formatted prompt string.
    """
    prompt = template
    # Placeholders in template should include {segment}, {instruction_type}, and optional params
    if additional_params is None:
        additional_params = {}
    prompt = prompt.format(segment=segment, instruction_type=instruction_type, **additional_params)
    return prompt

def generate_qa_pair(segment: str, instruction_type: str, prompt_template: str, model_api_key: Optional[str]=None) -> Tuple[str, str]:
    """
    Call GPT-4 API with prompt to generate QA pair from segment.
    Args:
        segment (str): Short text segment.
        instruction_type (str): Instruction type guiding generation.
        prompt_template (str): Prompt template string.
        model_api_key (str): Optional, API key for openai.
    Returns:
        (question, answer): The generated QA pair.
    """
    prompt = generate_prompt(prompt_template, segment, instruction_type)
    # Use openai API or fallback to local testing
    if model_api_key is None:
        # For local testing, mock response or implement alternative
        # Here, we assume local mock
        question = "Mock question based on segment."
        answer = "Mock answer for testing."
        return question, answer
    else:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {"role": "system", "content": "Generate a question-answer pair."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=512,
            top_p=0.95,
            api_key=model_api_key
        )
        response_text = response.choices[0].message['content'].strip()
        # Parse response into question and answer
        # Assume the response is formatted as "Q: ... \nA: ..."
        qa_match = re.match(r"Q[:\s]*(.+)\nA[:\s]*(.+)", response_text, re.IGNORECASE | re.DOTALL)
        if qa_match:
            question = qa_match.group(1).strip()
            answer = qa_match.group(2).strip()
        else:
            # fallback to entire response as answer, with generic question
            question = "Generated question."
            answer = response_text
        return question, answer

def assemble_context(segments: List[str], target_length: int, strategy: str='random') -> str:
    """
    Assemble long context string from segments to match target length.
    Args:
        segments (List[str]): List of segment texts.
        target_length (int): Desired total length in tokens.
        strategy (str): 'random' or 'stratified' placement.
    Returns:
        context_text (str): Concatenated context string.
    """
    # Tokenize segments and compute their token counts
    tokenized_segments = [tokenizer(seg, add_special_tokens=False) for seg in segments]
    segment_token_lengths = [len(t['input_ids']) for t in tokenized_segments]

    # Initialize context
    context = ''
    current_length = 0

    if strategy == 'random':
        # Randomly shuffle segments
        indices = list(range(len(segments)))
        random.shuffle(indices)
        for idx in indices:
            seg_text = segments[idx]
            seg_length = segment_token_lengths[idx]
            if current_length + seg_length > target_length:
                break
            context += seg_text + ' '
            current_length += seg_length
    elif strategy == 'stratified':
        # Stratify segments across the dataset, e.g., interleave with padding or fixed pattern
        # For simplicity, do similar as above
        for idx in range(len(segments)):
            if current_length >= target_length:
                break
            context += segments[idx] + ' '
            current_length += segment_token_lengths[idx]
    else:
        # Default to concatenation until target length
        for seg_text in segments:
            seg_length = len(tokenizer(seg_text, add_special_tokens=False)['input_ids'])
            if current_length + seg_length > target_length:
                break
            context += seg_text + ' '
            current_length += seg_length

    # Remove extra spaces
    context = context.strip()
    return context

def tokenize_and_process(text: str, max_length: int=None) -> Dict:
    """
    Tokenize input text, with optional maximum length truncation.
    Args:
        text (str): Input string.
        max_length (int): Max token length.
    Returns:
        dict: tokenized 'input_ids', 'attention_mask' and 'segment_indices'.
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length' if max_length else False,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'][0],
        'attention_mask': encoding['attention_mask'][0],
        # For segmentation, find segment boundaries based on delimiters or token ids
    }

def save_dataset(dataset: List[Dict], filename: str) -> None:
    """
    Save dataset to JSON file.
    Args:
        dataset (list): List of dict samples.
        filename (str): Path to save.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def load_dataset(filename: str) -> List[Dict]:
    """
    Load dataset from JSON file.
    Args:
        filename (str): Path to dataset file.
    Returns:
        dataset (list): List of dict samples.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def prepare_training_batch(dataset: List[Dict], batch_size: int=BATCH_SIZE, use_sliding_window:bool=USE_SLIDING_WINDOW, window_size:int=SLIDING_WINDOW_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of data suitable for training.
    Args:
        dataset (list): List of samples.
        batch_size (int): Batch size.
        use_sliding_window (bool): Whether to apply sliding window.
        window_size (int): Size of sliding window in tokens.
    Returns:
        input_ids (torch.Tensor): Batch of token IDs.
        attention_mask (torch.Tensor): Attention masks.
    """
    import torch

    input_ids_list = []
    attention_mask_list = []

    for _ in range(batch_size):
        sample = random.choice(dataset)
        long_context = sample['context']
        qa_pair = sample['qa']
        question, answer = qa_pair

        # Format input prompt
        prompt_text = f"Context:\n{long_context}\nQuestion: {question}\nAnswer:"
        tokenized = tokenizer(prompt_text, max_length=None)

        # Apply sliding window if enabled
        if use_sliding_window:
            total_tokens = len(tokenized['input_ids'])
            if total_tokens > window_size:
                start_idx = random.randint(0, total_tokens - window_size)
                input_ids = tokenized['input_ids'][start_idx:start_idx+window_size]
                attn_mask = tokenized['attention_mask'][start_idx:start_idx+window_size]
            else:
                input_ids = tokenized['input_ids']
                attn_mask = tokenized['attention_mask']
        else:
            input_ids = tokenized['input_ids']
            attn_mask = tokenized['attention_mask']

        # Pad sequences to max length in batch
        max_len = max(len(input_ids), window_size if use_sliding_window else len(input_ids))
        padded_input_ids = input_ids + [tokenizer.pad_token_id] * (max_len - len(input_ids))
        padded_attention_mask = attn_mask + [0] * (max_len - len(attn_mask))

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long)

    return input_ids_tensor, attention_mask_tensor

def format_prompt_template(instruction_type:str='fine_grained') -> str:
    """
    Return the prompt template string based on instruction_type.
    Args:
        instruction_type (str): For selecting different instruction prompts.
    Returns:
        template (str): The prompt template string.
    """
    if instruction_type == 'fine_grained':
        return ("Given a short segment of text, generate one question that is highly specific "
                "to the information contained in the segment. Also, provide a concise answer "
                "based on the segment.")
    elif instruction_type == 'multi_hop':
        return ("Given multiple segments, generate a question that requires reasoning and "
                "integration of information from at least two segments. Provide the answer accordingly.")
    elif instruction_type == 'reasoning':
        return ("Using the provided context, produce a question that encourages multi-hop "
                "reasoning. Supply a precise answer that synthesizes information from different parts.")
    else:
        return ("Generate a question and answer based on the segment, emphasizing detail and reasoning.")

def add_special_tokens(prompt:str, markers:Dict[str, str]) -> str:
    """
    Insert special delimiters or markers into prompts to denote segment boundaries or key info.
    Args:
        prompt (str): Original prompt string.
        markers (dict): Markers like {'segment_start':'<seg>', 'segment_end':'</seg>'}
    Returns:
        augmented_prompt (str): Modified prompt with markers.
    """
    for marker, token in markers.items():
        prompt = prompt.replace(marker, token)
    return prompt

def compute_metrics(generated_responses: List[str], references: List[str], task_type:str='qa') -> Dict:
    """
    Calculate evaluation metrics such as accuracy, F1, recall, etc.
    Args:
        generated_responses (List[str]): Model outputs.
        references (List[str]): Ground truth answers or labels.
        task_type (str): 'QA', 'retrieval', etc.
    Returns:
        dict: Metrics scores.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    metrics = {}
    if task_type == 'qa':
        # For QA, compute F1 or exact match
        # Placeholder: Implement exact match or F1 computation
        # For simplicity, use exact match
        correct = sum([1 if g.strip().lower() == r.strip().lower() else 0 for g, r in zip(generated_responses, references)])
        accuracy = correct / len(generated_responses)
        metrics['accuracy'] = accuracy
    elif task_type == 'retrieval':
        # For retrieval, match presence in responses
        match_scores = []
        for g, r in zip(generated_responses, references):
            g_norm = normalize_text(g)
            r_norm = normalize_text(r)
            match_scores.append(1.0 if r_norm in g_norm else 0.0)
        metrics['recall'] = np.mean(match_scores)
    else:
        # Other task metrics
        pass
    return metrics

def process_retrieval(model_output:str, target_keywords:List[str]) -> bool:
    """
    Simulate retrieval success by matching model output with target keywords.
    Args:
        model_output (str): The model generated content.
        target_keywords (List[str]): Keywords or key segments to find.
    Returns:
        success (bool): Whether retrieval is successful.
    """
    output_norm = normalize_text(model_output)
    for keyword in target_keywords:
        if normalize_text(keyword) in output_norm:
            return True
    return False

def normalize_text(text:str) -> str:
    """
    Normalize text for comparison: lowercase, remove punctuation, extra spaces.
    Args:
        text (str): Raw text.
    Returns:
        normalized (str): Normalized text.
    """
    import string
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()

def calculate_context_length(token_ids: List[int]) -> int:
    """
    Count tokens in a list of token IDs.
    Args:
        token_ids (List[int]): Token IDs.
    Returns:
        length (int): Number of tokens.
    """
    return len(token_ids)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\FILM\FILM_repo`
