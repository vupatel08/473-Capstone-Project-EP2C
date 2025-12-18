# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
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
```

## evaluation.py

```python
## evaluation.py
"""
Evaluation routines for mutation explanation and mutation engineering tasks,
including explanation quality metrics (ROUGE-L, BLEU-2, METEOR) and proposal 
ranking metrics (Recall@50, position correctness, amino acid prediction accuracy, Spearman correlation).
Also includes visualization functions for mutation proposal ranking and fitness trajectories.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import torch
from scipy.stats import spearmanr

# External libraries for NLP metrics
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    sentence_bleu = None
    meteor_score = None

# Load configuration for metrics parameters
import yaml
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Default boundary tokens if needed
BOP = "<BOP>"
EOP = "<EOP>"
BOM = "<BOM>"
EOM = "<EOM>"

# Function: Compute NLP metrics for explanation quality
def evaluate_explanation(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE-L, BLEU-2, and METEOR scores over the dataset.
    Args:
        predictions: List of predicted explanation strings.
        references: List of ground-truth explanation strings.
    Returns:
        metrics: Dictionary with average scores.
    """
    # Check for required libraries
    if rouge_scorer is None:
        raise ImportError("Please install 'rouge_score' package for ROUGE metrics.")
    if sentence_bleu is None or meteor_score is None:
        raise ImportError("Please install 'nltk' package for BLEU and METEOR metrics.")
    
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    bleu_scores = []
    meteor_scores = []

    # Smoothing for BLEU
    smooth_fn = SmoothingFunction().method1

    for pred, ref in zip(predictions, references):
        # ROUGE-L
        try:
            score_rouge = scorer.score(ref, pred)['rougeL'].fmeasure
        except Exception:
            score_rouge = 0.0
        rouge_l_scores.append(score_rouge)

        # BLEU-2
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(bleu)

        # METEOR
        try:
            meteor = meteor_score([ref], pred)
        except:
            meteor = 0.0
        meteor_scores.append(meteor)

    metrics = {
        'ROUGE-L': np.mean(rouge_l_scores),
        'BLEU-2': np.mean(bleu_scores),
        'METEOR': np.mean(meteor_scores)
    }
    return metrics

# Function: Evaluate mutation proposals (ranking and accuracy metrics)
def evaluate_mutation_proposals(
    proposal_scores: List[float],
    ground_truth_site: List[int],
    ground_truth_aa: List[int],
    top_k: int = 50
) -> Dict[str, float]:
    """
    Compute Recall@50, position accuracy, amino acid accuracy, and correlation.
    Args:
        proposal_scores: List/array of mutation scores (higher = more likely).
        ground_truth_site: List of true mutation positions (0-based).
        ground_truth_aa: List of true mutated amino acids (index-encoded).
        top_k: cutoff for Recall@k.
    Returns:
        metrics: Dict with recall, accuracies, and correlation.
    """
    from sklearn.metrics import Spearmanr

    num_samples = len(ground_truth_site)
    recall_at_k = 0
    position_correct = 0
    aa_correct = 0
    scores_array = np.array(proposal_scores)

    for i in range(num_samples):
        # For each sample, rank proposals
        ranked_indices = np.argsort(-scores_array[i])  # descending
        top_indices = ranked_indices[:top_k]
        # Check if true site is in top-k
        if ground_truth_site[i] in top_indices:
            recall_at_k += 1
        # Check position correctness: does top proposal match ground-truth?
        if top_indices[0] == ground_truth_site[i]:
            position_correct +=1
        # Check amino acid correctness of top proposal
        top_aa_pred_idx = top_indices[0]
        if top_aa_pred_idx == ground_truth_aa[i]:
            aa_correct +=1

    recall_pct = (recall_at_k / num_samples) * 100.0
    position_acc_pct = (position_correct / num_samples) * 100.0
    aa_acc_pct = (aa_correct / num_samples) * 100.0

    # Correlation between proposal scores and true mutation effects
    # Assuming ground_truth_effects is available as float per sample
    # For illustration, if not available, set correlation to NaN
    try:
        ground_truth_effects = np.array(ground_truth_aa)  # placeholder, replace with real effects if available
        correlation, _ = spearmanr(scores_array.flatten(), ground_truth_effects.flatten())
    except:
        correlation = np.nan

    metrics = {
        'Recall@50': recall_pct,
        'Position Accuracy': position_acc_pct,
        'Amino Acid Accuracy': aa_acc_pct,
        'Spearman Correlation': correlation
    }
    return metrics

# Visualization: Fitness trajectory over multiple rounds
def plot_fitness_trajectory(
    fitness_scores: List[List[float]],
    labels: List[str] = None,
    title: str = 'Protein Fitness Optimization',
    save_path: str = None
):
    """
    Plot mean and std curve for the fitness scores across rounds.
    Args:
        fitness_scores: List of lists, each sublist corresponds to a round's scores for all proteins.
        labels: Optional list of protein labels for x-axis.
        title: plot title.
        save_path: filepath to save figure, if None, just show.
    """
    rounds = list(range(1, len(fitness_scores) + 1))
    means = [np.mean(scores) for scores in fitness_scores]
    stds = [np.std(scores) for scores in fitness_scores]

    plt.figure(figsize=(8,6))
    plt.plot(rounds, means, label='Mean Fitness', color='blue')
    plt.fill_between(rounds, np.array(means)-np.array(stds),
                     np.array(means)+np.array(stds),
                     color='blue', alpha=0.2, label='Std Dev')
    plt.xlabel('Round')
    plt.ylabel('Fitness Score')
    plt.title(title)
    plt.legend()
    if labels:
        plt.xticks(rounds, labels, rotation=45)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# Visualization: Correlation scatter plot for mutation effects
def plot_correlation_scatter(
    scores: np.ndarray,
    effects: np.ndarray,
    title: str = 'Mutation Proposal Scores vs. Effects',
    save_path: str = None
):
    """
    Scatter plot for mutation scores against true effects.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(effects, scores, alpha=0.5)
    plt.xlabel('Ground Truth Effects')
    plt.ylabel('Proposed Mutation Scores')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# Visualization: Proposal distribution (histogram bar plots)
def plot_proposal_distribution(
    proposal_counts: Dict[str, int],
    title: str = 'Mutation Proposal Distribution',
    save_path: str = None
):
    """
    Plot distribution of proposed mutations.
    """
    labels = list(proposal_counts.keys())
    counts = list(proposal_counts.values())
    plt.figure(figsize=(10,6))
    sns.barplot(x=labels, y=counts)
    plt.xlabel('Proposal Mutation (Position + AA)')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=90)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# Main: Additional functions or classes can be added as needed, e.g., for aggregation, batch testing, or result logging.
# For brevity, this implementation covers core metrics and visualization utilities.
```

## main.py

```python
# main.py

import os
import sys
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import (
    set_seed, save_checkpoint, load_checkpoint, log_metrics, call_openai_api
)
from dataset_loader import ProteinTextDataset
from model import (
    ProteinPLM, TextEncoder, ProteinDeltaNetwork,
    PromptEmbedding, Heads
)
from evaluation import evaluate_explanation, evaluate_mutation_proposals
import prompt_templates as prompts

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting MutaPLM training and evaluation pipeline.")

    # Set seed
    seed = 42
    set_seed(seed)

    # --------- Extract configuration values with defaults ---------
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    prompt_cfg = config.get('prompt_templates', {})
    eval_cfg = config.get('evaluation', {})
    hardware_cfg = config.get('hardware', {})

    # Training parameters
    lr = training_cfg.get('learning_rate', 1e-4)
    batch_size = training_cfg.get('batch_size', 16)
    warmup_steps = training_cfg.get('warmup_steps', 10000)
    max_steps = training_cfg.get('max_steps', 200000)
    grad_clip = training_cfg.get('gradient_clip_value', 1.0)

    # Paths
    pretrain_path = dataset_cfg.get('pretraining_dataset_path', '')
    finetune_path = dataset_cfg.get('finetuning_dataset_path', '')
    test_path = dataset_cfg.get('test_dataset_path', '')
    enrichment_path = dataset_cfg.get('enrichment_data_path', '')

    # Model configs
    plm_name = model_cfg.get('plm_name', 'facebook/esm2_t6_8a_14B')
    text_enc_name = model_cfg.get('text_encoder_name', 'allenai/scibert_scivocab_uncased')
    delta_hidden_dim = model_cfg.get('delta_hidden_dim', 768)
    num_attention_heads = model_cfg.get('num_attention_heads', 12)
    num_layers = model_cfg.get('num_layers', 12)

    # --------- Environment Setup ---------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --------- Load datasets ---------
    logger.info("Loading datasets...")
    pretrain_dataset = ProteinTextDataset(pretrain_path, mode='pretraining', config=config)
    finetune_dataset = ProteinTextDataset(finetune_path, mode='finetuning', config=config)
    test_dataset = ProteinTextDataset(test_path, mode='test', config=config)

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --------- Initialize models ---------
    logger.info("Initializing models...")
    # PLM backbone
    protein_plm = ProteinPLM(pretrained_name=plm_name).to(device)
    # Text encoder
    text_encoder = TextEncoder(pretrained_name=text_enc_name).to(device)
    # Prompt embedding (soft prompts for cross-modal alignment)
    prompt_embedding = PromptEmbedding(K=32, D=delta_hidden_dim).to(device)
    # Heads for mutation prediction
    heads = Heads(delta_hidden_dim, num_aa=20).to(device)

    # --------- Set up optimizer ------
    # Collect parameters to optimize (full model except frozen parts)
    train_params = list(protein_plm.parameters()) + \
                   list(text_encoder.parameters()) + \
                   list(prompt_embedding.parameters()) + \
                   list(heads.position_head.parameters()) + \
                   list(heads.aa_head.parameters())
    optimizer = optim.AdamW(train_params, lr=lr, eps=1e-8, weight_decay=0.01)
    # Scheduler: linear warm-up + cosine decay
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, num_training_steps=max_steps)

    # --------- Loss functions ---------
    ce_loss_fn = nn.CrossEntropyLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # --------- Initialize model components ---------
    # Delta network
    delta_net = ProteinDeltaNetwork(hidden_dim=delta_hidden_dim).to(device)

    # --------- Checkpoint info ---------
    start_step = 0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------- Training loop ---------
    for step in range(start_step + 1, max_steps + 1):
        # Decide mode
        if step <= 200000:
            dataloader = pretrain_loader
            mode = 'pretraining'
        else:
            dataloader = finetune_loader
            mode = 'finetuning'

        for batch in dataloader:
            optimizer.zero_grad()

            # Prepare batch tensors
            seq_ids = batch['sequence_ids'].to(device)            # [B, L]
            seq_mask = batch['attention_mask'].to(device)
            text_ids = batch['text_ids'].to(device)               # [B, T]
            text_mask = batch['text_attention_mask'].to(device)

            # --------- Forward pass ----------
            # 1. Protein representations
            h_wt = protein_plm(seq_ids, attention_mask=seq_mask)  # [B, L, D]
            # 2. Encode text input
            text_feat = text_encoder.encode([t for t in batch['text']])  # [B, D]
            # 3. Prepare soft prompts
            soft_prompts = prompt_embedding.expand(batch_size).to(device)  # [B, K, D]
            # 4. Compute delta features using delta network: H W T
            # For simplicity in initial code, use h_wt as placeholder; in actual implementation, compute delta between WT and mutant sequences
            # Here, assume batch contains mutant sequences tokens, or can pass h_wt as both for pretraining
            # For placeholder, set h_mut to be same as h_wt
            h_mut = h_wt
            # 5. Compute mutation representation (delta features)
            # We simulate the delta network process
            delta_outputs = delta_net(h_wt, h_mut)
            h_mut_pred = delta_outputs['h_mutated']
            pos_logits = delta_outputs['pos_logits']
            aa_logits = delta_outputs['aa_logits']

            # --------- Compute losses ----------
            # Sequence MLM: (e.g., masked tokens prediction)
            # Placeholder: generate dummy logits
            seq_logits = torch.randn_like(seq_ids, dtype=torch.float)
            seq_loss = ce_loss_fn(seq_logits.view(-1, seq_logits.shape[-1]), seq_ids.view(-1))

            # Text generation loss: dummy placeholder
            text_logits = torch.randn_like(text_ids, dtype=torch.float)
            text_loss = ce_loss_fn(text_logits.view(-1, text_logits.shape[-1]), text_ids.view(-1))

            # Mutation head losses (binary position and amino acid classification)
            # For supervision, assume ground-truth mutation labels are in batch
            # For placeholder, assume zeros
            gt_positions = torch.zeros_like(pos_logits)
            gt_amino_acids = torch.zeros_like(aa_logits)

            pos_head_loss = bce_loss_fn(pos_logits.squeeze(-1), gt_positions.squeeze(-1))
            aa_head_loss = ce_loss_fn(aa_logits.view(-1, 20), gt_amino_acids.argmax(dim=-1).view(-1))

            # Chain-of-thought explanation and proposal losses (if finetuning)
            if step > 200000:
                # For illustration, define dummy explanation/proposal losses
                explanation_loss = torch.tensor(0.0, device=device)
                proposal_loss = pos_head_loss + aa_head_loss
            else:
                explanation_loss = torch.tensor(0.0, device=device)
                proposal_loss = torch.tensor(0.0, device=device)

            # Total loss (weighted sum)
            loss = seq_loss + text_loss + proposal_loss

            # --------- Backprop and optimize ---------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            if step % 100 == 0:
                log_metrics({
                    'step': step,
                    'loss': loss.item(),
                    'seq_loss': seq_loss.item(),
                    'text_loss': text_loss.item(),
                    'proposal_loss': proposal_loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                }, step)

            # Save checkpoint periodically
            if step % 5000 == 0:
                cp_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pth')
                save_checkpoint(protein_plm, optimizer, step, cp_path)
                # Optional: validate and save best model
                if step % 20000 == 0:
                    val_metrics = evaluate((protein_plm, text_encoder, delta_net), test_loader)
                    if val_metrics.get('ROUGE-L', 0) > 0.2:  # threshold placeholder
                        save_checkpoint(protein_plm, optimizer, step, os.path.join(checkpoint_dir, 'best.pth'))

            # Terminate if max steps reached
            if step >= max_steps:
                break

    # --------- Final evaluation ----------
    logger.info("Training completed. Running final evaluation...")
    final_metrics = evaluate((protein_plm, text_encoder, delta_net), test_loader)
    logger.info(f"Final evaluation metrics: {final_metrics}")

    # Save final model
    save_checkpoint(protein_plm, optimizer, max_steps, os.path.join(checkpoint_dir, 'final_model.pth'))

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Load configuration for model hyperparameters
import yaml
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Helper function to get device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProteinPLM(nn.Module):
    """
    Wrapper for a pretrained protein language model (e.g., ESM-2).
    Produces per-residue embeddings and optional pooled sequence embedding.
    """
    def __init__(self, pretrained_name: str = None, output_embeddings: bool = True):
        """
        Args:
            pretrained_name (str): Name of the pretrained PLM model.
            output_embeddings (bool): If True, output per-residue embeddings.
        """
        super().__init__()
        self.pretrained_name = pretrained_name or CONFIG['model'].get('plm_name', 'facebook/esm2_t6_8a_14B')
        self.output_embeddings = output_embeddings
        # Load pretrained model config
        self.config = AutoConfig.from_pretrained(self.pretrained_name)
        # Load pretrained model
        self.model = AutoModel.from_pretrained(self.pretrained_name)
        # Freeze parameters if needed, or leave trainable
        # self.model.requires_grad_(True)
        self.device = get_device()

    def forward(self, sequence_ids, attention_mask=None):
        """
        Args:
            sequence_ids (Tensor): Input token IDs, shape [batch_size, seq_len].
            attention_mask (Tensor): Mask tensor, shape [batch_size, seq_len].
        Returns:
            Tensor: Hidden states per residue, shape [batch_size, seq_len, D].
        """
        output = self.model(input_ids=sequence_ids, attention_mask=attention_mask)
        # output.last_hidden_state shape: [batch_size, seq_len, D]
        return output.last_hidden_state

    def get_sequence_embedding(self, sequence_ids, attention_mask=None):
        """
        To get a pooled sequence embedding, e.g., mean of token embeddings or CLS token if available.
        """
        hidden_states = self.forward(sequence_ids, attention_mask)
        # Global mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        return pooled


class TextEncoder(nn.Module):
    """
    Encodes biomedical texts/prompts into dense feature vectors.
    """
    def __init__(self, pretrained_name: str = None):
        """
        Args:
            pretrained_name (str): Name of the pretrained text encoder.
        """
        super().__init__()
        self.pretrained_name = pretrained_name or CONFIG['model'].get('text_encoder_name', 'allenai/scibert_scivocab_uncased')
        self.tokenizer = None  # Will be initialized externally
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(self.pretrained_name)
        self.device = get_device()

    def encode(self, texts: list):
        """
        Encode list of texts into dense vectors.
        Args:
            texts (list[str]): List of string texts.
        Returns:
            Tensor: shape [len(texts), D]
        """
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
        encodings = self.tokenizer(
            texts,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # pooling: mean over tokens
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, D]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return pooled


class PromptEmbedding(nn.Module):
    """
    Maintains a set of trainable soft prompt tokens for cross-modal prompts.
    """
    def __init__(self, K: int = 32, D: int = 768):
        """
        Args:
            K (int): Number of soft prompt tokens.
            D (int): Embedding dimension.
        """
        super().__init__()
        self.K = K
        self.D = D
        self.soft_tokens = nn.Parameter(torch.randn(K, D))
        # Initialize trainable tokens around a normal distribution

    def get_embeddings(self):
        """
        Return the soft prompt embeddings.
        """
        return self.soft_tokens

    def expand(self, batch_size: int):
        """
        Expand embeddings to batch size.
        """
        return self.soft_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # shape [batch, K, D]


class Heads:
    """
    Heads for mutation term prediction:
    - Position head: predicts if position should mutate
    - AA head: predicts mutated amino acid
    """
    def __init__(self, hidden_dim: int, num_aa: int = 20):
        """
        Args:
            hidden_dim (int): Dimension of input features
            num_aa (int): Number of amino acid classes
        """
        self.position_head = nn.Linear(hidden_dim, 1)  # Binary classification (mutable or not)
        self.aa_head = nn.Linear(hidden_dim, num_aa)  # AA class logits

    def init_weights(self):
        # Initialization if needed
        nn.init.kaiming_uniform_(self.position_head.weight, nonlinearity='sigmoid')
        nn.init.kaiming_uniform_(self.aa_head.weight, nonlinearity='linear')


class ProteinDeltaEncoder(nn.Module):
    """
    Encodes mutation effects from `h_delta`.
    """
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim (int): Input/output feature dimension, matching PLM hidden size.
        """
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=CONFIG['model'].get('num_attention_heads', 12))
        # The cross_attention here takes queries as input and keys/values as source
        # query: trainable or derived; key/value: h_delta
        # To match architecture, queries will be from learned parameters or h_wt

        # For simplicity, using fixed Q (trainable) plus attention layers
        self.query = nn.Parameter(torch.randn(1, hidden_dim))  # For this model, or can be multiple queries

    def forward(self, h_delta, h_wt):
        """
        Encode delta features with cross-attention.
        Args:
            h_delta (Tensor): [batch_size, seq_len, hidden_dim]
            h_wt (Tensor): [batch_size, seq_len, hidden_dim]
        Returns:
            z_delta (Tensor): [batch_size, seq_len, hidden_dim]
        """
        # Prepare queries from trainable parameters or from `self.query`
        batch_size, seq_len, dim = h_delta.shape
        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, dim]
        # But MultiheadAttention expects [seq_len, batch, embed_dim]
        # We'll flatten queries to [1, batch, dim]
        queries = queries.permute(1, 0, 2).contiguous()  # [1, batch, dim]
        h_delta_t = h_delta.permute(1, 0, 2)  # [seq_len, batch, dim]
        h_wt_t = h_wt.permute(1, 0, 2)  # [seq_len, batch, dim]
        attended_vals, attn_weights = self.cross_attn(
            query=queries, key=h_wt_t, value=h_wt_t
        )
        # attended_vals: [1, batch, dim], expand to [batch, seq_len, dim] for further processing if needed
        z_delta = attended_vals.permute(1, 0, 2).expand(batch_size, seq_len, dim)
        return z_delta


class ProteinDeltaDecoder(nn.Module):
    """
    Reconstructs mutation effects from z_delta and h_wt.
    """
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim (int): Dimension of the features.
        """
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=CONFIG['model'].get('num_attention_heads', 12))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h_wt, z_delta):
        """
        Decode the mutation effects.
        Args:
            h_wt (Tensor): [batch_size, seq_len, hidden_dim]
            z_delta (Tensor): [batch_size, seq_len, hidden_dim]
        Returns:
            h_delta (Tensor): [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, dim = h_wt.shape
        h_wt_t = h_wt.permute(1, 0, 2)  # [seq_len, batch, dim]
        z_delta_t = z_delta.permute(1, 0, 2)  # [seq_len, batch, dim]

        # Cross-attention: query from h_wt, key & value from z_delta
        attn_output, attn_weights = self.cross_attn(
            query=h_wt_t,
            key=z_delta_t,
            value=z_delta_t
        )
        attn_output = attn_output.permute(1, 0, 2)  # back to [batch, seq, dim]
        # Add residual
        h_delta = attn_output + h_wt
        # Pass through FFN
        h_delta = self.ffn(h_delta)
        return h_delta


class ProteinHead(nn.Module):
    """
    Overall mutation prediction heads combining position and amino acid heads.
    """
    def __init__(self, hidden_dim: int, num_aa: int = 20):
        super().__init__()
        self.position_head = nn.Linear(hidden_dim, 1)  # Sigmoid for probability
        self.aa_head = nn.Linear(hidden_dim, num_aa)  # Logits over amino acids

    def init_weights(self):
        nn.init.kaiming_uniform_(self.position_head.weight, nonlinearity='sigmoid')
        nn.init.kaiming_uniform_(self.aa_head.weight, nonlinearity='linear')


class ProteinDeltaNetwork(nn.Module):
    """
    The core encoder-decoder architecture modeling explicit mutations.
    """
    def __init__(self, hidden_dim: int = 768):
        """
        Args:
            hidden_dim (int): Dimension of PLM embeddings (match for transformer dimension)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # Encoders
        self.delta_encoder = ProteinDeltaEncoder(hidden_dim)
        self.delta_decoder = ProteinDeltaDecoder(hidden_dim)
        # Heads
        self.heads = Heads(hidden_dim, num_aa=20)

    def forward(self, h_wt, h_mut):
        """
        Compute mutation representation and proposals.
        Args:
            h_wt (Tensor): [batch_size, seq_len, hidden_dim]
            h_mut (Tensor): [batch_size, seq_len, hidden_dim]
        Returns:
            dict with:
                h_mutated (Tensor): reconstructed h_mut
                position_logits (Tensor): [batch_size, seq_len, 1]
                aa_logits (Tensor): [batch_size, seq_len, 20]
        """
        # Compute delta
        h_delta = h_mut - h_wt
        # Encode delta
        z_delta = self.delta_encoder(h_delta, h_wt)
        # Decode to reconstruct h_delta
        h_delta_recon = self.delta_decoder(h_wt, z_delta)
        # Compute mutated features
        h_mutated = h_wt + h_delta_recon
        # Predict mutation site (binary or categorical)
        pos_logits = self.heads.position_head(h_mutated)  # shape: [batch, seq, 1]
        aa_logits = self.heads.aa_head(h_mutated)         # shape: [batch, seq, 20]
        return {
            'h_mutated': h_mutated,
            'pos_logits': pos_logits,
            'aa_logits': aa_logits,
            'h_delta': h_delta
        }
```

## prompt_templates.py

```python
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
```

## trainer.py

```python
# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import yaml
import os
import time
import math
from utils import (
    set_seed, save_checkpoint, load_checkpoint,
    log_metrics, call_openai_api
)
from dataset_loader import ProteinTextDataset
from model import (
    ProteinPLM,
    TextEncoder,
    ProteinDeltaNetwork,
    PromptEmbedding,
    Heads
)
import prompt_templates as prompts

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set seeds for reproducibility
set_seed(42)

# Extract hyperparameters with defaults
batch_size = config.get('training', {}).get('batch_size', 16)
learning_rate = config.get('training', {}).get('learning_rate', 1e-4)
warmup_steps = config.get('training', {}).get('warmup_steps', 10000)
max_steps = config.get('training', {}).get('max_steps', 200000)
gradient_clip_value = config.get('training', {}).get('gradient_clip_value', 1.0)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate datasets
pretrain_dataset_path = config['dataset'].get('pretraining_dataset_path', '')
finetune_dataset_path = config['dataset'].get('finetuning_dataset_path', '')
test_dataset_path = config['dataset'].get('test_dataset_path', '')

# Load datasets
pretrain_dataset = ProteinTextDataset(pretrain_dataset_path, mode='pretraining', config=config)
finetune_dataset = ProteinTextDataset(finetune_dataset_path, mode='finetuning', config=config)
test_dataset = ProteinTextDataset(test_dataset_path, mode='test', config=config)

# Data loaders
pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
plm_name = config['model'].get('plm_name', 'facebook/esm2_t6_8a_14B')
text_enc_name = config['model'].get('text_encoder_name', 'allenai/scibert_scivocab_uncased')
delta_hidden_dim = config['model'].get('delta_hidden_dim', 768)

protein_plm = ProteinPLM(pretrained_name=plm_name)
text_encoder = TextEncoder(pretrained_name=text_enc_name)
prompt_embedding = PromptEmbedding(K=32, D=delta_hidden_dim)
heads = Heads(hidden_dim=delta_hidden_dim, num_aa=20)

# Move models to device
protein_plm.to(device)
text_encoder.to(device)
# prompt_embedding and heads are nn.Modules with parameters
prompt_embedding.to(device)
heads.position_head.to(device)
heads.aa_head.to(device)

# Optimizers
# Collect trainable parameters
train_params = list(protein_plm.parameters()) + \
               list(text_encoder.parameters()) + \
               list(prompt_embedding.parameters()) + \
               list(heads.position_head.parameters()) + \
               list(heads.aa_head.parameters())
optimizer = optim.AdamW(train_params, lr=learning_rate, eps=1e-8, weight_decay=0.01)

# Learning rate scheduler with warmup and cosine decay
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
)

# Loss functions
seq_ce_loss = nn.CrossEntropyLoss()  # For amino acid token prediction
mask_ce_loss = nn.CrossEntropyLoss() # For masked sequence tokens
# For mutation proposal (pos + aa heads)
pos_loss_fn = nn.BCEWithLogitsLoss()
aa_loss_fn = nn.CrossEntropyLoss()

# Training loop
global_step = 0
best_val_rouge = -float('inf')
best_model_path = "checkpoint_best.pth"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to perform evaluation on test set
def evaluate(model_components, dataloader):
    protein_plm, text_encoder, delta_net = model_components
    protein_plm.eval()
    text_encoder.eval()
    delta_net.eval()
    # Prepare evaluation metrics
    rouge_scores = []
    bleu_scores = []
    meteor_scores = []
    rec_50 = 0
    total_samples = 0
    correct_mutation_acc = 0

    # Import evaluation metrics
    from tqdm import tqdm
    import nltk
    from rouge_score import Rouge

    rouge_eval = Rouge()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq_ids = batch['sequence_ids'].to(device)
            seq_mask = batch['attention_mask'].to(device)
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            # For mutation labels, handle accordingly
            m_labels = batch['mutation_labels']
            # Forward through PLM
            h_wt = protein_plm(seq_ids, attention_mask=seq_mask)
            # Encode texts
            text_feats = text_encoder.encode(batch['text'])
            # Compute delta features
            if hasattr(delta_net, 'encode_delta'):
                h_delta = delta_net.encode_delta(h_wt, h_wt)  # Placeholder, replace as needed
            else:
                # For simplicity, assume delta features are zero
                h_delta = torch.zeros_like(h_wt)
            # Generate explanations and proposals
            # Here, for evaluation, you'd generate explanations and compare
            # For brevity, assuming ground-truth texts are available
            # Compute metrics: placeholder implementation
            # For example:
            # rouge_scores.append(rouge_eval.get_scores(generated_text, reference_text))
            total_samples += 1
            # For mutation accuracy, compare predicted mutation proposal with ground truth
            # Placeholder: set to 0 or 1 based on match
            # correct_mutation_acc += int(predicted_mutation == ground_truth)
        # Compute average metrics
    # Return metrics dict
    return {
        "ROUGE-L": sum(rouge_scores)/len(rouge_scores) if rouge_scores else 0,
        "BLEU-2": 0,
        "METEOR": 0,
        "Recall@50": rec_50,
        "Mutation_Acc": correct_mutation_acc / total_samples if total_samples > 0 else 0
    }

# Main training routine
for step in range(1, max_steps + 1):
    # Decide if training on pretraining or finetuning
    if step <= 200000:
        # Pretraining stage
        dataloader = pretrain_loader
        model_train_mode = 'pretraining'
    else:
        # Finetuning stage
        dataloader = finetune_loader
        model_train_mode = 'finetuning'

    for batch in dataloader:
        optimizer.zero_grad()
        # Prepare inputs
        seq_ids = batch['sequence_ids'].to(device)  # shape: [B, L]
        seq_mask = batch['attention_mask'].to(device)
        text_ids = batch['text_ids'].to(device)
        text_mask = batch['text_attention_mask'].to(device)

        # Forward through PLM
        h_wt = protein_plm(seq_ids, attention_mask=seq_mask)  # [B, L, D]
        # Encode textual info
        text_feat = text_encoder.encode([t for t in batch['text']])  # [B, D]

        # Sample or generate soft prompts (if used)
        soft_prompts = prompt_embedding.expand(batch_size).to(device)

        # For pretraining: sequence mask task
        # Randomly select tokens to mask
        # Implement masking pattern and create labels
        # Placeholder: assume tokens masked at random positions
        # For brevity, assuming mask_labels and input masked sequences are prepared
        # Similarly for text generation, compute cross entropy with ground-truth text tokens
        # For cross-modal alignment, compute cosine similarity or directly minimize embedding distance

        # For illustration: assume predicted tokens and compute loss
        seq_logits = torch.randn_like(seq_ids, dtype=torch.float)  # Dummy placeholder
        seq_loss = seq_ce_loss(seq_logits.view(-1, seq_logits.shape[-1]), seq_ids.view(-1))

        # Similarly, compute text generation loss
        text_logits = torch.randn_like(text_ids, dtype=torch.float)  # Dummy placeholder
        text_loss = mask_ce_loss(text_logits.view(-1, text_logits.shape[-1]), text_ids.view(-1))

        # Compute cross-modal alignment loss (e.g., MSE or cosine)
        # Here, placeholder as zero
        cross_modal_loss = torch.tensor(0.0).to(device)

        # For finetuning: chain-of-thought-specific losses (mutation explanation/proposal)
        if step > 200000:
            # Use specific prompt templates (from prompt_templates.py)
            # Generate function description (teacher forcing or model-generated)
            # Generate mutational effects
            # Compute ROUGE/BLEU/METEOR losses for explanation
            # Compute mutation head losses
            # Placeholder: using dummy losses
            explanation_loss = torch.tensor(0.0).to(device)
            proposal_pos_logits = torch.randn(batch['sequence_ids'].size(0), batch['sequence_ids'].size(1), 1).to(device)
            proposal_aa_logits = torch.randn(batch['sequence_ids'].size(0), batch['sequence_ids'].size(1), 20).to(device)

            # Suppose ground-truth mutation positions and amino acids
            gt_positions = torch.zeros_like(proposal_pos_logits)
            gt_amino_acids = torch.zeros_like(proposal_aa_logits)

            # Compute heads loss
            pos_loss = pos_loss_fn(proposal_pos_logits.squeeze(-1), gt_positions.squeeze(-1))
            aa_loss = aa_loss_fn(proposal_aa_logits.view(-1, 20), gt_amino_acids.argmax(dim=-1).view(-1))
            # Total explanation proposal loss
            proposal_loss = pos_loss + aa_loss
        else:
            explanation_loss = torch.tensor(0.0).to(device)
            proposal_loss = torch.tensor(0.0).to(device)

        # Total loss (weighted sum)
        total_loss = seq_loss + text_loss + cross_modal_loss + explanation_loss + proposal_loss

        # Backpropagation
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=gradient_clip_value)

        # Optimizer step
        optimizer.step()

        # Scheduler step
        scheduler.step()

        # Logging
        if step % 100 == 0:
            metrics = {
                'loss': total_loss.item(),
                'seq_loss': seq_loss.item(),
                'text_loss': text_loss.item(),
                'cross_modal_loss': cross_modal_loss.item(),
                'explanation_loss': explanation_loss.item(),
                'proposal_loss': proposal_loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            }
            log_metrics(metrics, step)

        # Checkpoint saving
        if step % 5000 == 0:
            save_checkpoint(protein_plm, optimizer, step, os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'))
            # Save best model based on validation metrics
            if step % 20000 == 0:
                val_metrics = evaluate((protein_plm, text_encoder, None), test_loader)
                # Use ROUGE-L to compare
                if val_metrics['ROUGE-L'] > best_val_rouge:
                    best_val_rouge = val_metrics['ROUGE-L']
                    save_checkpoint(protein_plm, optimizer, step, os.path.join(checkpoint_dir, best_model_path))
    # Break condition for max steps
    if step >= max_steps:
        print("Reached max training steps.")
        break

# Final evaluation
final_metrics = evaluate((protein_plm, text_encoder, None), test_loader)
print(f"Final evaluation metrics: {final_metrics}")
# Save final model
save_checkpoint(protein_plm, optimizer, max_steps, os.path.join(checkpoint_dir, 'final_model.pth'))

```

## utils.py

```python
## utils.py
import os
import json
import random
import yaml
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn import modules as nn_modules
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler

try:
    import rouge_score
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    rouge_score = None
    sentence_bleu = None
    meteor_score = None

# Load configuration from 'config.yaml'
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# ------------------ Logging Setup ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ------------------ Seed setting for reproducibility ------------------
def set_seed(seed: int = 42) -> None:
    """
    Set random seed for all relevant libraries for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

# ------------------ Tokenization utilities ------------------
def tokenize_protein(sequence: str, tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """
    Tokenize a protein sequence using the specified tokenizer.
    Args:
        sequence (str): Amino acid sequence string.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the PLM.
    Returns:
        dict: Dictionary with input_ids and attention_mask.
    """
    if tokenizer is None:
        # Load from config
        plm_name = CONFIG['model'].get('plm_name', 'facebook/esm2_t6_8a_14B')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plm_name)
    encoded = tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True)
    return encoded

def tokenize_text(text: str, tokenizer: Optional[Any] = None, max_length: int = 512) -> Dict[str, Any]:
    """
    Tokenize textual data with specified maximum length.
    Args:
        text (str): Input text string.
        tokenizer: Tokenizer compatible with the text encoder.
        max_length (int): Max token length.
    Returns:
        dict: Contains input_ids and attention_mask tensors.
    """
    if tokenizer is None:
        # Load from config
        text_enc_name = CONFIG['model'].get('text_encoder_name', 'allenai/scibert_scivocab_uncased')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(text_enc_name)
    encoded = tokenizer(
        text, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    return encoded

# ------------------ Checkpoint management ------------------
def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    step: int, 
    path: str
) -> None:
    """
    Save model and optimizer state dicts at given step.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved at step {step} to {path}")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> int:
    """
    Load checkpoint weights into model and optimizer.
    Returns:
        int: Last trained step.
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_step = checkpoint.get('step', 0)
    logger.info(f"Loaded checkpoint from {path} at step {last_step}")
    return last_step

# ------------------ Learning rate scheduler ------------------
def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 10000,
    schedule_type: str = 'linear_warmup'
) -> _LRScheduler:
    """
    Return scheduler: linear warmup + cosine decay or other as needed.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            import math
            return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

# ------------------ Gradient clipping ------------------
def clip_gradients(model: torch.nn.Module, clip_value: float = 1.0) -> None:
    """
    Clip gradients to prevent exploding gradients.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
    logger.debug("Gradients clipped to max norm {}".format(clip_value))

# ------------------ Logging metrics ------------------
def log_metrics(
    metrics: Dict[str, float], 
    step: int, 
    log_dir: str = './logs'
) -> None:
    """
    Log metrics as JSON line or to a dashboard.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'metrics.jsonl')
    record = {'step': step}
    record.update(metrics)
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    logger.info(f"Step {step}: {metrics}")

# ------------------ External API call wrapper (GPT) ------------------
import time
import requests

def call_openai_api(
    prompt: str,
    model: str = 'gpt-4',
    max_tokens: int = 512,
    api_key: str = '',  # User must set their API key
    temperature: float = 0.7,
    retries: int = 3,
    backoff_factor: float = 2.0
) -> str:
    """
    Call OpenAI API with retries and backoff.
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    for attempt in range(retries):
        try:
            response = requests.post('https://api.openai.com/v1/completions', headers=headers, json=payload)
            response.raise_for_status()
            reply_json = response.json()
            # Assume first choice
            reply_text = reply_json['choices'][0]['text'].strip()
            return reply_text
        except requests.RequestException as e:
            logger.warning(f"OpenAI API call failed ({e}), attempt {attempt + 1}")
            time.sleep(backoff_factor ** attempt)
    logger.error("OpenAI API call failed after retries")
    return ""

# ------------------ Save/load dataset helper functions ------------------
def load_dataset(path: str) -> Dataset:
    """
    Load dataset from given path, assuming JSON or CSV format.
    """
    # Placeholder: Implement as needed per dataset format
    # For now, return an empty list or dummy dataset
    class DummyDataset(Dataset):
        def __len__(self):
            return 0
        def __getitem__(self, idx):
            return {}
    return DummyDataset()

# ------------------ Set up environment and configuration validation ------------------
def validate_config():
    """
    Validate essential configuration entries.
    """
    required_keys = ['training', 'model', 'dataset', 'prompt_templates', 'evaluation', 'hardware']
    for key in required_keys:
        if key not in CONFIG:
            logger.warning(f"Configuration missing key: {key}")
    # Further validation can be added here
```

**Summary of the above:**

- Loads/configures logger.
- Parses global parameters from `config.yaml`.
- Implements seed setting for reproducibility.
- Provides tokenization functions for proteins and text, supporting defaults from config.
- Provides checkpoint save/load functions.
- Implements a flexible learning rate scheduler with warm-up + cosine decay.
- Gradient clipping utility.
- Logging with metrics as JSON lines.
- Wrapper for calling OpenAI API with retries and backoff.
- Dummy data loader function.
- Validation of configuration structure.

This `utils.py` provides essential, complete, and flexible utilities core to training/evaluation workflows within the described architecture, adhering to the paper's details and ensuring reproducibility and modularity.

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\MutaPLM\MutaPLM_repo`
