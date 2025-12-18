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
