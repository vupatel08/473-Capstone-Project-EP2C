## utils.py
import numpy as np
from scipy.special import expit  # Sigmoid function
from sklearn.metrics import roc_auc_score
import torch
from typing import List, Tuple, Dict
import torch.nn.functional as F

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to the [0, 1] range using min-max scaling.
    Handles negative scores such as log probabilities appropriately.
    
    Args:
        scores (List[float]): Raw detector scores or scores to normalize.
    
    Returns:
        List[float]: Normalized scores scaled between 0 and 1.
    """
    scores_arr = np.array(scores)
    min_score = np.min(scores_arr)
    max_score = np.max(scores_arr)
    if max_score - min_score < 1e-8:
        # Prevent division by zero; if all scores identical, return 0.5 for all
        return [0.5 for _ in scores]
    normalized = (scores_arr - min_score) / (max_score - min_score)
    return normalized.tolist()

def compute_preference(p1_score: float, p2_score: float, scale: str = 'log_prob') -> float:
    """
    Compute the probability that response 1 (w) is preferred over response 2 (l),
    based on their scores, using the Bradley-Terry model.
    
    Args:
        p1_score (float): Score for response w.
        p2_score (float): Score for response l.
        scale (str): Score type, default 'log_prob' indicating scores are log probabilities.
                     If 'prob', scores are assumed in [0,1].
    
    Returns:
        float: Preference probability p(y_w > y_l).
    """
    if scale == 'log_prob':
        # Assume p1_score and p2_score are in the log domain
        diff = p1_score - p2_score
        return expit(diff)  # σ(diff)
    elif scale == 'prob':
        # Scores are probabilities in [0, 1]
        diff = p1_score - p2_score
        return expit(diff)
    else:
        raise ValueError(f"Unknown scale: {scale}")

def preference_label(score_w: float, score_l: float) -> int:
    """
    Assign binary preference label: 1 if y_w is preferred over y_l, else 0.
    
    Args:
        score_w (float): Score (detector or log prob) for response w.
        score_l (float): Score for response l.
    
    Returns:
        int: 1 if w preferred, 0 otherwise.
    """
    return 1 if score_w > score_l else 0

def preference_probability(score_w: float, score_l: float) -> float:
    """
    Return the preference probability that y_w is preferred over y_l.
    Wraps compute_preference with scale='log_prob' by default.
    
    Args:
        score_w (float): Score for response w.
        score_l (float): Score for response l.
    
    Returns:
        float: Probability that y_w is preferred over y_l.
    """
    return compute_preference(score_w, score_l, scale='log_prob')

def sample_batch(dataset: List[Tuple], batch_size: int) -> List[Tuple]:
    """
    Randomly sample a batch of data tuples from the dataset.
    
    Args:
        dataset (List[Tuple]): Dataset of data points, e.g., (prompt, response, score).
        batch_size (int): Number of samples to draw.
    
    Returns:
        List[Tuple]: Randomly sampled batch.
    """
    import random
    return random.sample(dataset, min(batch_size, len(dataset)))

def generate_samples(prompt: str, model, max_tokens: int, temperature: float, top_p: float) -> str:
    """
    Generate a continuation from a model given a prompt.
    
    Args:
        prompt (str): Input prompt string.
        model: Model wrapper with generate method.
        max_tokens (int): Max number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
    
    Returns:
        str: Generated text continuation.
    """
    generated_text = model.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return generated_text

def compute_auroc(labels: List[int], scores: List[float]) -> float:
    """
    Compute the AUROC score given binary labels and scores.
    
    Args:
        labels (List[int]): Ground truth labels (0 or 1 or other binary labels).
        scores (List[float]): Scores assigned by detector.
    
    Returns:
        float: AUROC score.
    """
    return roc_auc_score(labels, scores)

def compute_perplexity(texts: List[str], model, tokenizer) -> float:
    """
    Compute the average perplexity over a list of texts given a language model.
    Uses negative log likelihoods from the model head.
    
    Args:
        texts (List[str]): List of text strings.
        model: Language model for evaluation.
        tokenizer: Corresponding tokenizer.
    
    Returns:
        float: Average perplexity over the texts.
    """
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = encodings['input_ids'].to(next(model.parameters()).device)
            attention_mask = encodings['attention_mask'].to(next(model.parameters()).device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            # Negative log likelihood
            neg_log_likelihood = outputs.loss * input_ids.shape[1]
            total_nll += neg_log_likelihood.item()
            total_tokens += input_ids.shape[1]
    if total_tokens == 0:
        return float('nan')
    avg_nll = total_nll / total_tokens
    perplexity = np.exp(avg_nll)
    return perplexity
