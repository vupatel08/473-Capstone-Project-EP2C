## evaluation.py
"""
This module provides functions to compute evaluation metrics during meta-training and final evaluation.
Metrics implemented:
- Classification accuracy (success rate)
- Rank correlation (Kendall's tau) for regression or prediction success ranking

It is designed to handle model predictions and true labels in NumPy or JAX device array formats.
"""

import numpy as np
from scipy.stats import kendalltau

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute evaluation metrics from model predictions and true labels.

    Args:
        predictions (np.ndarray): Model outputs.
            For classification: shape (batch_size, num_classes), e.g., logits or probabilities.
            For regression or ranking tasks: shape (batch_size,) scalar predictions.
        targets (np.ndarray): True labels.
            For classification: integer class labels, shape (batch_size,).
            For regression or ranking: real-valued scalar labels, shape (batch_size,).

    Returns:
        dict: Dictionary containing evaluation metrics:
            - 'accuracy': float, success rate for classification tasks.
            - 'kendall_tau': float, Kendall's rank correlation coefficient.
            - 'p_value': float, p-value associated with Kendall's tau.
    """
    metrics = {}

    # Convert inputs to numpy arrays if they are JAX DeviceArrays
    preds = np.array(predictions)
    truths = np.array(targets)

    # Determine if the task is classification or regression based on preds shape
    if preds.ndim > 1:
        # Assume classification if predictions are logits/probabilities
        predicted_labels = preds.argmax(axis=-1)
    else:
        # For scalar predictions, treat as regression or ranking
        predicted_labels = preds

    # Compute accuracy for classification
    correct_predictions = np.sum(predicted_labels == truths)
    accuracy = correct_predictions / len(truths)
    metrics['accuracy'] = accuracy

    # Compute Kendall's tau correlation coefficient
    # Flatten arrays in case of higher dimensions
    tau, p_value = kendalltau(preds.flatten(), truths.flatten())
    metrics['kendall_tau'] = tau
    metrics['p_value'] = p_value

    return metrics
