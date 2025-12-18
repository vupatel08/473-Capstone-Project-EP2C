#!/usr/bin/env python3
# main.py

import os
import sys
import yaml
import numpy as np

import torch

from dataset_loader import DatasetLoader
from model import SimpleMLP
from baseline_methods import (
    RangeHeuristic,
    NormAndThreshold,
    KNNDistanceDetector,
    PCAReconstructionError
)
from trainer import Trainer
from evaluation import Evaluator
from utils import set_seed, load_dataset, get_normalization_thresholds

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset_path = config['dataset']['path']
    window_size = config['dataset'].get('window_size', 4)
    train_split_ratio = config['dataset'].get('train_split_ratio', 0.8)

    model_type = config['model'].get('type', 'SimpleMLP')
    hidden_size = config['model'].get('hidden_size', 32)

    # Load dataset
    dataset_loader = DatasetLoader(
        dataset_path=dataset_path,
        window_size=window_size,
        train_split_ratio=train_split_ratio
    )
    dataset_loader.process_data()
    train_dict, test_dict = dataset_loader.get_train_test()

    X_train = train_dict['train_data']
    y_train_points = train_dict['train_labels']
    X_test = test_dict['test_data']
    y_test_points = test_dict['test_labels']

    # Determine input dimension for models
    input_dim = X_train.shape[1]
    # For univariate with windowing, shape of features is window_size * features
    # For simplicity, we proceed as-is.

    # Normalize data if necessary for models
    # For neural networks, already normalized if datasets set so;
    # will use normalize_data (z-score) for PCA, NN detectors as baseline.
    # For other baselines, normalization not critical unless specified.
    # For consistency, normalize train and test data for PCA, NN, etc.
    from utils import normalize_data
    X_train_norm, norm_params = normalize_data(X_train, method='zscore')
    X_test_norm, _ = normalize_data(X_test, method='zscore', fit_params=norm_params)

    # Initialize and train neural network model
    model = SimpleMLP(input_dim=input_dim, hidden_size=hidden_size)

    # For now, no validation set; using training stats for early stopping
    trainer = Trainer(
        model=model,
        train_data=X_train_norm,
        val_data=None,
        train_labels=y_train_points,
        val_labels=None,
        config=config,
        device='cpu'  # change to 'cuda' if GPU is available
    )
    trainer.train()

    # Extract training errors for thresholding in point-wise detection
    train_scores_nn = model.compute_error(X_train_norm)
    test_scores_nn = model.compute_error(X_test_norm)

    # Determine thresholds based on training set, per configuration
    threshold_method = config['evaluation'].get('thresholds', {}).get('method', 'percentile')
    percentile_value = config['evaluation'].get('thresholds', {}).get('percentile', 95)

    # Simple detection thresholds
    range_thresholds = None
    if threshold_method == 'percentile':
        range_thresholds = {
            'range': np.percentile(train_scores_nn, percentile_value),
            'pca': None,  # Will set later after PCA
            'knn': None,  # Will set later after k-NN
            'l2': None    # Will set later after norm
        }
    else:
        # Add other methods if needed
        range_thresholds = {
            'range': 0,
            'pca': None,
            'knn': None,
            'l2': None
        }

    # 1. Range heuristic baseline
    range_detector = RangeHeuristic(X_train)
    range_detector.fit_threshold(percentile=percentile_value)
    range_scores_test = range_detector.detect(X_test)

    # 2. L2-norm baseline
    norm_detector = NormAndThreshold(X_train)
    norm_detector.fit_threshold(percentile=percentile_value)
    l2_scores_test = norm_detector.detect(X_test)

    # 3. K-NN distance baseline
    k_value = config.get('baseline', {}).get('knn_k', 1)
    knn_detector = KNNDistanceDetector(X_train, k=k_value)
    knn_detector.fit_threshold(percentile=percentile_value)
    knn_scores_test = knn_detector.detect(X_test)

    # 4. PCA reconstruction error
    n_components = 30
    pca_detector = PCAReconstructionError(X_train, n_components=n_components)
    pca_scores_test = pca_detector.detect(X_test)

    # Store thresholds for detection
    range_thresh = np.percentile(train_scores_nn, percentile_value)
    norm_thresh = np.percentile(np.linalg.norm(X_train, axis=1), percentile_value)
    knn_thresh = np.percentile(knn_detector.train_distances, percentile_value)
    pca_thresh = np.percentile(pca_scores_test, percentile_value)

    # Apply thresholds to test scores and get binary labels
    y_pred_range = (range_scores_test > range_detector.threshold).astype(int)
    y_pred_l2 = (l2_scores_test > norm_detector.threshold).astype(int)
    y_pred_knn = (knn_scores_test > knn_detector.threshold).astype(int)
    y_pred_pca = (pca_scores_test > pca_detector.train_error_threshold).astype(int)

    # 5. Neural NN point-wise anomaly detection
    # Threshold: calculate on training scores
    threshold_nn_point = np.percentile(train_scores_nn, percentile_value)
    y_pred_nn_point = (test_scores_nn > threshold_nn_point).astype(int)

    # For range metrics, generate intervals
    def scores_to_intervals(scores: np.ndarray, threshold: float):
        binary = scores > threshold
        intervals = []
        start_idx = None
        for i, val in enumerate(binary):
            if val and start_idx is None:
                start_idx = i
            elif not val and start_idx is not None:
                intervals.append((start_idx, i-1))
                start_idx = None
        if start_idx is not None:
            intervals.append((start_idx, len(binary)-1))
        return intervals

    # Ground truth intervals for range-based metrics
    def get_gt_intervals(labels: np.ndarray):
        intervals = []
        in_a = False
        start_idx = 0
        for i, lbl in enumerate(labels):
            if lbl == 1 and not in_a:
                in_a = True
                start_idx = i
            elif lbl == 0 and in_a:
                intervals.append((start_idx, i-1))
                in_a = False
        if in_a:
            intervals.append((start_idx, len(labels)-1))
        return intervals

    gt_intervals = get_gt_intervals(y_test_points)

    # Initialize evaluators with different detection results
    evaluator_point = Evaluator(
        scores=test_scores_nn,
        labels=y_test_points,
        mode='pointwise',
        thresholds_method=threshold_method,
        percentile=percentile_value
    )
    evaluator_range = Evaluator(
        scores=test_scores_nn,
        labels=y_test_points,
        mode='range',
        thresholds_method=threshold_method,
        percentile=percentile_value
    )

    # Generate binary detection labels from scores
    def apply_threshold(scores: np.ndarray, threshold: float):
        return (scores > threshold).astype(int)

    # Thresholds for detection
    threshold_value_point = np.percentile(train_scores_nn, percentile_value)
    y_pred_nn_point_thresh = apply_threshold(test_scores_nn, threshold_value_point)

    # Generate predicted intervals for range metrics
    pred_intervals = scores_to_intervals(test_scores_nn, threshold_value_point)

    # Evaluate point-wise F1
    pm = evaluator_point.get_pointwise_metrics()
    # Evaluate range-wise metrics
    rm = evaluator_range.get_range_metrics()

    # Output metrics to console
    print("=== Evaluation Results ===")
    print("\n--- Neural Network Baseline (Point-wise) ---")
    print(f"Point-wise F1: {pm['f1']:.4f}")
    print(f"Precision: {pm['precision']:.4f}")
    print(f"Recall: {pm['recall']:.4f}")
    print(f"Threshold: {pm['threshold']:.4f}")

    print("\n--- Range-based Metrics ---")
    print(f"F1: {rm['f1']:.4f}")
    print(f"Range Precision: {rm['precision']:.4f}")
    print(f"Range Recall: {rm['recall']:.4f}")
    print(f"Range Threshold: {rm['threshold']:.4f}")
    # Optional: compute APRC or detailed range metrics here

    # Save metrics if desired
    # e.g., save to a JSON file for record
    import json
    results = {
        'pointwise_f1': pm['f1'],
        'pointwise_precision': pm['precision'],
        'pointwise_recall': pm['recall'],
        'pointwise_threshold': pm['threshold'],
        'range_f1': rm['f1'],
        'range_precision': rm['precision'],
        'range_recall': rm['recall'],
        'range_threshold': rm['threshold']
    }
    with open('results_summary.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
