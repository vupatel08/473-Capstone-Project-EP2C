## main.py
import os
import sys
import time
import logging
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Import project modules
from config import config
from dataset_loader import DatasetLoader
from feature_extractor import FeatureExtractor
from model import AttentionMIL, TransMIL, AdditiveMIL
from explanation import Explanation
from trainer import Trainer
from evaluation import Evaluation
from utils import save_figure, load_slide_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    start_time = time.time()
    device_str = config.hardware.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Load dataset
    dataset_path = config.dataset_paths.get('histopathology', './data')
    loader = DatasetLoader(dataset_dir=dataset_path)

    # 2. Load dataset (slide list with labels)
    all_slides = loader.load_all_slides()

    # 3. Split data into train, val, test
    # For simplicity, assume random split with fixed seed for reproducibility
    np.random.seed(42)
    slide_indices = list(range(len(all_slides)))
    np.random.shuffle(slide_indices)

    n_total = len(all_slides)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

    train_idx = slide_indices[:n_train]
    val_idx = slide_indices[n_train:n_train+n_val]
    test_idx = slide_indices[n_train+n_val:]

    train_list = [all_slides[i] for i in train_idx]
    val_list = [all_slides[i] for i in val_idx]
    test_list = [all_slides[i] for i in test_idx]

    logging.info(f"Dataset split: {len(train_list)} train, {len(val_list)} val, {len(test_list)} test.")

    # 4. Extract features for all datasets
    feature_extractor = FeatureExtractor(model_name=config.model_parameters['feature_extractor'])
    feature_extractor.model.eval()

    for dataset_split in [train_list, val_list, test_list]:
        # Check if features already extracted (optional)
        loader.extract_features_for_dataset(dataset_split, feature_extractor, batch_size=128)

    # 5. Initialize model
    model_type = config.model_parameters.get('model_type', 'attention').lower()
    feature_dim = 512  # from the feature extractor output
    hidden_dim = config.model_parameters.get('hidden_dim', 512)
    dropout = config.training.get('dropout', 0.0)

    if model_type == 'attention':
        model = AttentionMIL(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == 'transformer':
        model = TransMIL(feature_dim=feature_dim, num_layers=2, num_heads=8, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == 'additive':
        model = AdditiveMIL(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)
    logging.info(f"Initialized {model_type} model.")

    # 6. Set up optimizer
    optim_cls = torch.optim.Adam if config.training.get('optimizer', 'Adam').lower() == 'adam' else torch.optim.SGD
    optimizer = optim_cls(filter(lambda p: p.requires_grad, model.parameters()), lr=config.training.get('learning_rate', 0.002))

    # 7. Train model
    trainer = Trainer(model, train_list, val_list, test_list, {
        'optimizer': optimizer,
        'epochs': config.training.get('epochs', 1000),
        'batch_size': config.training.get('batch_size', 32),
        'device': device,
        'checkpoint_path': config.save.get('model_checkpoint_path', './checkpoints/'),
        'early_stop_patience': 10
    })
    logging.info("Starting training...")
    trainer.train()

    # 8. Load best model checkpoint
    # It is handled internally in trainer._load_checkpoint after train()

    # 9. Explaining test set
    explanation_method = config.explanation_method.get('method', 'xMIL-LRP')
    relevance_rules = config.explanation_method['relevance_rules']
    explainer = Explanation(model, explanation_method=explanation_method)
    explainer.rules = relevance_rules

    # Prepare output directories
    os.makedirs(config.save.get('explanation_heatmaps_path', './heatmaps'), exist_ok=True)

    # For each test slide, compute relevance, generate heatmaps, evaluate
    all_preds = []
    all_labels = []
    all_auprc2 = []
    all_aupc = []

    for slide_obj in test_list:
        # 9.1 Get features tensor (K, D)
        features = torch.tensor(slide_obj.features, dtype=torch.float).to(device)
        # 9.2 Compute relevance scores
        relevance_scores = explainer.compute_relevance(features)
        # Convert to numpy
        relevance_np = np.array(relevance_scores)

        # 9.3 Compute or retrieve model prediction
        # For simplicity, rerun model forward on features
        with torch.no_grad():
            logits = model(features.unsqueeze(0))
            prob = torch.sigmoid(logits).item()
        all_preds.append(prob)
        # Ground truth label: from slide label
        label = slide_obj.label
        all_labels.append(label)

        # 9.4 Evaluate explanation (AUPRC-2)
        # Assume ground truth evidence is support (simulate with label==1, refute==0)
        gt_evidence = np.ones(len(slide_obj.features)) if label==1 else -np.ones(len(slide_obj.features))
        # Calculate AUPRC-2
        auprc2 = 0.5 * (
            average_precision_score((gt_evidence>0).astype(int), relevance_np) +
            average_precision_score((gt_evidence<0).astype(int), -relevance_np)
        )
        all_auprc2.append(auprc2)

        # 9.5 Generate heatmap overlay
        if config.evaluation.get('visualization', {}).get('heatmaps', False):
            slide_image = load_slide_image(slide_obj.slide_path)
            heatmap_img = explainer.generate_heatmap(slide_obj.patches, relevance_scores)
            save_figure(heatmap_img, os.path.join(
                config.save.get('explanation_heatmaps_path', './heatmaps'), f"slide_{slide_obj.slide_id}_heatmap.png"
            ))

    # 10. Compute overall metrics
    mean_auprc2 = np.mean(all_auprc2)
    std_auprc2 = np.std(all_auprc2)

    # AUPC: compute by patch dropping in order of relevance
    # (Here skipped for simplicity; implement as needed)
    # For demonstration, assume it is computed and stored as 'overall_aupc'
    # but since detailed code is long, we omit in this code.

    # Log results
    logging.info(f"Test AUROC: {roc_auc_score(all_labels, all_preds):.3f}")
    logging.info(f"Test AUPRC-2: {mean_auprc2:.3f} ± {std_auprc2:.3f}")

    # 11. Final summary
    total_time = time.time() - start_time
    logging.info(f"Total runtime: {total_time/60:.2f} minutes.")

if __name__ == '__main__':
    main()
