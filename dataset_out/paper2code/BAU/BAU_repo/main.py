# main.py

import os
import torch
import numpy as np
import random
from tqdm import tqdm

from utils import set_seed
from dataset_loader import DatasetLoader
from model import Model
from losses import AlignmentLoss, compute_uniformity, compute_domain_uniformity
from prototype import PrototypeBank
from trainer import Trainer
from evaluation import evaluate_metrics
from config import cfg

def main():
    # 1. Load Configurations and Set Environment
    seed = cfg['misc'].get('seed', 42)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Prepare DatasetLoader
    dataset_paths = cfg['dataset']['datasets_paths']
    batch_size = cfg['training'].get('batch_size', 64)
    augmentation_prob = cfg['training'].get('augmentation_probability', 0.5)
    dataset_loader = DatasetLoader(
        dataset_paths=dataset_paths,
        split=cfg['dataset'].get('training_split', 'train'),
        batch_size=batch_size,
        augment_prob=augmentation_prob,
        num_identities=64,
        instances_per_id=4,
        augmentations_config=cfg['augmentation']
    )

    # 3. Prepare evaluation datasets (assuming required as per protocols)
    # For simplicity, assuming only one evaluation dataset with name 'test' in dataset_loader
    # For protocol evaluation, you need to set this accordingly.
    # e.g.,
    # eval_datasets_info = {
    #     'query': dataset_loader.query_dataset,
    #     'gallery': dataset_loader.gallery_dataset
    # }
    # But here, we focus on training loop; evaluation can be called separately.
    
    # 4. Instantiate Model
    model = Model(
        backbone_name=cfg['model'].get('backbone', 'resnet50'),
        feature_dim=cfg['model'].get('feature_dim', 512),
        normalize_features=True
    ).to(device)

    # 5. Initialize Loss Modules
    # For simplicity, only triplet loss wrapper provided
    from losses import TripletLossWrapper, CrossEntropyLossWrapper
    triplet_loss_fn = TripletLossWrapper(margin=cfg['training'].get('triplet_margin',0.3))
    ce_loss_fn = CrossEntropyLossWrapper()

    # 6. Initialize Prototype Bank
    # Assume total number of classes is known; here, mock with a fixed number or obtained from dataset info
    total_classes = 10000  # Placeholder; replace with actual total number of classes in datasets
    prototype_bank = PrototypeBank(
        num_classes=total_classes,
        feature_dim=cfg['model'].get('feature_dim', 512),
        momentum=cfg['training'].get('prototype_momentum', 0.999),
        device=device
    )
    
    # 7. Setup losses object container
    losses = {
        'triplet': triplet_loss_fn,
        'classification': ce_loss_fn
    }

    # 8. Instantiate Trainer
    trainer_obj = Trainer(
        model=model,
        losses=losses,
        prototypes=prototype_bank,
        data_loader=dataset_loader,
        config=cfg
    )

    # 9. Training Loop
    total_epochs = cfg['training'].get('epochs', 60)
    for epoch in range(total_epochs):
        print(f"\nStarting Epoch {epoch+1}/{total_epochs}")
        trainer_obj._current_epoch = epoch
        trainer_obj.train_epoch()

        # 10. Optional: Evaluate periodically, e.g., every 5 epochs
        if (epoch+1) % 5 == 0 or (epoch+1) == total_epochs:
            # Here, you'd load your validation/test dataset
            # For demonstration, assume placeholder:
            # test_datasets_info = {'query': query_dataset, 'gallery': gallery_dataset}
            # results = evaluate_metrics(model, test_datasets_info, protocol=cfg['evaluation'].get('protocol', 'Protocol-3'))
            # print(f"Validation results at epoch {epoch+1}: {results}")
            pass

    # 11. Save final model
    save_path = cfg['misc'].get('save_model_path', './results/model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved at {save_path}")

    # 12. Final evaluation call (if desired)
    # final_results = evaluate_metrics(model, test_datasets_info, protocol=cfg['evaluation'].get('protocol', 'Protocol-3'))
    # print(f"Final evaluation results: {final_results}")

if __name__ == '__main__':
    main()
