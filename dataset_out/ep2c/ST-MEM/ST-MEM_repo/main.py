# main.py
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_random_seed
from datasets import ECGDataset
from model import TransformerEncoder, TransformerDecoder
from trainer import ECGPretrainer
from evaluation import Evaluation

@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    # Set seed for reproducibility
    set_random_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset file paths and labels based on dataset paths
    # Here, assuming dataset paths in config, and datasets.py handles data loading
    # For simplicity, assume dataset paths are directory paths with files (.mat, .npy)
    # and labels are loaded within datasets.py (implemented accordingly).
    # Note: User must replace 'path/to/...' with actual dataset paths
    dataset_paths = cfg.dataset.dataset_paths

    # Define data splits: for pretraining, use all data; for downstream, use train/val/test
    # Load all datasets for pretraining
    pretrain_datasets = []
    for dataset_name, path in dataset_paths.items():
        # For preprocessing, we assume datasets.py can handle loading any dataset given directory path
        # and the loading logic is inside datasets.py
        pretrain_datasets.append((dataset_name, path))
    print("Pretraining datasets loaded from paths:", dataset_paths)

    # Create combined unlabeled dataset for pretraining
    # Each dataset should return raw signals, possibly labels if any, but not used in pretraining
    # For simplicity, combine all datasets into a single dataset
    # Here, assuming datasets.py can handle dataset name and path accordingly
    # possibly with a custom dataset that merges multiple datasets
    # Note: Implementation of Dataset merging not shown; assume pretraining Dataset combines all data
    pretrain_dataset = ECGDataset(
        file_paths=[],  # Will be set later after collecting all files
        label_paths=None,
        mode='pretrain',
        config=OmegaConf.to_container(cfg.dataset, resolve=True)
    )

    # Collect all file paths for pretraining
    all_files = []
    for dataset_name, dataset_path in pretrain_datasets:
        # Assume datasets.py can accept directory path and parse files internally
        # Instead, load paths here
        # For this example, assuming user provides full file list; 
        # alternatively, datasets.py can be extended
        # Placeholder: user must extend this part
        pass
    # For demonstration, we just keep empty list as placeholders
    pretrain_dataset.file_paths = all_files

    # Note: Since datasets.py is designed for file path inputs, user must provide proper paths
    # For simplicity, skipping actual file collection code here

    # Set DataLoader for pretraining
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True, num_workers=4)

    # Instantiate model components
    encoder = TransformerEncoder(
        num_layers=cfg.model.encoder_layers,
        embed_dim=cfg.model.encoder_embed_dim,
        n_heads=cfg.model.encoder_heads,
        patch_size=cfg.pretraining.patch_size,
        num_patches=cfg.pretraining.num_patches,
        lead_count=cfg.dataset.lead_count,
        dropout_rate=cfg.model.dropout_rate
    )
    decoder = TransformerDecoder(
        num_layers=cfg.model.decoder_layers,
        embed_dim=cfg.model.decoder_embed_dim,
        n_heads=cfg.model.decoder_heads,
        lead_count=cfg.dataset.lead_count,
        dropout_rate=cfg.model.dropout_rate
    )

    encoder.to(device)
    decoder.to(device)
    # Initialize pretrainer
    pretrainer = ECGPretrainer(encoder, decoder, pretrain_dataset, OmegaConf.to_container(cfg, resolve=True))
    # Load checkpoint if exists
    checkpoint_path = os.path.join('./checkpoints', 'pretrain_latest.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading pretraining checkpoint from {checkpoint_path}")
        pretrainer.load_checkpoint(checkpoint_path)

    # Run pretraining
    print("Starting self-supervised pretraining...")
    pretrainer.run()

    # Save the final encoder and decoder weights after pretraining
    encoder_path = os.path.join('./checkpoints', 'encoder_final.pt')
    decoder_path = os.path.join('./checkpoints', 'decoder_final.pt')
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Pretrained encoder saved to {encoder_path}")
    print(f"Pretrained decoder saved to {decoder_path}")

    # =========================
    # Downstream evaluation: fine-tuning
    # =========================

    # Load downstream datasets
    # Example: PTB-XL for arrhythmia classification
    # User must specify dataset paths and splits in config
    downstream_datasets = {}
    for task_name, task_info in cfg.downstream_tasks.items():
        # Prepare dataset for each task
        dataset_path = dataset_paths.get(task_info['dataset_name'])
        label_name = task_info['label_name']
        mode = 'train'  # load training subset; similarly for validation and test
        dataset_obj = ECGDataset(
            file_paths=[],  # User must load file paths for downstream datasets
            label_paths=None,
            mode=mode,
            config=OmegaConf.to_container(cfg.dataset, resolve=True)
        )
        # User must fill dataset paths
        # For this template, assume data is loaded externally
        downstream_datasets[task_name] = dataset_obj

    # For illustration, only process PTB-XL
    # Load PTB-XL train and val datasets for fine-tuning
    # For real implementation, user must prepare file_paths and labels
    # Here, assuming dataset objects are prepared accordingly

    # Load pretrained encoder weights
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()
    # Attach classifier head
    num_classes = cfg.downstream_tasks['arrhythmia_classification']['num_classes']
    classifier = nn.Linear(cfg.model.encoder_embed_dim, num_classes).to(device)

    # Define optimizer for fine-tuning
    finetune_params = list(encoder.parameters()) + list(classifier.parameters())
    finetune_optimizer = optim.AdamW(finetune_params, lr=cfg.training.learning_rate)

    # Fine-tuning loop
    # Dataset: training set with labels, no masking
    finetune_dataset = downstream_datasets['arrhythmia_classification']
    finetune_loader = DataLoader(finetune_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)

    epochs_ft = 100
    criterion = nn.CrossEntropyLoss()

    print("Starting downstream fine-tuning...")
    for epoch in range(1, epochs_ft + 1):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        progress_bar = tqdm(finetune_loader, desc=f"Fine-tune Epoch {epoch}/{epochs_ft}")
        for batch in progress_bar:
            signals = batch['patches'].to(device)
            labels = batch['label'].to(device)
            # Forward
            with torch.no_grad():
                embeddings = encoder(signals, lead_ids=batch['lead_idx'].to(device))
            # Pooling (e.g., mean pooling over sequence) or [CLS] token if implemented
            pooled = embeddings.mean(dim=1)  # shape: [B, embed_dim]
            logits = classifier(pooled)
            loss = criterion(logits, labels)
            # Backprop
            finetune_optimizer.zero_grad()
            loss.backward()
            finetune_optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch} downstream fine-tuning loss: {total_loss / len(finetune_loader):.4f}")

        # Save checkpoint periodically
        torch.save({
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': finetune_optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join('./checkpoints', f'finetune_epoch_{epoch}.pt'))

    # Evaluate on test set
    test_dataset = downstream_datasets['arrhythmia_classification']  # replace with test set
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)
    encoder.eval()
    classifier.eval()
    all_labels = []
    all_probs = []
    for batch in tqdm(test_loader, desc='Testing'):
        signals = batch['patches'].to(device)
        labels = batch['label'].to(device)
        with torch.no_grad():
            embeddings = encoder(signals, lead_ids=batch['lead_idx'].to(device))
            pooled = embeddings.mean(dim=1)
            logits = classifier(pooled)
            probs = torch.softmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute evaluation metrics
    from sklearn.metrics import roc_auc_score, f1_score
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.argmax(y_prob, axis=1)
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Test AUROC: {auroc:.4f}")
    print(f"Test F1 (macro): {f1_macro:.4f}")

if __name__ == "__main__":
    main()
