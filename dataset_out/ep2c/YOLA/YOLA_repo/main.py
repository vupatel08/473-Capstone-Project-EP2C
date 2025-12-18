## main.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import DatasetLoader
from model import DetectionModel
from evaluation import Evaluation

def main():
    # 1. Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set device
    gpus = config.get('hardware', {}).get('gpus', 1)
    device = torch.device('cuda' if torch.cuda.is_available() and gpus > 0 else 'cpu')
    if torch.cuda.device_count() > 1 and gpus > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print(f"Using device: {device}")

    # 3. Prepare Dataset Loaders
    dataset_name = config['dataset'].get('dataset_name', 'ExDark')
    input_size = config['dataset'].get('input_size', 608)
    dataset_path = config['dataset'].get('dataset_path', './datasets') # Ensure dataset path set
    synthetic_illumination = config['dataset'].get('synthetic_illumination', True)
    augmentation = config['dataset'].get('augmentation', {})

    # Load training dataset
    train_dataset = DatasetLoader(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        input_size=input_size,
        train_split_ratio=0.8,
        val_split_ratio=0.2,
        synthetic_illumination=synthetic_illumination,
        augmentation=augmentation,
        mode='train'
    )

    # Load validation dataset
    val_dataset = DatasetLoader(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        input_size=input_size,
        train_split_ratio=0.8,
        val_split_ratio=0.2,
        synthetic_illumination=False,
        augmentation={},
        mode='test'
    )

    batch_size = config['training'].get('batch_size', 16)
    num_workers = 4  # Adjust if needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # 4. Initialize Model
    model_config = config['model']
    model = DetectionModel(model_config)
    model = model.to(device)
    if torch.cuda.device_count() > 1 and gpus > 1:
        model = nn.DataParallel(model)  # For multi-GPU

    # 5. Define optimizer and scheduler
    learning_rate = config['training'].get('learning_rate', 0.001)
    weight_decay = config['training'].get('weight_decay', 5e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    step_size = config['training'].get('step_size', 10)
    gamma = config['training'].get('gamma', 0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    total_epochs = config['training'].get('epochs', 24)
    save_every = config['training'].get('save_model_every', 5)
    eval_every = config['training'].get('evaluation_epochs', 1)
    detection_loss_weight = config['loss'].get('detection_loss_weight', 1.0)
    ii_loss_weight = config['loss'].get('ii_loss_weight', 0.01)
    ii_loss_scale = config['loss'].get('ii_loss_scale', 1.0)
    beta = 1.0  # threshold for II Loss masking

    # Initialize training state
    best_mAP = 0.0
    model.train()

    # 6. Training Loop
    for epoch in range(total_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}") as pbar:
            for batch in pbar:
                images = batch['image'].to(device)
                targets = batch['targets']
                pair_images_list = batch['pair_image']
                # Generate paired batches for II Loss
                if synthetic_illumination:
                    pair_imgs = []
                    for p_img in pair_images_list:
                        if p_img is None:
                            pair_imgs.append(images)
                        else:
                            pair_imgs.append(p_img.to(device))
                    pair_batch = torch.stack(pair_imgs, dim=0)
                else:
                    pair_batch = None

                # Forward pass original images
                detections, features_orig = model(images)
                # Forward pass pairs to get features for II Loss
                if pair_batch is not None:
                    with torch.no_grad():
                        _, features_pair = model(pair_batch)
                else:
                    features_pair = None

                # Compute detection loss (placeholder - replace with actual detection criterion)
                loss_det = compute_detection_loss(detections, targets)

                # Compute II Loss
                if features_orig is not None and features_pair is not None:
                    batch_ii_loss = 0.0
                    for i in range(features_orig.size(0)):
                        diff = features_orig[i] - features_pair[i]
                        norm_diff = torch.norm(diff)
                        mask = (norm_diff < beta).float()
                        loss_i = (mask * (diff ** 2)).mean()
                        batch_ii_loss += loss_i
                    batch_ii_loss /= features_orig.size(0)
                else:
                    batch_ii_loss = 0.0

                total_loss = detection_loss_weight * loss_det + ii_loss_weight * batch_ii_loss * ii_loss_scale

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Enforce zero-mean kernels
                with torch.no_grad():
                    for kernel in model.module.kernels if hasattr(model, 'module') else model.kernels:
                        mean_w = torch.mean(kernel)
                        kernel -= mean_w

                epoch_loss += total_loss.item()
                pbar.set_postfix(loss=f"{total_loss.item():.3f}", det_loss=f"{loss_det:.3f}", ii_loss=f"{batch_ii_loss:.3f}")

        # Step LR scheduler
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f'./checkpoints/model_epoch_{epoch+1}.pth'
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        # Periodic evaluation
        if (epoch + 1) % eval_every == 0:
            evaluator = Evaluation(model, val_dataset, config)
            metrics = evaluator.run()  # returns dict with mAP etc.
            print(f"Validation Results at epoch {epoch+1}: {metrics}")

            # Save best model
            if metrics.get('mAP', 0) > best_mAP:
                best_mAP = metrics['mAP']
                torch.save(model.state_dict(), './checkpoints/best_model.pth')

        print(f"Epoch [{epoch+1}/{total_epochs}] Complete. Avg Loss: {epoch_loss/len(train_loader):.3f}")

    # 9. Final evaluation with best model
    print("Training complete. Loading best model for final evaluation...")
    model.load_state_dict(torch.load('./checkpoints/best_model.pth'))
    model.eval()
    final_evaluator = Evaluation(model, val_dataset, config)
    final_metrics = final_evaluator.run()
    print("Final Evaluation metrics:", final_metrics)

def collate_fn(batch):
    # Custom collate function to handle variable annotations
    images = [item['image'] for item in batch]
    targets = [item['targets'] for item in batch]
    pair_images = [item['pair_image'] for item in batch]
    meta = [item['metadata'] for item in batch]
    images = torch.stack(images, dim=0)
    return {'image': images, 'targets': targets, 'pair_image': pair_images, 'metadata': meta}

def compute_detection_loss(detections, targets):
    # Placeholder: replace with actual detection loss (e.g., YOLO, TOOD)
    # For demonstration, use L1 with zeros
    return nn.L1Loss()(detections, torch.zeros_like(detections))

if __name__ == '__main__':
    main()
