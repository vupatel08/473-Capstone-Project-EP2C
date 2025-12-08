## main.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from dataset_loader import get_dataset
from model import ResNetEncoder, DeepSetEncoder
from loss import compute_total_loss
from trainer import CareTrainer
from evaluation import evaluate_linear_probe, visualize_trajectories, plot_cosine_histogram

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Set device, seeds for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Prepare Dataset
    train_dataset, test_dataset = get_dataset(cfg)  # from dataset_loader.py
    batch_size = cfg['training'].get('batch_size', 256)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = (torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        if test_dataset is not None else None)

    # 2. Instantiate Model
    model_type = cfg['model'].get('type', 'resnet50')
    embedding_dim = cfg['model'].get('embedding_dim', 128)
    use_projection = cfg['model'].get('projection_head', True)

    if model_type == 'resnet50':
        model = ResNetEncoder({'embedding_dim': embedding_dim, 'projection_head': use_projection})
    elif model_type == 'deepset':
        model = DeepSetEncoder({'embedding_dim': embedding_dim})
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(device)

    # 3. Setup Optimizer and Scheduler
    train_cfg = cfg['training']
    lr = train_cfg.get('learning_rate', 1e-3)
    wd = train_cfg.get('weight_decay', 1e-6)
    opt_name = train_cfg.get('optimizer', 'Adam').lower()
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {opt_name} not supported.")

    # Optional scheduler
    num_epochs = train_cfg.get('epochs', 400)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 4. Setup Loss components
    lambda_eq = train_cfg.get('lambda_equiv', 0.001)
    temperature_infonce = train_cfg.get('temperature_infonce', 0.5)
    temperature_equiv = train_cfg.get('temperature_equiv', 0.1)
    batch_splits = train_cfg.get('batch_splits', 16)

    # 5. Instantiate the trainer object
    trainer = CareTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=cfg,
        lambda_eq=lambda_eq,
        temperature_infonce=temperature_infonce,
        temperature_equiv=temperature_equiv,
        batch_splits=batch_splits
    )

    # 6. Run training loops
    for epoch in range(1, num_epochs + 1):
        print(f"\nStarting epoch {epoch}/{num_epochs}")
        trainer.train_one_epoch(epoch)

        # Step scheduler
        trainer.scheduler.step()

        # 7. Evaluation and visualization at intervals
        if epoch % cfg['evaluation'].get('eval_interval', 10) == 0 or epoch == num_epochs:
            print(f"\nEvaluation at epoch {epoch}:\n")
            # 7a. Linear probing
            if trainer.test_loader is not None:
                probe_results = evaluate_linear_probe(trainer.model, trainer.test_loader, cfg['dataset']['name'])
                print(f"Linear probe accuracy: {probe_results['top1_acc']:.2f}%")
            # 7b. Embedding trajectories visualization
            # Select a sample from test set (or train)
            try:
                sample_idx = 0
                sample_data = None
                if hasattr(train_dataset, '__getitem__'):
                    sample_data = train_dataset[0]
                elif hasattr(test_dataset, '__getitem__'):
                    sample_data = test_dataset[0]
                else:
                    sample_data = None
            except:
                sample_data = None
            if sample_data is not None:
                # Assume for images or proteins
                if isinstance(sample_data, dict):
                    x_sample = sample_data['x']
                    label_sample = sample_data.get('label', '')
                else:
                    x_sample = sample_data[0]
                    label_sample = ''
                # Define a small rotation sequence for visualization
                def small_rotation_func(x):
                    angle_deg = np.random.uniform(-cfg['training']['augmentations'].get('rotation_small', 5))
                    rot_mat = None
                    if hasattr(trainer, 'model'):
                        # For image: rotation in 2D
                        # for point clouds: use generated rot_mat
                        # but here, just a placeholder: no rotation for images
                        pass
                    # For protein point clouds:
                    # rotation matrix implementation
                    # For simplicity, pass x unchanged (or implement rotation if data is point cloud)
                    return x
                # Generate trajectory visualization
                visualize_trajectories(
                    trainer.model,
                    x_sample,
                    augmentation_list=[small_rotation_func],
                    label=label_sample,
                    save_path=os.path.join(cfg['save'].get('logs_path', './logs'), 'trajectory_epoch_{}.png'.format(epoch))
                )
            # 7c. Cosine similarity histograms
            # For simplicity, we skip detailed implementation here, assuming routine calls
            # e.g.,
            # plot_cosine_histogram(z_pairs=[(z1, z2)], title='Pose pairs at epoch {}'.format(epoch),
            #                       save_path=os.path.join(cfg['save']['logs_path'], 'cosine_hist_epoch_{}.png'.format(epoch)))

            # 7d. Save checkpoint
            trainer.save_checkpoint(epoch)

    print("Training completed. Final model saved.")

if __name__ == "__main__":
    main()
