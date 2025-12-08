## train.py

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import modules from project
import dataset_loader
import model
import evaluation
import visualization

def main():
    # ---------------------------
    # 1. Load configuration
    # ---------------------------
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset parameters
    dataset_name = cfg['dataset']['dataset_name']
    train_size = cfg['dataset']['train_size']
    val_size = cfg['dataset']['validation_size']
    test_size = cfg['dataset']['test_size']
    freqs_min = cfg['dataset']['frequency_range']['min']
    freqs_max = cfg['dataset']['frequency_range']['max']
    freq_points = cfg['dataset']['frequency_points']
    mesh_dim = cfg['dataset']['discretization_mesh']
    # Model parameters
    model_arch = cfg['model']['architecture']
    encoder_type = cfg['model'].get('encoder', {}).get('type', 'implicit_shape_encoder')
    shape_rep = cfg['model'].get('encoder', {}).get('shape_representation', 'signed_distance_function')
    scalar_dim = 7  # as per description, adjust if needed
    channels = cfg['model'].get('channels', 64)
    depth = cfg['model'].get('depth', 4)
    response_type = cfg['model'].get('response_decoder', {}).get('type', 'velocity_field')
    # Training parameters
    lr = cfg['training'].get('learning_rate', 1e-3)
    batch_size = cfg['training'].get('batch_size', 16)
    max_epochs = cfg['training'].get('epochs', 300)
    early_patience = cfg['training'].get('early_stopping_patience', 20)
    v_loss_weight = cfg['training']['loss_weights'].get('velocity_loss_weight', 0.25)
    f_loss_weight = cfg['training']['loss_weights'].get('response_loss_weight', 0.75)
    # Other
    save_checkpoints = cfg['training'].get('save_checkpoints', True)
    save_best_only = cfg['training'].get('save_best_only', True)

    # ---------------------------
    # 2. Data loading
    # ---------------------------
    print("Loading datasets...")
    # Load datasets: assuming Dataset class supports train/val/test split
    full_dataset = dataset_loader.VibratingPlatesDataset(
        data_dir='./data',  # or adjust
        split='train'
    )
    # To get specific subset sizes, assume Dataset supports splitting
    # Or, if dataset_loader has train/val/test datasets, load accordingly
    # For simplicity, assume train/val/test DataLoaders are created here

    # Load full dataset and create splits
    # Note: in dataset_loader, __init__ has methods to get total samples, so:
    total_samples = len(full_dataset)
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8*total_samples)]
    val_idx = indices[int(0.8*total_samples):int(0.9*total_samples)]
    test_idx = indices[int(0.9*total_samples):]

    # Create subset datasets
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    test_subset = torch.utils.data.Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print("Datasets loaded.")
    # ---------------------------
    # 3. Model initialization
    # ---------------------------
    print(f"Initializing model architecture: {model_arch}")
    # Instantiate model (assumed in model.py)
    net = model.LearningVibrationModel()
    net.to(device)

    # Optional: load checkpoint if resuming
    start_epoch = 1
    best_val_loss = np.inf
    checkpoint_path = './checkpoint_best.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

    # ---------------------------
    # 4. Optimizer and scheduler
    # ---------------------------
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # ---------------------------
    # 5. Loss function setup
    # ---------------------------
    # We define loss functions inline during training

    # ---------------------------
    # 6. Training loop
    # ---------------------------
    print("Starting training...")
    no_grad = torch.no_grad

    # For early stopping
    best_epoch = start_epoch
    no_improve_epochs = 0
    total_batches = len(train_loader)

    for epoch in range(start_epoch, max_epochs +1):
        start_time = time.time()
        net.train()
        total_loss = 0.0
        total_velocity_loss = 0.0
        total_response_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Load batch data and move to device
            shape_sdf = batch['shape_enc'].to(device)               # shape (B, H, W)
            properties = batch['props'].to(device)                  # shape (B, scalar_dim)
            freqs = batch['freqs'].to(device)                       # shape (F,)
            velocity_gt = batch['velocity'].to(device)              # shape (B, F, H, W)
            response_gt = batch['response'].to(device)              # shape (B, F)
            shape_ids = batch['shape_id']                            # list of ids if needed

            # Zero grads
            optimizer.zero_grad()

            # For each sample, evaluate at multiple frequencies:
            # To vectorize, process all (B, F) pairs together
            B, F, H, W = velocity_gt.shape

            # Expand shape encoding and properties for F frequencies
            shape_input = shape_sdf  # shape (B, H, W)
            props = properties       # shape (B, scalar_dim)
            freq_batch = freqs.unsqueeze(0).expand(B, F)   # (B, F)
            freq_flat = freq_batch.reshape(-1)             # (B*F,)

            shape_input_exp = shape_input.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)   # (B*F, H, W)
            props_exp = props.unsqueeze(1).expand(-1, F, -1).reshape(-1, scalar_dim)             # (B*F, scalar_dim)

            # Forward pass via model
            # Depending on architecture, response is velocity field or direct scalar
            if net.response_type == 'velocity_field':
                velocity_pred = net(shape_input_exp, props_exp, freq_flat)
                # velocity_pred: (B*F, 2, H, W)
                # For loss, compute MSE on log velocities
                velocity_pred_log = torch.log(torch.clamp(velocity_pred**2 + 1e-8, min=1e-8))
                velocity_gt_flat = velocity_gt.reshape(-1, 2, H, W)
                velocity_gt_log = torch.log(torch.clamp(velocity_gt_flat**2 + 1e-8, min=1e-8))
                velocity_loss = nn.functional.mse_loss(velocity_pred_log, velocity_gt_log)
                # Derive response: measure the mean of squared velocities over the domain
                pred_response = ((velocity_pred_log**2).mean(dim=[2,3])) * 10  # scaled for decibel scale
                # Normalize true response as per description (already scaled/log)
                response_gt_exp = response_gt.reshape(-1)
                response_gt_mean = response_gt_exp.mean()
                response_gt_std = response_gt_exp.std()
                response_gt_norm = (response_gt_exp - response_gt_mean)/response_gt_std
                pred_response_norm = (pred_response.squeeze() - response_gt_mean)/response_gt_std
                response_loss = nn.functional.mse_loss(pred_response_norm, response_gt_norm)
            else:
                # Response directly predicted (scalar), shape (B*F,)
                pred_response = net(shape_input_exp, props_exp, freq_flat)
                response_gt_exp = response_gt.reshape(-1)
                # Normalize
                mean_resp = response_gt_exp.mean()
                std_resp = response_gt_exp.std()
                response_gt_norm = (response_gt_exp - mean_resp)/std_resp
                pred_response_norm = (pred_response - mean_resp)/std_resp
                response_loss = nn.functional.mse_loss(pred_response_norm, response_gt_norm)
                velocity_loss = torch.tensor(0.0, device=device)

            # Combine losses
            total_batch_loss = v_loss_weight * velocity_loss + f_loss_weight * response_loss
            total_batch_loss.backward()
            # Optional: clip gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_velocity_loss += velocity_loss.item()
            total_response_loss += response_loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_batches
        avg_v_loss = total_velocity_loss / total_batches
        avg_f_loss = total_response_loss / total_batches
        print(f"Epoch [{epoch}/{max_epochs}] - Time: {epoch_time:.2f}s - Loss: {avg_loss:.4f} (Vel: {avg_v_loss:.4f}, Resp: {avg_f_loss:.4f})")

        # ---------------------------
        # 7. Validation
        # ---------------------------
        net.eval()
        val_metrics = {'EMSE': [], 'EMD': [], 'PEAKS': []}
        with torch.no_grad():
            for v_batch in val_loader:
                shape_sdf = v_batch['shape_enc'].to(device)
                properties = v_batch['props'].to(device)
                freqs = v_batch['freqs'].to(device)
                velocity_gt = v_batch['velocity'].to(device)
                response_gt = v_batch['response'].to(device)

                B, F, H, W = velocity_gt.shape
                shape_input = shape_sdf
                props = properties
                freq_batch = freqs.unsqueeze(0).expand(B, F)
                freq_flat = freq_batch.reshape(-1)

                shape_input_exp = shape_input.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)
                props_exp = props.unsqueeze(1).expand(-1, F, -1).reshape(-1, scalar_dim)

                if net.response_type == 'velocity_field':
                    velocity_pred = net(shape_input_exp, props_exp, freq_flat)
                    velocity_pred_log = torch.log(torch.clamp(velocity_pred**2 + 1e-8, min=1e-8))
                    # Convert to response function for metric
                    pred_response = ((velocity_pred_log**2).mean(dim=[2,3])) * 10
                    # Compute metrics
                    # Retrieve ground-truth responses
                    resp_gt = response_gt.reshape(-1)
                    resp_pred = pred_response.squeeze().cpu()
                    # Compute metrics
                    emse_val = evaluation.compute_mse(resp_gt, resp_pred)
                    emd_val = evaluation.compute_emd(resp_gt, resp_pred)
                    peaks_true = evaluation.detect_peaks(resp_gt)
                    peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
                    pe = evaluation.peak_error(peaks_true, peaks_pred, resp_gt, resp_pred)
                else:
                    pred_response = net(shape_input_exp, props_exp, freq_flat)
                    resp_gt = response_gt.reshape(-1).cpu()
                    resp_pred = pred_response.cpu()
                    emse_val = evaluation.compute_mse(resp_gt, resp_pred)
                    emd_val = evaluation.compute_emd(resp_gt, resp_pred)
                    peaks_true = evaluation.detect_peaks(resp_gt)
                    peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
                    pe = evaluation.peak_error(peaks_true, peaks_pred, resp_gt, resp_pred)

                val_metrics['EMSE'].append(emse_val)
                val_metrics['EMD'].append(emd_val)
                val_metrics['PEAKS'].append(pe)

        mean_emse = np.mean(val_metrics['EMSE'])
        mean_emd = np.mean(val_metrics['EMD'])
        mean_pe = np.mean(val_metrics['PEAKS'])
        print(f"Validation - EMSE: {mean_emse:.4f}, EMD: {mean_emd:.4f}, PeakError: {mean_pe:.4f}")

        # Step learning rate scheduler
        scheduler.step(mean_emse)

        # Save checkpoint
        if save_checkpoints:
            is_best = (mean_emse < best_val_loss)
            if is_best:
                best_val_loss = mean_emse
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                print(f"Saved best model at epoch {epoch}")
            elif not save_best_only:
                # Save every epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, f'./checkpoint_epoch_{epoch}.pth')

        # Early stopping
        if mean_emse >= best_val_loss:
            no_improve_epochs += 1
            if no_improve_epochs >= early_patience:
                print(f"No improvement for {early_patience} epochs. Early stopping.")
                break
        else:
            no_improve_epochs = 0

    # ---------------------------
    # 8. Final evaluation on test set
    # ---------------------------
    print("Testing best model...")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    all_EMSE = []
    all_EMD = []
    all_PE = []

    with torch.no_grad():
        for batch in test_loader:
            shape_sdf = batch['shape_enc'].to(device)
            properties = batch['props'].to(device)
            freqs = batch['freqs'].to(device)
            velocity_gt = batch['velocity'].to(device)
            response_gt = batch['response'].to(device)

            B, F, H, W = velocity_gt.shape
            shape_input = shape_sdf
            props = properties
            freq_batch = freqs.unsqueeze(0).expand(B, F)
            freq_flat = freq_batch.reshape(-1)

            shape_input_exp = shape_input.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)
            props_exp = props.unsqueeze(1).expand(-1, F, -1).reshape(-1, scalar_dim)

            if net.response_type == 'velocity_field':
                velocity_pred = net(shape_input_exp, props_exp, freq_flat)
                velocity_pred_log = torch.log(torch.clamp(velocity_pred**2 + 1e-8, min=1e-8))
                resp_pred = ((velocity_pred_log**2).mean(dim=[2,3])) * 10
            else:
                resp_pred = net(shape_input_exp, props_exp, freq_flat)

            # Retrieve ground truth responses
            resp_gt = response_gt.reshape(-1).cpu()
            resp_pred = resp_pred.cpu()

            # Compute metrics
            emse_val = evaluation.compute_mse(resp_gt, resp_pred)
            emd_val = evaluation.compute_emd(resp_gt, resp_pred)
            peaks_true = evaluation.detect_peaks(resp_gt)
            peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
            pe = evaluation.peak_error(peaks_true, peaks_pred, resp_gt, resp_pred)

            all_EMSE.append(emse_val)
            all_EMD.append(emd_val)
            all_PE.append(pe)

    print(f"Test Results:\nEMSE: {np.mean(all_EMSE):.4f} ± {np.std(all_EMSE):.4f}")
    print(f"EMD: {np.mean(all_EMD):.4f} ± {np.std(all_EMD):.4f}")
    print(f"Peak Error: {np.mean(all_PE):.4f} ± {np.std(all_PE):.4f}")

    # Optional: generate visualization of example responses
    # For illustration, pick one sample
    # Here, could call visualization functions
    # For brevity, omitted

if __name__ == '__main__':
    main()
