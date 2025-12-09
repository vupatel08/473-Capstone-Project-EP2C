## main.py
import os
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
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
    freq_range = cfg['dataset']['frequency_range']
    freq_points = cfg['dataset']['frequency_points']
    discretization_mesh = cfg['dataset'].get('discretization_mesh', '181x121')
    mesh_dims = tuple(map(int, discretization_mesh.split('x')))
    freq_min = float(freq_range.get('min',1))
    freq_max = float(freq_range.get('max',300))
    freqs = np.linspace(freq_min, freq_max, freq_points)

    # Model parameters
    model_arch = cfg['model']['architecture']
    encoder_type = cfg['model'].get('encoder', {}).get('type', 'implicit_shape_encoder')
    shape_representation = cfg['model'].get('encoder', {}).get('shape_representation', 'signed_distance_function')
    scalar_dim = 7  # as indicated
    channels = cfg['model'].get('channels', 64)
    depth = cfg['model'].get('depth', 4)
    response_type = cfg['model'].get('response_decoder', {}).get('type', 'velocity_field')
    
    # Training parameters
    lr = cfg['training'].get('learning_rate', 1e-3)
    batch_size = cfg['training'].get('batch_size', 16)
    max_epochs = cfg['training'].get('epochs', 300)
    early_patience = cfg['training'].get('early_stopping_patience', 20)
    loss_weights = cfg['training']['loss_weights']
    v_loss_weight = loss_weights.get('velocity_loss_weight', 0.25)
    f_loss_weight = loss_weights.get('response_loss_weight', 0.75)
    save_checkpoints = cfg['training'].get('save_checkpoints', True)
    save_best_only = cfg['training'].get('save_best_only', True)

    # ---------------------------
    # 2. Dataset loading / generation
    # ---------------------------
    print("Loading datasets...")
    # Load or generate datasets
    # Here, we'll assume pre-generated datasets (more realistic).
    # If generating, implement a call to dataset_loader.generate_dataset()
    full_train_dataset = dataset_loader.VibratingPlatesDataset(
        data_dir='./data',
        split='train'
    )
    test_dataset = dataset_loader.VibratingPlatesDataset(
        data_dir='./data',
        split='test'
    )
    # For validation, take subset from train
    total_train_samples = len(full_train_dataset)
    indices = np.arange(total_train_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_idx = indices[:int(0.1*total_train_samples)]  # 10% for validation
    train_idx = indices[int(0.1*total_train_samples):]
    train_subset = torch.utils.data.Subset(full_train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_train_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Datasets loaded.")

    # ---------------------------
    # 3. Model instantiation
    # ---------------------------
    print(f"Initializing model: {model_arch}")
    net = model.LearningVibrationModel()
    net.to(device)

    # Load checkpoint if available
    checkpoint_path = './checkpoint_best.pth'
    start_epoch = 1
    best_val_loss = np.inf
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        chkpt = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(chkpt['model_state_dict'])
        start_epoch = chkpt['epoch'] + 1
        best_val_loss = chkpt['best_val_loss']

    # ---------------------------
    # 4. Optimizer and scheduler
    # ---------------------------
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # ---------------------------
    # 5. Training loop
    # ---------------------------
    print("Starting training...")
    for epoch in range(start_epoch, max_epochs+1):
        start_time = time.time()
        net.train()
        total_loss = 0.0
        total_vloss = 0.0
        total_floss = 0.0
        for batch in train_loader:
            shape_sdf = batch['shape_enc'].to(device)        # (B,H,W)
            properties = batch['props'].to(device)           # (B,7)
            freqs_np = batch['freqs'].to(device)             # (F,)
            velocities_gt = batch['velocity'].to(device)     # (B,F,H,W)
            responses_gt = batch['response'].to(device)      # (B,F)
            B, F, H, W = velocities_gt.shape

            optimizer.zero_grad()

            # Prepare expanded inputs for all frequencies in batch
            shape_input_exp = shape_sdf.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)  # (B*F,H,W)
            props_exp = properties.unsqueeze(1).expand(-1, F, -1).reshape(-1, 7)
            freq_flat = freqs_np.unsqueeze(0).expand(B, -1).reshape(-1)

            if net.response_type == 'velocity_field':
                pred_vel = net(shape_input_exp, props_exp, freq_flat)  # (B*F,2,H,W)
                # Log scale for loss
                pred_vel_log = torch.log(torch.clamp(pred_vel**2 + 1e-8, min=1e-8))
                gt_vel_flat = velocities_gt.reshape(-1, 2, H, W)
                gt_vel_log = torch.log(torch.clamp(gt_vel_flat**2 + 1e-8, min=1e-8))
                velocity_loss = nn.functional.mse_loss(pred_vel_log, gt_vel_log)

                # Derive frequency response (mean squared velocity)
                pred_resp = ((pred_vel_log**2).mean(dim=[2,3])) * 10  # scaled to dB
                resp_gt_flat = responses_gt.reshape(-1)
                resp_mean = resp_gt_flat.mean()
                resp_std = resp_gt_flat.std()
                resp_gt_norm = (resp_gt_flat - resp_mean)/resp_std
                resp_pred_norm = (pred_resp.squeeze() - resp_mean)/resp_std
                response_loss = nn.functional.mse_loss(resp_pred_norm, resp_gt_norm)
            else:
                # Direct scalar response prediction
                pred_resp = net(shape_input_exp, props_exp, freq_flat)  # (B*F,)
                resp_gt_flat = responses_gt.reshape(-1)
                mean_resp = resp_gt_flat.mean()
                std_resp = resp_gt_flat.std()
                resp_gt_norm = (resp_gt_flat - mean_resp)/std_resp
                resp_pred_norm = (pred_resp - mean_resp)/std_resp
                response_loss = nn.functional.mse_loss(resp_pred_norm, resp_gt_norm)
                velocity_loss = torch.tensor(0., device=device)

            total_batch_loss = v_loss_weight * velocity_loss + f_loss_weight * response_loss
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_vloss += velocity_loss.item()
            total_floss += response_loss.item()
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch}/{max_epochs}] in {epoch_time:.1f}s - Loss: {total_loss/len(train_loader):.4f} (Vel: {total_vloss/len(train_loader):.4f}, Resp: {total_floss/len(train_loader):.4f})")
        
        # ---------------------------
        # 6. Validation
        # ---------------------------
        net.eval()
        val_emse = []
        val_emd = []
        val_peak_err = []
        with torch.no_grad():
            for v_batch in val_loader:
                shape_sdf = v_batch['shape_enc'].to(device)
                properties = v_batch['props'].to(device)
                freqs_v = v_batch['freqs'].to(device)
                velocities_gt = v_batch['velocity'].to(device)
                responses_gt = v_batch['response'].to(device)
                B, F, H, W = velocities_gt.shape

                shape_input_exp = shape_sdf.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)
                props_exp = properties.unsqueeze(1).expand(-1, F, -1).reshape(-1,7)
                freq_flat = freqs_v.unsqueeze(0).expand(B, -1).reshape(-1)

                if net.response_type == 'velocity_field':
                    pred_vel = net(shape_input_exp, props_exp, freq_flat)
                    pred_vel_log = torch.log(torch.clamp(pred_vel**2 + 1e-8, min=1e-8))
                    pred_resp = ((pred_vel_log**2).mean(dim=[2,3])) * 10
                    resp_gt_flat = responses_gt.reshape(-1)
                    resp_pred = pred_resp.squeeze().cpu()
                else:
                    pred_resp = net(shape_input_exp, props_exp, freq_flat)
                    resp_gt_flat = responses_gt.reshape(-1).cpu()
                    resp_pred = pred_resp.cpu()

                # Compute metrics
                emse_value = evaluation.compute_mse(resp_gt_flat, resp_pred)
                emd_value = evaluation.compute_emd(resp_gt_flat, resp_pred)
                peaks_true = evaluation.detect_peaks(resp_gt_flat)
                peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
                pe_ratio, pe_shift = evaluation.compute_peak_errors(peaks_true, peaks_pred, *evaluation.match_peaks(peaks_true, peaks_pred), resp_gt_flat, resp_pred)
                val_emse.append(emse_value)
                val_emd.append(emd_value)
                val_peak_err.append(pe_ratio)  # For simplicity, just store ratio error
                
        # Average validation metrics
        val_emse_mean = np.mean(val_emse)
        val_emd_mean = np.mean(val_emd)
        val_peak_mean = np.mean(val_peak_err)
        print(f"Validation metrics -- EMSE: {val_emse_mean:.4f}, EMD: {val_emd_mean:.4f}, Peak Ratio Error: {val_peak_mean:.4f}")

        scheduler.step(val_emse_mean)

        # Save checkpoint if improved
        if save_checkpoints:
            is_better = val_emse_mean < best_val_loss
            if is_better:
                best_val_loss = val_emse_mean
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, checkpoint_path)
                print("Saved new best model.")

        # Early stopping
        if val_emse_mean >= best_val_loss:
            early_patience -= 1
            if early_patience == 0:
                print(f"No improvement for patience epochs. Stopping early.")
                break
        else:
            early_patience = cfg['training'].get('early_stopping_patience', 20)

    # ---------------------------
    # 7. Load best model and evaluate on test set
    # ---------------------------
    print("Loading best model for test evaluation...")
    if os.path.exists(checkpoint_path):
        chkpt = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(chkpt['model_state_dict'])
    net.eval()

    all_emse = []
    all_emd = []
    all_pek = []

    with torch.no_grad():
        for batch in test_loader:
            shape_sdf = batch['shape_enc'].to(device)  # (B, H, W)
            properties = batch['props'].to(device)
            freqs_t = batch['freqs'].to(device)
            velocities_gt = batch['velocity'].to(device)
            responses_gt = batch['response'].to(device)
            B, F, H, W = velocities_gt.shape

            shape_input_exp = shape_sdf.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)
            props_exp = properties.unsqueeze(1).expand(-1, F, -1).reshape(-1,7)
            freq_flat = freqs_t.unsqueeze(0).expand(B, -1).reshape(-1)

            if net.response_type == 'velocity_field':
                pred_vel = net(shape_input_exp, props_exp, freq_flat)
                pred_vel_log = torch.log(torch.clamp(pred_vel**2 + 1e-8, min=1e-8))
                pred_resp = ((pred_vel_log**2).mean(dim=[2,3]))*10
            else:
                pred_resp = net(shape_input_exp, props_exp, freq_flat)
            resp_gt_flat = responses_gt.reshape(-1).cpu()
            resp_pred = pred_resp.cpu()

            emse = evaluation.compute_mse(resp_gt_flat, resp_pred)
            emd = evaluation.compute_emd(resp_gt_flat, resp_pred)
            peaks_true = evaluation.detect_peaks(resp_gt_flat)
            peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
            pe_ratio, pe_shift = evaluation.compute_peak_errors(peaks_true, peaks_pred, *evaluation.match_peaks(peaks_true, peaks_pred), resp_gt_flat, resp_pred)
            all_emse.append(emse)
            all_emd.append(emd)
            all_pek.append(pe_shift if not np.isnan(pe_shift) else 0.0)

    # Report test metrics
    print(f"\n=== Test Results ===")
    print(f"EMSE: {np.mean(all_emse):.4f} ± {np.std(all_emse):.4f}")
    print(f"EMD: {np.mean(all_emd):.4f} ± {np.std(all_emd):.4f}")
    print(f"Peak shift (mean): {np.mean(all_pek):.4f}")

    # Optionally, visualize example responses/velocity fields
    # For brevity, not included here

if __name__ == "__main__":
    main()
