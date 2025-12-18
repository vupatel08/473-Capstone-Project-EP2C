## trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import kl_annealing, match_masks_hungarian, set_seed, save_checkpoint, load_checkpoint, adjust_learning_rate
from dataset_loader import SlotStateBuffer
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, dataset_loader, optimizer, scheduler, config, device=None):
        """
        Args:
            model (nn.Module): VONet model.
            dataset_loader (DataLoader): DataLoader for training sequences.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler): LR scheduler.
            config (dict): Configuration dict from YAML.
            device (torch.device): Compute device.
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_steps = config['training'].get('total_steps', 150000)
        self.batch_size = config['training'].get('batch_size', 32)
        self.segment_length = config['training'].get('segment_length', 3)
        self.use_replay = config['misc'].get('use_replay_buffer', True)
        self.replay_buffer_size = config['misc'].get('replay_buffer_size', 10000)
        self.zeros_clip_norm = config['optimization'].get('max_gradient_norm', 0.1)
        self.gradient_clipping = config['optimization'].get('gradient_clipping', True)
        self.model_save_path = config['misc'].get('model_save_path', './checkpoints/')
        self.result_save_path = config['misc'].get('result_save_path', './results/')
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.result_save_path, exist_ok=True)

        # Initialize replay buffer if used
        if self.use_replay:
            self.replay_buffer = SlotStateBuffer(self.replay_buffer_size)
        else:
            self.replay_buffer = None

        # State variables
        self.global_step = 0
        # For KL schedule
        self.kl_start_step = self.config['training'].get('kl_anneal_start_step', 0)
        self.kl_end_step = self.config['training'].get('kl_anneal_end_step', 50000)
        self.beta_final = self.config['training'].get('kl_final_weight', 0.7)

        # Visualization interval
        self.vis_interval = self.config['evaluation'].get('evaluation_interval', 10000)

        # Loss tracking
        self.losses = []

        # Set seed
        seed = self.config['misc'].get('seed', 42)
        set_seed(seed)

    def train(self):
        device = self.device
        model = self.model.to(device)
        model.train()

        pbar = tqdm(total=self.total_steps, desc='Training')
        # Initialize previous slot states (r_{0,k}) for each slot, shape [batch, K, D]
        # Start from Gaussian noise
        r_prev = torch.randn(1, self.model.num_slots, self.model.slot_dim, device=device)
        r_prev = r_prev.repeat(self.batch_size, 1, 1)  # batch dimension

        # Initialize previous context (c_{t-1,k}) as zeros, optional
        c_prev = torch.zeros(self.batch_size, self.model.num_slots, self.model.slot_dim, device=device)

        # Prepare optimizer and scheduler
        # Scheduler is assumed to handle LR updates
        # No explicit independent, so we step scheduler based on step count

        for step in range(self.global_step, self.total_steps):
            # Dynamic learning rate adjustment if needed
            adjust_learning_rate(self.optimizer, step, self.config['training'].get('learning_rate_schedule', {}))
            self.optimizer.zero_grad()

            try:
                # Load batch: shape [B, L, 3, 128, 128]
                batch = next(self._data_iterator)
            except StopIteration:
                self._data_iterator = iter(self.dataset_loader)
                batch = next(self._data_iterator)

            x_seq, gt_masks_seq, meta_seq = batch
            # x_seq: [B, L, 3, 128, 128], move to device
            x_seq = x_seq.to(device)
            B, L, C, H, W = x_seq.shape

            # For current step, consider only current frames
            # Assume we process the last frame in sequence for training
            # Alternatively, process all frames with separate losses (authors' approach is to process sequences)
            # Here, following the authors, process full sequence
            total_loss = 0.0

            # For handling the residual states, initialize from replay buffer if used
            if self.use_replay:
                # Sample states for batch segments
                buffer_samples = self.replay_buffer.sample(B)
                # buffer_samples is list of dicts with keys: 'r', etc.
                r_prev_batch = []
                for sample in buffer_samples:
                    r_prev_batch.append(sample['r'])  # shape [K, D]
                r_prev_batch = torch.stack(r_prev_batch, dim=0).to(device)  # [B, K, D]
            else:
                r_prev_batch = torch.randn(B, self.model.num_slots, self.model.slot_dim, device=device)

            # Initialize context vectors c_prev as zeros or from buffer
            # For simplicity, initialize as zeros
            c_prev_batch = torch.zeros(B, self.model.num_slots, self.model.slot_dim, device=device)

            # Save per-batch metrics
            batch_recon_loss = 0.0
            batch_kl_loss = 0.0
            batch_total_loss = 0.0

            # Process sequence: for each frame
            for t in range(self.segment_length):
                x_t = x_seq[:, t, :, :, :]  # [B, 3, 128,128]
                # Forward pass
                outputs = self._forward_single_frame(x_t, r_prev_batch, c_prev_batch, step)
                recon_scene = outputs['recon']
                masks = outputs['masks']
                r_t = outputs['r']
                r_prior = outputs['r_prior']
                mu_z = outputs['mu_z']
                logvar_z = outputs['logvar_z']
                mu_prior = outputs['mu_prior']
                logvar_prior = outputs['logvar_prior']

                # Compute reconstruction loss
                # Assuming Gaussian decoder, negative log likelihood
                recon_loss = self._compute_reconstruction_loss(x_t, recon_scene)
                # Compute KLD
                kld_loss = self._compute_kld(mu_z, logvar_z, mu_prior, logvar_prior)
                # Calculate current beta
                beta = kl_annealing(step, self.kl_start_step, self.kl_end_step, self.beta_final)

                loss = recon_loss + beta * kld_loss

                total_loss += loss

                # Prepare for next timestep
                # Update r_prev_batch
                r_prev_batch = r_t.detach()
                # Update c_prev_batch, here we assume c_prev is same as r_t for simplicity
                c_prev_batch = r_t.detach()

            # Backpropagation
            total_loss.backward()

            # Gradient clipping
            if self.gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), self.max_gradient_norm)

            # Optimizer step
            self.optimizer.step()

            # Save to metrics/log
            self.losses.append(total_loss.item() / self.segment_length)

            # Save slot states to replay buffer
            if self.use_replay:
                # Save current slot states (r_t) for each timestep (here last timestep)
                # As per design, buffer operates on states per frame, so we could store r_t for the last frame in sequence
                for t in range(self.segment_length):
                    # For simplicity, store only last step's r_t
                    state = {'r': r_prev_batch.clone().detach().cpu()}
                    self.replay_buffer.add([state])  # add individually

            # Step learning rate scheduler
            self.scheduler.step()

            # Periodic visualization and validation
            if (step+1) % self.vis_interval == 0 or step == self.total_steps -1:
                self._save_training_metrics(step)
                self._visualize_masks_and_recon(x_seq, masks, step)

            # Save checkpoint
            if (step+1) % 50000 == 0 or step == self.total_steps -1:
                save_checkpoint(model, self.optimizer, step+1, os.path.join(self.model_save_path, 'model_step_{}.pt'.format(step+1)))

            pbar.update(1)

        pbar.close()

    def _forward_single_frame(self, x_t, r_prev, c_prev, step):
        """
        Run model forward for a single frame, returning outputs dict.
        """
        # Extract backbone features
        features = self.model.extract_features(x_t)
        # Generate attention masks
        masks = self.model.generate_attention(features, c_prev)
        # masks shape: [B, K+1, H, W]
        # Foreground masks: exclude background (index 0)
        masks_fg = masks[:, 1:, :, :]  # [B, K, H, W]

        # Encode slot features
        slot_feats = self.model.encode_slots(features, masks_fg)
        # Update slot states
        r_t = self.model.update_slot_states(slot_feats, r_prev)  # [B, K, D]
        # Variational posterior for z_{t,k}
        mu_z, logvar_z = self.model.posterior_z(r_t)
        z = self.model.posterior_z.sample(mu_z, logvar_z)
        # Prior prediction r'
        r_prior, mu_prior, logvar_prior = self.model.predict_slot_prior(r_prev)
        # Scene reconstruction
        recon_scene = self.model.decode_scene(z)

        return {
            'recon': recon_scene,
            'masks': masks,
            'r': r_t,
            'r_prior': r_prior,
            'mu_z': mu_z,
            'logvar_z': logvar_z,
            'mu_prior': mu_prior,
            'logvar_prior': logvar_prior
        }

    def _compute_reconstruction_loss(self, x, recon):
        """
        Compute per-pixel negative log likelihood assuming Gaussian with fixed variance.
        """
        # Assuming standard Gaussian with unit variance
        # As per authors, often uses gaussian NLL
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
        return recon_loss

    def _compute_kld(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        Compute KL divergence between q(z|r) and p(z|r')
        """
        # KL divergence for diagonal Gaussians
        kld = 0.5 * (logvar_p - logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p) -1 )
        return kld.sum() / mu_q.shape[0]

    def _save_training_metrics(self, step):
        """
        Save metrics, losses, and plots
        """
        # Save losses list
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss over Steps')
        plt.savefig(os.path.join(self.result_save_path, 'loss_curve_step_{}.png'.format(step)))
        plt.close()

    def _visualize_masks_and_recon(self, x_seq, masks, step):
        """
        Create visualizations for attention masks and scene reconstructions.
        """
        B, L, C, H, W = x_seq.shape
        for t in range(min(L, self.config['evaluation'].get('visualization_frames', 3))):
            x_frame = x_seq[0, t]
            mask_probs = masks[0, 1:, :, :]  # assume all K masks
            thresh_mask = (mask_probs >= 0.3).float()
            # Save overlays
            save_mask_path = os.path.join(self.result_save_path, f'seq_{step}_frame_{t}_mask.png')
            # Assume utils.py has visualization functions
            from utils import visualize_attention_masks, visualize_reconstruction
            # For visualization, get input image as numpy
            visualize_attention_masks(x_frame.permute(1,2,0).cpu().numpy(), mask_probs.cpu(), save_mask_path, frame_idx=t)
            # Save reconstructed image
            recon_img = self.model.decode_scene(z).detach().cpu()
            save_recon_path = os.path.join(self.result_save_path, f'seq_{step}_frame_{t}_recon.png')
            visualize_reconstruction(x_frame.cpu(), recon_img[0], save_recon_path, frame_idx=t)
            

# Usage example (assuming proper config, and dataset loader implemented)
if __name__ == '__main__':
    import yaml
    from dataset_loader import MoviSequenceDataset
    from model import build_vonet_from_config
    import torch.optim as optim
    import torch.nn as nn
    import torch
    import os

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['misc'].get('device', 'cuda'))

    # Prepare dataset
    dataset = MoviSequenceDataset(
        dataset_dir='./dataset',  # set accordingly
        split_files=['./splits/train_split.txt'],
        sequence_length=config['training'].get('segment_length',3),
        training=True,
        transform=None,
        dataset_split='official_split',
        object_max_count=10 # or 16 for D/E
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training'].get('batch_size', 32),
        shuffle=True,
        num_workers=4,
        collate_fn=None  # optional, default collate
    )

    # Build model
    model = build_vonet_from_config(config)
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Define LR scheduler
    def lr_lambda(current_step):
        # Implement the schedule: warmup, plateau, decay
        schedule_params = config['training'].get('learning_rate_schedule', {})
        return ... # define as needed
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Instantiate trainer
    trainer = Trainer(model, data_loader, optimizer, scheduler, config)

    # Run training
    trainer.train()
