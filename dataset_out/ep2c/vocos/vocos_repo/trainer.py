## trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from spectral_utils import SpectralUtils
from dataset_loader import DatasetLoader
from model import SpectralPredictor
from discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator
import yaml
import os

class Trainer:
    def __init__(self, config: dict):
        # Extract configs with defaults
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Spectral utils initialization
        spectral_cfg = self.config['training']
        self.spectral_utils = SpectralUtils(
            sample_rate=self.config['dataset']['sample_rate'],
            n_fft=self.config['model']['fft_size'],
            hop_length=self.config['model']['hop_length'],
            n_mels=self.config['dataset']['mel_params']['n_mels']
        )
        # Instantiate dataset loader
        self.dataset_loader = DatasetLoader(self.config)
        # Instantiate generator
        self.net_G = SpectralPredictor(self.config['model']).to(self.device)
        # Instantiate discriminators
        self.D_list = []
        self.D_list.append(MultiPeriodDiscriminator().to(self.device))
        self.D_list.append(MultiResolutionDiscriminator().to(self.device))
        # Optimizers
        self.g_optimizer = torch.optim.AdamW(
            self.net_G.parameters(), lr=self.config['training']['learning_rate'],
            betas=tuple(self.config['training'].get('optimizer_betas', [0.9, 0.999])))
        )
        self.d_optimizer_list = []
        for D in self.D_list:
            self.d_optimizer_list.append(
                torch.optim.AdamW(D.parameters(), lr=self.config['training']['learning_rate'],
                                  betas=tuple(self.config['training'].get('optimizer_betas', [0.9, 0.999])))
            )
        # Training schedule
        self.total_iterations = self.config['training']['total_iterations']
        self.save_interval = self.config['training'].get('save_interval', 100000)
        self.log_interval = self.config['training'].get('log_interval', 1000)
        # Loss weights
        self.lambda_adv = 1.0
        self.lambda_mel = 1.0
        self.lambda_feat = 10.0
        # For simplicity, assume spectral regularization is off
        # Initialize logs and path
        self.checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Iteration counter
        self.global_step = 0

    def compute_adversarial_loss_D(self, real_scores, fake_scores):
        """Hinge loss for discriminator."""
        loss = 0
        for rs, fs in zip(real_scores, fake_scores):
            loss += torch.mean(F.relu(1.0 - rs)) + torch.mean(F.relu(1.0 + fs))
        return loss / len(real_scores)

    def compute_adversarial_loss_G(self, fake_scores):
        """Hinge loss for generator."""
        loss = 0
        for fs in fake_scores:
            loss += -torch.mean(fs)
        return loss / len(fake_scores)

    def run(self):
        for iteration in range(1, self.total_iterations + 1):
            self.global_step = iteration
            # Fetch batch
            mel_spec, real_waveform = self.dataset_loader.get_batch(self.config['training']['batch_size'])
            mel_spec = mel_spec.to(self.device)  # [B, N_mels, T]
            real_waveform = real_waveform.to(self.device)  # [B, samples]

            # Data augmentation (e.g., gain)
            # Not explicitly in code; assume dataset loader handles

            # Forward: generator
            m_logits, p_logits = self.net_G(mel_spec)  # [B, F, T], [B, F, T]

            # Convert network output to complex spectral coefficients
            spectral_coeffs = self.spectral_utils.prepare_spectral_outputs(m_logits, p_logits)
            # spectral_coeffs: [B, F, T], complex tensor

            # Waveform synthesis from spectral coefficients
            fake_waveform = self.spectral_utils.inverse_spectrogram(spectral_coeffs)
            # Clamp / normalize waveform if needed
            fake_waveform = fake_waveform.clamp_(-1.0, 1.0)

            # Discriminator steps
            real_scores_list = []
            fake_scores_list = []
            real_feats_list = []
            fake_feats_list = []

            for D, d_optimizer in zip(self.D_list, self.d_optimizer_list):
                # Real
                real_scores, real_feats = D(real_waveform)
                # Fake (detach to avoid grad through G when training D)
                fake_scores, fake_feats = D(fake_waveform.detach())

                real_scores_list.append(real_scores)
                real_feats_list.append(real_feats)
                fake_scores_list.append(fake_scores)
                fake_feats_list.append(fake_feats)

                # Discriminator optimizer step
                d_loss = self.compute_adversarial_loss_D([real_scores], [fake_scores])
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # --- Generator forward again for gradient --- #
            # Recompute discriminator scores for generator update
            gen_scores_list = []
            gen_feats_list = []
            for D in self.D_list:
                scores, feats = D(fake_waveform)
                gen_scores_list.append(scores)
                gen_feats_list.append(feats)

            # --- Compute Generator losses --- #
            # Adversarial loss
            g_adv_loss = self.compute_adversarial_loss_G(gen_scores_list)

            # Mel Spectrogram Loss between real and fake waveforms
            pred_mel = self.spectral_utils.compute_mel_spectrogram(fake_waveform)
            mel_true = self.spectral_utils.compute_mel_spectrogram(real_waveform)
            mel_loss = F.l1_loss(pred_mel, mel_true)

            # Feature matching loss
            feat_loss = 0
            num_Ds = len(self.D_list)
            for real_feats, fake_feats in zip(real_feats_list, gen_feats_list):
                for rf, ff in zip(real_feats, fake_feats):
                    feat_loss += F.l1_loss(ff, rf)
            feat_loss = feat_loss / (num_Ds * len(real_feats))

            # Total generator loss
            g_loss = self.lambda_adv * g_adv_loss + self.lambda_mel * mel_loss + self.lambda_feat * feat_loss

            # Optimize generator
            self.net_G.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging
            if iteration % self.log_interval == 0:
                print(f"Iter {iteration}: G_loss={g_loss.item():.4f}, adv={g_adv_loss.item():.4f}, mel={mel_loss.item():.4f}, feat={feat_loss.item():.4f}")

            # Save checkpoint
            if iteration % self.save_interval == 0:
                torch.save(self.net_G.state_dict(), os.path.join(self.checkpoint_dir, f"G_{iteration}.pt"))
                for idx, D in enumerate(self.D_list):
                    torch.save(D.state_dict(), os.path.join(self.checkpoint_dir, f"D{idx}_{iteration}.pt"))

            # Optionally: perform validation, evaluation metrics computation
            # For brevity, not included here.

        print("Training complete.")

