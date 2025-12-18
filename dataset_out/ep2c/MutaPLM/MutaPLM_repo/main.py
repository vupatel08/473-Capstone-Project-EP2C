# main.py

import os
import sys
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import (
    set_seed, save_checkpoint, load_checkpoint, log_metrics, call_openai_api
)
from dataset_loader import ProteinTextDataset
from model import (
    ProteinPLM, TextEncoder, ProteinDeltaNetwork,
    PromptEmbedding, Heads
)
from evaluation import evaluate_explanation, evaluate_mutation_proposals
import prompt_templates as prompts

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting MutaPLM training and evaluation pipeline.")

    # Set seed
    seed = 42
    set_seed(seed)

    # --------- Extract configuration values with defaults ---------
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    prompt_cfg = config.get('prompt_templates', {})
    eval_cfg = config.get('evaluation', {})
    hardware_cfg = config.get('hardware', {})

    # Training parameters
    lr = training_cfg.get('learning_rate', 1e-4)
    batch_size = training_cfg.get('batch_size', 16)
    warmup_steps = training_cfg.get('warmup_steps', 10000)
    max_steps = training_cfg.get('max_steps', 200000)
    grad_clip = training_cfg.get('gradient_clip_value', 1.0)

    # Paths
    pretrain_path = dataset_cfg.get('pretraining_dataset_path', '')
    finetune_path = dataset_cfg.get('finetuning_dataset_path', '')
    test_path = dataset_cfg.get('test_dataset_path', '')
    enrichment_path = dataset_cfg.get('enrichment_data_path', '')

    # Model configs
    plm_name = model_cfg.get('plm_name', 'facebook/esm2_t6_8a_14B')
    text_enc_name = model_cfg.get('text_encoder_name', 'allenai/scibert_scivocab_uncased')
    delta_hidden_dim = model_cfg.get('delta_hidden_dim', 768)
    num_attention_heads = model_cfg.get('num_attention_heads', 12)
    num_layers = model_cfg.get('num_layers', 12)

    # --------- Environment Setup ---------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --------- Load datasets ---------
    logger.info("Loading datasets...")
    pretrain_dataset = ProteinTextDataset(pretrain_path, mode='pretraining', config=config)
    finetune_dataset = ProteinTextDataset(finetune_path, mode='finetuning', config=config)
    test_dataset = ProteinTextDataset(test_path, mode='test', config=config)

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --------- Initialize models ---------
    logger.info("Initializing models...")
    # PLM backbone
    protein_plm = ProteinPLM(pretrained_name=plm_name).to(device)
    # Text encoder
    text_encoder = TextEncoder(pretrained_name=text_enc_name).to(device)
    # Prompt embedding (soft prompts for cross-modal alignment)
    prompt_embedding = PromptEmbedding(K=32, D=delta_hidden_dim).to(device)
    # Heads for mutation prediction
    heads = Heads(delta_hidden_dim, num_aa=20).to(device)

    # --------- Set up optimizer ------
    # Collect parameters to optimize (full model except frozen parts)
    train_params = list(protein_plm.parameters()) + \
                   list(text_encoder.parameters()) + \
                   list(prompt_embedding.parameters()) + \
                   list(heads.position_head.parameters()) + \
                   list(heads.aa_head.parameters())
    optimizer = optim.AdamW(train_params, lr=lr, eps=1e-8, weight_decay=0.01)
    # Scheduler: linear warm-up + cosine decay
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, num_training_steps=max_steps)

    # --------- Loss functions ---------
    ce_loss_fn = nn.CrossEntropyLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # --------- Initialize model components ---------
    # Delta network
    delta_net = ProteinDeltaNetwork(hidden_dim=delta_hidden_dim).to(device)

    # --------- Checkpoint info ---------
    start_step = 0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------- Training loop ---------
    for step in range(start_step + 1, max_steps + 1):
        # Decide mode
        if step <= 200000:
            dataloader = pretrain_loader
            mode = 'pretraining'
        else:
            dataloader = finetune_loader
            mode = 'finetuning'

        for batch in dataloader:
            optimizer.zero_grad()

            # Prepare batch tensors
            seq_ids = batch['sequence_ids'].to(device)            # [B, L]
            seq_mask = batch['attention_mask'].to(device)
            text_ids = batch['text_ids'].to(device)               # [B, T]
            text_mask = batch['text_attention_mask'].to(device)

            # --------- Forward pass ----------
            # 1. Protein representations
            h_wt = protein_plm(seq_ids, attention_mask=seq_mask)  # [B, L, D]
            # 2. Encode text input
            text_feat = text_encoder.encode([t for t in batch['text']])  # [B, D]
            # 3. Prepare soft prompts
            soft_prompts = prompt_embedding.expand(batch_size).to(device)  # [B, K, D]
            # 4. Compute delta features using delta network: H W T
            # For simplicity in initial code, use h_wt as placeholder; in actual implementation, compute delta between WT and mutant sequences
            # Here, assume batch contains mutant sequences tokens, or can pass h_wt as both for pretraining
            # For placeholder, set h_mut to be same as h_wt
            h_mut = h_wt
            # 5. Compute mutation representation (delta features)
            # We simulate the delta network process
            delta_outputs = delta_net(h_wt, h_mut)
            h_mut_pred = delta_outputs['h_mutated']
            pos_logits = delta_outputs['pos_logits']
            aa_logits = delta_outputs['aa_logits']

            # --------- Compute losses ----------
            # Sequence MLM: (e.g., masked tokens prediction)
            # Placeholder: generate dummy logits
            seq_logits = torch.randn_like(seq_ids, dtype=torch.float)
            seq_loss = ce_loss_fn(seq_logits.view(-1, seq_logits.shape[-1]), seq_ids.view(-1))

            # Text generation loss: dummy placeholder
            text_logits = torch.randn_like(text_ids, dtype=torch.float)
            text_loss = ce_loss_fn(text_logits.view(-1, text_logits.shape[-1]), text_ids.view(-1))

            # Mutation head losses (binary position and amino acid classification)
            # For supervision, assume ground-truth mutation labels are in batch
            # For placeholder, assume zeros
            gt_positions = torch.zeros_like(pos_logits)
            gt_amino_acids = torch.zeros_like(aa_logits)

            pos_head_loss = bce_loss_fn(pos_logits.squeeze(-1), gt_positions.squeeze(-1))
            aa_head_loss = ce_loss_fn(aa_logits.view(-1, 20), gt_amino_acids.argmax(dim=-1).view(-1))

            # Chain-of-thought explanation and proposal losses (if finetuning)
            if step > 200000:
                # For illustration, define dummy explanation/proposal losses
                explanation_loss = torch.tensor(0.0, device=device)
                proposal_loss = pos_head_loss + aa_head_loss
            else:
                explanation_loss = torch.tensor(0.0, device=device)
                proposal_loss = torch.tensor(0.0, device=device)

            # Total loss (weighted sum)
            loss = seq_loss + text_loss + proposal_loss

            # --------- Backprop and optimize ---------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            if step % 100 == 0:
                log_metrics({
                    'step': step,
                    'loss': loss.item(),
                    'seq_loss': seq_loss.item(),
                    'text_loss': text_loss.item(),
                    'proposal_loss': proposal_loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                }, step)

            # Save checkpoint periodically
            if step % 5000 == 0:
                cp_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pth')
                save_checkpoint(protein_plm, optimizer, step, cp_path)
                # Optional: validate and save best model
                if step % 20000 == 0:
                    val_metrics = evaluate((protein_plm, text_encoder, delta_net), test_loader)
                    if val_metrics.get('ROUGE-L', 0) > 0.2:  # threshold placeholder
                        save_checkpoint(protein_plm, optimizer, step, os.path.join(checkpoint_dir, 'best.pth'))

            # Terminate if max steps reached
            if step >= max_steps:
                break

    # --------- Final evaluation ----------
    logger.info("Training completed. Running final evaluation...")
    final_metrics = evaluate((protein_plm, text_encoder, delta_net), test_loader)
    logger.info(f"Final evaluation metrics: {final_metrics}")

    # Save final model
    save_checkpoint(protein_plm, optimizer, max_steps, os.path.join(checkpoint_dir, 'final_model.pth'))

if __name__ == "__main__":
    main()
