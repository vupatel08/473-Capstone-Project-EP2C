# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import yaml
import os
import time
import math
from utils import (
    set_seed, save_checkpoint, load_checkpoint,
    log_metrics, call_openai_api
)
from dataset_loader import ProteinTextDataset
from model import (
    ProteinPLM,
    TextEncoder,
    ProteinDeltaNetwork,
    PromptEmbedding,
    Heads
)
import prompt_templates as prompts

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set seeds for reproducibility
set_seed(42)

# Extract hyperparameters with defaults
batch_size = config.get('training', {}).get('batch_size', 16)
learning_rate = config.get('training', {}).get('learning_rate', 1e-4)
warmup_steps = config.get('training', {}).get('warmup_steps', 10000)
max_steps = config.get('training', {}).get('max_steps', 200000)
gradient_clip_value = config.get('training', {}).get('gradient_clip_value', 1.0)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate datasets
pretrain_dataset_path = config['dataset'].get('pretraining_dataset_path', '')
finetune_dataset_path = config['dataset'].get('finetuning_dataset_path', '')
test_dataset_path = config['dataset'].get('test_dataset_path', '')

# Load datasets
pretrain_dataset = ProteinTextDataset(pretrain_dataset_path, mode='pretraining', config=config)
finetune_dataset = ProteinTextDataset(finetune_dataset_path, mode='finetuning', config=config)
test_dataset = ProteinTextDataset(test_dataset_path, mode='test', config=config)

# Data loaders
pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
plm_name = config['model'].get('plm_name', 'facebook/esm2_t6_8a_14B')
text_enc_name = config['model'].get('text_encoder_name', 'allenai/scibert_scivocab_uncased')
delta_hidden_dim = config['model'].get('delta_hidden_dim', 768)

protein_plm = ProteinPLM(pretrained_name=plm_name)
text_encoder = TextEncoder(pretrained_name=text_enc_name)
prompt_embedding = PromptEmbedding(K=32, D=delta_hidden_dim)
heads = Heads(hidden_dim=delta_hidden_dim, num_aa=20)

# Move models to device
protein_plm.to(device)
text_encoder.to(device)
# prompt_embedding and heads are nn.Modules with parameters
prompt_embedding.to(device)
heads.position_head.to(device)
heads.aa_head.to(device)

# Optimizers
# Collect trainable parameters
train_params = list(protein_plm.parameters()) + \
               list(text_encoder.parameters()) + \
               list(prompt_embedding.parameters()) + \
               list(heads.position_head.parameters()) + \
               list(heads.aa_head.parameters())
optimizer = optim.AdamW(train_params, lr=learning_rate, eps=1e-8, weight_decay=0.01)

# Learning rate scheduler with warmup and cosine decay
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
)

# Loss functions
seq_ce_loss = nn.CrossEntropyLoss()  # For amino acid token prediction
mask_ce_loss = nn.CrossEntropyLoss() # For masked sequence tokens
# For mutation proposal (pos + aa heads)
pos_loss_fn = nn.BCEWithLogitsLoss()
aa_loss_fn = nn.CrossEntropyLoss()

# Training loop
global_step = 0
best_val_rouge = -float('inf')
best_model_path = "checkpoint_best.pth"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to perform evaluation on test set
def evaluate(model_components, dataloader):
    protein_plm, text_encoder, delta_net = model_components
    protein_plm.eval()
    text_encoder.eval()
    delta_net.eval()
    # Prepare evaluation metrics
    rouge_scores = []
    bleu_scores = []
    meteor_scores = []
    rec_50 = 0
    total_samples = 0
    correct_mutation_acc = 0

    # Import evaluation metrics
    from tqdm import tqdm
    import nltk
    from rouge_score import Rouge

    rouge_eval = Rouge()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq_ids = batch['sequence_ids'].to(device)
            seq_mask = batch['attention_mask'].to(device)
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            # For mutation labels, handle accordingly
            m_labels = batch['mutation_labels']
            # Forward through PLM
            h_wt = protein_plm(seq_ids, attention_mask=seq_mask)
            # Encode texts
            text_feats = text_encoder.encode(batch['text'])
            # Compute delta features
            if hasattr(delta_net, 'encode_delta'):
                h_delta = delta_net.encode_delta(h_wt, h_wt)  # Placeholder, replace as needed
            else:
                # For simplicity, assume delta features are zero
                h_delta = torch.zeros_like(h_wt)
            # Generate explanations and proposals
            # Here, for evaluation, you'd generate explanations and compare
            # For brevity, assuming ground-truth texts are available
            # Compute metrics: placeholder implementation
            # For example:
            # rouge_scores.append(rouge_eval.get_scores(generated_text, reference_text))
            total_samples += 1
            # For mutation accuracy, compare predicted mutation proposal with ground truth
            # Placeholder: set to 0 or 1 based on match
            # correct_mutation_acc += int(predicted_mutation == ground_truth)
        # Compute average metrics
    # Return metrics dict
    return {
        "ROUGE-L": sum(rouge_scores)/len(rouge_scores) if rouge_scores else 0,
        "BLEU-2": 0,
        "METEOR": 0,
        "Recall@50": rec_50,
        "Mutation_Acc": correct_mutation_acc / total_samples if total_samples > 0 else 0
    }

# Main training routine
for step in range(1, max_steps + 1):
    # Decide if training on pretraining or finetuning
    if step <= 200000:
        # Pretraining stage
        dataloader = pretrain_loader
        model_train_mode = 'pretraining'
    else:
        # Finetuning stage
        dataloader = finetune_loader
        model_train_mode = 'finetuning'

    for batch in dataloader:
        optimizer.zero_grad()
        # Prepare inputs
        seq_ids = batch['sequence_ids'].to(device)  # shape: [B, L]
        seq_mask = batch['attention_mask'].to(device)
        text_ids = batch['text_ids'].to(device)
        text_mask = batch['text_attention_mask'].to(device)

        # Forward through PLM
        h_wt = protein_plm(seq_ids, attention_mask=seq_mask)  # [B, L, D]
        # Encode textual info
        text_feat = text_encoder.encode([t for t in batch['text']])  # [B, D]

        # Sample or generate soft prompts (if used)
        soft_prompts = prompt_embedding.expand(batch_size).to(device)

        # For pretraining: sequence mask task
        # Randomly select tokens to mask
        # Implement masking pattern and create labels
        # Placeholder: assume tokens masked at random positions
        # For brevity, assuming mask_labels and input masked sequences are prepared
        # Similarly for text generation, compute cross entropy with ground-truth text tokens
        # For cross-modal alignment, compute cosine similarity or directly minimize embedding distance

        # For illustration: assume predicted tokens and compute loss
        seq_logits = torch.randn_like(seq_ids, dtype=torch.float)  # Dummy placeholder
        seq_loss = seq_ce_loss(seq_logits.view(-1, seq_logits.shape[-1]), seq_ids.view(-1))

        # Similarly, compute text generation loss
        text_logits = torch.randn_like(text_ids, dtype=torch.float)  # Dummy placeholder
        text_loss = mask_ce_loss(text_logits.view(-1, text_logits.shape[-1]), text_ids.view(-1))

        # Compute cross-modal alignment loss (e.g., MSE or cosine)
        # Here, placeholder as zero
        cross_modal_loss = torch.tensor(0.0).to(device)

        # For finetuning: chain-of-thought-specific losses (mutation explanation/proposal)
        if step > 200000:
            # Use specific prompt templates (from prompt_templates.py)
            # Generate function description (teacher forcing or model-generated)
            # Generate mutational effects
            # Compute ROUGE/BLEU/METEOR losses for explanation
            # Compute mutation head losses
            # Placeholder: using dummy losses
            explanation_loss = torch.tensor(0.0).to(device)
            proposal_pos_logits = torch.randn(batch['sequence_ids'].size(0), batch['sequence_ids'].size(1), 1).to(device)
            proposal_aa_logits = torch.randn(batch['sequence_ids'].size(0), batch['sequence_ids'].size(1), 20).to(device)

            # Suppose ground-truth mutation positions and amino acids
            gt_positions = torch.zeros_like(proposal_pos_logits)
            gt_amino_acids = torch.zeros_like(proposal_aa_logits)

            # Compute heads loss
            pos_loss = pos_loss_fn(proposal_pos_logits.squeeze(-1), gt_positions.squeeze(-1))
            aa_loss = aa_loss_fn(proposal_aa_logits.view(-1, 20), gt_amino_acids.argmax(dim=-1).view(-1))
            # Total explanation proposal loss
            proposal_loss = pos_loss + aa_loss
        else:
            explanation_loss = torch.tensor(0.0).to(device)
            proposal_loss = torch.tensor(0.0).to(device)

        # Total loss (weighted sum)
        total_loss = seq_loss + text_loss + cross_modal_loss + explanation_loss + proposal_loss

        # Backpropagation
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=gradient_clip_value)

        # Optimizer step
        optimizer.step()

        # Scheduler step
        scheduler.step()

        # Logging
        if step % 100 == 0:
            metrics = {
                'loss': total_loss.item(),
                'seq_loss': seq_loss.item(),
                'text_loss': text_loss.item(),
                'cross_modal_loss': cross_modal_loss.item(),
                'explanation_loss': explanation_loss.item(),
                'proposal_loss': proposal_loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            }
            log_metrics(metrics, step)

        # Checkpoint saving
        if step % 5000 == 0:
            save_checkpoint(protein_plm, optimizer, step, os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'))
            # Save best model based on validation metrics
            if step % 20000 == 0:
                val_metrics = evaluate((protein_plm, text_encoder, None), test_loader)
                # Use ROUGE-L to compare
                if val_metrics['ROUGE-L'] > best_val_rouge:
                    best_val_rouge = val_metrics['ROUGE-L']
                    save_checkpoint(protein_plm, optimizer, step, os.path.join(checkpoint_dir, best_model_path))
    # Break condition for max steps
    if step >= max_steps:
        print("Reached max training steps.")
        break

# Final evaluation
final_metrics = evaluate((protein_plm, text_encoder, None), test_loader)
print(f"Final evaluation metrics: {final_metrics}")
# Save final model
save_checkpoint(protein_plm, optimizer, max_steps, os.path.join(checkpoint_dir, 'final_model.pth'))

