## main.py
"""
Main script orchestrating the training and evaluation pipeline for REPRODUCING
the "Make Your LLM Fully Utilize the Context" (IN2) approach as described.
It sequentially:
- Loads configuration
- Generates synthetic long-context QA dataset
- Loads and possibly extends the model (RoPE, position embeddings)
- Fine-tunes the model on the dataset
- Performs probing and scaling evaluation
- Reports metrics and saves models
"""

import os
import sys
import yaml
import torch
from tqdm import tqdm

# Import custom modules
import utils
import dataset_generator
import dataset_loader
import model
import trainer
import evaluation

def main():
    # 1. Load configuration
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Extract core configs with defaults and strongly typed
    model_name = cfg.get('model', {}).get('name', 'mistral-7b-instruct-v0.2')
    rope_base = float(cfg.get('model', {}).get('rope_base', 1e6))
    max_position_embeddings = cfg.get('model', {}).get('max_position_embeddings', 0)
    use_sliding_window = cfg.get('evaluation', {}).get('use_sliding_window', True)
    sliding_window_size = int(cfg.get('evaluation', {}).get('sliding_window_size', 4096))
    training_cfg = cfg.get('training', {})
    dataset_cfg = cfg.get('dataset', {})
    long_context_cfg = cfg.get('long_context', {})
    generation_cfg = cfg.get('generation', {})
    eval_cfg = cfg.get('evaluation', {})

    learning_rate = float(training_cfg.get('learning_rate', 1e-6))
    batch_size = int(training_cfg.get('batch_size', 128))
    epochs = int(training_cfg.get('epochs', 1))
    total_steps = int(training_cfg.get('steps_per_epoch', 14000))
    warmup_ratio = float(training_cfg.get('warmup_steps', 0.03))
    warmup_steps = int(warmup_ratio * total_steps)

    dataset_size = int(dataset_cfg.get('size', 1_100_000))
    context_lengths = long_context_cfg.get('length_distribution', [4000, 8000, 16000, 32000])
    output_dir = cfg.get('output_dir', 'outputs')

    # 2. Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Generate dataset
    print("Loading raw texts for dataset generation...")
    raw_texts = utils.get_raw_texts()
    
    print("Initializing dataset generator...")
    gen = dataset_generator.DatasetGenerator(
        raw_texts=raw_texts,
        max_examples=dataset_size,
        batch_size=batch_size,
        context_lengths=context_lengths,
        model_name=model_name
    )
    print("Generating synthetic long-context QA dataset...")
    dataset = gen.generate_dataset()
    # Save the full dataset for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    full_dataset_path = os.path.join(output_dir, 'full_dataset.json')
    utils.save_dataset(dataset, full_dataset_path)
    print(f"Dataset saved to {full_dataset_path} with {len(dataset)} samples.")

    # 4. Initialize model
    print("Loading model...")
    # For simplicity, assume the model is a LongContextModel supporting extension
    model_obj = model.LongContextModel(
        model_name=model_name,
        rope_base=rope_base,
        extend_positional=(max_position_embeddings > 0),
        max_position_embeddings=max_position_embeddings,
    )
    # 5. Fine-tune model
    print("Loading dataset for training...")
    train_dataset = utils.load_dataset_from_json(full_dataset_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn,
    )
    print("Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model_obj.model.parameters(), lr=learning_rate)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup=int(0.03 * total_steps),
        num_training_steps=total_steps
    )

    print("Starting training...")
    model_obj.model.train()
    global_step = 0
    for epoch in range(epochs):
        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in epoch_iter:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to(input_ids.device)
            labels = batch['labels'].to(input_ids.device)
            outputs = model_obj.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            # gradient clip if desired
            torch.nn.utils.clip_grad_norm_(model_obj.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}/{total_steps}: loss={loss.item():.4f}")

            if global_step % 5000 == 0:
                ckpt_path = os.path.join(output_dir, f'checkpoint_step_{global_step}')
                os.makedirs(ckpt_path, exist_ok=True)
                model_obj.save_model(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break
    print("Training completed. Saving final model...")
    final_path = os.path.join(output_dir, 'final_model')
    model_obj.save_model(final_path)
    print(f"Model saved to {final_path}")

    # 6. Load model for evaluation
    print("Loading trained model for evaluation...")
    eval_model = model.LongContextModel(
        model_name=final_path,
        rope_base=rope_base,
        extend_positional=(max_position_embeddings > 0),
        max_position_embeddings=max_position_embeddings,
    )
    eval_model.model.eval()

    # 7. Run probing and evaluation
    print("Loading dataset for evaluation...")
    dataset_eval = utils.load_dataset_from_json(full_dataset_path)

    print("Starting probing tasks evaluation...")
    probing_results = evaluation.probe_long_context(eval_model, dataset_eval)

    print("Running scaling evaluation...")
    scaling_results = evaluation.scale_long_context(eval_model, dataset_eval)

    print("Evaluating real-world long document tasks...")
    real_tasks_results = evaluation.evaluate_real_world_tasks(eval_model)

    print("Evaluating short-context tasks...")
    short_results = evaluation.evaluate_short_tasks(eval_model)

    # 8. Save results
    results = {
        "probing": probing_results,
        "scaling": scaling_results,
        "real_tasks": real_tasks_results,
        "short_tasks": short_results,
    }
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    main()
