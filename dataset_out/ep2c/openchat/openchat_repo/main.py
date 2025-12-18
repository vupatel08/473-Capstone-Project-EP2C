# main.py

import os
import yaml
import torch
import random
import logging

from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Load configuration from 'config.yaml'
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set random seed for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 3. Load dataset
    dataset_cfg = config.get('dataset', {})
    dataset_path = dataset_cfg.get('dataset_path', 'data/sharegpt_mixed_quality.json')
    train_sample_size = dataset_cfg.get('train_sample_size', 128)
    eval_sample_size = dataset_cfg.get('eval_sample_size', 128)
    dataset_loader = DatasetLoader({
        'dataset_path': dataset_path,
        'train_sample_size': train_sample_size,
        'eval_sample_size': eval_sample_size,
        'seed': seed
    })
    train_data, eval_data = dataset_loader.load_data()
    logging.info(f"Loaded dataset with {len(train_data)} training samples and {len(eval_data)} evaluation samples.")

    # 4. Initialize the language model
    model_cfg = config.get('model', {})
    pretrained_name = model_cfg.get('pretrained_model_name', "huggingface/llama-13b")
    conditioning_token = model_cfg.get('conditioning_token', "<|class|>")
    model = Model(pretrained_name, conditioning_token)
    logging.info(f"Loaded pretrained model {pretrained_name} with class conditioning token '{conditioning_token}'.")

    # 5. Initialize the trainer
    training_cfg = config.get('training', {})
    beta = training_cfg.get('beta', 0.2)
    lr = training_cfg.get('learning_rate', 3e-5)
    batch_size = training_cfg.get('batch_size', 200)
    epochs = training_cfg.get('epochs', 3)
    max_grad_norm = training_cfg.get('max_grad_norm', 1.0)
    warmup_steps = training_cfg.get('warmup_steps', 1000)
    weight_decay = training_cfg.get('weight_decay', 0.01)

    trainer = Trainer(
        model=model,
        dataset=train_data,
        beta=beta,
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        seed=seed
    )

    # 6. Run training
    logging.info("Starting training process...")
    trainer.train()
    output_dir = config.get('output_dir', 'output/openchat_finetuned')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    trainer.save_model(output_dir)
    logging.info(f"Model saved to {output_dir}")

    # 7. Initialize evaluation
    eval_dataset = eval_data  # For simplicity, using eval_data; could load benchmark datasets separately
    eval_instance = Evaluation(
        model=model,
        eval_dataset=eval_dataset,
        scoring_model_name='gpt-4',  # or another LLM for scoring
        num_eval_samples=128,
        max_new_tokens=256,
        temperature=0.7
    )

    # 8. Evaluate on each benchmark
    benchmarks = config.get('evaluation', {}).get('eval_benchmarks', [
        "AlpacaEval", "MT-bench", "Vicuna-bench", "AGIEval"
    ])
    logging.info(f"Starting evaluation on benchmarks: {benchmarks}")
    results = eval_instance.evaluate_all(benchmarks, datasets_map={bm: eval_dataset for bm in benchmarks})

    # 9. Log evaluation metrics
    for bm in benchmarks:
        res = results[bm]
        logging.info(f"Benchmark {bm}: Win Rate={res.get('win_rate', 0):.2%}, "
                     f"Average Score={res.get('average_score', 0):.2f}")
    # Optional: Save evaluation results to file
    eval_results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    logging.info(f"Evaluation results saved to {eval_results_path}")

# Run main when executing the script
if __name__ == '__main__':
    main()
