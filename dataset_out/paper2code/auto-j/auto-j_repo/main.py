## main.py
import os
import sys
import yaml
import logging
import random
import time
import torch

# Import classes and functions from other modules
from dataset_loader import DatasetLoader, Dataset, Sample
from annotation import AnnotationGenerator
from model import AutoJudgeModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # Load configuration from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("Main")

    # Set environment variables (e.g., API key)
    api_key_env = config.get("annotation", {}).get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        logger.error(f"Environment variable {api_key_env} not set. Exiting.")
        sys.exit(1)

    # Set random seed for reproducibility
    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directories if applicable
    checkpoint_dir = config.get("model", {}).get("checkpoint_path", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Instantiate DatasetLoader
    data_paths = {
        "train_files": config.get("dataset", {}).get("train_files", []),
        "val_files": config.get("dataset", {}).get("val_files", []),
        "test_files": config.get("dataset", {}).get("test_files", [])
    }
    scenario_list = config.get("dataset", {}).get("scenario_list", [])
    dataset_loader = DatasetLoader(data_paths, scenario_list)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = dataset_loader.load_data(split='train')
    val_dataset = dataset_loader.load_data(split='val')
    test_dataset = dataset_loader.load_data(split='test')
    logger.info(f"Loaded {len(train_dataset.samples)} training samples.")
    logger.info(f"Loaded {len(val_dataset.samples)} validation samples.")
    logger.info(f"Loaded {len(test_dataset.samples)} test samples.")

    # Prepare scenario criteria prompts (assumed in config)
    scenario_criteria_prompts = {}
    scenario_instructions = {}
    for scenario in scenario_list:
        scenario_criteria_prompts[scenario] = ""
        scenario_instructions[scenario] = ""  # Will be filled/loaded from config if provided

    # Instantiate AnnotationGenerator
    annotation_gen = AnnotationGenerator(config)

    # Annotate training data
    logger.info("Annotating training data with GPT-4...")
    for scenario in scenario_list:
        scenario_samples = [s for s in train_dataset.samples if s.scenario == scenario]
        if not scenario_samples:
            continue
        scenario_prompt = scenario_instructions.get(scenario, "[Evaluate the response based on scenario guidelines.]")
        annotations = annotation_gen.generate_annotations(scenario_samples, scenario_prompt, task_type='pairwise')
        # Assign annotations back to samples
        for sample, annot in zip(scenario_samples, annotations):
            sample.annotation = annot
    # Similarly, annotate validation data
    logger.info("Annotating validation data...")
    for scenario in scenario_list:
        scenario_samples = [s for s in val_dataset.samples if s.scenario == scenario]
        if not scenario_samples:
            continue
        scenario_prompt = scenario_instructions.get(scenario, "[Evaluate the response based on scenario guidelines.]")
        annotations = annotation_gen.generate_annotations(scenario_samples, scenario_prompt, task_type='pairwise')
        for sample, annot in zip(scenario_samples, annotations):
            sample.annotation = annot
    # Annotate test data for both pairwise and single-response
    logger.info("Annotating test data for pairwise evaluation...")
    for scenario in scenario_list:
        scenario_samples = [s for s in test_dataset.samples if s.scenario == scenario]
        if not scenario_samples:
            continue
        scenario_prompt = scenario_instructions.get(scenario, "[Evaluate the response based on scenario guidelines.]")
        annotations = annotation_gen.generate_annotations(scenario_samples, scenario_prompt, task_type='pairwise')
        for sample, annot in zip(scenario_samples, annotations):
            sample.annotation = annot
    # For single-response evaluation set, optionally
    # Here, we re-use test samples, annotating critiques and ratings
    logger.info("Annotating test data for single-response critiques...")
    for scenario in scenario_list:
        scenario_samples = [s for s in test_dataset.samples if s.scenario == scenario]
        if not scenario_samples:
            continue
        scenario_prompt = scenario_instructions.get(scenario, "[Evaluate the response based on scenario guidelines.]")
        annotations = annotation_gen.generate_annotations(scenario_samples, scenario_prompt, task_type='single')
        for sample, annot in zip(scenario_samples, annotations):
            sample.annotation = annot

    # Optionally, filter data based on heuristic heuristics (done within annotation.generate_annotations)
    # Save annotated datasets to disk if needed
    # For brevity, proceed to model training

    # Initialize and load model
    logger.info("Loading model...")
    model = AutoJudgeModel(config)

    # Prepare data for training: format data accordingly
    # For simplicity, assume datasets are prepared with proper input_text and labels
    # Here, creating a training DataLoader wrapping train_dataset
    trainer = Trainer(model, train_dataset, config)

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save final model checkpoint
    final_ckpt_path = os.path.join(checkpoint_dir, "final_checkpoint")
    logger.info("Saving final model checkpoint...")
    model.save_checkpoint(final_ckpt_path)

    # Evaluation phase
    logger.info("Evaluating on test set...")
    evaluator = Evaluation(model, test_dataset, config)
    evaluator.report()

if __name__ == "__main__":
    main()
