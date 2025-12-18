## main.py
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm

from dataset_loader import DatasetLoader
from detectors import DetectorAPI
from model import ModelWrapper
from trainer import PPOTrainer
from evaluation import Evaluation
from utils import normalize_scores

def main():
    # 1. Load config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = 42
    if "seed" in config:
        seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Initialize device
    device = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Initialize DetectorAPI (assuming API-based here; customize in config if local)
    detector_config = config.get("detectors", {})
    detector_api = None
    if detector_config.get("api_endpoint", "") != "":
        detector_api = DetectorAPI(model_type="api",
                                   config={
                                       "api_endpoint": detector_config.get("api_endpoint", ""),
                                       "api_token": detector_config.get("api_token", "")
                                   })
    else:
        # For local model detector, modify accordingly
        detector_api = DetectorAPI(model_type="local",
                                   config={
                                       "model_name": detector_config.get("model_name", "roberta-base"),
                                       "device": device
                                   })

    # 4. Load prompt dataset
    prompts = []
    # Assuming openwebtext prompts: create a placeholder or load dataset
    # Here, we hardcode prompts or load from file if specified
    if "prompt_source" in config["dataset"]:
        source = config["dataset"]["prompt_source"]
        if source == "file" and "prompt_file_path" in config["dataset"]:
            path = config["dataset"]["prompt_file_path"]
            with open(path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
        elif source == "openwebtext":
            # Placeholder: sample prompts from dataset or define static prompts
            # For reproducibility, define simple prompts
            prompts = [
                "The quick brown fox",
                "In a far away land",
                "Once upon a time",
                "The future of AI is",
                "The history of art involves",
                "Advancements in science",
                "The benefits of exercise",
                "Understanding quantum mechanics"
            ]
        else:
            prompts = ["Sample prompt 1", "Sample prompt 2"]
    else:
        prompts = ["Sample prompt 1", "Sample prompt 2"]

    # 5. Instantiate denormalized model
    model_name = config["training"].get("model_name", "Llama-2-7b")
    model_wrapper = ModelWrapper(model_name=model_name, device=device)

    # 6. Generate baseline responses
    print("Generating baseline responses...")
    dataset = DatasetLoader(prompts=prompts)
    dataset.generate_responses(
        model=model_wrapper,
        responses_per_prompt=2,
        responses=None  # Will generate new responses internally
    )

    # 7. Obtain detector scores for generated responses (initial responses)
    detector_scores = dataset.compute_detector_scores()
    # print("Sample detector scores:", detector_scores[:2])  # Optional

    # 8. Create preference dataset based on detector scores
    print("Constructing preference pairs based on detector scores...")
    dataset.create_preference_pairs(detector_scores, threshold=0.0)  # threshold=0.0 for strict comparison
    preference_pairs = dataset.get_pairs()
    print(f"Total preference pairs: {len(preference_pairs)}")

    # 9. Instantiate PPO trainer
    training_params = config["training"]
    ppo_trainer = PPOTrainer(
        model=model_wrapper,
        dataset=dataset,
        detector=detector_api,
        kl_coeff=training_params.get("kl_coeff", 0.5),
        beta=training_params.get("beta", 0.5),
        lr=training_params.get("learning_rate", 1e-5),
        batch_size=training_params.get("batch_size", 16),
        total_steps=training_params.get("total_steps", 30000),
        save_interval=training_params.get("save_interval", 5000),
        device=device,
        sequence_length=training_params.get("sequence_length", 120),
        temperature=training_params.get("temperature", 0.7),
        top_p=training_params.get("top_p", 0.9)
    )

    # 10. Run RL fine tuning
    print("Starting RL fine-tuning to evade detector...")
    ppo_trainer.train()

    # 11. Save final model
    final_model_path = "llama2-7b-evading.pt"
    print(f"Saving final fine-tuned model to {final_model_path}")
    torch.save(model_wrapper.model.state_dict(), final_model_path)

    # 12. Generate validation/evaluation samples after training
    eval_prompts = prompts  # or load separate evaluation prompts if desired
    eval_responses = []
    print("Generating eval responses...")
    for prompt in tqdm(eval_prompts, desc="Generating eval samples"):
        resp = model_wrapper.generate(
            prompt=prompt,
            max_new_tokens=training_params.get("sequence_length", 120),
            temperature=training_params.get("temperature", 0.7),
            top_p=training_params.get("top_p", 0.9)
        )
        eval_responses.append((prompt, resp))

    # 13. Evaluate detector scores, AUROC, perplexity
    print("Evaluating on generated samples...")
    detector_scores_eval = {}
    for detector_name in [detectors.model_type for detectors in [detector_api]]:
        # get scores
        scores = []
        for _, resp in eval_responses:
            scores.append(detector_api.score(resp))
        detector_scores_eval[detector_name] = scores

    # Or use the evaluation.py module for a comprehensive report
    eval_metrics = Evaluation(
        model=model_wrapper,
        detectors=[detector_api],
        device=device
    ).evaluate_texts([resp for _, resp in eval_responses])

    print("Evaluation metrics after fine-tuning:")
    print(eval_metrics)

    # 14. Optional: Human evaluation
    # Can be implemented by an external survey or omitted here

if __name__ == "__main__":
    main()
