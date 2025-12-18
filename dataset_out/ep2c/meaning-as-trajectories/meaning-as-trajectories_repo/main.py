#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

This script orchestrates the process of loading configurations, initializing models,
loading datasets, sampling trajectories, computing divergences, performing evaluations,
and reporting results according to the methodology described in the paper.
It strictly follows the provided design and interface specifications.

Usage:
    python main.py
"""

import os
import yaml
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm

# Import custom modules (assuming they are in the same directory and follow the described interface)
import utils
from dataset_loader import DatasetLoader
from model import ModelWrapper, Trajectory
from sampling import sample_trajectories
from likelihood import compute_log_likelihood
from distance import approximate_divergence
from evaluation import Evaluation

def main():
    # 1. Load configuration
    config_path = "config.yaml"
    config = utils.load_config(config_path)

    # 2. Override with CLI args if provided; for simplicity, only seed and model_name
    args = utils.parse_args()
    if args.seed is not None:
        config['sampling']['seed'] = args.seed
        utils.set_seed(args.seed)
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.dataset_prompt_pairs_path:
        config['dataset']['prompt_pairs_path'] = args.dataset_prompt_pairs_path
    if args.dataset_wordnet_relations_path:
        config['dataset']['wordnet_relations_path'] = args.dataset_wordnet_relations_path
    if args.dataset_multimodal_data_path:
        config['dataset']['multimodal_data_path'] = args.dataset_multimodal_data_path
    if args.device:
        config['misc']['model_device'] = args.device

    # 3. Extract hyperparameters
    hp = utils.get_hyperparameters(config)

    # 4. Prepare device
    device = utils.get_device(hp['device'])

    # 5. Initialize model wrapper
    model_wrapper = ModelWrapper(
        model_name=hp['model_name'],
        model_type=hp['model_type'],
        device=device,
        verbose=config.get('verbose', False)
    )

    # 6. Load datasets
    dataset_loader = DatasetLoader(
        prompt_pairs_path=hp['prompt_pairs_path'],
        wordnet_relations_path=hp['wordnet_relations_path'],
        multimodal_data_path=hp['multimodal_data_path'],
        verbose=config.get('verbose', False)
    )

    # Load prompt pairs for semantic similarity evaluation
    prompt_pairs = dataset_loader.load_prompt_pairs()
    # Load WordNet relations for hyponym/hypernym tests
    wordnet_relations = dataset_loader.load_wordnet_relations()
    # Load multimodal data for multimodal experiments
    multimodal_data = dataset_loader.load_multimodal_inputs()

    # 7. Sampling: Generate trajectories for each prompt
    print("Sampling trajectories for prompt pairs...")
    prompt_samples_u = {}  # cache for u
    prompt_samples_v = {}  # cache for v
    trajectories_u = []
    trajectories_v = []

    # Helper function to get trajectories for a prompt string
    def get_trajectories_for_prompt(prompt: str) -> List[Trajectory]:
        return sample_trajectories(
            model_wrapper,
            prompt=prompt,
            n=hp['num_trajectories'],
            max_length=hp['max_length'],
            temperature=hp['temperature'],
            seed=hp['seed']
        )

    # 8. Compute divergence scores for semantic similarity tasks
    divergence_scores = []
    labels_similarity = []  # Human similarity labels from dataset if available
    for (prompt1, prompt2, label_score) in tqdm(prompt_pairs, desc="Semantic pairs"):
        # Sample trajectories for prompt1
        trajs_u = get_trajectories_for_prompt(prompt1)
        # Sample trajectories for prompt2
        trajs_v = get_trajectories_for_prompt(prompt2)
        # Store for potential later use
        trajectories_u.extend(trajs_u)
        trajectories_v.extend(trajs_v)
        # Approximate divergence
        div = approximate_divergence(
            trajs_u, trajs_v,
            model_wrapper, model_wrapper,
            tau=hp['likelihood_normalization_tau'],
            dist_type=hp['divergence']['type']
        )
        divergence_scores.append(div)
        labels_similarity.append(label_score)  # For correlation analysis

    # 9. Evaluate semantic similarity (correlation with human labels)
    eval_sim = Evaluation(divergence_scores, labels_similarity)
    spearman_corr = eval_sim.evaluate_similarity()

    print(f"Semantic Similarity Spearman correlation: {spearman_corr:.2f}")

    # 10. Infer entailment directions between prompt pairs
    # Using divergence scores in both directions (for the same pairs)
    divergence_uv = []
    divergence_vu = []
    entailment_labels = []  # ground truth: 1 if u entails v, 0 otherwise
    for (prompt1, prompt2, label_score) in tqdm(prompt_pairs, desc="Entailment inference"):
        # Sample for u
        trajs_u = get_trajectories_for_prompt(prompt1)
        # Sample for v
        trajs_v = get_trajectories_for_prompt(prompt2)
        # Store for record
        # divergence d(M_u, M_v)
        div_uv = approximate_divergence(trajs_u, trajs_v, model_wrapper, model_wrapper,
                                        tau=hp['likelihood_normalization_tau'],
                                        dist_type=hp['divergence']['type'])
        # divergence d(M_v, M_u)
        div_vu = approximate_divergence(trajs_v, trajs_u, model_wrapper, model_wrapper,
                                        tau=hp['likelihood_normalization_tau'],
                                        dist_type=hp['divergence']['type'])
        divergence_uv.append(div_uv)
        divergence_vu.append(div_vu)
        # For illustration, assume label_score > threshold indicates entailment
        # Here, since dataset labels may not be binary, we can threshold divergence or use label info
        # But for demonstration, assuming label_score > 0.5 maps to entailment
        entailment_labels.append(1 if label_score > 0.5 else 0)

    eval_entail = Evaluation(divergence_uv, entailment_labels)
    entailment_acc = eval_entail.evaluate_entailment(divergence_uv, divergence_vu, entailment_labels)

    print(f"Entailment accuracy: {entailment_acc:.2f}")

    # 11. WordNet hyponym/hypernym relation predictions
    word_pairs = [(w1, w2, label) for w1, w2, label in wordnet_relations]
    divergence_word_u = []
    divergence_word_v = []
    labels_word = []
    for (word_u, word_v, label) in tqdm(word_pairs, desc="WordNet hyponym/hypernym class"):
        # Generate trajectories for u
        trajs_u = get_trajectories_for_prompt(word_u)
        # Generate trajectories for v
        trajs_v = get_trajectories_for_prompt(word_v)
        div_u = approximate_divergence(
            trajs_u, trajs_v,
            model_wrapper, model_wrapper,
            tau=hp['likelihood_normalization_tau'],
            dist_type=hp['divergence']['type']
        )
        div_v = approximate_divergence(
            trajs_v, trajs_u,
            model_wrapper, model_wrapper,
            tau=hp['likelihood_normalization_tau'],
            dist_type=hp['divergence']['type']
        )
        divergence_word_u.append(div_u)
        divergence_word_v.append(div_v)
        labels_word.append(1 if label == 1 else 0)  # 1 if v is hyponym of u

    eval_word = Evaluation(divergence_word_u, labels_word)
    hyponym_acc = eval_word.evaluate_hypernymy(divergence_word_u, divergence_word_v, labels_word)

    print(f"WordNet hyponym prediction accuracy: {hyponym_acc:.2f}")

    # 12. Multimodal experiments (if applicable)
    # For each multimodal sample: generate trajectories from images and captions
    # and compute similarity scores between modalities
    print("Performing multimodal similarity evaluations...")
    multimodal_divergences = []
    human_labels_mm = []  # assume some labels or use similarity scores
    for sample in tqdm(multimodal_data):
        # Compose prompts for image and caption
        prompt_img = sample['prompt_image']  # e.g., "Describe this image: [IMAGE]"
        prompt_txt = sample['prompt_caption']  # e.g., "This is a caption for an image."
        # Sample trajectories
        trajs_img = sample_trajectories(
            model_wrapper,
            prompt=prompt_img,
            n=hp['num_trajectories'],
            max_length=hp['max_length'],
            temperature=hp['temperature'],
            seed=hp['seed']
        )
        trajs_txt = sample_trajectories(
            model_wrapper,
            prompt=prompt_txt,
            n=hp['num_trajectories'],
            max_length=hp['max_length'],
            temperature=hp['temperature'],
            seed=hp['seed'] + 100  # distinct seed for different modality
        )
        div = approximate_divergence(
            trajs_img, trajs_txt,
            model_wrapper, model_wrapper,
            tau=hp['likelihood_normalization_tau'],
            dist_type=hp['divergence']['type']
        )
        multimodal_divergences.append(div)
        # Placeholder for human label: e.g., similarity score between 0 and 5
        human_labels_mm.append(sample.get('human_similarity', 0))  

    # Compute correlation with human judgments
    eval_mm = Evaluation(multimodal_divergences, human_labels_mm)
    corr_mm = eval_mm.evaluate_similarity()

    print(f"Multimodal similarity (image-image/text) correlation: {corr_mm:.2f}")

    # 13. Save results and optionally generate visualizations
    # For example, save divergence matrices, trajectories, or hierarchies
    # For brevity, omitted here but can be implemented as needed.

    print("\n=== Summary of Results ===")
    print(f"Semantic similarity Spearman correlation: {spearman_corr:.2f}")
    print(f"Entailment classification accuracy: {entailment_acc:.2f}")
    print(f"WordNet hyponym accuracy: {hyponym_acc:.2f}")
    print(f"Multimodal similarity correlation: {corr_mm:.2f}")

if __name__ == "__main__":
    main()
