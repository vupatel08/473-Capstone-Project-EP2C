## main.py
import argparse
import yaml
import os
import torch
import numpy as np
import random

from dataset_loader import DatasetLoader
from model import MaskUNet, ResNetClassifier
from trainer import Trainer
from evaluation import Evaluator
from utils import load_background_images

def parse_args():
    parser = argparse.ArgumentParser(description='Adaptive Randomized Smoothing Experiment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'certify'],
                        help='Operation mode: train, evaluate, or certify')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', help='Compute device: auto, cpu, cuda')
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_arg == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device(device_arg)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize dataset loader
    dataset_loader = DatasetLoader(config)

    # Instantiate models
    mask_params = config['model']['mask_unet']
    classifier_params = config['model']['classifier']
    dataset_name = config['dataset']['name']
    arch = classifier_params.get('architecture', 'resnet50')
    # Initialize mask model
    mask_model = MaskUNet(
        base_channels=mask_params['base_channels'],
        channel_mult=mask_params['channel_mult'],
        step_size=mask_params.get('step_size', 40),
        gamma=mask_params.get('gamma', 0.5),
        momentum=mask_params.get('momentum', 0.9),
    ).to(device)

    # Initialize classifier
    num_classes = 10 if dataset_name=='CIFAR10' else 2 if dataset_name=='CelebA' else 1000
    classifier = ResNetClassifier(architecture=arch, num_classes=num_classes).to(device)

    # Initialize trainer
    trainer = Trainer(config)
    # Assign models for training
    trainer.mask_net = mask_module=mask_model
    trainer.classifier = classifier

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'evaluate':
        # Load trained models if checkpoints exist
        mask_path = './checkpoints/mask_net.pth'
        clf_path = './checkpoints/classifier.pth'
        if os.path.exists(mask_path):
            mask_model.load_state_dict(torch.load(mask_path))
        if os.path.exists(clf_path):
            classifier.load_state_dict(torch.load(clf_path))
        # Evaluate on validation/test set
        val_loader = dataset_loader.get_data_loader(
            batch_size=256, shuffle=False, num_workers=4
        )
        accuracy = trainer.evaluate(val_loader)
        print(f'Validation/Test Accuracy: {accuracy*100:.2f}%')
    elif args.mode == 'certify':
        # Load trained models
        mask_path = './checkpoints/mask_net.pth'
        clf_path = './checkpoints/classifier.pth'
        mask_model.load_state_dict(torch.load(mask_path))
        classifier.load_state_dict(torch.load(clf_path))
        mask_model.eval()
        classifier.eval()
        # Set parameters for certification
        n_samples = config['evaluation'].get('monte_carlo_samples', 1000)
        conf_level = config['evaluation'].get('certification_confidence', 0.99)
        error_tol = config['evaluation'].get('certification_error_tolerance', 0.01)

        # For each sample in test set, run certification
        test_loader = dataset_loader.get_data_loader(
            batch_size=1, shuffle=False, num_workers=4
        )
        total_samples = 0
        correct_predictions = 0
        certified_count = {}
        # Define radius thresholds for evaluation, e.g., [0, 0.1, 0.2, 0.5,...]
        radius_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
        for radius in radius_thresholds:
            certified_count[radius] = 0

        for X, label in test_loader:
            X = X.to(device)
            label = label.to(device)
            evaluator = Evaluator(mask_model, classifier, 
                                  sigma1=trainer.sigma1, sigma2=trainer.sigma2, device=device,
                                  background_scale=config['dataset'].get('background_scale', 640))
            result = evaluator.get_certified_radius(
                X[0], n_samples=n_samples, conf_level=conf_level, error_tol=error_tol
            )
            pred_class = result['predicted_class']
            radius_certified = False
            # Check certification at thresholds
            for thr in radius_thresholds:
                if result['radius'] >= thr:
                    radius_certified = True
            total_samples +=1

            # Count accuracy
            if pred_class == label.item():
                correct_predictions +=1
                # If radius certified above threshold, count for cert
                for thr in radius_thresholds:
                    if result['radius'] >= thr:
                        certified_count[thr] +=1

        # Print Results
        print(f'Accuracy on test samples: {correct_predictions/total_samples*100:.2f}%')
        for thr in radius_thresholds:
            print(f'Certified accuracy at radius >= {thr}: {certified_count[thr]/total_samples*100:.2f}%')

if __name__ == '__main__':
    main()
