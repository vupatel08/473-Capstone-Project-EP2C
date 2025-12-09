# config.py

import yaml

# Load configuration from the provided 'config.yaml' file
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Dataset configuration
DATASETS_PATHS = {
    'market1501': cfg['dataset']['datasets_paths'].get('market1501', ''),
    'msmt17': cfg['dataset']['datasets_paths'].get('msmt17', ''),
    'cuhk02': cfg['dataset']['datasets_paths'].get('cuhk02', ''),
    'cuhk03': cfg['dataset']['datasets_paths'].get('cuhk03', ''),
    'cuhksysu': cfg['dataset']['datasets_paths'].get('cuhksysu', ''),
    'prid': cfg['dataset']['datasets_paths'].get('prid', ''),
    'grid': cfg['dataset']['datasets_paths'].get('grid', ''),
    'viper': cfg['dataset']['datasets_paths'].get('viper', ''),
    'ilids': cfg['dataset']['datasets_paths'].get('ilids', ''),
}

# Model architecture parameters
MODEL = {
    'backbone': 'resnet50',  # Options: 'resnet50', 'vit-b/16', 'mobilenet_v2'
    'feature_dim': 512,
    'normalize_features': True,  # ensure features are normalized to sphere
}

# Training parameters
TRAINING = {
    'learning_rate': cfg['training'].get('learning_rate', 0.001),
    'batch_size': cfg['training'].get('batch_size', 64),
    'epochs': cfg['training'].get('epochs', 60),
    'warmup_epochs': cfg['training'].get('warmup_epochs', 5),
    'weight_decay': cfg['training'].get('weight_decay', 1e-4),
    'triplet_margin': cfg['training'].get('triplet_margin', 0.3),
    'lambda_alignment': cfg['training'].get('lambda_alignment', 1.0),
    'g_hard_triplet_loss': cfg['training'].get('g_hard_triplet_loss', True),
    'augmentation_prob': cfg['training'].get('augmentation_probability', 0.5),
    'neighbor_k': cfg['training'].get('neighbor_k', 10),
    'prototype_momentum': cfg['training'].get('prototype_momentum', 0.999),
    'seed': cfg['misc'].get('seed', 42),
}

# Evaluation parameters
EVALUATION = {
    'protocol': cfg['evaluation'].get('protocol', 'Protocol-3'),  # default 'Protocol-3'
    'metrics': cfg['evaluation'].get('metrics', ['mAP', 'Rank-1']),
    'batch_size': cfg['evaluation'].get('evaluation_batch_size', 64),
    # Additional evaluation settings can be added here
}

# Augmentation parameters
AUGMENTATION = {
    'random_erasing': cfg['augmentation'].get('random_erasing', True),
    'random_erasing_prob': cfg['augmentation'].get('random_erasing_prob', 0.25),
    'rand_augment': cfg['augmentation'].get('rand_augment', True),
    'rand_augment_prob': cfg['augmentation'].get('rand_augment_prob', 0.5),
    'color_jitter': cfg['augmentation'].get('color_jitter_prob', 0.3),
    'color_jitter_params': {
        'brightness': cfg['augmentation'].get('color_jitter_params', {}).get('brightness', 0.2),
        'contrast': cfg['augmentation'].get('color_jitter_params', {}).get('contrast', 0.2),
        'saturation': cfg['augmentation'].get('color_jitter_params', {}).get('saturation', 0.2),
        'hue': cfg['augmentation'].get('color_jitter_params', {}).get('hue', 0.1),
    },
}

# Additional miscellaneous parameters
MISC = {
    'neighbor_search_k': cfg['misc'].get('neighbor_search', {}).get('k', 10),
    'prototype_update_momentum': cfg['misc'].get('prototype_update', {}).get('momentum', 0.999),
    'save_model_path': cfg['misc'].get('save_model_path', './results/model.pth'),
    'log_interval': cfg['misc'].get('log_interval', 50),
    'seed': cfg['misc'].get('seed', 42),
}
