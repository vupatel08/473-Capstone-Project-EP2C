```python
```python
## main.py
import os
import yaml
import random
import numpy as np
import torch
import logging

from dataset_loader import DatasetLoader
from model import PretrainedModel
from prompt_module import PromptGenerator
from label_mapping import LabelMapper
from trainer import Trainer
from tuner import HyperparameterTuner
from evaluation import Evaluation

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set logging verbosity
    verbosity_str = config.get('logging', {}).get('verbosity', 'INFO').upper()
    import logging
    logging_level = getattr(logging, verbosity_str, logging.INFO)
    logging.basicConfig(level=logging_level)

    seed = config.get('misc', {}).get('seed', 42)
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset_name = config['dataset']['name']
    dataset_root = config['dataset'].get('root_dir', None)
    dataset_loader = DatasetLoader(
        datasets_list=[dataset_name],
        batch_size=config['training'].get('batch_size', 32),
        dataset_root=dataset_root
    )
    datasets_dict = dataset_loader.load_data()
    train_loader = datasets_dict[dataset_name]['train']
    val_loader = datasets_dict[dataset_name]['val']
    test_loader = datasets_dict[dataset_name]['test']
    dataset_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Initialize pre-trained backbone model
    backbone_name = config['model']['backbone']
    backbone = PretrainedModel(model_name=backbone_name, freeze=True).to(device)
    backbone.eval()

    # Initialize PromptGenerator
    prompt_size = config['model']['prompt_size']
    prompt_type = config['model']['prompt_type']
    prompt_init_type = config['model'].get('prompt_init_type', 'zeros')
    prompts = PromptGenerator(prompt_size=prompt_size,
                              prompt_type=prompt_type,
                              prompt_init_type=prompt_init_type)

    # Get class names for source and target
    # For source classes, assume ImageNet classes or mock list
    # For target, use dataset specific classes
    def get_source_class_names(model_name):
        # Placeholder: list of 1000 ImageNet classes
        return [f'class_{i}' for i in range(1000)]
    def get_target_class_names(dataset_name):
        # Placeholder: load dataset class names from dataset info
        num_classes = config['dataset'].get('num_classes', 10)
        return [f'class_{i}' for i in range(num_classes)]

    source_class_names = get_source_class_names(backbone_name)
    target_class_names = get_target_class_names(dataset_name)

    # Initialize LabelMapper based on hyperparameter choice
    mapping_strategy = None
    mapping_choice = None
    if 'label_mapping' in config:
        mapping_choice = config['label_mapping']
    else:
        mapping_choice = 'FreqMap'  # default
    map_params = {
        'batch_size': config['training'].get('batch_size', 32),
        'num_source_classes_per_target': config['dataset'].get('num_classes', None)
    }
    label_mapper = LabelMapper(
        strategy=mapping_choice,
        source_class_names=source_class_names,
        target_class_names=target_class_names,
        map_params=map_params,
        device=device
    )

    # Hyperparameter search space
    hp_search_space = {
        'prompt_size': config['hyperparameters'].get('prompt_size_options', [16]),
        'input_scale': config['hyperparameters'].get('input_scale_options', [1.0]),
        'model_choice': config['hyperparameters'].get('model_choices', ['resnet18']),
        'label_mapping': config['hyperparameters'].get('label_mapping_strategies', ['FreqMap'])
        # Additional hyperparameters like number of source classes per target can be added
    }

    # Initialize HyperparameterTuner
    tuner = HyperparameterTuner(
        config={
            'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'training': config['training'],
            'model': config['model'],
            'hyperparameters': config['hyperparameters'],
            'logging': config.get('logging', {}),
            'misc': config.get('misc', {})
        }
    )

    # Run hyperparameter search to find best config
    best_hyperparams = tuner.run()

    # Instantiate objects with best hyperparameters
    # Unpack best hyperparameters
    best_prompt_size = best_hyperparams['prompt_size']
    best_input_scale = best_hyperparams['input_scale']
    best_model_choice = best_hyperparams['model_choice']
    best_mapping_strategy = best_hyperparams['label_mapping']

    # Re-initialize model
    backbone_best = PretrainedModel(model_name=best_model_choice, freeze=True).to(device)
    backbone_best.eval()

    # Initialize prompts
    prompts_best = PromptGenerator(
        prompt_size=best_prompt_size,
        prompt_type=prompt_type,
        prompt_init_type=prompt_init_type
    )

    # Recompute class names if needed
    source_class_names_best = get_source_class_names(best_model_choice)
    target_class_names_best = get_target_class_names(dataset_name)

    label_mapper_best = LabelMapper(
        strategy=best_mapping_strategy,
        source_class_names=source_class_names_best,
        target_class_names=target_class_names_best,
        map_params=map_params,
        device=device
    )

    # Final full training with selected hyperparameters
    # Initialize optimizer for prompts and label mapping
    optimizer_params = list(prompts_best.prompt_tensor.parameters())
    if hasattr(prompts_best, 'real_coeffs'):
        optimizer_params += list(prompts_best.real_coeffs.parameters())
    if hasattr(prompts_best, 'imag_coeffs'):
        optimizer_params += list(prompts_best.imag_coeffs.parameters())
    if best_mapping_strategy == 'FullyMap':
        optimizer_params += list(label_mapper_best.linear_mapping.parameters())

    optimizer = torch.optim.Adam(optimizer_params,
                                 lr=config['training']['learning_rate'],
                                 weight_decay=config['training'].get('weight_decay', 0))
    # Optional LR scheduler
    lr_scheduler = None
    if config['training'].get('lr_scheduler', None) == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    # Initialize trainer and train over full epochs
    trainer = Trainer(
        model=backbone_best,
        prompts=prompts_best,
        dataset=dataset_dict,
        label_mapper=label_mapper_best,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config
    )
    # Run full training
    trainer.train()

    # Final evaluation on test set
    evaluator = Evaluation(model=backbone_best,
                           prompts=prompts_best,
                           dataset_loader=dataset_loader,
                           label_mapper=label_mapper_best,
                           config=config)
    results = evaluator.evaluate()

    # Save final model and prompts
    save_dir = config.get('logging', {}).get('log_dir', './logs')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': backbone_best.model.state_dict(),
        'prompts_state_dict': prompts_best.prompt_tensor.state_dict()
        # Save label mapper if needed
    }, os.path.join(save_dir, 'final_model_prompts.pth'))

    # Print summary
    print("Final Evaluation Results:")
    print(f"Accuracy: {results.get('accuracy', 'N/A'):.2f}%")
    print(f"IoU: {results.get('iou', 'N/A'):.4f}")
    if results.get('robust_accuracy', None) is not None:
        print(f"Robust Accuracy: {results['robust_accuracy']:.2f}%")
    if results.get('average_confidence', None) is not None:
        print(f"Average Confidence: {results['average_confidence']:.4f}")

if __name__ == '__main__':
    main()
```