## main.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Import custom modules
from dataset_loader import DatasetLoader
from model import ResNet18
from trainer import Trainer
from evaluation import Evaluation
from utils import set_seed

def main():
    # 1. Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Set seed for reproducibility
    seed = config.get('misc', {}).get('random_seed', 1993)
    set_seed(seed)
    
    # 3. Setup device (GPU/CPU)
    use_gpu = config.get('hardware', {}).get('gpu', True)
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    multi_gpu = config.get('hardware', {}).get('multi_gpu', False)
    
    # 4. Initialize Dataset Loader
    dataset_conf = config.get('dataset', {})
    dataset_name = dataset_conf.get('name', 'CIFAR-100')
    split_scheme = dataset_conf.get('split_scheme', 'class_incremental')
    classes_per_task = dataset_conf.get('classes_per_task', 10)
    total_tasks = dataset_conf.get('total_tasks', 10)
    seed = dataset_conf.get('seed', 1993)
    data_dir = dataset_conf.get('data_dir', './data')
    
    dataset_loader = DatasetLoader(
        dataset_name=dataset_name,
        split_scheme=split_scheme,
        classes_per_task=classes_per_task,
        total_tasks=total_tasks,
        seed=seed,
        data_dir=data_dir
    )
    
    # 5. Initialize Model
    model_arch = config.get('model', {}).get('architecture', 'ResNet18')
    # For simplicity, only ResNet18 supported here
    model = ResNet18()
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # 6. Setup optimizer and scheduler
    training_conf = config.get('training', {})
    lr = training_conf.get('learning_rate', 0.1)
    batch_size = training_conf.get('batch_size', 64)
    epochs = training_conf.get('epochs', 150)
    schedule_conf = training_conf.get('schedule', {})
    
    optimizer_params = dict(
        lr=lr,
        momentum=training_conf.get('optimizer_params', {}).get('momentum',0.9),
        weight_decay=training_conf.get('optimizer_params', {}).get('weight_decay',1e-4)
    )
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    # Scheduler
    milestones = schedule_conf.get('milestones', [])
    decay_factor = schedule_conf.get('decay_factor', 0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay_factor)
    
    # 7. Prepare output directories
    output_dir = config.get('logging', {}).get('output_dir', './logs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 8. Initialize Evaluation
    eval_conf = config.get('evaluation', {})
    landscape_period = eval_conf.get('landscape_visualization', False)
    eval_metrics_flag = eval_conf.get('metrics', True)
    eval_freq = eval_conf.get('evaluation_frequency', 1)  # per epoch
    landscape_eval_freq = eval_conf.get('landscape_eval_freq', 10)
    evaluator = Evaluation(model, dataset_loader, config, device)
    
    # 9. Loop over tasks (incremental phases)
    class_sets = dataset_loader.get_class_sets()
    
    for task_idx in range(len(class_sets)):
        print(f"\n======= Starting training for task {task_idx+1}/{len(class_sets)} =======")
        # 9.1 Data for current task
        train_loader = dataset_loader.get_task_dataloader(task_idx)
        # Optional rehearsal buffer: skipped for simplicity; implement if needed
        
        # 9.2 Instantiate trainer for current task
        trainer_instance = Trainer(model, dataset_loader, config, device, output_dir, seed)
        
        # 9.3 Train with C-Flat regularization
        trainer_instance.train_phase(task_idx)
        
        # 9.4 Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'model_task_{task_idx}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        
        # 9.5 Evaluation after each task
        metrics = evaluator.evaluate(task_idx, len(dataset_loader.get_class_sets()[task_idx]))
        print(f"Performance after task {task_idx+1}: {metrics}")
        # Save evaluation metrics if needed
        evaluator.process(epoch=epochs, task_idx=task_idx, seen_class_count=sum([len(c) for c in class_sets[:task_idx+1]]))
        
        # 9.6 Visualization of landscape and Hessian estimates
        if landscape_period:
            print(f"Visualizing landscape at end of task {task_idx+1}")
            evaluator.visualize_landscape(epoch=epochs)
        
    # 10. Final save and cleanup
    evaluator.save_final_results()
    print("Training complete. Results saved.")

if __name__ == '__main__':
    main()
