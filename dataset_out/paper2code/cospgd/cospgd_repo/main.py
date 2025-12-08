## main.py
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Import custom modules
from dataset_loader import DatasetLoader
from model import Model
from attack import Attack
from evaluation import Evaluation
from utils import set_seed, plot_metrics, save_image_batch

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set device and seed for reproducibility
    device_str = config.get('hardware', {}).get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    seed = config.get('hardware', {}).get('seed', 42)
    set_seed(seed)

    # Log setup
    print(f"Using device: {device}")
    print(f"Random seed: {seed}")

    # Load datasets
    datasets_cfg = config['datasets']
    data_loaders = {}
    for key, cfg in datasets_cfg.items():
        name = cfg['name']
        root_dir = cfg['root_dir']
        split = cfg['split']
        input_size = tuple(map(int, cfg.get('input_size', (512, 512)).split('x'))) if 'x' in str(cfg.get('input_size', '512')) else cfg.get('input_size', (512, 512))
        augment = cfg.get('augment', False)

        print(f"Loading dataset: {name}, split: {split}")
        dataset_loader = DatasetLoader(
            dataset_name=name,
            root_dir=root_dir,
            task=key,  # use key as task identifier
            split=split,
            augment=augment,
            input_size=input_size
        )
        data_loader = torch.utils.data.DataLoader(dataset_loader, batch_size=config.get('training', {}).get('batch_size', 16), shuffle=False)
        data_loaders[key] = data_loader

    # Load models
    models_cfg = config['models']
    models_obj = {}
    for key, m_cfg in models_cfg.items():
        model_name = key
        checkpoint_path = m_cfg['checkpoint_path']
        print(f"Loading model: {model_name} from {checkpoint_path}")
        model = Model(model_name, checkpoint_path, device).model
        model.eval()
        models_obj[key] = model

    # Prepare Attack parameters
    attack_params = config['attack_parameters']
    epsilon = attack_params.get('epsilon', 8/255)
    step_size = attack_params.get('step_size', 2/255)
    attack_iters_list = attack_params.get('attack_iters', [3,5,10,20,40,100])
    targeted = attack_params.get('targeted', False)
    target_label = attack_params.get('target_label', None)
    # Note: target_label can be dataset-specific; here kept generic

    # Prepare output directory
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    # For each dataset (e.g., semantic segmentation, optical flow, restoration)
    for task_name, dataloader in data_loaders.items():
        print(f"\n=== Starting evaluation on task: {task_name} ===")
        # Select dataset-specific model(s). For simplicity, pick one model per task.
        # In practice, you may want to attack multiple models.
        model_key = list(models_obj.keys())[0]
        model = models_obj[model_key]
        model_name = list(models_cfg.keys())[0]
        # Initialize Evaluation object
        eval_obj = Evaluation(
            model=Model(model_name, '', device),  # We can pass dummy if not used
            dataset_loader=None,  # Will set later
            task=task_name,
            device=device,
            save_dir=os.path.join(output_dir, task_name),
            verbose=True
        )

        # To run evaluation, we need to iterate over dataset
        # For each attack iteration count, perform attack
        all_metrics = {str(t): {'IoU': [], 'pixel_accuracy': [], 'EPE': [], 'EPE_f1_all': [], 'PSNR': [], 'SSIM': []}
                       for t in attack_iters_list}

        for batch in tqdm(dataloader, desc=f"Attacking dataset: {task_name}"):
            # Extract inputs based on task
            images = batch['image'].to(device)
            y = None
            # For semantic segmentation
            if task_name == 'semantic_segmentation':
                y = batch['label'].to(device)
            elif task_name == 'optical_flow':
                y = batch['flow'].to(device)
            elif task_name in ('image_restoration', 'image_denoising'):
                y = batch['target'].to(device)

            # Run attacks for each attack iteration count
            for T in attack_iters_list:
                attack_instance = Attack(
                    model=Model(model_name, '', device),
                    epsilon=epsilon,
                    step_size=step_size,
                    max_iters=T,
                    task=task_name,
                    targeted=targeted,
                    target=target_label,
                    device=device
                )

                # Generate adversarial example
                x_adv = attack_instance.attack(x_clean=images, y=y, targeted=targeted, target=target_label)

                # Get model prediction
                pred = model.predict(x_adv)

                # Evaluate metrics
                if task_name == 'semantic_segmentation':
                    true_mask = y.cpu().numpy()
                    pred_logits = pred.cpu()
                    pred_probs = torch.nn.functional.softmax(pred_logits, dim=1)
                    pred_mask = torch.argmax(pred_probs, dim=1).squeeze(0).cpu().numpy()
                    num_classes = pred_probs.shape[1]
                    # Evaluate IoU and pixel accuracy
                    iou = eval_obj.compute_iou(pred_mask, true_mask.squeeze(0), num_classes)
                    pix_acc = eval_obj.compute_pixel_accuracy(pred_mask, true_mask.squeeze(0))
                    all_metrics[str(T)]['IoU'].append(iou)
                    all_metrics[str(T)]['pixel_accuracy'].append(pix_acc)
                elif task_name == 'optical_flow':
                    true_flow = y.squeeze(0).permute(1,2,0).cpu().numpy()
                    pred_flow = pred.squeeze(0).permute(1,2,0).cpu().numpy()
                    epe = eval_obj.compute_epe(pred_flow, true_flow)
                    epe_f1 = eval_obj.compute_epe_f1_all(pred_flow, true_flow)
                    all_metrics[str(T)]['EPE'].append(epe)
                    all_metrics[str(T)]['EPE_f1_all'].append(epe_f1)
                elif task_name in ('image_restoration', 'image_denoising'):
                    true_img = y.squeeze(0).permute(1,2,0).cpu().numpy()
                    pred_img = pred.squeeze(0).permute(1,2,0).cpu().numpy()
                    psnr = eval_obj.compute_psnr(pred_img, true_img)
                    ssim = eval_obj.compute_ssim(pred_img, true_img)
                    all_metrics[str(T)]['PSNR'].append(psnr)
                    all_metrics[str(T)]['SSIM'].append(ssim)
                else:
                    continue

                # Optionally save adversarial images or predictions
                # save_image_batch(x_adv, os.path.join(output_dir, task_name, f"adv_{T}_{np.random.randint(0,10000)}.png"))

        # After processing dataset, compute mean metrics and save
        # Save per-iteration and overall results
        summary = {}
        for T_str, metrics_dict in all_metrics.items():
            summary[T_str] = {}
            for metric_name, values in metrics_dict.items():
                if len(values) > 0:
                    summary[T_str][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        # Save summary JSON
        summary_path = os.path.join(output_dir, task_name, 'evaluation_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Plot metrics trends over attack steps
        for metric in ['IoU', 'pixel_accuracy', 'EPE', 'EPE_f1_all', 'PSNR', 'SSIM']:
            x_vals = list(all_metrics.keys())
            y_vals = [np.mean(all_metrics[t][metric]) if len(all_metrics[t][metric])>0 else 0 for t in x_vals]
            plt.figure()
            plt.plot(x_vals, y_vals, marker='o')
            plt.xlabel('Attack iterations')
            plt.ylabel(metric)
            plt.title(f"{task_name} - {metric} over attack steps")
            plt.grid()
            plt.savefig(os.path.join(output_dir, task_name, f"{metric}_trend.png"))
            plt.close()

if __name__ == '__main__':
    main()
