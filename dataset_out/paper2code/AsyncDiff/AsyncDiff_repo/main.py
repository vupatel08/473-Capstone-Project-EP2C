## main.py
import os
import yaml
import torch
import random
import time
import numpy as np

from dataset_loader import DatasetLoader
from model import DiffusionComponent
from async_inference import AsyncScheduler
from communication import Communication
from evaluation import Evaluation
from utils import generate_schedules, freeze_model_parameters, load_full_model

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    seed = config.get('misc', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set device parameters
    hardware_cfg = config.get('hardware', {})
    num_devices = hardware_cfg.get('num_devices', 4)
    device_type = hardware_cfg.get('device_type', 'NVIDIA A5000')
    device_ids = list(range(num_devices))
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else torch.device('cpu')

    # Initialize dataset loader
    dataset_cfg = config.get('dataset', {})
    dataset_loader = DatasetLoader(dataset_cfg)
    train_loader, val_loader = dataset_loader.load_data()

    # Load or define the full diffusion model (pretrained)
    # Here, assuming 'full_model' is provided or loaded via utils
    full_model = load_full_model(config)  # Function to load the entire diffusion model
    # Generate layer slices / schedule based on model complexity
    num_components = config['model'].get('num_components', 4)
    layer_slices = generate_schedules(full_model, num_components)

    # Instantiate model components by splitting full model
    model_components = []
    for idx in range(num_components):
        comp = DiffusionComponent(
            component_id=idx,
            model_params={
                'full_model': full_model,
                'layer_slices': layer_slices,
                'num_components': num_components
            }
        )
        # Optional: load pretrained weights, freeze parameters if needed
        freeze_model_parameters(comp)
        model_components.append(comp)

    # Assign each component to a device
    device_managers = []
    for idx, dev_id in enumerate(device_ids):
        device_obj = torch.device(f"cuda:{dev_id}")
        # Move model component to device: optional, if model's internal layers support moving
        # For this, you may need to implement .to(device) on model or move parameters
        # Here, assuming component is transferred accordingly
        device_managers.append(DeviceManager(device=device_obj, component=model_components[idx]))

    # Initialize communication across devices
    comm = Communication()

    # Parameters from config
    total_steps = config['sampling'].get('timesteps', 50)
    warmup_steps = config['model'].get('warmup_steps', 5)
    stride = config['model'].get('stride', 2)

    # Instantiate asynchronous scheduler
    async_scheduler = AsyncScheduler(
        model_params={'full_model': full_model},
        device_ids=device_ids,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        num_components=num_components,
        stride=stride,
        dataset=dataset_loader,
        guidance_scale= config['sampling'].get('guidance_scale', 7.5),
    )

    # Run warm-up phase
    print("Starting warm-up phase...")
    start_time = time.time()
    async_scheduler.warmup(val_loader)  # Using validation loader for warm-up
    warmup_time = time.time() - start_time
    print(f"Warm-up completed in {warmup_time:.2f} seconds.")

    # Main inference: asynchronous, stride-aware denoising
    print("Starting main inference...")
    start_time = time.time()
    async_scheduler.execute(val_loader)
    total_inference_time = time.time() - start_time
    print(f"Async inference completed in {total_inference_time:.2f} seconds.")

    # After inference, retrieve generated images/videos and ground truth (if available)
    # For the purpose of this code, assume async_scheduler saves output images/videos
    # and provides a list of generated tensors
    generated_outputs = async_scheduler.get_generated_samples()

    # Load ground truth if available, or set None
    ground_truths = None
    prompts = None
    if 'ground_truths' in config:
        # Load ground truth images for evaluation
        ground_truths = load_ground_truths(config['ground_truths_path'])
    if 'prompts' in config:
        prompts = config['prompts']

    # Evaluate using provided metrics
    evaluator = Evaluation(config.get('evaluation', {}))
    eval_results = evaluator.evaluate(generated_outputs, ground_truths, prompts)

    # Save or print evaluation scores
    print("Evaluation Results:")
    for metric_name, score in eval_results.items():
        print(f"{metric_name}: {score:.4f}")

    # Save generated images/videos if needed
    save_dir = 'outputs/'
    os.makedirs(save_dir, exist_ok=True)
    for idx, img_tensor in enumerate(generated_outputs):
        # Convert tensor to PIL Image
        img = tensor_to_image(img_tensor)
        img.save(os.path.join(save_dir, f'generated_{idx}.png'))

    print("All done successfully.")

def load_ground_truths(gt_path: str):
    """
    Load ground truth images for evaluation.
    Args:
        gt_path (str): Directory path with ground truth images.
    Returns:
        List[torch.Tensor]: List of image tensors.
    """
    imgs = []
    for fname in sorted(os.listdir(gt_path)):
        if fname.endswith('.png') or fname.endswith('.jpg'):
            img_path = os.path.join(gt_path, fname)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((512, 512), Image.LANCZOS)
            tensor = T.ToTensor()(img).unsqueeze(0)
            imgs.append(tensor.squeeze(0))
    return imgs

def tensor_to_image(tensor: torch.Tensor):
    """
    Convert tensor in [0,1] to PIL image.
    """
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    image = to_pil(tensor.clamp(0,1))
    return image

if __name__ == "__main__":
    main()
