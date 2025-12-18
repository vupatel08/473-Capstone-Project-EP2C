# main.py
import os
import yaml
import torch
from dataset_loader import DatasetLoader
from model import DiffusionModel
from diffusion_sampler import DiffusionSampler
from evaluation import Evaluation

def main():
    # Load configuration from 'config.yaml'
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Dataset Loader
    dataset_cfg = cfg.get('dataset', {})
    dataset_path = dataset_cfg.get('dataset_path', './dataset')
    image_size = tuple(dataset_cfg.get('image_size', [512, 512]))
    dataset_type = dataset_cfg.get('dataset_type', 'unconditional')
    dataset_name = dataset_cfg.get('dataset_name', 'laion')
    dataset_loader = DatasetLoader(
        dataset_path=dataset_path,
        image_size=image_size,
        dataset_type=dataset_type,
        dataset_name=dataset_name
    )

    # Optional: Load datasets for training/evaluation
    # Here, we prepare dataloader for training
    train_loader = torch.utils.data.DataLoader(
        dataset_loader,
        batch_size=cfg['training'].get('batch_size', 8),
        shuffle=True,
        drop_last=True
    )

    # Initialize Diffusion Model
    model_cfg = cfg.get('model', {})
    architecture = model_cfg.get('architecture', 'SDXL')
    pretrained_ckpt = model_cfg.get('pretrained_checkpoint', '')
    freeze_backbone = model_cfg.get('freeze_backbone', False)
    attention_blur = model_cfg.get('attention_blur', True)

    diffusion_model = DiffusionModel(architecture=architecture, pretrained_path=pretrained_ckpt)
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    if freeze_backbone:
        for param in diffusion_model.parameters():
            param.requires_grad = False
        # Optionally, freeze only backbone parts, depending on model structure

    # Initialize Sampler with guidance and sigma schedule
    sampling_cfg = cfg.get('sampling', {})
    guidance_cfg = cfg.get('guidance', {})
    guidance_scale = guidance_cfg.get('guidance_scale', 3.0)
    sigma_schedule = guidance_cfg.get('sigma_schedule', [0,1,2,5,10,20,50,100])
    steps = sampling_cfg.get('steps', 1000)
    sampler_type = sampling_cfg.get('sampler_type', 'ddim')
    guidance_variant = guidance_cfg.get('guidance_variant', 'segmented_attention')  # e.g., 'none', 'segmented_attention'

    sampler = DiffusionSampler(
        model=diffusion_model,
        guidance_scale=guidance_scale,
        sigma_schedule=sigma_schedule,
        steps=steps,
        sampler_type=sampler_type,
        guidance_type=guidance_variant
    )

    # Optional: Load checkpoint if provided, or train
    # Here, assuming no training loop; focus on inference
    # If training desired, implement a Trainer class or call existing train()

    # Run inference over different sigma values as per sigma_schedule
    for sigma in sigma_schedule:
        # Set the current sigma in sampler (if needed)
        # Here, sigma is passed during sampling; assumes sampler handles it internally
        print(f"Generating samples with sigma={sigma} and guidance_scale={guidance_scale}")
        samples = []
        num_samples = cfg.get('sampling', {}).get('num_samples', 1)
        for _ in range(num_samples):
            # Generate images
            sample = sampler.sample(conditioning=None)  # No conditioning for unconditional
            samples.append(sample)

        # Save generated images
        save_dir = os.path.join("outputs", f"sigma_{sigma}")
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(samples):
            # Convert tensor to PIL image for saving
            img_cpu = img.squeeze(0).cpu()
            # Assume pixel range [-1,1], convert to [0,255]
            img_np = ((img_cpu + 1) / 2).clamp(0,1).permute(1,2,0).numpy() * 255
            img_pil = Image.fromarray(img_np.astype('uint8'))
            img_pil.save(os.path.join(save_dir, f"sample_{i+1}.png"))

    # Optional: Run evaluation with Evaluation class
    # Load generated images for metrics
    # For example, sample some images and compute FID, CLIP, LPIPS
    # Here, just as an example:
    # generated_images = torch.stack([...
    # evaluator = Evaluation()
    # metrics = evaluator.evaluate(generated_images, prompts=[])
    # print(metrics)

if __name__ == "__main__":
    main()
