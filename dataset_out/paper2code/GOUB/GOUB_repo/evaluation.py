## evaluation.py
import torch
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
from tqdm import tqdm
from torchvision.utils import save_image
from scipy import linalg
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from torch.nn.functional import interpolate
from dataset_loader import DatasetLoader

class Evaluation:
    def __init__(self, model, dataset, config, device=None):
        """
        Initialize Evaluation object.
        
        Args:
            model (nn.Module): trained neural net or sampler with a restore method.
            dataset (DatasetLoader): dataset object providing inputs and ground truths.
            config (dict): configuration dictionary from YAML.
            device (torch.device or None): device to perform evaluation on.
        """
        self.model = model
        self.dataset = dataset
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_list = config.get('evaluation', {}).get('metrics', ['PSNR', 'SSIM', 'LPIPS', 'FID'])
        self.dataset_name = config.get('dataset', {}).get('name', 'Unknown')
        self.use_mean_ode = True if 'use_mean_ode' not in config.get('restoration', {}) else config['restoration'].get('use_mean_ode', True)
        self.steps = config.get('inference', {}).get('steps', 100)
        self.normalize_img = lambda img: (img * 2.0) -1.0  # assume images in [0,1], output scaled to [-1,1]
        self.denormalize_img = lambda img: (img + 1.0) / 2.0  # [-1,1] to [0,1]
        # Initialize LPIPS
        if 'LPIPS' in self.metrics_list:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.lpips_fn = None
        # Initialize Inception model for FID
        if 'FID' in self.metrics_list:
            self.inception_model = inception_v3(pretrained=True, progress=False).to(self.device)
            self.inception_model.eval()
            # Remove final classification layer
            self.inception_model = torch.nn.Sequential(*list(self.inception_model.children())[:-1])
        else:
            self.inception_model = None

    def evaluate(self):
        """
        Main evaluation over dataset, computing specified metrics.
        Returns:
            dict: metrics results with keys as metric names and values as mean scores.
        """
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        real_activations = None
        fake_activations = None

        # Collect all images for FID if needed
        real_images_for_fid = []
        generated_images_for_fid = []

        for batch in tqdm(self.dataset, desc='Evaluating'):
            x_input, x_gt, _ = batch
            x_input = x_input.to(self.device)
            x_gt = x_gt.to(self.device)

            # Generate restoration
            if hasattr(self.model, 'restore'):
                # Use model's restore method for inference
                x_restored = self.model.restore(x_input)
            else:
                # fallback if model does not have restore method, raise error
                raise NotImplementedError("Model does not have a 'restore' method.")

            # Clamp and denormalize images to [0,1] for metrics
            pred_img = self.denormalize_img(self._clip_tensor(x_restored))
            true_img = self._clip_tensor(x_gt)

            # Convert tensors to numpy arrays for metrics
            pred_np = pred_img.cpu().numpy().transpose(0,2,3,1)
            true_np = true_img.cpu().numpy().transpose(0,2,3,1)

            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                pred_i = pred_np[i]
                true_i = true_np[i]
                # Calculate PSNR
                if 'PSNR' in self.metrics_list:
                    psnr = compare_psnr(true_i, pred_i, data_range=1.0)
                    psnr_scores.append(psnr)
                # Calculate SSIM
                if 'SSIM' in self.metrics_list:
                    # Optionally, compute only luminance channel
                    # Convert to YCbCr
                    true_y = self._to_y_channel(true_i)
                    pred_y = self._to_y_channel(pred_i)
                    ssim = compare_ssim(true_y, pred_y, data_range=1.0)
                    ssim_scores.append(ssim)
                # Calculate LPIPS
                if 'LPIPS' in self.metrics_list and self.lpips_fn:
                    # Input to LPIPS should be in [-1,1], shape: (3,H,W)
                    l_pred = torch.from_numpy(pred_i.transpose(2,0,1)).unsqueeze(0).to(self.device)
                    l_true = torch.from_numpy(true_i.transpose(2,0,1)).unsqueeze(0).to(self.device)
                    lpips_score = self.lpips_fn(l_pred, l_true).item()
                    lpips_scores.append(lpips_score)
                # Collect for FID
                if 'FID' in self.metrics_list:
                    real_images_for_fid.append(self._resize_for_fid(true_i))
                    generated_images_for_fid.append(self._resize_for_fid(pred_i))
        
        results = {}
        if 'PSNR' in self.metrics_list:
            results['PSNR'] = np.mean(psnr_scores)
        if 'SSIM' in self.metrics_list:
            results['SSIM'] = np.mean(ssim_scores)
        if 'LPIPS' in self.metrics_list:
            results['LPIPS'] = np.mean(lpips_scores)
        if 'FID' in self.metrics_list:
            # Compute FID
            fid_value = self._compute_fid(real_images_for_fid, generated_images_for_fid)
            results['FID'] = fid_value
        # Print results
        print("Evaluation Results:")
        for key, val in results.items():
            print(f"{key}: {val:.4f}")
        return results

    def _clip_tensor(self, tensor):
        """Ensure tensor values in [0,1]"""
        return torch.clamp(tensor, 0.0, 1.0)

    def _to_y_channel(self, img_np):
        """
        Convert RGB image (H,W,3) in [0,1] to luminance Y channel in [0,1] using YCbCr.
        """
        # Conversion weights for luminance
        r = img_np[:,:,0]
        g = img_np[:,:,1]
        b = img_np[:,:,2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return y

    def _resize_for_fid(self, img_np, size=(299,299)):
        """
        Resize image for Inception/FID. Input shape: (H,W,C), output: (C,H,W), scaled to size.
        """
        img_pil = TF.to_pil_image(img_np)
        img_resized = TF.resize(img_pil, size)
        img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(self.device)
        return img_tensor

    def _compute_fid(self, real_imgs, gen_imgs):
        """
        Compute FID score between two lists of images.
        Args:
            real_imgs (list of torch.Tensor): Each tensor shape (1,C,H,W)
            gen_imgs (list of torch.Tensor): Each tensor shape (1,C,H,W)
        Returns:
            float: FID score
        """
        # Extract features
        act1 = self._get_activations(real_imgs)
        act2 = self._get_activations(gen_imgs)
        mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
        fid_value = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value

    def _get_activations(self, imgs):
        """
        Obtain the activation features from Inception network.
        """
        features = []
        batch_size = 50  # adjust as needed
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = torch.cat(imgs[i:i+batch_size], dim=0)  # shape: (batch, C, H, W)
                preds = self.inception_model(batch).squeeze()
                # Use features: flatten spatial dimensions
                if len(preds.shape) > 2:
                    preds = torch.flatten(preds, start_dim=1)
                features.append(preds.cpu().numpy())
        return np.vstack(features)

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Compute FID between two distributions characterized by their mean and covariance.
        """
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if not np.isfinite(covmean).all():
            # fallback if sqrtm produces nan
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

