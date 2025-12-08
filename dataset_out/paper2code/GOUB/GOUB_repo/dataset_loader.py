## dataset_loader.py
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy.ndimage as ndi

class DatasetLoader(Dataset):
    def __init__(self, dataset_path: str, batch_size: int, mode: str = 'train', dataset_type: str = 'inpainting',
                 image_size: int = 128, mask_type: str = 'thin', scale_factor: int = 4, mask_prob: float = 0.5,
                 rain_overlay: bool = False):
        """
        DatasetLoader initializes datasets for various image restoration tasks:
        inpainting, super-resolution, deraining.

        Args:
            dataset_path (str): Root directory containing images.
            batch_size (int): Batch size (not used directly here, handled by DataLoader).
            mode (str): 'train' or 'test'.
            dataset_type (str): 'inpainting', 'super-resolution', 'deraining'.
            image_size (int): Size to resize images to (e.g., 128).
            mask_type (str): For inpainting, 'thin' or 'thick' masks.
            scale_factor (int): Downsampling scale for super-resolution.
            mask_prob (float): Probability of generating a mask.
            rain_overlay (bool): Whether to add rain effect for deraining task.
        """
        self.dataset_path = dataset_path
        self.mode = mode
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.mask_type = mask_type
        self.scale_factor = scale_factor
        self.mask_prob = mask_prob
        self.rain_overlay = rain_overlay

        # List all image files in dataset directory
        self.image_files = [os.path.join(root, fname)
                            for root, _, files in os.walk(self.dataset_path)
                            for fname in files if self._is_image_file(fname)]
        # Set transforms
        self.to_tensor = transforms.ToTensor()
        self.resize_transform = transforms.Resize((self.image_size, self.image_size))
        # For normalization (assumed the network training uses [-1,1])
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    def _is_image_file(self, filename):
        IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        return filename.lower().endswith(IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('RGB')
        # Resize image
        img = self.resize_transform(img)
        img_np = np.array(img).astype(np.float32) / 255.0  # [0,1]
        img_tensor = torch.from_numpy(img_np).permute(2,0,1)  # C x H x W

        if self.mode == 'train':
            if self.dataset_type == 'inpainting':
                # Generate mask
                mask = self._generate_mask(img_np.shape[1], img_np.shape[0])
                # Mask the image (set masked parts to 0)
                masked_img = img_np.copy()
                masked_img[mask==1] = 0.0
                input_tensor = torch.from_numpy(masked_img).permute(2,0,1)
                return input_tensor.float(), img_tensor.float(), mask.astype(np.float32)
            elif self.dataset_type == 'super-resolution':
                # Downsample image with bicubic interpolation
                low_res = ndi.zoom(img_np, (1.0/self.scale_factor, 1.0/self.scale_factor, 1), order=3)
                # Upsample back to original size to match target
                low_res_up = ndi.zoom(low_res, (self.scale_factor, self.scale_factor, 1), order=3)
                low_res_up = np.clip(low_res_up, 0, 1)
                input_tensor = torch.from_numpy(low_res_up).permute(2,0,1)
                return input_tensor.float(), img_tensor.float(), None
            elif self.dataset_type == 'deraining':
                # Add rain effect (simulate)
                rain_img = self._add_rain_effect(img_np)
                # Optionally add Gaussian noise
                noisy_img = rain_img + np.random.normal(0, 0.02, rain_img.shape)
                noisy_img = np.clip(noisy_img, 0.0, 1.0)
                input_tensor = torch.from_numpy(noisy_img).permute(2,0,1)
                return input_tensor.float(), img_tensor.float(), None
            else:
                # Default: return original
                return img_tensor.float(), img_tensor.float(), None
        else:
            # Mode = 'test' or validation: no augmentation
            return img_tensor.float(), img_tensor.float(), None

    def _generate_mask(self, width, height):
        """
        Creates a binary mask for inpainting:
        - 'thin': small random lines or narrow regions.
        - 'thick': large rectangular or irregular masks.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.mask_type == 'thin':
            # Generate small lines
            for _ in range(random.randint(1, 3)):
                x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
                x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
                rr, cc = self._line_coords(y1, x1, y2, x2)
                mask[rr, cc] = 1
        elif self.mask_type == 'thick':
            # Generate large irregular mask, e.g., rectangle
            for _ in range(random.randint(1, 2)):
                x_start = random.randint(0, width//2)
                y_start = random.randint(0, height//2)
                x_end = random.randint(x_start + 10, width)
                y_end = random.randint(y_start + 10, height)
                mask[y_start:y_end, x_start:x_end] = 1
        return mask

    def _line_coords(self, y1, x1, y2, x2):
        """
        Bresenham's line algorithm to generate line pixel coords.
        """
        import skimage.draw
        rr, cc = skimage.draw.line(y1, x1, y2, x2)
        return rr, cc

    def _add_rain_effect(self, img_np):
        """
        Overlay synthetic rain streaks over the image.
        Could be simple vertical streaks.
        """
        rain_layer = np.zeros_like(img_np)
        height, width = img_np.shape[0], img_np.shape[1]
        num_strikes = int(0.2 * width * height / (20*20))
        for _ in range(num_strikes):
            x_col = random.randint(0, width - 1)
            for y in range(0, height, 4):
                if random.random() < 0.3:
                    rain_layer[y:y+2, x_col:x_col+1] = 1.0
        # Blend rain layer with original image
        rain_color = np.array([0.8, 0.8, 0.8])  # light rain
        rain_effect = img_np + rain_layer * rain_color
        return np.clip(rain_effect, 0, 1)
