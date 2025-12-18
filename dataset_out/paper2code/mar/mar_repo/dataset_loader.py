## dataset_loader.py
import os
import random
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import numpy as np
from PIL import Image

# Import tokenizer utilities from the provided tokenizer_utils.py
import tokenizer_utils

class ImagenetDataset(Dataset):
    """
    Custom Dataset for loading ImageNet images, preprocessing,
    and encoding into token sequences for training/evaluation.
    """
    def __init__(self,
                 image_dir: str,
                 tokenizer_type: str = 'VQ-16',
                 tokenizer_path: str = '',
                 image_size: int = 256,
                 seq_length: int = 1024,
                 mode: str = 'continuous',  # 'discrete' or 'continuous'
                 transform: Optional[transforms.Compose] = None,
                 encode_on_the_fly: bool = True,
                 seed: int = 42,
                 max_samples: Optional[int] = None):
        """
        Initialize dataset.
        Args:
            image_dir: Path to ImageNet root directory (train or val).
            tokenizer_type: Type of tokenizer ('VQ-16', 'KL-16' etc).
            tokenizer_path: Path to pretrained tokenizer checkpoint.
            image_size: Size to resize images.
            seq_length: Expected token sequence length.
            mode: 'discrete' (int tokens) or 'continuous' (float vectors).
            transform: torchvision transforms for images.
            encode_on_the_fly: If True, encode images during __getitem__; else, encode all at init.
            seed: random seed for shuffling.
            max_samples: limit on dataset size for quick tests.
        """
        self.image_dir = image_dir
        self.tokenizer_type = tokenizer_type
        self.tokenizer_path = tokenizer_path
        self.image_size = image_size
        self.seq_length = seq_length
        self.mode = mode
        self.transform = transform
        self.encode_on_the_fly = encode_on_the_fly
        self.seed = seed
        self.max_samples = max_samples

        # Load image file paths
        self.image_paths = self._load_image_paths()
        if self.max_samples is not None:
            self.image_paths = self.image_paths[:self.max_samples]

        # Initialize tokenizer instance
        self.tokenizer = tokenizer_utils.get_tokenizer(tokenizer_type, tokenizer_path)

        # Optional: precompute token sequences for full dataset (disabled by default)
        # For memory efficiency, encode each on-the-fly unless precompute is desired

        # Set seed for reproducibility
        random.seed(self.seed)

    def _load_image_paths(self) -> List[str]:
        """
        Loads all image paths from the dataset directory, assuming ImageFolder structure.
        """
        # Supports standard ImageNet folder structure
        all_paths = []
        for split in ['train', 'val']:
            dir_path = os.path.join(self.image_dir, split)
            if os.path.exists(dir_path):
                for root, _, files in os.walk(dir_path):
                    for fname in files:
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            all_paths.append(os.path.join(root, fname))
        return all_paths

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: str) -> Image.Image:
        """
        Loads an image and applies resizing and normalization.
        """
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            # Default resize and tensor conversion
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return img

    def encode_image(self, image: Image.Image):
        """
        Encodes an image into token sequence using the tokenizer.
        """
        return self.tokenizer.encode(image)

    def decode_tokens(self, tokens):
        """
        Decodes token sequence back into an image PIL object.
        """
        return self.tokenizer.decode(tokens)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Loads and processes a single image and its token sequence.
        """
        img_path = self.image_paths[idx]
        img = self._load_image(img_path)

        # Encode image into tokens
        if self.encode_on_the_fly:
            tokens = self.encode_image(img)
        else:
            # For efficiency, precompute tokens and cache if desired (not implemented here)
            # Placeholder: always encode on the fly
            tokens = self.encode_image(img)

        # Convert tokens to appropriate tensor
        if self.tokenizer.mode == 'discrete':
            # Expect tokens as sequence of ints
            if len(tokens) < self.seq_length:
                # Pad with zeros
                padded_tokens = np.zeros(self.seq_length, dtype=np.int64)
                padded_tokens[:len(tokens)] = tokens
                tokens_tensor = torch.from_numpy(padded_tokens)
            elif len(tokens) > self.seq_length:
                # Truncate
                tokens_tensor = torch.from_numpy(np.array(tokens[:self.seq_length], dtype=np.int64))
            else:
                tokens_tensor = torch.from_numpy(np.array(tokens, dtype=np.int64))
            return img, tokens_tensor  # For training: input tokens
        
        elif self.tokenizer.mode == 'continuous':
            # For continuous tokens, tokens are numpy array of shape [seq_length, feature_dim]
            feats = tokens
            feats = feats.astype(np.float32)
            # Pad or truncate
            curr_len = feats.shape[0]
            feat_dim = feats.shape[1]
            if curr_len < self.seq_length:
                pad_feats = np.zeros((self.seq_length - curr_len, feat_dim), dtype=np.float32)
                feats = np.vstack([feats, pad_feats])
            elif curr_len > self.seq_length:
                feats = feats[:self.seq_length, :]
            tokens_tensor = torch.from_numpy(feats)
            return img, tokens_tensor  # For training: input features

        else:
            raise ValueError(f"Unknown tokenizer mode: {self.tokenizer.mode}")

    def get_batch(self, batch_size: int, shuffle: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a DataLoader for batching.
        """
        dataset = self
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        return dataloader
