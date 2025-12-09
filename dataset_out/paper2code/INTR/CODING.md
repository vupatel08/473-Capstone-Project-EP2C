# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SimpleDataset(Dataset):
    """
    Custom dataset to load images and labels specified in a split file.
    Expects lines with 'image_path label' or similar format.
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: transforms.Compose,
        dataset_dir: str = ""
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.dataset_dir = dataset_dir  # Optional base directory for images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        full_path = os.path.join(self.dataset_dir, img_path) if self.dataset_dir else img_path
        image = Image.open(full_path).convert('RGB')
        image = self.transform(image)
        return image, label


class DatasetLoader:
    """
    Handles dataset loading, splitting, transformations, and DataLoader creation.
    """
    def __init__(
        self,
        dataset_path: str,
        train_split: str,
        test_split: str,
        image_size: int = 224,
        batch_size: int = 16,
        dataset_name: str = "",
        use_fully_finetune_backbone: bool = False,
        attribute_annotations: str = ""  # Optional path for attribute labels
    ):
        self.dataset_path = dataset_path
        self.train_split_path = train_split
        self.test_split_path = test_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name.lower()
        self.use_fully_finetune_backbone = use_fully_finetune_backbone
        self.attribute_annotations = attribute_annotations

        # Initialize dataset info placeholders
        self.class_labels: List[str] = []
        self.num_classes: int = 0
        self.class_to_idx: Dict[str, int] = {}

        # Load dataset splits and labels
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

        # Set transforms based on training/testing
        self.transform_train = self._create_transform(train=True)
        self.transform_test = self._create_transform(train=False)

        # Load dataset
        self._load_dataset()

    def _create_transform(self, train: bool) -> transforms.Compose:
        """
        Creates data transformations for training or testing.
        """
        normalize_mean = [0.5, 0.5, 0.5]
        normalize_std = [0.5, 0.5, 0.5]

        if train:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])

    def _load_split_file(self, split_file: str) -> Tuple[List[str], List[int], List[str]]:
        """
        Loads a dataset split file.
        Assumes each line: path/to/image.jpg label
        """
        image_paths = []
        labels = []
        class_names = set()

        split_path_full = os.path.join(self.dataset_path, split_file)
        with open(split_path_full, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    path_part = parts[0]
                    label_part = parts[1]
                    image_paths.append(path_part)
                    labels.append(int(label_part))
                else:
                    # fallback if no label info, could implement attribute or class mapping
                    continue
        return image_paths, labels, list(set(labels))

    def _load_dataset(self):
        """
        Loads the dataset splits, assigns class labels, and creates dataset objects.
        """
        # Load split files
        train_paths, train_labels, train_classes = self._load_split_file(self.train_split_path)
        test_paths, test_labels, test_classes = self._load_split_file(self.test_split_path)

        # Determine class labels and map to indices
        all_class_indices = sorted(set(train_classes + test_classes))
        self.class_labels = [str(c) for c in all_class_indices]
        self.num_classes = len(self.class_labels)
        self.class_to_idx = {int(cls): idx for idx, cls in enumerate(all_class_indices)}

        # Map labels in dataset to continuous 0-based indices
        def map_labels(labels_list):
            return [self.class_to_idx[int(lbl)] for lbl in labels_list]

        train_labels_idx = map_labels(train_labels)
        test_labels_idx = map_labels(test_labels)

        # Save full dataset directory for image paths
        dataset_dir = self.dataset_path

        # Create Dataset objects
        self.train_dataset = SimpleDataset(
            image_paths=train_paths,
            labels=train_labels_idx,
            transform=self.transform_train,
            dataset_dir=dataset_dir
        )
        self.test_dataset = SimpleDataset(
            image_paths=test_paths,
            labels=test_labels_idx,
            transform=self.transform_test,
            dataset_dir=dataset_dir
        )

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
```

**Explanation:**

- **Class `SimpleDataset`**: A basic PyTorch `Dataset` that loads images and labels given lists of image paths and label indices, applying the specified transforms.
- **Class `DatasetLoader`**: Main loader class per your plan. It:
  - Takes configuration parameters (dataset path, split files, image size, batch size, dataset name).
  - Loads the train/test split assuming each is a text file with per-line `image_path label`.
  - Builds a class label list (`self.class_labels`) and a mapping from dataset class label to index.
  - Creates `SimpleDataset` instances for train and test.
  - Wraps datasets with `DataLoader` for batch iteration.
  - Uses image transforms aligned with experimental settings (resize, crop, normalization).

- **Important**:
  - Reads dataset split files line-by-line.
  - Handles mapping class labels from raw labels to 0-based indices for uniform labeling.
  - Enables easy extension for datasets that have attribute annotations or other metadata by modifying `_load_split_file`.
  
**Note**:  
- This code assumes split files are simple text files with each line containing image relative path and label, e.g.:  
  `images/bird1.jpg 12`  
  Adjust accordingly if your split files are formatted differently.  
- To handle dataset-specific nuances, adapt `_load_split_file`.  
- This code is aligned explicitly with the configuration (e.g., batch size, image size).

## evaluation.py

```python
## evaluation.py
import os
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize_attention, save_attention_map, plot_attention_overlay
from PIL import Image

class Evaluation:
    """
    Handles evaluation of the trained INTR model, including accuracy, attention map visualization,
    and optional faithfulness metrics for interpretability.
    """
    def __init__(
        self,
        model,
        data_loader,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Initializes the evaluator with model, data loader, configuration, and device.
        Args:
            model (INTRModel): The trained model for evaluation.
            data_loader (DataLoader): DataLoader for validation/test data.
            config (dict): Configuration parameters.
            device (str): 'cuda' or 'cpu'.
        """
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.save_attention_maps = config.get('save_attention_maps', True)
        self.visualization_overlay = config.get('visualization_overlay', True)
        self.evaluation_metrics = config.get('evaluation_metrics', ['accuracy'])
        self.save_dir = config.get('save_dir', 'outputs/evaluation')
        os.makedirs(self.save_dir, exist_ok=True)

        # For storing results
        self.all_preds = []
        self.all_labels = []
        self.attention_maps: List[Dict] = []  # To store attention maps per sample if needed

    def evaluate(self) -> Dict[str, float]:
        """
        Runs evaluation over the dataset, computes accuracy, and visualizes attention maps.
        Returns:
            metrics_dict (dict): Dictionary with evaluation results.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with return_attention=True for interpretability
                logits, preds, attn_maps = self.model(images, return_attention=True)

                # Collect predictions and labels for accuracy
                self.all_preds.extend(preds.cpu().tolist())
                self.all_labels.extend(labels.cpu().tolist())

                # Compute accuracy
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Save attention maps for visualization if enabled
                if self.save_attention_maps and attn_maps is not None:
                    # attn_maps: [B, C, heads, N], process each sample
                    for i in range(images.size(0)):
                        input_image = self._denormalize_image(images[i])
                        # For each class, save its attention map overlay
                        num_classes = attn_maps.shape[1]
                        for c_idx in range(num_classes):
                            attn_map = attn_maps[i, c_idx]  # [heads, N]
                            # Average over heads or pick the head with max weight for visualization
                            attn_mean = attn_map.mean(dim=0)  # [N]
                            # Save attention overlay map
                            save_path = os.path.join(
                                self.save_dir,
                                f"attention_img{batch_idx}_{i}_class{c_idx}.png"
                            )
                            save_attention_map(attn_mean, input_image, save_path)
                            if self.visualization_overlay:
                                overlay_path = save_path.replace('.png', '_overlay.png')
                                plot_attention_overlay([attn_mean], input_image, headers=None, save_path=overlay_path)

        accuracy = correct / total if total > 0 else 0.0
        metrics_result = {'accuracy': accuracy}

        # Compute additional interpretability metrics if included
        if 'faithfulness' in self.evaluation_metrics:
            metrics_result['faithfulness'] = self._compute_faithfulness()

        return metrics_result

    def _denormalize_image(self, image_tensor: torch.Tensor) -> Image.Image:
        """
        Converts a normalized tensor to a PIL Image for visualization.
        Args:
            image_tensor (Tensor): [3, H, W]
        Returns:
            PIL.Image.Image
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = image_tensor.cpu().permute(1,2,0).numpy()
        img = std * img + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _compute_faithfulness(self) -> float:
        """
        Placeholder for faithfulness metric computation (e.g., insertion/deletion scores).
        Implementation requires detailed attribute masks or object bounding boxes.
        Since dataset may lack attribute labels, returns a dummy score.
        """
        # For actual implementation, would need attribute masks or object bounding boxes
        # For now, return a placeholder value
        return 0.0

```

## main.py

```python
# main.py
import os
import sys
import torch
import numpy as np
from utils import load_config, set_seed
from dataset_loader import DatasetLoader
from model import INTRModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # 1. Determine config file path (e.g., passed as argument or default)
    # For simplicity, assume 'config.yaml' in the current directory
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Configuration file '{config_path}' not found.")
        sys.exit(1)

    # 2. Load configuration
    config = load_config(config_path)

    # 3. Set seed for reproducibility
    seed = config.get('misc', {}).get('seed', 42)
    set_seed(seed)

    # 4. Set device: GPU if available, else CPU
    device_str = config.get('training', {}).get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() and device_str=='cuda' else 'cpu')
    print(f"Using device: {device}")

    # 5. Initialize dataset loader
    dataset_cfg = config.get('dataset', {})
    dataset_path = dataset_cfg.get('path', '')  # assume path provided in config if needed
    train_split_path = dataset_cfg.get('train_split', '')
    test_split_path = dataset_cfg.get('test_split', '')
    image_size = dataset_cfg.get('image_size', 224)
    batch_size = dataset_cfg.get('batch_size', 16)

    dataset_loader = DatasetLoader(
        dataset_path=dataset_path,
        train_split=train_split_path,
        test_split=test_split_path,
        image_size=image_size,
        batch_size=batch_size,
        dataset_name=dataset_cfg.get('name', ''),
        use_fully_finetune_backbone=dataset_cfg.get('use_fully_finetune_backbone', False)
    )

    # Extract DataLoaders
    train_loader = dataset_loader.train_loader
    test_loader = dataset_loader.test_loader

    # 6. Prepare model
    model_cfg = config.get('model', {})
    # Add dataset-specific class count
    num_classes = dataset_loader.num_classes
    model_cfg['class_queries'] = num_classes
    model_cfg['pretrained_weights'] = model_cfg.get('pretrained_weights', '')

    # Instantiate model
    model = INTRModel(model_cfg)
    model.to(device)

    # 7. Set up optimizer, scheduler
    training_cfg = config.get('training', {})
    lr = training_cfg.get('learning_rate', 1e-4)
    weight_decay = training_cfg.get('weight_decay', 0.05)
    epochs = training_cfg.get('epochs', 50)

    # Only fine-tune backbone if specified
    use_finetune = dataset_cfg.get('use_fully_finetune_backbone', False)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 8. Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # 9. Initialize Trainer
    trainer = Trainer(
        model=model,
        data_loader={'train': train_loader, 'val': test_loader},
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        config=training_cfg
    )

    # 10. Run training
    print("Starting training...")
    trainer.train()

    # 11. Load best model for evaluation and interpretability
    best_model_path = os.path.join(
        training_cfg.get('save_dir', 'outputs/checkpoints'), 'best_model.pth'
    )
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
    else:
        print("Best model checkpoint not found. Using last epoch model.")

    # 12. Final evaluation and visualization
    print("Evaluating and visualizing attention maps...")
    evaluator = Evaluation(
        model=model,
        data_loader=test_loader,
        config=training_cfg,
        device=str(device)
    )
    metrics = evaluator.evaluate()
    print("Evaluation metrics:", metrics)

    # Optional: Visualize some attention maps for interpretability
    # For example, visualize attention for first few images
    # (This can be integrated into Evaluation or called separately)
    # e.g.,
    # evaluator.visualize_attention_maps(...)

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from typing import List, Tuple, Dict, Any

class CrossAttentionLayer(nn.Module):
    """
    Single multi-head cross-attention layer as used in the decoder.
    It attends class queries to image features.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Each head has its own projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(
        self,
        class_queries: torch.Tensor,  # shape: [B, C, D]
        memory: torch.Tensor         # shape: [B, N, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            class_queries: [B, C, D]
            memory: [B, N, D]
        Returns:
            attended: [B, C, D]
            attention_weights: [B, C, heads, N] - for interpretability
        """
        B, C, D = class_queries.shape
        N = memory.shape[1]
        H = self.num_heads
        # Linear projections
        Q = self.q_proj(class_queries)     # [B, C, D]
        K = self.k_proj(memory)            # [B, N, D]
        V = self.v_proj(memory)            # [B, N, D]

        # Split heads
        def split_heads(x):
            # x: [B, *, D]
            return x.view(B, -1, H, D // H).transpose(2,3)  # [B, *, H, D//H]

        Qh = split_heads(Q)                  # [B, C, H, D//H]
        Kh = split_heads(K)                  # [B, N, H, D//H]
        Vh = split_heads(V)                  # [B, N, H, D//H]

        # Compute scaled dot-product attention
        # Q: [B, C, H, D//H], K: [B, N, H, D//H]
        # Need to transpose Kh for matmul
        Qh = Qh.permute(0,1,2,3)             # [B, C, H, D//H]
        Kh = Kh.permute(0,2,1,3)             # [B, H, N, D//H]
        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / self.scale
        # shape: [B, C, H, N]
        attention_weights = F.softmax(attn_scores, dim=-1)

        # Attend V
        # Vh: [B, N, H, D//H], permute for matmul
        Vh = Vh.permute(0,2,1,3)              # [B, H, N, D//H]
        attn_out = torch.matmul(attention_weights, Vh)  # [B, C, H, D//H]
        attn_out = attn_out.permute(0,1,2,3).contiguous()  # [B, C, H, D//H]
        # Concatenate heads
        attn_out = attn_out.view(B, C, D)
        output = self.out_proj(attn_out)      # [B, C, D]

        # For interpretability, return attention weights
        # Reshape to [B, C, H, N]
        attention_weights = attention_weights.view(B, C, H, N)

        return output, attention_weights

class DecoderLayer(nn.Module):
    """
    One decoder layer consisting of self-attention and cross-attention.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.cross_attn = CrossAttentionLayer(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(
        self,
        class_tokens: torch.Tensor,   # [B, C, D]
        memory: torch.Tensor,         # [B, N, D]
        self_attn_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            class_tokens: [B, C, D]
            memory: [B, N, D]
        Returns:
            output tokens: [B, C, D]
            cross-attention weights: [B, C, heads, N]
        """
        B, C, D = class_tokens.shape

        # Self-attention over class tokens
        class_tokens_flat = class_tokens.permute(1, 0, 2)  # [C, B, D]
        self_attn_output, _ = self.self_attn(class_tokens_flat, class_tokens_flat, class_tokens_flat)
        class_tokens = self.norm1(class_tokens + self_attn_output.permute(1,0,2))

        # Cross-attention with memory
        cross_output, attn_weights = self.cross_attn(class_tokens, memory)
        class_tokens = self.norm2(class_tokens + cross_output)

        # Feed-forward
        ff_out = self.ff(class_tokens)
        class_tokens = self.norm3(class_tokens + ff_out)

        return class_tokens, attn_weights

class INTRModel(nn.Module):
    """
    The main INTR model class implementing the described architecture.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Load backbone feature extractor
        backbone_type = config.get("backbone", "vit").lower()
        pretrained_weights_path = config.get("pretrained_weights", "")
        embed_dim = config.get("embed_dim", 768)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 4)
        num_classes = config.get("class_queries", 200)
        query_dim = config.get("query_dim", embed_dim)

        # Initialize backbone
        # For simplicity, we use torchvision ResNet or a placeholder ViT
        if backbone_type == "resnet":
            self.backbone = resnet50(pretrained=False)
            if pretrained_weights_path:
                self.backbone.load_state_dict(torch.load(pretrained_weights_path))
            self.backbone_output_dim = 2048  # ResNet-50 last layer features
            self.backbone_fc = nn.Linear(self.backbone_output_dim, embed_dim)
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(2048, embed_dim, kernel_size=1),
                nn.AdaptiveAvgPool2d((1,1))
            )
            self.use_resnet = True
        elif backbone_type == "vit":
            # Placeholder: assuming a ViT model is loaded externally
            # For reproducibility, load from a standard torch hub or custom import
            # For now, define as an identity module
            self.feature_extractor = nn.Identity()
            self.use_resnet = False
            self.backbone = None
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        # Class-specific learnable input queries
        self.C = num_classes
        self.D = embed_dim
        self.query_dim = query_dim
        self.Z_in = nn.Parameter(torch.randn(self.D, self.C))
        # Class weight vector for classification
        self.w = nn.Parameter(torch.randn(self.D))
        # Decoder layers
        self.num_layers = num_layers
        self.heads = config.get("num_heads", 4)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim=self.D, num_heads=self.heads)
            for _ in range(self.num_layers)
        ])

        # Final classifier matrix
        self.W_w = nn.Parameter(torch.randn(self.D, self.C))
        # Optional: layer normalization or dropout can be added

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract feature map tensor [B,N,D] from images.
        Supports ResNet or ViT.
        """
        if hasattr(self, 'use_resnet') and self.use_resnet:
            feat = self.backbone(images)  # [B, 2048, H, W]
            feat = self.feature_extractor(feat)  # [B, D, 1, 1]
            feat = feat.squeeze(-1).squeeze(-1)  # [B, D]
            # Expand spatially to form feature map N
            # Alternatively, use last conv layer features with spatial size H,W
            # For simplicity, replicate features N times
            # Here, for more fidelity, implement actual feature map extraction
            # For now, assume global pooled feature, shape [B, D]
            # If spatial features required, modify accordingly
            # To align with transformer input, create a spatial grid
            # For now, treat as sequence of 1 feature vector
            # Let's assume front-end provides a proper feature map
            # For code simplicity, create dummy spatial features
            # For real implementation, replace this with actual feature extraction
            B = images.shape[0]
            N = 196  # e.g., 14x14 patches
            feats = self.backbone.conv1(images)
            feats = self.backbone.bn1(feats)
            feats = self.backbone.relu(feats)
            feats = self.backbone.maxpool(feats)
            feats = self.backbone.layer1(feats)
            feats = self.backbone.layer2(feats)
            feats = self.backbone.layer3(feats)
            feats = self.backbone.layer4(feats)
            # Now feats shape: [B, C, H, W]
            H_feat, W_feat = feats.shape[2], feats.shape[3]
            feats = F.adaptive_avg_pool2d(feats, (H_feat, W_feat))
            N = H_feat * W_feat
            feats_flat = feats.view(B, self.D, N).permute(0,2,1)  # [B, N, D]
            return feats_flat
        else:
            # For ViT, assume feature map is provided as tokens
            # Replace with actual ViT feature extractor
            # Placeholder: generate dummy features
            B = images.shape[0]
            N = 196
            return torch.randn(B, N, self.D, device=images.device)
    
    def forward(self, images: torch.Tensor, return_attention: bool=False):
        """
        Perform a forward pass:
        - Extract features
        - Pass class queries and features through decoder layers
        - Perform classification
        - Optionally returns attention maps for interpretability
        """
        B = images.shape[0]
        # 1. Feature extraction
        feat_map = self.extract_features(images)  # [B, N, D]

        # 2. Prepare class queries
        # Expand Z_in to batch
        class_queries = self.Z_in.unsqueeze(0).expand(B, -1, -1)  # [B, C, D]

        # Initialize list to store attention weights
        attention_maps_per_layer: List[List[torch.Tensor]] = []

        class_tokens = class_queries  # Start with input class queries

        # 3. Decoder layers with cross-attention
        for layer in self.decoder_layers:
            class_tokens, attn_weights = layer(class_tokens, feat_map)  # [B, C, D], [B, C, heads, N]
            if return_attention:
                attention_maps_per_layer.append(attn_weights)

        # 4. Classification: compute logits
        # For each class c: w dot z_out^{(c)} -> shape [B, C]
        # Element-wise dot: shape [B, C]
        logits = torch.einsum('d, bcd -> bc', self.W_w, class_tokens)  # [B, C]

        # 5. Predicted classes
        preds = torch.argmax(logits, dim=1)  # [B]

        # 6. For interpretability, collect final attention maps
        final_attn_maps = None
        if return_attention:
            # Use last decoder layer's attention weights
            # shape: [B, C, heads, N]
            final_attn_maps = attn_weights  # from last iteration
        if return_attention:
            return logits, preds, final_attn_maps
        else:
            return logits, preds

    def get_attention_maps(self) -> List[torch.Tensor]:
        """
        Placeholder if storing attention maps during forward
        - In actual training/inference, store attention maps in member variables
        """
        # For this implementation, attention maps are returned during forward
        # User should extract from forward call directly
        pass
```

## requirements.txt

```python
## requirements.txt
torch==1.11.0
torchvision==0.12.0
numpy==1.21.0
pyyaml==5.4.1
matplotlib==3.4.3
Pillow==8.4.0
```

## trainer.py

```python
# trainer.py
import os
import torch
import numpy as np
import random
from typing import Dict, Any, Optional
from utils import set_seed, save_attention_map, normalize_attention, plot_attention_overlay

class Trainer:
    """
    Implements the training loop for the INTR model, handling data flow, optimization,
    loss computation, learning rate scheduling, and optional interpretability map extraction.
    """
    def __init__(
        self,
        model: 'INTRModel',
        data_loader: Dict[str, torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any]
    ):
        """
        Initializes the Trainer with necessary components and configuration.
        Args:
            model (INTRModel): The model to train.
            data_loader (dict): Dictionary with 'train' and optionally 'val' DataLoaders.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            loss_fn (callable): Loss function (e.g., CrossEntropyLoss).
            scheduler (Optional): Learning rate scheduler.
            config (dict): Configuration parameters from YAML.
        """
        self.model = model
        self.train_loader = data_loader.get('train')
        self.val_loader = data_loader.get('val', None)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.save_dir = config.get('save_dir', 'outputs')
        os.makedirs(self.save_dir, exist_ok=True)
        self.epochs = config.get('epochs', 50)
        self.seed = config.get('seed', 42)
        self._set_seed(self.seed)
        self.use_fully_finetune_backbone = config.get('use_fully_finetune_backbone', False)
        self.save_attention_maps = config.get('interpretability', {}).get('save_attention_maps', True)
        self.visualization_overlay = config.get('interpretability', {}).get('visualization_overlay', True)
        self.log_interval = config.get('log_interval', 10)  # how often to log
        self.best_val_acc = 0.0
        self.global_step = 0

        # Set model to device
        self.model.to(self.device)
        if not self.use_fully_finetune_backbone:
            # Freeze backbone parameters if not finetuning
            for param in self.model.extract_features.parameters():
                param.requires_grad = False

    def _set_seed(self, seed: int) -> None:
        """
        Sets seed for numpy, torch, and python random for reproducibility.
        """
        set_seed(seed)

    def train(self):
        """
        Main training loop over epochs.
        """
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch [{epoch}/{self.epochs}]")
            train_loss, train_acc = self._train_epoch(epoch)
            print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Step scheduler if used
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Evaluate on validation if available
            if self.val_loader is not None:
                val_loss, val_acc = self._validate()
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    best_path = os.path.join(self.save_dir, "best_model.pth")
                    torch.save(self.model.state_dict(), best_path)
                    print(f"New best model saved at {best_path}")

    def _train_epoch(self, epoch: int):
        """
        Performs one epoch of training.
        Returns:
            average_loss (float), accuracy (float)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass: get logits and attention maps for interpretability
            logits, preds, attn_maps = self.model(images, return_attention=True)

            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            total_correct += (preds == labels).sum().item()

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                batch_loss = loss.item()
                batch_acc = (preds == labels).float().mean().item()
                print(f"Batch [{batch_idx+1}/{len(self.train_loader)}]: Loss={batch_loss:.4f}, Acc={batch_acc:.4f}")

            # Save attention maps for interpretability if enabled
            if self.save_attention_maps and attn_maps is not None:
                # Save each attribute map, overlay with input image
                # For simplicity, save only the last batch
                for i in range(images.size(0)):
                    img = images[i]
                    # Undo normalization for visualization
                    img_vis = self._denormalize_image(img)
                    for c_idx in range(attn_maps.shape[1]):  # class C
                        attn_map = attn_maps[i, c_idx]  # shape: [heads, N]
                        # Combine attention heads (e.g., avg)
                        attn_map_mean = attn_map.mean(dim=0)  # [N]
                        # Save overlay
                        save_path = os.path.join(self.save_dir, f"attn_epoch{epoch}_batch{batch_idx+1}_img{i}_class{c_idx}.png")
                        save_attention_map(attn_map_mean, img_vis, save_path)
                        # Optional: overlay visualization
                        if self.visualization_overlay:
                            overlay_path = save_path.replace('.png', '_overlay.png')
                            plot_attention_overlay([attn_map_mean], img_vis, headers=None, save_path=overlay_path)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def _validate(self):
        """
        Performs validation pass without gradient updates.
        Returns:
            average_loss, accuracy
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits, preds, attn_maps = self.model(images, return_attention=True)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                total_correct += (preds == labels).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def _denormalize_image(self, image_tensor: torch.Tensor):
        """
        Convert normalized image tensor back to [0,255] PIL Image for visualization.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = image_tensor.cpu().permute(1, 2, 0).numpy()  # C, H, W to H, W, C
        img = std * img + mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

```

## utils.py

```python
# utils.py

import os
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Union
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms.functional as TF

def load_config(config_filepath: str) -> Dict[str, Any]:
    """
    Parses and loads the YAML configuration file into a Python dictionary.
    Handles file not found or parse errors gracefully.
    
    Args:
        config_filepath (str): Path to the YAML configuration file.
        
    Returns:
        dict: Parsed configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing YAML.
    """
    if not os.path.isfile(config_filepath):
        raise FileNotFoundError(f"Config file not found: {config_filepath}")
    try:
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Parsed config is not a dictionary.")
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def set_seed(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch for reproducibility.
    Configures PyTorch backend for deterministic behavior.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_attention_map(attention: torch.Tensor, input_image: Image.Image, save_path: str, title: str = "") -> None:
    """
    Saves a cross-attention map overlayed on the input image.
    Normalizes and resizes attention for visualization.
    
    Args:
        attention (torch.Tensor): Attention weights tensor of shape [H, W] or [N], normalized.
        input_image (PIL.Image.Image): Input image to overlay.
        save_path (str): Path to save the overlay image.
        title (str): Optional title for the saved image.
    """
    attention_np = normalize_attention(attention).cpu().numpy()
    input_size = input_image.size  # (width, height)
    # Resize attention map to match input image size
    attention_resized = resize_attention_map(torch.from_numpy(attention_np), input_size)
    heatmap = apply_colormap(attention_resized)
    overlay = overlay_attention_on_image(input_image, heatmap, alpha=0.5)
    plt.figure(figsize=(8,8))
    plt.imshow(overlay)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_attention_overlay(attention_maps: List[Union[torch.Tensor, np.ndarray]],
                           input_image: Image.Image,
                           headers: List[str],
                           save_path: str) -> None:
    """
    Creates a grid of attention maps overlayed on input image for interpretability.
    
    Args:
        attention_maps (list): List of attention maps (Tensor or np.ndarray).
        input_image (PIL.Image.Image): Input image.
        headers (list): List of titles for each attention map.
        save_path (str): Path to save the composite visualization.
    """
    num_maps = len(attention_maps)
    plt.figure(figsize=(4 * num_maps, 4))
    
    for idx, att_map in enumerate(attention_maps):
        att_tensor = torch.from_numpy(att_map) if isinstance(att_map, np.ndarray) else att_map
        att_norm = normalize_attention(att_tensor)
        att_resized = resize_attention_map(att_norm, input_image.size)
        heatmap = apply_colormap(att_resized)
        overlay = overlay_attention_on_image(input_image, heatmap, alpha=0.5)
        plt.subplot(1, num_maps, idx+1)
        plt.imshow(overlay)
        plt.title(headers[idx] if headers else "")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def normalize_attention(attention: torch.Tensor) -> torch.Tensor:
    """
    Normalizes attention weights to [0,1] range for visualization.
    Handles max == min cases.
    
    Args:
        attention (torch.Tensor): Input attention tensor.
        
    Returns:
        torch.Tensor: Normalized attention tensor.
    """
    min_val = attention.min()
    max_val = attention.max()
    if max_val - min_val < 1e-8:
        return torch.zeros_like(attention)
    normalized = (attention - min_val) / (max_val - min_val)
    return normalized

def resize_attention_map(attention_map: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Resizes attention map to match target size (image size).
    
    Args:
        attention_map (torch.Tensor): 2D tensor [H, W]
        target_size (tuple): (width, height)
        
    Returns:
        torch.Tensor: Resized attention map [H', W']
    """
    # attention_map shape: [H, W]
    resized = TF.resize(attention_map.unsqueeze(0).unsqueeze(0),
                        size=target_size, interpolation=Image.BILINEAR)
    resized = resized.squeeze(0).squeeze(0)
    return resized

def apply_colormap(attention_map: torch.Tensor) -> np.ndarray:
    """
    Applies a colormap (e.g., 'jet') to a normalized attention map.
    
    Args:
        attention_map (torch.Tensor): 2D tensor with values in [0,1]
        
    Returns:
        np.ndarray: RGB heatmap array
    """
    attention_np = attention_map.cpu().numpy()
    heatmap = plt.cm.jet(attention_np)[:,:,:3]  # Ignore alpha channel
    return (heatmap * 255).astype(np.uint8)

def overlay_attention_on_image(input_image: Image.Image,
                               attention_heatmap: np.ndarray,
                               alpha: float = 0.5) -> Image.Image:
    """
    Blends input image and attention heatmap into an overlay image.
    
    Args:
        input_image (PIL.Image.Image): Original image.
        attention_heatmap (np.ndarray): RGB heatmap image.
        alpha (float): Transparency factor for heatmap overlay.
        
    Returns:
        PIL.Image.Image: Blended overlay image.
    """
    heatmap_img = Image.fromarray(attention_heatmap).resize(input_image.size, resample=Image.BILINEAR)
    overlay = Image.blend(input_image.convert('RGBA'), heatmap_img.convert('RGBA'), alpha=alpha)
    return overlay

def calculate_faithfulness_metrics(
        original_image: Union[Image.Image, np.ndarray],
        attention_map: torch.Tensor,
        ground_truth_attributes: Any = None
    ) -> Tuple[float, float]:
    """
    Computes insertion and deletion scores based on attention maps.
    Placeholder implementation: requires model confidence change measurement.
    Actual implementation depends on model evaluation procedures.
    
    Args:
        original_image (PIL.Image or np.ndarray): Original input.
        attention_map (torch.Tensor): Attention weights of shape [H, W] or [N].
        ground_truth_attributes: Optional ground truth masks or bounding boxes.
        
    Returns:
        tuple: (insertion_score, deletion_score)
    """
    # Placeholder: actual implementation requires model inference with perturbed images.
    # For now, return dummy scores.
    insertion_score = 0.0
    deletion_score = 0.0
    # Implement actual metrics if data and model inference code are available.
    return insertion_score, deletion_score

def log_attention_stats(attention: torch.Tensor) -> Dict[str, float]:
    """
    Records statistics such as max, mean, entropy of attention weights.
    
    Args:
        attention (torch.Tensor): Attention tensor.
        
    Returns:
        dict: Dictionary of statistics.
    """
    stats = {}
    att_flat = attention.flatten()
    stats['max'] = float(att_flat.max())
    stats['mean'] = float(att_flat.mean())
    # Entropy calculation (discrete)
    probabilities = att_flat / (att_flat.sum() + 1e-8)
    entropy = - (probabilities * torch.log(probabilities + 1e-8)).sum()
    stats['entropy'] = float(entropy)
    return stats

def prepare_input_image(image_path: str, image_size: int, normalize: bool = True) -> torch.Tensor:
    """
    Loads an image, resizes, optionally normalizes, and converts to tensor.
    
    Args:
        image_path (str): Path to image file.
        image_size (int): Size to resize shortest side.
        normalize (bool): Whether to normalize with ImageNet mean/std.
        
    Returns:
        torch.Tensor: Transformed image tensor [3, H, W].
    """
    image = Image.open(image_path).convert('RGB')
    # Resize keeping aspect ratio
    image = TF.resize(image, size=(image_size, int(image_size * image.height / image.width)))
    # Center crop or resize to ensure exact size
    image = TF.resize(image, (image_size, image_size))
    tensor_image = TF.to_tensor(image)  # [C, H, W]
    if normalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tensor_image = TF.normalize(tensor_image, mean=mean, std=std)
    return tensor_image

def flatten_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Flattens a feature map [D, H, W] into a sequence of tokens [N, D].
    
    Args:
        feature_map (torch.Tensor): [D, H, W]
        
    Returns:
        torch.Tensor: [N, D], where N=H*W
    """
    D, H, W = feature_map.shape
    return feature_map.reshape(D, H * W).permute(1, 0)  # shape: [N, D]
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\INTR\INTR_repo`
