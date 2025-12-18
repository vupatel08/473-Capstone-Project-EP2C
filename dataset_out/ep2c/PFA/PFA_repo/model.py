# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Dict, Optional, Tuple
import timm

class SegformerEncoder(nn.Module):
    """
    Segformer (MiT-B1) encoder based on timm's implementation.
    Extracts multi-scale features from the backbone.
    """
    def __init__(self, backbone_name: str = "mit_b1", pretrained: bool = True):
        super().__init__()
        # Load a pre-trained Segformer model from timm
        # Using timm, which supports 'mit_b1' as pre-trained model
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        # The backbone outputs features from 4 stages with different resolutions
        # For modularity, define feature extraction layers
        # We assume the backbone provides 'forward_features' method or similar
        # ttm's models may need custom extraction; here, rely on its standard interface
        # For simplicity, assume 'forward_features' returns list of features
        # If not, need to modify accordingly
        self.out_channels = [64, 128, 320, 512]  # for MiT-B1
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Use the backbone's forward_features method
        features = self.backbone.forward_features(x)
        # features is a list of tensors: [F1, F2, F3, F4]
        # Ensure they are in order from high resolution to low resolution
        return features


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for semantic segmentation.
    Fuses multi-scale features and refines into logits.
    """
    def __init__(self, num_classes: int = 21, feature_channels: List[int] = [64, 128, 320, 512]):
        super().__init__()
        # We will fuse multi-level features
        # Upsample lower resolution features and concatenate
        # For simplicity, use a series of convolutions and transposed convs
        self.conv1 = nn.Conv2d(feature_channels[-1], 256, kernel_size=1)
        self.conv2 = nn.Conv2d(feature_channels[-2], 256, kernel_size=1)
        self.conv3 = nn.Conv2d(feature_channels[-3], 256, kernel_size=1)
        self.conv4 = nn.Conv2d(feature_channels[-4], 256, kernel_size=1)
        
        # Decoder layers (could be transformer blocks, here simple convs)
        self.decode_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def fuse_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        '''
        Fuse multi-scale features into a single high-resolution feature map.
        '''
        # Expect features: [F1, F2, F3, F4]
        # Upsample lower resolution features to F1 size
        size = features[0].size()[2:]  # H, W
        upsampled = []
        for i, feat in enumerate(features):
            if feat.size()[2:] != size:
                feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
            upsampled.append(feat)
        # Concatenate all features along channel dimension
        fused = torch.cat(upsampled, dim=1)  # shape: (B, sum of channels, H, W)
        return fused

    def forward(self, features: List[torch.Tensor], prototypes: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        '''
        Forward pass, with optional augmentation via prototypes.
        '''
        fused = self.fuse_features(features)
        # Optionally, apply prototype-guided augmentation (implemented externally)
        # For core decoder, simply pass through
        logits = self.decode_head(fused)
        return logits


class PrototypeExtractor(nn.Module):
    """
    Extracts high-confidence feature prototypes for each class.
    """
    def __init__(self, topk_percentage: float = 0.5):
        super().__init__()
        self.topk_percentage = topk_percentage  # e.g., 0.5 for top 50%
        
    def compute_prototypes(self, features: torch.Tensor, predictions: torch.Tensor,
                           labels: torch.Tensor, class_ids: List[int]) -> Dict[int, torch.Tensor]:
        """
        Compute class-wise prototypes from high-confidence pixels.
        Args:
            features: [B, D, H, W]
            predictions: [B, C, H, W]
            labels: [B, H, W], sparse labels with ignore_idx for unlabeled
            class_ids: list of classes present in current batch
        Returns:
            prototypes: dict {class_idx: [K, D]} with K=number of prototypes per class
        """
        B, D, H, W = features.shape
        # Flatten spatial dimensions
        feats_flat = features.view(B, D, -1)  # [B, D, HW]
        pred_flat = predictions.view(B, predictions.shape[1], -1)  # [B, C, HW]
        label_flat = labels.view(B, -1)  # [B, HW]
        
        prototypes = {}
        for c in class_ids:
            class_prototypes = []
            for b in range(B):
                # Create mask for class c, excluding ignore_idx
                mask = (label_flat[b] == c)
                if mask.sum() == 0:
                    continue
                # Select confidence scores for class c
                class_pred_confidence = pred_flat[b, c, :]
                # Get top-k pixels
                k = max(1, int(self.topk_percentage * mask.sum().item()))
                topk_confidence, topk_indices = torch.topk(class_pred_confidence[mask], k=k)
                # Gather features for top pixels
                selected_indices = torch.where(mask)[0][topk_indices]
                sel_feats = feats_flat[b, :, selected_indices]  # [D, K]
                # Compute weighted average
                # Use confidence scores if desired for weighting
                # Here, simple average
                proto = sel_feats.mean(dim=1)  # [D]
                # Normalize
                proto = F.normalize(proto, p=2, dim=0)
                class_prototypes.append(proto)
            if len(class_prototypes) > 0:
                # Average over batch instances
                class_proto = torch.stack(class_prototypes, dim=0).mean(dim=0)  # [D]
                prototypes[c] = class_proto
        return prototypes


class FeatureAugmenter(nn.Module):
    """
    Augments features using prototypes via attention.
    Can be used with local or global prototypes.
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(2 * feature_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Linear(feature_dim, feature_dim)
        
    def augment_with_prototypes(self, features: torch.Tensor, prototypes: torch.Tensor, class_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform feature augmentation with prototypes.
        Args:
            features: [B, D, H, W]
            prototypes: [C, K, D] or [C, D]
            class_mask: [B, H, W], indicating class at each pixel
        Returns:
            augmented_features: [B, D, H, W]
        """
        B, D, H, W = features.shape
        device = features.device

        # Reshape features to [B, D, H*W]
        feat_flat = features.view(B, D, -1)  # [B, D, HW]
        feat_flat_t = feat_flat.permute(0, 2, 1)  # [B, HW, D]

        # For simplicity, assume prototypes are [C, D]
        # For each pixel, get the prototypes of its class using class_mask
        # We'll process cross class, for high efficiency, process per class
        # Create a new tensor for augmented features
        augmented_feat = feat_flat.clone()

        for c in range(prototypes.shape[0]):  # For each class
            class_indices = (class_mask == c).nonzero(as_tuple=False)
            if class_indices.shape[0] == 0:
                continue
            class_proto = prototypes[c]  # [D]
            # Expand to match features of class pixels
            proto_expanded = class_proto.unsqueeze(0).unsqueeze(0)  # [1,1,D]
            class_feat_indices = class_indices[:, 1:]  # [N, 3], spatial coords
            for idx in class_feat_indices:
                b_idx, y, x = idx
                feat_vector = features[b_idx, :, y, x]  # [D]
                # Compute attention weight via dot product
                attn = torch.matmul(feat_vector, class_proto)  # scalar
                # Attention weight via softmax over prototypes of this class
                # For simplicity, use sigmoid or softmax over 1 scalar
                # Here, since only one prototype, set weight
                weight = torch.sigmoid(attn)
                # Interpolated prototype
                proto_weighted = weight * class_proto
                # Linear transform
                feat_aug = self.linear(torch.cat([feat_vector, proto_weighted], dim=0))
                feat_aug = self.relu(feat_aug)
                # Residual connection
                features[b_idx, :, y, x] += self.proj(feat_aug)
        return features


class PrototypeMemoryBank:
    """
    Manages global prototypes for each class, updated with cosine similarity.
    """
    def __init__(self, num_classes: int = 21, prototypes_per_class: int = 5, feature_dim: int = 256,
                 momentum: float = 0.99):
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.feature_dim = feature_dim
        self.momentum = momentum
        # Initialize memory bank: tensor [C, K, D]
        self.prototypes = torch.zeros((self.num_classes, self.prototypes_per_class, self.feature_dim))
        self.full_mask = torch.zeros((self.num_classes,), dtype=torch.bool)  # track if full
        # For simplicity, fill with zeros initially
        self.initialized_counts = torch.zeros((self.num_classes,), dtype=torch.long)
        
    def update(self, class_indices: List[int], new_prototypes: torch.Tensor):
        """
        Update prototypes for given class indices.
        Args:
            class_indices: list of class indices [batch_size]
            new_prototypes: [len(class_indices), D]
        """
        for idx, c in enumerate(class_indices):
            proto = new_prototypes[idx]  # [D]
            if not self.full_mask[c]:
                # Fill remaining slots
                count = self.initialized_counts[c]
                remaining = self.prototypes_per_class - count
                if remaining >= 1:
                    self.prototypes[c, count:count+1, :] = proto.unsqueeze(0)
                    self.initialized_counts[c] += 1
                if self.initialized_counts[c] >= self.prototypes_per_class:
                    self.full_mask[c] = True
            else:
                # Replace the most similar prototype
                bank_protos = self.prototypes[c]  # [K, D]
                # Compute cosine similarity
                sim = F.cosine_similarity(bank_protos, proto.unsqueeze(0), dim=1)  # [K]
                min_sim_idx = torch.argmin(sim)
                # Update with momentum
                self.prototypes[c, min_sim_idx, :] = self.momentum * bank_protos[min_sim_idx] + (1 - self.momentum) * proto
                self.prototypes[c, min_sim_idx, :] = F.normalize(self.prototypes[c, min_sim_idx, :], p=2, dim=0)
    
    def get(self) -> torch.Tensor:
        """
        Return prototypes: shape [C, K, D]
        """
        return self.prototypes


class ScribbleSegModel(nn.Module):
    """
    Complete segmentation model with backbone, decoder, and prototype handling.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.num_classes = config['model'].get('num_classes', 21)
        self.backbone_name = config['model'].get('backbone', 'mit_b1')
        self.prototype_num_per_class = config['model'].get('proto_num_per_class', 5)
        self.prototype_momentum = config['model'].get('proto_momentum', 0.99)
        self.prototype_extraction_topk = config['model'].get('prototype_extraction_topk', 0.5)
        
        # Initialize backbone encoder
        self.encoder = SegformerEncoder(self.backbone_name)
        feat_channels = self.encoder.out_channels  # [64, 128, 320, 512]
        # For this model, previous code assumes features are 256 channels, so unify
        # Apply a projection for each feature to D
        self.feat_dim = 256
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(c, self.feat_dim, kernel_size=1) for c in feat_channels
        ])
        # Initialize decoder
        self.decoder = TransformerDecoder(num_classes=self.num_classes)
        # Initialize prototype extractor
        self.prototype_extractor = PrototypeExtractor(topk_percentage=self.prototype_extraction_topk)
        # Initialize feature augmenters
        self.local_augmenter = FeatureAugmenter(feature_dim=self.feat_dim)
        self.global_augmenter = FeatureAugmenter(feature_dim=self.feat_dim)
        # Initialize global prototype memory bank
        self.global_proto_bank = PrototypeMemoryBank(
            num_classes=self.num_classes,
            prototypes_per_class=self.prototype_num_per_class,
            feature_dim=self.feat_dim,
            momentum=self.prototype_momentum
        )
        # Flags for training phases
        self.global_prototypes_full = False
        self.use_prototypes = True  # can be toggled during training schedule
        # Placeholder for prototypes during training
        self.local_prototypes: Dict[int, torch.Tensor] = {}
        self.global_prototypes: torch.Tensor = torch.zeros(
            (self.num_classes, self.prototype_num_per_class, self.feat_dim),
            device='cpu'
        )
        
    def forward(self, images: torch.Tensor, predictions: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, training_phase: str = 'warmup') -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        Args:
            images: [B, 3, H, W]
            predictions: optional initial predictions [B, C, H, W]
            labels: optional labels for prototype extraction
            training_phase: 'warmup' / 'local_proto' / 'full_proto'
        Returns:
            logits: segmentation logits [B, C, H, W]
            extra_outputs: dict with prototypes etc.
        """
        B = images.size(0)
        # Encode
        feats = self.encoder.extract_features(images)  # List of 4 features
        # Project features to fix dimension
        feats_proj = [layer(feats[i]) for i, layer in enumerate(self.proj_layers)]
        # Compose feature tensor storing concatenated features
        # For augmentation, keep separate
        feats_for_aug = [feat for feat in feats_proj]
        # Predicted logits before augmentation
        preds = self.decoder(feats_for_aug)

        # Prepare for prototype extraction
        proto_info = {}
        if self.use_prototypes and (training_phase != 'warmup'):
            with torch.no_grad():
                # Use predictions and features to extract prototypes
                # Only proceed if labels provided
                if labels is not None:
                    # Compute class set present in batch
                    class_mask = labels.unique()
                    class_mask = class_mask[class_mask != 255].tolist()
                else:
                    class_mask = list(range(self.num_classes))
                # Compute class-wise prototypes from high-confidence pixels
                # Use features from the last layer for prototype extraction
                last_feat = feats_for_aug[-1]  # [B, D, H, W]
                pred_probs = F.softmax(preds, dim=1)
                # Extract local prototypes
                local_proto_dict = self.prototype_extractor.compute_prototypes(
                    last_feat, pred_probs, labels if labels is not None else torch.full_like(labels, 255), class_mask
                )
                # Store local prototypes
                self.local_prototypes = local_proto_dict

                # Update global prototypes if in full phase
                if self.global_prototypes_full:
                    class_idxs_empty = []
                    class_protos_tensor = []
                    for c in class_mask:
                        if c in local_proto_dict:
                            class_idxs_empty.append(c)
                            class_protos_tensor.append(local_proto_dict[c])
                    if len(class_idxs_empty) > 0:
                        # update memory bank
                        self.global_proto_bank.update(class_idxs_empty, torch.stack(class_protos_tensor))
                        # Update global prototypes tensor
                        self.global_prototypes = self.global_proto_bank.get()
        # Apply feature augmentation
        augmented_feats = feats_for_aug.copy()

        if self.use_prototypes:
            # For local prototypes
            for c, proto in self.local_prototypes.items():
                # Create class mask map
                class_mask_map = (labels == c).to(torch.long) if labels is not None else None
                if class_mask_map is not None:
                    class_mask_map = class_mask_map.squeeze(1)  # [B, H, W]
                else:
                    # fallback: assign zeros
                    class_mask_map = torch.zeros_like(labels)
                # Augment features for each level
                for lvl in range(len(augmented_feats)):
                    augmented_feats[lvl] = self.local_augmenter.augment_with_prototypes(
                        augmented_feats[lvl], proto.unsqueeze(0), class_mask_map
                    )
            # For global prototypes
            if self.global_prototypes_full:
                for c in range(self.num_classes):
                    proto = self.global_prototypes[c, :, :]  # [K, D]
                    if self.global_proto_bank.full_mask[c]:
                        class_mask_map = (labels == c).to(torch.long) if labels is not None else None
                        if class_mask_map is not None:
                            class_mask_map = class_mask_map.squeeze(1)
                        else:
                            class_mask_map = torch.zeros_like(labels)
                        for lvl in range(len(augmented_feats)):
                            augmented_feats[lvl] = self.global_augmenter.augment_with_prototypes(
                                augmented_feats[lvl], proto, class_mask_map
                            )

        # Recompute logits with augmented features
        final_logits = self.decoder(augmented_feats)

        return final_logits, {
            'local_prototypes': self.local_prototypes,
            'global_prototypes': self.global_prototypes
        }
