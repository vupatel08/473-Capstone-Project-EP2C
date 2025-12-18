## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50  # Placeholder, replace with actual backbone if needed
from collections import OrderedDict

class DetectionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Extract configuration parameters
        self.backbone_name = config['model'].get('backbone', 'darknet53')
        self.pretrain = config['model'].get('backbone_pretrain', True)
        self.kernel_size_list = config['model'].get('kernel_size_options', [3, 5])
        self.num_kernels = config['model'].get('num_kernels', 4)
        self.fuse_method = config['model'].get('fusion_method', 'concat')
        # Initialize backbone
        self.backbone = self._build_backbone(self.backbone_name, self.pretrain)
        # Initialize IIM with multiple kernels
        self.kernels = nn.ParameterList()
        for size in self.kernel_size_list:
            for _ in range(self.num_kernels):
                kernel = self._initialize_kernel(size)
                self.kernels.append(kernel)
        # Fusion layer: simple concatenation followed by 1x1 conv (can be replaced)
        fused_channels = self._calculate_fused_channels()
        self.fuse_conv = nn.Conv2d(fused_channels, fused_channels, kernel_size=1)
        # Detection head: Placeholder, replace with actual YOLOv3 or TOOD head
        self.det_head = self._build_detection_head()

    def _build_backbone(self, backbone_name, pretrained):
        # Here, we can load a Darknet-53 or Transformer backbone accordingly
        # For simplicity, using ResNet50 as placeholder
        # Replace with actual Darknet implementation if available
        backbone = resnet50(pretrained=pretrained)
        # Extract feature layers for detection, e.g., layer2, layer3, layer4
        # For actual implementation, adapt accordingly
        self.backbone_out_channels = [512, 1024, 2048]  # Placeholder
        return backbone

    def _initialize_kernel(self, size: int):
        # Initialize kernels with physics-inspired priors (cross-color ratios)
        # For demonstration, create simple approximate weights
        # For actual physics-based init, use equations from the paper
        weight_shape = (self.num_kernels, 3, size, size)  # for 3 input channels
        kernel = torch.zeros(weight_shape)
        for i in range(self.num_kernels):
            for c in range(3):
                # Example: random small weights around 0, or physics-based ratios
                # For illustration:
                if c == 0:  # R channel
                    kernel[i, c] = torch.full((size, size), fill_value=0.1 * (i+1))
                elif c == 1:  # G channel
                    kernel[i, c] = torch.full((size, size), fill_value=-0.1 * (i+1))
                else:  # B channel
                    kernel[i, c] = torch.full((size, size), fill_value=0.05 * (i+1))
        return nn.Parameter(kernel)

    def _calculate_fused_channels(self):
        # Determine number of output channels after fusion
        # For concat, sum channel dims; placeholder assumes specific
        total_channels = sum(self.backbone_out_channels)
        if self.fuse_method == 'concat':
            return total_channels
        elif self.fuse_method == 'add':
            return total_channels
        else:
            return total_channels

    def _build_detection_head(self):
        # Placeholder, replace with actual detection head (e.g., YOLO layer or TOOD head)
        # For illustration, a simple Conv2d output
        return nn.Conv2d(self._calculate_fused_channels(), 255, kernel_size=1)

    def log_feature_transform(self, x):
        # Helper to compute log of input
        # Clamp to prevent log(0)
        eps = 1e-6
        return torch.log(torch.clamp(x + eps, min=eps))

    def apply_zero_mean_projection(self):
        # Enforce zero-mean constraint on kernels after each update
        with torch.no_grad():
            for i in range(len(self.kernels)):
                kernel = self.kernels[i]
                mean = torch.mean(kernel, dim=[1,2,3], keepdim=True)
                self.kernels[i].data -= mean

    def forward(self, x):
        """
        Args:
            x: Input image tensor, shape (B, 3, H, W)
        Returns:
            detections: detection outputs (boxes, scores, labels)
            features: intermediate features for visualization if needed
        """
        # Extract backbone features
        features_dict = self._extract_backbone_features(x)
        backbone_feats = list(features_dict.values())  # assume dict with feature maps

        # Compute log of channels
        log_channels = [self.log_feature_transform(feat) for feat in backbone_feats]

        # Compute IIM features from kernels
        iim_feats = []
        for kernel in self.kernels:
            # Apply convolution per kernel to each log channel
            # Shape: (B, 1, H, W)
            # Because kernel shape: (out_channels, in_channels, k, k)
            kernel = self._enforce_zero_mean(kernel)
            out = []
            for c_idx, log_feat in enumerate(log_channels):
                conv = F.conv2d(log_feat, kernel[c_idx:c_idx+1], padding=kernel.shape[2]//2)
                out.append(conv)
            # Sum across channels to form kernel-specific feature
            fk = sum(out)
            iim_feats.append(fk)
        # Concatenate features from all kernels
        iim_features = torch.cat(iim_feats, dim=1)  # shape: (B, num_kernels * 1, H, W)

        # Fuse with backbone features
        # For simplicity, concatenate with last backbone feature
        if self.fuse_method == 'concat':
            fused_feats = torch.cat([backbone_feats[-1], iim_features], dim=1)
        elif self.fuse_method == 'add':
            # Make sure channels match
            fused_feats = backbone_feats[-1] + iim_features
        else:
            fused_feats = torch.cat([backbone_feats[-1], iim_features], dim=1)

        fused_feats = self.fuse_conv(fused_feats)

        # Detection head
        detections = self.det_head(fused_feats)

        return detections

    def _extract_backbone_features(self, x):
        # Run backbone and extract features at desired layers
        # Placeholder: use resnet's output features
        # For actual Darknet or detector backbone, adapt accordingly
        # Example with ResNet:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        feat1 = self.backbone.layer1(x)  # e.g., 64
        feat2 = self.backbone.layer2(feat1) # e.g., 128
        feat3 = self.backbone.layer3(feat2) # e.g., 256
        feat4 = self.backbone.layer4(feat3) # e.g., 512
        # Returning as dict to be flexible
        return OrderedDict([
            ('layer1', feat1),
            ('layer2', feat2),
            ('layer3', feat3),
            ('layer4', feat4),
        ])

    def _enforce_zero_mean(self, kernel: torch.nn.Parameter):
        # Project the kernel weights to satisfy zero-mean constraint
        mean = torch.mean(kernel)
        kernel.data -= mean
        return kernel

    def get_invariant_features(self, x):
        # Optional: method to extract invariant features for visualization
        log_channels = [self.log_feature_transform(feat) for feat in self._extract_backbone_features(x).values()]
        iim_feats = []
        for kernel in self.kernels:
            kernel = self._enforce_zero_mean(kernel)
            out = []
            for c_idx, log_feat in enumerate(log_channels):
                conv = F.conv2d(log_feat, kernel[c_idx:c_idx+1], padding=kernel.shape[2]//2)
                out.append(conv)
            fk = sum(out)
            iim_feats.append(fk)
        features = torch.cat(iim_feats, dim=1)
        return features

