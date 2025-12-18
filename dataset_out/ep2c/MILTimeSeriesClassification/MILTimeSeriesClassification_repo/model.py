## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 128, architecture_params: dict = None):
        """
        Fully Convolutional Network backbone for TSC.
        Args:
            embedding_dim (int): Dimension of output feature embeddings (channels).
            architecture_params (dict): Hyperparameters like residual_blocks, kernel_sizes.
        """
        super().__init__()
        # Default parameters if not provided
        residual_blocks = architecture_params.get('residual_blocks', 4) if architecture_params else 4
        kernel_sizes = architecture_params.get('kernel_sizes', [8, 5, 3]) if architecture_params else [8, 5, 3]
        # Construct layers
        self.layers = nn.ModuleList()
        in_channels = 1
        for _ in range(residual_blocks):
            for k in kernel_sizes:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, embedding_dim, kernel_size=k, padding=k//2),
                        nn.BatchNorm1d(embedding_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                )
                in_channels = embedding_dim
        self.final_conv = nn.Conv1d(in_channels, embedding_dim, kernel_size=1)

    def forward(self, x):
        """
        Input:
            x: (batch_size, 1, t)
        Output:
            embeddings: (batch_size, t, embedding_dim)
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.final_conv(out)
        out = out.transpose(1, 2)  # (batch, t, embedding_dim)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNetBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 128, architecture_params: dict = None):
        """
        ResNet backbone for TSC.
        Args:
            embedding_dim (int): output channels.
            architecture_params: includes 'residual_blocks' (int).
        """
        super().__init__()
        residual_blocks = architecture_params.get('residual_blocks', 4) if architecture_params else 4
        kernel_sizes = architecture_params.get('kernel_sizes', [8, 5, 3]) if architecture_params else [8, 5, 3]

        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, embedding_dim, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for _ in range(residual_blocks):
            self.res_blocks.append(ResNetBlock(embedding_dim, kernel_size=kernel_sizes[1]))

        self.final_conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)

    def forward(self, x):
        """
        Input:
            x: (batch_size, 1, t)
        Output:
            embeddings: (batch_size, t, embedding_dim)
        """
        out = self.initial_conv(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.final_conv(out)
        out = out.transpose(1, 2)  # (batch, t, embedding_dim)
        return out

class InceptionTimeBlock(nn.Module):
    def __init__(self, channels: int, kernel_sizes: list):
        super().__init__()
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=ks, padding=ks//2),
                    nn.BatchNorm1d(channels),
                    nn.ReLU()
                )
            )
        self.conv1x1 = nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))
        concat = torch.cat(branch_outs, dim=1)
        out = self.conv1x1(concat)
        out = self.bn(out)
        out = self.relu(out)
        return out

class InceptionResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_sizes: list):
        super().__init__()
        self.inception = InceptionTimeBlock(channels, kernel_sizes)
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.inception(x)
        residual = self.residual_conv(x)
        residual = self.bn(residual)
        out += residual
        out = self.relu(out)
        return out

class InceptionTimeBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 128, architecture_params: dict = None):
        """
        InceptionTime backbone for TSC.
        Args:
            embedding_dim (int): output channels.
            architecture_params: includes 'residual_blocks' and 'kernel_sizes'.
        """
        super().__init__()
        residual_blocks = architecture_params.get('residual_blocks', 4) if architecture_params else 4
        kernel_sizes = architecture_params.get('kernel_sizes', [8, 5, 3]) if architecture_params else [8, 5, 3]
        # Initial Conv layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, embedding_dim, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        # Residual Blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(residual_blocks):
            self.res_blocks.append(InceptionResidualBlock(embedding_dim, kernel_sizes))
        # Final conv
        self.final_conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)

    def forward(self, x):
        """
        Input:
            x: (batch_size, 1, t)
        Output:
            embeddings: (batch_size, t, embedding_dim)
        """
        out = self.initial_conv(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.final_conv(out)
        out = out.transpose(1, 2)  # (batch, t, embedding_dim)
        return out

class BackboneNetwork(nn.Module):
    def __init__(self, architecture: str='FCN', embedding_dim: int=128, architecture_params: dict=None):
        """
        Factory backbone model, supports FCN, ResNet, InceptionTime.
        """
        super().__init__()
        arch = architecture.lower()
        if arch == 'fcn':
            self.model = FCNBackbone(embedding_dim, architecture_params)
        elif arch == 'resnet':
            self.model = ResNetBackbone(embedding_dim, architecture_params)
        elif arch == 'inceptiontime':
            self.model = InceptionTimeBackbone(embedding_dim, architecture_params)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x):
        """
        Forward pass delegates to the specific backbone.
        """
        return self.model(x)
