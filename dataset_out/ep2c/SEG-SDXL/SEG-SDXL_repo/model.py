## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention import AttentionLayer
from typing import Optional

class DiffusionModel(nn.Module):
    """
    Encapsulates a pretrained diffusion backbone (e.g., SDXL),
    supporting attention modules with Gaussian blur (SEG) during inference.
    """
    def __init__(self, architecture: str, pretrained_path: str):
        """
        Initialize the diffusion model by loading the pretrained checkpoint and 
        locating attention layers.
        
        Args:
            architecture (str): Name/identifier of the model architecture.
            pretrained_path (str): Path to the pretrained checkpoint.
        """
        super().__init__()
        self.architecture = architecture
        self.pretrained_path = pretrained_path
        
        # Load the pretrained model (assuming a simple loading mechanism here)
        # For the example, using a generic U-Net-like backbone.
        # Replace with actual loading code for SDXL in practice.
        self.model = self._load_pretrained_model()
        self.attention_layers = self._extract_attention_layers(self.model)
        
        # Set model to eval and possibly FP16 if needed
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_pretrained_model(self):
        """
        Load the actual pretrained model.
        Note: Placeholder implementation; replace with actual model code.
        """
        # For demonstration, suppose model is a nn.Module with attention modules.
        # Load checkpoint
        loaded_model = torch.load(self.pretrained_path, map_location='cpu')
        model = loaded_model.get('model', loaded_model)  # Adapt as per checkpoint format
        # Ensure in eval mode
        model.eval()
        return model
    
    def _extract_attention_layers(self, model):
        """
        Traverse the model to find all AttentionLayer instances.
        """
        attention_layers = []

        def recurse_modules(module):
            for child in module.children():
                if isinstance(child, AttentionLayer):
                    attention_layers.append(child)
                recurse_modules(child)
        recurse_modules(model)
        return attention_layers

    def get_attention_layers(self):
        """
        Return list of attention layers for external control or debugging.
        """
        return self.attention_layers

    def forward(self, x: torch.Tensor, guidance_scale: float = 1.0, sigma: float = 0.0, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        Integrates Gaussian blur into attention modules via guidance.
        
        Args:
            x (Tensor): Input tensor, noisy image or latent.
            guidance_scale (float): Guidance scale (ignored here, handled externally during sampling).
            sigma (float): Std dev for Gaussian blur in attention. Applies to all attention layers.
            conditioning (Tensor or None): Conditioning input (text, class, etc), optional.
        
        Returns:
            Tensor: Model prediction (e.g., noise prediction).
        """
        # During the forward, pass sigma to attention layers
        # For example, injecting sigma into attention modules
        # Here, assuming attention modules accept sigma as a parameter during call.
        # If not, we need to set a global parameter or hook.
        # The code below assumes each attention layer has a method 'set_sigma'
        # which sets the current sigma for that layer.
        
        # For simplicity, assign sigma to each attention layer
        for attn in self.attention_layers:
            if hasattr(attn, 'set_sigma'):
                attn.set_sigma(sigma)
        
        # Run the model forward
        # It is expected that AttentionLayer internally uses the set sigma
        # during attention computation.
        output = self.model(x, conditioning=conditioning)
        return output
