## model.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModel, BertConfig, ViTModel
from typing import Optional, Dict, Any

class ImportanceScaledLinear(nn.Module):
    """
    A linear layer that supports importance sampling by applying importance masks and scales
    in the forward pass for unbiased gradient estimation.
    This layer can be used for attention projection layers or linear feed-forward layers
    in transformers, and can be adapted for other linear layers.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Buffers to hold importance mask and scale for weights
        self.register_buffer('importance_mask', torch.ones_like(self.linear.weight))
        self.register_buffer('importance_scale', torch.ones_like(self.linear.weight))
        # Buffers for bias if used
        if bias:
            self.register_buffer('importance_mask_bias', torch.ones_like(self.linear.bias))
            self.register_buffer('importance_scale_bias', torch.ones_like(self.linear.bias))
        else:
            self.importance_mask_bias = None
            self.importance_scale_bias = None

    def set_importance_mask_scale(self, mask: torch.Tensor, scale: torch.Tensor, bias_mask: Optional[torch.Tensor]=None, bias_scale: Optional[torch.Tensor]=None):
        """
        Set the importance mask and scale for weights (and bias if applicable).
        Args:
            mask (Tensor): importance masking tensor (same shape as weights)
            scale (Tensor): scaling factor tensor (inverse of sampling probability)
            bias_mask (Optional[Tensor]): mask for bias
            bias_scale (Optional[Tensor]): scale for bias
        """
        self.importance_mask.data.copy_(mask)
        self.importance_scale.data.copy_(scale)
        if bias_mask is not None:
            self.importance_mask_bias.data.copy_(bias_mask)
        if bias_scale is not None:
            self.importance_scale_bias.data.copy_(bias_scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward with importance sampling scaling applied to weights/bias.
        """
        weight = self.linear.weight * self.importance_mask * self.importance_scale
        bias = None
        if self.linear.bias is not None:
            bias = self.linear.bias * self.importance_mask_bias * self.importance_scale_bias
        return nn.functional.linear(input, weight, bias)

class ImportanceScaledAttention(nn.Module):
    """
    Wraps a MultiheadAttention module to support importance sampling.
    Importance masks/scales can be set externally for query, key, value, and output.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        from torch.nn import MultiheadAttention
        self.attn = MultiheadAttention(embed_dim, num_heads)
        # Buffers for importance masks and scales for query, key, value, and out-projection
        # Here we assume importance applies mainly to the out-projection and/or attention matrix
        # The implementation can be more detailed depending on specific layer
        self.register_buffer('importance_mask_out', torch.ones_like(self.attn.in_proj_weight))
        self.register_buffer('importance_scale_out', torch.ones_like(self.attn.in_proj_weight))
        # For simplicity, only handle out projection; can be extended
    def set_importance_mask_scale(self, mask: torch.Tensor, scale: torch.Tensor):
        self.importance_mask_out.copy_(mask)
        self.importance_scale_out.copy_(scale)
    def forward(self, query, key, value, **kwargs):
        # Save original weights if needed
        # Apply importance-scaled weights externally as masks and scales
        # For simplicity, assume the caller manages importance scaling outside
        # For demonstration, just pass through
        return self.attn(query, key, value, **kwargs)

class CustomBERTModel(nn.Module):
    """
    Wraps a HuggingFace BERT model and allows insertion of importance sampling masks/scaling.
    """
    def __init__(self, pretrained_name: str = 'bert-base-uncased', max_seq_length: int = 128):
        super().__init__()
        self.config = BertConfig.from_pretrained(pretrained_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_name)
        self.max_seq_length = max_seq_length
        self._register_hooks()

    def _register_hooks(self):
        """
        Register hooks for layers where importance sampling is applied.
        Typically on attention, intermediate, and output dense layers.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace linear with custom importance scaled linear
                setattr(self.model, name, ImportanceScaledLinear(module.in_features, module.out_features))
            elif 'attention' in name or 'intermediate' in name or 'output' in name:
                # These may contain nn.Linear; handled above
                pass

        # Note: For full control, one could implement custom transformer layers
        # here instead of replacing modules. For brevity, assume wrapping suffices.

    def forward(self, input_ids, attention_mask=None, importance_masks=None, importance_scales=None):
        """
        Forward pass with optional importance sampling masks and scales.
        importance_masks and importance_scales are dicts keyed by layer/module names.
        """
        # For modules replaced with ImportanceScaledLinear, we can set masks/scales externally
        if importance_masks is not None and importance_scales is not None:
            for name, mask in importance_masks.items():
                scale = importance_scales.get(name, None)
                module = dict(self.model.named_modules()).get(name, None)
                if module and hasattr(module, 'set_importance_mask_scale'):
                    module.set_importance_mask_scale(mask, scale)
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

class CustomViTModel(nn.Module):
    """
    Wraps a ViT model with importance sampling capabilities.
    """
    def __init__(self, pretrained_name: str = 'google/vit-base-patch16-224'):
        super().__init__()
        self.model = ViTModel.from_pretrained(pretrained_name)
        self._register_hooks()

    def _register_hooks(self):
        """
        Similar to BERT, replace or hook transformer layers.
        """
        # Depending on implementation, attach importance sampling hooks.
        # For simplicity, we leave as is; in real code, wrap attention and MLP modules.
        pass

    def forward(self, pixel_values, importance_masks=None, importance_scales=None):
        # Similar as above
        # Set importance masks and scales if provided
        return self.model(pixel_values=pixel_values)

# Factory function to create models from config
def get_model(config: Dict[str, Any]) -> nn.Module:
    model_type = config.get('type', 'bert-base-uncased')
    pretrained = config.get('pretrained', True)
    max_seq_length = config.get('max_seq_length', 128)
    if 'bert' in model_type.lower():
        model = CustomBERTModel(pretrained_name=model_type, max_seq_length=max_seq_length)
    elif 'vit' in model_type.lower():
        model = CustomViTModel(pretrained_name=model_type)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model
