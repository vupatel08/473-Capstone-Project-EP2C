## model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Union

class PEFTModule(nn.Module):
    """
    Abstract base class for PEFT modules.
    """
    def __init__(self):
        super().__init__()
        self.trainable_params = []

    def get_parameters(self):
        return self.parameters()

    def perturb_features(self, features: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply multiplicative Gaussian noise to features.
        """
        if sigma <= 0:
            return features
        noise = torch.normal(mean=1.0, std=sigma, size=features.shape, device=features.device)
        return features * noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override in subclasses. Processes input features.
        """
        raise NotImplementedError("PEFTModule subclasses must implement forward method.")

class LoRAModule(PEFTModule):
    """
    Implementation of LoRA: low-rank adaptation matrices decomposing weight updates.
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features

        # LoRA matrices
        self.W_d = nn.Parameter(torch.randn(out_dim, rank))
        self.W_u = nn.Parameter(torch.randn(rank, in_dim))
        # Optional bias
        if hasattr(original_layer, 'bias') and original_layer.bias is not None:
            self.bias = original_layer.bias
        else:
            self.bias = None

        self.trainable_params.extend([self.W_d, self.W_u])

    def forward(self, x: torch.Tensor, perturb: bool = False, sigma: float = 0.0) -> torch.Tensor:
        # Compute the low-rank delta
        delta_W = self.W_d @ self.W_u  # shape (out_dim, in_dim)
        weight = self.original_layer.weight + delta_W
        if perturb and sigma > 0:
            weight = self.perturb_features(weight, sigma)
        # Use the perturbed or original weight
        return nn.functional.linear(x, weight, self.bias)

class AdapterLayer(nn.Module):
    """
    Implementation of a residual adapter module with bottleneck architecture.
    """
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.trainable_params = [self.down_proj, self.up_proj]

    def forward(self, x: torch.Tensor, perturb: bool = False, sigma: float = 0.0) -> torch.Tensor:
        residual = x
        delta = self.down_proj(x)
        delta = self.activation(delta)
        delta = self.up_proj(delta)
        if perturb and sigma > 0:
            delta = self.perturb_features(delta, sigma)
        return residual + delta

class VPTPrompt(nn.Module):
    """
    Implementation of Visual Prompt Tuning (VPT): learnable prompt tokens.
    """
    def __init__(self, prompt_length: int, embedding_dim: int):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, embedding_dim))
        self.trainable_params = [self.prompt_embeddings]

    def forward(self, input_embeddings: torch.Tensor, perturb: bool = False, sigma: float = 0.0) -> torch.Tensor:
        if perturb and sigma > 0:
            noise = self.perturb_features(self.prompt_embeddings, sigma)
            prompt = self.prompt_embeddings + noise
        else:
            prompt = self.prompt_embeddings
        # Concatenate prompts to input embeddings
        return torch.cat([prompt.unsqueeze(0).expand(input_embeddings.size(0), -1, -1), input_embeddings], dim=1)

    def perturb_features(self, features: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return features
        noise = torch.normal(mean=1.0, std=sigma, size=features.shape, device=features.device)
        return features * noise

class TransformerModel(nn.Module):
    def __init__(self, pretrained_model_name: str, config: Dict):
        """
        Initialize the transformer backbone with PEFT modules.
        Args:
            pretrained_model_name: model identifier in transformers.
            config: configuration dictionary with keys:
                - peft_method: 'LoRA', 'Adapter', 'VPT'
                - peft_rank: int
                - adapter_params: float (scaling factor)
                - perturbation_sigma: float
                - adapter_perturbation: bool
                - output_regularization: bool
        """
        super().__init__()
        # Load the backbone
        self.pretrained_model_name = pretrained_model_name
        self.config = config
        self.peft_method = config.get('peft_method', 'LoRA')
        self.peft_rank = config.get('peft_rank', 16)
        self.adapter_params = config.get('adapter_params', 1.0)
        self.perturbation_sigma = config.get('perturbation_sigma', 0.2)
        self.adapter_perturbation = config.get('adapter_perturbation', True)
        self.output_regularization = config.get('output_regularization', True)

        # Depending on architecture, load model
        self.backbone_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, config=self.backbone_config)
        # Check if backbone architecture is vision or language
        # For example, for vision models:
        self.is_vision = hasattr(self.backbone, 'embeddings') or 'vit' in pretrained_model_name.lower()
        self._init_peft_modules()

        # Placeholder for storing features if needed
        self._peft_feature_layers = []

    def _init_peft_modules(self):
        """
        Initialize the PEFT modules depending on the method.
        For vision transformers, insert adapters or LoRA in attention/MLP layers.
        For NLP models, similarly modify linear layers or attention.
        """
        self.peft_modules = nn.ModuleList()

        # For demonstration, assume we modify all linear layers in self.backbone
        # In practice, target specific layers such as query, key, value, MLP
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                if self.peft_method == 'LoRA':
                    # Replace with LoRA module
                    lo_ra = LoRAModule(module, rank=self.peft_rank)
                    setattr(self.backbone, name, lo_ra)
                    self.peft_modules.append(lo_ra)
                elif self.peft_method == 'Adapter':
                    adapter = AdapterLayer(hidden_dim=module.out_features, bottleneck_dim=int(self.adapter_params * module.out_features))
                    setattr(self.backbone, name, adapter)
                    self.peft_modules.append(adapter)
                elif self.peft_method == 'VPT':
                    # For VPT, consider prompt tokens, handled elsewhere
                    pass
        # Additionally, for VPT, initialize prompt tokens if needed
        if 'VPT' in self.peft_method:
            # Example prompt length and embedding dims
            self.prompt_length = int(self.adapter_params * 10)  # arbitrary ratio
            self.embedding_dim = self.backbone.config.hidden_size
            self.vpt_prompt = VPTPrompt(prompt_length=self.prompt_length, embedding_dim=self.embedding_dim)

    def get_peft_module(self) -> nn.Module:
        """
        Return the list of PEFT modules for external access.
        """
        return self.peft_modules

    def perturb_features(self, features: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply multiplicative Gaussian noise to features if perturbation is enabled.
        """
        if not self.training or sigma <= 0:
            return features
        noise = torch.normal(mean=1.0, std=sigma, size=features.shape, device=features.device)
        return features * noise

    def extract_adapter_features(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward specific parts to get features passing through PEFT modules.
        For vision models, extract features after patch embedding or after adapter.
        For NLP, extract hidden states after PEFT modules.
        """
        # Forward input through backbone up to the PEFT parts
        x = input
        # Example: For vision, after patch embedding
        # For NLP, after embedding layer
        # To generalize, hook or override the forward method in subclasses
        # Here, assume backbone returns features as last hidden states
        # Return the features passing through PEFT modules
        # For simplicity, assume it's the output before final classification
        # In practice, hook into specific layers
        features = None
        # Forward with hook: we can implement hooks or modify the backbone
        # For demonstration:
        features = self.backbone(**x).last_hidden_state  # assumes dict input or tensor for vision
        return features

    def forward(self, inputs: Dict, perturb_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass with optional perturbation.
        Args:
            inputs: dict containing required inputs (images, token ids, etc.)
            perturb_params: optional dict, e.g.,
                - 'apply_perturb': True/False
                - 'sigma': float
                - 'perturb_features': True/False
        """
        apply_perturb = False
        sigma = 0.0
        perturb_features_flag = False
        if perturb_params:
            apply_perturb = perturb_params.get('apply_perturb', False)
            sigma = perturb_params.get('sigma', 0.0)
            perturb_features_flag = self.adapter_perturbation and apply_perturb

        # Forward input through the backbone
        if self.is_vision:
            # For vision, inputs could be images
            outputs = self.backbone(**inputs)
            features = outputs.last_hidden_state  # shape: batch x seq x hidden_dim
            # For classification, typically pooled or CLS token
            pooled_output = features[:, 0, :]  # assuming first token as CLS
        else:
            # For NLP, inputs could be tokenized dict
            outputs = self.backbone(**inputs, output_hidden_states=True)
            features = outputs.last_hidden_state
            pooled_output = features[:, 0, :]  # CLS token

        # Perturb features if required
        if perturb_features_flag:
            features = self.perturb_features(features, sigma)

        # Additional optional VPT prompt addition
        if self.peft_method == 'VPT' and hasattr(self, 'vpt_prompt'):
            # Assuming input embeddings are accessible
            # For vision, embedding is often initial patch embedding
            # For NLP, it's token embeddings
            # Here, assume we can get the input embeddings
            # For simplicity, only apply to NLP
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                embedding_layer = self.backbone.get_input_embeddings()
                input_embeddings = embedding_layer(input_ids)
                prompt_embeddings = self.vpt_prompt(input_embeddings, perturb=apply_perturb, sigma=sigma)
                # Re-encode with prompt concatenated
                # Note: For real implementation, need to pass these embeddings directly to the model
                return self.backbone(inputs_embeds=prompt_embeddings, attention_mask=inputs.get('attention_mask', None))
            # For vision, prompts may not be applied in this manner
            # placeholder handling
        # Forward the features through remaining of the model
        # For final classification head:
        # Assume the backbone has classifier or head attribute
        logits = self._classification_head(pooled_output)
        return logits

    def _classification_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Simple linear layer head for classification/regression
        """
        # For example, a linear layer with number of classes
        # For demonstration, replace with actual head if available
        # Unless specified, define a dummy linear classifier
        if not hasattr(self, '_head'):
            # For MNIST/CIFAR, set num_classes
            self.num_classes = 1000  # placeholder, update as needed
            self._head = nn.Linear(features.shape[-1], self.num_classes).to(features.device)
        return self._head(features)

    def get_parameters(self):
        """
        Return trainable parameters (PEFT modules + output head)
        """
        params = []
        for module in self.peft_modules:
            params.extend(list(module.parameters()))
        # Add classification head
        params.extend(list(self._classification_head.parameters()))
        return params

    def save(self, save_path: str):
        """
        Save model and PEFT modules
        """
        torch.save(self.state_dict(), save_path)

    def load(self, load_path: str):
        """
        Load model state
        """
        self.load_state_dict(torch.load(load_path))
