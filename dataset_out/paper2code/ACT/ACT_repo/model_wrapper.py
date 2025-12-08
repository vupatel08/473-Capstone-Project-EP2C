## model_wrapper.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional


class ModelWrapper:
    """
    Wraps a HuggingFace transformer language model to:
    - Load model and tokenizer
    - Register hooks to extract attention weights during inference
    - Provide method to retrieve attention maps
    - Generate outputs with optional modified attention
    """

    def __init__(self, model_name: str = "Llama-7b", device: str = "cuda", model_path: Optional[str] = None):
        """
        Initialize the model wrapper.
        Args:
            model_name (str): Identifier of the pretrained model.
            device (str): Device to run inference ("cpu" or "cuda").
            model_path (str, optional): Path to local model or HF model id.
        """
        import transformers  # import locally to delay loading if needed
        # Load model and tokenizer
        if model_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Prepare storage for attention maps
        # Structure: dict[layer_idx] -> dict['attentions': List[Tensor]]
        self.attention_maps: Dict[int, Dict[str, List[torch.Tensor]]] = {}
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Register hooks on transformer layers
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """
        Register hooks on all attention layers to capture attention weights during forward pass.
        Assumes model has attribute `model.transformer.h`, typical for HuggingFace models like GPT.
        """
        # Clear previous hooks if any
        self._remove_attention_hooks()

        # Find all attention modules
        # The structure depends on model origin; common attribute: model.transformer.h (list of layers)
        for layer_idx, layer in enumerate(self.model.transformer.h):
            # Some models use 'attn' attribute for attention modules
            # For GPT-like models, layer.attn is typical
            if hasattr(layer, 'attn'):
                attn_module = layer.attn

                def get_attention_hook(layer_index: int):
                    # Closure to retain layer index
                    def hook(module, input, output):
                        # output: tuple (attn_probs, attn_mask, attn_output, new_attn_weights)
                        # For most models, the first element is attention probabilities
                        # Capture attention probs: shape (batch, num_heads, seq_len, seq_len)
                        if isinstance(output, tuple):
                            attn_probs = output[0]
                        else:
                            attn_probs = output
                        # Store a clone to avoid mutation
                        self.attention_maps[layer_index] = {
                            'attentions': [attn_probs.detach().cpu()]
                        }
                    return hook

                handle = attn_module.register_forward_hook(get_attention_hook(layer_idx))
                self.hook_handles.append(handle)

    def _remove_attention_hooks(self):
        """
        Remove all registered hooks to avoid duplication.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def register_attention_hooks(self):
        """
        Public method to register hooks during inference.
        """
        self._register_attention_hooks()

    def clear_attention_maps(self):
        """
        Clear stored attention maps.
        """
        self.attention_maps.clear()

    def compute_attention(self) -> Dict[int, Dict[str, List[torch.Tensor]]]:
        """
        Return the latest stored attention maps.
        """
        return self.attention_maps

    def generate_output(
        self,
        input_ids: torch.Tensor,
        attention_maps: Optional[Dict[int, Dict[str, List[torch.Tensor]]]] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate output text with optional attention modifications.
        Args:
            input_ids (torch.Tensor): Tokenized input IDs, shape (1, seq_len).
            attention_maps: Optional attention maps to modify during generation.
            max_new_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature.
            do_sample (bool): Whether to sample or greedy decoding.
        Returns:
            Generated text string.
        """
        # If attention_maps provided, replace stored maps before generation
        if attention_maps is not None:
            self._apply_attention_modifications(attention_maps)

        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    def _apply_attention_modifications(self, attention_maps: Dict[int, Dict[str, List[torch.Tensor]]]):
        """
        Inject modified attention maps into the model by overriding attention in transformer layers.
        Note: HuggingFace transformers do not support direct input of attention weights.
        As a workaround, we can patch the attention probabilities after the hook, but this
        requires model internals modification or a custom forward hook.
        
        Since the task demands, here we will assume that the model's transformer layers have
        a method or attribute to override attentions or that we replace the internal attention matrices.
        For most models, this is non-trivial; for demonstration, we provide a patch approach.
        """
        # WARNING: Overriding internal attention probabilities requires model modification,
        # which is complex. For this implementation, we assume attention modifications are
        # feasible via hooks or a suitable interface.
        # The following is a placeholder: in practice, need model-specific hooks or code.

        # Example (hypothetical): Patch attention logits/weights if model allows
        pass

    def unload(self):
        """
        Cleanup: unregister hooks and free resources.
        """
        self._remove_attention_hooks()
        # Additional cleanup if needed
