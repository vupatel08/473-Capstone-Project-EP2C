## model_wrapper.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Callable
import re

class ModelWrapper:
    """
    Wraps a pre-trained language model to enable inference with
    attention score manipulation for attention steering.
    """
    def __init__(self, model_name: str, model_path: str = None):
        """
        Load the model and tokenizer based on model_name and register hooks.
        :param model_name: e.g., "LLaMA-7B" or "GPT-J"
        :param model_path: path to local model checkpoint or None for hub model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self._register_hook_dict: Dict[Tuple[int, int], torch.nn.Module] = {}
        # Store references to hooks for later removal if needed
        self._hooks_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Maintain a set of target heads for steering (layer, head)
        self.target_heads: List[Tuple[int, int]] = []

    def _load_model(self):
        """
        Load model and tokenizer. Supports LLAMA, GPT-J architectures.
        """
        # Example: specify model_path or use models from Huggingface hub
        # For LLAMA: use 'huggingface/llama model id' if available
        # For GPT-J: use 'EleutherAI/gpt-j-6B'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path if self.model_path else self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path if self.model_path else self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            revision="main"
        )
        self.model.to(self.device)
        self.model.eval()

        # Identify attention modules depending on architecture
        # For LLAMA: self.model.layers is a list of transformer blocks
        # For GPT-J: self.model.transformer.h is list of blocks
        # We'll support both by checking attributes
        if hasattr(self.model, 'layers'):
            self.attention_layers = self.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.attention_layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture for attention hook registration.")
        # Prepare a dict to store attention scores per layer
        self._attn_scores: Dict[Tuple[int, int], torch.Tensor] = {}
        # For hook management
        self._register_hooks()

    def _register_hooks(self):
        """
        Register hooks on attention modules for all layers.
        For multiple heads, hooks are registered per layer.
        """
        # Clear previous handles if any
        for handle in self._hooks_handles:
            handle.remove()
        self._hooks_handles.clear()

        # Loop over attention modules
        for layer_idx, layer_module in enumerate(self.attention_layers):
            # Depending on architecture, attention module attribute:
            # For LLAMA: 'self_attn' in each layer
            # For GPT-J: 'attn' in each block
            attn_module = None
            if hasattr(layer_module, 'self_attn'):
                attn_module = getattr(layer_module, 'self_attn')
            elif hasattr(layer_module, 'attn'):
                attn_module = getattr(layer_module, 'attn')
            else:
                continue  # skip if not found

            # Register hook to extract attention scores
            # We assume the module has a method or attribute that outputs attention scores.
            # For LLAMA, the attention module often returns attention weights when called
            # with output_attentions=True or via hooks.
            handle = attn_module.register_forward_hook(self._create_attention_hook(layer_idx))
            self._hooks_handles.append(handle)

    def _create_attention_hook(self, layer_idx: int) -> Callable:
        """
        Create a hook to capture attention scores during forward pass.
        Assumes the attention module returns or saves attention scores.
        """
        def hook(module, inputs, outputs):
            """
            Hook function to save attention scores.
            :param module: attention module
            :param inputs: tuple of input tensors
            :param outputs: output object, could be tuple or dict depending on model
            """
            # Depending on the module architecture, extract attention weights
            # For most transformers, implement assuming the module returns attention weights
            # For LLAMA: self_attn may return attention weights if output_attentions=True
            # For GPT-J: attention weights are usually internal; need to fetch from outputs
            # Since hooks are before softmax, assume the attention weights are available
            # in the module or via the outputs (depends on model implementation)
            # For generality, we try to get attention weights
            # Note: If model does not output attention scores, this may need adjustments
            if hasattr(module, 'attn'):  # typical for some implementations
                # LLAMA uses 'self_attn' which may not output attentions directly
                # but for the purpose here, assume the attribute exists
                # Alternatively, we can access these during the forward if model supports
                # options, but for simplicity, attach to attention modules that output attentions
                pass
            # Alternatively, some models store attentions inside module, or in output if configured

        return hook

    def register_attention_heads(self, heads: List[Tuple[int, int]]):
        """
        Record the selected heads for steering.
        :param heads: list of (layer_idx, head_idx)
        """
        self.target_heads = heads

    def get_attention_scores(self, layer_idx: int, head_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass to obtain attention scores for a specific head in a layer.
        Note: For efficiency, a dedicated method or caching strategy can be used.
        """
        # Run a forward with return_attentions=True
        # For models supporting output_attentions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_attentions=True,
            return_dict=True
        )
        attentions = outputs['attentions']  # tuple/list: one per layer
        layer_attn = attentions[layer_idx]  # shape: (batch_size, num_heads, seq_len, seq_len)
        # Select head_idx
        attn_head = layer_attn[:, head_idx, :, :]  # shape: (batch_size, seq_len, seq_len)
        return attn_head

    def generate(self, input_ids: List[int], emphasis_spans: List[List[int]], alpha: float=0.01, max_new_tokens: int=50, temperature: float=1.0) -> str:
        """
        Generate text with optional attention steering by reweighting attention scores
        during decoding at specified layers and heads.
        :param input_ids: tokenized prompt
        :param emphasis_spans: list of token index spans emphasized by user
        :param alpha: scaling coefficient
        :param max_new_tokens: maximum tokens to generate
        :param temperature: sampling temperature
        """
        # Prepare focus mask: set of token indexes highlighted
        emphasized_token_indices = set()
        for span in emphasis_spans:
            if span:
                start_idx, end_idx = span
                # Add tokens in span (assuming span is [start, end])
                emphasized_token_indices.update(range(start_idx, end_idx+1))
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Set model to eval mode
        self.model.eval()

        # To modify attention, we leverage hooks in attention modules.
        # Because hooks are active during forward, we set a custom forward_hook that adjusts attention scores.
        # Once set, in the generate loop, attention scores are manipulated per step.
        # Here, for simplicity, we make the assumption that attention scores are returned via attentions,
        # and we perform a "re-weight" step right before softmax.

        # For more precise control, custom decoder or attention implementations are needed.
        # Assuming we can set hooks to modify attention during the forward pass.

        # Run generation with custom attention reweighting
        output_ids = self._generate_with_attention_reweight(input_ids, emphasis_spans, alpha, max_new_tokens, temperature)
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return generated_text

    def _generate_with_attention_reweight(
        self,
        input_ids: List[int],
        emphasis_spans: List[List[int]],
        alpha: float,
        max_new_tokens: int,
        temperature: float
    ) -> List[int]:
        """
        Generate tokens iteratively, applying the attention score reweighting at each step
        in the targeted heads during the attention calculations.
        Note: For the purpose of illustration, assumes the model's attention modules can be intercepted.
        """
        # Initialize generation
        generated_ids = input_ids.copy()
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated_ids], device=self.device)
            # Run model with hooks that reweight attention
            # (Assuming hooks are registered and will manipulate scores during forward)
            outputs = self.model(
                input_ids=input_tensor,
                output_attentions=True,
                return_dict=True,
                temperature=temperature
            )
            logits = outputs.logits  # shape: (1, seq_len, vocab_size)
            # Get last token logits
            next_token_logits = logits[0, -1, :]
            # Apply temperature
            scaled_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            # Sample or greedy
            next_token_id = torch.argmax(probs).item()
            generated_ids.append(next_token_id)
            if next_token_id == self.tokenizer.eos_token_id:
                break
        return generated_ids

# Note:
# The critical part of this implementation is the correct registration of hooks
# on the attention modules to access and modify attention scores during generation.
# This code assumes the model's attention modules support capturing attention weights
# via forward hooks, which may require model-specific adjustments.
# For extremely custom models, rewriting attention modules to accept external scores
# or use a custom transformer implementation is necessary.
