# attention.py
import torch
import torch.nn.functional as F
import math
from typing import Optional

class AttentionLayer:
    """
    Encapsulates self-attention with optional Gaussian blurring on query or key tensors,
    supporting varying sigma values for Smoothed Energy Guidance (SEG).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        apply_blur: bool = False,
        blur_on: str = "query",  # 'query' or 'key'
        default_sigma: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the AttentionLayer.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            apply_blur (bool): Whether to enable Gaussian blurring.
            blur_on (str): Element to blur ('query' or 'key').
            default_sigma (float): Default sigma for Gaussian blur.
            device (torch.device): Computation device.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.apply_blur = apply_blur
        self.blur_on = blur_on
        self.default_sigma = default_sigma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Linear projections for Q, K, V
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        sigma: float = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_enabled: bool = False
    ) -> torch.Tensor:
        """
        Compute self-attention with optional Gaussian blurring on queries or keys.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, tokens, embed_dim].
            sigma (float): Sigma for Gaussian blur; if None, use default.
            attention_mask (torch.Tensor or None): Mask for attention.
            guidance_enabled (bool): If True, perform blurred attention (SEG).

        Returns:
            torch.Tensor: Attention output of shape [batch, tokens, embed_dim].
        """
        batch_size, seq_len, _ = x.shape
        sigma = sigma if sigma is not None else self.default_sigma

        # Project inputs
        Q = self.q_proj(x)  # shape: [batch, seq_len, embed_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head
        def reshape_for_heads(tensor):
            # [batch, seq_len, embed_dim] -> [batch, heads, seq_len, head_dim]
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        Q = reshape_for_heads(Q)
        K = reshape_for_heads(K)
        V = reshape_for_heads(V)

        # If guidance is enabled and sigma > 0, blur the queries
        if self.apply_blur and guidance_enabled and sigma > 1e-8:
            Q = self.apply_gaussian_blur(Q, sigma)

        # Compute scaled dot-product attention
        # Q, K: [batch, heads, seq_len, head_dim]
        # Attention scores: [batch, heads, seq_len, seq_len]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, head_dim]

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return output

    def apply_gaussian_blur(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply 2D Gaussian blur on the spatial tokens of tensor.
        The tensor shape: [batch, heads, seq_len, head_dim]
        We reshape to treat seq_len as 2D spatial if possible.

        Since tokens are 1D sequences, we interpret sequence length as spatial dimension.
        For higher-dimensional tokens, reshape accordingly.

        Args:
            tensor (torch.Tensor): Input tensor to blur.
            sigma (float): Standard deviation of Gaussian kernel.

        Returns:
            torch.Tensor: Blurred tensor with same shape as input.
        """
        # For simplicity, we assume seq_len is a perfect square and treat it as 2D
        batch_size, heads, seq_len, head_dim = tensor.shape

        # Determine spatial size (assumes square for simplicity)
        spatial_size = int(math.sqrt(seq_len))
        if spatial_size * spatial_size != seq_len:
            # fallback: treat as 1D, just pad to next square or use 1D convolution
            # For generality, perform 1D convolution along sequence length
            return self._apply_gaussian_blur_1d(tensor, sigma)

        # Reshape to [batch, heads, H, W, head_dim], combine head_dim later
        tensor_2d = tensor.view(batch_size, heads, spatial_size, spatial_size, head_dim)

        # Generate Gaussian kernel
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        gaussian_kernel = self._create_gaussian_kernel(kernel_size, sigma).to(self.device)

        # Blur along height
        tensor_blur_h = self._apply_conv2d_along_dim(tensor_2d, gaussian_kernel, dim=2)

        # Blur along width
        tensor_blur_w = self._apply_conv2d_along_dim(tensor_blur_h, gaussian_kernel, dim=3)

        # Flatten back
        tensor_blurred = tensor_blur_w.view(batch_size, heads, seq_len, head_dim)
        return tensor_blurred

    def _apply_gaussian_blur_1d(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply 1D Gaussian blur along the sequence dimension for tensors.
        Handles large sequences or when sequence length isn't perfect squares.

        Args:
            tensor (torch.Tensor): [batch, heads, seq_len, head_dim]
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            torch.Tensor: Blurred tensor.
        """
        batch_size, heads, seq_len, head_dim = tensor.shape
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        gaussian_kernel_1d = self._create_gaussian_kernel(kernel_size, sigma).to(self.device)
        # Shape: [kernel_size]
        # Expand for convolution: [1, 1, kernel_size]
        kernel = gaussian_kernel_1d.view(1, 1, -1)

        # Permute for conv1d: [batch*heads*head_dim, seq_len]
        tensor_perm = tensor.permute(0, 1, 3, 2).contiguous()  # [batch, heads, head_dim, seq_len]
        tensor_reshaped = tensor_perm.view(-1, 1, seq_len)  # [batch*heads*head_dim, 1, seq_len]

        # Pad to maintain size
        pad = (kernel_size // 2, kernel_size // 2)
        blurred = F.conv1d(
            tensor_reshaped,
            weight=kernel,
            padding=pad
        )
        # Reshape back
        blurred = blurred.view(batch_size, heads, head_dim, seq_len)
        blurred = blurred.permute(0,1,3,2)  # [batch, heads, seq_len, head_dim]
        return blurred

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Create a 1D Gaussian kernel normalized to sum to 1.

        Args:
            kernel_size (int): Size of the kernel.
            sigma (float): Standard deviation.

        Returns:
            torch.Tensor: 1D Gaussian kernel.
        """
        # Generate Gaussian
        center = kernel_size // 2
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - center
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def _apply_conv2d_along_dim(
        self,
        tensor: torch.Tensor,
        kernel: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
        """
        Apply 2D convolution along a specified spatial dimension in tensor.

        Args:
            tensor (torch.Tensor): [batch, heads, H, W, head_dim]
            kernel (torch.Tensor): 2D Gaussian kernel
            dim (int): Dimension along which to convolve (2 for H, 3 for W)

        Returns:
            torch.Tensor: Blurred tensor.
        """
        # Permute tensor to [batch, heads, H or W, other spatial, head_dim]
        permute_dims = [0, 1, 2, 3, 4]
        if dim == 2:
            # Convolve along height
            tensor_perm = tensor.permute(0, 1, 2, 3, 4)
        elif dim == 3:
            # Convolve along width
            tensor_perm = tensor.permute(0, 1, 3, 2, 4)
        else:
            raise ValueError("dim must be 2 or 3")

        # Merge batch, heads, spatial, other dims for convolution
        shape = tensor_perm.shape
        # [batch, heads, spatial_dim, other_dim, head_dim]
        tensor_flat = tensor_perm.contiguous().view(-1, 1, shape[dim], shape[dim+1] if dim==2 else shape[dim-1])
        # Prepare kernel for conv2d: assuming kernel is square
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

        # Pad to preserve size
        pad_size = kernel.shape[-1] // 2
        tensor_blurred = F.conv2d(
            tensor_flat,
            weight=kernel,
            padding=pad_size,
            groups=1
        )
        # Reshape back
        tensor_blurred = tensor_blurred.view(shape)
        # Permute back to original shape
        if dim == 2:
            tensor_blurred = tensor_blurred.permute(0, 1, 2, 3, 4)
        else:
            tensor_blurred = tensor_blurred.permute(0, 1, 3, 2, 4)

        return tensor_blurred

