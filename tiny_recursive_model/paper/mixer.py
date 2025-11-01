import math

import torch
import torch.nn.functional as F
from torch import nn

from tiny_recursive_model.paper.initialisers import trunc_normal_init_


def round_up_to_multiple(value: int, multiple: int) -> int:
    """Round up value to the nearest multiple."""
    return math.ceil(value / multiple) * multiple


class SwiGLU(nn.Module):
    """SwiGLU activation (Swish-Gated Linear Unit)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # Fused implementation: one matmul for gate+up instead of two (more efficient)
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Initialize weights to match original TRM implementation (truncated normal)
        trunc_normal_init_(self.gate_up_proj.weight, std=1.0 / math.sqrt(hidden_size))
        trunc_normal_init_(
            self.down_proj.weight, std=1.0 / math.sqrt(intermediate_size)
        )
        # PyTorch default: nn.Linear uses Kaiming uniform U(-sqrt(k), sqrt(k)) where k=1/in_features
        #   Kaiming has higher variance; LeCun normal (used here) is more conservative

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MixerLayer(nn.Module):
    """MLP-Mixer inspired layer: along sequence, then hidden dimension."""

    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        expansion_factor: float,
        rms_norm_eps: float,
    ) -> None:
        # TODO: other implementations:
        # - norm first, which might be more stable
        # - use LayerNorm (with mean/std rather than RMS)
        # - use GELU instead of SwiGLU
        # - use conv1d for token mixing
        # - use nn.Sequential approach
        super().__init__()

        # Compute intermediate sizes: 2/3 factor keeps parameter count equal to vanilla FFN
        seq_intermediate = round(expansion_factor * seq_len * 2 / 3)
        hidden_intermediate = round(expansion_factor * hidden_size * 2 / 3)
        # Note: original TRM implementation rounded up to a multiple of 256 for efficiency

        self.mlp_t = SwiGLU(hidden_size=seq_len, intermediate_size=seq_intermediate)
        self.norm_t = nn.RMSNorm(seq_len, eps=rms_norm_eps)
        self.mlp = SwiGLU(
            hidden_size=hidden_size, intermediate_size=hidden_intermediate
        )
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply MLP across sequence dimension
        x = x.transpose(1, 2)
        x = self.norm_t(x + self.mlp_t(x))
        x = x.transpose(1, 2)

        # Apply MLP across hidden dimension with residual connection
        return self.norm(x + self.mlp(x))
