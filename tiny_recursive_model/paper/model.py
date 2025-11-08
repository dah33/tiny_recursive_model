"""
Simplified TRM core model for Sudoku-Extreme (no ACT wrapper).
Configuration based on README.md Sudoku-Extreme settings:
- mlp_t=True (MLP on L instead of transformer)
- pos_encodings=none
- n_layers=2, H_cycles=3, L_cycles=6
- hidden_size=512, expansion=4
- scratch_slots=16

Use with PyTorch AMP for mixed precision training.
"""

import math

import torch
from einops import repeat
from torch import nn

from tiny_recursive_model.paper.initialisers import trunc_normal_init_
from tiny_recursive_model.paper.mixer import MixerLayer


def latent_recursion(net, x, y, z, n=6):  # all are (B, L, D)
    # the network learns to refine the latents if input is passed in, otherwise it refines the output
    for _ in range(n):  # latent reasoning
        z = net(y + z + x)
    y = net(y + z)  # refine output answer
    return y, z


def deep_recursion(net, output_head, Q_head, x, y, z, n=6, T=3):
    # recursing T−1 times to improve y and z (no gradients needed)
    with torch.no_grad():
        for _ in range(T - 1):
            y, z = latent_recursion(net, x, y, z, n)
    # recursing T−1 times to improve y and z (no gradients needed)
    y, z = latent_recursion(net, x, y, z, n)
    return (y.detach(), z.detach()), output_head(y), Q_head(y)


class HaltHead(nn.Module):
    """Predicts halt from the first position (a scratch slot)."""

    def __init__(self, in_features, scratch_slot=0, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)
        self.scratch_slot = scratch_slot

        # Initialize to encourage thinking (not halting early)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -5.0)
        # PyTorch default would be: Kaiming uniform for weights, uniform for bias
        #   Starting at -5 biases model to NOT halt initially (encourages more thinking)

    def forward(self, x):
        return self.linear(x[:, self.scratch_slot, :]).squeeze(-1)


class PredHead(nn.Module):
    """Prediction head that removes scratch slots from output."""

    def __init__(self, in_features, vocab_size, scratch_slots, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, vocab_size, bias=bias)
        self.scratch_slots = scratch_slots

        # Initialize with truncated normal
        trunc_normal_init_(self.linear.weight, std=1.0 / math.sqrt(in_features))
        # PyTorch default would be: Kaiming uniform U(-sqrt(k), sqrt(k)) where k=1/in_features
        #   Kaiming is optimized for ReLU; truncated normal is more conservative

    def forward(self, x):
        logits = self.linear(x)
        return logits[:, self.scratch_slots :]


class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 11,
        D: int = 512,
        L: int = 81,
        scratch_slots: int = 16,
        n_layers: int = 2,
        expansion_factor: float = 4.0,
        rms_norm_eps: float = 1e-5,
        n: int = 6,
        T: int = 3,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.D = D
        self.L = L
        self.scratch_slots = scratch_slots
        self.n_layers = n_layers
        self.expansion_factor = expansion_factor
        self.rms_norm_eps = rms_norm_eps
        self.n = n
        self.T = T

        # Embeddings
        self.input_embedding = nn.Embedding(vocab_size, D)

        # Initialisation from original TRM implementation
        # - Apply embedding scaling factor during forward pass
        # - This scales embeddings by sqrt(D) to balance gradient magnitudes and
        #   maintain variance ~1 consistent with other layers (Xavier/He init).
        # - TODO: May be redundant with Adam optimizer + RMSNorm
        self.embed_scale = math.sqrt(D)
        trunc_normal_init_(self.input_embedding.weight, std=1.0 / self.embed_scale)
        # PyTorch default: nn.Embedding uses N(0, 1) - much larger std than 1/sqrt(D)≈0.044

        # Learnable scratch pad values
        # - First scratch slot is used by halt_head for early stopping decision
        # - Initialized to zeros; could also use small random values
        self.scratch_init = nn.Parameter(torch.zeros(scratch_slots, D))
        # Typical: Learnable embeddings like this are usually initialised with
        #   small random values e.g., N(0, 0.02) like BERT positional
        #   embeddings

        # Main model
        layers = [
            MixerLayer(
                seq_len=L + scratch_slots,
                hidden_size=D,
                expansion_factor=expansion_factor,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(n_layers)
        ]
        self.model = nn.Sequential(*layers)
        
        # Optionally wrap with gradient checkpointing
        if use_checkpointing:
            from torch.utils.checkpoint import checkpoint_sequential
            self.model = checkpoint_sequential(
                self.model,
                segments=n_layers,
                use_reentrant=False,
            )

        # Output heads
        self.pred_head = PredHead(D, vocab_size, scratch_slots, bias=False)
        self.halt_head = HaltHead(D, scratch_slot=0, bias=True)

    def y_init(self, x_input: torch.Tensor) -> torch.Tensor:
        """Initialize y hidden state to zeros with same shape as embedded input."""
        x = self.embed_input(x_input)
        return torch.zeros_like(x)

    def z_init(self, x_input: torch.Tensor) -> torch.Tensor:
        """Initialize z hidden state to zeros with same shape as embedded input."""
        x = self.embed_input(x_input)
        return torch.zeros_like(x)

    def embed_input(self, x_input: torch.Tensor) -> torch.Tensor:
        # Apply embedding scaling to match original TRM implementation
        token_embeddings = self.embed_scale * self.input_embedding(x_input)
        scratch_inits = repeat(self.scratch_init, "L D -> B L D", B=x_input.shape[0])
        return torch.cat([scratch_inits, token_embeddings], dim=1)

    def forward(
        self,
        x_input: torch.Tensor,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ):
        # caller should do the N_supervision looping
        x = self.embed_input(x_input)
        y = y if y is not None else self.y_init(x_input)
        z = z if z is not None else self.z_init(x_input)
        return deep_recursion(
            self.model,
            self.pred_head,
            self.halt_head,
            x,
            y,
            z,
            n=self.n,
            T=self.T,
        )
