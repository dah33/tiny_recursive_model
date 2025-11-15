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
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tiny_recursive_model.paper.initialisers import trunc_normal_init_
from tiny_recursive_model.paper.mixer import MixerLayer


def latent_recursion(net, x, y, z, n=6):  # all are (B, L, D)
    # the network learns to refine the latents if input is passed in, otherwise it refines the output
    for _ in range(n):  # latent reasoning
        z = net(y + z + x)
    y = net(y + z)  # refine output answer
    return y, z


def deep_recursion(net, pred_head, halt_head, x, y, z, n=6, T=3):
    # recursing T−1 times to improve y and z (no gradients needed)
    with torch.no_grad():
        for _ in range(T - 1):
            y, z = latent_recursion(net, x, y, z, n)
    # recursing once to improve y and z
    y, z = latent_recursion(net, x, y, z, n)
    return (y.detach(), z.detach()), pred_head(y), halt_head(y)


class HaltHead(nn.Module):
    """Predicts halt from the first position (a scratchpad slot)."""

    def __init__(self, in_features, scratchpad_idx=0, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)
        self.scratchpad_idx = scratchpad_idx

        # Initialize to encourage thinking (not halting early)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -5.0)
        # PyTorch default would be: Kaiming uniform for weights, uniform for bias
        #   Starting at -5 biases model to NOT halt initially (encourages more thinking)

    def forward(self, x):
        return self.linear(x[:, self.scratchpad_idx, :]).squeeze(-1)


class PredHead(nn.Module):
    """Prediction head that removes scratchpad from output."""

    def __init__(self, in_features, vocab_size, n_scratchpad, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, vocab_size, bias=bias)
        self.n_scratchpad = n_scratchpad

        # Initialize with truncated normal
        trunc_normal_init_(self.linear.weight, std=1.0 / math.sqrt(in_features))
        # PyTorch default would be: Kaiming uniform U(-sqrt(k), sqrt(k)) where k=1/in_features
        #   Kaiming is optimized for ReLU; truncated normal is more conservative

    def forward(self, x):
        logits = self.linear(x)
        return logits[:, self.n_scratchpad :]


class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 11,
        D: int = 512,
        L: int = 81,
        n_scratchpad: int = 16,
        n_layers: int = 2,
        expansion_factor: float = 4.0,
        rms_norm_eps: float = 1e-5,
        n: int = 6,
        T: int = 3,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.D = D
        self.L = L
        self.n_scratchpad = n_scratchpad
        self.n_layers = n_layers
        self.expansion_factor = expansion_factor
        self.rms_norm_eps = rms_norm_eps
        self.n = n
        self.T = T

        # Precalculate total sequence length for y_init and z_init
        self.total_seq_len = n_scratchpad + L

        # Embeddings
        self.input_embedding = nn.Embedding(vocab_size, D)

        # Initialisation from original TRM implementation
        # - Apply embedding scaling factor during forward pass
        # - This scales embeddings by sqrt(D) to balance gradient magnitudes and
        #   maintain variance ~1 consistent with other layers (Xavier/He init).
        # - TODO: May be redundant with Adam optimizer + RMSNorm
        self.embedding_scale = math.sqrt(D)
        trunc_normal_init_(self.input_embedding.weight, std=1.0 / self.embedding_scale)
        # PyTorch default: nn.Embedding uses N(0, 1) - much larger std than 1/sqrt(D)≈0.044

        # Learnable scratchpad values
        # - First scratchpad slot is used by halt_head for early stopping decision
        # - Initialized to zeros; could also use small random values
        self.scratchpad = nn.Parameter(torch.zeros(n_scratchpad, D))
        # Typical: Learnable embeddings like this are usually initialised with
        #   small random values e.g., N(0, 0.02) like BERT positional
        #   embeddings

        # Main model
        layers = [
            MixerLayer(
                seq_len=self.total_seq_len,
                hidden_size=D,
                expansion_factor=expansion_factor,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(n_layers)
        ]
        self.model = nn.Sequential(*layers)
        self.activation_checkpointing = activation_checkpointing

        # Output heads
        self.pred_head = PredHead(D, vocab_size, n_scratchpad, bias=False)
        self.halt_head = HaltHead(D, scratchpad_idx=0, bias=True)

    def y_init(self, batch_size: int) -> torch.Tensor:
        """Initialize y hidden state to zeros with shape (B, L, D)."""
        return torch.zeros(
            batch_size, self.total_seq_len, self.D, device=self.scratchpad.device
        )

    def z_init(self, batch_size: int) -> torch.Tensor:
        return self.y_init(batch_size)  # same shape as y

    def embed_input(self, x_input: torch.Tensor) -> torch.Tensor:
        # Apply embedding scaling to match original TRM implementation
        token_embeddings = self.embedding_scale * self.input_embedding(x_input)
        scratchpad = self.scratchpad.expand(x_input.size(0), -1, -1)
        return torch.cat([scratchpad, token_embeddings], dim=1)

    def forward(
        self,
        x_input: torch.Tensor,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ):
        # Caller should do the N_supervision looping
        x = self.embed_input(x_input)
        batch_size = x_input.size(0)
        y = y if y is not None else self.y_init(batch_size)
        z = z if z is not None else self.z_init(batch_size)

        # Use activation checkpointing to reduce memory usage during training
        net = self.model
        if self.activation_checkpointing and self.training:
            net = partial(checkpoint, net, use_reentrant=False)

        return deep_recursion(
            net,
            self.pred_head,
            self.halt_head,
            x,
            y,
            z,
            n=self.n,
            T=self.T,
        )
