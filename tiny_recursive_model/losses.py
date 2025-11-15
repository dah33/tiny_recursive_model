import torch
from torch import Tensor


def log_stablemax(logits: Tensor, dim: int = -1, clamp_min: float = -10.0) -> Tensor:
    """
    Compute log-probabilities using stablemax normalization.
    
    StableMax applies s(x) = (x + 1) if x >= 0, else 1 / (1 - x), then normalizes.
    Logits are clamped to avoid extreme negative values that cause instability.

    Based on: https://github.com/QuixiAI/stablemax-orthogonal
    
    Args:
        logits: Input logits
        dim: Dimension to normalize over
        clamp_min: Minimum value to clamp logits to (default: -10.0)
    
    Returns:
        Log-probabilities
    """
    # Clamp extreme negative logits
    logits = torch.clamp(logits, min=clamp_min)
    s_logits = torch.where(logits >= 0, logits + 1, 1 / (1 - logits))
    s_sum = s_logits.sum(dim=dim, keepdim=True)
    return torch.log(s_logits / (s_sum + 1e-9) + 1e-9)


def stablemax_cross_entropy(
    logits: Tensor, target: Tensor, ignore_index: int = -100
) -> Tensor:
    """
    Stablemax cross entropy loss.

    Args:
        logits: Unnormalized logits of shape (N, C) where N is batch size and C is number of classes
        target: Target class indices of shape (N,)
        ignore_index: Index to ignore in loss calculation

    Returns:
        Scalar loss value (mean over valid tokens)
    """
    # Cast to float32 for numerical stability (PyTorch/Lightning convention)
    logprobs = log_stablemax(logits.float(), dim=-1)

    valid_mask = target != ignore_index
    transformed_labels = torch.where(valid_mask, target, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    per_token_loss = -torch.where(valid_mask, prediction_logprobs, 0)
    return per_token_loss.mean()
