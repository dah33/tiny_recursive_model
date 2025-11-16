import torch
from torch import Tensor


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: Tensor, target: Tensor, ignore_index: int = -100
) -> Tensor:
    """
    Stablemax cross entropy loss.

    Alternative implementation: https://github.com/QuixiAI/stablemax-orthogonal/blob/main/src/stablemax.py

    Args:
        logits: Unnormalized logits of shape (N, C) where N is batch size and C is number of classes
        target: Target class indices of shape (N,)
        ignore_index: Index to ignore in loss calculation

    Returns:
        Scalar loss value (mean over valid tokens)
    """
    logits = logits.float()  # Ensure float32 for stability
    logprobs = log_stablemax(logits, dim=-1)

    valid_mask = target != ignore_index
    transformed_labels = torch.where(valid_mask, target, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    per_token_loss = -torch.where(valid_mask, prediction_logprobs, 0)
    return per_token_loss.mean()
