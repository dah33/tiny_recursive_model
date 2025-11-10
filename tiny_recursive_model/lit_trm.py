from dataclasses import dataclass

import lightning as L
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from tiny_recursive_model.paper.model import TinyRecursiveModel


def softmax_cross_entropy(token_logits: Tensor, target_tokens: Tensor) -> Tensor:
    flat_logits = rearrange(token_logits, "B L D -> (B L) D")
    flat_target = rearrange(target_tokens, "B L -> (B L)")
    return F.cross_entropy(flat_logits, flat_target)  # mean over flattened B x L


def binary_cross_entropy(
    halt_logits: Tensor, token_logits: Tensor, target_tokens: Tensor
) -> Tensor:
    # 1 if the entire predicted sequence matches the target sequence
    is_seq_correct = (token_logits.argmax(dim=-1) == target_tokens).all(dim=-1)
    target = is_seq_correct.to(dtype=halt_logits.dtype)
    return F.binary_cross_entropy_with_logits(halt_logits, target)  # mean over batch


@dataclass
class TrainingCarry:
    x_input: Tensor
    y: Tensor
    z: Tensor
    y_true: Tensor
    supervision_count: Tensor
    completed: Tensor  # Boolean mask


class LitTRM(L.LightningModule):
    def __init__(
        self,
        *,
        # Model
        vocab_size: int = 11,
        D: int = 512,
        L: int = 81,
        scratch_slots: int = 16,
        n_layers: int = 2,
        expansion_factor: float = 4.0,
        rms_norm_eps: float = 1e-5,
        # Recursion
        T: int = 3,
        n: int = 6,
        halt_prob_threshold: float = 0.5,
        N_supervision: int = 16,
        activation_checkpointing: bool = False,
    ):
        super().__init__()

        # Save for checkpointing
        self.save_hyperparameters()

        # Hyperparameters used in training and forward
        self.halt_prob_threshold = halt_prob_threshold
        self.N_supervision = N_supervision
        self.activation_checkpointing = activation_checkpointing

        self.model = TinyRecursiveModel(
            vocab_size=vocab_size,
            D=D,
            L=L,
            scratch_slots=scratch_slots,
            n_layers=n_layers,
            expansion_factor=expansion_factor,
            rms_norm_eps=rms_norm_eps,
            n=n,
            T=T,
            activation_checkpointing=activation_checkpointing,
        )

    def on_train_epoch_start(self):
        # Initialize one carry per gradient accumulation step
        self.carries = [None] * self.trainer.accumulate_grad_batches

    def training_step(self, batch, batch_idx) -> Tensor:
        x_input_fresh, y_true_fresh, puzzle_ids = batch.values()

        # Determine which carry to use
        carry_idx = batch_idx % self.trainer.accumulate_grad_batches

        # Initialize carry on first use
        if self.carries[carry_idx] is None:
            self.carries[carry_idx] = TrainingCarry(
                x_input=x_input_fresh,
                y_true=y_true_fresh,
                y=self.model.y_init(x_input_fresh),
                z=self.model.z_init(x_input_fresh),
                supervision_count=torch.zeros_like(puzzle_ids, dtype=torch.long),
                completed=torch.zeros_like(puzzle_ids, dtype=torch.bool),
            )

        # Replace completed slots with fresh samples
        carry = self.carries[carry_idx]
        replace = carry.completed
        carry.x_input[replace] = x_input_fresh[replace]
        carry.y_true[replace] = y_true_fresh[replace]
        carry.y[replace] = self.model.y_init(x_input_fresh[replace])
        carry.z[replace] = self.model.z_init(x_input_fresh[replace])
        carry.supervision_count[replace] = 0

        # Single supervision step
        (y, z), y_hat, q_hat = self.model(carry.x_input, carry.y, carry.z)
        pred_loss = softmax_cross_entropy(y_hat, carry.y_true)
        halt_loss = binary_cross_entropy(q_hat, y_hat, carry.y_true)
        loss = pred_loss + halt_loss

        halt = q_hat.detach() >= self.halt_prob_threshold

        # Update carry state
        carry.supervision_count += 1
        carry.y = y
        carry.z = z
        completed = halt | (carry.supervision_count >= self.N_supervision)
        carry.completed = completed

        # Logging
        self.log("loss", loss, prog_bar=True)
        self.log("halt_loss", halt_loss, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        if completed.any():
            avg_sup = carry.supervision_count[completed].float().mean()
            self.log("avg_sup", avg_sup, prog_bar=True)

        return loss

    def forward(self, batch):
        x_input, y_true, puzzle_ids = batch.values()
        y, z = None, None
        halt = torch.zeros_like(puzzle_ids, dtype=torch.bool)
        final_loss, final_halt_loss = 0.0, 0.0
        batch_size = x_input.shape[0]
        sample_count = 0

        # No carry between supervision steps required
        for supervision_step in range(self.N_supervision):
            (y, z), y_hat, q_hat = self.model(x_input, y, z)

            pred_loss = softmax_cross_entropy(y_hat, y_true)
            halt_loss = binary_cross_entropy(q_hat, y_hat, y_true)
            loss = pred_loss + halt_loss
            sample_count += x_input.shape[0]

            halt = q_hat >= self.halt_prob_threshold

            # Accumulate losses if at final supervision step
            is_final_step = halt | (supervision_step == self.N_supervision - 1)
            final_loss += loss.item() * is_final_step.sum() / batch_size
            final_halt_loss += halt_loss.item() * is_final_step.sum() / batch_size

            if halt.all():
                break

            active = ~halt
            x_input = x_input[active]
            y = y[active]
            z = z[active]
            y_true = y_true[active]

        avg_sup = sample_count / batch_size * self.N_supervision
        return y_hat, (final_loss, final_halt_loss, avg_sup)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, _ = self.forward(batch)
        return y_hat.argmax(dim=-1)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        y_hat, (loss, halt_loss, avg_sup) = self.forward(batch)
        _, y_true, _ = batch.values()
        cell_acc = (y_hat.argmax(dim=-1) == y_true).float().mean()
        acc = (y_hat.argmax(dim=-1) == y_true).all(dim=-1).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_halt_loss", halt_loss)
        self.log("val_avg_sup", avg_sup)
        self.log("cell_acc", cell_acc, prog_bar=True)
        self.log("acc", acc, prog_bar=True)
