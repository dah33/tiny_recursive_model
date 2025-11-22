from dataclasses import dataclass

import lightning as L
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from tiny_recursive_model import losses
from tiny_recursive_model.paper.model import TinyRecursiveModel


def softmax_cross_entropy(
    token_logits: Tensor, target_tokens: Tensor, ignore_index: int = -100
) -> Tensor:
    flat_logits = rearrange(token_logits, "B L D -> (B L) D")
    flat_target = rearrange(target_tokens, "B L -> (B L)")
    return F.cross_entropy(flat_logits, flat_target, ignore_index=ignore_index)


def stablemax_cross_entropy(
    token_logits: Tensor, target_tokens: Tensor, ignore_index: int = -100
) -> Tensor:
    flat_logits = rearrange(token_logits, "B L D -> (B L) D")
    flat_target = rearrange(target_tokens, "B L -> (B L)")
    return losses.stablemax_cross_entropy(flat_logits, flat_target, ignore_index)


def binary_cross_entropy(
    halt_logits: Tensor, token_logits: Tensor, target_tokens: Tensor
) -> Tensor:
    # 1 if the entire predicted sequence matches the target sequence
    is_seq_correct = (token_logits.argmax(dim=-1) == target_tokens).all(dim=-1)
    target = is_seq_correct.to(dtype=halt_logits.dtype)
    return F.binary_cross_entropy_with_logits(halt_logits, target)


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
        N_supervision: int = 16,
        activation_checkpointing: bool = False,
        halt_loss_weight: float = 0.5,
        loss: str = "softmax",  # "softmax" or "stablemax"
    ):
        super().__init__()

        # Save for checkpointing
        self.save_hyperparameters()

        # Hyperparameters used in training and forward
        self.N_supervision = N_supervision
        self.activation_checkpointing = activation_checkpointing
        self.halt_loss_weight = halt_loss_weight

        self.model = TinyRecursiveModel(
            vocab_size=vocab_size,
            D=D,
            L=L,
            n_scratchpad=scratch_slots,
            n_layers=n_layers,
            expansion_factor=expansion_factor,
            rms_norm_eps=rms_norm_eps,
            n=n,
            T=T,
            activation_checkpointing=activation_checkpointing,
        )
        self.loss_fn = (
            stablemax_cross_entropy if loss == "stablemax" else softmax_cross_entropy
        )

    def on_fit_start(self):
        if hasattr(self.logger, "watch"):
            self.logger.watch(self.model, log="all")

    def on_train_epoch_start(self):
        # Initialize one carry per gradient accumulation step
        self.carry = [None] * self.trainer.accumulate_grad_batches

    def training_step(self, batch, batch_idx) -> Tensor:
        x_input, y_true, puzzle_ids = batch.values()

        # Determine which carry to use
        carry_idx = batch_idx % self.trainer.accumulate_grad_batches

        # Initialize carry on first use
        if self.carry[carry_idx] is None:
            self.carry[carry_idx] = TrainingCarry(
                x_input=x_input,
                y_true=y_true,
                y=self.model.y_init(x_input.size(0)),
                z=self.model.z_init(x_input.size(0)),
                supervision_count=torch.zeros_like(puzzle_ids, dtype=torch.long),
                completed=torch.zeros_like(puzzle_ids, dtype=torch.bool),
            )

        # Replace completed slots with fresh samples
        carry = self.carry[carry_idx]
        replace = carry.completed
        carry.x_input[replace] = x_input[replace]
        carry.y_true[replace] = y_true[replace]
        carry.y[replace] = self.model.y_init(x_input[replace].size(0))
        carry.z[replace] = self.model.z_init(x_input[replace].size(0))
        carry.supervision_count[replace] = 0
        carry.completed[replace] = False

        # Single supervision step
        (y, z), y_hat, q_hat = self.model(carry.x_input, carry.y, carry.z)
        pred_loss = self.loss_fn(y_hat, carry.y_true)
        halt_loss = binary_cross_entropy(q_hat, y_hat, carry.y_true)
        loss = pred_loss + self.halt_loss_weight * halt_loss
        halt = q_hat.detach() >= 0.0  # 50% probability threshold

        # Update carry state
        carry.y = y
        carry.z = z
        carry.supervision_count += 1
        carry.completed = halt | (carry.supervision_count >= self.N_supervision)

        # Logging
        self.log("loss", loss, prog_bar=True)
        self.log("pred_loss", pred_loss)
        self.log("halt_loss", halt_loss, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        preds = y_hat.argmax(dim=-1)
        cell_acc = (preds == carry.y_true).float().mean()
        acc = (preds == carry.y_true).all(dim=-1).float().mean()
        self.log("train_cell_acc", cell_acc)
        self.log("train_acc", acc)
        halt_prob = torch.sigmoid(q_hat.detach())
        supervision_step = carry.supervision_count.detach()
        for step in supervision_step.unique():
            mask = supervision_step == step
            prob = halt_prob[mask].mean()
            self.log(f"halt_prob_{int(step.item())}", prob)
        if carry.completed.any():
            avg_sup = carry.supervision_count[carry.completed].float().mean()
            self.log("avg_sup", avg_sup, prog_bar=True)

        return loss

    def forward(self, batch):
        x_input, y_true, puzzle_ids = batch.values()
        y, z = None, None
        halt = torch.zeros_like(puzzle_ids, dtype=torch.bool)
        final_loss, final_halt_loss = 0.0, 0.0
        batch_size = x_input.size(0)
        sample_count = 0
        y_hat_final = y_hat_active = None
        halt_probs = []

        # No carry between supervision steps required
        for supervision_step in range(self.N_supervision):
            (y, z), y_hat, q_hat = self.model(x_input, y, z)

            pred_loss = self.loss_fn(y_hat, y_true)
            halt_loss = binary_cross_entropy(q_hat, y_hat, y_true)
            loss = pred_loss + self.halt_loss_weight * halt_loss
            sample_count += x_input.size(0)

            halt = q_hat.detach() >= 0.0  # 50% probability threshold
            halt_probs.append(torch.sigmoid(q_hat.detach()).mean().item())

            # Accumulate losses if at final supervision step
            is_final_step = halt | (supervision_step == self.N_supervision - 1)
            final_loss += loss.item() * is_final_step.sum().item() / batch_size
            final_halt_loss += (
                halt_loss.item() * is_final_step.sum().item() / batch_size
            )

            # Save predictions: full final output and active subset view
            if y_hat_active is None:
                y_hat_final = y_hat_active = y_hat
            else:
                y_hat_active[:] = y_hat

            if halt.all():
                break

            active = ~halt
            x_input = x_input[active]
            y = y[active]
            z = z[active]
            y_true = y_true[active]
            y_hat_active = y_hat_active[active]

        avg_sup = sample_count / batch_size
        return y_hat_final, (final_loss, final_halt_loss, avg_sup, halt_probs)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, _ = self.forward(batch)
        return y_hat.argmax(dim=-1)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        y_hat, (loss, halt_loss, avg_sup, halt_probs) = self.forward(batch)
        _, y_true, _ = batch.values()
        cell_acc = (y_hat.argmax(dim=-1) == y_true).float().mean()
        acc = (y_hat.argmax(dim=-1) == y_true).all(dim=-1).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_pred_loss", loss - halt_loss)
        self.log("val_halt_loss", halt_loss)
        self.log("val_avg_sup", avg_sup)
        self.log("cell_acc", cell_acc, prog_bar=True)
        self.log("acc", acc, prog_bar=True)
        for idx, prob in enumerate(halt_probs, start=1):
            self.log(f"val_halt_prob_{idx}", prob)
