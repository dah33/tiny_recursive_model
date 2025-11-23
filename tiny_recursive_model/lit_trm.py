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


def accuracy(
    logits: Tensor, targets: Tensor, halt: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    preds = logits.argmax(dim=-1)
    cell_correct = preds == targets
    all_correct = cell_correct.all(dim=-1)
    cell_acc = cell_correct.float().mean()
    exact_acc = all_correct.float().mean()
    halt_acc = (all_correct == halt).float().mean()
    return cell_acc, exact_acc, halt_acc


@dataclass
class TrainingCarry:
    x_input: Tensor
    y: Tensor
    z: Tensor
    y_true: Tensor
    steps: Tensor
    halted: Tensor  # Boolean mask


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
        self.carry: list[TrainingCarry | None] = []

    def on_fit_start(self):
        # Watch model on wandb logger (if present)
        if hasattr(self.logger, "watch"):
            self.logger.watch(self.model, log="all")

    def on_train_epoch_start(self):
        # Initialize one carry per gradient accumulation step
        self.carry = [None] * self.trainer.accumulate_grad_batches

    def training_step(self, batch, batch_idx) -> Tensor:
        x_input, y_true, puzzle_ids = batch.values()

        # Determine which carry to use
        carry_idx: int = batch_idx % self.trainer.accumulate_grad_batches

        # Initialize carry on first use
        if self.carry[carry_idx] is None:
            self.carry[carry_idx] = TrainingCarry(
                x_input=x_input,
                y_true=y_true,
                y=self.model.y_init(x_input.size(0)),
                z=self.model.z_init(x_input.size(0)),
                steps=torch.zeros_like(puzzle_ids, dtype=torch.long),
                halted=torch.zeros_like(puzzle_ids, dtype=torch.bool),
            )

        # Replace halted puzzles with fresh ones
        carry = self.carry[carry_idx]
        replace = carry.halted
        carry.x_input[replace] = x_input[replace]
        carry.y_true[replace] = y_true[replace]
        carry.y[replace] = self.model.y_init(x_input[replace].size(0))
        carry.z[replace] = self.model.z_init(x_input[replace].size(0))
        carry.steps[replace] = 0
        carry.halted[replace] = False

        # Single supervision step
        (y, z), y_hat, q_hat = self.model(carry.x_input, carry.y, carry.z)
        pred_loss = self.loss_fn(y_hat, carry.y_true)
        halt_loss = binary_cross_entropy(q_hat, y_hat, carry.y_true)
        loss = pred_loss + self.halt_loss_weight * halt_loss
        y_hat, q_hat = y_hat.detach(), q_hat.detach()  # y, z already detached

        # Update carry state
        carry.y = y
        carry.z = z
        carry.steps += 1

        # Early stopping
        halt_pred = q_hat >= 0.0  # 50% probability threshold
        carry.halted = halt_pred | (carry.steps >= self.N_supervision)

        # Metrics for those just halted
        # - Halt accuracy
        # - Puzzle cell and exact accuracy
        # - Supervision steps taken
        just_halted = carry.halted  # as will be replaced next step!
        if just_halted.any():
            cell_acc, exact_acc, halt_acc = accuracy(
                y_hat[just_halted], carry.y_true[just_halted], halt_pred[just_halted]
            )
            steps = carry.steps[just_halted].float().mean()

        # Logging - for entire carry, including those not yet halted
        self.log("train/lm_loss", pred_loss, prog_bar=True)
        self.log("train/q_halt_loss", halt_loss, prog_bar=True)

        # Logging - for those just halted
        if just_halted.any():
            self.log("train/accuracy", cell_acc)
            self.log("train/exact_accuracy", exact_acc)
            self.log("train/q_halt_accuracy", halt_acc)
            self.log("train/steps", steps, prog_bar=True)

        return loss

    def forward(self, batch) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        x_input, y_true, _puzzle_ids = batch.values()
        y = self.model.y_init(x_input.size(0))
        z = self.model.z_init(x_input.size(0))

        # Run all puzzles for N_supervision steps, no early stopping
        for _ in range(self.N_supervision):
            (y, z), y_hat, q_hat = self.model(x_input, y, z)
        y_hat, q_hat = y_hat.detach(), q_hat.detach()

        # Loss and accuracy for final step
        pred_loss = self.loss_fn(y_hat, y_true)
        halt_pred = q_hat >= 0.0  # 50% probability threshold
        halt_loss = binary_cross_entropy(q_hat, y_hat, y_true)
        cell_acc, exact_acc, halt_acc = accuracy(y_hat, y_true, halt_pred)
        steps = self.N_supervision

        return pred_loss, halt_loss, cell_acc, exact_acc, halt_acc, steps

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, _ = self.forward(batch)
        return y_hat.argmax(dim=-1)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred_loss, halt_loss, cell_acc, exact_acc, halt_acc, steps = self.forward(batch)
        self.log("val/lm_loss", pred_loss)
        self.log("val/q_halt_loss", halt_loss)
        self.log("val/accuracy", cell_acc)
        self.log("val/exact_accuracy", exact_acc, prog_bar=True)
        self.log("val/q_halt_accuracy", halt_acc)
        self.log("val/steps", steps)  # always N_supervision--for symmetry

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred_loss, halt_loss, cell_acc, exact_acc, halt_acc, steps = self.forward(batch)
        self.log("test/lm_loss", pred_loss)
        self.log("test/q_halt_loss", halt_loss)
        self.log("test/accuracy", cell_acc)
        self.log("test/exact_accuracy", exact_acc, prog_bar=True)
        self.log("test/q_halt_accuracy", halt_acc)
        self.log("test/steps", steps)  # always N_supervision--for symmetry
